"""C1, round 3: is it more NEGATIVES or more TRAINING? Colder too. PLAN.md C1 (and C2).

    python sweeps/make_conlong.py --clean

sweep-conneg's step-matched control changed the question. At 50m, t0.01:

    width 64,  3 epochs   (~660 steps)    MAP@R 0.719
    width 256, 3 epochs   (~165 steps)    0.510
    width 256, 12 epochs  (~660 steps)    0.813

So wide batches lost only because they took fewer steps -- but the 0.813 arm also saw
4x the data. Its missing comparison is width 64 at 12 epochs, the SAME compute. If that
also reaches ~0.81, the gain is training length, not negatives. This grid crosses width
{64, 256, 512} with epochs {3, 12, 24}, so every width is seen at equal compute as well
as equal steps.

TEMPERATURE goes colder, {0.001, 0.002, 0.003}: 200m and 400m both peaked at 0.003, the
coldest value tried (400m 0.823 at t0.003 vs 0.808 at t0.005), so for large models the
optimum is still below the grid.

SCALES: 50m carries the full crossing (27 cells x 3 seeds = 81 arms), because it is
cheap. 400m -- the scale the temperature question is about -- runs widths {64, 256} x
epochs {3, 12} x the three temperatures (12 cells x 3 seeds = 36 arms). 400m at 24
epochs would be ~7 h per arm and is left out until 12 shows whether length pays.

117 arms. The 50m 3-epoch arms fit debug-scaling; everything else goes to capacity
(sweeps/arms/conlong_*.txt).
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-conlong"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_conlong-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

SCALES = {f"s{s:0>4}": (s, f"{FROZEN}/msdelta-{s}-production-01-checkpoint-220000")
          for s in ("50m", "100m", "200m", "400m")}
PK = {"pk064": ("16", "4"), "pk256": ("64", "4"), "pk512": ("128", "4")}
LEARNING_RATES = {"lr1e4": "1e-4"}
KL_WEIGHTS = {"kl10": "10"}
TEMPERATURES = {"t0001": "0.001", "t0002": "0.002", "t0003": "0.003"}
SEEDS = ("0", "1", "2")
EPOCHS = "3"


def arm_name(scale, lr, kl, t, pk, seed, ep=EPOCHS):
    tag = f"_ep{int(ep):02d}"
    return f"{scale}_{t}_{pk}{tag}_seed{seed}"


def render_arm(scale, lr, kl, t, pk, seed, ep=EPOCHS) -> tuple[str, str]:
    name = arm_name(scale, lr, kl, t, pk, seed, ep)
    overrides = {
        # 12 arms/node at TILES_PER_ARM=1, and a 400m optimizer checkpoint is 7.4 GB:
        # save_steps 200 put ~89 GB on Lustre at once and killed arms in 8853557,
        # 8853558 and 8853703 with "enforce fail ... unexpected pos" from torch.save.
        # Only final/ is ever consumed, so optimizer state is pure write amplification.
        "--save_only_model": "true",
        "--save_steps": "700",
        "--save_total_limit": "1",
        # This grid runs in 1h debug-scaling slots and is RESUBMITTED until done, so an
        # arm may be killed mid-run. Resuming from a save_only_model checkpoint restarts
        # Adam and the LR schedule mid-run -- a different trajectory from an
        # uninterrupted arm, silently. With no intermediate checkpoints an interrupted
        # arm restarts from step 0 with the same seed instead, which is equivalent to
        # never having been interrupted. The only cost is the killed arm's compute.
        # final/ is written explicitly by main() and does not depend on this.
        "--save_strategy": "no",
        "--pretrained_path": SCALES[scale][1],
        "--groups_per_batch": PK[pk][0],
        "--replicates": PK[pk][1],
        "--gradcache_chunk": "4",
        "--learning_rate": LEARNING_RATES[lr],
        "--kl_weight": KL_WEIGHTS[kl],
        "--temperature": TEMPERATURES[t],
        "--num_train_epochs": ep,
        "--seed": seed,
        # Pinned: the split must not move with the training seed, or arms are scored on
        # different held-out data and cannot be compared.
        "--split_seed": "0",
        "--run_name": f"{RUN_PREFIX}{name}",
        "--output_dir": f"./runs/{RUN_PREFIX}{name}",
    }
    lines, seen = [], set()
    tokens = TEMPLATE.read_text().split()
    for flag, value in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{flag} {overrides.get(flag, value)}")
        seen.add(flag)
    for flag, value in overrides.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")
    return name, "\n".join(lines) + "\n"


def description(scale, lr, kl, t, pk, seed, ep=EPOCHS) -> str:
    label = SCALES[scale][0]
    return (f"CONTRASTIVE HP ARM: {label}, lr {LEARNING_RATES[lr]}, KL "
            f"{KL_WEIGHTS[kl]}, temperature {TEMPERATURES[t]}, seed {seed} of "
            f"{len(SEEDS)}, PK sampling at P=2 K=2 for {EPOCHS} epochs. Asks whether "
            f"the contrastive hyperparameter choice transfers across scale the way the "
            f"denoise one did, where the same point won all four grids. 50m is re-run "
            f"rather than compared against its existing grid, which predates the FT14 "
            f"sampler fix (every epoch replayed the same 15.4% of the corpus) and is "
            f"n=1 per cell against a metric with seed sd 0.43-0.63.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clean", action="store_true")
    cli = ap.parse_args()
    for key, (label, ckpt) in SCALES.items():
        if not Path(ckpt, "model.safetensors").exists():
            raise SystemExit(f"{label} not frozen at {ckpt}")
    # Every arm names its epochs, including 3, so the grid reads uniformly.
    combos = [("s050m", "lr1e4", "kl10", t, pk, s, ep)
              for t in TEMPERATURES for pk in PK for ep in ("3", "12", "24") for s in SEEDS]
    combos += [("s400m", "lr1e4", "kl10", t, pk, s, ep)
               for t in TEMPERATURES for pk in ("pk064", "pk256") for ep in ("3", "12")
               for s in SEEDS]

    if cli.check:
        stale = [n for n, text in (render_arm(*c) for c in combos)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(combos)} arms match {TEMPLATE.relative_to(REPO)}")
        return 0

    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for combo in combos:
        name, text = render_arm(*combo)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(*combo))
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\ncontrastive-hp\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/  "
          f"(50m: 3 temps x 3 widths x 3 epoch budgets; 400m: 3 x 2 x 2; 3 seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
