"""Push C1 past both edges: more negatives, lower temperature. PLAN.md C1 (and C2).

    python sweeps/make_conneg.py --clean

sweep-conbig (job 8856460) found both axes at their limits. At every scale MAP@R rose
monotonically with batch width (P*K 4 -> 16 -> 64, e.g. 50m 0.51 -> 0.66 -> 0.72 at
t0.01) and with LOWER temperature (0.03 -> 0.02 -> 0.01), so the optimum lies beyond
64 negatives and below t0.01. "More negatives hurt" is not just retracted -- it was
backwards, and the four-spectrum batch the whole project trained with was the limiter.

MAIN GRID: 4 scales x t {0.003, 0.005, 0.01} x P*K {64, 128, 256, 512} x 3 seeds = 144.
K is held at 4 and P grows (16/32/64/128 groups), so every step is more negatives, not
more positives. t0.01 x 64 repeats the best conbig cell as an anchor between the grids.

STEP-COUNT CONTROLS (6 arms). At a fixed 3 epochs a wider batch means fewer optimizer
steps -- ~660 at 64, ~80 at 512. If wide batches then underperform, that is ambiguous
between "negatives stopped helping" and "it barely trained". 50m at t0.01 with 256 for
12 epochs and 512 for 24 epochs matches the 64-wide step count and settles it.

WHERE IT RUNS: 400m at 64 wide already took 53 min, at the 1 h debug-scaling limit, and
these arms save no mid-run checkpoints (an interrupted arm restarts clean), so 400m and
the long controls go to capacity while 50m/100m/200m run in debug-scaling rounds.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-conneg"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_conneg-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

SCALES = {f"s{s:0>4}": (s, f"{FROZEN}/msdelta-{s}-production-01-checkpoint-220000")
          for s in ("50m", "100m", "200m", "400m")}
PK = {"pk064": ("16", "4"), "pk128": ("32", "4"), "pk256": ("64", "4"), "pk512": ("128", "4")}
LEARNING_RATES = {"lr1e4": "1e-4"}
KL_WEIGHTS = {"kl10": "10"}
TEMPERATURES = {"t0003": "0.003", "t0005": "0.005", "t001": "0.01"}
SEEDS = ("0", "1", "2")
EPOCHS = "3"


def arm_name(scale, lr, kl, t, pk, seed, ep=EPOCHS):
    tag = "" if ep == EPOCHS else f"_ep{int(ep):02d}"
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
    combos = list(itertools.product(SCALES, LEARNING_RATES, KL_WEIGHTS,
                                    TEMPERATURES, PK, SEEDS))
    # step-matched controls: same optimizer-step count as 64-wide at 3 epochs
    combos += [("s050m", "lr1e4", "kl10", "t001", pk, s, ep)
               for pk, ep in (("pk256", "12"), ("pk512", "24")) for s in SEEDS]

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
          f"({len(SCALES)} scales x {len(PK)} widths x "
          f"{len(TEMPERATURES)} temps x {len(SEEDS)} seeds + 6 step-matched controls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
