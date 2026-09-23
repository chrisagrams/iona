"""Sweep P/K -- the number of negatives -- which nothing in this project has ever varied.

    python sweeps/make_conpk.py --clean

GATED ON sweep-gradcache-confirm. This grid cannot run at P*K > 4 without GradCache, and
GradCache has never produced a task number at a configuration that trains. If the
confirmation A/B fails, a null result here is uninterpretable and the grid is void.

WHY THIS AXIS. Every contrastive run in this project used groups_per_batch 2 x
replicates 2 -- FOUR spectra, so roughly two negatives -- against a loss whose entire
signal comes from negatives. The justification was "more negatives hurt", from
sweep-gradcache-v2. That result does not survive: it was decided on the separation
ratio, which does not predict retrieval, at lr2e-5/KL0/t0.2, which does not train.
Rescored on the task, 64 negatives against 4 is t=-0.21, p=0.84 -- the ordering does not
invert, it COLLAPSES. So the axis is unmeasured rather than settled.

WHY IT MIGHT EXPLAIN THE SCALE RESULT. Contrastive gets WORSE with model size here,
rho -0.80 against scale on MAP@100, MAP@R and R-Precision alike. A 400m encoder has no
more signal to exploit than a 50m one if the batch offers two negatives either way, so a
negative-starved setup is a mechanism that would produce exactly this shape. Not a
prediction -- a hypothesis this grid can falsify.

AXES:
    P x K     2x2 = 4      the setting every previous run used
              8x2 = 16
             16x4 = 64     32 GiB of DeltaMZBias without GradCache; 2 GiB with it
    chunk     4            fixed, so peak memory is constant across the sweep and any
                           difference is the negatives rather than the memory regime

Two cells, 50m@330k (the best cell) and 200m@330k (the worst), because if negatives are
what scale needs, the effect should be LARGER at 200m. Three seeds. 18 arms.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-conpk"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_conpk-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

SCALES = {"s050m_ck330k": ("50m@330k", f"{FROZEN}/msdelta-50m-production-01-checkpoint-330000"),
          "s200m_ck330k": ("200m@330k", f"{FROZEN}/msdelta-200m-production-01-checkpoint-330000")}
# P x K. The name carries the product because that is the number of negatives.
PK = {"pk04": ("2", "2"), "pk16": ("8", "2"), "pk64": ("16", "4")}
LEARNING_RATES = {"lr1e4": "1e-4"}
KL_WEIGHTS = {"kl10": "10"}
TEMPERATURES = {"t007": "0.07"}
SEEDS = ("0", "1", "2")
EPOCHS = "3"


def arm_name(scale, lr, kl, t, gc, seed):
    return f"{scale}_{gc}_seed{seed}"


def render_arm(scale, lr, kl, t, gc, seed) -> tuple[str, str]:
    name = arm_name(scale, lr, kl, t, gc, seed)
    overrides = {
        # 12 arms/node at TILES_PER_ARM=1, and a 400m optimizer checkpoint is 7.4 GB:
        # save_steps 200 put ~89 GB on Lustre at once and killed arms in 8853557,
        # 8853558 and 8853703 with "enforce fail ... unexpected pos" from torch.save.
        # Only final/ is ever consumed, so optimizer state is pure write amplification.
        "--save_only_model": "true",
        "--save_steps": "700",
        "--save_total_limit": "1",
        "--pretrained_path": SCALES[scale][1],
        "--gradcache_chunk": "4",
        "--groups_per_batch": PK[gc][0],
        "--replicates": PK[gc][1],
        "--learning_rate": LEARNING_RATES[lr],
        "--kl_weight": KL_WEIGHTS[kl],
        "--temperature": TEMPERATURES[t],
        "--num_train_epochs": EPOCHS,
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


def description(scale, lr, kl, t, gc, seed) -> str:
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
          f"({len(SCALES)} scale x checkpoint cells x {len(KL_WEIGHTS)} kl x "
          f"{len(TEMPERATURES)} temp x {len(SEEDS)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
