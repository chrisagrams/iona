"""Decompose contrastive variance: how much is the seed, how much is nondeterminism?

    python sweeps/make_seedvar_grid.py --clean

Paired with configs/sweep-repeat, which runs the SAME seed six times. Between them:

    sweep-repeat   fixed seed, 6 runs .... nondeterminism alone (XPU float order,
                                           node-to-node differences)
    sweep-seedvar  6 seeds, 1 run each ... seed effect PLUS that same nondeterminism

The difference in spread is the seed's own contribution. Neither number alone
distinguishes "this configuration is unstable" from "this machine is nondeterministic",
and the two have different remedies: the first wants a different learning rate, the
second wants averaging over repeats before any comparison.

The prompt for this was an accident. Arm lr5e4_kl10_t007 scored 7.83 in job 8842232 and
6.01 in 8843838 from byte-identical configs at the same seed, a gap of 1.82 -- larger
than most effects reported on this metric.

THE SPLIT IS HELD FIXED. `split_seed` was added and defaulted to 0 precisely for this
grid: the train/validation split used to be seeded from `training_args.seed`, so
changing the seed changed the HELD-OUT DATA and every arm was scored on a different
validation set. That confounds training variance with split variance and makes the
arms incomparable to each other. Here only training varies -- batch order via the PK
sampler, dropout, and initialisation of anything not loaded from the checkpoint.

Hyperparameters are the best contrastive arm (lr 5e-4, KL 10, temperature 0.07), which
is also the least stable setting in the grid -- its four arms span 1.35 to 7.83,
including a collapse to the floor. If variance is learning-rate dependent, this is
where it will be largest, and a follow-up at 2e-5 is the natural next step.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
OUT = REPO / "configs" / "sweep-seedvar"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_seedvar-"

LEARNING_RATE, KL_WEIGHT, TEMPERATURE = "5e-4", "10", "0.07"
SEEDS = ("0", "1", "2", "3", "4", "5")


def arm_name(seed: str) -> str:
    return f"seed{seed}"


def render_arm(seed: str) -> tuple[str, str]:
    name = arm_name(seed)
    overrides = {
        "--learning_rate": LEARNING_RATE,
        "--kl_weight": KL_WEIGHT,
        "--temperature": TEMPERATURE,
        "--seed": seed,
        # Held fixed on purpose: see the module docstring. Without this the validation
        # split moves with the seed and the arms score on different data.
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


def description(seed: str) -> str:
    return (f"SEED-VARIANCE ARM, seed {seed} of {', '.join(SEEDS)}: the best contrastive "
            f"configuration (lr {LEARNING_RATE}, KL {KL_WEIGHT}, temperature "
            f"{TEMPERATURE}) with the TRAINING seed varied and the train/validation "
            f"split held fixed at split_seed 0, so every arm is scored on identical "
            f"held-out data. Read against configs/sweep-repeat, which runs the same "
            f"configuration six times at a FIXED seed: that grid measures "
            f"nondeterminism alone, this one measures the seed effect on top of it, "
            f"and the difference in spread is the seed's own contribution. Motivated "
            f"by arm lr5e4_kl10_t007 scoring 7.83 and 6.01 on byte-identical configs "
            f"at the same seed, a gap larger than most differences this project has "
            f"reported on the separation ratio.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    cli = parser.parse_args()

    if cli.check:
        stale = [n for n, text in (render_arm(s) for s in SEEDS)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale)}")
            return 1
        print(f"  {len(SEEDS)} arms match {TEMPLATE.relative_to(REPO)}")
        return 0

    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for seed in SEEDS:
        name, text = render_arm(seed)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(seed))
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\nseedvar\n")
    print(f"arms={len(SEEDS)} under {OUT.relative_to(REPO)}/  (seeds {', '.join(SEEDS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
