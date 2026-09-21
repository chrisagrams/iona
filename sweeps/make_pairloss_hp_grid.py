"""Locate the pair-loss optimum, which the first grid left at its corner (FT17).

    python sweeps/make_pairloss_hp_grid.py --clean

The first pair-loss grid swept margin {0.5, 1.0} x positive_fraction {0.25, 0.5, 0.75}
x lr {2e-5, 1e-4}, one arm each, and its best cell was the CORNER in both margin and
learning rate:

    2.93  pf05_m10_lr1e4      <- highest margin, highest lr
    2.39  pf025_m10_lr1e4
    1.75  pf075_m10_lr1e4
    ...
    1.35  pf075_m05_lr2e5     <- the random-init floor: no learning at all

Every margin-0.5 arm sits near the floor and every margin-1.0 arm is above it, so the
margin was the dominant axis and both sampled values were too small. That makes the
comparison against PK softmax (6.60) uninformative: the pair formulation was
under-tuned, not beaten.

WHY THESE MARGINS. Embeddings are L2-normalised, so pair distance lives in [0, 2] and
d^2 = 2 - 2cos:

    1.00 .. cosine <= 0.5      the old edge
    1.41 .. cosine <= 0        ORTHOGONAL, the natural target for unrelated peptides
    1.70 .. cosine <= -0.44    past orthogonal, into actively opposing

1.41 is the principled value: two different peptides should be unrelated, not opposed.
1.70 is there so that if 1.41 wins it has not won at an edge again.

FOUR SEEDS per cell, because a single draw cannot rank cells against a metric whose
seed sd is 0.43-0.63 -- which is how the first grid's ordering below the top two arms
should be read: as unranked.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-pairloss-hp"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_pairhp-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"
CHECKPOINT = f"{FROZEN}/msdelta-50m-production-01-checkpoint-133233"

MARGINS = {"m100": "1.0", "m141": "1.414", "m170": "1.7"}
RATES = {"lr1e4": "1e-4", "lr5e4": "5e-4"}
FRACTIONS = {"pf025": "0.25", "pf05": "0.5"}
SEEDS = ("0", "1", "2", "3")
PAIRS_PER_BATCH = "4"
EPOCHS = "3"


def arm_name(m, lr, pf, seed):
    return f"{pf}_{m}_{lr}_seed{seed}"


def render_arm(m, lr, pf, seed) -> tuple[str, str]:
    name = arm_name(m, lr, pf, seed)
    overrides = {
        "--pretrained_path": CHECKPOINT,
        "--pair_loss": "true",
        "--pairs_per_batch": PAIRS_PER_BATCH,
        "--pair_margin": MARGINS[m],
        "--positive_fraction": FRACTIONS[pf],
        "--learning_rate": RATES[lr],
        "--num_train_epochs": EPOCHS,
        "--seed": seed,
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


def description(m, lr, pf, seed) -> str:
    meaning = {"m100": "cosine <= 0.5, the old grid's edge",
               "m141": "cosine <= 0, ORTHOGONAL -- the principled target",
               "m170": "cosine <= -0.44, past orthogonal into actively opposing"}[m]
    return (f"PAIR-LOSS HP ARM: margin {MARGINS[m]} ({meaning}), positive_fraction "
            f"{FRACTIONS[pf]}, lr {RATES[lr]}, seed {seed} of {len(SEEDS)}, "
            f"{PAIRS_PER_BATCH} pairs per batch, {EPOCHS} epochs on the 50m encoder. "
            f"The first pair grid's best cell was its corner in both margin and lr "
            f"(2.93 at margin 1.0 / lr 1e-4) and every margin-0.5 arm sat near the "
            f"1.35 floor, so the formulation was under-tuned rather than beaten by PK "
            f"softmax's 6.60. Four seeds because the separation ratio has seed sd "
            f"0.43-0.63 and single draws cannot rank cells.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clean", action="store_true")
    cli = ap.parse_args()
    if not Path(CHECKPOINT, "model.safetensors").exists():
        raise SystemExit(f"50m not frozen at {CHECKPOINT}")
    combos = list(itertools.product(MARGINS, RATES, FRACTIONS, SEEDS))

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
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\npairloss-hp\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/  "
          f"({len(MARGINS)} margins x {len(RATES)} lr x {len(FRACTIONS)} pf x "
          f"{len(SEEDS)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
