"""Does the contrastive hyperparameter choice transfer across scale, as it does for denoise?

    python sweeps/make_contrastive_hp_grid.py --clean

WHY BOTH SCALES AND NOT JUST 100m. The question is whether the 50m optimum holds at
100m, and for denoise the answer was yes -- the same point won all four grids, which is
what licensed repeating one configuration across scales. The existing 50m contrastive
grid cannot serve as the reference, for two independent reasons:

  it ran 2026-09-20, BEFORE the FT14 sampler fix, so every arm trained on the same
  15.4% of the corpus every epoch and the levels are depressed by roughly 13%;

  it is n=1 per cell, against a separation ratio whose seed sd is 0.43-0.63. A single
  draw cannot rank twelve cells -- the same mistake that produced the retracted
  "more epochs hurts" result.

So 50m is re-run alongside 100m, post-fix and with seeds, and the comparison is between
two grids measured the same way rather than against a stale record.

AXES are the original three: learning_rate x kl_weight x temperature. PK sampling at
P=2 K=2 and 3 epochs are held, being the configuration the epochs sweep settled on
(3 and 10 epochs are indistinguishable, and longer is mildly worse).

FOUR SEEDS per cell puts the standard error of a cell-to-cell difference at
0.5*sqrt(2/4) = 0.35, against arms that span several units. 96 arms at one tile each is
eight nodes for about fifteen minutes -- contrastive at 3 epochs is 5 minutes an arm, so
this costs almost nothing and there is no reason to under-power it.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-contrastive-hp-big"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_conhpb-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

SCALES = {"s200m": ("200m", f"{FROZEN}/msdelta-200m-production-01-checkpoint-192799"),
          "s400m": ("400m", f"{FROZEN}/msdelta-400m-production-01-checkpoint-181381")}
LEARNING_RATES = {"lr2e5": "2e-5", "lr1e4": "1e-4", "lr5e4": "5e-4"}
KL_WEIGHTS = {"kl0": "0", "kl10": "10"}
TEMPERATURES = {"t007": "0.07", "t02": "0.2"}
SEEDS = ("0", "1", "2", "3")
EPOCHS = "3"


def arm_name(scale, lr, kl, t, seed):
    return f"{scale}_{lr}_{kl}_{t}_seed{seed}"


def render_arm(scale, lr, kl, t, seed) -> tuple[str, str]:
    name = arm_name(scale, lr, kl, t, seed)
    overrides = {
        "--pretrained_path": SCALES[scale][1],
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


def description(scale, lr, kl, t, seed) -> str:
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
                                    TEMPERATURES, SEEDS))

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
          f"({len(SCALES)} scales x {len(LEARNING_RATES)} lr x {len(KL_WEIGHTS)} kl x "
          f"{len(TEMPERATURES)} temp x {len(SEEDS)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
