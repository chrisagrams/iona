"""Does the pair loss's decomposability pay, given enough steps to use it?

    python sweeps/make_pairaccum_grid.py --clean

THE ABLATION THAT PROMPTED THIS WAS ACCIDENTAL. The first pair grid set
gradient_accumulation_steps 16 deliberately -- that is the whole point of a decomposable
loss, 64 pairs per optimizer step at the memory of 8 spectra. The follow-up HP grid
dropped it by mistake, and the same cell scored:

    accum 16, 64 pairs/step,   273 optimizer steps    2.93
    accum  1,  4 pairs/step, 4,377 optimizer steps    6.92

So at three epochs, trading optimizer steps for breadth is a bad trade -- 273 updates
cannot train the model however wide each one is. That does NOT settle whether breadth
helps, because the wide configuration was never given a comparable number of updates.

THIS GRID GIVES IT ONE. accum 16 at 16 and 48 epochs is 1,459 and 4,377 optimizer steps,
matching and quadrupling what the narrow configuration got at three epochs. If breadth
is worth anything, it shows up here; if 48 epochs of 64-pair steps still loses to 3
epochs of 4-pair steps, the decomposability buys nothing at this corpus size.

Held at the HP grid's answers: positive_fraction 0.5, which beat 0.25 by nearly 2x and
was the dominant axis, and margin 1.0, which is indistinguishable from 1.414 and 1.7.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-pairaccum"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_pairacc-"
CHECKPOINT = ("/flare/UIC-HPC/khuss/msdelta/pretrained/"
              "msdelta-50m-production-01-checkpoint-133233")

EPOCHS = ("16", "48")
RATES = {"lr1e4": "1e-4", "lr5e4": "5e-4"}
SEEDS = ("0", "1", "2")
ACCUMULATION = "16"
PAIRS_PER_BATCH = "4"
MARGIN, FRACTION = "1.0", "0.5"


def arm_name(ep, lr, seed):
    return f"ep{int(ep):03d}_{lr}_seed{seed}"


def render_arm(ep, lr, seed):
    name = arm_name(ep, lr, seed)
    overrides = {
        "--pretrained_path": CHECKPOINT,
        "--pair_loss": "true",
        "--pairs_per_batch": PAIRS_PER_BATCH,
        "--gradient_accumulation_steps": ACCUMULATION,
        "--pair_margin": MARGIN,
        "--positive_fraction": FRACTION,
        "--learning_rate": RATES[lr],
        "--num_train_epochs": ep,
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


def description(ep, lr, seed):
    steps = 1459 * int(ep) // 16
    return (f"PAIR + ACCUMULATION ARM: {ep} epochs at gradient_accumulation_steps "
            f"{ACCUMULATION}, so an optimizer step sees {int(PAIRS_PER_BATCH)*16} pairs "
            f"at the memory of {int(PAIRS_PER_BATCH)*2} spectra, and the run takes about "
            f"{steps:,} optimizer steps. lr {RATES[lr]}, margin {MARGIN}, "
            f"positive_fraction {FRACTION}, seed {seed}. Tests whether the pair loss's "
            f"decomposability is worth anything once the wide steps get a comparable "
            f"update count: at 3 epochs the same shape took 273 steps and scored 2.93 "
            f"against 6.92 for 4,377 narrow steps, which measured the update count, not "
            f"the breadth.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clean", action="store_true")
    cli = ap.parse_args()
    if not Path(CHECKPOINT, "model.safetensors").exists():
        raise SystemExit("50m not frozen")
    combos = list(itertools.product(EPOCHS, RATES, SEEDS))
    if cli.check:
        stale = [n for n, t in (render_arm(*c) for c in combos)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != t
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(combos)} arms match {TEMPLATE.relative_to(REPO)}")
        return 0
    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for c in combos:
        name, text = render_arm(*c)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(*c))
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\npairaccum\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
