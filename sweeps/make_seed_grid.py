"""Repeat the winning denoise configuration across seeds, at two model scales.

    python sweeps/make_seed_grid.py --clean
    python sweeps/make_seed_grid.py --check

WHY THIS EXISTS (FT5). Both finished grids rank their arms by differences far smaller
than anything that has been shown to be reproducible:

  50m, 216 arms: the top eight span 0.0023 test AUROC, across three head widths and two
                 encoder_lr_scales. Head width 128/256/512 at the same lr and scale gave
                 0.9319 / 0.9319 / 0.9320.
  100m, 12 arms: the top five span 0.0012, across lr 5e-5 to 5e-4 and both batch sizes.

Neither number has an error bar, so "lr 2e-4 with es 0.5 is the best configuration" is
not currently a claim this project can make -- and neither is the more interesting one,
that 100m (0.9403) beats 50m (0.9320) by 0.0083. That gap is only ~4x the within-grid
top-cluster spread, and until the spread is known to be smaller than the gap, the scale
comparison rests on one sample each.

DESIGN. Six seeds at each of two scales, twelve arms, one wave on twelve nodes.

The hyperparameters are held fixed at the configuration both grids independently chose:
lr 2e-4, encoder_lr_scale 0.5, 4 epochs, head 512, effective batch 12. The 50m winner
(lr2e4_es05_ep4_h512_b12) and the 100m winner (lr2e4_es05_b12, which fixes ep4/h512) are
the same point, which is itself worth knowing -- it is what justifies not re-sweeping
those axes for 200m.

WHAT VARIES, AND WHAT MUST NOT. `--seed` moves head initialisation, data order and
dropout. It does NOT move the train/validation/test split: `build_denoising_datasets`
takes the splits as the dataset publishes them, so every arm here is scored on the
identical test rows. That is deliberate. Mixing split noise into training noise would
measure something real but not the thing being asked, which is "run this again and how
different is the answer".

READING IT. The spread across the six seeds at one scale is the resolution of both
grids. Any HP difference smaller than it was never measured, only observed -- including,
possibly, the 50m grid's entire top eight. The between-scale difference is a claim only
if it clears the within-scale spread.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-denoise-seeds"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_dnseed-"

SIZES = ("50m", "100m")      # overridden by --sizes
SEEDS = ("1", "2", "3", "4", "5", "6")
# The configuration both grids selected, held fixed. per_device 1 x 12 tiles x 1
# accumulation = effective batch 12, which is what b12 meant in the grids.
LEARNING_RATE = "2e-4"
ENCODER_SCALE = "0.5"
EPOCHS = "4"
HEAD = "512"
PER_DEVICE = "1"
ACCUMULATION = "1"


def template_for(size: str) -> Path:
    return REPO / "configs" / f"finetune-denoise-{size}-ds" / "training.args"


def arm_name(size: str, seed: str) -> str:
    return f"{size}_seed{seed}"


def render_arm(size: str, seed: str) -> tuple[str, str]:
    name = arm_name(size, seed)
    overrides = {
        "--learning_rate": LEARNING_RATE,
        "--encoder_lr_scale": ENCODER_SCALE,
        "--num_train_epochs": EPOCHS,
        "--head_hidden_size": HEAD,
        "--per_device_train_batch_size": PER_DEVICE,
        "--gradient_accumulation_steps": ACCUMULATION,
        "--seed": seed,
        "--run_name": f"{RUN_PREFIX}{name}",
        "--output_dir": f"./runs/{RUN_PREFIX}{name}",
    }
    lines, seen = [], set()
    tokens = template_for(size).read_text().split()
    for flag, value in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{flag} {overrides.get(flag, value)}")
        seen.add(flag)
    for flag, value in overrides.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")
    return name, "\n".join(lines) + "\n"


def description(size: str, seed: str) -> str:
    shared = template_for(size).parent / "DESCRIPTION.md"
    lead = " ".join(shared.read_text().split()) + " " if shared.exists() else ""
    return (f"{lead}SEED REPETITION (FT5): the {size} encoder at the configuration both "
            f"finished grids selected -- lr {LEARNING_RATE}, encoder at "
            f"{ENCODER_SCALE}x the head's rate, {EPOCHS} epochs, head width {HEAD}, "
            f"effective batch 12 -- repeated at seed {seed} of {', '.join(SEEDS)}. "
            f"Only the seed varies, and it moves head initialisation, data order and "
            f"dropout; the train/validation/test split comes from the dataset itself "
            f"and is identical across every arm, so these six numbers measure training "
            f"noise alone. Their spread is the resolution of the 216-arm 50m grid and "
            f"the 12-arm 100m grid, whose top clusters span 0.0023 and 0.0012 test "
            f"AUROC respectively. It also decides whether 100m's 0.9403 genuinely beats "
            f"50m's 0.9320, a gap of 0.0083 that currently rests on one sample each.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--sizes", default="50m,100m",
                        help="comma-separated scales. Defaults to the two whose grids "
                             "have finished and named a winner; 200m and 400m can only "
                             "join once theirs do, because the point is to repeat each "
                             "scale's OWN winning configuration.")
    parser.add_argument("--seeds", type=int, default=6,
                        help="repetitions per scale. scales x seeds should be a "
                             "multiple of 12 to fill a node-wave exactly.")
    global SIZES, SEEDS
    cli = parser.parse_args()
    SIZES = tuple(x.strip() for x in cli.sizes.split(",") if x.strip())
    SEEDS = tuple(str(i + 1) for i in range(cli.seeds))

    for size in SIZES:
        if not template_for(size).exists():
            raise SystemExit(f"no template at {template_for(size)}")
    combos = list(itertools.product(SIZES, SEEDS))

    if cli.check:
        stale = [n for n, text in (render_arm(*c) for c in combos)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(combos)} arms match the {'/'.join(SIZES)} templates")
        return 0

    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for combo in combos:
        name, text = render_arm(*combo)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(*combo))
    STAMP.write_text("\n".join(str(template_for(s).relative_to(REPO)) for s in SIZES)
                     + "\ndenoise-seeds\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/")
    for combo in combos:
        print(f"  {arm_name(*combo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
