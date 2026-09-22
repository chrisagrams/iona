"""Re-take the contrastive scale curve at the configuration that actually wins.

    python sweeps/make_contrastive_scale_v2.py --clean

THE CURVE ON RECORD IS UNQUOTABLE. sweep-contrastive-scale measured 5.86 / 7.00 / 7.02
/ 6.74 and was read as "contrastive saturates at 100m". Two things invalidate the
levels and cast doubt on the shape:

  FT14. It ran before the sampler fix, so every arm trained on the same 15.4% of the
  corpus every epoch. Fixing that is worth about +13% on its own.

  THE HYPERPARAMETERS. It used lr 2e-5 / KL 10 / temperature 0.07, which the 96-arm HP
  grid (job 8847624, 4 seeds per cell) ranks SEVENTH of twelve. The winner is lr 2e-5 /
  KL 0 / temperature 0.2, worth +2.76 at 50m and +3.17 at 100m -- roughly 40% more
  separation from hyperparameters alone.

The learning rate was the one axis chosen with evidence and it survives; KL and
temperature were carried over from an n=1 grid and never revisited. KL 0 winning is not
a contradiction of "KL prevents collapse": at lr 5e-4 the KL-0 arms still collapse to
the floor, so KL is a stabiliser that costs performance when stability is not the
binding constraint.

WHY THE SHAPE IS ALSO IN DOUBT, not just the levels: at the corrected setting 100m beat
50m in nine of twelve cells and by +1.34 at the best cell. "Saturates at 100m" was
measured where the metric was depressed and the ranking scrambled.

Six seeds per scale, as before, because the separation ratio has seed sd 0.43-0.63 and
the differences being claimed are of that order.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
OUT = REPO / "configs" / "sweep-contrastive-scale-v2"
STAMP = OUT / ".template"
RUN_PREFIX = "v3_conscale-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

LEARNING_RATE, KL_WEIGHT, TEMPERATURE = "2e-5", "0", "0.2"
SEEDS = ("0", "1", "2", "3", "4", "5")
# sort key -> (label, frozen checkpoint). Keys chosen so s400m sorts last.
SCALES = {
    "s050m": ("50m", f"{FROZEN}/msdelta-50m-production-01-checkpoint-133233"),
    "s100m": ("100m", f"{FROZEN}/msdelta-100m-production-01-checkpoint-138073"),
    "s200m": ("200m", f"{FROZEN}/msdelta-200m-production-01-checkpoint-192799"),
    "s400m": ("400m", f"{FROZEN}/msdelta-400m-production-01-checkpoint-181381"),
}


def arm_name(scale: str, seed: str) -> str:
    return f"{scale}_seed{seed}"


def render_arm(scale: str, seed: str) -> tuple[str, str]:
    name = arm_name(scale, seed)
    overrides = {
        "--pretrained_path": SCALES[scale][1],
        "--learning_rate": LEARNING_RATE,
        "--kl_weight": KL_WEIGHT,
        "--temperature": TEMPERATURE,
        "--seed": seed,
        # Pinned: the split must not move with the training seed or the arms are scored
        # on different held-out data and cannot be compared.
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


def description(scale: str, seed: str) -> str:
    label, ckpt = SCALES[scale]
    return (f"CONTRASTIVE SCALE ARM: the {label} encoder, seed {seed} of "
            f"{len(SEEDS)}, at lr {LEARNING_RATE} / KL {KL_WEIGHT} / temperature "
            f"{TEMPERATURE}, from the frozen checkpoint {Path(ckpt).name}. Contrastive "
            f"has only ever been run at 50m; denoise gains with scale, so this asks "
            f"whether the embedding does too. Six seeds per scale because the "
            f"separation ratio has sd 0.75 at a fixed seed (job 8844111) and a single "
            f"run per scale would resolve nothing -- with six, the standard error of a "
            f"scale-to-scale difference is 0.43. lr is 2e-5 rather than the nominal "
            f"grid winner 5e-4: means are indistinguishable at this noise level, so the "
            f"choice falls to worst case, and 5e-4 collapsed to the floor in one of its "
            f"four arms while 2e-5 never did. split_seed is pinned so every arm is "
            f"scored on identical held-out data.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    cli = parser.parse_args()
    for key, (label, ckpt) in SCALES.items():
        if not Path(ckpt, "model.safetensors").exists():
            raise SystemExit(f"{label} not frozen at {ckpt}")
    combos = list(itertools.product(SCALES, SEEDS))

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
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\ncontrastive-scale\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/  "
          f"({len(SCALES)} scales x {len(SEEDS)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
