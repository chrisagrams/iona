"""Does contrastive fine-tuning improve with encoder scale? With repeats, this time.

    python sweeps/make_contrastive_scale_grid.py --clean

Contrastive has only ever been run on the 50m. Denoise gains with scale (0.9320 ->
0.9403 -> 0.9446 at 50m/100m/200m), so the obvious question is whether the embedding
does too -- and it is the one remaining scale question in the project.

REPEATS ARE BUILT IN, NOT OPTIONAL. The separation ratio has sd 0.75 at a FIXED seed
(job 8844111: six identical runs gave 6.56 5.89 6.02 5.46 4.40 6.20, mean 5.75). A
single run per scale could not resolve anything: the standard error on a difference of
two single runs is 1.06, which is larger than the entire denoise scale effect expressed
proportionally. Six seeds per scale puts the standard error of a scale-to-scale
difference at 0.75*sqrt(2/6) = 0.43, so a real gap of ~1.3 becomes detectable.

That is also why this grid exists rather than four more single runs: the lesson from the
7.83 figure -- which turned out to be 2.8 sd above its own configuration's mean and then
anchored a dozen later comparisons -- is that one draw from this metric is not a
measurement.

HYPERPARAMETERS: lr 2e-5, KL 10, temperature 0.07. NOT the nominal grid winner, which
was lr 5e-4. With means indistinguishable the choice falls to worst case, and the four
arms at each rate show:

    lr 2e-5   min 5.33  max 7.71   never collapsed
    lr 1e-4   min 5.66  max 7.14   never collapsed
    lr 5e-4   min 1.35  max 7.83   COLLAPSED to the floor in one arm

lr 5e-4 owns both the best single draw in the grid and a total collapse, which is what
training at the edge of stability looks like. 2e-5 has the best floor. Choosing it also
keeps a scale comparison from being dominated by which scales happened to fall off the
edge.

Arm names are s050m/s100m/s200m/s400m so that sorting puts the largest last -- the
validation runner picks the last arm by name as its heaviest, and 400m is the one whose
memory is untested for this path.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
OUT = REPO / "configs" / "sweep-contrastive-scale"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_conscale-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

LEARNING_RATE, KL_WEIGHT, TEMPERATURE = "2e-5", "10", "0.07"
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
