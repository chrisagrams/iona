"""
RE-TAKEN AT A CONFIGURATION THAT ACTUALLY TRAINS. The first version of this grid ran at
lr 2e-5 / KL 0 / temperature 0.2, chosen by the separation ratio. Scored on MAP@100 that
point is statistically indistinguishable from NOT TRAINING AT ALL -- level with an
untrained encoder (p=0.98) and worse than it on Hit@1 (p=0.003) -- because the ratio
does not predict retrieval (OBSERVATIONS.md). KL 0 removes the leash to the pretrained
weights and t 0.2 is too soft; the two together cost more than either alone. This grid
now runs lr 1e-4 / KL 10 / t 0.07, top-ranked in all four scale x checkpoint cells.
The contrastive pretraining ablation, matched to the corrected scale curve.

    python sweeps/make_contrastive_random_v2.py --clean

WHAT THE OLD CONTROL SAID. sweep-contrastive-random (job 8842288) ran the same twelve
hyperparameter cells on a randomly initialised encoder, and every one landed on the
1.35 floor -- 1.34 to 1.36 across the whole grid, where the pretrained encoder reached
7.83. Contrastive training on a random encoder learns nothing at all. For denoise,
pretraining is worth a finite +0.033; for contrastive it looks like a precondition.

WHY IT NEEDS RE-RUNNING. That control inherited two things the pretrained side no
longer has. It predates the FT14 sampler fix, and it used lr 2e-5 / KL 10 / temperature
0.07, which the 96-arm HP grid ranks seventh of twelve -- the pretrained arms it was
compared against scored 7.83 where the corrected configuration now reaches 9.37 at 50m.
The random side has no room to fall, so the gap can only widen, but a headline claim
should be matched rather than inherited.

ONE THING THE NEW CONFIGURATION FIXES ON ITS OWN. The old grid's KL arms regularised a
random encoder toward a SEPARATE random function, which is not a control of anything;
only its kl_weight 0 arms were honest. The corrected configuration has KL 0, so every
arm here is clean.

Identical to sweep-contrastive-scale-v2 in every respect but --random_init true: same
four scales, same six seeds, same lr 2e-5 / KL 0 / temperature 0.2. The two grids
subtract.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
OUT = REPO / "configs" / "sweep-contrastive-scale-v2-random"
STAMP = OUT / ".template"
RUN_PREFIX = "v3_conrand-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

LEARNING_RATE, KL_WEIGHT, TEMPERATURE = "1e-4", "10", "0.07"
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
        # The architecture comes from this checkpoint; the weights do not. finetune_
        # contrastive builds from_config rather than loading and reinitialising, so no
        # buffer keeps a pretrained value.
        "--random_init": "true",
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
    return (f"CONTRASTIVE RANDOM-INIT ARM: the {label} encoder, seed {seed} of "
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
