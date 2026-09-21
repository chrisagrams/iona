"""Does contrastive training improve with more epochs, now that batches reshuffle?

    python sweeps/make_contrastive_epochs_grid.py --clean

REPLACES A RECORD THAT WAS NEVER RESOLVABLE. `configs/sweep-contrastive-epochs` reported
6.94 at 3 epochs, 4.82 at 10 and 4.46 at 50, and that "more epochs hurts" has been cited
since. It was four SINGLE runs at lr 5e-4, the least stable rate in the whole grid --
its four arms elsewhere span 1.35 (total collapse) to 7.83. A monotone-looking sequence
of four draws from that distribution carries no information about epochs at all.

It was also run under FT14: `GroupBatchSampler` never advanced its epoch counter, so
every epoch replayed byte-identical batches. On the real corpus at P=2 K=2 that is 1,796
of 11,674 rows -- 15.4% -- for 3 epochs and the same 15.4% for 50. "More epochs" meant
strictly more passes over one fixed slice, which is the setup under which more epochs
SHOULD hurt. With the sampler fixed, coverage grows:

    epoch   1     2     3     5    10    20    30
    rows   15%   29%   40%   57%   82%   ~95%  ~98%

so epochs and data are no longer independent knobs, and the old answer cannot carry over.

DESIGN. lr 2e-5 / temperature 0.07 / KL 10 -- the stable configuration used by
`sweep-contrastive-scale`, not the 5e-4 that collapsed. Four seeds per epoch count puts
the standard error of an epoch-to-epoch difference at 0.75*sqrt(2/4) = 0.53, against an
old claimed effect of 2.5.

THE 3-EPOCH ARM IS A CONTROL, NOT A DATA POINT. It is the scale grid's 50m arm with one
thing changed, the sampler, and the scale grid measured that at 5.86 +/- 0.63 over six
seeds. If ep3 here lands far from 5.86 the difference is attributable to FT14 and
nothing else; if it lands on top of it, FT14 cost nothing at 3 epochs and whatever the
longer arms show is about epochs.

Arms sort ep003 < ep010 < ep030 < ep100 so the validation runner's "last arm is the
heaviest" rule picks the 100-epoch arm, which is the one whose runtime is untested.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
OUT = REPO / "configs" / "sweep-contrastive-epochs-v2"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_conep-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"
CHECKPOINT = f"{FROZEN}/msdelta-50m-production-01-checkpoint-133233"

LEARNING_RATE, KL_WEIGHT, TEMPERATURE = "2e-5", "10", "0.07"
SEEDS = ("0", "1", "2", "3")
EPOCHS = ("3", "10", "30", "100")
# Measured: 268 s for 3 epochs at 50m (job 8845057), so ~89 s/epoch.
SECONDS_PER_EPOCH = 89


def arm_name(epochs: str, seed: str) -> str:
    return f"ep{int(epochs):03d}_seed{seed}"


def render_arm(epochs: str, seed: str) -> tuple[str, str]:
    name = arm_name(epochs, seed)
    overrides = {
        "--pretrained_path": CHECKPOINT,
        "--learning_rate": LEARNING_RATE,
        "--kl_weight": KL_WEIGHT,
        "--temperature": TEMPERATURE,
        "--num_train_epochs": epochs,
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


def description(epochs: str, seed: str) -> str:
    control = ("  THIS IS THE FT14 CONTROL: identical to the scale grid's 50m arm "
               "except that the sampler now reshuffles, and that measured 5.86 +/- 0.63 "
               "over six seeds.\n" if epochs == "3" else "")
    return (f"CONTRASTIVE EPOCHS ARM: {epochs} epochs, seed {seed} of {len(SEEDS)}, on "
            f"the 50m encoder at lr {LEARNING_RATE} / KL {KL_WEIGHT} / temperature "
            f"{TEMPERATURE}. Re-takes an epochs sweep whose record (6.94 / 4.82 / 4.46 "
            f"at 3 / 10 / 50) was four single runs at lr 5e-4, a rate that collapsed to "
            f"the 1.35 floor in another arm -- so the old 'more epochs hurts' was four "
            f"draws from a wide distribution, not a measurement. It also predates the "
            f"FT14 fix, under which every epoch replayed the identical 15.4% of the "
            f"corpus; epochs now buy data (3 epochs sees 40%, 10 sees 82%), so the old "
            f"answer cannot carry over. Four seeds put the standard error of an "
            f"epoch-to-epoch difference at 0.53.\n{control}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    cli = parser.parse_args()
    if not Path(CHECKPOINT, "model.safetensors").exists():
        raise SystemExit(f"50m not frozen at {CHECKPOINT}")
    combos = list(itertools.product(EPOCHS, SEEDS))

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
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\ncontrastive-epochs-v2\n")
    longest = max(int(e) for e in EPOCHS) * SECONDS_PER_EPOCH / 3600
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/  "
          f"({len(EPOCHS)} epoch counts x {len(SEEDS)} seeds)")
    print(f"  longest arm ~{longest:.1f} h at {SECONDS_PER_EPOCH} s/epoch (measured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
