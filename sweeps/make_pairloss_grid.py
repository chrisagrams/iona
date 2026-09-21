"""The pair formulation: independent per-pair terms instead of in-batch softmax.

    python sweeps/make_pairloss_grid.py --clean

WHY. PK sampling hands a whole batch to a softmax loss, which couples every row to
every other, so the batch cannot be split and memory caps how many peptides a step can
see. DeltaMZBias is O(batch * peaks^2); that is what forced P=2, K=2 and left every
anchor with one positive and two negatives from a single fixed other peptide.

A pair loss decomposes. Each pair contributes independently, so breadth comes from
gradient_accumulation_steps rather than from a batch that must fit at once, and the
in-group/out-group balance becomes a dial (`positive_fraction`) instead of a
consequence of P and K.

SHAPE. 4 pairs per minibatch = 8 spectra, with gradient_accumulation_steps 16, so an
optimizer step still sees 64 pairs at the memory of 8 spectra. The first attempt used
16 spectra and took a GPU scratch fault (job 8845056, the same 0xff00.... signature as
FT9); shrinking the minibatch while raising accumulation keeps the breadth and is
exactly the freedom a decomposable loss provides -- a softmax loss could not make that
trade. The sampler reshuffles
itself rather than waiting for set_epoch, which is the FT14 failure by construction.

AXES, 12 arms:
  positive_fraction  0.25 / 0.5 / 0.75   the balance, the thing PK could not control
  pair_margin        0.5 / 1.0           how far apart different peptides must get;
                                         embeddings are unit-norm so d is in [0,2] and
                                         d^2 = 2-2cos, making 1.0 ask for cosine <= 0.5
  learning_rate      2e-5 / 1e-4         2e-5 is the robust choice from the softmax
                                         grid; 1e-4 also never collapsed. 5e-4 is
                                         excluded: it owns both the best single draw
                                         and a collapse to the floor.

READING IT. The softmax reference is 5.75 +/- 0.75 over six repeats, and the random
floor is 1.35. With sd ~0.8 and one run per arm, only a difference above ~2 means
anything, so this grid is a SCREEN for whether the formulation is viable at all --
whether any arm clears the floor and lands near the softmax band. Whichever arms look
best then need repeats before being compared to anything.

WHAT IT GIVES UP, so the comparison is read fairly: a softmax loss puts every negative
in one denominator, so the hardest negative automatically takes the most gradient. Pair
terms weight negatives equally and go to exactly zero past the margin. If pair loss
matches the softmax band here, its ability to scale the number of peptides per step
without GradCache makes it the better foundation; if it lands well below, that
hard-negative weighting is the likely reason and hard-negative mining is the next step
rather than more tuning of these axes.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
OUT = REPO / "configs" / "sweep-pairloss"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_pair-"

POSITIVE_FRACTIONS = ("0.25", "0.5", "0.75")
MARGINS = ("0.5", "1.0")
LEARNING_RATES = ("2e-5", "1e-4")
PAIRS_PER_BATCH = "4"
ACCUMULATION = "16"


def arm_name(fraction: str, margin: str, lr: str) -> str:
    return (f"pf{fraction.replace('.', '')}_m{margin.replace('.', '')}"
            f"_lr{lr.replace('-', '')}")


def render_arm(fraction: str, margin: str, lr: str) -> tuple[str, str]:
    name = arm_name(fraction, margin, lr)
    overrides = {
        "--pair_loss": "true",
        "--pairs_per_batch": PAIRS_PER_BATCH,
        "--positive_fraction": fraction,
        "--pair_margin": margin,
        "--learning_rate": lr,
        "--gradient_accumulation_steps": ACCUMULATION,
        # per_device_train_batch_size is not consulted -- the sampler supplies whole
        # batches -- but it is set to match so the two do not disagree in the logs.
        "--per_device_train_batch_size": str(2 * int(PAIRS_PER_BATCH)),
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


def description(fraction: str, margin: str, lr: str) -> str:
    pairs = int(PAIRS_PER_BATCH)
    return (f"PAIR-LOSS ARM: independent per-pair terms instead of in-batch softmax. "
            f"{pairs} pairs per minibatch ({2 * pairs} spectra) with "
            f"gradient_accumulation_steps {ACCUMULATION}, so an optimizer step sees "
            f"{pairs * int(ACCUMULATION)} pairs at the memory of {2 * pairs} spectra -- "
            f"the point of the formulation, since pair terms decompose and softmax "
            f"terms do not. positive_fraction {fraction} sets the in-group/out-group "
            f"balance directly, which PK sampling could only change by changing the "
            f"batch shape. pair_margin {margin}: different peptides are pushed to at "
            f"least this distance and then ignored; embeddings are unit-norm so d lies "
            f"in [0,2] and d^2 = 2-2cos, making 1.0 ask for cosine <= 0.5. lr {lr}, "
            f"chosen from the settings that never collapsed in the softmax grid. Read "
            f"against a softmax reference of 5.75 +/- 0.75 and a random floor of 1.35; "
            f"with one run per arm only a gap above ~2 is meaningful, so this is a "
            f"viability screen and the best arms need repeats afterwards.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    cli = parser.parse_args()
    combos = list(itertools.product(POSITIVE_FRACTIONS, MARGINS, LEARNING_RATES))

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
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\npairloss\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
