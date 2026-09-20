"""Give the random encoder a real budget, so "it cannot learn" is a measurement.

    python sweeps/make_random_budget_grid.py --clean
    python sweeps/make_random_budget_grid.py --check

Job 8842288 ran the contrastive grid on a randomly initialised encoder and every one of
the twelve arms sat between 1.34 and 1.36 -- the random-init floor -- while the same
arms on a pretrained encoder reached 7.83. The conclusion drawn was that pretraining is
required for the contrastive objective to move at all.

The hole is smaller than it first looked, and the correction matters. Those runs are
1,347 optimizer steps, not the 90 reported earlier: the replicate corpus has 898 TRAINING
GROUPS and the PK sampler yields 449 batches an epoch at groups_per_batch 2, over 3
epochs. The 60-group figure came from job 8840665, a smoke run with max_samples applied,
and was mistaken for the real corpus.

So the random encoder already had 1,347 steps and did not move off 1.35. What remains
open is only whether it would move given MORE, which is a question about how much extra
training pretraining saves rather than about whether the result is real:

    3 epochs ->  1,347 steps   the budget already run, as the anchor
   10 epochs ->  4,490 steps
   30 epochs -> 13,470 steps   10x the original, ~45 min on one tile

Reading it: a ratio still pinned at 1.35 after 3000 steps makes the claim about
pretraining, not about the budget. A ratio that climbs means the original conclusion was
about step count and has to be withdrawn.

KL is kept at 10 to match the reference arm exactly, with the caveat that for a random
encoder it regularises toward a separate random model rather than toward anything worth
preserving. The kl0 arms of 8842288 were equally flat, so this is not what decides it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
OUT = REPO / "configs" / "sweep-random-budget"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_conrandbudget-"

# The best pretrained arm, held fixed so only the budget varies.
LEARNING_RATE = "5e-4"
KL_WEIGHT = "10"
TEMPERATURE = "0.07"
EPOCHS = ("3", "10", "30")
GROUPS, GROUPS_PER_BATCH = 898, 2


def steps_for(epochs: str) -> int:
    return GROUPS // GROUPS_PER_BATCH * int(epochs)


def arm_name(epochs: str) -> str:
    return f"rand_ep{epochs}"


def render_arm(epochs: str) -> tuple[str, str]:
    name = arm_name(epochs)
    overrides = {
        "--random_init": "true",
        "--learning_rate": LEARNING_RATE,
        "--kl_weight": KL_WEIGHT,
        "--temperature": TEMPERATURE,
        "--num_train_epochs": epochs,
        # Warmup has to track the budget: a fixed 5 is 6% of 90 steps and 0.2% of 3000.
        "--warmup_steps": str(max(1, round(0.06 * steps_for(epochs)))),
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


def description(epochs: str) -> str:
    shared = TEMPLATE.parent / "DESCRIPTION.md"
    lead = " ".join(shared.read_text().split()) + " " if shared.exists() else ""
    return (f"{lead}RANDOM-ENCODER BUDGET ARM: no pretrained weights, {epochs} epochs = "
            f"about {steps_for(epochs):,} optimizer steps, at the best pretrained arm's "
            f"hyperparameters (lr {LEARNING_RATE}, KL {KL_WEIGHT}, temperature "
            f"{TEMPERATURE}). Job 8842288 found every random arm pinned at the 1.35 "
            f"floor while the pretrained encoder reached 7.83, but those runs were only "
            f"~90 steps, where 'random cannot learn this' and 'random cannot learn this "
            f"YET' predict the same number. This varies only the budget. Still 1.35 at "
            f"3,000 steps means the result is about pretraining; a climbing ratio means "
            f"it was about step count and the earlier conclusion must be withdrawn.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    cli = parser.parse_args()
    if not TEMPLATE.exists():
        raise SystemExit(f"no template at {TEMPLATE}")

    if cli.check:
        stale = [n for n, text in (render_arm(e) for e in EPOCHS)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale)}")
            return 1
        print(f"  {len(EPOCHS)} arms match {TEMPLATE.relative_to(REPO)}")
        return 0

    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for epochs in EPOCHS:
        name, text = render_arm(epochs)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(epochs))
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\nrandom-budget\n")
    print(f"arms={len(EPOCHS)} under {OUT.relative_to(REPO)}/")
    for e in EPOCHS:
        print(f"  {arm_name(e):<12} {steps_for(e):>5,} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
