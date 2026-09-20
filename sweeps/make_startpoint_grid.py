"""Which pretraining checkpoint is the best STARTING POINT for contrastive fine-tuning?

    python sweeps/make_startpoint_grid.py --clean
    python sweeps/make_startpoint_grid.py --check

Every contrastive run so far started from checkpoint-133233, which is simply what the
published `final/` happened to be. Nothing has checked that it is a good choice.

The reason to ask now: probing the 50m along its own pretraining trajectory (job
8842038) found the FROZEN separation ratio is not monotone in pretraining step --

    step  10,000   1.67   <- the only reading in this project that beats the floor
    step  50,000   1.33
    step 100,000   1.38
    step 133,233   1.43   <- what every fine-tune has used
    step 180,000   1.35
    random floor   1.35

-- so early pretraining briefly carries linearly-readable replicate structure and then
trains it away, presumably as the encoder specialises for predicting masked intensities.

WHY THE ANSWER IS NOT OBVIOUS. Frozen readability does NOT predict fine-tuned
performance: frozen-pretrained equals random at 1.35, yet fine-tuned it reaches 7.83
while random stays at 1.35. So a better frozen ratio at 10,000 steps is not evidence of
a better starting point, and the two outcomes say opposite things:

  10k WINS  -> the linearly-readable structure matters after all, and every contrastive
               run to date started from the wrong checkpoint.
  10k LOSES -> the useful structure is the non-linear kind, it accumulates with
               pretraining, and 1.67 was a red herring. Closes the embedding axis.

PREDICTION, recorded so it can be wrong: 10k loses, and not narrowly. Pretraining is
worth more than 10x the fine-tuning budget (job 8843262), and checkpoint-10000 has had
13x LESS pretraining than 133,233.

Hyperparameters are pinned at the best contrastive arm -- lr 5e-4, KL 10, temperature
0.07 -- so the checkpoint is the only thing that varies.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
OUT = REPO / "configs" / "sweep-startpoint"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_start-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

# The best contrastive arm, held fixed.
LEARNING_RATE, KL_WEIGHT, TEMPERATURE = "5e-4", "10", "0.07"
# step -> the frozen copy holding it. 133233 is the incumbent.
STEPS = {
    "10000": f"{FROZEN}/msdelta-50m-production-01-checkpoint-10000",
    "50000": f"{FROZEN}/msdelta-50m-production-01-checkpoint-50000",
    "133233": f"{FROZEN}/msdelta-50m-production-01-checkpoint-133233",
}


def arm_name(step: str) -> str:
    return f"step{step}"


def render_arm(step: str) -> tuple[str, str]:
    name = arm_name(step)
    overrides = {
        "--pretrained_path": STEPS[step],
        "--learning_rate": LEARNING_RATE,
        "--kl_weight": KL_WEIGHT,
        "--temperature": TEMPERATURE,
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


def description(step: str) -> str:
    shared = TEMPLATE.parent / "DESCRIPTION.md"
    lead = " ".join(shared.read_text().split()) + " " if shared.exists() else ""
    frozen = {"10000": "1.67", "50000": "1.33", "133233": "1.43"}[step]
    role = ("the incumbent: what every contrastive run so far has used, because it is "
            "what the published final/ happened to be"
            if step == "133233" else
            "the only checkpoint whose FROZEN embedding beats the 1.35 random floor"
            if step == "10000" else
            "the trough, included so the comparison is a curve and not two points")
    return (f"{lead}START-POINT ARM: contrastive fine-tuning from pretraining "
            f"checkpoint {step} of the 50m, at the best arm's hyperparameters "
            f"(lr {LEARNING_RATE}, KL {KL_WEIGHT}, temperature {TEMPERATURE}) so the "
            f"checkpoint is the only variable. This one is {role}. Its frozen "
            f"separation ratio is {frozen}, against a random-init floor of 1.35. The "
            f"question is whether frozen readability predicts anything about "
            f"fine-tuned quality -- it demonstrably does not across the "
            f"pretrained/random comparison, where 1.35 == 1.35 frozen but 7.83 vs 1.35 "
            f"trained, so a higher frozen ratio here is not evidence of a better "
            f"start.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    cli = parser.parse_args()
    for step, path in STEPS.items():
        if not Path(path, "model.safetensors").exists():
            raise SystemExit(f"checkpoint {step} not frozen at {path}")

    if cli.check:
        stale = [n for n, text in (render_arm(s) for s in STEPS)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale)}")
            return 1
        print(f"  {len(STEPS)} arms match {TEMPLATE.relative_to(REPO)}")
        return 0

    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for step in STEPS:
        name, text = render_arm(step)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(step))
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\nstartpoint\n")
    print(f"arms={len(STEPS)} under {OUT.relative_to(REPO)}/")
    for step in STEPS:
        print(f"  {arm_name(step):<12} checkpoint-{step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
