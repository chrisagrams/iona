"""How much does the contrastive separation ratio vary run to run at a FIXED config?

    python sweeps/make_repeat_grid.py --clean

Discovered by accident: the arm lr5e4_kl10_t007 scored 7.83 in job 8842232 and its
byte-identical twin scored 6.01 in job 8843838. Same config, same seed 0, verified by
diff -- so this is run-to-run nondeterminism, not seed variance. Likely sources are XPU
float nondeterminism and dataloader worker ordering at dataloader_num_workers 4.

A gap of 1.82 is larger than most effects reported on this metric:

  scheduler fix 6.94 -> 7.83 ............ 0.89   inside it
  best arm 7.83 vs second 7.71 .......... 0.12   inside it
  random budget 1.35 -> 2.73 ............ 1.38   inside it
  layer-mix 4.14 vs matched 6.21 ........ 2.07   barely outside
  start point 6.01 vs 3.91 .............. 2.10   barely outside
  pretrained 7.83 vs random 1.35 ........ 6.48   survives comfortably

n=2 establishes that the variance is large; it does not measure it. This runs the SAME
config six times, changing nothing at all, so the spread can be quantified and every
comparison above re-read against a real error bar.

Two things it will also settle. Whether the variance scales with the mean -- if it does,
the random-encoder results near 1.35 are tighter than this and the budget curve may
survive after all. And whether dropping dataloader workers to 0 removes it, which is
the cheapest available fix if worker ordering is the cause.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
OUT = REPO / "configs" / "sweep-repeat"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_repeat-"

LEARNING_RATE, KL_WEIGHT, TEMPERATURE = "5e-4", "10", "0.07"
REPEATS = 6


def arm_name(i: int) -> str:
    return f"run{i}"


def render_arm(i: int) -> tuple[str, str]:
    name = arm_name(i)
    overrides = {
        "--learning_rate": LEARNING_RATE,
        "--kl_weight": KL_WEIGHT,
        "--temperature": TEMPERATURE,
        # seed stays at whatever the template says, deliberately: the point is to
        # measure variance at a FIXED seed, which is what the two observed runs had.
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


def description(i: int) -> str:
    return (f"REPEAT ARM {i} of {REPEATS}: the best contrastive configuration "
            f"(lr {LEARNING_RATE}, KL {KL_WEIGHT}, temperature {TEMPERATURE}), run "
            f"with NOTHING varied -- same seed, same data, same everything. The arm "
            f"lr5e4_kl10_t007 scored 7.83 in job 8842232 and 6.01 in job 8843838 from "
            f"byte-identical configs, so the separation ratio carries run-to-run "
            f"nondeterminism of at least 1.82 at fixed seed. That is larger than most "
            f"differences reported on this metric, including the 0.89 'improvement' "
            f"attributed to the scheduler fix. Six repeats turn an anecdote into an "
            f"error bar, after which every contrastive comparison in STATUS.md and "
            f"OBSERVATIONS.md has to be re-read against it.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    cli = parser.parse_args()

    if cli.check:
        stale = [n for n, text in (render_arm(i) for i in range(REPEATS))
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale)}")
            return 1
        print(f"  {REPEATS} arms match {TEMPLATE.relative_to(REPO)}")
        return 0

    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for i in range(REPEATS):
        name, text = render_arm(i)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(i))
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\nrepeat\n")
    print(f"arms={REPEATS} under {OUT.relative_to(REPO)}/  (all identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
