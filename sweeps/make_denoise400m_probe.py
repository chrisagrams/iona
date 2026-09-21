"""Is 400m's plateau real, and if so do the untested axes move it?

    python sweeps/make_denoise400m_probe.py --clean

400m scored 0.9436 against 200m's 0.9446 -- a deficit of 0.0010 against a top-cluster
spread of 0.0013. Two questions are tangled there and this grid separates them in one
job of twelve arms.

IS THERE ANYTHING TO EXPLAIN (3 arms). Three repeats of the 400m winner, nothing
varied. Denoise run-to-run variance has never been measured at any scale, and the
contrastive metric turned out to have sd 0.75 with a headline figure 2.8 sd above its
own mean. If the 400m repeats span more than 0.001, the 200m-vs-400m gap is noise and
there is no plateau to explain.

IF THERE IS, DO THE UNTESTED AXES MOVE IT (9 arms). The size grids swept lr,
encoder_lr_scale and batch, and the 400m optimum for lr is 2e-4 -- the MIDDLE value,
with both 5e-5 and 5e-4 worse -- so the lr range is not the constraint. What was never
tested above 50m:

  encoder_lr_scale 0.1 / 0.25 / 0.5   the size grids only ever used 0.5 and 1.0, while
                                      the 216-arm 50m grid also covered 0 and 0.1. A
                                      larger encoder has more pretrained structure to
                                      lose, so the gentler updates are exactly where a
                                      400m would differ from a 200m.
  num_train_epochs 2 / 4 / 8          fixed at 4 for every grid above 50m. The scratch
                                      ablation showed a random encoder wants 8, and
                                      nothing has checked whether a 400m wants 2.

lr is held at 2e-4 and batch at 12, both confirmed winners at every scale tested, so
the nine arms are a clean 3x3 over the two axes that were never looked at.

WHAT IT CANNOT FIX. If the 400m pretrained checkpoint is simply undertrained for its
capacity -- it sits at 181,381 steps, 33.6% of schedule, having seen no more data than
the 200m -- no fine-tuning hyperparameter will recover that, and the answer is more
pretraining rather than a better search. The repeats plus the 3x3 will say which
situation we are in.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-denoise-400m-ds" / "training.args"
OUT = REPO / "configs" / "sweep-denoise400m-probe"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_dn400mprobe-"

LEARNING_RATE = "2e-4"          # interior optimum at 400m, confirmed
BATCH = ("1", "1")              # per_device 1 x 12 tiles = effective 12, the winner
HEAD = "512"
SCALES = ("0.1", "0.25", "0.5")
EPOCHS = ("2", "4", "8")
REPEATS = 3                     # of the incumbent: es 0.5, 4 epochs


def arm_name(scale: str, epochs: str) -> str:
    return f"es{scale.replace('.', '')}_ep{epochs}"


def render(overrides: dict) -> str:
    lines, seen = [], set()
    tokens = TEMPLATE.read_text().split()
    for flag, value in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{flag} {overrides.get(flag, value)}")
        seen.add(flag)
    for flag, value in overrides.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")
    return "\n".join(lines) + "\n"


def base(name: str) -> dict:
    per_device, accumulation = BATCH
    return {
        "--learning_rate": LEARNING_RATE,
        "--head_hidden_size": HEAD,
        "--per_device_train_batch_size": per_device,
        "--gradient_accumulation_steps": accumulation,
        "--run_name": f"{RUN_PREFIX}{name}",
        "--output_dir": f"./runs/{RUN_PREFIX}{name}",
    }


def arms() -> list[tuple[str, str]]:
    out = []
    for scale, epochs in itertools.product(SCALES, EPOCHS):
        name = arm_name(scale, epochs)
        out.append((name, render(base(name) | {"--encoder_lr_scale": scale,
                                               "--num_train_epochs": epochs})))
    for i in range(REPEATS):
        name = f"repeat{i}"
        out.append((name, render(base(name) | {"--encoder_lr_scale": "0.5",
                                               "--num_train_epochs": "4"})))
    return out


def description(name: str) -> str:
    if name.startswith("repeat"):
        return ("400m PROBE, REPEAT ARM: the current 400m winner (lr 2e-4, "
                "encoder_lr_scale 0.5, 4 epochs, batch 12) with nothing varied. Denoise "
                "run-to-run variance has never been measured at any scale, and 400m's "
                "0.9436 sits 0.0010 below 200m's 0.9446 against a top-cluster spread of "
                "0.0013. If these three repeats span more than about 0.001 then the "
                "plateau is noise and there is nothing to explain. Read alongside the "
                "nine 3x3 arms in the same job.\n")
    scale, epochs = name.split("_")
    return (f"400m PROBE ARM: encoder_lr_scale {scale[2:]} and {epochs[2:]} epochs, at "
            f"lr {LEARNING_RATE} and batch 12, both confirmed winners at every scale. "
            f"These are the two axes never swept above 50m -- the size grids used only "
            f"encoder_lr_scale 0.5 and 1.0 and pinned epochs at 4, while the 216-arm "
            f"50m grid also covered 0 and 0.1. A larger encoder holds more pretrained "
            f"structure to lose, so gentler updates are where a 400m would plausibly "
            f"differ from a 200m; and a 400m may want fewer or more passes than the 4 "
            f"inherited from smaller models. lr is not a candidate: the 400m optimum is "
            f"2e-4, the middle of the range, with both neighbours worse.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    cli = parser.parse_args()
    if not TEMPLATE.exists():
        raise SystemExit(f"no template at {TEMPLATE}")
    built = arms()

    if cli.check:
        stale = [n for n, text in built
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(built)} arms match {TEMPLATE.relative_to(REPO)}")
        return 0

    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for name, text in built:
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(name))
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\ndenoise400m-probe\n")
    print(f"arms={len(built)} under {OUT.relative_to(REPO)}/  "
          f"({len(SCALES)}x{len(EPOCHS)} new + {REPEATS} repeats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
