"""Generate the from-scratch denoise ablation.

    python sweeps/make_scratch_grid.py --clean

Twelve arms on the 50m architecture with random weights, and the axes are deliberately
NOT the ones used for the pretrained size grids.

`encoder_lr_scale` is dropped and fixed at 1.0. It exists to slow the encoder relative to
a freshly initialised head so pretrained weights are not trampled before the head can
learn. A random encoder has nothing to protect, so scaling its rate down is not a
hyperparameter worth spending arms on -- it is just training more slowly.

`num_train_epochs` takes its place, at {4, 8}. The pretrained grid measured only 0.012
AUROC between 2 and 4 epochs because the encoder starts nearly right. A random encoder
has to learn the representation as well as the task, so the training length that suited
a pretrained start is exactly the assumption an ablation should not inherit -- and
under-training the control would manufacture the conclusion that pretraining helps.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-denoise-50m-scratch-ds" / "training.args"
OUT = REPO / "configs" / "sweep-denoise-scratch"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_dn50mscratch-"

LEARNING_RATES = ("5e-5", "2e-4", "5e-4")
EPOCHS = ("4", "8")
BATCHES = {"12": ("1", "1"), "48": ("4", "1")}
HEAD = "512"


def arm_name(lr: str, epochs: str, batch: str) -> str:
    return f"lr{lr.replace('-', '')}_ep{epochs}_b{batch}"


def render_arm(lr: str, epochs: str, batch: str) -> tuple[str, str]:
    name = arm_name(lr, epochs, batch)
    per_device, accumulation = BATCHES[batch]
    overrides = {
        "--learning_rate": lr, "--num_train_epochs": epochs,
        "--encoder_lr_scale": "1.0", "--head_hidden_size": HEAD,
        "--random_init": "true",
        "--per_device_train_batch_size": per_device,
        "--gradient_accumulation_steps": accumulation,
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


def description(lr: str, epochs: str, batch: str) -> str:
    shared = TEMPLATE.parent / "DESCRIPTION.md"
    lead = " ".join(shared.read_text().split()) + " " if shared.exists() else ""
    return (f"{lead}ABLATION ARM: lr={lr}, {epochs} epochs, effective batch {batch}, "
            f"encoder at full rate and head width {HEAD}. The pretrained 50m grid's best "
            f"arm reached 0.9320 test AUROC and its frozen-encoder arms averaged 0.760; "
            f"where this lands between those two numbers is what pretraining is worth.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    cli = parser.parse_args()
    combos = list(itertools.product(LEARNING_RATES, EPOCHS, BATCHES))
    if cli.check:
        stale = [n for n, text in (render_arm(*c) for c in combos)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text]
        if stale:
            print(f"  {len(stale)} stale: {', '.join(stale[:6])}")
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
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\nscratch\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/")
    for combo in combos:
        print(f"  {arm_name(*combo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
