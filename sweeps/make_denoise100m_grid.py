"""Generate the 100m denoise grid, narrowed by what the 50m grid measured.

    python sweeps/make_denoise100m_grid.py --clean
    python sweeps/make_denoise100m_grid.py --check

The 50m grid ran 216 arms across five axes. Two of them turned out not to matter, and
re-sweeping those at twice the cost per arm would buy nothing:

  encoder_lr_scale  spread 0.1114 -- KEPT, minus es0. A frozen encoder averages 0.7605
                    against 0.8719 unfrozen, so it is not a candidate, only a control,
                    and the 50m grid already ran that control 54 times.
  learning_rate     spread 0.1086 -- KEPT and EXTENDED UPWARD. 2e-4 was both the best
                    value and the top of the range, which means the optimum was never
                    bracketed; 5e-4 finds out. 5e-5 stays because a larger model often
                    prefers a lower rate and dropping the low end would assume otherwise.
  batch             spread 0.0340 -- KEPT as {12, 48}, dropping 144 which was worst. b12
                    means ~29k optimizer updates against b48's ~7k.
  num_train_epochs  spread 0.0124 -- FIXED at 4. Worth 0.012 AUROC, which does not
                    justify doubling the grid.
  head_hidden_size  spread 0.0072 -- FIXED at 512. The three widths span 0.007, which is
                    noise; 512 is nominally best and costs nothing at this scale.

Twelve arms, one node each, roughly three hours.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-denoise-100m-ds" / "training.args"
OUT = REPO / "configs" / "sweep-denoise-100m"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_dn100m-"

LEARNING_RATES = ("5e-5", "2e-4", "5e-4")
ENCODER_SCALES = ("0.5", "1.0")
BATCHES = {"12": ("1", "1"), "48": ("4", "1")}
EPOCHS = "4"
HEAD = "512"


def arm_name(lr: str, scale: str, batch: str) -> str:
    return f"lr{lr.replace('-', '')}_es{scale.replace('.', '')}_b{batch}"


def render_arm(lr: str, scale: str, batch: str) -> tuple[str, str]:
    name = arm_name(lr, scale, batch)
    per_device, accumulation = BATCHES[batch]
    overrides = {
        "--learning_rate": lr,
        "--encoder_lr_scale": scale,
        "--num_train_epochs": EPOCHS,
        "--head_hidden_size": HEAD,
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


def description(lr: str, scale: str, batch: str) -> str:
    shared = TEMPLATE.parent / "DESCRIPTION.md"
    lead = " ".join(shared.read_text().split()) + " " if shared.exists() else ""
    updates = 87_200 * int(EPOCHS) // int(batch)
    return (f"{lead}GRID ARM: lr={lr}, encoder at {scale}x the head's rate, effective "
            f"batch {batch} giving about {updates:,} optimizer updates, {EPOCHS} epochs, "
            f"head width {HEAD}. Epochs and head width are fixed rather than swept "
            f"because the 50m grid measured their spread at 0.012 and 0.007 AUROC "
            f"respectively, against 0.111 for encoder_lr_scale and 0.109 for the "
            f"learning rate. lr 5e-4 extends past the 50m range, where 2e-4 was both "
            f"the best value and the boundary, so the optimum was never bracketed.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    cli = parser.parse_args()
    combos = list(itertools.product(LEARNING_RATES, ENCODER_SCALES, BATCHES))

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
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\ndenoise100m\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/")
    for combo in combos:
        print(f"  {arm_name(*combo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
