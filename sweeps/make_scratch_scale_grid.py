"""The pretraining ablation at every scale, and at a longer budget at 50m.

    python sweeps/make_scratch_scale_grid.py --clean

TWO HOLES IN THE PRETRAINING CLAIM, both closed by this grid.

SCALE. "Pretraining is worth +0.046" has only ever been measured at 50m. Scaling is a
headline conclusion of this project, so whether that advantage grows, shrinks or holds
with model size is not a detail -- a pretraining benefit that vanishes at 200m would
change what the scale curve means. 100m/200m/400m at the matched 4 epochs.

BUDGET. The random encoder was still improving when its grid stopped: 0.8856 at 4
epochs, 0.9001 at 8. So "+0.032 against scratch at double budget" is a LOWER bound on
what scratch reaches, and the honest version of the claim needs to know where scratch
plateaus. 50m at 16 epochs, double again.

Configuration is the scratch grid's own winner -- lr 2e-4, encoder_lr_scale 1.0, head
512, effective batch 12 -- held fixed. Note encoder_lr_scale 1.0, not the 0.5 the
pretrained runs use: a randomly initialised encoder has nothing to preserve, and the
scratch grid selected 1.0 accordingly.

Three seeds per cell. The pretraining gap is ~0.046 against a seed noise of ~0.0005, so
precision is not the constraint here; three is enough to show the gap is not one draw,
and the interesting quantity -- how the gap MOVES with scale -- is the difference of
two means, which three seeds resolve to about 0.0004.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-denoise-scratch-scale"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_dnscr-"

LEARNING_RATE, ENCODER_SCALE, HEAD = "2e-4", "1.0", "512"
PER_DEVICE, ACCUMULATION = "1", "1"
SEEDS = ("1", "2", "3")
# (scale, epochs): the scale sweep at the matched budget, plus 50m at double-double.
CELLS = (("100m", "4"), ("200m", "4"), ("400m", "4"), ("50m", "16"))


def template_for(scale: str) -> Path:
    return REPO / "configs" / f"finetune-denoise-{scale}-ds" / "training.args"


def arm_name(cell, seed):
    scale, epochs = cell
    return f"{scale}_ep{int(epochs):02d}_seed{seed}"


def render_arm(cell, seed) -> tuple[str, str]:
    scale, epochs = cell
    name = arm_name(cell, seed)
    overrides = {
        "--random_init": "true",
        "--learning_rate": LEARNING_RATE,
        "--encoder_lr_scale": ENCODER_SCALE,
        "--head_hidden_size": HEAD,
        "--num_train_epochs": epochs,
        "--per_device_train_batch_size": PER_DEVICE,
        "--gradient_accumulation_steps": ACCUMULATION,
        "--seed": seed,
        "--run_name": f"{RUN_PREFIX}{name}",
        "--output_dir": f"./runs/{RUN_PREFIX}{name}",
    }
    lines, seen = [], set()
    tokens = template_for(scale).read_text().split()
    for flag, value in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{flag} {overrides.get(flag, value)}")
        seen.add(flag)
    for flag, value in overrides.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")
    return name, "\n".join(lines) + "\n"


def description(cell, seed) -> str:
    scale, epochs = cell
    why = ("Closes the BUDGET hole: the random encoder was still improving when its "
           "grid stopped (0.8856 at 4 epochs, 0.9001 at 8), so the pretraining "
           "advantage against scratch-at-double-budget is a lower bound until scratch "
           "plateaus." if scale == "50m" else
           "Closes the SCALE hole: pretraining's +0.046 has only been measured at 50m, "
           "and whether it grows or shrinks with size changes what the scale curve "
           "means.")
    return (f"PRETRAINING ABLATION: {scale} with a RANDOM encoder (--random_init true) "
            f"for {epochs} epochs, seed {seed} of {len(SEEDS)}, at the scratch grid's "
            f"own winning configuration -- lr {LEARNING_RATE}, encoder_lr_scale "
            f"{ENCODER_SCALE} (1.0, not the 0.5 the pretrained runs use: a random "
            f"encoder has nothing to preserve), head {HEAD}, effective batch 12. "
            f"{why}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clean", action="store_true")
    cli = ap.parse_args()
    for scale, _ in CELLS:
        if not template_for(scale).exists():
            raise SystemExit(f"no template for {scale}")
    combos = list(itertools.product(CELLS, SEEDS))

    if cli.check:
        stale = [n for n, text in (render_arm(*c) for c in combos)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(combos)} arms match their per-scale templates")
        return 0

    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for combo in combos:
        name, text = render_arm(*combo)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(*combo))
    STAMP.write_text("per-scale finetune-denoise-*-ds templates\nscratch-scale\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/  "
          f"({len(CELLS)} cells x {len(SEEDS)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
