"""Generate denoise fine-tune sweep arms.

    python sweeps/make_denoise_grid.py --stage core
    python sweeps/make_denoise_grid.py --stage full --dry-run

Two stages, because the full product is mostly waste.

`core` is the full cross of learning_rate x encoder_lr_scale at the baseline epochs and
head size. Those two axes are the ones that genuinely interact: the encoder's effective
rate is their PRODUCT, so `lr=2e-4, scale=0.1` and `lr=5e-5, scale=0.5` put nearly the
same rate on the encoder while putting a 4x different rate on the head. Sweeping either
alone cannot separate "the encoder moved too fast" from "the head moved too slow", and
`encoder_lr_scale=0` is a different model entirely (frozen encoder) rather than a point
on a continuum.

`stage2` then varies epochs and head width around whichever core arm won -- pass
`--base-lr` and `--base-scale` from the core results. Running those axes inside the full
product instead would multiply 12 arms into 72 to answer two questions that are very
unlikely to interact with the first two.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-denoise-50m" / "training.args"
OUT = REPO / "configs" / "sweep-denoise"

LEARNING_RATES = ("1e-5", "5e-5", "2e-4")
ENCODER_SCALES = ("0", "0.1", "0.5", "1.0")
EPOCHS = ("2", "4")
HEAD_SIZES = ("128", "256", "512")


def arm_name(lr: str, scale: str, epochs: str, head: str) -> str:
    """Readable and filesystem-safe, and it sorts sensibly in W&B."""
    return (f"lr{lr.replace('-', '')}_es{scale.replace('.', '')}"
            f"_ep{epochs}_h{head}")


def render_arm(lr: str, scale: str, epochs: str, head: str) -> tuple[str, str]:
    """The arm's name and exactly the bytes its training.args should contain."""
    name = arm_name(lr, scale, epochs, head)
    overrides = {
        "--learning_rate": lr,
        "--encoder_lr_scale": scale,
        "--num_train_epochs": epochs,
        "--head_hidden_size": head,
        "--run_name": f"dn50m-{name}",
        "--output_dir": f"./runs/dn50m-{name}",
    }
    lines, seen = [], set()
    tokens = TEMPLATE.read_text().split()
    for flag, value in zip(tokens[::2], tokens[1::2]):
        if flag in overrides:
            lines.append(f"{flag} {overrides[flag]}")
            seen.add(flag)
        else:
            lines.append(f"{flag} {value}")
    # A flag absent from the template still has to be written, or the arm silently
    # inherits the dataclass default instead of the value the grid asked for.
    for flag, value in overrides.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")

    return name, "\n".join(lines) + "\n"


def write_arm(lr: str, scale: str, epochs: str, head: str, dry_run: bool) -> str:
    name, text = render_arm(lr, scale, epochs, head)
    if not dry_run:
        directory = OUT / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "training.args").write_text(text)
    return name


def check(combos: list[tuple[str, str, str, str]]) -> int:
    """Fail if what is on disk is not what this generator would write today.

    The arms are generated from configs/finetune-denoise-50m/training.args, so every edit
    to that template silently invalidates them. That is not hypothetical: the grid sat at
    `max_peaks 1024` with no `per_device_train_batch_size` long after the template moved
    to 512 and a batch of 4, which is an 8x larger DeltaMZBias tensor -- roughly 118 GB
    against a 68.7 GB tile. All 72 arms would have OOMed on a 6-node allocation.
    """
    stale, missing = [], []
    for combo in combos:
        name, text = render_arm(*combo)
        path = OUT / name / "training.args"
        if not path.exists():
            missing.append(name)
        elif path.read_text() != text:
            stale.append(name)
    extra = sorted(d.name for d in OUT.iterdir() if d.is_dir()) if OUT.exists() else []
    expected = {render_arm(*c)[0] for c in combos}
    extra = [e for e in extra if e not in expected]
    for label, names in (("stale", stale), ("missing", missing), ("not in grid", extra)):
        if names:
            print(f"  {len(names)} {label}: {', '.join(names[:6])}"
                  f"{' ...' if len(names) > 6 else ''}")
    if stale or missing or extra:
        print(f"\nregenerate: python {Path(__file__).name} --stage full --clean")
        return 1
    print(f"  {len(combos)} arms match {TEMPLATE.relative_to(REPO)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("core", "stage2", "full"), default="core")
    parser.add_argument("--base-lr", default="5e-5", help="stage2: winning learning rate")
    parser.add_argument("--base-scale", default="0.1", help="stage2: winning encoder scale")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean", action="store_true", help="remove previously generated arms")
    parser.add_argument("--check", action="store_true",
                        help="verify the arms on disk match the template; write nothing")
    cli = parser.parse_args()

    if cli.clean and OUT.exists() and not cli.dry_run:
        shutil.rmtree(OUT)

    if cli.stage == "core":
        combos = [(lr, es, "2", "128") for lr, es in itertools.product(LEARNING_RATES, ENCODER_SCALES)]
    elif cli.stage == "stage2":
        combos = [(cli.base_lr, cli.base_scale, ep, h)
                  for ep, h in itertools.product(EPOCHS, HEAD_SIZES)]
        # The baseline point is already measured by core; running it again wastes a slot.
        combos = [c for c in combos if not (c[2] == "2" and c[3] == "128")]
    else:
        combos = list(itertools.product(LEARNING_RATES, ENCODER_SCALES, EPOCHS, HEAD_SIZES))

    if cli.check:
        return check(combos)

    names = [write_arm(*combo, dry_run=cli.dry_run) for combo in combos]
    # ~11 min at 2 epochs / head 128; epochs and head width both add to that.
    minutes = sum(11 * (2 if c[2] == "4" else 1) * (1.3 if c[3] == "512" else 1.0)
                  for c in combos)
    print(f"stage={cli.stage}  arms={len(names)}  est. {minutes/60:.1f} node-hours serial")
    for name in names:
        print(f"  {name}")
    if not cli.dry_run:
        print(f"\nwritten under {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
