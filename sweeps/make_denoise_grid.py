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
DEFAULT_TEMPLATE = REPO / "configs" / "finetune-denoise-50m-ds" / "training.args"
OUT = REPO / "configs" / "sweep-denoise"
# What the arms on disk were built from: template path on the first line, stage on the
# second. Written at generation time and read back by --check, so the check compares
# against the right base AND the right arm set without the sweep launcher having to know
# or be told either. The launcher used to pass --stage full itself, which silently broke
# the moment the grid moved to full+batch: the check reported the 144 batch arms as "not
# in grid" and job 8840323 refused to start. The caller should not have to remember.
STAMP = OUT / ".template"
TEMPLATE = DEFAULT_TEMPLATE

LEARNING_RATES = ("1e-6", "1e-5", "2e-4")
ENCODER_SCALES = ("0", "0.1", "0.5", "1.0")
EPOCHS = ("2", "4")
HEAD_SIZES = ("128", "256", "512")

# Effective batch, as (per_device_train_batch_size, gradient_accumulation_steps) on the
# twelve tiles of one node: effective = per_device * 12 * accumulation.
#
# Accumulation alone can only go UP from the default 48, and it buys the larger batch by
# taking optimizer steps away -- a 2-epoch run drops from 3,633 steps to 1,211 at
# accumulation 3. Going DOWN needs a smaller per-device batch, which is where the extra
# steps are: per_device=1 gives an effective 12 and 14,533 steps. A batch axis worth
# sweeping therefore has to use both knobs, so it is expressed as the effective batch and
# the pair is derived.
# Matches RUN_PREFIX in the pbs launchers. Runs from before the DDP and eval fixes are
# not comparable to these, so the names must not collide in W&B.
RUN_PREFIX = "v2_dn50m-"

BATCHES = {
    "12":  ("1", "1"),   # 14,533 steps at 2 epochs -- 4x the updates
    "48":  ("4", "1"),   #  3,633 steps -- the current default
    "144": ("4", "3"),   #  1,211 steps -- accumulation, fewer and larger updates
}


def arm_name(lr: str, scale: str, epochs: str, head: str, batch: str = "48") -> str:
    """Readable and filesystem-safe, and it sorts sensibly in W&B.

    The batch suffix is omitted at the default so that adding the axis does not rename
    every existing arm and orphan its W&B history.
    """
    name = (f"lr{lr.replace('-', '')}_es{scale.replace('.', '')}"
            f"_ep{epochs}_h{head}")
    return name if batch == "48" else f"{name}_b{batch}"


def render_arm(lr: str, scale: str, epochs: str, head: str,
               batch: str = "48") -> tuple[str, str]:
    """The arm's name and exactly the bytes its training.args should contain."""
    name = arm_name(lr, scale, epochs, head, batch)
    per_device, accumulation = BATCHES[batch]
    overrides = {
        "--learning_rate": lr,
        "--encoder_lr_scale": scale,
        "--num_train_epochs": epochs,
        "--head_hidden_size": head,
        "--per_device_train_batch_size": per_device,
        "--gradient_accumulation_steps": accumulation,
        "--run_name": f"{RUN_PREFIX}{name}",
        "--output_dir": f"./runs/{RUN_PREFIX}{name}",
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


def arm_description(lr: str, scale: str, epochs: str, head: str, batch: str) -> str:
    """Why this arm exists, in words, so no arm in a 216-arm grid is unexplained.

    describe_run() derives a sentence from the settings at run time; this says what the
    arm is FOR, which the settings cannot. The template's own DESCRIPTION.md leads, so the
    shared purpose is stated once and the arm adds only what makes it different.
    """
    shared = (TEMPLATE.parent / "DESCRIPTION.md")
    lead = " ".join(shared.read_text().split()) + " " if shared.exists() else ""
    per_device, accumulation = BATCHES[batch]
    updates = 87200 * int(epochs) // int(batch)
    encoder = ("encoder frozen (control for whether fine-tuning the encoder helps at all)"
               if scale == "0" else f"encoder learning at {scale}x the head's rate")
    return (
        f"{lead}GRID ARM: lr={lr}, {encoder}, {epochs} epochs, head width {head}, "
        f"effective batch {batch} ({per_device} per tile x 12 tiles x {accumulation} "
        f"accumulation) giving about {updates:,} optimizer updates. The batch axis exists "
        f"to test whether more, smaller updates beat fewer, larger ones; batch and update "
        f"count move together, so it cannot separate the two."
    )


def write_arm(lr: str, scale: str, epochs: str, head: str, batch: str = "48",
              *, dry_run: bool = False) -> str:
    name, text = render_arm(lr, scale, epochs, head, batch)
    if not dry_run:
        directory = OUT / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "training.args").write_text(text)
        (directory / "DESCRIPTION.md").write_text(
            arm_description(lr, scale, epochs, head, batch) + "\n")
    return name


def check(combos: list[tuple[str, ...]], stage: str = "full+batch") -> int:
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
        if not path.exists() or not (OUT / name / "DESCRIPTION.md").exists():
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
        print(f"\nregenerate: python {Path(__file__).name} --stage {stage} --clean")
        return 1
    print(f"  {len(combos)} arms match {TEMPLATE.relative_to(REPO)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("core", "stage2", "full", "full+batch"),
                        default="core")
    parser.add_argument("--base-lr", default="5e-5", help="stage2: winning learning rate")
    parser.add_argument("--base-scale", default="0.1", help="stage2: winning encoder scale")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean", action="store_true", help="remove previously generated arms")
    parser.add_argument("--check", action="store_true",
                        help="verify the arms on disk match the template; write nothing")
    parser.add_argument("--template", default=None,
                        help=f"base args file (default {DEFAULT_TEMPLATE.relative_to(REPO)}); "
                             "--check reads the recorded one instead")
    cli = parser.parse_args()

    global TEMPLATE
    if cli.check and STAMP.exists():
        recorded = STAMP.read_text().split()
        TEMPLATE = REPO / recorded[0]
        if len(recorded) > 1:
            cli.stage = recorded[1]
    elif cli.template:
        TEMPLATE = Path(cli.template)
        if not TEMPLATE.is_absolute():
            TEMPLATE = REPO / TEMPLATE
    if not TEMPLATE.exists():
        raise SystemExit(f"template not found: {TEMPLATE}")

    if cli.clean and OUT.exists() and not cli.dry_run:
        shutil.rmtree(OUT)

    if cli.stage == "core":
        combos = [(lr, es, "2", "128", "48")
                  for lr, es in itertools.product(LEARNING_RATES, ENCODER_SCALES)]
    elif cli.stage == "stage2":
        combos = [(cli.base_lr, cli.base_scale, ep, h, b)
                  for ep, h, b in itertools.product(EPOCHS, HEAD_SIZES, BATCHES)]
        # The baseline point is already measured by core; running it again wastes a slot.
        combos = [c for c in combos
                  if not (c[2] == "2" and c[3] == "128" and c[4] == "48")]
    elif cli.stage == "full":
        combos = [(*c, "48") for c in
                  itertools.product(LEARNING_RATES, ENCODER_SCALES, EPOCHS, HEAD_SIZES)]
    else:
        combos = list(itertools.product(LEARNING_RATES, ENCODER_SCALES, EPOCHS,
                                        HEAD_SIZES, BATCHES))

    if cli.check:
        return check(combos, cli.stage)

    names = [write_arm(*combo, dry_run=cli.dry_run) for combo in combos]
    if not cli.dry_run:
        STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\n{cli.stage}\n")
    # ~11 min at 2 epochs / head 128; epochs and head width both add to that.
    # 25 min for a 4-epoch arm on twelve tiles with DeepSpeed, measured on job 8840264
    # (235.4 samples/s, 87,200 samples per epoch); a 512-wide head adds ~30%, and an
    # effective batch of 12 costs ~40% more for the same samples through smaller kernels.
    minutes = sum(12.5 * (2 if c[2] == "4" else 1) * (1.3 if c[3] == "512" else 1.0)
                  * (1.4 if c[4] == "12" else 1.0) for c in combos)
    print(f"stage={cli.stage}  arms={len(names)}  est. {minutes/60:.1f} node-hours serial")
    for name in names:
        print(f"  {name}")
    if not cli.dry_run:
        print(f"\nwritten under {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
