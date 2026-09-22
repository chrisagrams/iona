"""Denoise and contrastive across pretraining checkpoints, at every scale.

    python sweeps/make_checkpoint_grid.py --task denoise --clean
    python sweeps/make_checkpoint_grid.py --task contrastive --clean

THE QUESTION. Every fine-tuned result in this project comes from exactly one pretrained
checkpoint per scale, and each was simply the latest available when it was frozen --
133k to 193k steps, which is 25 to 36% of the 540,423 all four scales target. The scale
conclusions therefore rest on models a quarter to a third pretrained, and they are not
even matched to each other on that axis.

The canonical ladder is six equally spaced points over [10,000 .. 540,423], the same
at every scale because a step is the same data everywhere (epoch 0.555 at step 100,000
for both 50m and 400m):

    10,000   120,000   220,000   330,000   430,000   540,423

This grid runs the MIDDLE TWO first -- 220,000 and 330,000 -- which bracket where the
existing results sit and are the cheapest place to see whether the curve moves at all.
400m has no 330,000 yet, so that cell is skipped rather than substituted.

HYPERPARAMETERS ARE NOT SWEPT. Denoise uses lr 2e-4 / encoder_lr_scale 0.5 / 4 epochs /
head 512 / effective batch 12, the point all four HP grids independently chose.
Contrastive uses lr 2e-5 / KL 0 / temperature 0.2, which the 96-arm grid selected and
which transferred between 50m and 100m at Spearman 0.95. BOTH WERE CHOSEN AT
CHECKPOINT 1. Whether they still hold at 220k and 330k is untested and is the obvious
thing to check if these results look strange -- it is on the TODO, not in this grid.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

# (scale, checkpoint) -- the middle two rungs, minus what does not exist yet.
CELLS = [(s, c) for s in ("50m", "100m", "200m", "400m") for c in ("220000", "330000")
         if not (s == "400m" and c == "330000")]

TASKS = {
    "denoise": dict(
        out="sweep-ckpt-denoise", prefix="v2_ckdn-",
        module="msdelta.finetune_denoise", seeds=("1", "2", "3"),
        template=lambda s: REPO / "configs" / f"finetune-denoise-{s}-ds" / "training.args",
        overrides={"--learning_rate": "2e-4", "--encoder_lr_scale": "0.5",
                   "--num_train_epochs": "4", "--head_hidden_size": "512",
                   "--per_device_train_batch_size": "1",
                   "--gradient_accumulation_steps": "1"}),
    "contrastive": dict(
        out="sweep-ckpt-contrastive", prefix="v2_ckcon-",
        module="msdelta.finetune_contrastive", seeds=("0", "1", "2", "3", "4", "5"),
        template=lambda s: REPO / "configs" / "finetune-contrastive-50m" / "training.args",
        overrides={"--learning_rate": "2e-5", "--kl_weight": "0",
                   "--temperature": "0.2", "--num_train_epochs": "3",
                   "--groups_per_batch": "2", "--replicates": "2",
                   "--split_seed": "0"}),
}


def arm_name(scale, ckpt, seed):
    return f"{scale}_ck{int(ckpt)//1000:03d}k_seed{seed}"


def render(task, scale, ckpt, seed):
    spec = TASKS[task]
    name = arm_name(scale, ckpt, seed)
    over = dict(spec["overrides"])
    over |= {
        "--pretrained_path": f"{FROZEN}/msdelta-{scale}-production-01-checkpoint-{ckpt}",
        "--seed": seed,
        "--run_name": f"{spec['prefix']}{name}",
        "--output_dir": f"./runs/{spec['prefix']}{name}",
    }
    lines, seen = [], set()
    for flag, value in zip(*[iter(spec["template"](scale).read_text().split())] * 2):
        lines.append(f"{flag} {over.get(flag, value)}")
        seen.add(flag)
    for flag, value in over.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")
    return name, "\n".join(lines) + "\n"


def describe(task, scale, ckpt, seed):
    pct = int(ckpt) / 540423 * 100
    return (f"CHECKPOINT ARM ({task}): {scale} from pretrained checkpoint {int(ckpt):,} "
            f"-- {pct:.0f}% of the 540,423 steps every scale targets -- seed {seed}. "
            f"Hyperparameters are held at the configuration chosen at checkpoint 1 and "
            f"are NOT swept here; whether that choice still holds this far into "
            f"pretraining is untested.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clean", action="store_true")
    cli = ap.parse_args()
    spec = TASKS[cli.task]
    out = REPO / "configs" / spec["out"]
    combos = [(s, c, sd) for (s, c) in CELLS for sd in spec["seeds"]]

    missing = [f"{s}/{c}" for s, c in CELLS
               if not Path(FROZEN, f"msdelta-{s}-production-01-checkpoint-{c}",
                           "model.safetensors").exists()]
    if missing and not cli.check:
        raise SystemExit("not frozen yet: " + ", ".join(missing))

    if cli.check:
        stale = [n for n, t in (render(cli.task, *c) for c in combos)
                 if not (out / n / "training.args").exists()
                 or (out / n / "training.args").read_text() != t
                 or not (out / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(combos)} arms match their templates")
        return 0

    if cli.clean and out.exists():
        shutil.rmtree(out)
    for c in combos:
        name, text = render(cli.task, *c)
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "training.args").write_text(text)
        (out / name / "DESCRIPTION.md").write_text(describe(cli.task, *c))
    (out / ".template").write_text(f"per-scale templates\nckpt-{cli.task}\n")
    print(f"arms={len(combos)} under configs/{spec['out']}/  "
          f"({len(CELLS)} cells x {len(spec['seeds'])} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
