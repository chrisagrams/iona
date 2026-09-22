"""Resolve the contrastive HP winner at 200m and 400m, where four seeds could not.

    python sweeps/make_conhp_seeds.py --clean

lr1e-4/KL10/t0.07 is top-ranked in all four scale x checkpoint cells, but its margin
over the runner-up is significant only at 50m. At the larger scales four seeds leave it
first and unresolved:

    200m   0.3409 vs 0.2596 (lr5e-4/KL10/t0.07)   gap 0.0813  sd 0.0851  t=1.45 p=0.20
    400m   0.3541 vs 0.3113 (lr2e-5/KL10/t0.07)   gap 0.0429  sd 0.0340  t=2.23 p=0.067

Powering those to p<0.05 needs n~10 at 200m and n~6 at 400m. This grid takes BOTH to
n=12 by adding seeds 4-11 to the winner and its own runner-up at each scale -- the
runner-up differs by scale, so the pairs are not the same four arms.

ONLY TWO CONFIGURATIONS PER SCALE, not the full twelve. The ranking question is already
answered at mean pairwise Spearman +0.80; what is open is one pairwise margin per scale,
and that is all this grid buys. Re-running twelve configurations at twelve seeds would
cost 288 arms to answer a question worth 32.

Seeds start at 4 so these compose with the existing four from job 8848820 rather than
repeating them. split_seed stays pinned at 0 so every arm is scored on the same
held-out rows.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-conhp-seeds"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_conhps-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

SCALES = {"s200m": ("200m", f"{FROZEN}/msdelta-200m-production-01-checkpoint-192799"),
          "s400m": ("400m", f"{FROZEN}/msdelta-400m-production-01-checkpoint-181381")}
LEARNING_RATES = {"lr2e5": "2e-5", "lr1e4": "1e-4", "lr5e4": "5e-4"}
KL_WEIGHTS = {"kl0": "0", "kl10": "10"}
TEMPERATURES = {"t007": "0.07", "t02": "0.2"}
SEEDS = ("4", "5", "6", "7", "8", "9", "10", "11")
EPOCHS = "3"


def arm_name(scale, lr, kl, t, seed):
    return f"{scale}_{lr}_{kl}_{t}_seed{seed}"


def render_arm(scale, lr, kl, t, seed) -> tuple[str, str]:
    name = arm_name(scale, lr, kl, t, seed)
    overrides = {
        # 12 arms/node at TILES_PER_ARM=1, and a 400m optimizer checkpoint is 7.4 GB:
        # save_steps 200 put ~89 GB on Lustre at once and killed arms in 8853557,
        # 8853558 and 8853703 with "enforce fail ... unexpected pos" from torch.save.
        # Only final/ is ever consumed, so optimizer state is pure write amplification.
        "--save_only_model": "true",
        "--save_steps": "700",
        "--save_total_limit": "1",
        "--pretrained_path": SCALES[scale][1],
        "--learning_rate": LEARNING_RATES[lr],
        "--kl_weight": KL_WEIGHTS[kl],
        "--temperature": TEMPERATURES[t],
        "--num_train_epochs": EPOCHS,
        "--seed": seed,
        # Pinned: the split must not move with the training seed, or arms are scored on
        # different held-out data and cannot be compared.
        "--split_seed": "0",
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


def description(scale, lr, kl, t, seed) -> str:
    label = SCALES[scale][0]
    return (f"CONTRASTIVE HP ARM: {label}, lr {LEARNING_RATES[lr]}, KL "
            f"{KL_WEIGHTS[kl]}, temperature {TEMPERATURES[t]}, seed {seed} of "
            f"{len(SEEDS)}, PK sampling at P=2 K=2 for {EPOCHS} epochs. Asks whether "
            f"the contrastive hyperparameter choice transfers across scale the way the "
            f"denoise one did, where the same point won all four grids. 50m is re-run "
            f"rather than compared against its existing grid, which predates the FT14 "
            f"sampler fix (every epoch replayed the same 15.4% of the corpus) and is "
            f"n=1 per cell against a metric with seed sd 0.43-0.63.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clean", action="store_true")
    cli = ap.parse_args()
    for key, (label, ckpt) in SCALES.items():
        if not Path(ckpt, "model.safetensors").exists():
            raise SystemExit(f"{label} not frozen at {ckpt}")
    # The runner-up differs by scale, so this is not a product over a shared set.
    PAIRS = {"s200m": [("lr1e4", "kl10", "t007"), ("lr5e4", "kl10", "t007")],
             "s400m": [("lr1e4", "kl10", "t007"), ("lr2e5", "kl10", "t007")]}
    combos = [(s, lr, kl, t, seed)
              for s, cfgs in PAIRS.items() for (lr, kl, t) in cfgs for seed in SEEDS]

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
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\ncontrastive-hp\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/  "
          f"({len(PAIRS)} scales x 2 configs x {len(SEEDS)} seeds, "
          f"seeds {SEEDS[0]}-{SEEDS[-1]} continuing job 8848820)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
