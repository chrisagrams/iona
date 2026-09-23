"""How much does best-model selection buy contrastive? Same cells, eval turned on.

    python sweeps/make_conhp_eval.py --clean

Every contrastive run in this project trained with --eval_strategy no. Nothing was
measured during training, nothing was selected, and `final/` is whatever the last step
produced. Three facts say that is not a safe default:

  the contrastive objective is SOLVED early -- on 50m@330k it falls below 10% of chance
  (ln 4 = 1.386) by epoch 0.18 of 3.0, so 94% of training optimises a solved task;

  `final/` is not even a loss minimum -- the last logged step of a real run sat at
  0.142 while epoch 2.90 had reached 0.0016, because a 4-spectrum batch makes every
  logged value one noisy draw;

  longer training measurably HURT in the one place it was tried -- the epochs ladder
  took the separation ratio from 6.94 at three epochs down to 4.82 at ten.

So this re-runs two cells with evaluation on and selection by the TASK, and compares
against the identical cells from job 8851663, which ran the same hyperparameters at six
seeds with selection off:

    50m@330k    MAP@R 0.3318 +/- 0.0133    the best cell in the ladder
    200m@330k   MAP@R 0.2281 +/- 0.0075    the worst, and the one that DECLINED with
                                           more pretraining -- if overfitting explains
                                           that, this is where selection should help most

WHAT CHANGES AND NOTHING ELSE: eval_strategy steps, load_best_model_at_end on
eval_retrieval/MAP@R, and eval_retrieval_rows set so ContrastiveTrainer.evaluate()
actually scores the task. Hyperparameters, checkpoints, batch shape and seeds are the
ladder's.

READ IT AS A PAIRED COMPARISON, not a new measurement: the reference is six seeds at the
same cells, so a gain smaller than the ~0.013 seed sem there is not a gain.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-conhp-eval"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_coneval-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

SCALES = {"s050m_ck330k": ("50m@330k", f"{FROZEN}/msdelta-50m-production-01-checkpoint-330000"),
          "s200m_ck330k": ("200m@330k", f"{FROZEN}/msdelta-200m-production-01-checkpoint-330000")}
# The single change under test.
SELECT = {"besteval": "on"}
LEARNING_RATES = {"lr1e4": "1e-4"}
KL_WEIGHTS = {"kl10": "10"}
TEMPERATURES = {"t007": "0.07"}
SEEDS = ("0", "1", "2", "3", "4", "5")
EPOCHS = "3"


def arm_name(scale, lr, kl, t, gc, seed):
    return f"{scale}_{gc}_seed{seed}"


def render_arm(scale, lr, kl, t, gc, seed) -> tuple[str, str]:
    name = arm_name(scale, lr, kl, t, gc, seed)
    overrides = {
        # 12 arms/node at TILES_PER_ARM=1, and a 400m optimizer checkpoint is 7.4 GB:
        # save_steps 200 put ~89 GB on Lustre at once and killed arms in 8853557,
        # 8853558 and 8853703 with "enforce fail ... unexpected pos" from torch.save.
        # Only final/ is ever consumed, so optimizer state is pure write amplification.
        "--save_only_model": "true",
        "--save_steps": "700",
        "--save_total_limit": "1",
        "--pretrained_path": SCALES[scale][1],
        # Evaluate on the TASK and keep the best encoder by it. eval_steps 200 gives
        # ~7 evaluations over 3 epochs; 800 rows keeps each one to a single forward
        # pass rather than a second training cost.
        "--eval_strategy": "steps",
        "--eval_steps": "200",
        "--eval_retrieval_rows": "800",
        "--load_best_model_at_end": "true",
        "--metric_for_best_model": "eval_retrieval/MAP@R",
        "--greater_is_better": "true",
        "--save_only_model": "false",   # load_best_model_at_end needs real checkpoints
        "--save_steps": "200",
        "--save_total_limit": "2",
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


def description(scale, lr, kl, t, gc, seed) -> str:
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
    combos = list(itertools.product(SCALES, LEARNING_RATES, KL_WEIGHTS,
                                    TEMPERATURES, SELECT, SEEDS))

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
          f"({len(SCALES)} scale x checkpoint cells x {len(KL_WEIGHTS)} kl x "
          f"{len(TEMPERATURES)} temp x {len(SEEDS)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
