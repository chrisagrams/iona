"""Push past the edge of the contrastive hyperparameter grid, on canonical checkpoints.

    python sweeps/make_conhp_edge.py --clean

THE RANKING QUESTION IS ALREADY ANSWERED and this does not re-ask it. Job 8849088 swept
all twelve configurations at 50m checkpoint-540423 against the same twelve at 133233 and
got Spearman +0.90; across four scale x checkpoint cells the mean pairwise rank
correlation is +0.80 and lr1e-4/KL10/t0.07 is first in every one. Re-running those twelve
axes at 220k and 330k would buy a thirteenth confirmation.

WHAT IS ACTUALLY OPEN is that the winner sits at the BOUNDARY of both axes that matter.
Temperature was swept over {0.07, 0.2} and won at 0.07, the lowest value tried. KL was
swept over {0, 10} and won at 10, the highest. By the rule this project already recorded
(FT17), an edge result means the optimum is UNLOCATED, and the marginal effects say these
are the two axes worth the compute: KL 0 -> 10 moves MAP@100 from 0.136 to 0.249 and
t 0.2 -> 0.07 moves it from 0.147 to 0.239, while the whole learning-rate axis is worth
0.207 -> 0.264. So lr is held at 1e-4 and the two live axes are extended OUTWARD:

    temperature  0.03  0.07        0.07 was the edge; 0.03 looks past it
    kl_weight      10   100        10 was the edge; 100 is already used elsewhere in
                                   this repo, so it is a known-runnable value

TWO SCALES, BECAUSE THE SCALE RESULT IS THE THING TO EXPLAIN. Contrastive gets WORSE with
scale -- rho -0.80 against model size on MAP@100, MAP@R and R-Precision alike, 50m best
and 200m worst. If that is a hyperparameter artefact rather than a property of the
encoders, the optimum should sit at a different place for 200m than for 50m, and this
grid would show it. 50m is the best cell and 200m the worst, so they bracket the effect.

TWO CHECKPOINTS, both canonical rungs (220000, 330000), so the answer is usable on the
ladder rather than on the original off-ladder checkpoints.

4 configurations x 2 scales x 2 checkpoints x 3 seeds = 48 arms. Seeds start at 0: these
are new cells, not a continuation of an existing job.

P/K IS NOT IN THIS GRID and is the other open axis. It cannot be swept without GradCache
-- DeltaMZBias is O(batch * peaks^2 * 2 * n_freqs), 32 GiB at batch 64 against a 64 GiB
tile -- and GradCache has never produced a task number at a configuration that trains.
That needs its own confirmation run first.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-conhp-edge"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_conhpe-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

SCALES = {"s050m_ck220k": ("50m@220k", f"{FROZEN}/msdelta-50m-production-01-checkpoint-220000"),
          "s050m_ck330k": ("50m@330k", f"{FROZEN}/msdelta-50m-production-01-checkpoint-330000"),
          "s200m_ck220k": ("200m@220k", f"{FROZEN}/msdelta-200m-production-01-checkpoint-220000"),
          "s200m_ck330k": ("200m@330k", f"{FROZEN}/msdelta-200m-production-01-checkpoint-330000")}
LEARNING_RATES = {"lr1e4": "1e-4"}
KL_WEIGHTS = {"kl10": "10", "kl100": "100"}
TEMPERATURES = {"t003": "0.03", "t007": "0.07"}
SEEDS = ("0", "1", "2")
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
    combos = list(itertools.product(SCALES, LEARNING_RATES, KL_WEIGHTS,
                                    TEMPERATURES, SEEDS))

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
