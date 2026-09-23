"""The contrastive sweep that closes both open axes, at every scale.

    python sweeps/make_conbig.py --clean

WHAT CONTRASTIVE IS TESTING: whether fine-tuning a pretrained encoder produces
embeddings that retrieve replicate spectra, and whether that improves with model scale
and with pretraining amount. Everything else is instrumentation for those two claims.

TWO AXES ARE STILL OPEN AND THEY INTERACT, which is why they are swept together rather
than one after the other.

  TEMPERATURE has been at the edge of the grid every single time it was looked at.
  0.2 -> 0.07 gained, 0.07 -> 0.03 gained 0.05 to 0.14 in all four cells tested. 0.03 is
  again the LOWEST value tried, so the optimum is still unlocated. Extended down to 0.01.

  P/K HAS NEVER BEEN VARIED. Every contrastive run in this project used
  groups_per_batch 2 x replicates 2 -- four spectra, about two negatives -- against a
  loss whose entire signal is negatives. The justification was "more negatives hurt",
  which is retracted: it was decided on the separation ratio (which does not predict
  retrieval) at a configuration that does not train, and rescoring it on the task gives
  64 vs 4 negatives at p=0.84. The axis is unmeasured, not settled.

WHY THEY INTERACT. Temperature sets how sharply the softmax concentrates on the hardest
negative; P/K sets how many negatives there are to concentrate on. The optimum
temperature at 2 negatives has no reason to be the optimum at 64, so sweeping one at a
fixed value of the other would find a local answer and mislead.

ALL FOUR SCALES, because the claim at stake is a scaling claim and the current evidence
for "contrastive gets worse with scale" is thin: rho -0.80 across four scales was
measured at t0.07 on the superseded configuration, and the only fresh evidence at t0.03
is 50m against 200m. 100m and 400m have never been run at a current configuration.

checkpoint-220000 for every scale: it is the ONLY canonical rung that exists at all
four, because 400m pretraining is 47% done and stops just past it.

HELD FIXED, and why each is defensible rather than merely convenient:
  kl_weight 10   bracketed on both sides -- 0 is far worse, 100 is far worse
  lr 1e-4        interior to {2e-5, 1e-4, 5e-4}, and the whole axis is worth less than
                 either open axis (0.207 -> 0.264 against 0.136 -> 0.249 for KL)
  gradcache_chunk 4   fixed so peak memory is constant across P/K and any difference is
                 the negatives rather than the memory regime. Required above P*K=4:
                 DeltaMZBias is 32 GiB at batch 64 against a 64 GiB tile. GradCache is
                 verified -- 19 unit tests, plus an end-to-end A/B at the live
                 configuration where on vs off is t=+0.46, p=0.65.

4 scales x 3 temperatures x 3 P/K x 3 seeds = 108 arms. Answers PLAN.md C1 (and C2 at
220k).
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-conbig"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_conbig-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

SCALES = {f"s{s:0>4}": (s, f"{FROZEN}/msdelta-{s}-production-01-checkpoint-220000")
          for s in ("50m", "100m", "200m", "400m")}
PK = {"pk04": ("2", "2"), "pk16": ("8", "2"), "pk64": ("16", "4")}
LEARNING_RATES = {"lr1e4": "1e-4"}
KL_WEIGHTS = {"kl10": "10"}
TEMPERATURES = {"t001": "0.01", "t002": "0.02", "t003": "0.03"}
SEEDS = ("0", "1", "2")
EPOCHS = "3"


def arm_name(scale, lr, kl, t, pk, seed):
    return f"{scale}_{t}_{pk}_seed{seed}"


def render_arm(scale, lr, kl, t, pk, seed) -> tuple[str, str]:
    name = arm_name(scale, lr, kl, t, pk, seed)
    overrides = {
        # 12 arms/node at TILES_PER_ARM=1, and a 400m optimizer checkpoint is 7.4 GB:
        # save_steps 200 put ~89 GB on Lustre at once and killed arms in 8853557,
        # 8853558 and 8853703 with "enforce fail ... unexpected pos" from torch.save.
        # Only final/ is ever consumed, so optimizer state is pure write amplification.
        "--save_only_model": "true",
        "--save_steps": "700",
        "--save_total_limit": "1",
        # This grid runs in 1h debug-scaling slots and is RESUBMITTED until done, so an
        # arm may be killed mid-run. Resuming from a save_only_model checkpoint restarts
        # Adam and the LR schedule mid-run -- a different trajectory from an
        # uninterrupted arm, silently. With no intermediate checkpoints an interrupted
        # arm restarts from step 0 with the same seed instead, which is equivalent to
        # never having been interrupted. The only cost is the killed arm's compute.
        # final/ is written explicitly by main() and does not depend on this.
        "--save_strategy": "no",
        "--pretrained_path": SCALES[scale][1],
        "--groups_per_batch": PK[pk][0],
        "--replicates": PK[pk][1],
        "--gradcache_chunk": "4",
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


def description(scale, lr, kl, t, pk, seed) -> str:
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
                                    TEMPERATURES, PK, SEEDS))

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
