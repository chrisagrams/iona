"""Train a learned mixture over encoder depths, at three encoder learning rates.

    python sweeps/make_layermix_grid.py --clean
    python sweeps/make_layermix_grid.py --check
    python sweeps/make_layermix_grid.py --clean --seeds 4   # 12 arms, fills a node

WHAT IS BEING TESTED. Every embedding measured in this project is read off ONE layer,
nearly always the last. Job 8841973 showed that is the wrong layer at every scale: the
separation ratio peaks mid-stack and sags at the output (50m 1.53 at block 8 against
1.43 at the output; 200m 1.44 at block 6 against 1.35). Choosing the best single layer
is a discrete search over a curve whose shape moves with model size. `pooling=layer_mix`
learns the mixture instead -- one trainable scalar per depth, softmaxed into a convex
combination, sequence-mean pooled to d_model -- and trains it under the same supervised
contrastive loss.

THE THREE ARMS. The question a learned readout raises immediately is whether any gain
came from the readout at all, or just from letting the encoder move. Only holding the
readout fixed and varying the encoder's rate answers it:

  frozen  encoder_lr_scale 0    ONLY the mixture trains. Whatever this reaches is what
                                depth selection alone is worth on a fixed encoder, and
                                it is directly comparable to the frozen probe's 1.53.
  els03   encoder_lr_scale 0.3  Mixture at full rate, encoder nudged.
  els10   encoder_lr_scale 1.0  Everything at one rate, the ordinary fine-tune.

Read them in that order. If frozen already captures most of the gain, the information
was always there and we were reading it wrong. If only els10 moves, depth was never the
problem and this line is finished.

TWO SETTINGS DIFFER FROM THE TEMPLATE, both forced rather than chosen:

  gradient_checkpointing false. Reading intermediate states needs forward hooks, and
  under checkpointing each block runs twice -- once under no_grad to find the
  boundaries, once to recompute -- so a hook fires twice and the first tensor is
  detached. encoder_layer_states refuses the combination rather than depending on
  recompute order. Batches here are groups_per_batch x replicates = 4 rows, so the
  activation memory checkpointing was saving is small.

  kl_weight 0 on the frozen arm. KL regularises the encoder against forgetting its
  pretraining behaviour; a frozen encoder cannot forget. Left at 100 it would compute
  a term whose gradient is identically zero, and whose VALUE is not zero only because
  the trainable copy runs with dropout and the reference does not -- pure noise in the
  log, at the cost of a second forward pass every step.

Everything else is held at the configuration that produced the best contrastive result
so far (ratio 6.94): batch 4 as 2 groups x 2 replicates, temperature 0.07, kl 100.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
OUT = REPO / "configs" / "sweep-layermix"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_lmix-"

# name -> encoder learning rate as a multiple of the mixture's
SCALES = {"frozen": "0.0", "els03": "0.3", "els10": "1.0"}
LAYER_MIX_LR = "5e-2"


def arm_name(scale: str, seed: str) -> str:
    return scale if seed == "0" else f"{scale}_s{seed}"


def render_arm(scale: str, seed: str) -> tuple[str, str]:
    name = arm_name(scale, seed)
    overrides = {
        "--pooling": "layer_mix",
        "--layer_mix_norm": "true",
        # Not the encoder's rate, and much larger than looks reasonable, because this
        # run is SHORT: the replicate corpus has 60 training groups, so P=2 gives 30
        # batches an epoch and 3 epochs is ~90 optimizer steps. warmup_steps is 100,
        # more than the whole run, so every rate here is still ramping when it ends
        # (mean multiplier ~0.45). A mixture logit therefore travels lr * 90 * 0.45:
        # 0.04 at 1e-3 and 0.41 at 1e-2, both indistinguishable from uniform. At 5e-2
        # it travels ~2.0, which puts the top weight near 0.43 against a uniform 0.091
        # -- concentrated enough to read, not so much that it collapses to one layer.
        "--layer_mix_lr": LAYER_MIX_LR,
        "--encoder_lr_scale": SCALES[scale],
        # See the module docstring: hooks cannot read intermediates under checkpointing.
        "--gradient_checkpointing": "false",
        "--seed": seed,
        "--run_name": f"{RUN_PREFIX}{name}",
        "--output_dir": f"./runs/{RUN_PREFIX}{name}",
    }
    if SCALES[scale] == "0.0":
        overrides["--kl_weight"] = "0.0"
    lines, seen = [], set()
    tokens = TEMPLATE.read_text().split()
    for flag, value in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{flag} {overrides.get(flag, value)}")
        seen.add(flag)
    for flag, value in overrides.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")
    return name, "\n".join(lines) + "\n"


def description(scale: str, seed: str) -> str:
    shared = TEMPLATE.parent / "DESCRIPTION.md"
    lead = " ".join(shared.read_text().split()) + " " if shared.exists() else ""
    if SCALES[scale] == "0.0":
        what = ("the encoder is FROZEN, so the trained layer mixture is the only thing "
                "that moves. This is the arm that says what depth selection alone is "
                "worth, directly against the frozen probe's best single layer (1.53 at "
                "block 8 of the 50m). KL is 0 here because a frozen encoder cannot "
                "forget, and the term's only non-zero content would be dropout noise")
    else:
        what = (f"the encoder trains at {SCALES[scale]}x the mixture's learning rate, so "
                f"the readout and the representation move together")
    return (f"{lead}LAYER-MIX ARM ({scale}): the spectrum embedding is a trained convex "
            f"mixture over all encoder depths -- one softmaxed scalar per layer "
            f"including the pre-block embedding, sequence-mean pooled to d_model -- "
            f"rather than a fixed readout of the final layer. Here {what}. Motivated by "
            f"job 8841973, which found the separation ratio peaks mid-stack and sags at "
            f"the output at every model scale, so every embedding measured so far was "
            f"read from the wrong layer. Gradient checkpointing is off because reading "
            f"intermediate states uses forward hooks, which fire twice per block under "
            f"checkpointing with the first output detached. Seed {seed}. The learned "
            f"mixture is logged per step as mix/layerNN, mix/argmax and mix/entropy: a "
            f"mixture that stays uniform means depth did not matter.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--seeds", type=int, default=1,
                        help="repetitions per arm. 1 gives the three arms asked for; 4 "
                             "gives 12 and fills a node's tiles, which is free in "
                             "wall-clock and is the only defence against reading a "
                             "difference that is inside seed noise.")
    cli = parser.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"no template at {TEMPLATE}")
    seeds = [str(i) for i in range(cli.seeds)]
    combos = list(itertools.product(SCALES, seeds))

    if cli.check:
        stale = [n for n, text in (render_arm(*c) for c in combos)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        extra = [d.name for d in OUT.glob("*") if d.is_dir()
                 and d.name not in {arm_name(*c) for c in combos}]
        if stale or extra:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}"
                  + (f"; {len(extra)} unexpected: {', '.join(extra[:4])}" if extra else ""))
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
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\nlayermix\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/")
    for combo in combos:
        print(f"  {arm_name(*combo)}  encoder_lr_scale={SCALES[combo[0]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
