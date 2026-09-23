"""Does GradCache reproduce the non-GradCache result at a configuration that trains?

    python sweeps/make_gradcache_confirm.py --clean

GradCache is the ONLY route past a batch of four: DeltaMZBias is
O(batch * peaks^2 * 2 * n_freqs), which is 32 GiB at batch 64 against a 64 GiB tile. So
every P/K sweep rests on it. Its gradient is exact and 19 unit tests now say so --
against a direct full-batch backward at relative L2 < 1e-4, across ragged chunk sizes,
P*K up to 32, temperature 0.07, padding, singleton groups, and chunk >= batch.

BUT NO UNIT TEST CAN ANSWER THE QUESTION THAT MATTERS. GradCache has only ever run
end-to-end at lr5e-4/KL10/t0.2 and at lr2e-5/KL0/t0.2, and the second of those is the
configuration since shown to be statistically indistinguishable from not training at
all. It has never produced a task number at a configuration that trains. Without this
run, a null result from a P/K sweep cannot be told apart from GradCache being wired up
wrong.

THE DESIGN IS AN A/B AND NOTHING ELSE. Identical cell, identical hyperparameters,
identical batch shape, six seeds either side, one variable:

    gradcache_chunk 0    off -- the path every contrastive number in this project used
    gradcache_chunk 2    on, and actually chunking, since the batch is 4

Chunk 2 rather than 4 deliberately: at chunk >= batch GradCache degenerates to the
direct path and the comparison proves nothing.

THE REFERENCE ALREADY EXISTS, which is why this is 12 arms and not 24. The checkpoint
ladder (8851663) ran this exact cell -- 50m at checkpoint-330000, lr1e-4/KL10/t0.07,
P=2 K=2 -- at six seeds and got MAP@R 0.3318 +/- 0.0133, MAP@100 0.4122 +/- 0.0152. The
chunk-0 arms here should land on that too; if they do not, the disagreement is something
other than GradCache and has to be chased before the chunk-2 comparison means anything.

PASS CONDITION: chunk 2 agrees with chunk 0 within seed noise. Then the P/K sweep can be
built on it. FAIL: they differ, and the sweep is blocked until the cause is found.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-gradcache-confirm"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_gcconf-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

SCALES = {"s050m_ck330k": ("50m@330k", f"{FROZEN}/msdelta-50m-production-01-checkpoint-330000")}
# The one variable. 0 is the path every contrastive number so far has used.
GRADCACHE = {"gcoff": "0", "gcon": "2"}
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
        "--gradcache_chunk": GRADCACHE[gc],
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
                                    TEMPERATURES, GRADCACHE, SEEDS))

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
