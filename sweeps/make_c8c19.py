"""C8 x C19: which contrastive recipe, trained on ms-contrastive-100k alone? PLAN.md C8, C19.

    python sweeps/make_c8c19.py            # write configs/sweep-c8c19
    python sweeps/make_c8c19.py --check    # verify the arms match this generator

A 2 x 2 factorial, 3 seeds each (12 arms = one node, one tile per arm):

    loss      supcon   SupCon softmax over the batch, temperature 0.002 (every run so far)
              sigmoid  C8: SigLIP-style independent per-pair sigmoid, learnable scale/bias
                       (init 10 / -10)
    batches   random   peptide groups drawn in random order (every run so far)
              mass     C19: groups that are neighbours in neutral mass (+-1 Da jitter per
                       epoch), so in-batch negatives are same-mass competitors (GLEAMS)

Everything else is the Iona stage-2 recipe (P85 x K3 per batch, KL 10, lr 1e-4 cosine,
experimental spectra only, replicate-corpus peptides excluded), but trained from the PRETRAINED
50M encoder at its final checkpoint (540,423) on ms-contrastive-100k ALONE -- the single-dataset
recipe these ablations are meant to fix -- for 3 epochs, the encoder saved every half epoch
(531 steps) so the same runs give the epoch curve. GradCache chunks are length-sorted and
trimmed (gradcache_trim_padding; same gradient, tested) and gradient checkpointing is
off: ~4x faster per step than the old settings (pbs/diag/gradcache_bench.py).
"""

from __future__ import annotations

import argparse
import shutil
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-c8c19"
TEMPLATE = REPO / "configs" / "sweep-con100k-best" / "cont050m_ep01_seed1" / "training.args"
PRETRAINED = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-50m-production-01-checkpoint-540423"
RUN_PREFIX = "v2_c8c19-"
SEEDS = ("0", "1", "2")
STEPS_PER_EPOCH = 1062          # ms-contrastive-100k train, P85 x K3 (sweep-con100k-best logs)
LOSSES = {"supcon": {}, "sigmoid": {"--loss": "sigmoid", "--sigmoid_init_scale": "10.0",
                                    "--sigmoid_init_bias": "-10.0"}}
# random is explicit: same_mass_batches defaults to true since 2026-09-27
BATCHES = {"random": {"--same_mass_batches": "false"},
           "mass": {"--same_mass_batches": "true", "--mass_jitter": "1.0"}}
COMMON = {"--pretrained_path": PRETRAINED, "--num_train_epochs": "3",
          "--save_steps": str(STEPS_PER_EPOCH // 2), "--save_total_limit": "10",
          "--gradcache_trim_padding": "true",
          # GradCache already bounds memory to one chunk (50M, chunk 4: ~7 GB of 64), so
          # recomputing every forward for checkpointing only costs time.
          "--gradient_checkpointing": "false"}


def parse(path: Path) -> list[tuple[str, str]]:
    tokens = path.read_text().split()
    return list(zip(tokens[::2], tokens[1::2]))


def arms() -> dict[str, tuple[str, str]]:
    base = parse(TEMPLATE)
    out = {}
    for (loss, lo), (batch, bo), seed in product(LOSSES.items(), BATCHES.items(), SEEDS):
        name = f"s050m_ck540k_{loss}_{batch}_seed{seed}"
        over = {**COMMON, **lo, **bo, "--seed": seed,
                "--run_name": RUN_PREFIX + name, "--output_dir": f"./runs/{RUN_PREFIX}{name}"}
        pairs = [(k, over.pop(k) if k in over else v) for k, v in base]
        pairs += list(over.items())
        args = "\n".join(f"{k} {v}" for k, v in pairs) + "\n"
        desc = (f"C8 x C19 ARM: loss={loss}, batches={batch}, seed {seed}. 50M from pretrained "
                f"checkpoint 540,423, ms-contrastive-100k only, 3 epochs, encoder every half epoch; "
                f"otherwise the Iona stage-2 recipe (P85 x K3, KL 10, lr 1e-4). sweeps/make_c8c19.py.\n")
        out[name] = (args, desc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    cli = ap.parse_args()
    want = arms()
    if cli.check:
        have = {d.name for d in OUT.iterdir() if d.is_dir()} if OUT.exists() else set()
        bad = sorted(set(want) ^ have) + [n for n in want if n in have and
                                          (OUT / n / "training.args").read_text() != want[n][0]]
        if bad:
            print("stale or missing arms:", bad)
            return 1
        print(f"{len(want)} arms match")
        return 0
    if OUT.exists():
        shutil.rmtree(OUT)
    for name, (args, desc) in want.items():
        (OUT / name).mkdir(parents=True)
        (OUT / name / "training.args").write_text(args)
        (OUT / name / "DESCRIPTION.md").write_text(desc)
    print(f"wrote {len(want)} arms to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
