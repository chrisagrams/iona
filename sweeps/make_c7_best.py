"""C7, best-model-first: continue our best contrastive encoders on ms-contrastive-100k.
PLAN.md C7 (Stage 3).

    python sweeps/make_c7_best.py
    python sweeps/make_c7_best.py --check

Does training on the large corpus put a learned encoder above the zero-parameter
baseline (binned cosine 0.1 Da: 0.730 experimental MAP@R on its test split)? Asked first
with the strongest starting point rather than from the pretrained checkpoint: each arm
CONTINUES a finished C1-recipe encoder (trained on the replicate corpus) for one epoch
of ms-contrastive-100k, same recipe and data flags as sweep-con100k.

    cont400m  from sweep-s400m_t0002_pk256_ep12_seed{s}-8857593   0.711-0.714 on the test
    cont050m  from sweep-s050m_t0002_pk256_ep24_seed{s}-8857593   0.655-0.657

The KL term now anchors to the STARTING encoder's own intensity head (a copy of
--pretrained_path), i.e. to the C1 model, not to the original pretrained checkpoint.

An encoder is saved every 300 steps (~1/4 epoch; ~5.7 h at 400m, ~1.8 h at 50m on one
tile), so msdelta.eval_grouped_retrieval can score the test split long before the epoch
ends. Checkpoints are model-only (save_only_model), ~1.6 GB each at 400m.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-con100k-best"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_con100k-best-"
RUNS = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs"
STARTS = {
    "cont400m": "sweep-s400m_t0002_pk256_ep12_seed{s}-8857593",
    "cont050m": "sweep-s050m_t0002_pk256_ep24_seed{s}-8857593",
}
SEEDS = ("0", "1", "2")
RECIPE = {
    "--dataset_repo": "chrisagrams/ms-contrastive-100k",
    "--dataset_format": "grouped",
    "--include_consensus": "false",
    "--exclude_replicate_peptides": "true",
    "--learning_rate": "1e-4",
    "--kl_weight": "10",
    "--temperature": "0.002",
    "--groups_per_batch": "85",
    "--replicates": "3",
    "--gradcache_chunk": "4",
    "--num_train_epochs": "1",
    "--save_only_model": "true",
    "--save_strategy": "steps",
    "--save_steps": "300",
    "--save_total_limit": "10",
    "--split_seed": "0",
}


def arm_name(start: str, seed: str) -> str:
    return f"{start}_ep01_seed{seed}"


def render_arm(start: str, seed: str) -> tuple[str, str]:
    name = arm_name(start, seed)
    over = dict(RECIPE, **{
        "--pretrained_path": f"{RUNS}/{STARTS[start].format(s=seed)}/final",
        "--seed": seed,
        "--run_name": f"{RUN_PREFIX}{name}",
        "--output_dir": f"./runs/{RUN_PREFIX}{name}",
    })
    lines, seen = [], set()
    tokens = TEMPLATE.read_text().split()
    for flag, value in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{flag} {over.get(flag, value)}")
        seen.add(flag)
    lines += [f"{f} {v}" for f, v in over.items() if f not in seen]
    return name, "\n".join(lines) + "\n"


def description(start: str, seed: str) -> str:
    return (f"C7 BEST-FIRST ARM: continue {STARTS[start].format(s=seed)} (a finished "
            f"C1-recipe encoder) for one epoch of ms-contrastive-100k, seed {seed}. Same "
            f"recipe as sweep-con100k (t 0.002, P85 x K3, GradCache 4, experimental "
            f"spectra only, replicate-corpus peptides excluded). Encoder saved every 300 "
            f"steps for early scoring against binned cosine (0.730).\n")


def combos():
    return [(start, s) for start in STARTS for s in SEEDS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    cli = ap.parse_args()
    for start, s in combos():
        p = Path(RUNS, STARTS[start].format(s=s), "final", "model.safetensors")
        if not p.exists():
            raise SystemExit(f"missing start encoder {p}")
    if cli.check:
        stale = [n for n, t in (render_arm(*c) for c in combos())
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != t
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(combos())} arms match {TEMPLATE.relative_to(REPO)}")
        return 0
    if OUT.exists():
        shutil.rmtree(OUT)
    for c in combos():
        name, text = render_arm(*c)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(*c))
    (OUT / ".template").write_text(f"{TEMPLATE.relative_to(REPO)}\ncontrastive-hp\n")
    print(f"arms={len(combos())} under {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
