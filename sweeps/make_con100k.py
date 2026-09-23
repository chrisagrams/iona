"""C7: contrastive fine-tuning on ms-contrastive-100k. PLAN.md C7.

    python sweeps/make_con100k.py --clean
    python sweeps/make_con100k.py --check

Every contrastive run so far trained on ms2-peptide-replicate-retrieval: ~900 peptides,
11,452 spectra, scored on 99 held-out groups. This corpus has 100,000 train analytes
(3 experimental replicates each, consensus off -- see grouped_retrieval) and a
peptide-disjoint 10,000-analyte test split, scored afterwards by
msdelta.eval_grouped_retrieval.

RECIPE: C1's best cell carried over (lr 1e-4, KL 10, t 0.002, GradCache 4), except
  - K = 3, because an analyte has 3 experimental spectra; P = 85 keeps the batch at
    255 spectra, C1's width of 256.
  - EPOCHS 1 and 3. One epoch here is ~300k spectra, already more than C1's entire
    24-epoch run (~275k), so epoch counts do not carry over.
  - peptides in the replicate corpus are excluded from training (the default), so these
    models can still be scored on it and fed to the reranking chain.

Arms: 50m@220k at 1 and 3 epochs, 400m@220k at 1 epoch; 3 seeds each.
Estimated one-tile cost from C1 throughput: 50m ~2.8 h/epoch, 400m ~10 h/epoch.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-con100k"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_con100k-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

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
    "--save_only_model": "true",
    "--save_strategy": "no",
    "--save_steps": "700",
    "--save_total_limit": "1",
    "--split_seed": "0",
}
SEEDS = ("0", "1", "2")
CELLS = [("50m", "220000", "1"), ("50m", "220000", "3"), ("400m", "220000", "1")]


def checkpoint(scale: str, ck: str) -> str:
    return f"{FROZEN}/msdelta-{scale}-production-01-checkpoint-{ck}"


def arm_name(scale, ck, ep, seed) -> str:
    return f"s{scale:0>4}_ck{int(ck) // 1000:03d}k_ep{int(ep):02d}_seed{seed}"


def render_arm(scale, ck, ep, seed) -> tuple[str, str]:
    name = arm_name(scale, ck, ep, seed)
    overrides = dict(RECIPE)
    overrides.update({
        "--pretrained_path": checkpoint(scale, ck),
        "--num_train_epochs": ep,
        "--seed": seed,
        "--run_name": f"{RUN_PREFIX}{name}",
        "--output_dir": f"./runs/{RUN_PREFIX}{name}",
    })
    lines, seen = [], set()
    tokens = TEMPLATE.read_text().split()
    for flag, value in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{flag} {overrides.get(flag, value)}")
        seen.add(flag)
    for flag, value in overrides.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")
    return name, "\n".join(lines) + "\n"


def description(scale, ck, ep, seed) -> str:
    return (f"C7 ARM (ms-contrastive-100k): {scale} at pretraining checkpoint {ck}, "
            f"{ep} epoch(s), seed {seed} of {len(SEEDS)}. C1's recipe with K=3 "
            f"(experimental replicates only), P=85; replicate-corpus peptides excluded. "
            f"Scored on the corpus's test split by msdelta.eval_grouped_retrieval.\n")


def combos():
    return [(s, c, e, seed) for s, c, e in CELLS for seed in SEEDS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clean", action="store_true")
    cli = ap.parse_args()
    for s, c, _ in CELLS:
        if not Path(checkpoint(s, c), "model.safetensors").exists():
            raise SystemExit(f"{s}@{c} not frozen at {checkpoint(s, c)}")
    if cli.check:
        stale = [n for n, text in (render_arm(*c) for c in combos())
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(combos())} arms match {TEMPLATE.relative_to(REPO)}")
        return 0
    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for combo in combos():
        name, text = render_arm(*combo)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(*combo))
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\ncontrastive-hp\n")
    (REPO / "sweeps" / "arms" / "con100k.txt").write_text(
        "\n".join(arm_name(*c) for c in combos()) + "\n")
    print(f"arms={len(combos())} under {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
