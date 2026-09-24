"""A4: LiT-style contrastive peptide student with distinguishable hard negatives. PLAN.md A4.

    python sweeps/make_a4.py
    python sweeps/make_a4.py --check

A1 regresses the student onto the frozen teacher's embedding (MSE), so nothing ever pushes
a WRONG peptide away: an adjacent-residue swap lands next to the truth (70.5% on the
near-miss test; A3 showed the readout is not the cause). A4 trains the same student with
the LiT objective (Zhai et al., CVPR 2022: contrastive against a FROZEN tower), SupCon-style
multi-positive, negatives = the batch's other peptides + synthetic rearrangements that are
spectrally distinguishable (never reversals -- the FDR decoys are reversals).

2 x 2 x 3 seeds, everything else = configs/a1-align-100k-050m-c7s600 (teacher C7-50m step
600, the A1 cache), batch 256 for more in-batch negatives:

    hn0 / hn4       hard negatives per peptide (0 isolates the loss change itself)
    mse0 / mse01    weight of the A1 MSE term kept beside the contrastive loss

Selection is on VALIDATION (in-training cross-modal Hit@1); the synthetic near-miss rate
is reported for information only. Headline = PSMs at 1% FDR on the MSFragger data, with
the cosine_null leakage AUROC (must stay ~0.5).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "configs" / "a1-align-100k-050m-c7s600" / "training.args"
OUT = REPO / "configs" / "sweep-a4"
CACHE = "/lus/flare/projects/UIC-HPC/khuss/msdelta/align-targets-a1-050m-c7s600"
NEG = {"hn0": "0", "hn4": "4"}
MSE = {"mse0": "0.0", "mse01": "0.1"}
SEEDS = ("0", "1", "2")
RUN_PREFIX = "v2_a4-"


def render(neg, mse, seed):
    name = f"lit_{neg}_{mse}_seed{seed}"
    over = {"--align_loss": "lit", "--align_temperature": "0.05", "--mse_weight": MSE[mse],
            "--hard_negatives": NEG[neg], "--neg_min_delta": "0.05",
            "--per_device_train_batch_size": "256", "--seed": seed, "--target_cache": CACHE,
            "--run_name": f"{RUN_PREFIX}{name}", "--output_dir": f"./runs/{RUN_PREFIX}{name}"}
    tokens = BASE.read_text().split()
    lines, seen = [], set()
    for f, v in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{f} {over.get(f, v)}"); seen.add(f)
    lines += [f"{f} {v}" for f, v in over.items() if f not in seen]
    return name, "\n".join(lines) + "\n"


def description(neg, mse, seed):
    return (f"A4 ARM: LiT cross-modal contrastive student, {NEG[neg]} hard negatives, MSE "
            f"weight {MSE[mse]}, seed {seed}. Otherwise configs/a1-align-100k-050m-c7s600 "
            f"(teacher C7-50m step 600, cache {CACHE}), batch 256. PLAN.md A4.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    cli = ap.parse_args()
    combos = [(n, m, s) for n in NEG for m in MSE for s in SEEDS]
    if cli.check:
        stale = [n for n, t in (render(*c) for c in combos)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != t
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(combos)} arms match {BASE.relative_to(REPO)}")
        return 0
    if OUT.exists():
        shutil.rmtree(OUT)
    for c in combos:
        name, text = render(*c)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(*c))
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
