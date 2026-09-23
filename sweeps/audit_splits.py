"""Do our train and test splits share peptides? Run on a compute node.

    python sweeps/audit_splits.py

  ms-denoise-100k    the dataset's own train/validation/test: peptide (and peptide+charge)
                     overlap, and the share of test spectra whose peptide is in train
  replicate corpus   the build_alignment_datasets split (seed 0) that contrastive, the
                     alignment student and R1 all use: overlap should be exactly zero
  rescorer           its spectrum-level 80/20 re-split of the held-out peptides: how many
                     rescorer-test spectra have a peptide in rescorer-train
"""
import collections
import os

# Offline BEFORE importing datasets: it reads these at import time, and compute nodes have
# no route to huggingface.co -- the first run retried HEAD requests until walltime.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
from datasets import load_dataset


def overlap(name, a, b):
    sa, sb = set(a), set(b)
    shared = sa & sb
    frac_rows = np.mean([x in sa for x in b]) if len(b) else float("nan")
    print(f"  {name:<44} train {len(sa):7d} unique | test {len(sb):6d} unique | "
          f"shared {len(shared):6d} | test ROWS whose key is in train: {frac_rows:.3f}")


print("=== ms-denoise-100k (dataset's own splits) ===")
d = load_dataset("chrisagrams/ms-denoise-100k")
tr = d["train"].select_columns(["peptide", "charge"])
for split in ("validation", "test"):
    te = d[split].select_columns(["peptide", "charge"])
    overlap(f"peptide: train vs {split}", tr["peptide"], te["peptide"])
    overlap(f"peptide+charge: train vs {split}",
            [f"{p}/{c}" for p, c in zip(tr["peptide"], tr["charge"])],
            [f"{p}/{c}" for p, c in zip(te["peptide"], te["charge"])])

print("\n=== replicate corpus, build_alignment_datasets split (seed 0, 10%) ===")
r = load_dataset("chrisagrams/ms2-peptide-replicate-retrieval")["test"].select_columns(["peptide", "charge"])
peps = sorted(set(r["peptide"]))
held = set(np.random.default_rng(0).choice(peps, size=max(1, int(len(peps) * 0.1)), replace=False).tolist())
tr_p = [p for p in r["peptide"] if p not in held]
te_p = [p for p in r["peptide"] if p in held]
overlap("peptide: alignment train vs validation", tr_p, te_p)
per = collections.Counter(te_p)
print(f"  held-out peptides: {len(per)}, spectra per peptide: median "
      f"{int(np.median(list(per.values())))}, max {max(per.values())}")

print("\n=== rescorer re-split of the held-out spectra (by spectrum, 20% test, seed 0) ===")
# mirrors rescoring.train_rescorer: groups are spectrum indices
idx = np.arange(len(te_p))
test_idx = set(np.random.default_rng(0).choice(idx, size=max(1, int(len(idx) * 0.2)), replace=False).tolist())
rs_tr = [te_p[i] for i in idx if i not in test_idx]
rs_te = [te_p[i] for i in idx if i in test_idx]
overlap("peptide: rescorer train vs rescorer test", rs_tr, rs_te)
