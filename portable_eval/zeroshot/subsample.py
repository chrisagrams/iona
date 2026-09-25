"""Shrink a prepared split for CPU runs, keeping WHOLE analyte groups (seed 0).

    python subsample.py IN_DIR OUT_DIR N_EXPERIMENTAL

Draws analyte groups (peptide + charge) at random until at least N experimental spectra are
kept; every row of a chosen group (its experimental replicates and its consensus spectrum)
is kept, so each query keeps all of its relevant spectra and MAP@R stays well defined.
"""
import sys

import numpy as np
from datasets import load_from_disk


def main(src, dst, n):
    d = load_from_disk(src)
    groups = np.array(d["analyte_id"]); exp = np.array([s == "experimental" for s in d["source"]])
    uniq = np.unique(groups)
    order = np.random.default_rng(0).permutation(len(uniq))
    exp_per = {g: 0 for g in uniq}
    for g, e in zip(groups, exp):
        exp_per[g] += int(e)
    keep, total = set(), 0
    for i in order:
        g = uniq[i]
        if exp_per[g] < 1:
            continue
        keep.add(g); total += exp_per[g]
        if total >= n:
            break
    idx = np.flatnonzero(np.isin(groups, list(keep)))
    d.select(idx).save_to_disk(dst)
    print(f"{src}: kept {len(keep):,} groups, {len(idx):,} rows ({total:,} experimental) -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
