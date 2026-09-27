"""Score external-baseline embeddings on the ms-contrastive-100k test eval, like our eval.

    PYTHONPATH=/home/khuss/code/msdelta /home/khuss/code/msdelta/.venv/bin/python \
        score_retrieval.py DATA_DIR EMBED.npy [OUT.json]

Uses the rows of meta.parquet with in_mp512 (== the prepared eval set, order checked at
export), groups by peptide_key(peptide, charge), and msdelta.contrastive.
retrieval_metrics_topk (cosine) for the `all` and `experimental` views, exactly as
eval_grouped_retrieval._variants. Also reports a Euclidean-distance ranking (GLEAMS'
native metric), the strict-valid-only subset, and the full export (no max_peaks cut).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from msdelta.contrastive import retrieval_metrics_topk
from msdelta.reranking import peptide_key


def metrics_euclid(emb, groups, k=100, chunk=2048):
    """retrieval_metrics_topk's MAP@R / Hit@1 with -L2 distance instead of cosine."""
    e = torch.as_tensor(emb, dtype=torch.float32)
    g = torch.as_tensor(groups, dtype=torch.long)
    n = len(e)
    n_rel_all = torch.bincount(g)[g] - 1
    k = min(k, n - 1)
    ranks = torch.arange(1, k + 1, dtype=torch.float32).unsqueeze(0)
    hit1 = mapr = 0.0
    scor = 0
    for s in range(0, n, chunk):
        q = torch.arange(s, min(s + chunk, n))
        sim = -torch.cdist(e[q], e)
        sim[torch.arange(len(q)), q] = float('-inf')
        idx = sim.topk(k, dim=1).indices
        hit = (g[idx] == g[q].unsqueeze(1)).float()
        nr = n_rel_all[q].float()
        keep = nr > 0
        hit, nr = hit[keep], nr[keep]
        prec = hit.cumsum(1) / ranks
        within = ranks <= nr.unsqueeze(1)
        hit1 += float(hit[:, 0].sum())
        mapr += float(((prec * hit * within).sum(1) / nr).sum())
        scor += int(keep.sum())
    return {"Hit@1": hit1 / scor, "MAP@R": mapr / scor, "queries": float(scor)}


def views(emb, meta, mask, tag):
    out = {}
    pep = np.asarray(meta["peptide"])[mask]
    chg = np.asarray(meta["charge"])[mask]
    src = np.asarray(meta["source"])[mask]
    e = emb[mask]
    keys = [peptide_key(p, int(c)) for p, c in zip(pep, chg)]
    groups = np.unique(np.array(keys), return_inverse=True)[1]
    exp = src == "experimental"
    for variant, m in (("all", np.ones(len(groups), bool)), ("experimental", exp)):
        g = np.unique(groups[m], return_inverse=True)[1]
        cos = retrieval_metrics_topk(torch.from_numpy(e[m]), g)
        euc = metrics_euclid(e[m], g)
        out[f"{tag}/{variant}"] = {"cos_MAP@R": cos["MAP@R"], "cos_Hit@1": cos["Hit@1"],
                                   "l2_MAP@R": euc["MAP@R"], "l2_Hit@1": euc["Hit@1"],
                                   "n": int(m.sum()), "queries": cos["queries"]}
    return out


def main(data_dir, emb_path, out_path=None):
    meta = pq.read_table(Path(data_dir) / "meta.parquet").to_pydict()
    emb = np.load(emb_path).astype(np.float32)
    missing = np.isnan(emb).any(1)
    emb[missing] = 0.0
    in_mp = np.asarray(meta["in_mp512"])
    print(f"[score] {len(emb):,} rows, {missing.sum()} without embedding (zeroed), "
          f"{in_mp.sum():,} in the prepared eval set")
    res = {"missing": int(missing.sum())}
    res |= views(emb, meta, in_mp, "mp512")
    res |= views(emb, meta, np.ones(len(emb), bool) & ~missing, "all_exported")
    sv = Path(data_dir) / "strict_valid.npy"
    if sv.exists():
        strict = np.load(sv)
        res["strict_valid_in_mp512"] = int((strict & in_mp).sum())
        res |= views(emb, meta, in_mp & strict, "mp512_strictvalid")
    for k, v in res.items():
        print(k, v if not isinstance(v, dict) else
              " ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}"
                       for a, b in v.items()))
    if out_path:
        Path(out_path).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
