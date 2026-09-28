"""Library search: experimental spectra queried against a consensus-only library (C25, K79-C).

Queries are the EXPERIMENTAL spectra; the gallery ("library") is ONLY the CONSENSUS spectra,
one per (peptide, charge) group. Consensus spectra are never queries and experimental
spectra are never library entries. Each query has exactly one correct library entry, its
own group's consensus; a query whose group has no consensus is UNSCORABLE -- counted in
`library/unscorable` and excluded. Consensus entries with no experimental query stay in the
library as distractors.

Per query we compute the exact rank of the correct entry among the library entries the
filter allows (no top-k cap). Ties are counted AGAINST the query (a library entry with the
same similarity as the correct one ranks ahead of it), so degenerate embeddings cannot
look good. A correct entry the filter excludes is a miss (rank = inf, reciprocal rank 0).

  Hit@1, Hit@5  correct entry at rank <= 1 / <= 5 (over all queries; excluded = miss).
  MRR           mean of 1/rank (excluded contributes 0).
  rank stats    (K97-C) R = exact rank of the correct entry among the WHOLE library, or,
                under a filter, among the filter-allowed candidates. rank_{mean,median,p90,
                p99,max} are over the FOUND queries only (the filter kept the correct entry),
                so excluded queries cannot poison them; `excluded` counts the rest.
                Percentiles use linear interpolation (numpy default). frac_rank_le_{1,5,10,
                100} is over ALL queries of the subset (excluded = not <= k), so
                frac_rank_le_1 == Hit@1.
(No MAP@R: with one correct entry it is just Hit@1 -- dropped, K97-C.) Ties count as
misses (K98-C). Only on data that has consensus spectra (K99-C).

Filtered split (project rule, notes/PLAN.md Rules; msdelta.eval.filtered_retrieval): every
number without a filter (`open`), with a 20 ppm filter and an isotope-tolerant 20 ppm filter,
comparing the query's precursor m/z with each LIBRARY entry's precursor m/z (same charge);
each on `full` (all scorable queries), `F` (the 20 ppm filter excludes the correct library
entry) and `Fbar` (it does not). With R = 1, F is also "every positive excluded", so there is
no separate F_all; `rescue` = unfiltered Hit@1 on F. net_loss / net_gain as in
filtered_retrieval. On ms-contrastive-100k the precursors are theoretical, so experimental
and consensus precursors of one group agree and F is (almost) empty there.

Keys (flat, for the per-model metrics JSON):
  library/{Hit@1,Hit@5,MRR,queries,unscorable,library_size}
  library/rank_{mean,median,p90,p99,max}, library/frac_rank_le_{1,5,10,100}   (unfiltered)
  library/{open,20ppm,iso20ppm}/{full,F,Fbar}/{Hit@1,Hit@5,MRR,n,excluded,
      rank_{mean,median,p90,p99,max},frac_rank_le_{1,5,10,100}}
      (metrics omitted when n=0; rank_* omitted when no query in the cell was found)
  library/{20ppm,iso20ppm}/{net_loss,net_gain}, library/rescue       (rescue only when F nonempty)
"""
from __future__ import annotations

import numpy as np
import torch

from msdelta.eval.filtered_retrieval import FILTERS, make_filters

SUBSETS = ("full", "F", "Fbar")
METRICS = ("Hit@1", "Hit@5", "MRR")
RANK_KS = (1, 5, 10, 100)
RANK_STATS = tuple(f"rank_{s}" for s in ("mean", "median", "p90", "p99", "max"))


def library_ranks(q_emb: torch.Tensor, q_groups: torch.Tensor, lib_emb: torch.Tensor,
                  lib_groups: torch.Tensor, allowed_fn, chunk: int = 1024) -> torch.Tensor:
    """Per query: rank (1-based, float) of its best allowed correct library entry, inf when
    the filter allows none. Ties count against the query."""
    n = len(q_emb)
    ranks = torch.full((n,), float("inf"), device=q_emb.device)
    for s in range(0, n, chunk):
        q = torch.arange(s, min(s + chunk, n), device=q_emb.device)
        sim = q_emb[q] @ lib_emb.T
        allowed = allowed_fn(q)
        correct = lib_groups[None] == q_groups[q, None]
        best = torch.where(correct & allowed, sim, torch.full_like(sim, float("-inf"))).max(1).values
        ahead = ((sim >= best[:, None]) & allowed & ~correct).sum(1)
        ranks[q] = torch.where(torch.isfinite(best), 1.0 + ahead.float(),
                               torch.full_like(best, float("inf")))
    return ranks


def _summ(ranks: torch.Tensor, mask: torch.Tensor) -> dict:
    c = int(mask.sum())
    if not c:
        return {"n": 0, **{m: None for m in METRICS}}
    r = ranks[mask]
    out = {"n": c, "Hit@1": float((r <= 1).float().mean()),
           "Hit@5": float((r <= 5).float().mean()), "MRR": float((1.0 / r).mean())}
    found = r[torch.isfinite(r)].double().cpu().numpy()
    out["excluded"] = c - len(found)
    out |= {f"frac_rank_le_{k}": float((r <= k).float().mean()) for k in RANK_KS}
    if len(found):
        out |= {"rank_mean": float(found.mean()), "rank_median": float(np.median(found)),
                "rank_p90": float(np.percentile(found, 90)),
                "rank_p99": float(np.percentile(found, 99)), "rank_max": float(found.max())}
    return out


def _extras(cell: dict) -> dict:
    """The rank-statistic entries of a _summ cell that are present."""
    keys = ("excluded", *(f"frac_rank_le_{k}" for k in RANK_KS), *RANK_STATS)
    return {k: float(cell[k]) for k in keys if k in cell}


def library_report(embeddings, groups, experimental, consensus, precursor=None, charge=None,
                   device=None, chunk: int = 1024) -> dict:
    """Flat `library/...` metrics for one embedding set (all rows; masks select roles)."""
    e = torch.nn.functional.normalize(torch.as_tensor(embeddings).float(), dim=-1)
    if device is not None:
        e = e.to(device)
    groups = np.asarray(groups)
    experimental = np.asarray(experimental, dtype=bool)
    consensus = np.asarray(consensus, dtype=bool)
    assert not (experimental & consensus).any(), "a row cannot be both query and library"
    lib_idx = np.flatnonzero(consensus)
    q_idx = np.flatnonzero(experimental)
    scorable = np.isin(groups[q_idx], groups[lib_idx])
    out = {"library/unscorable": float((~scorable).sum()),
           "library/library_size": float(len(lib_idx))}
    q_idx = q_idx[scorable]
    out["library/queries"] = float(len(q_idx))
    if not len(q_idx):
        return out
    dev = e.device
    qe, le = e[torch.from_numpy(q_idx).to(dev)], e[torch.from_numpy(lib_idx).to(dev)]
    qg = torch.as_tensor(groups[q_idx], dtype=torch.long, device=dev)
    lg = torch.as_tensor(groups[lib_idx], dtype=torch.long, device=dev)

    if precursor is None:
        n_lib = len(lib_idx)
        filters = {"open": lambda q: torch.ones(len(q), n_lib, dtype=torch.bool, device=dev)}
    else:
        prec = torch.as_tensor(np.asarray(precursor), dtype=torch.float64, device=dev)
        z = torch.as_tensor(np.asarray(charge), dtype=torch.long, device=dev)
        qi, li = torch.from_numpy(q_idx).to(dev), torch.from_numpy(lib_idx).to(dev)
        filters = make_filters(prec[qi], z[qi], gallery_prec=prec[li], gallery_charge=z[li])

    ranks = {w: library_ranks(qe, qg, le, lg, fn, chunk=chunk) for w, fn in filters.items()}
    top = _summ(ranks["open"], torch.ones(len(q_idx), dtype=torch.bool, device=dev))
    out |= {f"library/{m}": top[m] for m in METRICS}
    out |= {f"library/{k}": v for k, v in _extras(top).items() if k != "excluded"}
    if precursor is None:
        return out

    full = torch.ones(len(q_idx), dtype=torch.bool, device=dev)
    fail = ~torch.isfinite(ranks["20ppm"])       # 20 ppm excludes the correct entry
    masks = {"full": full, "F": fail, "Fbar": ~fail}
    open_hit = ranks["open"] <= 1
    nf = len(q_idx)
    for w in FILTERS:
        for s in SUBSETS:
            cell = _summ(ranks[w], masks[s])
            out[f"library/{w}/{s}/n"] = float(cell["n"])
            for m in METRICS:
                if cell[m] is not None:
                    out[f"library/{w}/{s}/{m}"] = cell[m]
            out |= {f"library/{w}/{s}/{k}": v for k, v in _extras(cell).items()}
        if w != "open":
            hit = ranks[w] <= 1
            out[f"library/{w}/net_loss"] = float((open_hit & ~hit).sum()) / nf
            out[f"library/{w}/net_gain"] = float((~open_hit & hit).sum()) / nf
    if "library/open/F/Hit@1" in out:
        out["library/rescue"] = out["library/open/F/Hit@1"]
    return out
