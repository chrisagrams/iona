"""Retrieval with and without a precursor filter, split by where the filter is WRONG.

Standard evaluation since 2026-09-27 (user): every retrieval number is reported
  - without a filter (`open`),
  - with a plain precursor filter (`20ppm`: same charge, |Δ m/z| <= 20 ppm),
  - with an isotope-tolerant filter (`iso20ppm`: same charge, |Δ m/z − k·1.00336/z| <= 20 ppm
    for some k in −2..2, the standard fix for mis-assigned monoisotopic peaks),
and each of those on
  - `full`: every query with at least one positive,
  - `F`: queries the plain filter fails -- at least one positive falls outside its window,
  - `Fbar`: the queries it passes (no positive excluded),
  - `F_all`: queries it makes unanswerable -- every positive excluded.
F is always defined by the plain 20 ppm filter on the MEASURED precursor m/z of query and
gallery spectrum. Positives a filter excludes count as misses (R unchanged), so a filter can
only lose on F. On data whose precursors are theoretical (ms-contrastive-100k), F is empty.

Plus, per filtered setting: `net_loss` (share of queries right at Hit@1 unfiltered but wrong
filtered), `net_gain` (the reverse), and `rescue` = unfiltered Hit@1 on F_all (how often the
model finds what the filter would have made unfindable).

Origin: pbs/diag/filter_failure_eval.py (job 8873366, OBSERVATIONS 2026-09-27).

CROSS-MODAL (spectrum -> peptide; `crossmodal_report`, K77-A): the same three filters, but
the query spectrum's MEASURED precursor m/z is compared with each CANDIDATE PEPTIDE's
THEORETICAL m/z at the query's charge, (M + z*PROTON)/z; with candidate charges given
(peptide+charge candidates) the candidate's charge must equal the query's. Each query has
exactly ONE correct candidate, so F (the 20 ppm filter excludes a positive) and F_all (it
excludes every positive) are the same set; both keys are written for a uniform schema.
Metrics are Hit@1, Hit@5 and MRR (with one relevant item AP = MRR and MAP@R = Hit@1).
That part is numpy-only and needs no torch, so yHydra's Python 3.8 env can load this file
by path (baselines_wip/yhydra_crossmodal.py); it also carries the legacy yHydra windows
(`legacy_20ppm`, `legacy_1.1Da`: neutral mass, no charge check) the paper's numbers use.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:     # yHydra's numpy-only env loads this file for the cross-modal part
    torch = None

ISOTOPE = 1.00336
ISO_K = (-2, -1, 0, 1, 2)
PPM = 20.0
FILTERS = ("open", "20ppm", "iso20ppm")
SUBSETS = ("full", "F", "Fbar", "F_all")


def make_filters(prec: torch.Tensor, charge: torch.Tensor, ppm: float = PPM, iso_k=ISO_K) -> dict:
    """Allowed-gallery functions q -> bool[len(q), n] on measured precursor m/z."""
    n = len(prec)

    def open_(q):
        return torch.ones(len(q), n, dtype=torch.bool, device=prec.device)

    def ppm_(q):
        tol = ppm * 1e-6 * prec[q, None]
        return (charge[q, None] == charge[None]) & ((prec[q, None] - prec[None]).abs() <= tol)

    def iso_(q):
        tol = ppm * 1e-6 * prec[q, None]
        d = prec[q, None] - prec[None]
        step = ISOTOPE / charge[q, None].double()
        ok = torch.zeros(len(q), n, dtype=torch.bool, device=prec.device)
        for k in iso_k:
            ok |= (d - k * step).abs() <= tol
        return ok & (charge[q, None] == charge[None])

    return {"open": open_, "20ppm": ppm_, "iso20ppm": iso_}


def filtered_metrics(e: torch.Tensor, g: torch.Tensor, allowed_fn, chunk: int = 1024, k: int = 100):
    """Per-query AP@R (nan when R=0) and Hit@1 (0/1). Same arithmetic as
    contrastive.retrieval_metrics_topk; excluded and self entries are -inf and never hits."""
    n = len(e)
    k = min(k, n - 1)
    n_rel = torch.bincount(g)[g] - 1
    ranks = torch.arange(1, k + 1, dtype=torch.float32, device=e.device).unsqueeze(0)
    ap_q = torch.full((n,), float("nan"), device=e.device)
    hit_q = torch.zeros(n, device=e.device)
    for s in range(0, n, chunk):
        q = torch.arange(s, min(s + chunk, n), device=e.device)
        sim = e[q] @ e.T
        sim[~allowed_fn(q)] = float("-inf")
        sim[torch.arange(len(q), device=e.device), q] = float("-inf")
        val, idx = sim.topk(k, dim=1)
        hit = ((g[idx] == g[q].unsqueeze(1)) & torch.isfinite(val)).float()
        r = n_rel[q].float()
        prec = hit.cumsum(1) / ranks
        within = ranks <= r.unsqueeze(1)
        ap = (prec * hit * within).sum(1) / r.clamp(min=1)
        ap_q[q] = torch.where(r > 0, ap, torch.nan)
        hit_q[q] = hit[:, 0]
    return ap_q, hit_q


def positive_outside(g: torch.Tensor, allowed_fn, chunk: int = 2048):
    """Per query: number of positives, and how many of them the filter excludes."""
    n = len(g)
    n_pos = torch.bincount(g)[g] - 1
    n_out = torch.zeros(n, dtype=torch.long, device=g.device)
    for s in range(0, n, chunk):
        q = torch.arange(s, min(s + chunk, n), device=g.device)
        pos = g[q, None] == g[None]
        pos[torch.arange(len(q), device=g.device), q] = False
        n_out[q] = (pos & ~allowed_fn(q)).sum(1)
    return n_pos, n_out


def subset_masks(g: torch.Tensor, filters: dict):
    n_pos, n_out = positive_outside(g, filters["20ppm"])
    valid = n_pos > 0
    masks = {"full": valid, "F": valid & (n_out > 0), "F_all": valid & (n_out == n_pos)}
    masks["Fbar"] = valid & ~masks["F"]
    return masks, n_pos, n_out


def summarise(ap, hit, masks, open_hit=None) -> dict:
    out = {}
    for sname, m in masks.items():
        c = int(m.sum())
        out[sname] = {"n": c,
                      "MAP@R": float(ap[m].mean()) if c else None,
                      "Hit@1": float(hit[m].mean()) if c else None}
    if open_hit is not None:
        full = masks["full"]
        lost = (open_hit > 0) & (hit == 0) & full
        gained = (open_hit == 0) & (hit > 0) & full
        nf = max(int(full.sum()), 1)
        out["net_loss"] = float(lost.sum()) / nf
        out["net_gain"] = float(gained.sum()) / nf
    return out


def filtered_report(embeddings: torch.Tensor, groups, precursor, charge, device=None) -> dict:
    """{filter: {subset: {n, MAP@R, Hit@1}, net_loss, net_gain}, rescue} for one embedding set."""
    e = torch.nn.functional.normalize(torch.as_tensor(embeddings).float(), dim=-1)
    if device is not None:
        e = e.to(device)
    g = torch.as_tensor(groups, dtype=torch.long, device=e.device)
    prec = torch.as_tensor(precursor, dtype=torch.float64, device=e.device)
    z = torch.as_tensor(charge, dtype=torch.long, device=e.device)
    filters = make_filters(prec, z)
    masks, _, _ = subset_masks(g, filters)
    out, open_hit = {}, None
    for w in FILTERS:
        ap, hit = filtered_metrics(e, g, filters[w])
        if w == "open":
            open_hit = hit
            out[w] = summarise(ap, hit, masks)
        else:
            out[w] = summarise(ap, hit, masks, open_hit)
    out["rescue"] = out["open"]["F_all"]["Hit@1"]
    return out


def flatten(report: dict, prefix: str) -> dict[str, float]:
    """Flat float keys for the per-model metrics JSON, e.g.
    `experimental/20ppm/F/MAP@R`, `experimental/20ppm/net_loss`, `experimental/rescue`.
    Empty subsets are omitted (their `.../n` is 0)."""
    flat = {}
    for w in FILTERS:
        for s in SUBSETS:
            cell = report[w][s]
            flat[f"{prefix}/{w}/{s}/n"] = float(cell["n"])
            for m in ("MAP@R", "Hit@1"):
                if cell[m] is not None:
                    flat[f"{prefix}/{w}/{s}/{m}"] = cell[m]
        for extra in ("net_loss", "net_gain"):
            if extra in report[w]:
                flat[f"{prefix}/{w}/{extra}"] = report[w][extra]
    if report["rescue"] is not None:
        flat[f"{prefix}/rescue"] = report["rescue"]
    return flat


# ------------------------------------------------------------------ cross-modal (K77-A)
PROTON = 1.007276
XM_METRICS = ("hit@1", "hit@5", "mrr")
LEGACY_WINDOWS = {"legacy_20ppm": (20.0, True), "legacy_1.1Da": (1.1, False)}


def crossmodal_filters(query_mz, query_charge, cand_mass, cand_charge=None, ppm: float = PPM,
                       iso_k=ISO_K) -> dict:
    """Allowed-candidate functions q (index array) -> bool[len(q), n_cand].

    query_mz: MEASURED precursor m/z per query spectrum; query_charge: its charge;
    cand_mass: THEORETICAL neutral monoisotopic mass per candidate peptide; cand_charge:
    per-candidate charge, or None when candidates carry no charge (yHydra's sequences).
      open        everything
      20ppm       |mz_q - mz_c(z_q)| <= ppm * mz_q, same charge
      iso20ppm    |mz_q - mz_c(z_q) - k*ISOTOPE/z_q| <= ppm * mz_q for some k, same charge
      legacy_*    the yHydra comparison's original windows (window_hits): neutral masses,
                  |M_c - (mz_q - PROTON) * z_q| <= tol (Da, or ppm of that mass), no charge
                  check. Kept so the published +-1.1 Da numbers are reproduced exactly.
    """
    qmz = np.asarray(query_mz, dtype=np.float64)
    z = np.asarray(query_charge, dtype=np.int64)
    cm = np.asarray(cand_mass, dtype=np.float64)
    cz = None if cand_charge is None else np.asarray(cand_charge, dtype=np.int64)
    nc = len(cm)

    def same(q):
        return np.ones((len(q), nc), dtype=bool) if cz is None else z[q, None] == cz[None]

    def delta(q):       # measured - theoretical m/z at the query's charge
        zq = z[q, None].astype(np.float64)
        return qmz[q, None] - (cm[None] + zq * PROTON) / zq

    def open_(q):
        return np.ones((len(q), nc), dtype=bool)

    def ppm_(q):
        return same(q) & (np.abs(delta(q)) <= ppm * 1e-6 * qmz[q, None])

    def iso_(q):
        d, tol = delta(q), ppm * 1e-6 * qmz[q, None]
        step = ISOTOPE / z[q, None].astype(np.float64)
        ok = np.zeros((len(q), nc), dtype=bool)
        for k in iso_k:
            ok |= np.abs(d - k * step) <= tol
        return ok & same(q)

    def legacy(tol, in_ppm):
        def f(q):
            m = (qmz[q] - PROTON) * z[q].astype(np.float64)
            lim = m[:, None] * tol * 1e-6 if in_ppm else tol
            return np.abs(cm[None] - m[:, None]) <= lim
        return f

    out = {"open": open_, "20ppm": ppm_, "iso20ppm": iso_}
    out.update({name: legacy(*spec) for name, spec in LEGACY_WINDOWS.items()})
    return out


def crossmodal_ranks(queries, cands, truth, allowed_fn=None, metric: str = "cos",
                     chunk: int = 1024):
    """Per query: rank of its true candidate among the ALLOWED candidates (0 = top; ties go
    to the true candidate), whether the filter allows it, and how many candidates it allows.
    metric "cos" (cosine) or "l2" (negative squared distance). float32, like the callers."""
    q = np.asarray(queries, np.float32)
    c = np.asarray(cands, np.float32)
    truth = np.asarray(truth, dtype=np.int64)
    if metric == "cos":
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
    n = len(q)
    rank = np.zeros(n, dtype=np.int64)
    inside = np.ones(n, dtype=bool)
    size = np.full(n, len(c), dtype=np.int64)
    for s in range(0, n, chunk):
        idx = np.arange(s, min(s + chunk, n))
        qq = q[idx]
        sim = qq @ c.T if metric == "cos" else -(
            (qq ** 2).sum(1)[:, None] + (c ** 2).sum(1)[None] - 2 * qq @ c.T)
        t = truth[idx]
        true = sim[np.arange(len(idx)), t]
        if allowed_fn is not None:
            ok = allowed_fn(idx)
            size[idx] = ok.sum(1)
            inside[idx] = ok[np.arange(len(idx)), t]
            sim = np.where(ok, sim, -np.inf)
        rank[idx] = (sim > true[:, None]).sum(1)
    return rank, inside, size


def _xm_per_query(rank, inside):
    return {"hit@1": (inside & (rank == 0)).astype(np.float64),
            "hit@5": (inside & (rank < 5)).astype(np.float64),
            "mrr": np.where(inside, 1.0 / (rank + 1), 0.0)}


def crossmodal_report(queries, cands, truth, query_mz, query_charge, cand_mass,
                      cand_charge=None, metric: str = "cos", chunk: int = 1024) -> dict:
    """{filter: {subset: {n, hit@1, hit@5, mrr}, net_loss, net_gain}, rescue} for
    spectrum -> peptide retrieval (module docstring). `truth[i]` indexes query i's correct
    candidate. F = F_all = queries whose correct candidate the plain 20 ppm filter excludes;
    Fbar = the rest. rescue = unfiltered Hit@1 on F."""
    filters = crossmodal_filters(query_mz, query_charge, cand_mass, cand_charge)
    per = {}
    for w in FILTERS:
        rank, inside, _ = crossmodal_ranks(queries, cands, truth,
                                           None if w == "open" else filters[w], metric, chunk)
        per[w] = (_xm_per_query(rank, inside), inside)
    passes = per["20ppm"][1]
    masks = {"full": np.ones(len(passes), dtype=bool), "F": ~passes, "Fbar": passes,
             "F_all": ~passes}
    open_hit = per["open"][0]["hit@1"]
    nf = max(len(passes), 1)
    out = {}
    for w in FILTERS:
        vals = per[w][0]
        out[w] = {s: dict({"n": int(m.sum())},
                          **{k: (float(vals[k][m].mean()) if m.any() else None)
                             for k in XM_METRICS})
                  for s, m in masks.items()}
        if w != "open":
            hit = vals["hit@1"]
            out[w]["net_loss"] = float(((open_hit > 0) & (hit == 0)).sum()) / nf
            out[w]["net_gain"] = float(((open_hit == 0) & (hit > 0)).sum()) / nf
    out["rescue"] = out["open"]["F"]["hit@1"]
    return out


def crossmodal_flatten(report: dict, prefix: str = "crossmodal") -> dict:
    """Flat float keys, e.g. `crossmodal/20ppm/F/hit@1`, `crossmodal/iso20ppm/net_loss`,
    `crossmodal/rescue`; prefix "" gives `20ppm/F/hit@1`. Empty subsets keep only `.../n` (0)."""
    pre = f"{prefix}/" if prefix else ""
    flat = {}
    for w in FILTERS:
        for s in SUBSETS:
            cell = report[w][s]
            flat[f"{pre}{w}/{s}/n"] = float(cell["n"])
            for m in XM_METRICS:
                if cell[m] is not None:
                    flat[f"{pre}{w}/{s}/{m}"] = cell[m]
        for extra in ("net_loss", "net_gain"):
            if extra in report[w]:
                flat[f"{pre}{w}/{extra}"] = report[w][extra]
    if report["rescue"] is not None:
        flat[f"{pre}rescue"] = report["rescue"]
    return flat


def measured_precursor(column):
    """Measured precursor m/z as float64, or None when absent or incomplete (0, NaN, None;
    grouped_retrieval writes 0.0 for a missing value); callers then skip filtered keys."""
    if column is None:
        return None
    p = np.array([np.nan if v is None else v for v in column], dtype=np.float64)
    if len(p) == 0 or not np.all(np.isfinite(p) & (p > 0)):
        return None
    return p
