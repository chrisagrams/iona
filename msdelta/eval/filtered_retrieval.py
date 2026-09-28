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
"""
from __future__ import annotations

import torch

ISOTOPE = 1.00336
ISO_K = (-2, -1, 0, 1, 2)
PPM = 20.0
FILTERS = ("open", "20ppm", "iso20ppm")
SUBSETS = ("full", "F", "Fbar", "F_all")


def make_filters(prec: torch.Tensor, charge: torch.Tensor, ppm: float = PPM, iso_k=ISO_K,
                 gallery_prec: torch.Tensor | None = None,
                 gallery_charge: torch.Tensor | None = None) -> dict:
    """Allowed-gallery functions q -> bool[len(q), n] on measured precursor m/z.

    q indexes the QUERY arrays (prec, charge). The gallery is the same set of spectra unless
    gallery_prec / gallery_charge are given (library search: experimental queries against a
    separate consensus library); n is the gallery size."""
    g_prec = prec if gallery_prec is None else gallery_prec
    g_charge = charge if gallery_charge is None else gallery_charge
    n = len(g_prec)

    def open_(q):
        return torch.ones(len(q), n, dtype=torch.bool, device=prec.device)

    def ppm_(q):
        tol = ppm * 1e-6 * prec[q, None]
        return ((charge[q, None] == g_charge[None])
                & ((prec[q, None] - g_prec[None]).abs() <= tol))

    def iso_(q):
        tol = ppm * 1e-6 * prec[q, None]
        d = prec[q, None] - g_prec[None]
        step = ISOTOPE / charge[q, None].double()
        ok = torch.zeros(len(q), n, dtype=torch.bool, device=prec.device)
        for k in iso_k:
            ok |= (d - k * step).abs() <= tol
        return ok & (charge[q, None] == g_charge[None])

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
