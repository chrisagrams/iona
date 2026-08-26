"""Δm bias-curve chemistry-alignment probe.

Evaluates each head's learned bias curve on a dense Δm grid, detects strong
peaks, and tests whether their locations align with chemically meaningful Δm
values (isotopes, neutral losses, amino-acid residues) *better than chance*.

The guardrail: with ~27 reference values packed into [2, 200] Da, some peak is
always near some reference — so "a peak landed near a residue mass" means
nothing without asking "more often than random peaks would?". Each head gets a
p-value from `binomtest(n_aligned, n_peaks, chance_rate)`. `alignment_metrics`
is the flat wandb dict logged inline every probe step; the encoder never sees a
spectrum, so it is cheap and safe at that cadence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import binomtest

from .model import MSEncoder
from .viz import ISOTOPES, NEUTRAL_LOSSES, RESIDUES_AA20


# ---------- references ----------

def reference_set(kinds: list[str]) -> dict[str, float]:
    refs: dict[str, float] = {}
    if "isotope" in kinds:
        refs.update({f"iso:{k}": v for k, v in ISOTOPES.items()})
    if "loss" in kinds:
        refs.update({f"loss:{k}": v for k, v in NEUTRAL_LOSSES.items()})
    if "residue" in kinds:
        refs.update({f"res:{k}": v for k, v in RESIDUES_AA20.items()})
    return refs


# ---------- core analysis ----------

@dataclass
class RangeSpec:
    name: str
    lo: float
    hi: float
    step: float
    kinds: list[str]
    tol: float          # Da; how close a peak must be to a reference to count
    prominence: float   # min peak prominence (logit units) to count as "strong"
    min_sep: float      # Da; minimum separation between detected peaks


@torch.no_grad()
def _eval_curves(enc: MSEncoder, lo: float, hi: float, step: float):
    # Build the grid on the bias module's device so this works both for a
    # CPU-loaded checkpoint (CLI) and the live GPU encoder (inline probe).
    dev = next(enc.bias_module.parameters()).device
    grid = torch.arange(lo, hi + step / 2, step, dtype=torch.float32, device=dev)
    curves = enc.bias_module.evaluate(grid).cpu().numpy()  # (N, H)
    return grid.cpu().numpy(), curves


def _chance_rate(abs_grid: np.ndarray, ref_vals: np.ndarray, tol: float) -> float:
    """Fraction of the (folded, |Δm|) axis within `tol` of any reference."""
    covered = np.zeros_like(abs_grid, dtype=bool)
    for v in ref_vals:
        covered |= np.abs(abs_grid - v) < tol
    return float(covered.mean())


def analyze_range(enc: MSEncoder, spec: RangeSpec) -> dict[str, Any]:
    refs = reference_set(spec.kinds)
    ref_items = list(refs.items())
    ref_vals = np.array([v for _, v in ref_items], dtype=np.float64)

    grid, curves = _eval_curves(enc, spec.lo, spec.hi, spec.step)
    H = curves.shape[1]
    distance = max(1, int(spec.min_sep / spec.step))

    # Chance computed on |Δm| since a feature at ±v both indicate the same
    # chemical relationship (Δm is signed; the bias may be asymmetric).
    abs_grid = np.abs(grid)
    chance = _chance_rate(abs_grid, ref_vals, spec.tol)

    per_head = []
    covered_refs: set[str] = set()
    for h in range(H):
        c = curves[:, h]
        pk, props = find_peaks(c, prominence=spec.prominence, distance=distance)
        peaks_dm = grid[pk]
        proms = props["prominences"]

        hits = []
        for dm, prom in zip(peaks_dm, proms):
            j = int(np.argmin(np.abs(ref_vals - abs(dm))))
            off = abs(dm) - ref_vals[j]
            if abs(off) < spec.tol:
                hits.append({
                    "dm": round(float(dm), 4),
                    "ref": ref_items[j][0],
                    "ref_dm": float(ref_vals[j]),
                    "offset": round(float(off), 4),
                    "prominence": round(float(prom), 3),
                })
                covered_refs.add(ref_items[j][0])

        n_pk = int(len(pk))
        n_hit = len(hits)
        # Binomial null: would n_hit aligned out of n_pk peaks be surprising
        # if each peak landed within tol of a ref with probability `chance`?
        if n_pk > 0:
            pval = float(binomtest(n_hit, n_pk, chance, alternative="greater").pvalue)
        else:
            pval = 1.0

        per_head.append({
            "head": h,
            "n_peaks": n_pk,
            "n_aligned": n_hit,
            "expected_by_chance": round(chance * n_pk, 2),
            "enrichment": round((n_hit / n_pk) / chance, 2) if n_pk and chance > 0 else 0.0,
            "p_value": pval,
            "hits": sorted(hits, key=lambda x: -x["prominence"]),
        })

    return {
        "range": spec.name,
        "bounds": [spec.lo, spec.hi],
        "targets": spec.kinds,
        "tol_da": spec.tol,
        "prominence": spec.prominence,
        "chance_rate": round(chance, 4),
        "n_refs": len(refs),
        "refs_covered": sorted(covered_refs),
        "n_refs_covered": len(covered_refs),
        "per_head": per_head,
    }


def alignment_metrics(
    enc: MSEncoder,
    fine_tol: float = 0.02,
    coarse_tol: float = 0.1,
    prominence: float = 0.3,
) -> dict[str, float]:
    """Flat wandb dict from the Δm bias-curve chemistry alignment (mode b).

    Pure bias-curve analysis — the encoder never sees a spectrum, so this is
    fast (peak-find on `bias_h(Δm)`) and safe to call inline every probe step.
    Tracks, per range and overall: how many head×range tests beat chance
    (p<0.05), how many survive Bonferroni at p<0.01 (the strict gate), the
    best p-value, the peak enrichment, and reference coverage. Watching
    `align/n_sig01_bonf` over training shows whether the bias is *sharpening*
    onto chemistry as the model learns, rather than only at the final ckpt.
    """
    specs = [
        RangeSpec("fine", -5.0, 5.0, 0.001, ["isotope"],
                  tol=fine_tol, prominence=prominence, min_sep=0.05),
        RangeSpec("coarse", 2.0, 200.0, 0.01, ["loss", "residue"],
                  tol=coarse_tol, prominence=prominence, min_sep=0.3),
    ]
    results = [analyze_range(enc, spec) for spec in specs]
    all_p = [ph["p_value"] for res in results for ph in res["per_head"]]
    n_tests = max(1, len(all_p))   # 8 heads × 2 ranges = 16

    out: dict[str, float] = {}
    for res in results:
        name = res["range"]
        pvals = [ph["p_value"] for ph in res["per_head"]]
        enrich = [ph["enrichment"] for ph in res["per_head"]]
        out[f"align/{name}_n_sig05"] = float(sum(p < 0.05 for p in pvals))
        out[f"align/{name}_best_p"] = float(min(pvals)) if pvals else 1.0
        out[f"align/{name}_max_enrich"] = float(max(enrich)) if enrich else 0.0
        out[f"align/{name}_coverage"] = res["n_refs_covered"] / max(1, res["n_refs"])

    out["align/n_sig05"] = float(sum(p < 0.05 for p in all_p))
    out["align/n_sig01_bonf"] = float(sum(p * n_tests < 0.01 for p in all_p))
    out["align/best_p"] = float(min(all_p)) if all_p else 1.0
    return out
