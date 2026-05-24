"""Quantitative chemistry-alignment analysis of a trained Δm bias module.

Loads a checkpoint, evaluates each head's bias curve on a dense Δm grid,
detects strong peaks, and tests whether their locations align with
chemically meaningful Δm values (isotopes, neutral losses, amino-acid
residues) *better than chance*. Implements the head-specialization
analysis from the project plan (§6.2a/b).

The key guardrail: alignment is scored against a binomial null. With ~27
reference values packed into [2, 200] Da, some peak is always near some
reference — so "a peak landed near a residue mass" means nothing without
asking "more often than random peaks would?". Each head gets a p-value
from `binomtest(n_aligned, n_peaks, chance_rate)`.

Usage:
    msdelta-analyze --ckpt runs/<run>/final.pt
    msdelta-analyze --ckpt runs/<run>/final.pt --plot --out runs/<run>/alignment.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import binomtest, spearmanr
from torch.utils.data import DataLoader

from .data import (
    ConsensusParquet,
    PreprocessConfig,
    pad_collate,
    split_paths,
)
from .model import DeltaBiasConfig, FourierConfig, ModelConfig, MSEncoder
from .viz import ISOTOPES, NEUTRAL_LOSSES, RESIDUES_AA20


# ---------- model loading ----------

def load_encoder(ckpt_path: str | Path) -> tuple[MSEncoder, dict[str, Any], int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    d = ckpt["cfg"]["model"]
    mcfg = ModelConfig(
        d_model=d["d_model"], n_heads=d["n_heads"], n_layers=d["n_layers"],
        ffn_mult=d["ffn_mult"], dropout=d["dropout"], max_peaks=d["max_peaks"],
        fourier_mz=FourierConfig(**d["fourier_mz"]),
        fourier_int=FourierConfig(**d["fourier_int"]),
        delta_bias=DeltaBiasConfig(**d["delta_bias"]),
    )
    enc = MSEncoder(mcfg)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    return enc, ckpt["cfg"], int(ckpt.get("step", -1))


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
    grid = torch.arange(lo, hi + step / 2, step, dtype=torch.float32)
    curves = enc.bias_module.evaluate(grid).cpu().numpy()  # (N, H)
    return grid.numpy(), curves


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


# ---------- reporting ----------

def print_report(result: dict[str, Any]) -> None:
    r = result
    print(f"\n=== {r['range'].upper()} range {r['bounds']} Da | "
          f"targets={'+'.join(r['targets'])} | tol={r['tol_da']} Da | "
          f"prominence≥{r['prominence']} ===")
    print(f"chance hit-rate {r['chance_rate']*100:.1f}%  "
          f"({r['n_refs']} reference values)")
    print(f"  {'head':>4} {'peaks':>6} {'aligned':>8} {'exp':>5} {'enrich':>7} {'p-value':>9}   top hits")
    for ph in r["per_head"]:
        sig = "*" if ph["p_value"] < 0.05 else " "
        hit_str = ", ".join(
            f"{h['ref'].split(':')[1]}@{h['dm']:+.3f}" for h in ph["hits"][:4]
        )
        print(f"  {ph['head']:>4} {ph['n_peaks']:>6} {ph['n_aligned']:>8} "
              f"{ph['expected_by_chance']:>5} {ph['enrichment']:>6.1f}x "
              f"{ph['p_value']:>8.1e}{sig}  {hit_str}")
    print(f"  coverage: {r['n_refs_covered']}/{r['n_refs']} references hit by ≥1 head"
          f"  → {', '.join(s.split(':')[1] for s in r['refs_covered'][:14])}")


def render_annotated(enc: MSEncoder, spec: RangeSpec, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    refs = reference_set(spec.kinds)
    ref_vals = np.array(list(refs.values()))
    grid, curves = _eval_curves(enc, spec.lo, spec.hi, spec.step)
    H = curves.shape[1]
    distance = max(1, int(spec.min_sep / spec.step))
    ncols = 4
    nrows = (H + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.2 * nrows), squeeze=False)
    for h in range(H):
        ax = axes[h // ncols][h % ncols]
        c = curves[:, h]
        ax.plot(grid, c, lw=0.7)
        for v in ref_vals:
            ax.axvline(v, color="tab:green", lw=0.3, alpha=0.5)
            if spec.lo < 0:
                ax.axvline(-v, color="tab:green", lw=0.3, alpha=0.5)
        pk, _ = find_peaks(c, prominence=spec.prominence, distance=distance)
        for p in pk:
            aligned = np.min(np.abs(ref_vals - abs(grid[p]))) < spec.tol
            ax.plot(grid[p], c[p], "v", ms=5,
                    color="red" if aligned else "gray")
        ax.set_title(f"head {h}", fontsize=8)
        ax.tick_params(labelsize=6)
    for h in range(H, nrows * ncols):
        axes[h // ncols][h % ncols].axis("off")
    fig.suptitle(f"{spec.name} bias peaks (red=aligned <{spec.tol} Da, gray=not)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------- functional probe (plan §6.2c) ----------
#
# The alignment score (b) reads the *learned bias curve* only — the encoder
# never sees a spectrum. The functional probe asks the load-bearing
# question: on real spectra, does the head's attention mass actually
# concentrate at the Δm values where its bias has peaks? A beautiful peak
# at 113 Da that the attention ignores is a vestigial parameter.
#
# For each head we histogram attention weights α^(h)_ij as a function of
# Δm_ij over validation batches (across all layers, since the bias is
# shared across layers), then divide by the pair-density histogram to get
# the *mean attention per pair* at each Δm — removing the confound that
# nearby peaks are simply more numerous. We compare that to bias_h(Δm) by
# Spearman correlation.


def build_val_loader(cfg: dict[str, Any], batch_size: int, num_workers: int = 0) -> DataLoader:
    dcfg = cfg["data"]
    pp = PreprocessConfig(
        intensity_threshold_frac=dcfg["intensity_threshold_frac"], top_n=dcfg["top_n"]
    )
    _, val_paths = split_paths(dcfg["root"], dcfg["n_val_files"])
    ds = ConsensusParquet(val_paths, preprocess=pp, seed=123)

    def collate(b):
        b = [x for x in b if x[0].numel() > 0]
        return pad_collate(b) if b else None

    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate)


@dataclass
class ProbeSpec:
    name: str
    lo: float
    hi: float
    n_bins: int


@torch.no_grad()
def functional_probe(
    enc: MSEncoder,
    loader: DataLoader,
    device: torch.device,
    spec: ProbeSpec,
    n_batches: int,
) -> dict[str, np.ndarray]:
    """Histogram attention mass vs Δm, density-normalized, per head."""
    enc.to(device).eval()
    enc.set_save_attn(True)
    H = enc.cfg.n_heads
    n_layers = enc.cfg.n_layers
    edges = torch.linspace(spec.lo, spec.hi, spec.n_bins + 1, device=device)
    centers = (0.5 * (edges[:-1] + edges[1:])).cpu().numpy()

    attn_sum = torch.zeros(H, spec.n_bins, device=device)  # Σ α over pairs×layers×batches
    pair_cnt = torch.zeros(spec.n_bins, device=device)      # # pairs (layer-independent)
    used = 0
    for batch in loader:
        if used >= n_batches:
            break
        if batch is None:
            continue
        batch = {k: v.to(device) for k, v in batch.items()}
        mz = batch["mz"]
        enc(mz, batch["log_int"], batch["key_padding_mask"])  # populates last_attn per block

        dm = mz.unsqueeze(-1) - mz.unsqueeze(-2)              # (B,K,K)
        kpm = batch["key_padding_mask"]
        K = mz.size(1)
        valid = (~kpm).unsqueeze(-1) & (~kpm).unsqueeze(-2)
        valid = valid & ~torch.eye(K, dtype=torch.bool, device=device)  # drop diagonal (bias=0 there)
        idx = torch.bucketize(dm, edges) - 1
        in_range = (idx >= 0) & (idx < spec.n_bins) & valid
        idx_flat = idx[in_range]

        pair_cnt.scatter_add_(0, idx_flat, torch.ones_like(idx_flat, dtype=torch.float))
        for blk in enc.blocks:
            attn = blk.attn.last_attn  # (B,H,K,K)
            for h in range(H):
                attn_sum[h].scatter_add_(0, idx_flat, attn[:, h][in_range].float())
        used += 1

    enc.set_save_attn(False)
    attn_sum_np = attn_sum.cpu().numpy()
    pair_cnt_np = pair_cnt.cpu().numpy()
    # mean attention per (pair, layer) at each Δm
    denom = np.maximum(pair_cnt_np * n_layers, 1.0)
    mean_attn = attn_sum_np / denom
    return {"centers": centers, "mean_attn": mean_attn, "pair_cnt": pair_cnt_np,
            "n_batches": used}


def report_probe(enc: MSEncoder, probe: dict[str, np.ndarray], spec: ProbeSpec,
                 min_pairs: int = 50) -> dict[str, Any]:
    centers = probe["centers"]
    mean_attn = probe["mean_attn"]            # (H, n_bins)
    pair_cnt = probe["pair_cnt"]
    dev = next(enc.bias_module.parameters()).device
    bias = enc.bias_module.evaluate(torch.from_numpy(centers).float().to(dev)).detach().cpu().numpy()

    ok = pair_cnt >= min_pairs  # only correlate where we have enough pairs
    print(f"\n=== PROBE {spec.name} [{spec.lo}, {spec.hi}] Da, {spec.n_bins} bins, "
          f"{probe['n_batches']} val batches ===")
    print(f"  (Spearman corr between bias_h(Δm) and mean attention-per-pair, "
          f"over {ok.sum()} bins with ≥{min_pairs} pairs)")
    print(f"  {'head':>4} {'spearman':>9} {'verdict':>14}")
    rows = []
    for h in range(mean_attn.shape[0]):
        if ok.sum() < 5:
            rho = float("nan")
        else:
            rho, _ = spearmanr(bias[ok, h], mean_attn[h, ok])
        verdict = ("uses bias" if rho > 0.3 else
                   "weak" if rho > 0.1 else "vestigial/ignored")
        print(f"  {h:>4} {rho:>9.2f} {verdict:>14}")
        rows.append({"head": h, "spearman": float(rho), "verdict": verdict})
    return {"range": spec.name, "min_pairs": min_pairs, "per_head": rows}


def plot_probe(enc: MSEncoder, probe: dict[str, np.ndarray], spec: ProbeSpec,
               out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = probe["centers"]
    mean_attn = probe["mean_attn"]
    dev = next(enc.bias_module.parameters()).device
    bias = enc.bias_module.evaluate(torch.from_numpy(centers).float().to(dev)).detach().cpu().numpy()
    H = mean_attn.shape[0]
    ncols = 4
    nrows = (H + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.4 * nrows), squeeze=False)
    for h in range(H):
        ax = axes[h // ncols][h % ncols]
        ax.plot(centers, mean_attn[h], color="tab:blue", lw=0.8, label="attn/pair")
        ax.set_title(f"head {h}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax2 = ax.twinx()
        ax2.plot(centers, bias[:, h], color="tab:red", lw=0.7, alpha=0.6, label="bias")
        ax2.tick_params(labelsize=6)
    for h in range(H, nrows * ncols):
        axes[h // ncols][h % ncols].axis("off")
    fig.suptitle(f"functional probe {spec.name}: mean attention/pair (blue) vs bias logit (red)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------- CLI ----------

def run_align(enc, cfg, step, args) -> list[dict[str, Any]]:
    specs = [
        RangeSpec("fine", -5.0, 5.0, 0.001, ["isotope"],
                  tol=args.fine_tol, prominence=args.prominence, min_sep=0.05),
        RangeSpec("coarse", 2.0, 200.0, 0.01, ["loss", "residue"],
                  tol=args.coarse_tol, prominence=args.prominence, min_sep=0.3),
    ]
    results = []
    for spec in specs:
        res = analyze_range(enc, spec)
        print_report(res)
        results.append(res)

    out = args.out or (Path(args.ckpt).parent / "alignment_scores.json")
    out.write_text(json.dumps({"ckpt": str(args.ckpt), "step": step, "ranges": results}, indent=2))
    print(f"\nwrote {out}")
    if args.plot:
        for spec in specs:
            fig_path = out.parent / f"alignment_{spec.name}.png"
            render_annotated(enc, spec, fig_path)
            print(f"wrote {fig_path}")

    sig = [(res["range"], ph["head"], ph["p_value"])
           for res in results for ph in res["per_head"] if ph["p_value"] < 0.05]
    if sig:
        print(f"\nALIGNMENT: {len(sig)} significant (p<0.05) head×range:")
        for rng, h, pv in sig:
            print(f"  {rng} head {h}: p={pv:.1e}")
    else:
        print("\nALIGNMENT: no head significantly above chance in either range.")
    return results


def run_probe(enc, cfg, step, args) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    loader = build_val_loader(cfg, batch_size=args.batch_size)
    specs = [ProbeSpec("fine", -5.0, 5.0, 200), ProbeSpec("coarse", -200.0, 200.0, 800)]
    out_dir = (args.out.parent if args.out else Path(args.ckpt).parent)
    results = []
    for spec in specs:
        probe = functional_probe(enc, loader, device, spec, n_batches=args.probe_batches)
        rep = report_probe(enc, probe, spec)
        results.append(rep)
        if args.plot:
            fig_path = out_dir / f"probe_{spec.name}.png"
            plot_probe(enc, probe, spec, fig_path)
            print(f"  wrote {fig_path}")
    (out_dir / "probe_scores.json").write_text(
        json.dumps({"ckpt": str(args.ckpt), "step": step, "ranges": results}, indent=2))
    print(f"wrote {out_dir / 'probe_scores.json'}")

    best = max((r["spearman"] for rep in results for r in rep["per_head"]), default=0.0)
    print(f"\nPROBE: best head Spearman(bias, attention) = {best:.2f} "
          f"→ {'bias is load-bearing' if best > 0.3 else 'bias largely vestigial'}")
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Δm bias analysis: alignment (b) and functional probe (c)")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--mode", choices=["align", "probe", "both"], default="align",
                   help="align=bias-curve chemistry alignment (b); probe=attention-vs-Δm (c)")
    p.add_argument("--out", type=Path, default=None,
                   help="JSON output path (default: <ckpt-dir>/alignment_scores.json)")
    p.add_argument("--plot", action="store_true", help="save plots next to the JSON")
    p.add_argument("--fine-tol", type=float, default=0.02, help="Da tol, isotope alignment")
    p.add_argument("--coarse-tol", type=float, default=0.1, help="Da tol, residue/loss alignment")
    p.add_argument("--prominence", type=float, default=0.3, help="min peak prominence (logits)")
    # probe-only
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64, help="probe val batch size")
    p.add_argument("--probe-batches", type=int, default=20, help="num val batches for the probe")
    args = p.parse_args(argv)

    enc, cfg, step = load_encoder(args.ckpt)
    print(f"loaded {args.ckpt}  (step {step}, {cfg['model']['n_heads']} heads, "
          f"bias scale ±{cfg['model']['delta_bias'].get('scale', 'inf')})")

    if args.mode in ("align", "both"):
        run_align(enc, cfg, step, args)
    if args.mode in ("probe", "both"):
        run_probe(enc, cfg, step, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
