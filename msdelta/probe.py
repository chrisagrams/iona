"""Frozen-encoder linear probes — the validation suite (plan §6.1 + chemistry).

Two tiers of linear probes on frozen, pooled encoder representations:

  Tier 1 (sanity — must pass or nothing else matters):
    - precursor m/z  (regression; label computed from peptide_charge)
    - peak count     (regression; ground truth from the spectrum)
    - log TIC        (regression; raw total ion current)

  Tier 2 (chemistry — the real test of the Δm bias):
    - charge state           (classification 1–5; label from peptide_charge)
    - neutral-loss pair      (binary; does Δm_ij match a known loss?)
    - isotope-peak position  (per-peak M+0/1/2/3; heuristic from Δm = 1.003/z)

Designed to run two ways from ONE implementation:
  - standalone:  `msdelta-probe --ckpt runs/<run>/final.pt`
  - inline:      train.py calls `run_all_probes(encoder, ...)` every N steps
                 and logs the returned flat dict to wandb.

Tier-2 labels for neutral-loss / isotope are *derived from Δm rules*, so they
test "is this info linearly decodable from the representation," not pure
discovery. Charge / precursor / count / TIC are ground-truth.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import PreprocessConfig, preprocess_spectrum, split_paths
from .model import MSEncoder

# Monoisotopic residue masses (Da); L and I share a mass.
_RES = {
    "G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276, "V": 99.06841,
    "T": 101.04768, "C": 103.00919, "L": 113.08406, "I": 113.08406, "N": 114.04293,
    "D": 115.02694, "Q": 128.05858, "K": 128.09496, "E": 129.04259, "M": 131.04049,
    "H": 137.05891, "F": 147.06841, "R": 156.10111, "Y": 163.06333, "W": 186.07931,
}
_WATER = 18.0105646
_PROTON = 1.0072765
_C13 = 1.0033548
_LOSSES = {"H2O": 18.0105646, "NH3": 17.0265491, "CO": 27.9949146, "CO2": 43.9898292}
_STD_PEP = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")


def parse_charge(pc: str) -> int | None:
    parts = pc.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def precursor_mz(pc: str) -> float | None:
    """Compute precursor m/z from 'PEPTIDE_z'; None if peptide has mods/non-std."""
    parts = pc.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    pep, z = parts[0], int(parts[1])
    if z < 1 or not _STD_PEP.fullmatch(pep):
        return None
    mass = sum(_RES[a] for a in pep) + _WATER
    return (mass + z * _PROTON) / z


# ---------- representation extraction ----------

def _iter_spectra(paths, n_spectra):
    count = 0
    for p in paths:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["peptide_charge", "m/z", "int"])
            pcs = tbl.column("peptide_charge").to_pylist()
            mzc, intc = tbl.column("m/z"), tbl.column("int")
            for i in range(len(tbl)):
                yield pcs[i], mzc[i].as_py(), intc[i].as_py()
                count += 1
                if count >= n_spectra:
                    return


def _pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mean ⊕ max over real peaks. tokens (B,K,D), mask (B,K) True=real."""
    m = mask.unsqueeze(-1)
    summed = (tokens * m).sum(1)
    mean = summed / m.sum(1).clamp_min(1)
    mx = tokens.masked_fill(~m, float("-inf")).max(1).values
    mx = torch.nan_to_num(mx, neginf=0.0)
    return torch.cat([mean, mx], dim=-1)


def _isotope_labels(mz_sorted_idx, mz: np.ndarray, z: int, tol: float = 0.01) -> np.ndarray:
    """Per-peak M+k position: # consecutive isotope steps (1.003/z) present below."""
    step = _C13 / max(z, 1)
    out = np.zeros(len(mz), dtype=np.int64)
    mzs = mz[mz_sorted_idx]
    for orig in mz_sorted_idx:
        k, target = 0, mz[orig] - step
        while k < 3:
            j = np.searchsorted(mzs, target)
            hit = (j < len(mzs) and abs(mzs[j] - target) < tol) or \
                  (j > 0 and abs(mzs[j - 1] - target) < tol)
            if not hit:
                break
            k += 1
            target -= step
        out[orig] = k
    return out


@torch.no_grad()
def extract_representations(
    enc: MSEncoder,
    paths: list[Path],
    n_spectra: int,
    batch_size: int,
    device: torch.device,
    pp: PreprocessConfig,
    max_peak_samples: int = 80_000,
    max_pair_samples: int = 80_000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    was_training = enc.training
    enc.to(device).eval()
    rng = np.random.default_rng(seed)

    spec, prec, pcount, logtic, charge, maxmz = [], [], [], [], [], []
    peak_rep, peak_iso, peak_mz = [], [], []
    pair_rep, pair_loss = [], []
    loss_vals = np.array(list(_LOSSES.values()))

    buf: list[tuple] = []  # (mz_p, li_p, meta)

    def flush():
        if not buf:
            return
        K = max(t[0].numel() for t in buf)
        B = len(buf)
        mz = torch.zeros(B, K); li = torch.zeros(B, K)
        mask = torch.zeros(B, K, dtype=torch.bool)
        chg = torch.zeros(B, dtype=torch.long)
        for b, (m, l, meta) in enumerate(buf):
            k = m.numel()
            mz[b, :k] = m; li[b, :k] = l; mask[b, :k] = True
            chg[b] = meta["z"]
        tokens = enc(mz.to(device), li.to(device), (~mask).to(device), charge=chg.to(device))  # (B,K,D)
        pooled = _pool(tokens, mask.to(device)).cpu().numpy()
        tok_cpu = tokens.cpu().numpy()
        for b, (m, l, meta) in enumerate(buf):
            k = m.numel()
            spec.append(pooled[b]); prec.append(meta["prec"]); pcount.append(k)
            logtic.append(meta["logtic"]); charge.append(meta["z"]); maxmz.append(float(m.max()))
            mzb = m.numpy()
            # per-peak samples: isotope label + the peak's OWN m/z. Tokens are
            # m/z-free, so recovering peak m/z from the token measures whether the
            # Δm bias re-injected the fragment m/z pattern into the representation.
            if len(peak_iso) < max_peak_samples and meta["z"]:
                order = np.argsort(mzb)
                iso = _isotope_labels(order, mzb, meta["z"])
                take = min(k, max(1, max_peak_samples // n_spectra * 4))
                sel = rng.choice(k, size=min(take, k), replace=False)
                for s in sel:
                    peak_rep.append(tok_cpu[b, s]); peak_iso.append(int(iso[s]))
                    peak_mz.append(float(mzb[s]))
            # neutral-loss pair samples (balanced per spectrum)
            if len(pair_loss) < max_pair_samples and k >= 4:
                dm = mzb[:, None] - mzb[None, :]
                ii, jj = np.where(dm > 0)
                d = dm[ii, jj]
                pos = np.abs(d[:, None] - loss_vals).min(1) < 0.01
                pidx = np.where(pos)[0]; nidx = np.where(~pos)[0]
                npos = min(len(pidx), 4)
                if npos:
                    psel = rng.choice(pidx, npos, replace=False)
                    nsel = rng.choice(nidx, min(npos, len(nidx)), replace=False)
                    for q in np.concatenate([psel, nsel]):
                        r = np.concatenate([tok_cpu[b, ii[q]], tok_cpu[b, jj[q]]])
                        pair_rep.append(r); pair_loss.append(int(pos[q]))
        buf.clear()

    for pc, mz_list, int_list in _iter_spectra(paths, n_spectra):
        mzt = torch.tensor(mz_list, dtype=torch.float32)
        it = torch.tensor(int_list, dtype=torch.float32)
        mz_p, li_p = preprocess_spectrum(mzt, it, pp)
        if mz_p.numel() == 0:
            continue
        z = parse_charge(pc)
        meta = {"prec": precursor_mz(pc) or np.nan,
                "logtic": float(torch.log1p(it.sum())),
                "z": z if (z and 1 <= z <= 5) else 0}
        buf.append((mz_p, li_p, meta))
        if len(buf) >= batch_size:
            flush()
    flush()

    if was_training:
        enc.train()
    return {
        "spec": np.array(spec), "prec": np.array(prec), "pcount": np.array(pcount, float),
        "logtic": np.array(logtic), "charge": np.array(charge), "maxmz": np.array(maxmz),
        "peak_rep": np.array(peak_rep) if peak_rep else np.zeros((0, 1)),
        "peak_iso": np.array(peak_iso),
        "peak_mz": np.array(peak_mz),
        "pair_rep": np.array(pair_rep) if pair_rep else np.zeros((0, 1)),
        "pair_loss": np.array(pair_loss),
    }


# ---------- probe runners ----------

def _split(n, frac=0.7, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * frac)
    return idx[:cut], idx[cut:]


def _regression(X, y, name, baseline_pred=None) -> dict[str, float]:
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < 50:
        return {f"probe/{name}_r2": float("nan")}
    tr, te = _split(len(y))
    sc = StandardScaler().fit(X[tr])
    m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
    pred = m.predict(sc.transform(X[te]))
    out = {f"probe/{name}_r2": float(r2_score(y[te], pred)),
           f"probe/{name}_mae": float(np.mean(np.abs(pred - y[te])))}
    if baseline_pred is not None:
        bp = baseline_pred[ok][te]
        out[f"probe/{name}_mae_baseline"] = float(np.mean(np.abs(bp - y[te])))
    return out


def _classification(X, y, name) -> dict[str, float]:
    if len(y) < 50 or len(np.unique(y)) < 2:
        return {f"probe/{name}_acc": float("nan")}
    tr, te = _split(len(y))
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    pred = m.predict(sc.transform(X[te]))
    maj = np.bincount(y[tr].astype(int)).argmax()
    out = {f"probe/{name}_acc": float(accuracy_score(y[te], pred)),
           f"probe/{name}_f1": float(f1_score(y[te], pred, average="macro")),
           f"probe/{name}_acc_baseline": float(np.mean(y[te] == maj))}
    if len(np.unique(y)) == 2:
        try:
            proba = m.predict_proba(sc.transform(X[te]))[:, 1]
            out[f"probe/{name}_auc"] = float(roc_auc_score(y[te], proba))
        except ValueError:
            pass
    return out


def run_all_probes(
    enc: MSEncoder,
    val_paths: list[Path],
    device: torch.device,
    pp: PreprocessConfig,
    n_spectra: int = 4000,
    batch_size: int = 128,
) -> dict[str, float]:
    """Flat metrics dict, suitable for wandb.log. Frozen-encoder linear probes."""
    d = extract_representations(enc, val_paths, n_spectra, batch_size, device, pp)
    X = d["spec"]
    metrics: dict[str, float] = {}
    # Tier 1
    metrics.update(_regression(X, d["prec"], "precursor_mz", baseline_pred=d["maxmz"]))
    metrics.update(_regression(X, d["pcount"], "peak_count"))
    metrics.update(_regression(X, d["logtic"], "log_tic"))
    # Tier 2
    metrics.update(_classification(X, d["charge"].astype(int), "charge"))
    if len(d["pair_loss"]) >= 50:
        metrics.update(_classification(d["pair_rep"], d["pair_loss"], "neutral_loss"))
    if len(d["peak_iso"]) >= 50:
        metrics.update(_classification(d["peak_rep"], d["peak_iso"], "isotope"))
    # Fragment-m/z pattern: recover each peak's OWN m/z from its (m/z-free) token.
    # High R² ⇒ the Δm bias re-injected fragment m/z into the representation —
    # the signal retrieval needs. This is the direct "did it learn the fragment
    # pattern" monitor for the m/z-free architecture.
    if len(d["peak_mz"]) >= 50:
        metrics.update(_regression(d["peak_rep"], d["peak_mz"], "fragment_mz"))
    return metrics


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    from .analyze import load_encoder  # reuse the loader

    p = argparse.ArgumentParser(description="Frozen-encoder linear probe suite")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--n-spectra", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    enc, cfg, step = load_encoder(args.ckpt)
    dcfg = cfg["data"]
    pp = PreprocessConfig(intensity_threshold_frac=dcfg["intensity_threshold_frac"], top_n=dcfg["top_n"])
    _, val_paths = split_paths(dcfg["root"], dcfg["n_val_files"])
    print(f"loaded {args.ckpt} (step {step}); probing on {args.n_spectra} val spectra...")

    m = run_all_probes(enc, val_paths, torch.device(args.device), pp,
                       n_spectra=args.n_spectra, batch_size=args.batch_size)

    def line(name, *keys):
        parts = [f"{k.split('/')[-1]}={m[k]:.3f}" for k in keys if k in m and np.isfinite(m[k])]
        print(f"  {name:<16} " + "  ".join(parts))

    print("\n=== TIER 1 (sanity) ===")
    line("precursor m/z", "probe/precursor_mz_mae", "probe/precursor_mz_mae_baseline", "probe/precursor_mz_r2")
    line("peak count", "probe/peak_count_r2", "probe/peak_count_mae")
    line("log TIC", "probe/log_tic_r2", "probe/log_tic_mae")
    print("\n=== TIER 2 (chemistry) ===")
    line("charge", "probe/charge_acc", "probe/charge_acc_baseline", "probe/charge_f1")
    line("neutral loss", "probe/neutral_loss_auc", "probe/neutral_loss_acc", "probe/neutral_loss_acc_baseline")
    line("isotope M+k", "probe/isotope_acc", "probe/isotope_acc_baseline", "probe/isotope_f1")
    line("fragment m/z", "probe/fragment_mz_r2", "probe/fragment_mz_mae")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
