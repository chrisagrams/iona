"""Measure information in frozen encoder representations."""

from __future__ import annotations

import itertools
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from msdelta.data.chemistry import (
    ISOTOPES,
    NEUTRAL_LOSSES,
    PROTON_MASS,
    RESIDUE_MASSES,
    WATER_MASS,
)
from msdelta.models.embedding import encode_batch, pool_tokens
from msdelta.models.modeling_msdelta import MSDeltaModel

_C13 = ISOTOPES["¹³C"]
_LOSSES = {name: NEUTRAL_LOSSES[name] for name in ("H₂O", "NH₃", "CO", "CO₂")}
_STD_PEP = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")


def parse_charge(pc: str) -> int | None:
    parts = pc.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def precursor_mz(pc: str) -> float | None:
    """Calculate precursor m/z for a standard peptide."""
    parts = pc.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    pep, z = parts[0], int(parts[1])
    if z < 1 or not _STD_PEP.fullmatch(pep):
        return None
    mass = sum(RESIDUE_MASSES[a] for a in pep) + WATER_MASS
    return (mass + z * PROTON_MASS) / z


def _isotope_labels(mz_sorted_idx, mz: np.ndarray, z: int, tol: float = 0.01) -> np.ndarray:
    """Count consecutive isotope steps below each peak."""
    step = _C13 / max(z, 1)
    out = np.zeros(len(mz), dtype=np.int64)
    mzs = mz[mz_sorted_idx]
    for orig in mz_sorted_idx:
        k, target = 0, mz[orig] - step
        while k < 3:
            j = np.searchsorted(mzs, target)
            hit = (j < len(mzs) and abs(mzs[j] - target) < tol) or (
                j > 0 and abs(mzs[j - 1] - target) < tol
            )
            if not hit:
                break
            k += 1
            target -= step
        out[orig] = k
    return out


@torch.no_grad()
def extract_representations(
    enc: MSDeltaModel,
    dataset,
    n_spectra: int,
    batch_size: int,
    device: torch.device,
    max_peak_samples: int = 80_000,
    max_pair_samples: int = 80_000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Extract spectrum, peak, and peak-pair representations."""
    was_training = enc.training
    enc.to(device).eval()
    rng = np.random.default_rng(seed)

    spec, prec, pcount, logtic, charge, maxmz = [], [], [], [], [], []
    peak_rep, peak_iso, peak_mz = [], [], []
    pair_rep, pair_loss = [], []
    loss_vals = np.array(list(_LOSSES.values()))

    buf: list[tuple] = []

    def flush():
        if not buf:
            return
        tokens, mask = encode_batch(enc, [t[0] for t in buf], [t[1] for t in buf], device)
        pooled = pool_tokens(tokens, mask).float().cpu().numpy()
        tok_cpu = tokens.float().cpu().numpy()
        for b, (m, _log_intensity, meta) in enumerate(buf):
            k = m.numel()
            spec.append(pooled[b])
            prec.append(meta["prec"])
            pcount.append(k)
            logtic.append(meta["logtic"])
            charge.append(meta["z"])
            maxmz.append(float(m.max()))
            mzb = m.numpy()
            if len(peak_iso) < max_peak_samples and meta["z"]:
                order = np.argsort(mzb)
                iso = _isotope_labels(order, mzb, meta["z"])
                take = min(k, max(1, max_peak_samples // n_spectra * 4))
                sel = rng.choice(k, size=min(take, k), replace=False)
                for s in sel:
                    peak_rep.append(tok_cpu[b, s])
                    peak_iso.append(int(iso[s]))
                    peak_mz.append(float(mzb[s]))
            if len(pair_loss) < max_pair_samples and k >= 4:
                dm = mzb[:, None] - mzb[None, :]
                ii, jj = np.where(dm > 0)
                d = dm[ii, jj]
                pos = np.abs(d[:, None] - loss_vals).min(1) < 0.01
                pidx = np.where(pos)[0]
                nidx = np.where(~pos)[0]
                npos = min(len(pidx), 4)
                if npos:
                    psel = rng.choice(pidx, npos, replace=False)
                    nsel = rng.choice(nidx, min(npos, len(nidx)), replace=False)
                    for q in np.concatenate([psel, nsel]):
                        r = np.concatenate([tok_cpu[b, ii[q]], tok_cpu[b, jj[q]]])
                        pair_rep.append(r)
                        pair_loss.append(int(pos[q]))
        buf.clear()

    for row in itertools.islice(dataset, n_spectra):
        mz_p = torch.tensor(row["mz"], dtype=torch.float32)
        li_p = torch.tensor(row["log_intensity"], dtype=torch.float32)
        if mz_p.numel() == 0:
            continue
        pc = row["peptide_charge"]
        z = parse_charge(pc)
        meta = {
            "prec": precursor_mz(pc) or np.nan,
            "logtic": float(row["log_tic"]),
            "z": z if (z and 1 <= z <= 5) else 0,
        }
        buf.append((mz_p, li_p, meta))
        if len(buf) >= batch_size:
            flush()
    flush()

    if was_training:
        enc.train()
    return {
        "spec": np.array(spec),
        "prec": np.array(prec),
        "pcount": np.array(pcount, float),
        "logtic": np.array(logtic),
        "charge": np.array(charge),
        "maxmz": np.array(maxmz),
        "peak_rep": np.array(peak_rep) if peak_rep else np.zeros((0, 1)),
        "peak_iso": np.array(peak_iso),
        "peak_mz": np.array(peak_mz),
        "pair_rep": np.array(pair_rep) if pair_rep else np.zeros((0, 1)),
        "pair_loss": np.array(pair_loss),
    }


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
    out = {
        f"probe/{name}_r2": float(r2_score(y[te], pred)),
        f"probe/{name}_mae": float(np.mean(np.abs(pred - y[te]))),
    }
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
    out = {
        f"probe/{name}_acc": float(accuracy_score(y[te], pred)),
        f"probe/{name}_f1": float(f1_score(y[te], pred, average="macro")),
        f"probe/{name}_acc_baseline": float(np.mean(y[te] == maj)),
    }
    if len(np.unique(y)) == 2:
        try:
            proba = m.predict_proba(sc.transform(X[te]))[:, 1]
            out[f"probe/{name}_auc"] = float(roc_auc_score(y[te], proba))
        except ValueError:
            pass
    return out


def run_all_probes(
    enc: MSDeltaModel,
    dataset,
    device: torch.device,
    n_spectra: int = 4000,
    batch_size: int = 128,
) -> dict[str, float]:
    """Run all linear probes and return metrics."""
    d = extract_representations(enc, dataset, n_spectra, batch_size, device)
    X = d["spec"]
    metrics: dict[str, float] = {}
    metrics.update(_regression(X, d["prec"], "precursor_mz", baseline_pred=d["maxmz"]))
    metrics.update(_regression(X, d["pcount"], "peak_count"))
    metrics.update(_regression(X, d["logtic"], "log_tic"))
    metrics.update(_classification(X, d["charge"].astype(int), "charge"))
    if len(d["pair_loss"]) >= 50:
        metrics.update(_classification(d["pair_rep"], d["pair_loss"], "neutral_loss"))
    if len(d["peak_iso"]) >= 50:
        metrics.update(_classification(d["peak_rep"], d["peak_iso"], "isotope"))
    if len(d["peak_mz"]) >= 50:
        metrics.update(_regression(d["peak_rep"], d["peak_mz"], "fragment_mz"))
    return metrics
