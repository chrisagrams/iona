"""Measure information in frozen encoder representations."""

from __future__ import annotations

import itertools
import re

import numpy as np
import torch
from datasets import Dataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence

from iona.chemistry import (
    ISOTOPES,
    NEUTRAL_LOSSES,
    PROTON_MASS,
    RESIDUE_MASSES,
    WATER_MASS,
)
from iona.embedding import pool_tokens
from iona.inference import PredictionTrainer, collate_spectra
from iona.modeling_iona import IonaModel

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


def _collate_probe(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Pad spectra and the sampled peak and pair indices of a batch."""
    batch = collate_spectra(rows)
    for key in ("peak_index", "pair_i", "pair_j"):
        batch[key] = pad_sequence(
            [torch.as_tensor(r[key], dtype=torch.long) for r in rows], batch_first=True
        )
    return batch


def _predict_probe(model, inputs):
    """Pool every spectrum and gather the tokens of its sampled peaks and pairs."""
    mask = inputs["attention_mask"]
    tokens = model(
        mz=inputs["mz"], log_intensity=inputs["log_intensity"], attention_mask=mask
    ).last_hidden_state

    def gather(index):
        return tokens.gather(1, index[..., None].expand(-1, -1, tokens.shape[-1])).float()

    return {
        "pooled": pool_tokens(tokens, mask).float(),
        "peaks": gather(inputs["peak_index"]),
        "pair_i": gather(inputs["pair_i"]),
        "pair_j": gather(inputs["pair_j"]),
    }


def extract_representations(
    enc: IonaModel,
    dataset,
    n_spectra: int,
    batch_size: int,
    device: torch.device,
    max_peak_samples: int = 80_000,
    max_pair_samples: int = 80_000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Extract spectrum, peak, and peak-pair representations.

    Every process of a distributed run must call this: encoding is sharded across ranks.
    """
    was_training = enc.training
    enc.to(device).eval()
    rng = np.random.default_rng(seed)

    prec, pcount, logtic, charge, maxmz = [], [], [], [], []
    peak_iso, peak_mz, pair_loss = [], [], []
    loss_vals = np.array(list(_LOSSES.values()))

    # Sample peaks and pairs in dataset order before encoding, which runs length-sorted: the
    # global sample caps would otherwise fill with the longest spectra first.
    rows = []
    for row in itertools.islice(dataset, n_spectra):
        mzb = np.asarray(row["mz"], dtype=np.float32)
        k = len(mzb)
        if k == 0:
            continue
        pc = row["peptide_charge"]
        z = parse_charge(pc)
        z = z if (z and 1 <= z <= 5) else 0
        prec.append(precursor_mz(pc) or np.nan)
        pcount.append(k)
        logtic.append(float(row["log_tic"]))
        charge.append(z)
        maxmz.append(float(mzb.max()))
        peak_index, pair_i, pair_j = [], [], []
        if len(peak_iso) < max_peak_samples and z:
            iso = _isotope_labels(np.argsort(mzb), mzb, z)
            take = min(k, max(1, max_peak_samples // n_spectra * 4))
            for s in rng.choice(k, size=min(take, k), replace=False):
                peak_index.append(int(s))
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
                    pair_i.append(int(ii[q]))
                    pair_j.append(int(jj[q]))
                    pair_loss.append(int(pos[q]))
        rows.append({
            "mz": row["mz"],
            "log_intensity": row["log_intensity"],
            "peak_index": peak_index,
            "pair_i": pair_i,
            "pair_j": pair_j,
        })

    spec, peak_rep, pair_rep = [], [], []
    if rows:
        trainer = PredictionTrainer(
            enc, _predict_probe, data_collator=_collate_probe, batch_size=batch_size,
            device=device,
        )
        out = trainer.predict_sorted(Dataset.from_list(rows))
        for i, row in enumerate(rows):
            n_peaks, n_pairs = len(row["peak_index"]), len(row["pair_i"])
            spec.append(out["pooled"][i])
            peak_rep.extend(out["peaks"][i, :n_peaks])
            pair_rep.extend(np.concatenate(
                [out["pair_i"][i, :n_pairs], out["pair_j"][i, :n_pairs]], axis=1
            ))

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


def probe_metrics(d: dict[str, np.ndarray]) -> dict[str, float]:
    """Fit all linear probes on ``extract_representations`` output and return metrics."""
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
