"""Spectrum-retrieval precision/recall eval — the model as an embedding model.

Pools the frozen encoder into one vector per spectrum and measures how well
same-peptide replicate spectra retrieve each other (leave-one-out), against a
classic binned-spectral-cosine baseline. Ground truth = peptide_charge.

The data has ~225 genuine replicate spectra per peptide, so positives are
abundant. Reports both retrieval metrics (mAP, P@1, R@k) and the threshold-free
pairwise precision-recall (average precision), for the learned embedding and
the baseline. Also reports train/val peptide overlap (a high overlap means the
result is "retrieve seen peptides", not generalization).

Usage:
    msdelta-retrieval --ckpt runs/<run>/final.pt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score

from .data import PreprocessConfig, charge_index, precursor_mz, preprocess_spectrum, split_paths
from .probe import _pool

_BARE = re.compile(r"_\d+$")  # strip the trailing _z charge suffix


def _bare_peptide(pc: str) -> str:
    return _BARE.sub("", pc)


def _iter(paths, max_rows):
    """Consensus parquet → (peptide_charge, mz_list, int_list)."""
    n = 0
    for p in paths:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["peptide_charge", "m/z", "int"])
            pcs = t.column("peptide_charge").to_pylist()
            mzc, itc = t.column("m/z"), t.column("int")
            for i in range(len(t)):
                yield pcs[i], mzc[i].as_py(), itc[i].as_py()
                n += 1
                if n >= max_rows:
                    return


def _iter_mgf(path, max_spectra):
    """Real experimental MGF → ('SEQ_charge', mz_list, int_list).

    Labels and peaks are inline (SEQ, CHARGE, then 'mz intensity' lines), so
    no PSM join is needed. These are noisy single-scan spectra — the faithful
    real-world retrieval test.
    """
    n = 0
    seq = charge = None
    mz: list[float] = []
    it: list[float] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                seq = charge = None; mz = []; it = []
            elif line == "END IONS":
                if seq and mz:
                    yield f"{seq}_{charge or '?'}", mz, it
                    n += 1
                    if n >= max_spectra:
                        return
            elif "=" in line:
                if line.startswith("SEQ="):
                    seq = line[4:]
                elif line.startswith("CHARGE="):
                    charge = line[7:].rstrip("+")
            else:
                parts = line.split()
                if len(parts) == 2:
                    mz.append(float(parts[0])); it.append(float(parts[1]))


# ---------- extraction ----------

@torch.no_grad()
def extract(enc, spectrum_iter, device, pp, *, max_peptides=150, per_peptide=25,
            batch_size=128, bin_width=1.0, mz_max=2000.0):
    """Bucket spectra by label (cap `per_peptide` each, up to `max_peptides`),
    then encode→pool. `spectrum_iter` yields (label, mz_list, int_list).
    Returns learned embeddings, binned-cosine baseline vectors, and labels."""
    enc.to(device).eval()
    n_bins = int(mz_max / bin_width)
    buckets: dict[str, list] = {}     # plain dict — do NOT auto-create entries
    binned_b: dict[str, list] = {}

    for pc, mz_list, int_list in spectrum_iter:
        existing = buckets.get(pc)
        if existing is not None:
            if len(existing) >= per_peptide:
                continue
        elif len(buckets) >= max_peptides:
            continue  # hit the peptide cap; skip new peptides (bounds memory)
        mzt = torch.tensor(mz_list, dtype=torch.float32)
        itt = torch.tensor(int_list, dtype=torch.float32)
        mp, lp, _ = preprocess_spectrum(mzt, itt, pp)   # intensity_prob unused by retrieval
        if mp.numel() == 0:
            continue
        buckets.setdefault(pc, []).append((mp, lp))
        b = np.zeros(n_bins, dtype=np.float32)
        np.add.at(b, np.clip((mp.numpy() / bin_width).astype(int), 0, n_bins - 1), lp.numpy())
        binned_b.setdefault(pc, []).append(b)
        # Early exit once every bucket is full — avoids scanning the whole file.
        if len(buckets) >= max_peptides and all(len(v) >= per_peptide for v in buckets.values()):
            break

    # flatten, keep only peptides with >=2 reps (need positives)
    specs, binned, labels, charges, prec_mzs = [], [], [], [], []
    for pc, items in buckets.items():
        if len(items) < 2:
            continue
        specs.extend(items)
        binned.extend(binned_b[pc])
        labels.extend([pc] * len(items))
        charges.extend([charge_index(pc)] * len(items))
        prec_mzs.extend([precursor_mz(pc)] * len(items))

    # encode in batches → pooled embedding
    embs = []
    for s in range(0, len(specs), batch_size):
        chunk = specs[s:s + batch_size]
        K = max(m.numel() for m, _ in chunk)
        mz = torch.zeros(len(chunk), K); li = torch.zeros(len(chunk), K)
        mask = torch.zeros(len(chunk), K, dtype=torch.bool)
        for b, (m, l) in enumerate(chunk):
            k = m.numel(); mz[b, :k] = m; li[b, :k] = l; mask[b, :k] = True
        chg = torch.tensor(charges[s:s + batch_size], dtype=torch.long, device=device)
        pmz = torch.tensor(prec_mzs[s:s + batch_size], dtype=torch.float32, device=device)
        tok = enc(mz.to(device), li.to(device), (~mask).to(device), charge=chg, precursor_mz=pmz)
        embs.append(_pool(tok, mask.to(device)).cpu().numpy())

    return {
        "emb": np.concatenate(embs) if embs else np.zeros((0, 1)),
        "binned": np.array(binned),
        "labels": np.array(labels),
    }


# ---------- metrics ----------

def all_but_top(X: np.ndarray, k: int) -> np.ndarray:
    """Anisotropy fix (Mu & Viswanath): center, remove the top-k principal
    directions. Transformer embeddings are dominated by a few common
    directions that drown the discriminative signal; removing them recovers
    most of the retrieval gap with zero training."""
    if k <= 0:
        return X
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k]
    return Xc - (Xc @ Vk.T) @ Vk


def _cosine_sim(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return Xn @ Xn.T


def _ap_from_rel(rel: np.ndarray) -> float:
    """Average precision of a relevance vector already sorted by score (desc)."""
    pos = np.where(rel)[0]
    if len(pos) == 0:
        return np.nan
    precs = (np.arange(len(pos)) + 1) / (pos + 1)
    return float(precs.mean())


def retrieval_metrics(sim: np.ndarray, labels: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    """Leave-one-out: every spectrum queries all others."""
    N = len(labels)
    np.fill_diagonal(sim, -np.inf)  # exclude self
    aps, p1, recall = [], [], {k: [] for k in ks}
    same = labels[None, :] == labels[:, None]
    for i in range(N):
        npos = int(same[i].sum())  # excludes self (diagonal of `same` is True but sim=-inf ranks it last)
        if npos == 0:
            continue
        order = np.argsort(-sim[i])
        rel = same[i][order]
        aps.append(_ap_from_rel(rel))
        p1.append(float(rel[0]))
        for k in ks:
            recall[k].append(rel[:k].sum() / npos)
    out = {"mAP": float(np.nanmean(aps)), "P@1": float(np.mean(p1))}
    for k in ks:
        out[f"R@{k}"] = float(np.mean(recall[k]))
    return out


def pairwise_ap(sim: np.ndarray, labels: np.ndarray) -> float:
    """Threshold-free precision-recall (average precision) over all spectrum pairs."""
    iu = np.triu_indices(len(labels), k=1)
    scores = sim[iu]
    y = (labels[iu[0]] == labels[iu[1]]).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, scores))


def _eval_block(name, X, labels):
    sim = _cosine_sim(X)
    m = retrieval_metrics(sim.copy(), labels)
    m["AUC-PR"] = pairwise_ap(sim, labels)
    print(f"  {name:<18} mAP={m['mAP']:.3f}  P@1={m['P@1']:.3f}  "
          f"R@5={m['R@5']:.3f}  R@10={m['R@10']:.3f}  AUC-PR={m['AUC-PR']:.3f}")
    return m


@torch.no_grad()
def retrieval_inline_metrics(
    enc,
    val_paths,
    device,
    pp,
    *,
    max_peptides: int = 100,
    per_peptide: int = 20,
    max_scan: int = 150_000,
    whiten: int = 0,
) -> dict[str, float]:
    """Flat wandb dict from spectrum retrieval — the model as an embedding model.

    Pools the encoder per spectrum and measures leave-one-out same-peptide
    retrieval vs. the binned-cosine baseline. Smaller caps than the CLI
    (100×20 vs 150×25) keep it cheap enough to run inline at probe cadence;
    `extract` early-exits once buckets fill, so it rarely scans all max_scan
    rows. Logs the learned mAP/P@1/AUC-PR, the baseline mAP, and the gap —
    so we can watch whether the embedding catches up to binned-cosine as the
    model trains. Restores the encoder's train/eval mode on exit.
    """
    was_training = enc.training
    try:
        d = extract(enc, _iter(val_paths, max_scan), device, pp,
                    max_peptides=max_peptides, per_peptide=per_peptide)
        labels = d["labels"]
        if len(labels) < 2 or len(set(labels)) < 2:
            return {}
        emb = all_but_top(d["emb"], whiten) if whiten > 0 else d["emb"]

        sim = _cosine_sim(emb)
        m = retrieval_metrics(sim.copy(), labels)
        m["AUC-PR"] = pairwise_ap(sim, labels)

        bsim = _cosine_sim(d["binned"])
        bm = retrieval_metrics(bsim.copy(), labels)

        return {
            "retrieval/mAP": m["mAP"],
            "retrieval/P@1": m["P@1"],
            "retrieval/R@5": m["R@5"],
            "retrieval/AUC_PR": m["AUC-PR"],
            "retrieval/binned_mAP": bm["mAP"],
            "retrieval/gap_vs_binned": m["mAP"] - bm["mAP"],
        }
    finally:
        if was_training:
            enc.train()


def peptide_overlap(train_paths, val_labels, max_rows=60_000) -> float:
    val_set = set(val_labels)
    train_set = set()
    for pc, _, _ in _iter(train_paths, max_rows):
        train_set.add(pc)
    inter = val_set & train_set
    return len(inter) / max(1, len(val_set))


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    from .analyze import load_encoder

    p = argparse.ArgumentParser(description="Spectrum-retrieval precision/recall eval")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--mgf", type=Path, default=None,
                   help="real-world MGF (SEQ/CHARGE inline); default = consensus val parquet")
    p.add_argument("--max-peptides", type=int, default=150)
    p.add_argument("--per-peptide", type=int, default=25)
    p.add_argument("--max-scan", type=int, default=300_000,
                   help="max spectra to scan from the source (MGF replicates are scattered)")
    p.add_argument("--bin-width", type=float, default=1.0, help="Da, baseline binning")
    p.add_argument("--whiten", type=int, default=0, metavar="K",
                   help="all-but-top-K anisotropy fix on the learned embedding "
                        "(0=off; ~5-20 typically recovers most of the retrieval gap)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--check-overlap", action="store_true",
                   help="report train/val peptide overlap (generalization sanity)")
    args = p.parse_args(argv)

    enc, cfg, step = load_encoder(args.ckpt)
    dcfg = cfg["data"]
    pp = PreprocessConfig(intensity_threshold_frac=dcfg["intensity_threshold_frac"], top_n=dcfg["top_n"])
    train_paths, val_paths = split_paths(dcfg["root"], dcfg["n_val_files"])

    if args.mgf:
        print(f"loaded {args.ckpt} (step {step}); extracting from MGF {args.mgf} ...")
        spectrum_iter = _iter_mgf(args.mgf, args.max_scan)
    else:
        print(f"loaded {args.ckpt} (step {step}); extracting from consensus val parquet ...")
        spectrum_iter = _iter(val_paths, args.max_scan)
    d = extract(enc, spectrum_iter, torch.device(args.device), pp,
                max_peptides=args.max_peptides, per_peptide=args.per_peptide,
                bin_width=args.bin_width)
    labels = d["labels"]
    n_pep = len(set(labels))
    emb = d["emb"]
    emb_label = "learned embed"
    if args.whiten > 0:
        emb = all_but_top(emb, args.whiten)
        emb_label = f"learned (whiten-{args.whiten})"
    print(f"{len(labels)} spectra across {n_pep} peptides "
          f"(~{len(labels)//max(1,n_pep)} reps each)\n")

    print("=== strict positives (same peptide_charge) ===")
    _eval_block(emb_label, emb, labels)
    _eval_block("binned cosine", d["binned"], labels)

    bare = np.array([_bare_peptide(x) for x in labels])
    if len(set(bare)) < n_pep:  # only meaningful if charges collapse
        print("\n=== loose positives (same peptide, any charge) ===")
        _eval_block(emb_label, emb, bare)
        _eval_block("binned cosine", d["binned"], bare)

    if args.check_overlap:
        ov = peptide_overlap(train_paths, labels)
        print(f"\ntrain/val peptide_charge overlap: {ov*100:.1f}% "
              f"({'retrieving SEEN peptides — not generalization' if ov > 0.5 else 'mostly novel peptides'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
