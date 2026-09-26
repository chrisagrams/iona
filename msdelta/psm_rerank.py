"""Rescore database-search PSMs with the lab's features + our embedding features.

    # 1. embeddings (GPU recommended): one table per run
    python -m msdelta.rerank_psm_embed --run RUN.parquet --out rows/RUN.parquet \
        --encoder Gaolaboratory/iona-contrastive-400m --student Gaolaboratory/iona-peptide-embedder-400m
    # 2a. rescore, per run (default; Percolator-style, trains on each run itself)
    python -m msdelta.psm_rerank score --rows rows/ --labfeat features/ --out psms.parquet
    # 2b. or apply the pretrained global model (no training on the new run)
    python -m msdelta.psm_rerank score --mode global --model Gaolaboratory/iona-rerank-400m \
        --rows rows/ --labfeat features/ --out psms.parquet
    # (train a global model yourself)
    python -m msdelta.psm_rerank train --rows rows/ --labfeat features/ --out model_dir/

Inputs are stage-1 tables from msdelta.rerank_psm_embed and the lab's per-candidate feature
tables. Output is the top candidate per spectrum with its score and q-value.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from sklearn.preprocessing import StandardScaler

from msdelta.rerank_psm_fdr import (WS_FEATURES, accepted, calibrate,
                                    fit_linear, load_lab_features, qvalues,
                                    top_per_spectrum, within_spectrum)


def load_table(rows: str, labfeat: str, lab_columns=None):
    """Stage-1 rows + lab features + within-spectrum embedding features, one frame."""
    files = sorted(glob.glob(str(Path(rows) / "*.parquet"))) if Path(rows).is_dir() else [rows]
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    lf, lab = load_lab_features(labfeat, df["run_id"].unique())
    if lab_columns is not None:                   # applying a trained model
        for c in lab_columns:
            if c not in lf.columns:
                lf[c] = 0.0 if "mod_count_" in c else np.nan
        lab = tuple(lab_columns)
        lf = lf[["candidate", *lab]]
    n = len(df)
    df = df.merge(lf, on="candidate", how="inner")
    if len(df) != n:
        raise SystemExit(f"lab features cover {len(df):,} of {n:,} candidates")
    df["matched_fraction"] = df["num_matched_ions"] / df["tot_num_ions"].clip(lower=1)
    df["abs_massdiff"] = df["massdiff"].abs()
    spec = df["spectrum_id"].astype("category").cat.codes.to_numpy()
    for k, v in within_spectrum(spec, pd.to_numeric(df["cosine"], errors="coerce")
                                .fillna(0).to_numpy(float)).items():
        df[k] = v
    return df, spec, lab


def feature_matrix(df, cols, medians=None):
    X = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    med = X.median() if medians is None else pd.Series(medians, dtype=float)
    return X.fillna(med).fillna(0.0).to_numpy(float), {k: float(v) for k, v in med.items()}


def engine_score(df):
    return (df["search_neglog10_evalue"].to_numpy(float)
            - 1e-6 * df["search_rank"].to_numpy(float))


def score_perrun(df, spec, cols, folds=3, iters=10, seed=0, max_neg=0):
    """Percolator-style: each run on its own, 3-fold by spectrum, iterative labels, linear."""
    X, _ = feature_matrix(df, cols)
    decoy = df["is_decoy"].to_numpy(bool); ms = engine_score(df)
    runs = df["run_id"].to_numpy(); score = np.zeros(len(df))
    for run in sorted(set(runs)):
        rows = np.flatnonzero(runs == run)
        codes = np.unique(spec[rows])
        sfold = dict(zip(codes, np.random.default_rng(seed).permutation(len(codes)) % folds))
        rfold = np.array([sfold[c] for c in spec[rows]])
        for f in range(folds):
            tr, te = rows[rfold != f], rows[rfold == f]
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            cur, w = ms[tr], None
            for _ in range(iters):
                top = top_per_spectrum(spec[tr], cur)
                q = qvalues(cur[top], decoy[tr][top])
                pos = top[(q <= 0.01) & ~decoy[tr][top]]; neg = np.flatnonzero(decoy[tr])
                if len(pos) == 0 or len(neg) == 0:
                    break
                if max_neg and len(neg) > max_neg:
                    neg = np.random.default_rng(f).choice(neg, max_neg, replace=False)
                sel = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                w = fit_linear(Xtr[sel], y, w0=w); cur = Xtr @ w[:-1] + w[-1]
            score[te] = (calibrate(ms[tr], decoy[tr], spec[tr], ms[te]) if w is None else
                         calibrate(cur, decoy[tr], spec[tr], Xte @ w[:-1] + w[-1]))
    return score


class GlobalModel:
    """MLP (2 x 64) over the lab + embedding features; one model for all runs."""

    def __init__(self, cols, medians, mean, std, state, hidden=64):
        self.cols, self.medians, self.mean, self.std = list(cols), medians, np.asarray(mean), np.asarray(std)
        self.state, self.hidden = state, hidden

    def net(self):
        n = torch.nn.Sequential(torch.nn.Linear(len(self.cols), self.hidden), torch.nn.ReLU(),
                                torch.nn.Dropout(0.1), torch.nn.Linear(self.hidden, self.hidden),
                                torch.nn.ReLU(), torch.nn.Linear(self.hidden, 1))
        if self.state is not None:
            n.load_state_dict(self.state)
        return n

    def score(self, df):
        X, _ = feature_matrix(df, self.cols, self.medians)
        X = (X - self.mean) / np.where(self.std > 0, self.std, 1.0)
        net = self.net().eval()
        with torch.no_grad():
            return torch.cat([net(c).squeeze(-1) for c in
                              torch.tensor(X, dtype=torch.float32).split(65536)]).numpy()

    def save(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in self.state.items()}, str(out / "model.safetensors"))
        (out / "rescorer.json").write_text(json.dumps({
            "features": self.cols, "medians": self.medians, "mean": self.mean.tolist(),
            "std": self.std.tolist(), "hidden": self.hidden, "kind": "mlp-2x64"}, indent=1))

    @classmethod
    def load(cls, path: str):
        p = Path(path)
        if not p.exists():
            p = Path(snapshot_download(path))
        cfg = json.loads((p / "rescorer.json").read_text())
        return cls(cfg["features"], cfg["medians"], cfg["mean"], cfg["std"],
                   load_file(str(p / "model.safetensors")), cfg["hidden"])


def train_global(df, spec, cols, seed=0, epochs=8, max_neg=1_000_000) -> GlobalModel:
    X, med = feature_matrix(df, cols)
    decoy = df["is_decoy"].to_numpy(bool); ms = engine_score(df)
    rank1 = np.flatnonzero(df["search_rank"].to_numpy() == 1)
    q = qvalues(ms[rank1], decoy[rank1])
    pos = rank1[(q <= 0.01) & ~decoy[rank1]]; neg = np.flatnonzero(decoy)
    if len(neg) > max_neg:
        neg = np.sort(np.random.default_rng(seed).choice(neg, max_neg, replace=False))
    idx = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    mean, std = X[idx].mean(0), X[idx].std(0)
    Xs = (X[idx] - mean) / np.where(std > 0, std, 1.0)
    model = GlobalModel(cols, med, mean, std, None)
    torch.manual_seed(seed); net = model.net()
    Xt = torch.tensor(Xs, dtype=torch.float32); Yt = torch.tensor(y, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - y.sum()) / max(y.sum(), 1.0)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(Xt), generator=g).split(4096):
            opt.zero_grad(); loss_fn(net(Xt[b]).squeeze(-1), Yt[b]).backward(); opt.step()
    model.state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    return model


def summarize(df, spec, score, label):
    top = top_per_spectrum(spec, score)
    decoy = df["is_decoy"].to_numpy(bool)[top]
    r = accepted(score[top], decoy, df["peptide"].to_numpy()[top])
    q = qvalues(score[top], decoy)
    out = pd.DataFrame({"spectrum_id": df["spectrum_id"].to_numpy()[top],
                        "run_id": df["run_id"].to_numpy()[top],
                        "candidate": df["candidate"].to_numpy()[top],
                        "peptide": df["peptide"].to_numpy()[top],
                        "is_decoy": decoy, "score": score[top], "q_value": q})
    print(f"[rerank] {label}: {r['psms_1pct']:,} PSMs and {r['peptides_1pct']:,} peptides at 1% "
          f"FDR over {len(top):,} spectra", flush=True)
    return out, r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("score")
    s.add_argument("--rows", required=True); s.add_argument("--labfeat", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mode", default="perrun", choices=["perrun", "global"])
    s.add_argument("--model", default="", help="global mode: a model dir or Hub repo id")
    s.add_argument("--seed", type=int, default=0)
    t = sp.add_parser("train")
    t.add_argument("--rows", required=True); t.add_argument("--labfeat", required=True)
    t.add_argument("--out", required=True); t.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args(argv)

    if cli.cmd == "train":
        df, spec, lab = load_table(cli.rows, cli.labfeat)
        model = train_global(df, spec, lab + WS_FEATURES, seed=cli.seed)
        model.save(Path(cli.out))
        print(f"[rerank] global model on {len(set(df['run_id'])):,} runs, "
              f"{len(model.cols)} features -> {cli.out}", flush=True)
        return 0

    if cli.mode == "global":
        if not cli.model:
            ap.error("--mode global needs --model")
        model = GlobalModel.load(cli.model)
        df, spec, _ = load_table(cli.rows, cli.labfeat,
                                 lab_columns=[c for c in model.cols if c.startswith("feat__")])
        score = model.score(df)
    else:
        df, spec, lab = load_table(cli.rows, cli.labfeat)
        score = score_perrun(df, spec, lab + WS_FEATURES, seed=cli.seed)
    summarize(df, spec, engine_score(df), "MSFragger (e-value)")
    out, _ = summarize(df, spec, score, f"rescored ({cli.mode})")
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cli.out)
    print(f"[rerank] wrote {len(out):,} top PSMs to {cli.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
