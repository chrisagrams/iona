"""Sweep the MS2 peptide-replicate-retrieval benchmark across all training
checkpoints in a run and plot Hit@1 / MAP / PairF1 vs training step — does the
embedding improve as the unsupervised pretraining proceeds?

Loads the benchmark spectra once, then for each checkpoint encodes + scores
(raw and all-but-top-16 whitened). Writes a JSON of results and a PNG.

    .venv/bin/python pbs/replicate_retrieval_sweep.py --run runs/v14_cap_XL_spark
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from msdelta.analyze import load_encoder
from msdelta.replicate_retrieval import (
    PreprocessConfig, embed_model, evaluate, load_spectra)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/v14_cap_XL_spark"))
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/ms2-peptide-replicate-retrieval"))
    ap.add_argument("--whiten", type=int, default=16)
    ap.add_argument("--kmeans-seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", type=Path,
                    default=Path("runs/v14_cap_XL_spark/replicate_retrieval_sweep"))
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpts = sorted(glob.glob(str(args.run / "step*.pt")))
    if not ckpts:
        raise SystemExit(f"no step*.pt under {args.run}")
    seeds = tuple(range(args.kmeans_seeds))
    device = torch.device(args.device)

    paths = [Path(p) for p in sorted(glob.glob(str(args.data_dir / "*.parquet")))]
    print(f"loading {len(paths)} parquet file(s) ...")
    mz_list, int_list, y, charges, precursors = load_spectra(paths)
    print(f"{len(y)} spectra, {int(y.max()) + 1} labels\n")

    rows = []
    for ck in ckpts:
        step = int(re.search(r"step0*(\d+)\.pt", ck).group(1))
        enc, cfg, _ = load_encoder(ck)
        dcfg = cfg["data"]
        pp = PreprocessConfig(intensity_threshold_frac=dcfg["intensity_threshold_frac"],
                              top_n=dcfg["top_n"])
        emb = embed_model(enc, mz_list, int_list, charges, precursors, device, pp,
                          batch_size=args.batch_size)
        print(f"step {step}:", flush=True)
        raw = evaluate(emb, y, kmeans_seeds=seeds, name="raw", device=device)
        wh = evaluate(emb, y, kmeans_seeds=seeds, whiten=args.whiten,
                      name=f"whiten-{args.whiten}", device=device)
        rows.append({"step": step, "raw": raw, "whiten": wh})
        del enc, emb
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda r: r["step"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {json_path}")

    steps = [r["step"] for r in rows]
    metrics = ["Hit@1", "MAP", "PairF1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, met in zip(axes, metrics):
        ax.plot(steps, [r["raw"][met] for r in rows], "o-", label="raw")
        ax.plot(steps, [r["whiten"][met] for r in rows], "s-",
                label=f"whiten-{args.whiten}")
        ax.set_title(met)
        ax.set_xlabel("training step")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("msdelta — MS2 peptide-replicate-retrieval benchmark vs training step")
    fig.tight_layout()
    png_path = args.out.with_suffix(".png")
    fig.savefig(png_path, dpi=130)
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
