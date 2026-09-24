"""Score saved contrastive encoders on ms-contrastive-100k's held-out split. No training.

    # once, on a compute node: flatten + preprocess the split to disk
    python -m msdelta.eval_grouped_retrieval prepare --out-data DIR --processor CKPT
    # then one process per tile, each taking every num_shards-th model
    python -m msdelta.eval_grouped_retrieval score --data DIR --models FILE \
        --shard I --num-shards N --out-dir DIR

THE QUESTION. Every contrastive number so far comes from ms2-peptide-replicate-retrieval's
99-group held-out split, too small to separate 0.86 from 0.87. This corpus holds out
~9,000 peptides the models never saw, so it measures (a) whether embeddings trained on
~855 peptides generalise, and (b) scale and recipe differences with real power.

The split is prepared WITH consensus spectra; every model is embedded once and scored
twice from that pass, `all` (consensus + 3 replicates, R=3) and `experimental` (the three
replicates only, R=2), so the two variants are over identical embeddings. Peptides in the
replicate corpus (what the models trained on) are excluded -- see grouped_retrieval.

--models is a text file, one `name path [pooling]` per line; pooling defaults to mean+max,
which every contrastive run in this project used (RUN.md records it).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def prepare(cli) -> int:
    from datasets import load_dataset

    from msdelta.grouped_retrieval import (build_grouped_split,
                                           replicate_corpus_peptides)
    from msdelta.processing_msdelta import MSDeltaProcessor

    processor = MSDeltaProcessor.from_pretrained(cli.processor, max_peaks=cli.max_peaks)
    exclude = replicate_corpus_peptides() if cli.exclude_replicate else set()
    print(f"[prepare] excluding {len(exclude):,} replicate-corpus peptides", flush=True)
    raw = load_dataset(cli.repo)[cli.split]
    if cli.max_analytes and len(raw) > cli.max_analytes:     # e.g. a train sample for PCA
        raw = raw.shuffle(seed=0).select(range(cli.max_analytes))
    rows = build_grouped_split(raw, processor, include_consensus=True,
                               exclude_peptides=exclude, num_proc=cli.num_proc)
    if Path(cli.out_data).exists():
        raise SystemExit(f"{cli.out_data} exists; refusing to overwrite")
    rows.save_to_disk(cli.out_data)
    Path(cli.out_data, "PREPARED.json").write_text(json.dumps({
        "repo": cli.repo, "split": cli.split, "processor": cli.processor,
        "max_peaks": cli.max_peaks, "excluded_peptides": len(exclude),
        "rows": len(rows)}, indent=1))
    print(f"[prepare] wrote {len(rows):,} spectra to {cli.out_data}", flush=True)
    return 0


def binned_embeddings(rows, width: float, max_mz: float = 2000.0) -> torch.Tensor:
    """The classic spectral-library dot product as an 'embedding': peak weights summed
    into fixed m/z bins, L2-normalised by the metric. No learning, no precursor filter.

    Rows carry PROCESSED spectra: raw m/z, but log_intensity = log1p(I) / max log1p(I)
    (MSDeltaProcessor._process_one) -- raw intensity is not recoverable from it. The
    weight is therefore log1p(I); cosine ignores the per-spectrum scale. Log (like the
    more usual sqrt) damps the base peak so a few ions do not decide the match."""
    n_bins = int(np.ceil(max_mz / width))
    out = torch.zeros(len(rows), n_bins)
    for i, (mz, li) in enumerate(zip(rows["mz"], rows["log_intensity"])):
        mz = np.asarray(mz, dtype=np.float64)
        inten = np.clip(np.asarray(li, dtype=np.float64), 0, None)
        keep = (mz >= 0) & (mz < max_mz)
        idx = torch.from_numpy((mz[keep] / width).astype(np.int64))
        out[i].index_add_(0, idx, torch.from_numpy(inten[keep]).float())
    return out


def score_model(path, pooling, rows, groups, experimental, collator, device,
                batch_size) -> dict:
    from msdelta.contrastive import (MSDeltaForContrastive, embed_dataset,
                                     retrieval_metrics_topk)
    from msdelta.modeling_msdelta import MSDeltaForPreTraining

    if path.startswith("binned:"):
        emb = binned_embeddings(rows, float(path.split(":", 1)[1]))
        return _variants(emb, groups, experimental, device, retrieval_metrics_topk)
    if path.startswith("pca:"):
        # pca:<bin width>:<dims>:<prepared TRAIN sample dir> -- PCA fitted on train
        # spectra only, test spectra projected; cosine retrieval in the PCA space.
        from datasets import load_from_disk
        _, width, dims, fit_dir = path.split(":", 3)
        fit = binned_embeddings(load_from_disk(fit_dir), float(width))
        fit = torch.nn.functional.normalize(fit, dim=-1)
        mean = fit.mean(0, keepdim=True)
        _, _, v = torch.pca_lowrank(fit - mean, q=int(dims), center=False, niter=4)
        x = torch.nn.functional.normalize(binned_embeddings(rows, float(width)), dim=-1)
        emb = (x - mean) @ v[:, :int(dims)]
        return _variants(emb, groups, experimental, device, retrieval_metrics_topk)
    encoder = MSDeltaForPreTraining.from_pretrained(path)
    # A projection head saved by finetune_contrastive (--projection_dim) is scored both
    # ways: `all/...` from the head output (the loss space) and `pooled_all/...` from the
    # pre-head features. Without a head the two are the same vector, scored once.
    head_file = Path(path) / "projection_head.pt"
    head = torch.load(head_file, map_location="cpu") if head_file.exists() else None
    model = MSDeltaForContrastive(
        encoder, None, pooling=pooling, kl_weight=0,
        projection_hidden=head["projection_hidden"] if head else 0,
        projection_dim=head["projection_dim"] if head else 0,
        projection_dropout=head["projection_dropout"] if head else 0.1)
    if head:
        model.projection.load_state_dict(head["state_dict"])
    model = model.to(device)
    out = {}
    try:
        readouts = ("head", "pooled") if head else ("head",)
        for readout in readouts:
            model.readout = readout
            emb, _ = embed_dataset(model, rows, collator, device, max_rows=len(rows),
                                   batch_size=batch_size)
            prefix = "pooled_" if readout == "pooled" else ""
            out |= {prefix + k: v for k, v in
                    _variants(emb, groups, experimental, device,
                              retrieval_metrics_topk).items()}
    finally:
        del model, encoder
        if device.type == "xpu":
            torch.xpu.empty_cache()
    return out


def _variants(emb, groups, experimental, device, metric) -> dict:
    """`all` (consensus + replicates) and `experimental` (replicates only), same pass."""
    out = {}
    for variant, mask in (("all", np.ones(len(groups), dtype=bool)),
                          ("experimental", experimental)):
        metrics = metric(emb[torch.from_numpy(mask)], groups[mask], device=device)
        out |= {f"{variant}/{k}": v for k, v in metrics.items()}
    return out


def score(cli) -> int:
    from datasets import load_from_disk

    from msdelta.finetune_contrastive import ContrastiveCollator
    from msdelta.grouped_retrieval import group_ids

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    models = []
    for line in Path(cli.models).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            parts = line.split()
            models.append((parts[0], parts[1], parts[2] if len(parts) > 2 else "mean+max"))
    mine = models[cli.shard::cli.num_shards]
    print(f"[score] shard {cli.shard}/{cli.num_shards}: {len(mine)} of {len(models)} "
          f"models on {device}", flush=True)

    rows = load_from_disk(cli.data)
    groups = group_ids(rows)
    experimental = np.array([s == "experimental" for s in rows["source"]])
    collator = ContrastiveCollator(max_peptide_length=64, pad_spectra_to=cli.max_peaks)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path, pooling in mine:
        target = out_dir / f"{name}.json"
        if target.exists():
            print(f"  {name}: already scored, skipping", flush=True)
            continue
        t0 = time.time()
        metrics = score_model(path, pooling, rows, groups, experimental, collator,
                              device, cli.batch_size)
        target.write_text(json.dumps({"name": name, "path": path, "pooling": pooling,
                                      "data": cli.data, "metrics": metrics}, indent=1))
        print(f"  {name}: all MAP@R {metrics.get('all/MAP@R', float('nan')):.4f}  "
              f"experimental MAP@R {metrics.get('experimental/MAP@R', float('nan')):.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out-data", required=True)
    p.add_argument("--processor", required=True, help="any pretrained checkpoint dir")
    p.add_argument("--repo", default="chrisagrams/ms-contrastive-100k")
    p.add_argument("--split", default="test")
    p.add_argument("--max-peaks", type=int, default=512)
    p.add_argument("--num-proc", type=int, default=16)
    p.add_argument("--max-analytes", type=int, default=0, help="random sample (0 = all)")
    p.add_argument("--no-exclude-replicate", dest="exclude_replicate",
                   action="store_false")
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--models", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--max-peaks", type=int, default=512)
    # DeltaMZBias is O(batch * peaks^2 * n_freqs); 16 x 512 peaks is 8 GiB.
    s.add_argument("--batch-size", type=int, default=16)
    cli = ap.parse_args(argv)
    return prepare(cli) if cli.cmd == "prepare" else score(cli)


if __name__ == "__main__":
    raise SystemExit(main())
