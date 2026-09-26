"""Score saved contrastive encoders on ms-contrastive-100k's held-out split. No training.

    # once, on a compute node: flatten + preprocess the split to disk
    python scripts/eval_grouped_retrieval.py prepare --out-data DIR --processor CKPT
    # then one process per tile, each taking every num_shards-th model
    python scripts/eval_grouped_retrieval.py score --data DIR --models FILE \
        --shard I --num-shards N --out-dir DIR

--models is a text file, one `name path [pooling]` per line (pooling defaults to mean+max).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk

from msdelta.contrastive import MSDeltaForContrastive, embed_dataset
from msdelta.data import build_grouped_split, corpus_peptides, group_ids
from msdelta.finetune_contrastive import ContrastiveCollator
from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.retrieval import retrieval_metrics


def prepare(cli) -> int:
    processor = MSDeltaProcessor.from_pretrained(cli.processor, max_peaks=cli.max_peaks)
    exclude = corpus_peptides(cli.exclude_peptides_from) if cli.exclude_peptides_from else set()
    print(f"[prepare] excluding {len(exclude):,} peptides from {cli.exclude_peptides_from}", flush=True)
    raw = load_dataset(cli.repo)[cli.split]
    if cli.max_analytes and len(raw) > cli.max_analytes:
        raw = raw.shuffle(seed=0).select(range(cli.max_analytes))
    rows = build_grouped_split(raw, processor, include_consensus=True,
                               exclude_peptides=exclude, num_proc=cli.num_proc)
    if Path(cli.out_data).exists():
        raise SystemExit(f"{cli.out_data} exists; refusing to overwrite")
    rows.save_to_disk(cli.out_data)
    Path(cli.out_data, "PREPARED.json").write_text(json.dumps({
        "repo": cli.repo, "split": cli.split, "processor": cli.processor,
        "max_peaks": cli.max_peaks, "exclude_peptides_from": cli.exclude_peptides_from,
        "excluded_peptides": len(exclude),
        "rows": len(rows)}, indent=1))
    print(f"[prepare] wrote {len(rows):,} spectra to {cli.out_data}", flush=True)
    return 0


def score_model(path, pooling, rows, groups, experimental, collator, device,
                batch_size) -> dict:
    encoder = MSDeltaForPreTraining.from_pretrained(path)
    model = MSDeltaForContrastive(encoder, None, pooling=pooling, kl_weight=0).to(device)
    try:
        emb, _ = embed_dataset(model, rows, collator, device, max_rows=len(rows),
                               batch_size=batch_size)
        out = _variants(emb, groups, experimental, device, retrieval_metrics)
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
        metrics = metric(emb[torch.from_numpy(mask)], groups[mask], device)
        _, inverse, counts = np.unique(groups[mask], return_inverse=True, return_counts=True)
        metrics["queries"] = float((counts[inverse] > 1).sum())
        out |= {f"{variant}/{k}": v for k, v in metrics.items()}
    return out


def score(cli) -> int:
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
    p.add_argument("--repo", required=True, help="grouped dataset repo")
    p.add_argument("--split", default="test")
    p.add_argument("--max-peaks", type=int, default=512)
    p.add_argument("--num-proc", type=int, default=16)
    p.add_argument("--max-analytes", type=int, default=0, help="random sample (0 = all)")
    p.add_argument("--exclude-peptides-from", default=None,
                   help="dataset repo whose peptides to drop")
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
