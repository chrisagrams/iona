"""Throughput of one contrastive GradCache step, across the settings that could make it faster.

    python pbs/diag/gradcache_bench.py --size 50m --out results/raw/finetune/contrastive/gradcache_bench/50m.json

One real training batch: 85 peptide groups x their (<= 3) experimental spectra from the
ms-contrastive-100k validation split, collated exactly as training does (padded to 512 peaks),
C7 recipe model (mean+max pooling, temperature 0.002, KL weight 10 to a frozen reference).
Each configuration: 1 warm-up step, then the median of 3 timed steps (forward + backward, no
optimizer update), and peak device memory. Varied: GradCache chunk size, trim_padding
(length-sorted, per-chunk trimmed chunks), gradient checkpointing, and the KL term (to price
the reference forward). An out-of-memory configuration is recorded as such, not fatal.
"""
import argparse
import json
import statistics
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

P = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-{}-production-01-checkpoint-540423"
DATA = "/lus/flare/projects/UIC-HPC/khuss/msdelta/eval-data/ms-contrastive-100k-validation-mp512"


def batch_rows(groups_per_batch=85, seed=0):
    from datasets import load_from_disk
    d = load_from_disk(DATA)
    src = np.array(d["source"]); aid = np.array(d["analyte_id"])
    exp = np.flatnonzero(src == "experimental")
    by = {}
    for i in exp:
        by.setdefault(aid[i], []).append(int(i))
    keys = [k for k, v in by.items() if len(v) >= 2]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(keys), groups_per_batch, replace=False)
    rows = [i for j in pick for i in by[keys[j]][:3]]
    return d.select(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", default="4,8,16,32,64")
    cli = ap.parse_args()
    from msdelta.contrastive import MSDeltaForContrastive, gradcache_step
    from msdelta.finetune_contrastive import ContrastiveCollator
    from msdelta.modeling_msdelta import MSDeltaForPreTraining

    dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    rows = batch_rows()
    feats = [{"mz": r["mz"], "log_intensity": r["log_intensity"], "peptide": r["peptide"],
              "charge": r["charge"]} for r in rows]
    batch = ContrastiveCollator(max_peptide_length=64, pad_spectra_to=512)(feats)
    lengths = batch["attention_mask"].sum(1)
    print(f"[bench] {cli.size}: batch {len(feats)} spectra, peaks median {int(lengths.median())} "
          f"max {int(lengths.max())}, padded to {batch['mz'].shape[1]}", flush=True)
    batch = {k: v.to(dev) for k, v in batch.items()}

    path = P.format(cli.size)
    enc = MSDeltaForPreTraining.from_pretrained(path).to(dev)
    ref = MSDeltaForPreTraining.from_pretrained(path).to(dev)
    model = MSDeltaForContrastive(enc, ref, pooling="mean+max", temperature=0.002, kl_weight=10.0).train()
    results = []
    configs = list(product([int(c) for c in cli.chunks.split(",")], [False, True], [True, False], [10.0]))
    configs += [(16, True, False, 0.0), (32, True, False, 0.0)]          # price the KL reference
    for chunk, trim, ckpt, kl in configs:
        model.kl_weight = kl
        (model.gradient_checkpointing_enable if ckpt else model.gradient_checkpointing_disable)()
        rec = dict(size=cli.size, chunk=chunk, trim_padding=trim, grad_ckpt=ckpt, kl_weight=kl)
        try:
            times = []
            for i in range(4):
                model.zero_grad(set_to_none=True)
                if dev.type == "xpu":
                    torch.xpu.synchronize(); torch.xpu.reset_peak_memory_stats()
                t = time.time()
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                    gradcache_step(model, batch, chunk_size=chunk, trim_padding=trim)
                if dev.type == "xpu":
                    torch.xpu.synchronize()
                if i:
                    times.append(time.time() - t)
            rec["step_s"] = statistics.median(times)
            rec["peak_gb"] = torch.xpu.max_memory_allocated() / 1e9 if dev.type == "xpu" else None
        except RuntimeError as e:
            rec["error"] = "OOM" if "memory" in str(e).lower() else str(e)[:200]
            model.zero_grad(set_to_none=True)
            if dev.type == "xpu":
                torch.xpu.empty_cache()
        print(f"[bench] {rec}", flush=True)
        results.append(rec)
        Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cli.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
