"""Denoise performance by spectrum length, including spectra LONGER than training saw.

    python -m msdelta.eval.eval_denoise_length --models FILE --out-dir DIR [--shard I --num-shards N]

Every denoise number in this project filters spectra to <= 512 peaks (max_peaks), and
build_denoising_datasets DROPS the rest -- 13.2% of the test split (1,309 of 9,893;
max 3,086 peaks). Nothing in the architecture caps length: peak tokens carry no absolute
position and m/z enters only as pairwise deltas, so a trained model can be run on any
spectrum; the cost is DeltaMZBias memory, O(peaks^2), hence batch size 1.

This scores the test split with the cap raised to --max-peaks and reports every metric
of finetune_denoise.denoise_metrics per peak-count bucket. The <=512 bucket is the
population every published number comes from and must reproduce it -- that is the check
on this script. PLAN.md FT1.

--models: one `name path` per line, path = a saved MSDeltaForDenoising (final/).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

BUCKETS = ((0, 512), (512, 768), (768, 1024), (1024, 1536), (1536, 1 << 30))


def bucket_name(lo, hi):
    return f"{lo + 1}-{hi}" if hi < (1 << 30) else f">{lo}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default="chrisagrams/ms-denoise-100k")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-peaks", type=int, default=3200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--fixed-width", type=int, default=1,
                    help="1: pad to the bucket width; 0: pad each batch to its own max")
    cli = ap.parse_args(argv)

    from datasets import load_dataset

    from msdelta.finetuning.denoise.finetune_denoise import denoise_metrics
    from msdelta.models.loading import load_strict
    from msdelta.models.modeling_msdelta import MSDeltaForDenoising
    from msdelta.models.processing_msdelta import MSDeltaProcessor

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    models = [l.split() for l in Path(cli.models).read_text().splitlines()
              if l.strip() and not l.startswith("#")][cli.shard::cli.num_shards]
    out_dir = Path(cli.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(cli.repo)[cli.split]
    raw = raw.filter(lambda e: 0 < len(e["mz"]) <= cli.max_peaks)
    for name, path in models:
        target = out_dir / f"{name}.json"
        if target.exists():
            print(f"  {name}: already scored", flush=True); continue
        t0 = time.time()
        processor = MSDeltaProcessor.from_pretrained(path, max_peaks=cli.max_peaks)
        data = raw.map(lambda e: processor.process_denoising_example(
            e["mz"], e["intensity"], e["noise"]), remove_columns=raw.column_names)
        # mask_token: frozen out of the graph during denoise training (never read without
        # mask_positions), and some denoise checkpoints omit it -- as in eval_checkpoint.
        # (Also MSDeltaForDenoising's class default since K105-S; kept explicit here.)
        model = load_strict(MSDeltaForDenoising, path,
                            allow_missing=("msdelta.embed.mask_token",)).to(device).eval()
        per = []           # (n_peaks, logits, labels) per spectrum
        skipped = []
        # FIXED widths, not each spectrum's own length: one shape per call made every
        # tile take a GPU page fault within seconds (8862542, 4/4 processes; the FT16
        # mechanism). Padding is masked, so per-peak logits are unchanged; a handful of
        # shapes also lets the short majority run batched.
        widths = [w for w in (512, 768, 1024, 1536, 2048, 3200) if w <= cli.max_peaks] \
            or [cli.max_peaks]
        lengths = np.array([len(m) for m in data["mz"]])
        with torch.no_grad():
            for w_i, width in enumerate(widths):
                lo = widths[w_i - 1] if w_i else 0
                idx = np.flatnonzero((lengths > lo) & (lengths <= width))
                bs = max(1, (512 * 512 * 16) // (width * width))   # pairwise bias ~ width^2
                for start in range(0, len(idx), bs):
                    chunk = [data[int(i)] for i in idx[start:start + bs]]
                    w = width if cli.fixed_width else max(len(r["mz"]) for r in chunk)
                    mz = torch.zeros(len(chunk), w); li = torch.zeros(len(chunk), w)
                    mask = torch.zeros(len(chunk), w, dtype=torch.long)
                    for r, row in enumerate(chunk):
                        n = len(row["mz"])
                        mz[r, :n] = torch.tensor(row["mz"]); li[r, :n] = torch.tensor(row["log_intensity"])
                        mask[r, :n] = 1
                    try:
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                            enabled=device.type == "xpu" and not cli.fp32):
                            logits = model(mz=mz.to(device), log_intensity=li.to(device),
                                           attention_mask=mask.to(device),
                                           return_dict=True).logits.float().cpu().numpy()
                    except RuntimeError as error:      # out of memory at the longest
                        skipped += [len(r["mz"]) for r in chunk]
                        print(f"  {name}: skipped {len(chunk)} spectra at width {width} "
                              f"({str(error)[:80]})", flush=True)
                        if device.type == "xpu":
                            torch.xpu.empty_cache()
                        continue
                    for r, row in enumerate(chunk):
                        n = len(row["mz"])
                        per.append((n, logits[r, :n], np.asarray(row["labels"])))
        del model
        if device.type == "xpu":
            torch.xpu.empty_cache()

        result = {"name": name, "path": path, "max_peaks": cli.max_peaks,
                  "skipped_lengths": skipped, "buckets": {}}
        for lo, hi in BUCKETS + ((0, 1 << 30),):
            rows = [(lg, lb) for n, lg, lb in per if lo < n <= hi]
            if not rows:
                continue
            width = max(len(lg) for lg, _ in rows)
            logits_2d = np.zeros((len(rows), width)); labels_2d = np.full((len(rows), width), -100.0)
            for r, (lg, lb) in enumerate(rows):
                logits_2d[r, :len(lg)] = lg; labels_2d[r, :len(lb)] = lb
            m = denoise_metrics(SimpleNamespace(predictions=logits_2d, label_ids=labels_2d))
            key = "all" if (lo, hi) == (0, 1 << 30) else bucket_name(lo, hi)
            result["buckets"][key] = {"spectra": len(rows), **m}
        target.write_text(json.dumps(result, indent=1))
        b = result["buckets"]
        print(f"  {name} ({time.time() - t0:.0f}s): " + "  ".join(
            f"{k} AUROC {v['auroc']:.4f} (n={v['spectra']})" for k, v in b.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
