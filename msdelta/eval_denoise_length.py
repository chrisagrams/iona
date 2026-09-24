"""Denoise performance by spectrum length, including spectra LONGER than training saw.

    python -m msdelta.eval_denoise_length --models FILE --out-dir DIR [--shard I --num-shards N]

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
    ap.add_argument("--max-peaks", type=int, default=4096)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    cli = ap.parse_args(argv)

    from datasets import load_dataset

    from msdelta.finetune_denoise import denoise_metrics
    from msdelta.modeling_msdelta import MSDeltaForDenoising
    from msdelta.processing_msdelta import MSDeltaProcessor

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
        model = MSDeltaForDenoising.from_pretrained(path).to(device).eval()
        per = []           # (n_peaks, logits, labels) per spectrum
        skipped = []
        order = np.argsort([len(m) for m in data["mz"]])   # short first; OOM only at the tail
        with torch.no_grad():
            for i in order:
                row = data[int(i)]
                n = len(row["mz"])
                mz = torch.tensor([row["mz"]], dtype=torch.float32, device=device)
                li = torch.tensor([row["log_intensity"]], dtype=torch.float32, device=device)
                mask = torch.ones_like(mz, dtype=torch.long)
                try:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=device.type == "xpu"):
                        logits = model(mz=mz, log_intensity=li, attention_mask=mask,
                                       return_dict=True).logits
                except RuntimeError as error:          # out of memory at the longest
                    skipped.append(n)
                    print(f"  {name}: skipped a {n}-peak spectrum ({str(error)[:80]})",
                          flush=True)
                    if device.type == "xpu":
                        torch.xpu.empty_cache()
                    continue
                per.append((n, logits.float().cpu().numpy()[0], np.asarray(row["labels"])))
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
