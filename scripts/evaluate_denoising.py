"""Evaluate Monte Carlo MSDelta signal scores on a labeled Hugging Face dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score

from msdelta import MSDeltaForDenoising, MSDeltaProcessor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Local path or Hub ID for an MSDelta checkpoint")
    parser.add_argument("--processor", default=None, help="Processor path; defaults to checkpoint")
    parser.add_argument("--dataset", default="chrisagrams/ms-denoise-100k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--min-contexts-per-peak", type=int, default=10)
    parser.add_argument("--mask-fraction", type=float, default=0.50)
    parser.add_argument("--loo-batch-size", type=int, default=64)
    parser.add_argument("--max-views", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MSDeltaForDenoising.from_pretrained(args.checkpoint).to(device)
    processor = MSDeltaProcessor.from_pretrained(args.processor or args.checkpoint)
    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    signal_targets: list[np.ndarray] = []
    signal_scores: list[np.ndarray] = []
    total_peaks = 0
    sufficiently_covered_peaks = 0
    evaluated_peaks = 0
    incomplete_spectra = 0
    spectra_processed = 0

    for spectrum_index, row in enumerate(dataset):
        if args.max_spectra is not None and spectrum_index >= args.max_spectra:
            break
        inputs = processor.prepare_denoising_inputs(row["mz"], row["intensity"])
        output = model.denoise(
            **inputs,
            min_contexts_per_peak=args.min_contexts_per_peak,
            mask_fraction=args.mask_fraction,
            loo_batch_size=args.loo_batch_size,
            max_views=args.max_views,
            seed=args.seed + spectrum_index,
        )

        scores = output.signal_score.numpy()
        signal = np.logical_not(np.asarray(row["noise"], dtype=np.bool_))
        valid = np.isfinite(scores)
        signal_targets.append(signal[valid])
        signal_scores.append(scores[valid])
        num_peaks = len(signal)
        total_peaks += num_peaks
        evaluated_peaks += int(valid.sum())
        sufficiently_covered_peaks += int((output.num_contexts >= args.min_contexts_per_peak).sum())
        incomplete_spectra += int(not output.coverage_complete)
        spectra_processed += 1
        if args.progress_every and spectra_processed % args.progress_every == 0:
            print(f"processed {spectra_processed} spectra", flush=True)

    targets = np.concatenate(signal_targets)
    scores = np.concatenate(signal_scores)

    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "spectra": spectra_processed,
        "total_peaks": total_peaks,
        "evaluated_peaks": evaluated_peaks,
        "signal_auroc": float(roc_auc_score(targets, scores)),
        "signal_average_precision": float(average_precision_score(targets, scores)),
        "context_coverage_rate": sufficiently_covered_peaks / max(total_peaks, 1),
        "incomplete_spectra": incomplete_spectra,
        "min_contexts_per_peak": args.min_contexts_per_peak,
        "mask_fraction": args.mask_fraction,
        "max_views": args.max_views,
        "seed": args.seed,
    }
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
