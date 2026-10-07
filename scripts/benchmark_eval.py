"""Time and score a pretrained checkpoint on the masked-intensity eval split.

Run one source tree per process by pointing PYTHONPATH at it; ``iona`` is imported from there,
so the same script benchmarks any branch. Masks are seeded per batch, so two trees see identical
inputs and their per-spectrum losses can be compared directly.

    PYTHONPATH=src/main python scripts/benchmark_eval.py run --expect-source src/main \
        --checkpoint CKPT --dataset-dir DATA --out-dir out/main
    python scripts/benchmark_eval.py compare out/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk

import iona
from iona.modeling_iona import IonaForPreTraining
from iona.processing_iona import IonaDataCollatorForPreTraining


def run(args: argparse.Namespace) -> None:
    source = Path(iona.__file__).resolve().parent
    if args.expect_source and not source.is_relative_to(Path(args.expect_source).resolve()):
        raise SystemExit(f"imported iona from {source}, expected under {args.expect_source}")
    print(f"[bench] iona={source} torch={torch.__version__}", flush=True)

    device = torch.device(args.device)
    model = IonaForPreTraining.from_pretrained(args.checkpoint).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    forward = torch.compile(model, dynamic=True) if args.compile else model
    backend = getattr(torch, device.type, None) if device.type != "cpu" else None

    datasets = load_from_disk(args.dataset_dir)
    if not isinstance(datasets, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {args.dataset_dir}")
    dataset = datasets[args.split].select_columns(["mz", "log_intensity", "labels"])
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    collator = IonaDataCollatorForPreTraining(
        mask_ratio=args.mask_ratio, pad_to_multiple_of=args.pad_to_multiple_of
    )
    print(
        f"[bench] checkpoint={args.checkpoint} params={n_params / 1e6:.2f}M "
        f"split={args.split} spectra={len(dataset):,} batch={args.batch_size}",
        flush=True,
    )

    batch_seconds, batch_spectra, batch_peaks, batch_padded = [], [], [], []
    kl_parts, cos_parts, loss_parts = [], [], []
    warmup_seconds = 0.0
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16)
    n_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    for index in range(n_batches):
        rows = dataset[index * args.batch_size : (index + 1) * args.batch_size]
        features = [dict(zip(rows, values)) for values in zip(*rows.values())]
        # The collator samples masks from the global CPU generator.
        torch.manual_seed(args.seed + index)
        batch = {k: v.to(device, non_blocking=True) for k, v in collator(features).items()}

        if backend is not None and index == args.warmup_batches:
            backend.reset_peak_memory_stats(device)
        if backend is not None:
            backend.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode(), autocast:
            output = forward(**batch)
        if backend is not None:
            backend.synchronize(device)
        elapsed = time.perf_counter() - start

        if index < args.warmup_batches:
            warmup_seconds += elapsed
        else:
            batch_seconds.append(elapsed)
            batch_spectra.append(len(features))
            batch_peaks.append(int(batch["attention_mask"].sum()))
            batch_padded.append(int(batch["mz"].shape[1]))

        # Recompute the loss per spectrum so the two trees can be compared row by row.
        with torch.inference_mode():
            selected = batch["mask_positions"]
            log_prob = F.log_softmax(
                output.logits.float().masked_fill(~selected, float("-inf")), dim=-1
            ).masked_fill(~selected, 0.0)
            target = batch["labels"].float().masked_fill(~selected, 0.0)
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            kl = (torch.xlogy(target, target) - target * log_prob).sum(dim=-1)
            pred = log_prob.exp().masked_fill(~selected, 0.0)
            cos = F.cosine_similarity(pred, target, dim=-1)
        kl_parts.append(kl.cpu().numpy())
        cos_parts.append(cos.cpu().numpy())
        loss_parts.append(float(output.loss))
        if (index + 1) % args.log_every == 0:
            print(f"[bench] batch {index + 1}/{n_batches}", flush=True)

    if not batch_seconds:
        raise SystemExit("no timed batches; lower --warmup-batches or raise --max-samples")
    kl = np.concatenate(kl_parts)
    cos = np.concatenate(cos_parts)
    timed = float(sum(batch_seconds))
    summary = {
        "source": str(source),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "params": n_params,
        "spectra": int(kl.size),
        "batch_size": args.batch_size,
        "compile": args.compile,
        "bf16": args.bf16,
        "eval_loss": float(kl.mean()),
        "model_loss_mean": float(np.mean(loss_parts)),
        "masked_cosine": float(cos.mean()),
        "warmup_batches": args.warmup_batches,
        "warmup_seconds": warmup_seconds,
        "timed_batches": len(batch_seconds),
        "timed_seconds": timed,
        "spectra_per_second": sum(batch_spectra) / timed,
        "peaks_per_second": sum(batch_peaks) / timed,
        "median_batch_ms": 1e3 * statistics.median(batch_seconds),
        "p90_batch_ms": 1e3 * float(np.percentile(batch_seconds, 90)),
        "peak_memory_gib": backend.max_memory_allocated(device) / 2**30 if backend else 0.0,
        "torch": torch.__version__,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "per_spectrum.npz", kl=kl, cosine=cos)
    np.savez(
        out_dir / "per_batch.npz",
        seconds=np.array(batch_seconds),
        spectra=np.array(batch_spectra),
        peaks=np.array(batch_peaks),
        padded_length=np.array(batch_padded),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def compare(args: argparse.Namespace) -> None:
    root = Path(args.results_dir)
    lines = [
        f"| model | {args.baseline} spectra/s | {args.candidate} spectra/s | speedup "
        f"| {args.baseline} GiB | {args.candidate} GiB "
        f"| {args.baseline} loss | {args.candidate} loss | max abs Δ loss "
        f"| {args.baseline} cosine | {args.candidate} cosine |",
        "|---" * 11 + "|",
    ]
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        base, cand = model_dir / args.baseline, model_dir / args.candidate
        if not (base / "summary.json").is_file() or not (cand / "summary.json").is_file():
            print(f"[compare] skipping {model_dir.name}: missing a summary", file=sys.stderr)
            continue
        b = json.loads((base / "summary.json").read_text())
        c = json.loads((cand / "summary.json").read_text())
        b_kl = np.load(base / "per_spectrum.npz")["kl"]
        c_kl = np.load(cand / "per_spectrum.npz")["kl"]
        delta = float(np.abs(b_kl - c_kl).max()) if b_kl.shape == c_kl.shape else float("nan")
        lines.append(
            f"| {model_dir.name} | {b['spectra_per_second']:.1f} | {c['spectra_per_second']:.1f} "
            f"| {c['spectra_per_second'] / b['spectra_per_second']:.3f}x "
            f"| {b['peak_memory_gib']:.2f} | {c['peak_memory_gib']:.2f} "
            f"| {b['eval_loss']:.6f} | {c['eval_loss']:.6f} | {delta:.2e} "
            f"| {b['masked_cosine']:.6f} | {c['masked_cosine']:.6f} |"
        )
    table = "\n".join(lines) + "\n"
    (root / "comparison.md").write_text(table)
    print(table, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="benchmark one checkpoint with one source tree")
    run_parser.add_argument("--checkpoint", required=True)
    run_parser.add_argument("--dataset-dir", required=True)
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--split", default="validation")
    run_parser.add_argument("--expect-source", help="fail unless iona is imported from here")
    run_parser.add_argument("--device", default="xpu")
    run_parser.add_argument("--batch-size", type=int, default=8)
    run_parser.add_argument("--max-samples", type=int, default=0, help="0 uses the whole split")
    run_parser.add_argument("--mask-ratio", type=float, default=0.5)
    run_parser.add_argument("--pad-to-multiple-of", type=int, default=64)
    run_parser.add_argument("--warmup-batches", type=int, default=10)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--log-every", type=int, default=100)
    run_parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    run_parser.set_defaults(func=run)

    compare_parser = commands.add_parser("compare", help="tabulate paired results")
    compare_parser.add_argument("results_dir", help="directory of <model>/<label>/ results")
    compare_parser.add_argument("--baseline", default="main")
    compare_parser.add_argument("--candidate", default="dev")
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
