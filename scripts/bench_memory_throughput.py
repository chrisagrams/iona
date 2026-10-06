"""Peak memory and throughput of one Iona configuration across batch sizes.

Three stages, normally driven by ``pbs/aurora-benchmark.pbs``:

* ``pool``: preprocess a fixed, shuffled pool of real spectra once and pickle it.
* ``experiment``: sweep batch sizes for one (mode, variant, compile, sorted) configuration,
  then search for the largest batch whose whole timed set fits, appending one CSV row per
  measurement. Every measurement runs in a fresh ``measure`` subprocess so an out-of-memory
  error cannot leak allocator state into the next one.
* ``measure``: run one batch size and print a ``RESULT`` JSON line.

The model code under test is whatever ``iona`` resolves to on ``PYTHONPATH``, so one copy of
this script can benchmark several checkouts. ``scripts/plot_memory_throughput.py`` turns the
CSV into the memory/throughput figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

GRID = (4, 8, 16, 32, 64, 128)
CSV_FIELDS = (
    "mode", "variant", "compiled", "sorted", "batch", "kind", "status", "peak_gib",
    "spectra_per_s", "n_batches", "padded_lengths", "device", "device_total_gib",
    "torch_version", "iona_path",
)


def accelerator():
    import torch

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", torch.xpu
    if torch.cuda.is_available():
        return "cuda", torch.cuda
    raise SystemExit("no XPU or CUDA device available")


def is_oom(error: BaseException) -> bool:
    import torch

    text = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in text or (
        "out_of_device_memory" in text
    )


# --------------------------------------------------------------------------- pool


def build_pool(args) -> None:
    """Pickle ``--pool-size`` non-empty preprocessed spectra, shuffled with a fixed seed."""
    from datasets import load_from_disk

    data = load_from_disk(args.dataset_dir)
    if hasattr(data, "keys"):
        data = data[args.split]
    data = data.shuffle(seed=0).select_columns(["mz", "log_intensity", "labels"])
    rows = []
    for row in data:
        if len(row["mz"]):
            rows.append(row)
        if len(rows) == args.pool_size:
            break
    Path(args.pool).parent.mkdir(parents=True, exist_ok=True)
    with open(args.pool, "wb") as f:
        pickle.dump(rows, f)
    print(f"[pool] wrote {len(rows)} spectra to {args.pool}")


# --------------------------------------------------------------------------- measure


def measure(args) -> dict:
    import numpy as np
    import torch

    import iona
    from iona.modeling_iona import IonaForPreTraining
    from iona.processing_iona import IonaDataCollatorForPreTraining

    device, acc = accelerator()
    rows = pickle.load(open(args.pool, "rb"))
    take = lambda start, n: [rows[(start + i) % len(rows)] for i in range(n)]  # noqa: E731

    def collate(chunk, mask_ratio=0.0, seed=0):
        torch.manual_seed(seed)
        batch = IonaDataCollatorForPreTraining(
            mask_ratio=mask_ratio, min_masked=1 if mask_ratio else 0, pad_to_multiple_of=64
        )(chunk)
        if not mask_ratio:
            batch.pop("mask_positions")
            batch.pop("labels")
        return {k: v.to(device) for k, v in batch.items()}

    model = IonaForPreTraining.from_pretrained(args.checkpoint, dtype=torch.float32).to(device)
    if args.compile:
        # One static kernel set per padded length (multiples of 64) instead of a dynamic one.
        torch._dynamo.config.automatic_dynamic_shapes = False
        torch._dynamo.config.cache_size_limit = 32
    autocast = lambda: torch.autocast(device, dtype=torch.bfloat16)  # noqa: E731
    B = args.batch
    lengths: list[int] = []

    if args.mode == "infer":
        model.eval()

        def forward(mz, log_intensity, attention_mask):
            return model.iona(mz=mz, log_intensity=log_intensity,
                              attention_mask=attention_mask).last_hidden_state

        fn = torch.compile(forward) if args.compile else forward
        n_batches = max(8, min(32, 2048 // B))

        if args.sorted:
            from datasets import Dataset

            from iona.data import map_length_sorted

            ds = Dataset.from_list(take(0, n_batches * B))

            def embed(chunk):
                bt = collate(chunk)
                lengths.append(bt["mz"].shape[1])
                h = fn(bt["mz"], bt["log_intensity"], bt["attention_mask"])
                mask = bt["attention_mask"].unsqueeze(-1)
                return {"emb": ((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy()}

            run_pass = lambda: map_length_sorted(ds, embed, B)  # noqa: E731
        else:
            batches = [collate(take(i * B, B)) for i in range(n_batches)]
            lengths = [bt["mz"].shape[1] for bt in batches]

            def run_pass():
                for bt in batches:
                    fn(bt["mz"], bt["log_intensity"], bt["attention_mask"])

        context = torch.inference_mode
    else:
        model.train()
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=1.3e-4, betas=(0.9, 0.95),
                                    weight_decay=0.01, fused=True)
        except (RuntimeError, ValueError):
            opt = torch.optim.AdamW(model.parameters(), lr=1.3e-4, betas=(0.9, 0.95),
                                    weight_decay=0.01)
        fwd = torch.compile(model) if args.compile else model
        n_batches = max(8, min(16, 1024 // B))
        if args.sorted:
            # group_by_length order: HF's megabatches of 50 batches over 50*B spectra. Keep the
            # longest batch (HF puts it first) plus random others so the peak stays worst-case.
            from transformers.trainer_pt_utils import get_length_grouped_indices

            pool = take(0, 50 * B)
            order = get_length_grouped_indices([len(r["mz"]) for r in pool], B, mega_batch_mult=50,
                                               generator=torch.Generator().manual_seed(0))
            groups = [order[i:i + B] for i in range(0, len(order), B)]
            pick = [0, *sorted(np.random.default_rng(0).choice(
                np.arange(1, len(groups)), n_batches - 1, replace=False))]
            batches = [collate([pool[j] for j in groups[k]], 0.5, int(k)) for k in pick]
        else:
            batches = [collate(take(i * B, B), 0.5, i) for i in range(n_batches)]
        lengths = [bt["mz"].shape[1] for bt in batches]

        def run_pass():
            for bt in batches:
                with autocast():
                    loss = fwd(**bt).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

        context = torch.enable_grad

    with context(), autocast():
        run_pass()  # warmup: every compile and the optimizer state happen here
        acc.synchronize()
        acc.reset_peak_memory_stats()
        lengths_before = len(lengths)
        start = time.perf_counter()
        run_pass()
        acc.synchronize()
        elapsed = time.perf_counter() - start
    if args.sorted and args.mode == "infer":
        lengths = lengths[lengths_before:]
    return {
        "status": "ok",
        "peak_gib": acc.max_memory_allocated() / 2**30,  # over every timed batch
        "spectra_per_s": n_batches * B / elapsed,
        "n_batches": n_batches,
        "padded_lengths": ";".join(str(n) for n in sorted(set(lengths))),
        "device": acc.get_device_name(0),
        "device_total_gib": acc.get_device_properties(0).total_memory / 2**30,
        "torch_version": torch.__version__,
        "iona_path": str(Path(iona.__file__).parent),
    }


def measure_main(args) -> None:
    try:
        result = measure(args)
    except Exception as error:  # noqa: BLE001 - reported to the driver, never swallowed
        if not is_oom(error):
            raise
        result = {"status": "oom"}
    print("RESULT " + json.dumps(result), flush=True)


# --------------------------------------------------------------------------- experiment


def experiment(args) -> None:
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    handle = open(out, "a", newline="")
    writer = csv.DictWriter(handle, CSV_FIELDS)
    if new_file:
        writer.writeheader()
    base = {"mode": args.mode, "variant": args.variant, "compiled": int(args.compile),
            "sorted": int(args.sorted)}

    def run(batch: int) -> dict:
        cmd = [sys.executable, __file__, "measure", "--mode", args.mode, "--batch", str(batch),
               "--checkpoint", args.checkpoint, "--pool", args.pool]
        cmd += ["--compile"] if args.compile else []
        cmd += ["--sorted"] if args.sorted else []
        proc = subprocess.run(cmd, capture_output=True, text=True)
        sys.stderr.write(proc.stderr[-4000:])
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[7:])
        return {"status": "error"}

    def record(batch: int, kind: str, result: dict) -> None:
        writer.writerow({**base, "batch": batch, "kind": kind,
                         **{k: result.get(k, "") for k in CSV_FIELDS if k in result}})
        handle.flush()
        print(f"[{args.variant} {args.mode} compile={int(args.compile)}] {kind} B={batch}: "
              f"{json.dumps(result)}", flush=True)

    results: dict[int, dict] = {}
    for batch in args.grid:
        if any(r["status"] == "oom" for r in results.values()):
            record(batch, "grid", {"status": "skipped"})  # a smaller batch already ran out of memory
            continue
        results[batch] = run(batch)
        record(batch, "grid", results[batch])
        if results[batch]["status"] == "error":
            raise SystemExit(f"measurement failed at batch {batch}; see stderr")
    ok = [b for b, r in results.items() if r["status"] == "ok"]
    if not args.max_search or not ok:
        return
    lo, hi = max(ok), max(ok) * 2
    while True:  # grow until a batch size fails, then bisect
        results[hi] = run(hi)
        if results[hi]["status"] == "error":
            raise SystemExit(f"measurement failed at batch {hi}; see stderr")
        if results[hi]["status"] != "ok":
            break
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        results[mid] = run(mid)
        if results[mid]["status"] == "error":
            raise SystemExit(f"measurement failed at batch {mid}; see stderr")
        lo, hi = (mid, hi) if results[mid]["status"] == "ok" else (lo, mid)
    record(lo, "max", results[lo])
    print(f"[{args.variant} {args.mode} compile={int(args.compile)}] largest batch {lo}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    pool = sub.add_parser("pool")
    pool.add_argument("--dataset-dir", required=True, help="load_from_disk path of preprocessed spectra")
    pool.add_argument("--split", default="validation")
    pool.add_argument("--pool-size", type=int, default=30_000)
    pool.add_argument("--pool", required=True)

    for name in ("experiment", "measure"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=("infer", "train"), required=True)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--pool", required=True)
        p.add_argument("--compile", action="store_true")
        p.add_argument("--sorted", action="store_true", help="length-sorted batches (PR #48)")
        if name == "measure":
            p.add_argument("--batch", type=int, required=True)
        else:
            p.add_argument("--variant", required=True, help="label written to the CSV")
            p.add_argument("--csv", required=True)
            p.add_argument("--grid", type=lambda s: [int(x) for x in s.split(",")], default=list(GRID))
            p.add_argument("--no-max-search", dest="max_search", action="store_false")

    args = parser.parse_args(argv)
    {"pool": build_pool, "experiment": experiment, "measure": measure_main}[args.stage](args)


if __name__ == "__main__":
    main()
