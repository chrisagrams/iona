"""Masked-peak (pretraining) validation loss of saved checkpoints, on one preprocessed dataset.

    python -m msdelta.pretraining.eval_mlm --checkpoints A,B --dataset DIR --split validation \
        --out FILE.json [--max_spectra N] [--batch_size B] [--seed S] [--mask_ratio R] \
        [--precision bf16|fp32] [--device xpu|cpu] [--pad_to_multiple_of M]

Scores every checkpoint (transformer or Pairformer, anything `MSDeltaForPreTraining` loads) on
the SAME spectra with the SAME masked peaks and the SAME loss, so their numbers can be compared.
Nothing is trained and nothing is reprocessed: the rows of a `msdelta.preprocess` DatasetDict are
fed as they are, whatever `max_peaks` a checkpoint's own preprocessor_config.json names (that
value is only read by the processor, which already ran when the dataset was built; the models
have no length limit). It is recorded per checkpoint, with a warning when the data holds longer
spectra than the checkpoint was trained on.

What is reused from `msdelta.train` (so the number is the trainer's `eval_loss`):
  * rows -> tensors: `MSDeltaDataCollatorForPreTraining` (padding, and the masking rule
    n_masked = min(len, max(min_masked, round(len * mask_ratio))) drawn with torch.randperm);
  * the loss: `MSDeltaForPreTraining.forward(..., labels=...)` (KL, reduction="batchmean");
  * the reduction: `Trainer.evaluation_loop` repeats each batch's loss batch_size times and
    `gather_for_metrics` cuts the last batch to its real size, so eval_loss is
    sum_b(n_b * loss_b) / N = the mean over spectra of the per-spectrum KL -- independent of the
    batch size. That is what `evaluate_model` computes (tests/test_eval_mlm.py checks it
    against `MSDeltaTrainer.evaluate()` on the same model and data);
  * precision: pretraining ran with --bf16 true, i.e. the forward under bf16 autocast (the
    production DeepSpeed runs held bf16 weights instead); --precision picks bf16 autocast
    (default) or fp32 here, the same for every checkpoint.
The trainer's other eval metrics are its speed metrics (runtime, samples/s, steps/s); there
is no compute_metrics. They are reported the same way.

Deterministic masks. The trainer's collator draws masks from the global torch RNG, so they
change with the RNG state, the batch composition and the DataLoader workers. Here
`SeededMaskingCollator` runs the same collator once per spectrum with the global CPU RNG
seeded from (seed, row index in the split) inside `torch.random.fork_rng`, so a spectrum's
masked peaks depend only on the seed and its row: not on the checkpoint, the batch size, the
order, --max_spectra (a prefix of the split keeps its masks) or anything run before. The
batches are built once per run and fed to every checkpoint; `mask_digest` / `input_digest`
(sha256 over each row's mask bits / inputs in row order, also independent of batching) let
two runs prove they used identical masks and spectra. torch.randperm's stream is fixed for a
given torch version, which is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from msdelta.models.processing_msdelta import MSDeltaDataCollatorForPreTraining

INDEX_KEY = "spectrum_index"
MODEL_INPUTS = ("mz", "log_intensity", "labels")


def spectrum_seed(seed: int, index: int) -> int:
    """The torch seed of one spectrum's mask: a hash of (run seed, row index)."""
    state = np.random.SeedSequence([int(seed), int(index)]).generate_state(2, dtype=np.uint32)
    return (int(state[0]) << 31) ^ int(state[1])        # < 2**63, what manual_seed accepts


class IndexedDataset(torch.utils.data.Dataset):
    """A preprocessed split whose rows also carry their row index (the mask seed)."""

    def __init__(self, dataset):
        columns = [c for c in MODEL_INPUTS if c in dataset.column_names]
        self.dataset = dataset.select_columns(columns)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        return {**self.dataset[int(index)], INDEX_KEY: int(index)}


@dataclass
class SeededMaskingCollator:
    """`MSDeltaDataCollatorForPreTraining` with per-spectrum deterministic masks.

    Padding and the masking rule are the base collator's; only the random draw is pinned:
    each row's mask is the one the base collator draws for that row alone with the RNG
    seeded by `spectrum_seed(seed, row index)`. The global RNG is left as it was.
    """

    base: MSDeltaDataCollatorForPreTraining = field(
        default_factory=MSDeltaDataCollatorForPreTraining)
    seed: int = 0

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        if any(INDEX_KEY not in feature for feature in features):
            raise ValueError(f"features must carry {INDEX_KEY!r} (wrap the split in IndexedDataset)")
        with torch.random.fork_rng(devices=[]):
            batch = self.base(features)                  # padding; its masks are replaced below
            mask_positions = torch.zeros_like(batch["mask_positions"])
            for row, feature in enumerate(features):
                length = len(feature["mz"])
                torch.manual_seed(spectrum_seed(self.seed, feature[INDEX_KEY]))
                mask_positions[row, :length] = self.base([feature])["mask_positions"][0, :length]
        batch["mask_positions"] = mask_positions
        return batch


def build_batches(dataset, collator: SeededMaskingCollator, batch_size: int) -> list[dict]:
    """The evaluation batches, in split order, as the trainer's eval DataLoader yields them
    (sequential sampler, last batch kept). Batched by hand: a DataLoader would draw its
    worker base seed from the global RNG."""
    rows = IndexedDataset(dataset)
    return [collator([rows[i] for i in range(start, min(start + batch_size, len(rows)))])
            for start in range(0, len(rows), batch_size)]


def batch_digests(batches: list[dict]) -> dict:
    """Counts and sha256 digests of the masks and inputs, row by row (batching-independent)."""
    masks, inputs = hashlib.sha256(), hashlib.sha256()
    n_spectra = n_peaks = n_masked = max_len = 0
    for batch in batches:
        lengths = batch["attention_mask"].sum(dim=1).tolist()
        for row, length in enumerate(lengths):
            mask = batch["mask_positions"][row, :length].numpy()
            masks.update(np.int64(length).tobytes() + np.packbits(mask).tobytes())
            for name in ("mz", "log_intensity", "labels"):
                inputs.update(batch[name][row, :length].numpy().astype(np.float32).tobytes())
            n_spectra += 1
            n_peaks += length
            n_masked += int(mask.sum())
            max_len = max(max_len, length)
    return {"n_spectra": n_spectra, "n_peaks": n_peaks, "n_masked_tokens": n_masked,
            "max_peaks_in_data": max_len, "mask_digest": masks.hexdigest(),
            "input_digest": inputs.hexdigest()}


def evaluate_model(model, batches: list[dict], device, precision: str = "bf16",
                   batch_size: int | None = None) -> dict:
    """The trainer's eval_loss and speed metrics of `model` on `batches`.

    Per batch: the model's own loss (forward with labels and mask_positions), weighted by the
    batch's number of spectra, as Trainer.evaluation_loop + gather_for_metrics weight it.
    """
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    device = torch.device(device)
    model.eval()
    total, n_spectra = 0.0, 0
    start = time.time()
    with torch.no_grad():
        for batch in batches:
            inputs = {name: tensor.to(device) for name, tensor in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=precision == "bf16"):
                loss = model(**inputs, return_dict=True).loss
            rows = batch["mz"].shape[0]
            total += float(loss.detach().float().cpu()) * rows
            n_spectra += rows
    runtime = time.time() - start
    # As transformers' speed_metrics: steps = ceil(samples / eval batch size).
    batch_size = batch_size or (batches[0]["mz"].shape[0] if batches else 1)
    steps = math.ceil(n_spectra / batch_size)
    return {"eval_loss": total / max(n_spectra, 1),
            "eval_runtime": round(runtime, 4),
            "eval_samples_per_second": round(n_spectra / runtime, 3) if runtime else None,
            "eval_steps_per_second": round(steps / runtime, 3) if runtime else None}


def _processor_max_peaks(path: str) -> int | None:
    config = Path(path, "preprocessor_config.json")
    if not config.is_file():
        return None
    return json.loads(config.read_text()).get("max_peaks")


def _code_info() -> dict:
    """The job's code snapshot (pbs/lib/code_snapshot.sh), else the checkout's git HEAD."""
    from msdelta.utils.provenance import code_provenance

    info = code_provenance()
    if info:
        return info
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--",
                                 "msdelta"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"checkout": str(repo)}
    return {"checkout": str(repo), "mode": "checkout", "commit": commit, "branch": branch,
            "dirty": bool(status.strip())}


def split_checkpoints(value: str) -> list[str]:
    """Checkpoint paths separated by ',' or '+' (qsub -v values cannot contain commas)."""
    return [p for p in value.replace("+", ",").split(",") if p.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True,
                    help="checkpoint dirs (config.json + model.safetensors), ',' or '+' separated")
    ap.add_argument("--dataset", required=True, help="preprocessed DatasetDict dir")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", required=True, help="JSON file to write")
    ap.add_argument("--max_spectra", type=int, default=None,
                    help="score only the first N rows of the split (their masks are unchanged)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0, help="mask seed")
    ap.add_argument("--mask_ratio", type=float, default=0.50,
                    help="fraction of peaks masked (0.50 in every pretraining config; the "
                         "collator's own default is 0.15)")
    ap.add_argument("--min_masked", type=int, default=1)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16",
                    help="bf16 autocast (as the trainer's eval with --bf16 true) or fp32")
    ap.add_argument("--device", default=None, help="default: xpu if available, else cpu")
    ap.add_argument("--pad_to_multiple_of", type=int, default=None,
                    help="pad every batch to a multiple of this (a fixed width gives the XPU one "
                         "shape); padding is masked, so the loss does not change")
    cli = ap.parse_args(argv)
    if cli.batch_size < 1:
        ap.error("--batch_size must be positive")

    import transformers
    from datasets import DatasetDict, load_from_disk

    from msdelta.models.modeling_msdelta import MSDeltaForPreTraining

    device = torch.device(cli.device or ("xpu" if torch.xpu.is_available() else "cpu"))
    checkpoints = split_checkpoints(cli.checkpoints)
    if not checkpoints:
        ap.error("--checkpoints is empty")

    datasets = load_from_disk(cli.dataset)
    dataset = datasets[cli.split] if isinstance(datasets, DatasetDict) else datasets
    if cli.max_spectra is not None:
        dataset = dataset.select(range(min(cli.max_spectra, len(dataset))))
    collator = SeededMaskingCollator(
        MSDeltaDataCollatorForPreTraining(mask_ratio=cli.mask_ratio, min_masked=cli.min_masked,
                                          pad_to_multiple_of=cli.pad_to_multiple_of),
        seed=cli.seed)
    t0 = time.time()
    batches = build_batches(dataset, collator, cli.batch_size)
    data = batch_digests(batches)
    print(f"[eval_mlm] {cli.dataset} [{cli.split}]: {data['n_spectra']} spectra, "
          f"{data['n_masked_tokens']} masked of {data['n_peaks']} peaks, seed {cli.seed}, "
          f"mask digest {data['mask_digest'][:12]} ({time.time() - t0:.0f}s to build)",
          flush=True)

    result = {
        "tool": "msdelta.pretraining.eval_mlm",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "dataset": str(Path(cli.dataset).resolve()), "split": cli.split,
        "max_spectra": cli.max_spectra, **data,
        "seed": cli.seed, "mask_ratio": cli.mask_ratio, "min_masked": cli.min_masked,
        "masking": "per spectrum: torch.manual_seed(spectrum_seed(seed, row index)) then "
                   "MSDeltaDataCollatorForPreTraining on that row alone",
        "loss": "MSDeltaForPreTraining KL(batchmean), weighted by batch size = mean over spectra",
        "batch_size": cli.batch_size, "pad_to_multiple_of": cli.pad_to_multiple_of,
        "precision": cli.precision, "device": str(device),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "code": _code_info(),
        "checkpoints": [],
    }
    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    failed = 0
    for path in checkpoints:
        entry: dict = {"path": str(path), "processor_max_peaks": _processor_max_peaks(path)}
        try:
            model = MSDeltaForPreTraining.from_pretrained(path, dtype=torch.float32).to(device)
            entry["architecture"] = getattr(model.config, "architecture", "transformer")
            entry["n_params"] = sum(p.numel() for p in model.parameters())
            if entry["processor_max_peaks"] and data["max_peaks_in_data"] > entry["processor_max_peaks"]:
                entry["warning"] = (f"data has spectra of up to {data['max_peaks_in_data']} peaks; "
                                    f"this checkpoint's processor capped at "
                                    f"{entry['processor_max_peaks']}")
                print(f"[eval_mlm] WARNING {path}: {entry['warning']}", flush=True)
            entry.update(evaluate_model(model, batches, device, cli.precision, cli.batch_size))
            entry["n_spectra"] = data["n_spectra"]
            entry["n_masked_tokens"] = data["n_masked_tokens"]
            print(f"[eval_mlm] {entry['architecture']:<11} {entry['n_params'] / 1e6:7.2f}M  "
                  f"eval_loss {entry['eval_loss']:.6f}  ({data['n_spectra']} spectra, "
                  f"{data['n_masked_tokens']} masked, {entry['eval_runtime']:.0f}s)  {path}",
                  flush=True)
            del model
            if device.type == "xpu":
                torch.xpu.empty_cache()
        except Exception as error:          # keep scoring the others; the exit code says so
            failed += 1
            entry["error"] = f"{type(error).__name__}: {error}"
            print(f"[eval_mlm] FAILED {path}: {entry['error']}", flush=True)
        result["checkpoints"].append(entry)
        out.write_text(json.dumps(result, indent=1))      # after each checkpoint
    print(f"[eval_mlm] wrote {out}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
