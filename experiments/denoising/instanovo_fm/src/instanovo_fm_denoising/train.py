"""Train a denoising head on a frozen InstaNovo-FM encoder with Accelerate.

Single process:  instanovo-fm-denoising --output-dir ...
Multiple GPUs:   accelerate launch --num_processes 2 -m instanovo_fm_denoising.train ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from tqdm import tqdm

from instanovo_fm_denoising.data import (
    DEFAULT_DATASET_REPO,
    PeakBudgetBatchSampler,
    build_denoising_datasets,
    collate_denoising,
    processor_kwargs,
)
from instanovo_fm_denoising.metrics import denoising_metrics, full_spectrum_metrics
from instanovo_fm_denoising.model import (
    DEFAULT_CHECKPOINT,
    InstaNovoFMDenoiser,
    count_parameters,
    resolve_checkpoint,
)

ACCELERATE_PRECISION = {"fp32": "no", "fp16": "fp16", "bf16": "bf16"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="InstaNovo-FM checkpoint file, or a registered model ID "
        "(downloaded to ~/.cache/instanovo-fm)",
    )
    parser.add_argument(
        "--encoder-init",
        choices=("pretrained", "random"),
        default="pretrained",
        help="'random' keeps the checkpoint's architecture but uses freshly "
        "initialized encoder weights (seeded by --seed)",
    )
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--head-hidden-size", type=int, default=128)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--peak-pair-budget", type=int, default=4_194_304)
    parser.add_argument(
        "--full-max-peaks",
        type=int,
        default=1024,
        help="largest raw spectrum included in denoise_full/* metrics "
        "(MSDelta's denoise_max_peaks)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device_type(name: str) -> str:
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "xpu" if xpu_available else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if name == "xpu" and not xpu_available:
        raise RuntimeError("--device xpu requested but XPU is not available")
    return name


def resolve_precision(precision: str, device_type: str) -> str:
    """Return the precision actually used, falling back to fp32 if unsupported."""
    if precision == "fp32":
        return precision
    if device_type == "cuda":
        supported = precision == "fp16" or torch.cuda.is_bf16_supported()
    elif device_type == "xpu":
        supported = True
    else:
        supported = precision == "bf16"
    if not supported:
        warnings.warn(f"{precision} autocast is not supported on {device_type}; using fp32")
        return "fp32"
    return precision


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_loader(
    dataset,
    accelerator: Accelerator,
    *,
    peak_pair_budget: int,
    seed: int,
    train: bool,
    num_workers: int,
) -> tuple[DataLoader, PeakBudgetBatchSampler]:
    # Sharding is done by the sampler rather than accelerator.prepare so that
    # variable-sized batches, epoch reshuffling, and exact evaluation
    # (no duplicated spectra) stay under our control.
    sampler = PeakBudgetBatchSampler(
        lengths=dataset["num_peaks"],
        peak_pair_budget=peak_pair_budget,
        seed=seed,
        shuffle=train,
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
        pad=train,
    )
    columns = ["mz", "intensity", "labels"]
    if not train:
        columns += ["source_index", "kept_index"]
    loader = DataLoader(
        dataset.with_format("numpy", columns=columns),
        batch_sampler=sampler,
        collate_fn=collate_denoising,
        num_workers=num_workers,
        pin_memory=accelerator.device.type == "cuda",
    )
    return loader, sampler


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: InstaNovoFMDenoiser,
    loader: DataLoader,
    accelerator: Accelerator,
    raw_validation,
    full_max_peaks: int,
) -> dict[str, float]:
    """Evaluate this process's shard, then gather every spectrum on all processes.

    Reports ``denoise/*`` over the peaks preprocessing retains and
    ``denoise_full/*`` over every original peak (see full_spectrum_metrics).
    """
    model.eval()
    shard: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    progress = tqdm(loader, desc="Denoising eval", disable=not accelerator.is_local_main_process)
    for batch in progress:
        source_index = batch.pop("source_index").tolist()
        kept_index = batch.pop("kept_index").numpy()
        batch = to_device(batch, accelerator.device)
        with accelerator.autocast():
            _, logits, valid = model(batch["mz"], batch["intensity"], batch["labels"])
        logits = logits.float().cpu().numpy()
        valid = valid.cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        for row, index in enumerate(source_index):
            row_valid = valid[row]
            if not np.array_equal(row_valid, kept_index[row] >= 0):
                raise RuntimeError("valid peak positions do not match kept_index")
            shard.append(
                (index, kept_index[row][row_valid], logits[row][row_valid], labels[row][row_valid])
            )
    spectra = gather_object(shard)

    logits = np.concatenate([row_logits for _, _, row_logits, _ in spectra])
    labels = np.concatenate([row_labels for _, _, _, row_labels in spectra])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits), torch.from_numpy(labels)
    ).item()
    retained = {"loss": loss, **denoising_metrics(logits, labels)}
    metrics = {f"denoise/{name}": value for name, value in retained.items()}

    scored = {index: (kept, row_logits) for index, kept, row_logits, _ in spectra}
    full = full_spectrum_metrics(raw_validation["noise"], scored, full_max_peaks)
    metrics.update({f"denoise_full/{name}": value for name, value in full.items()})
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device_type = resolve_device_type(args.device)
    precision = resolve_precision(args.precision, device_type)
    accelerator = Accelerator(
        mixed_precision=ACCELERATE_PRECISION[precision],
        cpu=device_type == "cpu",
    )
    if accelerator.device.type != device_type:
        raise RuntimeError(f"Accelerate selected {accelerator.device}, expected {device_type}")
    seed_everything(args.seed)

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Let the main process download the checkpoint and populate the datasets
    # cache before the others read them.
    with accelerator.main_process_first():
        checkpoint = resolve_checkpoint(args.checkpoint)
    # Built before the datasets: the preprocessing settings are part of the
    # checkpoint's config.
    model = InstaNovoFMDenoiser.from_checkpoint(
        checkpoint,
        head_hidden_size=args.head_hidden_size,
        head_dropout=args.head_dropout,
        random_init=args.encoder_init == "random",
    )
    preprocessing = processor_kwargs(model.encoder.cfg)
    accelerator.print(f"encoder init: {args.encoder_init}")
    accelerator.print(f"preprocessing: {preprocessing}")
    with accelerator.main_process_first():
        datasets, raw_validation = build_denoising_datasets(
            preprocessing,
            args.dataset_repo,
            train_split=args.train_split,
            validation_split=args.validation_split,
            cache_dir=args.cache_dir,
            num_proc=args.num_proc if args.num_proc > 1 else None,
        )
    train_loader, train_sampler = make_loader(
        datasets["train"],
        accelerator,
        peak_pair_budget=args.peak_pair_budget,
        seed=args.seed,
        train=True,
        num_workers=args.num_workers,
    )
    validation_loader, _ = make_loader(
        datasets["validation"],
        accelerator,
        peak_pair_budget=args.peak_pair_budget,
        seed=args.seed,
        train=False,
        num_workers=args.num_workers,
    )

    parameter_counts = {
        "encoder": count_parameters(model.encoder),
        "head": count_parameters(model.head),
        "total": count_parameters(model),
        "trainable": count_parameters(model, trainable_only=True),
    }
    for name, count in parameter_counts.items():
        accelerator.print(f"{name} parameters: {count:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        progress = tqdm(
            train_loader,
            desc=f"Denoising train (epoch {epoch + 1}/{args.epochs})",
            disable=not accelerator.is_local_main_process,
        )
        for batch in progress:
            batch = to_device(batch, accelerator.device)
            with accelerator.autocast():
                loss, _, _ = model(batch["mz"], batch["intensity"], batch["labels"])
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

    # Evaluate the unwrapped module: shards may have different batch counts,
    # which DDP forward passes are not designed for.
    denoiser = accelerator.unwrap_model(model)
    metrics = evaluate(
        denoiser, validation_loader, accelerator, raw_validation, args.full_max_peaks
    )

    if accelerator.is_main_process:
        for name, value in metrics.items():
            print(f"{name}: {value:.6f}")
        torch.save(denoiser.head.state_dict(), args.output_dir / "head.pt")
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        run_config = {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256sum(checkpoint),
            "preprocessing": preprocessing,
            "resolved_device": device_type,
            "resolved_precision": precision,
            "num_processes": accelerator.num_processes,
            "parameter_counts": parameter_counts,
            "num_train_spectra": len(datasets["train"]),
            "num_validation_spectra": len(datasets["validation"]),
            "versions": {
                "python": sys.version.split()[0],
                "instanovo_fm": version("instanovo-fm"),
                "accelerate": version("accelerate"),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }
        (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
