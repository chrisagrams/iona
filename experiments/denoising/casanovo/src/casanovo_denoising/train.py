"""Train a denoising head on a frozen Casanovo encoder with plain PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import warnings
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from casanovo_denoising.data import (
    DEFAULT_DATASET_REPO,
    PeakBudgetBatchSampler,
    build_denoising_datasets,
    collate_denoising,
)
from casanovo_denoising.metrics import denoising_metrics
from casanovo_denoising.model import CasanovoDenoiser, count_parameters

AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("xpu") if xpu_available else torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if name == "xpu" and not xpu_available:
        raise RuntimeError("--device xpu requested but XPU is not available")
    return torch.device(name)


def resolve_precision(precision: str, device: torch.device) -> str:
    """Return the precision actually used, falling back to fp32 if unsupported."""
    if precision == "fp32":
        return precision
    if device.type == "cuda":
        supported = precision == "fp16" or torch.cuda.is_bf16_supported()
    elif device.type == "xpu":
        supported = True
    else:
        supported = precision == "bf16"
    if not supported:
        warnings.warn(f"{precision} autocast is not supported on {device}; using fp32")
        return "fp32"
    return precision


def make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_loader(dataset, *, peak_pair_budget: int, seed: int, shuffle: bool, num_workers: int, device):
    sampler = PeakBudgetBatchSampler(
        lengths=dataset["num_peaks"],
        peak_pair_budget=peak_pair_budget,
        seed=seed,
        shuffle=shuffle,
    )
    loader = DataLoader(
        dataset.with_format("numpy", columns=["mz", "intensity", "labels"]),
        batch_sampler=sampler,
        collate_fn=collate_denoising,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return loader, sampler


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, autocast) -> dict[str, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in tqdm(loader, desc="Denoising eval"):
        batch = to_device(batch, device)
        with autocast():
            _, logits, valid = model(batch["mz"], batch["intensity"], batch["labels"])
        all_logits.append(logits[valid].float().cpu().numpy())
        all_labels.append(batch["labels"][valid].cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits), torch.from_numpy(labels)
    ).item()
    metrics = {"loss": loss, **denoising_metrics(logits, labels)}
    return {f"denoise/{name}": value for name, value in metrics.items()}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    amp_dtype = AMP_DTYPES.get(precision)

    def autocast():
        if amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_denoising_datasets(
        args.dataset_repo,
        train_split=args.train_split,
        validation_split=args.validation_split,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc if args.num_proc > 1 else None,
    )
    train_loader, train_sampler = make_loader(
        datasets["train"],
        peak_pair_budget=args.peak_pair_budget,
        seed=args.seed,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    validation_loader, _ = make_loader(
        datasets["validation"],
        peak_pair_budget=args.peak_pair_budget,
        seed=args.seed,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    model = CasanovoDenoiser.from_checkpoint(
        checkpoint,
        head_hidden_size=args.head_hidden_size,
        head_dropout=args.head_dropout,
    ).to(device)
    parameter_counts = {
        "encoder": count_parameters(model.encoder),
        "head": count_parameters(model.head),
        "total": count_parameters(model),
        "trainable": count_parameters(model, trainable_only=True),
    }
    for name, count in parameter_counts.items():
        print(f"{name} parameters: {count:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(device, enabled=precision == "fp16")

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        progress = tqdm(train_loader, desc=f"Denoising train (epoch {epoch + 1}/{args.epochs})")
        for batch in progress:
            batch = to_device(batch, device)
            with autocast():
                loss, _, _ = model(batch["mz"], batch["intensity"], batch["labels"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = evaluate(model, validation_loader, device, autocast)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    torch.save(model.head.state_dict(), args.output_dir / "head.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    run_config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256sum(checkpoint),
        "resolved_device": str(device),
        "resolved_precision": precision,
        "parameter_counts": parameter_counts,
        "num_train_spectra": len(datasets["train"]),
        "num_validation_spectra": len(datasets["validation"]),
        "versions": {
            "python": sys.version.split()[0],
            "casanovo": version("casanovo"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")


if __name__ == "__main__":
    main()
