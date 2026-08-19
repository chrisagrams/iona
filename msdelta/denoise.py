"""Post-train a peak-level signal/noise classifier on a frozen MSDelta encoder.

Example:
    msdelta-denoise \
      --encoder-config configs/scale_S.yaml \
      --encoder-checkpoint runs/scale_S/final \
      --output-dir runs/denoise-scale-S
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from safetensors.torch import load_file
from torch import nn
from transformers import Trainer, TrainingArguments

from .config import ModelArgs
from .data import PreprocessConfig
from .model import MSEncoder


def preprocess_labeled_spectrum(example: dict, cfg: PreprocessConfig) -> dict:
    """Apply encoder preprocessing while preserving peak/label alignment."""
    mz = torch.as_tensor(example["mz"], dtype=torch.float32)
    intensity = torch.as_tensor(example["intensity"], dtype=torch.float32)
    labels = torch.as_tensor(example["signal"], dtype=torch.float32)
    if not (mz.numel() == intensity.numel() == labels.numel()):
        raise ValueError("mz, intensity, and signal must have equal lengths")
    if intensity.numel() == 0 or float(intensity.max()) <= 0:
        return {"mz": [], "log_int": [], "labels": []}

    keep = intensity >= cfg.intensity_threshold_frac * intensity.max()
    mz, intensity, labels = mz[keep], intensity[keep], labels[keep]
    if mz.numel() > cfg.top_n:
        indices = torch.topk(intensity, cfg.top_n, sorted=False).indices
        mz, intensity, labels = mz[indices], intensity[indices], labels[indices]

    log_int = torch.log1p(intensity)
    log_int = log_int / log_int.max().clamp_min(1e-8)
    return {"mz": mz.tolist(), "log_int": log_int.tolist(), "labels": labels.tolist()}


class DenoiseCollator:
    """Pad spectra; -100 labels mark peaks excluded from loss and metrics."""

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        lengths = [len(row["mz"]) for row in features]
        width = max(max(lengths, default=0), 1)
        batch = len(features)
        mz = torch.zeros(batch, width, dtype=torch.float32)
        log_int = torch.zeros_like(mz)
        labels = torch.full((batch, width), -100.0, dtype=torch.float32)
        padding = torch.ones(batch, width, dtype=torch.bool)
        for i, (row, length) in enumerate(zip(features, lengths)):
            if not length:
                continue
            mz[i, :length] = torch.as_tensor(row["mz"])
            log_int[i, :length] = torch.as_tensor(row["log_int"])
            labels[i, :length] = torch.as_tensor(row["labels"])
            padding[i, :length] = False
        return {"mz": mz, "log_int": log_int,
                "key_padding_mask": padding, "labels": labels}


class PeakSignalHead(nn.Module):
    """Small per-token MLP producing one signal logit per peak."""

    def __init__(self, d_model: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens).squeeze(-1)


class MSDeltaForDenoising(nn.Module):
    """Frozen encoder plus trainable peak-level binary classifier."""

    def __init__(self, encoder: MSEncoder, hidden_dim: int = 128, dropout: float = 0.1,
                 pos_weight: float | None = None):
        super().__init__()
        self.encoder = encoder
        self.classifier = PeakSignalHead(encoder.cfg.d_model, hidden_dim, dropout)
        self.pos_weight = pos_weight
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Trainer calls model.train() every step; frozen dropout must stay off.
        self.encoder.eval()
        return self

    def forward(self, mz: torch.Tensor, log_int: torch.Tensor,
                key_padding_mask: torch.Tensor, labels: torch.Tensor | None = None):
        with torch.no_grad():
            tokens = self.encoder(mz, log_int, key_padding_mask)
        logits = self.classifier(tokens).float()
        output = {"logits": logits}
        if labels is not None:
            valid = labels != -100
            if not valid.any():
                loss = logits.sum() * 0.0
            else:
                weight = None if self.pos_weight is None else logits.new_tensor(self.pos_weight)
                loss = F.binary_cross_entropy_with_logits(
                    logits[valid], labels[valid].float(), pos_weight=weight)
            output["loss"] = loss
        return output


def _model_args_from_yaml(path: Path) -> ModelArgs:
    raw = yaml.safe_load(path.read_text()) or {}
    fields = ModelArgs.__dataclass_fields__
    return ModelArgs(**{key: value for key, value in raw.items() if key in fields})


def load_encoder(config_path: Path, checkpoint: Path) -> MSEncoder:
    """Load an encoder from a Trainer directory or a direct weight file."""
    encoder = MSEncoder(_model_args_from_yaml(config_path).to_model_config())
    if checkpoint.is_dir():
        safe = checkpoint / "model.safetensors"
        binary = checkpoint / "pytorch_model.bin"
        checkpoint = safe if safe.exists() else binary
    if not checkpoint.exists():
        raise FileNotFoundError(f"no model weights found at {checkpoint}")
    state = load_file(str(checkpoint)) if checkpoint.suffix == ".safetensors" else torch.load(
        checkpoint, map_location="cpu", weights_only=True)
    encoder_state = {
        key.removeprefix("module.").removeprefix("encoder."): value
        for key, value in state.items()
        if key.removeprefix("module.").startswith("encoder.")
    }
    # Also accept an encoder-only state dict.
    if not encoder_state:
        encoder_state = {key.removeprefix("module."): value for key, value in state.items()}
    encoder.load_state_dict(encoder_state, strict=True)
    return encoder


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    valid = labels != -100
    y = labels[valid].astype(np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-logits[valid]))
    predictions = probabilities >= 0.5
    tp = int(((predictions == 1) & (y == 1)).sum())
    fp = int(((predictions == 1) & (y == 0)).sum())
    fn = int(((predictions == 0) & (y == 1)).sum())
    tn = int(((predictions == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-config", required=True, type=Path)
    parser.add_argument("--encoder-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", default="chrisagrams/ms-denoise-12k")
    parser.add_argument("--output-dir", default="runs/denoise", type=Path)
    parser.add_argument("--top-n", default=None, type=int)
    parser.add_argument("--intensity-threshold-frac", default=0.0, type=float)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--pos-weight", default=None, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=10.0, type=float)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-2, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--wandb-project", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_args = _model_args_from_yaml(args.encoder_config)
    pp = PreprocessConfig(
        intensity_threshold_frac=args.intensity_threshold_frac,
        top_n=args.top_n or model_args.max_peaks,
    )
    dataset = load_dataset(args.dataset)
    dataset = dataset.map(
        lambda row: preprocess_labeled_spectrum(row, pp),
        remove_columns=dataset["train"].column_names,
        desc="preprocess labeled spectra",
    ).filter(lambda row: len(row["mz"]) > 0, desc="drop empty spectra")

    model = MSDeltaForDenoising(
        load_encoder(args.encoder_config, args.encoder_checkpoint),
        hidden_dim=args.hidden_dim, dropout=args.dropout, pos_weight=args.pos_weight,
    )
    report_to = ["wandb"] if args.wandb_project else []
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
        report_to=report_to,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        data_collator=DenoiseCollator(), compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
