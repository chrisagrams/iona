"""Train a Prosit-style fragment-intensity head on a frozen MSDelta encoder.

Data are the Prosit 2020 HCD splits (figshare article 12937092), converted once
by ``scripts/convert_prosit_hdf5.py`` into memory-mapped ``.npy`` arrays.

    uv run msdelta-intensity \
        --data-dir /mnt/vault-1/k8/prosit-dataset/arrays \
        --checkpoint runs/<run>/final \
        --output-dir runs/intensity/<name> \
        --wandb-project msdelta-intensity

``--encoder-init`` selects the controls that make the headline number
interpretable: ``random`` freezes an untrained encoder of the same shape, and
``none`` drops the encoder states so the head sees only Prosit's metadata.

By default the encoder is frozen and only the head trains. ``--finetune-encoder``
trains the encoder as well, at its own ``--encoder-learning-rate``; add
``--gradient-checkpointing`` (or lower ``--ion-pair-budget``) if memory runs out.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    EvalPrediction,
    FeatureExtractionMixin,
    Trainer,
    TrainingArguments,
    set_seed,
)

import wandb
from msdelta.callbacks import WalltimeCheckpointCallback
from msdelta.chemistry import PROSIT_RESIDUE_MASSES, PROTON_MASS, WATER_MASS
from msdelta.configuration_msdelta import MSDeltaConfig, MSDeltaIntensityPredictionConfig
from msdelta.denoising import PeakBudgetBatchSampler
from msdelta.modeling_msdelta import (
    MSDeltaForIntensityPrediction,
    MSDeltaForPreTraining,
    MSDeltaModel,
    masked_spectral_angle,
)
from msdelta.wandb_distributed import init_wandb_run

SPLITS = ("train", "val", "holdout")


def prosit_fragment_ladder(
    sequence_integer: torch.Tensor,
    precursor_charge: torch.Tensor,
    max_fragment_charge: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Theoretical b/y fragment m/z in Prosit's flat slot layout.

    Slot ``6 * i + 3 * s + (z - 1)`` holds ion number ``i + 1`` of series ``s``
    (0 = y, 1 = b) at fragment charge ``z``. An ion is possible when its number
    is below the peptide length and ``z`` does not exceed the precursor charge.
    Returns ``(mz, valid, peptide_length)``; impossible slots have m/z 0.
    """
    sequence = sequence_integer.long()
    batch_size, max_length = sequence.shape
    residue_masses = torch.tensor(
        PROSIT_RESIDUE_MASSES, dtype=torch.float64, device=sequence.device
    )
    length = (sequence > 0).sum(dim=-1)
    prefix = residue_masses[sequence].cumsum(dim=-1)
    total = prefix.gather(1, (length - 1).clamp_min(0).unsqueeze(-1))
    ion_number = torch.arange(1, max_length, device=sequence.device)
    b_neutral = prefix[:, :-1]
    y_index = (length.unsqueeze(-1) - 1 - ion_number).clamp_min(0)
    y_neutral = total - prefix.gather(1, y_index) + WATER_MASS
    neutral = torch.stack([y_neutral, b_neutral], dim=-1).unsqueeze(-1)
    charge = torch.arange(1, max_fragment_charge + 1, device=sequence.device)
    mz = (neutral + charge * PROTON_MASS) / charge
    valid = (ion_number[None, :, None, None] < length[:, None, None, None]) & (
        charge <= precursor_charge.long()[:, None, None, None]
    )
    valid = valid.expand_as(mz).reshape(batch_size, -1)
    mz = mz.reshape(batch_size, -1).masked_fill(~valid, 0.0)
    return mz.float(), valid, length


class MSDeltaIntensityProcessor(FeatureExtractionMixin):
    """Turn Prosit rows into prior-weighted theoretical ion lists for the encoder.

    MSDelta tokens carry no m/z, so a ladder of identical intensities would make
    every hidden state identical. Each possible ion therefore enters with the
    training-set mean intensity for its (precursor charge, peptide length, slot)
    cell, and only possible ions are kept, packed to the front of each row.

    The prior is part of the trained model's input contract, so it is stored in
    ``preprocessor_config.json`` and saved with every checkpoint.
    """

    model_input_names = [
        "mz",
        "log_intensity",
        "attention_mask",
        "ion_slots",
        "precursor_charge",
        "peptide_length",
        "collision_energy",
    ]

    def __init__(self, intensity_prior, max_fragment_charge: int = 3, **kwargs):
        # Round once here so a reloaded processor reproduces training inputs exactly
        # and the JSON stays compact.
        prior = np.round(np.asarray(intensity_prior, dtype=np.float64), 6)
        if prior.ndim != 3:
            raise ValueError("intensity_prior must be (precursor charge, length, slot)")
        if prior.shape[2] != (prior.shape[1] - 1) * 2 * max_fragment_charge:
            raise ValueError("intensity_prior slot count does not match Prosit's layout")
        super().__init__(intensity_prior=prior, max_fragment_charge=max_fragment_charge, **kwargs)
        self.intensity_prior = prior
        self.max_fragment_charge = max_fragment_charge

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features must not be empty")
        sequence = torch.stack(
            [torch.as_tensor(feature["sequence_integer"]) for feature in features]
        ).long()
        charge = torch.as_tensor([int(feature["precursor_charge"]) for feature in features])
        mz, valid, length = prosit_fragment_ladder(sequence, charge, self.max_fragment_charge)
        prior = torch.from_numpy(self.intensity_prior).float()[charge - 1, length - 1]
        log_intensity = torch.log1p(prior.masked_fill(~valid, 0.0))
        log_intensity = log_intensity / log_intensity.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        counts = valid.sum(dim=-1)
        order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
        ion_slots = order[:, : max(int(counts.max()), 1)]
        batch = {
            "mz": mz.gather(1, ion_slots),
            "log_intensity": log_intensity.gather(1, ion_slots),
            "attention_mask": (torch.arange(ion_slots.shape[1]) < counts.unsqueeze(-1)).long(),
            "ion_slots": ion_slots,
            "precursor_charge": charge,
            "peptide_length": length,
            "collision_energy": torch.as_tensor(
                [float(feature["collision_energy"]) for feature in features]
            ),
        }
        if "labels" in features[0]:
            labels = torch.stack(
                [torch.as_tensor(feature["labels"], dtype=torch.float32) for feature in features]
            )
            batch["labels"] = labels.masked_fill(~valid, -1.0)
        return batch


class PrositIntensityDataset(Dataset):
    """Memory-mapped Prosit split, optionally restricted to a row subset."""

    def __init__(self, directory: Path | str, indices: np.ndarray | None = None):
        directory = Path(directory)
        self.sequence_integer = np.load(directory / "sequence_integer.npy", mmap_mode="r")
        self.precursor_charge = np.load(directory / "precursor_charge.npy", mmap_mode="r")
        self.collision_energy = np.load(directory / "collision_energy.npy", mmap_mode="r")
        self.intensities = np.load(directory / "intensities.npy", mmap_mode="r")
        self.indices = np.arange(len(self.intensities)) if indices is None else np.sort(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def ion_counts(self) -> list[int]:
        """Possible b/y ions per spectrum: the encoder's sequence length."""
        length = (self.sequence_integer[self.indices] > 0).sum(axis=1)
        charge = np.minimum(self.precursor_charge[self.indices], 3)
        return ((length - 1) * 2 * charge).tolist()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        # Read whole batches in row order; memory-mapped fancy indexing is one
        # pass over the pages instead of one seek per spectrum.
        rows = self.indices[np.asarray(indices)]
        order = np.argsort(rows)
        sorted_rows = rows[order]
        batch = {
            "sequence_integer": torch.from_numpy(
                self.sequence_integer[sorted_rows].astype(np.int64)
            ),
            "precursor_charge": torch.from_numpy(
                self.precursor_charge[sorted_rows].astype(np.int64)
            ),
            "collision_energy": torch.from_numpy(self.collision_energy[sorted_rows].copy()),
            "labels": torch.from_numpy(self.intensities[sorted_rows].copy()),
        }
        restore = np.empty_like(order)
        restore[order] = np.arange(len(order))
        return [{name: values[i] for name, values in batch.items()} for i in restore]

    def subset(self, n_rows: int | None, seed: int) -> PrositIntensityDataset:
        """Return a seeded random subset of at most ``n_rows`` spectra."""
        if n_rows is None or n_rows >= len(self):
            return self
        chosen = np.random.default_rng(seed).choice(self.indices, n_rows, replace=False)
        subset = object.__new__(PrositIntensityDataset)
        subset.__dict__.update(self.__dict__)
        subset.indices = np.sort(chosen)
        return subset

    def shard(self, index: int, count: int) -> PrositIntensityDataset:
        """Return every ``count``-th spectrum starting at ``index``."""
        if count == 1:
            return self
        shard = object.__new__(PrositIntensityDataset)
        shard.__dict__.update(self.__dict__)
        shard.indices = self.indices[index::count]
        return shard


class IntensityTrainer(Trainer):
    """Batch spectra by padded ion-pair count; attention bias memory is O(N^2)."""

    def __init__(
        self,
        *args,
        ion_pair_budget: int,
        ion_budget: int | None = None,
        encoder_learning_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ion_pair_budget = ion_pair_budget
        self.ion_budget = ion_budget
        self.encoder_learning_rate = encoder_learning_rate

    def _save_rng_state(self, output_dir: str) -> None:
        # As in MSDeltaTrainer: many ranks racing os.makedirs on a distributed
        # filesystem can raise a spurious FileExistsError, so rank 0 creates it.
        if self.args.world_size > 1:
            if self.args.process_index == 0:
                os.makedirs(output_dir, exist_ok=True)
            self.accelerator.wait_for_everyone()
        super()._save_rng_state(output_dir)

    def create_optimizer(self, model=None):
        # Build the stock optimizer, then give encoder parameters their own
        # learning rate: a fresh head wants ~1e-3, pretrained weights far less.
        optimizer = super().create_optimizer(model)
        if self.encoder_learning_rate is None:
            return optimizer
        opt_model = cast(MSDeltaForIntensityPrediction, self.model if model is None else model)
        encoder_ids = {id(p) for p in opt_model.msdelta.parameters()}
        groups = []
        for group in optimizer.param_groups:
            settings = {k: v for k, v in group.items() if k != "params"}
            head = [p for p in group["params"] if id(p) not in encoder_ids]
            encoder = [p for p in group["params"] if id(p) in encoder_ids]
            if head:
                groups.append({**settings, "params": head})
            if encoder:
                groups.append({**settings, "params": encoder, "lr": self.encoder_learning_rate})
        optimizer.param_groups.clear()
        for group in groups:
            optimizer.add_param_group(group)
        return optimizer

    def _budget_loader(self, dataset: PrositIntensityDataset, num_processes: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_sampler=PeakBudgetBatchSampler(
                lengths=dataset.ion_counts(),
                peak_pair_budget=self.ion_pair_budget,
                seed=self.args.data_seed or self.args.seed,
                num_processes=num_processes,
                peak_budget=self.ion_budget,
            ),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_train_dataloader(self) -> DataLoader:
        dataset = cast(PrositIntensityDataset, self.train_dataset)
        dataloader = self._budget_loader(dataset, self.accelerator.num_processes)
        self.accelerator.even_batches = False
        return self.accelerator.prepare(dataloader)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        # Gathering eval predictions needs equal batches on every rank; the
        # variable-size training loader turns that off.
        self.accelerator.even_batches = True
        return super().get_eval_dataloader(eval_dataset)


def fit_intensity_prior(
    dataset: PrositIntensityDataset,
    config: MSDeltaIntensityPredictionConfig,
    *,
    batch_size: int = 65_536,
) -> torch.Tensor:
    """Mean max-normalized intensity per (precursor charge, peptide length, slot).

    Cells never seen in ``dataset`` fall back to the (charge, slot) mean, then
    to the slot mean, so every possible ion gets a finite prior.
    """
    shape = (config.max_precursor_charge, config.max_sequence_length, config.num_ion_slots)
    sums = np.zeros(shape)
    counts = np.zeros(shape)
    for start in range(0, len(dataset), batch_size):
        rows = dataset.indices[start : start + batch_size]
        intensities = dataset.intensities[rows].astype(np.float64)
        charge = dataset.precursor_charge[rows].astype(np.int64) - 1
        length = (dataset.sequence_integer[rows] > 0).sum(axis=1) - 1
        scored = intensities >= 0
        peak = np.where(scored, intensities, 0.0).max(axis=1, keepdims=True)
        keep = peak[:, 0] > 0
        normalized = np.where(scored, intensities, 0.0) / np.maximum(peak, 1e-12)
        np.add.at(sums, (charge[keep], length[keep]), normalized[keep])
        np.add.at(counts, (charge[keep], length[keep]), scored[keep])
    prior = sums / np.maximum(counts, 1)
    by_charge = sums.sum(axis=1) / np.maximum(counts.sum(axis=1), 1)
    by_slot = sums.sum(axis=(0, 1)) / np.maximum(counts.sum(axis=(0, 1)), 1)
    prior = np.where(counts > 0, prior, by_charge[:, None, :])
    prior = np.where(counts.sum(axis=1, keepdims=True) > 0, prior, by_slot)
    return torch.from_numpy(prior).float()


def summarize_spectral_angles(angles: np.ndarray, charges: np.ndarray, prefix: str) -> dict:
    """Mean/median angle overall and median per precursor charge."""
    metrics = {
        f"{prefix}spectral_angle_mean": float(angles.mean()),
        f"{prefix}spectral_angle_median": float(np.median(angles)),
        f"{prefix}n": int(len(angles)),
    }
    for charge in np.unique(charges):
        selected = angles[charges == charge]
        metrics[f"{prefix}spectral_angle_median_charge_{int(charge)}"] = float(np.median(selected))
        metrics[f"{prefix}n_charge_{int(charge)}"] = int(len(selected))
    return metrics


def _scored(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scored = labels >= 0
    return scored, (labels.clamp_min(0.0) * scored).sum(dim=-1) > 0


def intensity_metrics(prediction: EvalPrediction) -> dict[str, float]:
    """Trainer metrics on an in-memory validation subset."""
    predicted = torch.as_tensor(np.asarray(prediction.predictions))
    labels = torch.as_tensor(np.asarray(prediction.label_ids))
    scored, has_signal = _scored(labels)
    angles = masked_spectral_angle(predicted.clamp_min(0.0), labels, scored & (predicted >= 0))
    angles = angles[has_signal].numpy()
    return {
        "spectral_angle_mean": float(angles.mean()),
        "spectral_angle_median": float(np.median(angles)),
    }


@torch.no_grad()
def evaluate_split(
    model: MSDeltaForIntensityPrediction,
    dataset: PrositIntensityDataset,
    processor: MSDeltaIntensityProcessor,
    *,
    ion_pair_budget: int,
    num_workers: int,
    autocast_dtype: torch.dtype | None,
) -> dict[str, np.ndarray]:
    """Stream a split through the model and the input prior, keeping only angles."""
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_sampler=PeakBudgetBatchSampler(dataset.ion_counts(), ion_pair_budget, seed=0),
        num_workers=num_workers,
        collate_fn=processor,
    )
    intensity_prior = torch.from_numpy(processor.intensity_prior).float().to(device)
    model_angles, prior_angles, charges = [], [], []
    for batch in loader:
        batch = {name: values.to(device, non_blocking=True) for name, values in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            predicted = model(**batch).intensities
        scored, has_signal = _scored(labels)
        valid = predicted >= 0
        charge = batch["precursor_charge"]
        prior = intensity_prior[charge - 1, batch["peptide_length"] - 1]
        model_angles.append(masked_spectral_angle(predicted, labels, scored & valid)[has_signal])
        prior_angles.append(masked_spectral_angle(prior, labels, scored & valid)[has_signal])
        charges.append(charge[has_signal])
    return {
        "model": torch.cat(model_angles).cpu().numpy(),
        "prior": torch.cat(prior_angles).cpu().numpy(),
        "charge": torch.cat(charges).cpu().numpy(),
    }


def gather_arrays(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Concatenate each process's evaluation arrays (identity when not distributed)."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return parts
    gathered: list[dict[str, np.ndarray] | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, parts)
    return {name: np.concatenate([cast(dict, part)[name] for part in gathered]) for name in parts}


def load_encoder(checkpoint: Path | None, encoder_init: str) -> MSDeltaModel:
    """Load the pretrained encoder, or build a same-shaped random one."""
    if checkpoint is None:
        if encoder_init == "pretrained":
            raise ValueError("--checkpoint is required with --encoder-init pretrained")
        return MSDeltaModel(MSDeltaConfig())
    if encoder_init == "pretrained":
        return MSDeltaForPreTraining.from_pretrained(checkpoint).msdelta
    return MSDeltaModel(MSDeltaConfig.from_pretrained(checkpoint))


def main(argv: list[str] | None = None) -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.xpu.is_available():
        torch.xpu.set_device(local_rank)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--encoder-init", choices=("pretrained", "random", "none"), default="pretrained"
    )
    parser.add_argument("--head-hidden-size", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--prior-samples", type=int, default=2_000_000)
    parser.add_argument("--eval-samples", type=int, default=50_000)
    parser.add_argument("--max-holdout-samples", type=int)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--ion-pair-budget",
        type=int,
        default=2**21,
        help="max padded ions^2 summed over a training batch (bounds bias memory)",
    )
    parser.add_argument(
        "--ion-budget",
        type=int,
        help="also cap padded ions (spectra x longest) per training batch; "
        "needed with --finetune-encoder, where per-ion activations dominate memory",
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="head learning rate")
    parser.add_argument(
        "--finetune-encoder",
        action="store_true",
        help="train the encoder too (default: frozen encoder, head only)",
    )
    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=1e-5,
        help="encoder learning rate with --finetune-encoder",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="recompute encoder activations in backward to save memory when fine-tuning",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--ddp-backend", help="torch.distributed backend, e.g. xccl on Aurora (default: auto)"
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--wandb-project", help="log to this W&B project (off when unset)")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args(argv)
    if args.finetune_encoder and args.encoder_init == "none":
        parser.error(
            "--finetune-encoder needs an encoder; it cannot be used with --encoder-init none"
        )

    wandb_run = None
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(args.output_dir))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = init_wandb_run(
            project=args.wandb_project,
            run_name=args.run_name or args.output_dir.name,
            config={},
            entity=args.wandb_entity,
        )
    if wandb_run is not None:
        # A resumed job reattaches to the same run; its stored config comes back
        # JSON-round-tripped (e.g. int keys as strings), so allow value changes.
        wandb_run.config.update(
            {"cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
            allow_val_change=True,
        )
    try:
        return train_and_evaluate(args, wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def train_and_evaluate(args: argparse.Namespace, wandb_run: wandb.Run | None) -> int:
    set_seed(args.seed)
    data = {split: PrositIntensityDataset(args.data_dir / split) for split in SPLITS}
    train = data["train"].subset(args.max_train_samples, args.seed)
    validation = data["val"].subset(args.eval_samples, args.seed)
    holdout = data["holdout"].subset(args.max_holdout_samples, args.seed)

    encoder = load_encoder(args.checkpoint, args.encoder_init)
    config = MSDeltaIntensityPredictionConfig(
        encoder=cast(MSDeltaConfig, encoder.config),
        head_hidden_size=args.head_hidden_size,
        head_dropout=args.head_dropout,
        use_encoder_states=args.encoder_init != "none",
    )
    model = MSDeltaForIntensityPrediction(
        config, encoder=encoder, freeze_encoder=not args.finetune_encoder
    )
    processor = MSDeltaIntensityProcessor(
        fit_intensity_prior(train.subset(args.prior_samples, args.seed + 1), config),
        max_fragment_charge=config.max_fragment_charge,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}; train spectra: {len(train):,}")
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "model": config.to_dict(),
                "trainable_parameters": trainable,
                "splits": {"train": len(train), "val": len(validation), "holdout": len(holdout)},
            },
            allow_val_change=True,
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        run_name=args.run_name or args.output_dir.name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_ratio,  # a float < 1 is a fraction of total steps
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=50,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.num_workers,
        bf16=args.bf16,
        seed=args.seed,
        data_seed=args.seed,
        report_to=["wandb"] if args.wandb_project else [],
        gradient_checkpointing=args.gradient_checkpointing and args.finetune_encoder,
        ddp_find_unused_parameters=False,
        ddp_backend=args.ddp_backend,
    )
    trainer = IntensityTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=processor,
        processing_class=processor,
        compute_metrics=intensity_metrics,
        ion_pair_budget=args.ion_pair_budget,
        ion_budget=args.ion_budget,
        encoder_learning_rate=args.encoder_learning_rate if args.finetune_encoder else None,
    )
    if deadline := os.environ.get("MSDELTA_JOB_DEADLINE_EPOCH"):
        margin = float(os.environ.get("MSDELTA_CHECKPOINT_MARGIN_SECONDS", "900"))
        trainer.add_callback(WalltimeCheckpointCallback(float(deadline), margin))
    trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )
    if trainer.state.global_step < trainer.state.max_steps:
        # Stopped at the walltime margin after saving a checkpoint; the holdout
        # pass would not fit, so resume and evaluate in the next job.
        print(f"stopped early at step {trainer.state.global_step}; resume to finish")
        return 0
    trainer.save_model()

    # Each process scores its own slice of the holdout; rank 0 reports.
    accelerator = trainer.accelerator
    angles = gather_arrays(
        evaluate_split(
            model,
            holdout.shard(accelerator.process_index, accelerator.num_processes),
            processor,
            ion_pair_budget=args.ion_pair_budget,
            num_workers=args.num_workers,
            autocast_dtype=torch.bfloat16 if args.bf16 else None,
        )
    )
    if not trainer.is_world_process_zero():
        return 0
    metrics = {
        **summarize_spectral_angles(angles["model"], angles["charge"], "holdout/"),
        **summarize_spectral_angles(angles["prior"], angles["charge"], "holdout/prior_"),
    }
    prosit = np.load(args.data_dir / "holdout" / "prosit_spectral_angle.npy", mmap_mode="r")
    metrics["holdout/prosit_rnn_spectral_angle_median"] = float(np.median(prosit[holdout.indices]))
    metrics["encoder_init"] = args.encoder_init
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["world_size"] = accelerator.num_processes
    print(json.dumps(metrics, indent=2))
    (args.output_dir / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2))
    if wandb_run is not None:
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        wandb_run.log({**numeric, "train/global_step": trainer.state.global_step})
        wandb_run.summary.update(numeric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
