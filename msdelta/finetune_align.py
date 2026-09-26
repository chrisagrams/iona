"""Train a peptide encoder into the frozen spectrum encoder's embedding space.

Only the peptide encoder learns. Watch `crossmodal/hit@1`, not the loss.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

from msdelta.finetune_denoise import subset_splits
from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.reranking import (
    POOLING_MODES,
    AlignmentCollator,
    PeptideEncoder,
    REPLICATE_REPO,
    SequenceAlignmentModel,
    attach_teacher_embeddings,
    build_alignment_datasets,
    build_alignment_model,
    cross_modal_metrics,
    group_separation_metrics,
    peptide_key,
)
from msdelta.wandb_distributed import init_wandb_run


@dataclass
class AlignModelArguments:
    pretrained_path: str
    pooling: str = "mean+max"
    sequence_hidden_size: int = 256
    sequence_num_layers: int = 4
    sequence_num_heads: int = 8
    sequence_dropout: float = 0.1
    max_peptide_length: int = 64
    # pool, cls or attn.
    sequence_readout: str = "pool"
    # "mse" or "lit" (cross-modal SupCon against the teacher, + mse_weight x MSE).
    align_loss: str = "mse"
    align_temperature: float = 0.05
    mse_weight: float = 0.0
    hard_negatives: int = 0
    neg_min_delta: float = 0.05
    # Mass-aware batches and/or hard negatives within +-neg_ppm.
    mass_batches: bool = False
    mass_batch_jitter: float = 0.5
    neg_source: str = "swap"
    neg_ppm: float = 20.0


@dataclass
class AlignDataArguments:
    processor_name_or_path: str | None = None
    dataset_repo: str = REPLICATE_REPO
    dataset_format: str = "replicate"
    include_consensus: bool = False
    exclude_replicate_peptides: bool = True
    preprocessing_num_workers: int = 24
    max_peaks: int = 512
    validation_fraction: float = 0.1
    max_samples: int = 0
    # Precompute the teacher's embeddings and drop it from the training graph.
    precompute_targets: bool = True
    # Output of `python -m msdelta.precompute_align`; when set, no teacher is loaded.
    target_cache: str | None = None


@dataclass
class AlignTrainingArguments(TrainingArguments):
    wandb_project: str | None = None
    wandb_entity: str | None = None
    eval_alignment_rows: int = 2000


class SequenceAlignmentTrainer(Trainer):
    """Optimise the student alone, and score ranking rather than the loss."""

    mass_sampler = None          # a reranking.MassBatchSampler, or None

    def get_train_dataloader(self):
        if self.mass_sampler is None:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader
        return DataLoader(self.train_dataset, batch_sampler=self.mass_sampler,
                          collate_fn=self.data_collator,
                          num_workers=self.args.dataloader_num_workers,
                          pin_memory=self.args.dataloader_pin_memory)

    def create_optimizer(self):
        """Stock optimizer, unless a frozen teacher is in the module and must be excluded."""
        if self.optimizer is not None:
            return self.optimizer
        if getattr(self.model, "spectrum_model", None) is None:
            return super().create_optimizer()
        cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        decay = [p for n, p in self.model.sequence_encoder.named_parameters()
                 if p.requires_grad and p.ndim > 1]
        no_decay = [p for n, p in self.model.sequence_encoder.named_parameters()
                    if p.requires_grad and p.ndim <= 1]
        kwargs.pop("weight_decay", None)
        self.optimizer = cls([{"params": decay, "weight_decay": self.args.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], **kwargs)
        return self.optimizer

    @torch.no_grad()
    def evaluate_alignment(self, dataset, max_rows: int = 2000) -> dict[str, float]:
        """Rank candidate sequences against each spectrum, one candidate per peptide."""
        model = self.model
        was_training = model.training
        model.eval()
        device = next(model.sequence_encoder.parameters()).device
        rows = list(dataset.select(range(min(len(dataset), max_rows))))
        step = max(1, self.args.per_device_eval_batch_size)
        spectra, sequences, peptides, charges = [], [], [], []
        try:
            for start in range(0, len(rows), step):
                chunk = rows[start : start + step]
                batch = {k: v.to(device) for k, v in self.data_collator(chunk).items()}
                out = model(**batch, return_dict=True)
                spectra.append(out["target"].cpu())
                sequences.append(out["embeddings"].cpu())
                peptides.extend(f["peptide"] for f in chunk)
                charges.extend(int(f.get("charge", 0)) for f in chunk)
        finally:
            model.train(was_training)
        if not spectra:
            return {}
        first: dict[str, int] = {}
        for index, peptide in enumerate(peptides):
            first.setdefault(peptide, index)
        keep = sorted(first.values())
        slot = {peptides[i]: n for n, i in enumerate(keep)}
        sequence_embeddings, spectrum_embeddings = torch.cat(sequences), torch.cat(spectra)
        metrics = cross_modal_metrics(
            sequence_embeddings[keep], spectrum_embeddings,
            np.array([slot[p] for p in peptides]), np.arange(len(keep)),
        )
        # Replicate separation in both towers.
        groups = np.array([peptide_key(p, c) for p, c in zip(peptides, charges)])
        codes = np.unique(groups, return_inverse=True)[1]
        metrics.update(group_separation_metrics(spectrum_embeddings, codes, "sep_spectrum"))
        metrics.update(group_separation_metrics(sequence_embeddings, codes, "sep_sequence"))
        return metrics


def main(argv: list[str] | None = None) -> int:
    parser = HfArgumentParser(
        (AlignModelArguments, AlignDataArguments, AlignTrainingArguments)  # pyright: ignore[reportArgumentType]
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=argv, args_file_flag="--args_file"
    )
    out_dir = Path(training_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if training_args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", training_args.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(out_dir))
    set_seed(training_args.seed)

    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks,
    )
    cached = Path(data_args.target_cache) if data_args.target_cache else None
    if cached and cached.exists():
        # The teacher is never constructed; its width comes from the manifest.
        manifest = dict(
            line.split(": ", 1)
            for line in (cached / "MANIFEST.txt").read_text().splitlines() if ": " in line)
        if manifest["pooling"] != model_args.pooling:
            sys.exit(f"cache was built with pooling={manifest['pooling']!r}, "
                     f"this run asks for {model_args.pooling!r}")
        model = SequenceAlignmentModel(
            None,
            PeptideEncoder(embedding_size=int(manifest["embedding_size"]),
                           hidden_size=model_args.sequence_hidden_size,
                           num_layers=model_args.sequence_num_layers,
                           num_heads=model_args.sequence_num_heads,
                           max_length=model_args.max_peptide_length,
                           dropout=model_args.sequence_dropout,
                           pooling=model_args.pooling,
                           readout=model_args.sequence_readout),
            pooling=model_args.pooling, loss=model_args.align_loss,
            temperature=model_args.align_temperature, mse_weight=model_args.mse_weight)
    else:
        teacher = MSDeltaForPreTraining.from_pretrained(model_args.pretrained_path)
        model = build_alignment_model(
            teacher, pooling=model_args.pooling,
            hidden_size=model_args.sequence_hidden_size,
            num_layers=model_args.sequence_num_layers,
            num_heads=model_args.sequence_num_heads,
            dropout=model_args.sequence_dropout,
            max_peptide_length=model_args.max_peptide_length,
        )
    collator = AlignmentCollator(max_peptide_length=model_args.max_peptide_length,
                                 hard_negatives=model_args.hard_negatives,
                                 neg_min_delta=model_args.neg_min_delta,
                                 neg_seed=training_args.seed,
                                 neg_source=model_args.neg_source,
                                 neg_ppm=model_args.neg_ppm)

    wandb_run = None
    if training_args.wandb_project:
        wandb_run = init_wandb_run(
            project=training_args.wandb_project, run_name=training_args.run_name,
            entity=training_args.wandb_entity,
            config={"model": asdict(model_args), "data": asdict(data_args),
                    "training": training_args.to_dict()},
        )
    try:
        if training_args.process_index == 0:
            print(f"[align] embedding size {model.sequence_encoder.projection[-1].out_features}",
                  flush=True)

        if cached and cached.exists():
            from datasets import load_from_disk
            datasets = {name: load_from_disk(str(cached / name))
                        for name in ("train", "validation") if (cached / name).exists()}
            if training_args.process_index == 0:
                print(f"[align] targets from {cached} "
                      + " ".join(f"{k}={len(v):,}" for k, v in datasets.items()),
                      flush=True)
        else:
            with training_args.main_process_first(local=False, desc="alignment data"):
                from msdelta.grouped_retrieval import load_spectrum_datasets
                datasets = load_spectrum_datasets(
                    data_args.dataset_format, data_args.dataset_repo, processor,
                    include_consensus=data_args.include_consensus,
                    exclude_replicate_peptides=data_args.exclude_replicate_peptides,
                    num_proc=data_args.preprocessing_num_workers or None,
                    validation_fraction=data_args.validation_fraction,
                    seed=training_args.seed,
                )
            datasets = subset_splits(datasets, data_args.max_samples,
                                     training_args.process_index)
            if data_args.precompute_targets:
                # In-process fallback; main_process_first avoids a datasets.map cache race.
                with training_args.main_process_first(local=False,
                                                      desc="teacher embeddings"):
                    datasets = attach_teacher_embeddings(
                        datasets, model.spectrum_model, model_args.pooling,
                        batch_size=training_args.per_device_eval_batch_size,
                        max_peptide_length=model_args.max_peptide_length)
                model.spectrum_model = None
        if training_args.process_index == 0:
            print("[align] " + " ".join(f"{k}={len(v):,}" for k, v in datasets.items()),
                  flush=True)

        if model_args.neg_source == "mass" or model_args.mass_batches:
            from msdelta.reranking import MassBatchSampler, MassNegativePool, peptide_neutral_mass
            train_peps = datasets["train"]["peptide"]
            if model_args.neg_source == "mass":
                collator.neg_pool = MassNegativePool(train_peps)
                print(f"[align] mass negatives: {len(collator.neg_pool.peptides):,} training "
                      f"peptides, +-{model_args.neg_ppm:g} ppm", flush=True)
        trainer = SequenceAlignmentTrainer(
            model=model, args=training_args, train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"), data_collator=collator,
        )
        if model_args.mass_batches:
            masses = [peptide_neutral_mass(p) for p in train_peps]
            trainer.mass_sampler = MassBatchSampler(
                masses, training_args.per_device_train_batch_size,
                jitter=model_args.mass_batch_jitter, seed=training_args.seed)
            print(f"[align] mass-bucketed batches of {training_args.per_device_train_batch_size} "
                  f"(jitter +-{model_args.mass_batch_jitter:g} Da)", flush=True)
        trainer.train()

        if trainer.is_world_process_zero() and datasets.get("validation") is not None:
            metrics = trainer.evaluate_alignment(
                datasets["validation"], max_rows=training_args.eval_alignment_rows
            )
            print(f"[align] cross-modal: {metrics}", flush=True)
            trainer.log(metrics)
            trainer.save_metrics("crossmodal", metrics)
        if trainer.is_world_process_zero():
            trainer.save_model(str(out_dir / "final"))
            print(f"[align] saved to {out_dir / 'final'}", flush=True)
        return 0
    except BaseException:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
            wandb_run = None
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    sys.exit(main())
