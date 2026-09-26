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
from datasets import load_from_disk
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

from msdelta.finetune_denoise import subset_splits
from msdelta.grouped_retrieval import load_spectrum_datasets
from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.reranking import (
    AlignmentCollator,
    PeptideEncoder,
    REPLICATE_REPO,
    SequenceAlignmentModel,
    attach_teacher_embeddings,
    cross_modal_metrics,
    group_separation_metrics,
    peptide_key,
    pooled_width,
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
    # Output of `python scripts/precompute_align.py`; when set, no teacher is loaded.
    target_cache: str | None = None


@dataclass
class AlignTrainingArguments(TrainingArguments):
    wandb_project: str | None = None
    wandb_entity: str | None = None
    eval_alignment_rows: int = 2000


class SequenceAlignmentTrainer(Trainer):
    """Scores ranking rather than the loss."""

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
                out = model(**batch)
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
    teacher = None
    if cached and cached.exists():
        # The teacher is never constructed; its width comes from the manifest.
        manifest = dict(
            line.split(": ", 1)
            for line in (cached / "MANIFEST.txt").read_text().splitlines() if ": " in line)
        if manifest["pooling"] != model_args.pooling:
            sys.exit(f"cache was built with pooling={manifest['pooling']!r}, "
                     f"this run asks for {model_args.pooling!r}")
        embedding_size = int(manifest["embedding_size"])
    else:
        teacher = MSDeltaForPreTraining.from_pretrained(model_args.pretrained_path)
        embedding_size = pooled_width(teacher.config.hidden_size, model_args.pooling)
    model = SequenceAlignmentModel(
        PeptideEncoder(embedding_size=embedding_size,
                       hidden_size=model_args.sequence_hidden_size,
                       num_layers=model_args.sequence_num_layers,
                       num_heads=model_args.sequence_num_heads,
                       max_length=model_args.max_peptide_length,
                       dropout=model_args.sequence_dropout,
                       pooling=model_args.pooling))
    collator = AlignmentCollator(max_peptide_length=model_args.max_peptide_length)
    # `target` is the label, so evaluation reports eval_loss; no metrics need the predictions.
    training_args.label_names = ["target"]
    training_args.prediction_loss_only = True

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

        if teacher is None:
            datasets = {name: load_from_disk(str(cached / name))
                        for name in ("train", "validation") if (cached / name).exists()}
            if training_args.process_index == 0:
                print(f"[align] targets from {cached} "
                      + " ".join(f"{k}={len(v):,}" for k, v in datasets.items()),
                      flush=True)
        else:
            with training_args.main_process_first(local=False, desc="alignment data"):
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
            # main_process_first avoids a datasets.map cache race.
            with training_args.main_process_first(local=False, desc="teacher embeddings"):
                datasets = attach_teacher_embeddings(
                    datasets, teacher, model_args.pooling,
                    batch_size=training_args.per_device_eval_batch_size,
                    max_peptide_length=model_args.max_peptide_length)
            del teacher
        if training_args.process_index == 0:
            print("[align] " + " ".join(f"{k}={len(v):,}" for k, v in datasets.items()),
                  flush=True)

        trainer = SequenceAlignmentTrainer(
            model=model, args=training_args, train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"), data_collator=collator,
        )
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
