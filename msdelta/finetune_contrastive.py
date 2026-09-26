"""Fine-tune the spectrum encoder contrastively, with KL to its own pretrained head.

    python -m msdelta.finetune_contrastive --args_file configs/finetune/contrastive-replicate-50m/training.args
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (HfArgumentParser, Trainer, TrainerCallback, TrainingArguments,
                          set_seed)

from msdelta import grouped_retrieval as gr
from msdelta.contrastive import (GroupBatchSampler, MSDeltaForContrastive, gradcache_step,
                                 group_separation_summary, retrieval_summary,
                                 subset_by_group)
from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.reranking import AlignmentCollator, peptide_key
from msdelta.wandb_distributed import init_wandb_run


@dataclass
class ContrastiveModelArguments:
    pretrained_path: str = ""
    pooling: str = "mean+max"
    temperature: float = 0.07
    kl_weight: float = 100.0


@dataclass
class ContrastiveDataArguments:
    dataset_repo: str
    dataset_format: str = "replicate"
    include_consensus: bool = False
    # Corpus whose peptides are dropped from a grouped dataset.
    exclude_peptides_from: str | None = None
    processor_name_or_path: str | None = None
    max_peaks: int = 512
    validation_fraction: float = 0.1
    split_seed: int = 0
    preprocessing_num_workers: int = 24
    max_samples: int = 0
    groups_per_batch: int = 6
    replicates: int = 4
    gradcache_chunk: int = 0


@dataclass
class ContrastiveTrainingArguments(TrainingArguments):
    wandb_project: str | None = None
    wandb_entity: str | None = None
    eval_alignment_rows: int = 2000


class SaveEncoderCallback(TrainerCallback):
    """Save the inner HF encoder into every checkpoint, so intermediate checkpoints are loadable."""

    def __init__(self, model: nn.Module, processor=None):
        self.model = model
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        target = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "encoder"
        self.model.model.save_pretrained(str(target))
        if self.processor is not None:
            self.processor.save_pretrained(str(target))


class ContrastiveTrainer(Trainer):
    """Standard Trainer, with the PK sampler and the loss components surfaced."""

    def __init__(self, *args, groups=None, groups_per_batch=12, replicates=4,
                 gradcache_chunk=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.groups_per_batch = groups_per_batch
        self.replicates = replicates
        self.gradcache_chunk = gradcache_chunk

    def _get_train_sampler(self, *args, **kwargs):
        # Batches come from the PK sampler in get_train_dataloader.
        return None

    def get_train_dataloader(self):
        sampler = GroupBatchSampler(self.groups, self.groups_per_batch,
                                    self.replicates, seed=self.args.seed)
        return DataLoader(self.train_dataset, batch_sampler=sampler,
                          collate_fn=self.data_collator,
                          num_workers=self.args.dataloader_num_workers,
                          pin_memory=self.args.dataloader_pin_memory)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """GradCache when a chunk size is set, otherwise the ordinary path."""
        if not self.gradcache_chunk:
            return super().training_step(model, inputs, num_items_in_batch)
        model.train()
        inputs = self._prepare_inputs(inputs)
        inner = model.module if hasattr(model, "module") else model
        outputs = gradcache_step(inner, inputs, self.gradcache_chunk,
                                 accelerator=getattr(self, "accelerator", None))
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({"contrastive": float(outputs["contrastive"]),
                      "kl": float(outputs["kl"])})
        return outputs["loss"].detach()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({"contrastive": float(outputs["contrastive"]),
                      "kl": float(outputs["kl"])})
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


def load_contrastive_datasets(data_args, processor) -> dict:
    """train + validation for either corpus format. See ContrastiveDataArguments."""
    if data_args.dataset_format == "grouped":
        # K above the group size would pair a spectrum with itself.
        members = 3 + int(data_args.include_consensus)
        if data_args.replicates > members:
            raise ValueError(f"replicates={data_args.replicates} but grouped analytes "
                             f"have {members} spectra (include_consensus="
                             f"{data_args.include_consensus}); set replicates <= {members}")
    datasets = gr.load_spectrum_datasets(
        data_args.dataset_format, data_args.dataset_repo, processor,
        include_consensus=data_args.include_consensus,
        exclude_peptides_from=data_args.exclude_peptides_from,
        num_proc=data_args.preprocessing_num_workers or None,
        validation_fraction=data_args.validation_fraction, seed=data_args.split_seed)
    if data_args.dataset_format != "grouped":
        return datasets
    # Drop train analytes left with a single spectrum by max_peaks.
    train = datasets["train"]
    ids = gr.group_ids(train)
    keep = np.flatnonzero(np.bincount(ids)[ids] >= 2)
    if len(keep) < len(train):
        print(f"[contrastive] dropped {len(train) - len(keep):,} train spectra left "
              f"alone in their group by max_peaks", flush=True)
        datasets["train"] = train.select(keep)
    return datasets


@dataclass
class ContrastiveCollator(AlignmentCollator):
    """Spectra plus an integer group id per row."""

    def __call__(self, features):
        batch = {k: v for k, v in super().__call__(features).items()
                 if k in ("mz", "log_intensity", "attention_mask")}
        keys = [peptide_key(f["peptide"], int(f.get("charge", 0))) for f in features]
        lookup = {key: index for index, key in enumerate(dict.fromkeys(keys))}
        batch["group"] = torch.tensor([lookup[k] for k in keys], dtype=torch.long)
        return batch


def main(argv: list[str] | None = None) -> int:
    parser = HfArgumentParser(
        (ContrastiveModelArguments, ContrastiveDataArguments, ContrastiveTrainingArguments)  # pyright: ignore[reportArgumentType]
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=argv, args_file_flag="--args_file")
    out_dir = Path(training_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if training_args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", training_args.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(out_dir))
    set_seed(training_args.seed)

    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks)
    encoder = MSDeltaForPreTraining.from_pretrained(model_args.pretrained_path)
    reference = (MSDeltaForPreTraining.from_pretrained(model_args.pretrained_path)
                 if model_args.kl_weight > 0 else None)
    model = MSDeltaForContrastive(encoder, reference, pooling=model_args.pooling,
                                  temperature=model_args.temperature,
                                  kl_weight=model_args.kl_weight)

    wandb_run = None
    if training_args.wandb_project:
        wandb_run = init_wandb_run(
            project=training_args.wandb_project, run_name=training_args.run_name,
            entity=training_args.wandb_entity,
            config={"model": asdict(model_args), "data": asdict(data_args)})
    try:
        with training_args.main_process_first(local=False, desc="contrastive data"):
            datasets = load_contrastive_datasets(data_args, processor)
        if data_args.max_samples:
            # Subset by group so the PK sampler still has full groups.
            datasets = {name: subset_by_group(
                split, data_args.max_samples,
                lambda row: peptide_key(row["peptide"], int(row.get("charge", 0))),
                min_members=data_args.replicates)
                for name, split in datasets.items()}
            if training_args.process_index == 0:
                print("[contrastive] subset by group: "
                      + " ".join(f"{k}={len(v):,}" for k, v in datasets.items()), flush=True)
        groups = np.unique(
            np.array([peptide_key(p, c) for p, c in
                      zip(datasets["train"]["peptide"], datasets["train"]["charge"])]),
            return_inverse=True)[1]
        if training_args.process_index == 0:
            print(f"[contrastive] " + " ".join(f"{k}={len(v):,}" for k, v in datasets.items())
                  + f" train_groups={len(set(groups.tolist())):,}", flush=True)

        collator = ContrastiveCollator(max_peptide_length=64,
                                       pad_spectra_to=data_args.max_peaks)
        trainer = ContrastiveTrainer(
            model=model, args=training_args, train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"), data_collator=collator,
            groups=groups, groups_per_batch=data_args.groups_per_batch,
            replicates=data_args.replicates,
            gradcache_chunk=data_args.gradcache_chunk)
        trainer.add_callback(SaveEncoderCallback(model, processor))
        # Trainer.train() does not read args.resume_from_checkpoint on its own.
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        if trainer.state.global_step == 0:
            raise SystemExit("trained 0 optimizer steps -- too few groups/rows for one batch?")

        if datasets.get("validation") is not None:
            before_after = group_separation_summary(
                model, datasets["validation"], collator, trainer.args.device,
                max_rows=training_args.eval_alignment_rows)
            if trainer.is_world_process_zero():
                print(f"[contrastive] separation: {before_after}", flush=True)
                trainer.log(before_after)
                trainer.save_metrics("separation", before_after)

            retrieval = retrieval_summary(
                model, datasets["validation"], collator, trainer.args.device,
                max_rows=training_args.eval_alignment_rows)
            if retrieval and trainer.is_world_process_zero():
                print(f"[contrastive] retrieval: {retrieval}", flush=True)
                trainer.log({k: v for k, v in retrieval.items()
                             if isinstance(v, (int, float))})
                trainer.save_metrics("retrieval", retrieval)

        if trainer.is_world_process_zero():
            model.model.save_pretrained(str(out_dir / "final"))
            processor.save_pretrained(str(out_dir / "final"))
            print(f"[contrastive] saved to {out_dir / 'final'}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
