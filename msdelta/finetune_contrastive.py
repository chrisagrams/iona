"""Fine-tune the spectrum encoder contrastively, with KL to its own pretrained head.

    python -m msdelta.finetune_contrastive --args_file configs/finetune-contrastive-50m/training.args

Produces an encoder whose embedding space separates peptides, which is what the
alignment tower needs and what the pretrained checkpoint does not provide -- see
msdelta/contrastive.py for the measurement that motivated this.

The resulting checkpoint is then the `--pretrained_path` for the alignment run.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

from msdelta.contrastive import (GroupBatchSampler, MSDeltaForContrastive,
                                 embedding_size, group_separation_summary,
                                 subset_by_group)
from msdelta.finetune_denoise import MemoryProbe, load_description, select_device, subset_splits
from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.reranking import (AlignmentCollator, REPLICATE_REPO, build_alignment_datasets,
                               group_separation_metrics, peptide_key)
from msdelta.wandb_distributed import init_wandb_run


@dataclass
class ContrastiveModelArguments:
    pretrained_path: str = field(default="", metadata={"help": "encoder to start from"})
    pooling: str = "mean+max"
    temperature: float = field(default=0.07, metadata={"help": "SupCon temperature"})
    kl_weight: float = field(
        default=1.0,
        metadata={"help": "weight on KL to the frozen pretrained head. 0 disables the "
                          "regulariser and lets the encoder forget the chemistry."})


@dataclass
class ContrastiveDataArguments:
    dataset_repo: str = REPLICATE_REPO
    processor_name_or_path: str | None = None
    max_peaks: int = 512
    validation_fraction: float = 0.1
    preprocessing_num_workers: int = 24
    max_samples: int = 0
    # P*K IS the batch size -- the PK sampler supplies whole batches, so
    # per_device_train_batch_size is not consulted for training and is set to match only
    # so the two do not disagree in the logs. DeltaMZBias is O(batch * peaks^2), which at
    # 512 peaks is 12.9 GB for a batch of 48 and OOMed a 64 GB tile; 6x4 with gradient
    # checkpointing fits. Contrastive wants the largest batch that fits, since every
    # other row in it is a negative.
    groups_per_batch: int = field(default=6, metadata={"help": "P in the PK sampler"})
    replicates: int = field(default=4, metadata={"help": "K in the PK sampler"})


@dataclass
class ContrastiveTrainingArguments(TrainingArguments):
    wandb_project: str | None = None
    wandb_entity: str | None = None
    run_description: str | None = None


class ContrastiveTrainer(Trainer):
    """Standard Trainer, with the PK sampler and the loss components surfaced."""

    def __init__(self, *args, groups=None, groups_per_batch=12, replicates=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.groups_per_batch = groups_per_batch
        self.replicates = replicates

    def _get_train_sampler(self, *args, **kwargs):
        # Random batches hold about one positive PAIR; the PK sampler guarantees
        # groups_per_batch * replicates * (replicates-1) / 2 of them.
        return None

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader
        sampler = GroupBatchSampler(self.groups, self.groups_per_batch, self.replicates,
                                    seed=self.args.seed)
        return DataLoader(self.train_dataset, batch_sampler=sampler,
                          collate_fn=self.data_collator,
                          num_workers=self.args.dataloader_num_workers,
                          pin_memory=self.args.dataloader_pin_memory)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        # Both terms in the log: a run where the contrastive term falls while the KL
        # climbs is buying separation by forgetting, and the totals alone hide that.
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({"contrastive": float(outputs["contrastive"]),
                      "kl": float(outputs["kl"])})
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


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
    select_device()
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

    training_args.run_description = training_args.run_description or load_description()
    description = (
        f"Spectrum encoder from {Path(model_args.pretrained_path).parent.name} fine-tuned "
        f"with supervised contrastive loss on {data_args.dataset_repo} (replicates of one "
        f"peptide+charge are positives), temperature {model_args.temperature}, plus "
        f"KL={model_args.kl_weight} to the frozen pretrained intensity head so the encoder "
        f"separates peptides without forgetting the peak chemistry. "
        f"Batches are {data_args.groups_per_batch} groups x {data_args.replicates} "
        f"replicates. pooling={model_args.pooling}, embedding "
        f"{embedding_size(encoder, model_args.pooling)}, seed {training_args.seed}.")
    if training_args.run_description:
        description = f"{training_args.run_description} -- {description}"

    wandb_run = None
    if training_args.wandb_project:
        wandb_run = init_wandb_run(
            project=training_args.wandb_project, run_name=training_args.run_name,
            entity=training_args.wandb_entity, notes=description,
            tags=["contrastive", "spectrum-encoder", model_args.pooling],
            config={"model": asdict(model_args), "data": asdict(data_args)})
    try:
        if training_args.process_index == 0:
            (out_dir / "RUN.md").write_text(f"# {training_args.run_name}\n\n{description}\n")
            print(f"[contrastive] {description}", flush=True)

        with training_args.main_process_first(local=False, desc="replicate data"):
            datasets = build_alignment_datasets(
                data_args.dataset_repo, processor,
                num_proc=data_args.preprocessing_num_workers or None,
                validation_fraction=data_args.validation_fraction,
                seed=training_args.seed)
        if data_args.max_samples:
            # By group, not by row: see subset_by_group. A contiguous slice of this
            # corpus yields groups of about two, which makes the PK sampler draw
            # duplicates and the contrastive objective meaningless.
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

        collator = ContrastiveCollator(max_peptide_length=64)
        trainer = ContrastiveTrainer(
            model=model, args=training_args, train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"), data_collator=collator,
            groups=groups, groups_per_batch=data_args.groups_per_batch,
            replicates=data_args.replicates)
        trainer.add_callback(MemoryProbe(every=50))
        trainer.train()

        # The number this whole exercise exists to move: does the space separate peptides?
        if datasets.get("validation") is not None:
            before_after = group_separation_summary(
                model, datasets["validation"], collator, trainer.args.device,
                max_rows=training_args.eval_alignment_rows
                if hasattr(training_args, "eval_alignment_rows") else 2000)
            if trainer.is_world_process_zero():
                print(f"[contrastive] separation: {before_after}", flush=True)
                trainer.log(before_after)
                trainer.save_metrics("separation", before_after)

        if trainer.is_world_process_zero():
            # save_pretrained on the inner model, so the result is a drop-in
            # --pretrained_path for the alignment run.
            model.model.save_pretrained(str(out_dir / "final"))
            processor.save_pretrained(str(out_dir / "final"))
            print(f"[contrastive] saved to {out_dir / 'final'}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
