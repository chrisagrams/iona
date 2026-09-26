"""Fine-tune the spectrum encoder contrastively, with KL to its own pretrained head.

    python -m msdelta.finetune_contrastive --args_file configs/finetune-contrastive-replicate-50m/training.args
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import (HfArgumentParser, Trainer, TrainerCallback, TrainingArguments,
                          set_seed)

from msdelta.contrastive import (GroupBatchSampler, MSDeltaForContrastive,
                                 PairBatchSampler,
                                 gradcache_step,
                                 embedding_size, group_separation_summary,
                                 retrieval_summary,
                                 subset_by_group)
from msdelta.finetune_denoise import subset_splits
from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.reranking import (AlignmentCollator, REPLICATE_REPO, build_alignment_datasets,
                               group_separation_metrics, peptide_key)
from msdelta.wandb_distributed import init_wandb_run


@dataclass
class ContrastiveModelArguments:
    pretrained_path: str = ""
    pooling: str = "mean+max"
    random_init: bool = False
    pair_loss: bool = False
    pair_margin: float = 1.0
    pair_positive_weight: float = 1.0
    projection_dim: int = 0
    projection_hidden: int = 0
    projection_dropout: float = 0.1
    layer_mix_norm: bool = True
    layer_mix_lr: float = 1e-3
    encoder_lr_scale: float = 1.0
    temperature: float = 0.07
    kl_weight: float = 100.0


@dataclass
class ContrastiveDataArguments:
    dataset_repo: str = REPLICATE_REPO
    dataset_format: str = "replicate"
    include_consensus: bool = False
    exclude_replicate_peptides: bool = True
    processor_name_or_path: str | None = None
    max_peaks: int = 512
    validation_fraction: float = 0.1
    fixed_width_batches: bool = True
    split_seed: int = 0
    preprocessing_num_workers: int = 24
    max_samples: int = 0
    pairs_per_batch: int = 8
    positive_fraction: float = 0.5
    groups_per_batch: int = 6
    replicates: int = 4
    gradcache_chunk: int = 0


@dataclass
class ContrastiveTrainingArguments(TrainingArguments):
    wandb_project: str | None = None
    wandb_entity: str | None = None
    eval_retrieval_rows: int = 0
    eval_alignment_rows: int = 2000


class SaveEncoderCallback(TrainerCallback):
    """Save the inner HF encoder into every checkpoint, so intermediate checkpoints are loadable."""

    def __init__(self, model: nn.Module, processor=None):
        self.model = model
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        inner = getattr(self.model, "model", self.model)
        if not hasattr(inner, "save_pretrained"):
            return
        target = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "encoder"
        try:
            inner.save_pretrained(str(target))
            if self.processor is not None:
                self.processor.save_pretrained(str(target))
        except Exception as error:
            print(f"[contrastive] could not save encoder into {target}: {error}",
                  flush=True)


class ContrastiveTrainer(Trainer):
    """Standard Trainer, with the PK sampler and the loss components surfaced."""

    def __init__(self, *args, groups=None, groups_per_batch=12, replicates=4,
                 gradcache_chunk=0, encoder_lr_scale=1.0, layer_mix_lr=None,
                 pair_loss=False, pairs_per_batch=8, positive_fraction=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.groups_per_batch = groups_per_batch
        self.replicates = replicates
        self.gradcache_chunk = gradcache_chunk
        self.encoder_lr_scale = encoder_lr_scale
        self.layer_mix_lr = layer_mix_lr
        self.pair_loss = pair_loss
        self.pairs_per_batch = pairs_per_batch
        self.positive_fraction = positive_fraction

    def create_optimizer(self):
        """Separate parameter groups for encoder, readout and layer mixture."""
        if self.optimizer is not None:
            return self.optimizer
        if self.encoder_lr_scale == 1.0 and self.layer_mix_lr is None:
            return super().create_optimizer()
        optimizer_class, kwargs = type(self).get_optimizer_cls_and_kwargs(self.args, self.model)
        kwargs.pop("lr", None)
        encoder, readout, mixture = [], [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("layer_mix."):
                mixture.append(parameter)
            elif name.startswith("model."):
                encoder.append(parameter)
            else:
                readout.append(parameter)
        groups = []
        if encoder:
            groups.append({"params": encoder,
                           "lr": self.args.learning_rate * self.encoder_lr_scale})
        if readout:
            groups.append({"params": readout, "lr": self.args.learning_rate})
        if mixture:
            groups.append({"params": mixture,
                           "lr": self.layer_mix_lr or self.args.learning_rate})
        if not groups:
            raise ValueError("nothing to optimise: every parameter is frozen")
        self.optimizer = optimizer_class(groups, **kwargs)
        return self.optimizer

    def _log_layer_mix(self) -> None:
        """Log the layer-mixture weights."""
        inner = self.model.module if hasattr(self.model, "module") else self.model
        mixer = getattr(inner, "layer_mix", None)
        if mixer is None:
            return
        weights = mixer.weights.detach().float().cpu()
        self.log({f"mix/layer{i:02d}": float(w) for i, w in enumerate(weights)}
                 | {"mix/argmax": int(weights.argmax()),
                    "mix/max": float(weights.max()),
                    "mix/entropy": float(-(weights * weights.clamp_min(1e-9).log()).sum()),
                    "mix/gamma": float(mixer.gamma) if mixer.gamma is not None else 1.0})

    def _get_train_sampler(self, *args, **kwargs):
        # Batches come from the PK/pair sampler in get_train_dataloader.
        return None

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader
        if self.pair_loss:
            sampler = PairBatchSampler(self.groups, self.pairs_per_batch,
                                       self.positive_fraction, seed=self.args.seed)
        else:
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
            self._log_layer_mix()
        return outputs["loss"].detach()

    def evaluate(self, eval_dataset=None, ignore_keys=None,
                 metric_key_prefix: str = "eval"):
        """Add retrieval metrics to evaluate(), so load_best_model_at_end can select on them."""
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                   metric_key_prefix=metric_key_prefix)
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        rows = getattr(self.args, "eval_retrieval_rows", 0)
        if dataset is None or not rows:
            return metrics
        try:
            extra = retrieval_summary(self.model, dataset, self.data_collator,
                                      self.args.device, max_rows=rows)
        except Exception as error:      # never lose a run to the eval pass
            if self.is_world_process_zero():
                print(f"[contrastive] retrieval eval failed inside evaluate(): {error}", flush=True)
            return metrics
        scored = {f"{metric_key_prefix}_{k}": v for k, v in extra.items()
                  if isinstance(v, (int, float))}
        metrics.update(scored)
        self.log(scored)
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({"contrastive": float(outputs["contrastive"]),
                      "kl": float(outputs["kl"])})
            self._log_layer_mix()
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


def load_contrastive_datasets(data_args, processor) -> dict:
    """train + validation for either corpus format. See ContrastiveDataArguments."""
    from msdelta import grouped_retrieval as gr

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
        exclude_replicate_peptides=data_args.exclude_replicate_peptides,
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
    if model_args.random_init:
        from msdelta.configuration_msdelta import MSDeltaConfig
        config = MSDeltaConfig.from_pretrained(model_args.pretrained_path)
        encoder = MSDeltaForPreTraining(config)
        reference = MSDeltaForPreTraining(config) if model_args.kl_weight > 0 else None
        print("[contrastive] RANDOM INIT control: architecture of "
              f"{model_args.pretrained_path}, no pretrained weights", flush=True)
    else:
        encoder = MSDeltaForPreTraining.from_pretrained(model_args.pretrained_path)
        reference = (MSDeltaForPreTraining.from_pretrained(model_args.pretrained_path)
                     if model_args.kl_weight > 0 else None)
    model = MSDeltaForContrastive(encoder, reference, pooling=model_args.pooling,
                                  temperature=model_args.temperature,
                                  kl_weight=model_args.kl_weight,
                                  layer_mix_norm=model_args.layer_mix_norm,
                                  pair_loss=model_args.pair_loss,
                                  pair_margin=model_args.pair_margin,
                                  pair_positive_weight=model_args.pair_positive_weight,
                                  projection_hidden=model_args.projection_hidden,
                                  projection_dim=model_args.projection_dim,
                                  projection_dropout=model_args.projection_dropout)
    if model_args.encoder_lr_scale == 0:
        # Freeze outright; lr 0 would still let weight decay move it.
        encoder.requires_grad_(False)
        if model.layer_mix is None:
            raise SystemExit("a frozen encoder with a fixed pooling has nothing to "
                             "train; use pooling=layer_mix or a non-zero "
                             "encoder_lr_scale")

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

        collator = ContrastiveCollator(
            max_peptide_length=64,
            pad_spectra_to=(data_args.max_peaks
                            if data_args.fixed_width_batches else 0))
        trainer = ContrastiveTrainer(
            model=model, args=training_args, train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"), data_collator=collator,
            groups=groups, groups_per_batch=data_args.groups_per_batch,
            replicates=data_args.replicates,
            gradcache_chunk=data_args.gradcache_chunk,
            encoder_lr_scale=model_args.encoder_lr_scale,
            layer_mix_lr=(model_args.layer_mix_lr
                          if model_args.pooling == "layer_mix" else None),
            pair_loss=model_args.pair_loss,
            pairs_per_batch=data_args.pairs_per_batch,
            positive_fraction=data_args.positive_fraction)
        trainer.add_callback(SaveEncoderCallback(model, processor))
        # Trainer.train() does not read args.resume_from_checkpoint on its own.
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        if trainer.state.global_step == 0:
            raise SystemExit("trained 0 optimizer steps -- too few groups/rows for one batch?")

        if datasets.get("validation") is not None:
            before_after = group_separation_summary(
                model, datasets["validation"], collator, trainer.args.device,
                max_rows=training_args.eval_alignment_rows
                if hasattr(training_args, "eval_alignment_rows") else 2000)
            if trainer.is_world_process_zero():
                print(f"[contrastive] separation: {before_after}", flush=True)
                trainer.log(before_after)
                trainer.save_metrics("separation", before_after)

            try:
                retrieval = retrieval_summary(
                    model, datasets["validation"], collator, trainer.args.device,
                    max_rows=training_args.eval_alignment_rows
                    if hasattr(training_args, "eval_alignment_rows") else 2000)
            except Exception as error:
                retrieval = {}
                if trainer.is_world_process_zero():
                    print(f"[contrastive] retrieval eval failed: {error}", flush=True)
            if retrieval and trainer.is_world_process_zero():
                print(f"[contrastive] retrieval: {retrieval}", flush=True)
                trainer.log({k: v for k, v in retrieval.items()
                             if isinstance(v, (int, float))})
                trainer.save_metrics("retrieval", retrieval)

            # With a projection head, also score the pre-head features.
            if model.projection is not None:
                model.readout = "pooled"
                try:
                    pooled = retrieval_summary(
                        model, datasets["validation"], collator, trainer.args.device,
                        max_rows=training_args.eval_alignment_rows)
                finally:
                    model.readout = "head"
                if trainer.is_world_process_zero():
                    pooled = {f"retrieval_pooled/{k.split('/', 1)[-1]}": v
                              for k, v in pooled.items()}
                    print(f"[contrastive] retrieval (pre-head): {pooled}", flush=True)
                    trainer.log({k: v for k, v in pooled.items()
                                 if isinstance(v, (int, float))})
                    trainer.save_metrics("retrieval_pooled", pooled)

        if trainer.is_world_process_zero():
            model.model.save_pretrained(str(out_dir / "final"))
            processor.save_pretrained(str(out_dir / "final"))
            if model.projection is not None:
                torch.save({"state_dict": model.projection.state_dict(),
                            "pooling": model_args.pooling,
                            "projection_hidden": model_args.projection_hidden,
                            "projection_dim": model_args.projection_dim,
                            "projection_dropout": model_args.projection_dropout},
                           out_dir / "final" / "projection_head.pt")
            print(f"[contrastive] saved to {out_dir / 'final'}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
