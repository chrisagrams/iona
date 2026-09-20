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
from torch import nn
from transformers import (HfArgumentParser, Trainer, TrainerCallback, TrainingArguments,
                          set_seed)

from msdelta.contrastive import (GroupBatchSampler, MSDeltaForContrastive,
                                 gradcache_step,
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
    pooling: str = field(
        default="mean+max",
        metadata={"help": "mean, mean+max, weighted_mean, weighted_mean+max, or "
                          "layer_mix -- a trained convex mixture over every encoder "
                          "depth, sequence-mean pooled to d_model."},
    )
    random_init: bool = field(
        default=False,
        metadata={"help": "Same architecture, NO pretrained weights. The control that "
                          "asks whether contrastive training needs the pretrained "
                          "encoder at all: the frozen probe already showed the "
                          "pretrained embedding is indistinguishable from random "
                          "(1.35 either way), so if a random encoder also reaches ~7.8 "
                          "under this loss, pretraining contributes nothing to this "
                          "objective either."},
    )
    layer_mix_norm: bool = field(
        default=True,
        metadata={"help": "LayerNorm each depth before mixing. Off, the deepest blocks "
                          "dominate by magnitude alone and the learned weights are "
                          "decorative."},
    )
    layer_mix_lr: float = field(
        default=1e-3,
        metadata={"help": "Learning rate for the layer-mixture logits ONLY. They are a "
                          "different kind of parameter from network weights and their "
                          "scale is set by softmax geometry, not by the encoder: they "
                          "start equal and must travel O(1-5) apart before the mixture "
                          "is anything but uniform. At the encoder's 2e-5 that takes "
                          "~150k steps, so a full run would have measured an unweighted "
                          "average of all layers while appearing to learn one. At 1e-2 "
                          "it collapses onto a single layer inside 1000 steps, which "
                          "throws away the mixture just as completely. mix/entropy in "
                          "the log says which failure is happening: pinned at ln(L) is "
                          "too slow, crashing to 0 early is too fast."},
    )
    encoder_lr_scale: float = field(
        default=1.0,
        metadata={"help": "Encoder learning rate as a multiple of the head's. 0 freezes "
                          "the encoder outright, which is the control that says whether "
                          "the readout or the encoder is doing the work."},
    )
    temperature: float = field(default=0.07, metadata={"help": "SupCon temperature"})
    kl_weight: float = field(
        default=100.0,
        metadata={"help": "weight on KL to the frozen pretrained head. 0 disables the "
                          "regulariser and lets the encoder forget the chemistry. The "
                          "default is 100 because the two terms are on very different "
                          "scales: measured on real batches, contrastive is ~2.6 and the "
                          "KL ~0.007, so at weight 1.0 the regulariser is 0.3% of the "
                          "loss and constrains nothing. 100 puts them within an order of "
                          "magnitude, which is where a regulariser can actually trade "
                          "against the objective."})


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
    gradcache_chunk: int = field(
        default=0,
        metadata={"help": "spectra per forward when using GradCache (0 = off). With it "
                          "on, groups_per_batch x replicates can far exceed what fits: "
                          "peak memory is one chunk, not the batch. The gradient is "
                          "exact -- tests assert it against a full-batch backward."})


@dataclass
class ContrastiveTrainingArguments(TrainingArguments):
    wandb_project: str | None = None
    wandb_entity: str | None = None
    run_description: str | None = None


class SaveEncoderCallback(TrainerCallback):
    """Write a loadable HF encoder into every checkpoint the Trainer saves.

    MSDeltaForContrastive is a plain nn.Module wrapping a real PreTrainedModel, so the
    Trainer saves it as a bare state dict: `checkpoint-N/model.safetensors` carries
    `model.*` and `layer_mix.*` keys and no config.json, and nothing can load it without
    first reconstructing the wrapper by hand. The end-of-run `final/` is fine because
    main() saves the INNER model there explicitly; the intermediate checkpoints are the
    gap, and they are the ones you want when a run dies or when a mid-training encoder
    turns out to be the interesting one.

    Making the wrapper a PreTrainedModel would fix this properly, and should be done if
    this line survives -- it also brings resume and best-model selection. That is a
    config class, a from_pretrained that rebuilds the inner model, and care to keep the
    frozen reference encoder out of the serialised weights (it is a duplicate of the
    pretrained encoder and would roughly double every checkpoint). This callback buys
    the loadable artifact without any of that.

    The unwrapped model is captured at construction rather than taken from kwargs,
    because by save time the Trainer's model may be behind DeepSpeed or DDP.
    """

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
            # A failed side-artifact must not take down a training run whose real
            # checkpoint the Trainer has already written.
            print(f"[contrastive] could not save encoder into {target}: {error}",
                  flush=True)


class ContrastiveTrainer(Trainer):
    """Standard Trainer, with the PK sampler and the loss components surfaced."""

    def __init__(self, *args, groups=None, groups_per_batch=12, replicates=4,
                 gradcache_chunk=0, encoder_lr_scale=1.0, layer_mix_lr=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.groups_per_batch = groups_per_batch
        self.replicates = replicates
        self.gradcache_chunk = gradcache_chunk
        self.encoder_lr_scale = encoder_lr_scale
        self.layer_mix_lr = layer_mix_lr

    def create_optimizer(self):
        """Encoder and readout in separate groups, so one can move slower than the other.

        The point of the sweep this supports: with a trained layer mixture, is the gain
        coming from the readout or from moving the encoder? Only comparing the same
        readout at several encoder rates answers that.
        """
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
        """Which depths the mixture actually chose -- the result, not a diagnostic.

        A mixture that stays uniform means depth did not matter; one that concentrates
        says where the peptide structure lives, and can be read against the frozen
        layer probe that motivated this.
        """
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        # Both terms in the log: a run where the contrastive term falls while the KL
        # climbs is buying separation by forgetting, and the totals alone hide that.
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({"contrastive": float(outputs["contrastive"]),
                      "kl": float(outputs["kl"])})
            self._log_layer_mix()
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
    if model_args.random_init:
        # from_config, never from_pretrained-then-reinitialise: loading first would
        # leave any buffer the init does not touch still carrying pretrained values,
        # which is a subtler thing to be wrong about than it looks. Matches
        # build_denoising_model, which learned this the same way.
        from msdelta.configuration_msdelta import MSDeltaConfig
        config = MSDeltaConfig.from_pretrained(model_args.pretrained_path)
        encoder = MSDeltaForPreTraining(config)
        # The reference is a SEPARATE random model with the same config, not a copy of
        # the encoder, only if a KL target is asked for. Regularising a random encoder
        # toward a DIFFERENT random function is meaningless, so the honest control is
        # kl_weight 0; the arms that set it are kept for grid symmetry and say so.
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
                                  layer_mix_norm=model_args.layer_mix_norm)
    if model_args.encoder_lr_scale == 0:
        # A zero learning rate would still let weight decay and any stateful optimizer
        # move the encoder. Freezing is the thing being asked for, so freeze it.
        encoder.requires_grad_(False)
        if model.layer_mix is None:
            raise SystemExit("a frozen encoder with a fixed pooling has nothing to "
                             "train; use pooling=layer_mix or a non-zero "
                             "encoder_lr_scale")

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
            replicates=data_args.replicates,
            gradcache_chunk=data_args.gradcache_chunk,
            encoder_lr_scale=model_args.encoder_lr_scale,
            layer_mix_lr=(model_args.layer_mix_lr
                          if model_args.pooling == "layer_mix" else None))
        trainer.add_callback(MemoryProbe(every=50))
        trainer.add_callback(SaveEncoderCallback(model, processor))
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
