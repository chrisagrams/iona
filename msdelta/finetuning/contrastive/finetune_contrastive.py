"""Fine-tune the spectrum encoder contrastively, with KL to its own pretrained head.

    python -m msdelta.finetuning.contrastive.finetune_contrastive --args_file configs/finetune-contrastive-50m/training.args

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

from msdelta.finetuning.contrastive.contrastive import (GroupBatchSampler, MSDeltaForContrastive,
                                 PairBatchSampler,
                                 gradcache_step,
                                 embedding_size, group_separation_summary,
                                 retrieval_summary,
                                 subset_by_group)
from msdelta.finetuning.denoise.finetune_denoise import MemoryProbe, load_description, select_device, subset_splits
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining
from msdelta.models.processing_msdelta import MSDeltaProcessor
from msdelta.rescoring.reranking import (AlignmentCollator, REPLICATE_REPO, build_alignment_datasets,
                               group_separation_metrics, peptide_key)
from msdelta.utils.wandb_distributed import init_wandb_run


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
    pair_loss: bool = field(
        default=False,
        metadata={"help": "Replace the in-batch softmax with independent per-pair "
                          "terms: same peptide pulls together, different pushes apart "
                          "past a margin. Because pairs decompose, the number of "
                          "peptides a step sees stops being bounded by memory -- "
                          "gradient accumulation does the work GradCache does for the "
                          "softmax loss. Requires the pair sampler, switched on by the "
                          "same flag."},
    )
    pair_margin: float = field(
        default=1.0,
        metadata={"help": "Different-peptide pairs are pushed to at least this "
                          "distance and then ignored. Embeddings are unit-norm so "
                          "distance is in [0,2] and d^2 = 2-2cos; 1.0 asks for cosine "
                          "<= 0.5."},
    )
    pair_positive_weight: float = field(
        default=1.0,
        metadata={"help": "Rebalances the two terms when the sampler's positive "
                          "fraction is not 0.5."},
    )
    projection_dim: int = field(
        default=0,
        metadata={"help": "Width of an MLP projection head after pooling (master's "
                          "SpectrumRetrievalHead shape). 0 = no head: the loss and "
                          "retrieval use the normalised pooled vector, as every run so "
                          "far has. With a head, the loss sees the projection and "
                          "final/projection_head.pt is saved beside the encoder."})
    projection_hidden: int = field(
        default=0, metadata={"help": "Hidden width of the head; 0 = the pooled width."})
    projection_dropout: float = 0.1
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
    loss: str = field(
        default="supcon",
        metadata={"help": "contrastive loss: 'supcon' (softmax over the batch, every run so far) or "
                          "'sigmoid' (C8: SigLIP-style independent per-pair sigmoid, learnable "
                          "scale and bias; --temperature is then unused)"})
    sigmoid_init_scale: float = field(default=10.0, metadata={"help": "sigmoid loss: initial logit scale"})
    sigmoid_init_bias: float = field(default=-10.0, metadata={"help": "sigmoid loss: initial logit bias"})
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
    dataset_format: str = field(
        default="replicate",
        metadata={"help": "replicate: one spectrum per row, re-split here by peptide "
                          "(ms2-peptide-replicate-retrieval). grouped: one ANALYTE per "
                          "row, consensus + 3 experimental, with the corpus's own "
                          "peptide-disjoint train/validation/test splits "
                          "(ms-contrastive-100k); see msdelta/grouped_retrieval.py."})
    include_consensus: bool = field(
        default=False,
        metadata={"help": "grouped only: train on the consensus spectrum as a fourth "
                          "member. Off by default because a consensus is built FROM "
                          "the replicates, so pairing it with them is partly pairing "
                          "a spectrum with its own average."})
    exclude_replicate_peptides: bool = field(
        default=True,
        metadata={"help": "grouped only: drop peptides that appear anywhere in "
                          "ms2-peptide-replicate-retrieval (445 of 88,817 train "
                          "peptides), so a model trained here can still be scored on "
                          "that corpus -- and fed to the reranking chain built on it "
                          "-- without having seen its held-out peptides."})
    processor_name_or_path: str | None = None
    max_peaks: int = 512
    validation_fraction: float = 0.1
    fixed_width_batches: bool = field(
        default=True,
        metadata={"help": "Pad every spectrum batch to max_peaks instead of to the "
                          "longest spectrum in the batch. DeltaMZBias is O(batch * "
                          "width^2), so padding to the batch maximum makes memory "
                          "depend on which spectra the sampler happened to draw: the "
                          "same 200m config reserved 38.75 GB at seed 0 and 67.14 GB "
                          "at seed 3, and the wide draws took a GPU page fault at 200m "
                          "and 400m. Fixed width costs padding on narrow batches and "
                          "buys a memory figure that can be measured once and trusted. "
                          "Default on: an unpredictable crash is worse than a "
                          "predictable cost."},
    )
    split_seed: int = field(
        default=0,
        metadata={"help": "Seed for the train/validation split ONLY, deliberately "
                          "decoupled from training_args.seed. The split used to move "
                          "with the training seed, which meant a seed sweep scored "
                          "every arm on DIFFERENT held-out data and no two runs were "
                          "comparable. Hold this fixed and vary --seed to measure "
                          "training variance; vary this to measure split variance. "
                          "They are different questions."},
    )
    preprocessing_num_workers: int = 24
    max_samples: int = 0
    # P*K IS the batch size -- the PK sampler supplies whole batches, so
    # per_device_train_batch_size is not consulted for training and is set to match only
    # so the two do not disagree in the logs. DeltaMZBias is O(batch * peaks^2), which at
    # 512 peaks is 12.9 GB for a batch of 48 and OOMed a 64 GB tile; 6x4 with gradient
    # checkpointing fits. Contrastive wants the largest batch that fits, since every
    # other row in it is a negative.
    pairs_per_batch: int = field(
        default=8,
        metadata={"help": "Pair sampler only: pairs per minibatch. The batch is 2x "
                          "this many spectra, kept small on purpose -- pair terms are "
                          "independent, so breadth comes from "
                          "gradient_accumulation_steps rather than from a batch that "
                          "has to fit in memory all at once."},
    )
    positive_fraction: float = field(
        default=0.5,
        metadata={"help": "Pair sampler only: fraction of pairs drawn from the same "
                          "peptide."},
    )
    groups_per_batch: int = field(default=6, metadata={"help": "P in the PK sampler"})
    replicates: int = field(default=4, metadata={"help": "K in the PK sampler"})
    gradcache_chunk: int = field(
        default=0,
        metadata={"help": "spectra per forward when using GradCache (0 = off). With it "
                          "on, groups_per_batch x replicates can far exceed what fits: "
                          "peak memory is one chunk, not the batch. The gradient is "
                          "exact -- tests assert it against a full-batch backward."})
    same_mass_batches: bool = field(
        default=False,
        metadata={"help": "C19: build each batch from peptide groups that are neighbours in "
                          "neutral mass (sorted with +-mass_jitter Da jitter each epoch), so "
                          "in-batch negatives are same-mass competitors."})
    mass_jitter: float = field(default=1.0, metadata={"help": "same_mass_batches: jitter (Da)"})
    random_group_fraction: float = field(
        default=0.0,
        metadata={"help": "same_mass_batches ablation: fraction of groups served in random "
                          "order (0 = pure same-mass). See --random_mix."})
    random_mix: str = field(
        default="within",
        metadata={"help": "with random_group_fraction: 'within' = every batch mixes same-mass "
                          "and random groups; 'between' = that fraction of batches is random; "
                          "'regions' = every batch is two same-mass blocks from two mass "
                          "regions, the second holding that fraction of the groups"})
    gradcache_trim_padding: bool = field(
        default=False,
        metadata={"help": "GradCache: sort each batch's spectra by length and cut every chunk "
                          "to its own longest spectrum instead of max_peaks. Same gradient "
                          "(tested), much less compute on padding."})


@dataclass
class ContrastiveTrainingArguments(TrainingArguments):
    wandb_project: str | None = None
    wandb_entity: str | None = None
    run_description: str | None = None
    eval_retrieval_rows: int = field(
        default=0,
        metadata={"help": "Spectra to embed when scoring retrieval inside evaluate(). "
                          "0 disables it and leaves eval_loss as the only eval metric, "
                          "which is what every run before this did -- and eval_loss is "
                          "not selectable here, because the contrastive objective is "
                          "solved by epoch 0.18 of 3 and is measured on a 4-spectrum "
                          "batch. Set this AND metric_for_best_model to make "
                          "load_best_model_at_end pick on the task."})
    eval_alignment_rows: int = field(
        default=2000,
        metadata={"help": "Spectra embedded for the post-training separation and "
                          "retrieval summaries. Declared here rather than read off the "
                          "namespace with hasattr, which silently fell back to 2000."})


class SamplerEpochCallback(TrainerCallback):
    """Set the batch sampler's epoch from the step count at every epoch start (FT/K38).

    The samplers reshuffle by their own epoch counter (seeded [seed, epoch]). On a resume
    from checkpoint-N that counter restarts at 0, and transformers' own set_epoch does not
    reach a custom batch_sampler behind accelerate's wrappers, so the resumed epoch replayed
    epoch 0's batch order (then skipped N of them) instead of continuing the interrupted
    epoch. The epoch is global_step // optimizer-steps-per-epoch at every epoch start --
    fresh or resumed -- so a resumed run draws exactly the batches the uninterrupted run did.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        sampler = getattr(self.trainer, "_batch_sampler", None)
        if sampler is None or not hasattr(sampler, "set_epoch"):
            return
        steps_per_epoch = max(len(sampler) // max(args.gradient_accumulation_steps, 1), 1)
        sampler.set_epoch(state.global_step // steps_per_epoch)


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
                 gradcache_chunk=0, gradcache_trim_padding=False, group_masses=None,
                 mass_jitter=1.0, random_group_fraction=0.0, random_mix="within",
                 encoder_lr_scale=1.0, layer_mix_lr=None,
                 pair_loss=False, pairs_per_batch=8, positive_fraction=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups
        self.groups_per_batch = groups_per_batch
        self.replicates = replicates
        self.gradcache_chunk = gradcache_chunk
        self.gradcache_trim_padding = gradcache_trim_padding
        self.group_masses = group_masses
        self.mass_jitter = mass_jitter
        self.random_group_fraction = random_group_fraction
        self.random_mix = random_mix
        self.encoder_lr_scale = encoder_lr_scale
        self.layer_mix_lr = layer_mix_lr
        self.pair_loss = pair_loss
        self.pairs_per_batch = pairs_per_batch
        self.positive_fraction = positive_fraction

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
        if self.pair_loss:
            sampler = PairBatchSampler(self.groups, self.pairs_per_batch,
                                       self.positive_fraction, seed=self.args.seed)
        else:
            sampler = GroupBatchSampler(self.groups, self.groups_per_batch,
                                        self.replicates, seed=self.args.seed,
                                        group_masses=self.group_masses,
                                        mass_jitter=self.mass_jitter,
                                        random_fraction=self.random_group_fraction,
                                        random_mix=self.random_mix)
        self._batch_sampler = sampler          # SamplerEpochCallback sets its epoch
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
                                 accelerator=getattr(self, "accelerator", None),
                                 trim_padding=self.gradcache_trim_padding)
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({"contrastive": float(outputs["contrastive"]),
                      "kl": float(outputs["kl"])})
            self._log_layer_mix()
        return outputs["loss"].detach()

    def evaluate(self, eval_dataset=None, ignore_keys=None,
                 metric_key_prefix: str = "eval"):
        """Score RETRIEVAL at eval time, not just the contrastive loss.

        Without this the only eval number is eval_loss, and the contrastive loss is
        useless for model selection here: measured on 50m@330k it falls below 10% of
        chance (ln 4 = 1.386) by epoch 0.18 of 3.0, so 94% of training optimises a task
        that is already solved. Worse, it is computed on a FOUR-spectrum batch, so each
        value is one noisy draw -- the last logged step of a real run was 0.142 while
        epoch 2.90 had reached 0.0016. Selecting on that would pick noise.

        These metrics are what load_best_model_at_end reads, so putting them here is
        what makes `final/` the BEST encoder rather than whatever the last step left
        behind. They also reach wandb, because Trainer logs whatever evaluate returns.

        Kept cheap on purpose: retrieval_summary embeds eval_retrieval_rows spectra, so
        the cost is one forward pass over a subset and nothing is trained on it.
        """
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
        # Prefix to match HF's convention so metric_for_best_model can name them:
        #   retrieval/MAP@R  ->  eval_retrieval/MAP@R
        scored = {f"{metric_key_prefix}_{k}": v for k, v in extra.items()
                  if isinstance(v, (int, float))}
        metrics.update(scored)
        self.log(scored)
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        # Both terms in the log: a run where the contrastive term falls while the KL
        # climbs is buying separation by forgetting, and the totals alone hide that.
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({"contrastive": float(outputs["contrastive"]),
                      "kl": float(outputs["kl"])})
            self._log_layer_mix()
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


def load_contrastive_datasets(data_args, processor) -> dict:
    """train + validation for either corpus format. See ContrastiveDataArguments."""
    from msdelta import grouped_retrieval as gr

    if data_args.dataset_format == "grouped":
        # K above the group size makes the PK sampler draw with replacement, i.e. pair a
        # spectrum with ITSELF as a positive -- a free, meaningless positive per group.
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
    # max_peaks drops can leave an analyte with one spectrum: it has no positive, and
    # the sampler would pair it with itself. Drop such rows from TRAIN only; eval
    # metrics already skip queries with no relevant item.
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
        from msdelta.models.configuration_msdelta import MSDeltaConfig
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
                                  layer_mix_norm=model_args.layer_mix_norm,
                                  pair_loss=model_args.pair_loss,
                                  pair_margin=model_args.pair_margin,
                                  pair_positive_weight=model_args.pair_positive_weight,
                                  projection_hidden=model_args.projection_hidden,
                                  projection_dim=model_args.projection_dim,
                                  projection_dropout=model_args.projection_dropout,
                                  loss=model_args.loss,
                                  sigmoid_init_scale=model_args.sigmoid_init_scale,
                                  sigmoid_init_bias=model_args.sigmoid_init_bias)
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

        with training_args.main_process_first(local=False, desc="contrastive data"):
            datasets = load_contrastive_datasets(data_args, processor)
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
        train_peptides = datasets["train"]["peptide"]
        uniq_keys, groups = np.unique(
            np.array([peptide_key(p, c) for p, c in
                      zip(train_peptides, datasets["train"]["charge"])]),
            return_inverse=True)
        group_masses = None
        if data_args.same_mass_batches:
            from msdelta.rescoring.reranking import peptide_neutral_mass
            first = {}
            for row, g in enumerate(groups):
                first.setdefault(int(g), row)
            group_masses = {g: peptide_neutral_mass(train_peptides[row]) for g, row in first.items()}
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
            gradcache_trim_padding=data_args.gradcache_trim_padding,
            group_masses=group_masses, mass_jitter=data_args.mass_jitter,
            random_group_fraction=data_args.random_group_fraction,
            random_mix=data_args.random_mix,
            encoder_lr_scale=model_args.encoder_lr_scale,
            layer_mix_lr=(model_args.layer_mix_lr
                          if model_args.pooling == "layer_mix" else None),
            pair_loss=model_args.pair_loss,
            pairs_per_batch=data_args.pairs_per_batch,
            positive_fraction=data_args.positive_fraction)
        trainer.add_callback(MemoryProbe(every=50))
        trainer.add_callback(SamplerEpochCallback(trainer))
        trainer.add_callback(SaveEncoderCallback(model, processor))
        # Pass it explicitly. Trainer.train() defaults resume_from_checkpoint to None
        # and never falls back to args.resume_from_checkpoint, so the CLI flag parses
        # cleanly and is then IGNORED -- the run restarts from scratch while looking as
        # though it resumed. See pbs/aurora-finetune-sweep.pbs RESUME_JOB.
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # FT31: a run whose sampler yields no batch "finishes" at step 0 and then writes
        # final/ and metrics exactly like a real one; the C2/C4 smoke 8859890 reported
        # 14/14 ok that way. Fail loudly instead.
        if trainer.state.global_step == 0:
            raise SystemExit("trained 0 optimizer steps -- too few groups/rows for one batch?")

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

            # The TASK, on the same rows. The separation ratio is a proxy that has never
            # been checked against what it proxies for; reporting both on every run is
            # what makes the correlation measurable across a grid instead of assumed.
            # Never fatal: a missing or broken faiss must not lose a completed run.
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

            # With a projection head, also score the PRE-head features on the same rows:
            # which of the two is the better embedding is the question the head asks.
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
            # save_pretrained on the inner model, so the result is a drop-in
            # --pretrained_path for the alignment run.
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
