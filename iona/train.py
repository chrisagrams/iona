"""Train the masked-intensity model."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from accelerate.utils import DeepSpeedPlugin
from datasets import Dataset
from transformers import HfArgumentParser, Trainer, set_seed
from transformers.trainer_pt_utils import LengthGroupedSampler

from iona.callbacks import SidecarCallback, WalltimeCheckpointCallback, build_callbacks
from iona.configuration_iona import IonaConfig
from iona.data import (
    build_pretraining_datasets,
    load_pretraining_datasets_from_disk,
    subsample_train,
)
from iona.env import RankEnv
from iona.modeling_iona import IonaForPreTraining
from iona.posttraining import build_probe_data
from iona.processing_iona import IonaDataCollatorForPreTraining, IonaProcessor
from iona.training_args import DataArguments, IonaTrainingArguments, ModelArguments
from iona.viz import render_bias_panels
from iona.wandb_distributed import init_wandb_run


def full_data_max_steps(n_examples: int, args: IonaTrainingArguments) -> int:
    """Optimizer steps Trainer would take for n_examples at the configured epochs."""
    per_step = args.per_device_train_batch_size * args.world_size
    rounding = math.floor if args.dataloader_drop_last else math.ceil
    batches = max(rounding(n_examples / per_step), 1)
    steps_per_epoch = math.ceil(batches / args.gradient_accumulation_steps)
    return math.ceil(args.num_train_epochs * steps_per_epoch)


class IonaTrainer(Trainer):
    """Configure separate DeepSpeed plugins for pretraining and frozen-encoder probes."""

    def __init__(
        self, *args, use_denoising_probe: bool = False, use_retrieval_probe: bool = False, **kwargs
    ):
        self.use_denoising_probe = use_denoising_probe
        self.use_retrieval_probe = use_retrieval_probe
        super().__init__(*args, **kwargs)

    def _build_accelerator_args(self, **kwargs):
        args = super()._build_accelerator_args(**kwargs)
        pretrain_plugin = args.get("deepspeed_plugin")
        if pretrain_plugin is not None and (self.use_denoising_probe or self.use_retrieval_probe):
            plugins = {"pretrain": pretrain_plugin}
            if self.use_denoising_probe:
                plugins["denoise"] = DeepSpeedPlugin(hf_ds_config=self.args.deepspeed)
            if self.use_retrieval_probe:
                plugins["retrieval"] = DeepSpeedPlugin(hf_ds_config=self.args.deepspeed)
            args["deepspeed_plugin"] = plugins
        return args

    def _length_grouped_sampler(self, dataset, batch_size: int) -> LengthGroupedSampler | None:
        """Group by length from a numpy copy of the length column; None defers to Trainer.

        Trainer passes ``dataset["length"]`` straight to the sampler. In datasets 5 that is a
        lazy Column, which the sampler indexes once per row (~80 us each): hours at 100M rows,
        on every rank and every epoch.
        """
        if (
            self.args.train_sampling_strategy != "group_by_length"
            or not isinstance(dataset, Dataset)
            or self.args.length_column_name not in dataset.column_names
        ):
            return None
        # A numpy-formatted slice converts the whole column at once, following any index mapping.
        name = self.args.length_column_name
        lengths = dataset.select_columns([name]).with_format("numpy")[:][name]
        return LengthGroupedSampler(batch_size, lengths=lengths)  # pyright: ignore[reportArgumentType]

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        sampler = self._length_grouped_sampler(dataset, batch_size)
        return sampler if sampler is not None else super()._get_train_sampler(train_dataset)

    def _get_eval_sampler(self, eval_dataset):
        sampler = self._length_grouped_sampler(eval_dataset, self.args.eval_batch_size)
        return sampler if sampler is not None else super()._get_eval_sampler(eval_dataset)

    def _save_rng_state(self, output_dir: str) -> None:
        """Create the checkpoint directory once before ranks write their RNG states.

        Some distributed filesystems can report a spurious ``FileExistsError`` when
        many ranks concurrently call ``os.makedirs(..., exist_ok=True)``.  Trainer
        saves a distinct RNG state for every rank, so synchronize the directory
        creation without suppressing any of those files.
        """
        if self.args.world_size > 1:
            if self.args.process_index == 0:
                os.makedirs(output_dir, exist_ok=True)
            self.accelerator.wait_for_everyone()
        super()._save_rng_state(output_dir)


def main(argv: list[str] | None = None) -> int:
    env = RankEnv()
    if env.local_rank is not None and torch.xpu.is_available():
        torch.xpu.set_device(env.local_rank)

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, IonaTrainingArguments)  # pyright: ignore[reportArgumentType]
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=argv,
        args_file_flag="--args_file",
    )

    out_dir = Path(training_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(exist_ok=True)

    set_seed(training_args.seed)
    model_config = IonaConfig.from_pretrained(model_args.config_name)
    if model_args.config_overrides is not None:
        model_config.update_from_string(model_args.config_overrides)
        model_config._validate()
    processor_overrides = {}
    if data_args.max_peaks is not None:
        processor_overrides["max_peaks"] = data_args.max_peaks
    processor = IonaProcessor.from_pretrained(
        data_args.processor_name_or_path,
        **processor_overrides,
    )
    model = IonaForPreTraining(model_config)
    resolved = {
        "model": model_config.to_dict(),
        "processor": processor.to_dict(),
        "data": asdict(data_args),
        "training": training_args.to_dict(),
    }

    sidecar_callback = None
    wandb_run = None
    if training_args.wandb_project:
        wandb_run = init_wandb_run(
            project=training_args.wandb_project,
            run_name=training_args.run_name,
            config=resolved,
            shared=training_args.probe_execution == "sidecar",
            dir=env.wandb_dir or out_dir,
        )

    try:
        if training_args.process_index == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[model] {n_params / 1e6:.2f}M params", flush=True)

        if data_args.preprocessed_dataset_dir:
            if training_args.process_index == 0:
                print(
                    f"[data] loading finalized dataset from {data_args.preprocessed_dataset_dir}",
                    flush=True,
                )
            train_ds, val_ds = load_pretraining_datasets_from_disk(
                data_args.preprocessed_dataset_dir,
                train_split=data_args.dataset_train_split,
                validation_split=data_args.dataset_validation_split,
            )
        else:
            # Create the shared dataset cache on rank 0.
            with training_args.main_process_first(local=False, desc="dataset preprocessing"):
                if training_args.process_index == 0:
                    print(
                        f"[data] preprocessing with {data_args.preprocessing_num_workers} "
                        "CPU workers",
                        flush=True,
                    )
                train_ds, val_ds = build_pretraining_datasets(
                    data_args.dataset_repo_id,
                    processor,
                    train_split=data_args.dataset_train_split,
                    validation_split=data_args.dataset_validation_split,
                    num_proc=data_args.preprocessing_num_workers or None,
                    cache_dir=data_args.dataset_cache_dir,
                )
        if data_args.train_fraction < 1:
            full_n = len(train_ds)
            # Hold the step budget at the full-data run so only unique data varies.
            if training_args.max_steps <= 0:
                training_args.max_steps = full_data_max_steps(full_n, training_args)
                resolved["training"]["max_steps"] = training_args.max_steps
            train_ds = subsample_train(
                train_ds, data_args.train_fraction, data_args.train_subset_seed
            )
            if training_args.process_index == 0:
                print(
                    f"[data] train_fraction={data_args.train_fraction}: "
                    f"{len(train_ds):,} of {full_n:,} spectra, "
                    f"max_steps={training_args.max_steps:,}",
                    flush=True,
                )
        resolved["data"]["train_examples"] = len(train_ds)
        include_probes = training_args.probe_execution == "inline"
        denoising_datasets = denoising_processor = None
        retrieval_datasets = retrieval_evaluation_datasets = None
        if include_probes and training_args.denoise_steps:
            with training_args.main_process_first(local=False, desc="denoising preprocessing"):
                denoising_datasets, denoising_processor, _ = build_probe_data(
                    "denoise", data_args, training_args, processor
                )
        if include_probes and training_args.retrieval_steps:
            with training_args.main_process_first(local=False, desc="retrieval preprocessing"):
                retrieval_datasets, _, retrieval_evaluation_datasets = build_probe_data(
                    "retrieval", data_args, training_args, processor
                )

        callbacks = build_callbacks(
            model,
            val_ds,
            processor,
            training_args,
            out_dir,
            denoising_datasets=denoising_datasets,
            denoising_processor=denoising_processor,
            retrieval_datasets=retrieval_datasets,
            retrieval_evaluation_datasets=retrieval_evaluation_datasets,
            include_probes=include_probes,
        )
        if env.job_deadline_epoch is not None:
            callbacks.append(
                WalltimeCheckpointCallback(env.job_deadline_epoch, env.checkpoint_margin_seconds)
            )
        if training_args.probe_execution == "sidecar":
            sidecar_callback = SidecarCallback(out_dir, resolved)
            callbacks.append(sidecar_callback)
        trainer = IonaTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=IonaDataCollatorForPreTraining(
                mask_ratio=training_args.mask_ratio,
                pad_to_multiple_of=training_args.pad_to_multiple_of,
            ),
            processing_class=processor,
            use_denoising_probe=include_probes and bool(training_args.denoise_steps),
            use_retrieval_probe=include_probes and bool(training_args.retrieval_steps),
        )
        # Force Trainer to report the validation loss.
        trainer.can_return_loss = True
        for callback in callbacks:
            trainer.add_callback(callback)

        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        if trainer.is_world_process_zero():
            trainer.save_model(str(out_dir / "final"))
            panels = render_bias_panels(model.iona.bias_module, trainer.state.global_step)
            for name, fig in panels.items():
                fig.savefig(out_dir / "figs" / f"{name.replace('/', '_')}_final.png", dpi=110)
                plt.close(fig)
        return 0
    finally:
        if sidecar_callback is not None:
            sidecar_callback.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    sys.exit(main())
