"""Train the masked-intensity model."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from transformers import HfArgumentParser, Trainer, set_seed

from msdelta.callbacks import DenoisingProbeCallback, build_callbacks
from msdelta.configuration_msdelta import MSDeltaConfig
from msdelta.data import (
    build_denoising_datasets,
    build_pretraining_datasets,
    resolve_dataset_paths,
)
from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.processing_msdelta import MSDeltaDataCollatorForPreTraining, MSDeltaProcessor
from msdelta.training_args import DataArguments, ModelArguments, MSDeltaTrainingArguments
from msdelta.viz import render_bias_panels
from msdelta.wandb_distributed import init_wandb_run


class MSDeltaTrainer(Trainer):
    """Exclude Fourier frequencies from weight decay."""

    def get_decay_parameter_names(self, model):
        return [n for n in super().get_decay_parameter_names(model) if not n.endswith(".freqs")]


def main(argv: list[str] | None = None) -> int:
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, MSDeltaTrainingArguments)  # pyright: ignore[reportArgumentType]
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=argv,
        args_file_flag="--args_file",
    )

    out_dir = Path(training_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(exist_ok=True)

    if training_args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", training_args.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(out_dir))

    set_seed(training_args.seed)
    model_config = MSDeltaConfig.from_pretrained(model_args.config_name)
    if model_args.config_overrides is not None:
        model_config.update_from_string(model_args.config_overrides)
        model_config._validate()
    processor_overrides = {}
    if data_args.intensity_threshold_frac is not None:
        processor_overrides["intensity_threshold_frac"] = data_args.intensity_threshold_frac
    if data_args.max_peaks is not None:
        processor_overrides["max_peaks"] = data_args.max_peaks
    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path,
        **processor_overrides,
    )
    model = MSDeltaForPreTraining(model_config)
    resolved = {
        "model": model_config.to_dict(),
        "processor": processor.to_dict(),
        "data": asdict(data_args),
        "training": training_args.to_dict(),
    }

    wandb_run = None
    if training_args.wandb_project:
        wandb_run = init_wandb_run(
            project=training_args.wandb_project,
            run_name=training_args.run_name,
            config=resolved,
        )

    try:
        if training_args.process_index == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[model] {n_params / 1e6:.2f}M params", flush=True)

        # Create the shared dataset cache on rank 0.
        with training_args.main_process_first(local=False, desc="dataset preprocessing"):
            if training_args.process_index == 0:
                print(
                    f"[data] preprocessing with {data_args.preprocessing_num_workers} CPU workers",
                    flush=True,
                )
            train_paths, val_paths = resolve_dataset_paths(
                root=data_args.dataset_root,
                repo_id=data_args.dataset_repo_id,
                train_split=data_args.dataset_train_split,
                validation_split=data_args.dataset_validation_split,
                num_validation_files=data_args.num_validation_files,
            )
            train_ds, val_ds = build_pretraining_datasets(
                train_paths,
                val_paths,
                processor,
                num_proc=data_args.preprocessing_num_workers or None,
            )
        eval_size = training_args.validation_batches * training_args.per_device_eval_batch_size
        eval_ds = val_ds.select(range(min(len(val_ds), eval_size)))
        denoising_datasets = None
        denoising_processor = None
        if training_args.denoise_steps:
            denoising_processor = MSDeltaProcessor.from_pretrained(
                data_args.processor_name_or_path,
                max_peaks=training_args.denoise_max_peaks,
                intensity_threshold_frac=training_args.denoise_intensity_threshold_frac,
            )
            with training_args.main_process_first(local=False, desc="denoising preprocessing"):
                denoising_datasets = build_denoising_datasets(
                    training_args.denoise_dataset_repo,
                    denoising_processor,
                    num_proc=data_args.preprocessing_num_workers or None,
                )

        trainer = MSDeltaTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=MSDeltaDataCollatorForPreTraining(mask_ratio=training_args.mask_ratio),
            processing_class=processor,
        )
        # Force Trainer to report the validation loss.
        trainer.can_return_loss = True
        callbacks = build_callbacks(
            model,
            val_ds,
            processor,
            training_args,
            out_dir,
            denoising_datasets=denoising_datasets,
            denoising_processor=denoising_processor,
        )
        for cb in callbacks:
            trainer.add_callback(cb)

        denoising_callback = next(
            (cb for cb in callbacks if isinstance(cb, DenoisingProbeCallback)),
            None,
        )
        checkpoint = training_args.resume_from_checkpoint
        while True:
            trainer.train(resume_from_checkpoint=checkpoint)
            if denoising_callback is None or denoising_callback.pending_step is None:
                break

            step = denoising_callback.pending_step
            checkpoint = str(out_dir / f"checkpoint-{step}")
            if not Path(checkpoint).is_dir():
                raise RuntimeError(f"expected checkpoint was not saved: {checkpoint}")
            denoising_callback.run_pending_probe()

        if trainer.is_world_process_zero():
            trainer.save_model(str(out_dir / "final"))
            panels = render_bias_panels(model.msdelta.bias_module, trainer.state.global_step)
            for name, fig in panels.items():
                fig.savefig(out_dir / "figs" / f"{name.replace('/', '_')}_final.png", dpi=110)
                plt.close(fig)
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    sys.exit(main())
