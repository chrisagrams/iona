"""Train the masked-intensity model."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from accelerate.utils import DeepSpeedPlugin
from transformers import HfArgumentParser, Trainer, set_seed

from msdelta.utils.callbacks import SidecarCallback, build_callbacks
from msdelta.models.configuration_msdelta import MSDeltaConfig
from msdelta.data.data import build_pretraining_datasets, load_pretraining_datasets_from_disk
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining
from msdelta.pretraining.length_grouping import GlobalLengthGroupedSampler, spectrum_lengths
from msdelta.pretraining.posttraining import build_probe_data
from msdelta.pretraining.proposal_loss import tempered_intensity_kl
from msdelta.models.processing_msdelta import MSDeltaDataCollatorForPreTraining, MSDeltaProcessor
from msdelta.pretraining.training_args import DataArguments, ModelArguments, MSDeltaTrainingArguments
from msdelta.utils.viz import render_bias_panels
from msdelta.utils.wandb_distributed import init_wandb_run


class MSDeltaTrainer(Trainer):
    """Configure separate DeepSpeed plugins for pretraining and frozen-encoder probes."""

    def __init__(
        self, *args, use_denoising_probe: bool = False, use_retrieval_probe: bool = False, **kwargs
    ):
        self.use_denoising_probe = use_denoising_probe
        self.use_retrieval_probe = use_retrieval_probe
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        # K189-P: length-grouped global batches (see msdelta.pretraining.length_grouping).
        if not getattr(self.args, "length_grouped_batches", False):
            return super()._get_train_sampler(train_dataset)
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        global_batch = (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
                        * self.args.world_size)
        return GlobalLengthGroupedSampler(spectrum_lengths(dataset), global_batch,
                                          megabatches=self.args.length_group_megabatches, seed=self.args.seed)

    # K195a-P: with proposal_intensity_power set, every forward also computes the proposal loss. The optimised
    # objective is today's loss (default) or the proposal (train_on_proposal_loss). Logs keep today's loss under its
    # usual names ('loss', 'eval_loss') and add 'proposal_loss' / 'eval_proposal_loss': spectrum-weighted means since
    # the last log, summed over all ranks. Without proposal_intensity_power nothing here runs.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        power = getattr(self.args, "proposal_intensity_power", None)
        if power is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
        proposal = tempered_intensity_kl(logits, inputs["labels"], inputs["mask_positions"], power)
        n = float(inputs["mz"].shape[0])
        sums = torch.stack([loss.detach().float() * n, proposal.detach().float() * n, loss.new_tensor(n).float()])
        key = "train" if model.training else "eval"
        acc = getattr(self, "_k195_sums", None)
        if acc is None:
            acc = self._k195_sums = {}
        acc[key] = sums if key not in acc else acc[key] + sums
        # Only training optimises the proposal; evaluation returns today's loss, so eval_loss keeps its meaning.
        objective = proposal if (self.args.train_on_proposal_loss and model.training) else loss
        return (objective, outputs) if return_outputs else objective

    def log(self, logs, start_time=None):
        if getattr(self.args, "proposal_intensity_power", None) is not None and ("loss" in logs or "eval_loss" in logs):
            key = "eval" if "eval_loss" in logs else "train"
            acc = getattr(self, "_k195_sums", None) or {}
            sums = acc.pop(key, None)
            if sums is None:
                sums = torch.zeros(3, device=self.args.device)
            sums = self.accelerator.reduce(sums, reduction="sum")  # collective: log() runs on every rank
            if float(sums[2]) > 0:
                mean_loss, mean_proposal = (float(sums[0] / sums[2]), float(sums[1] / sums[2]))
                if key == "train":
                    logs["objective_loss"] = logs["loss"]  # what the optimiser saw (Trainer's own running mean)
                    logs["loss"], logs["proposal_loss"] = round(mean_loss, 6), round(mean_proposal, 6)
                else:
                    logs["eval_proposal_loss"] = mean_proposal
                    logs["eval_loss_check"] = mean_loss  # same quantity as eval_loss, from our own sums
        return super().log(logs, start_time) if start_time is not None else super().log(logs)

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


def main(argv: list[str] | None = None) -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.xpu.is_available():
        torch.xpu.set_device(local_rank)

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
    if training_args.torch_compile and training_args.compile_static_shapes:
        # K189-P: one static graph per padded shape (<= max_peaks / pad_to_multiple_of of them).
        import torch._dynamo as dynamo  # bound as `dynamo`: a bare `import torch.x` here would make `torch` local
        dynamo.config.automatic_dynamic_shapes = False
        dynamo.config.recompile_limit = training_args.compile_recompile_limit
        dynamo.config.accumulated_recompile_limit = max(
            dynamo.config.accumulated_recompile_limit, 4 * training_args.compile_recompile_limit)
    model_config = MSDeltaConfig.from_pretrained(model_args.config_name)
    if model_args.config_overrides is not None:
        model_config.update_from_string(model_args.config_overrides)
        model_config._validate()
    processor_overrides = {}
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

    sidecar_callback = None
    wandb_run = None
    if training_args.wandb_project:
        wandb_run = init_wandb_run(
            project=training_args.wandb_project,
            run_name=training_args.run_name,
            config=resolved,
            shared=training_args.probe_execution == "sidecar",
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
        if training_args.probe_execution == "sidecar":
            sidecar_callback = SidecarCallback(out_dir, resolved)
            callbacks.append(sidecar_callback)
        trainer = MSDeltaTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=MSDeltaDataCollatorForPreTraining(mask_ratio=training_args.mask_ratio,
                                                            pad_to_multiple_of=training_args.pad_to_multiple_of),
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
            panels = render_bias_panels(model.msdelta.bias_module, trainer.state.global_step)
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
