"""Shared setup for inline and independent posttraining probes."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_from_disk
from transformers import TrainingArguments, set_seed

from iona.data import (
    build_denoising_datasets,
    build_retrieval_datasets,
    build_retrieval_evaluation_datasets,
)
from iona.denoising import run_denoising_probe
from iona.env import RankEnv
from iona.modeling_iona import IonaForPreTraining
from iona.processing_iona import IonaProcessor
from iona.retrieval import run_retrieval_probe
from iona.wandb_distributed import init_wandb_run


def denoise_training_args(training_args, out_dir, *, ddp_backend=None):
    return TrainingArguments(
        output_dir=str(out_dir / "denoise-probes"),
        num_train_epochs=training_args.denoise_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=training_args.denoise_learning_rate,
        weight_decay=training_args.denoise_weight_decay,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=training_args.denoise_num_workers,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        seed=training_args.denoise_seed,
        data_seed=training_args.denoise_seed,
        report_to=[],
        ddp_find_unused_parameters=False,
        ddp_backend=ddp_backend,
    )


def retrieval_training_args(training_args, out_dir, *, ddp_backend=None):
    return TrainingArguments(
        output_dir=str(out_dir / "retrieval-probes"),
        num_train_epochs=training_args.retrieval_epochs,
        per_device_train_batch_size=training_args.retrieval_per_device_batch_size,
        per_device_eval_batch_size=training_args.retrieval_per_device_batch_size,
        learning_rate=training_args.retrieval_learning_rate,
        weight_decay=training_args.retrieval_weight_decay,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        remove_unused_columns=False,
        label_names=["group_ids"],
        dataloader_num_workers=training_args.retrieval_num_workers,
        dataloader_drop_last=True,
        prediction_loss_only=True,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        seed=training_args.retrieval_seed,
        data_seed=training_args.retrieval_seed,
        report_to=[],
        ddp_find_unused_parameters=False,
        ddp_backend=ddp_backend,
    )


def build_probe_data(kind, data_args, training_args, processor):
    """Build only the datasets and processor needed by one probe."""
    num_proc = data_args.preprocessing_num_workers or None
    if kind == "denoise":
        processor = IonaProcessor.from_pretrained(
            data_args.processor_name_or_path,
            max_peaks=training_args.denoise_max_peaks,
        )
        if getattr(data_args, "preprocessed_probe_dir", None):
            datasets = load_from_disk(Path(data_args.preprocessed_probe_dir) / "denoise")
            return datasets, processor, None
        datasets = build_denoising_datasets(
            training_args.denoise_dataset_repo, processor, num_proc=num_proc
        )
        return datasets, processor, None
    if kind != "retrieval":
        raise ValueError(f"Unknown probe: {kind}")
    if getattr(data_args, "preprocessed_probe_dir", None):
        root = Path(data_args.preprocessed_probe_dir)
        datasets = load_from_disk(root / "retrieval")
        evaluation = load_from_disk(root / "retrieval-evaluation")
        return datasets, processor, evaluation
    datasets = build_retrieval_datasets(
        training_args.retrieval_dataset_repo, processor, num_proc=num_proc
    )
    evaluation = build_retrieval_evaluation_datasets(
        datasets["validation"],
        processor,
        max_analytes=training_args.retrieval_validation_analytes,
        replicate_repo_id=training_args.replicate_retrieval_repo,
        num_proc=num_proc,
    )
    return datasets, processor, evaluation


def main(argv: list[str] | None = None) -> int:
    """Train one probe, log directly to the shared W&B run, and exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", choices=("denoise", "retrieval"), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    cli = parser.parse_args(argv)
    if cli.wandb and not (cli.wandb_run_id and cli.wandb_project):
        parser.error("--wandb requires --wandb-run-id and --wandb-project")
    resolved = json.loads(cli.settings_json)
    args = SimpleNamespace(**resolved["training"])
    data_args = SimpleNamespace(**resolved["data"])
    out_dir = Path(args.output_dir)
    env = RankEnv()
    local_rank = env.local_rank or 0
    if cli.device != "cpu":
        backend = torch.xpu if cli.device == "xpu" else torch.cuda
        if backend.device_count() != env.local_world_size:
            raise ValueError("Posttraining requires one visible accelerator per local worker")
        backend.set_device(local_rank)
    device = torch.device(cli.device if cli.device == "cpu" else f"{cli.device}:{local_rank}")
    set_seed(args.denoise_seed if cli.probe == "denoise" else args.retrieval_seed)
    run = None
    if cli.wandb and env.rank == 0:
        run = init_wandb_run(
            project=cli.wandb_project,
            run_name="",
            config={},
            shared=True,
            role=f"sidecar-{cli.probe}",
            run_id=cli.wandb_run_id,
            entity=cli.wandb_entity,
        )
    try:
        ddp_backend = "xccl" if cli.device == "xpu" else None
        probe_args = (
            denoise_training_args(args, out_dir, ddp_backend=ddp_backend)
            if cli.probe == "denoise"
            else retrieval_training_args(args, out_dir, ddp_backend=ddp_backend)
        )
        processor = IonaProcessor(**resolved["processor"])
        with probe_args.main_process_first(desc="probe dataset preparation"):
            datasets, processor, evaluation = build_probe_data(
                cli.probe, data_args, args, processor
            )
        module = IonaForPreTraining.from_pretrained(cli.checkpoint).to(device)
        destination = out_dir / f"{cli.probe}-probes" / f"step-{cli.step}"
        if cli.probe == "denoise":
            metrics = run_denoising_probe(
                module,
                datasets["train"],
                datasets["validation"],
                output_dir=destination,
                processor=processor,
                peak_pair_budget=args.denoise_peak_pair_budget,
                hidden_size=args.denoise_head_hidden_size,
                dropout=args.denoise_head_dropout,
                training_args=probe_args,
            )
        else:
            metrics = run_retrieval_probe(
                module,
                datasets["train"],
                datasets["validation"],
                output_dir=destination,
                processor=processor,
                evaluation_datasets=evaluation,
                projection_hidden_size=args.retrieval_projection_hidden_size,
                embedding_size=args.retrieval_embedding_size,
                dropout=args.retrieval_head_dropout,
                temperature=args.retrieval_temperature,
                training_args=probe_args,
            )
        if run is not None:
            run.define_metric("train/global_step")
            for name in metrics:
                run.define_metric(name, step_metric="train/global_step", step_sync=False)
            run.log({**metrics, "train/global_step": cli.step})
        return 0
    finally:
        if run is not None:
            run.finish()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
