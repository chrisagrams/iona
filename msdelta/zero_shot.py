"""Evaluate frozen MSDelta encoder embeddings across pretrained checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset
from tqdm.auto import tqdm

from msdelta.data import (
    build_retrieval_evaluation_datasets,
    build_retrieval_validation_dataset,
)
from msdelta.embedding import embed_spectra
from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.retrieval import retrieval_metrics


def all_but_top(embeddings: np.ndarray, components: int) -> np.ndarray:
    """Center embeddings and remove their leading principal directions."""
    if components <= 0:
        return embeddings
    limit = min(embeddings.shape)
    if components >= limit:
        raise ValueError(
            f"all-but-top components ({components}) must be smaller than "
            f"min(n_spectra, embedding_size) ({limit})"
        )
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    top = directions[:components]
    return centered - (centered @ top.T) @ top


def _dataset_inputs(
    dataset: Dataset,
    indices: np.ndarray,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], np.ndarray]:
    subset = dataset.select(indices.tolist())
    spectra = [
        (
            torch.tensor(mz, dtype=torch.float32),
            torch.tensor(log_intensity, dtype=torch.float32),
        )
        for mz, log_intensity in zip(subset["mz"], subset["log_intensity"])
    ]
    labels = np.asarray(dataset["retrieval_labels"], dtype=np.int64)
    if len(labels) < 2:
        raise ValueError("zero-shot retrieval requires at least two spectra")
    if max(np.bincount(labels, minlength=1)) < 2:
        raise ValueError("zero-shot retrieval requires at least one replicated analyte")
    return spectra, labels


def _gather_embeddings(
    embeddings: np.ndarray,
    row_ids: np.ndarray,
    accelerator: Accelerator | None,
    device: torch.device,
) -> np.ndarray:
    """Gather unevenly sharded embeddings and restore dataset row order."""
    if accelerator is None or accelerator.num_processes == 1:
        return embeddings
    local_embeddings = torch.as_tensor(embeddings, device=device)
    local_ids = torch.as_tensor(row_ids, device=device, dtype=torch.long)
    local_embeddings = accelerator.pad_across_processes(
        local_embeddings, dim=0, pad_index=0
    )
    local_ids = accelerator.pad_across_processes(local_ids, dim=0, pad_index=-1)
    gathered_embeddings = accelerator.gather(local_embeddings)
    gathered_ids = accelerator.gather(local_ids)
    keep = gathered_ids >= 0
    gathered_ids = gathered_ids[keep]
    order = torch.argsort(gathered_ids)
    return gathered_embeddings[keep][order].float().cpu().numpy()


@torch.no_grad()
def evaluate_zero_shot(
    module: MSDeltaForPreTraining,
    datasets: dict[str, Dataset],
    device: torch.device,
    *,
    batch_size: int = 128,
    all_but_top_components: int = 0,
    accelerator: Accelerator | None = None,
) -> dict[str, float]:
    """Score mean/max-pooled base-encoder embeddings without task training."""
    encoder = module.msdelta
    was_training = encoder.training
    metrics: dict[str, float] = {}
    rank = accelerator.process_index if accelerator is not None else 0
    world_size = accelerator.num_processes if accelerator is not None else 1
    is_main_process = accelerator is None or accelerator.is_main_process
    faiss_gpus = [device.index or 0] if device.type == "cuda" else None
    try:
        for name, dataset in datasets.items():
            row_ids = np.arange(rank, len(dataset), world_size)
            spectra, labels = _dataset_inputs(dataset, row_ids)
            local_embeddings = embed_spectra(
                encoder,
                spectra,
                device,
                batch_size=batch_size,
                progress_desc=f"{name} embeddings" if is_main_process else None,
            )
            embeddings = _gather_embeddings(
                local_embeddings, row_ids, accelerator, device
            )
            if is_main_process:
                raw = retrieval_metrics(embeddings, labels, device, gpus=faiss_gpus)
                metrics.update(
                    {f"zero_shot/{name}/{key}": value for key, value in raw.items()}
                )
                if all_but_top_components:
                    corrected = all_but_top(embeddings, all_but_top_components)
                    scores = retrieval_metrics(corrected, labels, device, gpus=faiss_gpus)
                    metrics.update(
                        {f"zero_shot/{name}/{key}_abt": value for key, value in scores.items()}
                    )
            if accelerator is not None:
                accelerator.wait_for_everyone()
    finally:
        if was_training:
            encoder.train()
    return metrics


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def _expand_checkpoints(paths: list[Path]) -> list[Path]:
    """Accept checkpoint directories or run directories containing checkpoints."""
    checkpoints: list[Path] = []
    for path in paths:
        if (path / "config.json").is_file():
            checkpoints.append(path)
            continue
        children = [child for child in path.glob("checkpoint-*") if child.is_dir()]
        if not children:
            raise ValueError(f"{path} is not a model checkpoint or a run with checkpoints")
        checkpoints.extend(children)
    return sorted(
        dict.fromkeys(checkpoints),
        key=lambda path: (_checkpoint_step(path) is None, _checkpoint_step(path) or 0, str(path)),
    )


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        type=Path,
        nargs="+",
        help="Checkpoint directories, or run directories containing checkpoint-* children.",
    )
    parser.add_argument(
        "--processor-name-or-path",
        type=Path,
        help="Processor to use for every checkpoint (defaults to the first checkpoint).",
    )
    parser.add_argument(
        "--retrieval-dataset-repo",
        default="chrisagrams/ms-contrastive-100k",
        help="Grouped retrieval dataset repository.",
    )
    parser.add_argument(
        "--replicate-retrieval-repo",
        help="Optional raw replicate-spectrum benchmark repository.",
    )
    parser.add_argument("--validation-analytes", type=int, default=1000)
    parser.add_argument("--preprocessing-num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--all-but-top-components",
        type=int,
        default=0,
        help="Also report embeddings after removing this many leading directions.",
    )
    parser.add_argument("--device", default="auto", help="Torch device, such as cuda:0 or xpu:0.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    accelerator = Accelerator(cpu=args.device == "cpu")
    checkpoints = _expand_checkpoints(args.checkpoints)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.validation_analytes < 1:
        raise ValueError("validation-analytes must be positive")
    if args.all_but_top_components < 0:
        raise ValueError("all-but-top-components must be nonnegative")
    if accelerator.num_processes > 1 and args.device != "auto":
        raise ValueError("--device must be auto when launched with multiple processes")

    device = accelerator.device if args.device == "auto" else _device(args.device)
    processor_path = args.processor_name_or_path or checkpoints[0]
    processor = MSDeltaProcessor.from_pretrained(processor_path)
    with accelerator.main_process_first():
        validation = build_retrieval_validation_dataset(
            args.retrieval_dataset_repo,
            processor,
            num_proc=args.preprocessing_num_workers or None,
        )
        datasets = build_retrieval_evaluation_datasets(
            validation,
            processor,
            max_analytes=args.validation_analytes,
            replicate_repo_id=args.replicate_retrieval_repo,
            num_proc=args.preprocessing_num_workers or None,
        )

    results = []
    checkpoint_bar = tqdm(
        checkpoints,
        desc="checkpoints",
        unit="checkpoint",
        disable=not accelerator.is_main_process,
    )
    for checkpoint in checkpoint_bar:
        checkpoint_bar.set_postfix_str(checkpoint.name)
        module = MSDeltaForPreTraining.from_pretrained(checkpoint).to(device)
        metrics = evaluate_zero_shot(
            module,
            datasets,
            device,
            batch_size=args.batch_size,
            all_but_top_components=args.all_but_top_components,
            accelerator=accelerator,
        )
        if accelerator.is_main_process:
            result = {
                "checkpoint": str(checkpoint),
                "step": _checkpoint_step(checkpoint),
                "metrics": metrics,
            }
            results.append(result)
            checkpoint_bar.write(json.dumps(result, sort_keys=True))
        accelerator.wait_for_everyone()
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

    if args.output and accelerator.is_main_process:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
