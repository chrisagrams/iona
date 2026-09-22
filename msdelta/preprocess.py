"""Build finalized pretraining and probe datasets in a single process."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

from datasets import DatasetDict
from transformers import HfArgumentParser

from msdelta.data import build_pretraining_datasets
from msdelta.posttraining import build_probe_data
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.training_args import DataArguments


@dataclass
class ProbePreprocessingArguments:
    """Probe data settings, without initializing a distributed Trainer."""

    probe_execution: str = "inline"
    denoise_steps: int = 0
    retrieval_steps: int = 0
    denoise_dataset_repo: str = "chrisagrams/ms-denoise-100k"
    retrieval_dataset_repo: str = "chrisagrams/ms-contrastive-100k"
    denoise_max_peaks: int = 1024
    retrieval_validation_analytes: int = 1000
    replicate_retrieval_repo: str | None = None
    probes_only: bool = False


def main(argv: list[str] | None = None) -> int:
    """Preprocess the train and validation datasets without distributed setup."""
    parser = HfArgumentParser((DataArguments, ProbePreprocessingArguments))
    parsed = parser.parse_args_into_dataclasses(
        args=argv,
        return_remaining_strings=True,
        args_file_flag="--args_file",
    )
    data_args, probe_args, _ = parsed

    processor_overrides = {}
    if data_args.max_peaks is not None:
        processor_overrides["max_peaks"] = data_args.max_peaks
    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path,
        **processor_overrides,
    )

    workers = data_args.preprocessing_num_workers or None
    if not probe_args.probes_only:
        print(
            f"[data] preprocessing {data_args.dataset_repo_id} with {workers or 1} CPU worker(s)",
            flush=True,
        )
        train_ds, validation_ds = build_pretraining_datasets(
            data_args.dataset_repo_id,
            processor,
            train_split=data_args.dataset_train_split,
            validation_split=data_args.dataset_validation_split,
            num_proc=workers,
            cache_dir=data_args.dataset_cache_dir,
            load_from_cache_file=False if data_args.preprocessed_dataset_dir else None,
        )
        print(
            f"[data] cache ready: train={len(train_ds):,}, validation={len(validation_ds):,}",
            flush=True,
        )
        if data_args.preprocessed_dataset_dir:
            print(
                f"[data] saving finalized dataset to {data_args.preprocessed_dataset_dir}",
                flush=True,
            )
            DatasetDict(
                {
                    data_args.dataset_train_split: train_ds,
                    data_args.dataset_validation_split: validation_ds,
                }
            ).save_to_disk(data_args.preprocessed_dataset_dir, num_proc=workers)
            print("[data] finalized dataset ready", flush=True)

    if data_args.preprocessed_probe_dir and probe_args.probe_execution != "off":
        root = Path(data_args.preprocessed_probe_dir)
        # Build from the source repositories, not from a previous finalized copy.
        source_args = replace(data_args, preprocessed_probe_dir=None)
        for kind, steps in (
            ("denoise", probe_args.denoise_steps),
            ("retrieval", probe_args.retrieval_steps),
        ):
            if steps <= 0:
                continue
            print(f"[data] preprocessing {kind} probes", flush=True)
            datasets, _, evaluation = build_probe_data(kind, source_args, probe_args, processor)
            datasets.save_to_disk(str(root / kind), num_proc=workers)
            if evaluation is not None:
                DatasetDict(evaluation).save_to_disk(
                    str(root / "retrieval-evaluation"), num_proc=workers
                )
            print(f"[data] finalized {kind} probes ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
