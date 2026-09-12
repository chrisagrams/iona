"""Build the pretraining dataset cache in a single process."""

from __future__ import annotations

import sys

from datasets import DatasetDict
from transformers import HfArgumentParser

from msdelta.data import build_pretraining_datasets
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.training_args import DataArguments


def main(argv: list[str] | None = None) -> int:
    """Preprocess the train and validation datasets without distributed setup."""
    parser = HfArgumentParser(DataArguments)
    parsed = parser.parse_args_into_dataclasses(
        args=argv,
        return_remaining_strings=True,
        args_file_flag="--args_file",
    )
    data_args, _ = parsed

    processor_overrides = {}
    if data_args.max_peaks is not None:
        processor_overrides["max_peaks"] = data_args.max_peaks
    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path,
        **processor_overrides,
    )

    workers = data_args.preprocessing_num_workers or None
    print(
        f"[data] preprocessing {data_args.dataset_repo_id} with "
        f"{workers or 1} CPU worker(s)",
        flush=True,
    )
    train_ds, validation_ds = build_pretraining_datasets(
        data_args.dataset_repo_id,
        processor,
        train_split=data_args.dataset_train_split,
        validation_split=data_args.dataset_validation_split,
        num_proc=workers,
        cache_dir=data_args.dataset_cache_dir,
        # Loading a multiprocessing map cache is serial in Hugging Face Datasets.
        # Recompute in parallel when building the finalized on-disk artifact.
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
