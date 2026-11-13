"""Build the de novo training and evaluation splits once, to disk.

    python scripts/preprocess_denovo.py \
        --args_file configs/finetune/denovo-massivekb-50m/training.args \
        --preprocessed_dataset_dir /path/to/denovo-preprocessed

Also downloads the encoder checkpoint, so a job running offline can load it from HF_HOME.
Training reads the splits with the same --preprocessed_dataset_dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import HfArgumentParser

from iona.finetune_denovo import DenovoDataArguments, DenovoModelArguments, build_denovo_datasets
from iona.processing_iona import IonaProcessor
from iona.tokenization_iona import IonaPeptideTokenizer


def main(argv: list[str] | None = None) -> int:
    parser = HfArgumentParser((DenovoModelArguments, DenovoDataArguments))  # pyright: ignore[reportArgumentType]
    # Training flags in the args file are ignored.
    model_args, data_args, _ignored = parser.parse_args_into_dataclasses(
        args=argv, args_file_flag="--args_file", return_remaining_strings=True
    )
    if not data_args.preprocessed_dataset_dir:
        sys.exit("--preprocessed_dataset_dir is required: it is where the splits are written")

    for name in (model_args.pretrained_path, data_args.processor_name_or_path):
        if name and not Path(name).exists():
            snapshot_download(name)
    processor = IonaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks,
    )
    datasets = build_denovo_datasets(model_args, data_args, processor, IonaPeptideTokenizer())
    datasets.save_to_disk(
        data_args.preprocessed_dataset_dir, num_proc=data_args.preprocessing_num_workers or None
    )
    print(f"[denovo] saved train={len(datasets['train']):,} "
          f"evaluation={len(datasets['evaluation']):,} to {data_args.preprocessed_dataset_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
