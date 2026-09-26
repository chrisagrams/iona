"""Compute the frozen teacher's spectrum embeddings once, to disk.

    python -m msdelta.precompute_align --args_file configs/finetune-align-100k-50m/training.args \
        --target_cache /path/to/cache

Separate from training on purpose. The teacher never learns, so its embedding for a
spectrum is the same in epoch 10 as in epoch 1 and there is no reason to recompute it --
but the stronger reason is that every alignment run that has faulted on twelve tiles
loaded the 49.8M teacher onto the device, used it, and abandoned it, while the bisect
that runs clean never loads one and the denoise fine-tune never abandons one (its encoder
IS the trained model). Doing this in its own process means the training job never
constructs a teacher at all, which removes that difference rather than reasoning about it.

Runs on ONE tile. It is pure inference over a few thousand spectra and takes a couple of
minutes; there is nothing to parallelise that would be worth the coordination.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import HfArgumentParser

from dataclasses import dataclass, field

from msdelta.finetune_align import AlignDataArguments, AlignModelArguments
from msdelta.finetune_denoise import subset_splits
from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.reranking import attach_teacher_embeddings, build_alignment_datasets


@dataclass
class PrecomputeArguments:
    """Deliberately NOT TrainingArguments.

    Parsing TrainingArguments pulls in HF's device and distributed validation -- it
    refuses bf16 without a GPU, and with a ddp_backend set it asks accelerate for a
    world size before any process group exists. None of that is relevant to a single
    process doing inference, so this takes the three settings that are, and the training
    flags in the shared args file are accepted and ignored.
    """

    batch_size: int = field(default=16, metadata={"help": "spectra per teacher forward"})
    seed: int = field(default=0, metadata={"help": "must match the run that consumes this"})
    # Sharded build, for caches too big for one tile inside a debug hour
    # (ms-contrastive-100k: ~283k spectra, ~1.5 h at 50m on one tile):
    #   stage prepare   one process: build the flattened splits, save to <cache>/_flat
    #   stage shard     one process per tile: contiguous shard i of n -> <cache>/_shards
    #   stage merge     one process: concatenate shards IN ORDER -> <cache>/{train,validation}
    # Contiguous shards concatenated in index order are exactly the unsharded row order,
    # so the merged cache is the one stage "all" (the default, unsharded) would write.
    stage: str = field(default="all", metadata={"help": "all, prepare, shard or merge"})
    pad_spectra_to: int = field(
        default=0, metadata={"help": "shard stage: fixed spectrum width per batch (use "
                                     "max_peaks when 12 shards share a node; see FT16)"})
    num_shards: int = 1
    shard_index: int = 0


def main(argv: list[str] | None = None) -> int:
    parser = HfArgumentParser(
        (AlignModelArguments, AlignDataArguments, PrecomputeArguments)  # pyright: ignore[reportArgumentType]
    )
    model_args, data_args, precompute_args, _ignored = parser.parse_args_into_dataclasses(
        args=argv, args_file_flag="--args_file", return_remaining_strings=True
    )
    cache = Path(data_args.target_cache or "")
    if not cache.name:
        sys.exit("--target_cache is required: it is where the embeddings are written")

    device = "xpu" if torch.xpu.is_available() else "cpu"
    stage = precompute_args.stage
    if stage not in ("all", "prepare", "shard", "merge"):
        sys.exit(f"--stage must be all, prepare, shard or merge, not {stage!r}")
    if stage == "shard":
        return _shard(cache, model_args, precompute_args, device)
    if stage == "merge":
        return _merge(cache, model_args, data_args, precompute_args)
    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks,
    )

    from msdelta.grouped_retrieval import load_spectrum_datasets
    datasets = load_spectrum_datasets(
        data_args.dataset_format, data_args.dataset_repo, processor,
        include_consensus=data_args.include_consensus,
        exclude_replicate_peptides=data_args.exclude_replicate_peptides,
        num_proc=data_args.preprocessing_num_workers or None,
        validation_fraction=data_args.validation_fraction,
        seed=precompute_args.seed,
    )
    datasets = subset_splits(datasets, data_args.max_samples)
    print(f"[precompute] device={device} "
          + " ".join(f"{k}={len(v):,}" for k, v in datasets.items()), flush=True)
    if stage == "prepare":
        for name, split in datasets.items():
            split.save_to_disk(str(cache / "_flat" / name))
        print(f"[precompute] prepared flat splits under {cache / '_flat'}", flush=True)
        return 0

    teacher = MSDeltaForPreTraining.from_pretrained(model_args.pretrained_path)
    datasets = attach_teacher_embeddings(
        datasets, teacher, model_args.pooling,
        batch_size=precompute_args.batch_size,
        max_peptide_length=model_args.max_peptide_length, device=device)

    return _write_cache(cache, datasets, model_args, data_args, precompute_args)


def _shard(cache, model_args, precompute_args, device) -> int:
    from datasets import load_from_disk

    n, i = precompute_args.num_shards, precompute_args.shard_index
    teacher = MSDeltaForPreTraining.from_pretrained(model_args.pretrained_path)
    for name in ("train", "validation"):
        flat = cache / "_flat" / name
        out = cache / "_shards" / name / f"{i:03d}"
        if not flat.exists():
            continue
        if out.exists():
            # A rerun after some shards faulted: finished pieces are kept, not recomputed.
            print(f"[precompute] shard {i}/{n} {name}: already done", flush=True)
            continue
        part = load_from_disk(str(flat)).shard(num_shards=n, index=i, contiguous=True)
        part = attach_teacher_embeddings(
            {name: part}, teacher, model_args.pooling,
            batch_size=precompute_args.batch_size,
            max_peptide_length=model_args.max_peptide_length, device=device,
            pad_spectra_to=getattr(precompute_args, "pad_spectra_to", 0))[name]
        part.save_to_disk(str(out))
        print(f"[precompute] shard {i}/{n} {name}: {len(part):,} rows", flush=True)
    return 0


def _merge(cache, model_args, data_args, precompute_args) -> int:
    from datasets import concatenate_datasets, load_from_disk

    n = precompute_args.num_shards
    datasets = {}
    for name in ("train", "validation"):
        parts = [cache / "_shards" / name / f"{i:03d}" for i in range(n)]
        if not parts[0].exists():
            continue
        missing = [str(p) for p in parts if not p.exists()]
        if missing:
            sys.exit(f"missing shards: {missing[:3]}")
        datasets[name] = concatenate_datasets([load_from_disk(str(p)) for p in parts])
        expected = len(load_from_disk(str(cache / "_flat" / name)))
        if len(datasets[name]) != expected:
            sys.exit(f"{name}: merged {len(datasets[name])} rows, expected {expected}")
    return _write_cache(cache, datasets, model_args, data_args, precompute_args)


def _write_cache(cache, datasets, model_args, data_args, precompute_args) -> int:
    cache.mkdir(parents=True, exist_ok=True)
    for name, split in datasets.items():
        split.save_to_disk(str(cache / name))
    width = len(datasets["train"][0]["target"])
    # The split is by PEPTIDE, and it is written here rather than recomputed at train
    # time: re-splitting with a different seed would put replicates of a held-out peptide
    # into training and turn the validation number into recall.
    (cache / "MANIFEST.txt").write_text(
        f"teacher: {model_args.pretrained_path}\n"
        f"pooling: {model_args.pooling}\n"
        f"embedding_size: {width}\n"
        f"max_peaks: {data_args.max_peaks}\n"
        f"validation_fraction: {data_args.validation_fraction}\n"
        f"seed: {precompute_args.seed}\n"
        f"dataset: {data_args.dataset_repo} ({data_args.dataset_format}, "
        f"include_consensus={data_args.include_consensus}, "
        f"exclude_replicate_peptides={data_args.exclude_replicate_peptides})\n"
        + "".join(f"{k}: {len(v)}\n" for k, v in datasets.items())
    )
    print(f"[precompute] wrote {width}-d targets to {cache}", flush=True)
    print((cache / "MANIFEST.txt").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
