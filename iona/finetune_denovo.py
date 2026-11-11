"""Train an autoregressive peptide decoder on a pretrained Iona encoder for de novo sequencing.

Trains on MassIVE-KB and evaluates on nine-species. Watch `eval_peptide_recall`, not the loss.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import HfArgumentParser, TrainingArguments, set_seed

from iona.configuration_iona import IonaConfig, IonaDenovoConfig
from iona.data import build_massive_kb_denovo_split, build_ninespecies_denovo_split
from iona.denovo import DenovoTrainer
from iona.modeling_iona import IonaForDenovo, IonaForPreTraining
from iona.processing_iona import IonaDataCollatorForDenovo, IonaProcessor
from iona.tokenization_iona import IonaPeptideTokenizer
from iona.wandb_distributed import init_wandb_run


@dataclass
class DenovoModelArguments:
    pretrained_path: str
    random_init: bool = False
    freeze_encoder: bool = True
    decoder_hidden_size: int = 512
    decoder_num_layers: int = 6
    decoder_num_heads: int = 8
    decoder_intermediate_size: int = 2048
    decoder_dropout: float = 0.1
    max_peptide_length: int = 64
    max_charge: int = 10


@dataclass
class DenovoDataArguments:
    processor_name_or_path: str | None = None
    train_dataset_repo: str = "chrisagrams/massive_kb_v1_shuffled"
    eval_dataset_repo: str = "InstaDeepAI/ms_ninespecies_benchmark"
    eval_split: str = "validation"
    max_peaks: int = 512
    preprocessing_num_workers: int = 24
    # 0 keeps every row; MassIVE-KB is already shuffled, so leading rows are a random sample.
    max_train_samples: int = 0
    max_eval_samples: int = 0
    # Output of scripts/preprocess_denovo.py; when set, the splits are loaded instead of built.
    preprocessed_dataset_dir: str | None = None


@dataclass
class DenovoTrainingArguments(TrainingArguments):
    wandb_project: str | None = None
    wandb_entity: str | None = None
    # Spectra decoded at each evaluation; 0 reports only the teacher-forced loss.
    decode_rows: int = 2000
    precursor_tolerance_ppm: float = 50.0


def build_denovo_model(
    model_args: DenovoModelArguments, tokenizer: IonaPeptideTokenizer
) -> IonaForDenovo:
    """Lift the encoder out of a pretraining checkpoint and attach a fresh decoder."""
    if model_args.random_init:
        # Same architecture, no pretrained weights.
        pretrained = IonaForPreTraining(IonaConfig.from_pretrained(model_args.pretrained_path))
    else:
        pretrained = IonaForPreTraining.from_pretrained(model_args.pretrained_path)
    config = IonaDenovoConfig(
        encoder=pretrained.config,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_hidden_size=model_args.decoder_hidden_size,
        decoder_num_layers=model_args.decoder_num_layers,
        decoder_num_heads=model_args.decoder_num_heads,
        decoder_intermediate_size=model_args.decoder_intermediate_size,
        decoder_dropout=model_args.decoder_dropout,
        max_peptide_length=model_args.max_peptide_length,
        max_charge=model_args.max_charge,
    )
    return IonaForDenovo(
        config,
        encoder=pretrained.iona,
        freeze_encoder=model_args.freeze_encoder,
    )


def build_denovo_datasets(
    model_args: DenovoModelArguments,
    data_args: DenovoDataArguments,
    processor: IonaProcessor,
    tokenizer: IonaPeptideTokenizer,
) -> DatasetDict:
    """Tokenize MassIVE-KB for training and nine-species for evaluation."""
    num_proc = data_args.preprocessing_num_workers or None
    train = load_dataset(data_args.train_dataset_repo, split="train")
    if data_args.max_train_samples > 0:
        train = train.select(range(min(data_args.max_train_samples, len(train))))
    train = build_massive_kb_denovo_split(
        train, processor, tokenizer, max_peptide_length=model_args.max_peptide_length,
        max_charge=model_args.max_charge, num_proc=num_proc,
    )
    evaluation = load_dataset(data_args.eval_dataset_repo, split=data_args.eval_split)
    if data_args.max_eval_samples > 0:
        evaluation = evaluation.select(range(min(data_args.max_eval_samples, len(evaluation))))
    evaluation = build_ninespecies_denovo_split(
        evaluation, processor, tokenizer, max_peptide_length=model_args.max_peptide_length,
        max_charge=model_args.max_charge, num_proc=num_proc,
    )
    return DatasetDict(train=train, evaluation=evaluation)


def main(argv: list[str] | None = None) -> int:
    parser = HfArgumentParser(
        (DenovoModelArguments, DenovoDataArguments, DenovoTrainingArguments)  # pyright: ignore[reportArgumentType]
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=argv, args_file_flag="--args_file"
    )
    out_dir = Path(training_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if training_args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", training_args.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(out_dir))
    if training_args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", training_args.wandb_entity)
    set_seed(training_args.seed)
    # The collator builds the model inputs; the loss needs only `labels`.
    training_args.remove_unused_columns = False
    training_args.label_names = ["labels"]
    training_args.prediction_loss_only = True

    processor = IonaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks,
    )
    tokenizer = IonaPeptideTokenizer()
    model = build_denovo_model(model_args, tokenizer)

    wandb_run = None
    if training_args.wandb_project:
        wandb_run = init_wandb_run(
            project=training_args.wandb_project,
            run_name=training_args.run_name,
            entity=training_args.wandb_entity,
            config={"model": asdict(model_args), "data": asdict(data_args),
                    "training": training_args.to_dict()},
        )
    try:
        if training_args.process_index == 0:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            origin = ("RANDOM INIT (control)" if model_args.random_init
                      else model_args.pretrained_path)
            print(f"[denovo] {total / 1e6:.2f}M params, {trainable / 1e6:.2f}M trainable; "
                  f"encoder from {origin}; freeze_encoder={model_args.freeze_encoder}", flush=True)

        if data_args.preprocessed_dataset_dir:
            datasets = load_from_disk(data_args.preprocessed_dataset_dir)
        else:
            with training_args.main_process_first(local=False, desc="de novo data"):
                datasets = build_denovo_datasets(model_args, data_args, processor, tokenizer)
        train, evaluation = datasets["train"], datasets["evaluation"]
        if training_args.process_index == 0:
            print(f"[denovo] train={len(train):,} eval={len(evaluation):,} "
                  f"({data_args.eval_dataset_repo}:{data_args.eval_split})", flush=True)

        trainer = DenovoTrainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=evaluation,
            data_collator=IonaDataCollatorForDenovo(),
            processing_class=tokenizer,
            decode_rows=training_args.decode_rows,
            precursor_tolerance_ppm=training_args.precursor_tolerance_ppm,
        )
        # Trainer.train() does not read args.resume_from_checkpoint on its own.
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        if trainer.is_world_process_zero():
            trainer.save_model(str(out_dir / "final"))
            processor.save_pretrained(str(out_dir / "final"))
            print(f"[denovo] saved to {out_dir / 'final'}", flush=True)
        return 0
    except BaseException:
        # Mark the W&B run as failed rather than finished.
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
            wandb_run = None
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    sys.exit(main())
