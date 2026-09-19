"""Train a peptide encoder into the frozen spectrum encoder's embedding space.

The first step of reranking: once spectra and sequences share a space, scoring a
database-search candidate is a dot product.

Only the peptide encoder learns. The spectrum encoder is frozen and in eval mode, so it
emits a fixed target per spectrum -- see msdelta/reranking.py for why that matters.

**Watch `crossmodal/hit@1`, not the loss.** L2 falls whenever predictions move toward the
mean target, and collapsing every sequence onto the centroid does exactly that while
destroying every ordering. The ranking metric is evaluated separately and is the number
that says whether the alignment is useful.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

from msdelta.finetune_denoise import (MemoryProbe, load_description, select_device,
                                      subset_splits)
from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.reranking import (
    POOLING_MODES,
    AlignmentCollator,
    PeptideEncoder,
    REPLICATE_REPO,
    SequenceAlignmentModel,
    attach_teacher_embeddings,
    build_alignment_datasets,
    build_alignment_model,
    cross_modal_metrics,
)
from msdelta.wandb_distributed import init_wandb_run


@dataclass
class AlignModelArguments:
    pretrained_path: str = field(
        metadata={"help": "Pretrained MSDelta checkpoint; its encoder becomes the frozen teacher."}
    )
    pooling: str = field(
        default="mean+max",
        metadata={"help": f"How token embeddings are reduced. One of {POOLING_MODES}. "
                          "Both towers must agree; the teacher's choice defines the space."},
    )
    sequence_hidden_size: int = 256
    sequence_num_layers: int = 4
    sequence_num_heads: int = 8
    sequence_dropout: float = 0.1
    max_peptide_length: int = 64


@dataclass
class AlignDataArguments:
    processor_name_or_path: str | None = None
    dataset_repo: str = REPLICATE_REPO
    preprocessing_num_workers: int = 24
    max_peaks: int = 512
    validation_fraction: float = 0.1
    # See subset_splits in finetune_denoise: caps ROWS, not steps, so a smoke test still
    # runs real epochs and therefore still exercises saving, load_best_model_at_end, the
    # cross-modal evaluation and the final save.
    max_samples: int = 0
    # Precompute the frozen teacher's embeddings and drop it from the training graph.
    # Default ON: it is both faster and the only configuration that has any prospect of
    # running on twelve tiles (FT9). --precompute_targets false keeps the old behaviour
    # for comparison.
    precompute_targets: bool = True
    # Path written by `python -m msdelta.precompute_align`. When set, training loads the
    # targets from disk and NEVER constructs a teacher -- no 49.8M model is loaded onto
    # the device and abandoned, which is the last difference between an alignment job
    # that faults on twelve tiles and a bisect that does not.
    target_cache: str | None = None


@dataclass
class AlignTrainingArguments(TrainingArguments):
    wandb_project: str | None = None
    wandb_entity: str | None = None
    run_description: str | None = None
    eval_alignment_rows: int = field(
        default=2000,
        metadata={"help": "Spectra scored by the cross-modal evaluation. Candidates are "
                          "deduplicated by peptide first, so this counts spectra."},
    )


class SequenceAlignmentTrainer(Trainer):
    """Optimise the student alone, and score ranking rather than the loss."""

    def create_optimizer(self):
        """Defer to the Trainer unless a teacher is actually present to exclude.

        This used to always build a flat parameter list over the student. That was
        needed when the frozen teacher was in the module -- optimiser state for tens of
        millions of weights that carry no gradient is memory bought for nothing -- but
        it had a cost that was not noticed: a flat list loses the Trainer's weight-decay
        grouping, so `weight_decay 0.01` was being applied to biases and LayerNorm gains
        as well, which the default deliberately exempts.

        With precomputed targets `spectrum_model` is None and the student IS the whole
        model, so there is nothing to exclude and the stock path is both correct and the
        one the denoise fine-tune uses successfully on twelve tiles.
        """
        if self.optimizer is not None:
            return self.optimizer
        if getattr(self.model, "spectrum_model", None) is None:
            return super().create_optimizer()
        cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        decay = [p for n, p in self.model.sequence_encoder.named_parameters()
                 if p.requires_grad and p.ndim > 1]
        no_decay = [p for n, p in self.model.sequence_encoder.named_parameters()
                    if p.requires_grad and p.ndim <= 1]
        kwargs.pop("weight_decay", None)
        self.optimizer = cls([{"params": decay, "weight_decay": self.args.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], **kwargs)
        return self.optimizer

    @torch.no_grad()
    def evaluate_alignment(self, dataset, max_rows: int = 2000) -> dict[str, float]:
        """Rank candidate sequences against each spectrum.

        Duplicate peptides collapse to one candidate, so Hit@1 answers "is the right
        SEQUENCE first" rather than "is one of this peptide's replicates first".
        """
        model = self.model
        was_training = model.training
        model.eval()
        device = next(model.sequence_encoder.parameters()).device
        rows = list(dataset.select(range(min(len(dataset), max_rows))))
        step = max(1, self.args.per_device_eval_batch_size)
        spectra, sequences, peptides = [], [], []
        try:
            for start in range(0, len(rows), step):
                chunk = rows[start : start + step]
                batch = {k: v.to(device) for k, v in self.data_collator(chunk).items()}
                out = model(**batch, return_dict=True)
                spectra.append(out["target"].cpu())
                sequences.append(out["embeddings"].cpu())
                peptides.extend(f["peptide"] for f in chunk)
        finally:
            model.train(was_training)
        if not spectra:
            return {}
        first: dict[str, int] = {}
        for index, peptide in enumerate(peptides):
            first.setdefault(peptide, index)
        keep = sorted(first.values())
        slot = {peptides[i]: n for n, i in enumerate(keep)}
        return cross_modal_metrics(
            torch.cat(sequences)[keep], torch.cat(spectra),
            np.array([slot[p] for p in peptides]), np.arange(len(keep)),
        )


def main(argv: list[str] | None = None) -> int:
    select_device()
    parser = HfArgumentParser(
        (AlignModelArguments, AlignDataArguments, AlignTrainingArguments)  # pyright: ignore[reportArgumentType]
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=argv, args_file_flag="--args_file"
    )
    out_dir = Path(training_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if training_args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", training_args.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(out_dir))
    set_seed(training_args.seed)

    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks,
    )
    cached = Path(data_args.target_cache) if data_args.target_cache else None
    if cached and cached.exists():
        # The teacher is never constructed. Its output width comes from the manifest the
        # precompute wrote, so nothing here needs the checkpoint -- not to size the
        # student, not to derive the split, not at all.
        manifest = dict(
            line.split(": ", 1)
            for line in (cached / "MANIFEST.txt").read_text().splitlines() if ": " in line)
        if manifest["pooling"] != model_args.pooling:
            sys.exit(f"cache was built with pooling={manifest['pooling']!r}, "
                     f"this run asks for {model_args.pooling!r}")
        model = SequenceAlignmentModel(
            None,
            PeptideEncoder(embedding_size=int(manifest["embedding_size"]),
                           hidden_size=model_args.sequence_hidden_size,
                           num_layers=model_args.sequence_num_layers,
                           num_heads=model_args.sequence_num_heads,
                           max_length=model_args.max_peptide_length,
                           dropout=model_args.sequence_dropout,
                           pooling=model_args.pooling),
            pooling=model_args.pooling)
    else:
        teacher = MSDeltaForPreTraining.from_pretrained(model_args.pretrained_path)
        model = build_alignment_model(
            teacher, pooling=model_args.pooling,
            hidden_size=model_args.sequence_hidden_size,
            num_layers=model_args.sequence_num_layers,
            num_heads=model_args.sequence_num_heads,
            dropout=model_args.sequence_dropout,
            max_peptide_length=model_args.max_peptide_length,
        )
    collator = AlignmentCollator(max_peptide_length=model_args.max_peptide_length)

    student = sum(p.numel() for p in model.sequence_encoder.parameters())
    frozen = (sum(p.numel() for p in model.spectrum_model.parameters())
              if model.spectrum_model is not None else 0)
    training_args.run_description = (training_args.run_description
                                     or load_description())
    description = (
        f"Peptide encoder aligned to a frozen {Path(model_args.pretrained_path).parent.name} "
        f"spectrum encoder under L2 on unit vectors. pooling={model_args.pooling}, "
        f"student {student/1e6:.2f}M, teacher {frozen/1e6:.2f}M frozen, "
        f"lr={training_args.learning_rate:g}, seed {training_args.seed}."
    )
    if training_args.run_description:
        description = f"{training_args.run_description} -- {description}"
    tags = ["reranking", "alignment", f"pool{model_args.pooling}",
            f"lr{training_args.learning_rate:g}", f"seed{training_args.seed}"]

    wandb_run = None
    if training_args.wandb_project:
        wandb_run = init_wandb_run(
            project=training_args.wandb_project, run_name=training_args.run_name,
            entity=training_args.wandb_entity, notes=description, tags=tags,
            config={"model": asdict(model_args), "data": asdict(data_args),
                    "training": training_args.to_dict()},
        )
    try:
        if training_args.process_index == 0:
            (out_dir / "RUN.md").write_text(f"# {training_args.run_name}\n\n{description}\n")
            print(f"[align] {description}", flush=True)
            print(f"[align] embedding size {model.sequence_encoder.projection[-1].out_features}",
                  flush=True)

        if cached and cached.exists():
            # Targets were written by `python -m msdelta.precompute_align`. Nothing in
            # this process loads or touches a teacher: the split came from the cache,
            # the student's width came from the manifest, and there is no 49.8M model to
            # put on the device and abandon.
            from datasets import load_from_disk
            datasets = {name: load_from_disk(str(cached / name))
                        for name in ("train", "validation") if (cached / name).exists()}
            if training_args.process_index == 0:
                print(f"[align] targets from {cached} "
                      + " ".join(f"{k}={len(v):,}" for k, v in datasets.items()),
                      flush=True)
        else:
            with training_args.main_process_first(local=False, desc="alignment data"):
                datasets = build_alignment_datasets(
                    data_args.dataset_repo, processor,
                    num_proc=data_args.preprocessing_num_workers or None,
                    validation_fraction=data_args.validation_fraction,
                    seed=training_args.seed,
                )
            datasets = subset_splits(datasets, data_args.max_samples,
                                     training_args.process_index)
            if data_args.precompute_targets:
                # In-process fallback, kept for the single-tile path. Under
                # main_process_first: datasets.map writes a cache file and twelve ranks
                # writing it at once is a race, as well as twelve times the work.
                with training_args.main_process_first(local=False,
                                                      desc="teacher embeddings"):
                    datasets = attach_teacher_embeddings(
                        datasets, model.spectrum_model, model_args.pooling,
                        batch_size=training_args.per_device_eval_batch_size,
                        max_peptide_length=model_args.max_peptide_length)
                model.spectrum_model = None
        if training_args.process_index == 0:
            print("[align] " + " ".join(f"{k}={len(v):,}" for k, v in datasets.items()),
                  flush=True)

        trainer = SequenceAlignmentTrainer(
            model=model, args=training_args, train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"), data_collator=collator,
        )
        trainer.add_callback(MemoryProbe(every=50))
        trainer.train()

        if trainer.is_world_process_zero() and datasets.get("validation") is not None:
            metrics = trainer.evaluate_alignment(
                datasets["validation"], max_rows=training_args.eval_alignment_rows
            )
            print(f"[align] cross-modal: {metrics}", flush=True)
            trainer.log(metrics)
            trainer.save_metrics("crossmodal", metrics)
        if trainer.is_world_process_zero():
            trainer.save_model(str(out_dir / "final"))
            print(f"[align] saved to {out_dir / 'final'}", flush=True)
        return 0
    except BaseException:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
            wandb_run = None
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    sys.exit(main())
