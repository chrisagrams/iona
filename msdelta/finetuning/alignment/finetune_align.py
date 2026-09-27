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

from msdelta.finetuning.denoise.finetune_denoise import (MemoryProbe, load_description, select_device,
                                      subset_splits)
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining
from msdelta.models.processing_msdelta import MSDeltaProcessor
from msdelta.rescoring.reranking import (
    POOLING_MODES,
    AlignmentCollator,
    PeptideEncoder,
    REPLICATE_REPO,
    SequenceAlignmentModel,
    attach_teacher_embeddings,
    build_alignment_datasets,
    build_alignment_model,
    cross_modal_metrics,
    group_separation_metrics,
    peptide_key,
)
from msdelta.utils.wandb_distributed import init_wandb_run


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
    # PeptideEncoder readout: pool (the `pooling` op, default), cls or attn. PLAN.md A3.
    sequence_readout: str = "pool"
    # A4. align_loss "mse" (A1 default) or "lit": LiT-style cross-modal SupCon against
    # the frozen teacher, + mse_weight x MSE, + hard_negatives distinguishable
    # rearrangements per peptide (never reversals; see reranking.hard_negatives).
    align_loss: str = "mse"
    align_temperature: float = 0.05
    mse_weight: float = 0.0
    hard_negatives: int = 0
    neg_min_delta: float = 0.05
    # A8 (mass-aware): batches of mass NEIGHBOURS (in-batch negatives ~ same-mass
    # competitors) and/or hard negatives drawn from training peptides within +-neg_ppm.
    mass_batches: bool = False
    mass_batch_jitter: float = 0.5
    neg_source: str = "swap"
    neg_ppm: float = 20.0


@dataclass
class AlignDataArguments:
    processor_name_or_path: str | None = None
    dataset_repo: str = REPLICATE_REPO
    # See ContrastiveDataArguments / grouped_retrieval.load_spectrum_datasets: `grouped`
    # reads ms-contrastive-100k with its own splits. Same defaults as contrastive so a
    # teacher and its student see the same rows.
    dataset_format: str = "replicate"
    include_consensus: bool = False
    exclude_replicate_peptides: bool = True
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
    # Path written by `python -m msdelta.finetuning.alignment.precompute_align`. When set, training loads the
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

    mass_sampler = None          # A8: a reranking.MassBatchSampler, or None (random batches)

    def get_train_dataloader(self):
        if self.mass_sampler is None:
            return super().get_train_dataloader()
        from torch.utils.data import DataLoader
        return DataLoader(self.train_dataset, batch_sampler=self.mass_sampler,
                          collate_fn=self.data_collator,
                          num_workers=self.args.dataloader_num_workers,
                          pin_memory=self.args.dataloader_pin_memory)

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
        spectra, sequences, peptides, charges = [], [], [], []
        try:
            for start in range(0, len(rows), step):
                chunk = rows[start : start + step]
                batch = {k: v.to(device) for k, v in self.data_collator(chunk).items()}
                out = model(**batch, return_dict=True)
                spectra.append(out["target"].cpu())
                sequences.append(out["embeddings"].cpu())
                peptides.extend(f["peptide"] for f in chunk)
                charges.extend(int(f.get("charge", 0)) for f in chunk)
        finally:
            model.train(was_training)
        if not spectra:
            return {}
        first: dict[str, int] = {}
        for index, peptide in enumerate(peptides):
            first.setdefault(peptide, index)
        keep = sorted(first.values())
        slot = {peptides[i]: n for n, i in enumerate(keep)}
        sequence_embeddings, spectrum_embeddings = torch.cat(sequences), torch.cat(spectra)
        metrics = cross_modal_metrics(
            sequence_embeddings[keep], spectrum_embeddings,
            np.array([slot[p] for p in peptides]), np.arange(len(keep)),
        )
        # Geometry, on BOTH towers. hit@1 says whether ranking works; this says why.
        # Replicates of one peptide at one charge should sit closer to each other than to
        # anything else -- if the TEACHER's space does not have that property, no student
        # trained to imitate it can, and the objective is not the thing to fix.
        groups = np.array([peptide_key(p, c) for p, c in zip(peptides, charges)])
        codes = np.unique(groups, return_inverse=True)[1]
        metrics.update(group_separation_metrics(spectrum_embeddings, codes, "sep_spectrum"))
        metrics.update(group_separation_metrics(sequence_embeddings, codes, "sep_sequence"))
        return metrics


def save_peptide_encoder(model, model_args, path) -> None:
    """Also write the student as a standard PeptideEncoderModel (config.json + weights), next to
    the raw final/ weights, so it loads with PeptideEncoderModel.from_pretrained like the
    spectrum encoder does. The raw final/ layout is unchanged for every existing loader."""
    from msdelta.models.peptide_encoder import PeptideEncoderConfig, PeptideEncoderModel
    encoder = getattr(model, "module", model).sequence_encoder
    config = PeptideEncoderConfig(
        embedding_size=encoder.projection[-1].out_features, hidden_size=model_args.sequence_hidden_size,
        num_layers=model_args.sequence_num_layers, num_heads=model_args.sequence_num_heads,
        max_length=model_args.max_peptide_length, n_charges=encoder.charge.num_embeddings,
        mod_n_freqs=encoder.mod_features.freqs.numel(), dropout=model_args.sequence_dropout,
        pooling=model_args.pooling, readout=model_args.sequence_readout,
        spectrum_model=model_args.pretrained_path, spectrum_pooling=model_args.pooling)
    peptide_encoder = PeptideEncoderModel(config)
    peptide_encoder.sequence_encoder.load_state_dict(encoder.state_dict())
    peptide_encoder.save_pretrained(str(path))
    print(f"[align] peptide encoder (standard layout) saved to {path}", flush=True)


# Name before the 2026-09-27 rename ("peptide embedder" -> "peptide encoder").
save_peptide_embedder = save_peptide_encoder

# Subdirectory of an alignment run's final/ holding the standard-layout peptide encoder. Runs
# before 2026-09-27 wrote it as final/peptide_embedder; peptide_encoder_dir() finds either.
PEPTIDE_ENCODER_SUBDIR = "peptide_encoder"
LEGACY_PEPTIDE_ENCODER_SUBDIR = "peptide_embedder"


def peptide_encoder_dir(final_dir) -> Path:
    """final/peptide_encoder of an alignment run, or final/peptide_embedder for a run saved
    before the rename; final/peptide_encoder (the new name) when neither exists yet."""
    final_dir = Path(final_dir)
    for name in (PEPTIDE_ENCODER_SUBDIR, LEGACY_PEPTIDE_ENCODER_SUBDIR):
        if (final_dir / name).is_dir():
            return final_dir / name
    return final_dir / PEPTIDE_ENCODER_SUBDIR


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
                           pooling=model_args.pooling,
                           readout=model_args.sequence_readout),
            pooling=model_args.pooling, loss=model_args.align_loss,
            temperature=model_args.align_temperature, mse_weight=model_args.mse_weight)
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
    collator = AlignmentCollator(max_peptide_length=model_args.max_peptide_length,
                                 hard_negatives=model_args.hard_negatives,
                                 neg_min_delta=model_args.neg_min_delta,
                                 neg_seed=training_args.seed,
                                 neg_source=model_args.neg_source,
                                 neg_ppm=model_args.neg_ppm)

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
            # Targets were written by `python -m msdelta.finetuning.alignment.precompute_align`. Nothing in
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
                from msdelta.data.grouped_retrieval import load_spectrum_datasets
                datasets = load_spectrum_datasets(
                    data_args.dataset_format, data_args.dataset_repo, processor,
                    include_consensus=data_args.include_consensus,
                    exclude_replicate_peptides=data_args.exclude_replicate_peptides,
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

        if model_args.neg_source == "mass" or model_args.mass_batches:
            from msdelta.rescoring.reranking import MassBatchSampler, MassNegativePool, peptide_neutral_mass
            train_peps = datasets["train"]["peptide"]
            if model_args.neg_source == "mass":
                collator.neg_pool = MassNegativePool(train_peps)
                print(f"[align] mass negatives: {len(collator.neg_pool.peptides):,} training "
                      f"peptides, +-{model_args.neg_ppm:g} ppm", flush=True)
        trainer = SequenceAlignmentTrainer(
            model=model, args=training_args, train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"), data_collator=collator,
        )
        if model_args.mass_batches:
            masses = [peptide_neutral_mass(p) for p in train_peps]
            trainer.mass_sampler = MassBatchSampler(
                masses, training_args.per_device_train_batch_size,
                jitter=model_args.mass_batch_jitter, seed=training_args.seed)
            print(f"[align] mass-bucketed batches of {training_args.per_device_train_batch_size} "
                  f"(jitter +-{model_args.mass_batch_jitter:g} Da)", flush=True)
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
            save_peptide_encoder(trainer.model, model_args, out_dir / "final" / PEPTIDE_ENCODER_SUBDIR)
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
