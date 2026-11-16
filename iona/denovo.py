"""Score de novo predictions and train the peptide decoder with the Hugging Face Trainer."""

from __future__ import annotations

import torch

from iona.train import IonaTrainer

# Casanovo's thresholds: prefix masses must agree within 0.5 Da, residues within 0.1 Da.
CUMULATIVE_MASS_TOLERANCE = 0.5
RESIDUE_MASS_TOLERANCE = 0.1


def match_residues(
    predicted: list[float],
    true: list[float],
    cumulative_tolerance: float = CUMULATIVE_MASS_TOLERANCE,
    residue_tolerance: float = RESIDUE_MASS_TOLERANCE,
) -> list[bool]:
    """Mark matched residues by mass, from the N-terminus and then from the C-terminus.

    Two residues match when the masses before them agree and their own masses agree, so
    isobaric residues such as L and I match. This follows Casanovo's prefix-suffix matching.
    """
    matched = [False] * max(len(predicted), len(true))
    i = j = 0
    predicted_mass = true_mass = 0.0
    while i < len(predicted) and j < len(true):
        next_predicted, next_true = predicted_mass + predicted[i], true_mass + true[j]
        if abs(next_predicted - next_true) < cumulative_tolerance:
            matched[max(i, j)] = abs(predicted[i] - true[j]) < residue_tolerance
            predicted_mass, true_mass = next_predicted, next_true
            i, j = i + 1, j + 1
        elif next_true > next_predicted:
            predicted_mass, i = next_predicted, i + 1
        else:
            true_mass, j = next_true, j + 1
    if all(matched):
        return matched

    stop = matched.index(False)
    i, j = len(predicted) - 1, len(true) - 1
    predicted_mass = true_mass = 0.0
    while i >= stop and j >= stop:
        next_predicted, next_true = predicted_mass + predicted[i], true_mass + true[j]
        if abs(next_predicted - next_true) < cumulative_tolerance:
            matched[max(i, j)] = abs(predicted[i] - true[j]) < residue_tolerance
            predicted_mass, true_mass = next_predicted, next_true
            i, j = i - 1, j - 1
        elif next_true > next_predicted:
            predicted_mass, i = next_predicted, i - 1
        else:
            true_mass, j = next_true, j - 1
    return matched


def residue_masses(ids, token_masses: list[float], eos_token_id: int):
    """Masses of the residue tokens before EOS."""
    masses = []
    for index in ids:
        if index == eos_token_id:
            break
        masses.append(token_masses[index])
    return masses


class DenovoMetrics:
    """Count matched peptides and residues over decoded spectra."""

    # spectra, matched peptides, matched residues, predicted residues, true residues,
    # predictions that fit the precursor
    N_COUNTS = 6

    def __init__(self):
        self.counts = [0] * self.N_COUNTS

    def update(self, predicted: list[float], true: list[float], fits: bool) -> list[bool]:
        """Score one prediction, given residue masses, and return its per-residue matches."""
        matched = match_residues(predicted, true)
        for index, value in enumerate(
            (1, int(all(matched)), sum(matched), len(predicted), len(true), int(fits))
        ):
            self.counts[index] += value
        return matched

    @staticmethod
    def compute_from(counts) -> dict[str, float]:
        spectra, peptides, residues, predicted, true, fits = (int(c) for c in counts)
        return {
            "aa_precision": residues / max(predicted, 1),
            "aa_recall": residues / max(true, 1),
            "peptide_recall": peptides / max(spectra, 1),
            "precursor_match_rate": fits / max(spectra, 1),
            "decoded_spectra": float(spectra),
        }

    def compute(self) -> dict[str, float]:
        return self.compute_from(self.counts)


def area_under_precision_coverage(scores: list[float], correct: list[bool]) -> float:
    """Area under the precision-coverage curve, ranking predictions by score as Casanovo does.

    Each prediction adds 1/n coverage at the precision reached once it is included.
    """
    if not scores:
        raise ValueError("no predictions to rank")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    matched = 0
    area = 0.0
    for rank, i in enumerate(order, start=1):
        matched += int(correct[i])
        area += matched / rank
    return area / len(scores)


class DenovoTrainer(IonaTrainer):
    """Score decoded peptides on evaluation."""

    def __init__(
        self,
        *args,
        decode_rows: int = 2000,
        precursor_tolerance_ppm: float = 50.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.decode_rows = decode_rows
        self.precursor_tolerance_ppm = precursor_tolerance_ppm

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """Teacher-forced loss from Trainer, plus metrics on decoded peptides."""
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if self.decode_rows > 0 and dataset is not None:
            decoded = {
                f"{metric_key_prefix}_{name}": value
                for name, value in self.decoding_metrics(dataset).items()
            }
            self.log(decoded)
            metrics.update(decoded)
        return metrics

    @torch.no_grad()
    def decoding_metrics(self, dataset) -> dict[str, float]:
        """Decode the first decode_rows spectra, split across processes, and score them."""
        model = self.accelerator.unwrap_model(self.model)
        tokenizer = self.processing_class
        token_masses = tokenizer.token_masses
        rows = range(min(len(dataset), self.decode_rows))
        shard = rows[self.args.process_index :: self.args.world_size]
        metrics = DenovoMetrics()
        step = self.args.per_device_eval_batch_size
        for start in range(0, len(shard), step):
            batch = self.data_collator([dataset[index] for index in shard[start : start + step]])
            labels = batch.pop("labels")
            batch = {key: value.to(self.args.device) for key, value in batch.items()}
            with self.autocast_smart_context_manager():
                output = model.generate(
                    **batch,
                    tokenizer=tokenizer,
                    precursor_tolerance_ppm=self.precursor_tolerance_ppm,
                )
            for sequence, label, fits in zip(
                output.sequences.tolist(), labels.tolist(), output.precursor_match.tolist()
            ):
                metrics.update(
                    residue_masses(sequence, token_masses, tokenizer.eos_token_id),
                    residue_masses(label, token_masses, tokenizer.eos_token_id),
                    fits,
                )
        counts = torch.tensor(metrics.counts, dtype=torch.long, device=self.args.device)
        return DenovoMetrics.compute_from(
            self.accelerator.reduce(counts, reduction="sum").tolist()
        )
