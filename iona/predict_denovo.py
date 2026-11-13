"""Sequence the spectra of MGF files de novo with a trained ``IonaForDenovo`` model.

Writes one CSV row per spectrum, which can be ranked by ``peptide_score`` into a
precision-coverage curve as Casanovo does.

Example:
    iona-denovo ninespecies-yeast.mgf --checkpoint runs/denovo-50m/final \\
        --output predictions.csv --evaluate --summary metrics.json

Multi-GPU:
    accelerate launch --num_processes 4 --module iona.predict_denovo run.mgf \\
        --checkpoint runs/denovo-50m/final --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from itertools import batched
from pathlib import Path

import pandas as pd
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from tqdm.auto import tqdm

from iona.chemistry import PROTON_MASS, WATER_MASS
from iona.data import normalize_annotation, top_peaks
from iona.denovo import DenovoMetrics, area_under_precision_coverage, residue_masses
from iona.modeling_iona import IonaForDenovo
from iona.processing_iona import IonaDataCollatorForDenovo, IonaProcessor
from iona.spectrum_files import InputSpectrum, count_spectra, read_spectra
from iona.tokenization_iona import IonaPeptideTokenizer

logger = logging.getLogger(__name__)

COLUMNS = [
    "file",
    "index",
    "title",
    "scan",
    "retention_time",
    "precursor_mz",
    "charge",
    "calc_mass_to_charge",
    "peptide",
    "peptide_score",
    "aa_scores",
    "precursor_match",
]
EVALUATION_COLUMNS = ["true_peptide", "correct", "aa_matches"]
EPS = sys.float_info.epsilon


def peptide_score(residue_scores: list[float], eos_score: float) -> float:
    """Probability of the whole peptide, as the latest Casanovo scores it.

    The product of every residue's probability and the EOS probability, each clipped to
    machine epsilon. The precursor does not affect it; see `precursor_match` for that.
    """
    log_score = sum(math.log(min(max(p, EPS), 1.0)) for p in [*residue_scores, eos_score])
    return math.exp(log_score)


def describe(source: Path, spectrum: InputSpectrum) -> str:
    return f"{source.name} spectrum {spectrum.index} ({spectrum.title!r})"


class DenovoPredictor:
    """Greedy-decode peptides for spectra with ``IonaForDenovo``."""

    def __init__(
        self,
        model: IonaForDenovo,
        processor: IonaProcessor,
        tokenizer: IonaPeptideTokenizer,
        *,
        batch_size: int = 256,
        precursor_tolerance_ppm: float = 50.0,
        compile_model: bool = True,
    ):
        self.model = model.eval()
        if compile_model:
            # generate() is a Python loop over a growing KV cache, so compile only the
            # encoder, which does most of the work.
            self.model.iona.compile(dynamic=True)
        self.device = next(model.parameters()).device
        self.processor = processor
        self.tokenizer = tokenizer
        self.token_masses = tokenizer.token_masses
        self.batch_size = batch_size
        self.precursor_tolerance_ppm = precursor_tolerance_ppm
        self.collator = IonaDataCollatorForDenovo(
            max_peptide_length=model.config.max_peptide_length
        )

    def _features(self, source: Path, spectrum: InputSpectrum) -> dict:
        """Model inputs for one spectrum; raises ValueError if the model cannot take it."""
        max_charge = self.model.config.max_charge
        if spectrum.charge > max_charge:
            raise ValueError(f"{describe(source, spectrum)}: charge {spectrum.charge} exceeds "
                             f"the model's max_charge={max_charge}")
        # Keep the max_peaks most intense peaks, as Casanovo does. top_peaks returns tensors,
        # which the processor needs: it reads a numpy array as a batch of scalars.
        mz, intensity = top_peaks(spectrum.mz, spectrum.intensity, self.processor.max_peaks)
        try:
            values = self.processor(mz, intensity, padding=False)
        except ValueError as error:
            raise ValueError(f"{describe(source, spectrum)}: {error}") from None
        return {
            "mz": values["mz"],
            "log_intensity": values["log_intensity"],
            "precursor_mz": spectrum.precursor_mz,
            "precursor_charge": spectrum.charge,
        }

    @torch.inference_mode()
    def predict(self, source: Path, spectra: list[InputSpectrum]) -> list[dict]:
        """One result per spectrum, in input order.

        Each result holds the residue `tokens`, their `scores`, the probability of EOS after
        them (`eos_score`, 0 when decoding hit max_peptide_length and EOS was forced), and
        `precursor_match`.
        """
        features = [self._features(source, spectrum) for spectrum in spectra]
        results: list[dict] = [{} for _ in spectra]
        # Batch spectra of similar peak counts together to limit padding.
        order = sorted(range(len(features)), key=lambda row: len(features[row]["mz"]))
        eos = self.tokenizer.eos_token_id
        max_length = self.model.config.max_peptide_length
        for rows in batched(order, self.batch_size):
            batch = self.collator([features[row] for row in rows])
            batch = {key: value.to(self.device) for key, value in batch.items()}
            output = self.model.generate(
                **batch,
                tokenizer=self.tokenizer,
                precursor_tolerance_ppm=self.precursor_tolerance_ppm,
            )
            for row, sequence, scores, fits in zip(
                rows,
                output.sequences.tolist(),
                output.token_scores.float().tolist(),
                output.precursor_match.tolist(),
            ):
                length = sequence.index(eos)
                results[row] = {
                    "tokens": sequence[:length],
                    "scores": scores[:length],
                    # generate() forces EOS at max_peptide_length; Casanovo scores a stop token
                    # it did not predict as 0.
                    "eos_score": scores[length] if length < max_length else 0.0,
                    "precursor_match": bool(fits),
                }
        return results

    def rows(
        self, source: Path, spectra: list[InputSpectrum], metrics: DenovoMetrics | None
    ) -> list[dict]:
        """Decode `spectra` into output rows, scoring them into `metrics` if given."""
        rows = []
        for spectrum, result in zip(spectra, self.predict(source, spectra)):
            tokens, scores = result["tokens"], result["scores"]
            predicted = [self.token_masses[token] for token in tokens]
            charge = spectrum.charge
            row = {
                "file": source.name,
                "index": spectrum.index,
                "title": spectrum.title,
                "scan": spectrum.scan,
                "retention_time": spectrum.retention_time,
                "precursor_mz": spectrum.precursor_mz,
                "charge": charge,
                "calc_mass_to_charge": (sum(predicted) + WATER_MASS + charge * PROTON_MASS)
                / charge,
                "peptide": "".join(self.tokenizer.convert_ids_to_tokens(tokens)),
                "peptide_score": peptide_score(scores, result["eos_score"]),
                "aa_scores": ",".join(f"{score:.5f}" for score in scores),
                "precursor_match": result["precursor_match"],
            }
            if metrics is not None:
                row.update(self._evaluate(source, spectrum, predicted, row, metrics))
            rows.append(row)
        return rows

    def _evaluate(
        self,
        source: Path,
        spectrum: InputSpectrum,
        predicted: list[float],
        row: dict,
        metrics: DenovoMetrics,
    ) -> dict:
        """Score a prediction against the spectrum's annotation."""
        if not spectrum.annotation:
            raise ValueError(f"{describe(source, spectrum)} has no SEQ annotation; --evaluate "
                             "needs one on every spectrum")
        try:
            true_peptide = normalize_annotation(spectrum.annotation)
        except ValueError as error:
            raise ValueError(f"{describe(source, spectrum)}: {error}") from None
        ids = self.tokenizer(true_peptide)["input_ids"]
        true = residue_masses(ids, self.token_masses, self.tokenizer.eos_token_id)
        matched = metrics.update(predicted, true, row["precursor_match"])
        return {
            "true_peptide": true_peptide,
            "correct": all(matched),
            # One flag per predicted residue, aligned with aa_scores.
            "aa_matches": ",".join(str(int(flag)) for flag in matched[: len(predicted)]),
        }


def resolve_device(device: str | None, state: PartialState) -> torch.device:
    """Pick this process's device, as ``load_denoiser`` does."""
    if state.num_processes > 1:
        if device is not None:
            raise ValueError("--device cannot be used with distributed inference")
        resolved = torch.device(state.device)
        if resolved.type == "cuda":
            resolved = torch.device("cuda", state.local_process_index % torch.cuda.device_count())
            torch.cuda.set_device(resolved)
            state.device = resolved
        return resolved
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def predict_file(
    source: Path,
    predictor: DenovoPredictor,
    state: PartialState,
    *,
    evaluate: bool,
    chunk_size: int,
    show_progress: bool,
) -> tuple[list[dict], dict]:
    """Sequence one file and return its rows, in file order, and its summary."""
    metrics = DenovoMetrics() if evaluate else None
    rows: list[dict] = []
    total = count_spectra(source) if show_progress else None
    with tqdm(desc=source.name, total=total, unit="spectra", disable=not show_progress) as bar:
        for chunk in batched(read_spectra(source), chunk_size):
            local = list(chunk[state.process_index :: state.num_processes])
            chunk_rows = predictor.rows(source, local, metrics)
            if state.num_processes > 1:
                chunk_rows = sorted(gather_object(chunk_rows), key=lambda row: row["index"])
            rows.extend(chunk_rows)
            bar.update(len(chunk))

    summary: dict = {"spectra": len(rows)}
    if metrics is not None:
        if state.num_processes > 1:
            metrics.counts = [sum(values) for values in zip(*gather_object([metrics.counts]))]
        summary.update(metrics.compute())
        # Every spectrum is decoded, so this repeats "spectra".
        summary.pop("decoded_spectra")
        summary["aupc"] = area_under_precision_coverage(
            [row["peptide_score"] for row in rows], [row["correct"] for row in rows]
        )
    return rows, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="MGF file(s) to sequence")
    parser.add_argument(
        "--checkpoint", required=True, help="IonaForDenovo checkpoint path or Hub ID"
    )
    parser.add_argument("--processor", help="processor path; defaults to the checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="CSV of predictions")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="score predictions against each spectrum's SEQ annotation",
    )
    parser.add_argument(
        "--summary", type=Path, help="write per-file counts and metrics to this JSON file"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="spectra per model batch")
    parser.add_argument(
        "--precursor-tolerance-ppm",
        type=float,
        default=50.0,
        help="tolerance for a peptide to fit its precursor (default: 50)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="spectra read into memory at a time (default: 8192)",
    )
    parser.add_argument("--device", help="torch device (default: cuda, then xpu, then cpu)")
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    state = PartialState()
    device = resolve_device(args.device, state)
    predictor = DenovoPredictor(
        IonaForDenovo.from_pretrained(args.checkpoint, dtype="auto").to(device),
        IonaProcessor.from_pretrained(args.processor or args.checkpoint),
        IonaPeptideTokenizer(),
        batch_size=args.batch_size,
        precursor_tolerance_ppm=args.precursor_tolerance_ppm,
        compile_model=not args.no_compile,
    )

    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for source in args.inputs:
        started = time.perf_counter()
        file_rows, summary = predict_file(
            source,
            predictor,
            state,
            evaluate=args.evaluate,
            chunk_size=args.chunk_size,
            show_progress=not args.no_progress and state.is_main_process,
        )
        rows.extend(file_rows)
        summaries[str(source)] = summary
        if state.is_main_process:
            logger.info("%s: %s in %.1fs", source.name, json.dumps(summary),
                        time.perf_counter() - started)

    if state.is_main_process:
        columns = COLUMNS + (EVALUATION_COLUMNS if args.evaluate else [])
        pd.DataFrame(rows, columns=columns).to_csv(args.output, index=False)
        if args.summary is not None:
            args.summary.write_text(json.dumps(summaries, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
