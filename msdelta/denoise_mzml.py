"""Denoise the spectra of mzML files with a trained ``MSDeltaForDenoising`` model.

Example:
    msdelta-denoise run.mzML --checkpoint runs/denoise-probes/step-190000 \\
        --output run.denoised.mzML
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from msdelta.denoising import PeakBudgetBatchSampler
from msdelta.modeling_msdelta import MSDeltaForDenoising
from msdelta.mzml import MzMLRewriter, Spectrum
from msdelta.processing_msdelta import MSDeltaProcessor

logger = logging.getLogger(__name__)

@dataclass
class DenoisingSummary:
    """Peak and spectrum counts accumulated over one mzML file."""

    spectra: int = 0
    denoised_spectra: int = 0
    peaks_in: int = 0
    peaks_out: int = 0
    noise_peaks: int = 0
    unscored_peaks: int = 0
    emptied_spectra: int = 0


def select_model_peaks(intensity: np.ndarray, max_peaks: int) -> np.ndarray:
    """Return indices of the most intense peaks, in their original order."""
    if intensity.size <= max_peaks:
        return np.arange(intensity.size)
    top = np.argpartition(intensity, intensity.size - max_peaks)[-max_peaks:]
    return np.sort(top)


class SpectrumDenoiser:
    """Score peaks as noise with ``MSDeltaForDenoising`` and build keep masks."""

    def __init__(
        self,
        model: MSDeltaForDenoising,
        processor: MSDeltaProcessor,
        *,
        noise_threshold: float = 0.5,
        peak_pair_budget: int = 4_194_304,
        keep_unscored: bool = False,
    ):
        if not 0.0 < noise_threshold <= 1.0:
            raise ValueError("noise_threshold must be in (0, 1]")
        if peak_pair_budget < processor.max_peaks**2:
            raise ValueError("peak_pair_budget must fit one spectrum of max_peaks peaks")
        self.model = model.eval()
        self.processor = processor
        self.noise_threshold = noise_threshold
        self.peak_pair_budget = peak_pair_budget
        self.keep_unscored = keep_unscored
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def noise_probabilities(self, spectra: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """Return one noise probability per peak for each ``(mz, intensity)`` pair."""
        lengths = [mz.size for mz, _ in spectra]
        results: list[np.ndarray | None] = [None] * len(spectra)
        sampler = PeakBudgetBatchSampler(lengths, self.peak_pair_budget, seed=0)
        for batch in sampler:
            inputs = self.processor(
                [spectra[i][0] for i in batch],
                [spectra[i][1] for i in batch],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs, return_dict=True).logits
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            for row, i in enumerate(batch):
                results[i] = probabilities[row, : lengths[i]]
        return results

    def keep_masks(self, spectra: Iterable[Spectrum]) -> list[np.ndarray]:
        """Return a boolean keep mask for every spectrum's peaks."""
        spectra = list(spectra)
        masks = [np.ones(spectrum.peak_count, dtype=bool) for spectrum in spectra]
        scored: list[int] = []
        selected: list[np.ndarray] = []
        for i, spectrum in enumerate(spectra):
            if spectrum.peak_count == 0 or spectrum.intensity.max() <= 0:
                continue
            indices = select_model_peaks(spectrum.intensity, self.processor.max_peaks)
            if not self.keep_unscored:
                masks[i][:] = False
                masks[i][indices] = True
            scored.append(i)
            selected.append(indices)
        probabilities = self.noise_probabilities(
            [(spectra[i].mz[idx], spectra[i].intensity[idx]) for i, idx in zip(scored, selected)]
        )
        for i, indices, noise in zip(scored, selected, probabilities):
            masks[i][indices] = noise < self.noise_threshold
        return masks


def _chunks(iterator: Iterator[Spectrum], size: int) -> Iterator[list[Spectrum]]:
    while chunk := list(islice(iterator, size)):
        yield chunk


def denoise_mzml(
    source: Path,
    destination: Path,
    denoiser: SpectrumDenoiser,
    *,
    ms_levels: set[int] | None = None,
    chunk_size: int = 1024,
) -> DenoisingSummary:
    """Write ``destination`` as a copy of ``source`` with noise peaks removed."""
    summary = DenoisingSummary()
    with MzMLRewriter(source, destination) as rewriter:
        for chunk in _chunks(rewriter.spectra(), chunk_size):
            targets = [
                spectrum
                for spectrum in chunk
                if ms_levels is None or spectrum.ms_level in ms_levels
            ]
            masks = dict(zip((s.index for s in targets), denoiser.keep_masks(targets)))
            for spectrum in chunk:
                summary.spectra += 1
                keep = masks.get(spectrum.index)
                if keep is None:
                    rewriter.write(spectrum)
                    continue
                n_in, n_out = spectrum.peak_count, int(keep.sum())
                summary.denoised_spectra += 1
                summary.peaks_in += n_in
                summary.peaks_out += n_out
                n_unscored = 0
                if not denoiser.keep_unscored:
                    n_unscored = min(max(n_in - denoiser.processor.max_peaks, 0), n_in - n_out)
                summary.unscored_peaks += n_unscored
                summary.noise_peaks += n_in - n_out - n_unscored
                summary.emptied_spectra += int(n_in > 0 and n_out == 0)
                rewriter.write(spectrum, keep)
    return summary


def load_denoiser(
    checkpoint: str,
    *,
    processor_path: str | None = None,
    device: str | None = None,
    **kwargs,
) -> SpectrumDenoiser:
    """Load a denoising checkpoint and its processor onto ``device``."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MSDeltaForDenoising.from_pretrained(checkpoint, dtype="auto").to(device)
    processor = MSDeltaProcessor.from_pretrained(processor_path or checkpoint)
    return SpectrumDenoiser(model, processor, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="mzML file(s) to denoise")
    parser.add_argument(
        "--checkpoint", required=True, help="MSDeltaForDenoising checkpoint path or Hub ID"
    )
    parser.add_argument("--processor", help="processor path; defaults to the checkpoint")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path, help="output mzML path (single input only)")
    output.add_argument(
        "--output-dir", type=Path, help="directory that receives one mzML per input"
    )
    parser.add_argument(
        "--ms-level",
        type=int,
        nargs="+",
        default=[2],
        help="MS levels to denoise; other spectra are copied unchanged (default: 2)",
    )
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=0.5,
        help="remove peaks whose noise probability is at least this value (default: 0.5)",
    )
    parser.add_argument(
        "--keep-unscored",
        action="store_true",
        help="keep peaks beyond the processor's max_peaks most intense peaks instead of "
        "dropping them; the model never scores those peaks",
    )
    parser.add_argument(
        "--peak-pair-budget",
        type=int,
        default=4_194_304,
        help="maximum padded peak pairs per model batch (default: 4194304)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="spectra read into memory between model batches (default: 1024)",
    )
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    parser.add_argument(
        "--summary", type=Path, help="write per-file peak and spectrum counts to this JSON file"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace outputs that already exist"
    )
    args = parser.parse_args(argv)
    if args.output is not None and len(args.inputs) != 1:
        parser.error("--output accepts exactly one input; use --output-dir for several")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    denoiser = load_denoiser(
        args.checkpoint,
        processor_path=args.processor,
        device=args.device,
        noise_threshold=args.noise_threshold,
        peak_pair_budget=args.peak_pair_budget,
        keep_unscored=args.keep_unscored,
    )
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for source in args.inputs:
        destination = args.output if args.output is not None else args.output_dir / source.name
        if destination.resolve() == source.resolve():
            raise SystemExit(f"refusing to overwrite the input file {source}")
        if destination.exists() and not args.overwrite:
            logger.info("skipping %s: %s exists", source, destination)
            continue
        started = time.perf_counter()
        partial = destination.with_name(destination.name + ".part")
        try:
            summary = denoise_mzml(
                source,
                partial,
                denoiser,
                ms_levels=set(args.ms_level),
                chunk_size=args.chunk_size,
            )
            partial.replace(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        summaries[str(source)] = asdict(summary)
        logger.info(
            "%s: %d/%d spectra denoised, %d -> %d peaks (%d noise, %d unscored) in %.1fs",
            source.name,
            summary.denoised_spectra,
            summary.spectra,
            summary.peaks_in,
            summary.peaks_out,
            summary.noise_peaks,
            summary.unscored_peaks,
            time.perf_counter() - started,
        )
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summaries, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
