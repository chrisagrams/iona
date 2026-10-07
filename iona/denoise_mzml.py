"""Denoise the spectra of mzML files with a trained ``IonaForDenoising`` model.

Example:
    iona-denoise run.mzML --checkpoint runs/denoise-probes/step-190000 \\
        --output run.denoised.mzML

Multi-GPU:
    accelerate launch --num_processes 4 --module iona.denoise_mzml run.mzML \\
        --checkpoint runs/denoise-probes/step-190000 --output run.denoised.mzML
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from itertools import batched
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from tqdm.auto import tqdm

from iona.denoising import PeakBudgetBatchSampler
from iona.modeling_iona import IonaForDenoising
from iona.mzml import MzMLRewriter, Spectrum
from iona.processing_iona import IonaProcessor

logger = logging.getLogger(__name__)

@dataclass
class DenoisingSummary:
    """Peak and spectrum counts accumulated over one mzML file."""

    spectra: int = 0
    denoised_spectra: int = 0
    peaks_in: int = 0
    peaks_out: int = 0
    noise_peaks: int = 0
    emptied_spectra: int = 0

    def record(self, peak_count: int, keep: np.ndarray | None) -> None:
        """Accumulate counts for one copied or denoised spectrum."""
        self.spectra += 1
        if keep is None:
            return
        kept = int(keep.sum())
        self.denoised_spectra += 1
        self.peaks_in += peak_count
        self.peaks_out += kept
        self.noise_peaks += peak_count - kept
        self.emptied_spectra += int(peak_count > 0 and kept == 0)


class SpectrumDenoiser:
    """Score peaks as noise with ``IonaForDenoising`` and build keep masks."""

    def __init__(
        self,
        model: IonaForDenoising,
        processor: IonaProcessor,
        *,
        noise_threshold: float = 0.5,
        peak_pair_budget: int = 4_194_304,
        distributed_state: PartialState | None = None,
        compile_model: bool = True,
    ):
        if not 0.0 < noise_threshold <= 1.0:
            raise ValueError("noise_threshold must be in (0, 1]")
        if peak_pair_budget < processor.max_peaks**2:
            raise ValueError("peak_pair_budget must fit one spectrum of max_peaks peaks")
        self.device = next(model.parameters()).device
        model = model.eval()
        if compile_model and self.device.type != "cpu":
            # Batches vary in both spectrum count and padded peak count, so compile dynamically
            # rather than recompiling per shape. Warmup costs a few seconds and pays for itself
            # after roughly a thousand spectra.
            model = torch.compile(model, dynamic=True)
        self.model = model
        self.processor = processor
        self.noise_threshold = noise_threshold
        self.peak_pair_budget = peak_pair_budget
        self.distributed_state = distributed_state

    def _windows(self, length: int) -> list[slice]:
        """Cover a peak sequence with the minimum number of evenly overlapping windows."""
        max_peaks = self.processor.max_peaks
        if length <= max_peaks:
            return [slice(0, length)]
        count = int(np.ceil(length / max_peaks))
        starts = np.linspace(0, length - max_peaks, count, dtype=int)
        return [slice(int(start), int(start) + max_peaks) for start in starts]

    def _local_noise_probabilities(
        self, spectra: list[tuple[np.ndarray, np.ndarray]]
    ) -> list[np.ndarray]:
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
        if any(result is None for result in results):
            raise RuntimeError("inference did not return every local spectrum window")
        return [result for result in results if result is not None]

    @torch.inference_mode()
    def noise_probabilities(self, spectra: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """Return one noise probability per peak for each ``(mz, intensity)`` pair."""
        indexed_spectra = list(enumerate(spectra))
        state = self.distributed_state
        if state is not None and state.num_processes > 1:
            with state.split_between_processes(indexed_spectra) as local_items:
                local_items = list(local_items)
        else:
            local_items = indexed_spectra

        local_probabilities = self._local_noise_probabilities(
            [spectrum for _, spectrum in local_items]
        )
        indexed_probabilities = [
            (index, probability)
            for (index, _), probability in zip(local_items, local_probabilities)
        ]
        if state is not None and state.num_processes > 1:
            indexed_probabilities = gather_object(indexed_probabilities)

        results: list[np.ndarray | None] = [None] * len(spectra)
        for index, probability in indexed_probabilities:
            results[index] = probability
        if any(result is None for result in results):
            raise RuntimeError("distributed inference did not return every spectrum window")
        return [result for result in results if result is not None]

    def keep_masks(self, spectra: Iterable[Spectrum]) -> list[np.ndarray]:
        """Return a boolean keep mask for every spectrum's peaks."""
        spectra = list(spectra)
        masks = [np.ones(spectrum.peak_count, dtype=bool) for spectrum in spectra]
        windows: list[tuple[int, slice]] = []
        for i, spectrum in enumerate(spectra):
            if spectrum.peak_count == 0 or spectrum.intensity.max() <= 0:
                continue
            windows.extend((i, window) for window in self._windows(spectrum.peak_count))
        probabilities = self.noise_probabilities(
            [(spectra[i].mz[window], spectra[i].intensity[window]) for i, window in windows]
        )
        probability_sums = [np.zeros(spectrum.peak_count, dtype=np.float32) for spectrum in spectra]
        prediction_counts = [np.zeros(spectrum.peak_count, dtype=np.int32) for spectrum in spectra]
        for (i, window), noise in zip(windows, probabilities):
            probability_sums[i][window] += noise
            prediction_counts[i][window] += 1
        for i, counts in enumerate(prediction_counts):
            scored = counts > 0
            masks[i][scored] = (
                probability_sums[i][scored] / counts[scored] < self.noise_threshold
            )
        return masks


def denoise_mzml(
    source: Path,
    destination: Path | None,
    denoiser: SpectrumDenoiser,
    *,
    ms_levels: set[int] | None = None,
    chunk_size: int = 1024,
    show_progress: bool = True,
) -> DenoisingSummary:
    """Write ``destination`` as a copy of ``source`` with noise peaks removed."""
    summary = DenoisingSummary()
    with (
        MzMLRewriter(source, destination) as rewriter,
        tqdm(desc=source.name, unit="spectra", disable=not show_progress) as progress,
    ):
        for chunk in batched(rewriter.spectra(), chunk_size):
            if progress.total is None and rewriter.spectrum_count is not None:
                progress.reset(total=rewriter.spectrum_count)
            targets = [
                spectrum
                for spectrum in chunk
                if ms_levels is None or spectrum.ms_level in ms_levels
            ]
            masks = dict(zip((s.index for s in targets), denoiser.keep_masks(targets)))
            for spectrum in chunk:
                keep = masks.get(spectrum.index)
                summary.record(spectrum.peak_count, keep)
                rewriter.write(spectrum, keep)
                progress.update()
    return summary


def load_denoiser(
    checkpoint: str,
    *,
    processor_path: str | None = None,
    device: str | torch.device | None = None,
    distributed_state: PartialState | None = None,
    **kwargs,
) -> SpectrumDenoiser:
    """Load a denoising checkpoint and its processor onto ``device``."""
    if distributed_state is not None and distributed_state.num_processes > 1:
        if device is not None:
            raise ValueError("--device cannot be used with distributed inference")
        if distributed_state.device.type == "cuda":
            device = torch.device(
                "cuda", distributed_state.local_process_index % torch.cuda.device_count()
            )
            torch.cuda.set_device(device)
            distributed_state.device = device
        else:
            device = distributed_state.device
    elif device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IonaForDenoising.from_pretrained(checkpoint, dtype="auto").to(device)
    processor = IonaProcessor.from_pretrained(processor_path or checkpoint)
    return SpectrumDenoiser(model, processor, distributed_state=distributed_state, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="mzML file(s) to denoise")
    parser.add_argument(
        "--checkpoint", required=True, help="IonaForDenoising checkpoint path or Hub ID"
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
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile")
    parser.add_argument(
        "--summary", type=Path, help="write per-file peak and spectrum counts to this JSON file"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace outputs that already exist"
    )
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    args = parser.parse_args(argv)
    if args.output is not None and len(args.inputs) != 1:
        parser.error("--output accepts exactly one input; use --output-dir for several")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state = PartialState()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    denoiser = load_denoiser(
        args.checkpoint,
        processor_path=args.processor,
        device=args.device,
        distributed_state=state,
        noise_threshold=args.noise_threshold,
        peak_pair_budget=args.peak_pair_budget,
        compile_model=not args.no_compile,
    )
    if args.output_dir is not None and state.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    state.wait_for_everyone()

    summaries: dict[str, dict] = {}
    for source in args.inputs:
        destination = args.output if args.output is not None else args.output_dir / source.name
        if destination.resolve() == source.resolve():
            raise SystemExit(f"refusing to overwrite the input file {source}")
        if destination.exists() and not args.overwrite:
            if state.is_main_process:
                logger.info("skipping %s: %s exists", source, destination)
            continue
        started = time.perf_counter()
        partial = destination.with_name(destination.name + ".part")
        try:
            summary = denoise_mzml(
                source,
                partial if state.is_main_process else None,
                denoiser,
                ms_levels=set(args.ms_level),
                chunk_size=args.chunk_size,
                show_progress=not args.no_progress and state.is_main_process,
            )
            if state.is_main_process:
                partial.replace(destination)
            state.wait_for_everyone()
        except BaseException:
            if state.is_main_process:
                partial.unlink(missing_ok=True)
            raise
        if state.is_main_process:
            summaries[str(source)] = asdict(summary)
            logger.info(
                "%s: %d/%d spectra denoised, %d -> %d peaks (%d noise) in %.1fs",
                source.name,
                summary.denoised_spectra,
                summary.spectra,
                summary.peaks_in,
                summary.peaks_out,
                summary.noise_peaks,
                time.perf_counter() - started,
            )
    if args.summary is not None and state.is_main_process:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summaries, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
