"""Evaluation: probes, chemical alignment, retrieval, denoising, and bias plots."""

from msdelta.eval.alignment import RangeSpec, alignment_metrics, analyze_range, reference_set
from msdelta.eval.denoising import denoising_metrics, run_denoising_probe
from msdelta.eval.embedding import embed_spectra, encode_batch, pool_tokens
from msdelta.eval.probe import extract_representations, run_all_probes
from msdelta.eval.retrieval import (
    all_but_top,
    load_benchmark,
    replicate_retrieval_inline_metrics,
    retrieval_inline_metrics,
    retrieval_metrics_tm,
)
from msdelta.eval.viz import plot_bias_curves, render_bias_panels

__all__ = [
    "RangeSpec",
    "alignment_metrics",
    "all_but_top",
    "analyze_range",
    "denoising_metrics",
    "embed_spectra",
    "encode_batch",
    "extract_representations",
    "load_benchmark",
    "plot_bias_curves",
    "pool_tokens",
    "reference_set",
    "render_bias_panels",
    "replicate_retrieval_inline_metrics",
    "retrieval_inline_metrics",
    "retrieval_metrics_tm",
    "run_all_probes",
    "run_denoising_probe",
]
