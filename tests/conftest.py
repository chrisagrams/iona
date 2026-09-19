"""Shared fixtures. Everything here is CPU-sized: the suite has to finish in seconds.

The models built here are deliberately tiny -- hidden_size 32, two layers, a handful of
peaks. Nothing in this suite is trying to measure quality; it is checking that shapes,
dtypes, masks and wiring are right, and those are all size-independent. A test that needs
a real 50m checkpoint is marked `slow` and skipped unless the checkpoint is readable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
CHECKPOINT = Path(os.environ.get(
    "MSDELTA_TEST_CHECKPOINT",
    "/flare/UIC-HPC/homes/cgrams/msdelta-runs/msdelta-50m-production-01/final"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the real checkpoint or the network")
    config.addinivalue_line("markers", "device: needs an XPU; run via pbs/run_tests.pbs")


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)


@pytest.fixture
def tiny_config():
    from msdelta.configuration_msdelta import MSDeltaConfig
    # delta_bias_n_freqs is 8 rather than the production 256 purely for speed: the bias
    # tensor is (B, L, L, 2*n_freqs) and at 256 even a toy batch is tens of megabytes.
    return MSDeltaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                         intermediate_size=64, delta_bias_n_freqs=8,
                         delta_bias_per_head_hidden=4)


@pytest.fixture
def tiny_denoising_config(tiny_config):
    """MSDeltaForDenoising takes a wrapper config holding the encoder, not a bare one."""
    from msdelta.configuration_msdelta import MSDeltaDenoisingConfig
    return MSDeltaDenoisingConfig(encoder=tiny_config, head_hidden_size=16)


@pytest.fixture
def spectra():
    """Two spectra of different lengths, so padding is always exercised."""
    mz = torch.zeros(2, 6)
    log_intensity = torch.zeros(2, 6)
    attention_mask = torch.zeros(2, 6, dtype=torch.long)
    mz[0, :6] = torch.tensor([100.1, 200.2, 300.3, 400.4, 500.5, 600.6])
    mz[1, :4] = torch.tensor([150.5, 250.5, 350.5, 450.5])
    log_intensity[0, :6] = torch.linspace(1.0, 2.0, 6)
    log_intensity[1, :4] = torch.linspace(0.5, 1.5, 4)
    attention_mask[0, :6] = 1
    attention_mask[1, :4] = 1
    return {"mz": mz, "log_intensity": log_intensity, "attention_mask": attention_mask}


@pytest.fixture
def peptides():
    return (["SAC[57.0215]GVC[57.0215]PGR", "AKPEPTIDEK", "MK"], [2, 3, 2])


@pytest.fixture
def student():
    from msdelta.reranking import PeptideEncoder
    return PeptideEncoder(embedding_size=64, hidden_size=32, num_layers=2, num_heads=4)


@pytest.fixture
def checkpoint():
    if not os.access(CHECKPOINT / "model.safetensors", os.R_OK):
        pytest.skip(f"checkpoint not readable: {CHECKPOINT}")
    return CHECKPOINT


def args_files():
    """Every training.args in the repo, named configs excluding generated sweep arms.

    Sweep arms are generated from a template and verified by make_denoise_grid --check,
    so parsing all 216 here would be slow and would only re-test the generator.
    """
    return sorted(p for p in (REPO / "configs").glob("*/training.args")
                  if "sweep-denoise" not in str(p))
