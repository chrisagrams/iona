"""Tests that a login node cannot run, because it has no XPU.

Run with `qsub -q debug -l select=1 -l walltime=00:30:00 pbs/run_tests.pbs`.

The suite on the login side is honest about its limit: CPU and XPU do not take the same
kernels, so a dtype test that passes on CPU proves nothing about the device. That is
exactly how the fused-kernel bug reached a compute node twice. These run on real silicon.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.device


@pytest.fixture(scope="module", autouse=True)
def _require_xpu():
    if not torch.xpu.is_available():
        pytest.skip("no XPU; run via pbs/run_tests.pbs")


class TestDeviceBasics:
    def test_tiles_match_what_the_launcher_assumes(self):
        """A mismatch here makes set_device(LOCAL_RANK) raise 'device index out of range'."""
        expected = os.environ.get("MSDELTA_XPUS_PER_HOST")
        if not expected:
            pytest.skip("MSDELTA_XPUS_PER_HOST unset")
        assert torch.xpu.device_count() == int(expected)

    def test_xccl_is_available(self):
        assert torch.distributed.is_xccl_available()

    def test_select_device_binds_without_raising(self):
        """select_device must cope with both ZE_AFFINITY_MASK conventions.

        A job-wide mask leaves every tile visible and each rank picks its own; a per-rank
        mask leaves exactly one, renumbered to 0. Guessing wrong killed job 8839150.
        """
        from msdelta.finetune_denoise import select_device
        select_device()

    def test_bf16_matmul_is_finite(self):
        a = torch.randn(64, 64, device="xpu", dtype=torch.bfloat16)
        assert torch.isfinite(a @ a).all()


class TestModelOnDevice:
    def test_tiny_forward(self, tiny_config, spectra):
        from msdelta.modeling_msdelta import MSDeltaModel
        model = MSDeltaModel(tiny_config).to("xpu").eval()
        batch = {k: v.to("xpu") for k, v in spectra.items()}
        with torch.no_grad():
            out = model(**batch)
        torch.xpu.synchronize()
        assert torch.isfinite(out.last_hidden_state).all()

    @pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
    def test_student_eval_under_xpu_autocast(self, student, peptides, weight_dtype):
        """The bug the CPU suite cannot see.

        TransformerEncoderLayer's fused kernel is guarded by the no-argument
        torch.is_autocast_enabled(), which reports CUDA state and is blind to
        torch.autocast("xpu"). Job 8840257 died here with bf16 activations against fp32
        weights, and job 8840336 with the reverse once DeepSpeed supplied bf16 weights.
        Both dtypes, on the device, under autocast -- the exact combination that failed.
        """
        import copy
        from msdelta.reranking import PeptideCollator
        model = copy.deepcopy(student).to("xpu").to(weight_dtype).eval()
        batch = {k: v.to("xpu") for k, v in PeptideCollator()(*peptides).items()}
        with torch.no_grad(), torch.autocast("xpu", dtype=torch.bfloat16):
            out = model(**batch)
        torch.xpu.synchronize()
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_student_trains_a_few_steps(self, student, peptides):
        from msdelta.reranking import PeptideCollator
        model = student.to("xpu")
        batch = {k: v.to("xpu") for k, v in PeptideCollator()(*peptides).items()}
        target = torch.nn.functional.normalize(
            torch.randn(3, 64, device="xpu"), dim=-1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        first = last = None
        for step in range(30):
            optimizer.zero_grad()
            loss = ((model(**batch) - target) ** 2).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            first = loss.item() if step == 0 else first
            last = loss.item()
        torch.xpu.synchronize()
        assert last < first

    def test_probes_do_not_break_a_device_forward(self, student, peptides, monkeypatch):
        """Probes synchronise the queue; that must not change the answer."""
        import msdelta.reranking as r
        from msdelta.reranking import PeptideCollator
        model = student.to("xpu").eval()
        batch = {k: v.to("xpu") for k, v in PeptideCollator()(*peptides).items()}
        monkeypatch.setattr(r, "PROBES", False)
        with torch.no_grad():
            quiet = model(**batch)
        monkeypatch.setattr(r, "PROBES", True)
        with torch.no_grad():
            probed = model(**batch)
        assert torch.allclose(quiet, probed, atol=1e-5)


@pytest.mark.slow
class TestRealCheckpoint:
    def test_loads_and_runs_on_device(self, checkpoint, spectra):
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        model = MSDeltaForPreTraining.from_pretrained(checkpoint).to("xpu").eval()
        batch = {k: v.to("xpu") for k, v in spectra.items()}
        with torch.no_grad():
            encoder = getattr(model, "msdelta", model)
            hidden = encoder(**batch).last_hidden_state
        torch.xpu.synchronize()
        assert torch.isfinite(hidden).all()

    def test_save_and_reload_round_trip(self, checkpoint, spectra, tmp_path):
        """A checkpoint that cannot be reloaded makes a finished run worthless."""
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        model = MSDeltaForPreTraining.from_pretrained(checkpoint).to("xpu").eval()
        batch = {k: v.to("xpu") for k, v in spectra.items()}
        encoder = getattr(model, "msdelta", model)
        with torch.no_grad():
            before = encoder(**batch).last_hidden_state.cpu()
        model.save_pretrained(tmp_path / "round-trip")
        reloaded = MSDeltaForPreTraining.from_pretrained(tmp_path / "round-trip")
        reloaded = reloaded.to("xpu").eval()
        with torch.no_grad():
            after = getattr(reloaded, "msdelta", reloaded)(**batch).last_hidden_state.cpu()
        assert torch.allclose(before, after, atol=1e-4)
