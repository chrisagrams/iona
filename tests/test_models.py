"""Forward passes, and the dtype and freezing behaviour around them."""

from __future__ import annotations

import copy

import pytest
import torch


class TestEncoderForward:
    def test_hidden_state_shape(self, tiny_config, spectra):
        from msdelta.modeling_msdelta import MSDeltaModel
        out = MSDeltaModel(tiny_config).eval()(**spectra)
        assert out.last_hidden_state.shape == (2, 6, tiny_config.hidden_size)
        assert torch.isfinite(out.last_hidden_state).all()

    @pytest.mark.parametrize("width", [10, 512])
    def test_padding_does_not_change_real_peaks(self, tiny_config, spectra, width):
        """Row 1 has four real peaks; padding the batch out must not move them.

        If the attention mask is wrong, padded positions contribute and the real peaks
        shift -- silently, and in a way no shape assertion catches.

        512 is here because the fix for the scratch GPU fault switched the collators
        from padding to the batch maximum to padding to a fixed max_peaks, and every
        number measured before that switch is only comparable with every number
        measured after it if width is genuinely inert. The batch max in production runs
        is often under 100, so the fixed-width path pads five to ten times further than
        the variable path ever did -- far enough that a mask leak too small to see at
        width 10 would be visible. Measured difference at 512: 3.6e-07, which is float32
        rounding, so the switch changed memory and nothing else.
        """
        from msdelta.modeling_msdelta import MSDeltaModel
        model = MSDeltaModel(tiny_config).eval()
        pad = width - spectra["mz"].shape[1]
        with torch.no_grad():
            short = model(**spectra).last_hidden_state[1, :4]
            wider = {k: torch.cat([v, torch.zeros_like(v[:, :1]).expand(-1, pad)], dim=1)
                     for k, v in spectra.items()}
            long = model(**wider).last_hidden_state[1, :4]
        assert torch.allclose(short, long, atol=1e-4)

    def test_denoising_head_emits_one_logit_per_peak(self, tiny_denoising_config, spectra):
        """One logit per peak under BCE, not two under cross-entropy."""
        from msdelta.modeling_msdelta import MSDeltaForDenoising
        model = MSDeltaForDenoising(tiny_denoising_config).eval()
        with torch.no_grad():
            out = model(**spectra)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        assert logits.shape[:2] == (2, 6)
        assert logits.shape[2:] in ((), (1,)), f"one logit per peak, got {logits.shape}"
