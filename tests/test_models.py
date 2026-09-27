"""Forward passes, and the dtype and freezing behaviour around them."""

from __future__ import annotations

import copy

import pytest
import torch

from msdelta.reranking import (AlignmentCollator, PeptideEncoder, SequenceAlignmentModel,
                               attach_teacher_embeddings, pooled_width,
                               teacher_embedding_size)


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


class TestPeptideEncoder:
    def test_output_is_unit_norm(self, student, peptides):
        from msdelta.reranking import PeptideCollator
        out = student.eval()(**PeptideCollator()(*peptides))
        assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)

    def test_width_follows_pooling(self, peptides):
        from msdelta.reranking import PeptideCollator
        batch = PeptideCollator()(*peptides)
        for mode in ("mean", "mean+max"):
            enc = PeptideEncoder(embedding_size=64, hidden_size=32, num_layers=1,
                                 num_heads=4, pooling=mode)
            assert enc.eval()(**batch).shape == (3, 64)
            assert pooled_width(32, mode) == enc.projection[0].in_features

    @pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
    def test_eval_under_autocast(self, student, peptides, weight_dtype):
        """The fused-kernel bug, both directions.

        TransformerEncoderLayer takes torch._transformer_encoder_layer_fwd only when grad
        is disabled -- eval, never training -- and its autocast guard calls the
        no-argument torch.is_autocast_enabled(), which reports CUDA state and is blind to
        XPU. Forcing fp32 fixed job 8840257 and broke 8840336 under DeepSpeed, which
        holds weights in bf16. Both weight dtypes have to work.
        """
        from msdelta.reranking import PeptideCollator
        model = copy.deepcopy(student).to(weight_dtype).eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(**PeptideCollator()(*peptides))
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_train_and_eval_agree_without_dropout(self, peptides):
        """Training and evaluation must not silently take different numerical paths."""
        from msdelta.reranking import PeptideCollator
        enc = PeptideEncoder(embedding_size=64, hidden_size=32, num_layers=1,
                             num_heads=4, dropout=0.0)
        batch = PeptideCollator()(*peptides)
        with torch.no_grad():
            trained = enc.train()(**batch)
            evaled = enc.eval()(**batch)
        assert torch.allclose(trained, evaled, atol=1e-5)


class TestAlignmentModel:
    def _model(self, tiny_config, pooling="mean+max"):
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        teacher = MSDeltaForPreTraining(tiny_config)
        width = teacher_embedding_size(teacher, pooling)
        return SequenceAlignmentModel(
            teacher, PeptideEncoder(embedding_size=width, hidden_size=32, num_layers=1,
                                    num_heads=4, pooling=pooling), pooling=pooling)

    def _rows(self):
        return [{"mz": [100.0, 200.0, 300.0], "log_intensity": [1.0, 2.0, 3.0],
                 "peptide": "PEPTIDE", "charge": 2},
                {"mz": [150.0, 250.0], "log_intensity": [0.5, 1.5],
                 "peptide": "MK", "charge": 3}]

    def test_teacher_is_frozen_and_in_eval(self, tiny_config):
        """requires_grad_(False) alone leaves dropout live and the target moving."""
        model = self._model(tiny_config).train()
        assert not any(p.requires_grad for p in model.spectrum_model.parameters())
        assert not model.spectrum_model.training
        assert model.sequence_encoder.training

    def test_pooling_mismatch_raises(self, tiny_config):
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        with pytest.raises(ValueError, match="pooling"):
            SequenceAlignmentModel(
                MSDeltaForPreTraining(tiny_config),
                PeptideEncoder(embedding_size=8, hidden_size=32, num_layers=1,
                               num_heads=4, pooling="mean"),
                pooling="mean+max")

    def test_default_loss_is_plain_mse(self):
        """The alignment recipe is squared L2 onto the frozen teacher's embedding (A1).
        The LiT/hard-negative alternatives (A4, A8) are not in it; the default must not drift."""
        from msdelta.reranking import PeptideCollator
        torch.manual_seed(0)
        student = PeptideEncoder(embedding_size=8, hidden_size=16, num_layers=1, num_heads=2,
                                 dropout=0.0)
        model = SequenceAlignmentModel(None, student).eval()
        batch = PeptideCollator()(["PEPTIDEK", "ACDK"], [2, 2])
        target = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
        out = model(**batch, target=target)
        expected = ((out["embeddings"] - target) ** 2).sum(-1).mean()
        assert float(out["loss"]) == pytest.approx(float(expected), abs=1e-6)

    def test_loss_is_finite_and_shapes_match(self, tiny_config):
        model = self._model(tiny_config).eval()
        with torch.no_grad():
            out = model(**AlignmentCollator()(self._rows()))
        assert torch.isfinite(out["loss"])
        assert out["embeddings"].shape == out["target"].shape

    def test_precomputed_target_equals_the_live_one(self, tiny_config):
        """The entire precompute path rests on this equality."""
        from datasets import Dataset
        model = self._model(tiny_config).eval()
        rows = self._rows()
        with torch.no_grad():
            live = model(**AlignmentCollator()(rows))
        cached = attach_teacher_embeddings(
            {"t": Dataset.from_list(rows)}, model.spectrum_model, model.pooling,
            batch_size=2, device="cpu")["t"]
        with torch.no_grad():
            out = model(**AlignmentCollator()(list(cached)))
        assert out["loss"].item() == pytest.approx(live["loss"].item(), abs=1e-5)

    def test_teacher_detaches(self, tiny_config):
        """Setting spectrum_model=None is what keeps it out of the DDP/ZeRO graph."""
        from datasets import Dataset
        model = self._model(tiny_config).eval()
        rows = self._rows()
        cached = attach_teacher_embeddings(
            {"t": Dataset.from_list(rows)}, model.spectrum_model, model.pooling,
            batch_size=2, device="cpu")["t"]
        before = sum(p.numel() for p in model.parameters())
        model.spectrum_model = None
        after = sum(p.numel() for p in model.parameters())
        assert after < before
        with torch.no_grad():
            assert torch.isfinite(model(**AlignmentCollator()(list(cached)))["loss"])

    def test_missing_teacher_and_missing_target_raises(self, tiny_config):
        """Failing loudly beats training against a silently wrong target."""
        model = self._model(tiny_config).eval()
        model.spectrum_model = None
        with pytest.raises(ValueError, match="no teacher"):
            model(**AlignmentCollator()(self._rows()))


class TestTargetCache:
    """The cached path must never construct a teacher.

    Every alignment run that faulted on twelve tiles loaded the 49.8M teacher onto the
    device, used it and abandoned it; the bisect that runs clean never loads one, and the
    denoise fine-tune never abandons one because its encoder IS the trained model. The
    point of precomputing to disk is that the training process has no teacher at all, so
    this asserts the absence rather than trusting the code path to stay that way.
    """

    def test_manifest_round_trips(self, tiny_config, tmp_path):
        from datasets import Dataset
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        from msdelta.reranking import attach_teacher_embeddings, teacher_embedding_size
        teacher = MSDeltaForPreTraining(tiny_config)
        rows = [{"mz": [100.0, 200.0], "log_intensity": [1.0, 2.0],
                 "peptide": "PEPTIDE", "charge": 2}]
        cached = attach_teacher_embeddings({"train": Dataset.from_list(rows)}, teacher,
                                           "mean+max", batch_size=1, device="cpu")["train"]
        width = teacher_embedding_size(teacher, "mean+max")
        assert len(cached[0]["target"]) == width

    def test_training_path_loads_no_teacher(self, tiny_config, tmp_path, monkeypatch):
        import msdelta.modeling_msdelta as mm
        from datasets import Dataset
        from msdelta.reranking import (AlignmentCollator, PeptideEncoder,
                                       SequenceAlignmentModel, attach_teacher_embeddings)

        teacher = mm.MSDeltaForPreTraining(tiny_config)
        rows = [{"mz": [100.0, 200.0], "log_intensity": [1.0, 2.0],
                 "peptide": "PEPTIDE", "charge": 2},
                {"mz": [150.0], "log_intensity": [0.5], "peptide": "MK", "charge": 3}]
        cached = list(attach_teacher_embeddings({"t": Dataset.from_list(rows)}, teacher,
                                                "mean+max", batch_size=2,
                                                device="cpu")["t"])
        width = len(cached[0]["target"])

        loads = []
        original = mm.MSDeltaForPreTraining.from_pretrained.__func__
        monkeypatch.setattr(mm.MSDeltaForPreTraining, "from_pretrained",
                            classmethod(lambda cls, *a, **k: (loads.append(a),
                                                              original(cls, *a, **k))[1]))
        model = SequenceAlignmentModel(
            None, PeptideEncoder(embedding_size=width, hidden_size=32, num_layers=1,
                                 num_heads=4), pooling="mean+max")
        out = model(**AlignmentCollator()(cached))
        assert loads == [], "the cached path constructed a teacher"
        assert model.spectrum_model is None
        assert torch.isfinite(out["loss"])
