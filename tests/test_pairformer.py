"""The Pairformer encoder option (``MSDeltaConfig.architecture = "pairformer"``).

Tiny CPU models only (2 layers, small widths, ~16 peaks). The default architecture is pinned
separately in ``TestTransformerUnchanged``: a config that never mentions the new fields must
build, save and run exactly as before.
"""

from __future__ import annotations

import json

import pytest
import torch

from msdelta.models.configuration_msdelta import (MSDeltaConfig, MSDeltaDenoisingConfig,
                                                  MSDeltaRetrievalConfig)
from msdelta.models.embedding import pool_tokens
from msdelta.models.modeling_msdelta import (DeltaMZBias, EncoderBlock, MSDeltaForDenoising,
                                             MSDeltaForPreTraining, MSDeltaForRetrieval,
                                             MSDeltaModel, PeakEmbed)
from msdelta.models.pairformer import (PairformerPeakEmbed, PairformerSingleBlock, PairStack,
                                       TriangleAttention, TriangleMultiplication)

N_PEAKS = 16


def _config(**overrides) -> MSDeltaConfig:
    base = dict(
        architecture="pairformer",
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        delta_bias_n_freqs=8,
        delta_bias_per_head_hidden=4,
        pair_channels=8,
        pair_tri_channels=8,
        pair_use_triangle_attention=True,
        pair_tri_attn_heads=2,
        pair_tri_attn_dim=4,
        pair_tri_attn_chunk=5,          # < N so the chunk loop runs more than once
        pair_opm_channels=4,
        pair_mass_defect_n_freqs=4,
    )
    base.update(overrides)
    return MSDeltaConfig(**base)


def _batch(n: int = N_PEAKS, seed: int = 0) -> dict[str, torch.Tensor]:
    """Two spectra; the second has n - 5 real peaks and 5 padding positions."""
    g = torch.Generator().manual_seed(seed)
    mz = torch.sort(torch.rand(2, n, generator=g) * 1500 + 100, dim=1).values
    log_intensity = torch.rand(2, n, generator=g)
    attention_mask = torch.ones(2, n, dtype=torch.long)
    attention_mask[1, n - 5:] = 0
    mz[1, n - 5:] = 0.0
    log_intensity[1, n - 5:] = 0.0
    return {"mz": mz, "log_intensity": log_intensity, "attention_mask": attention_mask}


def _model(config=None, seed: int = 0):
    torch.manual_seed(seed)
    return MSDeltaModel(config or _config()).eval()


class TestConfig:
    def test_defaults_are_the_transformer(self):
        assert MSDeltaConfig().architecture == "transformer"

    def test_round_trip(self, tmp_path):
        config = _config(pair_update="transition", pair_use_triangle_attention=False,
                         pair_bias_scale=3.0)
        config.save_pretrained(tmp_path)
        loaded = MSDeltaConfig.from_pretrained(tmp_path)
        assert loaded.to_dict() == config.to_dict()
        assert loaded.architecture == "pairformer" and loaded.pair_update == "transition"

    def test_round_trip_inside_head_configs(self):
        for cls in (MSDeltaDenoisingConfig, MSDeltaRetrievalConfig):
            wrapped = cls(encoder=_config().to_dict())
            assert wrapped.encoder.architecture == "pairformer"
            assert wrapped.encoder.pair_channels == 8

    @pytest.mark.parametrize("bad", [
        dict(architecture="rnn"),
        dict(pair_update="attention"),
        dict(pair_channels=0),
        dict(pair_update="static", pair_use_triangle_attention=True),
        dict(pair_bias_scale=-1.0),
        dict(pair_dropout=1.0),
    ])
    def test_invalid_settings_are_rejected(self, bad):
        with pytest.raises(ValueError):
            _config(**bad)

    def test_pair_settings_are_not_validated_for_the_transformer(self):
        MSDeltaConfig(pair_channels=0)   # ignored field, must not break a transformer


class TestForward:
    def test_module_tree_mirrors_the_transformer(self):
        model = _model()
        assert isinstance(model.embed, PairformerPeakEmbed) and isinstance(model.embed, PeakEmbed)
        assert isinstance(model.bias_module, PairStack)
        assert len(model.blocks) == 2
        assert all(isinstance(b, PairformerSingleBlock) for b in model.blocks)
        assert isinstance(model.bias_module.layers[0].tri_out, TriangleMultiplication)
        assert isinstance(model.bias_module.layers[0].tri_attn_start, TriangleAttention)

    @pytest.mark.parametrize("pair_update", ["static", "transition", "triangle"])
    @pytest.mark.parametrize("writeback", [False, True])
    def test_shapes_and_dtypes(self, pair_update, writeback):
        model = _model(_config(pair_update=pair_update, pair_use_writeback=writeback,
                               pair_use_triangle_attention=pair_update == "triangle"))
        with torch.no_grad():
            out = model(**_batch()).last_hidden_state
        assert out.shape == (2, N_PEAKS, 32)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_return_dict_false(self):
        with torch.no_grad():
            out = _model()(**_batch(), return_dict=False)
        assert isinstance(out, tuple) and out[0].shape == (2, N_PEAKS, 32)

    def test_bfloat16_forward(self):
        model = _model().to(torch.bfloat16)
        with torch.no_grad():
            out = model(**_batch()).last_hidden_state
        assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()

    def test_padding_does_not_change_real_peaks(self):
        """Row 1 alone (11 real peaks, no padding) vs the same row padded to 16 and to 40.

        Catches a missing mask in the triangle sums, the triangle attention keys, the
        write-back or the single-stream attention: any of them lets padding move real peaks.
        """
        model = _model()
        batch = _batch()
        real = int(batch["attention_mask"][1].sum())
        alone = {k: v[1:, :real] for k, v in batch.items()}
        with torch.no_grad():
            reference = model(**alone).last_hidden_state[0]
            padded = model(**batch).last_hidden_state[1, :real]
            wide = {k: torch.cat([v, torch.zeros_like(v[:, :24])], dim=1)
                    for k, v in batch.items()}
            wider = model(**wide).last_hidden_state[1, :real]
        assert torch.allclose(reference, padded, atol=1e-5)
        assert torch.allclose(reference, wider, atol=1e-5)

    def test_padding_values_are_irrelevant(self):
        """Garbage in the padded slots (not just zeros) must not leak either."""
        model = _model()
        batch = _batch()
        noisy = {k: v.clone() for k, v in batch.items()}
        noisy["mz"][1, -5:] = torch.tensor([150.0, 151.003355, 900.0, 18.0, 1999.0])
        noisy["log_intensity"][1, -5:] = 1.0
        with torch.no_grad():
            a = model(**batch).last_hidden_state[1, :-5]
            b = model(**noisy).last_hidden_state[1, :-5]
        assert torch.allclose(a, b, atol=1e-5)

    def test_permutation_equivariance(self):
        """Peaks are a set: permuting the input permutes the output."""
        model = _model()
        batch = {k: v[:1] for k, v in _batch().items()}
        perm = torch.randperm(N_PEAKS, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            out = model(**batch).last_hidden_state[0]
            out_p = model(**{k: v[:, perm] for k, v in batch.items()}).last_hidden_state[0]
        assert torch.allclose(out[perm], out_p, atol=1e-5)

    def test_masked_intensity_does_not_leak(self):
        """A masked peak's true intensity must not influence any output."""
        model = _model()
        batch = _batch()
        mask_positions = torch.zeros(2, N_PEAKS, dtype=torch.bool)
        mask_positions[:, 3] = True
        changed = {k: v.clone() for k, v in batch.items()}
        changed["log_intensity"][:, 3] = 0.987
        with torch.no_grad():
            a = model(**batch, mask_positions=mask_positions).last_hidden_state
            b = model(**changed, mask_positions=mask_positions).last_hidden_state
        assert torch.allclose(a, b, atol=1e-6)

    def test_writeback_starts_as_a_no_op(self):
        model = _model()
        for layer in model.bias_module.layers:
            assert layer.opm.out.weight.abs().sum() == 0
            assert layer.opm.out.bias.abs().sum() == 0

    def test_bias_module_diagnostic_surface(self):
        """render_bias_panels / alignment read .evaluate(grid) -> (grid, heads)."""
        from msdelta.utils.viz import render_bias_panels
        model = _model()
        grid = torch.linspace(-5, 5, 101)
        assert model.bias_module.evaluate(grid).shape == (101, 4)
        assert model.bias_module.evaluate_layers(grid).shape == (2, 101, 4)
        assert model.bias_module.ff.out_dim == 16
        panels = render_bias_panels(model.bias_module, step=0)
        assert set(panels) == {"bias/fine", "bias/coarse"}
        import matplotlib.pyplot as plt
        for fig in panels.values():
            plt.close(fig)


class TestTraining:
    def _pretraining_batch(self):
        batch = _batch()
        mask_positions = torch.zeros(2, N_PEAKS, dtype=torch.bool)
        mask_positions[:, [1, 4, 7]] = True
        labels = batch["log_intensity"].exp() * batch["attention_mask"]
        return {**batch, "mask_positions": mask_positions, "labels": labels}

    def test_gradient_reaches_every_parameter(self):
        torch.manual_seed(0)
        model = MSDeltaForPreTraining(_config()).train()
        # The write-back readout is zero at init, which zeroes the gradient into its input
        # projections; perturb it so the check covers them too.
        for layer in model.msdelta.bias_module.layers:
            torch.nn.init.normal_(layer.opm.out.weight, std=0.02)
        loss = model(**self._pretraining_batch()).loss
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        zero = [n for n, p in model.named_parameters()
                if p.grad is not None and p.grad.abs().sum() == 0]
        assert not missing, missing
        assert not zero, zero

    def test_gradient_checkpointing_matches(self):
        torch.manual_seed(0)
        model = MSDeltaForPreTraining(_config()).train()
        batch = self._pretraining_batch()
        plain = model(**batch).loss
        plain.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ckpt = model(**batch).loss
        ckpt.backward()
        assert torch.allclose(plain, ckpt)
        for n, p in model.named_parameters():
            assert torch.allclose(grads[n], p.grad, atol=1e-6), n

    def test_no_mask_positions_still_touches_the_stand_in(self):
        """Denoising/contrastive never pass mask_positions; under DDP every trainable
        parameter must still be in the graph (the mask token is frozen by finetune_denoise)."""
        torch.manual_seed(0)
        model = MSDeltaForDenoising(MSDeltaDenoisingConfig(encoder=_config(), head_hidden_size=8))
        labels = torch.randint(0, 2, (2, N_PEAKS))
        model(**_batch(), labels=labels).loss.backward()
        stand_in = model.msdelta.bias_module.pair_feats.mask_log_intensity
        assert stand_in.grad is not None


class TestHeads:
    def test_save_and_load_round_trip(self, tmp_path):
        torch.manual_seed(0)
        model = MSDeltaForPreTraining(_config()).eval()
        model.save_pretrained(tmp_path)
        saved = json.loads((tmp_path / "config.json").read_text())
        assert saved["architecture"] == "pairformer"
        loaded = MSDeltaForPreTraining.from_pretrained(tmp_path).eval()
        assert isinstance(loaded.msdelta.bias_module, PairStack)
        batch = _batch()
        with torch.no_grad():
            assert torch.equal(model(**batch).logits, loaded(**batch).logits)
        state, loaded_state = model.state_dict(), loaded.state_dict()
        assert state.keys() == loaded_state.keys()
        assert all(torch.equal(state[k], loaded_state[k]) for k in state)

    def test_pretraining_head(self):
        torch.manual_seed(0)
        model = MSDeltaForPreTraining(_config())
        out = TestTraining()._pretraining_batch()
        result = model(**out)
        assert result.logits.shape == (2, N_PEAKS)
        assert result.loss.ndim == 0 and torch.isfinite(result.loss)

    def test_denoising_on_a_pretrained_pairformer(self, tmp_path):
        """finetune_denoise's path: from_pretrained, then wrap .msdelta in MSDeltaForDenoising."""
        torch.manual_seed(0)
        MSDeltaForPreTraining(_config()).save_pretrained(tmp_path)
        pretrained = MSDeltaForPreTraining.from_pretrained(tmp_path)
        config = MSDeltaDenoisingConfig(encoder=MSDeltaConfig(**pretrained.config.to_dict()),
                                        head_hidden_size=8)
        model = MSDeltaForDenoising(config, encoder=pretrained.msdelta)
        model.msdelta.embed.mask_token.requires_grad_(False)
        labels = torch.randint(0, 2, (2, N_PEAKS))
        out = model(**_batch(), labels=labels)
        assert out.logits.shape == (2, N_PEAKS) and torch.isfinite(out.loss)
        out.loss.backward()
        # A denoiser saved and reloaded rebuilds the Pairformer from the nested config.
        model.save_pretrained(tmp_path / "denoise")
        again = MSDeltaForDenoising.from_pretrained(tmp_path / "denoise")
        assert isinstance(again.msdelta.bias_module, PairStack)

    def test_retrieval_and_contrastive_pooling(self):
        torch.manual_seed(0)
        config = MSDeltaRetrievalConfig(encoder=_config(), projection_hidden_size=16,
                                        embedding_size=8)
        model = MSDeltaForRetrieval(config)
        out = model(**_batch(), group_ids=torch.tensor([0, 0]))
        assert out.embeddings.shape == (2, 8) and torch.isfinite(out.loss)
        # The pooling the contrastive fine-tune and the eval scripts use.
        encoder = _model()
        batch = _batch()
        with torch.no_grad():
            tokens = encoder(**batch).last_hidden_state
        pooled = pool_tokens(tokens, batch["attention_mask"].bool())
        assert pooled.shape == (2, 64) and torch.isfinite(pooled).all()

    @pytest.mark.parametrize("pooling", ["mean+max", "layer_mix"])
    def test_contrastive_model(self, pooling):
        """finetune_contrastive's model: pooled embedding + KL to a frozen reference.

        layer_mix hooks encoder.embed and every encoder.blocks[i]; with the Pairformer each
        block's output is the single state, so the hooks see (B, N, hidden) as before.
        """
        from msdelta.finetuning.contrastive.contrastive import MSDeltaForContrastive
        torch.manual_seed(0)
        model = MSDeltaForContrastive(MSDeltaForPreTraining(_config()),
                                      reference=MSDeltaForPreTraining(_config()),
                                      pooling=pooling)
        batch = _batch()
        out = model(**batch, group=torch.tensor([0, 0]))
        assert out["embeddings"].shape[0] == 2
        assert torch.isfinite(out["loss"]) and out["kl"] >= 0
        out["loss"].backward()


class TestTransformerUnchanged:
    """The default architecture builds exactly the pre-existing module tree."""

    def test_default_model_tree(self):
        model = MSDeltaModel(MSDeltaConfig(hidden_size=32, num_attention_heads=4,
                                           num_hidden_layers=2, intermediate_size=64,
                                           delta_bias_n_freqs=8))
        assert type(model.embed) is PeakEmbed
        assert type(model.bias_module) is DeltaMZBias
        assert all(type(b) is EncoderBlock for b in model.blocks)
        assert not any("pair" in n for n, _ in model.named_parameters())

    def test_default_config_json_has_no_new_keys(self, tmp_path):
        MSDeltaConfig().save_pretrained(tmp_path)
        saved = json.loads((tmp_path / "config.json").read_text())
        assert "architecture" not in saved
        assert not [k for k in saved if k.startswith("pair_")]

    def test_old_config_without_the_field_loads_as_transformer(self, tmp_path):
        MSDeltaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                      intermediate_size=64, delta_bias_n_freqs=8).save_pretrained(tmp_path)
        loaded = MSDeltaConfig.from_pretrained(tmp_path)
        assert loaded.architecture == "transformer"
