"""Contrastive fine-tune of the spectrum encoder.

The objective exists because the pretrained encoder's embedding space does not separate
peptides: on the replicate corpus only 1% of peptide+charge groups have every replicate
nearer to each other than to any other peptide. These check the machinery that is meant
to fix that, including the two NaN paths that would have made every real batch NaN.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msdelta.contrastive import (GroupBatchSampler, head_kl, subset_by_group,
                                 supervised_contrastive_loss)


class TestSupervisedContrastiveLoss:
    def _grouped(self, noise=0.01):
        torch.manual_seed(0)
        centres = torch.eye(3).repeat_interleave(2, 0)
        return torch.nn.functional.normalize(centres + noise * torch.randn(6, 3), dim=-1)

    def test_separated_scores_better_than_random(self):
        groups = torch.tensor([0, 0, 1, 1, 2, 2])
        torch.manual_seed(0)
        random = torch.nn.functional.normalize(torch.randn(6, 3), dim=-1)
        assert supervised_contrastive_loss(self._grouped(), groups) < \
               supervised_contrastive_loss(random, groups)

    def test_is_finite_with_padding_and_self_masking(self):
        """-inf on the diagonal times a False mask is NaN; the loss must use where()."""
        groups = torch.tensor([0, 0, 1, 1, 2, 2])
        assert torch.isfinite(supervised_contrastive_loss(self._grouped(), groups))

    def test_no_positives_returns_zero_not_nan(self):
        """A batch of singleton groups is degenerate but must not poison training."""
        loss = supervised_contrastive_loss(torch.randn(4, 8), torch.tensor([0, 1, 2, 3]))
        assert torch.isfinite(loss) and loss == 0.0

    def test_more_than_two_positives_all_count(self):
        """SupCon, not InfoNCE: with K replicates all K-1 are positives."""
        embeddings = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [1.0, 0.01], [1.0, -0.01], [0.0, 1.0]]), dim=-1)
        tight = supervised_contrastive_loss(embeddings, torch.tensor([0, 0, 0, 1]))
        split = supervised_contrastive_loss(embeddings, torch.tensor([0, 0, 2, 1]))
        assert torch.isfinite(tight) and torch.isfinite(split)

    def test_temperature_changes_the_value(self):
        groups = torch.tensor([0, 0, 1, 1, 2, 2])
        embeddings = self._grouped(noise=0.3)
        assert supervised_contrastive_loss(embeddings, groups, 0.07) != \
               supervised_contrastive_loss(embeddings, groups, 0.5)


class TestHeadKL:
    def _masked(self):
        logits = torch.randn(3, 10)
        mask = torch.ones(3, 10, dtype=torch.long)
        mask[:, 7:] = 0
        return logits, mask

    def test_zero_against_itself(self):
        logits, mask = self._masked()
        assert head_kl(logits, logits, mask).abs() < 1e-6

    def test_positive_against_something_else(self):
        logits, mask = self._masked()
        assert head_kl(logits, torch.randn(3, 10), mask) > 0

    def test_finite_with_padding(self):
        """F.kl_div on -inf padded positions gives 0 * NaN; every real batch has padding."""
        logits, mask = self._masked()
        assert torch.isfinite(head_kl(logits, torch.randn(3, 10), mask))

    def test_padding_contributes_nothing(self):
        logits, mask = self._masked()
        altered = logits.clone()
        altered[:, 7:] = 999.0
        assert torch.allclose(head_kl(altered, logits, mask),
                              head_kl(logits, logits, mask), atol=1e-6)


class TestGroupBatchSampler:
    def test_every_batch_has_the_requested_shape(self):
        groups = np.repeat(np.arange(40), 13)
        sampler = GroupBatchSampler(groups, groups_per_batch=6, replicates=4, seed=0)
        batch = next(iter(sampler))
        assert len(batch) == 24
        _, counts = np.unique(groups[batch], return_counts=True)
        assert counts.tolist() == [4] * 6

    def test_positives_exist_by_construction(self):
        """The whole point: random batches of this corpus hold about one positive pair."""
        groups = np.repeat(np.arange(40), 13)
        sampler = GroupBatchSampler(groups, groups_per_batch=6, replicates=4, seed=0)
        _, counts = np.unique(groups[next(iter(sampler))], return_counts=True)
        assert sum(c * (c - 1) // 2 for c in counts) == 36

    def test_small_groups_are_sampled_not_dropped(self):
        """Dropping them biases training toward frequently observed peptides."""
        groups = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
        sampler = GroupBatchSampler(groups, groups_per_batch=2, replicates=4, seed=0)
        assert len(next(iter(sampler))) == 8

    def test_replicates_below_two_is_rejected(self):
        with pytest.raises(ValueError, match="positive pairs"):
            GroupBatchSampler(np.arange(10), replicates=1)

    def test_epochs_differ(self):
        groups = np.repeat(np.arange(40), 13)
        sampler = GroupBatchSampler(groups, groups_per_batch=6, replicates=4, seed=0)
        first = next(iter(sampler))
        sampler.set_epoch(1)
        assert next(iter(sampler)) != first


class TestSubsetByGroup:
    def test_keeps_whole_groups(self):
        """A contiguous slice of this corpus yields groups of ~2 and breaks the loss."""
        from datasets import Dataset
        rows = [{"peptide": f"P{i // 10}", "charge": 2} for i in range(100)]
        out = subset_by_group(Dataset.from_list(rows), 40,
                              lambda r: f"{r['peptide']}_{r['charge']}", min_members=4)
        counts = {}
        for row in out:
            counts[row["peptide"]] = counts.get(row["peptide"], 0) + 1
        assert all(count == 10 for count in counts.values())
        assert len(out) <= 40 + 10

    def test_rejects_groups_below_the_minimum(self):
        from datasets import Dataset
        rows = [{"peptide": "SMALL", "charge": 2}] * 2 + \
               [{"peptide": "BIG", "charge": 2}] * 10
        out = subset_by_group(Dataset.from_list(rows), 100,
                              lambda r: f"{r['peptide']}_{r['charge']}", min_members=4)
        assert {row["peptide"] for row in out} == {"BIG"}

    def test_zero_is_a_no_op(self):
        from datasets import Dataset
        rows = [{"peptide": "A", "charge": 2}] * 5
        data = Dataset.from_list(rows)
        assert len(subset_by_group(data, 0, lambda r: r["peptide"])) == 5


class TestContrastiveModel:
    def test_forward_reports_both_terms(self, tiny_config, spectra):
        from msdelta.contrastive import MSDeltaForContrastive
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        model = MSDeltaForContrastive(MSDeltaForPreTraining(tiny_config),
                                      MSDeltaForPreTraining(tiny_config),
                                      pooling="mean+max", kl_weight=1.0)
        out = model(**spectra, group=torch.tensor([0, 0]))
        assert torch.isfinite(out["loss"])
        assert "contrastive" in out and "kl" in out

    def test_reference_is_frozen_and_in_eval(self, tiny_config):
        from msdelta.contrastive import MSDeltaForContrastive
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        model = MSDeltaForContrastive(MSDeltaForPreTraining(tiny_config),
                                      MSDeltaForPreTraining(tiny_config)).train()
        assert not any(p.requires_grad for p in model.reference.parameters())
        assert not model.reference.training
        assert model.model.training

    def test_kl_weight_zero_needs_no_reference(self, tiny_config, spectra):
        from msdelta.contrastive import MSDeltaForContrastive
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        model = MSDeltaForContrastive(MSDeltaForPreTraining(tiny_config), None,
                                      kl_weight=0.0)
        out = model(**spectra, group=torch.tensor([0, 0]))
        assert torch.isfinite(out["loss"]) and float(out["kl"]) == 0.0

    def test_kl_without_reference_or_cache_raises(self, tiny_config, spectra):
        """Silently skipping the regulariser would let the encoder forget unnoticed."""
        from msdelta.contrastive import MSDeltaForContrastive
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        model = MSDeltaForContrastive(MSDeltaForPreTraining(tiny_config), None,
                                      kl_weight=1.0)
        with pytest.raises(ValueError, match="reference"):
            model(**spectra, group=torch.tensor([0, 0]))


class TestTrainerIntegration:
    """MSDeltaForContrastive is a plain nn.Module, so Trainer's hooks must be forwarded.

    Trainer calls gradient_checkpointing_enable() on whatever model it is handed. A
    PreTrainedModel has it; a bare nn.Module does not, and the run dies after data
    loading with an AttributeError -- which is exactly how job 8840597 failed. And
    checkpointing is not optional here: the PK sampler's batch of 24 spectra at 512
    peaks needs it to fit on a tile.
    """

    def _model(self, tiny_config):
        from msdelta.contrastive import MSDeltaForContrastive
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        return MSDeltaForContrastive(MSDeltaForPreTraining(tiny_config),
                                     MSDeltaForPreTraining(tiny_config))

    def test_gradient_checkpointing_hooks_exist(self, tiny_config):
        model = self._model(tiny_config)
        model.gradient_checkpointing_enable()
        assert model.is_gradient_checkpointing
        model.gradient_checkpointing_disable()
        assert not model.is_gradient_checkpointing

    def test_forward_works_with_checkpointing_on(self, tiny_config, spectra):
        model = self._model(tiny_config)
        model.gradient_checkpointing_enable()
        out = model(**spectra, group=torch.tensor([0, 0]))
        assert torch.isfinite(out["loss"])

    def test_checkpointing_is_not_applied_to_the_reference(self, tiny_config):
        """It runs under no_grad and stores no activations; checkpointing only costs."""
        model = self._model(tiny_config)
        model.gradient_checkpointing_enable()
        assert not getattr(model.reference, "is_gradient_checkpointing", False)


class TestGradCache:
    """GradCache must produce the EXACT full-batch gradient, not an approximation.

    It exists because the epochs ladder showed the constraint is negatives rather than
    steps: three epochs reached a separation ratio of 6.94 and ten epochs fell to 4.82,
    with the contrastive loss already at 0.005 against a chance value of 1.099. The task
    was solved and over-optimised, because a batch of four spectra offers four negatives.
    GradCache decouples the batch from memory so the objective can see hundreds.

    If the gradient were merely approximate the whole thing would be worthless, so it is
    checked against a direct full-batch backward -- with dropout OFF, because dropout
    masks are drawn per tensor shape and a chunked forward can never bit-match a
    full-batch one.
    """

    def _config(self):
        from msdelta.configuration_msdelta import MSDeltaConfig
        return MSDeltaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                             intermediate_size=64, delta_bias_n_freqs=8,
                             delta_bias_per_head_hidden=4, hidden_dropout_prob=0.0,
                             attention_probs_dropout_prob=0.0)

    def _model(self, kl_weight=10.0):
        from msdelta.contrastive import MSDeltaForContrastive
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        config = self._config()
        torch.manual_seed(0)
        return MSDeltaForContrastive(MSDeltaForPreTraining(config),
                                     MSDeltaForPreTraining(config),
                                     temperature=0.2, kl_weight=kl_weight).train()

    def _batch(self, size=8, length=12):
        torch.manual_seed(1)
        return {"mz": torch.rand(size, length) * 1000,
                "log_intensity": torch.rand(size, length),
                "attention_mask": torch.ones(size, length, dtype=torch.long),
                "group": torch.arange(size) // 2}

    def _flat_grad(self, model):
        return torch.cat([p.grad.flatten() for _, p in sorted(model.model.named_parameters())
                          if p.grad is not None])

    @pytest.mark.parametrize("chunk_size", [1, 2, 4])
    def test_gradient_matches_full_batch(self, chunk_size):
        from msdelta.contrastive import gradcache_step
        batch = self._batch()
        direct = self._model()
        direct(**batch)["loss"].backward()
        cached = self._model()
        gradcache_step(cached, batch, chunk_size=chunk_size)
        reference, got = self._flat_grad(direct), self._flat_grad(cached)
        # Relative L2 over the whole gradient, not per-parameter max ratio: a bias whose
        # gradient is near zero makes that ratio explode and reports a correct
        # implementation as 27x wrong, which is exactly what happened while writing this.
        assert (reference - got).norm() / reference.norm() < 1e-4

    def test_loss_matches_full_batch(self):
        from msdelta.contrastive import gradcache_step
        batch = self._batch()
        direct = self._model()
        expected = direct(**batch)
        cached = self._model()
        got = gradcache_step(cached, batch, chunk_size=2)
        assert float(got["loss"]) == pytest.approx(float(expected["loss"]), abs=1e-5)
        assert float(got["contrastive"]) == pytest.approx(float(expected["contrastive"]),
                                                          abs=1e-5)

    def test_works_without_the_kl_term(self):
        from msdelta.contrastive import gradcache_step
        batch = self._batch()
        direct = self._model(kl_weight=0.0)
        direct(**batch)["loss"].backward()
        cached = self._model(kl_weight=0.0)
        gradcache_step(cached, batch, chunk_size=2)
        reference, got = self._flat_grad(direct), self._flat_grad(cached)
        assert (reference - got).norm() / reference.norm() < 1e-4

    def test_rng_is_replayed_so_dropout_cannot_desync(self):
        """Without replay the two passes draw different masks and the gradient is wrong.

        Measured before the fix: losses 1.9495 vs 1.9409 and a badly wrong gradient.
        Here dropout is ON, so the check is self-consistency -- two identical GradCache
        steps from the same seed must agree exactly.
        """
        from msdelta.contrastive import MSDeltaForContrastive, gradcache_step
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        from msdelta.configuration_msdelta import MSDeltaConfig
        config = MSDeltaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                               intermediate_size=64, delta_bias_n_freqs=8,
                               delta_bias_per_head_hidden=4, hidden_dropout_prob=0.3)
        batch = self._batch()
        grads = []
        for _ in range(2):
            torch.manual_seed(0)
            model = MSDeltaForContrastive(MSDeltaForPreTraining(config),
                                          MSDeltaForPreTraining(config),
                                          temperature=0.2, kl_weight=10.0).train()
            torch.manual_seed(7)
            gradcache_step(model, batch, chunk_size=2)
            grads.append(self._flat_grad(model))
        assert torch.allclose(grads[0], grads[1], atol=1e-6)


class TestLayerMixPooler:
    """A trained mixture over depth, instead of picking one layer by hand.

    Motivated by job 8841973: the separation ratio peaks mid-stack at every scale and
    sags at the output, so every embedding measured in this repo was read from the wrong
    layer. These tests pin the properties that make the mixture mean what it says.
    """

    def _model(self, layers=4, hidden=32):
        from msdelta.configuration_msdelta import MSDeltaConfig
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        return MSDeltaForPreTraining(MSDeltaConfig(
            hidden_size=hidden, num_hidden_layers=layers,
            num_attention_heads=4, intermediate_size=hidden * 2))

    def _batch(self, batch=2, peaks=10, pad_from=7):
        mz = torch.rand(batch, peaks) * 1000 + 100
        log_intensity = torch.rand(batch, peaks)
        mask = torch.ones(batch, peaks, dtype=torch.bool)
        mask[0, pad_from:] = False
        return mz, log_intensity, mask

    def test_captures_one_state_per_depth_plus_the_embedding(self):
        from msdelta.contrastive import encoder_layer_states
        model = self._model(layers=4)
        states, _ = encoder_layer_states(model.msdelta, *self._batch())
        assert len(states) == 5

    def test_embedding_is_d_model_not_double(self):
        """A sequence mean only. Doubling it would silently change every consumer."""
        from msdelta.contrastive import MSDeltaForContrastive, embedding_size
        model = self._model(hidden=32)
        wrapped = MSDeltaForContrastive(model, pooling="layer_mix", kl_weight=0.0)
        pooled, _ = wrapped.embed(*self._batch())
        assert pooled.shape[-1] == 32 == embedding_size(model, "layer_mix")

    def test_mixture_starts_uniform_and_stays_convex(self):
        from msdelta.contrastive import LayerMixPooler
        weights = LayerMixPooler(5, 8).weights
        assert float(weights.sum()) == pytest.approx(1.0)
        assert float(weights.std()) < 1e-6

    def test_padding_cannot_reach_the_embedding(self, ):
        """(regression) The masked mean is the only thing keeping padded peaks out."""
        from msdelta.contrastive import MSDeltaForContrastive
        wrapped = MSDeltaForContrastive(self._model(), pooling="layer_mix",
                                        kl_weight=0.0).eval()
        mz, log_intensity, mask = self._batch()
        with torch.no_grad():
            before, _ = wrapped.embed(mz, log_intensity, mask)
            noisy = log_intensity.clone(); noisy[0, 7:] = 999.0
            after, _ = wrapped.embed(mz, noisy, mask)
            shifted = mz.clone(); shifted[0, 7:] = 5000.0
            moved, _ = wrapped.embed(shifted, log_intensity, mask)
        assert torch.allclose(before, after, atol=1e-6)
        assert torch.allclose(before, moved, atol=1e-6)

    def test_gradient_reaches_the_mix_and_the_whole_stack(self):
        """The mixture must train the encoder through every depth it draws on."""
        from msdelta.contrastive import MSDeltaForContrastive
        model = self._model()
        wrapped = MSDeltaForContrastive(model, pooling="layer_mix", kl_weight=0.0)
        wrapped.embed(*self._batch())[0].pow(2).sum().backward()
        assert wrapped.layer_mix.mix.grad is not None
        assert bool((wrapped.layer_mix.mix.grad != 0).any())
        first_block = [p.grad for p in model.msdelta.blocks[0].parameters()
                       if p.grad is not None]
        assert first_block and any(bool((g != 0).any()) for g in first_block)

    def test_normalisation_is_what_stops_deep_layers_dominating(self):
        """Without it the 'learned' mixture is decided by magnitude before step one.

        The 200m probe measured in-group distances of 0.0044 at block 2 against 0.0596
        at block 15, so an unnormalised convex sum is the deepest layer with extra steps.
        """
        from msdelta.contrastive import LayerMixPooler
        states = [torch.ones(2, 4, 8) * scale for scale in (0.01, 0.1, 10.0)]
        mask = torch.ones(2, 4, dtype=torch.bool)
        without = LayerMixPooler(3, 8, normalise=False)(states, mask)
        # Uniform weights over 0.01/0.1/10.0 -> 3.37, i.e. the largest layer and nothing
        # else. Normalised, every depth contributes on equal footing.
        assert float(without.mean()) == pytest.approx(3.37, abs=0.01)
        with_norm = LayerMixPooler(3, 8, normalise=True)(states, mask)
        assert abs(float(with_norm.mean())) < 1e-5

    def test_refuses_gradient_checkpointing_rather_than_mishandling_it(self):
        """Under checkpointing each block runs twice and the first output is detached."""
        from msdelta.contrastive import encoder_layer_states
        model = self._model()
        model.msdelta.gradient_checkpointing = True
        model.msdelta.train()
        with pytest.raises(RuntimeError, match="gradient checkpointing"):
            encoder_layer_states(model.msdelta, *self._batch())
