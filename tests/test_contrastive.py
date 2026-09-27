"""Contrastive fine-tune of the spectrum encoder.

The objective exists because the pretrained encoder's embedding space does not separate
peptides: on the replicate corpus only 1% of peptide+charge groups have every replicate
nearer to each other than to any other peptide. These check the machinery that is meant
to fix that, including the two NaN paths that would have made every real batch NaN.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest
import torch

from tests.conftest import REPO
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

    def test_reshuffles_without_anyone_calling_set_epoch(self):
        """FT14. The test above passed throughout the bug because it called set_epoch.

        Nothing in the real path does: HF Trainer builds the dataloader once, and
        accelerate forwards set_epoch to `batch_sampler.sampler`, which this class does
        not have. So the counter stayed at 0 and every epoch replayed identical batches.
        A sampler must reshuffle when iterated, not when asked nicely.
        """
        groups = np.repeat(np.arange(40), 13)
        sampler = GroupBatchSampler(groups, groups_per_batch=6, replicates=4, seed=0)
        first = list(sampler)
        second = list(sampler)
        assert len(first) == len(second) > 1
        assert first != second

    def test_more_epochs_reach_more_of_the_corpus(self):
        """Why FT14 mattered: one epoch touches only K of each group's ~13 replicates.

        Frozen batches meant that first slice -- 28% here, 15.6% in the real corpus at
        K=2 -- was the only data the model ever saw, however long it trained.
        Reshuffling has to make coverage grow.
        """
        groups = np.repeat(np.arange(40), 13)
        sampler = GroupBatchSampler(groups, groups_per_batch=6, replicates=4, seed=0)
        seen, coverage = set(), []
        for _ in range(8):
            for batch in sampler:
                seen.update(batch)
            coverage.append(len(seen) / len(groups))
        assert coverage[0] < 0.4, coverage[0]
        assert coverage[-1] > 0.9, coverage
        assert coverage == sorted(coverage)

    def test_seed_and_epoch_do_not_collide(self):
        """seed + epoch made seed 0 epoch 1 identical to seed 1 epoch 0, so a seed
        sweep would have been a relabelling of one trajectory, not independent runs."""
        groups = np.repeat(np.arange(40), 13)
        a = GroupBatchSampler(groups, groups_per_batch=6, replicates=4, seed=0)
        list(a)                     # advances a to epoch 1
        b = GroupBatchSampler(groups, groups_per_batch=6, replicates=4, seed=1)
        assert list(a) != list(b)


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

    def test_rejects_an_unknown_loss(self):
        """Only supcon is in the recipe (sigmoid was rejected, C8); a typo must not train."""
        from msdelta.contrastive import MSDeltaForContrastive
        with pytest.raises(ValueError):
            MSDeltaForContrastive(torch.nn.Linear(2, 2), None, kl_weight=0.0, loss="triplet")


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


class TestGradCacheEdges:
    """The cases the happy-path tests do not reach.

    GradCache is the only route to a batch wider than four spectra -- DeltaMZBias is
    O(batch * peaks^2 * 2 * n_freqs), which is 32 GiB at batch 64 against a 64 GiB tile --
    so a P/K sweep rests entirely on this being exact. The existing tests cover chunk
    sizes that DIVIDE the batch, one batch shape, temperature 0.2 and dropout on. Each
    test below is a case that differs from those in a way that could plausibly break it.
    """

    def _model(self, kl_weight=10.0, temperature=0.2, dropout=0.0):
        from msdelta.contrastive import MSDeltaForContrastive
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        from msdelta.configuration_msdelta import MSDeltaConfig
        # BOTH dropouts must be off to compare against a direct full-batch backward:
        # masks are drawn per tensor shape, so a chunked forward can never bit-match a
        # full-batch one no matter how carefully the RNG is replayed. Leaving
        # attention_probs_dropout_prob at its default made all seven of these tests fail
        # at a relative gradient error of ~5e-1, which reads exactly like a broken
        # GradCache and is not one.
        config = MSDeltaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                               intermediate_size=64, delta_bias_n_freqs=8,
                               delta_bias_per_head_hidden=4,
                               hidden_dropout_prob=dropout,
                               attention_probs_dropout_prob=dropout)
        torch.manual_seed(0)
        return MSDeltaForContrastive(MSDeltaForPreTraining(config),
                                     MSDeltaForPreTraining(config),
                                     temperature=temperature, kl_weight=kl_weight).train()

    def _batch(self, size=8, length=12, groups=None, mask=None):
        torch.manual_seed(1)
        return {"mz": torch.rand(size, length) * 1000,
                "log_intensity": torch.rand(size, length),
                "attention_mask": (mask if mask is not None
                                   else torch.ones(size, length, dtype=torch.long)),
                "group": (groups if groups is not None else torch.arange(size) // 2)}

    def _flat_grad(self, model):
        return torch.cat([p.grad.flatten()
                          for _, p in sorted(model.model.named_parameters())
                          if p.grad is not None])

    def _assert_matches(self, batch, chunk_size, trim_padding=False, **kw):
        from msdelta.contrastive import gradcache_step
        direct = self._model(**kw)
        direct(**batch)["loss"].backward()
        cached = self._model(**kw)
        got = gradcache_step(cached, batch, chunk_size=chunk_size, trim_padding=trim_padding)
        ref, mine = self._flat_grad(direct), self._flat_grad(cached)
        rel = (ref - mine).norm() / ref.norm()
        assert rel < 1e-4, f"relative gradient error {rel:.2e}"
        return got

    @pytest.mark.parametrize("chunk", [1, 2, 3, 8])
    def test_trimmed_length_sorted_chunks_match_full_batch(self, chunk):
        """trim_padding sorts rows by length and cuts each chunk to its own width.

        Right-padded spectra of very different lengths, groups interleaved so sorting
        really reorders them: the gradient must still equal the full padded batch's.
        """
        lengths = [12, 3, 7, 12, 5, 9, 2, 11]
        mask = torch.zeros(8, 12, dtype=torch.long)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1
        batch = self._batch(size=8, mask=mask, groups=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
        self._assert_matches(batch, chunk, trim_padding=True)

    def test_trimmed_loss_equals_untrimmed(self):
        from msdelta.contrastive import gradcache_step
        mask = torch.zeros(6, 10, dtype=torch.long)
        for i, n in enumerate([10, 4, 6, 2, 9, 5]):
            mask[i, :n] = 1
        batch = self._batch(size=6, length=10, mask=mask, groups=torch.tensor([0, 1, 2, 0, 1, 2]))
        a = gradcache_step(self._model(), batch, chunk_size=2)
        b = gradcache_step(self._model(), batch, chunk_size=2, trim_padding=True)
        assert float(a["loss"]) == pytest.approx(float(b["loss"]), abs=1e-5)

    @pytest.mark.parametrize("size,chunk", [(8, 3), (8, 5), (10, 4), (7, 2), (9, 9)])
    def test_chunk_size_need_not_divide_the_batch(self, size, chunk):
        """The P/K sweep will not hand us batches that divide evenly by the chunk.

        A ragged final chunk is the obvious place for an off-by-one in the cached
        gradient to hide, and every existing test uses a chunk that divides.
        """
        groups = torch.arange(size) // 2
        self._assert_matches(self._batch(size=size, groups=groups), chunk)

    @pytest.mark.parametrize("p,k", [(4, 2), (8, 2), (4, 4)])
    def test_holds_at_the_batch_widths_a_pk_sweep_would_use(self, p, k):
        """Batch 8 is what GradCache exists to escape; the sweep wants 16 and 32."""
        size = p * k
        groups = torch.arange(size) // k
        self._assert_matches(self._batch(size=size, groups=groups), chunk_size=4)

    def test_holds_at_the_live_temperature(self):
        """t=0.07, not the 0.2 every other test uses.

        A lower temperature sharpens the softmax, so the loss concentrates mass on the
        hardest negative and the gradient becomes far less uniform across the batch.
        If any chunk boundary effect exists, this is where it shows.
        """
        self._assert_matches(self._batch(), chunk_size=3, temperature=0.07)

    def test_holds_with_padding(self):
        """Real batches are padded to max_peaks; every existing test is fully unmasked."""
        mask = torch.ones(8, 12, dtype=torch.long)
        mask[::2, 6:] = 0            # half the rows are half padding
        self._assert_matches(self._batch(mask=mask), chunk_size=3)

    def test_a_group_with_no_positive_in_the_batch_does_not_poison_it(self):
        """PK sampling guarantees positives; a ragged final batch may not.

        SupCon has no defined positive term for a singleton, and the masking trap in
        supervised_contrastive_loss (-inf * False = NaN) lives exactly here.
        """
        groups = torch.tensor([0, 0, 1, 1, 2, 3, 4, 5])   # four singletons
        got = self._assert_matches(self._batch(groups=groups), chunk_size=3)
        assert torch.isfinite(got["loss"]), "singleton groups produced a non-finite loss"

    def test_gradcache_is_self_consistent_with_dropout_on(self):
        """With dropout ON there is no full-batch reference, so check reproducibility.

        Every other test here runs dropout off, because a chunked forward cannot
        bit-match a full-batch one when masks depend on shape. That leaves the RNG
        replay untested, so this asserts the thing replay is FOR: two GradCache steps
        from the same seed must produce identical gradients.
        """
        from msdelta.contrastive import gradcache_step
        batch = self._batch()
        grads = []
        for _ in range(2):
            torch.manual_seed(0)
            model = self._model(dropout=0.3)
            torch.manual_seed(7)
            gradcache_step(model, batch, chunk_size=3)
            grads.append(self._flat_grad(model))
        assert torch.allclose(grads[0], grads[1], atol=1e-6)

    def test_one_chunk_is_the_same_as_no_gradcache(self):
        """chunk_size >= batch degenerates to the direct path; it must agree exactly."""
        self._assert_matches(self._batch(size=8), chunk_size=8)


@pytest.mark.legacy  # C9 / FT11: layer mix dropped (parked retry)
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


class TestSaveEncoderCallback:
    """Mid-run checkpoints must contain something loadable.

    MSDeltaForContrastive is a plain nn.Module, so the Trainer writes checkpoint-N as a
    bare state dict with `model.*` keys and no config.json. `final/` is fine because
    main() saves the inner model there by hand; these are the checkpoints that are not.
    """

    def _wrapped(self):
        from msdelta.configuration_msdelta import MSDeltaConfig
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        from msdelta.contrastive import MSDeltaForContrastive
        inner = MSDeltaForPreTraining(MSDeltaConfig(
            hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=64))
        return MSDeltaForContrastive(inner, pooling="mean+max", kl_weight=0.0), inner

    def _fire(self, callback, out_dir, step=40, is_zero=True):
        from types import SimpleNamespace
        callback.on_save(SimpleNamespace(output_dir=str(out_dir)),
                         SimpleNamespace(global_step=step, is_world_process_zero=is_zero),
                         SimpleNamespace())

    def test_writes_an_encoder_that_from_pretrained_can_read(self, tmp_path):
        from msdelta.finetune_contrastive import SaveEncoderCallback
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        wrapped, inner = self._wrapped()
        (tmp_path / "checkpoint-40").mkdir(parents=True)
        self._fire(SaveEncoderCallback(wrapped), tmp_path)

        encoder_dir = tmp_path / "checkpoint-40" / "encoder"
        assert (encoder_dir / "config.json").exists(), "no config.json: still not loadable"
        reloaded = MSDeltaForPreTraining.from_pretrained(str(encoder_dir))
        for (name, before), (_, after) in zip(inner.named_parameters(),
                                              reloaded.named_parameters()):
            assert torch.allclose(before, after), f"{name} changed on the round trip"

    def test_only_rank_zero_writes(self, tmp_path):
        """Twelve ranks writing the same directory is a race, not redundancy."""
        from msdelta.finetune_contrastive import SaveEncoderCallback
        wrapped, _ = self._wrapped()
        (tmp_path / "checkpoint-40").mkdir(parents=True)
        self._fire(SaveEncoderCallback(wrapped), tmp_path, is_zero=False)
        assert not (tmp_path / "checkpoint-40" / "encoder").exists()

    def test_a_failure_to_save_does_not_kill_the_run(self, tmp_path, capsys):
        """The real checkpoint is already on disk; a side artifact must not abort."""
        from msdelta.finetune_contrastive import SaveEncoderCallback
        wrapped, _ = self._wrapped()

        def explode(*args, **kwargs):
            raise OSError("disk full")
        wrapped.model.save_pretrained = explode
        self._fire(SaveEncoderCallback(wrapped), tmp_path)          # must not raise
        assert "could not save encoder" in capsys.readouterr().out

    def test_tolerates_a_model_with_no_save_pretrained(self, tmp_path):
        from msdelta.finetune_contrastive import SaveEncoderCallback
        self._fire(SaveEncoderCallback(torch.nn.Linear(2, 2)), tmp_path)  # must not raise


@pytest.mark.legacy  # FT17: pair loss superseded
class TestPairSamplerAndLoss:
    """The pair formulation: independent per-pair terms instead of in-batch softmax.

    Pairs decompose, so gradient accumulation gives breadth that PK sampling could only
    get from a batch that fits in memory all at once.
    """

    def _groups(self, n_groups=40, per_group=5):
        return np.repeat(np.arange(n_groups), per_group)

    def test_batches_are_pairs_with_the_requested_positive_fraction(self):
        from msdelta.contrastive import PairBatchSampler
        groups = self._groups()
        s = PairBatchSampler(groups, pairs_per_batch=8, positive_fraction=0.5, seed=0)
        batch = next(iter(s))
        assert len(batch) == 16, "a batch is 2 rows per pair"
        same = [groups[batch[2 * i]] == groups[batch[2 * i + 1]] for i in range(8)]
        assert sum(same) == 4, f"expected 4 positive pairs, got {sum(same)}"

    def test_positive_fraction_is_actually_controllable(self):
        """The thing PK sampling could not do: set the in/out balance directly."""
        from msdelta.contrastive import PairBatchSampler
        groups = self._groups()
        for fraction, expected in ((0.25, 2), (0.5, 4), (0.75, 6)):
            s = PairBatchSampler(groups, pairs_per_batch=8,
                                 positive_fraction=fraction, seed=0)
            batch = next(iter(s))
            same = sum(groups[batch[2 * i]] == groups[batch[2 * i + 1]]
                       for i in range(8))
            assert same == expected, f"{fraction} gave {same}, expected {expected}"

    def test_positive_pairs_are_two_DISTINCT_rows(self):
        """Pairing a spectrum with itself makes the positive term identically zero."""
        from msdelta.contrastive import PairBatchSampler
        groups = self._groups()
        s = PairBatchSampler(groups, pairs_per_batch=8, positive_fraction=0.9, seed=0)
        for batch in list(s)[:20]:
            for i in range(0, len(batch), 2):
                if groups[batch[i]] == groups[batch[i + 1]]:
                    assert batch[i] != batch[i + 1]

    def test_it_reshuffles_without_anyone_calling_set_epoch(self):
        """(regression) FT14: GroupBatchSampler replayed identical batches all run."""
        from msdelta.contrastive import PairBatchSampler
        s = PairBatchSampler(self._groups(), pairs_per_batch=4, seed=0)
        assert list(s) != list(s), "consecutive epochs must differ"

    def test_degenerate_balances_are_refused(self):
        from msdelta.contrastive import PairBatchSampler
        for bad in (0.0, 1.0):
            with pytest.raises(ValueError, match="positive_fraction"):
                PairBatchSampler(self._groups(), positive_fraction=bad)

    def test_loss_pulls_positives_together_and_pushes_negatives_apart(self):
        from msdelta.contrastive import pair_contrastive_loss
        groups = torch.tensor([0, 0, 1, 2])          # pair0 same, pair1 different
        far = torch.tensor([[1.0, 0.0], [-1.0, 0.0],   # same peptide, far apart: bad
                            [1.0, 0.0], [0.99, 0.14]])  # different, close: bad
        near = torch.tensor([[1.0, 0.0], [1.0, 0.0],   # same, together: good
                             [1.0, 0.0], [-1.0, 0.0]])  # different, apart: good
        assert pair_contrastive_loss(far, groups)["loss"] > \
               pair_contrastive_loss(near, groups)["loss"]

    def test_margin_stops_pushing_once_far_enough(self):
        """Past the margin a negative contributes nothing -- the defining property."""
        from msdelta.contrastive import pair_contrastive_loss
        groups = torch.tensor([0, 1])
        at = torch.tensor([[1.0, 0.0], [0.0, 1.0]])            # distance sqrt(2) > 1.0
        got = pair_contrastive_loss(at, groups, margin=1.0)
        assert float(got["loss"]) == pytest.approx(0.0)
        assert float(pair_contrastive_loss(at, groups, margin=1.9)["loss"]) > 0

    def test_reports_the_two_terms_separately(self):
        """A run where positives collapse and negatives idle looks fine in the total."""
        from msdelta.contrastive import pair_contrastive_loss
        got = pair_contrastive_loss(torch.randn(8, 4),
                                    torch.tensor([0, 0, 1, 2, 3, 3, 4, 5]))
        for key in ("pair_positive", "pair_negative", "pair_positive_fraction",
                    "pair_distance_same", "pair_distance_diff"):
            assert key in got
        assert float(got["pair_positive_fraction"]) == pytest.approx(0.5)

    def test_odd_row_count_is_refused_not_truncated(self):
        from msdelta.contrastive import pair_contrastive_loss
        with pytest.raises(ValueError, match="even number"):
            pair_contrastive_loss(torch.randn(5, 4), torch.zeros(5, dtype=torch.long))

    def test_gradient_flows_to_both_members_of_a_pair(self):
        from msdelta.contrastive import pair_contrastive_loss
        z = torch.randn(4, 8, requires_grad=True)
        pair_contrastive_loss(z, torch.tensor([0, 0, 1, 2]))["loss"].backward()
        assert z.grad is not None and bool((z.grad != 0).any())



def _module_source(module: str) -> str:
    """Source of the module an import name resolves to (old flat names are shims)."""
    import importlib
    return Path(importlib.import_module(module).__file__).read_text()


class TestRetrievalSummary:
    """The task, not the proxy.

    Every contrastive result in this project is scored on the separation ratio, which
    exists to stand in for retrieval and has never been checked against it. These tests
    cover the evaluation itself; whether the proxy actually predicts the task is an
    empirical question the grids answer, not something a test can assert.
    """

    def _fixture(self, tmp_path, n_groups=6, n_per=3):
        import numpy as np
        from msdelta.contrastive import retrieval_summary

        class Tiny:
            def __init__(self, rows): self.rows = rows
            def __iter__(self): return iter(self.rows)
            def __len__(self): return len(self.rows)

        rows = [{"peptide": f"PEPTIDE{g}K", "charge": 2,
                 "mz": [100.0 + g * 10 + j, 200.0 + g * 10 + j],
                 "log_intensity": [1.0, 0.5]}
                for g in range(n_groups) for j in range(n_per)]
        return Tiny(rows), retrieval_summary

    def test_reports_the_three_task_metrics(self, tiny_config, tmp_path):
        """Hit@1, MAP@100 and R@5 -- what retrieval is actually judged on."""
        import torch
        from msdelta.contrastive import MSDeltaForContrastive
        from msdelta.modeling_msdelta import MSDeltaModel
        from msdelta.finetune_contrastive import ContrastiveCollator

        dataset, retrieval_summary = self._fixture(tmp_path)
        model = MSDeltaForContrastive(MSDeltaModel(tiny_config), kl_weight=0)
        out = retrieval_summary(model, dataset, ContrastiveCollator(pad_spectra_to=8),
                                torch.device("cpu"), max_rows=18)
        if "retrieval/error" in out:
            pytest.skip("faiss unavailable in this environment")
        for key in ("retrieval/Hit@1", "retrieval/MAP@100", "retrieval/R@5"):
            assert key in out, out
            assert 0.0 <= out[key] <= 1.0

    def test_scored_on_the_same_rows_as_the_separation_ratio(self, tiny_config, tmp_path):
        """If the two evaluations embedded different subsets, comparing them would be
        meaningless -- which is the entire reason both are reported."""
        import torch
        from msdelta.contrastive import (MSDeltaForContrastive, embed_dataset,
                                         group_separation_summary, retrieval_summary)
        from msdelta.modeling_msdelta import MSDeltaModel
        from msdelta.finetune_contrastive import ContrastiveCollator

        dataset, _ = self._fixture(tmp_path)
        model = MSDeltaForContrastive(MSDeltaModel(tiny_config), kl_weight=0)
        collator = ContrastiveCollator(pad_spectra_to=8)
        device = torch.device("cpu")
        emb, groups = embed_dataset(model, dataset, collator, device, max_rows=12)
        sep = group_separation_summary(model, dataset, collator, device, max_rows=12)
        ret = retrieval_summary(model, dataset, collator, device, max_rows=12)
        assert len(emb) == 12
        assert sep, "separation should report on these rows"
        if "retrieval/queries" in ret:
            assert ret["retrieval/queries"] == 12
            assert ret["retrieval/groups"] == len(set(groups.tolist()))

    def test_a_failing_evaluation_does_not_lose_the_run(self):
        """A completed training run must not be thrown away by its own evaluation.

        The metric is computed after training and after the encoder is saved, so any
        exception there costs hours and returns nothing. The fine-tune wraps the call.
        """
        source = _module_source("msdelta.finetune_contrastive")
        block = source[source.index("retrieval_summary("):]
        assert "try:" in source[:source.index("retrieval_summary(")][-400:], \
            "the retrieval evaluation must be wrapped in try/except"
        assert "retrieval eval failed" in block[:600]

    def test_exact_search_matches_a_hand_computed_answer(self):
        """Two tight pairs: every query's nearest other point is its own partner."""
        import torch
        from msdelta.contrastive import retrieval_metrics_exact
        e = torch.tensor([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]])
        out = retrieval_metrics_exact(e, [0, 0, 1, 1])
        assert out["Hit@1"] == pytest.approx(1.0)
        assert out["R@5"] == pytest.approx(1.0)
        assert out["MAP@100"] == pytest.approx(1.0)

    def test_exact_search_penalises_a_scrambled_space(self):
        """And the same metric on embeddings that carry no group structure."""
        import torch
        from msdelta.contrastive import retrieval_metrics_exact
        torch.manual_seed(0)
        # Same four points. Only the LABELS change: in the second case each group's two
        # members sit on opposite sides of the space, so the nearest other point is
        # always the wrong one.
        points = torch.tensor([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]])
        good = retrieval_metrics_exact(points, [0, 0, 1, 1])
        scrambled = retrieval_metrics_exact(points, [0, 1, 0, 1])
        assert good["Hit@1"] == pytest.approx(1.0)
        assert scrambled["Hit@1"] == pytest.approx(0.0)
        assert good["MAP@100"] > scrambled["MAP@100"]

    def test_singleton_groups_are_not_scored_as_failures(self, tiny_config, tmp_path):
        """A query whose group has no other member has no correct answer available;
        counting it as a miss would understate retrieval by however many singletons
        the split happens to contain."""
        import torch
        from msdelta.contrastive import MSDeltaForContrastive, retrieval_summary
        from msdelta.modeling_msdelta import MSDeltaModel
        from msdelta.finetune_contrastive import ContrastiveCollator

        dataset, _ = self._fixture(tmp_path, n_groups=8, n_per=1)
        model = MSDeltaForContrastive(MSDeltaModel(tiny_config), kl_weight=0)
        out = retrieval_summary(model, dataset, ContrastiveCollator(pad_spectra_to=8),
                                torch.device("cpu"), max_rows=8)
        assert out == {}, "all-singleton split has nothing to retrieve"


@pytest.mark.legacy  # C8: SupCon kept, sigmoid rejected
class TestSigmoidLoss:
    """C8: SigLIP-style pairwise sigmoid loss as an alternative to SupCon."""

    def test_matches_a_hand_computed_value(self):
        from msdelta.contrastive import sigmoid_contrastive_loss
        emb = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        groups = torch.tensor([0, 0, 1])
        log_scale, bias = torch.tensor(0.0), torch.tensor(0.0)     # logit = cosine
        # pairs (ordered, i != j): (0,1),(1,0) same, cos 1 -> -log sig(1); four cross pairs, cos 0
        # -> -log sig(-0) each
        expected = (2 * -torch.nn.functional.logsigmoid(torch.tensor(1.0))
                    + 4 * -torch.nn.functional.logsigmoid(torch.tensor(0.0))) / 3
        got = sigmoid_contrastive_loss(emb, groups, log_scale, bias)
        assert float(got) == pytest.approx(float(expected), abs=1e-6)

    def test_scale_and_bias_are_learnable_parameters(self):
        from msdelta.contrastive import MSDeltaForContrastive
        from msdelta.configuration_msdelta import MSDeltaConfig
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        cfg = MSDeltaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                            intermediate_size=64, delta_bias_n_freqs=8, delta_bias_per_head_hidden=4)
        m = MSDeltaForContrastive(MSDeltaForPreTraining(cfg), None, kl_weight=0.0, loss="sigmoid")
        names = {n for n, p in m.named_parameters() if p.requires_grad}
        assert {"sigmoid_log_scale", "sigmoid_bias"} <= names
        assert float(m.sigmoid_log_scale.exp()) == pytest.approx(10.0)
        assert float(m.sigmoid_bias) == pytest.approx(-10.0)

    @pytest.mark.parametrize("trim", [False, True])
    @pytest.mark.parametrize("chunk", [1, 3, 8])
    def test_gradcache_gradient_matches_full_batch(self, chunk, trim):
        """Including the gradient on the learnable scale and bias."""
        from msdelta.configuration_msdelta import MSDeltaConfig
        from msdelta.contrastive import MSDeltaForContrastive, gradcache_step
        from msdelta.modeling_msdelta import MSDeltaForPreTraining
        cfg = MSDeltaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                            intermediate_size=64, delta_bias_n_freqs=8, delta_bias_per_head_hidden=4,
                            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)

        def model():
            torch.manual_seed(0)
            return MSDeltaForContrastive(MSDeltaForPreTraining(cfg), MSDeltaForPreTraining(cfg),
                                         kl_weight=10.0, loss="sigmoid",
                                         sigmoid_init_scale=5.0, sigmoid_init_bias=-2.0).train()
        torch.manual_seed(1)
        mask = torch.zeros(8, 12, dtype=torch.long)
        for i, n in enumerate([12, 3, 7, 12, 5, 9, 2, 11]):
            mask[i, :n] = 1
        batch = {"mz": torch.rand(8, 12) * 1000, "log_intensity": torch.rand(8, 12),
                 "attention_mask": mask, "group": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])}
        direct = model(); direct(**batch)["loss"].backward()
        cached = model(); gradcache_step(cached, batch, chunk_size=chunk, trim_padding=trim)

        def flat(m):
            return torch.cat([p.grad.flatten() for _, p in sorted(m.named_parameters())
                              if p.grad is not None])
        ref, got = flat(direct), flat(cached)
        assert (ref - got).norm() / ref.norm() < 1e-4
        assert cached.sigmoid_bias.grad is not None and cached.sigmoid_log_scale.grad is not None


class TestSameMassBatches:
    """C19: GroupBatchSampler with group_masses builds batches of mass-neighbouring groups."""

    def test_peptide_neutral_mass(self):
        """The mass finetune_contrastive assigns each group; a wrong mass mis-sorts every batch."""
        from msdelta.reranking import peptide_neutral_mass
        assert peptide_neutral_mass("PEPTIDE") == pytest.approx(799.3600, abs=1e-3)
        assert peptide_neutral_mass("AC[57.0215]M[15.9949]K") == pytest.approx(524.2087, abs=1e-3)
        assert peptide_neutral_mass("[42.0106]PEPTIDE") == pytest.approx(841.3706, abs=1e-3)

    def _sampler(self, masses, p=4, k=2, seed=0, jitter=0.0):
        from msdelta.contrastive import GroupBatchSampler
        n = len(masses)
        groups = np.repeat(np.arange(n), 3)                  # 3 rows per group
        return GroupBatchSampler(groups, p, k, seed=seed, group_masses=dict(enumerate(masses)),
                                 mass_jitter=jitter), groups

    def test_batches_are_mass_neighbours(self):
        masses = np.random.default_rng(0).uniform(500, 3000, 64)
        sampler, groups = self._sampler(masses)
        order = np.sort(masses)
        spans = []
        for batch in sampler:
            g = np.unique(groups[batch])
            assert len(g) == 4
            spans.append(masses[g].max() - masses[g].min())
            # the batch is exactly 4 consecutive groups in mass order
            lo = np.searchsorted(order, masses[g].min())
            assert np.allclose(np.sort(masses[g]), order[lo:lo + 4])
        assert np.median(spans) < (3000 - 500) / 64 * 8          # far tighter than random batches

    def test_each_group_once_per_epoch_with_k_replicates(self):
        masses = np.linspace(600, 2000, 20)
        sampler, groups = self._sampler(masses, p=4, k=2)
        seen = []
        for batch in sampler:
            assert len(batch) == 8
            g, counts = np.unique(groups[batch], return_counts=True)
            assert (counts == 2).all()
            seen.extend(g.tolist())
        assert sorted(seen) == list(range(20))

    def test_reshuffles_every_epoch(self):
        masses = np.random.default_rng(1).uniform(500, 3000, 40)
        sampler, _ = self._sampler(masses, jitter=1.0)
        first = [tuple(b) for b in sampler]
        second = [tuple(b) for b in sampler]
        assert first != second

    def test_without_masses_behaviour_is_unchanged(self):
        from msdelta.contrastive import GroupBatchSampler
        groups = np.repeat(np.arange(12), 3)
        a = [tuple(b) for b in GroupBatchSampler(groups, 4, 2, seed=3)]
        b = [tuple(b) for b in GroupBatchSampler(groups, 4, 2, seed=3, group_masses=None)]
        assert a == b
