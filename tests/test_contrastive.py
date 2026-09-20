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
