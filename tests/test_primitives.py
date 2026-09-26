"""Small pieces that everything else stands on."""

from __future__ import annotations

import pytest
import torch

from msdelta.fourier import FourierFeatures
from msdelta.reranking import (PAD, POOLING_MODES, PeptideCollator, RESIDUE_TO_ID,
                               VOCAB_SIZE, AlignmentCollator, parse_peptide, peptide_key,
                               pool_sequence, pooled_width, probe, probe_index)


class TestFourierFeatures:
    def test_width_and_finiteness(self):
        features = FourierFeatures(8, 1e-2, 1e3)
        out = features(torch.tensor([[0.0, 1.0, 57.02, 1000.0]]))
        assert out.shape[-1] == features.out_dim
        assert torch.isfinite(out).all()

    def test_deterministic(self):
        features = FourierFeatures(4, 1e-2, 1e3)
        x = torch.tensor([[12.5]])
        assert torch.equal(features(x), features(x))

    def test_distinguishes_nearby_masses(self):
        """A modification encoder that maps 57.02 and 58.02 together is useless."""
        features = FourierFeatures(16, 1e-2, 1e3)
        a, b = features(torch.tensor([[57.0215]])), features(torch.tensor([[58.0055]]))
        assert not torch.allclose(a, b, atol=1e-3)


def _weights(mode, rows, length):
    """Weighted modes need per-peak weights; unweighted ones must not be given any."""
    return torch.rand(rows, length) if mode.startswith("weighted") else None


class TestPooling:
    @pytest.mark.parametrize("mode", POOLING_MODES)
    def test_width_matches_reality(self, mode):
        tokens = torch.randn(2, 5, 32)
        mask = torch.ones(2, 5, dtype=torch.long)
        pooled = pool_sequence(tokens, mask, mode, weights=_weights(mode, 2, 5))
        assert pooled.shape[-1] == pooled_width(32, mode)

    def test_uniform_weights_reproduce_the_plain_mean(self):
        """The weighted path must be a strict generalisation, not a different function."""
        tokens = torch.randn(2, 5, 32)
        mask = torch.ones(2, 5, dtype=torch.long)
        assert torch.allclose(pool_sequence(tokens, mask, "mean"),
                              pool_sequence(tokens, mask, "weighted_mean",
                                            weights=torch.ones(2, 5)), atol=1e-6)

    def test_weighted_modes_reject_missing_weights(self):
        """Silently falling back to an unweighted mean would be an invisible no-op."""
        with pytest.raises(ValueError, match="weights"):
            pool_sequence(torch.randn(1, 4, 8), torch.ones(1, 4, dtype=torch.long),
                          "weighted_mean")

    @pytest.mark.parametrize("mode", POOLING_MODES)
    def test_padding_is_ignored(self, mode):
        """Padded positions must not reach the mean or the max.

        Set the padded tail to something enormous: if it leaks in, max explodes and mean
        moves, and either way the two rows stop matching.
        """
        tokens = torch.randn(1, 3, 8)
        mask = torch.tensor([[1, 1, 0]])
        padded = tokens.clone()
        padded[0, 2] = 1e4
        weights = _weights(mode, 1, 3)
        assert torch.allclose(pool_sequence(tokens, mask, mode, weights=weights),
                              pool_sequence(padded, mask, mode, weights=weights),
                              atol=1e-5)

    def test_unknown_mode_raises(self):
        with pytest.raises((ValueError, KeyError)):
            pool_sequence(torch.randn(1, 2, 4), torch.ones(1, 2, dtype=torch.long), "nope")

    def test_weighting_actually_changes_the_result(self):
        """Guards against weights being accepted and then ignored."""
        tokens = torch.randn(1, 6, 8)
        mask = torch.ones(1, 6, dtype=torch.long)
        skewed = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        assert not torch.allclose(pool_sequence(tokens, mask, "mean"),
                                  pool_sequence(tokens, mask, "weighted_mean",
                                                weights=skewed), atol=1e-3)


class TestParsePeptide:
    def test_plain_residues(self):
        ids, mods = parse_peptide("PEPTIDE")
        assert len(ids) == 7
        assert all(m == 0.0 for m in mods)

    def test_modification_attaches_to_its_residue(self):
        ids, mods = parse_peptide("SAC[57.0215]GVC[57.0215]PGR")
        assert len(ids) == 9
        assert [i for i, m in enumerate(mods) if m] == [2, 5]
        assert mods[2] == pytest.approx(57.0215)

    def test_unknown_residue_stays_in_range(self):
        """An id past the embedding table is an unchecked read on GPU, not an IndexError."""
        ids, _ = parse_peptide("PEPXTIDE")
        assert max(ids) < VOCAB_SIZE and min(ids) >= 0

    def test_empty(self):
        ids, mods = parse_peptide("")
        assert ids == [] and mods == []

    def test_key_is_charge_aware(self):
        assert peptide_key("PEPTIDE", 2) != peptide_key("PEPTIDE", 3)
        assert peptide_key("PEPTIDE", 2, by_charge=False) == \
               peptide_key("PEPTIDE", 3, by_charge=False)


class TestPeptideCollator:
    def test_pads_to_longest_and_masks(self, peptides):
        batch = PeptideCollator()(*peptides)
        assert batch["residues"].shape == batch["sequence_mask"].shape
        assert batch["sequence_mask"][2].sum() == 2          # "MK"
        assert (batch["residues"][2, 2:] == PAD).all()

    def test_truncates_to_max_length(self):
        """A peptide longer than the position table would index past its end."""
        batch = PeptideCollator(max_length=4)(["PEPTIDEPEPTIDE"], [2])
        assert batch["residues"].shape[1] == 4

    def test_every_id_is_in_range(self, peptides):
        batch = PeptideCollator()(*peptides)
        assert int(batch["residues"].max()) < VOCAB_SIZE

    def test_all_residues_are_mapped(self):
        """Every residue the parser can emit must have a table slot."""
        assert max(RESIDUE_TO_ID.values()) < VOCAB_SIZE


class TestAlignmentCollator:
    def _rows(self, with_target=False):
        rows = [{"mz": [100.0, 200.0, 300.0], "log_intensity": [1.0, 2.0, 3.0],
                 "peptide": "PEPTIDE", "charge": 2},
                {"mz": [150.0], "log_intensity": [0.5], "peptide": "MK", "charge": 3}]
        if with_target:
            for row in rows:
                row["target"] = [0.1] * 8
        return rows

    def test_pads_spectra_and_masks(self):
        batch = AlignmentCollator()(self._rows())
        assert batch["mz"].shape == (2, 3)
        assert batch["attention_mask"][1].tolist() == [1, 0, 0]
        assert batch["mz"][1, 1:].eq(0).all()

    def test_passes_through_precomputed_targets(self):
        batch = AlignmentCollator()(self._rows(with_target=True))
        assert batch["target"].shape == (2, 8)

    def test_absent_target_is_absent(self):
        assert "target" not in AlignmentCollator()(self._rows())

    def test_empty_batch_raises(self):
        with pytest.raises(ValueError):
            AlignmentCollator()([])


class TestProbes:
    def test_off_by_default(self, monkeypatch):
        import msdelta.models.peptide_embedder as r
        monkeypatch.setattr(r, "PROBES", False)
        r.probe("x", bad=torch.tensor([float("nan")]))           # must not raise
        r.probe_index("x", "i", torch.tensor([999]), 4)

    def test_catches_non_finite(self, monkeypatch):
        import msdelta.models.peptide_embedder as r
        monkeypatch.setattr(r, "PROBES", True)
        with pytest.raises(RuntimeError, match="non-finite"):
            r.probe("stage", value=torch.tensor([1.0, float("inf")]))

    def test_catches_out_of_range_index(self, monkeypatch):
        import msdelta.models.peptide_embedder as r
        monkeypatch.setattr(r, "PROBES", True)
        with pytest.raises(RuntimeError, match="out of range"):
            r.probe_index("stage", "residues", torch.tensor([0, 23]), 23)

    def test_accepts_valid_index(self, monkeypatch):
        import msdelta.models.peptide_embedder as r
        monkeypatch.setattr(r, "PROBES", True)
        r.probe_index("stage", "residues", torch.tensor([0, 22]), 23)


class TestFixedWidthSpectrumPadding:
    """(regression) Padding to the batch maximum makes memory data-dependent.

    DeltaMZBias is O(batch * width^2), so a batch holding one wide spectrum costs
    several times one holding only narrow ones. Which spectra land together is the
    sampler's choice, so the same model and config reserved 38.75 GB at seed 0 and
    67.14 GB at seed 3, and at 200m/400m the wide draws took a GPU page fault. Every
    seed-0 arm survived; every other seed died, at every scale.
    """

    def _features(self, lengths):
        return [{"mz": [100.0 + i for i in range(n)],
                 "log_intensity": [1.0] * n,
                 "peptide": "PEPTIDE", "charge": 2} for n in lengths]

    def test_default_still_pads_to_the_batch_maximum(self):
        out = AlignmentCollator()(self._features([3, 7, 5]))
        assert out["mz"].shape == (3, 7)

    def test_fixed_width_makes_the_shape_independent_of_the_data(self):
        wide = AlignmentCollator(pad_spectra_to=64)(self._features([3, 7, 5]))
        narrow = AlignmentCollator(pad_spectra_to=64)(self._features([1, 2, 1]))
        assert wide["mz"].shape == narrow["mz"].shape == (3, 64), \
            "a fixed width must not depend on which spectra landed in the batch"

    def test_fixed_width_still_masks_the_padding(self):
        out = AlignmentCollator(pad_spectra_to=16)(self._features([3, 5]))
        assert out["attention_mask"].sum().item() == 8
        assert out["attention_mask"][0, 3:].sum().item() == 0
        assert out["attention_mask"][1, 5:].sum().item() == 0

    def test_fixed_width_preserves_the_values(self):
        out = AlignmentCollator(pad_spectra_to=16)(self._features([3]))
        assert torch.allclose(out["mz"][0, :3],
                              torch.tensor([100.0, 101.0, 102.0]))
        assert out["mz"][0, 3:].abs().sum().item() == 0
