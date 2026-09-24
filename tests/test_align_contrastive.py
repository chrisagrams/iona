"""A4: distinguishable hard negatives and the LiT-style cross-modal contrastive loss."""

import numpy as np
import pytest
import torch

import msdelta.reranking  # noqa: F401  -- module-level msdelta import, see FT33
from msdelta.reranking import hard_negatives, lit_contrastive_loss


def _negs(pep, k=50, seed=0):
    return hard_negatives(pep, np.random.default_rng(seed), k)


def test_never_original_or_reversal_and_cterm_fixed():
    pep = "PEPTIDEWAYK"
    negs = _negs(pep)
    assert negs and pep not in negs
    assert pep[::-1] not in negs and (pep[:-1][::-1] + pep[-1]) not in negs
    assert all(n.endswith("K") and sorted(n) == sorted(pep) for n in negs)


def test_isobaric_swaps_excluded():
    # I/L identical mass, K/Q 0.036 Da < 0.05 Da tolerance: no distinguishable swap
    assert _negs("ILR") == []            # only swap would be I<->L
    assert _negs("KQR") == []
    assert _negs("AAAAK") == []          # identical residues


def test_distinguishable_swap_found():
    negs = _negs("GWK")
    assert negs == ["WGK"]


def test_modifications_travel_with_residue():
    negs = _negs("GM[15.9949]K")
    assert negs == ["M[15.9949]GK"]


def test_lit_loss_prefers_alignment():
    torch.manual_seed(0)
    t = torch.randn(8, 16)
    group = torch.arange(8)
    aligned = lit_contrastive_loss(t, t.clone(), group, temperature=0.1)
    shuffled = lit_contrastive_loss(t, t[torch.randperm(8)], group, temperature=0.1)
    assert aligned < shuffled


def test_hard_negatives_raise_loss_and_masking_ignores_them():
    torch.manual_seed(1)
    t = torch.randn(4, 16); p = t.clone(); group = torch.arange(4)
    close = (t + 0.01 * torch.randn(4, 16))[:, None].repeat(1, 2, 1)   # near-copies
    valid = torch.ones(4, 2, dtype=torch.bool)
    base = lit_contrastive_loss(t, p, group, temperature=0.1)
    with_neg = lit_contrastive_loss(t, p, group, close, valid, temperature=0.1)
    masked = lit_contrastive_loss(t, p, group, close, ~valid, temperature=0.1)
    assert with_neg > base
    assert masked == pytest.approx(float(base), abs=1e-5)


def test_multi_positive_groups():
    t = torch.eye(4); p = torch.eye(4)
    same = torch.tensor([0, 0, 1, 1])
    assert torch.isfinite(lit_contrastive_loss(t, p, same, temperature=0.1))


def test_default_model_loss_is_mse():
    from msdelta.reranking import PeptideCollator, PeptideEncoder, SequenceAlignmentModel
    torch.manual_seed(0)
    student = PeptideEncoder(embedding_size=8, hidden_size=16, num_layers=1, num_heads=2,
                             dropout=0.0)
    model = SequenceAlignmentModel(None, student).eval()
    batch = PeptideCollator()(["PEPTIDEK", "ACDK"], [2, 2])
    target = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    out = model(**batch, target=target)
    expected = ((out["embeddings"] - target) ** 2).sum(-1).mean()
    assert float(out["loss"]) == pytest.approx(float(expected), abs=1e-6)


def test_collator_emits_negatives_and_groups():
    from msdelta.reranking import AlignmentCollator
    c = AlignmentCollator(hard_negatives=3)
    feats = [{"peptide": "PEPTIDEWK", "charge": 2, "mz": [1.0], "log_intensity": [1.0],
              "target": [0.0] * 8},
             {"peptide": "PEPTIDEWK", "charge": 2, "mz": [1.0], "log_intensity": [1.0],
              "target": [0.0] * 8},
             {"peptide": "GWK", "charge": 3, "mz": [1.0], "log_intensity": [1.0],
              "target": [0.0] * 8}]
    b = c(feats)
    assert b["neg_valid"].shape == (3, 3)
    assert b["neg_residues"].shape[0] == 9
    assert b["peptide_group"].tolist() == [0, 0, 1]
    assert b["neg_valid"][2].tolist() == [True, False, False]     # GWK has one swap
