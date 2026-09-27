"""student_readout(): the loaders (run_rescoring, rerank_psm_embed, eval_align_test,
PeptideEmbedderModel) detect a saved student's readout from its weights.

Training with cls/attn readouts was rejected (PLAN.md A3: keep mean+max pooling); their
forward-pass tests are opt-in (marker `legacy`, run with --legacy; see tests/README.md).
Detection stays in the default run: legacy A3 run directories must still load as what
they are, and 'pool' must add no parameters so every pre-A3 checkpoint loads.
"""

import pytest
import torch

import msdelta.reranking  # noqa: F401  -- module-level msdelta import, see FT33


def _student(readout):
    from msdelta.reranking import PeptideEncoder
    torch.manual_seed(0)
    return PeptideEncoder(embedding_size=16, hidden_size=32, num_layers=2, num_heads=4,
                          dropout=0.0, readout=readout).eval()


def _batch(peptides=("PEPTIDEK", "PEPTIEDK", "ACDK"), charges=(2, 2, 3)):
    from msdelta.reranking import PeptideCollator
    return PeptideCollator()(list(peptides), list(charges))


def test_pool_is_the_old_encoder():
    """readout='pool' must add no parameters: every pre-A3 checkpoint still loads."""
    names = {n for n, _ in _student("pool").named_parameters()}
    assert not any(n.startswith(("cls", "attn")) for n in names)


@pytest.mark.legacy  # A3: cls/attn readouts rejected
@pytest.mark.parametrize("readout", ["pool", "cls", "attn"])
def test_shapes_and_unit_norm(readout):
    out = _student(readout)(**_batch())
    assert out.shape == (3, 16)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)


@pytest.mark.legacy  # A3: cls/attn readouts rejected
@pytest.mark.parametrize("readout", ["cls", "attn"])
def test_padding_does_not_leak(readout):
    """The same peptide alone and in a batch with a longer one must embed identically."""
    s = _student(readout)
    alone = s(**_batch(("ACDK",), (3,)))
    batched = s(**_batch(("PEPTIDEKPEPTIDEK", "ACDK"), (2, 3)))[1:]
    assert torch.allclose(alone, batched, atol=1e-5)


@pytest.mark.parametrize("readout", ["pool", "cls", "attn"])
def test_readout_detected_from_weights(readout):
    from msdelta.reranking import student_readout
    state = {f"sequence_encoder.{k}": v for k, v in _student(readout).state_dict().items()}
    assert student_readout(state) == readout


def test_bad_readout_refused():
    with pytest.raises(ValueError):
        _student("max")
