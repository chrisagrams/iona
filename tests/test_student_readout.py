"""student_readout(): the loaders (run_rescoring, rerank_psm_embed, eval_align_test,
PeptideEmbedderModel) detect a saved student's readout from its weights.

Training with cls/attn readouts was rejected (PLAN.md A3: keep mean+max pooling); their
forward-pass tests were removed (see tests/README.md). Detection stays: legacy A3 run
directories must still load as what they are, and 'pool' must add no parameters so
every pre-A3 checkpoint loads.
"""

import pytest
import torch

import msdelta.reranking  # noqa: F401  -- module-level msdelta import, see FT33


def _student(readout):
    from msdelta.reranking import PeptideEncoder
    torch.manual_seed(0)
    return PeptideEncoder(embedding_size=16, hidden_size=32, num_layers=2, num_heads=4,
                          dropout=0.0, readout=readout).eval()


def test_pool_is_the_old_encoder():
    """readout='pool' must add no parameters: every pre-A3 checkpoint still loads."""
    names = {n for n, _ in _student("pool").named_parameters()}
    assert not any(n.startswith(("cls", "attn")) for n in names)


@pytest.mark.parametrize("readout", ["pool", "cls", "attn"])
def test_readout_detected_from_weights(readout):
    from msdelta.reranking import student_readout
    state = {f"sequence_encoder.{k}": v for k, v in _student(readout).state_dict().items()}
    assert student_readout(state) == readout


def test_bad_readout_refused():
    with pytest.raises(ValueError):
        _student("max")
