"""K195b-P: the retrieval probe's frozen encoder runs in sub-batches (FROZEN_ENCODER_CHUNK) to bound memory.

What must hold: embeddings and loss are the same as one full-batch pass (both architectures, padded batches whose
size is not a multiple of the chunk), and an unfrozen encoder is untouched (one pass, gradients flow).
"""

from __future__ import annotations

import pytest
import torch

import msdelta.models.modeling_msdelta as mm
from msdelta.models.configuration_msdelta import MSDeltaConfig, MSDeltaRetrievalConfig
from msdelta.models.modeling_msdelta import MSDeltaForRetrieval, MSDeltaModel
from test_eval_mlm import ARCHS, PAIR, TINY


def _batch(n=10, max_peaks=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(3, max_peaks + 1, (n,), generator=g)
    mz = torch.zeros(n, max_peaks); li = torch.zeros(n, max_peaks); am = torch.zeros(n, max_peaks, dtype=torch.long)
    for i, k in enumerate(lengths.tolist()):
        mz[i, :k] = torch.sort(torch.rand(k, generator=g) * 1500 + 100).values
        li[i, :k] = torch.rand(k, generator=g)
        am[i, :k] = 1
    groups = torch.arange(n) // 2
    return dict(mz=mz, log_intensity=li, attention_mask=am, group_ids=groups)


def _model(architecture, frozen):
    torch.manual_seed(0)
    enc_cfg = MSDeltaConfig(**TINY, **(PAIR if architecture == "pairformer" else {}))
    cfg = MSDeltaRetrievalConfig(encoder=enc_cfg, projection_hidden_size=16, embedding_size=8, head_dropout=0.0)
    return MSDeltaForRetrieval(cfg, encoder=MSDeltaModel(enc_cfg), freeze_encoder=frozen).eval()


@pytest.mark.parametrize("architecture", ARCHS)
@pytest.mark.parametrize("chunk", [1, 3, 4])
def test_chunked_frozen_pass_equals_one_pass(monkeypatch, architecture, chunk):
    model, batch = _model(architecture, True), _batch()
    monkeypatch.setattr(mm, "FROZEN_ENCODER_CHUNK", 10_000)
    with torch.no_grad():
        ref = model(**batch, return_dict=True)
    monkeypatch.setattr(mm, "FROZEN_ENCODER_CHUNK", chunk)
    with torch.no_grad():
        got = model(**batch, return_dict=True)
    torch.testing.assert_close(got.embeddings, ref.embeddings, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got.loss, ref.loss, rtol=1e-5, atol=1e-6)


def test_head_still_trains_when_frozen():
    model, batch = _model("transformer", True), _batch()
    model.train()
    model(**batch, return_dict=True).loss.backward()
    assert all(p.grad is None for p in model.msdelta.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.retrieval_head.parameters())


def test_unfrozen_encoder_untouched(monkeypatch):
    model, batch = _model("transformer", False), _batch()
    calls = []
    orig = model.msdelta.forward
    monkeypatch.setattr(model.msdelta, "forward", lambda *a, **k: calls.append(k["mz"].shape[0]) or orig(*a, **k))
    model.train()
    model(**batch, return_dict=True).loss.backward()
    assert calls == [10]
    assert any(p.grad is not None for p in model.msdelta.parameters())
