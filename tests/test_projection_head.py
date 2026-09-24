"""The optional contrastive projection head (--projection_dim): off changes nothing, on
puts the loss through the head, and GradCache stays exact through it."""

import pytest
import torch


def _config():
    from msdelta.configuration_msdelta import MSDeltaConfig
    return MSDeltaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                         intermediate_size=64, delta_bias_n_freqs=8,
                         delta_bias_per_head_hidden=4, hidden_dropout_prob=0.0,
                         attention_probs_dropout_prob=0.0)


def _model(projection_dim=0, kl_weight=10.0):
    from msdelta.contrastive import MSDeltaForContrastive
    from msdelta.modeling_msdelta import MSDeltaForPreTraining
    torch.manual_seed(0)
    encoder, reference = MSDeltaForPreTraining(_config()), MSDeltaForPreTraining(_config())
    torch.manual_seed(1)
    return MSDeltaForContrastive(encoder, reference, temperature=0.2, kl_weight=kl_weight,
                                 projection_dim=projection_dim, projection_hidden=24,
                                 projection_dropout=0.0).train()


def _batch(size=8, length=12):
    torch.manual_seed(2)
    return {"mz": torch.rand(size, length) * 1000,
            "log_intensity": torch.rand(size, length),
            "attention_mask": torch.ones(size, length, dtype=torch.long),
            "group": torch.arange(size) // 2}


def _embed(model, batch):
    return model.embed(batch["mz"], batch["log_intensity"], batch["attention_mask"])[0]


def test_head_off_is_the_old_model():
    """projection_dim 0 must leave the forward exactly as before: no head, no params."""
    model = _model(0)
    assert model.projection is None
    names = [n for n, _ in model.named_parameters()]
    assert not any(n.startswith("projection") for n in names)
    emb = _embed(model, _batch())
    assert emb.shape == (8, 64)                 # mean+max of hidden 32


def test_head_on_changes_width_and_readout_switches_back():
    batch = _batch()
    model = _model(16)
    assert _embed(model, batch).shape == (8, 16)
    model.readout = "pooled"
    pooled = _embed(model, batch)
    assert pooled.shape == (8, 64)
    assert torch.allclose(pooled, _embed(_model(0), batch), atol=1e-6)   # same encoder
    for emb in (_embed(model, batch), pooled):
        assert torch.allclose(emb.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_loss_trains_the_head():
    model = _model(16)
    model(**_batch())["loss"].backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.projection.parameters())


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_gradcache_exact_through_the_head(chunk_size):
    from msdelta.contrastive import gradcache_step
    batch = _batch()
    direct = _model(16)
    direct(**batch)["loss"].backward()
    cached = _model(16)
    gradcache_step(cached, batch, chunk_size=chunk_size)
    flat = lambda m: torch.cat([p.grad.flatten() for _, p in sorted(m.named_parameters())
                                if p.grad is not None])
    ref, got = flat(direct), flat(cached)
    assert (ref - got).norm() / ref.norm() < 1e-4
