"""K114-P decoupled pair / single streams: ``pair_update_every`` and ``pair_bias_lag``.

Tiny CPU models only. The defaults must reproduce the original Pairformer bit for bit: the
reference is the implementation at commit ``07f53424`` (before K114-P), loaded from git.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import msdelta.models.pairformer as pf
from msdelta.models.configuration_msdelta import MSDeltaConfig
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining

REPO = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "07f53424"  # dev_finetune_02 before K114-P
N_PEAKS = 12
N_LAYERS = 6

UPDATE_MODULES = ("opm", "tri_out", "tri_in", "tri_attn_start", "tri_attn_end", "transition")


def _config(**overrides) -> MSDeltaConfig:
    base = dict(
        architecture="pairformer", hidden_size=32, num_attention_heads=4,
        num_hidden_layers=N_LAYERS, intermediate_size=64, hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0, delta_bias_n_freqs=8, delta_bias_per_head_hidden=4,
        pair_channels=8, pair_tri_channels=8, pair_use_triangle_attention=True,
        pair_tri_attn_heads=2, pair_tri_attn_dim=4, pair_tri_attn_chunk=5, pair_opm_channels=4,
    )
    base.update(overrides)
    return MSDeltaConfig(**base)


def _batch(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    n = N_PEAKS
    mz = torch.sort(torch.rand(2, n, generator=g) * 1500 + 100, dim=1).values
    li = torch.rand(2, n, generator=g)
    am = torch.ones(2, n, dtype=torch.long)
    am[1, n - 4:] = 0
    mz[1, n - 4:] = 0.0
    li[1, n - 4:] = 0.0
    mask_positions = torch.zeros(2, n, dtype=torch.bool)
    mask_positions[:, 1] = True
    mask_positions[0, 5] = True
    labels = torch.rand(2, n, generator=g) * mask_positions
    labels = labels / labels.sum(1, keepdim=True)
    return dict(mz=mz, log_intensity=li, attention_mask=am, mask_positions=mask_positions,
                labels=labels)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return MSDeltaForPreTraining(cfg)


def _n_updates(model) -> int:
    return sum(layer.updates for layer in model.msdelta.bias_module.layers)


# ------------------------------------------------------------------ reference (pre-K114-P) --

@pytest.fixture(scope="module")
def reference_module():
    try:
        src = subprocess.run(
            ["git", "show", f"{REFERENCE_COMMIT}:msdelta/models/pairformer.py"], cwd=REPO,
            check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:  # e.g. a code snapshot without .git
        pytest.skip(f"reference implementation not available from git: {e}")
    name = "msdelta.models._pairformer_reference"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "msdelta.models"
    sys.modules[name] = mod
    exec(compile(src, f"<{REFERENCE_COMMIT}:pairformer.py>", "exec"), mod.__dict__)
    yield mod
    sys.modules.pop(name, None)


def _run(model, batch, checkpointing=False):
    model.train()
    if checkpointing:
        model.gradient_checkpointing_enable()
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.logits.detach(), out.loss.detach(), grads


@pytest.mark.parametrize("overrides", [
    {},
    {"pair_use_triangle_attention": False},
    {"pair_tri_attn_impl": "naive", "pair_tri_attn_checkpoint_chunks": True},
    {"pair_update": "transition", "pair_use_triangle_attention": False},
    {"pair_update": "static", "pair_use_triangle_attention": False},
    {"hidden_dropout_prob": 0.1, "attention_probs_dropout_prob": 0.1, "pair_dropout": 0.1},
])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_defaults_bit_identical_to_reference(reference_module, monkeypatch, overrides,
                                             checkpointing):
    # The reference (07f53424) predates the sdpa_view default (K117); pin the impl both sides know.
    cfg = _config(**{"pair_tri_attn_impl": "sdpa", "pair_writeback_impl": "materialize", **overrides})
    batch = _batch()
    new = _model(cfg)
    torch.manual_seed(123)  # dropout stream
    out_new = _run(new, batch, checkpointing)

    monkeypatch.setattr(pf, "build_pairformer", reference_module.build_pairformer)
    monkeypatch.setattr(pf, "encode_pairformer", reference_module.encode_pairformer)
    ref = _model(cfg)
    assert type(ref.msdelta.bias_module) is reference_module.PairStack
    torch.manual_seed(123)
    out_ref = _run(ref, batch, checkpointing)

    sd_new, sd_ref = new.state_dict(), ref.state_dict()
    assert list(sd_new) == list(sd_ref)
    for k in sd_ref:
        assert torch.equal(sd_new[k], sd_ref[k]), k
    assert torch.equal(out_new[0], out_ref[0])
    assert torch.equal(out_new[1], out_ref[1])
    assert out_new[2].keys() == out_ref[2].keys()
    for k in out_ref[2]:
        assert torch.equal(out_new[2][k], out_ref[2][k]), k
    grid = torch.linspace(-50, 50, 21)
    assert torch.equal(new.msdelta.bias_module.evaluate_layers(grid),
                       ref.msdelta.bias_module.evaluate_layers(grid))


def test_explicit_defaults_equal_implicit():
    a, b = _model(_config()), _model(_config(pair_update_every=1, pair_bias_lag=0))
    assert list(a.state_dict()) == list(b.state_dict())
    batch = _batch()
    assert torch.equal(_run(a, batch)[0], _run(b, batch)[0])


# ---------------------------------------------------------------------------- schedule --

@pytest.mark.parametrize("k,lag,expected", [
    (1, 0, [0, 1, 2, 3, 4, 5]),
    (2, 0, [0, 2, 4]),
    (4, 0, [0, 4]),
    (6, 0, [0]),
    (1, 1, [0, 1, 2, 3, 4]),
    (2, 1, [0, 2]),
    (4, 1, [0]),
])
def test_update_layers_and_modules(k, lag, expected):
    cfg = _config(pair_update_every=k, pair_bias_lag=lag)
    model = _model(cfg)
    layers = model.msdelta.bias_module.layers
    assert [i for i, layer in enumerate(layers) if layer.updates] == expected
    for i, layer in enumerate(layers):
        present = [m for m in UPDATE_MODULES if hasattr(layer, m)]
        assert present == (list(UPDATE_MODULES) if i in expected else []), i
        assert hasattr(layer, "to_bias") and hasattr(layer, "bias_norm")
    keys = set(model.state_dict())
    for i in range(N_LAYERS):
        assert any(k_.startswith(f"msdelta.bias_module.layers.{i}.to_bias") for k_ in keys)
    # Parameter count: the full model minus the skipped layers' update modules.
    full = _model(_config())
    per_update = sum(p.numel() for m in UPDATE_MODULES
                     for p in getattr(full.msdelta.bias_module.layers[0], m).parameters())
    n_full = sum(p.numel() for p in full.parameters())
    assert sum(p.numel() for p in model.parameters()) == n_full - per_update * (
        N_LAYERS - len(expected))


@pytest.mark.parametrize("bad", [
    dict(pair_update_every=0), dict(pair_update_every=N_LAYERS + 1),
    dict(pair_update_every=2.0), dict(pair_bias_lag=2), dict(pair_bias_lag=-1),
    dict(pair_bias_lag=1, pair_update_every=N_LAYERS),
])
def test_invalid_settings_rejected(bad):
    with pytest.raises(ValueError):
        _config(**bad)


def test_transformer_config_ignores_new_fields():
    d = MSDeltaConfig().to_dict()
    assert "pair_update_every" not in d and "pair_bias_lag" not in d


# ------------------------------------------------------------------- training behaviour --

@pytest.mark.parametrize("k,lag", [(2, 0), (4, 0), (6, 0), (1, 1), (2, 1), (4, 1)])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_fwd_bwd_every_parameter_gets_a_grad(k, lag, checkpointing):
    model = _model(_config(pair_update_every=k, pair_bias_lag=lag))
    logits, loss, grads = _run(model, _batch(), checkpointing)
    assert torch.isfinite(loss) and torch.isfinite(logits).all()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and n not in grads]
    assert not missing, missing  # DDP (find_unused_parameters=False) would fail on these
    # Checkpointing is numerically identical to the plain path.
    if checkpointing:
        ref = _run(_model(_config(pair_update_every=k, pair_bias_lag=lag)), _batch())
        torch.testing.assert_close(logits, ref[0], rtol=0, atol=1e-6)
        for n in ref[2]:
            torch.testing.assert_close(grads[n], ref[2][n], rtol=1e-5, atol=1e-6)


def test_k_equals_layers_is_one_update():
    model = _model(_config(pair_update_every=N_LAYERS))
    assert _n_updates(model) == 1
    _run(model, _batch())


def _block_outputs(model, batch):
    outs = []
    hooks = [b.register_forward_hook(lambda m, i, o: outs.append(o.detach().clone()))
             for b in model.msdelta.blocks]
    model.eval()
    with torch.no_grad():
        model(**batch)
    for h in hooks:
        h.remove()
    return outs


def _perturb_update(model, layer_idx):
    """Change what pair update ``layer_idx`` writes (its transition output projection)."""
    with torch.no_grad():
        model.msdelta.bias_module.layers[layer_idx].transition.out.weight.mul_(50.0)


@pytest.mark.parametrize("k", [1, 2])
def test_lag1_dependency(k):
    """lag 1: perturbing update m changes nothing before round m + 1 (the round that reads it)."""
    cfg = _config(pair_update_every=k, pair_bias_lag=1)
    batch = _batch()
    base = _model(cfg)
    ref = _block_outputs(base, batch)
    updates = [i for i, layer in enumerate(base.msdelta.bias_module.layers) if layer.updates]
    for m, layer_idx in enumerate(updates):
        model = _model(cfg)
        _perturb_update(model, layer_idx)
        out = _block_outputs(model, batch)
        first_reader = (m + 1) * k  # first layer of round m + 1
        for i in range(N_LAYERS):
            if i < first_reader:
                assert torch.equal(out[i], ref[i]), (layer_idx, i)
            else:
                assert not torch.equal(out[i], ref[i]), (layer_idx, i)


@pytest.mark.parametrize("k", [1, 2])
def test_lag0_dependency(k):
    """lag 0 (control): update m is read by its own round's single blocks immediately."""
    cfg = _config(pair_update_every=k)
    batch = _batch()
    ref = _block_outputs(_model(cfg), batch)
    for layer_idx in range(0, N_LAYERS, k):
        model = _model(cfg)
        _perturb_update(model, layer_idx)
        out = _block_outputs(model, batch)
        for i in range(N_LAYERS):
            if i < layer_idx:
                assert torch.equal(out[i], ref[i])
            else:
                assert not torch.equal(out[i], ref[i]), (layer_idx, i)


def test_lag1_round_is_independent_of_its_update():
    """lag 1: the single blocks of a round read no output of that round's pair update."""
    cfg = _config(pair_update_every=2, pair_bias_lag=1)
    model = _model(cfg).eval()
    batch = _batch()
    stack = model.msdelta.bias_module
    # Record the z each update receives and its output; the single blocks' bias must be the
    # readout of the round's INPUT z.
    seen = {}

    def hook(i):
        def fn(mod, args, out):
            seen[i] = (args[0].detach(), out[0].detach(), out[1].detach())
        return fn

    hs = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(stack.layers)]
    with torch.no_grad():
        model(**batch)
    for h in hs:
        h.remove()
    for i, layer in enumerate(stack.layers):
        z_in, z_out, bias = seen[i]
        round_start = i - i % 2
        z_round = seen[round_start][0]
        with torch.no_grad():
            expect = layer.read_bias(z_round).permute(0, 3, 1, 2)
            torch.testing.assert_close(bias, expect, rtol=0, atol=1e-6)
            if layer.updates:  # and not from what the update just produced
                fresh = layer.read_bias(z_out).permute(0, 3, 1, 2)
                assert not torch.allclose(bias, fresh, atol=1e-4), i


def test_evaluate_layers_with_skips_and_lag():
    for k, lag in [(2, 0), (3, 0), (2, 1)]:
        model = _model(_config(pair_update_every=k, pair_bias_lag=lag))
        curves = model.msdelta.bias_module.evaluate_layers(torch.linspace(-10, 10, 7))
        assert curves.shape == (N_LAYERS, 7, 4)
        assert torch.isfinite(curves).all()


# ------------------------------------------------------------------------- save / load --

@pytest.mark.parametrize("k,lag", [(1, 0), (3, 0), (2, 1)])
def test_save_load_round_trip_strict(tmp_path, k, lag):
    cfg = _config(pair_update_every=k, pair_bias_lag=lag)
    model = _model(cfg)
    model.save_pretrained(tmp_path)
    loaded = MSDeltaForPreTraining.from_pretrained(tmp_path)  # strict by default
    assert loaded.config.pair_update_every == k and loaded.config.pair_bias_lag == lag
    assert _n_updates(loaded) == _n_updates(model)
    sd, sdl = model.state_dict(), loaded.state_dict()
    assert list(sd) == list(sdl)
    for key in sd:
        assert torch.equal(sd[key], sdl[key]), key
    batch = _batch()
    model.eval()
    loaded.eval()
    with torch.no_grad():
        assert torch.equal(model(**batch).logits, loaded(**batch).logits)


def test_strict_load_refuses_mismatched_k(tmp_path):
    _model(_config(pair_update_every=2)).save_pretrained(tmp_path)
    # A k=2 checkpoint lacks the update modules a k=1 model needs.
    with pytest.raises(Exception):
        MSDeltaForPreTraining.from_pretrained(tmp_path, pair_update_every=1)


# ----------------------------------------------------------------------------- profiler --

def test_profiler_passthrough_and_calibration():
    spec = importlib.util.spec_from_file_location(
        "pairformer_profile_k114p", REPO / "pbs/diag/pairformer_profile.py")
    prof_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prof_mod)
    cfg = prof_mod.load_config("pairformer", str(REPO / prof_mod.PAIRFORMER_CONFIG), "",
                               {"pair_update_every": 5, "pair_bias_lag": 1})
    assert cfg.pair_update_every == 5 and cfg.pair_bias_lag == 1
    small = _config(pair_update_every=3)
    model = _model(small)
    prof = prof_mod.BlockProfiler(torch.device("cpu"))
    prof_mod.instrument(model, prof)
    batch = {k: v for k, v in _batch().items()}
    rec = prof_mod.profiled_mode(model, batch, torch.device("cpu"), prof, 1, 2)
    assert rec["blocks"]["b_trimul_out"]["calls"] == 2
    assert rec["blocks"]["g_bias_readout"]["calls"] == N_LAYERS
    cal = prof_mod.calibration(rec, N_LAYERS, _n_updates(model))
    assert cal["n_pair_updates"] == 2 and cal["pair_ops_total_ms_per_update"] > 0
    prof.remove()
