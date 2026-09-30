"""K172-P ``pair_concurrent``: pair update m on a side stream, concurrent with round m.

CPU only. On CPU the concurrent path has no streams, so it must be the sequential lag-1
computation bit for bit; the stream orchestration itself is checked with a recording stub.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

import msdelta.models.pairformer as pf
from msdelta.models.configuration_msdelta import MSDeltaConfig
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining

N_PEAKS = 12
N_LAYERS = 10


def _config(**overrides) -> MSDeltaConfig:
    base = dict(
        architecture="pairformer", hidden_size=32, num_attention_heads=4,
        num_hidden_layers=N_LAYERS, intermediate_size=64, hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0, delta_bias_n_freqs=8, delta_bias_per_head_hidden=4,
        pair_channels=8, pair_tri_channels=8, pair_use_triangle_attention=True,
        pair_tri_attn_heads=2, pair_tri_attn_dim=4, pair_tri_attn_chunk=5, pair_opm_channels=4,
        pair_bias_lag=1,
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
    model = MSDeltaForPreTraining(cfg)
    # The write-back's output projection starts at zero; perturb it so the s -> z path (and
    # its gradient through the side stream's update) is exercised.
    with torch.no_grad():
        for layer in model.msdelta.bias_module.layers:
            if hasattr(layer, "opm"):
                layer.opm.out.weight.normal_(0.0, 0.05)
    return model


def _run(model, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(123)  # dropout stream
    out = model(**batch)
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.logits.detach(), out.loss.detach(), grads


def _assert_same(a, b):
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    assert a[2].keys() == b[2].keys()
    for k in a[2]:
        assert torch.equal(a[2][k], b[2][k]), k


# --------------------------------------------------------------------------- numerics --

@pytest.mark.parametrize("k", [2, 5])
@pytest.mark.parametrize("overrides", [
    {},
    {"pair_use_triangle_attention": False, "pair_tri_mul": "outgoing"},
    {"hidden_dropout_prob": 0.1, "attention_probs_dropout_prob": 0.1, "pair_dropout": 0.1},
])
def test_cpu_concurrent_bit_identical_to_sequential_lag1(k, overrides):
    batch = _batch()
    seq = _model(_config(pair_update_every=k, **overrides))
    con = _model(_config(pair_update_every=k, pair_concurrent=True, **overrides))
    assert con.msdelta.bias_module.concurrent and not seq.msdelta.bias_module.concurrent
    assert list(seq.state_dict()) == list(con.state_dict())
    for key, v in seq.state_dict().items():
        assert torch.equal(v, con.state_dict()[key]), key
    out_seq, out_con = _run(seq, batch), _run(con, batch)
    _assert_same(out_seq, out_con)
    # Every parameter still gets a gradient (DDP without find_unused_parameters).
    missing = [n for n, p in con.named_parameters() if p.requires_grad and n not in out_con[2]]
    assert not missing, missing


def test_cpu_concurrent_eval_no_grad():
    batch = _batch()
    seq, con = (_model(_config(pair_update_every=3, pair_concurrent=c)).eval()
                for c in (False, True))
    with torch.no_grad():
        assert torch.equal(seq(**batch).logits, con(**batch).logits)


def test_concurrent_refuses_gradient_checkpointing():
    model = _model(_config(pair_update_every=2, pair_concurrent=True))
    model.gradient_checkpointing_enable()
    model.train()
    with pytest.raises(NotImplementedError, match="gradient checkpointing"):
        model(**_batch())


# ----------------------------------------------------------------------------- config --

@pytest.mark.parametrize("bad", [
    dict(pair_concurrent=True, pair_bias_lag=0),
    dict(pair_concurrent=1),
    dict(pair_concurrent="yes"),
])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        _config(**bad)


def test_config_default_and_round_trip(tmp_path):
    assert _config().pair_concurrent is False
    assert "pair_concurrent" not in MSDeltaConfig().to_dict()  # transformer drops pair_*
    model = _model(_config(pair_update_every=2, pair_concurrent=True))
    model.save_pretrained(tmp_path)
    loaded = MSDeltaForPreTraining.from_pretrained(tmp_path)
    assert loaded.config.pair_concurrent is True


# ------------------------------------------------------------------ stream orchestration --

class RecordingStreams(pf.NullStreams):
    """Stub streams: log every primitive, and which stream is current, in order."""

    def __init__(self):
        self.log: list[tuple] = []
        self.current = "main"
        self.n_events = 0

    def main(self):
        return "main"

    def side(self):
        return "side"

    def use(self, stream):
        streams = self

        class _Ctx:
            def __enter__(self):
                streams.log.append(("enter", stream))
                self.prev, streams.current = streams.current, stream

            def __exit__(self, *exc):
                streams.current = self.prev
                streams.log.append(("exit", stream))

        return _Ctx()

    def event(self, stream):
        self.n_events += 1
        name = f"e{self.n_events}"
        self.log.append(("event", stream, name))
        return name

    def wait(self, stream, event):
        self.log.append(("wait", stream, event))

    def keep(self, tensor, stream):
        self.log.append(("keep", id(tensor), stream))


def _instrument(model, streams):
    """Log each update / single block with the stream current when it is called."""
    stack = model.msdelta.bias_module
    handles = []
    for i, layer in enumerate(stack.layers):
        orig = layer.update

        def update(z, s, mask, pair_mask, i=i, orig=orig):
            out = orig(z, s, mask, pair_mask)
            streams.log.append(("update", i, streams.current, id(z), id(s), id(out)))
            return out

        layer.update = update
    for i, block in enumerate(model.msdelta.blocks):
        handles.append(block.register_forward_hook(
            lambda m, args, out, i=i: streams.log.append(("block", i, streams.current))))
    return handles


@pytest.mark.parametrize("k", [2, 3, 5])
def test_stream_orchestration_order(monkeypatch, k):
    cfg = _config(pair_update_every=k, pair_concurrent=True)
    model = _model(cfg)
    streams = RecordingStreams()
    monkeypatch.setattr(pf, "device_streams", lambda device: streams)
    _instrument(model, streams)
    batch = _batch()
    out = _run(model, batch)
    _assert_same(out, _run(_model(_config(pair_update_every=k)), batch))

    log = streams.log
    rounds = [list(range(a, min(a + k, N_LAYERS))) for a in range(0, N_LAYERS, k)]
    pos = 0
    pending_join = None
    for m, layers in enumerate(rounds):
        a = layers[0]
        if pending_join is not None:  # join before the next round
            assert log[pos] == ("wait", "main", pending_join), (m, log[pos])
            pos += 1
            pending_join = None
        if a + k < N_LAYERS:  # the round builds an update: fork
            ev, st, ready = log[pos]
            assert (ev, st) == ("event", "main"), (m, log[pos])
            assert log[pos + 1] == ("wait", "side", ready)
            assert log[pos + 2] == ("enter", "side")
            upd = log[pos + 3]
            assert upd[:3] == ("update", a, "side")
            z_in, s_in, z_out = upd[3:]
            assert log[pos + 4] == ("exit", "side")
            ev, st, join = log[pos + 5]
            assert (ev, st) == ("event", "side")
            # record_stream on everything crossing streams: inputs -> side, output -> main
            keeps = log[pos + 6: pos + 11]
            assert all(e[0] == "keep" for e in keeps), keeps
            assert ("keep", z_in, "side") in keeps and ("keep", s_in, "side") in keeps
            assert keeps[-1] == ("keep", z_out, "main")
            assert sum(1 for e in keeps if e[2] == "side") == 4  # z, s, mask, pair_mask
            pos += 11
            pending_join = join
        for i in layers:  # the round's single blocks, on main
            assert log[pos] == ("block", i, "main"), (m, i, log[pos])
            pos += 1
    assert pending_join is None  # the last round built no update, so nothing to join
    assert pos == len(log), log[pos:]
    n_updates = sum(1 for e in log if e[0] == "update")
    assert n_updates == len(rounds) - 1


def test_device_streams_selection():
    assert type(pf.device_streams(torch.device("cpu"))) is pf.NullStreams
    for dev in ("xpu", "cuda"):  # constructing touches no device, so this runs on CPU too
        if hasattr(torch, dev):
            assert type(pf.device_streams(torch.device(dev, 0))) is pf.DeviceStreams


def test_device_streams_primitives_with_fake_backend():
    """``DeviceStreams`` maps the interface onto the backend API (as torch.cuda / torch.xpu)."""
    calls = []

    class FakeStream:
        def __init__(self, device=None):
            calls.append(("Stream", device))

        def wait_event(self, event):
            calls.append(("wait_event", self, event))

    class FakeEvent:
        def record(self, stream):
            calls.append(("record", self, stream))

    class Backend:
        Stream, Event = FakeStream, FakeEvent

        @staticmethod
        def current_stream(device):
            return "cur"

        @staticmethod
        def stream(s):
            calls.append(("stream", s))
            return contextlib.nullcontext()

    dev = torch.device("xpu", 7)  # a device index nothing else uses (cache is per device)
    pf.DeviceStreams._side.pop(dev, None)
    ds = pf.DeviceStreams(Backend, dev)
    assert ds.main() == "cur"
    side = ds.side()
    assert ds.side() is side and calls.count(("Stream", dev)) == 1  # cached
    ev = ds.event(side)
    assert calls[-1] == ("record", ev, side)
    ds.wait(side, ev)
    assert calls[-1] == ("wait_event", side, ev)
    with ds.use(side):
        pass
    assert calls[-1] == ("stream", side)

    class T:
        def record_stream(self, s):
            calls.append(("record_stream", s))

    ds.keep(T(), side)
    assert calls[-1] == ("record_stream", side)
    pf.DeviceStreams._side.pop(dev, None)
