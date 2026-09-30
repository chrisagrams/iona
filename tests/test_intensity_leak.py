"""K147-P: can a masked peak's intensity reach the model? (audit; notes/K147_intensity_leak_audit.md)

Masked-intensity pretraining asks the model to predict the intensities of the masked peaks, so
nothing the model sees may depend on them. These tests run the REAL input path -- raw
(m/z, intensity) -> ``MSDeltaProcessor`` (log1p + max-normalisation) -> the pretraining
collator (padding + mask sampling) -> ``MSDeltaForPreTraining`` -- on tiny random-init CPU
models in eval mode, for the transformer and the Pairformer, and compare EVERY intermediate
state (each block's single state, each Pairformer pair layer's z and attention bias, the final
hidden state and the intensity logits) bit for bit.

What they establish:
* a masked peak whose intensity stays BELOW the spectrum's base peak cannot change any output
  (both architectures, every Pairformer pair feature switched on);
* the known leak: when the masked peak IS the base peak, the processor's max over all peaks
  (taken before masking) rescales every visible intensity, so outputs change -- xfail(strict)
  until the normalisation is changed; strict so that a fix makes these tests fail loudly and
  get flipped to plain tests;
* the Pairformer's relative-intensity pair feature never contains a masked peak's value;
* no other channel: labels never reach the logits, spectra in a batch do not see each other's
  masked peaks, the mask draw does not depend on intensities, peak order carries nothing.
"""

from __future__ import annotations

import pytest
import torch

from msdelta.models.configuration_msdelta import MSDeltaConfig
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining
from msdelta.models.pairformer import PairFeatures
from msdelta.models.processing_msdelta import (MSDeltaDataCollatorForPreTraining,
                                               MSDeltaProcessor)

MASK_RATIO = 0.5  # production value (configs/msdelta-base-50m, K96-S)
LENGTHS = (23, 17, 9)  # three spectra of different lengths: padding is always exercised

_TINY = dict(hidden_size=32, num_attention_heads=4, num_hidden_layers=2, intermediate_size=64,
             hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
             delta_bias_n_freqs=8, delta_bias_per_head_hidden=4)
_TINY_PAIR = dict(pair_channels=8, pair_tri_channels=8, pair_tri_attn_heads=2,
                  pair_tri_attn_dim=4, pair_tri_attn_chunk=5, pair_opm_channels=4,
                  pair_mass_defect_n_freqs=4)
CONFIGS = {
    "transformer": dict(_TINY),
    # Every pair feature and every pair-update path on (mass defect and triangle attention are
    # off by default; switched on here so they are covered too).
    "pairformer_all": dict(_TINY, **_TINY_PAIR, architecture="pairformer",
                           pair_use_mass_defect=True, pair_use_triangle_attention=True),
    # The feature flags of configs/stage0/pairformer/config.json (intensity, loss bank,
    # isotope, write-back, triangle multiplication; no mass defect, no triangle attention).
    "pairformer_stage0": dict(_TINY, **_TINY_PAIR, architecture="pairformer",
                              pair_use_mass_defect=False, pair_use_triangle_attention=False),
}
ARCHS = list(CONFIGS)


# ----------------------------------------------------------------------------- helpers

def _model(arch: str) -> MSDeltaForPreTraining:
    model = MSDeltaForPreTraining(MSDeltaConfig(**CONFIGS[arch])).eval()
    # Randomise EVERY parameter: at init some paths are exact no-ops (the Pairformer's
    # write-back projection and the intensity stand-in start at zero), which would hide a
    # leak through them.
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return model


def _raw_spectra(seed: int = 0) -> list[dict[str, torch.Tensor]]:
    """Raw spectra on the real data's scale: sorted m/z, absolute intensities ~1e2..1e6."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for n in LENGTHS:
        mz = torch.sort(torch.rand(n, generator=g) * 1400 + 100).values
        intensity = torch.exp(torch.randn(n, generator=g) * 1.5 + 9.0)
        out.append({"mz": mz, "intensity": intensity})
    return out


def _collate(raw: list[dict[str, torch.Tensor]], mask_seed: int = 7) -> dict[str, torch.Tensor]:
    """The pretraining input path: processor per spectrum, then the collator (fixed RNG)."""
    processor = MSDeltaProcessor(max_peaks=150)
    features = [processor(r["mz"], r["intensity"], padding=False, return_labels=True)
                for r in raw]
    with torch.random.fork_rng():
        torch.manual_seed(mask_seed)
        return MSDeltaDataCollatorForPreTraining(mask_ratio=MASK_RATIO)(features)


def _trace(model: MSDeltaForPreTraining, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Every intermediate state: per-block single state, per-pair-layer (z, bias), final."""
    states: dict[str, torch.Tensor] = {}
    hooks = []
    enc = model.msdelta
    for i, block in enumerate(enc.blocks):
        hooks.append(block.register_forward_hook(
            lambda _m, _i, out, i=i: states.__setitem__(f"block{i}", out.detach().clone())))
    if getattr(enc.config, "architecture", "transformer") == "pairformer":
        for i, layer in enumerate(enc.bias_module.layers):
            def hook(_m, _i, out, i=i):
                states[f"pair{i}.z"] = out[0].detach().clone()
                states[f"pair{i}.bias"] = out[1].detach().clone()
            hooks.append(layer.register_forward_hook(hook))
    else:
        hooks.append(enc.bias_module.register_forward_hook(
            lambda _m, _i, out: states.__setitem__("delta_bias", out.detach().clone())))
    try:
        with torch.no_grad():
            out = model(mz=batch["mz"], log_intensity=batch["log_intensity"],
                        attention_mask=batch["attention_mask"],
                        mask_positions=batch["mask_positions"], return_dict=True)
            hidden = enc(mz=batch["mz"], log_intensity=batch["log_intensity"],
                         attention_mask=batch["attention_mask"],
                         mask_positions=batch["mask_positions"]).last_hidden_state
    finally:
        for h in hooks:
            h.remove()
    states["last_hidden_state"] = hidden
    states["logits"] = out.logits
    return states


def _identical(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[str]:
    """Names of states that differ (bit-exact comparison; NaN-free by construction)."""
    assert a.keys() == b.keys()
    return [k for k in a if not torch.equal(a[k], b[k])]


def _masked_non_max(batch, raw) -> list[tuple[int, int]]:
    """(row, peak) pairs that are masked and strictly below their spectrum's base peak."""
    picks = []
    for row, r in enumerate(raw):
        n = r["intensity"].numel()
        top = r["intensity"].max()
        for k in range(n):
            if batch["mask_positions"][row, k] and r["intensity"][k] < top:
                picks.append((row, k))
    return picks


def _masked_max(batch, raw) -> list[tuple[int, int]]:
    picks = []
    for row, r in enumerate(raw):
        k = int(r["intensity"].argmax())
        if batch["mask_positions"][row, k]:
            picks.append((row, k))
    return picks


def _raw_with_max_masked(mask_seed: int = 7):
    """Raw spectra plus the batch, with row 0's base peak guaranteed to be masked."""
    raw = _raw_spectra()
    batch = _collate(raw, mask_seed)
    row0_masked = batch["mask_positions"][0, :LENGTHS[0]].nonzero().flatten()
    k = int(row0_masked[0])
    raw[0]["intensity"][k] = raw[0]["intensity"].max() * 3.0  # make a masked peak the max
    return raw, _collate(raw, mask_seed), k


# ----------------------------------------------------------------------------- (a) no leak

@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("factor", [0.0, 1e-3, 0.5, 0.999])
def test_masked_peak_below_max_does_not_change_any_output(arch, factor):
    """Moving every masked non-base peak anywhere in [0, base peak) changes nothing at all."""
    model = _model(arch)
    raw = _raw_spectra()
    batch = _collate(raw)
    picks = _masked_non_max(batch, raw)
    assert len(picks) >= 5
    changed = [{k: v.clone() for k, v in r.items()} for r in raw]
    for row, k in picks:
        top = changed[row]["intensity"].max()
        changed[row]["intensity"][k] = factor * top
    batch2 = _collate(changed)

    assert torch.equal(batch["mask_positions"], batch2["mask_positions"])
    visible = batch["attention_mask"].bool() & ~batch["mask_positions"]
    assert torch.equal(batch["log_intensity"][visible], batch2["log_intensity"][visible])
    assert not torch.equal(batch["labels"], batch2["labels"])  # the targets did change
    assert _identical(_trace(model, batch), _trace(model, batch2)) == []


@pytest.mark.parametrize("arch", ARCHS)
def test_masked_input_values_are_ignored_even_if_arbitrary(arch):
    """Model-level: any value in a masked slot of ``log_intensity`` (even > 1) is ignored."""
    model = _model(arch)
    batch = _collate(_raw_spectra())
    changed = dict(batch)
    li = batch["log_intensity"].clone()
    li[batch["mask_positions"]] = torch.linspace(-5.0, 50.0, int(batch["mask_positions"].sum()))
    changed["log_intensity"] = li
    assert _identical(_trace(model, batch), _trace(model, changed)) == []


# ----------------------------------------------------------------------------- (b) max leak

def test_masked_base_peak_rescales_visible_inputs():
    """Mechanism of the known leak (documented, passes): the max is taken before masking."""
    before = _collate(_raw_spectra())
    _, after, k = _raw_with_max_masked()
    assert after["mask_positions"][0, k]
    vis = after["attention_mask"][0].bool() & ~after["mask_positions"][0]
    # Base peak visible: the base peak's x is exactly 1.0 ...
    assert before["log_intensity"][0].max() == 1.0
    # ... base peak masked: no visible x reaches 1.0 and every visible value is rescaled.
    assert after["log_intensity"][0][vis].max() < 1.0
    assert (after["log_intensity"][0][vis] < before["log_intensity"][0][vis]).all()


@pytest.mark.xfail(strict=True, reason=(
    "K147-P known leak: MSDeltaProcessor normalises log1p(I) by the max over ALL peaks before "
    "the collator masks; when the base peak is masked every visible input is rescaled (and no "
    "visible x == 1). Flip to a plain test once normalisation uses visible peaks only."))
@pytest.mark.parametrize("arch", ARCHS)
def test_masked_base_peak_does_not_change_outputs(arch):
    model = _model(arch)
    raw, batch, k = _raw_with_max_masked()
    bigger = [{kk: v.clone() for kk, v in r.items()} for r in raw]
    bigger[0]["intensity"][k] *= 10.0  # the masked base peak grows further
    batch2 = _collate(bigger)
    assert torch.equal(batch["mask_positions"], batch2["mask_positions"])
    assert _identical(_trace(model, batch), _trace(model, batch2)) == []


# ----------------------------------------------------------------------------- (c) pair feature

def test_pair_relative_intensity_ignores_masked_peaks():
    """Every pair feature column, and in particular relative intensity, ignores masked values."""
    config = MSDeltaConfig(**CONFIGS["pairformer_all"])
    feats_mod = PairFeatures(config)
    with torch.no_grad():
        feats_mod.mask_log_intensity.fill_(0.37)
    batch = _collate(_raw_spectra())
    mp = batch["mask_positions"]
    li2 = batch["log_intensity"].clone()
    li2[mp] = torch.rand(int(mp.sum())) * 10
    with torch.no_grad():
        f1 = feats_mod(batch["mz"], batch["log_intensity"], mp)
        f2 = feats_mod(batch["mz"], li2, mp)
    assert torch.equal(f1, f2)
    # The intensity column is last: visible_i - visible_j with the stand-in for masked peaks.
    rel = f1[..., -1]
    vis = torch.where(mp, torch.tensor(0.37), batch["log_intensity"])
    assert torch.equal(rel, vis.unsqueeze(-1) - vis.unsqueeze(-2))
    # A pair of two masked peaks is exactly 0; masked-vs-visible is stand-in minus visible.
    i, j = mp[0].nonzero().flatten()[:2].tolist()
    assert rel[0, i, j] == 0


def test_pair_features_leak_through_the_max_only_via_visible_rescaling():
    """Under the max leak, pairs of two VISIBLE peaks change; masked peaks' own values never
    enter (their column is stand-in minus visible). Documents where (b) enters the pair path."""
    config = MSDeltaConfig(**CONFIGS["pairformer_all"])
    feats_mod = PairFeatures(config)
    raw, batch, k = _raw_with_max_masked()
    bigger = [{kk: v.clone() for kk, v in r.items()} for r in raw]
    bigger[0]["intensity"][k] *= 10.0
    batch2 = _collate(bigger)
    with torch.no_grad():
        rel1 = feats_mod(batch["mz"], batch["log_intensity"], batch["mask_positions"])[..., -1]
        rel2 = feats_mod(batch2["mz"], batch2["log_intensity"], batch2["mask_positions"])[..., -1]
    vis = batch["attention_mask"][0].bool() & ~batch["mask_positions"][0]
    vv = vis.unsqueeze(-1) & vis.unsqueeze(-2)
    assert not torch.equal(rel1[0][vv], rel2[0][vv])      # visible pairs rescaled: the leak
    assert torch.equal(rel1[1:], rel2[1:])               # other spectra untouched


# ----------------------------------------------------------------------------- (d) other paths

@pytest.mark.parametrize("arch", ARCHS)
def test_labels_do_not_reach_the_logits(arch):
    model = _model(arch)
    batch = _collate(_raw_spectra())
    common = {k: batch[k] for k in ("mz", "log_intensity", "attention_mask", "mask_positions")}
    with torch.no_grad():
        a = model(**common, labels=batch["labels"]).logits
        b = model(**common, labels=torch.rand_like(batch["labels"])).logits
        c = model(**common).logits
    assert torch.equal(a, b) and torch.equal(a, c)


@pytest.mark.parametrize("arch", ARCHS)
def test_other_spectra_in_the_batch_never_see_a_masked_base_peak(arch):
    """Batch-level statistics would be a cross-spectrum leak; even the max leak stays local."""
    model = _model(arch)
    raw, batch, k = _raw_with_max_masked()
    bigger = [{kk: v.clone() for kk, v in r.items()} for r in raw]
    bigger[0]["intensity"][k] *= 10.0
    t1, t2 = _trace(model, batch), _trace(model, _collate(bigger))
    for name in t1:
        assert torch.equal(t1[name][1:], t2[name][1:]), name


def test_mask_draw_does_not_depend_on_intensity():
    raw = _raw_spectra()
    flat = [{"mz": r["mz"], "intensity": torch.ones_like(r["intensity"])} for r in raw]
    assert torch.equal(_collate(raw)["mask_positions"], _collate(flat)["mask_positions"])
    # Mask count is round(ratio * n) per spectrum: a function of the peak count only.
    counts = _collate(raw)["mask_positions"].sum(1).tolist()
    assert counts == [max(1, round(n * MASK_RATIO)) for n in LENGTHS]


@pytest.mark.parametrize("arch", ARCHS)
def test_peak_order_carries_no_information(arch):
    """No positional encoding: permuting peaks permutes the outputs (so an intensity-sorted
    peak order could not leak). Float summation order differs, hence allclose here."""
    model = _model(arch)
    batch = _collate(_raw_spectra())
    n = LENGTHS[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(3))
    full = torch.arange(batch["mz"].shape[1])
    full[:n] = perm
    permuted = {k: v[:, full] if v.ndim == 2 else v for k, v in batch.items()}
    a, b = _trace(model, batch), _trace(model, permuted)
    assert torch.allclose(a["logits"][0, :n][perm], b["logits"][0, :n], atol=1e-4)
