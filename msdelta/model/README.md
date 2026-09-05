# `msdelta/model/` — the architecture

Everything that defines what MSDelta *is*. **This package imports nothing from `data/`,
`train/`, or `eval/`** — keep it that way, it is what makes the model loadable standalone from a
checkpoint.

* **[MODULES.md](MODULES.md) — every module in `modeling.py`**, one by one: diagrams, tensor
  shapes, measured parameter counts, and the change-impact table. Start here to read the code.
* [../../docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) — the design rationale and the
  experiment recipes.

| file | contents |
| --- | --- |
| `configuration.py` | `MSDeltaConfig` (15 architecture fields + `_validate()`), `MSDeltaDenoisingConfig` (composes an encoder config with a head) |
| `modeling.py` | `PeakEmbed`, `DeltaMZBias`, `BiasedMHA`, `EncoderBlock`, `MSDeltaModel`, `IntensityHead`, `PeakDenoisingHead`, `MSDeltaForPreTraining`, `MSDeltaForDenoising` — walked through in [MODULES.md](MODULES.md) |
| `fourier.py` | `FourierFeatures` plus the frequency-health metrics `interp_mae`, `dead_freqs`, `freq_drift` |
| `experimental.py` | ⚠ untracked scratch variant — **does not run as written**, see below |

## The one-sentence version

A peak token carries **only its log-intensity**; all m/z information enters as a learned per-head
additive attention bias over the signed pairwise mass difference `Δ = m/z_i − m/z_j`, computed
once and shared by every layer.

```
log_intensity ──► FourierFeatures(16) ──► MLP ──► tokens ─┐
                                                          ├─► EncoderBlock × L ──► LayerNorm ──► heads
mz ──► Δ = mzᵢ−mzⱼ ──► FourierFeatures(64) ──► per-head ──┘
                                               MLP → bias (B, heads, K, K)
```

## Two things that will bite you

**1. The relative imports in `modeling.py` and `experimental.py` are load-bearing.**

```python
from .configuration import MSDeltaConfig, MSDeltaDenoisingConfig
from .fourier import FourierFeatures
```

Hugging Face's `custom_object_save` follows relative imports to copy dependent modules into a
saved checkpoint, so a checkpoint stays loadable with `trust_remote_code=True` on a machine
without `msdelta` installed. `pyproject.toml` has a `TID252` per-file ignore for these two files.
Converting them to absolute imports would silently produce broken checkpoints.

**2. `experimental.py` does not run.** Two independent defects:

* It reads `config.delta_bias_scale`, a field deleted from `MSDeltaConfig` in commit `7396e38`.
  Constructing any model from it raises `AttributeError`.
* `MSDeltaModel_2`'s block loop never reassigns `hidden_states`, so every block receives the
  embedding output and all but the last result is discarded — effectively a 1-layer model with
  `L−1` blocks computing dead gradients.

It is also **not re-exported from `msdelta/model/__init__.py`**, deliberately: it re-declares
`MSDeltaModel` / `MSDeltaForPreTraining` and calls `register_for_auto_class` on them, so importing
it re-registers the HF auto-classes against different class objects. Import it explicitly, and
only if you mean to. Diagnosis and fixes:
[ARCHITECTURE.md §7](../../docs/ARCHITECTURE.md#7-the-scratch-variant-modelexperimentalpy).

## Changing the architecture

Recipes with exact edit sites are in
[ARCHITECTURE.md §6](../../docs/ARCHITECTURE.md#6-what-to-change-to-experiment-with-the-architecture).
The short version:

| experiment | edit | also update |
| --- | --- | --- |
| per-layer bias | `MSDeltaModel.__init__` → `ModuleList` of `DeltaMZBias` | four consumers that reach for `encoder.bias_module` |
| bounded bias (tanh) | `DeltaMZBias._curve` + new config field | `evaluate()` must apply the same transform |
| m/z in the token | `PeakEmbed` (second `FourierFeatures`, wider `mlp[0]`) | thread `mz` into `PeakEmbed.forward` |
| new objective | `MSDeltaForPreTraining.forward` | register any new head with its HF auto-class |
| break the K² ceiling | `DeltaMZBias.forward` + `BiasedMHA.forward` | nothing else — blocks and heads are untouched |

Whatever you change: add the knob to **both** `MSDeltaConfig.__init__` and `_validate()`, add it
to the `configs/*/config.json` tiers you plan to run, and remember that a stale config key is
silently ignored (that is how `zero_bias_diagonal` survives in the 400M config with no effect).

The bias module is **under 0.01% of parameters at every size tier** — you can afford to make it
much more expressive. The cost is activation memory, not weights.
