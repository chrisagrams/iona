# Every module in `modeling.py`

A module-by-module walkthrough of [`modeling.py`](modeling.py): ten classes and two output
dataclasses. For each one: what it does, exact tensor shapes, its parameter count, the config
fields that control it, and what breaks if you change it.

Parameter counts are **measured** on the `msdelta-base-50m` tier
(`H=640, L=10, A=10, I=2560`), total **49,710,811**. Symbols used throughout:

| symbol | meaning | 50m |
| --- | --- | --- |
| `B` | batch size | — |
| `K` | peaks per spectrum (padded) | ≤ 150 |
| `H` | `hidden_size` | 640 |
| `A` | `num_attention_heads` | 10 |
| `L` | `num_hidden_layers` | 10 |
| `I` | `intermediate_size` | 2560 |
| `d` | head dim = `H / A` | 64 |

---

## The module tree

```
MSDeltaForPreTraining                                            49,710,811   100%
├── msdelta: MSDeltaModel
│   ├── embed: PeakEmbed                                            432,016  0.87%
│   │   ├── ff_int: FourierFeatures                                      16
│   │   ├── mlp: Sequential(Linear 32→640, GELU, Linear 640→640)    431,360
│   │   └── mask_token: Parameter(640)                                  640
│   ├── bias_module: DeltaMZBias                                      41,674  0.08%   ◄── the
│   │   ├── ff: FourierFeatures                                           64          interpretable
│   │   └── head_mlps: ModuleList[10 × (Linear 128→32, GELU, →1)]    41,610          part
│   ├── blocks: ModuleList[10 × EncoderBlock]                     49,235,200 99.04%
│   │   └── EncoderBlock                                            4,923,520
│   │       ├── norm1: LayerNorm(640)                                   1,280
│   │       ├── attn: BiasedMHA                                     1,640,960
│   │       │   ├── qkv:  Linear(640 → 1920)                        1,230,720
│   │       │   └── out:  Linear(640 → 640)                           410,240
│   │       ├── norm2: LayerNorm(640)                                   1,280
│   │       └── ffn: Sequential(640→2560, GELU, drop, 2560→640, drop) 3,280,000
│   └── norm: LayerNorm(640)                                            1,280
└── intensity_head: IntensityHead(Linear 640→1)                            641
```

**99% of the model is the block stack. The bias module — the entire chemical-reasoning surface —
is 0.08%.** That asymmetry is the single most useful fact for planning experiments: you can make
the bias module far more expressive essentially for free in parameter terms. What it costs is
activation memory and wall-clock.

---

## 1. `MSDeltaForPreTrainingOutput` / `MSDeltaForDenoisingOutput`

Two-field `ModelOutput` dataclasses: `loss: Tensor | None`, `logits: Tensor | None`. They exist so
HF `Trainer` can find `.loss` by attribute and so `return_dict=False` callers still get tuples.
Nothing else. If you add an auxiliary loss or want to return attention weights, this is where the
new field goes — and remember `ModelOutput` iterates in declaration order, so **append, never
insert**, or you silently break tuple-unpacking callers.

---

## 2. `PeakEmbed` — the token, which contains no m/z

The whole architectural bet lives here: a token is built from **intensity alone**.

```
log_intensity (B,K) ∈ (0,1]
      │
      ▼
 FourierFeatures(n=16, f∈[1e-2,1e2], log-spaced, learnable)
      │   φ = 2π · x · |f|   →   [sin φ ‖ cos φ]
      ▼
   (B, K, 32)                          ← out_dim = 2 · n_freqs
      │
 Linear(32 → 640) → GELU → Linear(640 → 640)
      │
   (B, K, 640)
      │
 where(mask_positions, mask_token, ·)  ← learned Parameter(640) substituted in place
      │
      ▼
 hidden_states (B, K, 640)
```

| | |
| --- | --- |
| **in** | `log_intensity (B,K)`, `mask_positions (B,K) bool \| None` |
| **out** | `(B, K, H)` |
| **params** | 432,016 — `ff_int` 16, `mlp` 431,360, `mask_token` 640 |
| **config** | `fourier_int_n_freqs`, `fourier_int_f_min/f_max`, `fourier_int_learnable`, `hidden_size` |

Notes:

* `forward` takes **no `mz` argument at all.** That is not an oversight — it is the design. If you
  want the absolute-m/z ablation, this signature is what you change (and `MSDeltaModel.forward`
  already has `mz` in scope to pass it).
* Masking substitutes the token *after* the MLP, so a masked peak contributes an identical vector
  regardless of its true intensity. Its m/z is still visible to every other peak through the bias
  — which is exactly what makes the pretraining task well-posed.
* The Fourier frequencies span `[0.01, 100]` over an input in `(0, 1]`. The low end completes
  1/100th of a cycle across the whole range and is effectively dead; `FourierProbeCallback`'s
  `fourier/int_dead` counts exactly this.

---

## 3. `DeltaMZBias` — where all the m/z lives

The interpretable core. One learned curve per attention head over signed mass difference.

```
mz (B,K)
   │
   │  Δ = mz.unsqueeze(-1) − mz.unsqueeze(-2)         ← outer difference, signed
   ▼
Δ (B,K,K)          Δ[b,i,j] = m/z_i − m/z_j     (antisymmetric, zero diagonal)
   │
   ▼
FourierFeatures(n=64, f∈[1e-2,1e3], log-spaced, learnable, clamp |Δ|≤2000)
   │
(B,K,K,128)
   │
   ├──► head_mlps[0]:  Linear(128→32) → GELU → Linear(32→1) ──┐
   ├──► head_mlps[1]:  …                                      │  cat(dim=-1)
   ├──► …                                                     │
   └──► head_mlps[9]:  …                                      ▼
                                                        (B,K,K,10)
                                                              │ permute(0,3,1,2)
                                                              ▼
                                                    bias (B, 10, K, K)
```

| | |
| --- | --- |
| **in** | `mz (B,K)` |
| **out** | `(B, A, K, K)` — one `K×K` bias matrix per head |
| **params** | 41,674 — `ff` 64, `head_mlps` 41,610 (`A × (128·32+32 + 32+1)`) |
| **config** | `delta_bias_n_freqs`, `delta_bias_f_min/f_max`, `delta_bias_learnable`, `delta_bias_per_head_hidden`, `num_attention_heads` |

Three methods:

* **`forward(mz)`** — the `(B,A,K,K)` bias used in training.
* **`_curve(feats)`** — shared kernel: run every head MLP and concatenate. Casts features to the
  parameter dtype (they arrive as fp32 from `FourierFeatures`).
* **`evaluate(grid)`** — evaluates the same curves on a **1-D** Δ grid, returning `(G, A)` float32.
  This is the interpretability entry point: [`eval/viz.py`](../eval/viz.py) plots it and
  [`eval/alignment.py`](../eval/alignment.py) runs the peak↔chemistry test on it.

> **If you modify `_curve`, both `forward` and `evaluate` change together — that is the point.**
> Any transform applied in only one of them (a tanh clamp in `forward` but not `evaluate`, say)
> means your plots and your significance test describe a model you are not training.

Why `f_max = 1000`? Period = 1/1000 Da = 1 mDa, so the featurizer *can* in principle resolve
isotope fine structure. Whether it does is what `align/fine_*` measures.

**Cost.** Parameters are negligible; the intermediate `(B,K,K,2·n_freqs)` tensor is not. At
`B=64, K=150, n=64` that is 64·150·150·128 ≈ 184M floats before the head MLPs reduce it. This,
not the weights, is what OOMs when you raise `delta_bias_n_freqs` or `max_peaks`.

---

## 4. `BiasedMHA` — attention that adds the bias

Standard fused-QKV multi-head attention with one twist: the bias enters as an **additive float
`attn_mask`**, and padding is folded into the same tensor.

```
hidden_states (B,K,640)
      │
 Linear(640 → 1920)                          bias (B,10,K,K)   padding_mask (B,K)
      │                                            │                 │
 reshape (B,K,3,10,64) → unbind(2)                 └──── masked_fill(pad keys, −inf)
      │                                                        │
  q,k,v each (B,10,K,64)                            attention_bias (B,10,K,K)
      │                                                        │
      └────────────────► F.scaled_dot_product_attention ◄──────┘
                          softmax(QKᵀ/√d + attention_bias) · V
                                     │
                          context (B,10,K,64)
                                     │
                       transpose+reshape → (B,K,640)
                                     │
                     Linear(640→640) → Dropout
                                     │
                                     ▼
                               (B,K,640)
```

| | |
| --- | --- |
| **in** | `hidden_states (B,K,H)`, `bias (B,A,K,K)`, `padding_mask (B,K) bool` |
| **out** | `(B,K,H)` |
| **params** | 1,640,960 per block — `qkv` 1,230,720, `out` 410,240 |
| **config** | `hidden_size`, `num_attention_heads`, `attention_probs_dropout_prob`, `hidden_dropout_prob` |

Four things worth internalizing:

1. **The bias is added to the pre-softmax logits**, not multiplied and not concatenated. A head can
   suppress a mass relationship with a large negative value or promote it with a positive one.
2. **`padding_mask` marks keys, not queries.** Padded *rows* still produce output vectors — garbage
   ones. Every consumer downstream re-applies `attention_mask`; if you write new code that reads
   `last_hidden_state`, you must too.
3. **A float `attn_mask` disables FlashAttention.** SDPA falls back to the memory-efficient or math
   kernel and materializes the `(B,A,K,K)` bias. This is the architecture's hard scaling limit.
4. `padding_mask` is `~attention_mask.bool()` — inverted relative to the HF convention, computed
   once in `MSDeltaModel.forward`.

---

## 5. `EncoderBlock` — pre-norm transformer block

```
      x ─────────────────────────────┐
      │                              │
 LayerNorm(norm1)                    │
      │                              │
 BiasedMHA(·, bias, padding_mask)    │
      │                              │
      └──────────► + ◄───────────────┘
                   │
      x' ──────────┼──────────────────┐
                   │                  │
             LayerNorm(norm2)         │
                   │                  │
    Linear(640→2560) → GELU → Dropout │
    Linear(2560→640) → Dropout        │
                   │                  │
                   └────► + ◄─────────┘
                          │
                          ▼
```

| | |
| --- | --- |
| **in / out** | `(B,K,H)` → `(B,K,H)` |
| **params** | 4,923,520 — attn 1,640,960, ffn 3,280,000, two LayerNorms 2,560 |
| **config** | `hidden_size`, `intermediate_size`, `hidden_dropout_prob`, `layer_norm_eps` |

Pre-norm (`x + f(LN(x))`), so the residual stream is never normalized in place and deep stacks
train without warmup tricks. The FFN is a plain 4× GELU MLP with dropout after *both* linears.

> **Keep the class name `EncoderBlock` if you restructure this.**
> `MSDeltaPreTrainedModel._no_split_modules = ["EncoderBlock"]` is matched by string; renaming it
> silently breaks DeepSpeed/accelerate device-map splitting.

---

## 6. `MSDeltaPreTrainedModel` — the HF base class

Not a computational module — it carries the Hugging Face contract:

| attribute | value | why it matters |
| --- | --- | --- |
| `config_class` | `MSDeltaConfig` | `from_pretrained` builds the right config |
| `base_model_prefix` | `"msdelta"` | HF strips/adds this when moving weights between the base model and task heads — it is why the encoder attribute is named `self.msdelta` everywhere |
| `main_input_name` | `"mz"` | not `input_ids`; Trainer and pipelines use it to find the primary input |
| `supports_gradient_checkpointing` | `True` | enables `--gradient_checkpointing` |
| `_no_split_modules` | `["EncoderBlock"]` | device-map / ZeRO sharding boundary |

`_init_weights(module)` handles three cases: `nn.Linear` → `N(0, initializer_range)` with zero
bias; `nn.LayerNorm` → zero bias, unit weight; `PeakEmbed` → normal init for `mask_token`.

> Note what is **absent**: `FourierFeatures.freqs` is never touched by `_init_weights`. Frequencies
> keep their `logspace` initialization, which is correct — re-initializing them normally would
> destroy the whole point. If you add a new parameter that needs custom init, add a branch here.

---

## 7. `MSDeltaModel` — the encoder

Ties it together. **The bias is computed once and reused by every layer.**

```
mz (B,K)   log_intensity (B,K)   attention_mask (B,K)   mask_positions (B,K)
  │              │                     │                      │
  │              │              padding_mask = ~mask           │
  │              └──────────► PeakEmbed ◄─────────────────────┘
  │                              │
  │                         h (B,K,640)
  │                              │
  └──► DeltaMZBias ──► bias (B,10,K,K) ─────┐
                                            │  same tensor, every layer
       ┌────────────────────────────────────┤
       ▼                                    │
  EncoderBlock[0] ◄─────────────────────────┤
       ▼                                    │
  EncoderBlock[1] ◄─────────────────────────┤
       ⋮                                    │
  EncoderBlock[9] ◄─────────────────────────┘
       ▼
  LayerNorm
       ▼
  BaseModelOutput(last_hidden_state=(B,K,640))
```

| | |
| --- | --- |
| **in** | `mz`, `log_intensity`, `attention_mask=None`, `mask_positions=None`, `return_dict=None` |
| **out** | `BaseModelOutput` with `last_hidden_state (B,K,H)`, or a 1-tuple |
| **params** | 49,710,170 (everything except `intensity_head`) |

`forward` validates up front — `mz` must be 2-D, `log_intensity` / `attention_mask` /
`mask_positions` must match its shape — then defaults `attention_mask` to all-ones. Gradient
checkpointing wraps each block call when `self.gradient_checkpointing and self.training`.

**Sharing one bias across layers is the most consequential simplification in the model** and the
first thing worth ablating; see
[ARCHITECTURE.md §6.1](../../docs/ARCHITECTURE.md#61-per-layer-bias-curves-highest-leverage).

---

## 8. `IntensityHead` — pretraining head

A single `Linear(H → 1)`, squeezed and cast to fp32. **641 parameters** — 0.001% of the model.

```
(B,K,640) → Linear(640→1) → squeeze(-1) → .float() → (B,K)
```

Deliberately linear: the pretraining objective is meant to measure what the *encoder* knows, so
the head is given no capacity to compensate. The `.float()` keeps the loss in fp32 under bf16
autocast.

---

## 9. `PeakDenoisingHead` — denoising head

```
(B,K,640) → Linear(640→128) → GELU → Dropout(0.1) → Linear(128→1) → squeeze → .float() → (B,K)
```

**82,177 parameters.** Two layers rather than one because it is trained on a *frozen* encoder —
it has to work with representations it cannot influence. Config: `head_hidden_size`,
`head_dropout` on `MSDeltaDenoisingConfig`.

---

## 10. `MSDeltaForPreTraining` — masked relative intensity

```
MSDeltaModel ──► last_hidden_state ──► IntensityHead ──► logits (B,K)
                                                            │
                              labels (B,K)   mask_positions (B,K)
                                                            ▼
                    log_softmax over MASKED positions only
                    target = labels / Σ labels over the same set
                    loss = KL(log_prob ‖ target), reduction="batchmean"
```

```python
selected = mask_positions.bool()
log_prob = log_softmax(logits.masked_fill(~selected, -inf), -1).masked_fill(~selected, 0.0)
target   = labels.masked_fill(~selected, 0.0)
target   = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
loss     = F.kl_div(log_prob, target, reduction="batchmean")
```

This is **not** per-peak regression. The masked peaks are softmaxed *against each other*, so the
model predicts the **shape of the masked sub-spectrum**, not absolute heights. Consequences:

* Loss scale depends on how many peaks are masked (`batchmean` divides by batch size, not by
  masked count) — **do not compare losses across different `--mask_ratio` values.**
* Two distinct empty cases, both safe (verified): a **batch** with nothing masked anywhere
  short-circuits to `logits.new_zeros(())`, keeping DDP gradients well-defined; an individual
  **row** with nothing masked is handled by the masking arithmetic — its all-`-inf` `log_softmax`
  row is overwritten with zeros and its zero-sum target contributes nothing, so no NaN escapes.
* Guards: `labels.shape == logits.shape`, and `mask_positions` is required whenever `labels` is
  given.

---

## 11. `MSDeltaForDenoising` — frozen-encoder peak classifier

The only module with real lifecycle logic.

```python
MSDeltaForDenoising(config, encoder=None, freeze_encoder=False)
```

* `encoder=None` → builds a fresh `MSDeltaModel(config.encoder)` and calls `post_init()`.
* `encoder=<live model>` → **wraps that instance** and initializes *only* the head
  (`self.denoising_head.apply(self._init_weights)`). This is what lets
  [`eval/denoising.py`](../eval/denoising.py) probe the encoder mid-pretraining without
  reconstructing or copying it.

Freezing is enforced three ways, because any one alone would leak:

| mechanism | prevents |
| --- | --- |
| `requires_grad_(False)` | gradient accumulation into encoder weights |
| overridden `train(mode)` pinning the encoder to `.eval()` | dropout re-enabling when the parent is set to train |
| `forward` running the encoder under `torch.no_grad()` | activation storage |

Loss is `binary_cross_entropy_with_logits` over peaks where `labels != -100` **and**
`attention_mask` is set, with **noise as the positive class**. Empty-valid-set batches return
`logits.sum() * 0.0` — a real zero that still participates in the autograd graph.

---

## Auto-class registration

```python
MSDeltaModel.register_for_auto_class("AutoModel")
MSDeltaForPreTraining.register_for_auto_class("AutoModelForPreTraining")
MSDeltaForDenoising.register_for_auto_class("AutoModelForTokenClassification")
```

Runs at **import time**, which is why [`experimental.py`](experimental.py) is dangerous to import:
it re-declares these class names and re-registers them against different objects. New variants
should either skip registration or use distinct class names — see
[README.md](README.md#two-things-that-will-bite-you).

---

## Change-impact quick reference

| you change | also update |
| --- | --- |
| `DeltaMZBias._curve` | nothing — `forward` and `evaluate` both route through it (that is why it exists) |
| `DeltaMZBias` → per-layer | `eval/viz.py`, `eval/alignment.py::_eval_curves`, `train/callbacks.py::FourierProbeCallback`, `train/cli.py::main` — all reach for `encoder.bias_module` |
| `PeakEmbed.forward` signature | `MSDeltaModel.forward` call site; `FourierProbeCallback` reads `encoder.embed.ff_int` |
| rename `EncoderBlock` | `_no_split_modules` |
| rename the `self.msdelta` attribute | `base_model_prefix`, `_InlineCallback.encoder`, `run_denoising_probe`, `train/cli.py` |
| rename `FourierFeatures.freqs` | `MSDeltaTrainer.get_decay_parameter_names` — frequencies would start getting weight-decayed |
| add a field to an Output dataclass | append at the end; tuple order is positional |

Recipes with exact edit sites:
[ARCHITECTURE.md §6](../../docs/ARCHITECTURE.md#6-what-to-change-to-experiment-with-the-architecture).
