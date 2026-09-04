# MSDelta architecture

> Everything here is the state of `msdelta/model/modeling.py` + `msdelta/model/configuration.py`
> as it currently stands. Every claim is traceable to a specific class; the "What to change"
> sections at the end tell you which lines to touch for each kind of experiment.

---

## 1. The design commitment

A conventional transformer gives a token an identity (`content`) and a place (`position`).
MSDelta splits a mass-spec peak the same way, but makes an unusual choice about *where the
place goes*:

| | conventional | MSDelta |
| --- | --- | --- |
| token content | word embedding | `MLP(Fourier(log intensity))` — **intensity only** |
| position | absolute or relative index embedding | **nothing in the token** |
| where position acts | added to the token, or as an attention bias | **only as a per-head additive attention bias `B_h(m/z_i − m/z_j)`** |

The consequence, which is the whole point of the project: the encoder is **permutation
equivariant over peaks** and can only reason about mass through *differences*. It cannot memorize
"there is always a peak at 175.12"; it can only learn "a peak 18.011 Da below another peak means
something." That is exactly the structure of fragmentation chemistry.

The trade-off is that the bias is a dense `(B, heads, K, K)` tensor, which is what caps `max_peaks`
at 150 and forces the memory-efficient SDPA kernel instead of FlashAttention.

---

## 2. Full forward pass

```
INPUT   mz (B,K) float32        log_intensity (B,K) ∈ (0,1]      attention_mask (B,K)
                                mask_positions (B,K) bool | None

╔═══════════════════════════ TOKEN PATH (no m/z) ══════════════╗   ╔═════ BIAS PATH (m/z only) ═════╗
║                                                              ║   ║                                ║
║  log_intensity            (B,K)                              ║   ║  Δ = mz[:,:,None]-mz[:,None,:] ║
║        │                                                     ║   ║                       (B,K,K)  ║
║   FourierFeatures                                            ║   ║        │                       ║
║   n_freqs=16, 1e-2..1e2, log-spaced, LEARNABLE               ║   ║   FourierFeatures              ║
║   clamp |x| ≤ 2000; φ = 2π·x·|f|;  [sin φ ‖ cos φ]           ║   ║   n_freqs=64, 1e-2..1e3,       ║
║        │                  (B,K,32)                           ║   ║   log-spaced, LEARNABLE        ║
║   Linear(32→H) → GELU → Linear(H→H)                          ║   ║        │        (B,K,K,128)    ║
║        │                  (B,K,H)                            ║   ║   ┌────┴──────────────────┐    ║
║   where(mask_positions, mask_token, ·)   ← learned Parameter ║   ║   │ head_mlps: ModuleList │    ║
║        │                                                     ║   ║   │  A independent MLPs   │    ║
║        ▼                  hidden_states                      ║   ║   │  128 → 32 → 1 (GELU)  │    ║
╚══════════════════════════════════════════════════════════════╝   ║   └────┬──────────────────┘    ║
                             │                                     ║   cat over heads (B,K,K,A)     ║
                             │                                     ║   permute      → (B,A,K,K)     ║
                             │                                     ╚════════════════╤═══════════════╝
                             │                                                      │
                             │       ┌────────────────────────────────────────────────┐
                             │       │  bias  (B, A, K, K)   COMPUTED ONCE            │
                             │       │  reused unchanged by every layer               │
                             │       └────────────────────┬───────────────────────────┘
                             ▼                            │
              ╔══════════ EncoderBlock × num_hidden_layers ════════════════════════════╗
              ║   h ─┬─ LayerNorm ─ BiasedMHA(·, bias, padding_mask) ─┐                ║
              ║      └──────────────────── + ◄─────────────────────────┘   (pre-norm)  ║
              ║   h ─┬─ LayerNorm ─ FFN: H→I→GELU→drop→I→H→drop ───────┐               ║
              ║      └──────────────────── + ◄─────────────────────────┘               ║
              ╚═══════════════════════════════════════════════════════════════════════╝
                             │
                        final LayerNorm
                             │
                      last_hidden_state (B,K,H)
                             │
              ┌──────────────┴───────────────┐
              ▼                              ▼
   IntensityHead                    PeakDenoisingHead
   Linear(H→1) → (B,K)              H→head_hidden→GELU→drop→1 → (B,K)
              │                              │
   KL over MASKED peaks              BCE-with-logits, ignore label −100
   (MSDeltaForPreTraining)           (MSDeltaForDenoising, encoder frozen)
```

### Inside `BiasedMHA`

```python
qkv  = Linear(H, 3H)(h).reshape(B, K, 3, A, d_head)   # d_head = H // A
q, k, v = qkv.unbind(2)  →  each transposed to (B, A, K, d_head)

attention_bias = bias.masked_fill(padding_mask[:, None, None, :], -inf)   # (B,A,K,K)

context = F.scaled_dot_product_attention(q, k, v,
                                         attn_mask=attention_bias,        # ADDITIVE float mask
                                         dropout_p=attention_probs_dropout_prob)
out = Dropout(Linear(H, H)(context.transpose(1,2).reshape(B, K, H)))
```

Two things to internalize:

1. **The bias is an additive float `attn_mask`, not a boolean one.** SDPA adds it to the scaled
   logits before the softmax. Padding is folded into the same tensor as `-inf`. There is no
   separate masking step.
2. **`bias` is layer-independent.** The same curve shapes the attention in layer 0 and layer 19.
   Making the bias per-layer is the single highest-leverage architecture change available
   (§6.1).

### Padding, the diagonal, and dtype

* Padded **keys** are masked; padded **queries** are not, so padded rows produce garbage hidden
  states — every consumer (`pool_tokens`, both losses) re-applies `attention_mask` downstream.
* `Δ = 0` on the diagonal is *not* special-cased. Each head evaluates its curve at zero and that
  value becomes the self-attention bias. A `zero_bias_diagonal` option existed and was removed
  (commit `22c0a14`) — but `configs/msdelta-base-400m/config.json` still sets it, where it is now
  an inert attribute.
* `FourierFeatures.forward` always upcasts to fp32 and clamps `|x| ≤ 2000`; the consumers cast
  back to the parameter dtype. Under bf16 autocast the sin/cos are therefore computed in fp32,
  which matters — the high frequencies (up to 1000 Hz per Da) would alias badly otherwise.
* Both heads `.float()` their logits so the losses run in fp32.

---

## 3. The two losses

### Pretraining — masked relative intensity, as a distribution (`MSDeltaForPreTraining.forward`)

This is **not** per-peak regression. Logits over the masked positions are softmaxed *against each
other*, and matched to the true relative abundances renormalized over the same set:

```python
selected = mask_positions.bool()
log_prob = log_softmax(logits.masked_fill(~selected, -inf), dim=-1).masked_fill(~selected, 0.0)
target = labels.masked_fill(~selected, 0.0)
target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
loss = F.kl_div(log_prob, target, reduction="batchmean")
```

So the model predicts the **shape of the masked sub-spectrum**, not absolute heights. `labels`
comes out of the processor as `intensity / Σ intensity` over the *retained* peaks. `batchmean`
divides by batch size, so the per-spectrum loss scale drifts with the number of masked peaks —
worth remembering when comparing runs at different `mask_ratio` (the shipped args use `0.50`,
far above the BERT-style `0.15` default in the collator).

### Denoising — per-peak binary classification (`MSDeltaForDenoising.forward`)

Straight `binary_cross_entropy_with_logits` over peaks where `label != -100` and
`attention_mask` is set. When constructed with `freeze_encoder=True`, the encoder is
`requires_grad_(False)`, pinned to `.eval()` through an overridden `train()`, and run under
`torch.no_grad()` — so encoder dropout is off and no activations are kept.

---

## 4. Parameter budget

The bias module is essentially free; the encoder stack is everything.

| tier | H | layers | heads | intermediate | embed | **bias** | blocks | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50m | 640 | 10 | 10 | 2560 | 0.43M | **0.042M** | 49.2M | **49.7M** |
| 100m | 800 | 13 | 10 | 3200 | 0.67M | **0.042M** | 100.0M | **100.7M** |
| 200m | 1024 | 16 | 16 | 4096 | 1.08M | **0.067M** | 201.5M | **202.7M** |
| 400m | 1280 | 20 | 20 | 5120 | 1.68M | **0.083M** | 393.6M | **395.3M** |
| 1b | 2048 | 20 | 16 | 8192 | 4.27M | **0.067M** | 1007.2M | **1011.5M** |

The interpretable part of the model is **under 0.01% of its parameters**. You can make the bias
module dramatically more expressive (deeper per-head MLPs, more frequencies, per-layer biases)
before it registers on the parameter count. The costs are activation memory and wall-clock, not
weights.

### Memory: the bias tensor is the constraint

```
bias activation  =  B × A × K × K × 2 bytes (bf16)
```

At `B=64`, `A=10`, `K=150` that is ~28 MB per copy — modest. At the denoising settings
(`K=1024`) a single spectrum's bias is `A × 1024² ≈ 21 M` entries, which is exactly why
`PeakBudgetBatchSampler` exists: it packs batches under a `denoise_peak_pair_budget`
(4,194,304) of `batch × max_len²` rather than a fixed batch size.

Quadratic-in-`K` scaling with a *materialized* bias is the hard ceiling on this architecture.
Anything that raises `max_peaks` meaningfully needs the bias to stop being dense (§6.6).

---

## 5. Where each config field lands

`MSDeltaConfig` (validated by `_validate()`; re-validate after `update_from_string`):

| field | default | consumed by | effect |
| --- | --- | --- | --- |
| `hidden_size` | 256 | everywhere | `H`; must be divisible by `num_attention_heads` |
| `num_attention_heads` | 8 | `BiasedMHA`, `DeltaMZBias` | also = **number of independent bias curves** |
| `num_hidden_layers` | 6 | `MSDeltaModel` | depth |
| `intermediate_size` | 1024 | `EncoderBlock.ffn` | FFN width |
| `hidden_dropout_prob` | 0.1 | FFN + attn output proj | |
| `attention_probs_dropout_prob` | 0.1 | SDPA `dropout_p` | |
| `layer_norm_eps` | 1e-5 | all LayerNorms | |
| `initializer_range` | 0.02 | `_init_weights` | normal std for Linear + `mask_token` |
| `fourier_int_n_freqs` | 16 | `PeakEmbed.ff_int` | token feature dim = `2 × n_freqs` |
| `fourier_int_f_min/f_max` | 1e-2 / 1e2 | `PeakEmbed.ff_int` | logspace range over normalized log-intensity ∈ (0,1] |
| `fourier_int_learnable` | True | | frequencies become `nn.Parameter` |
| `delta_bias_n_freqs` | 64 | `DeltaMZBias.ff` | **mass resolution of the bias**; feature dim `2 × n` |
| `delta_bias_f_min/f_max` | 1e-2 / 1e3 | `DeltaMZBias.ff` | `f_max=1000` → period 1 mDa, i.e. it *can* resolve isotope fine structure |
| `delta_bias_per_head_hidden` | 32 | `DeltaMZBias.head_mlps` | hidden width of each per-head curve MLP |
| `delta_bias_learnable` | True | | |

`MSDeltaDenoisingConfig` composes `encoder: MSDeltaConfig` with `head_hidden_size` (128) and
`head_dropout` (0.1), and proxies `initializer_range` to the encoder.

One training-side detail that is easy to lose: `train/cli.py::MSDeltaTrainer.get_decay_parameter_names` strips
anything ending in `.freqs`, so **learned Fourier frequencies are excluded from weight decay**.
If you rename that parameter, you silently start decaying your frequencies toward zero.

---

## 6. What to change to experiment with the architecture

Each entry lists the file, the concrete edit, and what it costs.

### 6.1 Per-layer bias curves (highest leverage)

Today one `DeltaMZBias` serves all layers, so every layer sees identical mass structure. Layer 0
probably wants sharp isotope spikes; layer 15 probably wants broad residue-scale structure.

* `model/modeling.py` → `MSDeltaModel.__init__`: replace `self.bias_module = DeltaMZBias(config)`
  with `nn.ModuleList([DeltaMZBias(config) for _ in range(config.num_hidden_layers)])`, and in
  `forward` compute the bias inside the block loop.
* **You must also update the consumers**, which all reach for `encoder.bias_module`:
  `eval/viz.py::render_bias_panels`, `eval/alignment.py::_eval_curves`,
  `train/callbacks.py::FourierProbeCallback.run` (`enc.bias_module.ff`), and `train/cli.py::main`'s
  final panel dump. Keep a `bias_module` property
  aliasing layer 0, or teach the diagnostics to loop.
* Cost: `L ×` the bias compute and `L ×` the parameters of a module that is currently <0.1% of the
  model. Cheap. A middle ground — sharing the Fourier features but giving each layer its own
  `head_mlps` — is cheaper still.

### 6.2 Bounding the bias magnitude

Unbounded logit bias can saturate the softmax and make the curves hard to read. Git history shows
this was tried (`3cff5e5` add scale sweep, `7396e38` remove it), and the untracked
`msdelta/model/experimental.py` re-adds it:

```python
# DeltaMZBias._curve
out = torch.cat([mlp(feats) for mlp in self.head_mlps], dim=-1)
return self.scale * torch.tanh(out / self.scale)  # soft clamp to ±scale
```

To do this properly: add `delta_bias_scale: float = 8.0` to `MSDeltaConfig.__init__` (and a
positivity check in `_validate`), read it in `DeltaMZBias.__init__`, apply it in `_curve` so that
`evaluate()` — and therefore every plot and the alignment test — sees the same transform.
`TODO.txt` also lists sigmoid as an ablation against tanh.

### 6.3 Putting m/z back into the token (the leakage ablation)

`TODO.txt` asks "is the model cheating? precursors are an easily learned signal." The clean
ablation is the opposite direction: give the token absolute m/z and see how much the Δ bias still
buys you.

* `model/modeling.py` → `PeakEmbed`: add a second `FourierFeatures` over `mz`, concatenate with
  the intensity features, widen `self.mlp[0]` accordingly, and thread `mz` through
  `PeakEmbed.forward` (it currently only receives `log_intensity`).
* `MSDeltaModel.forward` already has `mz` in scope — one extra argument at the call site.
* This is also the hook for the "Hadamard for position" idea in `TODO.txt`: multiply rather than
  concatenate the two feature sets.

### 6.4 Attention/block structure

* **Bias with a learned per-head gate**: multiply the bias by a scalar `nn.Parameter` per head so
  the model can learn to ignore mass structure in some heads. Two lines in `DeltaMZBias`.
* **Post-norm / sandwich norm**: `EncoderBlock.forward` is pre-norm today; changing it is local to
  that one method.
* **QK-norm, RMSNorm, SwiGLU FFN, RoPE-free by construction** — all confined to `EncoderBlock`
  and `BiasedMHA`.
* **Residual / bias-insertion variants** from `TODO.txt`: see the scratch-file discussion below
  for what a correct deep-residual variant should look like.
* Note: any change that leaves `EncoderBlock`'s name intact keeps `_no_split_modules` and
  gradient checkpointing working. If you rename it, update `MSDeltaPreTrainedModel._no_split_modules`.

### 6.5 The Fourier front-end

`model/fourier.py::FourierFeatures` is shared by both paths, so a change there hits tokens *and* bias.
If you want them to diverge (e.g. learnable phases for the bias only), subclass rather than edit.

* Raising `delta_bias_n_freqs` is the direct knob on mass resolution and is nearly free in
  parameters (`+2n × per_head_hidden` per head) but grows the `(B,K,K,2n)` intermediate — that
  tensor, not the weights, is what will OOM you.
* `FourierProbeCallback` already tells you whether the frequencies you added are doing anything:
  `fourier/dm_dead` counts frequencies completing less than half a cycle over the observed data
  span, and `fourier/dm_drift_log10` tracks how far training moved them.
* `clamp_abs=2000.0` silently saturates `|Δ| > 2000 Da`. Fine for peptide MS/MS; revisit for
  intact-protein or lipid data.

### 6.6 Breaking the `K²` ceiling

If you want `max_peaks` ≫ 150, the dense bias has to go:

* **Low-rank bias**: factor `B_h(Δ) ≈ Σ_r u_r(m/z_i)·v_r(m/z_j)` so it can be folded into
  extra Q/K channels and FlashAttention comes back.
* **Bucketed bias**: quantize `Δ` onto a shared grid and gather, trading exactness for a much
  smaller intermediate.
* **Sparse/windowed attention** over sorted m/z, exploiting the fact that chemically meaningful
  Δ values are bounded.

Each of these is a rewrite of `DeltaMZBias.forward` + `BiasedMHA.forward` only; the block, heads,
and losses are untouched.

### 6.7 A new pretraining objective

`model/modeling.py::MSDeltaForPreTraining.forward` is self-contained. Swap the KL for per-peak MSE, add an auxiliary
Δ-prediction task, or add a contrastive spectrum-level loss — the encoder does not change. If you
add a new head class, register it with the matching HF auto-class at the bottom of the file so
checkpoints keep round-tripping.

### 6.8 Checklist for any architecture change

1. Add the knob to `MSDeltaConfig.__init__` **and** `_validate()`.
2. Add it to the `configs/*/config.json` tiers you intend to run (a missing key silently takes the
   dataclass default; a stale key is silently ignored — that is how `zero_bias_diagonal` survives
   in the 400M config).
3. Update anything reaching into module internals: `encoder.bias_module`, `encoder.embed.ff_int`,
   `encoder.bias_module.ff`, `.msdelta`. Grep for those four.
4. If a parameter should escape weight decay, make sure its name still ends in `.freqs`, or extend
   `train/cli.py::MSDeltaTrainer.get_decay_parameter_names`.
5. Sanity-check on the 50M tier with `--max_steps 50 --probe_steps 0 --denoise_steps 0
   --bias_curve_steps 10 --report_to none` before touching a real run.
6. Old checkpoints will not load — `from_pretrained` will report missing/unexpected keys. Version
   your `output_dir`.

---

## 7. The scratch variant: `model/experimental.py`

Imported by nothing, deliberately not re-exported from `msdelta/model/__init__.py`, and **it will
not run as written**. Two problems:

**(a) `config.delta_bias_scale` does not exist.** `DeltaMZBias.__init__` reads
`config.delta_bias_scale`, but that field was deleted from `MSDeltaConfig` in commit `7396e38`.
Constructing any model from this file raises `AttributeError`. Fix per §6.2.

**(b) `MSDeltaModel_2`'s residual loop is wrong.** As written:

```python
for block in self.blocks:
    hidden_states_residual = block(hidden_states, bias, padding_mask)  # hidden_states never updated
hidden_states = self.norm(hidden_states + hidden_states_residual)
```

`hidden_states` is never reassigned inside the loop, so every block receives the *embedding
output* and all but the last block's result is discarded. The model is effectively one layer deep
with `L−1` blocks computing dead gradients. If the intent was a global embedding→output skip:

```python
residual = hidden_states
for block in self.blocks:
    hidden_states = block(hidden_states, bias, padding_mask)
hidden_states = self.norm(hidden_states + residual)
```

Also note this file re-declares `MSDeltaForPreTrainingOutput`, `MSDeltaModel`,
`MSDeltaForPreTraining` and calls `register_for_auto_class` on them, so importing it *after*
`msdelta/__init__.py` re-registers the auto-classes against different class objects — which is
exactly why `msdelta/model/__init__.py` does not pull it in. Prefer one of:

* add the variant as a `config` flag inside `model/modeling.py` (best for ablations you want to
  sweep from an args file), or
* keep a separate file but give the classes distinct names and skip the auto-class registration.
