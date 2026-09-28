# Pairformer encoder: what our port computes

Code: `msdelta/models/pairformer.py` (branch `p1-pairformer`, 6c03366). The switch is
`MSDeltaConfig.architecture="pairformer"` (`configuration_msdelta.py`). `MSDeltaModel.__init__` /
`forward` (`modeling_msdelta.py`) call `build_pairformer` / `encode_pairformer`, then apply the
same final LayerNorm and heads as the transformer. Port source: `sweep/pairformer-aurora` @ bc2037b,
`msdelta/model/experiments/pairformer.py`. Review: `notes/PAIRFORMER_REVIEW.md` (main checkout).
Everything below was read from the code. Numbers marked *est.* are computed, not measured.

## 0. Notation

| symbol | meaning | config field | Stage-0 value (source 50m) |
|---|---|---|---|
| B, N | spectra per micro-batch; peaks after padding to the batch maximum | -- | 32, <=150 |
| h (= c_s) | single width | `hidden_size` | 512 |
| H, d | single-attention heads; head dim d = h/H | `num_attention_heads` | 8, 64 |
| I | single transition inner width | `intermediate_size` | 2048 |
| L | layers. Every layer has one pair layer and one single block | `num_hidden_layers` | 10 |
| c_z | pair width | `pair_channels` | 64 |
| c_t | triangle-multiplication hidden width | `pair_tri_channels` | 64 |
| c_o | write-back (outer-product) width | `pair_opm_channels` | 16 |
| e | pair-transition expansion | `pair_transition_expansion` | 2 |
| H_t, d_t | triangle-attention heads and dim (off by default) | `pair_tri_attn_heads/dim` | 4, 16 |
| F | Fourier frequencies for m/z and Δm/z | `delta_bias_n_freqs` | 64 |
| F_md | mass-defect frequencies | `pair_mass_defect_n_freqs` | 32 |
| K | neutral-loss / residue dictionary size (fixed) | -- | 56 |
| D | pair-feature width = 2F + 2F_md + K + 2 + 1 (all features on) | -- | 251 |

`FF(x) = [sin(2π f_k x), cos(2π f_k x)]_{k=1..n}` with fixed (non-learned) log-spaced `f_k`. The
input is clamped to ±2000 first (`fourier.py`).

## 1. Data flow

Inputs per batch, all `(B, N)`:
- `mz`: fp32.
- `log_intensity` ℓ: `log1p(I) / max log1p(I)` over all peaks of the spectrum (processor).
- `attention_mask` m: 1 = real peak.
- `mask_positions` M: the peaks masked for pretraining. Absent when fine-tuning.

Pair mask: `m_ij = m_i ∧ m_j`.

**Init.**
1. Single: `s_i = W₂ GELU(W₁ ℓ_i + b₁) + b₂ + W_mz FF_F(mz_i)`. The m/z term is present only
   when `pair_single_use_mz`. If `M_i`, the whole token (m/z part included) becomes
   `mask_token`. The masked peak's m/z still reaches the model through the pair features.
2. Pair: `z_ij = W_a s_i + W_b s_j + W_c f_ij`, where f is §2. `W_a ≠ W_b`, so z is directional
   (`z_ij ≠ z_ji`). The `s` used here is the masked `s`.

**Per layer l = 1..L**, in code order (`PairLayer.forward`, then `PairformerSingleBlock.forward`).
Every Δ below is added residually. pDrop = elementwise `Dropout(pair_dropout)`.

| # | block | equation (one line) | when |
|---|---|---|---|
| a | write-back s→z (AF3 Alg. 9, a single "sequence") | `ŝ=LN(s)`; `a_i=W_l ŝ_i·m_i`, `b_j=W_r ŝ_j·m_j`; `Δz_ij = W_o vec(a_i ⊗ b_j) + b_o` (c_o² → c_z); + pDrop | `pair_use_writeback` |
| b | triangle mult., outgoing (Alg. 12) | `ẑ=LN(z)`; `a_ik=σ(W_ag ẑ_ik)⊙W_ap ẑ_ik·m_ik`, same for b; `Δz_ij = σ(W_g ẑ_ij) ⊙ W_o LN(Σ_k a_ik ⊙ b_jk)`; + pDrop | `pair_update="triangle"` |
| c | triangle mult., incoming (Alg. 13) | as b with `Σ_k a_ki ⊙ b_kj` | `"triangle"` |
| d | triangle attn., starting node (Alg. 14) | `ẑ=LN(z)`; per head: `α_ijk = softmax_k(q_ij·k_ik/√d_t + β_jk − ∞·[k pad])`; `Δz_ij = W_o(σ(W_g ẑ_ij + b_g) ⊙ Σ_k α_ijk v_ik) + b_o`; + pDrop | `pair_use_triangle_attention` |
| e | triangle attn., ending node (Alg. 15) | d applied to zᵀ, output transposed back | same |
| f | pair transition (Alg. 11) | `x=LN(z)`; `Δz = W_3(SiLU(W_1 x) ⊙ W_2 x)`, inner e·c_z; no dropout | `"triangle"` or `"transition"` |
| g | pair bias readout | `β^h_ij = (W_β LN(z_ij))_h`; if `pair_bias_scale=c`: `β = c·tanh(β/c)` | always |
| h | single attention + pair bias + gate (Alg. 24, no conditioning) | `x=LN(s)`; `q,k,v = W_qkv x + b`; `o_i = Σ_j softmax_j(q_i·k_j/√d + β^h_ij − ∞·[j pad]) v_j`; `Δs = Drop(W_o(σ(W_g x + b_g) ⊙ o) + b_o)`. Attention-probability dropout is `attention_probs_dropout_prob`; the output dropout is `hidden_dropout_prob` | always |
| i | single transition | `x=LN(s)`; `Δs = Drop(W_3(SiLU(W_1 x) ⊙ W_2 x))`, inner I | always |

Step a reads the `s` produced by the previous layer. The pair updates for layer l therefore finish
before layer l's single update, which is the AF3 order. `pair_update="static"` skips b-f: z stays at
its init, and only each layer's own readout g is learned.

**Output.** `s_out = LN_final(s)`, shape `(B, N, h)`.
- Pretraining (`MSDeltaForPreTraining`): `logit_i = w·s_out_i + b`. The loss is
  `KL(p ‖ softmax_{i∈M}(logit))`, where `p_i = I_i / Σ_{j∈M} I_j` over the masked set, reduced
  with `batchmean` (divided by B).
- Denoise: a per-peak MLP `h → head_hidden → 1` with BCE over real peaks.
- Retrieval / contrastive: `[mean ‖ max]` pooling over real peaks, then MLP, then L2-normalise,
  then SupCon. The layer-mix pooling in `finetune_contrastive` hooks `encoder.embed` and every
  `encoder.blocks[l]`. Each block's output is `s` `(B, N, h)`, exactly as with `EncoderBlock`.
- Diagnostic `bias_module.evaluate(grid)`: this is the Δm/z-only part of the last layer's bias.
  It keeps `W_c`'s p1 columns, then per layer applies transition f and readout g. It uses no outer
  sum, no other features, no triangle and no write-back.

## 2. Pair input features `f_ij` (`PairFeatures`, computed in fp32, concatenated in this order)

With `Δ = mz_i − mz_j` (signed) and `σ_ij = σ_ppm·1e-6·max(mz_i, mz_j, 1)` Da (the tolerance grows
with the heavier peak):

| block | formula | width | config |
|---|---|---|---|
| p1 signed Δm/z | `FF_F(Δ)`, f_k = logspace(f_min, f_max, F). The transformer's `DeltaMZBias` and the token m/z (`W_mz`) use the same bank | 2F | `delta_bias_n_freqs/f_min/f_max` (always on) |
| p2 mass defect | `FF(|Δ| − ⌊|Δ|⌋)`, f_k = logspace(1, max(F_md, 2), F_md), non-integer | 2F_md | `pair_use_mass_defect`, `pair_mass_defect_n_freqs` |
| p3 loss / residue dictionary | `exp(−(|Δ| − m_k)² / 2σ²)`. The m_k are 56 monoisotopic masses: 32 small neutral losses (H at 1.007825 through 115.9137), 19 residues (L=I), and 5 sugar/modification residues | 56 | `pair_use_loss_bank`, `pair_loss_bank_sigma_ppm` |
| p5 13C isotope | `exp(−(|Δ| − k·1.003355)² / 2σ²)`, k = 1, 2 | 2 | `pair_use_isotope` |
| p6 relative intensity | `ℓ̃_i − ℓ̃_j`. `ℓ̃ = ℓ`, except a masked peak gets `ℓ̃ = c`, a learned scalar (init 0) | 1 | `pair_use_intensity` |

- p4 (complementarity `mz_i + mz_j ≈ M_prec + 2H⁺`) is **not ported**, because it needs the
  precursor.
- The dictionary is identical to the source's code: 56 unique values, checked by a diff. The
  source README says "55 entries", which is a miscount.
- p3 and p5 are functions of Δ that p1 already determines. What they add is linear availability
  through `W_c` (source README).
- H (1.007825) sits 4.5 mDa from the 13C spacing, so above roughly m/z 800 the p3 H column and p5
  k=1 are near-duplicates. This was kept on purpose in the source.
- `f` is materialised as `(B, N, N, D)` fp32: 0.72 GB at B32/N150/D251.

## 3. Shapes, parameters, cost

**Tensors.**
- `s`: `(B, N, h)`.
- `z`: `(B, N, N, c_z)`.
- bias: `(B, H, N, N)`. One is read per layer and consumed by that layer's single block.
- Under bf16 autocast every increment added to z is a Linear output, so the z stream is bf16. The
  LayerNorm outputs, `f`, and the triangle operands `a, b` are fp32 (see §6, item 9).

**Parameters per block** (weights + biases; LN = 2·width):

| block | formula | Stage-0 value |
|---|---|---|
| token embed | h² + 4h + 2F·h (the last term only with `pair_single_use_mz`) | 329,728 |
| z init | 2h·c_z + D·c_z (+1 stand-in c) | 81,601 |
| write-back / layer | 2h + 2h·c_o + c_o²·c_z + c_z | 33,856 |
| triangle mult. ×2 / layer | 2·(2c_z + 5c_z·c_t + 2c_t + c_z²) | 49,664 |
| triangle attn. ×2 / layer (off) | 2·(3c_z + 5c_z·H_t·d_t + c_z·H_t + H_t·d_t) | (41,984) |
| pair transition / layer | 3e·c_z² + 2c_z | 24,704 |
| bias readout / layer | 2c_z + c_z·H | 640 |
| single block / layer | 5h² + 9h + 3h·I | 4,461,056 |
| final LN + intensity head | 2h + h + 1 | 1,537 |
| **total** | | **46,112,066** (the pair branch is 1,170,241 = 2.5%) |

- This formula reproduces the meta-device count in `P1_card_draft.md` (46.3M at F=256).
- Transformer `msdelta-base-50m`: 49,813,771.
- Source 50m: 48.78M. The difference is AdaLayerNorm plus the global-conditioning MLP (≈2.7M),
  which are not ported.

**Forward FLOPs** (2 per multiply-add) at B=32, N=150 (B·N² = 720,000 pairs), c_z = c_t = 64,
c_o = 16, h = 512, I = 2048, H = 8, D = 251:

| block | multiply-adds | order in N | GFLOP fwd |
|---|---|---|---|
| features + W_c (once) | B·N²·D·c_z | N² | 23.1 |
| write-back: outer product / output Linear | B·N²·c_o² / B·N²·c_o²·c_z | N² | 0.37 / 23.6 |
| triangle mult. projections (each) | B·N²·(5c_z·c_t + c_z²) | N² | 35.4 |
| triangle mult. einsum (each) | B·N³·c_t | **N³** | 13.8 |
| triangle attn. (each, off) | B·N²·(5c_z·H_t·d_t + c_z·H_t) + 2B·N³·H_t·d_t | **N³** | 29.9 + 27.6 |
| pair transition | 3e·B·N²·c_z² | N² | 35.4 |
| bias readout | B·N²·c_z·H | N² | 0.74 |
| single linears (attention + transition) | B·N·(5h² + 3h·I) | N | 42.8 |
| single attention scores + AV | 2B·N²·h | N² | 1.5 |
| **per layer** (triangle attn. off) | | | **203** (pair 78%, triangle modules 49%) |
| **forward, 10 layers** | | | **2,050** (transformer 50m: 727, of which 236 is `DeltaMZBias` over `(B,N,N,512)`) |

- A training step with checkpointing costs about 4× forward (forward + backward + recompute):
  ≈ 8.2 TFLOP per tile per micro-batch *est.*
- At c = 64 the per-pair Linears (O(N²)) cost more than the cubic einsum until
  N ≈ (5c_z·c_t + c_z²)/c_t = 384.
- The source measured the triangle modules at 87% of step time: 0.569 steps/s with them, 4.529
  without (sweeps/README.md:115-117). They are 49% of FLOPs. The effective rate works out to
  ≈ 8.2 TFLOP / 1.76 s ≈ 5 TFLOP/s per tile, so time is dominated by memory traffic and kernel
  efficiency, not arithmetic *(derived)*.

**Memory at B=32, N=150.** One pair channel is 2.9 MB in fp32. `z` is 0.18 GB in fp32, 0.09 GB in
bf16.

| stored for backward (no checkpointing) | size |
|---|---|
| `f` (bf16 copy for the W_c gradient) | 0.36 GB, once |
| write-back `a⊗b` `(B,N,N,c_o²)` | 0.37 GB bf16 (0.74 GB if the fp32 operands survive into the einsum, §6 item 9) |
| triangle mult., each | ~12 z-sized tensors ≈ 2.3 GB (review K93) |
| triangle attn., each (off) | ≈ 3.2·B·N³·H_t·4 B ≈ 5.5 GB. Chunking bounds only no-grad memory (review #2) |
| pair transition | LN in/out plus 4 tensors of width e·c_z ≈ 1 GB *est.* |
| single block | O(B·N·I) + `(B,H,N,N)` ≈ 0.1 GB |
| **10 layers, no checkpointing** | **≈ 45 GB (review #8)** |

- With `gradient_checkpointing`, each `PairLayer` and each single block is its own checkpoint
  segment. Only the boundaries are kept: `z`, the bias `(B,H,N,N)` and `s`, about 0.2 GB per layer
  (2 GB total). The backward then recomputes one pair layer at a time (≈ 6-7 GB). Peak ≈ 10-15 GB
  per tile *est.*, out of 64 GB.
- Weights, gradients and Adam state in fp32: 16 B × 46.1M = 0.74 GB.

## 4. Masking, padding, leak protection, initialisation

**Padding.**
- Triangle multiplication masks `a` and `b` by `m_ij`, so a padded k adds 0 to `Σ_k`.
- Triangle attention adds fp32 min to the logits of padded keys k.
- The write-back zeroes the projections of padded peaks.
- Single attention gives padded keys −∞ (SDPA float mask).
- Padded rows of `s` and `z` are computed but never reach real rows.
- Tests cover padding invariance, invariance to the values in padded slots, and permutation
  equivariance (`tests/test_pairformer.py`, 37 CPU tests).

**Label-leak protection (pretraining).**
- A masked peak's whole token is `mask_token`.
- p6 uses the learned stand-in `c` for it. Without the stand-in, p6 would expose `log I_i`: the
  source measured that leak at skill 0.985.
- When no peaks are masked (fine-tuning), p6 adds `0·c` so that DDP sees the parameter as used.
- **Open leak, both architectures** (review #7): ℓ is normalised by the maximum over all peaks
  before masking. If the base peak is masked, no visible ℓ equals 1.

**Initialisation** (`MSDeltaPreTrainedModel._init_weights`):
- Every `nn.Linear` gets N(0, 0.02) weights and bias 0. That includes all gates (bias 0, so
  σ = 0.5 at init) and all residual output projections.
- LayerNorm: weight 1, bias 0 (affine).
- Zero-initialised: only the write-back output (`ZeroInitLinear`, weight and bias), so the
  write-back starts as a no-op.
- `ScalarInputLinear` (ℓ → h) keeps the PyTorch default init.
- `mask_token` is N(0, 0.02). The stand-in `c` is 0.
- The Fourier frequencies are fixed buffers.
- None of this follows the AF convention of gate bias 1 and zero-initialised final Linears
  (review #11).

## 5. Config fields (`MSDeltaConfig`)

| field | meaning | default | source 50m (`configs/pairformer-sweep-50m/config.json`, source name) |
|---|---|---|---|
| `architecture` | `"transformer"` or `"pairformer"` | transformer | (separate `model_type` `msdelta-pairformer`) |
| `hidden_size` | h | 256 | 512 |
| `num_attention_heads` | H | 8 | 8 |
| `num_hidden_layers` | L | 6 | 10 |
| `intermediate_size` | I, single SwiGLU inner width | 1024 | 2048 |
| `hidden_dropout_prob` | single output and transition dropout | 0.1 | 0.1 |
| `attention_probs_dropout_prob` | single attention-probability dropout | 0.1 | 0.1 |
| `layer_norm_eps` / `initializer_range` | | 1e-5 / 0.02 | defaults: 1e-5 / 0.02 |
| `delta_bias_n_freqs` / `_f_min` / `_f_max` | F and range of the Δm/z bank (p1) and the token m/z bank | 256 / 1e-3 / 190 | 64 / 0.01 / 1000 (learned, log-parameterised in the source) |
| `delta_bias_per_head_hidden` | used by the transformer only | 32 | 32 |
| `pair_channels` | c_z | 16 | 64 |
| `pair_transition_expansion` | e | 2 | 2 |
| `pair_tri_channels` | c_t | 16 | 64 (`tri_channels`) |
| `pair_update` | `static` / `transition` / `triangle` | triangle | triangle |
| `pair_use_triangle_attention` | blocks d-e | False | False (`use_triangle_attention`) |
| `pair_tri_attn_heads` / `_dim` / `_chunk` | H_t / d_t / query-row chunk | 2 / 8 / 32 | 4 / 16 / 32 (`tri_attn_*`) |
| `pair_use_writeback` | block a | True | True (`use_writeback`) |
| `pair_opm_channels` | c_o | 8 | 16 (`opm_channels`) |
| `pair_single_use_mz` | m/z term in the token | True | True (`single_use_mz`) |
| `pair_use_intensity` | p6 | True | True |
| `pair_use_mass_defect` / `pair_mass_defect_n_freqs` | p2 / F_md | True / 16 | True / 32 (`mass_defect_n_freqs`) |
| `pair_use_loss_bank` / `pair_loss_bank_sigma_ppm` | p3 / σ_ppm (also used by p5) | True / 20 | True / 10 (`loss_bank_sigma_ppm`) |
| `pair_use_isotope` | p5 | True | True |
| `pair_dropout` | pDrop on blocks a-e | 0.0 | 0.0 |
| `pair_bias_scale` | tanh cap on β (None = off) | None | None |

- The `pair_*` defaults are test-sized. A run must set every one of them (review #4, K95).
- A transformer config omits `architecture` and all `pair_*` fields from `to_dict()`.
- `pair_bias_scale` cannot be set through `config_overrides` (review #9). It has to go in
  `config.json`.
- Source-only fields, not ported:
  - `delta_bias_learnable`, `fourier_log_parameterized`: learned frequencies.
  - `fourier_int_*` (16 frequencies, 1-100, learned): intensity Fourier token.
  - `use_global_cond`, `global_cond_dim` = 128, `n_charges` = 8: precursor/charge AdaLayerNorm.
  - `pair_use_complementarity`: p4.

## 6. Differences

**vs AlphaFold 3 Pairformer (Alg. 17):**
1. There is an s→z write-back (outer product) at the start of every layer. AF3's Pairformer has
   no s→z path; its outer-product mean lives in the MSA module.
2. Sizes: AF3 has c_s 384, c_z 128, c_t 128, 48 blocks, 16 single heads, triangle attention 4×32
   and always on, transition expansion 4 for both streams. Ours has pair expansion 2, triangle
   attention off, and 10 layers.
3. Dropout: AF3 uses row/column-shared dropout of 0.25 on the triangle updates. Ours is
   elementwise and 0 in the source 50m.
4. Initialisation: AF3 uses gate bias 1 and zero-initialised output Linears. Ours uses gate bias 0
   and N(0, 0.02) outputs, with only the write-back zero-initialised.
5. Single attention: q, k, v, the gate and the output all have biases (AF3: only q). There is an
   optional tanh cap on the pair bias.
6. No recycling, no diffusion/conditioning, no template or MSA inputs.
7. Tokens are a padded, variable-size set of peaks, so everything is masked. z is initialised from
   chemistry features, not from relative-position or bond features.

**vs the source (bc2037b):**
1. No precursor inputs: no p4 and no GlobalCond/AdaLayerNorm. The source 50m had both on.
2. Our LayerNorms therefore have affine weights. The source's were affine-free, with zero-init
   AdaLN (review #5).
3. Token: ours is `MLP(scalar ℓ) + Linear(FF(mz))`. The source's was `MLP([FF_int(ℓ); FF(mz)])`,
   with a learned 16-frequency intensity bank.
4. All Fourier frequencies are fixed in ours; learned and log-parameterised in the source.
5. Selection is by a config field on the same `model_type` (every fine-tune/eval entry point
   works). The source used a separate model class and `--model_class`. Fields are renamed
   (`tri_channels` → `pair_tri_channels`, and so on).
6. The source's defaults were its 50m values (except σ = 20). Ours are test-sized.
7. Checkpointing: two segments per layer in ours (pair layer, single block); one per block in the
   source.
8. p6 keeps the stand-in in the graph when unmasked (DDP). Triangle-attention key masking uses an
   fp32 minimum on fp32 logits.
9. **Precision (unverified on XPU).**
   - Ours casts the pair mask to the dtype of the LayerNorm output: `pair_mask.to(z.dtype)` after
     `z = self.norm(z)`, which is fp32 under autocast. The triangle `a, b` and the write-back
     `a, b` therefore become fp32.
   - The source cast to the bf16 operand dtype ("so the pair branch stays low-precision").
   - If einsum is not autocast on XPU, our triangle einsum and the `(B,N,N,256)` write-back product
     run in fp32. That costs about 2× memory on those tensors and is slower. Check before
     optimising anything else.
10. The source "intrinsic" variant added charge-aware dictionary/isotope features (z = 1..3). They
    are not ported (review #6).
11. Data (processor, not model): the source kept peaks ≥1% of the base peak, then the top 150 by
    intensity. Our processor has no threshold, and it **drops** any spectrum longer than
    `max_peaks` instead of truncating it.

**Review findings** (`PAIRFORMER_REVIEW.md`) and status:

| # | finding | status |
|---|---|---|
| 1 | Non-integer mass-defect frequencies. A defect of −1 mDa and one of +1 mDa look unrelated, which splits CO/CO2/O from H2O/NH3 | inherited; K91 parked |
| 2 | Triangle-attention chunking does not bound training memory (~5.5 GB per module) | K102 (per-chunk checkpointing) open; the feature is off |
| 3 | Older code loads a Pairformer checkpoint as a random transformer, with only a warning | K94 approved (fail on missing keys) |
| 4 | Test-sized defaults | K95: set every field on the card |
| 5 | Absolute m/z in the tokens (the transformer has none); affine LNs | K92: not a problem; control arm optional |
| 6 | Intrinsic charge-aware features not ported | documented |
| 7 | ℓ normalised before masking (both architectures) | K96: later |
| 8 | ~45 GB without checkpointing | checkpointing required |
| 9 | `pair_bias_scale` not settable via overrides | nit |
| 10 | Token m/z clamped at 2000 | nit |
| 11 | Not AF-style gates / init | ablation #10 |
| 12 | Branch based on 14507a8 | rebase before landing |

Cost levers visible in the code:
- N: N² everywhere, N³ in the triangle einsum. The processor cap drops spectra.
- c_z and c_t.
- c_o²: the write-back output Linear is 23.6 GFLOP per layer and the largest activation.
- e.
- Mask/operand dtypes (item 9).
- Pair depth: one pair layer per single layer, and no sharing between layers.
