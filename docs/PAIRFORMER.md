# Pairformer encoder (`msdelta/model/experiments/pairformer.py`)

An AlphaFold3-style encoder for MS/MS spectra, built as a **controlled modification** of the
existing MSDelta baseline rather than a from-scratch model. It reuses the repo's `FourierFeatures`,
the `--model_class` experiment mechanism, the pretraining objective, the data pipeline, and the
diagnostic callbacks. The design follows the internal spec (AF3 Pairformer, Abramson et al. 2024,
Supplementary Alg. 11/12/13/14/17/24).

## What changed versus the baseline

| | baseline `MSDeltaModel` | `pair_stream` experiments | **Pairformer** |
| --- | --- | --- | --- |
| single `s` carries | intensity only | intensity only | **m/z + intensity** |
| pair `z` carries | Δm/z only (static) | Δm/z only (evolves, pointwise) | **Δm/z + intensity, evolves** |
| z ↔ z mixing | none | none | **triangle multiplication (+ optional attention)** |
| s → z | none | none | **optional outer-product-mean write-back** |
| global conditioning | none | none | **precursor m/z + charge via AdaLayerNorm** |

The two hidden states, refined together at every block:

```
s : (B, N, hidden_size)       one vector per peak
z : (B, N, N, pair_channels)  one vector per peak pair
```

`s_init` = MLP( Fourier(m/z) ++ Fourier(intensity) )  — spec §3
`z_init[i,j]` = W_a·s_i + W_b·s_j + W_c·pair_feats[i,j]  — spec §4, **W_a ≠ W_b** so z is directional.

`pair_feats[i,j]` couples mass and intensity:
- **p1** Fourier of the *signed* Δ_ij = mz_i − mz_j (sign says which peak is the parent)
- **p2** Fourier of the mass defect frac(|Δ|) (distinguishes a real formula difference from a coincidence)
- **p3** soft-match against a neutral-loss / residue / sugar dictionary (Gaussian, σ = ppm→Da at that m/z)
- **p4** complementarity: does mz_i + mz_j sum to the neutral precursor + 2 protons?
- **p5** isotope spacing (1.00336·k for k∈{1,2})
- **p6** relative intensity log(I_i / I_j) — keeps intensity coupling inside z

Each block (AF3 Alg. 17): refine z (write-back → triangle mult out/in → triangle attn → transition),
then read a per-head bias from z and update s with bias-conditioned attention + a SwiGLU transition,
both modulated by the global vector g through zero-initialised AdaLayerNorm.

`z → s` only; `s → z` is off unless `use_writeback` (B5). Information flow otherwise matches AF3.

## The build-phase ladder (spec §10)

Every phase is reachable from one module via config, so the ablation is clean (same data, same
optimizer, same single-stream dims for B1–B3/B5):

| phase | config | meaning |
| --- | --- | --- |
| B1 | `pair_update=static, use_triangle_attention=False, use_writeback=False` | static pair bias (DreaMS / Graphormer baseline) |
| B2 | `pair_update=transition, …=False, …=False` | z evolves by a SwiGLU transition only |
| B3 | `pair_update=triangle, …=False, …=False` | **+ triangle multiplication — the hypothesis** |
| B4 | `pair_update=triangle, use_triangle_attention=True` | + triangle attention (expensive) |
| B5 | `+ use_writeback=True` | + single→pair outer-product-mean |

`config.json` ships **B3 + write-back** as the default trainable model (triangle attention off).

Run the ladder on Polaris:

```bash
for P in B1 B2 B3 B5; do PHASE=$P qsub pbs/pairformer-50m.pbs; done
```

## Memory: triangle attention is the one to watch (spec §12)

Memory, not FLOPs, is the constraint. `z` is `B·N²·c_z`; triangle **attention** additionally
materialises a `(B, chunk, N, N, heads)` logit tensor — the dominant cost. It is chunked over the
query row (`tri_attn_chunk`) to bound that, but at full `max_peaks=150` / micro-batch 64 it will
still OOM. The `PHASE=B4` path in the launcher drops the micro-batch to 8 and caps `max_peaks` to 96.
If it still OOMs, lower the micro-batch before `max_peaks` — trap #1: aggressive top-K removes the
low-intensity intermediate peaks the triangle op needs to route through.

B1–B3 and B5 train comfortably at micro-batch 32 with `--gradient_checkpointing true` (default in the
args file). `torch_compile` stays **off**: K varies per batch and the triangle einsums / chunk loop
recompile on every new shape.

## Diagnostics compatibility

The pair module is exposed as `encoder.bias_module` with the baseline surface:
`.ff` (signed-Δ Fourier bank, for `FourierProbeCallback`) and `.evaluate(grid) → (grid, heads)` (for
`eval/viz.py` and `eval/alignment.py`). `evaluate` reports the **pure-Δ component** of the learned
bias — intensity coupling, the loss dictionary, complementarity, the outer-sum from s, triangle
mixing and write-back are all zeroed — because those only exist on a real N×N spectrum. It is a
faithful probe of the m/z-difference response, not the full bias. `encoder.embed.ff_int` is preserved
for the intensity-frequency probe.

## Global conditioning plumbing

`precursor_mz` and `charge` are computed in preprocessing (`msdelta/data/loading.py`) and are now
emitted by `MSDeltaDataCollatorForPreTraining` for **every** run. The baseline
`MSDeltaForPreTraining.forward` accepts and ignores them (via `**conditioning`); the Pairformer
threads them into `GlobalCond` → AdaLayerNorm. This data is proteomics, so collision energy / adduct
/ instrument (spec §5.5) are not available and are omitted.

## Correctness tests (spec §11)

`tests/test_pairformer.py` — run with `python tests/test_pairformer.py` (no pytest needed) or
`python -m pytest tests/test_pairformer.py`:

- **T1** permutation equivariance — peaks are a set; permuting inputs permutes outputs
- **T2** mask invariance — padded peaks never affect real-peak outputs (catches missing masks in triangle sums)
- **T3** pair asymmetry — z_ij ≠ z_ji (W_a ≠ W_b wiring)
- **T5** shape/dtype smoke + backward
- plus: the whole B1–B5 ladder constructs and runs, `evaluate` returns `(grid, heads)`, and the
  baseline still accepts the new collator keys.

## Not done (deliberate, future work)

- Prepending the precursor as an extra token (spec §2.4) — precursor enters via g and the p4
  complementarity feature instead, avoiding a data-pipeline change.
- Recycling (AF3 uses 4 passes) and the SIRIUS formula pair feature (spec p7, flag-gated arm).
- A parameter-matched comparison against the distance-only baseline: the Pairformer is ~49M vs the
  baseline ~50M, so the intended controlled comparison is across the B1–B5 ladder.
