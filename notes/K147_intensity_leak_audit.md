# K147-P: do masked intensities leak into the model? (audit, 2026-09-30)

Scope: masked-intensity pretraining (K96-S), both `architecture="transformer"` and `"pairformer"`.
Path audited: raw spectrum -> `MSDeltaProcessor._process_one` -> `MSDeltaDataCollatorForPreTraining`
-> `MSDeltaForPreTraining` (encoder -> `IntensityHead` -> KL loss). No model or processor code was
changed. Tests: `tests/test_intensity_leak.py` (tiny random-init CPU models, eval mode, every
parameter randomised so zero-init paths are live, real processor + collator, fixed mask; compares
every block output, every Pairformer pair state `z` and bias, the final hidden state and the logits
bit for bit). A mutation check (disabling the stand-in, or the mask token) makes the tests fail.

## Verdicts

| # | Path | Verdict | Evidence |
|---|------|---------|----------|
| P1 | Max-normalisation `x = log1p(I) / max_all log1p(I)` in the processor, before masking | **LEAKS** when the base peak is masked (below-max masked peaks: no effect) | `test_masked_base_peak_rescales_visible_inputs`, `test_masked_base_peak_does_not_change_outputs[*]` (xfail strict, all 3 archs) |
| P2 | Intensity token (`PeakEmbed`, `PairformerPeakEmbed`): masked token -> mask token (Pairformer: whole token incl. m/z Fourier) | no leak | `test_masked_peak_below_max_does_not_change_any_output[*]`, `test_masked_input_values_are_ignored_even_if_arbitrary[*]` |
| P3 | Pairformer relative-intensity pair feature `visible_i - visible_j` | no leak of the masked value (stand-in replaces it); carries P1 through visible-visible pairs | `test_pair_relative_intensity_ignores_masked_peaks`, `test_pair_features_leak_through_the_max_only_via_visible_rescaling` |
| P4 | Other pair features (signed-Δ Fourier, loss bank, isotope, mass defect): m/z only | no leak | covered by P2 tests with all features on (`pairformer_all`) |
| P5 | `z_init` outer sum `W_a s_i + W_b s_j`, OPM write-back, triangle mult./attention, pair bias | no leak (built from masked `s` and masked features only) | P2 tests (hooks on each pair layer's `z` and bias) |
| P6 | Transformer `DeltaMZBias` (m/z only) | no leak | P2 tests (`delta_bias` hooked) |
| P7 | Labels (`I / sum I`, uses the masked intensities) | only the loss; never the logits | `test_labels_do_not_reach_the_logits[*]` |
| P8 | Padding / batch: no cross-spectrum statistic (LayerNorm per token, attention per spectrum) | no leak, even of P1 across spectra | `test_other_spectra_in_the_batch_never_see_a_masked_base_peak[*]` |
| P9 | Mask draw: `randperm(n)[:round(0.5 n)]`, depends on peak count only | no leak | `test_mask_draw_does_not_depend_on_intensity` |
| P10 | Peak order / position: no positional encoding (permutation-equivariant); raw data is m/z-sorted | no leak | `test_peak_order_carries_no_information[*]` |
| P11 | Peak-count / cap150 selection (spectra >150 peaks dropped, not truncated) | not intensity-dependent | code read: processor raises on >max_peaks, `_preprocess_example` empties the row, `build_preprocessed_dataset` filters it |

Other per-spectrum statistics: the only one computed on the input side is the P1 max. `labels` uses
the sum (loss only). `log_tic`, `charge`, `precursor_mz` are stored in the preprocessed rows but the
pretraining collator drops them (they never reach the model).

## Precursor inputs
Neither architecture takes precursor m/z or charge: `MSDeltaModel.forward` accepts only
`(mz, log_intensity, attention_mask, mask_positions)` and the collator passes only those. The
Pairformer port dropped the source branch's precursor complementarity feature and precursor/charge
conditioning (module docstring), so it already follows the transformer's decision (no precursor).

## Pairformer defaults (configs/stage0/pairformer/config.json, same as the code defaults)
Pair features on: signed-Δm/z Fourier (256 freqs), neutral-loss/residue bank (σ 10 ppm), 13C isotope
(k=1,2), relative intensity (with stand-in). Off: mass defect, triangle attention. Single token:
intensity MLP + Fourier(m/z) (`pair_single_use_mz`). Write-back on, `pair_update="triangle"`.

## Size of the P1 leak (real data, CPU, no model)
2,000 spectra sampled from stage0-cap150 `preprocessed/validation` (67,933 rows; 22-150 peaks,
median 112), 50 mask draws each at mask_ratio 0.5:
- Base peak masked in **50.0%** of draws; no tied base peaks in the sample.
- "No visible x == 1.0" identifies that the base peak is masked with **100% accuracy** (0 false
  positives / negatives): 1 bit per spectrum, every spectrum.
- It also reveals the base peak's size: when flagged, the tallest visible x has median 0.960 (IQR
  0.930-0.981), i.e. log1p(I_max) is ~4% above log1p(I_top_visible) -- on this intensity scale
  (log1p ~ 10-14) roughly a 1.3-1.7x ratio, which matches the true base/second-peak ratio (median
  1.32, IQR 1.13-1.66). All visible x are rescaled by the same factor.
- The masked base peak holds a median **20%** (IQR 15-27%) of the masked target mass, the largest
  single share; the flag says "one masked peak is the biggest, about this much bigger than the
  tallest visible one" but not which masked peak it is (m/z context has to supply that).
How much the models exploit it is not measured here (needs trained checkpoints / a probe).

## Fix options (none applied)
1. Normalise over visible peaks only (`x = log1p(I) / max_visible log1p(I)`, computed after the
   mask is drawn, i.e. in the collator). Removes P1 fully; tests (b) then XPASS -> flip them.
2. Drop max-normalisation: absolute `log1p(I)` (or a fixed global scale). Also leak-free, changes
   the input distribution more.
3. Keep as is and caveat (both archs share the leak equally).
Any change alters the pretraining task. The existing transformer checkpoints were trained WITH the
leak, so a transformer-vs-Pairformer comparison must use the same normalisation in both arms
(either both leaky, as now, or both retrained with the fix); the loss values are not comparable
across normalisations. The same processor also feeds fine-tuning/eval (no masking there, so no leak
there; but option 1 would change fine-tuning inputs only if applied outside the pretraining
collator -- keep it in the collator).
