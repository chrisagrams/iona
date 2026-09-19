# Fine-tuning TODO

Open work on `dev_finetune`. Items are ordered by what would change a decision, not by
effort.

## FT1. Does the denoiser generalise beyond 1024 peaks? — **Open**

`max_peaks=1024` and `build_denoising_datasets` DROPS anything above it rather than
truncating, so 1.62% of train, 1.58% of validation and 1.71% of test never reach the
model. Those are not a random 1.6%: peak count correlates with noise, so the excluded
spectra are the largest and the noisiest — systematically the hardest cases. The reported
test AUROC of 0.9354 is therefore measured on a corpus that is slightly easier than the
real one, and says nothing at all about the regime it excludes.

Two distinct questions, and the second is the interesting one:

  **Coverage.** Retrain at a higher cap so those spectra are included. Costs pair memory
  quadratically — the Fourier tensor over every (i, j) m/z difference is
  `batch * L^2 * 2 * n_freqs`, so 2048 is 4x the memory of 1024 at equal batch and
  `peak_pair_budget` has to fall by the same factor.

  **Extrapolation.** Train at 1024 and evaluate on the >1024 spectra without retraining.
  This is the real generalisation question: the encoder has a learned per-head Δm/z bias
  rather than absolute position embeddings, so there is a genuine reason to expect it to
  extend past its training length — and a genuine reason to doubt it, since the bias MLPs
  have only ever seen the Δm/z distribution of shorter spectra.

Ordering matters: run extrapolation FIRST. It is one evaluation pass on an existing
checkpoint, it needs no training, and if the model already extends cleanly then the
expensive retrain buys nothing.

- [ ] Build the held-out >1024 slice (1,620 train / 159 validation / 169 test spectra)
      and score the existing 50m fine-tune on it with no retraining. Report AUROC/AUPRC
      against the <=1024 test number, and bucket by peak count so any degradation is
      visible as a trend rather than one average.
- [ ] Check whether degradation tracks peak count or noise fraction. Those are
      correlated, so a drop at high peak counts may be the task getting harder rather
      than the model failing to extrapolate. Stratify to separate them.
- [ ] Only if extrapolation fails: retrain at `max_peaks=2048` with
      `peak_pair_budget=262144`, and compare on the SAME >1024 slice.
- [ ] Decide whether `build_denoising_datasets` should truncate instead of drop. It
      currently discards whole spectra; truncating by m/z keeps them but silently removes
      the high-m/z tail, which is where the large fragment ions live.

## FT2. Test metrics never reach W&B — **Open**

Every run's W&B summary carries `eval/*` but `test_auroc` is `None`. The final test
evaluation prints to the PBS log and goes through `trainer.log(...)`, but is not landing
in the summary, so the headline number exists only in a job log. The comparison across
seeds and model sizes is unreadable in W&B until this is fixed.

- [ ] Log the test metrics explicitly to the run summary, not only through `trainer.log`.

## FT3. A crashed job reports `finished` in W&B — **Open**

`wandb_run.finish()` runs in a `finally` block, so a job that died mid-training is
indistinguishable from one that completed. `denoise-ft-50m-8839579` died on the metrics
error and shows `finished`.

- [ ] Mark the run failed when `main()` exits non-zero or raises.

## FT4. 32 stray label values survive the distributed gather — **Open**

`denoise_metrics` drops 32 labels per evaluation that are neither 0, 1 nor -100 —
bf16-quantised floats in the 4.09-5.22 range, the same 32 every time. That is 0.0013% of
2.5M peaks and has no measurable effect, but nothing should be writing those values into
a label buffer. A single-process eval and a two-rank gloo eval are both clean, so it
appears only at twelve ranks with bf16.

- [ ] Find the source. Suspect the padding index used when gathering variable-length
      label tensors across ranks.
