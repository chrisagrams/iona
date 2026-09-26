# Denoising: test loss vs pretraining (loss scaling)

The denoising test loss of every fine-tuned model in the pretraining ladder, against the number of
pretraining steps of its encoder, with a power-law fit.

## Files

| file | content |
|---|---|
| `D_denoise_loss_scaling.png` | left: test loss vs pretraining steps per model size, with the fit (dotted); right: reducible loss L − E_N on log-log axes |
| `D_denoise_loss_scaling.csv` | the plotted data: one row per (model size, pretraining checkpoint) |
| `plot_denoise_loss_scaling.py` | fits and regenerates the figure from the CSV (`python plot_denoise_loss_scaling.py`; matplotlib, numpy, scipy) |

CSV columns: `scale`, `parameters` (exact count, from the checkpoint), `pretraining_steps`, `n_seeds`,
`mean_test_loss`, `se_test_loss` (standard error over seeds), `per_seed_test_loss` (space-separated).

## Setup

- **Data and recipe**: the same runs as `../D_denoise.png` (`chrisagrams/ms-denoise-100k`, fine-tuning
  recipe lr 2e-4, encoder learning rate 0.5x, 4 epochs, head width 512, effective batch 12), 29 points:
  4 sizes x 7-8 pretraining checkpoints (10k to 540,423 steps), 3-13 seeds each.
- **Metric**: per-peak binary cross-entropy on the test split (`test_loss`), as logged at the end of
  fine-tuning.
- **Pretraining steps as the data axis**: every size was pretrained with the same global batch for the
  same 540,423 steps (3 epochs), so steps are proportional to pretraining data seen.

## Fit

L(N, S) = E_N + B · (S / 10⁵)^−β, one floor E_N per model size and one power law in pretraining steps S
shared by all sizes; weighted least squares (weights: standard error over seeds, floored at 5e-4).

| | value |
|---|---|
| β | 0.46 ± 0.04 |
| B | 0.021 |
| E_N (50M / 100M / 200M / 400M) | 0.3125 / 0.2955 / 0.2892 / 0.2906 (each ± 0.002) |
| R² | 0.997 |
| held out | fitted without the 540k checkpoint, the fit predicts it to within 0.002 |

- Reading: denoising loss decreases as a power law in pretraining steps with one exponent shared by all
  sizes. The asymptotic loss E_N improves up to 200M and not beyond (200M and 400M are equal within
  error).
- Why this form: a model with a power law in model size as well (E + A·N^−α + B·S^−β) fits worse
  (χ²/dof 5.5 vs 3.2) and needs an implausible α ≈ 2, because the loss saturates in size; one power law
  in compute (N × S) through all points fits poorly (R² 0.92, systematic by size).

## Caveats

- The right panel is partly circular: the floors E_N come from the same fit, so the straight line is in
  part a consequence of it. The independent check is the held-out prediction of the final checkpoint.
- These are the losses logged at the end of training; models reloaded from saved weights evaluate
  slightly worse (see `../README.md`). Points are comparable with each other.
- The 400M points at the last three checkpoints sit slightly above the shared line (within about 2 SE).

Provenance: msdelta repository, `sweeps/plot_denoise_loss_scaling.py` (reads the run directories via
`sweeps/plot_ladder.py` and writes the CSV).
