# Denoising: test loss vs pretraining (loss scaling)

The denoising test loss of every fine-tuned model in the pretraining ladder, against the number of
pretraining steps of its encoder, with a power-law fit.

## Files

| file | content |
|---|---|
| `D_denoise_loss_scaling.png` | left: test loss vs pretraining steps per model size, with the fit (dotted); right: reducible loss L − E_N on log-log axes |
| `D_denoise_loss_scaling.csv` | the plotted data: one row per (model size, pretraining checkpoint) |
| `plot_denoise_loss_scaling.py` | fits and regenerates the figure from the CSV (`python plot_denoise_loss_scaling.py`; matplotlib, numpy, scipy) |
| `fit_alternatives.py` | fits every functional form tried (below) to the same CSV and writes `fit_comparison.csv` and one figure per alternative (`python fit_alternatives.py`) |
| `fit_comparison.csv` | all four forms: parameters, standard errors, R², RMSE, χ²/dof, held-out errors |
| `D_loss_fit_chinchilla.png`, `D_loss_fit_compute.png`, `D_loss_fit_pure_power.png` | the three alternatives that were not used, each against the data |

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
- Every form tried (`fit_comparison.csv`; weighted least squares on the same 29 points):

| form | R² | RMSE | χ²/dof | held out 540k (max err) | held out 400M (max err) |
|---|---|---|---|---|---|
| **per-size floor + shared step power law** (used) | **0.997** | **0.0011** | **3.2** | **0.0020** | – (floor is per size) |
| E + A·N^−α + B·S^−β (Chinchilla form) | 0.995 | 0.0014 | 5.5 | 0.0029 | 0.0056 |
| E + A·(N·S)^−γ (one curve in compute) | 0.924 | 0.0053 | 67 | 0.0074 | 0.0126 |
| A·N^−α·S^−β (no floor) | 0.908 | 0.0058 | 70 | 0.0097 | 0.0163 |

- Why this form: the Chinchilla form fits the size axis with an implausible α ≈ 2.1 because denoising
  loss saturates in model size (200M and 400M are equal within error), so a power law in N does not
  hold; one curve in compute misses systematically by size (at equal compute 100M/200M sit below it,
  50M/400M above); dropping the floor fails outright. The per-size floor makes no claim about N and
  isolates the part that is a power law: pretraining steps.

## Caveats

- The right panel is partly circular: the floors E_N come from the same fit, so the straight line is in
  part a consequence of it. The independent check is the held-out prediction of the final checkpoint.
- These are the losses logged at the end of training; models reloaded from saved weights evaluate
  slightly worse (see `../README.md`). Points are comparable with each other.
- The 400M points at the last three checkpoints sit slightly above the shared line (within about 2 SE).

Provenance: msdelta repository, `sweeps/plot_denoise_loss_scaling.py` (reads the run directories via
`sweeps/plot_ladder.py` and writes the CSV).
