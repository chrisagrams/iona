# Denoising: hyperparameter search (50M)

A full grid over five fine-tuning hyperparameters for the 50M model, used to fix the recipe for all
denoising experiments.

## Files

| file | content |
|---|---|
| `D_hp_parallel.png` | parallel-coordinates plot: one axis per hyperparameter, one line per configuration, coloured by test AUROC; best in red |
| `D_hp_parallel.csv` | all 216 configurations with their test AUROC, sorted best first |
| `plot_hp_parallel.py` | regenerates the figure from the CSV (`python plot_hp_parallel.py`; matplotlib + numpy) |

CSV columns: `arm` (configuration name), `learning_rate`, `encoder_lr_scale` (encoder learning rate as a
fraction of the head's; 0 = frozen encoder), `num_train_epochs`, `head_hidden_size`, `eff_batch`
(effective batch size), `test_auroc`, `test_spectra_scored`.

## Setup

- **Model**: 50M encoder from pretraining checkpoint 133,233, fine-tuned for per-peak noise classification
  on `chrisagrams/ms-denoise-100k`.
- **Grid** (3 x 4 x 2 x 3 x 3 = 216): learning rate {1e-6, 1e-5, 2e-4}; encoder LR scale {0 (frozen),
  0.1, 0.5, 1.0}; epochs {2, 4}; head width {128, 256, 512}; effective batch {12, 48, 144}.
- **Metric**: test AUROC (8,584 test spectra), one run per configuration. Job 8840408.

## Results

- Best: lr 2e-4, encoder 0.5x, 4 epochs, head 512, effective batch 12, AUROC 0.9320.
- Learning rate and a trainable encoder matter most: every configuration above 0.90 has lr 2e-4 (or
  3 at lr 1e-5 with encoder 1.0x); every frozen-encoder and every lr 1e-6 run is below 0.90 (168 of 216
  runs, drawn grey). 4 epochs beat 2. Head width barely matters.

## Caveats

- Each configuration is a single run. The top 8 configurations lie within 0.0023 of each other, while
  seed-to-seed sd is about 0.0005, so the ordering within that cluster is not resolved. The chosen
  configuration is supported independently: it also ranked first in the separate 100M, 200M and 400M
  grids (12 configurations each), and again when the 50M grid was repeated at pretraining checkpoints
  1 and 540,423.
- The grey lines' vertical positions carry a small fixed jitter so that overlapping configurations
  stay visible; the axis values are exact.

Provenance: msdelta repository, `results/finetune/denoise/grid_denoise_50m.txt` (the committed grid table)
via `sweeps/plot_hp_parallel.py`.
