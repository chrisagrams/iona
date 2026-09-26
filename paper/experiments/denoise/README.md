# Denoising: scaling and pretraining

Fine-tuning the pretrained msdelta encoder to classify each MS2 peak as signal or noise, across
model size (50M, 100M, 200M, 400M) and pretraining checkpoint (10k to 540k steps), against the same
architecture trained from a random initialisation.

## Files

| file | content |
|---|---|
| `D_denoise.png` | test AUROC vs model size; one line per pretraining checkpoint, plus from scratch |
| `denoise_scaling_pretraining.csv` | the plotted data: one row per (scale, pretraining steps), with every seed's AUROC and AUPRC |
| `plot_denoise_scaling.py` | regenerates the figure from the CSV (`python plot_denoise_scaling.py`; matplotlib + numpy) |
| `HP/` | the hyperparameter search that fixed the fine-tuning recipe (own README) |

CSV columns: `scale`, `pretraining_steps` (0 = from scratch), `n_seeds`, `mean_auroc`, `sd_auroc`
(blank for a single run), `min_auroc`, `max_auroc`, `per_seed_auroc` (space-separated), `mean_auprc`, `sd_auprc`, `per_seed_auprc`
(same seeds, same order as `per_seed_auroc`).

## Setup

- **Data**: `chrisagrams/ms-denoise-100k`, spectra up to 512 peaks. Test split: 8,567 spectra scored.
- **Metric**: per-peak noise classification AUROC on the test split, pooled over peaks. AUPRC is also
  reported (CSV only), pooled over peaks with noise as the positive class, computed as the trapezoidal
  area under the precision-recall curve. Noise is about 53% of test peaks, so a random classifier
  scores AUPRC of about 0.53 (not 0.5).
- **Fine-tuning recipe (every pretrained point)**: lr 2e-4, encoder learning rate 0.5x the head's,
  4 epochs, head width 512, effective batch 12. This configuration won the hyperparameter search at
  every scale (see `HP/`).
- **Pretrained**: the encoder is initialised from `msdelta-<size>-production-01-checkpoint-<steps>`.
  3 seeds per point (5 at 50M / 540k). Seeds change head initialisation, data order and dropout;
  the split is fixed. Jobs 8848874, 8850494, 8856549, 8860472, 8861571, 8864759.
- **From scratch**: same architecture, random encoder, lr 2e-4, encoder learning rate 1.0x (nothing
  to preserve), 4 epochs, effective batch 12. 100M-400M: 3 seeds (job 8847663, resumed as 8856558).
  50M: a single run (job 8841984, configuration `lr2e4_ep4_b12`).

## Results (mean test AUROC)

| scale | scratch | 10k | 120k | 220k | 330k | 430k | 540k |
|---|---|---|---|---|---|---|---|
| 50M | 0.882 | 0.910 | 0.931 | 0.934 | 0.936 | 0.936 | 0.936 |
| 100M | 0.889 | 0.919 | 0.939 | 0.942 | 0.943 | 0.943 | 0.943 |
| 200M | 0.896 | 0.923 | 0.943 | 0.945 | 0.946 | 0.946 | 0.946 |
| 400M | 0.898 | 0.923 | 0.941 | 0.944 | 0.944 | 0.945 | 0.945 |

- Pretraining adds about +0.05 AUROC at every size. 10k steps (2% of pretraining) already recovers
  most of it; gains flatten after about 330k steps.
- AUROC rises with size up to 200M. 400M is slightly below 200M from 120k steps on (0.9449 vs 0.9458
  at 540k); longer fine-tuning at 400M does not recover it.
- Seed-to-seed sd is about 0.0005, so differences of 0.001 are resolved.

## Caveats

- 50M from scratch is one run, and its job scored 8,584 test spectra rather than 8,567 (an earlier
  evaluation of the same split); the point has no error bar.
- 50M at 10k steps has a large spread (one seed 0.901, two at 0.915); it is real seed variance.
- These are the AUROCs logged at the end of training. Models reloaded from saved weights currently
  re-evaluate about 0.014 lower (50M: 0.9227 vs 0.9366); the cause is under investigation. All points
  in the figure use the same logged evaluation, so they are comparable with each other.

Provenance: msdelta repository, `sweeps/plot_summary.py` (`denoise_cells`) read the run directories and
wrote the CSV.
