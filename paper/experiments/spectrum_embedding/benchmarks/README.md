# Spectrum embeddings: comparison with GLEAMS and binned cosine

Spectrum-to-spectrum retrieval on one in-distribution and three unseen benchmarks.

## Files

| file | content |
|---|---|
| `C_benchmarks.png` | MAP@R per benchmark for four methods |
| `C_benchmarks.csv` | every plotted value, per seed; binned cosine at both bin widths |
| `plot_benchmarks.py` | regenerates the figure from the CSV (`python plot_benchmarks.py`; matplotlib + numpy) |

CSV columns: `benchmark`, `model`, `scale`, `pretrain_ckpt`, `finetune_stage`, `seed`, `map_at_r`,
`hit_at_1`, `note`. For binned cosine the `note` gives the bin width; the plotted one (the better of
the two on that benchmark) is tagged `[plotted: best width]`.

## Methods

- **ours: C7 400M**: the fine-tuned 400M model (replicate corpus 12 epochs, then one epoch of
  ms-contrastive-100k, from pretraining checkpoint 220k); mean ± sd over its 3 seeds.
- **ours: replicate corpus only 400M**: the first stage alone (12 epochs); 3 seeds (1 on yeast 20k).
- **GLEAMS**: the published pretrained GLEAMS model, cosine similarity, no retraining.
- **binned cosine**: cosine similarity of spectra binned by m/z, with 1 Da (1.0005 Da) or 0.1 Da bins.

## Benchmarks

| benchmark | queries | notes |
|---|---|---|
| ms-contrastive-100k test | 25,137 | in-distribution for ours |
| HEK | 27,637 | confident (1% FDR) MSFragger PSMs from the HEK runs of `psm-rerank-hek-hct116`; low-resolution (ion-trap) MS2; spectra up to 512 peaks, groups capped at 20 |
| nine-species yeast | 86,184 | yeast test split of InstaDeepAI/ms_ninespecies_benchmark; high-resolution; groups of >= 2, capped at 20 |
| yeast 20k subset | 20,019 | a 20,000-spectrum whole-group sample of the above |

Metric: MAP@R over experimental spectra; relevant = same modified peptide and charge.

## Results (MAP@R)

| benchmark | ours C7 400M | replicate only 400M | GLEAMS | binned cosine |
|---|---|---|---|---|
| ms-contrastive-100k | **0.868** | 0.713 | 0.646 | 0.729 |
| HEK | 0.170 | 0.017 | 0.530 | **0.554** |
| nine-species yeast | 0.518 | 0.474 | 0.676 | **0.790** |
| yeast 20k subset | 0.600 | 0.520 | 0.770 | **0.916** |

- In-distribution, our fine-tuned model is clearly best. On unseen data, GLEAMS and binned cosine
  are better. HEK's low-resolution fragment spectra are the largest domain shift for our models.

## Caveats

- Binned cosine is shown at its better bin width per benchmark, which favours the baseline.
- "C7 400M" here is the mean of 3 seeds; `../C_transfer_ours.png` shows the single validation-selected seed.

Provenance: msdelta repository, `sweeps/package_contrastive.py` (our results from
`results/raw/finetune/contrastive/<benchmark>/`, GLEAMS from its per-benchmark metrics files).
