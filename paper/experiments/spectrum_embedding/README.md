# Spectrum embeddings: transfer, pretraining and scale

Contrastive fine-tuning of the pretrained msdelta encoder so that spectra of the same peptide (and
charge) embed close together; evaluated by spectrum-to-spectrum retrieval.

## Files

| file | content |
|---|---|
| `C_transfer.png`, `C_transfer.csv` | our models vs GLEAMS and binned cosine, in-distribution and on an unseen dataset |
| `C_transfer_ours.png`, `C_transfer_ours.csv` | our models on the in-distribution test set vs an unseen dataset |
| `C_pretraining_scaling.png`, `C_pretraining_scaling.csv` | retrieval vs pretraining checkpoint and model size |
| `C_pretraining_ablation.png`, `C_pretraining_ablation.csv` | same contrastive training from a random vs the pretrained encoder |
| `plot_spectrum_embedding.py` | regenerates all four figures from the CSVs (`python plot_spectrum_embedding.py`; matplotlib + numpy) |
| `0_shot/` | frozen pretrained encoders, no contrastive training (own README) |
| `benchmarks/` | comparison with GLEAMS and binned cosine on four benchmarks (own README) |

## Metric and data

- **MAP@R** over experimental spectra: each spectrum queries all others, relevant = same modified
  peptide and charge, cosine similarity (MAP@100 and Hit@1 for the ablation, see below).
- **ms-contrastive-100k test** (in-distribution): 25,137 experimental query spectra.
- **Yeast 20k subset** (unseen, high-resolution): 20,019 spectra of the yeast test split of the
  nine-species benchmark (InstaDeepAI/ms_ninespecies_benchmark), groups of >= 2 and capped at 20
  spectra; neither our models nor GLEAMS were trained on it.

## C_transfer_ours

| model | what it is | in-distribution | yeast 20k |
|---|---|---|---|
| fine-tuned 400M | stage 1: replicate corpus (`chrisagrams/ms2-peptide-replicate-retrieval`, 12 epochs); stage 2: one epoch of ms-contrastive-100k; from pretraining checkpoint 220k | 0.868 | 0.596 |
| fine-tuned 50M | same, except stage 1 ran 24 epochs | 0.839 | 0.523 |
| replicate corpus only 400M | the 400M stage-1 model alone (12 epochs); 3 seeds (1 on yeast) | 0.713 | 0.520 |
| replicate corpus only 50M | the 50M stage-1 model alone (24 epochs); 3 seeds (1 on yeast) | 0.656 | 0.499 |
| frozen + ABTT (best encoder) | no contrastive training; see `0_shot/` | 0.432 (200M@540k) | 0.709 (400M@220k) |

- The fine-tuned models are one seed each: the end-of-epoch seed with the best **validation** MAP@R
  on ms-contrastive-100k (400M: seed 0, 0.8639 vs 0.8629 / 0.8634; 50M: seed 1, 0.8357 vs 0.8314 / 0.8342).
  Another 400M seed scores higher on yeast (seed 1: 0.656); it was not selected because that would use
  the test set.
- Frozen + ABTT: the encoder, layer and number of removed components are the best on each benchmark,
  so these bars are upper bounds for that method.
- Reading: fine-tuning gives the best in-distribution retrieval by a wide margin, but on unseen data
  the frozen encoder with ABTT transfers better than any fine-tuned model.

## C_transfer

The `C_transfer_ours` models (same runs, same seed selection, same frozen + ABTT bars) next to the two
baselines, on the two benchmarks where every method was scored.

- **GLEAMS**: the published pretrained model, cosine similarity, no retraining.
- **binned cosine**: 1 Da and 0.1 Da bins are both in the CSV; the bar is the better of the two on that
  benchmark (tagged `[plotted: best width]`), which favours the baseline.
- "fine-tuned 400M/50M (replicate corpus only)" are the models called "replicate corpus only" elsewhere
  in this folder: the first contrastive stage alone, without the ms-contrastive-100k epoch.

| model | ms-contrastive-100k | yeast 20k |
|---|---|---|
| fine-tuned 400M | **0.868** | 0.596 |
| fine-tuned 50M | 0.839 | 0.523 |
| fine-tuned 400M (replicate corpus only) | 0.713 | 0.520 |
| fine-tuned 50M (replicate corpus only) | 0.656 | 0.499 |
| frozen + ABTT (best encoder) | 0.432 | 0.709 |
| GLEAMS | 0.646 | 0.770 |
| binned cosine (best width) | 0.729 | **0.916** |

- Reading: in-distribution our fine-tuned models lead every baseline; on the unseen yeast data both
  baselines are ahead of every model of ours, binned cosine by a wide margin.

## C_pretraining_scaling

- Recipe: replicate corpus only, SupCon loss, lr 1e-4, KL 10, temperature 0.002, 64 peptide groups x 4
  replicates per batch, 24 epochs; 3 seeds per point; evaluated on the ms-contrastive-100k test.
- 50M and 100M across pretraining checkpoints 10k-540k; 200M and 400M at 220k only.
  Jobs 8860092 / 8860863 (sizes at 220k) and 8860093 (checkpoints).
- Reading: MAP@R rises steeply from 10k to 120k pretraining steps (50M: 0.447 -> 0.627) and flattens
  after about 220k; 100M is at or above 50M at every checkpoint; 400M (0.703) is above the others at 220k.

## C_pretraining_ablation

- Identical contrastive training started from the pretrained encoder or from a random initialisation
  (verified per run in the training configuration), 6 seeds per point, each size at its original
  pretraining checkpoint (50M 133k, 100M 138k, 200M 193k, 400M 181k).
- **Different recipe and test set from the other figures**: an earlier short recipe (lr 1e-4, KL 10,
  temperature 0.07) scored on the small replicate-corpus evaluation set with MAP@100 and Hit@1. The
  numbers are therefore not comparable with the MAP@R values above; the figure stands on its own.
- Jobs 8853557 and 8854412 (pretrained), 8853558 and 8854760 (random init); the later job in each pair
  retried runs of the earlier one.
- Reading: from a random initialisation, contrastive training stays at chance at every size
  (MAP@100 about 0.01, Hit@1 about 0.03), against 0.34-0.41 and 0.75-0.78 from the pretrained encoder.

Provenance: msdelta repository, `sweeps/package_contrastive.py` read the per-run result files and wrote the CSVs.
