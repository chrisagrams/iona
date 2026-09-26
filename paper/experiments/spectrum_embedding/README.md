# Spectrum embeddings: transfer, pretraining and scale

Contrastive fine-tuning of the pretrained msdelta encoder so that spectra of the same peptide (and
charge) embed close together; evaluated by spectrum-to-spectrum retrieval.

## Files

| file | content |
|---|---|
| `C_transfer.png`, `C_transfer.csv` | our models vs GLEAMS and binned cosine, in-distribution and on three unseen nine-species subsets |
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
| frozen + ABTT (best encoder) | no contrastive training; see `0_shot/` | 0.660 (400M@330k) | 0.709 (400M@220k) |

- The fine-tuned models are one seed each: the end-of-epoch seed with the best **validation** MAP@R
  on ms-contrastive-100k (400M: seed 0, 0.8639 vs 0.8629 / 0.8634; 50M: seed 1, 0.8357 vs 0.8314 / 0.8342).
  Another 400M seed scores higher on yeast (seed 1: 0.656); it was not selected because that would use
  the test set.
- Frozen + ABTT: the encoder, layer and number of removed components are the best on each benchmark
  (of 24 encoders in-distribution, 6 on yeast), so in principle upper bounds; where checked (three 50M
  encoders), validation selection picked the same layer and D.
- Reading: fine-tuning gives the best in-distribution retrieval by a wide margin, but on unseen data
  the frozen encoder with ABTT transfers better than any fine-tuned model.

## C_transfer

The `C_transfer_ours` models (same runs, same seed selection) next to the two baselines, in-distribution
and on three unseen nine-species subsets (yeast, mouse, human). The frozen + ABTT bars are not in this
figure (they were not scored on every benchmark); see `C_transfer_ours` and `0_shot/`.

- **GLEAMS**: the published pretrained model, cosine similarity, no retraining.
- **binned cosine**: 1 Da and 0.1 Da bins are both in the CSV; the bar is the better of the two on that
  benchmark (tagged `[plotted: best width]`), which favours the baseline.
- "Iona spectrum encoder 400M/50M" are the models called "fine-tuned 400M/50M" elsewhere in this folder
  (released as `Gaolaboratory/iona-contrastive-400m` and `-50m`); "(replicate corpus only)" marks the models called
  "replicate corpus only" elsewhere: the first contrastive stage alone, without the ms-contrastive-100k epoch.

| model | ms-contrastive-100k | yeast 20k | mouse 20k | human 20k |
|---|---|---|---|---|
| Iona spectrum encoder 400M | **0.868** | 0.596 | 0.846 | **0.876** |
| Iona spectrum encoder 50M | 0.839 | 0.523 | 0.820 | 0.874 |
| Iona spectrum encoder 400M (replicate corpus only) | 0.713 | 0.520 | 0.787 | 0.816 |
| Iona spectrum encoder 50M (replicate corpus only) | 0.656 | 0.499 | 0.757 | 0.822 |
| GLEAMS | 0.646 | 0.770 | 0.834 | 0.841 |
| binned cosine (best width) | 0.729 | **0.916** | **0.916** | 0.809 |

- **Mouse 20k and human 20k**: the Mus musculus and H. sapiens spectra of the Noble lab nine-species
  benchmark (Zenodo 10.5281/zenodo.12819175, `nine-species-balanced.zip`; Tide + Percolator labels at 1%
  FDR), groups of peptide + charge with >= 2 spectra capped at 20, whole groups sampled (seed 0) to 20,000
  spectra (mouse: 20,003 in 4,026 groups, no spectrum above 512 peaks; human: 20,000 in 4,792 groups,
  240 spectra trimmed to their 512 most intense peaks). Neither overlaps our pretraining or fine-tuning
  data.
- Reading: in-distribution the Iona encoders lead every baseline. On unseen yeast both baselines are
  ahead of every model of ours; on unseen mouse the Iona 400M encoder is level with GLEAMS (0.846 vs
  0.834) and binned cosine leads (0.916); on unseen human both Iona encoders lead (0.876 / 0.874 vs GLEAMS
  0.841 and binned cosine 0.809).

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
