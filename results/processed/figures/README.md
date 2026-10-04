# Figures

One folder per track (D denoise, C contrastive, A alignment, R reranking). **Current** figures
sit at the top of each folder; `superseded/` keeps older ones for the record. Their numbers
came from an invalid metric, an old recipe or an earlier eval, so don't cite them.
Regenerate the current ones on a compute node: `qsub -q debug-scaling -l select=1 -l walltime=00:15:00 pbs/make_figures.pbs`.

## D_denoise — per-peak noise classification (PLAN D1–D3)
Test split of `ms-denoise-100k`, 3 seeds per point, mean ± sd. The x axis is pretraining grad steps, not compute.
Script: `sweeps/plot_ladder.py`.

| figure | what it shows |
|---|---|
| `ladder_compute.png` | **Headline.** AUROC, F1, AUPRC and test loss vs pretraining steps, one line per model size |
| `ladder_by_scale_{auroc,f1,auprc,loss}.png` | one panel per model size: metric vs pretraining checkpoint |
| `ladder_by_checkpoint_{auroc,f1,auprc,loss}.png` | one panel per checkpoint: metric vs model size |

`superseded/`:
- `scaling.png`, `pretrain_ablation.png`: first scale/scratch plots, made before the checkpoint ladder existed.
- `encoder_lr_scale.png`, `probe_heatmap.png`, `top_cluster.png`: HP-search diagnostics.

## C_contrastive — spectrum embeddings (PLAN C2, C4, C7, C10)
Retrieval on the **ms-contrastive-100k test split**: ~25k experimental-spectrum queries, ~9.9k peptides,
experimental MAP@R and Hit@1. Every panel draws binned cosine (0.1 and 1 Da) and the frozen
encoders' final layer as reference lines. Script: `sweeps/plot_contrastive_100k.py`.

| figure | what it shows |
|---|---|
| `c100k_scale.png` | **C2**: model size at checkpoint 220k (frozen C1 recipe, replicate-corpus training); dashed line = the same encoders frozen, best layer ("before" training) |
| `c100k_checkpoint.png` | **C4**: pretraining checkpoint, 50m and 100m |
| `c100k_c7.png` | **C7**: continuing the best models on ms-contrastive-100k; x = steps into the epoch |
| `c100k_zeroshot_layers.png` | **C10**: frozen encoders, retrieval at every block (left) and best block vs final layer (right) |
| `c100k_zeroshot_abtt.png` | Frozen encoders with all-but-the-top (mean + top-D principal directions, fitted on TRAIN, removed before cosine): left, raw vs ABTT per encoder (final layer / best block) against binned cosine and trained C7; right, best-block MAP@R vs D. Numbers: results/processed/tables/zeroshot-layers-abtt/summary.csv. 400m@220k/430k pending |
| `c100k_transfer.png` | small replicate-corpus eval vs this test, all 69 Stage-1 models (Spearman 0.78) |

**Since submission (K163/K188 consensus recipe: ms-contrastive-100k + consensus spectra, lr 4e-4, P170xK2, every
pretraining checkpoint; same 25,137 test queries and the same reference lines as above).** Script:
`sweeps/plot_contrastive_cons.py`.

| figure | what it shows |
|---|---|
| `c100k_cons_scale.png` | model size at checkpoint 220k (as `c100k_scale.png`) plus the 540k finals incl. 25m |
| `c100k_cons_checkpoint.png` | pretraining grad steps, one line per scale (as `c100k_checkpoint.png`); bottom row zoomed |
| `c100k_cons_filters.png` | experimental MAP@R and library-search Hit@1 with no / 20 ppm / iso-20 ppm precursor filter (no query fails 20 ppm on this split) |
| `c100k_old_vs_new.png` | **old vs new scaling**: new recipe (solid) vs the paper's replicate-corpus C2/C4 curves (dashed) and C7 (stars) |

**All evaluation sets vs baselines** (script `sweeps/plot_contrastive_datasets.py`; sets: validation, test, oodval
[new], mouse 20k, human 20k, yeast full 86k; baselines on the same queries: binned cosine 0.1/1 Da, GLEAMS
(`results/raw/finetune/contrastive/gleams/`), the paper's C7 models as stars).

| figure | what it shows |
|---|---|
| `datasets_scale.png` | MAP@R and Hit@1 vs model size at 540k, one column per set, open search |
| `datasets_checkpoint.png` | MAP@R vs pretraining steps per set (K188), one line per scale |
| `datasets_filters.png` | mouse / human: no filter vs 20 ppm vs iso-20 ppm, all / F (fail 20 ppm) / Fbar queries, vs binned 0.1 Da |

`C1_recipe/` (script `sweeps/plot_c1.py`), on the small replicate-corpus eval:
- `c1_width_temperature.png`: batch width × temperature × epochs grid.
- `c1_scale.png`: 50m vs 400m in that grid.

`superseded/`:
- `contrastive_ladder.png`: old recipe (t 0.07).
- `contrastive_scaling.png`, `contrastive_ablation.png`, `contrastive_breadth.png`: separation-ratio era, a metric C0 showed doesn't predict retrieval.
- `retrieval_vs_separation.png`: the C0 evidence itself.

## P_pretrain — masked-peak pretraining

| figure | what it shows |
|---|---|
| `k195a_losses.png` | K195a-P: today's loss vs proposal_loss (KL to intensity^0.5), 25m debug hours, arm A trains on today's loss, arm B on the proposal, vs Chris's 25m curve; dotted = each loss for a model that fits the other target perfectly (script `sweeps/k195_compare.py`) |

## A_alignment — peptide embeddings
No figures yet (results are in `notes/OBSERVATIONS.md`, A1/A3).

## R_reranking — PSM reranking
No figures yet (results in `results/raw/rerank/psm/*.json` and `notes/OBSERVATIONS.md`).
