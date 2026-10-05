# Summary tables (2026-09-25)

Source: notes/OBSERVATIONS.md (job ids there). Figures: results/figures/SUMMARY/.

## D: denoise

| scale | pretrained + fine-tuned | from scratch | gain |
|---|---|---|---|
| 50m | 0.9317 | 0.8821 | +0.050 |
| 100m | 0.9400 | 0.8886 | +0.051 |
| 200m | 0.9447 | 0.8963 | +0.048 |
| 400m | 0.9434 | 0.8981 | +0.045 |

Pretrained: mean ± sd over 6 seeds. Scratch: 3 seeds at 100m–400m; 50m is a single run at the same config (lr 2e-4, eff. batch 12, 4 epochs).
Caveat: the Hub cards show training-logged numbers; reloaded models re-evaluate ~0.014 AUROC lower (unresolved).

## C: spectrum retrieval (experimental MAP@R)

| dataset | ours C7 400m (3 seeds) | ours replicate-only | frozen + ABTT | GLEAMS | binned cosine |
|---|---|---|---|---|---|
| ms-contrastive-100k (in-distribution) | 0.868 (0.868–0.869) | 0.713 (0.711–0.714) | — | 0.646 | 0.730 |
| HEK (unseen, low-res MS2) | 0.170 (0.167–0.173) | 0.017 (0.016–0.018) | — | 0.530 | 0.554 |
| nine-species yeast (unseen, high-res) | 0.518 (0.475–0.565) | 0.474 (0.444–0.503) | — | 0.676 | 0.790 |
| yeast 20k subset (unseen, high-res) | 0.600 (0.550–0.656) | 0.521 (0.521–0.521) | 0.709 | 0.770 | 0.916 |

| scale | C2 (replicate corpus) | C7 (+ms-contrastive-100k) | zero-shot raw → ABTT (100k test) | zero-shot raw → ABTT (unseen yeast) |
|---|---|---|---|---|
| 50m | 0.657 | 0.839 | 0.208 → 0.401 | 0.385 → 0.602 |
| 100m | 0.665 | — | 0.140 → 0.328 | 0.178 → 0.421 |
| 200m | 0.658 | — | 0.191 → 0.432 | 0.339 → 0.654 |
| 400m | 0.703 | 0.868 | 0.216 → 0.414 | 0.510 → 0.709 |

Pretraining (C3): random init trained contrastively stays at chance at every scale.

## A: peptide embeddings

| model | test Hit@1 |
|---|---|
| A1 (teacher C7 50m) | 0.898 |
| A4 (A1 + LiT/hard neg.) | 0.895 |
| A-oodsel (teacher C7 400m s600, OOD-selected) | 0.918 |
| A2 (teacher C7 400m final) | 0.923 |

| dataset | window | yHydra | A1 | A2 | A-oodsel |
|---|---|---|---|---|---|
| in-distribution (ms-contrastive-100k) | open | 0.196 | 0.901 | 0.925 | 0.921 |
| in-distribution (ms-contrastive-100k) | ±1.1 Da | 0.754 | 0.973 | 0.979 | — |
| in-distribution (ms-contrastive-100k) | 20 ppm | 0.942 | 0.993 | 0.994 | 0.994 |
| HEK (unseen, low-res) | open | 0.016 | 0.061 | — | — |
| HEK (unseen, low-res) | ±1.1 Da | 0.345 | 0.601 | — | — |
| HEK (unseen, low-res) | 20 ppm | 0.606 | 0.679 | — | — |
| nine-species yeast (unseen, high-res) | open | 0.060 | 0.253 | 0.388 | 0.410 |
| nine-species yeast (unseen, high-res) | ±1.1 Da | 0.650 | 0.537 | 0.639 | 0.659 |
| nine-species yeast (unseen, high-res) | 20 ppm | 0.890 | 0.691 | 0.761 | 0.765 |

## R: reranking (PSMs at 1% FDR)

| method (8 runs) | PSMs |
|---|---|
| MSFragger (e-value) | 89,693 |
| ours global MLP, MSFragger feat. | 93,206 |
| ours global MLP, lab feat. | 104,785 |
| MS2Rescore search-only | 100,974 |
| ours per-run, MSFragger feat. | 99,154 |
| MS2Rescore full (MS2PIP+DeepLC) | 128,211 |
| MS2Rescore full + A2 emb. | 128,909 |
| ours per-run, lab feat. | 129,041 |
| ours per-run, lab + A2 emb. | 129,618 |

| base (per-run, 8 runs) | embedding | real − null per seed | mean |
|---|---|---|---|
| lab (8 runs) | A1 | +398 / +738 / +473 | +536 (0.42%) |
| lab (8 runs) | A2 | +564 / +924 / +544 | +677 (0.53%) |
| lab (8 runs) | A-oodsel | +901 / +912 / +572 | +795 (0.62%) |
| MSFragger (8 runs) | A1 | +1,495 / +1,215 / +1,186 | +1,299 (1.31%) |
| MSFragger (8 runs) | A2 | +2,705 / +2,373 / +2,386 | +2,488 (2.51%) |
| MSFragger (8 runs) | A-oodsel | +2,096 / +1,912 / +1,872 | +1,960 (1.97%) |

| HCT116 ×18 (unseen, high-res), A2, per-run, seed 0 | PSMs |
|---|---|
| MSFragger (e-value) | 330,139 |
| per-run, MSFragger feat. | 370,982 |
| per-run, MSFragger + A2 emb. | 392,350 |
| per-run, MSFragger + null | 371,587 |
| per-run, lab feat. | 522,539 |
| per-run, lab + A2 emb. | 530,977 |
| per-run, lab + null | 522,353 |

HCT116: lab + A2 real − null = +8,624 (+1.65%); MSFragger features + A2 real − null = +20,763 (+5.60%). Leakage AUROC 0.519 (cosine), 0.499 (null).
Controls: shuffled labels → 0 PSMs; per-run fold seeds 128,991–129,041 (lab); MS2Rescore repeat 128,211 vs 128,209.

