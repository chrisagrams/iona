# Fine-tuning plan

The one file that says **what we are trying to find out**. `STATUS.md` says where each
thing stands, `OBSERVATIONS.md` records results, `TODO.md` lists defects. Every job must
name the question below that it answers; a job that answers none should not be queued.

## Goal

Two conclusions, for two downstream tasks:

- **Scaling** — does performance improve with model size, and with pretraining amount?
- **Pretraining usefulness** — how much does pretraining buy over training from scratch?

Tasks: **denoise** (per-peak noise classification, metric AUROC/F1) and **contrastive**
(spectrum embeddings for retrieval/reranking, metric MAP@R with Precision@1 and
R-Precision beside it — Musgrave et al., ECCV 2020).

Axes: model size 50m/100m/200m/400m × canonical checkpoints
10000 / 120000 / 220000 / 330000 / 430000 / 540423 (6 rungs, max_steps 540423).
Compute on the x-axis is pretraining grad steps until real budgets are supplied.

## Questions

Status: ✅ answered · 🟡 partly · ⏳ running/queued · ⬜ not started

### Denoise

| id | question | status | evidence / job |
|---|---|---|---|
| D1 | Do AUROC and F1 improve with model size? | ✅ | AUROC 0.9317 / 0.9400 / 0.9447 / 0.9434, F1 tracks it, 6 seeds; 400m turnover real (t=−6.3) |
| D2 | How much does pretraining buy, in AUROC and F1? | ✅ | ~+0.05 AUROC at every scale (50m +0.046, 100m +0.051, 200m +0.048, 400m +0.045, 4 epochs). 16 epochs from scratch (50m 0.905) still far below pretrained at 4 |
| D3 | Do AUROC and F1 improve with pretraining checkpoint? | ✅ | yes, saturating: full curves at 50m (0.910 → 0.936) and 100m (0.919 → 0.943) over 10k → 540k, flat from ~330k; 10k already beats scratch by ~0.025. 200m rises through 330k; 400m flat 181k → 220k. 200m/400m later rungs wait on pretraining |
| D4 | Do denoise HPs hold across checkpoints? | ✅ | lr2e-4/es0.5 wins at ckpt 1 and 540423 |
| D5 | Does the denoiser generalise to spectra LONGER than its 512-peak training cap (13% of test, up to 3,086 peaks)? | 🟡 | `msdelta/eval_denoise_length.py` scores the Hub models by peak-count bucket. First run: longer buckets score higher (0.96-0.975 AUROC) BUT its <=512 check reads 0.9227 vs the published 0.9366 (50m), so NOT yet trusted; precision and padding ruled out, weights/data path under investigation |

### Contrastive

| id | question | status | evidence / job |
|---|---|---|---|
| C0 | Is our metric valid? | ✅ | separation ratio does NOT predict retrieval (ρ −0.04 within scale). Use MAP@R |
| C1 | What is the best training recipe? | ✅ | frozen as R* (`sweeps/make_confreeze.py`): t ~0.002, width 256, **24 epochs**, MAP@R 0.877 (50m). Training length was the main lever (+0.11 from 3 → 12 epochs); wider batches help only when trained long; temperature matters little once trained long |
| C2 | Does retrieval improve with model size? | ✅ | frozen recipe, 3 seeds, on the 100k test: 50m 0.657, 100m 0.665, 200m 0.658, **400m 0.703** -- flat to 200m, a step at 400m. The small eval ranks 400m last (0.867) and cannot resolve scale. Jobs 8860092 / 8860863 |
| C3 | How much does pretraining buy? | ✅ | ALL of it: random init trained contrastively lands at chance at every scale (MAP@100 ≈0.01, Hit@1 ≈0.03 vs 0.33–0.41 / 0.75–0.78 pretrained, t=16–96, 6 seeds). It never reaches even the UNTRAINED pretrained encoder. Since random = chance, the gap is just the pretrained score and updates with C2 |
| C4 | Does retrieval improve with pretraining checkpoint? | ✅ | 100k test, 3 seeds: 50m 0.447→0.627→0.655→0.664→0.665 (10k→540k), 100m 0.510→0.646→0.666→0.672→0.671; big gain to 120k, saturating after ~330k; 100m ≥ 50m at every rung. Job 8860093 |
| C5 | Do contrastive HPs transfer across scale/checkpoint? | ✅ | same winner in all 5 cells (4 scales + 50m@540k), mean pairwise ρ +0.81; t0.03 beat t0.07 in all 4 cells tested |
| C6 | Does best-model selection / early stopping matter? | ✅ | no: −0.002, −0.006 (n.s.); `final/` == last checkpoint |
| C7 | Does training on ms-contrastive-100k beat the replicate corpus, and does scale show there? | ⏳ | **yes, decisively**: 50m continued on it for 300 steps (~¼ epoch) reaches 0.81 exp MAP@R on its test split vs 0.656 before and binned cosine 0.730 (3 seeds; consensus view still 0.65 vs 0.76). Best-first run 8860522 continues 50m and 400m for one epoch, encoder every 300 steps; 400m step-300 read ~07:30 UTC |
| C8 | Does a per-pair sigmoid loss (SigLIP, Zhai et al. 2023) beat SupCon? | ⬜ | SupCon is a softmax over the batch with every positive in every denominator, so positives compete (target 1/(K-1) each). Sigmoid scores each pair independently (learnable temperature + bias). Ablation at the C1 recipe, 50m, 3 seeds, after C7 so data and loss do not change together |
| C9 | Does an MLP projection head (master's MSDeltaForRetrieval) help the embedding -- retrieving on the head output, or on the pre-head features (SimCLR)? | ✅ | **No; keep the no-head design.** 24 epochs, 3 seeds, small eval: no head 0.877, head output 0.825, pre-head 0.705 (3 epochs: 0.60 / 0.30 / 0.29). Job 8860587 |
| C10 | Zero-shot scaling: how does FROZEN-encoder retrieval scale with model size and pretraining? (kept to show how much training improves embeddings) | 🟡 | best-layer (~3/4 depth) exp MAP@R @220k: 50m 0.208, 100m 0.140, 200m 0.151, 400m 0.432 (job 8862567, 10 encoders). **To do**: every ladder checkpoint at every scale, for full curves |

### Alignment (peptide embedder)

A peptide encoder trained to land on the frozen spectrum encoder's embedding of that
peptide's spectra (L2 on normalised vectors). It is the sequence side of reranking and of
library search. Metric: peptide->spectrum Hit@1 / MAP@R on held-out peptides.

| id | question | status | evidence / job |
|---|---|---|---|
| A0 | What does the current student reach, and on what? | 🟡 | R1's student: Hit@1 0.73 on 94 held-out replicate-corpus peptides (50m teacher), trained on ~855. Blind to adjacent-residue swaps (57% vs near-miss decoys). Not yet scored on ms-contrastive-100k |
| A1 | Does training on ms-contrastive-100k (~88k peptides) give a much better student? | ✅ | **yes**: teacher C7-50m step 600; on the TEST split, all 25,137 spectra vs all 9,771 candidates: Hit@1 **0.898** (3 seeds, ±0.001), Hit@5 0.932, MRR 0.914 -- vs 0.73 for R1's student on 94 peptides. Jobs 8861093/8861292/8861309/8861310 |
| A2 | Does student quality track teacher quality? | ⬜ | **to do**: when C7 best-first (8860522) finishes, rebuild the teacher cache from its FINAL 50m and 400m encoders (sharded, `pbs/precompute_align_sharded.pbs`; ~10 min at 50m, ~25 min at 400m) and retrain the student on each (`sweeps/make_a1.py` TEACHERS; only --pretrained_path changes), 3 seeds, scored with `pbs/eval_align_test.pbs` on the test split |
| A3 | Does order-aware pooling fix the adjacent-swap blindness? | ⬜ | **now the main reranking lever**: with the A1 student the embedding alone beats mass-matched and reversed decoys 98-99% but near-miss swaps only 70.5%, and near-misses are why it costs -0.037 Hit@1 in the classifier. Try a CLS/attention pool; score on the swap test and Hit@1 |

### Reranking (downstream of contrastive)

The use contrastive exists for. The reranker scores (spectrum, candidate peptide) pairs;
the contrastive encoder reaches it as the teacher of an alignment tower, whose
`embedding_cosine` joins hand-built features in the rescorer. Metric: Hit@1 of the true
peptide among each spectrum's candidates, paired seeds (arms share split and init).

| id | question | status | evidence / job |
|---|---|---|---|
| R0 | What do we have to beat? | 🟡 | feature-only rescorer 0.893, but on an unfit benchmark: decoys not mass-matched (FT30), leaky split (FT29, fixed). To be re-taken on a realistic benchmark |
| R1 | Do the encoders that win on contrastive improve the rescorer over that baseline? | 🟡 | embedding costs ~0.11 Hit@1 for both teachers, but only because near-miss decoys (adjacent swaps) are the benchmark's only hard case and the student is blind to them; the benchmark needs truly mass-matched decoys (FT30) before this answers anything |
| R2 | If not, does a formulation without a shared per-spectrum vector (cross-encoder) fix it? | ⬜ | the −0.109 was diagnosed as errors correlated within a spectrum: every candidate is scored against the SAME cached teacher vector. Only if R1 fails |
| R3 | On the FULL Gaolaboratory/psm-rerank-hek-hct116 (175 runs, 7.8M spectra), how many more PSMs/peptides at 1% FDR does our rescorer accept than MSFragger, and what does the embedding add? | ⏳ | first pass on 16 runs (8 train / 8 held-out): MLP on engine + hand-built + embedding 96,367 vs MSFragger 89,693 PSMs@1% (+7.4%, CV by run); embedding adds ~0.1% on top. **Full dataset**: stage 1 (embeddings + hand features) on all runs; split by ACQUISITION SERIES for HEK293 (whole series held out; MudPIT runs of one series are one sample) and by fraction for HCT116; mixed and per-dataset rescorers; optional peptide-disjoint test (unmodified sequence, I/L collapsed) |

R1's first pass runs now with the current best encoders: they are ~2.3x better than the
one behind the −0.109, which is the question, and it proves the pipeline so the final
C1 winners can go through it immediately.

## Baselines to build (to-do)

External reference points, by track. Each is scored on the same split and metric as ours.

CONTRASTIVE (spectrum embedder; ms-contrastive-100k test, MAP@R / Hit@1)
- [x] **Binned cosine** (spectral-library dot product) — 0.730 exp MAP@R at 0.1 Da, 0.671 at 1 Da (8860250)
- [x] **PCA** — fitted on 10k ms-contrastive-100k TRAIN analytes, test projected (job 8863968): 0.1 Da bins -> 1280 dims 0.723 exp MAP@R (Hit@1 0.814), -> 256 dims 0.488; 1 Da -> 256 dims 0.553. Our C7 400m 0.859, 50m 0.839
- [ ] **GLEAMS** (Bittremieux et al., Nat Methods 2022) — GitHub install with an old TensorFlow in its OWN venv; needs raw m/z + intensity + precursor m/z + charge exported to MGF. ~0.5–1 day. Caveats: uses precursor info (we do not); trained on MassIVE-KB, which may overlap our test

ALIGNMENT (peptide embedder; peptide->spectrum Hit@1 / MRR on the test split)
- [ ] **yHydra** (Altenburg et al. 2022) — joint peptide/spectrum embedding, the direct competitor to the A1 student. 1–3 days, maintenance risk

RERANKING (on the incoming reranking dataset)
- [ ] **FDR vs the existing ranking** — PSMs/peptides at 1% FDR (target-decoy) of our reranking against the dataset's own search-engine ranking
- [ ] **MS²Rescore** (MS²PIP + DeepLC features) — ~1 day on top of the dataset's search results
- [ ] **Prosit / Oktoberfest** — predicted-spectrum rescoring; 1–2 days
- [x] ~~Sage + mokapot~~ — dropped 2026-09-24: no search engine needed (the dataset supplies MSFragger's results) and mokapot is not used as a baseline

## Order of work

CONTRASTIVE follows four stages (decided 2026-09-23). Training on ms-contrastive-100k
costs ~7 h/epoch at 50m and ~22 h at 400m on one tile, so the full suite trains on the
small replicate corpus and the large corpus is used to (a) TEST everything and (b) train
a few chosen models.

```
DENOISE                                   CONTRASTIVE
───────                                   ───────────
D1 ✅  D2 ✅  D3 ✅  D4 ✅                  STAGE 1  full suite on the replicate corpus
  D3 at 200m/400m: ends-big wave            C0 ✅ C1 ✅ C3 ✅ C5 ✅ C6 ✅
  (8860442 smoke → 18 arms)                 C2 ⏳ 8860092   C4 ⏳ 8860093 (50m/100m)
  400m 330k-540k: not pretrained yet                  │
      └─► denoise scaling figures                     ▼
          (AUROC and F1)                  STAGE 2  score every Stage-1 model on the
                                            ms-contrastive-100k test split
                                            69 models: small-eval vs 100k MAP@R Spearman
                                            0.78 (0.98 on the first 27). Recipe/length
                                            effects transfer; SCALE and CHECKPOINT effects
                                            do not -- they are read here only
                                                      │
                                                      ▼
                                          STAGE 3  a few configs trained ON ms-contrastive-100k
                                            (C7; 50m ep1 x3 queued 8860292; rest chosen
                                            from Stage 2)
                                                      │
                                                      ▼
                                          STAGE 4  score those on the same test split,
                                            vs binned cosine (0.730) and GLEAMS
                                                      │
                                   ┌──────────────────┴─────────────┐
                                   ▼                                ▼
                         C8 sigmoid vs SupCon          A1-A3 peptide embedder (teacher
                                                       = best Stage 3/4 encoder)
                                                                    │
                                                                    ▼
                                                     R0-R2 on the incoming reranking dataset
```

Reranking waits for the new reranking dataset (2026-09-23: being obtained). Until then
the work is good spectrum and peptide embedders; R0-R2 as written assume our synthetic
decoys and will be re-set against that dataset.

## Parked

Real questions, deliberately not on the path to the two conclusions. Each keeps its
write-up in TODO.md under the same number.

- **FT8** — is a warm-up freeze on the encoder worth anything?
- **FT12** — the sign of the pooled-vs-within AUROC gap, measured rather than argued.
- **ms-contrastive-100k only (lowest priority)** — C7's encoders are two-stage (replicate corpus, then ms-contrastive-100k). Training from the pretrained checkpoint on ms-contrastive-100k alone would show whether stage one adds anything. Not needed for the conclusions; two-stage is the recipe (decided 2026-09-24).
- **Linear vs non-linear probe on frozen features (not pursued)** — the zero-shot per-layer redo (OBSERVATIONS 2026-09-24, `results/finetune/contrastive/zeroshot-layers/`) shows frozen encoders retrieve well at ~3/4 depth, which partly overturns the old "non-linear, unreachable by any readout" claim. That data is kept ONLY as the starting point for how much training improves the embeddings (figure c100k_scale); the theory itself is not being chased.
- **Layer mixing (retry candidate)** — `pooling=layer_mix` was dropped (OBSERVATIONS, "Blending encoder depths is worse", job 8842351) on the SEPARATION RATIO at the old recipe (t 0.07, 3 epochs, P2xK2). C0 later showed that ratio does not predict retrieval, so "last layer is best" was never tested on MAP@R. Retry: 50m at the frozen C1 recipe, layer_mix vs last layer, 3 seeds each, scored on the small eval and the ms-contrastive-100k test (~2.4 h/arm, one node). No write-up in TODO.md; this entry is it.

## Rules

- Every job is tagged with a question id in its generator docstring.
- Grids are validated on debug before capacity. Never regenerate a grid dir a queued job
  points at (queued jobs read configs at run time).
- Contrastive is scored on MAP@R; the separation ratio is reported, never selected on.
- A result at the edge of a swept range means the optimum is unlocated.
