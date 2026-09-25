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
| D5 | Does the denoiser generalise to spectra LONGER than its 512-peak training cap (13% of test, up to 3,086 peaks)? | 🟡 | longer buckets score higher (0.96-0.975 AUROC), but trusted only once the gap below is explained. **Gap**: reloaded weights give 0.9227 test AUROC (50m) vs 0.9366 logged in training (validation 0.9324 vs 0.9438, same weights). Ruled out: weights (final == best ckpt, fp32 masters identical), fp32/bf16, padding, data, input rounding (m/z to bf16 -> 0.70), freqs. Left: the 12-rank DeepSpeed eval path itself. Hub cards show the logged numbers until resolved |

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
| C7 | Does training on ms-contrastive-100k beat the replicate corpus, and does scale show there? | ✅ | **yes**: one epoch, test exp MAP@R 50m 0.839, **400m 0.868** (Hit@1 0.913), vs replicate-only 0.656 / 0.714 and binned cosine 0.730; selected on VALIDATION (400m seed 0 0.864). Uploaded private: Gaolaboratory/iona-contrastive-400m, -50m (sha verified). Job 8860522 |
| C8 | Does a per-pair sigmoid loss (SigLIP, Zhai et al. 2023) beat SupCon? | ⬜ | SupCon is a softmax over the batch with every positive in every denominator, so positives compete (target 1/(K-1) each). Sigmoid scores each pair independently (learnable temperature + bias). Ablation at the C1 recipe, 50m, 3 seeds, after C7 so data and loss do not change together |
| C9 | Does an MLP projection head (master's MSDeltaForRetrieval) help the embedding -- retrieving on the head output, or on the pre-head features (SimCLR)? | ✅ | **No; keep the no-head design.** 24 epochs, 3 seeds, small eval: no head 0.877, head output 0.825, pre-head 0.705 (3 epochs: 0.60 / 0.30 / 0.29). Job 8860587 |
| C10 | Zero-shot scaling: how does FROZEN-encoder retrieval scale with model size and pretraining? (kept to show how much training improves embeddings) | 🟡 | best-layer exp MAP@R @220k: 50m 0.208, 100m 0.140, 200m 0.151, 400m 0.432. **All-but-the-top** (Mu & Viswanath 2018; mean + top-D PCs fitted on TRAIN removed): 50m best block 0.208 -> 0.395, final 0.112 -> 0.296 at D=32, still rising. Full run (D 8/32/64/128): 8 of 10 encoders done (job 8865987 hit the 1 h debug limit), ~2x everywhere, 200m@540k 0.432, 400m@10k 0.414 (OBSERVATIONS 2026-09-25). **Deferred (user, lower priority)**: (a) rerun 400m@220k and 400m@430k on capacity with 2 h walltime (`pbs/eval_zeroshot_layers.pbs`, same MODELS/ABTT; skips finished encoders); (b) extend D to 256 for ALL encoders, since D=128 was best for 200m/400m (edge of range, optimum unlocated) |

### Alignment (peptide embedder)

A peptide encoder trained to land on the frozen spectrum encoder's embedding of that
peptide's spectra (L2 on normalised vectors). It is the sequence side of reranking and of
library search. Metric: peptide->spectrum Hit@1 / MAP@R on held-out peptides.

| id | question | status | evidence / job |
|---|---|---|---|
| A0 | What does the current student reach, and on what? | 🟡 | R1's student: Hit@1 0.73 on 94 held-out replicate-corpus peptides (50m teacher), trained on ~855. Blind to adjacent-residue swaps (57% vs near-miss decoys). Not yet scored on ms-contrastive-100k |
| A1 | Does training on ms-contrastive-100k (~88k peptides) give a much better student? | ✅ | **yes**: teacher C7-50m step 600; on the TEST split, all 25,137 spectra vs all 9,771 candidates: Hit@1 **0.898** (3 seeds, ±0.001), Hit@5 0.932, MRR 0.914 -- vs 0.73 for R1's student on 94 peptides. Jobs 8861093/8861292/8861309/8861310 |
| A2 | Does student quality track teacher quality? | ✅ | **yes**: teacher C7 400m final (seed 0 by validation) -> test Hit@1 **0.923** (3 seeds +-0.0002) vs A1 0.898; Hit@5 0.949, MRR 0.935. Uploaded private: Gaolaboratory/iona-peptide-embedder-400m (seed 0, sha verified; standalone module). Next: reranking with A2 embeddings; yHydra comparisons |
| A3 | Does order-aware pooling fix the adjacent-swap blindness? | ✅ | **no**: cls 0.894 / attn 0.893 vs pool 0.899 test Hit@1; near-miss 68.5% vs 70.6%. Keep mean+max |
| A4 | Does a contrastive (LiT, SupCon-style multi-positive) student with distinguishable hard negatives (adjacent swaps / local shuffles, never reversals) beat A1's MSE student? | ⏳ | first sweep invalid (stray no_grad; fixed + gradient test). Rerun 8865869: validation Hit@1 LiT+4 hard negatives 0.931-0.934 vs A1 0.9305; best seed chosen on validation. Test eval + FDR vs A1 (with cosine_null + vectors) chained |
| A5 | vs yHydra (cross-modal spectrum -> peptide, identical queries/candidates, the 88.5% yHydra can represent) | 🟡 | ms-contrastive-100k test (in-distribution for us): open **A1 0.901 vs yHydra 0.196**; +-1.1 Da window 0.973 vs 0.754; 20 ppm 0.993 vs 0.942 (8-9x fewer errors). UNSEEN HEK ion-trap data: open 0.061 vs 0.016, +-1.1 Da 0.601 vs 0.345, 20 ppm 0.679 vs 0.606 -- ahead everywhere, both degraded (low-res domain shift). Still needed: an unseen HIGH-RES set **Nine-species (unseen)**: open yHydra 0.060 / A1 0.253 / A2 0.388 / A-oodsel 0.410; +-1.1 Da 0.650 / 0.537 / 0.639 / 0.659; 20 ppm 0.890 / 0.691 / 0.761 / 0.765 |
| A6 (low priority) | A2 + A4: the 400m teacher with A4's loss (LiT + 4 hard negatives, no MSE), 3 seeds (configs/sweep-a6, `make_a4.py --teacher 400m`). A4 on the 50m teacher did not change retrieval (0.895-0.898 vs 0.898) but improved reranking (MLP ms+embws 94,224 vs A1 93,843; null arm 93,557) | ⏳ | submits once A2's reranking job has started |
| A7 (LOWER PRIORITY, user 2026-09-25; c1rep cache held, random/frozen not submitted) | Pretraining / fine-tuning ablation for A: the SAME student trained against teachers = random-init 400m / frozen pretrained 400m@220k / replicate-only 400m (C1) / C7 400m (= A2, 0.923); 3 seeds each, test Hit@1 | ⏳ | queued 2026-09-25 (user) |
| A8 | Mass-aware student training: mass-bucketed batches (in-batch negatives are same-mass competitors) + same-mass hard negatives (+-20 ppm), LiT + small MSE; select on WINDOWED Hit@1 (validation + 8-other-species OOD set); eval yeast open/1.1 Da/20 ppm vs yHydra, then R | ⏳ | built (sweeps/make_a8.py, tests/test_mass_aware.py); chain: tests -> sweep -> test eval -> nine-species vs yHydra -> R |

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
| R3 | On the FULL Gaolaboratory/psm-rerank-hek-hct116 (175 runs), how many more PSMs/peptides at 1% FDR does our rescorer accept than MSFragger, and what does the embedding add? | ⏳ | stage 1 on all 175 runs done (46 re-run after the dataset's 00:34 reorganisation; code pinned to revision 87f5c27). Next: R4 matrix on the full set, CV by series |
| R4 | Does the embedding add PSMs on top of STRONG rescoring? | 🟡 | 8 runs: lab features global MLP 104,785 -> +emb 105,755; **per-run lab 129,041 (= MS2Rescore full 128,211)**, +embws +304/+697/+450 over 3 fold seeds (+0.37%, all positive); shuffled-label control 0; MS2Rescore full + A1 +310 (+0.24%); + A2 +698 with null +447 -> +0.20% net (HEK +251 net; the HCT116 '+4.1%' was noise). Ablation matrix with NULL-embedding arm fires after A4's re-embed **Focus (2026-09-25, user): R is where the embedding shows benefit.** Queued: A2 ablation seeds 1-2 (8 runs); MS2Rescore + A2 (cosws, null) + a plain repeat for noise; all 18 HCT116 runs (high-res) with A2: re-embed -> lab rescorers +-A2 +null -> MS2Rescore +-A2 +null. Then R3 (175 runs) with A2 |
| R5 | Embedding VECTORS: element-wise product spectrum x peptide, PCA 64 on training rows; null = product with a random spectrum | 🟡 | **confounded**: with A2 the null arm gains up to +2.4k PSMs (the product keeps the peptide's own embedding -> sequence-only / decoy-like signal). Needs a stricter control (same-mass-window spectrum, or a peptide-only arm) before any claim |
| C11 | A retrieval benchmark NEITHER we nor GLEAMS trained on: confident (1% FDR) MSFragger PSMs of psm-rerank-hek-hct116 (8 runs; spectra <=512 peaks, groups capped at 20 spectra -- the uncapped run failed on groups of up to 466; effectively HEK-only), same MAP@R/Hit@1. GLEAMS vs ours (replicate-only and C7) vs binned cosine | 🟡 | **ours does NOT transfer here**: binned 1 Da 0.554, GLEAMS 0.530, C7 400m 0.17, replicate-only 0.017 (job 8866238). Likely resolution domain shift (ion-trap CID). Needed: provenance of our training data, m/z-jitter test, a clean unseen HIGH-RES HCD set for the fair comparison. To do: C11b keeping all PSMs (top-512 peaks for ours), GLEAMS check vs its paper, provenance question (MassIVE-KB overlap) **Lab (2026-09-25): HEK = high-res MS1, LOW-res MS2; HCT116 = high-res MS1 and MS2.** So the collapse is the low-res fragment domain. C11-HCT116 with trimming CANCELLED (user: trimming changes the spectrum) |
| C13 | Unseen HIGH-RES HCD benchmark: nine-species (DeepNovo; InstaDeepAI/ms_ninespecies_benchmark, test split = yeast, 111k spectra, all <= 452 peaks: nothing trimmed or dropped). Groups modified peptide + charge, >= 2, capped at 20. C: GLEAMS vs ours (C7, replicate-only) vs binned; A: yHydra vs ours (open, +-1.1 Da, 20 ppm) | 🟡 | **ours LOSES**: binned 0.790, GLEAMS 0.676, C7 400m 0.47-0.56 (seed spread 0.09), replicate-only 0.44-0.50. C claim is in-distribution only. A (yHydra) running **Diagnostic (20k subset)**: frozen 400m@220k + ABTT 0.709 > every fine-tuned model (C7 <= 0.656; peaks at step 600) < GLEAMS 0.770 < binned 0.916 -- fine-tuning specialises. Next: OOD validation from the 8 other species to select a transferring teacher |
| C14 (running 2026-09-25 ~18:00; 20k-spectrum whole-group sample, trimmed to top-512) | HCT116 unseen high-res test WITH trimming (user OK'd trimming 2026-09-25): all 18 HCT116 runs, confident PSMs, spectra > 512 peaks trimmed to their top-512 (as reranking stage 1 does), groups capped at 20; C and A comparisons. Needs the trim option restored in c11_build.py | ⬜ | waiting on the choice of test set |
| C15 (queued overnight after C14; 8 non-human species x 5k, trimmed) | Noble-lab multi-species benchmark (Zenodo 10.5281/zenodo.12819175; main 12.4 GB / balanced 2.6 GB zipped): a newer, uniformly reprocessed nine-species set; not the same as C13's DeepNovo set | ⬜ | candidate |
| C16 (user 2026-09-25) | C7 recipe from the FINAL 200m checkpoint (540,423), 1 seed: stage 1 replicate corpus (12 ep) -> stage 2 one epoch ms-contrastive-100k (encoder every 300 steps); then C eval (100k test, nine-species), A student + yHydra, R reranking (per-run lab +-emb +null). `sweeps/make_c16.py` | ⏳ | smoke -> stage 1 (~2 h) -> stage 2 (~11-12 h); done ~Sep 26 04:00-06:00 UTC |
| C17 (not a priority, user 2026-09-25) | C7 recipe at 100m to add a scaling point (50/100/200/400m); needs a checkpoint choice (220k matches C7 50m/400m, 540k matches C16) | ⬜ | parked |
| C18 (camera-ready, user 2026-09-25) | GLEAMS-inspired: (a) train on MassIVE-KB / more diverse data; (b) precursor mass + charge as encoder inputs | ⬜ | deferred to camera-ready |
| C19 (proposed) | GLEAMS-inspired same-mass negatives for C: grouped batches of peptide groups with NEARBY masses; 50m, continue the replicate-corpus encoder 1 epoch ms-contrastive-100k (C7 recipe but batches); eval open + windowed retrieval, OOD set, yeast. ~9-10 h | ⬜ | awaiting go |
| C12 (low priority; likely rebuttal period; seed count TBD) | Repeat the C7 recipe from the FINAL 400m pretraining checkpoint (540,423; now backed up and verified identical to Chris's) instead of 220k, which was used because pretraining had not finished then: stage 1 (replicate corpus, 12 ep, t 0.002, KL 10) + stage 2 (1 epoch ms-contrastive-100k), 3 seeds, select on validation; compare with iona-contrastive-400m (220k base) on the 100k test, C11 and the ABTT zero-shot curve | ⏳ | added 2026-09-25 (user) |
| S25 (later; user 2026-09-25) | A NEW 25m scale exists (Gaolaboratory/iona-base-25m; Chris's runs msdelta-25m-production-01 / msdelta-base-25m-production-01 under cgrams/msdelta-runs). Recreate the scale results with it: denoise ladder (D1/D3), contrastive (C2/C4/C7), zero-shot + ABTT (C10), alignment (A1), so every scaling curve gains a 25m point | ⬜ | deferred; first check which 25m run is canonical and back its checkpoints up (as for 400m) |

R1's first pass runs now with the current best encoders: they are ~2.3x better than the
one behind the −0.109, which is the question, and it proves the pipeline so the final
C1 winners can go through it immediately.

## Baselines to build (to-do)

External reference points, by track. Each is scored on the same split and metric as ours.

CONTRASTIVE (spectrum embedder; ms-contrastive-100k test, MAP@R / Hit@1)
- [x] **Binned cosine** (spectral-library dot product) — 0.730 exp MAP@R at 0.1 Da, 0.671 at 1 Da (8860250)
- [x] **PCA** — fitted on 10k ms-contrastive-100k TRAIN analytes, test projected (job 8863968): 0.1 Da bins -> 1280 dims 0.723 exp MAP@R (Hit@1 0.814), -> 256 dims 0.488; 1 Da -> 256 dims 0.553. Our C7 400m 0.859, 50m 0.839
- [x] **GLEAMS** (Bittremieux et al., Nat Methods 2022) — pretrained: 0.646 / 0.746 on the 100k test (ours never trained on it: 0.714 / 0.810; binned 0.730). Unseen-data version = C11

ALIGNMENT (peptide embedder; peptide->spectrum Hit@1 / MRR on the test split)
- [x] **yHydra** (Altenburg et al. 2022) — cross-modal Hit@1 0.196 vs A1 0.901 on the 100k test (A5); unseen-data + mass-window versions queued

RERANKING (on the incoming reranking dataset)
- [x] **FDR vs the existing ranking** — PSMs/peptides at 1% FDR (target-decoy) of our reranking against the dataset's own search-engine ranking
- [x] **MS²Rescore** — full 128,211, search-only 100,974 PSMs@1% (8 runs); + our embedding 128,521
- [x] **Prosit / Oktoberfest** — 102,810 on the 6 HEK runs (HCT116 not run)
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
