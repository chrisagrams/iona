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

### Contrastive

| id | question | status | evidence / job |
|---|---|---|---|
| C0 | Is our metric valid? | ✅ | separation ratio does NOT predict retrieval (ρ −0.04 within scale). Use MAP@R |
| C1 | What is the best training recipe? | ✅ | frozen as R* (`sweeps/make_confreeze.py`): t ~0.002, width 256, **24 epochs**, MAP@R 0.877 (50m). Training length was the main lever (+0.11 from 3 → 12 epochs); wider batches help only when trained long; temperature matters little once trained long |
| C2 | Does retrieval improve with model size? | 🟡 | at 3 epochs 50m 0.751 < 100m 0.783 < 200m 0.788 < 400m 0.823, but at 12 epochs the 50m/400m gap shrinks to 0.013 (0.860 vs 0.873). Needs all 4 scales at the final recipe. **Frozen-recipe run queued: 8860092** (`configs/sweep-confreeze`, 4 scales @220k, 3 seeds) |
| C3 | How much does pretraining buy? | ✅ | ALL of it: random init trained contrastively lands at chance at every scale (MAP@100 ≈0.01, Hit@1 ≈0.03 vs 0.33–0.41 / 0.75–0.78 pretrained, t=16–96, 6 seeds). It never reaches even the UNTRAINED pretrained encoder. Since random = chance, the gap is just the pretrained score and updates with C2 |
| C4 | Does retrieval improve with pretraining checkpoint? | ⏳ | frozen C1 recipe, 50m+100m x 10k/120k/330k/430k/540k (+220k from C2), 3 seeds: **8860093**. Old ladder 8851663 (t0.07) superseded |
| C5 | Do contrastive HPs transfer across scale/checkpoint? | ✅ | same winner in all 5 cells (4 scales + 50m@540k), mean pairwise ρ +0.81; t0.03 beat t0.07 in all 4 cells tested |
| C6 | Does best-model selection / early stopping matter? | ✅ | no: −0.002, −0.006 (n.s.); `final/` == last checkpoint |
| C7 | Does training on ms-contrastive-100k beat the replicate corpus, and does scale show there? | ⏳ | **yes, decisively**: 50m continued on it for 300 steps (~¼ epoch) reaches 0.81 exp MAP@R on its test split vs 0.656 before and binned cosine 0.730 (3 seeds; consensus view still 0.65 vs 0.76). Best-first run 8860522 continues 50m and 400m for one epoch, encoder every 300 steps; 400m step-300 read ~07:30 UTC |
| C8 | Does a per-pair sigmoid loss (SigLIP, Zhai et al. 2023) beat SupCon? | ⬜ | SupCon is a softmax over the batch with every positive in every denominator, so positives compete (target 1/(K-1) each). Sigmoid scores each pair independently (learnable temperature + bias). Ablation at the C1 recipe, 50m, 3 seeds, after C7 so data and loss do not change together |
| C9 | Does an MLP projection head (master's MSDeltaForRetrieval) help the embedding -- retrieving on the head output, or on the pre-head features (SimCLR)? | ⏸ | **deprioritised 2026-09-24**: at 3 epochs the head HALVES retrieval (small eval 0.30 vs 0.60 no-head; 100k test 0.155 head / 0.184 pre-head vs 0.382), 3 seeds each. The 24-epoch arms (8860587) stay queued BEHIND the denoise ladder; the current no-head design stands unless they beat the control |

### Alignment (peptide embedder)

A peptide encoder trained to land on the frozen spectrum encoder's embedding of that
peptide's spectra (L2 on normalised vectors). It is the sequence side of reranking and of
library search. Metric: peptide->spectrum Hit@1 / MAP@R on held-out peptides.

| id | question | status | evidence / job |
|---|---|---|---|
| A0 | What does the current student reach, and on what? | 🟡 | R1's student: Hit@1 0.73 on 94 held-out replicate-corpus peptides (50m teacher), trained on ~855. Blind to adjacent-residue swaps (57% vs near-miss decoys). Not yet scored on ms-contrastive-100k |
| A1 | Does training on ms-contrastive-100k (~88k peptides) give a much better student? | ⏳ | `configs/a1-align-100k-400m-c1` (teacher: C1 400m ep12, 0.711 on 100k test); smoke 8860290 |
| A2 | Does student quality track teacher quality? | ⬜ | re-run A1 with the C7 teacher(s); only --pretrained_path changes |
| A3 | Does order-aware pooling fix the adjacent-swap blindness? | ⬜ | mean+max pooling is nearly a bag of residues; try a CLS/attention pool, scored on the swap test and Hit@1 |

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

R1's first pass runs now with the current best encoders: they are ~2.3x better than the
one behind the −0.109, which is the question, and it proves the pipeline so the final
C1 winners can go through it immediately.

## Baselines to build (to-do)

External reference points; each names the question it serves.

- [x] **Binned cosine** (spectral-library dot product), C7/C2 — in `eval_grouped_retrieval`, scored by 8860250
- [ ] **GLEAMS** (Bittremieux et al., Nat Methods 2022), spectrum→spectrum, C7 — not on PyPI; GitHub install with an old TensorFlow in its OWN venv (never the project env). ~0.5–1 day. Caveat: trained on MassIVE-KB, which may overlap our test spectra
- [ ] **Sage + mokapot** (PSMs at 1% FDR), R0–R2 — prebuilt Sage binary + mokapot in their own venv, raw mzML + FASTA; ~1–1.5 days setup, a few CPU node-hours. Wait for the incoming reranking dataset's format first: if it ships search results, only mokapot is needed (~0.5 day)
- [ ] later, if R needs them: MS²Rescore (~1 day on top of Sage+mokapot), Prosit/Oktoberfest (1–2 days), yHydra (1–3 days, cross-modal, A-track)

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
                                            first check: 27 models, small-eval vs 100k
                                            MAP@R Spearman 0.98 -- large effects transfer;
                                            differences below the small eval's resolution
                                            (50m vs 400m) do NOT, so scaling is read here
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

- **FT1** — does the denoiser generalise beyond 1024 peaks?
- **FT8** — is a warm-up freeze on the encoder worth anything?
- **FT12** — the sign of the pooled-vs-within AUROC gap, measured rather than argued.
- **Layer mixing (retry candidate)** — `pooling=layer_mix` was dropped (OBSERVATIONS, "Blending encoder depths is worse", job 8842351) on the SEPARATION RATIO at the old recipe (t 0.07, 3 epochs, P2xK2). C0 later showed that ratio does not predict retrieval, so "last layer is best" was never tested on MAP@R. Retry: 50m at the frozen C1 recipe, layer_mix vs last layer, 3 seeds each, scored on the small eval and the ms-contrastive-100k test (~2.4 h/arm, one node). No write-up in TODO.md; this entry is it.

## Rules

- Every job is tagged with a question id in its generator docstring.
- Grids are validated on debug before capacity. Never regenerate a grid dir a queued job
  points at (queued jobs read configs at run time).
- Contrastive is scored on MAP@R; the separation ratio is reported, never selected on.
- A result at the edge of a swept range means the optimum is unlocated.
