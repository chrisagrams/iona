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
| D2 | How much does pretraining buy, in AUROC and F1? | 🟡 | AUROC +0.046 (ep4), +0.032 (ep8) at 50m. All-scale scratch grid `8847663`: 7/12 arms done, 5 need resume |
| D3 | Do AUROC and F1 improve with pretraining checkpoint? | ⏳ | ladder `8850494` (220k+330k) queued; ends wave (10k/120k/430k/540k, 50m+100m) built, not submitted |
| D4 | Do denoise HPs hold across checkpoints? | ✅ | lr2e-4/es0.5 wins at ckpt 1 and 540423 |

### Contrastive

| id | question | status | evidence / job |
|---|---|---|---|
| C0 | Is our metric valid? | ✅ | separation ratio does NOT predict retrieval (ρ −0.04 within scale). Use MAP@R |
| C1 | What is the best training recipe? | 🟡 | lr1e-4 / KL10 / **t0.03** best so far, but t0.03 is the grid edge and P/K (negatives) never varied → job `8856399` (grid dir `sweep-conbig`) |
| C2 | Does retrieval improve with model size? | 🟡 | 50m > 200m at t0.03 (p=0.003). 100m/400m never run at current recipe → job `8856399` (grid dir `sweep-conbig`) |
| C3 | How much does pretraining buy? | ✅ | ALL of it: random init trained contrastively lands at chance at every scale (MAP@100 ≈0.01, Hit@1 ≈0.03 vs 0.33–0.41 / 0.75–0.78 pretrained, t=16–96, 6 seeds). It never reaches even the UNTRAINED pretrained encoder. Since random = chance, the gap is just the pretrained score and updates with C2 |
| C4 | Does retrieval improve with pretraining checkpoint? | 🟡 | ladder `8851663` ran at t0.07 — superseded recipe. Re-run after C1 |
| C5 | Do contrastive HPs transfer across scale/checkpoint? | ✅ | same winner in all 5 cells (4 scales + 50m@540k), mean pairwise ρ +0.81; t0.03 beat t0.07 in all 4 cells tested |
| C6 | Does best-model selection / early stopping matter? | ✅ | no: −0.002, −0.006 (n.s.); `final/` == last checkpoint |

## Order of work

```
DENOISE                                   CONTRASTIVE
───────                                   ───────────
D1 ✅  D4 ✅                               C0 ✅  C3 ✅  C5 ✅  C6 ✅
D2 🟡 resume 5 scratch arms                C1 🟡 job 8856399 (temp × P/K × 4 scales)
D3 ⏳ ladder 8850494 → ends wave                │   also answers C2 at 220k
      └─► denoise scaling figures               ▼
          (AUROC and F1)                  freeze recipe R*
                                                │
                                         ┌──────┴──────┐
                                         ▼             ▼
                                   C2 all scales  C4 checkpoint
                                   seeds at R*    ladder at R*
                                         └──────┬──────┘
                                                ▼
                                   contrastive scaling figures
```

Contrastive work downstream of C1 waits for it: re-running the ladder at a recipe that
is still moving would take every number twice (it already happened once, t0.07 → t0.03).

## Parked

Real questions, deliberately not on the path to the two conclusions. Each keeps its
write-up in TODO.md under the same number.

- **FT1** — does the denoiser generalise beyond 1024 peaks?
- **FT8** — is a warm-up freeze on the encoder worth anything?
- **FT12** — the sign of the pooled-vs-within AUROC gap, measured rather than argued.
- **Reranking** — the downstream use of contrastive. A feature-only rescorer reached
  Hit@1 0.889; an earlier embedding-based reranker HURT it (−0.109), but that embedding
  came from the superseded recipe. Undecided whether it is in scope.

## Rules

- Every job is tagged with a question id in its generator docstring.
- Grids are validated on debug before capacity. Never regenerate a grid dir a queued job
  points at (queued jobs read configs at run time).
- Contrastive is scored on MAP@R; the separation ratio is reported, never selected on.
- A result at the edge of a swept range means the optimum is unlocated.
