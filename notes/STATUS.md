# Project status

What is true right now. Overwritten, not appended: before changing the diagram, copy the
old one into `status-history/`. The questions it tracks are in `PLAN.md`; results with
their evidence are in `OBSERVATIONS.md`; hazards are in `TODO.md`.

Last updated: 2026-09-23.

```
LEGEND  [x] done  [~] RUNNING  [ ] queued / next  [>] blocked on something  [X] retracted


JOBS  (each tagged with the PLAN.md question it answers)
──────────────────────────────────────────────────────────────────────────
  [~] C1+C2  sweep-conbig on debug-scaling: temp 0.01/0.02/0.03 x P/K 4/16/64
             x 4 scales @220k, 108 arms. 1h rounds, resubmitted until done
             (round 1: 8856460); interrupted arms restart from step 0
  [~] D3     8850494 denoise ladder 220k+330k x 4 scales -- capacity, running
  [ ] D2     8856558 resume of the 5 unfinished scratch arms -- capacity, queued
  [ ] D3     8856549 ends wave 10k/120k/430k/540k x 50m+100m, 24 arms -- queued


DENOISE  (AUROC and F1)
──────────────────────────────────────────────────────────────────────────
  [x] D1 model size ...... AUROC 0.9317 / 0.9400 / 0.9447 / 0.9434, 6 seeds
                           400m turnover real (t=-6.3), not a budget artefact
  [~] D2 pretraining ..... +0.046 AUROC at ep4, +0.032 at ep8 (50m)
                           all-scale scratch grid: 7/12, rest resuming
  [ ] D3 checkpoints ..... only 50m has two points: 0.9272 -> 0.9365, 133k -> 540k
  [x] D4 HP transfer ..... lr2e-4 / es0.5 wins at ckpt 1 and at 540423


CONTRASTIVE  (MAP@R, Precision@1, R-Precision)
──────────────────────────────────────────────────────────────────────────
  [x] C0 metric .......... separation ratio does NOT predict retrieval
  [~] C1 recipe .......... SupCon, lr1e-4, KL10, t0.03, batch of 4 (P=2 K=2)
                           t0.03 is the grid edge; P/K never varied -> conbig
  [~] C2 model size ...... 50m > 200m at t0.03 (p=0.003); 100m/400m not run
                           at current recipe -> conbig
  [x] C3 pretraining ..... random init lands at CHANCE at every scale;
                           pretraining is the entire result
  [>] C4 checkpoints ..... ladder ran at t0.07 (superseded) -> redo after C1
  [x] C5 HP transfer ..... same winner in 5 cells, 4 scales, mean rho +0.81
  [x] C6 selection ....... best-model selection changes nothing; final/ is fine
  [X] "more negatives hurt" -- retracted, was a ratio artefact


RERANKING  (Hit@1, downstream of contrastive)
──────────────────────────────────────────────────────────────────────────
  [x] R0 baseline ........ feature-only rescorer Hit@1 0.889
  [>] R1 best encoders ... waits for C1; last attempt (old recipe) cost -0.109
  [>] R2 cross-encoder ... only if R1 shows no gain


DECISIONS NEEDED
──────────────────────────────────────────────────────────────────────────
  (none -- everything is queued or running)
```

Job history is not kept here any more: `pbs/job_history.sh` regenerates it from the
logs, which is more trustworthy than a hand-kept table.
