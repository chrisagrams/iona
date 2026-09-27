# Project status

What is true right now. Overwritten, not appended: before changing the diagram, copy the
old one into `status-history/`. The questions it tracks are in `PLAN.md`; results with
their evidence are in `OBSERVATIONS.md`; hazards are in `TODO.md`.

Last updated: 2026-09-23, 18:30.

```
LEGEND  [x] done  [~] RUNNING  [ ] queued / next  [>] blocked on something  [X] retracted


JOBS
──────────────────────────────────────────────────────────────────────────
  nothing running or queued -- every job has reported


DENOISE  (AUROC and F1)
──────────────────────────────────────────────────────────────────────────
  [x] D1 model size ...... 0.9317 / 0.9400 / 0.9447 / 0.9434, 400m turnover real
  [x] D2 pretraining ..... ~+0.05 at every scale (+0.046/+0.051/+0.048/+0.045)
  [x] D3 checkpoints ..... saturating: 50m 0.910 -> 0.936, 100m 0.919 -> 0.943
                           over 10k -> 540k, flat from ~330k; 400m flat
  [x] D4 HP transfer ..... lr2e-4 / es0.5 holds at 540423
  [x] splits ............. peptide-disjoint (audit 8859654)
  [>] 200m/400m later rungs wait on pretraining (200m 82%, 400m 47%)


CONTRASTIVE  (MAP@R)
──────────────────────────────────────────────────────────────────────────
  [x] C0 metric .......... separation ratio invalid; MAP@R
  [~] C1 recipe .......... t ~0.002, width 256, 24 epochs -> 0.877 (50m).
                           Training length was the lever (+0.11), not negatives
  [ ] C2 model size ...... all 4 scales at the final recipe
  [x] C3 pretraining ..... random init at chance at every scale
  [ ] C4 checkpoints ..... ladder at the final recipe
  [x] C5 / C6 ............ HPs transfer; best-model selection irrelevant


RERANKING  (Hit@1)
──────────────────────────────────────────────────────────────────────────
  [~] R0/R1 .............. embedding costs ~0.11, but the benchmark is unfit:
                           decoys not mass-matched (FT30), split leaked (FT29,
                           fixed). Near-miss decoys are its only hard case and
                           the student is blind to residue order
  [>] R1 re-take ......... needs truly mass-matched candidates


DECISIONS NEEDED
──────────────────────────────────────────────────────────────────────────
  [ ] freeze the C1 recipe (width 256, 24 epochs, t 0.002) and run C2 + C4?
  [ ] reranking benchmark: MassIVE-KB same-mass decoys, or ask Chris for real
      search-engine candidate lists?
  [ ] reranking vs retrieval: which does the project claim?
```

Job history is not kept here any more: `pbs/job_history.sh` regenerates it from the
logs, which is more trustworthy than a hand-kept table.
