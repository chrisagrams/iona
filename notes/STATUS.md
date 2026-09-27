# Project status

What is true right now. Overwritten, not appended: before changing the diagram, copy the
old one into `status-history/`. The questions it tracks are in `PLAN.md`; results with
their evidence are in `OBSERVATIONS.md`; hazards are in `TODO.md`.

Last updated: 2026-09-27, ~04:20 UTC.

```
LEGEND  [x] done  [~] RUNNING  [ ] queued / next  [>] blocked on something  [X] retracted

BRANCHES
──────────────────────────────────────────────────────────────────────────
  dev_finetune ...... frozen at paper submission (2026-09-26); main checkout,
                      until the mix job ends
  dev_finetune_02 ... working branch (~/code/msdelta-02); new jobs go from here
  master ............ upstream, never touched

JOBS
──────────────────────────────────────────────────────────────────────────
  [~] 8872806  C23 HP search, 50m @540k, 12 arms ............. finishing
  [~] 8873159  C21 mix: within75 / within50 / between75 x3 ... ~2.5 h left
  [~] 8873562  C21 two-region smoke + first code-snapshot job (debug-scaling)
  [>] 8873563  C21 two-region 75/25 + 50/50 x3 ............... held on the smoke
  [ ] 8873598  e2e tests from dev_finetune_02 (debug) ........ K38 re-check

CONTRASTIVE (single-dataset recipe: SupCon + same-mass, ms-contrastive-100k)
──────────────────────────────────────────────────────────────────────────
  [x] C8  sigmoid loss rejected      [x] C19 same-mass batches adopted
  [x] C24 filter cost measured (isotope-tolerant window removes the loss)
  [~] C23 HPs per scale (50m first)  [~] C21 far-mass knob
  [ ] C20 train with the consensus spectrum

ENGINEERING
──────────────────────────────────────────────────────────────────────────
  [x] reorg (task subpackages + shims, results raw/processed) merged into dev_finetune_02
  [x] opt-in legacy / e2e / golden tests; FT26 fixed; K38 fixed; per-job code snapshots
  [ ] cutover of the main checkout to dev_finetune_02 (after 8873159)
  [ ] rename peptide/spectrum "embedder" -> "encoder"
  [ ] K4 data/synthetic -> /flare (deletion protocol)
  [ ] P1 Pairformer port (HF-compliant)

REMINDER
──────────────────────────────────────────────────────────────────────────
  alignment caveats (PLAN.md -> Design decisions -> Alignment -> Caveats)
```

Job history is not kept here any more: `pbs/job_history.sh` regenerates it from the
logs, which is more trustworthy than a hand-kept table.
