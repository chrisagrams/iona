# Project status

What is true right now. Overwritten, not appended: before changing it, copy the old one into
`status-history/`. Questions: `PLAN.md`; results: `OBSERVATIONS.md`; decisions: `DECISIONS.md`;
waiting on the user: `OPEN_QUESTIONS.md` (top: "Your to-do").

Last updated: 2026-09-28 ~12:30 UTC (Aurora maintenance; this session may restart).

```
BRANCHES / CHECKOUTS
──────────────────────────────────────────────────────────────────────────
  ~/code/msdelta          dev_finetune_02 (working branch; main checkout)
  dev_finetune            frozen at paper submission; master untouched
  other worktrees         msdelta-pr, msdelta-rerank, msdelta-denoise-pr (older; kept, K109)
  merges                  scratch worktree -> tests -> pbs/checkout_ff (K111); jobs snapshot a commit

JOBS (capacity queue blocked by maintenance: "Insufficient amount of resource: at_queue")
──────────────────────────────────────────────────────────────────────────
  [ ] 8875260  K66-C 400m training (12 arms, 14 h walltime)   queued since 2026-09-27 23:37
  [ ] 8875263  K66-C 25m  training (12 arms, 10 h walltime)   queued since 2026-09-27 23:37
  [x] K66-C 100m, 200m: trained + scored on validation / oodval / test / mouse / human / yeast
      results/raw/finetune/contrastive/hp-scale-{100m,200m}-{...}/

SCORING WATCHERS (background loops in the Claude session -- they DIE if the session restarts)
──────────────────────────────────────────────────────────────────────────
  When a training job ends with "=== done: 12/12 arms ok" in pbs/logs/<job>.*.OU they submit scoring.
  Restart them (from ~/code/msdelta) -- they skip nothing twice only if the scoring dirs are checked first:
    bash <scratchpad>/k66_pipeline.sh 8875194 400m=8875260 025m=8875263     (val/ood/test/mouse/human, debug)
    bash <scratchpad>/k66_yeast.sh     (edit JOB map to 400m/025m only; yeast on capacity 3 h)
  The scripts live in the session scratchpad (/tmp/claude-40253/...). If they are gone, submit by hand
  for each finished scale s in {400m, 025m}:
    models file: sweeps/arms/score_hp_scale_<s>.txt  (one line per arm: "<name> <run>/final",
                 runs at $S/runs/sweep-s<s>_ck540k_*-<jobid>)
    qsub -q debug -l select=1 -l walltime=01:00:00 -v MODELS=<file>,SPLIT=validation,OUT_DIR=results/raw/finetune/contrastive/hp-scale-<s>-validation pbs/eval_grouped_retrieval.pbs
    same with SPLIT=test; DATA=$S/baselines/{nine_oodval20k,noble_mouse20k,noble_human20k}/prepared for oodval/mouse/human;
    yeast: -q capacity -l walltime=03:00:00, DATA=$S/baselines/nine_yeast/prepared (canonical, K76)
  ($S = /lus/flare/projects/UIC-HPC/khuss/msdelta)

DONE RECENTLY
──────────────────────────────────────────────────────────────────────────
  [x] C21 batch mixes (none beats same-mass), C20 consensus training, C24 filter cost
  [x] K66-C per-scale search: 100m/200m done (see OBSERVATIONS when written)
  [x] reorg + encoder renames; strict loading default; run-from-commit (pbs/qsub_ref);
      race-free snapshots + pbs/checkout_ff; job DAG built + first live trial passed (pbs/dagctl)
  [x] Pairformer ported (architecture="pairformer"), reviewed, diagrams (notes/PAIRFORMER.md);
      triangle attention SDPA default (K102/K115)
  [x] filtered metrics (with/without filter, passes/failures) + library search, on by default
  [x] MSConsensus-100M (190 GB) and MassIVE-KB (all splits) on /flare

NEXT
──────────────────────────────────────────────────────────────────────────
  [ ] K66-C 400m / 25m -> scoring -> full comparison + proposal to the user (winner per scale)
  [ ] Stage 0 (Pairformer vs transformer sanity run): preprocessing at cap 150, then the run
  [ ] then the winning C recipe on every pretraining checkpoint (card)
  [ ] alignment resumes on the new C models (caveats list)

REMINDERS FOR THE USER
──────────────────────────────────────────────────────────────────────────
  K63-I DAG review · C18-C MassIVE-KB review · Pairformer study (notes/PAIRFORMER.md)
```
