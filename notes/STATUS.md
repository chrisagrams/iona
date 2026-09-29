# Project status

What is true right now. Overwritten, not appended: before changing it, copy the old one into
`status-history/`. Questions: `PLAN.md`; results: `OBSERVATIONS.md`; decisions: `DECISIONS.md`;
waiting on the user: `OPEN_QUESTIONS.md` (top: "Your to-do").

Last updated: 2026-09-29 ~06:00 UTC.

SYSTEM CHANGE (2026-09 update): new OS image (SLES 15 SP7, kernel 6.4, GPU driver Agama 1146); default PE is
now 26.181.0 (oneAPI 2026.1, frameworks/2026.1.0 = PyTorch 2.13). Our .venv (frameworks/2025.3.1) needs the
old PE 26.26.0: every job script now sources pbs/lib/load_frameworks.sh (b85c7b2). On the login node run
python as: bash -c 'REPO_DIR=$PWD; source pbs/lib/load_frameworks.sh >/dev/null; .venv/bin/python ...'.
Validated on a compute node: e2e + golden 10/10 (job 8876790); device suite 11/11. The "4 known failures" were a test
false positive (K126-S, fixed); suite fully green (919 passed, 36 skipped after the K127 merges).

```
BRANCHES / CHECKOUTS
──────────────────────────────────────────────────────────────────────────
  ~/code/msdelta          dev_finetune_02 (working branch; main checkout)
  dev_finetune            frozen at paper submission; master untouched
  other worktrees         msdelta-pr, msdelta-rerank, msdelta-denoise-pr (older; kept, K109)
  merges                  scratch worktree -> tests -> pbs/checkout_ff (K111); jobs snapshot a commit

JOBS
──────────────────────────────────────────────────────────────────────────
  [ ] 8876832  K66-C 400m training (14 h), capacity, running since ~03:20
  [ ] 8876833  K66-C 25m  training (10 h), capacity, running since ~03:20
  [ ] 8878032  K136-C 50m lr4e-4_p170k2 x 3 seeds (capacity 10 h); scoring by feeder plan k136_score
      (log $S/logs/feeder_k136.log; restart: setsid nohup pbs/tools/feeder.sh pbs/tools/feeder_plans/k136_score.txt >> $S/logs/feeder_k136.log 2>&1 < /dev/null &)
  watchers running detached (setsid) on aurora-uan-0010: pbs/tools/k66c/k66_pipeline.sh, k66_yeast.sh;
  logs $S/logs/k66_pipeline.log (+ .nohup). If the login node restarts: rerun them with 400m=8876832 025m=8876833 (yeast: K66_JOB_400M=8876832 K66_JOB_025M=8876833)
  [x] 8877117  K114 profiler (partial; runtime abort after OOM, K128-P)
  [x] 8877118  K119 FlexAttention test aborted (K128-P)
  [x] 8877125 / 8877126  K100 library search on C20 models, validation / test -> results/raw/finetune/contrastive/c20-{validation,test}-lib
  C18 dry run 8877152 ok. FEEDER finished (all submitted or blocked) (was detached; aurora-uan-0010): pbs/tools/feeder.sh pbs/tools/feeder_plans/k127_batch.txt, log $S/logs/feeder_k127.log,
    state $S/feeder/k127_batch/ (one file per job: job ID or BLOCKED). Submits as the per-user queued limit allows:
    c18_dryrun -> stage0_pre -> stage0_pairformer (after pre ok) -> stage0_transformer (after pairformer ok).
    Restart after a login-node restart (idempotent; never double-submits):
      setsid nohup pbs/tools/feeder.sh pbs/tools/feeder_plans/k127_batch.txt >> $S/logs/feeder_k127.log 2>&1 < /dev/null &
  [x] K66-C 100m, 200m: trained + scored on validation / oodval / test / mouse / human / yeast
      results/raw/finetune/contrastive/hp-scale-{100m,200m}-{...}/

SCORING WATCHERS (background loops in the Claude session -- they DIE if the session restarts)
──────────────────────────────────────────────────────────────────────────
  When a training job ends with "=== done: 12/12 arms ok" in pbs/logs/<job>.*.OU they submit scoring.
  Restart them (from ~/code/msdelta; scripts saved in the repo, log in $S/logs/k66_pipeline.log):
    nohup bash pbs/tools/k66c/k66_pipeline.sh 8875194 400m=8875260 025m=8875263 &   (val/ood/test/mouse/human)
    nohup bash pbs/tools/k66c/k66_yeast.sh &                                          (yeast, capacity 3 h)
  Only if those jobs finished with "=== done: 12/12 arms ok" and their hp-scale-<s>-* results dirs don't exist yet.
  If the scripts fail, submit by hand for each finished scale s in {400m, 025m}:
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
  [ ] K114 / K119 results -> write up, K118/K119 follow-ups for the user
  [ ] K100 results (C20 library search) -> OBSERVATIONS
  [ ] K117 copy-cost bench: merged (8e726a6), NOT run -- waiting on K130-P settings approval
  [ ] K110 checkpoint inventory (read-only) -> notes/K110_checkpoint_inventory.md, deletion card for approval
  [ ] C18 dry run results -> confirm the open choices before the full run
  [ ] Stage 0: preprocessing done (8877174); Pairformer arm 8878133 (after fixes K132-I telegraf, K138-I
      CCL_KVS_MODE=pmi); transformer follows via feeder k127_batch
  [ ] K66-C 400m / 25m -> scoring -> full comparison + proposal to the user (winner per scale)
  [ ] C27 consensus-weighting card (draft; K139-C) -> 50m run
  [ ] then the winning C recipe on every pretraining checkpoint (card; no 25m, K137)
  [ ] alignment resumes on the new C models (caveats list)

REMINDERS FOR THE USER
──────────────────────────────────────────────────────────────────────────
  K63-I DAG review · Pairformer study (notes/PAIRFORMER.md)
```
