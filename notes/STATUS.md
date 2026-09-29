# Project status

What is true right now. Overwritten, not appended: before changing it, copy the old one into
`status-history/`. Questions: `PLAN.md`; results: `OBSERVATIONS.md`; decisions: `DECISIONS.md`;
waiting on the user: `OPEN_QUESTIONS.md` (top: "Your to-do").

Last updated: 2026-09-29 ~02:00 UTC (after Aurora's 2026-09 major update).

SYSTEM CHANGE (2026-09 update): new OS image (SLES 15 SP7, kernel 6.4, GPU driver Agama 1146); default PE is
now 26.181.0 (oneAPI 2026.1, frameworks/2026.1.0 = PyTorch 2.13). Our .venv (frameworks/2025.3.1) needs the
old PE 26.26.0: every job script now sources pbs/lib/load_frameworks.sh (b85c7b2). On the login node run
python as: bash -c 'REPO_DIR=$PWD; source pbs/lib/load_frameworks.sh >/dev/null; .venv/bin/python ...'.
Validated on a compute node: e2e + golden 10/10 (job 8876790); full suite 8876791.

```
BRANCHES / CHECKOUTS
──────────────────────────────────────────────────────────────────────────
  ~/code/msdelta          dev_finetune_02 (working branch; main checkout)
  dev_finetune            frozen at paper submission; master untouched
  other worktrees         msdelta-pr, msdelta-rerank, msdelta-denoise-pr (older; kept, K109)
  merges                  scratch worktree -> tests -> pbs/checkout_ff (K111); jobs snapshot a commit

JOBS (capacity queue blocked by maintenance: "Insufficient amount of resource: at_queue")
──────────────────────────────────────────────────────────────────────────
  [ ] 8876824  K66-C 400m training (14 h)  resubmitted 2026-09-29, afterok on validation jobs 8876790 + 8876791
  [ ] 8876825  K66-C 25m  training (10 h)  same
  watchers running detached (setsid) on aurora-uan-0010: pbs/tools/k66c/k66_pipeline.sh, k66_yeast.sh;
  logs $S/logs/k66_pipeline.log (+ .nohup). If the login node restarts: rerun them with 400m=8876824 025m=8876825
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
  [ ] K114 profiler + K119 FlexAttention test: ready on branch k114-k119-prep (pushed; worktree
      ~/code/msdelta-prof; runbook notes/K114_K119_runbook.md there). After maintenance, from that worktree:
        qsub -q debug -l select=1 -l walltime=00:45:00 -A UIC-HPC -l filesystems=home:flare -v REPO_DIR=$PWD pbs/diag/pairformer_profile.pbs
        qsub -q debug -l select=1 -l walltime=01:00:00 -A UIC-HPC -l filesystems=home:flare -v REPO_DIR=$PWD pbs/diag/flexattn_test.pbs
  [ ] K66-C 400m / 25m -> scoring -> full comparison + proposal to the user (winner per scale)
  [ ] Stage 0 (Pairformer vs transformer sanity run): prepared on branch stage0-prep (runbook
      notes/P1_stage0_runbook.md there; worktree ~/code/msdelta-stage0); merge pending K122-S; submit after maintenance
  [ ] then the winning C recipe on every pretraining checkpoint (card)
  [ ] alignment resumes on the new C models (caveats list)

REMINDERS FOR THE USER
──────────────────────────────────────────────────────────────────────────
  K63-I DAG review · C18-C MassIVE-KB review · Pairformer study (notes/PAIRFORMER.md)
```
