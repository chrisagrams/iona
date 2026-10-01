# Project status

What is true right now. Overwritten, not appended: before changing it, copy the old one into
`status-history/`. Questions: `PLAN.md`; results: `OBSERVATIONS.md`; decisions: `DECISIONS.md`;
waiting on the user: `OPEN_QUESTIONS.md` (top: "Your to-do").

Last updated: 2026-10-01 ~19:25 UTC (before a client reconnect). Every new setting needs the user's approval.

```
BRANCHES / CHECKOUTS
──────────────────────────────────────────────────────────────────────────
  ~/code/msdelta          dev_finetune_02 (working branch; main checkout). Jobs snapshot HEAD at START (memory
                          queued-jobs-snapshot-head): debug-test code before committing while jobs are queued.
  master/main             untouched (Chris)

DETACHED PROCESSES ON THE LOGIN NODE (setsid nohup; survive a session restart; logs in $S/logs/)
──────────────────────────────────────────────────────────────────────────
  feeder k185_p2u      K185/K186 runs + eval_mlm          log feeder_k185.log
  feeder k182d_tbase   K182 (d) transformer baselines     log feeder_k182d.log
  feeder k189_probe    K189 probe + grouped validation    log feeder_k189.log
  k188_gate.sh passed  starts feeder k188_cons_allck once every K185/K186 + K182 (d) training job has left the
                       queue                              log k188_gate.log
  feeder k160_cons     finished (only yeast scoring left, already submitted)
  Check: ps -u khuss -o pid,etime,args | grep -E "feeder|gate".  Restart (state files prevent double submission):
    setsid nohup pbs/tools/feeder.sh pbs/tools/feeder_plans/<plan>.txt >> $S/logs/feeder_<x>.log 2>&1 < /dev/null &
    setsid nohup pbs/tools/k188_gate.sh passed >> $S/logs/k188_gate.log 2>&1 < /dev/null &
  In-session watchers (scratchpad watch.sh) die with the session -- harmless, just re-check qstat.
  !! The detached helpers ALSO die when the user's last login session ends (login node not rebooted, uptime 3 d;
     loginctl Linger=no): 2026-10-01 ~19:30-20:00 all four died; restarted 20:01 UTC. Restart them after every
     reconnect.

THREAD C
──────────────────────────────────────────────────────────────────────────
  [x] K163-C consensus twins of the finals: trained (8882196) and scored on validation/test/oodval/mouse/human
      (OBSERVATIONS "K163-C"); [ ] yeast 8883903 queued. Consensus = default recipe (K188).
  [ ] K188-C consensus twins of all 78 K155 checkpoints: configs/sweep-cons (cons_allck.txt), smoke judged passed;
      one 4-node capacity job (~16 h) + six-set scoring, started by k188_gate.sh after the short P runs.
THREAD P
──────────────────────────────────────────────────────────────────────────
  [x] K180/K181 k sweep (OBSERVATIONS "K180-P"): L10 k10 0.0839 best.
  [ ] K185 (no pair updates: L10/L20 static, resubmitted after the train.py bug) and K186 (L20, 1/2/4 updates):
      8884427 L20 k10 running; L20 k20 8884397 queued; static reruns pending submission; eval_mlm after each.
  [ ] K182 (d) transformer baselines L10/L14/L20 x 512: 8885130, 8885131 queued, L20 pending.
  [ ] K189 length grouping (msdelta/pretraining/length_grouping.py; --length_grouped_batches, --compile_static_shapes):
      compiled 2-node speed at cap 512: Pairformer 1 update 0.560 -> 0.173 s/step, transformer 0.387 -> 0.133,
      2 updates 0.207 (grouped); 4 updates rerun 8886241 (debug). Validation run p2-L10-k10-grp (P2 data, vs 0.0839)
      pending submission (feeder k189_probe).
  [?] K187 card to present: full cap-512 build (~380 GB) + 10 x 640 Pairformer, 1 and 2 updates, 1 epoch
      (~160k steps at global 576, Chris's 3-epoch cosine stopped at 1 epoch, K191 a / K192 b), grouped + compiled,
      ~15 + ~18 node-h; compare with transformer-50m at 1 epoch (his 180k checkpoint: user to ask Chris).
WAITING ON THE USER: K187 card approval; Chris's 180k checkpoint; K182 (a) seeds; K179 ALCF question (not sent).

DONE RECENTLY
──────────────────────────────────────────────────────────────────────────
  K66-C 400m 8878459, 25m 8878461, K136 8878464: trained and scored on all six sets. lr 4e-4 P170xK2
    best on validation MAP@R at every scale; library Hit@1 agrees except 400m (P128) and 25m (lr 8e-4, noisy).
  C18 full MassIVE-KB prep 8879991 ok -> $S/data/massive-kb-contrastive (45 GB). Group statistics: to report.
  K152/K151: pair_tri_mul, factored write-back (default), pointwise write-back option; merged c2288d07.

QUEUE PROBES (user: keep queued): 8879772 next-eval, 8879810 legacy-reg (no nodes carry these labels).

REMINDERS FOR THE USER
──────────────────────────────────────────────────────────────────────────
  K63-I DAG review · Pairformer study (notes/PAIRFORMER.md) · K161-C parked option
```
