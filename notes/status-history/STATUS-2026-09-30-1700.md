# Project status

What is true right now. Overwritten, not appended: before changing it, copy the old one into
`status-history/`. Questions: `PLAN.md`; results: `OBSERVATIONS.md`; decisions: `DECISIONS.md`;
waiting on the user: `OPEN_QUESTIONS.md` (top: "Your to-do").

Last updated: 2026-09-30 ~17:00 UTC. The user is back: every setting needs approval again. Environment notes: notes/PRIMER.md, memory aurora-env-2026-09.

```
BRANCHES / CHECKOUTS
──────────────────────────────────────────────────────────────────────────
  ~/code/msdelta          dev_finetune_02 (working branch; main checkout)
  dev_finetune            frozen at paper submission; master untouched
  merges                  scratch worktree -> tests -> pbs/checkout_ff (K111); jobs snapshot a commit

THREAD C -- K160-C consensus twins; K155-C all-checkpoint run PAUSED
──────────────────────────────────────────────────────────────────────────
  [x] K155 8880712 paused 20:29 UTC (qdel). 27/78 done: 100m 220k/330k/430k, all 18 200m. 400m ~72%, 50m ~30%
      (resumable: RESUME_JOB=8880712). Unscored except via k160 (the 9 100m arms).
  [ ] 8882001 cons_train: 23 consensus twins, 1-node capacity, ~15 h (feeder k160_cons, PID 2055767, log
      $S/logs/feeder_k160.log); then scoring on six sets -> results/raw/finetune/contrastive/cons-<set>/
  [!] 400m final seed 0 twin held back: deterministic GPU page fault after step 20 in smoke (8881357, 8881848).
  Open: twins for the 18 finished 200m arms? Parked: K161-C scale both sets to every checkpoint.

THREAD P -- K156-P P2 half-epoch Pairformer runs (cz32 outgoing; k1, k1+tri-attn, k5)
──────────────────────────────────────────────────────────────────────────
  [x] p2_smoke 8880669 ok (all three arms train/eval/save; ~0.12-0.29 s/step at micro 3)
  [x] p2_k1 8880726 0.0907 / p2_k5 8880727 0.0926; [ ] p2_triattn 8880728 (~16k/23.4k at 16:35)
  [ ] K159-P 8881323 debug cost probe (micro 24/12/3 per tile)
  [ ] (was) p2_k1 8880726 / p2_k5 8880727 / p2_triattn 8880728  16-node capacity each (then eval_mlm of each final), stop at step 23,387, W&B CS_Pharm/pairformer_pretrain,
      runs under $S/runs/p2/, PBS logs $S/runs/p2/pbs-logs/
  FEEDER (detached): pbs/tools/feeder_plans/k156_p2.txt, log $S/logs/feeder_k156.log, state $S/feeder/k156_p2/.
    Restart: setsid nohup pbs/tools/feeder.sh pbs/tools/feeder_plans/k156_p2.txt >> $S/logs/feeder_k156.log 2>&1 < /dev/null &
  [x] eval_mlm merged; transformer 50m on P2 validation: 0.117 @10k, 0.077 @50k, 0.055 final (8880677)
  [x] 8880628  width profile (K154): pair update 33.5 ms at c_z=32 vs 4.7 ms per single block

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
