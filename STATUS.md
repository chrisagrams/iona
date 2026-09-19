# Project status

The schedule for the denoise fine-tune and the reranking pipeline. Hand-kept, so it says
what is *true right now*; `TODO.md` says what is *wrong and open*, and the two are meant
to be read together.

Regenerate the job table with `pbs/job_history.sh`, which reads the logs rather than
anyone's memory. Last updated: 2026-09-19.

## Where things stand

```
INFRASTRUCTURE ─────────────────────────────────────────────────── all green
  denoise, 1 tile ................. OK   8840190  test AUROC 0.8628
  denoise, 12 tiles DeepSpeed ..... OK   8840264  21.4x one tile
  alignment, 1 tile ............... OK   8840304  full pipeline, saved
  alignment, 12 tiles DeepSpeed ... ??   8840336  RUNNING
  grid, 72 arms, 1 tile ........... OK   8840232  72/72
  grid, 216 arms, full pipeline ... ??   8840323  RUNNING  <- gates capacity

SCIENCE ───────────────────────────────────────── nothing real run yet
  216-arm HP grid ................. blocked on 8840323  ~84 node-hrs / ~5 h
  alignment, full 10 epochs ....... ready               ~35 min on 1 tile
  denoise re-runs: 100m, scratch,
    seeds ......................... unblocked, unsubmitted

BUGS (detail in TODO.md) ──────────────────────────────────────────────────
  FT7  GPU fault is DDP ........... worked around with DeepSpeed, NOT fixed
  FT4  32 stray labels ............ DDP gather only; absent under ZeRO-2
  FT3  failure reports "finished" . open
  FT1  >1024 peaks ................ open
  FT5  seeds on the winner ........ waiting on the grid
  FT6  from-scratch control ....... 8839946 is a v1 run, must be redone
  FT8  encoder warm-up freeze ..... deferred, set to 0 everywhere
  FT2  ............................ closed, not a bug
```

## The one real number so far

`test AUROC 0.8628` from a 700-step denoise run (8840190), against a free raw-intensity
baseline of about 0.75. Everything else this session was infrastructure. **No tuned model
and no reranking result exists yet.**

## What has to happen, in order

1. **8840323 passes** -> submit the 216-arm grid to capacity. ~84 node-hours, ~5 h on 16
   nodes. Axes: `learning_rate` x `encoder_lr_scale` x `num_train_epochs` x
   `head_hidden_size` x effective batch.
2. **Full alignment training.** Fits in the debug hour on one tile (~35 min) if
   `eval_steps` goes from 200 to 2000; 143 evals at the current setting would cost 28 of
   those minutes. Faster still if 8840336 shows DeepSpeed works here.
3. **Grid winner** -> FT5 (seeds) and FT8 (freeze), both of which need it.
4. **Denoise re-runs** on the fixed pipeline: 100m, from-scratch control, seeds.
5. **Reranking end to end**: features + embedding distance -> classifier, which needs a
   trained alignment tower first.

## Conventions worth not rediscovering

- **Runs are prefixed `v2_`.** Anything without it predates the DDP, dtype and eval fixes
  and is not comparable. `RUN_PREFIX` in both launchers and the generator.
- **Validate on debug before capacity**, and validate with `MAX_SAMPLES`, not `MAX_STEPS`.
  Capping steps skips saving, `save_total_limit`, `load_best_model_at_end`, the test split
  and the final save -- which is exactly where the alignment bugs were hiding.
- **DDP is broken on this stack.** One tile, or DeepSpeed ZeRO-2 on twelve. The sweep
  launcher refuses multi-tile arms that name no deepspeed config.
- **Every experiment carries a description**: auto-derived from settings into W&B notes
  and `RUN.md`, plus a hand-written `DESCRIPTION.md` beside each args file for intent.
- Jobs read configs from a snapshot taken at job start, so the working tree can be edited
  while a sweep runs.

## Job history

Regenerate with `pbs/job_history.sh`.

```
JOB       TASK     PARALLELISM   OUTCOME                    NOTE
8840154   denoise  12 tiles DDP  GPU FAULT at 168/700
8840190   denoise  1 tile        COMPLETE 700/700           test AUROC 0.8628
8840223   align    12 tiles DDP  GPU FAULT at 3/200
8840232   grid     1 tile/arm    72/72 arms ok
8840238   align    1 tile        GPU FAULT at 73/200        batch 16
8840257   align    1 tile        ERROR in eval              bf16 vs fp32 fused kernel
8840264   denoise  12 tiles DS   COMPLETE 300/300           21.4x, 0 faults
8840277   align    1 tile        ERROR after eval           no eval_loss
8840291   grid     12 tiles DS   refused: stale arms        guard worked
8840304   align    1 tile        COMPLETE 400/400           eval_loss 0.048
8840313   grid     12 tiles DS   refused: stale arms        guard worked
8840323   grid     12 tiles DS   RUNNING                    216-arm full pipeline
8840336   align    12 tiles DS   RUNNING                    does ZeRO-2 work here
```
