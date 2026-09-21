# Fine-tuning TODO

Open work on `dev_finetune`. Items are ordered by what would change a decision, not by
effort. This file is what is WRONG; [STATUS.md](STATUS.md) is the schedule and what is
currently true. Read them together.

## FT7. The GPU page fault is DDP, not the model — **Isolated, workaround in hand**

Six jobs died with `Segmentation fault from GPU ... type: 0 (NotPresent), level: 2 (PDP),
access: 1 (Write)` partway through training. Memory was ruled out empirically: the probe
sat flat at 14.7 GB peak / 45.8 GB reserved of 68.7 GB right up to the fault.

Two jobs, same branch, same args file, same fixed batching, same `max_peaks=512`, same
`LambdaLR` freeze gate. The only difference is the rank count:

| job | ranks | outcome |
| --- | --- | --- |
| 8840154 | 1 host x 12 tiles, DDP over `xccl` | 2 GPU faults, died at step 168/700 |
| 8840190 | 1 host x 1 tile, no DDP | 700/700 clean, eval completed |

So the fault needs the reducer. That also explains why it only started once the encoder
was genuinely being all-reduced: the early runs accidentally excluded it (the DDP freeze
desync), the reducer then held only the 0.082M-parameter head, and 0.082M parameters of
gradient never tripped it. Nothing in the model, the data pipeline or the optimiser is
implicated -- the identical code runs to completion on one tile.

**Workaround, available now:** every sweep arm already runs on one tile
(`TILES_PER_ARM=1` in `pbs/aurora-finetune-sweep.pbs`, twelve arms per node). That path is
proven and is strictly better throughput for a grid search than twelve-way DDP on one arm,
so the grid and the denoise re-runs are not blocked on this.

### Working theory: torch DDP on xccl, and reducer size is NOT the discriminator

The environment is exonerated. `aurora-pretrain.pbs` and `aurora-finetune.pbs` set the
same `CCL_PROCESS_LAUNCHER`, `CCL_ATL_TRANSPORT`, `CCL_KVS_MODE`, `FI_MR_CACHE_MONITOR`,
`ZE_FLAT_DEVICE_HIERARCHY` and the same two `unset`s. What differs is the parallelism:
**pretrain runs DeepSpeed ZeRO-2** (`--deepspeed configs/deepspeed-zero2.json` in every
`configs/msdelta-base-*/training.args`), which partitions gradients itself and never
constructs a PyTorch DDP `Reducer`. So xccl is fine -- ZeRO-2 reduce-scatters 200M+
parameters over it routinely -- and nothing in this repo had exercised DDP before these
fine-tunes.

Job 8840223 then killed the size-based explanation. The alignment model's reducer holds
only the 4.11M-parameter student (the teacher is `requires_grad=False` and excluded), and
it faulted anyway -- at step 3 of 200, not 168:

| run | reducer holds | steps survived | fault |
| --- | --- | --- | --- |
| denoise, head only (8840007 era) | 0.082M | full runs | none |
| **align, student only (8840223)** | **4.11M** | **3** | `level: 0 (PTE)`, `access: 0 (Read)` |
| denoise, encoder in (8840154) | 49.9M | 168 | `level: 2 (PDP)`, `access: 1 (Write)` |
| denoise, 1 tile (8840190) | n/a, no DDP | 700 + eval + test | none |

Two different fault levels and two different access types, so plausibly two distinct
failure modes rather than one. What holds across all of it is narrow and worth stating
plainly: **any run whose DDP reducer is non-trivial faults; one tile never does.** Bucket
size, model, task and step count all vary; the presence of the reducer does not.

The step-3 timing is suggestive. DDP calls `rebuild_buckets()` exactly once, after the
first iteration, reallocating the bucket tensors and re-registering the gradient hooks --
which lands at iteration 2-3. A read fault at the *same* address on two ranks also looks
more like a broadcast or a bucket view reading a buffer that is not mapped on that rank
than like a random use-after-free. But a single data point is not enough to call it, and
the step-168 write fault does not fit the same story.

Probes, cheapest first, one debug run each:

- [ ] `--ddp_bucket_cap_mb 1024` so there is a single bucket and the rebuild is a no-op.
- [ ] `--ddp_broadcast_buffers false` -- tests the read-fault-on-broadcast reading.
- [ ] `TORCH_XPU_ALLOC_CONF=expandable_segments:False`, which stops segments being
      unmapped. Diagnostic, not a fix: if the fault goes away and the loss goes strange,
      it is a use-after-free.
- [ ] `CCL_ATL_TRANSPORT=ofi`.

**The multi-tile answer IS DeepSpeed. Confirmed by job 8840264:** the same denoise
config with `--deepspeed configs/deepspeed-zero2.json`, twelve tiles, 300 steps plus eval
plus test, **zero faults**, and `label_dropped`/`label_extra` of 0 -- so ZeRO-2's gather is
clean where DDP's is not, which also points at [FT4](#ft4) being a DDP-reducer problem
specifically. ZeRO-2 needs no change to the fine-tune script: the config carries no
`optimizer` or `scheduler` key, so Trainer's param groups and the `encoder_lr_scale`
LambdaLR gate survive untouched.

It is also 21.4x the throughput of one tile (235.4 vs 11.02 samples/s), which retires the
sweep's cost problem -- a 4-epoch arm goes from 8.8 h on a tile, or 21.2 h once twelve
arms contend for one node, to **0.41 h on a node**. The whole 72-arm grid is then about 22
node-hours, which is what `make_denoise_grid.py` estimated all along; the estimate was
never wrong, it just assumed the full-node parallelism that the DDP bug had taken away.

Caveat when switching the grid over: twelve tiles means an effective batch of 48 rather
than 4, so 12x fewer optimizer steps, and anything denominated in steps has to be rescaled
or it silently becomes a different experiment. `freeze_encoder_steps=500` is 1.1% of a
2-epoch run at batch 4 and 13.8% of the same run at batch 48. Same for `warmup_steps`.

**Nothing currently queued depends on DDP.** The grid runs one tile per arm and the
alignment validation was resubmitted as 8840238 on one tile.

## FT9. Twelve-tile alignment faults in the driver's scratch surface — **PARKED, for ALCF**

Not our code, not worth more of our time, and a good bug report for people with
driver-side tools. One tile runs the full ten epochs in seven minutes, so this is
throughput rather than a blocker.

**What the fault is.** `Segmentation fault from GPU at 0xff0N_ffffe00000, NotPresent,
level 1 (PDE), access 0 (Read)`, where N tracks the tile. That address is at the top of
each tile's virtual address space, which is where Intel places **scratch / private
memory** -- not user allocations. So no tensor of ours was ever the target: a kernel was
reading its own private surface at an address that had stopped being mapped.

**Every attempt, and how far it got:**

| configuration | fault at step |
| --- | --- |
| DDP, live teacher, batch 16 | 3 |
| ZeRO-2, live teacher, batch 4 | 56 |
| + targets precomputed | 22 |
| + fixed-shape batches | 120 |
| + stock optimizer | 206 |
| + ALCF fabric environment | **802** |
| one tile, any configuration | never |

Every change pushes it later and none prevents it, which is what an ACCUMULATING cause
looks like: each fix lowers allocation pressure and so delays whatever sweep unbinds the
surface, without supplying the missing re-bind.

**Eight hypotheses, all dead.** The DDP reducer (fails under ZeRO-2 too), the frozen
teacher in the graph (detached, still faults), bad hardware (three nodes), an
`nn.TransformerEncoder` fused path (the bisect ran four layers and the full student
clean), variable-shape batches (the working denoise job has them), abandoning the
teacher on the device (bisect variant clean), oneDNN fused SDPA
([pytorch#195319](https://github.com/pytorch/pytorch/issues/195319), MATH backend still
faults), and the private-surface eviction workarounds from
[intel/compute-runtime#973](https://github.com/intel/compute-runtime/issues/973) --
`UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` faults at address 0x0 on an atomic, which is
unrelated breakage, and `MakeEachAllocationResident=2` still faults on a high-address
read.

#973's MECHANISM still fits better than anything else -- a per-dispatch private surface
declared resident only at allocation, unbound by `evictUnusedAllocations()`, then reused
without re-binding -- and it explains the address, the accumulation and why less churn
buys more steps. Its published workarounds simply do not help on this driver.

- [ ] Report to ALCF support with: the address pattern, the step-count table above, and
      the fact that the same model under a raw training loop on the same twelve tiles is
      clean (`pbs/bisect_align.py`), so it needs the HF Trainer / accelerate stack to
      appear. They have `gdb-oneapi` and driver instrumentation; we have bisection, and
      bisection has run out.
- [ ] Re-test if the frameworks module is updated. #973's fix (commit 3d7a21dca9,
      2026-08-27) was unreleased as of that issue's last update.

## FT8. Is a warm-up freeze on the encoder worth anything? — **Open, deferred**

`freeze_encoder_steps` held the encoder still for the first N steps so the randomly
initialised head could not push large gradients back through pretrained weights before it
had learned anything itself. It is set to **0** everywhere now and the grid does not vary
it, deliberately: it is a second mechanism doing roughly what `encoder_lr_scale` already
does, the grid sweeps that over `0, 0.1, 0.5, 1.0`, and carrying both would confound them
-- an arm at `encoder_lr_scale=0.1` with a 500-step freeze is not cleanly either.

It also did not survive the move to twelve tiles unchanged: at an effective batch of 48
the same 500 steps is 13.8% of a 2-epoch run rather than 1.1%, so the setting silently
meant something different.

- [ ] Once the grid names a winning `learning_rate` / `encoder_lr_scale`, run that arm
      with a freeze of 0 / 1% / 5% of total steps and see whether it moves test AUROC.
      Express it as a FRACTION of training, not a step count, so it survives a change of
      batch size or tile count.
- [ ] If it does help, check whether it still helps at `encoder_lr_scale=0.1`, where the
      encoder is already moving slowly, or only at `1.0`.

## FT1. Does the denoiser generalise beyond 1024 peaks? — **Open**

`max_peaks=1024` and `build_denoising_datasets` DROPS anything above it rather than
truncating, so 1.62% of train, 1.58% of validation and 1.71% of test never reach the
model. Those are not a random 1.6%: peak count correlates with noise, so the excluded
spectra are the largest and the noisiest — systematically the hardest cases. The reported
test AUROC of 0.9354 is therefore measured on a corpus that is slightly easier than the
real one, and says nothing at all about the regime it excludes.

Two distinct questions, and the second is the interesting one:

  **Coverage.** Retrain at a higher cap so those spectra are included. Costs pair memory
  quadratically — the Fourier tensor over every (i, j) m/z difference is
  `batch * L^2 * 2 * n_freqs`, so 2048 is 4x the memory of 1024 at equal batch and
  `peak_pair_budget` has to fall by the same factor.

  **Extrapolation.** Train at 1024 and evaluate on the >1024 spectra without retraining.
  This is the real generalisation question: the encoder has a learned per-head Δm/z bias
  rather than absolute position embeddings, so there is a genuine reason to expect it to
  extend past its training length — and a genuine reason to doubt it, since the bias MLPs
  have only ever seen the Δm/z distribution of shorter spectra.

Ordering matters: run extrapolation FIRST. It is one evaluation pass on an existing
checkpoint, it needs no training, and if the model already extends cleanly then the
expensive retrain buys nothing.

- [ ] Build the held-out >1024 slice (1,620 train / 159 validation / 169 test spectra)
      and score the existing 50m fine-tune on it with no retraining. Report AUROC/AUPRC
      against the <=1024 test number, and bucket by peak count so any degradation is
      visible as a trend rather than one average.
- [ ] Check whether degradation tracks peak count or noise fraction. Those are
      correlated, so a drop at high peak counts may be the task getting harder rather
      than the model failing to extrapolate. Stratify to separate them.
- [ ] Only if extrapolation fails: retrain at `max_peaks=2048` with
      `peak_pair_budget=262144`, and compare on the SAME >1024 slice.
- [ ] Decide whether `build_denoising_datasets` should truncate instead of drop. It
      currently discards whole spectra; truncating by m/z keeps them but silently removes
      the high-m/z tail, which is where the large fragment ions live.

## FT6. Ablation: 50m trained from scratch — **SUBMITTED (job 8839946)**

Every fine-tuned number is uninterpretable without this control. If a randomly
initialised encoder of the same architecture also reaches ~0.935 test AUROC, pretraining
contributed nothing to denoising, the head simply learned the task from the labels, and
both the zero-shot investigation and the 50m-to-100m scaling result lose their point.

`configs/finetune-denoise-50m-scratch/training.args`, `--random_init true`. The
architecture is read from the 50m checkpoint's config; its tensors are never loaded, so
no buffer can retain a pretrained value that re-initialisation happens to miss.

`freeze_encoder_steps` drops to 0 and `encoder_lr_scale` rises to 1.0 for this arm alone.
Those defaults exist to shield pretrained weights from an untrained head's early
gradients; with no pretrained weights there is nothing to shield, and holding a random
encoder still for 500 steps while a head trains on its noise would handicap the control
rather than make the comparison fair.

- [x] Submit (8839946).
- [ ] Report against the pretrained 50m (0.93543) and the raw-intensity baseline (0.751).
- [ ] If it lands near 0.935: the task is learnable from labels alone and pretraining is
      not what produced the result. Say that plainly rather than burying it.
- [ ] If it lands well short: that gap IS the value of pretraining for denoising, and it
      is the number worth quoting, not the absolute 0.935.
- [ ] Consider the same control at 100m. The scaling result (+0.0047) only means
      "pretrained capacity helps" if a from-scratch 100m does NOT show the same gain --
      otherwise it is just a bigger model fitting the labels better.

## FT5. Multi-seed denoise on every scale — **Open, deprioritised**

Not urgent while the 200m, 400m and scratch grids are queued; those answer questions we
do not have answers to at all, where this sharpens ones we do. Keep it on the list.

WHY IT IS NOT OPTIONAL EVENTUALLY. Every grid ranks its arms on ONE seed each, and the
gaps being ranked are tiny:

  50m, 216 arms: the top eight span 0.0023 test AUROC, across three head widths and two
                 encoder_lr_scales. h128/h256/h512 at the same lr and scale gave
                 0.9319 / 0.9319 / 0.9320.
  100m, 12 arms: the top five span 0.0012, across lr 5e-5 to 5e-4 and both batch sizes.

If seed spread is comparable to those, the rankings are largely noise and each "winner"
is whichever arm drew a good seed. The replication is what licenses the word winner. It
also decides whether 100m's 0.9403 genuinely beats 50m's 0.9320 -- a gap of 0.0083, only
about 4x the within-grid top-cluster spread.

SHAPE. `sweeps/make_seed_grid.py --sizes ... --seeds N`, which now takes any set of
scales. Scales x seeds wants to be a multiple of 12 to fill a node-wave exactly:

  50m + 100m, 6 seeds     12 arms, one wave   <- generated and ready now
  all four scales, 3 seeds 12 arms, one wave
  all four scales, 6 seeds 24 arms, two waves <- the version worth having

DEPENDENCY, and the reason it cannot just be run now: each scale must repeat its OWN
winning configuration, and 200m and 400m have not got one yet -- their grids are 8841992
and 8842147. Generating 200m/400m seed arms today would only repeat a guess. Run this
after those land.

What varies is `--seed` alone: head initialisation, data order, dropout. The
train/validation/test split comes from the dataset, not the seed, so every arm is scored
on identical rows and the spread measures training noise and nothing else.

- [ ] After the 200m and 400m grids report, regenerate with all four scales.
- [ ] Report mean +/- sd per scale, and say plainly whether the top group of each grid
      is tied rather than ranked.
- [ ] Only then compare scales to each other, normalised by compute budget
      (`results/checkpoint_provenance.txt` has the pretraining step each started from).

## FT2. Test metrics never reach W&B — **NOT A BUG (closed)**

They were always there. HuggingFace's WandbCallback groups metrics into sections, so
`test_auroc` is logged as `test/auroc`. The original report queried the underscore form,
got None, and concluded the metrics were missing.

    test/auroc = 0.9354279541485844   test/auprc = 0.96199889848495

- [x] Verified: every `test/*` key is present in the run summary. No change needed.

## FT3. A Python-level failure reports `finished` in W&B — **Open**

Narrower than first written. W&B marks a run `crashed` by missing heartbeat, i.e. only
when the process dies WITHOUT calling finish(). A hard crash is therefore labelled
correctly -- 8839166 and 8839203 both died on a GPU page fault and both show `crashed`.

The mislabelled case is an exception that propagates through the `finally` block, which
calls finish() on the way out and stamps the run `finished`. 8839579 died on the metrics
ValueError and is indistinguishable from a completed run.

It matters at sweep scale: with 72 arms the W&B run list is the index. An arm that died
at step 500 sits beside a completed one, carrying plausible partial metrics, with nothing
to tell them apart.

- [ ] Call `finish(exit_code=1)` when main() raises, and keep the bare finish() only on
      the success path.

## FT4. 32 stray label values survive the distributed gather — **Confirmed as the gather**

`denoise_metrics` drops 32 labels per evaluation that are neither 0, 1 nor -100 —
bf16-quantised floats in the 4.09-5.22 range, the same 32 every time. That is 0.0013% of
2.5M peaks and has no measurable effect, but nothing should be writing those values into
a label buffer. A single-process eval and a two-rank gloo eval are both clean, so it
appears only at twelve ranks with bf16.

Job 8840190 closes the loop on the diagnosis: one tile, no distributed gather, a full
700-step train plus eval plus test over 1.68M peaks, and `label_dropped` / `label_extra`
are both exactly **0**. Single process clean, two-rank gloo clean, one-tile xpu clean,
twelve-rank xccl dirty -- it is the gather, not the loss, the model or the data.

- [ ] Find the source. Suspect the padding index used when gathering variable-length
      label tensors across ranks. Shares a root with [FT7](#ft7-the-gpu-page-fault-is-ddp-not-the-model--isolated-workaround-in-hand):
      both only appear once tensors cross ranks, so whichever is fixed first should be
      re-checked against the other.

## FT10 — RETRACTED: warmup never exceeded the run — **Closed, was not a bug**

The claim was that `--warmup_steps 100` exceeded the entire contrastive run, so the rate
ramped from zero and training ended before cosine decay engaged. It was derived from
`train_groups=60`, read out of job 8840665 -- a SMOKE run with `max_samples` applied.

The real corpus has **898 training groups**. At `groups_per_batch 2` that is 449 batches
an epoch and **1,347 optimizer steps** over 3 epochs, confirmed independently by the
run's own `train_runtime 264s x 5.101 steps/s`. Warmup 100 is 7.4% of that: entirely
normal, and never a bug.

What actually happened: warmup was changed from 100 to 5 for a wrong reason, and the
re-run scored 7.83 against the previous 6.94. That change is real but its explanation is
not established -- 7.4% to 0.4% warmup is a modest change and there are no error bars on
either number. Do not cite "the scheduler was broken" as the cause of the improvement.

- [ ] Re-derive a sensible warmup for 1,347 steps (100, i.e. the original, is defensible)
      and decide whether 5 is actually better, with seeds, rather than assuming.
- [x] Corpus size verified against the dataset rather than a smoke log.

See OBSERVATIONS.md, "Reading a smoke-run log as a real run cost three wrong
conclusions", for the other two things this number broke.

## FT5. Multi-seed denoise on every scale — **Open, deprioritised**

Not urgent while the 200m, 400m and scratch grids are queued; those answer questions we
do not have answers to at all, where this sharpens ones we do. Keep it on the list.

WHY IT IS NOT OPTIONAL EVENTUALLY. Every grid ranks its arms on ONE seed each, and the
gaps being ranked are tiny:

  50m, 216 arms: the top eight span 0.0023 test AUROC, across three head widths and two
                 encoder_lr_scales. h128/h256/h512 at the same lr and scale gave
                 0.9319 / 0.9319 / 0.9320.
  100m, 12 arms: the top five span 0.0012, across lr 5e-5 to 5e-4 and both batch sizes.

If seed spread is comparable to those, the rankings are largely noise and each "winner"
is whichever arm drew a good seed. The replication is what licenses the word winner. It
also decides whether 100m's 0.9403 genuinely beats 50m's 0.9320 -- a gap of 0.0083, only
about 4x the within-grid top-cluster spread.

SHAPE. `sweeps/make_seed_grid.py --sizes ... --seeds N`, which now takes any set of
scales. Scales x seeds wants to be a multiple of 12 to fill a node-wave exactly:

  50m + 100m, 6 seeds     12 arms, one wave   <- generated and ready now
  all four scales, 3 seeds 12 arms, one wave
  all four scales, 6 seeds 24 arms, two waves <- the version worth having

DEPENDENCY, and the reason it cannot just be run now: each scale must repeat its OWN
winning configuration, and 200m and 400m have not got one yet -- their grids are 8841992
and 8842147. Generating 200m/400m seed arms today would only repeat a guess. Run this
after those land.

What varies is `--seed` alone: head initialisation, data order, dropout. The
train/validation/test split comes from the dataset, not the seed, so every arm is scored
on identical rows and the spread measures training noise and nothing else.

- [ ] After the 200m and 400m grids report, regenerate with all four scales.
- [ ] Report mean +/- sd per scale, and say plainly whether the top group of each grid
      is tied rather than ranked.
- [ ] Only then compare scales to each other, normalised by compute budget
      (`results/checkpoint_provenance.txt` has the pretraining step each started from).

## FT2. Test metrics never reach W&B — **NOT A BUG (closed)**

They were always there. HuggingFace's WandbCallback groups metrics into sections, so
`test_auroc` is logged as `test/auroc`. The original report queried the underscore form,
got None, and concluded the metrics were missing.

    test/auroc = 0.9354279541485844   test/auprc = 0.96199889848495

- [x] Verified: every `test/*` key is present in the run summary. No change needed.

## FT3. A Python-level failure reports `finished` in W&B — **Open**

Narrower than first written. W&B marks a run `crashed` by missing heartbeat, i.e. only
when the process dies WITHOUT calling finish(). A hard crash is therefore labelled
correctly -- 8839166 and 8839203 both died on a GPU page fault and both show `crashed`.

The mislabelled case is an exception that propagates through the `finally` block, which
calls finish() on the way out and stamps the run `finished`. 8839579 died on the metrics
ValueError and is indistinguishable from a completed run.

It matters at sweep scale: with 72 arms the W&B run list is the index. An arm that died
at step 500 sits beside a completed one, carrying plausible partial metrics, with nothing
to tell them apart.

- [ ] Call `finish(exit_code=1)` when main() raises, and keep the bare finish() only on
      the success path.

## FT4. 32 stray label values survive the distributed gather — **Confirmed as the gather**

`denoise_metrics` drops 32 labels per evaluation that are neither 0, 1 nor -100 —
bf16-quantised floats in the 4.09-5.22 range, the same 32 every time. That is 0.0013% of
2.5M peaks and has no measurable effect, but nothing should be writing those values into
a label buffer. A single-process eval and a two-rank gloo eval are both clean, so it
appears only at twelve ranks with bf16.

Job 8840190 closes the loop on the diagnosis: one tile, no distributed gather, a full
700-step train plus eval plus test over 1.68M peaks, and `label_dropped` / `label_extra`
are both exactly **0**. Single process clean, two-rank gloo clean, one-tile xpu clean,
twelve-rank xccl dirty -- it is the gather, not the loss, the model or the data.

- [ ] Find the source. Suspect the padding index used when gathering variable-length
      label tensors across ranks. Shares a root with [FT7](#ft7-the-gpu-page-fault-is-ddp-not-the-model--isolated-workaround-in-hand):
      both only appear once tensors cross ranks, so whichever is fixed first should be
      re-checked against the other.

## FT10 — warmup_steps exceeds the whole run in every contrastive config

`configs/finetune-contrastive-50m/training.args` sets `--warmup_steps 100`. The
replicate corpus has 60 training groups, so the PK sampler with `groups_per_batch 2`
yields 30 batches an epoch, and `--num_train_epochs 3` is about 90 optimizer steps.

The learning rate therefore ramps from zero and the run ends at ~90% of nominal. Cosine
decay never engages, and the mean rate actually applied is roughly 0.45x the number in
the config. Every contrastive result in this project was produced that way, including
the 6.94 separation ratio, so the numbers stand -- but the `learning_rate` in those
configs is not the rate that was used, and any lr sweep over them was compressed into
the warmup ramp, which is a poor way to separate learning rates.

Not fixed yet on purpose: changing it changes the encoder's effective rate and would
break comparability with the 6.94 baseline that everything is measured against. Fix it
deliberately, with the baseline re-run alongside, rather than as a side effect.

Found while sizing `layer_mix_lr`, which had to be raised to 5e-2 to move at all inside
90 warmup-damped steps.

### FT10 update (2026-09-20)

Fixed for the contrastive family: `--warmup_steps 100` replaced by `--warmup_ratio 0.06`
across all 27 contrastive-family arm configs, so the schedule scales with the run and
cannot be wrong again if the epoch count changes. The mean applied rate barely moves
(0.445x -> 0.500x), but the PROFILE was the real damage: the old schedule ended the run
at 0.89x peak with cosine decay never engaging, so training stopped while still taking
near-maximum steps.

Denoise configs keep `warmup_steps 100` deliberately -- those runs are ~29,000 steps, so
100 is 0.3% and entirely appropriate. `finetune-align-contrastive` keeps 200 for the same
reason (~28,600 steps).

Contrastive baseline re-run under the corrected schedule: job 8842154. Until it lands,
every contrastive number in STATUS.md -- including the 6.94 -- was produced under the
warmup-only schedule.

## FT11. Make the layer mixture resist collapse structurally — **CLOSED, not needed**

Job 8842351 produced a genuine blend at `layer_mix_lr 3e-3` without any of this:
final entropy 2.065 against 2.398 uniform, weight spread over five depths. The
decision table below resolved to "nothing needed, the result stands".

And the result is that blending is WORSE than one layer -- 1.49 / 2.18 / 4.14 against
the collapsed run's 1.52 / 3.46 / 4.28. So there is nothing to protect: making the
mixture harder to collapse would only make it more reliably worse. Kept for the record
because the options are correct if a mixture is ever wanted for another purpose.

Original ticket follows.


Only worth doing if the corrected layer-mix run lands ambiguously. Read `mix/entropy`
at the end of each arm first (uniform over 11 depths is ln(11) = 2.398, one-hot is 0):

| entropy | what happened | action |
| --- | --- | --- |
| ~0 | collapsed again despite the lower rate | do this ticket |
| 0.5 - 1.5 | a genuine blend was trained | nothing needed, the result stands |
| ~2.4 | never moved; 3e-3 is too low for this run | raise the rate, do not add machinery |

WHY THE CURRENT FIX IS WEAK. `layer_mix_lr 3e-3` does not prevent collapse, it runs out
of budget before reaching it. Adam displaces a parameter by about `lr` per step when the
gradient sign is consistent, so total logit travel is bounded by
`lr x steps x mean_schedule_multiplier` -- about 2.0 over this 1,347-step run, which
caps the top weight near 0.43. That is pacing, not structure. It depends on the step
count, which changes whenever epochs or the corpus change, and on an estimate that was
wrong once already (it predicted 0.43 at 5e-2 and reality was 1.000 -- because the step
count fed to it was 90 rather than 1,347; with the right count it retrodicts the
collapse correctly, which is weak evidence it is calibrated, not proof).

The band is narrow: 1e-2 still collapses, 1e-3 barely moves. Anything that shifts the
run length walks out of it silently.

OPTIONS, cheapest first:

  Entropy penalty. Add `-lambda * H(w)` to the loss, so a mixture pays to concentrate
  and collapse has to be earned rather than drifted into. One term, one hyperparameter,
  and `mix/entropy` is already logged to tune it against. Downside: lambda is another
  thing to get right, and a large one forbids the single-layer answer even when that
  answer is correct.

  Softmax temperature. `softmax(z / T)` with T > 1 flattens the mapping so the same
  logit travel produces less concentration. Equivalent to rescaling the learning rate
  for this purpose, so it fixes the same problem the same way -- prefer it only if a
  fixed T is easier to keep right across run lengths than a fixed lr, which it is.

  Report the trajectory, not the endpoint. Cheapest of all and worth doing regardless:
  `mix/entropy` per step is already logged, so a mixture that collapsed at step 400 can
  be told from one that annealed smoothly, and the arm can be judged accordingly rather
  than silently averaged in. Currently only the final value is read.

  Reparameterise away from softmax. Non-negative weights normalised by their sum, or
  plain unconstrained weights with the scale absorbed by the LayerNorm, have no
  saturating region and therefore no vanishing-gradient trap: a losing layer keeps a
  gradient proportional to its usefulness rather than to its current weight. Biggest
  change, and the only one that removes the failure mode rather than avoiding it.

- [ ] Read `mix/entropy` from 8842351 and decide using the table above.
- [ ] If acting: entropy penalty first, reparameterisation only if that is not enough.
- [ ] Either way, record the entropy trajectory alongside the ratio so a collapsed arm
      is never reported as a blend again.

## FT12. The sign of the pooled-vs-within AUROC gap is not predictable by reasoning — measure it

Not a bug: a warning about how to treat the result of job 8842917, and about a class of
claim that keeps going wrong here.

Pooled AUROC ranks a peak against peaks from OTHER spectra; the model is used one
spectrum at a time. A per-spectrum offset in the logits moves the pooled number without
touching within-spectrum discrimination. WHICH WAY it moves depends on whether that
offset correlates positively or negatively with the spectrum's noise fraction, and that
is not something to settle by argument. Three constructions, all built while trying to
demonstrate the SAME effect:

| construction | pooled | within-spectrum | gap |
| --- | --- | --- | --- |
| one spectrum correct, one inverted and shifted up | 0.50 | 0.50 | none, they cancel |
| mostly-signal low, mostly-noise high, ordering correct | 0.36 | 1.00 | pooled DEFLATED |
| same logits, labels reversed so ordering is inverted | 0.64 | 0.00 | pooled INFLATED |

The last is the committed regression test: a model useless inside every spectrum scoring
0.64 on the headline metric. I reasoned about the sign three times and got it right once.

CONSEQUENCE FOR READING 8842917: do not predict the direction, and do not treat
"pooled and within agree" as the expected outcome or "they differ" as alarming. Read the
numbers. The quantities to look at together are `auroc_per_spectrum`, its `_sd` and
`_p10`, and `spectra_unscorable` -- a high mean with a low p10 means the model works on
most spectra and fails badly on a subset, which a single average hides.

WIDER POINT, worth remembering beyond this metric. This is the third time in this
project a pooled measurement has been trusted where a within-group one was the relevant
thing. The reranking embedding scored 0.846 pooled and cost 0.109 hit@1. The frozen
separation ratio pooled over all peptide pairs and did not predict fine-tuned
performance at all. Whenever a metric averages over a grouping the model does not see at
inference, check the within-group version before drawing a conclusion.

- [ ] Read 8842917 for the 50m and 100m winners; record both numbers in OBSERVATIONS.md.
- [ ] If they diverge, re-rank the grids by `auroc_per_spectrum` and check whether the
      winner changes. The top eight of the 50m grid span 0.0023 pooled, so a small
      systematic difference could reorder them.
- [ ] Backfill 200m and 400m winners the same way once those grids finish.

## FT13. The pretrained-vs-scratch denoise comparison has an unmeasured cell — **Open, cheap**

The two grids did not sweep the same epoch counts, so there are two valid comparisons
and one gap:

| epochs | pretrained 50m | scratch 50m |
| --- | --- | --- |
| 2 | 0.9272 | not run |
| 4 | **0.9320** | 0.8856 |
| 8 | **NOT RUN** | **0.9001** |

  matched at 4 epochs ......... pretraining worth +0.046
  scratch at double budget .... pretraining worth +0.032

Both are honest and they answer different questions; reporting only one is misleading
in a predictable direction. What is missing is pretrained at 8 epochs. Without it,
+0.032 cannot be called pretraining's advantage at equal wall-clock, because the
pretrained model might gain from the extra epochs too.

Probably it would not gain much -- ep2 to ep4 bought it only +0.005 against the random
encoder's +0.015 from ep4 to ep8, which is what saturation looks like. But that is an
inference and the measurement costs one 6-arm job.

- [ ] Run the pretrained 50m winner at 8 epochs, ideally with seeds so it can be read
      against FT5's spread rather than as a single number.
- [ ] Report all three cells together; never quote +0.032 alone.
- [ ] Same question applies at 100m/200m/400m if a scratch ablation is ever run there.

## FT14. The PK sampler never reshuffles — every epoch replays identical batches — **Open, real**

`GroupBatchSampler.__iter__` builds its RNG as `default_rng(self.seed + self.epoch)`,
and `set_epoch` is **never called anywhere in the codebase**. HuggingFace's Trainer calls
`set_epoch` on a DistributedSampler, not on a `batch_sampler`, and `get_train_dataloader`
passes this in as `batch_sampler=`. So `self.epoch` stays 0 for the whole run and every
epoch draws the identical permutation and the identical `rng.choice` replicates.

Verified directly: three consecutive iterations of the sampler produce byte-identical
batch lists, and `set_epoch(1)` does change them.

WHY IT MATTERS MORE HERE THAN IT WOULD ELSEWHERE. For ordinary supervised training,
replaying a fixed order costs some regularisation. For a PK sampler under a contrastive
loss it costs the objective itself: the point is that each epoch pairs different groups
against each other, so every group meets new negatives. Fixed batches mean a group only
ever sees the same 5 negatives, for all 3 epochs. "3 epochs" is closer to 1 epoch of
unique comparisons repeated three times.

That plausibly explains the earlier finding that MORE epochs made the ratio WORSE
(6.94 at 3 epochs, 4.82 at 10, 4.46 at 50) -- more passes over an identical batch
sequence is overfitting to a fixed set of contrasts, not additional learning.

- [ ] Call `set_epoch` from `ContrastiveTrainer`, or seed from a counter incremented in
      `__iter__`. The second is more robust since it cannot be forgotten by a caller.
- [ ] Re-run the epochs sweep afterwards: the "more epochs hurts" result is suspect and
      may reverse.
- [ ] Every contrastive number on record was produced under this, so none of them are
      wrong as measurements -- they just measure a weaker training procedure than
      intended.

## FT15. Contrastive results are not reproducible run to run — **Open, measuring**

Byte-identical config and seed, two runs, 7.83 (job 8842232) and 6.01 (job 8843838).

SOURCES RULED OUT by inspection:
  train/validation split .. seeded explicitly via `seed=training_args.seed`
  PK sampler .............. deterministic, and fixed across epochs (FT14)
  separation metric ....... `list(dataset)[:max_rows]`, a prefix, not a sample

REMAINING SOURCE: the training compute itself. Nothing in this repo sets
`torch.use_deterministic_algorithms`, and XPU reductions use atomics whose accumulation
order varies run to run. Small float differences then get amplified by the optimisation.

WHY THE AMPLIFICATION IS LARGE HERE, and this is the part worth acting on. The best arm
sits at lr 5e-4, which is the least stable setting in the grid:

  lr 2e-5 .. arms span 5.33 - 7.71   spread 2.38
  lr 1e-4 .. arms span 5.66 - 7.14   spread 1.48
  lr 5e-4 .. arms span 1.35 - 7.83   spread 6.48   <- includes a total collapse

At 5e-4 one arm reaches the best score in the grid and another collapses to the floor.
That is the signature of training at the edge of stability, where a float-level
difference decides which side of the edge a run lands on. So "lr 5e-4 is best" may mean
"lr 5e-4 has the highest variance and we sampled its upper tail once".

- [ ] Read job 8844111: the same config six times, nothing varied. That gives the error
      bar and says whether variance scales with the mean.
- [ ] If variance is lr-dependent as suspected, repeat a 2e-5 arm too and prefer the
      configuration with the best MEAN, not the best single draw.
- [ ] Consider `dataloader_num_workers 0` and `torch.use_deterministic_algorithms(True)`
      for comparison runs specifically; both cost throughput and neither is wanted for
      production training.
- [ ] Re-read every contrastive comparison in STATUS.md and OBSERVATIONS.md against the
      measured bar. Gaps under ~2 are currently unsupported.

## FT16. Two tiles share one card's HBM and both over-report it — **Diagnosed, fix in the launcher**

A node presents 12 tiles and has 6 physical cards. An Intel Max 1550 exposes two tiles
per card which SHARE that card's 128 GB of HBM, and each tile reports 68.7 GB
independently. A card's pair therefore claims 131 GB of a physical 128, with nothing
enforcing the real budget. When both siblings expand into it one takes a GPU fault at
0xff00...., which looks like a scratch-memory bug and is over-subscription.

Measured on the same 200m contrastive arm:

  12 arms per node (sibling active) .. reserved 67.11 GB/tile, 1 of 6 arms survived
  alone on a tile ..................... reserved 38.75 GB, 200/200 steps clean

Peak was 28.4 vs 28.5 GB either way, so the model always fitted in 68.7 GB.

FIX: `TILE_STRIDE=2` in pbs/aurora-finetune-sweep.pbs uses tiles 0,2,4,6,8,10 -- six
arms per node, one per card. Verified in job 8845548.

MEASURED LEVERS, same arm alone on a tile for 200 steps:

  max_peaks 512 .. peak 28.50 GB, reserved 38.75 GB -> a pair is 77.5 GB
  max_peaks 256 .. peak  9.89 GB, reserved 24.82 GB -> a pair is 49.6 GB

So halving the sequence length would also fit two siblings on a card and keep 12 arms
per node -- but it discards every spectrum over 256 peaks, and 512 already costs 19.4%
of the alignment pairs. TILE_STRIDE buys the same safety for free. Keep max_peaks 256
as the fallback only if a job genuinely needs all twelve tiles.

Note also that the grid reserved 67.11 GB where this same config reserves 38.75 GB
alone: the allocator grows MORE aggressively under contention, so the failure compounds
rather than merely being tight.

WHEN IT IS NEEDED: only when a pair would exceed 128 GB. Denoise under ZeRO-2 reserves
7.7 GB/tile, contrastive at 100m reserves 26 GB; both are fine at stride 1. Contrastive
at 200m+ reserves 67 GB and needs stride 2.

DO NOT RETRY: `PYTORCH_ALLOC_CONF=expandable_segments:True` would have let the allocator
give memory back and kept 12 arms per node. It is NOT SUPPORTED on this XPU build --
it fails immediately with "RuntimeError: could not create a memory". Tested, job on
node x4407c2s0b0n0.

- [x] FT9 is NOT the same bug -- checked, and the addresses rule it out:

        FT9         0xff00ffffffe00000  access 0 (Read)   top of space, tile field
                                                          varies 00/02/04 with the tile
        contrastive 0xff0000023a2ae000  access 1 (Write)  low in range, tile field
                                                          always 00

      FT9 is a READ at the very top of each tile's virtual address space, which is the
      scratch/private surface and exactly what it was originally diagnosed as. This one
      is a WRITE to a low address. Different access, different region, different
      structure. FT9 stays parked for ALCF; TILE_STRIDE does not touch it. The shared
      0xff00 prefix is just the region tag and is not evidence of a common cause --
      nearly closing FT9 on that similarity would have been wrong.
- [ ] The 50m pair-loss validation faulted while running single-tile with
      XPUS_PER_HOST=1, which this does not explain. Either a second cause or the tile
      assignment was not what the script intended. Check before trusting pair-loss runs.
