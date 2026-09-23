# Defects and hazards

What is broken or dangerous, FT-numbered, each with a status. Research questions do not
live here -- they are in PLAN.md. Where things stand is STATUS.md.

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

## FT8. Is a warm-up freeze on the encoder worth anything? — **Research question, moved to PLAN.md (parked)**

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

## FT1. Does the denoiser generalise beyond 1024 peaks? — **Research question, moved to PLAN.md (parked)**

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

## FT6. Ablation: 50m trained from scratch — **DONE** (0.8856 at ep4, 0.9001 at ep8; PLAN.md D2)

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

## FT5. Multi-seed denoise on every scale — **DONE** (6 seeds per scale, job 8845262; PLAN.md D1)

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
      (`results/finetune/checkpoint_provenance.txt` has the pretraining step each started from).

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

## FT12. The sign of the pooled-vs-within AUROC gap is not predictable by reasoning — **Research question, moved to PLAN.md (parked)**

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

## FT17. The pair-loss grid samples both margins on the low side — **SUPERSEDED** (that grid was ranked on the separation ratio, which does not predict retrieval; nothing downstream uses the pair loss)

Embeddings reach `pair_contrastive_loss` L2-normalised (`MSDeltaForContrastive.embed`
ends in `F.normalize`), verified rather than assumed, so pair distance lives in [0, 2]
and relates to cosine by d^2 = 2 - 2cos. The grid sweeps `pair_margin` over {0.5, 1.0}:

    margin 0.5 .. asks different peptides for cosine <= 0.875, a weak ask
    margin 1.0 .. asks for cosine <= 0.5
    margin 2.0 .. would demand antipodal, certainly too strong

Both sampled values sit in the bottom half of the usable range. If 1.0 wins it will
have won at the edge of the grid, which says the optimum is at or beyond it rather
than located.

But "at the edge" does not imply "beyond the edge is better", and assuming it does is
a mistake this project has now made once. The 400m probe looked exactly like this --
encoder_lr_scale rising monotonically to its largest sampled value of 0.5 -- and the
inference that 1.0 would be better was wrong: 1.0 had already been run in the 400m HP
grid and scores 0.9396 against 0.5's 0.9436. An edge result means the optimum is
UNLOCATED, in either direction.

- [ ] If margin 1.0 beats 0.5, extend to {1.0, 1.25, 1.5} before reading anything into
      the margin.
- [ ] But check the wider grid first. The analogous worry about the 400m probe -- that
      encoder_lr_scale was monotone up to its largest sampled value and 1.0 was never
      tried -- was WRONG: the 400m HP grid had already tested es 1.0, and at fixed
      effective batch 12 it scores 0.9396 against es 0.5's 0.9436, worse by 0.0040 or
      eight times the seed noise. encoder_lr_scale is an inverted U peaking at 0.5; the
      probe sampled only its rising half. Extrapolating a trend past the last sampled
      point is what produced that error, so before extending any grid edge, look for the
      point already measured somewhere else in the project.
- [ ] Do not read `positive_fraction` as a loss-balance knob here: `pair_positive_weight`
      is left at its 1.0 default across all twelve arms, so the sweep varies how often a
      positive pair is DRAWN and lets the loss weighting follow. That is the intended
      question -- sampling balance -- but the two are not separated by this grid.

## FT13. The pretrained-vs-scratch denoise comparison has an unmeasured cell — **DONE** (400m ep8: 0.9418 / 0.9423 / 0.9399, job 8845252)

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

I guessed here that it would not gain much -- ep2 to ep4 bought the pretrained model
only +0.005 against the random encoder's +0.015 from ep4 to ep8, which is what
saturation looks like. TREAT THAT GUESS AS UNSAFE. The 400m probe (job 8845252) has a
pretrained encoder still gaining at 4 epochs, +0.0021 from ep2 to ep4 against a seed
noise of 0.0005, so "extra epochs do nothing once pretrained" is not a safe default at
any scale. The measurement costs one 6-arm job, which is why it is being made rather
than argued.

- [x] Run the pretrained 50m winner at 8 epochs with seeds, so it reads against FT5's
      spread rather than as a single number. SUBMITTED as 8846027: six seeds, arms
      differing from the running 4-epoch grid (8845262) by exactly --num_train_epochs.
      Validated on debug as 8846010.
- [ ] Report all three cells together; never quote +0.032 alone.
- [ ] Same question applies at 100m/200m/400m if a scratch ablation is ever run there.

## FT14. The PK sampler never reshuffles — every epoch replays identical batches — **FIXED**

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

- [x] **Fixed.** `__iter__` advances `self.epoch` itself after yielding its last batch,
      so reshuffling cannot be forgotten by a caller. `set_epoch` still works and
      overrides it. The RNG is also seeded as the PAIR `[seed, epoch]` rather than the
      SUM: under `seed + epoch`, seed 0 epoch 1 drew exactly the batches of seed 1
      epoch 0, so a seed sweep would have been a relabelling of one trajectory. Both
      failure modes now have tests that were confirmed to FAIL on the old code
      (`test_reshuffles_without_anyone_calling_set_epoch`,
      `test_more_epochs_reach_more_of_the_corpus`, `test_seed_and_epoch_do_not_collide`).
      `PairBatchSampler` was written self-advancing from the start and needed no change.

      MEASURED COST ON THE REAL CORPUS (898 train groups x ~13 replicates = 11,674 rows,
      at the grid's P=2 K=2 x 3 epochs):

        before ... 1,796 rows = 15.4%, and the SAME 1,796 all three epochs
        after .... 4,651 rows = 39.8%
        coverage by epoch, fixed: 15% 29% 40% 49% 57% 64% 69% 74% 78% 82% 85% 87%

- [ ] Re-run the epochs sweep: the "more epochs hurts" result (6.94 at 3, 4.82 at 10,
      4.46 at 50) is exactly what replaying one fixed batch sequence would produce, and
      may reverse now that epoch 10 sees 82% of the corpus instead of 15%.
- [ ] Re-measure the contrastive SCALE curve (5.86 / 7.00 / 7.02 / 6.74). Not wrong as a
      measurement, but taken in a 15%-of-data regime; the fix changes the training
      distribution enough that the curve has to be re-taken before it means anything
      about scale.

## FT15. Contrastive results are not reproducible run to run — **Open, not rechecked since 2026-09-22** (seed sems at the current recipe are 0.004-0.02 MAP@R, small against the effects now being measured)

Byte-identical config and seed, two runs, 7.83 (job 8842232) and 6.01 (job 8843838).

SOURCES RULED OUT by inspection:
  train/validation split .. seeded explicitly via `seed=training_args.seed`
  PK sampler .............. deterministic; was also fixed across epochs until FT14 was
                            repaired, so it was never the source of run-to-run spread
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

## FT16. Contrastive faults at 200m+ inside PBS jobs, never over ssh — **Unresolved, not seen since**. Some of the arm losses attributed here may have been FT19 (Lustre write storm), which also hit only 200m/400m arms

THE ONE SURVIVING PATTERN. Every failure has been inside a PBS sweep job; every pass has
been over ssh into a sleeper node holding an allocation. Both directions, no exceptions:

  FAILED, inside PBS .... x4310c1s1b0n0, x4216c0s3b0n0, x4605c5s0b0n0, x4720c6s5b0n0
  PASSED, over ssh ...... x4400c3s1b0n0, x4407c2s0b0n0, x4305c7s4b0n0

Four nodes fail and three pass, so it is not a bad node. Job 8845596 tests it directly
by submitting the exact script that passes over ssh as a PBS job.

WHAT HAS BEEN RULED OUT, each by measurement rather than argument:

| hypothesis | test | verdict |
| --- | --- | --- |
| model too big for the tile | peak 28.5 GB of 68.7 | refuted |
| two tiles share a card's HBM | same-card pair, both clean at 38.75 GB | refuted |
| concurrency / arms per node | 1, 2, 4, 8, 12 arms all clean at 38.73 GB | refuted |
| fused SDPA kernel | MSDELTA_SDPA_MATH=1, switch verified live | no effect |
| allocator fragmentation | expandable_segments unsupported on XPU | untestable |
| host RAM exhaustion | 1102 GB free of 1134 | refuted |
| the invocation (config, env, epochs) | gridexact: 38.75 GB, 300+ steps clean | refuted |
| FT9 is the same bug | different access type and address region | refuted |

THE NUMBER ANY EXPLANATION MUST ACCOUNT FOR. Every controlled run reserves 38.73-38.75
GB. The failing grid arms reserved 47 to 67 GB for the same model and config. Nothing
outside a PBS job has reproduced that inflation.

A TRAP IN THE EVIDENCE, worth recording. In job 8845057 the launcher placed all 50m and
100m arms on one node and all 200m and 400m arms on the other, so "large models fail"
and "that node fails" were perfectly confounded. The failures look scale-dependent and
are not: 12 concurrent 200m arms pass over ssh. Any future reading of that job has to
account for the placement.

LEADING CANDIDATE: the PBS job context itself -- cgroup limits, CPU binding, or
inherited environment differing from an ssh session into the same allocation. Plausible
rather than shown; 8845596 decides it.

- [ ] Read 8845596. If it faults, bisect the PBS job context (cgroup, --cpu-bind,
      inherited env) against the ssh path.
- [ ] If it passes, instrument a real failing sweep instead of reproducing beside one:
      dump the full environment and cgroup state from inside a faulting arm.

### REFUTED: two tiles share one card's HBM and both over-report it

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

WHEN IT IS NEEDED: never, for this. The paragraph that stood here said contrastive at
200m+ "reserves 67 GB and needs stride 2", and that was the refuted theory giving
operational advice. The 67 GB was not a property of 200m: it was the collator padding
each batch to its widest spectrum, so the figure moved with the seed (38.75 GB at seed
0, 67.14 GB at seed 3, one identical config). Under fixed-width padding the same arms
reserve 29.64 GB at every 200m seed and 37.28 GB at every 400m seed, and 12 per node at
stride 1 is clean -- job 8845623, 12/12 arms, 0 faults.

TILE_STRIDE remains a legitimate knob for giving one arm more of a card's bandwidth. It
is not a remedy for a fault, and a fault that appears now is something new rather than
a sibling-tile collision.

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


## FT18. A queued job reads its config dir at run time, not submit time — **Hazard, guarded**

Regenerating a grid overwrote `configs/sweep-ckpt-denoise` while 8850494 sat queued
against it; it would have run 10000/120000/430000/540423 instead of the validated
220000/330000, and completed normally. Restored from git. Guards: the checkpoint-ladder
generator puts the wave in the directory name, and the rule is to `qstat` for jobs on a
config path before writing to it. To change a queued job's grid: `qdel` first.

## FT19. Twelve arms per node writing optimizer checkpoints swamps Lustre — **FIXED**

At TILES_PER_ARM=1 a node carries twelve arms, and a 400m optimizer checkpoint is 7.4 GB,
so save_steps 200 put ~89 GB on Lustre at once. torch.save failed with
`enforce fail at inline_container.cc:668 ... unexpected pos` and killed 6 arms of
8853557, 6 of 8853558 and 13 of 8853703, all 200m/400m. Fix: save_only_model true,
save_steps 700, save_total_limit 1 in every contrastive generator. load_best_model_at_end
needs full checkpoints, so a grid that selects re-enables them -- keep such grids small.

## FT20. The session scratchpad is invisible to compute nodes — **Hazard**

`/tmp/claude-*/.../scratchpad` is node-local on the login node. A PBS job that calls a
script there fails at once with "No such file or directory" (8856114). Anything a job
runs lives in the repo.

## FT21. A contrastive checkpoint-N/ root loads as a RANDOM encoder — **Hazard**

MSDeltaForContrastive is a plain nn.Module, so checkpoint-N/model.safetensors holds the
wrapper's bare state dict. Loading the root silently gives an untrained encoder
(MAP@R 0.0029, ratio nan -- job 8856058). The loadable encoder is
checkpoint-N/encoder/, written by SaveEncoderCallback. final/ is fine. The proper fix is
still making the wrapper a PreTrainedModel.

## FT22. Monitor scripts leak orphaned processes on the login node — **Cleaned, hazard stands**

Background monitors of the form `tail -F log | ugrep | sed &` ending in
`pkill -P $$ tail` leave the tails orphaned when the parent exits early. 69 such
processes from 11-12 days earlier were killed one by one on 2026-09-23 (the pid inside
the SCREEN session was spared). Pollers now use qstat loops with no child processes.

## FT23. RESUME_JOB re-ran finished contrastive arms, and resumed them inexactly — **FIXED**

The resume skip test was `test_results.json`, which contrastive never writes, so every
finished contrastive arm counted as unfinished. Now `test_results.json` OR (`final/` AND
`retrieval_results.json`). Separately, resuming from a save_only_model checkpoint restarts
Adam and the LR schedule mid-run; grids driven by repeated resubmission (sweep-conbig)
therefore save no intermediate checkpoints, so an interrupted arm restarts cleanly from
step 0 with its seed.

## FT24. The sweep's closing tally misreported every RESUME_JOB round — **FIXED**

It counted run dirs named after the CURRENT job, but a resume writes into the original
job's dirs, so 8846565 printed "0/3 arms ok, 3 never started" and exited 1 while all
three arms had finished and written test results. That false failure is why the D2
resume sat unsubmitted. The tally now counts arms holding a completion marker
(test_results.json, or final/ + retrieval_results.json) in the dirs they wrote to.

## FT25. Resume picked a half-written checkpoint — **FIXED**

A job killed while saving leaves its newest checkpoint without trainer_state.json. The
resume path took the highest step unconditionally, so all five D2 arms of 8856525 died
in 30-66 s with FileNotFoundError on checkpoint-N/trainer_state.json. Every one of the
five had exactly this shape: newest checkpoint incomplete, the three before it whole.
The runner now walks down from the newest and takes the first checkpoint that has a
trainer_state.json, logging each one it skips; the cost is one save interval (200
steps). Retried as 8856558.

## FT26. The device test suite (tests/gpu) segfaults — **Open**

First recorded run of pbs/run_tests.pbs (job 8856984): the CPU suite passed on the
compute node (375 passed, 26 skipped, 3 min), then `pytest tests/gpu` died with a
Segmentation fault. No earlier log of the device suite exists, so this is not a
regression we can date -- the suite has simply never been seen to pass. A verbose rerun (8857034, `-v`,
`-X faulthandler`) segfaults before pytest prints a single line, so the crash is at
import/collection -- almost certainly native code loaded by tests/gpu or its conftest --
not in any test body. Until it passes, commits are gated on the CPU suite only.

## FT27. A job can hang in node startup, and a plain qdel does not remove it — **Hazard**

8856642 ran "R" for 2+ hours without writing a config snapshot, a run dir or a log, and
qstat showed no resources_used for it at all (every healthy running job has them). The
script never started on its nodes. `qdel` left it in R for another hour, holding one of
the two per-user capacity run slots -- which is why D2 and the D3 ends wave sat queued.
`qdel -W force` cleared it. Check: a running job with no resources_used and no
.configs-<job> snapshot after ~10 minutes is hung; resubmit (8857336) and force-delete.

## FT28. The rescorer derived the precursor from the answer — **FIXED, impact negligible**

run_rescoring.py computed each spectrum's precursor m/z from the TRUE peptide's
theoretical mass, because the alignment cache dropped the measured `precursor` column,
so the truth's `mass_error_ppm` was exactly 0. Fixed: the cache keeps the measured
precursor, the rescorer uses it and exits if it is missing, and it prints the truth's
mass-error distribution as a check (median 1.7 ppm now; p95 ~1,350 ppm is ordinary
isotope-peak error). BUT the leak turned out not to matter: every R1 number reproduced
within noise after the fix, because the decoys' mass errors are huge either way
(FT30). The earlier R1 numbers are NOT void on this account. (An earlier version of this
entry said they were; that was wrong.)

## FT29. The rescorer's train/test split was by spectrum, not peptide — **FIXED**

train_rescorer held out 20% of SPECTRA. Each peptide has ~13 replicate spectra, so the
audit (8859654) found 100% of rescorer-test spectra had their peptide in rescorer-train,
and peptide-level features (length, charge, GRAVY, modifications) let it memorise them.
It now holds out whole peptides (split_keys = each row's true peptide). Every R1 number
so far used the leaky split. Note the pool is small: 94 held-out peptides in total.

## FT30. "Mass-matched" decoys are not mass-matched — **Open, blocks R1**

run_rescoring takes the ~4 nearest peptides by sorted mass from the held-out pool (94
peptides) with no tolerance check, so they sit far outside any search window and mass
error rejects them for free (Hit@1 0.999 without near-misses). A realistic reranking
benchmark needs candidates within ~20 ppm of the measured precursor, which needs a far
larger peptide pool (MassIVE-KB has millions) or real search-engine candidate lists.

## Naming trap: our R@5 is not the literature's Recall@K

Ours is the fraction of a query's relevant spectra in its top 5; the literature's
Recall@K is the fraction of QUERIES with at least one hit in the top K. Documented in
`retrieval_metrics_exact`; prefer MAP@R, R-Precision and Precision@1.

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
- Jobs read configs from a snapshot taken at job START, so the working tree can be edited
  while a sweep RUNS -- but NOT while it is QUEUED. See FT18.

