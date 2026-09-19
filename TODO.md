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

## FT9. The alignment tower faults on twelve tiles under BOTH backends — **Open**

Denoise runs clean on twelve tiles with DeepSpeed. Alignment does not, under either
parallelism, so this is the model rather than the reducer and is NOT the same thing as
FT7:

| config | result |
| --- | --- |
| align, 1 tile, batch 4 | clean, 400 steps + eval + cross-modal + save (8840304) |
| align, 1 tile, batch 16 | GPU fault at step 73 (8840238) |
| align, 12 tiles, DDP | GPU fault at step 3 (8840223) |
| align, 12 tiles, DeepSpeed ZeRO-2 | GPU fault at step 56 (8840356) |
| denoise, 12 tiles, DeepSpeed ZeRO-2 | clean (8840264) |

8840356 had 4 rows per tile -- the same per-tile batch as the clean single-tile run -- and
peaked at 5.43 GB of 68.7, so it is not per-tile memory either. Three mechanisms have
been proposed for these faults today and two were wrong, so no fourth is offered here.

**Act on the structural difference instead.** Alignment runs a frozen 49.81M-parameter
teacher inside the training forward under `no_grad`; denoise has nothing of the kind. The
teacher is frozen, so its embeddings are IDENTICAL every epoch, and recomputing them each
step spends ~92% of the parameters and all of the 512-peak attention reproducing a
constant.

**The teacher is not the cause.** Job 8840403 precomputed the embeddings, set
`spectrum_model = None`, and faulted anyway at training step 0 with the 4.11M student as
the entire wrapped module. Precomputing was still worth doing -- it is an order of
magnitude cheaper and it is the right design for a frozen teacher -- but it is not the
fix, and the hypothesis it was based on is dead.

The faults are deterministic but not node-specific: three different nodes
(x4720c1s4b0n0, x4407c4s2b0n0, x4405c5s2b0n0), different ranks (5, 0, 8, 7), and a
different address on each node -- but the SAME node gives the same rank and the same
address every time. 8840378 and 8840403 are bit-identical failures. So it reproduces, and
it is not bad hardware.

What is left in the faulting module is small: three `nn.Embedding`, a `FourierFeatures`,
some `nn.Linear`, and an `nn.TransformerEncoder`. The denoise model shares every one of
those EXCEPT the `nn.TransformerEncoder` -- and torch's transformer is already responsible
for one confirmed XPU bug today (the fused kernel whose autocast guard reads CUDA state).
That is the obvious next suspect, though at step 0 in training mode grad is enabled and
the fused path should not be reachable, so it is a suspect and not an answer.

- [ ] Minimal reproducer: the student alone, twelve tiles, DeepSpeed, ten steps, nothing
      else in the process. It is now a 4.11M-parameter model with six module types, which
      is small enough to bisect by deletion.
- [ ] Probes were enabled on 8840403 and none fired, which places the fault outside the
      instrumented region -- most likely in DeepSpeed initialisation or the first forward,
      before `student.in` is reached. Push a probe earlier than that.

**This does not block anything.** One tile at batch 4 runs the full pipeline (8840304) in
about 35 minutes, which fits the debug queue, and twelve-tile alignment is a throughput
optimisation rather than a requirement.

Until then the working path is one tile at batch 4: ~35 min for a full 10-epoch run,
which fits the debug queue.

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

## FT5. Seed replication belongs on the winning config, not the baseline — **Open**

Seeds 1-4 are running against the pre-sweep baseline (lr 5e-5, encoder_lr_scale 0.1,
2 epochs, head 128), which was a guess rather than a tuned point. Those runs give a noise
floor, but it is the noise floor of a configuration we are about to replace.

Once the grid picks a winner, re-run the seed replication on THAT config. Two reasons it
is not optional:

  The grid ranks 72 arms on one seed each. If seed spread is comparable to the spread
  between neighbouring arms, the ranking is largely noise and the "winner" is whichever
  arm drew a good seed. The replication is what licenses calling it a winner at all.

  Seed sensitivity is not constant across the space. A high learning rate or an unfrozen
  encoder can be stable at one seed and divergent at another, so the baseline's spread
  does not transfer to a more aggressive winning config.

- [ ] After the grid, run >=5 seeds on the winning arm and report mean +/- sd.
- [ ] Compare that spread against the gap between the top few arms. If they overlap, say
      so plainly and treat the top group as tied rather than ranked.
- [ ] Keep the existing baseline seed runs as the comparison point, so the two spreads
      can be read against each other.

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
