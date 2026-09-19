# Fine-tuning TODO

Open work on `dev_finetune`. Items are ordered by what would change a decision, not by
effort.

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

**Still to do**, because a 200M-parameter pretrain cannot fall back to one tile: bisect
the oneCCL environment. The launcher currently unsets `CCL_ZE_IPC`/`CCL_ZE_IPC_EXCHANGE`
and keeps `CCL_ATL_TRANSPORT=mpi`. A write to an unmapped page from the compute command
streamer is consistent with oneCCL holding a device pointer that the torch caching
allocator has since freed and remapped, which would make the suspects, in order:
`TORCH_XPU_ALLOC_CONF=expandable_segments`, CCL's own buffer cache, and the bucket rebuild
DDP performs after the first iteration. `aurora-pretrain.pbs` all-reduces 200M+ parameters
without this fault, so a diff against its exact environment is the cheapest first probe.

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
