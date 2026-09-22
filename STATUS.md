# Project status

The schedule for the denoise fine-tune and the reranking pipeline. Hand-kept, so it says
what is *true right now*; `TODO.md` says what is *wrong and open*, and the two are meant
to be read together.

Regenerate the job table with `pbs/job_history.sh`, which reads the logs rather than
anyone's memory. Last updated: 2026-09-20.

## Where things stand

```
LEGEND  [x] done  [~] RUNNING  [>] blocked  [X] closed  [ ] not started


JOBS
──────────────────────────────────────────────────────────────────────────
  [~] 400m denoise grid .. 8842147  capacity  just started, ~2.5h
      everything else in the queue has finished


DENOISE  — the line that works
──────────────────────────────────────────────────────────────────────────
  [x] 50m,  216 arms ..... 0.9320 auroc / 0.8632 f1
  [x] 100m,  12 arms ..... 0.9403 / 0.8723      (+0.0083)
  [x] 200m,  12 arms ..... 0.9446 / 0.8778      (+0.0043)
  [x] 400m,  12 arms ..... 0.9436 / 0.8768      (-0.0010)

        The prediction that stood here -- "+0.002, inside the ~0.001 within-grid
        spread" -- was wrong in both parts. 400m came in BELOW 200m, and the
        within-grid spread is not the error bar: it mixes real hyperparameter
        effects with noise, so it overstates noise and dismisses real effects.
        Measured seed noise at a fixed configuration is sd ~0.0005 (FT5, six
        seeds per scale, job 8845262):

              50m   0.9317 +/- 0.00055        200m  0.9447 +/- 0.00025
              100m  0.9400 +/- 0.00029        400m  0.9434 +/- 0.00045

        All 24 arms in. Every step resolves: +0.0083 (t=32.7), +0.0047
        (t=30.2), -0.0013 (t=-6.3). See results/finetune/denoise/denoise_scale_seeds.txt.

        200m -> 400m is -0.0013 at t = -6.3, a regression and not a plateau.
        AT THIS FINE-TUNING BUDGET: the 400m probe (8845252) has 400m still
        gaining at 4 epochs, +0.0021 from ep2, so 400m has not converged where
        it was scored.
  [x] scratch 50m, random encoder
        matched 4 epochs ....... 0.8856  -> pretraining worth +0.046
        scratch at 8 epochs .... 0.9001  -> pretraining worth +0.032
        never quote +0.032 alone; see FT13 for the missing cell
  [>] FT5 multi-seed, all scales .. blocked on 400m. Now a PRECONDITION,
        not a refinement: without seeds there is no basis for ranking 400m.
  [>] final model ................. after FT5


EMBEDDING  — one finding explains the whole axis
──────────────────────────────────────────────────────────────────────────
  [x] THE FINDING: pretrained structure is REAL but NON-LINEAR
        frozen              pretrained 1.35 == random 1.35
        after contrastive   pretrained 7.83 >> random 1.35
        No readout reaches it. Fine-tuning unlocks it. Both needed.

  [x] complete readout x encoder x init factorial -- every random cell
        is exactly 1.35; on the pretrained side only "final layer +
        encoder trains" works, every elaboration is worse
  [x] pretraining is worth >10x the fine-tuning budget
        random: 1.35 @1.3k steps -> 2.10 @4.5k -> 2.73 @13.5k
        pretrained: 7.83 @1.3k
  [X] layer / pooling / scale / longer pretraining .. noise on the floor
  [X] trained depth mixture ........................ 1.49 vs 1.53 by hand
  [X] genuine blend vs one layer ................... blend is WORSE
  [X] embedding -> reranker ........................ -0.109 hit@1
  [ ] contrastive from the 10k checkpoint .......... THE LAST LEAD
        1.67 frozen, the only reading that ever beat the floor.
        3 arms, one node, ~5 min. Needs checkpoint-10000 frozen first.


RERANKING
──────────────────────────────────────────────────────────────────────────
  [x] feature rescorer ... hit@1 0.889, no neural embedding
  [ ] cross-encoder ...... unblocked now: embeddings reach 7.83


INFRA & BUGS
──────────────────────────────────────────────────────────────────────────
  [x] per-spectrum AUROC .. pooled was NOT hiding a within-spectrum failure
        50m 0.9213 -> 0.9269, 100m 0.9331 -> 0.9377, p10 > 0.85
  [x] checkpoints frozen by step; validation gate; unknown-flag tests;
      SaveEncoderCallback; staleness guard on 10 grids
  [X] FT10 RETRACTED ... warmup was never the bug, I read a smoke log as real
  [X] FT11 closed ...... blend worked and was worse, nothing to protect
  [ ] FT12 ............. pooled-vs-within sign unpredictable; check applied
  [ ] FT13 ............. pretrained @ 8 epochs never run; fold into FT5
  [ ] MSDeltaForContrastive -> PreTrainedModel .. worth revisiting, the
      contrastive line survived
  [~] FT9 12-tile align fault .. parked for ALCF
  [ ] FT1 / FT3 / FT4 / FT8 .... open, none blocking
```

Model scales are NOT compared against each other directly; figures normalise by
compute budget. That is why the matched-checkpoint control was dropped -- and why
`results/finetune/checkpoint_provenance.txt` matters: `final/` is a moving export, so the
step each fine-tune started from is only recoverable while those checkpoints still
exist. It is recorded there rather than left to be reconstructed later.

## Reranking: the causal chain, validated end to end

The alignment tower was reaching only 2.3x chance at hit@1. The separation eval said why
with a measurement rather than a theory: the frozen pretrained encoder does not separate
peptides. Replicate spectra of one peptide sat at cosine 0.963 and spectra of DIFFERENT
peptides at 0.950 -- an out/in distance ratio of 1.34, essentially no structure. Nothing
in a masked-peak objective ever asked for peptide identity, so the student was imitating
that space faithfully and there was nothing there to imitate.

Fixing the TEACHER fixed retrieval, with no change to the student at all:

| | pretrained teacher | contrastive teacher |
| --- | --- | --- |
| teacher out/in ratio | 1.34 | **7.49** |
| cross-modal hit@1 | 0.0249 (2.3x chance) | **0.0446 (4.2x)** |
| hit@5 | 0.1157 | **0.2288** |
| MRR | 0.0991 | **0.1594** |

`eval_loss` went UP, 0.055 to 0.243, which is the right direction: the old target space
was easy to fit precisely because it was nearly collapsed. The loss was never the metric.

Not yet a usable reranker -- 4.2x chance over 94 candidates is not a reranker -- but the
mechanism is established and the levers are known:

1. **GradCache.** The contrastive encoder trains at batch 4, which is 4 negatives per
   step; a contrastive objective is largely a function of how many negatives it sees.
   Decoupling that from memory is the single biggest lever and the ratio is what drives
   the downstream number.
2. **The tail.** `clean` is still 0.010: some replicate pairs of one peptide stay far
   apart (`worst_in` 0.24 -> 0.84) even as the averages separate. Worth looking at
   whether those spectra genuinely resemble each other before trying to force them
   together.
3. **The student.** Worth tuning now, since the teacher no longer bottlenecks it.

Two findings from the sweep worth keeping. `kl_weight=10` beats `kl_weight=0` on the full
corpus while all six `kl=0` arms won at 740 steps -- the regulariser earns its place only
once there is enough training to overfit, and a smoke test would have locked in the wrong
answer. And rank arms by RATIO, not margin: margin is a difference and rises when a model
merely inflates the space, which one arm did.

## Denoise: the grid is finished and 100m beats 50m

216 arms at 50m, then 12 at 100m narrowed by what the first grid measured.

| | best test AUROC |
| --- | --- |
| free raw-intensity baseline | ~0.75 |
| first 700-step run | 0.8628 |
| 50m grid, 212 arms | 0.9320 (`lr2e4_es05_ep4_h512_b12`) |
| **100m grid, 12 arms** | **0.9403** (`lr2e4_es05_b12`) |

What the 50m grid established, by mean AUROC across arms:

| axis | effect |
| --- | --- |
| `encoder_lr_scale` | 0.760 frozen -> 0.872 at full rate. The largest factor by far |
| `learning_rate` | 0.776 -> 0.885 at 2e-4, monotone |
| batch | b144 0.813, b48 0.831, b12 0.849 |
| `num_train_epochs` | 0.824 -> 0.838 |
| `head_hidden_size` | 0.828 / 0.832 / 0.834 -- nothing |

**Fine-tuning the encoder is the whole game.** Frozen arms average 0.760, barely above the
free baseline, so what the pretrained model contributes comes from ADAPTING it rather
than from its features as they stand -- the same conclusion the reranking work reached
from the opposite direction.

Three things the 100m grid settled that the 50m grid could not:

- **2e-4 is a real interior optimum.** It was the best value AND the boundary at 50m, so
  the 100m grid extended to 5e-4, which is worse (0.9332 against 0.9392). Extending was
  the point of including it.
- **Keeping 5e-5 was sound reasoning and a wrong hypothesis.** Larger models often prefer
  lower rates; this one does not.
- **The b12 advantage is 50m-specific.** It beat b48 by 0.018 at 50m and loses to it by
  0.001 at 100m, so "more optimizer steps win" was about to become folklore on one data
  point.

The whole 100m grid spans 0.019 against the 50m grid's 0.26. Once the encoder is unfrozen
at a sane rate, the larger model barely cares about these hyperparameters, which is the
strongest argument that narrowing the grid was right.

## Pretraining IS essential — but nothing in the frozen embedding shows it

Job 8842288 ran the contrastive grid arm for arm on a RANDOMLY INITIALISED encoder.
All twelve arms land on the floor:

| arm | pretrained | random | delta |
| --- | --- | --- | --- |
| lr5e4_kl10_t007 | **7.83** | 1.35 | -6.48 |
| lr2e5_kl0_t02 | 7.71 | 1.35 | -6.36 |
| lr1e4_kl10_t02 | 7.14 | 1.36 | -5.78 |
| ... | ... | ... | ... |
| lr5e4_kl0_t007 | 1.35 | 1.35 | 0.00 |

Random spans 1.34 to 1.36 across every learning rate, temperature and KL weight. That is
the random-init floor to two decimal places: **the contrastive loss achieves literally
nothing on a random encoder**, while the same loss on the same data with the same
hyperparameters takes a pretrained encoder to 7.83.

THIS OVERTURNS THE OBVIOUS READING OF THE FLOOR RESULT. Job 8842086 found the frozen
pretrained embedding indistinguishable from random -- 1.35 either way, every layer,
every scale -- and the natural inference was that pretraining contributes nothing to
peptide identity. That inference is now dead. Both facts are true at once:

  frozen:            pretrained 1.35  ==  random 1.35
  after contrastive: pretrained 7.83  >>  random 1.35

So pretraining builds something that does NOT appear in the frozen embedding under any
readout we can construct, and that is nevertheless REQUIRED for the contrastive
objective to move at all. The pretrained weights are a learnable initialisation for this
task; random weights are not. "Nothing is there" was wrong; the right statement is
"nothing is there THAT A FROZEN READOUT CAN REACH".

It also retires the last worry about the readout work. Layer choice, pooling and mixture
were all measuring a quantity that does not predict fine-tuned performance, which is why
none of them moved it -- and why the 1.52 trained-mixture result closes that axis without
saying anything bad about the encoder.

THE HONEST CAVEAT. These runs are ~90 optimizer steps at batch 4. A randomly initialised
50m encoder has no realistic chance of learning a metric space in 90 steps, so this shows
pretraining is essential AT THE BUDGET WE USE, not that a random encoder could never get
there. Testing that properly means giving the random arm far more steps, which is cheap
here -- the whole grid is five minutes -- and is the obvious follow-up if the claim needs
to be stronger than "at equal budget".

## A trained readout over depth buys nothing; the corrected schedule buys 13%

Two jobs, both under the fixed warmup (FT10).

**Layer-mix (8842231).** One trainable softmaxed scalar per encoder depth, mean-pooled
to d_model, at three encoder learning rates. The frozen arm is the one that answers the
question, because only the readout moves in it:

| arm | encoder lr | ratio |
| --- | --- | --- |
| frozen | 0 | **1.52** |
| els03 | 0.3x | 3.46 |
| els10 | 1.0x | 4.28 |

Against a random-init floor of 1.35 and a best-hand-picked-layer of 1.53, a readout
TRAINED over all eleven depths reaches 1.52. It does not beat picking the best layer by
hand, and neither clears the floor by much. **Depth selection is not a lever**, learned
or otherwise, which is the last readout idea and closes that axis for good.

Two honest caveats. The mixture COLLAPSED -- `mix/max` 1.000 on layer09, entropy 3e-05
against a uniform 2.398, within the first ~15% of the run. `layer_mix_lr 5e-2` was too
high; my sizing treated Adam's displacement as linear in the learning rate and ignored
that softmax saturation is self-reinforcing. So this tested learned layer SELECTION, not
blending. And els10's 4.28 sits below the plain mean+max contrastive result at a
comparable setting (6.21), so the mixture is worse than the fixed readout when the
encoder trains -- though it is also d_model against mean+max's 2*d_model, so that
comparison is not clean either. Neither caveat touches the frozen arm, which is the
result that matters.

**Contrastive re-run (8842232).** The same 12-arm grid with warmup no longer exceeding
the whole run: best ratio **7.83**, against **6.94** under the broken schedule. The
6.94 that every earlier conclusion was measured against is superseded.

More interesting than the 13%: the spread is now 1.35 to 7.83, where the old grid's arms
sat much closer together. FT10 predicted exactly this -- a warmup longer than the run
compresses every learning rate into the same ramp, so the old sweep could not separate
them. It can now, and lr5e4 with no KL collapses to 1.35, the floor, while lr5e4 with
KL 10 is the best arm at 7.83. The KL term is doing real work at high learning rates.

## The frozen embedding never carried peptide identity: random scores the same

Job 8842086 ran the control that should have come first. A RANDOMLY INITIALISED encoder
-- same architecture, no pretrained weights -- on the same 1167 spectra and 99 groups:

| encoder | separation ratio |
| --- | --- |
| 50m random init | **1.35** |
| 100m random init | **1.35** |
| 200m random init | **1.35** |
| 50m pretrained, its own frontier | 1.35 |
| 100m pretrained, its own frontier | 1.36 |
| 200m pretrained, its own frontier | 1.35 |

**The pretrained encoders are indistinguishable from untrained ones.** Every layer of the
random models reads 1.35 as well -- there is no depth structure to find because there is
nothing there. Replicate spectra of one peptide have similar peaks, so ANY function that
does not actively destroy that similarity scores about 1.35 on this metric. That is the
floor, and essentially every frozen number this project has reported sits on it.

This retro-explains the whole line at once. Pooling mode moved 1.29 to 1.44; layer choice
moved 1.35 to 1.53; scale moved nothing; the teacher space sat at 1.34. All of those are
noise around a random-network floor, which is why none of them ever reached the reranker
and why the embedding measured as actively harmful (-0.109 hit@1). We were tuning the
readout of a representation that did not contain the signal.

Two things survive, and they are the only two:

**Contrastive training genuinely creates the structure.** 6.94 against a 1.35 floor is
not a readout effect. Training is the whole of the difference, and the earlier "+418%
against +14%" understated it, because the +14% was measurement noise on a floor.

**Pretraining briefly creates this structure and then destroys it.** The 50m at 10,000
steps reads 1.67, which is 0.32 ABOVE the random floor and the only frozen measurement
in this project that clearly clears it. By 50,000 steps it is back to 1.33. The 100m and
200m show the same sign at 10,000 (1.42, 1.43) and also decay to the floor. So the
masked-peak objective passes through a phase where the representation carries replicate
identity, and then trains it away -- it is not that the information was never learned.

WHAT THIS CHANGES. "Read a better layer", "pool differently", "use a bigger encoder" and
"pretrain longer" are all closed: there is nothing to read. The open questions are
whether the 10,000-step checkpoint is a better CONTRASTIVE starting point than the
frontier one, and whether a learned mixture can find anything the fixed readouts could
not -- which is what configs/sweep-layermix measures, now that its mixture can actually
move.

CALIBRATION, for anything reported later: on this data and metric the floor is 1.35, not
1.0. A ratio must be read against 1.35.

## The embedding direction has exhausted its identified levers

Every lever tried, and what it did to the separation ratio (pretrained = 1.34):

| lever | result |
| --- | --- |
| contrastive + KL, batch 4 | 1.34 -> **6.94**, the best figure reached |
| more steps at batch 4 | 6.94 -> 4.82 -> 4.46 across 3, 10 and 50 epochs |
| more negatives (GradCache, batch 64) | best 5.70, BELOW the batch-4 best |
| more steps at batch 64, to 17,893 | 5.70 -> 5.35 -> 4.37 -> 4.34 -> **3.35**, monotone |
| intensity-weighted pooling, frozen | 1.43 -> 1.44, nothing |
| denoiser P(signal) pooling, frozen | 1.34 -> **1.46**, the best readout available, and still nothing next to 6.94 |
| reading layer 8 instead of the output | 1.43 -> **1.53**; the stack peaks at 8 and DROPS at 9 |
| scaling the encoder 50m -> 100m -> 200m, frozen | NO EFFECT. At matched frontiers 1.35 / 1.36 / 1.35. The earlier "scale hurts" read a stale `final/` export |
| `mean` instead of `mean+max`, frozen | 1.34 -> 1.43, free but small |
| contrastive teacher -> alignment | cross-modal hit@1 +79% |
| that embedding -> reranker | **-0.109 hit@1**, five paired seeds |

Two of these deserve care, because the obvious reading of each is wrong.

**Longer training degrading the ratio is not overfitting.** At batch 4 the contrastive
loss reached 0.4% of chance, so "the task is solved and further steps overfit it" was a
natural explanation, and it is the one recorded earlier. GradCache disproves it: at batch
64 the loss is at 30.6% of chance and still falling -- nowhere near saturated -- and the
ratio falls monotonically anyway. Something about optimising in-batch discrimination
rearranges the space in a way the global separation metric dislikes, independent of
whether the training task has been learned.

**A feature can be strong alone and harmful in a model.** `embedding_cosine` separates
true from decoy pairs at AUROC 0.846, yet adding it costs eleven points of reranking
hit@1 while leaving pooled AUROC untouched. AUROC pools all pairs; hit@1 ranks within a
spectrum; the two diverge when a feature's errors are correlated within a spectrum, and
this one is structurally so -- every candidate for a spectrum is scored against the SAME
cached teacher vector.

**What works instead.** A hand-built feature rescorer reaches hit@1 0.889 on fragment
coverage, mass error and spectrum quality, with no neural embedding at all.

**The readout axis, measured exhaustively: 39 configurations, 3 scales.** Every
encoder block of the 50m, 100m and 200m pretrained checkpoints, plus four pooling modes
on each output, all frozen and training-free. The probe validates itself: at every scale
the last block's ratio equals the `mean`-pooled output ratio, as it must.

| scale | blocks | output `mean` | best block | best ratio | depth of best |
| --- | --- | --- | --- | --- | --- |
| 50m  | 10 | 1.43 | 8  | **1.53** | 89% |
| 100m | 13 | 1.36 | 11 | 1.41 | 92% |
| 200m | 16 | 1.35 | 6  | 1.44 | 40% |

Two things fall out, and the second is the one that matters.

**The mid-stack hump is real and universal.** No scale peaks at its output. The ratio
climbs through the stack and then drops in the last one to three blocks -- 50m
1.37/1.38/1.38/1.41/1.45/1.51/1.51/1.52/**1.53**/1.43, 200m rising to **1.44** at block 6
and sagging to 1.35 by block 15. That is the signature of a last layer specialised for
its pretraining head: predicting masked peak intensities is not the same objective as
representing peptide identity, and the final blocks have been optimised for the former.
Reading one block earlier is free.

**REFUTED, with the matched-step measurement: scale has no effect here, either way.**
The original claim -- output ratio 1.43 / 1.36 / 1.35 falling with capacity -- came from
reading each scale's published `final/`. Those are not the ends of training. Byte-
comparing `final/model.safetensors` against every checkpoint in each run:

| scale | `final/` is really | that run's last checkpoint | so it was read at |
| --- | --- | --- | --- |
| 50m  | **checkpoint-133233** | 180000 | 74% of its own frontier |
| 100m | **checkpoint-138073** | 190000 | 73% |
| 200m | checkpoint-192799 | 192799 | 100% |

`final/` is a periodically-refreshed export of a run still in progress, lagging its own
last checkpoint by 21 hours at 50m and 6 at 100m. The sweep compared two models at ~135k
steps against one at ~193k and read the difference as capacity.

Job 8842038 then probed all three along their own trajectories. Compared at matched
steps the effect disappears:

| pretrain step | 50m | 100m | 200m |
| --- | --- | --- | --- |
| 10,000 | **1.67** | 1.42 | 1.43 |
| 50,000 | 1.33 | 1.34 | 1.42 |
| 100,000 | 1.38 | 1.30 | 1.28 |
| 150,000 | 1.39 | 1.34 | 1.38 |
| each model's own frontier | **1.35** | **1.36** | **1.35** |

At their frontiers the three scales are identical to two decimal places. At the
intermediate steps the ordering changes from row to row -- 50m highest at 100k, 200m
highest at 50k, mixed at 150k -- which is what no effect looks like. Excluding the
10,000-step point, all fifteen measurements lie in [1.28, 1.42], a spread of 0.14, and
every between-model difference is smaller than that.

So: quadrupling the encoder neither helps nor hurts the frozen embedding. The earlier
"scale hurts" was an artefact of a stale export and is withdrawn. Note this is a
different finding from the denoise result on the same checkpoints, where 100m genuinely
does beat 50m (0.9403 vs 0.9320) -- fine-tuning uses the capacity, a frozen readout
does not.

It also answers the undertraining question directly: none of the three ratios is
climbing at its frontier. All three plateau from about 50,000 steps onward, at roughly a
third of the pretraining schedule. More pretraining will not supply this.

**The one real feature is the 50m at 10,000 steps**, 1.67 against ~1.36 everywhere else,
with its best block at 1.69 -- higher than any of the 39 configurations in the original
sweep. It is 50m-only: the 100m and 200m sit at 1.42 and 1.43 at the same step. Before
reading it as "early pretraining carries peptide identity and later training erodes it",
it needs its zero, because replicate spectra of one peptide have similar peaks and an
encoder that has learned nothing and merely passes its input through would separate them
too. `pooling_probe.py --random_init` supplies that, running as job 8842086. If a random
50m also sits near 1.7, the spike means pretraining has not yet destroyed input
similarity -- which points somewhere completely different.

Raw numbers in `results/layer_probe_trajectory.txt`; regenerate with
`sweeps/summarise_trajectory.py <log>`.

`clean` -- the fraction of groups whose every replicate is nearer to its own group than
to anything outside it -- sits at 0.010 for essentially all 39 configurations, meaning 1
of 99 groups, and at 0.000 for the earliest blocks. No layer of no model at no scale
separates replicates.

So the whole readout axis -- layer choice, pooling mode, peak weighting, and now model
scale -- spans 1.29 to 1.53. Contrastive training reaches 6.94. Every extraction trick
available, across three model sizes, is about 3% of what training buys, which is the
clearest available statement of where the information is not.

**What has not been tried.** The correlated-error structure is a property of the
two-tower formulation, not of embedding quality, so no amount of better embedding fixes
it. A cross-encoder scoring (spectrum, candidate) jointly has no shared per-spectrum
vector and is the natural next formulation -- reranking only scores the top-k candidates,
so it never needed a shared retrieval space in the first place.

## The embedding does not help the reranker, and hurts it

Measured, five paired seeds, both arms sharing each seed's split and initialisation:

| | with embedding | without | contribution |
| --- | --- | --- | --- |
| pairwise AUROC | 0.9061 | 0.9055 | **+0.0006 +- 0.0010** |
| hit@1 | 0.7803 | **0.8893** | **-0.1090 +- 0.0271** |

A hand-built feature rescorer reaches hit@1 0.889 on its own. Adding `embedding_cosine`
leaves pooled AUROC untouched and costs eleven points of hit@1, in every seed.

`embedding_cosine` is not weak on its own -- 0.846 AUROC separating true from decoy pairs.
But AUROC pools all 5,541 pairs while hit@1 ranks WITHIN each spectrum, and the two come
apart when a feature's errors are correlated within a spectrum. That is exactly this
feature's shape: every candidate for one spectrum is scored against the SAME cached
teacher vector, so when that vector is poor it misleads all of that spectrum's candidates
together. Fragment features are computed against the observed peaks per candidate, and
their errors do not line up that way.

So the embedding injects spectrum-level noise into precisely the comparison reranking
depends on. Whether the alignment tower can be rebuilt to avoid that is open; what is
settled is that its current output should not be a reranker feature.

**What this costs.** The contrastive work raised the separation ratio 1.34 -> 7.49 and
cross-modal hit@1 by 79%, and none of it reaches the reranker. The feature classifier
was already in the repo on `sweep/pairformer-aurora`, with `separation.py` stating the
prerequisite question and `negatives.py` the hard-negative problem, before any of today's
embedding work started.

## The one real number so far

`test AUROC 0.8628` from a 700-step denoise run (8840190), against a free raw-intensity
baseline of about 0.75. Everything else this session was infrastructure. **No tuned model
and no reranking result exists yet.**

## What has to happen, in order

1. **8840323 passes** -> submit the 216-arm grid to capacity. ~84 node-hours, ~5 h on 16
   nodes. Axes: `learning_rate` x `encoder_lr_scale` x `num_train_epochs` x
   `head_hidden_size` x effective batch.
2. **Full alignment training, one tile.** Twelve tiles faults under both backends
   (FT9), so one tile at batch 4 is the only proven path: ~35 min, which fits the debug
   hour if `eval_steps` goes from 200 to 2000. At the current 200 the 143 evals would eat
   28 of those minutes. The real fix is to precompute the frozen teacher's embeddings,
   which is cheaper anyway -- see FT9.
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

Regenerate with `pbs/job_history.sh`. Full probe results in `results/`.

```
JOB       TASK     PARALLELISM OUTCOME                    NOTE
8816450   denoise  8 tiles     ran to 40/40               
8821044   denoise  ?           no training                
8821246   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821247   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821248   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821249   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821250   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821251   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821272   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821273   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821275   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821276   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821277   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821278   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821285   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821292   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821293   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821295   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821296   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821297   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821309   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821311   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821312   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821314   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821315   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821316   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821326   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821327   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821328   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821329   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821330   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821331   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821345   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821347   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821349   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821351   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821352   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821353   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821363   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821364   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821365   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821366   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821368   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821369   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821376   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821377   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821378   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821380   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821381   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821382   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821403   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821404   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821405   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821406   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821407   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821408   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821428   denoise  ?           no training                
8821434   denoise  ?           no training                
8821438   denoise  ?           no training                
8821439   denoise  ?           no training                
8821469   denoise  ?           no training                
8821504   denoise  ?           no training                
8821931   denoise  ?           no training                
8822208   denoise  ?           no training                
8822209   denoise  ?           no training                
8838696   denoise  ?           no training                
8838813   denoise  ?           no training                
8838858   denoise  ?           no training                
8838985   denoise  ?           no training                
8839072   denoise  ?           no training                
8839122   denoise  ?           no training                
8839139   denoise  8 tiles     no training                
8839150   denoise  12 tiles    ERROR: RuntimeError: The device index is out of 
8839166   denoise  12 tiles    GPU FAULT at 19/60         
8839203   denoise  12 tiles    GPU FAULT at 2/103         
8839579   denoise  12 tiles    ERROR: ValueError: multiclass format is not sup 
8839683   denoise  12 tiles    COMPLETE 1/1               'test_auroc': 0.9354260932338447
8839842   denoise  12 tiles    COMPLETE 1/1               'test_auroc': 0.9354279541485844
8839881   denoise  12 tiles    COMPLETE 1/1               'test_auroc': 0.9401584363884974
8839890   denoise  12 tiles    COMPLETE 1/1               'test_auroc': 0.9355479921240085
8839920   denoise  ?           no training                
8839937   grid                 4/4 arms ok                
8839946   denoise  12 tiles    ERROR: RuntimeError: Expected to have finished  
8839957   grid                 4/4 arms ok                
8839997   grid     1 tile(s) each 4/4 arms ok                
8840007   denoise  12 tiles    GPU FAULT at 107/2018      
8840008   denoise  12 tiles    GPU FAULT at 26/2018       
8840010   denoise  12 tiles    GPU FAULT at 107/2018      
8840011   denoise  12 tiles    GPU FAULT at 29/2018       
8840057   denoise  12 tiles    no training                
8840058   denoise  12 tiles    no training                
8840095   denoise  12 tiles    GPU FAULT at 29/60         
8840120   denoise  12 tiles    GPU FAULT at 29/600        
8840154   denoise  12 tiles    GPU FAULT at 168/700       
8840190   denoise  1 tiles     COMPLETE 1/1               'test_auroc': 0.8628252912383914
8840223   align    12 tiles    GPU FAULT at 3/200         
8840232   grid     1 tile(s) each 72/72 arms ok              
8840238   align    1 tiles     GPU FAULT at 73/200        
8840257   align    1 tiles     ERROR: RuntimeError: expected scalar type BFloa 
8840264   denoise  12 tiles DS COMPLETE 1/1               'test_auroc': 0.7296013323843181
8840277   align    1 tiles     ERROR: KeyError: 'eval_loss' 
8840291   denoise  ?           refused: stale arms        
8840304   align    1 tiles     COMPLETE 400/400           'eval_loss': '0.04813'
8840313   denoise  ?           refused: stale arms        
8840323   denoise  ?           refused: stale arms        
8840336   align    12 tiles DS ERROR: RuntimeError: expected scalar type Float 
8840345   grid     12 tile(s) each 216/216 arms ok            'test_auroc': 0.5783115944609307
8840356   align    12 tiles DS GPU FAULT at 56/300        'eval_loss': '0.6521'
8840378   align    12 tiles    GPU FAULT at 1167/1167     
8840403   align    12 tiles    GPU FAULT at 0/300         
8840408   grid     12 tile(s) each 212/216 arms ok            'test_auroc': 0.8015630560943791
8840431   denoise  ?           no training                
8840444   denoise  ?           no training                
8840469   align    12 tiles    GPU FAULT at 120/300       'eval_loss': '0.05669'
8840487   denoise  ?           no training                
8840498   align    12 tiles    GPU FAULT at 22/2390       
8840516   align    12 tiles    GPU FAULT at 206/2390      
8840529   align    1 tiles     COMPLETE 28630/28630       'eval_loss': '0.05461'
8840537   align    12 tiles    GPU FAULT at 26/2390       
8840581   denoise  1 tiles     ERROR:                     
8840587   denoise  ?           no training                
8840597   denoise  1 tiles     ERROR: AttributeError: 'MSDeltaForContrastive'  
8840603   align    12 tiles    GPU FAULT at 802/2390      
8840611   denoise  1 tiles     ERROR:                     
8840621   denoise  1 tiles     ERROR:                     
8840632   denoise  1 tiles     ERROR:                     
8840651   denoise  1 tiles     COMPLETE 1/1               
8840656   denoise  1 tiles     COMPLETE 1/1               
8840665   denoise  ?           12/12 arms ok              
8840693   denoise  ?           0/12 arms ok               
8840715   denoise  ?           12/12 arms ok              
8840739   align    1 tiles     COMPLETE 28630/28630       'eval_loss': '0.2432'
8840792   denoise  ?           4/4 arms ok                
8840898   denoise  ?           no training                
8840907   denoise  ?           no training                
8840927   denoise  ?           no training                
8840941   denoise  ?           no training                
8841066   denoise  ?           3/3 arms ok                
8841165   denoise  ?           2/2 arms ok                
8841255   denoise  ?           ERROR: TypeError: MSDeltaModel.forward() missin 
8841270   denoise  ?           ran to 170/170             
8841285   denoise  ?           ran to 172/172             
8841310   denoise  ?           ran to 170/170             
8841345   denoise  ?           12/12 arms ok              'test_auroc': 0.9360134693643527
8841966   denoise  12 tiles    COMPLETE 1/1               'test_auroc': 0.8429291281565523
8841973   denoise  ?           ran to 266/266
```

## The 50m grid is complete at 216/216

The four arms that finished training but never wrote a test number (the
`load_best_model_at_end` failure, FT3's neighbour) were recovered by loading their final
checkpoint and re-running the test pass -- `pbs/recover_test_metrics.pbs`, job 8842037,
minutes rather than the ~79 node-hours a re-run would have cost.

| arm | val auroc | test, FINAL weights | rank by val |
| --- | --- | --- | --- |
| lr1e5_es0_ep4_h512_b12 | 0.7798 | 0.7882 | 165 |
| lr1e6_es01_ep4_h256_b12 | 0.8024 | 0.7995 | 129 |
| lr2e4_es0_ep2_h128_b12 | 0.7899 | 0.7964 | 146 |
| lr2e4_es10_ep4_h128_b12 | 0.9292 | 0.9230 | **8** |

These carry `weights_selected_by=final` in their `test_results.json`; every other arm
carries its best-validation weights. The difference is visible in the fourth row, which
is the only one that mattered: at rank 8 by validation it was the one arm that could
conceivably have displaced the winner, and at 0.9230 it does not -- the winner stays
`lr2e4_es05_ep4_h512_b12` at 0.9320. Its 0.9292 validation against 0.9230 final-weights
test also puts a number on what best-validation selection is worth here: about 0.006.

Checked BEFORE running the recovery rather than after, because if that arm had been
ranked first by validation the honest move would have been to re-run it properly rather
than report a final-weights number beside 212 best-validation ones.
