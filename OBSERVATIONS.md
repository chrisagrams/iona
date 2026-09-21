# Observations

Things we believe and why, separate from `STATUS.md` (what is running) and `TODO.md`
(what is broken). Each entry says what would overturn it.

---

## READ THIS FIRST: the contrastive separation ratio has sd 0.75 at a fixed seed

Six runs of one configuration, nothing varied, same seed 0:

    6.56  5.89  6.02  5.46  4.40  6.20      mean 5.75  sd 0.75  range 2.15

So the headline figure quoted throughout this project -- **7.83** -- is 2.8 sd above the
mean of its own configuration. It was a lucky draw, and it became the reference point
for later comparisons. The honest number for contrastive training is **5.75 +/- 0.75**.

The standard error on a difference between two SINGLE runs is 1.06. Applying that:

| claim | gap | t | status |
| --- | --- | --- | --- |
| pretrained vs random encoder | 6.48 | 6.1 | **stands** |
| every random cell of the factorial at the floor | ~4.4 | 4.1 | **stands** |
| frozen readout vs trained encoder | ~4.3 | 4.0 | **stands** |
| start point 133k vs 10k | 2.10 | 2.0 | suggestive only |
| layer-mix vs final layer | 2.07 | 2.0 | suggestive only |
| random budget 1.35 -> 2.73 | 1.38 | 1.3 | **not supported** |
| scheduler fix 6.94 -> 7.83 | 0.89 | 0.8 | **not supported** |
| which contrastive arm is best | 0.12 | 0.1 | **not supported** |

WHAT THIS DOES AND DOES NOT TOUCH. Every structural conclusion survives, because they
all rest on gaps of 4 or more: pretraining is necessary, fine-tuning is necessary,
neither alone works, the representation is non-linear. What dies is every RANKING --
which arm, which learning rate, which checkpoint, whether a change helped.

WHY THIS METRIC IS SO NOISY. The separation ratio is computed over 99 replicate groups;
denoise AUROC is computed over 1,680,125 peaks. Four orders of magnitude fewer units,
on a model trained at lr 5e-4, which is the least stable setting in the grid -- its four
arms span 1.35 to 7.83 including a collapse to the floor. Nondeterminism decides which
side of that edge a run lands on.

DOES DENOISE HAVE THE SAME PROBLEM? Unmeasured, and it is the obvious next worry since
the denoise scaling gaps are 0.0083 and 0.0043. But the relative scales are not
comparable: contrastive sd is 13% of its mean, while the denoise gaps are 0.88% and
0.46% of theirs. For denoise rankings to fail the same way its run-to-run sd would have
to be ~0.005 on a metric averaged over 1.68M peaks, which is implausible but not
impossible. FT5 measures it. Until FT5, treat the denoise scaling trend as likely but
unconfirmed -- which is what it was already labelled.

## WITHDRAWN: "more negatives do not help". The GradCache verdict was wrong three ways

Recorded in the levers table as a failed idea: "more negatives (GradCache, batch 64):
best 5.70, BELOW the batch-4 best". All five GradCache runs, jobs 8841066 and 8841165:

    gc_ep3 5.70 | gc_ep10 5.35 | gc_ep30 4.37 | gc_ep60 4.34 | gc_ep100 3.35

Three separate errors in reading them:

  WRONG REFERENCE. 5.70 was compared against the batch-4 BEST, 6.94 at the time and
  later 7.83. Both are single draws from a distribution whose mean is 5.75 with sd 0.75
  (job 8844111). Against the mean, 5.70 is a dead heat.

  CONFOUNDED. The GradCache config runs temperature 0.2; the batch-4 arm it was compared
  against runs 0.07. That is not an A/B on batch size, and temperature is not a small
  axis here -- at lr 5e-4 / KL 10 the two temperatures gave 7.83 and 4.33.

  MATCHED COMPARISON REVERSES IT. Holding lr, KL, temperature and epochs fixed and
  varying only the batch:

      batch 4,  lr5e4 kl10 t0.2 ......... 4.33
      batch 64, lr5e4 kl10 t0.2 (gc_ep3)  5.70      +1.37, t = 1.3

  Suggestive that more negatives HELP, which is the opposite of what was recorded.
  Not significant at t = 1.3, so the honest statement is that the effect of batch size
  has never been measured well enough to say either way.

The 5.70 -> 3.35 decline across 3 to 100 epochs is FT14, not a property of GradCache:
with `set_epoch` never called the batches are fixed, so extra epochs replay the same 15
negative groups instead of sampling new ones. That is overfitting a frozen set of
contrasts.

WHAT THIS MEANS FOR THE PLAN. GradCache at P=16/K=4 was the obvious next thing to try
and it was crossed off the list on a misreading. It goes back on, and it should be run
AFTER FT14 is fixed, since fixed batches are precisely what neutralises extra negatives:
60 negatives drawn fresh each epoch is a different proposition from the same 60 every
time.

## The pretrained encoder learns a real representation, but a NON-LINEAR one that only
## becomes useful after post-training

The central finding, and it took two contradictory-looking results to see it:

| | pretrained | random init |
| --- | --- | --- |
| frozen embedding, separation ratio | **1.35** | **1.35** |
| after contrastive fine-tuning | **7.83** | **1.35** |

Frozen, the pretrained encoder is indistinguishable from an untrained one -- at every
layer, every pooling mode, and all three model scales. Fine-tuned, it reaches 7.83 while
the random encoder does not move off the floor in any of twelve hyperparameter arms.

So pretraining is building structure that no linear readout we can construct can reach:
not the final layer, not any intermediate layer, not mean or max or intensity-weighted
pooling, not a trained convex mixture over all depths. The structure is there -- the
fine-tuning result proves it -- but it is entangled, and a projection cannot recover it.
Post-training is what makes it linearly accessible.

This reverses the natural reading of the floor result. "The frozen embedding carries no
peptide identity" is true. "Therefore pretraining is useless here" does not follow and
is false.

WHAT WOULD OVERTURN IT: a random encoder reaching comparable separation given a larger
budget. Partially tested -- random had 1,347 steps and stayed at 1.35 --
`configs/sweep-random-budget` extends that to 13,470.

WHY IT MATTERS FOR THE PROJECT: it kills the whole training-free probe programme as a
model-selection tool. The frozen separation ratio does not predict fine-tuned
performance, so it cannot be used to choose a checkpoint, a layer, or a scale. Every
comparison has to go through an actual fine-tune.

---

## Blending encoder depths is worse than using one, and the readout axis is closed

Job 8842351 ran the layer mixture with the learning rate corrected so it could not
saturate. It produced a real blend this time -- final entropy 2.065 against 2.398 for
uniform and 0 for one-hot -- and the blend is WORSE than the collapsed run at every
encoder learning rate:

| arm | v1, collapsed to one layer | v2, genuine blend |
| --- | --- | --- |
| frozen | 1.52 | **1.49** |
| els03 (0.3x) | 3.46 | **2.18** |
| els10 (1.0x) | 4.28 | **4.14** |

Both sit far below the plain final-layer `mean+max` contrastive result. So mixing depths
does not help, and combining them is actively worse than committing to one -- most
visibly at els03, 3.46 to 2.18.

The mixture learned the RIGHT thing about where the signal is and still lost. Its weight
concentrated on the top of the stack -- indices 6 to 10 hold about 83% of the mass, peak
at index 8 -- which is exactly the region the frozen layer probe independently found
best (blocks 5-8, ratios 1.51-1.53). It identified the useful depths correctly and
averaging them still degraded the embedding.

LIKELY MECHANISM, not established: LayerNorm puts every depth on a comparable scale but
cannot align their DIRECTIONS. Summing representations whose useful axes point
differently partially cancels them, so a weighted average of several good layers can be
worse than the best one alone. This would also explain why the effect grows with the
number of depths carrying real weight.

WHAT THIS CLOSES. Layer choice, pooling mode, peak weighting, model scale, pretraining
duration and now a trained mixture over depths have all been tested. Nothing on the
readout axis moves the embedding. Combined with the non-linearity finding above, the
reason is clear: the information is not recoverable by ANY linear function of the
activations, so no amount of choosing or combining them helps.

CAVEAT ON THE COMPARISON. The layer-mix arms run at the template's lr 2e-5 and KL 100,
while the 7.83 figure is the best arm of a sweep over lr, KL and temperature. The
nearest matched grid point is lr2e5_kl10_t007 at 6.21. Layer-mix is worse either way,
but 4.14 against 6.21 is the fair comparison, not 4.14 against 7.83.

I predicted a genuine blend would land close to the final-layer result rather than beat
it. It landed well below it. The direction was right and the magnitude was not.

## The complete readout x encoder x init factorial: only ONE cell works

Every combination of readout, encoder treatment and initialisation has now been run.
Separation ratio, random-init floor 1.35:

| readout | encoder | pretrained | random |
| --- | --- | --- | --- |
| final layer | frozen | 1.35 - 1.53 | **1.35** |
| final layer | 1.0x trains | **7.83** | **1.35** |
| depth mixture | frozen | 1.49 | **1.35** |
| depth mixture | 0.3x | 2.18 | **1.35** |
| depth mixture | 1.0x trains | 4.14 | **1.35** |

Every random cell is 1.35. Not approximately -- exactly the floor, whatever the readout
and whatever the encoder learning rate. Training the encoder does not help a random one,
and neither does a trained mixture over its depths.

On the pretrained side only ONE cell is interesting, and it is the plain one: final
layer with the encoder training, 7.83. Every elaboration of the readout makes it worse.

So both conditions are necessary and neither is sufficient:

  pretrained weights WITHOUT encoder training ...... 1.35 to 1.53, near the floor
  encoder training WITHOUT pretrained weights ...... 1.35, exactly the floor
  both together .................................... 7.83

That is the cleanest statement of the non-linearity finding. Pretraining deposits
structure that no readout can extract and that training cannot create from scratch --
it can only be unlocked, by fine-tuning the weights that hold it.

## Pretraining is worth more than 10x the fine-tuning budget, and probably far more

The value of pretraining is measured in extra training saved, not in a ceiling a random
encoder could never reach. Job 8843262 put numbers on it by varying only the budget on a
random encoder, at the best pretrained arm's hyperparameters:

| budget | steps | random encoder | pretrained |
| --- | --- | --- | --- |
| 3 epochs | 1,347 | **1.35** (the floor) | **7.83** |
| 10 epochs | 4,490 | 2.10 | -- |
| 30 epochs | 13,470 | **2.73** | -- |

A random encoder DOES learn. At 10x the budget it reaches 2.73, well clear of the 1.35
floor. So the earlier statement that "the contrastive loss achieves literally nothing on
a random encoder" was a claim about 1,347 steps and not about random encoders, and it
was over-claimed on a single budget point.

What the curve actually says. The ratio climbs roughly linearly in log(steps) at about
1.38 per decade, and the increments are already shrinking (+1.43 then +1.32 per decade).
Pretrained reaches 7.83 at 1,347 steps; random is at 2.73 after ten times that. So
pretraining is worth MORE THAN 10x the fine-tuning budget, which is the number this
experiment establishes.

A naive log-linear extrapolation says random would need ~7e7 steps -- around 50,000x --
to reach 7.83. DO NOT USE THAT NUMBER. Log-linear scaling has no reason to hold four
decades past the data, and the corpus has only 898 training groups, so the curve will
saturate somewhere well before then. The honest statement is the measured one: >10x, and
the gap is not closing fast enough for a 10x budget to matter.

The denoise side tells the same story with a much smaller magnitude, which is the
informative contrast -- but it has to be reported at MATCHED budget, because the two
grids did not sweep the same epoch counts. Full breakdown, best arm at each setting:

| epochs | pretrained 50m | scratch 50m |
| --- | --- | --- |
| 2 | 0.9272 (108 arms) | not run |
| 4 | **0.9320** (108 arms) | 0.8856 (6 arms) |
| 8 | **not run** | **0.9001** (6 arms) |

Two different comparisons, and both belong in any report:

  MATCHED at 4 epochs ......... 0.9320 vs 0.8856, pretraining worth **+0.046**
  scratch given DOUBLE budget . 0.9320 vs 0.9001, pretraining worth **+0.032**

Quoting only the second understates pretraining, because it hands the random encoder
twice the training. Quoting only the first ignores that the random encoder is still
improving at the point its grid stops -- all five of its top arms are 8 epochs, and ep4
to ep8 buys it +0.015 while ep2 to ep4 buys the pretrained model only +0.005.

THE MISSING CELL is pretrained at 8 epochs, which nothing has run. Without it we cannot
say whether the +0.032 figure is pretraining's true advantage at equal wall-clock or
whether the pretrained model would also gain from the extra epochs and restore the gap.
Given it gained only +0.005 going from 2 to 4 epochs it is probably close to saturated,
so +0.032 is likely near the honest number at double budget -- but that is an inference,
not a measurement, and the cell is cheap to fill.

## KL regularisation is not needed in general; it prevents collapse at high learning rates

An earlier note of mine said "the KL term is doing real work", which was too broad. What
the corrected contrastive grid actually shows:

| learning rate | KL 0 | KL 10 |
| --- | --- | --- |
| 2e-5 | **7.71** | 6.69 |
| 1e-4 | 6.41 | 7.14 |
| 5e-4 | **1.35** (collapsed) | **7.83** (best) |

At 2e-5 the best arm has NO KL at all. At 5e-4, KL 0 collapses to the floor while KL 10
gives the best result in the grid. So KL is a stabiliser that buys usable behaviour at
learning rates that would otherwise diverge, not a term the objective needs. If you run
at a low learning rate you can drop it.

---

## `final/` is a moving target, and reading one as a fixed checkpoint has cost us twice

The published `final/` directory of a pretraining run is a periodically refreshed export
of a run still in progress. It is not the end of training and it changes under you. The
50m `final/` was checkpoint-133233 against a last checkpoint of 180000.

This produced a wrong published conclusion -- "scale hurts the frozen embedding" --
which was really two models read at ~135k steps compared against one at ~193k. Withdrawn
after a matched-step probe showed no effect at all.

Everything now points at frozen copies under `/flare/UIC-HPC/khuss/msdelta/pretrained/`,
named by step number, with `results/checkpoint_provenance.txt` recording which is which.

---

## Reading a smoke-run log as a real run cost three wrong conclusions

Recorded because the failure mode is subtle and will recur. Job 8840665 was a contrastive
smoke run with `max_samples` applied; its log says `train_groups=60`. The real corpus has
**898 training groups**, so a 3-epoch run is **1,347 optimizer steps**, not the 90 that
60 groups implies. Three things were derived from the wrong number:

  FT10, "warmup_steps 100 exceeds the whole run" -- false. 100 of 1,347 is 7.4%, normal.
  `layer_mix_lr 5e-2`, sized for 90 steps, overshoots ~15x at 1,347. It is why the
    mixture collapsed onto one layer immediately instead of learning a blend.
  The random-init caveat, stated as "only 90 steps so random had no chance" -- it had
    1,347 and still did not move.

LESSON: a run's own log reports the corpus it actually loaded. Check that number against
the dataset before deriving anything from it, and never carry a figure across jobs.

---

## Pooled and within-spectrum AUROC agree on denoise; the headline numbers hold

Job 8842917 re-scored the finished winners with AUROC computed inside each spectrum and
averaged, alongside the pooled figure. Both from the same evaluation, so they are
directly comparable:

| winner | pooled | per-spectrum | sd | p10 | unscorable |
| --- | --- | --- | --- | --- | --- |
| 50m lr2e4_es05_ep4_h512_b12 | 0.9213 | **0.9269** | 0.067 | 0.856 | 17 of 8584 |
| 100m lr2e4_es05_b12 | 0.9331 | **0.9377** | 0.070 | 0.878 | 17 of 8584 |

Per-spectrum is slightly HIGHER than pooled, by 0.006 in both cases -- so pooling was
mildly deflating the number, not inflating it. The feared mechanism, a per-spectrum
offset propping up the pooled figure, is not operating. The ordering is preserved too:
100m beats 50m by the same ~0.011 on either metric.

The distribution is tight rather than a hidden split. sd around 0.07, and the 10th
percentile still above 0.85, so there is no substantial subset of spectra the model
fails on while the average looks fine. Only 17 spectra of 8,584 are unscorable for
being single-class.

So the denoise result stands on the axis the model is actually used on. Worth having
checked -- the direction was not predictable, see TODO FT12 -- but the answer is that
nothing was wrong.

NOTE ON THE ABSOLUTE NUMBERS: these are FINAL-weights evaluations (0.9213, 0.9331)
against the grids' best-validation figures (0.9320, 0.9403). The ~0.008-0.011 gap is
best-vs-final selection, measured earlier at ~0.006, and is not a discrepancy. The
pooled-vs-per-spectrum comparison above is unaffected because both come from the same
evaluation pass.

---

## Training the depth mixture beats a uniform average, and loses to picking one layer

The baseline the frozen layer-mix arm was missing, measured training-free at the same
checkpoint the arm used (job 8842806):

| readout on the frozen 50m | ratio |
| --- | --- |
| uniform mixture over all 11 depths, UNTRAINED | 1.43 |
| the same mixture after contrastive training | 1.49 |
| block 8 alone, picked by hand off the layer probe | **1.53** |
| mean pooling of the output layer | 1.43 |
| random-init floor | 1.35 |

So training those twelve weights did achieve something, +0.06 over the uniform average,
which the earlier report could not establish either way. And it still loses to reading
one layer chosen by eye. A trained linear reweighting of depths is worse than a good
guess, and both sit within 0.2 of a random network.

That closes the loop on the frozen arm: it is a real measurement, it is just a small
one, and it points the same way as everything else on this axis.

## Contrastive scaling saturates at 100m -- earlier than denoise, and with error bars

Six seeds per scale, so these are means rather than draws:

| scale | mean | sd | step | t |
| --- | --- | --- | --- | --- |
| 50m | 5.86 | 0.63 | -- | -- |
| 100m | **7.00** | 0.49 | +1.14 | **+3.5** |
| 200m | 7.02 | 0.44 | +0.02 | +0.1 |
| 400m | 6.74 | 0.28 | -0.28 | -1.3 |

One real step, 50m to 100m, and then nothing. Doubling past 100m buys no separation at
all, and 400m is slightly below 200m by less than the noise.

I previously said contrastive was "still climbing at 100m" while denoise had plateaued,
and used that divergence as the argument for debugging the GPU fault rather than
skipping it. With four points instead of two, BOTH objectives saturate -- contrastive
at 100m, denoise at 200m. The debugging was still worth it, but the claimed divergence
was an artefact of having only two points on one of the curves.

TAKEN TOGETHER WITH DENOISE: 0.9320 / 0.9403 / 0.9446 / 0.9436, saturating at 200m.
Neither objective rewards capacity past 200m, and contrastive stops paying at 100m. For
anything that has to choose a single encoder, 100m to 200m is the whole useful range,
and that is now measured on both objectives with seeds rather than inferred from single
runs.

CAVEAT ON COMPARABILITY: the 200m and 400m arms ran with fixed-width padding while the
50m and 100m arms ran with the old batch-maximum padding. Padded positions are masked
out of attention, so the computation should be identical and only memory differs -- but
that is an argument, not a measurement. Re-running 50m and 100m under fixed width would
settle it, and is cheap.

## Denoise scaling saturates at 200m; 400m buys nothing

All four grids are complete, 12 arms each except the 216-arm 50m:

| scale | best AUROC | best F1 | gain | top-cluster spread |
| --- | --- | --- | --- | --- |
| 50m | 0.9320 | 0.8632 | -- | 0.0023 |
| 100m | 0.9403 | 0.8723 | +0.0083 | 0.0012 |
| 200m | **0.9446** | **0.8778** | +0.0043 | 0.0010 |
| 400m | 0.9436 | 0.8768 | **-0.0010** | 0.0013 |

> **CORRECTED by FT5 (job 8845262).** The paragraph below called 200m -> 400m a plateau.
> Six seeds at each scale say it is a regression: 200m 0.9447 +/- 0.00025 against 400m
> 0.9434 +/- 0.00045, a gap of -0.0013 at t = -6.3. Doubling from 200m to 400m does not
> buy nothing -- it costs something measurable.
>
> THE REASONING IS WHERE THE ERROR WAS, not the arithmetic. "By less than either grid's
> top-cluster spread" treats the spread across a grid's best few ARMS as the error bar
> on a single arm. Those are different quantities: the top-cluster spread mixes real
> hyperparameter effects with noise, so it is an UPPER bound on noise and using it as
> the error bar is systematically too conservative. It dismisses real effects, always in
> the same direction. The right error bar is the seed spread at one fixed configuration,
> which is sd ~0.0005 -- and against that, even the naive -0.0010 was two sigma.
>
> Wherever this file or STATUS.md calls a denoise difference "within noise" on the
> strength of a grid's internal spread, the judgement needs redoing against 0.0005.

The increments halve and then reverse: +0.0083, +0.0043, -0.0010 (single seed) or
-0.0013 (six seeds). 200m is the largest scale worth using for denoise AT THIS
FINE-TUNING BUDGET, which is the caveat that matters: the 400m probe (job 8845252)
shows 400m still gaining at 4 epochs, +0.0021 from ep2 to ep4 against sd 0.0005. So
"400m is worse than 200m" and "400m has not finished converging" are both true, and
only the second is a statement about scale itself.

THE SAME HYPERPARAMETERS WIN AT EVERY SCALE. `lr2e4_es05_b12` is the top arm at 100m,
200m and 400m, and the same settings won the 216-arm 50m grid. Four scales, one answer.
The ranking WITHIN each top cluster is noise -- five arms inside 0.001 -- but the
hyperparameter choice itself transfers, which is what justified narrowing from 216 arms
to 12 and is now confirmed three times over.

400m also carries the first per-spectrum numbers measured at training time rather than
backfilled: 0.9485 per-spectrum against 0.9436 pooled, the same direction and roughly
the same size as the 50m and 100m backfills (+0.005). Pooling mildly deflates; nothing
is hidden.

CAVEAT ON COMPUTE BUDGET: the 400m started from checkpoint-181381 (33.6% of its
schedule) and the 200m from checkpoint-192799 (35.7%), so 400m had about 6% less
pretraining. Too small to explain a 0.0010 gap that is itself inside the noise, but it
belongs in any figure normalised by compute.

WHAT THIS MAKES URGENT. FT5 is no longer a refinement. The 200m-vs-400m difference is
0.0010 against top-cluster spreads of 0.0010 and 0.0013, so there is currently NO basis
for ranking them, and the plateau claim itself rests on single runs. After what the
contrastive metric turned out to hide -- sd 0.75, and a headline figure 2.8 sd above its
own mean -- assuming denoise is quieter without measuring it is exactly the mistake to
not repeat.

## The complete readout x encoder x init factorial: only ONE cell works

Every combination of readout, encoder treatment and initialisation has now been run.
Separation ratio, random-init floor 1.35:

| readout | encoder | pretrained | random |
| --- | --- | --- | --- |
| final layer | frozen | 1.35 - 1.53 | **1.35** |
| final layer | 1.0x trains | **7.83** | **1.35** |
| depth mixture | frozen | 1.49 | **1.35** |
| depth mixture | 0.3x | 2.18 | **1.35** |
| depth mixture | 1.0x trains | 4.14 | **1.35** |

Every random cell is 1.35. Not approximately -- exactly the floor, whatever the readout
and whatever the encoder learning rate. Training the encoder does not help a random one,
and neither does a trained mixture over its depths.

On the pretrained side only ONE cell is interesting, and it is the plain one: final
layer with the encoder training, 7.83. Every elaboration of the readout makes it worse.

So both conditions are necessary and neither is sufficient:

  pretrained weights WITHOUT encoder training ...... 1.35 to 1.53, near the floor
  encoder training WITHOUT pretrained weights ...... 1.35, exactly the floor
  both together .................................... 7.83

That is the cleanest statement of the non-linearity finding. Pretraining deposits
structure that no readout can extract and that training cannot create from scratch --
it can only be unlocked, by fine-tuning the weights that hold it.

## Pretraining is worth more than 10x the fine-tuning budget, and probably far more

The value of pretraining is measured in extra training saved, not in a ceiling a random
encoder could never reach. Job 8843262 put numbers on it by varying only the budget on a
random encoder, at the best pretrained arm's hyperparameters:

| budget | steps | random encoder | pretrained |
| --- | --- | --- | --- |
| 3 epochs | 1,347 | **1.35** (the floor) | **7.83** |
| 10 epochs | 4,490 | 2.10 | -- |
| 30 epochs | 13,470 | **2.73** | -- |

A random encoder DOES learn. At 10x the budget it reaches 2.73, well clear of the 1.35
floor. So the earlier statement that "the contrastive loss achieves literally nothing on
a random encoder" was a claim about 1,347 steps and not about random encoders, and it
was over-claimed on a single budget point.

What the curve actually says. The ratio climbs roughly linearly in log(steps) at about
1.38 per decade, and the increments are already shrinking (+1.43 then +1.32 per decade).
Pretrained reaches 7.83 at 1,347 steps; random is at 2.73 after ten times that. So
pretraining is worth MORE THAN 10x the fine-tuning budget, which is the number this
experiment establishes.

A naive log-linear extrapolation says random would need ~7e7 steps -- around 50,000x --
to reach 7.83. DO NOT USE THAT NUMBER. Log-linear scaling has no reason to hold four
decades past the data, and the corpus has only 898 training groups, so the curve will
saturate somewhere well before then. The honest statement is the measured one: >10x, and
the gap is not closing fast enough for a 10x budget to matter.

The denoise side tells the same story with a much smaller magnitude, which is the
informative contrast -- but it has to be reported at MATCHED budget, because the two
grids did not sweep the same epoch counts. Full breakdown, best arm at each setting:

| epochs | pretrained 50m | scratch 50m |
| --- | --- | --- |
| 2 | 0.9272 (108 arms) | not run |
| 4 | **0.9320** (108 arms) | 0.8856 (6 arms) |
| 8 | **not run** | **0.9001** (6 arms) |

Two different comparisons, and both belong in any report:

  MATCHED at 4 epochs ......... 0.9320 vs 0.8856, pretraining worth **+0.046**
  scratch given DOUBLE budget . 0.9320 vs 0.9001, pretraining worth **+0.032**

Quoting only the second understates pretraining, because it hands the random encoder
twice the training. Quoting only the first ignores that the random encoder is still
improving at the point its grid stops -- all five of its top arms are 8 epochs, and ep4
to ep8 buys it +0.015 while ep2 to ep4 buys the pretrained model only +0.005.

THE MISSING CELL is pretrained at 8 epochs, which nothing has run. Without it we cannot
say whether the +0.032 figure is pretraining's true advantage at equal wall-clock or
whether the pretrained model would also gain from the extra epochs and restore the gap.
Given it gained only +0.005 going from 2 to 4 epochs it is probably close to saturated,
so +0.032 is likely near the honest number at double budget -- but that is an inference,
not a measurement, and the cell is cheap to fill.

## KL regularisation is not needed in general; it prevents collapse at high learning rates

An earlier note of mine said "the KL term is doing real work", which was too broad. What
the corrected contrastive grid actually shows:

| learning rate | KL 0 | KL 10 |
| --- | --- | --- |
| 2e-5 | **7.71** | 6.69 |
| 1e-4 | 6.41 | 7.14 |
| 5e-4 | **1.35** (collapsed) | **7.83** (best) |

At 2e-5 the best arm has NO KL at all. At 5e-4, KL 0 collapses to the floor while KL 10
gives the best result in the grid. So KL is a stabiliser that buys usable behaviour at
learning rates that would otherwise diverge, not a term the objective needs. If you run
at a low learning rate you can drop it.

---

## `final/` is a moving target, and reading one as a fixed checkpoint has cost us twice

The published `final/` directory of a pretraining run is a periodically refreshed export
of a run still in progress. It is not the end of training and it changes under you. The
50m `final/` was checkpoint-133233 against a last checkpoint of 180000.

This produced a wrong published conclusion -- "scale hurts the frozen embedding" --
which was really two models read at ~135k steps compared against one at ~193k. Withdrawn
after a matched-step probe showed no effect at all.

Everything now points at frozen copies under `/flare/UIC-HPC/khuss/msdelta/pretrained/`,
named by step number, with `results/checkpoint_provenance.txt` recording which is which.

---

## Reading a smoke-run log as a real run cost three wrong conclusions

Recorded because the failure mode is subtle and will recur. Job 8840665 was a contrastive
smoke run with `max_samples` applied; its log says `train_groups=60`. The real corpus has
**898 training groups**, so a 3-epoch run is **1,347 optimizer steps**, not the 90 that
60 groups implies. Three things were derived from the wrong number:

  FT10, "warmup_steps 100 exceeds the whole run" -- false. 100 of 1,347 is 7.4%, normal.
  `layer_mix_lr 5e-2`, sized for 90 steps, overshoots ~15x at 1,347. It is why the
    mixture collapsed onto one layer immediately instead of learning a blend.
  The random-init caveat, stated as "only 90 steps so random had no chance" -- it had
    1,347 and still did not move.

LESSON: a run's own log reports the corpus it actually loaded. Check that number against
the dataset before deriving anything from it, and never carry a figure across jobs.

---

## Pooled and within-spectrum AUROC agree on denoise; the headline numbers hold

Job 8842917 re-scored the finished winners with AUROC computed inside each spectrum and
averaged, alongside the pooled figure. Both from the same evaluation, so they are
directly comparable:

| winner | pooled | per-spectrum | sd | p10 | unscorable |
| --- | --- | --- | --- | --- | --- |
| 50m lr2e4_es05_ep4_h512_b12 | 0.9213 | **0.9269** | 0.067 | 0.856 | 17 of 8584 |
| 100m lr2e4_es05_b12 | 0.9331 | **0.9377** | 0.070 | 0.878 | 17 of 8584 |

Per-spectrum is slightly HIGHER than pooled, by 0.006 in both cases -- so pooling was
mildly deflating the number, not inflating it. The feared mechanism, a per-spectrum
offset propping up the pooled figure, is not operating. The ordering is preserved too:
100m beats 50m by the same ~0.011 on either metric.

The distribution is tight rather than a hidden split. sd around 0.07, and the 10th
percentile still above 0.85, so there is no substantial subset of spectra the model
fails on while the average looks fine. Only 17 spectra of 8,584 are unscorable for
being single-class.

So the denoise result stands on the axis the model is actually used on. Worth having
checked -- the direction was not predictable, see TODO FT12 -- but the answer is that
nothing was wrong.

NOTE ON THE ABSOLUTE NUMBERS: these are FINAL-weights evaluations (0.9213, 0.9331)
against the grids' best-validation figures (0.9320, 0.9403). The ~0.008-0.011 gap is
best-vs-final selection, measured earlier at ~0.006, and is not a discrepancy. The
pooled-vs-per-spectrum comparison above is unaffected because both come from the same
evaluation pass.

---

## Training the depth mixture beats a uniform average, and loses to picking one layer

The baseline the frozen layer-mix arm was missing, measured training-free at the same
checkpoint the arm used (job 8842806):

| readout on the frozen 50m | ratio |
| --- | --- |
| uniform mixture over all 11 depths, UNTRAINED | 1.43 |
| the same mixture after contrastive training | 1.49 |
| block 8 alone, picked by hand off the layer probe | **1.53** |
| mean pooling of the output layer | 1.43 |
| random-init floor | 1.35 |

So training those twelve weights did achieve something, +0.06 over the uniform average,
which the earlier report could not establish either way. And it still loses to reading
one layer chosen by eye. A trained linear reweighting of depths is worse than a good
guess, and both sit within 0.2 of a random network.

That closes the loop on the frozen arm: it is a real measurement, it is just a small
one, and it points the same way as everything else on this axis.

## Denoise improves with scale, but the increments are shrinking faster than we can resolve

| scale | best test AUROC | best F1 | gain over previous | top-cluster spread |
| --- | --- | --- | --- | --- |
| 50m | 0.9320 | 0.8632 | -- | 0.0023 over 8 arms |
| 100m | 0.9403 | 0.8723 | +0.0083 | 0.0012 over 5 arms |
| 200m | **0.9446** | **0.8778** | +0.0043 | **0.0010 over 5 arms** |

The direction is consistent and the gains are real relative to the measurement so far.
But each doubling buys about half what the previous one did while the within-grid spread
stays near 0.001, so the trend and the noise are converging. Extrapolating, 400m would
gain ~0.002 -- which is INSIDE the top-cluster spread of a single grid.

That makes FT5 a precondition rather than a refinement at 400m. Without seeds we will
not be able to say whether 400m beats 200m at all, and a "400m is best" claim would rest
on one sample per scale with differences smaller than the spread between arms we already
know are equivalent.

One encouraging detail that is NOT about ranking: `lr2e4_es05_b12` is the top arm at both
100m and 200m, and the same hyperparameters won at 50m. The individual ranking inside a
grid's top cluster is arbitrary, but the hyperparameter choice appears to transfer across
scale, which is what justified narrowing from 216 arms to 12.

## Denoise is the line that works; the reranker does not need a neural embedding

50m grid best test AUROC **0.9320** over 216 arms; 100m best **0.9403** over 12. A
hand-built feature rescorer reaches **hit@1 0.889** on fragment coverage, mass error and
spectrum quality with no neural embedding at all -- and adding the embedding cosine made
it WORSE by 0.109 hit@1 over five paired seeds, because every candidate for a spectrum is
scored against one shared cached vector, so its errors correlate within a spectrum.

Neither grid ranking is known to be real yet: the 50m top eight span 0.0023 test AUROC
and the 100m top five span 0.0012, on one seed each. FT5 is what settles that.

## The same hyperparameter point wins at every scale

All four denoise HP grids independently ranked `lr2e4_es05_b12` first -- lr 2e-4,
encoder_lr_scale 0.5, effective batch 12:

    50m   0.9320      200m  0.9446
    100m  0.9403      400m  0.9436

This matters for FT5. The seed grid repeats one shared configuration at all four
scales, and that would confound scale with "how well 50m's hyperparameters transfer"
if the scales disagreed about the best point. They do not, so the scale curve is each
scale at its own optimum.

The weaker half of the claim: within each grid the top three arms span 0.0003-0.0008,
against a seed noise of ~0.0005. So "lr2e4_es05_b12 is best" is not resolved at any
single scale -- it is the same point landing at or near the top four times
independently, which is better evidence than any one grid provides.

WHAT WOULD OVERTURN IT: a scale whose grid picks a materially different point, or a
repeat of one grid whose top arm changes identity. The second is likely for the runner-
up ordering and would not disturb the claim; a change in the winner would.

## encoder_lr_scale peaks at 0.5 at 400m -- it is not monotone

    400m, lr 2e-4          es 0.1   es 0.25   es 0.5   es 1.0
    effective batch 12     0.9259*  0.9384*   0.9436   0.9396
    (* from the ep2 probe arms, which are 2 epochs rather than 4)

Reading only the probe, which sampled 0.1 / 0.25 / 0.5, the trend is monotone
increasing and the natural inference is that 1.0 would be better still. It is not:
0.9396 against 0.9436, worse by 0.0040, which is eight times the seed noise. The curve
is an inverted U and the probe sampled its rising half.

There is an interaction with batch size, so the peak is not at 0.5 unconditionally:

    effective batch 48     es 0.5 -> 0.9423    es 1.0 -> 0.9434

At the larger batch the ordering reverses. Both differences clear the noise, so this is
a real interaction rather than two noisy draws.

WHY IT IS RECORDED AS A LESSON AND NOT JUST A NUMBER: the error was extrapolating a
trend past the last sampled point, and the refuting measurement already existed
elsewhere in the project. Before extending a grid at its edge, check whether that point
has been run somewhere else first.

WHAT WOULD OVERTURN IT: es 0.75 at batch 12 landing above 0.9436, which would make the
peak a plateau rather than a maximum at 0.5.
