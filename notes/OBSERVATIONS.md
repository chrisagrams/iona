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
named by step number, with `results/finetune/checkpoint_provenance.txt` recording which is which.

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

## Contrastive scaling saturates at 100m -- MEASURED UNDER FT14, needs re-taking
<!-- Every arm trained on 15.4% of the corpus, the same 15.4% each epoch, because
     the PK sampler never reshuffled. The shape probably survives since the handicap
     applied at every scale, but the levels are depressed: the 50m cell rises from
     5.86 to 6.59 once the sampler is fixed. -->

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

## SUPERSEDED: "denoise scaling saturates at 200m; 400m buys nothing"
<!-- 400m is not a plateau, it is a regression. See "The denoise scale curve turns
     over at 400m (FT5, complete)" below for the six-seed measurement. -->

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
## SUPERSEDED: "increments shrinking faster than we can resolve"
<!-- They are resolvable. Seed noise at a fixed configuration is sd ~0.0005 and the
     scale steps run t=30-53. The "cannot resolve" judgement used each grid's
     internal arm spread as the error bar, which overstates noise. -->

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

## The denoise scale curve turns over at 400m (FT5, complete)

Six seeds at each of four scales, one fixed configuration, 24/24 arms (job 8845262):

| scale | mean test AUROC | sd | step | gap | t |
| --- | --- | --- | --- | --- | --- |
| 50m | 0.9317 | 0.00055 | -- | -- | -- |
| 100m | 0.9400 | 0.00029 | 50m -> 100m | +0.0083 | +32.7 |
| 200m | **0.9447** | 0.00025 | 100m -> 200m | +0.0047 | +30.2 |
| 400m | 0.9434 | 0.00045 | 200m -> 400m | **-0.0013** | **-6.3** |

The curve rises, decelerates, and turns over. Every step resolves, including the last.
This replaces the earlier reading of a plateau, which came from one seed per scale
judged against each grid's internal arm spread -- an upper bound on noise, not a
measurement of it.

The configuration is not a confound: `lr2e4_es05_b12` was independently the top arm of
all four HP grids, so each scale is repeated at its own selected point.

Grid bests sit ~0.0003 above the seed means at every scale, because a grid best is a
max over arms. The bias is consistent, so the gaps move very little, but no single grid
best should be quoted as that scale's expected score.

THE CLAIM IS ABOUT A BUDGET, NOT ABOUT SCALE. All of this is at 4 epochs, and the 400m
probe has 400m still gaining there: +0.0021 from ep2 to ep4, four times this noise. So
"400m is worse than 200m at 4 epochs" is established, and "400m is worse than 200m" is
not. The 8-epoch row -- 8846027 at 50m, the ep8-mid grid at 100m/200m, and the recovered
probe arms at 400m -- is what separates them.

WHAT WOULD OVERTURN IT: 400m at 8 epochs reaching or passing 200m at 8 epochs, which
would make the turnover an artefact of under-training the larger model rather than a
property of scale. That is a live possibility, not a hedge.

## Denoise has two variances and they differ by 2x; use the right one

Measured separately rather than assumed:

    fixed-seed   400m, 4 byte-identical configs INCLUDING --seed   sd 0.00022
                 0.9434 0.9435 0.9436 0.9439 (job 8845252 repeats)
                 XPU reduction nondeterminism alone -- nothing else varies.

    across-seed  each scale, 6 seeds                               sd ~0.00050
                 50m 0.00055, 100m 0.00029, 200m 0.00025, 400m 0.00045
                 adds head initialisation, data order and dropout.

Use across-seed for anything compared across configurations, which is almost
everything. Fixed-seed is only the right bar for "I re-ran the identical thing".

FOR SCALE: denoise is ~0.06% relative, contrastive is ~13% (sd 0.75 on a mean of
5.75). The two lines need completely different sample sizes, and intuitions carried
from one to the other will be wrong by a factor of 200. Most of the "within noise"
judgements this project got wrong on denoise came from importing contrastive caution.

WHAT WOULD OVERTURN IT: a scale whose six-seed sd lands far outside 0.0002-0.0006.

## Fixing the sampler (FT14) raised contrastive separation by ~13%

The PK sampler never advanced its epoch counter, so every epoch replayed identical
batches and one run only ever saw 15.4% of the corpus. With it fixed, three epochs
reach 39.8%. The same configuration, six seeds either side (the 50m cell of the scale
grid, against job 8845871 ep003):

    before fix, 15.4% of data    5.86 +/- 0.63
    after  fix, 39.8% of data    6.59 +/- 0.43
    difference +0.73    Welch t=2.37  df=8.8  p=0.043

STATED AT THE STRENGTH IT HAS: p=0.043 is suggestive, not settled, and it is one
grid. It is a different evidentiary regime from the denoise results in this file,
which run t=30-53. What corroborates it is the variance: sd falls 0.63 -> 0.43, which
is what more distinct spectra per run should do and is not something a lucky draw
produces.

CONSEQUENCE: every contrastive number in this project was measured under the
handicap. The handicap applied at every scale, so curve SHAPES probably survive, but
levels are depressed and no contrastive figure should be quoted as final until
re-taken.

WHAT WOULD OVERTURN IT: a repeat where the two distributions overlap, which at
p=0.043 is perfectly possible.

## "More contrastive epochs hurts" does not replicate

The cited record was 6.94 / 4.82 / 4.46 at 3 / 10 / 50 epochs -- an apparently clean
monotone decline that shaped how this project budgeted contrastive training. Six seeds
per point, at the stable lr 2e-5 and with the sampler fixed (job 8845871):

    epochs   mean   sd     old single run
    3        6.60   0.43   6.94
    10       6.65   0.29   4.82
    3 -> 10  +0.06  se 0.21  t=+0.27  p=0.79

Flat. The old 3 -> 10 step of -2.12 does not reproduce, and the old ep10 value of 4.82
sits about 6 sd below the new ep10 distribution -- a low draw, not a trend.

TWO CAUSES COMPOUNDED, and either alone would have been enough. The old points were
n=1 at lr 5e-4, the one rate whose other arms span 1.35 (total collapse) to 7.83. And
they predate FT14, so "more epochs" meant more passes over one frozen 15.4% slice,
which is precisely the setup in which more epochs SHOULD hurt.

Note the spread tightens with epochs, 0.43 -> 0.29, consistent with coverage: 3 epochs
now reaches 40% of the corpus and 10 reaches 82%.

WHAT IS NOT YET MEASURED: ep30 and ep100 are still running. "Flat from 3 to 10" and
"flat forever" are different claims and only the first is established.
    UPDATE 2026-09-23: ep30 and ep100 finished, 6/6 each: ratio 6.33 and 6.02. But
    this whole entry is on the separation ratio, which does not predict retrieval,
    at lr2e-5, a superseded recipe. Rescoring the saved encoders on MAP@R would settle
    it; see "Closing the loop" at the end of this file.

## The separation ratio does not predict retrieval, and contrastive FT trades Hit@1 for MAP@100

The ratio was the selection metric for every contrastive conclusion in this project. It
has now been scored against the task it proxies for, on the same validation rows, for
all 24 arms of job 8848049 plus the 50m arms' OWN base checkpoint
(50m-production-01-checkpoint-133233) as an untrained control. Same 1167 queries, same
99 groups, every encoder.

    scale    ratio    Hit@1   vs base        MAP@100  vs base
    untrained 1.34   0.6538   --             0.1727   --
    50m       9.37   0.5137   t=-5.4 p.003   0.1751   n.s.
    100m     10.36   0.4689   t=-16  p<.001  0.1561   t=-3.0 p=.032
    200m     11.50   0.4986   t=-24  p<.001  0.1676   t=-3.5 p=.017
    400m     13.84   0.5261   t=-15  p<.001  0.1976   t=+4.4 p=.007

THE PROXY FAILS VALIDATION. Spearman against the ratio, n=24: Hit@1 +0.28 (p=0.18),
R@5 +0.27 (p=0.20), MAP@100 +0.40 (p=0.055). None significant. Within a single scale,
where only the seed differs, the mean rank correlation with Hit@1 is -0.04 -- the ratio
cannot order runs at all. The ratio is monotone in scale (9.37 -> 13.84); neither task
metric is, and 100m sits below 50m on both.

THE TWO TASK METRICS DISAGREE IN SIGN, so reporting one alone misleads. Hit@1 asks
whether the single nearest neighbour is a replicate -- a LOCAL property. MAP@100 scores
the whole ranked list -- a GLOBAL one. Contrastive training pushes group centroids
apart, which is exactly what improves the second and damages the first. It costs Hit@1
at every scale without exception (0/24 arms beat the untrained encoder) and only beats
the untrained encoder on MAP@100 at 400m; at 100m and 200m it is significantly WORSE
than not training at all.

WHY THE UNTRAINED ENCODER IS SO STRONG ON Hit@1: replicate spectra of one peptide are
near-identical as raw peaks, so almost any embedding puts them adjacent. 0.654 is far
above the ~0.009 a random ranking would give. The pretrained representation already
solves the local problem; contrastive training then degrades it.

STATED AT THE STRENGTH IT HAS: 400m beating 50m on MAP@100 is t=+1.77, p=0.11 at six
seeds -- suggestive, not established. The 400m-vs-untrained result (p=0.007) is the
solid one.

CONSEQUENCE: "contrastive separation improves with scale" is a statement about the
proxy. On the task, scale buys MAP@100 only by 400m and buys no Hit@1 anywhere. Arms
must be selected on MAP@100 with Hit@1 reported beside it, never on the ratio.
finetune_contrastive.py already emits both per arm, so grids run with current code need
no rescoring pass; older jobs need msdelta.eval_retrieval.

WHAT IS NOT YET MEASURED: whether a different contrastive configuration (temperature,
loss, or fewer epochs) can raise MAP@100 without spending Hit@1. Every arm here used
one configuration per scale.
    UPDATE 2026-09-23: yes -- lr1e-4/KL10/t0.07 raises both (Hit@1 p=0.0007, MAP@100
    p=0.0002), and t0.03 raises MAP@R further. See the next entries.

## The ratio picked a contrastive configuration that does nothing; MAP@100 picks one that works

Rescoring the 50m half of the HP sweep at checkpoint-133233 (job 8847624, 12
configurations x 4 seeds) on the task rather than the proxy inverts the winner. The two
metrics pick near-opposite configurations:

    ratio's pick    lr2e-5 / KL0  / t0.2     MAP@100 rank  9 of 12
    MAP@100's pick  lr1e-4 / KL10 / t0.07    ratio   rank  9 of 12
    rank agreement across the 12 configs: Spearman +0.25 (p=0.43)

                        ratio    Hit@1              MAP@100
    untrained base      1.34    0.6538             0.1727
    lr1e-4/KL10/t0.07   6.34    0.7464 +- 0.0065   0.3837 +- 0.0086
    lr2e-5/KL0 /t0.2    9.59    0.5107 +- 0.0157   0.1725 +- 0.0080

THE CONFIGURATION THIS PROJECT HAS BEEN USING DOES NOTHING. lr2e-5/KL0/t0.2 is
significantly WORSE than the untrained encoder on Hit@1 (t=-9.1, p=0.003) and exactly
level with it on MAP@100 (t=-0.03, p=0.98). It has the highest separation ratio of all
twelve.

A PROPERLY SELECTED CONFIGURATION WORKS, DECISIVELY. lr1e-4/KL10/t0.07 beats the
untrained encoder on Hit@1 (t=+14.2, p=0.0007) and more than doubles MAP@100 (t=+24.5,
p=0.0002). Head to head against the config we used: t=+13.9 and t=+18.0, both p<0.0001.
All four seeds cluster tightly -- Hit@1 0.737/0.737/0.739/0.765, MAP 0.364-0.406 -- so
this is not one lucky draw.

CORRECTION TO THE ENTRY ABOVE. "Contrastive FT costs Hit@1 at every scale, 0/24 arms
beat the untrained encoder" is true of the SCALE GRID, which ran entirely at
lr2e-5/KL0/t0.2. It is not a property of contrastive training. Read as written it is
misleading, and the error came from generalising a grid that varied only scale while
holding a bad configuration fixed. The proxy-failure finding itself is unaffected and is
in fact strengthened: the two rescore halves independently reproduce it on a grid that
varies hyperparameters instead of scale (Hit@1 rho +0.33 p=0.12 and +0.20 p=0.36).

WHAT THE AXES SAY: t0.07 beats t0.2 almost everywhere, KL10 helps at the top, and
lr5e-4 with KL0 diverges outright (ratio 1.6-1.8, at the floor; Hit@1 0.06).

CONSEQUENCE: every contrastive number in this project -- the scale curve, the
pretraining ablation, the pair-loss comparison, the negatives result -- was measured at
an operating point statistically indistinguishable from not training. Curve shapes may
survive, levels do not, and none of it should be quoted until re-taken at
lr1e-4/KL10/t0.07.

WHAT IS NOT YET MEASURED: whether lr1e-4/KL10/t0.07 is still the winner at other
checkpoints and scales. Job 8849088 runs the same twelve configurations at 50m
checkpoint-540423, so the transfer question is answerable on MAP@100 on both sides once
it lands. Nothing above has been re-measured at 100m/200m/400m.
    UPDATE 2026-09-23: it is, at 540423 and at all four scales. See "Closing the loop".

## Contrastive hyperparameters transfer across BOTH scale and pretraining checkpoint

Twelve configurations (lr x KL x temperature), four seeds each, scored on MAP@100 in
four cells: 50m at checkpoint-133233 and checkpoint-540423 (a 4x pretraining gap), 200m
at 192799, 400m at 181381. Every pairwise ranking agrees.

    50m @133233  vs 50m @540423    rho +0.902  p=0.0001
    200m@192799  vs 400m@181381    rho +0.853  p=0.0004
    50m @133233  vs 200m@192799    rho +0.811  p=0.0014
    50m @133233  vs 400m@181381    rho +0.783  p=0.0026
    50m @540423  vs 400m@181381    rho +0.755  p=0.0045
    50m @540423  vs 200m@192799    rho +0.699  p=0.0114
    mean pairwise rho +0.801, minimum +0.699, all significant at n=12 configs

lr1e-4 / KL10 / t0.07 is top-ranked in ALL FOUR cells (MAP@100 0.3837 / 0.4405 / 0.3409
/ 0.3541). The configuration this project used before, lr2e-5 / KL0 / t0.2, ranks 9th,
8th, 9th, 9th -- consistently bad everywhere, not a 50m artefact.

STATED AT THE STRENGTH IT HAS: the RANKING transfers; the WINNER's margin is resolved
only at 50m.

    50m @133233   0.3837 vs 0.2686 (lr5e4_kl10_t007)  t=+6.49  p=0.0006
    50m @540423   0.4405 vs 0.2650 (lr1e4_kl0_t007)   t=+4.35  p=0.0048
    400m@181381   0.3541 vs 0.3113 (lr2e5_kl10_t007)  t=+2.23  p=0.067
    200m@192799   0.3409 vs 0.2596 (lr5e4_kl10_t007)  t=+1.45  p=0.196

At 200m and 400m it is first but not separated from the runner-up at four seeds. What
corroborates the choice anyway is that every runner-up chasing it is also KL10/t0.07 --
the AXES that matter are consistent even where the exact learning rate is not resolved.

CONSEQUENCE: no per-rung and no per-scale HP sweep is needed for contrastive. The
checkpoint ladder can run one configuration at every cell, which is what job 8851663
does.

CAVEAT ON PROVENANCE: three of the four cells sit at each scale's ORIGINAL checkpoint
(133233/192799/181381), not on the canonical ladder. They were used because those arms
already existed and only needed rescoring, so the transfer test cost no new training.
Only 50m@540423 is a canonical rung.

WHAT IS NOT YET MEASURED: 100m. Its 48 arms predate the retrieval code; jobs 8851492
and 8851650 are rescoring them, which will make this four scales instead of three.
    UPDATE 2026-09-23: measured. See "Closing the loop" -- same winner at 100m.

## Report the metric-learning standards, under the names the literature uses

Contrastive was being scored on Hit@1 and MAP@100, neither of which is what the metric
learning field reports, so our numbers could not be compared against published work.

THE REFERENCE: Musgrave, Belongie and Lim, "A Metric Learning Reality Check", ECCV 2020,
arXiv:2003.08505. They re-ran a decade of metric-learning papers under matched training
and found most reported gains vanished, and they argue Recall@K is a poor metric --
it saturates and is blind to ranking quality below the cutoff. Their recommendation is
MAP@R, with R the number of relevant items FOR THAT QUERY, alongside Precision@1 and
R-Precision.

WHY IT MATTERS HERE SPECIFICALLY. The replicate corpus has 1000 peptide+charge groups,
NO singletons, and group sizes from 11 to 120 with a median of 13. A fixed cutoff asks a
much harder question of a 120-replicate peptide than of an 11-replicate one, and MAP@100
averages the two as equals. MAP@R adapts the cutoff per query, which is exactly the
variable-group-size case it was designed for.

WHAT WAS NOT WRONG: the worry that MAP@100 is unreachable because no peptide has 100
replicates. Exactly one group of 1000 exceeds 100 members (0.8% of spectra), so MAP@100
essentially never truncates and average precision can reach 1.0 for 999 of 1000 groups.
The problem with it is difficulty normalisation, not reachability.

NAMING, since these get used inconsistently:
  Precision@1  identical to the Hit@1 we already reported; both names are emitted so
               older tables still line up.
  R-Precision  of a query's R relevant spectra, the fraction inside its own top R.
  MAP@R        average precision over the top R, zero-padded past the last hit.
  R@5          OURS IS NOT THE LITERATURE'S Recall@K. Ours is the fraction of a query's
               relevant spectra appearing in its top 5; theirs is the fraction of
               QUERIES with at least one hit in the top K. Kept only for continuity.

The implementation is checked against a brute-force loop on random embeddings, matching
to 1e-6 on all three across four trials, and returns 1.0 on a constructed perfect case.

WHAT IS NOT YET MEASURED: every contrastive number in this project is quoted in MAP@100.
The saved encoders all carry final/, so re-scoring them in MAP@R is a rescore rather
than a retrain, but it has not been done.

    UPDATE 2026-09-23: the ladder (8851663) is backfilled in MAP@R; the rest is not.

## Closing the loop, 2026-09-23: today's contrastive results in one place

Everything below is scored on the task (MAP@R, Precision@1, R-Precision, and MAP@100
where the runs predate MAP@R), never on the separation ratio.

TEMPERATURE 0.03 BEATS 0.07 EVERYWHERE, BY A LOT (job 8856116, lr1e-4, KL10, 3 seeds):

    cell        t0.03    t0.07    gain
    50m@220k    0.4462   0.3087   +0.1375
    50m@330k    0.4101   0.3548   +0.0553
    200m@220k   0.3655   0.2552   +0.1103
    200m@330k   0.3391   0.2170   +0.1221      MAP@R

KL 100 is worse than KL 10 in every cell (0.15-0.29), so KL is bracketed at 10. 0.03 is
again the LOWEST temperature tried; job 8856399's successor on debug-scaling extends it
to 0.01. The optimum is still unlocated.

50m STILL BEATS 200m AT THE BETTER TEMPERATURE, so that gap is not a recipe artefact:
+0.0807 at 220k (p=0.028) and +0.0711 at 330k (p=0.003). But only 50m and 200m have
been run at t0.03 -- 100m and 400m have not, and at t0.07 400m sat ABOVE 200m, so
"retrieval gets worse with scale" is not established as a curve. PLAN.md C2.

PRETRAINING IS ALL OF IT (jobs 8853557/8854412 vs 8853558/8854760, t0.07, 6 seeds):
a random-init encoder trained the same way lands at chance at every scale --
MAP@100 0.010-0.012, Hit@1 0.027-0.036 -- against 0.34-0.41 and 0.75-0.78 pretrained,
t=16-96. It never reaches even the UNTRAINED pretrained encoder (0.17 / 0.65). Because
random = chance, the pretraining gain is simply the pretrained score. PLAN.md C3.

HYPERPARAMETER RANKING TRANSFERS ACROSS ALL FOUR SCALES AND TWO CHECKPOINTS. The 100m
cell (rescored by 8851492/8851650) completes it: lr1e-4/KL10/t0.07 wins in all five
cells (50m@133k, 100m@138k, 200m@192k, 400m@181k, 50m@540k); 100m ranks with the others
at rho +0.91/+0.78/+0.74/+0.85; mean pairwise rho over the five cells +0.81, min +0.70.

BEST-MODEL SELECTION DOES NOT MATTER (job 8856159, eval on, select on MAP@R, 6 seeds vs
the identical ladder cells). In training, MAP@R does peak mid-run (step 800 of 1347,
0.3783 vs 0.3714 at the end, on the 800-row selection set), but on held-out rows the
change is -0.0021 (p=0.92) at 50m@330k and -0.0056 (p=0.53) at 200m@330k: selection
overfits a small eval set. Job 8856348 agrees from the other side -- final/ equals
checkpoint-1347 exactly and checkpoint-1200 is ~0.003 lower on three seeds.

GRADCACHE IS EXACT END TO END (job 8856115, 50m@330k, 6 seeds a side): chunk 2 vs off,
MAP@R 0.3441 vs 0.3371, t=+0.46, p=0.65; the off arms reproduce the ladder (0.3318).
Together with 19 unit tests this licenses sweeping P/K, which needs GradCache above a
batch of four.

"MORE NEGATIVES HURT" IS RETRACTED. It came from sweep-gradcache-v2 (8848463), decided
on the separation ratio at lr2e-5/KL0/t0.2. Rescored on the task (8854486), 64 vs 4
negatives is MAP@100 0.1681 vs 0.1707, t=-0.21, p=0.84 -- the ordering collapses rather
than inverts, and all three widths sit at the untrained level, as that recipe does. The
question is open, not answered the other way; job 8856399's successor sweeps it.

THE CONTRASTIVE TRAINING OBJECTIVE IS SOLVED ALMOST AT ONCE: on 50m@330k the loss falls
below 10% of chance (ln 4) by epoch 0.18 of 3.0, and at a batch of four each logged
value is one noisy draw (last step 0.142, epoch 2.90 reached 0.0016). A trivially easy
training task is itself a reason to expect wider batches to matter.

NOT YET MEASURED: the ep30/ep100 encoders on MAP@R; t below 0.03; P/K at the current
recipe; 100m and 400m at t0.03.

## More negatives help a lot; the four-spectrum batch was holding contrastive back

sweep-conbig (job 8856460, 107/108 arms), all scales at checkpoint-220000, lr1e-4,
KL10, GradCache chunk 4, 3 seeds per cell, MAP@R:

    t0.01        P*K 4    P*K 16   P*K 64
    50m          0.5116   0.6642   0.7183
    100m         0.5391   0.6978   0.7639
    200m         0.4478   0.6556   0.7197
    400m         0.5859   0.7082   0.7653

WIDER BATCHES WIN EVERYWHERE: 4 -> 64 adds 0.19-0.27 MAP@R at every scale and at every
temperature tried. "More negatives hurt" was not just unsupported, it was backwards;
every contrastive run before this trained at P*K 4.

LOWER TEMPERATURE WINS EVERYWHERE: t0.01 > 0.02 > 0.03 in all 12 (scale, width)
columns. Best MAP@R goes from 0.45 (t0.03, width 4) to 0.77.

BOTH AXES ARE AT THEIR EDGE AGAIN (64 is the widest, 0.01 the coldest), so the optimum
is still unlocated. sweep-conneg extends them: t {0.003, 0.005, 0.01} x width
{64, 128, 256, 512}, plus step-matched controls at 50m, since wider batches at a fixed
3 epochs take fewer optimizer steps (~660 at 64, ~80 at 512).

THE SCALE PICTURE CHANGES. At the best setting 400m (0.7653) and 100m (0.7639) beat
50m (0.7183); 200m (0.7197) is level with 50m. The earlier "retrieval gets worse with
scale" was measured at width 4 and does not survive. 200m being the odd one out at
220k matches earlier signs that the 200m run at 220k is weak (it also DECLINED from
220k to 330k in the ladder) -- to be checked at other checkpoints, not assumed.

STATED AT THE STRENGTH IT HAS: 3 seeds per cell; seed sems at this recipe have run
0.004-0.02, an order of magnitude below the width and temperature effects. The scale
ordering at the best cell (gaps of ~0.001-0.05) needs the next grid's seeds before it
is quoted as a curve.
