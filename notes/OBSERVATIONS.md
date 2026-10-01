# Observations

Things we believe and why, separate from `STATUS.md` (what is running) and `TODO.md`
(what is broken). Each entry says what would overturn it.

> **Paths (2026-09-26 reorg):** entries below keep the paths as they were when written. Since then
> `results/finetune` -> `results/raw/finetune`, `results/rerank` -> `results/raw/rerank`,
> `results/figures` -> `results/processed/figures`, and the top-level `results/*.csv` and
> `results/SUMMARY_TABLES.md` -> `results/processed/tables/`. See `results/README.md`.

> Since 2026-09-27 the 'peptide embedder' is called the peptide encoder (and the contrastive model the spectrum encoder); entries below keep the names they were written with.

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

## Temperature settles near 0.005; wider than 64 loses at fixed epochs; retrieval rises with scale

sweep-conneg, debug-scaling half (job 8856643, 108/108 arms): 50m/100m/200m at
checkpoint-220000, lr1e-4, KL10, K=4, 3 epochs, 3 seeds, MAP@R.

    best cell per scale       width 64
    50m    t0.005   0.7508     (t0.003 0.7482, t0.01 0.7186)
    100m   t0.005   0.7825     (t0.003 0.7731, t0.01 0.7636)
    200m   t0.003   0.7878     (t0.005 0.7697, t0.01 0.7200)

The t0.01 / width-64 anchor reproduces sweep-conbig to 0.0003 at every scale.

TEMPERATURE: interior at 50m and 100m (0.005 >= 0.003 > 0.01), but at 200m 0.003 is
best and is the coldest value tried, so the optimum may move colder with scale.
400m (capacity half, 8856642) decides whether that pattern holds.

WIDTH: beyond 64, MAP@R falls steeply at every scale and temperature, e.g. 50m t0.005
0.751 / 0.697 / 0.602 / 0.440 at 64 / 128 / 256 / 512. AMBIGUOUS AS IT STANDS: at a
fixed 3 epochs width 512 takes ~80 optimizer steps against ~660 at 64. The
step-matched controls (256 x 12 epochs, 512 x 24 epochs, 50m) in 8856642 separate
"fewer steps" from "more negatives stop helping". Do not quote "64 is optimal" before
they land.

SCALE: at each scale's best cell, 50m 0.751 < 100m 0.783 < 200m 0.788 -- the first
monotone contrastive scaling seen in this project. 200m's apparent weakness at t0.01
was a temperature effect: it needs a colder setting than the small models.

## D3: denoise improves with pretraining at every scale except 400m, which is flat

Job 8850494 (21/21), plus each scale's original checkpoint, 3 seeds per new cell,
lr2e-4 / es0.5 / 4 epochs:

                  original       220k           330k
    50m (133k)    0.9324/0.8635  0.9343/0.8661  0.9359/0.8677   (540k 0.9365/0.8680)
    100m (138k)   0.9395/0.8712  0.9418/0.8745  0.9426/0.8759
    200m (193k)   0.9437/0.8772  0.9447/0.8784  0.9458/0.8802
    400m (181k)   0.9433/0.8765  0.9435/0.8773  --
                  AUROC/F1

50m, 100m and 200m each gain +0.001-0.002 AUROC per extra stretch of pretraining, with
diminishing returns (50m: +0.0020, +0.0015, then +0.0006 over the last 210k steps), and
the larger model is ahead at every checkpoint. 400m does NOT move from 181k to 220k
(+0.0002, inside seed noise) and stays below 200m, so the 400m turnover of D1 is not an
artefact of its early checkpoint. Two nearby 400m points only: 330k does not exist yet.

## C1: wider batches win at matched steps; 400m wants colder still

sweep-conneg capacity half (8857336, 39/42 at the time of writing).

STEP-MATCHED CONTROL, 50m, t0.01: width 64 at 3 epochs 0.719; width 256 at 3 epochs
0.510; width 256 at 12 epochs (same ~660 optimizer steps as 64 at 3) 0.813. Wide batches
lost only because they took a quarter of the steps. But the 0.813 arm also saw 4x the
data -- its equal-COMPUTE comparison, width 64 at 12 epochs, has not been run. That is
what sweep-conlong settles.

400m at width 64, 3 epochs: t0.003 0.8226 > t0.005 0.8079 > t0.01 0.7660. With 200m
(best at t0.003) this makes two scales whose optimum is at the coldest value tried:
colder with scale. Best per scale at width 64 / 3 epochs:
50m 0.751 < 100m 0.783 < 200m 0.788 < 400m 0.823 -- monotone.

sweep-conlong (make_conlong.py, 117 arms) crosses width {64, 256, 512} with epochs
{3, 12, 24} at t {0.001, 0.002, 0.003}: full grid at 50m, widths {64, 256} x epochs
{3, 12} at 400m.

## D2: pretraining is worth ~+0.05 AUROC at every scale

Scratch grid 8847663 (all 12 arms, the last 5 via the 8856558 resume), 4 epochs, 3 seeds,
against each scale's pretrained arms at its original checkpoint (FT5, 6 seeds):

              pretrained   from scratch   gain
    50m       0.9317       0.8856         +0.046
    100m      0.9400       0.8886         +0.051
    200m      0.9447       0.8963         +0.048
    400m      0.9434       0.8981         +0.045

The gain is flat across scale. Scratch improves with size by about as much as pretrained
does (+0.013 from 50m to 400m).

CORRECTION (2026-09-25): the 50m scratch cell above is NOT a 3-seed mean at the shared config. It
is the best of the 50m scratch HP grid (8841984, arm lr5e4_ep4_b48, one run). At the config the
100m-400m scratch arms use (lr 2e-4, eff. batch 12, 4 epochs) 50m scratch is 0.8821 (one run),
so the matched gain at 50m is +0.050. Scratch per-seed: 100m 0.8907/0.8899/0.8854, 200m
0.8967/0.8961/0.8959, 400m 0.8956/0.8994/0.8993. The conclusion (~+0.05 at every scale) holds. Extra fine-tuning does not close it: 50m from scratch at
16 epochs reaches 0.9051, still 0.027 below pretrained 50m at 4.

## D3: denoise saturates by ~330k-430k pretraining steps; even 10k steps is most of the gain

Ends wave 8856549 (24/24) completes 6-point curves at the two scales whose pretraining
has finished (AUROC; F1 tracks it):

              10k      120k     220k     330k     430k     540k
    50m       0.9104   0.9312   0.9343   0.9359   0.9362   0.9361
    100m      0.9193   0.9394   0.9418   0.9426   0.9430   0.9429

Most of the benefit arrives early: 2% of pretraining (10k steps) already beats training
from scratch by ~0.025, and the curve is flat from ~330k on. The larger model is ahead at
every checkpoint.

## C1: training LENGTH was the main lever; wide batches help only when trained long

sweep-conlong (117/117), 220k, K=4, 3 seeds, MAP@R. 50m, t 0.002:

    width \ epochs     3       12      24
    64                 0.734   0.860   0.864
    256                0.605   0.846   0.877
    512                0.402   0.799   0.870

3 -> 12 epochs is worth ~+0.11; nothing else comes close. At EQUAL compute, width 64
edges width 256 at 12 epochs (0.860 vs 0.846) and loses at 24 (0.864 vs 0.877), so the
sweep-conneg control's jump (width 256 at 12 epochs, 0.813 vs 0.719) was mostly longer
training, not more negatives. Temperature stops mattering once trained long: 0.001-0.003
lie within ~0.01 at 12-24 epochs (at 3 epochs colder still helps, 0.749 vs 0.687 at
width 64). 512 at 24 epochs, t0.01 (control): 0.836.

400m (widths 64/256 x 3/12 epochs): best 0.877 (width 256, 12 epochs, t0.001); 0.873 at
width 64, 12 epochs. SCALE AT THE LONGER RECIPE: 50m 0.860 vs 400m 0.873 at width 64 /
12 epochs, a gap of 0.013 where it was 0.08 at 3 epochs. Much of the earlier scale gap
was an undertraining effect that longer training closes.

Working recipe: SupCon, lr1e-4, KL10, t ~0.002, width 256 (P=64, K=4), 24 epochs,
GradCache chunk 4: MAP@R ~0.877 at 50m.

## R1: the embedding costs ~0.11 Hit@1, but the benchmark cannot yet say why

Current teachers (50m t0.005, 400m t0.003, width 64, 220k), 5 paired seeds, alignment
config otherwise identical to the original run:

                                     50m       400m
    student cross-modal Hit@1        0.730     --       (old teacher: 0.045)
    rescorer, no embedding           0.893     0.893
    rescorer, with embedding         0.782     0.780
    embedding contributes            -0.111    -0.112
    no near-miss decoys, both arms   0.999     0.999

The student is 16x better at cross-modal retrieval than the one behind the original
-0.109, yet the cost is unchanged, so encoder quality is not the cause.

WITHIN-SPECTRUM DIAGNOSIS (--diagnose, 50m): all cosines sit at 0.97-0.98. The truth
beats mass-matched decoys on cosine 93.5% of the time and reversed ones 86.7%, but
near-miss decoys (two adjacent residues swapped) only 57.0%, a coin flip. Near-misses
win 581 of the 740 spectra where cosine picks wrong. A mean+max-pooled sequence
encoder is nearly blind to swapping neighbours; fragment ions are not, so the feature
adds confident noise on exactly the cases the other features solve.

THREE THINGS MAKE THIS BENCHMARK UNFIT, all found today:
  1. Precursor derived from the true peptide (FT28). Real, but INCONSEQUENTIAL: with
     the measured precursor (truth mass error median 1.7 ppm) every number above
     reproduces within noise.
  2. "Mass-matched" decoys are not mass-matched (FT30). They are the ~4 nearest
     peptides by mass among only 94 held-out peptides, far outside any search
     tolerance, so mass error rejects them for free (0.999). The only hard decoys left
     are near-misses, which the embedding cannot see.
  3. The rescorer split by spectrum (FT29): 100% of its test spectra had their peptide
     in its training data (audit 8859654). Now fixed to split by peptide.
The -0.11 is therefore a statement about adjacent-swap decoys under a leaky split, not
about realistic reranking.

## Split audit: denoise and contrastive are peptide-disjoint; only the rescorer leaked

Audit 8859654 (sweeps/audit_splits.py). ms-denoise-100k's own splits share ZERO
peptides (and zero peptide+charge) between train and validation or test. The
contrastive/alignment split (10% of peptides, seed 0) shares zero, as designed. The
rescorer's spectrum-level re-split shared 90 of 90 test peptides. Not checkable here:
overlap between test sets and the pretraining corpus, which needs its manifest.

## C7 (first read): on ms-contrastive-100k, an unlearned binned cosine beats every trained encoder (2026-09-23)

Test split, 9,950 analytes, replicate-corpus peptides excluded, 25,137 experimental-spectrum
queries (R=2); jobs 8860041/8860206/8860250, `results/finetune/contrastive/grouped100k-test/`.

    experimental MAP@R (Hit@1)                  3 seeds unless noted
    binned cosine, 0.1 Da bins (no learning)    0.730 (0.819)
    binned cosine, 1.0005 Da bins               0.671 (0.771)
    400m C1 recipe, 12 epochs                   0.713 (0.809)   0.711-0.714
    50m  C1 recipe, 24 epochs                   0.656 (0.766)   0.655-0.657
    400m 3 epochs t0.003                        0.580 (0.709)
    50m / 100m / 200m 3 epochs t0.003           0.40 / 0.32 / 0.35
    untrained pretrained encoder (220k)         0.08-0.16 (50m 0.112, 400m 0.165)

With the consensus spectrum in the gallery ("all", R=3) the gap is large: binned 0.759 vs
best trained 0.510. The learned space does not place a consensus spectrum near its
replicates; raw m/z bins do.

READING. (1) The 99-group replicate eval hid both the scale effect (400m 0.713 vs 50m
0.656 here; 0.873 vs 0.877 there) and the fact that the task is easy for exact-m/z
matching. (2) Every learned encoder so far is BELOW a zero-parameter baseline. A likely
cause, not yet tested: peak tokens carry no absolute m/z (iona-base model card: tokens are
built from normalised log intensity only; m/z enters as pairwise deltas), so the
embedding cannot use the one signal binned cosine lives on. (3) Contrastive training
still matters enormously (0.11 -> 0.66 at 50m), and longer training still helps.

NOT YET MEASURED: models trained on this corpus (C7, 8860292); precursor-window
filtering (what real library search does, for both methods); GLEAMS.

## C9 (quick): a projection head halves retrieval at 3 epochs (2026-09-24)

50m @220k, frozen C1 recipe at 3 epochs, 3 seeds; head = master's shape (pooled -> 512 ->
256, dropout 0.1), randomly initialised. Control = sweep-conlong s050m_t0002_pk256_ep03
(identical apart from the head). Jobs 8860538 (train), 8860569 (100k test).

                          small eval MAP@R    100k test exp MAP@R
    no head (control)     0.604               0.382
    head, pre-head        0.292               0.184
    head, head output     0.300               0.155

Starting loss 34.5 with the head vs 16.6 without (random head outputs at t 0.002), final
5.09 vs 3.58: the head mostly costs training time at this budget. Pre-head edges out the
head output on the larger test (SimCLR's direction), but both are far below no head.
NOT YET MEASURED: 24 epochs (8860587, deprioritised behind the denoise ladder).

## C7: a quarter epoch of ms-contrastive-100k puts 50m ABOVE binned cosine (2026-09-24)

sweep-con100k-best, cont050m (continue sweep-s050m_t0002_pk256_ep24 on ms-contrastive-100k,
same recipe, P85 x K3, experimental spectra, replicate-corpus peptides excluded), encoder at
step 300 of 1062 (~28% of one epoch), scored on the test split (job 8860719):

                                         exp MAP@R            exp Hit@1     all MAP@R
    cont050m step 300, seeds 0/1/2       0.808/0.799/0.811    0.871/0.867/0.873   0.65
    binned cosine 0.1 Da                 0.730                0.819         0.759
    400m, replicate corpus only          0.713                0.809         0.510
    50m starting point (replicate only)  0.656                0.766         0.462

+0.15 over its own start and +0.08 over binned cosine, seed spread 0.006. The replicate
corpus (~855 peptides) was the bottleneck. With the consensus spectrum in the gallery
("all") binned cosine still leads, 0.76 vs 0.65 (gap was 0.25); training never sees
consensus spectra (include_consensus false). NOT YET MEASURED: later steps, 400m.

## C2 on the ms-contrastive-100k test: flat 50m-200m, a step at 400m (2026-09-24)

Frozen C1 recipe (24 epochs, t 0.002, P64xK4, GradCache 4) at checkpoint 220k, 3 seeds,
trained on the replicate corpus; scored on the 100k test split (job 8860863):

              exp MAP@R (seeds)          mean    Hit@1   small eval
    50m       0.659 / 0.655 / 0.657      0.657   0.766   0.877
    100m      0.659 / 0.678 / 0.658      0.665   0.768   0.885
    200m      0.661 / 0.669 / 0.643      0.658   0.764   0.881
    400m      0.704 / 0.705 / 0.700      0.703   0.799   0.867

The small eval ranks 400m LAST; the 100k test ranks it first by +0.045. Scale reads from
the 100k test only. 400m at 24 epochs (0.703) sits just below 400m at 12 epochs (0.713,
C1), so 24 epochs is at or past 400m's optimum while 50m still gained 12 -> 24. All
replicate-corpus models remain below binned cosine (0.730).

## C9 settled: no projection head (2026-09-24)

24-epoch arms (job 8860587), frozen C1 recipe + master's head (pooled -> 512 -> 256), 50m
@220k, 3 seeds, small-eval MAP@R; control = C2 s050m_ck220k (identical apart from the head):

    no head (control)      0.878 / 0.879 / 0.874   mean 0.877
    head, head output      0.829 / 0.828 / 0.817   mean 0.825
    head, pre-head         0.713 / 0.699 / 0.704   mean 0.705

The head closes part of its 3-epoch deficit but still loses 0.05 at our full budget, and
the pre-head features (SimCLR's readout) lose 0.17. DECISION: keep the no-head design
(normalised mean+max of the last layer). Nothing is rerun. 100k-test scoring of these arms
is in 8861096 for the record.

## A1: a peptide encoder trained on ms-contrastive-100k reaches Hit@1 0.93 (2026-09-24)

Teacher: C7 50m continued, step 600, seed 1 (0.831 exp MAP@R on the 100k test). Cache built
sharded on 12 tiles (8861039 + 8861075); student = configs/a1-align-100k-050m-c7s600
(PeptideEncoder 4x256, L2 to the teacher's normalised embedding, batch 64, 3 epochs,
~12k steps, 257 s), job 8861093. In-training cross-modal check on 2,000 validation spectra
against 747 candidate peptides:

    Hit@1 0.932   Hit@5 0.962   MRR 0.945
    (R1's student, replicate corpus: Hit@1 0.73 on 94 held-out peptides)

NOT YET MEASURED: the 10k-analyte test split; seeds; a 400m teacher. C9's 100k-test scores
for the record: head arms 0.534 / 0.550 / 0.551 exp MAP@R vs 0.657 no-head (C2).
C7 50m keeps rising: step 300 0.81, 600 0.83, 900 0.84.

A1 ON THE TEST SPLIT (8861310, msdelta.eval_align_test): every experimental test spectrum
(25,137) against every test analyte (9,771 peptide+charge candidates), 3 student seeds:

    Hit@1 0.899 / 0.898 / 0.898   Hit@5 0.932   MRR 0.914
    (teacher's own spectrum->spectrum MAP@R on these rows: 0.8311, identical to its
    eval_grouped_retrieval score -- the two pipelines agree)

Sequence alone, no precursor filter, ~10k candidates: the right peptide ranks first for
~90% of spectra.

## CORRECTION: small-eval -> 100k-test transfer is rho 0.78, not 0.98 (2026-09-24)

With all 69 Stage-1 models scored (sweeps/plot_contrastive_100k.py, c100k_transfer.png),
Spearman between the replicate-corpus eval MAP@R and the 100k-test exp MAP@R is 0.78. The
0.98 quoted earlier came from the first 27 models, which varied mainly in recipe and
training length. The two groups that break it are the axes this study is about:
400m models sit ABOVE the trend (the small eval ranked 400m last in C2) and early
pretraining checkpoints (C4 10k) sit BELOW it. Recipe/length effects transfer; scale and
checkpoint effects must be read on the 100k test. PLAN Stage 2 note corrected.

C7 400m continued (8860522), 100k test, 3 seeds (job 8862067 for step 600):

    step 300    exp MAP@R 0.839   Hit@1 0.894   all 0.69
    step 600    exp MAP@R 0.859   Hit@1 0.908   all 0.726   (0.860 / 0.859 / 0.856)
    50m final   exp MAP@R 0.839   Hit@1 0.893   all 0.677
    binned 0.1  exp MAP@R 0.730   Hit@1 0.819   all 0.759

On the large corpus 400m pulls ahead of 50m (+0.02 at 600 of 1,062 steps, still rising),
and the consensus-view gap to binned cosine is nearly closed (0.726 vs 0.759).

## Zero-shot redone: frozen encoders DO retrieve, best at ~3/4 depth (2026-09-24)

msdelta.eval_zeroshot_layers (job 8862567): frozen pretrained encoders, mean+max pooling
of every block, ms-contrastive-100k test, experimental MAP@R. `final` at 220k reproduces
the earlier zero-shot numbers exactly (0.112 / 0.093 / 0.077 / 0.165).

    encoder      final   best block (of N)   best
    50m  @10k    0.072   8/10                0.086
    50m  @220k   0.112   8/10                0.208
    50m  @540k   0.102   8/10                0.177
    100m @220k   0.093   10/13               0.140
    100m @540k   0.087   10/13               0.125
    200m @220k   0.077   13/16               0.151
    200m @540k   0.080   13/16               0.191
    400m @10k    0.168   16/20               0.216
    400m @220k   0.164   16/20               0.432
    400m @430k   0.125   16/20               0.293

Same profile everywhere: 0 at the input embedding (intensity only, no m/z), rising to a
peak at ~75-80% depth, falling to the output -- the last blocks specialise for masked-
intensity prediction. 400m@220k at block 16 is 2.6x its own final layer. Longer
pretraining lowers the zero-shot peak at 50m and 400m (220k > 540k/430k), raises it at
200m.

OVERTURNS the "frozen embedding is indistinguishable from random at every layer" claim
(it was measured on the separation ratio, which C0 invalidated). Fine-tuning still adds a
lot (400m: 0.43 frozen best -> 0.70 replicate-FT -> 0.86 C7). NOT YET MEASURED: linear vs
MLP probe on the frozen ~3/4-depth features (the actual non-linearity test); a random-init
zero-shot reference on this test. Raises the value of the parked layer-mix retry.

## Reranking with the best 50m pair: the embedding alone beats real decoys 98-99%, near-miss swaps 70% (2026-09-24)

Spectrum side C7-50m (step 600, seed 1), peptide side A1 student (seed 0); 25,775
ms-contrastive-100k VALIDATION spectra (cache align-targets-a1-050m-c7s600); candidates =
truth + 2 near-miss (adjacent swap) + 2 mass-matched real peptides + reverse. Jobs 8862663
(embedding only), 8862729 (classifier, all decoys), 8862705 (classifier, no near-miss).

    embedding only (argmax cosine):  hit@1 0.701 all decoys | 0.975 without near-miss
      truth beats: mass_matched 98.7%, reverse 98.2%, near_miss 70.5%
    classifier, all decoys:          with emb 0.752 +- 0.008, without 0.789 +- 0.008
                                     embedding -0.037 hit@1 (R1: -0.11), +0.009 AUROC
    classifier, no near-miss:        with 0.996, without 0.995 (+0.001, n.s.)

The only thing the embedding cannot separate is adjacent-residue swaps: mean+max pooling
is near order-blind (A3). Without near-misses the hand-built features already solve this
synthetic benchmark, so it cannot show what the embedding adds -- that needs the incoming
reranking dataset's real candidates. Caveats FT30 (no ppm cutoff on mass-matched decoys)
and FT32 (distorted intensity features) apply to both classifier arms equally.

## PCA baseline: linear compression of binned spectra ~ binned cosine, well below ours (2026-09-24)

PCA fitted on 10k ms-contrastive-100k TRAIN analytes (35,731 spectra), test projected,
cosine retrieval (job 8863968), experimental MAP@R / Hit@1 / all-view MAP@R:

    PCA 0.1 Da -> 1280 dims   0.723 / 0.814 / 0.755    (binned cosine 0.1 Da: 0.730 / 0.819 / 0.759)
    PCA 1 Da   -> 256 dims    0.553 / 0.672 / 0.595
    PCA 0.1 Da -> 256 dims    0.488 / 0.611 / 0.533
    ours: C7 400m step 600    0.859 / 0.908 / 0.726;  C7 50m final 0.839 / 0.893 / 0.677

At our width, linear unsupervised compression loses almost nothing vs raw binned cosine;
the learned embeddings are +0.11-0.14 above both on experimental spectra (weaker only on
the consensus view, which training never sees).

## R baselines: MS²Rescore (full) beats our rescorer by +33% PSMs (2026-09-24)

Same 8 runs, 332,822 spectra, all top-10 candidates, OUR TDC (best per spectrum, runs
pooled, PSMs at 1% FDR). Built by an agent under baselines_wip/ (untracked); outputs in
$SCRATCH/baselines/{ms2rescore_out,okt_out}.

    method                                             all 8     HEK (6)   HCT116 (2)
    MSFragger rank-1                                   89,693    88,543     1,518
    ours MLP ms+hand                                   96,288    93,791     2,613
    ours MLP ms+hand+emb                               96,367    94,127     2,618
    MS²Rescore search-only (Percolator-style)         100,974    97,722     2,950
    MS²Rescore full (+MS²PIP +DeepLC)                 128,211   120,747     7,139
    Oktoberfest original (Percolator, search feats)       --    87,128        --
    Oktoberfest + Prosit                                  --   102,810        --

- MS²Rescore trains per run (Percolator-style internal 3-fold CV, ristretto engine, no
  mokapot in 4.0.2); ours is CV across runs. Even its search-only arm beats ours, so part
  of the gap is the training protocol, not features. The rest is predicted-spectrum
  (MS²PIP) and RT (DeepLC) features, which we do not have.
- MS²PIP: CID for HEK (ion trap), HCD2021 for HCT116 (CID there: 6,317 < 7,139).
- Oktoberfest: HEK only; N-term acetyl candidates dropped (~3.6%) and peptides > 30 aa,
  counted as not accepted (biases against it). Prosit calls went to koina.wilhelmlab.org.
- Leakage spot check (HEK-0628-5): decoy labels preserved; 172 decoys among 17,527 accepted.
- Consequence for R: the fair comparison is our embedding ADDED to a strong rescorer
  (MS²Rescore features, per-run training), not our MLP vs theirs.

## Teacher selection moved to VALIDATION; A1's teacher seed confirmed (2026-09-24)

All 12 C7-50m checkpoints re-scored on ms-contrastive-100k VALIDATION (experimental MAP@R;
25,057 queries). Test for comparison, never for selection from here on.

    step    seed0 val/test    seed1 val/test    seed2 val/test
    300     0.804 / --        0.795 / --        0.807 / --
    600     0.821 / 0.827     0.827 / 0.831     0.823 / 0.829
    900     0.830 / 0.838     0.835 / 0.837     0.832 / 0.839
    final   0.831 / 0.838     0.836 / 0.839     0.834 / 0.840

- At step 600 (the A1 teacher) validation picks seed 1, as test did: A1 stands unchanged.
- The best 50m teacher on validation is FINAL seed 1 (test would have picked final seed 2).
  Step 600 was used for A1 because it was the newest checkpoint at the time.
- A2 (400m) now picks its seed on validation (scratchpad auto_a2_val.sh); test is scored
  afterwards for the record only.

## R4 A/B: the lab's 254 features lift our rescorer to 104.8k; the A1 embedding adds ~1% (2026-09-25)

Global model (CV by run, fixed labels, decoys capped at 1M per fold), same 8 runs, PSMs at
1% FDR (job 8865505, results/rerank/psm/a1-050m-c7s600_r4_global.json):

    features                  linear     mlp
    ms (MSFragger)            91,780    93,206
    ms + emb                  92,345    93,787
    lab (254 non-empty)       99,512   104,785
    lab + emb                 99,206   105,755   (+970, +0.9%)
    lab + embws               99,611   105,179
    refs: MSFragger 89,693; old best mlp ms+hand+emb 96,367; MS2Rescore search-only
    100,974; MS2Rescore full (MS2PIP + DeepLC) 128,211

- lab features: +8.5k PSMs over our 22 hand features; the MLP beats MS2Rescore's
  search-only arm, still 23k below its predicted-spectrum arm.
- A1 embedding (50m teacher): +0.9% with the MLP, ~0 with the linear model; seed noise not
  yet measured. Within-spectrum versions add nothing over the raw cosine here.
- Embedding alone: 0 PSMs even ranked by lead (embedding:delta). Leakage AUROC 0.52.

## R4 C/D: per-run Percolator-style on the lab features = 129k, level with MS2Rescore full (2026-09-25)

Per run, 3-fold by spectrum, 10 label-refinement rounds, linear, mokapot calibration;
same 8 runs (job 8865619, results/rerank/psm/a1-050m-c7s600_r4_perrun.json):

    features            per-run linear    (global MLP)
    ms                  99,154            93,206
    ms + emb            100,490 (+1,336)  93,787
    ms + embws          100,614 (+1,460)  93,843
    lab                 129,041           104,785
    lab + emb           129,245 (+204)    105,755
    lab + embws         129,345 (+304)    105,179
    MS2Rescore: search-only 100,974; full (MS2PIP + DeepLC) 128,211

- Our per-run ms arm (99.2k) ~ MS2Rescore search-only (101.0k): the protocol reproduces.
- lab + per-run matches/exceeds MS2Rescore full without predicted-spectrum or RT features.
- The A1 embedding helps a weak base (+1.3-1.5%) and fades on a strong one (+0.2%).
- NOT YET VERIFIED: shuffled-label control (must collapse) and fold-seed noise.
- CONTROLS (job 8865754): shuffled training labels -> 0 PSMs (no leakage / FDR bug).
  Fold seeds 0/1/2: lab 129,041 / 128,991 / 129,040 (+-30); lab+embws 129,345 / 129,688 /
  129,490 -> embedding +304 / +697 / +450, mean +484 (+0.37%), positive on every seed and
  ~15x the seed spread.

## MS2Rescore full + our embedding columns: +0.24% overall, +4.1% on HCT116, flat on HEK (2026-09-25)

Same 8 runs, same MS2Rescore config (MS2PIP + DeepLC + search features, per-run Percolator-
style), extra rescoring:emb_* columns only (job 8865654; scored with our TDC, pooled):

    arm                        all        HEK293     HCT116
    MS2Rescore full            128,211    120,747    7,139
    + emb_cos                  128,431    120,716    7,324
    + emb_cos + within-spec    128,521    120,643    7,433   (+310 / -104 / +294 = +4.1%)

- Gain concentrated on HCT116 (the harder data: low ID rate, MS2PIP HCD model fits worse);
  HEK flat. Only 2 HCT116 runs here; MS2Rescore's own CV noise not measured yet.
- Consistent with our per-run rescorer (+0.37% on lab features): the A1 embedding adds a
  small, positive signal on top of strong rescoring. Full dataset (R3) will test HCT116.

## All-but-the-top (Mu & Viswanath 2018) roughly doubles zero-shot retrieval (2026-09-25)

Frozen 50m@220k, mean+max pooled, ms-contrastive-100k test (experimental MAP@R); mean and
top-D principal directions fitted on 25,859 TRAIN experimental spectra, removed, then
cosine (job 8865736, diag/abtt_smoke_out):

    layer     raw     D1      D4      D8      D16     D32
    block08   0.208   0.244   0.330   0.359   0.380   0.395
    final     0.112   0.142   0.205   0.241   0.277   0.296

Raw reproduces the earlier zero-shot run exactly. Still rising at D=32 -> full run over all
10 frozen encoders with D in {8, 32, 64, 128} queued. The pretrained space is strongly
anisotropic: a few common directions dominate cosine.

## A vs yHydra, cross-modal spectrum -> peptide: A1 Hit@1 0.90 vs yHydra 0.20 (2026-09-25)

ms-contrastive-100k test, the 22,869 experimental spectra whose peptide yHydra can represent
(88.5%; no mods beyond fixed CAM, length 7-42); candidates = the representable test
peptides (yHydra 7,848 sequences; ours 8,623 peptide+charge). Job 8865921,
$SCRATCH/baselines/yhydra/xmodal/.

    model                 Hit@1   Hit@5   MRR
    A1 (3 seeds)          0.901   0.933   0.916   (full set, 9,771 cands: 0.898)
    yHydra, L2 (native)   0.196   0.377   0.284
    yHydra, cosine        0.176   0.360   0.266

Caveats: in-distribution for A1 (trained on this dataset's train split), zero-shot for
yHydra; yHydra is used natively with a precursor-mass pre-filter (K=50), not open retrieval.
Pending: precursor-mass-window variant (both models) and an OOD cross-modal test on the
HEK/HCT116 confident PSMs (C11 export). yHydra >> chance (1/7,848), so its input encoding
(pyteomics alphabet order) is right.

## ABTT across frozen encoders: ~2x everywhere; best D 64-128 (edge of range for 200m/400m) (2026-09-25)

Exp MAP@R, ms-contrastive-100k test, train-fit (test-fit within 0.005 everywhere: no
transductive gain). Job 8865987 (8 of 10 encoders; 400m@220k/430k rerun on capacity):

    encoder       raw final  raw best   ABTT final  ABTT best (D)
    50m@10k       0.072      0.086      0.181       0.181 (64)
    50m@220k      0.112      0.208      0.296       0.401 (64)
    50m@540k      0.102      0.177      0.260       0.361 (64)
    100m@220k     0.093      0.140      0.259       0.328 (64)
    100m@540k     0.087      0.125      0.241       0.306 (64)
    200m@220k     0.077      0.151      0.230       0.366 (128)
    200m@540k     0.080      0.191      0.242       0.432 (128)
    400m@10k      0.168      0.216      0.385       0.414 (128)

Best block stays at ~3/4 depth. Still far below binned cosine 0.730 and trained C7 0.868.
D=128 is the top of the swept range for 200m/400m -> optimum unlocated (add D=256).
- Precursor-mass windows (yHydra's native setting; job 8866320), same 22,869 queries:
    window    median cands   yHydra (L2)   A1 (3 seeds)
    open      ~8k            0.196         0.901
    +-1.1 Da  14-15          0.754         0.972-0.973
    20 ppm    2              0.942         0.993
  The window does much of the work; our embedding still makes 8-9x fewer errors. True
  candidate inside the window for every query (mass calc verified). In-distribution for us.

## C11: on UNSEEN low-resolution ion-trap CID spectra our encoders do not transfer (2026-09-25)

HEK confident PSMs (psm-rerank-hek-hct116, 8 runs; <=512 peaks; groups capped at 20;
27,637 spectra; HCT116 almost entirely excluded by the peak cap). Job 8866238. Exp MAP@R:

    binned cosine, 1 Da bins          0.554
    GLEAMS (pretrained)               0.530   (Hit@1 0.684)
    binned cosine, 0.1 Da bins        0.247
    ours, C7 400m (3 seeds)           0.167-0.173
    ours, replicate-corpus only 400m  0.016-0.018

- Data is fine (binned 1 Da and GLEAMS work); our encoders collapse. 1 Da bins beating
  0.1 Da 2x says peak positions are only ~0.5 Da precise here.
- Most likely a resolution/fragmentation DOMAIN shift: our models depend on precise m/z
  (denoise: m/z rounded to bf16 -> AUROC 0.70). To check: instrument provenance of
  MSConsensus-100M / ms-contrastive-100k; an m/z-jitter test on the 100k test split.
- Consequences: the C claim vs GLEAMS is domain-limited (wins on ms-contrastive-100k
  0.714 vs 0.646, loses badly here); expect the same for A vs yHydra on this data; R's
  small embedding gain on these runs is consistent.

## A vs yHydra on UNSEEN data (C11 HEK ion-trap set): ours ahead everywhere, both degrade (2026-09-25)

26,628 spectra yHydra can represent (of 27,637); candidates 6,201 sequences (yHydra) /
7,380 peptide+charge (ours); A1 3 seeds within 0.002. $SCRATCH/baselines/c11_cap20/xmodal.

    candidates   truth inside   yHydra   A1
    open         100%           0.016    0.061
    +-1.1 Da     95.9%          0.345    0.601
    20 ppm       75.7%          0.606    0.679

- Ours ahead in every setting (widest at 1.1 Da); both collapse without a mass filter --
  the same low-res domain shift as C11 (our spectrum encoder: MAP@R 0.17 here).
- 20 ppm drops the true peptide for 24% of spectra (precursor accuracy of these runs), so
  it caps at 0.757; +-1.1 Da is the sensible window here.
- Lab, 2026-09-25: HEK runs are high-res MS1 / LOW-res MS2 (ion-trap fragments); HCT116 is
  high-res in both. C11's collapse is therefore the low-res MS2 domain. HCT116 = the unseen
  high-res test (C11-HCT116, queued; >512-peak spectra trimmed to top-512 for all methods).

## A4 on the TEST split: no retrieval gain over A1; A2 validation +1.4 (2026-09-25)

ms-contrastive-100k test, 25,848 spectra vs 9,771 candidates (job 8866281):

    A1 (MSE, 3 seeds)                       0.8975-0.8989
    A4 LiT only, no MSE (3 seeds)           0.8975-0.8980
    A4 LiT + 4 hard negatives (3 seeds)     0.8950-0.8971   <- picked on validation (0.934)
    A4 LiT + MSE 0.1, +-hard negatives      0.8923-0.8959

The in-training validation check (2,000 spectra vs a few hundred peptides) ranked A4 +0.2
above A1; that does not hold on test. A4's purpose (near-miss swaps) is judged by its FDR
eval (re-embed 8866354). A2 (400m C7-final teacher) validation 0.943-0.946 vs A1 0.929-0.932:
test eval 8866468 running.

## A2: the 400m teacher lifts the student to 0.923 test Hit@1 (+2.5 over A1) (2026-09-25)

ms-contrastive-100k test, 25,848 spectra vs 9,771 candidates. Teacher = C7 400m final,
seed 0 (chosen on validation; its own MAP@R 0.868). Students 8866356/7/9:
    A2  0.9229 / 0.9227 / 0.9226   Hit@5 0.949   MRR 0.935
    A1  0.8989 / 0.8982 / 0.8975   (teacher C7 50m step 600, MAP@R 0.831)
Student quality tracks teacher quality (A2 answered). Downstream (reranking, yHydra) next.

## Reranking with A2 (400m teacher) and A4, with the NULL control (2026-09-25)

Same 8 runs, MSFragger features, global CV-by-run, PSMs at 1% FDR (stage 2 of the re-embed
jobs; single seed). embws = cosine + within-spectrum features; nullws = the same five built
from the random-spectrum cosine.

    rescorer  base     A1 embws  A4 embws  A2 embws          A2 nullws
    MLP       93,206   93,843    94,224    94,508 (+1,302)   93,386 (+180)
    linear    91,780   --        92,306    92,576 (+796)     91,733 (-47)

- Real minus null: A2 +1,122 (MLP) / +843 (linear); A4 +667 / +593. The gain is the
  embedding's information, not an extra input. Leakage AUROC: A2 cosine 0.526, null 0.502.
- A2 > A4 > A1 for reranking (retrieval: A2 0.923 > A1 0.898 ~ A4 0.895). A4's hard
  negatives help reranking though not retrieval. Jobs 8866603 (A2), 8866354 (A4).
- Lab-feature version with A2 (global + per-run, + vectors): 8866608 running.

## A2 lab-feature reranking with null controls; vector-product arm is CONFOUNDED (2026-09-25)

8 runs, PSMs at 1% FDR, seed 0 (job 8866608; log copied to results/rerank/psm/a2-400m_r4_seed0.log).
real - null = gain of our embedding features minus the same features from a RANDOM spectrum:

    base                    features  real               null               real-null
    per-run linear, lab     embws     129,618 (+577)     129,054 (+13)      +564 (+0.44%)
    global MLP, lab         embws     105,672 (+887)     105,077 (+292)     +595
    global MLP, lab         embvec    106,371 (+1,586)   106,178 (+1,393)   +193
    per-run linear, ms      embws     101,885 (+2,731)   99,180 (+26)       +2,705 (+2.7%)
    per-run linear, ms      embvec    104,885 (+5,731)   101,539 (+2,385)   +3,346

- Within-spectrum cosine features: null adds ~0; A2 beats A1 on the strongest base
  (+0.44% vs +0.37%) and gives +2.7% on engine features alone.
- Vector product (R5): the NULL arm gains up to +2.4k -- the random-spectrum product still
  carries the PEPTIDE's own embedding, so the rescorer can learn sequence-only (decoy-like)
  patterns. Do not report embvec gains without a stricter control (e.g. product with a
  spectrum from the same precursor-mass window, or peptide-only features as their own arm).
- A2 vs yHydra in distribution: open 0.925 vs 0.196; +-1.1 Da 0.979 vs 0.754; 20 ppm 0.994
  vs 0.942 (A1: 0.901 / 0.973 / 0.993). A6 (A2 teacher + A4 loss) validation 0.9425-0.945
  = A2; test/reranking evals not yet run.

## A4 chain complete: hard negatives did not fix near-miss blindness; MLP single-seed noise ~ hundreds (2026-09-25)

- Synthetic near-miss (A4 best, informational): true peptide above adjacent swap 68.8%
  (A1 70.5%); vs reversed 98.3%, mass-matched 98.8%. A4's hard negatives did not help here.
- MLP ms+hand (single seed): A1 embws +930 / nullws +455; A4 embws +1,132 / nullws +836.
  Null arms of this size mean single-seed MLP differences of a few hundred PSMs are noise;
  the per-run linear arms (null ~0) are the clean evidence. R ablation (3 seeds) running.
- MLP ms (single seed), A1 re-embed: embws 93,817 / nullws 93,185 (base 93,206).

## C13 nine-species (yeast, UNSEEN, HIGH-RES): ours loses to GLEAMS and to binned cosine (2026-09-25)

InstaDeepAI/ms_ninespecies_benchmark test split (DeepNovo nine-species, yeast), 86,184
spectra in 20,978 groups (>=2, capped at 20), nothing trimmed (max 452 peaks). Job 8866804
(capacity rerun after the 1 h debug limit killed 8866521). Exp MAP@R / Hit@1:

    binned cosine 0.1 Da        0.789 / 0.891
    binned cosine 1 Da          0.790 / 0.892
    GLEAMS (pretrained)         0.676 / 0.820
    C7 400m, 3 seeds            0.514 / 0.565 / 0.475   (Hit@1 0.707 / 0.740 / 0.679)
    replicate-only 400m         0.444 / 0.474 / 0.503   (Hit@1 0.665 / 0.685 / 0.703)

- On clean unseen high-res data our encoders trail GLEAMS by 0.12-0.20 and binned cosine
  by 0.23-0.32 MAP@R: the C11 collapse was not only low-res MS2. Seeds disagree by 0.09
  (vs +-0.002 in distribution): the embedding is fitted to its training domain.
- C claim is IN-DISTRIBUTION only (ms-contrastive-100k 0.868). Diagnostics proposed:
  frozen+ABTT on this set, earlier C7 checkpoints, ms-contrastive-100k provenance/filtering.

## A vs yHydra on nine-species yeast (UNSEEN, high-res): we win open retrieval, yHydra wins with a mass filter (2026-09-25)

75,476 spectra yHydra can represent (of 86,184); candidates 14,424 sequences (yHydra) /
18,174 peptide+charge (ours). A1, job 8866807:

    candidates   median   yHydra   A1
    open         14-18k   0.060    0.253
    +-1.1 Da     20-25    0.650    0.537
    20 ppm       3-4      0.890    0.691

- With the mass filter (the practical setting) yHydra is clearly better on this unseen set;
  our embedding localises peptide space better (open 4x) but separates same-mass candidates
  worse -- consistent with the spectrum encoder's poor transfer (C13).
- A claim: clearly ahead in distribution (open 0.925 vs 0.196; 20 ppm 0.994 vs 0.942 with
  A2); on unseen data only without a mass filter. A2 version: 8866810 running.

## R per-run ablation, 3 seeds: A2 +0.53% on the strongest base, +2.5% on engine features (2026-09-25)

Per-run linear rescorer, 8 runs, PSMs at 1% FDR; real - null (null = same features from a
random spectrum). Jobs A1 8866860/8866890/8867044, A2 8866608/8867047/8867050.

    base        emb   seed0    seed1    seed2    mean
    lab (~129k) A2    +564     +924     +544     +677 (+0.53%)
    lab         A1    +398     +738     +473     +536 (+0.42%)
    MSFragger   A2    +2,705   +2,373   +2,386   +2,488 (+2.5%)
    MSFragger   A1    +1,495   +1,215   +1,186   +1,299 (+1.3%)

Null arms: -96 to +290. Positive on every seed; A2 > A1 on every seed and base.

## C diagnostic (nine-species yeast 20k subset, UNSEEN): the pretrained representation transfers better than any fine-tuned model (2026-09-25)

20,019 spectra in 4,773 groups (subset of C13's yeast set; easier, so absolute numbers are
higher than C13). ABTT fitted on 25k spectra of the 8 OTHER species. Job 8867207. Exp MAP@R:

    binned cosine 0.1 / 1 Da                     0.916 / 0.916
    GLEAMS                                       0.770  (Hit@1 0.886)
    FROZEN 400m@220k + ABTT (block14, D128)      0.709  (raw: best block 0.510, final 0.355)
    FROZEN 400m@430k + ABTT (block09, D128)      0.634
    C7 400m seed0: step 300 / 600 / 900 / final  0.606 / 0.655 / 0.596 / 0.596
    C7 400m final seeds 0 / 1 / 2                0.596 / 0.656 / 0.550
    C7 50m seed1: step 600 / final               0.543 / 0.523
    replicate-only 400m / 50m                    0.521 / 0.499

- Frozen 400m + ABTT (no labels) beats every contrastively fine-tuned model out of
  distribution: fine-tuning SPECIALISES (in-distribution 0.868) at the cost of transfer.
- Transfer peaks mid-epoch (step 600) then declines; seeds differ by 0.1 OOD (vs 0.002 in
  distribution). 400m > 50m OOD. For the frozen encoder 220k > 430k.
- Nothing beats binned cosine on these clean high-res spectra; GLEAMS stays ahead of us.
- Next: an OOD VALIDATION set from the 8 other species (nine-species train split) to select
  checkpoints/teachers for transfer; a student on the best-transferring C7 teacher -> R.

## MS2Rescore full + A2 embedding with NULL: +0.54% real, +0.35% null -> +0.20% net; the HCT116 "+4.1%" was noise (2026-09-25)

Same 8 runs, same MS2Rescore config; extra rescoring:emb_* columns (job 8867053):

    arm                      all             HEK             HCT116
    MS2Rescore full          128,211         120,747         7,139
    + A2 cosws               128,909 (+698)  121,004 (+257)  7,482 (+343)
    + A2 null                128,658 (+447)  120,753 (+6)    7,591 (+452)
    real - null              +251 (+0.20%)   +251            -109

- HEK: a clean small gain on top of MS2Rescore (+257 vs null +6).
- HCT116: null gains as much as the real embedding -> MS2Rescore's run-to-run spread on these
  two small runs is hundreds of PSMs. CORRECTION: the earlier A1 "+4.1% on HCT116" is within
  that noise and should not be cited. Measured noise: plain repeat 8867056 (queued) and the
  18-run HCT116 jobs.

## Overnight batch 2026-09-25 (afternoon UTC)

**OOD-selected teacher.** OOD validation = nine-species TRAIN split (8 non-yeast species),
20k spectra. C7 400m checkpoints (exp MAP@R): best s600_seed1 0.689; final seeds 0.665 /
0.682 / 0.642; seed 2 worst at every step. Student on s600_seed1 ("400m-oodsel"):
test Hit@1 0.918 / 0.918 / 0.917 (A2 0.923); in-distribution yHydra comparison 20 ppm 0.994.

**A vs yHydra on nine-species yeast (UNSEEN)**, 75,476 spectra:

    candidates   yHydra   A1      A2      A-oodsel
    open         0.060    0.253   0.388   0.410
    +-1.1 Da     0.650    0.537   0.639   0.659
    20 ppm       0.890    0.691   0.761   0.765

Selecting the teacher by OOD validation closes the gap: level with yHydra at +-1.1 Da,
still behind at 20 ppm; 7x better open.

**MS2Rescore noise.** Plain repeat of MS2Rescore full on the 8 runs: 128,209 (first run
128,211); HEK 120,706 (-41), HCT116 7,226 (+87). So run-to-run noise ~+-50 total, ~+-90 on
HCT116. A2 real +698 / null +447 (HCT116 null +452 >> noise): the null arm DOES carry signal
in MS2Rescore on HCT116 (student(peptide) . random spectrum is a peptide-only feature) --
real - null (+251, HEK) is the defensible number.

**Zero-shot + ABTT on yeast 20k (OOD scaling, frozen encoders, fitted on other species):**
50m@220k 0.602, 100m@220k 0.421, 200m@220k 0.609, 200m@540k 0.654, 400m@220k 0.709,
400m@430k 0.634 (raw best block 0.18-0.51). D=128 best everywhere (edge of range).

**HCT116 x18 A2 re-embed** done (2 h 41 min on 3 nodes; runs are ~144k spectra each).

## Audit of the MS2Rescore baseline and of the lab-feature classifier (2026-09-25 evening)

Prompted by "MS2Rescore full + null control" (128,658) sitting above "MS2Rescore full" (128,211).

- MS2Rescore 4.0.2 setup is correct: feature generators basic + MS2PIP + DeepLC + ms2; SVM rescoring;
  MS2PIP model CID (HEK) / HCD2021 (HCT116); one fragment tolerance (0.3 Da) annotates every spectrum
  and MS2PIP reuses it (core.py). DeepLC calibrates per run. "Non-mapped modifications" warning is
  benign: Oxidation / Acetyl / Carbamidomethyl are Unimod labels and resolve to the right masses
  (checked with psm_utils in the MS2Rescore env). Converter output: ProForma + charge, decoy flag,
  score = -log10 e-value, MSFragger features as rescoring: columns. Gains are in the published range
  (HEK-0628-5 +36% over the search engine; 8 runs 89,693 -> 128,211, +43%).
- The +447 of "+ null" is MS2Rescore retrained with 5 uninformative columns: HEK +6, HCT116 +452.
  The two HCT116 runs start from ~1k PSMs (1,026 at 1% before rescoring), liblinear warns it did not
  converge, and their totals swing by hundreds with any change (HCD2021 7,139; CID model 6,317;
  + embedding 7,482; + null 7,591). Plain repeat is stable (128,211 vs 128,209). Not information.
- Lab features: the prediction-based columns (predicted_* 25, rt_* 3, ccs_* 3) and the lab's own
  embedding_* columns (8) are EMPTY in the published table, so our loader drops them; "all features"
  = fragment-ion coverage/intensity, xcorr proxies, per-spectrum context (rank/margin/z of each score
  among the candidates), peptide/tryptic features and native scores of FOUR engines (MSFragger
  hyperscore, Comet / SEQUEST / ProLuCID xcorr, ProLuCID binomial/z). The multi-engine scores + context
  features plausibly explain parity with MS2Rescore full without predicted spectra or RT.
- Leakage scan (HEK-0628-5, all 342,412 candidates): the best single lab feature separates target vs
  decoy at AUROC 0.550 (MSFragger hyperscore 0.543). No leak.
- Resolution check: y-ion errors of the 300 best target PSMs, median 0.055 Da in BOTH HEK-0628-5 and
  HCT116 HILIC14 (20% within 0.02 Da); peak m/z decimals look alike. So HILIC14's MS2 does not look
  Orbitrap-accurate; 0.3 Da is appropriate there. "HCT116 = high-res MS2" (lab, 2026-09-25) needs
  confirming per run with the lab / raw metadata before we call HCT116 results "high-res".
- Correction: the 18-run HCT116 MS2Rescore job with A2 (8867717) already includes the null arm
  (EMB_SETS="cosws null").

## C14 zero-shot (frozen + ABTT) -- settings fixed BEFORE seeing C14 (2026-09-25 21:40 UTC)

Frozen encoders on the C14 HCT116 20k set, ABTT fitted on nine-species-train-fit25k (unlabelled, no
human data; same as the yeast run). Pre-registered setting = the layer / D each encoder preferred
on yeast 20k (train fit): 400m@220k block14 D128 (yeast 0.709), 200m@540k block12 D128 (0.654),
50m@220k block07 D128 (0.602). The best-over-layers/D number on C14 is reported only as an upper bound.
Result (job 8870383, 22:32 UTC): all near chance on C14 (MAP@R; pre-registered / best-over-settings upper bound / raw best layer):
400m@220k 0.033 / 0.038 / 0.034; 200m@540k 0.015 / 0.017 / 0.016; 50m@220k 0.006 / 0.007 / 0.006 (ABTT fitted on the
C14 spectra themselves: at most 0.047). Yeast 20k for the same encoders: 0.709 / 0.654 / 0.602. So frozen embeddings do not
transfer to HCT116 at all; fine-tuned C7 400M reaches 0.23-0.32, GLEAMS 0.658, binned 1 Da 0.742. Not a data-loading issue:
same 20,002 queries and prepared data as the fine-tuned eval. No further C compute (user, 2026-09-25).

## Cross-over results (job 8870274, 8 runs, 2026-09-25 23:07 UTC)

C. MS2Rescore WITHOUT DeepLC (basic + MS2PIP + ms2 features), one run each:
    plain 124,152 (HEK 117,816 / HCT116 5,872); + iona embedding 124,670 (118,267 / 6,130); + null 124,338 (117,824 / 6,203)
    net embedding - null = +332 (+0.27%): HEK +443 (null +8, clean), HCT116 -73 (noise, as before).
    DeepLC is worth +4,059 (128,211 with vs 124,152 without); the embedding in its place recovers +518 (net +332),
    i.e. it does NOT replace retention-time prediction.
B. iona-rerank (per-run linear) on MS2Rescore's own features (all-ranks export; MSFragger + MS2PIP + DeepLC + ms2),
   seeds 0/1/2:
    plain 124,792 / 124,863 / 124,652 (mean 124,769); + iona embedding 125,058 / 124,970 / 125,001 (125,010);
    + null 124,892 / 124,843 / 124,860 (124,865). Net embedding - null per seed +166 / +127 / +141 (+145, +0.12%):
    small but positive in every seed. Note iona-rerank on MS2Rescore's features (124.8k) is below MS2Rescore's own
    classifier on them (128.2k); with the rich features iona-rerank reaches 128.9k.

## MS2Rescore without MS2PIP (+ Iona embedding / + null in its place), 8 runs (job 8870729, 23:42 UTC)

    plain 111,421 (HEK 106,181 / HCT116 5,006); + Iona embedding 112,342 (107,182 / 4,955); + null 111,511 (106,271 / 4,887)
    net embedding - null = +831 (+0.75%): HEK +911 (null +90), HCT116 +68 (noise level).
    MS2PIP is worth +16,790 (128,211 with vs 111,421 without); the embedding in its place recovers +921 (net +831), so it
    does NOT replace predicted fragment spectra either, but its net gain is 2.5x the no-DeepLC case (+332).

## 18-run HCT116 MS2Rescore jobs killed by memory (8867717, 8870275, 8870544: PBS exit -20 after ~16-18 min)

MS2Rescore peak RSS 37 GB on HILIC14 (240k PSMs, 8-run set); the 18-run set's runs are ~5.7x larger (HILIC1: 1.37M PSMs,
144k spectra) -> ~200 GB each; 18 or 9 in parallel exceed the ~1 TB node. Would need MAXPAR<=3 and ~70 min per run:
~7 h per arm (plain, + embedding, + null) -> not feasible before the deadline alongside the other capacity jobs.

## MS2Rescore with MSFragger features only +/- Iona embedding (job 8870788, 8 runs, ~23:59 UTC)

    plain 100,974 (HEK 97,722 / HCT116 2,950) -- identical to the earlier search-only run, so reproducible
    + Iona embedding 102,751 (99,445 / 3,119); + null 101,007 (97,652 / 2,976)
    net embedding - null = +1,744 (+1.73%): HEK +1,793 (null -70), HCT116 +143 (emb +169 vs null +26).
    Clean: the null adds nothing, the embedding adds 1.7% in MS2Rescore's own classifier -- the same picture as
    Iona-rerank with MSFragger features (+2.5%).

## Mouse (Noble nine-species-balanced, Mus musculus) vs yHydra (job 8870879, 2026-09-26 00:53 UTC)

25,490 annotated spectra; shared (yHydra-representable) queries 19,710; Iona = iona-contrastive-400m + iona-peptide-embedder-400m.
    Hit@1          open     +-1.1 Da   20 ppm (ceiling 0.897: 10.3% of true peptides outside 20 ppm)
    yHydra (L2)    0.238    0.769      0.851
    Iona           0.925    0.986      0.894
    Iona, full set (every query / every candidate, open): 0.763
Far stronger than on yeast (DeepNovo nine-species test: open 0.39, 20 ppm 0.76 vs yHydra 0.89). CAVEAT before claiming:
mouse was NOT used to select this model, but the training corpus (ms-contrastive-100k) may overlap public mouse data
(the MassIVE-KB overlap question from C11 is still open) -- check provenance / peptide overlap before calling it unseen.

## 20 ppm ceilings are precursor isotope mis-assignment (verified on all spectra, 2026-09-26 ~01:15 UTC)

Recorded precursor mass vs the peptide's computed mass (the window test's own peptide_mass): HEK 27,637 spectra,
24.9% outside 20 ppm = +1 isotope 18.8% / +2 4.3% / -1 1.7%; mouse 25,490 spectra, 11.9% outside = all +1.
EVERY out-of-window case is within 20 ppm of an exact isotope offset (k x 1.00336 Da): no residue from our mass
calculation or modification mapping. The earlier "mostly" for mouse was unverified; now measured.

---

## Precursor filter width: C19's gain is biggest unfiltered and is NOT a within-window effect (job 8872964, 2026-09-26 23:53 UTC)

Same embeddings, only the candidate filter varies (`pbs/diag/mass_window_eval.py`; theoretical
masses; experimental spectra; `results/finetune/contrastive/mass-window/summary.json`).
50M@540k, ms-contrastive-100k only, 3 epochs, mean of 3 seeds, MAP@R:

    validation      open   100Da  25Da   5Da    1Da    0.1Da  100ppm 20ppm  10ppm
    frozen 50m      .103   .125   .159   .220   .325   .451   .532   .768   .862
    supcon mass     .868   .890   .915   .945   .969   .980   .981   .991   .994
    supcon random   .858   .871   .897   .932   .962   .976   .976   .988   .993
    mass - random  +.009  +.019  +.018  +.013  +.007  +.003  +.005  +.003  +.001
    OOD (8 species)
    supcon mass     .724   .774   .816   .869   .926   .958   .966   .986   .992
    supcon random   .645   .691   .741   .807   .884   .931   .940   .975   .986
    mass - random  +.080  +.083  +.076  +.063  +.042  +.027  +.026  +.011  +.006
    sigmoid: same pattern, larger gaps (OOD +.123 open, +.011 at 20 ppm)

1. A filter helps everything a lot; at 20 ppm same charge every SupCon model is >= 0.975, and the
   frozen encoder goes from 0.10 to 0.77 (validation) -- that is the filter, not the encoder.
2. C19's gain shrinks monotonically as the window narrows (OOD +0.080 open -> +0.011 at 20 ppm,
   still ~20x the seed spread). With a search engine's filter the benefit is small.
3. Only 1-2% of open-retrieval top-1 errors are within 1 Da of the query (chance ~0.1%), so
   errors are overwhelmingly FAR-mass confusions, and same-mass training reduced them
   (OOD 3,505 -> 2,658 errors). Harder negatives improved the embedding GLOBALLY rather than
   teaching within-window discrimination.
4. "Crowding" (other peptides within 1 Da) does not measure difficulty: open MAP@R RISES with
   crowding (validation .73 -> .87), likely because isolated masses are unusual peptides
   (very short/long/heavily modified). Not used further.

Would overturn: a mixed-batch or jitter ablation where near-mass errors, not far ones, move;
or the same pattern failing at another scale.

---

## Filter failures (isotope-offset precursors): open retrieval rescues ~80%, the isotope-tolerant filter beats both (job 8873366, 2026-09-27 03:04 UTC)

`pbs/diag/filter_failure_eval.py`, MEASURED precursor m/z of query and gallery, same charge. F = queries
with >= 1 positive outside 20 ppm (mouse 39.6%, human 34.8%, HEK 53.0%); F_all = all positives outside
(5.8% / 7.1% / 13.0%). Recorded precursors are never monoisotope-corrected: off-window cases are whole
isotope steps (mouse 10.8% / human 11.6% of spectra, all +1; HEK 24.9%). iso = |dmz - k*1.00336/z| <= 20 ppm,
k in -2..2. MAP@R open / 20ppm / iso, one seed:

    mouse     iona400m full .846/.822/.946   F .796/.571/.920   Fbar .879/.985/.964   rescue(F_all open Hit@1) .83
              c19_mass full .847/.823/.952   F .800/.573/.928                          rescue .84(human) .83(mouse)
              binned0.1 full .916/.825/.963 (beats our models on mouse; not on human: .809 vs .891 open)
    HEK       every encoder near-useless open (.09-.21); binned 1.0 Da .654 open / .889 iso beats all models

1. The 20 ppm filter loses: 5-6% of queries go right -> wrong at Hit@1 (mouse, human); F MAP@R drops ~.25.
2. Our encoders DO recover most of what it drops: open Hit@1 on F_all 0.79-0.84 (mouse, human).
3. The isotope-tolerant window removes that loss (net loss 0.0000) and is the best full-set filter for every
   trained model; on Fbar it is slightly below 20 ppm (~5x more candidates).
4. C19 same-mass vs random: +0.02-0.03 on F, similar on Fbar -- no F-specific effect.
5. HEK (low-res MS2) is out of domain for the encoders; the question can't be answered there.

Would overturn: more seeds reversing (4); a dataset where the isotope window's extra candidates cost more than it recovers.

---

## C23 50m HP search on the single-dataset recipe: learning rate is the lever, the rest transfers (jobs 8872806, 8873673, 8873676; 2026-09-27 04:34 UTC)

One factor at a time around base (lr 1e-4, t 0.002, KL 10, P85xK3), 50m@540k, 3 epochs, one seed; full
validation (exp MAP@R) and 8-species OOD. Base reproduces the c8c19 supcon_mass 3-seed mean (.867 vs .8675);
seed spread there ~±.001 validation, ~±.009 OOD.

    setting     val    OOD        setting     val    OOD
    lr4e-4     .879   .752        base       .867   .734
    lr2e-4     .876   .740        t0.001     .864   .730
    kl0        .874   .732        kl30       .859   .727
    kl1        .872   .743        t0.01      .859   .723
    p128k2     .871   .749        lr5e-5     .855   .717
    p170k3     .867   .736        t0.005     .865   .739

1. lr 4e-4 wins on both sets (+.012 / +.018) and is the edge of the range: optimum unlocated.
2. Temperature, KL, group shape: within noise of base or worse; the old recipe's values transfer.
3. Side finding (K54): MAP@R with consensus spectra in the gallery swings with KL (kl0 .848, base .744,
   kl30 .664) while experimental-only barely moves -- the KL anchor ties the embedding to how the
   pretrained model sees consensus-style spectra. Tested next with consensus in training (C20).
Next: sweep-hp50b (lr 8e-4, 1.6e-3, lr4e-4+p128k2, lr4e-4+kl1, x3 seeds; K53).


---

## C21 batch composition: no mix beats pure same-mass; C23/K53: lr 4e-4 + P128xK2 is the new best (jobs 8873159, 8873563, 8873825; scored 8874549/50/60/62, 2026-09-27 17:20 UTC)

50m@540k, ms-contrastive-100k only, SupCon, 3 epochs, 3 seeds each (± = half range); exp MAP@R. Mouse/human
are unseen species with MEASURED precursors (F = queries the 20 ppm filter fails, 40% / 35%).

    batches                              val        OOD        mouse open/20ppm/iso   mouse F open/20ppm/iso   human open
    random                               .858±.000  .645±.001  .828/.820/.942          .778/.571/.915           .879
    same-mass (100%)                     .868±.000  .724±.009  .847/.823/.951          .800/.573/.927           .891
    within75 (64 same-mass + 21 random)  .870±.002  .713±.006  .844/.823/.950          .797/.573/.925           .890
    within50                             .867±.001  .680±.007  .839/.822/.947          .791/.572/.922           .887
    between75                            .870±.000  .714±.004  .845/.823/.951          .799/.573/.926           .890
    regions75 (two mass regions)         .870±.001  .714±.010  .844/.823/.950          .797/.573/.925           .890
    regions50                            .870±.000  .715±.002  .843/.822/.949          .795/.572/.924           .890

1. Pure same-mass is best or tied everywhere OOD and on unseen species; every mix loses ~0.01 OOD (within50 ~0.04)
   while gaining only +0.002 in-distribution. Two-region = within75 = between75 within noise. Same-mass stays default.
2. The filter columns barely move between batch designs: the design changes open retrieval, not filtered.

K53 (lr follow-up, same-mass, 3 seeds):   lr1e-4 ref .868 / .724;  lr8e-4 .869 / .729;  lr1.6e-3 .821 / .549;
    lr4e-4+KL1 .879 / .751;  **lr4e-4 + P128xK2 .886±.002 / .784±.009**  (val / OOD)
3. The lr optimum is located (~4e-4: 8e-4 no better than 1e-4, 1.6e-3 degrades). P128xK2 (more peptides per batch,
   2 spectra each) on top of lr 4e-4 adds +0.007 val / +0.033 OOD -- the largest OOD gain of any single-dataset change.

---

## C20 consensus in training: headline unchanged, consensus-gallery retrieval fixed (job 8873850; scored 8874703/04/13/15, 2026-09-27 17:56 UTC)

50m@540k, same-mass, lr 1e-4, 3 seeds (± half range). "val all" = consensus spectra included in queries + gallery.

    training                      val exp     val all     OOD         mouse open/20ppm/iso   mouse F open  human
    experimental only (ref)       .868±.000   .753±.005   .724±.009   .847/.823/.951         .800          .891
    +consensus, 85x3 of 4         .865±.001   .893±.001   .736±.003   .843/.823/.952         .796          .888
    +consensus, 64x4              .861±.001   .890±.000   .719±.006   .841/.823/.951         .795          .889
    +consensus, 85x3, KL 0        .872±.001   .899±.001   .737±.010   .840/.823/.952         .793          .891

1. Experimental-only retrieval (the headline) barely moves (-.003 to +.004); OOD +.012 (K3); unseen mouse -.004 to -.007.
2. Retrieval WITH consensus spectra jumps .753 -> .89-.90: the K54 KL sensitivity came from consensus spectra never
   being seen in training; once they are, KL matters little (.893 at KL 10 vs .899 at KL 0).
3. So consensus training only pays if the use case searches against consensus spectra (a spectral library).
   Proposal (not decided): keep experimental-only as the default; revisit for library-search evaluations.

---

## C23 / K66-C per-scale search: 100m and 200m (jobs 8875262, 8875261; scored 2026-09-28 06:16-11:00 UTC). 400m / 25m pending (queue blocked by maintenance)

Single-dataset recipe (SupCon + same-mass batches, t 0.002, KL 10, 3 epochs, final pretraining checkpoint 540,423),
4 settings x 3 seeds per scale. Experimental MAP@R, mean ± half range over seeds; mouse/human/yeast are
unseen species (measured precursors); "mouse F" = queries the 20 ppm filter fails; library search on
validation (consensus-only library): Hit@1 and median rank of the true entry.

    100m: setting            val          OOD          test   mouse  human  yeast | mouse F  yeast 20ppm/iso | val lib Hit@1 (median rank)
        lr4e-4_p170k2    0.902±0.000 0.768±0.007 0.903  0.856  0.895  0.566 | 0.813    0.922/0.796    | 0.885 (1.0)
        lr2e-4_p128k2    0.900±0.002 0.760±0.007 0.900  0.856  0.896  0.547 | 0.811    0.910/0.778    | 0.815 (1.0)
        lr4e-4_p128k2    0.898±0.001 0.758±0.004 0.900  0.855  0.895  0.571 | 0.811    0.923/0.799    | 0.867 (1.0)
        lr8e-4_p128k2    0.880±0.001 0.773±0.024 0.883  0.844  0.890  0.637 | 0.801    0.952/0.850    | 0.824 (1.0)
    200m: setting            val          OOD          test   mouse  human  yeast | mouse F  yeast 20ppm/iso | val lib Hit@1 (median rank)
        lr4e-4_p170k2    0.909±0.000 0.802±0.028 0.912  0.861  0.898  0.628 | 0.817    0.951/0.847    | 0.920 (1.0)
        lr4e-4_p128k2    0.907±0.001 0.792±0.011 0.908  0.857  0.897  0.603 | 0.813    0.945/0.836    | 0.915 (1.0)
        lr2e-4_p128k2    0.905±0.001 0.818±0.004 0.909  0.859  0.898  0.662 | 0.814    0.965/0.871    | 0.900 (1.0)
        lr8e-4_p128k2    0.891±0.002 0.715±0.014 0.891  0.848  0.893  0.546 | 0.805    0.919/0.796    | 0.906 (1.0)

Observations (no decision -- selection is the user's call once all four scales are in):
1. By VALIDATION (the selection metric) lr 4e-4 + 170 x 2 leads at both scales, by 0.002-0.004 (above the
   ±0.001 seed spread at 100m, borderline at 200m); lr 8e-4 is clearly worst in-distribution at both.
2. Validation and the unseen sets disagree: OOD / yeast prefer lr 8e-4 at 100m and lr 2e-4 at 200m. OOD
   seed spread is large (up to ±0.028), so single OOD differences under ~0.03 are not reliable.
3. Scale helps: 200m beats 100m on every set for the lead setting (val .909 vs .902, OOD .802 vs .768,
   yeast .628 vs .566); both beat 50m's best (val .886, OOD .784).
4. Yeast (far OOD, expected -- user) stays at .55-.66 open. On yeast the plain 20 ppm filter (.91-.97) BEATS
   the isotope-tolerant one (.78-.87): yeast's recorded precursors show ~no isotope errors, so widening the
   window only adds wrong candidates (cf. mouse/human, where iso wins).
5. Library search (consensus-only library, validation): Hit@1 .82-.92, median rank 1 everywhere; ranking of
   settings roughly follows validation MAP@R at 200m but not at 100m (lr 2e-4: .815 vs lr 4e-4: .867-.885).

## K114-P: per-block Pairformer time (job 8877117, 2026-09-29; stage-0 config, 1 tile, bf16)

Raw: results/raw/diag/pairformer_profile/8877117.{json,log}. The job aborted (GPU page fault in the
compute runtime right after an OOM case, rc 134) during `pairformer_triattn` B=32 N=150; the 13
cases before that are complete. Missing: pairformer_triattn B=32 N=150 (gc on), N=256, N=512.

- Without triangle attention, the two triangle multiplications are ~2/3 of the step
  (b_trimul_out 33-37%, c_trimul_in 32-34%); writeback ~8-9%, pair transition ~8-9%; the single
  stream (attention + transition) is only 3-5%.
- With triangle attention (SDPA): the two triangle attentions take ~45% (22% + 23%) and the step
  roughly doubles (B=32 N=100: 555 -> 1066 ms; B=8 N=150: 310 -> 669 ms).
- One pair update costs 5-27 single-stream blocks (grows with N): N=100 ~5-13, N=150 ~10-26,
  N=256 ~25 (no triattn).
- Gradient checkpointing (gc on) costs +33-36% step time and cuts peak memory 7-8x
  (B=32 N=150: 54.6 -> 7.4 GB). Without it, B=32 fits only up to N=150 (no triattn) / N=100
  (triattn); N=512 does not fit even with gc at B=32.


## K100-C: library search on the C20 models (validation 8877125, test 8877126; 2026-09-29)

Raw: results/raw/finetune/contrastive/c20-{validation,test}-lib/. Experimental queries vs a consensus-only
library, one correct entry per query; ties count as misses. No query fails the precursor filter on this
set (F n = 0), so the filtered-failure columns are empty; filter passes = all queries.

| arm (3 seeds) | val Hit@1 open / 20 ppm / iso 20 ppm | test Hit@1 open / 20 ppm / iso 20 ppm |
|---|---|---|
| cons K3 KL0 | 0.939-0.940 / 0.997 / 0.992 | 0.942-0.943 / 0.997-0.998 / 0.993 |
| cons K3     | 0.936 / 0.997 / 0.991 | 0.939 / 0.997 / 0.991 |
| cons K4     | 0.930-0.931 / 0.996 / 0.988-0.990 | 0.933-0.935 / 0.996 / 0.989-0.990 |
| same-mass reference | 0.703-0.730 / 0.983-0.988 / 0.952-0.964 | 0.709-0.732 / 0.981-0.986 / 0.948-0.960 |

- Consensus in training lifts unfiltered library Hit@1 by ~0.21-0.23; with the 20 ppm filter the gap
  shrinks to ~0.01 (the filter removes most wrong candidates for the reference model).
- Experimental-only MAP@R is unchanged across arms (val 0.861-0.873, test 0.867-0.879), as in C20.
- Validation and test agree on the ranking (K3 KL0 >= K3 > K4 >> reference). Selection by validation only.
- As before, 20 ppm beats isotope-tolerant 20 ppm here.

## K141-P: the flat Stage 0 transformer is the transformer's normal early plateau (2026-09-29)

W&B CS_Pharm/msdelta-pretrain, run msdelta-50m-production-01 (master's recipe, warmup 2000): loss 0.97 -> 0.93
and FLAT to step ~650 with grad norm 0.03-0.05, then breaks through (0.89 @700, 0.52 @1000, 0.43 @1200).
Stage 0 transformer (8878340): flat 0.83 for all 300 steps, grad norm ~0.03 -- the same plateau; 300 steps
never reach the break-through. The Pairformer leaves its plateau early (0.90 -> 0.32 in 300 steps), as the user's earlier
msdelta-pairformer-50m did (10.5 -> 0.48 by step 400). So Stage 0 measured "time to leave the plateau", not
quality: comparisons need runs well past the break-through, normalised for FLOPs/time (K144 step 1).
Setup differences from master (not the cause, but to fix for real comparisons): warmup 11 vs 2000, 300 steps,
cap 150 peaks, no DeepSpeed / torch_compile, global batch 512.


## C27-C: weighting the consensus more heavily does not help library search (job 8878582; scored 2026-09-29/30)

50m@540k, K66 recipe (lr 4e-4, 128 x 2), 3 seeds; mean ± half range. ref_exp = K53 runs 8873825 (no consensus).
Library search = experimental query vs consensus-only library. No query fails the precursor filter here (F n=0).
Raw: results/raw/finetune/contrastive/c27-{validation,test,oodval,mouse,human,yeast}/ (yeast still scoring).

    validation     exp MAP@R      library Hit@1: open / 20 ppm / iso 20 ppm
    ref_exp        0.886±0.002    0.872±0.004 / 0.994 / 0.983
    cons_w1        0.874±0.002    0.947±0.000 / 0.998 / 0.995
    cons_w3        0.864±0.000    0.947±0.001 / 0.998 / 0.995
    cons_always    0.844±0.000    0.946±0.001 / 0.998 / 0.995
    cons_w3_kl0    0.862±0.002    0.945±0.000 / 0.998 / 0.995
    (test: same ranking; ref 0.888 / 0.872, w1 0.876 / 0.947, always 0.846 / 0.945)

1. Consensus in training (w1) lifts unfiltered library Hit@1 0.872 -> 0.947 (+0.075); with 20 ppm 0.994 -> 0.998.
2. Weighting it more (w3, always) adds NOTHING to library search (0.945-0.947) and costs experimental MAP@R
   in proportion to the weight: -0.012 (w1), -0.022 (w3), -0.042 (always) vs ref. KL 0 doesn't help either.
3. The approved guard (exp MAP@R drop <= seed spread) rejects every consensus arm, w1 included (-0.012 vs
   ±0.002): consensus trades ~0.012 experimental MAP@R for ~0.075 library Hit@1 (user's call, K139c/K135).
4. Note the reference is much better at library search at the new recipe (0.872) than at C20's old recipe (0.71).

## K117-P: triangle-attention copy cost and copy-free layouts (job 8879977, 2026-09-30; 1 tile, bf16)

Raw: results/raw/diag/triattn_copy_bench/8879977.{json,log} (+ traces). Profiler attribution worked on device
(basis "device"). One triangle-attention module (c_z 64, 4 heads x 16, chunk 32), B = 2 / 8, N = 200 / 256 / 512.
- Copies are 17-38% of device time in the current SDPA path; the attention kernel itself 42-56%.
- Copy-free layouts (hview = rows into the head dim with a stride-0 mask; cached = masks built once):
  forward 1.17-1.28x faster and peak memory ~2.5x lower (B=8, N=512: 14.3 -> 5.4-5.7 GB), all accurate.
  But a TRAINING step (fwd+bwd) gains only 0.98-1.08x: the backward pass dominates (B=8 N=512: 232 ms of
  which fwd 59).
- sdpa5d (the K116 layout) is slower everywhere and fails the accuracy rule at N=256.
- Takeaway: layout tweaks buy memory, not training speed; the time is in the attention kernel and its
  backward -> the fused-kernel route (K118/K119) is where training speed must come from. Adopting hview or
  cached is still worth it for memory (bigger batches / N=512). B=32 supplement queued (the Pairformer's
  usual micro-batch, K130).

## K128-P: FlexAttention works on XPU once ONEAPI_DEVICE_SELECTOR is unset; K114's missing cases (job 8879960, 2026-09-30)

- K119 root cause: Triton's Intel launcher aborts ("sycl ... No device of requested type available") when
  ONEAPI_DEVICE_SELECTOR is set -- both level_zero:gpu and level_zero:0 abort; UNSET works. Affects every
  Triton / torch.compile path in our jobs (they export ONEAPI_DEVICE_SELECTOR=level_zero:gpu; master's
  production configs set torch_compile true). ZE_AFFINITY_MASK still pins the tile.
- With it unset: compiled flex_attention for triangle attention matches SDPA exactly in fp32 (out and grads,
  rel. L2 ~1e-7). Speed (one module, fwd+bwd, bf16): flex is NOT faster -- B=8 N=100 11.5 vs 6.8 ms SDPA,
  N=150 20.6 vs 16.6; B=32 N=100 31.3 vs 23.0, N=150 71.7 vs 70.6 (parity at the largest size). Memory:
  the row-head layout uses ~2.6x less (B=32 N=150: 1.41 vs 3.72 GB). First call compiles for 12-27 s.
  Raw: results/raw/diag/flexattn/8879960_c_unset.{json,log}.
- K114 missing (full Pairformer with tri-attn, B=32): N=150 only fits with gradient checkpointing
  (3.8 s/step, 13.8 GB); N=256 12.9 s/step, 47 GB; N=512 OOM even with checkpointing.
- Takeaway for speed: neither layout tricks (K117) nor stock FlexAttention beat SDPA on time; both cut memory.
  Time must come from a custom fused kernel (K118) or from using triangle attention less (decoupled streams).

K117 supplement, B=32 (job 8880031): train step (fwd+bwd) hview 1.01x (N=150) / 1.06x (N=200), cached 1.04x /
1.08x; peak memory hview 2.85 vs 2.85 GB (N=150) and 3.58 vs 5.54 GB (N=200). -> sdpa_view (= hview) made the
default 2026-09-30 (only B=8 N=256 was slower, 0.98x). Note: sdpa5d is 1.25x at B=32 N=150 but slower at
N=200 forward and failed the accuracy rule at B=8 N=256 -- not adopted.

## K114-P validation: decoupled streams on the GPU (job 8880138, 2026-09-30; stage-0 Pairformer, B=32, N=150, 1 tile)

Step time (fwd+bwd) vs k = pair_update_every (lag 0 unless noted):
    no tri-attn (gc off):  k1 1240 ms | k2 684 (x0.55) | k3 574 (x0.46) | k5 347 (x0.28) | k10 239 (x0.19)
    tri-attn (gc on):      k1 3809 ms | k2 1986 (x0.52) | k3 1623 (x0.43) | k5 892 (x0.23) | k10 528 (x0.14)
    lag 1 drops one update (k2 lag1 = k3 lag0 timing; k5 lag1 = k10). All variants ran; tri-attn at k1 needs gc.
Cost model fitted from these: step ~= S + u x P, u = number of pair updates.
    no tri-attn: S (10 single blocks + rest) ~131 ms, P ~108 ms per pair update -> one pair update ~ 0.8x the
    WHOLE 10-block single stack (~10-25 single blocks).  tri-attn: S ~164 ms, P ~364 ms per update (~2.2x the
    whole single stack).
-> Time-balancing single vs pair work ("two singles per pair op") would need k >= 10 at this size; the pair
   update is ~25x (no tri-attn) to ~60x+ (tri-attn) a single block, not 2x. Raw: results/raw/diag/decoupled_profile/.

## K66-C / K136-C per-scale search complete: 25m, 50m (P170), 100m, 200m, 400m (2026-09-30)
Jobs: 25m 8878461, 400m 8878459, K136 50m-P170 8878464 (100m/200m earlier). Validation, experimental queries,
mean of 3 seeds (± half range of MAP@R). All validation queries pass the precursor filter (F n=0), so the
filtered columns are full = pass.

| scale | lr2e-4 P128 | lr4e-4 P128 | lr4e-4 P170 | lr8e-4 P128 |
|---|---|---|---|---|
| 25m  MAP@R / Hit@1 | .8655 / .9152 | .8674 / .9176 | **.8712±.0009 / .9197** | .8468 / .9024 |
| 50m  | (K66) | .886 (C27 ref) | **.8880±.0002 / .9303** | (K66) |
| 100m | .9001 / .9376 | .8981 / .9369 | **.9018±.0005 / .9393** | .8803 / .9253 |
| 200m | .9052 / .9414 | .9065 / .9423 | **.9090±.0005 / .9440** | .8909 / .9320 |
| 400m | .9170 / .9479 | .9176 / .9492 | **.9201±.0005 / .9504** | .9056 / .9420 |

20 ppm MAP@R 0.992-0.996, iso20 0.977-0.989 across all arms. lr 4e-4 with P170 x K2 is best at every scale
(margins 0.002-0.004 over the next arm, above the seed spread); lr 8e-4 is worst everywhere. Raw:
results/raw/finetune/contrastive/hp-scale-<scale>-<split>/. Other splits (test, oodval, mouse, human, yeast) scored, not yet tabulated.

## K154-P width profile (job 8880628, B=32 N=150, one tile, bf16, k=1, 2026-09-30)
Per layer, forward+backward (ms): single block = attention 3.1 + transition 1.6 = 4.7.
| layout | write-back | tri-mul | transition | readout+glue | pair update | step (gc off) | peak GB | step w/ tri-attn (gc on) vs without (gc on) |
|---|---|---|---|---|---|---|---|---|
| 64/64/16, both tri-mul, materialised write-back (old) | 9.9 | 45.5 + 42.6 | 10.5 | 7.0 | 115.5 | 1241 | 54.7 | 3813 vs 1670 |
| 64/64/16, outgoing, factored | 4.7 | 42.3 | 9.6 | 8.3 | 64.9 | 732 | 34.2 | 3178 vs 995 |
| 32/32/16, outgoing, factored | 2.7 | 20.9 | 4.9 | 5.0 | 33.5 | 418 | 18.9 | 2549 vs 561 |
| 16/16/16, outgoing, factored | 1.9 | 12.1 | 3.2 | 3.6 | 20.8 | 288 | 11.2 | 2294 vs 382 |
Pair update / single block: 25x (old) -> 14x -> 7x (32) -> 4.4x (16). The factored write-back is 2.1x faster
than the materialised one at the same widths. Triangle attention (4 heads x 16, fixed) does not shrink with
c_z: at 32 it makes the step 4.5x slower (gc on). Raw: results/raw/diag/width_profile/8880628_*.json.

## K66-C / K136-C: library search (the main metric) per scale, validation, unfiltered Hit@1 (mean of 3 seeds +- half range)
| scale | lr2e-4 P128 | lr4e-4 P128 | lr4e-4 P170 | lr8e-4 P128 |
|---|---|---|---|---|
| 25m | .702+-.018 | .726+-.014 | .708+-.065 | **.775+-.040** |
| 50m | - | - | .875+-.012 | - |
| 100m | .815+-.037 | .867+-.014 | **.885+-.008** | .824+-.020 |
| 200m | .900+-.007 | .915+-.009 | **.920+-.003** | .906+-.009 |
| 400m | .926+-.006 | **.946+-.003** | .940+-.005 | .938+-.006 |
20 ppm filtered library Hit@1 0.974-0.997, isotope-tolerant 0.929-0.992. Library Hit@1 is far noisier across seeds
than experimental MAP@R. Same raw files as above (library/* keys).

## C18 MassIVE-KB contrastive prep, full run (job 8879991, 2026-09-30): group statistics
Source chrisagrams/massive_kb_v1_shuffled rev 891f42f7, 30.5M spectra read; kept 25.4M after removing every
spectrum whose sequence (I/L collapsed) is in one of our evaluation sets (5.13M spectra, 31,283 sequences;
e.g. 94% of ms-contrastive-100k validation/test sequences and 98.5% of noble-human are in MassIVE-KB) and
1,170 with an unknown modification. Peptide-disjoint splits; spectra kept whole (max_peaks 1e6), mean 246 peaks.
| split | groups | spectra | singleton groups | median / p90 / p99 / max group size |
|---|---|---|---|---|
| train | 1,894,101 | 23,716,268 | 636,569 (2.7% of spectra) | 3 / 33 / 100 / 100 |
| validation | 67,072 | 824,126 | 22,778 | 3 / 32 / 100 / 100 |
| test | 66,538 | 835,055 | 22,447 | 3 / 34 / 100 / 100 |
Train histogram: 2: 277,732; 3: 132,478; 4-7: 240,628; 8-15: 181,568; 16-63: 317,135; 64-255: 107,991 groups.
Groups stop at 100 (the source keeps at most 100 spectra per peptide), so no very large groups exist.
Precursor m/z is theoretical (no measured value), so precursor-filter failure analyses are empty on this data.
Files: $S/data/massive-kb-contrastive/{manifest.json, group_sizes.parquet, overlap_report.md}.

## K157-P transformer baseline on the P2 validation set (job 8880677, eval_mlm, 2026-09-30)
Production 50m transformer (49.81M params), the full P2 validation split (67,933 spectra <= 150 peaks, 3,682,687
masked peaks, mask 0.5, seed 0, bf16): masked-peak eval loss 0.1169 at step 10k, 0.0769 at 50k, 0.0648 at 120k,
0.0554 at 540,423 (final). The P2 Pairformer runs stop at step 23,387 on the same schedule: interpolating in
log(step) between 10k and 50k gives ~0.096 for the transformer there (an estimate; no checkpoint at 23k).
Caveats: the transformer trained on spectra up to 512 peaks (this set is the <=150-peak subset) and K157-P
(held-out status of this shard) is unconfirmed. Raw: results/raw/diag/eval_mlm/8880677.json.

## K156-P P2 smoke (job 8880669, 2 nodes x 12 tiles, micro 3, global 72, 120 steps, 2026-09-30)
All three arms trained, evaluated (full P2 validation) and saved (checkpoint-120 + final). Steady step rate
(tqdm): k1 5.65 it/s (0.18 s/step), k1 + tri-attn 3.46 it/s (0.29 s), k5 8.64 it/s (0.12 s). Full validation
pass on 24 tiles: 25 s (k1), 59 s (tri-attn), 23 s (k5). Estimated 16-node runs (23,387 steps, if the step time
holds at 192 ranks): k1 ~1.2 h, tri-attn ~1.9 h, k5 ~0.75 h, ~60 node-h together plus evals.
Per-tile efficiency at micro 3 is low: k1 does 17 spectra/s/tile vs 76 at B=32 (profile 8880628). Fewer nodes
at a larger micro-batch (same global 576) would cost ~3-4x fewer node-hours at a similar wall time; kept 16
nodes as approved (K150 "fastest"), cost is small either way.
Eval loss at step 120 (warmup, lr ~1e-5): 0.817 for all three arms.

## K156-P first P2 result: Pairformer k=5 (job 8880727) vs the transformer, same masks (2026-09-30)
p2-cz32-k5 (45.27M params; c_z=c_t=32, c_o=16, outgoing tri-mul, pair updates on layers 1 and 6), 23,387 steps
x 576 = 13.47M spectra (0.5 epoch of the cap-150 half), 16 nodes, 49.5 min wall (~13 node-h), exit 0.
eval_mlm on the P2 validation set (identical mask and input digests to the transformer job 8880677):
  Pairformer k5 @ 23,387: 0.0926
  transformer 50m @ 10k: 0.1169, @ 50k: 0.0769 (log-step interpolation @ 23,387: ~0.096), final: 0.0554
In-training eval (random masks) ended at 0.0923. So at the same step of the same LR schedule the k5 Pairformer is
slightly below the transformer (by ~0.003-0.004; the transformer value is interpolated, and the cap-150 caveat
applies). Raw: results/raw/diag/eval_mlm/p2-k5-8880727.json; run $S/runs/p2/p2-cz32-k5-8880727.

## K156-P P2 k=1 (job 8880726), same masks (2026-09-30)
p2-cz32-k1 (45.57M params, pair update every layer), 23,387 steps, 16 nodes, 68.3 min wall (~18 node-h), exit 0.
eval_mlm (digest 89e46099fdba, as 8880677): 0.0907.
| model | eval loss @ 23,387 | wall (16 nodes) |
|---|---|---|
| Pairformer k1 | 0.0907 | 68 min |
| Pairformer k5 | 0.0926 | 50 min |
| transformer 50m (interpolated 10k-50k) | ~0.096 | - |
Pair updates on every layer buy 0.002 over two updates for 1.4x the wall time. Raw: results/raw/diag/eval_mlm/p2-k1-8880726.json.

## K156-P P2 tri-attn (job 8880728): no gain per step, 2.1x the cost (2026-09-30)
p2-cz32-k1-triattn (k1 + triangle attention), 23,387 steps, 16 nodes, train 136 min (~37 node-h), exit 0.
In-training eval (trainer masks, same validation split) at the end, all three P2 arms:
| arm | final in-training eval | eval_mlm (digest 89e46099fdba) | train time (16 nodes) |
|---|---|---|---|
| k1 | 0.0910 | 0.0907 | 64 min |
| k5 | 0.0923 | 0.0926 | 46 min |
| k1 + tri-attn | 0.0929 | NaN (bug, see TODO) | 136 min |
Tri-attn led early (6k: 0.154 vs 0.158), tied at 10k, and fell behind from 12k (16k: 0.1089 vs 0.1067).
-> Not useful at this size/budget: worst of the three per step, 2.1x k1's time. eval_mlm returned NaN on its
final (8881426) though training eval was finite -- likely a masking/padding edge case in triangle attention at
eval_mlm's batch shapes; to fix before any tri-attn number is trusted from eval_mlm.

## K167-I torch.compile works on frameworks/2026.1.0: k5 2.0x faster per step (held debug node 8882085, 2026-09-30)
.venv-2026 (torch 2.13, Intel triton_xpu 3.7.2), one node, tiles 2-11 (10 ranks), micro 24, global 480 (2 accumulation
steps), P2 configs, step time over steps 150-300. Raw: $S/runs/k168/k168c/summary.txt.
| arm | eager | compiled | first step (compile) |
|---|---|---|---|
| k5, pad to 150 | 0.313 s | 0.153 s (2.05x) | 106 s vs 7 s |
| k5, pad to longest (production) | 0.313 s | - | 5 s |
| k1, pad to 150 | (old env: ~0.75 s equivalent) | 0.267 s (~2.8x) | 119 s |
Fixed padding (pad_to_multiple_of 150 = one shape) costs nothing eager at cap 150. Earlier failures: 2025.3.1 .venv's
upstream triton (no intel backend), then IGC crashes with the shim (K162); on 2026.1.0 both gone. Not yet checked:
compiled vs eager loss curves agree (numerics), and multi-node compiled runs.

## K168-C 400m consensus seed-0 crash = out of GPU memory surfacing as a page fault (2026-09-30)
Held node 8882085, tiles 0-1 = one card, xpu-smi trace. As-is (trimmed GradCache chunks): card memory climbs to 95.9 GB
(~65 GB on the crashing tile, limit 64) and the run faults at the first evaluation (Python stack: prediction_step ->
SupCon loss), then the card drops to 30 GB = the untrimmed run alone. Untrimmed (--gradcache_trim_padding false): ~30 GB
but 82 s/step vs ~30 (2.7x slower). Cause: trimming cuts each chunk to its own longest spectrum, so the allocator keeps
one cached block per chunk width the seed happened to draw; consensus spectra are longer, seed 0 draws a bad mix, and
the evaluation's allocations on top exceed the tile (XPU reports it as "Segmentation fault from GPU ... NotPresent"
instead of OOM, as in FT7). Fix (memory only, numerics unchanged): free the XPU cache before every evaluation
(EvaluationCacheCallback registered in finetune_contrastive + before the post-training evaluation). Retest running.

## K173-P concurrent pair/single streams: NO overlap on XPU as run (job 8882322, 2026-10-01)
One tile (Max 1550), .venv-2026 (torch 2.13), B 24, N 150, P2 widths (c_z 32, outgoing, factored), bf16, fwd+bwd median.
| layers | k | lag1 sequential | lag1 concurrent | lag0 sequential | pair updates (lag1/lag0) |
|---|---|---|---|---|---|
| 10 | 1 | 276.4 ms | 276.4 | 298.2 | 9 / 10 |
| 10 | 2 | 173.7 | 174.2 | 194.0 | 4 / 5 |
| 10 | 3 | 153.7 | 153.9 | 173.3 | 3 / 4 |
| 10 | 5 | 114.0 | 113.7 | 132.9 | 1 / 2 |
| 10 | 7 | 113.6 | 114.0 | 132.8 | 1 / 2 |
| 14 | 1 | 382.1 | 381.7 | 403.5 | 13 / 14 |
| 14 | 2 | 236.5 | 236.9 | 257.2 | 6 / 7 |
| 14 | 3 | 196.4 | 196.5 | 216.4 | 4 / 5 |
| 14 | 5 | 156.3 | 156.1 | 175.5 | 2 / 3 |
| 14 | 7 | 136.4 | 136.1 | 155.1 | 1 / 2 |
- Concurrent == sequential to <0.5 ms everywhere: the side stream did not run alongside the main one. lag1 saves only
  the one pair update it does not build (~20 ms). Numerics: concurrent matches sequential (fp32 grad rel <=2.6e-7).
- Cost model from these rows: one pair update ~20.5 ms, one single layer (with its bias readout) ~5.7 ms, rest ~36 ms
  -> pair/single ~3.6x at B 24 on the 2026 stack -> time-balanced k ~3-4.
- Likely reasons: Aurora tiles run in 1-CCS mode by default (one compute engine per tile, so two queues serialise),
  and at B 24 each kernel already fills the tile. Raw: results/raw/diag/k172_overlap/8882322.{json,log}.

## K174-P two compute engines per tile: still no overlap, and 1.6x slower (job 8882544, 2026-10-01)
Same test as K173 (14 layers, B 24, N 150, .venv-2026), ZEX_NUMBER_OF_CCS 0:1 (default) vs 0:2. PyTorch sees 1 device both ways.
| k | 1 CCS: lag1 seq / conc / lag0 | 2 CCS: lag1 seq / conc / lag0 |
|---|---|---|
| 2 | 239.5 / 239.4 / 260.8 ms | 389.5 / 389.2 / 424.0 ms |
| 3 | 198.1 / 198.3 / 218.4 | 319.5 / 319.8 / 353.6 |
| 5 | 157.2 / 157.7 / 177.0 | 250.2 / 250.3 / 283.4 |
The 2-CCS mode took effect (everything ~1.6x slower = the work runs on half the tile's compute), but concurrent ==
sequential again: PyTorch's XPU streams all submit to the same engine, so the second engine sits idle. Same-tile
overlap is not reachable through torch.xpu streams on this stack. Raw: results/raw/diag/k172_overlap/8882544-ccs{1,2}.*

## K176-P pair stream on a second tile: real overlap, -14 to -23% step time; 2-CCS devices not exposed (job 8882596, 2026-10-01)
14 layers, B 24, N 150, .venv-2026. "tiles": main = tile 0, pair updates = tile 1 of the same card (FLAT, ZE_AFFINITY_MASK=0,1).
| k | lag1 seq (1 tile) | lag1 same-tile streams | lag1 two tiles | lag0 seq | peak GB (2 tiles: main / pair) | copies per forward |
|---|---|---|---|---|---|---|
| 2 | 237.2 ms | 237.3 | 184.7 (-22%) | 258.1 | 4.3 / 6.6 | 494 MB |
| 3 | 196.6 | 196.6 | 152.2 (-23%) | 216.3 | 4.2 / 4.5 | 341 MB |
| 5 | 156.0 | 156.0 | 133.5 (-14%) | 175.7 | 4.2 / 2.4 | 188 MB |
- First real overlap. Numerics match (fp32 grad rel <=2.6e-7; bf16 grad rel up to 4e-3 vs 1e-4 for same-device, to watch).
- Cost: two tiles per model copy. Tile-time per batch k3: 2 x 152 = 304 ms vs 197 ms on one tile -> 1.55x MORE compute per
  sample than plain data parallel; it buys wall time, not node-hours. Ideal overlap at k3 would be ~max(pair ~82+copies,
  single ~116) ms; 152 achieved -> copies (76.5 MB/round + backward) and the per-round barrier eat part of it.
- ccs2 (one tile split into 2 engines, each its own device): every selector tried exposed 0 devices -> no data; needs
  interactive debugging of the ZEX_NUMBER_OF_CCS / ONEAPI_DEVICE_SELECTOR / ZE_AFFINITY_MASK combination.

## K66-C / K136-C final-checkpoint arms on the other sets (mean of 3 seeds, experimental MAP@R, unfiltered)
| set | scale | lr2e-4 P128 | lr4e-4 P128 | lr4e-4 P170 | lr8e-4 P128 |
|---|---|---|---|---|---|
| test | 25m / 100m / 200m / 400m | .870 / .900 / .909 / .920 | .871 / .900 / .908 / .921 | **.875 / .903 / .912 / .923** | .849 / .883 / .891 / .910 |
| mouse | 25m / 100m / 200m / 400m | .839 / .856 / .859 / .867 | .837 / .855 / .857 / .867 | **.842 / .856 / .861 / .869** | .822 / .844 / .848 / .859 |
| human | 25m / 100m / 200m / 400m | .883 / .896 / .898 / .901 | .885 / .895 / .897 / .902 | .886 / .895 / .898 / .902 | .880 / .890 / .893 / .898 |
| oodval (8 species) | 25m / 100m / 200m / 400m | .786 / .760 / .818 / .777 | .781 / .758 / .792 / .772 | .794 / .768 / .802 / .767 | .762 / .773 / .715 / .784 |
| yeast | 25m / 100m / 200m / 400m | .728 / .547 / .662 / .598 | .727 / .571 / .603 / .623 | .736 / .566 / .628 / .606 | .707 / .637 / .546 / .648 |
50m P170: test .892, mouse .851, human .893, oodval .777, yeast .624. Library Hit@1 on test (the library exists only
for ms-contrastive-100k): 400m .925 / .944 / .938 / .936, 200m .898 / .914 / .917 / .905 (same arm order).
Test, mouse and human agree with validation (P170 best or tied, scale helps). The two out-of-distribution
species sets do NOT: no arm wins consistently, scale does not help (yeast: 25m .73 vs 400m .60-.65), and the
spread between arms is large. So the recipe choice is supported in-distribution only; OOD behaviour is an open
question (it echoes the earlier "OOD/yeast disagree with validation" note in K134).
