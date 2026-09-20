# Observations

Things we believe and why, separate from `STATUS.md` (what is running) and `TODO.md`
(what is broken). Each entry says what would overturn it.

---

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

## Fine-tuning is about how much EXTRA training is needed, not about a ceiling

Worth stating because the random-init control is easy to over-read. It does not show
that a randomly initialised encoder could never learn this task; with enough data and
steps it very likely could. What it shows is the thing that actually matters in
practice: at the budget we can afford, pretrained reaches 7.83 and random reaches
nothing. That IS the value of pretraining -- less extra training to a good result --
and the budget-limited comparison is the relevant one, not a confound in it.

---

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

## Denoise is the line that works; the reranker does not need a neural embedding

50m grid best test AUROC **0.9320** over 216 arms; 100m best **0.9403** over 12. A
hand-built feature rescorer reaches **hit@1 0.889** on fragment coverage, mass error and
spectrum quality with no neural embedding at all -- and adding the embedding cosine made
it WORSE by 0.109 hit@1 over five paired seeds, because every candidate for a spectrum is
scored against one shared cached vector, so its errors correlate within a spectrum.

Neither grid ranking is known to be real yet: the 50m top eight span 0.0023 test AUROC
and the 100m top five span 0.0012, on one seed each. FT5 is what settles that.
