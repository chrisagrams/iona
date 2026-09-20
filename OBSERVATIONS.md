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

## Denoise is the line that works; the reranker does not need a neural embedding

50m grid best test AUROC **0.9320** over 216 arms; 100m best **0.9403** over 12. A
hand-built feature rescorer reaches **hit@1 0.889** on fragment coverage, mass error and
spectrum quality with no neural embedding at all -- and adding the embedding cosine made
it WORSE by 0.109 hit@1 over five paired seeds, because every candidate for a spectrum is
scored against one shared cached vector, so its errors correlate within a spectrum.

Neither grid ranking is known to be real yet: the 50m top eight span 0.0023 test AUROC
and the 100m top five span 0.0012, on one seed each. FT5 is what settles that.
