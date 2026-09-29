# C27-C card (DRAFT, needs approval): weight the consensus spectrum more heavily in training

Status: APPROVED as drafted 2026-09-29 ("Yes run your proposal"). Implementation on branch c27-prep.

## Question
Does making the consensus spectrum the "canonical" member of each group -- sampling it more often, as if
it were in the group several times -- improve library search (experimental query vs consensus-only
library), and at what cost to experimental-only retrieval?

## Background
- C20 (job 8873850, old recipe lr 1e-4, P85xK3, 50m): consensus as a 4th member, sampled uniformly.
  Experimental MAP@R unchanged; library Hit@1 0.70-0.73 -> 0.93-0.94 (K100, validation and test).
- K66-C / K68 recipe (current): lr 4e-4, P128xK2, experimental spectra only. With K = 2 each group
  contributes ONE positive pair per batch, so what that pair contains is what the model learns.
- With uniform sampling from {consensus, e1, e2, e3} and K = 2, a pair contains the consensus with
  probability 1/2.

## Proposed change (code)
`--consensus_weight w` (float, default 1.0 = today's behavior; ignored without `--include_consensus`):
GroupBatchSampler draws the K members of a group WITHOUT replacement, the consensus with weight w and
each experimental spectrum with weight 1. Without replacement matters: a consensus paired with itself is
a free, meaningless positive (the reason K <= group size is enforced today). "As if it's there w times":
w = 3 makes the consensus as likely as the three experimental spectra together.
P(pair contains the consensus) at K = 2: w = 1 -> 0.50, w = 3 -> 0.80, w = inf ("always") -> 1.00.
(Alternative, not proposed: weight exp-consensus positive pairs in the SupCon loss; same intent, touches
the loss instead of the sampler.)

## Proposed arms (50m, final checkpoint 540,423, current K66 recipe otherwise)
| arm | include_consensus | consensus_weight | note |
|---|---|---|---|
| ref_exp | false | - | NOT retrained: the existing K53 50m lr 4e-4 P128xK2 runs, re-scored with library search |
| cons_w1 | true | 1 | C20 at the new recipe (uniform) |
| cons_w3 | true | 3 | "3 times" |
| cons_always | true | inf | every pair = consensus + one experimental |
| cons_w3_kl0 | true | 3, KL 0 | C20's best arm was KL 0 (K100: K3 KL0 >= K3) |
3 seeds each -> 12 trained arms = one capacity node (1 tile per arm), ~6 h at 50m (walltime 10 h).
Validation: 20-step debug smoke of cons_w3 and cons_always first (after the code change + unit test:
sampler never repeats a member; empirical consensus frequency matches the formula).

## Scoring
Same as K66-C (validation + OOD for selection; test, mouse, human, yeast for reporting), with and
without precursor filter (none / 20 ppm / iso 20 ppm), passes and failures, AND library search
(Hit@1/5, MRR, rank stats; ties = misses) -- the metric this card is about. ref_exp re-scored on the
same sets.

## Decision rule (proposal)
Pick by validation library Hit@1, provided validation experimental MAP@R drops by no more than the
seed spread vs ref_exp; report both.

## Open for the user (K139-C)
a) arms / weights above (w values; include the KL0 arm?); b) sampler weighting vs loss weighting;
c) decision rule; d) seeds (3).
