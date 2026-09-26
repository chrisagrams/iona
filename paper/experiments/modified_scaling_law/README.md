# Pretraining scaling law with an irreducible-loss term

The joint parameter/data scaling law of the pretraining runs, refitted in the Chinchilla form
(Hoffmann et al., 2022) with an irreducible loss E, and compared with the same law without E (the form
used by `scripts/plot_pretrain_efficiency_powerlaw.py` / `plots/pretrain_efficiency_powerlaw.pdf` at the
repository root, whose fit this reproduces exactly).

## Files

| file | content |
|---|---|
| `scaling_law_frontier_E_vs_noE.png` | eval loss vs compute for the five runs, with the compute-optimal frontier implied by each fit (orange dotted: without E; black dashed: with E) |
| `scaling_law_loss_vs_flops.png` | eval loss vs compute, each model size with its fitted curve (with E) dashed, plus the fitted frontier |
| `scaling_law_fit_loglog.png` | the fit with E along both axes: loss vs data per model size (left), loss vs model size at 10k/50k/200k/540k steps (right) |
| `pretrain_eval_loss.csv` | every logged evaluation of the five production runs (`model`, `run`, `parameters`, `step`, `peaks_seen`, `flops`, `eval_loss`, `post_warmup`) |
| `fit_parameters.csv` | both fits: parameters, log-space R², AIC, held-out errors |
| `final_loss_vs_parameters.png`, `final_loss_fit.csv` | end-of-training loss (540,423 steps) vs parameters, fitted with a pure power law and with a power law plus floor |
| `equal_steps.csv` | every size's eval loss at each logged step (equal data), and the best size |
| `equal_compute.csv` | every size's eval loss at shared compute budgets (log-log interpolated between logged steps), and the best size |
| `plot_modified_scaling_law.py` | fits and draws everything from `pretrain_eval_loss.csv` (`python plot_modified_scaling_law.py`; matplotlib, numpy, scipy) |

## Data and conventions

- The W&B export in `../pretrain/runs/`: msdelta 25M, 50M, 100M, 200M, 400M production runs, eval loss
  at 11 steps from 500 to 540,423 (3 epochs of the same corpus, one 540k-step cosine schedule per run).
- Fitted on the 40 post-warm-up evaluations (step > 2,000: 5k-540k, 8 per run), in log space.
- N = parameters; D = peaks seen = steps x 512 spectra x 512 peaks (every spectrum counted at the
  512-peak maximum, including padding); compute C = 6 N D. Same conventions as the repository's
  existing power-law figure. (The trainer's own FLOP counter implies about 271 peaks per spectrum, a
  constant factor of ~1.9 lower; that shifts all points sideways on a log axis and changes only B.)

## Fits

L(N, D) = E + A (N / 10⁸)^−α + B (D / 10¹⁰)^−β

| | E | A | α | B | β | log R² | AIC |
|---|---|---|---|---|---|---|---|
| without E | – | 0.053 | 0.14 | 0.029 | 0.57 | 0.989 | −260.9 |
| **with E** | **0.039 ± 0.004** | 0.013 | 0.55 ± 0.15 | 0.029 | 0.57 ± 0.03 | 0.991 | **−267.7** |

Held-out checks (largest absolute error on the held-out points):

| | without E | with E |
|---|---|---|
| 400M fitted from the other four sizes | 0.0067 | 0.0053 |
| final two checkpoints (500k, 540k) of every run | 0.0063 | 0.0052 |

Compute-optimal frontier implied by each fit:

| compute (FLOPs) | without E | with E |
|---|---|---|
| 1.8e20 (largest run) | 0.0528 | 0.0536 |
| 4e20 | 0.0483 | 0.0507 |
| 1e21 | 0.0436 | 0.0481 |
| 1e22 | 0.0338 | 0.0438 |

- Reading: E is well determined (about ten standard errors from zero), lowers AIC by 6.8 and improves
  both held-out predictions. Without it the size term absorbs the floor and its exponent is biased low
  (α 0.14 vs 0.55); the data exponent β is unchanged. Inside the data the two frontiers are close; they
  diverge in extrapolation (the form without E predicts steady returns, the form with E diminishing
  returns towards E ≈ 0.039). The 400M run's final checkpoints flatten above both frontiers.

## End of training: loss vs model size

One point per size, all at 540,423 steps (the same data), `final_loss_fit.csv`:

| | E | A | α | RMSE | largest residual |
|---|---|---|---|---|---|
| L = A (N/10⁸)^−α | – | 0.059 | 0.089 ± 0.015 | 0.0015 | +0.0018 (25M), +0.0017 (400M) |
| **L = E + A (N/10⁸)^−α** | **0.052 ± 0.001** | 0.005 | 0.83 ± 0.16 | **0.0004** | 0.0006 |

The pure power law misses systematically (above at both ends, below in between); with a floor every
point is within 0.0006. 400M (0.0543) is within 3% of the floor, so the step from 200M gains only 1.2%
(per doubling: 25M→50M −10.7%, 50M→100M −5.1%, 100M→200M −5.9%, 200M→400M −1.2%). At a fixed data
budget this floor is the irreducible loss plus the data term; separating them needs runs on different
amounts of data. (This floor, 0.052, is higher than the joint fit's E = 0.039 because the joint fit also
has a data term.)

## Crossovers (which size is best at a given budget)

- Equal steps (`equal_steps.csv`, equal data): from 10k steps on the larger model is better at every
  step; 400M is ahead of 200M from 10k (−1.8%) to 200k (−3.6%), narrowing to −1.2% at the end. At 5k
  (first post-warm-up evaluation) 400M is marginally behind 200M (0.1325 vs 0.1313).
- Equal compute (`equal_compute.csv`): the best size moves up with compute, 25M up to about 4e18 FLOPs,
  50M to about 2e19, then 200M, and 400M only beyond 200M's final budget (400M's final loss 0.0543 is
  below 200M's 0.0549, at twice the compute). 100M is never the best in these budgets. This is the
  pattern of Kaplan et al. (2020): each size is compute-optimal over a range of budgets.
- Equal-compute comparisons use mid-schedule checkpoints of fixed 540k-step cosine schedules, which
  disadvantage the larger model at a given budget (it is earlier in its schedule).

## Caveats

- The data axis comes from intermediate checkpoints of one run per size, all on the same 540k-step
  cosine schedule; mid-schedule losses are higher than a run sized to that budget would reach, and the
  end-of-schedule annealing drop is not modelled (the smaller runs' final checkpoints sit below their
  curves). Chinchilla fits separate runs per data budget; with one run per size this is the available
  approximation.
- D counts repeated data (3 epochs of one corpus).
- α is loosely determined (± 0.15) with five model sizes.

Provenance: msdelta repository session, from `../pretrain/runs/*.{csv,json}` (the exported W&B runs).
