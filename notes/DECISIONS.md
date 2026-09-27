# Decisions log

Every decision about what to run, which settings, what becomes a default, and how the project
is organised. Only the user makes these (2026-09-27: "never pick settings yourself from now on,
I need to approve them and all decisions before being submitted, and we need to record these
decisions").

Process:
1. Before anything is submitted (debug smoke included), an **experiment card** is proposed with an
   ID: question, what is fixed, every varied setting, scales / seeds, scoring, how it is validated
   on debug, time and cost, risks, what comes after.
2. The user approves, edits or rejects it. Nothing is submitted until approved, and only the
   approved settings are run.
3. The decision is logged here (date, ID, decision, the user's words where short), and the
   approved card text goes into the grid generator's docstring.

IDs carry a track suffix: -C contrastive (spectrum encoder), -A alignment (peptide encoder),
-D denoise, -R rescoring, -I infrastructure, -P Pairformer, -S cross-cutting.

## Log

| date (UTC) | ID | decision | by |
|---|---|---|---|
| 2026-09-26 | – | Contrastive: queue C8 (sigmoid loss) and C19 (same-mass batches) on ms-contrastive-100k alone; then an HP search per scale, "start only on 50m" | user |
| 2026-09-26 | K7 / K22 | Mixed batches: "75% same-mass, 25% random" (user); two-region design preferred over the per-anchor mask, "also try it at a 50/50 split" | user |
| 2026-09-27 | K15 | Negative-only random singles: not worth it; recorded as considered, never tested | user |
| 2026-09-27 | K10 / K11 | dev_finetune frozen at submission (post-deadline commits kept); dev_finetune_02 is the working branch; rename to "peptide encoder" / "spectrum encoder" | user |
| 2026-09-27 | K17 / P1 | Pairformer port: HF-compliant; unit tests AND a short debug pretraining comparison; future work uses the khuss scratch, never kelhus2 | user |
| 2026-09-27 | K29 | Per-job code snapshots so branch switches never affect jobs | user |
| 2026-09-27 | K37 | DeepSpeed/pyarrow fix only where our own code needs it; do not change working master code | user |
| 2026-09-27 | K38 | Fix the resume batch-order defect now | user |
| 2026-09-27 | K39 / K40 | Cutover of the main checkout after the mix job; new work from dev_finetune_02 immediately | user |
| 2026-09-27 | K45 | Download the MassIVE-KB test split (no content verification) | user |
| 2026-09-27 | K49 | Same-mass batches become the training default; then wait on the two-region models | user |
| 2026-09-27 | K53 | 50m follow-up: lr 8e-4, 1.6e-3, lr 4e-4 + P128×K2, lr 4e-4 + KL 1 (settings proposed by Claude, approved) | user |
| 2026-09-27 | K54 | Train with consensus spectra before drawing conclusions about the KL / consensus interaction | user |
| 2026-09-27 | – | Metrics: always report with and without precursor filtering, on filter passes and failures | user |
| 2026-09-27 | I2 | Job DAG + scheduler: notifications dir; multi-queue allocation; RESUME_JOB on partial failure; hung-job detection; runaway protection; atomic outputs; one DAG for all jobs (no per-DAG code pin). Build vs adopt still open (K63-I) | user |
| 2026-09-27 | K67-C | Pure same-mass batches stay the default after C21 | user |
| 2026-09-27 | K68-C | New single-dataset recipe: lr 4e-4 + P128×K2 | user |
| 2026-09-27 | K72-S | Experiment cards and this log; no settings chosen without approval | user |
| 2026-09-27 | K66-C | **Approved as carded:** per-scale search 25m/100m/200m/400m @540k; lr {2e-4, 4e-4, 8e-4} × P128×K2 and lr 4e-4 × P170×K2; 3 seeds; debug smoke (25m + 400m, 20 steps); walltime 14 h for 400m, 10 h others; scored on validation / OOD / mouse / human with filtered metrics (`sweeps/make_hp_scale.py`) | user |
| 2026-09-27 | P1-P | Go ahead with the Pairformer port (code + unit tests; the debug comparison run needs its own approved card) | user |
| 2026-09-27 | A | Alignment waits for the optimal C models | user |
| 2026-09-27 | C18-C | MassIVE-KB work is low priority, later; user is ~95% certain the datasets are distinct from ours | user |
| 2026-09-27 | K74-C | Validation is for selection; test is for reporting only; never select on test | user |
| 2026-09-27 | K56-C | Default contrastive testing: ms-contrastive-100k (validation for selection, test for reporting), human, mouse, yeast -- human/mouse/yeast have measured precursors, so the filter's mistakes are scored there | user |
| 2026-09-27 | K57-S | Remove the leftover folders (portable_eval cache file, empty results/figures) -- done | user |
| 2026-09-27 | K63-I | Build the thin custom DAG layer (not Balsam/Parsl/Snakemake) | user |
| 2026-09-27 | K75-C | Mark the cases where something was selected on test (PLAN.md); do not rescore old models | user |
| 2026-09-27 | C16-C | Archive C16 (superseded) | user |
| 2026-09-27 | K4-S | Move data/synthetic to /flare (done: $S/shelf/data-synthetic, checksums verified, staged removal) | user |
| 2026-09-27 | K76-C | Yeast in the default contrastive test set = the FULL nine-species yeast set (nine_yeast, 86k spectra) | user |
| 2026-09-27 | K78-C | Library search matters (it is what reranking improves): measure retrieval against a consensus-only library | user |
| 2026-09-27 | K77-A | (1) cross-modal eval gets the with/without-filter + pass/fail split: yes; (2) keep selecting alignment on validation LOSS for now (user asks how much it differs in practice) | user |

## Settings chosen WITHOUT explicit approval (before this log existed)

Recorded so the record is honest; their results stand but were not the user's choices.

| experiment | what Claude picked |
|---|---|
| C21 mix (sweep-mix) | added within50 and between75 arms beyond the requested 75/25; 3 seeds |
| C21 two-region (sweep-regions) | 3 seeds; P85 |
| C20 (sweep-c20) | the three arms (consensus K3, K4 at P64, K3 + KL 0), lr 1e-4, 3 seeds |
| C23 hp-single (50m) | the 12 one-factor arms and their values |
| K6 / C24 filter analyses | models, datasets, filter widths (100/25/5/1/0.1 Da; 100/20/10 ppm), isotope k range −2..2 |

## Pending approval

| ID | proposal |
|---|---|
