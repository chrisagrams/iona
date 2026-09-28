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
| 2026-09-27 | K79-C | Implement the library-search evaluation (consensus-only library); ms-contrastive-100k val/test; C20 + K66-C models | user |
| 2026-09-27 | K78-C | Building consensus libraries for mouse/human/yeast: later | user |
| 2026-09-27 | K80-A | Measure how correlated loss-based and Hit@1-based checkpoint selection are, and their effect on test; user prefers loss selection; decide after results (PLAN A10) | user |
| 2026-09-27 | K76-C / K81-C | Always test on the SAME yeast set: the full nine_yeast (86,184), frozen with CANONICAL.txt + sha256; scored on capacity (3 h walltime) | user |
| 2026-09-27 | K82-S | Remove the data/real-sample symlinks (done; targets untouched) | user |
| 2026-09-28 | K55-C | Which filtered number to headline: decide later | user |
| 2026-09-28 | K77-A | A8 / mass-aware peptide encoder follow-up: later, once the new C models exist | user |
| 2026-09-28 | K63-I | User will read the DAG build and confirm later (remind); write a spec sheet so it can be generalised into a feature later | user |
| 2026-09-28 | K84-I | First live trial of the DAG scheduler: merge i2-dag into dev_finetune_02 and run the 2-node trial (smoke then scoring), ticks by hand | user |
| 2026-09-28 | K92-P | Absolute m/z in the Pairformer tokens is NOT a problem: if including it improves the model, that is an improvement (the control arm may still be run) | user |
| 2026-09-28 | K94-P | Make fine-tuning loaders fail when checkpoint weights are missing (no silent random init) | user |
| 2026-09-28 | K96-S | Input-normalisation leak: revisit later (OPEN_QUESTIONS.md) | user |
| 2026-09-28 | Pairformer ablations | In principle (each still needs a card): #2 m/z control OK, #3 matched params vs matched compute -- very important, #4 pair input features OK, #5 peaks cap OK, #6 pair width OK, #8 triangle attention: find how to make it win; #11 precursor features: make the model work with AND without them | user |
| 2026-09-28 | K63-I etc. | Open questions are logged in notes/OPEN_QUESTIONS.md with full context; K86-K91 parked there | user |
| 2026-09-28 | K97-C | Library search: drop library/MAP@R; report average Hit@1, Hit@5 (and MRR) plus statistics of R = rank of the true library entry among the whole library (mean/median/p90/p99/max, fraction with R <= 1/5/10/100; filter-excluded counted separately) | user |
| 2026-09-28 | K98-C | Ties count as misses (conservative) | user |
| 2026-09-28 | K99-C | Library search only on datasets that have consensus spectra; queries without a consensus excluded (counted); consensus-only groups stay as distractors | user |
| 2026-09-28 | K90-S | Don't require merging to run code: jobs should take their code snapshot from a given git commit/branch (being built on branch k90-code-ref, with one tiny live check); tell the user any reason not to | user |
| 2026-09-28 | K55-C | No headline numbers for now; keep tracking all filtered numbers | user |
| 2026-09-28 | K83-C | Add the ±1.1 Da window as an evaluation filter (testing only; not in training) | user |
| 2026-09-28 | K78-C | No consensus building now; likely delegated | user |
| 2026-09-28 | C18-C | Write the MassIVE-KB prep script now (code; running it needs a card) | user |
| 2026-09-28 | K93-P | Write a Pairformer architecture note (notes/PAIRFORMER.md) for the user to study | user |
| 2026-09-28 | K101-P | Run Stage 0 (sanity run of both architectures) -- exact card being prepared (notes/P1_stage0_card.md) | user |

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
