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
| 2026-09-28 | K103-I | Claude may commit the DAG trial files and run the live trial (done: bd8fa7f; first live tick submitted 8875624) | user |
| 2026-09-28 | K105-S | Strict loading is the default in the model classes (every caller protected; explicit opt-out); covers K104-S | user |
| 2026-09-28 | K90-S (addendum) | A resumed job takes the configs of its ORIGINAL run and the code of the commit it was launched from | user |
| 2026-09-28 | K90-S / K106-S | Merge outstanding agent branches as soon as tests pass (done: 8 branches merged, worktrees removed); jobs run from any commit via pbs/qsub_ref; resume uses the original run's configs + commit | user |
| 2026-09-28 | P (Fourier) | Pairformer adopts master's Fourier settings (256 freqs, 1e-3 to 190); change later if studied | user |
| 2026-09-28 | K91-P | Decide after the user studies notes/PAIRFORMER.md | user |
| 2026-09-28 | K107-P | Stage 0: download MSConsensus-100M from Gaolaboratory to /flare (fits: 8.6/10 TB used); cap 150 OK for testing (final likely 512); add W&B logging; agent's unsourced choices OK for this test (real runs decide/HP-search them) | user |
| 2026-09-28 | C18-C | Review the MassIVE-KB prep later (remind) | user |
| 2026-09-28 | K109-S | Keep the three older worktrees (msdelta-pr, msdelta-rerank, msdelta-denoise-pr) for now | user |
| 2026-09-28 | K107-P (data) | MSConsensus-100M downloaded to /flare (Gaolaboratory, revision 78b3e74, 190 GB) | user |
| 2026-09-28 | K91-P / K113-P | Drop the Pairformer's separate mass-defect feature for now (default off; Stage 0 card updated) | user |
| 2026-09-28 | K102-P | Try (1) chunk checkpointing and (2) PyTorch fused scaled_dot_product_attention for triangle attention first; then decide whether a custom kernel is worth it | user |
| 2026-09-28 | K112-C | Spectrum-only database search: options = (a) our own intensity-prediction task (predict all intensities from fragment m/z) -> embed -> search; (b) generated consensus spectra; (c) peptides directly once alignment is done. Implication (user): consensus spectra must then matter in our training | user |
| 2026-09-28 | K111-S | Merge in a scratch worktree, test there, then fast-forward the main checkout; AND close the remaining window (race-free snapshots: git archive of the resolved commit when clean; locked copy when dirty) -- being built on k111-atomic-snapshot | user |
| 2026-09-28 | K115-P | SDPA should become the default for triangle attention; run the bf16 comparison first (incl. an all-bf16 variant) -- running on branch k115-bf16-check | user |
| 2026-09-28 | K115-P (done) | bf16 check passed (job 8875855): SDPA at least as accurate as the naive bf16 path; pair_tri_attn_impl default = "sdpa" | user |
| 2026-09-28 | Stage 0 prep | Green light to prepare Stage 0 (preprocessing + both arms + runbook); submit after maintenance | user |
| 2026-09-28 | K110-S | Checkpoint inventory waits until after maintenance | user |
| 2026-09-28 | K66-C write-up | Write up 100m/200m now (done: OBSERVATIONS) | user |
| 2026-09-28 | K114-P | Run a profiling job to get per-block Pairformer speed numbers (standalone pbs/diag script; prepared now, submitted after maintenance) | user |
| 2026-09-28 | K119-P | Run the FlexAttention support test on XPU (job-only Intel Triton path; prepared now, submitted after maintenance) | user |
| 2026-09-29 | K123-I | Validate the fixed environment on a compute node (e2e+golden 8876790: 10/10 passed; full suite incl. device 8876791 running) | user |
| 2026-09-29 | K124-C | Resubmit the K66-C 400m / 25m training (same card). Final IDs 8876832 (400m), 8876833 (25m); a first afterok-chained attempt (8876824/25) was deleted by PBS because the full-suite validation job exits 1 on the 4 known failures although validation passed (e2e+golden 10/10, device 11/11, 897 passed) | user |
| 2026-09-29 | K125-S | Move to the new frameworks/2026.1.0 venv when it no longer risks comparison issues within a task (i.e. between experiment phases) -- schedule it then (PLAN I3) | user |
| 2026-09-29 | K126-S | Fix the 4 long-standing test failures. Turned out to be a false positive in the TEST (the scripts export all variables on one line); test fixed, suite fully green (901 passed) | user |
| 2026-09-29 | K127-S | Merge k114-k119-prep and stage0-prep into dev_finetune_02 now (done: f1b06b2; 919 passed, 36 skipped) | user |
| 2026-09-29 | K122-S | Option (b): stage0-prep merged WITHOUT the train.py/training_args.py block-timing hook (block_timing.py + its test dropped); K114's standalone profiler covers per-block timing | user |
| 2026-09-29 | K100-C | Run library search on the C20 models (validation 8877125, test 8877126; new OUT_DIRs c20-{validation,test}-lib because the old C20 JSONs predate library metrics and would be skipped) | user |
| 2026-09-29 | K117-P | Run the copy-cost / layout / size-sweep bench (script being written on k117-prep) | user |
| 2026-09-29 | C18-C | Green light: run the MassIVE-KB dry run (2 shards) as in notes/C18_prepare_card.md | user |
| 2026-09-29 | K110-S | Do the checkpoint inventory now (read-only; deletion only after per-tier approval + deletion protocol) | user |
| 2026-09-29 | Stage 0 | Submit per runbook (preprocess -> Pairformer -> transformer); submitted sequentially by pbs/tools/feeder.sh because held jobs count toward the per-user queued limit | user |
| 2026-09-29 | K114/K119 | Submitted: K114 profiler 8877117 (debug), K119 FlexAttention test 8877118 (debug-scaling) | user |
| 2026-09-29 | K132-I | (a) telegraf 1.40.1 (official static binary, sha256 verified against the release notes) at $S/tools/bin/telegraf, linked from ~/.local/bin (on PATH via ~/.profile), so master's aurora-pretrain.pbs is unchanged; compute-node check 8877940: xpu-smi + telegraf serve /metrics (memory %, power, frequency; GPU utilization is N/A from xpu-smi 1.2.43, not needed). (c) master checked: it does `module use /soft/modulefiles; module load daos frameworks xpu-smi` and never provides telegraf -- Chris must have his own on PATH. Stage 0 Pairformer arm resubmitted: 8877949 (transformer follows via the feeder) | user |
| 2026-09-29 | K136-C | Run the missing 50m cell lr 4e-4 + P170xK2 (3 seeds, K66-C card otherwise): job 8878032 (capacity 10 h); scoring via feeder plan k136_score (same sets as K66-C + yeast) | user |
| 2026-09-29 | K137-C | No 25m in the all-checkpoint run for now (user wrote "K136-C", meaning the 25m question) | user |
| 2026-09-29 | C27-C | Consider a training where the consensus is weighted more heavily (oversampled, "more canonical"); test at 50m with library search vs current methods -> card notes/C27_consensus_weight_card.md (draft, settings K139-C need approval) | user |
| 2026-09-29 | C27-C / K139-C | "Yes run your proposal": card notes/C27_consensus_weight_card.md approved as drafted -- sampler weighting (--consensus_weight, without replacement), 50m arms cons_w1 / cons_w3 / cons_always / cons_w3_kl0 x 3 seeds, ref_exp = K53 runs 8873825 re-scored, decision rule = best validation library Hit@1 provided validation experimental MAP@R drops by no more than the seed spread; debug smoke first. Implementation on branch c27-prep | user |
| 2026-09-29 | K138-I | Keep running (debug) jobs until the Stage 0 DDP failure is understood: first pair 8878109 (Pairformer, debug) + 8878110 (transformer, debug-scaling), 20 min, TORCH_DISTRIBUTED_DEBUG=INFO, TORCH_CPP_LOG_LEVEL=INFO, CCL_LOG_LEVEL=info, output $S/runs/k138 | user |
| 2026-09-29 | K138-I (resolved) | Cause: CCL_KVS_MODE=mpi makes oneCCL call MPI before MPI_Init on the 2026-09 stack (rank 0 aborts at DDP setup; both arms, 8878109/8878110). ddp_smoke 8878122: KVS pmi and mpi4py-import both pass. Fix: aurora-pretrain.pbs CCL_KVS_MODE now overridable (default mpi, as master); Stage 0 passes CCL_KVS_MODE=pmi (bootstrap only, no effect on training numerics or speed). Stage 0 Pairformer resubmitted 8878133. For master/Chris: same failure expected there after the update | user (keep running jobs until understood) |
| 2026-09-29 | K138-I (2nd cause) | After the KVS fix, Stage 0 (8878133) got past DDP setup but DataLoader workers crashed: "OSError: AF_UNIX path too long". Held debug node 8878183 (user: debug interactively): PALS sets each rank's TMPDIR to /var/tmp/pbs.<job>/<uuid>/tmp = 109 chars > the 108-char AF_UNIX limit; reproduced and fixed live. Fix in aurora-pretrain.pbs rank wrapper: TMPDIR -> /tmp/msdelta-<job>-<rank> when longer than 60 chars. Stage 0 Pairformer resubmitted 8878234 | user |
| 2026-09-29 | K138-I (scope) | The TMPDIR bug hit EVERY mpiexec-launched training since the 2026-09 update, not only Stage 0: K66-C 400m 8876832 and 25m 8876833 and K136 8878032 never trained (all 12/3 arms: AF_UNIX crash at 03:23 / 13:53, no loss, no checkpoint); 25m was killed at walltime 10 h, 400m and 50m hung. Claude had reported them as "running" from qstat only. Fix moved to pbs/lib/load_frameworks.sh (short job TMPDIR, all scripts); validation: C27 smoke 8878345. qdel of the hung jobs was blocked by the permission classifier -> user | Claude (for the record) |
| 2026-09-29 | K140-S | User ran qdel 8876832 8878032; resubmission OK'd. Third launcher bug found by C27 smoke 8878345: CCL_KVS_MODE=mpi also kills single-rank mpiexec launches -> default pmi in every launcher (guard test). C27 smoke 8878405 trains. Resubmitted (same approved cards): K66-C 400m 8878459, 25m 8878461, K136 8878464 (scoring feeders/watchers repointed); C27 12 arms via feeder after the smoke | user |
| 2026-09-29 | K137-C (changed) | 25m IS included in the all-checkpoint/all-scale C run (reverses the earlier "no 25m for now"); only 25m@540k is on /flare -> the other 25m checkpoints must be fetched | user |
| 2026-09-29 | C priorities | After the recipe (HP + consensus + consensus weighting) is fixed: scale to all checkpoints x all scales; main eval = LIBRARY SEARCH; also re-collect the paper's result set; then downstream A (alignment) and R (reranking) | user |
| 2026-09-29 | P goal | Pairformer thread: find the architecture (design + input features) that beats the current transformer, accounting for compute; then scale and run real pretraining | user |
| 2026-09-29 | K145-C | 25m checkpoints: user will ask Chris later; not needed until the all-checkpoint run; if they're missing then, run without them and add later | user |
| 2026-09-29 | K144-P | Plan accepted with changes: (1) run, then normalise comparisons for FLOPs/time; (2) absolute m/z is NOT a confound -- including it is a design choice and a win if it helps (first iteration omitted it on a generalisation thesis); (3) decoupled streams (pair rep updated less often than singles) may be the most important idea; (5) speed: optimise every kernel/trick; (6) win on pretraining/scaling laws -> scale to more tasks -> new model paper. (4) input features: explained in chat | user |
| 2026-09-29 | W&B | Pairformer-thread runs log to CS_Pharm/pairformer_pretrain (both arms of P comparisons); pass WANDB_ENTITY=CS_Pharm (without it the key's personal entity kelhus2-uic is used) | user |
| 2026-09-29 | K146-I | Keep the next-eval probe (8879772) queued; legacy-reg probe 8879810 also queued ("Can Never Run": no nodes) | user |
| 2026-09-30 | K141-P | No own transformer runs for P comparisons: the baseline is the existing transformer pretraining logs (W&B CS_Pharm/msdelta-pretrain) | user |
| 2026-09-30 | K144-P (features) | First make sure the Pairformer's features don't leak intensities (K147 audit); then the CURRENT feature set + the transformer's precursor decision is the feature list; feature explorations later. First priority: speed optimisations and whether triangle multiplication / attention are useful | user |
| 2026-09-30 | K142-C | Auto-resume the 400m C training if it hits walltime (pbs/tools/k66_400m_resume.sh running) | user |
| 2026-09-30 | P2 | Start a full Pairformer pretraining test run in pairformer_pretrain -> card notes/P2_full_pretrain_card.md (K148-P, draft) | user |
| 2026-09-30 | K96-S | Max-normalisation leak: very likely not a big deal (parked) | user |
| 2026-09-30 | K149 | No leak/leak-free training comparison: the Pairformer uses the current leak-free model; just run it and compare with Chris's existing transformer pretraining | user |
| 2026-09-30 | K148-P | Options (a)+(c): cap 150 with larger spectra dropped, with and without triangle attention; not the full dataset -- about 0.25 or 0.5 epoch, to see how the eval looks | user |
| 2026-09-30 | K147-P (done) | Leak audit merged (tests/test_intensity_leak.py, notes/K147_intensity_leak_audit.md): the ONLY leak in either architecture is the max-normalisation one (K96; parked by the user as very likely not a big deal): base peak masked 50% of the time, detectable 100%. Masked peaks below the max change nothing (bit-identical); the Pairformer's pair features never see a masked intensity; m/z features clean; neither model takes precursor m/z or charge. Both arms of the P comparison use the same normalisation as the transformer checkpoints | Claude (audit) |
| 2026-09-30 | K128-P | Rerun: job 8879960 (K114 missing triattn cases N=150/256/512 + K119 variants, one process each) | user |
| 2026-09-30 | K150-P | Optimise the Pairformer BEFORE the 0.5-epoch comparison runs (preprocessing continues) | user |
| 2026-09-30 | K129-I | Core dumps off in every job (ulimit -c 0 in load_frameworks.sh; opt-out MSDELTA_CORE_DUMPS=1) | user |
| 2026-09-30 | K110-S | Approved and DONE (deletion protocol: lists rebuilt + re-verified live -- no final/, unreferenced, job not live, K110f reruns exist; dry run; tested on a copy; staged live): K110g 123 dirs (12 GiB; runs/allocfix kept -- referenced by pbs/diag/allocfix.sh), K110f 25 crashed runs (328 GiB), K110a-lite 4,836 optimizer/scheduler/rng/global_step items in checkpoints of 262 old finished runs (781 GiB; weights and finals kept). Lists + logs: $S/k110-logs/. Project 8.75/10 TB after. Still open: full K110a (delete those checkpoints' weights too -- user said "we would only need the models"; to confirm), K110b/c/d/h/i/j | user |
| 2026-09-30 | C18-C | Full prep: clean (peptide-disjoint) splits; KEEP oversize spectra whole (MAX_PEAKS=1000000; loaders drop/trim if needed); no group cap yet -- look at group-size statistics first. Job 8879991 | user |
| 2026-09-30 | K117-P | Run the copy-cost bench (job 8879977) | user |
| 2026-09-30 | K130-P | Bench settings approved; batch sizes checked against master: master configs micro 64/tile (transformer; PBS default global 512), the production transformer runs logged micro 2 x accum 4, the Pairformer branch (exp_pairformer) and Stage 0 use micro 32 (Chris also ran 8). The bench's B=2,8 miss 32 -> supplementary run B=32, N=150/200 (N>=256 at B=32 OOMs with tri-attn, K114) | user |
| 2026-09-30 | K130-P | Use the batch size of the best pretraining run: best Pairformer run on W&B (CS_Pharm/pairformer_pretrain pf-baseline, eval 0.060) used micro 32 -> the B=32 supplement (8880031) is the relevant one; best transformer run (400m production, eval 0.054) used micro 1 x accum 4 | user |
| 2026-09-30 | K117-P | Adopt the copy-free triangle-attention layout as the default (agent on branch k117-adopt) | user |
| 2026-09-30 | K114-P | Decoupled streams WILL be implemented (agent on branch decoupled-streams; knobs default to the current model); checking whether triangle attention is worth it stays important | user |

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
