# Open questions

Every question Claude raises in chat is also written here, with enough context to be read
on its own later. When the user decides, the entry moves to `DECISIONS.md` (with the decision)
and is deleted here. IDs: running number + track suffix (-C contrastive / spectrum encoder,
-A alignment / peptide encoder, -D denoise, -R rescoring, -I infrastructure, -P Pairformer,
-S cross-cutting).

Status: **parked** = user said to come back later; **open** = waiting on the user.

---

## Alignment (peptide encoder) -- parked until the new C models exist

### K86-A: which 20 ppm number the paper cites for the yHydra comparison (parked 2026-09-28)
Context: the alignment evaluation (peptide <-> spectrum retrieval) was upgraded (branch
`a-filtered-eval`, 58c9b28) to report every number without a precursor filter, with a plain 20 ppm
filter and with an isotope-tolerant 20 ppm filter, on filter passes and failures (project rule).
The OLD yHydra comparison script had its own windows, "±1.1 Da" and "20 ppm", which compare
NEUTRAL masses and ignore charge; those numbers are in the paper. The NEW standard 20 ppm filter
compares m/z at the query's charge and requires the charge to match. Both are now computed; they
can differ slightly.
Question: cite the old window (consistent with the published numbers) or switch to the new standard?

### K87-A: what to do when some spectra have no precursor value (parked)
Context: the filtered cross-modal metrics need each query spectrum's measured precursor m/z. Right
now, if ANY row lacks it (0 / NaN), all filtered numbers for that dataset are skipped.
Options: (a) keep skipping; (b) score only rows that have a precursor and report how many were
excluded.

### K88-A: port the mouse-vs-yHydra comparison into the repo (parked)
Context: `pbs/mouse_vs_yhydra.pbs` runs `compare.py` from the shelved `portable_eval` package on
/flare (not in the repo), so it does not get the new with/without-filter split.
Question: port it to the repo's shared filtered-evaluation code (like the yHydra script already was)?

### K89-A: small follow-ups to the filtered alignment evaluation (parked)
(1) `sweeps/package_alignment.py` only reads the old keys; update it so tables/figures can show the
filtered numbers. (2) The during-training validation check (`finetune_align.evaluate_alignment`)
still uses the plain metric; proposal: leave it (it is a health check, not the test evaluation).

---

## Infrastructure / cross-cutting

### K90-S: how finished branches get merged (parked 2026-09-28)
Context: work is done in separate git worktrees/branches so it cannot disturb running jobs or other
work. Finished, unmerged branches: `a-filtered-eval` (filtered alignment eval), `p1-pairformer`
(Pairformer port), `c25-library-search` (library-search evaluation); `i2-dag` (job scheduler) is
being merged under K84-I.
Question: merge each into `dev_finetune_02` as soon as you approve its content, or review each
branch yourself first?

### K63-I: your review of the job-DAG scheduler (open; reminder requested)
Context: built on branch `i2-dag` (tested only on a simulated batch system). Spec sheet:
`notes/DAG_SPEC.md`. First live trial approved (K84-I). You said you would read it and say whether
you agree with the design.

### K96-S: pretraining input normalisation may leak which masked peak is the base peak (parked)
Context: in `msdelta/models/processing_msdelta.py` the model's INPUT intensity feature is
log1p(I) divided by the maximum of log1p(I) over ALL peaks, computed before masking. The TARGET is
a proper distribution: I / sum(I), renormalised over the masked peaks, and the loss is KL between
it and a softmax over the masked peaks. If the tallest peak is masked, no visible peak has input
value 1.0, which hints that a masked peak is the tallest. Affects every pretrained model (both
architectures); comparisons stay fair, absolute pretraining scores may be inflated. Not yet
measured.
Options: (a) record as a caveat; (b) measure how often/how much it matters; (c) normalise over
visible peaks for NEW pretraining runs only (master's pipeline untouched).

---

## Contrastive (spectrum encoder)

### K55-C: which filtered number to headline (parked)
Context: every retrieval number is reported three ways: no filter, plain 20 ppm, isotope-tolerant
20 ppm (the latter never drops correct matches on real data, at ~5x more candidates). Full tables
show all three. The question is only which ONE "with filter" number goes into headline figures /
summaries when there is room for one. Related: K83-C.

### K83-C: add the ±1.1 Da window as a fourth standard filter? (open)
Context: ±1.1 Da on neutral mass (used in the yHydra comparisons and the paper) also tolerates ±1
isotope errors, but admits every peptide within ~1 Da (dozens of candidates); isotope-tolerant
20 ppm is much tighter and also tolerates ±2. Adding ±1.1 Da would make our tables line up with the
paper's yHydra numbers. Cost: one more column in the same pass.

### K78-C: build consensus libraries for mouse / human / yeast (parked)
Context: library search (C25) needs consensus spectra; only ms-contrastive-100k has them. Building
them for the species sets means merging replicates into consensus spectra (a method choice:
peak merging, minimum replicates).

### K97-C: drop or keep `library/MAP@R` (open)
Context: in library search each query has exactly ONE correct library entry, so MAP@R (hits within
the top R, R = 1) equals Hit@1; the key just repeats it. Drop it, or keep it for table consistency?

### K98-C: ties in library search count as misses (open)
Context: if the correct library entry has exactly the same similarity as a wrong one, the new
library-search code counts it as a miss (conservative; identical embeddings cannot score well). The
older retrieval metric breaks ties arbitrarily. Keep the conservative rule?

### K99-C: groups without consensus / consensus without queries (open)
Context: in ms-contrastive-100k validation, 48 groups have no consensus spectrum (their 201
experimental queries cannot be scored; they are counted and excluded) and 201 groups have a
consensus but no experimental spectra (kept in the library as distractors, as in a real library).
Keep both behaviours?

### K100-C: when to run library search (open)
Context: approved card K79-C: ms-contrastive-100k validation + test, on the C20 models and the K66-C
models. Proposal: after K97-K99 and merging `c25-library-search`, score the C20 models now (one
debug job per split) and include library search in the K66-C scoring automatically.

### C18-C: MassIVE-KB (parked, low priority)
Context: MassIVE-KB (chrisagrams/massive_kb_v1_shuffled, 30.5M spectra, all splits now on /flare)
could be a larger contrastive training set. No prep script exists. User is ~95% sure it does not
overlap our evaluation sets; an overlap check (one debug job) and asking Chris about its provenance
were proposed.

---

## Pairformer (P1)

### K85-P: the first Pairformer vs transformer comparison run (open; see K91-K95 and the review)
Context: `notes/P1_card_draft.md` (branch p1-pairformer) drafts a short debug pretraining run
comparing the ported Pairformer with our transformer at ~50M parameters (loss curves, step time).
It needs choices: size match (options A-D), pair settings, Fourier frequencies, peaks cap, data,
steps/batch/lr. Review: `notes/PAIRFORMER_REVIEW.md`. User asked (2026-09-28): defaults first, or
HP search first? -- answered in chat; decision pending.

### K91-P: mass-defect features are poorly encoded (parked 2026-09-28)
Context: the Pairformer's pair features include the fractional mass of each peak-pair difference
(mass defect), encoded with non-integer log-spaced Fourier frequencies. So a defect of −1 mDa and
+1 mDa look unrelated to the model (feature distance 5.5 vs 0.36 for a 2 mDa step), splitting losses
just below an integer mass (CO, CO2, O) from those just above (H2O, NH3). Inherited from the source.
Options: fix before any comparison (integer frequencies, or feed the signed defect) or run as-is
and test the fix as an ablation.

### K93-P / K95-P: explained in chat 2026-09-28 (memory needs; configuration defaults); decisions
pending on the K85-P card.
