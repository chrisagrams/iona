# C18-C: prepare MassIVE-KB as contrastive training data (RUN card)

> **PROPOSED, NOT APPROVED.** Nothing in this card has been submitted. The prep script exists
> (`data/prepare_massive_kb.py`, `pbs/prepare_massive_kb.pbs`, branch `c18-massivekb-prep`);
> running it needs your approval of this card. Choices marked **[choose]** are open.

## What it does

Turns `chrisagrams/massive_kb_v1_shuffled` (revision `891f42f72e84bd38c97b0b4356115d146fa51507`,
30,504,897 spectra, already on /flare in the hub cache) into one-spectrum-per-row training data in
the same form as the prepared ms-contrastive-100k splits, after removing every spectrum whose
peptide sequence appears in any of our evaluation sets. Details: `data/README.md` and the script's
docstring. It trains nothing, uses no GPU and writes only under `$S/data/`.

Steps inside one job: unit tests -> exclusion set (once) -> prepare -> check.

## Already measured on the login node (light work, 2,000 rows)

A sample of 2,000 rows (500 from each of train_000, train_137, val_000, test_000) went through
the full pipeline in scratch space, against the REAL exclusion set:

- Modification tokens in the sample, all mapped: `C[+57.021]` 428, `M[+15.995]` 137,
  N-term `[+43.006]` 105, `N[+0.984]` 72, N-term `[+42.011]` 65, N-term `[-17.027]` 31,
  `Q[+0.984]` 23. No unknown tokens and no non-standard residues in the sample. The combined
  N-terminal carbamyl + NH3 loss (Noble writes `[25.9793]`) was not seen; if it occurs in the full
  data it is dropped and counted under `unknown_tokens` in the manifest.
- Charges: 2: 960, 3: 778, 4: 217, 5: 45. Peaks per spectrum: median 151, p90 375, max 1,470;
  79 of 2,000 (4.0%) have more than 512 peaks and are dropped (54 counted as `oversize`, the
  rest were already excluded as eval peptides). On the 1,921 others, the script's vectorised
  processing matches `MSDeltaProcessor` exactly for m/z, and to within 1.2e-7 (1 ulp) for
  log_intensity.
- **Overlap: 371 of 2,000 sampled spectra (18.6%) have a sequence (I/L collapsed) in at least one
  eval set.** Per eval set (spectra of the 2,000; overlapping): ms-contrastive-100k validation 94,
  test 68, HEK (c11) 90, human 59, HCT116 50, mouse 27, replicate corpus 22, 8-species OOD 12
  (only 2 without I/L collapse), yeast 2. So MassIVE-KB is NOT disjoint from our eval sets at the
  sequence level. The full run's `overlap_report.md` gives the exact figures.
- Kept: 1,575 of 2,000 (78.8%). `check` on that output: 0 problems.

Exclusion set (built in scratch from the real eval sets, 31 s): union 56,437 peptides / 52,850
sequences / 52,417 I/L-collapsed sequences; the yeast file's sha256 matches `CANONICAL.txt`.

## Estimates (EXTRAPOLATED from the 2,000-row run, not measured at scale)

- CPU: ~80 us per spectrum to parse, filter, process and write (measured on 500-row shards),
  plus ~12 us per spectrum to read and decode (one real 20,000-row row group decoded in
  0.21-0.27 s). About 95 us x 30.5M = ~48 CPU-minutes, i.e. about 1 minute on 64 workers.
- I/O: read 49 GB and write ~45 GB on Lustre. At an assumed 1-3 GB/s per node, this takes
  roughly 1-2 minutes each way. Aggregation plus `check` (a Python pass over ~24M peptide
  strings) should take a few minutes.
- **Wall time: about 10-15 minutes expected. Request 1 h.** A walltime kill loses nothing:
  resubmit with `RESUME=1`, and the job skips shards that are already done.
- Memory: one 20,000-row row group per worker (~36 MB decoded, ~0.2-0.5 GB RSS). With 64 workers
  that is at most ~32 GB, plus a few GB for the group-size aggregation. One node has plenty.
- **Disk: ~45 GB** for the output. The sample measured ~1.85 KB per kept spectrum, and ~24M
  spectra are expected to be kept. Add ~2 GB for `_shards/` group tables and <1 MB for the
  exclusion set. /flare has 30 PB free.

## Proposed runs **[approve]**

Both run on 1 node, CPU only, from the code snapshot of the commit on `c18-massivekb-prep`.
Neither overwrites anything: the script refuses if its output exists.

1. **Dry run** (2 shards per split, ~240k spectra; expected <5 min):

   ```bash
   cd /home/khuss/code/msdelta-c18
   qsub -q debug -l select=1 -l walltime=00:30:00 \
        -v OUT=/lus/flare/projects/UIC-HPC/khuss/msdelta/data/massive-kb-contrastive-dryrun,SHARDS=2 \
        pbs/prepare_massive_kb.pbs
   ```

   This also builds the exclusion set once, at `$S/data/massive-kb-exclusion/`, and the full run
   reuses it. Look at `manifest.json` (`unknown_tokens`, `dropped_by_reason`, `groups`) and
   `overlap_report.md` before step 2.

2. **Full run** (only after the dry run has been reviewed):

   ```bash
   cd /home/khuss/code/msdelta-c18
   qsub -q debug -l select=1 -l walltime=01:00:00 pbs/prepare_massive_kb.pbs
   ```

   Output goes to `$S/data/massive-kb-contrastive/`. If the job is killed: resubmit the same
   command with `-v RESUME=1`. If the debug queue is busy, `-q capacity` with the same
   resources.

## Open choices **[choose]**

1. **Split policy.** The default is `peptide`: all three source splits are pooled and each
   spectrum is re-assigned by a hash of its I/L-collapsed sequence (3.3% validation, 3.3% test,
   the source's proportions). The splits are then peptide-disjoint, as in ms-contrastive-100k.
   The source's own splits look spectrum-random (the README says spectra were shuffled into
   shards), so they probably share peptides; `--split-policy source` keeps them anyway. The
   manifest records how many spectra moved between splits.
2. **I/L collapse for exclusion.** Default ON (per the request). It matters for the nine-species
   sets: in the sample, 12 OOD spectra matched with collapse and 2 without.
3. **Oversize spectra.** Default: drop (4.0% of the sample), as training and eval prep do.
   `--oversize top` would keep the 512 most intense peaks instead, but that is NOT what our
   training does.
4. **Group sizes.** MassIVE-KB groups (peptide, charge) can be large. Nothing caps them here;
   `group_sizes.parquet` and the manifest histogram will show the distribution. A cap (as C11
   capped at 20) could be applied at load time.
5. **Missing eval sources?** The full PSM-rerank dataset (`Gaolaboratory/psm-rerank-hek-hct116`
   beyond c11/c14), the other nine-species test splits, and ms-contrastive-100k-train10k (the
   ABTT fit sample) are not excluded. Add them with `exclusion --source NAME=PATH` if they count
   as evaluation.
6. **Using the output.** The contrastive trainer cannot read this format yet. It needs a new
   `--dataset_format` (a small change, a separate card). ms-contrastive-100k's analytes have 3
   spectra; MassIVE-KB groups vary in size, so `replicates` and the PK sampler settings need
   choosing then.
