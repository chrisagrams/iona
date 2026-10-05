# K197-C: where binned cosine 0.1 Da beats our spectrum encoders

`report.md`: MAP@R / Hit@1 per method and late fusion (alpha x encoder + (1 - alpha) x binned cosine) per set; what
each method's wrong top-1 hit is; MAP@R by group size, charge, peak count and replicate similarity; the queries the
400m misses at top-1 but binned gets right (shared peaks, precursor mass of the wrong hit).

## How it was made

- Per-query diagnostic jobs: `pbs/diag/k197_binned_edge.pbs` (diag 8903703; yeast20k rerun 8903748) ->
  `$MSDELTA_DIAG/k197/<set>_{25m,400m,binned0.1,fuse*,meta}.npz`, `<set>_summary.json`, `<set>.log`.
- Report: `.venv/bin/python sweeps/k197_analyze.py` (login; reads `$MSDELTA_DIAG/k197/`, writes `report.md`).
- Built: 2026-10-04 03:36 UTC. Notes: notes/OBSERVATIONS.md "K197-C"; potential direction notes/proposals/K197d_binned_fusion.md.
