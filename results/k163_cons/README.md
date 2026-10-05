# K163-C: consensus twins vs their no-consensus partners

Per scale (25m-400m, 540k checkpoint), mean ± sd over 3 seeds: library search Hit@1 with no filter / 20 ppm /
isotope-tolerant 20 ppm, F = queries failing the 20 ppm filter, Fbar = passing it, and experimental MAP@R; columns
"no-cons | cons | delta". One file per evaluation set (`validation.md`, `test.md`, `yeast.md`).

## How it was made

- Command: `.venv/bin/python sweeps/compare_cons.py validation test yeast` (login; prints each table and writes
  `results/k163_cons/<set>.md`).
- Inputs: twins `$MSDELTA_EVAL/contrastive/cons-<set>/s<scale>_ck540k_lr4e-4_p170k2_cons_seed<i>.json`, partners
  `$MSDELTA_EVAL/contrastive/hp-scale-<scale>-<set>/s<scale>_ck540k_lr4e-4_p170k2_seed<i>.json`.
- Built: validation and test 2026-10-01, yeast 2026-10-03 (UTC). Notes: notes/OBSERVATIONS.md "K163".
