# K188-C: consensus fine-tunes on every pretraining checkpoint

Per scale and pretraining checkpoint (thousand steps), mean ± sd over the seeds present (n); 540k rows are the K163
consensus finals. Library search Hit@1 and experimental retrieval MAP@R with no filter / 20 ppm / iso-20 ppm; F / Fbar
columns (queries failing / passing the 20 ppm filter) where a set has failures. One file per evaluation set.

## How it was made

- Command: `.venv/bin/python sweeps/allck_cons.py [sets...]` (login; default all six sets; writes
  `results/k188_allck_cons/<set>.md`).
- Inputs: `$MSDELTA_EVAL/contrastive/cons-allck-<set>/s<scale>_ck<NNN>k_lr4e-4_p170k2_cons_seed<i>.json` (job 8901080)
  and the 540k finals in `$MSDELTA_EVAL/contrastive/cons-<set>/`.
- Built: validation, test, oodval, mouse 2026-10-03 19:30; human 2026-10-03 21:48; yeast 2026-10-04 03:37 (UTC).
  Notes: notes/OBSERVATIONS.md "K188".
