# Peptide embeddings: cross-modal retrieval vs yHydra

A peptide encoder (the student) is trained to map a peptide sequence and charge to the embedding
that our fine-tuned spectrum encoder (the teacher) gives that peptide's spectra. A spectrum can then
be identified by nearest-neighbour search among peptide embeddings.

## Files

| file | content |
|---|---|
| `A_windows.png`, `A_windows.csv` | Hit@1 vs precursor-mass window on three datasets, ours vs yHydra |
| `A_teacher_student.md`, `A_teacher_student.csv` | student quality vs teacher quality (table) |
| `plot_peptide_embedding.py` | regenerates the figure and the table from the CSVs (`python plot_peptide_embedding.py`; matplotlib + numpy) |

## Task and metric

Each query spectrum is embedded with the spectrum encoder and compared with the embeddings of all
candidate peptides; Hit@1 = fraction of queries whose top candidate is the true peptide.
Candidates can be restricted to those whose mass matches the spectrum's precursor mass:
**open search** (all candidates), **±1.1 Da**, or **20 ppm**.

## A_windows

| dataset | queries | ours (model) |
|---|---|---|
| ms-contrastive-100k test (in-distribution for ours) | 22,869 | student of the fine-tuned 400M teacher; 3 seeds |
| HEK (unseen, low-resolution MS2) | 26,628 | student of the fine-tuned 400M teacher; 3 seeds |
| nine-species yeast (unseen, high-resolution) | 75,476 | student of a 400M teacher selected on the eight other species (seed 1, step 600); 1 seed |

- Queries are the spectra whose peptide yHydra can represent, identical for both methods.
- **yHydra**: the published pretrained model, L2 distance (better than cosine for yHydra on all three datasets).
- Our candidates are (peptide, charge) pairs; yHydra's are sequences, so our candidate lists are
  slightly larger (yeast, 20 ppm: median 4 vs 3 candidates).
- "ours" is our best evaluated model on each dataset, chosen on these results; the per-row model is in
  the CSV (`method`, `teacher`, `job`).
- **Ceiling (HEK, 20 ppm)**: for 24% of HEK spectra the true peptide is more than 20 ppm from the recorded
  precursor mass, so it is excluded from the candidate list before scoring; no method can exceed 0.76.
  The other datasets have no such cases.

| dataset | window | yHydra | ours |
|---|---|---|---|
| in-distribution | open / ±1.1 Da / 20 ppm | 0.20 / 0.75 / 0.94 | 0.92 / 0.98 / 0.99 |
| HEK | open / ±1.1 Da / 20 ppm | 0.02 / 0.34 / 0.61 | 0.15 / 0.69 / 0.70 |
| nine-species yeast | open / ±1.1 Da / 20 ppm | 0.06 / 0.65 / 0.89 | 0.41 / 0.66 / 0.77 |

Reading: our embedding identifies peptides far better without a mass filter, and stays ahead on
HEK at every window; on unseen high-resolution yeast, yHydra (designed around the precursor-mass
filter) is ahead at 20 ppm.

## A_teacher_student

Students trained identically (MSE to the teacher's embedding, mean+max pooling) against three teachers;
test split of ms-contrastive-100k, 25,848 spectra searched against all 9,771 candidate peptides (open
search), 3 seeds each. Teacher quality = the teacher's own spectrum-retrieval MAP@R on the same test split.

| teacher | teacher MAP@R | student Hit@1 |
|---|---|---|
| fine-tuned 50M, step 600 | 0.831 | 0.898 |
| fine-tuned 400M seed 1, step 600 (selected on the eight other species) | 0.859 | 0.918 |
| fine-tuned 400M seed 0, end of epoch | 0.868 | 0.923 |

A better teacher gives a better student. Three teachers are too few for a curve, so this is kept as a table.

Provenance: msdelta repository, `sweeps/package_alignment.py` (student test results from
`results/finetune/align/`, yHydra comparisons from the per-dataset cross-modal result files).
