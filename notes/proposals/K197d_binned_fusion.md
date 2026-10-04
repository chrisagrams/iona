# K197d-C: combine our spectrum encoder with binned cosine (no training) -- potential direction

Status: **POTENTIAL DIRECTION, to explore later** (user, 2026-10-04: "Record as K197d as a potential direction to
explore"). Nothing scheduled.

## Evidence (K197-C, notes/OBSERVATIONS.md; results/summary/k197_binned_edge.md)

Late fusion, score = alpha x our cosine + (1 - alpha) x binned cosine (0.1 Da), consensus recipe at 540k, seed 0,
open search, experimental MAP@R:

| set | binned | 25m | 400m | fusion 25m (alpha) | fusion 400m (alpha) |
|---|---:|---:|---:|---:|---:|
| test | 0.729 | 0.864 | 0.917 | 0.891 (0.9) | 0.917 (1.0) |
| oodval | 0.906 | 0.785 | 0.822 | 0.930 (0.8) | 0.936 (0.8) |
| mouse | 0.916 | 0.835 | 0.868 | 0.924 (0.8) | 0.926 (0.9) |
| human | 0.809 | 0.881 | 0.899 | 0.903 (0.9) | 0.907 (0.9) |
| yeast20k | 0.916 | 0.857 | 0.808 | 0.925 (0.8) | 0.923 (0.8) |

Why it works: our encoder sometimes rates spectra that share few fragment peaks as identical; binned cosine vetoes
those, and our encoder carries the cross-lab invariance binned lacks.

## What exploring it would involve

1. Pick alpha on validation / oodval only (not on the sets we report), then report test / mouse / human / yeast.
2. A single embedding whose cosine IS the fusion: [sqrt(alpha) x ours, sqrt(1 - alpha) x binned], binned part sparse
   (20,000 bins at 0.1 Da) or PCA-compressed (the pca:<width>:<dims> baselines exist) -- check how much PCA loses.
3. With the precursor filters (none / 20 ppm / iso-20 ppm) and on filter passes / failures, as for every result.
4. Library search (consensus library) as well as replicate retrieval.

Cost: one debug job (the K197 diagnostic already does steps 1 and the fusion scoring; add the PCA variant and filters).

Related training-side options (not proposals yet): K197a peak-subsampling augmentation, K197b peak-overlap hard
negatives, K197c relational distillation from binned cosine (notes/OPEN_QUESTIONS.md K197-C).
