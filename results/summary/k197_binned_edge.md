# K197-C: where binned cosine 0.1 Da beats our spectrum encoders

Per-query diagnostic (pbs/diag/k197_binned_edge.py): consensus-recipe encoders at 540k, seed 0; experimental queries vs experimental gallery, open search, peptide+charge groups (the evaluation's definitions).

## 1. Methods and late fusion (MAP@R)

fusion = alpha * encoder cosine + (1 - alpha) * binned cosine; alpha 0 = binned, 1 = encoder.

| set | binned | 25m | 400m | best fusion 25m (alpha) | best fusion 400m (alpha) |
|---|---:|---:|---:|---:|---:|
| test | 0.729 | 0.864 | 0.917 | 0.891 (0.9) | 0.917 (1.0) |
| oodval | 0.906 | 0.785 | 0.822 | 0.930 (0.8) | 0.936 (0.8) |
| mouse | 0.916 | 0.835 | 0.868 | 0.924 (0.8) | 0.926 (0.9) |
| human | 0.809 | 0.881 | 0.899 | 0.903 (0.9) | 0.907 (0.9) |

## 2. What the wrong top-1 hit is (share of all queries)

| set | method | top-1 wrong | same peptide, other charge | same sequence, other mods | isobaric | different peptide |
|---|---|---:|---:|---:|---:|---:|
| test | binned0.1 | 0.181 | 0.018 | 0.001 | 0.001 | 0.161 |
| test | 25m | 0.088 | 0.002 | 0.002 | 0.000 | 0.084 |
| test | 400m | 0.052 | 0.001 | 0.001 | 0.000 | 0.049 |
| oodval | binned0.1 | 0.025 | 0.000 | 0.001 | 0.000 | 0.023 |
| oodval | 25m | 0.095 | 0.000 | 0.002 | 0.000 | 0.092 |
| oodval | 400m | 0.077 | 0.000 | 0.001 | 0.000 | 0.075 |
| mouse | binned0.1 | 0.029 | 0.001 | 0.016 | 0.001 | 0.011 |
| mouse | 25m | 0.072 | 0.000 | 0.029 | 0.001 | 0.041 |
| mouse | 400m | 0.057 | 0.000 | 0.029 | 0.001 | 0.027 |
| human | binned0.1 | 0.073 | 0.006 | 0.033 | 0.001 | 0.033 |
| human | 25m | 0.067 | 0.001 | 0.036 | 0.001 | 0.030 |
| human | 400m | 0.059 | 0.001 | 0.035 | 0.001 | 0.023 |

## 3. MAP@R by stratum (share of queries in brackets)

### test

| stratum | bin | share | binned | 25m | 400m | 400m - binned |
|---|---|---:|---:|---:|---:|---:|
| R | 1 | 0.16 | 0.654 | 0.776 | 0.858 | +0.204 |
| R | 2-4 | 0.84 | 0.744 | 0.881 | 0.929 | +0.185 |
| charge | 2 | 0.56 | 0.712 | 0.872 | 0.921 | +0.209 |
| charge | 3 | 0.36 | 0.746 | 0.858 | 0.912 | +0.167 |
| charge | 4+ | 0.08 | 0.778 | 0.833 | 0.913 | +0.135 |
| peaks | <105 | 0.25 | 0.835 | 0.865 | 0.919 | +0.084 |
| peaks | 105-172 | 0.25 | 0.803 | 0.890 | 0.934 | +0.131 |
| peaks | 172-268 | 0.25 | 0.714 | 0.876 | 0.924 | +0.210 |
| peaks | >=268 | 0.25 | 0.567 | 0.824 | 0.892 | +0.325 |
| replicate cos (binned) | -1.0-0.4 | 0.32 | 0.502 | 0.686 | 0.798 | +0.296 |
| replicate cos (binned) | 0.4-0.6 | 0.58 | 0.810 | 0.938 | 0.969 | +0.159 |
| replicate cos (binned) | 0.6-0.8 | 0.10 | 0.982 | 0.993 | 0.996 | +0.013 |
| replicate cos (binned) | 0.8-2.0 | 0.00 | 1.000 | 1.000 | 1.000 | +0.000 |

### oodval

| stratum | bin | share | binned | 25m | 400m | 400m - binned |
|---|---|---:|---:|---:|---:|---:|
| R | 1 | 0.18 | 0.927 | 0.804 | 0.833 | -0.094 |
| R | 2-4 | 0.32 | 0.907 | 0.775 | 0.812 | -0.095 |
| R | 5-9 | 0.27 | 0.903 | 0.783 | 0.825 | -0.078 |
| R | 10-19 | 0.24 | 0.890 | 0.785 | 0.823 | -0.068 |
| charge | 2 | 0.59 | 0.902 | 0.808 | 0.842 | -0.061 |
| charge | 3 | 0.33 | 0.913 | 0.766 | 0.810 | -0.103 |
| charge | 4+ | 0.07 | 0.896 | 0.687 | 0.711 | -0.186 |
| peaks | <71 | 0.25 | 0.886 | 0.701 | 0.751 | -0.136 |
| peaks | 71-122 | 0.25 | 0.910 | 0.831 | 0.861 | -0.048 |
| peaks | 122-185 | 0.25 | 0.905 | 0.809 | 0.849 | -0.056 |
| peaks | >=185 | 0.25 | 0.921 | 0.799 | 0.826 | -0.095 |
| replicate cos (binned) | -1.0-0.4 | 0.28 | 0.759 | 0.534 | 0.605 | -0.154 |
| replicate cos (binned) | 0.4-0.6 | 0.57 | 0.953 | 0.857 | 0.887 | -0.066 |
| replicate cos (binned) | 0.6-0.8 | 0.15 | 0.994 | 0.972 | 0.973 | -0.021 |
| replicate cos (binned) | 0.8-2.0 | 0.00 | 1.000 | 1.000 | 1.000 | +0.000 |

### mouse

| stratum | bin | share | binned | 25m | 400m | 400m - binned |
|---|---|---:|---:|---:|---:|---:|
| R | 1 | 0.10 | 0.923 | 0.787 | 0.834 | -0.088 |
| R | 2-4 | 0.32 | 0.943 | 0.853 | 0.889 | -0.055 |
| R | 5-9 | 0.40 | 0.941 | 0.875 | 0.902 | -0.039 |
| R | 10-19 | 0.18 | 0.805 | 0.742 | 0.771 | -0.034 |
| charge | 2 | 0.92 | 0.921 | 0.843 | 0.876 | -0.045 |
| charge | 3 | 0.08 | 0.854 | 0.741 | 0.775 | -0.079 |
| charge | 4+ | 0.00 | 0.927 | 0.885 | 0.969 | +0.042 |
| peaks | <43 | 0.24 | 0.925 | 0.718 | 0.786 | -0.139 |
| peaks | 43-66 | 0.26 | 0.935 | 0.862 | 0.888 | -0.047 |
| peaks | 66-98 | 0.25 | 0.925 | 0.887 | 0.907 | -0.018 |
| peaks | >=98 | 0.25 | 0.878 | 0.868 | 0.885 | +0.008 |
| replicate cos (binned) | -1.0-0.4 | 0.05 | 0.756 | 0.389 | 0.494 | -0.262 |
| replicate cos (binned) | 0.4-0.6 | 0.55 | 0.893 | 0.797 | 0.839 | -0.055 |
| replicate cos (binned) | 0.6-0.8 | 0.40 | 0.966 | 0.943 | 0.953 | -0.013 |
| replicate cos (binned) | 0.8-2.0 | 0.00 | 1.000 | 1.000 | 1.000 | +0.000 |

### human

| stratum | bin | share | binned | 25m | 400m | 400m - binned |
|---|---|---:|---:|---:|---:|---:|
| R | 1 | 0.19 | 0.820 | 0.864 | 0.881 | +0.061 |
| R | 2-4 | 0.37 | 0.815 | 0.891 | 0.910 | +0.095 |
| R | 5-9 | 0.24 | 0.803 | 0.890 | 0.905 | +0.102 |
| R | 10-19 | 0.21 | 0.797 | 0.870 | 0.889 | +0.092 |
| charge | 2 | 0.58 | 0.812 | 0.916 | 0.932 | +0.120 |
| charge | 3 | 0.37 | 0.815 | 0.847 | 0.867 | +0.052 |
| charge | 4+ | 0.05 | 0.737 | 0.738 | 0.757 | +0.020 |
| peaks | <73 | 0.25 | 0.832 | 0.804 | 0.838 | +0.006 |
| peaks | 73-138 | 0.25 | 0.876 | 0.911 | 0.925 | +0.049 |
| peaks | 138-233 | 0.25 | 0.820 | 0.918 | 0.929 | +0.109 |
| peaks | >=233 | 0.25 | 0.711 | 0.892 | 0.904 | +0.194 |
| replicate cos (binned) | -1.0-0.4 | 0.12 | 0.590 | 0.633 | 0.694 | +0.105 |
| replicate cos (binned) | 0.4-0.6 | 0.66 | 0.802 | 0.897 | 0.911 | +0.109 |
| replicate cos (binned) | 0.6-0.8 | 0.22 | 0.950 | 0.967 | 0.973 | +0.023 |
| replicate cos (binned) | 0.8-2.0 | 0.00 | 1.000 | 1.000 | 1.000 | +0.000 |

## 4. Queries the 400m misses at top-1 but binned gets right

Wrong top-1 hits: share of queries, and how many of those wrong hits sit within 20 ppm / isotope-tolerant 20 ppm of the query's precursor mass. Then, for a 400-query sample of the queries 400m misses and binned hits: binned cosine (shared peaks) between the query and our wrong hit vs its true replicate, our own cosines, and median peak counts.

| set | binned wrong (20ppm / iso) | 400m wrong (20ppm / iso) | sample | binned cos: our wrong hit / true replicate | our cos: wrong hit / replicate | peaks: all / these queries |
|---|---|---|---:|---|---|---|
| test | 0.181 (0.10 / 0.11) | 0.052 (0.02 / 0.03) | 367 | 0.12 / 0.33 | 0.96 / 0.96 | 173 / 127 |
| oodval | 0.025 (0.03 / 0.07) | 0.077 (0.01 / 0.02) | 400 | 0.06 / 0.39 | 0.97 / 0.96 | 122 / 81 |
| mouse | 0.029 (0.22 / 0.49) | 0.057 (0.10 / 0.28) | 400 | 0.26 / 0.50 | 0.98 / 0.97 | 66 / 34 |
| human | 0.073 (0.16 / 0.51) | 0.059 (0.11 / 0.50) | 400 | 0.35 / 0.49 | 0.98 / 0.98 | 138 / 57 |
