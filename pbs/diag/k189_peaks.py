"""K189-P: peak-count distribution of MSConsensus-100M (raw shards), to price training at Chris's max_peaks 512.

    python pbs/diag/k189_peaks.py OUT.json shard.parquet [shard.parquet ...]

Counts peaks per spectrum (length of the `mz` list); reports the fraction kept at caps 150 / 256 / 512 (spectra over the
cap are DROPPED by msdelta.data, not truncated), quantiles of the kept lengths, and the mean padded length of random
batches of 24 kept spectra (what a micro-batch of 24 pays with pad-to-longest), plus N^2 and N^3 means for pair costs.
"""
import json
import sys

import numpy as np
import pyarrow.parquet as pq

out, shards = sys.argv[1], sys.argv[2:]
n = np.concatenate([np.diff(pq.read_table(f, columns=["mz"]).column("mz").combine_chunks().offsets.to_numpy())
                    for f in shards])
rep = {"shards": shards, "spectra": int(n.size)}
rng = np.random.default_rng(0)
for cap in (150, 256, 512):
    k = n[(n > 0) & (n <= cap)]
    b = rng.permutation(k)[: (k.size // 24) * 24].reshape(-1, 24).max(axis=1)
    rep[f"cap{cap}"] = {"kept_frac": float(k.size / n.size),
                        "q": {q: float(np.quantile(k, q / 100)) for q in (10, 25, 50, 75, 90, 99)},
                        "mean": float(k.mean()), "mean_N2": float((k.astype(float) ** 2).mean()),
                        "mean_N3": float((k.astype(float) ** 3).mean()),
                        "batch24_padded_mean": float(b.mean()), "batch24_padded_q50": float(np.median(b))}
json.dump(rep, open(out, "w"), indent=2)
print(json.dumps(rep, indent=2))
