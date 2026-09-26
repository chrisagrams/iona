"""Stage 1b of the PSM-reranking eval: our hand-built fragment features per candidate.

    python -m msdelta.rerank_psm_handfeat --run HEK293/0718-1.parquet --out OUT.parquet

The msdelta.rescoring features from the raw peaks, at an ion-trap
tolerance of max(250 ppm, 0.05 Da). Keyed by `candidate` to join stage 1's rows. CPU only.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from msdelta.rerank_psm_embed import to_notation
from msdelta.rescoring import FEATURE_NAMES, extract_features

REPO_ID = "Gaolaboratory/psm-rerank-hek-hct116"
# Pinned: later revisions moved the run tables.
REVISION = "87f5c2756f5de8da8a664ef7e1a4dac8de067a88"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ppm", type=float, default=250.0)
    ap.add_argument("--da-floor", type=float, default=0.05)
    cli = ap.parse_args(argv)

    t0 = time.time()
    path = hf_hub_download(REPO_ID, cli.run, repo_type="dataset", revision=REVISION)
    rows = pq.read_table(path, columns=["charge", "precursor_mz", "mz", "intensity",
                                        "candidates"]).to_pylist()
    out = {"candidate": [], **{f"hf_{n}": [] for n in FEATURE_NAMES}}
    for r in rows:
        mz = np.asarray(r["mz"], dtype=np.float64)
        it = np.asarray(r["intensity"], dtype=np.float64)
        for c in r["candidates"]:
            v = extract_features(to_notation(c["sequence"], c["modifications"]), mz, it,
                                 float(r["precursor_mz"]), int(r["charge"]),
                                 cli.ppm, cli.da_floor)
            out["candidate"].append(c["candidate_id"])
            for n, x in zip(FEATURE_NAMES, v):
                out[f"hf_{n}"].append(float(x))
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(out), cli.out)
    print(f"[handfeat] {cli.run}: {len(out['candidate']):,} candidates "
          f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
