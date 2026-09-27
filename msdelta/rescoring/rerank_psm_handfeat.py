"""Stage 1b of the PSM-reranking eval: our hand-built fragment features per candidate.

    python -m msdelta.rescoring.rerank_psm_handfeat --run HEK293/0718-1.parquet --out OUT.parquet

The 22 spectrum/candidate/match features of msdelta.rescoring.rescoring (everything but
embedding_cosine), computed from the RAW centroided peaks and raw intensities in the
dataset -- so FT32 (the old pipeline rebuilding intensity from a normalised log) does not
apply here. Fragment tolerance is the dataset card's ion-trap setting, max(250 ppm,
0.05 Da); the package default of 20 ppm would match almost nothing on this MS2.

Keyed by `candidate` (the dataset's candidate_id) to join stage 1's rows. CPU only.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

REPO_ID = "Gaolaboratory/psm-rerank-hek-hct116"
# Pinned: the 2026-09-25 00:34 update moved <dataset>/<run>.parquet to spectra/ and added
# features/. Every table so far was built from this revision; keep them all on it.
REVISION = "87f5c2756f5de8da8a664ef7e1a4dac8de067a88"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ppm", type=float, default=250.0)
    ap.add_argument("--da-floor", type=float, default=0.05)
    cli = ap.parse_args(argv)

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from msdelta.rescoring.rerank_psm_embed import to_notation
    from msdelta.rescoring.rescoring import FEATURE_NAMES, extract_features

    t0 = time.time()
    path = hf_hub_download(REPO_ID, cli.run, repo_type="dataset", revision=REVISION)
    rows = pq.read_table(path, columns=["charge", "precursor_mz", "mz", "intensity",
                                        "candidates"]).to_pylist()
    names = [n for n in FEATURE_NAMES if n != "embedding_cosine"]
    out = {"candidate": [], **{f"hf_{n}": [] for n in names}}
    for r in rows:
        mz = np.asarray(r["mz"], dtype=np.float64)
        it = np.asarray(r["intensity"], dtype=np.float64)
        for c in r["candidates"]:
            v = extract_features(to_notation(c["sequence"], c["modifications"]), mz, it,
                                 float(r["precursor_mz"]), int(r["charge"]), 0.0,
                                 cli.ppm, cli.da_floor)
            out["candidate"].append(c["candidate_id"])
            for n, x in zip(FEATURE_NAMES, v):
                if n != "embedding_cosine":
                    out[f"hf_{n}"].append(float(x))
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(out), cli.out)
    print(f"[handfeat] {cli.run}: {len(out['candidate']):,} candidates "
          f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
