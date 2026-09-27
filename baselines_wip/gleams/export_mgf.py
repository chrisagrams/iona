"""Export RAW ms-contrastive-100k test spectra to MGF for external baselines (GLEAMS etc.).

Run with the project venv python (read-only use), on a compute node:
    /home/khuss/code/msdelta/.venv/bin/python export_mgf.py OUT_DIR

Mirrors msdelta.grouped_retrieval.build_grouped_split's order: analytes in parquet order,
peptides of chrisagrams/ms2-peptide-replicate-retrieval (any split) excluded, each analyte
flattened as [consensus, exp1, exp2, exp3]. Every spectrum is exported; meta.parquet
records `in_mp512` (0 < n_peaks <= 512), the subset kept in the prepared eval set, and
the export checks that subset's (analyte_id, source) sequence against the prepared set.

Outputs: experimental.mgf, consensus.mgf, meta.parquet (row = global row index; the
TITLE of each MGF entry starts with `row=<row>`).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HUB = "/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface/hub"
TEST = (f"{HUB}/datasets--chrisagrams--ms-contrastive-100k/snapshots/"
        "613d9b1901debc66877264bedbaa6def46ac0861/test/data.parquet")
REPL = f"{HUB}/datasets--chrisagrams--ms2-peptide-replicate-retrieval/snapshots/*/*.parquet"
PREPARED = "/lus/flare/projects/UIC-HPC/khuss/msdelta/eval-data/ms-contrastive-100k-test-mp512"
MAX_PEAKS = 512


def write_spec(f, row, aid, src, sid, precursor, charge, mz, inten):
    f.write("BEGIN IONS\n")
    f.write(f"TITLE=row={row};analyte_id={aid};source={src};spectrum_id={sid}\n")
    f.write(f"PEPMASS={precursor:.6f}\nCHARGE={int(charge)}+\nRTINSECONDS=0\n")
    for m, i in zip(mz, inten):
        f.write(f"{m:.5f} {i:.4f}\n")
    f.write("END IONS\n")


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(REPL))
    assert files, REPL
    exclude = set()
    for p in files:
        exclude.update(pq.read_table(p, columns=["peptide"]).column("peptide").to_pylist())
    print(f"[export] {len(exclude):,} replicate-corpus peptides from {len(files)} files")

    t = pq.read_table(TEST).to_pylist()
    for a in t[:5]:
        print("[export] sample", a["peptide"], a["charge"], a["precursor"])
    meta = {k: [] for k in ("row", "analyte_id", "peptide", "charge", "precursor", "source",
                            "spectrum_id", "n_peaks", "in_mp512")}
    n_excl = 0
    row = 0
    with open(out / "experimental.mgf", "w") as fe, open(out / "consensus.mgf", "w") as fc:
        for a in t:
            if a["peptide"] in exclude:
                n_excl += 1
                continue
            spectra = [("consensus", "", a["consensus"])] + [
                ("experimental", e.get("spectrum_id") or "", e) for e in a["experimental"]]
            for src, sid, s in spectra:
                mz = np.asarray(s["mz"], dtype=np.float64)
                it = np.asarray(s["intensity"], dtype=np.float64)
                order = np.argsort(mz, kind="stable")
                mz, it = mz[order], it[order]
                f = fc if src == "consensus" else fe
                if len(mz):
                    write_spec(f, row, a["analyte_id"], src, sid, float(a["precursor"] or 0),
                               a["charge"], mz, it)
                for k, v in (("row", row), ("analyte_id", str(a["analyte_id"] or "")),
                             ("peptide", a["peptide"]), ("charge", int(a["charge"])),
                             ("precursor", float(a["precursor"] or 0)), ("source", src),
                             ("spectrum_id", sid), ("n_peaks", len(mz)),
                             ("in_mp512", bool(0 < len(mz) <= MAX_PEAKS))):
                    meta[k].append(v)
                row += 1
    pq.write_table(pa.table(meta), out / "meta.parquet")
    n_mp = int(np.sum(meta["in_mp512"]))
    print(f"[export] {len(t):,} analytes, {n_excl:,} excluded; {row:,} spectra "
          f"({sum(s == 'experimental' for s in meta['source']):,} experimental); "
          f"{n_mp:,} with 0 < peaks <= {MAX_PEAKS}")

    # Alignment check against the prepared eval set.
    from datasets import load_from_disk
    prep = load_from_disk(PREPARED).select_columns(["analyte_id", "source", "peptide",
                                                    "charge"])
    keep = np.asarray(meta["in_mp512"])
    mine = list(zip(np.asarray(meta["analyte_id"])[keep], np.asarray(meta["source"])[keep],
                    np.asarray(meta["peptide"])[keep], np.asarray(meta["charge"])[keep]))
    theirs = list(zip(prep["analyte_id"], prep["source"], prep["peptide"], prep["charge"]))
    same = len(mine) == len(theirs) and all(
        (a, b, c, int(d)) == (w, x, y, int(z)) for (a, b, c, d), (w, x, y, z)
        in zip(mine, theirs))
    report = {"analytes": len(t), "excluded_analytes": n_excl, "spectra": row,
              "in_mp512": n_mp, "prepared_rows": len(theirs),
              "aligned_with_prepared": bool(same)}
    (out / "EXPORT.json").write_text(json.dumps(report, indent=1))
    print(f"[export] {report}")
    if not same:
        print("[export] WARNING: in_mp512 subset does NOT match the prepared set order")


if __name__ == "__main__":
    main(sys.argv[1])
