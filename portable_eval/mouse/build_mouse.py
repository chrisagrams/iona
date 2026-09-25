"""Build the nine-species MOUSE benchmark (Noble lab, nine-species-balanced) for the yHydra comparison.

    python build_mouse.py nine-species-balanced.zip OUT_DIR [N]     # N: only the first N spectra (a quick test)

Source: Zenodo 10.5281/zenodo.12819175, nine-species-balanced.zip, folder Mus-musculus/
(9 MGF files, 25,490 annotated spectra; 6,843 modified peptides, 7,177 (peptide, charge) pairs;
precursor tolerance 10 ppm, fragment 0.05 Da).
Every annotated spectrum is kept as published (max 414 peaks, so nothing is trimmed).

Peptides are converted to msdelta notation: a residue modification follows its residue
(M+15.995 -> M[15.9949]); an N-terminal modification leads the sequence (-17.027Q... ->
[-17.0265]Q...; stacked N-terminal modifications are summed into one mass). Rounded masses in the MGF are replaced by their exact values; any other
modification string stops the build (never silently mapped).

Writes OUT_DIR/meta.parquet (row, spectrum_id, peptide, charge, precursor [m/z], source,
in_mp512, mz, intensity), OUT_DIR/experimental.mgf (TITLE=row=<n>;...) for yHydra, and an empty
OUT_DIR/consensus.mgf (the yHydra embedding script reads both).
"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EXACT = {"+57.021": 57.0215, "+15.995": 15.9949, "+0.984": 0.9840, "-17.027": -17.0265,
         "+43.006": 43.0058, "+42.011": 42.0106}
MOD = re.compile(r"[+-]\d+\.\d+")


def to_notation(seq: str) -> str:
    out, i, nterm = [], 0, 0.0
    while (lead := MOD.match(seq, i)):          # stacked N-terminal mods (+43.006-17.027Q...) are summed
        if lead.group() not in EXACT:
            raise SystemExit(f"unknown N-terminal modification {lead.group()} in {seq}")
        nterm += EXACT[lead.group()]; i = lead.end()
    if i:
        out.append(f"[{nterm:.4f}]")
    while i < len(seq):
        if not seq[i].isalpha():
            raise SystemExit(f"cannot parse {seq!r} at {i}")
        out.append(seq[i]); i += 1
        m = MOD.match(seq, i)
        if m:
            if m.group() not in EXACT:
                raise SystemExit(f"unknown modification {m.group()} in {seq}")
            out.append(f"[{EXACT[m.group()]:.4f}]"); i = m.end()
    return "".join(out)


def spectra(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if "/Mus-musculus/" in n and n.endswith(".mgf"))
        if not names:
            raise SystemExit("no Mus-musculus/*.mgf in the archive")
        for name in names:
            cur, mz, it = {}, [], []
            for line in z.read(name).decode().splitlines():
                if line == "BEGIN IONS":
                    cur, mz, it = {"file": Path(name).stem}, [], []
                elif line == "END IONS":
                    if cur.get("SEQ"):
                        yield cur, mz, it
                elif "=" in line and not line[0].isdigit():
                    k, v = line.split("=", 1); cur[k] = v
                elif line and line[0].isdigit():
                    a, b = line.split()[:2]; mz.append(float(a)); it.append(float(b))


def main(zip_path, out_dir, limit=0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cols = {k: [] for k in ("row", "spectrum_id", "peptide", "charge", "precursor", "source", "in_mp512",
                            "mz", "intensity")}
    with open(out / "experimental.mgf", "w") as mgf:
        for row, (p, mz, it) in enumerate(spectra(zip_path)):
            if limit and row >= limit:
                break
            z = int(p["CHARGE"].rstrip("+"))
            sid = f"{p['file']}:scan={p.get('SCANS', row)}"
            cols["row"].append(row); cols["spectrum_id"].append(sid); cols["peptide"].append(to_notation(p["SEQ"]))
            cols["charge"].append(z); cols["precursor"].append(float(p["PEPMASS"].split()[0]))
            cols["source"].append("experimental"); cols["in_mp512"].append(len(mz) <= 512)
            cols["mz"].append(mz); cols["intensity"].append(it)
            mgf.write(f"BEGIN IONS\nTITLE=row={row};spectrum_id={sid}\nPEPMASS={cols['precursor'][-1]}\nCHARGE={z}+\n")
            mgf.write("".join(f"{a:.5f} {b:.2f}\n" for a, b in zip(mz, it)))
            mgf.write("END IONS\n")
    (out / "consensus.mgf").write_text("")    # embed_yhydra reads experimental + consensus; mouse has none
    pq.write_table(pa.table(cols), out / "meta.parquet")
    peps = {(p, c) for p, c in zip(cols["peptide"], cols["charge"])}
    print(f"mouse: {len(cols['row']):,} spectra, {len(set(cols['peptide'])):,} peptides, "
          f"{len(peps):,} (peptide, charge) candidates, max peaks {max(map(len, cols['mz']))} -> {out}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else 0)
