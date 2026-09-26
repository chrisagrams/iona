"""C15: Noble-lab multi-species benchmark (balanced), an unseen HIGH-RES HCD set.

    PYTHONPATH=. .venv/bin/python baselines_wip/noble_build.py OUT_DIR

Source: Zenodo 10.5281/zenodo.12819175, `nine-species-balanced.zip` (annotated MGF per
species; nine PRIDE projects, all Thermo Q Exactive; Tide + Percolator at 1% PSM FDR;
peptides made disjoint across species). H. sapiens is EXCLUDED until our pretraining
data's provenance is known (the human project is the one most likely to overlap it).

Per species: groups = peptide + charge with >= 2 spectra, each capped at MAX_PER_GROUP,
then whole groups drawn at random (seed 0) until PER_SPECIES spectra. Spectra above 512
peaks are TRIMMED to their 512 most intense peaks (m/z order kept; user OK'd 2026-09-25):
~52% of the yeast spectra are affected -- this copy is not peak-filtered, unlike C13's.
The trimmed fraction is reported. Every method sees the identical spectrum.

Modifications -> our notation: residue X+m -> X[m] (C+57.021 -> C[57.0215], M+15.995 ->
M[15.9949], N/Q+0.984 -> [0.9840]); N-terminal prefixes +42.011 / +43.006 / -17.027 (and
their combination) -> a leading [m]. Anything else stops the build.

Same two output formats as c11_build.py / nine_build.py.
"""
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ZIP = "/lus/flare/projects/UIC-HPC/khuss/msdelta/eval-data/nine-species-noble/nine-species-balanced.zip"
PROCESSOR = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-50m-production-01-checkpoint-220000"
MAX_PEAKS = 512
MAX_PER_GROUP = 20
PER_SPECIES = int(os.environ.get("NOBLE_PER_SPECIES", "5000"))
EXCLUDE = {"H.-sapiens"}
# NOBLE_SPECIES=Mus-musculus[,...]: build only these (e.g. the mouse 20k transfer set).
ONLY = {s for s in os.environ.get("NOBLE_SPECIES", "").split(",") if s}
RESIDUE_MODS = {"C+57.021": "C[57.0215]", "M+15.995": "M[15.9949]",
                "N+0.984": "N[0.9840]", "Q+0.984": "Q[0.9840]"}
NTERM = {"+42.011": 42.0106, "+43.006": 43.0058, "-17.027": -17.0265}


def to_notation(seq: str) -> str:
    """Noble SEQ= string -> our bracket notation; refuse anything unrecognised."""
    nterm = 0.0
    m = re.match(r"^((?:[+-]\d+\.\d+)+)", seq)
    if m:
        for part in re.findall(r"[+-]\d+\.\d+", m.group(1)):
            if part not in NTERM:
                raise SystemExit(f"unmapped N-terminal modification {part!r} in {seq!r}")
            nterm += NTERM[part]
        seq = seq[m.end():]
    for k, v in RESIDUE_MODS.items():
        seq = seq.replace(k, v)
    if re.search(r"[+-]\d", seq) or not re.fullmatch(r"(?:[A-Z](?:\[[\d.]+\])?)+", seq):
        raise SystemExit(f"unmapped modification in {seq!r}")
    return (f"[{nterm:.4f}]" if nterm else "") + seq


def read_mgf(text):
    spec = None
    for line in text.splitlines():
        if line.startswith("BEGIN IONS"):
            spec = {"mz": [], "it": []}
        elif line.startswith("END IONS"):
            yield spec; spec = None
        elif spec is not None:
            if "=" in line:
                k, v = line.split("=", 1); spec[k] = v
            elif line.strip():
                a, b = line.split()[:2]; spec["mz"].append(float(a)); spec["it"].append(float(b))


def main(out_dir, *_ignored):
    from datasets import Dataset
    from msdelta.grouped_retrieval import _process
    from msdelta.processing_msdelta import MSDeltaProcessor

    out = Path(out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    processor = MSDeltaProcessor.from_pretrained(PROCESSOR, max_peaks=MAX_PEAKS)
    zf = zipfile.ZipFile(ZIP)
    by_species = defaultdict(list)
    for name in sorted(zf.namelist()):
        if not name.endswith(".mgf"):
            continue
        species = name.split("/")[1]
        if species in EXCLUDE or (ONLY and species not in ONLY):
            continue
        run = Path(name).stem
        for i, s in enumerate(read_mgf(zf.read(name).decode())):
            by_species[species].append((run, i, s))
    recs, stats = [], {}
    rng = np.random.default_rng(0)
    for species in sorted(by_species):
        items = by_species[species]
        groups = defaultdict(list)
        for run, i, s in items:
            groups[(to_notation(s["SEQ"]), int(s["CHARGE"].rstrip("+")))].append((run, i, s))
        keys = [k for k, v in groups.items() if len(v) >= 2]
        order = rng.permutation(len(keys))
        chosen, n = [], 0
        for j in order:
            members = groups[keys[j]]
            members = [members[t] for t in rng.permutation(len(members))[:MAX_PER_GROUP]]
            chosen.append((keys[j], members)); n += len(members)
            if n >= PER_SPECIES:
                break
        trimmed = 0
        for (pep, z), members in chosen:
            for run, i, s in members:
                mz = np.asarray(s["mz"], np.float64); it = np.asarray(s["it"], np.float64)
                ok = (it > 0) & np.isfinite(mz); mz, it = mz[ok], it[ok]
                if len(mz) > MAX_PEAKS:
                    top = np.sort(np.argsort(it)[-MAX_PEAKS:]); mz, it = mz[top], it[top]
                    trimmed += 1
                if len(mz) == 0:
                    continue
                recs.append({"spectrum_id": f"{species}:{run}:{i}", "species": species,
                             "peptide": pep, "charge": z,
                             "precursor": float(s["PEPMASS"].split()[0]), "mz": mz,
                             "intensity": it})
        kept = sum(len(m) for _, m in chosen)
        stats[species] = {"annotated": len(items), "groups_ge2": len(keys),
                          "kept": kept, "groups_kept": len(chosen), "trimmed": trimmed}
        print(f"[noble] {species}: {len(items):,} spectra, {len(keys):,} groups >=2 -> "
              f"{kept:,} kept in {len(chosen):,} groups; {trimmed:,} trimmed to {MAX_PEAKS}",
              flush=True)

    prep = {k: [] for k in ("mz", "log_intensity", "peptide", "charge", "precursor",
                            "source", "analyte_id")}
    for r in recs:
        m, li = _process(processor, r["mz"], r["intensity"])
        prep["mz"].append(m); prep["log_intensity"].append(li)
        prep["peptide"].append(r["peptide"]); prep["charge"].append(r["charge"])
        prep["precursor"].append(r["precursor"]); prep["source"].append("experimental")
        prep["analyte_id"].append(f"{r['peptide']}_{r['charge']}")
    Dataset.from_dict(prep).save_to_disk(str(out / "prepared"))
    info = {"source": "zenodo 10.5281/zenodo.12819175 nine-species-balanced.zip",
            "excluded_species": sorted(EXCLUDE), "only_species": sorted(ONLY),
            "per_species": stats,
            "spectra": len(recs), "max_per_group": MAX_PER_GROUP,
            "per_species_target": PER_SPECIES, "trimmed_to_top512": True,
            "labels": "Tide + Percolator, 1% PSM FDR (Noble lab)"}
    (out / "prepared" / "PREPARED.json").write_text(json.dumps(info, indent=1))
    exp = out / "export"; exp.mkdir(parents=True)
    with open(exp / "experimental.mgf", "w") as f:
        for i, r in enumerate(recs):
            f.write("BEGIN IONS\n")
            f.write(f"TITLE=row={i};analyte_id={r['peptide']}_{r['charge']};"
                    f"source=experimental;spectrum_id={r['spectrum_id']}\n")
            f.write(f"PEPMASS={r['precursor']:.6f}\nCHARGE={r['charge']}+\nRTINSECONDS=0\n")
            for m, it in zip(r["mz"], r["intensity"]):
                f.write(f"{m:.5f} {it:.4f}\n")
            f.write("END IONS\n")
    (exp / "consensus.mgf").write_text("")
    pq.write_table(pa.table({
        "row": np.arange(len(recs)), "analyte_id": [f"{r['peptide']}_{r['charge']}" for r in recs],
        "peptide": [r["peptide"] for r in recs], "charge": [r["charge"] for r in recs],
        "precursor": [r["precursor"] for r in recs], "source": ["experimental"] * len(recs),
        "spectrum_id": [r["spectrum_id"] for r in recs], "species": [r["species"] for r in recs],
        "n_peaks": [len(r["mz"]) for r in recs], "in_mp512": [True] * len(recs)}),
        exp / "meta.parquet")
    (exp / "EXPORT.json").write_text(json.dumps(info, indent=1))
    print(f"[noble] wrote {len(recs):,} spectra to {out}/prepared and {out}/export", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
