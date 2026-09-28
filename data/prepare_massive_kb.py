#!/usr/bin/env python
"""MassIVE-KB v1 -> one-spectrum-per-row contrastive training data, eval peptides removed.

Source: `chrisagrams/massive_kb_v1_shuffled` (revision REVISION below), read as its parquet
shards straight from the HF hub cache -- NOT through `datasets.load_dataset`, which would
first convert all 50 GB into a second Arrow cache. Each row there is one spectrum:
`m/z`, `int` (raw intensity) and `peptide_charge` (`SEQ_z`, mods inline as `C[+57.021]`,
`[-17.027]QSP...`). There is no measured precursor m/z and no run/source information.

Three steps (subcommands), each safe to re-run:

  exclusion  Collect every peptide of every evaluation set we report on (ms-contrastive-100k
             validation + test, the replicate corpus, the nine-species OOD set, mouse, human,
             yeast (canonical), HEK, HCT116) into one JSON with per-source counts. Small; run
             it once and pass it to `prepare`.
  prepare    Stream every source shard in parallel (one process per shard, one row group in
             memory at a time) and, per spectrum:
               1. split `peptide_charge` into peptide and charge;
               2. DROP it if its peptide SEQUENCE (modifications stripped, I and L collapsed
                  unless --no-il-collapse) is in the exclusion set -- counted per source;
               3. map the modification notation to ours (MOD_MAP); an unknown token, or a
                  known token on a residue it does not belong to, drops the spectrum and is
                  counted by its exact string -- never guessed;
               4. precursor m/z = THEORETICAL, from the mapped peptide and the charge. There
                  is no measured value, so a precursor-mass filter can never fail on this
                  data: any filter-failure analysis over it is empty by construction;
               5. process the spectrum exactly as training does for ms-contrastive-100k
                  (msdelta.data.grouped_retrieval.build_grouped_split): spectra with more than
                  --max-peaks (512) peaks are DROPPED, not truncated (--oversize top keeps the
                  512 most intense instead, NOT what training does); mz kept as is;
                  log_intensity = log1p(int) / max(log1p(int)); empty, non-finite, negative or
                  all-zero spectra are dropped as the processor would reject them. Output is
                  already processed (like the prepared eval sets), stored as float32;
               6. assign the train / validation / test split (see --split-policy);
               7. group by (peptide, charge) = peptide_key; group sizes are recorded.
             Output is written to <out>.partial and renamed to <out> only when complete;
             an existing <out> is never overwritten. --resume continues a .partial.
  check      Validate a finished output: counts vs manifest, schema, no excluded sequence,
             split disjointness, group sizes, and processed-value sanity on a sample.

Output layout (<out>, e.g. $S/data/massive-kb-contrastive):

  train/ validation/ test/     parquet parts <srcsplit>-<shard>.parquet, columns
                               analyte_id, peptide, charge, precursor, mz, source,
                               log_intensity (same names as the prepared eval sets)
  group_sizes.parquet          split, analyte_id (peptide_key), count
  overlap_report.json / .md    how many MassIVE-KB spectra / sequences each eval set removes
  exclusion.json               copy of the exclusion set used
  manifest.json                source revision, counts, drops by reason, code commit (last)
  _shards/                     per-source-shard stats (what --resume uses)

Run it on a compute node (pbs/prepare_massive_kb.pbs), never on a login node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Run by path (python data/prepare_massive_kb.py), so make the repo (or the job's code
# snapshot this file sits in) importable; only the exclusion step imports msdelta.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

REPO_ID = "chrisagrams/massive_kb_v1_shuffled"
REVISION = "891f42f72e84bd38c97b0b4356115d146fa51507"
SOURCE_SPLITS = {"train": "train", "validation": "val", "test": "test"}   # ours -> directory
EXPECTED_ROWS = {"train": 28_508_636, "validation": 1_000_234, "test": 996_027}
SPLITS = ("train", "validation", "test")
S_DEFAULT = "/lus/flare/projects/UIC-HPC/khuss/msdelta"

# MassIVE-KB token -> our token. Key: (residue, token), residue "^" = peptide N-terminus.
# Our notation is the one every eval set uses: unsigned for positive masses, 4 decimals,
# N-terminal mods as a leading bracket. Every entry was seen in a sample of the source;
# anything else (including the combined N-terminal carbamyl + NH3 loss, which the Noble
# sets write [25.9793]) is dropped and reported, so the table can be extended on evidence.
MOD_MAP: dict[tuple[str, str], str] = {
    ("C", "[+57.021]"): "[57.0215]",     # carbamidomethyl
    ("M", "[+15.995]"): "[15.9949]",     # oxidation
    ("N", "[+0.984]"): "[0.9840]",       # deamidation
    ("Q", "[+0.984]"): "[0.9840]",       # deamidation
    ("^", "[+42.011]"): "[42.0106]",     # N-terminal acetylation
    ("^", "[+43.006]"): "[43.0058]",     # N-terminal carbamylation
    ("^", "[-17.027]"): "[-17.0265]",    # N-terminal NH3 loss (pyro-Glu from Q / pyro-cmC)
}
RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")
_TOKEN = re.compile(r"\[[^\]]*\]")
_PC = re.compile(r"^(.+)_(\d+)$")

WATER = 18.010565                        # as msdelta.rescoring.reranking.peptide_neutral_mass
PROTON = 1.007276466812                  # msdelta.data.chemistry.PROTON_MASS

DROP_REASONS = ("bad_format", "bad_charge", "excluded_eval", "unknown_residue",
                "unknown_modification", "oversize", "invalid_spectrum")


# ------------------------------------------------------------------------ pure helpers

class Drop(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def split_peptide_charge(value: str) -> tuple[str, int]:
    """`[-17.027]QSPLGR_2` -> (`[-17.027]QSPLGR`, 2)."""
    match = _PC.match(value or "")
    if not match:
        raise Drop("bad_format" if "_" not in (value or "") else "bad_charge", value)
    charge = int(match.group(2))
    if not 1 <= charge <= 10:
        raise Drop("bad_charge", value)
    return match.group(1), charge


def sequence_key(peptide: str, il_collapse: bool) -> str:
    """Bare residue sequence: bracketed mods and any non-residue marker (the replicate
    corpus's `n` N-terminus) removed; I -> L when il_collapse. Works on either notation."""
    bare = re.sub(r"[^A-Z]", "", _TOKEN.sub("", peptide))
    return bare.replace("I", "L") if il_collapse else bare


def map_modifications(peptide: str) -> tuple[str, list[str]]:
    """MassIVE-KB peptide -> our notation, plus the source tokens seen (with position).

    Raises Drop("unknown_residue") / Drop("unknown_modification", <token in context>)."""
    out: list[str] = []
    seen: list[str] = []
    last = "^"                    # residue a bracket would bind to; "^" before any residue
    index = 0
    while index < len(peptide):
        char = peptide[index]
        if char == "[":
            end = peptide.find("]", index)
            if end < 0:
                raise Drop("unknown_modification", peptide[index:])
            token = peptide[index:end + 1]
            context = f"{last}{token}"
            ours = MOD_MAP.get((last, token))
            if ours is None:
                raise Drop("unknown_modification", context)
            if last == "^" and out and out[-1].startswith("["):
                raise Drop("unknown_modification", f"^{out[-1]}{token}")   # stacked N-term
            out.append(ours)
            seen.append(context)
            index = end + 1
            continue
        if char not in RESIDUES:
            raise Drop("unknown_residue", char)
        out.append(char)
        last = char
        index += 1
    if last == "^":
        raise Drop("bad_format", peptide)       # no residues at all
    return "".join(out), seen


_RESIDUE_MASSES: dict[str, float] = {}


def neutral_mass(peptide: str) -> float:
    """Same formula and residue masses as msdelta.rescoring.reranking.peptide_neutral_mass
    (tested equal). Reads pyteomics directly: importing msdelta pulls in torch and
    transformers (~10 s and a GPU probe per worker) for a table of 20 numbers."""
    if not _RESIDUE_MASSES:
        from pyteomics import mass
        _RESIDUE_MASSES.update({r: mass.std_aa_mass[r] for r in RESIDUES})
    RESIDUE_MASSES = _RESIDUE_MASSES
    mods = sum(float(x) for x in re.findall(r"\[([-+]?\d+\.?\d*)\]", peptide))
    return sum(RESIDUE_MASSES[r] for r in _TOKEN.sub("", peptide)) + mods + WATER


def precursor_mz(peptide: str, charge: int) -> float:
    """THEORETICAL precursor m/z: MassIVE-KB stores no measured one."""
    return (neutral_mass(peptide) + charge * PROTON) / charge


def peptide_key(peptide: str, charge: int) -> str:
    """msdelta.models.peptide_encoder.peptide_key (charge-aware), duplicated to stay torch-free."""
    return f"{peptide}_{charge}"


def assign_split(seq_il: str, validation_fraction: float, test_fraction: float) -> str:
    """Stable split by a hash of the I/L-collapsed bare sequence: every charge, mod form
    and I/L variant of a sequence lands in one split, on every run and every machine."""
    h = int.from_bytes(hashlib.blake2b(seq_il.encode(), digest_size=8).digest(), "big")
    u = h / 2.0 ** 64
    if u < validation_fraction:
        return "validation"
    if u < validation_fraction + test_fraction:
        return "test"
    return "train"


def parse_row(value: str, il_collapse: bool, excluded: dict[str, list[str]]) -> dict:
    """Everything that depends only on `peptide_charge`. Raises Drop."""
    raw, charge = split_peptide_charge(value)
    seq = sequence_key(raw, il_collapse)
    hit = excluded.get(seq)
    if hit:
        raise Drop("excluded_eval", seq)
    peptide, tokens = map_modifications(raw)
    return {"peptide": peptide, "charge": charge, "tokens": tokens,
            "seq_il": sequence_key(raw, True),
            "precursor": precursor_mz(peptide, charge)}


def process_spectra(offsets: np.ndarray, mz: np.ndarray, intensity: np.ndarray,
                    max_peaks: int, oversize: str):
    """Vectorised MSDeltaProcessor._process_one over one record batch.

    Returns (status, new_offsets, mz_out, log_intensity_out) where status per row is
    "" (kept), "oversize" or "invalid_spectrum"; the value arrays hold kept rows only."""
    offsets = offsets.astype(np.int64)
    lengths = np.diff(offsets)
    n = len(lengths)
    status = np.array([""] * n, dtype=object)
    finite = np.isfinite(mz) & np.isfinite(intensity)
    bad_peak = (~finite) | (intensity < 0)
    starts = offsets[:-1]
    nonempty = lengths > 0
    any_bad = np.zeros(n, dtype=bool)
    row_max = np.zeros(n, dtype=np.float32)
    if len(mz):
        idx = starts[nonempty]
        any_bad[nonempty] = np.logical_or.reduceat(bad_peak, idx)
        safe = np.where(finite, intensity, 0).astype(np.float32)
        row_max[nonempty] = np.maximum.reduceat(safe, idx)
    invalid = (~nonempty) | any_bad | (row_max <= 0)
    over = lengths > max_peaks
    status[invalid] = "invalid_spectrum"
    if oversize == "drop":
        status[over & ~invalid] = "oversize"
    out_mz, out_li, out_len = [], [], []
    for row in np.flatnonzero(status == ""):
        a, b = offsets[row], offsets[row + 1]
        m = mz[a:b].astype(np.float32)
        it = intensity[a:b].astype(np.float32)
        if b - a > max_peaks:                          # oversize == "top"
            top = np.sort(np.argsort(-it, kind="stable")[:max_peaks])
            m, it = m[top], it[top]
        li = np.log1p(it)
        li = li / max(li.max(), np.float32(1e-8))
        out_mz.append(m)
        out_li.append(li.astype(np.float32))
        out_len.append(len(m))
    new_offsets = np.zeros(len(out_len) + 1, dtype=np.int64)
    np.cumsum(out_len, out=new_offsets[1:])
    cat = (lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.float32))
    return status, new_offsets, cat(out_mz), cat(out_li)


# ------------------------------------------------------------------------ exclusion step

def default_sources(s_root: str) -> dict[str, str]:
    e, b = f"{s_root}/eval-data", f"{s_root}/baselines"
    return {
        "ms-contrastive-100k-validation": f"{e}/ms-contrastive-100k-validation-mp512",
        "ms-contrastive-100k-test": f"{e}/ms-contrastive-100k-test-mp512",
        "replicate-corpus": "hf:chrisagrams/ms2-peptide-replicate-retrieval",
        "nine-species-ood-8species": f"{b}/nine_oodval20k/prepared",
        "noble-mouse": f"{b}/noble_mouse20k/prepared",
        "noble-human": f"{b}/noble_human20k/prepared",
        "nine-species-yeast-canonical": f"{b}/nine_yeast/prepared",
        "hek-c11": f"{b}/c11_cap20/prepared",
        "hct116-c14": f"{b}/c14_hct116_20k/prepared",
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source_peptides(spec: str) -> tuple[list[str], dict]:
    """Peptide column of one eval source: a `save_to_disk` directory or `hf:<repo>`."""
    if spec.startswith("hf:"):
        repo = spec[3:]
        if repo == "chrisagrams/ms2-peptide-replicate-retrieval":
            from msdelta.data.grouped_retrieval import replicate_corpus_peptides
            peptides = sorted(replicate_corpus_peptides(repo))
        else:
            from datasets import load_dataset
            raw = load_dataset(repo)
            peptides = [p for split in raw.values() for p in split["peptide"]]
        return peptides, {"path": spec}
    from datasets import load_from_disk
    path = Path(spec)
    ds = load_from_disk(str(path))
    info = {"path": str(path), "rows": len(ds),
            "arrow_sha256": {p.name: _sha256(p) for p in sorted(path.glob("*.arrow"))}}
    return list(ds["peptide"]), info


def build_exclusion(sources: dict[str, str]) -> dict:
    peptides, info = {}, {}
    for name, spec in sources.items():
        rows, meta = load_source_peptides(spec)
        distinct = sorted(set(rows))
        peptides[name] = distinct
        info[name] = {**meta, "peptides": len(distinct),
                      "sequences": len({sequence_key(p, False) for p in distinct}),
                      "sequences_il": len({sequence_key(p, True) for p in distinct})}
        print(f"[exclusion] {name}: {info[name]}", flush=True)
    union = {p for ps in peptides.values() for p in ps}
    return {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sources": info,
            "union": {"peptides": len(union),
                      "sequences": len({sequence_key(p, False) for p in union}),
                      "sequences_il": len({sequence_key(p, True) for p in union})},
            "peptides": peptides}


def exclusion_index(exclusion: dict, il_collapse: bool) -> dict[str, list[str]]:
    """sequence key -> the eval sources containing it."""
    index: dict[str, list[str]] = {}
    for name, peptides in exclusion["peptides"].items():
        for key in {sequence_key(p, il_collapse) for p in peptides}:
            index.setdefault(key, []).append(name)
    return index


def cmd_exclusion(args) -> int:
    out = Path(args.out)
    target = out / "exclusion.json"
    if target.exists():
        print(f"refusing to overwrite {target}", file=sys.stderr)
        return 2
    sources = {} if args.no_defaults else default_sources(args.scratch_root)
    for item in args.source or []:
        name, _, spec = item.partition("=")
        sources[name] = spec
    for name in args.drop_source or []:
        sources.pop(name, None)
    data = build_exclusion(sources)
    out.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.rename(target)
    counts = {k: v for k, v in data.items() if k != "peptides"}
    (out / "exclusion_counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts, indent=1))
    return 0


# ------------------------------------------------------------------------ prepare step

SCHEMA = None


def _schema():
    import pyarrow as pa
    global SCHEMA
    if SCHEMA is None:
        SCHEMA = pa.schema([("analyte_id", pa.string()), ("peptide", pa.string()),
                            ("charge", pa.int16()), ("precursor", pa.float32()),
                            ("mz", pa.list_(pa.float32())), ("source", pa.string()),
                            ("log_intensity", pa.list_(pa.float32()))])
    return SCHEMA


_CFG: dict = {}


def _init_worker(cfg):
    _CFG.clear()
    _CFG.update(cfg)
    _CFG["index"] = exclusion_index(cfg["exclusion"], cfg["il_collapse"])
    _CFG["strict_index"] = exclusion_index(cfg["exclusion"], False)


def process_shard(task: tuple[str, str]) -> dict:
    """One source shard -> parts in each target split + a stats JSON. Runs in a worker."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    src_split, path = task
    cfg = _CFG
    partial = Path(cfg["partial"])
    stem = f"{src_split}-{Path(path).stem.split('_')[-1]}"
    stats_path = partial / "_shards" / f"{stem}.json"
    t0 = time.time()
    drops = Counter()
    unknown = Counter()
    tokens = Counter()
    excl_spectra = Counter()                # per eval source, active key
    excl_spectra_strict = Counter()         # per eval source, no I/L collapse
    excl_keys: dict[str, set] = {}
    kept_by_split = Counter()
    moved = Counter()                       # "src->dst"
    charges = Counter()
    groups = Counter()                      # (split, analyte_id)
    peaks_sum = 0
    rows_in = 0
    writers = {}
    cache: dict[str, dict | Drop] = {}
    limit = cfg["limit"]
    pf = pq.ParquetFile(path)
    try:
        for batch in pf.iter_batches(batch_size=cfg["batch_size"],
                                     columns=["m/z", "int", "peptide_charge"]):
            if limit and rows_in >= limit:
                break
            if limit and rows_in + batch.num_rows > limit:
                batch = batch.slice(0, limit - rows_in)
            rows_in += batch.num_rows
            names = batch.column(2).to_pylist()
            parsed = []
            for value in names:
                hit = cache.get(value)
                if hit is None:
                    try:
                        hit = parse_row(value, cfg["il_collapse"], cfg["index"])
                    except Drop as drop:
                        hit = drop
                    cache[value] = hit
                parsed.append(hit)
            mz_col, int_col = batch.column(0), batch.column(1)
            offsets = mz_col.offsets.to_numpy()
            if not np.array_equal(offsets, int_col.offsets.to_numpy()):
                raise ValueError(f"{path}: m/z and int offsets differ")
            base = offsets[0]
            mz = mz_col.values.to_numpy(zero_copy_only=False)[base:offsets[-1]]
            it = int_col.values.to_numpy(zero_copy_only=False)[base:offsets[-1]]
            status, new_off, out_mz, out_li = process_spectra(
                offsets - base, mz, it, cfg["max_peaks"], cfg["oversize"])
            # Row-level decisions: parse drops take precedence over spectrum drops.
            keep_rows, keep_pos = [], []
            kept_i = -1
            for i, p in enumerate(parsed):
                if status[i] == "":
                    kept_i += 1
                if isinstance(p, Drop):
                    drops[p.reason] += 1
                    if p.reason == "unknown_modification":
                        unknown[p.detail] += 1
                    elif p.reason == "unknown_residue":
                        unknown[f"residue:{p.detail}"] += 1
                    elif p.reason == "excluded_eval":
                        for src in cfg["index"][p.detail]:
                            excl_spectra[src] += 1
                            excl_keys.setdefault(src, set()).add(p.detail)
                        raw = split_peptide_charge(names[i])[0]
                        for src in cfg["strict_index"].get(sequence_key(raw, False), []):
                            excl_spectra_strict[src] += 1
                    continue
                if status[i] != "":
                    drops[status[i]] += 1
                    continue
                keep_rows.append(i)
                keep_pos.append(kept_i)
            if not keep_rows:
                continue
            # Slice the processed values down to rows that also passed parsing.
            sel_off = np.zeros(len(keep_pos) + 1, dtype=np.int64)
            lens = new_off[np.array(keep_pos) + 1] - new_off[np.array(keep_pos)]
            np.cumsum(lens, out=sel_off[1:])
            gather = np.concatenate([np.arange(new_off[k], new_off[k + 1]) for k in keep_pos])
            v_mz, v_li = out_mz[gather], out_li[gather]
            by_split: dict[str, list[int]] = {}
            for j, i in enumerate(keep_rows):
                p = parsed[i]
                if cfg["split_policy"] == "peptide":
                    dst = assign_split(p["seq_il"], cfg["validation_fraction"],
                                       cfg["test_fraction"])
                else:
                    dst = src_split
                by_split.setdefault(dst, []).append(j)
            for dst, js in by_split.items():
                rows = [parsed[keep_rows[j]] for j in js]
                ids = [peptide_key(r["peptide"], r["charge"]) for r in rows]
                sub_off = np.zeros(len(js) + 1, dtype=np.int64)
                np.cumsum(lens[js], out=sub_off[1:])
                take = np.concatenate([np.arange(sel_off[j], sel_off[j + 1]) for j in js])
                table = pa.table({
                    "analyte_id": pa.array(ids, pa.string()),
                    "peptide": pa.array([r["peptide"] for r in rows], pa.string()),
                    "charge": pa.array([r["charge"] for r in rows], pa.int16()),
                    "precursor": pa.array([r["precursor"] for r in rows], pa.float32()),
                    "mz": pa.ListArray.from_arrays(pa.array(sub_off.astype(np.int32)),
                                                   pa.array(v_mz[take], pa.float32())),
                    "source": pa.array(["experimental"] * len(js), pa.string()),
                    "log_intensity": pa.ListArray.from_arrays(
                        pa.array(sub_off.astype(np.int32)), pa.array(v_li[take], pa.float32())),
                }, schema=_schema())
                if dst not in writers:
                    (partial / dst).mkdir(parents=True, exist_ok=True)
                    writers[dst] = pq.ParquetWriter(partial / dst / f"{stem}.parquet",
                                                    _schema(), compression="zstd")
                writers[dst].write_table(table)
                kept_by_split[dst] += len(js)
                moved[f"{src_split}->{dst}"] += len(js)
                peaks_sum += int(sub_off[-1])
                for r, key in zip(rows, ids):
                    charges[r["charge"]] += 1
                    groups[(dst, key)] += 1
                    for t in r["tokens"]:
                        tokens[t] += 1
    finally:
        for w in writers.values():
            w.close()
    if groups:
        import pyarrow as pa
        import pyarrow.parquet as pq
        keys = list(groups)
        pq.write_table(pa.table({"split": [k[0] for k in keys],
                                 "analyte_id": [k[1] for k in keys],
                                 "count": pa.array([groups[k] for k in keys], pa.int64())}),
                       partial / "_shards" / f"{stem}.groups.parquet")
    stats = {"shard": stem, "path": path, "source_split": src_split, "rows_in": rows_in,
             "rows_in_file": pf.metadata.num_rows, "kept": sum(kept_by_split.values()),
             "kept_by_split": dict(kept_by_split), "moved": dict(moved),
             "drops": dict(drops), "unknown": dict(unknown), "tokens": dict(tokens),
             "excluded_spectra": dict(excl_spectra),
             "excluded_spectra_strict": dict(excl_spectra_strict),
             "excluded_keys": {k: sorted(v) for k, v in excl_keys.items()},
             "charges": {str(k): v for k, v in charges.items()}, "peaks_sum": peaks_sum,
             "seconds": round(time.time() - t0, 2)}
    tmp = stats_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats))
    tmp.rename(stats_path)                    # written LAST: marks the shard done
    return stats


def source_dir_default() -> Path:
    hf = os.environ.get("HF_HOME", f"{S_DEFAULT}/huggingface")
    return Path(hf) / "hub" / f"datasets--{REPO_ID.replace('/', '--')}" / "snapshots" / REVISION


def list_tasks(source_dir: Path, splits, shards: int) -> list[tuple[str, str]]:
    tasks = []
    for split in splits:
        files = sorted((source_dir / SOURCE_SPLITS[split]).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet shards under {source_dir / SOURCE_SPLITS[split]}")
        tasks += [(split, str(f)) for f in (files[:shards] if shards else files)]
    return tasks


def code_commit() -> dict:
    snap = os.environ.get("MSDELTA_CODE_DIR")
    if snap and (Path(snap) / "SNAPSHOT.txt").exists():
        text = (Path(snap) / "SNAPSHOT.txt").read_text()
        return {"snapshot": snap, "snapshot_txt": text}
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--",
                                "data/prepare_massive_kb.py"],
                               capture_output=True, text=True).stdout.strip()
    except OSError:
        commit, dirty = "", ""
    return {"repo": str(repo), "commit": commit, "script_dirty": bool(dirty)}


def summarise_groups(partial: Path) -> tuple[dict, "object"]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    parts = sorted((partial / "_shards").glob("*.groups.parquet"))
    if not parts:
        table = pa.table({"split": pa.array([], pa.string()),
                          "analyte_id": pa.array([], pa.string()),
                          "count": pa.array([], pa.int64())})
    else:
        agg = pa.concat_tables(pq.read_table(p) for p in parts) \
                .group_by(["split", "analyte_id"]).aggregate([("count", "sum")])
        table = pa.table({"split": agg["split"], "analyte_id": agg["analyte_id"],
                          "count": agg["count_sum"]}).sort_by([("split", "ascending"),
                                                                ("analyte_id", "ascending")])
    summary = {}
    splits = table.column("split").to_numpy(zero_copy_only=False)
    counts = table.column("count").to_numpy()
    for split in SPLITS:
        c = counts[splits == split]
        if len(c) == 0:
            continue
        summary[split] = {
            "groups": int(len(c)), "spectra": int(c.sum()),
            "singleton_groups": int((c == 1).sum()),
            "spectra_in_groups_ge2": int(c[c >= 2].sum()),
            "size_quantiles": {q: float(np.quantile(c, float(q)))
                               for q in ("0.5", "0.9", "0.99", "1.0")},
            "size_histogram": {b: int(n) for b, n in zip(
                ["1", "2", "3", "4-7", "8-15", "16-63", "64-255", "256+"],
                np.histogram(c, [1, 2, 3, 4, 8, 16, 64, 256, np.inf])[0])},
        }
    return summary, table


def overlap_report(shards: list[dict], exclusion: dict, il_collapse: bool) -> dict:
    spectra, strict, keys = Counter(), Counter(), {}
    for s in shards:
        spectra.update(s["excluded_spectra"])
        strict.update(s["excluded_spectra_strict"])
        for k, v in s["excluded_keys"].items():
            keys.setdefault(k, set()).update(v)
    union_keys = set().union(*keys.values()) if keys else set()
    report = {"key": "sequence, I/L collapsed" if il_collapse else "sequence",
              "note": "spectra counts are per source and overlap (a spectrum whose sequence "
                      "is in two eval sets is counted under both); 'total' is unique",
              "total_spectra_removed": sum(s["drops"].get("excluded_eval", 0) for s in shards),
              "total_sequences_removed": len(union_keys), "sources": {}}
    for name, info in exclusion["sources"].items():
        n_eval = info["sequences_il" if il_collapse else "sequences"]
        found = len(keys.get(name, ()))
        report["sources"][name] = {
            "eval_sequences": n_eval,
            "eval_sequences_in_massive_kb": found,
            "fraction_of_eval_sequences_in_massive_kb": round(found / max(n_eval, 1), 4),
            "massive_kb_spectra_removed": spectra.get(name, 0),
            "massive_kb_spectra_matching_without_il_collapse": strict.get(name, 0),
        }
    return report


def report_markdown(report: dict) -> str:
    lines = [f"# MassIVE-KB overlap with our evaluation sets (key: {report['key']})", "",
             f"Removed in total: {report['total_spectra_removed']:,} spectra, "
             f"{report['total_sequences_removed']:,} distinct sequences. {report['note']}.", "",
             "| eval set | eval sequences | of which in MassIVE-KB | fraction | "
             "MassIVE-KB spectra removed | same, no I/L collapse |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, r in report["sources"].items():
        lines.append(f"| {name} | {r['eval_sequences']:,} | {r['eval_sequences_in_massive_kb']:,}"
                     f" | {r['fraction_of_eval_sequences_in_massive_kb']:.1%} | "
                     f"{r['massive_kb_spectra_removed']:,} | "
                     f"{r['massive_kb_spectra_matching_without_il_collapse']:,} |")
    return "\n".join(lines) + "\n"


def cmd_prepare(args) -> int:
    from multiprocessing import get_context

    out = Path(args.out)
    partial = out.with_name(out.name + ".partial")
    if out.exists():
        print(f"refusing to overwrite existing output {out}", file=sys.stderr)
        return 2
    if partial.exists() and not args.resume:
        print(f"{partial} exists (an interrupted run?); pass --resume to continue it. "
              f"It is never deleted automatically.", file=sys.stderr)
        return 2
    exclusion_path = Path(args.exclusion)
    if exclusion_path.is_dir():
        exclusion_path = exclusion_path / "exclusion.json"
    exclusion = json.loads(exclusion_path.read_text())
    source_dir = Path(args.source_dir) if args.source_dir else source_dir_default()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    tasks = list_tasks(source_dir, splits, args.shards)
    settings = {
        "max_peaks": args.max_peaks, "oversize": args.oversize,
        "il_collapse": args.il_collapse, "split_policy": args.split_policy,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction, "limit": args.limit,
        "shards": args.shards, "splits": splits, "batch_size": args.batch_size,
    }
    (partial / "_shards").mkdir(parents=True, exist_ok=True)
    settings_path = partial / "_settings.json"
    if settings_path.exists():
        previous = json.loads(settings_path.read_text())
        if previous != settings:
            print(f"--resume with different settings: {previous} vs {settings}",
                  file=sys.stderr)
            return 2
    else:
        settings_path.write_text(json.dumps(settings, indent=1))
    done = {p.stem for p in (partial / "_shards").glob("*.json")}
    todo = [t for t in tasks
            if f"{t[0]}-{Path(t[1]).stem.split('_')[-1]}" not in done]
    print(f"[prepare] {len(tasks)} shards, {len(tasks) - len(todo)} already done, "
          f"{len(todo)} to go, {args.workers} workers; source {source_dir}", flush=True)
    cfg = {**settings, "exclusion": exclusion, "partial": str(partial)}
    t0 = time.time()
    if todo:
        if args.workers <= 1:
            _init_worker(cfg)
            results = map(process_shard, todo)
        else:
            pool = get_context("fork").Pool(args.workers, initializer=_init_worker,
                                            initargs=(cfg,), maxtasksperchild=8)
            results = pool.imap_unordered(process_shard, todo)
        for n, stats in enumerate(results, 1):
            print(f"[prepare] {n}/{len(todo)} {stats['shard']}: {stats['rows_in']:,} in, "
                  f"{stats['kept']:,} kept, {stats['seconds']}s "
                  f"(elapsed {time.time() - t0:.0f}s)", flush=True)
        if args.workers > 1:
            pool.close()
            pool.join()
    wanted = {f"{t[0]}-{Path(t[1]).stem.split('_')[-1]}" for t in tasks}
    shards = [json.loads((partial / "_shards" / f"{s}.json").read_text()) for s in sorted(wanted)]
    return finalise(out, partial, shards, exclusion, exclusion_path, source_dir, settings,
                    time.time() - t0)


def finalise(out, partial, shards, exclusion, exclusion_path, source_dir, settings, seconds):
    import pyarrow.parquet as pq

    group_summary, group_table = summarise_groups(partial)
    pq.write_table(group_table, partial / "group_sizes.parquet")
    report = overlap_report(shards, exclusion, settings["il_collapse"])
    (partial / "overlap_report.json").write_text(json.dumps(report, indent=1))
    (partial / "overlap_report.md").write_text(report_markdown(report))
    shutil.copyfile(exclusion_path, partial / "exclusion.json")

    total = lambda key: sum(s[key] for s in shards)          # noqa: E731
    merge = lambda key: dict(sum((Counter(s[key]) for s in shards), Counter()))  # noqa: E731
    rows_in_by_split = Counter()
    rows_file_by_split = Counter()
    for s in shards:
        rows_in_by_split[s["source_split"]] += s["rows_in"]
        rows_file_by_split[s["source_split"]] += s["rows_in_file"]
    files = {split: sorted(p.name for p in (partial / split).glob("*.parquet"))
             for split in SPLITS if (partial / split).exists()}
    kept = merge("kept_by_split")
    drops = merge("drops")
    manifest = {
        "dataset": "massive-kb-contrastive",
        "source": {"repo": REPO_ID, "revision": REVISION, "dir": str(source_dir),
                   "rows_read_by_split": dict(rows_in_by_split),
                   "rows_in_files_read_by_split": dict(rows_file_by_split),
                   "expected_rows_full_dataset": EXPECTED_ROWS,
                   "shards_read": len(shards)},
        "settings": settings,
        "precursor": "THEORETICAL m/z from the mapped peptide and charge (no measured value "
                     "in the source): precursor-mass filters cannot fail on this data, so a "
                     "filter-failure analysis over it is empty by construction",
        "processing": "already processed like msdelta.data.grouped_retrieval.build_grouped_split: "
                      "mz as is; log_intensity = log1p(int)/max(log1p(int)); float32; spectra "
                      f"over max_peaks={settings['max_peaks']} "
                      + ("DROPPED (as training does)" if settings["oversize"] == "drop"
                         else "cut to the most intense peaks (NOT what training does)"),
        "rows_in": total("rows_in"), "rows_kept": total("kept"), "kept_by_split": kept,
        "moved_between_splits": merge("moved"),
        "dropped_by_reason": {r: drops.get(r, 0) for r in DROP_REASONS},
        "unknown_tokens": merge("unknown"),
        "modification_tokens_kept": merge("tokens"),
        "mod_map": {f"{k[0]}{k[1]}": v for k, v in MOD_MAP.items()},
        "charges_kept": merge("charges"),
        "mean_peaks_kept": round(total("peaks_sum") / max(total("kept"), 1), 2),
        "groups": group_summary,
        "exclusion": {"file": str(exclusion_path), "sha256": _sha256(exclusion_path),
                      "sources": exclusion["sources"], "union": exclusion["union"]},
        "overlap": {k: v for k, v in report.items() if k != "sources"},
        "files": files,
        "code": code_commit(),
        "seconds": round(seconds, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    balance = manifest["rows_in"] - manifest["rows_kept"] - sum(drops.values())
    manifest["unaccounted_rows"] = balance
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if out.exists():
        print(f"refusing to overwrite {out} (appeared while running)", file=sys.stderr)
        return 2
    partial.rename(out)
    print(json.dumps({k: manifest[k] for k in ("rows_in", "rows_kept", "kept_by_split",
                                                 "dropped_by_reason", "unknown_tokens")},
                     indent=1))
    print(report_markdown(report))
    print(f"[prepare] wrote {out}")
    return 0 if balance == 0 else 1


# ------------------------------------------------------------------------ check step

def cmd_check(args) -> int:
    import pyarrow.parquet as pq

    out = Path(args.out)
    problems: list[str] = []
    manifest = json.loads((out / "manifest.json").read_text())
    settings = manifest["settings"]
    exclusion_file = out / "exclusion.json"
    if _sha256(exclusion_file) != manifest["exclusion"]["sha256"]:
        problems.append("exclusion.json does not match the manifest's sha256")
    exclusion = json.loads(exclusion_file.read_text())
    index = exclusion_index(exclusion, settings["il_collapse"])
    if manifest.get("unaccounted_rows", 0) != 0:
        problems.append(f"manifest: {manifest['unaccounted_rows']} rows unaccounted for")
    seen_split: dict[str, str] = {}
    group_counts: Counter = Counter()
    rng = np.random.default_rng(0)
    sampled = 0
    for split in SPLITS:
        files = sorted((out / split).glob("*.parquet")) if (out / split).exists() else []
        if sorted(f.name for f in files) != manifest["files"].get(split, []):
            problems.append(f"{split}: files differ from the manifest")
        rows = 0
        for f in files:
            pf = pq.ParquetFile(f)
            if not pf.schema_arrow.equals(_schema()):
                problems.append(f"{f}: schema {pf.schema_arrow}")
                continue
            rows += pf.metadata.num_rows
            cols = pf.read(columns=["peptide", "charge", "analyte_id"])
            for pep, charge, aid in zip(cols.column(0).to_pylist(), cols.column(1).to_pylist(),
                                        cols.column(2).to_pylist()):
                key = sequence_key(pep, settings["il_collapse"])
                if key in index:
                    problems.append(f"{f.name}: excluded sequence {pep}")
                if aid != peptide_key(pep, charge):
                    problems.append(f"{f.name}: analyte_id {aid} != {pep}_{charge}")
                group_counts[(split, aid)] += 1
                if settings["split_policy"] == "peptide":
                    il = sequence_key(pep, True)
                    other = seen_split.setdefault(il, split)
                    if other != split:
                        problems.append(f"sequence {il} in both {other} and {split}")
            if sampled < args.sample and pf.metadata.num_row_groups:
                t = pf.read_row_group(int(rng.integers(pf.metadata.num_row_groups)))
                for row in t.slice(0, min(args.sample - sampled, 50)).to_pylist():
                    sampled += 1
                    mz, li = np.asarray(row["mz"]), np.asarray(row["log_intensity"])
                    if not (0 < len(mz) == len(li) <= settings["max_peaks"]):
                        problems.append(f"{f.name}: bad peak count {len(mz)}/{len(li)}")
                    elif not (np.isfinite(mz).all() and abs(li.max() - 1) < 1e-5
                              and li.min() >= 0):
                        problems.append(f"{f.name}: log_intensity not normalised")
                    if abs(row["precursor"] - precursor_mz(row["peptide"], row["charge"])) > 1e-3:
                        problems.append(f"{f.name}: precursor mismatch for {row['peptide']}")
        if rows != manifest["kept_by_split"].get(split, 0):
            problems.append(f"{split}: {rows} rows, manifest says "
                            f"{manifest['kept_by_split'].get(split, 0)}")
    groups = pq.read_table(out / "group_sizes.parquet").to_pylist()
    recorded = Counter({(g["split"], g["analyte_id"]): g["count"] for g in groups})
    if recorded != group_counts:
        problems.append("group_sizes.parquet does not match the data")
    for p in problems[:50]:
        print("PROBLEM:", p)
    print(f"[check] {out}: {sum(manifest['kept_by_split'].values()):,} rows, "
          f"{len(group_counts):,} groups, {sampled} spectra sampled, "
          f"{len(problems)} problem(s)")
    return 1 if problems else 0


# ------------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("exclusion", help="build the eval-peptide exclusion set")
    ex.add_argument("--out", required=True, help="directory for exclusion.json")
    ex.add_argument("--scratch-root", default=os.environ.get("SCRATCH_ROOT", S_DEFAULT))
    ex.add_argument("--source", action="append", metavar="NAME=PATH",
                    help="extra source: a save_to_disk dir or hf:<repo> (repeatable)")
    ex.add_argument("--drop-source", action="append", metavar="NAME")
    ex.add_argument("--no-defaults", action="store_true", help="only the --source entries")

    pr = sub.add_parser("prepare", help="build the training data")
    pr.add_argument("--out", required=True)
    pr.add_argument("--exclusion", required=True, help="exclusion.json or its directory")
    pr.add_argument("--source-dir", help=f"snapshot dir (default: $HF_HOME/hub/... @{REVISION[:8]})")
    pr.add_argument("--splits", default="train,validation,test", help="source splits to read")
    pr.add_argument("--shards", type=int, default=0, help="first N shards per split (0 = all)")
    pr.add_argument("--limit", type=int, default=0, help="first N rows per shard (0 = all)")
    pr.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 64))
    pr.add_argument("--batch-size", type=int, default=20000)
    pr.add_argument("--max-peaks", type=int, default=512)
    pr.add_argument("--oversize", choices=("drop", "top"), default="drop",
                    help="drop (as training does) or keep the max_peaks most intense peaks")
    pr.add_argument("--il-collapse", action=argparse.BooleanOptionalAction, default=True,
                    help="exclude eval sequences after I->L (default on)")
    pr.add_argument("--split-policy", choices=("peptide", "source"), default="peptide",
                    help="peptide: re-split by a hash of the I/L-collapsed sequence "
                         "(peptide-disjoint, like ms-contrastive-100k); source: keep the "
                         "source's splits (spectrum-level, may share peptides)")
    pr.add_argument("--validation-fraction", type=float, default=0.033)
    pr.add_argument("--test-fraction", type=float, default=0.033)
    pr.add_argument("--resume", action="store_true", help="continue <out>.partial")

    ck = sub.add_parser("check", help="validate a finished output")
    ck.add_argument("out")
    ck.add_argument("--sample", type=int, default=2000, help="spectra to check values of")

    args = parser.parse_args(argv)
    return {"exclusion": cmd_exclusion, "prepare": cmd_prepare, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
