"""Stream an MGF into globally-shuffled parquet shards.

The MassIVE-KB MGF is ordered by source raw file (``SulfenM_RKO_LCA_A01 →
A02 → …``) with the same peptide repeated in consecutive entries. Feeding
that order to training — even with `ConsensusParquet`'s row-group-local
shuffle — leaves long runs of correlated spectra inside a row-group and
produces the loss spikes we saw before. The fix is a *global* shuffle.

We do it in a single streaming pass with bounded memory: each spectrum is
assigned to a uniformly-random shard. Every shard is therefore an i.i.d.
sample of the whole file, so a row-group read from any shard is already a
random draw. Combined with the dataset's existing per-row-group + within-
row-group permutation, the effective ordering is a near-perfect global
shuffle — no 130 GB sort, peak memory ≈ (n_shards × rows_per_group) rows.

Output schema matches the existing consensus parquet exactly:
  ``m/z``            list<float32>
  ``int``            list<float32>
  ``peptide_charge`` string   ("SEQ_z", mods in bracket form, see below)

Mod notation: the MGF carries inline mods like ``M+15.995`` / ``C+57.021``.
We rewrite them to the bracket form ``M[+15.995]`` that ``data.precursor_mz``
parses (``data._MOD_RE`` accepts an optional sign). Residue letters are left
untouched so ``_MOD_RE.sub("", pep)`` still recovers the bare sequence.

One-off data-prep script (not part of the installed package). Run with the
project venv's python, one invocation per input MGF; see pbs/convert_massivekb.sh:
  python scripts/convert_mgf.py --input .../massivekb_..._train.mgf \\
      --out-dir .../parquet_shuffled --prefix train --n-shards 256
  python scripts/convert_mgf.py --input .../massivekb_..._val.mgf \\
      --out-dir .../parquet_shuffled --prefix val --n-shards 2

`split_paths(root, n_val_files)` then takes the last `n_val_files` shards
(sorted) as validation; name val shards so they sort last (``val_*`` >
``train_*``) and set ``n_val_files`` to the val shard count.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Inline mod → bracket: "M+15.995" → "M[+15.995]", "Q-17.027" → "Q[-17.027]".
# Keeps the sign; data._MOD_RE is sign-tolerant. Residue letters are kept.
_INLINE_MOD = re.compile(r"([+-][0-9]+(?:\.[0-9]+)?)")

SCHEMA = pa.schema([
    ("m/z", pa.list_(pa.float32())),
    ("int", pa.list_(pa.float32())),
    ("peptide_charge", pa.string()),
])


def normalize_seq(seq: str) -> str:
    """Rewrite inline +/- mods to bracket form data.py understands."""
    return _INLINE_MOD.sub(r"[\1]", seq)


def parse_charge(charge_field: str | None) -> int:
    """'2+' / '2' / '3-' → int z (0 if unparseable)."""
    if not charge_field:
        return 0
    m = re.match(r"\s*(\d+)", charge_field)
    return int(m.group(1)) if m else 0


class ShardWriter:
    """Holds one open ParquetWriter per shard plus a row buffer; flushes a
    buffer as a single row-group when it reaches ``rows_per_group``."""

    def __init__(self, out_dir: Path, prefix: str, n_shards: int, rows_per_group: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.n_shards = n_shards
        self.rows_per_group = rows_per_group
        width = max(3, len(str(n_shards - 1)))
        self.paths = [out_dir / f"{prefix}_{i:0{width}d}.parquet" for i in range(n_shards)]
        self._writers: list[pq.ParquetWriter | None] = [None] * n_shards
        self._mz: list[list] = [[] for _ in range(n_shards)]
        self._int: list[list] = [[] for _ in range(n_shards)]
        self._pc: list[list] = [[] for _ in range(n_shards)]

    def add(self, shard: int, mz: list[float], inten: list[float], pc: str) -> None:
        self._mz[shard].append(mz)
        self._int[shard].append(inten)
        self._pc[shard].append(pc)
        if len(self._pc[shard]) >= self.rows_per_group:
            self._flush(shard)

    def _flush(self, shard: int) -> None:
        if not self._pc[shard]:
            return
        tbl = pa.table(
            {
                "m/z": pa.array(self._mz[shard], type=pa.list_(pa.float32())),
                "int": pa.array(self._int[shard], type=pa.list_(pa.float32())),
                "peptide_charge": pa.array(self._pc[shard], type=pa.string()),
            },
            schema=SCHEMA,
        )
        if self._writers[shard] is None:
            self._writers[shard] = pq.ParquetWriter(self.paths[shard], SCHEMA, compression="zstd")
        self._writers[shard].write_table(tbl)  # one row-group per flush
        self._mz[shard].clear()
        self._int[shard].clear()
        self._pc[shard].clear()

    def close(self) -> None:
        for shard in range(self.n_shards):
            self._flush(shard)
            if self._writers[shard] is not None:
                self._writers[shard].close()


def iter_mgf(path: Path):
    """Yield (peptide_charge, mz_list, int_list) per spectrum. Skips spectra
    with no SEQ, no charge, or no peaks (can't form a usable training row)."""
    seq = None
    charge = 0
    mz: list[float] = []
    inten: list[float] = []
    in_ions = False
    with open(path, "r", buffering=1 << 20) as fh:
        for line in fh:
            if line[0] == "B" and line.startswith("BEGIN IONS"):
                seq, charge, mz, inten, in_ions = None, 0, [], [], True
                continue
            if not in_ions:
                continue
            if line[0] == "E" and line.startswith("END IONS"):
                in_ions = False
                if seq and charge and mz:
                    yield f"{normalize_seq(seq)}_{charge}", mz, inten
                continue
            c0 = line[0]
            if c0.isdigit() or c0 == ".":
                # peak line: "mz intensity"
                sp = line.split()
                if len(sp) >= 2:
                    mz.append(float(sp[0]))
                    inten.append(float(sp[1]))
            elif line.startswith("SEQ="):
                seq = line[4:].strip()
            elif line.startswith("CHARGE="):
                charge = parse_charge(line[7:])
            # TITLE/PEPMASS/RTINSECONDS are not needed (precursor m/z is
            # recomputed from SEQ_z in data.precursor_mz).


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="source .mgf")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--prefix", required=True, help="shard name prefix, e.g. 'train' or 'val'")
    ap.add_argument("--n-shards", type=int, default=256,
                    help="more shards = finer global shuffle + lower peak memory")
    ap.add_argument("--rows-per-group", type=int, default=20000,
                    help="rows per parquet row-group (the dataset's read unit)")
    ap.add_argument("--seed", type=int, default=0, help="shard-assignment RNG seed")
    ap.add_argument("--report-every", type=int, default=1_000_000)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    writer = ShardWriter(args.out_dir, args.prefix, args.n_shards, args.rows_per_group)

    n = 0
    skipped = 0
    t0 = time.time()
    print(f"[convert] {args.input} → {args.out_dir} ({args.n_shards} shards, prefix={args.prefix!r})",
          flush=True)
    try:
        for pc, mz, inten in iter_mgf(args.input):
            shard = int(rng.integers(args.n_shards))  # uniform → global shuffle
            writer.add(shard, mz, inten, pc)
            n += 1
            if n % args.report_every == 0:
                rate = n / max(1e-6, time.time() - t0)
                print(f"[convert] {n:,} spectra  ({rate:,.0f}/s)", flush=True)
    finally:
        writer.close()

    dt = time.time() - t0
    print(f"[convert] done: {n:,} spectra written, {skipped:,} skipped, "
          f"{dt/60:.1f} min ({n/max(1e-6,dt):,.0f}/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
