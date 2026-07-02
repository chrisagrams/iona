"""Globally shuffle the consensus parquet corpus into a peptide-disjoint
train/val split, in one bounded-memory streaming pass.

Why: the consensus shards (consensus_00..89.parquet) store each peptide's
~330 replicate spectra in consecutive rows, and a peptide's replicates are
spread across multiple shards (measured: 41% of peptides span >1 shard, so a
whole-shard val holdout leaks ~59% of val spectra into train). Feeding that
order to training — even with ConsensusParquet's read-time row-group-local
shuffle — leaves long runs of correlated spectra, and any random split leaks
replicates across train/val. We fix both here:

  * global shuffle — each output shard is assigned uniformly at random per
    spectrum, so every shard is an i.i.d. sample of the whole corpus. Combined
    with ConsensusParquet's per-row-group + within-row-group permutation, the
    effective training order is a near-perfect global shuffle. No 200 GB sort;
    peak memory ≈ (n_shards × rows_per_group) buffered rows.

  * peptide-disjoint val — the train/val decision is a stable hash of the bare
    peptide (sequence+mods, charge stripped), so ALL replicates of a peptide
    land in the same split. val therefore shares no peptide with train.

The pass is embarrassingly parallel: both decisions above need zero cross-worker
coordination (the split is a pure hash; the shard is a per-spectrum random draw),
and the only shared resource — the output parquet files — is partitioned by giving
each worker its OWN set of output shards (name tagged `_wNN_`). We fan out with a
ProcessPoolExecutor over the input shards; a 32-core node cuts the ~18 h serial
run to ~35 min. This preserves the global-shuffle guarantee because training
(msdelta.data.ConsensusParquet) pools the (file, row-group) units of EVERY output
file and reads them in globally-shuffled order — so per-worker output files are
just extra units in that same shuffled pool, not a coarser split.

Output schema matches the consensus parquet exactly (m/z list<float32>, int
list<float32>, peptide_charge string). Files are written flat into one dir:
  <out>/<prefix>_train_NNN.parquet   (n_train_shards)
  <out>/<prefix>_val_NN.parquet      (n_val_shards)
"val" sorts after "train", so local training reads them with
`data.root=<out>` + `data.n_val_files=<n_val_shards>` (see msdelta.data.split_paths).
For the HF archive, scripts/upload_hf_dataset.py lifts train_*/val_* into
train/ and val/ sub-prefixes so the repo also works via `data.hf_repo`.

One-off data-prep script (not part of the installed package). See
pbs/shuffle_consensus.sh:
  python scripts/shuffle_parquet.py \\
      --input-dir /eagle/.../consensus_100M \\
      --out-dir   /eagle/.../consensus_100M_shuffled
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("m/z", pa.list_(pa.float32())),
    ("int", pa.list_(pa.float32())),
    ("peptide_charge", pa.string()),
])


def bare_peptide(peptide_charge: str) -> str:
    """'PEPTIDE[+15.995]_2' → 'PEPTIDE[+15.995]' (charge stripped). This is the
    split key: hashing it keeps every charge/replicate of a peptide together."""
    return peptide_charge.rsplit("_", 1)[0]


def in_val(pep: str, val_permille: int) -> bool:
    """Stable (non-salted) hash → deterministic peptide-disjoint val membership.
    val_permille is out of 1000, so 10 ≈ 1% of peptides held out."""
    h = hashlib.blake2b(pep.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(h, "big") % 1000) < val_permille


class ShardWriter:
    """One open ParquetWriter per shard + a row buffer; flushes a buffer as a
    single row-group when it reaches rows_per_group. (Same design as
    scripts/convert_mgf.py's writer.)"""

    def __init__(self, paths: list[Path], rows_per_group: int):
        self.paths = paths
        self.rows_per_group = rows_per_group
        n = len(paths)
        self._writers: list[pq.ParquetWriter | None] = [None] * n
        self._mz: list[list] = [[] for _ in range(n)]
        self._int: list[list] = [[] for _ in range(n)]
        self._pc: list[list] = [[] for _ in range(n)]
        self.rows = 0

    def add(self, shard: int, mz, inten, pc: str) -> None:
        self._mz[shard].append(mz)
        self._int[shard].append(inten)
        self._pc[shard].append(pc)
        self.rows += 1
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
        for shard in range(len(self.paths)):
            self._flush(shard)
            if self._writers[shard] is not None:
                self._writers[shard].close()


def shard_paths(out_dir: Path, prefix: str, split: str, n: int,
                worker: int | None = None) -> list[Path]:
    width = max(3, len(str(n - 1)))
    tag = "" if worker is None else f"w{worker:02d}_"
    return [out_dir / f"{prefix}_{split}_{tag}{i:0{width}d}.parquet" for i in range(n)]


def _process_shards(task: dict) -> tuple[int, int, int]:
    """One worker: stream its slice of input shards through the same global-shuffle
    + peptide-disjoint-split logic as the serial path, writing to its OWN worker-
    tagged output shards. Runs in a child process (see main's ProcessPoolExecutor).

    Independence across workers is by construction: `in_val` is a pure hash (same
    verdict everywhere) and each worker draws shard assignment / subsampling from
    its own RNG stream (seed + worker), so per-worker keep-probabilities compose to
    the same expected global targets. Arrow is pinned to 1 thread so N workers don't
    oversubscribe the node."""
    pa.set_cpu_count(1)
    wid = task["worker"]
    inputs = task["inputs"]
    p_train, p_val = task["p_train"], task["p_val"]
    val_permille = task["val_permille"]
    rng = np.random.default_rng(task["seed"] + wid)

    train_w = ShardWriter(shard_paths(task["out_dir"], task["prefix"], "train",
                                      task["n_train"], worker=wid), task["rows_per_group"])
    val_w = ShardWriter(shard_paths(task["out_dir"], task["prefix"], "val",
                                    task["n_val"], worker=wid), task["rows_per_group"])

    n = 0
    t0 = time.time()
    report_every = task["report_every"]
    # Replicates are consecutive within a shard, so cache the split decision for
    # the current peptide instead of re-hashing it ~330 times (per-worker cache).
    cur_pep = None
    cur_is_val = False
    try:
        for p in inputs:
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["m/z", "int", "peptide_charge"])
                mz_l = tbl.column("m/z").to_pylist()
                int_l = tbl.column("int").to_pylist()
                pc_l = tbl.column("peptide_charge").to_pylist()
                for mz, inten, pc in zip(mz_l, int_l, pc_l):
                    if not pc or not mz:
                        continue
                    pep = bare_peptide(pc)
                    if pep != cur_pep:
                        cur_pep = pep
                        cur_is_val = in_val(pep, val_permille)
                    if cur_is_val:
                        if p_val < 1.0 and rng.random() >= p_val:
                            continue
                        val_w.add(int(rng.integers(task["n_val"])), mz, inten, pc)
                    else:
                        if p_train < 1.0 and rng.random() >= p_train:
                            continue
                        train_w.add(int(rng.integers(task["n_train"])), mz, inten, pc)
                    n += 1
                    if n % report_every == 0:
                        rate = n / max(1e-6, time.time() - t0)
                        print(f"[shuffle][w{wid:02d}] {n:,} spectra  (train {train_w.rows:,} / "
                              f"val {val_w.rows:,})  {rate:,.0f}/s", flush=True)
                del tbl, mz_l, int_l, pc_l
    finally:
        train_w.close()
        val_w.close()
    return n, train_w.rows, val_w.rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path,
                    help="dir of source consensus_*.parquet shards")
    ap.add_argument("--input-glob", default="consensus_*.parquet",
                    help="glob for source shards within --input-dir")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--prefix", default="consensus_shuf", help="output shard name prefix")
    ap.add_argument("--n-train-shards", type=int, default=256,
                    help="more shards = finer global shuffle + lower peak memory")
    ap.add_argument("--n-val-shards", type=int, default=4)
    ap.add_argument("--val-permille", type=int, default=10,
                    help="per-1000 of PEPTIDES held out to val (10 = ~1%%); "
                         "val is peptide-disjoint from train")
    ap.add_argument("--rows-per-group", type=int, default=20000,
                    help="rows per parquet row-group (ConsensusParquet's read unit)")
    ap.add_argument("--target-train-spectra", type=int, default=0,
                    help="subsample train to ~this many spectra (0 = keep all). "
                         "Uniform per-spectrum, so peptide/replicate proportions are "
                         "preserved. Use to match MassIVE-KB's ~28.5M for an "
                         "apples-to-apples same-epoch comparison.")
    ap.add_argument("--target-val-spectra", type=int, default=0,
                    help="subsample val to ~this many spectra (0 = keep all)")
    ap.add_argument("--seed", type=int, default=0, help="shard-assignment RNG seed")
    ap.add_argument("--report-every", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes fanned out over input shards "
                         "(0 = os.cpu_count(), capped at #input shards). Each worker "
                         "writes its own `_wNN_` output shards; totals are ~n-train-"
                         "/n-val-shards spread across workers.")
    args = ap.parse_args(argv)

    inputs = sorted(args.input_dir.glob(args.input_glob))
    if not inputs:
        raise SystemExit(f"no shards match {args.input_dir}/{args.input_glob}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Total input spectra from parquet metadata (instant) → keep-probabilities
    # that hit the requested target counts. val fraction of spectra ≈ val_permille
    # /1000 (replicate counts vary per peptide, but the target is approximate).
    total_in = sum(pq.ParquetFile(p).metadata.num_rows for p in inputs)
    exp_val = total_in * args.val_permille / 1000.0
    exp_train = total_in - exp_val
    p_train = 1.0 if args.target_train_spectra <= 0 else min(1.0, args.target_train_spectra / max(1.0, exp_train))
    p_val = 1.0 if args.target_val_spectra <= 0 else min(1.0, args.target_val_spectra / max(1.0, exp_val))
    print(f"[shuffle] input spectra: {total_in:,}  (est train {exp_train:,.0f} / val {exp_val:,.0f})", flush=True)
    if p_train < 1.0:
        print(f"[shuffle] subsampling train to ~{args.target_train_spectra:,} (keep p={p_train:.4f})", flush=True)
    if p_val < 1.0:
        print(f"[shuffle] subsampling val to ~{args.target_val_spectra:,} (keep p={p_val:.4f})", flush=True)

    # Fan out over input shards. Round-robin assignment (inputs[w::n_workers])
    # balances load since shards are ~equal size. Each worker owns a slice of the
    # n_train/n_val shards so the totals stay ≈ the requested counts, just split
    # into per-worker files; every output file is an independent read-unit for
    # training, so the global shuffle is unchanged (see module docstring).
    n_workers = args.workers or (os.cpu_count() or 8)
    n_workers = max(1, min(n_workers, len(inputs)))
    per_train = max(1, round(args.n_train_shards / n_workers))
    per_val = max(1, round(args.n_val_shards / n_workers))
    tot_train_shards = per_train * n_workers
    tot_val_shards = per_val * n_workers

    tasks = [
        {
            "worker": w,
            "inputs": inputs[w::n_workers],
            "out_dir": args.out_dir,
            "prefix": args.prefix,
            "n_train": per_train,
            "n_val": per_val,
            "val_permille": args.val_permille,
            "rows_per_group": args.rows_per_group,
            "p_train": p_train,
            "p_val": p_val,
            "seed": args.seed,
            "report_every": args.report_every,
        }
        for w in range(n_workers)
        if inputs[w::n_workers]
    ]

    t0 = time.time()
    print(f"[shuffle] {len(inputs)} input shards → {args.out_dir}  "
          f"({n_workers} workers, train={tot_train_shards} [{per_train}/worker], "
          f"val={tot_val_shards} [{per_val}/worker], "
          f"val_permille={args.val_permille}, seed={args.seed})", flush=True)

    n = train_rows = val_rows = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_process_shards, t) for t in tasks]
        for fut in as_completed(futures):
            wn, wtr, wvr = fut.result()
            n += wn
            train_rows += wtr
            val_rows += wvr

    dt = time.time() - t0
    print(f"[shuffle] done: {n:,} spectra "
          f"(train {train_rows:,} / val {val_rows:,}), "
          f"{dt/60:.1f} min ({n/max(1e-6,dt):,.0f}/s)", flush=True)
    print(f"[shuffle] local training: data.root={args.out_dir}  "
          f"data.n_val_files={tot_val_shards}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
