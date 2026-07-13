"""Fast, parallel, single-pass full-corpus spectrum encoder.

Replaces the serial `embed_model` CPU-preprocessing loop
(msdelta/replicate_retrieval.py) with a `DataLoader`+`IterableDataset` so peak
filtering/top-N/log1p runs in parallel worker processes while the GPU stays
fed. Encodes every spectrum of the 4G held-out set ONCE per checkpoint and
caches the pooled embedding to disk (memmap — never materialised in RAM).

Memory: the 4G shards have one row-group per file (~250k rows / ~320MB raw
each); a worker holds at most one shard's row-group at a time, not the whole
corpus. The output embedding matrix (4.18M x 2048 float32 ~= 34GB, ~17GB at
fp16) is written via `np.memmap` so it's never resident all at once, and only
one checkpoint's model + matrix exist at a time (encode_one deletes the model
and empties the CUDA cache before returning).

Alignment: `CorpusParquet` is a DETERMINISTIC, single-pass (no shuffle, no
repeat) stream in stable file+row order, and every yielded spectrum carries
its `global_row` (a monotonic index over the full, sorted shard list) through
the collate function. The main process scatters each batch's pooled vectors
into `emb[global_row]`, so worker interleaving / prefetch reordering can never
misplace a row. `build_metadata` performs a second, independent pass reading
only the scalar columns (peptide/charge/scores/precursor) and writes them
under the same `global_row` indexing, so row i of the embedding always lines
up with row i of the metadata table.

Usage:
    # correctness check (old serial embed_model vs this path)
    .venv/bin/python scripts/encode_corpus.py --verify --limit 2000 \\
        --ckpt runs/consensus_xl_28M_7235042/final.pt

    # timed small run
    .venv/bin/python scripts/encode_corpus.py --limit 100000 \\
        --ckpt runs/consensus_xl_28M_7235042/final.pt --out-dir runs/emb_test

    # full corpus (one checkpoint at a time; repeat --ckpt for more)
    .venv/bin/python scripts/encode_corpus.py \\
        --ckpt runs/consensus_xl_28M_7235042/final.pt --ckpt runs/other/final.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from msdelta.analyze import load_encoder
from msdelta.data import N_CHARGES, PreprocessConfig, preprocess_spectrum
from msdelta.probe import _pool

MZ_COL, INT_COL, CHARGE_COL, PREC_COL = "mz", "intensity", "charge", "precursor"


# ---------- deterministic single-pass dataset ----------

class CorpusParquet(IterableDataset):
    """Deterministic, single-pass stream over 4G-schema parquet shards
    (columns: peptide, charge, mz, intensity, precursor, ...), in stable
    file-then-row order. Yields (mz_p, log_int_p, charge_idx, precursor_mz,
    global_row) for every spectrum that preprocesses to >=1 peak; spectra
    that preprocess to zero peaks are skipped (their embedding row stays the
    zero-initialised memmap default, matching `embed_model`'s behaviour).

    NOT for training: no shuffle, one epoch, order is required for the
    global_row alignment to mean anything.
    """

    # Target number of work units (independent of num_workers/shard count).
    # The 4G shards are one big row-group each (~250k rows), so without
    # subdividing, a small --limit run would only touch one shard/unit and
    # only one worker would ever have anything to do. Subdividing each
    # row-group into ~TARGET_UNITS-sized chunks fixes that (and, as a bonus,
    # bounds per-worker memory to a fraction of a row-group instead of a
    # whole one for the full-corpus run too).
    TARGET_UNITS = 64

    def __init__(
        self,
        paths: list[str | Path],
        preprocess: PreprocessConfig,
        limit: int | None = None,
    ):
        super().__init__()
        self.preprocess = preprocess
        sorted_paths = sorted(Path(p) for p in paths)

        # Pass 1: macro (path, row_group, base_offset, n_rows) units from
        # parquet metadata only, stopping once `limit` rows are covered so a
        # small --limit run doesn't even look at shards it won't read.
        macro: list[tuple[Path, int, int, int]] = []
        cum = 0
        for p in sorted_paths:
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                if limit is not None and cum >= limit:
                    break
                n_full = pf.metadata.row_group(rg).num_rows
                n = n_full if limit is None else min(n_full, limit - cum)
                macro.append((p, rg, cum, n))
                cum += n
            if limit is not None and cum >= limit:
                break
        self.total_rows = cum

        # Pass 2: split each macro unit into row-range sub-chunks of
        # `chunk_rows` rows. Units: (path, row_group, rg_local_start,
        # length, rg_base_offset) -- global_row = rg_base_offset + local i.
        chunk_rows = max(1, -(-self.total_rows // self.TARGET_UNITS))  # ceil div
        self.units: list[tuple[Path, int, int, int, int]] = []
        for p, rg, base, n in macro:
            start = 0
            while start < n:
                ln = min(chunk_rows, n - start)
                self.units.append((p, rg, start, ln, base))
                start += ln

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, int, float, int]]:
        info = get_worker_info()
        worker_id, num_workers = (0, 1) if info is None else (info.id, info.num_workers)
        my_units = self.units[worker_id::num_workers]

        # Tiny 1-slot cache: adjacent units in this worker's slice sometimes
        # share the same (path, row_group) after subdivision.
        cache_key = None
        cache_cols = None
        cols = [MZ_COL, INT_COL, CHARGE_COL, PREC_COL]

        for path, rg, start, ln, base in my_units:
            key = (path, rg)
            if key != cache_key:
                tbl = pq.ParquetFile(path).read_row_group(rg, columns=cols)
                cache_cols = (tbl.column(MZ_COL), tbl.column(INT_COL),
                              tbl.column(CHARGE_COL), tbl.column(PREC_COL))
                cache_key = key
            mz_col, int_col, chg_col, prec_col = cache_cols
            for i in range(start, start + ln):
                mz_list = mz_col[i].as_py()
                if not mz_list:
                    continue
                int_list = int_col[i].as_py()
                mz_t = torch.tensor(mz_list, dtype=torch.float32)
                int_t = torch.tensor(int_list, dtype=torch.float32)
                mz_p, li_p, _ = preprocess_spectrum(mz_t, int_t, self.preprocess)
                if mz_p.numel() == 0:
                    continue
                chg_raw = chg_col[i].as_py()
                chg_idx = min(max(int(chg_raw) if chg_raw is not None else 0, 0), N_CHARGES - 1)
                prec_raw = prec_col[i].as_py()
                prec = float(prec_raw) if (prec_raw is not None and prec_raw == prec_raw) else 0.0
                yield mz_p, li_p, chg_idx, prec, base + i


def _encode_collate(batch):
    """Pad a list of (mz_p, log_int_p, charge, precursor_mz, global_row)."""
    B = len(batch)
    Ks = [int(item[0].numel()) for item in batch]
    K_max = max(max(Ks), 1)
    mz = torch.zeros(B, K_max, dtype=torch.float32)
    log_int = torch.zeros(B, K_max, dtype=torch.float32)
    key_padding_mask = torch.ones(B, K_max, dtype=torch.bool)  # True = padding
    charge = torch.zeros(B, dtype=torch.long)
    prec_mz = torch.zeros(B, dtype=torch.float32)
    global_row = torch.zeros(B, dtype=torch.long)
    for b, (mz_p, li_p, chg, pmz, grow) in enumerate(batch):
        K = int(mz_p.numel())
        charge[b] = chg
        prec_mz[b] = pmz
        global_row[b] = grow
        if K:
            mz[b, :K] = mz_p
            log_int[b, :K] = li_p
            key_padding_mask[b, :K] = False
    return mz, log_int, key_padding_mask, charge, prec_mz, global_row


def make_loader(paths, pp, limit, batch_size, num_workers, prefetch_factor=4):
    ds = CorpusParquet(paths, pp, limit=limit)
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = False
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        collate_fn=_encode_collate, pin_memory=True, **kwargs,
    ), ds.total_rows


# ---------- metadata (independent of checkpoint; scalar columns only) ----------

def build_metadata(paths: list[Path], limit: int | None, out_path: Path) -> int:
    """Second, independent single-pass over the same shard order reading only
    scalar columns (peptide/charge/scores/precursor) -- no mz/intensity, so
    it's cheap. Writes a parquet with `global_row` matching the embedding
    matrix's row index exactly, plus a factorized peptide/charge label."""
    import pandas as pd

    sorted_paths = sorted(paths)
    peptide, charge, max_score, mean_score, precursor = [], [], [], [], []
    cum = 0
    for p in sorted_paths:
        if limit is not None and cum >= limit:
            break
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            if limit is not None and cum >= limit:
                break
            n = pf.metadata.row_group(rg).num_rows
            take = n if limit is None else min(n, limit - cum)
            tbl = pf.read_row_group(rg, columns=["peptide", "charge", "max_score",
                                                  "mean_score", "precursor"])
            peptide.extend(tbl.column("peptide").to_pylist()[:take])
            charge.extend(tbl.column("charge").to_pylist()[:take])
            max_score.extend(tbl.column("max_score").to_pylist()[:take])
            mean_score.extend(tbl.column("mean_score").to_pylist()[:take])
            precursor.extend(tbl.column("precursor").to_pylist()[:take])
            cum += take

    df = pd.DataFrame({
        "global_row": np.arange(cum, dtype=np.int64),
        "peptide": peptide,
        "charge": charge,
        "max_score": max_score,
        "mean_score": mean_score,
        "precursor": precursor,
    })
    key = df["peptide"].astype(str) + "_" + df["charge"].astype(str)
    df["peptide_charge_id"] = pd.factorize(key, sort=False)[0].astype(np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return cum


# ---------- encode one checkpoint ----------

@torch.no_grad()
def encode_one(
    ckpt: Path,
    paths: list[Path],
    out_dir: Path,
    device: torch.device,
    limit: int | None,
    batch_size: int,
    num_workers: int,
    fp16: bool,
) -> dict:
    enc, cfg, step = load_encoder(ckpt)
    enc.to(device).eval()
    dcfg = cfg["data"]
    pp = PreprocessConfig(intensity_threshold_frac=dcfg["intensity_threshold_frac"],
                          top_n=dcfg["top_n"])

    loader, total_rows = make_loader(paths, pp, limit, batch_size, num_workers)
    d_model = cfg["model"]["d_model"]
    dim = 2 * d_model  # mean ⊕ max pool

    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if fp16 else np.float32
    emb_path = out_dir / f"{ckpt.stem}.{'fp16' if fp16 else 'fp32'}.npy"
    emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=dtype, shape=(total_rows, dim))

    n_done = 0
    t0 = time.monotonic()
    last_print = t0
    for mz, log_int, kpm, charge, prec, grow in loader:
        mz = mz.to(device, non_blocking=True)
        log_int = log_int.to(device, non_blocking=True)
        kpm = kpm.to(device, non_blocking=True)
        charge = charge.to(device, non_blocking=True)
        prec = prec.to(device, non_blocking=True)

        tok = enc(mz, log_int, kpm, charge=charge, precursor_mz=prec)
        pooled = _pool(tok, ~kpm).to(torch.float32).cpu().numpy()

        idx = grow.numpy()
        emb[idx] = pooled.astype(dtype, copy=False)
        n_done += len(idx)

        now = time.monotonic()
        if now - last_print > 5.0:
            rate = n_done / (now - t0)
            remaining = max(total_rows - n_done, 0)
            eta_s = remaining / rate if rate > 0 else float("inf")
            print(f"  [{ckpt.name}] {n_done}/{total_rows} "
                  f"({rate:.1f} spec/s, ETA {eta_s/60:.1f} min)", flush=True)
            last_print = now

    emb.flush()
    del emb  # drop the memmap handle
    dt = time.monotonic() - t0
    rate = n_done / dt if dt > 0 else 0.0
    print(f"[{ckpt.name}] done: {n_done} spectra in {dt:.1f}s ({rate:.1f} spec/s) -> {emb_path}")

    del enc
    torch.cuda.empty_cache()
    return {"ckpt": str(ckpt), "step": step, "n": n_done, "seconds": dt,
            "rate": rate, "path": str(emb_path)}


# ---------- correctness verify: old embed_model vs this path ----------

def verify(ckpt: Path, paths: list[Path], device: torch.device, n: int, num_workers: int):
    """Encode the first `n` spectra (in file+row order) both ways and assert
    the pooled embeddings agree within float tolerance."""
    from datasets import Dataset

    from msdelta.replicate_retrieval import embed_model

    enc, cfg, step = load_encoder(ckpt)
    dcfg = cfg["data"]
    pp = PreprocessConfig(intensity_threshold_frac=dcfg["intensity_threshold_frac"],
                          top_n=dcfg["top_n"])

    # ---- old path: read the first n rows into an Arrow-backed Dataset ----
    rows_needed = n
    tables = []
    for p in sorted(paths):
        if rows_needed <= 0:
            break
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            if rows_needed <= 0:
                break
            tbl = pf.read_row_group(rg, columns=[MZ_COL, INT_COL, CHARGE_COL, PREC_COL])
            take = min(len(tbl), rows_needed)
            tables.append(tbl.slice(0, take))
            rows_needed -= take
    import pyarrow as pa
    full_tbl = pa.concat_tables(tables)
    ds = Dataset(full_tbl)
    print(f"verify: old embed_model over {len(ds)} spectra ...")
    old_emb = embed_model(enc, ds, device, pp, batch_size=128)

    # ---- new path: same first n rows via CorpusParquet + DataLoader ----
    print(f"verify: new DataLoader path over {n} spectra ({num_workers} workers) ...")
    loader, total_rows = make_loader(paths, pp, n, batch_size=128, num_workers=num_workers)
    new_emb = np.zeros_like(old_emb)
    enc.to(device).eval()
    with torch.no_grad():
        for mz, log_int, kpm, charge, prec, grow in loader:
            mz, log_int = mz.to(device), log_int.to(device)
            kpm, charge, prec = kpm.to(device), charge.to(device), prec.to(device)
            tok = enc(mz, log_int, kpm, charge=charge, precursor_mz=prec)
            pooled = _pool(tok, ~kpm).cpu().numpy()
            new_emb[grow.numpy()] = pooled

    diff = np.abs(old_emb - new_emb)
    max_diff = float(diff.max())
    print(f"\nVERIFY max|old - new| over {len(ds)} spectra = {max_diff:.3e}")
    ok = max_diff < 1e-4
    print("VERIFY " + ("PASSED" if ok else "FAILED") + f" (threshold 1e-4)")
    return ok, max_diff


# ---------- CLI ----------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", action="append", required=True, type=Path,
                   help="checkpoint path; repeatable. Encoded one at a time.")
    p.add_argument("--data-dir", type=Path, default=Path("/home/cgrams/datasets/4G_dataset"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/emb_full"))
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=256,
                   help="the Δm-bias einsum is O(B*K^2*H*D_h) -- 1024 OOMs at top_n=150, "
                        "16 heads (needs ~44GB/batch-item-1024); 256 is a safe default. "
                        "Throughput is CPU-preprocessing-bound, not batch-size-bound, so "
                        "there's little upside to pushing this higher.")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="cap total spectra (testing)")
    p.add_argument("--verify", action="store_true",
                   help="correctness check (old embed_model vs new path) instead of encoding")
    p.add_argument("--verify-n", type=int, default=2000)
    p.add_argument("--skip-metadata", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    if args.num_workers > 16:
        print(f"capping --num-workers {args.num_workers} -> 16 (memory-safety cap)")
        args.num_workers = 16

    paths = sorted(args.data_dir.glob("*.parquet"))
    if not paths:
        print(f"no parquet shards found under {args.data_dir}", file=sys.stderr)
        return 1
    device = torch.device(args.device)

    if args.verify:
        ckpt = args.ckpt[0]
        ok, max_diff = verify(ckpt, paths, device, args.verify_n, args.num_workers)
        return 0 if ok else 1

    if not args.skip_metadata:
        meta_path = args.out_dir / "metadata.parquet"
        print(f"building metadata ({'all' if args.limit is None else args.limit} rows) -> {meta_path}")
        t0 = time.monotonic()
        n_meta = build_metadata(paths, args.limit, meta_path)
        print(f"  wrote {n_meta} rows in {time.monotonic()-t0:.1f}s")

    results = []
    for ckpt in args.ckpt:
        print(f"\n=== encoding {ckpt} ===")
        res = encode_one(ckpt, paths, args.out_dir, device, args.limit,
                         args.batch_size, args.num_workers, args.fp16)
        results.append(res)

    print("\n=== summary ===")
    for r in results:
        print(f"  {r['ckpt']}: {r['n']} spectra, {r['rate']:.1f} spec/s, {r['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
