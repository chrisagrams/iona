#!/bin/bash
#PBS -N shuf_consensus
#PBS -A FRAME-IDP
#PBS -q capacity
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o /home/cgrams/msdelta/pbs/logs/
#
# One-time data prep: globally shuffle the consensus parquet corpus into a
# peptide-disjoint train/val split (see scripts/shuffle_parquet.py). Runs on a
# compute node — 110M spectra, ~200 GB read + ~200 GB write, and ~40 GB of
# buffered rows peak (256 train shards x 20k rows/group), too heavy for the
# login node. Fanned out over the node's cores (WORKERS, default all 32);
# ~35 min expected (the serial version walltimed at ~18 h projected).
#
# Two datasets, selected by env (override with `qsub -v`):
#   FULL (default):    the whole ~109M-spectra shuffled train split.
#       qsub pbs/shuffle_consensus.sh
#   MASSIVEKB-MATCHED: train subsampled to ~28.5M so it runs the identical
#       schedule as massivekb_xl (90k steps ≈ 1 epoch) for an apples-to-apples
#       comparison (see configs/consensus_xl_28M.yaml). Name the OUT dir by its
#       spectra count so the two datasets are self-documenting.
#       qsub -v TARGET_TRAIN=28508636,OUT=/eagle/UIC-HPC/cgrams/msbench-datasets/consensus_match_shuffled_28M pbs/shuffle_consensus.sh
#
# Output: $OUT/{consensus_shuf_train_wNN_000.parquet ... , consensus_shuf_val_wNN_00.parquet ...}
#   -> local training reads it with data.root=$OUT + data.n_val_files=<N>, where
#      <N> is the "data.n_val_files=" value the script prints at the end (= WORKERS
#      * per-worker val shards; val_* shards still sort after train_*).

set -euo pipefail

REPO=/home/cgrams/msdelta
VENV=$REPO/.venv
SRC=${SRC:-/eagle/UIC-HPC/cgrams/msbench-datasets/consensus_100M}
OUT=${OUT:-/eagle/UIC-HPC/cgrams/msbench-datasets/consensus_100M_shuffled}

N_TRAIN=${N_TRAIN:-256}
N_VAL=${N_VAL:-4}
VAL_PERMILLE=${VAL_PERMILLE:-10}   # ~1% of PEPTIDES held out, peptide-disjoint from train
TARGET_TRAIN=${TARGET_TRAIN:-0}    # 0 = keep all; e.g. 28508636 to match MassIVE-KB
WORKERS=${WORKERS:-0}              # 0 = all node cores (capped at #input shards)

mkdir -p "$OUT"
cd "$REPO"
echo "$(date -Is) shuffling $SRC → $OUT (train=$N_TRAIN val=$N_VAL val_permille=$VAL_PERMILLE target_train=$TARGET_TRAIN workers=$WORKERS)"

# One Arrow thread per worker process (the script also calls pa.set_cpu_count(1));
# parallelism comes from the ProcessPool, so keep every thread pool at 1 to avoid
# oversubscribing the node's cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$VENV/bin/python" "$REPO/scripts/shuffle_parquet.py" \
    --input-dir "$SRC" \
    --out-dir "$OUT" \
    --n-train-shards "$N_TRAIN" \
    --n-val-shards "$N_VAL" \
    --val-permille "$VAL_PERMILLE" \
    --target-train-spectra "$TARGET_TRAIN" \
    --workers "$WORKERS" \
    --seed 0

echo "$(date -Is) done. shards:"
ls -1 "$OUT" | head
echo "  ... $(ls -1 "$OUT"/*.parquet | wc -l) parquet files total"
