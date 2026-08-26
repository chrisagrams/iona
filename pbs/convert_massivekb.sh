#!/bin/bash
#PBS -N mkb_convert
#PBS -A FRAME-IDP
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o /home/cgrams/msdelta/pbs/logs/
#
# One-time data prep: stream the MassIVE-KB MGFs into globally-shuffled
# parquet shards (the format ConsensusParquet reads). The shuffle is the
# whole point — the MGF is ordered by source raw file, which is what caused
# the earlier training-loss spikes. See scripts/convert_mgf.py.
#
# Output: $OUT/{train_000.parquet ... , val_00.parquet, val_01.parquet}
#   - train: 256 shards (~28.5M spectra, finer shuffle / lower peak memory)
#   - val:   2 shards   → set data.n_val_files=2 (they sort after train_*)
#
# Pure-Python single stream over 130 GB ≈ 30–60 min; run off the login node.

set -euo pipefail

REPO=/home/cgrams/msdelta
VENV=$REPO/.venv
SRC=/eagle/UIC-HPC/cgrams/msbench-datasets/massive_kb_v1_shared
OUT=$SRC/parquet_shuffled
mkdir -p "$OUT"

cd "$REPO"
echo "$(date -Is) converting MassIVE-KB → $OUT"

# Validation first (cheap, fails fast if anything's wrong).
"$VENV/bin/python" "$REPO/scripts/convert_mgf.py" \
    --input "$SRC/massivekb_82c0124b_val.mgf" \
    --out-dir "$OUT" --prefix val --n-shards 2 --seed 1

# Train: the big one.
"$VENV/bin/python" "$REPO/scripts/convert_mgf.py" \
    --input "$SRC/massivekb_82c0124b_train.mgf" \
    --out-dir "$OUT" --prefix train --n-shards 256 --seed 0

echo "$(date -Is) done. shards:"
ls -1 "$OUT" | head
echo "  ... $(ls -1 "$OUT"/*.parquet | wc -l) parquet files total"
