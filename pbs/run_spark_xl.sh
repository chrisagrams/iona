#!/bin/bash
# Launch a single msdelta tier on the DGX Spark (GB10, 1 GPU). This is the
# non-PBS analogue of pbs/_run_tier.sh: it overlays data.root + out_dir onto
# an in-repo config (which stays system-agnostic) and runs msdelta-train
# directly. No PBS, no ALCF proxy, no 4-GPU fan-out.
#
# Usage:
#   wandb login            # once, interactive
#   pbs/run_spark_xl.sh    # defaults to configs/v14_cap_XL.yaml
#   pbs/run_spark_xl.sh configs/v14_cap_L.yaml
#
# Set SMOKE=1 for a 100-step debug run.

set -euo pipefail

SRC_CFG=${1:-configs/v14_cap_XL.yaml}

REPO=/home/cgrams/msdelta
VENV=$REPO/.venv

case "$SRC_CFG" in
    /*) ;;
    *) SRC_CFG="$REPO/$SRC_CFG" ;;
esac
TAG=$(basename "$SRC_CFG" .yaml)

# DGX Spark local paths.
DATA_ROOT=/home/cgrams/datasets/consensus_100M/
OUT_DIR=$REPO/runs
mkdir -p "$OUT_DIR"

RUN_NAME=${TAG}_spark
LOGDIR=$REPO/pbs/logs/spark
mkdir -p "$LOGDIR"

# Overlay data.root + out_dir (and shrink schedule if SMOKE=1).
TMP_CFG=$(mktemp -t msdelta_${TAG}.XXXX.yaml)
SMOKE=${SMOKE:-0} "$VENV/bin/python" - "$SRC_CFG" "$DATA_ROOT" "$OUT_DIR" "$TMP_CFG" <<'PY'
import os, sys, yaml
src, root, out_dir, dst = sys.argv[1:]
with open(src) as fh:
    cfg = yaml.safe_load(fh)
cfg["data"]["root"] = root
cfg["log"]["out_dir"] = out_dir
if os.environ.get("SMOKE") == "1":
    cfg["train"]["total_steps"] = 100
    cfg["train"]["warmup_steps"] = 20
    cfg["train"]["val_batches"] = 5
    cfg["log"]["log_every"] = 10
    cfg["log"]["val_every"] = 50
    cfg["log"]["bias_curve_every"] = 100
    cfg["log"]["ckpt_every"] = 100
    cfg["log"]["probe_every"] = 10_000_000
    cfg["log"]["wandb_project"] = cfg["log"].get("wandb_project", "msdelta") + "-smoke"
with open(dst, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
PY

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Fragmentation fix that pairs with the inline probe block (train.py calls
# torch.cuda.empty_cache() before probes; this keeps segments expandable).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resume the same wandb run on re-launch against the same name.
export WANDB_DIR=$OUT_DIR/$RUN_NAME
export WANDB_RESUME=allow
export WANDB_RUN_ID=$RUN_NAME

echo "[$(hostname)] $(date -Is) tag=$TAG run=$RUN_NAME cfg=$TMP_CFG data=$DATA_ROOT"

cd "$REPO"
exec "$VENV/bin/msdelta-train" \
    --config "$TMP_CFG" \
    --run-name "$RUN_NAME"
