#!/bin/bash
# Resume ONE tier on ONE GPU inside an already-running PBS job — rescue a
# crashed sub-run without losing the allocation. Mirrors _run_tier.sh but:
#   - resumes from <run_dir>/last.pt (--resume)
#   - sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (the OOM fix:
#     the probe block OOM'd on a fragmented 40GB A100, "20 GiB reserved but
#     unallocated"). Combined with the empty_cache() added to the probe
#     block in train.py, this keeps the probe forwards from OOMing.
#
# Run it ON THE COMPUTE NODE (ssh to the node the job holds), pinned to the
# freed GPU. Use tmux/nohup so it survives an ssh drop.
#
# Usage:  bash pbs/resume_run.sh <config> <gpu> <run_dir>
#   e.g.  bash pbs/resume_run.sh configs/massivekb_xl.yaml 3 \
#               /eagle/UIC-HPC/cgrams/msdelta-runs/v14_cap_XL_7174804

set -euo pipefail

SRC_CFG=${1:?config path required}
GPU=${2:?gpu id required}
RUN_DIR=${3:?run dir required (the existing .../v14_cap_XL_<jobid>)}

REPO=/home/cgrams/msdelta
VENV=$REPO/.venv
case "$SRC_CFG" in /*) ;; *) SRC_CFG="$REPO/$SRC_CFG" ;; esac
TAG=$(basename "$SRC_CFG" .yaml)
RUN_NAME=$(basename "$RUN_DIR")
RESUME="$RUN_DIR/last.pt"
[[ -f "$RESUME" ]] || { echo "no checkpoint at $RESUME" >&2; exit 1; }

DATA_ROOT=/eagle/UIC-HPC/cgrams/msbench-datasets/consensus_100M/
OUT_DIR=$(dirname "$RUN_DIR")
LOGDIR=$REPO/pbs/logs/resume_${RUN_NAME}
mkdir -p "$LOGDIR"

# Same data.root + out_dir splice as _run_tier.sh.
TMP_CFG=$(mktemp -t msdelta_${TAG}.XXXX.yaml)
"$VENV/bin/python" - "$SRC_CFG" "$DATA_ROOT" "$OUT_DIR" "$TMP_CFG" <<'PY'
import sys, yaml
src, root, out_dir, dst = sys.argv[1:]
cfg = yaml.safe_load(open(src))
cfg["data"]["root"] = root
cfg["log"]["out_dir"] = out_dir
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY

export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,.alcf.anl.gov,.anl.gov
export NO_PROXY=$no_proxy

# THE OOM FIX — avoid the caching-allocator fragmentation that crashed the
# probe forward on the d=1024 tier.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Rejoin the same wandb run (resume=allow); re-logged steps ≤ last logged
# step are dropped by wandb, new steps continue the trajectory.
export WANDB_DIR=$LOGDIR
export WANDB_RESUME=allow
export WANDB_RUN_ID=$RUN_NAME

echo "[$(hostname)] $(date -Is) resume $RUN_NAME on gpu $GPU from $RESUME"
cd "$REPO"
exec "$VENV/bin/msdelta-train" \
    --config "$TMP_CFG" --run-name "$RUN_NAME" --resume "$RESUME" \
    >"$LOGDIR/${TAG}.out" 2>"$LOGDIR/${TAG}.err"
