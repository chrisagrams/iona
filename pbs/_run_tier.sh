#!/bin/bash
# Per-GPU helper. Pins one msdelta run (one config) to one local GPU.
# Used by both scale_*.pbs and mask_sweep.pbs.
#
# Args:
#   $1 = config path     absolute or repo-relative path to a yaml in configs/
#   $2 = gpu_id          0|1|2|3 — index into the node's 4 A100s
#
# Side effects:
#   - writes /eagle/UIC-HPC/cgrams/msdelta-runs/$RUN_NAME/   (ckpts, figs)
#   - writes $REPO/pbs/logs/$JOBID/$TAG.{out,err}
#   - writes a temp overlay yaml that rewrites data.root and log.out_dir
#     so the in-repo configs stay system-agnostic. SMOKE=1 additionally
#     shrinks the training schedule for 10-min debug-queue runs.

set -euo pipefail

SRC_CFG=${1:?config path required (e.g. configs/scale_S.yaml)}
GPU=${2:?gpu id required (0|1|2|3)}

REPO=/home/cgrams/msdelta
VENV=$REPO/.venv

# Resolve to absolute and derive a short tag (basename without .yaml).
case "$SRC_CFG" in
    /*) ;;
    *) SRC_CFG="$REPO/$SRC_CFG" ;;
esac
TAG=$(basename "$SRC_CFG" .yaml)

DATA_ROOT=/eagle/UIC-HPC/cgrams/msbench-datasets/consensus_100M/
OUT_DIR=/eagle/UIC-HPC/cgrams/msdelta-runs
mkdir -p "$OUT_DIR"

JOBID=${PBS_JOBID%%.*}
JOBID=${JOBID:-local}
RUN_NAME=${TAG}_${JOBID}
LOGDIR=$REPO/pbs/logs/${JOBID}
mkdir -p "$LOGDIR"

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
    cfg["log"]["probe_every"] = 10_000_000   # skip — slow + not needed for smoke
    cfg["log"]["wandb_project"] = cfg["log"].get("wandb_project", "msdelta") + "-smoke"
with open(dst, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
PY

# One GPU per process; CUDA_VISIBLE_DEVICES masks the others so torch
# only ever sees device 0 from its perspective.
export CUDA_VISIBLE_DEVICES=$GPU

# Polaris compute node has 64 cores / 4 GPUs = 16 cores/GPU. Leave room
# for the dataloader workers (num_workers=8 in the configs).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Polaris compute nodes route outbound HTTPS through ALCF's proxy; without
# this wandb.init silently falls back to offline. no_proxy excludes the
# internal loopback / cluster traffic so we don't try to proxy NCCL or
# the local wandb daemon.
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,.alcf.anl.gov,.anl.gov
export NO_PROXY=$no_proxy

# Per-run wandb run id so resubmits join the same run if the job is
# re-launched against the same name; remove last.pt + bump JOBID for a
# clean restart.
export WANDB_DIR=$LOGDIR
export WANDB_RESUME=allow
export WANDB_RUN_ID=$RUN_NAME

echo "[$(hostname)] $(date -Is) tag=$TAG gpu=$GPU run=$RUN_NAME cfg=$TMP_CFG"

cd "$REPO"
exec "$VENV/bin/msdelta-train" \
    --config "$TMP_CFG" \
    --run-name "$RUN_NAME" \
    >"$LOGDIR/${TAG}.out" 2>"$LOGDIR/${TAG}.err"
