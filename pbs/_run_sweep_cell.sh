#!/bin/bash
# One LR-range-test cell: a full-node 4-GPU DDP run of configs/massivekb_xl.yaml
# with a flat (constant-after-warmup) LR and a short horizon, for co-tuning the
# peak LR against grad_clip. Invoked once per node by pbs/lr_sweep.pbs (which
# ssh's into each allocated node and runs this in the background).
#
# Why constant_with_warmup: a truncated cosine would anneal the LR to ~0 inside
# the 4k-step window (at total=4000/warmup=800 the cosine tail sits at ~10% of
# peak by the end), so configs would be ranked on where their cooldown landed
# rather than on training dynamics. A flat post-warmup LR keeps every cell at
# near-peak the whole way, which is what an LR-range test needs.
#
# Args:
#   $1 = lr          peak learning rate (e.g. 1.3e-4)
#   $2 = grad_clip   max_grad_norm (e.g. 5.0)
#   $3 = jobid       PBS job id, for the run name + log dir
#
# Reads: chris' shared eagle HF cache + msdelta-runs; writes one wandb run
#   named mkbxl-sweep-lr<lr>-clip<clip>-<jobid> under project msfound.

set -euo pipefail

LR=${1:?peak lr required (e.g. 1.3e-4)}
CLIP=${2:?grad_clip required (e.g. 5.0)}
JOBID=${3:-local}

REPO=/home/cgrams/msdelta
VENV=$REPO/.venv
CFG=$REPO/configs/massivekb_xl.yaml

RUN_NAME=mkbxl-sweep-lr${LR}-clip${CLIP}-${JOBID}
LOGDIR=$REPO/pbs/logs/${JOBID}
mkdir -p "$LOGDIR"

cd "$REPO"
echo "[$(hostname)] $(date -Is) cell lr=$LR clip=$CLIP run=$RUN_NAME gpus=$(nvidia-smi -L | wc -l)"

# 4 ranks share the node's cores; leave headroom for the 6 loader workers/rank.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ALCF proxy for rank-0 wandb; keep loopback/cluster traffic off the proxy so
# NCCL rendezvous (127.0.0.1 for --standalone) and the local wandb daemon work.
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,.alcf.anl.gov,.anl.gov
export NO_PROXY=$no_proxy

# HF dataset cache on eagle (pre-populated); read straight from disk, no Hub.
export HF_HOME=/eagle/UIC-HPC/cgrams/hf
export HF_HUB_OFFLINE=1

export WANDB_DIR=$LOGDIR
export WANDB_RESUME=allow
export WANDB_RUN_ID=$RUN_NAME

# Each cell is its own single-node --standalone DDP job (rdzv on localhost);
# distinct physical nodes never collide on the port. The overrides:
#   * constant_with_warmup + short horizon  -> flat-LR screen (see header)
#   * probe/bias/ckpt pushed past the horizon -> spend the hour on the training
#     signal, not the inline science (we only need train/loss, eval/loss,
#     train/grad_norm to rank cells)
exec "$VENV/bin/torchrun" \
    --standalone --nnodes=1 --nproc_per_node=4 \
    -m msdelta.train \
    --config "$CFG" \
    --run-name "$RUN_NAME" \
    --lr "$LR" \
    --grad_clip "$CLIP" \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 800 \
    --total_steps 4000 \
    --val_every 500 \
    --probe_every 100000000 \
    --bias_curve_every 100000000 \
    --ckpt_every 100000000 \
    >"$LOGDIR/lr${LR}-clip${CLIP}.out" 2>"$LOGDIR/lr${LR}-clip${CLIP}.err"
