#!/bin/bash
# One cell of the parallelism diagnostic (see pbs/parallelism_diag.pbs).
#
# Every cell trains configs/massivekb_xl.yaml at a FIXED global batch of 512 and
# a FIXED lr of 1.3e-4 — Spark's known-good settings — varying ONLY how the work
# is parallelised. That makes the three curves directly comparable and isolates
# the one axis every previous experiment confounded:
#
#   A  1 GPU, no DeepSpeed, bs64 x gacc8   -> replicates massivekb_xl-spark on A100
#   B  4 GPU, plain DDP,    bs64 x gacc2   -> A vs B isolates multi-GPU itself
#   C  4 GPU, ZeRO-2,       bs64 x gacc2   -> B vs C isolates DeepSpeed
#
# Args:
#   $1 = tag         short cell name (A-1gpu-nods | B-4gpu-nods | C-4gpu-zero2)
#   $2 = nproc       GPUs / torchrun ranks on this node (1 or 4)
#   $3 = deepspeed   True | False  (--deepspeed override)
#   $4 = grad_accum  accumulation steps, chosen so nproc*64*gacc == 512
#   $5 = jobid       PBS job id, for the run name + log dir
#
# LR schedule: `constant_with_warmup` with warmup 2000 rather than the config's
# cosine. Spark ran cosine over 90000 steps, so across the first 3000 steps its
# LR is within 0.03% of flat 1.3e-4 — constant_with_warmup reproduces that
# exactly, while a cosine truncated to total_steps=3000 would anneal to ~0 inside
# the window and make the cells incomparable to Spark (and to each other).

set -euo pipefail

TAG=${1:?cell tag required}
NPROC=${2:?nproc required (1|4)}
DS=${3:?deepspeed required (True|False)}
GACC=${4:?grad_accum required}
JOBID=${5:-local}

REPO=/home/cgrams/msdelta
VENV=$REPO/.venv
CFG=$REPO/configs/massivekb_xl.yaml

RUN_NAME=mkbxl-par-${TAG}-${JOBID}
LOGDIR=$REPO/pbs/logs/${JOBID}
mkdir -p "$LOGDIR"

cd "$REPO"
echo "[$(hostname)] $(date -Is) cell=$TAG nproc=$NPROC deepspeed=$DS gacc=$GACC" \
     "gbs=$((NPROC * 64 * GACC)) run=$RUN_NAME"

# Threads scale with rank count so all cells get the same per-rank CPU budget
# (64 hardware threads / 4 ranks vs / 1 rank); the 6 loader workers per rank do
# the actual CPU-heavy preprocessing and are set by the config.
export OMP_NUM_THREADS=$(( 8 / NPROC ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ALCF proxy for rank-0 wandb; keep loopback/cluster traffic off the proxy so the
# node-local NCCL rendezvous (--standalone) and the wandb daemon still work.
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,.alcf.anl.gov,.anl.gov
export NO_PROXY=$no_proxy

# HF dataset cache on eagle (pre-populated); read from disk, no Hub round-trip.
export HF_HOME=/eagle/UIC-HPC/cgrams/hf
export HF_HUB_OFFLINE=1

export WANDB_DIR=$LOGDIR
export WANDB_RESUME=allow
export WANDB_RUN_ID=$RUN_NAME

# Inline science off entirely: build_callbacks registers a probe only when its
# cadence is truthy, so 0 means the callbacks are never even constructed. That
# matters beyond wall-clock — the configs now carry replicate_retrieval_repo,
# whose callback would fetch an external HF dataset, and this job runs with
# HF_HUB_OFFLINE=1. ckpt stays large rather than 0 (save_strategy="steps" wants
# a positive save_steps); it simply never fires inside 3000 steps.
exec "$VENV/bin/torchrun" \
    --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m msdelta.train \
    --config "$CFG" \
    --run-name "$RUN_NAME" \
    --deepspeed "$DS" \
    --grad_accum_steps "$GACC" \
    --lr 1.3e-4 \
    --grad_clip 1.0 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 2000 \
    --total_steps 3000 \
    --val_every 250 \
    --probe_every 0 \
    --bias_curve_every 0 \
    --ckpt_every 100000000 \
    >"$LOGDIR/${TAG}.out" 2>"$LOGDIR/${TAG}.err"
