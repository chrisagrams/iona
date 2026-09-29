#!/bin/bash
# ONE arm, but reproducing the GRID's invocation exactly rather than my hand-built one.
# Every controlled test so far reserves 38.73 GB; the grid reserves 67.11. Concurrency
# and card placement are ruled out, so the difference must be in HOW the grid launches:
# full epochs rather than --max_steps, W&B on, checkpointing on, SaveEncoderCallback
# firing, and the launcher's extra environment (ZES_ENABLE_SYSMAN, FI_CXI_* tuning).
cd /home/khuss/code/msdelta
source "${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}/pbs/lib/load_frameworks.sh"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
export PYTHONPATH=/home/khuss/code/msdelta
export HF_HOME=/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MSDELTA_XPUS_PER_HOST=1 MSDELTA_WORLD_SIZE=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=${CCL_KVS_MODE:-pmi}  # K138-I: "mpi" breaks on the 2026-09 stack
export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OFLOW_BUF_SIZE=8388608 FI_CXI_CQ_FILL_PERCENT=20
export ZES_ENABLE_SYSMAN=1          # <- the launcher sets this; nothing of mine did
unset CCL_ZE_IPC CCL_ZE_IPC_EXCHANGE
export MASTER_ADDR=$(hostname) MASTER_PORT=29900
export WANDB_MODE=disabled WANDB_DISABLED=true
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
C200=/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-200m-production-01-checkpoint-192799
OUT=/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/gridexact; mkdir -p "$OUT"

go() {  # tag, extra args
  local tag=$1; shift
  mpiexec --envall --pmi=pmix -n 1 --ppn 1 --cpu-bind depth --depth 8 \
    /home/khuss/code/msdelta/pbs/rank_wrapper.sh \
    /home/khuss/code/msdelta/.venv/bin/python -m msdelta.finetune_contrastive \
      --args_file configs/finetune-contrastive-50m/training.args \
      --pretrained_path "$C200" --output_dir "$OUT/$tag" "$@" \
      > "$OUT/$tag.log" 2>&1
  local s=$?
  local m=$(grep -oE 'peak [0-9.]+ GB reserved [0-9.]+ GB' "$OUT/$tag.log" | tail -1)
  if grep -q 'Segmentation fault from GPU' "$OUT/$tag.log"; then echo "$tag GPUFAULT  $m"
  elif [[ $s -eq 0 ]]; then echo "$tag PASS  $m"; else echo "$tag EXIT($s)  $m"; fi
}
# Full epochs and full save behaviour -- exactly what the grid runs. Capped by time,
# not by max_steps, so the scheduler and checkpointing behave as they do in the grid.
go full_grid_config --report_to none
