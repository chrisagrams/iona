# Source from an Aurora PBS job to prepare the job-wide training environment.
# Loads modules, mounts DAOS, computes the walltime deadline, validates inputs,
# and exports everything pbs/aurora-run.sh needs to launch one training run.
# Nothing here is specific to a single run, so several runs can share it.
#
# Inputs (environment): ARGS_FILE, DAOS_POOL, LOG_ROOT, and optionally DAOS_CONT,
# CHECKPOINT_DIR, HF_HOME, XPUS_PER_HOST, MICRO_BATCH_SIZE, GLOBAL_BATCH_SIZE,
# PROBE_EXECUTION, DENOISE_XPU_TILES, RETRIEVAL_XPU_TILES, TELEGRAF_BIN.
# Outputs: JOB_HOSTS, a comma-separated list of the job's unique hosts.

module use /soft/modulefiles
module load daos
module load frameworks/2026.1.0
module load xpu-smi

set -euo pipefail

JOB_START_EPOCH=$(date +%s)
CHECKPOINT_MARGIN_SECONDS=${CHECKPOINT_MARGIN_SECONDS:-900}
if ! [[ $CHECKPOINT_MARGIN_SECONDS =~ ^[0-9]+$ ]]; then
    echo "CHECKPOINT_MARGIN_SECONDS must be a nonnegative integer" >&2
    exit 2
fi

if [[ -n ${JOB_WALLTIME_SECONDS:-} ]]; then
    if ! [[ $JOB_WALLTIME_SECONDS =~ ^[0-9]+$ ]] || ((JOB_WALLTIME_SECONDS < 1)); then
        echo "JOB_WALLTIME_SECONDS must be a positive integer" >&2
        exit 2
    fi
else
    PBS_WALLTIME=$(qstat -f "${PBS_JOBID:?PBS_JOBID is required}" | awk -F ' = ' \
        '/^[[:space:]]*Resource_List.walltime = / {print $2; exit}')
    IFS=: read -r -a WALLTIME_PARTS <<<"$PBS_WALLTIME"
    if ((${#WALLTIME_PARTS[@]} != 3)) ||
        ! [[ ${WALLTIME_PARTS[0]} =~ ^[0-9]+$ && ${WALLTIME_PARTS[1]} =~ ^[0-9]+$ && ${WALLTIME_PARTS[2]} =~ ^[0-9]+$ ]]; then
        echo "could not parse PBS Resource_List.walltime: $PBS_WALLTIME" >&2
        exit 2
    fi
    JOB_WALLTIME_SECONDS=$((
        10#${WALLTIME_PARTS[0]} * 3600 +
        10#${WALLTIME_PARTS[1]} * 60 +
        10#${WALLTIME_PARTS[2]}
    ))
fi
if ((CHECKPOINT_MARGIN_SECONDS >= JOB_WALLTIME_SECONDS)); then
    echo "CHECKPOINT_MARGIN_SECONDS must be less than the requested walltime" >&2
    exit 2
fi
export IONA_JOB_DEADLINE_EPOCH=$((JOB_START_EPOCH + JOB_WALLTIME_SECONDS))
export IONA_CHECKPOINT_MARGIN_SECONDS=$CHECKPOINT_MARGIN_SECONDS

REPO_DIR=${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}
VENV_DIR=${VENV_DIR:-$REPO_DIR/.venv}
ARGS_FILE=${ARGS_FILE:?set ARGS_FILE to an Iona training args file}
DAOS_POOL=${DAOS_POOL:?set DAOS_POOL to the project DAOS pool}
DAOS_CONT=${DAOS_CONT:-msdelta-training}
LOG_ROOT=${LOG_ROOT:?set LOG_ROOT to a Flare base directory for logs}
if [[ $LOG_ROOT != /* ]]; then
    echo "LOG_ROOT must be an absolute shared-filesystem path" >&2
    exit 2
fi

if ! command -v launch-dfuse-with-caching.sh >/dev/null 2>&1; then
    echo "launch-dfuse-with-caching.sh is required for Hugging Face mmap access" >&2
    exit 2
fi
launch-dfuse-with-caching.sh "${DAOS_POOL}:${DAOS_CONT}"
DAOS_ROOT=/tmp/$DAOS_POOL/$DAOS_CONT
if [[ ! -d $DAOS_ROOT ]]; then
    echo "DAOS dfuse mount not found: $DAOS_ROOT" >&2
    exit 2
fi

CHECKPOINT_DIR=${CHECKPOINT_DIR:-$DAOS_ROOT/checkpoints}
HF_HOME=${HF_HOME:-$DAOS_ROOT/hf-cache}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-$HF_HOME/datasets}
PREPROCESSED_DATASET_DIR=${PREPROCESSED_DATASET_DIR:-$DATASET_CACHE_DIR/msdelta-preprocessed}
PREPROCESSED_PROBE_DIR=${PREPROCESSED_PROBE_DIR:-$DATASET_CACHE_DIR/msdelta-probes}
if [[ ! -d $PREPROCESSED_DATASET_DIR ]]; then
    echo "preprocessed dataset not found: $PREPROCESSED_DATASET_DIR" >&2
    exit 2
fi

if [[ $ARGS_FILE != /* ]]; then
    ARGS_FILE=$REPO_DIR/$ARGS_FILE
fi
if [[ $CHECKPOINT_DIR != /* ]]; then
    CHECKPOINT_DIR=$REPO_DIR/$CHECKPOINT_DIR
fi
if [[ ! -f $ARGS_FILE ]]; then
    echo "arguments file not found: $ARGS_FILE" >&2
    exit 2
fi
if [[ ! -x $VENV_DIR/bin/python ]]; then
    echo "Python interpreter not found: $VENV_DIR/bin/python" >&2
    exit 2
fi
XPU_TELEGRAF_CONFIG=${IONA_XPU_TELEGRAF_CONFIG:-$REPO_DIR/pbs/xpu-telegraf.conf}
TELEGRAF_BIN=${TELEGRAF_BIN:-$(command -v telegraf || true)}
if [[ -n $TELEGRAF_BIN && ! -x $TELEGRAF_BIN ]]; then
    echo "TELEGRAF_BIN is not executable: $TELEGRAF_BIN" >&2
    exit 2
fi
if [[ ! -f $XPU_TELEGRAF_CONFIG ]]; then
    echo "XPU Telegraf configuration not found: $XPU_TELEGRAF_CONFIG" >&2
    exit 2
fi
if ! command -v mpiexec >/dev/null 2>&1; then
    echo "mpiexec is required for an Aurora launch" >&2
    exit 2
fi
if [[ ! -f ${PBS_NODEFILE:-} ]]; then
    echo "PBS_NODEFILE is required for an Aurora launch" >&2
    exit 2
fi

JOB_ID=${PBS_JOBID:-local}
JOB_ID=${JOB_ID%%.*}
mapfile -t CONFIG_VALUES < <(
    "$VENV_DIR/bin/python" - "$ARGS_FILE" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
tokens = path.read_text().split()


def argument(name: str, default: str | None = None) -> str | None:
    values = [tokens[index + 1] for index, token in enumerate(tokens[:-1]) if token == name]
    return values[-1] if values else default


run_name = argument("--run_name", path.parent.name)
wandb_project = argument("--wandb_project", "")
print(run_name)
print(wandb_project)
print(argument("--probe_execution", "sidecar"))
print(argument("--denoise_steps", "0"))
print(argument("--retrieval_steps", "0"))
PY
)
CONFIG_RUN_NAME=${CONFIG_VALUES[0]}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-64}
CONFIG_WANDB_PROJECT=${CONFIG_VALUES[1]}
PROBE_EXECUTION=${PROBE_EXECUTION:-${CONFIG_VALUES[2]}}
DENOISE_STEPS=${CONFIG_VALUES[3]}
RETRIEVAL_STEPS=${CONFIG_VALUES[4]}
if [[ $PROBE_EXECUTION != off ]]; then
    for spec in "$DENOISE_STEPS:denoise" "$RETRIEVAL_STEPS:retrieval"; do
        steps=${spec%%:*}
        kind=${spec#*:}
        if ((steps > 0)) && [[ ! -f $PREPROCESSED_PROBE_DIR/$kind/dataset_dict.json ]]; then
            echo "finalized $kind probes not found: $PREPROCESSED_PROBE_DIR/$kind; run aurora-preprocess.pbs" >&2
            exit 2
        fi
    done
    if ((RETRIEVAL_STEPS > 0)) && [[ ! -f $PREPROCESSED_PROBE_DIR/retrieval-evaluation/dataset_dict.json ]]; then
        echo "finalized retrieval evaluation not found: $PREPROCESSED_PROBE_DIR/retrieval-evaluation" >&2
        exit 2
    fi
fi
mkdir -p "$HF_HOME" "$DATASET_CACHE_DIR"
TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-$USER}

cd "$REPO_DIR"
JOB_HOSTS=$(awk '!seen[$0]++ {print $0}' "$PBS_NODEFILE" | paste -sd, -)
XPUS_PER_HOST=${XPUS_PER_HOST:-8}
if ((XPUS_PER_HOST < 1 || XPUS_PER_HOST > 12)); then
    echo "XPUS_PER_HOST must be between 1 and 12 on Aurora" >&2
    exit 2
fi
if [[ $XPUS_PER_HOST == 8 ]]; then
    PRETRAIN_XPU_TILES=${ZE_AFFINITY_MASK:-0,1,2,3,6,7,8,9}
elif [[ $XPUS_PER_HOST == 12 ]]; then
    PRETRAIN_XPU_TILES=${ZE_AFFINITY_MASK:-0,1,2,3,4,5,6,7,8,9,10,11}
else
    PRETRAIN_XPU_TILES=${ZE_AFFINITY_MASK:-$(seq -s, 0 $((XPUS_PER_HOST - 1)))}
fi
DENOISE_XPU_TILES=${DENOISE_XPU_TILES:-4,5}
RETRIEVAL_XPU_TILES=${RETRIEVAL_XPU_TILES:-10,11}
if [[ $PROBE_EXECUTION == sidecar ]]; then
    if [[ ${ZE_FLAT_DEVICE_HIERARCHY:-FLAT} != FLAT ]]; then
        echo "sidecars require ZE_FLAT_DEVICE_HIERARCHY=FLAT" >&2
        exit 2
    fi
    used_tiles=,
    for spec in "$DENOISE_STEPS:$DENOISE_XPU_TILES" "$RETRIEVAL_STEPS:$RETRIEVAL_XPU_TILES"; do
        steps=${spec%%:*}
        ((steps > 0)) || continue
        tiles=${spec#*:}
        if [[ ! $tiles =~ ^([0-9]|1[01])(,([0-9]|1[01]))*$ ]]; then
            echo "invalid sidecar tile list: $tiles" >&2
            exit 2
        fi
        IFS=, read -r -a probe_tiles <<<"$tiles"
        for tile in "${probe_tiles[@]}"; do
            if [[ ,$PRETRAIN_XPU_TILES, == *",$tile,"* || $used_tiles == *",$tile,"* ]]; then
                echo "sidecar tile $tile overlaps pretraining or another probe assignment" >&2
                exit 2
            fi
            used_tiles+="$tile,"
        done
    done
fi
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}

if [[ $XPUS_PER_HOST == 8 ]]; then
    export ZE_AFFINITY_MASK=$PRETRAIN_XPU_TILES
    export CPU_BIND=${CPU_BIND:-verbose,list:4-7:8-11:12-15:16-19:56-59:60-63:64-67:68-71}
elif [[ $XPUS_PER_HOST == 12 ]]; then
    export ZE_AFFINITY_MASK=$PRETRAIN_XPU_TILES
    export CPU_BIND=${CPU_BIND:-verbose,list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79}
else
    ZE_AFFINITY_MASK=$PRETRAIN_XPU_TILES
    export ZE_AFFINITY_MASK
    export CPU_BIND=${CPU_BIND:-none}
fi
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}
export ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}
unset SYCL_DEVICE_FILTER

VISIBLE_XPUS=$("$VENV_DIR/bin/python" -c 'import torch; print(torch.xpu.device_count())')
if [[ $VISIBLE_XPUS != "$XPUS_PER_HOST" ]]; then
    echo "expected $XPUS_PER_HOST visible XPU tiles, found $VISIBLE_XPUS" >&2
    exit 2
fi

echo "$(date -Is) job=${PBS_JOBID:-local} hosts=$JOB_HOSTS xpus_per_host=$XPUS_PER_HOST global_batch=$GLOBAL_BATCH_SIZE micro_batch=$MICRO_BATCH_SIZE"
echo "daos=${DAOS_POOL}:${DAOS_CONT} dataset=$PREPROCESSED_DATASET_DIR checkpoints=$CHECKPOINT_DIR"
echo "walltime=${JOB_WALLTIME_SECONDS}s checkpoint_margin=${CHECKPOINT_MARGIN_SECONDS}s deadline=$IONA_JOB_DEADLINE_EPOCH"
echo "inductor_cache=$TORCHINDUCTOR_CACHE_DIR"
"$VENV_DIR/bin/python" -c 'import torch; print(f"torch={torch.__version__} xccl={torch.distributed.is_xccl_available()}")'

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-userfaultfd}
unset CCL_ZE_IPC
unset CCL_ZE_IPC_EXCHANGE
export HF_HOME
export HF_DATASETS_CACHE=$DATASET_CACHE_DIR
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TORCHINDUCTOR_CACHE_DIR
# Python multiprocessing binds AF_UNIX sockets under TMPDIR, which must stay
# well under the 107-byte socket path limit; the PBS job TMPDIR can exceed it.
export TMPDIR=${IONA_TMPDIR:-/tmp}
export PYTHONPATH=$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}
export IONA_XPUS_PER_HOST=$XPUS_PER_HOST
export IONA_XPU_TELEGRAF_CONFIG=$XPU_TELEGRAF_CONFIG
export IONA_TELEGRAF_BIN=$TELEGRAF_BIN
export IONA_TELEGRAF_STARTUP_SECONDS=${IONA_TELEGRAF_STARTUP_SECONDS:-120}
export IONA_XPU_METRICS_PORT=${IONA_XPU_METRICS_PORT:-9274}

ALCF_PROXY=${ALCF_PROXY:-http://proxy.alcf.anl.gov:3128}
export HTTP_PROXY=${HTTP_PROXY:-$ALCF_PROXY}
export HTTPS_PROXY=${HTTPS_PROXY:-$ALCF_PROXY}
export http_proxy=${http_proxy:-$ALCF_PROXY}
export https_proxy=${https_proxy:-$ALCF_PROXY}
export ftp_proxy=${ftp_proxy:-$ALCF_PROXY}
export no_proxy=${no_proxy:-admin,localhost,*.cm.aurora.alcf.anl.gov,aurora-*,*.aurora.alcf.anl.gov,*.alcf.anl.gov}
export no_proxy="$no_proxy,127.0.0.1"
export NO_PROXY="${NO_PROXY:-$no_proxy},127.0.0.1"

export WANDB_RESUME=${WANDB_RESUME:-allow}
export WANDB_PROJECT=${WANDB_PROJECT:-$CONFIG_WANDB_PROJECT}

# Consumed by pbs/aurora-run.sh, which runs as a separate process.
export REPO_DIR VENV_DIR ARGS_FILE LOG_ROOT CHECKPOINT_DIR JOB_ID JOB_HOSTS
export DATASET_CACHE_DIR PREPROCESSED_DATASET_DIR PREPROCESSED_PROBE_DIR
export CONFIG_RUN_NAME PROBE_EXECUTION DENOISE_XPU_TILES RETRIEVAL_XPU_TILES
export XPUS_PER_HOST MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE
