#!/usr/bin/env bash
# Invoked by the posttraining callback on Aurora's global-rank-zero host.
# Usage: bash pbs/aurora-probe.sh xpu:TILE[,TILE...] /path/to/python [posttraining arguments]
# Inherits the frameworks runtime, data/cache paths, and network settings.
set -euo pipefail

if (($# < 2)); then
    echo "usage: $0 xpu:TILE[,TILE...] PYTHON [posttraining arguments]" >&2
    exit 2
fi
probe_device=$1
probe_python=$2
shift 2
if [[ ! $probe_device =~ ^xpu:([0-9]|1[01])(,([0-9]|1[01]))*$ ]]; then
    echo "Aurora probes require xpu:TILE[,TILE...] with tiles between 0 and 11" >&2
    exit 2
fi
probe_mask=${probe_device#xpu:}
if [[ ${ZE_FLAT_DEVICE_HIERARCHY:-FLAT} != FLAT ]]; then
    echo "Aurora probe tile assignments require FLAT device numbering" >&2
    exit 2
fi
IFS=, read -r -a probe_tiles <<<"$probe_mask"
probe_seen=,
for probe_tile in "${probe_tiles[@]}"; do
    if [[ ,${ZE_AFFINITY_MASK:-}, == *",$probe_tile,"* || $probe_seen == *",$probe_tile,"* ]]; then
        echo "probe tile $probe_tile overlaps pretraining or is repeated" >&2
        exit 2
    fi
    probe_seen+="$probe_tile,"
done
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK="$probe_mask"

# This is a standalone Trainer, not another member of pretraining's DDP group.
while IFS= read -r probe_variable; do
    case "$probe_variable" in
        RANK|WORLD_SIZE|LOCAL_RANK|LOCAL_WORLD_SIZE|MASTER_ADDR|MASTER_PORT|\
        PMI_*|PMIX_*|PALS_*|OMPI_*|MV2_*|MPI_*|CCL_*|ACCELERATE_*|TORCHELASTIC_*)
            unset "$probe_variable"
            ;;
    esac
done < <(compgen -e)

# Use an independent W&B service while logging to the same shared run.
# Inheriting the parent service can collide with its already initialized run.
unset WANDB_SERVICE

# A fresh rendezvous on a free port keeps simultaneous probes independent.
exec "$probe_python" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${#probe_tiles[@]}" --max-restarts=0 \
    --module msdelta.posttraining "$@"
