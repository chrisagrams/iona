#!/usr/bin/env bash
# Invoked by the posttraining callback on Aurora's global-rank-zero host.
# Usage: bash pbs/aurora-probe.sh xpu:TILE /path/to/python [posttraining arguments]
# Inherits the frameworks runtime, data/cache paths, and network settings.
set -euo pipefail

if (($# < 2)); then
    echo "usage: $0 xpu:TILE PYTHON [posttraining arguments]" >&2
    exit 2
fi
probe_device=$1
probe_python=$2
shift 2
if [[ ! $probe_device =~ ^xpu:([0-9]|1[01])$ ]]; then
    echo "Aurora probes require xpu:TILE with TILE between 0 and 11" >&2
    exit 2
fi
probe_tile=${probe_device#xpu:}
if [[ ${ZE_FLAT_DEVICE_HIERARCHY:-FLAT} != FLAT ]]; then
    echo "Aurora probe tile assignments require FLAT device numbering" >&2
    exit 2
fi
case ",${ZE_AFFINITY_MASK:-}," in
    *",$probe_tile,"*)
        echo "probe tile $probe_tile overlaps the pretraining device mask" >&2
        exit 2
        ;;
esac
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK="$probe_tile"

# This is a standalone Trainer, not another member of pretraining's DDP group.
while IFS= read -r probe_variable; do
    case "$probe_variable" in
        RANK|WORLD_SIZE|LOCAL_RANK|LOCAL_WORLD_SIZE|MASTER_ADDR|MASTER_PORT|\
        PMI_*|PMIX_*|PALS_*|OMPI_*|MV2_*|MPI_*|CCL_*|ACCELERATE_*|TORCHELASTIC_*)
            unset "$probe_variable"
            ;;
    esac
done < <(compgen -e)

exec "$probe_python" -m msdelta.posttraining "$@"
