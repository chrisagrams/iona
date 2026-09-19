#!/bin/bash
# Translate the Aurora launcher's rank variables into the ones torch.distributed expects,
# then exec the real command.
#
# This lives in a file rather than inline in `mpiexec ... bash -c '...'` because the
# inline form has to be single-quoted inside a shell function, and embedding a single
# quote there needs '"'"' sequences that are easy to get wrong. When that quoting
# collapsed in job 8839937 the outer shell expanded $RANK itself, died on `set -u`, and
# every arm silently never ran.
set -euo pipefail

export RANK="${PMIX_RANK:-${PALS_RANKID:-${PMI_RANK:-}}}"
[[ -n $RANK ]] || { echo "launcher provided no PMIX_RANK/PALS_RANKID/PMI_RANK" >&2; exit 2; }
export WORLD_SIZE="${MSDELTA_WORLD_SIZE:?}"
export LOCAL_RANK="${PALS_LOCAL_RANKID:-${MPI_LOCALRANKID:-$(( RANK % MSDELTA_XPUS_PER_HOST ))}}"
export LOCAL_WORLD_SIZE="${MSDELTA_XPUS_PER_HOST:?}"
exec "$@"
