#!/usr/bin/env bash
# Per-rank wrapper that iona.launch runs under mpiexec on Aurora.
# Usage: aurora-rank.sh COMMAND [ARGS...]
# Maps the Aurora launcher's rank variables to the torch.distributed names, which
# torch reads from the environment before any Iona code runs, and on each host's
# local rank zero runs Telegraf for the duration of COMMAND. The IONA_* and LOG_DIR
# variables come from iona.env.AuroraPlatform via iona.launch.
set -euo pipefail

export RANK="${PMIX_RANK:-${PALS_RANKID:-${PMI_RANK:-}}}"
if [[ -z $RANK ]]; then
    echo "Aurora launcher did not provide PMIX_RANK, PALS_RANKID, or PMI_RANK" >&2
    exit 2
fi
export WORLD_SIZE="$IONA_WORLD_SIZE"
export LOCAL_RANK="${PALS_LOCAL_RANKID:-${MPI_LOCALRANKID:-$((RANK % IONA_DEVICES_PER_HOST))}}"
export LOCAL_WORLD_SIZE="$IONA_DEVICES_PER_HOST"
if [[ $LOCAL_RANK == 0 && -z ${IONA_TELEGRAF_BIN:-} ]]; then
    echo "warning: telegraf not found (set IONA_TELEGRAF_BIN); skipping XPU metrics" >&2
elif [[ $LOCAL_RANK == 0 ]]; then
    telegraf_bin=/tmp/telegraf-$USER
    cp "$IONA_TELEGRAF_BIN" "$telegraf_bin"
    "$telegraf_bin" --config "$IONA_XPU_TELEGRAF_CONFIG" \
        >"$LOG_DIR/xpu-telegraf-${HOSTNAME}.log" 2>&1 &
    telegraf_pid=$!
    trap 'kill "$telegraf_pid" 2>/dev/null || true; wait "$telegraf_pid" 2>/dev/null || true' EXIT
    export IONA_XPU_METRICS_URL="http://127.0.0.1:${IONA_XPU_METRICS_PORT}/metrics"
    for ((attempt = 0; attempt < IONA_TELEGRAF_STARTUP_SECONDS * 2; attempt++)); do
        if curl --noproxy 127.0.0.1 --fail --silent \
            "$IONA_XPU_METRICS_URL" >/dev/null; then
            metrics_endpoint_ready=1
            break
        fi
        if ! kill -0 "$telegraf_pid" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done
    if [[ ${metrics_endpoint_ready:-0} != 1 ]]; then
        echo "XPU metrics endpoint failed to start; see $LOG_DIR/xpu-telegraf-${HOSTNAME}.log" >&2
        exit 2
    fi
    "$@"
    exit $?
fi
exec "$@"
