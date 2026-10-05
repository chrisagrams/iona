#!/bin/bash
# Alert when a running PBS job stops making progress (notes/AGENT_PLAYBOOK_2.md A7, K198e). Alert only: it never stops
# the job (qdel needs the user's approval).
#
#   pbs/tools/in_screen.sh watch-<job> $S/logs/watch-<job>.log pbs/tools/stall_watch.sh <job> <progress file>...
#
# Every POLL s (default 300) while the job is running (R): when none of the progress files (e.g. the train log, which
# tqdm rewrites every step) has changed size or mtime for STALL_MIN minutes (default 30; STALL_SEC overrides), write ONE
# alert, $NOTIFY_DIR/watchdog/<UTC>_watchdog_<job>_stalled.json, and print it; when progress resumes, a "resumed" alert,
# then it re-arms. Alerts carry source "watchdog" and sit in a subfolder, so the DAG's job-notification reader
# (pbs/dag/contract.py) never takes them for job ends. The clock starts when the job is first seen running; a queued
# or held job is only waited for. A failing qstat is no verdict: the round is skipped. When the job has finished (F) or
# is unknown to PBS, a "finished" alert with its exit status is written and the watcher exits 0.
set -u
JOB=${1:?usage: $0 <job id> <progress file>...}
shift
(( $# )) || { echo "usage: $0 <job id> <progress file>..." >&2; exit 2; }
FILES=("$@")
POLL=${POLL:-300}
STALL_SEC=${STALL_SEC:-$(( ${STALL_MIN:-30} * 60 ))}
S=${S:-/lus/flare/projects/UIC-HPC/khuss/msdelta}
NOTIFY_DIR=${NOTIFY_DIR:-$S/notifications}
QSTAT=${QSTAT:-qstat}   # tests put a fake qstat here

_iso() { date -u -d "@$1" +%FT%TZ; }
_json_str() { local s=${1//\\/\\\\}; s=${s//\"/\\\"}; printf '"%s"' "${s//$'\n'/ }"; }
alert() {   # alert <event> <text>
    local now d f tmp files="" p
    now=$(date +%s)
    d=$NOTIFY_DIR/watchdog
    mkdir -p "$d"
    for p in "${FILES[@]}"; do files+="${files:+,}$(_json_str "$p")"; done
    f=$d/$(date -u -d "@$now" +%Y%m%dT%H%M%SZ)_watchdog_${JOB}_$1.json
    tmp=$d/.$(basename "$f").tmp
    printf '{"schema":1,"source":"watchdog","job_id":%s,"event":%s,"time":%s,"text":%s,"files":[%s],"host":%s}\n' \
        "$(_json_str "$JOB")" "$(_json_str "$1")" "$(_json_str "$(_iso "$now")")" "$(_json_str "$2")" "$files" \
        "$(_json_str "$(hostname)")" > "$tmp" && mv "$tmp" "$f"
    echo "[$(_iso "$now")] ALERT $1: $2 ($f)"
}
signature() {   # size:mtime of every progress file ("-" when missing)
    local p out=""
    for p in "${FILES[@]}"; do out+="$(stat -c '%s:%Y' "$p" 2>/dev/null || echo -) "; done
    printf '%s' "$out"
}

echo "[$(_iso "$(date +%s)")] watching job $JOB on $(hostname): poll ${POLL}s, stall after ${STALL_SEC}s, files: ${FILES[*]}"
last_sig="" last_change=0 stalled=0 seen_running=0
while :; do
    out=$("$QSTAT" -x -f "$JOB" 2>&1); rc=$?
    state=$(sed -n 's/^[[:space:]]*job_state = //p' <<<"$out" | head -1)
    if [[ -z $state ]]; then
        if grep -q "Unknown Job Id" <<<"$out"; then
            alert finished "job $JOB is unknown to PBS (finished and purged, or never existed)"
            exit 0
        fi
        echo "[$(_iso "$(date +%s)")] qstat failed (rc $rc): $(head -1 <<<"$out") -- no verdict, retrying"
        sleep "$POLL"; continue
    fi
    now=$(date +%s)
    case $state in
        F|X)
            ex=$(sed -n 's/^[[:space:]]*Exit_status = //p' <<<"$out" | head -1)
            alert finished "job $JOB finished (state $state, exit status ${ex:-unknown})"
            exit 0 ;;
        R)
            sig=$(signature)
            if (( ! seen_running )); then
                seen_running=1 last_sig=$sig last_change=$now
                echo "[$(_iso "$now")] job $JOB is running"
            elif [[ $sig != "$last_sig" ]]; then
                last_sig=$sig last_change=$now
                if (( stalled )); then
                    stalled=0
                    alert resumed "job $JOB is making progress again"
                fi
            elif (( ! stalled && now - last_change >= STALL_SEC )); then
                stalled=1
                alert stalled "job $JOB is running but its progress files have not changed for $(( (now - last_change) / 60 )) min (since $(_iso "$last_change")); check it -- the watchdog does not stop jobs"
            fi ;;
        *)  : ;;   # Q, H, W, E...: wait
    esac
    sleep "$POLL"
done
