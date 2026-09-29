#!/bin/bash
# Submit a list of approved jobs as queue slots free up (Aurora caps queued jobs per user).
#
#   setsid nohup pbs/tools/feeder.sh <plan-file> > $S/logs/feeder.log 2>&1 < /dev/null &
#
# Plan file, one job per line, '#' comments:
#   <name>|<after>|<qsub arguments...>
# <after> is '-' or the name of an earlier line: the job is submitted only after that job has
# finished with exit status 0; if it fails, this job is marked BLOCKED and never submitted.
# Commands run from the repo root; "$S" is available in the arguments.
# State (one file per job name, holding the job ID or BLOCKED) lives in $STATE_DIR, so a
# restarted feeder never submits twice. Exits when every line is submitted or blocked.
set -u
PLAN=$1
REPO=$(cd "$(dirname "$0")/../.." && pwd)
S=${S:-/lus/flare/projects/UIC-HPC/khuss/msdelta}
STATE_DIR=${STATE_DIR:-$S/feeder/$(basename "$PLAN" .txt)}
POLL=${POLL:-180}
mkdir -p "$STATE_DIR" "$S/notifications"
cd "$REPO" || exit 1

log() { echo "$(date -Is) $*"; }
exit_status() {  # prints the exit status of a finished job, nothing while it is queued/running
    qstat -xf "$1" 2>/dev/null | awk '/job_state = F/{f=1} /Exit_status =/{e=$3} END{if (f) print e}'
}

while true; do
    pending=0
    while IFS='|' read -r name after args; do
        [[ -z ${name// } || $name == \#* ]] && continue
        [[ -s $STATE_DIR/$name ]] && continue
        pending=1
        if [[ $after != - ]]; then
            dep=$(cat "$STATE_DIR/$after" 2>/dev/null)
            [[ -z $dep ]] && continue
            if [[ $dep == BLOCKED ]]; then
                echo BLOCKED > "$STATE_DIR/$name"; log "$name BLOCKED ($after blocked)"; continue
            fi
            st=$(exit_status "$dep")
            [[ -z $st ]] && continue
            if [[ $st != 0 ]]; then
                echo BLOCKED > "$STATE_DIR/$name"
                log "$name BLOCKED: $after ($dep) exited $st"
                echo "$(date -Is) feeder: $name not submitted, $after ($dep) exited $st" \
                    >> "$S/notifications/feeder.txt"
                continue
            fi
        fi
        out=$(eval "qsub $args" 2>&1)
        if [[ $out =~ ^[0-9]+\. ]]; then
            echo "${out%%.*}" > "$STATE_DIR/$name"; log "$name submitted: ${out%%.*}"
        elif [[ $out == *"per-user limit"* ]]; then
            break                       # queue full: retry the same job next round
        else
            log "$name qsub error (will retry): $out"
        fi
    done < "$PLAN"
    (( pending )) || { log "all jobs submitted or blocked"; exit 0; }
    sleep "$POLL"
done
