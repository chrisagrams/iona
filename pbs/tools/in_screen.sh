#!/bin/bash
# Run a long-lived login-node helper (a feeder, a watcher) in a named GNU screen session the user can attach to
# (notes/AGENT_PLAYBOOK_2.md A7, K198e; tmux is not installed on the Aurora login nodes):
#
#   pbs/tools/in_screen.sh <name> <log> <command> [args...]
#   screen -r <name>       # attach on the SAME login node (printed below); Ctrl-a d detaches
#   screen -ls             # this login node's sessions
#
# The session runs the command once and closes when it ends (no shell is kept open); its output goes to the screen
# window and is appended to <log>, followed by an "exited rc=N" line. Refuses if this login node already has a session
# of that name. Each start is recorded in $S/watchers/<name>.txt (login node, start time, command, log, how to attach
# and restart), so STATUS can say where every helper runs.
set -u
name=${1:?usage: $0 <name> <log> <command> [args...]}
log=${2:?usage: $0 <name> <log> <command> [args...]}
shift 2
(( $# )) || { echo "usage: $0 <name> <log> <command> [args...]" >&2; exit 2; }
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "in_screen: bad session name '$name'" >&2; exit 2; }
S=${S:-/lus/flare/projects/UIC-HPC/khuss/msdelta}
WATCHERS=${WATCHERS:-$S/watchers}

if screen -ls 2>/dev/null | grep -qE "^[[:space:]]*[0-9]+\.${name//./\\.}[[:space:]]"; then
    echo "in_screen: a session '$name' already runs on $(hostname); attach with: screen -r $name" >&2
    exit 1
fi
mkdir -p "$(dirname "$log")" "$WATCHERS" || exit 1
cmd=$(printf '%q ' "$@")
qlog=$(printf '%q' "$log")
inner="{ $cmd; } 2>&1 | tee -a $qlog; rc=\${PIPESTATUS[0]}; echo \"[\$(date -u +%FT%TZ)] $name exited rc=\$rc\" | tee -a $qlog"
screen -dmS "$name" bash -c "$inner" || { echo "in_screen: screen failed to start '$name'" >&2; exit 1; }
{
    echo "name: $name"
    echo "host: $(hostname)"
    echo "started: $(date -u +%FT%TZ)"
    echo "command: $cmd"
    echo "log: $log"
    echo "attach: ssh $(hostname) then screen -r $name"
    echo "restart: $(printf '%q ' "$0" "$name" "$log")$cmd"
} > "$WATCHERS/$name.txt"
echo "started screen session '$name' on $(hostname); log $log; attach: screen -r $name"
