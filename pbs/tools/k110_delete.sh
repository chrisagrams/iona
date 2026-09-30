#!/bin/bash
# K110 staged deletion (user-approved tiers only; deletion protocol: triple-check, dry run, test on a copy,
# staged live). Usage:
#   pbs/tools/k110_delete.sh <list-file> [--live]      # without --live: dry run (prints what it would delete)
# Every path is re-checked at deletion time: under the allowed root, exists, no final/ (whole-run tiers),
# its job is not in qstat. Anything failing a check is skipped and reported.
set -uo pipefail
LIST=$1; LIVE=${2:-}
ROOT=${K110_ROOT:-/lus/flare/projects/UIC-HPC/khuss/msdelta/runs}
live_jobs=$(qstat -u "$USER" 2>/dev/null | awk 'NR>5{print $1}' | cut -d. -f1)
n=0; skipped=0
while read -r p; do
    [[ -z $p || $p == \#* ]] && continue
    reason=
    [[ $p == "$ROOT"/* ]] || reason="outside $ROOT"
    [[ -e $p ]] || reason="missing"
    base=$(basename "$(dirname "$p")")/$(basename "$p")
    if [[ -z $reason && -d $p && ${K110_MODE:-run} == run ]]; then
        [[ -d $p/final ]] && reason="has final/"
        job=${p##*-}; grep -qx "$job" <<<"$live_jobs" && reason="job $job is live"
    fi
    if [[ -n $reason ]]; then echo "SKIP $p ($reason)"; skipped=$((skipped+1)); continue; fi
    if [[ $LIVE == --live ]]; then rm -rf -- "$p" && echo "DELETED $p"; else echo "WOULD DELETE $p"; fi
    n=$((n+1))
done < "$LIST"
echo "$([[ $LIVE == --live ]] && echo deleted || echo would delete): $n, skipped: $skipped"
