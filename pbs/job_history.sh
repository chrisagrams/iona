#!/bin/bash
# Reconstruct what every finetune job actually did, from its log.
#
#   pbs/job_history.sh              # every job with a log
#   pbs/job_history.sh 88403        # jobs whose id starts with this
#
# STATUS.md carries this table by hand, and a hand-kept table drifts. This regenerates it
# from the logs, which cannot drift: a job either has a fault line or it does not, either
# reached its last step or did not. Paste the output into STATUS.md.
cd "$(dirname "$0")/.." || exit 1
filter=${1:-}
printf '%-9s %-8s %-11s %-26s %s\n' JOB TASK PARALLELISM OUTCOME NOTE
for log in pbs/logs/${filter}*.OU; do
    [ -f "$log" ] || continue
    job=$(basename "$log"); job=${job%%.*}
    text=$(tr '\r' '\n' < "$log" 2>/dev/null)

    task=$(grep -qa "\[align\]" <<<"$text" && echo align \
        || { grep -qa "denoise sweep" <<<"$text" && echo grid || echo denoise; })
    tiles=$(grep -oaE "hosts *: [0-9]+ x [0-9]+ tiles" <<<"$text" | head -1 \
            | grep -oE "[0-9]+ tiles")
    grep -qa -- "--deepspeed\|deepspeed configs" configs/*/training.args 2>/dev/null
    ds=$(grep -qa "DeepSpeed info\|deepspeed" <<<"$text" && echo " DS" || echo "")
    par="${tiles:-?}${ds}"
    grep -qa "denoise sweep" <<<"$text" && par="$(grep -oaE '[0-9]+ tile\(s\) each' <<<"$text" | head -1)"

    faults=$(grep -ca "Segmentation fault from GPU" <<<"$text")
    last=$(grep -oaE "[0-9]+/[0-9]+ \[" <<<"$text" | tail -1 | tr -d ' [')
    if   grep -qa "=== done:" <<<"$text"; then
        outcome=$(grep -oaE "=== done: [0-9]+/[0-9]+ arms ok" <<<"$text" | head -1 | sed 's/=== done: //')
    elif [ "$faults" -gt 0 ]; then outcome="GPU FAULT at ${last:-?}"
    elif grep -qa "^\[rank0\]: [A-Za-z]*Error\|^Traceback" <<<"$text"; then
        outcome="ERROR: $(grep -oaE "^[A-Za-z]*(Error|Exception): .*" <<<"$text" | head -1 | cut -c1-40)"
    elif grep -qa "stale against their template" <<<"$text"; then outcome="refused: stale arms"
    elif grep -qa "saved to\|test:" <<<"$text"; then outcome="COMPLETE ${last:-}"
    elif [ -n "$last" ]; then outcome="ran to $last"
    else outcome="no training"; fi

    note=$(grep -oaE "'test_auroc': [0-9.]+|'eval_loss': '[0-9.]+'" <<<"$text" | tail -1)
    printf '%-9s %-8s %-11s %-26s %s\n' "$job" "$task" "$par" "$outcome" "$note"
done
