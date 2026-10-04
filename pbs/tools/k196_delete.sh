#!/bin/bash
# K196 staged deletion (user-approved tiers only; deletion protocol: triple-check, dry run, test on a redundant
# copy, staged live). Fail-closed successor of k110_delete.sh.
# Usage:
#   pbs/tools/k196_delete.sh <mode> <list-file> [--live]     # without --live: dry run
#   modes: ckpt  -- each line runs/<run>-<job>/checkpoint-<N>; the run must have final/model.safetensors (>1 MB)
#          run   -- each line runs/<run>-<job>; the run must NOT have final/
#          ckweights -- each line runs/<run>-<job>/checkpoint-<N>/model.safetensors (D1-lite); the run must have
#                   final/model.safetensors and the checkpoint must keep encoder/model.safetensors (>1 MB)
#          path  -- each line any path (dir or file) under the root, outside the never-touch areas below
#          scoredft -- (K196b) each line runs/sweep-s<scale>_ck<NNN>k_lr4e-4_p170k2_cons_seed<i>-<job>, NNN != 540: a
#                   finished consensus fine-tune on an intermediate pretraining checkpoint. Deleted only if, for EVERY
#                   set in K196_SETS, $K196_SCORED/cons-allck-<set>/<run without 'sweep-' and '-<job>'>.json exists and its "path" is exactly
#                   <live root>/<line>/final (so the scores are of this very run) and the job is not live
# List lines are paths RELATIVE to the root, so the same list runs on the redundant copy (K196_ROOT=<copy>)
# and live. Every line is re-checked at deletion time; anything failing a check is skipped and reported.
#   K196_ROOT     root (default /lus/flare/projects/UIC-HPC/khuss/msdelta)
#   K196_PROTECT  file of relative paths that must never be deleted, nor anything above or below them
set -uo pipefail
MODE=${1:?mode}; LIST=${2:?list}; LIVE=${3:-}
ROOT=${K196_ROOT:-/lus/flare/projects/UIC-HPC/khuss/msdelta}
ROOT=${ROOT%/}
case $MODE in ckpt|ckweights|run|path|scoredft) ;; *) echo "bad mode $MODE" >&2; exit 2;; esac
[[ -f $LIST ]] || { echo "no list $LIST" >&2; exit 2; }
[[ -d $ROOT && ! -L $ROOT ]] || { echo "bad root $ROOT" >&2; exit 2; }

# Fail closed: without a working qstat we cannot know which jobs are live.
qs=$(qstat -u "$USER" 2>&1) || { echo "ABORT: qstat failed: $qs" >&2; exit 3; }
live_jobs=$(awk '$1 ~ /^[0-9]+\./ {split($1,a,"."); print a[1]}' <<<"$qs")
echo "live/queued jobs: $(echo $live_jobs)"

protect=()
if [[ -n ${K196_PROTECT:-} ]]; then
    [[ -f $K196_PROTECT ]] || { echo "ABORT: protect file $K196_PROTECT missing" >&2; exit 2; }
    mapfile -t protect < <(grep -v '^\s*$' "$K196_PROTECT")
fi

LIVE_ROOT=/lus/flare/projects/UIC-HPC/khuss/msdelta   # what the scored JSONs' "path" fields name (copy tests too)
SCORED=${K196_SCORED:-$(cd "$(dirname "$0")/../.." && pwd)/results/raw/finetune/contrastive}
read -r -a SETS <<<"${K196_SETS:-validation test oodval mouse human yeast}"
if [[ $MODE == scoredft ]]; then
    [[ -d $SCORED ]] || { echo "ABORT: scored dir $SCORED missing" >&2; exit 2; }
    command -v python3 >/dev/null || { echo "ABORT: no python3 to read the scored JSONs" >&2; exit 2; }
fi

# Areas never touched in any mode (relative to the root).
NEVER='^(pretrained|pretrained-random|eval-data|huggingface/hub|huggingface/datasets/chrisagrams___[^/]+|data/[^/]+/(preprocessed|raw)|data/probe-cap512|data/stage0-cap150|feeder|manifests|k110-logs|k196-logs|code-snapshots)(/|$)'

n=0; skipped=0
while IFS= read -r rel || [[ -n $rel ]]; do
    [[ -z $rel || $rel == \#* ]] && continue
    rel=${rel%/}; p=$ROOT/$rel; reason=
    if [[ $rel == /* || $rel == *..* || $rel == *'*'* || $rel == *'?'* ]]; then reason="not a plain relative path"
    elif [[ ! -e $p && ! -L $p ]]; then reason="missing"
    elif [[ -L $p ]]; then reason="is a symlink"
    elif [[ $(readlink -f -- "$p") != "$p" ]]; then reason="symlink in its path"
    elif [[ $rel =~ $NEVER ]]; then reason="never-touch area"
    elif [[ $rel == */.* || $rel == .* ]]; then reason="hidden path (.configs-<job> etc.)"
    fi
    if [[ -z $reason ]]; then
        for k in "${protect[@]}"; do
            k=${k%/}
            if [[ $rel == "$k" || $rel == "$k"/* || $k == "$rel"/* ]]; then reason="protected ($k)"; break; fi
        done
    fi
    if [[ -z $reason ]]; then
        case $MODE in
        ckpt)
            if [[ ! $rel =~ ^runs/[^/]+-([0-9]{7})/checkpoint-[0-9]+$ ]]; then reason="not runs/<run>-<job>/checkpoint-N"
            else
                job=${BASH_REMATCH[1]}; run=${p%/*}
                if [[ ! -d $p ]]; then reason="not a dir"
                elif [[ ! -f $run/final/model.safetensors || $(stat -c %s "$run/final/model.safetensors") -lt 1000000 ]]; then reason="run has no final/model.safetensors"
                elif grep -qx "$job" <<<"$live_jobs"; then reason="job $job is live"
                fi
            fi ;;
        ckweights)
            if [[ ! $rel =~ ^runs/[^/]+-([0-9]{7})/checkpoint-[0-9]+/model\.safetensors$ ]]; then reason="not runs/<run>-<job>/checkpoint-N/model.safetensors"
            else
                job=${BASH_REMATCH[1]}; ck=${p%/*}; run=${ck%/*}
                if [[ ! -f $p ]]; then reason="not a file"
                elif [[ ! -f $ck/encoder/model.safetensors || -L $ck/encoder/model.safetensors || $(stat -c %s "$ck/encoder/model.safetensors") -lt 1000000 ]]; then reason="no encoder/model.safetensors to keep"
                elif [[ ! -f $run/final/model.safetensors || $(stat -c %s "$run/final/model.safetensors") -lt 1000000 ]]; then reason="run has no final/model.safetensors"
                elif grep -qx "$job" <<<"$live_jobs"; then reason="job $job is live"
                fi
            fi ;;
        run)
            if [[ ! $rel =~ ^runs/[^/]+-([0-9]{7})$ ]]; then reason="not runs/<run>-<job>"
            else
                job=${BASH_REMATCH[1]}
                if [[ ! -d $p ]]; then reason="not a dir"
                elif [[ -e $p/final ]]; then reason="has final/"
                elif grep -qx "$job" <<<"$live_jobs"; then reason="job $job is live"
                fi
            fi ;;
        scoredft)
            if [[ ! $rel =~ ^runs/sweep-(s[0-9]+m_ck([0-9]{3})k_lr4e-4_p170k2_cons_seed[0-9])-([0-9]{7})$ ]]; then
                reason="not runs/sweep-s<scale>_ck<NNN>k_lr4e-4_p170k2_cons_seed<i>-<job>"
            else
                name=${BASH_REMATCH[1]}; ck=${BASH_REMATCH[2]}; job=${BASH_REMATCH[3]}
                if [[ $ck == 540 ]]; then reason="final pretraining checkpoint (540k) -- never in K196b"
                elif [[ ! -d $p ]]; then reason="not a dir"
                elif grep -qx "$job" <<<"$live_jobs"; then reason="job $job is live"
                else
                    for set in "${SETS[@]}"; do
                        js=$SCORED/cons-allck-$set/$name.json
                        if [[ ! -f $js ]]; then reason="not scored on $set"; break; fi
                        got=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("path",""))' "$js" 2>/dev/null)
                        if [[ $got != "$LIVE_ROOT/$rel/final" ]]; then reason="$set score is of another run ($got)"; break; fi
                    done
                fi
            fi ;;
        path)  # explicit allowlist of shapes (K196 D3/D4); anything else is skipped
            if [[ $rel =~ ^huggingface/datasets/parquet/default-[0-9a-f]{16}$ || $rel =~ ^data/[^/]+/datasets$ ]]; then
                [[ -d $p ]] || reason="not a dir"
            elif [[ $rel =~ ^runs/(validate-sweep-[^/]+|armcount|armcount12|gridexact|p2-cost|p2-smoke|sibling2|k168|quarantine|pf-timing|allocfix2|ddp2)$ ]]; then
                [[ -d $p ]] || reason="not a dir"
            elif [[ $rel =~ /core\.x[0-9]+c[0-9]+s[0-9]+b[0-9]+n[0-9]+\.[0-9]+$ ]]; then
                if [[ ! -f $p ]]; then reason="not a file"
                elif ! file -b -- "$p" | grep -q "core file"; then reason="not an ELF core dump"; fi
            else
                reason="not an allowed D3/D4 shape"
            fi
            if [[ -z $reason ]]; then
                for j in $live_jobs; do [[ $rel == *"$j"* ]] && { reason="names live job $j"; break; }; done
            fi ;;
        esac
    fi
    if [[ -n $reason ]]; then echo "SKIP $rel ($reason)"; skipped=$((skipped+1)); continue; fi
    if [[ $LIVE == --live ]]; then
        if { if [[ $MODE == ckweights ]]; then rm -f -- "$p"; else rm -rf --one-file-system -- "$p"; fi; } && [[ ! -e $p ]]; then echo "DELETED $rel"; else echo "FAILED $rel"; skipped=$((skipped+1)); continue; fi
    else
        echo "WOULD DELETE $rel"
    fi
    n=$((n+1))
done < "$LIST"
echo "$([[ $LIVE == --live ]] && echo deleted || echo would delete): $n, skipped: $skipped"
