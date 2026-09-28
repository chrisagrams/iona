# Job-end reporting for the job DAG (notes/PLAN.md I2; pbs/dag/README.md). Source it; do
# not execute it. Source it AFTER SCRATCH_ROOT is defined, as early as possible, so that an
# early `exit 2` (stale grid, bad arm) is reported too:
#
#     source "$REPO_DIR/pbs/lib/job_finish.sh"
#
# It installs an EXIT trap that ALWAYS writes one notification file when the job ends
#
#     $SCRATCH_ROOT/notifications/<UTC %Y%m%dT%H%M%SZ>_<jobid>_<node>_<status>.json
#
# status: ok | failed | partial (some sweep arms ok) | walltime | killed (SIGTERM early),
# with runtime, outputs, checks, arms, snapshot commit and a one-line summary. On success,
# and only then, it first writes the success manifest -- LAST, after every output:
#
#     $SCRATCH_ROOT/manifests/<jobid>/_SUCCESS.json
#
# Success = exit status 0, every declared output exists, every check passed and, for a
# sweep, JF_ARMS_OK == JF_ARMS_TOTAL. Both files are written atomically (dot-tmp + mv).
#
# The script declares what it produced and what it verified:
#     jf_output PATH...          outputs that must exist at exit
#     jf_check NAME CMD [ARG...]  run CMD now; record NAME as passed / failed
#     JF_ARMS_OK=n JF_ARMS_TOTAL=m   (sweeps, set before exiting)
#     JF_SUMMARY="..."           one line for the notification
#
# It never changes the job's exit status, and it is a NO-OP when DRY_RUN is set, when
# PBS_JOBID is unset (login-node runs, tests of the scripts) or when JF_DISABLE=1.
# SIGTERM (walltime or qdel) is trapped only to record it: the job still exits 143.
# Pure bash + coreutils: it must finish inside PBS's kill delay.
# Overrides: JF_NOTIFY_DIR, JF_MANIFEST_DIR. The scheduler passes DAG_NODE, DAG_PIPELINE,
# DAG_KIND and DAG_WALLTIME_SEC with -v (never secrets).

_jf_enabled=1
if [[ -n ${DRY_RUN:-} || -z ${PBS_JOBID:-} || -n ${JF_DISABLE:-} ]]; then
    _jf_enabled=
fi

jf_output() {
    [[ -n $_jf_enabled ]] || return 0
    _JF_OUTPUTS+=("$@")
}

jf_check() {
    [[ -n $_jf_enabled ]] || return 0
    local name=$1; shift
    if "$@"; then
        _JF_CHECKS+=("$name"$'\t'1); return 0
    fi
    _JF_CHECKS+=("$name"$'\t'0); return 1
}

_jf_str() {  # JSON string literal
    local s=$1
    s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/\\n}; s=${s//$'\t'/\\t}; s=${s//$'\r'/}
    printf '"%s"' "$s"
}

_jf_num() {  # a JSON number or null
    [[ $1 =~ ^[0-9]+$ ]] && printf '%s' "$1" || printf 'null'
}

_jf_write() {  # _jf_write PATH CONTENT -- atomic
    local path=$1 dir tmp
    dir=$(dirname "$path"); tmp="$dir/.$(basename "$path").tmp.$$"
    mkdir -p "$dir" && printf '%s\n' "$2" > "$tmp" && mv -f "$tmp" "$path"
}

_jf_on_exit() {
    local rc=$?
    [[ $BASHPID == "$_JF_PID" ]] || return 0
    set +e
    trap - EXIT TERM
    local end job node commit="" branch="" dirty=false snap status summary
    local outputs="" checks="" all_ok=1 p name ok missing=0 failed_checks=0
    end=$(date +%s)
    job=${PBS_JOBID%%.*}
    node=${DAG_NODE:-${PBS_JOBNAME:-job}}
    snap=${MSDELTA_CODE_DIR:-}/SNAPSHOT.txt
    if [[ -n ${MSDELTA_CODE_DIR:-} && -f $snap ]]; then
        commit=$(sed -n 's/^commit: *//p' "$snap" | head -1)
        branch=$(sed -n 's/^branch: *//p' "$snap" | head -1)
        sed -n '/^uncommitted changes/,$p' "$snap" | tail -n +2 | grep -q '[^[:space:]]' \
            && dirty=true
    fi
    for p in "${_JF_OUTPUTS[@]}"; do
        if [[ -e $p ]]; then ok=true; else ok=false; missing=$(( missing + 1 )); fi
        outputs+="${outputs:+,}{\"path\":$(_jf_str "$p"),\"exists\":$ok}"
    done
    for p in "${_JF_CHECKS[@]}"; do
        name=${p%$'\t'*}; ok=${p##*$'\t'}
        [[ $ok == 1 ]] && ok=true || { ok=false; failed_checks=$(( failed_checks + 1 )); }
        checks+="${checks:+,}{\"name\":$(_jf_str "$name"),\"ok\":$ok}"
    done
    local arms_ok=${JF_ARMS_OK:-} arms_total=${JF_ARMS_TOTAL:-} runtime=$(( end - _JF_START ))
    if [[ -n $_JF_SIGNAL ]]; then
        local wt=${DAG_WALLTIME_SEC:-}
        if [[ $wt =~ ^[0-9]+$ ]] && (( runtime * 100 >= wt * 95 )); then
            status=walltime
        else
            status=killed
        fi
    elif (( rc == 0 && missing == 0 && failed_checks == 0 )) \
         && [[ -z $arms_total || $arms_ok == "$arms_total" ]]; then
        status=ok
    elif [[ $arms_ok =~ ^[0-9]+$ && $arms_total =~ ^[0-9]+$ ]] \
         && (( arms_ok > 0 && arms_ok < arms_total )); then
        status=partial
    else
        status=failed
    fi
    summary=${JF_SUMMARY:-}
    [[ -n $summary ]] || summary="exit $rc${_JF_SIGNAL:+ (SIG$_JF_SIGNAL)}${arms_total:+, arms $arms_ok/$arms_total}"
    (( missing )) && summary+=", $missing output(s) missing"
    (( failed_checks )) && summary+=", $failed_checks check(s) failed"

    local common manifest="" mdir ndir stamp
    common="\"job_id\":$(_jf_str "$job"),\"job_name\":$(_jf_str "${PBS_JOBNAME:-}"),\"node\":$(_jf_str "$node"),\"pipeline\":$(_jf_str "${DAG_PIPELINE:-}"),\"kind\":$(_jf_str "${DAG_KIND:-}"),\"commit\":$(_jf_str "$commit"),\"branch\":$(_jf_str "$branch"),\"dirty\":$dirty,\"snapshot\":$(_jf_str "${MSDELTA_CODE_DIR:-}"),\"outputs\":[${outputs}],\"checks\":[${checks}],\"arms_ok\":$(_jf_num "$arms_ok"),\"arms_total\":$(_jf_num "$arms_total"),\"resume_job\":$(_jf_str "${RESUME_JOB:-}"),\"runtime_sec\":$runtime"
    mdir=${JF_MANIFEST_DIR:-${SCRATCH_ROOT:-.}/manifests}/$job
    if [[ $status == ok ]]; then
        manifest=$mdir/_SUCCESS.json
        if ! _jf_write "$manifest" "{\"schema\":1,${common},\"written\":$(_jf_str "$(date -u +%FT%TZ)")}"; then
            status=failed; summary+=", manifest write failed"; manifest=""
        fi
    fi
    ndir=${JF_NOTIFY_DIR:-${SCRATCH_ROOT:-.}/notifications}
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    local nfile="$ndir/${stamp}_${job}_${node//[^A-Za-z0-9.~-]/-}_${status}.json"
    _jf_write "$nfile" "{\"schema\":1,\"source\":\"job\",${common},\"status\":$(_jf_str "$status"),\"exit_code\":$rc,\"signal\":$(_jf_str "$_JF_SIGNAL"),\"start\":$(_jf_str "$(date -u -d @"$_JF_START" +%FT%TZ)"),\"end\":$(_jf_str "$(date -u -d @"$end" +%FT%TZ)"),\"queue\":$(_jf_str "${PBS_QUEUE:-}"),\"nodes\":$(_jf_num "$( [[ -r ${PBS_NODEFILE:-} ]] && sort -u "$PBS_NODEFILE" | wc -l)"),\"host\":$(_jf_str "$(hostname 2>/dev/null)"),\"workdir\":$(_jf_str "${PBS_O_WORKDIR:-$PWD}"),\"manifest\":$( [[ -n $manifest ]] && _jf_str "$manifest" || printf null),\"summary\":$(_jf_str "$summary")}" \
        && echo "=== job_finish: $status ($summary) -> $nfile" \
        || echo "=== job_finish: could not write the notification to $ndir" >&2
}

if [[ -n $_jf_enabled ]]; then
    _JF_PID=$BASHPID
    _JF_START=$(date +%s)
    _JF_OUTPUTS=()
    _JF_CHECKS=()
    _JF_SIGNAL=
    trap '_jf_on_exit' EXIT
    trap '_JF_SIGNAL=TERM; exit 143' TERM
fi
