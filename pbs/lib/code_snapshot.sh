# Run this job's Python from a frozen copy of the code, so that switching branches or
# editing the checkout while the job is queued or running cannot change what it runs.
#
# Source it AFTER REPO_DIR (and SCRATCH_ROOT, if set) are defined and after the line that
# exports PYTHONPATH=$REPO_DIR:
#
#     source "$REPO_DIR/pbs/lib/code_snapshot.sh"
#
# What it does:
#   - copies the code directories (msdelta, sweeps, pbs minus logs, tests, baselines_wip
#     *.py/*.sh/*.pbs, data *.py/*.sh) to $SCRATCH_ROOT/code-snapshots/<job id>/ with rsync;
#     never .keys, results, runs, data files, logs, or configs (configs are read at job start,
#     and the sweep runner already snapshots its own grid);
#   - writes SNAPSHOT.txt there: source checkout, branch, commit, and any uncommitted changes;
#   - points Python at the copy: PYTHONPATH=<copy> (the checkout is removed from it) and
#     PYTHONSAFEPATH=1, so `python -m msdelta...` does not pick the checkout up from the cwd;
#   - exports MSDELTA_CODE_DIR=<copy> for scripts launched by path (use
#     "$MSDELTA_CODE_DIR/pbs/diag/x.py", not pbs/diag/x.py).
# The working directory is NOT changed: relative outputs (results/, pbs/logs, ./runs) still
# land in the checkout, as before. Secrets are still read from the checkout by load_keys.sh.
#
# MSDELTA_NO_SNAPSHOT=1 turns it off (runs the checkout directly, the old behaviour).
# Snapshots are small (~3 MB) and are never deleted automatically.

if [[ -n ${MSDELTA_NO_SNAPSHOT:-} ]]; then
    export MSDELTA_CODE_DIR=$REPO_DIR
    echo "=== code snapshot: disabled (MSDELTA_NO_SNAPSHOT), running the checkout $REPO_DIR"
else
    _snap_id=${PBS_JOBID%%.*}
    _snap_id=${_snap_id:-local-$(date +%Y%m%d-%H%M%S)-$$}
    _snap_root=${SCRATCH_ROOT:-/lus/flare/projects/UIC-HPC/$USER/msdelta}/code-snapshots
    export MSDELTA_CODE_DIR=$_snap_root/$_snap_id
    mkdir -p "$MSDELTA_CODE_DIR"
    if ! rsync -a \
            --include='/msdelta/***' --include='/sweeps/***' --include='/tests/***' \
            --exclude='/pbs/logs/' --include='/pbs/***' \
            --include='/baselines_wip/' --include='/baselines_wip/**/' \
            --include='/baselines_wip/**.py' --include='/baselines_wip/**.sh' \
            --include='/baselines_wip/**.pbs' \
            --include='/data/' --include='/data/*.py' --include='/data/*.sh' \
            --include='/pyproject.toml' --include='/pytest.ini' \
            --exclude='__pycache__/' --exclude='*' \
            "$REPO_DIR/" "$MSDELTA_CODE_DIR/"; then
        echo "code snapshot: rsync failed; refusing to run from a partial copy" >&2
        exit 3
    fi
    {
        echo "source:   $REPO_DIR"
        echo "branch:   $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)"
        echo "commit:   $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null)"
        echo "taken:    $(date -Is)"
        echo "job:      ${PBS_JOBID:-none}"
        echo "uncommitted changes in the checkout at snapshot time (code dirs):"
        git -C "$REPO_DIR" status --porcelain -- msdelta sweeps pbs tests baselines_wip data 2>/dev/null \
            | grep -v '^?? pbs/logs' | sed 's/^/    /'
    } > "$MSDELTA_CODE_DIR/SNAPSHOT.txt"
    # Replace the checkout with the copy on PYTHONPATH, keep everything else.
    _snap_pp=""
    IFS=':' read -r -a _snap_parts <<< "${PYTHONPATH:-}"
    for _p in "${_snap_parts[@]}"; do
        [[ -z $_p || $_p == "$REPO_DIR" || $_p == "$REPO_DIR/" ]] && continue
        _snap_pp=${_snap_pp:+$_snap_pp:}$_p
    done
    export PYTHONPATH=$MSDELTA_CODE_DIR${_snap_pp:+:$_snap_pp}
    export PYTHONSAFEPATH=1
    echo "=== code snapshot: $MSDELTA_CODE_DIR ($(sed -n 's/^commit: *//p' "$MSDELTA_CODE_DIR/SNAPSHOT.txt" | cut -c1-8) on $(sed -n 's/^branch: *//p' "$MSDELTA_CODE_DIR/SNAPSHOT.txt"))"
    unset _snap_id _snap_root _snap_pp _snap_parts _p
fi
