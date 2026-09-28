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
#   - writes SNAPSHOT.txt there: source checkout, mode, branch, the full commit sha, a
#     `dirty: yes|no` flag and the list of uncommitted changes (plus uncommitted.diff when
#     dirty, so a dirty run can be rebuilt as commit + patch);
#   - points Python at the copy: PYTHONPATH=<copy> (the checkout is removed from it) and
#     PYTHONSAFEPATH=1, so `python -m msdelta...` does not pick the checkout up from the cwd;
#   - exports MSDELTA_CODE_DIR=<copy> for scripts launched by path (use
#     "$MSDELTA_CODE_DIR/pbs/diag/x.py", not pbs/diag/x.py).
# The working directory is NOT changed: relative outputs (results/, pbs/logs, ./runs) still
# land in the checkout, as before. Secrets are still read from the checkout by load_keys.sh.
#
# MSDELTA_NO_SNAPSHOT=1 turns it off (runs the checkout directly, the old behaviour).
# Snapshots are small (~3 MB) and are never deleted automatically.
#
# Running a COMMIT instead of the working tree (K90; notes/DAG_SPEC.md section 9, "Running code from a
# commit"): with MSDELTA_CODE_REF=<sha> -- set by pbs/qsub_ref, which resolves a branch or
# tag to a sha at submit time -- the snapshot is `git archive <sha>` of the same code paths
# plus configs/, never the working tree. SNAPSHOT.txt then says `mode: git-archive`, the
# ref, the full commit, and `dirty: n/a`. A sha the repository does not have fails the job
# (exit 3). Sourcing it again in the same shell only re-applies the PYTHONPATH change.
# Without MSDELTA_CODE_REF nothing below changes.
#
# MSDELTA_CODE_REUSE=<an earlier job's snapshot dir> runs that snapshot as is, copying
# nothing (aurora-finetune-sweep.pbs sets it when resuming a job whose snapshot was dirty).
#
# For scripts, in both modes (they are identity functions without a ref):
#   MSDELTA_REF_PREFIX       "" without a ref, "$MSDELTA_CODE_DIR/" with one: prefix a
#                            repo-relative path the script reads ("${MSDELTA_REF_PREFIX}tests/x.py")
#   msdelta_code_path P      where to read a repo-relative file P that may be committed OR
#                            ad hoc: with a ref, P resolves into the snapshot if the commit has
#                            it, else stays P (an uncommitted models/arms file is read from
#                            where it is). Absolute paths are returned unchanged.
MSDELTA_REF_PREFIX=
msdelta_code_path() {
    if [[ -n ${MSDELTA_CODE_REF:-} && -n ${MSDELTA_CODE_DIR:-} && $1 != /* \
          && -e $MSDELTA_CODE_DIR/$1 ]]; then
        printf '%s\n' "$MSDELTA_CODE_DIR/$1"
    else
        printf '%s\n' "$1"
    fi
}

if [[ -n ${MSDELTA_NO_SNAPSHOT:-} ]]; then
    if [[ -n ${MSDELTA_CODE_REF:-} ]]; then
        echo "code snapshot: MSDELTA_CODE_REF=$MSDELTA_CODE_REF needs the snapshot;" \
             "unset MSDELTA_NO_SNAPSHOT" >&2
        exit 3
    fi
    export MSDELTA_CODE_DIR=$REPO_DIR
    echo "=== code snapshot: disabled (MSDELTA_NO_SNAPSHOT), running the checkout $REPO_DIR"
elif [[ -n ${MSDELTA_CODE_REUSE:-} ]]; then
    # An EARLIER job's snapshot, used as is (a resumed sweep whose original job ran a dirty
    # working tree: the commit alone would not reproduce it). Nothing is copied.
    if [[ ! -f $MSDELTA_CODE_REUSE/SNAPSHOT.txt ]]; then
        echo "code snapshot: MSDELTA_CODE_REUSE=$MSDELTA_CODE_REUSE has no SNAPSHOT.txt" >&2
        exit 3
    fi
    export MSDELTA_CODE_DIR=$MSDELTA_CODE_REUSE
    MSDELTA_REF_PREFIX=$MSDELTA_CODE_DIR/
    _snap_pp=""
    IFS=':' read -r -a _snap_parts <<< "${PYTHONPATH:-}"
    for _p in "${_snap_parts[@]}"; do
        [[ -z $_p || $_p == "$REPO_DIR" || $_p == "$REPO_DIR/" || $_p == "$MSDELTA_CODE_DIR" ]] && continue
        _snap_pp=${_snap_pp:+$_snap_pp:}$_p
    done
    export PYTHONPATH=$MSDELTA_CODE_DIR${_snap_pp:+:$_snap_pp}
    export PYTHONSAFEPATH=1
    echo "=== code snapshot: REUSING $MSDELTA_CODE_DIR ($(sed -n 's/^commit: *//p' "$MSDELTA_CODE_DIR/SNAPSHOT.txt" | cut -c1-12), job $(sed -n 's/^job: *//p' "$MSDELTA_CODE_DIR/SNAPSHOT.txt"))"
    unset _snap_pp _snap_parts _p
elif [[ -n ${MSDELTA_CODE_REF:-} ]]; then
    _snap_sha=$(git -C "$REPO_DIR" rev-parse --verify --quiet "${MSDELTA_CODE_REF}^{commit}" \
                    2>/dev/null) || _snap_sha=""
    if [[ ! $MSDELTA_CODE_REF =~ ^[0-9a-f]{7,40}$ || -z $_snap_sha ]]; then
        echo "code snapshot: MSDELTA_CODE_REF=$MSDELTA_CODE_REF is not a commit sha in $REPO_DIR" >&2
        echo "  (branch and tag names are resolved at submit time: use pbs/qsub_ref <ref> ...)" >&2
        exit 3
    fi
    if [[ ${_MSDELTA_REF_SNAP_DONE:-} != "$_snap_sha" ]]; then
        _snap_id=${PBS_JOBID:-}; _snap_id=${_snap_id%%.*}
        _snap_id=${_snap_id:-local-$(date +%Y%m%d-%H%M%S)-$$}
        _snap_root=${SCRATCH_ROOT:-/lus/flare/projects/UIC-HPC/$USER/msdelta}/code-snapshots
        export MSDELTA_CODE_DIR=$_snap_root/$_snap_id
        mkdir -p "$MSDELTA_CODE_DIR"
        # The same code paths as the rsync below, plus configs/ (sweep grids, args files and
        # the generators' templates are committed and must match the code); only those the
        # commit has, since git archive fails on a missing pathspec.
        _snap_specs=()
        for _p in $(git -C "$REPO_DIR" ls-tree --name-only "$_snap_sha" -- \
                        msdelta sweeps pbs tests configs pyproject.toml pytest.ini); do
            _snap_specs+=("$_p")
        done
        while IFS= read -r _p; do
            [[ -n $_p ]] && _snap_specs+=("$_p")
        done < <(git -C "$REPO_DIR" ls-tree -r --name-only "$_snap_sha" -- baselines_wip \
                     | grep -E '\.(py|sh|pbs)$'
                 git -C "$REPO_DIR" ls-tree --name-only "$_snap_sha" -- data/ \
                     | grep -E '^data/[^/]+\.(py|sh)$')
        if (( ${#_snap_specs[@]} == 0 )); then
            echo "code snapshot: commit $_snap_sha has none of the code paths" >&2
            exit 3
        fi
        if ! git -C "$REPO_DIR" archive --format=tar --output="$MSDELTA_CODE_DIR/.archive.tar" \
                 "$_snap_sha" -- "${_snap_specs[@]}" ':(exclude)pbs/logs' \
           || ! tar -x -f "$MSDELTA_CODE_DIR/.archive.tar" -C "$MSDELTA_CODE_DIR"; then
            echo "code snapshot: git archive of $_snap_sha failed; refusing to run from a partial copy" >&2
            exit 3
        fi
        rm -f "$MSDELTA_CODE_DIR/.archive.tar"
        {
            echo "source:   $REPO_DIR"
            echo "mode:     git-archive"
            echo "ref:      ${MSDELTA_CODE_REF_NAME:-$MSDELTA_CODE_REF}"
            echo "branch:   ${MSDELTA_CODE_REF_NAME:-$MSDELTA_CODE_REF}"
            echo "commit:   $_snap_sha"
            echo "taken:    $(date -Is)"
            echo "job:      ${PBS_JOBID:-none}"
            echo "dirty:    n/a (built from the commit, not from the checkout's working tree)"
        } > "$MSDELTA_CODE_DIR/SNAPSHOT.txt"
        _MSDELTA_REF_SNAP_DONE=$_snap_sha
        echo "=== code snapshot: $MSDELTA_CODE_DIR (git archive of ${_snap_sha:0:12}," \
             "ref ${MSDELTA_CODE_REF_NAME:-$MSDELTA_CODE_REF}; checkout $REPO_DIR not used for code)"
    fi
    MSDELTA_REF_PREFIX=$MSDELTA_CODE_DIR/
    _snap_pp=""
    IFS=':' read -r -a _snap_parts <<< "${PYTHONPATH:-}"
    for _p in "${_snap_parts[@]}"; do
        [[ -z $_p || $_p == "$REPO_DIR" || $_p == "$REPO_DIR/" || $_p == "$MSDELTA_CODE_DIR" ]] && continue
        _snap_pp=${_snap_pp:+$_snap_pp:}$_p
    done
    export PYTHONPATH=$MSDELTA_CODE_DIR${_snap_pp:+:$_snap_pp}
    export PYTHONSAFEPATH=1
    unset _snap_sha _snap_id _snap_root _snap_pp _snap_parts _snap_specs _p
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
    # Every job records the exact commit it was launched from and whether the code it
    # copied differed from it, so any job can be reproduced (or resumed) from that commit.
    # A dirty tree also leaves uncommitted.diff (`git diff HEAD --binary` of the code
    # paths: tracked changes only; untracked files are in the copy itself).
    _snap_dirty=$(git -C "$REPO_DIR" status --porcelain -- msdelta sweeps pbs tests baselines_wip data 2>/dev/null \
                  | grep -v '^?? pbs/logs')
    {
        echo "source:   $REPO_DIR"
        echo "mode:     working-tree (rsync)"
        echo "branch:   $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)"
        echo "commit:   $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null)"
        echo "taken:    $(date -Is)"
        echo "job:      ${PBS_JOBID:-none}"
        if [[ -n $_snap_dirty ]]; then
            echo "dirty:    yes (the copy is the commit plus the changes below; see uncommitted.diff)"
        else
            echo "dirty:    no (the copy is exactly the commit's code paths)"
        fi
        echo "uncommitted changes in the checkout at snapshot time (code dirs):"
        [[ -z $_snap_dirty ]] || printf '%s\n' "$_snap_dirty" | sed 's/^/    /'
    } > "$MSDELTA_CODE_DIR/SNAPSHOT.txt"
    if [[ -n $_snap_dirty ]]; then
        git -C "$REPO_DIR" diff HEAD --binary -- msdelta sweeps pbs tests baselines_wip data \
            pyproject.toml pytest.ini ':(exclude)pbs/logs' > "$MSDELTA_CODE_DIR/uncommitted.diff" 2>/dev/null
    fi
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
    unset _snap_id _snap_root _snap_pp _snap_parts _p _snap_dirty
fi
