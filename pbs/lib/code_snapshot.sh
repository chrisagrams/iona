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
#     *.py/*.sh/*.pbs, data *.py/*.sh, pyproject.toml, pytest.ini) to
#     $SCRATCH_ROOT/code-snapshots/<job id>/; never .keys, results, runs, data files, logs, or
#     configs (configs are read at job start, and the sweep runner already snapshots its own
#     grid). HOW it copies (K111, race-free against a concurrent fast-forward of the checkout):
#       * HEAD is resolved to a full sha FIRST. If those code paths have no uncommitted
#         changes (git status; untracked pbs/logs ignored), the copy is `git archive <sha>`
#         of them: committed objects, so a fast-forward running at that moment cannot give a
#         mix of old and new files. SNAPSHOT.txt: `mode: git-archive (clean HEAD)`.
#       * Only a DIRTY tree is copied with rsync (`mode: working-tree (rsync)`), holding a
#         SHARED flock on <git common dir>/msdelta-checkout.lock for the copy and the status
#         record; pbs/checkout_ff takes that lock EXCLUSIVELY while it fast-forwards, so the
#         two never overlap. HEAD is also read before and after the copy: if it moved (a
#         checkout change made without pbs/checkout_ff), the copy is retried once, then the
#         job fails (exit 3). MSDELTA_SNAPSHOT_LOCK_TIMEOUT (s, default 600) bounds the wait.
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

# The code paths of commit $1 that the snapshot copies, as git pathspecs, in _snap_specs
# (only those the commit has: git archive fails on a missing pathspec). $2 non-empty adds
# configs/ (commit mode). These are the rsync mode's paths below.
_msdelta_snap_specs() {
    local sha=$1 with_configs=${2:-} p
    local top=(msdelta sweeps pbs tests pyproject.toml pytest.ini)
    [[ -n $with_configs ]] && top+=(configs)
    _snap_specs=()
    for p in $(git -C "$REPO_DIR" ls-tree --name-only "$sha" -- "${top[@]}"); do
        _snap_specs+=("$p")
    done
    while IFS= read -r p; do
        [[ -n $p ]] && _snap_specs+=("$p")
    done < <(git -C "$REPO_DIR" ls-tree -r --name-only "$sha" -- baselines_wip \
                 | grep -E '\.(py|sh|pbs)$'
             git -C "$REPO_DIR" ls-tree --name-only "$sha" -- data/ \
                 | grep -E '^data/[^/]+\.(py|sh)$')
}

# `git archive` of commit $1's code paths, extracted into $2; $3 non-empty adds configs/,
# $4 non-empty also leaves out __pycache__ (as the rsync mode does). Returns 3 on failure.
_msdelta_snap_archive() {
    local sha=$1 dest=$2 excl=(':(exclude)pbs/logs')
    [[ -n ${4:-} ]] && excl+=(':(exclude,glob)**/__pycache__/**')
    _msdelta_snap_specs "$sha" "${3:-}"
    if (( ${#_snap_specs[@]} == 0 )); then
        echo "code snapshot: commit $sha has none of the code paths" >&2
        return 3
    fi
    if ! git -C "$REPO_DIR" archive --format=tar --output="$dest/.archive.tar" \
             "$sha" -- "${_snap_specs[@]}" "${excl[@]}" \
       || ! tar -x -f "$dest/.archive.tar" -C "$dest"; then
        echo "code snapshot: git archive of $sha failed; refusing to run from a partial copy" >&2
        return 3
    fi
    rm -f "$dest/.archive.tar"
}

# Uncommitted changes (git status --porcelain) in the copied code paths, untracked pbs/logs
# left out. Fails (status 1) when git status does.
_msdelta_snap_status() {
    local out
    out=$(git -C "$REPO_DIR" status --porcelain -- msdelta sweeps pbs tests baselines_wip data \
              pyproject.toml pytest.ini 2>/dev/null) || return 1
    [[ -z $out ]] || printf '%s\n' "$out" | grep -v '^?? pbs/logs' || true
}

# Tests only: MSDELTA_SNAPSHOT_TEST_HOOK is a bash command run at fixed points ($1 = the
# point: clean-resolved, rsync-copied), used to simulate a checkout changing mid-snapshot.
_msdelta_snap_hook() {
    [[ -z ${MSDELTA_SNAPSHOT_TEST_HOOK:-} ]] || bash -c "$MSDELTA_SNAPSHOT_TEST_HOOK" hook "$1"
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
        # The same code paths as the no-ref snapshot, plus configs/ (sweep grids, args files
        # and the generators' templates are committed and must match the code).
        _msdelta_snap_archive "$_snap_sha" "$MSDELTA_CODE_DIR" configs || exit 3
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
    # K111: resolve HEAD FIRST; a clean tree is copied from that commit's objects (git
    # archive), which a concurrent fast-forward of the checkout cannot mix up. HEAD is read
    # again after the status check, so "clean" is known to hold for exactly this sha.
    _snap_sha=$(git -C "$REPO_DIR" rev-parse --verify --quiet 'HEAD^{commit}' 2>/dev/null) \
        || _snap_sha=""
    _snap_branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null) || _snap_branch=""
    _snap_mode=""
    if [[ -n $_snap_sha ]] && _snap_dirty=$(_msdelta_snap_status) && [[ -z $_snap_dirty ]] \
       && [[ $(git -C "$REPO_DIR" rev-parse --verify --quiet 'HEAD^{commit}' 2>/dev/null) \
             == "$_snap_sha" ]]; then
        _msdelta_snap_hook clean-resolved
        _msdelta_snap_archive "$_snap_sha" "$MSDELTA_CODE_DIR" "" no-pycache || exit 3
        _snap_mode="git-archive (clean HEAD)"
    else
        # A dirty tree (or HEAD moved during the check): rsync the working tree while holding
        # the SHARED checkout lock, so pbs/checkout_ff (EXCLUSIVE) cannot fast-forward the
        # checkout mid-copy. The status and diff are recorded under the same lock.
        _snap_lock_fd=""
        _snap_gcd=$(cd "$REPO_DIR" 2>/dev/null && _g=$(git rev-parse --git-common-dir 2>/dev/null) \
                    && cd "$_g" && pwd) || _snap_gcd=""
        if [[ -n $_snap_gcd ]]; then
            _snap_lock=$_snap_gcd/msdelta-checkout.lock
            if ! exec {_snap_lock_fd}>>"$_snap_lock"; then
                echo "code snapshot: cannot open the checkout lock $_snap_lock; refusing to copy" \
                     "a working tree that may be changing" >&2
                exit 3
            fi
            if ! flock -s -w "${MSDELTA_SNAPSHOT_LOCK_TIMEOUT:-600}" "$_snap_lock_fd"; then
                echo "code snapshot: no shared lock on $_snap_lock within" \
                     "${MSDELTA_SNAPSHOT_LOCK_TIMEOUT:-600}s (pbs/checkout_ff holding it?);" \
                     "refusing to copy a working tree that may be changing" >&2
                exit 3
            fi
        fi
        for _snap_try in 1 2; do
            if (( _snap_try == 2 )); then
                rm -rf "$MSDELTA_CODE_DIR"; mkdir -p "$MSDELTA_CODE_DIR"
            fi
            _snap_sha=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null) || _snap_sha=""
            _snap_branch=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null) \
                || _snap_branch=""
            # __pycache__ first: after the /msdelta/*** etc. includes it never matched (before
            # K111 the copy had the checkout's __pycache__ dirs; git archive never has them).
            if ! rsync -a --exclude='__pycache__/' \
                    --include='/msdelta/***' --include='/sweeps/***' --include='/tests/***' \
                    --exclude='/pbs/logs/' --include='/pbs/***' \
                    --include='/baselines_wip/' --include='/baselines_wip/**/' \
                    --include='/baselines_wip/**.py' --include='/baselines_wip/**.sh' \
                    --include='/baselines_wip/**.pbs' \
                    --include='/data/' --include='/data/*.py' --include='/data/*.sh' \
                    --include='/pyproject.toml' --include='/pytest.ini' \
                    --exclude='*' \
                    "$REPO_DIR/" "$MSDELTA_CODE_DIR/"; then
                echo "code snapshot: rsync failed; refusing to run from a partial copy" >&2
                exit 3
            fi
            _msdelta_snap_hook rsync-copied
            _snap_dirty=$(_msdelta_snap_status) || _snap_dirty=""
            _snap_after=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null) || _snap_after=""
            [[ $_snap_after == "$_snap_sha" ]] && break
            echo "code snapshot: HEAD of $REPO_DIR moved from ${_snap_sha:0:12} to" \
                 "${_snap_after:0:12} during the copy (changed without pbs/checkout_ff?)" >&2
            if (( _snap_try == 2 )); then
                echo "code snapshot: HEAD moved twice; refusing to run from a copy that may" \
                     "mix two commits" >&2
                exit 3
            fi
            echo "code snapshot: copying again" >&2
        done
        _snap_mode="working-tree (rsync)"
    fi
    # Every job records the exact commit it was launched from and whether the code it
    # copied differed from it, so any job can be reproduced (or resumed) from that commit.
    # A dirty tree also leaves uncommitted.diff (`git diff HEAD --binary` of the code
    # paths: tracked changes only; untracked files are in the copy itself).
    {
        echo "source:   $REPO_DIR"
        echo "mode:     $_snap_mode"
        echo "branch:   $_snap_branch"
        echo "commit:   $_snap_sha"
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
    if [[ -n ${_snap_lock_fd:-} ]]; then
        exec {_snap_lock_fd}>&-
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
    unset _snap_id _snap_root _snap_pp _snap_parts _p _snap_dirty _snap_sha _snap_branch \
          _snap_mode _snap_lock _snap_lock_fd _snap_gcd _g _snap_try _snap_after _snap_specs
fi
