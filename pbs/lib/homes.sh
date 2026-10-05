# Export the data homes (configs/homes.env) to a job or shell:   source "$REPO_DIR/pbs/lib/homes.sh"
# Values already set in the environment win (e.g. a test pointing MSDELTA_EVAL at a scratch copy).
# A code snapshot (pbs/lib/code_snapshot.sh) has no configs/: the file is then read from the checkout ($REPO_DIR), or,
# with neither, the values must already be exported (a job sources this from the checkout before it snapshots).
_homes_env="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/configs/homes.env"
[[ -f $_homes_env ]] || _homes_env="${REPO_DIR:-/nonexistent}/configs/homes.env"
if [[ -f $_homes_env ]]; then
    while IFS='=' read -r _k _v; do
        [[ $_k =~ ^MSDELTA_[A-Z_]+$ ]] || continue
        [[ -n ${!_k:-} ]] || eval "export $_k=\"$_v\""
    done < "$_homes_env"
    export MSDELTA_RESULTS="${MSDELTA_RESULTS:-$(dirname "$(dirname "$_homes_env")")/results}"
elif [[ -z ${MSDELTA_STORAGE:-} || -z ${MSDELTA_RUNS:-} || -z ${MSDELTA_EVAL:-} || -z ${MSDELTA_DIAG:-} \
        || -z ${MSDELTA_DERIVED:-} || -z ${MSDELTA_RESULTS:-} ]]; then
    echo "homes.sh: configs/homes.env not found (looked next to ${BASH_SOURCE[0]} and in REPO_DIR=${REPO_DIR:-unset})" \
         "and the MSDELTA_* homes are not all exported" >&2
    unset _homes_env
    return 1 2>/dev/null || exit 1
fi
unset _homes_env _k _v
