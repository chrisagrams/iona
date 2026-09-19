# Shared credential loader. Source it; do not execute it.
#
#     source "$REPO_DIR/pbs/load_keys.sh"
#
# Exports HF_TOKEN and WANDB_API_KEY from the first key store that has them, and prints
# only whether each was found -- never the value.
#
# Search order, first hit wins:
#     $REPO_DIR/.keys/{hf,wandb}      repo-local store (gitignored)
#     ~/.keys/{hf,wandb}              per-user store
#     ~/.hf_token                     single-file fallback
#
# Two rules this exists to enforce:
#
#  1. Secrets are never passed through `qsub -v`. Everything in a job's variable list is
#     visible to anyone who can run `qstat -f <jobid>`. Jobs read the files at runtime
#     instead -- $HOME is mounted on Aurora compute nodes, so this works from inside a job.
#  2. Secrets are never passed on a command line. Process arguments are readable via `ps`
#     on a shared login node. Everything here goes through the environment or a file.

_msdelta_read_key() {
    local name=$1 path
    for path in "${REPO_DIR:-$PWD}/.keys/$name" "$HOME/.keys/$name"; do
        if [[ -r $path ]]; then
            tr -d '\r\n' <"$path"
            return 0
        fi
    done
    if [[ $name == hf && -r $HOME/.hf_token ]]; then
        tr -d '\r\n' <"$HOME/.hf_token"
        return 0
    fi
    return 1
}

_msdelta_check_perms() {
    local path=$1 mode
    [[ -e $path ]] || return 0
    mode=$(stat -c '%a' "$path" 2>/dev/null) || return 0
    if [[ $mode != 600 && $mode != 400 && $mode != 700 ]]; then
        echo "WARNING: $path is mode $mode -- tighten with: chmod 600 $path" >&2
    fi
}

for _p in "${REPO_DIR:-$PWD}/.keys/hf" "$HOME/.keys/hf" "$HOME/.keys/wandb" "$HOME/.hf_token"; do
    _msdelta_check_perms "$_p"
done
unset _p

if [[ -z ${HF_TOKEN:-} ]]; then
    if HF_TOKEN=$(_msdelta_read_key hf); then
        export HF_TOKEN
        echo "keys   : HF_TOKEN loaded (${#HF_TOKEN} chars)"
    else
        unset HF_TOKEN
        echo "keys   : no HF token found -- gated datasets will 401" >&2
    fi
fi

if [[ -z ${WANDB_API_KEY:-} ]]; then
    if WANDB_API_KEY=$(_msdelta_read_key wandb); then
        export WANDB_API_KEY
        echo "keys   : WANDB_API_KEY loaded (${#WANDB_API_KEY} chars)"
    else
        unset WANDB_API_KEY
        echo "keys   : no W&B key found -- runs will not report to CS_Pharm" >&2
    fi
fi
