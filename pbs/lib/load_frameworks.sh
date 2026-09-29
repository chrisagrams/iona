# Load the Python/oneAPI environment our .venv was built on, on the post-2026-09 Aurora image.
#
# Aurora's 2026-09 major update (https://docs.alcf.anl.gov/aurora/system-updates/#major-update-2026-09)
# made PE 26.181.0 (oneAPI 2026.1, frameworks/2026.1.0, PyTorch 2.13) the default and kept PE 26.26.0
# (oneAPI 2025.3.1) "recompiled for SLES 15 SP7". Our .venv is built on frameworks/2025.3.1
# (PyTorch 2.10); `module load frameworks/2025.3.1` on top of the NEW default fails (its pti-gpu
# dependency resolves to the 2026.1 stack), and a bare `module load frameworks` now silently picks
# 2026.1.0. So: purge, point MODULEPATH at the 26.26.0 trees only, then load oneAPI 2025.3.1, its
# MPICH, frameworks/2025.3.1 and Cray PALS (mpiexec). Verified on the login node 2026-09-29.
#
# Usage (in any PBS script, after REPO_DIR is known):   source "$REPO_DIR/pbs/lib/load_frameworks.sh"
# Override: FRAMEWORKS_MODULE=frameworks/2026.1.0 loads the NEW default stack instead (needs a venv
# built for it -- ours is not).
_lf_had_u=0; [[ $- == *u* ]] && _lf_had_u=1; set +u
if ! type module >/dev/null 2>&1; then source /usr/share/lmod/lmod/init/bash; fi
_lf_mod=${FRAMEWORKS_MODULE:-frameworks/2025.3.1}
if [[ $_lf_mod == frameworks/2025.3.1 ]]; then
    module --force purge >/dev/null 2>&1
    export MODULEPATH=/usr/share/lmod/modulefiles/Linux:/usr/share/lmod/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core:/opt/cray/pals/lmod/modulefiles/core:/opt/cray/modulefiles:/opt/aurora/26.26.0/modulefiles
    module load oneapi/release/2025.3.1 >/dev/null 2>&1 \
        && module load mpich/prd/5.0.0.aurora_test.87e2045 >/dev/null 2>&1 \
        && module load frameworks/2025.3.1 >/dev/null 2>&1 \
        && module load cray-pals cray-libpals >/dev/null 2>&1
    _lf_rc=$?
else
    module load "$_lf_mod" >/dev/null 2>&1; _lf_rc=$?
fi
if (( _lf_rc != 0 )) || ! module is-loaded "${_lf_mod}" 2>/dev/null; then
    echo "load_frameworks.sh: FAILED to load ${_lf_mod} (rc=${_lf_rc}); loaded: $(module -t list 2>&1 | tr '\n' ' ')" >&2
    (( _lf_had_u )) && set -u
    return 3 2>/dev/null || exit 3
fi
echo "=== environment: ${_lf_mod} ($(module -t list 2>&1 | grep -E '^(oneapi|mpich|frameworks)' | tr '\n' ' '))"
(( _lf_had_u )) && set -u
unset _lf_had_u _lf_mod _lf_rc
