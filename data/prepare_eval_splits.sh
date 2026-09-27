#!/bin/bash
# Prepare the ms-contrastive-100k evaluation splits used by the retrieval evaluations
# (msdelta.eval.eval_grouped_retrieval score, msdelta.eval.eval_zeroshot_layers), on a compute node:
#
#   bash data/prepare_eval_splits.sh            # writes $EVAL/ms-contrastive-100k-*-mp512
#
# Each split is flattened to one spectrum per row, preprocessed with the processor of a pretrained
# checkpoint (512 peaks max; only the processor's settings matter, not its weights), and the
# replicate-corpus peptides are excluded (949). Existing outputs are left alone.
set -euo pipefail
S=${SCRATCH_ROOT:-/lus/flare/projects/UIC-HPC/khuss/msdelta}
EVAL=${EVAL:-$S/eval-data}
PROCESSOR=${PROCESSOR:-/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-50m-production-01-checkpoint-220000}
PY=${PY:-.venv/bin/python}
prep() {   # out split [extra args]
    local out=$EVAL/$1; shift
    [[ -d $out ]] && { echo "exists: $out"; return; }
    $PY -m msdelta.eval.eval_grouped_retrieval prepare --out-data "$out" --processor "$PROCESSOR" --split "$@"
}
prep ms-contrastive-100k-test-mp512 test                      # 35,734 rows
prep ms-contrastive-100k-validation-mp512 validation          # 35,654 rows
prep ms-contrastive-100k-train10k-mp512 train --max-analytes 10000   # 35,731 rows; the ABTT fit sample
# CPU-sized validation subset (whole groups, >= 5,000 experimental spectra):
[[ -d $EVAL/ms-contrastive-100k-validation-mp512-sub5k ]] || \
    $PY data/subsample_prepared.py $EVAL/ms-contrastive-100k-validation-mp512 \
        $EVAL/ms-contrastive-100k-validation-mp512-sub5k 5000
