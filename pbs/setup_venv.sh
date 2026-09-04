#!/bin/bash
# Install the locked environment on a Polaris login node.

set -euo pipefail

REPO=/home/cgrams/msdelta
cd "$REPO"

# Limit the BLAS threads.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv sync --frozen

echo "==== smoke test ===="
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
# A login node does not have a GPU.
print("cuda available (login node, expect False):", torch.cuda.is_available())

import msdelta
import msdelta.data, msdelta.eval, msdelta.model, msdelta.train.cli
print("msdelta package imports OK")
PY
