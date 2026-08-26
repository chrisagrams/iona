#!/bin/bash
# One-shot venv setup for msdelta on Polaris. Login-node safe: only
# downloads + installs (no GPU work). Re-runnable.
#
# Builds .venv via uv against the pinned uv.lock; the project's
# pyproject pins torch>=2.10,<2.12 from the pytorch-cu128 index, so
# `uv sync` pulls the CUDA-12.8 wheel that matches Polaris's driver.

set -euo pipefail

REPO=/home/cgrams/msdelta
cd "$REPO"

# Keep BLAS thread fan-out small on the shared login node.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# uv handles python provisioning (.python-version → 3.13); --frozen
# guarantees we materialise exactly what's in uv.lock.
uv sync --frozen

echo "==== smoke test ===="
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
# Login node has no GPU — expect False; this is just an import check.
print("cuda available (login node, expect False):", torch.cuda.is_available())

# Confirm msdelta itself imports.
import msdelta.train, msdelta.model, msdelta.data
print("msdelta package imports OK")
PY
