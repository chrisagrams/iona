"""Regenerate the golden references. A DELIBERATE act -- read this first.

    qsub -q debug -l select=1 -l walltime=01:00:00 -A UIC-HPC -l filesystems=home:flare \
         -v REPO_DIR=$PWD,SUITE=regenerate-golden pbs/run_e2e.pbs
    # or, on a compute node:  python -m tests.golden.regenerate [--xpu]

The references in tests/golden/reference/ pin what the CURRENT code computes for frozen
checkpoints on frozen inputs. Regenerating overwrites that pin, so it is right only when
an output change is intended and understood (a deliberate change to pooling, the
processor, the model code) -- never to make a failing golden test pass. Commit the new
reference together with the change that moved it, and say in the message why the
numbers moved.

Writes
  reference/golden.npz    CPU fp32 outputs. Unit-norm embeddings are stored as float16
                          (the comparison adds the float16 rounding to its tolerance);
                          head log-probabilities and metrics stay float32/float64.
  reference/MANIFEST.json checkpoint paths, sha256 of every weight file, input row
                          indices and peptides, code commit, torch version, dtypes,
                          tolerances, and (with --xpu) the measured bf16-on-XPU deviation
                          from the fp32 reference that the bf16 tolerance is set against.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from tests.golden import common

# Absolute tolerances. fp32: CPU recompute against a CPU reference (ulp-level drift
# across CPU models and thread counts). bf16: XPU autocast against the fp32 reference.
TOLERANCES = {
    "fp32": {"embeddings": 1e-4, "head_logprob": 1e-4, "retrieval": 1e-4},
    "bf16": {"embeddings": 1e-2, "head_logprob": 0.25, "retrieval": 5e-2},
}
FLOAT16 = ("pretrained25m/embeddings", "contrastive50m/embeddings", "peptide400m/embeddings")


def deviation(a: dict, b: dict) -> dict:
    out = {}
    for key, ref in a.items():
        if ref.dtype.kind in "fc":
            out[key] = float(np.abs(np.asarray(b[key], np.float64) - ref).max())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xpu", action="store_true",
                    help="also run bf16 autocast on XPU and record its deviation")
    cli = ap.parse_args(argv)
    missing = common.missing_inputs()
    if missing:
        raise SystemExit(f"cannot regenerate, missing: {missing}")
    import torch

    t0 = time.time()
    fp32 = common.compute("cpu", "fp32")
    print(f"[golden] cpu fp32 in {time.time() - t0:.0f}s", flush=True)
    stored = {k: (v.astype(np.float16) if k in FLOAT16 else v) for k, v in fp32.items()}
    common.REFERENCE.mkdir(exist_ok=True)
    np.savez_compressed(common.REFERENCE / "golden.npz", **stored)

    bf16 = None
    if cli.xpu:
        if not torch.xpu.is_available():
            raise SystemExit("--xpu asked for but no XPU is visible")
        t0 = time.time()
        bf16 = deviation(fp32, common.compute("xpu", "bf16"))
        print(f"[golden] xpu bf16 in {time.time() - t0:.0f}s; max |bf16 - fp32|: {bf16}",
              flush=True)

    _, peptides, charges = common.inputs()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "msdelta"],
                                capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent).stdout.strip())
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_commit": commit + ("+dirty(msdelta/)" if dirty else ""),
        "torch": torch.__version__,
        "reference_device": "cpu",
        "reference_dtype": "fp32",
        "stored_float16": list(FLOAT16),
        "checkpoints": {**{k: str(v) for k, v in common.SPECTRUM_MODELS.items()},
                        "peptide400m": common.PEPTIDE_MODEL},
        "peptide400m_snapshot": common.peptide_model_dir().name,
        "sha256": common.weight_hashes(),
        "inputs": {"eval_data": str(common.EVAL_DATA),
                   "row_indices": [0, common.N_ROWS - 1],
                   "head_rows": common.HEAD_ROWS,
                   "peptides": [[p, c] for p, c in zip(peptides, charges)]},
        "pooling": common.POOLING,
        "batch_size": common.BATCH_SIZE,
        "tolerances": TOLERANCES,
        "measured_bf16_xpu_max_abs_deviation": bf16,
        "retrieval_fp32": {name: dict(zip(fp32[f"{name}/retrieval_keys"].tolist(),
                                          fp32[f"{name}/retrieval"].tolist()))
                           for name in common.SPECTRUM_MODELS},
    }
    (common.REFERENCE / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    size = sum(p.stat().st_size for p in common.REFERENCE.iterdir())
    print(f"[golden] wrote {common.REFERENCE} ({size / 1e6:.2f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
