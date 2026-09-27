"""Moved to `msdelta.rescoring.rerank_psm_handfeat`. This alias keeps `import msdelta.rerank_psm_handfeat` (and pickles that name it) and
`python -m msdelta.rerank_psm_handfeat` working; new code should import `msdelta.rescoring.rerank_psm_handfeat`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.rescoring.rerank_psm_handfeat", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.rescoring.rerank_psm_handfeat")
