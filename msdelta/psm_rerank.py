"""Moved to `msdelta.rescoring.psm_rerank`. This alias keeps `import msdelta.psm_rerank` (and pickles that name it) and
`python -m msdelta.psm_rerank` working; new code should import `msdelta.rescoring.psm_rerank`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.rescoring.psm_rerank", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.rescoring.psm_rerank")
