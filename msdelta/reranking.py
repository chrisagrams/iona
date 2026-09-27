"""Moved to `msdelta.rescoring.reranking`. This alias keeps `import msdelta.reranking` (and pickles that name it) and
`python -m msdelta.reranking` working; new code should import `msdelta.rescoring.reranking`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.rescoring.reranking", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.rescoring.reranking")
