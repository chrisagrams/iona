"""Moved to `msdelta.eval.eval_retrieval`. This alias keeps `import msdelta.eval_retrieval` (and pickles that name it) and
`python -m msdelta.eval_retrieval` working; new code should import `msdelta.eval.eval_retrieval`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.eval.eval_retrieval", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.eval.eval_retrieval")
