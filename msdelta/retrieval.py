"""Moved to `msdelta.eval.retrieval`. This alias keeps `import msdelta.retrieval` (and pickles that name it) and
`python -m msdelta.retrieval` working; new code should import `msdelta.eval.retrieval`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.eval.retrieval", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.eval.retrieval")
