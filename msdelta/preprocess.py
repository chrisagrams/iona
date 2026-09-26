"""Moved to `msdelta.data.preprocess`. This alias keeps `import msdelta.preprocess` (and pickles that name it) and
`python -m msdelta.preprocess` working; new code should import `msdelta.data.preprocess`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.data.preprocess", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.data.preprocess")
