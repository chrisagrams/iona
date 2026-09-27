"""Moved to `msdelta.pretraining.train`. This alias keeps `import msdelta.train` (and pickles that name it) and
`python -m msdelta.train` working; new code should import `msdelta.pretraining.train`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.pretraining.train", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.pretraining.train")
