"""Moved to `msdelta.pretraining.posttraining`. This alias keeps `import msdelta.posttraining` (and pickles that name it) and
`python -m msdelta.posttraining` working; new code should import `msdelta.pretraining.posttraining`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.pretraining.posttraining", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.pretraining.posttraining")
