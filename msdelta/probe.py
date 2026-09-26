"""Moved to `msdelta.pretraining.probe`. This alias keeps `import msdelta.probe` (and pickles that name it) and
`python -m msdelta.probe` working; new code should import `msdelta.pretraining.probe`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.pretraining.probe", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.pretraining.probe")
