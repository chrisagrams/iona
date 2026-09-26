"""Moved to `msdelta.pretraining.training_args`. This alias keeps `import msdelta.training_args` (and pickles that name it) and
`python -m msdelta.training_args` working; new code should import `msdelta.pretraining.training_args`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.pretraining.training_args", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.pretraining.training_args")
