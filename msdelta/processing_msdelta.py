"""Moved to `msdelta.models.processing_msdelta`. This alias keeps `import msdelta.processing_msdelta` (and pickles that name it) and
`python -m msdelta.processing_msdelta` working; new code should import `msdelta.models.processing_msdelta`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.models.processing_msdelta", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.models.processing_msdelta")
