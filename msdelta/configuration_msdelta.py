"""Moved to `msdelta.models.configuration_msdelta`. This alias keeps `import msdelta.configuration_msdelta` (and pickles that name it) and
`python -m msdelta.configuration_msdelta` working; new code should import `msdelta.models.configuration_msdelta`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.models.configuration_msdelta", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.models.configuration_msdelta")
