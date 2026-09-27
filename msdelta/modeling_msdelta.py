"""Moved to `msdelta.models.modeling_msdelta`. This alias keeps `import msdelta.modeling_msdelta` (and pickles that name it) and
`python -m msdelta.modeling_msdelta` working; new code should import `msdelta.models.modeling_msdelta`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.models.modeling_msdelta", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.models.modeling_msdelta")
