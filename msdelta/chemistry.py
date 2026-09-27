"""Moved to `msdelta.data.chemistry`. This alias keeps `import msdelta.chemistry` (and pickles that name it) and
`python -m msdelta.chemistry` working; new code should import `msdelta.data.chemistry`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.data.chemistry", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.data.chemistry")
