"""Moved to `msdelta.utils.viz`. This alias keeps `import msdelta.viz` (and pickles that name it) and
`python -m msdelta.viz` working; new code should import `msdelta.utils.viz`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.utils.viz", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.utils.viz")
