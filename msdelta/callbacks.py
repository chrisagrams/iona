"""Moved to `msdelta.utils.callbacks`. This alias keeps `import msdelta.callbacks` (and pickles that name it) and
`python -m msdelta.callbacks` working; new code should import `msdelta.utils.callbacks`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.utils.callbacks", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.utils.callbacks")
