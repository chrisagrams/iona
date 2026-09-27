"""Moved to `msdelta.rescoring.run_rescoring`. This alias keeps `import msdelta.run_rescoring` (and pickles that name it) and
`python -m msdelta.run_rescoring` working; new code should import `msdelta.rescoring.run_rescoring`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.rescoring.run_rescoring", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.rescoring.run_rescoring")
