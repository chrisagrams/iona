"""Moved to `msdelta.finetuning.alignment.alignment`. This alias keeps `import msdelta.alignment` (and pickles that name it) and
`python -m msdelta.alignment` working; new code should import `msdelta.finetuning.alignment.alignment`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.finetuning.alignment.alignment", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.finetuning.alignment.alignment")
