"""Moved to `msdelta.finetuning.alignment.precompute_align`. This alias keeps `import msdelta.precompute_align` (and pickles that name it) and
`python -m msdelta.precompute_align` working; new code should import `msdelta.finetuning.alignment.precompute_align`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.finetuning.alignment.precompute_align", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.finetuning.alignment.precompute_align")
