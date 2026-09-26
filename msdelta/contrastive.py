"""Moved to `msdelta.finetuning.contrastive.contrastive`. This alias keeps `import msdelta.contrastive` (and pickles that name it) and
`python -m msdelta.contrastive` working; new code should import `msdelta.finetuning.contrastive.contrastive`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.finetuning.contrastive.contrastive", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.finetuning.contrastive.contrastive")
