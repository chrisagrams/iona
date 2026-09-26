"""Moved to `msdelta.utils.wandb_distributed`. This alias keeps `import msdelta.wandb_distributed` (and pickles that name it) and
`python -m msdelta.wandb_distributed` working; new code should import `msdelta.utils.wandb_distributed`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.utils.wandb_distributed", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.utils.wandb_distributed")
