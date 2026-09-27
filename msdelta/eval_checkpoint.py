"""Moved to `msdelta.eval.eval_checkpoint`. This alias keeps `import msdelta.eval_checkpoint` (and pickles that name it) and
`python -m msdelta.eval_checkpoint` working; new code should import `msdelta.eval.eval_checkpoint`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.eval.eval_checkpoint", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.eval.eval_checkpoint")
