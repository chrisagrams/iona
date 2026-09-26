"""Moved to `msdelta.eval.eval_denoise_length`. This alias keeps `import msdelta.eval_denoise_length` (and pickles that name it) and
`python -m msdelta.eval_denoise_length` working; new code should import `msdelta.eval.eval_denoise_length`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.eval.eval_denoise_length", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.eval.eval_denoise_length")
