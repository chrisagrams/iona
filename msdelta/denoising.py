"""Moved to `msdelta.finetuning.denoise.denoising`. This alias keeps `import msdelta.denoising` (and pickles that name it) and
`python -m msdelta.denoising` working; new code should import `msdelta.finetuning.denoise.denoising`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.finetuning.denoise.denoising", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.finetuning.denoise.denoising")
