"""Moved to `msdelta.finetuning.denoise.finetune_denoise`. This alias keeps `import msdelta.finetune_denoise` (and pickles that name it) and
`python -m msdelta.finetune_denoise` working; new code should import `msdelta.finetuning.denoise.finetune_denoise`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.finetuning.denoise.finetune_denoise", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.finetuning.denoise.finetune_denoise")
