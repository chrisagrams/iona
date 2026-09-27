"""Moved to `msdelta.finetuning.contrastive.finetune_contrastive`. This alias keeps `import msdelta.finetune_contrastive` (and pickles that name it) and
`python -m msdelta.finetune_contrastive` working; new code should import `msdelta.finetuning.contrastive.finetune_contrastive`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.finetuning.contrastive.finetune_contrastive", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.finetuning.contrastive.finetune_contrastive")
