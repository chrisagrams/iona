"""Moved to `msdelta.models.embedding`. This alias keeps `import msdelta.embedding` (and pickles that name it) and
`python -m msdelta.embedding` working; new code should import `msdelta.models.embedding`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.models.embedding", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.models.embedding")
