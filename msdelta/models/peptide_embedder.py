"""Renamed to `msdelta.models.peptide_encoder` (2026-09-27: "peptide embedder" -> "peptide encoder"). This alias keeps
`import msdelta.models.peptide_embedder` (and pickles that name it), PeptideEmbedderConfig / PeptideEmbedderModel and
`python -m msdelta.models.peptide_embedder` working; new code should import `msdelta.models.peptide_encoder`."""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("msdelta.models.peptide_encoder", run_name="__main__", alter_sys=True)
else:
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module("msdelta.models.peptide_encoder")
