"""msdelta.data: datasets, preprocessing, grouped retrieval data and peptide chemistry.

Also the old flat module `msdelta.data`, now `msdelta.data.data`: attribute lookups that are not
a submodule of this package fall through to it lazily (PEP 562), so `from msdelta.data import X`
keeps working without importing it eagerly.
"""


def __getattr__(name):
    import importlib
    return getattr(importlib.import_module("msdelta.data.data"), name)
