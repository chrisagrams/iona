"""msdelta.rescoring: PSM rescoring and reranking with the embeddings.

Also the old flat module `msdelta.rescoring`, now `msdelta.rescoring.rescoring`: attribute lookups that are not
a submodule of this package fall through to it lazily (PEP 562), so `from msdelta.rescoring import X`
keeps working without importing it eagerly.
"""


def __getattr__(name):
    import importlib
    return getattr(importlib.import_module("msdelta.rescoring.rescoring"), name)
