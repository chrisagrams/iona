"""Stand-in on the PYTHONPATH of tests/test_imports.py's child interpreters: importing faiss
fails exactly as it does on Aurora, where faiss is not installed, even on a machine where
it is. Not a test module (pytest collects only test_*.py)."""

raise ModuleNotFoundError("No module named 'faiss' (blocked by tests/_nofaiss)", name="faiss")
