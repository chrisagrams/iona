"""Training entry point, argument dataclasses, callbacks, and W&B setup.

This module is intentionally kept import-light: it does NOT eagerly import
`msdelta.train.cli`, so `from msdelta.train.args import DataArguments` does not drag in
matplotlib, wandb, and the whole `msdelta.eval` package.

Entry points:
    msdelta-train                  → msdelta.train.cli:main
    python -m msdelta.train        → msdelta/train/__main__.py
"""
