"""Resolve a model class from a dotted path.

Lets an architecture experiment live in its own module and be selected from a training
args file, so trying a new architecture needs no edit to shared training code.
"""

from __future__ import annotations

import importlib

from msdelta.model.modeling import MSDeltaForPreTraining


def load_model_class(spec: str | None) -> type:
    """Return ``MSDeltaForPreTraining``, or the class named by ``pkg.module.Class``."""
    if not spec:
        return MSDeltaForPreTraining
    module_name, _, class_name = spec.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"model_class must be a dotted path like 'pkg.module.Class', got {spec!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as error:
        raise ValueError(f"module '{module_name}' has no attribute '{class_name}'") from error
