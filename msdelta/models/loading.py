"""Load a checkpoint and REFUSE it unless its weights match the model being built (K94-P).

`PreTrainedModel.from_pretrained` only warns when checkpoint keys are missing or unexpected:
the missing ones are left at their random initialisation and loading carries on. That is how
a Pairformer checkpoint (config.architecture = "pairformer") read by code that only knows the
transformer produced a mostly random transformer (41 missing / 143 unexpected keys) that was
then evaluated as if trained. Every place our fine-tuning / evaluation code loads a model from
a path goes through `load_strict`, which turns that warning into an error.

Allow-lists are per call site and name exactly the keys a site legitimately does not get from
the checkpoint (fnmatch patterns, matched against the full key). Keep them narrow: an
allow-list that matches encoder keys defeats the point.
"""

from __future__ import annotations

import inspect
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Iterable, TypeVar

# Encoder architectures this code can build. A config that names anything else (the
# `architecture` field added on the Pairformer branch) is refused before its weights are
# trusted. "transformer" is also what an absent field means.
SUPPORTED_ARCHITECTURES = frozenset({"transformer"})

M = TypeVar("M")


class CheckpointMismatchError(RuntimeError):
    """A checkpoint's weights or config do not match the model this code builds."""


def _unexpected_architecture(config) -> str | None:
    """The config's `architecture` if this code cannot build it, else None.

    Nested encoder configs (MSDeltaDenoisingConfig.encoder, MSDeltaRetrievalConfig.encoder)
    are checked too. When the config class itself declares `architecture` as a constructor
    argument (the Pairformer branch), that class validates the value and builds the matching
    model, so it is trusted here; only an UNDECLARED field -- one this code's config class
    merely carried along from config.json -- is refused unless it says "transformer".
    """
    for cfg in (config, getattr(config, "encoder", None)):
        if cfg is None or not hasattr(cfg, "architecture"):
            continue
        value = getattr(cfg, "architecture")
        if value is None or value in SUPPORTED_ARCHITECTURES:
            continue
        if "architecture" in inspect.signature(type(cfg).__init__).parameters:
            continue
        return value
    return None


def check_architecture(config, source: str = "checkpoint") -> None:
    """Raise if `config` names an encoder architecture this code does not build."""
    value = _unexpected_architecture(config)
    if value is not None:
        raise CheckpointMismatchError(
            f"{source}: config.architecture={value!r}, but this code only builds "
            f"{sorted(SUPPORTED_ARCHITECTURES)}. Loading it would construct the wrong "
            f"encoder and leave most weights randomly initialised. Use the code that "
            f"trained it (e.g. the p1-pairformer branch for 'pairformer').")


def _filter(keys: Iterable[str], allowed: Iterable[str]) -> list[str]:
    allowed = tuple(allowed)
    return [k for k in keys if not any(fnmatchcase(k, pattern) for pattern in allowed)]


def check_keys(missing: Iterable[str] = (), unexpected: Iterable[str] = (),
               mismatched: Iterable = (), *, allow_missing: Iterable[str] = (),
               allow_unexpected: Iterable[str] = (), source: str = "checkpoint",
               error: type[Exception] = CheckpointMismatchError) -> None:
    """Raise `error` listing every missing / unexpected / shape-mismatched key not allowed.

    Shared by `load_strict` and by callers that load a raw state dict themselves
    (eval_checkpoint.load_weights) so the rule and the message are the same everywhere.
    """
    missing = _filter(missing, allow_missing)
    unexpected = _filter(unexpected, allow_unexpected)
    mismatched = list(mismatched)
    if not (missing or unexpected or mismatched):
        return
    lines = [f"{source}: checkpoint does not match the model being built -- refusing to "
             f"continue with randomly initialised weights"]
    for label, keys in (("missing", missing), ("unexpected", unexpected),
                        ("mismatched", mismatched)):
        if keys:
            shown = ", ".join(str(k) for k in keys[:12])
            more = f" ... (+{len(keys) - 12} more)" if len(keys) > 12 else ""
            lines.append(f"  {label} ({len(keys)}): {shown}{more}")
    raise error("\n".join(lines))


def load_strict(cls: type[M], path, *, allow_missing: Iterable[str] = (),
                allow_unexpected: Iterable[str] = (), **kwargs) -> M:
    """`cls.from_pretrained(path, **kwargs)`, failing loudly on any weight/config mismatch.

    allow_missing / allow_unexpected: fnmatch patterns for keys this call site legitimately
    does not receive from / does not use from the checkpoint. Everything else must match.
    """
    kwargs.pop("output_loading_info", None)
    source = str(path)
    # Refuse a foreign architecture BEFORE building anything: the wrong encoder may not even
    # fail to build, it just gets the wrong weights.
    config = kwargs.get("config")
    config_class = getattr(cls, "config_class", None)
    if config is None and config_class is not None and (Path(str(path)) / "config.json").is_file():
        try:
            config = config_class.from_pretrained(path)
        except Exception:        # an older layout the model's own loader converts; checked below
            config = None
    if config is not None and not isinstance(config, (str, Path)):
        check_architecture(config, source)
    loaded = cls.from_pretrained(path, output_loading_info=True, **kwargs)
    if not (isinstance(loaded, tuple) and len(loaded) == 2 and isinstance(loaded[1], dict)):
        raise TypeError(f"{cls.__name__}.from_pretrained did not return loading info; "
                        f"cannot verify {source}")
    model, info = loaded
    check_architecture(model.config, source)
    errors = [e for e in info.get("error_msgs") or [] if e]
    if errors:
        raise CheckpointMismatchError(f"{source}: errors while loading: {errors[:5]}")
    check_keys(info.get("missing_keys") or (), info.get("unexpected_keys") or (),
               info.get("mismatched_keys") or (), allow_missing=allow_missing,
               allow_unexpected=allow_unexpected, source=source)
    return model
