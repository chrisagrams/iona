"""The package reorganisation must not break a single old import path or `python -m` line.

Every flat module msdelta/<old>.py is now a shim that replaces itself in sys.modules with
its new home (msdelta/{models,data,pretraining,finetuning/...,eval,rescoring,utils}/...),
and runs the new module's main under `python -m`. Configs, PBS scripts and notebooks all
use the old names, so these tests pin, for EVERY shim:

  (a) `import msdelta.<old>` is the very same module object as the new path, so module
      state (and monkeypatching in tests) is shared, not duplicated;
  (b) `from msdelta.<old> import X` works for a public name defined in the new module;
  (c) `python -m msdelta.<old> --help` reaches the new module's argument parser -- for the
      entry points the PBS scripts call;
  (d) the PEP 562 fall-through of the msdelta.data and msdelta.rescoring packages to the
      old flat msdelta.data / msdelta.rescoring modules;
  (e) every module under msdelta/ imports with faiss absent: faiss is not in the Aurora
      venv, and only msdelta.eval.retrieval.retrieval_metrics needs it (lazy import).
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import re
import subprocess
import sys

import pytest

from tests.conftest import REPO

SHIM = re.compile(r'_sys\.modules\[__name__\]\s*=\s*_importlib\.import_module\("([\w.]+)"\)')


def _shims() -> dict[str, str]:
    """old module name -> new module name, read from the shim files themselves."""
    out = {}
    for path in sorted((REPO / "msdelta").glob("*.py")):
        match = SHIM.search(path.read_text())
        if match:
            out[f"msdelta.{path.stem}"] = match.group(1)
    return out


SHIMS = _shims()


def test_shims_were_found():
    """Guards the regex: if it silently matched nothing every test below would vanish."""
    assert len(SHIMS) >= 30
    assert SHIMS["msdelta.contrastive"] == "msdelta.finetuning.contrastive.contrastive"
    assert SHIMS["msdelta.train"] == "msdelta.pretraining.train"


def test_no_flat_module_is_left_unshimmed():
    """A flat module that is not a shim is a second copy of code that has moved."""
    flat = {f"msdelta.{p.stem}" for p in (REPO / "msdelta").glob("*.py")
            if p.stem != "__init__"}
    assert flat == set(SHIMS)


@pytest.mark.parametrize("old,new", sorted(SHIMS.items()))
def test_old_import_is_the_new_module(old, new):
    old_module = importlib.import_module(old)
    new_module = importlib.import_module(new)
    assert old_module is new_module
    assert sys.modules[old] is new_module


def _public_name(module) -> str:
    """A public function or class DEFINED in the module (not merely imported by it)."""
    for name, obj in sorted(vars(module).items()):
        if (not name.startswith("_") and (inspect.isfunction(obj) or inspect.isclass(obj))
                and getattr(obj, "__module__", None) == module.__name__):
            return name
    for name in sorted(vars(module)):                 # constants-only module
        if name.isupper():
            return name
    raise AssertionError(f"{module.__name__} defines nothing public")


@pytest.mark.parametrize("old,new", sorted(SHIMS.items()))
def test_from_old_import_name(old, new):
    name = _public_name(importlib.import_module(new))
    namespace: dict = {}
    exec(f"from {old} import {name}", namespace)
    assert namespace[name] is getattr(importlib.import_module(new), name)


# The entry points the PBS scripts and sweeps run as `python -m msdelta.<old>`.
ENTRY_POINTS = [
    "msdelta.train",
    "msdelta.finetune_contrastive",
    "msdelta.finetune_denoise",
    "msdelta.finetune_align",
    "msdelta.precompute_align",
    "msdelta.eval_grouped_retrieval",
    "msdelta.eval_zeroshot_layers",
    "msdelta.psm_rerank",
    "msdelta.rerank_psm_fdr",
]


def _no_faiss_env() -> dict:
    """Run a child with faiss unimportable, as on Aurora, even where it is installed."""
    blocker = REPO / "tests" / "_nofaiss"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(blocker), str(REPO), env.get("PYTHONPATH", "")])
    return env


# ONE child interpreter does (c) and (e): every interpreter start pays torch and
# transformers (~10-30 s on a login node), so a child per entry point or per module would
# make this file the slowest in the suite. The child first runs each entry point the way
# `python -m` does -- runpy on the OLD name as __main__, so the shim's own __main__ branch
# is what dispatches -- then imports every module. faiss is blocked throughout.
CHILD = r"""
import contextlib, importlib, io, json, runpy, sys, traceback

entry_points, modules = json.loads(sys.argv[1])
helps = {}
for old in entry_points:
    # A shim imported earlier in this loop would be replaced in sys.modules by its new
    # module, and runpy would then find the new module's spec and skip the shim.
    sys.modules.pop(old, None)
    out, argv = io.StringIO(), sys.argv
    sys.argv = [old, "--help"]
    code = None
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_module(old, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        code = exc.code
    except BaseException:
        code = "raised: " + traceback.format_exc()[-1500:]
    finally:
        sys.argv = argv
    helps[old] = {"code": code, "stdout": out.getvalue()[:1500]}   # usage line is first

failed = {}
for name in modules:
    try:
        importlib.import_module(name)
    except BaseException:
        failed[name] = traceback.format_exc()[-1500:]
faiss_loaded = sys.modules.get("faiss") is not None
print("@@RESULT@@" + json.dumps({"helps": helps, "failed": failed, "faiss": faiss_loaded}))
"""


def _all_modules() -> list[str]:
    import msdelta
    return sorted(m.name for m in pkgutil.walk_packages(msdelta.__path__, "msdelta."))


@pytest.fixture(scope="module")
def child():
    import json
    payload = json.dumps([ENTRY_POINTS, _all_modules()])
    result = subprocess.run([sys.executable, "-c", CHILD, payload], cwd=REPO,
                            env=_no_faiss_env(), capture_output=True, text=True, timeout=600)
    marker = [l for l in result.stdout.splitlines() if l.startswith("@@RESULT@@")]
    assert marker, f"child died (rc {result.returncode}):\n{result.stderr[-3000:]}"
    return json.loads(marker[-1][len("@@RESULT@@"):])


@pytest.mark.parametrize("old", ENTRY_POINTS)
def test_python_m_old_name_runs_the_new_main(child, old):
    assert old in SHIMS, f"{old} is not a shim any more; update ENTRY_POINTS"
    run = child["helps"][old]
    assert run["code"] in (0, None), run["code"]
    assert "usage" in run["stdout"].lower(), run["stdout"]


def test_python_m_really_works_from_a_shell():
    """The child above goes through runpy by hand; one real `python -m` proves the same
    path from a command line (psm_rerank: the cheapest entry point to start)."""
    result = subprocess.run([sys.executable, "-m", "msdelta.psm_rerank", "--help"], cwd=REPO,
                            env=_no_faiss_env(), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage" in result.stdout.lower()
    assert "{score,train}" in result.stdout


def test_data_package_falls_through_to_the_old_flat_module():
    import msdelta.data
    flat = importlib.import_module("msdelta.data.data")
    name = _public_name(flat)
    assert getattr(msdelta.data, name) is getattr(flat, name)
    namespace: dict = {}
    exec(f"from msdelta.data import {name}", namespace)
    assert namespace[name] is getattr(flat, name)
    # submodules still resolve as submodules, not through the fall-through
    assert importlib.import_module("msdelta.data.grouped_retrieval").__name__ == \
        "msdelta.data.grouped_retrieval"
    with pytest.raises(AttributeError):
        msdelta.data.no_such_attribute_anywhere


def test_rescoring_package_falls_through_to_the_old_flat_module():
    import msdelta.rescoring
    flat = importlib.import_module("msdelta.rescoring.rescoring")
    for name in ("FEATURE_NAMES", "build_candidates", "extract_features", "train_rescorer"):
        assert getattr(msdelta.rescoring, name) is getattr(flat, name)
    with pytest.raises(AttributeError):
        msdelta.rescoring.no_such_attribute_anywhere


def test_every_module_imports_without_faiss(child):
    assert "msdelta.eval.retrieval" in _all_modules()
    assert not child["failed"], "\n\n".join(f"{k}:\n{v}" for k, v in child["failed"].items())
    assert not child["faiss"], "something imported faiss (the blocker should have refused)"


def test_faiss_is_imported_only_inside_retrieval_metrics():
    source = (REPO / "msdelta" / "eval" / "retrieval.py").read_text()
    assert not re.search(r"^(import faiss|from faiss)", source, re.MULTILINE)
    from msdelta.eval.retrieval import retrieval_metrics
    assert "import faiss" in inspect.getsource(retrieval_metrics)
