"""Every job script loads the Python environment through pbs/lib/load_frameworks.sh (2026-09 Aurora update).

After the update a bare `module load frameworks` picks frameworks/2026.1.0 (PyTorch 2.13; our .venv is
built on 2025.3.1) and `module load frameworks/2025.3.1` fails on top of the new default PE, so no script
may load the frameworks module directly."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "pbs" / "lib" / "load_frameworks.sh"


def _scripts():
    for p in ROOT.rglob("*"):
        if p.suffix in (".pbs", ".sh") and p.is_file() and ".venv" not in p.parts and p != HELPER:
            yield p


def test_helper_exists_and_pins_the_old_stack():
    t = HELPER.read_text()
    assert "oneapi/release/2025.3.1" in t and "frameworks/2025.3.1" in t and "/opt/aurora/26.26.0/modulefiles" in t


def test_no_script_loads_frameworks_directly():
    offenders = [str(p.relative_to(ROOT)) for p in _scripts()
                 if re.search(r"^[^#\n]*module load[^\n]*frameworks", p.read_text(), re.M)]
    assert not offenders, f"load the environment via pbs/lib/load_frameworks.sh instead: {offenders}"
