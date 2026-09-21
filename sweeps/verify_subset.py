"""Check that a hand-picked subset grid still matches the grid it was copied from.

    python sweeps/verify_subset.py configs/sweep-conscale-retry configs/sweep-contrastive-scale

The staleness guard refuses any grid with no registered check, which is correct -- but
some grids are a deliberate subset of another rather than the output of a generator.
This gives those a real check instead of an exemption: every arm must exist in the
parent and be byte-identical to it.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    subset, parent = Path(sys.argv[1]), Path(sys.argv[2])
    arms = sorted(d for d in subset.glob("*") if d.is_dir())
    if not arms:
        print(f"  {subset} holds no arms")
        return 1
    bad = []
    for arm in arms:
        origin = parent / arm.name / "training.args"
        mine = arm / "training.args"
        if not origin.exists():
            bad.append(f"{arm.name} (not in {parent.name})")
        elif origin.read_text() != mine.read_text():
            bad.append(f"{arm.name} (differs from {parent.name})")
    if bad:
        print(f"  {len(bad)} arm(s) out of step: {', '.join(bad[:5])}")
        return 1
    print(f"  {len(arms)} arms match {parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
