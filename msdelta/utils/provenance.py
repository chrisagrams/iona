"""Which code produced an output: the job's code snapshot (pbs/lib/code_snapshot.sh).

`code_provenance()` reads $MSDELTA_CODE_DIR/SNAPSHOT.txt and returns its `commit`,
`branch`, `ref`, `mode` and `dirty` fields (mode is `git-archive` when the job ran a commit:
via pbs/qsub_ref (with `ref`, dirty then "n/a"), or the checkout's clean HEAD (K111, no `ref`,
dirty False); `working-tree` when it ran the checkout's dirty working tree (rsync), dirty then
whether that tree had uncommitted code changes). Empty dict outside a snapshot.
"""

from __future__ import annotations

import os
from pathlib import Path


def code_provenance() -> dict:
    code_dir = os.environ.get("MSDELTA_CODE_DIR")
    if not code_dir:
        return {}
    snap = Path(code_dir, "SNAPSHOT.txt")
    if not snap.is_file():
        return {"snapshot": code_dir}
    fields, dirty_lines, in_dirty = {}, [], False
    for line in snap.read_text().splitlines():
        if in_dirty:
            if line.strip():
                dirty_lines.append(line.strip())
            continue
        if line.startswith("uncommitted changes"):
            in_dirty = True
            continue
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    out = {"snapshot": code_dir,
           "mode": fields.get("mode", "working-tree").split()[0],
           "commit": fields.get("commit", ""),
           "branch": fields.get("branch", "")}
    if "ref" in fields:
        out["ref"] = fields["ref"]
    # A git-archive snapshot has no working tree to be dirty: "n/a". Older snapshots have
    # no `dirty:` line, only the list of changes.
    flag = fields.get("dirty", "").split()[:1]
    if flag == ["n/a"]:
        out["dirty"] = "n/a"
    elif flag:
        out["dirty"] = flag[0] == "yes"
    else:
        out["dirty"] = bool(dirty_lines)
    return out
