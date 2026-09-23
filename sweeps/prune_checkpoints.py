"""Delete intermediate checkpoints from COMPLETED DENOISE runs. Dry run by default.

    python sweeps/prune_checkpoints.py                  # dry run, shows what would go
    python sweeps/prune_checkpoints.py --limit 5        # dry run, first 5 runs only
    python sweeps/prune_checkpoints.py --limit 5 --execute

CONTRASTIVE (--task contrastive), added once it was measured rather than assumed.
Contrastive trains with no best-model selection, so `final/` is the LAST step, and
deleting its checkpoints was held until that was shown not to matter. It does not:
job 8856348 scored checkpoint-1200, checkpoint-1347 and final/ on three seeds -- final/
equals the last checkpoint exactly and 1200 is ~0.003 MAP@R lower -- and job 8856159
found selecting the best checkpoint by MAP@R changes held-out MAP@R by -0.002 and
-0.006, both noise. Only the last two checkpoints survived save_total_limit anyway, so
no training-dynamics study was possible from them.

WHY DENOISE WAS FIRST. Denoise trains with load_best_model_at_end on eval_auprc, so `final/`
holds the BEST checkpoint, not the last one, and the intermediates are strictly
redundant. Contrastive trains with eval_strategy=no: `final/` there is whatever the last
step produced, and until that is shown to be the best its checkpoints must not be
touched.

WHAT IS LOST. Four paths read intermediate checkpoints -- RESUME_JOB in
aurora-finetune-sweep.pbs, recover_test_metrics, backfill_per_spectrum, and
eval_checkpoint on a named step. All of them are RECOVERY paths for runs that did not
finish. A run holding both final/ and all_results.json has nothing left to recover, so
for that run they are dead weight. That is the whole safety argument.

EVERY predicate below must hold. Any one failing skips the run:
  1. under the runs root, resolved, no symlink escape
  2. all_results.json parses and carries test_auroc  -> a COMPLETED DENOISE run
  3. final/ exists and holds real weights           -> the model we keep is really there
  4. the run's job id is not queued or running      -> nothing is writing to it
  5. the target is a directory literally named checkpoint-<digits>
  6. nothing directly under the run dir changed in the last --quiet-hours (default 6):
     a resubmitted round writes into dirs named after the FIRST job, so a live job's id
     need not appear in the dir name and predicate 4 alone cannot see it
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

TASK = "denoise"
RUNS = Path("/lus/flare/projects/UIC-HPC/khuss/msdelta/runs").resolve()
CKPT = re.compile(r"^checkpoint-\d+$")
WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def active_job_ids() -> set[str]:
    """Job ids currently queued or running. A run under one of these is off limits."""
    try:
        out = subprocess.run(["qstat", "-u", os.environ.get("USER", "khuss")],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        # Cannot prove nothing is running -> refuse to delete anything.
        raise SystemExit("qstat failed; refusing to prune without knowing what is live")
    return {m.group(1) for m in re.finditer(r"^(\d+)\.", out, re.M)}


def recently_touched(run: Path, hours: float) -> bool:
    cutoff = time.time() - hours * 3600
    return any(e.stat(follow_symlinks=False).st_mtime > cutoff for e in os.scandir(run)) \
        or run.stat().st_mtime > cutoff


def candidates(active: set[str], quiet_hours: float = 6.0):
    for run in sorted(RUNS.glob("sweep-*")):
        run = run.resolve()
        if not str(run).startswith(str(RUNS) + os.sep):       # 1
            continue
        job = run.name.rsplit("-", 1)[-1]
        if job in active:                                      # 4
            continue
        if recently_touched(run, quiet_hours):                 # 6
            continue
        results = run / "all_results.json"
        if not results.exists():
            continue
        try:
            payload = json.loads(results.read_text())
        except Exception:
            continue
        kind = ("denoise" if "test_auroc" in payload else
                "contrastive" if "sep_spectrum/ratio" in payload else None)
        if kind != TASK:                                       # 2  one task per run
            continue
        final = run / "final"
        if not final.is_dir() or not any((final / w).exists() for w in WEIGHTS):
            continue                                           # 3
        targets = [c for c in run.glob("checkpoint-*")
                   if c.is_dir() and not c.is_symlink() and CKPT.match(c.name)]  # 5
        if targets:
            yield run, sorted(targets)


def size_of(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def main() -> int:
    global TASK, RUNS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("denoise", "contrastive"), default="denoise",
                    help="which task's runs to prune; never both at once")
    ap.add_argument("--limit", type=int, help="only consider the first N runs")
    ap.add_argument("--runs", type=Path, default=RUNS,
                    help="runs root; point at a COPY to test the script before a live run")
    ap.add_argument("--quiet-hours", type=float, default=6.0,
                    help="skip runs with anything modified this recently")
    ap.add_argument("--execute", action="store_true",
                    help="actually delete. Without it nothing is removed.")
    cli = ap.parse_args()
    TASK = cli.task
    RUNS = cli.runs.resolve()
    print(f"  runs root: {RUNS}   task: {TASK}")

    active = active_job_ids()
    print(f"  active job ids (excluded): {sorted(active) or 'none'}")
    runs = list(candidates(active, cli.quiet_hours))
    if cli.limit:
        runs = runs[:cli.limit]
    freed = 0
    for run, targets in runs:
        for t in targets:
            n = size_of(t)
            freed += n
            action = "DELETE " if cli.execute else "would  "
            print(f"  {action} {n/2**30:7.2f} GB  {t.relative_to(RUNS)}")
            if cli.execute:
                shutil.rmtree(t)
        # Prove the survivor is still intact, every time, not once at the end.
        final = run / "final"
        assert final.is_dir() and any((final / w).exists() for w in WEIGHTS), \
            f"final/ vanished under {run}"
    print(f"\n  {len(runs)} runs, {sum(len(t) for _, t in runs)} checkpoint dirs, "
          f"{freed/2**40:.3f} TB {'freed' if cli.execute else 'would be freed'}")
    if not cli.execute:
        print("  DRY RUN -- nothing was deleted. Re-run with --execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
