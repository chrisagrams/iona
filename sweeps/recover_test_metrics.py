"""Recompute the test metric for sweep arms that died after training finished.

    python sweeps/recover_test_metrics.py 8840408 --dry-run
    python sweeps/recover_test_metrics.py 8840408

`load_best_model_at_end` fails on arms with many checkpoints: the ranks disagree about
which one is best, and a rank asking for a checkpoint that save_total_limit has already
rotated away raises "Can't find a valid checkpoint". It happens AFTER training completes
and after every evaluation is logged, so the only casualty is the held-out test number --
the final checkpoint is on disk and the weights are intact.

That affects the b12 arms specifically: 72 to 145 checkpoints each, against 6 to 36 for
b48 and b144. Rather than burn 79 node-hours re-running them, load what they saved and
evaluate it.

One caveat, stated in the output: these arms are evaluated with their FINAL weights,
where a successful arm is evaluated with its best-validation weights. For ranking
hyperparameters that difference is small and it is visible in the report rather than
hidden, but it is not nothing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCRATCH = Path("/lus/flare/projects/UIC-HPC/khuss/msdelta/runs")


def arm_state(directory: Path) -> dict:
    """What a finished-but-failed arm left behind."""
    checkpoints = sorted(directory.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))
    state = {"arm": directory.name, "checkpoints": [c.name for c in checkpoints]}
    if not checkpoints:
        return state | {"status": "no checkpoints"}
    final = checkpoints[-1]
    state["final"] = str(final)
    state["has_weights"] = (final / "model.safetensors").exists()
    trainer_state = final / "trainer_state.json"
    if trainer_state.exists():
        saved = json.loads(trainer_state.read_text())
        evaluations = [h for h in saved.get("log_history", []) if "eval_auroc" in h]
        state["evaluations"] = len(evaluations)
        state["best_metric"] = saved.get("best_metric")
        state["global_step"] = saved.get("global_step")
        if evaluations:
            last = evaluations[-1]
            state["last_eval_auroc"] = last.get("eval_auroc")
            state["last_eval_auprc"] = last.get("eval_auprc")
            best = max(evaluations, key=lambda h: h.get("eval_auprc", -1))
            state["best_eval_auroc"] = best.get("eval_auroc")
            state["best_eval_auprc"] = best.get("eval_auprc")
    state["expected_steps"] = expected_steps(directory.name)
    has_test = (directory / "test_results.json").exists()
    if has_test:
        state["status"] = "complete"
    elif state.get("global_step", 0) >= state["expected_steps"] - 1:
        # Trained to the end but never wrote a test result: this is the failure mode.
        state["status"] = "needs test metric"
    else:
        # Still training, or killed part-way. Either way it is not recoverable yet, and
        # calling it a failure would confuse an in-flight arm with a broken one.
        state["status"] = "in progress"
    return state


def expected_steps(name: str, samples_per_epoch: int = 87_200) -> int:
    """Total optimizer steps for an arm, from its own name.

    Arms are named lr..._es..._ep{E}_h{H}[_b{B}], and the batch suffix is omitted at the
    default of 48, which is exactly the convention that keeps older arms from being
    renamed when the axis was added.
    """
    epochs = int(re.search(r"_ep(\d+)", name).group(1))
    batch = re.search(r"_b(\d+)", name)
    return samples_per_epoch * epochs // (int(batch.group(1)) if batch else 48)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", help="PBS job id whose arms to inspect")
    parser.add_argument("--root", default=str(SCRATCH))
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is recoverable and stop")
    cli = parser.parse_args()

    directories = sorted(Path(cli.root).glob(f"sweep-*-{cli.job}"))
    if not directories:
        print(f"no arm directories for job {cli.job} under {cli.root}")
        return 1

    incomplete, running = [], 0
    for directory in directories:
        state = arm_state(directory)
        if state["status"] == "complete":
            continue
        if state["status"] == "in progress":
            running += 1
            continue
        incomplete.append(state)

    complete = len(directories) - len(incomplete) - running
    print(f"{len(directories)} arm directories: {complete} complete, {running} still "
          f"training, {len(incomplete)} finished but missing the test metric\n")
    if not incomplete:
        return 0
    width = max(len(s["arm"]) for s in incomplete)
    print(f"  {'arm':<{width}} {'steps':>7} {'evals':>6} {'best auprc':>11} {'last auroc':>11}  weights")
    for state in incomplete:
        print(f"  {state['arm']:<{width}} {state.get('global_step', 0):>7} "
              f"{state.get('evaluations', 0):>6} {state.get('best_eval_auprc') or 0:>11.4f} "
              f"{state.get('last_eval_auroc') or 0:>11.4f}  "
              f"{'yes' if state.get('has_weights') else 'MISSING'}")

    print("\nEvery arm above finished training and logged its evaluations; only the")
    print("held-out test pass is missing. Re-evaluating uses FINAL weights, where a")
    print("successful arm used best-validation weights -- report that difference rather")
    print("than quietly mixing the two.")
    if cli.dry_run:
        print("\n(dry run) re-run without --dry-run to submit the evaluation job")
        return 0
    print("\nSubmit with:")
    print(f"  qsub -q debug -l select=1 -l walltime=00:59:00 \\")
    print(f"    -v JOB={cli.job} pbs/recover_test_metrics.pbs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
