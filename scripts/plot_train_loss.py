#!/usr/bin/env python
"""Fetch `train/loss` from wandb for one or more runs and plot them together.

Runs are addressed by their display name (the name shown in the wandb UI), which
for these training runs is the config `wandb_run_name` with the PBS job id
appended, e.g. `consensus_xl_28M_7235042`. Each run may live in a different
project, so a run is specified as `project/run_name` (optionally
`entity/project/run_name`).

Examples
--------
    # the two runs from the default entity
    python scripts/plot_train_loss.py \
        msdelta-consensus/consensus_xl_28M_7235042 \
        msdelta-capacity-kl/v14_cap_XL_hf_7232144

    # write to a file instead of showing a window
    python scripts/plot_train_loss.py -o train_loss.png <run> <run>

Requires `wandb login` (or WANDB_API_KEY) to have been done already.
"""
import argparse
import sys

import matplotlib.pyplot as plt
import wandb


def parse_run_spec(spec, default_entity):
    """`project/name`, `entity/project/name`, or `project:name` -> (entity, project, name)."""
    # allow a `:` between project and run name so run names with `/` are unambiguous
    if ":" in spec:
        proj_part, name = spec.rsplit(":", 1)
        parts = proj_part.split("/")
    else:
        parts = spec.split("/")
        name = parts.pop()
    if len(parts) == 1:
        entity, project = default_entity, parts[0]
    elif len(parts) == 2:
        entity, project = parts
    else:
        raise ValueError(
            f"cannot parse run spec {spec!r}; expected project/run_name "
            "or entity/project/run_name"
        )
    if entity is None:
        raise ValueError(
            f"no entity for {spec!r}; pass --entity or use entity/project/run_name"
        )
    return entity, project, name


def find_run(api, entity, project, name):
    """Resolve a run by display name (falls back to run id)."""
    runs = list(api.runs(f"{entity}/{project}", filters={"display_name": name}))
    if not runs:
        # maybe they passed a run id rather than a display name
        try:
            return api.run(f"{entity}/{project}/{name}")
        except Exception:
            raise SystemExit(
                f"no run named {name!r} found in {entity}/{project}"
            )
    if len(runs) > 1:
        ids = ", ".join(r.id for r in runs)
        print(
            f"warning: {len(runs)} runs named {name!r} in {entity}/{project} "
            f"(ids: {ids}); using the first",
            file=sys.stderr,
        )
    return runs[0]


def fetch_loss(run, metric, x_key):
    """Return (xs, ys) for `metric` over the full logged history."""
    xs, ys = [], []
    for row in run.scan_history(keys=[x_key, metric]):
        if row.get(metric) is None:
            continue
        xs.append(row.get(x_key))
        ys.append(row[metric])
    return xs, ys


def main():
    p = argparse.ArgumentParser(
        description="Plot wandb train/loss for one or more runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "runs",
        nargs="+",
        help="run specs as project/run_name (or entity/project/run_name)",
    )
    p.add_argument(
        "--entity",
        default=None,
        help="default wandb entity for specs that omit one (defaults to your "
        "wandb default entity)",
    )
    p.add_argument("--metric", default="train/loss", help="metric key to plot")
    p.add_argument("--x-key", default="step", help="x-axis key")
    p.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="rolling-mean window (in points) to smooth the curves; 0 disables",
    )
    p.add_argument("--log-y", action="store_true", help="log-scale the y axis")
    p.add_argument("-o", "--out", default=None, help="save to this file instead of showing")
    p.add_argument("--title", default=None, help="plot title")
    args = p.parse_args()

    api = wandb.Api()
    default_entity = args.entity or api.default_entity

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for spec in args.runs:
        entity, project, name = parse_run_spec(spec, default_entity)
        run = find_run(api, entity, project, name)
        xs, ys = fetch_loss(run, args.metric, args.x_key)
        if not ys:
            print(
                f"warning: no {args.metric!r} points for {name} ({project})",
                file=sys.stderr,
            )
            continue
        label = f"{name}  ({project})"
        if args.smooth and args.smooth > 1 and len(ys) >= args.smooth:
            import numpy as np

            kernel = np.ones(args.smooth) / args.smooth
            ys_s = np.convolve(ys, kernel, mode="valid")
            xs_s = xs[args.smooth - 1 :]
            ax.plot(xs, ys, alpha=0.2, linewidth=0.8)
            ax.plot(xs_s, ys_s, label=label, linewidth=1.6)
        else:
            ax.plot(xs, ys, label=label, linewidth=1.4)
        print(f"{name}: {len(ys)} points, final {args.metric}={ys[-1]:.4f}")

    ax.set_xlabel(args.x_key)
    ax.set_ylabel(args.metric)
    if args.log_y:
        ax.set_yscale("log")
    ax.set_title(args.title or args.metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
