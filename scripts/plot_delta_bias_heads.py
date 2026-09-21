from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _delta_bias_plotting import (
    GUIDE_RESIDUES,
    evaluate_bias,
    load_bias_module,
    plt,
    references_in_range,
    save_figure,
    set_publication_style,
)

from msdelta.chemistry import ISOTOPES, NEUTRAL_LOSSES, RESIDUES_AA20

_MAX_AUTO_CURVES = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Figure A command line."""
    parser = argparse.ArgumentParser(description="Plot the learned per-head Δm attention bias (Figure A).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretraining checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="figure path (.pdf/.svg/.png)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or xpu")
    parser.add_argument("--mode", choices=("heatmap", "curves"), default="heatmap")
    parser.add_argument("--dm-min", type=float, default=0.0, help="lowest Δm in Da")
    parser.add_argument("--dm-max", type=float, default=250.0, help="highest Δm in Da")
    parser.add_argument("--dm-step", type=float, default=0.01, help="grid spacing in Da")
    parser.add_argument("--heads", type=int, nargs="+", default=None, help="heads to draw")
    parser.add_argument("--symmetric", action="store_true", help="fold b(+Δm) with b(-Δm)")
    parser.add_argument("--show-references", action="store_true", help="draw chemical guides")
    parser.add_argument("--color-limit", type=float, default=None, help="heatmap color bound")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default=None, help="figure title")
    return parser.parse_args(argv)


def guide_references(lo: float, hi: float) -> list[tuple[str, float, str]]:
    """Return the restrained ¹³C / H₂O / NH₃ / residue guides inside ``[lo, hi]``."""
    wanted = {
        "¹³C": ISOTOPES["¹³C"],
        "H₂O": NEUTRAL_LOSSES["H₂O"],
        "NH₃": NEUTRAL_LOSSES["NH₃"],
    }
    wanted.update({name: RESIDUES_AA20[name] for name in GUIDE_RESIDUES})
    by_mass = {ref.name: ref for ref in references_in_range(lo, hi)}
    guides = [
        (name, value, by_mass[name].color) for name, value in wanted.items() if name in by_mass
    ]
    return sorted(guides, key=lambda item: item[1])


def draw_references(ax: plt.Axes, lo: float, hi: float, *, label: bool) -> None:
    """Draw thin vertical guides at the selected chemical masses."""
    guides = guide_references(lo, hi)
    if not guides:
        return
    y_top = ax.get_ylim()[1]
    for name, value, color in guides:
        ax.axvline(value, color=color, linewidth=0.4, alpha=0.55, zorder=0)
        if label:
            ax.annotate(
                name,
                xy=(value, y_top),
                xytext=(0, 2),
                textcoords="offset points",
                fontsize=5.5,
                rotation=90,
                ha="center",
                va="bottom",
                color=color,
                clip_on=False,
            )


def select_heads(curves: np.ndarray, heads: list[int] | None) -> list[int]:
    """Return the requested heads, or the highest-variance heads by default."""
    n_heads = curves.shape[1]
    if heads:
        bad = [head for head in heads if not 0 <= head < n_heads]
        if bad:
            raise ValueError(f"heads {bad} are outside the checkpoint's range 0..{n_heads - 1}")
        return list(dict.fromkeys(heads))
    variance = curves.var(axis=0)
    order = np.argsort(variance)[::-1][: min(_MAX_AUTO_CURVES, n_heads)]
    return sorted(int(head) for head in order)


def plot_heatmap(
    grid: np.ndarray,
    curves: np.ndarray,
    heads: list[int] | None,
    color_limit: float | None,
    show_references: bool,
    title: str | None,
) -> plt.Figure:
    """Render the head × Δm bias heatmap with a zero-centered diverging map."""
    n_heads = curves.shape[1]
    rows = list(range(n_heads)) if not heads else select_heads(curves, heads)
    image = curves[:, rows].T

    limit = color_limit
    if limit is None:
        limit = float(np.percentile(np.abs(image), 99.0))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.max(np.abs(image))) or 1.0

    height = max(1.7, 0.22 * len(rows) + 1.1)
    fig, ax = plt.subplots(figsize=(7.0, height))
    mesh = ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
        extent=(float(grid[0]), float(grid[-1]), -0.5, len(rows) - 0.5),
    )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([str(head) for head in rows])
    ax.set_xlabel("Mass difference Δm (Da)")
    ax.set_ylabel("Attention head")
    if show_references:
        draw_references(ax, float(grid[0]), float(grid[-1]), label=True)
    bar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.03)
    bar.set_label("Attention bias (logits)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_curves(
    grid: np.ndarray,
    curves: np.ndarray,
    heads: list[int] | None,
    show_references: bool,
    title: str | None,
) -> plt.Figure:
    """Render selected head curves as colored lines on one axis."""
    rows = select_heads(curves, heads)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    colors = plt.get_cmap("tab10")
    for position, head in enumerate(rows):
        ax.plot(grid, curves[:, head], color=colors(position % 10), label=f"head {head}")
    ax.set_xlim(float(grid[0]), float(grid[-1]))
    ax.set_xlabel("Mass difference Δm (Da)")
    ax.set_ylabel("Attention bias (logits)")
    ax.axhline(0.0, color="0.6", linewidth=0.4, zorder=0)
    if show_references:
        draw_references(ax, float(grid[0]), float(grid[-1]), label=True)
    ax.legend(ncols=min(len(rows), 4), loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    """Build and save Figure A."""
    args = parse_args(argv)
    if args.dm_step <= 0:
        raise SystemExit("--dm-step must be positive")
    if args.dm_max <= args.dm_min:
        raise SystemExit("--dm-max must exceed --dm-min")

    set_publication_style()
    bias_module = load_bias_module(args.checkpoint, args.device)

    n_points = int(round((args.dm_max - args.dm_min) / args.dm_step)) + 1
    grid = args.dm_min + args.dm_step * np.arange(n_points, dtype=np.float64)
    curves = evaluate_bias(bias_module, grid, symmetric=args.symmetric)

    if args.mode == "heatmap":
        fig = plot_heatmap(
            grid, curves, args.heads, args.color_limit, args.show_references, args.title
        )
    else:
        fig = plot_curves(grid, curves, args.heads, args.show_references, args.title)

    save_figure(fig, args.output, args.dpi)
    print(
        f"wrote {args.output} ({args.mode}, {curves.shape[1]} heads, "
        f"{n_points} grid points over [{args.dm_min}, {args.dm_max}] Da)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
