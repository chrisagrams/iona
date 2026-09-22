from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _delta_bias_plotting import (
    chemical_references,
    evaluate_bias,
    load_bias_module,
    local_peak_scores,
    plt,
    references_in_range,
    save_figure,
    set_publication_style,
)
from scipy.ndimage import gaussian_filter1d

from msdelta.chemistry import ISOTOPES, NEUTRAL_LOSSES, RESIDUES_AA20

_MAX_AUTO_CURVES = 4
_MAX_CONTROL_SAMPLE_ROUNDS = 100
_RESIDUE_DISPLAY_NAMES = {
    "G": "Gly",
    "A": "Ala",
    "S": "Ser",
    "P": "Pro",
    "V": "Val",
    "T": "Thr",
    "C": "Cys",
    "L/I": "Leu/Ile",
    "N": "Asn",
    "D": "Asp",
    "Q": "Gln",
    "K": "Lys",
    "E": "Glu",
    "M": "Met",
    "H": "His",
    "F": "Phe",
    "R": "Arg",
    "Y": "Tyr",
    "W": "Trp",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Figure A command line."""
    parser = argparse.ArgumentParser(
        description="Plot the learned per-head Δm attention bias (Figure A)."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretraining checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="figure path (.pdf/.svg/.png)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or xpu")
    parser.add_argument("--mode", choices=("heatmap", "curves"), default="heatmap")
    parser.add_argument("--dm-min", type=float, default=0.0, help="lowest Δm in Da")
    parser.add_argument("--dm-max", type=float, default=200.0, help="highest Δm in Da")
    parser.add_argument("--dm-step", type=float, default=0.01, help="grid spacing in Da")
    parser.add_argument("--heads", type=int, nargs="+", default=None, help="heads to draw")
    parser.add_argument("--symmetric", action="store_true", help="fold b(+Δm) with b(-Δm)")
    parser.add_argument(
        "--show-references",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw chemical guides (enabled by default)",
    )
    parser.add_argument(
        "--heatmap-bin-width-da",
        type=float,
        default=2.0,
        help="width of mean-aggregation bins in the heatmap (default: 2 Da)",
    )
    parser.add_argument(
        "--heatmap-normalization",
        choices=("per-head", "global", "none"),
        default="per-head",
        help="center/scale heatmap values (default: per-head)",
    )
    parser.add_argument(
        "--color-limit",
        type=float,
        default=None,
        help="symmetric heatmap color bound after normalization",
    )
    parser.add_argument(
        "--smoothing-sigma-da",
        type=float,
        default=0.0,
        help="optional Gaussian smoothing before binning in Da (default: 0)",
    )
    parser.add_argument(
        "--reference-n-random",
        type=int,
        default=10_000,
        help="random masses used for residue/head significance tests",
    )
    parser.add_argument(
        "--reference-fdr",
        type=float,
        default=0.05,
        help="FDR threshold for showing enriched residues (default: 0.05)",
    )
    parser.add_argument(
        "--max-residue-guides",
        type=int,
        default=8,
        help="maximum significant residues to label; 0 shows all (default: 8)",
    )
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument(
        "--flank-offsets",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.4, 0.8],
        help="offsets in Da used for the local enrichment baseline",
    )
    parser.add_argument(
        "--exclusion-tolerance",
        type=float,
        default=0.1,
        help="minimum distance between random controls and known chemical masses",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values with the original shape."""
    flat = np.asarray(p_values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat)
    ranked = flat[order]
    adjusted = ranked * flat.size / np.arange(1, flat.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(flat)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result.reshape(p_values.shape)


def _sample_control_masses(
    rng: np.random.Generator,
    n_random: int,
    lo: float,
    hi: float,
    excluded: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Sample masses away from known chemistry for the enrichment null."""
    kept: list[np.ndarray] = []
    found = 0
    for _ in range(_MAX_CONTROL_SAMPLE_ROUNDS):
        if found >= n_random:
            break
        draw = rng.uniform(lo, hi, size=max(n_random - found, 1024) * 2)
        distance = np.full(draw.shape, np.inf)
        for value in excluded:
            np.minimum(distance, np.abs(draw - value), out=distance)
        good = draw[distance >= tolerance]
        if good.size:
            kept.append(good)
            found += good.size
    if found < n_random:
        raise RuntimeError("could not sample enough reference-control masses")
    return np.concatenate(kept)[:n_random]


def significant_residue_heads(
    bias_module,
    lo: float,
    hi: float,
    *,
    n_random: int,
    seed: int,
    fdr: float,
    flank_offsets: list[float],
    exclusion_tolerance: float,
    symmetric: bool,
    max_residues: int,
) -> tuple[dict[str, int], int, int]:
    """Count and rank residues with significant positive local peaks."""
    targets = [(name, mass) for name, mass in RESIDUES_AA20.items() if lo <= mass <= hi]
    if not targets:
        return {}, int(bias_module.n_heads), 0

    names = [name for name, _ in targets]
    masses = np.array([mass for _, mass in targets], dtype=np.float64)
    observed = local_peak_scores(bias_module, masses, flank_offsets, symmetric=symmetric)
    excluded = np.array(
        [ref.mass for ref in chemical_references() if lo <= ref.mass <= hi],
        dtype=np.float64,
    )
    controls = _sample_control_masses(
        np.random.default_rng(seed),
        n_random,
        float(masses.min()),
        float(masses.max()),
        excluded,
        exclusion_tolerance,
    )
    null_scores = local_peak_scores(bias_module, controls, flank_offsets, symmetric=symmetric)
    exceedances = np.count_nonzero(
        null_scores[:, np.newaxis, :] >= observed[np.newaxis, :, :], axis=0
    )
    p_values = (exceedances + 1.0) / (n_random + 1.0)
    significant = _benjamini_hochberg(p_values) <= fdr
    counts = significant.sum(axis=1)
    n_significant = int(np.count_nonzero(counts))

    null_mean = null_scores.mean(axis=0)
    null_scale = null_scores.std(axis=0)
    null_scale = np.where(null_scale > 0.0, null_scale, 1.0)
    standardized_effect = (observed - null_mean) / null_scale
    strengths = np.divide(
        np.where(significant, standardized_effect, 0.0).sum(axis=1),
        counts,
        out=np.full(counts.shape, -np.inf, dtype=np.float64),
        where=counts > 0,
    )
    ranked = sorted(
        np.flatnonzero(counts),
        key=lambda index: (-int(counts[index]), -float(strengths[index]), masses[index]),
    )
    if max_residues > 0:
        ranked = ranked[:max_residues]
    selected = {names[index]: int(counts[index]) for index in ranked}
    return selected, observed.shape[1], n_significant


def guide_references(
    lo: float, hi: float, residue_head_counts: dict[str, int]
) -> list[tuple[str, float, str, int | None]]:
    """Return fixed chemistry plus significantly enriched residue guides."""
    wanted = {
        "¹³C": ISOTOPES["¹³C"],
        "H₂O": NEUTRAL_LOSSES["H₂O"],
        "NH₃": NEUTRAL_LOSSES["NH₃"],
    }
    wanted.update({name: RESIDUES_AA20[name] for name in residue_head_counts})
    by_mass = {ref.name: ref for ref in references_in_range(lo, hi)}
    guides = [
        (name, value, by_mass[name].color, residue_head_counts.get(name))
        for name, value in wanted.items()
        if name in by_mass
    ]
    return sorted(guides, key=lambda item: item[1])


def draw_references(
    ax: plt.Axes,
    lo: float,
    hi: float,
    *,
    label: bool,
    residue_head_counts: dict[str, int],
    n_heads: int,
    bin_edges: np.ndarray | None = None,
) -> None:
    """Draw grid-aligned selections at the chosen chemical masses."""
    guides = guide_references(lo, hi, residue_head_counts)
    if not guides:
        return
    y_top = ax.get_ylim()[1]
    crowded_offsets = {"H₂O": -18, "NH₃": 18}
    residue_levels: dict[str, int] = {}
    last_mass_by_level = [-np.inf] * 4
    for name, value, _color, significant_heads in guides:
        if significant_heads is None:
            continue
        level = next(
            (
                candidate
                for candidate, last_mass in enumerate(last_mass_by_level)
                if value - last_mass >= 7.0
            ),
            len(last_mass_by_level) - 1,
        )
        residue_levels[name] = level
        last_mass_by_level[level] = value

    for name, value, _color, significant_heads in guides:
        if bin_edges is None:
            left = max(lo, value - 0.25)
            right = min(hi, value + 0.25)
        else:
            bin_index = int(np.searchsorted(bin_edges, value, side="right") - 1)
            bin_index = int(np.clip(bin_index, 0, len(bin_edges) - 2))
            left, right = float(bin_edges[bin_index]), float(bin_edges[bin_index + 1])

        if label:
            ax.axvspan(
                left,
                right,
                facecolor="none",
                edgecolor="0.12",
                linewidth=0.8,
                zorder=3,
            )
        else:
            ax.axvspan(
                left,
                right,
                facecolor="0.15",
                edgecolor="none",
                alpha=0.12,
                zorder=0,
            )
        if label:
            if significant_heads is None:
                annotation = f"{name}\n{value:.3f}"
                y_offset = 10
            else:
                annotation = (
                    f"{_RESIDUE_DISPLAY_NAMES.get(name, name)}\n{significant_heads}/{n_heads}"
                )
                y_offset = 10 + 22 * residue_levels[name]
            ax.annotate(
                annotation,
                xy=(value, y_top),
                xytext=(crowded_offsets.get(name, 0), y_offset),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                va="bottom",
                color="black",
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


def bin_curves(
    grid: np.ndarray, curves: np.ndarray, bin_width_da: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-aggregate dense bias samples into fixed-width mass bins."""
    lo, hi = float(grid[0]), float(grid[-1])
    n_bins = max(1, int(np.ceil((hi - lo) / bin_width_da)))
    edges = np.minimum(lo + bin_width_da * np.arange(n_bins + 1), hi)
    edges[-1] = hi

    counts = np.histogram(grid, bins=edges)[0]
    if np.any(counts == 0):
        raise ValueError("heatmap bins must not be narrower than the evaluation grid")
    binned = np.column_stack(
        [
            np.histogram(grid, bins=edges, weights=curves[:, head])[0] / counts
            for head in range(curves.shape[1])
        ]
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, edges, binned


def normalize_heatmap(curves: np.ndarray, mode: str) -> np.ndarray:
    """Remove irrelevant offsets and robustly scale heatmap rows."""
    if mode == "none":
        return curves.copy()

    centered = curves - np.median(curves, axis=0, keepdims=True)
    if mode == "global":
        scale = np.percentile(np.abs(centered), 99.0)
    elif mode == "per-head":
        scale = np.percentile(np.abs(centered), 99.0, axis=0, keepdims=True)
    else:
        raise ValueError(f"unknown heatmap normalization {mode!r}")
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    return centered / scale


def plot_heatmap(
    grid: np.ndarray,
    curves: np.ndarray,
    heads: list[int] | None,
    color_limit: float | None,
    show_references: bool,
    bin_width_da: float,
    normalization: str,
    residue_head_counts: dict[str, int],
    reference_n_heads: int,
) -> plt.Figure:
    """Render the binned, normalized head-by-mass heatmap."""
    n_heads = curves.shape[1]
    rows = list(range(n_heads)) if not heads else select_heads(curves, heads)
    _binned_grid, bin_edges, binned_curves = bin_curves(grid, curves, bin_width_da)
    display_curves = normalize_heatmap(binned_curves, normalization)
    image = display_curves[:, rows].T

    limit = color_limit
    if limit is None:
        limit = float(np.percentile(np.abs(image), 99.0))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.max(np.abs(image))) or 1.0

    fig, ax = plt.subplots(figsize=(12.0, 4.1))
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.12, top=0.70)
    mesh = ax.pcolormesh(
        bin_edges,
        np.arange(len(rows) + 1) + 0.5,
        image,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        shading="flat",
        rasterized=True,
    )
    ax.set_xlim(float(grid[0]), float(grid[-1]))
    ax.set_ylim(len(rows) + 0.5, 0.5)
    if len(rows) > 8:
        tick_positions = np.unique(np.linspace(1, len(rows), 5, dtype=int))
    else:
        tick_positions = np.arange(1, len(rows) + 1)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(rows[position - 1] + 1) for position in tick_positions])
    ax.set_xlabel("Mass difference Δm (Da)")
    ax.set_ylabel("Attention head")
    if show_references:
        draw_references(
            ax,
            float(grid[0]),
            float(grid[-1]),
            label=True,
            residue_head_counts=residue_head_counts,
            n_heads=reference_n_heads,
            bin_edges=bin_edges,
        )
    bar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.022)
    if normalization == "per-head":
        color_title = "Head-normalized\nrelative bias"
    elif normalization == "global":
        color_title = "Centered relative\nbias (shared scale)"
    else:
        color_title = r"Relative bias $b^{(h)}(\Delta m)$"
    bar.ax.set_title(color_title, fontsize=7, pad=6)

    return fig


def plot_curves(
    grid: np.ndarray,
    curves: np.ndarray,
    heads: list[int] | None,
    show_references: bool,
    residue_head_counts: dict[str, int],
    reference_n_heads: int,
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
        draw_references(
            ax,
            float(grid[0]),
            float(grid[-1]),
            label=True,
            residue_head_counts=residue_head_counts,
            n_heads=reference_n_heads,
        )
    ax.legend(ncols=min(len(rows), 4), loc="best")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    """Build and save Figure A."""
    args = parse_args(argv)
    if args.dm_step <= 0:
        raise SystemExit("--dm-step must be positive")
    if args.dm_max <= args.dm_min:
        raise SystemExit("--dm-max must exceed --dm-min")
    if args.heatmap_bin_width_da <= 0:
        raise SystemExit("--heatmap-bin-width-da must be positive")
    if args.heatmap_bin_width_da < args.dm_step:
        raise SystemExit("--heatmap-bin-width-da must be at least --dm-step")
    if args.color_limit is not None and args.color_limit <= 0:
        raise SystemExit("--color-limit must be positive")
    if args.reference_n_random <= 0:
        raise SystemExit("--reference-n-random must be positive")
    if not 0.0 < args.reference_fdr <= 1.0:
        raise SystemExit("--reference-fdr must be in (0, 1]")
    if args.max_residue_guides < 0:
        raise SystemExit("--max-residue-guides must be non-negative")
    if not args.flank_offsets or any(value <= 0 for value in args.flank_offsets):
        raise SystemExit("--flank-offsets must contain positive values")
    if args.exclusion_tolerance < 0:
        raise SystemExit("--exclusion-tolerance must be non-negative")

    set_publication_style()
    bias_module = load_bias_module(args.checkpoint, args.device)
    if args.show_references:
        residue_head_counts, reference_n_heads, n_significant_residues = significant_residue_heads(
            bias_module,
            args.dm_min,
            args.dm_max,
            n_random=args.reference_n_random,
            seed=args.reference_seed,
            fdr=args.reference_fdr,
            flank_offsets=args.flank_offsets,
            exclusion_tolerance=args.exclusion_tolerance,
            symmetric=args.symmetric,
            max_residues=args.max_residue_guides,
        )
    else:
        residue_head_counts = {}
        reference_n_heads = int(bias_module.n_heads)
        n_significant_residues = 0

    n_points = int(round((args.dm_max - args.dm_min) / args.dm_step)) + 1
    grid = args.dm_min + args.dm_step * np.arange(n_points, dtype=np.float64)
    curves = evaluate_bias(bias_module, grid, symmetric=args.symmetric)
    if args.smoothing_sigma_da < 0:
        raise SystemExit("--smoothing-sigma-da must be non-negative")
    if args.smoothing_sigma_da > 0:
        curves = gaussian_filter1d(
            curves,
            sigma=args.smoothing_sigma_da / args.dm_step,
            axis=0,
            mode="nearest",
        )

    if args.mode == "heatmap":
        fig = plot_heatmap(
            grid,
            curves,
            args.heads,
            args.color_limit,
            args.show_references,
            args.heatmap_bin_width_da,
            args.heatmap_normalization,
            residue_head_counts,
            reference_n_heads,
        )
    else:
        fig = plot_curves(
            grid,
            curves,
            args.heads,
            args.show_references,
            residue_head_counts,
            reference_n_heads,
        )

    save_figure(fig, args.output, args.dpi)
    print(
        f"wrote {args.output} ({args.mode}, {curves.shape[1]} heads, "
        f"{n_points} grid points over [{args.dm_min}, {args.dm_max}] Da, "
        f"showing {len(residue_head_counts)} of {n_significant_residues} "
        "FDR-significant residues)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
