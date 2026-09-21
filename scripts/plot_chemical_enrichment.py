from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _delta_bias_plotting import (
    CATEGORY_COLORS,
    CATEGORY_ISOTOPE,
    CATEGORY_LABELS,
    CATEGORY_NEUTRAL_LOSS,
    CATEGORY_RESIDUE,
    chemical_references,
    load_bias_module,
    local_peak_scores,
    plt,
    save_figure,
    set_publication_style,
    write_csv,
)

CATEGORY_ORDER = (CATEGORY_RESIDUE, CATEGORY_NEUTRAL_LOSS, CATEGORY_ISOTOPE)

_N_BOOTSTRAP = 10_000
_RATIO_EPS = 1e-6
_MAX_SAMPLE_ROUNDS = 100
_CONTROL_BLOCK_MASSES = 20_000

CSV_HEADER = (
    "category",
    "n_targets",
    "n_heads",
    "mass_min",
    "mass_max",
    "chemical_mean",
    "chemical_ci_low",
    "chemical_ci_high",
    "control_mean",
    "control_p2_5",
    "control_p97_5",
    "effect_difference",
    "enrichment_ratio",
    "p_value",
    "n_random",
    "seed",
    "flank_offsets",
    "symmetric",
    "dm_min",
    "dm_max",
    "exclusion_tolerance",
    "checkpoint",
)


@dataclass
class CategoryStats:
    """Enrichment statistics for one chemical category."""

    category: str
    names: list[str]
    masses: np.ndarray
    n_heads: int
    chemical_mean: float
    chemical_ci: tuple[float, float]
    control_mean: float
    control_band: tuple[float, float]
    effect_difference: float
    enrichment_ratio: float | None
    p_value: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Figure B command line."""
    parser = argparse.ArgumentParser(description="Quantify chemical enrichment of the learned Δm bias (Figure B).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretraining checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="figure path (.pdf/.svg/.png)")
    parser.add_argument("--stats-output", type=Path, default=None, help="statistics CSV path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or xpu")
    parser.add_argument("--dm-min", type=float, default=0.0, help="lowest Δm considered")
    parser.add_argument("--dm-max", type=float, default=200.0, help="highest Δm considered")
    parser.add_argument("--n-random", type=int, default=10_000, help="matched control sets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symmetric", action="store_true", help="fold b(+Δm) with b(-Δm)")
    parser.add_argument(
        "--flank-offsets",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.4, 0.8],
        help="offsets in Da used for the local baseline median",
    )
    parser.add_argument(
        "--exclusion-tolerance",
        type=float,
        default=0.1,
        help="keep random controls this far from every known chemical target",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default=None, help="figure title")
    return parser.parse_args(argv)


def collect_targets(dm_min: float, dm_max: float) -> dict[str, list[tuple[str, float]]]:
    """Group in-range chemical references by category."""
    grouped: dict[str, list[tuple[str, float]]] = {name: [] for name in CATEGORY_ORDER}
    for ref in chemical_references():
        if ref.category in grouped and dm_min <= ref.mass <= dm_max:
            grouped[ref.category].append((ref.name, ref.mass))
    return {key: value for key, value in grouped.items() if value}


def sample_controls(
    rng: np.random.Generator,
    n_sets: int,
    n_per_set: int,
    lo: float,
    hi: float,
    excluded: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Draw ``n_sets`` matched control sets uniformly on ``[lo, hi]``.

    Candidates within ``tolerance`` of any known chemical target are rejected
    and redrawn, so the control never accidentally samples real chemistry.
    Returns an array of shape ``[n_sets, n_per_set]``.
    """
    if hi <= lo:
        raise ValueError(f"empty control range [{lo}, {hi}]")
    needed = n_sets * n_per_set
    kept: list[np.ndarray] = []
    found = 0
    for _ in range(_MAX_SAMPLE_ROUNDS):
        if found >= needed:
            break
        draw = rng.uniform(lo, hi, size=max(needed - found, 1024) * 2)
        distance = np.full(draw.shape, np.inf)
        for value in excluded:
            np.minimum(distance, np.abs(draw - value), out=distance)
        good = draw[distance >= tolerance]
        if good.size:
            kept.append(good)
            found += good.size
    if found < needed:
        raise RuntimeError(
            "could not draw enough control masses; lower --exclusion-tolerance "
            "or widen the mass range"
        )
    return np.concatenate(kept)[:needed].reshape(n_sets, n_per_set)


def bootstrap_ci(rng: np.random.Generator, values: np.ndarray) -> tuple[float, float]:
    """Return the bootstrap 95% CI of the mean of ``values``."""
    flat = values.reshape(-1)
    if flat.size == 0:
        return (float("nan"), float("nan"))
    picks = rng.integers(0, flat.size, size=(_N_BOOTSTRAP, flat.size))
    means = flat[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(low), float(high))


def analyze_category(
    bias_module,
    category: str,
    targets: list[tuple[str, float]],
    excluded: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> CategoryStats:
    """Compare a category's local-peak scores against matched random controls."""
    names = [name for name, _ in targets]
    masses = np.array([value for _, value in targets], dtype=np.float64)
    lo, hi = float(masses.min()), float(masses.max())

    chemical = local_peak_scores(bias_module, masses, args.flank_offsets, symmetric=args.symmetric)
    n_heads = chemical.shape[1]
    chemical_mean = float(chemical.mean())
    chemical_ci = bootstrap_ci(rng, chemical)

    controls = sample_controls(
        rng,
        args.n_random,
        masses.size,
        lo,
        hi,
        excluded,
        args.exclusion_tolerance,
    )
    block = max(1, _CONTROL_BLOCK_MASSES // max(masses.size, 1))
    set_means = np.empty(args.n_random, dtype=np.float64)
    for start in range(0, args.n_random, block):
        piece = controls[start : start + block]
        scores = local_peak_scores(
            bias_module, piece.reshape(-1), args.flank_offsets, symmetric=args.symmetric
        )
        set_means[start : start + piece.shape[0]] = scores.reshape(
            piece.shape[0], masses.size, n_heads
        ).mean(axis=(1, 2))
    control_mean = float(set_means.mean())
    band = np.percentile(set_means, [2.5, 97.5])

    ratio: float | None = None
    if abs(control_mean) > _RATIO_EPS and control_mean > 0.0:
        ratio = chemical_mean / control_mean

    p_value = float((1 + int(np.count_nonzero(set_means >= chemical_mean))) / (args.n_random + 1))

    return CategoryStats(
        category=category,
        names=names,
        masses=masses,
        n_heads=n_heads,
        chemical_mean=chemical_mean,
        chemical_ci=chemical_ci,
        control_mean=control_mean,
        control_band=(float(band[0]), float(band[1])),
        effect_difference=chemical_mean - control_mean,
        enrichment_ratio=ratio,
        p_value=p_value,
    )


def format_p(p_value: float, n_random: int) -> str:
    """Format a Monte Carlo p-value, respecting its resolution floor."""
    floor = 1.0 / (n_random + 1)
    if p_value <= floor:
        return f"p < {floor:.1e}"
    return f"p = {p_value:.3g}"


def plot_enrichment(stats: list[CategoryStats], n_random: int, title: str | None) -> plt.Figure:
    """Draw paired chemical/control means with error bars, one pair per category."""
    fig, ax = plt.subplots(figsize=(1.6 * len(stats) + 2.2, 3.0))
    offset = 0.16
    positions = np.arange(len(stats), dtype=float)

    for index, entry in enumerate(stats):
        color = CATEGORY_COLORS[entry.category]
        chem_err = np.array(
            [
                [entry.chemical_mean - entry.chemical_ci[0]],
                [entry.chemical_ci[1] - entry.chemical_mean],
            ]
        )
        ctrl_err = np.array(
            [
                [entry.control_mean - entry.control_band[0]],
                [entry.control_band[1] - entry.control_mean],
            ]
        )
        ax.errorbar(
            positions[index] - offset,
            entry.chemical_mean,
            yerr=np.abs(chem_err),
            fmt="o",
            markersize=4.5,
            color=color,
            capsize=2.5,
            elinewidth=0.8,
            label="Chemical targets" if index == 0 else None,
        )
        ax.errorbar(
            positions[index] + offset,
            entry.control_mean,
            yerr=np.abs(ctrl_err),
            fmt="s",
            markersize=4.0,
            color="0.45",
            markerfacecolor="white",
            capsize=2.5,
            elinewidth=0.8,
            label="Matched random control" if index == 0 else None,
        )
        top = max(entry.chemical_ci[1], entry.control_band[1])
        ax.annotate(
            format_p(entry.p_value, n_random),
            xy=(positions[index], top),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            color="0.25",
        )

    ax.axhline(0.0, color="0.7", linewidth=0.5, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{CATEGORY_LABELS[e.category]}\n(n={len(e.names)})" for e in stats], fontsize=7
    )
    ax.set_xlim(-0.5, len(stats) - 0.5)
    ax.set_ylabel("Local peak score (logits)")
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def stats_rows(stats: list[CategoryStats], args: argparse.Namespace) -> list[list[object]]:
    """Render the statistics table, repeating run metadata on every row."""
    offsets = " ".join(f"{value:g}" for value in args.flank_offsets)
    rows: list[list[object]] = []
    for entry in stats:
        rows.append(
            [
                entry.category,
                len(entry.names),
                entry.n_heads,
                f"{entry.masses.min():.6f}",
                f"{entry.masses.max():.6f}",
                f"{entry.chemical_mean:.10g}",
                f"{entry.chemical_ci[0]:.10g}",
                f"{entry.chemical_ci[1]:.10g}",
                f"{entry.control_mean:.10g}",
                f"{entry.control_band[0]:.10g}",
                f"{entry.control_band[1]:.10g}",
                f"{entry.effect_difference:.10g}",
                "" if entry.enrichment_ratio is None else f"{entry.enrichment_ratio:.10g}",
                f"{entry.p_value:.10g}",
                args.n_random,
                args.seed,
                offsets,
                int(args.symmetric),
                f"{args.dm_min:g}",
                f"{args.dm_max:g}",
                f"{args.exclusion_tolerance:g}",
                str(args.checkpoint),
            ]
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Build and save Figure B along with its statistics CSV."""
    args = parse_args(argv)
    if args.dm_max <= args.dm_min:
        raise SystemExit("--dm-max must exceed --dm-min")
    if args.n_random < 1:
        raise SystemExit("--n-random must be at least 1")
    if args.exclusion_tolerance < 0:
        raise SystemExit("--exclusion-tolerance must be non-negative")

    stats_output = args.stats_output or args.output.with_name(args.output.stem + ".csv")
    set_publication_style()
    rng = np.random.default_rng(args.seed)
    bias_module = load_bias_module(args.checkpoint, args.device)

    grouped = collect_targets(args.dm_min, args.dm_max)
    if not grouped:
        raise SystemExit(f"no chemical references fall inside [{args.dm_min}, {args.dm_max}] Da")
    excluded = np.array([ref.mass for ref in chemical_references()], dtype=np.float64)

    stats: list[CategoryStats] = []
    for category in CATEGORY_ORDER:
        if category not in grouped:
            continue
        entry = analyze_category(bias_module, category, grouped[category], excluded, args, rng)
        stats.append(entry)
        ratio = "n/a" if entry.enrichment_ratio is None else f"{entry.enrichment_ratio:.3g}"
        print(
            f"{CATEGORY_LABELS[category]}: n={len(entry.names)} heads={entry.n_heads} "
            f"chem={entry.chemical_mean:.4g} ctrl={entry.control_mean:.4g} "
            f"diff={entry.effect_difference:.4g} ratio={ratio} "
            f"{format_p(entry.p_value, args.n_random)}"
        )

    fig = plot_enrichment(stats, args.n_random, args.title)
    save_figure(fig, args.output, args.dpi)
    write_csv(stats_output, CSV_HEADER, stats_rows(stats, args))
    print(f"wrote {args.output} and {stats_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
