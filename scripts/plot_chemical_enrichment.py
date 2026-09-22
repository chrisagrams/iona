from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _delta_bias_plotting import (
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
CATEGORY_RANDOM = "random"
RANDOM_LABEL = "Random masses\n(negative control)"
_N_BOOTSTRAP = 10_000
_MAX_SAMPLE_ROUNDS = 100
_CONTROL_BLOCK_MASSES = 20_000

CSV_HEADER = (
    "category",
    "n_targets",
    "n_heads",
    "known_mean",
    "known_ci_low",
    "known_ci_high",
    "random_mean",
    "random_p2_5",
    "random_p97_5",
    "effect_difference",
    "enrichment_ratio",
    "p_value",
    "p_value_holm",
    "n_random_sets",
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
    """Random-set enrichment statistics for one chemical category."""

    category: str
    names: list[str]
    masses: np.ndarray
    n_heads: int
    known_mean: float
    known_ci: tuple[float, float]
    random_mean: float
    random_interval: tuple[float, float]
    effect_difference: float
    enrichment_ratio: float
    p_value: float
    p_value_holm: float = float("nan")
    random_set_means: np.ndarray | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Figure B command line."""
    parser = argparse.ArgumentParser(
        description="Compare delta-bias contrast at known and random mass differences."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretraining checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="figure path (.pdf/.svg/.png)")
    parser.add_argument("--stats-output", type=Path, default=None, help="statistics CSV path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or xpu")
    parser.add_argument("--dm-min", type=float, default=0.0, help="lowest random Δm")
    parser.add_argument("--dm-max", type=float, default=200.0, help="highest random Δm")
    parser.add_argument("--n-random", type=int, default=10_000, help="random mass sets")
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
        help="keep random masses this far from known chemical masses",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def collect_targets(dm_min: float, dm_max: float) -> dict[str, list[tuple[str, float]]]:
    """Group in-range chemical references by category."""
    grouped: dict[str, list[tuple[str, float]]] = {name: [] for name in CATEGORY_ORDER}
    for ref in chemical_references():
        if ref.category in grouped and dm_min <= ref.mass <= dm_max:
            grouped[ref.category].append((ref.name, ref.mass))
    return {category: targets for category, targets in grouped.items() if targets}


def sample_random_sets(
    rng: np.random.Generator,
    n_sets: int,
    n_per_set: int,
    lo: float,
    hi: float,
    excluded: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Draw equally sized random non-chemical mass sets from the plot range."""
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
        raise RuntimeError("could not draw enough random masses")
    return np.concatenate(kept)[:needed].reshape(n_sets, n_per_set)


def mean_absolute_contrast(
    bias_module,
    masses: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Return the across-head mean absolute local contrast for each mass."""
    scores = local_peak_scores(bias_module, masses, args.flank_offsets, symmetric=args.symmetric)
    return np.abs(scores).mean(axis=1)


def bootstrap_mean_ci(rng: np.random.Generator, values: np.ndarray) -> tuple[float, float]:
    """Bootstrap target masses to obtain a 95% interval for their mean."""
    picks = rng.integers(0, values.size, size=(_N_BOOTSTRAP, values.size))
    means = values[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def score_random_sets(
    bias_module,
    controls: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Return one mean local-contrast score for each random mass set."""
    n_sets, n_targets = controls.shape
    block = max(1, _CONTROL_BLOCK_MASSES // n_targets)
    set_means = np.empty(n_sets, dtype=np.float64)
    for start in range(0, n_sets, block):
        piece = controls[start : start + block]
        mass_scores = mean_absolute_contrast(bias_module, piece.reshape(-1), args)
        set_means[start : start + piece.shape[0]] = mass_scores.reshape(
            piece.shape[0], n_targets
        ).mean(axis=1)
    return set_means


def analyze_category(
    bias_module,
    category: str,
    targets: list[tuple[str, float]],
    excluded: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> CategoryStats:
    """Compare known masses with equally sized random mass sets."""
    names = [name for name, _ in targets]
    masses = np.array([value for _, value in targets], dtype=np.float64)
    known_scores = mean_absolute_contrast(bias_module, masses, args)
    known_mean = float(known_scores.mean())
    known_ci = bootstrap_mean_ci(rng, known_scores)

    controls = sample_random_sets(
        rng,
        args.n_random,
        masses.size,
        args.dm_min,
        args.dm_max,
        excluded,
        args.exclusion_tolerance,
    )
    random_means = score_random_sets(bias_module, controls, args)
    random_mean = float(random_means.mean())
    random_interval = np.percentile(random_means, [2.5, 97.5])
    p_value = float((1 + np.count_nonzero(random_means >= known_mean)) / (args.n_random + 1))
    return CategoryStats(
        category=category,
        names=names,
        masses=masses,
        n_heads=int(bias_module.n_heads),
        known_mean=known_mean,
        known_ci=known_ci,
        random_mean=random_mean,
        random_interval=(float(random_interval[0]), float(random_interval[1])),
        effect_difference=known_mean - random_mean,
        enrichment_ratio=known_mean / random_mean,
        p_value=p_value,
        random_set_means=random_means,
    )


def make_random_control(source: CategoryStats) -> CategoryStats:
    """Split one null distribution into independent negative-control halves."""
    if source.random_set_means is None:
        raise ValueError("source category has no random-set distribution")
    first, second = np.array_split(source.random_set_means, 2)
    first_mean, second_mean = float(first.mean()), float(second.mean())
    first_interval = np.percentile(first, [2.5, 97.5])
    second_interval = np.percentile(second, [2.5, 97.5])
    return CategoryStats(
        category=CATEGORY_RANDOM,
        names=[f"random_{index + 1}" for index in range(len(source.names))],
        masses=np.array([], dtype=np.float64),
        n_heads=source.n_heads,
        known_mean=first_mean,
        known_ci=(float(first_interval[0]), float(first_interval[1])),
        random_mean=second_mean,
        random_interval=(float(second_interval[0]), float(second_interval[1])),
        effect_difference=first_mean - second_mean,
        enrichment_ratio=first_mean / second_mean,
        p_value=float("nan"),
        p_value_holm=float("nan"),
    )


def apply_holm_correction(stats: list[CategoryStats]) -> None:
    """Attach family-wise-error-adjusted permutation p-values."""
    order = np.argsort([entry.p_value for entry in stats])
    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted = (len(stats) - rank) * stats[int(index)].p_value
        running_max = max(running_max, adjusted)
        stats[int(index)].p_value_holm = min(1.0, running_max)


def format_p(p_value: float) -> str:
    """Format a multiple-comparison-corrected permutation p-value."""
    if p_value < 0.001:
        return r"corrected $p < 0.001$"
    return rf"corrected $p = {p_value:.3f}$"


def plot_enrichment(stats: list[CategoryStats]) -> plt.Figure:
    """Draw a simple known-versus-random enrichment comparison."""
    fig, ax = plt.subplots(figsize=(6.1, 4.25))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.22, top=0.80)
    positions = np.arange(len(stats), dtype=float)
    width = 0.34

    for index, entry in enumerate(stats):
        ax.bar(
            positions[index] - width / 2,
            entry.known_mean,
            width,
            color="#4c8bc4",
            edgecolor="white",
            linewidth=0.4,
            label="Category target Δm" if index == 0 else None,
        )
        ax.bar(
            positions[index] + width / 2,
            entry.random_mean,
            width,
            color="#b9b9b9",
            edgecolor="white",
            linewidth=0.4,
            label="Random Δm sets" if index == 0 else None,
        )
        top = max(entry.known_mean, entry.random_mean)
        if entry.category == CATEGORY_RANDOM:
            ax.annotate(
                f"{entry.enrichment_ratio:.1f}×\nnegative control",
                xy=(positions[index], top),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
            continue
        bracket_low = top * 1.05
        bracket_high = top * 1.11
        ax.plot(
            [
                positions[index] - width / 2,
                positions[index] - width / 2,
                positions[index] + width / 2,
                positions[index] + width / 2,
            ],
            [bracket_low, bracket_high, bracket_high, bracket_low],
            color="black",
            linewidth=0.7,
            clip_on=False,
        )
        ax.annotate(
            f"{entry.enrichment_ratio:.1f}×\n{format_p(entry.p_value_holm)}",
            xy=(positions[index], bracket_high),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            RANDOM_LABEL if entry.category == CATEGORY_RANDOM else CATEGORY_LABELS[entry.category]
            for entry in stats
        ],
        fontsize=8,
    )
    ax.set_xlim(-0.5, len(stats) - 0.5)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Mean absolute local bias contrast (logits)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.99),
        ncols=2,
        fontsize=7,
    )
    return fig


def stats_rows(stats: list[CategoryStats], args: argparse.Namespace) -> list[list[object]]:
    """Render category statistics and run metadata for the CSV output."""
    offsets = " ".join(f"{value:g}" for value in args.flank_offsets)
    return [
        [
            entry.category,
            len(entry.names),
            entry.n_heads,
            f"{entry.known_mean:.10g}",
            f"{entry.known_ci[0]:.10g}",
            f"{entry.known_ci[1]:.10g}",
            f"{entry.random_mean:.10g}",
            f"{entry.random_interval[0]:.10g}",
            f"{entry.random_interval[1]:.10g}",
            f"{entry.effect_difference:.10g}",
            f"{entry.enrichment_ratio:.10g}",
            f"{entry.p_value:.10g}",
            f"{entry.p_value_holm:.10g}",
            args.n_random,
            args.seed,
            offsets,
            int(args.symmetric),
            f"{args.dm_min:g}",
            f"{args.dm_max:g}",
            f"{args.exclusion_tolerance:g}",
            str(args.checkpoint),
        ]
        for entry in stats
    ]


def main(argv: list[str] | None = None) -> int:
    """Build and save the random-set permutation version of Figure B."""
    args = parse_args(argv)
    if args.dm_max <= args.dm_min:
        raise SystemExit("--dm-max must exceed --dm-min")
    if args.n_random < 1:
        raise SystemExit("--n-random must be at least 1")
    if not args.flank_offsets or any(value <= 0 for value in args.flank_offsets):
        raise SystemExit("--flank-offsets must contain positive values")
    if args.exclusion_tolerance < 0:
        raise SystemExit("--exclusion-tolerance must be non-negative")

    stats_output = args.stats_output or args.output.with_name(args.output.stem + ".csv")
    set_publication_style()
    rng = np.random.default_rng(args.seed)
    bias_module = load_bias_module(args.checkpoint, args.device)
    grouped = collect_targets(args.dm_min, args.dm_max)
    excluded = np.array(
        [ref.mass for ref in chemical_references() if args.dm_min <= ref.mass <= args.dm_max],
        dtype=np.float64,
    )
    stats = [
        analyze_category(bias_module, category, grouped[category], excluded, args, rng)
        for category in CATEGORY_ORDER
        if category in grouped
    ]
    apply_holm_correction(stats)
    residue_entry = next((entry for entry in stats if entry.category == CATEGORY_RESIDUE), None)
    if residue_entry is not None:
        stats.append(make_random_control(residue_entry))

    for entry in stats:
        if entry.category == CATEGORY_RANDOM:
            print(
                f"Random negative control: first={entry.known_mean:.4g} "
                f"second={entry.random_mean:.4g} ratio={entry.enrichment_ratio:.3g}"
            )
            continue
        print(
            f"{CATEGORY_LABELS[entry.category]}: n={len(entry.names)} "
            f"known={entry.known_mean:.4g} random={entry.random_mean:.4g} "
            f"ratio={entry.enrichment_ratio:.3g} p={entry.p_value:.4g} "
            f"Holm={entry.p_value_holm:.4g}"
        )

    fig = plot_enrichment(stats)
    save_figure(fig, args.output, args.dpi)
    write_csv(stats_output, CSV_HEADER, stats_rows(stats, args))
    print(f"wrote {args.output} and {stats_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
