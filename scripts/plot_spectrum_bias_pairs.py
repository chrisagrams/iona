from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path

import numpy as np
from _delta_bias_plotting import (
    evaluate_bias,
    load_bias_module,
    plt,
    save_figure,
    set_publication_style,
    write_csv,
)
from datasets import load_dataset
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from pyteomics import mass

from msdelta.chemistry import RESIDUE_MASSES

DATASET_ID = "chrisagrams/MSConsensus-100M"
DATASET_SPLIT = "validation"

ANNOTATION_STYLE: dict[str, tuple[str, str]] = {
    "b_ion": ("#1358b0", "Observed b-ion ladder transition"),
    "y_ion": ("#d7191c", "Observed y-ion ladder transition"),
}

_CALLOUT = (
    "Only transitions whose two endpoints match\n"
    "the peptide's theoretical b/y ladder are shown.\n"
    "Hn identifies the attention-bias head with the\n"
    "largest learned score at that observed Δm."
)

CSV_HEADER = (
    "lower_index",
    "upper_index",
    "lower_mz",
    "upper_mz",
    "delta_mass",
    "score",
    "bias_head",
    "head_bias",
    "annotation_type",
    "annotation_label",
    "reference_mass",
    "mass_error",
)


@dataclass
class PairRecord:
    """One scored peak pair and its chemical interpretation."""

    lower_index: int
    upper_index: int
    lower_mz: float
    upper_mz: float
    delta_mass: float
    score: float
    head: int
    head_bias: float
    annotation_type: str = "unassigned"
    annotation_label: str = ""
    reference_mass: float | None = None
    mass_error: float | None = None

    @property
    def annotated(self) -> bool:
        """Return whether the pair received a chemical assignment."""
        return self.annotation_type != "unassigned"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Figure C command line."""
    parser = argparse.ArgumentParser(description="Draw bias-favored peak pairs over a single spectrum (Figure C).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretraining checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="figure path (.pdf/.svg/.png)")
    parser.add_argument("--pairs-output", type=Path, default=None, help="annotated pairs CSV")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or xpu")
    parser.add_argument("--row-index", type=int, default=0, help="validation-split row to plot")
    parser.add_argument("--table-pairs", type=int, default=10, help="pairs shown in table")
    parser.add_argument("--min-dm", type=float, default=0.5, help="lowest pair Δm in Da")
    parser.add_argument("--max-dm", type=float, default=250.0, help="highest pair Δm in Da")
    parser.add_argument("--min-bias", type=float, default=0.0, help="lowest per-head bias to draw")
    parser.add_argument(
        "--fragment-tolerance", type=float, default=0.01, help="b/y ion match tolerance"
    )
    parser.add_argument("--symmetric", action="store_true", help="fold b(+Δm) with b(-Δm)")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def load_spectrum_row(row_index: int) -> dict[str, object]:
    """Stream one evaluation spectrum from MSConsensus-100M."""
    if row_index < 0:
        raise SystemExit("--row-index must be non-negative when streaming the dataset")
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
    try:
        return next(islice(dataset, row_index, row_index + 1))
    except StopIteration:
        raise SystemExit(f"--row-index {row_index} is past the end of the dataset") from None


def extract_peaks(row: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return the sorted m/z array and base-peak-normalized intensity."""
    for column in ("mz", "intensity"):
        if column not in row or row[column] is None:
            raise SystemExit(f"the selected row has no {column!r} column")
    mz = np.asarray(row["mz"], dtype=np.float64)
    intensity = np.asarray(row["intensity"], dtype=np.float64)
    if mz.ndim != 1 or intensity.ndim != 1:
        raise SystemExit("m/z and intensity must be one-dimensional arrays")
    if mz.shape != intensity.shape:
        raise SystemExit(f"m/z has {mz.size} values but intensity has {intensity.size}")
    if mz.size < 2:
        raise SystemExit("at least two peaks are required")
    if not (np.isfinite(mz).all() and np.isfinite(intensity).all()):
        raise SystemExit("m/z and intensity must be finite")
    if (intensity < 0).any():
        raise SystemExit("intensities must be non-negative")

    order = np.argsort(mz, kind="stable")
    mz = mz[order]
    intensity = intensity[order]
    base = intensity.max()
    if base <= 0:
        raise SystemExit("the spectrum has no positive intensity")
    return mz, intensity / base


def read_peptide(row: dict[str, object]) -> str | None:
    """Return the unmodified peptide sequence, or None when unusable."""
    value = row.get("peptide")
    if value is None:
        return None
    peptide = str(value).strip().upper()
    if not peptide or not set(peptide) <= set("ACDEFGHIKLMNPQRSTVWY"):
        return None
    return peptide


def read_charge(row: dict[str, object]) -> int | None:
    """Return the precursor charge, or None when missing or unparsable."""
    value = row.get("charge")
    if value is None or value == "":
        return None
    try:
        charge = int(float(str(value)))
    except ValueError:
        return None
    return charge if charge > 0 else None


def theoretical_ions(peptide: str, max_charge: int) -> dict[tuple[str, int], list[float]]:
    """Return b- and y-ion m/z ladders keyed by ``(series, charge)``.

    Index ``i`` of each list is the ion with ``i + 1`` residues.
    """
    ladders: dict[tuple[str, int], list[float]] = {}
    for charge in range(1, max_charge + 1):
        ladders[("b", charge)] = [
            float(mass.fast_mass(peptide[:i], ion_type="b", charge=charge))
            for i in range(1, len(peptide))
        ]
        ladders[("y", charge)] = [
            float(mass.fast_mass(peptide[-i:], ion_type="y", charge=charge))
            for i in range(1, len(peptide))
        ]
    return ladders


def match_ions(
    mz: np.ndarray, ladders: dict[tuple[str, int], list[float]], tolerance: float
) -> dict[int, set[tuple[str, int, int]]]:
    """Map each peak index to the ``(series, charge, ladder_index)`` ions it matches."""
    assignments: dict[int, set[tuple[str, int, int]]] = {}
    for (series, charge), values in ladders.items():
        for ladder_index, target in enumerate(values):
            position = int(np.argmin(np.abs(mz - target)))
            if abs(mz[position] - target) <= tolerance:
                assignments.setdefault(position, set()).add((series, charge, ladder_index))
    return assignments


def ladder_annotation(
    peptide: str,
    lower: set[tuple[str, int, int]],
    upper: set[tuple[str, int, int]],
    series: str,
    delta: float,
) -> tuple[str, str, float, float] | None:
    """Describe a same-series, same-charge ladder step spanning one or two residues.

    Returns ``(annotation_type, label, reference_mass, mass_error)`` or None.
    """
    for low_series, low_charge, low_index in sorted(lower):
        if low_series != series:
            continue
        for high_series, high_charge, high_index in sorted(upper):
            if high_series != series or high_charge != low_charge:
                continue
            gap = high_index - low_index
            if gap not in (1, 2):
                continue
            if series == "b":
                residues = peptide[low_index + 1 : high_index + 1]
            else:
                residues = peptide[-(high_index + 1) : -(low_index + 1)]
            reference = sum(RESIDUE_MASSES[aa] for aa in residues) / low_charge
            charge_tag = "" if low_charge == 1 else f"{low_charge}+"
            step = "" if gap == 1 else " (skip)"
            label = f"{series}{low_index + 1}→{series}{high_index + 1}{charge_tag} {residues}{step}"
            return (f"{series}_ion", label.strip(), reference, delta - reference)
    return None


def annotate_ladder_pair(
    record: PairRecord,
    peptide: str,
    assignments: dict[int, set[tuple[str, int, int]]],
) -> None:
    """Annotate ``record`` only when both endpoints form a theoretical b/y ladder step."""
    lower = assignments.get(record.lower_index, set())
    upper = assignments.get(record.upper_index, set())
    for series in ("b", "y"):
        hit = ladder_annotation(peptide, lower, upper, series, record.delta_mass)
        if hit is not None:
            record.annotation_type = hit[0]
            record.annotation_label = hit[1]
            record.reference_mass = hit[2]
            record.mass_error = hit[3]
            return


def score_candidate_pairs(
    bias_module,
    mz: np.ndarray,
    min_dm: float,
    max_dm: float,
    symmetric: bool,
) -> tuple[list[PairRecord], np.ndarray]:
    """Score every in-window peak pair, retaining the per-head bias matrix."""
    lower_idx, upper_idx = np.triu_indices(mz.size, k=1)
    deltas = mz[upper_idx] - mz[lower_idx]
    keep = (deltas >= min_dm) & (deltas <= max_dm)
    lower_idx, upper_idx, deltas = lower_idx[keep], upper_idx[keep], deltas[keep]
    if deltas.size == 0:
        return [], np.empty((0, 0), dtype=np.float64)

    bias = evaluate_bias(bias_module, deltas, symmetric=symmetric)
    records = [
        PairRecord(
            lower_index=int(lower_idx[i]),
            upper_index=int(upper_idx[i]),
            lower_mz=float(mz[lower_idx[i]]),
            upper_mz=float(mz[upper_idx[i]]),
            delta_mass=float(deltas[i]),
            score=0.0,
            head=0,
            head_bias=0.0,
        )
        for i in range(deltas.size)
    ]
    return records, bias


def select_ladder_transitions(
    records: list[PairRecord], bias: np.ndarray, min_bias: float
) -> list[PairRecord]:
    """Attach each supported ladder transition to its highest-bias attention head."""
    selected: list[PairRecord] = []
    for index, record in enumerate(records):
        if not record.annotated:
            continue
        head_index = int(np.argmax(bias[index]))
        head_bias = float(bias[index, head_index])
        if head_bias < min_bias:
            continue
        selected.append(
            replace(
                record,
                score=head_bias,
                head=head_index + 1,
                head_bias=head_bias,
            )
        )
    return selected


def assign_levels(records: list[PairRecord]) -> list[int]:
    """Stack overlapping arcs on progressively higher levels to limit collisions."""
    levels: list[float] = []
    assigned: list[int] = []
    for record in sorted(records, key=lambda item: item.lower_mz):
        for level, occupied_until in enumerate(levels):
            if record.lower_mz > occupied_until:
                levels[level] = record.upper_mz
                assigned.append(level)
                break
        else:
            levels.append(record.upper_mz)
            assigned.append(len(levels) - 1)
    order = sorted(range(len(records)), key=lambda i: records[i].lower_mz)
    result = [0] * len(records)
    for position, index in enumerate(order):
        result[index] = assigned[position]
    return result


def plot_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    records: list[PairRecord],
    peptide: str | None = None,
    charge: int | None = None,
    table_pairs: int = 4,
) -> plt.Figure:
    """Draw the spectrum, relationship arcs, compact table, and interpretation."""
    levels = assign_levels(records)
    n_levels = (max(levels) + 1) if levels else 1
    arc_base, arc_step = 1.03, 0.13
    top = arc_base + arc_step * n_levels + 0.1

    fig = plt.figure(figsize=(10.2, 5.1))
    layout = fig.add_gridspec(
        2,
        5,
        left=0.07,
        right=0.985,
        bottom=0.07,
        top=0.82,
        height_ratios=(2.15, 1.0),
        hspace=0.45,
        wspace=0.28,
    )
    ax = fig.add_subplot(layout[0, :])
    ax.vlines(mz, 0.0, intensity, color="black", linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=0.6)

    for record, level in zip(records, levels):
        color = ANNOTATION_STYLE[record.annotation_type][0]
        apex = arc_base + arc_step * level
        start_y = min(float(intensity[record.lower_index]) + 0.025, apex - 0.05)
        end_y = min(float(intensity[record.upper_index]) + 0.025, apex - 0.05)
        span_fraction = (record.upper_mz - record.lower_mz) / max(float(np.ptp(mz)), 1.0)
        radius = -(0.22 + min(0.35, 0.7 * span_fraction) + 0.08 * level)
        arrow = FancyArrowPatch(
            (record.lower_mz, start_y),
            (record.upper_mz, end_y),
            connectionstyle=f"arc3,rad={radius}",
            arrowstyle="-|>",
            mutation_scale=5,
            color=color,
            linewidth=0.8,
            alpha=0.95,
        )
        ax.add_patch(arrow)
        mid_x = 0.5 * (record.lower_mz + record.upper_mz)
        explanation = record.annotation_label or f"Δm={record.delta_mass:.2f}"
        label = f"H{record.head}: {explanation}"
        ax.annotate(
            label,
            xy=(mid_x, apex - 0.01),
            ha="center",
            va="bottom",
            fontsize=6,
            color=color,
        )

    span = float(mz[-1] - mz[0]) or 1.0
    ax.set_xlim(float(mz[0]) - 0.03 * span, float(mz[-1]) + 0.03 * span)
    ax.set_ylim(0.0, top)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.spines["left"].set_bounds(0.0, 1.0)

    used = [key for key in ANNOTATION_STYLE if any(r.annotation_type == key for r in records)]
    if used:
        handles = [
            Line2D(
                [],
                [],
                color=ANNOTATION_STYLE[key][0],
                linewidth=1.2,
                label=ANNOTATION_STYLE[key][1],
            )
            for key in used
        ]
        legend = ax.legend(
            handles=handles,
            loc="upper right",
            title="Sequence-supported transitions",
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            borderpad=0.5,
            fontsize=7,
        )
        legend.get_frame().set_linewidth(0.6)

    table_ax = fig.add_subplot(layout[1, :3])
    table_ax.axis("off")
    table_ax.set_title("Observed peptide ladder transitions", loc="left", fontsize=8, weight="bold")
    ranked = sorted(records, key=lambda record: (record.lower_mz, record.upper_mz))
    table_records = ranked[: max(table_pairs, 0)]
    cells = [
        [
            f"H{record.head}",
            f"{record.lower_mz:.1f}",
            f"{record.upper_mz:.1f}",
            f"{record.delta_mass:.2f}",
            record.annotation_label or "Unassigned",
        ]
        for record in table_records
    ]
    table = table_ax.table(
        cellText=cells,
        colLabels=["Head", r"$m/z_i$", r"$m/z_j$", r"$\Delta m$ (Da)", "Explanation"],
        cellLoc="center",
        colWidths=[0.10, 0.13, 0.13, 0.16, 0.48],
        bbox=(0.0, 0.0, 1.0, 0.92),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, _column), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f1f1f1")

    callout_ax = fig.add_subplot(layout[1, 3:])
    callout_ax.axis("off")
    callout_ax.text(
        0.03,
        0.58,
        _CALLOUT,
        transform=callout_ax.transAxes,
        va="center",
        ha="left",
        fontsize=7,
        style="italic",
        linespacing=1.3,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#edf4ff", "edgecolor": "#b7c7df"},
    )

    if peptide:
        charge_text = f"   (z = {charge})" if charge is not None else ""
        fig.text(0.08, 0.89, f"Peptide:   {peptide}{charge_text}", fontsize=8, weight="bold")
    return fig


def pair_rows(records: list[PairRecord]) -> list[list[object]]:
    """Render the selected pairs as CSV rows."""
    return [
        [
            record.lower_index,
            record.upper_index,
            f"{record.lower_mz:.6f}",
            f"{record.upper_mz:.6f}",
            f"{record.delta_mass:.6f}",
            f"{record.score:.10g}",
            record.head,
            f"{record.head_bias:.10g}",
            record.annotation_type,
            record.annotation_label,
            "" if record.reference_mass is None else f"{record.reference_mass:.6f}",
            "" if record.mass_error is None else f"{record.mass_error:.6f}",
        ]
        for record in records
    ]


def main(argv: list[str] | None = None) -> int:
    """Build and save Figure C along with its annotated-pairs CSV."""
    args = parse_args(argv)
    if args.max_dm <= args.min_dm:
        raise SystemExit("--max-dm must exceed --min-dm")
    if args.table_pairs < 0:
        raise SystemExit("--table-pairs must be non-negative")

    pairs_output = args.pairs_output or args.output.with_name(args.output.stem + "_pairs.csv")
    set_publication_style()

    row = load_spectrum_row(args.row_index)
    mz, intensity = extract_peaks(row)
    peptide = read_peptide(row)
    charge = read_charge(row)
    if peptide is None:
        raise SystemExit("the selected spectrum has no usable unmodified peptide sequence")
    if charge is None:
        raise SystemExit("the selected spectrum has no usable precursor charge")

    max_fragment_charge = max(1, charge - 1)
    ladders = theoretical_ions(peptide, max_fragment_charge)
    assignments = match_ions(mz, ladders, args.fragment_tolerance)

    bias_module = load_bias_module(args.checkpoint, args.device)
    candidates, bias = score_candidate_pairs(
        bias_module, mz, args.min_dm, args.max_dm, args.symmetric
    )
    if not candidates:
        raise SystemExit(f"no peak pairs fall inside [{args.min_dm}, {args.max_dm}] Da")

    for record in candidates:
        annotate_ladder_pair(record, peptide, assignments)

    selected = select_ladder_transitions(candidates, bias, args.min_bias)
    if not selected:
        raise SystemExit("no observed peak pairs support a theoretical b/y ladder transition")
    selected.sort(key=lambda item: (item.lower_mz, item.upper_mz))

    fig = plot_spectrum(
        mz,
        intensity,
        selected,
        peptide=peptide,
        charge=charge,
        table_pairs=args.table_pairs,
    )
    save_figure(fig, args.output, args.dpi)
    write_csv(pairs_output, CSV_HEADER, pair_rows(selected))

    print(
        f"wrote {args.output} and {pairs_output} "
        f"({len(mz)} peaks, {len(candidates)} candidate pairs, {bias.shape[1]} heads, "
        f"{len(selected)} sequence-supported transitions drawn)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
