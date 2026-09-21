from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from _delta_bias_plotting import (
    evaluate_bias,
    load_bias_module,
    plt,
    save_figure,
    set_publication_style,
    write_csv,
)
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from pyteomics import mass

from msdelta.chemistry import ISOTOPES, NEUTRAL_LOSSES, RESIDUE_MASSES, RESIDUES_AA20

ANNOTATION_STYLE: dict[str, tuple[str, str]] = {
    "b_ion": ("tab:blue", "b-ion ladder"),
    "y_ion": ("tab:red", "y-ion ladder"),
    "neutral_loss": ("tab:orange", "Neutral loss"),
    "isotope": ("tab:green", "Isotope spacing"),
    "residue": ("tab:purple", "Residue mass"),
    "unassigned": ("0.55", "Unassigned"),
}

CSV_HEADER = (
    "lower_index",
    "upper_index",
    "lower_mz",
    "upper_mz",
    "delta_mass",
    "score",
    "max_bias_head",
    "max_head_bias",
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
    max_head: int
    max_bias: float
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
    parser.add_argument("--spectra", type=Path, required=True, help="CSV/JSON/JSONL/Parquet file")
    parser.add_argument("--output", type=Path, required=True, help="figure path (.pdf/.svg/.png)")
    parser.add_argument("--pairs-output", type=Path, default=None, help="annotated pairs CSV")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or xpu")
    parser.add_argument("--row-index", type=int, default=0, help="row to plot when no id is given")
    parser.add_argument("--id-column", default=None, help="column holding the spectrum id")
    parser.add_argument("--spectrum-id", default=None, help="spectrum id to select")
    parser.add_argument("--mz-column", default="mz")
    parser.add_argument("--intensity-column", default="intensity")
    parser.add_argument("--peptide-column", default="peptide")
    parser.add_argument("--charge-column", default="charge")
    parser.add_argument("--top-pairs", type=int, default=12, help="pairs drawn and written")
    parser.add_argument("--top-k-heads", type=int, default=3, help="heads averaged per pair score")
    parser.add_argument("--min-dm", type=float, default=0.5, help="lowest pair Δm in Da")
    parser.add_argument("--max-dm", type=float, default=250.0, help="highest pair Δm in Da")
    parser.add_argument("--mass-tolerance", type=float, default=0.05, help="Δm match tolerance")
    parser.add_argument(
        "--fragment-tolerance", type=float, default=0.05, help="b/y ion match tolerance"
    )
    parser.add_argument("--symmetric", action="store_true", help="fold b(+Δm) with b(-Δm)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default=None, help="figure title")
    return parser.parse_args(argv)


def _maybe_parse_array(value: object) -> object:
    """Parse a JSON-list string such as ``"[100.1, 200.2]"`` into a list."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [float(part) for part in text[1:-1].replace(",", " ").split()]
    return value


def _select_row(
    rows: list[dict[str, object]], row_index: int, id_column: str | None, spectrum_id: str | None
) -> dict[str, object]:
    """Pick a row by id when one is supplied, otherwise by position."""
    if id_column and spectrum_id is not None:
        for row in rows:
            if str(row.get(id_column, "")) == str(spectrum_id):
                return row
        raise SystemExit(f"no row with {id_column}={spectrum_id!r}")
    if not -len(rows) <= row_index < len(rows):
        raise SystemExit(f"--row-index {row_index} is out of range for {len(rows)} rows")
    return rows[row_index]


def load_spectrum_row(
    path: Path, row_index: int, id_column: str | None, spectrum_id: str | None
) -> dict[str, object]:
    """Load one spectrum record from a CSV, JSON, JSONL, or Parquet file."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table = pq.read_table(path)
        rows = table.to_pylist()
    elif suffix == ".jsonl" or suffix == ".ndjson":
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload if isinstance(payload, list) else [payload]
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [dict(record) for record in csv.DictReader(handle)]
    else:
        raise SystemExit(f"unsupported spectra suffix {path.suffix!r}")

    if not rows:
        raise SystemExit(f"{path} contains no rows")
    row = _select_row(rows, row_index, id_column, spectrum_id)
    return {key: _maybe_parse_array(value) for key, value in row.items()}


def extract_peaks(
    row: dict[str, object], mz_column: str, intensity_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return the sorted m/z array and base-peak-normalized intensity."""
    for column in (mz_column, intensity_column):
        if column not in row or row[column] is None:
            raise SystemExit(f"the selected row has no {column!r} column")
    mz = np.asarray(row[mz_column], dtype=np.float64)
    intensity = np.asarray(row[intensity_column], dtype=np.float64)
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


def read_peptide(row: dict[str, object], column: str) -> str | None:
    """Return the unmodified peptide sequence, or None when unusable."""
    value = row.get(column)
    if value is None:
        return None
    peptide = str(value).strip().upper()
    if not peptide or not set(peptide) <= set("ACDEFGHIKLMNPQRSTVWY"):
        return None
    return peptide


def read_charge(row: dict[str, object], column: str) -> int | None:
    """Return the precursor charge, or None when missing or unparsable."""
    value = row.get(column)
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


def charge_scaled(values: dict[str, float], max_charge: int) -> list[tuple[str, float]]:
    """Expand reference masses to the charge-scaled differences they produce."""
    scaled: list[tuple[str, float]] = []
    for name, value in values.items():
        for charge in range(1, max_charge + 1):
            label = name if charge == 1 else f"{name} z{charge}"
            scaled.append((label, value / charge))
    return scaled


def closest_reference(
    delta: float, candidates: list[tuple[str, float]], tolerance: float
) -> tuple[str, float] | None:
    """Return the closest candidate within ``tolerance`` of ``delta``."""
    best: tuple[str, float] | None = None
    best_error = tolerance
    for name, value in candidates:
        error = abs(delta - value)
        if error <= best_error:
            best, best_error = (name, value), error
    return best


def annotate_pair(
    record: PairRecord,
    peptide: str | None,
    assignments: dict[int, set[tuple[str, int, int]]],
    losses: list[tuple[str, float]],
    isotopes: list[tuple[str, float]],
    residues: list[tuple[str, float]],
    tolerance: float,
) -> None:
    """Assign the highest-priority chemical interpretation to ``record`` in place."""
    if peptide is not None:
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

    for kind, candidates in (
        ("neutral_loss", losses),
        ("isotope", isotopes),
        ("residue", residues),
    ):
        hit = closest_reference(record.delta_mass, candidates, tolerance)
        if hit is not None:
            record.annotation_type = kind
            record.annotation_label = hit[0]
            record.reference_mass = hit[1]
            record.mass_error = record.delta_mass - hit[1]
            return


def score_pairs(
    bias_module,
    mz: np.ndarray,
    min_dm: float,
    max_dm: float,
    top_k_heads: int,
    symmetric: bool,
) -> list[PairRecord]:
    """Score every in-window peak pair with one batched bias evaluation."""
    lower_idx, upper_idx = np.triu_indices(mz.size, k=1)
    deltas = mz[upper_idx] - mz[lower_idx]
    keep = (deltas >= min_dm) & (deltas <= max_dm)
    lower_idx, upper_idx, deltas = lower_idx[keep], upper_idx[keep], deltas[keep]
    if deltas.size == 0:
        return []

    bias = evaluate_bias(bias_module, deltas, symmetric=symmetric)
    k = max(1, min(top_k_heads, bias.shape[1]))
    scores = np.sort(bias, axis=1)[:, -k:].mean(axis=1)
    max_heads = np.argmax(bias, axis=1)
    max_bias = bias[np.arange(bias.shape[0]), max_heads]

    return [
        PairRecord(
            lower_index=int(lower_idx[i]),
            upper_index=int(upper_idx[i]),
            lower_mz=float(mz[lower_idx[i]]),
            upper_mz=float(mz[upper_idx[i]]),
            delta_mass=float(deltas[i]),
            score=float(scores[i]),
            max_head=int(max_heads[i]),
            max_bias=float(max_bias[i]),
        )
        for i in range(deltas.size)
    ]


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
    title: str | None,
) -> plt.Figure:
    """Draw the stick spectrum with colored arcs over the selected peak pairs."""
    levels = assign_levels(records)
    n_levels = (max(levels) + 1) if levels else 1
    arc_base, arc_step = 1.08, 0.16
    top = arc_base + arc_step * n_levels + 0.12

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.vlines(mz, 0.0, intensity, color="black", linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=0.6)

    for record, level in zip(records, levels):
        color = ANNOTATION_STYLE[record.annotation_type][0]
        apex = arc_base + arc_step * level
        start = (record.lower_mz, min(intensity[record.lower_index] + 0.02, apex - 0.01))
        end = (record.upper_mz, min(intensity[record.upper_index] + 0.02, apex - 0.01))
        mid_x = 0.5 * (record.lower_mz + record.upper_mz)
        path = MplPath(
            [start, (start[0], apex), (mid_x, apex), (end[0], apex), end],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO],
        )
        ax.add_patch(PathPatch(path, edgecolor=color, facecolor="none", linewidth=0.8, alpha=0.9))
        label = record.annotation_label or f"{record.delta_mass:.2f}"
        ax.annotate(
            label,
            xy=(mid_x, apex),
            xytext=(0, 1.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.5,
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
        ax.legend(handles=handles, loc="upper center", ncols=min(len(handles), 6), frameon=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
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
            record.max_head,
            f"{record.max_bias:.10g}",
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
    if args.top_pairs < 1:
        raise SystemExit("--top-pairs must be at least 1")

    pairs_output = args.pairs_output or args.output.with_name(args.output.stem + "_pairs.csv")
    set_publication_style()

    row = load_spectrum_row(args.spectra, args.row_index, args.id_column, args.spectrum_id)
    mz, intensity = extract_peaks(row, args.mz_column, args.intensity_column)
    peptide = read_peptide(row, args.peptide_column)
    charge = read_charge(row, args.charge_column)
    if peptide is None or charge is None:
        print("peptide or charge unavailable; skipping sequence-specific b/y annotation")
        peptide = None

    max_fragment_charge = max(1, (charge - 1) if charge else 1)
    assignments: dict[int, set[tuple[str, int, int]]] = {}
    if peptide is not None:
        ladders = theoretical_ions(peptide, max_fragment_charge)
        assignments = match_ions(mz, ladders, args.fragment_tolerance)

    bias_module = load_bias_module(args.checkpoint, args.device)
    records = score_pairs(
        bias_module, mz, args.min_dm, args.max_dm, args.top_k_heads, args.symmetric
    )
    if not records:
        raise SystemExit(f"no peak pairs fall inside [{args.min_dm}, {args.max_dm}] Da")

    losses = charge_scaled(NEUTRAL_LOSSES, max_fragment_charge)
    isotopes = list(ISOTOPES.items())
    residues = list(RESIDUES_AA20.items())
    for record in records:
        annotate_pair(
            record,
            peptide,
            assignments,
            losses,
            isotopes,
            residues,
            args.mass_tolerance,
        )

    records.sort(key=lambda item: (item.annotated, item.score), reverse=True)
    selected = records[: args.top_pairs]
    selected.sort(key=lambda item: item.lower_mz)

    fig = plot_spectrum(mz, intensity, selected, args.title)
    save_figure(fig, args.output, args.dpi)
    write_csv(pairs_output, CSV_HEADER, pair_rows(selected))

    annotated = sum(1 for record in selected if record.annotated)
    print(
        f"wrote {args.output} and {pairs_output} "
        f"({len(mz)} peaks, {len(records)} candidate pairs, "
        f"{len(selected)} drawn, {annotated} annotated)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
