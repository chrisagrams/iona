from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from msdelta.chemistry import ISOTOPES, NEUTRAL_LOSSES, RESIDUES_AA20  # noqa: E402
from msdelta.modeling_msdelta import MSDeltaForPreTraining  # noqa: E402

CATEGORY_ISOTOPE = "isotope"
CATEGORY_NEUTRAL_LOSS = "neutral_loss"
CATEGORY_RESIDUE = "residue"

CATEGORY_COLORS: dict[str, str] = {
    CATEGORY_ISOTOPE: "tab:green",
    CATEGORY_NEUTRAL_LOSS: "tab:orange",
    CATEGORY_RESIDUE: "tab:purple",
}

CATEGORY_LABELS: dict[str, str] = {
    CATEGORY_ISOTOPE: "Isotope spacings",
    CATEGORY_NEUTRAL_LOSS: "Neutral losses",
    CATEGORY_RESIDUE: "Residue masses",
}

GUIDE_RESIDUES: tuple[str, ...] = (
    "G",
    "A",
    "S",
    "P",
    "V",
    "L/I",
    "N",
    "D",
    "K",
    "E",
    "M",
    "F",
    "R",
    "Y",
    "W",
)

_EVAL_CHUNK = 65_536


@dataclass(frozen=True)
class ChemicalReference:
    """A labeled chemical mass difference used as a reference marker."""

    name: str
    mass: float
    category: str

    @property
    def color(self) -> str:
        """Return the plotting color for this reference's category."""
        return CATEGORY_COLORS[self.category]


def chemical_references() -> list[ChemicalReference]:
    """Return every labeled chemical mass difference, ordered by mass."""
    records = [ChemicalReference(name, value, CATEGORY_ISOTOPE) for name, value in ISOTOPES.items()]
    records += [
        ChemicalReference(name, value, CATEGORY_NEUTRAL_LOSS)
        for name, value in NEUTRAL_LOSSES.items()
    ]
    records += [
        ChemicalReference(name, value, CATEGORY_RESIDUE) for name, value in RESIDUES_AA20.items()
    ]
    return sorted(records, key=lambda ref: ref.mass)


def references_in_range(lo: float, hi: float) -> list[ChemicalReference]:
    """Return the references whose mass falls inside ``[lo, hi]``."""
    return [ref for ref in chemical_references() if lo <= ref.mass <= hi]


def resolve_device(requested: str) -> str:
    """Resolve ``auto``/``cpu``/``cuda``/``xpu`` to a concrete torch device string."""
    name = requested.strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            return "xpu"
        return "cpu"
    if name in ("cpu", "cuda", "xpu"):
        return name
    raise ValueError(f"unknown device {requested!r}")


def load_bias_module(checkpoint: Path, device: str):
    """Load a pretraining checkpoint and return its evaluation-mode bias module."""
    model = MSDeltaForPreTraining.from_pretrained(str(checkpoint))
    model.eval()
    model.to(resolve_device(device))
    return model.msdelta.bias_module


@torch.no_grad()
def evaluate_bias(bias_module, masses: np.ndarray, *, symmetric: bool) -> np.ndarray:
    """Evaluate every head's bias curve on ``masses``.

    Returns an array of shape ``[n_masses, n_heads]``. With ``symmetric`` the
    signed curve is folded as ``0.5 * (b(+dm) + b(-dm))``, which removes the
    sign convention the model is free to pick per head.
    """
    values = np.asarray(masses, dtype=np.float64).reshape(-1)
    if values.size == 0:
        n_heads = int(bias_module.n_heads)
        return np.zeros((0, n_heads), dtype=np.float64)

    device = next(bias_module.parameters()).device
    out: list[np.ndarray] = []
    for start in range(0, values.size, _EVAL_CHUNK):
        chunk = torch.as_tensor(
            values[start : start + _EVAL_CHUNK], dtype=torch.float32, device=device
        )
        curve = bias_module.evaluate(chunk)
        if symmetric:
            curve = 0.5 * (curve + bias_module.evaluate(-chunk))
        out.append(curve.float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def local_peak_scores(
    bias_module,
    masses: np.ndarray,
    flank_offsets: Sequence[float],
    *,
    symmetric: bool,
) -> np.ndarray:
    """Score each mass as ``bias(m) - median(bias(m ± offset))`` per head.

    The flanking median turns the raw bias into a local-peak statistic, so a
    head that simply runs high over a whole mass region does not score.
    Returns an array of shape ``[n_masses, n_heads]``.
    """
    centers = np.asarray(masses, dtype=np.float64).reshape(-1)
    offsets = np.asarray(list(flank_offsets), dtype=np.float64).reshape(-1)
    if offsets.size == 0:
        raise ValueError("at least one flank offset is required")

    flanks = np.concatenate([centers + off for off in offsets] + [centers - off for off in offsets])
    values = evaluate_bias(bias_module, np.concatenate([centers, flanks]), symmetric=symmetric)
    n = centers.size
    center_values = values[:n]
    flank_values = values[n:].reshape(2 * offsets.size, n, values.shape[1])
    return center_values - np.median(flank_values, axis=0)


def set_publication_style() -> None:
    """Apply a restrained, print-oriented Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    """Save ``fig`` to ``output``, honoring the PDF/SVG/PNG suffix."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Write ``rows`` to ``path`` with a leading ``header`` line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
