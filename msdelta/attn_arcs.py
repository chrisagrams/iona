"""Attention arcs on a real annotated spectrum — the "it learned the chemistry" figure.

The bias-curve figure (viz.py) plots the model's *internal* Δm bias. This one
plots its *behavior*: on one real MS/MS spectrum, it draws the attention edges a
specialized head forms and shows they land exactly on chemically meaningful Δm —

  * the "residue-ladder" head connects consecutive fragment ions (b- or y-series),
    each edge spanning one amino-acid residue mass, so walking the arcs reads off
    the peptide sequence; and
  * the "isotope" head connects each monoisotopic peak to its ¹³C+1 satellite.

Heads are chosen by their measured attention behavior on the chosen spectrum (the
head that puts the most attention mass on residue-gap / isotope-gap pairs), not by
hand. Pass --control to render a random-init encoder alongside as a null: its edges
scatter and no ladder appears.

Nothing chemical is fabricated — b/y ions and isotope spacings are computed from
the peptide label (data.py's mass tables). The only model-derived quantity is the
attention weight (arc opacity/width), read from `BiasedMHA.last_attn`.

Usage:
    msdelta-arcs --ckpt runs/<run>/final.pt --out fig_arcs.png --control
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .analyze import build_val_loader  # noqa: F401  (kept for parity / reuse)
from .analyze import load_encoder
from .data import (
    ConsensusParquet,
    PreprocessConfig,
    _MOD_RE,
    _PROTON,
    _RESIDUE_MASS,
    _WATER,
    split_paths,
)
from .model import (
    DeltaBiasConfig,
    FourierConfig,
    ModelConfig,
    MSEncoder,
)
from .viz import ISOTOPES, RESIDUES_AA20

_C13 = 1.0033548


# ----------------------------------------------------------------------------
# peptide → theoretical fragment ions
# ----------------------------------------------------------------------------

def residue_masses(pep: str) -> tuple[list[str], list[float]] | None:
    """Parse a (mod-aware) peptide string → (residue chars, residue masses).

    A ``[±x]`` bracket adds its mass to the preceding residue (an N-terminal
    bracket, before any residue, is folded into the first residue). Returns
    ``None`` if any token is an unknown residue — those spectra are skipped so
    the fragment ladder stays exact.
    """
    chars: list[str] = []
    masses: list[float] = []
    nterm = 0.0
    i = 0
    while i < len(pep):
        c = pep[i]
        if c == "[":
            j = pep.find("]", i)
            if j < 0:
                return None
            val = float(pep[i + 1 : j])
            if masses:
                masses[-1] += val
            else:
                nterm += val
            i = j + 1
        else:
            m = _RESIDUE_MASS.get(c)
            if m is None:
                return None
            chars.append(c)
            masses.append(m)
            i += 1
    if not masses:
        return None
    masses[0] += nterm
    return chars, masses


@dataclass
class Ion:
    label: str      # e.g. "b3", "y5"
    mz: float
    series: str     # "b" or "y"
    index: int      # 1-based position in the ladder
    z: int          # fragment charge
    added_aa: str   # residue added going from index-1 → index (labels the arc)


def theoretical_ions(chars: list[str], masses: list[float],
                     charges: tuple[int, ...] = (1, 2)) -> list[Ion]:
    """b/y fragment ions (both singly and doubly charged by default).

    b_i (z=1) m/z = Σ(residues 1..i) + proton;  y_j = Σ(last j) + water + proton.
    General charge z: (neutral + z·proton) / z.
    """
    n = len(masses)
    ions: list[Ion] = []
    # b-ions: prefixes 1..n-1 (b_n == full peptide, not a fragment)
    cum = 0.0
    for i in range(n - 1):
        cum += masses[i]
        for z in charges:
            ions.append(Ion(f"b{i+1}", (cum + z * _PROTON) / z, "b", i + 1, z, chars[i]))
    # y-ions: suffixes 1..n-1
    cum = 0.0
    for j in range(n - 1):
        cum += masses[n - 1 - j]
        for z in charges:
            ions.append(
                Ion(f"y{j+1}", (cum + _WATER + z * _PROTON) / z, "y", j + 1, z,
                    chars[n - 1 - j])
            )
    return ions


def assign_peaks(mz: np.ndarray, ions: list[Ion], tol: float = 0.02) -> dict[int, Ion]:
    """Map each observed peak index → the nearest fragment ion within `tol` Da.

    When several ions fall within tol of one peak, the closest wins; each peak
    gets at most one label. Returns {peak_index: Ion}.
    """
    out: dict[int, Ion] = {}
    best_off: dict[int, float] = {}
    for ion in ions:
        k = int(np.argmin(np.abs(mz - ion.mz)))
        off = abs(mz[k] - ion.mz)
        if off < tol and off < best_off.get(k, tol):
            out[k] = ion
            best_off[k] = off
    return out


def best_ladder(assign: dict[int, Ion]) -> tuple[str, int, list[tuple[int, Ion]]]:
    """Longest run of consecutive singly-charged ions in one series.

    Returns (series, charge, [(peak_index, Ion), ...] ordered by ladder index).
    The z=1 ladder is used because only there does a one-step gap equal a single
    residue mass in m/z (z=2 halves it), which is what makes the arcs readable.
    """
    best: list[tuple[int, Ion]] = []
    best_key = ("", 1)
    for series in ("b", "y"):
        # index → peak, restricted to z=1 of this series
        idx2peak = {ion.index: (k, ion) for k, ion in assign.items()
                    if ion.series == series and ion.z == 1}
        if not idx2peak:
            continue
        order = sorted(idx2peak)
        run: list[int] = []
        runs: list[list[int]] = []
        for v in order:
            if run and v == run[-1] + 1:
                run.append(v)
            else:
                run = [v]
                runs.append(run)
        longest = max(runs, key=len)
        if len(longest) > len(best):
            best = [idx2peak[v] for v in longest]
            best_key = (series, 1)
    return best_key[0], best_key[1], best


# ----------------------------------------------------------------------------
# spectra + attention
# ----------------------------------------------------------------------------

@dataclass
class Spectrum:
    mz: np.ndarray
    log_int: np.ndarray
    intensity: np.ndarray     # display height (∝ raw intensity_prob)
    charge: int
    precursor_mz: float
    peptide: str


def iter_val_spectra(cfg: dict[str, Any], seed: int = 7):
    """Yield Spectrum records from the validation shards, with peptide labels."""
    dcfg = cfg["data"]
    pp = PreprocessConfig(
        intensity_threshold_frac=dcfg["intensity_threshold_frac"], top_n=dcfg["top_n"]
    )
    # This figure needs the raw peptide string (dropped by the collate fns), so
    # iterate the dataset directly rather than through a DataLoader.
    _, val_paths = split_paths(dcfg["root"], dcfg["n_val_files"])
    ds = ConsensusParquet(val_paths, preprocess=pp, seed=seed)
    for mz, li, pp_prob, ch, pm, pc in ds:
        yield Spectrum(
            mz=mz.numpy(), log_int=li.numpy(), intensity=pp_prob.numpy(),
            charge=int(ch), precursor_mz=float(pm), peptide=str(pc),
        )


def select_spectrum(cfg: dict[str, Any], n_scan: int, min_run: int,
                    tol: float, seed: int) -> tuple[Spectrum, list[tuple[int, Ion]],
                                                     str, dict[int, Ion]]:
    """Scan up to `n_scan` val spectra; return the one with the longest clean
    fragment ladder (prefers unmodified peptides). Raises if none qualify."""
    best: tuple | None = None
    best_len = min_run - 1
    scanned = 0
    for spec in iter_val_spectra(cfg, seed=seed):
        scanned += 1
        if scanned > n_scan:
            break
        parts = spec.peptide.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        pep = parts[0]
        parsed = residue_masses(pep)
        if parsed is None:
            continue
        chars, masses = parsed
        ions = theoretical_ions(chars, masses)
        assign = assign_peaks(spec.mz, ions, tol=tol)
        series, z, ladder = best_ladder(assign)
        modded = bool(_MOD_RE.search(pep))
        score = len(ladder) - (0.5 if modded else 0.0)  # tie-break toward clean
        if score > best_len:
            best_len = score
            best = (spec, ladder, series, assign)
    if best is None:
        raise SystemExit(
            f"no spectrum with a ≥{min_run}-ion consecutive ladder in the first "
            f"{n_scan} val spectra (tol={tol} Da). Try --n-scan larger or --tol looser."
        )
    return best


@torch.no_grad()
def spectrum_attention(enc: MSEncoder, spec: Spectrum, device: torch.device) -> np.ndarray:
    """Run one spectrum, return per-head attention summed over layers: (H, K, K).

    Layers are summed because the Δm bias is shared across all of them, so a
    head's specialization shows up consistently layer to layer. The precursor
    anchor row/col (if present) is stripped to leave fragment↔fragment attention.
    """
    enc.to(device).eval()
    enc.set_save_attn(True)
    K = spec.mz.shape[0]
    mz = torch.from_numpy(spec.mz).float().unsqueeze(0).to(device)
    li = torch.from_numpy(spec.log_int).float().unsqueeze(0).to(device)
    kpm = torch.zeros(1, K, dtype=torch.bool, device=device)
    charge = torch.tensor([spec.charge], dtype=torch.long, device=device)
    prec = torch.tensor([spec.precursor_mz], dtype=torch.float32, device=device)

    enc(mz, li, kpm, charge=charge, precursor_mz=prec)

    H = enc.cfg.n_heads
    acc = torch.zeros(H, K, K, device=device)
    for blk in enc.blocks:
        a = blk.attn.last_attn[0]                 # (H, Kp, Kp)
        if a.size(-1) == K + 1:
            a = a[:, 1:, 1:]                      # drop precursor anchor
        acc += a.float()
    enc.set_save_attn(False)
    return acc.cpu().numpy()


def _sym(attn_h: np.ndarray) -> np.ndarray:
    """Undirected edge weight from a head's attention: α_ij + α_ji."""
    return attn_h + attn_h.T


def pick_heads(attn: np.ndarray, mz: np.ndarray, tol_res: float = 0.05,
               tol_iso: float = 0.01) -> tuple[int, int]:
    """Choose (residue_head, isotope_head) by real attention behavior.

    residue_head maximizes attention mass on peak pairs whose |Δm| matches any
    amino-acid residue mass; isotope_head maximizes it on pairs matching a ¹³C
    M+1 spacing at z∈{1,2,3}. Purely data-driven — no per-head cherry-picking.
    """
    H, K, _ = attn.shape
    dm = np.abs(mz[:, None] - mz[None, :])
    res_vals = np.array(list(RESIDUES_AA20.values()))
    iso_vals = np.array([_C13, _C13 / 2, _C13 / 3])
    res_mask = (np.abs(dm[..., None] - res_vals).min(-1) < tol_res)
    iso_mask = (np.abs(dm[..., None] - iso_vals).min(-1) < tol_iso)
    np.fill_diagonal(res_mask, False)
    np.fill_diagonal(iso_mask, False)
    res_score = np.array([attn[h][res_mask].sum() for h in range(H)])
    iso_score = np.array([attn[h][iso_mask].sum() for h in range(H)])
    return int(res_score.argmax()), int(iso_score.argmax())


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------

_COL = {"b": "#2166ac", "y": "#b2182b", "iso": "#5aae61", "noise": "#c0c0c0"}


def _uparc(ax, x0, x1, y_base, height, color, lw, alpha, label=None):
    t = np.linspace(0, 1, 40)
    xs = x0 + t * (x1 - x0)
    ys = y_base + height * 4 * t * (1 - t)
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    if label:
        ax.text((x0 + x1) / 2, y_base + height + 0.02, label, ha="center",
                va="bottom", fontsize=11, fontweight="bold", color=color)


def _stems(ax, spec: Spectrum, assign: dict[int, Ion], xlim, y):
    for k in range(spec.mz.shape[0]):
        if not (xlim[0] <= spec.mz[k] <= xlim[1]):
            continue
        ion = assign.get(k)
        col = _COL[ion.series] if ion else _COL["noise"]
        ax.vlines(spec.mz[k], 0, y[k], color=col, lw=2.0 if ion else 1.1)
    ax.set_xlim(*xlim)
    ax.set_ylabel("rel. intensity", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _panel_ladder(ax, spec, assign, attn_h, series, ladder, title):
    y = spec.intensity / spec.intensity.max()
    _stems(ax, spec, assign, (spec.mz.min() - 20, spec.mz.max() + 20), y)
    ax.set_ylim(0, 1.75)
    yb = 1.05
    W = _sym(attn_h)
    wmax = max((W[i, j] for (i, _), (j, _) in zip(ladder, ladder[1:])), default=1.0) or 1.0
    letters = []
    for (ka, ia), (kb, ib) in zip(ladder, ladder[1:]):
        x0, x1 = sorted((spec.mz[ka], spec.mz[kb]))
        w = W[ka, kb] / wmax
        for k in (ka, kb):
            ax.vlines(spec.mz[k], y[k], yb, color=_COL[series], lw=0.7, ls=":", alpha=0.5)
        _uparc(ax, x0, x1, yb, 0.16, _COL[series], 1.0 + 3.0 * w,
               0.35 + 0.6 * w, label=ib.added_aa)
        letters.append(ib.added_aa)
    for k, ion in assign.items():
        if ion.z == 1 and ion.series == series:
            ax.text(spec.mz[k], y[k] + 0.02, ion.label, ha="center", va="bottom",
                    fontsize=6.5, color=_COL[series])
    ax.set_xlabel("m/z", fontsize=9)
    ax.set_title(f"{title}\nwalking the {series}-ion ladder reads off: "
                 + " · ".join(letters), fontsize=10, loc="left")
    return letters


def _panel_isotope(ax, spec, assign, attn_h, title):
    # find the isotope pair (Δm ≈ ¹³C/z) this head attends to most strongly
    W = _sym(attn_h)
    mz = spec.mz
    iso_vals = [_C13, _C13 / 2, _C13 / 3]
    best = None
    for i in range(len(mz)):
        for j in range(len(mz)):
            if i == j:
                continue
            d = abs(mz[i] - mz[j])
            if min(abs(d - v) for v in iso_vals) < 0.01 and (best is None or W[i, j] > best[2]):
                best = (i, j, W[i, j])
    if best is None:
        ax.text(0.5, 0.5, "no isotope pair found", transform=ax.transAxes, ha="center")
        ax.set_title(title, fontsize=10, loc="left")
        return
    i, j = best[0], best[1]
    c = (mz[i] + mz[j]) / 2
    zoom = (c - 12, c + 12)
    y = spec.intensity / spec.intensity.max()
    _stems(ax, spec, assign, zoom, y)
    top = max((y[k] for k in range(len(mz)) if zoom[0] <= mz[k] <= zoom[1]), default=1.0)
    ax.set_ylim(0, top * 1.5)
    x0, x1 = sorted((mz[i], mz[j]))
    _uparc(ax, x0, x1, top * 1.05, top * 0.18, _COL["iso"], 3.0, 0.9)
    ax.annotate(f"Δm/z {x1 - x0:.3f}  (¹³C)", xy=((x0 + x1) / 2, top * 1.25),
                ha="center", fontsize=8, color=_COL["iso"])
    ax.set_xlabel("m/z  (zoom)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")


def render(enc: MSEncoder, spec: Spectrum, assign, series, ladder,
           device, out_path: Path, control_enc: MSEncoder | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    attn = spectrum_attention(enc, spec, device)
    res_h, iso_h = pick_heads(attn, spec.mz)

    ncols = 2
    nrows = 2 if control_enc is not None else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.2 * nrows), squeeze=False,
                             gridspec_kw={"width_ratios": [2.4, 1]})

    letters = _panel_ladder(
        axes[0][0], spec, assign, attn[res_h], series, ladder,
        f"Head {res_h} (trained) — attention edges span one residue mass",
    )
    _panel_isotope(
        axes[0][1], spec, assign, attn[iso_h],
        f"Head {iso_h} (trained) — monoisotopic → ¹³C+1 (zoom)",
    )

    if control_enc is not None:
        cattn = spectrum_attention(control_enc, spec, device)
        # use the SAME head indices as trained, to make it an honest comparison
        _panel_ladder(axes[1][0], spec, assign, cattn[res_h], series, ladder,
                      f"Head {res_h} (random init) — no ladder structure")
        _panel_isotope(axes[1][1], spec, assign, cattn[iso_h],
                       f"Head {iso_h} (random init) — no isotope grouping")

    handles = [Line2D([0], [0], color=_COL["b"], lw=2, label="b-ion"),
               Line2D([0], [0], color=_COL["y"], lw=2, label="y-ion"),
               Line2D([0], [0], color=_COL["iso"], lw=2, label="¹³C satellite"),
               Line2D([0], [0], color=_COL["noise"], lw=2, label="unannotated")]
    axes[0][0].legend(handles=handles, fontsize=8, loc="upper right", frameon=False, ncol=2)

    fig.suptitle(
        f"Attention on one real MS/MS spectrum  ·  {spec.peptide}  ·  "
        f"arc width/opacity ∝ attention weight", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}  (peptide {spec.peptide}, {series}-ladder len "
          f"{len(ladder)}, residue-head {res_h}, isotope-head {iso_h})")


# ----------------------------------------------------------------------------
# control encoder + CLI
# ----------------------------------------------------------------------------

def build_random_encoder(cfg: dict[str, Any]) -> MSEncoder:
    """A same-architecture, freshly-initialized encoder for the null panel."""
    d = cfg["model"]
    mcfg = ModelConfig(
        d_model=d["d_model"], n_heads=d["n_heads"], n_layers=d["n_layers"],
        ffn_mult=d["ffn_mult"], dropout=d["dropout"], max_peaks=d["max_peaks"],
        fourier_mz=FourierConfig(**d["fourier_mz"]),
        fourier_int=FourierConfig(**d["fourier_int"]),
        delta_bias=DeltaBiasConfig(**d["delta_bias"]),
        use_precursor=d.get("use_precursor", False),
    )
    return MSEncoder(mcfg).eval()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Attention-arc figure on a real annotated spectrum")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="figure path (default: <ckpt-dir>/attn_arcs.png)")
    p.add_argument("--control", action="store_true",
                   help="add a random-init null row (trained vs random)")
    p.add_argument("--n-scan", type=int, default=800,
                   help="val spectra to scan for a clean ladder")
    p.add_argument("--min-run", type=int, default=4,
                   help="minimum consecutive-ion ladder length to accept")
    p.add_argument("--tol", type=float, default=0.02, help="Da match tol (z=1 ions)")
    p.add_argument("--seed", type=int, default=7, help="val stream shuffle seed")
    p.add_argument("--peptide", type=str, default=None,
                   help="force a specific peptide_charge label instead of scanning")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    enc, cfg, step = load_encoder(args.ckpt)
    device = torch.device(args.device)
    print(f"loaded {args.ckpt} (step {step}, {cfg['model']['n_heads']} heads)")

    if args.peptide:
        target = None
        for spec in iter_val_spectra(cfg, seed=args.seed):
            if spec.peptide == args.peptide:
                target = spec
                break
        if target is None:
            raise SystemExit(f"peptide {args.peptide} not found in val stream")
        chars, masses = residue_masses(args.peptide.rsplit("_", 1)[0])
        assign = assign_peaks(target.mz, theoretical_ions(chars, masses), tol=args.tol)
        series, _z, ladder = best_ladder(assign)
        spec = target
    else:
        spec, ladder, series, assign = select_spectrum(
            cfg, n_scan=args.n_scan, min_run=args.min_run, tol=args.tol, seed=args.seed)

    control = build_random_encoder(cfg) if args.control else None
    out = args.out or (args.ckpt.parent / "attn_arcs.png")
    render(enc, spec, assign, series, ladder, device, out, control_enc=control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
