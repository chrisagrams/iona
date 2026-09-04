"""Provide mass references for model diagnostics."""

from pyteomics import mass

WATER_MASS = mass.calculate_mass(formula="H2O")
PROTON_MASS = mass.nist_mass["H+"][0][0]

RESIDUE_MASSES: dict[str, float] = {
    residue: mass.std_aa_mass[residue] for residue in "GASPVTCLINDQKEMHFRYW"
}

RESIDUES_AA20: dict[str, float] = {
    **{residue: value for residue, value in RESIDUE_MASSES.items() if residue not in {"I", "L"}},
    "L/I": RESIDUE_MASSES["L"],
}

_C13 = mass.nist_mass["C"][13][0] - mass.nist_mass["C"][12][0]
ISOTOPES: dict[str, float] = {
    "¹³C z3": _C13 / 3,
    "¹³C z2": _C13 / 2,
    "2¹³C z3": 2 * _C13 / 3,
    "¹³C": _C13,
    "2¹³C": 2 * _C13,
}

_NEUTRAL_LOSS_FORMULAS = {
    "NH₃": "NH3",
    "H₂O": "H2O",
    "CO": "CO",
    "CO₂": "CO2",
    "HPO₃": "HPO3",
    "H₃PO₄": "H3PO4",
    "Hexose": "C6H10O5",
}
NEUTRAL_LOSSES: dict[str, float] = {
    name: mass.calculate_mass(formula=formula) for name, formula in _NEUTRAL_LOSS_FORMULAS.items()
}
