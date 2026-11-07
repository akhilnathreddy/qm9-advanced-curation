"""Physical constants and unit conversions, defined in one place.

Every conversion used anywhere in the pipeline must come from this module so
that a single audit covers all of them. Values are CODATA 2018 recommended
values (https://physics.nist.gov/cuu/Constants/). They are written out
explicitly rather than taken from ``scipy.constants`` so that a SciPy upgrade
that moves to a newer CODATA release cannot silently change released numbers.
"""

import math
from typing import Final

CODATA_RELEASE: Final = "CODATA 2018"

# --- Energy -----------------------------------------------------------------
HARTREE_TO_EV: Final = 27.211386245988
HARTREE_TO_KCAL_PER_MOL: Final = 627.5094740631
HARTREE_TO_INV_CM: Final = 219474.6313632
INV_CM_TO_HARTREE: Final = 1.0 / HARTREE_TO_INV_CM
EV_TO_KCAL_PER_MOL: Final = HARTREE_TO_KCAL_PER_MOL / HARTREE_TO_EV

# Chemical accuracy, conventionally 1 kcal/mol.
CHEMICAL_ACCURACY_KCAL_PER_MOL: Final = 1.0
CHEMICAL_ACCURACY_EV: Final = CHEMICAL_ACCURACY_KCAL_PER_MOL / EV_TO_KCAL_PER_MOL

# --- Thermal ----------------------------------------------------------------
BOLTZMANN_EV_PER_K: Final = 8.617333262e-5
BOLTZMANN_HARTREE_PER_K: Final = BOLTZMANN_EV_PER_K / HARTREE_TO_EV
QM9_TEMPERATURE_K: Final = 298.15

# --- Length / volume --------------------------------------------------------
BOHR_TO_ANGSTROM: Final = 0.529177210903
BOHR2_TO_ANGSTROM2: Final = BOHR_TO_ANGSTROM**2
BOHR3_TO_ANGSTROM3: Final = BOHR_TO_ANGSTROM**3

# --- Dipole -----------------------------------------------------------------
# 1 e*a0 expressed in debye.
E_BOHR_TO_DEBYE: Final = 2.541746473


def thermal_energy_hartree(temperature_k: float = QM9_TEMPERATURE_K) -> float:
    """k_B * T in Hartree; equals H - U for an ideal gas at that temperature."""
    return BOLTZMANN_HARTREE_PER_K * temperature_k


def zpve_from_wavenumbers_hartree(wavenumbers_inv_cm: list[float]) -> float:
    """Harmonic zero-point vibrational energy, 1/2 * sum(nu), in Hartree."""
    return 0.5 * sum(wavenumbers_inv_cm) * INV_CM_TO_HARTREE


# --- Rotational constants -----------------------------------------------------
PLANCK_J_S: Final = 6.62607015e-34  # exact (SI 2019)
AMU_KG: Final = 1.66053906660e-27
# B [GHz] = ROT_CONST_GHZ_AMU_ANG2 / I [amu * Angstrom^2]
ROT_CONST_GHZ_AMU_ANG2: Final = PLANCK_J_S / (8 * math.pi**2 * AMU_KG * 1e-20) / 1e9

# Most abundant isotope masses (amu), as used by Gaussian for rotational constants.
# Source: NIST Atomic Weights and Isotopic Compositions.
ISOTOPE_MASS_AMU: Final = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
}
