import math

import pytest

from qm9_curation import units


def test_hartree_ev_kcal_are_mutually_consistent() -> None:
    assert units.HARTREE_TO_EV * units.EV_TO_KCAL_PER_MOL == pytest.approx(
        units.HARTREE_TO_KCAL_PER_MOL, rel=1e-12
    )


def test_chemical_accuracy_in_ev() -> None:
    # 1 kcal/mol is about 0.0434 eV.
    assert units.CHEMICAL_ACCURACY_EV == pytest.approx(0.0433641, abs=1e-6)


def test_thermal_energy_at_qm9_temperature() -> None:
    # k_B * 298.15 K is about 9.44e-4 Hartree; H - U should equal this in QM9.
    assert units.thermal_energy_hartree() == pytest.approx(9.4418e-4, rel=1e-4)


def test_inverse_cm_round_trip() -> None:
    assert units.INV_CM_TO_HARTREE * units.HARTREE_TO_INV_CM == pytest.approx(1.0)


def test_zpve_of_single_mode() -> None:
    # A single 1000 cm^-1 mode contributes 500 cm^-1 of zero-point energy.
    expected = 500.0 / units.HARTREE_TO_INV_CM
    assert units.zpve_from_wavenumbers_hartree([1000.0]) == pytest.approx(expected)


def test_bohr_powers() -> None:
    assert units.BOHR3_TO_ANGSTROM3 == pytest.approx(units.BOHR_TO_ANGSTROM**3)
    assert math.isclose(units.BOHR2_TO_ANGSTROM2, 0.280028520, rel_tol=1e-8)
