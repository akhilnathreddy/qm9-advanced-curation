from hypothesis import given
from hypothesis import strategies as st
from pytest import approx

from qm9_curation.units import zpve_from_wavenumbers_hartree

wavenumbers = st.lists(st.floats(min_value=1.0, max_value=5000.0), min_size=1, max_size=60)


@given(wavenumbers, wavenumbers)
def test_zpve_is_additive_over_modes(a: list[float], b: list[float]) -> None:
    total = zpve_from_wavenumbers_hartree(a + b)
    assert total == approx(zpve_from_wavenumbers_hartree(a) + zpve_from_wavenumbers_hartree(b))


@given(wavenumbers)
def test_zpve_is_order_independent(a: list[float]) -> None:
    assert zpve_from_wavenumbers_hartree(a) == approx(zpve_from_wavenumbers_hartree(a[::-1]))
