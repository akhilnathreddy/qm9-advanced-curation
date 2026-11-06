from typing import Any

from qm9_curation import fixture


def synthetic_traits(n: int) -> dict[int, dict[str, Any]]:
    traits = {}
    for i in range(1, n + 1):
        traits[i] = {
            "n_heavy": 9 if i % 10 else 1 + i % 8,
            "linear": i in (4, 5),
            "doubled_frequencies": i % 97 == 0,
            "negative_frequency": i == 50,
            "fortran_exponent": i % 89 == 0,
        }
    return traits


def test_selection_is_deterministic_sized_and_keeps_edge_cases() -> None:
    traits = synthetic_traits(5000)
    a = fixture.select(traits, {7, 8, 9}, {50, 123}, seed=1, size=300)
    b = fixture.select(traits, {7, 8, 9}, {50, 123}, seed=1, size=300)
    assert a == b
    assert len(a) == 300
    assert {4, 5, 50, 123, 7, 8, 9} <= a.keys()
    assert a[4] == "linear" and a[50] == "negative_frequency"


def test_every_stratum_is_represented() -> None:
    traits = synthetic_traits(5000)
    chosen = fixture.select(traits, set(), set(), seed=2, size=200)
    assert {traits[i]["n_heavy"] for i in chosen} == {t["n_heavy"] for t in traits.values()}


def test_record_traits_on_linear_molecule() -> None:
    acetylene = (
        "4\ngdb 4\t0.\t35.61\t35.61\t0.\t16.28\t-0.2845\t0.0506\t0.3351\t59.52\t0.0268"
        "\t-77.31\t-77.31\t-77.30\t-77.33\t8.574\t\n"
        "C\t 0.5995\t 0.\t 1.\t-0.207\nC\t-0.5995\t 0.\t 1.\t-0.207\n"
        "H\t-1.6616\t 0.\t 1.\t 0.207\nH\t 1.6616\t 0.\t 1.\t 0.207\n"
        "549.7\t549.7\t795.2\t795.2\t2078.1\t3455.8\t3557.8\nC#C\tC#C\t\n"
        "InChI=1S/C2H2/c1-2/h1-2H\tInChI=1S/C2H2/c1-2/h1-2H\n"
    )
    t = fixture.record_traits(acetylene)
    assert t["linear"] and t["n_heavy"] == 2 and not t["doubled_frequencies"]
