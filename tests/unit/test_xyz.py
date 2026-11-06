"""Parser tests. Golden values were read by hand from the source records."""

import pytest

from qm9_curation.parse.xyz import ParseError, parse_record, serialize_record, to_float


@pytest.mark.parametrize("index", [1, 3, 4, 212, 6620])
def test_golden_records_round_trip_exactly(golden: dict[int, str], index: int) -> None:
    text = golden[index]
    assert serialize_record(parse_record(text, f"golden_{index}")) == text


def test_methane_golden_values(golden: dict[int, str]) -> None:
    r = parse_record(golden[1], "m")
    assert r.index == 1 and r.n_atoms == 5
    assert [a.element for a in r.atoms] == ["C", "H", "H", "H", "H"]
    assert to_float(r.properties[0]) == 157.7118  # A, GHz
    assert to_float(r.properties[3]) == 0.0  # mu, Debye; written "0."
    assert to_float(r.properties[10]) == -40.47893  # U0, Hartree
    assert to_float(r.properties[14]) == 6.469  # Cv
    assert to_float(r.atoms[0].x) == -0.0126981359
    assert to_float(r.atoms[0].charge) == -0.535689
    assert len(r.frequency_sets) == 1 and r.n_modes == 9
    assert r.smiles_gdb == r.smiles_relaxed == "C"
    assert r.inchi_relaxed == "InChI=1S/CH4/h1H4"


def test_water_golden_values(golden: dict[int, str]) -> None:
    r = parse_record(golden[3], "w")
    assert to_float(r.properties[3]) == 1.8511  # dipole
    assert to_float(r.properties[7]) == 0.3615  # gap
    assert r.frequencies == ("1671.4222", "3803.6305", "3907.698")


def test_acetylene_is_linear(golden: dict[int, str]) -> None:
    r = parse_record(golden[4], "a")
    assert r.n_modes == 3 * 4 - 5 == len(r.frequencies)
    assert r.properties[0] == "0."  # rotational constant A undefined for linear
    assert r.smiles_gdb == "C#C"


def test_fortran_exponent_is_read(golden: dict[int, str]) -> None:
    r = parse_record(golden[212], "f")
    assert r.atoms[0].x == "2.1997*^-6"
    assert to_float(r.atoms[0].x) == 2.1997e-6


def test_doubled_frequency_sets_are_split(golden: dict[int, str]) -> None:
    r = parse_record(golden[6620], "d")
    first, second = r.frequency_sets
    assert len(first) == len(second) == r.n_modes == 3 * 14 - 6
    assert first[0] == "-2.0226" and second[0] == "-1.979"
    assert first != second


@pytest.mark.parametrize(
    ("mutate", "line_no", "reason"),
    [
        (lambda t: t.replace("gdb 1\t", "gdb9 1\t"), 2, "header"),
        (lambda t: t.replace("\t6.469\t\n", "\t6.469\n"), 2, "15 + trailing tab"),
        (lambda t: t.replace("157.7118", "15x.7"), 2, "not a number"),
        (lambda t: t.replace("H\t 0.002150416", "Cl\t 0.002150416"), 4, "element"),
        (lambda t: t.replace("\t 1.0858041578", "\t  1.0858041578"), 3, "not a number"),
        (lambda t: t.replace("\t-0.535689", "\t+0.535689"), 3, "not a number"),
        (lambda t: t.replace("C\tC\t\n", "C\tC\n"), 9, "SMILES"),
        (lambda t: t.replace("\t3151.7078", ""), 8, "frequencies"),
        (lambda t: t.replace("InChI=1S/CH4/h1H4\tInChI", "InChI=1S/CH4/h1H4 InChI"), 10, "InChI"),
    ],
)
def test_malformed_records_fail_with_location(
    golden: dict[int, str], mutate: object, line_no: int, reason: str
) -> None:
    text = mutate(golden[1])  # type: ignore[operator]
    assert text != golden[1]
    with pytest.raises(ParseError) as info:
        parse_record(text, "bad")
    assert info.value.line_no == line_no
    assert reason in info.value.reason


def test_layout_deviation_not_caught_by_tokenizing_is_still_rejected(
    golden: dict[int, str],
) -> None:
    # A positive number without its leading space tokenizes fine but breaks the layout.
    text = golden[1].replace("H\t 0.002150416", "H\t0.002150416")
    with pytest.raises(ParseError, match="layout differs") as info:
        parse_record(text, "bad")
    assert info.value.line_no == 4


def test_missing_trailing_newline_and_truncation(golden: dict[int, str]) -> None:
    with pytest.raises(ParseError, match="newline"):
        parse_record(golden[1].rstrip("\n"), "bad")
    with pytest.raises(ParseError, match="lines, expected"):
        parse_record("\n".join(golden[1].split("\n")[:8]) + "\n", "bad")
