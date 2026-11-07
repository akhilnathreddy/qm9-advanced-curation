import math
from pathlib import Path

import polars as pl
import pytest

from qm9_curation.parse import build
from qm9_curation.schema import SCHEMAS

GOLDEN_DIR = Path(__file__).parents[1] / "fixtures" / "golden"
SHA = "0" * 64


@pytest.mark.parametrize(
    ("elements", "formula"),
    [
        ("CHHHH", "CH4"),
        ("OHH", "H2O"),
        ("CCHH", "C2H2"),
        ("NHHH", "H3N"),
        ("FCCOHN", "C2HFNO"),
        ("CCNN", "C2N2"),  # no hydrogen: H must not appear (regression, cyanogen qm9_id 25)
        ("FCOO", "CFO2"),
        ("OO", "O2"),
    ],
)
def test_hill_formula(elements: str, formula: str) -> None:
    assert build.hill_formula(elements) == formula


@pytest.mark.parametrize("token", ["0.", "157.7118", "-0.0126981359", "2.1997*^-6", "6.469"])
def test_value_is_exact_for_source_style_tokens(token: str) -> None:
    assert build.value_is_exact(token)


def test_value_is_exact_detects_precision_beyond_float64() -> None:
    assert not build.value_is_exact("0.12345678901234567890123")


def test_golden_tables_are_valid_and_consistent() -> None:
    frames, report = build.build(build.iter_directory(GOLDEN_DIR), SHA)
    assert all(report["gate_2_conditions"].values()), report
    mol = frames["molecules"]
    assert mol["qm9_id"].to_list() == [1, 3, 4, 212, 6620]
    assert mol["formula"].to_list()[:3] == ["CH4", "H2O", "C2H2"]
    assert mol.filter(pl.col("qm9_id") == 4)["is_linear"].item() is True
    assert mol.filter(pl.col("qm9_id") == 6620)["n_frequency_sets"].item() == 2
    assert report["parse_log_events"] == {"doubled_frequencies": 1, "fortran_exponent": 1}
    atoms = frames["atoms"].filter(pl.col("qm9_id") == 212)
    assert atoms["x_ang"][0] == 2.1997e-6


def test_parse_failure_is_reported_not_raised() -> None:
    members = [("dsgdb9nsd_000001.xyz", b"5\nnot a record\n")]
    _, report = build.build(members, SHA)
    assert report["parse_failures"][0]["member"] == "dsgdb9nsd_000001.xyz"
    assert report["gate_2_conditions"]["all_records_parsed"] is False


def test_schema_rejects_non_finite_coordinates() -> None:
    frames, _ = build.build(build.iter_directory(GOLDEN_DIR), SHA)
    bad = frames["atoms"].with_columns(
        pl.when(pl.col("atom_idx") == 0).then(math.nan).otherwise(pl.col("x_ang")).alias("x_ang")
    )
    assert build.validate_frames({"atoms": bad})["atoms"] is not None
    assert build.validate_frames({"atoms": frames["atoms"]})["atoms"] is None


def test_tables_follow_schema_column_order() -> None:
    frames, _ = build.build(build.iter_directory(GOLDEN_DIR), SHA)
    for name, frame in frames.items():
        assert frame.columns == list(SCHEMAS[name].to_schema().columns)


def test_written_tables_are_deterministic(tmp_path: Path) -> None:
    frames, _ = build.build(build.iter_directory(GOLDEN_DIR), SHA)
    assert build.write_tables(frames, tmp_path / "a") == build.write_tables(frames, tmp_path / "b")
