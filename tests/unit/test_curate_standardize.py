from pathlib import Path

import polars as pl
import pytest

from qm9_curation.curate.standardize import geometry_identifiers, standardize
from qm9_curation.inventory import parse_atomref
from qm9_curation.parse import build as parse_build
from qm9_curation.units import HARTREE_TO_EV

GOLDEN = Path(__file__).parents[1] / "fixtures" / "golden"
ATOMREF = {
    "H": {
        "zpve": 0.0,
        "u0": -0.500273,
        "u298": -0.498857,
        "h298": -0.497912,
        "g298": -0.510927,
        "cv": 2.981,
    },
    "C": {
        "zpve": 0.0,
        "u0": -37.846772,
        "u298": -37.845355,
        "h298": -37.844411,
        "g298": -37.861317,
        "cv": 2.981,
    },
    "N": {
        "zpve": 0.0,
        "u0": -54.583861,
        "u298": -54.582445,
        "h298": -54.581501,
        "g298": -54.598897,
        "cv": 2.981,
    },
    "O": {
        "zpve": 0.0,
        "u0": -75.064579,
        "u298": -75.063163,
        "h298": -75.062219,
        "g298": -75.079532,
        "cv": 2.981,
    },
    "F": {
        "zpve": 0.0,
        "u0": -99.718730,
        "u298": -99.717314,
        "h298": -99.716370,
        "g298": -99.733544,
        "cv": 2.981,
    },
}


def golden_standardized() -> pl.DataFrame:
    frames, _ = parse_build.build(parse_build.iter_directory(GOLDEN), "0" * 64)
    mol = frames["molecules"]
    validation = mol.select(
        "qm9_id",
        pl.col("is_linear").alias("linear_by_geometry"),
        pl.lit(0, dtype=pl.UInt8).alias("zpve_consistent_set"),
        pl.lit("consistent").alias("identity_class"),
    )
    return standardize(mol, frames["atoms"], validation, ATOMREF)


def test_methane_atomization_energy_matches_hand_calculation() -> None:
    row = golden_standardized().filter(pl.col("qm9_id") == 1).row(0, named=True)
    # U0(CH4) - U0(C) - 4 U0(H) = -40.47893 + 37.846772 + 4 * 0.500273 = -0.631066 Ha
    assert row["u0_atomization_ev"] == pytest.approx(-0.631066 * HARTREE_TO_EV, abs=1e-9)
    assert row["gap_ev"] == pytest.approx(0.5048 * HARTREE_TO_EV)
    assert row["rot_a_ghz"] == row["rot_a_ghz_std"] == 157.7118  # source kept, std equal


def test_linear_molecule_gets_null_a_with_reason() -> None:
    row = golden_standardized().filter(pl.col("qm9_id") == 4).row(0, named=True)
    assert row["rot_a_ghz"] == 0.0  # source value untouched
    assert row["rot_a_ghz_std"] is None and "linear" in row["rot_a_missing_reason"]


def test_atomref_parser_agrees_with_test_constants(tmp_path: Path) -> None:
    text = "\n".join(
        f"   {el}   "
        + "   ".join(f"{v[k]:.6f}" for k in ("zpve", "u0", "u298", "h298", "g298"))
        + f"   {v['cv']:.3f}"
        for el, v in ATOMREF.items()
    )
    (tmp_path / "atomref.txt").write_text(text)
    assert parse_atomref(tmp_path / "atomref.txt") == ATOMREF


def test_geometry_identifiers_and_failure_reason() -> None:
    water = geometry_identifiers(
        ["O", "H", "H"], [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0), (-0.24, 0.93, 0.0)]
    )
    assert (
        water["smiles_geometry"] == "O"
        and water["inchikey_geometry"] == "XLYOFNOQVPJJNP-UHFFFAOYSA-N"
    )
    # A lone carbon with three hydrogens has no closed-shell neutral bond assignment.
    methyl = geometry_identifiers(
        ["C", "H", "H", "H"], [(0, 0, 0), (1.09, 0, 0), (-0.55, 0.94, 0), (-0.55, -0.94, 0)]
    )
    assert methyl["smiles_geometry"] is None and methyl["geometry_identifier_error"] is not None
