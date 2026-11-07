from pathlib import Path

import polars as pl
import pytest

from qm9_curation.checks import build as checks_build
from qm9_curation.checks.registry import CHECK_NAMES
from qm9_curation.parse import build as parse_build
from qm9_curation.profiling.build import build_profile

pytestmark = pytest.mark.pipeline


def test_fixture_validation_end_to_end(qm9_1k_dir: Path, tmp_path: Path) -> None:
    frames, _ = parse_build.build(parse_build.iter_directory(qm9_1k_dir), "0" * 64)
    interim, raw = tmp_path / "interim", tmp_path / "raw"
    parse_build.write_tables(frames, interim)
    unchar = set(pl.read_parquet(interim / "molecules.parquet")["qm9_id"].head(20).to_list())
    _, per_molecule, _ = build_profile(
        frames["molecules"], frames["atoms"], frames["frequencies"], unchar
    )
    per_molecule.write_parquet(interim / "molecule_profile.parquet")
    raw.mkdir()
    (raw / "uncharacterized.txt").write_text("".join(f"  {i}  X X X 1.0\n" for i in sorted(unchar)))

    flags, report = checks_build.build(interim, raw, workers=2)
    assert all(report["gate_4_conditions"].values()), report["gate_4_conditions"]
    assert flags.height == 1000
    assert set(report["checks"]) == set(CHECK_NAMES)
    assert report["checks"]["real_minimum"]["n_failed"] == 5
    assert report["checks"]["not_uncharacterized"]["n_failed"] == 20
    for name in (
        "gap_consistency",
        "enthalpy_minus_energy",
        "zpve_from_frequencies",
        "rotational_constants",
    ):
        assert report["checks"][name]["n_failed"] == 0, name
    assert report["doubled_frequency_resolution"]["records"] >= 10
