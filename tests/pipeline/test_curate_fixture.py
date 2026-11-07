from pathlib import Path

import polars as pl
import pytest
from tests.unit.test_curate_standardize import ATOMREF

from qm9_curation.checks import build as checks_build
from qm9_curation.config import load_config
from qm9_curation.curate.build import curated_frame, gate_conditions
from qm9_curation.curate.standardize import identifiers_from_geometry
from qm9_curation.parse import build as parse_build
from qm9_curation.profiling.build import build_profile

pytestmark = pytest.mark.pipeline


def test_fixture_curation_end_to_end(qm9_1k_dir: Path, tmp_path: Path) -> None:
    frames, _ = parse_build.build(parse_build.iter_directory(qm9_1k_dir), "0" * 64)
    interim, raw = tmp_path / "interim", tmp_path / "raw"
    parse_build.write_tables(frames, interim)
    unchar = set(frames["molecules"]["qm9_id"].head(20).to_list())
    _, profile, _ = build_profile(
        frames["molecules"], frames["atoms"], frames["frequencies"], unchar
    )
    profile.write_parquet(interim / "molecule_profile.parquet")
    raw.mkdir()
    (raw / "uncharacterized.txt").write_text("".join(f"  {i}  X X X 1.0\n" for i in sorted(unchar)))
    validation, _ = checks_build.build(interim, raw, workers=2)

    ids = identifiers_from_geometry(frames["atoms"], workers=2)
    rules = load_config("curation_rules")["rules"]
    curated = curated_frame(
        frames["molecules"], frames["atoms"], validation, profile, ATOMREF, rules, ids
    )
    conditions = gate_conditions(curated, frames["molecules"])
    assert all(conditions.values()), conditions
    assert curated.height == 1000
    assert curated.filter(pl.col("quality_tier") == "quarantine").height == 5
    assert (
        curated.filter(pl.col("uncharacterized_listed") & (pl.col("quality_tier") == "gold")).height
        == 0
    )
