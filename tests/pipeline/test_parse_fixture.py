from pathlib import Path

import polars as pl
import pytest

from qm9_curation.parse import build

pytestmark = pytest.mark.pipeline


def test_fixture_parses_losslessly_into_valid_tables(qm9_1k_dir: Path) -> None:
    frames, report = build.build(build.iter_directory(qm9_1k_dir), "0" * 64)
    assert all(report["gate_2_conditions"].values()), report
    assert report["n_parsed"] == 1000
    mol = frames["molecules"]
    assert int(mol["is_linear"].sum()) == 5
    assert int((mol["n_frequency_sets"] == 2).sum()) >= 10
    assert frames["frequencies"].filter(pl.col("wavenumber_cm1") < 0)["qm9_id"].n_unique() == 5
