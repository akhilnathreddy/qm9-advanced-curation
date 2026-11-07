from pathlib import Path

import pytest

from qm9_curation.features import build as feature_build
from qm9_curation.parse import build as parse_build

pytestmark = pytest.mark.pipeline


def test_fixture_features_end_to_end(qm9_1k_dir: Path) -> None:
    frames, _ = parse_build.build(parse_build.iter_directory(qm9_1k_dir), "0" * 64)
    tables, report = feature_build.build(frames["atoms"], workers=2)
    assert all(report["gate_6_conditions"].values()), report["gate_6_conditions"]
    assert tables["composition"].height == 1000
    assert report["descriptors_kept"] > 150
    again, _ = feature_build.build(frames["atoms"], workers=2)
    for name, frame in tables.items():
        assert frame.equals(again[name]), name  # deterministic
