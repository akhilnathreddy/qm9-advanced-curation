from pathlib import Path

import polars as pl
import pytest
from tests.pipeline.test_review_fixture import curated_fixture

from qm9_curation.config import load_config
from qm9_curation.splits import build as split_build

pytestmark = pytest.mark.pipeline


def test_fixture_splits_end_to_end(qm9_1k_dir: Path, tmp_path: Path) -> None:
    curated, features = curated_fixture(qm9_1k_dir, tmp_path)
    curated = curated.sort("qm9_id")
    cfg = load_config("splits")
    cfg["fraction_tolerance"] = 0.03  # 1,000 records in a few hundred formula/scaffold units
    cfg["audit"]["nn_sample"] = 50
    groups = split_build.group_table(curated, features["ecfp4"], cfg, workers=2)
    splits, rules = split_build.build_splits(curated, features["composition"], groups, cfg)
    report = {
        "splits": split_build.audit(
            splits, groups, rules, curated, features["composition"], features["ecfp4"], cfg
        )
    }
    conditions = split_build.gate_conditions(report, groups, curated.height)
    assert all(conditions.values()), conditions
    assert splits.height == 1000 * len(cfg["splits"]) * len(cfg["seeds"])
    size = splits.filter(
        (pl.col("split_name") == "size_extrapolation") & (pl.col("subset") == "test")
    )
    assert size.join(curated, on="qm9_id")["n_heavy"].min() == 9

    again, _ = split_build.build_splits(curated, features["composition"], groups, cfg)
    assert splits.equals(again)  # deterministic
