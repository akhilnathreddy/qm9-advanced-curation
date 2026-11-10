"""End to end on the 1k fixture: splits -> benchmark grid -> analysis -> Gate 9 conditions."""

import json
from pathlib import Path
from typing import Any

import pytest
from tests.pipeline.test_review_fixture import curated_fixture

from qm9_curation.config import PROJECT_ROOT, load_config
from qm9_curation.splits import build as split_build

pytest.importorskip("torch")
pytest.importorskip("sklearn")

pytestmark = pytest.mark.pipeline


def small_config() -> dict[str, Any]:
    cfg = load_config("benchmark")
    cfg["seeds"] = [0, 1]
    cfg["experiments"]["rq3"]["splits"] = ["random", "isomer_group", "size_extrapolation"]
    cfg["experiments"]["stress"]["splits"] = ["element_holdout"]
    cfg["experiments"]["rq5"]["n_grid"] = [100, 200]
    cfg["coverage"] = {
        "mean": ["rq3", "stress"],
        "dressed_atom": ["rq3"],
        "gbm": ["rq3", "rq5", "rq2"],
        "krr": ["rq3"],
        "dmpnn": ["rq3_random", "rq5"],
        "schnet": ["rq3_random"],
    }
    m = cfg["models"]
    m["gbm"].update(max_iter=15, max_leaf_nodes=7, min_samples_leaf=5)
    m["krr"]["max_train"] = 300
    for g in ("dmpnn", "schnet"):
        m[g].update(epochs=2, batch_size=64, device="cpu")
    m["dmpnn"]["arch"] = {"hidden": 32, "depth": 2, "ffn_hidden": 32}
    m["schnet"]["arch"] = {
        "hidden": 16,
        "filters": 16,
        "interactions": 2,
        "n_rbf": 10,
        "cutoff": 5.0,
    }
    cfg["tuning"]["gbm"]["grid"] = {"max_leaf_nodes": [3, 7]}
    cfg["tuning"]["krr"] = {"n_train": 200, "grid": {"sigma_factor": [0.5, 1.0], "lambda": [1e-6]}}
    cfg["statistics"]["n_boot"] = 200
    return cfg


def test_fixture_benchmark_end_to_end(qm9_1k_dir: Path, tmp_path: Path) -> None:
    from qm9_curation.benchmark import analysis, run
    from qm9_curation.benchmark.tracking import Tracker

    curated, features = curated_fixture(qm9_1k_dir, tmp_path)
    processed, splits_dir = tmp_path / "processed", tmp_path / "splits"
    (processed / "features").mkdir(parents=True)
    splits_dir.mkdir()
    curated.write_parquet(processed / "molecules.parquet")
    for name, frame in features.items():
        frame.write_parquet(processed / "features" / f"{name}.parquet")
    scfg = load_config("splits")
    scfg["fraction_tolerance"] = 0.03
    groups = split_build.group_table(curated.sort("qm9_id"), features["ecfp4"], scfg, workers=2)
    splits, _ = split_build.build_splits(curated, features["composition"], groups, scfg)
    splits.write_parquet(splits_dir / "splits.parquet")
    groups.write_parquet(splits_dir / "split_groups.parquet")

    cfg = small_config()
    ctx = run.Context(cfg, processed, splits_dir, tmp_path / "interim")
    prov = run.provenance(processed, splits_dir)
    mlruns, runs_dir = tmp_path / "mlruns", tmp_path / "runs"
    runner = run.Runner(ctx, runs_dir, Tracker(mlruns, "test", prov), prov, log=False)
    run.grid(runner, list(cfg["models"]), ["rq3", "stress", "rq2", "rq5"])
    n_files = len(list(runs_dir.glob("*.json")))
    run.grid(runner, list(cfg["models"]), ["rq3", "stress", "rq2", "rq5"])  # resumes: no new runs
    assert len(list(runs_dir.glob("*.json"))) == n_files

    plan = PROJECT_ROOT / "docs" / "analysis_plan.md"
    report, table = analysis.build(runs_dir, processed, plan, cfg, mlruns)
    conditions = report["gate_9_conditions"]
    assert all(conditions.values()), conditions
    assert table.height == n_files
    rq = report["research_questions"]
    assert {rq[k]["verdict"] for k in rq} <= set(analysis.VERDICTS)
    assert rq["rq2"]["tests"] and rq["rq5"]["largest_common_n"]
    assert "schnet" in report["rq3_cells"] and "isomer_group" not in report["rq3_cells"]["schnet"]
    # Deterministic: a second analysis gives the same report.
    again, _ = analysis.build(runs_dir, processed, plan, cfg, mlruns)
    assert json.dumps(again, sort_keys=True, default=float) == json.dumps(
        report, sort_keys=True, default=float
    )
    assert "| RQ3 |" in analysis.summary_markdown(report)
    assert len(analysis.figures(report, tmp_path / "fig")) == 2
