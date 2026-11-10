import numpy as np
import polars as pl
import pytest
from scipy import sparse

from qm9_curation.benchmark import analysis, metrics, models
from qm9_curation.benchmark.data import Inputs, tier_subsample
from qm9_curation.benchmark.tracking import run_key


def toy_inputs(n: int = 300, seed: int = 0) -> Inputs:
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 6, size=(n, 5)).astype(np.float64)
    coulomb = np.sort(rng.normal(size=(n, 29)), axis=1)
    desc = rng.normal(size=(n, 8)).astype(np.float32)
    bits = rng.integers(0, 2, size=(n, 2048)) * (rng.random((n, 2048)) < 0.02)
    y = comp @ np.array([-0.6, -5.0, -3.0, -2.0, -1.0]) + 0.1 * desc[:, 0]
    frame = pl.DataFrame(
        {
            "qm9_id": np.arange(1, n + 1),
            "quality_tier": ["gold"] * (n - 20) + ["silver"] * 10 + ["bronze"] * 10,
            "leakage_group_id": np.arange(1, n + 1),
            "u0_atomization_ev": y,
            "gap_ev": 5.0 + desc[:, 1],
        }
    )
    return Inputs(
        frame=frame,
        composition=comp,
        descriptors=desc,
        ecfp=sparse.csr_matrix(bits.astype(np.float32)),
        coulomb=coulomb,
        excluded={},
    )


def test_metrics_basic() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    m = metrics.metrics(y, y + np.array([0.01, -0.01, 0.1, -0.1]))
    assert m["mae"] == pytest.approx(0.055)
    assert m["within_chemical_accuracy"] == 0.5
    assert m["r2"] < 1.0


def test_holm_matches_hand_computation() -> None:
    adj = metrics.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert metrics.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])  # 2 x 0.8 caps at 1


def test_unpaired_ci_brackets_mean_and_detects_difference() -> None:
    rng = np.random.default_rng(1)
    small = [np.abs(rng.normal(0, 1, 500)) for _ in range(3)]
    large = [np.abs(rng.normal(0, 2, 500)) for _ in range(3)]
    a = metrics.unpaired_mae_ci(small, 500, 0)
    b = metrics.unpaired_mae_ci(large, 500, 0)
    assert a["ci_low"] < a["mae"] < a["ci_high"]
    assert a["ci_high"] < b["ci_low"]
    assert metrics.unpaired_greater_pvalue(b["boot"], a["boot"]) < 0.01
    assert metrics.unpaired_greater_pvalue(a["boot"], b["boot"]) > 0.99


def test_paired_difference_zero_and_signed() -> None:
    zero = metrics.paired_difference_ci([np.zeros(100), np.zeros(80)], 200, 0)
    assert zero["estimate"] == 0.0 and zero["ci_low"] == 0.0 == zero["ci_high"]
    neg = metrics.paired_difference_ci([np.full(100, -0.1) + np.linspace(-0.01, 0.01, 100)], 200, 0)
    assert neg["ci_high"] < 0 and neg["p_less_than_zero"] < 0.01


def test_dressed_atom_recovers_linear_target() -> None:
    inp = toy_inputs()
    rows = np.arange(inp.frame.height)
    y = inp.composition @ np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + 7.0
    coef = models.dressed_atom(inp, rows, y)
    assert coef == pytest.approx([1, 2, 3, 4, 5, 7])


@pytest.mark.parametrize("model", ["mean", "dressed_atom", "gbm", "krr"])
def test_classical_levels_predict_requested_rows(model: str) -> None:
    inp = toy_inputs()
    train, val, test = np.arange(0, 200), np.arange(200, 250), np.arange(250, 280)
    params = {
        "gbm": {
            "learning_rate": 0.2,
            "max_iter": 30,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 5,
            "l2_regularization": 0.0,
            "min_bit_frequency": 0.01,
        },
        "krr": {"max_train": 150, "sigma_factor": 1.0, "lambda": 1e-6},
    }.get(model, {})
    fit = models.fit_predict(
        model, inp, "u0_atomization_ev", train, val, {"val": val, "test": test}, params, 0
    )
    assert set(fit.predictions) == {"val", "test"}
    assert fit.predictions["test"].shape == (30,)
    y = inp.frame["u0_atomization_ev"].to_numpy()
    mae = np.abs(fit.predictions["test"] - y[test]).mean()
    if model != "mean":
        assert mae < 0.5  # composition explains this target almost entirely
    if model == "krr":
        assert fit.info["n_train_used"] == 150
    if model == "gbm":
        assert 1 <= fit.info["best_iteration"] <= 30


def test_krr_interpolates_training_points() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 4))
    y = np.sin(x[:, 0])
    pred = models.krr_fit_predict(x, y, {"train": x}, sigma=2.0, lam=1e-10)["train"]
    assert np.abs(pred - y).max() < 1e-5


def test_tier_subsample_is_seeded_and_filtered() -> None:
    inp = toy_inputs()
    rows = np.arange(inp.frame.height)
    gold = tier_subsample(inp, rows, ["gold"], 100, seed=4)
    assert len(gold) == 100 and set(inp.frame["quality_tier"].to_numpy()[gold]) == {"gold"}
    assert np.array_equal(gold, tier_subsample(inp, rows, ["gold"], 100, seed=4))
    assert not np.array_equal(gold, tier_subsample(inp, rows, ["gold"], 100, seed=5))
    with pytest.raises(ValueError):
        tier_subsample(inp, rows, ["silver"], 50, seed=0)


def test_run_key_depends_on_spec_and_params() -> None:
    spec = {"model": "gbm", "seed": 0}
    assert run_key(spec, {"version": 1}) == run_key(dict(spec), {"version": 1})
    assert run_key(spec, {"version": 1}) != run_key(spec, {"version": 2})
    assert run_key(spec, {"version": 1}) != run_key({**spec, "seed": 1}, {"version": 1})


def _cell(mae: float, lo: float, hi: float, boot_center: float) -> dict[str, object]:
    return {"mae": mae, "ci_low": lo, "ci_high": hi, "_boot": np.full(100, boot_center)}


def test_rq3_rule_needs_ci_separation_and_holm() -> None:
    cells = {
        "gbm": {
            "gap_ev": {
                "random": _cell(0.10, 0.09, 0.11, 0.10),
                "isomer_group": _cell(0.20, 0.18, 0.22, 0.20),
                "cluster_ecfp": _cell(0.105, 0.095, 0.115, 0.105),
            }
        }
    }
    res = analysis.rq3_result(cells, ["isomer_group", "cluster_ecfp"], alpha=0.05)
    by = {t["split"]: t for t in res["tests"]}
    assert by["isomer_group"]["inflates"] and not by["cluster_ecfp"]["inflates"]
    assert res["per_split"]["isomer_group"]["verdict"] == "supported"
    assert res["per_split"]["cluster_ecfp"]["verdict"] == "refuted"
    assert res["verdict"] == "mixed"


def test_rq4_verdict_against_threshold() -> None:
    cells = {"gbm": {"u0_atomization_ev": {"random": _cell(0.08, 0.07, 0.09, 0.08)}}}
    assert analysis.rq4_result(cells, 0.123)["verdict"] == "supported"
    cells = {"gbm": {"u0_atomization_ev": {"random": _cell(0.2, 0.19, 0.21, 0.2)}}}
    assert analysis.rq4_result(cells, 0.123)["verdict"] == "refuted"
    cells = {"gbm": {"u0_atomization_ev": {"random": _cell(0.12, 0.11, 0.13, 0.12)}}}
    assert analysis.rq4_result(cells, 0.123)["verdict"] == "inconclusive"


def test_plan_status_detects_edits_before_amendments(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import hashlib

    head = "# Plan\n\nfrozen text\n"
    plan = tmp_path / "plan.md"
    plan.write_text(head + "\n## Amendments\n- **2026-11-09** — something.\n")
    cfg = {
        "analysis_plan": {
            "frozen_commit": "x",
            "frozen_sha256": hashlib.sha256(head.encode()).hexdigest(),
        }
    }
    status = analysis.plan_status(plan, cfg)
    assert status["frozen_text_unchanged"] and len(status["amendments"]) == 1
    plan.write_text(head.replace("frozen", "edited") + "\n## Amendments\nNone.\n")
    assert not analysis.plan_status(plan, cfg)["frozen_text_unchanged"]


def test_rq5_skips_models_outside_coverage(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from qm9_curation.config import load_config

    cfg = load_config("benchmark")
    assert "rq5" not in cfg["coverage"]["dmpnn"] and cfg["not_run"]["dmpnn"]["rq5"]
    sentinel = [{"spec": {"seed": 0}, "key": "missing"}]
    idx = {
        ("rq5", "dmpnn", t, "random", tiers, n): sentinel
        for t in cfg["targets"]
        for tiers in cfg["experiments"]["rq5"]["tier_sets"]
        for n in cfg["experiments"]["rq5"]["n_grid"]
    }
    # Uncovered model: its runs are never read, so no curve and no verdict input.
    out = analysis.rq5_result(tmp_path, idx, cfg)
    assert out["curves"] == {} and out["largest_common_n"] == []


def test_rq1_counts_listed_records_passing_the_independent_check() -> None:
    molecules = pl.DataFrame(
        {
            "uncharacterized_listed": [True, True, False, False],
            "identity_class": ["consistent", "proton_moved", "consistent", "rearranged"],
            # Being on the list counts as one identity failure.
            "n_failed_identity": [1, 2, 0, 1],
            "n_failed_hard": [0, 0, 0, 0],
            "n_failed_soft": [0, 0, 1, 0],
            "quality_tier": ["silver", "bronze", "gold", "bronze"],
            "tier_reasons": ["R4", "R3", "", "R3"],
        }
    )
    out = analysis.rq1_result(molecules)
    assert out["listed_passing_identity"] == 1
    assert out["unlisted_failing_identity"] == 1 and out["verdict"] == "supported"
