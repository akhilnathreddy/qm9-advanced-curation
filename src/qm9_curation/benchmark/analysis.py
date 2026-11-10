"""Apply the frozen analysis plan to the finished runs (Phase 9).

Run as ``python -m qm9_curation.benchmark.analysis``. Reads
``data/benchmark/runs`` and writes:

* ``data/benchmark/run_metrics.parquet`` — one row per run (spec + metrics);
* ``reports/benchmark/benchmark_report.json`` — RQ1-RQ5 results with CIs,
  verdicts, the stress tests, sanity alarms, not-run cells, provenance and
  the Gate 9 conditions;
* ``reports/benchmark/summary.md`` — the result tables in Markdown;
* ``reports/benchmark/figures/`` — learning curves and split MAEs.

Deterministic given the runs (seeded bootstrap), so a forced second run is
byte-identical (Gate 9).

Verdicts (docs/analysis_plan.md, decision rules and the 2026-11-09
clarification): a cell meets its rule when its CI condition holds **and**
its Holm-adjusted one-sided bootstrap (or Mann-Whitney) p-value is below
alpha. A question is ``supported`` when every cell meets its rule,
``refuted`` when no cell does and the evidence goes the other way where the
rule says so, ``mixed`` when some cells do, and ``inconclusive`` otherwise.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from qm9_curation.benchmark.metrics import (
    holm,
    metrics,
    paired_difference_ci,
    unpaired_greater_pvalue,
    unpaired_mae_ci,
)
from qm9_curation.benchmark.models import LEVELS
from qm9_curation.config import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR, load_config
from qm9_curation.review.stats import mann_whitney_greater

RUNS_DIR = DATA_DIR / "benchmark" / "runs"
VERDICTS = ("supported", "refuted", "mixed", "inconclusive")
AMENDMENTS_HEADER = "\n## Amendments\n"


# ---------------------------------------------------------------- loading


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))]
    return sorted(runs, key=lambda r: r["key"])


def predictions(runs_dir: Path, run: Mapping[str, Any], role: str) -> pl.DataFrame:
    table = pl.read_parquet(runs_dir / f"{run['key']}.parquet")
    return table.filter(pl.col("role") == role).sort("qm9_id")


def run_table(runs: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    rows = []
    for r in runs:
        s = r["spec"]
        row = {
            "key": r["key"],
            "experiment": s["experiment"],
            "model": s["model"],
            "level": LEVELS[s["model"]],
            "target": s["target"],
            "split": s["split"],
            "seed": s["seed"],
            "tiers": s["tiers"],
            "n_train_requested": s["n_train"],
            "fold": s["fold"],
            "grid_point": json.dumps(s["grid_point"], sort_keys=True),
            "n_train": r["n_train"],
            "seconds": float(r["seconds"]),
            "mlflow_run_id": r["mlflow_run_id"],
        }
        for k in (
            "val_mae",
            "test_mae",
            "test_rmse",
            "test_r2",
            "test_within_chemical_accuracy",
            "oof_mae",
        ):
            row[k] = float(r["metrics"][k]) if k in r["metrics"] else None
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).sort("key")


def index(runs: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    out: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for r in runs:
        s = r["spec"]
        out[s["experiment"], s["model"], s["target"], s["split"], s["tiers"], s["n_train"]].append(
            r
        )
    for v in out.values():
        v.sort(key=lambda r: (r["spec"]["seed"], r["spec"]["fold"] or 0))
    return out


# ---------------------------------------------------------------- RQ3 / stress


def split_cells(
    runs_dir: Path,
    idx: Mapping[tuple[Any, ...], list[Mapping[str, Any]]],
    experiment: str,
    splits: Sequence[str],
    targets: Sequence[str],
    stats_cfg: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """model -> target -> split -> pooled metrics, per-seed MAE, bootstrap CI."""
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for model in LEVELS:
        for target in targets:
            for split in splits:
                exp = "rq3" if experiment == "rq3" else "stress"
                runs = idx.get((exp, model, target, split, "all", None))
                if not runs:
                    continue
                preds = [predictions(runs_dir, r, "test") for r in runs]
                errs = [np.abs(p["pred"].to_numpy() - p["y"].to_numpy()) for p in preds]
                ci = unpaired_mae_ci(
                    errs, int(stats_cfg["n_boot"]), int(stats_cfg["bootstrap_seed"])
                )
                pooled = pl.concat(preds)
                cell = {
                    "seeds": [r["spec"]["seed"] for r in runs],
                    "mae_per_seed": [float(e.mean()) for e in errs],
                    "val_mae_per_seed": [float(r["metrics"]["val_mae"]) for r in runs],
                    "n_train_mean": float(np.mean([r["n_train"] for r in runs])),
                    "pooled": metrics(pooled["y"].to_numpy(), pooled["pred"].to_numpy()),
                    "mae": ci["mae"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "_boot": ci["boot"],
                }
                out[model][target][split] = cell
    return out


def rq3_result(cells: Mapping[str, Any], others: Sequence[str], alpha: float) -> dict[str, Any]:
    tests = []
    for model, by_target in cells.items():
        for target, by_split in by_target.items():
            ref = by_split.get("random")
            if ref is None:
                continue
            for split in others:
                other = by_split.get(split)
                if other is None:
                    continue
                tests.append(
                    {
                        "model": model,
                        "target": target,
                        "split": split,
                        "random_mae": ref["mae"],
                        "random_ci_high": ref["ci_high"],
                        "split_mae": other["mae"],
                        "split_ci_low": other["ci_low"],
                        "ratio": other["mae"] / ref["mae"],
                        "ci_rule": ref["ci_high"] < other["ci_low"],
                        "p": unpaired_greater_pvalue(other["_boot"], ref["_boot"]),
                    }
                )
    for t, p_adj in zip(tests, holm([t["p"] for t in tests]), strict=True):
        t["p_holm"] = p_adj
        t["inflates"] = bool(t["ci_rule"] and p_adj < alpha)
    per_split = {}
    for split in others:
        cell = [t["inflates"] for t in tests if t["split"] == split]
        per_split[split] = {
            "cells": len(cell),
            "cells_inflating": int(sum(cell)),
            "verdict": _all_some_none(cell),
        }
    verdicts = [v["verdict"] for v in per_split.values()]
    overall = (
        "supported"
        if verdicts and all(v == "supported" for v in verdicts)
        else "refuted"
        if verdicts and all(v == "refuted" for v in verdicts)
        else "mixed"
        if verdicts
        else "inconclusive"
    )
    return {"tests": tests, "per_split": per_split, "verdict": overall}


def _all_some_none(flags: Sequence[bool]) -> str:
    if not flags:
        return "inconclusive"
    if all(flags):
        return "supported"
    return "mixed" if any(flags) else "refuted"


# ---------------------------------------------------------------- RQ1


def rq1_result(molecules: pl.DataFrame) -> dict[str, Any]:
    unlisted = molecules.filter(~pl.col("uncharacterized_listed"))
    listed = molecules.filter(pl.col("uncharacterized_listed"))
    hard = pl.col("n_failed_hard") > 0
    ident = pl.col("n_failed_identity") > 0
    extra = unlisted.filter(hard | ident)
    reasons = (
        unlisted.filter(pl.col("quality_tier") != "gold")
        .group_by("quality_tier", "tier_reasons")
        .len()
        .sort("len", "tier_reasons", descending=[True, False])
    )
    return {
        "listed_uncharacterized": listed.height,
        "unlisted_records": unlisted.height,
        "unlisted_failing_hard_or_identity": extra.height,
        "unlisted_failing_hard": int(unlisted.filter(hard).height),
        "unlisted_failing_identity": int(unlisted.filter(ident).height),
        "unlisted_with_soft_flags": int((unlisted["n_failed_soft"] > 0).sum()),
        "unlisted_not_gold_by_reason": [
            {"tier": t, "reasons": r, "n": int(n)} for t, r, n in reasons.rows()
        ],
        # The independent check only: n_failed_identity also counts being on the
        # list itself, so every listed record "fails" it.
        "listed_passing_identity": int(
            listed.filter(pl.col("identity_class") == "consistent").height
        ),
        "test": "descriptive",
        "verdict": "supported" if extra.height > 0 else "refuted",
        "note": (
            "The plan defines no threshold for 'non-trivial'. The verdict records that "
            "records beyond the list fail at least one hard or identity check; the "
            "count itself is the result."
        ),
    }


# ---------------------------------------------------------------- RQ2


def rq2_result(
    runs_dir: Path,
    idx: Mapping[tuple[Any, ...], list[Mapping[str, Any]]],
    molecules: pl.DataFrame,
    targets: Sequence[str],
    alpha: float,
) -> dict[str, Any]:
    tiers = molecules.select("qm9_id", "quality_tier")
    tests = []
    models = {}
    for target in targets:
        runs = [r for k, v in idx.items() if k[0] == "rq2" and k[2] == target for r in v]
        if not runs:
            continue
        models[target] = runs[0]["spec"]["model"]
        oof = pl.concat([predictions(runs_dir, r, "oof") for r in runs]).join(tiers, on="qm9_id")
        oof = oof.with_columns((pl.col("pred") - pl.col("y")).abs().alias("abs_residual"))
        gold = oof.filter(pl.col("quality_tier") == "gold")["abs_residual"].to_numpy()
        for tier in ("silver", "bronze"):
            other = oof.filter(pl.col("quality_tier") == tier)["abs_residual"].to_numpy()
            if other.size == 0:
                continue
            mw = mann_whitney_greater(other, gold)
            tests.append(
                {
                    "target": target,
                    "model": models[target],
                    "tier": tier,
                    **mw,
                    "mae_tier": float(other.mean()),
                    "mae_gold": float(gold.mean()),
                }
            )
    for t, p_adj in zip(tests, holm([t["pvalue"] for t in tests]), strict=True):
        t["p_holm"] = p_adj
        t["significant"] = bool(p_adj < alpha and t["prob_a_greater"] > 0.5)
    return {
        "model_per_target": models,
        "tests": tests,
        "verdict": _all_some_none([t["significant"] for t in tests]),
    }


# ---------------------------------------------------------------- RQ4


def rq4_result(
    cells: Mapping[str, Any], threshold: float, target: str = "u0_atomization_ev"
) -> dict[str, Any]:
    random = {m: c[target]["random"] for m, c in cells.items() if "random" in c.get(target, {})}
    if not random:
        return {"verdict": "inconclusive", "reason": "no random-split runs"}
    best = min(random, key=lambda m: random[m]["mae"])
    c = random[best]
    verdict = (
        "supported"
        if c["ci_high"] < threshold
        else "refuted"
        if c["ci_low"] > threshold
        else "inconclusive"
    )
    return {
        "best_model": best,
        "mae": c["mae"],
        "ci_low": c["ci_low"],
        "ci_high": c["ci_high"],
        "threshold_ev": threshold,
        "all_models": {m: {k: random[m][k] for k in ("mae", "ci_low", "ci_high")} for m in random},
        "verdict": verdict,
    }


# ---------------------------------------------------------------- RQ5


def rq5_result(
    runs_dir: Path,
    idx: Mapping[tuple[Any, ...], list[Mapping[str, Any]]],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    r5, st = cfg["experiments"]["rq5"], cfg["statistics"]
    tier_sets = list(r5["tier_sets"])
    curves: dict[str, Any] = {}
    finals = []
    for model in r5["models"]:
        if "rq5" not in cfg["coverage"].get(model, []):
            continue  # reported under not_run
        for target in cfg["targets"]:
            points = []
            for n in r5["n_grid"]:
                by_tier = {
                    t: idx.get(("rq5", model, target, r5["split"], t, n), []) for t in tier_sets
                }
                if not all(by_tier.values()):
                    continue
                seeds = sorted(
                    set.intersection(*({r["spec"]["seed"] for r in v} for v in by_tier.values()))
                )
                errs: dict[str, dict[int, pl.DataFrame]] = {}
                for t, v in by_tier.items():
                    errs[t] = {
                        r["spec"]["seed"]: predictions(runs_dir, r, "test")
                        .with_columns((pl.col("pred") - pl.col("y")).abs().alias("e"))
                        .select("qm9_id", "e")
                        for r in v
                        if r["spec"]["seed"] in seeds
                    }
                point: dict[str, Any] = {"n": n, "seeds": seeds}
                for t in tier_sets:
                    point[f"mae_{t}"] = float(np.mean([errs[t][s]["e"].mean() for s in seeds]))
                for t in tier_sets[1:]:
                    diffs = []
                    for s in seeds:
                        joined = errs[t][s].join(errs["all"][s], on="qm9_id", suffix="_all")
                        if joined.height != errs["all"][s].height:
                            raise ValueError("RQ5 test sets differ between tier sets")
                        diffs.append((joined["e"] - joined["e_all"]).to_numpy())
                    point[f"diff_{t}_minus_all"] = paired_difference_ci(
                        diffs, int(st["n_boot"]), int(st["bootstrap_seed"])
                    )
                points.append(point)
            if not points:
                continue
            curves.setdefault(model, {})[target] = points
            last = points[-1]
            d = last["diff_gold_minus_all"]
            finals.append(
                {
                    "model": model,
                    "target": target,
                    "n": last["n"],
                    **{k: d[k] for k in ("estimate", "ci_low", "ci_high")},
                    "p_less": d["p_less_than_zero"],
                    "p_greater": d["p_greater_than_zero"],
                }
            )
    alpha = float(st["alpha"])
    for key in ("p_less", "p_greater"):
        for f, p_adj in zip(finals, holm([f[key] for f in finals]), strict=True):
            f[f"{key}_holm"] = p_adj
    for f in finals:
        f["gold_not_worse"] = bool(f["ci_high"] <= 0 and f["p_less_holm"] < alpha)
        f["gold_worse"] = bool(f["ci_low"] > 0 and f["p_greater_holm"] < alpha)
    primary = [f for f in finals if f["model"] == r5["models"][0]]
    verdict = (
        "supported"
        if primary and all(f["gold_not_worse"] for f in primary)
        else "refuted"
        if primary and all(f["gold_worse"] for f in primary)
        else "mixed"
        if any(f["gold_not_worse"] or f["gold_worse"] for f in primary)
        else "inconclusive"
    )
    return {
        "primary_model": r5["models"][0],
        "curves": curves,
        "largest_common_n": finals,
        "verdict": verdict,
    }


# ---------------------------------------------------------------- learning-curve slopes


def slopes(curves: Mapping[str, Any]) -> dict[str, Any]:
    """Least-squares slope of log MAE on log N, per model, target and tier set."""
    out: dict[str, Any] = {}
    for model, by_target in curves.items():
        for target, points in by_target.items():
            n = np.log([p["n"] for p in points])
            for key in [k for k in points[0] if k.startswith("mae_")]:
                mae = np.log([p[key] for p in points])
                out.setdefault(model, {}).setdefault(target, {})[key[4:]] = (
                    float(np.polyfit(n, mae, 1)[0]) if len(points) > 1 else None
                )
    return out


# ---------------------------------------------------------------- plan and gate


def plan_status(plan_path: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    text = plan_path.read_text(encoding="utf-8")
    head, _, amendments = text.partition(AMENDMENTS_HEADER)
    entries = [line for line in amendments.splitlines() if line.startswith("- **20")]
    return {
        "frozen_commit": cfg["analysis_plan"]["frozen_commit"],
        "frozen_sha256": cfg["analysis_plan"]["frozen_sha256"],
        "current_sha256_before_amendments": hashlib.sha256(head.encode()).hexdigest(),
        "frozen_text_unchanged": hashlib.sha256(head.encode()).hexdigest()
        == cfg["analysis_plan"]["frozen_sha256"],
        "amendments": entries,
    }


def expected_cells(cfg: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    """(experiment, model, target, split) cells the coverage says must exist (5 seeds each)."""
    cells = []
    for model, cov in cfg["coverage"].items():
        for target in cfg["targets"]:
            for exp in ("rq3", "stress"):
                for split in cfg["experiments"][exp]["splits"]:
                    if exp in cov or (exp == "rq3" and split == "random" and "rq3_random" in cov):
                        cells.append((exp, model, target, split))
    return cells


def gate_conditions(
    report: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    mlruns: Path | None,
) -> dict[str, bool]:
    idx = index(runs)
    seeds = set(cfg["seeds"])
    cells_ok = all(
        {r["spec"]["seed"] for r in idx.get((e, m, t, s, "all", None), [])} == seeds
        for e, m, t, s in expected_cells(cfg)
    )
    keys = [
        (
            r["spec"]["experiment"],
            r["spec"]["model"],
            r["spec"]["target"],
            r["spec"]["split"],
            r["spec"]["seed"],
            r["spec"]["tiers"],
            r["spec"]["n_train"],
            r["spec"]["fold"],
        )
        for r in runs
        if r["spec"]["experiment"] != "tuning"
    ]
    tracked = all(
        r.get("mlflow_run_id")
        and r["provenance"].get("git_commit")
        and r["provenance"].get("dataset_version")
        for r in runs
    )
    if mlruns is not None:
        found = {p.name for p in mlruns.glob("*/*") if p.is_dir()}
        tracked = tracked and all(r.get("mlflow_run_id") in found for r in runs)
    rq = report["research_questions"]
    with_ci = [
        *(t for t in rq["rq3"]["tests"]),
        rq["rq4"],
        *rq["rq5"]["largest_common_n"],
    ]
    return {
        "analysis_plan_frozen_text_unchanged": bool(
            report["analysis_plan"]["frozen_text_unchanged"]
        ),
        "every_rq_has_a_verdict": all(
            rq[k]["verdict"] in VERDICTS for k in ("rq1", "rq2", "rq3", "rq4", "rq5")
        ),
        "confirmatory_results_have_cis": bool(rq["rq3"]["tests"])
        and bool(rq["rq5"]["largest_common_n"])
        and all(np.isfinite(c.get("ci_low", c.get("split_ci_low", np.nan))) for c in with_ci),
        "every_planned_cell_has_every_seed": cells_ok,
        "test_predicted_once_per_cell": len(keys) == len(set(keys)),
        "tuning_runs_never_predict_test": all(
            "test_mae" not in r["metrics"] for r in runs if r["spec"]["experiment"] == "tuning"
        ),
        "every_run_tracked_with_commit_and_data_version": bool(tracked),
        "no_unexplained_sanity_alarm": not report["sanity_alarms"],
        "not_run_cells_have_reasons": all(
            isinstance(v, str) and v for d in cfg["not_run"].values() for v in d.values()
        ),
    }


def sanity_alarms(cells: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    s = cfg["statistics"]["sanity"]
    alarms = []
    for model, by_target in cells.items():
        for target, by_split in by_target.items():
            ref = s["published_best_ev"].get(target)
            if (
                ref is not None
                and "random" in by_split
                and by_split["random"]["mae"] < s["factor"] * ref
            ):
                alarms.append(
                    {
                        "model": model,
                        "target": target,
                        "mae": by_split["random"]["mae"],
                        "published": ref,
                    }
                )
    return alarms


def strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def build(
    runs_dir: Path,
    processed: Path,
    plan_path: Path,
    cfg: dict[str, Any],
    mlruns: Path | None,
) -> tuple[dict[str, Any], pl.DataFrame]:
    runs = load_runs(runs_dir)
    idx = index(runs)
    targets = list(cfg["targets"])
    st = cfg["statistics"]
    rq3_splits = cfg["experiments"]["rq3"]["splits"]
    cells = split_cells(runs_dir, idx, "rq3", rq3_splits, targets, st)
    stress = split_cells(
        runs_dir, idx, "stress", cfg["experiments"]["stress"]["splits"], targets, st
    )
    molecules = pl.read_parquet(processed / "molecules.parquet")
    rq5 = rq5_result(runs_dir, idx, cfg)
    tuning: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in runs:
        if r["spec"]["experiment"] == "tuning":
            s = r["spec"]
            tuning.setdefault(s["model"], {}).setdefault(s["target"], []).append(
                {**s["grid_point"], "val_mae": r["metrics"]["val_mae"]}
            )
    report: dict[str, Any] = {
        "config": cfg,
        "analysis_plan": plan_status(plan_path, cfg),
        "n_runs": len(runs),
        "n_runs_by_experiment": dict(
            sorted(
                {
                    e: sum(r["spec"]["experiment"] == e for r in runs)
                    for e in {r["spec"]["experiment"] for r in runs}
                }.items()
            )
        ),
        "compute_hours": sum(float(r["seconds"]) for r in runs) / 3600.0,
        "provenance": sorted({json.dumps(r["provenance"], sort_keys=True) for r in runs}),
        "tuning": {
            m: {t: sorted(v, key=lambda x: json.dumps(x, sort_keys=True)) for t, v in d.items()}
            for m, d in tuning.items()
        },
        "rq3_cells": cells,
        "stress_cells": stress,
        "research_questions": {
            "rq1": rq1_result(molecules),
            "rq2": rq2_result(runs_dir, idx, molecules, targets, float(st["alpha"])),
            "rq3": rq3_result(cells, rq3_splits[1:], float(st["alpha"])),
            "rq4": rq4_result(cells, float(st["rq4_threshold_ev"])),
            "rq5": rq5,
        },
        "rq5_learning_curve_slopes": slopes(rq5["curves"]),
        "sanity_alarms": sanity_alarms(cells, cfg),
        "not_run": cfg["not_run"],
    }
    report["gate_9_conditions"] = gate_conditions(report, runs, cfg, mlruns)
    return strip_private(report), run_table(runs)


# ---------------------------------------------------------------- outputs


def _row(*cells: Any) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def summary_markdown(report: Mapping[str, Any]) -> str:
    rq = report["research_questions"]
    lines = [
        "# Phase 9 benchmark summary",
        "",
        "Generated by `qm9_curation.benchmark.analysis`.",
        "",
    ]
    lines += ["## Verdicts", "", "| RQ | Verdict |", "|---|---|"]
    lines += [f"| {k.upper()} | {rq[k]['verdict']} |" for k in ("rq1", "rq2", "rq3", "rq4", "rq5")]
    for name, cells in (
        ("RQ3 splits", report["rq3_cells"]),
        ("Stress tests", report["stress_cells"]),
    ):
        lines += ["", f"## {name}: Gold-test MAE (eV), pooled over seeds [95% CI]", ""]
        for target in report["config"]["targets"]:
            splits = sorted(
                {s for m in cells.values() for s in m.get(target, {})},
                key=list(
                    [
                        *report["config"]["experiments"]["rq3"]["splits"],
                        *report["config"]["experiments"]["stress"]["splits"],
                    ]
                ).index,
            )
            lines += [
                f"**{target}**",
                "",
                "| Level | Model | " + " | ".join(splits) + " |",
                "|---|---|" + "---|" * len(splits),
            ]
            for model in sorted(cells, key=LEVELS.__getitem__):
                row = cells[model].get(target, {})
                vals = [
                    f"{row[s]['mae']:.4f} [{row[s]['ci_low']:.4f}, {row[s]['ci_high']:.4f}]"
                    if s in row
                    else "not run"
                    for s in splits
                ]
                lines.append(f"| {LEVELS[model]} | {model} | " + " | ".join(vals) + " |")
            lines.append("")
    lines += [
        "## RQ3 inflation tests (other split vs. random)",
        "",
        "| Model | Target | Split | Ratio | CI rule | p (Holm) | Inflates |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in rq["rq3"]["tests"]:
        lines.append(
            _row(t["model"], t["target"], t["split"], f"{t['ratio']:.2f}", t["ci_rule"],
                 f"{t['p_holm']:.4f}", t["inflates"])
        )  # fmt: skip
    lines += [
        "",
        "## RQ2 (out-of-fold |residual|, one-sided Mann-Whitney vs. Gold)",
        "",
        "| Target | Model | Tier | n | MAE tier | MAE Gold | P(tier > Gold) | p (Holm) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in rq["rq2"]["tests"]:
        lines.append(
            _row(t["target"], t["model"], t["tier"], t["n_a"], f"{t['mae_tier']:.4f}",
                 f"{t['mae_gold']:.4f}", f"{t['prob_a_greater']:.3f}", f"{t['p_holm']:.2e}")
        )  # fmt: skip
    r4 = rq["rq4"]
    if "mae" in r4:
        lines += [
            "",
            "## RQ4",
            "",
            f"Best random-split U0 atomization MAE: {r4['best_model']} {r4['mae']:.4f} eV "
            f"[{r4['ci_low']:.4f}, {r4['ci_high']:.4f}] vs. {r4['threshold_ev']} eV: "
            f"{r4['verdict']}.",
        ]
    lines += ["", "## RQ5 (Gold-test MAE, eV, seed mean)", ""]
    for model, by_target in rq["rq5"]["curves"].items():
        for target, points in by_target.items():
            lines += [
                f"**{model} / {target}**",
                "",
                "| N | all | gold_silver | gold | gold - all [95% CI] |",
                "|---|---|---|---|---|",
            ]
            for p in points:
                d = p["diff_gold_minus_all"]
                lines.append(
                    _row(p["n"], f"{p['mae_all']:.4f}", f"{p['mae_gold_silver']:.4f}",
                         f"{p['mae_gold']:.4f}",
                         f"{d['estimate']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
                )  # fmt: skip
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def figures(report: Mapping[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    curves = report["research_questions"]["rq5"]["curves"]
    targets = list(report["config"]["targets"])
    if curves:
        fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), squeeze=False)
        for ax, target in zip(axes[0], targets, strict=True):
            for model, by_target in curves.items():
                for tiers, style in (("all", "-"), ("gold_silver", "--"), ("gold", ":")):
                    pts = by_target.get(target, [])
                    if pts:
                        ax.loglog(
                            [p["n"] for p in pts],
                            [p[f"mae_{tiers}"] for p in pts],
                            style,
                            marker="o",
                            ms=3,
                            label=f"{model} / {tiers}",
                        )
            ax.set_xlabel("training molecules N")
            ax.set_ylabel("Gold-test MAE (eV)")
            ax.set_title(target)
            ax.legend(fontsize=7)
        fig.tight_layout()
        paths.append(out_dir / "rq5_learning_curves.png")
        fig.savefig(paths[-1], dpi=120, metadata={"Software": None})
        plt.close(fig)
    cells = report["rq3_cells"]
    splits = report["config"]["experiments"]["rq3"]["splits"]
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 4), squeeze=False)
    width = 0.8 / max(1, len(cells))
    for ax, target in zip(axes[0], targets, strict=True):
        for k, model in enumerate(sorted(cells, key=LEVELS.__getitem__)):
            row = cells[model].get(target, {})
            xs = [i + k * width for i, s in enumerate(splits) if s in row]
            ys = [row[s]["mae"] for s in splits if s in row]
            err = [
                [row[s]["mae"] - row[s]["ci_low"] for s in splits if s in row],
                [row[s]["ci_high"] - row[s]["mae"] for s in splits if s in row],
            ]
            ax.bar(xs, ys, width, yerr=err, label=model)
        ax.set_xticks(
            [i + 0.4 - width / 2 for i in range(len(splits))], splits, rotation=20, fontsize=8
        )
        ax.set_yscale("log")
        ax.set_ylabel("Gold-test MAE (eV)")
        ax.set_title(target)
        ax.legend(fontsize=7)
    fig.tight_layout()
    paths.append(out_dir / "rq3_mae_by_split.png")
    fig.savefig(paths[-1], dpi=120, metadata={"Software": None})
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "docs" / "analysis_plan.md")
    parser.add_argument("--mlruns", type=Path, default=PROJECT_ROOT / "mlruns")
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR / "benchmark")
    parser.add_argument(
        "--metrics-out", type=Path, default=DATA_DIR / "benchmark" / "run_metrics.parquet"
    )
    args = parser.parse_args(argv)

    cfg = load_config("benchmark")
    mlruns = args.mlruns if args.mlruns.exists() else None
    report, table = build(args.runs_dir, args.processed_dir, args.plan, cfg, mlruns)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.metrics_out, compression="zstd", compression_level=9, statistics=True)
    summary = args.out_dir / "summary.md"
    summary.write_text(summary_markdown(report), encoding="utf-8")
    figures(report, args.out_dir / "figures")
    report["output_sha256"] = {
        "run_metrics": hashlib.sha256(args.metrics_out.read_bytes()).hexdigest(),
        "summary": hashlib.sha256(summary.read_bytes()).hexdigest(),
    }
    (args.out_dir / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=float) + "\n",
        encoding="utf-8",
    )
    for condition, ok in report["gate_9_conditions"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {condition}")


if __name__ == "__main__":
    main()
