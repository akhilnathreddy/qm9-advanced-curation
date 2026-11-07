"""Statistical, scientific and multi-fidelity validation (Phase 7).

Run as ``python -m qm9_curation.review.build``. Outputs:

* ``data/processed/review.parquet`` — one row per record: composition,
  thermochemistry, multi-fidelity and model-residual scores, and the soft
  ``review_reasons`` they raise. Tiers are copied, never changed.
* ``data/processed/g4mp2_xref.parquet`` — G4(MP2) values of Kim et al. (2019,
  CC0) joined by QM9 index, with status and B3LYP - G4(MP2) differences.
* ``reports/statistical/statistical_report.json`` — distribution tests,
  correlations, residual tables, the multi-fidelity analysis, cross-checks
  against the authors' findings, and the Gate 7 conditions.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from qm9_curation.config import PROCESSED_DIR, REPORTS_DIR, load_config
from qm9_curation.curate.build import STANDARD_TARGETS
from qm9_curation.curate.tiers import TIER_ORDER
from qm9_curation.external import kim2019
from qm9_curation.external.acquire import KIM2019_DIR
from qm9_curation.review import multifidelity, scientific
from qm9_curation.review.stats import compare_groups, mann_whitney_greater, summary
from qm9_curation.review.suspicion import suspicion_scores

# Soft review reasons, each from a robust z above the review threshold or a
# status reported by the external source.
REASONS = {
    "composition_outlier": "composition_robust_z",
    "thermal_energy_outlier": "thermal_energy_robust_z",
    "entropy_term_outlier": "minus_ts_robust_z",
    "multi_fidelity_outlier": "delta_robust_z",
    "model_suspicion": "suspicion_score",
}
STATUS_REASONS = {
    "geometry_changed": "g4mp2_geometry_changed",
    "not_converged": "g4mp2_not_converged",
}


def review_table(
    curated: pl.DataFrame,
    composition_res: pl.DataFrame,
    thermal: pl.DataFrame,
    xref: pl.DataFrame,
    suspicion: pl.DataFrame,
    threshold: float,
) -> pl.DataFrame:
    frame = (
        curated.select("qm9_id", "quality_tier")
        .join(composition_res, on="qm9_id", how="left")
        .join(thermal, on="qm9_id", how="left")
        .join(
            xref.select("qm9_id", "g4mp2_status", "delta_residual_ev", "delta_robust_z"),
            on="qm9_id",
            how="left",
        )
        .join(suspicion, on="qm9_id", how="left")
    )
    flags = [
        pl.when(pl.col(col).abs() > threshold).then(pl.lit(reason)).otherwise(None)
        for reason, col in REASONS.items()
    ] + [
        pl.when(pl.col("g4mp2_status") == status).then(pl.lit(reason)).otherwise(None)
        for status, reason in STATUS_REASONS.items()
    ]
    return (
        frame.with_columns(pl.concat_list(flags).list.drop_nulls().list.sort().alias("_r"))
        .with_columns(
            pl.col("_r").list.join(",").alias("review_reasons"),
            pl.col("_r").list.len().cast(pl.UInt8).alias("n_review_reasons"),
        )
        .drop("_r")
        .sort("qm9_id")
    )


def reason_counts(review: pl.DataFrame) -> dict[str, Any]:
    exploded = review.select(
        "quality_tier", pl.col("review_reasons").str.split(",").alias("r")
    ).explode("r")
    exploded = exploded.filter(pl.col("r") != "")
    out: dict[str, Any] = {}
    for reason in [*REASONS, *STATUS_REASONS.values()]:
        sub = exploded.filter(pl.col("r") == reason)
        out[reason] = {
            "total": sub.height,
            **dict(sorted(sub.group_by("quality_tier").len().rows())),
        }
    out["records_with_any_reason"] = int((review["n_review_reasons"] > 0).sum())
    return out


def rq2_preliminary(review: pl.DataFrame, target: str) -> dict[str, Any]:
    """Out-of-fold |residual| of non-Gold vs Gold records (Phase 9 repeats with stronger models)."""
    col = f"oof_residual_{target}"
    frame = review.filter(pl.col(col).is_not_null()).with_columns(pl.col(col).abs().alias("a"))
    gold = frame.filter(pl.col("quality_tier") == "gold")["a"].to_numpy()
    out = {
        "non_gold_vs_gold": mann_whitney_greater(
            frame.filter(pl.col("quality_tier") != "gold")["a"].to_numpy(), gold
        )
    }
    for tier in ("silver", "bronze"):
        out[f"{tier}_vs_gold"] = mann_whitney_greater(
            frame.filter(pl.col("quality_tier") == tier)["a"].to_numpy(), gold
        )
    return out


def gate_conditions(
    review: pl.DataFrame,
    curated: pl.DataFrame,
    xref: pl.DataFrame,
    composition: pl.DataFrame,
    kim: kim2019.Kim2019,
    info: dict[str, Any],
    report: dict[str, Any],
    ext_cfg: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, bool]:
    expected = ext_cfg["expected"]
    observed = {
        "n_molecules": kim.molecules.height,
        "n_not_converged": len(kim.not_converged),
        "n_multi_molecule": len(kim.multi_molecule),
        "n_duplicate_pairs": len(kim.duplicate_pairs),
        "n_energy_change_above_1_kcal_mol": int((xref["g4mp2_status"] == "geometry_changed").sum()),
    }
    published = {f["name"]: f["md5"] for f in ext_cfg["files"]}
    with_invalid = xref.join(composition, on="qm9_id").filter(
        pl.any_horizontal(pl.col(f"n_{e}") > 0 for e in info["invalid_reference_elements"])
        if info["invalid_reference_elements"]
        else pl.lit(False)
    )
    dist = report["distributions"]["vs_gold"]
    tops = report["composition"]["largest_residuals"]
    finite = all(
        np.isfinite(v["ks_statistic"]) and np.isfinite(v["wasserstein"])
        for per_target in dist.values()
        for v in per_target.values()
    )
    known = {*REASONS, *STATUS_REASONS.values()}
    used = {r for s in review["review_reasons"].to_list() for r in s.split(",") if r}
    return {
        "external_source_verified": manifest["license"] == ext_cfg["license"]
        and all(f["md5"] == published[f["name"]] for f in manifest["files"]),
        "external_counts_match_publication": observed == expected,
        "external_formulas_match_qm9": bool(info["formulas_agree"]),
        "authors_atomization_reproduced": report["multi_fidelity"][
            "authors_atomization_max_abs_diff_hartree"
        ]
        < 1e-9,
        "atomization_withheld_where_reference_invalid": bool(
            with_invalid["g4mp2_u0_atomization_ev"].is_null().all()
        ),
        "one_review_row_per_molecule": review.height == curated.height
        and review["qm9_id"].is_unique().all(),
        "review_flags_do_not_change_tiers": review.select("qm9_id", "quality_tier").equals(
            curated.select("qm9_id", "quality_tier").sort("qm9_id")
        ),
        "distribution_tests_cover_every_target_and_tier": finite
        and set(dist) == set(STANDARD_TARGETS),
        "largest_residuals_annotated": all(r["notes"] for r in tops),
        "review_reasons_known": used <= known,
    }


def build(
    curated: pl.DataFrame,
    composition: pl.DataFrame,
    ecfp: pl.DataFrame,
    kim: kim2019.Kim2019,
    stat_cfg: dict[str, Any],
    ext_cfg: dict[str, Any],
    authors_hard_to_converge: list[int],
) -> tuple[dict[str, pl.DataFrame], dict[str, Any], dict[str, Any]]:
    threshold = float(stat_cfg["robust_z_review"])
    top_n = int(stat_cfg["top_n"])
    comp_res, comp_coefs = scientific.composition_residuals(curated, composition)
    thermal = scientific.thermal_scores(curated)
    xref, info = multifidelity.xref_table(
        curated,
        composition,
        kim,
        ext_cfg["ground_state_multiplicity"],
        float(stat_cfg["multi_fidelity"]["geometry_change_hartree"]),
    )
    sus_cfg = stat_cfg["suspicion"]
    suspicion = suspicion_scores(
        curated,
        composition,
        ecfp,
        sus_cfg["targets"],
        int(sus_cfg["n_folds"]),
        float(sus_cfg["ridge_alpha"]),
    )
    review = review_table(curated, comp_res, thermal, xref, suspicion, threshold)

    tiers = [
        t
        for t in TIER_ORDER
        if t != "gold"
        and curated.filter(pl.col("quality_tier") == t).height >= stat_cfg["min_tier_size"]
    ]
    comp_top = scientific.annotate_top(
        comp_res, curated, "composition_residual_ev", "composition_robust_z", top_n
    )
    sus_top = scientific.annotate_top(
        suspicion, curated, "suspicion_score", "suspicion_score", top_n
    )
    n_offset = multifidelity.nitrogen_offset(curated, composition, xref, kim)
    authors_diff = multifidelity.authors_atomization_reproduced(kim)
    report: dict[str, Any] = {
        "review_threshold_robust_z": threshold,
        "distributions": {
            "reference": "gold",
            "groups": ["all", *tiers],
            "vs_gold": compare_groups(curated, STANDARD_TARGETS, "quality_tier", "gold", tiers),
        },
        "inter_target": {
            "atomization_collinearity": scientific.collinearity(curated),
            **scientific.thermal_summary(thermal, curated),
        },
        "composition": {
            "fit_on": "gold",
            "coefficients_ev": comp_coefs,
            "residual_gold": summary(
                comp_res.join(curated.select("qm9_id", "quality_tier"), on="qm9_id")
                .filter(pl.col("quality_tier") == "gold")["composition_residual_ev"]
                .to_numpy()
            ),
            "largest_residuals": comp_top,
            "descriptor_rates_top_vs_gold": scientific.note_counts(comp_top, curated),
        },
        "multi_fidelity": {
            "source": {
                k: ext_cfg[k] for k in ("name", "paper", "paper_doi", "collection_doi", "license")
            },
            "atom_references": kim.atoms.to_dicts(),
            "invalid_reference_elements": info["invalid_reference_elements"],
            "nitrogen_offset_ev": {
                "fitted_with_published_references": n_offset["n_N"],
                "n_doublet_excitation_nist_ev": ext_cfg["n_doublet_excitation_ev"],
                "all_coefficients_ev": n_offset,
            },
            "total_energy_fit_ev": info["total_energy_fit"],
            "status_counts": dict(sorted(xref.group_by("g4mp2_status").len().rows())),
            "authors_atomization_max_abs_diff_hartree": authors_diff,
            **multifidelity.delta_statistics(xref, curated, top_n),
        },
        "cross_checks": multifidelity.cross_checks(xref, curated, kim, authors_hard_to_converge),
        "suspicion": {
            "model": "ridge on element counts + ECFP4 counts, out-of-fold",
            **sus_cfg,
            "oof_mae": {
                t: float(np.abs(suspicion[f"oof_residual_{t}"].to_numpy()).mean())
                for t in sus_cfg["targets"]
            },
            "largest": sus_top,
            "rq2_preliminary": {t: rq2_preliminary(review, t) for t in sus_cfg["targets"]},
        },
        "review_reasons": reason_counts(review),
    }
    return {"review": review, "g4mp2_xref": xref}, report, info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--external-dir", type=Path, default=KIM2019_DIR)
    parser.add_argument(
        "--report", type=Path, default=REPORTS_DIR / "statistical" / "statistical_report.json"
    )
    args = parser.parse_args(argv)

    ext_cfg = load_config("external")["kim2019"]
    stat_cfg = load_config("statistical")
    hard = load_config("dataset")["known_issues"]["hard_to_converge"]
    curated = pl.read_parquet(args.processed_dir / "molecules.parquet")
    feats = args.processed_dir / "features"
    composition = pl.read_parquet(feats / "composition.parquet")
    ecfp = pl.read_parquet(feats / "ecfp4.parquet")
    kim = kim2019.load(args.external_dir / ext_cfg["files"][0]["name"], ext_cfg["members"])
    manifest = json.loads((args.external_dir / "MANIFEST.json").read_text(encoding="utf-8"))

    tables, report, info = build(curated, composition, ecfp, kim, stat_cfg, ext_cfg, hard)
    report["gate_7_conditions"] = gate_conditions(
        tables["review"],
        curated,
        tables["g4mp2_xref"],
        composition,
        kim,
        info,
        report,
        ext_cfg,
        manifest,
    )
    report["output_sha256"] = {}
    for name, frame in tables.items():
        out = args.processed_dir / f"{name}.parquet"
        frame.write_parquet(out, compression="zstd", compression_level=9, statistics=True)
        report["output_sha256"][name] = hashlib.sha256(out.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    for condition, ok in report["gate_7_conditions"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {condition}")
    print(
        "review reasons:",
        {k: v["total"] if isinstance(v, dict) else v for k, v in report["review_reasons"].items()},
    )


if __name__ == "__main__":
    main()
