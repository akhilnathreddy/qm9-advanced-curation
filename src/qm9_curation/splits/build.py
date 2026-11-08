"""Build the split suite and its leakage audit (Phase 8).

Run as ``python -m qm9_curation.splits.build``. Outputs:

* ``data/splits/splits.parquet`` — long format ``(qm9_id, split_name, seed,
  subset)``: every split assigns all records for every seed;
* ``data/splits/split_groups.parquet`` — per record: leakage group, formula,
  scaffold key, ECFP cluster, and the split unit of each split;
* ``reports/splits/split_report.json`` — group statistics, leakage
  assertions per split and seed, the seed-0 distribution-shift report, and the
  Gate 8 conditions.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from qm9_curation.config import PROCESSED_DIR, REPORTS_DIR, SPLITS_DIR, load_config
from qm9_curation.parallel import default_workers
from qm9_curation.splits import assign
from qm9_curation.splits.audit import check_assignment, naive_random_leakage, shift_report
from qm9_curation.splits.fingerprints import bit_matrix, leader_clusters
from qm9_curation.splits.groups import LEAKAGE_KEYS, components, leakage_groups, scaffold_keys

GROUP_COLUMNS = {
    "leakage": None,
    "formula": "formula",
    "scaffold": "scaffold_key",
    "cluster": "cluster_id",
}


def group_table(
    curated: pl.DataFrame, ecfp: pl.DataFrame, cfg: dict[str, Any], workers: int
) -> pl.DataFrame:
    curated = curated.sort("qm9_id")
    threshold = float(cfg["splits"]["cluster_ecfp"]["distance_threshold"])
    frame = (
        curated.select(
            "qm9_id",
            "formula",
            *LEAKAGE_KEYS[:3],
            pl.col("inchikey_geometry").str.slice(0, 14).alias("ik_block"),
        )
        .with_columns(leakage_groups(curated))
        .join(scaffold_keys(curated, workers), on="qm9_id", how="left")
        .join(leader_clusters(ecfp, threshold), on="qm9_id", how="left")
    )
    ids = frame["qm9_id"].to_list()
    leak = frame["leakage_group_id"].to_list()
    units = {}
    for name, spec in cfg["splits"].items():
        col = GROUP_COLUMNS[spec["group"]] if spec["kind"] == "grouped" else None
        keys = [leak] if col is None else [frame[col].to_list(), leak]
        units[f"unit_{name}"] = pl.Series(components(ids, keys), dtype=pl.UInt32)
    return frame.with_columns(**units).sort("qm9_id")


def build_splits(
    curated: pl.DataFrame,
    composition: pl.DataFrame,
    groups: pl.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, pl.Series]]:
    frame = (
        curated.select("qm9_id", "n_heavy", "gap_ev", "u0_atomization_ev")
        .join(composition.select("qm9_id", "n_F", "n_N", "n_O"), on="qm9_id")
        .sort("qm9_id")
    )
    rules: dict[str, pl.Series] = {}
    parts = []
    for name, spec in cfg["splits"].items():
        units = groups.select("qm9_id", pl.col(f"unit_{name}").alias("unit_id"))
        if spec["kind"] == "holdout":
            rules[name] = assign.holdout_rule(frame, spec["test_when"])
        for seed in cfg["seeds"]:
            if spec["kind"] == "grouped":
                out = assign.grouped(units, cfg["fractions"], seed)
            else:
                out = assign.holdout(units, rules[name], cfg["fractions"], seed)
            parts.append(
                out.sort("qm9_id").select(
                    "qm9_id",
                    pl.lit(name).alias("split_name"),
                    pl.lit(seed, dtype=pl.UInt8).alias("seed"),
                    pl.col("subset").cast(pl.Enum(assign.SUBSETS)),
                )
            )
    splits = pl.concat(parts).with_columns(pl.col("split_name").cast(pl.Enum(list(cfg["splits"]))))
    return splits, rules


def audit(
    splits: pl.DataFrame,
    groups: pl.DataFrame,
    rules: dict[str, pl.Series],
    curated: pl.DataFrame,
    composition: pl.DataFrame,
    ecfp: pl.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    ecfp = ecfp.sort("qm9_id")
    fp = bit_matrix(ecfp)
    fp_ids = ecfp["qm9_id"].to_numpy()
    out: dict[str, Any] = {}
    for name, spec in cfg["splits"].items():
        cols = groups.select(
            "qm9_id", *LEAKAGE_KEYS, "leakage_group_id", pl.col(f"unit_{name}").alias("unit_id")
        )
        per_seed = {}
        for seed in cfg["seeds"]:
            assigned = (
                splits.filter((pl.col("split_name") == name) & (pl.col("seed") == seed))
                .select("qm9_id", pl.col("subset").cast(pl.String))
                .join(cols, on="qm9_id")
                .sort("qm9_id")
            )
            per_seed[str(seed)] = check_assignment(
                assigned,
                groups.height,
                spec,
                cfg["fractions"],
                float(cfg["fraction_tolerance"]),
                rules.get(name),
            )
            if seed == cfg["seeds"][0]:
                shift = shift_report(
                    assigned,
                    curated,
                    composition,
                    fp_ids,
                    fp,
                    cfg["audit"]["shift_targets"],
                    int(cfg["audit"]["nn_sample"]),
                )
        units = groups.group_by(f"unit_{name}").len()
        out[name] = {
            "spec": spec,
            "n_units": units.height,
            "largest_unit": int(units["len"].to_numpy().max()),
            "seeds": per_seed,
            "shift_seed0": shift,
        }
    return out


def group_stats(groups: pl.DataFrame) -> dict[str, Any]:
    def sizes(col: str) -> dict[str, int]:
        g = groups.filter(pl.col(col).is_not_null()).group_by(col).len()
        return {
            "n": g.height,
            "largest": int(g["len"].to_numpy().max()),
            "records_in_groups_of_2_or_more": int(g.filter(pl.col("len") > 1)["len"].sum()),
        }

    acyclic = groups.filter(pl.col("scaffold_key").str.starts_with("acyclic:"))
    return {
        "leakage_group_id": sizes("leakage_group_id"),
        "formula": sizes("formula"),
        "scaffold_key": {
            **sizes("scaffold_key"),
            "acyclic_records": acyclic.height,
            "acyclic_buckets": acyclic["scaffold_key"].n_unique(),
        },
        "cluster_id": {
            **sizes("cluster_id"),
            "records_without_fingerprint": int(groups["cluster_id"].null_count()),
        },
    }


def gate_conditions(
    report: dict[str, Any], groups: pl.DataFrame, n_records: int
) -> dict[str, bool]:
    per = [s for split in report["splits"].values() for s in split["seeds"].values()]
    shifts = [split["shift_seed0"] for split in report["splits"].values()]
    return {
        "every_record_has_groups": groups.height == n_records
        and groups["leakage_group_id"].null_count() == 0,
        "every_split_and_seed_covers_every_record_once": all(
            p["covers_every_record_once"] for p in per
        ),
        "no_leakage_key_in_two_subsets": all(p["no_key_in_two_subsets"] for p in per),
        "holdout_rules_respected": all(p.get("holdout_rule_respected", True) for p in per),
        "grouped_fractions_within_tolerance": all(
            p.get("fractions_within_tolerance", True) for p in per
        ),
        "shift_report_for_every_split": all(
            np.isfinite(s["nn_tanimoto_test_to_train"]["quantiles"]["median"])
            and all(np.isfinite(t["wasserstein"]) for t in s["targets"].values())
            for s in shifts
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "splits" / "split_report.json")
    parser.add_argument("--workers", type=int, default=default_workers())
    args = parser.parse_args(argv)

    cfg = load_config("splits")
    curated = pl.read_parquet(args.processed_dir / "molecules.parquet").sort("qm9_id")
    composition = pl.read_parquet(args.processed_dir / "features" / "composition.parquet")
    ecfp = pl.read_parquet(args.processed_dir / "features" / "ecfp4.parquet")

    groups = group_table(curated, ecfp, cfg, args.workers)
    splits, rules = build_splits(curated, composition, groups, cfg)
    report: dict[str, Any] = {
        "config": cfg,
        "groups": group_stats(groups),
        "naive_record_level_random": naive_random_leakage(
            groups, cfg["fractions"]["test"], cfg["seeds"][0]
        ),
        "splits": audit(splits, groups, rules, curated, composition, ecfp, cfg),
    }
    report["gate_8_conditions"] = gate_conditions(report, groups, curated.height)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report["output_sha256"] = {}
    for name, frame in (("splits", splits), ("split_groups", groups)):
        out = args.out_dir / f"{name}.parquet"
        frame.write_parquet(out, compression="zstd", compression_level=9, statistics=True)
        report["output_sha256"][name] = hashlib.sha256(out.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    for condition, ok in report["gate_8_conditions"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {condition}")


if __name__ == "__main__":
    main()
