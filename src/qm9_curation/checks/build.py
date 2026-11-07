"""Run all four validation layers and write the flag table (Phase 4).

Run as ``python -m qm9_curation.checks.build``. Outputs:

* ``data/interim/validation.parquet`` — one row per molecule: ``<check>__pass``
  (and ``<check>__residual`` where a residual exists) for every check in
  ``registry.CHECKS``, failure counts per severity, ``flag_bitmask``, and the
  identity / ZPVE / linearity details;
* ``reports/validation/validation_report.json`` — per-check failure counts,
  residual quantiles, the identity-vs-uncharacterized confusion matrix (RQ1),
  tolerances used, and the Gate 4 conditions.
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl

from qm9_curation.checks.format import run_format
from qm9_curation.checks.identity import identity_record
from qm9_curation.checks.physics import run_physics
from qm9_curation.checks.registry import CHECK_NAMES, CHECKS, SEVERITIES
from qm9_curation.checks.statistical import run_statistical
from qm9_curation.config import INTERIM_DIR, RAW_DIR, REPORTS_DIR, load_config
from qm9_curation.inventory import UNCHARACTERIZED_NAME, parse_uncharacterized

QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)
CHUNK = 2000


def _silence_stdout() -> None:
    """Send this process's C-level stdout to /dev/null.

    RDKit's extended-Hückel backend prints a warning for every short X-H bond
    (O-H ~0.96 Å is "suspicious" to it), tens of thousands of lines per run.
    Used as the worker initializer, so the main process keeps its output.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)


def _identity_chunk(rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    return [identity_record(*row) for row in rows]


def run_identity(
    mol: pl.DataFrame, atoms: pl.DataFrame, unchar: set[int], workers: int
) -> pl.DataFrame:
    per_mol = (
        atoms.sort(["qm9_id", "atom_idx"])
        .group_by("qm9_id", maintain_order=True)
        .agg(pl.col("element"), pl.concat_list("x_ang", "y_ang", "z_ang").alias("xyz"))
    )
    j = (
        mol.select("qm9_id", "smiles_gdb", "smiles_relaxed")
        .join(per_mol, on="qm9_id")
        .sort("qm9_id")
    )
    rows = [(q, g, r, e, [tuple(c) for c in xyz]) for q, g, r, e, xyz in j.iter_rows()]
    chunks = [rows[i : i + CHUNK] for i in range(0, len(rows), CHUNK)]
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_silence_stdout) as pool:
        results = [r for chunk in pool.map(_identity_chunk, chunks) for r in chunk]
    ident = pl.DataFrame(results, infer_schema_length=None)
    return ident.select(
        pl.col("qm9_id").cast(pl.UInt32),
        "identity_class",
        (pl.col("geom_n_fragments") == 1).alias("single_fragment__pass"),
        pl.col("geom_n_fragments").cast(pl.Float64).alias("single_fragment__residual"),
        pl.col("gdb_matches_geometry").alias("gdb_matches_geometry__pass"),
        (~pl.col("qm9_id").is_in(sorted(unchar))).alias("not_uncharacterized__pass"),
        pl.col("relaxed_matches_geometry")
        .fill_null(False)
        .alias("relaxed_smiles_matches_geometry__pass"),
        "geom_bond_orders_assigned",
        "geom_inchi_matches_gdb",
    )


def assemble(parts: list[pl.DataFrame]) -> pl.DataFrame:
    out = parts[0]
    for part in parts[1:]:
        out = out.join(part, on="qm9_id", how="left")
    fails = {s: [] for s in SEVERITIES}  # type: dict[str, list[pl.Expr]]
    bits = []
    for i, check in enumerate(CHECKS):
        failed = ~pl.col(f"{check.name}__pass").fill_null(False)
        fails[check.severity].append(failed.cast(pl.UInt8))
        bits.append(
            pl.when(failed)
            .then(pl.lit(1 << i, dtype=pl.UInt32))
            .otherwise(pl.lit(0, dtype=pl.UInt32))
        )
    return out.with_columns(
        *(
            pl.sum_horizontal(exprs).cast(pl.UInt8).alias(f"n_failed_{sev}")
            for sev, exprs in fails.items()
        ),
        pl.sum_horizontal(bits).cast(pl.UInt32).alias("flag_bitmask"),
    ).sort("qm9_id")


def summarize(flags: pl.DataFrame, tolerances: dict[str, Any]) -> dict[str, Any]:
    per_check = {}
    for check in CHECKS:
        col = flags[f"{check.name}__pass"]
        entry: dict[str, Any] = {
            "layer": check.layer,
            "severity": check.severity,
            "description": check.description,
            "n_failed": int((~col.fill_null(False)).sum()),
            "n_null": col.null_count(),
        }
        res_col = f"{check.name}__residual"
        if res_col in flags.columns:
            # Some checks are pass/fail only (residual column all null): no quantiles then.
            res = flags[res_col].drop_nulls().drop_nans()
            if res.len():
                entry["residual_quantiles"] = {
                    str(q): float(res.quantile(q, "nearest"))  # type: ignore[arg-type]
                    for q in QUANTILES
                }
        entry["failed_examples"] = (
            flags.filter(~pl.col(f"{check.name}__pass").fill_null(False))["qm9_id"]
            .head(10)
            .to_list()
        )
        per_check[check.name] = entry
    confusion = (
        flags.with_columns((~pl.col("not_uncharacterized__pass")).alias("uncharacterized"))
        .group_by("identity_class", "uncharacterized")
        .len()
        .sort("identity_class", "uncharacterized")
    )
    doubled = flags.filter(pl.col("zpve_other_set_residual").is_not_null())
    tol = tolerances["physics"]
    return {
        "checks": per_check,
        "identity_vs_uncharacterized": [
            {"identity_class": c, "uncharacterized": u, "n": n} for c, u, n in confusion.rows()
        ],
        "doubled_frequency_resolution": {
            "records": doubled.height,
            "consistent_set_counts": {
                str(k): v
                for k, v in doubled["zpve_consistent_set"]
                .value_counts()
                .sort("zpve_consistent_set")
                .rows()
            },
            "both_sets_consistent": int(
                (doubled["zpve_other_set_residual"] <= tol["zpve_rel_tol"]).sum()
            ),
        },
        "linear_by_geometry": flags.filter(pl.col("linear_by_geometry"))["qm9_id"].to_list(),
        "records_with_any_failure": {
            sev: int((flags[f"n_failed_{sev}"] > 0).sum()) for sev in SEVERITIES
        },
        "records_clean": int((flags["flag_bitmask"] == 0).sum()),
        "tolerances": tolerances,
    }


def build(interim: Path, raw: Path, workers: int) -> tuple[pl.DataFrame, dict[str, Any]]:
    mol = pl.read_parquet(interim / "molecules.parquet")
    atoms = pl.read_parquet(interim / "atoms.parquet")
    freq = pl.read_parquet(interim / "frequencies.parquet")
    profile = pl.read_parquet(interim / "molecule_profile.parquet")
    dataset = load_config("dataset")
    tolerances = load_config("validation")
    unchar = set(parse_uncharacterized(raw / UNCHARACTERIZED_NAME))

    # Cheap vectorized layers first so a configuration error fails in seconds,
    # before the slow per-molecule identity perception.
    physics = run_physics(mol, atoms, freq, tolerances["physics"])
    statistical = run_statistical(
        profile,
        tolerances["statistical"]["robust_z_review"],
        dataset["known_issues"]["hard_to_converge"],
    )
    fmt = run_format(mol, atoms, freq, dataset["expected"]["max_heavy_atoms"])
    identity = run_identity(mol, atoms, unchar, workers)
    flags = assemble([fmt, identity, physics, statistical])
    report = summarize(flags, tolerances)
    pass_cols = [f"{c}__pass" for c in CHECK_NAMES]
    report["gate_4_conditions"] = {
        "one_flag_row_per_molecule": flags.height == mol.height
        and flags["qm9_id"].is_unique().all(),
        "every_registered_check_present": all(c in flags.columns for c in pass_cols),
        "no_missing_check_results": all(flags[c].null_count() == 0 for c in pass_cols),
        "all_tolerances_set": all(v is not None for v in tolerances["physics"].values())
        and all(v is not None for v in tolerances["statistical"].values()),
    }
    return flags, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--report", type=Path, default=REPORTS_DIR / "validation" / "validation_report.json"
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    flags, report = build(args.interim_dir, args.raw_dir, args.workers)
    out = args.interim_dir / "validation.parquet"
    flags.write_parquet(out, compression="zstd", compression_level=9, statistics=True)
    report["output_sha256"] = {"validation": hashlib.sha256(out.read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for condition, ok in report["gate_4_conditions"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {condition}")
    for name, entry in report["checks"].items():
        print(
            f"  L{entry['layer']} {entry['severity']:<8} {name:<34} failed {entry['n_failed']:>6,}"
        )


if __name__ == "__main__":
    main()
