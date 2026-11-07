"""Build the curated molecule table with quality tiers (Phase 5).

Run as ``python -m qm9_curation.curate.build``. Outputs:

* ``data/processed/molecules.parquet`` — one row per source record (none
  removed): standardized targets, untouched ``source_*`` values, identifiers,
  group ids, ``quality_tier`` and ``tier_reasons``;
* ``reports/curation/curation_report.json`` — tier counts, the rule
  waterfall, per-rule impact, and the Gate 5 conditions.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandera.errors
import polars as pl

from qm9_curation.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, load_config
from qm9_curation.curate.standardize import identifiers_from_geometry, standardize
from qm9_curation.curate.tiers import (
    TIER_ORDER,
    assign_tiers,
    choose_representatives,
    rule_impact,
    tier_counts,
    waterfall,
)
from qm9_curation.inventory import ATOMREF_NAME, parse_atomref
from qm9_curation.parallel import default_workers
from qm9_curation.schema import PROPERTY_COLUMNS, CuratedMolecules
from qm9_curation.units import HARTREE_TO_EV

STANDARD_TARGETS = (
    "mu_debye", "alpha_ang3", "homo_ev", "lumo_ev", "gap_ev", "r2_ang2", "zpve_ev",
    "u0_ev", "u298_ev", "h298_ev", "g298_ev", "cv_cal_per_mol_k",
    "u0_atomization_ev", "u298_atomization_ev", "h298_atomization_ev", "g298_atomization_ev",
)  # fmt: skip


def curated_frame(
    molecules: pl.DataFrame,
    atoms: pl.DataFrame,
    validation: pl.DataFrame,
    profile: pl.DataFrame,
    atomref: dict[str, dict[str, float]],
    rules: list[dict[str, Any]],
    geometry_ids: pl.DataFrame,
) -> pl.DataFrame:
    std = standardize(molecules, atoms, validation, atomref)
    reps = choose_representatives(profile.select("qm9_id", "structure_group_id"), validation)
    tiered = assign_tiers(validation.join(reps, on="qm9_id"), rules).select(
        "qm9_id",
        "quality_tier",
        pl.col("tier_reasons").list.join(",").alias("tier_reasons"),
        "structure_group_id",
        "structure_representative_id",
        "is_structure_representative",
        "flag_bitmask",
        "n_failed_hard",
        "n_failed_identity",
        "n_failed_soft",
        (~pl.col("not_uncharacterized__pass")).alias("uncharacterized_listed"),
    )
    groups = profile.select(
        "qm9_id", "tautomer_group_id", "connectivity_group_id", "canonical_smiles_gdb"
    )
    return (
        std.join(tiered, on="qm9_id")
        .join(groups, on="qm9_id")
        .join(geometry_ids, on="qm9_id")
        .select(
            "qm9_id",
            "quality_tier",
            "tier_reasons",
            "formula",
            "n_atoms",
            "n_heavy",
            "smiles_gdb",
            "canonical_smiles_gdb",
            "smiles_geometry",
            "inchikey_geometry",
            "geometry_identifier_error",
            "identity_class",
            "structure_group_id",
            "structure_representative_id",
            "is_structure_representative",
            "tautomer_group_id",
            "connectivity_group_id",
            pl.col("linear_by_geometry").alias("is_linear"),
            "gdb_charge_separated",
            "uncharacterized_listed",
            *STANDARD_TARGETS,
            pl.col("rot_a_ghz_std").alias("rot_a_ghz"),
            "rot_a_missing_reason",
            "rot_b_ghz",
            "rot_c_ghz",
            *(pl.col(c).alias(f"source_{c}") for c in PROPERTY_COLUMNS),
            "zpve_consistent_set",
            "flag_bitmask",
            "n_failed_hard",
            "n_failed_identity",
            "n_failed_soft",
            "source_member",
            "source_sha256",
            "source_archive_sha256",
        )
        .sort("qm9_id")
    )


def gate_conditions(curated: pl.DataFrame, molecules: pl.DataFrame) -> dict[str, bool]:
    gold = curated.filter(pl.col("quality_tier") == "gold")
    reps = curated.filter(pl.col("is_structure_representative"))
    preserved = curated.select("qm9_id", *(f"source_{c}" for c in PROPERTY_COLUMNS)).join(
        molecules.select("qm9_id", *PROPERTY_COLUMNS), on="qm9_id"
    )
    inverse_err = curated.select(
        (
            (pl.col("u0_ev") / HARTREE_TO_EV - pl.col("source_u0_ha")).abs()
            / pl.col("source_u0_ha").abs()
        ).max()
    ).item()
    try:
        CuratedMolecules.validate(curated, lazy=True)
        schema_ok = True
    except pandera.errors.SchemaErrors:
        schema_ok = False
    return {
        "one_row_per_source_record": curated.height == molecules.height
        and curated["qm9_id"].is_unique().all(),
        "every_record_has_a_tier": curated["quality_tier"].is_in(list(TIER_ORDER)).all(),
        "every_non_gold_record_has_a_reason": (
            curated.filter(pl.col("quality_tier") != "gold")["tier_reasons"].str.len_chars() > 0
        ).all(),
        "gold_records_have_no_reason": (gold["tier_reasons"] == "").all(),
        "one_representative_per_structure_group": reps.height
        == curated["structure_group_id"].n_unique()
        and reps["structure_group_id"].is_unique().all(),
        "gold_is_duplicate_free": gold["structure_group_id"].is_unique().all(),
        "source_values_preserved_exactly": all(
            (preserved[f"source_{c}"] == preserved[c]).all() for c in PROPERTY_COLUMNS
        ),
        "unit_conversion_invertible": inverse_err < 1e-14,
        "schema_valid": schema_ok,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--report", type=Path, default=REPORTS_DIR / "curation" / "curation_report.json"
    )
    parser.add_argument("--workers", type=int, default=default_workers())
    args = parser.parse_args(argv)

    t = {
        n: pl.read_parquet(args.interim_dir / f"{n}.parquet")
        for n in ("molecules", "atoms", "validation", "molecule_profile")
    }
    rules = load_config("curation_rules")["rules"]
    atomref = parse_atomref(args.raw_dir / ATOMREF_NAME)
    geometry_ids = identifiers_from_geometry(t["atoms"], args.workers)
    curated = curated_frame(
        t["molecules"],
        t["atoms"],
        t["validation"],
        t["molecule_profile"],
        atomref,
        rules,
        geometry_ids,
    )

    reps = choose_representatives(
        t["molecule_profile"].select("qm9_id", "structure_group_id"), t["validation"]
    )
    rule_input = t["validation"].join(reps, on="qm9_id")
    report: dict[str, Any] = {
        "tier_counts": tier_counts(curated),
        "waterfall": waterfall(rule_input, rules),
        "rule_impact_if_switched_off": rule_impact(rule_input, rules),
        "rules": rules,
        "duplicates": {
            "structure_groups_with_2_or_more": int(
                curated.filter(pl.col("structure_group_id") != pl.col("qm9_id"))
                .select("structure_group_id")
                .n_unique()
            ),
            "non_representative_records": int((~curated["is_structure_representative"]).sum()),
        },
        "geometry_identifiers": {
            "assigned": int(curated["smiles_geometry"].is_not_null().sum()),
            "failed": dict(
                curated.filter(pl.col("geometry_identifier_error").is_not_null())
                .group_by("geometry_identifier_error")
                .len()
                .rows()
            ),
        },
        "atom_references_hartree": atomref,
        "gate_5_conditions": gate_conditions(curated, t["molecules"]),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "molecules.parquet"
    curated.write_parquet(out, compression="zstd", compression_level=9, statistics=True)
    report["output_sha256"] = {"molecules": hashlib.sha256(out.read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for condition, ok in report["gate_5_conditions"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {condition}")
    print("tiers:", report["tier_counts"])


if __name__ == "__main__":
    main()
