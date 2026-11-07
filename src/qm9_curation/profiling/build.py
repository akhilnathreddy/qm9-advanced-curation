"""Profile the interim tables and write the Phase 3 data-quality report.

Run as ``python -m qm9_curation.profiling.build``. Outputs:

* ``data/interim/molecule_profile.parquet`` — one row per molecule: RDKit
  profile, geometry statistics and duplicate-group ids (used by later phases);
* ``data/interim/geometric_duplicate_pairs.parquet``;
* ``reports/data_quality/profile.json`` — machine-readable profile, triaged
  issues and Gate 3 conditions;
* ``reports/data_quality/profile.html`` — the human-readable report.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from qm9_curation.config import INTERIM_DIR, RAW_DIR, REPORTS_DIR
from qm9_curation.inventory import UNCHARACTERIZED_NAME, parse_uncharacterized
from qm9_curation.profiling import geometry
from qm9_curation.profiling.chemistry import chemistry_profile
from qm9_curation.profiling.duplicates import duplicate_groups, summarize_groups
from qm9_curation.profiling.issues import TRIAGE_CLASSES, build_issues, clean_checks
from qm9_curation.profiling.properties import pearson_matrix, property_stats, robust_z
from qm9_curation.profiling.report import render
from qm9_curation.profiling.summaries import chemistry_summary, composition_summary
from qm9_curation.schema import PROPERTY_COLUMNS

SECTIONS = ("composition", "chemistry", "duplicates", "properties", "geometry", "issues")
SAME_MOLECULE_MAX_KCAL = 0.1
# The largest A of any non-linear QM9 molecule is ~800 GHz (water); values above this are
# linear molecules whose A came out finite only through numerical noise.
ROT_A_EFFECTIVELY_INFINITE_GHZ = 1e4


def _ids(frame: pl.DataFrame, n: int = 10) -> list[int]:
    return [int(i) for i in frame["qm9_id"].head(n).to_list()]


def build_profile(
    molecules: pl.DataFrame, atoms: pl.DataFrame, frequencies: pl.DataFrame, unchar: set[int]
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    chem = chemistry_profile(molecules)
    geo = geometry.profile_geometry(atoms, molecules.select("qm9_id", "formula"))
    groups = duplicate_groups(chem, geo.duplicate_pairs)
    z = robust_z(molecules)
    max_z = z.select(
        "qm9_id",
        pl.max_horizontal(*(pl.col(f"z_{c}").abs() for c in PROPERTY_COLUMNS)).alias(
            "max_abs_robust_z"
        ),
    )
    neg = frequencies.filter(pl.col("wavenumber_cm1") < 0)["qm9_id"].unique().sort()
    per_molecule = (
        chem.join(geo.per_molecule, on="qm9_id")
        .join(groups, on="qm9_id")
        .join(max_z, on="qm9_id")
        .with_columns(pl.col("qm9_id").is_in(sorted(unchar)).alias("uncharacterized"))
        .sort("qm9_id")
    )

    u0 = dict(zip(molecules["qm9_id"].to_list(), molecules["u0_ha"].to_list(), strict=True))
    pair_du = [
        abs(u0[a] - u0[b]) * 627.5094740631
        for a, b in geo.duplicate_pairs.select("qm9_id_a", "qm9_id_b").rows()
    ]
    gm = geo.per_molecule
    profile: dict[str, Any] = {
        "composition": composition_summary(molecules, atoms),
        "chemistry": chemistry_summary(chem, molecules, unchar),
        "duplicates": summarize_groups(groups, molecules.select("qm9_id", "u0_ha")),
        "properties": property_stats(molecules),
        "property_correlations": pearson_matrix(molecules),
        "geometry": {
            col: {
                stat: gm.select(getattr(pl.col(col), stat)()).item()
                for stat in ("min", "median", "max")
            }
            for col in ("min_interatomic_ang", "max_interatomic_ang", "radius_of_gyration_ang")
        }
        | {
            "geometric_duplicate_pairs": geo.duplicate_pairs.height,
            "geometric_pair_max_u0_diff_kcal_per_mol": max(pair_du, default=0.0),
            "molecules_with_negative_frequency": neg.len(),
        },
        "parse": {
            "doubled_frequencies": int((molecules["n_frequency_sets"] == 2).sum()),
            "linear_molecules": int(molecules["is_linear"].sum()),
        },
        # Linearity shows up three ways in the source; Phase 4 decides it from the geometry.
        "rotational": {
            "a_zero_and_3n_minus_5_frequencies": int(
                ((molecules["rot_a_ghz"] == 0) & molecules["is_linear"]).sum()
            ),
            "a_zero_but_3n_minus_6_frequencies": int(
                ((molecules["rot_a_ghz"] == 0) & ~molecules["is_linear"]).sum()
            ),
            "a_above_1e4_ghz": int((molecules["rot_a_ghz"] > ROT_A_EFFECTIVELY_INFINITE_GHZ).sum()),
        },
        "thresholds": {
            "eigen_radius": geometry.EIGEN_RADIUS,
            "distance_tol_ang": geometry.DISTANCE_TOL_ANG,
            "same_molecule_max_kcal_per_mol": SAME_MOLECULE_MAX_KCAL,
        },
    }
    j = per_molecule.join(
        molecules.select("qm9_id", "smiles_gdb", "smiles_relaxed", "is_linear", "n_frequency_sets"),
        on="qm9_id",
    )
    examples = {
        "structure": _ids(j.filter(pl.col("structure_group_size") > 1)),
        "tautomer": _ids(j.filter(pl.col("tautomer_group_size") > 1)),
        "relaxed_unparseable": _ids(j.filter(~pl.col("relaxed_rdkit_ok"))),
        "radicals": _ids(j.filter(pl.col("n_radical_electrons") > 0)),
        "fragments": _ids(j.filter(pl.col("smiles_relaxed").str.contains(r"\."))),
        "disagree": _ids(
            j.filter(
                (pl.col("flat_smiles_gdb") != pl.col("flat_smiles_relaxed"))
                & ~pl.col("uncharacterized")
            )
        ),
        "gdb_charged": _ids(j.filter(pl.col("smiles_gdb").str.contains(r"\+|-\]"))),
        "doubled": _ids(j.filter(pl.col("n_frequency_sets") == 2)),
        "negative_freq": [int(i) for i in neg.to_list()],
        "linear": _ids(j.filter(pl.col("is_linear"))),
        "linearity": _ids(
            molecules.filter(
                ((pl.col("rot_a_ghz") == 0) & ~pl.col("is_linear"))
                | (pl.col("rot_a_ghz") > ROT_A_EFFECTIVELY_INFINITE_GHZ)
            ).sort("qm9_id")
        ),
        "robust_z": _ids(
            j.filter(pl.col("max_abs_robust_z") > 6).sort("max_abs_robust_z", descending=True)
        ),
    }
    profile["issues"] = build_issues(profile, examples)
    profile["clean_checks"] = clean_checks(profile)
    profile["gate_3_conditions"] = {
        "all_sections_present": all(s in profile for s in SECTIONS),
        "profile_row_per_molecule": per_molecule.height == molecules.height
        and per_molecule["qm9_id"].is_unique().all(),
        "all_issues_triaged": all(i["triage"] in TRIAGE_CLASSES for i in profile["issues"]),
        "geometric_duplicates_energy_consistent": max(pair_du, default=0.0)
        < SAME_MOLECULE_MAX_KCAL,
        "clean_checks_passed": all(c["passed"] for c in profile["clean_checks"]),
    }
    return profile, per_molecule, geo.duplicate_pairs


def _write_parquet(frame: pl.DataFrame, path: Path) -> str:
    frame.write_parquet(path, compression="zstd", compression_level=9, statistics=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR / "data_quality")
    args = parser.parse_args(argv)
    tables = {
        n: pl.read_parquet(args.interim_dir / f"{n}.parquet")
        for n in ("molecules", "atoms", "frequencies")
    }
    unchar = set(parse_uncharacterized(args.raw_dir / UNCHARACTERIZED_NAME))
    profile, per_molecule, pairs = build_profile(
        tables["molecules"], tables["atoms"], tables["frequencies"], unchar
    )

    page = render(profile, tables["molecules"], per_molecule)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "profile.html").write_text(page, encoding="utf-8")
    profile["output_sha256"] = {
        "molecule_profile": _write_parquet(
            per_molecule, args.interim_dir / "molecule_profile.parquet"
        ),
        "geometric_duplicate_pairs": _write_parquet(
            pairs, args.interim_dir / "geometric_duplicate_pairs.parquet"
        ),
        "profile_html": hashlib.sha256(page.encode("utf-8")).hexdigest(),
    }
    (args.report_dir / "profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for condition, ok in profile["gate_3_conditions"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {condition}")
    for issue in profile["issues"]:
        print(f"  {issue['id']}  {issue['count']:>7,}  {issue['triage']:<20} {issue['title']}")


if __name__ == "__main__":
    main()
