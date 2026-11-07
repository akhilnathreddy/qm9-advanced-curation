"""Multi-fidelity cross-check: QM9 B3LYP vs. G4(MP2) of Kim et al. (RQ4).

Two differences per molecule, both at 0 K and in eV:

* ``delta_u0_atomization_ev`` = AE(B3LYP, QM9 atom references) - AE(G4(MP2),
  authors' atom references). Defined only when every atom reference of the
  molecule is in its electronic ground state. The authors' nitrogen atom is a
  doublet (``Multiplicity = 2`` in ``atom_ref/N.log``); the ground state is
  the quartet, so this difference is withheld for every N-containing molecule.
* ``delta_residual_ev``: U0(B3LYP) - U0(G4(MP2)) minus a per-element linear
  term fitted on Gold. Because AE differences differ from total-energy
  differences only by a per-element constant, this residual is the same
  whichever atom references are used, so it is defined for every converged
  molecule. It is the part of the B3LYP error that a per-atom correction
  cannot remove: the label-noise floor for RQ4.

Records whose geometry the authors' re-optimization changed by more than
1 kcal/mol (their criterion) are not same-geometry comparisons and are left
out of the statistics.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import polars as pl

from qm9_curation.external.kim2019 import ELEMENTS, Kim2019, formula_counts
from qm9_curation.review.scientific import constitution
from qm9_curation.review.stats import ELEMENT_COUNTS, fit_per_element, robust_z_by_size, summary
from qm9_curation.units import EV_TO_KCAL_PER_MOL, HARTREE_TO_EV, HARTREE_TO_KCAL_PER_MOL

STATUSES = ("ok", "geometry_changed", "not_converged", "missing")


def _atom_sum(refs: Mapping[str, float]) -> pl.Expr:
    total = pl.lit(0.0)
    for e in ELEMENTS:
        total = total + pl.col(f"n_{e}").cast(pl.Float64) * refs[e]
    return total


def xref_table(
    curated: pl.DataFrame,
    composition: pl.DataFrame,
    kim: Kim2019,
    ground_state: Mapping[str, int],
    geometry_change_hartree: float,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """One row per QM9 record with the G4(MP2) values, status and differences."""
    atoms = {r["element"]: r for r in kim.atoms.iter_rows(named=True)}
    invalid = sorted(e for e in ELEMENTS if atoms[e]["multiplicity"] != ground_state[e])
    refs = {e: atoms[e]["g4mp2_u0_ha"] for e in ELEMENTS}
    has_invalid = pl.lit(False)
    for e in invalid:
        has_invalid = has_invalid | (pl.col(f"n_{e}") > 0)

    frame = (
        curated.select(
            "qm9_id",
            "quality_tier",
            "n_heavy",
            "source_u0_ha",
            "source_zpve_ha",
            "u0_atomization_ev",
        )
        .join(composition.select("qm9_id", *ELEMENT_COUNTS), on="qm9_id")
        .join(kim.molecules, on="qm9_id", how="left")
        .with_columns(
            (pl.col("g4mp2_b3lyp_ha") - (pl.col("source_u0_ha") - pl.col("source_zpve_ha"))).alias(
                "b3lyp_reopt_energy_change_ha"
            ),
            pl.col("qm9_id").is_in(kim.not_converged).alias("kim_not_converged"),
            pl.col("qm9_id").is_in(kim.multi_molecule).alias("kim_multi_molecule"),
        )
        .with_columns(
            pl.when(pl.col("g4mp2_u0_ha").is_not_null() & ~pl.col("kim_not_converged"))
            .then(
                pl.when(pl.col("b3lyp_reopt_energy_change_ha").abs() > geometry_change_hartree)
                .then(pl.lit("geometry_changed"))
                .otherwise(pl.lit("ok"))
            )
            .when(pl.col("kim_not_converged"))
            .then(pl.lit("not_converged"))
            .otherwise(pl.lit("missing"))
            .alias("g4mp2_status"),
            pl.when(has_invalid)
            .then(None)
            .otherwise((pl.col("g4mp2_u0_ha") - _atom_sum(refs)) * HARTREE_TO_EV)
            .alias("g4mp2_u0_atomization_ev"),
            pl.when(pl.col("g4mp2_u0_ha").is_null())
            .then(pl.lit("no converged G4(MP2) energy"))
            .when(has_invalid)
            .then(pl.lit(f"published atom reference not in ground state: {','.join(invalid)}"))
            .otherwise(None)
            .alias("g4mp2_atomization_missing_reason"),
            ((pl.col("source_u0_ha") - pl.col("g4mp2_u0_ha")) * HARTREE_TO_EV).alias(
                "delta_u0_total_ev"
            ),
        )
        .with_columns(
            (pl.col("u0_atomization_ev") - pl.col("g4mp2_u0_atomization_ev")).alias(
                "delta_u0_atomization_ev"
            )
        )
    )
    usable = (pl.col("g4mp2_status") == "ok") & (pl.col("quality_tier") == "gold")
    coefs, pred = fit_per_element(frame.filter(usable), "delta_u0_total_ev")
    frame = frame.with_columns(
        pl.when(pl.col("g4mp2_status") == "ok")
        .then(pl.col("delta_u0_total_ev") - pred)
        .otherwise(None)
        .alias("delta_residual_ev")
    )
    frame = frame.with_columns(robust_z_by_size(frame, "delta_residual_ev").alias("delta_robust_z"))
    formulas_ok = all(
        formula_counts(f) == {e: n for e, n in zip(ELEMENTS, counts, strict=True) if n}
        for f, *counts in frame.filter(pl.col("kim_formula").is_not_null())
        .select("kim_formula", *ELEMENT_COUNTS)
        .iter_rows()
    )
    info = {
        "invalid_reference_elements": invalid,
        "total_energy_fit": coefs,
        "formulas_agree": formulas_ok,
    }
    columns = [
        "qm9_id",
        "g4mp2_status",
        "g4mp2_u0_ha",
        "g4mp2_b3lyp_ha",
        "g4mp2_zpve_ha",
        "b3lyp_reopt_energy_change_ha",
        "g4mp2_u0_atomization_ev",
        "g4mp2_atomization_missing_reason",
        "delta_u0_total_ev",
        "delta_u0_atomization_ev",
        "delta_residual_ev",
        "delta_robust_z",
        "kim_not_converged",
        "kim_multi_molecule",
    ]
    return frame.select(columns).sort("qm9_id"), info


def nitrogen_offset(
    curated: pl.DataFrame, composition: pl.DataFrame, xref: pl.DataFrame, kim: Kim2019
) -> dict[str, float]:
    """Per-element fit of the AE difference computed with *all* published references.

    With the doublet N reference, the fitted N coefficient carries the N
    doublet-quartet excitation on top of the genuine B3LYP-vs-G4(MP2) per-N
    term, so it is compared with the measured atomic splitting.
    """
    refs = {r["element"]: r["g4mp2_u0_ha"] for r in kim.atoms.iter_rows(named=True)}
    frame = (
        xref.filter(pl.col("g4mp2_status") == "ok")
        .join(curated.select("qm9_id", "quality_tier", "u0_atomization_ev"), on="qm9_id")
        .filter(pl.col("quality_tier") == "gold")
        .join(composition.select("qm9_id", *ELEMENT_COUNTS), on="qm9_id")
        .with_columns(
            (
                pl.col("u0_atomization_ev")
                - (pl.col("g4mp2_u0_ha") - _atom_sum(refs)) * HARTREE_TO_EV
            ).alias("d")
        )
    )
    coefs, _ = fit_per_element(frame, "d")
    return coefs


def delta_statistics(xref: pl.DataFrame, curated: pl.DataFrame, top_n: int) -> dict[str, Any]:
    frame = xref.join(
        curated.select("qm9_id", "quality_tier", "smiles_geometry", "formula"), on="qm9_id"
    )
    ok_gold = frame.filter((pl.col("g4mp2_status") == "ok") & (pl.col("quality_tier") == "gold"))
    by_tier = {
        tier: summary(
            frame.filter((pl.col("g4mp2_status") == "ok") & (pl.col("quality_tier") == tier))[
                "delta_residual_ev"
            ].to_numpy()
        )
        for tier in ("gold", "silver", "bronze")
    }
    top = (
        frame.filter(pl.col("delta_robust_z").is_not_null())
        .sort(pl.col("delta_robust_z").abs(), "qm9_id", descending=[True, False])
        .head(top_n)
        .select(
            "qm9_id",
            "quality_tier",
            "formula",
            "smiles_geometry",
            "delta_u0_total_ev",
            "delta_residual_ev",
            "delta_robust_z",
        )
    )
    return {
        "atomization_difference_gold_valid_references": summary(
            ok_gold["delta_u0_atomization_ev"].drop_nulls().to_numpy()
        ),
        "residual_difference_gold": summary(ok_gold["delta_residual_ev"].to_numpy()),
        "residual_difference_by_tier": by_tier,
        "largest_residuals": top.to_dicts(),
    }


def cross_checks(
    xref: pl.DataFrame,
    curated: pl.DataFrame,
    kim: Kim2019,
    authors_hard_to_converge: list[int],
) -> dict[str, Any]:
    """Compare the authors' findings with this pipeline's own (RQ1 corroboration)."""
    fragmented = set(curated.filter(pl.col("identity_class") == "fragmented")["qm9_id"].to_list())
    multi = set(kim.multi_molecule)
    group = dict(
        zip(curated["qm9_id"].to_list(), curated["structure_group_id"].to_list(), strict=True)
    )
    conn = dict(
        zip(curated["qm9_id"].to_list(), curated["connectivity_group_id"].to_list(), strict=True)
    )
    sizes = curated.group_by("structure_group_id").len()
    ours_dup = set(
        curated.join(sizes.filter(pl.col("len") > 1), on="structure_group_id")["qm9_id"].to_list()
    )
    pair_ids = {i for p in kim.duplicate_pairs for i in p}
    rows = {r["qm9_id"]: r for r in curated.select(
        "qm9_id", "quality_tier", "identity_class", "smiles_geometry", "u0_ev"
    ).iter_rows(named=True)}  # fmt: skip
    not_ours = [
        {
            "pair": [a, b],
            "tiers": [rows[a]["quality_tier"], rows[b]["quality_tier"]],
            "identity_classes": [rows[a]["identity_class"], rows[b]["identity_class"]],
            "same_geometry_smiles": rows[a]["smiles_geometry"] == rows[b]["smiles_geometry"],
            "same_constitution": constitution(rows[a]["smiles_geometry"])
            == constitution(rows[b]["smiles_geometry"]),
            "abs_du0_kcal_mol": abs(rows[a]["u0_ev"] - rows[b]["u0_ev"]) * EV_TO_KCAL_PER_MOL,
        }
        for a, b in kim.duplicate_pairs
        if group[a] != group[b]
    ]
    not_conv = curated.filter(pl.col("qm9_id").is_in(kim.not_converged))
    changed = xref.filter(pl.col("g4mp2_status") == "geometry_changed").join(
        curated.select("qm9_id", "quality_tier", "identity_class"), on="qm9_id"
    )
    tiers = {
        "not_converged": dict(sorted(not_conv.group_by("quality_tier").len().rows())),
        "geometry_changed": dict(sorted(changed.group_by("quality_tier").len().rows())),
    }
    return {
        "multi_molecule_vs_fragmented": {
            "authors": len(multi),
            "ours": len(fragmented),
            "identical_sets": multi == fragmented,
            "authors_only": sorted(multi - fragmented),
            "ours_only": sorted(fragmented - multi),
        },
        "duplicate_pairs": {
            "authors_pairs": len(kim.duplicate_pairs),
            "same_structure_group": sum(group[a] == group[b] for a, b in kim.duplicate_pairs),
            "same_connectivity_group": sum(conn[a] == conn[b] for a, b in kim.duplicate_pairs),
            "authors_pairs_not_ours": not_ours,
            "authors_pairs_not_ours_same_geometry_smiles": sum(
                p["same_geometry_smiles"] for p in not_ours
            ),
            "authors_pairs_not_ours_same_constitution": sum(
                p["same_constitution"] for p in not_ours
            ),
            "our_duplicate_records": len(ours_dup),
            "our_duplicate_records_in_authors_pairs": len(ours_dup & pair_ids),
        },
        "not_converged": {
            "n": len(kim.not_converged),
            "also_on_qm9_authors_hard_to_converge_list": sorted(
                set(kim.not_converged) & set(authors_hard_to_converge)
            ),
        },
        "geometry_changed": {
            "n": changed.height,
            "records": changed.sort("qm9_id")
            .select(
                "qm9_id",
                "quality_tier",
                "identity_class",
                (pl.col("b3lyp_reopt_energy_change_ha") * HARTREE_TO_KCAL_PER_MOL).alias(
                    "kcal_mol"
                ),
            )
            .to_dicts(),
        },
        "tiers": tiers,
    }


def authors_atomization_reproduced(kim: Kim2019) -> float:
    """Max |our AE (authors' refs) - authors' atomization.csv| in Hartree."""
    refs = {r["element"]: r["g4mp2_u0_ha"] for r in kim.atoms.iter_rows(named=True)}
    counts = kim.molecules.select(
        "qm9_id",
        "g4mp2_u0_ha",
        *(
            pl.col("kim_formula")
            .str.extract(rf"{e}(\d+)")
            .cast(pl.Int64)
            .fill_null(0)
            .alias(f"n_{e}")
            for e in ELEMENTS
        ),
    )
    ours = counts.select("qm9_id", (_atom_sum(refs) - pl.col("g4mp2_u0_ha")).alias("ae"))
    joined = ours.join(kim.authors_atomization, on="qm9_id")
    diff = joined["ae"].to_numpy() - joined["authors_g4mp2_atomization_ha"].to_numpy()
    return float(np.abs(diff).max())
