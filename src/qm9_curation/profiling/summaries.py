"""Dataset-level summaries of composition and cheminformatics (Phase 3)."""

from typing import Any

import polars as pl


def _counts(series: pl.Series) -> dict[str, int]:
    vc = series.value_counts().sort(series.name)
    return {str(k): int(v) for k, v in vc.rows()}


def composition_summary(molecules: pl.DataFrame, atoms: pl.DataFrame) -> dict[str, Any]:
    formula_sizes = molecules.group_by("formula").len()
    top = formula_sizes.sort(["len", "formula"], descending=[True, False]).head(10)
    elements_present = atoms.group_by("qm9_id").agg(pl.col("element").unique())
    per_element = {
        e: int(elements_present.filter(pl.col("element").list.contains(e)).height)
        for e in ("C", "N", "O", "F")
    }
    return {
        "n_molecules": molecules.height,
        "element_atom_counts": _counts(atoms["element"]),
        "molecules_containing_element": per_element,
        "n_atoms_histogram": _counts(molecules["n_atoms"]),
        "n_heavy_histogram": _counts(molecules["n_heavy"]),
        "n_unique_formulas": formula_sizes.height,
        "isomers_per_formula": {
            "singleton_formulas": int((formula_sizes["len"] == 1).sum()),
            "median": float(formula_sizes["len"].median()),  # type: ignore[arg-type]
            "max": int(formula_sizes["len"].max()),  # type: ignore[arg-type]
        },
        "largest_formula_groups": [{"formula": f, "n": int(n)} for f, n in top.rows()],
    }


def chemistry_summary(
    chem: pl.DataFrame, molecules: pl.DataFrame, unchar: set[int]
) -> dict[str, Any]:
    j = chem.join(
        molecules.select("qm9_id", "smiles_gdb", "smiles_relaxed"), on="qm9_id"
    ).with_columns(pl.col("qm9_id").is_in(sorted(unchar)).alias("uncharacterized"))
    agree = j["flat_smiles_gdb"] == j["flat_smiles_relaxed"]
    crosstab = (
        j.with_columns(
            pl.when(pl.col("flat_smiles_relaxed").is_null())
            .then(pl.lit("relaxed_unparseable"))
            .when(agree)
            .then(pl.lit("agree"))
            .otherwise(pl.lit("disagree"))
            .alias("smiles_comparison")
        )
        .group_by("smiles_comparison", "uncharacterized")
        .len()
        .sort("smiles_comparison", "uncharacterized")
    )
    scaffolds = j.filter(pl.col("murcko_scaffold").is_not_null())
    top_generic = (
        scaffolds.group_by("generic_scaffold")
        .len()
        .sort(["len", "generic_scaffold"], descending=[True, False])
        .head(10)
    )
    ring_sizes = j["ring_sizes"].drop_nulls().str.split(",").explode()
    ring_sizes = ring_sizes.filter(ring_sizes != "")
    return {
        "rdkit_parse_failures": {
            "gdb": int((~j["gdb_rdkit_ok"]).sum()),
            "relaxed": int((~j["relaxed_rdkit_ok"]).sum()),
            "relaxed_and_uncharacterized": int(
                (~j["relaxed_rdkit_ok"] & j["uncharacterized"]).sum()
            ),
        },
        "gdb_vs_relaxed_flat_smiles": [
            {"comparison": c, "uncharacterized": u, "n": int(n)} for c, u, n in crosstab.rows()
        ],
        "relaxed_smiles_with_fragments": int(j["smiles_relaxed"].str.contains(r"\.").sum()),
        "relaxed_smiles_with_radicals": int((j["n_radical_electrons"] > 0).sum()),
        "relaxed_smiles_with_charges": int((j["n_charged_atoms"] > 0).sum()),
        "gdb_smiles_with_charges": int(j["smiles_gdb"].str.contains(r"\+|-\]").sum()),
        "gdb_smiles_with_stereo": int(j["smiles_gdb"].str.contains(r"[@/\\]").sum()),
        "relaxed_smiles_with_stereo": int(j["smiles_relaxed"].str.contains(r"[@/\\]").sum()),
        "molecules_with_stereocentres": int((j["n_stereocentres"] > 0).sum()),
        "rings": {
            "acyclic": int((j["n_rings"] == 0).sum()),
            "acyclic_fraction": round(float((j["n_rings"] == 0).mean()), 4),  # type: ignore[arg-type]
            "n_rings_histogram": _counts(j["n_rings"].drop_nulls()),
            "ring_size_counts": _counts(ring_sizes),
            "molecules_with_aromatic_atoms": int((j["n_aromatic_atoms"] > 0).sum()),
        },
        "scaffolds": {
            "unique_murcko": scaffolds["murcko_scaffold"].n_unique(),
            "unique_generic": scaffolds["generic_scaffold"].n_unique(),
            "empty_scaffold_molecules": int((scaffolds["murcko_scaffold"] == "").sum()),
            "top_generic": [{"scaffold": s, "n": int(n)} for s, n in top_generic.rows()],
        },
    }
