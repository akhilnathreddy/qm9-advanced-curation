"""Duplicate and near-duplicate structure groups (Phase 3).

Four levels, from loosest to strictest notion of "the same molecule":

* ``connectivity`` — InChIKey first block (skeleton; ignores H position,
  stereo and charge);
* ``tautomer`` — full standard InChIKey of the GDB-17 molecule. Standard InChI
  uses a mobile-H layer, so **tautomers share a key**. GDB-17 enumerates
  tautomers as distinct molecules with distinct energies; these groups are not
  duplicates, but they matter for leakage-safe splits;
* ``structure`` — the same molecule entered more than once: identical RDKit
  canonical GDB-17 SMILES (e.g. two Kekulé forms of one aromatic ring), joined
  by union-find with pairs whose relaxed geometries are near-identical (e.g.
  charge-separated resonance forms that relax to the same structure);
* ``geometry`` — the geometric pairs on their own (``geometry.py``).

Every group id is the smallest ``qm9_id`` in the group; singletons get their
own id, so the columns are always filled.
"""

import polars as pl


def _group_by_key(frame: pl.DataFrame, key: str, name: str) -> pl.DataFrame:
    return (
        frame.select("qm9_id", key)
        .with_columns(
            pl.col("qm9_id").min().over(key).alias(f"{name}_group_id"),
            pl.len().over(key).cast(pl.UInt32).alias(f"{name}_group_size"),
        )
        .drop(key)
    )


def _union_find(ids: list[int], edges: list[tuple[int, int]]) -> dict[int, int]:
    parent = {i: i for i in ids}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in ids}


def duplicate_groups(chem: pl.DataFrame, geometric_pairs: pl.DataFrame) -> pl.DataFrame:
    base = chem.select(
        "qm9_id",
        pl.col("inchikey_gdb").str.slice(0, 14).alias("connectivity_key"),
        "inchikey_gdb",
        "canonical_smiles_gdb",
    )
    out = base.select("qm9_id")
    out = out.join(_group_by_key(base, "connectivity_key", "connectivity"), on="qm9_id")
    out = out.join(_group_by_key(base, "inchikey_gdb", "tautomer"), on="qm9_id")

    smiles_dup = base.filter(pl.col("canonical_smiles_gdb").is_duplicated())
    smiles_edges = (
        smiles_dup.with_columns(pl.col("qm9_id").min().over("canonical_smiles_gdb").alias("root"))
        .filter(pl.col("qm9_id") != pl.col("root"))
        .select("root", "qm9_id")
        .rows()
    )
    geo_edges = geometric_pairs.select("qm9_id_a", "qm9_id_b").rows()
    ids = base["qm9_id"].to_list()
    root = _union_find(ids, [*smiles_edges, *geo_edges])
    in_geo = set(geometric_pairs["qm9_id_a"].to_list()) | set(geometric_pairs["qm9_id_b"].to_list())
    structure = pl.DataFrame(
        {
            "qm9_id": ids,
            "structure_group_id": [root[i] for i in ids],
            "in_smiles_duplicate": [i in set(smiles_dup["qm9_id"].to_list()) for i in ids],
            "in_geometric_duplicate": [i in in_geo for i in ids],
        },
        schema={
            "qm9_id": pl.UInt32,
            "structure_group_id": pl.UInt32,
            "in_smiles_duplicate": pl.Boolean,
            "in_geometric_duplicate": pl.Boolean,
        },
    ).with_columns(
        pl.len().over("structure_group_id").cast(pl.UInt32).alias("structure_group_size")
    )
    return out.join(structure, on="qm9_id").sort("qm9_id")


def summarize_groups(groups: pl.DataFrame, energies: pl.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for level in ("connectivity", "tautomer", "structure"):
        sizes = groups.filter(pl.col(f"{level}_group_size") > 1).group_by(f"{level}_group_id").len()
        summary[level] = {
            "n_groups_with_2_or_more": sizes.height,
            "n_molecules_in_such_groups": int(sizes["len"].sum()) if sizes.height else 0,
            "largest_group": int(sizes["len"].max()) if sizes.height else 1,  # type: ignore[arg-type]
        }
    # Energy spread inside each multi-member structure group, as evidence they are one molecule.
    spread = (
        groups.filter(pl.col("structure_group_size") > 1)
        .join(energies, on="qm9_id")
        .group_by("structure_group_id")
        .agg(((pl.col("u0_ha").max() - pl.col("u0_ha").min()) * 627.5094740631).alias("du0"))
    )
    summary["structure"]["max_u0_spread_kcal_per_mol"] = (  # type: ignore[index]
        float(spread["du0"].max()) if spread.height else 0.0  # type: ignore[arg-type]
    )
    summary["structure"]["found_by"] = {  # type: ignore[index]
        "smiles_only": int(
            (groups["in_smiles_duplicate"] & ~groups["in_geometric_duplicate"]).sum()
        ),
        "geometry_only": int(
            (~groups["in_smiles_duplicate"] & groups["in_geometric_duplicate"]).sum()
        ),
        "both": int((groups["in_smiles_duplicate"] & groups["in_geometric_duplicate"]).sum()),
    }
    return summary
