"""Grouping keys that splits keep within one subset.

* **Leakage group** (every split): the connected components of records that
  share any of
  - ``structure_group_id``: exact duplicates (ADR 0007);
  - ``tautomer_group_id``: standard InChI of the GDB-17 label;
  - ``connectivity_group_id``: GDB-17 connectivity;
  - the connectivity block of the InChIKey perceived from the geometry, which
    also covers stereoisomers, other conformers and records that relaxed to the
    same molecule (Phase 7 found 21 such pairs missed by the first three).
* **Scaffold key:** generic Bemis-Murcko scaffold of the geometry structure.
  Acyclic molecules would all share the empty scaffold (14,117 of them), so
  they are bucketed by heavy-atom composition (e.g. ``acyclic:C4N1O1``).
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import polars as pl
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from qm9_curation.parallel import map_chunks

LEAKAGE_KEYS = ("structure_group_id", "tautomer_group_id", "connectivity_group_id", "ik_block")


def components(ids: Sequence[int], keys: Iterable[Sequence[Any]]) -> list[int]:
    """Union-find over records sharing any non-null key; label = smallest id in the component."""
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for column in keys:
        first: dict[Any, int] = {}
        for i, k in enumerate(column):
            if k is None:
                continue
            if k in first:
                a, b = find(i), find(first[k])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first[k] = i
    label: dict[int, int] = {}
    for i in sorted(range(len(ids)), key=lambda i: ids[i]):
        label.setdefault(find(i), ids[i])  # first visit in id order = smallest id
    return [label[find(i)] for i in range(len(ids))]


def leakage_groups(curated: pl.DataFrame) -> pl.Series:
    frame = curated.sort("qm9_id").with_columns(
        pl.col("inchikey_geometry").str.slice(0, 14).alias("ik_block")
    )
    labels = components(frame["qm9_id"].to_list(), (frame[k].to_list() for k in LEAKAGE_KEYS))
    return pl.Series("leakage_group_id", labels, dtype=pl.UInt32)


def scaffold_key(smiles: str | None) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    if mol.GetRingInfo().NumRings() == 0:
        counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
        heavy = "".join(f"{e}{counts[e]}" for e in ("C", "N", "O", "F") if counts[e])
        return f"acyclic:{heavy}"
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return str(Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core)))


def _scaffold_chunk(rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    return [{"qm9_id": q, "scaffold_key": scaffold_key(s)} for q, s in rows]


def scaffold_keys(curated: pl.DataFrame, workers: int) -> pl.DataFrame:
    """Geometry SMILES where available, else the GDB-17 label (22 records)."""
    rows = list(
        curated.sort("qm9_id")
        .select("qm9_id", pl.coalesce("smiles_geometry", "canonical_smiles_gdb"))
        .iter_rows()
    )
    return pl.DataFrame(
        map_chunks(_scaffold_chunk, rows, workers),
        schema={"qm9_id": pl.UInt32, "scaffold_key": pl.String},
        orient="row",
    )
