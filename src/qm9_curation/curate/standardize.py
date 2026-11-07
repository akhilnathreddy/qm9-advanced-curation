"""Standardized representation of each molecule (Phase 5).

Rules (ADR 0009):

* **Source values are never overwritten.** Converted values get new,
  unit-suffixed columns; the source-unit columns are carried through unchanged.
* **Units:** energies in eV, polarizability in Å³, spatial extent in Å²;
  dipole (D), rotational constants (GHz) and Cv (cal/(mol·K)) keep their
  source units. Every factor comes from ``units.py``.
* **Atomization energies** use only the atom references shipped with the
  release (``atomref.txt``, same level of theory): E_mol - sum(E_atom), so they
  are negative for bound molecules.
* **Linear molecules** (by geometry, ADR 0008): rotational constant A is
  undefined, stored as null with ``rot_a_missing_reason``.
* **Identifiers from the geometry:** canonical isomeric SMILES and standard
  InChIKey of the structure perceived from 3D, with stereo assigned from the
  coordinates. Null when RDKit cannot assign bond orders, with the reason.
"""

from collections.abc import Mapping
from typing import Any

import polars as pl
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from qm9_curation.checks.identity import xyz_block
from qm9_curation.parallel import map_chunks
from qm9_curation.units import BOHR2_TO_ANGSTROM2, BOHR3_TO_ANGSTROM3, HARTREE_TO_EV

ENERGY_COLUMNS = ("homo", "lumo", "gap", "zpve", "u0", "u298", "h298", "g298")
ATOMIZATION = {
    "u0": "u0",
    "u298": "u298",
    "h298": "h298",
    "g298": "g298",
}  # molecule col -> atomref key


def converted_columns() -> list[pl.Expr]:
    exprs = [(pl.col(f"{e}_ha") * HARTREE_TO_EV).alias(f"{e}_ev") for e in ENERGY_COLUMNS]
    exprs += [
        (pl.col("alpha_bohr3") * BOHR3_TO_ANGSTROM3).alias("alpha_ang3"),
        (pl.col("r2_bohr2") * BOHR2_TO_ANGSTROM2).alias("r2_ang2"),
    ]
    return exprs


def atom_reference_sums(
    atoms: pl.DataFrame, atomref: Mapping[str, Mapping[str, float]]
) -> pl.DataFrame:
    """Σ E_atom (Hartree) per molecule for each thermochemical quantity."""
    ref = pl.DataFrame(
        [
            {"element": el, **{f"ref_{k}": v[k] for k in ATOMIZATION.values()}}
            for el, v in atomref.items()
        ]
    )
    return (
        atoms.select("qm9_id", "element")
        .join(ref, on="element", how="left")
        .group_by("qm9_id")
        .agg(pl.col(f"ref_{k}").sum() for k in ATOMIZATION.values())
    )


def atomization_columns() -> list[pl.Expr]:
    return [
        ((pl.col(f"{mol_col}_ha") - pl.col(f"ref_{ref_key}")) * HARTREE_TO_EV).alias(
            f"{mol_col}_atomization_ev"
        )
        for mol_col, ref_key in ATOMIZATION.items()
    ]


def geometry_identifiers(
    elements: list[str], coords: list[tuple[float, float, float]]
) -> dict[str, Any]:
    mol = Chem.MolFromXYZBlock(xyz_block(elements, coords))
    try:
        rdDetermineBonds.DetermineConnectivity(mol, useHueckel=True, charge=0)
        rdDetermineBonds.DetermineBondOrders(mol, charge=0)
        Chem.AssignStereochemistryFrom3D(mol)
        no_h = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(no_h)
        inchikey = Chem.MolToInchiKey(mol) or None
    except (ValueError, RuntimeError) as exc:
        return {
            "smiles_geometry": None,
            "inchikey_geometry": None,
            "geometry_identifier_error": type(exc).__name__,
        }
    return {
        "smiles_geometry": smiles,
        "inchikey_geometry": inchikey,
        "geometry_identifier_error": None,
    }


def _identifier_chunk(rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    return [{"qm9_id": q, **geometry_identifiers(e, c)} for q, e, c in rows]


def identifiers_from_geometry(atoms: pl.DataFrame, workers: int) -> pl.DataFrame:
    per_mol = (
        atoms.sort(["qm9_id", "atom_idx"])
        .group_by("qm9_id", maintain_order=True)
        .agg(pl.col("element"), pl.concat_list("x_ang", "y_ang", "z_ang").alias("xyz"))
        .sort("qm9_id")
    )
    rows = [(q, e, [tuple(c) for c in xyz]) for q, e, xyz in per_mol.iter_rows()]
    out = pl.DataFrame(
        map_chunks(_identifier_chunk, rows, workers),
        schema={
            "qm9_id": pl.UInt32,
            "smiles_geometry": pl.String,
            "inchikey_geometry": pl.String,
            "geometry_identifier_error": pl.String,
        },
        orient="row",
    )
    return out.sort("qm9_id")


def standardize(
    molecules: pl.DataFrame,
    atoms: pl.DataFrame,
    validation: pl.DataFrame,
    atomref: Mapping[str, Mapping[str, float]],
) -> pl.DataFrame:
    """Converted units, atomization energies and linear-molecule handling (no RDKit)."""
    linear = validation.select(
        "qm9_id", "linear_by_geometry", "zpve_consistent_set", "identity_class"
    )
    return (
        molecules.join(atom_reference_sums(atoms, atomref), on="qm9_id")
        .join(linear, on="qm9_id")
        .with_columns(*converted_columns(), *atomization_columns())
        .with_columns(
            pl.when(pl.col("linear_by_geometry"))
            .then(None)
            .otherwise(pl.col("rot_a_ghz"))
            .alias("rot_a_ghz_std"),
            pl.when(pl.col("linear_by_geometry"))
            .then(pl.lit("linear molecule: A undefined"))
            .otherwise(None)
            .alias("rot_a_missing_reason"),
            pl.col("smiles_gdb").str.contains(r"\+|-\]").alias("gdb_charge_separated"),
        )
        .drop(*(f"ref_{k}" for k in ATOMIZATION.values()))
        .sort("qm9_id")
    )
