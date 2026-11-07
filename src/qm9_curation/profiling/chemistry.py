"""Per-molecule cheminformatics profile with RDKit (Phase 3).

Descriptive only: this records what RDKit makes of each source string
(parse status, canonical forms, InChIKeys, rings, scaffolds, charges,
stereocentres). It does not judge validity; Phase 4 does that with
explicit checks and the 3D geometry.

Both SMILES are profiled: ``gdb`` is the intended molecule from GDB-17,
``relaxed`` is the authors' SMILES for the optimized geometry.
"""

from typing import Any

import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

CHEM_PROFILE_SCHEMA: dict[str, pl.DataType] = {
    "qm9_id": pl.UInt32(),
    "gdb_rdkit_ok": pl.Boolean(),
    "relaxed_rdkit_ok": pl.Boolean(),
    "canonical_smiles_gdb": pl.String(),
    "canonical_smiles_relaxed": pl.String(),
    "flat_smiles_gdb": pl.String(),
    "flat_smiles_relaxed": pl.String(),
    "inchikey_gdb": pl.String(),
    "inchikey_relaxed": pl.String(),
    "n_rings": pl.Int16(),
    "ring_sizes": pl.String(),
    "n_aromatic_atoms": pl.Int16(),
    "murcko_scaffold": pl.String(),
    "generic_scaffold": pl.String(),
    "n_charged_atoms": pl.Int16(),
    "n_radical_electrons": pl.Int16(),
    "n_stereocentres": pl.Int16(),
}


def _smiles(mol: Chem.Mol | None, isomeric: bool) -> str | None:
    return None if mol is None else Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def _inchikey(inchi: str) -> str | None:
    key = Chem.InchiToInchiKey(inchi)
    return key or None


def profile_molecule(
    qm9_id: int, smiles_gdb: str, smiles_relaxed: str, inchi_gdb: str, inchi_relaxed: str
) -> dict[str, Any]:
    gdb = Chem.MolFromSmiles(smiles_gdb)
    relaxed = Chem.MolFromSmiles(smiles_relaxed)
    row: dict[str, Any] = {
        "qm9_id": qm9_id,
        "gdb_rdkit_ok": gdb is not None,
        "relaxed_rdkit_ok": relaxed is not None,
        "canonical_smiles_gdb": _smiles(gdb, isomeric=True),
        "canonical_smiles_relaxed": _smiles(relaxed, isomeric=True),
        "flat_smiles_gdb": _smiles(gdb, isomeric=False),
        "flat_smiles_relaxed": _smiles(relaxed, isomeric=False),
        "inchikey_gdb": _inchikey(inchi_gdb),
        "inchikey_relaxed": _inchikey(inchi_relaxed),
    }
    # Ring and scaffold statistics describe the relaxed structure (the geometry is the
    # ground truth); fall back to the GDB-17 molecule when the relaxed SMILES won't parse.
    mol = relaxed if relaxed is not None else gdb
    if mol is None:
        return {**row, **{k: None for k in CHEM_PROFILE_SCHEMA if k not in row}}
    ring_info = mol.GetRingInfo()
    core = MurckoScaffold.GetScaffoldForMol(mol)
    row.update(
        {
            "n_rings": ring_info.NumRings(),
            "ring_sizes": ",".join(str(len(r)) for r in sorted(ring_info.AtomRings(), key=len)),
            "n_aromatic_atoms": sum(a.GetIsAromatic() for a in mol.GetAtoms()),
            "murcko_scaffold": Chem.MolToSmiles(core),
            "generic_scaffold": Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core)),
            "n_charged_atoms": sum(a.GetFormalCharge() != 0 for a in mol.GetAtoms()),
            "n_radical_electrons": sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
            "n_stereocentres": len(
                Chem.FindMolChiralCenters(
                    mol, includeUnassigned=True, useLegacyImplementation=False
                )
            ),
        }
    )
    return row


def chemistry_profile(molecules: pl.DataFrame) -> pl.DataFrame:
    cols = ["qm9_id", "smiles_gdb", "smiles_relaxed", "inchi_gdb", "inchi_relaxed"]
    rows = [profile_molecule(*r) for r in molecules.select(cols).iter_rows()]
    return pl.DataFrame(rows, schema=CHEM_PROFILE_SCHEMA, orient="row").sort("qm9_id")
