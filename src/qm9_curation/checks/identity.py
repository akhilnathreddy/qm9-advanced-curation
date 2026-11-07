"""Layer 2 — chemical identity from three independent sources.

* **A** — the GDB-17 SMILES: the molecule the record is supposed to be.
* **B** — the authors' relaxed SMILES (Open Babel bond perception).
* **C** — bonds re-derived here from the B3LYP coordinates with RDKit
  ``rdDetermineBonds``: connectivity from extended-Hückel bond orders, then
  integer bond orders for a neutral molecule. The default covalent-radius
  connectivity bonds the bridgehead carbons of strained cages
  (bicyclo[1.1.1]pentanes, 1.88 Å apart) and was rejected (ADR 0008).

Graphs are compared as **skeletons**: every atom including hydrogens, all
bonds made single, charges and aromaticity removed. Two skeletons are equal
exactly when the molecules have the same atoms connected the same way, so the
comparison ignores bond-order and charge notation (which is where Open Babel,
RDKit and GDB-17 disagree most) while still catching a hydrogen that moved.
A second, heavy-atom-only skeleton separates proton transfer from changes to
the carbon/heteroatom framework.
"""

from dataclasses import dataclass
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

IDENTITY_CLASSES = ("consistent", "proton_moved", "rearranged", "fragmented", "unperceived")
# InChI without stereo layers: GDB-17 has no stereo, the 3D structure always does.
INCHI_NO_STEREO = "/SNon"


def skeleton(mol: Chem.Mol, keep_h: bool = True) -> str:
    """Canonical SMILES of the bare connectivity graph."""
    rw = Chem.RWMol(mol)
    if not keep_h:
        for atom in sorted(
            (a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1), reverse=True
        ):
            rw.RemoveAtom(atom)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return str(Chem.MolToSmiles(out, canonical=True, allHsExplicit=False))


def mol_from_smiles_with_h(smiles: str) -> Chem.Mol | None:
    """Parse leniently so Open Babel-style SMILES (e.g. ``[NH3]C...[O]``) still yield a graph."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    return Chem.AddHs(mol)


def xyz_block(elements: list[str], coords: list[tuple[float, float, float]]) -> str:
    lines = [
        f"{e} {x:.10f} {y:.10f} {z:.10f}" for e, (x, y, z) in zip(elements, coords, strict=True)
    ]
    return f"{len(elements)}\n\n" + "\n".join(lines) + "\n"


@dataclass(frozen=True)
class GeometryGraph:
    full: str
    heavy: str
    n_fragments: int
    inchi: str | None  # standard InChI after bond-order assignment; None if that failed


def perceive(elements: list[str], coords: list[tuple[float, float, float]]) -> GeometryGraph:
    mol = Chem.MolFromXYZBlock(xyz_block(elements, coords))
    rdDetermineBonds.DetermineConnectivity(mol, useHueckel=True, charge=0)
    n_fragments = len(Chem.GetMolFrags(mol))
    full, heavy = skeleton(mol), skeleton(mol, keep_h=False)
    inchi = None
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=0)
        inchi = Chem.MolToInchi(mol, options=INCHI_NO_STEREO) or None
    except (ValueError, RuntimeError):
        inchi = None
    return GeometryGraph(full=full, heavy=heavy, n_fragments=n_fragments, inchi=inchi)


def classify(geom: GeometryGraph, gdb_full: str, gdb_heavy: str) -> str:
    if geom.n_fragments > 1:
        return "fragmented"
    if geom.full == gdb_full:
        return "consistent"
    if geom.heavy == gdb_heavy:
        return "proton_moved"
    return "rearranged"


def identity_record(
    qm9_id: int,
    smiles_gdb: str,
    smiles_relaxed: str,
    elements: list[str],
    coords: list[tuple[float, float, float]],
) -> dict[str, Any]:
    gdb = mol_from_smiles_with_h(smiles_gdb)
    relaxed = mol_from_smiles_with_h(smiles_relaxed)
    geom = perceive(elements, coords)
    gdb_full = skeleton(gdb) if gdb is not None else ""
    gdb_heavy = skeleton(gdb, keep_h=False) if gdb is not None else ""
    relaxed_full = skeleton(relaxed) if relaxed is not None else None
    parsed_gdb = Chem.MolFromSmiles(smiles_gdb)
    inchi_gdb = (
        Chem.MolToInchi(parsed_gdb, options=INCHI_NO_STEREO) if parsed_gdb is not None else None
    )
    return {
        "qm9_id": qm9_id,
        "identity_class": classify(geom, gdb_full, gdb_heavy) if gdb is not None else "unperceived",
        "geom_n_fragments": geom.n_fragments,
        "gdb_matches_geometry": geom.full == gdb_full,
        "relaxed_matches_geometry": None if relaxed_full is None else geom.full == relaxed_full,
        "gdb_matches_relaxed": None if relaxed_full is None else gdb_full == relaxed_full,
        "geom_bond_orders_assigned": geom.inchi is not None,
        "geom_inchi_matches_gdb": None if geom.inchi is None else geom.inchi == inchi_gdb,
        "geometry_skeleton": geom.full,
    }
