"""Per-molecule RDKit features from the structure perceived in 3D (Phase 6).

The molecule is perceived from the coordinates exactly as in Phase 5
(Hückel connectivity, bond orders for a neutral molecule, stereo from 3D),
so features describe the computed structure, not the GDB-17 label. RDKit
keeps the XYZ atom order, so graph atom indices match ``atoms.atom_idx``.

Produces, per molecule: the full RDKit descriptor set, ECFP4 counts and MACCS
keys (both stored sparsely), and atom / bond rows for graph models.
"""

from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdDetermineBonds, rdFingerprintGenerator

from qm9_curation.checks.identity import xyz_block

ECFP_RADIUS = 2
ECFP_SIZE = 2048
DESCRIPTOR_NAMES = tuple(name for name, _ in Descriptors._descList)
_ECFP = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_SIZE)
HYBRIDIZATION = {
    Chem.HybridizationType.S: 0,
    Chem.HybridizationType.SP: 1,
    Chem.HybridizationType.SP2: 2,
    Chem.HybridizationType.SP3: 3,
}


def perceive(elements: list[str], coords: list[tuple[float, float, float]]) -> Chem.Mol | None:
    mol = Chem.MolFromXYZBlock(xyz_block(elements, coords))
    try:
        rdDetermineBonds.DetermineConnectivity(mol, useHueckel=True, charge=0)
        rdDetermineBonds.DetermineBondOrders(mol, charge=0)
        Chem.AssignStereochemistryFrom3D(mol)
    except (ValueError, RuntimeError):
        return None
    return mol


def descriptors(mol: Chem.Mol) -> dict[str, float]:
    heavy = Chem.RemoveHs(mol)
    values = Descriptors.CalcMolDescriptors(heavy, missingVal=float("nan"))
    return {name: float(values[name]) for name in DESCRIPTOR_NAMES}


def ecfp_counts(mol: Chem.Mol) -> tuple[list[int], list[int]]:
    sparse = _ECFP.GetCountFingerprint(Chem.RemoveHs(mol)).GetNonzeroElements()
    keys = sorted(sparse)
    return keys, [sparse[k] for k in keys]


def maccs_bits(mol: Chem.Mol) -> list[int]:
    return list(MACCSkeys.GenMACCSKeys(Chem.RemoveHs(mol)).GetOnBits())  # type: ignore[attr-defined]


def graph_rows(qm9_id: int, mol: Chem.Mol) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "qm9_id": qm9_id,
            "atom_idx": a.GetIdx(),
            "atomic_number": a.GetAtomicNum(),
            "degree": a.GetDegree(),
            "total_h": a.GetTotalNumHs(includeNeighbors=True),
            "formal_charge": a.GetFormalCharge(),
            "hybridization": HYBRIDIZATION.get(a.GetHybridization(), 4),
            "is_aromatic": a.GetIsAromatic(),
            "in_ring": a.IsInRing(),
        }
        for a in mol.GetAtoms()
    ]
    conf = mol.GetConformer()
    edges = []
    for b in mol.GetBonds():
        i, j = sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        edges.append(
            {
                "qm9_id": qm9_id,
                "atom_i": i,
                "atom_j": j,
                "bond_order": float(b.GetBondTypeAsDouble()),
                "is_aromatic": b.GetIsAromatic(),
                "is_conjugated": b.GetIsConjugated(),
                "in_ring": b.IsInRing(),
                "length_ang": float((conf.GetAtomPosition(i) - conf.GetAtomPosition(j)).Length()),
            }
        )
    edges.sort(key=lambda e: (e["atom_i"], e["atom_j"]))
    return nodes, edges


def molecule_features(
    qm9_id: int, elements: list[str], coords: list[tuple[float, float, float]]
) -> dict[str, Any]:
    mol = perceive(elements, coords)
    if mol is None:
        return {"qm9_id": qm9_id, "ok": False}
    bits, counts = ecfp_counts(mol)
    nodes, edges = graph_rows(qm9_id, mol)
    return {
        "qm9_id": qm9_id,
        "ok": True,
        "descriptors": descriptors(mol),
        "ecfp4_bits": bits,
        "ecfp4_counts": counts,
        "maccs_bits": maccs_bits(mol),
        "nodes": nodes,
        "edges": edges,
    }
