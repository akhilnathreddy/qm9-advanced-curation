import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from qm9_curation.checks.identity import (
    classify,
    identity_record,
    mol_from_smiles_with_h,
    perceive,
    skeleton,
)


def embed(smiles: str, seed: int = 7) -> tuple[list[str], list[tuple[float, float, float]]]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    pos = mol.GetConformer().GetPositions()
    return [a.GetSymbol() for a in mol.GetAtoms()], [tuple(map(float, p)) for p in pos]


def test_skeleton_ignores_bond_orders_charges_and_aromaticity() -> None:
    a = mol_from_smiles_with_h("C1=CN=CC=N1")  # Kekulé form
    b = mol_from_smiles_with_h("c1cnccn1")  # aromatic form
    c = mol_from_smiles_with_h("[NH3+]CC([O-])=O")
    d = mol_from_smiles_with_h("[NH3]CC(=O)[O]")  # Open Babel style, not RDKit-valid
    assert a is not None and b is not None and c is not None and d is not None
    assert skeleton(a) == skeleton(b)
    assert skeleton(c) == skeleton(d)


def test_skeleton_detects_proton_transfer_but_heavy_skeleton_does_not() -> None:
    zwitterion = mol_from_smiles_with_h("[NH3+]CC([O-])=O")
    neutral = mol_from_smiles_with_h("NCC(=O)O")
    assert zwitterion is not None and neutral is not None
    assert skeleton(zwitterion) != skeleton(neutral)
    assert skeleton(zwitterion, keep_h=False) == skeleton(neutral, keep_h=False)


def test_consistent_molecule_is_classified_consistent() -> None:
    elements, coords = embed("CC(=O)OC")
    rec = identity_record(1, "CC(=O)OC", "CC(=O)OC", elements, coords)
    assert rec["identity_class"] == "consistent"
    assert rec["gdb_matches_geometry"] and rec["relaxed_matches_geometry"]
    assert rec["geom_inchi_matches_gdb"] is True


def test_strained_cage_bridgeheads_are_not_bonded() -> None:
    # bicyclo[1.1.1]pentane: bridgehead C...C ~1.87 Å must not become a bond (ADR 0008).
    elements, coords = embed("C1C2CC1C2")
    assert perceive(elements, coords).heavy == skeleton(
        mol_from_smiles_with_h("C1C2CC1C2"), keep_h=False
    )  # type: ignore[arg-type]


def test_fragmented_geometry() -> None:
    e1, c1 = embed("C=C1C=CC(=O)O1")
    e2, c2 = embed("N")
    far = [(x + 8.0, y, z) for x, y, z in c2]
    rec = identity_record(2, "[NH3+]CC1=CC=C([O-])O1", "N.C=C1C=CC(=O)O1", e1 + e2, c1 + far)
    assert rec["identity_class"] == "fragmented" and rec["geom_n_fragments"] == 2
    assert rec["relaxed_matches_geometry"] is True  # the authors' SMILES recorded the split


def test_proton_moved_and_rearranged_classes() -> None:
    geom = perceive(*embed("NCC(=O)O"))
    gdb = mol_from_smiles_with_h("[NH3+]CC([O-])=O")
    assert gdb is not None
    assert classify(geom, skeleton(gdb), skeleton(gdb, keep_h=False)) == "proton_moved"
    other = mol_from_smiles_with_h("CC(=O)OC")
    assert other is not None
    assert classify(geom, skeleton(other), skeleton(other, keep_h=False)) == "rearranged"


def test_coordinates_are_used_not_just_order() -> None:
    elements, coords = embed("CCO")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(elements))
    shuffled = perceive([elements[i] for i in perm], [coords[i] for i in perm])
    assert shuffled.full == perceive(elements, coords).full
