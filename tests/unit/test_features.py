import numpy as np
import polars as pl
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation

from qm9_curation.features.composition import composition
from qm9_curation.features.coulomb import bag_of_bonds, coulomb_eigen_features
from qm9_curation.features.inputs import (
    ALLOWED_ATOM_COLUMNS,
    LeakageError,
    require_allowed,
    restrict,
)
from qm9_curation.features.molecular import molecule_features
from qm9_curation.units import BOHR_TO_ANGSTROM

Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def embedded(smiles: str) -> tuple[list[str], np.ndarray]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=11)
    AllChem.MMFFOptimizeMolecule(mol)
    return [a.GetSymbol() for a in mol.GetAtoms()], mol.GetConformer().GetPositions()


def atoms_frame(
    qm9_id: int, elements: list[str], xyz: np.ndarray, **extra: list[float]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "qm9_id": [qm9_id] * len(elements),
            "atom_idx": list(range(len(elements))),
            "element": elements,
            "atomic_number": [Z[e] for e in elements],
            "x_ang": xyz[:, 0],
            "y_ang": xyz[:, 1],
            "z_ang": xyz[:, 2],
            **extra,
        }
    ).with_columns(
        pl.col("qm9_id").cast(pl.UInt32), pl.col("atom_idx", "atomic_number").cast(pl.UInt8)
    )


def test_leakage_guard_rejects_quantum_by_products() -> None:
    elements, xyz = embedded("CCO")
    frame = atoms_frame(1, elements, xyz, mulliken_charge_e=[0.0] * len(elements))
    with pytest.raises(LeakageError, match="mulliken_charge_e"):
        composition(frame)
    with pytest.raises(LeakageError):
        coulomb_eigen_features(frame)
    assert list(restrict(frame).columns) == list(ALLOWED_ATOM_COLUMNS)
    require_allowed(restrict(frame))
    assert not {"mulliken_charge_e", "wavenumber_cm1", "zpve_ha", "rot_a_ghz"} & set(
        ALLOWED_ATOM_COLUMNS
    )


def test_composition_counts() -> None:
    elements, xyz = embedded("NCC(=O)OF")
    row = composition(atoms_frame(7, elements, xyz)).row(0, named=True)
    assert (row["n_C"], row["n_N"], row["n_O"], row["n_F"], row["n_H"]) == (2, 1, 2, 1, 4)
    assert row["n_heavy"] == 6 and row["n_atoms"] == 10


def test_coulomb_eigenvalues_invariant_to_rotation_translation_permutation() -> None:
    elements, xyz = embedded("CC(=O)N")
    perm = np.random.default_rng(3).permutation(len(elements))
    moved = (xyz @ Rotation.random(random_state=5).as_matrix().T + 4.2)[perm]
    a = coulomb_eigen_features(atoms_frame(1, elements, xyz)).drop("qm9_id").row(0)
    b = (
        coulomb_eigen_features(atoms_frame(1, [elements[i] for i in perm], moved))
        .drop("qm9_id")
        .row(0)
    )
    np.testing.assert_allclose(a, b, atol=1e-9)
    assert a[len(elements) :] == (0.0,) * (29 - len(elements))  # zero padding


def test_coulomb_uses_bohr() -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [BOHR_TO_ANGSTROM, 0.0, 0.0]])  # 1 Bohr apart
    eig = coulomb_eigen_features(atoms_frame(1, ["H", "H"], xyz)).drop("qm9_id").row(0)
    # M = [[0.5, 1], [1, 0.5]] -> eigenvalues 1.5, -0.5
    assert eig[:2] == pytest.approx((1.5, -0.5))


def test_bag_of_bonds_separates_self_terms_and_is_permutation_invariant() -> None:
    elements, xyz = embedded("CO")
    z = np.array([Z[e] for e in elements], dtype=float)
    # Methanol: every atom pairs with every other, so 4 C-H and 4 O-H pair terms.
    sizes = {(0, 1): 4, (0, 6): 1, (0, 8): 1, (1, 1): 6, (1, 6): 4, (1, 8): 4, (6, 8): 1, (6, 6): 0}
    v = bag_of_bonds(z, xyz, sizes)
    perm = np.random.default_rng(1).permutation(len(z))
    np.testing.assert_allclose(v, bag_of_bonds(z[perm], xyz[perm], sizes))
    assert v[sizes[(0, 1)] : sizes[(0, 1)] + 1] == pytest.approx([0.5 * 6**2.4])  # (0, 6) self bag
    with pytest.raises(ValueError, match="bag"):
        bag_of_bonds(z, xyz, {**sizes, (1, 6): 3})


def test_rdkit_features_are_invariant_to_atom_order_and_pose() -> None:
    elements, xyz = embedded("CC1=CC(=O)NC1")
    perm = np.random.default_rng(9).permutation(len(elements))
    moved = xyz @ Rotation.random(random_state=2).as_matrix().T
    a = molecule_features(1, elements, [tuple(p) for p in xyz])
    b = molecule_features(1, [elements[i] for i in perm], [tuple(p) for p in moved[perm]])
    assert a["ok"] and b["ok"]
    assert a["descriptors"] == pytest.approx(b["descriptors"], nan_ok=True)
    assert (a["ecfp4_bits"], a["ecfp4_counts"], a["maccs_bits"]) == (
        b["ecfp4_bits"],
        b["ecfp4_counts"],
        b["maccs_bits"],
    )

    def node_key(n: dict[str, object]) -> tuple[object, ...]:
        return (n["atomic_number"], n["degree"], n["total_h"], n["hybridization"], n["is_aromatic"])

    assert sorted(map(node_key, a["nodes"])) == sorted(map(node_key, b["nodes"]))
    assert len(a["edges"]) == len(b["edges"])


def test_graph_atom_index_follows_source_order() -> None:
    elements, xyz = embedded("OC")
    f = molecule_features(1, elements, [tuple(p) for p in xyz])
    assert [n["atomic_number"] for n in f["nodes"]] == [Z[e] for e in elements]
    lengths = {(e["atom_i"], e["atom_j"]): e["length_ang"] for e in f["edges"]}
    assert lengths[(0, 1)] == pytest.approx(float(np.linalg.norm(xyz[0] - xyz[1])))
