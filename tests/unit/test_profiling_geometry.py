import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from qm9_curation.profiling import geometry

WATER = np.array([[-0.0344, 0.9775, 0.0076], [0.0648, 0.0206, 0.0015], [0.8718, 1.3008, 0.0007]])
Z_WATER = np.array([8.0, 1.0, 1.0])


def atoms_frame(molecules: dict[int, tuple[np.ndarray, list[str]]]) -> pl.DataFrame:
    z_of = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    rows = []
    for qid, (xyz, elements) in molecules.items():
        for i, (pos, e) in enumerate(zip(xyz, elements, strict=True)):
            rows.append((qid, i, e, z_of[e], *pos.tolist(), 0.0))
    return pl.DataFrame(
        rows,
        schema={
            "qm9_id": pl.UInt32,
            "atom_idx": pl.UInt8,
            "element": pl.String,
            "atomic_number": pl.UInt8,
            "x_ang": pl.Float64,
            "y_ang": pl.Float64,
            "z_ang": pl.Float64,
            "mulliken_charge_e": pl.Float64,
        },
        orient="row",
    )


@settings(max_examples=50)
@given(st.integers(min_value=0, max_value=10_000), st.permutations([0, 1, 2]))
def test_fingerprints_are_invariant_to_rotation_translation_permutation(
    seed: int, perm: list[int]
) -> None:
    rot = Rotation.random(random_state=seed).as_matrix()
    moved = (WATER @ rot.T + np.array([1.5, -2.0, 0.3]))[perm]
    d0 = geometry.pairwise_distances(WATER[None])
    d1 = geometry.pairwise_distances(moved[None])
    e0 = geometry.coulomb_eigenvalues(Z_WATER[None], d0)
    e1 = geometry.coulomb_eigenvalues(Z_WATER[perm][None], d1)
    np.testing.assert_allclose(e0, e1, atol=1e-9)
    np.testing.assert_allclose(
        geometry.sorted_distance_vectors(d0), geometry.sorted_distance_vectors(d1), atol=1e-12
    )


def test_duplicate_search_finds_moved_copy_but_not_a_distorted_one() -> None:
    rot = Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()
    copy = (WATER @ rot.T + 3.0)[[2, 0, 1]]
    distorted = WATER.copy()
    distorted[2] += [0.0, 0.3, 0.0]
    elements = ["O", "H", "H"]
    atoms = atoms_frame(
        {1: (WATER, elements), 2: (copy, ["H", "O", "H"]), 3: (distorted, elements)}
    )
    formulas = pl.DataFrame(
        {"qm9_id": [1, 2, 3], "formula": ["H2O"] * 3},
        schema={"qm9_id": pl.UInt32, "formula": pl.String},
    )
    result = geometry.profile_geometry(atoms, formulas)
    assert result.duplicate_pairs.select("qm9_id_a", "qm9_id_b").rows() == [(1, 2)]
    water = result.per_molecule.filter(pl.col("qm9_id") == 1)
    assert 0.9 < water["min_interatomic_ang"].item() < 1.0


def test_different_formulas_are_never_paired() -> None:
    atoms = atoms_frame({1: (WATER, ["O", "H", "H"]), 2: (WATER, ["O", "H", "H"])})
    formulas = pl.DataFrame(
        {"qm9_id": [1, 2], "formula": ["H2O", "XH2"]},
        schema={"qm9_id": pl.UInt32, "formula": pl.String},
    )
    assert geometry.profile_geometry(atoms, formulas).duplicate_pairs.height == 0
