"""Geometry profile and near-duplicate geometry search (Phase 3).

Works on molecules batched by atom count so every array operation is
vectorized. Two permutation-invariant fingerprints are used:

* sorted eigenvalues of the Coulomb matrix (cheap; used to find candidates);
* the sorted list of all interatomic distances (used to confirm a candidate).

Both are invariant to rotation, translation and atom order. They are also
identical for mirror images, so a "geometric duplicate" pair may be two
enantiomers; the report says so rather than assuming.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.spatial import cKDTree

# Chosen from a threshold scan (docs/profiling.md): up to these values every pair found has
# |dU0| <= 0.023 kcal/mol (same molecule); looser values start pairing distinct molecules.
EIGEN_RADIUS = 0.1  # candidate search radius in Coulomb-eigenvalue space
DISTANCE_TOL_ANG = 0.05  # max difference in any sorted interatomic distance


@dataclass(frozen=True)
class Batch:
    """All molecules with the same atom count."""

    qm9_ids: NDArray[np.int64]
    z: NDArray[np.float64]  # (m, n)
    xyz: NDArray[np.float64]  # (m, n, 3)


def batches(atoms: pl.DataFrame) -> Iterator[Batch]:
    atoms = atoms.sort(["qm9_id", "atom_idx"])
    counts = atoms.group_by("qm9_id", maintain_order=True).len()
    with_n = atoms.join(counts.rename({"len": "n"}), on="qm9_id")
    for (n,), part in with_n.group_by("n", maintain_order=True):
        m = part.height // int(n)
        yield Batch(
            qm9_ids=part["qm9_id"].to_numpy()[:: int(n)].astype(np.int64),
            z=part["atomic_number"].to_numpy().astype(np.float64).reshape(m, int(n)),
            xyz=part.select("x_ang", "y_ang", "z_ang").to_numpy().reshape(m, int(n), 3),
        )


def pairwise_distances(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    diff = xyz[:, :, None, :] - xyz[:, None, :, :]
    return np.sqrt((diff**2).sum(-1))


def coulomb_eigenvalues(z: NDArray[np.float64], d: NDArray[np.float64]) -> NDArray[np.float64]:
    n = z.shape[1]
    with np.errstate(divide="ignore"):
        cm = z[:, :, None] * z[:, None, :] / d
    idx = np.arange(n)
    cm[:, idx, idx] = 0.5 * z**2.4
    return np.sort(np.linalg.eigvalsh(cm), axis=1)[:, ::-1]


def sorted_distance_vectors(d: NDArray[np.float64]) -> NDArray[np.float64]:
    iu = np.triu_indices(d.shape[1], k=1)
    return np.sort(d[:, iu[0], iu[1]], axis=1)


@dataclass
class GeometryProfile:
    per_molecule: pl.DataFrame
    duplicate_pairs: pl.DataFrame


def profile_geometry(atoms: pl.DataFrame, formulas: pl.DataFrame) -> GeometryProfile:
    formula_of = dict(zip(formulas["qm9_id"].to_list(), formulas["formula"].to_list(), strict=True))
    rows: list[pl.DataFrame] = []
    pairs: list[tuple[int, int, float]] = []
    for batch in batches(atoms):
        n = batch.z.shape[1]
        d = pairwise_distances(batch.xyz)
        off = np.where(np.eye(n, dtype=bool), np.inf, d)
        centred = batch.xyz - batch.xyz.mean(axis=1, keepdims=True)
        rows.append(
            pl.DataFrame(
                {
                    "qm9_id": batch.qm9_ids,
                    "min_interatomic_ang": off.min(axis=(1, 2)) if n > 1 else np.nan,
                    "max_interatomic_ang": d.max(axis=(1, 2)),
                    "radius_of_gyration_ang": np.sqrt((centred**2).sum(-1).mean(-1)),
                }
            )
        )
        eig = coulomb_eigenvalues(batch.z, d)
        sdv = sorted_distance_vectors(d)
        groups: dict[str, list[int]] = {}
        for i, qid in enumerate(batch.qm9_ids.tolist()):
            groups.setdefault(formula_of[qid], []).append(i)
        for members in groups.values():
            if len(members) < 2:
                continue
            idx = np.array(members)
            for a, b in sorted(cKDTree(eig[idx]).query_pairs(EIGEN_RADIUS)):
                i, j = idx[a], idx[b]
                gap = float(np.abs(sdv[i] - sdv[j]).max())
                if gap <= DISTANCE_TOL_ANG:
                    lo, hi = sorted((int(batch.qm9_ids[i]), int(batch.qm9_ids[j])))
                    pairs.append((lo, hi, gap))
    per_molecule = pl.concat(rows).sort("qm9_id")
    duplicate_pairs = pl.DataFrame(
        pairs,
        schema={"qm9_id_a": pl.UInt32, "qm9_id_b": pl.UInt32, "max_distance_diff_ang": pl.Float64},
        orient="row",
    ).sort(["qm9_id_a", "qm9_id_b"])
    return GeometryProfile(per_molecule=per_molecule, duplicate_pairs=duplicate_pairs)
