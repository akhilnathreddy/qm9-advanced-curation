"""Coulomb-matrix representations (Rupp et al., 2012), in atomic units.

M_ii = 0.5 Z_i^2.4, M_ij = Z_i Z_j / |R_i - R_j| with distances in Bohr.
Sorted eigenvalues are materialized (29 values, zero-padded). Bag-of-Bonds is
provided as a function and computed on demand: as a dense table it would be
~1,000 columns per molecule for little benefit over recomputing it.
"""

from collections import defaultdict

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qm9_curation.features.inputs import require_allowed
from qm9_curation.profiling.geometry import batches, coulomb_eigenvalues, pairwise_distances
from qm9_curation.units import BOHR_TO_ANGSTROM

MAX_ATOMS = 29


def coulomb_eigen_features(atoms: pl.DataFrame, max_atoms: int = MAX_ATOMS) -> pl.DataFrame:
    require_allowed(atoms)
    frames = []
    for batch in batches(atoms):
        d_bohr = pairwise_distances(batch.xyz) / BOHR_TO_ANGSTROM
        eig = coulomb_eigenvalues(batch.z, d_bohr)
        padded = np.zeros((eig.shape[0], max_atoms))
        padded[:, : eig.shape[1]] = eig
        frames.append(
            pl.DataFrame(
                {
                    "qm9_id": batch.qm9_ids.astype(np.uint32),
                    **{f"cm_eig_{k:02d}": padded[:, k] for k in range(max_atoms)},
                }
            )
        )
    return pl.concat(frames).sort("qm9_id")


def bag_of_bonds(
    z: NDArray[np.float64], xyz_ang: NDArray[np.float64], bag_sizes: dict[tuple[int, int], int]
) -> NDArray[np.float64]:
    """Bag-of-Bonds vector for one molecule (Hansen et al., 2015).

    ``bag_sizes`` fixes the length of each bag: ``(0, Z)`` holds the self terms
    of element Z, ``(Z_a, Z_b)`` with a <= b the pair terms. Entries are sorted
    in descending order within a bag and zero-padded.
    """
    d = np.linalg.norm(xyz_ang[:, None] - xyz_ang[None], axis=-1) / BOHR_TO_ANGSTROM
    bags: dict[tuple[int, int], list[float]] = defaultdict(list)
    n = len(z)
    for i in range(n):
        bags[(0, int(z[i]))].append(0.5 * z[i] ** 2.4)
        for j in range(i + 1, n):
            key = (int(min(z[i], z[j])), int(max(z[i], z[j])))
            bags[key].append(z[i] * z[j] / d[i, j])
    parts = []
    for key in sorted(bag_sizes):
        values = sorted(bags.get(key, []), reverse=True)
        if len(values) > bag_sizes[key]:
            raise ValueError(f"bag {key} has {len(values)} entries, size is {bag_sizes[key]}")
        parts.append(np.pad(np.asarray(values, dtype=float), (0, bag_sizes[key] - len(values))))
    return np.concatenate(parts)
