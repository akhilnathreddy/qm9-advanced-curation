"""ECFP4 bit fingerprints: Tanimoto similarity and leader clustering.

Similarities are computed with sparse matrix products (Tanimoto = c / (a + b - c)
on binary vectors), in fixed-size blocks, so results are deterministic and
memory stays bounded.
"""

import numpy as np
import polars as pl
from rdkit import DataStructs
from rdkit.SimDivFilters import rdSimDivPickers
from scipy import sparse

from qm9_curation.features.molecular import ECFP_SIZE

BLOCK = 2000


def bit_matrix(ecfp: pl.DataFrame) -> sparse.csr_matrix:
    """Binary CSR matrix (rows in ``ecfp`` order) from the sparse ECFP4 table."""
    lengths = ecfp["bits"].list.len().to_numpy()
    rows = np.repeat(np.arange(ecfp.height), lengths)
    cols = ecfp["bits"].explode().to_numpy().astype(np.int64)
    data = np.ones(len(cols), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(ecfp.height, ECFP_SIZE))


def tanimoto_block(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    common = (a @ b.T).toarray()
    na = np.asarray(a.sum(axis=1)).ravel()[:, None]
    nb = np.asarray(b.sum(axis=1)).ravel()[None, :]
    union = na + nb - common
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(union > 0, common / union, 0.0)
    return np.asarray(sim, dtype=np.float64)


def max_similarity(query: sparse.csr_matrix, reference: sparse.csr_matrix) -> np.ndarray:
    """For each query row, the largest Tanimoto similarity to any reference row."""
    out = np.empty(query.shape[0])
    for start in range(0, query.shape[0], BLOCK // 10):
        stop = start + BLOCK // 10
        out[start:stop] = tanimoto_block(query[start:stop], reference).max(axis=1)
    return out


def nearest(query: sparse.csr_matrix, reference: sparse.csr_matrix) -> np.ndarray:
    """Index of the most similar reference row (lowest index on ties)."""
    out = np.empty(query.shape[0], dtype=np.int64)
    for start in range(0, query.shape[0], BLOCK):
        out[start : start + BLOCK] = tanimoto_block(query[start : start + BLOCK], reference).argmax(
            axis=1
        )
    return out


def leader_clusters(ecfp: pl.DataFrame, distance_threshold: float) -> pl.DataFrame:
    """Sphere-exclusion leaders (RDKit LeaderPicker), then each molecule to its nearest leader.

    Rows are taken in ``qm9_id`` order, so the leaders are deterministic. The
    cluster id is the leader's ``qm9_id``.
    """
    ecfp = ecfp.sort("qm9_id")
    fps = []
    for bits in ecfp["bits"].to_list():
        bv = DataStructs.ExplicitBitVect(ECFP_SIZE)
        bv.SetBitsFromList(list(bits))
        fps.append(bv)
    picker = rdSimDivPickers.LeaderPicker()
    leaders = sorted(picker.LazyBitVectorPick(fps, len(fps), distance_threshold))
    mat = bit_matrix(ecfp)
    assigned = np.asarray(leaders)[nearest(mat, mat[leaders])]
    ids = ecfp["qm9_id"].to_numpy()
    return pl.DataFrame(
        {"qm9_id": ids, "cluster_id": ids[assigned]},
        schema={"qm9_id": pl.UInt32, "cluster_id": pl.UInt32},
    )
