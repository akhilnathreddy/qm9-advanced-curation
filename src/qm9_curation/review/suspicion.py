"""ML-assisted suspicion score from out-of-fold ridge residuals.

A linear model on composition + ECFP4 counts is close to a bond-additivity
model, so a record far from its out-of-fold prediction is unusual for its
local structure. Folds keep every exact-structure group together
(``structure_group_id mod k``), so a duplicate never predicts its twin.
The score ranks records for review only; Phase 9 recomputes it with the
strongest benchmark model.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy import sparse

from qm9_curation.features.molecular import ECFP_SIZE
from qm9_curation.review.stats import ELEMENT_COUNTS, robust_z


def design_matrix(composition: pl.DataFrame, ecfp: pl.DataFrame) -> sparse.csr_matrix:
    """Rows in ``ecfp`` order: 5 element counts, then 2048 ECFP4 count columns."""
    comp = ecfp.select("qm9_id").join(composition, on="qm9_id", how="left")
    counts = comp.select(ELEMENT_COUNTS).to_numpy().astype(np.float64)
    lengths = ecfp["bits"].list.len().to_numpy()
    rows = np.repeat(np.arange(ecfp.height), lengths)
    cols = ecfp["bits"].explode().to_numpy().astype(np.int64) + len(ELEMENT_COUNTS)
    vals = ecfp["counts"].explode().to_numpy().astype(np.float64)
    fp = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(ecfp.height, len(ELEMENT_COUNTS) + ECFP_SIZE)
    )
    return (fp + sparse.csr_matrix(np.pad(counts, ((0, 0), (0, ECFP_SIZE))))).tocsr()


def ridge_oof(x: sparse.csr_matrix, y: np.ndarray, folds: np.ndarray, alpha: float) -> np.ndarray:
    """Out-of-fold predictions of a centred ridge regression (closed form)."""
    pred = np.empty_like(y)
    for k in np.unique(folds):
        train = folds != k
        xt = x[train]
        mu = np.asarray(xt.mean(axis=0)).ravel()
        ym = y[train].mean()
        n = xt.shape[0]
        gram = (xt.T @ xt).toarray() - n * np.outer(mu, mu)
        rhs = xt.T @ (y[train] - ym)
        w = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
        pred[~train] = (x[~train] @ w) - mu @ w + ym
    return pred


def suspicion_scores(
    curated: pl.DataFrame,
    composition: pl.DataFrame,
    ecfp: pl.DataFrame,
    targets: Sequence[str],
    n_folds: int,
    alpha: float,
) -> pl.DataFrame:
    """Per target: out-of-fold residual and its robust z; score = max |z| over targets."""
    ecfp = ecfp.sort("qm9_id")
    frame = ecfp.select("qm9_id").join(
        curated.select("qm9_id", "structure_group_id", *targets), on="qm9_id"
    )
    x = design_matrix(composition, ecfp)
    folds = (frame["structure_group_id"].to_numpy() % n_folds).astype(np.int64)
    cols = []
    for t in targets:
        y = frame[t].to_numpy().astype(np.float64)
        cols.append(pl.Series(f"oof_residual_{t}", y - ridge_oof(x, y, folds, alpha)))
    out = frame.select("qm9_id").with_columns(cols)
    out = out.with_columns(
        robust_z(pl.col(f"oof_residual_{t}")).alias(f"oof_robust_z_{t}") for t in targets
    )
    return out.with_columns(
        pl.max_horizontal(pl.col(f"oof_robust_z_{t}").abs() for t in targets).alias(
            "suspicion_score"
        )
    )
