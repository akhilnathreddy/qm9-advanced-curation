"""Baseline ladder (plan v2 §7.1): one ``fit_predict`` per level.

| Level | Name | Input |
|---|---|---|
| 0 | ``mean`` | — |
| 1 | ``dressed_atom`` | element counts (+ intercept), least squares |
| 2 | ``gbm`` | RDKit descriptors + ECFP4 counts, histogram gradient boosting |
| 3 | ``krr`` | sorted Coulomb-matrix eigenvalues, Laplacian-kernel ridge |
| 4 | ``dmpnn`` | Phase 6 molecular graph (Chemprop D-MPNN architecture) |
| 5 | ``schnet`` | coordinates (SchNet) |

Levels 2-5 learn the residual of level 1 fitted on the same training rows
(Δ-learning on the dressed-atom baseline), so every model sees the
stoichiometric part removed in the same way. Every hyperparameter choice uses
validation rows only; test rows are only predicted.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist

from qm9_curation.benchmark.data import Inputs

LEVELS = {"mean": 0, "dressed_atom": 1, "gbm": 2, "krr": 3, "dmpnn": 4, "schnet": 5}


@dataclass
class Fit:
    predictions: dict[str, np.ndarray]  # role -> predictions for the rows asked
    info: dict[str, Any] = field(default_factory=dict)


def _design(inputs: Inputs, rows: np.ndarray) -> np.ndarray:
    return np.column_stack([inputs.composition[rows], np.ones(len(rows))])


def dressed_atom(inputs: Inputs, train: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(_design(inputs, train), y, rcond=None)
    return np.asarray(coef)


def _gbm_matrix(inputs: Inputs, rows: np.ndarray, bits: np.ndarray) -> np.ndarray:
    fp = inputs.ecfp[rows][:, bits].toarray()
    return np.hstack([inputs.descriptors[rows], fp]).astype(np.float32)


def _gbm(
    inputs: Inputs,
    train: np.ndarray,
    r_train: np.ndarray,
    val: np.ndarray,
    r_val: np.ndarray,
    predict: Mapping[str, np.ndarray],
    params: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    freq = np.asarray((inputs.ecfp[train] > 0).mean(axis=0)).ravel()
    bits = np.flatnonzero(freq >= float(params["min_bit_frequency"]))
    model = HistGradientBoostingRegressor(
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=False,
        random_state=seed,
    )
    model.fit(_gbm_matrix(inputs, train, bits), r_train)
    # Number of boosting iterations chosen on validation (early stopping on val only).
    val_mae = [
        float(np.mean(np.abs(p - r_val)))
        for p in model.staged_predict(_gbm_matrix(inputs, val, bits))
    ]
    best = int(np.argmin(val_mae))
    out = {}
    for role, rows in predict.items():
        for k, p in enumerate(model.staged_predict(_gbm_matrix(inputs, rows, bits))):
            if k == best:
                out[role] = p
                break
    return out, {"n_bits": len(bits), "best_iteration": best + 1, "val_mae_curve_end": val_mae[-1]}


def laplacian_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    k: np.ndarray = np.exp(-cdist(a, b, metric="cityblock") / sigma)
    return k


def krr_scale(x: np.ndarray, seed: int, n: int = 2000) -> float:
    """Median pairwise L1 distance on a sample: the kernel-width unit."""
    rng = np.random.default_rng(seed)
    s = x[rng.choice(len(x), size=min(n, len(x)), replace=False)]
    d = cdist(s, s, metric="cityblock")
    return float(np.median(d[np.triu_indices(len(s), 1)]))


def krr_fit_predict(
    x_train: np.ndarray,
    r_train: np.ndarray,
    predict: Mapping[str, np.ndarray],
    sigma: float,
    lam: float,
    block: int = 4096,
) -> dict[str, np.ndarray]:
    k = laplacian_kernel(x_train, x_train, sigma)
    k[np.diag_indices_from(k)] += lam
    mu = float(r_train.mean())
    alpha = linalg.cho_solve(linalg.cho_factor(k, overwrite_a=True), r_train - mu)
    del k
    out = {}
    for role, x in predict.items():
        out[role] = np.concatenate(
            [
                laplacian_kernel(x[i : i + block], x_train, sigma) @ alpha + mu
                for i in range(0, len(x), block)
            ]
        )
    return out


def _krr(
    inputs: Inputs,
    train: np.ndarray,
    r_train: np.ndarray,
    predict: Mapping[str, np.ndarray],
    params: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    info: dict[str, Any] = {"n_train_requested": len(train)}
    cap = int(params["max_train"])
    if len(train) > cap:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(train), size=cap, replace=False))
        train, r_train = train[keep], r_train[keep]
    x = inputs.coulomb
    sigma = float(params["sigma_factor"]) * krr_scale(x[train], seed)
    out = krr_fit_predict(
        x[train], r_train, {k: x[v] for k, v in predict.items()}, sigma, float(params["lambda"])
    )
    info.update({"n_train_used": len(train), "sigma": sigma})
    return out, info


def fit_predict(
    model: str,
    inputs: Inputs,
    target: str,
    train: np.ndarray,
    val: np.ndarray,
    predict: Mapping[str, np.ndarray],
    params: Mapping[str, Any],
    seed: int,
    packed: Any = None,
    mean_readout: bool = False,
) -> Fit:
    """Train on ``train`` rows, select on ``val`` rows, predict every role in ``predict``."""
    y = inputs.frame[target].to_numpy().astype(np.float64)
    if model == "mean":
        mu = float(y[train].mean())
        return Fit({role: np.full(len(rows), mu) for role, rows in predict.items()})
    coef = dressed_atom(inputs, train, y[train])
    base = {role: _design(inputs, rows) @ coef for role, rows in predict.items()}
    if model == "dressed_atom":
        return Fit(base, {"coefficients": coef.tolist()})
    r = y - _design(inputs, np.arange(len(y))) @ coef
    if model == "gbm":
        resid, info = _gbm(inputs, train, r[train], val, r[val], predict, params, seed)
    elif model == "krr":
        resid, info = _krr(inputs, train, r[train], predict, params, seed)
    elif model in ("dmpnn", "schnet"):
        from qm9_curation.benchmark import gnn

        ids = inputs.ids
        res = gnn.train(
            packed,
            packed.index(ids[train]),
            r[train],
            packed.index(ids[val]),
            r[val],
            {**params, "model": model},
            mean_readout,
            seed,
        )
        resid = {role: res.predict(packed.index(ids[rows])) for role, rows in predict.items()}
        info = {
            "best_epoch": res.best_epoch,
            "epochs": len(res.history),
            "seconds_training": res.seconds,
            "device": res.device,
            "history": res.history,
        }
    else:
        raise ValueError(f"unknown model {model}")
    return Fit({role: base[role] + resid[role] for role in predict}, info)
