"""Covariance estimation and null models.

CIPHER represents the control cell state as a gene-gene covariance ``Sigma``.
The forward/reverse problems are then linear in ``Sigma``.  Null covariances
(mean-field, fully shuffled, ZINB) provide baselines that destroy the real
covariance structure while preserving marginal statistics.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom
from scipy.sparse import issparse

from .utils import to_dense


def compute_covariance(X, ridge_abs: float = 0.0, ridge_rel: float = 0.0) -> np.ndarray:
    """Gene-gene covariance of a (cells x genes) matrix, optionally ridged.

    ``ridge_abs`` and ``ridge_rel`` add ``ridge_abs + ridge_rel * mean_diag`` to
    the diagonal, which stabilises the inverse used by the reverse problem.
    """
    X = to_dense(X).astype(np.float64, copy=False)
    Sigma = np.cov(X, rowvar=False)
    Sigma = np.atleast_2d(Sigma)
    if ridge_abs or ridge_rel:
        diag = np.diag(Sigma)
        good = np.isfinite(diag) & (diag > 0)
        scale = float(np.mean(diag[good])) if np.any(good) else 1.0
        Sigma = Sigma + (ridge_abs + ridge_rel * scale) * np.eye(Sigma.shape[0])
    return Sigma


def meanfield_covariance(X, seed: int = 0) -> np.ndarray:
    """Null Sigma: shuffle each gene independently across cells (kills covariance,
    keeps per-gene marginals)."""
    rng = np.random.default_rng(seed)
    Xs = to_dense(X).astype(np.float64, copy=True)
    for g in range(Xs.shape[1]):
        rng.shuffle(Xs[:, g])
    return compute_covariance(Xs)


def shuffled_covariance(X, seed: int = 0) -> np.ndarray:
    """Null Sigma: fully permute every entry of ``X`` (keeps only the global
    value distribution)."""
    rng = np.random.default_rng(seed)
    Xd = to_dense(X).astype(np.float64, copy=False)
    flat = rng.permutation(Xd.reshape(-1))
    return compute_covariance(flat.reshape(Xd.shape))


def _sample_zinb(mean, var, zero_prob, size, rng):
    if var <= mean:
        var = mean + 1e-3
    p = mean / var
    r = mean ** 2 / (var - mean)
    nb = nbinom.rvs(n=r, p=p, size=size, random_state=rng)
    nb = nb.astype(np.float64)
    nb[rng.random(size) < zero_prob] = 0
    return nb


def zinb_covariance(X, n_cells: int | None = None, seed: int = 42) -> np.ndarray:
    """Null Sigma from a zero-inflated negative-binomial resample of ``X`` with
    randomized per-gene variance scaling (as in the preprint's ZINB null)."""
    rng = np.random.default_rng(seed)
    Xd = to_dense(X).astype(np.float64, copy=False)
    n_obs, n_genes = Xd.shape
    n_cells = n_obs if n_cells is None else n_cells
    mu = Xd.mean(axis=0)
    var_emp = Xd.var(axis=0)
    zero_prob = (Xd == 0).mean(axis=0)
    n_half = n_genes // 2
    combined = np.concatenate([rng.uniform(-1, -0.5, n_half),
                               rng.uniform(0.5, 1, n_genes - n_half)])
    rng.shuffle(combined)
    target_var = (10 ** combined) * var_emp
    synth = np.zeros((n_cells, n_genes))
    for g in range(n_genes):
        synth[:, g] = _sample_zinb(mu[g], target_var[g], zero_prob[g], n_cells, rng)
    return compute_covariance(synth)


NULL_MODELS = {
    "meanfield": meanfield_covariance,
    "shuffled": shuffled_covariance,
    "zinb": zinb_covariance,
}


def null_covariance(X, kind: str, seed: int = 0) -> np.ndarray:
    """Dispatch to a named null covariance model (``meanfield``/``shuffled``/``zinb``)."""
    if kind not in NULL_MODELS:
        raise ValueError(f"Unknown null model {kind!r}; choose from {list(NULL_MODELS)}.")
    return NULL_MODELS[kind](X, seed=seed)
