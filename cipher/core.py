"""Core CIPHER math: the forward projection and the reverse (driver) inference.

Both problems share one linear model of the mean perturbation response::

    delta_x  ~=  Sigma @ u

where ``Sigma`` is the control covariance and ``u`` is a (sparse) vector of
perturbation strengths per gene.

* **Forward** — the perturbed gene ``g`` is known, so ``u`` is supported on ``g``
  alone and the predicted shift is a rank-1 projection onto column ``Sigma[:, g]``.
* **Reverse** — ``delta_x`` is observed and we solve for ``u`` to rank candidate
  driver genes.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import pinv

EPS = 1e-8


# --------------------------------------------------------------------------- #
# forward problem
# --------------------------------------------------------------------------- #
def forward_predict(Sigma, delta_x, gene_idx, eps: float = EPS):
    """Best rank-1 reconstruction of ``delta_x`` from column ``gene_idx``.

    Returns ``(prediction, u_opt)`` where ``prediction = u_opt * Sigma[:, gene_idx]``
    and ``u_opt`` minimises ``||delta_x - u * Sigma[:, g]||``.
    """
    sigma_col = np.asarray(Sigma[:, gene_idx], dtype=np.float64)
    delta_x = np.asarray(delta_x, dtype=np.float64)
    denom = float(np.dot(sigma_col, sigma_col)) + eps
    u_opt = float(np.dot(sigma_col, delta_x) / denom)
    return u_opt * sigma_col, u_opt


def forward_metrics(delta_x, prediction, mean_pert=None, eps: float = EPS) -> dict:
    """Goodness-of-fit metrics between observed ``delta_x`` and ``prediction``.

    * ``R2``  — variance of the shift explained (denominator ``sum(delta_x**2)``)
    * ``R20`` — explained relative to the perturbed mean (``sum(mean_pert**2)``)
    * ``Spearman`` / ``Pearson`` — rank / linear correlation
    """
    from scipy.stats import spearmanr, pearsonr

    delta_x = np.asarray(delta_x, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.abs(delta_x) > 0
    if not np.any(valid):
        return {"R2": np.nan, "R20": np.nan, "Spearman": np.nan, "Pearson": np.nan}
    resid = np.sum((delta_x[valid] - prediction[valid]) ** 2)
    r2 = 1.0 - resid / (np.sum(delta_x[valid] ** 2) + eps)
    if mean_pert is not None:
        mean_pert = np.asarray(mean_pert, dtype=np.float64)
        r20 = 1.0 - resid / (np.sum(mean_pert[valid] ** 2) + eps)
    else:
        r20 = np.nan
    if np.std(prediction[valid]) < eps:
        spear = pear = np.nan
    else:
        spear = spearmanr(delta_x[valid], prediction[valid]).statistic
        pear = pearsonr(delta_x[valid], prediction[valid]).statistic
    return {"R2": float(r2), "R20": float(r20) if mean_pert is not None else np.nan,
            "Spearman": float(spear) if np.isfinite(spear) else np.nan,
            "Pearson": float(pear) if np.isfinite(pear) else np.nan}


# --------------------------------------------------------------------------- #
# reverse problem (driver inference)
# --------------------------------------------------------------------------- #
def matched_filter_scores(Sigma, delta_x, eps: float = EPS) -> np.ndarray:
    """Per-gene ``u_opt`` from the forward model — how strongly each single-gene
    perturbation would reproduce ``delta_x``.  No matrix inverse required, so it
    is robust for ill-conditioned ``Sigma``."""
    Sigma = np.asarray(Sigma, dtype=np.float64)
    delta_x = np.asarray(delta_x, dtype=np.float64)
    num = Sigma.T @ delta_x
    den = np.einsum("ij,ij->j", Sigma, Sigma) + eps
    return num / den


def reverse_operator(Sigma, method: str = "pinv", ridge: float = 1e-2, eps: float = EPS):
    """Build a reusable ``delta_x -> per-gene scores`` operator.

    The expensive factorization (inverse / decomposition) is done once so the
    same ``Sigma`` can score many perturbations cheaply.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    if method == "matched_filter":
        den = np.einsum("ij,ij->j", Sigma, Sigma) + eps
        SigmaT = Sigma.T
        return lambda dx: (SigmaT @ np.asarray(dx, np.float64)) / den
    if method == "pinv":
        A_inv = pinv(Sigma)
        return lambda dx: A_inv @ np.asarray(dx, np.float64)
    if method == "ridge":
        diag = np.diag(Sigma)
        good = np.isfinite(diag) & (diag > 0)
        scale = float(np.mean(diag[good])) if np.any(good) else 1.0
        A_inv = np.linalg.inv(Sigma + ridge * scale * np.eye(Sigma.shape[0]))
        return lambda dx: A_inv @ np.asarray(dx, np.float64)
    if method == "lstsq":
        # the minimum-norm least-squares solution equals pinv(Sigma) @ dx;
        # factorize once so scoring many perturbations stays cheap.
        A_inv = pinv(Sigma)
        return lambda dx: A_inv @ np.asarray(dx, np.float64)
    raise ValueError(f"Unknown reverse method {method!r}.")


def reverse_scores(Sigma, delta_x, method: str = "pinv", ridge: float = 1e-2,
                   eps: float = EPS) -> np.ndarray:
    """Score every gene as a candidate driver of ``delta_x``.

    Larger ``|score|`` == more likely the true driver.  Methods:

    * ``pinv``           — ``pinv(Sigma) @ delta_x`` (Moore-Penrose inverse)
    * ``ridge``          — ``(Sigma + ridge*mean_diag*I)^-1 @ delta_x``
    * ``lstsq``          — least-squares solve of ``Sigma u = delta_x``
    * ``matched_filter`` — per-column forward ``u_opt`` (see above)
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    delta_x = np.asarray(delta_x, dtype=np.float64)
    if method == "matched_filter":
        return matched_filter_scores(Sigma, delta_x, eps=eps)
    if method == "pinv":
        return pinv(Sigma) @ delta_x
    if method == "ridge":
        diag = np.diag(Sigma)
        good = np.isfinite(diag) & (diag > 0)
        scale = float(np.mean(diag[good])) if np.any(good) else 1.0
        A = Sigma + ridge * scale * np.eye(Sigma.shape[0])
        return np.linalg.solve(A, delta_x)
    if method == "lstsq":
        return np.linalg.lstsq(Sigma, delta_x, rcond=None)[0]
    raise ValueError(f"Unknown reverse method {method!r}.")


# --------------------------------------------------------------------------- #
# ranking / evaluation helpers
# --------------------------------------------------------------------------- #
def rank_of(scores, target_idx) -> int:
    """0-indexed rank of ``target_idx`` when sorting genes by descending ``|score|``
    (0 == top). Ties in ``|score|`` are broken by ascending gene index (stable order)."""
    s = np.abs(np.asarray(scores, dtype=np.float64))
    order = np.argsort(-s, kind="stable")
    return int(np.where(order == target_idx)[0][0])


def top_k_hit(scores, target_idx, k: int = 10) -> bool:
    return rank_of(scores, target_idx) < k


def one_vs_rest_auc(scores, target_idx) -> float:
    """ROC-AUC of ``|score|`` for identifying the single true driver among all genes."""
    from sklearn.metrics import roc_auc_score

    s = np.abs(np.asarray(scores, dtype=np.float64))
    n = s.size
    if n < 2:
        return np.nan
    y = np.zeros(n)
    y[target_idx] = 1
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return np.nan
