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
#: metric keys returned by :func:`forward_metrics`, in a stable order.
FORWARD_METRICS = (
    "r2_uncentered", "r2_centered", "pearson", "spearman", "cosine",
    "mse", "rmse", "mae", "sign_accuracy", "sign_accuracy_nonzero_true",
)


def forward_fit(sigma_col, delta_x, mask=None, eps: float = EPS):
    """Least-squares scalar ``a_hat`` for ``delta_x ~= a * sigma_col``.

    The fit uses only the genes in ``mask`` (all genes if ``mask`` is ``None``).
    Returns ``(a_hat, denom)`` where ``denom = <sigma_col, sigma_col>`` on the
    masked genes; ``a_hat`` is ``nan`` when ``denom`` is non-finite or ~0.
    """
    sigma_col = np.asarray(sigma_col, dtype=np.float64)
    delta_x = np.asarray(delta_x, dtype=np.float64)
    if mask is not None:
        sigma_col = sigma_col[mask]
        delta_x = delta_x[mask]
    denom = float(np.dot(sigma_col, sigma_col))
    if not np.isfinite(denom) or denom <= eps:
        return np.nan, denom
    return float(np.dot(sigma_col, delta_x) / denom), denom


def forward_predict(Sigma, delta_x, gene_idx, eps: float = EPS):
    """Best rank-1 reconstruction of ``delta_x`` from column ``gene_idx``.

    Returns ``(prediction, a_hat)`` where ``prediction = a_hat * Sigma[:, gene_idx]``
    and ``a_hat`` minimises ``||delta_x - a * Sigma[:, g]||`` over all genes.
    """
    sigma_col = np.asarray(Sigma[:, gene_idx], dtype=np.float64)
    a_hat, _ = forward_fit(sigma_col, delta_x, eps=eps)
    if not np.isfinite(a_hat):
        a_hat = 0.0
    return a_hat * sigma_col, a_hat


def gene_holdout_masks(n_genes, target_idx=None, holdout_frac=0.0, seed=0, rng=None,
                       exclude_target_fit=False, exclude_target_eval=False):
    """Boolean ``(train_mask, test_mask)`` over genes for out-of-sample forward fitting.

    With ``holdout_frac <= 0`` train == test == all genes.  Otherwise a random
    ``holdout_frac`` of genes forms the test (evaluation) set and the remainder the
    train (fit) set.  The target gene can optionally be excluded from the fit and/or
    the evaluation.
    """
    n_genes = int(n_genes)
    base_fit = np.ones(n_genes, dtype=bool)
    base_eval = np.ones(n_genes, dtype=bool)
    if target_idx is not None and 0 <= int(target_idx) < n_genes:
        if exclude_target_fit:
            base_fit[int(target_idx)] = False
        if exclude_target_eval:
            base_eval[int(target_idx)] = False
    holdout_frac = float(holdout_frac)
    if holdout_frac <= 0:
        return base_fit.copy(), base_eval.copy()
    if holdout_frac >= 1.0:
        raise ValueError("holdout_frac must be < 1.0")
    if rng is None:
        rng = np.random.default_rng(seed)
    n_test = max(1, int(round(holdout_frac * n_genes)))
    test_idx = rng.choice(np.arange(n_genes, dtype=np.int64), size=n_test, replace=False)
    test_mask = np.zeros(n_genes, dtype=bool)
    test_mask[test_idx] = True
    return (~test_mask) & base_fit, test_mask & base_eval


def _pearson(x, y):
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    from scipy.stats import rankdata
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return np.nan
    return _pearson(rankdata(x), rankdata(y))


def _cosine(x, y):
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return np.nan
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 0 else np.nan


def forward_metrics(y_true, y_pred, eps: float = EPS) -> dict:
    """Forward goodness-of-fit metrics between observed and predicted shift.

    Every metric is computed over the genes where both are finite.  Keys
    (:data:`FORWARD_METRICS`):

    * ``r2_uncentered`` — ``1 - SSE / sum(y_true**2)`` (fraction of raw shift power explained)
    * ``r2_centered``   — standard R² (``1 - SSE / sum((y_true - mean)**2)``)
    * ``pearson`` / ``spearman`` — linear / rank correlation
    * ``cosine`` — cosine similarity of the two shift vectors
    * ``mse`` / ``rmse`` / ``mae`` — error magnitudes
    * ``sign_accuracy`` — fraction of genes whose predicted shift sign matches
    * ``sign_accuracy_nonzero_true`` — same, restricted to genes with nonzero true shift
    """
    yt = np.asarray(y_true, np.float64).ravel()
    yp = np.asarray(y_pred, np.float64).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    out = {k: np.nan for k in FORWARD_METRICS}
    if yt.size == 0:
        return out
    err = yt - yp
    sse = float(np.sum(err ** 2))
    denom_u = float(np.sum(yt * yt))
    if denom_u > 0:
        out["r2_uncentered"] = 1.0 - sse / denom_u
    if yt.size >= 2:
        denom_c = float(np.sum((yt - np.mean(yt)) ** 2))
        if denom_c > 0:
            out["r2_centered"] = 1.0 - sse / denom_c
    out["pearson"] = _pearson(yt, yp)
    out["spearman"] = _spearman(yt, yp)
    out["cosine"] = _cosine(yt, yp)
    out["mse"] = float(np.mean(err ** 2))
    out["rmse"] = float(np.sqrt(out["mse"]))
    out["mae"] = float(np.mean(np.abs(err)))
    st, sp = np.sign(yt), np.sign(yp)
    out["sign_accuracy"] = float(np.mean(st == sp))
    nz = st != 0
    out["sign_accuracy_nonzero_true"] = float(np.mean(st[nz] == sp[nz])) if np.any(nz) else np.nan
    return out


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
