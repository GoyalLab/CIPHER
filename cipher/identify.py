"""Perturbation / drug **identification** — which perturbation a held-out replicate is.

The inverse problem (:mod:`cipher.inverse`) ranks *genes* as the driver of a single
observed shift.  Identification is a different task: given a *train* mean shift and a
*held-out test* mean shift per perturbation, decide **which perturbation** a test
replicate belongs to, among all candidates.

For each candidate ``j`` CIPHER fits the signed intervention vector ``u*_j`` from its
train shift (the ridge/posterior estimate, :func:`cipher.recover_u`) and predicts its
shift ``Pred_j = Sigma0 @ u*_j``.  A test query ``i`` is scored by the Gaussian
log-likelihood of its shift under every candidate, using the per-eigenmode sampling
variance ``h`` (shared across candidates, so the log-determinant cancels):

    LL(i, j) = -0.5 * sum_k (Pred_j[k] - (dx_test_i · V)[k])**2 / h_test_i[k]

Candidates are ranked by ``LL``; the top-1 is the predicted label, and
``margin = LL_top1 - LL_top2`` is a per-query confidence.  This drives top-k accuracy
and a correctness-vs-margin ROC/PR calibration curve.  ``Sigma0`` is the control
covariance (use a shuffled/mean-field null for an ablation).

This works on chemical screens too (drug identification): the labels are compounds and
no gene targets are needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics as _metrics
from .inverse import recover_u


@dataclass
class IdentificationResult:
    """Per-query identification outcome plus top-k accuracy and margin calibration."""
    labels: np.ndarray            # candidate labels (length n_candidates)
    y_pred: np.ndarray            # predicted label per query
    rank: np.ndarray              # 1-based rank of the true label per query
    margin: np.ndarray            # LL_top1 - LL_top2 confidence per query
    acc1: float
    topk_acc: dict
    roc: tuple | None             # correctness-vs-margin ROC (fpr, tpr)
    prc: tuple | None             # correctness-vs-margin PR (precision, recall)
    method: str
    dataset_name: str = ""

    @property
    def n_queries(self) -> int:
        return len(self.rank)

    def __repr__(self) -> str:
        return (f"IdentificationResult(dataset={self.dataset_name!r}, method={self.method!r}, "
                f"n={self.n_queries}, acc1={self.acc1:.3f}, "
                f"top5={self.topk_acc.get(5, float('nan')):.3f})")


def _finish(labels, rank, margin, y_pred, method, dataset_name, topk):
    rank = np.asarray(rank, dtype=np.int64)
    acc1 = float(np.mean(rank <= 1)) if rank.size else float("nan")
    topk_acc = {int(k): (float(np.mean(rank <= k)) if rank.size else float("nan")) for k in topk}
    correct = (rank <= 1).astype(int)
    roc = prc = None
    m = np.asarray(margin, dtype=np.float64)
    fin = np.isfinite(m)
    if fin.sum() >= 2 and 0 < correct[fin].sum() < fin.sum():
        roc = _metrics.roc_curve(correct[fin], m[fin])
        prc = _metrics.pr_curve(correct[fin], m[fin])
    return IdentificationResult(labels=np.asarray(labels), y_pred=np.asarray(y_pred),
                                rank=rank, margin=m, acc1=acc1, topk_acc=topk_acc,
                                roc=roc, prc=prc, method=method, dataset_name=dataset_name)


def _resolve_true_index(labels, dx_train, dx_test, true_index):
    n_c, n_q = dx_train.shape[0], dx_test.shape[0]
    labels = np.arange(n_c) if labels is None else np.asarray(labels)
    if true_index is None:
        if n_q != n_c:
            raise ValueError("Pass true_index when dx_test rows don't align 1:1 with dx_train "
                             "(candidates).")
        true_index = np.arange(n_q)
    return labels, np.asarray(true_index, dtype=np.int64)


def identify_perturbations(model, dx_train, dx_test, tau2=1.0, *, labels=None,
                           true_index=None, pert_eigvar_test=None, nu_test=None,
                           topk=(1, 3, 5, 10), dataset_name=""):
    """Identify which candidate perturbation each held-out ``dx_test`` replicate belongs to.

    Parameters
    ----------
    model : cipher.PosteriorInverseModel
        Built from the control covariance (:func:`cipher.build_model`); supplies the
        eigenbasis ``V``/``lambda``, control count ``n0`` and the candidates' sampling
        variances.
    dx_train, dx_test : ndarray (n_candidates, p)
        Train and held-out mean shifts, one row per perturbation (same perturbation per
        row unless ``true_index`` is given).
    tau2 : float
        Prior variance for the ``u*`` fit (``1/ridge``); ``1.0`` matches the notebooks.
    pert_eigvar_test, nu_test : optional
        Test-split projected variances / cell counts for ``h_test`` (default: the model's).

    Notes
    -----
    Cost is ``O(n_candidates**2 * p)``; fine for hundreds of perturbations, heavier for
    thousands.
    """
    V, lam = model.V, model.eigenvalues
    dx_train = np.asarray(dx_train, dtype=np.float64)
    dx_test = np.asarray(dx_test, dtype=np.float64)
    labels, true_index = _resolve_true_index(labels, dx_train, dx_test, true_index)
    n_q = dx_test.shape[0]

    u_hat = recover_u(model, dx_train, tau2)          # (n_c, p) signed u*
    pred_eig = (u_hat @ V) * lam[None, :]             # (n_c, K) predicted dx in eigenbasis
    pev = model.pert_eigvar if pert_eigvar_test is None else np.asarray(pert_eigvar_test, np.float64)
    nu = model.nu if nu_test is None else np.asarray(nu_test, np.float64).reshape(-1)
    h_test = np.maximum(lam[None, :] / model.n0 + pev / nu[:, None], model.ridge)  # (n_q, K)
    z_raw = dx_test @ V                               # (n_q, K)

    rank = np.empty(n_q, dtype=np.int64)
    margin = np.empty(n_q, dtype=np.float64)
    y_pred = np.empty(n_q, dtype=labels.dtype)
    n_c = pred_eig.shape[0]
    for i in range(n_q):
        diff = pred_eig - z_raw[i][None, :]
        lls = -0.5 * np.sum(diff * diff / h_test[i][None, :], axis=1)   # (n_c,)
        order = np.argsort(-lls, kind="stable")
        y_pred[i] = labels[order[0]]
        rank[i] = int(np.where(order == true_index[i])[0][0]) + 1
        margin[i] = float(lls[order[0]] - lls[order[1]]) if n_c > 1 else np.nan
    return _finish(labels, rank, margin, y_pred, "likelihood", dataset_name, topk)


def identify_lfc(lfc_train, lfc_test, *, labels=None, true_index=None,
                 topk=(1, 3, 5, 10), dataset_name=""):
    """Cosine-similarity identification baseline on log-fold-change rows.

    Ranks candidates by the cosine similarity of the query's ``lfc_test`` row to each
    candidate's ``lfc_train`` row; ``margin`` is the top-1 minus top-2 cosine.
    """
    A = np.asarray(lfc_train, dtype=np.float64)
    B = np.asarray(lfc_test, dtype=np.float64)
    labels, true_index = _resolve_true_index(labels, A, B, true_index)

    def _unit(M):
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.maximum(n, 1e-12)

    S = _unit(B) @ _unit(A).T                          # (n_q, n_c) cosine
    n_q, n_c = S.shape
    rank = np.empty(n_q, dtype=np.int64)
    margin = np.empty(n_q, dtype=np.float64)
    y_pred = np.empty(n_q, dtype=labels.dtype)
    for i in range(n_q):
        order = np.argsort(-S[i], kind="stable")
        y_pred[i] = labels[order[0]]
        rank[i] = int(np.where(order == true_index[i])[0][0]) + 1
        margin[i] = float(S[i, order[0]] - S[i, order[1]]) if n_c > 1 else np.nan
    return _finish(labels, rank, margin, y_pred, "lfc_cosine", dataset_name, topk)
