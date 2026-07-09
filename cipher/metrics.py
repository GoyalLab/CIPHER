"""Binary-classification metrics in pure numpy (ROC / PR curves, AUC, AP).

Used by the inverse-problem evaluation (:mod:`cipher.inverse`) to score how well a
per-gene driver score separates the true perturbed gene (positive) from the other
genes (negatives), pooled across perturbations.  Kept dependency-free (no
scikit-learn) and verified to match ``sklearn.metrics`` numerically.
"""
from __future__ import annotations

import numpy as np


def _binary_clf_curve(y_true, y_score):
    """Cumulative (fps, tps, thresholds) at each distinct score, high score first.

    Mirrors ``sklearn.metrics._binary_clf_curve``.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[order]
    y_score = y_score[order]
    distinct = np.where(np.diff(y_score))[0]
    threshold_idx = np.r_[distinct, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idx]
    fps = 1.0 + threshold_idx - tps
    return fps, tps, y_score[threshold_idx]


def roc_auc(y_true, y_score) -> float:
    """ROC-AUC via the tie-aware Mann-Whitney statistic (== ``roc_auc_score``)."""
    y_true = np.asarray(y_true).ravel().astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = int(np.count_nonzero(y_true))
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(y_score.size, dtype=np.float64)
    s = y_score[order]
    # average ranks (1-based) for ties
    ranks_sorted = np.arange(1, s.size + 1, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i + 1
        while j < s.size and s[j] == s[i]:
            j += 1
        ranks_sorted[i:j] = (i + 1 + j) / 2.0
        i = j
    ranks[order] = ranks_sorted
    sum_pos = float(np.sum(ranks[y_true]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve(y_true, y_score):
    """Return ``(fpr, tpr)`` arrays for plotting (with a leading (0, 0) point)."""
    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]
    P, N = tps[-1], fps[-1]
    if P <= 0 or N <= 0:
        return None
    return fps / N, tps / P


def pr_curve(y_true, y_score):
    """Return ``(precision, recall)`` arrays (recall descending, ending at recall 0)."""
    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    P = tps[-1]
    if P <= 0:
        return None
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / P
    # match sklearn.precision_recall_curve: reverse, and append the (recall=0, precision=1) point
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    return precision, recall


def average_precision(y_true, y_score) -> float:
    """Average precision ``sum_n (R_n - R_{n-1}) P_n`` (== ``average_precision_score``)."""
    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    P = tps[-1]
    if P <= 0:
        return float("nan")
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / P
    recall = np.r_[0.0, recall]           # R_0 = 0
    return float(np.sum(np.diff(recall) * precision))


def auc_trapezoid(x, y) -> float:
    """Trapezoidal area under a curve, ordering by ``x`` then ``y`` (matches ``sklearn.auc``).

    The secondary sort on ``y`` keeps tied-``x`` points in step order so the area is
    correct for staircase ROC/PR curves.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.lexsort((y, x))
    return float(np.trapz(y[order], x[order]))
