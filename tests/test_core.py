"""Unit tests for the core CIPHER math and normalizations.

These exercise the pure-numpy building blocks on small, hand-constructed inputs
where the correct answer is known analytically -- no dataset, no pymc, fast.
"""
from __future__ import annotations

import numpy as np
import pytest

from cipher.core import (
    forward_predict,
    forward_fit,
    forward_metrics,
    gene_holdout_masks,
    reverse_scores,
    reverse_operator,
    matched_filter_scores,
    rank_of,
    top_k_hit,
    one_vs_rest_auc,
)
from cipher.normalize import normalize_matrix, fit_pflog_alpha


def _spd_sigma(p: int = 30, seed: int = 1) -> np.ndarray:
    """A well-conditioned symmetric positive-definite covariance-like matrix."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(p, p))
    return (A @ A.T) / p + 0.1 * np.eye(p)


# --------------------------------------------------------------------------- #
# forward problem
# --------------------------------------------------------------------------- #
def test_forward_predict_recovers_a_hat():
    """On a rank-1 shift ``delta_x = a * Sigma[:, g]`` the optimum recovers ``a``."""
    Sigma = _spd_sigma(20, seed=0)
    g, a_true = 7, 1.37
    delta_x = a_true * Sigma[:, g]
    pred, a_hat = forward_predict(Sigma, delta_x, g)
    assert a_hat == pytest.approx(a_true, rel=1e-4)
    np.testing.assert_allclose(pred, delta_x, rtol=1e-4, atol=1e-6)


def test_forward_fit_uses_train_mask_only():
    """``forward_fit`` recovers the scalar even when the target gene is excluded."""
    Sigma = _spd_sigma(20, seed=0)
    g = 3
    col = Sigma[:, g]
    dx = 2.0 * col
    mask = np.ones(20, dtype=bool)
    mask[g] = False
    a_hat, denom = forward_fit(col, dx, mask=mask)
    assert a_hat == pytest.approx(2.0, rel=1e-6)
    assert denom > 0


def test_forward_metrics_perfect_prediction():
    """A perfect prediction gives every correlation/R² == 1 and zero error."""
    y = np.random.default_rng(2).normal(size=25)
    m = forward_metrics(y, y)
    for k in ("r2_uncentered", "r2_centered", "pearson", "spearman", "cosine",
              "sign_accuracy"):
        assert m[k] == pytest.approx(1.0, abs=1e-6)
    assert m["mse"] == pytest.approx(0.0, abs=1e-12)
    assert m["mae"] == pytest.approx(0.0, abs=1e-12)


def test_forward_metrics_uncentered_vs_centered():
    """Uncentered and centered R² use different denominators (sum y² vs var)."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y + 0.5   # constant bias -> SSE = 5*0.25 = 1.25
    m = forward_metrics(y, y_pred)
    assert m["r2_centered"] == pytest.approx(1.0 - 1.25 / 10.0, abs=1e-9)
    assert m["r2_uncentered"] == pytest.approx(1.0 - 1.25 / 55.0, abs=1e-9)


def test_gene_holdout_masks():
    tr, te = gene_holdout_masks(100, holdout_frac=0.0)
    assert tr.all() and te.all()
    tr, te = gene_holdout_masks(100, holdout_frac=0.3, seed=0)
    assert te.sum() == 30 and tr.sum() == 70
    assert not np.any(tr & te)                    # train/test disjoint
    tr, te = gene_holdout_masks(100, target_idx=5, holdout_frac=0.0,
                                exclude_target_fit=True, exclude_target_eval=True)
    assert not tr[5] and not te[5]


# --------------------------------------------------------------------------- #
# reverse problem
# --------------------------------------------------------------------------- #
def test_reverse_ranks_true_gene_top():
    """With ``delta_x = Sigma @ e_g`` every inverse solver must rank ``g`` first."""
    Sigma = _spd_sigma(30, seed=3)
    g = 11
    delta_x = Sigma[:, g]
    for method in ("pinv", "ridge", "lstsq"):
        scores = reverse_scores(Sigma, delta_x, method=method, ridge=1e-6)
        assert rank_of(scores, g) == 0
        assert int(np.argmax(np.abs(scores))) == g


def test_reverse_operator_matches_reverse_scores():
    """The reusable operator returns the same scores as the functional form."""
    Sigma = _spd_sigma(28, seed=4)
    delta_x = Sigma[:, 5]
    op = reverse_operator(Sigma, method="pinv")
    np.testing.assert_allclose(op(delta_x), reverse_scores(Sigma, delta_x, "pinv"),
                               atol=1e-8)


def test_matched_filter_scores_rank_true_gene():
    """The matched filter (no inverse) also puts the true gene at the top."""
    Sigma = _spd_sigma(24, seed=5)
    g = 4
    delta_x = Sigma[:, g]
    scores = matched_filter_scores(Sigma, delta_x)
    assert scores.shape == (24,)
    assert rank_of(scores, g) < 3
    # dispatched through reverse_scores this is identical
    np.testing.assert_allclose(
        scores, reverse_scores(Sigma, delta_x, method="matched_filter"), atol=1e-10)


# --------------------------------------------------------------------------- #
# ranking helpers
# --------------------------------------------------------------------------- #
def test_one_vs_rest_auc_is_one_when_target_is_max():
    scores = np.array([0.1, -0.2, 0.05, -0.9, 0.3])   # |-0.9| is the strict max
    assert one_vs_rest_auc(scores, 3) == pytest.approx(1.0)


def test_one_vs_rest_auc_tie_handling():
    """Ties count as 0.5 (Mann-Whitney), matching sklearn.roc_auc_score exactly.

    |scores| = [0.5, 0.5, 0.1], target=0 -> negatives {0.5, 0.1}: one strictly
    below (win) and one tie -> AUC = (1 + 0.5) / 2 = 0.75.
    """
    scores = np.array([0.5, -0.5, 0.1])
    assert one_vs_rest_auc(scores, 0) == pytest.approx(0.75)
    # all-equal scores -> pure chance, AUC == 0.5
    assert one_vs_rest_auc(np.full(6, 2.0), 3) == pytest.approx(0.5)


def test_rank_of_and_top_k_hit():
    scores = np.array([0.1, -0.5, 0.3, 0.05])         # |.| order: idx 1,2,0,3
    assert rank_of(scores, 1) == 0
    assert rank_of(scores, 2) == 1
    assert rank_of(scores, 0) == 2
    assert rank_of(scores, 3) == 3
    assert top_k_hit(scores, 1, k=1)
    assert not top_k_hit(scores, 0, k=2)


# --------------------------------------------------------------------------- #
# normalizations
# --------------------------------------------------------------------------- #
def test_normalize_matrix_modes_preserve_shape(control_counts):
    X = control_counts[:60, :]
    for mode in ["raw", "log1p", "frequency", "libsize10k", "log1CP10k"]:
        Y = normalize_matrix(X, mode)
        assert Y.shape == X.shape
        assert np.all(np.isfinite(Y))
    # pflog needs a pseudocount from the dispersion fit
    _, pc, _, _ = fit_pflog_alpha(X)
    Yp = normalize_matrix(X, "pflog", pseudocount=pc)
    assert Yp.shape == X.shape
    assert np.all(np.isfinite(Yp))


def test_fit_pflog_alpha_positive(control_counts):
    alpha, pc, meanvar_df, _ = fit_pflog_alpha(control_counts)
    assert np.isfinite(alpha) and alpha > 0
    assert pc == pytest.approx(1.0 / (4.0 * alpha))
    assert "genewise_alpha" in meanvar_df.columns
    assert len(meanvar_df) == control_counts.shape[1]
