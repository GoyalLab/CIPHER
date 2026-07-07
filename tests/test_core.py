"""Unit tests for the core CIPHER math and normalizations.

These exercise the pure-numpy building blocks on small, hand-constructed inputs
where the correct answer is known analytically -- no dataset, no pymc, fast.
"""
from __future__ import annotations

import numpy as np
import pytest

from cipher.core import (
    forward_predict,
    forward_metrics,
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
def test_forward_predict_recovers_u_opt():
    """On a rank-1 shift ``delta_x = u * Sigma[:, g]`` the optimum recovers ``u``."""
    Sigma = _spd_sigma(20, seed=0)
    g, u_true = 7, 1.37
    delta_x = u_true * Sigma[:, g]
    pred, u_opt = forward_predict(Sigma, delta_x, g)
    assert u_opt == pytest.approx(u_true, rel=1e-4)
    np.testing.assert_allclose(pred, delta_x, rtol=1e-4, atol=1e-6)


def test_forward_metrics_perfect_prediction():
    """A perfect prediction gives R2 == R20 == Pearson == 1."""
    delta_x = np.random.default_rng(2).normal(size=25)
    m = forward_metrics(delta_x, delta_x, mean_pert=delta_x)
    assert m["R2"] == pytest.approx(1.0, abs=1e-8)
    assert m["R20"] == pytest.approx(1.0, abs=1e-8)
    assert m["Pearson"] == pytest.approx(1.0, abs=1e-6)
    assert m["Spearman"] == pytest.approx(1.0, abs=1e-6)


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
