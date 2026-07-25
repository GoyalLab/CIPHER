"""Tests for the identification-recipe capabilities: convex shrink, within-group
covariance, HVG dispersion selection, and build_model h_mode (sigma_count / within_cov)."""
from __future__ import annotations

import numpy as np
import pytest

import cipher
from cipher import (compute_covariance, within_group_covariance, select_hvg_dispersion,
                    build_model, project_diag_cov, identification_eigvar, recover_u)


def test_compute_covariance_convex_shrink():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 15))
    S = compute_covariance(X)
    Ssh = compute_covariance(X, shrink=0.1)
    d = float(np.mean(np.diag(S)))
    np.testing.assert_allclose(Ssh, 0.9 * S + 0.1 * d * np.eye(15), atol=1e-10)
    # convex shrinkage keeps the trace (unlike an additive ridge)
    np.testing.assert_allclose(np.trace(S), np.trace(Ssh), rtol=1e-10)


def test_within_group_covariance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 12))
    W = within_group_covariance(X, shrink=0.05)
    assert W.shape == (12, 12)
    np.testing.assert_allclose(W, compute_covariance(X, shrink=0.05))
    assert within_group_covariance(X[:1]) is None      # <2 cells -> None


def test_select_hvg_dispersion_picks_high_variance_and_forces_targets():
    rng = np.random.default_rng(2)
    # count-like data (positive means) so var/mean dispersion is meaningful
    X = rng.poisson(5.0, (400, 20)).astype(float)      # near-Poisson => dispersion ~1
    X[:, :5] += rng.gamma(2.0, 6.0, (400, 5))          # genes 0..4 overdispersed
    hvg = select_hvg_dispersion(X, 5)
    assert set(hvg) == set(range(5))                    # the 5 overdispersed genes
    # a low-dispersion target gene is force-included and the result stays sorted
    forced = select_hvg_dispersion(X, 5, force_include=[19])
    assert 19 in forced
    assert list(forced) == sorted(forced)


def _psd(p, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(p, p))
    return A @ A.T / p + 0.2 * np.eye(p), rng


def test_project_diag_cov_matches_full_projection():
    Sigma, rng = _psd(10, 3)
    _, V = np.linalg.eigh(Sigma)
    cov = _psd(10, 4)[0]
    np.testing.assert_allclose(project_diag_cov(V, cov), np.diag(V.T @ cov @ V), atol=1e-10)
    stack = np.stack([_psd(10, s)[0] for s in range(3)])
    got = project_diag_cov(V, stack)
    assert got.shape == (3, 10)
    for i in range(3):
        np.testing.assert_allclose(got[i], np.diag(V.T @ stack[i] @ V), atol=1e-10)


def test_build_model_sigma_count_uses_eigenvalues():
    Sigma, rng = _psd(12, 5)
    nu = np.full(4, 80.0)
    m = build_model(Sigma, None, n0=200.0, nu=nu, h_mode="sigma_count")
    # H noise per mode is exactly the control eigenvalue (var_pert ignored)
    np.testing.assert_allclose(m.pert_eigvar, np.broadcast_to(m.eigenvalues[None, :], (4, 12)))
    # test-side helper matches
    np.testing.assert_allclose(identification_eigvar(m, "sigma_count", n_query=3),
                               np.broadcast_to(m.eigenvalues[None, :], (3, 12)))


def test_build_model_within_cov_projects_full_covariance():
    Sigma, rng = _psd(10, 6)
    covs = np.stack([within_group_covariance(rng.normal(size=(50, 10)), 0.05) for _ in range(4)])
    m = build_model(Sigma, None, n0=150.0, nu=np.full(4, 50.0), h_mode="within_cov", pert_cov=covs)
    np.testing.assert_allclose(m.pert_eigvar, np.maximum(project_diag_cov(m.V, covs), 0.0))
    with pytest.raises(ValueError):
        build_model(Sigma, None, n0=150.0, nu=np.full(4, 50.0), h_mode="within_cov")  # no pert_cov


def test_sigma_count_identifies_and_recovers_u():
    """sigma_count identification still recovers distinct single-gene perturbations."""
    Sigma, rng = _psd(40, 7)
    genes = rng.choice(40, size=12, replace=False)
    dx_train = np.vstack([3.0 * Sigma[:, g] + rng.normal(0, 0.05, 40) for g in genes])
    dx_test = np.vstack([3.0 * Sigma[:, g] + rng.normal(0, 0.05, 40) for g in genes])
    m = build_model(Sigma, None, n0=200.0, nu=np.full(12, 60.0), h_mode="sigma_count")
    res = cipher.identify_perturbations(
        m, dx_train, dx_test, tau2=1.0, labels=[f"P{g}" for g in genes],
        pert_eigvar_test=identification_eigvar(m, "sigma_count", n_query=12),
        nu_test=np.full(12, 60.0))
    assert res.acc1 > 0.6
    assert recover_u(m, dx_train, 1.0).shape == dx_train.shape
