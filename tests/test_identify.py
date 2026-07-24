"""Tests for signed u* recovery and perturbation identification."""
from __future__ import annotations

import numpy as np
import pytest

import cipher


def _model(p=40, n_pert=12, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(p, p))
    Sigma = A @ A.T / p + 0.2 * np.eye(p)
    var_pert = np.abs(rng.normal(size=(n_pert, p))) + 0.2
    nu = np.full(n_pert, 60.0)
    return cipher.build_model(Sigma, var_pert, n0=200.0, nu=nu), Sigma, rng


def test_recover_u_matches_closed_form():
    """recover_u returns V @ [ d/(d^2+1/tau2) * z ] — the signed posterior mean."""
    model, Sigma, rng = _model()
    dx = rng.normal(size=(model.pert_eigvar.shape[0], model.V.shape[0]))
    tau2 = 1.0
    u = cipher.recover_u(model, dx, tau2)
    # rebuild by hand in the eigenbasis
    V, lam = model.V, model.eigenvalues
    h = np.maximum(lam[None, :] / model.n0 + model.pert_eigvar / model.nu[:, None], model.ridge)
    d = lam[None, :] / np.sqrt(h)
    z = (dx @ V) / np.sqrt(h)
    post_var = 1.0 / np.maximum(d * d + 1.0 / tau2, 1e-12)
    ref = (d * post_var * z) @ V.T
    np.testing.assert_allclose(u, ref, atol=1e-10)
    assert u.shape == dx.shape


def test_recover_u_signed_and_score_is_its_magnitude():
    """The posterior *score* is the uncertainty-inflated magnitude of the signed u*."""
    from cipher.inverse import posterior_scores_batch, posterior_mean_batch
    model, Sigma, rng = _model()
    dx = rng.normal(size=(3, model.V.shape[0]))
    u = posterior_mean_batch(model, dx, 0, 3, 1.0)
    score = posterior_scores_batch(model, dx, 0, 3, 1.0)
    assert np.any(u < 0) and np.all(score >= 0)          # u* is signed, score is a magnitude
    assert np.all(score + 1e-9 >= np.abs(u))             # inflated magnitude >= |mean|


def test_identify_perturbations_recovers_identity():
    """Distinct single-gene perturbations are identifiable from held-out replicates."""
    model, Sigma, rng = _model(p=50, n_pert=15, seed=2)
    p = Sigma.shape[0]
    genes = rng.choice(p, size=15, replace=False)
    # each perturbation's shift ~ a * Sigma[:, g]; train/test are noisy replicates
    dx_train = np.vstack([3.0 * Sigma[:, g] + rng.normal(0, 0.05, p) for g in genes])
    dx_test = np.vstack([3.0 * Sigma[:, g] + rng.normal(0, 0.05, p) for g in genes])
    res = cipher.identify_perturbations(model, dx_train, dx_test, tau2=1.0,
                                        labels=[f"P{g}" for g in genes])
    assert res.acc1 > 0.6                    # top-1 well above chance (1/15)
    assert res.topk_acc[5] >= res.acc1
    assert res.rank.shape == (15,)
    assert res.roc is None or len(res.roc) == 2

    # LFC cosine baseline runs and returns the same-shaped result
    lfc = cipher.identify_lfc(dx_train, dx_test, labels=[f"P{g}" for g in genes])
    assert lfc.method == "lfc_cosine" and lfc.rank.shape == (15,)
