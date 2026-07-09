"""Tests for the fullH_diag posterior inverse and its numpy metrics."""
from __future__ import annotations

import numpy as np
import pytest

import cipher
from cipher import metrics


# --------------------------------------------------------------------------- #
# metrics (numpy, no sklearn)
# --------------------------------------------------------------------------- #
def test_roc_auc_matches_known_values():
    # perfect separation -> 1.0 ; reversed -> 0.0 ; all-tie -> 0.5
    y = np.array([0, 0, 1, 1])
    assert metrics.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert metrics.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert metrics.roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_average_precision_and_curves():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.average_precision(y, s) == pytest.approx(1.0)
    fpr, tpr = metrics.roc_curve(y, s)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)
    precision, recall = metrics.pr_curve(y, s)
    assert recall.max() == pytest.approx(1.0)
    # degenerate single-class -> None / nan
    assert metrics.roc_curve(np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0])) is None
    assert np.isnan(metrics.roc_auc(np.array([1, 1]), np.array([1.0, 2.0])))


# --------------------------------------------------------------------------- #
# posterior inverse on the synthetic dataset
# --------------------------------------------------------------------------- #
def test_posterior_inverse_recovers_drivers(h5ad_path, synth):
    res = cipher.posterior_inverse_prediction(
        h5ad_path, normalization="log1p", method="posterior",
        negatives_per_pert=200, cov_max_cells=None,
        expression_threshold=0.0, min_samples=5, progress=False)
    s = res.summary
    assert s["n_valid"] == len(synth.target_labels)
    assert s["mean_per_pert_auc"] > 0.8          # correlated synthetic is well separated
    assert s["pooled_auc"] > 0.8
    assert 0.0 <= s["pooled_average_precision"] <= 1.0
    assert res.roc is not None and res.prc is not None
    assert np.isfinite(res.tau2) and res.tau2 > 0
    # per-perturbation results are populated
    assert set(["perturbation", "target_gene", "per_pert_auc", "rank"]).issubset(res.results.columns)


def test_pip_method_runs(h5ad_path):
    res = cipher.posterior_inverse_prediction(
        h5ad_path, normalization="log1p", method="pip",
        negatives_per_pert=200, cov_max_cells=None,
        expression_threshold=0.0, min_samples=5, progress=False)
    assert res.method == "pip"
    assert res.summary["mean_per_pert_auc"] > 0.7


def test_posterior_inverse_from_precomputed(tmp_path, h5ad_path):
    cfg = cipher.PreprocessConfig(min_samples_per_pert=5, expression_threshold=0.0,
                                  cov_max_cells=None, save_mean_var=True)
    outdir = cipher.preprocess_dataset(h5ad_path, tmp_path / "pp", modes=["log1p"],
                                       config=cfg, progress=False)
    res = cipher.posterior_inverse_from_precomputed(outdir, "log1p", method="posterior",
                                                    progress=False)
    assert res.summary["n_valid"] > 0
    assert res.summary["mean_per_pert_auc"] > 0.8
    assert res.summary["pooled_auc"] > 0.8


def test_build_model_and_fit_tau2():
    rng = np.random.default_rng(0)
    p, n_pert = 25, 6
    A = rng.normal(size=(p, p))
    Sigma = A @ A.T / p + 0.1 * np.eye(p)
    var_pert = np.abs(rng.normal(size=(n_pert, p))) + 0.1
    nu = np.full(n_pert, 50.0)
    model = cipher.build_model(Sigma, var_pert, n0=100.0, nu=nu)
    assert model.eigenvalues.shape == (p,)
    assert model.pert_eigvar.shape == (n_pert, p)
    dx = rng.normal(size=(n_pert, p))
    eb = cipher.fit_tau2(model, dx)
    assert eb["tau2_use"] > 0
