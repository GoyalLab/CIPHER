"""Regression tests for control/perturbation detection and precompute self-containment."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import anndata as ad
import pytest

import cipher


def test_control_label_detected_within_pinned_pert_key(tmp_path):
    """When the caller pins pert_key, the control label must come from THAT column,
    not from a different (higher-scoring) obs column."""
    rng = np.random.default_rng(0)
    n_ctrl, n_pert, n_genes = 120, 40, 30
    X = rng.poisson(3.0, size=(n_ctrl + 2 * n_pert, n_genes)).astype("float32")
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    # two candidate columns with DIFFERENT control labels over the same cells
    pert_col = ["control"] * n_ctrl + ["G0"] * n_pert + ["G1"] * n_pert
    guide_col = ["non-targeting"] * n_ctrl + ["G0"] * n_pert + ["G1"] * n_pert
    obs = pd.DataFrame({"perturbation": pert_col, "guide_target": guide_col},
                       index=[f"c{i}" for i in range(X.shape[0])])
    path = tmp_path / "twocol.h5ad"
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(str(path))

    # pin the guide_target column, whose control label is 'non-targeting'
    ds = cipher.load_dataset(str(path), pert_key="guide_target",
                             expression_threshold=0.0, min_samples=5)
    assert ds.pert_key == "guide_target"
    assert ds.control_label == "non-targeting"
    assert int(ds.control_mask.sum()) == n_ctrl


def test_precomputed_target_indices_survive_without_dataset_npy(h5ad_path, tmp_path):
    """target_gene_indices are embedded in perturbation_stats.h5, so load_precomputed
    stays correct (and silent) even if the dataset-level .npy files are gone."""
    cfg = cipher.PreprocessConfig(min_samples_per_pert=5, expression_threshold=0.0,
                                  cov_max_cells=None)
    outdir = cipher.preprocess_dataset(h5ad_path, tmp_path / "pp", modes=["log1p"],
                                       config=cfg, progress=False)
    pc_full = cipher.load_precomputed(outdir, "log1p")
    assert (pc_full.target_gene_indices >= 0).all()

    for fname in ("target_gene_indices.npy", "perturbations.npy"):
        os.remove(os.path.join(outdir, fname))

    with warnings.catch_warnings():
        warnings.simplefilter("error")   # no fragile-fallback warning may fire
        pc = cipher.load_precomputed(outdir, "log1p")
    assert np.array_equal(pc.target_gene_indices, pc_full.target_gene_indices)
