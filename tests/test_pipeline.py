"""Integration tests that drive the CIPHER pipeline on the synthetic dataset.

Everything runs on the small in-memory / tmp ``.h5ad`` fixture, uses
``progress=False`` and caps work (``cov_max_cells=None`` is safe because the
synthetic control set is tiny), so the whole module stays fast.
"""
from __future__ import annotations

import cipher
from cipher import cli


def test_load_dataset_metadata(h5ad_path, synth):
    ds = cipher.load_dataset(h5ad_path, expression_threshold=0.0, min_samples=5)
    assert ds.n_perturbations == len(synth.target_labels)      # 8
    assert set(ds.perturbations) == set(synth.target_labels)
    assert ds.n_genes == synth.n_genes
    # each perturbation label is its own target gene symbol
    for pert, idx in zip(ds.perturbations, ds.target_gene_indices):
        assert idx >= 0
        assert ds.gene_names[idx] == pert


def test_forward_beats_meanfield_null(h5ad_path):
    res = cipher.forward_prediction(
        h5ad_path, normalization="log1p", nulls=("meanfield",),
        expression_threshold=0.0, min_samples=5, cov_max_cells=None, progress=False)
    real = res.summary["mean_r2_uncentered_real"]
    null = res.summary["mean_r2_uncentered_meanfield"]
    assert res.summary["n_perturbations"] == 8
    assert real > 0.5
    assert real > null + 0.2       # real signal notably beats the marginals-only null
    # the full metric set is present per perturbation
    for col in ("r2_uncentered_real", "r2_centered_real", "pearson_real",
                "cosine_real", "sign_accuracy_real", "a_hat"):
        assert col in res.results.columns


def test_forward_gene_holdout_out_of_sample(h5ad_path):
    """Out-of-sample gene holdout: a_hat fit on train genes still explains held-out genes."""
    res = cipher.forward_prediction(
        h5ad_path, normalization="log1p", nulls=("meanfield",), holdout_frac=0.5,
        min_train_genes=20, min_test_genes=20, expression_threshold=0.0,
        min_samples=5, cov_max_cells=None, progress=False)
    assert res.summary["n_perturbations"] == 8
    assert res.holdout_frac == 0.5
    assert (res.results["n_test_genes"] < res.results["n_train_genes"] + 1).all()
    assert res.summary["mean_r2_uncentered_real"] > res.summary["mean_r2_uncentered_meanfield"]


def test_reverse_recovers_driver(h5ad_path, synth):
    res = cipher.reverse_prediction(
        h5ad_path, normalization="log1p", method="pinv", top_k=10,
        expression_threshold=0.0, min_samples=5, cov_max_cells=None, progress=False)
    assert res.summary["mean_auc"] > 0.6
    random_topk = 10.0 / synth.n_genes       # chance level for hitting one gene in top-10
    assert res.summary["top10_accuracy"] > random_topk


def test_condition_drivers_ranks_perturbed_gene(h5ad_path):
    ds = cipher.load_dataset(h5ad_path, expression_threshold=0.0, min_samples=5)
    target = ds.perturbations[0]
    control_X = ds.control_matrix(dense=True)
    condition_X = ds.perturbation_matrix(target, dense=True)
    dr = cipher.condition_drivers_from_matrices(
        control_X, condition_X, ds.gene_names, normalization="log1p", method="pinv")
    assert target in dr.ranking["gene"].head(5).tolist()
    rank = int(dr.ranking.loc[dr.ranking["gene"] == target, "rank"].iloc[0])
    assert rank < 5


def test_preprocess_then_precomputed_agree(tmp_path, h5ad_path):
    cfg = cipher.PreprocessConfig(
        min_samples_per_pert=5, expression_threshold=0.0, cov_max_cells=None)
    out = cipher.preprocess_dataset(
        h5ad_path, tmp_path / "pp", modes=["log1p"], config=cfg, progress=False)
    assert "log1p" in cipher.list_modes(out)

    pc = cipher.load_precomputed(out, "log1p")
    fwd = cipher.forward_from_precomputed(out, "log1p", progress=False)
    rev = cipher.reverse_from_precomputed(out, "log1p", method="pinv", progress=False)

    assert fwd.summary["n_perturbations"] == rev.summary["n_perturbations"]
    assert fwd.summary["n_perturbations"] == len(pc.perturbations)
    # the precomputed artifacts reproduce the same real signal as the live pipeline
    assert fwd.summary["mean_r2_uncentered_real"] > 0.5
    assert rev.summary["mean_auc"] > 0.6


def test_cli_forward_returns_zero(tmp_path, h5ad_path):
    rc = cli.main([
        "forward", h5ad_path, "-o", str(tmp_path / "cli_out"),
        "--normalization", "log1p", "--min-samples", "5",
        "--expression-threshold", "0.0", "--max-perturbations", "5",
    ])
    assert rc == 0


def test_load_matched_datasets_shares_gene_axis(tmp_path):
    """load_matched_datasets restricts two datasets to their shared gene set so a
    covariance from one can be applied to the other (cross-dataset transfer)."""
    import numpy as np, pandas as pd, anndata as ad, cipher

    def _make(genes, seed):
        r = np.random.default_rng(seed)
        X = [r.poisson(2.0, (200, len(genes))).astype("float32")]
        labels = ["control"] * 200
        for g in range(6):
            Y = r.poisson(2.0, (40, len(genes))).astype("float32"); Y[:, g] += 5
            X.append(Y); labels += [genes[g]] * 40
        return ad.AnnData(np.vstack(X), obs=pd.DataFrame({"perturbation": labels}),
                          var=pd.DataFrame(index=genes))

    gA = [f"G{i}" for i in range(50)]
    gB = [f"G{i}" for i in range(15, 65)]           # overlap G15..G49 == 35 genes
    _make(gA, 1).write_h5ad(str(tmp_path / "A.h5ad"))
    _make(gB, 2).write_h5ad(str(tmp_path / "B.h5ad"))
    ds_a, ds_b = cipher.load_matched_datasets(
        str(tmp_path / "A.h5ad"), str(tmp_path / "B.h5ad"),
        expression_threshold=0.0, min_samples=5)
    assert list(ds_a.gene_names) == list(ds_b.gene_names)   # identical, ordered gene axis
    assert ds_a.n_genes == ds_b.n_genes == 35
    # a covariance from A applies to B's shifts on the shared axis
    SigA = cipher.compute_covariance(ds_a.control_matrix())
    assert SigA.shape == (35, 35)
