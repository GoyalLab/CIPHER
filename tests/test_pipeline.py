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
    real = res.summary["mean_R2_real"]
    null = res.summary["mean_R2_meanfield"]
    assert res.summary["n_perturbations"] == 8
    assert real > 0.5
    assert real > null + 0.2       # real signal notably beats the marginals-only null


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
    assert fwd.summary["mean_R2_real"] > 0.5
    assert rev.summary["mean_auc"] > 0.6


def test_cli_forward_returns_zero(tmp_path, h5ad_path):
    rc = cli.main([
        "forward", h5ad_path, "-o", str(tmp_path / "cli_out"),
        "--normalization", "log1p", "--min-samples", "5",
        "--expression-threshold", "0.0", "--max-perturbations", "5",
    ])
    assert rc == 0
