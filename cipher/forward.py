"""Forward prediction: predict the transcriptomic shift of a known perturbation.

For each perturbed gene ``g`` CIPHER models the mean expression shift as a rank-1
projection onto the control covariance column ``Sigma[:, g]``::

    dx_pert ~= a_hat * Sigma[:, g],   a_hat = <Sigma_col, dx> / <Sigma_col, Sigma_col>

The scalar ``a_hat`` is fit on a *train* set of genes and the fit is scored on a
(held-out) *test* set — with ``holdout_frac=0`` train and test are all genes.
Every gene-wise metric in :data:`cipher.core.FORWARD_METRICS` is reported per
perturbation (uncentered/centered R², Pearson, Spearman, cosine, MSE/RMSE/MAE,
sign accuracy).  Optional null covariances give the baseline expected from
marginal statistics with the gene-gene covariance destroyed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data import Dataset, load_dataset
from .normalize import normalize_matrix, library_size, fit_pflog_alpha
from .covariance import compute_covariance, null_covariance
from .core import forward_fit, forward_metrics, gene_holdout_masks, FORWARD_METRICS
from .utils import ensure_dir, stable_seed

#: metrics summarized (as ``mean_<metric>_real``) at the dataset level.
_SUMMARY_METRICS = ("r2_uncentered", "r2_centered", "pearson", "spearman", "cosine")


@dataclass
class ForwardResult:
    """Per-perturbation forward-prediction metrics plus dataset-level summary."""
    results: pd.DataFrame
    summary: dict
    normalization: str
    dataset_name: str
    holdout_frac: float = 0.0
    nulls: tuple = field(default_factory=tuple)

    def save(self, outdir) -> Path:
        outdir = ensure_dir(outdir)
        path = Path(outdir) / f"{self.dataset_name}_forward_{self.normalization}.csv"
        self.results.to_csv(path, index=False)
        return path

    def __repr__(self) -> str:
        mr2 = self.summary.get("mean_r2_uncentered_real", float("nan"))
        mp = self.summary.get("mean_pearson_real", float("nan"))
        return (f"ForwardResult(dataset={self.dataset_name!r}, norm={self.normalization!r}, "
                f"n={len(self.results)}, holdout={self.holdout_frac}, "
                f"mean_r2_uncentered={mr2:.3f}, mean_pearson={mp:.3f})")


def _as_dataset(data, **load_kwargs) -> Dataset:
    if isinstance(data, Dataset):
        return data
    if isinstance(data, (str, os.PathLike)):
        return load_dataset(data, **load_kwargs)
    raise TypeError("Pass an .h5ad path or a cipher.Dataset (from load_dataset).")


def _forward_row(pert, target_gene, gene_idx, dx, sigma_col, null_cols, n_genes, rng,
                 holdout_frac, exclude_fit, exclude_eval, min_train, min_test):
    """Fit a_hat on train genes and score the held-out test genes for one perturbation.

    Returns a row dict, or ``None`` when there are too few usable genes.
    """
    dx = np.asarray(dx, dtype=np.float64)
    sigma_col = np.asarray(sigma_col, dtype=np.float64)
    finite = np.isfinite(dx) & np.isfinite(sigma_col)
    train, test = gene_holdout_masks(n_genes, gene_idx, holdout_frac, rng=rng,
                                     exclude_target_fit=exclude_fit,
                                     exclude_target_eval=exclude_eval)
    train &= finite
    test &= finite
    n_tr, n_te = int(train.sum()), int(test.sum())
    if n_tr < min_train or n_te < min_test:
        return None
    a_hat, _ = forward_fit(sigma_col, dx, mask=train)
    if not np.isfinite(a_hat):
        return None
    y_true = dx[test]
    row = {"perturbation": pert, "target_gene": target_gene, "a_hat": a_hat,
           "n_train_genes": n_tr, "n_test_genes": n_te}
    row.update({f"{k}_real": v for k, v in forward_metrics(y_true, a_hat * sigma_col[test]).items()})
    for name, col in null_cols.items():
        a_n, _ = forward_fit(col, dx, mask=train)
        y_pred_n = (a_n if np.isfinite(a_n) else 0.0) * col[test]
        row[f"r2_uncentered_{name}"] = forward_metrics(y_true, y_pred_n)["r2_uncentered"]
    return row


def _build_result(rows, ds_name, normalization, holdout_frac, nulls):
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", ds_name)
    df["normalization"] = normalization
    df["holdout_frac"] = holdout_frac
    have = len(df) > 0
    summary = {"dataset": ds_name, "normalization": normalization,
               "holdout_frac": float(holdout_frac), "n_perturbations": int(len(df))}
    for metric in _SUMMARY_METRICS:
        col = f"{metric}_real"
        summary[f"mean_{col}"] = float(df[col].mean()) if have and col in df else float("nan")
    for k in nulls:
        col = f"r2_uncentered_{k}"
        summary[f"mean_{col}"] = float(df[col].mean()) if have and col in df else float("nan")
    return ForwardResult(results=df, summary=summary, normalization=normalization,
                         dataset_name=ds_name, holdout_frac=float(holdout_frac), nulls=tuple(nulls))


def forward_prediction(
    data,
    normalization: str = "log1p",
    nulls=("meanfield", "shuffled"),
    holdout_frac: float = 0.0,
    max_perturbations: int | None = None,
    cov_max_cells: int | None = 10000,
    ridge_abs: float = 0.0,
    ridge_rel: float = 0.0,
    exclude_target_from_fit: bool = False,
    exclude_target_from_eval: bool = False,
    min_train_genes: int = 50,
    min_test_genes: int = 50,
    seed: int = 0,
    split_seed: int = 0,
    progress: bool = True,
    **load_kwargs,
) -> ForwardResult:
    """Run CIPHER forward prediction end-to-end on a Perturb-seq dataset.

    Parameters
    ----------
    data : str | Path | cipher.Dataset
        Path to an ``.h5ad`` file or a pre-loaded :class:`~cipher.data.Dataset`.
    normalization : str
        One of the modes in :data:`cipher.normalize.NORMALIZATION_MODES`.
    nulls : sequence of str
        Null covariance models to benchmark (``meanfield``/``shuffled``/``zinb``);
        each reports ``r2_uncentered_<null>``.
    holdout_frac : float
        Fraction of genes held out for evaluation. ``0`` (default) fits and scores
        ``a_hat`` on all genes; the paper's forward figures use ``0.5`` (out-of-sample).
    exclude_target_from_fit / exclude_target_from_eval : bool
        Drop the perturbed target gene from the fit / evaluation set.
    min_train_genes / min_test_genes : int
        Skip a perturbation if fewer usable genes remain after masking.
    cov_max_cells : int, optional
        Subsample this many control cells when estimating the covariance.
    load_kwargs
        Forwarded to :func:`cipher.data.load_dataset` (e.g. ``expression_threshold``).
    """
    ds = _as_dataset(data, **load_kwargs)
    ds_name = ds.name
    nulls = tuple(nulls or ())
    rng_seed = stable_seed(seed, ds_name)
    split_rng = np.random.default_rng(stable_seed(split_seed, f"{ds_name}:{normalization}"))

    control_raw = ds.control_matrix(dense=True)   # ALL control cells
    pseudocount = None
    if normalization == "pflog":
        _, pseudocount, _, _ = fit_pflog_alpha(control_raw, ds.gene_names)

    def _norm(X):
        return normalize_matrix(X, normalization, libsize=library_size(X), pseudocount=pseudocount)

    # control_mean over ALL controls; covariance from up to cov_max_cells
    # (normalizations are row-independent, so subsampling normalized rows is exact).
    control_norm = _norm(control_raw)
    control_mean = control_norm.mean(axis=0)
    if cov_max_cells is not None and control_norm.shape[0] > cov_max_cells:
        rng = np.random.default_rng(rng_seed)
        sel = np.sort(rng.choice(control_norm.shape[0], cov_max_cells, replace=False))
        control_cov_norm = control_norm[sel]
    else:
        control_cov_norm = control_norm
    Sigma_real = compute_covariance(control_cov_norm, ridge_abs=ridge_abs, ridge_rel=ridge_rel)
    null_sigmas = {k: null_covariance(control_cov_norm, k, seed=rng_seed) for k in nulls}
    n_genes = ds.n_genes

    perts = ds.perturbations
    tgi = ds.target_gene_indices
    if max_perturbations is not None:
        perts, tgi = perts[:max_perturbations], tgi[:max_perturbations]

    rows = []
    it = list(zip(perts, tgi))
    if progress:
        it = tqdm(it, desc=f"forward:{ds_name}:{normalization}")
    for pert, gene_idx in it:
        if gene_idx is None or gene_idx < 0:
            continue
        dx = _norm(ds.perturbation_matrix(pert, dense=True)).mean(axis=0) - control_mean
        null_cols = {k: Sn[:, gene_idx] for k, Sn in null_sigmas.items()}
        row = _forward_row(pert, ds.gene_names[gene_idx], gene_idx, dx, Sigma_real[:, gene_idx],
                           null_cols, n_genes, split_rng, holdout_frac,
                           exclude_target_from_fit, exclude_target_from_eval,
                           min_train_genes, min_test_genes)
        if row is not None:
            rows.append(row)

    return _build_result(rows, ds_name, normalization, holdout_frac, nulls)


def forward_from_precomputed(
    dataset_dir, mode: str,
    holdout_frac: float = 0.0,
    exclude_target_from_fit: bool = False,
    exclude_target_from_eval: bool = False,
    min_train_genes: int = 50,
    min_test_genes: int = 50,
    split_seed: int = 0,
    progress: bool = True,
) -> ForwardResult:
    """Forward metrics from a preprocessed dataset directory (real Sigma only).

    Mirrors the paper's final forward recompute: it fits ``a_hat`` per perturbation
    on the precomputed ``Sigma`` column and ``dx`` (with an optional gene holdout)
    and reports the full metric set.
    """
    from .io import load_precomputed

    pc = load_precomputed(dataset_dir, mode)
    Sigma = pc.sigma(mmap=True)
    ds_name = Path(dataset_dir).name
    n_genes = len(pc.gene_names)
    split_rng = np.random.default_rng(stable_seed(split_seed, f"{ds_name}:{mode}"))

    rows = []
    idx_it = range(len(pc.perturbations))
    if progress:
        idx_it = tqdm(idx_it, desc=f"forward:{ds_name}:{mode}")
    for i in idx_it:
        gene_idx = int(pc.target_gene_indices[i])
        if gene_idx < 0:
            continue
        row = _forward_row(pc.perturbations[i], pc.gene_names[gene_idx], gene_idx,
                           pc.dx[i], np.asarray(Sigma[:, gene_idx], dtype=np.float64), {},
                           n_genes, split_rng, holdout_frac,
                           exclude_target_from_fit, exclude_target_from_eval,
                           min_train_genes, min_test_genes)
        if row is not None:
            rows.append(row)

    return _build_result(rows, ds_name, mode, holdout_frac, ())
