"""Condition-driver inference for generic control-vs-condition datasets.

This is the reverse problem applied outside Perturb-seq: given a *control* group
and a *condition* group (e.g. resting vs stimulated, healthy vs disease), CIPHER
computes the mean shift ``delta_x`` and uses the control covariance to rank the
genes most likely to *drive* the condition.  There is no ground-truth label, so
the output is a ranked candidate list rather than an accuracy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import normalize_matrix, library_size, fit_pflog_alpha
from .covariance import compute_covariance
from .core import reverse_scores, matched_filter_scores
from .utils import to_dense, ensure_dir


@dataclass
class DriverResult:
    """Ranked candidate driver genes for a condition."""
    ranking: pd.DataFrame       # gene, driver_score, abs_score, delta_x, rank (sorted)
    delta_x: np.ndarray
    gene_names: np.ndarray
    method: str
    normalization: str
    name: str = "condition"

    def top(self, n: int = 20) -> pd.DataFrame:
        return self.ranking.head(n)

    def save(self, outdir) -> Path:
        outdir = ensure_dir(outdir)
        path = Path(outdir) / f"{self.name}_drivers_{self.normalization}_{self.method}.csv"
        self.ranking.to_csv(path, index=False)
        return path

    def __repr__(self) -> str:
        top = ", ".join(self.ranking["gene"].head(5).astype(str))
        return (f"DriverResult(name={self.name!r}, method={self.method!r}, "
                f"n_genes={len(self.ranking)}, top5=[{top}])")


def condition_drivers_from_matrices(
    control_X,
    condition_X,
    gene_names,
    normalization: str = "log1p",
    method: str = "matched_filter",
    ridge: float = 1e-2,
    name: str = "condition",
) -> DriverResult:
    """Rank driver genes from raw control and condition (cells x genes) matrices.

    ``control_X`` and ``condition_X`` must share the same gene axis (``gene_names``).
    ``method`` defaults to ``matched_filter`` (no matrix inverse — the most robust
    choice for wide gene panels); use ``pinv``/``ridge`` for the inverse solution.
    """
    control_X = to_dense(control_X).astype(np.float64)
    condition_X = to_dense(condition_X).astype(np.float64)
    gene_names = np.asarray(gene_names)
    if control_X.shape[1] != condition_X.shape[1]:
        raise ValueError("control_X and condition_X must have the same number of genes.")
    if control_X.shape[1] != gene_names.shape[0]:
        raise ValueError("gene_names length must match the number of genes.")

    pseudocount = None
    if normalization == "pflog":
        _, pseudocount, _, _ = fit_pflog_alpha(control_X, gene_names)

    def _norm(X):
        return normalize_matrix(X, normalization, libsize=library_size(X), pseudocount=pseudocount)

    control_norm = _norm(control_X)
    condition_norm = _norm(condition_X)
    control_mean = control_norm.mean(axis=0)
    delta_x = condition_norm.mean(axis=0) - control_mean
    Sigma = compute_covariance(control_norm)

    if method == "matched_filter":
        scores = matched_filter_scores(Sigma, delta_x)
    else:
        scores = reverse_scores(Sigma, delta_x, method=method, ridge=ridge)

    ranking = pd.DataFrame({
        "gene": gene_names, "driver_score": scores, "abs_score": np.abs(scores),
        "delta_x": delta_x,
    }).sort_values("abs_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(len(ranking)))
    return DriverResult(ranking=ranking, delta_x=delta_x, gene_names=gene_names,
                        method=method, normalization=normalization, name=name)


def condition_drivers(
    data,
    condition_key: str,
    control_value,
    condition_value=None,
    gene_symbols_col: str | None = None,
    normalization: str = "log1p",
    method: str = "matched_filter",
    ridge: float = 1e-2,
    name: str | None = None,
) -> DriverResult:
    """Rank condition-driver genes from an AnnData / ``.h5ad`` with a group column.

    Parameters
    ----------
    data : str | Path | AnnData
        Dataset with an ``obs[condition_key]`` grouping cells.
    control_value :
        Value of ``condition_key`` marking control cells.
    condition_value : optional
        Value marking condition cells; if ``None`` every non-control cell is used.
    gene_symbols_col : optional
        ``var`` column holding gene symbols (else ``var_names`` is used).
    """
    import anndata as ad
    if isinstance(data, (str, os.PathLike)):
        adata = ad.read_h5ad(str(data))
        default_name = Path(str(data)).stem
    else:
        adata = data
        default_name = "condition"
    name = name or default_name

    if condition_key not in adata.obs.columns:
        raise KeyError(f"{condition_key!r} not in obs columns {list(adata.obs.columns)}")
    if gene_symbols_col and gene_symbols_col in adata.var.columns:
        gene_names = np.asarray(adata.var[gene_symbols_col].astype(str))
    else:
        gene_names = np.asarray(adata.var_names.astype(str))

    col = adata.obs[condition_key].astype(str)
    control_mask = (col == str(control_value)).to_numpy()
    if condition_value is None:
        condition_mask = (col != str(control_value)).to_numpy()
    else:
        condition_mask = (col == str(condition_value)).to_numpy()
    if control_mask.sum() < 2 or condition_mask.sum() < 2:
        raise ValueError(f"Need >=2 cells per group; got control={control_mask.sum()}, "
                         f"condition={condition_mask.sum()}.")

    control_X = adata[control_mask].X
    condition_X = adata[condition_mask].X
    return condition_drivers_from_matrices(
        control_X, condition_X, gene_names,
        normalization=normalization, method=method, ridge=ridge, name=name)
