"""Forward prediction: predict the transcriptomic shift of a known perturbation.

For each perturbed gene ``g`` CIPHER predicts the mean expression shift
``delta_x`` as a rank-1 projection onto the control covariance column
``Sigma[:, g]`` and scores it against the observed shift (R2 / R20 / Spearman /
Pearson).  Null covariances give the baseline expected from marginal statistics
alone.
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
from .core import forward_predict, forward_metrics
from .utils import ensure_dir, stable_seed


@dataclass
class ForwardResult:
    """Per-perturbation forward-prediction metrics plus dataset-level summary."""
    results: pd.DataFrame
    summary: dict
    normalization: str
    dataset_name: str
    nulls: tuple = field(default_factory=tuple)

    def save(self, outdir) -> Path:
        outdir = ensure_dir(outdir)
        path = Path(outdir) / f"{self.dataset_name}_forward_{self.normalization}.csv"
        self.results.to_csv(path, index=False)
        return path

    def __repr__(self) -> str:
        n = len(self.results)
        mr2 = self.summary.get("mean_R2_real", float("nan"))
        return (f"ForwardResult(dataset={self.dataset_name!r}, norm={self.normalization!r}, "
                f"n_perturbations={n}, mean_R2={mr2:.3f})")


def _as_dataset(data, **load_kwargs) -> Dataset:
    if isinstance(data, Dataset):
        return data
    if isinstance(data, (str, os.PathLike)):
        return load_dataset(data, **load_kwargs)
    # assume an AnnData
    import anndata as ad
    if isinstance(data, ad.AnnData):
        import tempfile
        raise TypeError("Pass an .h5ad path or a cipher.Dataset (from load_dataset), not a raw AnnData.")
    raise TypeError(f"Unsupported data type: {type(data)}")


def forward_prediction(
    data,
    normalization: str = "log1p",
    nulls=("meanfield", "shuffled"),
    max_perturbations: int | None = None,
    cov_max_cells: int | None = 10000,
    ridge_abs: float = 0.0,
    ridge_rel: float = 0.0,
    seed: int = 0,
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
        Null covariance models to benchmark against (``meanfield``/``shuffled``/``zinb``).
    max_perturbations : int, optional
        Only evaluate the first ``N`` perturbations (useful for smoke tests).
    cov_max_cells : int, optional
        Subsample this many control cells when estimating the covariance.
    load_kwargs
        Forwarded to :func:`cipher.data.load_dataset` (e.g. ``expression_threshold``).
    """
    ds = _as_dataset(data, **load_kwargs)
    ds_name = ds.name
    nulls = tuple(nulls or ())
    rng_seed = stable_seed(seed, ds_name)

    control_raw = ds.control_matrix(dense=True)   # ALL control cells

    # pflog dispersion fit from the full raw control matrix (matches the canonical pipeline)
    pseudocount = None
    if normalization == "pflog":
        _, pseudocount, _, _ = fit_pflog_alpha(control_raw, ds.gene_names)

    def _norm(X):
        return normalize_matrix(X, normalization, libsize=library_size(X), pseudocount=pseudocount)

    # control_mean is computed over ALL controls; covariance uses up to cov_max_cells.
    # Every normalization is row-independent, so subsampling rows of the normalized
    # matrix is identical to normalizing the subsampled rows.
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

    perts = ds.perturbations
    tgi = ds.target_gene_indices
    if max_perturbations is not None:
        perts, tgi = perts[:max_perturbations], tgi[:max_perturbations]

    rows = []
    it = zip(perts, tgi)
    if progress:
        it = tqdm(list(it), desc=f"forward:{ds_name}:{normalization}")
    for pert, gene_idx in it:
        if gene_idx is None or gene_idx < 0:
            continue
        pert_norm = _norm(ds.perturbation_matrix(pert, dense=True))
        mean_pert = pert_norm.mean(axis=0)
        delta_x = mean_pert - control_mean
        pred, _ = forward_predict(Sigma_real, delta_x, gene_idx)
        row = {"perturbation": pert, "target_gene": ds.gene_names[gene_idx]}
        m = forward_metrics(delta_x, pred, mean_pert)
        row.update({"R2_real": m["R2"], "R20_real": m["R20"],
                    "Spearman_real": m["Spearman"], "Pearson_real": m["Pearson"]})
        for k, Sn in null_sigmas.items():
            pred_n, _ = forward_predict(Sn, delta_x, gene_idx)
            row[f"R2_{k}"] = forward_metrics(delta_x, pred_n)["R2"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(0, "dataset", ds_name)
    df["normalization"] = normalization
    summary = {"dataset": ds_name, "normalization": normalization,
               "n_perturbations": int(len(df)),
               "mean_R2_real": float(df["R2_real"].mean()) if len(df) else float("nan")}
    for k in nulls:
        summary[f"mean_R2_{k}"] = float(df[f"R2_{k}"].mean()) if len(df) else float("nan")
    return ForwardResult(results=df, summary=summary, normalization=normalization,
                         dataset_name=ds_name, nulls=nulls)


def forward_from_precomputed(dataset_dir, mode: str, progress: bool = True) -> ForwardResult:
    """Forward metrics computed from a preprocessed dataset directory (real Sigma only)."""
    from .io import load_precomputed

    pc = load_precomputed(dataset_dir, mode)
    Sigma = pc.sigma(mmap=True)
    ds_name = Path(dataset_dir).name
    rows = []
    idx_it = range(len(pc.perturbations))
    if progress:
        idx_it = tqdm(idx_it, desc=f"forward:{ds_name}:{mode}")
    for i in idx_it:
        gene_idx = int(pc.target_gene_indices[i])
        if gene_idx < 0:
            continue
        delta_x = pc.dx[i]
        pred, _ = forward_predict(Sigma, delta_x, gene_idx)
        m = forward_metrics(delta_x, pred, pc.mean_pert[i])
        rows.append({"perturbation": pc.perturbations[i], "target_gene": pc.gene_names[gene_idx],
                     "R2_real": m["R2"], "R20_real": m["R20"],
                     "Spearman_real": m["Spearman"], "Pearson_real": m["Pearson"]})
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", ds_name)
    df["normalization"] = mode
    summary = {"dataset": ds_name, "normalization": mode, "n_perturbations": int(len(df)),
               "mean_R2_real": float(df["R2_real"].mean()) if len(df) else float("nan")}
    return ForwardResult(results=df, summary=summary, normalization=mode, dataset_name=ds_name)
