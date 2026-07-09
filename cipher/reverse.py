"""Reverse prediction: recover the perturbation label from the expression shift.

Given an observed mean shift ``delta_x`` CIPHER ranks genes by how likely each is
the driver.  On a Perturb-seq dataset (where the true perturbed gene is known)
this yields a rank / ROC-AUC per perturbation.  The default solver is the
fullH_diag empirical-Bayes posterior inverse (:mod:`cipher.inverse`, the most
accurate); lightweight linear baselines (``matched_filter``/``pinv``/``ridge``/
``lstsq``) and an optional PyMC horseshoe variant are also available.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data import Dataset, load_dataset
from .normalize import normalize_matrix, library_size, fit_pflog_alpha
from .covariance import compute_covariance
from .core import reverse_operator, rank_of, top_k_hit, one_vs_rest_auc
from .utils import ensure_dir, stable_seed


@dataclass
class ReverseResult:
    """Per-perturbation driver-recovery metrics plus dataset-level summary."""
    results: pd.DataFrame
    summary: dict
    method: str
    normalization: str
    dataset_name: str
    top_k: int = 10

    def save(self, outdir) -> Path:
        outdir = ensure_dir(outdir)
        path = Path(outdir) / f"{self.dataset_name}_reverse_{self.normalization}_{self.method}.csv"
        self.results.to_csv(path, index=False)
        return path

    def __repr__(self) -> str:
        topk_acc = self.summary.get(f"top{self.top_k}_accuracy", float("nan"))
        return (f"ReverseResult(dataset={self.dataset_name!r}, method={self.method!r}, "
                f"n={len(self.results)}, mean_auc={self.summary.get('mean_auc', float('nan')):.3f}, "
                f"top{self.top_k}_acc={topk_acc:.3f})")


def _as_dataset(data, **load_kwargs) -> Dataset:
    if isinstance(data, Dataset):
        return data
    if isinstance(data, (str, os.PathLike)):
        return load_dataset(data, **load_kwargs)
    raise TypeError("Pass an .h5ad path or a cipher.Dataset (from load_dataset).")


def _inverse_to_reverse(inv, top_k: int) -> "ReverseResult":
    """Convert a posterior :class:`cipher.InverseResult` to a :class:`ReverseResult`
    (per-perturbation rank / AUC / top-k accuracy), matching the linear-method output."""
    src = inv.results
    n_genes = int(inv.summary.get("n_genes", 0))
    rank1 = pd.to_numeric(src["rank"], errors="coerce").to_numpy()
    keep = np.isfinite(rank1)
    src = src.loc[keep].reset_index(drop=True)
    rank1 = rank1[keep]
    target_rank = (rank1 - 1).astype(int)
    auc = pd.to_numeric(src["per_pert_auc"], errors="coerce").to_numpy()
    have = len(src) > 0
    hit_col = f"top{top_k}_hit"
    df = pd.DataFrame({
        "dataset": inv.dataset_name,
        "perturbation": src["perturbation"].to_numpy() if have else [],
        "target_gene": src["target_gene"].to_numpy() if have else [],
        "target_rank": target_rank,
        "percentile": 1.0 - target_rank / max(n_genes - 1, 1),
        hit_col: rank1 <= top_k,
        "auc": auc,
    })
    df["normalization"] = inv.normalization
    df["method"] = inv.method
    summary = {
        "dataset": inv.dataset_name, "normalization": inv.normalization, "method": inv.method,
        "n_perturbations": int(len(df)), "n_genes": n_genes,
        "mean_auc": float(np.nanmean(auc)) if have else float("nan"),
        "median_rank": float(np.median(target_rank)) if have else float("nan"),
        "top1_accuracy": float(np.mean(rank1 <= 1)) if have else float("nan"),
        f"top{top_k}_accuracy": float(np.mean(rank1 <= top_k)) if have else float("nan"),
        "tau2": float(inv.summary.get("tau2", float("nan"))),
    }
    return ReverseResult(results=df, summary=summary, method=inv.method,
                         normalization=inv.normalization, dataset_name=inv.dataset_name, top_k=top_k)


def reverse_prediction(
    data,
    normalization: str = "log1p",
    method: str = "posterior",
    top_k: int = 10,
    max_perturbations: int | None = None,
    cov_max_cells: int | None = 10000,
    ridge: float = 1e-2,
    seed: int = 0,
    progress: bool = True,
    **load_kwargs,
) -> ReverseResult:
    """Run CIPHER reverse prediction (driver recovery) on a Perturb-seq dataset.

    Parameters
    ----------
    method : str
        Driver solver. Default ``"posterior"`` — the fullH_diag empirical-Bayes
        posterior inverse (:mod:`cipher.inverse`), the most accurate method;
        ``"pip"`` is its single-effect variant. The lightweight linear baselines
        ``"matched_filter"`` / ``"pinv"`` / ``"ridge"`` / ``"lstsq"`` are also
        available (no eigendecomposition; ``matched_filter`` needs only ``Sigma``).
        For the pooled ROC / precision-recall curves of the posterior inverse use
        :func:`cipher.posterior_inverse_prediction` directly.
    top_k : int
        Rank cutoff for the ``top_k_hit`` / top-k accuracy summary.
    See :func:`cipher.forward.forward_prediction` for the shared parameters.
    """
    ds = _as_dataset(data, **load_kwargs)
    if method in {"posterior", "pip"}:
        from .inverse import posterior_inverse_prediction
        inv = posterior_inverse_prediction(
            ds, normalization=normalization, method=method, cov_max_cells=cov_max_cells,
            max_perturbations=max_perturbations, negatives_per_pert=0, seed=seed,
            progress=progress)
        return _inverse_to_reverse(inv, top_k)
    ds_name = ds.name
    rng_seed = stable_seed(seed, ds_name)

    control_raw = ds.control_matrix(dense=True)   # ALL control cells

    pseudocount = None
    if normalization == "pflog":
        _, pseudocount, _, _ = fit_pflog_alpha(control_raw, ds.gene_names)

    def _norm(X):
        return normalize_matrix(X, normalization, libsize=library_size(X), pseudocount=pseudocount)

    # control_mean over ALL controls; covariance from up to cov_max_cells (row-independent).
    control_norm = _norm(control_raw)
    control_mean = control_norm.mean(axis=0)
    if cov_max_cells is not None and control_norm.shape[0] > cov_max_cells:
        rng = np.random.default_rng(rng_seed)
        sel = np.sort(rng.choice(control_norm.shape[0], cov_max_cells, replace=False))
        control_cov_norm = control_norm[sel]
    else:
        control_cov_norm = control_norm
    Sigma = compute_covariance(control_cov_norm)
    solve = reverse_operator(Sigma, method=method, ridge=ridge)
    n_genes = ds.n_genes

    perts = ds.perturbations
    tgi = ds.target_gene_indices
    if max_perturbations is not None:
        perts, tgi = perts[:max_perturbations], tgi[:max_perturbations]

    rows = []
    it = list(zip(perts, tgi))
    if progress:
        it = tqdm(it, desc=f"reverse:{ds_name}:{normalization}:{method}")
    for pert, gene_idx in it:
        if gene_idx is None or gene_idx < 0:
            continue
        mean_pert = _norm(ds.perturbation_matrix(pert, dense=True)).mean(axis=0)
        delta_x = mean_pert - control_mean
        scores = solve(delta_x)
        r = rank_of(scores, gene_idx)
        rows.append({
            "perturbation": pert, "target_gene": ds.gene_names[gene_idx],
            "target_rank": r, "percentile": 1.0 - r / max(n_genes - 1, 1),
            f"top{top_k}_hit": bool(top_k_hit(scores, gene_idx, top_k)),
            "auc": one_vs_rest_auc(scores, gene_idx),
        })

    df = pd.DataFrame(rows)
    df.insert(0, "dataset", ds_name)
    df["normalization"] = normalization
    df["method"] = method
    hit_col = f"top{top_k}_hit"
    summary = {
        "dataset": ds_name, "normalization": normalization, "method": method,
        "n_perturbations": int(len(df)), "n_genes": int(n_genes),
        "mean_auc": float(df["auc"].mean()) if len(df) else float("nan"),
        "median_rank": float(df["target_rank"].median()) if len(df) else float("nan"),
        "top1_accuracy": float((df["target_rank"] == 0).mean()) if len(df) else float("nan"),
        f"top{top_k}_accuracy": float(df[hit_col].mean()) if len(df) else float("nan"),
    }
    return ReverseResult(results=df, summary=summary, method=method,
                         normalization=normalization, dataset_name=ds_name, top_k=top_k)


def reverse_from_precomputed(dataset_dir, mode: str, method: str = "posterior",
                             top_k: int = 10, ridge: float = 1e-2,
                             progress: bool = True) -> ReverseResult:
    """Reverse metrics from a preprocessed dataset directory.

    ``method`` defaults to the fullH_diag posterior inverse (``"posterior"`` /
    ``"pip"``); the linear baselines (``matched_filter``/``pinv``/``ridge``/``lstsq``)
    are also available.
    """
    if method in {"posterior", "pip"}:
        from .inverse import posterior_inverse_from_precomputed
        inv = posterior_inverse_from_precomputed(dataset_dir, mode, method=method,
                                                 negatives_per_pert=0, progress=progress)
        return _inverse_to_reverse(inv, top_k)
    from .io import load_precomputed

    pc = load_precomputed(dataset_dir, mode)
    Sigma = np.asarray(pc.sigma(mmap=False), dtype=np.float64)
    solve = reverse_operator(Sigma, method=method, ridge=ridge)
    ds_name = Path(dataset_dir).name
    n_genes = len(pc.gene_names)
    rows = []
    it = range(len(pc.perturbations))
    if progress:
        it = tqdm(it, desc=f"reverse:{ds_name}:{mode}:{method}")
    for i in it:
        gene_idx = int(pc.target_gene_indices[i])
        if gene_idx < 0:
            continue
        scores = solve(pc.dx[i])
        r = rank_of(scores, gene_idx)
        rows.append({"perturbation": pc.perturbations[i], "target_gene": pc.gene_names[gene_idx],
                     "target_rank": r, "percentile": 1.0 - r / max(n_genes - 1, 1),
                     f"top{top_k}_hit": bool(top_k_hit(scores, gene_idx, top_k)),
                     "auc": one_vs_rest_auc(scores, gene_idx)})
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", ds_name)
    df["normalization"] = mode
    df["method"] = method
    hit_col = f"top{top_k}_hit"
    summary = {"dataset": ds_name, "normalization": mode, "method": method,
               "n_perturbations": int(len(df)), "n_genes": int(n_genes),
               "mean_auc": float(df["auc"].mean()) if len(df) else float("nan"),
               "median_rank": float(df["target_rank"].median()) if len(df) else float("nan"),
               "top1_accuracy": float((df["target_rank"] == 0).mean()) if len(df) else float("nan"),
               f"top{top_k}_accuracy": float(df[hit_col].mean()) if len(df) else float("nan")}
    return ReverseResult(results=df, summary=summary, method=method,
                         normalization=mode, dataset_name=ds_name, top_k=top_k)


# --------------------------------------------------------------------------- #
# optional Bayesian (horseshoe) reverse inference
# --------------------------------------------------------------------------- #
def bayesian_reverse(Sigma, delta_x, top_k: int = 200, draws: int = 1000,
                     tune: int = 1000, target_accept: float = 0.95,
                     inclusion_threshold: float = 0.05, seed: int = 0):
    """Sparse Bayesian driver inference with a horseshoe prior (requires PyMC).

    Pre-selects the ``top_k`` genes by ``|pinv(Sigma) @ delta_x|``, fits a
    horseshoe-regularized linear model ``delta_x ~= Sigma_sub @ u`` and returns
    ``(pip, u_mean, u_std, selected_indices)`` where ``pip`` is the posterior
    inclusion probability (over the full gene set).
    """
    try:
        import pymc as pm
    except ImportError as e:  # pragma: no cover
        raise ImportError("bayesian_reverse requires PyMC (`pip install pymc`).") from e
    from scipy.linalg import pinv

    Sigma = np.asarray(Sigma, dtype=np.float64)
    delta_x = np.asarray(delta_x, dtype=np.float64)
    G = Sigma.shape[0]
    u_hat = pinv(Sigma) @ delta_x
    sel = np.sort(np.argsort(np.abs(u_hat))[::-1][:top_k])
    Sigma_sub = Sigma[np.ix_(sel, sel)]
    dx_sub = delta_x[sel]

    with pm.Model():
        lam = pm.HalfCauchy("lambda", beta=1.0, shape=len(sel))
        log_tau = pm.Normal("log_tau", mu=-4, sigma=1)
        tau = pm.Deterministic("tau", pm.math.exp(log_tau))
        z = pm.Normal("z", 0, 1, shape=len(sel))
        u = pm.Deterministic("u", z * tau * lam)
        log_sigma = pm.Normal("log_sigma_obs", mu=-2, sigma=2)
        sigma_obs = pm.Deterministic("sigma_obs", pm.math.exp(log_sigma))
        pm.Normal("obs", mu=pm.math.dot(Sigma_sub, u), sigma=sigma_obs, observed=dx_sub)
        trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept,
                          chains=4, cores=4, random_seed=seed, progressbar=False)

    u_samples = trace.posterior["u"].stack(sample=("chain", "draw")).values
    pip_local = np.mean(np.abs(u_samples) > inclusion_threshold, axis=1)
    pip = np.zeros(G)
    pip[sel] = pip_local
    u_mean = np.zeros(G)
    u_std = np.zeros(G)
    u_mean[sel] = np.mean(u_samples, axis=1)
    u_std[sel] = np.std(u_samples, axis=1)
    return pip, u_mean, u_std, sel
