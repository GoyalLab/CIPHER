"""Fast posterior inverse problem (``fullH_diag``) — driver recovery.

This is CIPHER's principled solution to the inverse problem: given an observed
mean shift ``dx`` for a perturbation, rank every gene by how likely it is to be
the driver.  Unlike the simple linear solvers in :mod:`cipher.reverse`
(pinv/ridge/matched_filter), it accounts for the *heteroscedastic sampling noise*
of ``dx`` and places an empirical-Bayes prior on the perturbation vector.

Model (in the eigenbasis ``Sigma0 = V diag(lambda) V.T`` of the control covariance):

* the sampling covariance of ``dx = mean_pert - control_mean`` is, per eigenmode,
  ``h_k = lambda_k / n0 + projvar_k / nu``  where ``n0`` = #control cells,
  ``nu`` = #cells for the perturbation, and ``projvar = var_pert @ (V*V)`` is the
  perturbation variance projected onto the eigenvectors;
* whitening gives ``z = (dx @ V) / sqrt(h)`` and design ``d = lambda / sqrt(h)``;
* a prior ``u_k ~ N(0, tau2)`` (``tau2`` fit by empirical Bayes) yields a closed-form
  Gaussian posterior for the per-gene perturbation strength.

Two per-gene scores are provided:

* ``posterior`` — ``max(|mean + std|, |mean - std|)`` of the posterior perturbation
  strength (an uncertainty-inflated magnitude); the main CIPHER inverse score.
* ``pip`` — the single-effect posterior inclusion probability from the per-gene
  log Bayes factor.

Driver recovery is scored per perturbation (one-vs-rest ROC-AUC over all genes,
rank of the true gene) and pooled across perturbations into ROC / PR curves.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import metrics as _metrics
from .utils import ensure_dir

# ---- defaults mirroring the reference notebook -----------------------------
LOGTAU2_BOUNDS = (-2.0, -1.0)
EB_GRID_N = 100
PLATEAU_DELTA = 1.92
FULLH_RIDGE_REL = 1e-8
FULLH_RIDGE_ABS = 1e-12
PERT_BATCH = 128
ROC_NEGATIVES_PER_PERT = 512


def _nan0(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# paper dataset groupings (for reproducing the CRISPRi / CRISPRa figure)
# --------------------------------------------------------------------------- #
CRISPRA_DATASETS = [
    "akana_etal_2026_crispra_perturbseq", "schemidt_etal_2022_crispra_perturbseq",
    "kaden25_rpe1_ctrl_10k_min100_greedy_4gb", "kaden25_fibroblast_ctrl_10k_min100_greedy_4gb",
    "NormanWeissman2019_filtered", "TianKampmann2021_CRISPRa",
]
CRISPRI_DATASETS = [
    "XAtlas2025_HEK293T_filtered", "XAtlas2025_HCT116_filtered",
    "Marson2025_D1_Rest_filtered", "Marson2025_D1_Stim8hr_filtered", "Marson2025_D1_Stim48hr_filtered",
    "Marson2025_D2_Stim8hr_filtered", "Marson2025_D2_Stim48hr_filtered",
    "Marson2025_D3_Rest_filtered", "Marson2025_D3_Stim8hr_filtered", "Marson2025_D3_Stim48hr_filtered",
    "Marson2025_D4_Rest_filtered", "Marson2025_D4_Stim8hr_filtered", "Marson2025_D4_Stim48hr_filtered",
    "ReplogleWeissman2022_rpe1", "ReplogleWeissman2022_K562_essential",
    "GSE264667_jurkat_raw_singlecell_01", "GSE264667_hepg2_raw_singlecell_01",
    "FrangiehIzar2021_RNA", "TianKampmann2019_day7neuron", "TianKampmann2021_CRISPRi",
    "TianKampmann2019_iPSC",
]


def dataset_group(name) -> str:
    """Classify a dataset name as ``"CRISPRa"``, ``"CRISPRi"``, or ``"unknown"``.

    Uses the paper's dataset lists (substring match) so results can be grouped for
    the per-dataset ROC/PR figure.
    """
    name = str(name)
    if any(k in name for k in CRISPRA_DATASETS):
        return "CRISPRa"
    if any(k in name for k in CRISPRI_DATASETS):
        return "CRISPRi"
    return "unknown"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@dataclass
class PosteriorInverseModel:
    """Eigendecomposition of ``Sigma0`` plus the per-perturbation noise ingredients."""
    eigenvalues: np.ndarray       # (p,)
    V: np.ndarray                 # (p, p) eigenvectors (columns)
    V2: np.ndarray                # (p, p) elementwise V*V
    pert_eigvar: np.ndarray       # (n_pert, p) perturbation variance projected onto eigenvectors
    n0: float                     # number of control cells
    nu: np.ndarray                # (n_pert,) cells per perturbation
    ridge: float

    def batch_terms(self, dx_batch, row_start, row_end):
        """Whitened design ``d`` and data ``z`` for a batch of perturbations."""
        lam = self.eigenvalues
        h = lam[None, :] / self.n0 + self.pert_eigvar[row_start:row_end, :] / self.nu[row_start:row_end, None]
        h = np.maximum(h, self.ridge)
        inv_sqrt_h = 1.0 / np.sqrt(h)
        DX = _nan0(np.asarray(dx_batch, dtype=np.float64))
        d = lam[None, :] * inv_sqrt_h
        z = (DX @ self.V) * inv_sqrt_h
        return d, z


def build_model(Sigma, var_pert, n0, nu, control_var=None, symmetrize=True):
    """Build a :class:`PosteriorInverseModel` from a covariance and per-perturbation variances.

    ``var_pert`` is ``(n_pert, p)`` gene-wise variances of the perturbed cells
    (the gene-diagonal approximation of the projected variance).  If ``var_pert``
    is ``None`` and ``control_var`` is given, the control variance is used for every
    perturbation.
    """
    Sigma = _nan0(np.asarray(Sigma, dtype=np.float64))
    p = Sigma.shape[0]
    if symmetrize:
        Sigma = 0.5 * (Sigma + Sigma.T)
    eigenvalues, V = np.linalg.eigh(Sigma)
    eigenvalues = np.maximum(_nan0(eigenvalues), 0.0)
    V = _nan0(V)
    V2 = V * V

    nu = np.asarray(nu, dtype=np.float64).reshape(-1)
    n_pert = nu.shape[0]
    if var_pert is not None:
        var_pert = np.maximum(_nan0(np.asarray(var_pert, dtype=np.float64)), 0.0)
    elif control_var is not None:
        cv = np.maximum(_nan0(np.asarray(control_var, dtype=np.float64)).reshape(-1), 0.0)
        var_pert = np.broadcast_to(cv[None, :], (n_pert, p)).copy()
    else:
        raise ValueError("Need var_pert (or control_var) to build the fullH_diag model.")
    pert_eigvar = np.maximum(_nan0(var_pert @ V2), 0.0)

    positive = eigenvalues[eigenvalues > 0]
    scale = float(np.median(positive) / n0) if positive.size else 1.0
    ridge = max(float(FULLH_RIDGE_ABS), float(FULLH_RIDGE_REL) * scale)
    return PosteriorInverseModel(eigenvalues=eigenvalues, V=V, V2=V2, pert_eigvar=pert_eigvar,
                                 n0=float(n0), nu=nu, ridge=float(ridge))


# --------------------------------------------------------------------------- #
# empirical Bayes for tau2
# --------------------------------------------------------------------------- #
def fit_tau2(model, dx, batch=PERT_BATCH, plateau=True, progress=False):
    """Empirical-Bayes estimate of the prior variance ``tau2``.

    Minimises the marginal negative log-likelihood ``0.5 * sum(log(1 + tau2 d^2) +
    z^2 / (1 + tau2 d^2))`` over ``log tau2``.  When ``plateau`` is set, returns the
    largest ``tau2`` on a grid within ``PLATEAU_DELTA`` NLL of the minimum
    (a conservative choice that avoids over-shrinking).
    """
    from scipy.optimize import minimize_scalar

    n_pert = dx.shape[0]
    d2_all = np.empty(dx.shape, dtype=np.float32)
    z2_all = np.empty(dx.shape, dtype=np.float32)
    it = range(0, n_pert, batch)
    if progress:
        it = tqdm(it, desc="EB tau2", leave=False)
    for a in it:
        b = min(a + batch, n_pert)
        d, z = model.batch_terms(np.asarray(dx[a:b]), a, b)
        d2_all[a:b] = (d * d).astype(np.float32)
        z2_all[a:b] = (z * z).astype(np.float32)

    def nll(log_tau2):
        tau2 = float(np.exp(log_tau2))
        mv = np.maximum(1.0 + tau2 * d2_all, 1e-12)
        return 0.5 * float(np.sum(np.log(mv) + z2_all / mv, dtype=np.float64))

    res = minimize_scalar(nll, bounds=LOGTAU2_BOUNDS, method="bounded")
    tau2_opt = float(np.exp(res.x))

    grid = np.linspace(LOGTAU2_BOUNDS[0], LOGTAU2_BOUNDS[1], int(EB_GRID_N))
    nll_grid = np.array([nll(v) for v in grid])
    imin = int(np.nanargmin(nll_grid))
    acceptable = np.where(nll_grid <= nll_grid[imin] + PLATEAU_DELTA)[0]
    iplateau = int(acceptable[-1]) if acceptable.size else imin
    tau2_plateau = float(np.exp(grid[iplateau]))
    return {"tau2_opt": tau2_opt, "tau2_plateau": tau2_plateau,
            "tau2_use": tau2_plateau if plateau else tau2_opt,
            "grid_logtau2": grid, "nll_grid": nll_grid}


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def posterior_scores_batch(model, dx_batch, row_start, row_end, tau2):
    """Per-gene posterior score ``max(|mean+std|, |mean-std|)`` for a batch."""
    d, z = model.batch_terms(dx_batch, row_start, row_end)
    d2 = d * d
    post_var_eig = 1.0 / np.maximum(d2 + 1.0 / tau2, 1e-12)
    post_mean_eig = d * post_var_eig * z
    post_mean = post_mean_eig @ model.V.T
    post_var_diag = post_var_eig @ model.V2.T
    post_std = np.sqrt(np.maximum(post_var_diag, 0.0))
    score = np.maximum(np.abs(post_mean + post_std), np.abs(post_mean - post_std))
    return _nan0(score)


def pip_scores_batch(model, dx_batch, row_start, row_end, tau2):
    """Per-gene single-effect posterior inclusion probability for a batch."""
    d, z = model.batch_terms(dx_batch, row_start, row_end)
    denom_stat = np.maximum((d * d) @ model.V2.T, 1e-12)
    num_stat = (d * z) @ model.V.T
    post_denom = 1.0 + tau2 * denom_stat
    logbf = -0.5 * np.log(post_denom) + 0.5 * tau2 * num_stat * num_stat / post_denom
    logbf = np.nan_to_num(logbf, nan=-1e30, posinf=1e30, neginf=-1e30)
    logbf -= np.max(logbf, axis=1, keepdims=True)
    pip = np.exp(logbf)
    pip /= np.maximum(np.sum(pip, axis=1, keepdims=True), 1e-300)
    return pip


# --------------------------------------------------------------------------- #
# per-perturbation + pooled evaluation
# --------------------------------------------------------------------------- #
class _ScoreAccumulator:
    """Accumulates per-perturbation AUC/rank and pooled positives/negatives."""

    def __init__(self, n_perts, n_genes, target_idx, seed=0, negatives_per_pert=ROC_NEGATIVES_PER_PERT):
        self.n_genes = int(n_genes)
        self.target_idx = np.asarray(target_idx, dtype=np.int64)
        self.negatives_per_pert = int(negatives_per_pert)
        self.rng = np.random.default_rng(seed)
        self.per_auc = np.full(int(n_perts), np.nan)
        self.rank = np.full(int(n_perts), np.nan)
        self._pos = []
        self._neg = []

    def update(self, score_batch, row_start):
        scores = np.nan_to_num(np.asarray(score_batch, dtype=np.float64),
                               nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        bsz, n_genes = scores.shape
        rows = np.arange(row_start, row_start + bsz)
        targets = self.target_idx[rows]
        valid = (targets >= 0) & (targets < n_genes)
        scores, rows, targets = scores[valid], rows[valid], targets[valid]
        if rows.size == 0:
            return
        tscore = scores[np.arange(scores.shape[0]), targets]
        finite = np.isfinite(tscore)
        scores, rows, targets, tscore = scores[finite], rows[finite], targets[finite], tscore[finite]
        if rows.size == 0:
            return
        below = np.sum(scores < tscore[:, None], axis=1)
        ties = np.sum(scores == tscore[:, None], axis=1)
        denom = max(n_genes - 1, 1)
        self.per_auc[rows] = (below + 0.5 * np.maximum(ties - 1, 0)) / denom
        self.rank[rows] = 1 + np.sum(scores > tscore[:, None], axis=1)
        self._pos.append(tscore.astype(np.float32))

        k = min(self.negatives_per_pert, n_genes - 1)
        if k <= 0:
            return
        chunks = []
        for r in range(scores.shape[0]):
            t = int(targets[r])
            idx = self.rng.integers(0, n_genes - 1, size=k)
            idx = idx + (idx >= t)
            vals = scores[r, idx]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                chunks.append(vals.astype(np.float32))
        if chunks:
            self._neg.append(np.concatenate(chunks))

    def finish(self):
        good = np.isfinite(self.per_auc)
        gr = np.isfinite(self.rank)
        summary = {
            "n_valid": int(np.sum(good)),
            "mean_per_pert_auc": float(np.nanmean(self.per_auc)) if np.any(good) else np.nan,
            "median_per_pert_auc": float(np.nanmedian(self.per_auc)) if np.any(good) else np.nan,
            "median_rank": float(np.nanmedian(self.rank)) if np.any(gr) else np.nan,
            "top1": float(np.mean(self.rank[gr] <= 1)) if np.any(gr) else np.nan,
            "top5": float(np.mean(self.rank[gr] <= 5)) if np.any(gr) else np.nan,
            "top10": float(np.mean(self.rank[gr] <= 10)) if np.any(gr) else np.nan,
            "pooled_auc": np.nan, "pooled_average_precision": np.nan,
        }
        roc = prc = None
        if self._pos and self._neg:
            pos = np.concatenate(self._pos).astype(np.float64)
            neg = np.concatenate(self._neg).astype(np.float64)
            scores = np.concatenate([pos, neg])
            labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
            fin = np.isfinite(scores)
            scores, labels = scores[fin], labels[fin]
            if labels.size >= 2 and 0 < labels.sum() < labels.size:
                summary["pooled_auc"] = _metrics.roc_auc(labels, scores)
                summary["pooled_average_precision"] = _metrics.average_precision(labels, scores)
                roc = _metrics.roc_curve(labels, scores)
                prc = _metrics.pr_curve(labels, scores)
        return summary, roc, prc


@dataclass
class InverseResult:
    """Posterior-inverse driver-recovery metrics, ROC/PR curves and summary."""
    results: pd.DataFrame          # per perturbation: target_gene, per_pert_auc, rank
    summary: dict
    roc: tuple | None              # (fpr, tpr)
    prc: tuple | None              # (precision, recall)
    method: str
    normalization: str
    dataset_name: str
    tau2: float = float("nan")
    scores: np.ndarray | None = None

    def save(self, outdir) -> Path:
        outdir = ensure_dir(outdir)
        path = Path(outdir) / f"{self.dataset_name}_inverse_{self.normalization}_{self.method}.csv"
        self.results.to_csv(path, index=False)
        return path

    def __repr__(self) -> str:
        s = self.summary
        return (f"InverseResult(dataset={self.dataset_name!r}, method={self.method!r}, "
                f"n={len(self.results)}, pooled_auc={s.get('pooled_auc', float('nan')):.3f}, "
                f"mean_per_pert_auc={s.get('mean_per_pert_auc', float('nan')):.3f}, "
                f"pooled_AP={s.get('pooled_average_precision', float('nan')):.3f})")


def _run_scoring(model, dx, target_idx, method, tau2, gene_names, perturbations,
                 negatives_per_pert, seed, batch, dataset_name, normalization,
                 return_scores, progress):
    n_pert, n_gene = dx.shape
    acc = _ScoreAccumulator(n_pert, n_gene, target_idx, seed=seed,
                            negatives_per_pert=negatives_per_pert)
    score_fn = pip_scores_batch if method == "pip" else posterior_scores_batch
    out = np.empty((n_pert, n_gene), dtype=np.float32) if return_scores else None
    it = range(0, n_pert, batch)
    if progress:
        it = tqdm(it, desc=f"inverse:{dataset_name}:{normalization}:{method}", leave=False)
    for a in it:
        b = min(a + batch, n_pert)
        sc = score_fn(model, np.asarray(dx[a:b], dtype=np.float64), a, b, tau2)
        sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if out is not None:
            out[a:b] = sc
        acc.update(sc, a)
    summary, roc, prc = acc.finish()

    tg = np.asarray(gene_names)[np.clip(np.asarray(target_idx), 0, n_gene - 1)]
    tg = np.where(np.asarray(target_idx) >= 0, tg, "")
    df = pd.DataFrame({
        "dataset": dataset_name, "perturbation": np.asarray(perturbations).astype(str),
        "target_gene": tg, "target_idx": np.asarray(target_idx, dtype=np.int64),
        "per_pert_auc": acc.per_auc, "rank": acc.rank,
    })
    df["normalization"] = normalization
    df["method"] = method
    summary = {"dataset": dataset_name, "normalization": normalization, "method": method,
               "tau2": float(tau2), **summary}
    return InverseResult(results=df, summary=summary, roc=roc, prc=prc, method=method,
                         normalization=normalization, dataset_name=dataset_name,
                         tau2=float(tau2), scores=out)


def posterior_inverse_from_precomputed(
    dataset_dir, mode: str, method: str = "posterior",
    tau2: float | None = None, plateau: bool = True,
    negatives_per_pert: int = ROC_NEGATIVES_PER_PERT, seed: int = 0,
    batch: int = PERT_BATCH, return_scores: bool = False, progress: bool = True,
) -> InverseResult:
    """Run the fullH_diag posterior inverse on a preprocessed dataset directory.

    Requires the precompute artifacts written by :func:`cipher.preprocess_dataset`
    with ``save_mean_var=True`` (``Sigma_full_ridge.npy`` and a
    ``perturbation_stats.h5`` containing ``dx``, ``var_pert``, ``n_cells_pert`` and
    the ``n_control`` attribute).

    ``method`` is ``"posterior"`` (default) or ``"pip"``.  ``tau2`` is fit by
    empirical Bayes when ``None``.
    """
    import h5py
    from .io import load_precomputed

    dataset_dir = Path(dataset_dir)
    pc = load_precomputed(dataset_dir, mode)
    Sigma = np.asarray(pc.sigma(mmap=False), dtype=np.float64)
    stats_path = dataset_dir / "normalizations" / mode / "perturbation_stats.h5"
    with h5py.File(stats_path, "r") as h5:
        nu = np.asarray(h5["n_cells_pert"][:], dtype=np.float64).reshape(-1)
        var_pert = np.asarray(h5["var_pert"][:], dtype=np.float64) if "var_pert" in h5 else None
        control_var = np.asarray(h5["control_var"][:], dtype=np.float64) if "control_var" in h5 else None
        n0 = _read_n0(h5)
    if n0 is None:
        raise ValueError(f"Could not find the control-cell count (n_control) in {stats_path}; "
                         "re-run preprocess_dataset to record it.")

    model = build_model(Sigma, var_pert, n0, nu, control_var=control_var)
    ds_name = dataset_dir.name
    if tau2 is None:
        tau2 = fit_tau2(model, pc.dx, batch=batch, plateau=plateau)["tau2_use"]
    return _run_scoring(model, pc.dx, pc.target_gene_indices, method, float(tau2),
                        pc.gene_names, pc.perturbations, negatives_per_pert, seed, batch,
                        ds_name, mode, return_scores, progress)


def _read_n0(h5):
    for name in ("n_control", "n_cells_control", "n_control_cells", "control_n", "n0"):
        if name in h5.attrs:
            v = float(h5.attrs[name])
            if np.isfinite(v) and v > 0:
                return v
        if name in h5:
            arr = np.asarray(h5[name][()]).reshape(-1)
            if arr.size and np.isfinite(arr[0]) and arr[0] > 0:
                return float(arr[0])
    return None


def posterior_inverse_prediction(
    data, normalization: str = "log1p", method: str = "posterior",
    tau2: float | None = None, plateau: bool = True,
    negatives_per_pert: int = ROC_NEGATIVES_PER_PERT, seed: int = 0,
    batch: int = PERT_BATCH, cov_max_cells: int | None = 10000,
    max_perturbations: int | None = None, return_scores: bool = False,
    progress: bool = True, **load_kwargs,
) -> InverseResult:
    """Run the fullH_diag posterior inverse in memory from an ``.h5ad`` / Dataset.

    Computes the covariance, per-perturbation ``dx`` and gene-wise variances on the
    fly (moderate datasets; for large ones preprocess to disk and use
    :func:`posterior_inverse_from_precomputed`).
    """
    from .data import Dataset, load_dataset
    from .normalize import normalize_matrix, library_size, fit_pflog_alpha, mean_var
    from .covariance import compute_covariance
    from .utils import stable_seed

    ds = data if isinstance(data, Dataset) else load_dataset(data, **load_kwargs)
    ds_name = ds.name
    control_raw = ds.control_matrix(dense=True)
    pseudocount = None
    if normalization == "pflog":
        _, pseudocount, _, _ = fit_pflog_alpha(control_raw, ds.gene_names)

    def _norm(X):
        return normalize_matrix(X, normalization, libsize=library_size(X), pseudocount=pseudocount)

    control_norm = _norm(control_raw)
    control_mean = control_norm.mean(axis=0)
    control_var = control_norm.var(axis=0, ddof=1)
    n0 = float(control_norm.shape[0])
    cov_norm = control_norm
    if cov_max_cells is not None and control_norm.shape[0] > cov_max_cells:
        rng = np.random.default_rng(stable_seed(seed, f"{ds_name}:{normalization}"))
        sel = np.sort(rng.choice(control_norm.shape[0], cov_max_cells, replace=False))
        cov_norm = control_norm[sel]
    Sigma = compute_covariance(cov_norm)

    perts = ds.perturbations
    tgi = ds.target_gene_indices
    if max_perturbations is not None:
        perts, tgi = perts[:max_perturbations], np.asarray(tgi)[:max_perturbations]
    n_pert, p = len(perts), ds.n_genes
    dx = np.empty((n_pert, p), dtype=np.float64)
    var_pert = np.empty((n_pert, p), dtype=np.float64)
    nu = np.empty(n_pert, dtype=np.float64)
    it = enumerate(perts)
    if progress:
        it = tqdm(list(it), desc=f"inverse-stats:{ds_name}:{normalization}", leave=False)
    for i, pert in it:
        Yp = _norm(ds.perturbation_matrix(pert, dense=True))
        nu[i] = Yp.shape[0]
        m, v = mean_var(Yp)
        dx[i] = m - control_mean
        var_pert[i] = v

    model = build_model(Sigma, var_pert, n0, nu, control_var=control_var)
    if tau2 is None:
        tau2 = fit_tau2(model, dx, batch=batch, plateau=plateau)["tau2_use"]
    return _run_scoring(model, dx, np.asarray(tgi, dtype=np.int64), method, float(tau2),
                        ds.gene_names, perts, negatives_per_pert, seed, batch,
                        ds_name, normalization, return_scores, progress)
