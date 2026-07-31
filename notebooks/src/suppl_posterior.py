"""Shared engine for the tau2-selection supplement (analytic Gaussian CIPHER posterior).

A notebook-only helper for reproducing the supplementary "choosing tau2" figure.
NOT part of the installable ``cipher`` package.

The tau2-sweep engine works on the model:

    y = Sigma u + eps,  eps ~ N(0, H),  u ~ N(0, tau2 I)
    C_tau = H + tau2 Sigma Sigma^T

Whitened by H = L L^T (z = L^-1 y, W = L^-1 Sigma, B = W W^T = Q diag(lam) Q^T)
so log p(y | tau2) and the posterior mean/R2 are cheap to sweep over tau2.

This module keeps that shared engine (``prepare_whitened_eigendecomposition``,
``evaluate_tau2_grid``, ``plot_tau_sweep`` and helpers) and both dataset compute
pipelines that produce Sigma / H / delta_x from cells:

  * KRAS (pancreatic) pipeline -- canonical names
    (``build_selected_matrices_from_cells`` + helpers).
  * Melanoma (GSE233766) pipeline -- ``*_mel`` names, because its
    ``compute_covariance`` / ``build_H_from_sample_means`` / ``compute_de_scores``
    genuinely differ from the KRAS versions (different shrinkage defaults,
    ``mode=`` vs ``approx=`` signature, different DE columns / selection rule).
    The two pipelines differ only in the ``def`` names and the matching
    sibling-call names so both can coexist in one module.

The compute functions read their configuration from module-level globals
(H5AD_PATH, BASE_OUTDIR/OUTDIR, TOP_N_DE, ...); the notebook sets those on this
module (``sp.NAME = ...``) before each section runs.
"""
from __future__ import annotations


# --- library imports required by the functions below
#     (resolved at call time so placement after the docstring is sufficient) ---
import os, re, glob, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Patch
from matplotlib import gridspec
from scipy.sparse import issparse, csr_matrix
from scipy.stats import wilcoxon, ttest_rel, ks_2samp, mannwhitneyu, pearsonr, spearmanr
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
except Exception:
    roc_auc_score = average_precision_score = roc_curve = precision_recall_curve = None
try:
    import scanpy as sc
except Exception:
    sc = None
try:
    import anndata as ad
except Exception:
    ad = None
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *a, **k): return x
# --- end fixup imports ---

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import issparse
from scipy.linalg import cho_factor, cho_solve, solve_triangular, eigh
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

try:
    import scanpy as sc
except Exception:  # pragma: no cover
    sc = None

# Patch old/broken scanpy.
if sc is not None:
    try:
        sc.read_h5ad
    except AttributeError:
        print("[patch] scanpy has no read_h5ad; using anndata.read_h5ad instead")
        sc.read_h5ad = ad.read_h5ad


# ============================================================
# SWEEP-ENGINE CONFIG (shared defaults; notebook may override)
# ============================================================

SYMMETRIZE = True
CHOLESKY_JITTER_REL = 1e-10
CHOLESKY_MAX_TRIES = 8
PLATEAU_DELTA_LOGML = 1.0
COMPUTE_POSTERIOR_DIAGNOSTICS = True
DPI = 300


# ============================================================
# SHARED SWEEP ENGINE
# ============================================================

def load_first_existing(base_dir, names):
    """
    Load the first existing .npy file among candidate names.
    """
    base_dir = Path(base_dir)

    for name in names:
        path = base_dir / name
        if path.exists():
            arr = np.load(path, allow_pickle=True)
            print(f"[load] {name}: shape={getattr(arr, 'shape', None)}")
            return arr, path

    raise FileNotFoundError(
        "None of these files were found in "
        f"{base_dir}:\n  " + "\n  ".join(names)
    )


def maybe_load_gene_names(base_dir):
    """
    Optional: load gene names if present.
    Not required for the tau sweep.
    """
    base_dir = Path(base_dir)

    candidates = [
        "selected_genes.npy",
        "selected_gene_names.npy",
        "gene_names.npy",
    ]

    for name in candidates:
        path = base_dir / name
        if path.exists():
            g = np.load(path, allow_pickle=True)
            g = np.asarray(g, dtype=str)
            print(f"[load] {name}: n={len(g)}")
            return g

    # TSV fallback
    tsv_candidates = [
        "selected_gene_names.tsv",
        "posterior_summary_selected.tsv",
        "posterior_summary.tsv",
    ]

    for name in tsv_candidates:
        path = base_dir / name
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            if "gene" in df.columns:
                g = df["gene"].astype(str).values
                print(f"[load] {name}: n={len(g)}")
                return g

    print("[load] no gene-name file found; continuing without gene names")
    return None


def symmetrize(A):
    return 0.5 * (A + A.T)


def r2_from_pred(y, yhat, eps=1e-12):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    return 1.0 - np.sum((y - yhat) ** 2) / (np.sum(y ** 2) + eps)


def safe_cholesky_factor(A, name="matrix"):
    """
    Cholesky with increasing diagonal jitter if needed.
    """
    A = np.asarray(A, dtype=np.float64)

    if SYMMETRIZE:
        A = symmetrize(A)

    p = A.shape[0]
    diag = np.diag(A)
    scale = float(np.nanmean(np.abs(diag))) if len(diag) else 1.0
    scale = max(scale, 1e-12)

    last_err = None

    for k in range(CHOLESKY_MAX_TRIES):
        jitter = CHOLESKY_JITTER_REL * scale * (10.0 ** k)
        Aj = A + jitter * np.eye(p)

        try:
            cho = cho_factor(Aj, lower=True, check_finite=False)
            if k > 0:
                print(f"[cholesky] {name}: succeeded with jitter={jitter:.3e}")
            return cho, jitter
        except Exception as e:
            last_err = e

    raise np.linalg.LinAlgError(
        f"Cholesky failed for {name} after {CHOLESKY_MAX_TRIES} tries. "
        f"Last error: {repr(last_err)}"
    )


def prepare_whitened_eigendecomposition(Sigma, H, y):
    """
    Prepare all objects needed to evaluate log p(y | tau2) quickly.

    Model:
        C_tau = H + tau2 Sigma Sigma^T

    Whiten by H:
        H = L L^T
        z = L^{-1} y
        W = L^{-1} Sigma
        B = W W^T = Q diag(lam) Q^T
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if SYMMETRIZE:
        Sigma = symmetrize(Sigma)
        H = symmetrize(H)

    p = len(y)

    if Sigma.shape != (p, p):
        raise ValueError(f"Sigma shape {Sigma.shape} does not match y length {p}.")
    if H.shape != (p, p):
        raise ValueError(f"H shape {H.shape} does not match y length {p}.")

    choH, H_jitter = safe_cholesky_factor(H, name="H")
    L = choH[0]

    # Because cho_factor(lower=True) stores L in the lower triangle.
    # solve_triangular ignores the irrelevant upper entries when lower=True.
    z = solve_triangular(L, y, lower=True, check_finite=False)
    W = solve_triangular(L, Sigma, lower=True, check_finite=False)

    B = W @ W.T
    B = symmetrize(B)

    print("[eig] computing eigenvalues of whitened Sigma Sigma^T ...")
    lam, Q = eigh(B, check_finite=False)

    # Numerical cleanup: tiny negative eigenvalues can occur from roundoff.
    lam = np.asarray(lam, dtype=np.float64)
    lam[lam < 0] = np.maximum(lam[lam < 0], -1e-10)
    lam = np.clip(lam, 0.0, None)

    zQ = Q.T @ z

    logdetH = 2.0 * np.sum(np.log(np.diag(L) + 1e-300))

    # Useful for posterior mean:
    # mu(tau2) = tau2 * W^T Q diag(1 / (1 + tau2 lam)) Q^T z
    if COMPUTE_POSTERIOR_DIAGNOSTICS:
        WTQ = W.T @ Q
    else:
        WTQ = None

    prep = {
        "Sigma": Sigma,
        "H": H,
        "y": y,
        "choH": choH,
        "H_jitter": H_jitter,
        "L": L,
        "z": z,
        "W": W,
        "B_eigvals": lam,
        "B_eigvecs": Q,
        "zQ": zQ,
        "logdetH": logdetH,
        "WTQ": WTQ,
    }

    print("[prep]")
    print(f"  p genes: {p}")
    print(f"  H jitter: {H_jitter:.3e}")
    print(f"  logdet(H): {logdetH:.6g}")
    print(f"  whitened eig min/median/max: {lam.min():.3e}, {np.median(lam):.3e}, {lam.max():.3e}")

    return prep


def evaluate_tau2_grid(prep, tau2_grid, include_tau2_zero=True):
    """
    Evaluate log marginal likelihood and diagnostics over tau2.
    """
    Sigma = prep["Sigma"]
    y = prep["y"]
    lam = prep["B_eigvals"]
    zQ = prep["zQ"]
    logdetH = prep["logdetH"]
    WTQ = prep["WTQ"]

    p = len(y)
    tau2_grid = np.asarray(tau2_grid, dtype=np.float64)
    tau2_grid = tau2_grid[np.isfinite(tau2_grid)]
    tau2_grid = tau2_grid[tau2_grid > 0]
    tau2_grid = np.unique(np.sort(tau2_grid))

    rows = []

    def eval_one(tau2):
        tau2 = float(tau2)

        denom = 1.0 + tau2 * lam

        logdetC = logdetH + np.sum(np.log1p(tau2 * lam))
        quad = np.sum((zQ ** 2) / denom)

        log_marginal = -0.5 * (
            p * np.log(2.0 * np.pi)
            + logdetC
            + quad
        )

        row = {
            "tau2": tau2,
            "tau": np.sqrt(tau2),
            "log10_tau2": np.log10(tau2) if tau2 > 0 else -np.inf,
            "log10_tau": np.log10(np.sqrt(tau2)) if tau2 > 0 else -np.inf,
            "log_marginal": log_marginal,
            "logdetC": logdetC,
            "quad": quad,
            "mean_log1p_tau2_lambda": np.mean(np.log1p(tau2 * lam)),
            "max_tau2_lambda": tau2 * np.max(lam),
            "median_tau2_lambda": tau2 * np.median(lam),
        }

        if COMPUTE_POSTERIOR_DIAGNOSTICS:
            # posterior mean:
            # mu = tau2 * W^T Q [zQ / (1 + tau2 lam)]
            coeff = zQ / denom
            mu = tau2 * (WTQ @ coeff)

            yhat = Sigma @ mu

            row.update({
                "r2": r2_from_pred(y, yhat),
                "mu_l2": float(np.linalg.norm(mu)),
                "mu_l1": float(np.sum(np.abs(mu))),
                "mu_max_abs": float(np.max(np.abs(mu))),
                "yhat_l2": float(np.linalg.norm(yhat)),
                "resid_l2": float(np.linalg.norm(y - yhat)),
                "y_l2": float(np.linalg.norm(y)),
            })

        return row

    if include_tau2_zero:
        # Exact tau2 -> 0 model:
        # C = H.
        # logdetC = logdetH.
        # quad = z^T z = sum zQ^2.
        quad0 = float(np.sum(zQ ** 2))
        log_marginal0 = -0.5 * (
            p * np.log(2.0 * np.pi)
            + logdetH
            + quad0
        )

        row0 = {
            "tau2": 0.0,
            "tau": 0.0,
            "log10_tau2": -np.inf,
            "log10_tau": -np.inf,
            "log_marginal": log_marginal0,
            "logdetC": logdetH,
            "quad": quad0,
            "mean_log1p_tau2_lambda": 0.0,
            "max_tau2_lambda": 0.0,
            "median_tau2_lambda": 0.0,
        }

        if COMPUTE_POSTERIOR_DIAGNOSTICS:
            yhat0 = np.zeros_like(y)
            row0.update({
                "r2": r2_from_pred(y, yhat0),
                "mu_l2": 0.0,
                "mu_l1": 0.0,
                "mu_max_abs": 0.0,
                "yhat_l2": 0.0,
                "resid_l2": float(np.linalg.norm(y)),
                "y_l2": float(np.linalg.norm(y)),
            })

        rows.append(row0)

    for i, tau2 in enumerate(tau2_grid):
        if (i + 1) % 50 == 0 or i == 0 or i == len(tau2_grid) - 1:
            print(f"[sweep] {i + 1}/{len(tau2_grid)} tau2={tau2:.3e}")

        rows.append(eval_one(tau2))

    df = pd.DataFrame(rows)

    max_logml = df["log_marginal"].max()
    df["delta_log_marginal"] = df["log_marginal"] - max_logml
    df["relative_likelihood"] = np.exp(np.maximum(df["delta_log_marginal"], -745.0))

    # Finite-difference slope with respect to log10(tau2), excluding tau2=0.
    df["d_logml_d_log10_tau2"] = np.nan
    positive = df["tau2"].values > 0
    if positive.sum() >= 3:
        x = df.loc[positive, "log10_tau2"].values
        ylog = df.loc[positive, "log_marginal"].values
        slope = np.gradient(ylog, x)
        df.loc[positive, "d_logml_d_log10_tau2"] = slope

    return df


def plot_tau_sweep(df, outdir):
    outdir = Path(outdir)

    df_pos = df[df["tau2"] > 0].copy()

    best = df.loc[df["log_marginal"].idxmax()].copy()
    best_tau2 = float(best["tau2"])
    best_tau = float(best["tau"])
    best_logml = float(best["log_marginal"])

    plateau = df[df["delta_log_marginal"] >= -PLATEAU_DELTA_LOGML].copy()

    if len(plateau) > 0:
        plateau_pos = plateau[plateau["tau2"] > 0]
        if len(plateau_pos) > 0:
            plateau_min_tau2 = plateau_pos["tau2"].min()
            plateau_max_tau2 = plateau_pos["tau2"].max()
        else:
            plateau_min_tau2 = 0.0
            plateau_max_tau2 = 0.0
    else:
        plateau_min_tau2 = np.nan
        plateau_max_tau2 = np.nan

    # ------------------------------------------------------------
    # 1) Absolute log marginal
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df_pos["tau2"], df_pos["log_marginal"], lw=2)

    if best_tau2 > 0:
        ax.axvline(best_tau2, ls="--", lw=1)
    ax.scatter([best_tau2 if best_tau2 > 0 else df_pos["tau2"].min()], [best_logml], s=60, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau^2$")
    ax.set_ylabel(r"$\log p(\Delta x \mid \tau^2)$")
    ax.set_title("Marginal likelihood vs prior variance")

    txt = (
        f"best tau2 = {best_tau2:.3e}\n"
        f"best tau  = {best_tau:.3e}\n"
        f"best logML = {best_logml:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.tight_layout()
    plt.savefig(outdir / "tau2_log_marginal.png", dpi=DPI)
    plt.savefig(outdir / "tau2_log_marginal.svg")
    plt.show()

    # ------------------------------------------------------------
    # 2) Relative log marginal
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df_pos["tau2"], df_pos["delta_log_marginal"], lw=2)
    ax.axhline(0, lw=1)
    ax.axhline(-PLATEAU_DELTA_LOGML, ls="--", lw=1)

    if best_tau2 > 0:
        ax.axvline(best_tau2, ls="--", lw=1)

    if np.isfinite(plateau_min_tau2) and plateau_min_tau2 > 0 and plateau_max_tau2 > 0:
        ax.axvspan(plateau_min_tau2, plateau_max_tau2, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau^2$")
    ax.set_ylabel(r"$\log p(\Delta x \mid \tau^2) - \max$")
    ax.set_title("Relative marginal likelihood")

    plt.tight_layout()
    plt.savefig(outdir / "tau2_relative_log_marginal.png", dpi=DPI)
    plt.savefig(outdir / "tau2_relative_log_marginal.svg")
    plt.show()

    # ------------------------------------------------------------
    # 3) Relative likelihood on [0, 1]
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df_pos["tau2"], df_pos["relative_likelihood"], lw=2)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)

    if best_tau2 > 0:
        ax.axvline(best_tau2, ls="--", lw=1)

    ax.set_xlabel(r"$\tau^2$")
    ax.set_ylabel("relative likelihood")
    ax.set_title("Relative evidence scale")

    plt.tight_layout()
    plt.savefig(outdir / "tau2_relative_likelihood.png", dpi=DPI)
    plt.savefig(outdir / "tau2_relative_likelihood.svg")
    plt.show()

    # ------------------------------------------------------------
    # 4) Slope wrt log tau2
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df_pos["tau2"], df_pos["d_logml_d_log10_tau2"], lw=2)
    ax.axhline(0, lw=1)

    if best_tau2 > 0:
        ax.axvline(best_tau2, ls="--", lw=1)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau^2$")
    ax.set_ylabel(r"$d\log p / d\log_{10}\tau^2$")
    ax.set_title("Evidence slope")

    plt.tight_layout()
    plt.savefig(outdir / "tau2_log_marginal_slope.png", dpi=DPI)
    plt.savefig(outdir / "tau2_log_marginal_slope.svg")
    plt.show()

    # ------------------------------------------------------------
    # 5) Same plot but x-axis tau, not tau2
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df_pos["tau"], df_pos["delta_log_marginal"], lw=2)
    ax.axhline(0, lw=1)
    ax.axhline(-PLATEAU_DELTA_LOGML, ls="--", lw=1)

    if best_tau > 0:
        ax.axvline(best_tau, ls="--", lw=1)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\log p(\Delta x \mid \tau^2) - \max$")
    ax.set_title("Relative marginal likelihood vs tau")

    plt.tight_layout()
    plt.savefig(outdir / "tau_relative_log_marginal.png", dpi=DPI)
    plt.savefig(outdir / "tau_relative_log_marginal.svg")
    plt.show()

    # ------------------------------------------------------------
    # 6) Posterior diagnostics if present
    # ------------------------------------------------------------
    if "r2" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(df_pos["tau2"], df_pos["r2"], lw=2)

        if best_tau2 > 0:
            ax.axvline(best_tau2, ls="--", lw=1)

        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau^2$")
        ax.set_ylabel(r"$R^2$ of $\Sigma\hat u$")
        ax.set_title("Posterior fit vs tau2")

        plt.tight_layout()
        plt.savefig(outdir / "tau2_posterior_r2.png", dpi=DPI)
        plt.savefig(outdir / "tau2_posterior_r2.svg")
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(df_pos["tau2"], df_pos["mu_l2"], lw=2, label=r"$||\hat u||_2$")
        ax.plot(df_pos["tau2"], df_pos["mu_max_abs"], lw=2, label=r"$\max |\hat u_g|$")

        if best_tau2 > 0:
            ax.axvline(best_tau2, ls="--", lw=1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\tau^2$")
        ax.set_ylabel("posterior force scale")
        ax.set_title("Posterior force size vs tau2")
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(outdir / "tau2_posterior_force_scale.png", dpi=DPI)
        plt.savefig(outdir / "tau2_posterior_force_scale.svg")
        plt.show()

    return {
        "best_tau2": best_tau2,
        "best_tau": best_tau,
        "best_log_marginal": best_logml,
        "plateau_min_tau2": plateau_min_tau2,
        "plateau_max_tau2": plateau_max_tau2,
    }


# ============================================================
# SHARED PIPELINE HELPERS (identical for both pipelines)
# ============================================================

def to_dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}\nPWD: {os.getcwd()}")
    print(f"[OK] Found file: {path}")


def normalize_gene_list(genes):
    out = []
    seen = set()

    for g in genes:
        gg = str(g).strip().upper()
        if gg and gg not in seen:
            out.append(gg)
            seen.add(gg)

    return out


# ============================================================
# KRAS (PANCREATIC) COMPUTE PIPELINE
# ============================================================

def drop_housekeeping_prefixes(
    var_names,
    bad_prefixes=("RPL", "RPS", "MT-", "MT.", "HSP", "HSP90", "EIF"),
):
    """
    Preserves the uploaded script's behavior.
    It checks startswith on raw var_names.
    """
    names = np.asarray(var_names, dtype=str)
    keep = np.ones(len(names), dtype=bool)

    for p in bad_prefixes:
        keep &= ~np.char.startswith(names, p)

    return keep


def filter_by_expression_and_variance_percentile(
    adata,
    naive_mask,
    min_cells_frac=0.01,
    min_expr=1.0,
    hi_quantile=0.90,
    var_drop_q=1.0,
    filter_subsample_cells=0,
    seed=0,
):
    rng = np.random.default_rng(seed)

    if filter_subsample_cells and adata.n_obs > filter_subsample_cells:
        idx = rng.choice(
            np.arange(adata.n_obs),
            size=int(filter_subsample_cells),
            replace=False,
        )
        adata_sub = adata[idx].copy()
        nm = np.asarray(naive_mask)[idx]
    else:
        adata_sub = adata
        nm = np.asarray(naive_mask)

    # RAW DATA
    X = to_dense(adata_sub.X).astype(np.float64)

    frac_on = np.mean(X >= min_expr, axis=0)
    q_hi = np.quantile(X, hi_quantile, axis=0)

    keep_expr = (frac_on >= min_cells_frac) | (q_hi >= min_expr)

    Xn = X[nm]
    vars_ = Xn.var(axis=0)

    if np.any(keep_expr):
        var_cut = np.quantile(vars_[keep_expr], var_drop_q)
        keep_var = vars_ <= var_cut
    else:
        keep_var = np.ones_like(keep_expr, dtype=bool)

    keep = keep_expr & keep_var

    print(f"[filter] kept {keep.sum()} / {len(keep)} genes after expression/variance filtering")

    if keep.sum() == 0:
        raise ValueError("All genes removed by filtering.")

    return adata[:, keep].copy()


def compute_covariance(X, shrinkage=1e-3):
    """
    RAW data covariance:
        C = centered X covariance
        C += shrinkage * mean(diag(C)) * I
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)

    C = (Xc.T @ Xc) / max(1, X.shape[0] - 1)
    C += float(shrinkage) * np.eye(C.shape[0]) * (np.mean(np.diag(C)) + 1e-12)

    if SYMMETRIZE:
        C = symmetrize(C)

    return C


def build_H_from_sample_means(
    X0,
    X1,
    shrinkage=1e-4,
    ridge=1e-6,
    approx="diag",
):
    """
    Same H convention as uploaded pancreatic script.

    S0 = cov(X0)
    S1 = cov(X1)

    full:
        H = S0/n0 + S1/n1

    diag:
        H = S0/n0 + diag(diag(S1))/n1
        NOTE: this keeps full control covariance.

    naive:
        H = (1/n0 + 1/n1) S0
    """
    X0 = np.asarray(X0, dtype=np.float64)
    X1 = np.asarray(X1, dtype=np.float64)

    n0 = X0.shape[0]
    n1 = X1.shape[0]

    print("[H] n0,n1", n0, n1)

    S0 = compute_covariance(X0, shrinkage=shrinkage)
    S1 = compute_covariance(X1, shrinkage=shrinkage)

    if approx == "full":
        H = S0 / max(n0, 1) + S1 / max(n1, 1)

    elif approx == "diag":
        H = S0 / max(n0, 1) + np.diag(np.diag(S1)) / max(n1, 1)

    elif approx == "naive":
        H = (1.0 / max(n0, 1) + 1.0 / max(n1, 1)) * S0

    else:
        H = np.diag(np.ones(X0.shape[1]))

    scale = np.mean(np.diag(H)) + 1e-12
    H += ridge * scale * np.eye(H.shape[0])

    if SYMMETRIZE:
        H = symmetrize(H)

    return H, S0, S1


def compute_de_scores(X0, X1, gene_names, eps=1e-12):
    X0 = np.asarray(X0, dtype=np.float64)
    X1 = np.asarray(X1, dtype=np.float64)

    mean0 = X0.mean(axis=0)
    mean1 = X1.mean(axis=0)

    delta = mean1 - mean0

    with np.errstate(divide="ignore", invalid="ignore"):
        log2fc = np.log2((mean1 + 1e-10) / (mean0 + 1e-10))

    log2fc = np.nan_to_num(log2fc, nan=0.0, posinf=0.0, neginf=0.0)

    n0 = X0.shape[0]
    n1 = X1.shape[0]

    v0 = X0.var(axis=0, ddof=1)
    v1 = X1.var(axis=0, ddof=1)

    se = np.sqrt(v0 / max(n0, 1) + v1 / max(n1, 1)) + eps
    t_stat = delta / se
    t_stat = np.nan_to_num(t_stat, nan=0.0, posinf=0.0, neginf=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_val = ttest_ind(X1, X0, axis=0, equal_var=False)

    p_val = np.asarray(p_val, dtype=np.float64)
    p_val = np.nan_to_num(p_val, nan=1.0, posinf=1.0, neginf=1.0)

    reject_fdr, p_adj, _, _ = multipletests(
        p_val,
        alpha=FDR_ALPHA,
        method="fdr_bh",
    )

    de = pd.DataFrame({
        "gene": np.asarray(gene_names, dtype=str),
        "mean_cond0": mean0,
        "mean_cond1": mean1,
        "delta": delta,
        "log2fc": log2fc,
        "abs_log2fc": np.abs(log2fc),
        "t_stat": t_stat,
        "abs_t": np.abs(t_stat),
        "p_value": p_val,
        "p_adj": p_adj,
        "significant_fdr_0p05": reject_fdr.astype(int),
        "neglog10_p": -np.log10(np.maximum(p_val, 1e-300)),
        "neglog10_padj": -np.log10(np.maximum(p_adj, 1e-300)),
    })

    de = de.sort_values(
        ["significant_fdr_0p05", "abs_log2fc", "p_adj"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    de["rank"] = np.arange(1, len(de) + 1)
    de["gene_upper"] = de["gene"].astype(str).str.upper()

    return de


def build_selected_matrices_from_cells():
    check_file(H5AD_PATH)

    print(f"[run] h5ad = {H5AD_PATH}")
    print(f"[run] outdir = {BASE_OUTDIR}")

    adata = sc.read_h5ad(H5AD_PATH)
    adata.var_names_make_unique()

    print(f"[data] loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    if CONDITION_KEY not in adata.obs:
        raise KeyError(
            f"'{CONDITION_KEY}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    print(f"\n[data] available {CONDITION_KEY}:")
    print(pd.Series(adata.obs[CONDITION_KEY]).value_counts())

    adata = adata[
        adata.obs[CONDITION_KEY].isin([COND0, COND1])
    ].copy()

    m0 = np.asarray(adata.obs[CONDITION_KEY].values == COND0)
    m1 = np.asarray(adata.obs[CONDITION_KEY].values == COND1)

    print(f"\n[contrast] {COND1} - {COND0}")
    print(f"  {COND0}: {m0.sum()} cells")
    print(f"  {COND1}: {m1.sum()} cells")

    if m0.sum() < 5 or m1.sum() < 5:
        raise ValueError(
            f"Too few cells: {COND0}={m0.sum()}, {COND1}={m1.sum()}"
        )

    # --------------------------------------------------------
    # Filtering
    # --------------------------------------------------------
    adata = filter_by_expression_and_variance_percentile(
        adata,
        naive_mask=m0,
        min_cells_frac=MIN_CELLS_FRAC,
        min_expr=MIN_EXPR,
        hi_quantile=HI_QUANTILE,
        var_drop_q=VAR_DROP_Q,
        filter_subsample_cells=FILTER_SUBSAMPLE_CELLS,
        seed=SEED,
    )

    if DROP_HOUSEKEEPING:
        keep = drop_housekeeping_prefixes(adata.var_names)
        print(f"[filter] housekeeping-prefix filter kept {keep.sum()} / {len(keep)} genes")
        adata = adata[:, keep].copy()

    m0 = np.asarray(adata.obs[CONDITION_KEY].values == COND0)
    m1 = np.asarray(adata.obs[CONDITION_KEY].values == COND1)

    # RAW DATA
    X0 = to_dense(adata[m0].X).astype(np.float64)
    X1 = to_dense(adata[m1].X).astype(np.float64)

    gene_names_all = np.asarray(adata.var_names, dtype=str)
    gene_names_all_upper = np.char.upper(gene_names_all)

    print(f"\n[matrix after filtering]")
    print(f"  X0: {X0.shape}")
    print(f"  X1: {X1.shape}")

    # --------------------------------------------------------
    # DE ranking
    # --------------------------------------------------------
    de_df = compute_de_scores(X0, X1, gene_names_all)

    de_path = BASE_OUTDIR / "all_genes_de_ranking.tsv"
    de_df.to_csv(de_path, sep="\t", index=False)
    print(f"[saved] {de_path}")

    sig_de = de_df.loc[
        (de_df["p_adj"] < FDR_ALPHA)
        & (de_df["abs_log2fc"] >= float(MIN_ABS_LOG2FC))
    ].copy()

    sig_de = sig_de.sort_values(
        ["abs_log2fc", "p_adj"],
        ascending=[False, True],
    ).reset_index(drop=True)

    sig_de["selected_rank"] = np.arange(1, len(sig_de) + 1)

    selected_de = sig_de.head(min(TOP_N_DE, len(sig_de))).copy()

    if len(selected_de) == 0:
        raise ValueError(
            f"No genes passed selection: p_adj < {FDR_ALPHA} and "
            f"abs(log2FC) >= {MIN_ABS_LOG2FC}."
        )

    selected_genes = selected_de["gene"].values
    selected_set_upper = set(g.upper() for g in selected_genes)

    selected_de_path = BASE_OUTDIR / "selected_de_table.tsv"
    selected_de.to_csv(selected_de_path, sep="\t", index=False)
    print(f"[saved] {selected_de_path}")

    sig_de_path = BASE_OUTDIR / "all_sig_highlfc_de_table.tsv"
    sig_de.to_csv(sig_de_path, sep="\t", index=False)
    print(f"[saved] {sig_de_path}")

    print(f"\n[DE]")
    print(f"  total genes tested: {len(de_df)}")
    print(f"  FDR < {FDR_ALPHA} and |log2FC| >= {MIN_ABS_LOG2FC}: {len(sig_de)}")
    print(f"  selected top: {len(selected_de)}")

    print("\n[top selected genes]")
    print(selected_de[[
        "selected_rank",
        "gene",
        "mean_cond0",
        "mean_cond1",
        "delta",
        "log2fc",
        "abs_log2fc",
        "t_stat",
        "p_adj",
    ]].head(30).to_string(index=False))

    # --------------------------------------------------------
    # Tracked-gene status
    # --------------------------------------------------------
    genes_to_check = normalize_gene_list(GENES_TO_CHECK)

    rows = []
    for g in genes_to_check:
        present_after_filter = g in set(gene_names_all_upper)
        in_selected = g in selected_set_upper

        full_hit = de_df.loc[de_df["gene_upper"] == g]
        sel_hit = sig_de.loc[sig_de["gene_upper"] == g]

        rows.append({
            "gene": g,
            "present_after_initial_filtering": present_after_filter,
            "in_selected_set": in_selected,
            "full_rank": int(full_hit["rank"].iloc[0]) if len(full_hit) > 0 else np.nan,
            "selected_rank": int(sel_hit["selected_rank"].iloc[0]) if len(sel_hit) > 0 else np.nan,
            "delta": float(full_hit["delta"].iloc[0]) if len(full_hit) > 0 else np.nan,
            "log2fc": float(full_hit["log2fc"].iloc[0]) if len(full_hit) > 0 else np.nan,
            "abs_log2fc": float(full_hit["abs_log2fc"].iloc[0]) if len(full_hit) > 0 else np.nan,
            "t_stat": float(full_hit["t_stat"].iloc[0]) if len(full_hit) > 0 else np.nan,
            "p_value": float(full_hit["p_value"].iloc[0]) if len(full_hit) > 0 else np.nan,
            "p_adj": float(full_hit["p_adj"].iloc[0]) if len(full_hit) > 0 else np.nan,
        })

    gene_status_df = pd.DataFrame(rows)

    gene_status_path = BASE_OUTDIR / "tracked_gene_status.tsv"
    gene_status_df.to_csv(gene_status_path, sep="\t", index=False)
    print(f"[saved] {gene_status_path}")

    print("\n[tracked genes]")
    print(gene_status_df.to_string(index=False))

    # --------------------------------------------------------
    # Restrict to selected genes, preserving selected-DE order
    # --------------------------------------------------------
    selected_mask = np.isin(gene_names_all, selected_genes)

    X0_sel_unordered = X0[:, selected_mask]
    X1_sel_unordered = X1[:, selected_mask]
    gene_names_unordered = gene_names_all[selected_mask]

    name_to_idx = {g: i for i, g in enumerate(gene_names_unordered)}
    order_de = [name_to_idx[g] for g in selected_genes if g in name_to_idx]

    X0_sel = X0_sel_unordered[:, order_de]
    X1_sel = X1_sel_unordered[:, order_de]
    gene_names = gene_names_unordered[order_de]

    print(f"\n[selected matrix]")
    print(f"  X0_sel: {X0_sel.shape}")
    print(f"  X1_sel: {X1_sel.shape}")
    print(f"  selected genes: {len(gene_names)}")

    # --------------------------------------------------------
    # y, log2fc, Sigma, H from cells
    # --------------------------------------------------------
    y = X1_sel.mean(axis=0) - X0_sel.mean(axis=0)

    de_exact = de_df.set_index("gene", drop=False)
    log2fc = np.asarray(
        [float(de_exact.loc[g, "log2fc"]) for g in gene_names],
        dtype=np.float64,
    )

    Sigma = compute_covariance(
        X0_sel,
        shrinkage=SIGMA_SHRINKAGE,
    )

    H, S0, S1 = build_H_from_sample_means(
        X0_sel,
        X1_sel,
        shrinkage=H_SHRINKAGE,
        ridge=H_RIDGE,
        approx=H_MODE,
    )

    print("\n[computed from cells]")
    print(f"  y:      {y.shape}")
    print(f"  Sigma:  {Sigma.shape}")
    print(f"  H:      {H.shape}")
    print(f"  H_MODE: {H_MODE}")
    print(f"  ||y||:  {np.linalg.norm(y):.6g}")
    print(f"  diag Sigma mean/min/max: {np.mean(np.diag(Sigma)):.3e}, {np.min(np.diag(Sigma)):.3e}, {np.max(np.diag(Sigma)):.3e}")
    print(f"  diag H mean/min/max:     {np.mean(np.diag(H)):.3e}, {np.min(np.diag(H)):.3e}, {np.max(np.diag(H)):.3e}")

    # Save matrices exactly used
    np.save(BASE_OUTDIR / "selected_gene_names.npy", gene_names)
    np.save(BASE_OUTDIR / "selected_log2fc.npy", log2fc)
    np.save(BASE_OUTDIR / "delta_x_selected.npy", y)
    np.save(BASE_OUTDIR / "Sigma_selected_from_cells.npy", Sigma)
    np.save(BASE_OUTDIR / "H_selected_from_cells.npy", H)
    np.save(BASE_OUTDIR / "S0_selected_from_cells.npy", S0)
    np.save(BASE_OUTDIR / "S1_selected_from_cells.npy", S1)

    pd.DataFrame({"gene": gene_names}).to_csv(
        BASE_OUTDIR / "selected_gene_names.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame({
        "gene": gene_names,
        "delta_x": y,
        "log2fc": log2fc,
    }).to_csv(
        BASE_OUTDIR / "selected_gene_delta_and_lfc.tsv",
        sep="\t",
        index=False,
    )

    run_config = {
        "h5ad_path": H5AD_PATH,
        "condition_key": CONDITION_KEY,
        "cond0": COND0,
        "cond1": COND1,
        "top_n_de": TOP_N_DE,
        "fdr_alpha": FDR_ALPHA,
        "min_abs_log2fc": MIN_ABS_LOG2FC,
        "drop_housekeeping": DROP_HOUSEKEEPING,
        "min_cells_frac": MIN_CELLS_FRAC,
        "min_expr": MIN_EXPR,
        "hi_quantile": HI_QUANTILE,
        "var_drop_q": VAR_DROP_Q,
        "filter_subsample_cells": FILTER_SUBSAMPLE_CELLS,
        "seed": SEED,
        "Sigma_shrinkage": SIGMA_SHRINKAGE,
        "H_shrinkage": H_SHRINKAGE,
        "H_ridge": H_RIDGE,
        "H_mode": H_MODE,
        "n0": X0_sel.shape[0],
        "n1": X1_sel.shape[0],
        "n_selected_genes": len(gene_names),
    }

    pd.Series(run_config).to_csv(
        BASE_OUTDIR / "run_config_used_for_tau2_sweep.tsv",
        sep="\t",
        header=False,
    )

    return {
        "de_df": de_df,
        "sig_de": sig_de,
        "selected_de": selected_de,
        "gene_status_df": gene_status_df,
        "gene_names": gene_names,
        "log2fc": log2fc,
        "X0_sel": X0_sel,
        "X1_sel": X1_sel,
        "y": y,
        "Sigma": Sigma,
        "H": H,
        "S0": S0,
        "S1": S1,
    }


# ============================================================
# MELANOMA (GSE233766) COMPUTE PIPELINE
# (only def names + sibling-call names carry a *_mel suffix so the two
# divergent pipelines coexist in one module -- bodies are otherwise unchanged)
# ============================================================

def read_h5ad_robust(path):
    try:
        import anndata as ad
        return ad.read_h5ad(path)
    except Exception as e1:
        try:
            import scanpy as sc
            return sc.read(path)
        except Exception as e2:
            raise RuntimeError(
                "Could not read .h5ad file.\n"
                f"anndata error: {repr(e1)}\n"
                f"scanpy error: {repr(e2)}"
            )


def drop_bad_gene_prefixes(
    gene_names,
    bad_prefixes=(
        "RPL", "RPS",
        "MT-", "MT.",
        "MTRNR", "MTRNR2L",
        "HSP", "HSP90",
        "EIF",
        "MALAT1",
    ),
):
    names = np.asarray(gene_names, dtype=str)
    keep = np.ones(len(names), dtype=bool)
    upper = np.char.upper(names)

    for p in bad_prefixes:
        keep &= ~np.char.startswith(upper, p.upper())

    return keep


def filter_genes_basic(
    adata,
    min_cells_frac=0.001,
    min_expr=0.01,
    min_mean=0.001,
    max_mean=np.inf,
    max_var_quantile=1.0,
    seed=0,
    filter_subsample_cells=0,
):
    rng = np.random.default_rng(seed)

    if filter_subsample_cells and adata.n_obs > filter_subsample_cells:
        idx = rng.choice(
            np.arange(adata.n_obs),
            size=int(filter_subsample_cells),
            replace=False,
        )
        ad = adata[idx].copy()
    else:
        ad = adata

    X = to_dense(ad.X).astype(np.float64)

    frac_on = np.mean(X >= min_expr, axis=0)
    mean = X.mean(axis=0)
    var = X.var(axis=0)

    keep = (
        (frac_on >= min_cells_frac)
        & (mean >= min_mean)
        & (mean <= max_mean)
    )

    if max_var_quantile < 1.0 and np.any(keep):
        var_cut = np.quantile(var[keep], max_var_quantile)
        keep &= var <= var_cut

    print(f"[filter] basic expression filter kept {keep.sum()} / {len(keep)} genes")

    if keep.sum() == 0:
        raise ValueError("All genes removed by expression filter.")

    return adata[:, keep].copy()


def compute_de_scores_mel(
    X0,
    X1,
    gene_names,
    logfc_pseudocount=1.0,
    eps=1e-12,
):
    X0 = np.asarray(X0, dtype=np.float64)
    X1 = np.asarray(X1, dtype=np.float64)
    gene_names = np.asarray(gene_names, dtype=str)

    mean0 = X0.mean(axis=0)
    mean1 = X1.mean(axis=0)
    delta = mean1 - mean0

    v0 = X0.var(axis=0, ddof=1)
    v1 = X1.var(axis=0, ddof=1)

    n0 = X0.shape[0]
    n1 = X1.shape[0]

    se = np.sqrt(v0 / max(n0, 1) + v1 / max(n1, 1)) + eps
    t_stat = delta / se
    t_stat = np.nan_to_num(t_stat, nan=0.0, posinf=0.0, neginf=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_value = ttest_ind(X1, X0, axis=0, equal_var=False)

    p_value = np.asarray(p_value, dtype=np.float64)
    p_value = np.nan_to_num(p_value, nan=1.0, posinf=1.0, neginf=1.0)

    _, p_adj, _, _ = multipletests(
        p_value,
        alpha=FDR_ALPHA,
        method="fdr_bh",
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        log2fc = np.log2(
            (mean1 + logfc_pseudocount)
            / (mean0 + logfc_pseudocount)
        )

    log2fc = np.nan_to_num(log2fc, nan=0.0, posinf=0.0, neginf=0.0)

    de = pd.DataFrame({
        "gene": gene_names,
        "gene_upper": np.char.upper(gene_names),
        "mean_cond0": mean0,
        "mean_cond1": mean1,
        "delta": delta,
        "abs_delta": np.abs(delta),
        "log2fc": log2fc,
        "abs_log2fc": np.abs(log2fc),
        "t_stat": t_stat,
        "abs_t": np.abs(t_stat),
        "p_value": p_value,
        "p_adj": p_adj,
        "neglog10_p": -np.log10(np.maximum(p_value, 1e-300)),
        "neglog10_padj": -np.log10(np.maximum(p_adj, 1e-300)),
        "fdr_sig": (p_adj < FDR_ALPHA).astype(int),
    })

    de = de.sort_values(
        ["fdr_sig", "abs_t", "p_adj"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    de["global_rank"] = np.arange(1, len(de) + 1)

    return de


def select_de_genes(
    de_df,
    top_n_de=2000,
    fdr_alpha=0.05,
    min_abs_log2fc=0.01,
    min_abs_delta=0.02,
    rank_by="abs_t",
    fill_to_top_n=True,
):
    if rank_by not in de_df.columns:
        raise ValueError(f"rank_by='{rank_by}' not found in DE table.")

    passing = de_df.loc[
        (de_df["p_adj"] < fdr_alpha)
        & (de_df["abs_log2fc"] >= min_abs_log2fc)
        & (de_df["abs_delta"] >= min_abs_delta)
    ].copy()

    passing = passing.sort_values(
        [rank_by, "p_adj"],
        ascending=[False, True],
    ).reset_index(drop=True)

    if fill_to_top_n and len(passing) < top_n_de:
        already = set(passing["gene"].astype(str))
        filler = de_df.loc[
            ~de_df["gene"].astype(str).isin(already)
        ].copy()

        filler = filler.sort_values(
            [rank_by, "p_adj"],
            ascending=[False, True],
        ).reset_index(drop=True)

        need = top_n_de - len(passing)
        selected = pd.concat(
            [passing, filler.head(need)],
            axis=0,
        ).reset_index(drop=True)
    else:
        selected = passing.head(top_n_de).copy().reset_index(drop=True)

    selected = selected.head(top_n_de).copy()
    selected["selected_rank"] = np.arange(1, len(selected) + 1)
    selected["passed_primary_filter"] = (
        (selected["p_adj"] < fdr_alpha)
        & (selected["abs_log2fc"] >= min_abs_log2fc)
        & (selected["abs_delta"] >= min_abs_delta)
    ).astype(int)

    print("\n[selection]")
    print(f"  primary passing genes: {len(passing)}")
    print(f"  final selected genes: {len(selected)}")
    print(f"  rank_by: {rank_by}")
    print(f"  fill_to_top_n: {fill_to_top_n}")

    return selected, passing


def compute_covariance_mel(X, shrinkage=1e-6):
    """
    Same style as uploaded pipeline:

        Xc = X - mean(X)
        C = Xc.T @ Xc / (n - 1)
        C += shrinkage * mean(diag(C)) * I
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)

    C = (Xc.T @ Xc) / max(1, X.shape[0] - 1)

    diag_mean = np.mean(np.diag(C)) + 1e-12
    C += float(shrinkage) * diag_mean * np.eye(C.shape[0])

    if SYMMETRIZE:
        C = symmetrize(C)

    return C


def build_H_from_sample_means_mel(
    X0,
    X1,
    shrinkage=1e-6,
    ridge=1e-6,
    mode="naive",
):
    """
    Same H choices as uploaded pipeline.

    X0 = cond0/control/Naive cells in selected genes
    X1 = cond1/perturbed/Resistant cells in selected genes

    S0 = covariance(X0)
    S1 = covariance(X1)

    mode == "diag":
        H = diag(diag(S0)/n0 + diag(S1)/n1)

    mode == "full":
        H = S0/n0 + S1/n1

    mode == "naive":
        H = (1/n0 + 1/n1) S0

    Then:
        H += ridge * mean(diag(H)) * I
    """
    X0 = np.asarray(X0, dtype=np.float64)
    X1 = np.asarray(X1, dtype=np.float64)

    n0 = X0.shape[0]
    n1 = X1.shape[0]

    S0 = compute_covariance_mel(X0, shrinkage=shrinkage)
    S1 = compute_covariance_mel(X1, shrinkage=shrinkage)

    if mode == "diag":
        hdiag = np.diag(S0) / max(n0, 1) + np.diag(S1) / max(n1, 1)
        H = np.diag(hdiag)

    elif mode == "full":
        H = S0 / max(n0, 1) + S1 / max(n1, 1)

    elif mode == "naive":
        H = (1.0 / max(n0, 1) + 1.0 / max(n1, 1)) * S0

    else:
        raise ValueError("H_MODE must be 'diag', 'full', or 'naive'.")

    scale = np.mean(np.diag(H)) + 1e-12
    H += float(ridge) * scale * np.eye(H.shape[0])

    if SYMMETRIZE:
        H = symmetrize(H)

    return H, S0, S1


def build_selected_matrices_from_cells_mel():
    check_file(H5AD_PATH)

    adata = read_h5ad_robust(H5AD_PATH)
    adata.var_names_make_unique()

    print(f"[data] loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    if CONDITION_KEY not in adata.obs.columns:
        raise KeyError(
            f"{CONDITION_KEY} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    print(f"\n[data] available {CONDITION_KEY}:")
    print(pd.Series(adata.obs[CONDITION_KEY]).value_counts())

    adata = adata[
        adata.obs[CONDITION_KEY].isin([COND0, COND1])
    ].copy()

    m0 = np.asarray(adata.obs[CONDITION_KEY].values == COND0)
    m1 = np.asarray(adata.obs[CONDITION_KEY].values == COND1)

    print(f"\n[contrast] {COND1} - {COND0}")
    print(f"  {COND0}: {m0.sum()} cells")
    print(f"  {COND1}: {m1.sum()} cells")

    if m0.sum() < 5 or m1.sum() < 5:
        raise ValueError(f"Too few cells: {COND0}={m0.sum()}, {COND1}={m1.sum()}")

    # -------------------------------
    # Gene filtering
    # -------------------------------
    adata = filter_genes_basic(
        adata,
        min_cells_frac=MIN_CELLS_FRAC,
        min_expr=MIN_EXPR,
        min_mean=MIN_MEAN,
        max_mean=MAX_MEAN,
        max_var_quantile=MAX_VAR_QUANTILE,
        seed=SEED,
        filter_subsample_cells=FILTER_SUBSAMPLE_CELLS,
    )

    if DROP_HOUSEKEEPING:
        keep = drop_bad_gene_prefixes(adata.var_names)
        print(f"[filter] bad-prefix filter kept {keep.sum()} / {len(keep)} genes")
        adata = adata[:, keep].copy()

    m0 = np.asarray(adata.obs[CONDITION_KEY].values == COND0)
    m1 = np.asarray(adata.obs[CONDITION_KEY].values == COND1)

    X0_all = to_dense(adata[m0].X).astype(np.float64)
    X1_all = to_dense(adata[m1].X).astype(np.float64)
    gene_names_all = np.asarray(adata.var_names, dtype=str)

    print(f"\n[matrix after filtering]")
    print(f"  X0_all: {X0_all.shape}")
    print(f"  X1_all: {X1_all.shape}")

    # -------------------------------
    # DE
    # -------------------------------
    de_df = compute_de_scores_mel(
        X0_all,
        X1_all,
        gene_names_all,
        logfc_pseudocount=LOGFC_PSEUDOCOUNT,
    )

    de_df.to_csv(OUTDIR / "all_genes_de.tsv", sep="\t", index=False)

    print("\n[DE diagnostic counts]")
    print(f"  total genes tested: {len(de_df)}")
    print(f"  FDR < {FDR_ALPHA}: {(de_df['p_adj'] < FDR_ALPHA).sum()}")

    for cut in [0.0, 0.01, 0.1, 0.25, 0.5, 1.0]:
        n = (
            (de_df["p_adj"] < FDR_ALPHA)
            & (de_df["abs_log2fc"] >= cut)
        ).sum()
        print(f"  FDR < {FDR_ALPHA} and abs_log2FC >= {cut}: {n}")

    selected_de, primary_passing_de = select_de_genes(
        de_df=de_df,
        top_n_de=TOP_N_DE,
        fdr_alpha=FDR_ALPHA,
        min_abs_log2fc=MIN_ABS_LOG2FC,
        min_abs_delta=MIN_ABS_DELTA,
        rank_by=RANK_BY,
        fill_to_top_n=FILL_TO_TOP_N,
    )

    selected_de.to_csv(OUTDIR / "selected_de.tsv", sep="\t", index=False)
    primary_passing_de.to_csv(OUTDIR / "primary_passing_de.tsv", sep="\t", index=False)

    print("\n[top selected genes]")
    cols = [
        "selected_rank",
        "gene",
        "passed_primary_filter",
        "mean_cond0",
        "mean_cond1",
        "delta",
        "log2fc",
        "t_stat",
        "p_adj",
    ]
    print(selected_de[cols].head(30).to_string(index=False))

    # -------------------------------
    # Highlight gene status
    # -------------------------------
    highlight_rows = []
    genes_to_highlight = normalize_gene_list(GENES_TO_HIGHLIGHT)

    print("\n[highlight gene checks]")
    for gene in genes_to_highlight:
        gene_status = de_df.loc[
            de_df["gene"].str.upper() == gene.upper()
        ].copy()

        in_selected = gene.upper() in set(selected_de["gene"].str.upper())

        if len(gene_status) > 0:
            row = gene_status.iloc[0].to_dict()
            row["highlight_gene"] = gene
            row["in_selected"] = in_selected
            highlight_rows.append(row)

            print(f"\n{gene}:")
            print(gene_status[[
                "gene",
                "global_rank",
                "mean_cond0",
                "mean_cond1",
                "delta",
                "log2fc",
                "t_stat",
                "p_adj",
            ]].to_string(index=False))
            print(f"in selected set: {in_selected}")
        else:
            print(f"\n{gene}: not found after filtering")
            highlight_rows.append({
                "highlight_gene": gene,
                "gene": gene,
                "found_after_filtering": False,
                "in_selected": False,
            })

    pd.DataFrame(highlight_rows).to_csv(
        OUTDIR / "highlight_gene_status.tsv",
        sep="\t",
        index=False,
    )

    # -------------------------------
    # Restrict to selected genes, preserving selected-DE order
    # -------------------------------
    selected_genes = selected_de["gene"].values.astype(str)
    selected_set = set(selected_genes)

    selected_mask = np.asarray([g in selected_set for g in gene_names_all])

    X0_sel_unordered = X0_all[:, selected_mask]
    X1_sel_unordered = X1_all[:, selected_mask]
    gene_names_sel_unordered = gene_names_all[selected_mask]

    name_to_idx = {
        g: i for i, g in enumerate(gene_names_sel_unordered)
    }

    order = [name_to_idx[g] for g in selected_genes if g in name_to_idx]

    X0_sel = X0_sel_unordered[:, order]
    X1_sel = X1_sel_unordered[:, order]
    gene_names_sel = gene_names_sel_unordered[order]

    print(f"\n[selected matrix]")
    print(f"  X0_sel: {X0_sel.shape}")
    print(f"  X1_sel: {X1_sel.shape}")
    print(f"  selected genes: {len(gene_names_sel)}")

    # -------------------------------
    # Compute y, Sigma, H from cells
    # -------------------------------
    y = X1_sel.mean(axis=0) - X0_sel.mean(axis=0)

    Sigma = compute_covariance_mel(
        X0_sel,
        shrinkage=SIGMA_SHRINKAGE,
    )

    H, S0, S1 = build_H_from_sample_means_mel(
        X0_sel,
        X1_sel,
        shrinkage=H_SHRINKAGE,
        ridge=H_RIDGE,
        mode=H_MODE,
    )

    print("\n[computed from cells]")
    print(f"  delta_x y: {y.shape}")
    print(f"  Sigma:     {Sigma.shape}")
    print(f"  H:         {H.shape}")
    print(f"  H_MODE:    {H_MODE}")
    print(f"  n0:        {X0_sel.shape[0]}")
    print(f"  n1:        {X1_sel.shape[0]}")
    print(f"  ||y||:     {np.linalg.norm(y):.6g}")
    print(f"  diag Sigma mean/min/max: {np.mean(np.diag(Sigma)):.3e}, {np.min(np.diag(Sigma)):.3e}, {np.max(np.diag(Sigma)):.3e}")
    print(f"  diag H mean/min/max:     {np.mean(np.diag(H)):.3e}, {np.min(np.diag(H)):.3e}, {np.max(np.diag(H)):.3e}")

    # Save the exact matrices used for auditability
    np.save(OUTDIR / "selected_gene_names.npy", gene_names_sel)
    np.save(OUTDIR / "X0_selected_mean.npy", X0_sel.mean(axis=0))
    np.save(OUTDIR / "X1_selected_mean.npy", X1_sel.mean(axis=0))
    np.save(OUTDIR / "delta_x.npy", y)
    np.save(OUTDIR / "Sigma_from_cells.npy", Sigma)
    np.save(OUTDIR / "H_from_cells.npy", H)
    np.save(OUTDIR / "S0_from_cells.npy", S0)
    np.save(OUTDIR / "S1_from_cells.npy", S1)

    pd.DataFrame({"gene": gene_names_sel}).to_csv(
        OUTDIR / "selected_gene_names.tsv",
        sep="\t",
        index=False,
    )

    run_config = {
        "h5ad_path": H5AD_PATH,
        "condition_key": CONDITION_KEY,
        "cond0": COND0,
        "cond1": COND1,
        "top_n_de": TOP_N_DE,
        "fdr_alpha": FDR_ALPHA,
        "min_abs_log2fc": MIN_ABS_LOG2FC,
        "min_abs_delta": MIN_ABS_DELTA,
        "rank_by": RANK_BY,
        "fill_to_top_n": FILL_TO_TOP_N,
        "drop_housekeeping": DROP_HOUSEKEEPING,
        "min_cells_frac": MIN_CELLS_FRAC,
        "min_expr": MIN_EXPR,
        "min_mean": MIN_MEAN,
        "max_var_quantile": MAX_VAR_QUANTILE,
        "logfc_pseudocount": LOGFC_PSEUDOCOUNT,
        "Sigma_shrinkage": SIGMA_SHRINKAGE,
        "H_shrinkage": H_SHRINKAGE,
        "H_ridge": H_RIDGE,
        "H_mode": H_MODE,
        "n0": X0_sel.shape[0],
        "n1": X1_sel.shape[0],
        "n_selected_genes": len(gene_names_sel),
    }

    pd.Series(run_config).to_csv(
        OUTDIR / "run_config_used_for_tau2_sweep.tsv",
        sep="\t",
        header=False,
    )

    return {
        "adata": adata,
        "de_df": de_df,
        "selected_de": selected_de,
        "primary_passing_de": primary_passing_de,
        "gene_names": gene_names_sel,
        "X0_sel": X0_sel,
        "X1_sel": X1_sel,
        "y": y,
        "Sigma": Sigma,
        "H": H,
        "S0": S0,
        "S1": S1,
    }
