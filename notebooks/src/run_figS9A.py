"""Fig S9A -- analytical finite-sampling noise covariance H calibration null.

Tests whether CIPHER's analytical CLT sampling-noise covariance H = Sigma0*(1/nA + 1/nB)
-- the same eigenbasis noise the inverse uses -- is better calibrated than a trace-matched
isotropic model, using pure control-control pseudobulk nulls with no perturbation signal.
Two independent bootstrap pseudobulks are drawn from the same control pool, dx is whitened
in the Sigma0 eigenbasis, and Q_CLT/d is checked against chi2_d/d (QQ plots, KS, per-dimension
NLL) while the isotropic baseline is not. Companion analyses score the per-perturbation-size
Gaussian NLL of the CLT vs isotropic model, both recomputing Sigma0 from each h5ad and using
the precomputed Sigma_full_ridge covariance. Outputs are QQ/hist/scatter PNG+SVG, per-dataset
JSON/NPZ, and aggregate CSVs.

Helpers in notebooks/src (not part of the cipher package). Config constants are module globals
the notebook overrides via R.__dict__.update; DATA_DIR/SUPPL/OUTDIR injected.
"""
import os
import re
import json
import math
import time
import glob
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.sparse import issparse
from scipy.stats import chi2, norm, kstest

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fall back to the text bar
    from tqdm import tqdm

import cipher

try:
    from IPython.display import display
except Exception:  # pragma: no cover - allow standalone import without IPython
    def display(*args, **kwargs):
        for a in args:
            print(a)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================
# CONFIG (injected by the notebook via R.__dict__.update)
# =========================================================

DATA_DIR = None
SUPPL = None
OUTDIR = None

DATASET_NAMES = [
    "NormanWeissman2019_filtered.h5ad",
    "ReplogleWeissman2022_rpe1.h5ad",
    "ReplogleWeissman2022_K562_essential.h5ad",
    "GSE264667_jurkat_raw_singlecell_01.h5ad",
    "GSE264667_hepg2_raw_singlecell_01.h5ad",
    "FrangiehIzar2021_RNA.h5ad",
    "TianKampmann2019_day7neuron.h5ad",
    "TianKampmann2021_CRISPRi.h5ad",
    "TianKampmann2021_CRISPRa.h5ad",
    "TianKampmann2019_iPSC.h5ad",
]

EXPRESSION_THRESHOLD = 1.0
MIN_SAMPLES = 100
N_NULL_REPS = 300
# Reuse per-dataset outputs that already exist, so a run stopped by a wall-clock limit resumes
# instead of recomputing finished datasets. Set False (or CIPHER_FIGS9A_RESUME=0) to force a
# full recompute.
RESUME_COMPLETED = bool(int(os.environ.get("CIPHER_FIGS9A_RESUME", "1")))
N_REPS_PER_PERT = 30
SEED = 0
RUN_SIGMA_METHODS = ["true"]
ONLY_RUN_CRISPRI_A_LISTS = True

CRISPRa_KEYWORDS = [
    "akana_etal_2026_crispra_perturbseq",
    "schemidt_etal_2022_crispra_perturbseq",
    "kaden25_rpe1_ctrl_10k_min100_greedy_4gb",
    "kaden25_fibroblast_ctrl_10k_min100_greedy_4gb",
    "NormanWeissman2019_filtered",
    "TianKampmann2021_CRISPRa",
]

CRISPRi_KEYWORDS = [
    "XAtlas2025_HEK293T_filtered",
    "Marson2025_D3_Stim8hr_filtered",
    "Marson2025_D4_Stim48hr_filtered",
    "Marson2025_D1_Stim48hr_filtered",
    "Marson2025_D1_Rest_filtered",
    "Marson2025_D4_Stim8hr_filtered",
    "Marson2025_D1_Stim8hr_filtered",
    "Marson2025_D4_Rest_filtered",
    "Marson2025_D2_Stim48hr_filtered",
    "Marson2025_D3_Stim48hr_filtered",
    "Marson2025_D3_Rest_filtered",
    "Marson2025_D2_Stim8hr_filtered",
    "XAtlas2025_HCT116_filtered",
    "ReplogleWeissman2022_rpe1",
    "ReplogleWeissman2022_K562_essential",
    "GSE264667_jurkat_raw_singlecell_01",
    "GSE264667_hepg2_raw_singlecell_01",
    "FrangiehIzar2021_RNA",
    "TianKampmann2019_day7neuron",
    "TianKampmann2021_CRISPRi",
    "TianKampmann2019_iPSC",
]

# Cross-section state populated by the run_* functions and read by the plot_* functions.
_CLT_SUMMARIES = None
_PERPERT_ALL_DF = None
_PERPERT_RUN_DIR = None
_PC_ALL_DF = None
_PC_RUN_DIR = None


# =========================================================
# SHARED BASIC HELPERS
# =========================================================

def to_dense(X):
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def _symmetrize(A):
    return 0.5 * (A + A.T)


def _shrink_cov(S, shrink=1e-3):
    S = _symmetrize(S)
    dbar = float(np.mean(np.diag(S))) if S.size else 1.0
    return (1.0 - shrink) * S + shrink * dbar * np.eye(S.shape[0], dtype=S.dtype)


def _eig_psd(S, jitter=1e-8):
    S = _symmetrize(S) + jitter * np.eye(S.shape[0], dtype=S.dtype)
    lam, V = np.linalg.eigh(S)
    lam = np.maximum(lam, jitter)
    return lam, V


def _subsample_rows(X, max_rows, rng):
    n = X.shape[0]
    if max_rows is None or max_rows <= 0 or n <= max_rows:
        return X
    idx = rng.choice(n, size=max_rows, replace=False)
    return X[idx]


def _mean_axis0(X):
    return np.asarray(X.mean(axis=0)).ravel()


def _cov_rowvar_false(X):
    if X.shape[0] <= 1:
        return np.eye(X.shape[1], dtype=np.float64)
    return np.cov(X, rowvar=False)


def _safe_gene_name_array(adata):
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    return np.array(adata.var_names.tolist(), dtype=str)


def pert_to_gene_safe(pert: str) -> str:
    p = str(pert).strip()
    p = re.sub(r"([_\-\s]+)(KD|KO|OE|overexp|overexpression)$", "", p, flags=re.IGNORECASE)
    p = re.sub(r"^(sg)(?=[A-Z0-9])", "", p)
    p = re.sub(r"^(sgRNA|gRNA|sgrna|grna|sg)([_\-\s]+)", "", p, flags=re.IGNORECASE)

    for s in ["_", "+", "-", "|", " "]:
        if s in p:
            p = p.split(s)[0]
            break

    return p


def _sample_mean_from_rows(X, n, rng, replace=True):
    n = int(n)
    N = X.shape[0]

    if n <= 0:
        raise ValueError("Sample size must be positive.")

    if replace:
        idx = rng.choice(N, size=n, replace=True)
    else:
        if n > N:
            raise ValueError(f"Cannot sample n={n} without replacement from N={N}.")
        idx = rng.choice(N, size=n, replace=False)

    return _mean_axis0(X[idx])


def _subsample_vector_entries(z, max_entries, rng):
    z = np.asarray(z).ravel()
    if z.size <= max_entries:
        return z.copy()
    idx = rng.choice(z.size, size=max_entries, replace=False)
    return z[idx].copy()


def _downsample_pool(existing, new_values, max_total, rng):
    """
    Reservoir-style downsampling for storing whitened coordinate residuals.
    """
    if existing is None or len(existing) == 0:
        if len(new_values) <= max_total:
            return new_values.copy()
        return _subsample_vector_entries(new_values, max_total, rng)

    existing = np.asarray(existing).ravel()
    new_values = np.asarray(new_values).ravel()

    combined_n = existing.size + new_values.size

    if combined_n <= max_total:
        return np.concatenate([existing, new_values])

    keep_existing_prob = existing.size / combined_n
    n_keep_existing = rng.binomial(max_total, keep_existing_prob)
    n_keep_new = max_total - n_keep_existing

    n_keep_existing = min(n_keep_existing, existing.size)
    n_keep_new = min(n_keep_new, new_values.size)

    idx_old = rng.choice(existing.size, size=n_keep_existing, replace=False)
    idx_new = rng.choice(new_values.size, size=n_keep_new, replace=False)

    return np.concatenate([existing[idx_old], new_values[idx_new]])


def set_equal_axes(ax, x, y, pad_frac=0.04):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    vals = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if vals.size == 0:
        return

    lo = np.nanpercentile(vals, 0.5)
    hi = np.nanpercentile(vals, 99.5)

    span = hi - lo
    if span <= 0:
        span = abs(hi) if hi != 0 else 1.0

    lo = lo - pad_frac * span
    hi = hi + pad_frac * span

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="gray")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def run_clt_isotropic_null():
    global _CLT_SUMMARIES

    PERT_KEY = "perturbation"
    CONTROL_LABEL = "control"
    SEP = "_"
    COV_SHRINK0 = 1e-3
    JITTER = 1e-8
    COV_MAX_CELLS_PER_GROUP = 4000
    CONTROL_NULL_NA_CAP = 4000
    PERT_LIKE_NB_CAP = 4000
    SAMPLE_WITH_REPLACEMENT = True
    MAX_Z_COORDS_FOR_QQ = 500_000
    C_CLT = "purple"
    C_ISO = "blue"
    C_BASE = "#9e9e9e"
    LINE_W = 2.2

    def qq_rmse_chi2(Q, df):
        Q = np.asarray(Q, dtype=float).ravel()
        Q = Q[np.isfinite(Q)]

        if Q.size < 5:
            return np.nan

        p = (np.arange(1, Q.size + 1) - 0.5) / Q.size
        theory = chi2.ppf(p, df=df) / float(df)
        empirical = np.sort(Q) / float(df)

        return float(np.sqrt(np.mean((empirical - theory) ** 2)))


    def chi2_calibration_summary(Q, df, nll):
        Q = np.asarray(Q, dtype=float).ravel()
        Q = Q[np.isfinite(Q)]
        q_scaled = Q / float(df)

        if Q.size < 5:
            return {
                "n": int(Q.size),
                "df": int(df),
                "mean_Q_over_d": np.nan,
                "var_Q_over_d": np.nan,
                "expected_mean_Q_over_d": 1.0,
                "expected_var_Q_over_d": 2.0 / float(df),
                "abs_mean_error": np.nan,
                "abs_var_error": np.nan,
                "qq_rmse_Q_over_d": np.nan,
                "ks_stat": np.nan,
                "ks_pvalue": np.nan,
                "mean_nll_per_dim": np.nan,
            }

        expected_mean = 1.0
        expected_var = 2.0 / float(df)

        try:
            ks_stat, ks_p = kstest(Q, lambda x: chi2.cdf(x, df=df))
        except Exception:
            ks_stat, ks_p = np.nan, np.nan

        nll = np.asarray(nll, dtype=float).ravel()
        nll = nll[np.isfinite(nll)]

        return {
            "n": int(Q.size),
            "df": int(df),
            "mean_Q_over_d": float(np.mean(q_scaled)),
            "var_Q_over_d": float(np.var(q_scaled, ddof=1)),
            "expected_mean_Q_over_d": float(expected_mean),
            "expected_var_Q_over_d": float(expected_var),
            "abs_mean_error": float(abs(np.mean(q_scaled) - expected_mean)),
            "abs_var_error": float(abs(np.var(q_scaled, ddof=1) - expected_var)),
            "qq_rmse_Q_over_d": float(qq_rmse_chi2(Q, df)),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "mean_nll_per_dim": float(np.mean(nll) / float(df)) if nll.size > 0 else np.nan,
        }


    def z_normal_qq_rmse(z):
        z = np.asarray(z, dtype=float).ravel()
        z = z[np.isfinite(z)]

        if z.size < 10:
            return np.nan

        z = np.sort(z)
        p = (np.arange(1, z.size + 1) - 0.5) / z.size
        theory = norm.ppf(p)

        lo = int(0.005 * z.size)
        hi = int(0.995 * z.size)

        if hi <= lo:
            lo, hi = 0, z.size

        return float(np.sqrt(np.mean((z[lo:hi] - theory[lo:hi]) ** 2)))

    def plot_q_chi2_qq(dataset_name, Q_clt, Q_iso, df, outdir):
        Q_clt = np.asarray(Q_clt, dtype=float).ravel()
        Q_iso = np.asarray(Q_iso, dtype=float).ravel()

        Q_clt = Q_clt[np.isfinite(Q_clt)]
        Q_iso = Q_iso[np.isfinite(Q_iso)]

        n = min(Q_clt.size, Q_iso.size)

        if n < 5:
            return

        Q_clt = np.sort(Q_clt)[:n]
        Q_iso = np.sort(Q_iso)[:n]

        p = (np.arange(1, n + 1) - 0.5) / n
        theory = chi2.ppf(p, df=df) / float(df)

        emp_clt = np.sort(Q_clt) / float(df)
        emp_iso = np.sort(Q_iso) / float(df)

        rmse_clt = qq_rmse_chi2(Q_clt, df)
        rmse_iso = qq_rmse_chi2(Q_iso, df)

        maxv = np.nanpercentile(np.concatenate([theory, emp_clt, emp_iso]), 99.5)
        maxv = max(maxv, 1.2)

        plt.figure(figsize=(6.5, 6.5))
        plt.plot([0, maxv], [0, maxv], linestyle="--", color=C_BASE, linewidth=1.2)
        plt.plot(
            theory,
            emp_iso,
            color=C_ISO,
            linewidth=LINE_W,
            label=f"isotropic QQ-RMSE={rmse_iso:.4g}",
        )
        plt.plot(
            theory,
            emp_clt,
            color=C_CLT,
            linewidth=LINE_W,
            label=f"CLT QQ-RMSE={rmse_clt:.4g}",
        )

        plt.xlabel(r"Theoretical $\chi^2_d/d$ quantile")
        plt.ylabel(r"Empirical $Q_H/d$ quantile")
        plt.title(f"{dataset_name}: pure control-sampling null")
        plt.xlim(0, maxv)
        plt.ylim(0, maxv)
        plt.legend(frameon=False)
        plt.tight_layout()

        png = os.path.join(outdir, f"{dataset_name}__Q_CHI2_QQ.png")
        svg = os.path.join(outdir, f"{dataset_name}__Q_CHI2_QQ.svg")

        plt.savefig(png, dpi=250, bbox_inches="tight")
        plt.savefig(svg, bbox_inches="tight")
        plt.close()


    def plot_q_hist(dataset_name, Q_clt, Q_iso, df, outdir):
        q_clt = np.asarray(Q_clt, dtype=float).ravel() / float(df)
        q_iso = np.asarray(Q_iso, dtype=float).ravel() / float(df)

        q_clt = q_clt[np.isfinite(q_clt)]
        q_iso = q_iso[np.isfinite(q_iso)]

        if q_clt.size < 5 or q_iso.size < 5:
            return

        maxv = np.nanpercentile(np.concatenate([q_clt, q_iso]), 99.0)
        maxv = max(maxv, 1.5)

        bins = np.linspace(0, maxv, 45)

        plt.figure(figsize=(7.0, 5.0))
        plt.hist(q_iso, bins=bins, density=True, alpha=0.45, label="isotropic")
        plt.hist(q_clt, bins=bins, density=True, alpha=0.45, label="CLT")
        plt.axvline(1.0, linestyle="--", color=C_BASE, linewidth=1.3, label=r"expected mean $\chi^2_d/d=1$")
        plt.xlabel(r"$Q_H/d$")
        plt.ylabel("Density")
        plt.title(f"{dataset_name}: null whitened norm")
        plt.legend(frameon=False)
        plt.tight_layout()

        png = os.path.join(outdir, f"{dataset_name}__Q_OVER_D_HIST.png")
        svg = os.path.join(outdir, f"{dataset_name}__Q_OVER_D_HIST.svg")

        plt.savefig(png, dpi=250, bbox_inches="tight")
        plt.savefig(svg, bbox_inches="tight")
        plt.close()


    def plot_z_normal_qq(dataset_name, z_clt, z_iso, outdir):
        z_clt = np.asarray(z_clt, dtype=float).ravel()
        z_iso = np.asarray(z_iso, dtype=float).ravel()

        z_clt = z_clt[np.isfinite(z_clt)]
        z_iso = z_iso[np.isfinite(z_iso)]

        n = min(z_clt.size, z_iso.size)

        if n < 100:
            return

        rng = np.random.default_rng(123)

        if z_clt.size > n:
            z_clt = rng.choice(z_clt, size=n, replace=False)
        if z_iso.size > n:
            z_iso = rng.choice(z_iso, size=n, replace=False)

        z_clt = np.sort(z_clt)
        z_iso = np.sort(z_iso)

        p = (np.arange(1, n + 1) - 0.5) / n
        theory = norm.ppf(p)

        # Trim extreme plotting limits for readability.
        lo = int(0.001 * n)
        hi = int(0.999 * n)

        if hi <= lo:
            lo, hi = 0, n

        theory_p = theory[lo:hi]
        clt_p = z_clt[lo:hi]
        iso_p = z_iso[lo:hi]

        rmse_clt = z_normal_qq_rmse(z_clt)
        rmse_iso = z_normal_qq_rmse(z_iso)

        maxv = np.nanpercentile(np.abs(np.concatenate([theory_p, clt_p, iso_p])), 99.5)
        maxv = max(maxv, 3.0)

        plt.figure(figsize=(6.5, 6.5))
        plt.plot([-maxv, maxv], [-maxv, maxv], linestyle="--", color=C_BASE, linewidth=1.2)
        plt.plot(
            theory_p,
            iso_p,
            color=C_ISO,
            linewidth=LINE_W,
            label=f"isotropic z QQ-RMSE={rmse_iso:.4g}",
        )
        plt.plot(
            theory_p,
            clt_p,
            color=C_CLT,
            linewidth=LINE_W,
            label=f"CLT z QQ-RMSE={rmse_clt:.4g}",
        )

        plt.xlabel("Theoretical N(0,1) quantile")
        plt.ylabel("Empirical whitened-coordinate quantile")
        plt.title(f"{dataset_name}: coordinate-wise whitening")
        plt.xlim(-maxv, maxv)
        plt.ylim(-maxv, maxv)
        plt.legend(frameon=False)
        plt.tight_layout()

        png = os.path.join(outdir, f"{dataset_name}__Z_NORMAL_QQ.png")
        svg = os.path.join(outdir, f"{dataset_name}__Z_NORMAL_QQ.svg")

        plt.savefig(png, dpi=250, bbox_inches="tight")
        plt.savefig(svg, bbox_inches="tight")
        plt.close()


    def plot_aggregate_summary(all_summaries, outdir):
        if len(all_summaries) == 0:
            return

        datasets = [s["dataset"] for s in all_summaries]

        clt_rmse = np.array([s["clt"]["qq_rmse_Q_over_d"] for s in all_summaries], dtype=float)
        iso_rmse = np.array([s["isotropic"]["qq_rmse_Q_over_d"] for s in all_summaries], dtype=float)

        clt_nll = np.array([s["clt"]["mean_nll_per_dim"] for s in all_summaries], dtype=float)
        iso_nll = np.array([s["isotropic"]["mean_nll_per_dim"] for s in all_summaries], dtype=float)

        clt_meanerr = np.array([s["clt"]["abs_mean_error"] for s in all_summaries], dtype=float)
        iso_meanerr = np.array([s["isotropic"]["abs_mean_error"] for s in all_summaries], dtype=float)

        x = np.arange(len(datasets))
        width = 0.38

        # QQ-RMSE barplot
        plt.figure(figsize=(max(11, 0.75 * len(datasets)), 5))
        plt.bar(x - width / 2, iso_rmse, width=width, label="isotropic")
        plt.bar(x + width / 2, clt_rmse, width=width, label="CLT")
        plt.xticks(x, datasets, rotation=60, ha="right")
        plt.ylabel(r"QQ-RMSE for $Q_H/d$ vs $\chi^2_d/d$")
        plt.title("Pure sampling-noise calibration: lower is better")
        plt.legend(frameon=False)
        plt.tight_layout()

        png = os.path.join(outdir, "AGGREGATE__QQ_RMSE_Q_OVER_D.png")
        svg = os.path.join(outdir, "AGGREGATE__QQ_RMSE_Q_OVER_D.svg")

        plt.savefig(png, dpi=250, bbox_inches="tight")
        plt.savefig(svg, bbox_inches="tight")
        plt.close()

        # NLL difference barplot
        nll_gain = iso_nll - clt_nll

        plt.figure(figsize=(max(11, 0.75 * len(datasets)), 5))
        plt.bar(x, nll_gain)
        plt.axhline(0.0, linestyle="--", color=C_BASE, linewidth=1.2)
        plt.xticks(x, datasets, rotation=60, ha="right")
        plt.ylabel("Mean NLL per dim: isotropic - CLT")
        plt.title("Pure sampling-noise likelihood gain; positive means CLT fits better")
        plt.tight_layout()

        png = os.path.join(outdir, "AGGREGATE__NLL_GAIN_ISO_MINUS_CLT.png")
        svg = os.path.join(outdir, "AGGREGATE__NLL_GAIN_ISO_MINUS_CLT.svg")

        plt.savefig(png, dpi=250, bbox_inches="tight")
        plt.savefig(svg, bbox_inches="tight")
        plt.close()

        # Mean error barplot
        plt.figure(figsize=(max(11, 0.75 * len(datasets)), 5))
        plt.bar(x - width / 2, iso_meanerr, width=width, label="isotropic")
        plt.bar(x + width / 2, clt_meanerr, width=width, label="CLT")
        plt.xticks(x, datasets, rotation=60, ha="right")
        plt.ylabel(r"$|\mathbb{E}[Q_H/d] - 1|$")
        plt.title("Whitened norm mean calibration: lower is better")
        plt.legend(frameon=False)
        plt.tight_layout()

        png = os.path.join(outdir, "AGGREGATE__MEAN_Q_OVER_D_ERROR.png")
        svg = os.path.join(outdir, "AGGREGATE__MEAN_Q_OVER_D_ERROR.svg")

        plt.savefig(png, dpi=250, bbox_inches="tight")
        plt.savefig(svg, bbox_inches="tight")
        plt.close()

    def run_one_dataset(data_path, outdir, seed=0):
        rng = np.random.default_rng(seed)

        dataset_name = os.path.basename(data_path).replace(".h5ad", "")

        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        adata = ad.read_h5ad(data_path)
        _safe_gene_name_array(adata)

        if PERT_KEY not in adata.obs.columns:
            raise ValueError(f"{data_path} does not contain obs['{PERT_KEY}'].")

        # -----------------------------------------------------
        # Gene filtering, same spirit as baseline script.
        # Keep genes above expression threshold plus perturbed genes.
        # -----------------------------------------------------
        gene_means = adata.X.mean(axis=0).A1 if issparse(adata.X) else np.asarray(adata.X).mean(axis=0)
        valid_genes = set(adata.var_names[np.where(gene_means >= float(EXPRESSION_THRESHOLD))[0]])

        all_perturbed_genes = set()

        for pert in adata.obs[PERT_KEY].astype(str).unique():
            if pert == CONTROL_LABEL:
                continue

            for g in str(pert).split(SEP):
                if g in set(adata.var_names):
                    all_perturbed_genes.add(g)

            parsed = pert_to_gene_safe(pert)
            if parsed in set(adata.var_names):
                all_perturbed_genes.add(parsed)

        keep_genes = list(valid_genes | all_perturbed_genes)
        adata = adata[:, adata.var_names.isin(keep_genes)].copy()

        gene_names = _safe_gene_name_array(adata)
        G = len(gene_names)

        # -----------------------------------------------------
        # Perturbation filtering only used to mimic realistic n_B sizes.
        # The actual null is control-control and has no perturbation signal.
        # -----------------------------------------------------
        pert_counts = adata.obs[PERT_KEY].astype(str).value_counts()
        valid_perts = pert_counts[pert_counts >= int(MIN_SAMPLES)].index.astype(str).tolist()

        adata = adata[adata.obs[PERT_KEY].astype(str).isin(valid_perts)].copy()
        obs_pert = adata.obs[PERT_KEY].astype(str).values

        if CONTROL_LABEL not in set(obs_pert):
            raise ValueError(f"Control label '{CONTROL_LABEL}' not found in {dataset_name}.")

        X0_all = to_dense(adata[obs_pert == CONTROL_LABEL].X).astype(np.float32, copy=False)
        n0_full = int(X0_all.shape[0])

        perts_noncontrol = [p for p in np.unique(obs_pert) if p != CONTROL_LABEL]
        pert_sizes = []

        for p in perts_noncontrol:
            n_p = int(np.sum(obs_pert == p))
            if n_p >= MIN_SAMPLES:
                pert_sizes.append(n_p)

        if len(pert_sizes) == 0:
            # fallback: use a perturbation-like size from controls
            pert_sizes = [min(max(MIN_SAMPLES, 100), max(2, n0_full // 4))]

        pert_sizes = np.asarray(pert_sizes, dtype=int)

        print(f"n_cells after filtering: {adata.n_obs}")
        print(f"n_genes after filtering: {G}")
        print(f"n_control cells:         {n0_full}")
        print(f"n_valid perturbations:   {len(perts_noncontrol)}")
        print(f"pert-like n_B range:     {int(np.min(pert_sizes))} to {int(np.max(pert_sizes))}")

        # -----------------------------------------------------
        # Estimate control covariance Σ0 and eigenbasis.
        # -----------------------------------------------------
        X0_cov = _subsample_rows(X0_all, COV_MAX_CELLS_PER_GROUP, rng).astype(np.float64, copy=False)
        n0_cov = int(X0_cov.shape[0])

        print(f"Estimating Σ0 from {n0_cov} control cells...")

        Sigma0 = cipher.compute_covariance(X0_cov, shrink=COV_SHRINK0)

        lam, V = _eig_psd(Sigma0, jitter=JITTER)

        d_eff = int(G)
        lam_mean = float(np.mean(lam))

        print(f"Eigenbasis dimension d = {d_eff}")
        print(f"mean eigenvalue = {lam_mean:.6g}")

        # -----------------------------------------------------
        # Pure sampling-noise null.
        # -----------------------------------------------------
        Q_clt = []
        Q_iso = []

        nll_clt = []
        nll_iso = []

        nA_used = []
        nB_used = []

        z_clt_pool = None
        z_iso_pool = None

        nA_default = n0_full if CONTROL_NULL_NA_CAP is None else min(int(CONTROL_NULL_NA_CAP), n0_full)
        nA_default = max(2, int(nA_default))

        for b in tqdm(range(N_NULL_REPS), desc=f"{dataset_name}: pure sampling-noise null"):
            nB = int(rng.choice(pert_sizes))

            if PERT_LIKE_NB_CAP is not None:
                nB = min(nB, int(PERT_LIKE_NB_CAP))

            nB = max(2, int(nB))

            if not SAMPLE_WITH_REPLACEMENT:
                # Need two non-overlapping samples if sampling without replacement.
                # Cap sizes so nA + nB <= n0_full.
                nB = min(nB, max(2, n0_full // 2))
                nA = min(nA_default, max(2, n0_full - nB))

                perm = rng.permutation(n0_full)
                idxA = perm[:nA]
                idxB = perm[nA:nA + nB]

                meanA = _mean_axis0(X0_all[idxA])
                meanB = _mean_axis0(X0_all[idxB])

            else:
                # Independent bootstrap pseudobulks from the same control distribution.
                nA = nA_default
                meanA = _sample_mean_from_rows(X0_all, nA, rng, replace=True)
                meanB = _sample_mean_from_rows(X0_all, nB, rng, replace=True)

            dx = meanA - meanB

            # Work in Σ0 eigenbasis.
            y = V.T @ dx

            # CLT noise diagonal in Σ0 eigenbasis.
            h_clt = lam * ((1.0 / float(nA)) + (1.0 / float(nB)))
            h_clt = np.maximum(h_clt, JITTER)

            # Trace-matched isotropic noise.
            sigma2_iso = float(np.mean(h_clt))
            sigma2_iso = max(sigma2_iso, JITTER)

            q_clt = float(np.sum((y * y) / h_clt))
            q_iso = float(np.sum((y * y) / sigma2_iso))

            # Gaussian NLLs up to the exact normalizing constants.
            this_nll_clt = 0.5 * (
                np.sum(np.log(2.0 * np.pi * h_clt)) + q_clt
            )

            this_nll_iso = 0.5 * (
                d_eff * np.log(2.0 * np.pi * sigma2_iso) + q_iso
            )

            Q_clt.append(q_clt)
            Q_iso.append(q_iso)

            nll_clt.append(this_nll_clt)
            nll_iso.append(this_nll_iso)

            nA_used.append(nA)
            nB_used.append(nB)

            z_clt = y / np.sqrt(h_clt)
            z_iso = y / np.sqrt(sigma2_iso)

            z_clt_pool = _downsample_pool(z_clt_pool, z_clt, MAX_Z_COORDS_FOR_QQ, rng)
            z_iso_pool = _downsample_pool(z_iso_pool, z_iso, MAX_Z_COORDS_FOR_QQ, rng)

        Q_clt = np.asarray(Q_clt, dtype=float)
        Q_iso = np.asarray(Q_iso, dtype=float)

        nll_clt = np.asarray(nll_clt, dtype=float)
        nll_iso = np.asarray(nll_iso, dtype=float)

        nA_used = np.asarray(nA_used, dtype=int)
        nB_used = np.asarray(nB_used, dtype=int)

        # -----------------------------------------------------
        # Calibration summaries.
        # -----------------------------------------------------
        summary_clt = chi2_calibration_summary(Q_clt, d_eff, nll_clt)
        summary_iso = chi2_calibration_summary(Q_iso, d_eff, nll_iso)

        z_summary = {
            "z_clt_normal_qq_rmse": float(z_normal_qq_rmse(z_clt_pool)),
            "z_iso_normal_qq_rmse": float(z_normal_qq_rmse(z_iso_pool)),
            "z_clt_mean": float(np.mean(z_clt_pool)),
            "z_iso_mean": float(np.mean(z_iso_pool)),
            "z_clt_var": float(np.var(z_clt_pool, ddof=1)),
            "z_iso_var": float(np.var(z_iso_pool, ddof=1)),
            "n_z_coords_used": int(len(z_clt_pool)),
        }

        nll_gain = float(np.mean(nll_iso - nll_clt) / float(d_eff))

        winner_by_qq = "CLT" if summary_clt["qq_rmse_Q_over_d"] < summary_iso["qq_rmse_Q_over_d"] else "isotropic"
        winner_by_nll = "CLT" if nll_gain > 0 else "isotropic"

        summary = {
            "dataset": dataset_name,
            "data_path": data_path,
            "n_cells_after_filtering": int(adata.n_obs),
            "n_genes": int(G),
            "n_control_cells": int(n0_full),
            "n_control_cells_for_cov": int(n0_cov),
            "n_null_reps": int(N_NULL_REPS),
            "control_null_nA_cap": None if CONTROL_NULL_NA_CAP is None else int(CONTROL_NULL_NA_CAP),
            "pert_like_nB_cap": None if PERT_LIKE_NB_CAP is None else int(PERT_LIKE_NB_CAP),
            "sample_with_replacement": bool(SAMPLE_WITH_REPLACEMENT),
            "nA_mean": float(np.mean(nA_used)),
            "nB_mean": float(np.mean(nB_used)),
            "nB_min": int(np.min(nB_used)),
            "nB_max": int(np.max(nB_used)),
            "clt": summary_clt,
            "isotropic": summary_iso,
            "z_coordinate_summary": z_summary,
            "mean_nll_per_dim_gain_iso_minus_clt": nll_gain,
            "winner_by_chi2_qq_rmse": winner_by_qq,
            "winner_by_gaussian_nll": winner_by_nll,
        }

        print("\nCalibration summary:")
        print(f"  CLT QQ-RMSE       = {summary_clt['qq_rmse_Q_over_d']:.6g}")
        print(f"  isotropic QQ-RMSE = {summary_iso['qq_rmse_Q_over_d']:.6g}")
        print(f"  winner by QQ      = {winner_by_qq}")
        print(f"  NLL gain iso-CLT  = {nll_gain:.6g} per dim")
        print(f"  winner by NLL     = {winner_by_nll}")
        print(f"  CLT mean Q/d      = {summary_clt['mean_Q_over_d']:.6g}")
        print(f"  iso mean Q/d      = {summary_iso['mean_Q_over_d']:.6g}")
        print(f"  CLT z var         = {z_summary['z_clt_var']:.6g}")
        print(f"  iso z var         = {z_summary['z_iso_var']:.6g}")

        # -----------------------------------------------------
        # Save arrays and summaries.
        # -----------------------------------------------------
        with open(os.path.join(outdir, f"{dataset_name}__sampling_noise_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        np.savez_compressed(
            os.path.join(outdir, f"{dataset_name}__sampling_noise_null_arrays.npz"),
            Q_clt=Q_clt.astype(np.float64),
            Q_iso=Q_iso.astype(np.float64),
            Q_clt_over_d=(Q_clt / float(d_eff)).astype(np.float64),
            Q_iso_over_d=(Q_iso / float(d_eff)).astype(np.float64),
            nll_clt=nll_clt.astype(np.float64),
            nll_iso=nll_iso.astype(np.float64),
            nA_used=nA_used.astype(np.int32),
            nB_used=nB_used.astype(np.int32),
            z_clt=np.asarray(z_clt_pool, dtype=np.float32),
            z_iso=np.asarray(z_iso_pool, dtype=np.float32),
            eigenvalues=lam.astype(np.float32),
        )

        # -----------------------------------------------------
        # Plots.
        # -----------------------------------------------------
        plot_q_chi2_qq(dataset_name, Q_clt, Q_iso, d_eff, outdir)
        plot_q_hist(dataset_name, Q_clt, Q_iso, d_eff, outdir)
        plot_z_normal_qq(dataset_name, z_clt_pool, z_iso_pool, outdir)

        print(f"\n[saved] {dataset_name} outputs -> {outdir}")

        return summary

    run_outdir = os.path.join(OUTDIR, "clt_isotropic_null")
    os.makedirs(run_outdir, exist_ok=True)
    data_paths = [os.path.join(DATA_DIR, name) for name in DATASET_NAMES]

    all_summaries = []

    for i, data_path in enumerate(data_paths):
        try:
            # Resume guard: each dataset writes its own summary JSON + null-array npz, so a run
            # stopped by a wall-clock limit can be relaunched and will reuse the datasets it
            # already finished. The cached summary is loaded (not just skipped) so the combined
            # JSON/CSV written below still covers every dataset.
            _ds = os.path.basename(data_path).replace(".h5ad", "")
            _cached = os.path.join(run_outdir, f"{_ds}__sampling_noise_summary.json")
            _cached_npz = os.path.join(run_outdir, f"{_ds}__sampling_noise_null_arrays.npz")
            if RESUME_COMPLETED and os.path.exists(_cached) and os.path.exists(_cached_npz):
                with open(_cached) as _f:
                    all_summaries.append(json.load(_f))
                print(f"[resume] {_ds}: reusing cached sampling-noise summary")
                continue

            summary = run_one_dataset(
                data_path=data_path,
                outdir=run_outdir,
                seed=SEED + i,
            )
            all_summaries.append(summary)

        except Exception as e:
            print("\n" + "!" * 100)
            print(f"[ERROR] Failed on {data_path}")
            print(repr(e))
            print("!" * 100 + "\n")

    # Save combined JSON.
    with open(os.path.join(run_outdir, "ALL_DATASETS__sampling_noise_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Save compact CSV summary.
    csv_path = os.path.join(run_outdir, "ALL_DATASETS__sampling_noise_summary_table.csv")

    cols = [
        "dataset",
        "n_genes",
        "n_control_cells",
        "n_null_reps",
        "nA_mean",
        "nB_mean",
        "clt_mean_Q_over_d",
        "iso_mean_Q_over_d",
        "clt_var_Q_over_d",
        "iso_var_Q_over_d",
        "clt_abs_mean_error",
        "iso_abs_mean_error",
        "clt_qq_rmse_Q_over_d",
        "iso_qq_rmse_Q_over_d",
        "clt_ks_stat",
        "iso_ks_stat",
        "clt_ks_pvalue",
        "iso_ks_pvalue",
        "clt_mean_nll_per_dim",
        "iso_mean_nll_per_dim",
        "mean_nll_per_dim_gain_iso_minus_clt",
        "z_clt_normal_qq_rmse",
        "z_iso_normal_qq_rmse",
        "z_clt_var",
        "z_iso_var",
        "winner_by_chi2_qq_rmse",
        "winner_by_gaussian_nll",
    ]

    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")

        for s in all_summaries:
            row = [
                s["dataset"],
                s["n_genes"],
                s["n_control_cells"],
                s["n_null_reps"],
                s["nA_mean"],
                s["nB_mean"],
                s["clt"]["mean_Q_over_d"],
                s["isotropic"]["mean_Q_over_d"],
                s["clt"]["var_Q_over_d"],
                s["isotropic"]["var_Q_over_d"],
                s["clt"]["abs_mean_error"],
                s["isotropic"]["abs_mean_error"],
                s["clt"]["qq_rmse_Q_over_d"],
                s["isotropic"]["qq_rmse_Q_over_d"],
                s["clt"]["ks_stat"],
                s["isotropic"]["ks_stat"],
                s["clt"]["ks_pvalue"],
                s["isotropic"]["ks_pvalue"],
                s["clt"]["mean_nll_per_dim"],
                s["isotropic"]["mean_nll_per_dim"],
                s["mean_nll_per_dim_gain_iso_minus_clt"],
                s["z_coordinate_summary"]["z_clt_normal_qq_rmse"],
                s["z_coordinate_summary"]["z_iso_normal_qq_rmse"],
                s["z_coordinate_summary"]["z_clt_var"],
                s["z_coordinate_summary"]["z_iso_var"],
                s["winner_by_chi2_qq_rmse"],
                s["winner_by_gaussian_nll"],
            ]

            f.write(",".join([str(x) for x in row]) + "\n")

    print("[saved]", csv_path)

    # Aggregate plots.
    plot_aggregate_summary(all_summaries, run_outdir)

    _CLT_SUMMARIES = all_summaries


def plot_clt_isotropic_posthoc():
    POSTHOC_OUTDIR_NAME = "posthoc_plots"
    SHOW_FIGS = True
    SAVE_FIGS = True
    MAKE_CALIBRATED_CLT = True
    MAX_Z_PLOT = 250_000
    DPI = 250
    SAVE_SVG = True

    RUN_DIR = os.path.join(OUTDIR, "clt_isotropic_null")
    PLOT_OUTDIR = os.path.join(RUN_DIR, POSTHOC_OUTDIR_NAME)
    os.makedirs(PLOT_OUTDIR, exist_ok=True)

    summary_csv = os.path.join(RUN_DIR, "ALL_DATASETS__sampling_noise_summary_table.csv")
    summary_json = os.path.join(RUN_DIR, "ALL_DATASETS__sampling_noise_summary.json")

    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    summary_df = pd.read_csv(summary_csv)

    print("\nLoaded summary table:")
    display(summary_df)

    # ============================================================
    # HELPERS
    # ============================================================

    def finite(x):
        x = np.asarray(x, dtype=float).ravel()
        return x[np.isfinite(x)]


    def qq_rmse_chi2(Q, df):
        Q = finite(Q)
        if len(Q) < 5:
            return np.nan

        p = (np.arange(1, len(Q) + 1) - 0.5) / len(Q)
        theory = chi2.ppf(p, df=df) / float(df)
        empirical = np.sort(Q) / float(df)

        return float(np.sqrt(np.mean((empirical - theory) ** 2)))


    def qq_rmse_normal(z, trim=0.005):
        z = finite(z)
        if len(z) < 20:
            return np.nan

        z = np.sort(z)
        p = (np.arange(1, len(z) + 1) - 0.5) / len(z)
        theory = norm.ppf(p)

        lo = int(trim * len(z))
        hi = int((1.0 - trim) * len(z))

        if hi <= lo:
            lo, hi = 0, len(z)

        return float(np.sqrt(np.mean((z[lo:hi] - theory[lo:hi]) ** 2)))


    def chi2_summary(Q, df, label):
        Q = finite(Q)
        qd = Q / float(df)

        if len(Q) < 5:
            return {
                "model": label,
                "n": len(Q),
                "mean_Q_over_d": np.nan,
                "var_Q_over_d": np.nan,
                "abs_mean_error": np.nan,
                "qq_rmse": np.nan,
                "ks_stat": np.nan,
                "ks_p": np.nan,
            }

        try:
            ks_stat, ks_p = kstest(Q, lambda x: chi2.cdf(x, df=df))
        except Exception:
            ks_stat, ks_p = np.nan, np.nan

        return {
            "model": label,
            "n": len(Q),
            "mean_Q_over_d": float(np.mean(qd)),
            "var_Q_over_d": float(np.var(qd, ddof=1)),
            "expected_var_Q_over_d": float(2.0 / df),
            "abs_mean_error": float(abs(np.mean(qd) - 1.0)),
            "qq_rmse": float(qq_rmse_chi2(Q, df)),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
        }


    def savefig(fig, path_base):
        if SAVE_FIGS:
            fig.savefig(path_base + ".png", dpi=DPI, bbox_inches="tight")
            if SAVE_SVG:
                fig.savefig(path_base + ".svg", bbox_inches="tight")

        if SHOW_FIGS:
            plt.show()
        else:
            plt.close(fig)


    def dataset_npz_path(dataset):
        path = os.path.join(RUN_DIR, f"{dataset}__sampling_noise_null_arrays.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path


    def maybe_subsample(x, max_n, seed=123):
        x = finite(x)
        if len(x) <= max_n:
            return x

        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_n, replace=False)
        return x[idx]


    def add_identity_line(ax, x, y):
        vals = finite(np.concatenate([finite(x), finite(y)]))
        if len(vals) == 0:
            return

        lo = 0.0
        hi = np.nanpercentile(vals, 99.5)
        hi = max(hi, 1.2)

        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="gray")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)


    # ============================================================
    # PER-DATASET PLOTS
    # ============================================================

    all_posthoc_rows = []

    for _, row in summary_df.iterrows():

        dataset = row["dataset"]
        df_dim = int(row["n_genes"])

        print("\n" + "=" * 100)
        print(dataset)
        print("=" * 100)

        arr = np.load(dataset_npz_path(dataset))

        Q_clt = finite(arr["Q_clt"])
        Q_iso = finite(arr["Q_iso"])

        qd_clt = Q_clt / float(df_dim)
        qd_iso = Q_iso / float(df_dim)

        nll_clt = finite(arr["nll_clt"])
        nll_iso = finite(arr["nll_iso"])

        z_clt = maybe_subsample(arr["z_clt"], MAX_Z_PLOT, seed=1)
        z_iso = maybe_subsample(arr["z_iso"], MAX_Z_PLOT, seed=2)

        # -----------------------------
        # Optional scalar-calibrated CLT
        # -----------------------------
        if MAKE_CALIBRATED_CLT:
            alpha = float(np.mean(qd_clt))

            if not np.isfinite(alpha) or alpha <= 0:
                alpha = 1.0

            Q_clt_cal = Q_clt / alpha
            qd_clt_cal = Q_clt_cal / float(df_dim)

            # If H_cal = alpha H_CLT:
            # NLL_cal = NLL_clt + 0.5 * d * log(alpha) + 0.5 * (Q_clt / alpha - Q_clt)
            nll_clt_cal = nll_clt + 0.5 * df_dim * np.log(alpha) + 0.5 * (Q_clt / alpha - Q_clt)

            z_clt_cal = z_clt / math.sqrt(alpha)
        else:
            alpha = np.nan
            Q_clt_cal = None
            qd_clt_cal = None
            nll_clt_cal = None
            z_clt_cal = None

        # -----------------------------
        # Print compact stats
        # -----------------------------
        stats = []
        stats.append(chi2_summary(Q_clt, df_dim, "CLT"))
        stats.append(chi2_summary(Q_iso, df_dim, "isotropic"))

        if MAKE_CALIBRATED_CLT:
            stats.append(chi2_summary(Q_clt_cal, df_dim, "CLT calibrated"))

        stats_df = pd.DataFrame(stats)

        print("Chi-square / Q calibration:")
        display(stats_df)

        z_stats = pd.DataFrame([
            {
                "model": "CLT",
                "z_mean": float(np.mean(z_clt)),
                "z_var": float(np.var(z_clt, ddof=1)),
                "z_normal_qq_rmse": qq_rmse_normal(z_clt),
            },
            {
                "model": "isotropic",
                "z_mean": float(np.mean(z_iso)),
                "z_var": float(np.var(z_iso, ddof=1)),
                "z_normal_qq_rmse": qq_rmse_normal(z_iso),
            },
        ])

        if MAKE_CALIBRATED_CLT:
            z_stats = pd.concat([
                z_stats,
                pd.DataFrame([{
                    "model": "CLT calibrated",
                    "z_mean": float(np.mean(z_clt_cal)),
                    "z_var": float(np.var(z_clt_cal, ddof=1)),
                    "z_normal_qq_rmse": qq_rmse_normal(z_clt_cal),
                }])
            ], ignore_index=True)

        print("Coordinate-wise z calibration:")
        display(z_stats)

        for dct in stats:
            all_posthoc_rows.append({
                "dataset": dataset,
                "model": dct["model"],
                "n_genes": df_dim,
                "alpha_clt_scale": alpha,
                **dct,
            })

        # ========================================================
        # Plot 1: Q/d histograms
        # ========================================================

        fig, ax = plt.subplots(figsize=(7.2, 5.0))

        all_qd = [qd_clt, qd_iso]
        labels = ["CLT", "isotropic"]

        if MAKE_CALIBRATED_CLT:
            all_qd.append(qd_clt_cal)
            labels.append("CLT calibrated")

        maxv = np.nanpercentile(np.concatenate(all_qd), 99.0)
        maxv = max(maxv, 1.5)

        bins = np.linspace(0, maxv, 45)

        for vals, label in zip(all_qd, labels):
            ax.hist(vals, bins=bins, density=True, alpha=0.40, label=label)

        ax.axvline(1.0, linestyle="--", linewidth=1.3, color="gray", label="expected mean = 1")
        ax.set_xlabel(r"$Q_H / d$")
        ax.set_ylabel("Density")
        ax.set_title(f"{dataset}: null whitened norm")
        ax.legend(frameon=False)

        fig.tight_layout()
        savefig(fig, os.path.join(PLOT_OUTDIR, f"{dataset}__Q_OVER_D_hist"))

        # ========================================================
        # Plot 2: Chi-square QQ for Q/d
        # ========================================================

        fig, ax = plt.subplots(figsize=(6.5, 6.5))

        n = min(len(Q_clt), len(Q_iso))
        p = (np.arange(1, n + 1) - 0.5) / n
        theory = chi2.ppf(p, df=df_dim) / float(df_dim)

        emp_clt = np.sort(Q_clt)[:n] / float(df_dim)
        emp_iso = np.sort(Q_iso)[:n] / float(df_dim)

        ax.plot(
            theory,
            emp_clt,
            linewidth=2,
            label=f"CLT RMSE={qq_rmse_chi2(Q_clt, df_dim):.3g}",
        )

        ax.plot(
            theory,
            emp_iso,
            linewidth=2,
            label=f"isotropic RMSE={qq_rmse_chi2(Q_iso, df_dim):.3g}",
        )

        if MAKE_CALIBRATED_CLT:
            emp_cal = np.sort(Q_clt_cal)[:n] / float(df_dim)
            ax.plot(
                theory,
                emp_cal,
                linewidth=2,
                label=f"CLT calibrated RMSE={qq_rmse_chi2(Q_clt_cal, df_dim):.3g}",
            )

        add_identity_line(ax, theory, np.concatenate([emp_clt, emp_iso] + ([emp_cal] if MAKE_CALIBRATED_CLT else [])))

        ax.set_xlabel(r"Theoretical $\chi^2_d/d$ quantile")
        ax.set_ylabel(r"Empirical $Q_H/d$ quantile")
        ax.set_title(f"{dataset}: chi-square QQ")
        ax.legend(frameon=False)

        fig.tight_layout()
        savefig(fig, os.path.join(PLOT_OUTDIR, f"{dataset}__Q_CHI2_QQ"))

        # ========================================================
        # Plot 3: z-coordinate Normal QQ
        # ========================================================

        fig, ax = plt.subplots(figsize=(6.5, 6.5))

        z_plot_items = [
            ("CLT", z_clt),
            ("isotropic", z_iso),
        ]

        if MAKE_CALIBRATED_CLT:
            z_plot_items.append(("CLT calibrated", z_clt_cal))

        n_z = min(len(z) for _, z in z_plot_items)

        p = (np.arange(1, n_z + 1) - 0.5) / n_z
        theory_z = norm.ppf(p)

        # Trim tails for readability
        lo = int(0.001 * n_z)
        hi = int(0.999 * n_z)
        if hi <= lo:
            lo, hi = 0, n_z

        theory_plot = theory_z[lo:hi]

        plotted_zs = []

        for label, z in z_plot_items:
            z = np.sort(z)[:n_z]
            z_plot = z[lo:hi]
            plotted_zs.append(z_plot)

            ax.plot(
                theory_plot,
                z_plot,
                linewidth=2,
                label=f"{label} RMSE={qq_rmse_normal(z):.3g}",
            )

        all_z_for_lim = np.concatenate([theory_plot] + plotted_zs)
        lim = np.nanpercentile(np.abs(all_z_for_lim), 99.5)
        lim = max(lim, 3.0)

        ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1.2, color="gray")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        ax.set_xlabel("Theoretical N(0,1) quantile")
        ax.set_ylabel("Empirical whitened-coordinate quantile")
        ax.set_title(f"{dataset}: coordinate-wise z QQ")
        ax.legend(frameon=False)

        fig.tight_layout()
        savefig(fig, os.path.join(PLOT_OUTDIR, f"{dataset}__Z_NORMAL_QQ"))

        # ========================================================
        # Plot 4: compact Q/d boxplot
        # ========================================================

        fig, ax = plt.subplots(figsize=(7.0, 4.8))

        box_data = [qd_clt, qd_iso]
        box_labels = ["CLT", "isotropic"]

        if MAKE_CALIBRATED_CLT:
            box_data.append(qd_clt_cal)
            box_labels.append("CLT\ncalibrated")

        ax.boxplot(box_data, labels=box_labels, showfliers=False)

        # Add lightly jittered points
        rng = np.random.default_rng(0)
        for i, vals in enumerate(box_data, start=1):
            vals_plot = vals
            if len(vals_plot) > 600:
                vals_plot = rng.choice(vals_plot, size=600, replace=False)
            jitter = rng.normal(0, 0.045, size=len(vals_plot))
            ax.scatter(np.full(len(vals_plot), i) + jitter, vals_plot, s=10, alpha=0.35)

        ax.axhline(1.0, linestyle="--", linewidth=1.2, color="gray")
        ax.set_ylabel(r"$Q_H/d$")
        ax.set_title(f"{dataset}: draw-level calibrated norm")
        ax.set_ylim(bottom=0)

        fig.tight_layout()
        savefig(fig, os.path.join(PLOT_OUTDIR, f"{dataset}__Q_OVER_D_boxplot"))

        # ========================================================
        # Plot 5: NLL per draw
        # ========================================================

        fig, ax = plt.subplots(figsize=(7.0, 4.8))

        nll_data = [
            nll_clt / float(df_dim),
            nll_iso / float(df_dim),
        ]

        nll_labels = ["CLT", "isotropic"]

        if MAKE_CALIBRATED_CLT:
            nll_data.append(nll_clt_cal / float(df_dim))
            nll_labels.append("CLT\ncalibrated")

        ax.boxplot(nll_data, labels=nll_labels, showfliers=False)

        rng = np.random.default_rng(1)
        for i, vals in enumerate(nll_data, start=1):
            vals_plot = vals
            if len(vals_plot) > 600:
                vals_plot = rng.choice(vals_plot, size=600, replace=False)
            jitter = rng.normal(0, 0.045, size=len(vals_plot))
            ax.scatter(np.full(len(vals_plot), i) + jitter, vals_plot, s=10, alpha=0.35)

        ax.set_ylabel("Gaussian NLL per dimension")
        ax.set_title(f"{dataset}: draw-level likelihood")

        fig.tight_layout()
        savefig(fig, os.path.join(PLOT_OUTDIR, f"{dataset}__NLL_PER_DIM_boxplot"))


    # ============================================================
    # AGGREGATE POSTHOC SUMMARY
    # ============================================================

    posthoc_df = pd.DataFrame(all_posthoc_rows)

    posthoc_csv = os.path.join(PLOT_OUTDIR, "POSTHOC__calibration_summary.csv")
    posthoc_df.to_csv(posthoc_csv, index=False)

    print("\nSaved posthoc summary:")
    print(posthoc_csv)

    display(posthoc_df)


    # ------------------------------------------------------------
    # Aggregate plot 1: QQ-RMSE
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(max(11, 0.8 * summary_df.shape[0]), 5.2))

    datasets = summary_df["dataset"].tolist()
    x = np.arange(len(datasets))

    models = ["CLT", "isotropic"]
    if MAKE_CALIBRATED_CLT:
        models.append("CLT calibrated")

    width = 0.8 / len(models)

    for j, model in enumerate(models):
        vals = []
        for ds in datasets:
            sub = posthoc_df[(posthoc_df["dataset"] == ds) & (posthoc_df["model"] == model)]
            vals.append(float(sub["qq_rmse"].iloc[0]))
        ax.bar(x - 0.4 + width / 2 + j * width, vals, width=width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=60, ha="right")
    ax.set_ylabel(r"QQ-RMSE for $Q_H/d$ vs $\chi^2_d/d$")
    ax.set_title("Aggregate chi-square QQ calibration; lower is better")
    ax.legend(frameon=False)

    fig.tight_layout()
    savefig(fig, os.path.join(PLOT_OUTDIR, "AGGREGATE__QQ_RMSE"))


    # ------------------------------------------------------------
    # Aggregate plot 2: mean Q/d
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(max(11, 0.8 * summary_df.shape[0]), 5.2))

    for j, model in enumerate(models):
        vals = []
        for ds in datasets:
            sub = posthoc_df[(posthoc_df["dataset"] == ds) & (posthoc_df["model"] == model)]
            vals.append(float(sub["mean_Q_over_d"].iloc[0]))
        ax.bar(x - 0.4 + width / 2 + j * width, vals, width=width, label=model)

    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=60, ha="right")
    ax.set_ylabel(r"Mean $Q_H/d$")
    ax.set_title(r"Aggregate mean norm calibration; ideal is $Q_H/d=1$")
    ax.legend(frameon=False)

    fig.tight_layout()
    savefig(fig, os.path.join(PLOT_OUTDIR, "AGGREGATE__MEAN_Q_OVER_D"))


    # ------------------------------------------------------------
    # Aggregate plot 3: abs mean error
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(max(11, 0.8 * summary_df.shape[0]), 5.2))

    for j, model in enumerate(models):
        vals = []
        for ds in datasets:
            sub = posthoc_df[(posthoc_df["dataset"] == ds) & (posthoc_df["model"] == model)]
            vals.append(float(sub["abs_mean_error"].iloc[0]))
        ax.bar(x - 0.4 + width / 2 + j * width, vals, width=width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=60, ha="right")
    ax.set_ylabel(r"$|\mathbb{E}[Q_H/d]-1|$")
    ax.set_title("Aggregate mean calibration error; lower is better")
    ax.legend(frameon=False)

    fig.tight_layout()
    savefig(fig, os.path.join(PLOT_OUTDIR, "AGGREGATE__ABS_MEAN_ERROR"))


    # ------------------------------------------------------------
    # Aggregate plot 4: original NLL gain, isotropic - CLT
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(max(11, 0.8 * summary_df.shape[0]), 5.2))

    nll_gain = summary_df["mean_nll_per_dim_gain_iso_minus_clt"].astype(float).values

    ax.bar(x, nll_gain)
    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=60, ha="right")
    ax.set_ylabel("Mean NLL per dim: isotropic - CLT")
    ax.set_title("Likelihood gain; positive means CLT fits better than isotropic")

    fig.tight_layout()
    savefig(fig, os.path.join(PLOT_OUTDIR, "AGGREGATE__NLL_GAIN_ISO_MINUS_CLT"))


    # ------------------------------------------------------------
    # Aggregate plot 5: z variance
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(max(11, 0.8 * summary_df.shape[0]), 5.2))

    z_clt_var = summary_df["z_clt_var"].astype(float).values
    z_iso_var = summary_df["z_iso_var"].astype(float).values

    ax.bar(x - 0.2, z_iso_var, width=0.4, label="isotropic")
    ax.bar(x + 0.2, z_clt_var, width=0.4, label="CLT")

    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=60, ha="right")
    ax.set_ylabel("Whitened z-coordinate variance")
    ax.set_title("Coordinate-wise variance calibration; ideal is 1")
    ax.legend(frameon=False)

    fig.tight_layout()
    savefig(fig, os.path.join(PLOT_OUTDIR, "AGGREGATE__Z_VARIANCE"))


    # ------------------------------------------------------------
    # Aggregate plot 6: CLT vs isotropic scatter summaries
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6.6, 6.2))

    clt_rmse = summary_df["clt_qq_rmse_Q_over_d"].astype(float).values
    iso_rmse = summary_df["iso_qq_rmse_Q_over_d"].astype(float).values

    ax.scatter(iso_rmse, clt_rmse, s=70)

    for ds, xi, yi in zip(datasets, iso_rmse, clt_rmse):
        ax.text(xi, yi, ds, fontsize=8, ha="left", va="bottom")

    lim = np.nanpercentile(np.concatenate([clt_rmse, iso_rmse]), 98)
    lim = max(lim, 1.0)

    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.2, color="gray")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("isotropic QQ-RMSE")
    ax.set_ylabel("CLT QQ-RMSE")
    ax.set_title("Dataset-level QQ-RMSE comparison\nbelow diagonal means CLT better")

    fig.tight_layout()
    savefig(fig, os.path.join(PLOT_OUTDIR, "AGGREGATE__CLT_VS_ISO_QQ_RMSE_SCATTER"))


    # ------------------------------------------------------------
    # Aggregate plot 7: calibration scale alpha
    # ------------------------------------------------------------

    if MAKE_CALIBRATED_CLT:
        fig, ax = plt.subplots(figsize=(max(11, 0.8 * summary_df.shape[0]), 5.2))

        alpha_vals = []
        for ds in datasets:
            sub = posthoc_df[(posthoc_df["dataset"] == ds) & (posthoc_df["model"] == "CLT")]
            alpha_vals.append(float(sub["alpha_clt_scale"].iloc[0]))

        ax.bar(x, alpha_vals)
        ax.axhline(1.0, linestyle="--", linewidth=1.2, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=60, ha="right")
        ax.set_ylabel(r"CLT scale factor $\alpha = \mathbb{E}[Q_{\rm CLT}/d]$")
        ax.set_title(r"Scalar noise-scale correction needed for CLT")

        fig.tight_layout()
        savefig(fig, os.path.join(PLOT_OUTDIR, "AGGREGATE__CLT_SCALE_ALPHA"))

    print("\nDONE.")
    print("Plots saved to:", os.path.basename(PLOT_OUTDIR))


def run_perpert_nll_recomputed_cov():
    global _PERPERT_ALL_DF, _PERPERT_RUN_DIR

    PERT_KEY = "perturbation"
    CONTROL_LABEL = "control"
    SEP = "_"
    COV_SHRINK0 = 1e-3
    JITTER = 1e-8
    COV_MAX_CELLS_PER_GROUP = 4000
    CONTROL_NULL_NA_CAP = 4000
    PERT_LIKE_NB_CAP = 4000
    SAMPLE_WITH_REPLACEMENT = True
    N_LABEL_TOP = 10

    def gaussian_nll_from_eigbasis(dx, lam, V, nA, nB):
        """
        Compute NLL under:
            H_CLT = Sigma0 * (1/nA + 1/nB)
            H_iso = mean(eig(H_CLT)) * I

        Work in Sigma0 eigenbasis.
        """
        d = len(lam)

        y = V.T @ dx

        h_clt = lam * ((1.0 / float(nA)) + (1.0 / float(nB)))
        h_clt = np.maximum(h_clt, JITTER)

        sigma2_iso = float(np.mean(h_clt))
        sigma2_iso = max(sigma2_iso, JITTER)

        q_clt = float(np.sum((y * y) / h_clt))
        q_iso = float(np.sum((y * y) / sigma2_iso))

        nll_clt = 0.5 * (np.sum(np.log(2.0 * np.pi * h_clt)) + q_clt)
        nll_iso = 0.5 * (d * np.log(2.0 * np.pi * sigma2_iso) + q_iso)

        return nll_clt, nll_iso, q_clt, q_iso

    def run_one_dataset(data_path, outdir, seed=0):
        rng = np.random.default_rng(seed)

        dataset_name = os.path.basename(data_path).replace(".h5ad", "")

        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        adata = ad.read_h5ad(data_path)
        _safe_gene_name_array(adata)

        if PERT_KEY not in adata.obs.columns:
            raise ValueError(f"{data_path} does not contain obs['{PERT_KEY}'].")

        # --------------------------------------------------------
        # Gene filtering, matching your original script
        # --------------------------------------------------------
        gene_means = adata.X.mean(axis=0).A1 if issparse(adata.X) else np.asarray(adata.X).mean(axis=0)
        valid_genes = set(adata.var_names[np.where(gene_means >= float(EXPRESSION_THRESHOLD))[0]])

        all_perturbed_genes = set()

        for pert in adata.obs[PERT_KEY].astype(str).unique():
            if pert == CONTROL_LABEL:
                continue

            for g in str(pert).split(SEP):
                if g in set(adata.var_names):
                    all_perturbed_genes.add(g)

            parsed = pert_to_gene_safe(pert)
            if parsed in set(adata.var_names):
                all_perturbed_genes.add(parsed)

        keep_genes = list(valid_genes | all_perturbed_genes)
        adata = adata[:, adata.var_names.isin(keep_genes)].copy()

        gene_names = _safe_gene_name_array(adata)
        G = len(gene_names)

        # --------------------------------------------------------
        # Perturbation filtering
        # --------------------------------------------------------
        pert_counts = adata.obs[PERT_KEY].astype(str).value_counts()
        valid_perts = pert_counts[pert_counts >= int(MIN_SAMPLES)].index.astype(str).tolist()

        if CONTROL_LABEL not in valid_perts:
            raise ValueError(f"Control label '{CONTROL_LABEL}' has fewer than MIN_SAMPLES or is missing.")

        adata = adata[adata.obs[PERT_KEY].astype(str).isin(valid_perts)].copy()
        obs_pert = adata.obs[PERT_KEY].astype(str).values

        X0_all = to_dense(adata[obs_pert == CONTROL_LABEL].X).astype(np.float32, copy=False)
        n0_full = int(X0_all.shape[0])

        perts_noncontrol = [p for p in np.unique(obs_pert) if p != CONTROL_LABEL]

        print(f"n_cells after filtering: {adata.n_obs}")
        print(f"n_genes after filtering: {G}")
        print(f"n_control cells:         {n0_full}")
        print(f"n_valid perturbations:   {len(perts_noncontrol)}")

        # --------------------------------------------------------
        # Estimate Sigma0
        # --------------------------------------------------------
        X0_cov = _subsample_rows(X0_all, COV_MAX_CELLS_PER_GROUP, rng).astype(np.float64, copy=False)
        n0_cov = int(X0_cov.shape[0])

        print(f"Estimating Sigma0 from {n0_cov} control cells...")

        Sigma0 = cipher.compute_covariance(X0_cov, shrink=COV_SHRINK0)

        lam, V = _eig_psd(Sigma0, jitter=JITTER)
        d = int(G)

        nA_default = n0_full if CONTROL_NULL_NA_CAP is None else min(int(CONTROL_NULL_NA_CAP), n0_full)
        nA_default = max(2, int(nA_default))

        rows = []

        # --------------------------------------------------------
        # Per-perturbation-size control-control null
        # --------------------------------------------------------
        for p in tqdm(perts_noncontrol, desc=f"{dataset_name}: per-pert NLL null"):

            n_p_raw = int(np.sum(obs_pert == p))

            nB = n_p_raw
            if PERT_LIKE_NB_CAP is not None:
                nB = min(nB, int(PERT_LIKE_NB_CAP))
            nB = max(2, int(nB))

            nll_clt_reps = []
            nll_iso_reps = []
            q_clt_reps = []
            q_iso_reps = []

            for r in range(N_REPS_PER_PERT):

                if SAMPLE_WITH_REPLACEMENT:
                    nA = nA_default
                    meanA = _sample_mean_from_rows(X0_all, nA, rng, replace=True)
                    meanB = _sample_mean_from_rows(X0_all, nB, rng, replace=True)
                else:
                    nB_eff = min(nB, max(2, n0_full // 2))
                    nA = min(nA_default, max(2, n0_full - nB_eff))

                    perm = rng.permutation(n0_full)
                    idxA = perm[:nA]
                    idxB = perm[nA:nA + nB_eff]

                    meanA = _mean_axis0(X0_all[idxA])
                    meanB = _mean_axis0(X0_all[idxB])

                dx = meanA - meanB

                nll_clt, nll_iso, q_clt, q_iso = gaussian_nll_from_eigbasis(
                    dx=dx,
                    lam=lam,
                    V=V,
                    nA=nA,
                    nB=nB,
                )

                nll_clt_reps.append(nll_clt)
                nll_iso_reps.append(nll_iso)
                q_clt_reps.append(q_clt)
                q_iso_reps.append(q_iso)

            nll_clt_reps = np.asarray(nll_clt_reps, dtype=float)
            nll_iso_reps = np.asarray(nll_iso_reps, dtype=float)
            q_clt_reps = np.asarray(q_clt_reps, dtype=float)
            q_iso_reps = np.asarray(q_iso_reps, dtype=float)

            rows.append({
                "dataset": dataset_name,
                "perturbation": p,
                "n_p_raw": n_p_raw,
                "nB_used": nB,
                "nA_used": nA_default,
                "n_reps": N_REPS_PER_PERT,

                "clt_nll_per_dim_mean": float(np.mean(nll_clt_reps) / d),
                "iso_nll_per_dim_mean": float(np.mean(nll_iso_reps) / d),
                "clt_nll_per_dim_std": float(np.std(nll_clt_reps / d, ddof=1)),
                "iso_nll_per_dim_std": float(np.std(nll_iso_reps / d, ddof=1)),

                "nll_gain_iso_minus_clt_per_dim": float(np.mean((nll_iso_reps - nll_clt_reps) / d)),
                "nll_gain_iso_minus_clt_per_dim_std": float(np.std((nll_iso_reps - nll_clt_reps) / d, ddof=1)),

                "clt_Q_over_d_mean": float(np.mean(q_clt_reps / d)),
                "iso_Q_over_d_mean": float(np.mean(q_iso_reps / d)),
                "clt_better_fraction": float(np.mean(nll_clt_reps < nll_iso_reps)),
            })

        df = pd.DataFrame(rows)

        csv_path = os.path.join(outdir, f"{dataset_name}__per_perturbation_size_nll_null.csv")
        df.to_csv(csv_path, index=False)

        print(f"[saved] {csv_path}")

        # --------------------------------------------------------
        # Plot 1: main scatter, one point per perturbation
        # --------------------------------------------------------
        x = df["clt_nll_per_dim_mean"].values
        y = df["iso_nll_per_dim_mean"].values
        gains = df["nll_gain_iso_minus_clt_per_dim"].values
        nB_vals = df["nB_used"].values

        fig, ax = plt.subplots(figsize=(6.8, 6.4))

        sc = ax.scatter(
            x,
            y,
            c=np.log10(nB_vals),
            s=45,
            alpha=0.75,
            edgecolor="none",
        )

        set_equal_axes(ax, x, y)

        ax.set_xlabel(r"CLT NLL / dimension")
        ax.set_ylabel(r"Isotropic NLL / dimension")
        ax.set_title(f"{dataset_name}\nOne point per perturbation-size null")

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(r"$\log_{10}(n_B)$")

        frac_above = float(np.mean(y > x))
        mean_gain = float(np.mean(gains))
        median_gain = float(np.median(gains))

        ax.text(
            0.03,
            0.97,
            (
                f"fraction CLT better = {frac_above:.3f}\n"
                f"mean iso-CLT NLL/d = {mean_gain:.3g}\n"
                f"median iso-CLT NLL/d = {median_gain:.3g}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.5),
        )

        # Label strongest CLT wins
        if N_LABEL_TOP is not None and N_LABEL_TOP > 0 and len(df) > 0:
            top = df.sort_values("nll_gain_iso_minus_clt_per_dim", ascending=False).head(N_LABEL_TOP)
            for _, rr in top.iterrows():
                ax.text(
                    rr["clt_nll_per_dim_mean"],
                    rr["iso_nll_per_dim_mean"],
                    str(rr["perturbation"]),
                    fontsize=7,
                    ha="left",
                    va="bottom",
                )

        fig.tight_layout()

        png = os.path.join(outdir, f"{dataset_name}__NLL_PER_DIM_SCATTER_iso_vs_clt.png")
        svg = os.path.join(outdir, f"{dataset_name}__NLL_PER_DIM_SCATTER_iso_vs_clt.svg")

        fig.savefig(png, dpi=250, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.show()

        # --------------------------------------------------------
        # Plot 2: distribution of per-perturbation gains
        # --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7.0, 4.8))

        ax.hist(gains, bins=40, alpha=0.8)
        ax.axvline(0.0, linestyle="--", linewidth=1.3, color="gray")
        ax.axvline(mean_gain, linestyle="-", linewidth=1.5, color="black", label=f"mean = {mean_gain:.3g}")

        ax.set_xlabel(r"Isotropic NLL/d $-$ CLT NLL/d")
        ax.set_ylabel("Number of perturbations")
        ax.set_title(f"{dataset_name}: per-perturbation-size likelihood gain")
        ax.legend(frameon=False)

        fig.tight_layout()

        png = os.path.join(outdir, f"{dataset_name}__NLL_GAIN_HIST.png")
        svg = os.path.join(outdir, f"{dataset_name}__NLL_GAIN_HIST.svg")

        fig.savefig(png, dpi=250, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.show()

        # --------------------------------------------------------
        # Plot 3: gain vs sample size
        # --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7.0, 4.8))

        ax.scatter(nB_vals, gains, s=35, alpha=0.75)
        ax.axhline(0.0, linestyle="--", linewidth=1.2, color="gray")
        ax.set_xscale("log")

        ax.set_xlabel(r"Perturbation sample size $n_B$")
        ax.set_ylabel(r"Isotropic NLL/d $-$ CLT NLL/d")
        ax.set_title(f"{dataset_name}: likelihood gain vs perturbation-size")

        fig.tight_layout()

        png = os.path.join(outdir, f"{dataset_name}__NLL_GAIN_vs_nB.png")
        svg = os.path.join(outdir, f"{dataset_name}__NLL_GAIN_vs_nB.svg")

        fig.savefig(png, dpi=250, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.show()

        # --------------------------------------------------------
        # Dataset summary
        # --------------------------------------------------------
        summary = {
            "dataset": dataset_name,
            "data_path": data_path,
            "n_cells_after_filtering": int(adata.n_obs),
            "n_genes": int(G),
            "n_control_cells": int(n0_full),
            "n_control_cells_for_cov": int(n0_cov),
            "n_valid_perturbations": int(len(perts_noncontrol)),
            "n_reps_per_pert": int(N_REPS_PER_PERT),
            "mean_gain_iso_minus_clt_per_dim": float(np.mean(gains)),
            "median_gain_iso_minus_clt_per_dim": float(np.median(gains)),
            "fraction_perturbations_clt_better": float(np.mean(gains > 0)),
            "mean_clt_nll_per_dim": float(np.mean(x)),
            "mean_iso_nll_per_dim": float(np.mean(y)),
            "mean_clt_Q_over_d": float(df["clt_Q_over_d_mean"].mean()),
            "mean_iso_Q_over_d": float(df["iso_Q_over_d_mean"].mean()),
        }

        json_path = os.path.join(outdir, f"{dataset_name}__per_perturbation_size_nll_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\nDataset summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        return df, summary

    run_outdir = os.path.join(OUTDIR, "perpert_nll_recomputed")
    os.makedirs(run_outdir, exist_ok=True)
    data_paths = [os.path.join(DATA_DIR, name) for name in DATASET_NAMES]

    all_dfs = []
    all_summaries = []

    for i, data_path in enumerate(data_paths):
        try:
            df_i, summary_i = run_one_dataset(
                data_path=data_path,
                outdir=run_outdir,
                seed=SEED + i,
            )

            all_dfs.append(df_i)
            all_summaries.append(summary_i)

        except Exception as e:
            print("\n" + "!" * 100)
            print(f"[ERROR] Failed on {data_path}")
            print(repr(e))
            print("!" * 100 + "\n")


    # ============================================================
    # AGGREGATE PLOTS
    # ============================================================

    if len(all_dfs) > 0:
        all_df = pd.concat(all_dfs, ignore_index=True)
        all_summary_df = pd.DataFrame(all_summaries)

        all_csv = os.path.join(run_outdir, "ALL_DATASETS__per_perturbation_size_nll_null.csv")
        all_summary_csv = os.path.join(run_outdir, "ALL_DATASETS__per_perturbation_size_nll_summary.csv")

        all_df.to_csv(all_csv, index=False)
        all_summary_df.to_csv(all_summary_csv, index=False)

        print(f"[saved] {all_csv}")
        print(f"[saved] {all_summary_csv}")

        # --------------------------------------------------------
        # Aggregate scatter: all perturbation-size points
        # --------------------------------------------------------
        x = all_df["clt_nll_per_dim_mean"].values
        y = all_df["iso_nll_per_dim_mean"].values
        gains = all_df["nll_gain_iso_minus_clt_per_dim"].values

        fig, ax = plt.subplots(figsize=(7.0, 6.6))

        dataset_codes, dataset_uniques = pd.factorize(all_df["dataset"])

        sc = ax.scatter(
            x,
            y,
            c=dataset_codes,
            s=28,
            alpha=0.65,
            edgecolor="none",
        )

        set_equal_axes(ax, x, y)

        ax.set_xlabel(r"CLT NLL / dimension")
        ax.set_ylabel(r"Isotropic NLL / dimension")
        ax.set_title("All datasets: one point per perturbation-size null")

        frac_clt_better = float(np.mean(y > x))
        mean_gain = float(np.mean(gains))
        median_gain = float(np.median(gains))

        ax.text(
            0.03,
            0.97,
            (
                f"fraction CLT better = {frac_clt_better:.3f}\n"
                f"mean iso-CLT NLL/d = {mean_gain:.3g}\n"
                f"median iso-CLT NLL/d = {median_gain:.3g}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.5),
        )

        fig.tight_layout()

        png = os.path.join(run_outdir, "ALL_DATASETS__NLL_PER_DIM_SCATTER_iso_vs_clt.png")
        svg = os.path.join(run_outdir, "ALL_DATASETS__NLL_PER_DIM_SCATTER_iso_vs_clt.svg")

        fig.savefig(png, dpi=250, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.show()

        # --------------------------------------------------------
        # Aggregate dataset-level summary barplot
        # --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(max(11, 0.8 * len(all_summary_df)), 5.0))

        xloc = np.arange(len(all_summary_df))
        vals = all_summary_df["mean_gain_iso_minus_clt_per_dim"].values

        ax.bar(xloc, vals)
        ax.axhline(0.0, linestyle="--", linewidth=1.2, color="gray")

        ax.set_xticks(xloc)
        ax.set_xticklabels(all_summary_df["dataset"], rotation=60, ha="right")
        ax.set_ylabel(r"Mean isotropic NLL/d $-$ CLT NLL/d")
        ax.set_title("Per-perturbation-size null likelihood gain by dataset")

        fig.tight_layout()

        png = os.path.join(run_outdir, "ALL_DATASETS__MEAN_NLL_GAIN_BY_DATASET.png")
        svg = os.path.join(run_outdir, "ALL_DATASETS__MEAN_NLL_GAIN_BY_DATASET.svg")

        fig.savefig(png, dpi=250, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.show()

        # --------------------------------------------------------
        # Aggregate fraction CLT better by dataset
        # --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(max(11, 0.8 * len(all_summary_df)), 5.0))

        vals = all_summary_df["fraction_perturbations_clt_better"].values

        ax.bar(xloc, vals)
        ax.axhline(0.5, linestyle="--", linewidth=1.2, color="gray")

        ax.set_xticks(xloc)
        ax.set_xticklabels(all_summary_df["dataset"], rotation=60, ha="right")
        ax.set_ylabel("Fraction of perturbation-size nulls where CLT wins")
        ax.set_ylim(0, 1.05)
        ax.set_title("How often CLT beats isotropic across perturbation labels")

        fig.tight_layout()

        png = os.path.join(run_outdir, "ALL_DATASETS__FRACTION_CLT_BETTER_BY_DATASET.png")
        svg = os.path.join(run_outdir, "ALL_DATASETS__FRACTION_CLT_BETTER_BY_DATASET.svg")

        fig.savefig(png, dpi=250, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.show()

    _PERPERT_ALL_DF = pd.concat(all_dfs, ignore_index=True) if len(all_dfs) > 0 else pd.DataFrame()
    _PERPERT_RUN_DIR = run_outdir


def plot_perpert_nll_scatter():
    SAVE_FIG = True
    SAVE_SVG = True
    SHOW_FIG = True
    DPI = 300
    POINT_SIZE = 32
    POINT_ALPHA = 0.72

    RUN_DIR = os.path.join(OUTDIR, "perpert_nll_recomputed")

    # -----------------------------
    # LOAD AGGREGATE CSV
    # -----------------------------

    aggregate_csv = os.path.join(RUN_DIR, "ALL_DATASETS__per_perturbation_size_nll_null.csv")

    if os.path.exists(aggregate_csv):
        all_df = pd.read_csv(aggregate_csv)
    else:
        # Fallback: concatenate per-dataset files if aggregate file is missing
        files = sorted(glob.glob(os.path.join(RUN_DIR, "*__per_perturbation_size_nll_null.csv")))
        files = [f for f in files if "ALL_DATASETS" not in os.path.basename(f)]

        if len(files) == 0:
            raise FileNotFoundError(
                f"Could not find aggregate or per-dataset NLL-null CSV files in {RUN_DIR}"
            )

        all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    print(f"Loaded {len(all_df)} rows")
    print("Datasets:")
    for ds, n in all_df["dataset"].value_counts().items():
        print(f"  {ds}: {n}")

    required_cols = [
        "dataset",
        "clt_nll_per_dim_mean",
        "iso_nll_per_dim_mean",
        "nll_gain_iso_minus_clt_per_dim",
    ]

    missing = [c for c in required_cols if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -----------------------------
    # PREPARE VALUES
    # -----------------------------

    x = all_df["clt_nll_per_dim_mean"].values.astype(float)
    y = all_df["iso_nll_per_dim_mean"].values.astype(float)
    gains = all_df["nll_gain_iso_minus_clt_per_dim"].values.astype(float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(gains)
    all_df = all_df.loc[ok].copy()

    x = all_df["clt_nll_per_dim_mean"].values.astype(float)
    y = all_df["iso_nll_per_dim_mean"].values.astype(float)
    gains = all_df["nll_gain_iso_minus_clt_per_dim"].values.astype(float)

    frac_clt_better = float(np.mean(y > x))
    mean_gain = float(np.mean(gains))
    median_gain = float(np.median(gains))

    # -----------------------------
    # COLORS
    # -----------------------------

    datasets = sorted(all_df["dataset"].unique().tolist())
    n_ds = len(datasets)

    # tab20 handles up to 20 nicely
    cmap = plt.get_cmap("tab20", max(n_ds, 1))
    dataset_to_color = {ds: cmap(i) for i, ds in enumerate(datasets)}

    # -----------------------------
    # AXIS LIMITS
    # -----------------------------

    vals = np.concatenate([x, y])
    lo = np.nanpercentile(vals, 0.5)
    hi = np.nanpercentile(vals, 99.5)

    pad = 0.08 * (hi - lo)
    lo = lo - pad
    hi = hi + pad

    # Keep full diagonal visible
    lo = min(lo, np.nanmin(vals))
    hi = max(hi, np.nanmax(vals))

    # -----------------------------
    # PLOT
    # -----------------------------

    fig, ax = plt.subplots(figsize=(11.2, 8.0))

    for ds in datasets:
        sub = all_df[all_df["dataset"] == ds]

        ax.scatter(
            sub["clt_nll_per_dim_mean"],
            sub["iso_nll_per_dim_mean"],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=dataset_to_color[ds],
            edgecolor="none",
            label=f"{ds} (n={len(sub)})",
        )

    # y=x diagonal
    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1.4,
        color="gray",
        zorder=0,
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("CLT NLL / dimension")
    ax.set_ylabel("Isotropic NLL / dimension")
    ax.set_title("All datasets: one point per perturbation-size null")

    ax.text(
        0.03,
        0.97,
        (
            f"fraction CLT better = {frac_clt_better:.3f}\n"
            f"mean iso-CLT NLL/d = {mean_gain:.3g}\n"
            f"median iso-CLT NLL/d = {median_gain:.3g}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, linewidth=0.6),
    )

    # Legend outside plot
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Dataset",
        title_fontsize=9,
        markerscale=1.1,
    )

    plt.tight_layout()

    # -----------------------------
    # SAVE
    # -----------------------------

    out_png = os.path.join(RUN_DIR, "ALL_DATASETS__NLL_PER_DIM_SCATTER_by_dataset_legend.png")
    out_svg = os.path.join(RUN_DIR, "ALL_DATASETS__NLL_PER_DIM_SCATTER_by_dataset_legend.svg")
    out_pdf = os.path.join(RUN_DIR, "ALL_DATASETS__NLL_PER_DIM_SCATTER_by_dataset_legend.pdf")

    if SAVE_FIG:
        fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
        print("[saved]", out_png)

        if SAVE_SVG:
            fig.savefig(out_svg, bbox_inches="tight")
            fig.savefig(out_pdf, bbox_inches="tight")
            print("[saved]", out_svg)
            print("[saved]", out_pdf)

    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)

    # -----------------------------
    # OPTIONAL: DATASET-LEVEL SUMMARY TABLE
    # -----------------------------

    summary = (
        all_df
        .groupby("dataset")
        .agg(
            n_points=("dataset", "size"),
            mean_clt_nll_per_dim=("clt_nll_per_dim_mean", "mean"),
            mean_iso_nll_per_dim=("iso_nll_per_dim_mean", "mean"),
            mean_gain_iso_minus_clt=("nll_gain_iso_minus_clt_per_dim", "mean"),
            median_gain_iso_minus_clt=("nll_gain_iso_minus_clt_per_dim", "median"),
            fraction_clt_better=("nll_gain_iso_minus_clt_per_dim", lambda z: float(np.mean(np.asarray(z) > 0))),
        )
        .reset_index()
    )

    summary_csv = os.path.join(RUN_DIR, "ALL_DATASETS__NLL_SCATTER_by_dataset_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print("[saved]", summary_csv)

    display(summary)


def run_perpert_nll_precomputed_cov():
    global _PC_ALL_DF, _PC_RUN_DIR

    PRECOMP_ROOT = os.path.join(SUPPL, "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED")
    DATA_ROOTS = [DATA_DIR]
    DATA_PATH_MAP = {}
    SEARCH_DATA_ROOTS_RECURSIVELY = True
    EXPRESSION_CUTOFF = EXPRESSION_THRESHOLD
    SIGMA_FILES = {
        "shuffle": "Sigma_shuffled_ridge.npy",
        "mean_field": "Sigma_meanfield_ridge.npy",
        "true": "Sigma_full_ridge.npy",
    }
    PERT_KEY_CANDIDATES = [
        "perturbation",
        "gene",
        "target_gene",
        "condition",
        "guide_target",
        "perturbation_name",
    ]
    CONTROL_LABEL = "control"
    CONTROL_POOL_MAX_CELLS = 10000
    CONTROL_NULL_NA_CAP = 4000
    PERT_LIKE_NB_CAP = 4000
    SAMPLE_WITH_REPLACEMENT = True
    N_LABEL_TOP = 10
    JITTER = 1e-8
    SIGMA_SYMMETRIZE = True
    EIGEN_FLOOR = JITTER
    TRANSFORM_DTYPE = np.float32
    MAKE_PER_DATASET_PLOTS = True
    SHOW_PER_DATASET_FIGURES = False
    MAKE_AGGREGATE_PLOTS = True
    SHOW_AGGREGATE_FIGURES = True
    RUN_ONLY_N_DATASETS = None

    def ensure_dir(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path


    def cutoff_to_tag(value):
        return f"{float(value):.1f}".replace(".", "p")


    EXPRESSION_TAG = cutoff_to_tag(EXPRESSION_CUTOFF)


    def decode_arr(x):
        out = []
        for value in np.asarray(x):
            if isinstance(value, bytes):
                out.append(value.decode("utf-8"))
            else:
                out.append(str(value))
        return np.asarray(out, dtype=object)


    def nan0(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


    def sym(A):
        return 0.5 * (A + A.T)

    def mean_axis0(X):
        return np.asarray(X.mean(axis=0)).ravel()


    def is_match(name, keywords):
        name = str(name)
        return any(str(keyword) in name for keyword in keywords)


    def dataset_name_from_precomp_dir(path):
        return Path(path).name.split("__mean_ge_")[0]


    def find_one(root, filename):
        root = Path(root)
        direct = root / filename
        if direct.exists():
            return direct
        hits = list(root.rglob(filename))
        if not hits:
            raise FileNotFoundError(f"Could not find {filename!r} under {root}")
        return hits[0]


    def read_h5_scalar(h5, names):
        expanded = list(names) + [
            "n_control",
            "n_controls",
            "control_cells",
            "n_ctrl",
            "ctrl_n",
            "n0",
        ]

        seen = set()
        expanded = [x for x in expanded if not (x in seen or seen.add(x))]

        for name in expanded:
            if name in h5:
                value = np.asarray(h5[name][()]).reshape(-1)
                if value.size:
                    value = float(value[0])
                    if np.isfinite(value) and value > 0:
                        return value
            if name in h5.attrs:
                value = float(h5.attrs[name])
                if np.isfinite(value) and value > 0:
                    return value

        return None


    # ============================================================
    # FIND PRECOMPUTED DATASET DIRECTORIES
    # ============================================================

    def find_dataset_dirs():
        root = Path(PRECOMP_ROOT)
        pattern = f"*__mean_ge_{EXPRESSION_TAG}"

        all_dirs = sorted(path for path in root.glob(pattern) if path.is_dir())

        print(f"[precomp root]      {root}")
        print(f"[expression cutoff] mean >= {EXPRESSION_CUTOFF}")
        print(f"[folder tag]        __mean_ge_{EXPRESSION_TAG}")
        print(f"[found total]       {len(all_dirs)}")

        if not all_dirs:
            print("\n[available cutoff folders]")
            for path in sorted(root.glob("*__mean_ge_*"))[:100]:
                print(" ", path.name)
            raise FileNotFoundError(f"No folders found for pattern {pattern!r} under {root}")

        if ONLY_RUN_CRISPRI_A_LISTS:
            selected_keywords = CRISPRa_KEYWORDS + CRISPRi_KEYWORDS
            selected = []
            for path in all_dirs:
                dataset = dataset_name_from_precomp_dir(path)
                if is_match(dataset, selected_keywords):
                    selected.append(path)
        else:
            selected = all_dirs

        print(f"[selected]          {len(selected)}")
        for path in selected:
            dataset = dataset_name_from_precomp_dir(path)
            if is_match(dataset, CRISPRa_KEYWORDS):
                group = "CRISPRa"
            elif is_match(dataset, CRISPRi_KEYWORDS):
                group = "CRISPRi"
            else:
                group = "unknown"
            print(f"  [{group}] {dataset}")

        if RUN_ONLY_N_DATASETS is not None:
            selected = selected[:int(RUN_ONLY_N_DATASETS)]
            print(f"[debug] truncated to first {len(selected)} datasets")

        if not selected:
            raise FileNotFoundError("No datasets matched the active CRISPRa/CRISPRi lists.")

        return selected


    # ============================================================
    # FIND RAW H5AD FILE FOR CONTROL SAMPLING
    # ============================================================

    def find_h5ad_for_dataset(dataset):
        dataset = str(dataset)

        if dataset in DATA_PATH_MAP:
            path = Path(DATA_PATH_MAP[dataset])
            if path.exists():
                return path
            raise FileNotFoundError(f"DATA_PATH_MAP[{dataset!r}] points to missing file: {path}")

        candidates = []
        for root in DATA_ROOTS:
            root = Path(root)
            candidates.extend([
                root / f"{dataset}.h5ad",
                root / dataset / f"{dataset}.h5ad",
            ])

        for path in candidates:
            if path.exists():
                return path

        if SEARCH_DATA_ROOTS_RECURSIVELY:
            hits = []
            for root in DATA_ROOTS:
                root = Path(root)
                if root.exists():
                    hits.extend(root.rglob(f"{dataset}.h5ad"))
            hits = sorted(set(hits))
            if hits:
                return hits[0]

        searched = "\n".join(str(x) for x in candidates[:20])
        raise FileNotFoundError(
            f"Could not find raw .h5ad for dataset {dataset!r}.\n"
            f"Add it to DATA_PATH_MAP or DATA_ROOTS. Tried examples:\n{searched}"
        )


    # ============================================================
    # ANNDATA CONTROL POOL LOADING
    # ============================================================

    def detect_pert_key(adata):
        for key in PERT_KEY_CANDIDATES:
            if key in adata.obs.columns:
                return key
        raise KeyError(
            f"Could not find a perturbation key in adata.obs. Tried {PERT_KEY_CANDIDATES}. "
            f"Available columns include: {list(adata.obs.columns[:30])}"
        )


    def first_occurrence_indexer(names, query):
        mapping = {}
        for i, name in enumerate(map(str, names)):
            if name not in mapping:
                mapping[name] = i
        return np.asarray([mapping.get(str(x), -1) for x in query], dtype=np.int64)


    def get_gene_indexer_for_precomputed_genes(adata, genes):
        genes = np.asarray(genes, dtype=object)

        idx = first_occurrence_indexer(adata.var_names.astype(str), genes)
        if np.all(idx >= 0):
            return idx, "var_names"

        candidate_cols = [
            "gene_name",
            "gene_names",
            "gene_symbol",
            "gene_symbols",
            "feature_name",
            "feature_names",
            "symbol",
            "name",
        ]

        best_idx = idx
        best_source = "var_names"
        best_found = int(np.sum(idx >= 0))

        for col in candidate_cols:
            if col not in adata.var.columns:
                continue
            idx_col = first_occurrence_indexer(adata.var[col].astype(str).values, genes)
            found = int(np.sum(idx_col >= 0))
            if found > best_found:
                best_idx = idx_col
                best_source = f"var[{col!r}]"
                best_found = found
            if np.all(idx_col >= 0):
                return idx_col, f"var[{col!r}]"

        missing = genes[best_idx < 0]
        preview = ", ".join(map(str, missing[:20]))
        raise ValueError(
            f"Could not map all precomputed genes into the raw AnnData. "
            f"Best source={best_source}, matched={best_found}/{len(genes)}. "
            f"First missing genes: {preview}"
        )


    def load_control_pool_for_precomputed_genes(data_path, genes, rng):
        """
        Load only a random pool of control cells and only the precomputed genes.
        This is the only place the .h5ad expression matrix is touched.
        """
        data_path = Path(data_path)
        print(f"[h5ad] loading control pool only: {os.path.basename(str(data_path))}")

        adata = ad.read_h5ad(data_path, backed="r")
        try:
            pert_key = detect_pert_key(adata)
            obs_pert = adata.obs[pert_key].astype(str).values
            control_idx_all = np.where(obs_pert == str(CONTROL_LABEL))[0]

            if control_idx_all.size < 2:
                raise ValueError(
                    f"Found only {control_idx_all.size} control cells with "
                    f"obs[{pert_key!r}] == {CONTROL_LABEL!r}."
                )

            n_pool = int(control_idx_all.size)
            if CONTROL_POOL_MAX_CELLS is not None and CONTROL_POOL_MAX_CELLS > 0:
                n_pool = min(n_pool, int(CONTROL_POOL_MAX_CELLS))

            pool_idx = rng.choice(control_idx_all, size=n_pool, replace=False)
            pool_idx = np.sort(pool_idx)

            gene_idx, gene_source = get_gene_indexer_for_precomputed_genes(adata, genes)

            col_order = np.argsort(gene_idx)
            gene_idx_sorted = gene_idx[col_order]
            restore_cols = np.argsort(col_order)

            X = adata[pool_idx, gene_idx_sorted].X
            X = to_dense(X)
            X = np.asarray(X[:, restore_cols], dtype=TRANSFORM_DTYPE, order="C")

            if np.any(~np.isfinite(X)):
                print("[warn] non-finite values in control pool; replacing with zero")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(
                    TRANSFORM_DTYPE,
                    copy=False,
                )

            min_value = float(np.min(X)) if X.size else 0.0
            if min_value < 0:
                print(f"[warn] control expression pool contains negative values, min={min_value:.4g}")

            print(
                f"[control pool] key={pert_key!r} | source={gene_source} | "
                f"cells={X.shape[0]:,}/{control_idx_all.size:,} | genes={X.shape[1]:,}"
            )

            return X, {
                "data_path": str(data_path),
                "pert_key": pert_key,
                "gene_source": gene_source,
                "n_control_total_in_h5ad": int(control_idx_all.size),
                "n_control_pool_loaded": int(X.shape[0]),
            }

        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()


    # ============================================================
    # PRECOMPUTED STATS LOADING
    # ============================================================

    def load_precomputed_stats(ds_dir):
        ds_dir = Path(ds_dir)
        stats_path = find_one(ds_dir, "perturbation_stats.h5")

        with h5py.File(stats_path, "r") as h5:
            required = ["gene_names", "perturbations"]
            missing = [key for key in required if key not in h5]
            if missing:
                raise KeyError(f"Missing required keys in {stats_path}: {missing}")

            genes = decode_arr(h5["gene_names"][:])
            perturbations = decode_arr(h5["perturbations"][:])

            if "n_cells_pert" in h5:
                n_cells_pert = np.asarray(h5["n_cells_pert"][:], dtype=np.int64).reshape(-1)
            elif "n_pert" in h5:
                n_cells_pert = np.asarray(h5["n_pert"][:], dtype=np.int64).reshape(-1)
            elif "nu" in h5:
                n_cells_pert = np.asarray(h5["nu"][:], dtype=np.int64).reshape(-1)
            else:
                raise KeyError(
                    f"Could not find perturbation sample sizes in {stats_path}. "
                    "Expected n_cells_pert, n_pert, or nu."
                )

            n_control = read_h5_scalar(
                h5,
                ["n_cells_control", "n_control_cells", "control_n", "n0", "n_control"],
            )

        if n_cells_pert.shape[0] != perturbations.shape[0]:
            raise ValueError(
                f"n_cells_pert length {n_cells_pert.shape[0]} != "
                f"n perturbations {perturbations.shape[0]} in {stats_path}"
            )

        keep = np.asarray(perturbations, dtype=str) != str(CONTROL_LABEL)
        keep &= np.asarray(n_cells_pert >= int(MIN_SAMPLES), dtype=bool)

        perturbations = perturbations[keep]
        n_cells_pert = n_cells_pert[keep]

        return {
            "stats_path": str(stats_path),
            "genes": genes,
            "perturbations": perturbations,
            "n_cells_pert": n_cells_pert,
            "n_control_from_stats": None if n_control is None else float(n_control),
        }


    # ============================================================
    # LIKELIHOOD HELPERS
    # ============================================================

    def load_precomputed_sigma(ds_dir, method):
        if method not in SIGMA_FILES:
            raise KeyError(f"Unknown Sigma method {method!r}. Available: {list(SIGMA_FILES)}")

        sigma_path = find_one(ds_dir, SIGMA_FILES[method])
        Sigma = np.asarray(np.load(sigma_path, mmap_mode="r"), dtype=np.float64)
        Sigma = nan0(Sigma)

        if SIGMA_SYMMETRIZE:
            Sigma = sym(Sigma)

        print(f"[Sigma] {method}: {os.path.basename(str(sigma_path))} | shape={Sigma.shape}")
        return Sigma, str(sigma_path)


    def eig_precomputed_sigma(Sigma):
        lam, V = np.linalg.eigh(Sigma)
        lam = np.nan_to_num(lam, nan=0.0, posinf=0.0, neginf=0.0)
        lam = np.maximum(lam, float(EIGEN_FLOOR))
        V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
        return lam.astype(np.float64, copy=False), V.astype(np.float64, copy=False)


    def sample_mean_from_pool(X, n, rng, replace=True):
        n = int(n)
        N = int(X.shape[0])
        if n <= 0:
            raise ValueError("Sample size must be positive.")
        if replace:
            idx = rng.choice(N, size=n, replace=True)
        else:
            if n > N:
                raise ValueError(f"Cannot sample n={n} without replacement from pool N={N}.")
            idx = rng.choice(N, size=n, replace=False)
        return mean_axis0(X[idx])


    def nlls_for_null_reps_in_eigbasis(X0_eig, lam, nA, nB, n_reps, rng):
        """
        X0_eig is the control expression pool projected into Sigma eigenbasis.
        If dx = meanA - meanB in original coordinates, then y = V.T dx.
        Because X0_eig = X0 @ V, y = meanA_eig - meanB_eig.
        """
        d = int(lam.size)

        nA = int(nA)
        nB = int(nB)

        h_clt = lam * ((1.0 / float(nA)) + (1.0 / float(nB)))
        h_clt = np.maximum(h_clt, float(JITTER))
        inv_h_clt = 1.0 / h_clt

        sigma2_iso = max(float(np.mean(h_clt)), float(JITTER))
        inv_sigma2_iso = 1.0 / sigma2_iso

        const_clt = 0.5 * float(np.sum(np.log(2.0 * np.pi * h_clt)))
        const_iso = 0.5 * float(d * np.log(2.0 * np.pi * sigma2_iso))

        nll_clt = np.empty(int(n_reps), dtype=np.float64)
        nll_iso = np.empty(int(n_reps), dtype=np.float64)
        q_clt = np.empty(int(n_reps), dtype=np.float64)
        q_iso = np.empty(int(n_reps), dtype=np.float64)

        for r in range(int(n_reps)):
            if SAMPLE_WITH_REPLACEMENT:
                meanA = sample_mean_from_pool(X0_eig, nA, rng, replace=True)
                meanB = sample_mean_from_pool(X0_eig, nB, rng, replace=True)
            else:
                N = int(X0_eig.shape[0])
                nB_eff = min(nB, max(2, N // 2))
                nA_eff = min(nA, max(2, N - nB_eff))
                perm = rng.permutation(N)
                meanA = mean_axis0(X0_eig[perm[:nA_eff]])
                meanB = mean_axis0(X0_eig[perm[nA_eff:nA_eff + nB_eff]])

            y = meanA - meanB
            y2 = np.asarray(y, dtype=np.float64) ** 2

            q_clt[r] = float(np.sum(y2 * inv_h_clt))
            q_iso[r] = float(np.sum(y2) * inv_sigma2_iso)

            nll_clt[r] = const_clt + 0.5 * q_clt[r]
            nll_iso[r] = const_iso + 0.5 * q_iso[r]

        return nll_clt, nll_iso, q_clt, q_iso


    # ============================================================
    # PLOTTING
    # ============================================================

    def save_or_show(fig, path_base, show=False):
        path_base = Path(path_base)
        fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


    def plot_dataset_outputs(df, output_dir, dataset, method):
        output_dir = Path(output_dir)
        x = df["clt_nll_per_dim_mean"].values
        y = df["iso_nll_per_dim_mean"].values
        gains = df["nll_gain_iso_minus_clt_per_dim"].values
        nB_vals = df["nB_used"].values

        fig, ax = plt.subplots(figsize=(6.8, 6.4))
        sc = ax.scatter(
            x,
            y,
            c=np.log10(np.maximum(nB_vals, 1)),
            s=45,
            alpha=0.75,
            edgecolor="none",
        )
        set_equal_axes(ax, x, y)
        ax.set_xlabel("CLT NLL / dimension")
        ax.set_ylabel("Isotropic NLL / dimension")
        ax.set_title(f"{dataset} | {method} Sigma")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("log10(n_B)")

        frac_clt_better = float(np.mean(y > x))
        mean_gain = float(np.mean(gains))
        median_gain = float(np.median(gains))
        ax.text(
            0.03,
            0.97,
            (
                f"fraction CLT better = {frac_clt_better:.3f}\n"
                f"mean iso-CLT NLL/d = {mean_gain:.3g}\n"
                f"median iso-CLT NLL/d = {median_gain:.3g}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.5),
        )

        if N_LABEL_TOP is not None and N_LABEL_TOP > 0 and len(df) > 0:
            top = df.sort_values("nll_gain_iso_minus_clt_per_dim", ascending=False).head(int(N_LABEL_TOP))
            for _, rr in top.iterrows():
                ax.text(
                    rr["clt_nll_per_dim_mean"],
                    rr["iso_nll_per_dim_mean"],
                    str(rr["perturbation"]),
                    fontsize=7,
                    ha="left",
                    va="bottom",
                )

        fig.tight_layout()
        save_or_show(
            fig,
            output_dir / f"{dataset}__{method}__NLL_PER_DIM_SCATTER_iso_vs_clt",
            SHOW_PER_DATASET_FIGURES,
        )

        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.hist(gains, bins=40, alpha=0.8)
        ax.axvline(0.0, linestyle="--", linewidth=1.3, color="gray")
        ax.axvline(mean_gain, linestyle="-", linewidth=1.5, color="black", label=f"mean = {mean_gain:.3g}")
        ax.set_xlabel("Isotropic NLL/d - CLT NLL/d")
        ax.set_ylabel("Number of perturbations")
        ax.set_title(f"{dataset} | {method} Sigma")
        ax.legend(frameon=False)
        fig.tight_layout()
        save_or_show(
            fig,
            output_dir / f"{dataset}__{method}__NLL_GAIN_HIST",
            SHOW_PER_DATASET_FIGURES,
        )

        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.scatter(nB_vals, gains, s=35, alpha=0.75)
        ax.axhline(0.0, linestyle="--", linewidth=1.2, color="gray")
        ax.set_xscale("log")
        ax.set_xlabel("Perturbation sample size n_B")
        ax.set_ylabel("Isotropic NLL/d - CLT NLL/d")
        ax.set_title(f"{dataset} | {method} Sigma")
        fig.tight_layout()
        save_or_show(
            fig,
            output_dir / f"{dataset}__{method}__NLL_GAIN_vs_nB",
            SHOW_PER_DATASET_FIGURES,
        )


    def plot_aggregate_outputs(all_df, run_outdir):
        run_outdir = Path(run_outdir)

        for method, dfm in all_df.groupby("sigma_method"):
            x = dfm["clt_nll_per_dim_mean"].values
            y = dfm["iso_nll_per_dim_mean"].values
            gains = dfm["nll_gain_iso_minus_clt_per_dim"].values

            fig, ax = plt.subplots(figsize=(7.0, 6.6))
            dataset_codes, _ = pd.factorize(dfm["dataset"])
            ax.scatter(x, y, c=dataset_codes, s=28, alpha=0.65, edgecolor="none")
            set_equal_axes(ax, x, y)
            ax.set_xlabel("CLT NLL / dimension")
            ax.set_ylabel("Isotropic NLL / dimension")
            ax.set_title(f"All datasets | {method} Sigma")

            ax.text(
                0.03,
                0.97,
                (
                    f"fraction CLT better = {float(np.mean(y > x)):.3f}\n"
                    f"mean iso-CLT NLL/d = {float(np.mean(gains)):.3g}\n"
                    f"median iso-CLT NLL/d = {float(np.median(gains)):.3g}\n"
                    f"points = {len(dfm):,}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.5),
            )
            fig.tight_layout()
            save_or_show(
                fig,
                run_outdir / f"ALL_DATASETS__{method}__NLL_PER_DIM_SCATTER_iso_vs_clt",
                SHOW_AGGREGATE_FIGURES,
            )

            fig, ax = plt.subplots(figsize=(8.0, 5.0))
            dataset_order = (
                dfm.groupby("dataset")["nll_gain_iso_minus_clt_per_dim"]
                .median()
                .sort_values(ascending=False)
                .index
            )
            data = [
                dfm.loc[dfm["dataset"] == ds, "nll_gain_iso_minus_clt_per_dim"].values
                for ds in dataset_order
            ]
            ax.boxplot(data, labels=dataset_order, showfliers=False)
            ax.axhline(0.0, linestyle="--", linewidth=1.2, color="gray")
            ax.set_ylabel("Isotropic NLL/d - CLT NLL/d")
            ax.set_title(f"Per-dataset NLL gain | {method} Sigma")
            ax.tick_params(axis="x", rotation=90)
            fig.tight_layout()
            save_or_show(
                fig,
                run_outdir / f"ALL_DATASETS__{method}__NLL_GAIN_BOXPLOT",
                SHOW_AGGREGATE_FIGURES,
            )


    # ============================================================
    # RUN ONE DATASET
    # ============================================================

    def run_one_dataset(ds_dir, run_outdir, seed=0):
        ds_dir = Path(ds_dir)
        dataset = dataset_name_from_precomp_dir(ds_dir)
        rng = np.random.default_rng(seed)

        dataset_outdir = ensure_dir(Path(run_outdir) / dataset)

        print("\n" + "=" * 100)
        print(f"[dataset] {dataset}")
        print(f"[precomp] {os.path.basename(str(ds_dir))}")
        print("=" * 100)

        stats = load_precomputed_stats(ds_dir)
        genes = stats["genes"]
        perturbations = stats["perturbations"]
        n_cells_pert = stats["n_cells_pert"]

        print(f"[stats] {os.path.basename(str(stats['stats_path']))}")
        print(f"[stats] genes={len(genes):,} | perturbations >= {MIN_SAMPLES}: {len(perturbations):,}")
        if stats["n_control_from_stats"] is not None:
            print(f"[stats] n_control={stats['n_control_from_stats']:.0f}")

        data_path = find_h5ad_for_dataset(dataset)
        X0_pool, pool_info = load_control_pool_for_precomputed_genes(data_path, genes, rng)

        n_pool, n_genes = X0_pool.shape
        if n_genes != len(genes):
            raise ValueError(f"Control pool has {n_genes} genes but precomputed stats has {len(genes)} genes.")

        if CONTROL_NULL_NA_CAP is None:
            nA_default = n_pool
        else:
            if SAMPLE_WITH_REPLACEMENT:
                nA_default = int(CONTROL_NULL_NA_CAP)
            else:
                nA_default = min(int(CONTROL_NULL_NA_CAP), n_pool)
        nA_default = max(2, int(nA_default))

        dataset_all_rows = []
        dataset_summary_rows = []

        for method in RUN_SIGMA_METHODS:
            print("\n" + "-" * 100)
            print(f"[method] {method}")
            print("-" * 100)

            Sigma, sigma_path = load_precomputed_sigma(ds_dir, method)
            if Sigma.shape != (len(genes), len(genes)):
                raise ValueError(
                    f"Sigma shape {Sigma.shape} does not match precomputed gene count {(len(genes), len(genes))}."
                )

            print(f"[eig] decomposing precomputed {method} Sigma")
            lam, V = eig_precomputed_sigma(Sigma)
            del Sigma
            gc.collect()

            print(f"[project] projecting control pool into {method} Sigma eigenbasis")
            X0_eig = (
                X0_pool.astype(TRANSFORM_DTYPE, copy=False)
                @ V.astype(TRANSFORM_DTYPE, copy=False)
            ).astype(TRANSFORM_DTYPE, copy=False)

            del V
            gc.collect()

            rows = []
            d = int(lam.size)

            for p, n_p_raw in tqdm(
                list(zip(perturbations, n_cells_pert)),
                desc=f"{dataset}: {method} control-null NLL",
            ):
                n_p_raw = int(n_p_raw)
                nB = n_p_raw
                if PERT_LIKE_NB_CAP is not None:
                    nB = min(nB, int(PERT_LIKE_NB_CAP))
                nB = max(2, int(nB))

                if not SAMPLE_WITH_REPLACEMENT:
                    nB = min(nB, max(2, n_pool // 2))
                    nA = min(nA_default, max(2, n_pool - nB))
                else:
                    nA = nA_default

                nll_clt, nll_iso, q_clt, q_iso = nlls_for_null_reps_in_eigbasis(
                    X0_eig=X0_eig,
                    lam=lam,
                    nA=nA,
                    nB=nB,
                    n_reps=N_REPS_PER_PERT,
                    rng=rng,
                )

                gain = (nll_iso - nll_clt) / float(d)

                rows.append({
                    "dataset": dataset,
                    "sigma_method": method,
                    "perturbation": str(p),
                    "n_p_raw": n_p_raw,
                    "nB_used": int(nB),
                    "nA_used": int(nA),
                    "n_reps": int(N_REPS_PER_PERT),
                    "n_control_pool_loaded": int(n_pool),
                    "n_genes": int(d),
                    "sigma_path": sigma_path,

                    "clt_nll_per_dim_mean": float(np.mean(nll_clt) / d),
                    "iso_nll_per_dim_mean": float(np.mean(nll_iso) / d),
                    "clt_nll_per_dim_std": float(np.std(nll_clt / d, ddof=1)) if len(nll_clt) > 1 else 0.0,
                    "iso_nll_per_dim_std": float(np.std(nll_iso / d, ddof=1)) if len(nll_iso) > 1 else 0.0,

                    "nll_gain_iso_minus_clt_per_dim": float(np.mean(gain)),
                    "nll_gain_iso_minus_clt_per_dim_std": float(np.std(gain, ddof=1)) if len(gain) > 1 else 0.0,

                    "clt_Q_over_d_mean": float(np.mean(q_clt / d)),
                    "iso_Q_over_d_mean": float(np.mean(q_iso / d)),
                    "clt_better_fraction": float(np.mean(nll_clt < nll_iso)),
                })

            df = pd.DataFrame(rows)
            method_csv = dataset_outdir / f"{dataset}__{method}__per_perturbation_size_nll_null.csv"
            df.to_csv(method_csv, index=False)
            print(f"[saved] {method_csv}")

            if MAKE_PER_DATASET_PLOTS and len(df) > 0:
                plot_dataset_outputs(df, dataset_outdir, dataset, method)

            summary = {
                "dataset": dataset,
                "sigma_method": method,
                "precomp_dir": str(ds_dir),
                "stats_path": stats["stats_path"],
                "data_path": str(data_path),
                "sigma_path": sigma_path,
                "expression_cutoff": float(EXPRESSION_CUTOFF),
                "n_genes": int(d),
                "n_valid_perturbations": int(len(df)),
                "n_reps_per_pert": int(N_REPS_PER_PERT),
                "n_control_pool_loaded": int(n_pool),
                "n_control_total_in_h5ad": int(pool_info["n_control_total_in_h5ad"]),
                "n_control_from_stats": stats["n_control_from_stats"],
                "mean_gain_iso_minus_clt_per_dim": float(df["nll_gain_iso_minus_clt_per_dim"].mean()) if len(df) else np.nan,
                "median_gain_iso_minus_clt_per_dim": float(df["nll_gain_iso_minus_clt_per_dim"].median()) if len(df) else np.nan,
                "fraction_perturbations_clt_better": float(np.mean(df["nll_gain_iso_minus_clt_per_dim"].values > 0)) if len(df) else np.nan,
                "mean_clt_nll_per_dim": float(df["clt_nll_per_dim_mean"].mean()) if len(df) else np.nan,
                "mean_iso_nll_per_dim": float(df["iso_nll_per_dim_mean"].mean()) if len(df) else np.nan,
                "mean_clt_Q_over_d": float(df["clt_Q_over_d_mean"].mean()) if len(df) else np.nan,
                "mean_iso_Q_over_d": float(df["iso_Q_over_d_mean"].mean()) if len(df) else np.nan,
                **pool_info,
            }

            summary_json = dataset_outdir / f"{dataset}__{method}__summary.json"
            with open(summary_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[saved] {summary_json}")

            dataset_all_rows.append(df)
            dataset_summary_rows.append(summary)

            del X0_eig, lam, df
            gc.collect()

        del X0_pool
        gc.collect()

        if dataset_all_rows:
            all_df = pd.concat(dataset_all_rows, ignore_index=True)
        else:
            all_df = pd.DataFrame()

        return all_df, dataset_summary_rows

    run_outdir = ensure_dir(Path(OUTDIR) / "perpert_nll_precomputed")

    all_dfs = []
    all_summaries = []
    errors = []

    dataset_dirs = find_dataset_dirs()

    for i, ds_dir in enumerate(dataset_dirs):
        dataset = dataset_name_from_precomp_dir(ds_dir)
        try:
            df_i, summary_rows_i = run_one_dataset(
                ds_dir=ds_dir,
                run_outdir=run_outdir,
                seed=SEED + i,
            )

            if df_i is not None and len(df_i) > 0:
                all_dfs.append(df_i)
            all_summaries.extend(summary_rows_i)

        except Exception as e:
            error = {
                "dataset": dataset,
                "precomp_dir": str(ds_dir),
                "error_type": type(e).__name__,
                "error": repr(e),
            }
            errors.append(error)
            print("\n" + "!" * 100)
            print(f"[ERROR] Failed on {dataset}")
            print(repr(e))
            print("!" * 100 + "\n")

        gc.collect()


    # ============================================================
    # AGGREGATE SAVE + PLOTS
    # ============================================================

    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        all_summary_df = pd.DataFrame(all_summaries)

        all_csv = run_outdir / "ALL_DATASETS__per_perturbation_size_nll_null.csv"
        all_summary_csv = run_outdir / "ALL_DATASETS__per_perturbation_size_nll_summary.csv"

        all_df.to_csv(all_csv, index=False)
        all_summary_df.to_csv(all_summary_csv, index=False)

        print(f"[saved] {all_csv}")
        print(f"[saved] {all_summary_csv}")

        if MAKE_AGGREGATE_PLOTS:
            plot_aggregate_outputs(all_df, run_outdir)

    else:
        print("[warn] No successful dataset outputs to aggregate.")

    errors_path = run_outdir / "ALL_DATASETS__errors.json"
    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=2)
    print(f"[saved] {errors_path}")

    _PC_ALL_DF = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    _PC_RUN_DIR = str(run_outdir)


def plot_combined_boxplot_panels():
    SCATTER_DATASETS = [
        "TianKampmann2021_CRISPRi",
        "TianKampmann2019_day7neuron",
        "TianKampmann2019_iPSC",
    ]
    DISPLAY_NAMES = {
        "TianKampmann2021_CRISPRi": "Tian 2021 CRISPRi",
        "TianKampmann2019_day7neuron": "Tian 2019 day7 neuron",
        "TianKampmann2019_iPSC": "Tian 2019 iPSC",
    }
    SHOW_FIGURE = True
    BOXPLOT_FLIERS = False
    POINT_SIZE = 28
    POINT_ALPHA = 0.75
    EPS = 1e-12

    def save_svg(fig, path_base, show=False):
        path_base = Path(path_base)
        svg_path = path_base.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"[saved] {svg_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)


    def make_combined_boxplot_plus_three_panels(all_df, run_outdir, show=True):
        run_outdir = Path(run_outdir)

        required_cols = [
            "dataset",
            "sigma_method",
            "perturbation",
            "clt_nll_per_dim_mean",
            "iso_nll_per_dim_mean",
            "nll_gain_iso_minus_clt_per_dim",
        ]
        missing = [c for c in required_cols if c not in all_df.columns]
        if missing:
            raise ValueError(f"Missing required columns in all_df: {missing}")

        # relative difference metric requested by user
        df = all_df.copy()
        df["relative_abs_diff"] = (
            np.abs(df["iso_nll_per_dim_mean"] - df["clt_nll_per_dim_mean"])
            / np.maximum(np.abs(df["iso_nll_per_dim_mean"]), EPS)
        )

        for method, dfm in df.groupby("sigma_method"):
            # ----------------------------------------------------
            # top boxplot order from all datasets
            # ----------------------------------------------------
            dataset_order = (
                dfm.groupby("dataset")["nll_gain_iso_minus_clt_per_dim"]
                .median()
                .sort_values(ascending=False)
                .index
                .tolist()
            )

            # ----------------------------------------------------
            # figure layout:
            #   top row spans all 3 columns
            #   bottom row has 3 panels
            # ----------------------------------------------------
            fig = plt.figure(figsize=(18, 9))
            gs = GridSpec(
                2, 3,
                figure=fig,
                height_ratios=[2.2, 1.5],
                hspace=0.42,
                wspace=0.28,
            )

            ax_top = fig.add_subplot(gs[0, :])
            ax1 = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[1, 1])
            ax3 = fig.add_subplot(gs[1, 2])
            bottom_axes = [ax1, ax2, ax3]

            # ====================================================
            # TOP: LONG BOXPLOT FOR ALL DATASETS
            # ====================================================
            data = [
                dfm.loc[
                    dfm["dataset"] == ds,
                    "nll_gain_iso_minus_clt_per_dim",
                ].values
                for ds in dataset_order
            ]

            ax_top.boxplot(
                data,
                labels=dataset_order,
                showfliers=BOXPLOT_FLIERS,
            )
            ax_top.axhline(0.0, linestyle="--", linewidth=1.2, color="gray")
            ax_top.set_ylabel("Isotropic NLL/d - CLT NLL/d")
            ax_top.set_title(f"Per-dataset NLL gain | {method} Sigma")
            ax_top.tick_params(axis="x", rotation=90)

            # ====================================================
            # BOTTOM: 3 PER-DATASET SCATTER PANELS
            # y = |NLL_ISO - NLL_CLT| / |NLL_ISO|
            #
            # x = perturbation rank after sorting by this quantity
            #     (largest on left)
            # ====================================================
            for ax, dataset in zip(bottom_axes, SCATTER_DATASETS):
                sub = dfm[dfm["dataset"] == dataset].copy()

                if len(sub) == 0:
                    ax.text(
                        0.5, 0.5, "dataset not found",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=12,
                    )
                    ax.set_title(DISPLAY_NAMES.get(dataset, dataset))
                    ax.set_xlabel("Perturbations")
                    ax.set_ylabel(r"$|NLL_{iso}-NLL_{CLT}| / |NLL_{iso}|$")
                    continue

                sub = sub.sort_values("relative_abs_diff", ascending=False).reset_index(drop=True)
                sub["rank"] = np.arange(1, len(sub) + 1)

                ax.scatter(
                    sub["rank"].values,
                    sub["relative_abs_diff"].values,
                    s=POINT_SIZE,
                    alpha=POINT_ALPHA,
                    edgecolor="none",
                )

                ax.set_title(DISPLAY_NAMES.get(dataset, dataset))
                ax.set_xlabel("Perturbations (ranked)")
                ax.set_ylabel(r"$|NLL_{iso}-NLL_{CLT}| / |NLL_{iso}|$")

                y = sub["relative_abs_diff"].values
                mean_y = float(np.mean(y))
                med_y = float(np.median(y))
                frac_nonzero = float(np.mean(y > 0))
                npts = len(sub)

                ax.text(
                    0.03,
                    0.97,
                    (
                        f"mean = {mean_y:.3g}\n"
                        f"median = {med_y:.3g}\n"
                        f"nonzero frac = {frac_nonzero:.3f}\n"
                        f"points = {npts:,}"
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        alpha=0.85,
                        linewidth=0.5,
                    ),
                )

            fig.suptitle(
                f"NLL comparison summary | {method} Sigma",
                y=0.98,
                fontsize=16,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.965])

            save_svg(
                fig,
                run_outdir / f"ALL_DATASETS__{method}__BOXPLOT_plus_3_RELATIVE_DIFF_SCATTERS",
                show=show,
            )

    if _PC_ALL_DF is None or _PC_RUN_DIR is None:
        raise RuntimeError("Run run_perpert_nll_precomputed_cov() first to populate the aggregate dataframe.")

    make_combined_boxplot_plus_three_panels(
        all_df=_PC_ALL_DF,
        run_outdir=_PC_RUN_DIR,
        show=SHOW_FIGURE,
    )
