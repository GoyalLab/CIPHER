"""Analytic-Gaussian-posterior resistance-driver engine for Fig M7 / Fig S17.

Helper module living in ``notebooks/src`` -- NOT part of the installable
``cipher`` package. A notebook-only helper for reproducing the pancreatic
naive-vs-resistant supplementary figures.

The engine selects significant high-LFC genes from a naive-vs-resistant
contrast, fits an analytic Gaussian posterior on the mean-shift, and scores /
ranks genes by ``log(1 + |posterior mean mu|)``.

Notes on de-duplication
------------------------
Several helpers were re-defined with the SAME name.  Where bodies were
byte-identical or logically identical (``to_dense``, ``normalize_gene_list``,
``ordinal_rank_desc``, ``normalize_gene_name``, ``assign_labels_to_top_band``)
a single copy is kept here.  Two helpers -- ``drop_housekeeping_prefixes`` and
``filter_by_expression_and_variance_percentile`` -- genuinely diverge between
the main pipeline and the naive-only path: the main-pipeline versions live here,
while the naive-only path keeps its own divergent copies INLINE in the notebook
so both behaviours reproduce exactly.
"""
from __future__ import annotations


# --- library imports required by the helper functions (resolved at call time
#     so placement after the docstring is sufficient) ---
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

# The reference figures were generated in an environment WITHOUT adjustText, so every
# gene-label panel used the deterministic (bounded-offset) placement below. Keep that off
# by default so the reproduction matches the reference; set True to use adjustText instead.
USE_ADJUSTTEXT = False

import os
import re
import math

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
import scipy.stats
from scipy.stats import ttest_ind
from scipy.linalg import cho_factor, cho_solve
from statsmodels.stats.multitest import multipletests

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import scanpy as sc


# ============================================================
# PATCH OLD/BROKEN SCANPY
# ============================================================

try:
    sc.read_h5ad
except AttributeError:
    print("[patch] scanpy has no read_h5ad; using anndata.read_h5ad instead")
    sc.read_h5ad = ad.read_h5ad


# ============================================================
# UTILITIES
# ============================================================

def to_dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}\nPWD: {os.getcwd()}")
    print(f"[OK] Found file: {path}")


def tau2_to_label(tau2):
    tau2 = float(tau2)
    s = f"{tau2:.0e}" if tau2 < 1e-3 or tau2 >= 1e3 else str(tau2)
    s = s.replace("+", "").replace("-", "m").replace(".", "p")
    return f"tau2_{s}"


def r2_from_pred(y, yhat, eps=1e-12):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    return 1.0 - np.sum((y - yhat) ** 2) / (np.sum(y ** 2) + eps)


def stable_logdet_from_cholesky(cho):
    c, _ = cho
    return 2.0 * np.sum(np.log(np.diag(c) + 1e-300))


def drop_housekeeping_prefixes(
    var_names,
    bad_prefixes=("RPL", "RPS", "MT-", "MT.", "HSP", "HSP90", "EIF"),
):
    names = np.asarray(var_names, dtype=str)
    keep = np.ones(len(names), dtype=bool)
    for prefix in bad_prefixes:
        keep &= ~np.char.startswith(names, prefix)
    return keep


def normalize_gene_list(genes):
    out = []
    seen = set()
    for gene in genes:
        gene = str(gene).strip().upper()
        if gene and gene not in seen:
            out.append(gene)
            seen.add(gene)
    return out


def ordinal_rank_desc(values, valid=None):
    """Return 1-based descending ordinal ranks."""
    values = np.asarray(values, dtype=np.float64)

    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(values)

    ranks = np.full(len(values), np.nan, dtype=np.float64)
    idx = np.where(valid)[0]

    if len(idx) == 0:
        return ranks

    order = idx[np.argsort(-values[idx], kind="mergesort")]
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    return ranks


def get_absmu_rank_order(mu):
    """Rank genes by |posterior mean mu|, largest first."""
    mu = np.asarray(mu, dtype=np.float64)
    score = np.abs(mu)
    score = np.where(np.isfinite(score), score, -np.inf)
    return np.argsort(-score, kind="mergesort")


def remap_top_idx_to_ranked_positions(order, top_idx):
    top_idx = np.asarray(top_idx, dtype=int)
    if len(top_idx) == 0:
        return np.asarray([], dtype=int)

    top_mask = np.zeros(len(order), dtype=bool)
    top_mask[top_idx] = True
    return np.where(top_mask[order])[0]


# ============================================================
# FILTERING
# ============================================================

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
    return adata[:, keep].copy()


# ============================================================
# COVARIANCE AND SAMPLE-MEAN NOISE
# ============================================================

def compute_covariance(X, shrinkage=1e-3):
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)

    C = (Xc.T @ Xc) / max(1, X.shape[0] - 1)
    scale = np.mean(np.diag(C)) + 1e-12
    C += float(shrinkage) * scale * np.eye(C.shape[0])
    return C


def build_H_from_sample_means(
    X0,
    X1,
    shrinkage=1e-4,
    ridge=1e-6,
    approx="diag",
):
    X0 = np.asarray(X0, dtype=np.float64)
    X1 = np.asarray(X1, dtype=np.float64)

    n0 = X0.shape[0]
    n1 = X1.shape[0]
    print("n0,n1", n0, n1)

    S0 = compute_covariance(X0, shrinkage=shrinkage)
    S1 = compute_covariance(X1, shrinkage=shrinkage)

    if approx == "full":
        H = S0 / max(n0, 1) + S1 / max(n1, 1)
    elif approx == "diag":
        H = S0 / max(n0, 1) + np.diag(np.diag(S1)) / max(n1, 1)
    elif approx == "naive":
        H = (1.0 / max(n0, 1) + 1.0 / max(n1, 1)) * S0
    else:
        H = np.eye(X0.shape[1], dtype=np.float64)

    scale = np.mean(np.diag(H)) + 1e-12
    H += float(ridge) * scale * np.eye(H.shape[0])
    return H


# ============================================================
# DIFFERENTIAL EXPRESSION
# ============================================================

def compute_de_scores(X0, X1, gene_names, eps=1e-12):
    X0 = np.asarray(X0, dtype=np.float64)
    X1 = np.asarray(X1, dtype=np.float64)

    mean0 = X0.mean(axis=0)
    mean1 = X1.mean(axis=0)

    delta = mean1 - mean0
    log2fc = np.log2((mean1 + 1e-10) / (mean0 + 1e-10))

    n0 = X0.shape[0]
    n1 = X1.shape[0]

    v0 = X0.var(axis=0, ddof=1)
    v1 = X1.var(axis=0, ddof=1)
    se = np.sqrt(v0 / max(n0, 1) + v1 / max(n1, 1)) + eps
    t_stat = delta / se

    _, p_value = ttest_ind(X1, X0, axis=0, equal_var=False)
    p_value = np.asarray(p_value, dtype=np.float64)
    p_value = np.nan_to_num(p_value, nan=1.0, posinf=1.0, neginf=1.0)

    reject_fdr, p_adj, _, _ = multipletests(
        p_value,
        alpha=0.05,
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
        "p_value": p_value,
        "p_adj": p_adj,
        "significant_fdr_0p05": reject_fdr.astype(int),
        "neglog10_p": -np.log10(np.maximum(p_value, 1e-300)),
        "neglog10_padj": -np.log10(np.maximum(p_adj, 1e-300)),
    })

    de = de.sort_values(
        ["significant_fdr_0p05", "abs_log2fc", "p_adj"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    de["rank"] = np.arange(1, len(de) + 1)
    de["gene_upper"] = de["gene"].astype(str).str.upper()
    return de


# ============================================================
# ANALYTIC GAUSSIAN POSTERIOR
# ============================================================

def analytic_gaussian_posterior(Sigma, y, H, tau2=1.0):
    Sigma = np.asarray(Sigma, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    H = np.asarray(H, dtype=np.float64)

    tau2 = float(tau2)
    if tau2 <= 0:
        raise ValueError("tau2 must be positive.")

    choH = cho_factor(H, lower=True, check_finite=False)
    Hinv_y = cho_solve(choH, y, check_finite=False)
    Hinv_Sigma = cho_solve(choH, Sigma, check_finite=False)

    Prec = Sigma.T @ Hinv_Sigma + np.eye(Sigma.shape[1]) / tau2
    choP = cho_factor(Prec, lower=True, check_finite=False)

    Cov = cho_solve(choP, np.eye(Prec.shape[0]), check_finite=False)
    mu = cho_solve(choP, Sigma.T @ Hinv_y, check_finite=False)

    yhat = Sigma @ mu
    r2 = r2_from_pred(y, yhat)

    C = H + tau2 * (Sigma @ Sigma.T)
    choC = cho_factor(C, lower=True, check_finite=False)
    logdetC = stable_logdet_from_cholesky(choC)
    quad = y @ cho_solve(choC, y, check_finite=False)

    log_marginal = -0.5 * (
        len(y) * np.log(2.0 * np.pi)
        + logdetC
        + quad
    )

    return {
        "mu": mu,
        "Cov": Cov,
        "std": np.sqrt(np.maximum(np.diag(Cov), 0.0)),
        "yhat": yhat,
        "r2": r2,
        "log_marginal": log_marginal,
    }


# ============================================================
# LABEL HELPERS
# ============================================================

def _label_top_points(ax, xs, ys, labels, idxs, fontsize=8):
    for i in idxs:
        if np.isfinite(xs[i]) and np.isfinite(ys[i]):
            ax.text(xs[i], ys[i], labels[i], fontsize=fontsize)


def _highlight_genes_ranked(
    ax,
    gene_names_ranked,
    genes,
    yvals,
    xvals,
    fontsize=10,
):
    genes = set(normalize_gene_list(genes))

    for j, gene in enumerate(np.asarray(gene_names_ranked, dtype=str)):
        if (
            gene.upper() in genes
            and np.isfinite(xvals[j])
            and np.isfinite(yvals[j])
        ):
            ax.scatter([xvals[j]], [yvals[j]], s=120, marker="*", zorder=6)
            ax.text(xvals[j], yvals[j], f" {gene}", fontsize=fontsize)


def _draw_repelled_gene_labels(
    ax,
    x,
    y,
    gene_names,
    label_idx,
    fontsize=9,
):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    gene_names = np.asarray(gene_names, dtype=str)
    label_idx = sorted(set(int(i) for i in label_idx))

    if len(label_idx) == 0:
        return

    text_effect = [pe.withStroke(linewidth=3.5, foreground="white")]

    try:
        if not USE_ADJUSTTEXT:
            raise ImportError("adjustText disabled; reference used deterministic placement")
        from adjustText import adjust_text

        texts = []
        for k, i in enumerate(label_idx):
            texts.append(
                ax.text(
                    x[i] + 0.20 + 0.05 * (k % 3),
                    y[i] * (1.05 + 0.04 * (k % 4)),
                    gene_names[i],
                    fontsize=fontsize,
                    ha="left",
                    va="center",
                    zorder=20,
                    path_effects=text_effect,
                )
            )

        adjust_text(
            texts,
            ax=ax,
            x=x[label_idx],
            y=y[label_idx],
            expand_text=(1.35, 1.55),
            expand_points=(1.35, 1.55),
            force_text=(0.40, 0.95),
            force_points=(0.25, 0.70),
            force_pull=(0.02, 0.08),
            lim=600,
            arrowprops=dict(
                arrowstyle="-",
                lw=0.7,
                alpha=0.65,
                color="0.25",
            ),
        )
        return

    except Exception as exc:
        print(
            "[labels] adjustText unavailable or failed; "
            f"using fallback label placement. Reason: {exc}"
        )

    # Deterministic fallback with alternating offsets and leader lines.
    for k, i in enumerate(label_idx):
        dx = 24 + 10 * (k % 4)
        dy = (18 + 7 * (k % 5)) * (1 if k % 2 == 0 else -1)

        ax.annotate(
            gene_names[i],
            xy=(x[i], y[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            ha="left",
            va="center",
            zorder=20,
            arrowprops=dict(
                arrowstyle="-",
                lw=0.75,
                alpha=0.65,
                color="0.25",
                shrinkA=0,
                shrinkB=4,
            ),
            path_effects=text_effect,
        )


# ============================================================
# PLOT 1: |z| VS |mu| RANK
# ============================================================

def plot_absz_vs_absmu_rank(
    mu,
    std,
    gene_names,
    top_idx,
    genes_to_check,
    outpath_png,
    outpath_svg,
):
    mu = np.asarray(mu, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    gene_names = np.asarray(gene_names, dtype=str)

    z = mu / (std + 1e-12)
    abs_z = np.abs(z)
    order = get_absmu_rank_order(mu)

    mu_ranked = mu[order]
    z_ranked = z[order]
    abs_z_ranked = abs_z[order]
    gene_names_ranked = gene_names[order]

    x = np.arange(len(mu_ranked))
    top_ranked_idx = remap_top_idx_to_ranked_positions(order, top_idx)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(x, abs_z_ranked, lw=1)

    if len(top_ranked_idx) > 0:
        ax.scatter(x[top_ranked_idx], abs_z_ranked[top_ranked_idx], s=25)
        _label_top_points(
            ax,
            x,
            abs_z_ranked,
            gene_names_ranked,
            top_ranked_idx,
        )

    _highlight_genes_ranked(
        ax,
        gene_names_ranked,
        genes_to_check,
        abs_z_ranked,
        x,
    )

    ax.set_xlabel("Rank by |posterior mean μ|")
    ax.set_ylabel("|posterior z|")
    ax.set_title("Posterior |z|-score vs genes ranked by |μ|")

    plt.tight_layout()
    plt.savefig(outpath_png, dpi=300)
    plt.savefig(outpath_svg)
    plt.show()

    ranked_df = pd.DataFrame({
        "abs_mu_rank": np.arange(1, len(mu_ranked) + 1),
        "gene": gene_names_ranked,
        "mu": mu_ranked,
        "abs_mu": np.abs(mu_ranked),
        "z": z_ranked,
        "abs_z": abs_z_ranked,
    })

    table_path = outpath_png.replace(".png", "_table.tsv")
    ranked_df.to_csv(table_path, sep="\t", index=False)

    print(f"[saved] {outpath_png}")
    print(f"[saved] {outpath_svg}")
    print(f"[saved] {table_path}")


# ============================================================
# PLOT 2: log(1 + |mu|) VS |mu| RANK
# ============================================================

def plot_log1p_absmu_vs_absmu_rank(
    mu,
    gene_names,
    top_idx,
    genes_to_check,
    outpath_png,
    outpath_svg,
):
    mu = np.asarray(mu, dtype=np.float64)
    gene_names = np.asarray(gene_names, dtype=str)

    order = get_absmu_rank_order(mu)
    mu_ranked = mu[order]
    gene_names_ranked = gene_names[order]

    x = np.arange(len(mu_ranked))
    valid = np.isfinite(mu_ranked)

    log1p_abs_mu_ranked = np.full_like(mu_ranked, np.nan, dtype=np.float64)
    log1p_abs_mu_ranked[valid] = np.log1p(np.abs(mu_ranked[valid]))

    if np.sum(~valid) > 0:
        print(
            f"[plot] WARNING: {(~valid).sum()} genes have nonfinite mu "
            "and were skipped."
        )

    top_ranked_idx = remap_top_idx_to_ranked_positions(order, top_idx)
    top_ranked_idx = np.asarray(
        [i for i in top_ranked_idx if valid[i]],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(x[valid], log1p_abs_mu_ranked[valid], lw=1)

    if len(top_ranked_idx) > 0:
        ax.scatter(
            x[top_ranked_idx],
            log1p_abs_mu_ranked[top_ranked_idx],
            s=25,
        )
        _label_top_points(
            ax,
            x,
            log1p_abs_mu_ranked,
            gene_names_ranked,
            top_ranked_idx,
        )

    _highlight_genes_ranked(
        ax,
        gene_names_ranked,
        genes_to_check,
        log1p_abs_mu_ranked,
        x,
    )

    ax.axhline(0, lw=1, alpha=0.6)
    ax.set_xlabel("Rank by |posterior mean μ|")
    ax.set_ylabel("log(1 + |posterior mean μ|)")
    ax.set_title("Posterior score: log(1 + |μ|) vs genes ranked by |μ|")

    plt.tight_layout()
    plt.savefig(outpath_png, dpi=300)
    plt.savefig(outpath_svg)
    plt.show()

    ranked_df = pd.DataFrame({
        "abs_mu_rank": np.arange(1, len(mu_ranked) + 1),
        "gene": gene_names_ranked,
        "mu": mu_ranked,
        "abs_mu": np.abs(mu_ranked),
        "log1p_abs_mu": log1p_abs_mu_ranked,
        "valid_log1p_abs_mu": valid,
    })

    table_path = outpath_png.replace(".png", "_table.tsv")
    ranked_df.to_csv(table_path, sep="\t", index=False)

    print(f"[saved] {outpath_png}")
    print(f"[saved] {outpath_svg}")
    print(f"[saved] {table_path}")


# ============================================================
# PLOT 3: log(1 + |mu|) VS POSITIVE log2FC
# ============================================================

def plot_log1p_absmu_vs_positive_lfc_ylog(
    mu,
    log2fc,
    gene_names,
    top_idx,
    genes_to_check,
    outpath_png,
    outpath_svg,
    label_all_tracked=True,
    label_top_by_absmu=True,
    label_fontsize=9,
):
    mu = np.asarray(mu, dtype=np.float64)
    log2fc = np.asarray(log2fc, dtype=np.float64)
    gene_names = np.asarray(gene_names, dtype=str)

    log1p_abs_mu = np.full_like(mu, np.nan, dtype=np.float64)
    finite_mu = np.isfinite(mu)
    log1p_abs_mu[finite_mu] = np.log1p(np.abs(mu[finite_mu]))

    valid = (
        np.isfinite(log2fc)
        & np.isfinite(log1p_abs_mu)
        & (log2fc >= 0)
        & (log1p_abs_mu > 0)
    )

    n_valid = int(valid.sum())
    n_skipped = int((~valid).sum())

    print(f"[plot] positive-log2FC genes shown: {n_valid}")
    print(
        f"[plot] skipped {n_skipped} genes because log2FC < 0, "
        "log(1 + |mu|) <= 0, or values were nonfinite."
    )

    if n_valid == 0:
        raise ValueError(
            "No valid genes for positive-log2FC plot. "
            "Need log2FC >= 0 and log(1 + |mu|) > 0."
        )

    label_idx = set()
    tracked_upper = set(normalize_gene_list(genes_to_check))

    if label_all_tracked:
        for i, gene in enumerate(gene_names):
            if gene.upper() in tracked_upper and valid[i]:
                label_idx.add(i)

    if label_top_by_absmu and top_idx is not None:
        top_idx = np.asarray(top_idx, dtype=int)
        for i in top_idx:
            if 0 <= i < len(gene_names) and valid[i]:
                label_idx.add(int(i))

    label_idx = sorted(label_idx)

    print(f"[plot] labeled genes: {len(label_idx)}")
    if len(label_idx) > 0:
        print("[plot] labels:", ", ".join(gene_names[label_idx]))

    fig, ax = plt.subplots(figsize=(10.5, 7.8))

    ax.scatter(
        log2fc[valid],
        log1p_abs_mu[valid],
        s=20,
        alpha=0.72,
        rasterized=True,
        zorder=2,
    )

    if len(label_idx) > 0:
        ax.scatter(
            log2fc[label_idx],
            log1p_abs_mu[label_idx],
            s=115,
            marker="*",
            edgecolor="black",
            linewidth=0.45,
            zorder=8,
        )

    xmax = float(np.nanmax(log2fc[valid]))
    ymin = float(np.nanmin(log1p_abs_mu[valid]))
    ymax = float(np.nanmax(log1p_abs_mu[valid]))

    ax.set_xlim(0, xmax * 1.25 + 0.5)
    ax.set_yscale("log")
    ax.set_ylim(max(ymin / 1.7, 1e-8), ymax * 1.85)

    ax.set_xlabel("log2FC")
    ax.set_ylabel("log(1 + |posterior mean μ|)")
    ax.set_title("Posterior log(1 + |μ|) vs positive log2FC")

    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.08)

    _draw_repelled_gene_labels(
        ax=ax,
        x=log2fc,
        y=log1p_abs_mu,
        gene_names=gene_names,
        label_idx=label_idx,
        fontsize=label_fontsize,
    )

    plt.tight_layout()
    plt.savefig(outpath_png, dpi=300)
    plt.savefig(outpath_svg)
    plt.show()

    scatter_df = pd.DataFrame({
        "gene": gene_names,
        "mu": mu,
        "abs_mu": np.abs(mu),
        "log1p_abs_mu": log1p_abs_mu,
        "log2fc": log2fc,
        "abs_log2fc": np.abs(log2fc),
        "valid_positive_lfc_ylog": valid,
        "labeled": [i in set(label_idx) for i in range(len(gene_names))],
    })

    table_path = outpath_png.replace(".png", "_table.tsv")
    scatter_df.to_csv(table_path, sep="\t", index=False)

    print(f"[saved] {outpath_png}")
    print(f"[saved] {outpath_svg}")
    print(f"[saved] {table_path}")


# ============================================================
# PLOT 4: CIPHER RANK VS LFC RANK
# ============================================================

def plot_cipher_rank_vs_lfc_rank_loglog(
    mu,
    log2fc,
    gene_names,
    genes_to_check,
    outpath_png,
    outpath_svg,
    top_idx=None,
    label_all_selected_genes=False,
    max_all_labels=100,
):
    mu = np.asarray(mu, dtype=np.float64)
    log2fc = np.asarray(log2fc, dtype=np.float64)
    gene_names = np.asarray(gene_names, dtype=str)

    abs_mu = np.abs(mu)
    log1p_abs_mu = np.full_like(mu, np.nan, dtype=np.float64)
    finite_mu = np.isfinite(mu)
    log1p_abs_mu[finite_mu] = np.log1p(abs_mu[finite_mu])

    abs_log2fc = np.abs(log2fc)
    valid = np.isfinite(abs_mu) & np.isfinite(abs_log2fc)

    cipher_rank = ordinal_rank_desc(abs_mu, valid=valid)
    lfc_rank = ordinal_rank_desc(abs_log2fc, valid=valid)

    n_valid = int(valid.sum())
    if n_valid == 0:
        print("[plot] WARNING: no valid genes for rank scatter.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        lfc_rank[valid],
        cipher_rank[valid],
        s=18,
        alpha=0.75,
    )

    diag = np.array([1, n_valid], dtype=float)
    ax.plot(diag, diag, "--", lw=1, alpha=0.5)

    genes_to_label = set(normalize_gene_list(genes_to_check))
    label_idx = set()

    for i, gene in enumerate(gene_names):
        if gene.upper() in genes_to_label and valid[i]:
            label_idx.add(i)

    if top_idx is not None:
        for i in np.asarray(top_idx, dtype=int):
            if 0 <= i < len(gene_names) and valid[i]:
                label_idx.add(int(i))

    if label_all_selected_genes:
        all_valid_idx = np.where(valid)[0]
        if len(all_valid_idx) <= max_all_labels:
            label_idx.update(int(i) for i in all_valid_idx)
        else:
            print(
                f"[plot] label_all_selected_genes=True but n_valid={len(all_valid_idx)} "
                f"> max_all_labels={max_all_labels}. "
                "Only genes_to_check/top_idx are labeled."
            )

    label_idx = sorted(label_idx)

    if len(label_idx) > 0:
        ax.scatter(
            lfc_rank[label_idx],
            cipher_rank[label_idx],
            s=90,
            marker="*",
            zorder=6,
        )
        _label_top_points(
            ax,
            lfc_rank,
            cipher_rank,
            gene_names,
            label_idx,
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, n_valid * 1.15)
    ax.set_ylim(0.8, n_valid * 1.15)

    ax.set_xlabel("LFC rank, by |log2FC|")
    ax.set_ylabel("CIPHER rank, by |posterior mean μ|")
    ax.set_title("CIPHER rank vs LFC rank, log-log")

    ax.text(
        0.02,
        0.98,
        "rank 1 = highest\nlower-left = high by both",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(outpath_png, dpi=300)
    plt.savefig(outpath_svg)
    plt.show()

    rank_df = pd.DataFrame({
        "gene": gene_names,
        "mu": mu,
        "abs_mu": abs_mu,
        "log1p_abs_mu": log1p_abs_mu,
        "log2fc": log2fc,
        "abs_log2fc": abs_log2fc,
        "cipher_rank_by_abs_mu": cipher_rank,
        "lfc_rank_by_abs_log2fc": lfc_rank,
        "valid": valid,
    })

    table_path = outpath_png.replace(".png", "_table.tsv")
    rank_df.to_csv(table_path, sep="\t", index=False)

    print(f"[saved] {outpath_png}")
    print(f"[saved] {outpath_svg}")
    print(f"[saved] {table_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_top_sig_highlfc_gaussian_pipeline(
    h5ad_path,
    outdir="analytic_gaussian_top_sig_highlfc",
    condition_key="Condition",
    cond0="Naive",
    cond1="Resistant",
    genes_to_check=None,
    top_n_de=1000,
    min_abs_log2fc=0.5,
    drop_housekeeping=True,
    Sigma_shrinkage=1e-3,
    H_shrinkage=1e-3,
    H_ridge=1e-6,
    H_mode="diag",
    tau2=1.0,
    min_cells_frac=0.0001,
    min_expr=0.0,
    hi_quantile=1.0,
    var_drop_q=1.0,
    filter_subsample_cells=0,
    seed=0,
    top_k_plot=20,
    label_all_selected_genes=False,
):
    if genes_to_check is None:
        genes_to_check = [
            "TGFA", "ATRNL1", "ATXN1", "TNNT2",
            "ANKRD1", "ROBO2", "PSG9", "PSG4",
        ]

    genes_to_check = normalize_gene_list(genes_to_check)

    safe_makedirs(outdir)
    check_file(h5ad_path)

    print(f"[run] tau2 = {tau2}")
    print(f"[run] outdir = {outdir}")

    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()

    if condition_key not in adata.obs:
        raise KeyError(
            f"'{condition_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    adata = adata[adata.obs[condition_key].isin([cond0, cond1])].copy()

    m0 = adata.obs[condition_key].values == cond0
    m1 = adata.obs[condition_key].values == cond1

    if m0.sum() < 5 or m1.sum() < 5:
        raise ValueError(f"Too few cells: {cond0}={m0.sum()}, {cond1}={m1.sum()}")

    adata = filter_by_expression_and_variance_percentile(
        adata,
        naive_mask=m0,
        min_cells_frac=min_cells_frac,
        min_expr=min_expr,
        hi_quantile=hi_quantile,
        var_drop_q=var_drop_q,
        filter_subsample_cells=filter_subsample_cells,
        seed=seed,
    )

    if drop_housekeeping:
        keep = drop_housekeeping_prefixes(adata.var_names)
        adata = adata[:, keep].copy()

    m0 = adata.obs[condition_key].values == cond0
    m1 = adata.obs[condition_key].values == cond1

    # Uses adata.X exactly as supplied.
    X0 = to_dense(adata[m0].X).astype(np.float64)
    X1 = to_dense(adata[m1].X).astype(np.float64)

    gene_names_all = np.asarray(adata.var_names, dtype=str)
    gene_names_all_upper = np.char.upper(gene_names_all)

    # --------------------------------------------------------
    # DE SELECTION
    # --------------------------------------------------------

    de_df = compute_de_scores(X0, X1, gene_names_all)
    de_df.to_csv(
        os.path.join(outdir, "all_genes_de_ranking.tsv"),
        sep="\t",
        index=False,
    )

    sig_de = de_df.loc[
        (de_df["p_adj"] < 0.05)
        & (de_df["abs_log2fc"] >= float(min_abs_log2fc))
    ].copy()

    sig_de = sig_de.sort_values(
        ["abs_log2fc", "p_adj"],
        ascending=[False, True],
    ).reset_index(drop=True)

    sig_de["selected_rank"] = np.arange(1, len(sig_de) + 1)
    selected_de = sig_de.head(min(top_n_de, len(sig_de))).copy()

    if len(selected_de) == 0:
        raise ValueError(
            f"No genes passed selection: p_adj < 0.05 and "
            f"abs(log2FC) >= {min_abs_log2fc}."
        )

    selected_genes = selected_de["gene"].values
    selected_set_upper = set(g.upper() for g in selected_genes)

    # --------------------------------------------------------
    # TRACKED-GENE STATUS
    # --------------------------------------------------------

    rows = []
    for gene in genes_to_check:
        present_after_filter = gene in set(gene_names_all_upper)
        in_selected = gene in selected_set_upper

        full_hit = de_df.loc[de_df["gene_upper"] == gene]
        sel_hit = sig_de.loc[sig_de["gene_upper"] == gene]

        rows.append({
            "gene": gene,
            "present_after_initial_filtering": present_after_filter,
            "in_selected_set": in_selected,
            "full_rank": int(full_hit["rank"].iloc[0]) if len(full_hit) else np.nan,
            "selected_rank": int(sel_hit["selected_rank"].iloc[0]) if len(sel_hit) else np.nan,
            "delta": float(full_hit["delta"].iloc[0]) if len(full_hit) else np.nan,
            "log2fc": float(full_hit["log2fc"].iloc[0]) if len(full_hit) else np.nan,
            "abs_log2fc": float(full_hit["abs_log2fc"].iloc[0]) if len(full_hit) else np.nan,
            "t_stat": float(full_hit["t_stat"].iloc[0]) if len(full_hit) else np.nan,
            "p_value": float(full_hit["p_value"].iloc[0]) if len(full_hit) else np.nan,
            "p_adj": float(full_hit["p_adj"].iloc[0]) if len(full_hit) else np.nan,
        })

    gene_status_df = pd.DataFrame(rows)
    gene_status_df.to_csv(
        os.path.join(outdir, "tracked_gene_status.tsv"),
        sep="\t",
        index=False,
    )

    print(f"[DE] total genes tested: {len(de_df)}")
    print(f"[DE] FDR<0.05 and |log2FC|>={min_abs_log2fc}: {len(sig_de)}")
    print(f"[DE] selected top {len(selected_de)} genes")
    print("\n[tracked genes]")
    print(gene_status_df.to_string(index=False))

    # --------------------------------------------------------
    # RESTRICT TO SELECTED GENES
    # --------------------------------------------------------

    selected_mask = np.isin(gene_names_all, selected_genes)
    adata_sel = adata[:, selected_mask].copy()
    gene_names = np.asarray(adata_sel.var_names, dtype=str)

    m0 = adata_sel.obs[condition_key].values == cond0
    m1 = adata_sel.obs[condition_key].values == cond1

    X0_sel = to_dense(adata_sel[m0].X).astype(np.float64)
    X1_sel = to_dense(adata_sel[m1].X).astype(np.float64)

    name_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    order_de = [name_to_idx[gene] for gene in selected_genes if gene in name_to_idx]

    X0_sel = X0_sel[:, order_de]
    X1_sel = X1_sel[:, order_de]
    gene_names = gene_names[order_de]

    y = X1_sel.mean(axis=0) - X0_sel.mean(axis=0)

    de_exact = de_df.set_index("gene", drop=False)
    log2fc = np.asarray(
        [float(de_exact.loc[gene, "log2fc"]) for gene in gene_names],
        dtype=np.float64,
    )
    abs_log2fc = np.abs(log2fc)

    # --------------------------------------------------------
    # POSTERIOR
    # --------------------------------------------------------

    Sigma = compute_covariance(X0_sel, shrinkage=Sigma_shrinkage)

    H = build_H_from_sample_means(
        X0_sel,
        X1_sel,
        shrinkage=H_shrinkage,
        ridge=H_ridge,
        approx=H_mode,
    )

    post = analytic_gaussian_posterior(
        Sigma=Sigma,
        y=y,
        H=H,
        tau2=tau2,
    )

    mu = post["mu"]
    Cov = post["Cov"]
    std = post["std"]
    yhat = post["yhat"]
    z = mu / (std + 1e-12)

    abs_mu = np.abs(mu)
    log1p_abs_mu = np.full_like(mu, np.nan, dtype=np.float64)
    finite_mu = np.isfinite(mu)
    log1p_abs_mu[finite_mu] = np.log1p(abs_mu[finite_mu])

    valid_rank = np.isfinite(abs_mu) & np.isfinite(abs_log2fc)
    cipher_rank_by_abs_mu = ordinal_rank_desc(abs_mu, valid=valid_rank)
    lfc_rank_by_abs_log2fc = ordinal_rank_desc(abs_log2fc, valid=valid_rank)

    top_by_absz = np.argsort(-np.abs(z))[:min(top_k_plot, len(gene_names))]
    top_by_absmu = np.argsort(-abs_mu)[:min(top_k_plot, len(gene_names))]

    # --------------------------------------------------------
    # SAVE NUMERIC OUTPUTS
    # --------------------------------------------------------

    np.save(os.path.join(outdir, "Sigma_selected.npy"), Sigma)
    np.save(os.path.join(outdir, "delta_x_selected.npy"), y)
    np.save(os.path.join(outdir, "H_selected.npy"), H)
    np.save(os.path.join(outdir, "posterior_mu_selected.npy"), mu)
    np.save(os.path.join(outdir, "posterior_cov_selected.npy"), Cov)
    np.save(os.path.join(outdir, "posterior_std_selected.npy"), std)
    np.save(os.path.join(outdir, "posterior_z_selected.npy"), z)
    np.save(os.path.join(outdir, "posterior_abs_mu_selected.npy"), abs_mu)
    np.save(os.path.join(outdir, "posterior_log1p_abs_mu_selected.npy"), log1p_abs_mu)
    np.save(os.path.join(outdir, "selected_log2fc.npy"), log2fc)
    np.save(os.path.join(outdir, "selected_abs_log2fc.npy"), abs_log2fc)
    np.save(os.path.join(outdir, "cipher_rank_by_abs_mu.npy"), cipher_rank_by_abs_mu)
    np.save(os.path.join(outdir, "lfc_rank_by_abs_log2fc.npy"), lfc_rank_by_abs_log2fc)

    pd.DataFrame({"gene": gene_names}).to_csv(
        os.path.join(outdir, "selected_gene_names.tsv"),
        sep="\t",
        index=False,
    )

    selected_de.to_csv(
        os.path.join(outdir, "selected_de_table.tsv"),
        sep="\t",
        index=False,
    )

    summary = pd.DataFrame({
        "gene": gene_names,
        "gene_upper": np.char.upper(gene_names),
        "mu": mu,
        "abs_mu": abs_mu,
        "log1p_abs_mu": log1p_abs_mu,
        "std": std,
        "z": z,
        "abs_z": np.abs(z),
        "delta_x": y,
        "log2fc": log2fc,
        "abs_log2fc": abs_log2fc,
        "cipher_rank_by_abs_mu": cipher_rank_by_abs_mu,
        "lfc_rank_by_abs_log2fc": lfc_rank_by_abs_log2fc,
        "selected_index": np.arange(len(gene_names)),
        "selected_rank_by_abs_lfc_then_padj": np.arange(1, len(gene_names) + 1),
    })

    summary.to_csv(
        os.path.join(outdir, "posterior_summary_selected.tsv"),
        sep="\t",
        index=False,
    )

    absmu_order = get_absmu_rank_order(mu)
    summary_absmu_ranked = summary.iloc[absmu_order].copy()
    summary_absmu_ranked["abs_mu_rank"] = np.arange(
        1,
        len(summary_absmu_ranked) + 1,
    )
    summary_absmu_ranked.to_csv(
        os.path.join(outdir, "posterior_summary_ranked_by_abs_mu.tsv"),
        sep="\t",
        index=False,
    )

    lfc_order = np.argsort(-abs_log2fc, kind="mergesort")
    summary_lfc_ranked = summary.iloc[lfc_order].copy()
    summary_lfc_ranked["lfc_rank"] = np.arange(
        1,
        len(summary_lfc_ranked) + 1,
    )
    summary_lfc_ranked.to_csv(
        os.path.join(outdir, "posterior_summary_ranked_by_lfc.tsv"),
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    plot_absz_vs_absmu_rank(
        mu=mu,
        std=std,
        gene_names=gene_names,
        top_idx=top_by_absz,
        genes_to_check=genes_to_check,
        outpath_png=os.path.join(outdir, "posterior_absz_vs_abs_mu_rank.png"),
        outpath_svg=os.path.join(outdir, "posterior_absz_vs_abs_mu_rank.svg"),
    )

    plot_log1p_absmu_vs_absmu_rank(
        mu=mu,
        gene_names=gene_names,
        top_idx=top_by_absmu,
        genes_to_check=genes_to_check,
        outpath_png=os.path.join(
            outdir,
            "posterior_log1p_abs_mu_vs_abs_mu_rank.png",
        ),
        outpath_svg=os.path.join(
            outdir,
            "posterior_log1p_abs_mu_vs_abs_mu_rank.svg",
        ),
    )

    plot_log1p_absmu_vs_positive_lfc_ylog(
        mu=mu,
        log2fc=log2fc,
        gene_names=gene_names,
        top_idx=top_by_absmu,
        genes_to_check=genes_to_check,
        outpath_png=os.path.join(
            outdir,
            "posterior_log1p_abs_mu_vs_positive_log2fc_ylog_labeled.png",
        ),
        outpath_svg=os.path.join(
            outdir,
            "posterior_log1p_abs_mu_vs_positive_log2fc_ylog_labeled.svg",
        ),
        label_all_tracked=True,
        label_top_by_absmu=True,
        label_fontsize=9,
    )

    plot_cipher_rank_vs_lfc_rank_loglog(
        mu=mu,
        log2fc=log2fc,
        gene_names=gene_names,
        genes_to_check=genes_to_check,
        top_idx=top_by_absmu,
        label_all_selected_genes=label_all_selected_genes,
        outpath_png=os.path.join(outdir, "cipher_rank_vs_lfc_rank_loglog.png"),
        outpath_svg=os.path.join(outdir, "cipher_rank_vs_lfc_rank_loglog.svg"),
    )

    # --------------------------------------------------------
    # RUN SUMMARY
    # --------------------------------------------------------

    with open(os.path.join(outdir, "run_summary.txt"), "w") as handle:
        handle.write(f"h5ad_path: {h5ad_path}\n")
        handle.write(f"contrast: {cond1} - {cond0}\n")
        handle.write(f"tracked_genes: {', '.join(genes_to_check)}\n")
        handle.write(
            f"selection_rule: top {top_n_de} genes with BH-FDR p_adj < 0.05 "
            f"and abs(log2FC) >= {min_abs_log2fc}, ranked by abs(log2FC)\n"
        )
        handle.write("cipher_score: log(1 + |posterior mean mu|)\n")
        handle.write("cipher_rank: descending |posterior mean mu|\n")
        handle.write("rank_plot_1: posterior abs z vs absolute-mu rank\n")
        handle.write("rank_plot_2: log1p(abs(mu)) vs absolute-mu rank\n")
        handle.write(
            "scatter_plot: log1p(abs(mu)) vs positive signed log2FC; "
            "x linear, y log, tracked labels shown\n"
        )
        handle.write("rank_scatter_plot: CIPHER rank vs LFC rank on log-log axes\n")
        handle.write(f"n_genes_tested: {len(de_df)}\n")
        handle.write(f"n_fdr_sig_and_high_lfc: {len(sig_de)}\n")
        handle.write(f"n_selected_genes: {len(selected_de)}\n")
        handle.write(f"tau2: {tau2}\n")
        handle.write(f"H_mode: {H_mode}\n")
        handle.write(f"R2: {post['r2']:.8g}\n")
        handle.write(f"log_marginal: {post['log_marginal']:.8g}\n")

    print("\n" + "=" * 70)
    print(f"[done] tau2 = {tau2}")
    print(f"[done] total genes tested = {len(de_df)}")
    print(f"[done] FDR<0.05 and |log2FC|>={min_abs_log2fc}: {len(sig_de)}")
    print(f"[done] selected top {len(selected_de)} genes")
    print(f"[done] posterior R2 = {post['r2']:.4f}")
    print("[done] CIPHER score = log(1 + |mu|)")
    print("[done] CIPHER rank = rank by |mu|")
    print(f"[done] outputs written to: {outdir}")

    return {
        "adata_selected": adata_sel,
        "de_df": de_df,
        "sig_de": sig_de,
        "selected_de": selected_de,
        "selected_genes": selected_genes,
        "gene_names": gene_names,
        "gene_status_df": gene_status_df,
        "Sigma": Sigma,
        "delta_x": y,
        "H": H,
        "posterior": post,
        "summary": summary,
        "summary_absmu_ranked": summary_absmu_ranked,
        "summary_lfc_ranked": summary_lfc_ranked,
        "log2fc": log2fc,
        "abs_log2fc": abs_log2fc,
        "abs_mu": abs_mu,
        "log1p_abs_mu": log1p_abs_mu,
        "cipher_rank_by_abs_mu": cipher_rank_by_abs_mu,
        "lfc_rank_by_abs_log2fc": lfc_rank_by_abs_log2fc,
    }


# ============================================================
# LABEL-BAND HELPERS (posterior redraw cells)
# ============================================================

def normalize_gene_name(gene):
    """Normalize gene names for case-insensitive matching."""
    return str(gene).strip().upper()


def assign_labels_to_top_band(
    x,
    y,
    gene_names,
    label_indices,
    n_rows,
    curve_ymax,
    band_gap,
    row_spacing,
    x_min,
    x_max,
):
    """
    Assign labels to a fixed grid above the posterior curve.

    Labels are sorted by CIPHER rank and placed from left to right.
    Several horizontal rows are used to prevent label overlap.

    Leader lines later connect each label to its actual data point.
    """
    label_indices = np.asarray(
        label_indices,
        dtype=int,
    )

    if len(label_indices) == 0:
        return pd.DataFrame(
            columns=[
                "index",
                "gene",
                "point_x",
                "point_y",
                "label_x",
                "label_y",
                "label_row",
                "label_column",
            ]
        )

    # Sort by position on the ranked curve.
    sorted_indices = label_indices[
        np.argsort(
            x[label_indices],
            kind="mergesort",
        )
    ]

    n_labels = len(sorted_indices)

    n_rows = max(
        1,
        min(int(n_rows), n_labels),
    )

    n_columns = int(
        math.ceil(n_labels / n_rows)
    )

    # Keep the first and last label columns away from the axes.
    left = (
        x_min
        + 0.045 * (x_max - x_min)
    )

    right = (
        x_max
        - 0.045 * (x_max - x_min)
    )

    if n_columns == 1:
        column_positions = np.array(
            [(left + right) / 2.0]
        )
    else:
        column_positions = np.linspace(
            left,
            right,
            n_columns,
        )

    first_row_y = curve_ymax + band_gap

    rows = []

    for order_position, idx in enumerate(
        sorted_indices
    ):
        column = order_position // n_rows
        row = order_position % n_rows

        label_x = column_positions[column]
        label_y = (
            first_row_y
            + row * row_spacing
        )

        rows.append({
            "index": int(idx),
            "gene": str(gene_names[idx]),
            "point_x": float(x[idx]),
            "point_y": float(y[idx]),
            "label_x": float(label_x),
            "label_y": float(label_y),
            "label_row": int(row + 1),
            "label_column": int(column + 1),
        })

    return pd.DataFrame(rows)


# ============================================================
# NAIVE-ONLY RANKED-EXPRESSION HELPERS (cell 8)
# ============================================================

def draw_repelled_labels(
    ax,
    x,
    y,
    gene_names,
    label_indices,
    fontsize=9,
):
    """
    Use adjustText when available, otherwise use deterministic
    annotation offsets.
    """

    label_indices = sorted(
        set(int(i) for i in label_indices)
    )

    if len(label_indices) == 0:
        return

    text_effect = [
        pe.withStroke(
            linewidth=3.5,
            foreground="white",
        )
    ]

    try:
        if not USE_ADJUSTTEXT:
            raise ImportError("adjustText disabled; reference used deterministic placement")
        from adjustText import adjust_text

        texts = []

        for i in label_indices:
            texts.append(
                ax.text(
                    x[i],
                    y[i],
                    gene_names[i],
                    fontsize=fontsize,
                    ha="left",
                    va="bottom",
                    zorder=20,
                    path_effects=text_effect,
                )
            )

        adjust_text(
            texts,
            ax=ax,
            x=x[label_indices],
            y=y[label_indices],
            expand_text=(1.25, 1.40),
            expand_points=(1.20, 1.35),
            force_text=(0.40, 0.80),
            force_points=(0.20, 0.50),
            force_pull=(0.02, 0.08),
            lim=600,
            arrowprops=dict(
                arrowstyle="-",
                lw=0.7,
                alpha=0.65,
                color="0.25",
            ),
        )

    except Exception as exc:
        print(
            "[labels] adjustText unavailable or failed; "
            f"using fallback placement. Reason: {exc}"
        )

        for k, i in enumerate(label_indices):
            dx = 18 + 8 * (k % 5)
            dy = (18 + 6 * (k % 4)) * (
                1 if k % 2 == 0 else -1
            )

            ax.annotate(
                gene_names[i],
                xy=(x[i], y[i]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                ha="left",
                va="center",
                zorder=20,
                arrowprops=dict(
                    arrowstyle="-",
                    lw=0.7,
                    alpha=0.65,
                    color="0.25",
                    shrinkA=0,
                    shrinkB=4,
                ),
                path_effects=text_effect,
            )


# ============================================================
# NAIVE-ONLY ALL-GENES HELPERS (cell 9)
# ============================================================

def gene_means(X):
    """
    Compute gene-wise means without densifying sparse matrices.
    """
    if issparse(X):
        return np.asarray(
            X.mean(axis=0)
        ).ravel().astype(np.float64)

    return np.asarray(
        X,
        dtype=np.float64,
    ).mean(axis=0)


def draw_clear_gene_labels(
    ax,
    x,
    y,
    labels,
    indices,
    fontsize=9,
    ARROW_LINEWIDTH=1.0,
):
    """
    Draw labels for tracked genes only.

    Uses adjustText when available. Otherwise, uses deterministic
    label offsets with clear black arrows.
    """
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    labels = np.asarray(
        labels,
        dtype=str,
    )

    indices = sorted({
        int(i)
        for i in indices
        if 0 <= int(i) < len(x)
        and np.isfinite(x[int(i)])
        and np.isfinite(y[int(i)])
        and y[int(i)] > 0
    })

    if not indices:
        return

    text_outline = [
        pe.withStroke(
            linewidth=4.0,
            foreground="white",
        )
    ]

    try:
        if not USE_ADJUSTTEXT:
            raise ImportError("adjustText disabled; reference used deterministic placement")
        from adjustText import adjust_text

        texts = []

        for k, i in enumerate(indices):

            # Start genes near the right edge with labels on the left.
            if x[i] > 0.72 * np.nanmax(x):
                initial_dx = -35
                horizontal_alignment = "right"
            else:
                initial_dx = 35
                horizontal_alignment = "left"

            initial_dy = (
                20 + 7 * (k % 5)
            ) * (
                1 if k % 2 == 0 else -1
            )

            text = ax.annotate(
                labels[i],
                xy=(
                    x[i],
                    y[i],
                ),
                xytext=(
                    initial_dx,
                    initial_dy,
                ),
                textcoords="offset points",
                fontsize=fontsize,
                fontweight="bold",
                ha=horizontal_alignment,
                va="center",
                color="black",
                zorder=30,
                clip_on=False,
                path_effects=text_outline,
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    linewidth=ARROW_LINEWIDTH,
                    alpha=0.9,
                    shrinkA=3,
                    shrinkB=6,
                    mutation_scale=10,
                    connectionstyle="arc3,rad=0.04",
                ),
            )

            texts.append(text)

        adjust_text(
            texts,
            ax=ax,
            x=x[indices],
            y=y[indices],
            expand=(1.35, 1.60),
            force_text=(0.8, 1.3),
            force_static=(0.5, 0.9),
            force_pull=(0.02, 0.08),
            ensure_inside_axes=False,
            iter_lim=1000,
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                linewidth=ARROW_LINEWIDTH,
                alpha=0.9,
                shrinkA=3,
                shrinkB=6,
                mutation_scale=10,
            ),
        )

    except Exception as exc:
        print(
            "[labels] adjustText unavailable or failed; "
            f"using deterministic placement: {exc}"
        )

        for k, i in enumerate(indices):

            direction = (
                1 if k % 2 == 0 else -1
            )

            dx = (
                35 + 12 * (k % 4)
            )

            dy = direction * (
                18 + 11 * (k % 6)
            )

            if x[i] > 0.72 * np.nanmax(x):
                dx = -45 - 12 * (k % 4)
                ha = "right"
            else:
                ha = "left"

            ax.annotate(
                labels[i],
                xy=(
                    x[i],
                    y[i],
                ),
                xytext=(
                    dx,
                    dy,
                ),
                textcoords="offset points",
                fontsize=fontsize,
                fontweight="bold",
                ha=ha,
                va="center",
                color="black",
                zorder=30,
                clip_on=False,
                path_effects=text_outline,
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    linewidth=ARROW_LINEWIDTH,
                    alpha=0.9,
                    shrinkA=3,
                    shrinkB=6,
                    mutation_scale=10,
                    connectionstyle="arc3,rad=0.05",
                ),
            )
