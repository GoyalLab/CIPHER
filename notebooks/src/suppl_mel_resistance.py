"""Notebook-only helpers for Fig M7 / Fig S17 — melanoma naive-vs-resistant
analytic Gaussian posterior (CIPHER score) resistance-driver analysis.

Helper module living in ``notebooks/src`` — NOT part of the installable
``cipher`` package. Functions are deduped by name; where a function was
redefined more than once, the LAST definition is kept.

A notebook-only helper for reproducing the supplementary figures.
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

import os
import re
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pathlib import Path

from scipy.sparse import issparse, csr_matrix, diags
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import ttest_ind, norm
from scipy.stats import t as student_t

try:
    from statsmodels.stats.multitest import multipletests
except Exception:  # pragma: no cover
    multipletests = None

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

try:
    from sklearn.decomposition import TruncatedSVD, PCA
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    TruncatedSVD = PCA = StandardScaler = None


# ============================================================
# CONFIG.
#
# Several helper functions read these config constants as MODULE-level
# globals. They are provided here as env-driven defaults so the module is
# importable/usable standalone and pyflakes-clean; the reproduction notebook
# redefines them in its config cell and re-injects (``_M.__dict__.update(...)``)
# so notebook values win. Data paths are rebased onto $CIPHER_DATA_DIR/suppl
# and outputs nest under OUTDIR.
# ============================================================
_DATA_DIR = os.environ.get("CIPHER_DATA_DIR", "")
_SUPPL = os.path.join(_DATA_DIR, "suppl")
_BASE_OUT = os.environ.get("SUPPL_OUT", "resources/repro/figM7_S17")

# analytic Gaussian posterior pipeline output dir
OUTDIR = os.path.join(_BASE_OUT, "analytic_gaussian_FN1_IGFBP7_stable_diag")

# h5ad loaded by the analytic pipeline / rerun fallback (melanoma naive vs resistant)
H5AD_PATH = os.path.join(_SUPPL, "GSE233766", "Xtot_naive_resistant_melanoma_unbalanced.h5ad")

# saved posterior / DE artifacts that run_pipeline writes into OUTDIR
POSTERIOR_SUMMARY_PATH = os.path.join(OUTDIR, "posterior_summary.tsv")
POSTERIOR_MU_PATH = os.path.join(OUTDIR, "posterior_mu.npy")
SELECTED_GENES_PATH = os.path.join(OUTDIR, "selected_genes.npy")
SELECTED_DE_PATH = os.path.join(OUTDIR, "selected_de.tsv")
ALL_GENES_DE_PATH = os.path.join(OUTDIR, "all_genes_de.tsv")

# DE / statistics
FDR_ALPHA = 0.05

# plot style (values for the analytic-posterior rank panels)
DPI = 300
HIGHLIGHT_SIZE = 220
LABEL_FONTSIZE = 8
HIGHLIGHT_FONTSIZE = 11
ARROW_LINEWIDTH = 1.2


__all__ = ['_gene_key', 'add_gene_ranks', 'analytic_gaussian_posterior', 'analytic_gaussian_posterior_yvariant', 'annotate_highlights', 'as_gene_list', 'assign_labels_to_top_band', 'build_H_from_sample_means', 'check_file', 'compute_covariance', 'compute_de_scores', 'compute_log2fc_from_h5ad', 'compute_pca_embedding', 'draw_clear_gene_labels', 'drop_bad_gene_prefixes', 'filter_genes_basic', 'find_lfc_column', 'gene_means', 'get_gene_expr', 'get_gene_idx', 'get_mu_rank_order', 'highlight_gene', 'highlight_genes', 'highlight_genes_ranked', 'import_umap', 'label_gene', 'label_points', 'label_top_points', 'load_de_table', 'load_posterior_summary', 'looks_like_raw_counts', 'make_posterior_summary', 'maybe_rerun_pipeline', 'normalize_gene_list', 'normalize_gene_name', 'normalize_total_log1p', 'ordinal_rank_desc', 'plot_absz_vs_mu_rank', 'plot_de_diagnostics', 'plot_log1p_mu_vs_mu_rank', 'plot_posterior', 'posterior_summary', 'r2_score_centered_at_zero', 'read_h5ad_robust', 'remap_top_idx_to_ranked_positions', 'run_pipeline', 'safe_makedirs', 'save_fig_all_formats', 'select_de_genes', 'select_hvgs_by_variance', 'sparse_mean_var', 'sparse_or_dense_frac_on', 'sparse_or_dense_mean', 'sparse_or_dense_mean_var', 'stable_cho_factor', 'standardize_lfc_columns', 'to_dense', 'welch_de_from_moments']


def read_h5ad_robust(path):
    """
    Load an h5ad file using anndata first and scanpy as a fallback.
    """

    try:
        import anndata as ad
        return ad.read_h5ad(path)

    except Exception as anndata_error:
        try:
            import scanpy as sc
            return sc.read(path)

        except Exception as scanpy_error:
            raise RuntimeError(
                "Could not read the h5ad file.\n\n"
                f"anndata error:\n{repr(anndata_error)}\n\n"
                f"scanpy error:\n{repr(scanpy_error)}"
            )


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}\nPWD: {os.getcwd()}")
    print(f"[OK] Found file: {path}")


def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def to_dense(X):
    """
    Convert a sparse or dense expression matrix to a NumPy array.
    """

    if issparse(X):
        return X.toarray()

    return np.asarray(X)


def drop_bad_gene_prefixes(
    gene_names,
    bad_prefixes=(
        "RPL",
        "RPS",
        "MT-",
        "MT.",
        "MTRNR",
        "MTRNR2L",
        "HSP",
        "HSP90",
        "EIF",
        "MALAT1",
    ),
):
    """
    Return a Boolean mask excluding common housekeeping/junk genes.
    """

    names = np.asarray(gene_names, dtype=str)
    upper_names = np.char.upper(names)

    keep = np.ones(len(names), dtype=bool)

    for prefix in bad_prefixes:
        keep &= ~np.char.startswith(
            upper_names,
            prefix.upper(),
        )

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
    """
    Filter genes based on expression frequency, mean, and variance.
    """

    rng = np.random.default_rng(seed)

    if (
        filter_subsample_cells
        and adata.n_obs > filter_subsample_cells
    ):
        sampled_indices = rng.choice(
            np.arange(adata.n_obs),
            size=int(filter_subsample_cells),
            replace=False,
        )

        adata_for_filter = adata[sampled_indices].copy()

    else:
        adata_for_filter = adata

    X = to_dense(adata_for_filter.X).astype(
        np.float64,
        copy=False,
    )

    fraction_expressed = np.mean(
        X >= min_expr,
        axis=0,
    )

    gene_mean = np.mean(
        X,
        axis=0,
    )

    gene_variance = np.var(
        X,
        axis=0,
    )

    keep = (
        (fraction_expressed >= min_cells_frac)
        & (gene_mean >= min_mean)
        & (gene_mean <= max_mean)
    )

    if max_var_quantile < 1.0 and np.any(keep):
        variance_cutoff = np.quantile(
            gene_variance[keep],
            max_var_quantile,
        )

        keep &= gene_variance <= variance_cutoff

    print(
        "[filter] Basic expression filter kept "
        f"{int(keep.sum()):,} / {len(keep):,} genes"
    )

    if keep.sum() == 0:
        raise ValueError(
            "The basic expression filter removed every gene."
        )

    return adata[:, keep].copy()


def compute_de_scores(
    X0,
    X1,
    gene_names,
    logfc_pseudocount=1.0,
    eps=1e-12,
):
    """
    Calculate mean expression, delta expression, log2FC,
    Welch t statistics, p-values, and BH-adjusted p-values.
    """

    X0 = np.asarray(
        X0,
        dtype=np.float64,
    )

    X1 = np.asarray(
        X1,
        dtype=np.float64,
    )

    gene_names = np.asarray(
        gene_names,
        dtype=str,
    )

    mean0 = X0.mean(axis=0)
    mean1 = X1.mean(axis=0)

    delta = mean1 - mean0

    variance0 = X0.var(
        axis=0,
        ddof=1,
    )

    variance1 = X1.var(
        axis=0,
        ddof=1,
    )

    n0 = X0.shape[0]
    n1 = X1.shape[0]

    standard_error = np.sqrt(
        variance0 / max(n0, 1)
        + variance1 / max(n1, 1)
    ) + eps

    t_stat = delta / standard_error

    t_stat = np.nan_to_num(
        t_stat,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        _, p_value = ttest_ind(
            X1,
            X0,
            axis=0,
            equal_var=False,
        )

    p_value = np.asarray(
        p_value,
        dtype=np.float64,
    )

    p_value = np.nan_to_num(
        p_value,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    _, p_adjusted, _, _ = multipletests(
        p_value,
        alpha=FDR_ALPHA,
        method="fdr_bh",
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        log2fc = np.log2(
            (mean1 + logfc_pseudocount)
            / (mean0 + logfc_pseudocount)
        )

    log2fc = np.nan_to_num(
        log2fc,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    de = pd.DataFrame(
        {
            "gene": gene_names,
            "mean_cond0": mean0,
            "mean_cond1": mean1,
            "delta": delta,
            "abs_delta": np.abs(delta),
            "log2fc": log2fc,
            "abs_log2fc": np.abs(log2fc),
            "t_stat": t_stat,
            "abs_t": np.abs(t_stat),
            "p_value": p_value,
            "p_adj": p_adjusted,
            "neglog10_p": -np.log10(
                np.maximum(p_value, 1e-300)
            ),
            "neglog10_padj": -np.log10(
                np.maximum(p_adjusted, 1e-300)
            ),
            "fdr_sig": (
                p_adjusted < FDR_ALPHA
            ).astype(int),
        }
    )

    de = (
        de
        .sort_values(
            by=[
                "fdr_sig",
                "abs_t",
                "p_adj",
                "gene",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    de["global_rank"] = np.arange(
        1,
        len(de) + 1,
    )

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
    """
    Select genes passing the specified DE criteria and optionally
    fill the set to top_n_de using the selected ranking statistic.
    """

    if rank_by not in de_df.columns:
        raise KeyError(
            f"DE ranking column '{rank_by}' was not found."
        )

    passing = de_df.loc[
        (de_df["p_adj"] < fdr_alpha)
        & (
            de_df["abs_log2fc"]
            >= min_abs_log2fc
        )
        & (
            de_df["abs_delta"]
            >= min_abs_delta
        )
    ].copy()

    passing = (
        passing
        .sort_values(
            by=[
                rank_by,
                "p_adj",
                "gene",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if fill_to_top_n and len(passing) < top_n_de:
        already_selected = set(
            passing["gene"].astype(str)
        )

        filler = de_df.loc[
            ~de_df["gene"]
            .astype(str)
            .isin(already_selected)
        ].copy()

        filler = (
            filler
            .sort_values(
                by=[
                    rank_by,
                    "p_adj",
                    "gene",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        number_needed = (
            top_n_de - len(passing)
        )

        selected = pd.concat(
            [
                passing,
                filler.head(number_needed),
            ],
            axis=0,
            ignore_index=True,
        )

    else:
        selected = (
            passing
            .head(top_n_de)
            .copy()
            .reset_index(drop=True)
        )

    selected = (
        selected
        .head(top_n_de)
        .copy()
        .reset_index(drop=True)
    )

    selected["selected_rank"] = np.arange(
        1,
        len(selected) + 1,
    )

    selected["passed_primary_filter"] = (
        (selected["p_adj"] < fdr_alpha)
        & (
            selected["abs_log2fc"]
            >= min_abs_log2fc
        )
        & (
            selected["abs_delta"]
            >= min_abs_delta
        )
    ).astype(int)

    print("\n[selection]")
    print(
        f"  Primary passing genes: {len(passing):,}"
    )
    print(
        f"  Final selected genes:  {len(selected):,}"
    )
    print(
        f"  Selection ranking:     {rank_by}"
    )
    print(
        f"  Fill to top N:         {fill_to_top_n}"
    )

    return selected, passing


def compute_covariance(
    X,
    shrinkage=1e-6,
):
    """
    Calculate a covariance matrix with diagonal shrinkage.
    """

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    centered = X - X.mean(
        axis=0,
        keepdims=True,
    )

    covariance = (
        centered.T @ centered
    ) / max(1, X.shape[0] - 1)

    diagonal_scale = (
        np.mean(np.diag(covariance))
        + 1e-12
    )

    covariance += (
        float(shrinkage)
        * diagonal_scale
        * np.eye(covariance.shape[0])
    )

    return covariance


def build_H_from_sample_means(
    X0,
    X1,
    shrinkage=1e-6,
    ridge=1e-6,
    mode="naive",
):
    """
    Construct the covariance of the estimated mean shift.
    """

    X0 = np.asarray(
        X0,
        dtype=np.float64,
    )

    X1 = np.asarray(
        X1,
        dtype=np.float64,
    )

    n0 = X0.shape[0]
    n1 = X1.shape[0]

    covariance0 = compute_covariance(
        X0,
        shrinkage=shrinkage,
    )

    covariance1 = compute_covariance(
        X1,
        shrinkage=shrinkage,
    )

    if mode == "diag":
        diagonal = (
            np.diag(covariance0) / max(n0, 1)
            + np.diag(covariance1) / max(n1, 1)
        )

        H = np.diag(diagonal)

    elif mode == "full":
        H = (
            covariance0 / max(n0, 1)
            + covariance1 / max(n1, 1)
        )

    elif mode == "naive":
        H = (
            1.0 / max(n0, 1)
            + 1.0 / max(n1, 1)
        ) * covariance0

    else:
        raise ValueError(
            "H_MODE must be one of: "
            "'diag', 'full', or 'naive'."
        )

    diagonal_scale = (
        np.mean(np.diag(H))
        + 1e-12
    )

    H += (
        float(ridge)
        * diagonal_scale
        * np.eye(H.shape[0])
    )

    return H


def r2_score_centered_at_zero(
    observed,
    predicted,
    eps=1e-12,
):
    """
    Calculate the uncentered R² used by the original pipeline.
    """

    observed = np.asarray(
        observed,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    numerator = np.sum(
        (observed - predicted) ** 2
    )

    denominator = (
        np.sum(observed ** 2)
        + eps
    )

    return 1.0 - numerator / denominator


def analytic_gaussian_posterior_yvariant(Sigma, y, H, tau2=1.0):
    Sigma = np.asarray(Sigma, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    H = np.asarray(H, dtype=np.float64)

    choH = cho_factor(H, lower=True, check_finite=False)

    Hinv_y = cho_solve(choH, y, check_finite=False)
    Hinv_Sigma = cho_solve(choH, Sigma, check_finite=False)

    Precision = Sigma.T @ Hinv_Sigma + np.eye(Sigma.shape[1]) / float(tau2)
    choP = cho_factor(Precision, lower=True, check_finite=False)

    Cov = cho_solve(choP, np.eye(Precision.shape[0]), check_finite=False)
    mu = cho_solve(choP, Sigma.T @ Hinv_y, check_finite=False)

    std = np.sqrt(np.maximum(np.diag(Cov), 0.0))
    yhat = Sigma @ mu
    r2 = r2_score_centered_at_zero(y, yhat)

    return {
        "mu": mu,
        "Cov": Cov,
        "std": std,
        "yhat": yhat,
        "r2": r2,
    }


def analytic_gaussian_posterior(
    Sigma,
    response,
    H,
    tau2=1e-6,
):
    """
    Compute the analytic Gaussian posterior over gene effects.
    """

    Sigma = np.asarray(
        Sigma,
        dtype=np.float64,
    )

    response = np.asarray(
        response,
        dtype=np.float64,
    ).reshape(-1)

    H = np.asarray(
        H,
        dtype=np.float64,
    )

    H_factor = stable_cho_factor(
        H,
        matrix_name="H",
    )

    H_inverse_response = cho_solve(
        H_factor,
        response,
        check_finite=False,
    )

    H_inverse_Sigma = cho_solve(
        H_factor,
        Sigma,
        check_finite=False,
    )

    precision = (
        Sigma.T @ H_inverse_Sigma
        + np.eye(Sigma.shape[1]) / float(tau2)
    )

    precision_factor = stable_cho_factor(
        precision,
        matrix_name="posterior precision",
    )

    identity = np.eye(
        precision.shape[0],
    )

    posterior_covariance = cho_solve(
        precision_factor,
        identity,
        check_finite=False,
    )

    posterior_mean = cho_solve(
        precision_factor,
        Sigma.T @ H_inverse_response,
        check_finite=False,
    )

    posterior_std = np.sqrt(
        np.maximum(
            np.diag(posterior_covariance),
            0.0,
        )
    )

    predicted_response = (
        Sigma @ posterior_mean
    )

    r2 = r2_score_centered_at_zero(
        response,
        predicted_response,
    )

    return {
        "mu": posterior_mean,
        "Cov": posterior_covariance,
        "std": posterior_std,
        "yhat": predicted_response,
        "r2": r2,
    }


def posterior_summary(mu, std, gene_names, delta_x, effect_threshold=None):
    mu = np.asarray(mu, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    gene_names = np.asarray(gene_names, dtype=str)

    if effect_threshold is None:
        effect_threshold = float(np.median(std))

    z = mu / (std + 1e-12)

    upper = (effect_threshold - mu) / (std + 1e-12)
    lower = (-effect_threshold - mu) / (std + 1e-12)

    pip = 1.0 - (norm.cdf(upper) - norm.cdf(lower))

    p_pos = 1.0 - norm.cdf((0.0 - mu) / (std + 1e-12))
    p_neg = norm.cdf((0.0 - mu) / (std + 1e-12))
    sign_conf = np.maximum(p_pos, p_neg)

    ci_lo = mu - 1.96 * std
    ci_hi = mu + 1.96 * std
    zero_excluded = ((ci_lo > 0) | (ci_hi < 0)).astype(int)

    df = pd.DataFrame({
        "gene": gene_names,
        "mu": mu,
        "std": std,
        "z": z,
        "abs_z": np.abs(z),
        "pip": pip,
        "p_pos": p_pos,
        "p_neg": p_neg,
        "sign_conf": sign_conf,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "zero_excluded": zero_excluded,
        "delta_x": delta_x,
    }).sort_values("pip", ascending=False).reset_index(drop=True)

    return df, effect_threshold


def label_points(ax, x, y, labels, idxs, fontsize=8):
    for i in idxs:
        ax.text(x[i], y[i], str(labels[i]), fontsize=fontsize)


def highlight_gene(ax, gene_names, gene_to_check, x, y):
    gene_names = np.asarray(gene_names, dtype=str)
    hits = np.where(np.char.upper(gene_names) == gene_to_check.upper())[0]

    if len(hits) > 0:
        j = hits[0]
        ax.scatter([x[j]], [y[j]], s=140, marker="*", zorder=10)
        ax.text(x[j], y[j], f" {gene_to_check}", fontsize=11)


def plot_de_diagnostics(de_df, selected_de, genes_to_highlight, outdir, top_k_label=30):
    genes_to_highlight = as_gene_list(genes_to_highlight)

    # selected rank plot
    fig, ax = plt.subplots(figsize=(15, 5))

    vals = selected_de["abs_t"].values
    x = np.arange(len(vals))
    labels = selected_de["gene"].values.astype(str)

    ax.plot(x, vals, lw=1)
    top = np.arange(min(top_k_label, len(vals)))
    ax.scatter(x[top], vals[top], s=25)
    label_points(ax, x, vals, labels, top)

    highlight_genes(ax, labels, genes_to_highlight, x, vals)

    ax.set_xlabel("Selected gene rank")
    ax.set_ylabel("|t statistic|")
    ax.set_title("Selected genes ranked by |t statistic|")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "selected_gene_rank_abs_t.png"), dpi=200)
    plt.show()

    # volcano
    fig, ax = plt.subplots(figsize=(7, 6))

    x = de_df["log2fc"].values
    y = de_df["neglog10_padj"].values
    sig = de_df["p_adj"].values < 0.05
    all_genes = de_df["gene"].values.astype(str)

    ax.scatter(x[~sig], y[~sig], s=8, alpha=0.35)
    ax.scatter(x[sig], y[sig], s=8, alpha=0.7)

    top_idx = np.argsort(-de_df["abs_t"].values)[:min(top_k_label, len(de_df))]
    ax.scatter(x[top_idx], y[top_idx], s=25)
    label_points(ax, x, y, all_genes, top_idx)

    highlight_genes(ax, all_genes, genes_to_highlight, x, y)

    ax.axhline(-np.log10(0.05), ls="--", lw=1)
    ax.set_xlabel("log2FC")
    ax.set_ylabel("-log10 adjusted p-value")
    ax.set_title("DE volcano")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "de_volcano.png"), dpi=200)
    plt.show()

    # p_adj histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(de_df["p_adj"].values, bins=80)
    ax.set_xlabel("adjusted p-value")
    ax.set_ylabel("count")
    ax.set_title("Adjusted p-value distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "p_adj_hist.png"), dpi=200)
    plt.show()

    # t histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(de_df["t_stat"].values, bins=80)
    ax.set_xlabel("t statistic")
    ax.set_ylabel("count")
    ax.set_title("t statistic distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "t_stat_hist.png"), dpi=200)
    plt.show()


def plot_posterior(summary, y, yhat, r2, genes_to_highlight, outdir, top_k=50):
    """
    Replacement for your existing plot_posterior(...).

    Makes:
      1. Top PIP bar plot
      2. Posterior effect vs PIP
      3. Posterior |z| across all selected genes
      4. Clipped posterior |z| plot so the smaller peaks are visible
      5. Zoomed top |z| peaks plot
      6. Highlighted genes only posterior z plot
      7. Observed vs predicted delta_x plot
    """

    genes_to_highlight = as_gene_list(genes_to_highlight)

    # ------------------------------------------------------------
    # Original summary order may be by PIP.
    # Keep one PIP-ranked dataframe and one z-ranked dataframe.
    # ------------------------------------------------------------
    df_pip = summary.sort_values("pip", ascending=False).reset_index(drop=True).copy()
    df_z = summary.sort_values("abs_z", ascending=False).reset_index(drop=True).copy()

    df_z["posterior_abs_z_rank"] = np.arange(1, len(df_z) + 1)

    df_z.to_csv(
        os.path.join(outdir, "posterior_all_genes_ranked_by_abs_z.tsv"),
        sep="\t",
        index=False,
    )

    # ============================================================
    # 1. Top PIP bar plot
    # ============================================================
    gene_names = df_pip["gene"].values.astype(str)
    pip = df_pip["pip"].values
    mu = df_pip["mu"].values
    std = df_pip["std"].values

    n = len(df_pip)
    top = np.arange(min(top_k, n))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(top))))

    vals = pip[top][::-1]
    labs = gene_names[top][::-1]

    ax.barh(np.arange(len(vals)), vals)
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(labs)
    ax.set_xlabel("PIP-like activity probability")
    ax.set_xlim(0, 1)
    ax.set_title("Top posterior activity genes")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "top_pip_genes.png"), dpi=250)
    plt.show()

    # ============================================================
    # 2. Posterior effect vs PIP
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(mu, pip, s=15, alpha=0.6)
    ax.scatter(mu[top], pip[top], s=35)

    label_points(ax, mu, pip, gene_names, top, fontsize=8)
    highlight_genes(ax, gene_names, genes_to_highlight, mu, pip, fontsize=13)

    ax.axvline(0, lw=1)
    ax.set_xlabel("posterior mean effect")
    ax.set_ylabel("PIP-like activity probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("Posterior effect vs PIP")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "effect_vs_pip.png"), dpi=250)
    plt.show()

    # ============================================================
    # 3. Posterior |z| across all selected genes
    # ============================================================
    genes_z = df_z["gene"].values.astype(str)
    z_signed = df_z["z"].values
    abs_z = df_z["abs_z"].values
    x = np.arange(len(df_z))

    top_z = np.arange(min(top_k, len(df_z)))

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(x, abs_z, lw=0.8, alpha=0.65)
    ax.scatter(x, abs_z, s=12, alpha=0.65)

    ax.scatter(x[top_z], abs_z[top_z], s=40, zorder=10)
    label_points(ax, x, abs_z, genes_z, top_z, fontsize=8)

    highlight_genes(ax, genes_z, genes_to_highlight, x, abs_z, fontsize=14)

    ax.set_xlabel("Genes ranked by |posterior z|")
    ax.set_ylabel("|posterior z|")
    ax.set_title("Posterior |z| across selected genes")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "posterior_abs_z_all_selected_genes.png"), dpi=250)
    plt.show()

    # ============================================================
    # 4. Clipped posterior |z| plot
    #    This is for visibility when FN1/IGFBP7 dominate the scale.
    # ============================================================
    clip_quantile = 0.985
    y_clip = np.quantile(abs_z, clip_quantile)
    y_clip = max(y_clip, np.percentile(abs_z, 90))

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(x, abs_z, lw=0.8, alpha=0.65)
    ax.scatter(x, abs_z, s=12, alpha=0.65)

    visible_top = [i for i in top_z if abs_z[i] <= y_clip * 1.05]

    ax.scatter(x[top_z], abs_z[top_z], s=40, zorder=10)

    for i in visible_top:
        ax.text(x[i], abs_z[i], genes_z[i], fontsize=8)

    highlight_genes(ax, genes_z, genes_to_highlight, x, abs_z, fontsize=14)

    ax.set_ylim(0, y_clip * 1.2)
    ax.set_xlabel("Genes ranked by |posterior z|")
    ax.set_ylabel("|posterior z|")
    ax.set_title(f"Posterior |z| across selected genes, clipped at q={clip_quantile}")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "posterior_abs_z_all_selected_genes_clipped.png"), dpi=250)
    plt.show()

    # ============================================================
    # 5. Zoomed top posterior |z| peaks
    # ============================================================
    zoom_top_n = min(250, len(df_z))
    x_zoom = np.arange(zoom_top_n)

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(x_zoom, abs_z[:zoom_top_n], lw=0.8, alpha=0.65)
    ax.scatter(x_zoom, abs_z[:zoom_top_n], s=18, alpha=0.75)

    top_zoom = np.arange(min(top_k, zoom_top_n))

    ax.scatter(x_zoom[top_zoom], abs_z[top_zoom], s=45, zorder=10)
    label_points(ax, x_zoom, abs_z[:zoom_top_n], genes_z[:zoom_top_n], top_zoom, fontsize=8)

    highlight_genes(
        ax,
        genes_z[:zoom_top_n],
        genes_to_highlight,
        x_zoom,
        abs_z[:zoom_top_n],
        fontsize=14,
    )

    ax.set_xlabel(f"Top {zoom_top_n} genes ranked by |posterior z|")
    ax.set_ylabel("|posterior z|")
    ax.set_title(f"Zoomed posterior |z| peaks: top {zoom_top_n}")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "posterior_abs_z_top_zoom.png"), dpi=250)
    plt.show()

    # ============================================================
    # 6. Signed posterior z across all selected genes
    # ============================================================
    fig, ax = plt.subplots(figsize=(18, 6))

    ax.axhline(0, lw=1)
    ax.plot(x, z_signed, lw=0.8, alpha=0.65)
    ax.scatter(x, z_signed, s=12, alpha=0.65)

    ax.scatter(x[top_z], z_signed[top_z], s=40, zorder=10)
    label_points(ax, x, z_signed, genes_z, top_z, fontsize=8)

    highlight_genes(ax, genes_z, genes_to_highlight, x, z_signed, fontsize=14)

    ax.set_xlabel("Genes ranked by |posterior z|")
    ax.set_ylabel("signed posterior z = mu / std")
    ax.set_title("Signed posterior z across selected genes")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "posterior_signed_z_all_selected_genes.png"), dpi=250)
    plt.show()

    # ============================================================
    # 7. Highlighted genes only
    # ============================================================
    highlight_rows = []

    for gene in genes_to_highlight:
        hit = df_z.loc[df_z["gene"].str.upper() == gene.upper()].copy()
        if len(hit) > 0:
            highlight_rows.append(hit.iloc[0])

    if len(highlight_rows) > 0:
        hdf = pd.DataFrame(highlight_rows).copy()
        hdf = hdf.sort_values("z", ascending=True)

        hdf.to_csv(
            os.path.join(outdir, "posterior_highlight_gene_z_values.tsv"),
            sep="\t",
            index=False,
        )

        fig, ax = plt.subplots(figsize=(7, max(3, 0.6 * len(hdf) + 1)))

        ypos = np.arange(len(hdf))
        vals = hdf["z"].values

        ax.barh(ypos, vals)
        ax.axvline(0, lw=1)

        ax.set_yticks(ypos)
        ax.set_yticklabels(hdf["gene"].values)
        ax.set_xlabel("signed posterior z")
        ax.set_title("Posterior z for highlighted genes")

        for i, val in enumerate(vals):
            ax.text(val, i, f" {val:.2f}", va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "posterior_z_highlight_genes_only.png"), dpi=250)
        plt.show()

    # ============================================================
    # 8. Observed vs predicted delta
    # ============================================================
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(y, yhat, s=20, alpha=0.7)

    lim = max(np.max(np.abs(y)), np.max(np.abs(yhat))) + 1e-12
    ax.plot([-lim, lim], [-lim, lim], "--", lw=1)

    ax.set_xlabel("observed Δx")
    ax.set_ylabel("predicted Δx")
    ax.set_title(f"Posterior fit, R²={r2:.3f}")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "observed_vs_predicted_delta_x.png"), dpi=250)
    plt.show()

    print("\n[posterior plots saved]")
    print("  top_pip_genes.png")
    print("  effect_vs_pip.png")
    print("  posterior_abs_z_all_selected_genes.png")
    print("  posterior_abs_z_all_selected_genes_clipped.png")
    print("  posterior_abs_z_top_zoom.png")
    print("  posterior_signed_z_all_selected_genes.png")
    print("  posterior_z_highlight_genes_only.png")
    print("  posterior_all_genes_ranked_by_abs_z.tsv")


def run_pipeline(
    h5ad_path,
    outdir="analytic_gaussian_fixed_selection",
    condition_key="Condition",
    cond0="Naive",
    cond1="Resistant",
    genes_to_highlight=("FN1", "IGFBP7"),

    top_n_de=1000,
    fdr_alpha=0.05,
    min_abs_log2fc=0.0,
    min_abs_delta=0.0,
    rank_by="abs_t",
    fill_to_top_n=True,

    drop_housekeeping=True,
    min_cells_frac=0.001,
    min_expr=0.01,
    min_mean=0.001,
    max_mean=np.inf,
    max_var_quantile=1.0,
    filter_subsample_cells=0,

    logfc_pseudocount=0.1,

    Sigma_shrinkage=1e-3,
    H_shrinkage=1e-3,
    H_ridge=1e-4,
    H_mode="diag",
    tau2=1.0,
    effect_threshold=None,

    top_k_plot=50,
    seed=0,
):
    genes_to_highlight = as_gene_list(genes_to_highlight)

    safe_makedirs(outdir)
    check_file(h5ad_path)

    adata = read_h5ad_robust(h5ad_path)
    adata.var_names_make_unique()

    print(f"[data] loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    if condition_key not in adata.obs.columns:
        raise KeyError(
            f"{condition_key} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    print(f"[data] available {condition_key}:")
    print(pd.Series(adata.obs[condition_key]).value_counts())

    adata = adata[adata.obs[condition_key].isin([cond0, cond1])].copy()

    m0 = np.asarray(adata.obs[condition_key].values == cond0)
    m1 = np.asarray(adata.obs[condition_key].values == cond1)

    print(f"\n[contrast] {cond1} - {cond0}")
    print(f"  {cond0}: {m0.sum()} cells")
    print(f"  {cond1}: {m1.sum()} cells")

    if m0.sum() < 5 or m1.sum() < 5:
        raise ValueError(f"Too few cells: {cond0}={m0.sum()}, {cond1}={m1.sum()}")

    adata = filter_genes_basic(
        adata,
        min_cells_frac=min_cells_frac,
        min_expr=min_expr,
        min_mean=min_mean,
        max_mean=max_mean,
        max_var_quantile=max_var_quantile,
        seed=seed,
        filter_subsample_cells=filter_subsample_cells,
    )

    if drop_housekeeping:
        keep = drop_bad_gene_prefixes(adata.var_names)
        print(f"[filter] bad-prefix filter kept {keep.sum()} / {len(keep)} genes")
        adata = adata[:, keep].copy()

    m0 = np.asarray(adata.obs[condition_key].values == cond0)
    m1 = np.asarray(adata.obs[condition_key].values == cond1)

    X0 = to_dense(adata[m0].X).astype(np.float64)
    X1 = to_dense(adata[m1].X).astype(np.float64)
    gene_names_all = np.asarray(adata.var_names, dtype=str)

    print(f"[matrix] X0: {X0.shape}")
    print(f"[matrix] X1: {X1.shape}")

    de_df = compute_de_scores(
        X0,
        X1,
        gene_names_all,
        logfc_pseudocount=logfc_pseudocount,
    )

    de_df.to_csv(os.path.join(outdir, "all_genes_de.tsv"), sep="\t", index=False)

    print("\n[DE diagnostic counts]")
    print(f"  total genes tested: {len(de_df)}")
    print(f"  FDR < 0.05: {(de_df['p_adj'] < 0.05).sum()}")

    for cut in [0.0, 0.1, 0.25, 0.5, 1.0]:
        n = ((de_df["p_adj"] < 0.05) & (de_df["abs_log2fc"] >= cut)).sum()
        print(f"  FDR < 0.05 and abs_log2FC >= {cut}: {n}")

    selected_de, primary_passing_de = select_de_genes(
        de_df,
        top_n_de=top_n_de,
        fdr_alpha=fdr_alpha,
        min_abs_log2fc=min_abs_log2fc,
        min_abs_delta=min_abs_delta,
        rank_by=rank_by,
        fill_to_top_n=fill_to_top_n,
    )

    selected_de.to_csv(os.path.join(outdir, "selected_de.tsv"), sep="\t", index=False)
    primary_passing_de.to_csv(os.path.join(outdir, "primary_passing_de.tsv"), sep="\t", index=False)

    print("\n[top selected genes]")
    print(selected_de[[
        "selected_rank",
        "gene",
        "passed_primary_filter",
        "mean_cond0",
        "mean_cond1",
        "delta",
        "log2fc",
        "t_stat",
        "p_adj",
    ]].head(30).to_string(index=False))

    print("\n[highlight gene checks]")
    highlight_rows = []
    for gene in genes_to_highlight:
        gene_status = de_df.loc[de_df["gene"].str.upper() == gene.upper()].copy()
        in_selected = gene.upper() in set(selected_de["gene"].str.upper())

        if len(gene_status) > 0:
            row = gene_status.iloc[0].to_dict()
            row["highlight_gene"] = gene
            row["in_selected"] = in_selected
            highlight_rows.append(row)

            print(f"\n{gene}:")
            print(gene_status[[
                "gene", "global_rank", "mean_cond0", "mean_cond1",
                "delta", "log2fc", "t_stat", "p_adj"
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
        os.path.join(outdir, "highlight_gene_status.tsv"),
        sep="\t",
        index=False,
    )

    plot_de_diagnostics(
        de_df=de_df,
        selected_de=selected_de,
        genes_to_highlight=genes_to_highlight,
        outdir=outdir,
        top_k_label=min(top_k_plot, len(selected_de)),
    )

    selected_genes = selected_de["gene"].values.astype(str)
    selected_set = set(selected_genes)

    selected_mask = np.asarray([g in selected_set for g in gene_names_all])
    adata_sel = adata[:, selected_mask].copy()
    gene_names_sel_unordered = np.asarray(adata_sel.var_names, dtype=str)

    X0_sel_unordered = to_dense(adata_sel[m0].X).astype(np.float64)
    X1_sel_unordered = to_dense(adata_sel[m1].X).astype(np.float64)

    name_to_idx = {g: i for i, g in enumerate(gene_names_sel_unordered)}
    order = [name_to_idx[g] for g in selected_genes if g in name_to_idx]

    X0_sel = X0_sel_unordered[:, order]
    X1_sel = X1_sel_unordered[:, order]
    gene_names_sel = gene_names_sel_unordered[order]

    print(f"\n[selected matrix] X0 selected: {X0_sel.shape}")
    print(f"[selected matrix] X1 selected: {X1_sel.shape}")

    delta_x = X1_sel.mean(axis=0) - X0_sel.mean(axis=0)

    Sigma = compute_covariance(
        X0_sel,
        shrinkage=Sigma_shrinkage,
    )

    H = build_H_from_sample_means(
        X0_sel,
        X1_sel,
        shrinkage=H_shrinkage,
        ridge=H_ridge,
        mode=H_mode,
    )

    post = analytic_gaussian_posterior_yvariant(
        Sigma=Sigma,
        y=delta_x,
        H=H,
        tau2=tau2,
    )

    summary, used_effect_threshold = posterior_summary(
        mu=post["mu"],
        std=post["std"],
        gene_names=gene_names_sel,
        delta_x=delta_x,
        effect_threshold=effect_threshold,
    )

    summary.to_csv(os.path.join(outdir, "posterior_summary.tsv"), sep="\t", index=False)

    np.save(os.path.join(outdir, "selected_genes.npy"), gene_names_sel)
    np.save(os.path.join(outdir, "Sigma.npy"), Sigma)
    np.save(os.path.join(outdir, "H.npy"), H)
    np.save(os.path.join(outdir, "delta_x.npy"), delta_x)
    np.save(os.path.join(outdir, "posterior_mu.npy"), post["mu"])
    np.save(os.path.join(outdir, "posterior_std.npy"), post["std"])
    np.save(os.path.join(outdir, "posterior_cov.npy"), post["Cov"])
    np.save(os.path.join(outdir, "posterior_yhat.npy"), post["yhat"])

    print("\n[posterior]")
    print(f"  R2: {post['r2']:.4f}")
    print(f"  tau2: {tau2}")
    print(f"  H_mode: {H_mode}")
    print(f"  effect_threshold: {used_effect_threshold}")

    print("\n[top posterior genes]")
    print(summary.head(30).to_string(index=False))

    print("\n[highlight posterior status]")
    for gene in genes_to_highlight:
        hit = summary.loc[summary["gene"].str.upper() == gene.upper()]
        if len(hit) > 0:
            print(f"\n{gene}:")
            print(hit.to_string(index=False))
        else:
            print(f"\n{gene}: not in posterior selected set")

    plot_posterior(
        summary=summary,
        y=delta_x,
        yhat=post["yhat"],
        r2=post["r2"],
        genes_to_highlight=genes_to_highlight,
        outdir=outdir,
        top_k=min(top_k_plot, len(summary)),
    )

    with open(os.path.join(outdir, "run_summary.txt"), "w") as f:
        f.write(f"h5ad_path: {h5ad_path}\n")
        f.write(f"contrast: {cond1} - {cond0}\n")
        f.write(f"condition_key: {condition_key}\n")
        f.write(f"genes_to_highlight: {genes_to_highlight}\n")
        f.write(f"n_cells_cond0: {m0.sum()}\n")
        f.write(f"n_cells_cond1: {m1.sum()}\n")
        f.write(f"n_genes_after_filtering: {len(gene_names_all)}\n")
        f.write(f"top_n_de_requested: {top_n_de}\n")
        f.write(f"n_selected_genes: {len(selected_de)}\n")
        f.write(f"fdr_alpha: {fdr_alpha}\n")
        f.write(f"min_abs_log2fc: {min_abs_log2fc}\n")
        f.write(f"min_abs_delta: {min_abs_delta}\n")
        f.write(f"rank_by: {rank_by}\n")
        f.write(f"fill_to_top_n: {fill_to_top_n}\n")
        f.write(f"primary_passing_genes: {len(primary_passing_de)}\n")
        f.write(f"Sigma_shrinkage: {Sigma_shrinkage}\n")
        f.write(f"H_shrinkage: {H_shrinkage}\n")
        f.write(f"H_ridge: {H_ridge}\n")
        f.write(f"H_mode: {H_mode}\n")
        f.write(f"tau2: {tau2}\n")
        f.write(f"effect_threshold: {used_effect_threshold}\n")
        f.write(f"posterior_R2: {post['r2']}\n")

    print("\n" + "=" * 70)
    print("[DONE]")
    print(f"outputs written to: {outdir}")
    print("=" * 70)

    return {
        "adata_selected": adata_sel,
        "de_df": de_df,
        "selected_de": selected_de,
        "primary_passing_de": primary_passing_de,
        "selected_genes": gene_names_sel,
        "X0_selected": X0_sel,
        "X1_selected": X1_sel,
        "delta_x": delta_x,
        "Sigma": Sigma,
        "H": H,
        "posterior": post,
        "summary": summary,
    }


def as_gene_list(genes):
    if genes is None:
        return []
    if isinstance(genes, str):
        return [genes]
    return list(genes)


def highlight_genes(ax, gene_names, genes_to_highlight, x, y, fontsize=11):
    genes_to_highlight = as_gene_list(genes_to_highlight)
    gene_names = np.asarray(gene_names, dtype=str)
    upper_names = np.char.upper(gene_names)

    for gene in genes_to_highlight:
        hits = np.where(upper_names == gene.upper())[0]
        if len(hits) > 0:
            j = hits[0]
            ax.scatter([x[j]], [y[j]], s=160, marker="*", zorder=20)
            ax.text(x[j], y[j], f" {gene}", fontsize=fontsize, zorder=21)


def normalize_gene_name(gene):
    """
    Normalize a gene name for case-insensitive matching.
    """
    return str(gene).strip().upper()


def ordinal_rank_desc(values, valid=None):
    """
    Return 1-based descending ordinal ranks.

    Rank 1 is assigned to the largest finite value.
    Invalid values receive NaN.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if valid is None:
        valid = np.isfinite(values)

    else:
        valid = (
            np.asarray(valid, dtype=bool)
            & np.isfinite(values)
        )

    ranks = np.full(
        len(values),
        np.nan,
        dtype=np.float64,
    )

    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:
        return ranks

    ordered_indices = valid_indices[
        np.argsort(
            -values[valid_indices],
            kind="mergesort",
        )
    ]

    ranks[ordered_indices] = np.arange(
        1,
        len(ordered_indices) + 1,
        dtype=np.float64,
    )

    return ranks


def sparse_or_dense_mean(X):
    """
    Compute a column mean for either sparse or dense matrices.
    """
    if issparse(X):
        return np.asarray(
            X.mean(axis=0)
        ).ravel().astype(np.float64)

    return np.asarray(
        X,
        dtype=np.float64,
    ).mean(axis=0)


def find_lfc_column(df):
    """
    Return the first recognized signed-log2FC column, if present.
    """
    candidates = [
        "log2fc",
        "log2FC",
        "log2_fc",
        "log_fc",
        "logFC",
        "lfc",
        "LFC",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    lowercase_to_original = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lowercase_to_original:
            return lowercase_to_original[
                candidate.lower()
            ]

    return None


def compute_log2fc_from_h5ad(
    h5ad_path,
    genes,
    condition_key,
    cond0,
    cond1,
    pseudocount=1.0,
):
    """
    Compute signed log2FC directly from the H5AD file for the
    supplied genes.

    log2FC is defined as:

        log2((mean_cond1 + pseudocount) /
             (mean_cond0 + pseudocount))

    Only the requested genes are loaded into memory.
    """
    h5ad_path = Path(
        h5ad_path
    )

    if not h5ad_path.exists():
        raise FileNotFoundError(
            "Cannot recompute log2FC because the H5AD file "
            "was not found:\n"
            f"  {h5ad_path}"
        )

    print(
        "[LFC] Signed log2FC was not available in the saved "
        "posterior summary."
    )

    print(
        f"[LFC] Recomputing log2FC from:\n"
        f"      {h5ad_path}"
    )

    adata = ad.read_h5ad(
        h5ad_path
    )

    adata.var_names_make_unique()

    if condition_key not in adata.obs.columns:
        raise KeyError(
            f"'{condition_key}' was not found in adata.obs.\n"
            f"Available columns: {list(adata.obs.columns)}"
        )

    condition_values = (
        adata.obs[condition_key]
        .astype(str)
        .to_numpy()
    )

    mask0 = (
        condition_values
        == str(cond0)
    )

    mask1 = (
        condition_values
        == str(cond1)
    )

    if mask0.sum() == 0:
        raise ValueError(
            f"No cells were found for "
            f"{condition_key} == '{cond0}'."
        )

    if mask1.sum() == 0:
        raise ValueError(
            f"No cells were found for "
            f"{condition_key} == '{cond1}'."
        )

    print(
        f"[LFC] {cond0} cells: {int(mask0.sum())}"
    )

    print(
        f"[LFC] {cond1} cells: {int(mask1.sum())}"
    )

    var_names = np.asarray(
        adata.var_names,
        dtype=str,
    )

    exact_lookup = {
        gene: index
        for index, gene in enumerate(var_names)
    }

    uppercase_lookup = {}

    for index, gene in enumerate(var_names):
        gene_upper = normalize_gene_name(gene)

        if gene_upper not in uppercase_lookup:
            uppercase_lookup[gene_upper] = index

    genes = np.asarray(
        genes,
        dtype=str,
    )

    matched_summary_positions = []
    matched_adata_indices = []
    missing_genes = []

    for summary_position, gene in enumerate(genes):
        adata_index = exact_lookup.get(
            gene
        )

        if adata_index is None:
            adata_index = uppercase_lookup.get(
                normalize_gene_name(gene)
            )

        if adata_index is None:
            missing_genes.append(
                gene
            )
            continue

        matched_summary_positions.append(
            summary_position
        )

        matched_adata_indices.append(
            adata_index
        )

    log2fc = np.full(
        len(genes),
        np.nan,
        dtype=np.float64,
    )

    mean0 = np.full(
        len(genes),
        np.nan,
        dtype=np.float64,
    )

    mean1 = np.full(
        len(genes),
        np.nan,
        dtype=np.float64,
    )

    if len(matched_adata_indices) == 0:
        raise ValueError(
            "None of the posterior-summary genes were found "
            "in the H5AD file."
        )

    # Load only the matched genes.
    X0 = adata[
        mask0,
        matched_adata_indices,
    ].X

    X1 = adata[
        mask1,
        matched_adata_indices,
    ].X

    matched_mean0 = sparse_or_dense_mean(
        X0
    )

    matched_mean1 = sparse_or_dense_mean(
        X1
    )

    matched_summary_positions = np.asarray(
        matched_summary_positions,
        dtype=int,
    )

    mean0[
        matched_summary_positions
    ] = matched_mean0

    mean1[
        matched_summary_positions
    ] = matched_mean1

    numerator = (
        matched_mean1
        + float(pseudocount)
    )

    denominator = (
        matched_mean0
        + float(pseudocount)
    )

    valid_ratio = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator > 0)
        & (denominator > 0)
    )

    matched_log2fc = np.full(
        len(matched_mean0),
        np.nan,
        dtype=np.float64,
    )

    matched_log2fc[
        valid_ratio
    ] = np.log2(
        numerator[valid_ratio]
        / denominator[valid_ratio]
    )

    log2fc[
        matched_summary_positions
    ] = matched_log2fc

    if missing_genes:
        print(
            f"[LFC warning] {len(missing_genes)} genes from "
            "the posterior summary were absent from the H5AD file."
        )

        print(
            "[LFC warning] Missing genes:",
            ", ".join(
                missing_genes[:30]
            ),
        )

        if len(missing_genes) > 30:
            print(
                f"[LFC warning] ... and "
                f"{len(missing_genes) - 30} more."
            )

    return (
        log2fc,
        mean0,
        mean1,
    )


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
    Place highlighted labels in a dedicated grid above the
    entire curve.

    Labels are sorted by CIPHER rank and placed left-to-right,
    minimizing leader-line crossings.
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

    sorted_indices = label_indices[
        np.argsort(
            x[label_indices],
            kind="mergesort",
        )
    ]

    n_labels = len(
        sorted_indices
    )

    n_rows = max(
        1,
        min(
            int(n_rows),
            n_labels,
        ),
    )

    n_columns = int(
        math.ceil(
            n_labels / n_rows
        )
    )

    plot_width = (
        x_max - x_min
    )

    left = (
        x_min
        + 0.14 * plot_width
    )

    right = (
        x_max
        - 0.14 * plot_width
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

    first_row_y = (
        curve_ymax
        + band_gap
    )

    rows = []

    for order_position, idx in enumerate(
        sorted_indices
    ):
        column = (
            order_position // n_rows
        )

        row = (
            order_position % n_rows
        )

        label_x = (
            column_positions[column]
        )

        label_y = (
            first_row_y
            + row * row_spacing
        )

        rows.append(
            {
                "index": int(idx),
                "gene": str(gene_names[idx]),
                "point_x": float(x[idx]),
                "point_y": float(y[idx]),
                "label_x": float(label_x),
                "label_y": float(label_y),
                "label_row": int(row + 1),
                "label_column": int(column + 1),
            }
        )

    return pd.DataFrame(
        rows
    )


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
):
    """
    Draw bold labels with clear arrows for FN1 and IGFBP7.
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

    indices = [
        int(i)
        for i in indices
        if 0 <= int(i) < len(x)
        and np.isfinite(x[int(i)])
        and np.isfinite(y[int(i)])
        and y[int(i)] > 0
    ]

    if not indices:
        return

    text_outline = [
        pe.withStroke(
            linewidth=4.5,
            foreground="white",
        )
    ]

    indices = sorted(
        indices,
        key=lambda i: x[i],
    )

    for k, i in enumerate(indices):

        # Separate the two labels vertically.
        dy = 42 if k % 2 == 0 else -42

        # Keep labels inside the figure when the gene is near
        # the right side of the ranked curve.
        if x[i] > 0.72 * np.nanmax(x):
            dx = -65
            ha = "right"
        else:
            dx = 65
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
            fontsize=LABEL_FONTSIZE,
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
                alpha=0.95,
                shrinkA=4,
                shrinkB=8,
                mutation_scale=13,
                connectionstyle="arc3,rad=0.08",
            ),
        )


def normalize_gene_list(genes):
    out = []
    seen = set()
    for g in genes:
        gg = str(g).strip().upper()
        if gg and gg not in seen:
            out.append(gg)
            seen.add(gg)
    return out


def stable_cho_factor(
    matrix,
    matrix_name,
    max_attempts=8,
):
    """
    Compute a Cholesky factor, adding progressively larger diagonal
    jitter only if numerical factorization initially fails.
    """

    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    matrix = 0.5 * (
        matrix + matrix.T
    )

    scale = max(
        float(np.mean(np.diag(matrix))),
        1e-12,
    )

    last_error = None

    for attempt in range(max_attempts):
        if attempt == 0:
            jitter = 0.0
        else:
            jitter = (
                scale
                * 10.0 ** (-12 + attempt)
            )

        try:
            adjusted = matrix + (
                jitter
                * np.eye(matrix.shape[0])
            )

            factor = cho_factor(
                adjusted,
                lower=True,
                check_finite=False,
            )

            if jitter > 0:
                print(
                    f"[numerics] Added jitter={jitter:.3e} "
                    f"to {matrix_name}"
                )

            return factor

        except Exception as error:
            last_error = error

    raise np.linalg.LinAlgError(
        f"Could not factorize {matrix_name}. "
        f"Last error: {repr(last_error)}"
    )


def make_posterior_summary(
    mu,
    std,
    gene_names,
    delta_x,
    effect_threshold=None,
):
    """
    Create the saved posterior-summary table.
    """

    mu = np.asarray(
        mu,
        dtype=np.float64,
    )

    std = np.asarray(
        std,
        dtype=np.float64,
    )

    gene_names = np.asarray(
        gene_names,
        dtype=str,
    )

    delta_x = np.asarray(
        delta_x,
        dtype=np.float64,
    )

    if effect_threshold is None:
        effect_threshold = float(
            np.median(std)
        )

    safe_std = std + 1e-12

    z_score = mu / safe_std

    upper = (
        effect_threshold - mu
    ) / safe_std

    lower = (
        -effect_threshold - mu
    ) / safe_std

    pip = 1.0 - (
        norm.cdf(upper)
        - norm.cdf(lower)
    )

    probability_positive = (
        1.0
        - norm.cdf(
            (0.0 - mu) / safe_std
        )
    )

    probability_negative = norm.cdf(
        (0.0 - mu) / safe_std
    )

    sign_confidence = np.maximum(
        probability_positive,
        probability_negative,
    )

    ci95_lo = mu - 1.96 * std
    ci95_hi = mu + 1.96 * std

    zero_excluded = (
        (ci95_lo > 0)
        | (ci95_hi < 0)
    ).astype(int)

    summary = pd.DataFrame(
        {
            "gene": gene_names,
            "mu": mu,
            "std": std,
            "z": z_score,
            "abs_z": np.abs(z_score),
            "pip": pip,
            "p_pos": probability_positive,
            "p_neg": probability_negative,
            "sign_conf": sign_confidence,
            "ci95_lo": ci95_lo,
            "ci95_hi": ci95_hi,
            "zero_excluded": zero_excluded,
            "delta_x": delta_x,
        }
    )

    summary = (
        summary
        .sort_values(
            by=[
                "pip",
                "abs_z",
                "gene",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return summary, effect_threshold


def _gene_key(s):
    return str(s).upper()


def load_posterior_summary():
    """
    Returns a dataframe with at least:
      gene, mu
    """
    if os.path.exists(POSTERIOR_SUMMARY_PATH):
        print(f"[load] found posterior summary: {POSTERIOR_SUMMARY_PATH}")
        df = pd.read_csv(POSTERIOR_SUMMARY_PATH, sep="\t")
        return df

    if os.path.exists(POSTERIOR_MU_PATH) and os.path.exists(SELECTED_GENES_PATH):
        print("[load] found posterior_mu.npy and selected_genes.npy")
        mu = np.load(POSTERIOR_MU_PATH)
        genes = np.load(SELECTED_GENES_PATH, allow_pickle=True).astype(str)

        df = pd.DataFrame({
            "gene": genes,
            "mu": mu,
        })
        return df

    return None


def load_de_table():
    """
    Prefer selected_de.tsv because posterior was computed on selected genes.
    Fallback to all_genes_de.tsv.
    Returns dataframe with at least:
      gene, log2fc
    """
    if os.path.exists(SELECTED_DE_PATH):
        print(f"[load] found selected DE table: {SELECTED_DE_PATH}")
        return pd.read_csv(SELECTED_DE_PATH, sep="\t")

    if os.path.exists(ALL_GENES_DE_PATH):
        print(f"[load] found all-gene DE table: {ALL_GENES_DE_PATH}")
        return pd.read_csv(ALL_GENES_DE_PATH, sep="\t")

    return None


def maybe_rerun_pipeline():
    """
    Rerun only when saved data needed for these plots are missing.
    Requires run_pipeline(...) already defined in the notebook/session.
    """
    if "run_pipeline" not in globals():
        raise RuntimeError(
            "Needed saved posterior/DE files are missing, and run_pipeline(...) "
            "is not defined in this notebook/session.\n\n"
            "Run the original full pipeline code first, or paste/run the original "
            "run_pipeline definition, then rerun this plotting block."
        )

    print("[saved data incomplete] rerunning run_pipeline(...)")

    results = run_pipeline(
        h5ad_path=H5AD_PATH,
        outdir=OUTDIR,

        condition_key="Condition",
        cond0="Naive",
        cond1="Resistant",

        genes_to_highlight=["IGFBP7", "FN1"],

        # DE gene set
        top_n_de=2000,
        fdr_alpha=0.05,
        min_abs_log2fc=0.01,
        min_abs_delta=0.02,
        rank_by="abs_t",
        fill_to_top_n=True,

        # gene filtering
        drop_housekeeping=True,
        min_cells_frac=0.01,
        min_expr=0.01,
        min_mean=0.001,
        max_mean=np.inf,
        max_var_quantile=1.0,
        filter_subsample_cells=0,

        # DE logFC only; posterior uses delta_x, not logFC
        logfc_pseudocount=1.0,

        # covariance / posterior
        Sigma_shrinkage=1e-6,
        H_shrinkage=1e-6,
        H_ridge=1e-6,
        H_mode="naive",
        tau2=1e-6,
        effect_threshold=None,

        top_k_plot=20,
        seed=0,
    )

    return results


def standardize_lfc_columns(de):
    """
    Make sure DE table has log2fc and abs_log2fc.
    """
    de = de.copy()

    possible_lfc_cols = [
        "log2fc",
        "log2FC",
        "lfc",
        "LFC",
        "logFC",
        "log_fold_change",
        "logfoldchange",
    ]

    lfc_col = None
    for c in possible_lfc_cols:
        if c in de.columns:
            lfc_col = c
            break

    if lfc_col is None:
        raise KeyError(
            "Could not find an LFC column in the DE table. "
            f"Available columns: {list(de.columns)}"
        )

    de["log2fc"] = pd.to_numeric(de[lfc_col], errors="coerce")
    de["abs_log2fc"] = np.abs(de["log2fc"].values)

    return de


def add_gene_ranks(df):
    """
    Adds signed and absolute posterior/LFC ranks.
    """
    df = df.copy()

    df["abs_mu"] = np.abs(df["mu"].values)
    df["abs_log2fc"] = np.abs(df["log2fc"].values)

    # Signed ranks: largest positive effect gets rank 1.
    df["rank_by_mu"] = df["mu"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    df["rank_by_log2fc"] = df["log2fc"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    # Absolute ranks: largest magnitude gets rank 1.
    df["rank_by_abs_mu"] = df["abs_mu"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    df["rank_by_abs_log2fc"] = df["abs_log2fc"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    return df


def annotate_highlights(ax, df, x_col, y_col, genes, fontsize=13):
    """
    Scatter + label highlighted genes if present and finite.
    """
    for gene in genes:
        hit = df.loc[df["gene"].str.upper() == gene.upper()]
        if len(hit) == 0:
            print(f"[highlight] {gene}: not found")
            continue

        row = hit.iloc[0]

        x = row[x_col]
        y = row[y_col]

        if not np.isfinite(x) or not np.isfinite(y):
            print(f"[highlight] {gene}: non-finite x/y for {x_col}, {y_col}")
            continue

        ax.scatter(
            x,
            y,
            s=HIGHLIGHT_SIZE,
            marker="*",
            edgecolor="black",
            linewidth=0.9,
            zorder=20,
        )

        ax.text(
            x,
            y,
            f" {gene}",
            fontsize=fontsize,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=21,
        )


def get_mu_rank_order(mu):
    """
    Rank genes by posterior mean mu descending.
    Rank 1 has largest mu.
    """
    mu = np.asarray(mu, dtype=np.float64)
    return np.argsort(-mu)


def remap_top_idx_to_ranked_positions(order, top_idx):
    """
    top_idx are original positions.
    order maps ranked positions -> original positions.
    Return ranked positions corresponding to top_idx.
    """
    top_idx = np.asarray(top_idx, dtype=int)
    top_mask = np.zeros(len(order), dtype=bool)
    top_mask[top_idx] = True
    ranked_positions = np.where(top_mask[order])[0]
    return ranked_positions


def label_top_points(ax, xs, ys, labels, idxs, fontsize=8):
    for i in idxs:
        if np.isfinite(ys[i]):
            ax.text(xs[i], ys[i], str(labels[i]), fontsize=fontsize)


def highlight_genes_ranked(ax, gene_names_ranked, genes, yvals, xvals, fontsize=11):
    genes = set(normalize_gene_list(genes))

    for j, g in enumerate(np.asarray(gene_names_ranked, dtype=str)):
        if g.upper() in genes and np.isfinite(yvals[j]):
            ax.scatter(
                [xvals[j]],
                [yvals[j]],
                s=150,
                marker="*",
                edgecolor="black",
                linewidth=0.8,
                zorder=10,
            )
            ax.text(
                xvals[j],
                yvals[j],
                f" {g}",
                fontsize=fontsize,
                fontweight="bold",
                va="center",
                zorder=11,
            )


def save_fig_all_formats(fig, outbase):
    out_png = outbase + ".png"
    out_pdf = outbase + ".pdf"
    out_svg = outbase + ".svg"

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")

    print(f"[saved] {out_png}")
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_svg}")


def plot_log1p_mu_vs_mu_rank(
    mu,
    gene_names,
    genes_to_highlight,
    outbase,
    top_k_label=20,
):
    """
    x-axis = genes ranked by posterior mean mu descending.
    y-axis = log(1 + posterior mean mu).
    """
    mu = np.asarray(mu, dtype=np.float64)
    gene_names = np.asarray(gene_names, dtype=str)

    order = get_mu_rank_order(mu)

    mu_ranked = mu[order]
    gene_names_ranked = gene_names[order]

    x = np.arange(len(mu_ranked))

    valid = np.isfinite(mu_ranked) & (mu_ranked > -1)

    y = np.full_like(mu_ranked, np.nan, dtype=np.float64)
    y[valid] = np.log1p(mu_ranked[valid])

    if np.sum(~valid) > 0:
        print(
            f"[plot] WARNING: {(~valid).sum()} genes have mu <= -1 or nonfinite mu; "
            "log(1 + mu) is undefined and skipped."
        )

    top_idx_original = np.argsort(-np.abs(mu))[:min(top_k_label, len(mu))]
    top_ranked_idx = remap_top_idx_to_ranked_positions(order, top_idx_original)
    top_ranked_idx = np.asarray([i for i in top_ranked_idx if valid[i]], dtype=int)

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(
        x[valid],
        y[valid],
        lw=1,
        alpha=0.9,
    )

    ax.scatter(
        x[valid],
        y[valid],
        s=12,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )

    ax.scatter(
        x[top_ranked_idx],
        y[top_ranked_idx],
        s=28,
        zorder=5,
    )

    label_top_points(
        ax,
        x,
        y,
        gene_names_ranked,
        top_ranked_idx,
        fontsize=LABEL_FONTSIZE,
    )

    highlight_genes_ranked(
        ax,
        gene_names_ranked,
        genes_to_highlight,
        y,
        x,
        fontsize=HIGHLIGHT_FONTSIZE,
    )

    ax.axhline(0, color="black", lw=1, alpha=0.6)

    ax.set_xlabel("Rank by posterior mean μ")
    ax.set_ylabel("log(1 + posterior mean μ)")
    ax.set_title("Posterior score: log(1 + μ) vs genes ranked by μ")

    plt.tight_layout()
    save_fig_all_formats(fig, outbase)
    plt.show()

    ranked_df = pd.DataFrame({
        "mu_rank": np.arange(1, len(mu_ranked) + 1),
        "gene": gene_names_ranked,
        "mu": mu_ranked,
        "log1p_mu": y,
        "valid_log1p": valid,
    })

    table_path = outbase + "_table.tsv"
    ranked_df.to_csv(table_path, sep="\t", index=False)
    print(f"[saved] {table_path}")

    return ranked_df


def plot_absz_vs_mu_rank(
    mu,
    std=None,
    z=None,
    gene_names=None,
    genes_to_highlight=None,
    outbase=None,
    top_k_label=20,
):
    """
    x-axis = genes ranked by posterior mean mu descending.
    y-axis = |posterior z|.
    """
    mu = np.asarray(mu, dtype=np.float64)
    gene_names = np.asarray(gene_names, dtype=str)

    if z is None:
        if std is None:
            raise ValueError("Need either z or std to plot |posterior z|.")
        std = np.asarray(std, dtype=np.float64)
        z = mu / (std + 1e-12)
    else:
        z = np.asarray(z, dtype=np.float64)

    absz = np.abs(z)

    order = get_mu_rank_order(mu)

    mu_ranked = mu[order]
    z_ranked = z[order]
    absz_ranked = absz[order]
    gene_names_ranked = gene_names[order]

    x = np.arange(len(mu_ranked))

    valid = np.isfinite(absz_ranked)

    top_idx_original = np.argsort(-absz)[:min(top_k_label, len(absz))]
    top_ranked_idx = remap_top_idx_to_ranked_positions(order, top_idx_original)
    top_ranked_idx = np.asarray([i for i in top_ranked_idx if valid[i]], dtype=int)

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(
        x[valid],
        absz_ranked[valid],
        lw=1,
        alpha=0.9,
    )

    ax.scatter(
        x[valid],
        absz_ranked[valid],
        s=12,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )

    ax.scatter(
        x[top_ranked_idx],
        absz_ranked[top_ranked_idx],
        s=28,
        zorder=5,
    )

    label_top_points(
        ax,
        x,
        absz_ranked,
        gene_names_ranked,
        top_ranked_idx,
        fontsize=LABEL_FONTSIZE,
    )

    highlight_genes_ranked(
        ax,
        gene_names_ranked,
        genes_to_highlight,
        absz_ranked,
        x,
        fontsize=HIGHLIGHT_FONTSIZE,
    )

    ax.set_xlabel("Rank by posterior mean μ")
    ax.set_ylabel("|posterior z|")
    ax.set_title("Posterior |z|-score vs genes ranked by μ")

    plt.tight_layout()
    save_fig_all_formats(fig, outbase)
    plt.show()

    ranked_df = pd.DataFrame({
        "mu_rank": np.arange(1, len(mu_ranked) + 1),
        "gene": gene_names_ranked,
        "mu": mu_ranked,
        "z": z_ranked,
        "abs_z": absz_ranked,
    })

    table_path = outbase + "_table.tsv"
    ranked_df.to_csv(table_path, sep="\t", index=False)
    print(f"[saved] {table_path}")

    return ranked_df


def import_umap():
    try:
        from umap import UMAP
        return UMAP
    except Exception as e1:
        try:
            from umap.umap_ import UMAP
            return UMAP
        except Exception as e2:
            raise ImportError(
                "Could not import UMAP from umap-learn.\n\n"
                "Run:\n"
                "    pip uninstall -y umap\n"
                "    pip install umap-learn\n\n"
                f"First error: {repr(e1)}\n"
                f"Second error: {repr(e2)}"
            )


def get_gene_idx(var_names, gene):
    names = np.asarray(var_names, dtype=str)
    hits = np.where(np.char.upper(names) == gene.upper())[0]
    if len(hits) == 0:
        return None
    return int(hits[0])


def get_gene_expr(X, gene_idx):
    x = X[:, gene_idx]
    if issparse(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def sparse_mean_var(X):
    if issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = mean_sq - mean**2
        var = np.maximum(var, 0)
    else:
        mean = np.asarray(X.mean(axis=0)).ravel()
        var = np.asarray(X.var(axis=0)).ravel()
    return mean, var


def looks_like_raw_counts(X, n_cells_check=2000):
    n = min(X.shape[0], n_cells_check)

    if issparse(X):
        vals = X[:n].data
    else:
        vals = np.asarray(X[:n]).ravel()

    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]

    if len(vals) == 0:
        return False

    vals_sample = vals[:min(len(vals), 200000)]
    integer_like = np.mean(np.abs(vals_sample - np.round(vals_sample)) < 1e-6)
    q99 = np.percentile(vals_sample, 99)

    return (integer_like > 0.8) and (q99 > 20)


def normalize_total_log1p(X, target_sum=1e4):
    if issparse(X):
        X = X.tocsr(copy=True)
        cell_sums = np.asarray(X.sum(axis=1)).ravel()

        scale = np.divide(
            target_sum,
            cell_sums,
            out=np.zeros_like(cell_sums, dtype=float),
            where=cell_sums > 0,
        )

        X = diags(scale) @ X
        X.data = np.log1p(X.data)

        return X.tocsr()

    X = np.asarray(X, dtype=np.float32).copy()
    cell_sums = X.sum(axis=1)

    scale = np.divide(
        target_sum,
        cell_sums,
        out=np.zeros_like(cell_sums, dtype=float),
        where=cell_sums > 0,
    )

    X *= scale[:, None]
    X = np.log1p(X)

    return X


def select_hvgs_by_variance(X, n_hvg=3000):
    mean, var = sparse_mean_var(X)

    valid = np.isfinite(mean) & np.isfinite(var) & (mean > 0)
    score = np.zeros_like(var)
    score[valid] = var[valid]

    n_hvg = min(n_hvg, X.shape[1])
    hvg_idx = np.argsort(-score)[:n_hvg]

    return np.asarray(hvg_idx, dtype=int), mean, var


def compute_pca_embedding(X, n_pcs=50, seed=0):
    n_comps = min(n_pcs, X.shape[0] - 1, X.shape[1] - 1)

    if n_comps < 2:
        raise ValueError(f"Too few cells/genes for PCA: X shape={X.shape}")

    if issparse(X):
        print("[PCA] sparse input: using TruncatedSVD")
        model = TruncatedSVD(n_components=n_comps, random_state=seed)
        Z = model.fit_transform(X)
        Z = StandardScaler().fit_transform(Z)
    else:
        print("[PCA] dense input: using PCA")
        Xd = np.asarray(X, dtype=np.float32)
        Xd = StandardScaler().fit_transform(Xd)
        model = PCA(n_components=n_comps, random_state=seed)
        Z = model.fit_transform(Xd)

    return Z


def sparse_or_dense_mean_var(X):
    """
    Returns column-wise mean and unbiased variance.
    Works for sparse or dense X.
    """
    n = X.shape[0]

    if issparse(X):
        X = X.tocsr()
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var_pop = mean_sq - mean**2
        var_pop = np.maximum(var_pop, 0)
        var = var_pop * n / max(n - 1, 1)
    else:
        X = np.asarray(X)
        mean = X.mean(axis=0)
        var = X.var(axis=0, ddof=1)

    return mean.astype(float), var.astype(float)


def sparse_or_dense_frac_on(X, min_expr=0.0):
    """
    Fraction of cells with expression > min_expr or >= min_expr.
    For min_expr <= 0, uses X > 0.
    """
    n = X.shape[0]

    if issparse(X):
        X = X.tocsr()
        if min_expr <= 0:
            frac = np.asarray((X > 0).mean(axis=0)).ravel()
        else:
            frac = np.asarray((X >= min_expr).mean(axis=0)).ravel()
    else:
        X = np.asarray(X)
        if min_expr <= 0:
            frac = np.mean(X > 0, axis=0)
        else:
            frac = np.mean(X >= min_expr, axis=0)

    return frac.astype(float)


def welch_de_from_moments(mean0, var0, n0, mean1, var1, n1, eps=1e-300):
    """
    Welch t-test from means/variances, vectorized.
    Returns t_stat and two-sided p-value.
    """
    se2 = var0 / max(n0, 1) + var1 / max(n1, 1)
    se = np.sqrt(np.maximum(se2, eps))

    t_stat = (mean1 - mean0) / se

    # Welch-Satterthwaite df
    a = var0 / max(n0, 1)
    b = var1 / max(n1, 1)

    denom = (a**2) / max(n0 - 1, 1) + (b**2) / max(n1 - 1, 1)
    df = (a + b)**2 / np.maximum(denom, eps)

    pval = 2.0 * student_t.sf(np.abs(t_stat), df)
    pval = np.nan_to_num(pval, nan=1.0, posinf=1.0, neginf=1.0)

    return t_stat, pval


def label_gene(ax, row, x_col="log2fc", y_col="neglog10_p", fontsize=9):
    ax.text(
        row[x_col],
        row[y_col],
        f" {row['gene']}",
        fontsize=fontsize,
        ha="left",
        va="center",
    )
