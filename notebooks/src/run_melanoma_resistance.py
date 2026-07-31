"""Notebook-only run/orchestration module for Fig M7 / Fig S17 — melanoma
naive-vs-resistant analytic Gaussian posterior (CIPHER score) resistance drivers.

Each function is one main-flow cell of ``notebooks/suppl/melanoma_resistance_figM7_S17.ipynb``,
relocated here (statements dedented from the cell, then indented one level into the
def body — same variables, same plt/savefig calls, same logic). Functions read
notebook config (MEL_H5AD, BC50_H5AD, FOUR_COND_H5AD, BASE_OUT, DATA_DIR, SUPPL,
OUTDIR, ...) as MODULE GLOBALS, injected at runtime by the notebook's injection cell.
Every main-flow cell is self-contained (redefines its own local config and reloads its
inputs from disk), so no cross-section module state is needed.

NOT part of the installable ``cipher`` package — a notebook-only helper for
reproducing the supplementary figures.
"""
from src.suppl_mel_resistance import *

# --- library imports the extracted cell bodies use (mirrors the cluster module) ---
import os, re, glob, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Patch
from matplotlib import gridspec
from scipy.sparse import issparse, csr_matrix, diags
from scipy.stats import ttest_ind, norm
try:
    from statsmodels.stats.multitest import multipletests
except Exception:  # pragma: no cover
    multipletests = None
try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None
try:
    import scanpy as sc
except Exception:  # pragma: no cover
    sc = None
# --- end imports ---

def fit_analytic_posterior_pipeline():
    # ============================================================
    # ANALYTIC GAUSSIAN POSTERIOR PIPELINE
    # MULTI-GENE HIGHLIGHT VERSION:
    #   - highlights FN1 and IGFBP7
    #   - robust h5ad loading
    #   - selects FDR genes first, then fills to top_n_de by abs_t
    #   - filters MTRNR/RPL/RPS/MT/HSP/EIF/MALAT1 junk genes
    # ============================================================


    # ============================================================
    # I/O
    # ============================================================


    # ============================================================
    # filtering
    # ============================================================


    # ============================================================
    # DE
    # ============================================================


    # ============================================================
    # covariance / posterior
    # ============================================================


    # ============================================================
    # plotting helpers
    # ============================================================


    # ============================================================
    # main
    # ============================================================


    # ============================================================
    # RUN
    # ============================================================

    results = run_pipeline(
            h5ad_path=MEL_H5AD,
            # h5ad_path=BC50_H5AD,
            outdir=os.path.join(BASE_OUT, "analytic_gaussian_FN1_IGFBP7_stable_diag"),

            condition_key="Condition",
            cond0="Naive",
            cond1="Resistant",

            genes_to_highlight=[ "IGFBP7", "FN1"],

            # DE gene set
            top_n_de=2000,
            fdr_alpha=0.05,
            min_abs_log2fc=.01,
            min_abs_delta=0.02,
            rank_by="abs_t",
            fill_to_top_n=True,

            # gene filtering
            drop_housekeeping=True,
            min_cells_frac=0.01,
            min_expr=0.01,
            min_mean=0.001,
            max_mean=np.inf,
            max_var_quantile=1.,
            filter_subsample_cells=0,

            # DE logFC only; posterior uses delta_x, not logFC
            logfc_pseudocount=1.,

            # covariance / posterior
            Sigma_shrinkage=1e-6,
            H_shrinkage=1e-6,
            H_ridge=1e-6,
            H_mode="naive",
            tau2= 1e-6,
            effect_threshold=None,

            top_k_plot=20,
            seed=0,
        )


def plot_log1p_mu_vs_rank():
    # ============================================================
    # LOAD SAVED POSTERIOR OR RERUN, THEN PLOT log(1 + mu) vs RANK
    # Highlights FN1 and IGFBP7
    # Saves PNG, PDF, SVG
    # ============================================================


    # ============================================================
    # CONFIG: match your run_pipeline settings
    # ============================================================

    H5AD_PATH = MEL_H5AD
    # H5AD_PATH = BC50_H5AD

    OUTDIR = os.path.join(BASE_OUT, "analytic_gaussian_FN1_IGFBP7_stable_diag")
    os.makedirs(OUTDIR, exist_ok=True)

    HIGHLIGHT_GENES = ["FN1", "IGFBP7"]

    DPI = 300
    POINT_SIZE = 18
    HIGHLIGHT_SIZE = 220

    POSTERIOR_SUMMARY_PATH = os.path.join(OUTDIR, "posterior_summary.tsv")
    POSTERIOR_MU_PATH = os.path.join(OUTDIR, "posterior_mu.npy")
    SELECTED_GENES_PATH = os.path.join(OUTDIR, "selected_genes.npy")

    # ============================================================
    # LOAD SAVED RESULTS IF POSSIBLE
    # ============================================================

    if os.path.exists(POSTERIOR_SUMMARY_PATH):
        print(f"[load] found saved posterior summary: {POSTERIOR_SUMMARY_PATH}")
        summary = pd.read_csv(POSTERIOR_SUMMARY_PATH, sep="\t")

    elif os.path.exists(POSTERIOR_MU_PATH) and os.path.exists(SELECTED_GENES_PATH):
        print("[load] found saved posterior_mu.npy and selected_genes.npy")
        mu = np.load(POSTERIOR_MU_PATH)
        genes = np.load(SELECTED_GENES_PATH, allow_pickle=True).astype(str)

        summary = pd.DataFrame({
            "gene": genes,
            "mu": mu,
        })

    else:
        print("[saved data not found] rerunning run_pipeline(...)")

        if "run_pipeline" not in globals():
            raise RuntimeError(
                "run_pipeline(...) is not defined in this notebook/session.\n"
                "Run the full pipeline code first, then run this plotting block."
            )

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

        summary = results["summary"].copy()

    # ============================================================
    # CHECK COLUMNS
    # ============================================================

    if "gene" not in summary.columns:
        raise KeyError("summary must contain a 'gene' column.")

    if "mu" not in summary.columns:
        raise KeyError("summary must contain a 'mu' column.")

    summary = summary.copy()
    summary["gene"] = summary["gene"].astype(str)
    summary["mu"] = pd.to_numeric(summary["mu"], errors="coerce")

    summary = summary[np.isfinite(summary["mu"].values)].copy()

    if len(summary) == 0:
        raise ValueError("No finite mu values found.")

    # ============================================================
    # RANK BY POSTERIOR MU
    # ============================================================

    summary = summary.sort_values("mu", ascending=False).reset_index(drop=True)
    summary["rank_by_mu"] = np.arange(1, len(summary) + 1)

    # Direct requested transform: log(1 + mu)
    # This is undefined for mu <= -1.
    summary["log1p_mu"] = np.nan
    valid_log = summary["mu"].values > -1.0
    summary.loc[valid_log, "log1p_mu"] = np.log1p(summary.loc[valid_log, "mu"].values)

    n_invalid = int((~valid_log).sum())
    if n_invalid > 0:
        print(
            f"[warning] {n_invalid} genes have mu <= -1, so log(1 + mu) is undefined. "
            "They will be omitted from the plotted y-values."
        )

    # Also save a signed version in the table in case useful later
    summary["signed_log1p_abs_mu"] = np.sign(summary["mu"]) * np.log1p(np.abs(summary["mu"]))

    out_table = os.path.join(OUTDIR, "posterior_mu_rank_log1p.tsv")
    summary.to_csv(out_table, sep="\t", index=False)
    print(f"[saved table] {out_table}")

    # ============================================================
    # PRINT HIGHLIGHT GENE STATUS
    # ============================================================

    print("\n[highlight genes]")
    for gene in HIGHLIGHT_GENES:
        hit = summary.loc[summary["gene"].str.upper() == gene.upper()]
        if len(hit) == 0:
            print(f"{gene}: not found")
        else:
            print(hit[["gene", "rank_by_mu", "mu", "log1p_mu", "signed_log1p_abs_mu"]].to_string(index=False))

    # ============================================================
    # PLOT log(1 + mu) vs RANK
    # ============================================================

    plot_df = summary[np.isfinite(summary["log1p_mu"].values)].copy()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(
        plot_df["rank_by_mu"],
        plot_df["log1p_mu"],
        s=POINT_SIZE,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )

    ax.axhline(0, lw=1)
    ax.set_xlabel("Rank by posterior mean effect μ")
    ax.set_ylabel("log(1 + μ)")
    ax.set_title("Posterior effect rank plot")

    # Highlight FN1 and IGFBP7
    for gene in HIGHLIGHT_GENES:
        hit = summary.loc[summary["gene"].str.upper() == gene.upper()]

        if len(hit) == 0:
            continue

        row = hit.iloc[0]

        if not np.isfinite(row["log1p_mu"]):
            print(f"[warning] {gene} has mu={row['mu']:.4g}, so log(1+mu) is undefined and cannot be plotted.")
            continue

        ax.scatter(
            row["rank_by_mu"],
            row["log1p_mu"],
            s=HIGHLIGHT_SIZE,
            marker="*",
            edgecolor="black",
            linewidth=0.8,
            zorder=20,
        )

        ax.text(
            row["rank_by_mu"],
            row["log1p_mu"],
            f" {gene}",
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=21,
        )

    plt.tight_layout()

    out_png = os.path.join(OUTDIR, "posterior_log1p_mu_vs_rank_FN1_IGFBP7.png")
    out_pdf = os.path.join(OUTDIR, "posterior_log1p_mu_vs_rank_FN1_IGFBP7.pdf")
    out_svg = os.path.join(OUTDIR, "posterior_log1p_mu_vs_rank_FN1_IGFBP7.svg")

    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_svg, format="svg", bbox_inches="tight")

    plt.show()

    print("\n[DONE]")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")

    # ============================================================
    # OPTIONAL: ALSO PLOT SIGNED log(1 + |mu|)
    # This works even when mu is negative.
    # ============================================================

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(
        summary["rank_by_mu"],
        summary["signed_log1p_abs_mu"],
        s=POINT_SIZE,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )

    ax.axhline(0, lw=1)
    ax.set_xlabel("Rank by posterior mean effect μ")
    ax.set_ylabel("sign(μ) log(1 + |μ|)")
    ax.set_title("Signed posterior effect rank plot")

    for gene in HIGHLIGHT_GENES:
        hit = summary.loc[summary["gene"].str.upper() == gene.upper()]

        if len(hit) == 0:
            continue

        row = hit.iloc[0]

        ax.scatter(
            row["rank_by_mu"],
            row["signed_log1p_abs_mu"],
            s=HIGHLIGHT_SIZE,
            marker="*",
            edgecolor="black",
            linewidth=0.8,
            zorder=20,
        )

        ax.text(
            row["rank_by_mu"],
            row["signed_log1p_abs_mu"],
            f" {gene}",
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=21,
        )

    plt.tight_layout()

    out_png2 = os.path.join(OUTDIR, "posterior_signed_log1p_abs_mu_vs_rank_FN1_IGFBP7.png")
    out_pdf2 = os.path.join(OUTDIR, "posterior_signed_log1p_abs_mu_vs_rank_FN1_IGFBP7.pdf")
    out_svg2 = os.path.join(OUTDIR, "posterior_signed_log1p_abs_mu_vs_rank_FN1_IGFBP7.svg")

    plt.savefig(out_png2, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_pdf2, bbox_inches="tight")
    plt.savefig(out_svg2, format="svg", bbox_inches="tight")

    plt.show()

    print("\n[DONE optional signed plot]")
    print(f"Saved: {out_png2}")
    print(f"Saved: {out_pdf2}")
    print(f"Saved: {out_svg2}")


def plot_cipher_score_vs_rank():
    # ============================================================
    # LOAD SAVED POSTERIOR OR RERUN, THEN PLOT log(1 + |mu|) vs RANK
    #
    # - Defines the CIPHER score as:
    #       CIPHER score = log(1 + |mu|)
    # - Ranks genes by decreasing |mu|
    # - Highlights FN1 and IGFBP7
    # - Saves the ranked table and PNG, PDF, and SVG figures
    # ============================================================


    # ============================================================
    # CONFIG: match the original run_pipeline settings
    # ============================================================

    H5AD_PATH = MEL_H5AD
    # H5AD_PATH = (
    #     "GSE233766_RAW/"
    #     "Xtot_naive_resistant_unbalanced_resistant_BC50_clone_size_gt1.h5ad"
    # )

    OUTDIR = os.path.join(BASE_OUT, "analytic_gaussian_FN1_IGFBP7_stable_diag")
    os.makedirs(OUTDIR, exist_ok=True)

    HIGHLIGHT_GENES = ["FN1", "IGFBP7"]

    DPI = 300
    POINT_SIZE = 18
    HIGHLIGHT_SIZE = 220

    POSTERIOR_SUMMARY_PATH = os.path.join(
        OUTDIR,
        "posterior_summary.tsv",
    )

    POSTERIOR_MU_PATH = os.path.join(
        OUTDIR,
        "posterior_mu.npy",
    )

    SELECTED_GENES_PATH = os.path.join(
        OUTDIR,
        "selected_genes.npy",
    )


    # ============================================================
    # LOAD SAVED RESULTS IF POSSIBLE
    # ============================================================

    if os.path.exists(POSTERIOR_SUMMARY_PATH):

        print(
            f"[load] Found saved posterior summary:\n"
            f"       {POSTERIOR_SUMMARY_PATH}"
        )

        summary = pd.read_csv(
            POSTERIOR_SUMMARY_PATH,
            sep="\t",
        )

    elif (
        os.path.exists(POSTERIOR_MU_PATH)
        and os.path.exists(SELECTED_GENES_PATH)
    ):

        print(
            "[load] Found posterior_mu.npy and selected_genes.npy"
        )

        mu = np.load(POSTERIOR_MU_PATH)

        genes = np.load(
            SELECTED_GENES_PATH,
            allow_pickle=True,
        ).astype(str)

        if len(mu) != len(genes):
            raise ValueError(
                "posterior_mu.npy and selected_genes.npy have "
                f"different lengths: {len(mu)} versus {len(genes)}."
            )

        summary = pd.DataFrame(
            {
                "gene": genes,
                "mu": mu,
            }
        )

    else:

        print(
            "[saved data not found] Rerunning run_pipeline(...)"
        )

        if "run_pipeline" not in globals():
            raise RuntimeError(
                "run_pipeline(...) is not defined in this notebook/session.\n"
                "Run the full pipeline code first, then rerun this block."
            )

        results = run_pipeline(
            h5ad_path=H5AD_PATH,
            outdir=OUTDIR,

            condition_key="Condition",
            cond0="Naive",
            cond1="Resistant",

            genes_to_highlight=[
                "IGFBP7",
                "FN1",
            ],

            # DE gene set
            top_n_de=2000,
            fdr_alpha=0.05,
            min_abs_log2fc=0.01,
            min_abs_delta=0.02,
            rank_by="abs_t",
            fill_to_top_n=True,

            # Gene filtering
            drop_housekeeping=True,
            min_cells_frac=0.01,
            min_expr=0.01,
            min_mean=0.001,
            max_mean=np.inf,
            max_var_quantile=1.0,
            filter_subsample_cells=0,

            # DE logFC only; posterior uses delta_x, not logFC
            logfc_pseudocount=1.0,

            # Covariance and posterior settings
            Sigma_shrinkage=1e-6,
            H_shrinkage=1e-6,
            H_ridge=1e-6,
            H_mode="naive",
            tau2=1e-6,
            effect_threshold=None,

            top_k_plot=20,
            seed=0,
        )

        if "summary" not in results:
            raise KeyError(
                "run_pipeline(...) did not return a 'summary' entry."
            )

        summary = results["summary"].copy()


    # ============================================================
    # VALIDATE AND CLEAN THE POSTERIOR SUMMARY
    # ============================================================

    required_columns = [
        "gene",
        "mu",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in summary.columns
    ]

    if missing_columns:
        raise KeyError(
            "Posterior summary is missing required columns: "
            + ", ".join(missing_columns)
        )

    summary = summary.copy()

    summary["gene"] = (
        summary["gene"]
        .astype(str)
        .str.strip()
    )

    summary["mu"] = pd.to_numeric(
        summary["mu"],
        errors="coerce",
    )

    finite_mask = np.isfinite(
        summary["mu"].to_numpy(dtype=float)
    )

    n_removed = int((~finite_mask).sum())

    if n_removed > 0:
        print(
            f"[warning] Removing {n_removed} rows with non-finite mu."
        )

    summary = (
        summary.loc[finite_mask]
        .copy()
        .reset_index(drop=True)
    )

    if summary.empty:
        raise ValueError(
            "No finite posterior mean values were found."
        )


    # ============================================================
    # COMPUTE log(1 + |mu|) AND RANK BY |mu|
    # ============================================================

    summary["abs_mu"] = np.abs(
        summary["mu"].to_numpy(dtype=float)
    )

    summary["cipher_score"] = np.log1p(
        summary["abs_mu"].to_numpy(dtype=float)
    )

    summary = (
        summary
        .sort_values(
            by=[
                "abs_mu",
                "gene",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    summary["cipher_rank"] = np.arange(
        1,
        len(summary) + 1,
        dtype=int,
    )

    # Convenient aliases with explicit names
    summary["log1p_abs_mu"] = summary["cipher_score"]
    summary["rank_by_abs_mu"] = summary["cipher_rank"]


    # ============================================================
    # SAVE FULL RANKED TABLE
    # ============================================================

    preferred_columns = [
        "gene",
        "mu",
        "abs_mu",
        "cipher_score",
        "cipher_rank",
    ]

    remaining_columns = [
        column
        for column in summary.columns
        if column not in preferred_columns
    ]

    summary = summary[
        preferred_columns + remaining_columns
    ]

    out_table = os.path.join(
        OUTDIR,
        "posterior_mu_log1p_abs_mu_rank.tsv",
    )

    summary.to_csv(
        out_table,
        sep="\t",
        index=False,
    )

    print(
        f"[saved table] {out_table}"
    )


    # ============================================================
    # PRINT HIGHLIGHT-GENE RESULTS
    # ============================================================

    print("\n[highlight genes]")

    gene_upper = summary["gene"].str.upper()

    for gene in HIGHLIGHT_GENES:

        hit = summary.loc[
            gene_upper == gene.upper()
        ]

        if hit.empty:
            print(f"{gene}: not found")
            continue

        print(
            hit[
                [
                    "gene",
                    "mu",
                    "abs_mu",
                    "cipher_score",
                    "cipher_rank",
                ]
            ].to_string(index=False)
        )


    # ============================================================
    # PLOT log(1 + |mu|) AGAINST RANK BY |mu|
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.scatter(
        summary["cipher_rank"],
        summary["cipher_score"],
        s=POINT_SIZE,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    ax.set_xlabel(
        r"Rank by posterior effect magnitude $|\mu|$"
    )

    ax.set_ylabel(
        r"CIPHER score: $\log(1 + |\mu|)$"
    )

    ax.set_title(
        "Posterior effect-magnitude rank plot"
    )

    ax.set_xlim(
        -100,
        len(summary) + 1,
    )

    ax.set_ylim(
        bottom=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    # ============================================================
    # HIGHLIGHT FN1 AND IGFBP7
    # ============================================================

    for gene in HIGHLIGHT_GENES:

        hit = summary.loc[
            summary["gene"].str.upper() == gene.upper()
        ]

        if hit.empty:
            continue

        row = hit.iloc[0]

        x = float(row["cipher_rank"])
        y = float(row["cipher_score"])

        ax.scatter(
            x,
            y,
            s=HIGHLIGHT_SIZE,
            marker="*",
            edgecolor="black",
            linewidth=0.8,
            zorder=20,
        )

        ax.annotate(
            gene,
            xy=(x, y),
            xytext=(7, 2),
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=21,
        )


    plt.tight_layout()


    # ============================================================
    # SAVE FIGURE
    # ============================================================

    out_png = os.path.join(
        OUTDIR,
        "posterior_log1p_abs_mu_vs_rank_FN1_IGFBP7.png",
    )

    out_pdf = os.path.join(
        OUTDIR,
        "posterior_log1p_abs_mu_vs_rank_FN1_IGFBP7.pdf",
    )

    out_svg = os.path.join(
        OUTDIR,
        "posterior_log1p_abs_mu_vs_rank_FN1_IGFBP7.svg",
    )

    fig.savefig(
        out_png,
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        out_svg,
        format="svg",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


    # ============================================================
    # DONE
    # ============================================================

    print("\n[DONE]")
    print(f"Saved table: {out_table}")
    print(f"Saved PNG:   {out_png}")
    print(f"Saved PDF:   {out_pdf}")
    print(f"Saved SVG:   {out_svg}")


def plot_cipher_rank_offcurve_labels():
    # ============================================================
    # MELANOMA: LOAD SAVED POSTERIOR OR RERUN, THEN PLOT
    #
    #     CIPHER score = log(1 + |mu|)
    #               versus
    #     CIPHER rank by decreasing |mu|
    #
    # FEATURES:
    #   - loads saved posterior results when available
    #   - reruns run_pipeline(...) only when necessary
    #   - ranks genes by decreasing |mu|
    #   - highlights FN1 and IGFBP7
    #   - places all highlighted labels in a dedicated band above
    #     the curve with leader lines
    #   - labels cannot overlap the posterior curve
    #   - obtains signed log2FC values from the saved summary when
    #     available, otherwise recomputes them from the H5AD file
    #
    # SAVES:
    #   1) ranked posterior table
    #   2) compact LFC/CIPHER comparison table containing:
    #        gene
    #        log2FC
    #        |log2FC|
    #        LFC rank
    #        posterior mu
    #        |posterior mu|
    #        log(1 + |posterior mu|)
    #        CIPHER rank
    #   3) label-layout table
    #   4) PNG, PDF, and SVG figures
    # ============================================================


    # ============================================================
    # CONFIG: MATCH THE ORIGINAL MELANOMA RUN
    # ============================================================

    H5AD_PATH = Path(
        MEL_H5AD
    )

    # Alternative dataset:
    #
    # H5AD_PATH = Path(
    #     "GSE233766_RAW/"
    #     "Xtot_naive_resistant_unbalanced_resistant_BC50_clone_size_gt1.h5ad"
    # )

    OUTDIR = Path(BASE_OUT) / "analytic_gaussian_FN1_IGFBP7_stable_diag"

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONDITION_KEY = "Condition"
    COND0 = "Naive"
    COND1 = "Resistant"

    HIGHLIGHT_GENES = [
        "FN1",
        "IGFBP7",
    ]

    # The original melanoma pipeline used this pseudocount for DE logFC.
    LOGFC_PSEUDOCOUNT = 1.0

    FIGSIZE = (12, 7.5)
    DPI = 300

    LINE_WIDTH = 1.35
    POINT_SIZE = 18
    HIGHLIGHT_SIZE = 220
    LABEL_FONTSIZE = 12

    # Number of rows in the off-curve label band.
    # With FN1 and IGFBP7, one row is sufficient.
    LABEL_ROWS = 1

    # Label-band spacing relative to the score range.
    LABEL_BAND_GAP_FRACTION = 0.13
    LABEL_ROW_SPACING_FRACTION = 0.10

    POSTERIOR_SUMMARY_PATH = (
        OUTDIR
        / "posterior_summary.tsv"
    )

    POSTERIOR_MU_PATH = (
        OUTDIR
        / "posterior_mu.npy"
    )

    SELECTED_GENES_PATH = (
        OUTDIR
        / "selected_genes.npy"
    )

    OUTPUT_STEM = (
        OUTDIR
        / "posterior_log1p_abs_mu_vs_rank_FN1_IGFBP7_offcurve"
    )


    # ============================================================
    # HELPERS
    # ============================================================


    # ============================================================
    # LOAD SAVED RESULTS WHEN POSSIBLE
    # ============================================================

    if POSTERIOR_SUMMARY_PATH.exists():

        print(
            "[load] Found saved posterior summary:\n"
            f"       {POSTERIOR_SUMMARY_PATH}"
        )

        summary = pd.read_csv(
            POSTERIOR_SUMMARY_PATH,
            sep="\t",
            low_memory=False,
        )

    elif (
        POSTERIOR_MU_PATH.exists()
        and SELECTED_GENES_PATH.exists()
    ):

        print(
            "[load] Found posterior_mu.npy and selected_genes.npy"
        )

        mu = np.load(
            POSTERIOR_MU_PATH
        )

        genes = np.load(
            SELECTED_GENES_PATH,
            allow_pickle=True,
        ).astype(str)

        if len(mu) != len(genes):
            raise ValueError(
                "posterior_mu.npy and selected_genes.npy have "
                f"different lengths: {len(mu)} versus {len(genes)}."
            )

        summary = pd.DataFrame({
            "gene": genes,
            "mu": mu,
        })

    else:

        print(
            "[saved data not found] Rerunning run_pipeline(...)"
        )

        if "run_pipeline" not in globals():
            raise RuntimeError(
                "run_pipeline(...) is not defined in this notebook/session.\n"
                "Run the full melanoma pipeline code first, then rerun this block."
            )

        results = run_pipeline(
            h5ad_path=str(H5AD_PATH),
            outdir=str(OUTDIR),

            condition_key=CONDITION_KEY,
            cond0=COND0,
            cond1=COND1,

            genes_to_highlight=[
                "IGFBP7",
                "FN1",
            ],

            # DE gene set
            top_n_de=2000,
            fdr_alpha=0.05,
            min_abs_log2fc=0.01,
            min_abs_delta=0.02,
            rank_by="abs_t",
            fill_to_top_n=True,

            # Gene filtering
            drop_housekeeping=True,
            min_cells_frac=0.01,
            min_expr=0.01,
            min_mean=0.001,
            max_mean=np.inf,
            max_var_quantile=1.0,
            filter_subsample_cells=0,

            # DE logFC only; posterior uses delta_x, not logFC
            logfc_pseudocount=LOGFC_PSEUDOCOUNT,

            # Covariance and posterior settings
            Sigma_shrinkage=1e-6,
            H_shrinkage=1e-6,
            H_ridge=1e-6,
            H_mode="naive",
            tau2=1e-6,
            effect_threshold=None,

            top_k_plot=20,
            seed=0,
        )

        if "summary" not in results:
            raise KeyError(
                "run_pipeline(...) did not return a 'summary' entry."
            )

        summary = results[
            "summary"
        ].copy()


    # ============================================================
    # VALIDATE POSTERIOR SUMMARY
    # ============================================================

    required_columns = {
        "gene",
        "mu",
    }

    missing_columns = required_columns.difference(
        summary.columns
    )

    if missing_columns:
        raise KeyError(
            "Posterior summary is missing required columns: "
            f"{sorted(missing_columns)}\n"
            f"Available columns: {list(summary.columns)}"
        )

    summary = summary.copy()

    summary["gene"] = (
        summary["gene"]
        .astype(str)
        .str.strip()
    )

    summary["gene_upper"] = (
        summary["gene"]
        .map(normalize_gene_name)
    )

    summary["mu"] = pd.to_numeric(
        summary["mu"],
        errors="coerce",
    )


    # ============================================================
    # OBTAIN OR RECOMPUTE SIGNED log2FC
    # ============================================================

    existing_lfc_column = find_lfc_column(
        summary
    )

    if existing_lfc_column is not None:

        print(
            f"[LFC] Using saved signed log2FC column: "
            f"{existing_lfc_column}"
        )

        summary["log2fc"] = pd.to_numeric(
            summary[existing_lfc_column],
            errors="coerce",
        )

        # Use saved means if available.
        mean0_candidates = [
            "mean_cond0",
            "mean0",
            "mean_naive",
        ]

        mean1_candidates = [
            "mean_cond1",
            "mean1",
            "mean_resistant",
        ]

        mean0_column = next(
            (
                column
                for column in mean0_candidates
                if column in summary.columns
            ),
            None,
        )

        mean1_column = next(
            (
                column
                for column in mean1_candidates
                if column in summary.columns
            ),
            None,
        )

        if mean0_column is not None:
            summary["mean_cond0"] = pd.to_numeric(
                summary[mean0_column],
                errors="coerce",
            )
        elif "mean_cond0" not in summary.columns:
            summary["mean_cond0"] = np.nan

        if mean1_column is not None:
            summary["mean_cond1"] = pd.to_numeric(
                summary[mean1_column],
                errors="coerce",
            )
        elif "mean_cond1" not in summary.columns:
            summary["mean_cond1"] = np.nan

    else:

        recomputed_log2fc, mean0, mean1 = (
            compute_log2fc_from_h5ad(
                h5ad_path=H5AD_PATH,
                genes=summary["gene"].to_numpy(dtype=str),
                condition_key=CONDITION_KEY,
                cond0=COND0,
                cond1=COND1,
                pseudocount=LOGFC_PSEUDOCOUNT,
            )
        )

        summary["log2fc"] = recomputed_log2fc
        summary["mean_cond0"] = mean0
        summary["mean_cond1"] = mean1


    # ============================================================
    # COMPUTE CIPHER AND LFC SCORES
    # ============================================================

    summary["abs_mu"] = np.abs(
        summary["mu"].to_numpy(dtype=np.float64)
    )

    summary["log1p_abs_mu"] = np.log1p(
        summary["abs_mu"].to_numpy(dtype=np.float64)
    )

    summary["cipher_score"] = (
        summary["log1p_abs_mu"]
    )

    summary["abs_log2fc"] = np.abs(
        summary["log2fc"].to_numpy(dtype=np.float64)
    )


    # ============================================================
    # REMOVE NONFINITE POSTERIOR VALUES
    # ============================================================

    finite_posterior = (
        np.isfinite(summary["mu"])
        & np.isfinite(summary["abs_mu"])
        & np.isfinite(summary["log1p_abs_mu"])
    )

    n_removed = int(
        (~finite_posterior).sum()
    )

    if n_removed > 0:
        print(
            f"[warning] Removing {n_removed} rows with nonfinite "
            "posterior values."
        )

    summary = (
        summary.loc[finite_posterior]
        .copy()
        .reset_index(drop=True)
    )

    if summary.empty:
        raise ValueError(
            "No finite posterior mean values were found."
        )


    # ============================================================
    # RANK BY CIPHER SCORE
    # ============================================================

    # Because log1p is strictly monotonic for nonnegative values,
    # ranking by |mu| and log(1 + |mu|) gives the same order.
    summary = (
        summary
        .sort_values(
            by=[
                "abs_mu",
                "gene",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    summary["cipher_rank"] = np.arange(
        1,
        len(summary) + 1,
        dtype=int,
    )

    summary["rank_by_abs_mu"] = (
        summary["cipher_rank"]
    )


    # ============================================================
    # COMPUTE LFC RANK
    # ============================================================

    valid_lfc = (
        np.isfinite(summary["log2fc"])
        & np.isfinite(summary["abs_log2fc"])
    )

    summary["lfc_rank"] = ordinal_rank_desc(
        summary["abs_log2fc"].to_numpy(dtype=float),
        valid=valid_lfc.to_numpy(dtype=bool),
    )


    # ============================================================
    # ORDER COLUMNS AND SAVE FULL RANKED TABLE
    # ============================================================

    preferred_columns = [
        "gene",
        "mean_cond0",
        "mean_cond1",
        "log2fc",
        "abs_log2fc",
        "lfc_rank",
        "mu",
        "abs_mu",
        "log1p_abs_mu",
        "cipher_score",
        "cipher_rank",
    ]

    preferred_columns = [
        column
        for column in preferred_columns
        if column in summary.columns
    ]

    remaining_columns = [
        column
        for column in summary.columns
        if column not in preferred_columns
    ]

    summary = summary[
        preferred_columns
        + remaining_columns
    ]

    ranked_table_path = (
        OUTDIR
        / "posterior_mu_log1p_abs_mu_rank.tsv"
    )

    summary.to_csv(
        ranked_table_path,
        sep="\t",
        index=False,
    )

    print(
        f"[saved table] {ranked_table_path}"
    )


    # ============================================================
    # SAVE COMPACT LFC/CIPHER COMPARISON TABLE
    # ============================================================

    lfc_cipher_table = summary[
        [
            "gene",
            "log2fc",
            "abs_log2fc",
            "lfc_rank",
            "mu",
            "abs_mu",
            "log1p_abs_mu",
            "cipher_rank",
        ]
    ].copy()

    lfc_cipher_table = lfc_cipher_table.rename(
        columns={
            "log2fc": "LFC_log2FC",
            "abs_log2fc": "LFC_abs_log2FC",
            "lfc_rank": "LFC_rank",
            "mu": "posterior_mu",
            "abs_mu": "posterior_abs_mu",
            "log1p_abs_mu": "CIPHER_log1p_abs_mu",
            "cipher_rank": "CIPHER_rank",
        }
    )

    lfc_cipher_table = (
        lfc_cipher_table
        .sort_values(
            "CIPHER_rank",
            ascending=True,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    lfc_cipher_table_path = (
        OUTDIR
        / "lfc_rank_and_cipher_score_rank_table.tsv"
    )

    lfc_cipher_table.to_csv(
        lfc_cipher_table_path,
        sep="\t",
        index=False,
    )

    print(
        f"[saved table] {lfc_cipher_table_path}"
    )


    # ============================================================
    # PRINT HIGHLIGHT-GENE RESULTS
    # ============================================================

    print(
        "\n[highlight genes]"
    )

    for gene in HIGHLIGHT_GENES:

        hit = summary.loc[
            summary["gene_upper"]
            == normalize_gene_name(gene)
        ]

        if hit.empty:
            print(
                f"{gene}: not found"
            )
            continue

        columns_to_print = [
            "gene",
            "log2fc",
            "lfc_rank",
            "mu",
            "abs_mu",
            "log1p_abs_mu",
            "cipher_rank",
        ]

        print(
            hit[
                columns_to_print
            ].to_string(index=False)
        )


    # ============================================================
    # PREPARE PLOT DATA
    # ============================================================

    x = summary[
        "cipher_rank"
    ].to_numpy(dtype=float)

    y = summary[
        "log1p_abs_mu"
    ].to_numpy(dtype=float)

    gene_names = summary[
        "gene"
    ].to_numpy(dtype=str)

    gene_names_upper = summary[
        "gene_upper"
    ].to_numpy(dtype=str)

    highlight_gene_set = {
        normalize_gene_name(gene)
        for gene in HIGHLIGHT_GENES
    }

    highlight_indices = np.where(
        np.isin(
            gene_names_upper,
            list(highlight_gene_set),
        )
    )[0]

    present_highlights = set(
        gene_names_upper[highlight_indices]
    )

    missing_highlights = [
        gene
        for gene in HIGHLIGHT_GENES
        if normalize_gene_name(gene)
        not in present_highlights
    ]

    if missing_highlights:
        print(
            "[warning] Highlighted genes absent from the posterior table:",
            ", ".join(missing_highlights),
        )


    # ============================================================
    # COMPUTE LABEL BAND
    # ============================================================

    n_genes = len(summary)

    curve_ymax = float(
        np.nanmax(y)
    )

    curve_ymin = float(
        np.nanmin(y)
    )

    y_scale = max(
        curve_ymax - curve_ymin,
        curve_ymax,
        1e-3,
    )

    band_gap = max(
        0.08,
        LABEL_BAND_GAP_FRACTION * y_scale,
    )

    row_spacing = max(
        0.06,
        LABEL_ROW_SPACING_FRACTION * y_scale,
    )

    x_left_margin = max(
        25.0,
        0.04 * n_genes,
    )

    x_right_margin = max(
        30.0,
        0.04 * n_genes,
    )

    x_min = (
        1.0
        - x_left_margin
    )

    x_max = (
        float(n_genes)
        + x_right_margin
    )

    label_layout = assign_labels_to_top_band(
        x=x,
        y=y,
        gene_names=gene_names,
        label_indices=highlight_indices,
        n_rows=LABEL_ROWS,
        curve_ymax=curve_ymax,
        band_gap=band_gap,
        row_spacing=row_spacing,
        x_min=x_min,
        x_max=x_max,
    )

    if len(label_layout) > 0:
        label_band_top = float(
            label_layout["label_y"].max()
        )
    else:
        label_band_top = (
            curve_ymax
            + band_gap
        )

    y_bottom = 0.0

    y_top = (
        label_band_top
        + max(
            0.08,
            0.10 * y_scale,
        )
    )


    # ============================================================
    # MAKE FIGURE
    # ============================================================

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # Full ranked curve.
    ax.plot(
        x,
        y,
        linewidth=LINE_WIDTH,
        zorder=2,
    )

    ax.scatter(
        x,
        y,
        s=POINT_SIZE,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    # Highlight FN1 and IGFBP7 on the curve.
    if len(highlight_indices) > 0:

        ax.scatter(
            x[highlight_indices],
            y[highlight_indices],
            s=HIGHLIGHT_SIZE,
            marker="*",
            edgecolor="black",
            linewidth=0.8,
            zorder=20,
        )


    # ============================================================
    # DRAW DEDICATED OFF-CURVE LABEL BAND
    # ============================================================

    label_boundary_y = (
        curve_ymax
        + 0.50 * band_gap
    )

    ax.axhline(
        label_boundary_y,
        linewidth=0.8,
        linestyle=":",
        alpha=0.45,
        zorder=1,
    )

    for row in label_layout.itertuples(
        index=False
    ):

        ax.annotate(
            row.gene,

            # Actual point on the posterior curve.
            xy=(
                row.point_x,
                row.point_y,
            ),

            # Label position above the entire curve.
            xytext=(
                row.label_x,
                row.label_y,
            ),

            textcoords="data",
            fontsize=LABEL_FONTSIZE,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=30,
            clip_on=False,

            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.65",
                "linewidth": 0.7,
                "alpha": 1.0,
            },

            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.9,
                "color": "0.30",
                "alpha": 0.85,
                "shrinkA": 4,
                "shrinkB": 7,
                "connectionstyle": "arc3,rad=0.0",
            },
        )


    # ============================================================
    # AXIS FORMATTING
    # ============================================================

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_bottom,
        y_top,
    )

    ax.set_xlabel(
        r"Rank by posterior effect magnitude $|\mu|$",
        fontsize=12,
    )

    ax.set_ylabel(
        r"CIPHER score: $\log(1+|\mu|)$",
        fontsize=12,
    )

    ax.set_title(
        r"Posterior score: $\log(1+|\mu|)$ vs genes ranked by $|\mu|$",
        fontsize=14,
        pad=14,
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.grid(False)

    plt.tight_layout()


    # ============================================================
    # OUTPUT PATHS
    # ============================================================

    out_png = OUTPUT_STEM.with_suffix(
        ".png"
    )

    out_pdf = OUTPUT_STEM.with_suffix(
        ".pdf"
    )

    out_svg = OUTPUT_STEM.with_suffix(
        ".svg"
    )

    label_layout_path = (
        OUTDIR
        / "posterior_log1p_abs_mu_FN1_IGFBP7_label_layout.tsv"
    )


    # ============================================================
    # SAVE FIGURE AND LABEL TABLE
    # ============================================================

    fig.savefig(
        out_png,
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        out_svg,
        format="svg",
        bbox_inches="tight",
    )

    label_layout.to_csv(
        label_layout_path,
        sep="\t",
        index=False,
    )

    plt.show()
    plt.close(
        fig
    )


    # ============================================================
    # DONE
    # ============================================================

    print(
        "\n[DONE]"
    )

    print(
        f"Saved ranked posterior table: {ranked_table_path}"
    )

    print(
        f"Saved LFC/CIPHER table:      {lfc_cipher_table_path}"
    )

    print(
        f"Saved label layout:          {label_layout_path}"
    )

    print(
        f"Saved PNG:                   {out_png}"
    )

    print(
        f"Saved PDF:                   {out_pdf}"
    )

    print(
        f"Saved SVG:                   {out_svg}"
    )

    print(
        "\n[LFC/CIPHER comparison columns]"
    )

    print(
        ", ".join(
            lfc_cipher_table.columns
        )
    )


def plot_naive_control_mean_expression():
    # ============================================================
    # MELANOMA — NAIVE / CONTROL CELLS ONLY — ALL GENES
    #
    # Single panel:
    #   All genes ranked by mean expression in Naive cells
    #
    # Highlights only:
    #   - FN1
    #   - IGFBP7
    #
    # No automatic top-gene labels.
    # No Scanpy dependency.
    # Uses adata.X exactly as stored.
    # ============================================================


    # ============================================================
    # CONFIG
    # ============================================================

    H5AD_PATH = Path(
        MEL_H5AD
    )

    # Alternative dataset:
    #
    # H5AD_PATH = Path(
    #     "GSE233766_RAW/"
    #     "Xtot_naive_resistant_unbalanced_resistant_BC50_clone_size_gt1.h5ad"
    # )

    OUTDIR = Path(BASE_OUT) / "melanoma_naive_all_genes_mean_FN1_IGFBP7"

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONDITION_KEY = "Condition"
    CONTROL_LABEL = "Naive"

    GENES_TO_CHECK = [
        "FN1",
        "IGFBP7",
    ]

    FIGSIZE = (12, 7)
    DPI = 300

    STAR_SIZE = 260
    LABEL_FONTSIZE = 13
    ARROW_LINEWIDTH = 1.2


    # ============================================================
    # HELPERS
    # ============================================================


    # ============================================================
    # LOAD DATA
    # ============================================================

    if not H5AD_PATH.exists():
        raise FileNotFoundError(
            f"Could not find:\n{H5AD_PATH}\n\n"
            f"Current working directory:\n{Path.cwd()}"
        )

    print(
        f"[load] {H5AD_PATH}"
    )

    adata = ad.read_h5ad(
        H5AD_PATH
    )

    adata.var_names_make_unique()

    print(
        f"[load] original shape: "
        f"{adata.n_obs:,} cells × "
        f"{adata.n_vars:,} genes"
    )

    if CONDITION_KEY not in adata.obs.columns:
        raise KeyError(
            f"Missing obs column '{CONDITION_KEY}'.\n"
            f"Available columns:\n"
            f"{list(adata.obs.columns)}"
        )

    conditions = (
        adata.obs[CONDITION_KEY]
        .astype(str)
    )

    print("\n[conditions]")

    print(
        conditions
        .value_counts()
        .to_string()
    )

    control_mask = (
        conditions.to_numpy()
        == CONTROL_LABEL
    )

    if control_mask.sum() == 0:
        raise ValueError(
            f"No cells have "
            f"{CONDITION_KEY} == '{CONTROL_LABEL}'."
        )

    adata_control = adata[
        control_mask
    ].copy()

    print(
        f"\n[control] retained "
        f"{adata_control.n_obs:,} "
        f"{CONTROL_LABEL} cells"
    )


    # ============================================================
    # COMPUTE MEAN EXPRESSION FOR ALL GENES
    # ============================================================

    gene_names = np.asarray(
        adata_control.var_names,
        dtype=str,
    )

    control_mean = gene_means(
        adata_control.X
    )

    valid_mean = (
        np.isfinite(control_mean)
        & (control_mean > 0)
    )

    print(
        f"[statistics] positive finite means: "
        f"{valid_mean.sum():,} / "
        f"{len(control_mean):,}"
    )


    # ============================================================
    # RANK GENES BY DECREASING NAIVE MEAN
    # ============================================================

    sortable_mean = np.where(
        valid_mean,
        control_mean,
        -np.inf,
    )

    order = np.argsort(
        -sortable_mean,
        kind="mergesort",
    )

    ranked_genes = gene_names[
        order
    ]

    ranked_mean = control_mean[
        order
    ]

    ranks = np.arange(
        1,
        len(ranked_genes) + 1,
        dtype=int,
    )

    ranked_valid = (
        np.isfinite(ranked_mean)
        & (ranked_mean > 0)
    )

    ranked_upper = np.char.upper(
        ranked_genes.astype(str)
    )

    tracked_upper = {
        gene.upper()
        for gene in GENES_TO_CHECK
    }

    tracked_indices = np.asarray(
        [
            i
            for i, gene in enumerate(ranked_upper)
            if gene in tracked_upper
            and ranked_valid[i]
        ],
        dtype=int,
    )


    # ============================================================
    # BUILD AND SAVE RANKED TABLE
    # ============================================================

    ranked_table = pd.DataFrame({
        "control_mean_rank": ranks,
        "gene": ranked_genes,
        "control_mean": ranked_mean,
    })

    ranked_table_path = (
        OUTDIR
        / "melanoma_naive_all_genes_ranked_by_mean.tsv"
    )

    ranked_table.to_csv(
        ranked_table_path,
        sep="\t",
        index=False,
    )


    # ============================================================
    # SINGLE MEAN-EXPRESSION PLOT
    # ============================================================

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.plot(
        ranks[ranked_valid],
        ranked_mean[ranked_valid],
        linewidth=1.6,
        zorder=2,
    )

    ax.scatter(
        ranks[ranked_valid],
        ranked_mean[ranked_valid],
        s=10,
        alpha=0.48,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    # Highlight FN1 and IGFBP7.
    if len(tracked_indices) > 0:

        ax.scatter(
            ranks[tracked_indices],
            ranked_mean[tracked_indices],
            s=STAR_SIZE,
            marker="*",
            facecolor="limegreen",
            edgecolor="black",
            linewidth=1.0,
            zorder=20,
            clip_on=False,
        )

        draw_clear_gene_labels(
            ax=ax,
            x=ranks,
            y=ranked_mean,
            labels=ranked_genes,
            indices=tracked_indices,
        )

    ax.set_xlabel(
        "Gene rank",
        fontsize=13,
    )

    ax.set_ylabel(
        "Mean expression in Naive cells",
        fontsize=13,
    )

    ax.set_title(
        "Melanoma Naive mean gene expression",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlim(
        0.5,
        len(ranked_genes) + 0.5,
    )

    ax.set_yscale(
        "log"
    )

    ax.grid(
        True,
        which="major",
        alpha=0.18,
    )

    ax.grid(
        True,
        which="minor",
        alpha=0.05,
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    plt.subplots_adjust(
        left=0.12,
        right=0.96,
        bottom=0.13,
        top=0.90,
    )


    # ============================================================
    # SAVE FIGURE
    # ============================================================

    png_path = (
        OUTDIR
        / "melanoma_naive_all_genes_ranked_mean_FN1_IGFBP7.png"
    )

    pdf_path = (
        OUTDIR
        / "melanoma_naive_all_genes_ranked_mean_FN1_IGFBP7.pdf"
    )

    svg_path = (
        OUTDIR
        / "melanoma_naive_all_genes_ranked_mean_FN1_IGFBP7.svg"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.show()


    # ============================================================
    # FN1 / IGFBP7 SUMMARY
    # ============================================================

    tracked_table = ranked_table.loc[
        ranked_table["gene"]
        .astype(str)
        .str.upper()
        .isin(tracked_upper)
    ].copy()

    tracked_table = tracked_table.sort_values(
        "control_mean_rank"
    )

    tracked_path = (
        OUTDIR
        / "melanoma_naive_FN1_IGFBP7_mean_statistics.tsv"
    )

    tracked_table.to_csv(
        tracked_path,
        sep="\t",
        index=False,
    )

    available_upper = set(
        np.char.upper(
            gene_names.astype(str)
        )
    )

    missing_genes = sorted(
        tracked_upper
        - available_upper
    )

    print("\n" + "=" * 78)
    print("MELANOMA NAIVE CELLS — RANKED MEAN EXPRESSION")
    print("=" * 78)

    print(
        f"Naive cells: {adata_control.n_obs:,}"
    )

    print(
        f"Genes:       {adata_control.n_vars:,}"
    )

    print("\n[FN1 and IGFBP7]")

    if len(tracked_table) > 0:
        print(
            tracked_table.to_string(
                index=False
            )
        )
    else:
        print(
            "FN1 and IGFBP7 were not found in the ranked table."
        )

    if missing_genes:
        print("\n[genes absent from dataset]")
        print(
            ", ".join(missing_genes)
        )

    print("\n[saved]")
    print(png_path)
    print(pdf_path)
    print(svg_path)
    print(ranked_table_path)
    print(tracked_path)


def plot_cipher_rank_highlight_groups():
    # ============================================================
    # MELANOMA: LOAD SAVED POSTERIOR OR RERUN, THEN PLOT
    #
    #     CIPHER score = log(1 + |mu|)
    #               versus
    #     CIPHER rank by decreasing |mu|
    #
    # FEATURES:
    #   - loads saved posterior results when available
    #   - reruns run_pipeline(...) only when necessary
    #   - ranks genes by decreasing |mu|
    #   - highlights one subset with large GREEN dots
    #   - highlights another subset with large GREY dots
    #   - places all highlighted labels in a dedicated band above
    #     the curve with leader lines
    #   - labels cannot overlap the posterior curve
    #   - obtains signed log2FC values from the saved summary when
    #     available, otherwise recomputes them from the H5AD file
    #
    # SAVES:
    #   1) ranked posterior table
    #   2) compact LFC/CIPHER comparison table containing:
    #        gene
    #        log2FC
    #        |log2FC|
    #        LFC rank
    #        posterior mu
    #        |posterior mu|
    #        log(1 + |posterior mu|)
    #        CIPHER rank
    #        highlight group
    #   3) label-layout table
    #   4) PNG, PDF, and SVG figures
    # ============================================================


    # ============================================================
    # CONFIG: MATCH THE ORIGINAL MELANOMA RUN
    # ============================================================

    H5AD_PATH = Path(
        MEL_H5AD
    )

    # Alternative dataset:
    #
    # H5AD_PATH = Path(
    #     "GSE233766_RAW/"
    #     "Xtot_naive_resistant_unbalanced_resistant_BC50_clone_size_gt1.h5ad"
    # )

    OUTDIR = Path(BASE_OUT) / "analytic_gaussian_FN1_IGFBP7_stable_diag"

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONDITION_KEY = "Condition"
    COND0 = "Naive"
    COND1 = "Resistant"


    # ============================================================
    # HIGHLIGHT GROUPS
    #
    # Put each highlighted gene in exactly one list.
    #
    # The union of GREEN_GENES and GREY_GENES determines which
    # genes receive:
    #   - a large dot
    #   - an off-curve label
    #   - a leader line
    # ============================================================

    GREEN_GENES = [
        "FN1", "IGFBP7",
    ]

    GREY_GENES = [

    ]

    HIGHLIGHT_GENES = (
        GREEN_GENES
        + GREY_GENES
    )


    # ============================================================
    # ANALYSIS AND FIGURE SETTINGS
    # ============================================================

    # The original melanoma pipeline used this pseudocount for DE logFC.
    LOGFC_PSEUDOCOUNT = 1.0

    FIGSIZE = (12, 7.5)
    DPI = 300

    LINE_WIDTH = 1.35
    POINT_SIZE = 18

    # Both highlight groups remain larger than the ordinary points.
    HIGHLIGHT_SIZE = 220

    GREEN_HIGHLIGHT_COLOR = "green"
    GREY_HIGHLIGHT_COLOR = "0.60"

    HIGHLIGHT_EDGE_COLOR = "black"
    HIGHLIGHT_EDGE_WIDTH = 0.8

    LABEL_FONTSIZE = 12

    # Number of rows in the off-curve label band.
    LABEL_ROWS = 1

    # Label-band spacing relative to the score range.
    LABEL_BAND_GAP_FRACTION = 0.13
    LABEL_ROW_SPACING_FRACTION = 0.10

    POSTERIOR_SUMMARY_PATH = (
        OUTDIR
        / "posterior_summary.tsv"
    )

    POSTERIOR_MU_PATH = (
        OUTDIR
        / "posterior_mu.npy"
    )

    SELECTED_GENES_PATH = (
        OUTDIR
        / "selected_genes.npy"
    )

    OUTPUT_STEM = (
        OUTDIR
        / "posterior_log1p_abs_mu_vs_rank_green_grey_offcurve"
    )


    # ============================================================
    # HELPERS
    # ============================================================


    # ============================================================
    # NORMALIZE AND VALIDATE HIGHLIGHT GROUPS
    # ============================================================

    green_gene_set = set(
        normalize_gene_list(
            GREEN_GENES
        )
    )

    grey_gene_set = set(
        normalize_gene_list(
            GREY_GENES
        )
    )

    overlapping_genes = (
        green_gene_set
        & grey_gene_set
    )

    if overlapping_genes:
        raise ValueError(
            "The following genes were assigned to both GREEN_GENES "
            "and GREY_GENES:\n"
            f"{sorted(overlapping_genes)}"
        )

    highlight_gene_set = (
        green_gene_set
        | grey_gene_set
    )

    if len(highlight_gene_set) == 0:
        raise ValueError(
            "GREEN_GENES and GREY_GENES are both empty."
        )


    # ============================================================
    # LOAD SAVED RESULTS WHEN POSSIBLE
    # ============================================================

    if POSTERIOR_SUMMARY_PATH.exists():

        print(
            "[load] Found saved posterior summary:\n"
            f"       {POSTERIOR_SUMMARY_PATH}"
        )

        summary = pd.read_csv(
            POSTERIOR_SUMMARY_PATH,
            sep="\t",
            low_memory=False,
        )

    elif (
        POSTERIOR_MU_PATH.exists()
        and SELECTED_GENES_PATH.exists()
    ):

        print(
            "[load] Found posterior_mu.npy and selected_genes.npy"
        )

        mu = np.load(
            POSTERIOR_MU_PATH
        )

        genes = np.load(
            SELECTED_GENES_PATH,
            allow_pickle=True,
        ).astype(str)

        if len(mu) != len(genes):
            raise ValueError(
                "posterior_mu.npy and selected_genes.npy have "
                f"different lengths: {len(mu)} versus {len(genes)}."
            )

        summary = pd.DataFrame(
            {
                "gene": genes,
                "mu": mu,
            }
        )

    else:

        print(
            "[saved data not found] Rerunning run_pipeline(...)"
        )

        if "run_pipeline" not in globals():
            raise RuntimeError(
                "run_pipeline(...) is not defined in this "
                "notebook/session.\n"
                "Run the full melanoma pipeline code first, "
                "then rerun this block."
            )

        results = run_pipeline(
            h5ad_path=str(H5AD_PATH),
            outdir=str(OUTDIR),

            condition_key=CONDITION_KEY,
            cond0=COND0,
            cond1=COND1,

            genes_to_highlight=HIGHLIGHT_GENES,

            # DE gene set
            top_n_de=2000,
            fdr_alpha=0.05,
            min_abs_log2fc=0.01,
            min_abs_delta=0.02,
            rank_by="abs_t",
            fill_to_top_n=True,

            # Gene filtering
            drop_housekeeping=True,
            min_cells_frac=0.01,
            min_expr=0.01,
            min_mean=0.001,
            max_mean=np.inf,
            max_var_quantile=1.0,
            filter_subsample_cells=0,

            # DE logFC only; posterior uses delta_x, not logFC
            logfc_pseudocount=LOGFC_PSEUDOCOUNT,

            # Covariance and posterior settings
            Sigma_shrinkage=1e-6,
            H_shrinkage=1e-6,
            H_ridge=1e-6,
            H_mode="naive",
            tau2=1e-6,
            effect_threshold=None,

            top_k_plot=20,
            seed=0,
        )

        if "summary" not in results:
            raise KeyError(
                "run_pipeline(...) did not return a 'summary' entry."
            )

        summary = results[
            "summary"
        ].copy()


    # ============================================================
    # VALIDATE POSTERIOR SUMMARY
    # ============================================================

    required_columns = {
        "gene",
        "mu",
    }

    missing_columns = required_columns.difference(
        summary.columns
    )

    if missing_columns:
        raise KeyError(
            "Posterior summary is missing required columns: "
            f"{sorted(missing_columns)}\n"
            f"Available columns: {list(summary.columns)}"
        )

    summary = summary.copy()

    summary["gene"] = (
        summary["gene"]
        .astype(str)
        .str.strip()
    )

    summary["gene_upper"] = (
        summary["gene"]
        .map(normalize_gene_name)
    )

    summary["mu"] = pd.to_numeric(
        summary["mu"],
        errors="coerce",
    )


    # ============================================================
    # OBTAIN OR RECOMPUTE SIGNED log2FC
    # ============================================================

    existing_lfc_column = find_lfc_column(
        summary
    )

    if existing_lfc_column is not None:

        print(
            f"[LFC] Using saved signed log2FC column: "
            f"{existing_lfc_column}"
        )

        summary["log2fc"] = pd.to_numeric(
            summary[existing_lfc_column],
            errors="coerce",
        )

        # Use saved means if available.
        mean0_candidates = [
            "mean_cond0",
            "mean0",
            "mean_naive",
        ]

        mean1_candidates = [
            "mean_cond1",
            "mean1",
            "mean_resistant",
        ]

        mean0_column = next(
            (
                column
                for column in mean0_candidates
                if column in summary.columns
            ),
            None,
        )

        mean1_column = next(
            (
                column
                for column in mean1_candidates
                if column in summary.columns
            ),
            None,
        )

        if mean0_column is not None:
            summary["mean_cond0"] = pd.to_numeric(
                summary[mean0_column],
                errors="coerce",
            )

        elif "mean_cond0" not in summary.columns:
            summary["mean_cond0"] = np.nan

        if mean1_column is not None:
            summary["mean_cond1"] = pd.to_numeric(
                summary[mean1_column],
                errors="coerce",
            )

        elif "mean_cond1" not in summary.columns:
            summary["mean_cond1"] = np.nan

    else:

        recomputed_log2fc, mean0, mean1 = (
            compute_log2fc_from_h5ad(
                h5ad_path=H5AD_PATH,
                genes=summary["gene"].to_numpy(dtype=str),
                condition_key=CONDITION_KEY,
                cond0=COND0,
                cond1=COND1,
                pseudocount=LOGFC_PSEUDOCOUNT,
            )
        )

        summary["log2fc"] = (
            recomputed_log2fc
        )

        summary["mean_cond0"] = (
            mean0
        )

        summary["mean_cond1"] = (
            mean1
        )


    # ============================================================
    # COMPUTE CIPHER AND LFC SCORES
    # ============================================================

    summary["abs_mu"] = np.abs(
        summary["mu"].to_numpy(
            dtype=np.float64
        )
    )

    summary["log1p_abs_mu"] = np.log1p(
        summary["abs_mu"].to_numpy(
            dtype=np.float64
        )
    )

    summary["cipher_score"] = (
        summary["log1p_abs_mu"]
    )

    summary["abs_log2fc"] = np.abs(
        summary["log2fc"].to_numpy(
            dtype=np.float64
        )
    )


    # ============================================================
    # REMOVE NONFINITE POSTERIOR VALUES
    # ============================================================

    finite_posterior = (
        np.isfinite(summary["mu"])
        & np.isfinite(summary["abs_mu"])
        & np.isfinite(summary["log1p_abs_mu"])
    )

    n_removed = int(
        (~finite_posterior).sum()
    )

    if n_removed > 0:
        print(
            f"[warning] Removing {n_removed} rows with nonfinite "
            "posterior values."
        )

    summary = (
        summary.loc[finite_posterior]
        .copy()
        .reset_index(drop=True)
    )

    if summary.empty:
        raise ValueError(
            "No finite posterior mean values were found."
        )


    # ============================================================
    # RANK BY CIPHER SCORE
    # ============================================================

    # Because log1p is strictly monotonic for nonnegative values,
    # ranking by |mu| and log(1 + |mu|) gives the same order.
    summary = (
        summary
        .sort_values(
            by=[
                "abs_mu",
                "gene",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    summary["cipher_rank"] = np.arange(
        1,
        len(summary) + 1,
        dtype=int,
    )

    summary["rank_by_abs_mu"] = (
        summary["cipher_rank"]
    )


    # ============================================================
    # COMPUTE LFC RANK
    # ============================================================

    valid_lfc = (
        np.isfinite(summary["log2fc"])
        & np.isfinite(summary["abs_log2fc"])
    )

    summary["lfc_rank"] = ordinal_rank_desc(
        summary["abs_log2fc"].to_numpy(
            dtype=float
        ),
        valid=valid_lfc.to_numpy(
            dtype=bool
        ),
    )


    # ============================================================
    # ADD GREEN/GREY GROUP ASSIGNMENTS
    # ============================================================

    summary["highlight_group"] = "unlabeled"

    summary.loc[
        summary["gene_upper"].isin(
            green_gene_set
        ),
        "highlight_group",
    ] = "green"

    summary.loc[
        summary["gene_upper"].isin(
            grey_gene_set
        ),
        "highlight_group",
    ] = "grey"

    summary["highlighted"] = (
        summary["highlight_group"]
        != "unlabeled"
    )


    # ============================================================
    # ORDER COLUMNS AND SAVE FULL RANKED TABLE
    # ============================================================

    preferred_columns = [
        "gene",
        "mean_cond0",
        "mean_cond1",
        "log2fc",
        "abs_log2fc",
        "lfc_rank",
        "mu",
        "abs_mu",
        "log1p_abs_mu",
        "cipher_score",
        "cipher_rank",
        "highlighted",
        "highlight_group",
    ]

    preferred_columns = [
        column
        for column in preferred_columns
        if column in summary.columns
    ]

    remaining_columns = [
        column
        for column in summary.columns
        if column not in preferred_columns
    ]

    summary = summary[
        preferred_columns
        + remaining_columns
    ]

    ranked_table_path = (
        OUTDIR
        / "posterior_mu_log1p_abs_mu_rank_green_grey.tsv"
    )

    summary.to_csv(
        ranked_table_path,
        sep="\t",
        index=False,
    )

    print(
        f"[saved table] {ranked_table_path}"
    )


    # ============================================================
    # SAVE COMPACT LFC/CIPHER COMPARISON TABLE
    # ============================================================

    lfc_cipher_table = summary[
        [
            "gene",
            "log2fc",
            "abs_log2fc",
            "lfc_rank",
            "mu",
            "abs_mu",
            "log1p_abs_mu",
            "cipher_rank",
            "highlight_group",
        ]
    ].copy()

    lfc_cipher_table = lfc_cipher_table.rename(
        columns={
            "log2fc": "LFC_log2FC",
            "abs_log2fc": "LFC_abs_log2FC",
            "lfc_rank": "LFC_rank",
            "mu": "posterior_mu",
            "abs_mu": "posterior_abs_mu",
            "log1p_abs_mu": "CIPHER_log1p_abs_mu",
            "cipher_rank": "CIPHER_rank",
        }
    )

    lfc_cipher_table = (
        lfc_cipher_table
        .sort_values(
            "CIPHER_rank",
            ascending=True,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    lfc_cipher_table_path = (
        OUTDIR
        / "lfc_rank_and_cipher_score_rank_table_green_grey.tsv"
    )

    lfc_cipher_table.to_csv(
        lfc_cipher_table_path,
        sep="\t",
        index=False,
    )

    print(
        f"[saved table] {lfc_cipher_table_path}"
    )


    # ============================================================
    # PRINT HIGHLIGHT-GENE RESULTS
    # ============================================================

    print(
        "\n[highlight genes]"
    )

    for gene in HIGHLIGHT_GENES:

        hit = summary.loc[
            summary["gene_upper"]
            == normalize_gene_name(gene)
        ]

        if hit.empty:
            print(
                f"{gene}: not found"
            )
            continue

        columns_to_print = [
            "gene",
            "log2fc",
            "lfc_rank",
            "mu",
            "abs_mu",
            "log1p_abs_mu",
            "cipher_rank",
            "highlight_group",
        ]

        print(
            hit[
                columns_to_print
            ].to_string(index=False)
        )


    # ============================================================
    # PREPARE PLOT DATA
    # ============================================================

    x = summary[
        "cipher_rank"
    ].to_numpy(
        dtype=float
    )

    y = summary[
        "log1p_abs_mu"
    ].to_numpy(
        dtype=float
    )

    gene_names = summary[
        "gene"
    ].to_numpy(
        dtype=str
    )

    gene_names_upper = summary[
        "gene_upper"
    ].to_numpy(
        dtype=str
    )

    green_indices = np.where(
        np.isin(
            gene_names_upper,
            list(green_gene_set),
        )
    )[0]

    grey_indices = np.where(
        np.isin(
            gene_names_upper,
            list(grey_gene_set),
        )
    )[0]

    highlight_indices = np.sort(
        np.concatenate(
            [
                green_indices,
                grey_indices,
            ]
        )
    )

    present_highlights = set(
        gene_names_upper[
            highlight_indices
        ]
    )

    missing_green_genes = [
        gene
        for gene in GREEN_GENES
        if normalize_gene_name(gene)
        not in present_highlights
    ]

    missing_grey_genes = [
        gene
        for gene in GREY_GENES
        if normalize_gene_name(gene)
        not in present_highlights
    ]

    print(
        f"\n[plot] green genes found: "
        f"{len(green_indices)} / {len(GREEN_GENES)}"
    )

    print(
        f"[plot] grey genes found: "
        f"{len(grey_indices)} / {len(GREY_GENES)}"
    )

    if len(green_indices) > 0:
        print(
            "[plot] green genes:",
            ", ".join(
                gene_names[green_indices]
            ),
        )

    if len(grey_indices) > 0:
        print(
            "[plot] grey genes:",
            ", ".join(
                gene_names[grey_indices]
            ),
        )

    if missing_green_genes:
        print(
            "[warning] Green genes absent from the posterior table:",
            ", ".join(
                missing_green_genes
            ),
        )

    if missing_grey_genes:
        print(
            "[warning] Grey genes absent from the posterior table:",
            ", ".join(
                missing_grey_genes
            ),
        )


    # ============================================================
    # COMPUTE LABEL BAND
    # ============================================================

    n_genes = len(
        summary
    )

    curve_ymax = float(
        np.nanmax(y)
    )

    curve_ymin = float(
        np.nanmin(y)
    )

    y_scale = max(
        curve_ymax - curve_ymin,
        curve_ymax,
        1e-3,
    )

    band_gap = max(
        0.08,
        LABEL_BAND_GAP_FRACTION
        * y_scale,
    )

    row_spacing = max(
        0.06,
        LABEL_ROW_SPACING_FRACTION
        * y_scale,
    )

    x_left_margin = max(
        25.0,
        0.04 * n_genes,
    )

    x_right_margin = max(
        30.0,
        0.04 * n_genes,
    )

    x_min = (
        1.0
        - x_left_margin
    )

    x_max = (
        float(n_genes)
        + x_right_margin
    )

    label_layout = assign_labels_to_top_band(
        x=x,
        y=y,
        gene_names=gene_names,
        label_indices=highlight_indices,
        n_rows=LABEL_ROWS,
        curve_ymax=curve_ymax,
        band_gap=band_gap,
        row_spacing=row_spacing,
        x_min=x_min,
        x_max=x_max,
    )

    if len(label_layout) > 0:

        label_layout["highlight_group"] = (
            label_layout["gene"]
            .map(normalize_gene_name)
            .map(
                lambda gene: (
                    "green"
                    if gene in green_gene_set
                    else "grey"
                )
            )
        )

        label_band_top = float(
            label_layout["label_y"].max()
        )

    else:

        label_layout["highlight_group"] = pd.Series(
            dtype=str
        )

        label_band_top = (
            curve_ymax
            + band_gap
        )

    y_bottom = 0.0

    y_top = (
        label_band_top
        + max(
            0.08,
            0.10 * y_scale,
        )
    )


    # ============================================================
    # MAKE FIGURE
    # ============================================================

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )


    # ------------------------------------------------------------
    # Full ranked curve
    # ------------------------------------------------------------

    ax.plot(
        x,
        y,
        linewidth=LINE_WIDTH,
        zorder=2,
    )

    ax.scatter(
        x,
        y,
        s=POINT_SIZE,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )


    # ------------------------------------------------------------
    # Highlight green genes with large green circular dots
    # ------------------------------------------------------------

    if len(green_indices) > 0:

        ax.scatter(
            x[green_indices],
            y[green_indices],
            s=HIGHLIGHT_SIZE,
            marker="o",
            facecolor=GREEN_HIGHLIGHT_COLOR,
            edgecolor=HIGHLIGHT_EDGE_COLOR,
            linewidth=HIGHLIGHT_EDGE_WIDTH,
            alpha=1.0,
            zorder=21,
            label="Green gene group",
        )


    # ------------------------------------------------------------
    # Highlight grey genes with large grey circular dots
    # ------------------------------------------------------------

    if len(grey_indices) > 0:

        ax.scatter(
            x[grey_indices],
            y[grey_indices],
            s=HIGHLIGHT_SIZE,
            marker="o",
            facecolor=GREY_HIGHLIGHT_COLOR,
            edgecolor=HIGHLIGHT_EDGE_COLOR,
            linewidth=HIGHLIGHT_EDGE_WIDTH,
            alpha=1.0,
            zorder=20,
            label="Grey gene group",
        )


    # ============================================================
    # DRAW DEDICATED OFF-CURVE LABEL BAND
    # ============================================================

    label_boundary_y = (
        curve_ymax
        + 0.50 * band_gap
    )

    ax.axhline(
        label_boundary_y,
        linewidth=0.8,
        linestyle=":",
        alpha=0.45,
        zorder=1,
    )

    for row in label_layout.itertuples(
        index=False
    ):

        ax.annotate(
            row.gene,

            # Actual point on the posterior curve.
            xy=(
                row.point_x,
                row.point_y,
            ),

            # Label position above the entire curve.
            xytext=(
                row.label_x,
                row.label_y,
            ),

            textcoords="data",
            fontsize=LABEL_FONTSIZE,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=30,
            clip_on=False,

            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.65",
                "linewidth": 0.7,
                "alpha": 1.0,
            },

            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.9,
                "color": "0.30",
                "alpha": 0.85,
                "shrinkA": 4,
                "shrinkB": 7,
                "connectionstyle": "arc3,rad=0.0",
            },
        )


    # ============================================================
    # AXIS FORMATTING
    # ============================================================

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_bottom,
        y_top,
    )

    ax.set_xlabel(
        r"Rank by posterior effect magnitude $|\mu|$",
        fontsize=12,
    )

    ax.set_ylabel(
        r"CIPHER score: $\log(1+|\mu|)$",
        fontsize=12,
    )

    ax.set_title(
        r"Posterior score: $\log(1+|\mu|)$ vs genes ranked by $|\mu|$",
        fontsize=14,
        pad=14,
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.grid(False)

    # Uncomment to display a legend.
    # ax.legend(
    #     frameon=False,
    #     loc="upper right",
    # )

    plt.tight_layout()


    # ============================================================
    # OUTPUT PATHS
    # ============================================================

    out_png = OUTPUT_STEM.with_suffix(
        ".png"
    )

    out_pdf = OUTPUT_STEM.with_suffix(
        ".pdf"
    )

    out_svg = OUTPUT_STEM.with_suffix(
        ".svg"
    )

    label_layout_path = (
        OUTDIR
        / "posterior_log1p_abs_mu_green_grey_label_layout.tsv"
    )


    # ============================================================
    # SAVE FIGURE AND LABEL TABLE
    # ============================================================

    fig.savefig(
        out_png,
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        out_svg,
        format="svg",
        bbox_inches="tight",
    )

    label_layout.to_csv(
        label_layout_path,
        sep="\t",
        index=False,
    )

    plt.show()

    plt.close(
        fig
    )


    # ============================================================
    # DONE
    # ============================================================

    print(
        "\n[DONE]"
    )

    print(
        f"Green genes:                  "
        f"{', '.join(GREEN_GENES)}"
    )

    print(
        f"Grey genes:                   "
        f"{', '.join(GREY_GENES)}"
    )

    print(
        f"Saved ranked posterior table: {ranked_table_path}"
    )

    print(
        f"Saved LFC/CIPHER table:       {lfc_cipher_table_path}"
    )

    print(
        f"Saved label layout:           {label_layout_path}"
    )

    print(
        f"Saved PNG:                    {out_png}"
    )

    print(
        f"Saved PDF:                    {out_pdf}"
    )

    print(
        f"Saved SVG:                    {out_svg}"
    )

    print(
        "\n[LFC/CIPHER comparison columns]"
    )

    print(
        ", ".join(
            lfc_cipher_table.columns
        )
    )


def build_lfc_cipher_table():
    # ============================================================
    # LOAD SAVED DE + POSTERIOR RESULTS IF AVAILABLE;
    # OTHERWISE RECOMPUTE EVERYTHING FROM THE H5AD FILE.
    #
    # FINAL OUTPUT:
    #   full_gene_lfc_cipher_ranking.tsv
    #   full_gene_lfc_cipher_ranking.csv
    #
    # Primary columns:
    #   gene
    #   lfc
    #   lfc_rank
    #   mu
    #   cipher_score = log(1 + mu)
    #   cipher_rank
    #
    # By default:
    #   - LFC rank is based on descending |LFC|
    #   - CIPHER rank is based on descending log(1 + mu)
    # ============================================================


    # ============================================================
    # CONFIGURATION
    # ============================================================

    H5AD_PATH = (
        MEL_H5AD
    )

    # Alternative dataset:
    # H5AD_PATH = (
    #     "GSE233766_RAW/"
    #     "Xtot_naive_resistant_unbalanced_resistant_BC50_clone_size_gt1.h5ad"
    # )

    OUTDIR = os.path.join(BASE_OUT, "analytic_gaussian_FN1_IGFBP7_stable_diag")
    os.makedirs(OUTDIR, exist_ok=True)

    CONDITION_KEY = "Condition"
    COND0 = "Naive"
    COND1 = "Resistant"

    HIGHLIGHT_GENES = ["FN1", "IGFBP7"]

    # Set to True to ignore saved results and rerun everything.
    FORCE_RECOMPUTE = False

    # LFC ranking:
    #   "absolute" -> rank by |log2FC|
    #   "signed"   -> largest positive log2FC first
    LFC_RANK_MODE = "absolute"

    # DE selection
    TOP_N_DE = 2000
    FDR_ALPHA = 0.05
    MIN_ABS_LOG2FC = 0.01
    MIN_ABS_DELTA = 0.02
    DE_RANK_BY = "abs_t"
    FILL_TO_TOP_N = True

    # Gene filtering
    DROP_HOUSEKEEPING = True
    MIN_CELLS_FRAC = 0.01
    MIN_EXPR = 0.01
    MIN_MEAN = 0.001
    MAX_MEAN = np.inf
    MAX_VAR_QUANTILE = 1.0
    FILTER_SUBSAMPLE_CELLS = 0

    # DE log-fold-change pseudocount
    LOGFC_PSEUDOCOUNT = 1.0

    # Covariance and posterior
    SIGMA_SHRINKAGE = 1e-6
    H_SHRINKAGE = 1e-6
    H_RIDGE = 1e-6
    H_MODE = "naive"
    TAU2 = 1e-6
    EFFECT_THRESHOLD = None

    SEED = 0


    # ============================================================
    # FILE PATHS
    # ============================================================

    ALL_GENES_DE_PATH = os.path.join(
        OUTDIR,
        "all_genes_de.tsv",
    )

    SELECTED_DE_PATH = os.path.join(
        OUTDIR,
        "selected_de.tsv",
    )

    PRIMARY_PASSING_DE_PATH = os.path.join(
        OUTDIR,
        "primary_passing_de.tsv",
    )

    POSTERIOR_SUMMARY_PATH = os.path.join(
        OUTDIR,
        "posterior_summary.tsv",
    )

    SELECTED_GENES_PATH = os.path.join(
        OUTDIR,
        "selected_genes.npy",
    )

    SIGMA_PATH = os.path.join(
        OUTDIR,
        "Sigma.npy",
    )

    H_PATH = os.path.join(
        OUTDIR,
        "H.npy",
    )

    DELTA_X_PATH = os.path.join(
        OUTDIR,
        "delta_x.npy",
    )

    POSTERIOR_MU_PATH = os.path.join(
        OUTDIR,
        "posterior_mu.npy",
    )

    POSTERIOR_STD_PATH = os.path.join(
        OUTDIR,
        "posterior_std.npy",
    )

    POSTERIOR_COV_PATH = os.path.join(
        OUTDIR,
        "posterior_cov.npy",
    )

    POSTERIOR_YHAT_PATH = os.path.join(
        OUTDIR,
        "posterior_yhat.npy",
    )

    FINAL_TSV_PATH = os.path.join(
        OUTDIR,
        "full_gene_lfc_cipher_ranking.tsv",
    )

    FINAL_CSV_PATH = os.path.join(
        OUTDIR,
        "full_gene_lfc_cipher_ranking.csv",
    )


    # ============================================================
    # BASIC UTILITIES
    # ============================================================


    # ============================================================
    # GENE FILTERING
    # ============================================================


    # ============================================================
    # DIFFERENTIAL EXPRESSION
    # ============================================================


    # ============================================================
    # COVARIANCE AND POSTERIOR
    # ============================================================


    # ============================================================
    # CHECK WHETHER SAVED RESULTS CAN BE USED
    # ============================================================

    posterior_exists = os.path.exists(
        POSTERIOR_SUMMARY_PATH
    )

    selected_de_exists = os.path.exists(
        SELECTED_DE_PATH
    )

    all_de_exists = os.path.exists(
        ALL_GENES_DE_PATH
    )

    saved_results_available = (
        posterior_exists
        and (
            selected_de_exists
            or all_de_exists
        )
    )


    # ============================================================
    # LOAD SAVED RESULTS OR RECOMPUTE
    # ============================================================

    if (
        saved_results_available
        and not FORCE_RECOMPUTE
    ):
        print("=" * 80)
        print("[LOAD] USING SAVED DE AND POSTERIOR RESULTS")
        print("=" * 80)

        posterior_summary = pd.read_csv(
            POSTERIOR_SUMMARY_PATH,
            sep="\t",
        )

        if selected_de_exists:
            de_table = pd.read_csv(
                SELECTED_DE_PATH,
                sep="\t",
            )

            de_source_path = SELECTED_DE_PATH

        else:
            de_table = pd.read_csv(
                ALL_GENES_DE_PATH,
                sep="\t",
            )

            de_source_path = ALL_GENES_DE_PATH

        print(
            f"[load] Posterior: {POSTERIOR_SUMMARY_PATH}"
        )
        print(
            f"[load] DE table:  {de_source_path}"
        )
        print(
            f"[load] Posterior rows: {len(posterior_summary):,}"
        )
        print(
            f"[load] DE rows:        {len(de_table):,}"
        )

    else:
        print("=" * 80)
        print("[RUN] SAVED RESULTS NOT FOUND OR RECOMPUTE REQUESTED")
        print("=" * 80)

        check_file(H5AD_PATH)

        adata = read_h5ad_robust(
            H5AD_PATH
        )

        adata.var_names_make_unique()

        print(
            f"[data] Loaded {adata.n_obs:,} cells "
            f"x {adata.n_vars:,} genes"
        )

        if CONDITION_KEY not in adata.obs.columns:
            raise KeyError(
                f"'{CONDITION_KEY}' was not found in adata.obs.\n"
                f"Available columns:\n"
                f"{list(adata.obs.columns)}"
            )

        print(
            f"\n[data] Values in {CONDITION_KEY}:"
        )

        print(
            pd.Series(
                adata.obs[CONDITION_KEY]
            ).value_counts()
        )

        condition_values = (
            adata.obs[CONDITION_KEY]
            .astype(str)
            .to_numpy()
        )

        contrast_mask = np.isin(
            condition_values,
            [COND0, COND1],
        )

        adata = adata[
            contrast_mask
        ].copy()

        condition_values = (
            adata.obs[CONDITION_KEY]
            .astype(str)
            .to_numpy()
        )

        cond0_mask = (
            condition_values == COND0
        )

        cond1_mask = (
            condition_values == COND1
        )

        print(
            f"\n[contrast] {COND1} - {COND0}"
        )
        print(
            f"  {COND0}: {int(cond0_mask.sum()):,} cells"
        )
        print(
            f"  {COND1}: {int(cond1_mask.sum()):,} cells"
        )

        if (
            cond0_mask.sum() < 5
            or cond1_mask.sum() < 5
        ):
            raise ValueError(
                "Too few cells for the requested contrast: "
                f"{COND0}={int(cond0_mask.sum())}, "
                f"{COND1}={int(cond1_mask.sum())}"
            )

        adata = filter_genes_basic(
            adata=adata,
            min_cells_frac=MIN_CELLS_FRAC,
            min_expr=MIN_EXPR,
            min_mean=MIN_MEAN,
            max_mean=MAX_MEAN,
            max_var_quantile=MAX_VAR_QUANTILE,
            seed=SEED,
            filter_subsample_cells=FILTER_SUBSAMPLE_CELLS,
        )

        if DROP_HOUSEKEEPING:
            keep_gene_mask = drop_bad_gene_prefixes(
                adata.var_names
            )

            print(
                "[filter] Bad-prefix filter kept "
                f"{int(keep_gene_mask.sum()):,} / "
                f"{len(keep_gene_mask):,} genes"
            )

            adata = adata[
                :,
                keep_gene_mask,
            ].copy()

        condition_values = (
            adata.obs[CONDITION_KEY]
            .astype(str)
            .to_numpy()
        )

        cond0_mask = (
            condition_values == COND0
        )

        cond1_mask = (
            condition_values == COND1
        )

        X0 = to_dense(
            adata[cond0_mask].X
        ).astype(
            np.float64,
            copy=False,
        )

        X1 = to_dense(
            adata[cond1_mask].X
        ).astype(
            np.float64,
            copy=False,
        )

        all_gene_names = np.asarray(
            adata.var_names,
            dtype=str,
        )

        print(
            f"[matrix] {COND0}: {X0.shape}"
        )
        print(
            f"[matrix] {COND1}: {X1.shape}"
        )

        all_genes_de = compute_de_scores(
            X0=X0,
            X1=X1,
            gene_names=all_gene_names,
            logfc_pseudocount=LOGFC_PSEUDOCOUNT,
        )

        all_genes_de.to_csv(
            ALL_GENES_DE_PATH,
            sep="\t",
            index=False,
        )

        selected_de, primary_passing_de = (
            select_de_genes(
                de_df=all_genes_de,
                top_n_de=TOP_N_DE,
                fdr_alpha=FDR_ALPHA,
                min_abs_log2fc=MIN_ABS_LOG2FC,
                min_abs_delta=MIN_ABS_DELTA,
                rank_by=DE_RANK_BY,
                fill_to_top_n=FILL_TO_TOP_N,
            )
        )

        selected_de.to_csv(
            SELECTED_DE_PATH,
            sep="\t",
            index=False,
        )

        primary_passing_de.to_csv(
            PRIMARY_PASSING_DE_PATH,
            sep="\t",
            index=False,
        )

        selected_gene_names = (
            selected_de["gene"]
            .astype(str)
            .to_numpy()
        )

        gene_to_original_index = {
            gene: index
            for index, gene in enumerate(
                all_gene_names
            )
        }

        missing_selected_genes = [
            gene
            for gene in selected_gene_names
            if gene not in gene_to_original_index
        ]

        if missing_selected_genes:
            raise ValueError(
                "Some selected genes could not be found in the "
                "filtered expression matrix:\n"
                f"{missing_selected_genes[:20]}"
            )

        selected_indices = np.asarray(
            [
                gene_to_original_index[gene]
                for gene in selected_gene_names
            ],
            dtype=int,
        )

        X0_selected = X0[
            :,
            selected_indices,
        ]

        X1_selected = X1[
            :,
            selected_indices,
        ]

        print(
            f"\n[selected matrix] {COND0}: "
            f"{X0_selected.shape}"
        )

        print(
            f"[selected matrix] {COND1}: "
            f"{X1_selected.shape}"
        )

        delta_x = (
            X1_selected.mean(axis=0)
            - X0_selected.mean(axis=0)
        )

        Sigma = compute_covariance(
            X0_selected,
            shrinkage=SIGMA_SHRINKAGE,
        )

        H = build_H_from_sample_means(
            X0=X0_selected,
            X1=X1_selected,
            shrinkage=H_SHRINKAGE,
            ridge=H_RIDGE,
            mode=H_MODE,
        )

        posterior = analytic_gaussian_posterior(
            Sigma=Sigma,
            response=delta_x,
            H=H,
            tau2=TAU2,
        )

        posterior_summary, used_effect_threshold = (
            make_posterior_summary(
                mu=posterior["mu"],
                std=posterior["std"],
                gene_names=selected_gene_names,
                delta_x=delta_x,
                effect_threshold=EFFECT_THRESHOLD,
            )
        )

        posterior_summary.to_csv(
            POSTERIOR_SUMMARY_PATH,
            sep="\t",
            index=False,
        )

        np.save(
            SELECTED_GENES_PATH,
            selected_gene_names,
        )

        np.save(
            SIGMA_PATH,
            Sigma,
        )

        np.save(
            H_PATH,
            H,
        )

        np.save(
            DELTA_X_PATH,
            delta_x,
        )

        np.save(
            POSTERIOR_MU_PATH,
            posterior["mu"],
        )

        np.save(
            POSTERIOR_STD_PATH,
            posterior["std"],
        )

        np.save(
            POSTERIOR_COV_PATH,
            posterior["Cov"],
        )

        np.save(
            POSTERIOR_YHAT_PATH,
            posterior["yhat"],
        )

        print("\n[posterior]")
        print(
            f"  R²:               {posterior['r2']:.6f}"
        )
        print(
            f"  tau²:              {TAU2}"
        )
        print(
            f"  H mode:            {H_MODE}"
        )
        print(
            f"  Effect threshold:  {used_effect_threshold:.6g}"
        )

        de_table = selected_de.copy()

        print("\n[saved recomputed results]")
        print(
            f"  {ALL_GENES_DE_PATH}"
        )
        print(
            f"  {SELECTED_DE_PATH}"
        )
        print(
            f"  {POSTERIOR_SUMMARY_PATH}"
        )


    # ============================================================
    # VALIDATE LOADED/COMPUTED TABLES
    # ============================================================

    required_posterior_columns = {
        "gene",
        "mu",
    }

    missing_posterior_columns = (
        required_posterior_columns
        - set(posterior_summary.columns)
    )

    if missing_posterior_columns:
        raise KeyError(
            "The posterior summary is missing columns: "
            f"{sorted(missing_posterior_columns)}"
        )

    if "gene" not in de_table.columns:
        raise KeyError(
            "The DE table does not contain a 'gene' column."
        )

    lfc_candidates = [
        "log2fc",
        "log2FC",
        "log2_fc",
        "log2_fold_change",
        "log2FoldChange",
        "lfc",
        "LFC",
        "logfc",
        "logFC",
    ]

    lfc_column = None

    for candidate in lfc_candidates:
        if candidate in de_table.columns:
            lfc_column = candidate
            break

    if lfc_column is None:
        lowercase_columns = {
            str(column).lower(): column
            for column in de_table.columns
        }

        for candidate in lfc_candidates:
            if candidate.lower() in lowercase_columns:
                lfc_column = lowercase_columns[
                    candidate.lower()
                ]
                break

    if lfc_column is None:
        raise KeyError(
            "Could not identify the LFC column in the DE table.\n"
            f"Available columns:\n{list(de_table.columns)}"
        )

    print(
        f"\n[table] Using DE LFC column: {lfc_column}"
    )


    # ============================================================
    # CLEAN TABLES BEFORE MERGING
    # ============================================================

    posterior_summary = (
        posterior_summary
        .copy()
    )

    de_table = (
        de_table
        .copy()
    )

    posterior_summary["gene"] = (
        posterior_summary["gene"]
        .astype(str)
        .str.strip()
    )

    de_table["gene"] = (
        de_table["gene"]
        .astype(str)
        .str.strip()
    )

    posterior_summary["mu"] = pd.to_numeric(
        posterior_summary["mu"],
        errors="coerce",
    )

    de_table[lfc_column] = pd.to_numeric(
        de_table[lfc_column],
        errors="coerce",
    )

    posterior_summary = posterior_summary.loc[
        posterior_summary["gene"].ne("")
    ].copy()

    de_table = de_table.loc[
        de_table["gene"].ne("")
    ].copy()

    # Avoid accidental row multiplication if duplicate gene rows exist.
    posterior_summary = (
        posterior_summary
        .drop_duplicates(
            subset="gene",
            keep="first",
        )
    )

    de_table = (
        de_table
        .drop_duplicates(
            subset="gene",
            keep="first",
        )
    )


    # ============================================================
    # MERGE DE AND POSTERIOR INFORMATION
    # ============================================================

    full_table = posterior_summary.merge(
        de_table,
        on="gene",
        how="left",
        suffixes=(
            "_posterior",
            "_de",
        ),
        validate="one_to_one",
    )

    full_table["lfc"] = pd.to_numeric(
        full_table[lfc_column],
        errors="coerce",
    )

    full_table["abs_lfc"] = np.abs(
        full_table["lfc"]
    )


    # ============================================================
    # CALCULATE LFC RANK
    # ============================================================

    if LFC_RANK_MODE.lower() == "absolute":
        lfc_ranking_values = (
            full_table["abs_lfc"]
        )

        lfc_rank_description = (
            "descending absolute log2 fold change"
        )

    elif LFC_RANK_MODE.lower() == "signed":
        lfc_ranking_values = (
            full_table["lfc"]
        )

        lfc_rank_description = (
            "descending signed log2 fold change"
        )

    else:
        raise ValueError(
            "LFC_RANK_MODE must be either "
            "'absolute' or 'signed'."
        )

    full_table["lfc_rank"] = pd.Series(
        pd.NA,
        index=full_table.index,
        dtype="Int64",
    )

    valid_lfc = np.isfinite(
        lfc_ranking_values.to_numpy(
            dtype=float
        )
    )

    full_table.loc[
        valid_lfc,
        "lfc_rank",
    ] = (
        lfc_ranking_values.loc[valid_lfc]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )


    # ============================================================
    # CALCULATE CIPHER SCORE AND CIPHER RANK
    #
    # CIPHER score = log(1 + mu)
    #
    # This score is mathematically defined only when mu > -1.
    # ============================================================

    mu_values = full_table["mu"].to_numpy(
        dtype=float
    )

    valid_cipher_score = (
        np.isfinite(mu_values)
        & (mu_values > -1.0)
    )

    full_table["cipher_score"] = np.nan

    full_table.loc[
        valid_cipher_score,
        "cipher_score",
    ] = np.log1p(
        mu_values[valid_cipher_score]
    )

    full_table["cipher_rank"] = pd.Series(
        pd.NA,
        index=full_table.index,
        dtype="Int64",
    )

    full_table.loc[
        valid_cipher_score,
        "cipher_rank",
    ] = (
        full_table.loc[
            valid_cipher_score,
            "cipher_score",
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    full_table["cipher_score_defined"] = (
        valid_cipher_score
    )

    n_invalid_cipher = int(
        (~valid_cipher_score).sum()
    )

    if n_invalid_cipher > 0:
        print(
            f"[warning] {n_invalid_cipher:,} genes have "
            "nonfinite mu or mu <= -1. Their log(1 + mu) "
            "score and CIPHER rank are left missing."
        )


    # ============================================================
    # ADD DIRECTION AND HIGHLIGHT INFORMATION
    # ============================================================

    full_table["lfc_direction"] = np.select(
        [
            full_table["lfc"] > 0,
            full_table["lfc"] < 0,
            full_table["lfc"] == 0,
        ],
        [
            "up",
            "down",
            "unchanged",
        ],
        default="missing",
    )

    highlight_upper = {
        gene.upper()
        for gene in HIGHLIGHT_GENES
    }

    full_table["is_highlight_gene"] = (
        full_table["gene"]
        .str.upper()
        .isin(highlight_upper)
    )


    # ============================================================
    # SORT BY CIPHER SCORE
    # ============================================================

    full_table = (
        full_table
        .sort_values(
            by=[
                "cipher_score",
                "abs_lfc",
                "gene",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            na_position="last",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


    # ============================================================
    # PUT REQUESTED COLUMNS FIRST
    # ============================================================

    requested_columns = [
        "gene",
        "lfc",
        "lfc_rank",
        "mu",
        "cipher_score",
        "cipher_rank",
        "abs_lfc",
        "lfc_direction",
        "cipher_score_defined",
        "is_highlight_gene",
    ]

    remaining_columns = [
        column
        for column in full_table.columns
        if column not in requested_columns
    ]

    full_table = full_table[
        requested_columns
        + remaining_columns
    ]


    # ============================================================
    # SAVE THE FULL TABLE
    # ============================================================

    full_table.to_csv(
        FINAL_TSV_PATH,
        sep="\t",
        index=False,
    )

    full_table.to_csv(
        FINAL_CSV_PATH,
        index=False,
    )


    # ============================================================
    # PRINT TOP RESULTS
    # ============================================================

    display_columns = [
        "gene",
        "lfc",
        "lfc_rank",
        "mu",
        "cipher_score",
        "cipher_rank",
    ]

    print("\n" + "=" * 90)
    print("TOP 30 GENES BY CIPHER SCORE")
    print("=" * 90)

    print(
        full_table[
            display_columns
        ]
        .head(30)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )


    # ============================================================
    # PRINT HIGHLIGHT GENES
    # ============================================================

    print("\n" + "=" * 90)
    print("HIGHLIGHT GENES")
    print("=" * 90)

    for gene in HIGHLIGHT_GENES:
        matches = full_table.loc[
            full_table["gene"]
            .str.upper()
            .eq(gene.upper())
        ]

        if matches.empty:
            print(
                f"{gene}: not found"
            )
            continue

        print(
            matches[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.6g}",
            )
        )


    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    number_with_lfc = int(
        full_table["lfc"].notna().sum()
    )

    number_with_cipher_score = int(
        full_table["cipher_score"].notna().sum()
    )

    print("\n" + "=" * 90)
    print("[DONE]")
    print("=" * 90)

    print(
        f"Number of posterior genes:       {len(full_table):,}"
    )
    print(
        f"Genes with an LFC:               {number_with_lfc:,}"
    )
    print(
        f"Genes with a CIPHER score:       {number_with_cipher_score:,}"
    )
    print(
        f"LFC ranking definition:         {lfc_rank_description}"
    )
    print(
        "CIPHER score definition:       log(1 + posterior mu)"
    )
    print(
        f"Saved TSV:                      {FINAL_TSV_PATH}"
    )
    print(
        f"Saved CSV:                      {FINAL_CSV_PATH}"
    )


def plot_posterior_vs_lfc_rank():
    # ============================================================
    # LOAD SAVED POSTERIOR/DE OR RERUN, THEN PLOT:
    #   1) posterior rank vs LFC rank
    #   2) log(1 + mu) vs log2FC, with linear LFC axis
    #
    # Highlights FN1 and IGFBP7
    # Saves PNG, PDF, SVG
    # ============================================================


    # ============================================================
    # CONFIG: match your run_pipeline settings
    # ============================================================

    H5AD_PATH = MEL_H5AD
    # H5AD_PATH = BC50_H5AD

    OUTDIR = os.path.join(BASE_OUT, "analytic_gaussian_FN1_IGFBP7_stable_diag")
    os.makedirs(OUTDIR, exist_ok=True)

    HIGHLIGHT_GENES = ["FN1", "IGFBP7"]

    DPI = 300
    POINT_SIZE = 18
    HIGHLIGHT_SIZE = 230

    POSTERIOR_SUMMARY_PATH = os.path.join(OUTDIR, "posterior_summary.tsv")
    POSTERIOR_MU_PATH = os.path.join(OUTDIR, "posterior_mu.npy")
    SELECTED_GENES_PATH = os.path.join(OUTDIR, "selected_genes.npy")

    SELECTED_DE_PATH = os.path.join(OUTDIR, "selected_de.tsv")
    ALL_GENES_DE_PATH = os.path.join(OUTDIR, "all_genes_de.tsv")

    # Rank choices.
    # For signed rank comparison:
    POSTERIOR_RANK_MODE = "mu"        # "mu" or "abs_mu"
    LFC_RANK_MODE = "log2fc"          # "log2fc" or "abs_log2fc"

    # For log(1 + mu) vs LFC:
    # log1p(mu) is defined only for mu > -1.
    # Setting this True focuses on positive posterior effects.
    POSITIVE_MU_ONLY_FOR_LOG1P_PLOT = False

    # ============================================================
    # HELPERS
    # ============================================================


    # ============================================================
    # LOAD OR RERUN
    # ============================================================

    summary = load_posterior_summary()
    de = load_de_table()

    if summary is None or de is None:
        results = maybe_rerun_pipeline()

        if summary is None:
            summary = results["summary"].copy()

        if de is None:
            if "selected_de" in results:
                de = results["selected_de"].copy()
            elif os.path.exists(SELECTED_DE_PATH):
                de = pd.read_csv(SELECTED_DE_PATH, sep="\t")
            elif os.path.exists(ALL_GENES_DE_PATH):
                de = pd.read_csv(ALL_GENES_DE_PATH, sep="\t")
            else:
                raise RuntimeError("Could not find or reconstruct DE table with log2fc.")

    # ============================================================
    # CHECK / CLEAN / MERGE
    # ============================================================

    if "gene" not in summary.columns:
        raise KeyError("posterior summary must contain a 'gene' column.")

    if "mu" not in summary.columns:
        raise KeyError("posterior summary must contain a 'mu' column.")

    if "gene" not in de.columns:
        raise KeyError("DE table must contain a 'gene' column.")

    summary = summary.copy()
    summary["gene"] = summary["gene"].astype(str)
    summary["mu"] = pd.to_numeric(summary["mu"], errors="coerce")
    summary = summary[np.isfinite(summary["mu"].values)].copy()

    if len(summary) == 0:
        raise ValueError("No finite posterior mu values found.")

    de = standardize_lfc_columns(de)
    de["gene"] = de["gene"].astype(str)
    de = de[np.isfinite(de["log2fc"].values)].copy()

    summary["gene_key"] = summary["gene"].map(_gene_key)
    de["gene_key"] = de["gene"].map(_gene_key)

    # Avoid duplicate gene merge problems.
    de_small_cols = ["gene_key", "log2fc", "abs_log2fc"]
    for extra_col in ["mean_cond0", "mean_cond1", "delta", "p_value", "p_adj", "t_stat", "abs_t"]:
        if extra_col in de.columns:
            de_small_cols.append(extra_col)

    de_small = de[de_small_cols].drop_duplicates("gene_key", keep="first").copy()

    merged = summary.merge(
        de_small,
        on="gene_key",
        how="left",
        suffixes=("", "_de"),
    )

    n_missing_lfc = int(merged["log2fc"].isna().sum())
    if n_missing_lfc > 0:
        print(f"[warning] {n_missing_lfc} posterior genes are missing log2fc and will be omitted from LFC plots.")

    merged = merged[np.isfinite(merged["mu"].values) & np.isfinite(merged["log2fc"].values)].copy()

    if len(merged) == 0:
        raise ValueError("No genes have both finite posterior mu and finite log2fc.")

    merged = add_gene_ranks(merged)

    # log(1 + mu), only valid for mu > -1
    merged["log1p_mu"] = np.nan
    valid_log1p = merged["mu"].values > -1.0
    merged.loc[valid_log1p, "log1p_mu"] = np.log1p(merged.loc[valid_log1p, "mu"].values)

    merged["signed_log1p_abs_mu"] = np.sign(merged["mu"].values) * np.log1p(np.abs(merged["mu"].values))

    n_invalid_log1p = int((~valid_log1p).sum())
    if n_invalid_log1p > 0:
        print(
            f"[warning] {n_invalid_log1p} genes have mu <= -1, so log(1 + mu) is undefined. "
            "They are omitted from the log1p(mu) plot."
        )

    # Choose rank columns.
    if POSTERIOR_RANK_MODE == "mu":
        posterior_rank_col = "rank_by_mu"
        posterior_rank_label = "Posterior rank by μ"
    elif POSTERIOR_RANK_MODE == "abs_mu":
        posterior_rank_col = "rank_by_abs_mu"
        posterior_rank_label = "Posterior rank by |μ|"
    else:
        raise ValueError("POSTERIOR_RANK_MODE must be 'mu' or 'abs_mu'.")

    if LFC_RANK_MODE == "log2fc":
        lfc_rank_col = "rank_by_log2fc"
        lfc_rank_label = "LFC rank by log2FC"
    elif LFC_RANK_MODE == "abs_log2fc":
        lfc_rank_col = "rank_by_abs_log2fc"
        lfc_rank_label = "LFC rank by |log2FC|"
    else:
        raise ValueError("LFC_RANK_MODE must be 'log2fc' or 'abs_log2fc'.")

    # Save merged table.
    out_table = os.path.join(OUTDIR, "posterior_mu_lfc_ranks.tsv")
    merged.to_csv(out_table, sep="\t", index=False)
    print(f"[saved table] {out_table}")

    # ============================================================
    # PRINT HIGHLIGHT GENE STATUS
    # ============================================================

    print("\n[highlight genes]")
    cols_to_print = [
        "gene",
        "mu",
        "log1p_mu",
        "signed_log1p_abs_mu",
        "log2fc",
        "abs_log2fc",
        "rank_by_mu",
        "rank_by_abs_mu",
        "rank_by_log2fc",
        "rank_by_abs_log2fc",
    ]

    for gene in HIGHLIGHT_GENES:
        hit = merged.loc[merged["gene"].str.upper() == gene.upper()]
        if len(hit) == 0:
            print(f"{gene}: not found after posterior/DE merge")
        else:
            print(hit[[c for c in cols_to_print if c in hit.columns]].to_string(index=False))

    # ============================================================
    # PLOT 1: POSTERIOR RANK VS LFC RANK
    # ============================================================

    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    ax.scatter(
        merged[lfc_rank_col],
        merged[posterior_rank_col],
        s=POINT_SIZE,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )

    annotate_highlights(
        ax=ax,
        df=merged,
        x_col=lfc_rank_col,
        y_col=posterior_rank_col,
        genes=HIGHLIGHT_GENES,
    )

    # Rank 1 should visually be top-left.
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlabel(lfc_rank_label)
    ax.set_ylabel(posterior_rank_label)
    ax.set_title("Posterior rank vs LFC rank")

    # Optional diagonal reference.
    max_rank = int(max(merged[lfc_rank_col].max(), merged[posterior_rank_col].max()))
    ax.plot(
        [1, max_rank],
        [1, max_rank],
        "--",
        lw=1,
        alpha=0.7,
        zorder=0,
    )

    plt.tight_layout()

    out_rank_png = os.path.join(OUTDIR, "posterior_rank_vs_lfc_rank_FN1_IGFBP7.png")
    out_rank_pdf = os.path.join(OUTDIR, "posterior_rank_vs_lfc_rank_FN1_IGFBP7.pdf")
    out_rank_svg = os.path.join(OUTDIR, "posterior_rank_vs_lfc_rank_FN1_IGFBP7.svg")

    plt.savefig(out_rank_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_rank_pdf, bbox_inches="tight")
    plt.savefig(out_rank_svg, format="svg", bbox_inches="tight")

    plt.show()

    print("\n[saved rank plot]")
    print(f"Saved: {out_rank_png}")
    print(f"Saved: {out_rank_pdf}")
    print(f"Saved: {out_rank_svg}")

    # ============================================================
    # PLOT 2: log(1 + mu) VS LFC
    # ============================================================

    plot_df = merged[np.isfinite(merged["log1p_mu"].values)].copy()

    if POSITIVE_MU_ONLY_FOR_LOG1P_PLOT:
        plot_df = plot_df[plot_df["mu"].values > 0].copy()

    if len(plot_df) == 0:
        raise ValueError("No genes available for log(1 + mu) vs LFC plot.")

    fig, ax = plt.subplots(figsize=(7.0, 5.8))

    ax.scatter(
        plot_df["log2fc"],
        plot_df["log1p_mu"],
        s=POINT_SIZE,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )

    ax.axhline(0, lw=1)
    ax.axvline(0, lw=1)

    annotate_highlights(
        ax=ax,
        df=plot_df,
        x_col="log2fc",
        y_col="log1p_mu",
        genes=HIGHLIGHT_GENES,
    )

    ax.set_xlabel("log2 fold change, Resistant / Naive")
    ax.set_ylabel("log(1 + posterior mean effect μ)")
    ax.set_title("Posterior effect vs LFC")

    # This is intentionally linear in LFC because y has already been log-transformed.
    ax.set_xscale("linear")
    ax.set_yscale("linear")

    plt.tight_layout()

    out_lfc_png = os.path.join(OUTDIR, "posterior_log1p_mu_vs_lfc_FN1_IGFBP7.png")
    out_lfc_pdf = os.path.join(OUTDIR, "posterior_log1p_mu_vs_lfc_FN1_IGFBP7.pdf")
    out_lfc_svg = os.path.join(OUTDIR, "posterior_log1p_mu_vs_lfc_FN1_IGFBP7.svg")

    plt.savefig(out_lfc_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_lfc_pdf, bbox_inches="tight")
    plt.savefig(out_lfc_svg, format="svg", bbox_inches="tight")

    plt.show()

    print("\n[saved log1p(mu) vs LFC plot]")
    print(f"Saved: {out_lfc_png}")
    print(f"Saved: {out_lfc_pdf}")
    print(f"Saved: {out_lfc_svg}")

    print("\n[DONE]")


def plot_log1p_mu_and_absz():
    # ============================================================
    # LOAD SAVED POSTERIOR OR RERUN, THEN PLOT IN SAME STYLE:
    #   1) log(1 + posterior mean mu) vs genes ranked by posterior mean mu
    #   2) |posterior z| vs genes ranked by posterior mean mu, if std/z available
    #
    # Highlights FN1 and IGFBP7
    # Saves PNG, PDF, SVG
    # ============================================================


    # ============================================================
    # CONFIG: match your run_pipeline settings
    # ============================================================

    H5AD_PATH = MEL_H5AD
    # H5AD_PATH = BC50_H5AD

    OUTDIR = os.path.join(BASE_OUT, "analytic_gaussian_FN1_IGFBP7_stable_diag")
    os.makedirs(OUTDIR, exist_ok=True)

    HIGHLIGHT_GENES = ["FN1", "IGFBP7"]

    DPI = 300

    TOP_K_LABEL = 20
    LABEL_FONTSIZE = 8
    HIGHLIGHT_FONTSIZE = 11

    POSTERIOR_SUMMARY_PATH = os.path.join(OUTDIR, "posterior_summary.tsv")
    POSTERIOR_SUMMARY_SELECTED_PATH = os.path.join(OUTDIR, "posterior_summary_selected.tsv")
    POSTERIOR_MU_PATH = os.path.join(OUTDIR, "posterior_mu.npy")
    POSTERIOR_STD_PATH = os.path.join(OUTDIR, "posterior_std.npy")
    SELECTED_GENES_PATH = os.path.join(OUTDIR, "selected_genes.npy")


    # ============================================================
    # STYLE HELPERS
    # ============================================================


    # ============================================================
    # LOAD SAVED RESULTS IF POSSIBLE
    # ============================================================

    if os.path.exists(POSTERIOR_SUMMARY_PATH):
        print(f"[load] found saved posterior summary: {POSTERIOR_SUMMARY_PATH}")
        summary = pd.read_csv(POSTERIOR_SUMMARY_PATH, sep="\t")

    elif os.path.exists(POSTERIOR_SUMMARY_SELECTED_PATH):
        print(f"[load] found saved selected posterior summary: {POSTERIOR_SUMMARY_SELECTED_PATH}")
        summary = pd.read_csv(POSTERIOR_SUMMARY_SELECTED_PATH, sep="\t")

    elif os.path.exists(POSTERIOR_MU_PATH) and os.path.exists(SELECTED_GENES_PATH):
        print("[load] found saved posterior_mu.npy and selected_genes.npy")

        mu = np.load(POSTERIOR_MU_PATH)
        genes = np.load(SELECTED_GENES_PATH, allow_pickle=True).astype(str)

        summary = pd.DataFrame({
            "gene": genes,
            "mu": mu,
        })

        if os.path.exists(POSTERIOR_STD_PATH):
            std = np.load(POSTERIOR_STD_PATH)
            summary["std"] = std
            summary["z"] = summary["mu"] / (summary["std"] + 1e-12)

    else:
        print("[saved data not found] rerunning run_pipeline(...)")

        if "run_pipeline" not in globals():
            raise RuntimeError(
                "run_pipeline(...) is not defined in this notebook/session.\n"
                "Run the full pipeline code first, then run this plotting block."
            )

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

            top_k_plot=0,
            seed=0,
        )

        summary = results["summary"].copy()


    # ============================================================
    # CLEAN SUMMARY
    # ============================================================

    if "gene" not in summary.columns:
        raise KeyError("summary must contain a 'gene' column.")

    if "mu" not in summary.columns:
        raise KeyError("summary must contain a 'mu' column.")

    summary = summary.copy()
    summary["gene"] = summary["gene"].astype(str)
    summary["gene_upper"] = summary["gene"].str.upper()
    summary["mu"] = pd.to_numeric(summary["mu"], errors="coerce")

    if "std" in summary.columns:
        summary["std"] = pd.to_numeric(summary["std"], errors="coerce")

    if "z" in summary.columns:
        summary["z"] = pd.to_numeric(summary["z"], errors="coerce")
    elif "std" in summary.columns:
        summary["z"] = summary["mu"] / (summary["std"] + 1e-12)

    summary = summary[np.isfinite(summary["mu"].values)].copy()

    if len(summary) == 0:
        raise ValueError("No finite mu values found.")

    # Save cleaned table
    clean_table_path = os.path.join(OUTDIR, "posterior_summary_clean_for_mu_rank_plots.tsv")
    summary.to_csv(clean_table_path, sep="\t", index=False)
    print(f"[saved] {clean_table_path}")


    # ============================================================
    # PRINT HIGHLIGHT GENE STATUS
    # ============================================================

    print("\n[highlight genes]")
    tmp_ranked = summary.sort_values("mu", ascending=False).reset_index(drop=True).copy()
    tmp_ranked["mu_rank"] = np.arange(1, len(tmp_ranked) + 1)
    tmp_ranked["log1p_mu"] = np.where(tmp_ranked["mu"] > -1, np.log1p(tmp_ranked["mu"]), np.nan)
    tmp_ranked["signed_log1p_abs_mu"] = np.sign(tmp_ranked["mu"]) * np.log1p(np.abs(tmp_ranked["mu"]))

    cols = ["gene", "mu_rank", "mu", "log1p_mu", "signed_log1p_abs_mu"]
    if "z" in tmp_ranked.columns:
        cols += ["z"]

    for gene in HIGHLIGHT_GENES:
        hit = tmp_ranked.loc[tmp_ranked["gene_upper"] == gene.upper()]
        if len(hit) == 0:
            print(f"{gene}: not found")
        else:
            print(hit[cols].to_string(index=False))


    # ============================================================
    # FINAL PLOTS IN MATCHED STYLE
    # ============================================================

    mu = summary["mu"].values
    gene_names = summary["gene"].values

    ranked_log1p = plot_log1p_mu_vs_mu_rank(
        mu=mu,
        gene_names=gene_names,
        genes_to_highlight=HIGHLIGHT_GENES,
        outbase=os.path.join(OUTDIR, "posterior_log1p_mu_vs_mu_rank_FN1_IGFBP7"),
        top_k_label=TOP_K_LABEL,
    )

    if "z" in summary.columns:
        ranked_absz = plot_absz_vs_mu_rank(
            mu=mu,
            z=summary["z"].values,
            gene_names=gene_names,
            genes_to_highlight=HIGHLIGHT_GENES,
            outbase=os.path.join(OUTDIR, "posterior_absz_vs_mu_rank_FN1_IGFBP7"),
            top_k_label=TOP_K_LABEL,
        )

    elif "std" in summary.columns:
        ranked_absz = plot_absz_vs_mu_rank(
            mu=mu,
            std=summary["std"].values,
            gene_names=gene_names,
            genes_to_highlight=HIGHLIGHT_GENES,
            outbase=os.path.join(OUTDIR, "posterior_absz_vs_mu_rank_FN1_IGFBP7"),
            top_k_label=TOP_K_LABEL,
        )

    else:
        print("[skip] No std or z column found, so |posterior z| plot was skipped.")


    print("\n[DONE]")


def inspect_four_conditions():
    path = FOUR_COND_H5AD
    condition_key = "Condition"

    a = ad.read_h5ad(path, backed="r")

    print("=" * 80)
    print("FILE:", path)
    print("SHAPE BEFORE ANY SUBSETTING:", a.shape)

    print("\nCONDITION COUNTS BEFORE ANY SUBSETTING:")
    print(a.obs[condition_key].astype(str).value_counts())

    print("\nNaive + Resistant only:")
    m = a.obs[condition_key].astype(str).isin(["Naive", "Resistant"])
    print(a.obs.loc[m, condition_key].astype(str).value_counts())
    print("subset shape would be:", (m.sum(), a.n_vars))


def umap_fn1_igfbp7_bc50():
    # ============================================================
    # SIMPLE UMAP WITHOUT scanpy.pp / scanpy.tl / scanpy.pl
    # Makes 3 UMAPs:
    #   1. Naive vs Resistant
    #   2. FN1 expression
    #   3. IGFBP7 expression
    # ============================================================


    # ============================================================
    # CONFIG
    # ============================================================

    H5AD_PATH = BC50_H5AD

    OUTDIR = os.path.join(BASE_OUT, "umap_FN1_IGFBP7_no_scanpy")
    os.makedirs(OUTDIR, exist_ok=True)

    CONDITION_KEY = "Condition"
    COND0 = "Naive"
    COND1 = "Resistant"

    GENES_TO_PLOT = ["FN1", "IGFBP7"]

    # Set to None for all cells.
    # Use 50000 if UMAP is slow.
    MAX_CELLS = None

    RANDOM_SEED = 0

    N_HVG = 3000
    N_PCS = 50
    N_NEIGHBORS = 15
    MIN_DIST = 0.4

    # If data are raw counts, normalize_total + log1p manually.
    # If already log-normalized, set False.
    AUTO_NORMALIZE = True
    TARGET_SUM = 1e4

    POINT_SIZE = 5
    DPI = 300


    # ============================================================
    # HELPERS
    # ============================================================


    # ============================================================
    # LOAD DATA
    # ============================================================

    check_file(H5AD_PATH)

    adata = ad.read_h5ad(H5AD_PATH)
    adata.var_names_make_unique()

    print(f"[loaded] {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    if CONDITION_KEY not in adata.obs.columns:
        raise KeyError(
            f"{CONDITION_KEY} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    print("\n[condition counts before subset]")
    print(adata.obs[CONDITION_KEY].value_counts())

    adata = adata[adata.obs[CONDITION_KEY].isin([COND0, COND1])].copy()

    print("\n[condition counts after subset]")
    print(adata.obs[CONDITION_KEY].value_counts())

    if adata.n_obs < 5:
        raise ValueError("Too few cells after subsetting.")

    # optional subsample
    if MAX_CELLS is not None and adata.n_obs > MAX_CELLS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(adata.n_obs, size=int(MAX_CELLS), replace=False)
        adata = adata[idx].copy()
        print(f"\n[subsampled] {adata.n_obs:,} cells")

    # ensure X is sparse csr or dense float
    if issparse(adata.X):
        X = adata.X.tocsr()
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    # ============================================================
    # NORMALIZE / LOG
    # ============================================================

    if AUTO_NORMALIZE and looks_like_raw_counts(X):
        print("\n[preprocess] X looks like raw counts: normalize_total + log1p")
        X_proc = normalize_total_log1p(X, target_sum=TARGET_SUM)
    else:
        print("\n[preprocess] assuming X is already normalized/log-like")
        X_proc = X.copy() if hasattr(X, "copy") else np.array(X, copy=True)

    # ============================================================
    # GENE EXPRESSION FOR PLOTTING
    # ============================================================

    expr = {}

    for gene in GENES_TO_PLOT:
        j = get_gene_idx(adata.var_names, gene)
        if j is None:
            print(f"[WARN] {gene} not found. Filling with zeros.")
            expr[gene] = np.zeros(adata.n_obs)
        else:
            expr[gene] = get_gene_expr(X_proc, j)
            print(
                f"[gene] {gene}: "
                f"mean={expr[gene].mean():.4g}, "
                f"max={expr[gene].max():.4g}, "
                f"frac>0={(expr[gene] > 0).mean():.4f}"
            )

    # ============================================================
    # HVG SELECTION
    # ============================================================

    print("\n[HVG] selecting top variable genes")
    hvg_idx, gene_mean, gene_var = select_hvgs_by_variance(X_proc, n_hvg=N_HVG)

    print(f"[HVG] selected {len(hvg_idx):,} genes")

    X_hvg = X_proc[:, hvg_idx]

    # ============================================================
    # PCA + UMAP
    # ============================================================

    print("\n[PCA] computing PCA/SVD embedding")
    Z = compute_pca_embedding(X_hvg, n_pcs=N_PCS, seed=RANDOM_SEED)
    print(f"[PCA] Z shape: {Z.shape}")

    print("\n[UMAP] computing UMAP")
    UMAP = import_umap()

    umap_model = UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_SEED,
    )

    U = umap_model.fit_transform(Z)
    print(f"[UMAP] U shape: {U.shape}")

    # ============================================================
    # PLOTTING
    # ============================================================

    condition = np.asarray(adata.obs[CONDITION_KEY].astype(str))

    is_naive = condition == COND0
    is_resistant = condition == COND1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ----------------------------
    # 1. condition UMAP
    # ----------------------------
    ax = axes[0]

    ax.scatter(
        U[is_naive, 0],
        U[is_naive, 1],
        s=POINT_SIZE,
        alpha=0.7,
        label=COND0,
    )

    ax.scatter(
        U[is_resistant, 0],
        U[is_resistant, 1],
        s=POINT_SIZE,
        alpha=0.7,
        label=COND1,
    )

    ax.set_title(f"{COND0} vs {COND1}")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    # ----------------------------
    # 2. FN1 expression UMAP
    # ----------------------------
    ax = axes[1]

    fn1 = expr["FN1"]
    vmax_fn1 = np.percentile(fn1, 99)

    sc1 = ax.scatter(
        U[:, 0],
        U[:, 1],
        c=np.clip(fn1, 0, vmax_fn1),
        s=POINT_SIZE,
        alpha=0.85,
        cmap="viridis",
    )

    ax.set_title("FN1 expression")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    cb1 = plt.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label("Expression")

    # ----------------------------
    # 3. IGFBP7 expression UMAP
    # ----------------------------
    ax = axes[2]

    igfbp7 = expr["IGFBP7"]
    vmax_igfbp7 = np.percentile(igfbp7, 99)

    sc2 = ax.scatter(
        U[:, 0],
        U[:, 1],
        c=np.clip(igfbp7, 0, vmax_igfbp7),
        s=POINT_SIZE,
        alpha=0.85,
        cmap="viridis",
    )

    ax.set_title("IGFBP7 expression")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    cb2 = plt.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label("Expression")

    plt.tight_layout()

    out_png = os.path.join(OUTDIR, "UMAP_condition_FN1_IGFBP7.png")
    out_pdf = os.path.join(OUTDIR, "UMAP_condition_FN1_IGFBP7.pdf")

    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    print("\n[DONE]")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def umap_fn1_igfbp7_melanoma():
    # ============================================================
    # SIMPLE UMAP WITHOUT scanpy.pp / scanpy.tl / scanpy.pl
    # Makes 3 UMAPs:
    #   1. Naive vs Resistant
    #   2. FN1 expression
    #   3. IGFBP7 expression
    # ============================================================


    # ============================================================
    # CONFIG
    # ============================================================

    H5AD_PATH = MEL_H5AD

    OUTDIR = os.path.join(BASE_OUT, "umap_FN1_IGFBP7_no_scanpy")
    os.makedirs(OUTDIR, exist_ok=True)

    CONDITION_KEY = "Condition"
    COND0 = "Naive"
    COND1 = "Resistant"

    GENES_TO_PLOT = ["FN1", "IGFBP7"]

    # Set to None for all cells.
    # Use 50000 if UMAP is slow.
    MAX_CELLS = None

    RANDOM_SEED = 0

    N_HVG = 3000
    N_PCS = 50
    N_NEIGHBORS = 15
    MIN_DIST = 0.4

    # If data are raw counts, normalize_total + log1p manually.
    # If already log-normalized, set False.
    AUTO_NORMALIZE = True
    TARGET_SUM = 1e4

    POINT_SIZE = 5
    DPI = 300


    # ============================================================
    # HELPERS
    # ============================================================


    # ============================================================
    # LOAD DATA
    # ============================================================

    check_file(H5AD_PATH)

    adata = ad.read_h5ad(H5AD_PATH)
    adata.var_names_make_unique()

    print(f"[loaded] {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    if CONDITION_KEY not in adata.obs.columns:
        raise KeyError(
            f"{CONDITION_KEY} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    print("\n[condition counts before subset]")
    print(adata.obs[CONDITION_KEY].value_counts())

    adata = adata[adata.obs[CONDITION_KEY].isin([COND0, COND1])].copy()

    print("\n[condition counts after subset]")
    print(adata.obs[CONDITION_KEY].value_counts())

    if adata.n_obs < 5:
        raise ValueError("Too few cells after subsetting.")

    # optional subsample
    if MAX_CELLS is not None and adata.n_obs > MAX_CELLS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(adata.n_obs, size=int(MAX_CELLS), replace=False)
        adata = adata[idx].copy()
        print(f"\n[subsampled] {adata.n_obs:,} cells")

    # ensure X is sparse csr or dense float
    if issparse(adata.X):
        X = adata.X.tocsr()
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    # ============================================================
    # NORMALIZE / LOG
    # ============================================================

    if AUTO_NORMALIZE and looks_like_raw_counts(X):
        print("\n[preprocess] X looks like raw counts: normalize_total + log1p")
        X_proc = normalize_total_log1p(X, target_sum=TARGET_SUM)
    else:
        print("\n[preprocess] assuming X is already normalized/log-like")
        X_proc = X.copy() if hasattr(X, "copy") else np.array(X, copy=True)

    # ============================================================
    # GENE EXPRESSION FOR PLOTTING
    # ============================================================

    expr = {}

    for gene in GENES_TO_PLOT:
        j = get_gene_idx(adata.var_names, gene)
        if j is None:
            print(f"[WARN] {gene} not found. Filling with zeros.")
            expr[gene] = np.zeros(adata.n_obs)
        else:
            expr[gene] = get_gene_expr(X_proc, j)
            print(
                f"[gene] {gene}: "
                f"mean={expr[gene].mean():.4g}, "
                f"max={expr[gene].max():.4g}, "
                f"frac>0={(expr[gene] > 0).mean():.4f}"
            )

    # ============================================================
    # HVG SELECTION
    # ============================================================

    print("\n[HVG] selecting top variable genes")
    hvg_idx, gene_mean, gene_var = select_hvgs_by_variance(X_proc, n_hvg=N_HVG)

    print(f"[HVG] selected {len(hvg_idx):,} genes")

    X_hvg = X_proc[:, hvg_idx]

    # ============================================================
    # PCA + UMAP
    # ============================================================

    print("\n[PCA] computing PCA/SVD embedding")
    Z = compute_pca_embedding(X_hvg, n_pcs=N_PCS, seed=RANDOM_SEED)
    print(f"[PCA] Z shape: {Z.shape}")

    print("\n[UMAP] computing UMAP")
    UMAP = import_umap()

    umap_model = UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_SEED,
    )

    U = umap_model.fit_transform(Z)
    print(f"[UMAP] U shape: {U.shape}")

    # ============================================================
    # PLOTTING
    # ============================================================

    condition = np.asarray(adata.obs[CONDITION_KEY].astype(str))

    is_naive = condition == COND0
    is_resistant = condition == COND1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ----------------------------
    # 1. condition UMAP
    # ----------------------------
    ax = axes[0]

    ax.scatter(
        U[is_naive, 0],
        U[is_naive, 1],
        s=POINT_SIZE,
        alpha=0.7,
        label=COND0,
    )

    ax.scatter(
        U[is_resistant, 0],
        U[is_resistant, 1],
        s=POINT_SIZE,
        alpha=0.7,
        label=COND1,
    )

    ax.set_title(f"{COND0} vs {COND1}")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    # ----------------------------
    # 2. FN1 expression UMAP
    # ----------------------------
    ax = axes[1]

    fn1 = expr["FN1"]
    vmax_fn1 = np.percentile(fn1, 99)

    sc1 = ax.scatter(
        U[:, 0],
        U[:, 1],
        c=np.clip(fn1, 0, vmax_fn1),
        s=POINT_SIZE,
        alpha=0.85,
        cmap="viridis",
    )

    ax.set_title("FN1 expression")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    cb1 = plt.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label("Expression")

    # ----------------------------
    # 3. IGFBP7 expression UMAP
    # ----------------------------
    ax = axes[2]

    igfbp7 = expr["IGFBP7"]
    vmax_igfbp7 = np.percentile(igfbp7, 99)

    sc2 = ax.scatter(
        U[:, 0],
        U[:, 1],
        c=np.clip(igfbp7, 0, vmax_igfbp7),
        s=POINT_SIZE,
        alpha=0.85,
        cmap="viridis",
    )

    ax.set_title("IGFBP7 expression")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    cb2 = plt.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label("Expression")

    plt.tight_layout()

    out_png = os.path.join(OUTDIR, "UMAP_condition_FN1_IGFBP7.png")
    out_pdf = os.path.join(OUTDIR, "UMAP_condition_FN1_IGFBP7.pdf")

    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    print("\n[DONE]")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def umap_fn1_igfbp7_variant():
    # ============================================================
    # SIMPLE UMAP WITHOUT scanpy.pp / scanpy.tl / scanpy.pl
    # Saves PNG, PDF, SVG
    # Scatter points are rasterized for smaller SVG/PDF files
    # Makes 3 UMAPs:
    #   1. Naive vs Resistant
    #   2. FN1 expression
    #   3. IGFBP7 expression
    # ============================================================


    # ============================================================
    # CONFIG
    # ============================================================

    H5AD_PATH = MEL_H5AD

    OUTDIR = os.path.join(BASE_OUT, "umap_FN1_IGFBP7_no_scanpy")
    os.makedirs(OUTDIR, exist_ok=True)

    CONDITION_KEY = "Condition"
    COND0 = "Naive"
    COND1 = "Resistant"

    GENES_TO_PLOT = ["FN1", "IGFBP7"]

    MAX_CELLS = None  # set to 50000 if slow

    RANDOM_SEED = 0

    N_HVG = 3000
    N_PCS = 50
    N_NEIGHBORS = 15
    MIN_DIST = 0.4

    AUTO_NORMALIZE = True
    TARGET_SUM = 1e4

    POINT_SIZE = 5
    DPI = 300

    # ============================================================
    # HELPERS
    # ============================================================


    # ============================================================
    # LOAD DATA
    # ============================================================

    check_file(H5AD_PATH)

    adata = ad.read_h5ad(H5AD_PATH)
    adata.var_names_make_unique()

    print(f"[loaded] {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    if CONDITION_KEY not in adata.obs.columns:
        raise KeyError(
            f"{CONDITION_KEY} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    print("\n[condition counts before subset]")
    print(adata.obs[CONDITION_KEY].value_counts())

    adata = adata[adata.obs[CONDITION_KEY].isin([COND0, COND1])].copy()

    print("\n[condition counts after subset]")
    print(adata.obs[CONDITION_KEY].value_counts())

    if adata.n_obs < 5:
        raise ValueError("Too few cells after subsetting.")

    if MAX_CELLS is not None and adata.n_obs > MAX_CELLS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(adata.n_obs, size=int(MAX_CELLS), replace=False)
        adata = adata[idx].copy()
        print(f"\n[subsampled] {adata.n_obs:,} cells")

    if issparse(adata.X):
        X = adata.X.tocsr()
    else:
        X = np.asarray(adata.X, dtype=np.float32)

    # ============================================================
    # NORMALIZE / LOG
    # ============================================================

    if AUTO_NORMALIZE and looks_like_raw_counts(X):
        print("\n[preprocess] X looks like raw counts: normalize_total + log1p")
        X_proc = normalize_total_log1p(X, target_sum=TARGET_SUM)
    else:
        print("\n[preprocess] assuming X is already normalized/log-like")
        X_proc = X.copy() if hasattr(X, "copy") else np.array(X, copy=True)

    # ============================================================
    # GENE EXPRESSION FOR PLOTTING
    # ============================================================

    expr = {}

    for gene in GENES_TO_PLOT:
        j = get_gene_idx(adata.var_names, gene)

        if j is None:
            print(f"[WARN] {gene} not found. Filling with zeros.")
            expr[gene] = np.zeros(adata.n_obs)
        else:
            expr[gene] = get_gene_expr(X_proc, j)
            print(
                f"[gene] {gene}: "
                f"mean={expr[gene].mean():.4g}, "
                f"max={expr[gene].max():.4g}, "
                f"frac>0={(expr[gene] > 0).mean():.4f}"
            )

    # ============================================================
    # HVG SELECTION
    # ============================================================

    print("\n[HVG] selecting top variable genes")
    hvg_idx, gene_mean, gene_var = select_hvgs_by_variance(X_proc, n_hvg=N_HVG)

    print(f"[HVG] selected {len(hvg_idx):,} genes")

    X_hvg = X_proc[:, hvg_idx]

    # ============================================================
    # PCA + UMAP
    # ============================================================

    print("\n[PCA] computing PCA/SVD embedding")
    Z = compute_pca_embedding(X_hvg, n_pcs=N_PCS, seed=RANDOM_SEED)
    print(f"[PCA] Z shape: {Z.shape}")

    print("\n[UMAP] computing UMAP")
    UMAP = import_umap()

    umap_model = UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_SEED,
    )

    U = umap_model.fit_transform(Z)
    print(f"[UMAP] U shape: {U.shape}")

    # ============================================================
    # PLOTTING
    # ============================================================

    condition = np.asarray(adata.obs[CONDITION_KEY].astype(str))

    is_naive = condition == COND0
    is_resistant = condition == COND1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ----------------------------
    # 1. condition UMAP
    # ----------------------------
    ax = axes[0]

    ax.scatter(
        U[is_naive, 0],
        U[is_naive, 1],
        s=POINT_SIZE,
        alpha=0.7,
        label=COND0,
        rasterized=True,
    )

    ax.scatter(
        U[is_resistant, 0],
        U[is_resistant, 1],
        s=POINT_SIZE,
        alpha=0.7,
        label=COND1,
        rasterized=True,
    )

    ax.set_title(f"{COND0} vs {COND1}")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    # ----------------------------
    # 2. FN1 expression UMAP
    # ----------------------------
    ax = axes[1]

    fn1 = expr["FN1"]
    vmax_fn1 = np.percentile(fn1, 99)

    sc1 = ax.scatter(
        U[:, 0],
        U[:, 1],
        c=np.clip(fn1, 0, vmax_fn1),
        s=POINT_SIZE,
        alpha=0.85,
        cmap="viridis",
        rasterized=True,
    )

    ax.set_title("FN1 expression")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])

    cb1 = plt.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label("Expression")

    # ----------------------------
    # 3. IGFBP7 expression UMAP
    # ----------------------------
    ax = axes[2]

    igfbp7 = expr["IGFBP7"]
    vmax_igfbp7 = np.percentile(igfbp7, 99)

    sc2 = ax.scatter(
        U[:, 0],
        U[:, 1],
        c=np.clip(igfbp7, 0, vmax_igfbp7),
        s=POINT_SIZE,
        alpha=0.85,
        cmap="viridis",
        rasterized=True,
    )

    ax.set_title("IGFBP7 expression")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])

    cb2 = plt.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label("Expression")

    plt.tight_layout()

    # ============================================================
    # SAVE FIGURE
    # ============================================================

    out_png = os.path.join(OUTDIR, "UMAP_condition_FN1_IGFBP7.png")
    out_pdf = os.path.join(OUTDIR, "UMAP_condition_FN1_IGFBP7.pdf")
    out_svg = os.path.join(OUTDIR, "UMAP_condition_FN1_IGFBP7.svg")

    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_svg, format="svg", bbox_inches="tight")

    plt.show()

    print("\n[DONE]")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")


def volcano_bc50_allgenes():
    # ============================================================
    # SIMPLE ALL-GENE VOLCANO PLOT: log2FC vs -log10(p-value)
    # No scanpy required
    # ============================================================


    # ============================================================
    # CONFIG
    # ============================================================

    H5AD_PATH = BC50_H5AD

    OUTDIR = os.path.join(BASE_OUT, "volcano_all_genes_FN1_IGFBP7")
    os.makedirs(OUTDIR, exist_ok=True)

    CONDITION_KEY = "Condition"
    COND0 = "Naive"
    COND1 = "Resistant"

    # Volcano is COND1 - COND0
    # positive log2FC = higher in Resistant
    HIGHLIGHT_GENES = ["FN1", "IGFBP7"]

    LOGFC_PSEUDOCOUNT = 0.0001

    FDR_ALPHA = 0.05
    PVAL_ALPHA = 0.05
    MIN_ABS_LOG2FC_FOR_COLOR = 1.0

    TOP_N_LABEL = 10
    DPI = 300

    # Use all genes, or set simple filters if you want.
    MIN_MEAN = 0.0
    MIN_FRAC_ON = 0.0
    MIN_EXPR_ON = 0.0

    # ============================================================
    # HELPERS
    # ============================================================


    # ============================================================
    # LOAD
    # ============================================================

    check_file(H5AD_PATH)

    adata = ad.read_h5ad(H5AD_PATH)
    adata.var_names_make_unique()

    print(f"[loaded] {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    if CONDITION_KEY not in adata.obs.columns:
        raise KeyError(
            f"{CONDITION_KEY} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    print("\n[condition counts before subset]")
    print(adata.obs[CONDITION_KEY].value_counts())

    adata = adata[adata.obs[CONDITION_KEY].isin([COND0, COND1])].copy()

    print("\n[condition counts after subset]")
    print(adata.obs[CONDITION_KEY].value_counts())

    cond = np.asarray(adata.obs[CONDITION_KEY].astype(str))
    m0 = cond == COND0
    m1 = cond == COND1

    n0 = int(m0.sum())
    n1 = int(m1.sum())

    print(f"\n[contrast] {COND1} - {COND0}")
    print(f"{COND0}: {n0:,} cells")
    print(f"{COND1}: {n1:,} cells")

    if n0 < 2 or n1 < 2:
        raise ValueError("Need at least 2 cells in each condition.")

    X0 = adata.X[m0]
    X1 = adata.X[m1]

    gene_names = np.asarray(adata.var_names, dtype=str)

    # ============================================================
    # COMPUTE ALL-GENE DE
    # ============================================================

    print("\n[DE] computing means/variances")
    mean0, var0 = sparse_or_dense_mean_var(X0)
    mean1, var1 = sparse_or_dense_mean_var(X1)

    print("[DE] computing detection fractions")
    frac0 = sparse_or_dense_frac_on(X0, min_expr=MIN_EXPR_ON)
    frac1 = sparse_or_dense_frac_on(X1, min_expr=MIN_EXPR_ON)

    mean_all = (mean0 * n0 + mean1 * n1) / (n0 + n1)
    frac_all = (frac0 * n0 + frac1 * n1) / (n0 + n1)

    keep = (
        np.isfinite(mean_all)
        & np.isfinite(frac_all)
        & (mean_all >= MIN_MEAN)
        & (frac_all >= MIN_FRAC_ON)
    )

    print(f"[filter] kept {keep.sum():,} / {len(keep):,} genes for volcano")

    gene_names = gene_names[keep]
    mean0 = mean0[keep]
    mean1 = mean1[keep]
    var0 = var0[keep]
    var1 = var1[keep]
    frac0 = frac0[keep]
    frac1 = frac1[keep]
    mean_all = mean_all[keep]
    frac_all = frac_all[keep]

    delta = mean1 - mean0

    with np.errstate(divide="ignore", invalid="ignore"):
        log2fc = np.log2((mean1 + LOGFC_PSEUDOCOUNT) / (mean0 + LOGFC_PSEUDOCOUNT))

    log2fc = np.nan_to_num(log2fc, nan=0.0, posinf=0.0, neginf=0.0)

    print("[DE] computing Welch p-values")
    t_stat, pval = welch_de_from_moments(mean0, var0, n0, mean1, var1, n1)

    _, p_adj, _, _ = multipletests(pval, alpha=FDR_ALPHA, method="fdr_bh")

    neglog10_p = -np.log10(np.maximum(pval, 1e-300))
    neglog10_padj = -np.log10(np.maximum(p_adj, 1e-300))

    de = pd.DataFrame({
        "gene": gene_names,
        "mean_naive": mean0,
        "mean_resistant": mean1,
        "delta_resistant_minus_naive": delta,
        "log2fc_resistant_minus_naive": log2fc,
        "abs_log2fc": np.abs(log2fc),
        "frac_on_naive": frac0,
        "frac_on_resistant": frac1,
        "mean_all": mean_all,
        "frac_on_all": frac_all,
        "t_stat": t_stat,
        "p_value": pval,
        "p_adj": p_adj,
        "neglog10_p": neglog10_p,
        "neglog10_padj": neglog10_padj,
    })

    de["sig_fdr"] = de["p_adj"] < FDR_ALPHA
    de["sig_pval"] = de["p_value"] < PVAL_ALPHA

    de["volcano_class"] = "not significant"
    de.loc[
        (de["p_adj"] < FDR_ALPHA) &
        (de["log2fc_resistant_minus_naive"] >= MIN_ABS_LOG2FC_FOR_COLOR),
        "volcano_class"
    ] = f"higher in {COND1}"

    de.loc[
        (de["p_adj"] < FDR_ALPHA) &
        (de["log2fc_resistant_minus_naive"] <= -MIN_ABS_LOG2FC_FOR_COLOR),
        "volcano_class"
    ] = f"higher in {COND0}"

    de = de.sort_values(["p_adj", "abs_log2fc"], ascending=[True, False]).reset_index(drop=True)

    out_tsv = os.path.join(OUTDIR, "all_genes_DE_volcano_stats.tsv")
    de.to_csv(out_tsv, sep="\t", index=False)

    print(f"\n[saved stats] {out_tsv}")

    print("\n[top genes by adjusted p-value]")
    print(
        de[[
            "gene",
            "mean_naive",
            "mean_resistant",
            "delta_resistant_minus_naive",
            "log2fc_resistant_minus_naive",
            "p_value",
            "p_adj",
        ]].head(30).to_string(index=False)
    )

    print("\n[highlight genes]")
    for g in HIGHLIGHT_GENES:
        hit = de.loc[de["gene"].str.upper() == g.upper()]
        if len(hit) == 0:
            print(f"{g}: not found")
        else:
            print(hit[[
                "gene",
                "mean_naive",
                "mean_resistant",
                "delta_resistant_minus_naive",
                "log2fc_resistant_minus_naive",
                "p_value",
                "p_adj",
            ]].to_string(index=False))


    # ============================================================
    # VOLCANO PLOT
    # ============================================================

    x = de["log2fc_resistant_minus_naive"].values
    y = de["neglog10_p"].values

    cls = de["volcano_class"].values

    fig, ax = plt.subplots(figsize=(8, 7))

    # background
    m_bg = cls == "not significant"
    ax.scatter(
        x[m_bg],
        y[m_bg],
        s=8,
        alpha=0.35,
        linewidths=0,
        label="not significant",
    )

    # higher in Naive
    label_naive = f"higher in {COND0}"
    m_naive = cls == label_naive
    ax.scatter(
        x[m_naive],
        y[m_naive],
        s=10,
        alpha=0.75,
        linewidths=0,
        label=label_naive,
    )

    # higher in Resistant
    label_res = f"higher in {COND1}"
    m_res = cls == label_res
    ax.scatter(
        x[m_res],
        y[m_res],
        s=10,
        alpha=0.75,
        linewidths=0,
        label=label_res,
    )

    # threshold lines
    ax.axvline(MIN_ABS_LOG2FC_FOR_COLOR, linestyle="--", linewidth=1)
    ax.axvline(-MIN_ABS_LOG2FC_FOR_COLOR, linestyle="--", linewidth=1)
    ax.axhline(-np.log10(PVAL_ALPHA), linestyle="--", linewidth=1)

    # label top genes by p-value
    top = de.head(min(TOP_N_LABEL, len(de))).copy()
    for _, row in top.iterrows():
        label_gene(
            ax,
            {
                "gene": row["gene"],
                "log2fc": row["log2fc_resistant_minus_naive"],
                "neglog10_p": row["neglog10_p"],
            },
            fontsize=8,
        )

    # highlight FN1 / IGFBP7
    for gene in HIGHLIGHT_GENES:
        hit = de.loc[de["gene"].str.upper() == gene.upper()]
        if len(hit) == 0:
            continue

        row = hit.iloc[0]

        ax.scatter(
            row["log2fc_resistant_minus_naive"],
            row["neglog10_p"],
            s=220,
            marker="*",
            edgecolor="black",
            linewidth=0.8,
            zorder=20,
        )

        ax.text(
            row["log2fc_resistant_minus_naive"],
            row["neglog10_p"],
            f" {gene}",
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=21,
        )

    ax.set_xlabel(f"log2FC ({COND1} / {COND0})")
    ax.set_ylabel("-log10 p-value")
    ax.set_title(f"Volcano plot: {COND1} vs {COND0}")

    ax.legend(frameon=False, loc="best")

    plt.tight_layout()

    out_png = os.path.join(OUTDIR, "volcano_log2fc_pvalue_all_genes.png")
    out_pdf = os.path.join(OUTDIR, "volcano_log2fc_pvalue_all_genes.pdf")

    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    print("\n[DONE]")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_tsv}")

