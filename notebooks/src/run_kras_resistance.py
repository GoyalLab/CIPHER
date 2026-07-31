"""Notebook run-module for ``notebooks/suppl/kras_resistance_figM7_S17.ipynb``.

Holds the main-flow of the Fig M7 / Fig S17 KRAS-resistance supplement,
relocated out of the notebook so the notebook is a thin driver (markdown headers,
one config cell, one call per figure panel). Each function is one notebook code
cell; configuration is read as module globals injected at runtime by the notebook
(``R.__dict__.update(_cfg)``). Notebook-only; NOT shipped with the installable
``cipher`` package.
"""
from src.suppl_resistance import *

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import anndata as ad


def run_gaussian_pipeline():
    TAU2 = 1e-0

    PIPELINE_OUTDIR = os.path.join(_BASE, f"analytic_gaussian_top_sig_highlfc_{tau2_to_label(TAU2)}")

    results = run_top_sig_highlfc_gaussian_pipeline(
        h5ad_path=H5AD_PATH,
        outdir=PIPELINE_OUTDIR,
        condition_key="Condition",
        cond0="Naive",
        cond1="Resistant",
        genes_to_check=[
            "ANKRD1", "IGFN1", "PSG9", "PSG4", "SAMD9",
            "TSPAN12", "TNNT2", "ATXN1", "TGFA", "ROBO2",
            "DACT1", "SYBU", "MAPK4", "APOL1", "CDKN1A",
            "STX11", "PARP8", "ATF3", "KRTAP5-AS1", "ATRNL1","GBP3", "HS6ST3",
        ],
        top_n_de=2000,
        min_abs_log2fc=0.5,
        drop_housekeeping=True,
        Sigma_shrinkage=1e-6,
        H_shrinkage=1e-6,
        H_ridge=1e-6,
        H_mode="naive",
        tau2=TAU2,
        min_cells_frac=0.00001,
        min_expr=0.01,
        hi_quantile=1.0,
        var_drop_q=1.0,
        filter_subsample_cells=0,
        seed=0,
        top_k_plot=0,

        # False = label only genes_to_check on the rank plot.
        # True = label every selected gene only when selected set <= max_all_labels.
        label_all_selected_genes=True,
    )


def fig_posterior_score_curve():
    OUTDIR = Path(os.path.join(_BASE, "analytic_gaussian_top_sig_highlfc_tau2_1p0"))

    SUMMARY_PATH = OUTDIR / "posterior_summary_selected.tsv"

    GENES_TO_LABEL = [
        "ANKRD1",
        "IGFN1",
        "PSG9",
        "PSG4",
        "SAMD9",
        "TSPAN12",
        "TNNT2",
        "ATXN1",
        "TGFA",
        "ROBO2",
        "DACT1",
        "SYBU",
        "MAPK4",
        "APOL1",
        "CDKN1A",
        "STX11",
        "PARP8",
        "ATF3",
        "KRTAP5-AS1",
        "ATRNL1",
        "GBP3",
        "HS6ST3",
    ]

    FIGSIZE = (20, 9)

    DPI = 300

    LINE_WIDTH = 1.35

    POINT_SIZE = 12

    HIGHLIGHT_SIZE = 150

    LABEL_FONTSIZE = 10

    LABEL_ROWS = 3

    LABEL_BAND_GAP_FRACTION = 0.09

    LABEL_ROW_SPACING_FRACTION = 0.085

    X_LEFT_MARGIN_FRACTION = 0.035

    X_RIGHT_MARGIN_FRACTION = 0.045

    OUTPUT_STEM = (
        OUTDIR
        / "posterior_log1p_abs_mu_vs_abs_mu_rank_labels_above_curve"
    )

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Could not find the saved posterior table:\n"
            f"  {SUMMARY_PATH}\n\n"
            "Run the posterior-fitting script first or update OUTDIR."
        )

    print(f"[load] {SUMMARY_PATH}")

    summary = pd.read_csv(
        SUMMARY_PATH,
        sep="\t",
        low_memory=False,
    )

    required_columns = {"gene", "mu"}

    missing_columns = required_columns.difference(summary.columns)

    if missing_columns:
        raise KeyError(
            "Saved posterior table is missing required columns: "
            f"{sorted(missing_columns)}\n"
            f"Available columns: {list(summary.columns)}"
        )

    summary["gene"] = summary["gene"].astype(str)

    summary["gene_upper"] = summary["gene"].map(normalize_gene_name)

    summary["mu"] = pd.to_numeric(summary["mu"], errors="coerce")

    summary["abs_mu"] = np.abs(summary["mu"])

    summary["log1p_abs_mu"] = np.log1p(summary["abs_mu"])

    finite_mask = (
        np.isfinite(summary["mu"])
        & np.isfinite(summary["abs_mu"])
        & np.isfinite(summary["log1p_abs_mu"])
    )

    n_nonfinite = int((~finite_mask).sum())

    if n_nonfinite > 0:
        print(
            f"[warning] dropping {n_nonfinite} genes with nonfinite "
            "posterior values."
        )

    plot_df = summary.loc[finite_mask].copy()

    if len(plot_df) == 0:
        raise ValueError("No genes have finite posterior mu values.")

    plot_df = plot_df.sort_values(
        ["abs_mu", "gene"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    plot_df["abs_mu_rank"] = np.arange(
        1,
        len(plot_df) + 1,
    )

    x = plot_df["abs_mu_rank"].to_numpy(dtype=float)

    y = plot_df["log1p_abs_mu"].to_numpy(dtype=float)

    gene_names = plot_df["gene"].to_numpy(dtype=str)

    gene_names_upper = plot_df["gene_upper"].to_numpy(dtype=str)

    requested_gene_set = {
        normalize_gene_name(gene)
        for gene in GENES_TO_LABEL
    }

    label_indices = np.where(
        np.isin(
            gene_names_upper,
            list(requested_gene_set),
        )
    )[0]

    present_gene_set = set(
        gene_names_upper[label_indices]
    )

    missing_genes = [
        gene
        for gene in GENES_TO_LABEL
        if normalize_gene_name(gene) not in present_gene_set
    ]

    print(f"[plot] genes available: {len(plot_df)}")

    print(f"[plot] requested labels: {len(GENES_TO_LABEL)}")

    print(f"[plot] labels found: {len(label_indices)}")

    if len(label_indices) > 0:
        print(
            "[plot] labeled genes:",
            ", ".join(gene_names[label_indices]),
        )

    if missing_genes:
        print(
            "[warning] requested genes absent from saved posterior table:",
            ", ".join(missing_genes),
        )

    n_genes = len(plot_df)

    curve_ymax = float(np.nanmax(y))

    curve_ymin = float(np.nanmin(y))

    y_scale = max(
        curve_ymax - curve_ymin,
        curve_ymax,
        1.0,
    )

    band_gap = max(
        0.65,
        LABEL_BAND_GAP_FRACTION * y_scale,
    )

    row_spacing = max(
        0.58,
        LABEL_ROW_SPACING_FRACTION * y_scale,
    )

    x_left_margin = max(
        30.0,
        X_LEFT_MARGIN_FRACTION * n_genes,
    )

    x_right_margin = max(
        60.0,
        X_RIGHT_MARGIN_FRACTION * n_genes,
    )

    x_min = 1.0 - x_left_margin

    x_max = float(n_genes) + x_right_margin

    label_layout = assign_labels_to_top_band(
        x=x,
        y=y,
        gene_names=gene_names,
        label_indices=label_indices,
        n_rows=LABEL_ROWS,
        curve_ymax=curve_ymax,
        band_gap=band_gap,
        row_spacing=row_spacing,
        x_min=x_min,
        x_max=x_max,
    )

    if len(label_layout) > 0:
        label_band_top = float(label_layout["label_y"].max())
    else:
        label_band_top = curve_ymax + band_gap

    y_bottom = -max(
        0.25,
        0.025 * y_scale,
    )

    y_top = label_band_top + max(
        0.75,
        0.08 * y_scale,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)

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
        alpha=0.28,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    if len(label_indices) > 0:
        ax.scatter(
            x[label_indices],
            y[label_indices],
            s=HIGHLIGHT_SIZE,
            marker="*",
            edgecolor="black",
            linewidth=0.45,
            zorder=12,
        )

    label_boundary_y = curve_ymax + 0.50 * band_gap

    ax.axhline(
        label_boundary_y,
        linewidth=0.8,
        linestyle=":",
        alpha=0.45,
        zorder=1,
    )

    for row in label_layout.itertuples(index=False):
        ax.annotate(
            row.gene,

            # Actual posterior point.
            xy=(
                row.point_x,
                row.point_y,
            ),

            # Label location in the dedicated top band.
            xytext=(
                row.label_x,
                row.label_y,
            ),

            textcoords="data",
            ha="center",
            va="center",
            fontsize=LABEL_FONTSIZE,
            zorder=30,
            clip_on=False,

            # White box ensures that every gene name is readable,
            # even where leader lines pass behind it.
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "0.70",
                "linewidth": 0.55,
                "alpha": 1.0,
            },

            # Leader line connecting the label to its posterior point.
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.75,
                "color": "0.35",
                "alpha": 0.75,
                "shrinkA": 3,
                "shrinkB": 6,
                "connectionstyle": "arc3,rad=0.0",
            },
        )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_bottom,
        y_top,
    )

    ax.axhline(
        0,
        linewidth=0.85,
        alpha=0.55,
        zorder=1,
    )

    ax.set_xlabel(
        r"Rank by $|\mathrm{posterior\ mean}\ \mu|$",
        fontsize=12,
    )

    ax.set_ylabel(
        r"$\log(1+|\mathrm{posterior\ mean}\ \mu|)$",
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

    ax.grid(False)

    plt.tight_layout()

    png_path = OUTPUT_STEM.with_suffix(".png")

    pdf_path = OUTPUT_STEM.with_suffix(".pdf")

    svg_path = OUTPUT_STEM.with_suffix(".svg")

    table_path = (
        OUTPUT_STEM.parent
        / f"{OUTPUT_STEM.name}_table.tsv"
    )

    label_layout_path = (
        OUTPUT_STEM.parent
        / f"{OUTPUT_STEM.name}_label_layout.tsv"
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
        bbox_inches="tight",
    )

    plot_df["labeled"] = np.isin(
        plot_df["gene_upper"],
        list(requested_gene_set),
    )

    plot_df[
        [
            "abs_mu_rank",
            "gene",
            "mu",
            "abs_mu",
            "log1p_abs_mu",
            "labeled",
        ]
    ].to_csv(
        table_path,
        sep="\t",
        index=False,
    )

    label_layout.to_csv(
        label_layout_path,
        sep="\t",
        index=False,
    )

    print(f"[saved] {png_path}")

    print(f"[saved] {pdf_path}")

    print(f"[saved] {svg_path}")

    print(f"[saved] {table_path}")

    print(f"[saved] {label_layout_path}")

    plt.show()


def fig_cipher_rank_lfc_curve():
    OUTDIR = Path(
        os.path.join(_BASE, "analytic_gaussian_top_sig_highlfc_tau2_1p0")
    )

    SUMMARY_PATH = (
        OUTDIR
        / "posterior_summary_selected.tsv"
    )

    GENES_TO_LABEL = [
        "ANKRD1",
        "IGFN1",
        "PSG9",
        "PSG4",
        "SAMD9",
        "TSPAN12",
        "TNNT2",
        "ATXN1",
        "TGFA",
        "ROBO2",
        "DACT1",
        "SYBU",
        "MAPK4",
        "APOL1",
        "CDKN1A",
        "STX11",
        "PARP8",
        "ATF3",
        "KRTAP5-AS1",
        "ATRNL1",
        "GBP3",
        "HS6ST3",
    ]

    FIGSIZE = (20, 9)

    DPI = 300

    LINE_WIDTH = 1.35

    POINT_SIZE = 12

    HIGHLIGHT_SIZE = 150

    LABEL_FONTSIZE = 10

    LABEL_ROWS = 3

    LABEL_BAND_GAP_FRACTION = 0.09

    LABEL_ROW_SPACING_FRACTION = 0.085

    X_LEFT_MARGIN_FRACTION = 0.035

    X_RIGHT_MARGIN_FRACTION = 0.045

    OUTPUT_STEM = (
        OUTDIR
        / "posterior_log1p_abs_mu_vs_abs_mu_rank_labels_above_curve"
    )

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Could not find the saved posterior table:\n"
            f"  {SUMMARY_PATH}\n\n"
            "Run the posterior-fitting script first or update OUTDIR."
        )

    print(f"[load] {SUMMARY_PATH}")

    summary = pd.read_csv(
        SUMMARY_PATH,
        sep="\t",
        low_memory=False,
    )

    required_columns = {
        "gene",
        "mu",
        "log2fc",
    }

    missing_columns = required_columns.difference(
        summary.columns
    )

    if missing_columns:
        raise KeyError(
            "Saved posterior table is missing required columns: "
            f"{sorted(missing_columns)}\n"
            f"Available columns: {list(summary.columns)}"
        )

    summary["gene"] = (
        summary["gene"]
        .astype(str)
    )

    summary["gene_upper"] = (
        summary["gene"]
        .map(normalize_gene_name)
    )

    summary["mu"] = pd.to_numeric(
        summary["mu"],
        errors="coerce",
    )

    summary["log2fc"] = pd.to_numeric(
        summary["log2fc"],
        errors="coerce",
    )

    summary["abs_mu"] = np.abs(
        summary["mu"]
    )

    summary["log1p_abs_mu"] = np.log1p(
        summary["abs_mu"]
    )

    summary["abs_log2fc"] = np.abs(
        summary["log2fc"]
    )

    valid_cipher = (
        np.isfinite(summary["mu"])
        & np.isfinite(summary["abs_mu"])
        & np.isfinite(summary["log1p_abs_mu"])
    )

    valid_lfc = (
        np.isfinite(summary["log2fc"])
        & np.isfinite(summary["abs_log2fc"])
    )

    valid_both = (
        valid_cipher
        & valid_lfc
    )

    summary["cipher_rank_by_abs_mu"] = (
        ordinal_rank_desc(
            summary["abs_mu"].to_numpy(),
            valid=valid_both,
        )
    )

    summary["lfc_rank_by_abs_log2fc"] = (
        ordinal_rank_desc(
            summary["abs_log2fc"].to_numpy(),
            valid=valid_both,
        )
    )

    n_invalid = int(
        (~valid_both).sum()
    )

    if n_invalid > 0:
        print(
            f"[warning] dropping {n_invalid} genes with nonfinite "
            "mu, log(1 + |mu|), or log2FC."
        )

    plot_df = (
        summary.loc[valid_both]
        .copy()
    )

    if len(plot_df) == 0:
        raise ValueError(
            "No genes have finite posterior and LFC values."
        )

    plot_df = plot_df.sort_values(
        [
            "abs_mu",
            "abs_log2fc",
            "gene",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    plot_df["cipher_rank_by_abs_mu"] = np.arange(
        1,
        len(plot_df) + 1,
    )

    plot_df["lfc_rank_by_abs_log2fc"] = (
        ordinal_rank_desc(
            plot_df["abs_log2fc"].to_numpy(),
        )
    )

    x = (
        plot_df["cipher_rank_by_abs_mu"]
        .to_numpy(dtype=float)
    )

    y = (
        plot_df["log1p_abs_mu"]
        .to_numpy(dtype=float)
    )

    gene_names = (
        plot_df["gene"]
        .to_numpy(dtype=str)
    )

    gene_names_upper = (
        plot_df["gene_upper"]
        .to_numpy(dtype=str)
    )

    requested_gene_set = {
        normalize_gene_name(gene)
        for gene in GENES_TO_LABEL
    }

    label_indices = np.where(
        np.isin(
            gene_names_upper,
            list(requested_gene_set),
        )
    )[0]

    present_gene_set = set(
        gene_names_upper[label_indices]
    )

    missing_genes = [
        gene
        for gene in GENES_TO_LABEL
        if normalize_gene_name(gene)
        not in present_gene_set
    ]

    print(f"[plot] genes available: {len(plot_df)}")

    print(f"[plot] requested labels: {len(GENES_TO_LABEL)}")

    print(f"[plot] labels found: {len(label_indices)}")

    if len(label_indices) > 0:
        print(
            "[plot] labeled genes:",
            ", ".join(
                gene_names[label_indices]
            ),
        )

    if missing_genes:
        print(
            "[warning] requested genes absent from saved table:",
            ", ".join(missing_genes),
        )

    n_genes = len(plot_df)

    curve_ymax = float(
        np.nanmax(y)
    )

    curve_ymin = float(
        np.nanmin(y)
    )

    y_scale = max(
        curve_ymax - curve_ymin,
        curve_ymax,
        1.0,
    )

    band_gap = max(
        0.65,
        LABEL_BAND_GAP_FRACTION * y_scale,
    )

    row_spacing = max(
        0.58,
        LABEL_ROW_SPACING_FRACTION * y_scale,
    )

    x_left_margin = max(
        30.0,
        X_LEFT_MARGIN_FRACTION * n_genes,
    )

    x_right_margin = max(
        60.0,
        X_RIGHT_MARGIN_FRACTION * n_genes,
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
        label_indices=label_indices,
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

    y_bottom = -max(
        0.25,
        0.025 * y_scale,
    )

    y_top = (
        label_band_top
        + max(
            0.75,
            0.08 * y_scale,
        )
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

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
        alpha=0.28,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    if len(label_indices) > 0:
        ax.scatter(
            x[label_indices],
            y[label_indices],
            s=HIGHLIGHT_SIZE,
            marker="*",
            edgecolor="black",
            linewidth=0.45,
            zorder=12,
        )

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

            # Actual point on curve.
            xy=(
                row.point_x,
                row.point_y,
            ),

            # Label in dedicated band.
            xytext=(
                row.label_x,
                row.label_y,
            ),

            textcoords="data",
            ha="center",
            va="center",
            fontsize=LABEL_FONTSIZE,
            zorder=30,
            clip_on=False,

            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "0.70",
                "linewidth": 0.55,
                "alpha": 1.0,
            },

            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.75,
                "color": "0.35",
                "alpha": 0.75,
                "shrinkA": 3,
                "shrinkB": 6,
                "connectionstyle": "arc3,rad=0.0",
            },
        )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_bottom,
        y_top,
    )

    ax.axhline(
        0,
        linewidth=0.85,
        alpha=0.55,
        zorder=1,
    )

    ax.set_xlabel(
        r"Rank by $|\mathrm{posterior\ mean}\ \mu|$",
        fontsize=12,
    )

    ax.set_ylabel(
        r"$\log(1+|\mathrm{posterior\ mean}\ \mu|)$",
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

    ax.grid(False)

    plt.tight_layout()

    png_path = OUTPUT_STEM.with_suffix(
        ".png"
    )

    pdf_path = OUTPUT_STEM.with_suffix(
        ".pdf"
    )

    svg_path = OUTPUT_STEM.with_suffix(
        ".svg"
    )

    ranked_plot_table_path = (
        OUTPUT_STEM.parent
        / f"{OUTPUT_STEM.name}_table.tsv"
    )

    lfc_cipher_table_path = (
        OUTPUT_STEM.parent
        / "lfc_rank_and_cipher_score_rank_table.tsv"
    )

    label_layout_path = (
        OUTPUT_STEM.parent
        / f"{OUTPUT_STEM.name}_label_layout.tsv"
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
        bbox_inches="tight",
    )

    plot_df["labeled"] = np.isin(
        plot_df["gene_upper"],
        list(requested_gene_set),
    )

    plot_df[
        [
            "cipher_rank_by_abs_mu",
            "gene",
            "mu",
            "abs_mu",
            "log1p_abs_mu",
            "log2fc",
            "abs_log2fc",
            "lfc_rank_by_abs_log2fc",
            "labeled",
        ]
    ].to_csv(
        ranked_plot_table_path,
        sep="\t",
        index=False,
    )

    lfc_cipher_table = plot_df[
        [
            "gene",
            "log2fc",
            "abs_log2fc",
            "lfc_rank_by_abs_log2fc",
            "mu",
            "abs_mu",
            "log1p_abs_mu",
            "cipher_rank_by_abs_mu",
        ]
    ].copy()

    lfc_cipher_table = lfc_cipher_table.rename(
        columns={
            "log2fc": "LFC_log2FC",
            "abs_log2fc": "LFC_abs_log2FC",
            "lfc_rank_by_abs_log2fc": "LFC_rank",
            "mu": "posterior_mu",
            "abs_mu": "posterior_abs_mu",
            "log1p_abs_mu": "CIPHER_log1p_abs_mu",
            "cipher_rank_by_abs_mu": "CIPHER_rank",
        }
    )

    lfc_cipher_table = lfc_cipher_table.sort_values(
        "CIPHER_rank",
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)

    lfc_cipher_table.to_csv(
        lfc_cipher_table_path,
        sep="\t",
        index=False,
    )

    label_layout.to_csv(
        label_layout_path,
        sep="\t",
        index=False,
    )

    print(f"[saved] {png_path}")

    print(f"[saved] {pdf_path}")

    print(f"[saved] {svg_path}")

    print(f"[saved] {ranked_plot_table_path}")

    print(f"[saved] {lfc_cipher_table_path}")

    print(f"[saved] {label_layout_path}")

    print("\n[LFC/CIPHER table columns]")

    print(
        ", ".join(
            lfc_cipher_table.columns
        )
    )

    plt.show()


def fig_cipher_rank_green_grey():
    OUTDIR = Path(
        os.path.join(_BASE, "analytic_gaussian_top_sig_highlfc_tau2_1p0")
    )

    SUMMARY_PATH = (
        OUTDIR
        / "posterior_summary_selected.tsv"
    )

    GREEN_GENES = [
        "ANKRD1",
        "IGFN1",
        "PSG9",
        "PSG4",
        "SAMD9",
        "TSPAN12",
        "TNNT2",
        "ATXN1",
        "TGFA",
        "ROBO2",
        "ATRNL1",

    ]

    GREY_GENES = [
        "SYBU",
        "MAPK4",
        "APOL1",
        "CDKN1A",
        "STX11",
        "PARP8",
        "ATF3",
        "KRTAP5-AS1",
        "GBP3",
        "HS6ST3",
        "DACT1",
    ]

    GENES_TO_LABEL = (
        GREEN_GENES
        + GREY_GENES
    )

    FIGSIZE = (20, 9)

    DPI = 300

    LINE_WIDTH = 1.35

    POINT_SIZE = 12

    HIGHLIGHT_SIZE = 150

    GREEN_HIGHLIGHT_COLOR = "green"

    GREY_HIGHLIGHT_COLOR = "0.60"

    HIGHLIGHT_EDGE_COLOR = "black"

    HIGHLIGHT_EDGE_WIDTH = 0.65

    LABEL_FONTSIZE = 10

    LABEL_ROWS = 3

    LABEL_BAND_GAP_FRACTION = 0.09

    LABEL_ROW_SPACING_FRACTION = 0.085

    X_LEFT_MARGIN_FRACTION = 0.035

    X_RIGHT_MARGIN_FRACTION = 0.045

    OUTPUT_STEM = (
        OUTDIR
        / "posterior_log1p_abs_mu_vs_abs_mu_rank_green_grey_labels"
    )

    green_gene_set = set(
        normalize_gene_list(GREEN_GENES)
    )

    grey_gene_set = set(
        normalize_gene_list(GREY_GENES)
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

    requested_gene_set = (
        green_gene_set
        | grey_gene_set
    )

    if len(requested_gene_set) == 0:
        raise ValueError(
            "GREEN_GENES and GREY_GENES are both empty."
        )

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Could not find the saved posterior table:\n"
            f"  {SUMMARY_PATH}\n\n"
            "Run the posterior-fitting script first or update OUTDIR."
        )

    print(f"[load] {SUMMARY_PATH}")

    summary = pd.read_csv(
        SUMMARY_PATH,
        sep="\t",
        low_memory=False,
    )

    required_columns = {
        "gene",
        "mu",
        "log2fc",
    }

    missing_columns = required_columns.difference(
        summary.columns
    )

    if missing_columns:
        raise KeyError(
            "Saved posterior table is missing required columns: "
            f"{sorted(missing_columns)}\n"
            f"Available columns: {list(summary.columns)}"
        )

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

    summary["log2fc"] = pd.to_numeric(
        summary["log2fc"],
        errors="coerce",
    )

    summary["abs_mu"] = np.abs(
        summary["mu"]
    )

    summary["log1p_abs_mu"] = np.log1p(
        summary["abs_mu"]
    )

    summary["abs_log2fc"] = np.abs(
        summary["log2fc"]
    )

    valid_cipher = (
        np.isfinite(summary["mu"])
        & np.isfinite(summary["abs_mu"])
        & np.isfinite(summary["log1p_abs_mu"])
    )

    valid_lfc = (
        np.isfinite(summary["log2fc"])
        & np.isfinite(summary["abs_log2fc"])
    )

    valid_both = (
        valid_cipher
        & valid_lfc
    )

    summary["cipher_rank_by_abs_mu"] = (
        ordinal_rank_desc(
            summary["abs_mu"].to_numpy(),
            valid=valid_both,
        )
    )

    summary["lfc_rank_by_abs_log2fc"] = (
        ordinal_rank_desc(
            summary["abs_log2fc"].to_numpy(),
            valid=valid_both,
        )
    )

    n_invalid = int(
        (~valid_both).sum()
    )

    if n_invalid > 0:
        print(
            f"[warning] dropping {n_invalid} genes with nonfinite "
            "mu, log(1 + |mu|), or log2FC."
        )

    plot_df = (
        summary.loc[valid_both]
        .copy()
    )

    if len(plot_df) == 0:
        raise ValueError(
            "No genes have finite posterior and LFC values."
        )

    plot_df = plot_df.sort_values(
        [
            "abs_mu",
            "abs_log2fc",
            "gene",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    plot_df["cipher_rank_by_abs_mu"] = np.arange(
        1,
        len(plot_df) + 1,
    )

    plot_df["lfc_rank_by_abs_log2fc"] = (
        ordinal_rank_desc(
            plot_df["abs_log2fc"].to_numpy(),
        )
    )

    x = (
        plot_df["cipher_rank_by_abs_mu"]
        .to_numpy(dtype=float)
    )

    y = (
        plot_df["log1p_abs_mu"]
        .to_numpy(dtype=float)
    )

    gene_names = (
        plot_df["gene"]
        .to_numpy(dtype=str)
    )

    gene_names_upper = (
        plot_df["gene_upper"]
        .to_numpy(dtype=str)
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

    label_indices = np.sort(
        np.concatenate(
            [
                green_indices,
                grey_indices,
            ]
        )
    )

    present_gene_set = set(
        gene_names_upper[label_indices]
    )

    missing_green_genes = [
        gene
        for gene in GREEN_GENES
        if normalize_gene_name(gene)
        not in present_gene_set
    ]

    missing_grey_genes = [
        gene
        for gene in GREY_GENES
        if normalize_gene_name(gene)
        not in present_gene_set
    ]

    print(f"[plot] genes available: {len(plot_df)}")

    print(f"[plot] requested green genes: {len(GREEN_GENES)}")

    print(f"[plot] requested grey genes: {len(GREY_GENES)}")

    print(f"[plot] green genes found: {len(green_indices)}")

    print(f"[plot] grey genes found: {len(grey_indices)}")

    print(f"[plot] total labels found: {len(label_indices)}")

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
            "[warning] green genes absent from saved table:",
            ", ".join(missing_green_genes),
        )

    if missing_grey_genes:
        print(
            "[warning] grey genes absent from saved table:",
            ", ".join(missing_grey_genes),
        )

    plot_df["highlight_group"] = "unlabeled"

    plot_df.loc[
        plot_df["gene_upper"].isin(green_gene_set),
        "highlight_group",
    ] = "green"

    plot_df.loc[
        plot_df["gene_upper"].isin(grey_gene_set),
        "highlight_group",
    ] = "grey"

    plot_df["labeled"] = (
        plot_df["highlight_group"]
        != "unlabeled"
    )

    n_genes = len(plot_df)

    curve_ymax = float(
        np.nanmax(y)
    )

    curve_ymin = float(
        np.nanmin(y)
    )

    y_scale = max(
        curve_ymax - curve_ymin,
        curve_ymax,
        1.0,
    )

    band_gap = max(
        0.65,
        LABEL_BAND_GAP_FRACTION * y_scale,
    )

    row_spacing = max(
        0.58,
        LABEL_ROW_SPACING_FRACTION * y_scale,
    )

    x_left_margin = max(
        30.0,
        X_LEFT_MARGIN_FRACTION * n_genes,
    )

    x_right_margin = max(
        60.0,
        X_RIGHT_MARGIN_FRACTION * n_genes,
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
        label_indices=label_indices,
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

    y_bottom = -max(
        0.25,
        0.025 * y_scale,
    )

    y_top = (
        label_band_top
        + max(
            0.75,
            0.08 * y_scale,
        )
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

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
        alpha=0.28,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

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
            zorder=13,
            label="Green gene group",
        )

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
            zorder=12,
            label="Grey gene group",
        )

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

            # Actual point on curve.
            xy=(
                row.point_x,
                row.point_y,
            ),

            # Label in dedicated band.
            xytext=(
                row.label_x,
                row.label_y,
            ),

            textcoords="data",
            ha="center",
            va="center",
            fontsize=LABEL_FONTSIZE,
            zorder=30,
            clip_on=False,

            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "0.70",
                "linewidth": 0.55,
                "alpha": 1.0,
            },

            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.75,
                "color": "0.35",
                "alpha": 0.75,
                "shrinkA": 3,
                "shrinkB": 6,
                "connectionstyle": "arc3,rad=0.0",
            },
        )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_bottom,
        y_top,
    )

    ax.axhline(
        0,
        linewidth=0.85,
        alpha=0.55,
        zorder=1,
    )

    ax.set_xlabel(
        r"Rank by $|\mathrm{posterior\ mean}\ \mu|$",
        fontsize=12,
    )

    ax.set_ylabel(
        r"$\log(1+|\mathrm{posterior\ mean}\ \mu|)$",
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

    ax.grid(False)

    plt.tight_layout()

    png_path = OUTPUT_STEM.with_suffix(
        ".png"
    )

    pdf_path = OUTPUT_STEM.with_suffix(
        ".pdf"
    )

    svg_path = OUTPUT_STEM.with_suffix(
        ".svg"
    )

    ranked_plot_table_path = (
        OUTPUT_STEM.parent
        / f"{OUTPUT_STEM.name}_table.tsv"
    )

    lfc_cipher_table_path = (
        OUTPUT_STEM.parent
        / "lfc_rank_and_cipher_score_rank_table_green_grey.tsv"
    )

    label_layout_path = (
        OUTPUT_STEM.parent
        / f"{OUTPUT_STEM.name}_label_layout.tsv"
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
        bbox_inches="tight",
    )

    plot_df[
        [
            "cipher_rank_by_abs_mu",
            "gene",
            "mu",
            "abs_mu",
            "log1p_abs_mu",
            "log2fc",
            "abs_log2fc",
            "lfc_rank_by_abs_log2fc",
            "labeled",
            "highlight_group",
        ]
    ].to_csv(
        ranked_plot_table_path,
        sep="\t",
        index=False,
    )

    lfc_cipher_table = plot_df[
        [
            "gene",
            "log2fc",
            "abs_log2fc",
            "lfc_rank_by_abs_log2fc",
            "mu",
            "abs_mu",
            "log1p_abs_mu",
            "cipher_rank_by_abs_mu",
            "highlight_group",
        ]
    ].copy()

    lfc_cipher_table = lfc_cipher_table.rename(
        columns={
            "log2fc": "LFC_log2FC",
            "abs_log2fc": "LFC_abs_log2FC",
            "lfc_rank_by_abs_log2fc": "LFC_rank",
            "mu": "posterior_mu",
            "abs_mu": "posterior_abs_mu",
            "log1p_abs_mu": "CIPHER_log1p_abs_mu",
            "cipher_rank_by_abs_mu": "CIPHER_rank",
        }
    )

    lfc_cipher_table = lfc_cipher_table.sort_values(
        "CIPHER_rank",
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)

    lfc_cipher_table.to_csv(
        lfc_cipher_table_path,
        sep="\t",
        index=False,
    )

    label_layout.to_csv(
        label_layout_path,
        sep="\t",
        index=False,
    )

    print(f"[saved] {png_path}")

    print(f"[saved] {pdf_path}")

    print(f"[saved] {svg_path}")

    print(f"[saved] {ranked_plot_table_path}")

    print(f"[saved] {lfc_cipher_table_path}")

    print(f"[saved] {label_layout_path}")

    print("\n[LFC/CIPHER table columns]")

    print(
        ", ".join(
            lfc_cipher_table.columns
        )
    )

    plt.show()


def fig_naive_control_ranked_mean():
    H5AD_PATH = os.path.join(SUPPL, "pancreatic_naive_vs_resistant.h5ad")

    OUTDIR = os.path.join(_BASE, "naive_control_ranked_mean_expression")

    os.makedirs(OUTDIR, exist_ok=True)

    CONDITION_KEY = "Condition"

    CONTROL_LABEL = "Naive"

    GENES_TO_CHECK = [
        "ANKRD1",
        "IGFN1",
        "PSG9",
        "PSG4",
        "SAMD9",
        "TSPAN12",
        "TNNT2",
        "ATXN1",
        "TGFA",
        "ROBO2",
        "DACT1",
        "SYBU",
        "MAPK4",
        "APOL1",
        "CDKN1A",
        "STX11",
        "PARP8",
        "ATF3",
        "KRTAP5-AS1",
        "ATRNL1",
        "GBP3",
        "HS6ST3",
    ]

    TOP_N_LABELS = 0

    MIN_CELLS_FRAC = 0.00001

    MIN_EXPR = 0.01

    HI_QUANTILE = 1.0

    VAR_DROP_Q = 1.0

    FILTER_SUBSAMPLE_CELLS = 0

    SEED = 0

    DROP_HOUSEKEEPING = True

    BAD_PREFIXES = (
        "RPL",
        "RPS",
        "MT-",
        "MT.",
        "HSP",
        "HSP90",
        "EIF",
    )

    FIGSIZE = (16, 6)

    DPI = 300

    def drop_housekeeping_prefixes(
        var_names,
        bad_prefixes=BAD_PREFIXES,
    ):
        names = np.asarray(var_names, dtype=str)
        names_upper = np.char.upper(names)

        keep = np.ones(len(names), dtype=bool)

        for prefix in bad_prefixes:
            keep &= ~np.char.startswith(names_upper, prefix.upper())

        return keep

    def filter_by_expression_and_variance_percentile(
        adata,
        control_mask,
        min_cells_frac=0.01,
        min_expr=1.0,
        hi_quantile=0.90,
        var_drop_q=1.0,
        filter_subsample_cells=0,
        seed=0,
    ):
        """
        Same filtering logic as the original supplied pipeline.

        Because this script contains only Naive/control cells,
        control_mask will be all True.
        """

        rng = np.random.default_rng(seed)

        if (
            filter_subsample_cells
            and adata.n_obs > filter_subsample_cells
        ):
            idx = rng.choice(
                np.arange(adata.n_obs),
                size=int(filter_subsample_cells),
                replace=False,
            )

            adata_sub = adata[idx].copy()
            cm = np.asarray(control_mask, dtype=bool)[idx]

        else:
            adata_sub = adata
            cm = np.asarray(control_mask, dtype=bool)

        X = to_dense(adata_sub.X).astype(np.float64)

        # Fraction of cells in which each gene passes min_expr.
        frac_on = np.mean(X >= min_expr, axis=0)

        # High expression quantile for each gene.
        q_hi = np.quantile(X, hi_quantile, axis=0)

        keep_expr = (
            (frac_on >= min_cells_frac)
            | (q_hi >= min_expr)
        )

        # Variance is calculated using only control cells.
        X_control = X[cm]
        variances = X_control.var(axis=0)

        if np.any(keep_expr):
            variance_cutoff = np.quantile(
                variances[keep_expr],
                var_drop_q,
            )
            keep_variance = variances <= variance_cutoff
        else:
            keep_variance = np.ones_like(
                keep_expr,
                dtype=bool,
            )

        keep = keep_expr & keep_variance

        print(
            f"[filter] kept {keep.sum():,} / "
            f"{len(keep):,} genes"
        )

        return adata[:, keep].copy()

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(
            f"Could not find:\n{H5AD_PATH}\n\n"
            f"Current working directory:\n{os.getcwd()}"
        )

    print(f"[load] {H5AD_PATH}")

    adata = ad.read_h5ad(H5AD_PATH)

    adata.var_names_make_unique()

    print(
        f"[load] original shape: "
        f"{adata.n_obs:,} cells x {adata.n_vars:,} genes"
    )

    if CONDITION_KEY not in adata.obs.columns:
        raise KeyError(
            f"'{CONDITION_KEY}' was not found in adata.obs.\n"
            f"Available columns:\n"
            f"{list(adata.obs.columns)}"
        )

    available_conditions = (
        adata.obs[CONDITION_KEY]
        .astype(str)
        .value_counts()
    )

    print("\n[conditions]")

    print(available_conditions.to_string())

    if CONTROL_LABEL not in set(
        adata.obs[CONDITION_KEY].astype(str)
    ):
        raise ValueError(
            f"Control label '{CONTROL_LABEL}' was not found.\n"
            f"Available values:\n"
            f"{available_conditions.index.tolist()}"
        )

    control_mask = (
        adata.obs[CONDITION_KEY]
        .astype(str)
        .values
        == CONTROL_LABEL
    )

    adata_control = adata[control_mask].copy()

    print(
        f"\n[control] retained {adata_control.n_obs:,} "
        f"{CONTROL_LABEL} cells"
    )

    if adata_control.n_obs < 2:
        raise ValueError(
            f"Too few {CONTROL_LABEL} cells: "
            f"{adata_control.n_obs}"
        )

    all_control_mask = np.ones(
        adata_control.n_obs,
        dtype=bool,
    )

    adata_control = (
        filter_by_expression_and_variance_percentile(
            adata=adata_control,
            control_mask=all_control_mask,
            min_cells_frac=MIN_CELLS_FRAC,
            min_expr=MIN_EXPR,
            hi_quantile=HI_QUANTILE,
            var_drop_q=VAR_DROP_Q,
            filter_subsample_cells=FILTER_SUBSAMPLE_CELLS,
            seed=SEED,
        )
    )

    if DROP_HOUSEKEEPING:
        keep = drop_housekeeping_prefixes(
            adata_control.var_names
        )

        print(
            f"[housekeeping] removing "
            f"{(~keep).sum():,} genes"
        )

        adata_control = adata_control[:, keep].copy()

    print(
        f"[processed] final shape: "
        f"{adata_control.n_obs:,} cells x "
        f"{adata_control.n_vars:,} genes"
    )

    X_control = to_dense(
        adata_control.X
    ).astype(np.float64)

    gene_names = np.asarray(
        adata_control.var_names,
        dtype=str,
    )

    mean_expression = np.asarray(
        X_control.mean(axis=0)
    ).reshape(-1)

    fraction_expressing = np.asarray(
        np.mean(X_control > 0, axis=0)
    ).reshape(-1)

    variance_expression = np.asarray(
        X_control.var(axis=0)
    ).reshape(-1)

    order = np.argsort(
        -mean_expression,
        kind="mergesort",
    )

    gene_ranked = gene_names[order]

    mean_ranked = mean_expression[order]

    fraction_ranked = fraction_expressing[order]

    variance_ranked = variance_expression[order]

    rank = np.arange(
        1,
        len(gene_ranked) + 1,
    )

    ranked_df = pd.DataFrame({
        "rank_by_control_mean": rank,
        "gene": gene_ranked,
        "control_mean_expression": mean_ranked,
        "control_fraction_expressing": fraction_ranked,
        "control_variance": variance_ranked,
    })

    table_path = os.path.join(
        OUTDIR,
        "naive_control_ranked_mean_expression.tsv",
    )

    ranked_df.to_csv(
        table_path,
        sep="\t",
        index=False,
    )

    tracked_upper = set(
        normalize_gene_list(GENES_TO_CHECK)
    )

    gene_ranked_upper = np.char.upper(
        gene_ranked.astype(str)
    )

    tracked_indices = [
        i
        for i, gene in enumerate(gene_ranked_upper)
        if gene in tracked_upper
    ]

    top_indices = list(
        range(
            min(
                TOP_N_LABELS,
                len(gene_ranked),
            )
        )
    )

    label_indices = sorted(
        set(top_indices + tracked_indices)
    )

    print(
        f"\n[labels] top-expression genes: "
        f"{len(top_indices)}"
    )

    print(
        f"[labels] tracked genes found after filtering: "
        f"{len(tracked_indices)} / "
        f"{len(tracked_upper)}"
    )

    if tracked_indices:
        print(
            "[labels] tracked genes shown:",
            ", ".join(
                gene_ranked[tracked_indices]
            ),
        )

    missing_tracked = sorted(
        tracked_upper
        - set(gene_ranked_upper)
    )

    if missing_tracked:
        print(
            "[labels] tracked genes absent after filtering:",
            ", ".join(missing_tracked),
        )

    fig, ax = plt.subplots(figsize=FIGSIZE)

    x = rank.astype(float)

    y = mean_ranked.astype(float)

    ax.plot(
        x,
        y,
        linewidth=1.4,
        zorder=2,
    )

    ax.scatter(
        x,
        y,
        s=9,
        alpha=0.55,
        rasterized=True,
        zorder=3,
    )

    if top_indices:
        ax.scatter(
            x[top_indices],
            y[top_indices],
            s=40,
            zorder=6,
            label=f"Top {len(top_indices)} by mean",
        )

    if tracked_indices:
        ax.scatter(
            x[tracked_indices],
            y[tracked_indices],
            s=145,
            marker="*",
            edgecolor="black",
            linewidth=0.5,
            zorder=8,
            label="Genes to check",
        )

    draw_repelled_labels(
        ax=ax,
        x=x,
        y=y,
        gene_names=gene_ranked,
        label_indices=label_indices,
        fontsize=9,
    )

    ax.set_xlabel(
        "Gene rank by mean expression in Naive cells"
    )

    ax.set_ylabel(
        "Mean expression in Naive cells"
    )

    ax.set_title(
        "Ranked mean gene expression in the Naive/control population"
    )

    ax.set_xlim(
        0.5,
        len(gene_ranked) + 0.5,
    )

    ax.set_yscale(
        "symlog",
        linthresh=1e-3,
    )

    ax.grid(
        True,
        which="major",
        alpha=0.18,
    )

    ax.grid(
        True,
        which="minor",
        alpha=0.06,
    )

    if top_indices or tracked_indices:
        ax.legend(
            frameon=False,
            loc="upper right",
        )

    plt.tight_layout()

    png_path = os.path.join(
        OUTDIR,
        "naive_control_ranked_mean_expression.png",
    )

    pdf_path = os.path.join(
        OUTDIR,
        "naive_control_ranked_mean_expression.pdf",
    )

    svg_path = os.path.join(
        OUTDIR,
        "naive_control_ranked_mean_expression.svg",
    )

    plt.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.show()

    tracked_table = ranked_df.loc[
        ranked_df["gene"]
        .astype(str)
        .str.upper()
        .isin(tracked_upper)
    ].copy()

    tracked_table = tracked_table.sort_values(
        "rank_by_control_mean"
    )

    tracked_path = os.path.join(
        OUTDIR,
        "naive_control_tracked_gene_expression.tsv",
    )

    tracked_table.to_csv(
        tracked_path,
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 72)

    print("[done] Naive/control ranked-expression analysis")

    print("=" * 72)

    print(f"[saved] {png_path}")

    print(f"[saved] {pdf_path}")

    print(f"[saved] {svg_path}")

    print(f"[saved] {table_path}")

    print(f"[saved] {tracked_path}")

    print("\n[tracked-gene ranks]")

    if len(tracked_table):
        print(
            tracked_table[
                [
                    "rank_by_control_mean",
                    "gene",
                    "control_mean_expression",
                    "control_fraction_expressing",
                    "control_variance",
                ]
            ].to_string(index=False)
        )
    else:
        print(
            "None of the tracked genes remained after filtering."
        )


def fig_naive_all_genes_mean():
    H5AD_PATH = os.path.join(SUPPL, "pancreatic_naive_vs_resistant.h5ad")

    OUTDIR = os.path.join(_BASE, "naive_all_genes_mean_clear_labels")

    os.makedirs(OUTDIR, exist_ok=True)

    CONDITION_KEY = "Condition"

    CONTROL_LABEL = "Naive"

    GENES_TO_CHECK = [
        "ANKRD1",
        "IGFN1",
        "PSG9",
        "PSG4",
        "SAMD9",
        "TSPAN12",
        "TNNT2",
        "ATXN1",
        "TGFA",
        "ROBO2",
        "DACT1",
        "SYBU",
        "MAPK4",
        "APOL1",
        "CDKN1A",
        "STX11",
        "PARP8",
        "ATF3",
        "KRTAP5-AS1",
        "ATRNL1",
        "GBP3",
        "HS6ST3",
    ]

    FIGSIZE = (13, 7.5)

    DPI = 300

    STAR_SIZE = 190

    LABEL_FONTSIZE = 9

    ARROW_LINEWIDTH = 1.0

    if not os.path.exists(H5AD_PATH):
        raise FileNotFoundError(
            f"Could not find:\n{H5AD_PATH}\n\n"
            f"Current working directory:\n{os.getcwd()}"
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
        f"[statistics] positive finite gene means: "
        f"{valid_mean.sum():,} / "
        f"{len(control_mean):,}"
    )

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
        str(gene).strip().upper()
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

    ranked_table = pd.DataFrame({
        "control_mean_rank": ranks,
        "gene": ranked_genes,
        "control_mean": ranked_mean,
    })

    ranked_table_path = os.path.join(
        OUTDIR,
        "naive_all_genes_ranked_by_mean.tsv",
    )

    ranked_table.to_csv(
        ranked_table_path,
        sep="\t",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.plot(
        ranks[ranked_valid],
        ranked_mean[ranked_valid],
        linewidth=1.5,
        zorder=2,
    )

    ax.scatter(
        ranks[ranked_valid],
        ranked_mean[ranked_valid],
        s=8,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    if len(tracked_indices) > 0:

        ax.scatter(
            ranks[tracked_indices],
            ranked_mean[tracked_indices],
            s=STAR_SIZE,
            marker="*",
            facecolor="limegreen",
            edgecolor="black",
            linewidth=0.8,
            zorder=20,
            clip_on=False,
        )

        draw_clear_gene_labels(
            ax=ax,
            x=ranks,
            y=ranked_mean,
            labels=ranked_genes,
            indices=tracked_indices,
            fontsize=LABEL_FONTSIZE,
        )

    ax.set_xlabel(
        "Gene rank",
        fontsize=12,
    )

    ax.set_ylabel(
        "Mean expression in Naive cells",
        fontsize=12,
    )

    ax.set_title(
        "Naive control mean gene expression",
        fontsize=14,
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
        labelsize=10,
    )

    plt.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.13,
        top=0.90,
    )

    png_path = os.path.join(
        OUTDIR,
        "naive_all_genes_ranked_mean_clear_labels.png",
    )

    pdf_path = os.path.join(
        OUTDIR,
        "naive_all_genes_ranked_mean_clear_labels.pdf",
    )

    svg_path = os.path.join(
        OUTDIR,
        "naive_all_genes_ranked_mean_clear_labels.svg",
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

    tracked_table = ranked_table.loc[
        ranked_table["gene"]
        .astype(str)
        .str.upper()
        .isin(tracked_upper)
    ].copy()

    tracked_table = tracked_table.sort_values(
        "control_mean_rank"
    )

    tracked_path = os.path.join(
        OUTDIR,
        "naive_tracked_gene_mean_statistics.tsv",
    )

    tracked_table.to_csv(
        tracked_path,
        sep="\t",
        index=False,
    )

    available_gene_upper = set(
        np.char.upper(
            gene_names.astype(str)
        )
    )

    missing_tracked = sorted(
        tracked_upper
        - available_gene_upper
    )

    print("\n" + "=" * 78)

    print("NAIVE CONTROL — RANKED MEAN EXPRESSION")

    print("=" * 78)

    print(
        f"Naive cells: {adata_control.n_obs:,}"
    )

    print(
        f"Genes:       {adata_control.n_vars:,}"
    )

    print("\n[tracked genes]")

    if len(tracked_table) > 0:
        print(
            tracked_table.to_string(
                index=False
            )
        )
    else:
        print(
            "None of the tracked genes were found."
        )

    if missing_tracked:
        print("\n[tracked genes absent from dataset]")
        print(
            ", ".join(missing_tracked)
        )

    print("\n[saved]")

    print(png_path)

    print(pdf_path)

    print(svg_path)

    print(ranked_table_path)

    print(tracked_path)
