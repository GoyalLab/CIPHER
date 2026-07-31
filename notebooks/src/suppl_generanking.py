"""Helper functions for Fig S9 (inverse-inference true-perturbation gene ranking).

Helper in notebooks/src -- NOT part of the installable ``cipher`` package; a
notebook-only helper for reproducing the supplementary figure.

Only the functions that are identical (or unique) across the notebook
cells are collected here. The two plotting functions that are
redefined incompatibly across the three figure-variant cells
(``add_summary_marker`` and ``plot_metric_panel``) are intentionally kept inline
in the notebook so each variant reproduces exactly.

Several functions read notebook-level globals (e.g. ``METHOD_LABELS``,
``METHOD_ORDER``, ``VIOLIN_WIDTH``, ``CRISPRa_KEYWORDS``, ``DISPLAY_NAME_MAP``);
those constants live in the notebook config/main-flow cells.
"""
from __future__ import annotations


# --- library imports required by the extracted functions (added during cleanup fixup;
#     resolved at call time so placement after the docstring is sufficient) ---
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
from pathlib import Path

import numpy as np
import pandas as pd

# --- plot-config constants the helpers close over; the
#     notebook re-injects its own (possibly variant-specific) values into this
#     module namespace before calling the helpers, so these serve as defaults
#     and keep static analysis clean ---
METHOD_LABELS = {
    "score_pval": "-log10(p)",
    "score_pip_full": "PIP",
    "score_lfc_abs": "|LFC|",
    "score_shuffle": "shuffle",
    "score_mean_field": "mean-field",
    "score_true": "true Sigma",
}

METHOD_ORDER = [
    "|LFC|",
    # "Mean-field",
    "True covariance",
]

VIOLIN_WIDTH = 0.72

CRISPRa_KEYWORDS = [
    "akana_etal_2026_crispra_perturbseq",
    "schemidt_etal_2022_crispra_perturbseq",
    "schmidt_etal_2022_crispra_perturbseq",
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

DISPLAY_NAME_MAP = {
    # CRISPRa
    "akana_etal_2026_crispra_perturbseq": "Akana",
    "schemidt_etal_2022_crispra_perturbseq": "Schemidt",
    "schmidt_etal_2022_crispra_perturbseq": "Schmidt",
    "kaden25_rpe1_ctrl_10k_min100_greedy_4gb": "KadenRPE",
    "kaden25_fibroblast_ctrl_10k_min100_greedy_4gb": "KadenFIB",
    "NormanWeissman2019_filtered": "Norman19",
    "TianKampmann2021_CRISPRa": "Tian21_a",

    # CRISPRi
    "XAtlas2025_HEK293T_filtered": "XAtlas25_HEK",
    "XAtlas2025_HCT116_filtered": "XAtlas25_HCT",

    "Marson2025_D1_Rest_filtered": "Marson25_D1_Rest",
    "Marson2025_D1_Stim8hr_filtered": "Marson25_D1_S8",
    "Marson2025_D1_Stim48hr_filtered": "Marson25_D1_S48",

    "Marson2025_D2_Stim8hr_filtered": "Marson25_D2_S8",
    "Marson2025_D2_Stim48hr_filtered": "Marson25_D2_S48",

    "Marson2025_D3_Rest_filtered": "Marson25_D3_Rest",
    "Marson2025_D3_Stim8hr_filtered": "Marson25_D3_S8",
    "Marson2025_D3_Stim48hr_filtered": "Marson25_D3_S48",

    "Marson2025_D4_Rest_filtered": "Marson25_D4_Rest",
    "Marson2025_D4_Stim8hr_filtered": "Marson25_D4_S8",
    "Marson2025_D4_Stim48hr_filtered": "Marson25_D4_S48",

    "ReplogleWeissman2022_rpe1": "Replogle22_RPE1",
    "ReplogleWeissman2022_K562_essential": "Replogle22_K562",

    "GSE264667_jurkat_raw_singlecell_01": "GSE264667_Jurkat",
    "GSE264667_hepg2_raw_singlecell_01": "GSE264667_HepG2",

    "FrangiehIzar2021_RNA": "Frangieh21",

    "TianKampmann2019_day7neuron": "Tian19_Neuron",
    "TianKampmann2019_iPSC": "Tian19_iPSC",
    "TianKampmann2021_CRISPRi": "Tian21_i",
}
# --- end plot-config constants ---

__all__ = ['add_violin', 'calculate_dataset_metrics', 'calculate_pooled_perturbation_metrics', 'classify_dataset', 'cutoff_to_tag', 'dataset_sem', 'decode_string_array', 'find_latest_run', 'is_match', 'label_best_and_worst', 'ordinal_rank', 'short_dataset_name', 'standard_error', 'summarize_ranks']


def cutoff_to_tag(value):
    return f"{float(value):.1f}".replace(".", "p")


def decode_string_array(values):
    output = []

    for value in np.asarray(values):
        if isinstance(value, bytes):
            output.append(value.decode("utf-8"))
        else:
            output.append(str(value))

    return np.asarray(output, dtype=object)


def is_match(name, keywords):
    name = str(name)
    return any(str(keyword) in name for keyword in keywords)


def classify_dataset(dataset):
    if is_match(dataset, CRISPRa_KEYWORDS):
        return "CRISPRa"

    if is_match(dataset, CRISPRi_KEYWORDS):
        return "CRISPRi"

    return "other"


def short_dataset_name(dataset):
    if dataset in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[dataset]

    name = str(dataset)
    name = name.replace("_filtered", "")
    name = name.replace("_RNA", "")
    name = name.replace("NormanWeissman", "Norman")
    name = name.replace("ReplogleWeissman", "Replogle")
    name = name.replace("TianKampmann", "Tian")
    name = name.replace("FrangiehIzar", "Frangieh")
    name = name.replace("XAtlas2025", "XAtlas25")
    name = name.replace("Marson2025", "Marson25")
    name = name.replace("Stim8hr", "S8")
    name = name.replace("Stim48hr", "S48")
    name = re.sub(r"_+", "_", name)

    return name[:50]


def find_latest_run(out_root, expression_cutoff):
    expression_tag = cutoff_to_tag(expression_cutoff)
    base = Path(out_root) / f"mean_ge_{expression_tag}"

    if not base.exists():
        raise FileNotFoundError(
            f"Could not find output directory:\n{base}"
        )

    runs = sorted(
        path
        for path in base.glob("run_*")
        if path.is_dir()
    )

    if not runs:
        raise FileNotFoundError(
            f"No run_* directories found under:\n{base}"
        )

    return runs[-1]


def standard_error(values):
    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna()

    if len(values) <= 1:
        return np.nan

    return float(
        values.std(ddof=1) / np.sqrt(len(values))
    )


def ordinal_rank(value):
    if pd.isna(value):
        return ""

    value = float(value)

    if not value.is_integer():
        return f"{value:.2f}"

    rank = int(value)

    if 10 <= rank % 100 <= 20:
        suffix = "th"
    else:
        suffix = {
            1: "st",
            2: "nd",
            3: "rd",
        }.get(rank % 10, "th")

    return f"{rank}{suffix}"


def summarize_ranks(
    dataframe,
    group_columns,
    rank_column="absolute_rank",
    count_name="n_perturbations",
):
    if dataframe.empty:
        return pd.DataFrame()

    rows = []

    for group_values, group in dataframe.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        values = pd.to_numeric(
            group[rank_column],
            errors="coerce",
        ).dropna()

        row = {
            column: value
            for column, value in zip(
                group_columns,
                group_values,
            )
        }

        row.update({
            count_name: int(len(values)),
            "mean_rank": (
                float(values.mean())
                if len(values)
                else np.nan
            ),
            "sem_rank": standard_error(values),
            "std_rank": (
                float(values.std(ddof=1))
                if len(values) > 1
                else np.nan
            ),
            "median_rank": (
                float(values.median())
                if len(values)
                else np.nan
            ),
            "min_rank": (
                float(values.min())
                if len(values)
                else np.nan
            ),
            "max_rank": (
                float(values.max())
                if len(values)
                else np.nan
            ),
            "top1_fraction": (
                float(np.mean(values <= 1))
                if len(values)
                else np.nan
            ),
            "top5_fraction": (
                float(np.mean(values <= 5))
                if len(values)
                else np.nan
            ),
            "top10_fraction": (
                float(np.mean(values <= 10))
                if len(values)
                else np.nan
            ),
            "top25_fraction": (
                float(np.mean(values <= 25))
                if len(values)
                else np.nan
            ),
            "top50_fraction": (
                float(np.mean(values <= 50))
                if len(values)
                else np.nan
            ),
        })

        if pd.notna(row["sem_rank"]):
            row["mean_plus_minus_sem"] = (
                f"{row['mean_rank']:.3f} "
                f"± {row['sem_rank']:.3f}"
            )
        else:
            row["mean_plus_minus_sem"] = (
                f"{row['mean_rank']:.3f} ± NA"
            )

        rows.append(row)

    return pd.DataFrame(rows)


def dataset_sem(values):
    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna()

    if len(values) <= 1:
        return np.nan

    return float(values.std(ddof=1) / np.sqrt(len(values)))


def calculate_dataset_metrics(rank_df):
    """
    Calculate one value per dataset and method for:
      - mean rank
      - median rank
      - top-1 fraction
    """
    output = (
        rank_df.groupby(
            [
                "dataset",
                "dataset_display",
                "dataset_group",
                "method",
            ],
            dropna=False,
            sort=False,
        )
        .agg(
            n_perturbations=(
                "absolute_rank",
                "count",
            ),
            mean_rank=(
                "absolute_rank",
                "mean",
            ),
            median_rank=(
                "absolute_rank",
                "median",
            ),
            top1_fraction=(
                "absolute_rank",
                lambda x: np.mean(
                    pd.to_numeric(
                        x,
                        errors="coerce",
                    ).dropna() <= 1
                ),
            ),
        )
        .reset_index()
    )

    output["method_label"] = output["method"].map(
        METHOD_LABELS
    )

    output["method_label"] = pd.Categorical(
        output["method_label"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    return output.sort_values(
        [
            "dataset_group",
            "dataset",
            "method_label",
        ],
        kind="stable",
    ).reset_index(drop=True)


def add_violin(
    ax,
    values_by_method,
    positions,
):
    """
    Add violin distributions while gracefully handling methods
    with too few finite observations.
    """
    valid_values = []
    valid_positions = []

    for values, position in zip(
        values_by_method,
        positions,
    ):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        # Matplotlib violinplot requires at least two values
        # with nonzero variation for a useful density.
        if len(values) >= 2 and np.nanstd(values) > 0:
            valid_values.append(values)
            valid_positions.append(position)

    if not valid_values:
        return

    violin = ax.violinplot(
        valid_values,
        positions=valid_positions,
        widths=VIOLIN_WIDTH,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in violin["bodies"]:
        body.set_alpha(0.20)
        body.set_edgecolor("black")
        body.set_linewidth(1.0)


def label_best_and_worst(
    ax,
    values,
    x_position,
    lower_is_better,
):
    """
    Label the best- and worst-performing datasets for one method.

    The values Series has a MultiIndex containing:
      dataset, dataset_display, dataset_group
    """
    values = values.dropna()

    if values.empty:
        return

    if lower_is_better:
        best_index = values.idxmin()
        worst_index = values.idxmax()
    else:
        best_index = values.idxmax()
        worst_index = values.idxmin()

    best_value = float(values.loc[best_index])
    worst_value = float(values.loc[worst_index])

    best_dataset = str(best_index[1])
    worst_dataset = str(worst_index[1])

    ax.annotate(
        best_dataset,
        xy=(x_position, best_value),
        xytext=(7, -9),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "linewidth": 0.7,
            "color": "black",
        },
        zorder=12,
    )

    if worst_index != best_index:
        ax.annotate(
            worst_dataset,
            xy=(x_position, worst_value),
            xytext=(7, 9),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.7,
                "color": "black",
            },
            zorder=12,
        )


def calculate_pooled_perturbation_metrics(rank_df):
    """
    Pool all perturbations across all datasets and calculate one
    summary value per method.

    Unlike an average of dataset-level metrics, each perturbation
    contributes equally to these pooled summaries.
    """
    output = (
        rank_df.groupby(
            "method",
            dropna=False,
            sort=False,
        )
        .agg(
            n_perturbations=(
                "absolute_rank",
                "count",
            ),
            mean_rank=(
                "absolute_rank",
                "mean",
            ),
            median_rank=(
                "absolute_rank",
                "median",
            ),
            top1_fraction=(
                "absolute_rank",
                lambda x: np.mean(
                    pd.to_numeric(
                        x,
                        errors="coerce",
                    ).dropna() <= 1
                ),
            ),
            rank_sem=(
                "absolute_rank",
                dataset_sem,
            ),
        )
        .reset_index()
    )

    output["method_label"] = output["method"].map(
        METHOD_LABELS
    )

    output["method_label"] = pd.Categorical(
        output["method_label"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    return output.sort_values(
        "method_label",
        kind="stable",
    ).reset_index(drop=True)
