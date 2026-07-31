"""Notebook-only helpers for Fig S15 (observed vs CIPHER-predicted effective
number of responding genes / response breadth).

NOT part of the
installable ``cipher`` package -- a notebook-only helper for reproducing the
supplementary figure.

The three analysis cells were independent scripts that each redefined a few
same-named helpers with slightly different bodies/config. This module keeps one
canonical version of every helper (the version used by the final published
figure, cell 45); the two variant helpers needed by the earlier exploratory
cells (cell 31's ``number_to_tag`` and cell 36's ``bootstrap_dataset_spearman`` /
``calculate_dataset_medians`` / ``calculate_cutoff_summary`` / ``add_loglog_scatter``)
are redefined inline in those notebook cells so every cell reproduces 1:1.

Config constants that the extracted functions close over are defined here at
module scope with the values shared across the cells that call those functions.
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --- config constants the extracted helpers close over (shared values) ---
DPI = 300
SAVE_PDF = True
SAVE_SVG = True
SHOW_FIGURES = True
DATASET_COLUMN = "dataset_base"
PERTURBATION_COLUMN = "perturbation"
MIN_PERTURBATIONS_PER_DATASET = 3
MIN_DATASETS_FOR_SPEARMAN = 3
N_BOOTSTRAP = 2000
SCATTER_SIZE = 16
SCATTER_ALPHA = 0.17
IDENTITY_LINE_WIDTH = 1.7
GRID_ALPHA = 0.18

__all__ = [
    'save_figure',
    'number_to_tag',
    'safe_pearson',
    'safe_spearman',
    'safe_r2_uncentered',
    'safe_r2_centered',
    'rank_zscore',
    'bootstrap_correlation_interval',
    'safe_centered_r2',
    'safe_uncentered_r2',
    'make_filtered_dataframe',
    'bootstrap_dataset_spearman',
    'calculate_dataset_medians',
    'filter_by_r2',
    'calculate_sweep_row',
    'calculate_shared_log_limits',
    'add_perturbation_loglog_panel',
]

def save_figure(
    figure,
    output_base,
):
    output_base = Path(
        output_base
    )

    figure.savefig(
        output_base.with_suffix(
            ".png"
        ),
        dpi=DPI,
        bbox_inches="tight",
    )

    if SAVE_PDF:
        figure.savefig(
            output_base.with_suffix(
                ".pdf"
            ),
            bbox_inches="tight",
        )

    if SAVE_SVG:
        figure.savefig(
            output_base.with_suffix(
                ".svg"
            ),
            bbox_inches="tight",
        )

    if SHOW_FIGURES:
        plt.show()

    plt.close(
        figure
    )

def number_to_tag(
    value,
):
    return (
        f"{float(value):.2f}"
        .replace(
            "-",
            "m",
        )
        .replace(
            ".",
            "p",
        )
    )

def safe_pearson(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    finite = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
    )

    x = x[
        finite
    ]

    y = y[
        finite
    ]

    if len(
        x
    ) < 3:
        return np.nan

    if (
        np.std(
            x
        ) <= 0
        or np.std(
            y
        ) <= 0
    ):
        return np.nan

    return float(
        stats.pearsonr(
            x,
            y,
        ).statistic
    )

def safe_spearman(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    finite = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
    )

    x = x[
        finite
    ]

    y = y[
        finite
    ]

    if len(
        x
    ) < 3:
        return np.nan

    if (
        np.std(
            x
        ) <= 0
        or np.std(
            y
        ) <= 0
    ):
        return np.nan

    return float(
        stats.spearmanr(
            x,
            y,
        ).statistic
    )

def safe_r2_uncentered(
    observed,
    predicted,
):
    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    finite = (
        np.isfinite(
            observed
        )
        & np.isfinite(
            predicted
        )
    )

    observed = observed[
        finite
    ]

    predicted = predicted[
        finite
    ]

    if len(
        observed
    ) == 0:
        return np.nan

    denominator = float(
        np.sum(
            observed**2
        )
    )

    if denominator <= 0:
        return np.nan

    numerator = float(
        np.sum(
            (
                observed
                - predicted
            )**2
        )
    )

    return float(
        1.0
        - numerator
        / denominator
    )

def safe_r2_centered(
    observed,
    predicted,
):
    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    finite = (
        np.isfinite(
            observed
        )
        & np.isfinite(
            predicted
        )
    )

    observed = observed[
        finite
    ]

    predicted = predicted[
        finite
    ]

    if len(
        observed
    ) < 2:
        return np.nan

    denominator = float(
        np.sum(
            (
                observed
                - np.mean(
                    observed
                )
            )**2
        )
    )

    if denominator <= 0:
        return np.nan

    numerator = float(
        np.sum(
            (
                observed
                - predicted
            )**2
        )
    )

    return float(
        1.0
        - numerator
        / denominator
    )

def rank_zscore(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    output = np.full(
        len(
            values
        ),
        np.nan,
        dtype=float,
    )

    finite = np.isfinite(
        values
    )

    if np.sum(
        finite
    ) < 2:
        return output

    ranks = stats.rankdata(
        values[
            finite
        ],
        method="average",
    )

    rank_std = float(
        np.std(
            ranks,
            ddof=0,
        )
    )

    if (
        not np.isfinite(
            rank_std
        )
        or rank_std <= 0
    ):
        return output

    output[
        finite
    ] = (
        ranks
        - np.mean(
            ranks
        )
    ) / rank_std

    return output

def bootstrap_correlation_interval(
    x,
    y,
    correlation_function,
    n_bootstrap=2000,
    seed=12345,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    finite = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
    )

    x = x[
        finite
    ]

    y = y[
        finite
    ]

    if len(
        x
    ) < 3:
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_values = []

    for _ in range(
        int(
            n_bootstrap
        )
    ):
        indices = rng.integers(
            0,
            len(
                x
            ),
            size=len(
                x
            ),
        )

        value = correlation_function(
            x[
                indices
            ],
            y[
                indices
            ],
        )

        if np.isfinite(
            value
        ):
            bootstrap_values.append(
                value
            )

    if not bootstrap_values:
        return (
            np.nan,
            np.nan,
        )

    bootstrap_values = np.asarray(
        bootstrap_values,
        dtype=float,
    )

    return (
        float(
            np.quantile(
                bootstrap_values,
                0.025,
            )
        ),
        float(
            np.quantile(
                bootstrap_values,
                0.975,
            )
        ),
    )

def safe_centered_r2(
    observed,
    predicted,
):
    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    finite = (
        np.isfinite(
            observed
        )
        & np.isfinite(
            predicted
        )
    )

    observed = observed[
        finite
    ]

    predicted = predicted[
        finite
    ]

    if len(
        observed
    ) < 2:
        return np.nan

    denominator = float(
        np.sum(
            (
                observed
                - np.mean(
                    observed
                )
            ) ** 2
        )
    )

    if (
        not np.isfinite(
            denominator
        )
        or denominator <= 0
    ):
        return np.nan

    numerator = float(
        np.sum(
            (
                observed
                - predicted
            ) ** 2
        )
    )

    return float(
        1.0
        - numerator
        / denominator
    )

def safe_uncentered_r2(
    observed,
    predicted,
):
    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    finite = (
        np.isfinite(
            observed
        )
        & np.isfinite(
            predicted
        )
    )

    observed = observed[
        finite
    ]

    predicted = predicted[
        finite
    ]

    if len(
        observed
    ) == 0:
        return np.nan

    denominator = float(
        np.sum(
            observed ** 2
        )
    )

    if (
        not np.isfinite(
            denominator
        )
        or denominator <= 0
    ):
        return np.nan

    numerator = float(
        np.sum(
            (
                observed
                - predicted
            ) ** 2
        )
    )

    return float(
        1.0
        - numerator
        / denominator
    )

def make_filtered_dataframe(
    full_dataframe,
    r2_cutoff,
):
    filtered = full_dataframe.loc[
        full_dataframe[
            "fit_r2"
        ]
        >= float(
            r2_cutoff
        )
    ].copy()

    return filtered

def bootstrap_dataset_spearman(
    observed_dataset_medians,
    predicted_dataset_medians,
    n_bootstrap,
    seed,
):
    observed_dataset_medians = np.asarray(
        observed_dataset_medians,
        dtype=float,
    )

    predicted_dataset_medians = np.asarray(
        predicted_dataset_medians,
        dtype=float,
    )

    finite = (
        np.isfinite(
            observed_dataset_medians
        )
        & np.isfinite(
            predicted_dataset_medians
        )
    )

    observed_dataset_medians = (
        observed_dataset_medians[
            finite
        ]
    )

    predicted_dataset_medians = (
        predicted_dataset_medians[
            finite
        ]
    )

    n_datasets = len(
        observed_dataset_medians
    )

    if (
        n_bootstrap <= 0
        or n_datasets < MIN_DATASETS_FOR_SPEARMAN
    ):
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_correlations = []

    for _ in range(
        int(
            n_bootstrap
        )
    ):
        indices = rng.integers(
            low=0,
            high=n_datasets,
            size=n_datasets,
        )

        bootstrap_correlation = safe_spearman(
            observed_dataset_medians[
                indices
            ],
            predicted_dataset_medians[
                indices
            ],
        )

        if np.isfinite(
            bootstrap_correlation
        ):
            bootstrap_correlations.append(
                bootstrap_correlation
            )

    if len(
        bootstrap_correlations
    ) == 0:
        return (
            np.nan,
            np.nan,
        )

    bootstrap_correlations = np.asarray(
        bootstrap_correlations,
        dtype=float,
    )

    return (
        float(
            np.quantile(
                bootstrap_correlations,
                0.025,
            )
        ),
        float(
            np.quantile(
                bootstrap_correlations,
                0.975,
            )
        ),
    )

def calculate_dataset_medians(
    filtered_dataframe,
    cutoff,
):
    if len(
        filtered_dataframe
    ) == 0:
        return pd.DataFrame(
            columns=[
                "r2_cutoff",
                DATASET_COLUMN,
                "n_perturbations",
                "median_observed_neff",
                "median_predicted_neff",
            ]
        )

    dataset_medians = (
        filtered_dataframe
        .groupby(
            DATASET_COLUMN,
            sort=False,
        )
        .agg(
            n_perturbations=(
                PERTURBATION_COLUMN,
                "size",
            ),
            median_observed_neff=(
                "observed_neff",
                "median",
            ),
            median_predicted_neff=(
                "predicted_neff",
                "median",
            ),
        )
        .reset_index()
    )

    dataset_medians = dataset_medians.loc[
        dataset_medians[
            "n_perturbations"
        ]
        >= MIN_PERTURBATIONS_PER_DATASET
    ].copy()

    dataset_medians[
        "r2_cutoff"
    ] = float(
        cutoff
    )

    dataset_medians[
        "median_predicted_to_observed_ratio"
    ] = (
        dataset_medians[
            "median_predicted_neff"
        ]
        / dataset_medians[
            "median_observed_neff"
        ]
    )

    return dataset_medians

def filter_by_r2(
    dataframe,
    cutoff,
):
    return dataframe.loc[
        dataframe[
            "fit_r2"
        ]
        >= float(
            cutoff
        )
    ].copy()

def calculate_sweep_row(
    complete_dataframe,
    filtered_dataframe,
    dataset_medians,
    cutoff,
    bootstrap_seed,
):
    observed = filtered_dataframe[
        "observed_neff"
    ].to_numpy(
        dtype=float
    )

    predicted = filtered_dataframe[
        "predicted_neff"
    ].to_numpy(
        dtype=float
    )

    dataset_observed = dataset_medians[
        "median_observed_neff"
    ].to_numpy(
        dtype=float
    )

    dataset_predicted = dataset_medians[
        "median_predicted_neff"
    ].to_numpy(
        dtype=float
    )

    dataset_spearman = safe_spearman(
        dataset_observed,
        dataset_predicted,
    )

    (
        dataset_spearman_ci_low,
        dataset_spearman_ci_high,
    ) = bootstrap_dataset_spearman(
        observed_dataset_medians=dataset_observed,
        predicted_dataset_medians=dataset_predicted,
        n_bootstrap=N_BOOTSTRAP,
        seed=bootstrap_seed,
    )

    perturbation_pearson_log10 = safe_pearson(
        np.log10(
            observed
        ),
        np.log10(
            predicted
        ),
    )

    perturbation_spearman = safe_spearman(
        observed,
        predicted,
    )

    median_absolute_relative_error = float(
        np.median(
            np.abs(
                predicted
                - observed
            )
            / observed
        )
    )

    return {
        "r2_cutoff": float(
            cutoff
        ),
        "n_perturbations": int(
            len(
                filtered_dataframe
            )
        ),
        "fraction_perturbations_retained": float(
            len(
                filtered_dataframe
            )
            / max(
                len(
                    complete_dataframe
                ),
                1,
            )
        ),
        "n_datasets": int(
            len(
                dataset_medians
            )
        ),
        "dataset_median_spearman": (
            dataset_spearman
        ),
        "dataset_median_spearman_ci_low": (
            dataset_spearman_ci_low
        ),
        "dataset_median_spearman_ci_high": (
            dataset_spearman_ci_high
        ),
        "perturbation_pearson_log10": (
            perturbation_pearson_log10
        ),
        "perturbation_spearman": (
            perturbation_spearman
        ),
        "median_absolute_relative_error": (
            median_absolute_relative_error
        ),
    }

def calculate_shared_log_limits(
    dataframes,
    observed_column,
    predicted_column,
):
    all_observed = np.concatenate(
        [
            dataframe[
                observed_column
            ].to_numpy(
                dtype=float
            )
            for dataframe in dataframes
        ]
    )

    all_predicted = np.concatenate(
        [
            dataframe[
                predicted_column
            ].to_numpy(
                dtype=float
            )
            for dataframe in dataframes
        ]
    )

    minimum_positive = float(
        min(
            np.min(
                all_observed
            ),
            np.min(
                all_predicted
            ),
        )
    )

    maximum_positive = float(
        max(
            np.max(
                all_observed
            ),
            np.max(
                all_predicted
            ),
        )
    )

    minimum_limit = float(
        10 ** np.floor(
            np.log10(
                minimum_positive
            )
        )
    )

    maximum_limit = float(
        10 ** np.ceil(
            np.log10(
                maximum_positive
            )
        )
    )

    return (
        minimum_limit,
        maximum_limit,
    )

def add_perturbation_loglog_panel(
    axis,
    filtered_dataframe,
    cutoff,
    minimum_limit,
    maximum_limit,
    panel_label,
):
    observed = filtered_dataframe[
        "observed_neff"
    ].to_numpy(
        dtype=float
    )

    predicted = filtered_dataframe[
        "predicted_neff"
    ].to_numpy(
        dtype=float
    )

    pearson_log10 = safe_pearson(
        np.log10(
            observed
        ),
        np.log10(
            predicted
        ),
    )

    spearman = safe_spearman(
        observed,
        predicted,
    )

    median_relative_error = float(
        np.median(
            np.abs(
                predicted
                - observed
            )
            / observed
        )
    )

    n_datasets = int(
        filtered_dataframe[
            DATASET_COLUMN
        ].nunique()
    )

    axis.scatter(
        observed,
        predicted,
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
        rasterized=True,
    )

    axis.plot(
        [
            minimum_limit,
            maximum_limit,
        ],
        [
            minimum_limit,
            maximum_limit,
        ],
        linestyle="--",
        linewidth=IDENTITY_LINE_WIDTH,
        label="Observed = predicted",
    )

    axis.set_xscale(
        "log"
    )

    axis.set_yscale(
        "log"
    )

    axis.set_xlim(
        minimum_limit,
        maximum_limit,
    )

    axis.set_ylim(
        minimum_limit,
        maximum_limit,
    )

    axis.text(
        0.04,
        0.96,
        (
            f"R² cutoff ≥ {cutoff:.2f}\n"
            f"n = {len(filtered_dataframe):,}\n"
            f"Datasets = {n_datasets}\n"
            f"Pearson(log10) = {pearson_log10:.3f}\n"
            f"Spearman = {spearman:.3f}\n"
            f"Median |relative error| = "
            f"{median_relative_error:.3f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.3,
    )

    axis.text(
        -0.12,
        1.07,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    axis.set_xlabel(
        "Observed effective number of genes"
    )

    axis.set_ylabel(
        "CIPHER-predicted effective number of genes"
    )

    axis.set_title(
        f"Perturbations with forward R² ≥ {cutoff:.2f}"
    )

    axis.grid(
        alpha=GRID_ALPHA
    )

