"""Notebook-only run module for Fig S15 (observed vs CIPHER-predicted effective
number of responding genes / response breadth).

Relocates the three big main-flow cells of
``notebooks/suppl/figS15_effective_N.ipynb`` VERBATIM into one function per
analysis block. The cell bodies are unchanged (same variables, same plt/savefig
calls, same logic); only the illegal in-function ``from src.suppl_effn import *``
line is commented out (the same names are provided by the module-level import
below). Config is read as MODULE GLOBALS -- the thin notebook injects them into
this module's __dict__ at runtime. NOT part of the installable ``cipher`` package.
"""
from src.suppl_effn import *

# --- library imports mirroring the cluster module (resolved at call time) ---
import os, re, glob, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Patch
from matplotlib import gridspec
from scipy import stats
from scipy.sparse import issparse, csr_matrix
from scipy.stats import wilcoxon, ttest_rel, ks_2samp, mannwhitneyu, pearsonr, spearmanr
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *a, **k):
        return x
# --- end imports ---


def block1_observed_vs_predicted_neff():
    """Block 1 -- observed vs CIPHER-predicted N_eff (R2>=0.5) + permutation null (.npz)."""
    # reset shared helpers to the module canonical (guards against cross-cell shadowing)

    # from src.suppl_effn import *  (provided by module-level import in header)

    # --- helpers with cell-specific variants, kept inline for a faithful 1:1 reproduction ---

    def number_to_tag(
        value,
    ):
        return (
            f"{float(value):g}"
            .replace(
                "-",
                "m",
            )
            .replace(
                ".",
                "p",
            )
        )

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            return iterable

    ANALYSIS_ROOT = Path(OUTDIR)

    OBSERVED_NEFF_COLUMN = (
        "observed_excluding_target_neff_shannon"
    )

    PREDICTED_NEFF_COLUMN = (
        "predicted_excluding_target_neff_shannon"
    )

    FILTER_MODE = "r2"

    R2_THRESHOLD = 0.5

    PEARSON_THRESHOLD = 0.90

    R2_COLUMN = (
        "r2_uncentered_excluding_target"
    )

    PEARSON_COLUMN = (
        "pearson_excluding_target"
    )

    N_PERMUTATIONS = 10000

    PERMUTATION_BATCH_SIZE = 250

    RANDOM_SEED = 314159

    DPI = 300

    SAVE_PDF = True

    SAVE_SVG = True

    SHOW_FIGURES = True

    SCATTER_SIZE = 18

    SCATTER_ALPHA = 0.20

    MIN_PERTURBATIONS_PER_DATASET = 3

    normalized_filter_mode = str(
        FILTER_MODE
    ).strip().lower()

    if normalized_filter_mode == "none":
        FILTER_TAG = (
            "all_perturbations"
        )

    elif normalized_filter_mode == "r2":
        FILTER_TAG = (
            f"r2_ge_{number_to_tag(R2_THRESHOLD)}"
        )

    elif normalized_filter_mode == "pearson":
        FILTER_TAG = (
            f"pearson_ge_{number_to_tag(PEARSON_THRESHOLD)}"
        )

    elif normalized_filter_mode == "either":
        FILTER_TAG = (
            f"r2_ge_{number_to_tag(R2_THRESHOLD)}"
            f"_or_pearson_ge_{number_to_tag(PEARSON_THRESHOLD)}"
        )

    elif normalized_filter_mode == "both":
        FILTER_TAG = (
            f"r2_ge_{number_to_tag(R2_THRESHOLD)}"
            f"_and_pearson_ge_{number_to_tag(PEARSON_THRESHOLD)}"
        )

    else:
        raise ValueError(
            "FILTER_MODE must be one of:\n"
            "  none\n"
            "  r2\n"
            "  pearson\n"
            "  either\n"
            "  both\n\n"
            f"Received: {FILTER_MODE!r}"
        )

    OUT_TABLE_DIR = (
        ANALYSIS_ROOT
        / "tables"
        / "observed_vs_cipher_neff"
        / FILTER_TAG
    )

    OUT_FIGURE_DIR = (
        ANALYSIS_ROOT
        / "figures"
        / "observed_vs_cipher_neff"
        / FILTER_TAG
    )

    OUT_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not BREADTH_PATH.exists():
        raise FileNotFoundError(
            f"Could not find:\n"
            f"    {BREADTH_PATH}"
        )

    breadth_df = pd.read_csv(
        BREADTH_PATH,
        sep="\t",
    )

    required_columns = [
        "dataset_base",
        "perturbation",
        "target_gene",
        "status",
        OBSERVED_NEFF_COLUMN,
        PREDICTED_NEFF_COLUMN,
        R2_COLUMN,
        PEARSON_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in breadth_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "The response-breadth table is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
            + "\n\nAvailable columns:\n  "
            + "\n  ".join(
                map(
                    str,
                    breadth_df.columns,
                )
            )
        )

    analysis_df = breadth_df.loc[
        breadth_df[
            "status"
        ].astype(
            str
        )
        == "ok"
    ].copy()

    numeric_columns = [
        OBSERVED_NEFF_COLUMN,
        PREDICTED_NEFF_COLUMN,
        R2_COLUMN,
        PEARSON_COLUMN,
    ]

    for column in numeric_columns:
        analysis_df[
            column
        ] = pd.to_numeric(
            analysis_df[
                column
            ],
            errors="coerce",
        )

    analysis_df = analysis_df.loc[
        np.isfinite(
            analysis_df[
                OBSERVED_NEFF_COLUMN
            ]
        )
        & np.isfinite(
            analysis_df[
                PREDICTED_NEFF_COLUMN
            ]
        )
        & (
            analysis_df[
                OBSERVED_NEFF_COLUMN
            ]
            > 0
        )
        & (
            analysis_df[
                PREDICTED_NEFF_COLUMN
            ]
            > 0
        )
    ].copy()

    analysis_df[
        "observed_neff"
    ] = analysis_df[
        OBSERVED_NEFF_COLUMN
    ].astype(
        float
    )

    analysis_df[
        "predicted_neff"
    ] = analysis_df[
        PREDICTED_NEFF_COLUMN
    ].astype(
        float
    )

    analysis_df[
        "fit_r2"
    ] = analysis_df[
        R2_COLUMN
    ].astype(
        float
    )

    analysis_df[
        "fit_pearson"
    ] = analysis_df[
        PEARSON_COLUMN
    ].astype(
        float
    )

    analysis_df[
        "passes_r2_filter"
    ] = (
        np.isfinite(
            analysis_df[
                "fit_r2"
            ]
        )
        & (
            analysis_df[
                "fit_r2"
            ]
            >= float(
                R2_THRESHOLD
            )
        )
    )

    analysis_df[
        "passes_pearson_filter"
    ] = (
        np.isfinite(
            analysis_df[
                "fit_pearson"
            ]
        )
        & (
            analysis_df[
                "fit_pearson"
            ]
            >= float(
                PEARSON_THRESHOLD
            )
        )
    )

    n_before_filter = len(
        analysis_df
    )

    if normalized_filter_mode == "none":
        keep = np.ones(
            len(
                analysis_df
            ),
            dtype=bool,
        )

    elif normalized_filter_mode == "r2":
        keep = analysis_df[
            "passes_r2_filter"
        ].to_numpy(
            dtype=bool
        )

    elif normalized_filter_mode == "pearson":
        keep = analysis_df[
            "passes_pearson_filter"
        ].to_numpy(
            dtype=bool
        )

    elif normalized_filter_mode == "either":
        keep = (
            analysis_df[
                "passes_r2_filter"
            ].to_numpy(
                dtype=bool
            )
            | analysis_df[
                "passes_pearson_filter"
            ].to_numpy(
                dtype=bool
            )
        )

    elif normalized_filter_mode == "both":
        keep = (
            analysis_df[
                "passes_r2_filter"
            ].to_numpy(
                dtype=bool
            )
            & analysis_df[
                "passes_pearson_filter"
            ].to_numpy(
                dtype=bool
            )
        )

    else:
        raise RuntimeError(
            f"Unexpected FILTER_MODE: {FILTER_MODE!r}"
        )

    analysis_df[
        "passes_selected_filter"
    ] = keep

    filter_summary_df = (
        analysis_df
        .groupby(
            "dataset_base",
            sort=False,
        )
        .agg(
            n_before_filter=(
                "perturbation",
                "size",
            ),
            n_passing_r2=(
                "passes_r2_filter",
                "sum",
            ),
            n_passing_pearson=(
                "passes_pearson_filter",
                "sum",
            ),
            n_passing_selected_filter=(
                "passes_selected_filter",
                "sum",
            ),
        )
        .reset_index()
    )

    filter_summary_df[
        "fraction_passing_selected_filter"
    ] = (
        filter_summary_df[
            "n_passing_selected_filter"
        ]
        / filter_summary_df[
            "n_before_filter"
        ].clip(
            lower=1
        )
    )

    analysis_df = analysis_df.loc[
        analysis_df[
            "passes_selected_filter"
        ]
    ].copy()

    n_after_filter = len(
        analysis_df
    )

    if n_after_filter == 0:
        raise RuntimeError(
            "No perturbations passed the requested fit-quality filter.\n\n"
            f"FILTER_MODE = {FILTER_MODE!r}\n"
            f"R2_THRESHOLD = {R2_THRESHOLD:g}\n"
            f"PEARSON_THRESHOLD = {PEARSON_THRESHOLD:g}"
        )

    analysis_df[
        "neff_difference"
    ] = (
        analysis_df[
            "predicted_neff"
        ]
        - analysis_df[
            "observed_neff"
        ]
    )

    analysis_df[
        "absolute_neff_error"
    ] = np.abs(
        analysis_df[
            "neff_difference"
        ]
    )

    analysis_df[
        "relative_neff_error"
    ] = (
        analysis_df[
            "neff_difference"
        ]
        / analysis_df[
            "observed_neff"
        ]
    )

    analysis_df[
        "absolute_relative_neff_error"
    ] = np.abs(
        analysis_df[
            "relative_neff_error"
        ]
    )

    analysis_df[
        "neff_ratio"
    ] = (
        analysis_df[
            "predicted_neff"
        ]
        / analysis_df[
            "observed_neff"
        ]
    )

    analysis_df[
        "log10_observed_neff"
    ] = np.log10(
        analysis_df[
            "observed_neff"
        ]
    )

    analysis_df[
        "log10_predicted_neff"
    ] = np.log10(
        analysis_df[
            "predicted_neff"
        ]
    )

    analysis_df[
        "log10_neff_ratio"
    ] = np.log10(
        analysis_df[
            "neff_ratio"
        ]
    )

    analysis_df.to_csv(
        OUT_TABLE_DIR
        / "observed_vs_cipher_neff_per_perturbation.tsv",
        sep="\t",
        index=False,
    )

    filter_summary_df.to_csv(
        OUT_TABLE_DIR
        / "fit_quality_filter_summary_by_dataset.tsv",
        sep="\t",
        index=False,
    )

    analysis_df[
        "observed_rank_z"
    ] = np.nan

    analysis_df[
        "predicted_rank_z"
    ] = np.nan

    dataset_index_groups = []

    for dataset_base, index_values in (
        analysis_df.groupby(
            "dataset_base",
            sort=False,
        ).groups.items()
    ):
        indices = np.asarray(
            list(
                index_values
            )
        )

        if len(
            indices
        ) < MIN_PERTURBATIONS_PER_DATASET:
            continue

        observed_rank_z = rank_zscore(
            analysis_df.loc[
                indices,
                "observed_neff",
            ].to_numpy(
                dtype=float
            )
        )

        predicted_rank_z = rank_zscore(
            analysis_df.loc[
                indices,
                "predicted_neff",
            ].to_numpy(
                dtype=float
            )
        )

        analysis_df.loc[
            indices,
            "observed_rank_z",
        ] = observed_rank_z

        analysis_df.loc[
            indices,
            "predicted_rank_z",
        ] = predicted_rank_z

        valid_rank = (
            np.isfinite(
                observed_rank_z
            )
            & np.isfinite(
                predicted_rank_z
            )
        )

        if np.sum(
            valid_rank
        ) >= MIN_PERTURBATIONS_PER_DATASET:
            dataset_index_groups.append(
                (
                    dataset_base,
                    indices,
                )
            )

    analysis_df.to_csv(
        OUT_TABLE_DIR
        / "observed_vs_cipher_neff_per_perturbation.tsv",
        sep="\t",
        index=False,
    )

    observed_neff = analysis_df[
        "observed_neff"
    ].to_numpy(
        dtype=float
    )

    predicted_neff = analysis_df[
        "predicted_neff"
    ].to_numpy(
        dtype=float
    )

    observed_log_neff = analysis_df[
        "log10_observed_neff"
    ].to_numpy(
        dtype=float
    )

    predicted_log_neff = analysis_df[
        "log10_predicted_neff"
    ].to_numpy(
        dtype=float
    )

    overall_pearson = safe_pearson(
        observed_neff,
        predicted_neff,
    )

    overall_spearman = safe_spearman(
        observed_neff,
        predicted_neff,
    )

    overall_log_pearson = safe_pearson(
        observed_log_neff,
        predicted_log_neff,
    )

    overall_log_spearman = safe_spearman(
        observed_log_neff,
        predicted_log_neff,
    )

    overall_centered_r2 = safe_r2_centered(
        observed_neff,
        predicted_neff,
    )

    overall_uncentered_r2 = safe_r2_uncentered(
        observed_neff,
        predicted_neff,
    )

    median_absolute_error = float(
        np.median(
            analysis_df[
                "absolute_neff_error"
            ]
        )
    )

    mean_absolute_error = float(
        np.mean(
            analysis_df[
                "absolute_neff_error"
            ]
        )
    )

    median_absolute_relative_error = float(
        np.median(
            analysis_df[
                "absolute_relative_neff_error"
            ]
        )
    )

    median_ratio = float(
        np.median(
            analysis_df[
                "neff_ratio"
            ]
        )
    )

    pearson_ci_low, pearson_ci_high = (
        bootstrap_correlation_interval(
            observed_neff,
            predicted_neff,
            safe_pearson,
            n_bootstrap=2000,
            seed=RANDOM_SEED,
        )
    )

    spearman_ci_low, spearman_ci_high = (
        bootstrap_correlation_interval(
            observed_neff,
            predicted_neff,
            safe_spearman,
            n_bootstrap=2000,
            seed=RANDOM_SEED + 1,
        )
    )

    overall_summary_df = pd.DataFrame(
        [
            {
                "filter_mode": FILTER_MODE,
                "r2_threshold": R2_THRESHOLD,
                "pearson_threshold": PEARSON_THRESHOLD,
                "n_before_filter": n_before_filter,
                "n_after_filter": n_after_filter,
                "fraction_retained": (
                    n_after_filter
                    / max(
                        n_before_filter,
                        1,
                    )
                ),
                "n_datasets": int(
                    analysis_df[
                        "dataset_base"
                    ].nunique()
                ),
                "pearson_neff": overall_pearson,
                "pearson_neff_ci_low": pearson_ci_low,
                "pearson_neff_ci_high": pearson_ci_high,
                "spearman_neff": overall_spearman,
                "spearman_neff_ci_low": spearman_ci_low,
                "spearman_neff_ci_high": spearman_ci_high,
                "pearson_log10_neff": overall_log_pearson,
                "spearman_log10_neff": overall_log_spearman,
                "centered_r2_neff": overall_centered_r2,
                "uncentered_r2_neff": overall_uncentered_r2,
                "median_observed_neff": float(
                    np.median(
                        observed_neff
                    )
                ),
                "median_predicted_neff": float(
                    np.median(
                        predicted_neff
                    )
                ),
                "mean_absolute_error": mean_absolute_error,
                "median_absolute_error": median_absolute_error,
                "median_absolute_relative_error": (
                    median_absolute_relative_error
                ),
                "median_predicted_to_observed_ratio": median_ratio,
            }
        ]
    )

    overall_summary_df.to_csv(
        OUT_TABLE_DIR
        / "observed_vs_cipher_neff_overall_summary.tsv",
        sep="\t",
        index=False,
    )

    per_dataset_rows = []

    for dataset_base, dataset_df in analysis_df.groupby(
        "dataset_base",
        sort=False,
    ):
        dataset_observed = dataset_df[
            "observed_neff"
        ].to_numpy(
            dtype=float
        )

        dataset_predicted = dataset_df[
            "predicted_neff"
        ].to_numpy(
            dtype=float
        )

        dataset_log_observed = np.log10(
            dataset_observed
        )

        dataset_log_predicted = np.log10(
            dataset_predicted
        )

        per_dataset_rows.append(
            {
                "dataset_base": dataset_base,
                "n_perturbations": len(
                    dataset_df
                ),
                "pearson_neff": safe_pearson(
                    dataset_observed,
                    dataset_predicted,
                ),
                "spearman_neff": safe_spearman(
                    dataset_observed,
                    dataset_predicted,
                ),
                "pearson_log10_neff": safe_pearson(
                    dataset_log_observed,
                    dataset_log_predicted,
                ),
                "spearman_log10_neff": safe_spearman(
                    dataset_log_observed,
                    dataset_log_predicted,
                ),
                "centered_r2_neff": safe_r2_centered(
                    dataset_observed,
                    dataset_predicted,
                ),
                "uncentered_r2_neff": safe_r2_uncentered(
                    dataset_observed,
                    dataset_predicted,
                ),
                "median_observed_neff": float(
                    np.median(
                        dataset_observed
                    )
                ),
                "median_predicted_neff": float(
                    np.median(
                        dataset_predicted
                    )
                ),
                "median_predicted_to_observed_ratio": float(
                    np.median(
                        dataset_predicted
                        / dataset_observed
                    )
                ),
                "median_absolute_error": float(
                    np.median(
                        np.abs(
                            dataset_predicted
                            - dataset_observed
                        )
                    )
                ),
                "median_absolute_relative_error": float(
                    np.median(
                        np.abs(
                            dataset_predicted
                            - dataset_observed
                        )
                        / dataset_observed
                    )
                ),
            }
        )

    per_dataset_df = pd.DataFrame(
        per_dataset_rows
    )

    per_dataset_df.to_csv(
        OUT_TABLE_DIR
        / "observed_vs_cipher_neff_per_dataset.tsv",
        sep="\t",
        index=False,
    )

    valid_rank_rows = (
        np.isfinite(
            analysis_df[
                "observed_rank_z"
            ]
        )
        & np.isfinite(
            analysis_df[
                "predicted_rank_z"
            ]
        )
    )

    rank_analysis_df = analysis_df.loc[
        valid_rank_rows
    ].copy()

    if len(
        rank_analysis_df
    ) < 3:
        raise RuntimeError(
            "Too few perturbations remain for the pooled "
            "within-dataset rank analysis."
        )

    observed_pooled_rank_statistic = float(
        np.mean(
            rank_analysis_df[
                "observed_rank_z"
            ].to_numpy(
                dtype=float
            )
            * rank_analysis_df[
                "predicted_rank_z"
            ].to_numpy(
                dtype=float
            )
        )
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    null_statistics = np.zeros(
        N_PERMUTATIONS,
        dtype=np.float64,
    )

    total_valid_rank_rows = 0

    print(
        "\n"
        + "=" * 110
    )

    print(
        "RUNNING WITHIN-DATASET PERMUTATION TEST"
    )

    print(
        f"Datasets:      {len(dataset_index_groups)}"
    )

    print(
        f"Permutations:  {N_PERMUTATIONS:,}"
    )

    print(
        "=" * 110
    )

    for dataset_number, (
        dataset_base,
        indices,
    ) in enumerate(
        dataset_index_groups,
        start=1,
    ):
        dataset_observed_rank = analysis_df.loc[
            indices,
            "observed_rank_z",
        ].to_numpy(
            dtype=float
        )

        dataset_predicted_rank = analysis_df.loc[
            indices,
            "predicted_rank_z",
        ].to_numpy(
            dtype=float
        )

        finite = (
            np.isfinite(
                dataset_observed_rank
            )
            & np.isfinite(
                dataset_predicted_rank
            )
        )

        dataset_observed_rank = dataset_observed_rank[
            finite
        ]

        dataset_predicted_rank = dataset_predicted_rank[
            finite
        ]

        if len(
            dataset_observed_rank
        ) < MIN_PERTURBATIONS_PER_DATASET:
            continue

        total_valid_rank_rows += len(
            dataset_observed_rank
        )

        print(
            f"[{dataset_number}/{len(dataset_index_groups)}] "
            f"{dataset_base}: "
            f"{len(dataset_observed_rank):,} perturbations"
        )

        for permutation_start in tqdm(
            range(
                0,
                N_PERMUTATIONS,
                PERMUTATION_BATCH_SIZE,
            ),
            desc=dataset_base,
            ncols=110,
            leave=False,
        ):
            permutation_end = min(
                permutation_start
                + PERMUTATION_BATCH_SIZE,
                N_PERMUTATIONS,
            )

            for permutation_index in range(
                permutation_start,
                permutation_end,
            ):
                permuted_predicted = rng.permutation(
                    dataset_predicted_rank
                )

                null_statistics[
                    permutation_index
                ] += float(
                    np.sum(
                        dataset_observed_rank
                        * permuted_predicted
                    )
                )

    if total_valid_rank_rows <= 0:
        raise RuntimeError(
            "No valid rows were available for permutation testing."
        )

    null_statistics /= total_valid_rank_rows

    permutation_p_positive = float(
        (
            1
            + np.sum(
                null_statistics
                >= observed_pooled_rank_statistic
            )
        )
        / (
            N_PERMUTATIONS
            + 1
        )
    )

    permutation_p_two_sided = float(
        (
            1
            + np.sum(
                np.abs(
                    null_statistics
                )
                >= abs(
                    observed_pooled_rank_statistic
                )
            )
        )
        / (
            N_PERMUTATIONS
            + 1
        )
    )

    permutation_summary_df = pd.DataFrame(
        [
            {
                "pooled_within_dataset_rank_statistic": (
                    observed_pooled_rank_statistic
                ),
                "permutation_p_positive": (
                    permutation_p_positive
                ),
                "permutation_p_two_sided": (
                    permutation_p_two_sided
                ),
                "n_permutations": N_PERMUTATIONS,
                "n_valid_perturbations": (
                    total_valid_rank_rows
                ),
                "n_datasets": len(
                    dataset_index_groups
                ),
            }
        ]
    )

    permutation_summary_df.to_csv(
        OUT_TABLE_DIR
        / "observed_vs_cipher_neff_permutation_summary.tsv",
        sep="\t",
        index=False,
    )

    np.savez_compressed(
        OUT_TABLE_DIR
        / "observed_vs_cipher_neff_permutation_null.npz",
        observed_statistic=(
            observed_pooled_rank_statistic
        ),
        null_statistics=null_statistics,
    )

    figure, axis = plt.subplots(
        figsize=(
            7.5,
            7.0,
        )
    )

    axis.scatter(
        observed_neff,
        predicted_neff,
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
    )

    minimum_value = float(
        min(
            np.min(
                observed_neff
            ),
            np.min(
                predicted_neff
            ),
        )
    )

    maximum_value = float(
        max(
            np.max(
                observed_neff
            ),
            np.max(
                predicted_neff
            ),
        )
    )

    axis.plot(
        [
            minimum_value,
            maximum_value,
        ],
        [
            minimum_value,
            maximum_value,
        ],
        linestyle="--",
        linewidth=1.5,
        label="Observed = predicted",
    )

    axis.text(
        0.04,
        0.96,
        (
            f"n = {len(analysis_df):,}\n"
            f"Pearson = {overall_pearson:.3f}\n"
            f"Spearman = {overall_spearman:.3f}\n"
            f"Centered R² = {overall_centered_r2:.3f}\n"
            f"Median ratio = {median_ratio:.3f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    axis.set_xlabel(
        "Observed effective number of genes"
    )

    axis.set_ylabel(
        "CIPHER-predicted effective number of genes"
    )

    axis.set_title(
        "Observed versus CIPHER-predicted response breadth\n"
        f"{FILTER_TAG}"
    )

    axis.legend(
        frameon=False
    )

    axis.grid(
        alpha=0.20
    )

    figure.tight_layout()

    save_figure(
        figure,
        OUT_FIGURE_DIR
        / "observed_vs_cipher_neff_scatter",
    )

    figure, axis = plt.subplots(
        figsize=(
            7.5,
            7.0,
        )
    )

    axis.scatter(
        observed_neff,
        predicted_neff,
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
    )

    positive_minimum = float(
        min(
            np.min(
                observed_neff[
                    observed_neff > 0
                ]
            ),
            np.min(
                predicted_neff[
                    predicted_neff > 0
                ]
            ),
        )
    )

    positive_maximum = float(
        max(
            np.max(
                observed_neff
            ),
            np.max(
                predicted_neff
            ),
        )
    )

    axis.plot(
        [
            positive_minimum,
            positive_maximum,
        ],
        [
            positive_minimum,
            positive_maximum,
        ],
        linestyle="--",
        linewidth=1.5,
        label="Observed = predicted",
    )

    axis.set_xscale(
        "log"
    )

    axis.set_yscale(
        "log"
    )

    axis.text(
        0.04,
        0.96,
        (
            f"n = {len(analysis_df):,}\n"
            f"Pearson(log10) = {overall_log_pearson:.3f}\n"
            f"Spearman = {overall_spearman:.3f}\n"
            f"Median |relative error| = "
            f"{median_absolute_relative_error:.3f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    axis.set_xlabel(
        "Observed effective number of genes"
    )

    axis.set_ylabel(
        "CIPHER-predicted effective number of genes"
    )

    axis.set_title(
        "Observed versus CIPHER-predicted response breadth\n"
        "Log-log scale"
    )

    axis.legend(
        frameon=False
    )

    axis.grid(
        alpha=0.20
    )

    figure.tight_layout()

    save_figure(
        figure,
        OUT_FIGURE_DIR
        / "observed_vs_cipher_neff_loglog_scatter",
    )

    figure, axis = plt.subplots(
        figsize=(
            7.5,
            7.0,
        )
    )

    axis.scatter(
        rank_analysis_df[
            "observed_rank_z"
        ],
        rank_analysis_df[
            "predicted_rank_z"
        ],
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
    )

    rank_limit = float(
        max(
            np.nanmax(
                np.abs(
                    rank_analysis_df[
                        "observed_rank_z"
                    ]
                )
            ),
            np.nanmax(
                np.abs(
                    rank_analysis_df[
                        "predicted_rank_z"
                    ]
                )
            ),
        )
    )

    axis.plot(
        [
            -rank_limit,
            rank_limit,
        ],
        [
            -rank_limit,
            rank_limit,
        ],
        linestyle="--",
        linewidth=1.5,
    )

    axis.text(
        0.04,
        0.96,
        (
            f"n = {len(rank_analysis_df):,}\n"
            f"Pooled rank statistic = "
            f"{observed_pooled_rank_statistic:.3f}\n"
            f"Permutation p = "
            f"{permutation_p_positive:.3g}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    axis.set_xlabel(
        "Observed effective-number rank\n"
        "standardized within dataset"
    )

    axis.set_ylabel(
        "CIPHER-predicted effective-number rank\n"
        "standardized within dataset"
    )

    axis.set_title(
        "Within-dataset correspondence of response breadth"
    )

    axis.grid(
        alpha=0.20
    )

    figure.tight_layout()

    save_figure(
        figure,
        OUT_FIGURE_DIR
        / "observed_vs_cipher_neff_within_dataset_rank_scatter",
    )

    forest_df = per_dataset_df.loc[
        np.isfinite(
            per_dataset_df[
                "spearman_neff"
            ]
        )
    ].copy()

    forest_df = forest_df.sort_values(
        "spearman_neff",
        ascending=True,
    )

    if len(
        forest_df
    ) > 0:
        figure, axis = plt.subplots(
            figsize=(
                11,
                max(
                    8,
                    0.38
                    * len(
                        forest_df
                    ),
                ),
            )
        )

        y_positions = np.arange(
            len(
                forest_df
            )
        )

        axis.plot(
            forest_df[
                "spearman_neff"
            ],
            y_positions,
            linestyle="none",
            marker="o",
        )

        axis.axvline(
            0,
            linestyle="--",
            linewidth=1.0,
        )

        axis.set_yticks(
            y_positions
        )

        axis.set_yticklabels(
            [
                (
                    f"{row.dataset_base} "
                    f"(n={int(row.n_perturbations)})"
                )
                for row in forest_df.itertuples(
                    index=False
                )
            ],
            fontsize=8,
        )

        axis.set_xlabel(
            "Spearman correlation between observed and\n"
            "CIPHER-predicted effective number of genes"
        )

        axis.set_title(
            "Per-dataset correspondence of response breadth\n"
            f"{FILTER_TAG}"
        )

        axis.grid(
            axis="x",
            alpha=0.20,
        )

        figure.tight_layout()

        save_figure(
            figure,
            OUT_FIGURE_DIR
            / "observed_vs_cipher_neff_per_dataset_spearman",
        )

    median_df = per_dataset_df.loc[
        np.isfinite(
            per_dataset_df[
                "median_observed_neff"
            ]
        )
        & np.isfinite(
            per_dataset_df[
                "median_predicted_neff"
            ]
        )
    ].copy()

    if len(
        median_df
    ) > 0:
        figure, axis = plt.subplots(
            figsize=(
                7.5,
                7.0,
            )
        )

        axis.scatter(
            median_df[
                "median_observed_neff"
            ],
            median_df[
                "median_predicted_neff"
            ],
            s=45,
            alpha=0.80,
        )

        median_minimum = float(
            min(
                median_df[
                    "median_observed_neff"
                ].min(),
                median_df[
                    "median_predicted_neff"
                ].min(),
            )
        )

        median_maximum = float(
            max(
                median_df[
                    "median_observed_neff"
                ].max(),
                median_df[
                    "median_predicted_neff"
                ].max(),
            )
        )

        axis.plot(
            [
                median_minimum,
                median_maximum,
            ],
            [
                median_minimum,
                median_maximum,
            ],
            linestyle="--",
            linewidth=1.5,
        )

        for row in median_df.itertuples(
            index=False
        ):
            axis.annotate(
                str(
                    row.dataset_base
                ),
                (
                    row.median_observed_neff,
                    row.median_predicted_neff,
                ),
                xytext=(
                    3,
                    3,
                ),
                textcoords="offset points",
                fontsize=6,
                alpha=0.80,
            )

        median_dataset_spearman = safe_spearman(
            median_df[
                "median_observed_neff"
            ],
            median_df[
                "median_predicted_neff"
            ],
        )

        axis.text(
            0.04,
            0.96,
            (
                f"Datasets = {len(median_df)}\n"
                f"Spearman = {median_dataset_spearman:.3f}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

        axis.set_xscale(
            "log"
        )

        axis.set_yscale(
            "log"
        )

        axis.set_xlabel(
            "Median observed effective number of genes"
        )

        axis.set_ylabel(
            "Median CIPHER-predicted effective number of genes"
        )

        axis.set_title(
            "Dataset-level median response breadth"
        )

        axis.grid(
            alpha=0.20
        )

        figure.tight_layout()

        save_figure(
            figure,
            OUT_FIGURE_DIR
            / "observed_vs_cipher_neff_dataset_medians",
        )

    figure, axis = plt.subplots(
        figsize=(
            8,
            5.8,
        )
    )

    finite_log_ratio = analysis_df[
        "log10_neff_ratio"
    ].to_numpy(
        dtype=float
    )

    finite_log_ratio = finite_log_ratio[
        np.isfinite(
            finite_log_ratio
        )
    ]

    axis.hist(
        finite_log_ratio,
        bins=50,
    )

    axis.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
        label="Predicted = observed",
    )

    axis.axvline(
        np.median(
            finite_log_ratio
        ),
        linestyle="-",
        linewidth=1.5,
        label=(
            "Median predicted/observed = "
            f"{median_ratio:.3f}"
        ),
    )

    axis.set_xlabel(
        "log10(CIPHER-predicted N_eff / observed N_eff)"
    )

    axis.set_ylabel(
        "Number of perturbations"
    )

    axis.set_title(
        "Bias in CIPHER-predicted response breadth\n"
        f"{FILTER_TAG}"
    )

    axis.legend(
        frameon=False
    )

    axis.grid(
        alpha=0.20
    )

    figure.tight_layout()

    save_figure(
        figure,
        OUT_FIGURE_DIR
        / "observed_vs_cipher_neff_ratio_histogram",
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "OBSERVED VS CIPHER-PREDICTED EFFECTIVE NUMBER ANALYSIS COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        f"Observed column:      {OBSERVED_NEFF_COLUMN}"
    )

    print(
        f"Predicted column:     {PREDICTED_NEFF_COLUMN}"
    )

    print(
        f"Filter mode:          {FILTER_MODE}"
    )

    print(
        f"R2 threshold:         {R2_THRESHOLD:g}"
    )

    print(
        f"Pearson threshold:    {PEARSON_THRESHOLD:g}"
    )

    print(
        f"Before filtering:     {n_before_filter:,}"
    )

    print(
        f"After filtering:      {n_after_filter:,}"
    )

    print(
        f"Fraction retained:    "
        f"{n_after_filter / max(n_before_filter, 1):.2%}"
    )

    print(
        f"Datasets retained:    "
        f"{analysis_df['dataset_base'].nunique():,}"
    )

    print(
        "-" * 120
    )

    print(
        f"Overall Pearson:      {overall_pearson:.4f}"
    )

    print(
        f"Pearson 95% CI:       "
        f"[{pearson_ci_low:.4f}, {pearson_ci_high:.4f}]"
    )

    print(
        f"Overall Spearman:     {overall_spearman:.4f}"
    )

    print(
        f"Spearman 95% CI:      "
        f"[{spearman_ci_low:.4f}, {spearman_ci_high:.4f}]"
    )

    print(
        f"Log10 Pearson:        {overall_log_pearson:.4f}"
    )

    print(
        f"Centered R2:          {overall_centered_r2:.4f}"
    )

    print(
        f"Uncentered R2:        {overall_uncentered_r2:.4f}"
    )

    print(
        f"Median abs error:     {median_absolute_error:.4f}"
    )

    print(
        f"Median abs rel error: "
        f"{median_absolute_relative_error:.4f}"
    )

    print(
        f"Median pred/obs ratio:{median_ratio: .4f}"
    )

    print(
        "-" * 120
    )

    print(
        f"Pooled within-dataset rank statistic: "
        f"{observed_pooled_rank_statistic:.4f}"
    )

    print(
        f"Permutation p, positive: "
        f"{permutation_p_positive:.6g}"
    )

    print(
        f"Permutation p, two-sided: "
        f"{permutation_p_two_sided:.6g}"
    )

    print(
        "-" * 120
    )

    print(
        f"Tables saved to:\n"
        f"    {OUT_TABLE_DIR}"
    )

    print(
        f"Figures saved to:\n"
        f"    {OUT_FIGURE_DIR}"
    )

    print(
        "=" * 120
    )


def block2_full_r2_cutoff_sweep():
    """Block 2 -- full R2-cutoff sweep (0.00 to 0.90) four-panel composite."""
    # reset shared helpers to the module canonical (guards against cross-cell shadowing)

    # from src.suppl_effn import *  (provided by module-level import in header)

    # --- helpers with cell-specific variants, kept inline for a faithful 1:1 reproduction ---

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
                0,
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
            bootstrap_indices = rng.integers(
                low=0,
                high=n_datasets,
                size=n_datasets,
            )

            bootstrap_observed = (
                observed_dataset_medians[
                    bootstrap_indices
                ]
            )

            bootstrap_predicted = (
                predicted_dataset_medians[
                    bootstrap_indices
                ]
            )

            correlation = safe_spearman(
                bootstrap_observed,
                bootstrap_predicted,
            )

            if np.isfinite(
                correlation
            ):
                bootstrap_correlations.append(
                    correlation
                )

        if len(
            bootstrap_correlations
        ) == 0:
            return (
                np.nan,
                np.nan,
                0,
            )

        bootstrap_correlations = np.asarray(
            bootstrap_correlations,
            dtype=float,
        )

        lower = float(
            np.quantile(
                bootstrap_correlations,
                0.025,
            )
        )

        upper = float(
            np.quantile(
                bootstrap_correlations,
                0.975,
            )
        )

        return (
            lower,
            upper,
            len(
                bootstrap_correlations
            ),
        )

    def calculate_dataset_medians(
        filtered_dataframe,
        r2_cutoff,
    ):
        if len(
            filtered_dataframe
        ) == 0:
            return pd.DataFrame(
                columns=[
                    "r2_cutoff",
                    "dataset_base",
                    "n_perturbations",
                    "median_observed_neff",
                    "median_predicted_neff",
                    "median_predicted_to_observed_ratio",
                    "median_absolute_relative_error",
                    "median_fit_r2",
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
                median_fit_r2=(
                    "fit_r2",
                    "median",
                ),
            )
            .reset_index()
        )

        dataset_medians = dataset_medians.loc[
            dataset_medians[
                "n_perturbations"
            ]
            >= int(
                MIN_PERTURBATIONS_PER_DATASET
            )
        ].copy()

        dataset_medians[
            "r2_cutoff"
        ] = float(
            r2_cutoff
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

        dataset_medians[
            "median_absolute_relative_error"
        ] = (
            np.abs(
                dataset_medians[
                    "median_predicted_neff"
                ]
                - dataset_medians[
                    "median_observed_neff"
                ]
            )
            / dataset_medians[
                "median_observed_neff"
            ]
        )

        dataset_medians = dataset_medians[
            [
                "r2_cutoff",
                DATASET_COLUMN,
                "n_perturbations",
                "median_observed_neff",
                "median_predicted_neff",
                "median_predicted_to_observed_ratio",
                "median_absolute_relative_error",
                "median_fit_r2",
            ]
        ].copy()

        return dataset_medians

    def calculate_cutoff_summary(
        full_dataframe,
        filtered_dataframe,
        dataset_medians,
        r2_cutoff,
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

        if len(
            dataset_medians
        ) >= MIN_DATASETS_FOR_SPEARMAN:
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

            dataset_pearson_log10 = safe_pearson(
                np.log10(
                    dataset_observed
                ),
                np.log10(
                    dataset_predicted
                ),
            )

            (
                dataset_spearman_ci_low,
                dataset_spearman_ci_high,
                n_valid_bootstrap,
            ) = bootstrap_dataset_spearman(
                observed_dataset_medians=dataset_observed,
                predicted_dataset_medians=dataset_predicted,
                n_bootstrap=N_BOOTSTRAP,
                seed=bootstrap_seed,
            )
        else:
            dataset_spearman = np.nan
            dataset_pearson_log10 = np.nan
            dataset_spearman_ci_low = np.nan
            dataset_spearman_ci_high = np.nan
            n_valid_bootstrap = 0

        if len(
            filtered_dataframe
        ) >= 3:
            perturbation_spearman = safe_spearman(
                observed,
                predicted,
            )

            perturbation_pearson_log10 = safe_pearson(
                np.log10(
                    observed
                ),
                np.log10(
                    predicted
                ),
            )

            centered_r2 = safe_centered_r2(
                observed,
                predicted,
            )

            uncentered_r2 = safe_uncentered_r2(
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

            median_predicted_to_observed_ratio = float(
                np.median(
                    predicted
                    / observed
                )
            )
        else:
            perturbation_spearman = np.nan
            perturbation_pearson_log10 = np.nan
            centered_r2 = np.nan
            uncentered_r2 = np.nan
            median_absolute_relative_error = np.nan
            median_predicted_to_observed_ratio = np.nan

        return {
            "r2_cutoff": float(
                r2_cutoff
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
                        full_dataframe
                    ),
                    1,
                )
            ),
            "n_datasets": int(
                len(
                    dataset_medians
                )
            ),
            "minimum_perturbations_per_dataset": int(
                MIN_PERTURBATIONS_PER_DATASET
            ),
            "dataset_median_spearman": dataset_spearman,
            "dataset_median_spearman_ci_low": (
                dataset_spearman_ci_low
            ),
            "dataset_median_spearman_ci_high": (
                dataset_spearman_ci_high
            ),
            "dataset_median_pearson_log10": (
                dataset_pearson_log10
            ),
            "n_valid_bootstrap_samples": int(
                n_valid_bootstrap
            ),
            "perturbation_spearman": (
                perturbation_spearman
            ),
            "perturbation_pearson_log10": (
                perturbation_pearson_log10
            ),
            "perturbation_centered_r2": (
                centered_r2
            ),
            "perturbation_uncentered_r2": (
                uncentered_r2
            ),
            "median_absolute_relative_error": (
                median_absolute_relative_error
            ),
            "median_predicted_to_observed_ratio": (
                median_predicted_to_observed_ratio
            ),
            "median_observed_neff": (
                float(
                    np.median(
                        observed
                    )
                )
                if len(
                    observed
                ) > 0
                else np.nan
            ),
            "median_predicted_neff": (
                float(
                    np.median(
                        predicted
                    )
                )
                if len(
                    predicted
                ) > 0
                else np.nan
            ),
        }

    def add_loglog_scatter(
        axis,
        filtered_dataframe,
        r2_cutoff,
        shared_minimum,
        shared_maximum,
        panel_label=None,
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
                shared_minimum,
                shared_maximum,
            ],
            [
                shared_minimum,
                shared_maximum,
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
            shared_minimum,
            shared_maximum,
        )

        axis.set_ylim(
            shared_minimum,
            shared_maximum,
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

        n_datasets = int(
            filtered_dataframe[
                DATASET_COLUMN
            ].nunique()
        )

        annotation = (
            f"R² cutoff ≥ {r2_cutoff:.2f}\n"
            f"n = {len(filtered_dataframe):,}\n"
            f"Datasets = {n_datasets}\n"
            f"Pearson(log10) = "
            f"{perturbation_pearson_log10:.3f}\n"
            f"Spearman = "
            f"{perturbation_spearman:.3f}\n"
            f"Median |relative error| = "
            f"{median_absolute_relative_error:.3f}"
        )

        axis.text(
            0.04,
            0.96,
            annotation,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
        )

        if panel_label is not None:
            axis.text(
                -0.12,
                1.07,
                panel_label,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=15,
                fontweight="bold",
            )

        axis.set_xlabel(
            "Observed effective number of genes"
        )

        axis.set_ylabel(
            "CIPHER-predicted effective number of genes"
        )

        axis.set_title(
            f"Perturbations with forward R² ≥ {r2_cutoff:.2f}"
        )

        axis.grid(
            alpha=GRID_ALPHA
        )

    ANALYSIS_ROOT = Path(OUTDIR)

    OUT_TABLE_DIR = (
        ANALYSIS_ROOT
        / "tables"
        / "observed_vs_cipher_neff_r2_cutoff_sweep"
    )

    OUT_FIGURE_DIR = (
        ANALYSIS_ROOT
        / "figures"
        / "observed_vs_cipher_neff_r2_cutoff_sweep"
    )

    OUT_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATASET_COLUMN = "dataset_base"

    PERTURBATION_COLUMN = "perturbation"

    TARGET_GENE_COLUMN = "target_gene"

    STATUS_COLUMN = "status"

    OBSERVED_NEFF_COLUMN = (
        "observed_excluding_target_neff_shannon"
    )

    PREDICTED_NEFF_COLUMN = (
        "predicted_excluding_target_neff_shannon"
    )

    R2_COLUMN = (
        "r2_uncentered_excluding_target"
    )

    R2_CUTOFFS = np.round(
        np.arange(
            0.0,
            0.9001,
            0.05,
        ),
        2,
    )

    SHOWCASE_R2_CUTOFFS = [
        0.30,
        0.60,
        0.90,
    ]

    MIN_PERTURBATIONS_PER_DATASET = 3

    MIN_DATASETS_FOR_SPEARMAN = 3

    N_BOOTSTRAP = 2000

    RANDOM_SEED = 314159

    DPI = 300

    SAVE_PDF = True

    SAVE_SVG = True

    SHOW_FIGURES = True

    SCATTER_SIZE = 16

    SCATTER_ALPHA = 0.17

    DATASET_MEDIAN_SIZE = 48

    GRID_ALPHA = 0.18

    IDENTITY_LINE_WIDTH = 1.6

    SWEEP_LINE_WIDTH = 2.0

    SWEEP_MARKER_SIZE = 6

    if not BREADTH_PATH.exists():
        raise FileNotFoundError(
            f"Could not find:\n"
            f"    {BREADTH_PATH}"
        )

    breadth_df = pd.read_csv(
        BREADTH_PATH,
        sep="\t",
    )

    required_columns = [
        DATASET_COLUMN,
        PERTURBATION_COLUMN,
        TARGET_GENE_COLUMN,
        STATUS_COLUMN,
        OBSERVED_NEFF_COLUMN,
        PREDICTED_NEFF_COLUMN,
        R2_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in breadth_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "The response-breadth table is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
            + "\n\nAvailable columns:\n  "
            + "\n  ".join(
                map(
                    str,
                    breadth_df.columns,
                )
            )
        )

    analysis_df = breadth_df.loc[
        breadth_df[
            STATUS_COLUMN
        ].astype(
            str
        )
        == "ok"
    ].copy()

    for column in [
        OBSERVED_NEFF_COLUMN,
        PREDICTED_NEFF_COLUMN,
        R2_COLUMN,
    ]:
        analysis_df[
            column
        ] = pd.to_numeric(
            analysis_df[
                column
            ],
            errors="coerce",
        )

    analysis_df = analysis_df.loc[
        np.isfinite(
            analysis_df[
                OBSERVED_NEFF_COLUMN
            ]
        )
        & np.isfinite(
            analysis_df[
                PREDICTED_NEFF_COLUMN
            ]
        )
        & np.isfinite(
            analysis_df[
                R2_COLUMN
            ]
        )
        & (
            analysis_df[
                OBSERVED_NEFF_COLUMN
            ]
            > 0
        )
        & (
            analysis_df[
                PREDICTED_NEFF_COLUMN
            ]
            > 0
        )
    ].copy()

    analysis_df[
        "observed_neff"
    ] = analysis_df[
        OBSERVED_NEFF_COLUMN
    ].astype(
        float
    )

    analysis_df[
        "predicted_neff"
    ] = analysis_df[
        PREDICTED_NEFF_COLUMN
    ].astype(
        float
    )

    analysis_df[
        "fit_r2"
    ] = analysis_df[
        R2_COLUMN
    ].astype(
        float
    )

    analysis_df[
        "predicted_to_observed_ratio"
    ] = (
        analysis_df[
            "predicted_neff"
        ]
        / analysis_df[
            "observed_neff"
        ]
    )

    analysis_df[
        "absolute_relative_error"
    ] = (
        np.abs(
            analysis_df[
                "predicted_neff"
            ]
            - analysis_df[
                "observed_neff"
            ]
        )
        / analysis_df[
            "observed_neff"
        ]
    )

    analysis_df[
        "log10_observed_neff"
    ] = np.log10(
        analysis_df[
            "observed_neff"
        ]
    )

    analysis_df[
        "log10_predicted_neff"
    ] = np.log10(
        analysis_df[
            "predicted_neff"
        ]
    )

    analysis_df.to_csv(
        OUT_TABLE_DIR
        / "all_valid_observed_vs_cipher_neff.tsv",
        sep="\t",
        index=False,
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "R2-CUTOFF SWEEP: OBSERVED VS CIPHER-PREDICTED N_EFF"
    )

    print(
        "=" * 110
    )

    print(
        f"Input table:                  {BREADTH_PATH}"
    )

    print(
        f"Valid perturbations:          {len(analysis_df):,}"
    )

    print(
        f"Datasets before filtering:    "
        f"{analysis_df[DATASET_COLUMN].nunique():,}"
    )

    print(
        f"R2 cutoff range:              "
        f"{R2_CUTOFFS.min():.2f} to {R2_CUTOFFS.max():.2f}"
    )

    print(
        f"R2 cutoff step:               "
        f"{R2_CUTOFFS[1] - R2_CUTOFFS[0]:.2f}"
    )

    print(
        f"Min perts per dataset:        "
        f"{MIN_PERTURBATIONS_PER_DATASET}"
    )

    print(
        "=" * 110
    )

    sweep_rows = []

    all_dataset_median_frames = []

    filtered_tables = {}

    for cutoff_index, r2_cutoff in enumerate(
        R2_CUTOFFS
    ):
        filtered_df = make_filtered_dataframe(
            full_dataframe=analysis_df,
            r2_cutoff=r2_cutoff,
        )

        dataset_medians_df = (
            calculate_dataset_medians(
                filtered_dataframe=filtered_df,
                r2_cutoff=r2_cutoff,
            )
        )

        summary_row = calculate_cutoff_summary(
            full_dataframe=analysis_df,
            filtered_dataframe=filtered_df,
            dataset_medians=dataset_medians_df,
            r2_cutoff=r2_cutoff,
            bootstrap_seed=(
                RANDOM_SEED
                + cutoff_index
            ),
        )

        sweep_rows.append(
            summary_row
        )

        all_dataset_median_frames.append(
            dataset_medians_df
        )

        filtered_tables[
            float(
                np.round(
                    r2_cutoff,
                    2,
                )
            )
        ] = filtered_df

        print(
            f"R2 >= {r2_cutoff:.2f}: "
            f"{len(filtered_df):,} perturbations; "
            f"{len(dataset_medians_df):,} datasets; "
            f"dataset-level Spearman = "
            f"{summary_row['dataset_median_spearman']:.4f}"
        )

    sweep_df = pd.DataFrame(
        sweep_rows
    )

    all_dataset_medians_df = pd.concat(
        all_dataset_median_frames,
        ignore_index=True,
    )

    sweep_df.to_csv(
        OUT_TABLE_DIR
        / "r2_cutoff_sweep_summary.tsv",
        sep="\t",
        index=False,
    )

    all_dataset_medians_df.to_csv(
        OUT_TABLE_DIR
        / "dataset_medians_at_each_r2_cutoff.tsv",
        sep="\t",
        index=False,
    )

    for r2_cutoff in SHOWCASE_R2_CUTOFFS:
        rounded_cutoff = float(
            np.round(
                r2_cutoff,
                2,
            )
        )

        if rounded_cutoff not in filtered_tables:
            showcase_df = make_filtered_dataframe(
                full_dataframe=analysis_df,
                r2_cutoff=rounded_cutoff,
            )
        else:
            showcase_df = filtered_tables[
                rounded_cutoff
            ]

        showcase_df.to_csv(
            OUT_TABLE_DIR
            / (
                "observed_vs_cipher_neff_"
                f"r2_ge_{number_to_tag(rounded_cutoff)}.tsv"
            ),
            sep="\t",
            index=False,
        )

    showcase_frames = []

    for r2_cutoff in SHOWCASE_R2_CUTOFFS:
        rounded_cutoff = float(
            np.round(
                r2_cutoff,
                2,
            )
        )

        showcase_df = filtered_tables.get(
            rounded_cutoff,
            make_filtered_dataframe(
                full_dataframe=analysis_df,
                r2_cutoff=rounded_cutoff,
            ),
        )

        if len(
            showcase_df
        ) == 0:
            raise RuntimeError(
                f"No perturbations passed R2 >= "
                f"{rounded_cutoff:.2f}."
            )

        showcase_frames.append(
            showcase_df
        )

    all_showcase_observed = np.concatenate(
        [
            dataframe[
                "observed_neff"
            ].to_numpy(
                dtype=float
            )
            for dataframe in showcase_frames
        ]
    )

    all_showcase_predicted = np.concatenate(
        [
            dataframe[
                "predicted_neff"
            ].to_numpy(
                dtype=float
            )
            for dataframe in showcase_frames
        ]
    )

    shared_positive_minimum = float(
        min(
            np.min(
                all_showcase_observed
            ),
            np.min(
                all_showcase_predicted
            ),
        )
    )

    shared_positive_maximum = float(
        max(
            np.max(
                all_showcase_observed
            ),
            np.max(
                all_showcase_predicted
            ),
        )
    )

    shared_log_minimum = float(
        10 ** (
            np.floor(
                np.log10(
                    shared_positive_minimum
                )
            )
            - 0.05
        )
    )

    shared_log_maximum = float(
        10 ** (
            np.ceil(
                np.log10(
                    shared_positive_maximum
                )
            )
            + 0.05
        )
    )

    figure = plt.figure(
        figsize=(
            16,
            13,
        )
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[
            0.92,
            1.15,
        ],
        hspace=0.34,
        wspace=0.28,
    )

    sweep_axis = figure.add_subplot(
        grid[
            0,
            :,
        ]
    )

    scatter_axes = [
        figure.add_subplot(
            grid[
                1,
                panel_index,
            ]
        )
        for panel_index in range(
            3
        )
    ]

    sweep_x = sweep_df[
        "r2_cutoff"
    ].to_numpy(
        dtype=float
    )

    sweep_y = sweep_df[
        "dataset_median_spearman"
    ].to_numpy(
        dtype=float
    )

    ci_low = sweep_df[
        "dataset_median_spearman_ci_low"
    ].to_numpy(
        dtype=float
    )

    ci_high = sweep_df[
        "dataset_median_spearman_ci_high"
    ].to_numpy(
        dtype=float
    )

    sweep_axis.plot(
        sweep_x,
        sweep_y,
        marker="o",
        markersize=SWEEP_MARKER_SIZE,
        linewidth=SWEEP_LINE_WIDTH,
    )

    finite_ci = (
        np.isfinite(
            ci_low
        )
        & np.isfinite(
            ci_high
        )
    )

    if np.any(
        finite_ci
    ):
        sweep_axis.fill_between(
            sweep_x[
                finite_ci
            ],
            ci_low[
                finite_ci
            ],
            ci_high[
                finite_ci
            ],
            alpha=0.18,
            label="Dataset-bootstrap 95% CI",
        )

    for showcase_cutoff in SHOWCASE_R2_CUTOFFS:
        showcase_match = np.flatnonzero(
            np.isclose(
                sweep_x,
                showcase_cutoff,
            )
        )

        if len(
            showcase_match
        ) == 1:
            match_index = int(
                showcase_match[
                    0
                ]
            )

            sweep_axis.scatter(
                [
                    sweep_x[
                        match_index
                    ]
                ],
                [
                    sweep_y[
                        match_index
                    ]
                ],
                s=85,
                zorder=5,
            )

            sweep_axis.annotate(
                (
                    f"R² ≥ {showcase_cutoff:.2f}\n"
                    f"ρ = {sweep_y[match_index]:.3f}\n"
                    f"n = "
                    f"{int(sweep_df.iloc[match_index]['n_perturbations']):,}, "
                    f"D = "
                    f"{int(sweep_df.iloc[match_index]['n_datasets'])}"
                ),
                xy=(
                    sweep_x[
                        match_index
                    ],
                    sweep_y[
                        match_index
                    ],
                ),
                xytext=(
                    8,
                    8,
                ),
                textcoords="offset points",
                fontsize=8.5,
            )

    sweep_axis.set_xticks(
        R2_CUTOFFS
    )

    sweep_axis.set_xticklabels(
        [
            f"{value:.2f}"
            for value in R2_CUTOFFS
        ],
        rotation=45,
        ha="right",
    )

    finite_sweep_y = sweep_y[
        np.isfinite(
            sweep_y
        )
    ]

    if len(
        finite_sweep_y
    ) > 0:
        y_padding = max(
            0.03,
            0.12
            * (
                np.max(
                    finite_sweep_y
                )
                - np.min(
                    finite_sweep_y
                )
            ),
        )

        sweep_axis.set_ylim(
            max(
                -1.0,
                np.min(
                    finite_sweep_y
                )
                - y_padding,
            ),
            min(
                1.0,
                np.max(
                    finite_sweep_y
                )
                + y_padding,
            ),
        )

    sweep_axis.set_xlabel(
        "Minimum perturbation-level forward R²"
    )

    sweep_axis.set_ylabel(
        "Spearman correlation across dataset medians"
    )

    sweep_axis.set_title(
        "Dataset-level agreement between observed and "
        "CIPHER-predicted response breadth"
    )

    sweep_axis.text(
        -0.055,
        1.08,
        "A",
        transform=sweep_axis.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    sweep_axis.grid(
        alpha=GRID_ALPHA
    )

    if np.any(
        finite_ci
    ):
        sweep_axis.legend(
            frameon=False,
            loc="best",
        )

    panel_labels = [
        "B",
        "C",
        "D",
    ]

    for axis, filtered_df, r2_cutoff, panel_label in zip(
        scatter_axes,
        showcase_frames,
        SHOWCASE_R2_CUTOFFS,
        panel_labels,
    ):
        add_loglog_scatter(
            axis=axis,
            filtered_dataframe=filtered_df,
            r2_cutoff=r2_cutoff,
            shared_minimum=shared_log_minimum,
            shared_maximum=shared_log_maximum,
            panel_label=panel_label,
        )

    scatter_axes[
        -1
    ].legend(
        frameon=False,
        loc="lower right",
    )

    figure.suptitle(
        (
            "Observed versus CIPHER-predicted effective number of genes\n"
            "Sensitivity to perturbation-level forward-prediction R²"
        ),
        fontsize=18,
        y=0.985,
    )

    figure.subplots_adjust(
        top=0.91,
        bottom=0.07,
        left=0.07,
        right=0.98,
    )

    save_figure(
        figure,
        OUT_FIGURE_DIR
        / "r2_cutoff_sweep_and_three_loglog_scatter_panels",
    )

    figure, axis = plt.subplots(
        figsize=(
            9.2,
            6.3,
        )
    )

    axis.plot(
        sweep_x,
        sweep_y,
        marker="o",
        markersize=SWEEP_MARKER_SIZE,
        linewidth=SWEEP_LINE_WIDTH,
    )

    if np.any(
        finite_ci
    ):
        axis.fill_between(
            sweep_x[
                finite_ci
            ],
            ci_low[
                finite_ci
            ],
            ci_high[
                finite_ci
            ],
            alpha=0.18,
            label="Dataset-bootstrap 95% CI",
        )

    for showcase_cutoff in SHOWCASE_R2_CUTOFFS:
        showcase_match = np.flatnonzero(
            np.isclose(
                sweep_x,
                showcase_cutoff,
            )
        )

        if len(
            showcase_match
        ) == 1:
            match_index = int(
                showcase_match[
                    0
                ]
            )

            axis.scatter(
                [
                    sweep_x[
                        match_index
                    ]
                ],
                [
                    sweep_y[
                        match_index
                    ]
                ],
                s=90,
                zorder=5,
            )

            axis.annotate(
                (
                    f"ρ = "
                    f"{sweep_y[match_index]:.3f}"
                ),
                xy=(
                    sweep_x[
                        match_index
                    ],
                    sweep_y[
                        match_index
                    ],
                ),
                xytext=(
                    7,
                    7,
                ),
                textcoords="offset points",
                fontsize=9,
            )

    axis.set_xticks(
        R2_CUTOFFS
    )

    axis.set_xticklabels(
        [
            f"{value:.2f}"
            for value in R2_CUTOFFS
        ],
        rotation=45,
        ha="right",
    )

    if len(
        finite_sweep_y
    ) > 0:
        axis.set_ylim(
            max(
                -1.0,
                np.min(
                    finite_sweep_y
                )
                - y_padding,
            ),
            min(
                1.0,
                np.max(
                    finite_sweep_y
                )
                + y_padding,
            ),
        )

    axis.set_xlabel(
        "Minimum perturbation-level forward R²"
    )

    axis.set_ylabel(
        "Dataset-level Spearman correlation\n"
        "between median observed and predicted N_eff"
    )

    axis.set_title(
        "Dataset-level response-breadth agreement across R² cutoffs"
    )

    axis.grid(
        alpha=GRID_ALPHA
    )

    if np.any(
        finite_ci
    ):
        axis.legend(
            frameon=False
        )

    figure.tight_layout()

    save_figure(
        figure,
        OUT_FIGURE_DIR
        / "dataset_median_spearman_vs_r2_cutoff",
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(
            18,
            5.8,
        ),
        sharex=True,
        sharey=True,
    )

    for axis, filtered_df, r2_cutoff, panel_label in zip(
        axes,
        showcase_frames,
        SHOWCASE_R2_CUTOFFS,
        [
            "A",
            "B",
            "C",
        ],
    ):
        add_loglog_scatter(
            axis=axis,
            filtered_dataframe=filtered_df,
            r2_cutoff=r2_cutoff,
            shared_minimum=shared_log_minimum,
            shared_maximum=shared_log_maximum,
            panel_label=panel_label,
        )

    axes[
        -1
    ].legend(
        frameon=False,
        loc="lower right",
    )

    figure.suptitle(
        (
            "Observed versus CIPHER-predicted response breadth "
            "at three forward-R² cutoffs"
        ),
        fontsize=17,
        y=1.02,
    )

    figure.tight_layout()

    save_figure(
        figure,
        OUT_FIGURE_DIR
        / "observed_vs_cipher_neff_three_r2_cutoffs",
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(
            18,
            5.8,
        ),
        sharex=True,
        sharey=True,
    )

    showcase_dataset_median_frames = []

    for r2_cutoff in SHOWCASE_R2_CUTOFFS:
        cutoff_median_df = all_dataset_medians_df.loc[
            np.isclose(
                all_dataset_medians_df[
                    "r2_cutoff"
                ],
                r2_cutoff,
            )
        ].copy()

        showcase_dataset_median_frames.append(
            cutoff_median_df
        )

    all_dataset_median_observed = np.concatenate(
        [
            dataframe[
                "median_observed_neff"
            ].to_numpy(
                dtype=float
            )
            for dataframe in showcase_dataset_median_frames
            if len(
                dataframe
            ) > 0
        ]
    )

    all_dataset_median_predicted = np.concatenate(
        [
            dataframe[
                "median_predicted_neff"
            ].to_numpy(
                dtype=float
            )
            for dataframe in showcase_dataset_median_frames
            if len(
                dataframe
            ) > 0
        ]
    )

    dataset_shared_minimum = float(
        min(
            np.min(
                all_dataset_median_observed
            ),
            np.min(
                all_dataset_median_predicted
            ),
        )
    )

    dataset_shared_maximum = float(
        max(
            np.max(
                all_dataset_median_observed
            ),
            np.max(
                all_dataset_median_predicted
            ),
        )
    )

    dataset_shared_log_minimum = float(
        10 ** (
            np.floor(
                np.log10(
                    dataset_shared_minimum
                )
            )
            - 0.05
        )
    )

    dataset_shared_log_maximum = float(
        10 ** (
            np.ceil(
                np.log10(
                    dataset_shared_maximum
                )
            )
            + 0.05
        )
    )

    for axis, dataset_median_df, r2_cutoff, panel_label in zip(
        axes,
        showcase_dataset_median_frames,
        SHOWCASE_R2_CUTOFFS,
        [
            "A",
            "B",
            "C",
        ],
    ):
        observed_dataset_medians = dataset_median_df[
            "median_observed_neff"
        ].to_numpy(
            dtype=float
        )

        predicted_dataset_medians = dataset_median_df[
            "median_predicted_neff"
        ].to_numpy(
            dtype=float
        )

        dataset_spearman = safe_spearman(
            observed_dataset_medians,
            predicted_dataset_medians,
        )

        axis.scatter(
            observed_dataset_medians,
            predicted_dataset_medians,
            s=DATASET_MEDIAN_SIZE,
            alpha=0.82,
            linewidths=0,
        )

        axis.plot(
            [
                dataset_shared_log_minimum,
                dataset_shared_log_maximum,
            ],
            [
                dataset_shared_log_minimum,
                dataset_shared_log_maximum,
            ],
            linestyle="--",
            linewidth=IDENTITY_LINE_WIDTH,
        )

        axis.set_xscale(
            "log"
        )

        axis.set_yscale(
            "log"
        )

        axis.set_xlim(
            dataset_shared_log_minimum,
            dataset_shared_log_maximum,
        )

        axis.set_ylim(
            dataset_shared_log_minimum,
            dataset_shared_log_maximum,
        )

        axis.text(
            0.04,
            0.96,
            (
                f"R² cutoff ≥ {r2_cutoff:.2f}\n"
                f"Datasets = {len(dataset_median_df)}\n"
                f"Spearman = {dataset_spearman:.3f}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

        axis.text(
            -0.12,
            1.07,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
        )

        axis.set_xlabel(
            "Median observed effective number of genes"
        )

        axis.set_ylabel(
            "Median CIPHER-predicted effective number of genes"
        )

        axis.set_title(
            f"Dataset medians: R² ≥ {r2_cutoff:.2f}"
        )

        axis.grid(
            alpha=GRID_ALPHA
        )

    figure.suptitle(
        "Dataset-level median response breadth at three R² cutoffs",
        fontsize=17,
        y=1.02,
    )

    figure.tight_layout()

    save_figure(
        figure,
        OUT_FIGURE_DIR
        / "dataset_median_neff_three_r2_cutoffs",
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "R2-CUTOFF SWEEP COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        f"Observed N_eff column:\n"
        f"    {OBSERVED_NEFF_COLUMN}"
    )

    print(
        f"Predicted N_eff column:\n"
        f"    {PREDICTED_NEFF_COLUMN}"
    )

    print(
        f"Forward-fit R2 column:\n"
        f"    {R2_COLUMN}"
    )

    print(
        f"\nValid perturbations before cutoff filtering: "
        f"{len(analysis_df):,}"
    )

    print(
        f"Datasets before cutoff filtering: "
        f"{analysis_df[DATASET_COLUMN].nunique():,}"
    )

    print(
        "\nSweep results:"
    )

    print(
        sweep_df[
            [
                "r2_cutoff",
                "n_perturbations",
                "fraction_perturbations_retained",
                "n_datasets",
                "dataset_median_spearman",
                "dataset_median_spearman_ci_low",
                "dataset_median_spearman_ci_high",
                "perturbation_pearson_log10",
                "perturbation_spearman",
                "median_absolute_relative_error",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\nTables saved to:"
    )

    print(
        f"    {OUT_TABLE_DIR}"
    )

    print(
        "\nFigures saved to:"
    )

    print(
        f"    {OUT_FIGURE_DIR}"
    )

    print(
        "=" * 120
    )


def block3_r2_sweep_published():
    """Block 3 -- R2 sweep 0.30 to 0.90 (published composite) + dataset-median plot."""
    # reset shared helpers to the module canonical (guards against cross-cell shadowing)

    # from src.suppl_effn import *  (provided by module-level import in header)

    ANALYSIS_ROOT = Path(OUTDIR)

    OUT_TABLE_DIR = (
        ANALYSIS_ROOT
        / "tables"
        / "observed_vs_cipher_neff_r2_0p30_to_0p90"
    )

    OUT_FIGURE_DIR = (
        ANALYSIS_ROOT
        / "figures"
        / "observed_vs_cipher_neff_r2_0p30_to_0p90"
    )

    OUT_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATASET_COLUMN = "dataset_base"

    PERTURBATION_COLUMN = "perturbation"

    TARGET_GENE_COLUMN = "target_gene"

    STATUS_COLUMN = "status"

    OBSERVED_NEFF_COLUMN = (
        "observed_excluding_target_neff_shannon"
    )

    PREDICTED_NEFF_COLUMN = (
        "predicted_excluding_target_neff_shannon"
    )

    R2_COLUMN = (
        "r2_uncentered_excluding_target"
    )

    R2_CUTOFFS = np.round(
        np.arange(
            0.30,
            0.9001,
            0.05,
        ),
        2,
    )

    SHOWCASE_R2_CUTOFFS = [
        0.30,
        0.60,
        0.90,
    ]

    DATASET_MEDIAN_PLOT_R2_CUTOFF = 0.50

    MIN_PERTURBATIONS_PER_DATASET = 3

    MIN_DATASETS_FOR_SPEARMAN = 3

    N_BOOTSTRAP = 2000

    RANDOM_SEED = 314159

    DPI = 300

    SAVE_PDF = True

    SAVE_SVG = True

    SHOW_FIGURES = True

    SCATTER_SIZE = 16

    SCATTER_ALPHA = 0.17

    DATASET_MEDIAN_POINT_SIZE = 58

    GRID_ALPHA = 0.18

    IDENTITY_LINE_WIDTH = 1.7

    SWEEP_LINE_WIDTH = 2.0

    SWEEP_MARKER_SIZE = 6

    ANNOTATE_DATASET_NAMES = True

    DATASET_LABEL_FONT_SIZE = 6.5

    if not BREADTH_PATH.exists():
        raise FileNotFoundError(
            f"Could not find:\n"
            f"    {BREADTH_PATH}"
        )

    breadth_df = pd.read_csv(
        BREADTH_PATH,
        sep="\t",
    )

    required_columns = [
        DATASET_COLUMN,
        PERTURBATION_COLUMN,
        TARGET_GENE_COLUMN,
        STATUS_COLUMN,
        OBSERVED_NEFF_COLUMN,
        PREDICTED_NEFF_COLUMN,
        R2_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in breadth_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
            + "\n\nAvailable columns:\n  "
            + "\n  ".join(
                map(
                    str,
                    breadth_df.columns,
                )
            )
        )

    analysis_df = breadth_df.loc[
        breadth_df[
            STATUS_COLUMN
        ].astype(
            str
        )
        == "ok"
    ].copy()

    for column in [
        OBSERVED_NEFF_COLUMN,
        PREDICTED_NEFF_COLUMN,
        R2_COLUMN,
    ]:
        analysis_df[
            column
        ] = pd.to_numeric(
            analysis_df[
                column
            ],
            errors="coerce",
        )

    analysis_df = analysis_df.loc[
        np.isfinite(
            analysis_df[
                OBSERVED_NEFF_COLUMN
            ]
        )
        & np.isfinite(
            analysis_df[
                PREDICTED_NEFF_COLUMN
            ]
        )
        & np.isfinite(
            analysis_df[
                R2_COLUMN
            ]
        )
        & (
            analysis_df[
                OBSERVED_NEFF_COLUMN
            ]
            > 0
        )
        & (
            analysis_df[
                PREDICTED_NEFF_COLUMN
            ]
            > 0
        )
    ].copy()

    analysis_df[
        "observed_neff"
    ] = analysis_df[
        OBSERVED_NEFF_COLUMN
    ].astype(
        float
    )

    analysis_df[
        "predicted_neff"
    ] = analysis_df[
        PREDICTED_NEFF_COLUMN
    ].astype(
        float
    )

    analysis_df[
        "fit_r2"
    ] = analysis_df[
        R2_COLUMN
    ].astype(
        float
    )

    sweep_rows = []

    dataset_median_frames = []

    filtered_by_cutoff = {}

    dataset_medians_by_cutoff = {}

    for cutoff_index, cutoff in enumerate(
        R2_CUTOFFS
    ):
        filtered_df = filter_by_r2(
            dataframe=analysis_df,
            cutoff=cutoff,
        )

        dataset_medians_df = (
            calculate_dataset_medians(
                filtered_dataframe=filtered_df,
                cutoff=cutoff,
            )
        )

        sweep_row = calculate_sweep_row(
            complete_dataframe=analysis_df,
            filtered_dataframe=filtered_df,
            dataset_medians=dataset_medians_df,
            cutoff=cutoff,
            bootstrap_seed=(
                RANDOM_SEED
                + cutoff_index
            ),
        )

        cutoff_key = float(
            np.round(
                cutoff,
                2,
            )
        )

        filtered_by_cutoff[
            cutoff_key
        ] = filtered_df

        dataset_medians_by_cutoff[
            cutoff_key
        ] = dataset_medians_df

        sweep_rows.append(
            sweep_row
        )

        dataset_median_frames.append(
            dataset_medians_df
        )

        print(
            f"R² >= {cutoff:.2f}: "
            f"n = {len(filtered_df):,}, "
            f"datasets = {len(dataset_medians_df)}, "
            f"dataset Spearman = "
            f"{sweep_row['dataset_median_spearman']:.4f}"
        )

    sweep_df = pd.DataFrame(
        sweep_rows
    )

    all_dataset_medians_df = pd.concat(
        dataset_median_frames,
        ignore_index=True,
    )

    sweep_df.to_csv(
        OUT_TABLE_DIR
        / "r2_cutoff_sweep_summary.tsv",
        sep="\t",
        index=False,
    )

    all_dataset_medians_df.to_csv(
        OUT_TABLE_DIR
        / "dataset_medians_at_each_r2_cutoff.tsv",
        sep="\t",
        index=False,
    )

    showcase_frames = []

    for cutoff in SHOWCASE_R2_CUTOFFS:
        cutoff_key = float(
            np.round(
                cutoff,
                2,
            )
        )

        if cutoff_key not in filtered_by_cutoff:
            raise KeyError(
                f"Showcase cutoff {cutoff_key:.2f} "
                "was not found in the sweep."
            )

        showcase_df = filtered_by_cutoff[
            cutoff_key
        ].copy()

        if len(
            showcase_df
        ) == 0:
            raise RuntimeError(
                f"No perturbations passed R² >= "
                f"{cutoff_key:.2f}."
            )

        showcase_frames.append(
            showcase_df
        )

        showcase_df.to_csv(
            OUT_TABLE_DIR
            / (
                "observed_vs_predicted_neff_"
                f"r2_ge_{number_to_tag(cutoff_key)}.tsv"
            ),
            sep="\t",
            index=False,
        )

    (
        perturbation_minimum_limit,
        perturbation_maximum_limit,
    ) = calculate_shared_log_limits(
        dataframes=showcase_frames,
        observed_column="observed_neff",
        predicted_column="predicted_neff",
    )

    figure = plt.figure(
        figsize=(
            16,
            12.5,
        )
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[
            0.88,
            1.12,
        ],
        hspace=0.36,
        wspace=0.28,
    )

    sweep_axis = figure.add_subplot(
        grid[
            0,
            :,
        ]
    )

    scatter_axes = [
        figure.add_subplot(
            grid[
                1,
                index,
            ]
        )
        for index in range(
            3
        )
    ]

    sweep_x = sweep_df[
        "r2_cutoff"
    ].to_numpy(
        dtype=float
    )

    sweep_y = sweep_df[
        "dataset_median_spearman"
    ].to_numpy(
        dtype=float
    )

    ci_low = sweep_df[
        "dataset_median_spearman_ci_low"
    ].to_numpy(
        dtype=float
    )

    ci_high = sweep_df[
        "dataset_median_spearman_ci_high"
    ].to_numpy(
        dtype=float
    )

    sweep_axis.plot(
        sweep_x,
        sweep_y,
        marker="o",
        markersize=SWEEP_MARKER_SIZE,
        linewidth=SWEEP_LINE_WIDTH,
    )

    finite_ci = (
        np.isfinite(
            ci_low
        )
        & np.isfinite(
            ci_high
        )
    )

    if np.any(
        finite_ci
    ):
        sweep_axis.fill_between(
            sweep_x[
                finite_ci
            ],
            np.clip(
                ci_low[
                    finite_ci
                ],
                0.0,
                1.0,
            ),
            np.clip(
                ci_high[
                    finite_ci
                ],
                0.0,
                1.0,
            ),
            alpha=0.18,
            label="Dataset-bootstrap 95% CI",
        )

    for showcase_cutoff in SHOWCASE_R2_CUTOFFS:
        matches = np.flatnonzero(
            np.isclose(
                sweep_x,
                showcase_cutoff,
            )
        )

        if len(
            matches
        ) != 1:
            continue

        match_index = int(
            matches[
                0
            ]
        )

        sweep_axis.scatter(
            sweep_x[
                match_index
            ],
            sweep_y[
                match_index
            ],
            s=85,
            zorder=5,
        )

    sweep_axis.set_xlim(
        0.285,
        0.915,
    )

    sweep_axis.set_ylim(
        0.0,
        1.0,
    )

    sweep_axis.set_xticks(
        R2_CUTOFFS
    )

    sweep_axis.set_xticklabels(
        [
            f"{cutoff:.2f}"
            for cutoff in R2_CUTOFFS
        ],
        rotation=45,
        ha="right",
    )

    sweep_axis.set_yticks(
        np.arange(
            0.0,
            1.01,
            0.1,
        )
    )

    sweep_axis.set_xlabel(
        "Minimum perturbation-level forward R²"
    )

    sweep_axis.set_ylabel(
        "Spearman correlation across dataset medians"
    )

    sweep_axis.set_title(
        "Dataset-level agreement between observed and "
        "CIPHER-predicted response breadth"
    )

    sweep_axis.text(
        -0.055,
        1.08,
        "A",
        transform=sweep_axis.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    sweep_axis.grid(
        alpha=GRID_ALPHA
    )

    if np.any(
        finite_ci
    ):
        sweep_axis.legend(
            frameon=False,
            loc="lower right",
        )

    for (
        axis,
        filtered_df,
        cutoff,
        panel_label,
    ) in zip(
        scatter_axes,
        showcase_frames,
        SHOWCASE_R2_CUTOFFS,
        [
            "B",
            "C",
            "D",
        ],
    ):
        add_perturbation_loglog_panel(
            axis=axis,
            filtered_dataframe=filtered_df,
            cutoff=cutoff,
            minimum_limit=(
                perturbation_minimum_limit
            ),
            maximum_limit=(
                perturbation_maximum_limit
            ),
            panel_label=panel_label,
        )

    scatter_axes[
        -1
    ].legend(
        frameon=False,
        loc="lower right",
    )

    figure.suptitle(
        (
            "Observed versus CIPHER-predicted effective number of genes\n"
            "Sensitivity to perturbation-level forward-prediction R²"
        ),
        fontsize=18,
        y=0.985,
    )

    figure.subplots_adjust(
        top=0.91,
        bottom=0.07,
        left=0.07,
        right=0.98,
    )

    save_figure(
        figure,
        OUT_FIGURE_DIR
        / "r2_cutoff_sweep_and_three_neff_scatter_panels",
    )

    median_cutoff_key = float(
        np.round(
            DATASET_MEDIAN_PLOT_R2_CUTOFF,
            2,
        )
    )

    if median_cutoff_key not in dataset_medians_by_cutoff:
        raise KeyError(
            "DATASET_MEDIAN_PLOT_R2_CUTOFF must be one of:\n"
            + "\n".join(
                f"  - {cutoff:.2f}"
                for cutoff in R2_CUTOFFS
            )
        )

    median_plot_df = dataset_medians_by_cutoff[
        median_cutoff_key
    ].copy()

    if len(
        median_plot_df
    ) < MIN_DATASETS_FOR_SPEARMAN:
        raise RuntimeError(
            f"Only {len(median_plot_df)} datasets remain at "
            f"R² >= {median_cutoff_key:.2f}."
        )

    median_plot_df.to_csv(
        OUT_TABLE_DIR
        / (
            "dataset_medians_for_plot_"
            f"r2_ge_{number_to_tag(median_cutoff_key)}.tsv"
        ),
        sep="\t",
        index=False,
    )

    median_observed = median_plot_df[
        "median_observed_neff"
    ].to_numpy(
        dtype=float
    )

    median_predicted = median_plot_df[
        "median_predicted_neff"
    ].to_numpy(
        dtype=float
    )

    median_dataset_spearman = safe_spearman(
        median_observed,
        median_predicted,
    )

    median_dataset_pearson_log10 = safe_pearson(
        np.log10(
            median_observed
        ),
        np.log10(
            median_predicted
        ),
    )

    (
        dataset_minimum_limit,
        dataset_maximum_limit,
    ) = calculate_shared_log_limits(
        dataframes=[
            median_plot_df
        ],
        observed_column="median_observed_neff",
        predicted_column="median_predicted_neff",
    )

    figure, axis = plt.subplots(
        figsize=(
            9.2,
            8.2,
        )
    )

    axis.scatter(
        median_observed,
        median_predicted,
        s=DATASET_MEDIAN_POINT_SIZE,
        alpha=0.82,
        linewidths=0,
    )

    axis.plot(
        [
            dataset_minimum_limit,
            dataset_maximum_limit,
        ],
        [
            dataset_minimum_limit,
            dataset_maximum_limit,
        ],
        linestyle="--",
        linewidth=IDENTITY_LINE_WIDTH,
        label="Observed = predicted",
    )

    if ANNOTATE_DATASET_NAMES:
        for row in median_plot_df.itertuples(
            index=False
        ):
            axis.annotate(
                str(
                    getattr(
                        row,
                        DATASET_COLUMN,
                    )
                ),
                (
                    row.median_observed_neff,
                    row.median_predicted_neff,
                ),
                xytext=(
                    4,
                    3,
                ),
                textcoords="offset points",
                fontsize=DATASET_LABEL_FONT_SIZE,
                alpha=0.80,
            )

    axis.set_xscale(
        "log"
    )

    axis.set_yscale(
        "log"
    )

    axis.set_xlim(
        dataset_minimum_limit,
        dataset_maximum_limit,
    )

    axis.set_ylim(
        dataset_minimum_limit,
        dataset_maximum_limit,
    )

    axis.text(
        0.04,
        0.96,
        (
            f"R² cutoff ≥ {median_cutoff_key:.2f}\n"
            f"Datasets = {len(median_plot_df)}\n"
            f"Spearman = {median_dataset_spearman:.3f}\n"
            f"Pearson(log10) = "
            f"{median_dataset_pearson_log10:.3f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )

    axis.set_xlabel(
        "Median observed effective number of genes"
    )

    axis.set_ylabel(
        "Median CIPHER-predicted effective number of genes"
    )

    axis.set_title(
        "Dataset-level median response breadth"
    )

    axis.grid(
        alpha=GRID_ALPHA
    )

    axis.legend(
        frameon=False,
        loc="lower right",
    )

    figure.tight_layout()

    save_figure(
        figure,
        OUT_FIGURE_DIR
        / (
            "dataset_level_median_response_breadth_"
            f"r2_ge_{number_to_tag(median_cutoff_key)}"
        ),
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "OBSERVED VS CIPHER-PREDICTED N_EFF ANALYSIS COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        f"Input:\n"
        f"    {BREADTH_PATH}"
    )

    print(
        f"\nR² sweep:\n"
        f"    {R2_CUTOFFS.min():.2f} to "
        f"{R2_CUTOFFS.max():.2f} in steps of "
        f"{R2_CUTOFFS[1] - R2_CUTOFFS[0]:.2f}"
    )

    print(
        "\nPanel A:"
    )

    print(
        "    No point annotations"
    )

    print(
        "    Y-axis fixed from 0 to 1"
    )

    print(
        f"\nFigure 2 dataset-median cutoff:\n"
        f"    R² >= {median_cutoff_key:.2f}"
    )

    print(
        f"\nFigure 2 datasets:\n"
        f"    {len(median_plot_df)}"
    )

    print(
        f"\nFigure 2 dataset Spearman:\n"
        f"    {median_dataset_spearman:.4f}"
    )

    print(
        "\nSweep summary:"
    )

    print(
        sweep_df[
            [
                "r2_cutoff",
                "n_perturbations",
                "n_datasets",
                "dataset_median_spearman",
                "dataset_median_spearman_ci_low",
                "dataset_median_spearman_ci_high",
                "perturbation_pearson_log10",
                "perturbation_spearman",
                "median_absolute_relative_error",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        f"\nTables saved to:\n"
        f"    {OUT_TABLE_DIR}"
    )

    print(
        f"\nFigures saved to:\n"
        f"    {OUT_FIGURE_DIR}"
    )

    print(
        "=" * 120
    )


