"""figS16 forward-prediction main flow (relocated verbatim from the notebook; notebook-only)."""
from __future__ import annotations
import os, re, glob, json, math, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from scipy.sparse import issparse, csr_matrix
from scipy.stats import wilcoxon, ttest_rel, ks_2samp, mannwhitneyu, pearsonr, spearmanr
from src.suppl_forward6 import *

def run_figS16():
    global PRECOMPUTE_ROOT, all_dataset_dirs, all_rows, composite_output_base, composite_plot_summary, dataset_dir, dataset_dirs, dataset_iterator, dataset_metrics, dataset_plot_summary, display_columns, marson_datasets_discovered, marson_datasets_excluded, marson_datasets_retained, marson_token_lower, metrics_df, metrics_path, mode, n_ok_rows, ok_metrics, output_base, plot_manifest_df, plot_manifest_path, plot_manifest_rows, plot_summary_df, plot_summary_frames, plot_summary_path, run_status, run_summary, run_summary_path, selected_dataset_names, shared_mode_order, status_df, status_path, summary_df, summary_path
    # ============================================================
    # RUN ALL
    # ============================================================

    ensure_dir(OUTDIR)
    ensure_dir(PLOT_DIR)
    ensure_dir(PER_DATASET_PLOT_DIR)
    ensure_dir(COMPOSITE_PLOT_DIR)

    PRECOMPUTE_ROOT = choose_precompute_root(
        PRECOMPUTE_ROOT_CANDIDATES
    )

    all_dataset_dirs = discover_dataset_dirs(
        PRECOMPUTE_ROOT
    )

    dataset_dirs = select_dataset_dirs(
        all_dataset_dirs
    )

    selected_dataset_names = [
        dataset_dir.name
        for dataset_dir in dataset_dirs
    ]

    print("\n" + "=" * 120)
    print(
        "RECOMPUTING FORWARD METRICS "
        "FROM PRECOMPUTED DATA"
    )
    print("=" * 120)

    print(f"PRECOMPUTE_ROOT:       {PRECOMPUTE_ROOT}")
    print(f"OUTDIR:                {OUTDIR}")
    print(f"datasets discovered:   {len(all_dataset_dirs)}")
    print(f"datasets retained:     {len(dataset_dirs)}")
    print(f"HOLDOUT_FRAC:          {HOLDOUT_FRAC}")
    print(f"train=test:            {HOLDOUT_FRAC <= 0}")

    print("=" * 120)

    all_rows = []
    run_status = []

    dataset_iterator = tqdm(
        dataset_dirs,
        desc="datasets",
        ncols=TQDM_NCOLS,
    )

    for dataset_dir in dataset_iterator:
        for mode in MODES:
            (
                rows,
                status,
            ) = compute_forward_for_dataset_mode(
                dataset_dir,
                mode,
            )

            all_rows.extend(rows)
            run_status.append(status)

            gc.collect()


    # ============================================================
    # SAVE RECOMPUTED METRICS
    # ============================================================

    metrics_df = pd.DataFrame(
        all_rows
    )

    status_df = pd.DataFrame(
        run_status
    )

    summary_df = summarize_metrics(
        metrics_df
    )

    metrics_path = (
        OUTDIR
        / "forward_metrics_per_perturbation_FIRST_ORDER_TO_RAW.tsv"
    )

    status_path = (
        OUTDIR
        / "forward_run_status_FIRST_ORDER_TO_RAW.tsv"
    )

    summary_path = (
        OUTDIR
        / "forward_metrics_per_dataset_mode_FIRST_ORDER_TO_RAW.tsv"
    )

    metrics_df.to_csv(
        metrics_path,
        sep="\t",
        index=False,
    )

    status_df.to_csv(
        status_path,
        sep="\t",
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        sep="\t",
        index=False,
    )


    # ============================================================
    # PREPARE SUCCESSFUL METRIC ROWS
    # ============================================================

    if (
        len(metrics_df) > 0
        and "status" in metrics_df.columns
    ):
        ok_metrics = metrics_df[
            metrics_df["status"].astype(str) == "ok"
        ].copy()

    else:
        ok_metrics = pd.DataFrame()


    # One common mode order for every dataset-specific plot and
    # the pooled composite plot.
    shared_mode_order = determine_shared_mode_order(
        ok_metrics,
        metric_column=ORDER_MODES_BY_METRIC,
    )

    print("\n" + "=" * 120)
    print("PLOTTING ORDER")
    print("=" * 120)

    for index, mode in enumerate(
        shared_mode_order,
        start=1,
    ):
        print(
            f"{index:2d}. "
            f"{MODE_LABELS.get(mode, mode)}"
        )

    print("=" * 120)


    # ============================================================
    # MAKE ONE FIGURE FOR EACH DATASET
    # ============================================================

    plot_summary_frames = []
    plot_manifest_rows = []

    print("\n" + "=" * 120)
    print("MAKING DATASET-SPECIFIC FIGURES")
    print("=" * 120)

    for dataset_index, dataset_name in enumerate(
        selected_dataset_names,
        start=1,
    ):
        dataset_metrics = ok_metrics[
            ok_metrics["dataset"].astype(str)
            == str(dataset_name)
        ].copy()

        output_base = (
            PER_DATASET_PLOT_DIR
            / (
                f"{dataset_index:02d}_"
                f"{safe_filename(dataset_name)}_"
                f"pearson_and_r2"
            )
        )

        print(
            f"[{dataset_index:02d}/{len(selected_dataset_names):02d}] "
            f"{dataset_name}"
        )

        dataset_plot_summary = plot_combined_metrics(
            dataframe=dataset_metrics,
            mode_order=shared_mode_order,
            title=(
                f"{dataset_name}\n"
                "CIPHER forward prediction across normalizations"
            ),
            output_base=output_base,
            scope="dataset",
            dataset_name=dataset_name,
        )

        plot_summary_frames.append(
            dataset_plot_summary
        )

        plot_manifest_rows.append(
            {
                "plot_order": dataset_index,
                "scope": "dataset",
                "dataset": dataset_name,
                "n_metric_rows": int(
                    len(dataset_metrics)
                ),
                "png_path": str(
                    output_base.with_suffix(".png")
                ),
                "svg_path": str(
                    output_base.with_suffix(".svg")
                ),
            }
        )


    # ============================================================
    # MAKE POOLED COMPOSITE FIGURE LAST
    # ============================================================

    print("\n" + "=" * 120)
    print("MAKING POOLED COMPOSITE FIGURE")
    print("=" * 120)

    composite_output_base = (
        COMPOSITE_PLOT_DIR
        / "all_datasets_composite_pearson_and_r2"
    )

    composite_plot_summary = plot_combined_metrics(
        dataframe=ok_metrics,
        mode_order=shared_mode_order,
        title=(
            "All retained datasets\n"
            "CIPHER forward prediction across normalizations"
        ),
        output_base=composite_output_base,
        scope="composite",
        dataset_name="ALL_DATASETS",
    )

    plot_summary_frames.append(
        composite_plot_summary
    )

    plot_manifest_rows.append(
        {
            "plot_order": len(selected_dataset_names) + 1,
            "scope": "composite",
            "dataset": "ALL_DATASETS",
            "n_metric_rows": int(
                len(ok_metrics)
            ),
            "png_path": str(
                composite_output_base.with_suffix(".png")
            ),
            "svg_path": str(
                composite_output_base.with_suffix(".svg")
            ),
        }
    )


    # ============================================================
    # SAVE PLOT SUMMARIES
    # ============================================================

    if plot_summary_frames:
        plot_summary_df = pd.concat(
            plot_summary_frames,
            ignore_index=True,
        )

    else:
        plot_summary_df = pd.DataFrame()

    plot_manifest_df = pd.DataFrame(
        plot_manifest_rows
    )

    plot_summary_path = (
        OUTDIR
        / "plot_metric_summary_dataset_specific_and_composite.tsv"
    )

    plot_manifest_path = (
        OUTDIR
        / "plot_manifest_dataset_specific_and_composite.tsv"
    )

    plot_summary_df.to_csv(
        plot_summary_path,
        sep="\t",
        index=False,
    )

    plot_manifest_df.to_csv(
        plot_manifest_path,
        sep="\t",
        index=False,
    )


    # ============================================================
    # RUN SUMMARY
    # ============================================================

    n_ok_rows = 0

    if (
        len(metrics_df) > 0
        and "status" in metrics_df.columns
    ):
        n_ok_rows = int(
            np.sum(
                metrics_df["status"].astype(str) == "ok"
            )
        )

    marson_token_lower = str(
        MARSON_NAME_TOKEN
    ).lower()

    marson_datasets_discovered = [
        dataset_dir.name
        for dataset_dir in all_dataset_dirs
        if marson_token_lower in dataset_dir.name.lower()
    ]

    marson_datasets_retained = [
        dataset_dir.name
        for dataset_dir in dataset_dirs
        if marson_token_lower in dataset_dir.name.lower()
    ]

    marson_datasets_excluded = [
        dataset_name
        for dataset_name in marson_datasets_discovered
        if dataset_name not in marson_datasets_retained
    ]

    run_summary = {
        "precompute_root": str(PRECOMPUTE_ROOT),
        "outdir": str(OUTDIR),

        "n_dataset_dirs_discovered": int(
            len(all_dataset_dirs)
        ),

        "n_dataset_dirs_retained": int(
            len(dataset_dirs)
        ),

        "dataset_dirs_retained": selected_dataset_names,

        "marson_name_token": str(
            MARSON_NAME_TOKEN
        ),

        "marson_dataset_to_keep_requested": (
            MARSON_DATASET_TO_KEEP
        ),

        "marson_datasets_discovered": (
            marson_datasets_discovered
        ),

        "marson_datasets_retained": (
            marson_datasets_retained
        ),

        "marson_datasets_excluded": (
            marson_datasets_excluded
        ),

        "modes": MODES,

        "mode_labels": MODE_LABELS,

        "shared_plot_mode_order": shared_mode_order,

        "order_modes_by_metric": (
            ORDER_MODES_BY_METRIC
        ),

        "holdout_frac": float(
            HOLDOUT_FRAC
        ),

        "train_equals_test": bool(
            HOLDOUT_FRAC <= 0
        ),

        "exclude_target_gene_from_fit": bool(
            EXCLUDE_TARGET_GENE_FROM_FIT
        ),

        "exclude_target_gene_from_eval": bool(
            EXCLUDE_TARGET_GENE_FROM_EVAL
        ),

        "min_sigma_col_norm2": float(
            MIN_SIGMA_COL_NORM2
        ),

        "min_train_genes": int(
            MIN_TRAIN_GENES
        ),

        "min_test_genes": int(
            MIN_TEST_GENES
        ),

        "n_metric_rows": int(
            len(metrics_df)
        ),

        "n_ok_rows": int(
            n_ok_rows
        ),

        "n_dataset_specific_plots": int(
            len(selected_dataset_names)
        ),

        "n_composite_plots": 1,

        "metrics_path": str(
            metrics_path
        ),

        "status_path": str(
            status_path
        ),

        "summary_path": str(
            summary_path
        ),

        "plot_summary_path": str(
            plot_summary_path
        ),

        "plot_manifest_path": str(
            plot_manifest_path
        ),

        "per_dataset_plot_dir": str(
            PER_DATASET_PLOT_DIR
        ),

        "composite_plot_dir": str(
            COMPOSITE_PLOT_DIR
        ),

        "composite_png": str(
            composite_output_base.with_suffix(".png")
        ),

        "composite_svg": str(
            composite_output_base.with_suffix(".svg")
        ),
    }

    run_summary_path = (
        OUTDIR
        / "forward_recompute_summary.json"
    )

    with open(
        run_summary_path,
        "w",
    ) as file_handle:
        json.dump(
            run_summary,
            file_handle,
            indent=2,
            default=json_default,
        )


    # ============================================================
    # FINAL PRINTING
    # ============================================================

    print("\n" + "=" * 120)
    print("DONE")
    print("=" * 120)

    print(
        f"Per-perturbation metrics:  "
        f"{metrics_path}"
    )

    print(
        f"Per-dataset/mode summary:  "
        f"{summary_path}"
    )

    print(
        f"Run status:                "
        f"{status_path}"
    )

    print(
        f"Plot metric summary:       "
        f"{plot_summary_path}"
    )

    print(
        f"Plot manifest:             "
        f"{plot_manifest_path}"
    )

    print(
        f"Run summary JSON:          "
        f"{run_summary_path}"
    )

    print(
        f"Dataset-specific plots:    "
        f"{PER_DATASET_PLOT_DIR}"
    )

    print(
        f"Composite plot:            "
        f"{composite_output_base}"
    )

    print(
        f"Dataset plots made:        "
        f"{len(selected_dataset_names)}"
    )

    print(
        "Composite plots made:      1"
    )

    print("=" * 120)

    if len(status_df) > 0:
        print("\n[run status counts]")

        print(
            status_df["status"]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    if len(summary_df) > 0:
        display_columns = [
            "dataset",
            "mode",
            "n_perturbations_ok",
            "pearson_mean",
            "spearman_mean",
            "cosine_mean",
            "r2_uncentered_mean",
            "r2_centered_mean",
            "mse_mean",
            "mae_mean",
            "sign_accuracy_mean",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in summary_df.columns
        ]

        print("\n[summary head]")

        print(
            summary_df[
                display_columns
            ]
            .head(50)
            .to_string(
                index=False
            )
        )

    if len(plot_manifest_df) > 0:
        print("\n[plot manifest]")

        print(
            plot_manifest_df[
                [
                    "plot_order",
                    "scope",
                    "dataset",
                    "n_metric_rows",
                    "png_path",
                ]
            ].to_string(
                index=False
            )
        )

    gc.collect()

