"""Notebook-only run module for Fig S9 (inverse-inference true-perturbation gene ranking).

Holds the relocated main-flow orchestration for ``notebooks/suppl/figS9_generanking.ipynb``.
Each function corresponds one-to-one to a former big notebook cell and contains that
cell's statements VERBATIM (only dedented from the cell and re-indented one level into
the ``def`` body). Nothing was rewritten -- same variables, same computations, same
matplotlib / savefig calls, same printed output.

Config lives as MODULE GLOBALS, exactly the names the original single-namespace notebook
used. The thin notebook injects ``DATA_DIR`` / ``SUPPL`` / ``OUTDIR`` (and any other
UPPER-case config) into this module and into ``src.suppl_generanking`` before calling
these functions; each function additionally declares its own UPPER-case config names
``global`` so their in-body assignments land in this module's namespace -- this reproduces
the original notebook, where every cell's top-level assignments accumulated in one shared
namespace, and it makes the verbatim ``src.suppl_generanking`` injection line resolve the
variant-specific plot constants correctly.

NOT part of the installable ``cipher`` package.
"""
from src.suppl_generanking import *

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_summarize():
    global OUT_ROOT, EXPRESSION_CUTOFF, RUN_OUT, METHODS_TO_LOAD, METHOD_LABELS, CRISPRa_KEYWORDS, CRISPRi_KEYWORDS, DISPLAY_NAME_MAP
    OUT_ROOT = os.path.join(SUPPL, "posterior_inverse_fast_from_prerun_fullH_diag")

    EXPRESSION_CUTOFF = 1.0

    RUN_OUT = None

    METHODS_TO_LOAD = [
        "score_pval",
        "score_pip_full",
        "score_lfc_abs",
        "score_shuffle",
        "score_mean_field",
        "score_true",
    ]

    METHOD_LABELS = {
        "score_pval": "-log10(p)",
        "score_pip_full": "PIP",
        "score_lfc_abs": "|LFC|",
        "score_shuffle": "shuffle",
        "score_mean_field": "mean-field",
        "score_true": "true Sigma",
    }

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

    # inject this notebook's config into the helper module so its functions
    # resolve the plot-config constants as module globals (matching the
    # original single-namespace notebook)
    import src.suppl_generanking as _M
    _M.__dict__.update({k: v for k, v in list(globals().items()) if k.isupper()})

    if RUN_OUT is None:
        RUN_OUT = find_latest_run(
            out_root=OUT_ROOT,
            expression_cutoff=EXPRESSION_CUTOFF,
        )
    else:
        RUN_OUT = Path(RUN_OUT)

    RUN_OUT = Path(RUN_OUT)

    if not RUN_OUT.exists():
        raise FileNotFoundError(
            f"Run directory does not exist:\n{RUN_OUT}"
        )

    print("=" * 100)

    print("LOAD SAVED TRUE-TARGET ABSOLUTE RANKS")

    print("=" * 100)

    print("[run]", RUN_OUT)

    combined_summary_path = (
        RUN_OUT / "ALL_DATASETS_inverse_summary.csv"
    )

    if combined_summary_path.exists():
        combined_summary = pd.read_csv(
            combined_summary_path,
            low_memory=False,
        )

        if "dataset" not in combined_summary.columns:
            raise ValueError(
                f"{combined_summary_path} is missing the "
                "'dataset' column."
            )

        datasets = sorted(
            combined_summary["dataset"]
            .dropna()
            .astype(str)
            .unique()
        )

        print(
            f"[datasets] found {len(datasets)} datasets "
            f"in {combined_summary_path.name}"
        )

    else:
        # Fallback to every subdirectory containing perpert_metrics.npz.
        datasets = sorted({
            path.parent.name
            for path in RUN_OUT.rglob("perpert_metrics.npz")
        })

        print(
            f"[datasets] combined summary not found; "
            f"discovered {len(datasets)} datasets from "
            "perpert_metrics.npz files"
        )

    if not datasets:
        raise RuntimeError(
            f"No dataset outputs found under:\n{RUN_OUT}"
        )

    all_rows = []

    manifest_rows = []

    for dataset_number, dataset in enumerate(
        datasets,
        start=1,
    ):
        dataset_dir = RUN_OUT / dataset
        metrics_path = dataset_dir / "perpert_metrics.npz"

        print()
        print(
            f"[{dataset_number:02d}/{len(datasets):02d}] "
            f"{dataset}"
        )

        if not metrics_path.exists():
            print(
                f"    [skip] missing {metrics_path.name}"
            )

            manifest_rows.append({
                "dataset": dataset,
                "dataset_display": short_dataset_name(dataset),
                "dataset_group": classify_dataset(dataset),
                "status": "missing_perpert_metrics",
                "metrics_path": str(metrics_path),
                "n_perturbations": 0,
                "n_valid_targets": 0,
                "methods_loaded": "",
            })
            continue

        with np.load(
            metrics_path,
            allow_pickle=True,
        ) as metrics:
            available_keys = set(metrics.files)

            required_keys = {
                "perturbations",
                "target_genes",
                "target_idx",
            }

            missing_required = (
                required_keys - available_keys
            )

            if missing_required:
                print(
                    "    [skip] missing required arrays:",
                    sorted(missing_required),
                )

                manifest_rows.append({
                    "dataset": dataset,
                    "dataset_display": short_dataset_name(dataset),
                    "dataset_group": classify_dataset(dataset),
                    "status": (
                        "missing_required_arrays: "
                        + "|".join(sorted(missing_required))
                    ),
                    "metrics_path": str(metrics_path),
                    "n_perturbations": 0,
                    "n_valid_targets": 0,
                    "methods_loaded": "",
                })
                continue

            perturbations = decode_string_array(
                metrics["perturbations"]
            )

            target_genes = decode_string_array(
                metrics["target_genes"]
            )

            target_idx = np.asarray(
                metrics["target_idx"],
                dtype=np.int64,
            ).reshape(-1)

            n_perturbations = len(perturbations)

            if len(target_genes) != n_perturbations:
                raise ValueError(
                    f"{dataset}: target_genes has length "
                    f"{len(target_genes)}, but perturbations has "
                    f"length {n_perturbations}."
                )

            if len(target_idx) != n_perturbations:
                raise ValueError(
                    f"{dataset}: target_idx has length "
                    f"{len(target_idx)}, but perturbations has "
                    f"length {n_perturbations}."
                )

            methods_found = []

            for method in METHODS_TO_LOAD:
                rank_key = f"{method}_rank"

                if rank_key not in available_keys:
                    print(
                        f"    [method skip] {rank_key} not found"
                    )
                    continue

                ranks = np.asarray(
                    metrics[rank_key],
                    dtype=np.float64,
                ).reshape(-1)

                if len(ranks) != n_perturbations:
                    raise ValueError(
                        f"{dataset}: {rank_key} has length "
                        f"{len(ranks)}, expected {n_perturbations}."
                    )

                methods_found.append(method)

                for perturbation_index in range(
                    n_perturbations
                ):
                    perturbation = str(
                        perturbations[perturbation_index]
                    )

                    target_gene = str(
                        target_genes[perturbation_index]
                    )

                    target_gene_index = int(
                        target_idx[perturbation_index]
                    )

                    absolute_rank = float(
                        ranks[perturbation_index]
                    )

                    target_matched = (
                        target_gene_index >= 0
                    )

                    rank_is_valid = (
                        target_matched
                        and np.isfinite(absolute_rank)
                        and absolute_rank >= 1
                    )

                    all_rows.append({
                        "dataset": dataset,
                        "dataset_display": (
                            short_dataset_name(dataset)
                        ),
                        "dataset_group": (
                            classify_dataset(dataset)
                        ),
                        "perturbation_index": (
                            perturbation_index
                        ),
                        "perturbation": perturbation,
                        "true_target_gene": target_gene,
                        "true_target_index": (
                            target_gene_index
                        ),
                        "target_matched": target_matched,
                        "method": method,
                        "method_display": (
                            METHOD_LABELS.get(
                                method,
                                method,
                            )
                        ),
                        "absolute_rank": (
                            absolute_rank
                            if rank_is_valid
                            else np.nan
                        ),
                        "rank_ordinal": (
                            ordinal_rank(absolute_rank)
                            if rank_is_valid
                            else ""
                        ),
                        "rank_is_valid": rank_is_valid,
                        "top1": (
                            int(absolute_rank <= 1)
                            if rank_is_valid
                            else np.nan
                        ),
                        "top5": (
                            int(absolute_rank <= 5)
                            if rank_is_valid
                            else np.nan
                        ),
                        "top10": (
                            int(absolute_rank <= 10)
                            if rank_is_valid
                            else np.nan
                        ),
                        "top25": (
                            int(absolute_rank <= 25)
                            if rank_is_valid
                            else np.nan
                        ),
                        "top50": (
                            int(absolute_rank <= 50)
                            if rank_is_valid
                            else np.nan
                        ),
                        "metrics_path": str(metrics_path),
                    })

            n_valid_targets = int(
                np.sum(target_idx >= 0)
            )

            manifest_rows.append({
                "dataset": dataset,
                "dataset_display": short_dataset_name(dataset),
                "dataset_group": classify_dataset(dataset),
                "status": (
                    "ok"
                    if methods_found
                    else "no_requested_rank_arrays"
                ),
                "metrics_path": str(metrics_path),
                "n_perturbations": n_perturbations,
                "n_valid_targets": n_valid_targets,
                "n_unmatched_targets": (
                    n_perturbations - n_valid_targets
                ),
                "methods_loaded": "|".join(
                    methods_found
                ),
            })

            print(
                f"    [perturbations] {n_perturbations:,}"
            )
            print(
                f"    [matched targets] "
                f"{n_valid_targets:,}/{n_perturbations:,}"
            )
            print(
                f"    [methods] "
                f"{', '.join(methods_found)}"
            )

    rank_df = pd.DataFrame(all_rows)

    manifest_df = pd.DataFrame(manifest_rows)

    if rank_df.empty:
        manifest_path = (
            Path(OUTDIR) / "TRUE_TARGET_RANK_LOAD_MANIFEST.csv"
        )

        manifest_df.to_csv(
            manifest_path,
            index=False,
        )

        raise RuntimeError(
            "No rank arrays were loaded.\n"
            f"Inspect:\n{manifest_path}"
        )

    rank_df = rank_df.sort_values(
        [
            "method",
            "dataset_group",
            "dataset",
            "perturbation_index",
        ],
        kind="stable",
    ).reset_index(drop=True)

    valid_rank_df = rank_df[
        rank_df["rank_is_valid"]
    ].copy()

    invalid_rank_df = rank_df[
        ~rank_df["rank_is_valid"]
    ].copy()

    print()

    print(
        f"[loaded rows] {len(rank_df):,}"
    )

    print(
        f"[valid rank rows] {len(valid_rank_df):,}"
    )

    print(
        f"[invalid rank rows] {len(invalid_rank_df):,}"
    )

    per_dataset_summary = summarize_ranks(
        dataframe=valid_rank_df,
        group_columns=[
            "dataset",
            "dataset_display",
            "dataset_group",
            "method",
            "method_display",
        ],
        rank_column="absolute_rank",
        count_name="n_perturbations",
    )

    per_dataset_summary = per_dataset_summary.sort_values(
        [
            "method",
            "dataset_group",
            "mean_rank",
        ],
        kind="stable",
    ).reset_index(drop=True)

    group_pooled_summary = summarize_ranks(
        dataframe=valid_rank_df,
        group_columns=[
            "dataset_group",
            "method",
            "method_display",
        ],
        rank_column="absolute_rank",
        count_name="n_perturbations",
    )

    group_pooled_summary = group_pooled_summary.sort_values(
        [
            "method",
            "dataset_group",
        ],
        kind="stable",
    ).reset_index(drop=True)

    pooled_input = valid_rank_df.copy()

    pooled_input["pooled_group"] = "all_datasets"

    pooled_summary = summarize_ranks(
        dataframe=pooled_input,
        group_columns=[
            "pooled_group",
            "method",
            "method_display",
        ],
        rank_column="absolute_rank",
        count_name="n_perturbations",
    )

    pooled_summary = pooled_summary.sort_values(
        ["method"],
        kind="stable",
    ).reset_index(drop=True)

    dataset_means = (
        valid_rank_df.groupby(
            [
                "dataset",
                "dataset_display",
                "dataset_group",
                "method",
                "method_display",
            ],
            dropna=False,
            sort=False,
        )
        .agg(
            dataset_mean_rank=(
                "absolute_rank",
                "mean",
            ),
            dataset_median_rank=(
                "absolute_rank",
                "median",
            ),
            n_perturbations=(
                "absolute_rank",
                "count",
            ),
        )
        .reset_index()
    )

    dataset_balanced_rows = []

    for (
        dataset_group,
        method,
        method_display,
    ), group in dataset_means.groupby(
        [
            "dataset_group",
            "method",
            "method_display",
        ],
        dropna=False,
        sort=False,
    ):
        values = pd.to_numeric(
            group["dataset_mean_rank"],
            errors="coerce",
        ).dropna()

        mean_value = (
            float(values.mean())
            if len(values)
            else np.nan
        )

        sem_value = standard_error(values)

        dataset_balanced_rows.append({
            "dataset_group": dataset_group,
            "method": method,
            "method_display": method_display,
            "n_datasets": int(len(values)),
            "mean_dataset_rank": mean_value,
            "sem_across_datasets": sem_value,
            "median_dataset_rank": (
                float(values.median())
                if len(values)
                else np.nan
            ),
            "min_dataset_mean_rank": (
                float(values.min())
                if len(values)
                else np.nan
            ),
            "max_dataset_mean_rank": (
                float(values.max())
                if len(values)
                else np.nan
            ),
            "mean_plus_minus_sem": (
                f"{mean_value:.3f} ± {sem_value:.3f}"
                if pd.notna(sem_value)
                else f"{mean_value:.3f} ± NA"
            ),
        })

    dataset_balanced_summary = pd.DataFrame(
        dataset_balanced_rows
    ).sort_values(
        [
            "method",
            "dataset_group",
        ],
        kind="stable",
    ).reset_index(drop=True)

    all_ranks_path = (
        Path(OUTDIR) / "ALL_DATASETS_TRUE_TARGET_ABSOLUTE_RANKS.csv"
    )

    valid_ranks_path = (
        Path(OUTDIR) / "ALL_DATASETS_VALID_TRUE_TARGET_RANKS.csv"
    )

    invalid_ranks_path = (
        Path(OUTDIR) / "ALL_DATASETS_INVALID_OR_UNMATCHED_TARGET_RANKS.csv"
    )

    per_dataset_summary_path = (
        Path(OUTDIR) / "PER_DATASET_MEAN_TRUE_TARGET_RANK_SEM.csv"
    )

    group_pooled_summary_path = (
        Path(OUTDIR) / "CRISPRI_CRISPRA_POOLED_TRUE_TARGET_RANK_SUMMARY.csv"
    )

    pooled_summary_path = (
        Path(OUTDIR) / "ALL_DATASETS_POOLED_TRUE_TARGET_RANK_SUMMARY.csv"
    )

    dataset_means_path = (
        Path(OUTDIR) / "PER_DATASET_MEAN_RANKS_FOR_BALANCED_SUMMARY.csv"
    )

    dataset_balanced_summary_path = (
        Path(OUTDIR) / "DATASET_BALANCED_TRUE_TARGET_RANK_SUMMARY.csv"
    )

    manifest_path = (
        Path(OUTDIR) / "TRUE_TARGET_RANK_LOAD_MANIFEST.csv"
    )

    text_summary_path = (
        Path(OUTDIR) / "TRUE_TARGET_RANK_MEAN_SEM_SUMMARY.txt"
    )

    rank_df.to_csv(
        all_ranks_path,
        index=False,
    )

    valid_rank_df.to_csv(
        valid_ranks_path,
        index=False,
    )

    invalid_rank_df.to_csv(
        invalid_ranks_path,
        index=False,
    )

    per_dataset_summary.to_csv(
        per_dataset_summary_path,
        index=False,
    )

    group_pooled_summary.to_csv(
        group_pooled_summary_path,
        index=False,
    )

    pooled_summary.to_csv(
        pooled_summary_path,
        index=False,
    )

    dataset_means.to_csv(
        dataset_means_path,
        index=False,
    )

    dataset_balanced_summary.to_csv(
        dataset_balanced_summary_path,
        index=False,
    )

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    with open(
        text_summary_path,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "TRUE-PERTURBATION ABSOLUTE-RANK SUMMARY\n"
        )
        handle.write("=" * 80 + "\n\n")

        handle.write(
            "Rank 1 means that the true perturbed gene was "
            "the highest-ranked candidate.\n"
        )
        handle.write(
            "Lower absolute rank is better.\n\n"
        )

        for method in METHODS_TO_LOAD:
            method_rows = per_dataset_summary[
                per_dataset_summary["method"] == method
            ].copy()

            if method_rows.empty:
                continue

            method_display = METHOD_LABELS.get(
                method,
                method,
            )

            handle.write(
                f"METHOD: {method_display} [{method}]\n"
            )
            handle.write("-" * 80 + "\n")

            for dataset_group in [
                "CRISPRi",
                "CRISPRa",
                "other",
            ]:
                subset = method_rows[
                    method_rows["dataset_group"]
                    == dataset_group
                ].copy()

                if subset.empty:
                    continue

                subset = subset.sort_values(
                    "mean_rank",
                    ascending=True,
                )

                handle.write(
                    f"\n{dataset_group}\n"
                )

                for _, row in subset.iterrows():
                    sem_text = (
                        f"{row['sem_rank']:.3f}"
                        if pd.notna(row["sem_rank"])
                        else "NA"
                    )

                    handle.write(
                        f"  {row['dataset_display']}: "
                        f"{row['mean_rank']:.3f} ± {sem_text} "
                        f"(n={int(row['n_perturbations'])}, "
                        f"median={row['median_rank']:.1f}, "
                        f"range={row['min_rank']:.0f}-"
                        f"{row['max_rank']:.0f})\n"
                    )

            pooled_hit = pooled_summary[
                pooled_summary["method"] == method
            ]

            if not pooled_hit.empty:
                row = pooled_hit.iloc[0]

                sem_text = (
                    f"{row['sem_rank']:.3f}"
                    if pd.notna(row["sem_rank"])
                    else "NA"
                )

                handle.write(
                    "\nPOOLED ACROSS ALL PERTURBATIONS\n"
                )
                handle.write(
                    f"  {row['mean_rank']:.3f} ± {sem_text} "
                    f"(n={int(row['n_perturbations'])}, "
                    f"median={row['median_rank']:.1f})\n"
                )

            balanced_hits = dataset_balanced_summary[
                dataset_balanced_summary["method"]
                == method
            ]

            if not balanced_hits.empty:
                handle.write(
                    "\nDATASET-BALANCED\n"
                )

                for _, row in balanced_hits.iterrows():
                    sem_text = (
                        f"{row['sem_across_datasets']:.3f}"
                        if pd.notna(
                            row["sem_across_datasets"]
                        )
                        else "NA"
                    )

                    handle.write(
                        f"  {row['dataset_group']}: "
                        f"{row['mean_dataset_rank']:.3f} "
                        f"± {sem_text} "
                        f"(n datasets={int(row['n_datasets'])})\n"
                    )

            handle.write("\n\n")

    print()

    print("=" * 100)

    print("PER-DATASET MEAN ABSOLUTE RANK ± SEM")

    print("=" * 100)

    columns_to_print = [
        "dataset_group",
        "dataset_display",
        "method",
        "n_perturbations",
        "mean_rank",
        "sem_rank",
        "median_rank",
        "top1_fraction",
        "top10_fraction",
    ]

    print(
        per_dataset_summary[
            columns_to_print
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()

    print("=" * 100)

    print("POOLED ACROSS ALL PERTURBATIONS")

    print("=" * 100)

    print(
        pooled_summary[
            [
                "method",
                "n_perturbations",
                "mean_rank",
                "sem_rank",
                "median_rank",
                "top1_fraction",
                "top10_fraction",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()

    print("=" * 100)

    print("FIRST 30 TRUE-TARGET RANKS")

    print("=" * 100)

    print(
        valid_rank_df[
            [
                "dataset_display",
                "perturbation",
                "true_target_gene",
                "method",
                "absolute_rank",
                "rank_ordinal",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print()

    print("=" * 100)

    print("SAVED")

    print("=" * 100)

    for path in [
        all_ranks_path,
        valid_ranks_path,
        invalid_ranks_path,
        per_dataset_summary_path,
        group_pooled_summary_path,
        pooled_summary_path,
        dataset_means_path,
        dataset_balanced_summary_path,
        manifest_path,
        text_summary_path,
    ]:
        print(path)

    print()

    print(
        "[complete] "
        f"{valid_rank_df['dataset'].nunique():,} datasets, "
        f"{valid_rank_df['perturbation'].nunique():,} unique "
        f"perturbation names, "
        f"{valid_rank_df['method'].nunique():,} methods, and "
        f"{len(valid_rank_df):,} valid "
        "dataset × perturbation × method rank observations."
    )


def figure_variant_a():
    global OUT_ROOT, EXPRESSION_CUTOFF, RUN_OUT, INPUT_FILENAME, METHODS, METHOD_LABELS, METHOD_ORDER, DPI, FONTSIZE, JITTER_SD, RANDOM_SEED, DRAW_PAIRED_LINES, DRAW_VIOLINS, VIOLIN_WIDTH, LOG_RANK_AXIS
    OUT_ROOT = os.path.join(SUPPL, "posterior_inverse_fast_from_prerun_fullH_diag")

    EXPRESSION_CUTOFF = 1.0

    RUN_OUT = None

    INPUT_FILENAME = "ALL_DATASETS_VALID_TRUE_TARGET_RANKS.csv"

    METHODS = [
        "score_lfc_abs",
        # "score_mean_field",
        "score_true",
    ]

    METHOD_LABELS = {
        "score_lfc_abs": "|LFC|",
        # "score_mean_field": "Mean-field",
        "score_true": "True covariance",
    }

    METHOD_ORDER = [
        "|LFC|",
        # "Mean-field",
        "True covariance",
    ]

    DPI = 300

    FONTSIZE = 15

    JITTER_SD = 0.045

    RANDOM_SEED = 0

    DRAW_PAIRED_LINES = True

    DRAW_VIOLINS = True

    VIOLIN_WIDTH = 0.72

    LOG_RANK_AXIS = False

    # inject this notebook's config into the helper module so its functions
    # resolve the plot-config constants as module globals (matching the
    # original single-namespace notebook)
    import src.suppl_generanking as _M
    _M.__dict__.update({k: v for k, v in list(globals().items()) if k.isupper()})

    def add_summary_marker(
        ax,
        x_position,
        values,
    ):
        """
        Add mean +/- SEM for the dataset-level metric.
        """
        values = pd.to_numeric(
            pd.Series(values),
            errors="coerce",
        ).dropna()

        if values.empty:
            return

        mean_value = float(values.mean())
        sem_value = dataset_sem(values)

        if pd.notna(sem_value):
            ax.errorbar(
                x_position,
                mean_value,
                yerr=sem_value,
                fmt="D",
                markersize=7,
                markeredgecolor="black",
                markerfacecolor="white",
                ecolor="black",
                elinewidth=1.8,
                capsize=4,
                capthick=1.5,
                zorder=10,
            )
        else:
            ax.scatter(
                [x_position],
                [mean_value],
                marker="D",
                s=55,
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
                zorder=10,
            )

    def plot_metric_panel(
        ax,
        dataset_metric_df,
        metric_column,
        ylabel,
        title,
        rng,
        percentage=False,
    ):
        positions = np.arange(len(METHOD_ORDER))

        metric_wide = (
            dataset_metric_df.pivot_table(
                index=[
                    "dataset",
                    "dataset_display",
                    "dataset_group",
                ],
                columns="method_label",
                values=metric_column,
                aggfunc="first",
                observed=False,
            )
            .reindex(columns=METHOD_ORDER)
        )

        # --------------------------------------------------------
        # Violin distributions
        # --------------------------------------------------------

        values_by_method = [
            metric_wide[label]
            .dropna()
            .to_numpy(dtype=float)
            for label in METHOD_ORDER
        ]

        if DRAW_VIOLINS:
            add_violin(
                ax=ax,
                values_by_method=values_by_method,
                positions=positions,
            )

        # --------------------------------------------------------
        # Paired lines between methods
        # --------------------------------------------------------

        if DRAW_PAIRED_LINES:
            for _, row in metric_wide.iterrows():
                values = row.to_numpy(dtype=float)
                finite = np.isfinite(values)

                if np.sum(finite) >= 2:
                    ax.plot(
                        positions[finite],
                        values[finite],
                        linewidth=0.7,
                        alpha=0.16,
                        color="black",
                        zorder=1,
                    )

        # --------------------------------------------------------
        # Dataset dots
        # --------------------------------------------------------

        for method_index, method_label in enumerate(
            METHOD_ORDER
        ):
            values = metric_wide[method_label].dropna()

            if values.empty:
                continue

            jitter = rng.normal(
                loc=0.0,
                scale=JITTER_SD,
                size=len(values),
            )

            ax.scatter(
                np.full(len(values), method_index) + jitter,
                values.to_numpy(dtype=float),
                s=42,
                alpha=0.78,
                edgecolor="black",
                linewidth=0.45,
                zorder=5,
            )

            add_summary_marker(
                ax=ax,
                x_position=method_index,
                values=values,
            )

        # --------------------------------------------------------
        # Axes
        # --------------------------------------------------------

        ax.set_xticks(positions)
        ax.set_xticklabels(
            METHOD_ORDER,
            rotation=20,
            ha="right",
        )

        ax.set_ylabel(ylabel)
        ax.set_title(
            title,
            fontsize=FONTSIZE + 1,
            fontweight="bold",
        )

        ax.set_xlim(
            -0.65,
            len(METHOD_ORDER) - 0.35,
        )

        if percentage:
            ax.set_ylim(-0.025, 1.025)

            ticks = np.linspace(0, 1, 6)
            ax.set_yticks(ticks)
            ax.set_yticklabels(
                [f"{100 * value:.0f}%" for value in ticks]
            )

        if LOG_RANK_AXIS and metric_column in {
            "mean_rank",
            "median_rank",
        }:
            ax.set_yscale("log")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if RUN_OUT is None:
        RUN_OUT = find_latest_run(
            out_root=OUT_ROOT,
            expression_cutoff=EXPRESSION_CUTOFF,
        )
    else:
        RUN_OUT = Path(RUN_OUT)

    RUN_OUT = Path(RUN_OUT)

    input_path = Path(OUTDIR) / INPUT_FILENAME

    if not input_path.exists():
        raise FileNotFoundError(
            "Could not find the saved valid-rank table:\n"
            f"{input_path}\n\n"
            "Run the rank-extraction script first."
        )

    print("=" * 80)

    print("DATASET-LEVEL RANK METRIC PLOT")

    print("=" * 80)

    print("[run]", RUN_OUT)

    print("[input]", input_path)

    rank_df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    required_columns = {
        "dataset",
        "method",
        "absolute_rank",
    }

    missing_columns = required_columns - set(rank_df.columns)

    if missing_columns:
        raise ValueError(
            f"Input table is missing columns: {sorted(missing_columns)}"
        )

    if "dataset_display" not in rank_df.columns:
        rank_df["dataset_display"] = rank_df["dataset"].astype(str)

    if "dataset_group" not in rank_df.columns:
        rank_df["dataset_group"] = "all"

    rank_df["absolute_rank"] = pd.to_numeric(
        rank_df["absolute_rank"],
        errors="coerce",
    )

    rank_df = rank_df[
        rank_df["method"].isin(METHODS)
        & np.isfinite(rank_df["absolute_rank"])
        & (rank_df["absolute_rank"] >= 1)
    ].copy()

    if rank_df.empty:
        raise RuntimeError(
            "No valid rows remained after restricting to:\n"
            f"{METHODS}"
        )

    methods_found = sorted(
        rank_df["method"].unique()
    )

    missing_methods = [
        method
        for method in METHODS
        if method not in methods_found
    ]

    if missing_methods:
        print(
            "[warning] missing requested methods:",
            missing_methods,
        )

    print(
        f"[rank rows] {len(rank_df):,}"
    )

    print(
        f"[datasets] {rank_df['dataset'].nunique():,}"
    )

    print(
        f"[methods] {methods_found}"
    )

    dataset_metrics = calculate_dataset_metrics(
        rank_df
    )

    metrics_output_path = (
        Path(OUTDIR) / "DATASET_RANK_METRICS_LFC_MEANFIELD_TRUE.csv"
    )

    dataset_metrics.to_csv(
        metrics_output_path,
        index=False,
    )

    print()

    print("[dataset-level metrics]")

    print(
        dataset_metrics[
            [
                "dataset_group",
                "dataset_display",
                "method_label",
                "n_perturbations",
                "mean_rank",
                "median_rank",
                "top1_fraction",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    across_dataset_summary = (
        dataset_metrics.groupby(
            [
                "method",
                "method_label",
            ],
            observed=False,
            dropna=False,
        )
        .agg(
            n_datasets=(
                "dataset",
                "nunique",
            ),
            mean_of_mean_ranks=(
                "mean_rank",
                "mean",
            ),
            sem_of_mean_ranks=(
                "mean_rank",
                dataset_sem,
            ),
            mean_of_median_ranks=(
                "median_rank",
                "mean",
            ),
            sem_of_median_ranks=(
                "median_rank",
                dataset_sem,
            ),
            mean_top1_fraction=(
                "top1_fraction",
                "mean",
            ),
            sem_top1_fraction=(
                "top1_fraction",
                dataset_sem,
            ),
        )
        .reset_index()
        .sort_values(
            "method_label",
            kind="stable",
        )
    )

    summary_output_path = (
        Path(OUTDIR)
        / "ACROSS_DATASET_RANK_METRICS_LFC_MEANFIELD_TRUE.csv"
    )

    across_dataset_summary.to_csv(
        summary_output_path,
        index=False,
    )

    print()

    print("[across-dataset mean ± SEM]")

    print(
        across_dataset_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    plt.rcParams.update({
        "font.size": FONTSIZE,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    })

    rng = np.random.default_rng(RANDOM_SEED)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.6),
    )

    plot_metric_panel(
        ax=axes[0],
        dataset_metric_df=dataset_metrics,
        metric_column="mean_rank",
        ylabel="Mean absolute rank",
        title="Mean rank",
        rng=rng,
        percentage=False,
    )

    plot_metric_panel(
        ax=axes[1],
        dataset_metric_df=dataset_metrics,
        metric_column="median_rank",
        ylabel="Median absolute rank",
        title="Median rank",
        rng=rng,
        percentage=False,
    )

    plot_metric_panel(
        ax=axes[2],
        dataset_metric_df=dataset_metrics,
        metric_column="top1_fraction",
        ylabel="Top-1 fraction",
        title="True target ranked first",
        rng=rng,
        percentage=True,
    )

    for panel_letter, ax in zip(
        ["A", "B", "C"],
        axes,
    ):
        ax.text(
            -0.18,
            1.06,
            panel_letter,
            transform=ax.transAxes,
            fontsize=FONTSIZE + 7,
            fontweight="bold",
            ha="left",
            va="top",
        )

    fig.suptitle(
        "True perturbation ranking across datasets",
        fontsize=FONTSIZE + 2,
        y=1.02,
    )

    fig.tight_layout()

    output_stem = (
        Path(OUTDIR)
        / "RANK_METRICS_LFC_MEANFIELD_TRUE_VIOLIN_DOT"
    )

    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    plt.show()

    print()

    print("=" * 80)

    print("SAVED")

    print("=" * 80)

    print(metrics_output_path)

    print(summary_output_path)

    print(output_stem.with_suffix(".png"))

    print(output_stem.with_suffix(".pdf"))

    print(output_stem.with_suffix(".svg"))


def figure_variant_b():
    global OUT_ROOT, EXPRESSION_CUTOFF, RUN_OUT, INPUT_FILENAME, METHODS, METHOD_LABELS, METHOD_ORDER, DPI, FONTSIZE, JITTER_SD, RANDOM_SEED, DRAW_PAIRED_LINES, DRAW_VIOLINS, VIOLIN_WIDTH, LOG_RANK_AXIS
    OUT_ROOT = os.path.join(SUPPL, "posterior_inverse_fast_from_prerun_fullH_diag")

    EXPRESSION_CUTOFF = 1.0

    RUN_OUT = None

    INPUT_FILENAME = "ALL_DATASETS_VALID_TRUE_TARGET_RANKS.csv"

    METHODS = [
        "score_lfc_abs",
        # "score_mean_field",
        "score_true",
    ]

    METHOD_LABELS = {
        "score_lfc_abs": "|LFC|",
        # "score_mean_field": "Mean-field",
        "score_true": "True covariance",
    }

    METHOD_ORDER = [
        "|LFC|",
        # "Mean-field",
        "True covariance",
    ]

    DPI = 300

    FONTSIZE = 15

    JITTER_SD = 0.045

    RANDOM_SEED = 0

    DRAW_PAIRED_LINES = False

    DRAW_VIOLINS = True

    VIOLIN_WIDTH = 0.72

    LOG_RANK_AXIS = False

    # inject this notebook's config into the helper module so its functions
    # resolve the plot-config constants as module globals (matching the
    # original single-namespace notebook)
    import src.suppl_generanking as _M
    _M.__dict__.update({k: v for k, v in list(globals().items()) if k.isupper()})

    def add_summary_marker(
        ax,
        x_position,
        values,
        percentage=False,
    ):
        """
        Add mean +/- SEM for the dataset-level metric and write the
        mean value above the white diamond.
        """
        values = pd.to_numeric(
            pd.Series(values),
            errors="coerce",
        ).dropna()

        if values.empty:
            return

        mean_value = float(values.mean())
        sem_value = dataset_sem(values)

        if pd.notna(sem_value):
            ax.errorbar(
                x_position,
                mean_value,
                yerr=sem_value,
                fmt="D",
                markersize=7,
                markeredgecolor="black",
                markerfacecolor="white",
                ecolor="black",
                elinewidth=1.8,
                capsize=4,
                capthick=1.5,
                zorder=10,
            )
        else:
            ax.scatter(
                [x_position],
                [mean_value],
                marker="D",
                s=55,
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
                zorder=10,
            )

        if percentage:
            mean_text = f"{100 * mean_value:.1f}%"
        else:
            mean_text = f"{mean_value:.1f}"

        ax.annotate(
            mean_text,
            xy=(x_position, mean_value),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            zorder=11,
        )

    def plot_metric_panel(
        ax,
        dataset_metric_df,
        metric_column,
        ylabel,
        title,
        rng,
        percentage=False,
    ):
        positions = np.arange(len(METHOD_ORDER))

        metric_wide = (
            dataset_metric_df.pivot_table(
                index=[
                    "dataset",
                    "dataset_display",
                    "dataset_group",
                ],
                columns="method_label",
                values=metric_column,
                aggfunc="first",
                observed=False,
            )
            .reindex(columns=METHOD_ORDER)
        )

        # --------------------------------------------------------
        # Violin distributions
        # --------------------------------------------------------

        values_by_method = [
            metric_wide[label]
            .dropna()
            .to_numpy(dtype=float)
            for label in METHOD_ORDER
        ]

        if DRAW_VIOLINS:
            add_violin(
                ax=ax,
                values_by_method=values_by_method,
                positions=positions,
            )

        # --------------------------------------------------------
        # Paired lines between methods
        # --------------------------------------------------------

        if DRAW_PAIRED_LINES:
            for _, row in metric_wide.iterrows():
                values = row.to_numpy(dtype=float)
                finite = np.isfinite(values)

                if np.sum(finite) >= 2:
                    ax.plot(
                        positions[finite],
                        values[finite],
                        linewidth=0.7,
                        alpha=0.16,
                        color="black",
                        zorder=1,
                    )

        # --------------------------------------------------------
        # Dataset dots
        # --------------------------------------------------------

        for method_index, method_label in enumerate(
            METHOD_ORDER
        ):
            values = metric_wide[method_label].dropna()

            if values.empty:
                continue

            jitter = rng.normal(
                loc=0.0,
                scale=JITTER_SD,
                size=len(values),
            )

            ax.scatter(
                np.full(len(values), method_index) + jitter,
                values.to_numpy(dtype=float),
                s=42,
                alpha=0.78,
                edgecolor="black",
                linewidth=0.45,
                zorder=5,
            )

            add_summary_marker(
                ax=ax,
                x_position=method_index,
                values=values,
                percentage=percentage,
            )

            label_best_and_worst(
                ax=ax,
                values=values,
                x_position=method_index,
                lower_is_better=not percentage,
            )

        # --------------------------------------------------------
        # Axes
        # --------------------------------------------------------

        ax.set_xticks(positions)
        ax.set_xticklabels(
            METHOD_ORDER,
            rotation=20,
            ha="right",
        )

        ax.set_ylabel(ylabel)
        ax.set_title(
            title,
            fontsize=FONTSIZE + 1,
            fontweight="bold",
        )

        ax.set_xlim(
            -0.65,
            len(METHOD_ORDER) - 0.35,
        )

        if percentage:
            ax.set_ylim(-0.025, 1.10)

            ticks = np.linspace(0, 1, 6)
            ax.set_yticks(ticks)
            ax.set_yticklabels(
                [f"{100 * value:.0f}%" for value in ticks]
            )

        if LOG_RANK_AXIS and metric_column in {
            "mean_rank",
            "median_rank",
        }:
            ax.set_yscale("log")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if RUN_OUT is None:
        RUN_OUT = find_latest_run(
            out_root=OUT_ROOT,
            expression_cutoff=EXPRESSION_CUTOFF,
        )
    else:
        RUN_OUT = Path(RUN_OUT)

    RUN_OUT = Path(RUN_OUT)

    input_path = Path(OUTDIR) / INPUT_FILENAME

    if not input_path.exists():
        raise FileNotFoundError(
            "Could not find the saved valid-rank table:\n"
            f"{input_path}\n\n"
            "Run the rank-extraction script first."
        )

    print("=" * 80)

    print("DATASET-LEVEL RANK METRIC PLOT")

    print("=" * 80)

    print("[run]", RUN_OUT)

    print("[input]", input_path)

    rank_df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    required_columns = {
        "dataset",
        "method",
        "absolute_rank",
    }

    missing_columns = required_columns - set(rank_df.columns)

    if missing_columns:
        raise ValueError(
            f"Input table is missing columns: {sorted(missing_columns)}"
        )

    if "dataset_display" not in rank_df.columns:
        rank_df["dataset_display"] = rank_df["dataset"].astype(str)

    if "dataset_group" not in rank_df.columns:
        rank_df["dataset_group"] = "all"

    rank_df["absolute_rank"] = pd.to_numeric(
        rank_df["absolute_rank"],
        errors="coerce",
    )

    rank_df = rank_df[
        rank_df["method"].isin(METHODS)
        & np.isfinite(rank_df["absolute_rank"])
        & (rank_df["absolute_rank"] >= 1)
    ].copy()

    if rank_df.empty:
        raise RuntimeError(
            "No valid rows remained after restricting to:\n"
            f"{METHODS}"
        )

    methods_found = sorted(
        rank_df["method"].unique()
    )

    missing_methods = [
        method
        for method in METHODS
        if method not in methods_found
    ]

    if missing_methods:
        print(
            "[warning] missing requested methods:",
            missing_methods,
        )

    print(
        f"[rank rows] {len(rank_df):,}"
    )

    print(
        f"[datasets] {rank_df['dataset'].nunique():,}"
    )

    print(
        f"[methods] {methods_found}"
    )

    dataset_metrics = calculate_dataset_metrics(
        rank_df
    )

    metrics_output_path = (
        Path(OUTDIR) / "DATASET_RANK_METRICS_LFC_MEANFIELD_TRUE.csv"
    )

    dataset_metrics.to_csv(
        metrics_output_path,
        index=False,
    )

    print()

    print("[dataset-level metrics]")

    print(
        dataset_metrics[
            [
                "dataset_group",
                "dataset_display",
                "method_label",
                "n_perturbations",
                "mean_rank",
                "median_rank",
                "top1_fraction",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    across_dataset_summary = (
        dataset_metrics.groupby(
            [
                "method",
                "method_label",
            ],
            observed=False,
            dropna=False,
        )
        .agg(
            n_datasets=(
                "dataset",
                "nunique",
            ),
            mean_of_mean_ranks=(
                "mean_rank",
                "mean",
            ),
            sem_of_mean_ranks=(
                "mean_rank",
                dataset_sem,
            ),
            mean_of_median_ranks=(
                "median_rank",
                "mean",
            ),
            sem_of_median_ranks=(
                "median_rank",
                dataset_sem,
            ),
            mean_top1_fraction=(
                "top1_fraction",
                "mean",
            ),
            sem_top1_fraction=(
                "top1_fraction",
                dataset_sem,
            ),
        )
        .reset_index()
        .sort_values(
            "method_label",
            kind="stable",
        )
    )

    summary_output_path = (
        Path(OUTDIR)
        / "ACROSS_DATASET_RANK_METRICS_LFC_MEANFIELD_TRUE.csv"
    )

    across_dataset_summary.to_csv(
        summary_output_path,
        index=False,
    )

    print()

    print("[across-dataset mean ± SEM]")

    print(
        across_dataset_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    plt.rcParams.update({
        "font.size": FONTSIZE,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    })

    rng = np.random.default_rng(RANDOM_SEED)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.6),
    )

    plot_metric_panel(
        ax=axes[0],
        dataset_metric_df=dataset_metrics,
        metric_column="mean_rank",
        ylabel="Mean absolute rank",
        title="Mean rank",
        rng=rng,
        percentage=False,
    )

    plot_metric_panel(
        ax=axes[1],
        dataset_metric_df=dataset_metrics,
        metric_column="median_rank",
        ylabel="Median absolute rank",
        title="Median rank",
        rng=rng,
        percentage=False,
    )

    plot_metric_panel(
        ax=axes[2],
        dataset_metric_df=dataset_metrics,
        metric_column="top1_fraction",
        ylabel="Top-1 fraction",
        title="True target ranked first",
        rng=rng,
        percentage=True,
    )

    for panel_letter, ax in zip(
        ["A", "B", "C"],
        axes,
    ):
        ax.text(
            -0.18,
            1.06,
            panel_letter,
            transform=ax.transAxes,
            fontsize=FONTSIZE + 7,
            fontweight="bold",
            ha="left",
            va="top",
        )

    fig.suptitle(
        "True perturbation ranking across datasets",
        fontsize=FONTSIZE + 2,
        y=1.02,
    )

    fig.tight_layout()

    output_stem = (
        Path(OUTDIR)
        / "RANK_METRICS_LFC_MEANFIELD_TRUE_VIOLIN_DOT"
    )

    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    plt.show()

    print()

    print("=" * 80)

    print("SAVED")

    print("=" * 80)

    print(metrics_output_path)

    print(summary_output_path)

    print(output_stem.with_suffix(".png"))

    print(output_stem.with_suffix(".pdf"))

    print(output_stem.with_suffix(".svg"))


def figure_variant_c():
    global OUT_ROOT, EXPRESSION_CUTOFF, RUN_OUT, INPUT_FILENAME, METHODS, METHOD_LABELS, METHOD_ORDER, DPI, FONTSIZE, JITTER_SD, RANDOM_SEED, DRAW_PAIRED_LINES, DRAW_VIOLINS, VIOLIN_WIDTH, LOG_RANK_AXIS
    OUT_ROOT = os.path.join(SUPPL, "posterior_inverse_fast_from_prerun_fullH_diag")

    EXPRESSION_CUTOFF = 1.0

    RUN_OUT = None

    INPUT_FILENAME = "ALL_DATASETS_VALID_TRUE_TARGET_RANKS.csv"

    METHODS = [
        "score_lfc_abs",
        # "score_mean_field",
        "score_true",
    ]

    METHOD_LABELS = {
        "score_lfc_abs": "|LFC|",
        # "score_mean_field": "Mean-field",
        "score_true": "True covariance",
    }

    METHOD_ORDER = [
        "|LFC|",
        # "Mean-field",
        "True covariance",
    ]

    DPI = 300

    FONTSIZE = 15

    JITTER_SD = 0.045

    RANDOM_SEED = 0

    DRAW_PAIRED_LINES = False

    DRAW_VIOLINS = True

    VIOLIN_WIDTH = 0.72

    LOG_RANK_AXIS = False

    # inject this notebook's config into the helper module so its functions
    # resolve the plot-config constants as module globals (matching the
    # original single-namespace notebook)
    import src.suppl_generanking as _M
    _M.__dict__.update({k: v for k, v in list(globals().items()) if k.isupper()})

    def add_summary_marker(
        ax,
        x_position,
        summary_value,
        sem_value=None,
        percentage=False,
    ):
        """
        Add the pooled perturbation summary marker.

        The white diamond summarizes all perturbations pooled across
        datasets. It is not an unweighted average across datasets.
        """
        summary_value = pd.to_numeric(
            pd.Series([summary_value]),
            errors="coerce",
        ).iloc[0]

        if not np.isfinite(summary_value):
            return

        if sem_value is not None:
            sem_value = pd.to_numeric(
                pd.Series([sem_value]),
                errors="coerce",
            ).iloc[0]

        if sem_value is not None and np.isfinite(sem_value):
            ax.errorbar(
                x_position,
                summary_value,
                yerr=sem_value,
                fmt="D",
                markersize=7,
                markeredgecolor="black",
                markerfacecolor="white",
                ecolor="black",
                elinewidth=1.8,
                capsize=4,
                capthick=1.5,
                zorder=10,
            )
        else:
            ax.scatter(
                [x_position],
                [summary_value],
                marker="D",
                s=55,
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
                zorder=10,
            )

        if percentage:
            summary_text = f"{100 * summary_value:.1f}%"
        else:
            summary_text = f"{summary_value:.1f}"

        ax.annotate(
            summary_text,
            xy=(x_position, summary_value),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            zorder=11,
        )

    def plot_metric_panel(
        ax,
        dataset_metric_df,
        pooled_metric_df,
        metric_column,
        ylabel,
        title,
        rng,
        percentage=False,
    ):
        positions = np.arange(len(METHOD_ORDER))

        metric_wide = (
            dataset_metric_df.pivot_table(
                index=[
                    "dataset",
                    "dataset_display",
                    "dataset_group",
                ],
                columns="method_label",
                values=metric_column,
                aggfunc="first",
                observed=False,
            )
            .reindex(columns=METHOD_ORDER)
        )

        # --------------------------------------------------------
        # Violin distributions
        # --------------------------------------------------------

        values_by_method = [
            metric_wide[label]
            .dropna()
            .to_numpy(dtype=float)
            for label in METHOD_ORDER
        ]

        if DRAW_VIOLINS:
            add_violin(
                ax=ax,
                values_by_method=values_by_method,
                positions=positions,
            )

        # --------------------------------------------------------
        # Paired lines between methods
        # --------------------------------------------------------

        if DRAW_PAIRED_LINES:
            for _, row in metric_wide.iterrows():
                values = row.to_numpy(dtype=float)
                finite = np.isfinite(values)

                if np.sum(finite) >= 2:
                    ax.plot(
                        positions[finite],
                        values[finite],
                        linewidth=0.7,
                        alpha=0.16,
                        color="black",
                        zorder=1,
                    )

        # --------------------------------------------------------
        # Dataset dots
        # --------------------------------------------------------

        for method_index, method_label in enumerate(
            METHOD_ORDER
        ):
            values = metric_wide[method_label].dropna()

            if values.empty:
                continue

            jitter = rng.normal(
                loc=0.0,
                scale=JITTER_SD,
                size=len(values),
            )

            ax.scatter(
                np.full(len(values), method_index) + jitter,
                values.to_numpy(dtype=float),
                s=42,
                alpha=0.78,
                edgecolor="black",
                linewidth=0.45,
                zorder=5,
            )

            pooled_row = pooled_metric_df[
                pooled_metric_df["method_label"] == method_label
            ]

            if not pooled_row.empty:
                summary_value = float(
                    pooled_row.iloc[0][metric_column]
                )

                # A standard error is meaningful here for the pooled
                # mean rank. For the pooled median and top-1 fraction,
                # show the pooled point without an error bar.
                if metric_column == "mean_rank":
                    sem_value = float(
                        pooled_row.iloc[0]["rank_sem"]
                    )
                else:
                    sem_value = None

                add_summary_marker(
                    ax=ax,
                    x_position=method_index,
                    summary_value=summary_value,
                    sem_value=sem_value,
                    percentage=percentage,
                )

            label_best_and_worst(
                ax=ax,
                values=values,
                x_position=method_index,
                lower_is_better=not percentage,
            )

        # --------------------------------------------------------
        # Axes
        # --------------------------------------------------------

        ax.set_xticks(positions)
        ax.set_xticklabels(
            METHOD_ORDER,
            rotation=20,
            ha="right",
        )

        ax.set_ylabel(ylabel)
        ax.set_title(
            title,
            fontsize=FONTSIZE + 1,
            fontweight="bold",
        )

        ax.set_xlim(
            -0.65,
            len(METHOD_ORDER) - 0.35,
        )

        if percentage:
            ax.set_ylim(-0.025, 1.10)

            ticks = np.linspace(0, 1, 6)
            ax.set_yticks(ticks)
            ax.set_yticklabels(
                [f"{100 * value:.0f}%" for value in ticks]
            )

        if LOG_RANK_AXIS and metric_column in {
            "mean_rank",
            "median_rank",
        }:
            ax.set_yscale("log")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if RUN_OUT is None:
        RUN_OUT = find_latest_run(
            out_root=OUT_ROOT,
            expression_cutoff=EXPRESSION_CUTOFF,
        )
    else:
        RUN_OUT = Path(RUN_OUT)

    RUN_OUT = Path(RUN_OUT)

    input_path = Path(OUTDIR) / INPUT_FILENAME

    if not input_path.exists():
        raise FileNotFoundError(
            "Could not find the saved valid-rank table:\n"
            f"{input_path}\n\n"
            "Run the rank-extraction script first."
        )

    print("=" * 80)

    print("DATASET-LEVEL RANK METRIC PLOT")

    print("=" * 80)

    print("[run]", RUN_OUT)

    print("[input]", input_path)

    rank_df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    required_columns = {
        "dataset",
        "method",
        "absolute_rank",
    }

    missing_columns = required_columns - set(rank_df.columns)

    if missing_columns:
        raise ValueError(
            f"Input table is missing columns: {sorted(missing_columns)}"
        )

    if "dataset_display" not in rank_df.columns:
        rank_df["dataset_display"] = rank_df["dataset"].astype(str)

    if "dataset_group" not in rank_df.columns:
        rank_df["dataset_group"] = "all"

    rank_df["absolute_rank"] = pd.to_numeric(
        rank_df["absolute_rank"],
        errors="coerce",
    )

    rank_df = rank_df[
        rank_df["method"].isin(METHODS)
        & np.isfinite(rank_df["absolute_rank"])
        & (rank_df["absolute_rank"] >= 1)
    ].copy()

    if rank_df.empty:
        raise RuntimeError(
            "No valid rows remained after restricting to:\n"
            f"{METHODS}"
        )

    methods_found = sorted(
        rank_df["method"].unique()
    )

    missing_methods = [
        method
        for method in METHODS
        if method not in methods_found
    ]

    if missing_methods:
        print(
            "[warning] missing requested methods:",
            missing_methods,
        )

    print(
        f"[rank rows] {len(rank_df):,}"
    )

    print(
        f"[datasets] {rank_df['dataset'].nunique():,}"
    )

    print(
        f"[methods] {methods_found}"
    )

    dataset_metrics = calculate_dataset_metrics(
        rank_df
    )

    metrics_output_path = (
        Path(OUTDIR) / "DATASET_RANK_METRICS_LFC_MEANFIELD_TRUE.csv"
    )

    dataset_metrics.to_csv(
        metrics_output_path,
        index=False,
    )

    print()

    print("[dataset-level metrics]")

    print(
        dataset_metrics[
            [
                "dataset_group",
                "dataset_display",
                "method_label",
                "n_perturbations",
                "mean_rank",
                "median_rank",
                "top1_fraction",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    pooled_perturbation_summary = (
        calculate_pooled_perturbation_metrics(
            rank_df
        )
    )

    summary_output_path = (
        Path(OUTDIR)
        / "POOLED_PERTURBATION_RANK_METRICS_LFC_MEANFIELD_TRUE.csv"
    )

    pooled_perturbation_summary.to_csv(
        summary_output_path,
        index=False,
    )

    print()

    print("[all perturbations pooled across datasets]")

    print(
        pooled_perturbation_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    plt.rcParams.update({
        "font.size": FONTSIZE,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    })

    rng = np.random.default_rng(RANDOM_SEED)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.6),
    )

    plot_metric_panel(
        ax=axes[0],
        dataset_metric_df=dataset_metrics,
        pooled_metric_df=pooled_perturbation_summary,
        metric_column="mean_rank",
        ylabel="Mean absolute rank",
        title="Mean rank",
        rng=rng,
        percentage=False,
    )

    plot_metric_panel(
        ax=axes[1],
        dataset_metric_df=dataset_metrics,
        pooled_metric_df=pooled_perturbation_summary,
        metric_column="median_rank",
        ylabel="Median absolute rank",
        title="Median rank",
        rng=rng,
        percentage=False,
    )

    plot_metric_panel(
        ax=axes[2],
        dataset_metric_df=dataset_metrics,
        pooled_metric_df=pooled_perturbation_summary,
        metric_column="top1_fraction",
        ylabel="Top-1 fraction",
        title="True target ranked first",
        rng=rng,
        percentage=True,
    )

    for panel_letter, ax in zip(
        ["A", "B", "C"],
        axes,
    ):
        ax.text(
            -0.18,
            1.06,
            panel_letter,
            transform=ax.transAxes,
            fontsize=FONTSIZE + 7,
            fontweight="bold",
            ha="left",
            va="top",
        )

    fig.suptitle(
        "True perturbation ranking across datasets",
        fontsize=FONTSIZE + 2,
        y=1.02,
    )

    fig.tight_layout()

    output_stem = (
        Path(OUTDIR)
        / "RANK_METRICS_LFC_MEANFIELD_TRUE_VIOLIN_DOT"
    )

    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    plt.show()

    print()

    print("=" * 80)

    print("SAVED")

    print("=" * 80)

    print(metrics_output_path)

    print(summary_output_path)

    print(output_stem.with_suffix(".png"))

    print(output_stem.with_suffix(".pdf"))

    print(output_stem.with_suffix(".svg"))
