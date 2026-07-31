"""Fig S19 engine -- true (precomputed) covariance heatmaps paired with low-dimensional
expression UMAPs across five Perturb-seq datasets, plus a shared-color-scale composite.
For each dataset the true full-ridge covariance Sigma_full_ridge.npy is loaded from disk
and drawn over its top-variance genes next to a 2D UMAP of cells colored by perturbation;
one entry point uses a CP10k/log1p + z-score expression UMAP, another a PFlog (NB-alpha)
normalized PCA/UMAP, and a third assembles the five-dataset composite on one covariance
color scale. Covariance is read, never recomputed, so no CIPHER forward/inverse math runs.

Helpers in notebooks/src (not part of the cipher package). Config constants are module
globals the notebook overrides via R.__dict__.update; DATA_DIR/OUTDIR injected.
"""

import os
import gc
import re
import hashlib
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# INJECTED PLACEHOLDERS (overwritten by the notebook config cell)
# ============================================================

DATA_DIR = None
SUPPL = None
OUTDIR = None

# Cross-call state: figS19_true_cov_umap() records where it wrote per-dataset outputs so
# figS19_composite_shared_scale() can reconstruct its records from disk.
TRUE_COV_OUT_ROOT = None


# ============================================================
# SHARED CONFIG (consistent across the CP10k and PFlog variants)
# ============================================================

EXPRESSION_THRESHOLD = 1.0
RANDOM_SEED = 0
DPI = 300

EXPRESSION_LAYER = None
TRY_ADATA_RAW = True
H5AD_PATH_OVERRIDES = {}

N_UMAP_GENES = 1000
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.25
UMAP_POINT_SIZE = 5
UMAP_POINT_ALPHA = 0.70

N_COVARIANCE_GENES = 120
COVARIANCE_COLOR_QUANTILE = 0.99
MAX_COVARIANCE_TICK_LABELS = 14

TRUE_SIGMA_CANDIDATES = [
    "Sigma_full_ridge.npy",
    "Sigma_true_ridge.npy",
    "Sigma_full.npy",
    "Sigma_true.npy",
]

PERTURBATION_COLUMN_CANDIDATES = [
    "perturbation",
    "perturbation_name",
    "perturbation_id",
    "condition",
    "condition_name",
    "condition_ID",
    "target_gene",
    "target",
    "gene",
    "guide",
    "guide_id",
    "guide_identity",
    "sgRNA",
    "sgrna",
    "gRNA",
    "grna",
    "covariate",
]

CONTROL_PATTERNS = [
    "control",
    "ctrl",
    "non-target",
    "non_target",
    "nontarget",
    "non targeting",
    "non-targeting",
    "ntc",
    "negative control",
    "vehicle",
    "untreated",
    "mock",
    "scramble",
    "scrambled",
]


def figS19_true_cov_umap():
    """Cell-15 pipeline: per-dataset true-covariance heatmap + CP10k/log1p expression UMAP.

    Nested helper defs and the driver body are the source cell moved here verbatim (indented);
    the collision-prone per-variant constants and the DATA_DIR/SUPPL/OUTDIR-derived paths are
    local, while the shared constants are read from the module globals above.
    """
    global TRUE_COV_OUT_ROOT
    if DATA_DIR is None or SUPPL is None or OUTDIR is None:
        raise RuntimeError("DATA_DIR, SUPPL, and OUTDIR must be injected by the notebook.")

    PRECOMPUTE_ROOT = Path(SUPPL) / "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED"
    H5AD_SEARCH_ROOTS = [Path(SUPPL), Path(DATA_DIR)]
    OUT_ROOT = Path(OUTDIR) / "true_cov_umap"
    TRUE_COV_OUT_ROOT = OUT_ROOT

    # Per-variant tunables (cell 15: CP10k/log1p + z-score expression UMAP)
    DATASET_QUERIES = [
        "TianKampmann2021_CRISPRa",
        "NormanWeissman2019_filtered",
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "XAtlas2025_HEK293T_filtered",
    ]
    N_TOP_PERTURBATIONS = 5
    MAX_UMAP_CELLS = 25_000
    MAX_CONTROL_CELLS = 20_000
    N_PCS = 40

    def threshold_to_tag(value):
        return str(value).replace(".", "p")


    def decode_string_array(values):
        return np.asarray(
            [
                value.decode("utf-8")
                if isinstance(value, bytes)
                else str(value)
                for value in np.asarray(values, dtype=object)
            ],
            dtype=object,
        )


    def safe_filename(value):
        return re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(value),
        ).strip("_")


    def dataset_name_from_folder(folder):
        return re.sub(
            r"__mean_ge_[^/]+$",
            "",
            Path(folder).name,
        )


    def normalized_match_string(value):
        return re.sub(
            r"[^a-z0-9]+",
            "",
            str(value).lower(),
        )


    def to_dense(matrix):
        if sparse.issparse(matrix):
            return matrix.toarray()

        return np.asarray(matrix)


    def even_tick_indices(n_values, max_ticks):
        if n_values <= max_ticks:
            return np.arange(
                n_values,
                dtype=int,
            )

        return np.unique(
            np.linspace(
                0,
                n_values - 1,
                max_ticks,
            )
            .round()
            .astype(int)
        )


    def series_to_string_array(
        series,
        missing_value="unknown",
    ):
        """
        Safely convert a pandas Series to strings.

        This explicitly converts categorical data to ordinary Python
        objects before replacing missing values. It therefore avoids:

            TypeError:
            Cannot setitem on a Categorical with a new category
        """

        values = series.astype(
            object
        ).to_numpy(
            copy=True
        )

        missing = pd.isna(
            values
        )

        values[missing] = (
            missing_value
        )

        return np.asarray(
            [
                str(value)
                for value in values
            ],
            dtype=object,
        )


    def is_control_label(value):
        value = str(
            value
        ).strip().lower()

        return any(
            pattern in value
            for pattern in CONTROL_PATTERNS
        )


    def collapse_control_labels(labels):
        labels = np.asarray(
            labels,
            dtype=object,
        )

        return np.asarray(
            [
                "Control"
                if is_control_label(label)
                else str(label)
                for label in labels
            ],
            dtype=object,
        )


    # ============================================================
    # PRECOMPUTED DATASET DISCOVERY
    # ============================================================

    def find_precomputed_dataset_folders(
        root,
        expression_threshold,
    ):
        root = Path(root)

        tag = threshold_to_tag(
            expression_threshold
        )

        return sorted(
            folder
            for folder in root.glob(
                f"*__mean_ge_{tag}"
            )
            if folder.is_dir()
        )


    def filter_dataset_folders(
        folders,
        dataset_queries,
    ):
        if dataset_queries is None:
            return folders

        queries = [
            str(query).lower()
            for query in dataset_queries
        ]

        return [
            folder
            for folder in folders
            if any(
                query in folder.name.lower()
                for query in queries
            )
        ]


    def find_true_sigma_path(
        sigma_directory,
    ):
        sigma_directory = Path(
            sigma_directory
        )

        for filename in TRUE_SIGMA_CANDIDATES:
            candidate = (
                sigma_directory
                / filename
            )

            if candidate.exists():
                return candidate

        fallback_patterns = [
            "*Sigma*full*.npy",
            "*Sigma*true*.npy",
            "*sigma*full*.npy",
            "*sigma*true*.npy",
        ]

        hits = []

        for pattern in fallback_patterns:
            hits.extend(
                sorted(
                    sigma_directory.glob(
                        pattern
                    )
                )
            )

        hits = [
            path
            for path in hits
            if not any(
                excluded
                in path.name.lower()
                for excluded in [
                    "meanfield",
                    "mean_field",
                    "shuffle",
                    "shuffled",
                    "_mf",
                ]
            )
        ]

        if hits:
            return hits[0]

        available = sorted(
            path.name
            for path
            in sigma_directory.glob(
                "*.npy"
            )
        )

        raise FileNotFoundError(
            "Could not locate a true/full covariance matrix in:\n"
            f"{sigma_directory}\n\n"
            f"Available covariance files:\n{available}"
        )


    def load_precomputed_information(
        dataset_folder,
    ):
        dataset_folder = Path(
            dataset_folder
        )

        genes_path = (
            dataset_folder
            / "genes.npy"
        )

        perturbations_path = (
            dataset_folder
            / "perturbations.npy"
        )

        sigma_directory = (
            dataset_folder
            / "sigmas"
        )

        required = [
            genes_path,
            perturbations_path,
            sigma_directory,
        ]

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required precomputed files:\n"
                + "\n".join(missing)
            )

        true_sigma_path = (
            find_true_sigma_path(
                sigma_directory
            )
        )

        genes = decode_string_array(
            np.load(
                genes_path,
                allow_pickle=True,
            )
        )

        perturbations = decode_string_array(
            np.load(
                perturbations_path,
                allow_pickle=True,
            )
        )

        Sigma_true = np.load(
            true_sigma_path,
            mmap_mode="r",
        )

        expected_shape = (
            len(genes),
            len(genes),
        )

        if Sigma_true.shape != expected_shape:
            raise ValueError(
                f"True covariance shape is {Sigma_true.shape}; "
                f"expected {expected_shape}."
            )

        return {
            "genes": genes,
            "perturbations": perturbations,
            "Sigma_true": Sigma_true,
            "true_sigma_path": true_sigma_path,
        }


    # ============================================================
    # H5AD DISCOVERY
    # ============================================================

    def index_h5ad_files(
        search_roots,
    ):
        indexed = {}

        for root in search_roots:
            root = Path(root)

            if not root.exists():
                continue

            for path in root.rglob(
                "*.h5ad"
            ):
                try:
                    key = str(
                        path.resolve()
                    )
                except Exception:
                    key = str(path)

                indexed[key] = path

        return sorted(
            indexed.values(),
            key=lambda path: str(path),
        )


    def find_matching_h5ad(
        dataset_folder,
        dataset_name,
        indexed_h5ad_files,
    ):
        for key in [
            dataset_folder.name,
            dataset_name,
        ]:
            if key not in H5AD_PATH_OVERRIDES:
                continue

            override = Path(
                H5AD_PATH_OVERRIDES[key]
            )

            if not override.exists():
                raise FileNotFoundError(
                    "Configured H5AD override does not exist:\n"
                    f"{override}"
                )

            return override

        target = normalized_match_string(
            dataset_name
        )

        candidates = []

        for path in indexed_h5ad_files:
            stem = normalized_match_string(
                path.stem
            )

            exact = (
                stem == target
            )

            target_inside_stem = (
                target in stem
            )

            stem_inside_target = (
                stem in target
                and len(stem)
                >= 0.80 * len(target)
            )

            if not (
                exact
                or target_inside_stem
                or stem_inside_target
            ):
                continue

            candidates.append(
                {
                    "path": path,
                    "exact": exact,
                    "target_inside_stem": (
                        target_inside_stem
                    ),
                    "length_difference": abs(
                        len(stem)
                        - len(target)
                    ),
                }
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda row: (
                not row["exact"],
                not row[
                    "target_inside_stem"
                ],
                row[
                    "length_difference"
                ],
                len(
                    str(row["path"])
                ),
            )
        )

        if len(candidates) > 1:
            print(
                f"[h5ad] Multiple matches for {dataset_name}:"
            )

            for row in candidates[:5]:
                print(
                    f"        {os.path.basename(row['path'])}"
                )

            print(
                f"[h5ad] Selected: "
                f"{os.path.basename(candidates[0]['path'])}"
            )

        return candidates[0][
            "path"
        ]


    # ============================================================
    # EXPRESSION SOURCE
    # ============================================================

    def expression_source_from_adata(
        adata,
        requested_layer,
    ):
        if requested_layer is not None:
            if requested_layer in adata.layers:
                return {
                    "matrix": adata.layers[
                        requested_layer
                    ],
                    "var_names": np.asarray(
                        adata.var_names,
                        dtype=object,
                    ),
                    "name": (
                        f"layers[{requested_layer!r}]"
                    ),
                }

            warnings.warn(
                f"Layer {requested_layer!r} was not found. "
                f"Available layers: {list(adata.layers.keys())}. "
                "Using adata.X."
            )

        return {
            "matrix": adata.X,
            "var_names": np.asarray(
                adata.var_names,
                dtype=object,
            ),
            "name": "X",
        }


    def choose_best_gene_source(
        adata,
        precomputed_genes,
    ):
        precomputed_gene_set = set(
            map(
                str,
                precomputed_genes,
            )
        )

        primary = (
            expression_source_from_adata(
                adata,
                requested_layer=(
                    EXPRESSION_LAYER
                ),
            )
        )

        primary["overlap"] = sum(
            str(gene)
            in precomputed_gene_set
            for gene
            in primary["var_names"]
        )

        best = primary

        if (
            TRY_ADATA_RAW
            and adata.raw is not None
        ):
            raw_names = np.asarray(
                adata.raw.var_names,
                dtype=object,
            )

            raw_overlap = sum(
                str(gene)
                in precomputed_gene_set
                for gene in raw_names
            )

            if raw_overlap > best[
                "overlap"
            ]:
                best = {
                    "matrix": adata.raw.X,
                    "var_names": raw_names,
                    "name": "raw.X",
                    "overlap": raw_overlap,
                }

        return best


    # ============================================================
    # PERTURBATION COLUMN DETECTION
    # ============================================================

    def find_perturbation_column(
        obs,
    ):
        for column in (
            PERTURBATION_COLUMN_CANDIDATES
        ):
            if column in obs.columns:
                return column

        possible_columns = []

        for column in obs.columns:
            values = obs[column]

            string_like = (
                pd.api.types.is_object_dtype(
                    values
                )
                or pd.api.types.is_string_dtype(
                    values
                )
                or isinstance(
                    values.dtype,
                    pd.CategoricalDtype,
                )
            )

            if not string_like:
                continue

            n_unique = values.nunique(
                dropna=True
            )

            if not (
                2
                <= n_unique
                <= max(
                    5000,
                    int(
                        0.75
                        * len(obs)
                    ),
                )
            ):
                continue

            safe_strings = (
                series_to_string_array(
                    values
                )
            )

            possible_columns.append(
                {
                    "column": column,
                    "n_unique": n_unique,
                    "contains_control": any(
                        is_control_label(
                            value
                        )
                        for value
                        in safe_strings
                    ),
                }
            )

        if not possible_columns:
            return None

        possible_columns.sort(
            key=lambda row: (
                not row[
                    "contains_control"
                ],
                row["n_unique"],
            )
        )

        return possible_columns[0][
            "column"
        ]


    # ============================================================
    # CAPPED PROPORTIONAL SAMPLING
    # ============================================================

    def capped_proportional_allocation(
        available_counts,
        total_cap,
        control_cap,
    ):
        """
        Allocate an integer cell budget approximately in proportion to
        original category abundance, subject to total and control caps.
        """

        available_counts = (
            pd.Series(
                available_counts,
                dtype="int64",
            )
            .clip(lower=0)
            .astype(int)
        )

        capacities = (
            available_counts.copy()
        )

        if "Control" in capacities.index:
            capacities.loc[
                "Control"
            ] = min(
                int(
                    capacities.loc[
                        "Control"
                    ]
                ),
                int(control_cap),
            )

        target_total = min(
            int(total_cap),
            int(
                capacities.sum()
            ),
        )

        continuous_allocation = pd.Series(
            0.0,
            index=available_counts.index,
        )

        active = list(
            available_counts.index
        )

        remaining = float(
            target_total
        )

        while (
            active
            and remaining > 1e-10
        ):
            residual_capacity = (
                capacities.loc[active]
                - continuous_allocation.loc[
                    active
                ]
            )

            active = [
                category
                for category in active
                if residual_capacity.loc[
                    category
                ] > 1e-10
            ]

            if not active:
                break

            weights = (
                available_counts.loc[
                    active
                ]
                .astype(float)
            )

            if weights.sum() <= 0:
                weights[:] = 1.0

            proposal = (
                remaining
                * weights
                / weights.sum()
            )

            residual_capacity = (
                capacities.loc[active]
                - continuous_allocation.loc[
                    active
                ]
            )

            newly_capped = [
                category
                for category in active
                if proposal.loc[
                    category
                ]
                >= residual_capacity.loc[
                    category
                ]
                - 1e-10
            ]

            if not newly_capped:
                continuous_allocation.loc[
                    active
                ] += proposal

                remaining = 0.0
                break

            for category in newly_capped:
                addition = float(
                    residual_capacity.loc[
                        category
                    ]
                )

                continuous_allocation.loc[
                    category
                ] += addition

                remaining -= addition

            active = [
                category
                for category in active
                if category
                not in newly_capped
            ]

        integer_allocation = (
            np.floor(
                continuous_allocation
                + 1e-12
            )
            .astype(int)
        )

        cells_remaining = (
            target_total
            - int(
                integer_allocation.sum()
            )
        )

        fractional_parts = (
            continuous_allocation
            - integer_allocation
        )

        while cells_remaining > 0:
            eligible = [
                category
                for category
                in integer_allocation.index
                if integer_allocation.loc[
                    category
                ]
                < capacities.loc[
                    category
                ]
            ]

            if not eligible:
                break

            priority = pd.DataFrame(
                {
                    "fractional_part": (
                        fractional_parts.loc[
                            eligible
                        ]
                    ),
                    "abundance": (
                        available_counts.loc[
                            eligible
                        ]
                    ),
                }
            ).sort_values(
                [
                    "fractional_part",
                    "abundance",
                ],
                ascending=False,
            )

            chosen = priority.index[0]

            integer_allocation.loc[
                chosen
            ] += 1

            fractional_parts.loc[
                chosen
            ] = 0.0

            cells_remaining -= 1

        if int(
            integer_allocation.sum()
        ) > int(total_cap):
            raise RuntimeError(
                "Allocation exceeded the total-cell cap."
            )

        if (
            "Control"
            in integer_allocation.index
            and int(
                integer_allocation.loc[
                    "Control"
                ]
            )
            > int(control_cap)
        ):
            raise RuntimeError(
                "Allocation exceeded the control-cell cap."
            )

        return integer_allocation


    def select_top_perturbations_and_controls(
        raw_labels,
        n_top_perturbations,
        max_total_cells,
        max_control_cells,
        seed,
    ):
        rng = np.random.default_rng(
            seed
        )

        raw_labels = np.asarray(
            raw_labels,
            dtype=object,
        )

        collapsed_labels = (
            collapse_control_labels(
                raw_labels
            )
        )

        all_counts = pd.Series(
            collapsed_labels,
            dtype="object",
        ).value_counts()

        perturbation_counts = (
            all_counts.drop(
                labels=["Control"],
                errors="ignore",
            )
            .sort_values(
                ascending=False
            )
        )

        if len(
            perturbation_counts
        ) == 0:
            raise ValueError(
                "No non-control perturbations were detected."
            )

        top_perturbations = (
            perturbation_counts
            .head(
                n_top_perturbations
            )
            .index
            .astype(str)
            .tolist()
        )

        retained_categories = []

        if "Control" in all_counts.index:
            retained_categories.append(
                "Control"
            )

        retained_categories.extend(
            top_perturbations
        )

        retained_mask = np.isin(
            collapsed_labels,
            retained_categories,
        )

        retained_indices = np.flatnonzero(
            retained_mask
        )

        retained_labels = collapsed_labels[
            retained_indices
        ]

        available_counts = (
            pd.Series(
                retained_labels,
                dtype="object",
            )
            .value_counts()
            .reindex(
                retained_categories,
                fill_value=0,
            )
            .astype(int)
        )

        allocation = (
            capped_proportional_allocation(
                available_counts=(
                    available_counts
                ),
                total_cap=(
                    max_total_cells
                ),
                control_cap=(
                    max_control_cells
                ),
            )
        )

        selected_parts = []

        for category in retained_categories:
            category_indices = (
                retained_indices[
                    retained_labels
                    == category
                ]
            )

            n_take = int(
                allocation.loc[
                    category
                ]
            )

            if n_take <= 0:
                continue

            if n_take >= len(
                category_indices
            ):
                chosen = (
                    category_indices
                )

            else:
                chosen = rng.choice(
                    category_indices,
                    size=n_take,
                    replace=False,
                )

            selected_parts.append(
                np.asarray(
                    chosen,
                    dtype=int,
                )
            )

        if not selected_parts:
            raise RuntimeError(
                "The UMAP sampler selected no cells."
            )

        selected_indices = np.sort(
            np.concatenate(
                selected_parts
            )
        )

        selected_labels = collapsed_labels[
            selected_indices
        ]

        selected_counts = (
            pd.Series(
                selected_labels,
                dtype="object",
            )
            .value_counts()
            .reindex(
                retained_categories,
                fill_value=0,
            )
            .astype(int)
        )

        n_selected_controls = int(
            selected_counts.get(
                "Control",
                0,
            )
        )

        if len(
            selected_indices
        ) > max_total_cells:
            raise RuntimeError(
                "Selected more than 10,000 UMAP cells."
            )

        if (
            n_selected_controls
            > max_control_cells
        ):
            raise RuntimeError(
                "Selected more than 5,000 control cells."
            )

        summary = pd.DataFrame(
            {
                "category": (
                    retained_categories
                ),
                "is_control": [
                    category == "Control"
                    for category
                    in retained_categories
                ],
                "available_cells": [
                    int(
                        available_counts.loc[
                            category
                        ]
                    )
                    for category
                    in retained_categories
                ],
                "selected_cells": [
                    int(
                        selected_counts.loc[
                            category
                        ]
                    )
                    for category
                    in retained_categories
                ],
            }
        )

        available_total = max(
            int(
                summary[
                    "available_cells"
                ].sum()
            ),
            1,
        )

        selected_total = max(
            int(
                summary[
                    "selected_cells"
                ].sum()
            ),
            1,
        )

        summary[
            "available_fraction"
        ] = (
            summary[
                "available_cells"
            ]
            / available_total
        )

        summary[
            "selected_fraction"
        ] = (
            summary[
                "selected_cells"
            ]
            / selected_total
        )

        summary[
            "fraction_difference"
        ] = (
            summary[
                "selected_fraction"
            ]
            - summary[
                "available_fraction"
            ]
        )

        print(
            "\n[UMAP retained categories]"
        )

        print(
            summary.to_string(
                index=False,
                formatters={
                    "available_fraction": (
                        lambda value:
                        f"{value:.4f}"
                    ),
                    "selected_fraction": (
                        lambda value:
                        f"{value:.4f}"
                    ),
                    "fraction_difference": (
                        lambda value:
                        f"{value:+.4f}"
                    ),
                },
            )
        )

        print(
            f"\n[UMAP sample] "
            f"{len(selected_indices):,} cells total; "
            f"{n_selected_controls:,} controls"
        )

        return {
            "selected_indices": (
                selected_indices
            ),
            "selected_labels": (
                selected_labels
            ),
            "top_perturbations": (
                top_perturbations
            ),
            "summary": summary,
            "collapsed_labels": (
                collapsed_labels
            ),
        }


    # ============================================================
    # GENE SELECTION
    # ============================================================

    def select_overlapping_high_variance_genes(
        precomputed_genes,
        adata_var_names,
        Sigma_true,
        n_genes,
    ):
        precomputed_genes = np.asarray(
            precomputed_genes,
            dtype=object,
        )

        adata_var_names = np.asarray(
            adata_var_names,
            dtype=object,
        )

        # Preserve the first occurrence when var names are duplicated.
        adata_gene_to_index = {}

        for index, gene in enumerate(
            adata_var_names
        ):
            gene = str(gene)

            if gene not in (
                adata_gene_to_index
            ):
                adata_gene_to_index[
                    gene
                ] = index

        covariance_variance = np.asarray(
            np.diag(
                Sigma_true
            ),
            dtype=float,
        )

        records = []

        for precomputed_index, gene in enumerate(
            precomputed_genes
        ):
            gene = str(gene)

            if gene not in (
                adata_gene_to_index
            ):
                continue

            variance = covariance_variance[
                precomputed_index
            ]

            if not np.isfinite(
                variance
            ):
                continue

            records.append(
                {
                    "precomputed_index": (
                        precomputed_index
                    ),
                    "adata_index": (
                        adata_gene_to_index[
                            gene
                        ]
                    ),
                    "gene": gene,
                    "covariance_variance": float(
                        variance
                    ),
                }
            )

        if len(records) < 3:
            raise ValueError(
                "Fewer than three genes overlap between "
                "the covariance and expression matrices."
            )

        records.sort(
            key=lambda row: (
                row[
                    "covariance_variance"
                ]
            ),
            reverse=True,
        )

        records = records[
            :min(
                n_genes,
                len(records),
            )
        ]

        # Backed array indexing is safest with increasing indices.
        records.sort(
            key=lambda row: (
                row["adata_index"]
            )
        )

        return {
            "precomputed_indices": np.asarray(
                [
                    row[
                        "precomputed_index"
                    ]
                    for row in records
                ],
                dtype=int,
            ),
            "adata_indices": np.asarray(
                [
                    row["adata_index"]
                    for row in records
                ],
                dtype=int,
            ),
            "genes": np.asarray(
                [
                    row["gene"]
                    for row in records
                ],
                dtype=object,
            ),
            "covariance_variances": np.asarray(
                [
                    row[
                        "covariance_variance"
                    ]
                    for row in records
                ],
                dtype=float,
            ),
        }


    def select_covariance_heatmap_genes(
        precomputed_genes,
        Sigma_true,
        n_genes,
    ):
        diagonal = np.asarray(
            np.diag(
                Sigma_true
            ),
            dtype=float,
        )

        valid = np.flatnonzero(
            np.isfinite(
                diagonal
            )
        )

        if len(valid) == 0:
            raise ValueError(
                "The covariance diagonal contains no finite values."
            )

        n_keep = min(
            n_genes,
            len(valid),
        )

        selected = valid[
            np.argsort(
                diagonal[valid]
            )[-n_keep:]
        ]

        selected = selected[
            np.argsort(
                diagonal[selected]
            )[::-1]
        ]

        return {
            "indices": (
                selected.astype(int)
            ),
            "genes": np.asarray(
                precomputed_genes[
                    selected
                ],
                dtype=object,
            ),
            "variances": (
                diagonal[selected]
            ),
        }


    # ============================================================
    # EXPRESSION PREPROCESSING
    # ============================================================

    def matrix_looks_like_counts(
        X,
    ):
        X = np.asarray(
            X,
            dtype=float,
        )

        finite = X[
            np.isfinite(X)
        ]

        if finite.size == 0:
            return False

        if finite.size > 100_000:
            rng = np.random.default_rng(
                RANDOM_SEED
            )

            finite = rng.choice(
                finite,
                size=100_000,
                replace=False,
            )

        if np.min(finite) < 0:
            return False

        integer_fraction = np.mean(
            np.abs(
                finite
                - np.round(finite)
            )
            < 1e-6
        )

        return bool(
            integer_fraction
            >= 0.95
        )


    def preprocess_expression(
        X,
    ):
        X = np.asarray(
            X,
            dtype=np.float64,
        )

        counts_like = (
            matrix_looks_like_counts(
                X
            )
        )

        if counts_like:
            library_size = np.sum(
                X,
                axis=1,
            )

            safe_library_size = np.maximum(
                library_size,
                1e-12,
            )

            normalized = (
                X
                / safe_library_size[
                    :,
                    None,
                ]
                * 1e4
            )

            X_processed = np.log1p(
                normalized
            )

            normalization_name = (
                "library-size normalization + log1p"
            )

        else:
            X_processed = X.copy()

            normalization_name = (
                "existing expression values"
            )

        X_processed[
            ~np.isfinite(
                X_processed
            )
        ] = 0.0

        scaler = StandardScaler(
            with_mean=True,
            with_std=True,
        )

        X_scaled = scaler.fit_transform(
            X_processed
        )

        X_scaled[
            ~np.isfinite(
                X_scaled
            )
        ] = 0.0

        return {
            "X_scaled": X_scaled,
            "normalization_name": (
                normalization_name
            ),
            "counts_like": (
                counts_like
            ),
        }


    # ============================================================
    # PCA + UMAP
    # ============================================================

    def calculate_umap(
        X,
    ):
        X = np.asarray(
            X,
            dtype=np.float64,
        )

        n_components = min(
            N_PCS,
            X.shape[0] - 1,
            X.shape[1],
        )

        if n_components < 2:
            raise ValueError(
                "Cannot calculate an embedding from "
                f"matrix shape {X.shape}."
            )

        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=RANDOM_SEED,
        )

        X_pca = pca.fit_transform(
            X
        )

        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(
                    UMAP_N_NEIGHBORS,
                    max(
                        2,
                        X_pca.shape[0] - 1,
                    ),
                ),
                min_dist=UMAP_MIN_DIST,
                metric="euclidean",
                random_state=RANDOM_SEED,
            )

            embedding = (
                reducer.fit_transform(
                    X_pca
                )
            )

            method = "UMAP"

        except ImportError:
            warnings.warn(
                "umap-learn is unavailable. Using the "
                "first two principal components instead."
            )

            embedding = X_pca[
                :,
                :2,
            ]

            method = "PCA fallback"

        return {
            "embedding": embedding,
            "method": method,
            "pca_variance_fraction": float(
                np.sum(
                    pca.explained_variance_ratio_
                )
            ),
        }


    # ============================================================
    # COVARIANCE PREPARATION
    # ============================================================

    def prepare_covariance_heatmap(
        Sigma_true,
        covariance_selection,
    ):
        indices = covariance_selection[
            "indices"
        ]

        covariance = np.asarray(
            Sigma_true[
                np.ix_(
                    indices,
                    indices,
                )
            ],
            dtype=np.float64,
        )

        covariance = 0.5 * (
            covariance
            + covariance.T
        )

        off_diagonal_mask = ~np.eye(
            covariance.shape[0],
            dtype=bool,
        )

        display_values = covariance[
            off_diagonal_mask
        ]

        display_values = display_values[
            np.isfinite(
                display_values
            )
        ]

        if (
            display_values.size == 0
            or np.all(
                display_values == 0
            )
        ):
            display_values = covariance[
                np.isfinite(
                    covariance
                )
            ]

        color_limit = float(
            np.quantile(
                np.abs(
                    display_values
                ),
                COVARIANCE_COLOR_QUANTILE,
            )
        )

        if (
            not np.isfinite(
                color_limit
            )
            or color_limit <= 0
        ):
            color_limit = float(
                np.nanmax(
                    np.abs(
                        covariance
                    )
                )
            )

        if (
            not np.isfinite(
                color_limit
            )
            or color_limit <= 0
        ):
            color_limit = 1.0

        return {
            "matrix": covariance,
            "color_limit": color_limit,
        }


    # ============================================================
    # UMAP COLORING
    # ============================================================

    def make_category_color_map(
        labels,
    ):
        labels = np.asarray(
            labels,
            dtype=object,
        )

        counts = pd.Series(
            labels,
            dtype="object",
        ).value_counts()

        categories = (
            counts.index.tolist()
        )

        non_control_categories = [
            category
            for category in categories
            if category != "Control"
        ]

        color_map = {}

        if "Control" in categories:
            color_map[
                "Control"
            ] = (
                0.18,
                0.18,
                0.18,
                1.0,
            )

        colors = plt.cm.tab10(
            np.linspace(
                0.0,
                0.8,
                max(
                    len(
                        non_control_categories
                    ),
                    1,
                ),
            )
        )

        for category, color in zip(
            non_control_categories,
            colors,
        ):
            color_map[
                category
            ] = color

        return color_map, counts


    def draw_perturbation_umap(
        ax,
        embedding,
        labels,
        embedding_method,
    ):
        labels = np.asarray(
            labels,
            dtype=object,
        )

        color_map, counts = (
            make_category_color_map(
                labels
            )
        )

        # Draw large groups first so smaller groups stay visible.
        plot_order = (
            counts
            .sort_values(
                ascending=False
            )
            .index
            .tolist()
        )

        for category in plot_order:
            mask = (
                labels == category
            )

            ax.scatter(
                embedding[
                    mask,
                    0,
                ],
                embedding[
                    mask,
                    1,
                ],
                s=UMAP_POINT_SIZE,
                alpha=UMAP_POINT_ALPHA,
                linewidths=0,
                color=color_map[
                    category
                ],
                rasterized=True,
            )

        legend_order = []

        if "Control" in counts.index:
            legend_order.append(
                "Control"
            )

        legend_order.extend(
            [
                category
                for category
                in counts.index
                if category != "Control"
            ]
        )

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=6,
                markerfacecolor=(
                    color_map[
                        category
                    ]
                ),
                markeredgecolor="none",
                label=(
                    f"{category} "
                    f"(n={int(counts[category]):,})"
                ),
            )
            for category in legend_order
        ]

        ax.legend(
            handles=handles,
            title="Cell category",
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            bbox_to_anchor=(
                1.02,
                1.0,
            ),
            loc="upper left",
            borderaxespad=0.0,
        )

        # No text labels are placed inside the UMAP.
        ax.set_title(
            f"Cell expression {embedding_method}\n"
            "Control + five most abundant perturbations"
        )

        ax.set_xlabel(
            f"{embedding_method} 1"
        )

        ax.set_ylabel(
            f"{embedding_method} 2"
        )

        ax.set_xticks([])
        ax.set_yticks([])


    # ============================================================
    # PROCESS ONE DATASET
    # ============================================================

    def process_one_dataset(
        dataset_folder,
        indexed_h5ad_files,
    ):
        dataset_folder = Path(
            dataset_folder
        )

        dataset_name = (
            dataset_name_from_folder(
                dataset_folder
            )
        )

        print(
            "\n"
            + "=" * 110
        )

        print(
            f"DATASET: {dataset_name}"
        )

        print(
            f"FOLDER:  {os.path.basename(dataset_folder)}"
        )

        print(
            "=" * 110
        )

        h5ad_path = find_matching_h5ad(
            dataset_folder=dataset_folder,
            dataset_name=dataset_name,
            indexed_h5ad_files=(
                indexed_h5ad_files
            ),
        )

        if h5ad_path is None:
            raise FileNotFoundError(
                "No matching h5ad file was found for "
                f"{dataset_name}.\n"
                "Add its location to H5AD_SEARCH_ROOTS "
                "or H5AD_PATH_OVERRIDES."
            )

        print(
            f"[h5ad] {os.path.basename(h5ad_path)}"
        )

        precomputed = (
            load_precomputed_information(
                dataset_folder
            )
        )

        precomputed_genes = (
            precomputed["genes"]
        )

        Sigma_true = (
            precomputed["Sigma_true"]
        )

        output_directory = (
            OUT_ROOT
            / safe_filename(
                dataset_name
            )
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        adata = None

        try:
            adata = ad.read_h5ad(
                h5ad_path,
                backed="r",
            )

            print(
                f"[adata] shape={adata.shape}; "
                f"layers={list(adata.layers.keys())}"
            )

            expression_source = (
                choose_best_gene_source(
                    adata=adata,
                    precomputed_genes=(
                        precomputed_genes
                    ),
                )
            )

            print(
                f"[expression] "
                f"source={expression_source['name']}; "
                f"gene overlap="
                f"{expression_source['overlap']:,}"
            )

            perturbation_column = (
                find_perturbation_column(
                    adata.obs
                )
            )

            if perturbation_column is None:
                raise ValueError(
                    "Could not identify a perturbation "
                    "column in adata.obs.\n"
                    f"Available columns:\n"
                    f"{list(adata.obs.columns)}"
                )

            print(
                f"[metadata] perturbation column="
                f"{perturbation_column}"
            )

            # ====================================================
            # CATEGORICAL-SAFE LABEL EXTRACTION
            #
            # Do not use:
            #
            #   series.fillna("unknown")
            #
            # because the Series may be categorical.
            # ====================================================

            all_raw_labels = (
                series_to_string_array(
                    adata.obs[
                        perturbation_column
                    ],
                    missing_value="unknown",
                )
            )

            sampling = (
                select_top_perturbations_and_controls(
                    raw_labels=(
                        all_raw_labels
                    ),
                    n_top_perturbations=(
                        N_TOP_PERTURBATIONS
                    ),
                    max_total_cells=(
                        MAX_UMAP_CELLS
                    ),
                    max_control_cells=(
                        MAX_CONTROL_CELLS
                    ),
                    seed=RANDOM_SEED,
                )
            )

            selected_cell_indices = (
                sampling[
                    "selected_indices"
                ]
            )

            selected_plot_labels = (
                sampling[
                    "selected_labels"
                ]
            )

            top_perturbations = (
                sampling[
                    "top_perturbations"
                ]
            )

            sampling_summary = (
                sampling["summary"]
            )

            all_collapsed_labels = (
                sampling[
                    "collapsed_labels"
                ]
            )

            selected_raw_labels = (
                all_raw_labels[
                    selected_cell_indices
                ]
            )

            selected_cell_names = np.asarray(
                adata.obs_names[
                    selected_cell_indices
                ],
                dtype=object,
            )

            n_controls_total = int(
                np.sum(
                    all_collapsed_labels
                    == "Control"
                )
            )

            n_controls_selected = int(
                np.sum(
                    selected_plot_labels
                    == "Control"
                )
            )

            print(
                "[metadata] retained perturbations: "
                + ", ".join(
                    top_perturbations
                )
            )

            print(
                f"[sample] selected "
                f"{len(selected_cell_indices):,}/"
                f"{adata.n_obs:,} total cells; "
                f"controls={n_controls_selected:,}"
            )

            sampling_summary_path = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__umap_category_sampling.csv"
                )
            )

            sampling_summary.to_csv(
                sampling_summary_path,
                index=False,
            )

            # ----------------------------------------------------
            # SELECT UMAP GENES
            # ----------------------------------------------------

            umap_gene_selection = (
                select_overlapping_high_variance_genes(
                    precomputed_genes=(
                        precomputed_genes
                    ),
                    adata_var_names=(
                        expression_source[
                            "var_names"
                        ]
                    ),
                    Sigma_true=(
                        Sigma_true
                    ),
                    n_genes=(
                        N_UMAP_GENES
                    ),
                )
            )

            print(
                f"[genes] UMAP uses "
                f"{len(umap_gene_selection['genes']):,} genes"
            )

            # Row selection first, followed by column selection,
            # is safer for backed HDF5 matrices.
            expression_rows = (
                expression_source[
                    "matrix"
                ][
                    selected_cell_indices,
                    :,
                ]
            )

            expression_subset = (
                expression_rows[
                    :,
                    umap_gene_selection[
                        "adata_indices"
                    ],
                ]
            )

            X = to_dense(
                expression_subset
            ).astype(
                np.float64,
                copy=False,
            )

            del expression_rows
            del expression_subset

            # ----------------------------------------------------
            # PCA + UMAP
            # ----------------------------------------------------

            preprocessing = (
                preprocess_expression(
                    X
                )
            )

            embedding_result = (
                calculate_umap(
                    preprocessing[
                        "X_scaled"
                    ]
                )
            )

            embedding = (
                embedding_result[
                    "embedding"
                ]
            )

            normalization_name = (
                preprocessing[
                    "normalization_name"
                ]
            )

            print(
                f"[embedding] "
                f"method={embedding_result['method']}; "
                f"PCA variance captured="
                f"{embedding_result['pca_variance_fraction']:.3f}; "
                f"normalization={normalization_name}"
            )

            # ----------------------------------------------------
            # TRUE COVARIANCE MATRIX
            # ----------------------------------------------------

            covariance_gene_selection = (
                select_covariance_heatmap_genes(
                    precomputed_genes=(
                        precomputed_genes
                    ),
                    Sigma_true=(
                        Sigma_true
                    ),
                    n_genes=(
                        N_COVARIANCE_GENES
                    ),
                )
            )

            covariance_plot = (
                prepare_covariance_heatmap(
                    Sigma_true=(
                        Sigma_true
                    ),
                    covariance_selection=(
                        covariance_gene_selection
                    ),
                )
            )

            covariance_matrix = (
                covariance_plot[
                    "matrix"
                ]
            )

            covariance_genes = (
                covariance_gene_selection[
                    "genes"
                ]
            )

            color_limit = (
                covariance_plot[
                    "color_limit"
                ]
            )

            # ----------------------------------------------------
            # FIGURE
            # ----------------------------------------------------

            figure, axes = plt.subplots(
                1,
                2,
                figsize=(19, 8),
                gridspec_kw={
                    "width_ratios": [
                        1.0,
                        1.2,
                    ]
                },
            )

            covariance_axis = axes[0]
            umap_axis = axes[1]

            covariance_image = (
                covariance_axis.imshow(
                    covariance_matrix,
                    aspect="equal",
                    interpolation="nearest",
                    cmap="coolwarm",
                    vmin=-color_limit,
                    vmax=color_limit,
                )
            )

            covariance_ticks = (
                even_tick_indices(
                    len(
                        covariance_genes
                    ),
                    MAX_COVARIANCE_TICK_LABELS,
                )
            )

            covariance_axis.set_xticks(
                covariance_ticks
            )

            covariance_axis.set_yticks(
                covariance_ticks
            )

            covariance_axis.set_xticklabels(
                covariance_genes[
                    covariance_ticks
                ],
                rotation=90,
                fontsize=7,
            )

            covariance_axis.set_yticklabels(
                covariance_genes[
                    covariance_ticks
                ],
                fontsize=7,
            )

            covariance_axis.set_xlabel(
                "genes"
            )

            covariance_axis.set_ylabel(
                "genes"
            )

            covariance_axis.set_title(
                "True covariance matrix\n"
                f"top {len(covariance_genes)} "
                "variance genes"
            )

            figure.colorbar(
                covariance_image,
                ax=covariance_axis,
                fraction=0.046,
                pad=0.03,
                label="covariance",
            )

            draw_perturbation_umap(
                ax=umap_axis,
                embedding=embedding,
                labels=(
                    selected_plot_labels
                ),
                embedding_method=(
                    embedding_result[
                        "method"
                    ]
                ),
            )

            figure.suptitle(
                dataset_name,
                fontsize=17,
                y=0.99,
            )

            figure.tight_layout(
                rect=[
                    0.0,
                    0.0,
                    0.88,
                    0.96,
                ]
            )

            output_stem = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__true_covariance_and_umap"
                )
            )

            png_path = Path(
                str(output_stem)
                + ".png"
            )

            svg_path = Path(
                str(output_stem)
                + ".svg"
            )

            figure.savefig(
                png_path,
                dpi=DPI,
                bbox_inches="tight",
            )

            figure.savefig(
                svg_path,
                bbox_inches="tight",
            )

            plt.show()
            plt.close(
                figure
            )

            # ----------------------------------------------------
            # SAVE UMAP COORDINATES
            # ----------------------------------------------------

            umap_dataframe = pd.DataFrame(
                {
                    "cell_index": (
                        selected_cell_indices
                    ),
                    "cell_name": (
                        selected_cell_names
                    ),
                    "original_perturbation_label": (
                        selected_raw_labels
                    ),
                    "plot_label": (
                        selected_plot_labels
                    ),
                    "is_control": (
                        selected_plot_labels
                        == "Control"
                    ),
                    "embedding_1": (
                        embedding[:, 0]
                    ),
                    "embedding_2": (
                        embedding[:, 1]
                    ),
                    "embedding_method": (
                        embedding_result[
                            "method"
                        ]
                    ),
                }
            )

            umap_csv_path = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__umap_coordinates.csv"
                )
            )

            umap_dataframe.to_csv(
                umap_csv_path,
                index=False,
            )

            # ----------------------------------------------------
            # SAVE COVARIANCE GENES
            # ----------------------------------------------------

            covariance_gene_dataframe = (
                pd.DataFrame(
                    {
                        "gene_index": (
                            covariance_gene_selection[
                                "indices"
                            ]
                        ),
                        "gene": (
                            covariance_genes
                        ),
                        "variance": (
                            covariance_gene_selection[
                                "variances"
                            ]
                        ),
                    }
                )
            )

            covariance_gene_csv_path = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__covariance_genes.csv"
                )
            )

            covariance_gene_dataframe.to_csv(
                covariance_gene_csv_path,
                index=False,
            )

            # ----------------------------------------------------
            # SAVE UMAP GENES
            # ----------------------------------------------------

            umap_gene_dataframe = pd.DataFrame(
                {
                    "precomputed_gene_index": (
                        umap_gene_selection[
                            "precomputed_indices"
                        ]
                    ),
                    "adata_gene_index": (
                        umap_gene_selection[
                            "adata_indices"
                        ]
                    ),
                    "gene": (
                        umap_gene_selection[
                            "genes"
                        ]
                    ),
                    "covariance_variance": (
                        umap_gene_selection[
                            "covariance_variances"
                        ]
                    ),
                }
            )

            umap_gene_csv_path = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__umap_genes.csv"
                )
            )

            umap_gene_dataframe.to_csv(
                umap_gene_csv_path,
                index=False,
            )

            print(
                f"[saved] {os.path.basename(png_path)}"
            )

            print(
                f"[saved] {os.path.basename(svg_path)}"
            )

            print(
                f"[saved] {os.path.basename(umap_csv_path)}"
            )

            print(
                f"[saved] {os.path.basename(sampling_summary_path)}"
            )

            result = {
                "dataset": (
                    dataset_name
                ),
                "status": "ok",
                "precomputed_folder": str(
                    dataset_folder
                ),
                "h5ad_path": str(
                    h5ad_path
                ),
                "true_sigma_path": str(
                    precomputed[
                        "true_sigma_path"
                    ]
                ),
                "perturbation_column": (
                    perturbation_column
                ),
                "expression_source": (
                    expression_source[
                        "name"
                    ]
                ),
                "n_cells_total": int(
                    len(
                        all_raw_labels
                    )
                ),
                "n_cells_umap": int(
                    len(
                        selected_cell_indices
                    )
                ),
                "n_controls_total": int(
                    n_controls_total
                ),
                "n_controls_umap": int(
                    n_controls_selected
                ),
                "n_plot_categories": int(
                    len(
                        pd.unique(
                            selected_plot_labels
                        )
                    )
                ),
                "top_perturbations": (
                    "|".join(
                        top_perturbations
                    )
                ),
                "n_umap_genes": int(
                    len(
                        umap_gene_selection[
                            "genes"
                        ]
                    )
                ),
                "n_covariance_genes": int(
                    len(
                        covariance_genes
                    )
                ),
                "normalization": (
                    normalization_name
                ),
                "embedding_method": (
                    embedding_result[
                        "method"
                    ]
                ),
                "png": str(
                    png_path
                ),
                "svg": str(
                    svg_path
                ),
                "umap_csv": str(
                    umap_csv_path
                ),
                "umap_sampling_summary": str(
                    sampling_summary_path
                ),
            }

            del X
            del preprocessing
            del embedding

            gc.collect()

            return result

        finally:
            if adata is not None:
                try:
                    adata.file.close()
                except Exception:
                    pass

            del Sigma_true
            gc.collect()


    # ============================================================
    # RUN ALL DATASETS
    # ============================================================

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_folders = (
        find_precomputed_dataset_folders(
            root=PRECOMPUTE_ROOT,
            expression_threshold=(
                EXPRESSION_THRESHOLD
            ),
        )
    )

    dataset_folders = (
        filter_dataset_folders(
            folders=dataset_folders,
            dataset_queries=(
                DATASET_QUERIES
            ),
        )
    )

    if len(
        dataset_folders
    ) == 0:
        raise FileNotFoundError(
            "No matching precomputed dataset folders "
            f"were found under:\n{PRECOMPUTE_ROOT}"
        )

    print(
        f"[datasets] found "
        f"{len(dataset_folders):,} datasets"
    )

    print(
        "[h5ad] indexing h5ad files once..."
    )

    indexed_h5ad_files = (
        index_h5ad_files(
            H5AD_SEARCH_ROOTS
        )
    )

    print(
        f"[h5ad] found "
        f"{len(indexed_h5ad_files):,} h5ad files"
    )

    results = []

    for dataset_number, dataset_folder in enumerate(
        dataset_folders,
        start=1,
    ):
        print(
            f"\n[{dataset_number}/"
            f"{len(dataset_folders)}]"
        )

        try:
            result = (
                process_one_dataset(
                    dataset_folder=(
                        dataset_folder
                    ),
                    indexed_h5ad_files=(
                        indexed_h5ad_files
                    ),
                )
            )

            results.append(
                result
            )

        except Exception as error:
            dataset_name = (
                dataset_name_from_folder(
                    dataset_folder
                )
            )

            print(
                "\n"
                + "!" * 110
            )

            print(
                f"[ERROR] {dataset_name}"
            )

            print(
                repr(error)
            )

            print(
                "!" * 110
            )

            results.append(
                {
                    "dataset": (
                        dataset_name
                    ),
                    "status": "error",
                    "precomputed_folder": str(
                        dataset_folder
                    ),
                    "error": repr(
                        error
                    ),
                }
            )

            gc.collect()


    # ============================================================
    # SAVE PROCESSING SUMMARY
    # ============================================================

    summary_dataframe = pd.DataFrame(
        results
    )

    summary_path = (
        OUT_ROOT
        / "dataset_processing_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    n_success = int(
        np.sum(
            summary_dataframe[
                "status"
            ]
            == "ok"
        )
    )

    n_errors = int(
        np.sum(
            summary_dataframe[
                "status"
            ]
            == "error"
        )
    )

    print(
        "\n"
        + "=" * 110
    )

    print("DONE")

    print(
        f"Successful datasets: "
        f"{n_success:,}"
    )

    print(
        f"Errored datasets:    "
        f"{n_errors:,}"
    )

    print(
        f"Summary:             "
        f"{os.path.basename(summary_path)}"
    )

    print(
        f"Figures:             "
        f"{os.path.basename(OUT_ROOT)}"
    )

    print(
        "=" * 110
    )

    if n_errors > 0:
        error_columns = [
            column
            for column in [
                "dataset",
                "error",
            ]
            if column
            in summary_dataframe.columns
        ]

        print(
            "\nDatasets with errors:"
        )

        print(
            summary_dataframe.loc[
                summary_dataframe[
                    "status"
                ]
                == "error",
                error_columns,
            ].to_string(
                index=False
            )
        )


def figS19_composite_shared_scale():
    """Cell-3 composite: pool off-diagonal covariance across datasets onto one shared color
    scale and draw the five-dataset covariance/UMAP composite.

    The source cell was an orphaned manual cell (it referenced an undefined `selected`, a
    `RESULT_ROOT`, and helper functions that were lost), so its per-dataset records are
    reconstructed here from figS19_true_cov_umap()'s on-disk outputs and the missing display
    helpers are provided locally; the plotting body below is the source cell (indented).
    """
    if SUPPL is None or OUTDIR is None:
        raise RuntimeError("SUPPL and OUTDIR must be injected by the notebook.")

    RESULT_ROOT = Path(OUTDIR)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT = TRUE_COV_OUT_ROOT if TRUE_COV_OUT_ROOT is not None else (Path(OUTDIR) / "true_cov_umap")
    PRECOMPUTE_ROOT = Path(SUPPL) / "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED"

    # Composite display config (reconstructed; the source cell's config was orphaned).
    MAX_UMAP_POINTS = 15_000
    MAX_COVARIANCE_GENES = N_COVARIANCE_GENES
    TOP_BOTTOM_HEIGHT_RATIO = [1.0, 1.0]
    UMAP_ALPHA = UMAP_POINT_ALPHA
    SHOW_COMPACT_UMAP_LEGENDS = True
    MAX_LEGEND_LABELS = 6

    def _find_sigma_path(sigma_directory):
        sigma_directory = Path(sigma_directory)
        for filename in TRUE_SIGMA_CANDIDATES:
            candidate = sigma_directory / filename
            if candidate.exists():
                return candidate
        return None

    def subsample_umap_dataframe(dataframe, max_points, seed):
        if len(dataframe) <= max_points:
            return dataframe
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(dataframe), size=max_points, replace=False))
        return dataframe.iloc[keep].reset_index(drop=True)

    def clean_dataset_title(name):
        return str(name).replace("_", " ")

    def deterministic_category_color(category):
        if str(category) == "Control":
            return (0.18, 0.18, 0.18, 1.0)
        digest = int(hashlib.md5(str(category).encode()).hexdigest()[:8], 16)
        return plt.cm.tab10((digest % 10) / 10.0)

    # Reconstruct the per-dataset records from figS19_true_cov_umap()'s on-disk outputs.
    tag = str(EXPRESSION_THRESHOLD).replace(".", "p")
    selected = []
    if OUT_ROOT.exists():
        for dataset_dir in sorted(p for p in OUT_ROOT.iterdir() if p.is_dir()):
            umap_files = sorted(dataset_dir.glob("*__umap_coordinates.csv"))
            covariance_files = sorted(dataset_dir.glob("*__covariance_genes.csv"))
            if not umap_files or not covariance_files:
                continue
            dataset = umap_files[0].name[: -len("__umap_coordinates.csv")]
            sigma_path = None
            for folder in sorted(PRECOMPUTE_ROOT.glob(f"{dataset}__mean_ge_{tag}")):
                sigma_path = _find_sigma_path(folder / "sigmas")
                if sigma_path is not None:
                    break
            if sigma_path is None:
                continue
            selected.append(
                {
                    "dataset": dataset,
                    "umap_path": str(umap_files[0]),
                    "covariance_gene_path": str(covariance_files[0]),
                    "sigma_path": str(sigma_path),
                }
            )

    if not selected:
        raise FileNotFoundError(
            "No per-dataset outputs found under the true-covariance UMAP directory; "
            "run figS19_true_cov_umap() first."
        )

    N_DATASETS = len(selected)
    FIGSIZE = (4.2 * N_DATASETS, 8.5)

    # ============================================================
    # LOAD MATRICES AND UMAP COORDINATES
    #
    # IMPORTANT:
    #   All covariance matrices will use the SAME symmetric color scale.
    #   The common limit is calculated from the pooled off-diagonal
    #   covariance values across all five datasets.
    # ============================================================

    loaded = []
    pooled_covariance_values = []

    for dataset_index, row in enumerate(selected):
        print(f"\nLoading {row['dataset']}...")

        # --------------------------------------------------------
        # LOAD SAVED UMAP COORDINATES
        # --------------------------------------------------------

        umap_dataframe = pd.read_csv(
            row["umap_path"]
        )

        required_umap_columns = [
            "embedding_1",
            "embedding_2",
        ]

        missing_umap_columns = [
            column
            for column in required_umap_columns
            if column not in umap_dataframe.columns
        ]

        if missing_umap_columns:
            raise ValueError(
                f"{os.path.basename(row['umap_path'])} is missing columns:\n"
                f"{missing_umap_columns}"
            )

        if "plot_label" not in umap_dataframe.columns:
            if (
                "original_perturbation_label"
                not in umap_dataframe.columns
            ):
                raise ValueError(
                    f"{os.path.basename(row['umap_path'])} has neither "
                    "'plot_label' nor "
                    "'original_perturbation_label'."
                )

            umap_dataframe["plot_label"] = (
                umap_dataframe[
                    "original_perturbation_label"
                ]
                .fillna("unknown")
                .astype(str)
            )

        umap_dataframe["plot_label"] = (
            umap_dataframe["plot_label"]
            .fillna("unknown")
            .astype(str)
        )

        # Ensure all saved control cells are displayed as Control.
        if "is_control" in umap_dataframe.columns:
            control_mask = (
                umap_dataframe["is_control"]
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                        "yes",
                    ]
                )
            )

            umap_dataframe.loc[
                control_mask,
                "plot_label",
            ] = "Control"

        umap_dataframe = subsample_umap_dataframe(
            dataframe=umap_dataframe,
            max_points=MAX_UMAP_POINTS,
            seed=RANDOM_SEED + dataset_index,
        )

        # --------------------------------------------------------
        # LOAD SAVED COVARIANCE-GENE SELECTION
        # --------------------------------------------------------

        covariance_genes = pd.read_csv(
            row["covariance_gene_path"]
        )

        if "gene_index" not in covariance_genes.columns:
            raise ValueError(
                f"{os.path.basename(row['covariance_gene_path'])} "
                "is missing 'gene_index'."
            )

        covariance_genes = covariance_genes.iloc[
            :MAX_COVARIANCE_GENES
        ].copy()

        covariance_indices = (
            pd.to_numeric(
                covariance_genes["gene_index"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .to_numpy()
        )

        # --------------------------------------------------------
        # LOAD TRUE COVARIANCE MATRIX
        # --------------------------------------------------------

        Sigma_true = np.load(
            row["sigma_path"],
            mmap_mode="r",
        )

        valid_indices = covariance_indices[
            (covariance_indices >= 0)
            & (covariance_indices < Sigma_true.shape[0])
        ]

        if len(valid_indices) < 2:
            raise ValueError(
                f"Too few valid covariance indices for "
                f"{row['dataset']}."
            )

        covariance = np.asarray(
            Sigma_true[
                np.ix_(
                    valid_indices,
                    valid_indices,
                )
            ],
            dtype=np.float64,
        )

        covariance = 0.5 * (
            covariance + covariance.T
        )

        # --------------------------------------------------------
        # COLLECT OFF-DIAGONAL VALUES FOR ONE GLOBAL SCALE
        # --------------------------------------------------------

        off_diagonal_mask = ~np.eye(
            covariance.shape[0],
            dtype=bool,
        )

        off_diagonal_values = covariance[
            off_diagonal_mask
        ]

        off_diagonal_values = off_diagonal_values[
            np.isfinite(off_diagonal_values)
        ]

        if off_diagonal_values.size > 0:
            pooled_covariance_values.append(
                off_diagonal_values
            )

        loaded.append(
            {
                **row,
                "umap": umap_dataframe,
                "covariance": covariance,
            }
        )

        del Sigma_true


    # ============================================================
    # CALCULATE ONE GLOBAL COVARIANCE COLOR LIMIT
    # ============================================================

    if len(pooled_covariance_values) == 0:
        raise RuntimeError(
            "No finite off-diagonal covariance values were found."
        )

    pooled_covariance_values = np.concatenate(
        pooled_covariance_values
    )

    GLOBAL_COVARIANCE_LIMIT = float(
        np.quantile(
            np.abs(pooled_covariance_values),
            COVARIANCE_COLOR_QUANTILE,
        )
    )

    if (
        not np.isfinite(GLOBAL_COVARIANCE_LIMIT)
        or GLOBAL_COVARIANCE_LIMIT <= 0
    ):
        GLOBAL_COVARIANCE_LIMIT = float(
            np.max(
                np.abs(
                    pooled_covariance_values
                )
            )
        )

    if (
        not np.isfinite(GLOBAL_COVARIANCE_LIMIT)
        or GLOBAL_COVARIANCE_LIMIT <= 0
    ):
        GLOBAL_COVARIANCE_LIMIT = 1.0

    print(
        "\nGlobal covariance display range:"
    )

    print(
        f"  {-GLOBAL_COVARIANCE_LIMIT:.6g} "
        f"to {GLOBAL_COVARIANCE_LIMIT:.6g}"
    )

    print(
        f"  based on pooled "
        f"{100 * COVARIANCE_COLOR_QUANTILE:.1f}th percentile "
        "of absolute off-diagonal covariance"
    )


    # ============================================================
    # MAKE 10-PANEL COMPOSITE
    #
    # TOP ROW:
    #   Five true covariance matrices on one common color scale
    #
    # BOTTOM ROW:
    #   Five corresponding UMAPs
    # ============================================================

    figure, axes = plt.subplots(
        2,
        N_DATASETS,
        figsize=FIGSIZE,
        gridspec_kw={
            "height_ratios": TOP_BOTTOM_HEIGHT_RATIO,
            "hspace": 0.22,
            "wspace": 0.08,
        },
    )

    panel_letters = list(
        "ABCDEFGHIJ"
    )

    # Keep one covariance image object for the shared colorbar.
    shared_covariance_image = None


    for dataset_index, data in enumerate(loaded):
        covariance_axis = axes[
            0,
            dataset_index,
        ]

        umap_axis = axes[
            1,
            dataset_index,
        ]

        # --------------------------------------------------------
        # TOP ROW: TRUE COVARIANCE
        #
        # Every panel uses exactly the same:
        #
        #   vmin = -GLOBAL_COVARIANCE_LIMIT
        #   vmax = +GLOBAL_COVARIANCE_LIMIT
        # --------------------------------------------------------

        covariance = data["covariance"]

        covariance_image = covariance_axis.imshow(
            covariance,
            aspect="equal",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-GLOBAL_COVARIANCE_LIMIT,
            vmax=GLOBAL_COVARIANCE_LIMIT,
            rasterized=True,
        )

        if shared_covariance_image is None:
            shared_covariance_image = covariance_image

        # No gene labels or gene ticks.
        covariance_axis.set_xticks([])
        covariance_axis.set_yticks([])

        covariance_axis.set_xlabel("")
        covariance_axis.set_ylabel("")

        covariance_axis.set_title(
            clean_dataset_title(
                data["dataset"]
            ),
            fontsize=12,
            pad=8,
        )

        covariance_axis.text(
            0.02,
            0.98,
            panel_letters[dataset_index],
            transform=covariance_axis.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.5,
            },
        )

        # --------------------------------------------------------
        # BOTTOM ROW: UMAP
        # --------------------------------------------------------

        umap_dataframe = data["umap"]

        label_counts = (
            umap_dataframe["plot_label"]
            .value_counts()
        )

        # Plot large groups first so smaller groups remain visible.
        plot_order = (
            label_counts
            .sort_values(
                ascending=False
            )
            .index
            .tolist()
        )

        for category in plot_order:
            category_mask = (
                umap_dataframe["plot_label"]
                == category
            )

            category_points = umap_dataframe.loc[
                category_mask,
                [
                    "embedding_1",
                    "embedding_2",
                ],
            ]

            umap_axis.scatter(
                category_points["embedding_1"],
                category_points["embedding_2"],
                s=UMAP_POINT_SIZE,
                alpha=UMAP_ALPHA,
                linewidths=0,
                color=deterministic_category_color(
                    category
                ),
                rasterized=True,
            )

        # No labels, ticks, or gene names inside the UMAP.
        umap_axis.set_xticks([])
        umap_axis.set_yticks([])

        umap_axis.set_xlabel("")
        umap_axis.set_ylabel("")

        for spine in umap_axis.spines.values():
            spine.set_visible(False)

        umap_axis.text(
            0.02,
            0.98,
            panel_letters[
                N_DATASETS + dataset_index
            ],
            transform=umap_axis.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.5,
            },
        )

        if SHOW_COMPACT_UMAP_LEGENDS:
            legend_categories = []

            if "Control" in label_counts.index:
                legend_categories.append(
                    "Control"
                )

            legend_categories.extend(
                [
                    category
                    for category in label_counts.index
                    if category != "Control"
                ][
                    :max(
                        0,
                        MAX_LEGEND_LABELS
                        - len(legend_categories),
                    )
                ]
            )

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markersize=4,
                    markerfacecolor=(
                        deterministic_category_color(
                            category
                        )
                    ),
                    markeredgecolor="none",
                    label=str(category),
                )
                for category in legend_categories
            ]

            umap_axis.legend(
                handles=legend_handles,
                frameon=False,
                fontsize=5.5,
                loc="lower left",
                handletextpad=0.2,
                borderaxespad=0.1,
                labelspacing=0.2,
            )


    # ============================================================
    # ONE SHARED COVARIANCE SCALE BAR
    #
    # It spans all five covariance panels and therefore communicates
    # that covariance color magnitude is directly comparable across
    # every dataset.
    # ============================================================

    shared_colorbar = figure.colorbar(
        shared_covariance_image,
        ax=axes[0, :].ravel().tolist(),
        orientation="horizontal",
        fraction=0.045,
        pad=0.055,
        aspect=70,
    )

    shared_colorbar.set_label(
        "Covariance",
        fontsize=11,
    )

    shared_colorbar.ax.tick_params(
        labelsize=8,
        length=3,
    )


    # ============================================================
    # ROW LABELS
    # ============================================================

    figure.text(
        0.007,
        0.735,
        "True covariance",
        rotation=90,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    figure.text(
        0.007,
        0.255,
        "Expression UMAP",
        rotation=90,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    figure.suptitle(
        "Control covariance structure and cell-expression organization",
        fontsize=17,
        y=0.995,
    )

    figure.subplots_adjust(
        left=0.025,
        right=0.995,
        bottom=0.025,
        top=0.94,
    )


    # ============================================================
    # SAVE
    # ============================================================

    output_png = (
        RESULT_ROOT
        / "FIVE_DATASET__TRUE_COVARIANCE_AND_UMAP_SHARED_SCALE.png"
    )

    output_pdf = (
        RESULT_ROOT
        / "FIVE_DATASET__TRUE_COVARIANCE_AND_UMAP_SHARED_SCALE.pdf"
    )

    output_svg = (
        RESULT_ROOT
        / "FIVE_DATASET__TRUE_COVARIANCE_AND_UMAP_SHARED_SCALE.svg"
    )

    figure.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    figure.savefig(
        output_svg,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(figure)

    print("\nSaved shared-scale composite figure:")
    print(f"  PNG: {os.path.basename(output_png)}")
    print(f"  PDF: {os.path.basename(output_pdf)}")
    print(f"  SVG: {os.path.basename(output_svg)}")


def figS19_pflog_cov_umap():
    """Cell-20 pipeline: same true-covariance heatmap with a PFlog (NB-alpha) PCA/UMAP embedding.

    Nested helper defs and the driver body are the source cell moved here verbatim (indented);
    per-variant and PFlog-specific constants plus the DATA_DIR/SUPPL/OUTDIR-derived paths are
    local, while the shared constants are read from the module globals above.
    """
    if DATA_DIR is None or SUPPL is None or OUTDIR is None:
        raise RuntimeError("DATA_DIR, SUPPL, and OUTDIR must be injected by the notebook.")

    PRECOMPUTE_ROOT = Path(SUPPL) / "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED"
    H5AD_SEARCH_ROOTS = [Path(SUPPL), Path(DATA_DIR)]
    OUT_ROOT = Path(OUTDIR) / "pflog_cov_umap"

    # Per-variant tunables (cell 20: PFlog / NB-overdispersion normalized embedding)
    DATASET_QUERIES = [
        "TianKampmann2021_CRISPRa",
        "NormanWeissman2019_filtered",
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "XAtlas2025_HEK293T_filtered",
    ]
    N_TOP_PERTURBATIONS = 1
    MAX_UMAP_CELLS = 10_000
    MAX_CONTROL_CELLS = 5_000
    N_PCS = 200

    COUNTS_LAYER_CANDIDATES = [
        "counts",
        "raw_counts",
        "count",
    ]

    MAX_ALPHA_FIT_CELLS = 1_000
    MAX_ALPHA_FIT_GENES = 6_000
    ALPHA_N_BINS = 30
    ALPHA_MIN_GENES_PER_BIN = 6
    ALPHA_MIN_MEAN = 0.05
    ALPHA_MIN_DETECTION_FRACTION = 0.005
    ALPHA_MIN_OVERPOISSON_BINS = 3
    MIN_ALPHA = 1e-8
    MAX_ALPHA = 1e3

    GENE_CHUNK_SIZE = 4_096
    N_PFLOG_PCA_GENES = 3_000
    MIN_PCA_GENE_DETECTION_FRACTION = 0.01
    MIN_PFLOG_VARIANCE = 1e-10

    INVALID_PERTURBATION_LABELS = {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "unknown",
        "unassigned",
        "unclassified",
        "unlabeled",
        "unlabelled",
        "missing",
        "ambiguous",
        "multiple",
        "multiplet",
        "doublet",
        "no guide",
        "no_guide",
        "noguide",
        "not assigned",
        "not_assigned",
        "not detected",
        "not_detected",
    }

    INVALID_PERTURBATION_SUBSTRINGS = [
        "unassigned",
        "unclassified",
        "unlabeled",
        "unlabelled",
        "ambiguous",
        "multiplet",
        "doublet",
        "no guide",
        "no_guide",
        "noguide",
        "not assigned",
        "not_assigned",
    ]

    def threshold_to_tag(value):
        return str(value).replace(
            ".",
            "p",
        )


    def safe_filename(value):
        return re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(value),
        ).strip("_")


    def dataset_name_from_folder(folder):
        return re.sub(
            r"__mean_ge_[^/]+$",
            "",
            Path(folder).name,
        )


    def normalized_name(value):
        return re.sub(
            r"[^a-z0-9]+",
            "",
            str(value).lower(),
        )


    def decode_string_array(values):
        return np.asarray(
            [
                value.decode("utf-8")
                if isinstance(value, bytes)
                else str(value)
                for value in np.asarray(
                    values,
                    dtype=object,
                )
            ],
            dtype=object,
        )


    def series_to_string_array(
        series,
        missing_value="unknown",
    ):
        """
        Convert categorical or ordinary pandas labels into a normal
        object array without calling fillna on a categorical Series.
        """

        values = (
            series
            .astype(object)
            .to_numpy(copy=True)
        )

        values[
            pd.isna(values)
        ] = missing_value

        return np.asarray(
            [
                str(value)
                for value in values
            ],
            dtype=object,
        )


    def even_tick_indices(
        n_values,
        max_ticks,
    ):
        if n_values <= max_ticks:
            return np.arange(
                n_values,
                dtype=int,
            )

        return np.unique(
            np.linspace(
                0,
                n_values - 1,
                max_ticks,
            )
            .round()
            .astype(int)
        )


    def is_control_label(value):
        value = str(
            value
        ).strip().lower()

        return any(
            pattern in value
            for pattern in CONTROL_PATTERNS
        )


    def collapse_control_labels(labels):
        labels = np.asarray(
            labels,
            dtype=object,
        )

        return np.asarray(
            [
                "Control"
                if is_control_label(label)
                else str(label)
                for label in labels
            ],
            dtype=object,
        )


    def is_invalid_perturbation_label(value):
        value = str(
            value
        ).strip().lower()

        if value in INVALID_PERTURBATION_LABELS:
            return True

        return any(
            token in value
            for token
            in INVALID_PERTURBATION_SUBSTRINGS
        )


    def ensure_sorted_unique(indices):
        return np.unique(
            np.asarray(
                indices,
                dtype=int,
            )
        )


    # ============================================================
    # PRECOMPUTED DATASET DISCOVERY
    # ============================================================

    def find_precomputed_dataset_folders(
        root,
        expression_threshold,
    ):
        root = Path(root)

        tag = threshold_to_tag(
            expression_threshold
        )

        return sorted(
            folder
            for folder in root.glob(
                f"*__mean_ge_{tag}"
            )
            if folder.is_dir()
        )


    def select_dataset_folders(
        folders,
        dataset_queries,
    ):
        selected = []

        for query in dataset_queries:
            query_lower = str(
                query
            ).lower()

            matches = [
                folder
                for folder in folders
                if query_lower
                in folder.name.lower()
            ]

            if len(matches) == 0:
                warnings.warn(
                    "No precomputed folder matched "
                    f"{query!r}."
                )

                continue

            if len(matches) > 1:
                raise RuntimeError(
                    f"Dataset query {query!r} matched "
                    "multiple precomputed folders:\n"
                    + "\n".join(
                        str(path)
                        for path in matches
                    )
                )

            selected.append(
                matches[0]
            )

        return selected


    def find_true_sigma_path(
        sigma_directory,
    ):
        sigma_directory = Path(
            sigma_directory
        )

        for filename in TRUE_SIGMA_CANDIDATES:
            path = (
                sigma_directory
                / filename
            )

            if path.exists():
                return path

        hits = []

        for pattern in [
            "*Sigma*full*.npy",
            "*Sigma*true*.npy",
            "*sigma*full*.npy",
            "*sigma*true*.npy",
        ]:
            hits.extend(
                sorted(
                    sigma_directory.glob(
                        pattern
                    )
                )
            )

        hits = [
            path
            for path in hits
            if not any(
                excluded
                in path.name.lower()
                for excluded in [
                    "meanfield",
                    "mean_field",
                    "shuffle",
                    "shuffled",
                    "_mf",
                ]
            )
        ]

        if hits:
            return hits[0]

        available = sorted(
            path.name
            for path
            in sigma_directory.glob(
                "*.npy"
            )
        )

        raise FileNotFoundError(
            "Could not locate a true/full covariance "
            f"matrix in:\n{sigma_directory}\n"
            f"Available files: {available}"
        )


    def load_precomputed_information(
        dataset_folder,
    ):
        dataset_folder = Path(
            dataset_folder
        )

        genes_path = (
            dataset_folder
            / "genes.npy"
        )

        sigma_directory = (
            dataset_folder
            / "sigmas"
        )

        if not genes_path.exists():
            raise FileNotFoundError(
                f"Missing {genes_path}"
            )

        if not sigma_directory.exists():
            raise FileNotFoundError(
                f"Missing {sigma_directory}"
            )

        genes = decode_string_array(
            np.load(
                genes_path,
                allow_pickle=True,
            )
        )

        true_sigma_path = (
            find_true_sigma_path(
                sigma_directory
            )
        )

        Sigma_true = np.load(
            true_sigma_path,
            mmap_mode="r",
        )

        if Sigma_true.shape != (
            len(genes),
            len(genes),
        ):
            raise ValueError(
                f"Sigma shape {Sigma_true.shape} "
                f"does not match {len(genes)} genes."
            )

        return {
            "genes": genes,
            "Sigma_true": Sigma_true,
            "true_sigma_path": (
                true_sigma_path
            ),
        }


    # ============================================================
    # H5AD DISCOVERY
    # ============================================================

    def index_h5ad_files(search_roots):
        indexed = {}

        for root in search_roots:
            root = Path(root)

            if not root.exists():
                continue

            for path in root.rglob(
                "*.h5ad"
            ):
                try:
                    key = str(
                        path.resolve()
                    )
                except Exception:
                    key = str(path)

                indexed[key] = path

        return sorted(
            indexed.values(),
            key=lambda path: str(path),
        )


    def find_matching_h5ad(
        dataset_folder,
        dataset_name,
        indexed_h5ad_files,
    ):
        for key in [
            dataset_folder.name,
            dataset_name,
        ]:
            if key not in H5AD_PATH_OVERRIDES:
                continue

            path = Path(
                H5AD_PATH_OVERRIDES[key]
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"H5AD override does not exist: {path}"
                )

            return path

        target = normalized_name(
            dataset_name
        )

        candidates = []

        for path in indexed_h5ad_files:
            stem = normalized_name(
                path.stem
            )

            exact = (
                stem == target
            )

            contains = (
                target in stem
            )

            if exact or contains:
                candidates.append(
                    {
                        "path": path,
                        "exact": exact,
                        "length_difference": abs(
                            len(stem)
                            - len(target)
                        ),
                    }
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda row: (
                not row["exact"],
                row["length_difference"],
                len(
                    str(row["path"])
                ),
            )
        )

        if len(candidates) > 1:
            print(
                f"[h5ad] Multiple matches for {dataset_name}:"
            )

            for row in candidates[:5]:
                print(
                    f"        {os.path.basename(row['path'])}"
                )

            print(
                "[h5ad] Selected: "
                f"{os.path.basename(candidates[0]['path'])}"
            )

        return candidates[0][
            "path"
        ]


    # ============================================================
    # COUNT SOURCE AND BACKED MATRIX ACCESS
    # ============================================================

    def choose_count_source(adata):
        for layer in COUNTS_LAYER_CANDIDATES:
            if layer in adata.layers:
                return {
                    "kind": "layer",
                    "layer": layer,
                    "name": (
                        f"layers[{layer!r}]"
                    ),
                    "var_names": np.asarray(
                        adata.var_names,
                        dtype=object,
                    ),
                }

        return {
            "kind": "X",
            "layer": None,
            "name": "X",
            "var_names": np.asarray(
                adata.var_names,
                dtype=object,
            ),
        }


    def materialize_matrix(matrix):
        if hasattr(
            matrix,
            "to_memory",
        ):
            matrix = matrix.to_memory()

        if sparse.issparse(
            matrix
        ):
            return matrix.tocsr()

        return np.asarray(
            matrix
        )


    def read_count_block(
        adata,
        source,
        row_indices,
        column_indices,
    ):
        """
        Read a selected count block from backed AnnData.

        Sorted row and column indices reduce HDF5 fancy-index overhead.
        """

        row_indices = ensure_sorted_unique(
            row_indices
        )

        if isinstance(
            column_indices,
            slice,
        ):
            column_indexer = (
                column_indices
            )

        else:
            column_indexer = (
                ensure_sorted_unique(
                    column_indices
                )
            )

        view = adata[
            row_indices,
            column_indexer,
        ]

        if source["kind"] == "layer":
            matrix = view.layers[
                source["layer"]
            ]

        else:
            matrix = view.X

        return materialize_matrix(
            matrix
        )


    def matrix_looks_like_counts(
        matrix,
        seed=0,
    ):
        if sparse.issparse(
            matrix
        ):
            values = matrix.data
        else:
            values = np.asarray(
                matrix
            ).ravel()

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            return True

        if values.size > 100_000:
            rng = np.random.default_rng(
                seed
            )

            values = rng.choice(
                values,
                size=100_000,
                replace=False,
            )

        if np.min(values) < 0:
            return False

        integer_fraction = np.mean(
            np.abs(
                values
                - np.round(values)
            )
            < 1e-6
        )

        return bool(
            integer_fraction
            >= 0.95
        )


    def validate_count_source(
        adata,
        source,
        row_indices,
    ):
        sample_rows = ensure_sorted_unique(
            row_indices[
                :min(
                    len(row_indices),
                    300,
                )
            ]
        )

        n_sample_genes = min(
            adata.n_vars,
            1_000,
        )

        sample = read_count_block(
            adata,
            source,
            sample_rows,
            slice(
                0,
                n_sample_genes,
            ),
        )

        if not matrix_looks_like_counts(
            sample,
            seed=RANDOM_SEED,
        ):
            raise ValueError(
                f"{source['name']} does not appear to contain "
                "raw nonnegative integer counts. PFlog requires "
                "raw count data."
            )

        del sample


    # ============================================================
    # PERTURBATION COLUMN DETECTION
    # ============================================================

    def find_perturbation_column(obs):
        for column in (
            PERTURBATION_COLUMN_CANDIDATES
        ):
            if column in obs.columns:
                return column

        candidates = []

        for column in obs.columns:
            values = obs[column]

            string_like = (
                pd.api.types.is_object_dtype(
                    values
                )
                or pd.api.types.is_string_dtype(
                    values
                )
                or isinstance(
                    values.dtype,
                    pd.CategoricalDtype,
                )
            )

            if not string_like:
                continue

            n_unique = values.nunique(
                dropna=True
            )

            if not (
                2
                <= n_unique
                <= max(
                    5_000,
                    int(
                        0.75
                        * len(obs)
                    ),
                )
            ):
                continue

            labels = (
                series_to_string_array(
                    values
                )
            )

            candidates.append(
                {
                    "column": column,
                    "n_unique": n_unique,
                    "contains_control": any(
                        is_control_label(
                            value
                        )
                        for value in labels
                    ),
                }
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda row: (
                not row[
                    "contains_control"
                ],
                row[
                    "n_unique"
                ],
            )
        )

        return candidates[0][
            "column"
        ]


    # ============================================================
    # PROPORTIONAL CELL ALLOCATION
    # ============================================================

    def capped_proportional_allocation(
        available_counts,
        total_cap,
        control_cap,
    ):
        available_counts = (
            pd.Series(
                available_counts,
                dtype="int64",
            )
            .clip(lower=0)
        )

        capacities = (
            available_counts.copy()
        )

        if "Control" in capacities.index:
            capacities.loc[
                "Control"
            ] = min(
                int(
                    capacities.loc[
                        "Control"
                    ]
                ),
                int(control_cap),
            )

        target_total = min(
            int(total_cap),
            int(
                capacities.sum()
            ),
        )

        continuous = pd.Series(
            0.0,
            index=available_counts.index,
        )

        active = list(
            available_counts.index
        )

        remaining = float(
            target_total
        )

        while (
            active
            and remaining > 1e-10
        ):
            residual = (
                capacities.loc[
                    active
                ]
                - continuous.loc[
                    active
                ]
            )

            active = [
                category
                for category in active
                if residual.loc[
                    category
                ] > 1e-10
            ]

            if not active:
                break

            weights = (
                available_counts.loc[
                    active
                ]
                .astype(float)
            )

            if weights.sum() <= 0:
                weights[:] = 1.0

            proposal = (
                remaining
                * weights
                / weights.sum()
            )

            residual = (
                capacities.loc[
                    active
                ]
                - continuous.loc[
                    active
                ]
            )

            capped = [
                category
                for category in active
                if proposal.loc[
                    category
                ]
                >= residual.loc[
                    category
                ] - 1e-10
            ]

            if not capped:
                continuous.loc[
                    active
                ] += proposal

                remaining = 0.0
                break

            for category in capped:
                amount = float(
                    residual.loc[
                        category
                    ]
                )

                continuous.loc[
                    category
                ] += amount

                remaining -= amount

            active = [
                category
                for category in active
                if category not in capped
            ]

        allocation = (
            np.floor(
                continuous
                + 1e-12
            )
            .astype(int)
        )

        cells_left = (
            target_total
            - int(
                allocation.sum()
            )
        )

        fractional = (
            continuous
            - allocation
        )

        while cells_left > 0:
            eligible = [
                category
                for category
                in allocation.index
                if allocation.loc[
                    category
                ]
                < capacities.loc[
                    category
                ]
            ]

            if not eligible:
                break

            priority = pd.DataFrame(
                {
                    "fractional": (
                        fractional.loc[
                            eligible
                        ]
                    ),
                    "available": (
                        available_counts.loc[
                            eligible
                        ]
                    ),
                }
            ).sort_values(
                [
                    "fractional",
                    "available",
                ],
                ascending=False,
            )

            chosen = priority.index[0]

            allocation.loc[
                chosen
            ] += 1

            fractional.loc[
                chosen
            ] = 0.0

            cells_left -= 1

        return allocation.astype(
            int
        )


    def select_umap_cells(
        raw_labels,
        seed,
    ):
        rng = np.random.default_rng(
            seed
        )

        raw_labels = np.asarray(
            raw_labels,
            dtype=object,
        )

        collapsed = (
            collapse_control_labels(
                raw_labels
            )
        )

        valid_noncontrol = np.asarray(
            [
                (
                    label != "Control"
                    and not is_invalid_perturbation_label(
                        label
                    )
                )
                for label in collapsed
            ],
            dtype=bool,
        )

        perturbation_counts = (
            pd.Series(
                collapsed[
                    valid_noncontrol
                ],
                dtype="object",
            )
            .value_counts()
            .sort_values(
                ascending=False
            )
        )

        if perturbation_counts.empty:
            raise ValueError(
                "No valid non-control perturbations "
                "were found."
            )

        top_perturbations = (
            perturbation_counts
            .head(
                N_TOP_PERTURBATIONS
            )
            .index
            .astype(str)
            .tolist()
        )

        retained_categories = []

        if np.any(
            collapsed == "Control"
        ):
            retained_categories.append(
                "Control"
            )

        retained_categories.extend(
            top_perturbations
        )

        retained_mask = np.isin(
            collapsed,
            retained_categories,
        )

        retained_indices = np.flatnonzero(
            retained_mask
        )

        retained_labels = collapsed[
            retained_indices
        ]

        available_counts = (
            pd.Series(
                retained_labels,
                dtype="object",
            )
            .value_counts()
            .reindex(
                retained_categories,
                fill_value=0,
            )
            .astype(int)
        )

        allocation = (
            capped_proportional_allocation(
                available_counts=(
                    available_counts
                ),
                total_cap=(
                    MAX_UMAP_CELLS
                ),
                control_cap=(
                    MAX_CONTROL_CELLS
                ),
            )
        )

        selected_parts = []

        for category in retained_categories:
            category_indices = retained_indices[
                retained_labels
                == category
            ]

            n_take = int(
                allocation.loc[
                    category
                ]
            )

            if n_take <= 0:
                continue

            if n_take >= len(
                category_indices
            ):
                chosen = (
                    category_indices
                )

            else:
                chosen = rng.choice(
                    category_indices,
                    size=n_take,
                    replace=False,
                )

            selected_parts.append(
                np.asarray(
                    chosen,
                    dtype=int,
                )
            )

        if not selected_parts:
            raise RuntimeError(
                "No cells were selected."
            )

        selected_indices = np.sort(
            np.concatenate(
                selected_parts
            )
        )

        selected_labels = collapsed[
            selected_indices
        ]

        selected_counts = (
            pd.Series(
                selected_labels,
                dtype="object",
            )
            .value_counts()
            .reindex(
                retained_categories,
                fill_value=0,
            )
            .astype(int)
        )

        summary = pd.DataFrame(
            {
                "category": (
                    retained_categories
                ),
                "is_control": [
                    category == "Control"
                    for category
                    in retained_categories
                ],
                "available_cells": [
                    int(
                        available_counts.loc[
                            category
                        ]
                    )
                    for category
                    in retained_categories
                ],
                "selected_cells": [
                    int(
                        selected_counts.loc[
                            category
                        ]
                    )
                    for category
                    in retained_categories
                ],
            }
        )

        summary[
            "available_fraction"
        ] = (
            summary[
                "available_cells"
            ]
            / max(
                int(
                    summary[
                        "available_cells"
                    ].sum()
                ),
                1,
            )
        )

        summary[
            "selected_fraction"
        ] = (
            summary[
                "selected_cells"
            ]
            / max(
                int(
                    summary[
                        "selected_cells"
                    ].sum()
                ),
                1,
            )
        )

        invalid_counts = (
            pd.Series(
                collapsed[
                    (
                        collapsed
                        != "Control"
                    )
                    & ~valid_noncontrol
                ],
                dtype="object",
            )
            .value_counts()
            .head(20)
        )

        print(
            "\n[excluded non-perturbation labels]"
        )

        if invalid_counts.empty:
            print("  None")
        else:
            print(
                invalid_counts.to_string()
            )

        print(
            "\n[UMAP retained categories]"
        )

        print(
            summary.to_string(
                index=False,
                formatters={
                    "available_fraction": (
                        lambda value:
                        f"{value:.4f}"
                    ),
                    "selected_fraction": (
                        lambda value:
                        f"{value:.4f}"
                    ),
                },
            )
        )

        return {
            "selected_indices": (
                selected_indices
            ),
            "selected_labels": (
                selected_labels
            ),
            "top_perturbations": (
                top_perturbations
            ),
            "summary": summary,
            "collapsed_labels": (
                collapsed
            ),
        }


    # ============================================================
    # SPARSE MEAN / VARIANCE
    # ============================================================

    def matrix_mean_variance_detection(
        matrix,
    ):
        n_cells = int(
            matrix.shape[0]
        )

        if n_cells < 2:
            raise ValueError(
                "At least two cells are required."
            )

        if sparse.issparse(
            matrix
        ):
            matrix = matrix.tocsr()

            sums = np.asarray(
                matrix.sum(
                    axis=0
                )
            ).ravel().astype(
                np.float64
            )

            squared = matrix.copy()

            squared.data = (
                squared.data.astype(
                    np.float64,
                    copy=False,
                )
                ** 2
            )

            squared_sums = np.asarray(
                squared.sum(
                    axis=0
                )
            ).ravel()

            detection = (
                np.asarray(
                    matrix.getnnz(
                        axis=0
                    )
                ).ravel()
                / n_cells
            )

        else:
            matrix = np.asarray(
                matrix,
                dtype=np.float64,
            )

            sums = np.sum(
                matrix,
                axis=0,
                dtype=np.float64,
            )

            squared_sums = np.sum(
                matrix ** 2,
                axis=0,
                dtype=np.float64,
            )

            detection = np.mean(
                matrix > 0,
                axis=0,
            )

        means = (
            sums
            / n_cells
        )

        variances = (
            squared_sums
            - n_cells
            * means ** 2
        ) / (
            n_cells - 1
        )

        variances = np.maximum(
            variances,
            0.0,
        )

        return {
            "mean": means,
            "variance": variances,
            "detection_fraction": (
                detection
            ),
        }


    # ============================================================
    # FAST ALPHA FIT
    # ============================================================

    def weighted_median(
        values,
        weights,
    ):
        values = np.asarray(
            values,
            dtype=float,
        )

        weights = np.asarray(
            weights,
            dtype=float,
        )

        order = np.argsort(
            values
        )

        values = values[
            order
        ]

        weights = weights[
            order
        ]

        cumulative = np.cumsum(
            weights
        )

        cutoff = (
            0.5
            * np.sum(weights)
        )

        index = np.searchsorted(
            cumulative,
            cutoff,
            side="left",
        )

        return float(
            values[
                min(
                    index,
                    len(values) - 1,
                )
            ]
        )


    def fit_alpha_from_binned_statistics(
        means,
        variances,
        detection,
    ):
        """
        Robustly fit one dataset-level NB alpha from approximately
        ALPHA_N_BINS mean-expression bins.
        """

        means = np.asarray(
            means,
            dtype=float,
        )

        variances = np.asarray(
            variances,
            dtype=float,
        )

        detection = np.asarray(
            detection,
            dtype=float,
        )

        valid = (
            np.isfinite(means)
            & np.isfinite(variances)
            & np.isfinite(detection)
            & (
                means
                >= ALPHA_MIN_MEAN
            )
            & (
                variances
                >= 0
            )
            & (
                detection
                >= ALPHA_MIN_DETECTION_FRACTION
            )
        )

        means = means[
            valid
        ]

        variances = variances[
            valid
        ]

        if len(means) < (
            2
            * ALPHA_MIN_GENES_PER_BIN
        ):
            raise ValueError(
                "Too few expressed genes passed "
                "the alpha-fit filters."
            )

        log_means = np.log10(
            means
        )

        lower = float(
            np.min(
                log_means
            )
        )

        upper = float(
            np.max(
                log_means
            )
        )

        if upper <= lower:
            raise ValueError(
                "Gene means do not span a usable range."
            )

        edges = np.linspace(
            lower,
            upper,
            ALPHA_N_BINS + 1,
        )

        bin_ids = np.clip(
            np.digitize(
                log_means,
                edges,
            ) - 1,
            0,
            ALPHA_N_BINS - 1,
        )

        rows = []

        for bin_id in range(
            ALPHA_N_BINS
        ):
            mask = (
                bin_ids == bin_id
            )

            n_genes = int(
                np.sum(mask)
            )

            if n_genes < (
                ALPHA_MIN_GENES_PER_BIN
            ):
                continue

            mean_median = float(
                np.median(
                    means[
                        mask
                    ]
                )
            )

            variance_median = float(
                np.median(
                    variances[
                        mask
                    ]
                )
            )

            alpha_bin = (
                variance_median
                - mean_median
            ) / max(
                mean_median ** 2,
                1e-30,
            )

            rows.append(
                {
                    "bin": (
                        bin_id
                    ),
                    "n_genes": (
                        n_genes
                    ),
                    "mean_median": (
                        mean_median
                    ),
                    "variance_median": (
                        variance_median
                    ),
                    "alpha_bin": (
                        alpha_bin
                    ),
                    "over_poisson": (
                        variance_median
                        > mean_median
                    ),
                }
            )

        bins = pd.DataFrame(
            rows
        )

        if bins.empty:
            raise ValueError(
                "No alpha-fit bins contained "
                "enough genes."
            )

        usable = bins.loc[
            bins[
                "over_poisson"
            ]
            & np.isfinite(
                bins[
                    "alpha_bin"
                ]
            )
            & (
                bins[
                    "alpha_bin"
                ]
                > 0
            )
        ].copy()

        if len(usable) >= (
            ALPHA_MIN_OVERPOISSON_BINS
        ):
            alpha = weighted_median(
                usable[
                    "alpha_bin"
                ],
                usable[
                    "n_genes"
                ],
            )

            method = (
                "weighted median of binned alpha"
            )

        else:
            gene_alpha = (
                variances
                - means
            ) / np.maximum(
                means ** 2,
                1e-30,
            )

            gene_alpha = gene_alpha[
                np.isfinite(
                    gene_alpha
                )
                & (
                    gene_alpha > 0
                )
            ]

            if len(gene_alpha) == 0:
                alpha = MIN_ALPHA
                method = (
                    "minimum-alpha fallback"
                )

            else:
                alpha = float(
                    np.median(
                        gene_alpha
                    )
                )

                method = (
                    "gene-level median fallback"
                )

        alpha = float(
            np.clip(
                alpha,
                MIN_ALPHA,
                MAX_ALPHA,
            )
        )

        pseudocount = (
            1.0
            / (
                4.0
                * alpha
            )
        )

        return {
            "alpha": alpha,
            "pseudocount": (
                pseudocount
            ),
            "bins": bins,
            "fit_method": method,
            "n_genes_fit": int(
                len(means)
            ),
            "n_usable_bins": int(
                len(usable)
            ),
        }


    def fit_fast_alpha(
        adata,
        source,
        control_indices,
        selected_indices,
    ):
        rng = np.random.default_rng(
            RANDOM_SEED
        )

        if len(control_indices) >= 100:
            alpha_cell_pool = (
                control_indices
            )

            alpha_source_name = (
                "control cells"
            )

        else:
            alpha_cell_pool = (
                selected_indices
            )

            alpha_source_name = (
                "selected cells"
            )

        n_alpha_cells = min(
            MAX_ALPHA_FIT_CELLS,
            len(alpha_cell_pool),
        )

        if n_alpha_cells == len(
            alpha_cell_pool
        ):
            alpha_cell_indices = (
                ensure_sorted_unique(
                    alpha_cell_pool
                )
            )

        else:
            alpha_cell_indices = np.sort(
                rng.choice(
                    alpha_cell_pool,
                    size=n_alpha_cells,
                    replace=False,
                )
            )

        n_alpha_genes = min(
            MAX_ALPHA_FIT_GENES,
            adata.n_vars,
        )

        if n_alpha_genes == adata.n_vars:
            alpha_gene_indices = np.arange(
                adata.n_vars,
                dtype=int,
            )

        else:
            alpha_gene_indices = np.sort(
                rng.choice(
                    adata.n_vars,
                    size=n_alpha_genes,
                    replace=False,
                )
            )

        print(
            f"[PFlog alpha] reading "
            f"{len(alpha_cell_indices):,} cells x "
            f"{len(alpha_gene_indices):,} genes"
        )

        alpha_matrix = read_count_block(
            adata,
            source,
            alpha_cell_indices,
            alpha_gene_indices,
        )

        alpha_stats = (
            matrix_mean_variance_detection(
                alpha_matrix
            )
        )

        alpha_result = (
            fit_alpha_from_binned_statistics(
                means=(
                    alpha_stats[
                        "mean"
                    ]
                ),
                variances=(
                    alpha_stats[
                        "variance"
                    ]
                ),
                detection=(
                    alpha_stats[
                        "detection_fraction"
                    ]
                ),
            )
        )

        alpha_result[
            "fit_cell_indices"
        ] = alpha_cell_indices

        alpha_result[
            "fit_gene_indices"
        ] = alpha_gene_indices

        alpha_result[
            "fit_cell_source"
        ] = alpha_source_name

        del alpha_matrix
        del alpha_stats

        gc.collect()

        return alpha_result


    def save_alpha_diagnostic(
        alpha_result,
        output_path,
        dataset_name,
    ):
        bins = alpha_result[
            "bins"
        ]

        alpha = alpha_result[
            "alpha"
        ]

        figure, axis = plt.subplots(
            figsize=(
                5.2,
                4.4,
            )
        )

        axis.scatter(
            bins[
                "mean_median"
            ],
            bins[
                "variance_median"
            ],
            s=np.clip(
                bins[
                    "n_genes"
                ],
                12,
                80,
            ),
            alpha=0.8,
            label="Binned medians",
        )

        positive = bins.loc[
            bins[
                "mean_median"
            ] > 0,
            "mean_median",
        ]

        if len(positive) > 0:
            grid = np.logspace(
                np.log10(
                    positive.min()
                ),
                np.log10(
                    positive.max()
                ),
                300,
            )

            axis.plot(
                grid,
                grid,
                linestyle="--",
                linewidth=1.2,
                label="Poisson",
            )

            axis.plot(
                grid,
                grid
                + alpha
                * grid ** 2,
                linewidth=1.5,
                label="NB fit",
            )

        axis.set_xscale(
            "log"
        )

        axis.set_yscale(
            "log"
        )

        axis.set_xlabel(
            "Gene mean"
        )

        axis.set_ylabel(
            "Gene variance"
        )

        axis.set_title(
            f"{dataset_name}\n"
            f"alpha={alpha:.4g}, "
            f"pseudocount="
            f"{alpha_result['pseudocount']:.4g}"
        )

        axis.legend(
            frameon=False,
            fontsize=8,
        )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=DPI,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )


    # ============================================================
    # ONE-PASS PFLOG ROW CENTER + GENE STATISTICS
    # ============================================================

    def fast_full_gene_pflog_statistics(
        adata,
        source,
        selected_indices,
        pseudocount,
    ):
        """
        One full-gene pass that calculates:

          1. raw gene means
          2. raw gene detection fractions
          3. exact all-gene PFlog row centers

        Sparse identity:

          log(x + p) =
              log(p) + [log(x + p) - log(p)]

        Only nonzero entries need the bracketed correction.
        """

        selected_indices = (
            ensure_sorted_unique(
                selected_indices
            )
        )

        n_cells = len(
            selected_indices
        )

        n_genes = int(
            adata.n_vars
        )

        gene_means = np.zeros(
            n_genes,
            dtype=np.float64,
        )

        gene_detection = np.zeros(
            n_genes,
            dtype=np.float64,
        )

        row_log_sums = np.zeros(
            n_cells,
            dtype=np.float64,
        )

        log_pseudocount = float(
            np.log(
                pseudocount
            )
        )

        n_chunks = int(
            np.ceil(
                n_genes
                / GENE_CHUNK_SIZE
            )
        )

        for chunk_number, start in enumerate(
            range(
                0,
                n_genes,
                GENE_CHUNK_SIZE,
            ),
            start=1,
        ):
            end = min(
                start
                + GENE_CHUNK_SIZE,
                n_genes,
            )

            print(
                f"\r[PFlog genome pass] "
                f"chunk {chunk_number}/{n_chunks}",
                end="",
                flush=True,
            )

            block = read_count_block(
                adata,
                source,
                selected_indices,
                slice(
                    start,
                    end,
                ),
            )

            width = (
                end - start
            )

            if sparse.issparse(
                block
            ):
                block = block.tocsr()

                gene_sums = np.asarray(
                    block.sum(
                        axis=0
                    )
                ).ravel()

                gene_means[
                    start:end
                ] = (
                    gene_sums
                    / n_cells
                )

                gene_detection[
                    start:end
                ] = (
                    np.asarray(
                        block.getnnz(
                            axis=0
                        )
                    ).ravel()
                    / n_cells
                )

                # Every zero contributes log(pseudocount).
                row_log_sums += (
                    width
                    * log_pseudocount
                )

                # Nonzero entries receive an additional correction.
                delta = block.copy()

                delta.data = (
                    np.log(
                        delta.data.astype(
                            np.float64,
                            copy=False,
                        )
                        + pseudocount
                    )
                    - log_pseudocount
                )

                row_log_sums += np.asarray(
                    delta.sum(
                        axis=1
                    )
                ).ravel()

                del delta

            else:
                block = np.asarray(
                    block,
                    dtype=np.float64,
                )

                block[
                    ~np.isfinite(
                        block
                    )
                ] = 0.0

                gene_means[
                    start:end
                ] = np.mean(
                    block,
                    axis=0,
                    dtype=np.float64,
                )

                gene_detection[
                    start:end
                ] = np.mean(
                    block > 0,
                    axis=0,
                )

                row_log_sums += np.sum(
                    np.log(
                        block
                        + pseudocount
                    ),
                    axis=1,
                    dtype=np.float64,
                )

            del block

        print()

        row_center = (
            row_log_sums
            / float(
                n_genes
            )
        )

        return {
            "gene_mean": (
                gene_means
            ),
            "gene_detection_fraction": (
                gene_detection
            ),
            "row_center": (
                row_center
            ),
        }


    # ============================================================
    # PCA GENE SELECTION
    # ============================================================

    def select_pflog_pca_genes(
        gene_means,
        gene_detection,
        var_names,
    ):
        gene_means = np.asarray(
            gene_means,
            dtype=float,
        )

        gene_detection = np.asarray(
            gene_detection,
            dtype=float,
        )

        valid = (
            np.isfinite(
                gene_means
            )
            & np.isfinite(
                gene_detection
            )
            & (
                gene_means > 0
            )
            & (
                gene_detection
                >= MIN_PCA_GENE_DETECTION_FRACTION
            )
        )

        candidates = np.flatnonzero(
            valid
        )

        if len(candidates) < 2:
            raise ValueError(
                "Too few genes passed the PFlog "
                "PCA-gene filters."
            )

        ranked = candidates[
            np.argsort(
                gene_means[
                    candidates
                ]
            )[::-1]
        ]

        selected_unsorted = ranked[
            :min(
                N_PFLOG_PCA_GENES,
                len(ranked),
            )
        ]

        selected = np.sort(
            selected_unsorted
        )

        return {
            "indices": (
                selected.astype(int)
            ),
            "genes": np.asarray(
                var_names[
                    selected
                ],
                dtype=object,
            ),
            "raw_mean": (
                gene_means[
                    selected
                ]
            ),
            "detection_fraction": (
                gene_detection[
                    selected
                ]
            ),
        }


    # ============================================================
    # MATERIALIZE ONLY SELECTED PFLOG GENES
    # ============================================================

    def make_dense_pflog_matrix(
        count_matrix,
        pseudocount,
        row_center,
    ):
        """
        Materialize shifted-log values only for selected PCA genes.
        """

        log_pseudocount = float(
            np.log(
                pseudocount
            )
        )

        if sparse.issparse(
            count_matrix
        ):
            matrix = (
                count_matrix
                .tocoo()
            )

            transformed = np.full(
                matrix.shape,
                log_pseudocount,
                dtype=np.float32,
            )

            transformed[
                matrix.row,
                matrix.col,
            ] = np.log(
                matrix.data.astype(
                    np.float64,
                    copy=False,
                )
                + pseudocount
            ).astype(
                np.float32
            )

        else:
            transformed = np.log(
                np.asarray(
                    count_matrix,
                    dtype=np.float64,
                )
                + pseudocount
            ).astype(
                np.float32
            )

        transformed -= np.asarray(
            row_center,
            dtype=np.float32,
        )[
            :,
            None,
        ]

        transformed[
            ~np.isfinite(
                transformed
            )
        ] = 0.0

        return transformed


    # ============================================================
    # PCA + UMAP
    # ============================================================

    def calculate_pflog_pca_umap(
        X_pflog,
    ):
        X_pflog = np.asarray(
            X_pflog,
            dtype=np.float32,
        )

        gene_variance = np.var(
            X_pflog,
            axis=0,
            ddof=1,
        )

        keep = (
            np.isfinite(
                gene_variance
            )
            & (
                gene_variance
                > MIN_PFLOG_VARIANCE
            )
        )

        X_use = X_pflog[
            :,
            keep,
        ]

        n_components = min(
            N_PCS,
            X_use.shape[0] - 1,
            X_use.shape[1],
        )

        if n_components < 2:
            raise ValueError(
                "Too few cells or variable PFlog genes "
                f"for PCA: {X_use.shape}"
            )

        print(
            f"[PCA] matrix={X_use.shape}; "
            f"components={n_components}"
        )

        # No StandardScaler:
        # PFlog itself is the variance-stabilizing transform.
        # PCA centers columns internally.
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            iterated_power=3,
            random_state=RANDOM_SEED,
        )

        X_pca = pca.fit_transform(
            X_use
        )

        print(
            f"[PCA] variance captured="
            f"{np.sum(pca.explained_variance_ratio_):.4f}"
        )

        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(
                    UMAP_N_NEIGHBORS,
                    max(
                        2,
                        X_pca.shape[0] - 1,
                    ),
                ),
                min_dist=UMAP_MIN_DIST,
                metric="euclidean",
                random_state=RANDOM_SEED,
                low_memory=True,
            )

            embedding = (
                reducer.fit_transform(
                    X_pca
                )
            )

            embedding_method = (
                "UMAP"
            )

        except ImportError:
            warnings.warn(
                "umap-learn is unavailable. "
                "Using PC1 and PC2."
            )

            embedding = X_pca[
                :,
                :2,
            ]

            embedding_method = (
                "PCA fallback"
            )

        return {
            "embedding": embedding,
            "embedding_method": (
                embedding_method
            ),
            "n_pcs": int(
                n_components
            ),
            "pca_variance_fraction": float(
                np.sum(
                    pca.explained_variance_ratio_
                )
            ),
            "gene_keep_mask": (
                keep
            ),
            "n_genes_after_variance_filter": int(
                X_use.shape[1]
            ),
        }


    # ============================================================
    # COVARIANCE PREPARATION
    # ============================================================

    def prepare_covariance_matrix(
        precomputed_genes,
        Sigma_true,
    ):
        diagonal = np.asarray(
            np.diag(
                Sigma_true
            ),
            dtype=float,
        )

        valid = np.flatnonzero(
            np.isfinite(
                diagonal
            )
        )

        if len(valid) == 0:
            raise ValueError(
                "Covariance diagonal contains "
                "no finite values."
            )

        n_keep = min(
            N_COVARIANCE_GENES,
            len(valid),
        )

        selected = valid[
            np.argsort(
                diagonal[
                    valid
                ]
            )[
                -n_keep:
            ]
        ]

        selected = selected[
            np.argsort(
                diagonal[
                    selected
                ]
            )[::-1]
        ]

        covariance = np.asarray(
            Sigma_true[
                np.ix_(
                    selected,
                    selected,
                )
            ],
            dtype=np.float64,
        )

        covariance = 0.5 * (
            covariance
            + covariance.T
        )

        off_diagonal = covariance[
            ~np.eye(
                covariance.shape[0],
                dtype=bool,
            )
        ]

        off_diagonal = off_diagonal[
            np.isfinite(
                off_diagonal
            )
        ]

        if len(off_diagonal) == 0:
            display_values = covariance[
                np.isfinite(
                    covariance
                )
            ]

        else:
            display_values = (
                off_diagonal
            )

        color_limit = float(
            np.quantile(
                np.abs(
                    display_values
                ),
                COVARIANCE_COLOR_QUANTILE,
            )
        )

        if (
            not np.isfinite(
                color_limit
            )
            or color_limit <= 0
        ):
            color_limit = float(
                np.nanmax(
                    np.abs(
                        covariance
                    )
                )
            )

        if (
            not np.isfinite(
                color_limit
            )
            or color_limit <= 0
        ):
            color_limit = 1.0

        return {
            "matrix": covariance,
            "indices": selected.astype(
                int
            ),
            "genes": np.asarray(
                precomputed_genes[
                    selected
                ],
                dtype=object,
            ),
            "variances": diagonal[
                selected
            ],
            "color_limit": (
                color_limit
            ),
        }


    # ============================================================
    # UMAP PLOTTING
    # ============================================================

    def make_category_color_map(labels):
        labels = np.asarray(
            labels,
            dtype=object,
        )

        counts = pd.Series(
            labels,
            dtype="object",
        ).value_counts()

        categories = (
            counts.index.tolist()
        )

        noncontrol = [
            category
            for category in categories
            if category != "Control"
        ]

        color_map = {}

        if "Control" in categories:
            color_map[
                "Control"
            ] = (
                0.18,
                0.18,
                0.18,
                1.0,
            )

        colors = plt.cm.tab10(
            np.linspace(
                0.0,
                0.8,
                max(
                    len(noncontrol),
                    1,
                ),
            )
        )

        for category, color in zip(
            noncontrol,
            colors,
        ):
            color_map[
                category
            ] = color

        return color_map, counts


    def draw_umap(
        axis,
        embedding,
        labels,
        n_pcs,
    ):
        labels = np.asarray(
            labels,
            dtype=object,
        )

        color_map, counts = (
            make_category_color_map(
                labels
            )
        )

        # Draw large populations first.
        plot_order = (
            counts
            .sort_values(
                ascending=False
            )
            .index
            .tolist()
        )

        for category in plot_order:
            mask = (
                labels == category
            )

            axis.scatter(
                embedding[
                    mask,
                    0,
                ],
                embedding[
                    mask,
                    1,
                ],
                s=UMAP_POINT_SIZE,
                alpha=UMAP_POINT_ALPHA,
                linewidths=0,
                color=color_map[
                    category
                ],
                rasterized=True,
            )

        legend_order = []

        if "Control" in counts.index:
            legend_order.append(
                "Control"
            )

        legend_order.extend(
            [
                category
                for category
                in counts.index
                if category != "Control"
            ]
        )

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=6,
                markerfacecolor=(
                    color_map[
                        category
                    ]
                ),
                markeredgecolor="none",
                label=(
                    f"{category} "
                    f"(n={int(counts[category]):,})"
                ),
            )
            for category in legend_order
        ]

        axis.legend(
            handles=handles,
            title="Cell category",
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            bbox_to_anchor=(
                1.02,
                1.0,
            ),
            loc="upper left",
            borderaxespad=0.0,
        )

        # No perturbation names are placed inside the UMAP.
        axis.set_title(
            "PFlog expression UMAP\n"
            f"first {n_pcs} PCs"
        )

        axis.set_xlabel(
            "UMAP 1"
        )

        axis.set_ylabel(
            "UMAP 2"
        )

        axis.set_xticks([])
        axis.set_yticks([])


    # ============================================================
    # PROCESS ONE DATASET
    # ============================================================

    def process_one_dataset(
        dataset_folder,
        indexed_h5ad_files,
    ):
        dataset_folder = Path(
            dataset_folder
        )

        dataset_name = (
            dataset_name_from_folder(
                dataset_folder
            )
        )

        print(
            "\n"
            + "=" * 110
        )

        print(
            f"DATASET: {dataset_name}"
        )

        print(
            f"FOLDER:  {os.path.basename(dataset_folder)}"
        )

        print(
            "=" * 110
        )

        h5ad_path = find_matching_h5ad(
            dataset_folder=dataset_folder,
            dataset_name=dataset_name,
            indexed_h5ad_files=(
                indexed_h5ad_files
            ),
        )

        if h5ad_path is None:
            raise FileNotFoundError(
                f"No matching h5ad file was found "
                f"for {dataset_name}."
            )

        print(
            f"[h5ad] {os.path.basename(h5ad_path)}"
        )

        precomputed = (
            load_precomputed_information(
                dataset_folder
            )
        )

        precomputed_genes = (
            precomputed[
                "genes"
            ]
        )

        Sigma_true = (
            precomputed[
                "Sigma_true"
            ]
        )

        output_directory = (
            OUT_ROOT
            / safe_filename(
                dataset_name
            )
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        adata = None

        try:
            adata = ad.read_h5ad(
                h5ad_path,
                backed="r",
            )

            print(
                f"[adata] shape={adata.shape}; "
                f"layers={list(adata.layers.keys())}"
            )

            count_source = (
                choose_count_source(
                    adata
                )
            )

            print(
                f"[counts] source="
                f"{count_source['name']}"
            )

            perturbation_column = (
                find_perturbation_column(
                    adata.obs
                )
            )

            if perturbation_column is None:
                raise ValueError(
                    "Could not identify a perturbation "
                    f"column. Available columns:\n"
                    f"{list(adata.obs.columns)}"
                )

            print(
                f"[metadata] perturbation column="
                f"{perturbation_column}"
            )

            all_raw_labels = (
                series_to_string_array(
                    adata.obs[
                        perturbation_column
                    ]
                )
            )

            cell_selection = (
                select_umap_cells(
                    raw_labels=all_raw_labels,
                    seed=RANDOM_SEED,
                )
            )

            selected_indices = (
                cell_selection[
                    "selected_indices"
                ]
            )

            selected_labels = (
                cell_selection[
                    "selected_labels"
                ]
            )

            collapsed_labels = (
                cell_selection[
                    "collapsed_labels"
                ]
            )

            top_perturbations = (
                cell_selection[
                    "top_perturbations"
                ]
            )

            sampling_summary = (
                cell_selection[
                    "summary"
                ]
            )

            control_indices = np.flatnonzero(
                collapsed_labels
                == "Control"
            )

            selected_raw_labels = (
                all_raw_labels[
                    selected_indices
                ]
            )

            selected_cell_names = np.asarray(
                adata.obs_names[
                    selected_indices
                ],
                dtype=object,
            )

            n_controls_selected = int(
                np.sum(
                    selected_labels
                    == "Control"
                )
            )

            print(
                f"[sample] selected "
                f"{len(selected_indices):,}/"
                f"{adata.n_obs:,} cells; "
                f"controls={n_controls_selected:,}"
            )

            print(
                "[sample] perturbations: "
                + ", ".join(
                    top_perturbations
                )
            )

            validate_count_source(
                adata=adata,
                source=count_source,
                row_indices=selected_indices,
            )

            sampling_path = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__umap_category_sampling.csv"
                )
            )

            sampling_summary.to_csv(
                sampling_path,
                index=False,
            )

            # ----------------------------------------------------
            # FAST ALPHA FIT
            # ----------------------------------------------------

            alpha_result = fit_fast_alpha(
                adata=adata,
                source=count_source,
                control_indices=(
                    control_indices
                ),
                selected_indices=(
                    selected_indices
                ),
            )

            alpha = alpha_result[
                "alpha"
            ]

            pseudocount = (
                alpha_result[
                    "pseudocount"
                ]
            )

            print(
                f"[PFlog alpha] "
                f"alpha={alpha:.6g}; "
                f"pseudocount={pseudocount:.6g}; "
                f"method={alpha_result['fit_method']}; "
                f"usable bins="
                f"{alpha_result['n_usable_bins']}"
            )

            alpha_bins_path = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__pflog_nb_bins.csv"
                )
            )

            alpha_result[
                "bins"
            ].to_csv(
                alpha_bins_path,
                index=False,
            )

            alpha_plot_path = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__pflog_nb_fit.png"
                )
            )

            save_alpha_diagnostic(
                alpha_result=(
                    alpha_result
                ),
                output_path=(
                    alpha_plot_path
                ),
                dataset_name=(
                    dataset_name
                ),
            )

            # ----------------------------------------------------
            # ONE FULL-GENE PASS
            # ----------------------------------------------------

            full_stats = (
                fast_full_gene_pflog_statistics(
                    adata=adata,
                    source=count_source,
                    selected_indices=(
                        selected_indices
                    ),
                    pseudocount=(
                        pseudocount
                    ),
                )
            )

            # ----------------------------------------------------
            # SELECT PCA GENES
            # ----------------------------------------------------

            pca_gene_selection = (
                select_pflog_pca_genes(
                    gene_means=(
                        full_stats[
                            "gene_mean"
                        ]
                    ),
                    gene_detection=(
                        full_stats[
                            "gene_detection_fraction"
                        ]
                    ),
                    var_names=(
                        count_source[
                            "var_names"
                        ]
                    ),
                )
            )

            print(
                f"[PFlog] selected "
                f"{len(pca_gene_selection['indices']):,} "
                "PCA genes"
            )

            # ----------------------------------------------------
            # READ ONLY SELECTED PCA GENES
            # ----------------------------------------------------

            selected_gene_counts = (
                read_count_block(
                    adata=adata,
                    source=count_source,
                    row_indices=(
                        selected_indices
                    ),
                    column_indices=(
                        pca_gene_selection[
                            "indices"
                        ]
                    ),
                )
            )

            X_pflog = (
                make_dense_pflog_matrix(
                    count_matrix=(
                        selected_gene_counts
                    ),
                    pseudocount=(
                        pseudocount
                    ),
                    row_center=(
                        full_stats[
                            "row_center"
                        ]
                    ),
                )
            )

            del selected_gene_counts

            # ----------------------------------------------------
            # PCA + UMAP
            # ----------------------------------------------------

            embedding_result = (
                calculate_pflog_pca_umap(
                    X_pflog
                )
            )

            embedding = (
                embedding_result[
                    "embedding"
                ]
            )

            # ----------------------------------------------------
            # COVARIANCE MATRIX
            # ----------------------------------------------------

            covariance_result = (
                prepare_covariance_matrix(
                    precomputed_genes=(
                        precomputed_genes
                    ),
                    Sigma_true=(
                        Sigma_true
                    ),
                )
            )

            covariance_matrix = (
                covariance_result[
                    "matrix"
                ]
            )

            covariance_genes = (
                covariance_result[
                    "genes"
                ]
            )

            color_limit = (
                covariance_result[
                    "color_limit"
                ]
            )

            # ----------------------------------------------------
            # DRAW FIGURE
            # ----------------------------------------------------

            figure, axes = plt.subplots(
                1,
                2,
                figsize=(
                    19,
                    8,
                ),
                gridspec_kw={
                    "width_ratios": [
                        1.0,
                        1.2,
                    ]
                },
            )

            covariance_axis = axes[0]
            umap_axis = axes[1]

            covariance_image = (
                covariance_axis.imshow(
                    covariance_matrix,
                    aspect="equal",
                    interpolation="nearest",
                    cmap="coolwarm",
                    vmin=-color_limit,
                    vmax=color_limit,
                )
            )

            covariance_ticks = (
                even_tick_indices(
                    len(
                        covariance_genes
                    ),
                    MAX_COVARIANCE_TICK_LABELS,
                )
            )

            covariance_axis.set_xticks(
                covariance_ticks
            )

            covariance_axis.set_yticks(
                covariance_ticks
            )

            covariance_axis.set_xticklabels(
                covariance_genes[
                    covariance_ticks
                ],
                rotation=90,
                fontsize=7,
            )

            covariance_axis.set_yticklabels(
                covariance_genes[
                    covariance_ticks
                ],
                fontsize=7,
            )

            covariance_axis.set_xlabel(
                "genes"
            )

            covariance_axis.set_ylabel(
                "genes"
            )

            covariance_axis.set_title(
                "True covariance matrix\n"
                f"top {len(covariance_genes)} "
                "variance genes"
            )

            figure.colorbar(
                covariance_image,
                ax=covariance_axis,
                fraction=0.046,
                pad=0.03,
                label="covariance",
            )

            draw_umap(
                axis=umap_axis,
                embedding=embedding,
                labels=selected_labels,
                n_pcs=(
                    embedding_result[
                        "n_pcs"
                    ]
                ),
            )

            figure.suptitle(
                f"{dataset_name}\n"
                f"PFlog alpha={alpha:.4g}; "
                f"pseudocount={pseudocount:.4g}",
                fontsize=16,
                y=0.995,
            )

            figure.tight_layout(
                rect=[
                    0.0,
                    0.0,
                    0.88,
                    0.95,
                ]
            )

            output_stem = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__true_covariance_and_pflog_umap"
                )
            )

            png_path = Path(
                str(output_stem)
                + ".png"
            )

            svg_path = Path(
                str(output_stem)
                + ".svg"
            )

            figure.savefig(
                png_path,
                dpi=DPI,
                bbox_inches="tight",
            )

            figure.savefig(
                svg_path,
                bbox_inches="tight",
            )

            plt.show()

            plt.close(
                figure
            )

            # ----------------------------------------------------
            # SAVE UMAP
            # ----------------------------------------------------

            umap_dataframe = pd.DataFrame(
                {
                    "cell_index": (
                        selected_indices
                    ),
                    "cell_name": (
                        selected_cell_names
                    ),
                    "original_perturbation_label": (
                        selected_raw_labels
                    ),
                    "plot_label": (
                        selected_labels
                    ),
                    "is_control": (
                        selected_labels
                        == "Control"
                    ),
                    "embedding_1": (
                        embedding[
                            :,
                            0,
                        ]
                    ),
                    "embedding_2": (
                        embedding[
                            :,
                            1,
                        ]
                    ),
                    "embedding_method": (
                        embedding_result[
                            "embedding_method"
                        ]
                    ),
                    "normalization": (
                        "PFlog"
                    ),
                    "pflog_alpha": (
                        alpha
                    ),
                    "pflog_pseudocount": (
                        pseudocount
                    ),
                    "n_pcs_used": (
                        embedding_result[
                            "n_pcs"
                        ]
                    ),
                }
            )

            umap_path = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__umap_coordinates.csv"
                )
            )

            umap_dataframe.to_csv(
                umap_path,
                index=False,
            )

            # ----------------------------------------------------
            # SAVE PCA GENES
            # ----------------------------------------------------

            pca_gene_dataframe = pd.DataFrame(
                {
                    "adata_gene_index": (
                        pca_gene_selection[
                            "indices"
                        ]
                    ),
                    "gene": (
                        pca_gene_selection[
                            "genes"
                        ]
                    ),
                    "raw_mean": (
                        pca_gene_selection[
                            "raw_mean"
                        ]
                    ),
                    "detection_fraction": (
                        pca_gene_selection[
                            "detection_fraction"
                        ]
                    ),
                    "retained_after_pflog_variance_filter": (
                        embedding_result[
                            "gene_keep_mask"
                        ]
                    ),
                }
            )

            pca_gene_path = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__pflog_pca_genes.csv"
                )
            )

            pca_gene_dataframe.to_csv(
                pca_gene_path,
                index=False,
            )

            # ----------------------------------------------------
            # SAVE COVARIANCE GENES
            # ----------------------------------------------------

            covariance_gene_dataframe = (
                pd.DataFrame(
                    {
                        "gene_index": (
                            covariance_result[
                                "indices"
                            ]
                        ),
                        "gene": (
                            covariance_result[
                                "genes"
                            ]
                        ),
                        "variance": (
                            covariance_result[
                                "variances"
                            ]
                        ),
                    }
                )
            )

            covariance_gene_path = (
                output_directory
                / (
                    safe_filename(
                        dataset_name
                    )
                    + "__covariance_genes.csv"
                )
            )

            covariance_gene_dataframe.to_csv(
                covariance_gene_path,
                index=False,
            )

            print(
                f"[saved] {os.path.basename(png_path)}"
            )

            print(
                f"[saved] {os.path.basename(umap_path)}"
            )

            result = {
                "dataset": (
                    dataset_name
                ),
                "status": "ok",
                "h5ad_path": str(
                    h5ad_path
                ),
                "counts_source": (
                    count_source[
                        "name"
                    ]
                ),
                "perturbation_column": (
                    perturbation_column
                ),
                "n_cells_total": int(
                    adata.n_obs
                ),
                "n_cells_umap": int(
                    len(
                        selected_indices
                    )
                ),
                "n_controls_total": int(
                    len(
                        control_indices
                    )
                ),
                "n_controls_umap": int(
                    n_controls_selected
                ),
                "top_perturbations": (
                    "|".join(
                        top_perturbations
                    )
                ),
                "alpha_fit_source": (
                    alpha_result[
                        "fit_cell_source"
                    ]
                ),
                "n_alpha_fit_cells": int(
                    len(
                        alpha_result[
                            "fit_cell_indices"
                        ]
                    )
                ),
                "n_alpha_fit_genes_sampled": int(
                    len(
                        alpha_result[
                            "fit_gene_indices"
                        ]
                    )
                ),
                "n_alpha_fit_genes_used": (
                    alpha_result[
                        "n_genes_fit"
                    ]
                ),
                "alpha_fit_method": (
                    alpha_result[
                        "fit_method"
                    ]
                ),
                "pflog_alpha": (
                    alpha
                ),
                "pflog_pseudocount": (
                    pseudocount
                ),
                "n_pca_genes_selected": int(
                    len(
                        pca_gene_selection[
                            "indices"
                        ]
                    )
                ),
                "n_pca_genes_after_variance_filter": (
                    embedding_result[
                        "n_genes_after_variance_filter"
                    ]
                ),
                "n_pcs_used": (
                    embedding_result[
                        "n_pcs"
                    ]
                ),
                "pca_variance_fraction": (
                    embedding_result[
                        "pca_variance_fraction"
                    ]
                ),
                "embedding_method": (
                    embedding_result[
                        "embedding_method"
                    ]
                ),
                "png": str(
                    png_path
                ),
                "svg": str(
                    svg_path
                ),
                "umap_csv": str(
                    umap_path
                ),
                "sampling_csv": str(
                    sampling_path
                ),
                "alpha_bins_csv": str(
                    alpha_bins_path
                ),
                "alpha_plot": str(
                    alpha_plot_path
                ),
                "pca_genes_csv": str(
                    pca_gene_path
                ),
                "covariance_genes_csv": str(
                    covariance_gene_path
                ),
            }

            del X_pflog
            del embedding
            del full_stats

            gc.collect()

            return result

        finally:
            if adata is not None:
                try:
                    adata.file.close()
                except Exception:
                    pass

            del Sigma_true

            gc.collect()


    # ============================================================
    # RUN
    # ============================================================

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_dataset_folders = (
        find_precomputed_dataset_folders(
            root=PRECOMPUTE_ROOT,
            expression_threshold=(
                EXPRESSION_THRESHOLD
            ),
        )
    )

    dataset_folders = (
        select_dataset_folders(
            folders=(
                all_dataset_folders
            ),
            dataset_queries=(
                DATASET_QUERIES
            ),
        )
    )

    if len(
        dataset_folders
    ) == 0:
        raise FileNotFoundError(
            "No requested precomputed dataset "
            f"folders were found under:\n"
            f"{PRECOMPUTE_ROOT}"
        )

    print(
        f"[datasets] selected "
        f"{len(dataset_folders)} datasets"
    )

    print(
        "[h5ad] indexing h5ad files once..."
    )

    indexed_h5ad_files = (
        index_h5ad_files(
            H5AD_SEARCH_ROOTS
        )
    )

    print(
        f"[h5ad] found "
        f"{len(indexed_h5ad_files)} h5ad files"
    )

    results = []

    for dataset_number, dataset_folder in enumerate(
        dataset_folders,
        start=1,
    ):
        print(
            f"\n[{dataset_number}/"
            f"{len(dataset_folders)}]"
        )

        try:
            result = (
                process_one_dataset(
                    dataset_folder=(
                        dataset_folder
                    ),
                    indexed_h5ad_files=(
                        indexed_h5ad_files
                    ),
                )
            )

            results.append(
                result
            )

        except Exception as error:
            dataset_name = (
                dataset_name_from_folder(
                    dataset_folder
                )
            )

            print(
                "\n"
                + "!" * 110
            )

            print(
                f"[ERROR] {dataset_name}"
            )

            print(
                repr(error)
            )

            print(
                "!" * 110
            )

            results.append(
                {
                    "dataset": (
                        dataset_name
                    ),
                    "status": "error",
                    "precomputed_folder": str(
                        dataset_folder
                    ),
                    "error": repr(
                        error
                    ),
                }
            )

            gc.collect()


    # ============================================================
    # SAVE SUMMARY
    # ============================================================

    summary_dataframe = pd.DataFrame(
        results
    )

    summary_path = (
        OUT_ROOT
        / "dataset_processing_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    n_success = int(
        np.sum(
            summary_dataframe[
                "status"
            ]
            == "ok"
        )
    )

    n_errors = int(
        np.sum(
            summary_dataframe[
                "status"
            ]
            == "error"
        )
    )

    print(
        "\n"
        + "=" * 110
    )

    print("DONE")

    print(
        f"Successful datasets: "
        f"{n_success}"
    )

    print(
        f"Errored datasets:    "
        f"{n_errors}"
    )

    print(
        f"Summary:             "
        f"{os.path.basename(summary_path)}"
    )

    print(
        f"Figures:             "
        f"{os.path.basename(OUT_ROOT)}"
    )

    print(
        "=" * 110
    )

    if n_errors > 0:
        error_columns = [
            column
            for column in [
                "dataset",
                "error",
            ]
            if column
            in summary_dataframe.columns
        ]

        print(
            "\nDatasets with errors:"
        )

        print(
            summary_dataframe.loc[
                summary_dataframe[
                    "status"
                ]
                == "error",
                error_columns,
            ].to_string(
                index=False
            )
        )

