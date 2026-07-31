"""Forward-problem metric recomputation + figures for Fig S16 (notebook-only).

Provides the authoritative function definitions, including
``print_aggregate_metric_summary`` and the aggregate-printing
variant of ``plot_combined_metrics``. NOT part of the installable ``cipher``
package -- a notebook-only helper for reproducing the supplementary figure.

Refits CIPHER's first-order forward map (dx ~= a * Sigma[:, target]) in each
normalization's native space, pushes the prediction back to raw-count response
units, and scores per-perturbation metrics (Pearson, uncentered R^2, sign accuracy,
...) across datasets and normalization modes, then renders the per-dataset and
pooled composite panels.

The module keeps the CONFIG constants that the functions reference as module-level
globals (so ``from src.suppl_forward6 import *`` resolves them); only the data/output
PATHS (PRECOMPUTE_ROOT_CANDIDATES, OUTDIR, ...) are set in the reproduction notebook
so they resolve under $CIPHER_DATA_DIR/suppl and the repro output directory.
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
import gc
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import h5py
from tqdm.auto import tqdm


# ============================================================
# RECOMPUTE FORWARD METRICS + MAKE DATASET-SPECIFIC FIGURES
#
# Dataset selection:
#   - retain every non-Marson dataset
#   - retain only one Marson dataset
#   - by default, retain the alphabetically first Marson dataset
#
# Plotting:
#   - one combined figure per retained dataset
#   - each figure contains:
#         Pearson dot/box plot
#         Pearson histograms by normalization
#         uncentered R² dot/box plot
#         uncentered R² histograms by normalization
#   - one pooled composite figure is produced after all
#     dataset-specific figures
#
# The same normalization order is used in every figure.
# By default, modes are ordered by pooled mean Pearson.
# ============================================================




# ============================================================
# CONFIG
# ============================================================





# ------------------------------------------------------------
# Normalizations
# ------------------------------------------------------------

MODES = [
    "raw",
    "log1p",
    "log1CP10k",
    "frequency",
    "pflog",
]

MODE_LABELS = {
    "raw": "Raw",
    "log1p": "log1p",
    "log1CP10k": "Library-size 10k",
    "frequency": "Frequency",
    "pflog": "Lior/PFlog",
}

MODE_COLORS = {
    "raw": "tab:blue",
    "log1p": "tab:red",
    "log1CP10k": "tab:green",
    "frequency": "tab:purple",
    "pflog": "tab:orange",
}


# ------------------------------------------------------------
# Dataset selection
# ------------------------------------------------------------

MARSON_NAME_TOKEN = "marson"

# None means:
#     retain the alphabetically first Marson dataset.
#
# To force a specific Marson dataset, use its exact directory name:
#
# MARSON_DATASET_TO_KEEP = "Marson2025_D1_Rest8hr"
#
MARSON_DATASET_TO_KEEP = None


# ------------------------------------------------------------
# Train/test split over genes
# ------------------------------------------------------------

HOLDOUT_FRAC = 0.5
SPLIT_SEED = 0

EXCLUDE_TARGET_GENE_FROM_FIT = False
EXCLUDE_TARGET_GENE_FROM_EVAL = False

MIN_SIGMA_COL_NORM2 = 1e-20
MIN_TRAIN_GENES = 50
MIN_TEST_GENES = 50


# ------------------------------------------------------------
# Sign metrics
# ------------------------------------------------------------

SIGN_EPS_TRUE = 0.0
SIGN_EPS_PRED = 0.0


# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------

MMAP_SIGMA = True
TQDM_NCOLS = 110

# ------------------------------------------------------------
# First-order pushback to raw-count response space
# ------------------------------------------------------------

# All methods are fit in their native space, but predictions and
# observed responses are compared in raw-count response units.
EVALUATE_ALL_MODES_IN_RAW_SPACE = True

# log1CP10k uses log(1 + 10,000 * x / library_size).
CP10K_SCALE = 10_000.0

# When no saved mean library size is available, use the sum of the
# retained-gene raw control means. This is exact only if the retained
# genes comprise the library used during normalization.
ALLOW_LIBRARY_SIZE_FROM_RETAINED_GENES = True

# PFlog uses pc = 1 / (4 * alpha). The code first searches saved HDF5
# datasets/attributes and JSON metadata for pc or alpha. Set a numeric
# fallback only if the precompute did not save either quantity.
PFLOG_PSEUDOCOUNT_FALLBACK = None


# ------------------------------------------------------------
# Figure settings
# ------------------------------------------------------------

DPI = 300
DISPLAY_RANGE = (-1.0, 1.0)

DOT_ALPHA = 0.18
DOT_SIZE = 12

DATASET_MEAN_DOT_SIZE = 27
GRAND_MEAN_SIZE = 115

JITTER_WIDTH = 0.12
BOX_WIDTH = 0.58
LINE_ALPHA = 0.16

RANDOM_DOT_SEED = 1

HIST_BINS = 70
HIST_YMIN = 1e-2

FIGSIZE_BASE_WIDTH_PER_MODE = 2.35
FIG_HEIGHT = 15.0

SAVE_SVG = True
SAVE_PNG = True

# Set False if 18 large figures are too much notebook output.
SHOW_FIGS = True

# Use pooled Pearson to define one common mode order for all plots.
# Set to None to preserve the order in MODES.
ORDER_MODES_BY_METRIC = "pearson"


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._")
    return value or "unnamed"


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return str(obj)


def stable_int_seed(*parts, base_seed=0):
    text = "::".join(
        [str(part) for part in parts] + [str(base_seed)]
    )

    digest = hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()

    return int(
        int(digest[:8], 16) % (2**32 - 1)
    )


def choose_precompute_root(candidates):
    for root in candidates:
        root = Path(root)

        if not root.exists():
            continue

        dataset_dirs = [
            dataset_dir
            for dataset_dir in root.iterdir()
            if (
                dataset_dir.is_dir()
                and (dataset_dir / "normalizations").exists()
            )
        ]

        if dataset_dirs:
            return root

    raise FileNotFoundError(
        "Could not find a usable precompute root. Checked:\n"
        + "\n".join(str(path) for path in candidates)
    )


def discover_dataset_dirs(root):
    root = Path(root)

    return sorted(
        [
            dataset_dir
            for dataset_dir in root.iterdir()
            if (
                dataset_dir.is_dir()
                and (dataset_dir / "normalizations").exists()
            )
        ],
        key=lambda path: path.name.lower(),
    )


def select_dataset_dirs(dataset_dirs):
    """
    Retain all non-Marson datasets and only one Marson dataset.
    """
    dataset_dirs = sorted(
        [Path(path) for path in dataset_dirs],
        key=lambda path: path.name.lower(),
    )

    token = str(MARSON_NAME_TOKEN).lower()

    marson_dirs = [
        path
        for path in dataset_dirs
        if token in path.name.lower()
    ]

    non_marson_dirs = [
        path
        for path in dataset_dirs
        if token not in path.name.lower()
    ]

    kept_marson = None

    if marson_dirs:
        if MARSON_DATASET_TO_KEEP is None:
            kept_marson = marson_dirs[0]

        else:
            requested = str(MARSON_DATASET_TO_KEEP)

            exact_matches = [
                path
                for path in marson_dirs
                if path.name == requested
            ]

            case_insensitive_matches = [
                path
                for path in marson_dirs
                if path.name.lower() == requested.lower()
            ]

            if len(exact_matches) == 1:
                kept_marson = exact_matches[0]

            elif len(case_insensitive_matches) == 1:
                kept_marson = case_insensitive_matches[0]

            else:
                raise ValueError(
                    "MARSON_DATASET_TO_KEEP did not uniquely match "
                    "a discovered Marson dataset.\n"
                    f"Requested: {requested!r}\n"
                    "Available Marson datasets:\n"
                    + "\n".join(
                        f"  - {path.name}"
                        for path in marson_dirs
                    )
                )

    selected = list(non_marson_dirs)

    if kept_marson is not None:
        selected.append(kept_marson)

    selected = sorted(
        selected,
        key=lambda path: path.name.lower(),
    )

    excluded_marson_dirs = [
        path
        for path in marson_dirs
        if path != kept_marson
    ]

    print("\n" + "=" * 120)
    print("DATASET SELECTION")
    print("=" * 120)

    print(f"All discovered datasets:       {len(dataset_dirs)}")
    print(f"Non-Marson datasets retained:  {len(non_marson_dirs)}")
    print(f"Marson datasets discovered:    {len(marson_dirs)}")

    if kept_marson is None:
        print("Marson dataset retained:        none found")
    else:
        print(f"Marson dataset retained:        {kept_marson.name}")

    print(f"Marson datasets excluded:      {len(excluded_marson_dirs)}")

    for path in excluded_marson_dirs:
        print(f"  - {path.name}")

    print(f"Total datasets retained:       {len(selected)}")
    print("=" * 120)

    return selected


def safe_load_strings_npy(path):
    values = np.load(
        path,
        allow_pickle=True,
    )

    return np.asarray(values).astype(str)


def safe_read_h5_strings(h5_file, key):
    values = h5_file[key][:]

    output = []

    for value in values:
        if isinstance(value, bytes):
            output.append(
                value.decode("utf-8")
            )
        else:
            output.append(
                str(value)
            )

    return np.asarray(
        output,
        dtype=str,
    )


def load_dataset_level_files(dataset_dir):
    dataset_dir = Path(dataset_dir)

    genes_path = dataset_dir / "genes.npy"
    perturbations_path = dataset_dir / "perturbations.npy"
    target_indices_path = dataset_dir / "target_gene_indices.npy"
    target_genes_path = dataset_dir / "target_genes.npy"

    if not genes_path.exists():
        raise FileNotFoundError(
            f"Missing genes.npy: {genes_path}"
        )

    if not perturbations_path.exists():
        raise FileNotFoundError(
            f"Missing perturbations.npy: {perturbations_path}"
        )

    if not target_indices_path.exists():
        raise FileNotFoundError(
            f"Missing target_gene_indices.npy: {target_indices_path}"
        )

    genes = safe_load_strings_npy(
        genes_path
    )

    perturbations = safe_load_strings_npy(
        perturbations_path
    )

    target_gene_indices = np.load(
        target_indices_path
    ).astype(np.int64)

    if target_genes_path.exists():
        target_genes = safe_load_strings_npy(
            target_genes_path
        )

    else:
        target_genes = np.asarray(
            [
                (
                    genes[int(index)]
                    if 0 <= int(index) < len(genes)
                    else ""
                )
                for index in target_gene_indices
            ],
            dtype=str,
        )

    if len(target_genes) != len(perturbations):
        target_genes = np.asarray(
            [
                (
                    genes[int(index)]
                    if 0 <= int(index) < len(genes)
                    else ""
                )
                for index in target_gene_indices
            ],
            dtype=str,
        )

    if len(perturbations) != len(target_gene_indices):
        raise ValueError(
            f"Length mismatch in {dataset_dir}: "
            f"perturbations={len(perturbations)}, "
            f"target_gene_indices={len(target_gene_indices)}"
        )

    perturbation_to_target_index = {
        str(perturbation): int(index)
        for perturbation, index in zip(
            perturbations,
            target_gene_indices,
        )
    }

    perturbation_to_target_gene = {
        str(perturbation): str(gene)
        for perturbation, gene in zip(
            perturbations,
            target_genes,
        )
    }

    return {
        "genes": genes,
        "perturbations": perturbations,
        "target_genes": target_genes,
        "target_gene_indices": target_gene_indices,
        "pert_to_target_idx": perturbation_to_target_index,
        "pert_to_target_gene": perturbation_to_target_gene,
    }


def mode_files_ready(mode_dir):
    mode_dir = Path(mode_dir)

    sigma_path = mode_dir / "Sigma_full_ridge.npy"
    stats_path = mode_dir / "perturbation_stats.h5"
    temporary_sigma_path = mode_dir / "Sigma_full_ridge.npy.tmp"

    if not mode_dir.exists():
        return (
            False,
            "missing_mode_dir",
            sigma_path,
            stats_path,
        )

    if temporary_sigma_path.exists():
        return (
            False,
            "sigma_tmp_exists_still_writing",
            sigma_path,
            stats_path,
        )

    if not sigma_path.exists():
        return (
            False,
            "missing_sigma",
            sigma_path,
            stats_path,
        )

    if not stats_path.exists():
        return (
            False,
            "missing_stats",
            sigma_path,
            stats_path,
        )

    return (
        True,
        "ready",
        sigma_path,
        stats_path,
    )


def load_sigma(sigma_path):
    mmap_mode = "r" if MMAP_SIGMA else None

    return np.load(
        sigma_path,
        mmap_mode=mmap_mode,
    )


# ============================================================
# RAW-SPACE REFERENCE AND FIRST-ORDER PUSHBACK HELPERS
# ============================================================

def _read_first_h5_scalar_or_vector(h5_file, names):
    """Return the first matching HDF5 dataset/attribute, or None."""
    for name in names:
        if name in h5_file:
            value = np.asarray(h5_file[name][:] if h5_file[name].shape else h5_file[name][()])
            return value

        if name in h5_file.attrs:
            return np.asarray(h5_file.attrs[name])

    return None


def _search_json_metadata(paths, keys):
    """Search a small set of JSON files for the first matching key."""
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue

        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue

        stack = [payload]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for key, value in current.items():
                    if str(key).lower() in keys:
                        return value
                    stack.append(value)
            elif isinstance(current, list):
                stack.extend(current)

    return None


def load_raw_reference_information(dataset_dir, n_genes):
    """
    Load the raw control mean, raw perturbation responses, and a reference
    mean library size used for the first-order inverse normalizations.
    """
    dataset_dir = Path(dataset_dir)
    raw_mode_dir = dataset_dir / "normalizations" / "raw"
    raw_stats_path = raw_mode_dir / "perturbation_stats.h5"

    if not raw_stats_path.exists():
        raise FileNotFoundError(
            f"Raw perturbation statistics are required for raw-space "
            f"evaluation but were not found: {raw_stats_path}"
        )

    control_mean_file_candidates = [
        dataset_dir / "raw_control_mean.npy",
        dataset_dir / "control_mean_raw.npy",
        raw_mode_dir / "control_mean.npy",
        raw_mode_dir / "mean_control.npy",
        raw_mode_dir / "mu_control.npy",
    ]

    raw_control_mean = None
    control_mean_source = None

    for path in control_mean_file_candidates:
        if path.exists():
            candidate = np.asarray(np.load(path), dtype=np.float64).ravel()
            if candidate.size == n_genes:
                raw_control_mean = candidate
                control_mean_source = str(path)
                break

    with h5py.File(raw_stats_path, "r") as raw_h5:
        if raw_control_mean is None:
            candidate = _read_first_h5_scalar_or_vector(
                raw_h5,
                [
                    "control_mean",
                    "mean_control",
                    "raw_control_mean",
                    "mu_control",
                    "x0_mean",
                ],
            )

            if candidate is not None:
                candidate = np.asarray(candidate, dtype=np.float64).ravel()
                if candidate.size == n_genes:
                    raw_control_mean = candidate
                    control_mean_source = f"{raw_stats_path}:HDF5"

        library_value = _read_first_h5_scalar_or_vector(
            raw_h5,
            [
                "mean_control_library_size",
                "control_mean_library_size",
                "mean_library_size_control",
                "mean_library_size",
                "library_size_mean",
            ],
        )

    if raw_control_mean is None:
        raise KeyError(
            "Could not locate a raw control-mean vector. Save one as "
            "raw_control_mean.npy / control_mean_raw.npy, or store an HDF5 "
            "dataset named control_mean in the raw perturbation_stats.h5."
        )

    if not np.all(np.isfinite(raw_control_mean)):
        raise ValueError("The raw control-mean vector contains nonfinite values.")

    mean_library_size = None
    library_size_source = None

    if library_value is not None:
        library_array = np.asarray(library_value, dtype=np.float64).ravel()
        if library_array.size >= 1 and np.isfinite(library_array[0]):
            mean_library_size = float(library_array[0])
            library_size_source = f"{raw_stats_path}:HDF5"

    if mean_library_size is None:
        json_value = _search_json_metadata(
            [
                dataset_dir / "metadata.json",
                dataset_dir / "precompute_metadata.json",
                raw_mode_dir / "metadata.json",
                raw_mode_dir / "normalization_metadata.json",
            ],
            {
                "mean_control_library_size",
                "control_mean_library_size",
                "mean_library_size_control",
                "mean_library_size",
            },
        )

        if json_value is not None:
            try:
                mean_library_size = float(json_value)
                library_size_source = "JSON metadata"
            except Exception:
                mean_library_size = None

    if mean_library_size is None and ALLOW_LIBRARY_SIZE_FROM_RETAINED_GENES:
        mean_library_size = float(np.sum(raw_control_mean))
        library_size_source = "sum(raw_control_mean over retained genes)"

    if mean_library_size is None or not np.isfinite(mean_library_size) or mean_library_size <= 0:
        raise ValueError(
            "A positive mean control library size is required for frequency "
            "and log1CP10k pushback."
        )

    return {
        "raw_stats_path": raw_stats_path,
        "raw_control_mean": raw_control_mean,
        "control_mean_source": control_mean_source,
        "mean_library_size": mean_library_size,
        "library_size_source": library_size_source,
    }


def load_pflog_pseudocount(dataset_dir, mode_dir, stats_path):
    """Load PFlog pc directly, or infer pc = 1/(4 alpha)."""
    direct_pc_names = [
        "pflog_pseudocount",
        "pseudocount",
        "pc",
    ]
    alpha_names = [
        "pflog_alpha",
        "alpha",
        "nb_alpha",
        "dispersion_alpha",
    ]

    direct_pc = None
    alpha = None

    with h5py.File(stats_path, "r") as h5_file:
        direct_pc = _read_first_h5_scalar_or_vector(h5_file, direct_pc_names)
        alpha = _read_first_h5_scalar_or_vector(h5_file, alpha_names)

    if direct_pc is not None:
        value = float(np.asarray(direct_pc).ravel()[0])
        if np.isfinite(value) and value > 0:
            return value, f"{stats_path}:pc"

    if alpha is not None:
        value = float(np.asarray(alpha).ravel()[0])
        if np.isfinite(value) and value > 0:
            return 1.0 / (4.0 * value), f"{stats_path}:alpha"

    json_paths = [
        Path(mode_dir) / "metadata.json",
        Path(mode_dir) / "normalization_metadata.json",
        Path(dataset_dir) / "metadata.json",
        Path(dataset_dir) / "precompute_metadata.json",
    ]

    direct_pc = _search_json_metadata(
        json_paths,
        {name.lower() for name in direct_pc_names},
    )

    if direct_pc is not None:
        try:
            value = float(direct_pc)
            if np.isfinite(value) and value > 0:
                return value, "JSON:pc"
        except Exception:
            pass

    alpha = _search_json_metadata(
        json_paths,
        {name.lower() for name in alpha_names},
    )

    if alpha is not None:
        try:
            value = float(alpha)
            if np.isfinite(value) and value > 0:
                return 1.0 / (4.0 * value), "JSON:alpha"
        except Exception:
            pass

    if PFLOG_PSEUDOCOUNT_FALLBACK is not None:
        value = float(PFLOG_PSEUDOCOUNT_FALLBACK)
        if np.isfinite(value) and value > 0:
            return value, "PFLOG_PSEUDOCOUNT_FALLBACK"

    raise KeyError(
        "PFlog pushback requires its pseudocount pc or fitted alpha. "
        "Save pc/alpha in the PFlog HDF5 or JSON metadata, or set "
        "PFLOG_PSEUDOCOUNT_FALLBACK explicitly."
    )


def first_order_native_shift_to_raw(
    native_shift,
    mode,
    raw_control_mean,
    mean_library_size,
    pflog_pseudocount=None,
):
    """
    First-order inverse-normalization map around the raw control mean.

    raw:
        dx_raw = dy

    log1p, y = log(1 + x):
        dx_raw ~= (1 + <x0>) * dy

    log1CP10k, y = log(1 + CP10K_SCALE * x / L0):
        dx_raw ~= (<x0> + L0 / CP10K_SCALE) * dy

    frequency, y = x / L0:
        dx_raw ~= L0 * dy

    PFlog, y_j = log(x_j + pc) - gene-wise log center:
        dx_raw ~= (<x0_j> + pc) * dy_j
        under the fixed-global-center approximation.
    """
    native_shift = np.asarray(native_shift, dtype=np.float64)
    raw_control_mean = np.asarray(raw_control_mean, dtype=np.float64)

    if native_shift.shape != raw_control_mean.shape:
        raise ValueError(
            f"Shape mismatch: native_shift={native_shift.shape}, "
            f"raw_control_mean={raw_control_mean.shape}"
        )

    if mode == "raw":
        scale = np.ones_like(raw_control_mean)

    elif mode == "log1p":
        scale = 1.0 + raw_control_mean

    elif mode == "log1CP10k":
        scale = raw_control_mean + float(mean_library_size) / CP10K_SCALE

    elif mode == "frequency":
        scale = np.full_like(raw_control_mean, float(mean_library_size))

    elif mode == "pflog":
        if pflog_pseudocount is None:
            raise ValueError("pflog_pseudocount is required for PFlog pushback.")
        scale = raw_control_mean + float(pflog_pseudocount)

    else:
        raise ValueError(f"No first-order raw pushback is defined for mode={mode!r}")

    return scale * native_shift, scale


# ============================================================
# METRIC HELPERS
# ============================================================

def safe_pearson(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    ).ravel()

    y = np.asarray(
        y,
        dtype=np.float64,
    ).ravel()

    finite = np.isfinite(x) & np.isfinite(y)

    x = x[finite]
    y = y[finite]

    if x.size < 3:
        return np.nan

    x_std = float(np.std(x))
    y_std = float(np.std(y))

    if x_std <= 0 or y_std <= 0:
        return np.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def safe_spearman(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    ).ravel()

    y = np.asarray(
        y,
        dtype=np.float64,
    ).ravel()

    finite = np.isfinite(x) & np.isfinite(y)

    x = x[finite]
    y = y[finite]

    if x.size < 3:
        return np.nan

    x_rank = (
        pd.Series(x)
        .rank(method="average")
        .to_numpy(dtype=np.float64)
    )

    y_rank = (
        pd.Series(y)
        .rank(method="average")
        .to_numpy(dtype=np.float64)
    )

    return safe_pearson(
        x_rank,
        y_rank,
    )


def safe_cosine(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    ).ravel()

    y = np.asarray(
        y,
        dtype=np.float64,
    ).ravel()

    finite = np.isfinite(x) & np.isfinite(y)

    x = x[finite]
    y = y[finite]

    if x.size == 0:
        return np.nan

    denominator = float(
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.dot(x, y) / denominator
    )


def r2_uncentered(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).ravel()

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).ravel()

    finite = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if y_true.size == 0:
        return np.nan

    denominator = float(
        np.sum(y_true * y_true)
    )

    if denominator <= 0:
        return np.nan

    residual_sum_squares = float(
        np.sum(
            (y_true - y_pred) ** 2
        )
    )

    return float(
        1.0
        - residual_sum_squares / denominator
    )


def r2_centered(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).ravel()

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).ravel()

    finite = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if y_true.size < 2:
        return np.nan

    denominator = float(
        np.sum(
            (y_true - np.mean(y_true)) ** 2
        )
    )

    if denominator <= 0:
        return np.nan

    residual_sum_squares = float(
        np.sum(
            (y_true - y_pred) ** 2
        )
    )

    return float(
        1.0
        - residual_sum_squares / denominator
    )


def mse_rmse_mae(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).ravel()

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).ravel()

    finite = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if y_true.size == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    error = y_true - y_pred

    mse = float(
        np.mean(error**2)
    )

    rmse = float(
        np.sqrt(mse)
    )

    mae = float(
        np.mean(np.abs(error))
    )

    return (
        mse,
        rmse,
        mae,
    )


def sign_accuracy(
    y_true,
    y_pred,
    eps_true=0.0,
    eps_pred=0.0,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).ravel()

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).ravel()

    finite = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if y_true.size == 0:
        return (
            np.nan,
            np.nan,
            0,
            0,
        )

    true_sign = np.zeros_like(
        y_true,
        dtype=np.int8,
    )

    predicted_sign = np.zeros_like(
        y_pred,
        dtype=np.int8,
    )

    true_sign[y_true > eps_true] = 1
    true_sign[y_true < -eps_true] = -1

    predicted_sign[y_pred > eps_pred] = 1
    predicted_sign[y_pred < -eps_pred] = -1

    accuracy_all = float(
        np.mean(
            true_sign == predicted_sign
        )
    )

    true_nonzero = true_sign != 0
    n_nonzero = int(np.sum(true_nonzero))

    if n_nonzero == 0:
        accuracy_nonzero = np.nan

    else:
        accuracy_nonzero = float(
            np.mean(
                true_sign[true_nonzero]
                == predicted_sign[true_nonzero]
            )
        )

    return (
        accuracy_all,
        accuracy_nonzero,
        int(y_true.size),
        n_nonzero,
    )


def make_gene_train_test_masks(
    n_genes,
    target_idx=None,
    holdout_frac=0.0,
    rng=None,
    exclude_target_fit=False,
    exclude_target_eval=False,
):
    base_fit_mask = np.ones(
        n_genes,
        dtype=bool,
    )

    base_eval_mask = np.ones(
        n_genes,
        dtype=bool,
    )

    if target_idx is not None:
        target_idx = int(target_idx)

        if 0 <= target_idx < n_genes:
            if exclude_target_fit:
                base_fit_mask[target_idx] = False

            if exclude_target_eval:
                base_eval_mask[target_idx] = False

    holdout_frac = float(holdout_frac)

    if holdout_frac <= 0:
        return (
            base_fit_mask.copy(),
            base_eval_mask.copy(),
        )

    if holdout_frac >= 1.0:
        raise ValueError(
            "HOLDOUT_FRAC must be < 1.0"
        )

    if rng is None:
        rng = np.random.default_rng(
            SPLIT_SEED
        )

    n_test = max(
        1,
        int(
            round(
                holdout_frac * n_genes
            )
        ),
    )

    test_indices = rng.choice(
        np.arange(
            n_genes,
            dtype=np.int64,
        ),
        size=n_test,
        replace=False,
    )

    test_mask = np.zeros(
        n_genes,
        dtype=bool,
    )

    test_mask[test_indices] = True

    train_mask = ~test_mask

    train_mask &= base_fit_mask
    test_mask &= base_eval_mask

    return (
        train_mask,
        test_mask,
    )


# ============================================================
# RECOMPUTE FORWARD METRICS
# ============================================================

def compute_forward_for_dataset_mode(
    dataset_dir,
    mode,
):
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_dir.name

    mode_dir = dataset_dir / "normalizations" / mode
    rows = []

    ready, reason, sigma_path, stats_path = mode_files_ready(mode_dir)

    if not ready:
        return (
            rows,
            {
                "dataset": dataset_name,
                "mode": mode,
                "status": reason,
                "sigma_path": str(sigma_path),
                "stats_path": str(stats_path),
                "n_rows": 0,
            },
        )

    sigma = None

    try:
        dataset_files = load_dataset_level_files(dataset_dir)
        genes = dataset_files["genes"]
        perturbation_to_target_index = dataset_files["pert_to_target_idx"]
        perturbation_to_target_gene = dataset_files["pert_to_target_gene"]
        n_genes = int(len(genes))

        raw_reference = load_raw_reference_information(
            dataset_dir=dataset_dir,
            n_genes=n_genes,
        )

        raw_control_mean = raw_reference["raw_control_mean"]
        mean_library_size = raw_reference["mean_library_size"]
        raw_stats_path = raw_reference["raw_stats_path"]

        pflog_pseudocount = None
        pflog_pseudocount_source = "not_applicable"

        if mode == "pflog":
            pflog_pseudocount, pflog_pseudocount_source = load_pflog_pseudocount(
                dataset_dir=dataset_dir,
                mode_dir=mode_dir,
                stats_path=stats_path,
            )

        sigma = load_sigma(sigma_path)
        expected_shape = (n_genes, n_genes)

        if sigma.shape != expected_shape:
            raise ValueError(
                f"Bad Sigma shape: Sigma.shape={sigma.shape}, "
                f"expected={expected_shape}"
            )

        # Use the same gene split for every normalization in a dataset.
        rng = np.random.default_rng(
            stable_int_seed(dataset_name, base_seed=SPLIT_SEED)
        )

        with h5py.File(stats_path, "r") as native_h5, h5py.File(raw_stats_path, "r") as raw_h5:
            if "dx" not in native_h5:
                raise KeyError(f"No dx dataset in {stats_path}")
            if "dx" not in raw_h5:
                raise KeyError(f"No dx dataset in {raw_stats_path}")

            native_dx_dataset = native_h5["dx"]
            raw_dx_dataset = raw_h5["dx"]

            if native_dx_dataset.shape[1] != n_genes:
                raise ValueError(
                    "Native dx gene dimension mismatch: "
                    f"dx.shape={native_dx_dataset.shape}, n_genes={n_genes}"
                )

            if raw_dx_dataset.shape[1] != n_genes:
                raise ValueError(
                    "Raw dx gene dimension mismatch: "
                    f"dx.shape={raw_dx_dataset.shape}, n_genes={n_genes}"
                )

            native_perturbations = (
                safe_read_h5_strings(native_h5, "perturbations")
                if "perturbations" in native_h5
                else dataset_files["perturbations"]
            )

            raw_perturbations = (
                safe_read_h5_strings(raw_h5, "perturbations")
                if "perturbations" in raw_h5
                else dataset_files["perturbations"]
            )

            if native_dx_dataset.shape[0] != len(native_perturbations):
                raise ValueError("Native dx perturbation dimension mismatch.")
            if raw_dx_dataset.shape[0] != len(raw_perturbations):
                raise ValueError("Raw dx perturbation dimension mismatch.")

            raw_perturbation_to_row = {
                str(perturbation): int(index)
                for index, perturbation in enumerate(raw_perturbations)
            }

            perturbation_iterator = tqdm(
                range(len(native_perturbations)),
                desc=f"{dataset_name} / {mode}",
                ncols=TQDM_NCOLS,
                leave=False,
            )

            for index in perturbation_iterator:
                perturbation = str(native_perturbations[index])

                if perturbation not in perturbation_to_target_index:
                    rows.append({
                        "dataset": dataset_name,
                        "mode": mode,
                        "mode_label": MODE_LABELS.get(mode, mode),
                        "perturbation": perturbation,
                        "target_gene": "",
                        "target_gene_index": -1,
                        "status": "pert_missing_from_top_level_map",
                    })
                    continue

                if perturbation not in raw_perturbation_to_row:
                    rows.append({
                        "dataset": dataset_name,
                        "mode": mode,
                        "mode_label": MODE_LABELS.get(mode, mode),
                        "perturbation": perturbation,
                        "target_gene": "",
                        "target_gene_index": -1,
                        "status": "pert_missing_from_raw_mode",
                    })
                    continue

                target_index = int(perturbation_to_target_index[perturbation])
                target_gene = str(
                    perturbation_to_target_gene.get(perturbation, "")
                )

                if target_index < 0 or target_index >= n_genes:
                    rows.append({
                        "dataset": dataset_name,
                        "mode": mode,
                        "mode_label": MODE_LABELS.get(mode, mode),
                        "perturbation": perturbation,
                        "target_gene": target_gene,
                        "target_gene_index": target_index,
                        "status": "bad_target_index",
                    })
                    continue

                native_dx = np.asarray(
                    native_dx_dataset[index, :],
                    dtype=np.float64,
                )

                raw_row = raw_perturbation_to_row[perturbation]
                raw_dx = np.asarray(
                    raw_dx_dataset[raw_row, :],
                    dtype=np.float64,
                )

                sigma_column = np.asarray(
                    sigma[:, target_index],
                    dtype=np.float64,
                )

                finite_base = (
                    np.isfinite(native_dx)
                    & np.isfinite(raw_dx)
                    & np.isfinite(sigma_column)
                    & np.isfinite(raw_control_mean)
                )

                train_mask, test_mask = make_gene_train_test_masks(
                    n_genes=n_genes,
                    target_idx=target_index,
                    holdout_frac=HOLDOUT_FRAC,
                    rng=rng,
                    exclude_target_fit=EXCLUDE_TARGET_GENE_FROM_FIT,
                    exclude_target_eval=EXCLUDE_TARGET_GENE_FROM_EVAL,
                )

                train_mask &= finite_base
                test_mask &= finite_base

                n_train = int(np.sum(train_mask))
                n_test = int(np.sum(test_mask))

                if n_train < MIN_TRAIN_GENES or n_test < MIN_TEST_GENES:
                    rows.append({
                        "dataset": dataset_name,
                        "mode": mode,
                        "mode_label": MODE_LABELS.get(mode, mode),
                        "perturbation": perturbation,
                        "target_gene": target_gene,
                        "target_gene_index": target_index,
                        "status": "too_few_genes",
                        "n_train_genes": n_train,
                        "n_test_genes": n_test,
                    })
                    continue

                # Fit alpha entirely in the method's native normalization space.
                x_train_native = sigma_column[train_mask]
                y_train_native = native_dx[train_mask]

                denominator = float(
                    np.dot(x_train_native, x_train_native)
                )

                if not np.isfinite(denominator) or denominator <= MIN_SIGMA_COL_NORM2:
                    rows.append({
                        "dataset": dataset_name,
                        "mode": mode,
                        "mode_label": MODE_LABELS.get(mode, mode),
                        "perturbation": perturbation,
                        "target_gene": target_gene,
                        "target_gene_index": target_index,
                        "status": "tiny_sigma_col_norm",
                        "sigma_col_norm2_train": denominator,
                        "n_train_genes": n_train,
                        "n_test_genes": n_test,
                    })
                    continue

                a_hat = float(
                    np.dot(x_train_native, y_train_native) / denominator
                )

                native_prediction_full = a_hat * sigma_column

                raw_prediction_full, raw_push_scale = first_order_native_shift_to_raw(
                    native_shift=native_prediction_full,
                    mode=mode,
                    raw_control_mean=raw_control_mean,
                    mean_library_size=mean_library_size,
                    pflog_pseudocount=pflog_pseudocount,
                )

                # Every method is now scored against the same raw-space dx.
                y_true = raw_dx[test_mask]
                y_pred = raw_prediction_full[test_mask]

                pearson = safe_pearson(y_true, y_pred)
                spearman = safe_spearman(y_true, y_pred)
                cosine = safe_cosine(y_true, y_pred)
                uncentered_r2 = r2_uncentered(y_true, y_pred)
                centered_r2 = r2_centered(y_true, y_pred)
                mse, rmse, mae = mse_rmse_mae(y_true, y_pred)

                (
                    accuracy_all,
                    accuracy_nonzero,
                    n_sign_all,
                    n_sign_nonzero,
                ) = sign_accuracy(
                    y_true,
                    y_pred,
                    eps_true=SIGN_EPS_TRUE,
                    eps_pred=SIGN_EPS_PRED,
                )

                rows.append({
                    "dataset": dataset_name,
                    "mode": mode,
                    "mode_label": MODE_LABELS.get(mode, mode),
                    "perturbation": perturbation,
                    "target_gene": target_gene,
                    "target_gene_index": target_index,
                    "status": "ok",
                    "a_hat_native_space": a_hat,
                    "a_hat": a_hat,
                    "evaluation_space": "raw",
                    "pushback_order": 1,
                    "pearson": pearson,
                    "spearman": spearman,
                    "cosine": cosine,
                    "r2_uncentered": uncentered_r2,
                    "r2_centered": centered_r2,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "sign_accuracy": accuracy_all,
                    "sign_accuracy_nonzero_true": accuracy_nonzero,
                    "n_sign_genes": n_sign_all,
                    "n_sign_nonzero_true": n_sign_nonzero,
                    "n_train_genes": n_train,
                    "n_test_genes": n_test,
                    "holdout_frac": float(HOLDOUT_FRAC),
                    "train_equals_test": bool(HOLDOUT_FRAC <= 0),
                    "exclude_target_gene_from_fit": bool(EXCLUDE_TARGET_GENE_FROM_FIT),
                    "exclude_target_gene_from_eval": bool(EXCLUDE_TARGET_GENE_FROM_EVAL),
                    "sigma_col_norm2_train": denominator,
                    "dx_norm_test": float(np.linalg.norm(y_true)),
                    "pred_norm_test": float(np.linalg.norm(y_pred)),
                    "raw_push_scale_mean_test": float(np.mean(raw_push_scale[test_mask])),
                    "raw_push_scale_min_test": float(np.min(raw_push_scale[test_mask])),
                    "raw_push_scale_max_test": float(np.max(raw_push_scale[test_mask])),
                    "raw_control_mean_source": raw_reference["control_mean_source"],
                    "mean_library_size": float(mean_library_size),
                    "library_size_source": raw_reference["library_size_source"],
                    "pflog_pseudocount": (
                        float(pflog_pseudocount)
                        if pflog_pseudocount is not None
                        else np.nan
                    ),
                    "pflog_pseudocount_source": pflog_pseudocount_source,
                    "sigma_path": str(sigma_path),
                    "stats_path": str(stats_path),
                    "raw_stats_path": str(raw_stats_path),
                })

        n_ok = sum(row.get("status") == "ok" for row in rows)

        return (
            rows,
            {
                "dataset": dataset_name,
                "mode": mode,
                "status": "ok",
                "n_rows": int(len(rows)),
                "n_ok": int(n_ok),
                "evaluation_space": "raw",
                "pushback_order": 1,
                "control_mean_source": raw_reference["control_mean_source"],
                "mean_library_size": float(mean_library_size),
                "library_size_source": raw_reference["library_size_source"],
                "pflog_pseudocount": (
                    float(pflog_pseudocount)
                    if pflog_pseudocount is not None
                    else np.nan
                ),
                "pflog_pseudocount_source": pflog_pseudocount_source,
                "sigma_path": str(sigma_path),
                "stats_path": str(stats_path),
                "raw_stats_path": str(raw_stats_path),
            },
        )

    except Exception as error:
        return (
            rows,
            {
                "dataset": dataset_name,
                "mode": mode,
                "status": "error",
                "error": repr(error),
                "sigma_path": str(sigma_path),
                "stats_path": str(stats_path),
                "n_rows": int(len(rows)),
            },
        )

    finally:
        if sigma is not None:
            del sigma
        gc.collect()


def summarize_metrics(metrics_df):
    if (
        metrics_df is None
        or len(metrics_df) == 0
        or "status" not in metrics_df.columns
    ):
        return pd.DataFrame()

    ok = metrics_df[
        metrics_df["status"].astype(str) == "ok"
    ].copy()

    if len(ok) == 0:
        return pd.DataFrame()

    metric_columns = [
        "pearson",
        "spearman",
        "cosine",
        "r2_uncentered",
        "r2_centered",
        "mse",
        "rmse",
        "mae",
        "sign_accuracy",
        "sign_accuracy_nonzero_true",
        "a_hat",
        "n_train_genes",
        "n_test_genes",
        "dx_norm_test",
        "pred_norm_test",
    ]

    rows = []

    grouped = ok.groupby(
        [
            "dataset",
            "mode",
        ],
        sort=False,
    )

    for (
        dataset,
        mode,
    ), subset in grouped:
        row = {
            "dataset": dataset,
            "mode": mode,
            "mode_label": MODE_LABELS.get(
                mode,
                mode,
            ),
            "n_perturbations_ok": int(
                len(subset)
            ),
        }

        for column in metric_columns:
            if column not in subset.columns:
                continue

            values = pd.to_numeric(
                subset[column],
                errors="coerce",
            ).to_numpy(
                dtype=np.float64
            )

            values = values[
                np.isfinite(values)
            ]

            if values.size == 0:
                row[f"{column}_mean"] = np.nan
                row[f"{column}_median"] = np.nan
                row[f"{column}_sem"] = np.nan

            else:
                row[f"{column}_mean"] = float(
                    np.mean(values)
                )

                row[f"{column}_median"] = float(
                    np.median(values)
                )

                row[f"{column}_sem"] = (
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                        / np.sqrt(values.size)
                    )
                    if values.size > 1
                    else 0.0
                )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING HELPERS
# ============================================================

METRIC_SPECS = [
    {
        "metric": "pearson",
        "label": "Pearson (raw-space response)",
        "hist_label": "Pearson",
    },
    {
        "metric": "r2_uncentered",
        "label": "Uncentered R² (raw-space response)",
        "hist_label": "Uncentered R²",
    },
]


def metric_display_df(
    dataframe,
    metric_column,
    display_range=DISPLAY_RANGE,
):
    if dataframe is None or len(dataframe) == 0:
        return pd.DataFrame(
            columns=dataframe.columns
            if dataframe is not None
            else []
        )

    output = dataframe.copy()

    output[metric_column] = pd.to_numeric(
        output[metric_column],
        errors="coerce",
    )

    output = output[
        np.isfinite(
            output[metric_column]
        )
    ].copy()

    return output


def determine_shared_mode_order(
    dataframe,
    metric_column=ORDER_MODES_BY_METRIC,
):
    """
    Determine one common normalization order for every plot.

    If metric_column is None, preserve the order in MODES.
    Otherwise, order modes by their pooled mean for metric_column.
    """
    if metric_column is None:
        return list(MODES)

    display_df = metric_display_df(
        dataframe,
        metric_column,
        display_range=DISPLAY_RANGE,
    )

    if len(display_df) == 0:
        return list(MODES)

    means = (
        display_df
        .groupby("mode")[metric_column]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    ordered_modes = [
        mode
        for mode in means.index.tolist()
        if mode in MODES
    ]

    missing_modes = [
        mode
        for mode in MODES
        if mode not in ordered_modes
    ]

    return ordered_modes + missing_modes


def savefig(
    figure,
    output_base,
):
    output_base = Path(output_base)

    ensure_dir(
        output_base.parent
    )

    if SAVE_SVG:
        figure.savefig(
            output_base.with_suffix(".svg"),
            bbox_inches="tight",
        )

    if SAVE_PNG:
        figure.savefig(
            output_base.with_suffix(".png"),
            dpi=DPI,
            bbox_inches="tight",
        )

    if SHOW_FIGS:
        plt.show()

    plt.close(figure)


def add_empty_figure(
    title,
    output_base,
    message="No successful metric rows were available.",
):
    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.axis("off")

    axis.text(
        0.5,
        0.58,
        title,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    axis.text(
        0.5,
        0.40,
        message,
        ha="center",
        va="center",
        fontsize=13,
    )

    savefig(
        figure,
        output_base,
    )


def add_metric_block(
    figure,
    grid,
    dataframe,
    metric_spec,
    mode_order,
    top_row,
    histogram_row,
    random_seed,
):
    """
    Add one dot/box panel plus one row of per-mode histograms.

    Returns summary rows for this metric block.
    """
    metric_column = metric_spec["metric"]
    metric_label = metric_spec["label"]
    histogram_label = metric_spec["hist_label"]

    display_df = metric_display_df(
        dataframe,
        metric_column,
        display_range=DISPLAY_RANGE,
    )

    x_positions = {
        mode: index + 1
        for index, mode in enumerate(mode_order)
    }

    rng = np.random.default_rng(
        random_seed
    )

    summary_rows = []

    # --------------------------------------------------------
    # Top dot/box panel
    # --------------------------------------------------------

    top_axis = figure.add_subplot(
        grid[top_row, :]
    )

    nonempty_modes = []
    box_values = []
    box_positions = []

    for mode in mode_order:
        values = display_df.loc[
            display_df["mode"] == mode,
            metric_column,
        ].to_numpy(
            dtype=float
        )

        values = values[
            np.isfinite(values)
        ]

        if values.size > 0:
            nonempty_modes.append(mode)
            box_values.append(values)
            box_positions.append(
                x_positions[mode]
            )

    if box_values:
        top_axis.boxplot(
            box_values,
            positions=box_positions,
            widths=BOX_WIDTH,
            patch_artist=False,
            showfliers=False,
            medianprops={
                "linewidth": 1.8,
            },
            boxprops={
                "linewidth": 1.4,
            },
            whiskerprops={
                "linewidth": 1.4,
            },
            capprops={
                "linewidth": 1.4,
            },
        )

    for mode in mode_order:
        subset = display_df[
            display_df["mode"] == mode
        ]

        y_values = subset[
            metric_column
        ].to_numpy(
            dtype=float
        )

        y_values = y_values[
            np.isfinite(y_values)
        ]

        if y_values.size == 0:
            continue

        center_x = x_positions[mode]

        x_values = (
            center_x
            + rng.uniform(
                -JITTER_WIDTH,
                JITTER_WIDTH,
                size=len(y_values),
            )
        )

        top_axis.scatter(
            x_values,
            y_values,
            s=DOT_SIZE,
            alpha=DOT_ALPHA,
            color=MODE_COLORS.get(
                mode,
                None,
            ),
            edgecolors="none",
            zorder=2,
        )

    # Dataset-level means.
    #
    # For a dataset-specific plot, this creates one line connecting
    # the mode means for that dataset.
    #
    # For the pooled composite, this creates one line per dataset.
    if len(display_df) > 0:
        dataset_means = (
            display_df
            .groupby(
                [
                    "dataset",
                    "mode",
                ],
                as_index=False,
            )[metric_column]
            .mean()
            .rename(
                columns={
                    metric_column: "dataset_mode_mean"
                }
            )
        )

        for dataset, subset in dataset_means.groupby(
            "dataset",
            sort=False,
        ):
            x_values = []
            y_values = []

            for mode in mode_order:
                values = subset.loc[
                    subset["mode"] == mode,
                    "dataset_mode_mean",
                ].to_numpy(
                    dtype=float
                )

                if values.size > 0:
                    x_values.append(
                        x_positions[mode]
                    )

                    y_values.append(
                        float(values[0])
                    )

            if len(x_values) >= 2:
                top_axis.plot(
                    x_values,
                    y_values,
                    color="black",
                    alpha=LINE_ALPHA,
                    linewidth=1.0,
                    zorder=1,
                )

            if len(x_values) >= 1:
                top_axis.scatter(
                    x_values,
                    y_values,
                    s=DATASET_MEAN_DOT_SIZE,
                    color="black",
                    alpha=0.45,
                    edgecolors="none",
                    zorder=4,
                )

    # Grand mean across all perturbations represented in the plot.
    for mode in mode_order:
        values = display_df.loc[
            display_df["mode"] == mode,
            metric_column,
        ].to_numpy(
            dtype=float
        )

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            summary_rows.append(
                {
                    "metric": metric_column,
                    "metric_label": metric_label,
                    "mode": mode,
                    "mode_label": MODE_LABELS.get(
                        mode,
                        mode,
                    ),
                    "mean_all": np.nan,
                    "median_all": np.nan,
                    "n_all": 0,
                }
            )

            continue

        grand_mean = float(
            np.mean(values)
        )

        grand_median = float(
            np.median(values)
        )

        top_axis.scatter(
            [x_positions[mode]],
            [grand_mean],
            marker="D",
            s=GRAND_MEAN_SIZE,
            color=MODE_COLORS.get(
                mode,
                "black",
            ),
            edgecolors="none",
            zorder=5,
        )

        summary_rows.append(
            {
                "metric": metric_column,
                "metric_label": metric_label,
                "mode": mode,
                "mode_label": MODE_LABELS.get(
                    mode,
                    mode,
                ),
                "mean_all": grand_mean,
                "median_all": grand_median,
                "n_all": int(values.size),
            }
        )

    top_axis.set_xticks(
        [
            x_positions[mode]
            for mode in mode_order
        ]
    )

    top_axis.set_xticklabels(
        [
            MODE_LABELS.get(
                mode,
                mode,
            )
            for mode in mode_order
        ],
        fontsize=12,
    )

    top_axis.set_ylabel(
        metric_label,
        fontsize=13,
    )

    top_axis.set_title(
        metric_label,
        fontsize=16,
        pad=7,
    )

    top_axis.set_xlim(
        0.4,
        len(mode_order) + 0.6,
    )

    top_axis.set_ylim(
        *DISPLAY_RANGE
    )

    top_axis.grid(
        axis="y",
        alpha=0.25,
    )

    top_axis.set_axisbelow(
        True
    )

    for spine_name in [
        "top",
        "right",
        "left",
        "bottom",
    ]:
        top_axis.spines[
            spine_name
        ].set_linewidth(1.0)

    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------

    histogram_axes = []
    global_histogram_max = HIST_YMIN

    for mode_index, mode in enumerate(mode_order):
        axis = figure.add_subplot(
            grid[
                histogram_row,
                mode_index,
            ]
        )

        histogram_axes.append(axis)

        values = display_df.loc[
            display_df["mode"] == mode,
            metric_column,
        ].to_numpy(
            dtype=float
        )

        values = values[
            np.isfinite(values)
        ]

        if values.size > 0:
            counts, _, _ = axis.hist(
                values,
                bins=HIST_BINS,
                range=DISPLAY_RANGE,
                density=True,
                color=MODE_COLORS.get(
                    mode,
                    None,
                ),
                alpha=0.55,
            )

            positive_counts = counts[
                counts > 0
            ]

            if positive_counts.size > 0:
                global_histogram_max = max(
                    global_histogram_max,
                    float(
                        np.max(positive_counts)
                    ),
                )

            mean_value = float(
                np.mean(values)
            )

            axis.axvline(
                mean_value,
                color="black",
                linewidth=1.3,
                alpha=0.8,
            )

        else:
            axis.text(
                0.5,
                0.5,
                "No data",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=8,
            )

        axis.set_title(
            MODE_LABELS.get(
                mode,
                mode,
            ),
            fontsize=9,
        )

        axis.set_xlim(
            *DISPLAY_RANGE
        )

        axis.set_yscale(
            "log"
        )

        axis.grid(
            axis="y",
            alpha=0.22,
        )

        axis.set_xlabel(
            histogram_label,
            fontsize=8,
        )

        axis.tick_params(
            axis="both",
            labelsize=7,
        )

        if mode_index == 0:
            axis.set_ylabel(
                "Density",
                fontsize=8,
            )

        else:
            axis.set_ylabel("")

            axis.tick_params(
                axis="y",
                labelleft=False,
            )

    histogram_top = max(
        global_histogram_max * 1.8,
        HIST_YMIN * 10,
    )

    for axis in histogram_axes:
        axis.set_ylim(
            HIST_YMIN,
            histogram_top,
        )

    return summary_rows


def print_aggregate_metric_summary(
    dataframe,
    mode_order,
    title,
):
    """
    Print mean and median across all individual perturbation rows
    represented in the current figure.

    For a dataset-specific figure, this summarizes all perturbations
    in that dataset. For the pooled composite, this pools perturbations
    across all retained datasets, so datasets with more perturbations
    contribute more observations.
    """
    if dataframe is None or len(dataframe) == 0:
        print("\nNo successful perturbation rows available.\n")
        return pd.DataFrame()

    summary_rows = []

    for mode in mode_order:
        mode_df = dataframe[
            dataframe["mode"].astype(str) == str(mode)
        ].copy()

        if len(mode_df) == 0:
            continue

        row = {
            "normalization": MODE_LABELS.get(mode, mode),
        }

        for metric_column, output_prefix in [
            ("pearson", "pearson"),
            ("r2_uncentered", "r2_uncentered"),
        ]:
            if metric_column not in mode_df.columns:
                values = np.asarray([], dtype=np.float64)
            else:
                values = pd.to_numeric(
                    mode_df[metric_column],
                    errors="coerce",
                ).to_numpy(dtype=np.float64)

                values = values[np.isfinite(values)]

            row[f"{output_prefix}_mean"] = (
                float(np.mean(values))
                if values.size > 0
                else np.nan
            )

            row[f"{output_prefix}_median"] = (
                float(np.median(values))
                if values.size > 0
                else np.nan
            )

            row[f"{output_prefix}_n"] = int(values.size)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 118)
    print(title)
    print(
        "AGGREGATE MEAN AND MEDIAN ACROSS ALL INDIVIDUAL "
        "PERTURBATIONS IN THIS FIGURE"
    )
    print("=" * 118)

    if len(summary_df) == 0:
        print("No finite Pearson or uncentered R² values were available.")
    else:
        display_df = summary_df.rename(
            columns={
                "normalization": "Normalization",
                "pearson_mean": "Pearson mean",
                "pearson_median": "Pearson median",
                "pearson_n": "Pearson n",
                "r2_uncentered_mean": "Uncentered R² mean",
                "r2_uncentered_median": "Uncentered R² median",
                "r2_uncentered_n": "Uncentered R² n",
            }
        )

        print(
            display_df.to_string(
                index=False,
                formatters={
                    "Pearson mean": lambda value: f"{value:.4f}",
                    "Pearson median": lambda value: f"{value:.4f}",
                    "Uncentered R² mean": lambda value: f"{value:.4f}",
                    "Uncentered R² median": lambda value: f"{value:.4f}",
                },
            )
        )

    print("=" * 118 + "\n")

    return summary_df


def plot_combined_metrics(
    dataframe,
    mode_order,
    title,
    output_base,
    scope,
    dataset_name=None,
):
    """
    Make one figure containing both:
        - Pearson comparison
        - Pearson histograms
        - uncentered R² comparison
        - uncentered R² histograms

    Immediately after displaying the figure, print aggregate means
    and medians across all individual perturbations for Pearson and
    uncentered R², separately for each normalization.
    """
    if dataframe is None or len(dataframe) == 0:
        add_empty_figure(
            title=title,
            output_base=output_base,
        )

        print(f"\n{title}")
        print("No successful perturbation rows were available.\n")

        return pd.DataFrame(
            [
                {
                    "scope": scope,
                    "dataset": dataset_name,
                    "status": "no_metric_rows",
                }
            ]
        )

    dataframe = dataframe.copy()

    if "status" in dataframe.columns:
        dataframe = dataframe[
            dataframe["status"].astype(str) == "ok"
        ].copy()

    if len(dataframe) == 0:
        add_empty_figure(
            title=title,
            output_base=output_base,
        )

        print(f"\n{title}")
        print("No successful perturbation rows were available.\n")

        return pd.DataFrame(
            [
                {
                    "scope": scope,
                    "dataset": dataset_name,
                    "status": "no_metric_rows",
                }
            ]
        )

    available_modes = set(
        dataframe["mode"].astype(str)
    )

    plot_mode_order = [
        mode
        for mode in mode_order
        if mode in available_modes
    ]

    if len(plot_mode_order) == 0:
        add_empty_figure(
            title=title,
            output_base=output_base,
            message=(
                "No requested normalization modes had "
                "successful metric rows."
            ),
        )

        print(f"\n{title}")
        print(
            "No requested normalization modes had "
            "successful metric rows.\n"
        )

        return pd.DataFrame(
            [
                {
                    "scope": scope,
                    "dataset": dataset_name,
                    "status": "no_requested_modes",
                }
            ]
        )

    figure_width = max(
        10.0,
        FIGSIZE_BASE_WIDTH_PER_MODE
        * len(plot_mode_order),
    )

    figure = plt.figure(
        figsize=(
            figure_width,
            FIG_HEIGHT,
        )
    )

    grid = GridSpec(
        4,
        len(plot_mode_order),
        height_ratios=[
            3.15,
            1.30,
            3.15,
            1.30,
        ],
        hspace=0.42,
        wspace=0.14,
        figure=figure,
    )

    figure.suptitle(
        title,
        fontsize=19,
        y=0.995,
    )

    all_summary_rows = []

    for metric_index, metric_spec in enumerate(
        METRIC_SPECS
    ):
        metric_summary_rows = add_metric_block(
            figure=figure,
            grid=grid,
            dataframe=dataframe,
            metric_spec=metric_spec,
            mode_order=plot_mode_order,
            top_row=2 * metric_index,
            histogram_row=2 * metric_index + 1,
            random_seed=stable_int_seed(
                title,
                metric_spec["metric"],
                base_seed=RANDOM_DOT_SEED,
            ),
        )

        for row in metric_summary_rows:
            row["scope"] = scope
            row["dataset"] = dataset_name
            row["status"] = "ok"

        all_summary_rows.extend(
            metric_summary_rows
        )

    figure.subplots_adjust(
        top=0.955,
        bottom=0.055,
        left=0.070,
        right=0.985,
    )

    savefig(
        figure,
        output_base,
    )

    # This print occurs after plt.show(), so it appears immediately
    # below the corresponding figure in a Jupyter notebook.
    aggregate_summary_df = print_aggregate_metric_summary(
        dataframe=dataframe,
        mode_order=plot_mode_order,
        title=title,
    )

    # Save the same aggregate table beside the figure files.
    aggregate_output_path = Path(
        str(output_base)
        + "_aggregate_means_medians.tsv"
    )

    aggregate_summary_df.to_csv(
        aggregate_output_path,
        sep="\t",
        index=False,
    )

    return pd.DataFrame(
        all_summary_rows
    )


