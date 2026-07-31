"""Helpers for Fig S14 — covariance-PC top genes (A/B) and forward held-out
Pearson vs response magnitude (C).

Covers panels A/B (covariance-PC top genes) and panel C (forward held-out Pearson).
NOT part of the installable ``cipher`` package — a notebook-only helper for reproducing the
supplementary figure.

The config constants below are the module-level globals the helper functions read.
The reproduction notebook overrides the path-valued
constants (``PRECOMPUTE_ROOT``, ``PRECOMPUTE_ROOT_CANDIDATES``, ``OUTDIR``, ...) to point
under ``$CIPHER_DATA_DIR/suppl`` and the repro output dir.
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
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.stats import linregress, spearmanr

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

from tqdm.auto import tqdm


# ============================================================
# CONFIG — panels A/B (pcs123)
# ============================================================

PRECOMPUTE_ROOT = Path(
    "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED"
)

EXPRESSION_THRESHOLD = 1.0

# None = use every matching precomputed dataset folder.
DATASET_FOLDERS = None

N_PCS = 3
N_TOP_GENES = 25

DPI = 300
SHOW_FIGURES = True

# Prefer the unregularized covariance.
# Fall back to the ridge covariance if necessary.
SIGMA_CANDIDATES = [
    "Sigma_full.npy",
    "Sigma_true.npy",
    "Sigma_control.npy",
    "Sigma.npy",
    "Sigma_full_ridge.npy",
    "Sigma_true_ridge.npy",
]

PC_COLORS = {
    1: "#3274A1",
    2: "#E1812C",
    3: "#3A923A",
}

GLOBAL_GENE_COLOR = "0.72"


# ============================================================
# CONFIG — panel C (forward dx vs Sigma)
# ============================================================

PRECOMPUTE_ROOT_CANDIDATES = [
    Path("precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_SAFE_mean_control_ge_1p0"),
    Path("precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_mean_control_ge_1p0"),
]

OUTDIR = Path("cipher_raw_pearson_vs_dx_magnitude")
PLOT_DIR = OUTDIR / "plots"
PER_DATASET_PLOT_DIR = PLOT_DIR / "per_dataset"

RAW_MODE = "raw"

# Retain all non-Marson datasets and one Marson dataset.
MARSON_NAME_TOKEN = "marson"
MARSON_DATASET_TO_KEEP = None  # None -> alphabetically first Marson dataset

# Optional exact dataset-name whitelist; None keeps all selected datasets.
DATASETS_TO_KEEP = None

# One common split for every perturbation in a dataset is recommended.
HOLDOUT_FRAC = 0.50
SPLIT_SEED = 0
USE_ONE_SHARED_SPLIT_PER_DATASET = True
EXCLUDE_TARGET_GENE_FROM_FIT = False
EXCLUDE_TARGET_GENE_FROM_EVAL = False
MIN_TRAIN_GENES = 50
MIN_TEST_GENES = 50
MIN_SIGMA_COL_NORM2 = 1e-20

# Primary diagnostic uses train magnitude and test Pearson.
MAGNITUDE_SOURCE = "train"  # "train" or "test"
USE_RMS_MAGNITUDE = True
LOG10_FLOOR = 1e-12
LOW_SIGNAL_QUANTILE = 0.25
HIGH_SIGNAL_QUANTILE = 0.75

# Plotting.
SAVE_PNG = True
SAVE_SVG = True
SCATTER_SIZE = 16
SCATTER_ALPHA = 0.23
GRID_ALPHA = 0.20
N_QUANTILE_BINS_PER_DATASET = 8
MIN_POINTS_PER_BIN = 5
N_POOLED_PERCENTILE_BINS = 10
LABEL_LOW_SIGNAL_NEGATIVE = 4
MMAP_SIGMA = True
TQDM_NCOLS = 110


# ============================================================
# HELPERS — panels A/B
# ============================================================

def threshold_to_tag(value):
    return str(value).replace(".", "p")


def decode_str_array(values):
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
            for value in np.asarray(values, dtype=object)
        ],
        dtype=object,
    )


def clean_dataset_name(folder):
    suffix = (
        f"__mean_ge_"
        f"{threshold_to_tag(EXPRESSION_THRESHOLD)}"
    )

    name = folder.name

    if name.endswith(suffix):
        name = name[:-len(suffix)]

    return name


def safe_filename(value):
    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(value),
    )

    return value.strip("_")


def find_dataset_folders():
    if DATASET_FOLDERS is not None:
        return [
            Path(folder)
            for folder in DATASET_FOLDERS
        ]

    tag = threshold_to_tag(
        EXPRESSION_THRESHOLD
    )

    return sorted(
        folder
        for folder in PRECOMPUTE_ROOT.glob(
            f"*__mean_ge_{tag}"
        )
        if folder.is_dir()
    )


def find_sigma_path(folder):
    sigma_dir = folder / "sigmas"

    if not sigma_dir.exists():
        raise FileNotFoundError(
            f"Missing sigma directory: {sigma_dir}"
        )

    for filename in SIGMA_CANDIDATES:
        path = sigma_dir / filename

        if path.exists():
            return path

    # Conservative fallback:
    # find a Sigma file that is not mean-field/shuffled.
    possible = []

    for path in sorted(
        sigma_dir.glob("*.npy")
    ):
        lower = path.name.lower()

        if any(
            word in lower
            for word in [
                "meanfield",
                "mean_field",
                "mean-field",
                "shuffle",
                "shuffled",
                "_mf",
            ]
        ):
            continue

        if "sigma" in lower:
            possible.append(path)

    if possible:
        return possible[0]

    available = sorted(
        path.name
        for path in sigma_dir.glob("*.npy")
    )

    raise FileNotFoundError(
        f"Could not find real/full covariance in {sigma_dir}\n"
        f"Available files: {available}"
    )


def is_global_gene(gene):
    """
    Flag broad genes that commonly dominate global PCs.

    This is only for visual highlighting.
    """

    gene = str(gene).upper()

    patterns = [
        r"^RPL",       # large ribosomal subunit
        r"^RPS",       # small ribosomal subunit
        r"^MT-",       # mitochondrial genes
        r"^HIST",      # histones
        r"^H[1-4][A-Z]",
        r"^EEF",       # elongation factors
        r"^EIF",       # initiation factors
        r"^PABPC",
        r"^RACK1$",
        r"^TPT1$",
        r"^NPM1$",
        r"^SRP[0-9]",
    ]

    return any(
        re.match(pattern, gene)
        for pattern in patterns
    )


def make_symmetric_operator(Sigma):
    """
    Matrix-free symmetric covariance operator.

    The saved covariance should already be symmetric. Averaging
    Sigma @ v and Sigma.T @ v protects against small numerical
    asymmetries without constructing another p x p matrix.
    """

    p = Sigma.shape[0]

    def matvec(vector):
        forward = np.asarray(
            Sigma @ vector,
            dtype=np.float64,
        )

        backward = np.asarray(
            Sigma.T @ vector,
            dtype=np.float64,
        )

        return 0.5 * (
            forward + backward
        )

    return LinearOperator(
        shape=(p, p),
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.float64,
    )


def compute_top_three_pcs(Sigma):
    """
    Return the three largest covariance eigenvalues and eigenvectors.
    """

    p = Sigma.shape[0]

    if p < N_PCS:
        raise ValueError(
            f"Only {p} genes but {N_PCS} PCs requested."
        )

    # Dense solution for very small matrices.
    if p <= N_PCS + 1:
        dense = np.asarray(
            Sigma,
            dtype=np.float64,
        )

        dense = 0.5 * (
            dense + dense.T
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            dense
        )

        order = np.argsort(
            eigenvalues
        )[::-1][:N_PCS]

        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

    else:
        operator = make_symmetric_operator(
            Sigma
        )

        rng = np.random.default_rng(0)

        initial_vector = rng.normal(
            size=p
        )

        initial_vector /= np.linalg.norm(
            initial_vector
        )

        eigenvalues, eigenvectors = eigsh(
            operator,
            k=N_PCS,
            which="LA",
            tol=1e-6,
            maxiter=10000,
            ncv=min(
                p,
                max(20, 2 * N_PCS + 1),
            ),
            v0=initial_vector,
        )

        order = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

    # The sign of an eigenvector is arbitrary.
    # Orient it so its largest absolute loading is positive.
    for pc_index in range(N_PCS):
        vector = eigenvectors[:, pc_index]

        anchor = int(
            np.argmax(
                np.abs(vector)
            )
        )

        if vector[anchor] < 0:
            eigenvectors[:, pc_index] *= -1.0

    return eigenvalues, eigenvectors


def make_top_gene_table(
    dataset,
    genes,
    eigenvalues,
    eigenvectors,
    total_variance,
    sigma_path,
):
    rows = []

    for pc_index in range(N_PCS):
        pc_number = pc_index + 1
        vector = eigenvectors[:, pc_index]

        order = np.argsort(
            np.abs(vector)
        )[::-1]

        top_indices = order[:N_TOP_GENES]

        eigenvalue = float(
            eigenvalues[pc_index]
        )

        explained_fraction = (
            eigenvalue / total_variance
        )

        for rank, gene_index in enumerate(
            top_indices,
            start=1,
        ):
            gene = str(
                genes[gene_index]
            )

            loading = float(
                vector[gene_index]
            )

            rows.append(
                {
                    "dataset": dataset,
                    "pc": pc_number,
                    "rank": rank,
                    "gene": gene,
                    "gene_index": int(
                        gene_index
                    ),
                    "loading": loading,
                    "absolute_loading": abs(
                        loading
                    ),
                    "loading_squared": (
                        loading ** 2
                    ),
                    "is_global_gene": (
                        is_global_gene(gene)
                    ),
                    "eigenvalue": eigenvalue,
                    "variance_fraction": (
                        explained_fraction
                    ),
                    "variance_percent": (
                        100.0
                        * explained_fraction
                    ),
                    "sigma_file": str(
                        sigma_path
                    ),
                }
            )

    return pd.DataFrame(rows)


def plot_dataset(
    dataset,
    top_gene_table,
    eigenvalues,
    total_variance,
    sigma_path,
):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 9),
    )

    for pc_number, ax in enumerate(
        axes,
        start=1,
    ):
        pc_table = (
            top_gene_table.loc[
                top_gene_table["pc"]
                == pc_number
            ]
            .sort_values(
                "loading",
                ascending=True,
            )
            .reset_index(drop=True)
        )

        y_positions = np.arange(
            len(pc_table)
        )

        bar_colors = [
            (
                GLOBAL_GENE_COLOR
                if is_global
                else PC_COLORS[pc_number]
            )
            for is_global in pc_table[
                "is_global_gene"
            ]
        ]

        ax.barh(
            y_positions,
            pc_table["loading"],
            color=bar_colors,
            edgecolor="none",
            height=0.76,
        )

        ax.axvline(
            0.0,
            color="black",
            linewidth=1.0,
        )

        ax.set_yticks(
            y_positions
        )

        ax.set_yticklabels(
            pc_table["gene"],
            fontsize=9,
        )

        # Bold the genes that are not broad global genes.
        for tick_label, is_global in zip(
            ax.get_yticklabels(),
            pc_table["is_global_gene"],
        ):
            if not is_global:
                tick_label.set_fontweight(
                    "bold"
                )

        variance_percent = (
            100.0
            * eigenvalues[pc_number - 1]
            / total_variance
        )

        ax.set_title(
            f"PC{pc_number}\n"
            f"{variance_percent:.2f}% of covariance variance",
            fontsize=13,
        )

        ax.set_xlabel(
            "signed gene loading"
        )

        ax.grid(
            axis="x",
            linestyle=":",
            alpha=0.30,
        )

    legend_handles = [
        Patch(
            facecolor=GLOBAL_GENE_COLOR,
            edgecolor="none",
            label=(
                "Ribosomal / mitochondrial / "
                "translation / histone"
            ),
        ),
        Patch(
            facecolor=PC_COLORS[2],
            edgecolor="none",
            label=(
                "Other potentially dataset-specific gene"
            ),
        ),
    ]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    covariance_label = (
        "ridge covariance"
        if "ridge" in sigma_path.name.lower()
        else "full covariance"
    )

    figure.suptitle(
        f"{dataset}: top genes in covariance PCs 1–3\n"
        f"Top {N_TOP_GENES} genes ranked by absolute loading; "
        f"{covariance_label}",
        fontsize=16,
        y=0.99,
    )

    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.10,
        top=0.87,
        wspace=0.50,
    )

    return figure


# ============================================================
# HELPERS — panel C (forward dx vs Sigma)
# ============================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_seed(*parts):
    text = "::".join([str(x) for x in parts] + [str(SPLIT_SEED)])
    return int(int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**32 - 1))


def save_figure(fig, output_base):
    output_base = Path(output_base)
    ensure_dir(output_base.parent)
    if SAVE_PNG:
        fig.savefig(output_base.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def load_strings_npy(path):
    return np.asarray(np.load(path, allow_pickle=True)).astype(str)


def read_h5_strings(h5, key):
    return np.asarray([
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in h5[key][:]
    ], dtype=str)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(np.dot(x, y) / denominator)


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return np.nan, np.nan
    try:
        from scipy.stats import spearmanr
        result = spearmanr(x, y, nan_policy="omit")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        xr = pd.Series(x).rank(method="average").to_numpy(float)
        yr = pd.Series(y).rank(method="average").to_numpy(float)
        return safe_pearson(xr, yr), np.nan


def choose_precompute_root():
    checked = []
    for root in PRECOMPUTE_ROOT_CANDIDATES:
        checked.append(str(root))
        if not root.exists():
            continue
        usable = [
            p for p in root.iterdir()
            if p.is_dir() and (p / "normalizations" / RAW_MODE).exists()
        ]
        if usable:
            return root
    raise FileNotFoundError(
        "Could not find a raw precompute root. Checked:\n  - "
        + "\n  - ".join(checked)
    )


def discover_and_select_datasets(root):
    dataset_dirs = sorted([
        p for p in Path(root).iterdir()
        if p.is_dir() and (p / "normalizations" / RAW_MODE).exists()
    ], key=lambda p: p.name.lower())

    token = MARSON_NAME_TOKEN.lower()
    marson = [p for p in dataset_dirs if token in p.name.lower()]
    selected = [p for p in dataset_dirs if token not in p.name.lower()]

    kept_marson = None
    if marson:
        if MARSON_DATASET_TO_KEEP is None:
            kept_marson = marson[0]
        else:
            matches = [
                p for p in marson
                if p.name.lower() == str(MARSON_DATASET_TO_KEEP).lower()
            ]
            if len(matches) != 1:
                raise ValueError(
                    "MARSON_DATASET_TO_KEEP did not uniquely match. Available:\n  - "
                    + "\n  - ".join(p.name for p in marson)
                )
            kept_marson = matches[0]
        selected.append(kept_marson)

    if DATASETS_TO_KEEP is not None:
        requested = {str(x).lower() for x in DATASETS_TO_KEEP}
        selected = [p for p in selected if p.name.lower() in requested]

    selected = sorted(selected, key=lambda p: p.name.lower())
    if not selected:
        raise RuntimeError("No datasets selected.")

    print("\n" + "=" * 100)
    print("DATASETS")
    print("=" * 100)
    print(f"Discovered: {len(dataset_dirs)}")
    print(f"Retained:   {len(selected)}")
    print(f"Marson:     {kept_marson.name if kept_marson else 'none'}")
    for p in selected:
        print(f"  - {p.name}")
    print("=" * 100)
    return selected


def load_dataset_maps(dataset_dir):
    dataset_dir = Path(dataset_dir)
    genes = load_strings_npy(dataset_dir / "genes.npy")
    perturbations = load_strings_npy(dataset_dir / "perturbations.npy")
    target_indices = np.load(dataset_dir / "target_gene_indices.npy").astype(int)

    target_genes_path = dataset_dir / "target_genes.npy"
    if target_genes_path.exists():
        target_genes = load_strings_npy(target_genes_path)
    else:
        target_genes = np.asarray([
            genes[i] if 0 <= i < len(genes) else ""
            for i in target_indices
        ], dtype=str)

    if len(perturbations) != len(target_indices):
        raise ValueError(f"Perturbation/target mismatch in {dataset_dir.name}")
    if len(target_genes) != len(perturbations):
        target_genes = np.asarray([
            genes[i] if 0 <= i < len(genes) else ""
            for i in target_indices
        ], dtype=str)

    return {
        "genes": genes,
        "perturbations": perturbations,
        "target_index": dict(zip(perturbations.astype(str), target_indices.astype(int))),
        "target_gene": dict(zip(perturbations.astype(str), target_genes.astype(str))),
    }


def get_raw_paths(dataset_dir):
    mode_dir = Path(dataset_dir) / "normalizations" / RAW_MODE
    sigma_path = mode_dir / "Sigma_full_ridge.npy"
    stats_path = mode_dir / "perturbation_stats.h5"
    if (mode_dir / "Sigma_full_ridge.npy.tmp").exists():
        return False, "sigma_tmp_exists", sigma_path, stats_path
    if not sigma_path.exists():
        return False, "missing_sigma", sigma_path, stats_path
    if not stats_path.exists():
        return False, "missing_stats", sigma_path, stats_path
    return True, "ready", sigma_path, stats_path


def make_gene_split(n_genes, dataset, perturbation=None):
    if not 0 <= HOLDOUT_FRAC < 1:
        raise ValueError("HOLDOUT_FRAC must satisfy 0 <= frac < 1")
    if HOLDOUT_FRAC == 0:
        mask = np.ones(n_genes, dtype=bool)
        return mask.copy(), mask.copy()

    parts = [dataset, "gene_split"]
    if not USE_ONE_SHARED_SPLIT_PER_DATASET:
        parts.append(perturbation)
    rng = np.random.default_rng(stable_seed(*parts))

    n_test = min(max(int(round(HOLDOUT_FRAC * n_genes)), 1), n_genes - 1)
    test_idx = rng.choice(n_genes, size=n_test, replace=False)
    test_mask = np.zeros(n_genes, dtype=bool)
    test_mask[test_idx] = True
    return ~test_mask, test_mask


def compute_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    dataset = dataset_dir.name
    ready, reason, sigma_path, stats_path = get_raw_paths(dataset_dir)
    if not ready:
        return [], {"dataset": dataset, "status": reason, "n_ok": 0}

    rows = []
    sigma = None
    try:
        maps = load_dataset_maps(dataset_dir)
        n_genes = len(maps["genes"])
        sigma = np.load(sigma_path, mmap_mode="r" if MMAP_SIGMA else None)
        if sigma.shape != (n_genes, n_genes):
            raise ValueError(
                f"Bad Sigma shape {sigma.shape}; expected {(n_genes, n_genes)}"
            )

        if USE_ONE_SHARED_SPLIT_PER_DATASET:
            shared_train, shared_test = make_gene_split(n_genes, dataset)

        with h5py.File(stats_path, "r") as h5:
            if "dx" not in h5:
                raise KeyError(f"No dx array in {stats_path}")
            dx_data = h5["dx"]
            if dx_data.shape[1] != n_genes:
                raise ValueError(f"dx shape {dx_data.shape}; n_genes={n_genes}")

            perturbations = (
                read_h5_strings(h5, "perturbations")
                if "perturbations" in h5
                else maps["perturbations"]
            )
            if len(perturbations) != dx_data.shape[0]:
                raise ValueError("Perturbation count does not match dx rows")

            iterator = tqdm(
                range(len(perturbations)),
                desc=f"{dataset} / raw",
                ncols=TQDM_NCOLS,
                leave=False,
            )

            for row_index in iterator:
                perturbation = str(perturbations[row_index])
                base = {
                    "dataset": dataset,
                    "mode": RAW_MODE,
                    "perturbation": perturbation,
                    "holdout_frac": HOLDOUT_FRAC,
                    "shared_split_per_dataset": USE_ONE_SHARED_SPLIT_PER_DATASET,
                }

                if perturbation not in maps["target_index"]:
                    rows.append({
                        **base,
                        "status": "pert_missing_from_map",
                        "target_gene": "",
                        "target_gene_index": -1,
                    })
                    continue

                target_idx = int(maps["target_index"][perturbation])
                target_gene = str(maps["target_gene"].get(perturbation, ""))
                if not 0 <= target_idx < n_genes:
                    rows.append({
                        **base,
                        "status": "bad_target_index",
                        "target_gene": target_gene,
                        "target_gene_index": target_idx,
                    })
                    continue

                dx = np.asarray(dx_data[row_index, :], dtype=float)
                sigma_col = np.asarray(sigma[:, target_idx], dtype=float)

                if USE_ONE_SHARED_SPLIT_PER_DATASET:
                    train_mask = shared_train.copy()
                    test_mask = shared_test.copy()
                else:
                    train_mask, test_mask = make_gene_split(
                        n_genes, dataset, perturbation
                    )

                if EXCLUDE_TARGET_GENE_FROM_FIT:
                    train_mask[target_idx] = False
                if EXCLUDE_TARGET_GENE_FROM_EVAL:
                    test_mask[target_idx] = False

                finite = np.isfinite(dx) & np.isfinite(sigma_col)
                train_mask &= finite
                test_mask &= finite
                n_train = int(train_mask.sum())
                n_test = int(test_mask.sum())

                if n_train < MIN_TRAIN_GENES or n_test < MIN_TEST_GENES:
                    rows.append({
                        **base,
                        "status": "too_few_genes",
                        "target_gene": target_gene,
                        "target_gene_index": target_idx,
                        "n_train_genes": n_train,
                        "n_test_genes": n_test,
                    })
                    continue

                dx_train = dx[train_mask]
                sigma_train = sigma_col[train_mask]
                denominator = float(np.dot(sigma_train, sigma_train))
                if not np.isfinite(denominator) or denominator <= MIN_SIGMA_COL_NORM2:
                    rows.append({
                        **base,
                        "status": "tiny_sigma_col_norm",
                        "target_gene": target_gene,
                        "target_gene_index": target_idx,
                        "n_train_genes": n_train,
                        "n_test_genes": n_test,
                    })
                    continue

                a_hat = float(np.dot(sigma_train, dx_train) / denominator)
                dx_test = dx[test_mask]
                pred_test = a_hat * sigma_col[test_mask]

                dx_norm_train = float(np.linalg.norm(dx_train))
                dx_norm_test = float(np.linalg.norm(dx_test))

                rows.append({
                    **base,
                    "status": "ok",
                    "target_gene": target_gene,
                    "target_gene_index": target_idx,
                    "a_hat": a_hat,
                    "pearson": safe_pearson(dx_test, pred_test),
                    "n_train_genes": n_train,
                    "n_test_genes": n_test,
                    "sigma_col_norm2_train": denominator,
                    "dx_norm_train": dx_norm_train,
                    "dx_rms_train": dx_norm_train / np.sqrt(n_train),
                    "dx_norm_test": dx_norm_test,
                    "dx_rms_test": dx_norm_test / np.sqrt(n_test),
                    "pred_norm_test": float(np.linalg.norm(pred_test)),
                })

        n_ok = sum(row.get("status") == "ok" for row in rows)
        return rows, {
            "dataset": dataset,
            "status": "ok",
            "n_rows": len(rows),
            "n_ok": n_ok,
            "sigma_path": str(sigma_path),
            "stats_path": str(stats_path),
        }

    except Exception as error:
        return rows, {
            "dataset": dataset,
            "status": "error",
            "error": repr(error),
            "n_rows": len(rows),
            "n_ok": sum(row.get("status") == "ok" for row in rows),
        }
    finally:
        if sigma is not None:
            del sigma
        gc.collect()


def prepare_analysis(metrics):
    df = metrics.copy()
    numeric = [
        "pearson", "n_train_genes", "n_test_genes",
        "dx_norm_train", "dx_rms_train", "dx_norm_test", "dx_rms_test",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    source = MAGNITUDE_SOURCE.lower()
    if source not in {"train", "test"}:
        raise ValueError("MAGNITUDE_SOURCE must be 'train' or 'test'")

    magnitude_column = (
        f"dx_rms_{source}" if USE_RMS_MAGNITUDE else f"dx_norm_{source}"
    )
    magnitude_label = (
        rf"RMS($\Delta x$), {source} genes"
        if USE_RMS_MAGNITUDE
        else rf"$\|\Delta x\|_2$, {source} genes"
    )

    df["dx_magnitude"] = pd.to_numeric(df[magnitude_column], errors="coerce")
    df = df[
        np.isfinite(df["pearson"])
        & np.isfinite(df["dx_magnitude"])
        & (df["dx_magnitude"] >= 0)
    ].copy()

    df["log10_dx_magnitude"] = np.log10(
        np.maximum(df["dx_magnitude"], LOG10_FLOOR)
    )

    # Only for pooling across raw datasets; delta_x itself is untouched.
    df["dx_strength_percentile"] = (
        df.groupby("dataset")["dx_magnitude"]
        .rank(method="average", pct=True)
    )
    df["dataset_mean_pearson"] = df.groupby("dataset")["pearson"].transform("mean")
    df["pearson_within_dataset_centered"] = (
        df["pearson"] - df["dataset_mean_pearson"]
    )
    df.attrs["magnitude_label"] = magnitude_label
    return df


def quantile_trend(df, x_column, n_bins, min_points=5):
    work = df[[x_column, "pearson"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < max(3, min_points):
        return pd.DataFrame()
    q = min(int(n_bins), work[x_column].nunique(), len(work))
    if q < 2:
        return pd.DataFrame()
    try:
        work["bin"] = pd.qcut(work[x_column], q=q, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    rows = []
    for bin_id, part in work.groupby("bin", sort=True):
        if len(part) < min_points:
            continue
        x = part[x_column].to_numpy(float)
        y = part["pearson"].to_numpy(float)
        rows.append({
            "bin": int(bin_id),
            "x": np.median(x),
            "y": np.median(y),
            "q25": np.quantile(y, 0.25),
            "q75": np.quantile(y, 0.75),
            "n": len(part),
        })
    return pd.DataFrame(rows)


def pooled_percentile_trend(df):
    work = df.copy()
    bins = np.floor(work["dx_strength_percentile"] * N_POOLED_PERCENTILE_BINS).astype(int)
    work["bin"] = np.clip(bins, 0, N_POOLED_PERCENTILE_BINS - 1)

    # Average perturbations within dataset/bin first.
    stage1 = work.groupby(["dataset", "bin"], as_index=False).agg(
        x=("dx_strength_percentile", "mean"),
        y=("pearson", "mean"),
    )
    # Then give each dataset equal weight.
    stage2 = stage1.groupby("bin", as_index=False).agg(
        x=("x", "mean"),
        y=("y", "mean"),
        sd=("y", "std"),
        n_datasets=("dataset", "nunique"),
    )
    stage2["sem"] = stage2["sd"] / np.sqrt(stage2["n_datasets"].clip(lower=1))
    return stage2


def make_statistics(df):
    rows = []
    for dataset, part in df.groupby("dataset", sort=False):
        x = part["log10_dx_magnitude"].to_numpy(float)
        y = part["pearson"].to_numpy(float)
        percentile = part["dx_strength_percentile"].to_numpy(float)
        rho, pvalue = safe_spearman(x, y)
        low = percentile <= LOW_SIGNAL_QUANTILE
        high = percentile >= HIGH_SIGNAL_QUANTILE
        rows.append({
            "scope": "dataset",
            "dataset": dataset,
            "n": len(part),
            "spearman_rho": rho,
            "spearman_pvalue": pvalue,
            "median_pearson_lowest_quartile": np.median(y[low]) if low.any() else np.nan,
            "median_pearson_highest_quartile": np.median(y[high]) if high.any() else np.nan,
            "fraction_negative_lowest_quartile": np.mean(y[low] < 0) if low.any() else np.nan,
            "fraction_negative_highest_quartile": np.mean(y[high] < 0) if high.any() else np.nan,
        })

    rho, pvalue = safe_spearman(df["log10_dx_magnitude"], df["pearson"])
    rows.append({
        "scope": "pooled_absolute_raw_scale",
        "dataset": "ALL_DATASETS",
        "n": len(df),
        "spearman_rho": rho,
        "spearman_pvalue": pvalue,
    })

    rho, pvalue = safe_spearman(
        df["dx_strength_percentile"],
        df["pearson_within_dataset_centered"],
    )
    rows.append({
        "scope": "pooled_within_dataset",
        "dataset": "ALL_DATASETS",
        "n": len(df),
        "spearman_rho": rho,
        "spearman_pvalue": pvalue,
    })
    return pd.DataFrame(rows)


def style_axis(axis):
    axis.axhline(0, color="black", linewidth=1, alpha=0.45)
    axis.set_ylim(-1.05, 1.05)
    axis.grid(alpha=GRID_ALPHA)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_per_dataset(df):
    x_label = df.attrs["magnitude_label"]
    for index, (dataset, part) in enumerate(df.groupby("dataset", sort=True), 1):
        fig, axis = plt.subplots(figsize=(6.5, 5.3))
        axis.scatter(
            part["log10_dx_magnitude"], part["pearson"],
            s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
            edgecolors="none", rasterized=True,
        )

        trend = quantile_trend(
            part, "log10_dx_magnitude",
            N_QUANTILE_BINS_PER_DATASET, MIN_POINTS_PER_BIN,
        )
        if len(trend):
            axis.errorbar(
                trend["x"], trend["y"],
                yerr=np.vstack([trend["y"] - trend["q25"], trend["q75"] - trend["y"]]),
                marker="o", markersize=5.5, linewidth=2.3,
                capsize=2.5, color="black", zorder=5,
            )

        rho, pvalue = safe_spearman(part["log10_dx_magnitude"], part["pearson"])
        title = f"{dataset}\nSpearman rho = {rho:.3f}, n = {len(part)}"
        if np.isfinite(pvalue):
            title += f", p = {pvalue:.2g}"
        axis.set_title(title)
        axis.set_xlabel(f"log10 {x_label}")
        axis.set_ylabel("Held-out Pearson")
        style_axis(axis)

        low_negative = part[
            part["dx_strength_percentile"] <= LOW_SIGNAL_QUANTILE
        ].nsmallest(LABEL_LOW_SIGNAL_NEGATIVE, "pearson")
        for _, point in low_negative.iterrows():
            axis.annotate(
                str(point["perturbation"]),
                (point["log10_dx_magnitude"], point["pearson"]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=7, alpha=0.85,
            )

        fig.tight_layout()
        save_figure(
            fig,
            PER_DATASET_PLOT_DIR
            / f"{index:02d}_{safe_filename(dataset)}_raw_pearson_vs_dx",
        )


def plot_pooled(df):
    x_label = df.attrs["magnitude_label"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), sharey=True)

    # A: absolute raw-count scale.
    axis = axes[0]
    axis.scatter(
        df["log10_dx_magnitude"], df["pearson"],
        s=10, alpha=0.10, edgecolors="none", rasterized=True,
    )
    for _, part in df.groupby("dataset", sort=False):
        trend = quantile_trend(
            part, "log10_dx_magnitude",
            N_QUANTILE_BINS_PER_DATASET, MIN_POINTS_PER_BIN,
        )
        if len(trend):
            axis.plot(trend["x"], trend["y"], linewidth=1, alpha=0.28)

    rho, pvalue = safe_spearman(df["log10_dx_magnitude"], df["pearson"])
    title = f"Absolute raw-count scale\nNaive pooled rho = {rho:.3f}, n = {len(df)}"
    if np.isfinite(pvalue):
        title += f", p = {pvalue:.2g}"
    axis.set_title(title)
    axis.set_xlabel(f"log10 {x_label}")
    axis.set_ylabel("Held-out Pearson")
    style_axis(axis)

    # B: preferred pooled analysis.
    axis = axes[1]
    axis.scatter(
        df["dx_strength_percentile"], df["pearson"],
        s=10, alpha=0.10, edgecolors="none", rasterized=True,
    )
    trend = pooled_percentile_trend(df)
    axis.errorbar(
        trend["x"], trend["y"], yerr=trend["sem"],
        marker="o", markersize=5.5, linewidth=2.3,
        capsize=2.5, color="black", zorder=5,
    )

    rho, pvalue = safe_spearman(
        df["dx_strength_percentile"],
        df["pearson_within_dataset_centered"],
    )
    title = f"Within-dataset response strength\nrho = {rho:.3f}, n = {len(df)}"
    if np.isfinite(pvalue):
        title += f", p = {pvalue:.2g}"
    axis.set_title(title)
    axis.set_xlabel("Within-dataset response-magnitude percentile")
    style_axis(axis)

    fig.suptitle(
        "CIPHER raw counts: held-out Pearson versus response magnitude",
        fontsize=16, y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, PLOT_DIR / "all_datasets_pooled_raw_pearson_vs_dx")


def main():
    ensure_dir(OUTDIR)
    ensure_dir(PLOT_DIR)
    ensure_dir(PER_DATASET_PLOT_DIR)

    root = choose_precompute_root()
    dataset_dirs = discover_and_select_datasets(root)

    print("\n" + "=" * 100)
    print("RAW CIPHER: PEARSON VS RESPONSE MAGNITUDE")
    print("=" * 100)
    print(f"Root:                     {root}")
    print(f"Output:                   {OUTDIR}")
    print(f"Holdout fraction:         {HOLDOUT_FRAC}")
    print(f"Shared split/dataset:     {USE_ONE_SHARED_SPLIT_PER_DATASET}")
    print(f"Magnitude source:         {MAGNITUDE_SOURCE}")
    print(f"Use RMS:                  {USE_RMS_MAGNITUDE}")
    print("=" * 100)

    all_rows, status_rows = [], []
    for dataset_dir in tqdm(dataset_dirs, desc="datasets", ncols=TQDM_NCOLS):
        rows, status = compute_dataset(dataset_dir)
        all_rows.extend(rows)
        status_rows.append(status)
        gc.collect()

    metrics = pd.DataFrame(all_rows)
    status = pd.DataFrame(status_rows)
    metrics_path = OUTDIR / "raw_forward_metrics_per_perturbation.tsv"
    status_path = OUTDIR / "raw_forward_run_status.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    status.to_csv(status_path, sep="\t", index=False)

    if len(metrics) == 0 or "status" not in metrics:
        raise RuntimeError("No rows produced; inspect raw_forward_run_status.tsv")

    ok = metrics[metrics["status"] == "ok"].copy()
    if len(ok) == 0:
        raise RuntimeError("No successful rows; inspect raw_forward_run_status.tsv")

    analysis = prepare_analysis(ok)
    if len(analysis) == 0:
        raise RuntimeError("No finite Pearson/magnitude rows remained")

    analysis_path = OUTDIR / "raw_pearson_vs_dx_per_perturbation.tsv"
    analysis.to_csv(analysis_path, sep="\t", index=False)

    statistics = make_statistics(analysis)
    statistics_path = OUTDIR / "raw_pearson_vs_dx_statistics.tsv"
    statistics.to_csv(statistics_path, sep="\t", index=False)

    low_negative = analysis[
        (analysis["dx_strength_percentile"] <= LOW_SIGNAL_QUANTILE)
        & (analysis["pearson"] < 0)
    ].sort_values(
        ["dataset", "pearson", "dx_strength_percentile"]
    )
    low_negative_path = OUTDIR / "raw_low_signal_negative_pearson_perturbations.tsv"
    low_negative.to_csv(low_negative_path, sep="\t", index=False)

    plot_per_dataset(analysis)
    plot_pooled(analysis)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Successful perturbations: {len(analysis)}")
    print(f"Datasets:                 {analysis['dataset'].nunique()}")
    print(f"Low-signal negatives:     {len(low_negative)}")
    print(f"Metrics:                  {metrics_path}")
    print(f"Analysis:                 {analysis_path}")
    print(f"Statistics:               {statistics_path}")
    print(f"Low-signal negatives:     {low_negative_path}")
    print(f"Plots:                    {PLOT_DIR}")
    print("=" * 100)
    print("\nStatistics:\n")
    print(statistics.to_string(index=False))
