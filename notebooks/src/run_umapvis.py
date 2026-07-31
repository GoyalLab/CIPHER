"""Fig S -- CIPHER raw-count direct-UMAP visualization.

Per dataset: keep valid single-gene perturbations, select N_HVG genes by raw control
variance, compute the raw-count control covariance (cipher.compute_covariance), fit UMAP
directly on the raw HVG matrix (no PCA), select perturbations near target Pearson values,
and draw the CIPHER UMAP panels. The rank-1 forward prediction uses cipher.forward_predict.

Helpers in notebooks/src (not part of the cipher package). Config constants below are module
globals the notebook overrides via R.__dict__.update; DATA_DIR / OUTDIR are injected there.
"""
import os
import cipher
try:
    import umap
except Exception:  # pragma: no cover - umap-learn may expose a different import name
    import umap.umap_ as umap

# ============================================================
# FAST RAW-COUNT CIPHER UMAPS — NO PCA
#
# For each dataset:
#   1. load the h5ad fully into memory;
#   2. resolve the perturbation column and control label;
#   3. keep valid single-gene perturbations with enough cells;
#   4. filter genes by raw control mean and force-include targets;
#   5. select exactly N_HVG genes by raw control variance;
#   6. compute the raw-count control covariance on those HVGs;
#   7. select unique perturbations closest to Pearson 0.30 and 0.60;
#   8. fit UMAP DIRECTLY on the raw N_HVG-dimensional matrix;
#   9. make four CIPHER UMAP panels for each selected perturbation.
#
# There is no PFlog, library-size normalization, log transform, or PCA.
# ============================================================

from pathlib import Path
import gc
import hashlib
import json
import re
import time

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from scipy import sparse


# ============================================================
# CONFIG
# ============================================================

DATASET_NAMES = [
    "akana_etal_2026_crispra_perturbseq.h5ad",
    "schemidt_etal_2022_crispra_perturbseq.h5ad",
    "kaden25_rpe1_ctrl_10k_min100_greedy_4gb.h5ad",
    "kaden25_fibroblast_ctrl_10k_min100_greedy_4gb.h5ad",
]

# OUT_ROOT and DATA_PATHS are built inside run_umapvis_pipeline() from the
# notebook-injected DATA_DIR / OUTDIR globals (no hard-coded paths).
DATA_DIR = None
OUTDIR = None
OUT_ROOT = None
DATA_PATHS = None

RAW_LAYER = None
COUNT_LAYER_CANDIDATES = ["counts", "raw_counts", "count"]

PREFERRED_PERT_KEY = "perturbation"
DESIRED_CONTROL = "control"

EXPRESSION_THRESHOLD = 1.0
EXPRESSION_CUTOFF_SOURCE = "control"  # "control" or "all"
MIN_PERT_CELLS = 100

N_HVG = 2000
TARGET_PEARSONS = [0.4,0.5,0.7]
RANK_METRIC = "pearson"  # selection is based on closeness to TARGET_PEARSONS

MAX_COV_CONTROLS = 10_000
MAX_UMAP_CELLS = 20_000
MAX_UMAP_CONTROLS = 10_000
MAX_UMAP_CELLS_PER_PERT = 3_000

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.25
UMAP_METRIC = "euclidean"

ALPHA_RIDGE = 1e-10

VECTOR_BINS = 18
VECTOR_MIN_CELLS = 15

POINT_SIZE = 4
POINT_ALPHA = 0.55
DPI = 300
SHOW_FIGURES = True
SAVE_SVG = True

SEED = 0


# ============================================================
# METADATA CANDIDATES
# ============================================================

PERT_KEY_CANDIDATES = [
    "perturbation",
    "perturbation_name",
    "perturbation_id",
    "condition",
    "condition_name",
    "condition_ID",
    "target_gene",
    "target",
    "gene",
    "guide_target",
    "guide_targets",
    "sgRNA_target",
    "guide",
    "guide_id",
    "guide_identity",
    "sgRNA",
    "sgrna",
    "gRNA",
    "grna",
    "covariate",
    "treatment",
]

GENE_SYMBOL_COLUMNS = [
    "gene_symbol",
    "gene_symbols",
    "gene_name",
    "gene_names",
    "symbol",
    "feature_name",
]

CONTROL_PATTERNS = [
    "control",
    "ctrl",
    "ntc",
    "non-targeting",
    "nontargeting",
    "non_targeting",
    "non-target",
    "nontarget",
    "negative",
    "neg",
    "safe",
    "scramble",
    "scrambled",
    "mock",
    "untreated",
    "vehicle",
    "empty",
]

INVALID_LABELS = {
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

INVALID_SUBSTRINGS = [
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

TARGET_OVERRIDES = {
    # ("dataset_name", "perturbation_label"): "TP53",
    # "perturbation_label": "TP53",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return re.sub(r"_+", "_", value).strip("_")


def stable_seed(name):
    digest = hashlib.md5(str(name).encode()).hexdigest()
    return int((SEED + int(digest[:8], 16)) % (2**32 - 1))


def as_csr_or_dense(X):
    if hasattr(X, "to_memory"):
        X = X.to_memory()

    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32, copy=False)
        X.eliminate_zeros()
        X.sort_indices()
        return X

    return np.asarray(X, dtype=np.float32)


def mean_var(X):
    n = int(X.shape[0])
    if n == 0:
        raise ValueError("Cannot calculate statistics from zero cells.")

    if sparse.issparse(X):
        sums = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
        sums_sq = np.asarray(X.multiply(X).sum(axis=0)).ravel().astype(np.float64)
    else:
        X = np.asarray(X)
        sums = np.sum(X, axis=0, dtype=np.float64)
        sums_sq = np.einsum("ij,ij->j", X, X, optimize=True).astype(np.float64)

    mean = sums / float(n)
    if n > 1:
        var = (sums_sq - n * mean**2) / float(n - 1)
    else:
        var = np.zeros_like(mean)

    return np.nan_to_num(mean), np.maximum(np.nan_to_num(var), 0.0)


def sample_rows(indices, maximum, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= maximum:
        return np.sort(indices)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def finite_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return np.nan

    x = x[valid]
    y = y[valid]

    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def uncentered_r2(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    valid = np.isfinite(y) & np.isfinite(yhat)

    denominator = float(np.sum(y[valid] ** 2))
    if denominator <= 0:
        return np.nan

    numerator = float(np.sum((y[valid] - yhat[valid]) ** 2))
    return 1.0 - numerator / denominator


def symmetric_limit(values, quantile=0.99):
    values = np.abs(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 1.0

    limit = float(np.quantile(values, quantile))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def looks_like_counts(X):
    values = X.data if sparse.issparse(X) else np.asarray(X).ravel()
    values = values[np.isfinite(values)]

    if len(values) == 0 or np.min(values) < 0:
        return False

    if len(values) > 100_000:
        values = np.random.default_rng(SEED).choice(
            values,
            size=100_000,
            replace=False,
        )

    return bool(np.mean(np.abs(values - np.round(values)) < 1e-6) >= 0.95)


# ============================================================
# LOADING AND METADATA RESOLUTION
# ============================================================

def choose_raw_matrix(adata):
    if RAW_LAYER is not None:
        if RAW_LAYER not in adata.layers:
            raise KeyError(
                f"RAW_LAYER={RAW_LAYER!r} missing; layers={list(adata.layers.keys())}"
            )
        return adata.layers[RAW_LAYER], f"layers[{RAW_LAYER!r}]"

    for layer in COUNT_LAYER_CANDIDATES:
        if layer in adata.layers:
            return adata.layers[layer], f"layers[{layer!r}]"

    return adata.X, "X"


def resolve_gene_names(adata):
    def usable(values):
        values = np.asarray(values, dtype=str)
        normalized = np.char.lower(np.char.strip(values))
        valid = ~np.isin(normalized, ["", "nan", "none", "null"])
        return valid.mean() >= 0.8 and len(np.unique(values[valid])) > 2

    names = None
    source = None

    for column in GENE_SYMBOL_COLUMNS:
        if column not in adata.var.columns:
            continue
        candidate = adata.var[column].astype(str).to_numpy()
        if usable(candidate):
            names = candidate
            source = f"var[{column!r}]"
            break

    if names is None:
        candidate = np.asarray(adata.var_names.astype(str))
        if not usable(candidate):
            raise ValueError("No usable gene-symbol column or var_names found.")
        names = candidate
        source = "var_names"

    keep_columns = np.flatnonzero(~pd.Index(names).duplicated(keep="first"))

    print(
        f"[genes] source={source}; unique={len(keep_columns):,}/{len(names):,}"
    )

    return names[keep_columns], keep_columns, source


def find_control(values):
    counts = pd.Series(values).astype(str).value_counts()

    if DESIRED_CONTROL in counts.index:
        return DESIRED_CONTROL, int(counts.loc[DESIRED_CONTROL]), "exact_requested"

    exact = [
        label
        for label in counts.index.astype(str)
        if any(label.lower() == pattern for pattern in CONTROL_PATTERNS)
    ]
    if exact:
        selected = counts.loc[exact].idxmax()
        return str(selected), int(counts.loc[selected]), "exact_pattern"

    contains = [
        label
        for label in counts.index.astype(str)
        if any(pattern in label.lower() for pattern in CONTROL_PATTERNS)
    ]
    if contains:
        selected = counts.loc[contains].idxmax()
        return str(selected), int(counts.loc[selected]), "contains_pattern"

    return None, 0, "none"


def infer_target(label, gene_set):
    lookup = {str(gene).upper(): str(gene) for gene in gene_set}
    label = str(label).strip()

    override = TARGET_OVERRIDES.get(label)
    if override is not None:
        key = str(override).upper()
        return (lookup[key], "override") if key in lookup else ("", "override_missing")

    if label.upper() in lookup:
        return lookup[label.upper()], "exact"

    cleaned = re.sub(
        r"([_\-\s]+)(KD|KO|OE|overexp|overexpression|activation|inhibition)$",
        "",
        label,
        flags=re.I,
    )
    cleaned = re.sub(
        r"^(sgRNA|gRNA|sgrna|grna)([_\-\s]+)",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"^(sg)(?=[A-Z0-9])", "", cleaned)

    if cleaned.upper() in lookup:
        return lookup[cleaned.upper()], "stripped"

    for separator in ["_", "+", "|", ";", ",", " "]:
        if separator not in cleaned:
            continue

        hits = []
        for token in cleaned.split(separator):
            key = token.upper()
            if key in lookup and lookup[key] not in hits:
                hits.append(lookup[key])

        if len(hits) == 1:
            return hits[0], "parsed_single"
        if len(hits) > 1:
            return "", "multi_gene"
        break

    return "", "unmatched"


def resolve_perturbation_key(obs, gene_names):
    gene_set = set(map(str, gene_names))
    candidates = [
        column
        for column in dict.fromkeys([PREFERRED_PERT_KEY] + PERT_KEY_CANDIDATES)
        if column in obs.columns
    ]

    if not candidates:
        raise KeyError(f"No perturbation-key candidate found. obs={list(obs.columns)}")

    rows = []

    for column in candidates:
        values = obs[column].astype(str)
        control, n_control, control_mode = find_control(values)

        if control is None:
            rows.append(
                {
                    "column": column,
                    "score": -1e9,
                    "control": "",
                    "n_control": 0,
                    "control_mode": "none",
                    "target_hit_rate": 0.0,
                }
            )
            continue

        counts = values.value_counts()
        hits = 0
        checked = 0

        for label in [x for x in counts.index.astype(str) if x != control][:3000]:
            if int(counts.loc[label]) < MIN_PERT_CELLS:
                continue
            target, reason = infer_target(label, gene_set)
            checked += 1
            hits += int(bool(target) and reason != "multi_gene")

        hit_rate = hits / max(checked, 1)
        score = (
            1000.0
            + 200.0 * hit_rate
            + 10.0 * np.log10(n_control + 1.0)
            + 2.0 * np.log10(len(counts) + 1.0)
        )

        if control_mode == "exact_requested":
            score += 25.0
        elif control_mode == "exact_pattern":
            score += 15.0
        elif control_mode == "contains_pattern":
            score += 5.0

        rows.append(
            {
                "column": column,
                "score": score,
                "control": control,
                "n_control": n_control,
                "control_mode": control_mode,
                "target_hit_rate": hit_rate,
            }
        )

    table = pd.DataFrame(rows).sort_values("score", ascending=False)
    print("\n[perturbation-key candidates]")
    print(table.to_string(index=False))

    best = table.iloc[0]
    if float(best["score"]) < 0:
        raise ValueError("Could not resolve a perturbation key and control label.")

    print(
        f"[using] obs[{best['column']!r}], control={best['control']!r}, "
        f"n_control={int(best['n_control']):,}"
    )

    return str(best["column"]), str(best["control"]), table


# ============================================================
# FILTERING AND HVG SELECTION
# ============================================================

def filter_dataset(X, labels, genes, control, dataset):
    labels = np.asarray(labels, dtype=str)
    genes = np.asarray(genes, dtype=str)
    counts = pd.Series(labels).value_counts()
    gene_set = set(genes)

    control_rows = np.flatnonzero(labels == control)
    if len(control_rows) < 2:
        raise ValueError(f"Only {len(control_rows)} controls found.")

    if EXPRESSION_CUTOFF_SOURCE == "control":
        cutoff_rows = control_rows
    elif EXPRESSION_CUTOFF_SOURCE == "all":
        cutoff_rows = np.arange(len(labels), dtype=np.int64)
    else:
        raise ValueError("EXPRESSION_CUTOFF_SOURCE must be 'control' or 'all'.")

    cutoff_mean, _ = mean_var(X[cutoff_rows])
    expressed_genes = set(genes[cutoff_mean >= EXPRESSION_THRESHOLD])

    perturbations = []
    targets = {}
    triage_rows = []

    for label in counts.index.astype(str):
        if label == control or int(counts.loc[label]) < MIN_PERT_CELLS:
            continue

        low = label.strip().lower()
        if low in INVALID_LABELS or any(token in low for token in INVALID_SUBSTRINGS):
            continue

        override = TARGET_OVERRIDES.get(
            (dataset, label),
            TARGET_OVERRIDES.get(label),
        )

        if override is not None:
            lookup = {str(gene).upper(): str(gene) for gene in gene_set}
            key = str(override).upper()
            target, reason = (
                (lookup[key], "override")
                if key in lookup
                else ("", "override_missing")
            )
        else:
            target, reason = infer_target(label, gene_set)

        keep = bool(target) and reason != "multi_gene"

        triage_rows.append(
            {
                "perturbation": label,
                "target_gene": target,
                "parse_reason": reason,
                "n_cells": int(counts.loc[label]),
                "selected": keep,
            }
        )

        if keep:
            perturbations.append(label)
            targets[label] = target

    if not perturbations:
        raise ValueError("No valid single-gene perturbations after filtering.")

    keep_gene_set = expressed_genes | set(targets.values())
    keep_columns = np.flatnonzero(pd.Index(genes).isin(keep_gene_set))
    keep_cells = np.isin(labels, np.asarray([control] + perturbations, dtype=object))

    print(
        f"[filter] cells={keep_cells.sum():,}/{len(labels):,}; "
        f"genes={len(keep_columns):,}/{len(genes):,}; "
        f"perts={len(perturbations):,}"
    )

    return {
        "X": X[keep_cells][:, keep_columns],
        "labels": labels[keep_cells],
        "genes": genes[keep_columns],
        "targets": targets,
        "triage": pd.DataFrame(triage_rows),
        "keep_cells": keep_cells,
        "keep_columns": keep_columns,
    }


def select_hvgs(X, labels, genes, control, targets):
    control_rows = np.flatnonzero(labels == control)
    control_mean, control_var = mean_var(X[control_rows])

    eligible = np.flatnonzero(
        (control_mean >= EXPRESSION_THRESHOLD)
        & np.isfinite(control_var)
        & (control_var > 0)
    )

    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    forced = np.unique([gene_to_index[gene] for gene in targets.values()])

    if len(forced) > N_HVG:
        raise ValueError(f"{len(forced)} target genes exceed N_HVG={N_HVG}.")

    ranked = eligible[np.argsort(control_var[eligible])[::-1]]
    forced_set = set(map(int, forced))
    extras = [index for index in ranked if int(index) not in forced_set]
    extras = extras[: max(0, N_HVG - len(forced))]

    selected = np.asarray(list(forced) + list(extras), dtype=int)
    selected = np.unique(selected)

    if len(selected) < N_HVG:
        remaining = [
            index
            for index in np.argsort(control_var)[::-1]
            if int(index) not in set(map(int, selected))
            and np.isfinite(control_var[index])
            and control_var[index] > 0
        ]
        selected = np.r_[selected, remaining[: N_HVG - len(selected)]]

    selected = np.asarray(selected[:N_HVG], dtype=int)

    print(
        f"[HVG] selected={len(selected):,}; forced targets={len(forced):,}; "
        f"raw direct-UMAP dimensions={len(selected):,}"
    )

    return {
        "X": X[:, selected],
        "genes": genes[selected],
        "control_mean": control_mean[selected],
        "control_var": control_var[selected],
        "indices": selected,
    }


# ============================================================
# COVARIANCE AND CIPHER
# ============================================================

def compute_control_covariance(X, control_rows, rng):
    chosen = sample_rows(control_rows, MAX_COV_CONTROLS, rng)
    X_control = X[chosen]
    n = X_control.shape[0]

    # Gene-gene control covariance via the cipher package. cipher.compute_covariance
    # is np.cov(rowvar=False) (sample covariance, ddof=1) -- identical to the previous
    # explicit (X^T X - n*mean mean^T)/(n-1) computation.
    covariance = cipher.compute_covariance(X_control)
    full_control_mean = np.asarray(X[control_rows].mean(axis=0)).ravel().astype(np.float64)

    return full_control_mean, covariance, chosen


def compute_group_means(X, labels, perturbations):
    lookup = {label: index for index, label in enumerate(perturbations)}

    cell_indices = []
    group_indices = []

    for cell_index, label in enumerate(labels):
        group_index = lookup.get(label)
        if group_index is not None:
            cell_indices.append(cell_index)
            group_indices.append(group_index)

    cell_indices = np.asarray(cell_indices, dtype=int)
    group_indices = np.asarray(group_indices, dtype=int)
    counts = np.bincount(group_indices, minlength=len(perturbations))

    group_matrix = sparse.csr_matrix(
        (
            1.0 / counts[group_indices],
            (group_indices, cell_indices),
        ),
        shape=(len(perturbations), X.shape[0]),
    )

    means = group_matrix @ X
    if sparse.issparse(means):
        means = means.toarray()

    return np.asarray(means, dtype=np.float64), counts


def fit_single_gene_cipher(observed_response, covariance, target_index):
    # Rank-1 CIPHER forward projection via the cipher package:
    #   predicted = alpha * Sigma[:, target],  alpha = <col, dx> / <col, col>.
    predicted_response, alpha = cipher.forward_predict(
        covariance, observed_response, target_index
    )
    predicted_response = np.asarray(predicted_response, dtype=np.float64)
    alpha = float(alpha)

    return {
        "dx": np.asarray(observed_response, dtype=np.float64),
        "pred": predicted_response,
        "alpha": alpha,
        "pearson": finite_pearson(observed_response, predicted_response),
        "r2_uncentered": uncentered_r2(observed_response, predicted_response),
    }


# ============================================================
# DIRECT RAW-HVG UMAP — NO PCA
# ============================================================

def fit_direct_umap(X, positions, seed):
    try:
        import umap
    except ImportError as error:
        raise ImportError("Install UMAP with: pip install umap-learn") from error

    X_fit = X[np.asarray(positions, dtype=int)]

    if sparse.issparse(X_fit):
        X_fit = X_fit.tocsr().astype(np.float32, copy=False)
    else:
        X_fit = np.asarray(X_fit, dtype=np.float32)

    print(
        f"[UMAP direct] matrix={X_fit.shape}; no PCA; metric={UMAP_METRIC}"
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(UMAP_N_NEIGHBORS, max(2, X_fit.shape[0] - 1)),
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=seed,
        transform_seed=seed,
        low_memory=True,
    )

    embedding = reducer.fit_transform(X_fit)

    return X_fit, reducer, np.asarray(embedding, dtype=np.float64)


def transform_direct_umap(X_new, reducer):
    if sparse.issparse(X_new):
        X_new = X_new.tocsr().astype(np.float32, copy=False)
    else:
        X_new = np.asarray(X_new, dtype=np.float32)

    return np.asarray(reducer.transform(X_new), dtype=np.float64)


def binned_vector_field(start, end):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    x = start[:, 0]
    y = start[:, 1]
    dx = end[:, 0] - x
    dy = end[:, 1] - y

    x_edges = np.linspace(x.min(), x.max(), VECTOR_BINS + 1)
    y_edges = np.linspace(y.min(), y.max(), VECTOR_BINS + 1)

    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, VECTOR_BINS - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, VECTOR_BINS - 1)

    rows = []
    for ix in range(VECTOR_BINS):
        for iy in range(VECTOR_BINS):
            mask = (x_bin == ix) & (y_bin == iy)
            if int(mask.sum()) < VECTOR_MIN_CELLS:
                continue

            rows.append(
                {
                    "x": float(x[mask].mean()),
                    "y": float(y[mask].mean()),
                    "dx": float(dx[mask].mean()),
                    "dy": float(dy[mask].mean()),
                    "n": int(mask.sum()),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================

def clean_umap_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")


def plot_cipher_umaps(
    dataset,
    rank,
    requested_pearson,
    perturbation,
    target,
    fit,
    pair_embedding,
    pair_is_control,
    control_embedding,
    perturbation_embedding,
    target_expression,
    counterfactual_embedding,
    vector_field,
    score_z,
    output_directory,
):
    figure, axes = plt.subplots(1, 4, figsize=(20, 4.9))

    control_centroid = control_embedding.mean(axis=0)
    perturbation_centroid = perturbation_embedding.mean(axis=0)
    predicted_centroid = counterfactual_embedding.mean(axis=0)

    # A. Observed shift.
    axes[0].scatter(
        pair_embedding[pair_is_control, 0],
        pair_embedding[pair_is_control, 1],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        color="0.72",
        linewidths=0,
        rasterized=True,
    )
    axes[0].scatter(
        pair_embedding[~pair_is_control, 0],
        pair_embedding[~pair_is_control, 1],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        color="tab:orange",
        linewidths=0,
        rasterized=True,
    )
    axes[0].scatter(*control_centroid, s=130, marker="X", color="black", edgecolor="white")
    axes[0].scatter(
        *perturbation_centroid,
        s=130,
        marker="X",
        color="tab:orange",
        edgecolor="white",
    )
    axes[0].annotate(
        "",
        xy=perturbation_centroid,
        xytext=control_centroid,
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "black"},
    )
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color="0.72", label="Control"),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color="tab:orange",
                label=perturbation,
            ),
        ],
        frameon=False,
        fontsize=8,
    )
    axes[0].set_title("A  Observed population shift")
    clean_umap_axis(axes[0])

    # B. Target-gene fluctuation.
    target_expression = np.asarray(target_expression, dtype=float)
    target_z = (target_expression - target_expression.mean()) / max(
        target_expression.std(),
        1e-12,
    )
    target_limit = symmetric_limit(target_z)

    axes[1].scatter(
        perturbation_embedding[:, 0],
        perturbation_embedding[:, 1],
        s=POINT_SIZE,
        alpha=0.12,
        color="black",
        linewidths=0,
        rasterized=True,
    )
    target_scatter = axes[1].scatter(
        control_embedding[:, 0],
        control_embedding[:, 1],
        c=target_z,
        s=POINT_SIZE,
        alpha=0.75,
        cmap="coolwarm",
        vmin=-target_limit,
        vmax=target_limit,
        linewidths=0,
        rasterized=True,
    )

    low_mask = target_expression <= np.quantile(target_expression, 0.10)
    high_mask = target_expression >= np.quantile(target_expression, 0.90)

    if low_mask.any() and high_mask.any():
        axes[1].annotate(
            "",
            xy=control_embedding[high_mask].mean(axis=0),
            xytext=control_embedding[low_mask].mean(axis=0),
            arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "black"},
        )

    plt.colorbar(
        target_scatter,
        ax=axes[1],
        fraction=0.046,
        pad=0.03,
        label=f"control {target} raw expression (z)",
    )
    axes[1].set_title(f"B  Spontaneous control fluctuation\nlow to high {target}")
    clean_umap_axis(axes[1])

    # C. Counterfactual displacement.
    axes[2].scatter(
        pair_embedding[:, 0],
        pair_embedding[:, 1],
        s=POINT_SIZE,
        alpha=0.15,
        color="0.5",
        linewidths=0,
        rasterized=True,
    )

    if len(vector_field):
        axes[2].quiver(
            vector_field["x"],
            vector_field["y"],
            vector_field["dx"],
            vector_field["dy"],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
            color="tab:blue",
        )

    axes[2].annotate(
        "",
        xy=perturbation_centroid,
        xytext=control_centroid,
        arrowprops={"arrowstyle": "-|>", "lw": 2.5, "color": "black"},
    )
    axes[2].annotate(
        "",
        xy=predicted_centroid,
        xytext=control_centroid,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 2.5,
            "color": "tab:blue",
            "linestyle": "--",
        },
    )
    axes[2].legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=2, label="Observed"),
            Line2D(
                [0],
                [0],
                color="tab:blue",
                linewidth=2,
                linestyle="--",
                label="CIPHER",
            ),
        ],
        frameon=False,
        fontsize=8,
    )
    axes[2].set_title("C  CIPHER counterfactual displacement")
    clean_umap_axis(axes[2])

    # D. Response coordinate.
    score_limit = symmetric_limit(score_z)
    score_scatter = axes[3].scatter(
        pair_embedding[:, 0],
        pair_embedding[:, 1],
        c=score_z,
        s=POINT_SIZE,
        alpha=0.75,
        cmap="coolwarm",
        vmin=-score_limit,
        vmax=score_limit,
        linewidths=0,
        rasterized=True,
    )
    plt.colorbar(
        score_scatter,
        ax=axes[3],
        fraction=0.046,
        pad=0.03,
        label="CIPHER coordinate (control SD)",
    )
    axes[3].set_title("D  CIPHER response coordinate")
    clean_umap_axis(axes[3])

    figure.suptitle(
        f"{dataset} | target Pearson={requested_pearson:.2f}: {perturbation}\n"
        f"raw counts, direct UMAP on {N_HVG:,} HVGs, no PCA | "
        f"actual Pearson={fit['pearson']:.3f}, "
        f"uncentered $R^2$={fit['r2_uncentered']:.3f}, "
        f"alpha={fit['alpha']:.3g}",
        fontsize=13,
        y=1.03,
    )

    figure.tight_layout()

    output_stem = output_directory / (
        f"target_pearson_{requested_pearson:.2f}__"
        f"{safe_name(perturbation)}__raw_hvg_direct_umap"
    )

    figure.savefig(str(output_stem) + ".png", dpi=DPI, bbox_inches="tight")
    if SAVE_SVG:
        figure.savefig(str(output_stem) + ".svg", bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()

    plt.close(figure)


# ============================================================
# DATASET RUNNER
# ============================================================

def run_dataset(path):
    path = Path(path)
    dataset = path.stem
    rng = np.random.default_rng(stable_seed(dataset))
    output_directory = OUT_ROOT / safe_name(dataset)
    output_directory.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    print("\n" + "=" * 100)
    print(f"DATASET: {dataset}")
    print(f"H5AD:   {path}")
    print("=" * 100)

    if not path.exists():
        raise FileNotFoundError(path)

    adata = ad.read_h5ad(path)
    raw_matrix, raw_source = choose_raw_matrix(adata)
    X = as_csr_or_dense(raw_matrix)

    print(f"[load] shape={adata.shape}; source={raw_source}; type={type(X).__name__}")

    if not looks_like_counts(X):
        raise ValueError(f"{raw_source} does not look like raw nonnegative integer counts.")

    genes, unique_gene_columns, gene_source = resolve_gene_names(adata)
    X = X[:, unique_gene_columns]

    perturbation_key, control_label, key_table = resolve_perturbation_key(
        adata.obs,
        genes,
    )
    key_table.to_csv(output_directory / "perturbation_key_candidates.csv", index=False)

    original_labels = adata.obs[perturbation_key].astype(str).to_numpy()
    original_obs_names = np.asarray(adata.obs_names, dtype=object)

    filtered = filter_dataset(
        X=X,
        labels=original_labels,
        genes=genes,
        control=control_label,
        dataset=dataset,
    )
    filtered["triage"].to_csv(
        output_directory / "perturbation_target_triage.csv",
        index=False,
    )

    X_filtered = filtered["X"]
    labels = filtered["labels"]
    filtered_genes = filtered["genes"]
    targets = filtered["targets"]
    kept_global_indices = np.flatnonzero(filtered["keep_cells"])
    kept_obs_names = original_obs_names[filtered["keep_cells"]]

    hvg = select_hvgs(
        X=X_filtered,
        labels=labels,
        genes=filtered_genes,
        control=control_label,
        targets=targets,
    )

    X_hvg = hvg["X"]
    hvg_genes = hvg["genes"]
    gene_to_hvg = {gene: index for index, gene in enumerate(hvg_genes)}

    pd.DataFrame(
        {
            "gene": hvg_genes,
            "raw_control_mean": hvg["control_mean"],
            "raw_control_variance": hvg["control_var"],
            "is_target": np.isin(hvg_genes, list(targets.values())),
        }
    ).to_csv(output_directory / "selected_hvgs.csv", index=False)

    control_rows = np.flatnonzero(labels == control_label)
    control_mean, covariance, covariance_rows = compute_control_covariance(
        X_hvg,
        control_rows,
        rng,
    )

    print(
        f"[covariance] controls={len(covariance_rows):,}/{len(control_rows):,}; "
        f"shape={covariance.shape}"
    )

    perturbations = list(targets)
    perturbation_means, perturbation_counts = compute_group_means(
        X_hvg,
        labels,
        perturbations,
    )

    fits = {}
    ranking_rows = []

    for perturbation_index, perturbation in enumerate(perturbations):
        target = targets[perturbation]
        target_index = gene_to_hvg[target]
        observed_response = perturbation_means[perturbation_index] - control_mean

        fit = fit_single_gene_cipher(
            observed_response,
            covariance,
            target_index,
        )
        fit["target"] = target
        fit["target_index"] = target_index
        fits[perturbation] = fit

        ranking_rows.append(
            {
                "perturbation": perturbation,
                "target_gene": target,
                "n_cells": int(perturbation_counts[perturbation_index]),
                "alpha": fit["alpha"],
                "pearson": fit["pearson"],
                "r2_uncentered": fit["r2_uncentered"],
            }
        )

    sort_columns = [RANK_METRIC] + [
        column
        for column in ["pearson", "r2_uncentered", "n_cells"]
        if column != RANK_METRIC
    ]

    ranking = (
        pd.DataFrame(ranking_rows)
        .dropna(subset=[RANK_METRIC])
        .sort_values(sort_columns, ascending=False)
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(output_directory / "all_cipher_performance.csv", index=False)

    if ranking.empty:
        raise RuntimeError("No perturbations had finite CIPHER performance.")

    # Select one UNIQUE perturbation nearest to each requested Pearson value.
    # Targets are processed in the order given in TARGET_PEARSONS.
    selected_rows = []
    used_perturbations = set()

    for requested_pearson in TARGET_PEARSONS:
        candidates = ranking.loc[
            ~ranking["perturbation"].astype(str).isin(used_perturbations)
        ].copy()

        if candidates.empty:
            raise RuntimeError(
                "Not enough unique perturbations to satisfy all TARGET_PEARSONS."
            )

        candidates["requested_pearson"] = float(requested_pearson)
        candidates["pearson_absolute_error"] = (
            candidates["pearson"] - float(requested_pearson)
        ).abs()

        chosen = candidates.sort_values(
            ["pearson_absolute_error", "n_cells"],
            ascending=[True, False],
        ).iloc[0].copy()

        selected_rows.append(chosen)
        used_perturbations.add(str(chosen["perturbation"]))

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected.insert(0, "plot_index", np.arange(1, len(selected) + 1))
    selected.to_csv(
        output_directory / "selected_target_pearson_performance.csv",
        index=False,
    )

    top_perturbations = selected["perturbation"].astype(str).tolist()
    requested_pearson_by_perturbation = dict(
        zip(
            selected["perturbation"].astype(str),
            selected["requested_pearson"].astype(float),
        )
    )

    print("\n[selected perturbations nearest requested Pearson values]")
    print(
        selected[
            [
                "plot_index",
                "requested_pearson",
                "perturbation",
                "target_gene",
                "pearson",
                "pearson_absolute_error",
                "r2_uncentered",
                "n_cells",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # One shared UMAP fit directly on raw 2000-HVG values.
    # --------------------------------------------------------

    selected_controls = sample_rows(control_rows, MAX_UMAP_CONTROLS, rng)
    remaining_capacity = max(0, MAX_UMAP_CELLS - len(selected_controls))
    cells_per_pert = max(
        1,
        min(
            MAX_UMAP_CELLS_PER_PERT,
            remaining_capacity // max(len(top_perturbations), 1),
        ),
    )

    embedding_parts = [selected_controls]
    for perturbation in top_perturbations:
        perturbation_rows = np.flatnonzero(labels == perturbation)
        embedding_parts.append(sample_rows(perturbation_rows, cells_per_pert, rng))

    embedding_rows = np.sort(np.unique(np.concatenate(embedding_parts)))
    embedding_labels = labels[embedding_rows]
    embedding_is_control = embedding_labels == control_label

    X_embedding, reducer, embedding = fit_direct_umap(
        X_hvg,
        embedding_rows,
        stable_seed(dataset + "_direct_umap"),
    )

    control_embedding = embedding[embedding_is_control]
    X_control_embedding = X_embedding[embedding_is_control]

    pd.DataFrame(
        {
            "filtered_cell_index": embedding_rows,
            "original_cell_index": kept_global_indices[embedding_rows],
            "cell_name": kept_obs_names[embedding_rows],
            "perturbation": embedding_labels,
            "is_control": embedding_is_control,
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "normalization": "raw",
            "embedding_input": f"direct_{len(hvg_genes)}_HVGs_no_PCA",
        }
    ).to_csv(
        output_directory / "shared_direct_umap_cells.csv.gz",
        index=False,
        compression="gzip",
    )

    plot_rows = []

    for rank, perturbation in enumerate(top_perturbations, start=1):
        fit = fits[perturbation]
        requested_pearson = requested_pearson_by_perturbation[perturbation]
        pair_mask = embedding_is_control | (embedding_labels == perturbation)
        pair_embedding = embedding[pair_mask]
        pair_is_control = embedding_is_control[pair_mask]
        X_pair = X_embedding[pair_mask]
        perturbation_embedding = pair_embedding[~pair_is_control]

        # Direct raw-HVG counterfactual: x_control + dx_CIPHER.
        if sparse.issparse(X_control_embedding):
            counterfactual_raw = X_control_embedding.toarray().astype(np.float32)
        else:
            counterfactual_raw = np.asarray(X_control_embedding, dtype=np.float32).copy()

        counterfactual_raw += fit["pred"].astype(np.float32)[None, :]
        counterfactual_raw = np.maximum(counterfactual_raw, 0.0)

        counterfactual_embedding = transform_direct_umap(
            counterfactual_raw,
            reducer,
        )

        vector_field = binned_vector_field(
            control_embedding,
            counterfactual_embedding,
        )

        target_index = fit["target_index"]
        if sparse.issparse(X_control_embedding):
            target_expression = X_control_embedding[:, target_index].toarray().ravel()
        else:
            target_expression = np.asarray(
                X_control_embedding[:, target_index]
            ).ravel()

        direction = fit["pred"] / max(np.linalg.norm(fit["pred"]), 1e-12)
        pair_projection = np.asarray(X_pair @ direction).ravel()
        pair_score = pair_projection - float(control_mean @ direction)

        control_score_mean = float(pair_score[pair_is_control].mean())
        control_score_sd = float(pair_score[pair_is_control].std())
        score_z = (pair_score - control_score_mean) / max(control_score_sd, 1e-12)

        plot_cipher_umaps(
            dataset=dataset,
            rank=rank,
            requested_pearson=requested_pearson,
            perturbation=perturbation,
            target=fit["target"],
            fit=fit,
            pair_embedding=pair_embedding,
            pair_is_control=pair_is_control,
            control_embedding=control_embedding,
            perturbation_embedding=perturbation_embedding,
            target_expression=target_expression,
            counterfactual_embedding=counterfactual_embedding,
            vector_field=vector_field,
            score_z=score_z,
            output_directory=output_directory,
        )

        pd.DataFrame(
            {
                "gene": hvg_genes,
                "observed_raw_mean_shift": fit["dx"],
                "cipher_predicted_raw_mean_shift": fit["pred"],
                "is_target": hvg_genes == fit["target"],
            }
        ).to_csv(
            output_directory
            / (
                f"target_pearson_{requested_pearson:.2f}__"
                f"{safe_name(perturbation)}__gene_response.csv.gz"
            ),
            index=False,
            compression="gzip",
        )

        plot_rows.append(
            {
                "rank": rank,
                "requested_pearson": requested_pearson,
                "pearson_absolute_error": abs(fit["pearson"] - requested_pearson),
                "perturbation": perturbation,
                "target_gene": fit["target"],
                "pearson": fit["pearson"],
                "r2_uncentered": fit["r2_uncentered"],
                "alpha": fit["alpha"],
            }
        )

    pd.DataFrame(plot_rows).to_csv(
        output_directory / "plot_summary.csv",
        index=False,
    )

    elapsed = time.perf_counter() - start_time
    metadata = {
        "dataset": dataset,
        "raw_source": raw_source,
        "gene_source": gene_source,
        "perturbation_key": perturbation_key,
        "control_label": control_label,
        "expression_threshold": EXPRESSION_THRESHOLD,
        "expression_cutoff_source": EXPRESSION_CUTOFF_SOURCE,
        "min_pert_cells": MIN_PERT_CELLS,
        "n_hvg": int(len(hvg_genes)),
        "embedding": "direct UMAP on raw HVGs; no PCA",
        "umap_metric": UMAP_METRIC,
        "rank_metric": RANK_METRIC,
        "target_pearsons": [float(x) for x in TARGET_PEARSONS],
        "selected_perturbations": top_perturbations,
        "elapsed_seconds": float(elapsed),
    }

    (output_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print(f"[done] {dataset}: {elapsed:.1f} seconds")

    del adata
    del X
    del X_filtered
    del X_hvg
    del covariance
    del X_embedding
    gc.collect()

    return {
        "dataset": dataset,
        "status": "ok",
        "n_ranked": len(ranking),
        "top": ";".join(top_perturbations),
        "seconds": elapsed,
    }


# ============================================================
# RUN
# ============================================================

def run_umapvis_pipeline():
    """Reproduce the CIPHER raw-count direct-UMAP panels for DATASET_NAMES.

    Resolves DATA_PATHS / OUT_ROOT from the notebook-injected DATA_DIR / OUTDIR, then
    runs the per-dataset pipeline. Verbose per-dataset progress is captured by
    route_logs in the notebook; only concise lines and the plots reach the cell.
    """
    global OUT_ROOT, DATA_PATHS
    if DATA_DIR is None or OUTDIR is None:
        raise RuntimeError("DATA_DIR and OUTDIR must be injected by the notebook.")
    DATA_PATHS = [os.path.join(DATA_DIR, name) for name in DATASET_NAMES]
    OUT_ROOT = Path(OUTDIR) / "umapvis"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for dataset_index, path in enumerate(DATA_PATHS, start=1):
        print(f"\n[{dataset_index}/{len(DATA_PATHS)}] {os.path.basename(path)}")
        try:
            results.append(run_dataset(path))
        except Exception as error:
            print(f"[ERROR] {os.path.basename(path)}: {error!r}")
            results.append({
                "dataset": Path(path).stem,
                "status": "error",
                "error": repr(error),
            })
            gc.collect()

    summary = pd.DataFrame(results)
    summary.to_csv(OUT_ROOT / "dataset_summary.csv", index=False)
    print("\n[done] umapvis pipeline")
    print(summary.to_string(index=False))
    return summary
