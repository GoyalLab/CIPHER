"""Forward-problem engine for Fig S7 (covariance-vs-correlation coordinates + high-variance-filtered forward prediction).

Helper in notebooks/src -- NOT part of the installable ``cipher`` package; a notebook-only helper for
reproducing the supplementary figure.

Colliding helper names across the notebook cells differed only in docstrings / whitespace / return style
(semantically identical); the canonical copy is kept here. ``make_gene_removal_mask`` and ``process_one_folder``
are NOT here: they read per-cell config globals, so they stay inline in the notebook cells.
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
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def threshold_to_tag(x):
    return str(x).replace(".", "p")

def find_dataset_folders(root, expression_threshold):
    root = Path(root)
    tag = threshold_to_tag(expression_threshold)
    folders = sorted([p for p in root.glob(f"*__mean_ge_{tag}") if p.is_dir()])
    print(f"[folders] found {len(folders)} folders for threshold={expression_threshold}")
    return folders

def find_sigma_path(sigdir, candidates, glob_patterns=None, exclude_paths=None, label="sigma"):
    sigdir = Path(sigdir)
    exclude_paths = set(Path(x).resolve() for x in (exclude_paths or []))

    for name in candidates:
        p = sigdir / name
        if p.exists() and p.resolve() not in exclude_paths:
            print(f"[sigma] {label}: {p}")
            return p

    if glob_patterns is not None:
        hits = []

        for pat in glob_patterns:
            hits.extend(sorted(sigdir.glob(pat)))

        hits_unique = []
        seen = set()

        for h in hits:
            hr = h.resolve()

            if hr in seen:
                continue

            if hr in exclude_paths:
                continue

            seen.add(hr)
            hits_unique.append(h)

        if len(hits_unique) > 0:
            print(f"[sigma] {label}: {hits_unique[0]}")
            return hits_unique[0]

    available = sorted([p.name for p in sigdir.glob("*.npy")])
    raise FileNotFoundError(
        f"Could not find {label} Sigma in {sigdir}.\n"
        f"Available .npy files: {available}"
    )

def decode_str_array(x):
    return np.asarray(
        [
            y.decode("utf-8") if isinstance(y, bytes) else str(y)
            for y in np.asarray(x, dtype=object)
        ],
        dtype=object,
    )

def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

def safe_nanmedian(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.median(x))

def pert_to_gene_safe(pert, gene_set):
    p0 = str(pert).strip()

    if p0 in gene_set:
        return p0

    p = p0
    p = re.sub(r"([_\-\s]+)(KD|KO|OE|overexp|overexpression)$", "", p, flags=re.IGNORECASE)
    p = re.sub(r"^(sg)(?=[A-Z0-9])", "", p)
    p = re.sub(r"^(sgRNA|gRNA|sgrna|grna|sg)([_\-\s]+)", "", p, flags=re.IGNORECASE)

    for s in ["_", "+", "-", "|", " "]:
        if s in p:
            p = p.split(s)[0]
            break

    if p in gene_set:
        return p

    if p0 in gene_set:
        return p0

    return None

def load_target_indices(folder, perts, genes):
    folder = Path(folder)

    gene_set = set(genes.tolist())
    gene_to_idx = {g: i for i, g in enumerate(genes.tolist())}

    target_map_path = folder / "perturbation_target_map.tsv"
    pert_to_target = {}

    if target_map_path.exists():
        tm = pd.read_csv(target_map_path, sep="\t")

        if "perturbation" in tm.columns and "target_gene" in tm.columns:
            for _, row in tm.iterrows():
                p = str(row["perturbation"])
                g = str(row["target_gene"])

                if g not in {"", "nan", "None"}:
                    pert_to_target[p] = g

    target_genes = []
    target_idx = []
    matched = []

    for p in perts:
        p = str(p)

        if p in pert_to_target and pert_to_target[p] in gene_to_idx:
            g = pert_to_target[p]
        else:
            g = pert_to_gene_safe(p, gene_set)

        if g is None or g not in gene_to_idx:
            target_genes.append("")
            target_idx.append(-1)
            matched.append(False)
        else:
            target_genes.append(g)
            target_idx.append(gene_to_idx[g])
            matched.append(True)

    return (
        np.asarray(target_genes, dtype=object),
        np.asarray(target_idx, dtype=np.int64),
        np.asarray(matched, dtype=bool),
    )

def get_sigma_sd(Sigma, eps=1e-12):
    """
    sd_i = sqrt(Sigma_ii), with zero protection.
    """
    diag = np.asarray(np.diag(Sigma), dtype=np.float64)
    sd = np.sqrt(np.maximum(diag, 0.0))
    sd_safe = sd.copy()
    sd_safe[~np.isfinite(sd_safe)] = 0.0
    sd_safe[sd_safe < eps] = np.nan
    return sd_safe

def full_sigma_columns(Sigma, gene_idx):
    gene_idx = np.asarray(gene_idx, dtype=np.int64)
    cols = np.asarray(Sigma[:, gene_idx], dtype=np.float64)

    if cols.ndim == 1:
        cols = cols[:, None]

    return cols.T.copy()

def full_sigma_columns_masked(Sigma, old_gene_idx, keep_gene_idx_old):
    """
    Pulls Sigma rows restricted to kept genes and columns for target genes.

    Output shape:
        batch x n_kept_genes

    This avoids materializing a full cleaned Sigma matrix.
    """
    old_gene_idx = np.asarray(old_gene_idx, dtype=np.int64)
    keep_gene_idx_old = np.asarray(keep_gene_idx_old, dtype=np.int64)

    cols = np.asarray(Sigma[np.ix_(keep_gene_idx_old, old_gene_idx)], dtype=np.float32)

    if cols.ndim == 1:
        cols = cols[:, None]

    return cols.T.copy()

def corr_columns_from_sigma(Sigma, gene_idx, sd, eps=1e-12):
    """
    Returns rows:
        R[g, :] = Sigma[:, g] / (sd[:] * sd[g])

    Output shape: n_targets x n_genes.
    """
    gene_idx = np.asarray(gene_idx, dtype=np.int64)

    Sigma_cols = np.asarray(Sigma[:, gene_idx], dtype=np.float64)
    if Sigma_cols.ndim == 1:
        Sigma_cols = Sigma_cols[:, None]

    sd_all = np.asarray(sd, dtype=np.float64)
    sd_g = sd_all[gene_idx]

    denom = sd_all[:, None] * sd_g[None, :]

    R_cols = np.full_like(Sigma_cols, np.nan, dtype=np.float64)
    good = np.isfinite(denom) & (np.abs(denom) > eps)
    R_cols[good] = Sigma_cols[good] / denom[good]

    R_rows = R_cols.T.copy()
    return R_rows

def standardize_dx(dx, sd, eps=1e-12):
    """
    dz = D^{-1} dx.
    """
    dx = np.asarray(dx, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)

    dz = np.full_like(dx, np.nan, dtype=np.float64)
    good = np.isfinite(sd) & (np.abs(sd) > eps)
    dz[:, good] = dx[:, good] / sd[good][None, :]
    return dz

def make_gene_splits(p, n_splits=5, train_frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    all_idx = np.arange(p)

    n_train = int(np.round(train_frac * p))
    n_train = max(1, min(p - 1, n_train))

    splits = []

    for s in range(n_splits):
        perm = rng.permutation(all_idx)
        train_idx = np.sort(perm[:n_train])
        test_idx = np.sort(perm[n_train:])

        splits.append(
            {
                "split": int(s),
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )

    return splits

def pearson_rows(y, yhat, eps=1e-12):
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)

    out = np.full(y.shape[0], np.nan, dtype=np.float64)

    for i in range(y.shape[0]):
        yi = y[i]
        pi = yhat[i]

        mask = np.isfinite(yi) & np.isfinite(pi)

        if mask.sum() < 3:
            continue

        yi = yi[mask]
        pi = pi[mask]

        yc = yi - yi.mean()
        pc = pi - pi.mean()

        num = float(np.dot(yc, pc))
        den = float(np.sqrt(np.dot(yc, yc)) * np.sqrt(np.dot(pc, pc)))

        if den > eps:
            out[i] = num / den

    return out

def weighted_beta_fit_corrcoords(
    dx_raw,
    R_basis,
    Sigma_basis,
    sd,
    target_gene_idx,
    train_idx,
    test_idx,
    eps=1e-12,
):
    """
    Exact raw-space-preserving fit in correlation coordinates.

    Coordinates:
        dz = dx / sd
        Rg = Sigma_g / (sd * sd_g)

    Preserve raw Euclidean LS:
        minimize ||dx - alpha Sigma_g||^2

    In corr coords:
        dx - alpha Sigma_g
        = D dz - alpha D Rg sd_g
        = D (dz - beta Rg), beta = alpha sd_g

    Therefore minimize:
        ||D(dz - beta Rg)||^2
        = sum_i sd_i^2 (dz_i - beta Rg_i)^2

    So:
        beta = sum_i sd_i^2 dz_i Rg_i / sum_i sd_i^2 Rg_i^2

    Then:
        alpha = beta / sd_g
    """
    dx_raw = np.asarray(dx_raw, dtype=np.float64)
    R_basis = np.asarray(R_basis, dtype=np.float64)
    Sigma_basis = np.asarray(Sigma_basis, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    target_gene_idx = np.asarray(target_gene_idx, dtype=np.int64)

    dz = standardize_dx(dx_raw, sd, eps=eps)

    sd_g = sd[target_gene_idx]

    w = sd ** 2
    w[~np.isfinite(w)] = np.nan

    dz_tr = dz[:, train_idx]
    R_tr = R_basis[:, train_idx]
    w_tr = w[train_idx][None, :]

    valid_tr = np.isfinite(dz_tr) & np.isfinite(R_tr) & np.isfinite(w_tr)

    num = np.nansum(np.where(valid_tr, w_tr * dz_tr * R_tr, 0.0), axis=1)
    den = np.nansum(np.where(valid_tr, w_tr * R_tr * R_tr, 0.0), axis=1)

    beta = np.full(dx_raw.shape[0], np.nan, dtype=np.float64)
    good = np.isfinite(den) & (den > eps) & np.isfinite(sd_g) & (np.abs(sd_g) > eps)
    beta[good] = num[good] / den[good]

    alpha = np.full_like(beta, np.nan)
    alpha[good] = beta[good] / sd_g[good]

    # z-coordinate predictions
    dz_hat_train = beta[:, None] * R_basis[:, train_idx]
    dz_hat_test = beta[:, None] * R_basis[:, test_idx]

    dz_train = dz[:, train_idx]
    dz_test = dz[:, test_idx]

    pearson_z_train = pearson_rows(dz_train, dz_hat_train, eps=eps)
    pearson_z_test = pearson_rows(dz_test, dz_hat_test, eps=eps)

    mse_z_train = np.nanmean((dz_train - dz_hat_train) ** 2, axis=1)
    mse_z_test = np.nanmean((dz_test - dz_hat_test) ** 2, axis=1)

    # raw-space predictions recovered from beta
    # beta Rg in z-space => D beta Rg = alpha Sigma_g in raw space
    dx_hat_train = alpha[:, None] * Sigma_basis[:, train_idx]
    dx_hat_test = alpha[:, None] * Sigma_basis[:, test_idx]

    dx_train = dx_raw[:, train_idx]
    dx_test = dx_raw[:, test_idx]

    pearson_raw_train = pearson_rows(dx_train, dx_hat_train, eps=eps)
    pearson_raw_test = pearson_rows(dx_test, dx_hat_test, eps=eps)

    mse_raw_train = np.nanmean((dx_train - dx_hat_train) ** 2, axis=1)
    mse_raw_test = np.nanmean((dx_test - dx_hat_test) ** 2, axis=1)

    return {
        "beta": beta,
        "alpha": alpha,

        "pearson_z_train": pearson_z_train,
        "pearson_z_test": pearson_z_test,
        "mse_z_train": mse_z_train,
        "mse_z_test": mse_z_test,

        "pearson_raw_train": pearson_raw_train,
        "pearson_raw_test": pearson_raw_test,
        "mse_raw_train": mse_raw_train,
        "mse_raw_test": mse_raw_test,
    }

def ordinary_beta_fit_corrspace(
    dx_raw,
    R_basis,
    sd,
    train_idx,
    test_idx,
    eps=1e-12,
):
    """
    Ordinary correlation-space fit.

    Solve:

        min_beta sum_i (dz_i - beta R_ig)^2

    over train genes only.

    This is NOT raw-space equivalent.

    It gives the beta that best matches standardized expression shifts.
    """
    dx_raw = np.asarray(dx_raw, dtype=np.float64)
    R_basis = np.asarray(R_basis, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)

    dz = standardize_dx(dx_raw, sd, eps=eps)

    dz_tr = dz[:, train_idx]
    R_tr = R_basis[:, train_idx]

    valid = np.isfinite(dz_tr) & np.isfinite(R_tr)

    num = np.nansum(np.where(valid, dz_tr * R_tr, 0.0), axis=1)
    den = np.nansum(np.where(valid, R_tr * R_tr, 0.0), axis=1)

    beta = np.full(dx_raw.shape[0], np.nan, dtype=np.float64)
    good = np.isfinite(den) & (den > eps)
    beta[good] = num[good] / den[good]

    # z-space predictions
    dz_train = dz[:, train_idx]
    dz_test = dz[:, test_idx]

    dz_hat_train = beta[:, None] * R_basis[:, train_idx]
    dz_hat_test = beta[:, None] * R_basis[:, test_idx]

    pearson_z_train = pearson_rows(dz_train, dz_hat_train, eps=eps)
    pearson_z_test = pearson_rows(dz_test, dz_hat_test, eps=eps)

    mse_z_train = np.nanmean((dz_train - dz_hat_train) ** 2, axis=1)
    mse_z_test = np.nanmean((dz_test - dz_hat_test) ** 2, axis=1)

    # Raw-space reconstruction from standardized prediction:
    # dx_hat_i = sd_i * dz_hat_i
    sd_train = sd[train_idx][None, :]
    sd_test = sd[test_idx][None, :]

    dx_train = dx_raw[:, train_idx]
    dx_test = dx_raw[:, test_idx]

    dx_hat_train = sd_train * dz_hat_train
    dx_hat_test = sd_test * dz_hat_test

    pearson_raw_train = pearson_rows(dx_train, dx_hat_train, eps=eps)
    pearson_raw_test = pearson_rows(dx_test, dx_hat_test, eps=eps)

    mse_raw_train = np.nanmean((dx_train - dx_hat_train) ** 2, axis=1)
    mse_raw_test = np.nanmean((dx_test - dx_hat_test) ** 2, axis=1)

    return {
        "beta": beta,

        "pearson_z_train": pearson_z_train,
        "pearson_z_test": pearson_z_test,
        "mse_z_train": mse_z_train,
        "mse_z_test": mse_z_test,

        "pearson_raw_train": pearson_raw_train,
        "pearson_raw_test": pearson_raw_test,
        "mse_raw_train": mse_raw_train,
        "mse_raw_test": mse_raw_test,
    }

def ordinary_beta_fit_corrspace_zonly(
    dx_raw,
    R_basis,
    sd,
    train_idx,
    test_idx,
    eps=1e-12,
):
    """
    Ordinary correlation-space fit.

    Solves:

        min_beta sum_{i in train} (dz_i - beta R_ig)^2

    and evaluates only in z-space.

    No raw-space reconstruction is performed.
    """
    dx_raw = np.asarray(dx_raw, dtype=np.float64)
    R_basis = np.asarray(R_basis, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)

    dz = standardize_dx(dx_raw, sd, eps=eps)

    dz_tr = dz[:, train_idx]
    R_tr = R_basis[:, train_idx]

    valid = np.isfinite(dz_tr) & np.isfinite(R_tr)

    num = np.nansum(np.where(valid, dz_tr * R_tr, 0.0), axis=1)
    den = np.nansum(np.where(valid, R_tr * R_tr, 0.0), axis=1)

    beta = np.full(dx_raw.shape[0], np.nan, dtype=np.float64)
    good = np.isfinite(den) & (den > eps)
    beta[good] = num[good] / den[good]

    dz_train = dz[:, train_idx]
    dz_test = dz[:, test_idx]

    dz_hat_train = beta[:, None] * R_basis[:, train_idx]
    dz_hat_test = beta[:, None] * R_basis[:, test_idx]

    pearson_z_train = pearson_rows(dz_train, dz_hat_train, eps=eps)
    pearson_z_test = pearson_rows(dz_test, dz_hat_test, eps=eps)

    mse_z_train = np.nanmean((dz_train - dz_hat_train) ** 2, axis=1)
    mse_z_test = np.nanmean((dz_test - dz_hat_test) ** 2, axis=1)

    return {
        "beta": beta,
        "pearson_z_train": pearson_z_train,
        "pearson_z_test": pearson_z_test,
        "mse_z_train": mse_z_train,
        "mse_z_test": mse_z_test,
    }

def fit_a_train_eval_test_pearson(y, basis, train_idx, test_idx, eps=1e-12):
    """
    Fit scalar a on train genes, evaluate Pearson on test genes.

    Formula:
        a = <dx_train, Sigma_train> / <Sigma_train, Sigma_train>

        Pearson_test = corr(dx_test, a Sigma_test)
    """
    y = np.asarray(y, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)

    y_tr = y[:, train_idx]
    b_tr = basis[:, train_idx]

    num = np.einsum("ij,ij->i", y_tr, b_tr, optimize=True)
    den = np.einsum("ij,ij->i", b_tr, b_tr, optimize=True)

    a = np.full(y.shape[0], np.nan, dtype=np.float64)
    good = den > eps
    a[good] = num[good] / den[good]

    yhat_tr = a[:, None] * b_tr

    y_te = y[:, test_idx]
    b_te = basis[:, test_idx]
    yhat_te = a[:, None] * b_te

    pearson_train = pearson_rows(y_tr, yhat_tr, eps=eps)
    pearson_test = pearson_rows(y_te, yhat_te, eps=eps)

    train_mse = np.nanmean((y_tr - yhat_tr) ** 2, axis=1)
    test_mse = np.nanmean((y_te - yhat_te) ** 2, axis=1)

    return {
        "pearson_train": pearson_train,
        "pearson_test": pearson_test,
        "a": a,
        "train_mse": train_mse,
        "test_mse": test_mse,
    }
