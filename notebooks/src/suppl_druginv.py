"""Notebook-only helper engine for Fig 5 / Fig S11 (sci-Plex drug inverse).

Implements: covariance-aware drug identification + ROC/PR/top-k,
the 5-panel summary figure, the dense CIPHER u* drug x gene heatmap, the
global drug-class clustering permutation test, and the per-cell-line variant.

This module is NOT part of the installable ``cipher`` package -- it is a
notebook-only helper for reproducing the supplementary figures.  Notable
path/config conventions:
  * hard-coded data/output paths -> environment-driven paths (see below);
  * the two ``main()`` drivers renamed to ``run_dense_ustar_heatmap`` (cell for
    the pan-cell-line u* heatmap) and ``run_cell_line_specific`` (per-cell-line
    three-panel figures) so both can coexist in one module;
  * inside those two cell-line-clustering drivers, the output directory global
    ``OUTDIR`` is referenced as ``CELL_LINE_OUTDIR`` and the u* heatmap driver's
    save DPI as ``USTAR_DPI`` (pure path/plotting-config swaps).

The cipher package now ships equivalent primitives
(``cipher.build_model(h_mode="within_cov")`` / ``cipher.recover_u`` /
``cipher.identify_perturbations`` / ``cipher.compute_covariance`` /
``cipher.select_hvg_dispersion``); the local versions are kept here verbatim so
the supplement reproduces bit-for-bit.
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
import json
import time
import warnings
from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd

from scipy.sparse import issparse, csr_matrix
from scipy.cluster.hierarchy import linkage, leaves_list, cophenet
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, ListedColormap

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (roc_curve, auc,
                                 precision_recall_curve, average_precision_score)
except Exception:  # pragma: no cover
    LogisticRegression = None
    roc_curve = auc = precision_recall_curve = average_precision_score = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, *a, **k):
            return x

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    requests = None
    HTTPAdapter = None
    Retry = None

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# ENVIRONMENT-DRIVEN PATHS (replace the notebook's hard-coded paths)
# ============================================================
DATA_DIR = os.environ.get("CIPHER_DATA_DIR", "")
OUTBASE = os.environ.get("SUPPL_OUT", os.path.join("resources", "repro", "fig5_S11"))

# full-gene sci-Plex3 lives in the cipher_data base (canonical), not under suppl/
DATA_PATH = Path(DATA_DIR) / "SrivatsanTrapnell2020_sciplex3.h5ad"

# shared u* heatmap output tree (cells for the dense u* heatmap + clustering)
OUTDIR = Path(OUTBASE) / "sciplex_u_star_heatmap"
INDIR = OUTDIR
MATRIX_CSV = INDIR / "drug_level_u_star_heatmap_z.csv"
CONDITION_CSV = INDIR / "condition_level_u_star.csv.gz"
CELL_LINE_OUTDIR = INDIR / "cell_line_specific_three_panel"

# save DPI for the pan-cell-line u* heatmap driver (source used 300 there)
USTAR_DPI = 300


# ============================================================
# (cell 1) covariance-aware drug identification helpers
# ============================================================

def _is_bad_label(x):
    if x is None:
        return True
    s = str(x).strip()
    if s == "":
        return True
    sl = s.lower()
    return sl in ("nan", "none", "null")

def _split_train_test_indices(labels, test_frac=0.5, seed=0, min_test=20):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(str)
    uniq = np.unique(labels)
    train_idx, test_idx = [], []
    for u in uniq:
        idx = np.where(labels == u)[0]
        rng.shuffle(idx)
        n = idx.size
        nt = int(np.round(test_frac * n))
        nt = max(min_test, nt) if n >= (min_test + 5) else max(1, int(np.round(test_frac * n)))
        nt = min(nt, n - 1)  # keep >=1 train
        test_idx.extend(idx[:nt].tolist())
        train_idx.extend(idx[nt:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)

def _mean_rows_sparse(X_csr):
    n = X_csr.shape[0]
    if n == 0:
        return np.zeros(X_csr.shape[1], dtype=np.float64)
    return np.asarray(X_csr.mean(axis=0)).ravel()

def _symmetrize(A):
    return 0.5 * (A + A.T)

def _shrink_cov(S, shrink=1e-3):
    S = _symmetrize(S)
    d = float(np.mean(np.diag(S)))
    return (1.0 - shrink) * S + shrink * d * np.eye(S.shape[0], dtype=S.dtype)

def _eig_psd(S, jitter=1e-8):
    S = _symmetrize(S) + jitter * np.eye(S.shape[0], dtype=S.dtype)
    lam, V = np.linalg.eigh(S)  # ascending
    lam = np.maximum(lam, 0.0)
    return lam, V

def _diag_VtSV(S, V):
    SV = S @ V
    return np.sum(V * SV, axis=0)

def _ll_diag_gauss(resid, hdiag, jitter=1e-12):
    h = np.maximum(hdiag, jitter)
    return float(-0.5 * (np.sum(np.log(h)) + np.sum((resid * resid) / h)))

def _estimate_u_ridge_in_Vbasis(y, lam, h, ridge_lambda=1.0, jitter=1e-12):
    hi = np.maximum(h, jitter)
    denom = (lam * lam) / hi + max(ridge_lambda, jitter)
    return (lam * y / hi) / (denom + jitter)

def _cosine_sim(A, B, eps=1e-12):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return An @ Bn.T

def _topk_by_freq(y, k):
    uniq, counts = np.unique(y, return_counts=True)
    order = np.argsort(counts)[::-1]
    return uniq[order][:k].tolist()

def _confusion_topk_arrays(y_true, y_pred, topk=40, normalize_rows=True):
    keep = _topk_by_freq(y_true, topk)
    d2i = {d: i for i, d in enumerate(keep)}
    C = np.zeros((len(keep), len(keep)), float)
    for t, p in zip(y_true, y_pred):
        if t in d2i and p in d2i:
            C[d2i[t], d2i[p]] += 1.0
    C_counts = C.copy()
    if normalize_rows:
        row = C.sum(axis=1, keepdims=True)
        C = C / np.maximum(row, 1e-12)
    return keep, C_counts, C

def _plot_confusion_topk(y_true, y_pred, topk=40, title="", normalize_rows=True, figsize=(11, 9), savepath=None):
    keep, C_counts, C_plot = _confusion_topk_arrays(y_true, y_pred, topk=topk, normalize_rows=normalize_rows)
    plt.figure(figsize=figsize)
    im = plt.imshow(C_plot, aspect="auto", interpolation="nearest", cmap="magma")
    cb = plt.colorbar(im, fraction=0.046, pad=0.02)
    cb.set_label("P(pred|true) row-normalized" if normalize_rows else "Count", fontsize=14)
    plt.xticks(range(len(keep)), keep, rotation=90, fontsize=10)
    plt.yticks(range(len(keep)), keep, fontsize=10)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title(title + f"  (top-{len(keep)})", fontsize=16)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=220, bbox_inches="tight")
    plt.show()
    return keep, C_counts, C_plot

def _plot_rank_hist(rank_true, title, max_rank=50, savepath=None):
    rank_true = np.asarray(rank_true, int)
    mx = int(min(rank_true.max(), max_rank))
    plt.figure(figsize=(7, 4))
    plt.hist(rank_true, bins=np.arange(1, mx + 2) - 0.5)
    plt.xlabel("Rank of TRUE label (1=best)", fontsize=14)
    plt.ylabel("Count (test drugs)", fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=220, bbox_inches="tight")
    plt.show()

def _log1p_mean_from_mean(mu, eps=1e-8):
    return np.log1p(np.maximum(mu, 0.0) + eps)

def _zscore_fit_transform(X):
    m = X.mean(axis=0, keepdims=True)
    s = X.std(axis=0, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return (X - m) / s, (m, s)

def _zscore_transform(X, params):
    m, s = params
    return (X - m) / s

def _roc_pr_from_scores(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    if np.unique(y).size < 2:
        return None
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    return (fpr, tpr, roc_auc, prec, rec, ap)

def _shuffle_columns_independently(X, seed=0):
    rng = np.random.default_rng(seed)
    Xs = X.copy()
    n, p = Xs.shape
    for j in range(p):
        perm = rng.permutation(n)
        Xs[:, j] = Xs[perm, j]
    return Xs

def _topk_accuracy_from_orders(orders, K):
    # orders: (n_samples, n_classes) array of indices sorted descending by score
    n = orders.shape[0]
    hit = 0
    for i in range(n):
        if i in set(orders[i, :K].tolist()):
            hit += 1
    return float(hit / max(n, 1))

def _topk_acc_from_rank(rank_true, K):
    rank_true = np.asarray(rank_true, int)
    return float(np.mean(rank_true <= int(K)))

def _pad_2d(list_of_lists, fill="", width=10, dtype=object):
    out = np.full((len(list_of_lists), width), fill, dtype=dtype)
    for i, row in enumerate(list_of_lists):
        m = min(width, len(row))
        out[i, :m] = np.array(row[:m], dtype=dtype)
    return out

def _run_lr(drugs_fit, mu_train, mu_test, mu0,
            train_idx, test_idx, labels_all, X_csr,
            Sigma0, cov_shrinkd, jitter, ridge_lambda, use_hdiag,
            control_label, tag="LR", save_topll_k=0):

    lam, V = _eig_psd(Sigma0, jitter=jitter)
    def proj_V(dX): return V.T @ dX

    ctrl_idx_local = np.where(labels_all == control_label)[0]
    n0_local = int(ctrl_idx_local.size)

    # Fit u on train
    uhat = {}
    for d in tqdm(drugs_fit, desc=f"{tag}: fit u_d (train)"):
        dX = (mu_train[d] - mu0).astype(np.float64)
        y = proj_V(dX)

        if use_hdiag:
            idx_tr = train_idx[labels_all[train_idx] == d]
            Xd = X_csr[idx_tr].toarray().astype(np.float64)
            nd = int(Xd.shape[0])
            Sd = _shrink_cov(np.cov(Xd, rowvar=False), shrink=cov_shrinkd)
            diagVtSdV = _diag_VtSV(Sd, V)
            h = (lam / max(n0_local, 1)) + (diagVtSdV / max(nd, 1))
            h = np.maximum(h, 1e-12)
        else:
            idx_tr = train_idx[labels_all[train_idx] == d]
            nd = int(idx_tr.size)
            alpha = (1.0 / max(n0_local, 1)) + (1.0 / max(nd, 1))
            h = np.maximum(alpha * lam, 1e-12)

        uhat[d] = _estimate_u_ridge_in_Vbasis(y, lam, h, ridge_lambda=ridge_lambda)

    # Score on test
    nD = len(drugs_fit)
    y_true = np.array(drugs_fit, dtype=object)
    y_pred = np.empty(nD, dtype=object)
    rank_true = np.zeros(nD, dtype=int)
    margin = np.zeros(nD, dtype=float)
    orders_all = np.zeros((nD, nD), dtype=int)

    # NEW: store only top-K LLs (and labels) per test drug
    top_labels = None
    top_lls = None
    if int(save_topll_k) > 0:
        k = int(min(save_topll_k, nD))
        top_labels = np.empty((nD, k), dtype=object)
        top_lls = np.full((nD, k), np.nan, dtype=float)

    for i, d in enumerate(tqdm(drugs_fit, desc=f"{tag}: score+order (test)")):
        dX_te = (mu_test[d] - mu0).astype(np.float64)
        y_te = proj_V(dX_te)

        if use_hdiag:
            idx_te = test_idx[labels_all[test_idx] == d]
            Xd_te = X_csr[idx_te].toarray().astype(np.float64)
            ndt = int(Xd_te.shape[0])
            Sd_te = _shrink_cov(np.cov(Xd_te, rowvar=False), shrink=cov_shrinkd)
            diagVtSdV = _diag_VtSV(Sd_te, V)
            h_te = (lam / max(n0_local, 1)) + (diagVtSdV / max(ndt, 1))
            h_te = np.maximum(h_te, 1e-12)
        else:
            idx_te = test_idx[labels_all[test_idx] == d]
            ndt = int(idx_te.size)
            alpha = (1.0 / max(n0_local, 1)) + (1.0 / max(ndt, 1))
            h_te = np.maximum(alpha * lam, 1e-12)

        lls = np.array([
            _ll_diag_gauss(y_te - lam * uhat[cand], h_te, jitter=1e-12)
            for cand in drugs_fit
        ], dtype=float)

        order = np.argsort(lls)[::-1]
        orders_all[i, :] = order
        y_pred[i] = drugs_fit[int(order[0])]
        rank_true[i] = int(np.where(order == i)[0][0] + 1)
        margin[i] = float(lls[order[0]] - lls[order[1]]) if nD >= 2 else 0.0

        if top_labels is not None:
            k = top_labels.shape[1]
            top_idx = order[:k]
            top_labels[i, :] = np.array([drugs_fit[j] for j in top_idx], dtype=object)
            top_lls[i, :] = lls[top_idx]

    return y_true, y_pred, rank_true, margin, orders_all, top_labels, top_lls


# ============================================================
# (cell 2) 5-panel summary figure helpers + style
# ============================================================

FONTSIZE = 20
CORNFLOWER = "#6495ED"
PURPLE = "#9467bd"
SALMON = "#FA8072"
COLORS = {"LR-TRUE": CORNFLOWER, "LR-MF": CORNFLOWER, "LFC": PURPLE, "CLF": SALMON}
LINESTYLE = {"LR-TRUE": "-", "LR-MF": "--", "LFC": "-", "CLF": "-"}

def _pick_dataset_name(outdir):
    cand = sorted(glob.glob(os.path.join(outdir, "*__ALL_METHODS__predictions_and_metrics.npz")))
    if len(cand) == 0:
        raise FileNotFoundError(f"No '*__ALL_METHODS__predictions_and_metrics.npz' in {outdir}")
    base = os.path.basename(cand[-1])
    return base.split("__ALL_METHODS__predictions_and_metrics.npz")[0]

def _safe_arr(z, key, default=None):
    return z[key] if key in z.files else default

def _plot_example(ax, cand_labels, scores, true_label=None,
                  other_color=CORNFLOWER, true_color=SALMON):
    # scores are LOG LIKELIHOODS (LL). Sort desc and barh.
    cand_labels = list(map(str, cand_labels))
    scores = np.asarray(scores, float)
    order = np.argsort(scores)[::-1]
    cand_labels = [cand_labels[i] for i in order]
    scores = scores[order]

    t = None if true_label is None else str(true_label)

    # color ONLY the TRUE candidate bar
    colors = [true_color if (t is not None and lab == t) else other_color for lab in cand_labels]

    y = np.arange(len(cand_labels))
    ax.barh(y, scores, color=colors, alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(cand_labels, fontsize=FONTSIZE * 0.6)
    ax.invert_yaxis()
    ax.set_xlabel("log-likelihood (LR-TRUE)", fontsize=FONTSIZE * 0.9)
    ax.tick_params(axis="x", labelsize=FONTSIZE * 0.75)

def _plot_topk(ax, ks, curves_dict):
    ax.set_xlabel("k", fontsize=FONTSIZE)
    ax.set_ylabel("Top-k accuracy", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE * 0.75)
    for name, y in curves_dict.items():
        if y is None:
            continue
        ax.plot(ks, y, linewidth=3.0, color=COLORS[name], linestyle=LINESTYLE[name], label=name)
    ax.set_xlim(float(np.min(ks)), float(np.max(ks)))
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=FONTSIZE * 0.7, frameon=True, loc="lower right")

def _plot_roc(ax, curves):
    ax.set_xlabel("False Positive Rate", fontsize=FONTSIZE)
    ax.set_ylabel("True Positive Rate", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE * 0.75)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.5, color="0.6")
    for name, (fpr, tpr, aucv) in curves.items():
        if fpr is None or tpr is None or len(fpr) == 0:
            continue
        ax.plot(
            fpr, tpr,
            linewidth=3.0,
            color=COLORS[name],
            linestyle=LINESTYLE[name],
            label=f"{name} AUC={aucv:.3f}" if np.isfinite(aucv) else name
        )
    ax.set_xlim(0, 1.)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=FONTSIZE * 0.7, frameon=True, loc="lower right")

def _plot_pr(ax, curves):
    ax.set_xlabel("Recall", fontsize=FONTSIZE)
    ax.set_ylabel("Precision", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE * 0.75)
    for name, (rec, prec, apv) in curves.items():
        if rec is None or prec is None or len(rec) == 0:
            continue
        ax.plot(
            rec, prec,
            linewidth=3.0,
            color=COLORS[name],
            linestyle=LINESTYLE[name],
            label=f"{name} AP={apv:.3f}" if np.isfinite(apv) else name
        )
    ax.set_xlim(0, 1.)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=FONTSIZE * 0.7, frameon=True, loc="lower left")

def _choose_best_example_by_ll(allz, topk_bars, fallback_index=0):
    """
    Criterion:
      - top1 LL > 0
      - top2 LL < 0
      - choose example maximizing (top1 - top2)
    Fallback to fallback_index if needed.
    """
    labs = _safe_arr(allz, "lr_top20_labels", None)
    lls  = _safe_arr(allz, "lr_top20_lls", None)
    if labs is None or lls is None or lls.size == 0:
        return int(fallback_index), None, None

    k = int(min(topk_bars, lls.shape[1]))
    top1 = lls[:, 0]
    top2 = lls[:, 1] if lls.shape[1] >= 2 else np.full_like(top1, np.nan)
    ok = np.isfinite(top1) & np.isfinite(top2) & (top1 > 0.0) & (top2 < 0.0)
    if np.any(ok):
        margin = top1 - top2
        i_best = int(np.argmax(np.where(ok, margin, -np.inf)))
        return i_best, labs[i_best, :k].astype(object), lls[i_best, :k].astype(float)

    ok2 = np.isfinite(top1) & np.isfinite(top2)
    if np.any(ok2):
        margin = top1 - top2
        i_best = int(np.argmax(np.where(ok2, margin, -np.inf)))
        return i_best, labs[i_best, :k].astype(object), lls[i_best, :k].astype(float)

    return int(fallback_index), None, None


# ============================================================
# (cell 24) dense CIPHER u* drug x gene heatmap
# ============================================================

MATRIX_SOURCE = "X"

ENSEMBL_VAR_COLUMN = "ensembl_id"

PERT_KEY = "perturbation"

CELL_LINE_KEY = "cell_line"

DOSE_KEY = "dose_value"

CONTROL_LABELS = {
    "control",
    "ctrl",
    "vehicle",
    "dmso",
    "untreated",
    "mock",
    "negative_control",
    "negative control",
}

N_HVG = 1000

MIN_CONTROL_MEAN = 0.02

MIN_CONTROL_DETECTION = 0.005

MIN_CELL_LINES_FOR_HVG = 1

DOSE_MODE = "highest"

CELL_LINES_TO_USE = None

MIN_TREATED_CELLS = 30

MIN_CONTROL_CELLS = 100

MAX_CONTROL_CELLS_FOR_COV = 15000

COV_SHRINK = 1e-3

RIDGE_PRECISION = 1.0

JITTER = 1e-8

H_FLOOR = 1e-12

SEED = 7

NORMALIZE_EACH_CONDITION_U_BY_RMS = True

CONDITION_WEIGHTING = "equal"

ROW_ZSCORE_FOR_HEATMAP = True

HEATMAP_CLIP = 3.0

MAX_GENE_LABELS = 0

ENSEMBL_SERVER = "https://rest.ensembl.org"

ENSEMBL_BATCH_SIZE = 900

REQUEST_TIMEOUT = 60

FORCE_REFRESH_SYMBOL_CACHE = False

def normalize_label(value):
    text = str(value).strip().lower()

    text = text.replace(
        "_",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text

NORMALIZED_CONTROL_LABELS = {
    normalize_label(value)
    for value in CONTROL_LABELS
}

def is_control(value):
    return (
        normalize_label(value)
        in NORMALIZED_CONTROL_LABELS
    )

def stable_ensembl_id(value):
    """
    Extract a stable human Ensembl gene ID from strings such as:

        ENSG00000146648
        ENSG00000146648.17
        ENSG00000146648|EGFR
    """
    match = re.search(
        r"ENSG\d{6,}(?:\.\d+)?",
        str(value).upper(),
    )

    if match is None:
        return None

    return (
        match.group(0)
        .split(".")[0]
    )

def as_csr(X):
    if issparse(X):
        return X.tocsr()

    return csr_matrix(
        np.asarray(X)
    )

def mean_and_variance(X):
    """
    Gene-wise sample mean and unbiased sample variance.
    """
    n_cells = X.shape[0]

    if n_cells == 0:
        return (
            np.zeros(
                X.shape[1],
                dtype=float,
            ),
            np.zeros(
                X.shape[1],
                dtype=float,
            ),
        )

    mean = np.asarray(
        X.mean(axis=0)
    ).ravel().astype(float)

    second_moment = np.asarray(
        X.multiply(X).mean(axis=0)
    ).ravel().astype(float)

    variance = np.maximum(
        second_moment - mean * mean,
        0.0,
    )

    if n_cells > 1:
        variance *= (
            n_cells
            / (n_cells - 1)
        )

    return mean, variance

def detection_fraction(X):
    if X.shape[0] == 0:
        return np.zeros(
            X.shape[1],
            dtype=float,
        )

    return np.asarray(
        (X != 0).mean(axis=0)
    ).ravel().astype(float)

def parse_dose(value):
    """
    Convert common dose strings into a sortable numeric value.

    Plain numeric values remain unchanged.
    Strings with nM/uM/mM/M are converted to molar.
    """
    text = (
        str(value)
        .strip()
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
    )

    try:
        return float(text)
    except ValueError:
        pass

    match = re.search(
        r"([-+]?\d*\.?\d+(?:e[-+]?\d+)?)"
        r"\s*(nm|um|mm|m)?",
        text,
    )

    if match is None:
        return np.nan

    value_numeric = float(
        match.group(1)
    )

    unit = match.group(2)

    conversion = {
        None: 1.0,
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
    }

    return (
        value_numeric
        * conversion[unit]
    )

def row_rms_normalize(matrix):
    rms = np.sqrt(
        np.mean(
            matrix * matrix,
            axis=1,
            keepdims=True,
        )
    )

    return (
        matrix
        / np.maximum(
            rms,
            H_FLOOR,
        )
    )

def row_zscore(matrix):
    row_mean = matrix.mean(
        axis=1,
        keepdims=True,
    )

    row_std = matrix.std(
        axis=1,
        keepdims=True,
    )

    return (
        matrix - row_mean
    ) / np.maximum(
        row_std,
        H_FLOOR,
    )

def hierarchical_cluster_order(
    matrix,
    cluster_rows=True,
):
    """
    Correlation-distance hierarchical clustering.
    """
    data = (
        matrix
        if cluster_rows
        else matrix.T
    )

    if data.shape[0] <= 1:
        return np.arange(
            data.shape[0]
        )

    distances = pdist(
        data,
        metric="correlation",
    )

    distances = np.nan_to_num(
        distances,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    linkage_matrix = linkage(
        distances,
        method="average",
    )

    return leaves_list(
        linkage_matrix
    )

def make_http_session():
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.0,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "POST",
        ],
        respect_retry_after_header=True,
    )

    session = requests.Session()

    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=retry
        ),
    )

    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": (
            "SciPlex-u-star-heatmap/1.0 "
            "(academic research)"
        ),
    })

    return session

try:
    HTTP_SESSION = make_http_session() if requests is not None else None
except Exception:  # pragma: no cover
    HTTP_SESSION = None

def batched(values, batch_size):
    values = list(values)

    for start in range(
        0,
        len(values),
        batch_size,
    ):
        yield values[
            start:start + batch_size
        ]

def map_ensembl_ids_to_symbols(
    ensembl_ids,
):
    """
    Map stable ENSG IDs to Ensembl display names.

    This is used only for labels. If the API is unavailable, the
    corresponding Ensembl IDs remain as the heatmap labels.
    """
    symbol_map = {}

    cache_dir = (
        OUTDIR
        / "ensembl_symbol_cache"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    unique_ids = sorted(
        set(ensembl_ids)
    )

    batches = list(
        batched(
            unique_ids,
            ENSEMBL_BATCH_SIZE,
        )
    )

    for batch in tqdm(
        batches,
        desc="Mapping Ensembl IDs to symbols",
    ):
        cache_hash = sha256(
            json.dumps(
                batch,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        cache_path = (
            cache_dir
            / f"{cache_hash}.json"
        )

        if (
            cache_path.exists()
            and not FORCE_REFRESH_SYMBOL_CACHE
        ):
            response = json.loads(
                cache_path.read_text()
            )

        else:
            try:
                response_object = HTTP_SESSION.post(
                    ENSEMBL_SERVER
                    + "/lookup/id",
                    json={
                        "ids": batch,
                    },
                    timeout=REQUEST_TIMEOUT,
                )

                response_object.raise_for_status()

                response = (
                    response_object.json()
                )

                cache_path.write_text(
                    json.dumps(
                        response,
                        indent=2,
                    )
                )

                time.sleep(0.1)

            except Exception as error:
                print(
                    "WARNING: Ensembl symbol lookup "
                    f"failed for one batch: {error}"
                )

                response = {}

        if not isinstance(
            response,
            dict,
        ):
            continue

        for ensembl_id in batch:
            record = response.get(
                ensembl_id
            )

            if not isinstance(
                record,
                dict,
            ):
                continue

            display_name = str(
                record.get(
                    "display_name"
                )
                or ""
            ).strip()

            if display_name:
                symbol_map[
                    ensembl_id
                ] = display_name.upper()

    return symbol_map

def make_unique_gene_labels(
    ensembl_ids,
    symbol_map,
):
    """
    Use symbols where available. If two Ensembl IDs map to the same
    symbol, append the Ensembl ID so column labels remain unique.
    """
    base_labels = [
        symbol_map.get(
            ensembl_id,
            ensembl_id,
        )
        for ensembl_id
        in ensembl_ids
    ]

    label_counts = (
        pd.Series(base_labels)
        .value_counts()
        .to_dict()
    )

    output = []

    for label, ensembl_id in zip(
        base_labels,
        ensembl_ids,
    ):
        if label_counts[label] == 1:
            output.append(label)
        else:
            output.append(
                f"{label}|{ensembl_id}"
            )

    return np.asarray(
        output,
        dtype=object,
    )

def solve_dense_u_star(
    response,
    treated_variance,
    n_treated,
    n_control,
    covariance_eigenvalues,
    covariance_eigenvectors,
):
    """
    Solve the dense ridge/MAP intervention vector:

        Delta x approximately Sigma_0 u

    in the eigenbasis of the control covariance.

    The sampling covariance is approximated mode-wise as:

        H approximately Sigma_0 / n_control
                      + diag(var_treated) / n_treated

    Returns:
        u* in the original 500-gene coordinates.
    """
    eigenvalues = (
        covariance_eigenvalues
    )

    eigenvectors = (
        covariance_eigenvectors
    )

    eigenvectors_squared = (
        eigenvectors
        * eigenvectors
    )

    response_in_eigenbasis = (
        eigenvectors.T
        @ response
    )

    projected_treated_variance = (
        treated_variance
        @ eigenvectors_squared
    )

    sampling_variance = (
        eigenvalues
        / max(
            int(n_control),
            1,
        )
        + projected_treated_variance
        / max(
            int(n_treated),
            1,
        )
    )

    sampling_variance = np.maximum(
        sampling_variance,
        H_FLOOR,
    )

    posterior_precision = (
        eigenvalues
        * eigenvalues
        / sampling_variance
        + max(
            RIDGE_PRECISION,
            H_FLOOR,
        )
    )

    u_in_eigenbasis = (
        eigenvalues
        * response_in_eigenbasis
        / sampling_variance
    ) / posterior_precision

    return (
        eigenvectors
        @ u_in_eigenbasis
    )

def run_dense_ustar_heatmap():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            DATA_PATH.resolve()
        )

    print(
        f"Loading {DATA_PATH}"
    )

    adata = ad.read_h5ad(
        DATA_PATH
    )

    # ------------------------------------------------------------
    # Choose expression source.
    # ------------------------------------------------------------

    if MATRIX_SOURCE == "X":
        X_full = adata.X
        var_df = adata.var.copy()

    elif MATRIX_SOURCE == "raw":
        if adata.raw is None:
            raise ValueError(
                "MATRIX_SOURCE='raw' but adata.raw is None"
            )

        X_full = adata.raw.X
        var_df = adata.raw.var.copy()

    elif MATRIX_SOURCE.startswith(
        "layer:"
    ):
        layer_name = (
            MATRIX_SOURCE
            .split(
                ":",
                1,
            )[1]
        )

        if layer_name not in adata.layers:
            raise KeyError(
                f"Missing layer: {layer_name}"
            )

        X_full = adata.layers[
            layer_name
        ]

        var_df = adata.var.copy()

    else:
        raise ValueError(
            "MATRIX_SOURCE must be 'X', "
            "'raw', or 'layer:<name>'"
        )

    X_full = as_csr(
        X_full
    )

    for key in [
        PERT_KEY,
        CELL_LINE_KEY,
        DOSE_KEY,
    ]:
        if key not in adata.obs.columns:
            raise KeyError(
                f"Missing adata.obs['{key}']. "
                f"Available columns: "
                f"{list(adata.obs.columns)}"
            )

    if ENSEMBL_VAR_COLUMN not in var_df.columns:
        raise KeyError(
            f"Missing adata.var['{ENSEMBL_VAR_COLUMN}']. "
            f"Available columns: "
            f"{list(var_df.columns)}"
        )

    drug = (
        adata.obs[PERT_KEY]
        .astype(str)
        .to_numpy()
    )

    cell_line = (
        adata.obs[CELL_LINE_KEY]
        .astype(str)
        .to_numpy()
    )

    dose = (
        adata.obs[DOSE_KEY]
        .astype(str)
        .to_numpy()
    )

    control_mask = np.array(
        [
            is_control(value)
            for value in drug
        ],
        dtype=bool,
    )

    if not np.any(
        control_mask
    ):
        raise RuntimeError(
            "No control cells were found. "
            "Update CONTROL_LABELS."
        )

    # ============================================================
    # IDENTIFY VALID UNIQUE ENSEMBL FEATURES
    # ============================================================

    raw_ensembl_ids = np.asarray(
        var_df[
            ENSEMBL_VAR_COLUMN
        ]
    ).astype(str)

    stable_ids_all = np.array(
        [
            stable_ensembl_id(value)
            or ""
            for value
            in raw_ensembl_ids
        ],
        dtype=object,
    )

    valid_feature_indices = np.where(
        stable_ids_all != ""
    )[0]

    if len(
        valid_feature_indices
    ) == 0:
        raise RuntimeError(
            "No valid ENSG identifiers were found in "
            f"adata.var['{ENSEMBL_VAR_COLUMN}']."
        )

    if "ncounts" in var_df.columns:
        feature_ncounts = (
            pd.to_numeric(
                var_df["ncounts"],
                errors="coerce",
            )
            .fillna(-np.inf)
            .to_numpy()
        )
    else:
        feature_ncounts = np.zeros(
            adata.n_vars,
            dtype=float,
        )

    feature_table = pd.DataFrame({
        "original_feature_index":
            valid_feature_indices,
        "ensembl_id":
            stable_ids_all[
                valid_feature_indices
            ],
        "ncounts":
            feature_ncounts[
                valid_feature_indices
            ],
    })

    # Retain one matrix column per Ensembl ID.
    # For duplicates, keep the feature with the largest ncounts.
    feature_table = (
        feature_table
        .sort_values(
            [
                "ensembl_id",
                "ncounts",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            "ensembl_id",
            keep="first",
        )
        .sort_values(
            "original_feature_index"
        )
        .reset_index(
            drop=True
        )
    )

    unique_feature_indices = (
        feature_table[
            "original_feature_index"
        ]
        .to_numpy(
            dtype=int
        )
    )

    unique_ensembl_ids = (
        feature_table[
            "ensembl_id"
        ]
        .astype(str)
        .to_numpy()
    )

    X_unique = X_full[
        :,
        unique_feature_indices,
    ].tocsr()

    print(
        f"Total h5ad features: {adata.n_vars}"
    )

    print(
        "Valid unique Ensembl genes:",
        len(unique_ensembl_ids),
    )

    # ============================================================
    # SELECT EXACTLY 500 CONTROL HVGs
    # ============================================================

    control_cell_lines = sorted(
        set(
            cell_line[
                control_mask
            ]
        )
    )

    if CELL_LINES_TO_USE is not None:
        requested_lines = set(
            map(
                str,
                CELL_LINES_TO_USE,
            )
        )

        control_cell_lines = [
            line
            for line
            in control_cell_lines
            if line in requested_lines
        ]

    if not control_cell_lines:
        raise RuntimeError(
            "No requested cell lines contain controls."
        )

    dispersion_rank_sum = np.zeros(
        len(unique_ensembl_ids),
        dtype=float,
    )

    dispersion_rank_count = np.zeros(
        len(unique_ensembl_ids),
        dtype=int,
    )

    hvg_detail_tables = []

    for line_name in control_cell_lines:
        control_indices = np.where(
            control_mask
            & (
                cell_line
                == line_name
            )
        )[0]

        if (
            len(control_indices)
            < MIN_CONTROL_CELLS
        ):
            print(
                f"Skipping {line_name} for HVG ranking: "
                f"only {len(control_indices)} controls"
            )

            continue

        control_mean, control_variance = (
            mean_and_variance(
                X_unique[
                    control_indices
                ]
            )
        )

        control_detection = (
            detection_fraction(
                X_unique[
                    control_indices
                ]
            )
        )

        eligible = (
            (
                control_mean
                >= MIN_CONTROL_MEAN
            )
            & (
                control_detection
                >= MIN_CONTROL_DETECTION
            )
        )

        dispersion = np.full(
            len(unique_ensembl_ids),
            np.nan,
            dtype=float,
        )

        dispersion[
            eligible
        ] = np.log1p(
            control_variance[
                eligible
            ]
            / np.maximum(
                control_mean[
                    eligible
                ],
                H_FLOOR,
            )
        )

        dispersion_percentile = np.full(
            len(unique_ensembl_ids),
            np.nan,
            dtype=float,
        )

        eligible_indices = np.where(
            eligible
        )[0]

        if len(
            eligible_indices
        ):
            dispersion_percentile[
                eligible_indices
            ] = (
                pd.Series(
                    dispersion[
                        eligible_indices
                    ]
                )
                .rank(
                    pct=True,
                    method="average",
                )
                .to_numpy()
            )

        finite = np.isfinite(
            dispersion_percentile
        )

        dispersion_rank_sum[
            finite
        ] += dispersion_percentile[
            finite
        ]

        dispersion_rank_count[
            finite
        ] += 1

        hvg_detail_tables.append(
            pd.DataFrame({
                "cell_line":
                    line_name,
                "ensembl_id":
                    unique_ensembl_ids,
                "control_mean":
                    control_mean,
                "control_variance":
                    control_variance,
                "control_detection":
                    control_detection,
                "dispersion_percentile":
                    dispersion_percentile,
            })
        )

    if not hvg_detail_tables:
        raise RuntimeError(
            "No cell line had enough control cells "
            "for HVG selection."
        )

    average_dispersion_rank = np.divide(
        dispersion_rank_sum,
        dispersion_rank_count,
        out=np.full(
            len(unique_ensembl_ids),
            np.nan,
            dtype=float,
        ),
        where=(
            dispersion_rank_count
            > 0
        ),
    )

    hvg_candidates = np.where(
        (
            dispersion_rank_count
            >= MIN_CELL_LINES_FOR_HVG
        )
        & np.isfinite(
            average_dispersion_rank
        )
    )[0]

    if len(
        hvg_candidates
    ) < N_HVG:
        raise RuntimeError(
            f"Only {len(hvg_candidates)} genes passed "
            f"the HVG filters; cannot select {N_HVG}. "
            "Lower MIN_CONTROL_MEAN or "
            "MIN_CONTROL_DETECTION."
        )

    selected_local_indices = (
        hvg_candidates[
            np.argsort(
                average_dispersion_rank[
                    hvg_candidates
                ]
            )[::-1][
                :N_HVG
            ]
        ]
    )

    selected_ensembl_ids = (
        unique_ensembl_ids[
            selected_local_indices
        ]
    )

    X = X_unique[
        :,
        selected_local_indices,
    ].tocsr()

    print(
        f"Selected exactly {X.shape[1]} control HVGs"
    )

    # ============================================================
    # CREATE READABLE GENE LABELS
    # ============================================================

    symbol_map = (
        map_ensembl_ids_to_symbols(
            selected_ensembl_ids
        )
    )

    gene_labels = make_unique_gene_labels(
        selected_ensembl_ids,
        symbol_map,
    )

    selected_annotation = pd.DataFrame({
        "matrix_column":
            np.arange(
                N_HVG
            ),
        "ensembl_id":
            selected_ensembl_ids,
        "gene_symbol": [
            symbol_map.get(
                ensembl_id,
                "",
            )
            for ensembl_id
            in selected_ensembl_ids
        ],
        "heatmap_label":
            gene_labels,
        "average_dispersion_percentile":
            average_dispersion_rank[
                selected_local_indices
            ],
        "n_cell_lines_contributing":
            dispersion_rank_count[
                selected_local_indices
            ],
        "original_h5ad_feature_index":
            unique_feature_indices[
                selected_local_indices
            ],
    })

    selected_annotation.to_csv(
        OUTDIR
        / "selected_500_hvgs.csv",
        index=False,
    )

    all_hvg_details = pd.concat(
        hvg_detail_tables,
        ignore_index=True,
    )

    all_hvg_details[
        all_hvg_details[
            "ensembl_id"
        ].isin(
            selected_ensembl_ids
        )
    ].to_csv(
        OUTDIR
        / "selected_500_hvgs_by_cell_line.csv",
        index=False,
    )

    # ============================================================
    # BUILD DRUG/CELL-LINE/DOSE CONDITIONS
    # ============================================================

    cell_table = pd.DataFrame({
        "cell_index":
            np.arange(
                adata.n_obs,
                dtype=int,
            ),
        "drug":
            drug,
        "cell_line":
            cell_line,
        "dose":
            dose,
        "dose_numeric": [
            parse_dose(value)
            for value in dose
        ],
        "is_control":
            control_mask,
    })

    treated_cells = cell_table.loc[
        (
            ~cell_table[
                "is_control"
            ]
        )
        & (
            cell_table[
                "cell_line"
            ].isin(
                control_cell_lines
            )
        )
    ].copy()

    condition_table = (
        treated_cells
        .groupby(
            [
                "drug",
                "cell_line",
                "dose",
                "dose_numeric",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "n_cells"
        )
        .reset_index()
    )

    condition_table = condition_table[
        condition_table[
            "n_cells"
        ] >= MIN_TREATED_CELLS
    ].copy()

    if DOSE_MODE == "highest":
        selected_conditions = []

        for _, group in condition_table.groupby(
            [
                "drug",
                "cell_line",
            ],
            sort=False,
        ):
            numeric_doses = group[
                "dose_numeric"
            ].to_numpy(
                dtype=float
            )

            if np.isfinite(
                numeric_doses
            ).any():
                selected_conditions.append(
                    group.loc[
                        group[
                            "dose_numeric"
                        ].idxmax()
                    ]
                )
            else:
                selected_conditions.append(
                    group.iloc[-1]
                )

        condition_table = pd.DataFrame(
            selected_conditions
        ).reset_index(
            drop=True
        )

    elif DOSE_MODE != "all":
        raise ValueError(
            "DOSE_MODE must be 'highest' or 'all'."
        )

    print(
        "Candidate drug/cell-line conditions:",
        len(condition_table),
    )

    # ============================================================
    # FIT u* FOR EACH DRUG/CELL-LINE CONDITION
    # ============================================================

    condition_u_vectors = []
    condition_metadata = []

    for line_number, line_name in enumerate(
        control_cell_lines
    ):
        print(
            "\n" + "=" * 70
        )

        print(
            "CELL LINE:",
            line_name,
        )

        print(
            "=" * 70
        )

        control_indices = np.where(
            control_mask
            & (
                cell_line
                == line_name
            )
        )[0]

        if (
            len(control_indices)
            < MIN_CONTROL_CELLS
        ):
            print(
                f"Skipping {line_name}: "
                f"only {len(control_indices)} controls"
            )

            continue

        control_mean, _ = mean_and_variance(
            X[
                control_indices
            ]
        )

        covariance_indices = (
            control_indices.copy()
        )

        if (
            len(covariance_indices)
            > MAX_CONTROL_CELLS_FOR_COV
        ):
            line_rng = np.random.default_rng(
                SEED
                + 1009
                * (
                    line_number
                    + 1
                )
            )

            covariance_indices = line_rng.choice(
                covariance_indices,
                MAX_CONTROL_CELLS_FOR_COV,
                replace=False,
            )

        dense_controls = (
            X[
                covariance_indices
            ]
            .toarray()
            .astype(float)
        )

        control_covariance = np.cov(
            dense_controls,
            rowvar=False,
        )

        control_covariance = (
            0.5
            * (
                control_covariance
                + control_covariance.T
            )
        )

        diagonal_mean = float(
            np.mean(
                np.diag(
                    control_covariance
                )
            )
        )

        control_covariance = (
            (
                1.0
                - COV_SHRINK
            )
            * control_covariance
            + COV_SHRINK
            * diagonal_mean
            * np.eye(
                N_HVG
            )
            + JITTER
            * np.eye(
                N_HVG
            )
        )

        covariance_eigenvalues, covariance_eigenvectors = (
            np.linalg.eigh(
                control_covariance
            )
        )

        covariance_eigenvalues = np.maximum(
            covariance_eigenvalues,
            JITTER,
        )

        del dense_controls
        del control_covariance

        line_conditions = condition_table[
            condition_table[
                "cell_line"
            ] == line_name
        ]

        for _, condition in tqdm(
            line_conditions.iterrows(),
            total=len(
                line_conditions
            ),
            desc=f"Solving u*: {line_name}",
        ):
            treated_indices = np.where(
                (
                    ~control_mask
                )
                & (
                    drug
                    == str(
                        condition[
                            "drug"
                        ]
                    )
                )
                & (
                    cell_line
                    == line_name
                )
                & (
                    dose
                    == str(
                        condition[
                            "dose"
                        ]
                    )
                )
            )[0]

            if (
                len(treated_indices)
                < MIN_TREATED_CELLS
            ):
                continue

            treated_mean, treated_variance = (
                mean_and_variance(
                    X[
                        treated_indices
                    ]
                )
            )

            response = (
                treated_mean
                - control_mean
            )

            u_star = solve_dense_u_star(
                response=response,
                treated_variance=treated_variance,
                n_treated=len(
                    treated_indices
                ),
                n_control=len(
                    control_indices
                ),
                covariance_eigenvalues=
                    covariance_eigenvalues,
                covariance_eigenvectors=
                    covariance_eigenvectors,
            )

            condition_u_vectors.append(
                u_star
            )

            dose_numeric = condition[
                "dose_numeric"
            ]

            condition_metadata.append({
                "drug":
                    str(
                        condition[
                            "drug"
                        ]
                    ),
                "cell_line":
                    line_name,
                "dose":
                    str(
                        condition[
                            "dose"
                        ]
                    ),
                "dose_numeric":
                    (
                        float(
                            dose_numeric
                        )
                        if np.isfinite(
                            dose_numeric
                        )
                        else np.nan
                    ),
                "n_treated_cells":
                    int(
                        len(
                            treated_indices
                        )
                    ),
                "n_control_cells":
                    int(
                        len(
                            control_indices
                        )
                    ),
                "u_rms":
                    float(
                        np.sqrt(
                            np.mean(
                                u_star
                                * u_star
                            )
                        )
                    ),
                "response_rms":
                    float(
                        np.sqrt(
                            np.mean(
                                response
                                * response
                            )
                        )
                    ),
            })

    if not condition_u_vectors:
        raise RuntimeError(
            "No condition-level u* vectors were inferred."
        )

    condition_u_matrix = np.vstack(
        condition_u_vectors
    )

    condition_metadata = pd.DataFrame(
        condition_metadata
    )

    condition_output = pd.concat(
        [
            condition_metadata,
            pd.DataFrame(
                condition_u_matrix,
                columns=gene_labels,
            ),
        ],
        axis=1,
    )

    condition_output.to_csv(
        OUTDIR
        / "condition_level_u_star.csv.gz",
        index=False,
        compression="gzip",
    )

    # ============================================================
    # AGGREGATE TO EXACTLY ONE ROW PER DRUG
    # ============================================================

    if NORMALIZE_EACH_CONDITION_U_BY_RMS:
        condition_u_for_aggregation = (
            row_rms_normalize(
                condition_u_matrix
            )
        )
    else:
        condition_u_for_aggregation = (
            condition_u_matrix.copy()
        )

    drug_names = []
    drug_raw_vectors = []
    drug_pattern_vectors = []
    drug_count_rows = []

    for drug_name, group in condition_metadata.groupby(
        "drug",
        sort=True,
    ):
        condition_indices = (
            group.index
            .to_numpy(
                dtype=int
            )
        )

        if CONDITION_WEIGHTING == "equal":
            weights = np.ones(
                len(
                    condition_indices
                ),
                dtype=float,
            )

        elif CONDITION_WEIGHTING == "sqrt_cells":
            weights = np.sqrt(
                group[
                    "n_treated_cells"
                ].to_numpy(
                    dtype=float
                )
            )

        else:
            raise ValueError(
                "CONDITION_WEIGHTING must be "
                "'equal' or 'sqrt_cells'."
            )

        weights = (
            weights
            / weights.sum()
        )

        raw_mean_vector = np.sum(
            condition_u_matrix[
                condition_indices
            ]
            * weights[
                :,
                None,
            ],
            axis=0,
        )

        pattern_mean_vector = np.sum(
            condition_u_for_aggregation[
                condition_indices
            ]
            * weights[
                :,
                None,
            ],
            axis=0,
        )

        drug_names.append(
            drug_name
        )

        drug_raw_vectors.append(
            raw_mean_vector
        )

        drug_pattern_vectors.append(
            pattern_mean_vector
        )

        drug_count_rows.append({
            "drug":
                drug_name,
            "n_conditions_aggregated":
                int(
                    len(
                        condition_indices
                    )
                ),
            "cell_lines":
                ";".join(
                    sorted(
                        group[
                            "cell_line"
                        ].unique()
                    )
                ),
            "doses":
                ";".join(
                    sorted(
                        group[
                            "dose"
                        ].unique()
                    )
                ),
            "total_treated_cells":
                int(
                    group[
                        "n_treated_cells"
                    ].sum()
                ),
        })

    drug_raw_matrix = np.vstack(
        drug_raw_vectors
    )

    drug_pattern_matrix = np.vstack(
        drug_pattern_vectors
    )

    if ROW_ZSCORE_FOR_HEATMAP:
        heatmap_matrix = row_zscore(
            drug_pattern_matrix
        )
    else:
        heatmap_matrix = (
            drug_pattern_matrix.copy()
        )

    heatmap_matrix = np.clip(
        heatmap_matrix,
        -HEATMAP_CLIP,
        HEATMAP_CLIP,
    )

    def save_drug_gene_matrix(
        matrix,
        filename,
    ):
        table = pd.DataFrame(
            matrix,
            index=drug_names,
            columns=gene_labels,
        )

        table.index.name = "drug"

        table.to_csv(
            OUTDIR
            / filename
        )

    save_drug_gene_matrix(
        drug_raw_matrix,
        "drug_level_u_star_raw_mean.csv",
    )

    save_drug_gene_matrix(
        drug_pattern_matrix,
        "drug_level_u_star_pattern.csv",
    )

    save_drug_gene_matrix(
        heatmap_matrix,
        "drug_level_u_star_heatmap_z.csv",
    )

    pd.DataFrame(
        drug_count_rows
    ).to_csv(
        OUTDIR
        / "drug_condition_counts.csv",
        index=False,
    )

    # ============================================================
    # CLUSTER DRUGS AND GENES
    # ============================================================

    drug_order = hierarchical_cluster_order(
        heatmap_matrix,
        cluster_rows=True,
    )

    gene_order = hierarchical_cluster_order(
        heatmap_matrix,
        cluster_rows=False,
    )

    ordered_heatmap = heatmap_matrix[
        np.ix_(
            drug_order,
            gene_order,
        )
    ]

    ordered_drug_names = np.asarray(
        drug_names,
        dtype=object,
    )[
        drug_order
    ]

    ordered_gene_labels = gene_labels[
        gene_order
    ]

    ordered_ensembl_ids = (
        selected_ensembl_ids[
            gene_order
        ]
    )

    pd.DataFrame({
        "cluster_order":
            np.arange(
                len(
                    ordered_drug_names
                )
            ),
        "drug":
            ordered_drug_names,
    }).to_csv(
        OUTDIR
        / "clustered_drug_order.csv",
        index=False,
    )

    pd.DataFrame({
        "cluster_order":
            np.arange(
                len(
                    ordered_gene_labels
                )
            ),
        "gene":
            ordered_gene_labels,
        "ensembl_id":
            ordered_ensembl_ids,
    }).to_csv(
        OUTDIR
        / "clustered_gene_order.csv",
        index=False,
    )

    # ============================================================
    # DRAW CLUSTERED HEATMAP
    # ============================================================

    figure_height = min(
        max(
            14,
            0.18
            * len(
                ordered_drug_names
            )
            + 4,
        ),
        42,
    )

    figure, axis = plt.subplots(
        figsize=(
            24,
            figure_height,
        )
    )

    image = axis.imshow(
        ordered_heatmap,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-HEATMAP_CLIP,
        vmax=HEATMAP_CLIP,
    )

    axis.set_yticks(
        np.arange(
            len(
                ordered_drug_names
            )
        )
    )

    axis.set_yticklabels(
        ordered_drug_names,
        fontsize=5,
    )

    if (
        len(
            ordered_gene_labels
        )
        <= MAX_GENE_LABELS
    ):
        gene_tick_positions = np.arange(
            len(
                ordered_gene_labels
            )
        )
    else:
        gene_tick_positions = np.unique(
            np.linspace(
                0,
                len(
                    ordered_gene_labels
                )
                - 1,
                MAX_GENE_LABELS,
            ).astype(int)
        )

    axis.set_xticks(
        gene_tick_positions
    )

    axis.set_xticklabels(
        ordered_gene_labels[
            gene_tick_positions
        ],
        rotation=90,
        fontsize=5,
    )

    axis.set_xlabel(
        "Top 500 control HVGs"
    )

    axis.set_ylabel(
        "Drug"
    )

    axis.set_title(
        "Clustered dense CIPHER intervention vectors $u^*$\n"
        "one aggregated row per drug"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.02,
        pad=0.01,
    )

    colorbar.set_label(
        "Row-standardized aggregated $u^*$"
    )

    figure.tight_layout()

    figure.savefig(
        OUTDIR
        / "sciplex_drug_by_gene_u_star_clustered.png",
        dpi=USTAR_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTDIR
        / "sciplex_drug_by_gene_u_star_clustered.pdf",
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # ============================================================
    # SAVE RUN CONFIGURATION
    # ============================================================

    run_config = {
        "data_path":
            str(
                DATA_PATH
            ),
        "matrix_source":
            MATRIX_SOURCE,
        "ensembl_var_column":
            ENSEMBL_VAR_COLUMN,
        "perturbation_key":
            PERT_KEY,
        "cell_line_key":
            CELL_LINE_KEY,
        "dose_key":
            DOSE_KEY,
        "n_hvg":
            N_HVG,
        "dose_mode":
            DOSE_MODE,
        "cell_lines_used":
            control_cell_lines,
        "minimum_treated_cells":
            MIN_TREATED_CELLS,
        "minimum_control_cells":
            MIN_CONTROL_CELLS,
        "maximum_control_cells_for_covariance":
            MAX_CONTROL_CELLS_FOR_COV,
        "covariance_shrinkage":
            COV_SHRINK,
        "ridge_precision":
            RIDGE_PRECISION,
        "normalize_each_condition_u_by_rms":
            NORMALIZE_EACH_CONDITION_U_BY_RMS,
        "condition_weighting":
            CONDITION_WEIGHTING,
        "row_zscore_for_heatmap":
            ROW_ZSCORE_FOR_HEATMAP,
        "n_condition_vectors":
            int(
                condition_u_matrix.shape[0]
            ),
        "n_drugs":
            int(
                heatmap_matrix.shape[0]
            ),
        "n_genes":
            int(
                heatmap_matrix.shape[1]
            ),
    }

    (
        OUTDIR
        / "run_config.json"
    ).write_text(
        json.dumps(
            run_config,
            indent=2,
        )
    )

    print(
        "\nDONE"
    )

    print(
        "Output:",
        OUTDIR.resolve(),
    )

    print(
        "Condition-level u* matrix:",
        condition_u_matrix.shape,
    )

    print(
        "Drug-level heatmap matrix:",
        heatmap_matrix.shape,
    )

    print(
        "Clustered heatmap:",
        (
            OUTDIR
            / "sciplex_drug_by_gene_u_star_clustered.png"
        ).resolve(),
    )


# ============================================================
# (cell 28) global drug-class clustering permutation test
# ============================================================

DISTANCE_METRIC = "correlation"

LINKAGE_METHOD = "average"

N_PERMUTATIONS = 100_000

MIN_CATEGORY_SIZE_FOR_TEST = 3

EXCLUDE_CATEGORIES_FROM_TEST = {
    "Other / miscellaneous",
}

N_BLOCKS_TO_HIGHLIGHT = 5

MIN_BLOCK_SIZE_TO_HIGHLIGHT = 2

EXCLUDE_CATEGORIES_FROM_BLOCK_RANKING = {
    "Other / miscellaneous",
}

HEATMAP_VMIN = -3.0

HEATMAP_VMAX = 3.0

HEATMAP_CMAP = "RdBu_r"

SHOW_GENE_TICK_LABELS = False

SHOW_DRUG_TICK_LABELS = False

SHOW_HEATMAP_AXIS_LABELS = False

BLOCK_LINEWIDTH = 1.8

BLOCK_LABEL_FONTSIZE = 7.5

FIGSIZE = (
    19.0,
    12.5,
)

DPI = 400

N_HISTOGRAM_BINS = 60

SHOW_FIGURE = True

def normalize_drug_name(name):
    text = str(
        name
    ).strip().lower()

    text = (
        text.replace(
            "μ",
            "u",
        )
        .replace(
            "µ",
            "u",
        )
    )

    text = re.sub(
        r"\([^)]*\)",
        " ",
        text,
    )

    text = re.sub(
        r"[_/]+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9+\-. ]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text

def contains_any(
    text,
    patterns,
):
    return any(
        pattern in text
        for pattern in patterns
    )

HDAC_KEYWORDS = [
    "belinostat",
    "entinostat",
    "mocetinostat",
    "panobinostat",
    "quisinostat",
    "givinostat",
    "pracinostat",
    "tacedinaline",
    "tucidinostat",
    "abexinostat",
    "resminostat",
    "trichostatin a",
    "dacinostat",
    "ar-42",
    "m344",
    "tubastatin a",
    "droxinostat",
    "mc1568",
    "pci-34051",
    "itsa-1",
    "cudc-101",
    "cudc-907",
    "valproic acid",
    "divalproex",
    "sodium phenylbutyrate",
]

OTHER_EPIGENETIC_KEYWORDS = [
    "jq1",
    "pfi-1",
    "rg108",
    "azacitidine",
    "decitabine",
    "gsk-lsd1",
    "tranylcypromine",
    "tazemetostat",
    "eed226",
    "unc1999",
    "unc0379",
    "unc0631",
    "brd4770",
    "a-366",
    "gsk j1",
    "selisistat",
    "sirtinol",
    "srt2104",
    "srt3025",
    "srt1720",
    "tmp195",
]

JAK_STAT_KEYWORDS = [
    "fedratinib",
    "tg101209",
    "wp1066",
    "s3i-201",
    "momelotinib",
    "filgotinib",
    "tofacitinib",
    "baricitinib",
    "cep-33779",
    "cerdulatinib",
    "kw-2449",
    "ruxolitinib",
    "s-ruxolitinib",
    "bms-911543",
    "whi-p154",
    "az 960",
    "azd1480",
    "ag-490",
    "nvp-bsk805",
]

RTK_MULTIKINASE_KEYWORDS = [
    "motesanib",
    "linifanib",
    "pd173074",
    "cediranib",
    "nintedanib",
    "glesatinib",
    "vandetanib",
    "crizotinib",
    "enmd-2076",
    "pelitinib",
    "bosutinib",
    "dasatinib",
    "ac480",
    "lapatinib",
    "nilotinib",
    "regorafenib",
    "sorafenib",
    "ki8751",
    "tie2 kinase inhibitor",
    "gandotinib",
    "bms-754807",
    "bms-536924",
    "sgi-1776",
]

AURORA_CELL_CYCLE_KEYWORDS = [
    "amg-900",
    "barasertib",
    "zm 447439",
    "mk-5108",
    "sns-314",
    "at9283",
    "hesperadin",
    "tak-901",
    "tozasertib",
    "alisertib",
    "mln8054",
    "danusertib",
    "pha-680632",
    "gsk1070916",
    "bms-265246",
    "cyc116",
    "flavopiridol",
    "roscovitine",
    "jnj-7706621",
    "aurora a inhibitor i",
]

PI3K_MTOR_MAPK_KEYWORDS = [
    "tgx-221",
    "temsirolimus",
    "xav-939",
    "pd98059",
    "g007-lk",
    "sb431542",
    "ki16425",
    "pf-573228",
    "sl-327",
    "trametinib",
    "fasudil",
]

DNA_DAMAGE_KEYWORDS = [
    "iniparib",
    "pj34",
    "ino-1001",
    "rucaparib",
    "veliparib",
    "ag-14361",
    "thiotepa",
    "busulfan",
    "streptozotocin",
    "capecitabine",
    "mercaptopurine",
    "raltitrexed",
    "fluorouracil",
    "5-fu",
    "carmofur",
    "cyclocytidine",
    "lomustine",
    "altretamine",
    "pirarubicin",
    "clevudine",
]

HORMONE_RECEPTOR_KEYWORDS = [
    "fulvestrant",
    "toremifene",
    "andarine",
    "prednisone",
    "meprednisone",
    "triamcinolone",
    "aminoglutethimide",
    "2-methoxyestradiol",
]

APOPTOSIS_BCL2_KEYWORDS = [
    "abt-737",
    "navitoclax",
    "obatoclax",
]

CATEGORY_ORDER = [
    "HDAC inhibitor",
    "Other epigenetic",
    "JAK/STAT",
    "RTK / multikinase",
    "Aurora / cell-cycle",
    "PI3K/mTOR/MAPK",
    "DNA damage / antimet.",
    "Hormone / receptor",
    "Apoptosis / BCL2",
    "Other / miscellaneous",
]

CATEGORY_COLORS = {
    "HDAC inhibitor":
        "#D62728",
    "Other epigenetic":
        "#9467BD",
    "JAK/STAT":
        "#1F77B4",
    "RTK / multikinase":
        "#2CA02C",
    "Aurora / cell-cycle":
        "#FF7F0E",
    "PI3K/mTOR/MAPK":
        "#17BECF",
    "DNA damage / antimet.":
        "#8C564B",
    "Hormone / receptor":
        "#E377C2",
    "Apoptosis / BCL2":
        "#7F7F7F",
    "Other / miscellaneous":
        "#BCBD22",
}

def categorize_drug(
    drug_name,
):
    text = normalize_drug_name(
        drug_name
    )

    if contains_any(
        text,
        HDAC_KEYWORDS,
    ):
        return "HDAC inhibitor"

    if contains_any(
        text,
        OTHER_EPIGENETIC_KEYWORDS,
    ):
        return "Other epigenetic"

    if contains_any(
        text,
        JAK_STAT_KEYWORDS,
    ):
        return "JAK/STAT"

    if contains_any(
        text,
        RTK_MULTIKINASE_KEYWORDS,
    ):
        return "RTK / multikinase"

    if contains_any(
        text,
        AURORA_CELL_CYCLE_KEYWORDS,
    ):
        return "Aurora / cell-cycle"

    if contains_any(
        text,
        PI3K_MTOR_MAPK_KEYWORDS,
    ):
        return "PI3K/mTOR/MAPK"

    if contains_any(
        text,
        DNA_DAMAGE_KEYWORDS,
    ):
        return "DNA damage / antimet."

    if contains_any(
        text,
        HORMONE_RECEPTOR_KEYWORDS,
    ):
        return "Hormone / receptor"

    if contains_any(
        text,
        APOPTOSIS_BCL2_KEYWORDS,
    ):
        return "Apoptosis / BCL2"

    return "Other / miscellaneous"

def category_color(
    category,
):
    return CATEGORY_COLORS.get(
        category,
        "#333333",
    )

def benjamini_hochberg(
    p_values,
):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    n_values = len(
        p_values
    )

    if n_values == 0:
        return np.array(
            [],
            dtype=float,
        )

    order = np.argsort(
        p_values
    )

    ranked = p_values[
        order
    ]

    adjusted = (
        ranked
        * n_values
        / np.arange(
            1,
            n_values + 1,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.minimum(
        adjusted,
        1.0,
    )

    output = np.empty(
        n_values,
        dtype=float,
    )

    output[
        order
    ] = adjusted

    return output

def empirical_upper_tail_p(
    observed,
    null_values,
):
    null_values = np.asarray(
        null_values,
        dtype=float,
    )

    return (
        1
        + np.sum(
            null_values
            >= observed
        )
    ) / (
        len(
            null_values
        )
        + 1
    )

def safe_z_score(
    observed,
    null_values,
):
    null_values = np.asarray(
        null_values,
        dtype=float,
    )

    null_mean = float(
        np.mean(
            null_values
        )
    )

    null_sd = float(
        np.std(
            null_values,
            ddof=1,
        )
    )

    if (
        not np.isfinite(
            null_sd
        )
        or null_sd <= 0
    ):
        return np.nan

    return (
        observed
        - null_mean
    ) / null_sd

def format_probability(
    value,
):
    if not np.isfinite(
        value
    ):
        return "NA"

    if value < 0.001:
        return f"{value:.1e}"

    return f"{value:.3f}"

def hierarchical_order(
    matrix,
    cluster_rows=True,
):
    data = (
        matrix
        if cluster_rows
        else matrix.T
    )

    if data.shape[0] <= 1:
        return (
            np.arange(
                data.shape[0]
            ),
            None,
            np.array(
                [],
                dtype=float,
            ),
        )

    distances = pdist(
        data,
        metric=DISTANCE_METRIC,
    )

    distances = np.nan_to_num(
        distances,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    linkage_matrix = linkage(
        distances,
        method=LINKAGE_METHOD,
    )

    order = leaves_list(
        linkage_matrix
    )

    return (
        order,
        linkage_matrix,
        distances,
    )

def find_contiguous_blocks(
    categories,
):
    categories = list(
        map(
            str,
            categories,
        )
    )

    if len(
        categories
    ) == 0:
        return pd.DataFrame(
            columns=[
                "category",
                "start_row",
                "end_row",
                "n_rows",
            ]
        )

    blocks = []
    start = 0

    for i in range(
        1,
        len(categories) + 1,
    ):
        at_end = (
            i
            == len(categories)
        )

        changed = (
            not at_end
            and categories[i]
            != categories[start]
        )

        if at_end or changed:
            end = i - 1

            blocks.append({
                "category":
                    categories[start],
                "start_row":
                    int(start),
                "end_row":
                    int(end),
                "n_rows":
                    int(
                        end
                        - start
                        + 1
                    ),
            })

            start = i

    return pd.DataFrame(
        blocks
    )

def category_clustering_scores(
    cophenetic_distance_matrix,
    labels,
    categories,
):
    labels = np.asarray(
        labels,
        dtype=object,
    )

    category_results = {}

    for category in categories:
        inside = np.where(
            labels
            == category
        )[0]

        outside = np.where(
            labels
            != category
        )[0]

        if (
            len(inside) < 2
            or len(outside) == 0
        ):
            category_results[
                category
            ] = {
                "n_drugs":
                    int(
                        len(
                            inside
                        )
                    ),
                "within_distance":
                    np.nan,
                "between_distance":
                    np.nan,
                "cluster_score":
                    np.nan,
            }

            continue

        within_matrix = (
            cophenetic_distance_matrix[
                np.ix_(
                    inside,
                    inside,
                )
            ]
        )

        upper_indices = np.triu_indices(
            len(
                inside
            ),
            k=1,
        )

        within_values = (
            within_matrix[
                upper_indices
            ]
        )

        between_values = (
            cophenetic_distance_matrix[
                np.ix_(
                    inside,
                    outside,
                )
            ]
        ).ravel()

        within_distance = float(
            np.mean(
                within_values
            )
        )

        between_distance = float(
            np.mean(
                between_values
            )
        )

        category_results[
            category
        ] = {
            "n_drugs":
                int(
                    len(
                        inside
                    )
                ),
            "within_distance":
                within_distance,
            "between_distance":
                between_distance,
            "cluster_score":
                (
                    between_distance
                    - within_distance
                ),
        }

    finite_scores = [
        result[
            "cluster_score"
        ]
        for result
        in category_results.values()
        if np.isfinite(
            result[
                "cluster_score"
            ]
        )
    ]

    global_score = (
        float(
            np.mean(
                finite_scores
            )
        )
        if finite_scores
        else np.nan
    )

    return (
        global_score,
        category_results,
    )


# ============================================================
# (cell 30) per-cell-line three-panel u* clustering figures
# ============================================================

CELL_LINES = [
    "A549",
    "K562",
    "MCF7",
]

NORMALIZE_EACH_U_BY_RMS = True

SHOW_FIGURES = True

def choose_drug_vectors(
    cell_line_df,
    gene_columns,
):
    work = cell_line_df.copy()

    if (
        "dose_numeric"
        in work.columns
    ):
        work[
            "_dose_numeric"
        ] = pd.to_numeric(
            work[
                "dose_numeric"
            ],
            errors="coerce",
        )

    else:
        work[
            "_dose_numeric"
        ] = np.nan

    if (
        "dose"
        in work.columns
    ):
        parsed_doses = (
            work[
                "dose"
            ]
            .map(
                parse_dose
            )
        )

        work[
            "_dose_numeric"
        ] = work[
            "_dose_numeric"
        ].where(
            work[
                "_dose_numeric"
            ].notna(),
            parsed_doses,
        )

    selected_rows = []

    for drug_name, group in work.groupby(
        "drug",
        sort=True,
    ):
        if DOSE_MODE == "highest":
            if (
                group[
                    "_dose_numeric"
                ]
                .notna()
                .any()
            ):
                maximum_dose = group[
                    "_dose_numeric"
                ].max()

                selected_group = group[
                    group[
                        "_dose_numeric"
                    ]
                    == maximum_dose
                ]

            else:
                selected_group = group

            selected_row = (
                selected_group
                .iloc[0]
                .copy()
            )

            # Average duplicate vectors at the selected dose.
            selected_row.loc[
                gene_columns
            ] = (
                selected_group[
                    gene_columns
                ]
                .mean(
                    axis=0
                )
                .to_numpy()
            )

            selected_rows.append(
                selected_row
            )

        elif DOSE_MODE == "all_mean":
            vectors = group[
                gene_columns
            ].to_numpy(
                dtype=float
            )

            if NORMALIZE_EACH_U_BY_RMS:
                vectors = row_rms_normalize(
                    vectors
                )

            selected_row = (
                group
                .iloc[0]
                .copy()
            )

            selected_row.loc[
                gene_columns
            ] = vectors.mean(
                axis=0
            )

            selected_row[
                "dose"
            ] = "all_mean"

            selected_row[
                "dose_numeric"
            ] = np.nan

            selected_rows.append(
                selected_row
            )

        else:
            raise ValueError(
                "DOSE_MODE must be 'highest' or 'all_mean'."
            )

    return pd.DataFrame(
        selected_rows
    ).reset_index(
        drop=True
    )

def prepare_cell_line_matrix(
    cell_line_name,
    condition_df,
    gene_columns,
):
    cell_df = condition_df[
        condition_df[
            "cell_line"
        ].astype(str)
        == str(
            cell_line_name
        )
    ].copy()

    if cell_df.empty:
        raise RuntimeError(
            f"No condition rows found for {cell_line_name}."
        )

    selected_df = choose_drug_vectors(
        cell_df,
        gene_columns,
    )

    drug_names = (
        selected_df[
            "drug"
        ]
        .astype(str)
        .to_numpy(
            dtype=object,
        )
    )

    matrix = selected_df[
        gene_columns
    ].to_numpy(
        dtype=float,
    )

    # ------------------------------------------------------------
    # Remove invalid or constant drugs
    # ------------------------------------------------------------

    valid_drugs = (
        np.all(
            np.isfinite(
                matrix
            ),
            axis=1,
        )
        & (
            np.std(
                matrix,
                axis=1,
            )
            > 1e-12
        )
    )

    matrix = matrix[
        valid_drugs
    ]

    drug_names = drug_names[
        valid_drugs
    ]

    # ------------------------------------------------------------
    # Normalize each condition-level u*
    # ------------------------------------------------------------

    if (
        NORMALIZE_EACH_U_BY_RMS
        and DOSE_MODE
        == "highest"
    ):
        matrix = row_rms_normalize(
            matrix
        )

    if ROW_ZSCORE_FOR_HEATMAP:
        matrix = row_zscore(
            matrix
        )

    matrix = np.clip(
        matrix,
        -HEATMAP_CLIP,
        HEATMAP_CLIP,
    )

    # ------------------------------------------------------------
    # Remove genes constant across drugs in this cell line
    # ------------------------------------------------------------

    valid_genes = (
        np.all(
            np.isfinite(
                matrix
            ),
            axis=0,
        )
        & (
            np.std(
                matrix,
                axis=0,
            )
            > 1e-12
        )
    )

    matrix = matrix[
        :,
        valid_genes,
    ]

    gene_names = np.asarray(
        gene_columns,
        dtype=object,
    )[
        valid_genes
    ]

    categories = np.array(
        [
            categorize_drug(
                drug
            )
            for drug
            in drug_names
        ],
        dtype=object,
    )

    return {
        "cell_line":
            cell_line_name,
        "matrix_unordered":
            matrix,
        "drug_names_unordered":
            drug_names,
        "gene_names_unordered":
            gene_names,
        "categories_unordered":
            categories,
    }

def analyze_cell_line(
    prepared,
):
    cell_line_name = prepared[
        "cell_line"
    ]

    matrix_unordered = prepared[
        "matrix_unordered"
    ]

    drug_names_unordered = prepared[
        "drug_names_unordered"
    ]

    gene_names_unordered = prepared[
        "gene_names_unordered"
    ]

    categories_unordered = prepared[
        "categories_unordered"
    ]

    cell_outdir = (
        CELL_LINE_OUTDIR
        / cell_line_name
    )

    cell_outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # CLUSTER BOTH AXES
    # ============================================================

    (
        drug_order,
        drug_linkage,
        drug_distances,
    ) = hierarchical_order(
        matrix_unordered,
        cluster_rows=True,
    )

    (
        gene_order,
        gene_linkage,
        gene_distances,
    ) = hierarchical_order(
        matrix_unordered,
        cluster_rows=False,
    )

    matrix_clustered = matrix_unordered[
        np.ix_(
            drug_order,
            gene_order,
        )
    ]

    drug_names_clustered = (
        drug_names_unordered[
            drug_order
        ]
    )

    gene_names_clustered = (
        gene_names_unordered[
            gene_order
        ]
    )

    categories_clustered = (
        categories_unordered[
            drug_order
        ]
    )

    # ============================================================
    # CONTIGUOUS CATEGORY BLOCKS
    # ============================================================

    blocks_df = find_contiguous_blocks(
        categories_clustered
    )

    blocks_df[
        "category_color"
    ] = blocks_df[
        "category"
    ].map(
        category_color
    )

    blocks_df[
        "eligible_for_highlight"
    ] = (
        (
            blocks_df[
                "n_rows"
            ]
            >= MIN_BLOCK_SIZE_TO_HIGHLIGHT
        )
        & (
            ~blocks_df[
                "category"
            ].isin(
                EXCLUDE_CATEGORIES_FROM_BLOCK_RANKING
            )
        )
    )

    eligible_blocks = (
        blocks_df[
            blocks_df[
                "eligible_for_highlight"
            ]
        ]
        .copy()
        .sort_values(
            [
                "n_rows",
                "start_row",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index()
        .rename(
            columns={
                "index":
                    "original_block_index",
            }
        )
    )

    eligible_blocks[
        "highlight_rank"
    ] = np.arange(
        1,
        len(
            eligible_blocks
        ) + 1,
    )

    top_blocks_df = (
        eligible_blocks
        .head(
            N_BLOCKS_TO_HIGHLIGHT
        )
        .copy()
    )

    # ============================================================
    # SELECT TESTABLE CATEGORIES
    # ============================================================

    category_counts = (
        pd.Series(
            categories_unordered
        )
        .value_counts()
    )

    categories_to_test = [
        category
        for category, count
        in category_counts.items()
        if (
            count
            >= MIN_CATEGORY_SIZE_FOR_TEST
            and category
            not in EXCLUDE_CATEGORIES_FROM_TEST
        )
    ]

    if len(
        categories_to_test
    ) < 2:
        raise RuntimeError(
            f"{cell_line_name}: fewer than two categories "
            "passed the permutation-test filters.\n"
            f"{category_counts}"
        )

    test_mask = np.isin(
        categories_unordered,
        categories_to_test,
    )

    test_matrix = (
        matrix_unordered[
            test_mask
        ]
    )

    test_labels = (
        categories_unordered[
            test_mask
        ]
    )

    test_drug_names = (
        drug_names_unordered[
            test_mask
        ]
    )

    # Correlation distance is unaffected by row scaling, but this
    # makes the tested representation explicit.
    test_matrix_centered = (
        test_matrix
        - test_matrix.mean(
            axis=1,
            keepdims=True,
        )
    )

    test_matrix_scaled = (
        test_matrix_centered
        / np.maximum(
            test_matrix_centered.std(
                axis=1,
                keepdims=True,
            ),
            1e-12,
        )
    )

    test_condensed_distances = pdist(
        test_matrix_scaled,
        metric=DISTANCE_METRIC,
    )

    test_condensed_distances = np.nan_to_num(
        test_condensed_distances,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    test_linkage = linkage(
        test_condensed_distances,
        method=LINKAGE_METHOD,
    )

    (
        cophenetic_correlation,
        cophenetic_condensed,
    ) = cophenet(
        test_linkage,
        test_condensed_distances,
    )

    cophenetic_distance_matrix = squareform(
        cophenetic_condensed
    )

    # ============================================================
    # OBSERVED SCORES
    # ============================================================

    (
        observed_global_score,
        observed_category_results,
    ) = category_clustering_scores(
        cophenetic_distance_matrix=
            cophenetic_distance_matrix,
        labels=
            test_labels,
        categories=
            categories_to_test,
    )

    # ============================================================
    # CELL-LINE-SPECIFIC PERMUTATION NULL
    # ============================================================

    cell_seed = (
        SEED
        + sum(
            ord(character)
            for character
            in cell_line_name
        )
    )

    rng = np.random.default_rng(
        cell_seed
    )

    null_global_scores = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    null_category_scores = {
        category:
            np.empty(
                N_PERMUTATIONS,
                dtype=float,
            )
        for category
        in categories_to_test
    }

    for permutation_index in tqdm(
        range(
            N_PERMUTATIONS
        ),
        desc=(
            f"{cell_line_name}: "
            "permuting drug-class labels"
        ),
    ):
        permuted_labels = rng.permutation(
            test_labels
        )

        (
            permuted_global_score,
            permuted_category_results,
        ) = category_clustering_scores(
            cophenetic_distance_matrix=
                cophenetic_distance_matrix,
            labels=
                permuted_labels,
            categories=
                categories_to_test,
        )

        null_global_scores[
            permutation_index
        ] = permuted_global_score

        for category in categories_to_test:
            null_category_scores[
                category
            ][
                permutation_index
            ] = (
                permuted_category_results[
                    category
                ][
                    "cluster_score"
                ]
            )

    # ============================================================
    # GLOBAL STATISTICS
    # ============================================================

    global_p_value = empirical_upper_tail_p(
        observed_global_score,
        null_global_scores,
    )

    global_z_score = safe_z_score(
        observed_global_score,
        null_global_scores,
    )

    global_statistics_df = pd.DataFrame([
        {
            "cell_line":
                cell_line_name,
            "n_drugs":
                len(
                    test_drug_names
                ),
            "n_categories":
                len(
                    categories_to_test
                ),
            "n_genes":
                matrix_unordered.shape[1],
            "distance_metric":
                DISTANCE_METRIC,
            "linkage_method":
                LINKAGE_METHOD,
            "cophenetic_correlation":
                float(
                    cophenetic_correlation
                ),
            "observed_global_cluster_score":
                observed_global_score,
            "null_mean_global_cluster_score":
                float(
                    np.mean(
                        null_global_scores
                    )
                ),
            "null_sd_global_cluster_score":
                float(
                    np.std(
                        null_global_scores,
                        ddof=1,
                    )
                ),
            "global_z_score":
                global_z_score,
            "global_empirical_p_value":
                global_p_value,
            "n_permutations":
                N_PERMUTATIONS,
            "dose_mode":
                DOSE_MODE,
        }
    ])

    # ============================================================
    # CATEGORY-SPECIFIC STATISTICS
    # ============================================================

    category_rows = []

    for category in categories_to_test:
        observed = (
            observed_category_results[
                category
            ]
        )

        null_values = (
            null_category_scores[
                category
            ]
        )

        category_rows.append({
            "cell_line":
                cell_line_name,
            "category":
                category,
            "n_drugs":
                observed[
                    "n_drugs"
                ],
            "observed_within_cophenetic_distance":
                observed[
                    "within_distance"
                ],
            "observed_between_cophenetic_distance":
                observed[
                    "between_distance"
                ],
            "observed_cluster_score":
                observed[
                    "cluster_score"
                ],
            "null_mean_cluster_score":
                float(
                    np.mean(
                        null_values
                    )
                ),
            "null_sd_cluster_score":
                float(
                    np.std(
                        null_values,
                        ddof=1,
                    )
                ),
            "z_score":
                safe_z_score(
                    observed[
                        "cluster_score"
                    ],
                    null_values,
                ),
            "empirical_p_value":
                empirical_upper_tail_p(
                    observed[
                        "cluster_score"
                    ],
                    null_values,
                ),
        })

    category_statistics_df = pd.DataFrame(
        category_rows
    )

    category_statistics_df[
        "fdr_q_value"
    ] = benjamini_hochberg(
        category_statistics_df[
            "empirical_p_value"
        ].to_numpy()
    )

    category_statistics_df = (
        category_statistics_df
        .sort_values(
            "z_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ============================================================
    # SAVE MATRICES AND STATISTICS
    # ============================================================

    pd.DataFrame(
        matrix_clustered,
        index=
            drug_names_clustered,
        columns=
            gene_names_clustered,
    ).rename_axis(
        "drug"
    ).to_csv(
        cell_outdir
        / (
            f"{cell_line_name}_"
            "drug_by_gene_u_star_heatmap_z.csv"
        )
    )

    pd.DataFrame({
        "cluster_order":
            np.arange(
                len(
                    drug_names_clustered
                )
            ),
        "drug":
            drug_names_clustered,
        "category":
            categories_clustered,
        "category_color": [
            category_color(
                category
            )
            for category
            in categories_clustered
        ],
    }).to_csv(
        cell_outdir
        / (
            f"{cell_line_name}_"
            "clustered_drug_order.csv"
        ),
        index=False,
    )

    pd.DataFrame({
        "cluster_order":
            np.arange(
                len(
                    gene_names_clustered
                )
            ),
        "gene":
            gene_names_clustered,
    }).to_csv(
        cell_outdir
        / (
            f"{cell_line_name}_"
            "clustered_gene_order.csv"
        ),
        index=False,
    )

    blocks_df.to_csv(
        cell_outdir
        / (
            f"{cell_line_name}_"
            "contiguous_category_blocks.csv"
        ),
        index=False,
    )

    top_blocks_df.to_csv(
        cell_outdir
        / (
            f"{cell_line_name}_"
            "top_5_category_blocks.csv"
        ),
        index=False,
    )

    global_statistics_df.to_csv(
        cell_outdir
        / (
            f"{cell_line_name}_"
            "global_clustering_statistics.csv"
        ),
        index=False,
    )

    category_statistics_df.to_csv(
        cell_outdir
        / (
            f"{cell_line_name}_"
            "category_clustering_statistics.csv"
        ),
        index=False,
    )

    null_payload = {
        "global_null":
            null_global_scores,
        "observed_global":
            np.array([
                observed_global_score
            ]),
        "categories":
            np.array(
                categories_to_test,
                dtype=object,
            ),
    }

    for category in categories_to_test:
        safe_category = re.sub(
            r"[^a-z0-9]+",
            "_",
            category.lower(),
        ).strip(
            "_"
        )

        null_payload[
            f"null_{safe_category}"
        ] = null_category_scores[
            category
        ]

        null_payload[
            f"observed_{safe_category}"
        ] = np.array([
            observed_category_results[
                category
            ][
                "cluster_score"
            ]
        ])

    np.savez_compressed(
        cell_outdir
        / (
            f"{cell_line_name}_"
            "null_distributions.npz"
        ),
        **null_payload,
    )

    return {
        "cell_line":
            cell_line_name,
        "cell_outdir":
            cell_outdir,
        "matrix_clustered":
            matrix_clustered,
        "drug_names_clustered":
            drug_names_clustered,
        "gene_names_clustered":
            gene_names_clustered,
        "categories_clustered":
            categories_clustered,
        "blocks_df":
            blocks_df,
        "top_blocks_df":
            top_blocks_df,
        "categories_to_test":
            categories_to_test,
        "observed_global_score":
            observed_global_score,
        "null_global_scores":
            null_global_scores,
        "global_p_value":
            global_p_value,
        "global_z_score":
            global_z_score,
        "cophenetic_correlation":
            cophenetic_correlation,
        "category_statistics_df":
            category_statistics_df,
    }

def make_three_panel_figure(
    result,
):
    cell_line_name = result[
        "cell_line"
    ]

    cell_outdir = result[
        "cell_outdir"
    ]

    matrix_clustered = result[
        "matrix_clustered"
    ]

    drug_names_clustered = result[
        "drug_names_clustered"
    ]

    gene_names_clustered = result[
        "gene_names_clustered"
    ]

    categories_clustered = result[
        "categories_clustered"
    ]

    blocks_df = result[
        "blocks_df"
    ]

    top_blocks_df = result[
        "top_blocks_df"
    ]

    observed_global_score = result[
        "observed_global_score"
    ]

    null_global_scores = result[
        "null_global_scores"
    ]

    global_p_value = result[
        "global_p_value"
    ]

    global_z_score = result[
        "global_z_score"
    ]

    category_statistics_df = result[
        "category_statistics_df"
    ]

    # ============================================================
    # CATEGORY STRIP
    # ============================================================

    categories_present = [
        category
        for category
        in CATEGORY_ORDER
        if category
        in set(
            categories_clustered
        )
    ]

    category_code_lookup = {
        category:
            index
        for index, category
        in enumerate(
            categories_present
        )
    }

    category_codes = np.array(
        [
            category_code_lookup[
                category
            ]
            for category
            in categories_clustered
        ],
        dtype=int,
    ).reshape(
        -1,
        1,
    )

    category_cmap = ListedColormap([
        category_color(
            category
        )
        for category
        in categories_present
    ])

    # ============================================================
    # FIGURE LAYOUT
    # ============================================================

    figure = plt.figure(
        figsize=FIGSIZE,
        constrained_layout=False,
    )

    outer_grid = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        figure=figure,
        width_ratios=[
            1.72,
            1.0,
        ],
        height_ratios=[
            1.0,
            1.0,
        ],
        left=0.045,
        right=0.975,
        bottom=0.065,
        top=0.865,
        wspace=0.28,
        hspace=0.30,
    )

    heatmap_grid = (
        gridspec
        .GridSpecFromSubplotSpec(
            nrows=1,
            ncols=4,
            subplot_spec=outer_grid[
                :,
                0,
            ],
            width_ratios=[
                1.25,
                0.14,
                8.0,
                0.18,
            ],
            wspace=0.025,
        )
    )

    block_label_axis = figure.add_subplot(
        heatmap_grid[
            0,
            0,
        ]
    )

    category_strip_axis = figure.add_subplot(
        heatmap_grid[
            0,
            1,
        ]
    )

    heatmap_axis = figure.add_subplot(
        heatmap_grid[
            0,
            2,
        ]
    )

    heatmap_colorbar_axis = figure.add_subplot(
        heatmap_grid[
            0,
            3,
        ]
    )

    global_null_axis = figure.add_subplot(
        outer_grid[
            0,
            1,
        ]
    )

    category_score_axis = figure.add_subplot(
        outer_grid[
            1,
            1,
        ]
    )

    # ============================================================
    # PANEL A: CELL-LINE-SPECIFIC HEATMAP
    # ============================================================

    heatmap_image = heatmap_axis.imshow(
        matrix_clustered,
        aspect="auto",
        interpolation="nearest",
        cmap=HEATMAP_CMAP,
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        rasterized=True,
    )

    if SHOW_DRUG_TICK_LABELS:
        heatmap_axis.set_yticks(
            np.arange(
                len(
                    drug_names_clustered
                )
            )
        )

        heatmap_axis.set_yticklabels(
            drug_names_clustered,
            fontsize=2.7,
        )

    else:
        heatmap_axis.set_yticks(
            []
        )

    if SHOW_GENE_TICK_LABELS:
        heatmap_axis.set_xticks(
            np.arange(
                len(
                    gene_names_clustered
                )
            )
        )

        heatmap_axis.set_xticklabels(
            gene_names_clustered,
            rotation=90,
            fontsize=2.5,
        )

    else:
        heatmap_axis.set_xticks(
            []
        )

    heatmap_axis.set_xlabel(
        ""
    )

    heatmap_axis.set_ylabel(
        ""
    )

    heatmap_axis.set_title(
        (
            "Inferred dense intervention programs\n"
            f"{cell_line_name}"
        ),
        pad=8,
    )

    # ------------------------------------------------------------
    # Category strip
    # ------------------------------------------------------------

    category_strip_axis.imshow(
        category_codes,
        aspect="auto",
        interpolation="nearest",
        cmap=category_cmap,
        vmin=-0.5,
        vmax=(
            len(
                categories_present
            )
            - 0.5
        ),
    )

    category_strip_axis.set_xticks(
        []
    )

    category_strip_axis.set_yticks(
        []
    )

    category_strip_axis.set_title(
        "Class",
        fontsize=8,
        pad=8,
    )

    # ------------------------------------------------------------
    # Block label axis
    # ------------------------------------------------------------

    block_label_axis.set_xlim(
        0,
        1,
    )

    block_label_axis.set_ylim(
        len(
            drug_names_clustered
        ) - 0.5,
        -0.5,
    )

    block_label_axis.axis(
        "off"
    )

    # ------------------------------------------------------------
    # Subtle separators
    # ------------------------------------------------------------

    for block in blocks_df.itertuples(
        index=False
    ):
        if (
            block.end_row
            < len(
                drug_names_clustered
            ) - 1
        ):
            boundary = (
                block.end_row
                + 0.5
            )

            heatmap_axis.axhline(
                boundary,
                color="black",
                linewidth=0.20,
                alpha=0.22,
                zorder=5,
            )

            category_strip_axis.axhline(
                boundary,
                color="black",
                linewidth=0.20,
                alpha=0.22,
                zorder=5,
            )

    # ------------------------------------------------------------
    # Top five contiguous blocks
    # ------------------------------------------------------------

    for block in top_blocks_df.itertuples(
        index=False
    ):
        rank = int(
            block.highlight_rank
        )

        block_color = category_color(
            block.category
        )

        y_start = (
            block.start_row
            - 0.5
        )

        block_height = (
            block.n_rows
        )

        category_strip_axis.add_patch(
            Rectangle(
                (
                    -0.5,
                    y_start,
                ),
                width=1.0,
                height=block_height,
                fill=False,
                edgecolor=block_color,
                linewidth=BLOCK_LINEWIDTH,
                zorder=30,
                clip_on=False,
            )
        )

        heatmap_axis.add_patch(
            Rectangle(
                (
                    -0.5,
                    y_start,
                ),
                width=
                    matrix_clustered.shape[1],
                height=
                    block_height,
                fill=False,
                edgecolor=
                    block_color,
                linewidth=
                    BLOCK_LINEWIDTH,
                zorder=30,
                clip_on=False,
            )
        )

        block_label_axis.text(
            0.98,
            (
                y_start
                + block_height / 2
            ),
            (
                f"#{rank} "
                f"{block.category}\n"
                f"n={int(block.n_rows)}"
            ),
            ha="right",
            va="center",
            fontsize=
                BLOCK_LABEL_FONTSIZE,
            fontweight="bold",
            color=
                block_color,
            bbox={
                "facecolor":
                    "white",
                "edgecolor":
                    block_color,
                "linewidth":
                    0.8,
                "alpha":
                    0.95,
                "boxstyle":
                    "round,pad=0.22",
            },
            zorder=40,
        )

    # ------------------------------------------------------------
    # Heatmap colorbar
    # ------------------------------------------------------------

    heatmap_colorbar = figure.colorbar(
        heatmap_image,
        cax=
            heatmap_colorbar_axis,
    )

    heatmap_colorbar.set_label(
        "Row-standardized $u^*$",
        fontsize=8,
    )

    heatmap_colorbar.ax.tick_params(
        labelsize=7,
    )

    block_label_axis.text(
        -0.15,
        1.02,
        "A",
        transform=
            block_label_axis.transAxes,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # ============================================================
    # PANEL B: CELL-LINE-SPECIFIC GLOBAL NULL
    # ============================================================

    global_null_axis.hist(
        null_global_scores,
        bins=N_HISTOGRAM_BINS,
        density=True,
        color="#9ECAE1",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.95,
    )

    global_null_axis.axvline(
        observed_global_score,
        color="#D62728",
        linewidth=2.6,
        label="Observed",
        zorder=10,
    )

    null_mean = float(
        np.mean(
            null_global_scores
        )
    )

    global_null_axis.axvline(
        null_mean,
        color="#4D4D4D",
        linewidth=1.8,
        linestyle="--",
        label="Null mean",
        zorder=10,
    )

    global_null_axis.set_title(
        (
            "Global drug-class clustering\n"
            f"{cell_line_name}"
        )
    )

    global_null_axis.set_xlabel(
        "Mean category cophenetic-separation score"
    )

    global_null_axis.set_ylabel(
        "Permutation density"
    )

    global_null_axis.legend(
        frameon=False,
        loc="upper right",
    )

    global_null_axis.text(
        0.035,
        0.965,
        (
            f"Observed = "
            f"{observed_global_score:.4f}\n"
            f"Null mean = "
            f"{null_mean:.4f}\n"
            f"$z$ = "
            f"{global_z_score:.2f}\n"
            f"Empirical $P$ = "
            f"{format_probability(global_p_value)}"
        ),
        transform=
            global_null_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={
            "facecolor":
                "white",
            "edgecolor":
                "0.7",
            "linewidth":
                0.7,
            "alpha":
                0.94,
            "boxstyle":
                "round,pad=0.30",
        },
    )

    global_null_axis.spines[
        "top"
    ].set_visible(
        False
    )

    global_null_axis.spines[
        "right"
    ].set_visible(
        False
    )

    global_null_axis.text(
        -0.14,
        1.04,
        "B",
        transform=
            global_null_axis.transAxes,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # ============================================================
    # PANEL C: CELL-LINE-SPECIFIC CATEGORY EFFECTS
    # ============================================================

    category_plot_df = (
        category_statistics_df
        .sort_values(
            "z_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    bar_positions = np.arange(
        len(
            category_plot_df
        )
    )

    bar_colors = [
        category_color(
            category
        )
        for category
        in category_plot_df[
            "category"
        ]
    ]

    category_score_axis.barh(
        bar_positions,
        category_plot_df[
            "z_score"
        ],
        color=bar_colors,
        edgecolor="none",
        alpha=0.90,
    )

    category_score_axis.set_yticks(
        bar_positions
    )

    category_score_axis.set_yticklabels(
        category_plot_df[
            "category"
        ]
    )

    category_score_axis.invert_yaxis()

    category_score_axis.axvline(
        0,
        color="black",
        linewidth=0.8,
    )

    category_score_axis.set_xlabel(
        "Permutation $z$-score"
    )

    category_score_axis.set_title(
        (
            "Category-specific clustering\n"
            f"{cell_line_name}"
        )
    )

    maximum_z = float(
        np.nanmax(
            category_plot_df[
                "z_score"
            ]
        )
    )

    minimum_z = float(
        np.nanmin(
            category_plot_df[
                "z_score"
            ]
        )
    )

    z_span = max(
        maximum_z
        - minimum_z,
        1.0,
    )

    label_offset = (
        0.025
        * z_span
    )

    for position, statistic in enumerate(
        category_plot_df.itertuples(
            index=False
        )
    ):
        category_score_axis.text(
            statistic.z_score
            + (
                label_offset
                if statistic.z_score >= 0
                else -label_offset
            ),
            position,
            (
                f"$q$="
                f"{format_probability(statistic.fdr_q_value)}"
            ),
            va="center",
            ha=(
                "left"
                if statistic.z_score >= 0
                else "right"
            ),
            fontsize=7.5,
            fontweight=(
                "bold"
                if statistic.fdr_q_value < 0.05
                else "normal"
            ),
        )

    category_score_axis.set_xlim(
        min(
            -0.5,
            minimum_z
            - 0.10 * z_span,
        ),
        maximum_z
        + 0.25 * z_span,
    )

    category_score_axis.spines[
        "top"
    ].set_visible(
        False
    )

    category_score_axis.spines[
        "right"
    ].set_visible(
        False
    )

    category_score_axis.text(
        -0.14,
        1.04,
        "C",
        transform=
            category_score_axis.transAxes,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # ============================================================
    # CENTERED CATEGORY LEGEND
    # ============================================================

    legend_handles = [
        Patch(
            facecolor=category_color(
                category
            ),
            edgecolor="none",
            label=category,
        )
        for category
        in categories_present
    ]

    figure.legend(
        handles=
            legend_handles,
        loc=
            "upper center",
        bbox_to_anchor=(
            0.5,
            0.985,
        ),
        bbox_transform=
            figure.transFigure,
        ncol=5,
        frameon=False,
        fontsize=8,
        handlelength=1.1,
        handleheight=0.9,
        columnspacing=1.6,
        labelspacing=0.7,
        borderaxespad=0.0,
    )

    dose_description = (
        "highest dose per drug"
        if DOSE_MODE
        == "highest"
        else "all doses averaged within drug"
    )

    figure.suptitle(
        (
            "SciPlex dense CIPHER intervention-program "
            f"clustering: {cell_line_name}"
        ),
        fontsize=15,
        fontweight="bold",
        y=0.925,
    )

    figure.text(
        0.73,
        0.025,
        (
            "Null generated by permuting drug-class labels "
            "while preserving category sizes; "
            f"{dose_description}."
        ),
        ha="center",
        va="center",
        fontsize=7.5,
        color="0.35",
    )

    # ============================================================
    # SAVE AND DISPLAY
    # ============================================================

    png_path = (
        cell_outdir
        / (
            f"{cell_line_name}_"
            "three_panel_drug_u_star.png"
        )
    )

    pdf_path = (
        cell_outdir
        / (
            f"{cell_line_name}_"
            "three_panel_drug_u_star.pdf"
        )
    )

    svg_path = (
        cell_outdir
        / (
            f"{cell_line_name}_"
            "three_panel_drug_u_star.svg"
        )
    )

    figure.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    print(
        f"\nSaved {cell_line_name} figure:"
    )

    print(
        " ",
        png_path.resolve(),
    )

    print(
        " ",
        pdf_path.resolve(),
    )

    print(
        " ",
        svg_path.resolve(),
    )

    if SHOW_FIGURES:
        plt.show()

    else:
        plt.close(
            figure
        )

def run_cell_line_specific():
    CELL_LINE_OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CONDITION_CSV.exists():
        raise FileNotFoundError(
            "Missing condition-level u* file:\n"
            f"{CONDITION_CSV.resolve()}\n\n"
            "Run the dense u* inference script first."
        )

    print(
        "Loading:",
        CONDITION_CSV.resolve(),
    )

    condition_df = pd.read_csv(
        CONDITION_CSV
    )

    required_metadata = {
        "drug",
        "cell_line",
    }

    missing_metadata = (
        required_metadata
        - set(
            condition_df.columns
        )
    )

    if missing_metadata:
        raise ValueError(
            "Missing required metadata columns: "
            f"{sorted(missing_metadata)}"
        )

    metadata_columns = {
        "drug",
        "cell_line",
        "dose",
        "dose_numeric",
        "replicate",
        "n_treated_cells",
        "n_control_cells",
        "u_rms",
        "response_rms",
    }

    gene_columns = [
        column
        for column
        in condition_df.columns
        if column
        not in metadata_columns
    ]

    if not gene_columns:
        raise RuntimeError(
            "No gene columns were found in "
            "condition_level_u_star.csv.gz."
        )

    condition_df[
        gene_columns
    ] = condition_df[
        gene_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    print(
        "Condition rows:",
        len(
            condition_df
        ),
    )

    print(
        "Gene columns:",
        len(
            gene_columns
        ),
    )

    all_global_results = []
    all_category_results = []

    for cell_line_name in CELL_LINES:
        print(
            "\n" + "=" * 78
        )

        print(
            "CELL LINE:",
            cell_line_name
        )

        print(
            "=" * 78
        )

        prepared = prepare_cell_line_matrix(
            cell_line_name=
                cell_line_name,
            condition_df=
                condition_df,
            gene_columns=
                gene_columns,
        )

        result = analyze_cell_line(
            prepared
        )

        make_three_panel_figure(
            result
        )

        global_csv = (
            result[
                "cell_outdir"
            ]
            / (
                f"{cell_line_name}_"
                "global_clustering_statistics.csv"
            )
        )

        category_csv = (
            result[
                "cell_outdir"
            ]
            / (
                f"{cell_line_name}_"
                "category_clustering_statistics.csv"
            )
        )

        all_global_results.append(
            pd.read_csv(
                global_csv
            )
        )

        all_category_results.append(
            pd.read_csv(
                category_csv
            )
        )

        print(
            f"\n{cell_line_name} summary:"
        )

        print(
            f"  Drugs in heatmap: "
            f"{result['matrix_clustered'].shape[0]}"
        )

        print(
            f"  Genes in heatmap: "
            f"{result['matrix_clustered'].shape[1]}"
        )

        print(
            f"  Global score: "
            f"{result['observed_global_score']:.6f}"
        )

        print(
            f"  Global z-score: "
            f"{result['global_z_score']:.3f}"
        )

        print(
            f"  Global empirical P: "
            f"{result['global_p_value']:.6g}"
        )

        print(
            "\n  Category results:"
        )

        print(
            result[
                "category_statistics_df"
            ][
                [
                    "category",
                    "n_drugs",
                    "z_score",
                    "empirical_p_value",
                    "fdr_q_value",
                ]
            ].to_string(
                index=False
            )
        )

    # ============================================================
    # COMBINED SUMMARY TABLES ACROSS CELL LINES
    # ============================================================

    combined_global_df = pd.concat(
        all_global_results,
        ignore_index=True,
    )

    combined_category_df = pd.concat(
        all_category_results,
        ignore_index=True,
    )

    combined_global_df.to_csv(
        CELL_LINE_OUTDIR
        / "all_cell_lines_global_clustering_statistics.csv",
        index=False,
    )

    combined_category_df.to_csv(
        CELL_LINE_OUTDIR
        / "all_cell_lines_category_clustering_statistics.csv",
        index=False,
    )

    print(
        "\n" + "=" * 78
    )

    print(
        "ALL CELL-LINE-SPECIFIC ANALYSES COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "\nCombined global statistics:"
    )

    print(
        combined_global_df[
            [
                "cell_line",
                "n_drugs",
                "observed_global_cluster_score",
                "global_z_score",
                "global_empirical_p_value",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nOutput directory:"
    )

    print(
        CELL_LINE_OUTDIR.resolve()
    )


__all__ = [
    'DATA_DIR',
    'OUTBASE',
    'DATA_PATH',
    'OUTDIR',
    'INDIR',
    'MATRIX_CSV',
    'CONDITION_CSV',
    'CELL_LINE_OUTDIR',
    'USTAR_DPI',
    '_is_bad_label',
    '_split_train_test_indices',
    '_mean_rows_sparse',
    '_symmetrize',
    '_shrink_cov',
    '_eig_psd',
    '_diag_VtSV',
    '_ll_diag_gauss',
    '_estimate_u_ridge_in_Vbasis',
    '_cosine_sim',
    '_topk_by_freq',
    '_confusion_topk_arrays',
    '_plot_confusion_topk',
    '_plot_rank_hist',
    '_log1p_mean_from_mean',
    '_zscore_fit_transform',
    '_zscore_transform',
    '_roc_pr_from_scores',
    '_shuffle_columns_independently',
    '_topk_accuracy_from_orders',
    '_topk_acc_from_rank',
    '_pad_2d',
    '_run_lr',
    'FONTSIZE',
    'CORNFLOWER',
    'PURPLE',
    'SALMON',
    'COLORS',
    'LINESTYLE',
    '_pick_dataset_name',
    '_safe_arr',
    '_plot_example',
    '_plot_topk',
    '_plot_roc',
    '_plot_pr',
    '_choose_best_example_by_ll',
    'MATRIX_SOURCE',
    'ENSEMBL_VAR_COLUMN',
    'PERT_KEY',
    'CELL_LINE_KEY',
    'DOSE_KEY',
    'CONTROL_LABELS',
    'N_HVG',
    'MIN_CONTROL_MEAN',
    'MIN_CONTROL_DETECTION',
    'MIN_CELL_LINES_FOR_HVG',
    'DOSE_MODE',
    'CELL_LINES_TO_USE',
    'MIN_TREATED_CELLS',
    'MIN_CONTROL_CELLS',
    'MAX_CONTROL_CELLS_FOR_COV',
    'COV_SHRINK',
    'RIDGE_PRECISION',
    'JITTER',
    'H_FLOOR',
    'SEED',
    'NORMALIZE_EACH_CONDITION_U_BY_RMS',
    'CONDITION_WEIGHTING',
    'ROW_ZSCORE_FOR_HEATMAP',
    'HEATMAP_CLIP',
    'MAX_GENE_LABELS',
    'ENSEMBL_SERVER',
    'ENSEMBL_BATCH_SIZE',
    'REQUEST_TIMEOUT',
    'FORCE_REFRESH_SYMBOL_CACHE',
    'normalize_label',
    'NORMALIZED_CONTROL_LABELS',
    'is_control',
    'stable_ensembl_id',
    'as_csr',
    'mean_and_variance',
    'detection_fraction',
    'parse_dose',
    'row_rms_normalize',
    'row_zscore',
    'hierarchical_cluster_order',
    'make_http_session',
    'batched',
    'map_ensembl_ids_to_symbols',
    'make_unique_gene_labels',
    'solve_dense_u_star',
    'run_dense_ustar_heatmap',
    'DISTANCE_METRIC',
    'LINKAGE_METHOD',
    'N_PERMUTATIONS',
    'MIN_CATEGORY_SIZE_FOR_TEST',
    'EXCLUDE_CATEGORIES_FROM_TEST',
    'N_BLOCKS_TO_HIGHLIGHT',
    'MIN_BLOCK_SIZE_TO_HIGHLIGHT',
    'EXCLUDE_CATEGORIES_FROM_BLOCK_RANKING',
    'HEATMAP_VMIN',
    'HEATMAP_VMAX',
    'HEATMAP_CMAP',
    'SHOW_GENE_TICK_LABELS',
    'SHOW_DRUG_TICK_LABELS',
    'SHOW_HEATMAP_AXIS_LABELS',
    'BLOCK_LINEWIDTH',
    'BLOCK_LABEL_FONTSIZE',
    'FIGSIZE',
    'DPI',
    'N_HISTOGRAM_BINS',
    'SHOW_FIGURE',
    'normalize_drug_name',
    'contains_any',
    'HDAC_KEYWORDS',
    'OTHER_EPIGENETIC_KEYWORDS',
    'JAK_STAT_KEYWORDS',
    'RTK_MULTIKINASE_KEYWORDS',
    'AURORA_CELL_CYCLE_KEYWORDS',
    'PI3K_MTOR_MAPK_KEYWORDS',
    'DNA_DAMAGE_KEYWORDS',
    'HORMONE_RECEPTOR_KEYWORDS',
    'APOPTOSIS_BCL2_KEYWORDS',
    'CATEGORY_ORDER',
    'CATEGORY_COLORS',
    'categorize_drug',
    'category_color',
    'benjamini_hochberg',
    'empirical_upper_tail_p',
    'safe_z_score',
    'format_probability',
    'hierarchical_order',
    'find_contiguous_blocks',
    'category_clustering_scores',
    'CELL_LINES',
    'NORMALIZE_EACH_U_BY_RMS',
    'SHOW_FIGURES',
    'choose_drug_vectors',
    'prepare_cell_line_matrix',
    'analyze_cell_line',
    'make_three_panel_figure',
    'run_cell_line_specific',
]
