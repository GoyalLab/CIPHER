"""Expression normalizations used by CIPHER.

Every normalization is applied to the *raw count* matrix.  The available modes
mirror the ones benchmarked in the preprint:

    raw        : x
    log1p      : log(1 + x)
    frequency  : x / total_counts_cell
    libsize10k : 1e4 * x / total_counts_cell
    log1CP10k  : log(1 + 1e4 * x / total_counts_cell)
    pflog      : log(x + 1/(4*alpha)) - cellwise_mean_over_genes log(x + 1/(4*alpha))

``pflog`` ("Poisson-fit log") needs a per-dataset dispersion ``alpha`` fit from
control cells via ``Var = mu + alpha * mu**2``; the pseudocount is ``1/(4*alpha)``.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.sparse import issparse

WORK_DTYPE = np.float32
CP10K_SCALE = 10000.0

#: ordered mapping of mode name -> human readable formula
NORMALIZATION_MODES = OrderedDict([
    ("raw",        "x"),
    ("log1p",      "log(1 + x)"),
    ("frequency",  "x / total_counts_cell"),
    ("libsize10k", "10000 * x / total_counts_cell"),
    ("log1CP10k",  "log(1 + 10000 * x / total_counts_cell)"),
    ("pflog",      "log(x + 1/(4 alpha)) - cellwise_mean_genes log(x + 1/(4 alpha))"),
])

# fit knobs for the pflog dispersion
PFLOG_N_BINS = 40
PFLOG_MIN_GENES_PER_BIN = 20
PFLOG_MIN_ALPHA = 1e-8
PFLOG_MAX_ALPHA = 1e8


# --------------------------------------------------------------------------- #
# library size helpers
# --------------------------------------------------------------------------- #
def library_size(X) -> np.ndarray:
    """Per-cell total counts (row sums) of a raw matrix."""
    if issparse(X):
        return np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    return np.sum(np.asarray(X), axis=1, dtype=np.float64)


def safe_library_size(libsize) -> np.ndarray:
    """Row sums with non-positive / non-finite entries replaced by 1."""
    libsize = np.asarray(libsize, dtype=np.float64)
    bad = (~np.isfinite(libsize)) | (libsize <= 0)
    if np.any(bad):
        libsize = libsize.copy()
        libsize[bad] = 1.0
    return libsize


def mean_var(X):
    """Column-wise mean and (unbiased) variance for sparse or dense ``X``."""
    n, p = int(X.shape[0]), int(X.shape[1])
    if n == 0:
        return np.full(p, np.nan), np.full(p, np.nan)
    if issparse(X):
        s = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
        ss = np.asarray(X.multiply(X).sum(axis=0)).ravel().astype(np.float64)
    else:
        Xa = np.asarray(X)
        s = np.sum(Xa, axis=0, dtype=np.float64)
        ss = np.einsum("ij,ij->j", Xa, Xa, optimize=True).astype(np.float64)
    mean = s / float(n)
    if n > 1:
        var = np.maximum((ss - n * mean * mean) / float(n - 1), 0.0)
    else:
        var = np.full(p, np.nan)
    return (np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0))


# --------------------------------------------------------------------------- #
# zero-preserving transforms (raw / log1p / frequency / libsize10k / log1CP10k)
# --------------------------------------------------------------------------- #
def transform_zero_preserving(X, mode, libsize_rows=None):
    """Apply a zero-preserving ``mode`` to ``X`` (sparse stays sparse).

    ``pflog`` is *not* zero preserving (centering makes it dense); use
    :func:`transform_pflog` for that mode.
    """
    if mode == "pflog":
        raise ValueError("pflog is dense after centering; use transform_pflog().")

    if issparse(X):
        Y = X.tocsr(copy=True).astype(WORK_DTYPE, copy=False)
        Y.eliminate_zeros()
        if mode == "raw":
            return Y
        if mode == "log1p":
            Y.data = np.log1p(Y.data).astype(WORK_DTYPE, copy=False)
            return Y
        if mode in {"frequency", "libsize10k", "log1CP10k"}:
            if libsize_rows is None:
                raise ValueError(f"libsize_rows required for {mode}")
            scale = 1.0 / safe_library_size(libsize_rows)
            if mode in {"libsize10k", "log1CP10k"}:
                scale = scale * CP10K_SCALE
            row_ids = np.repeat(np.arange(Y.shape[0], dtype=np.int64), np.diff(Y.indptr))
            Y.data = (Y.data.astype(np.float64, copy=False) * scale[row_ids]).astype(WORK_DTYPE)
            if mode == "log1CP10k":
                Y.data = np.log1p(Y.data).astype(WORK_DTYPE, copy=False)
            Y.eliminate_zeros()
            return Y
        raise ValueError(f"Unknown mode={mode}")

    Y = np.asarray(X, dtype=WORK_DTYPE).copy()
    if mode == "raw":
        return Y
    if mode == "log1p":
        np.log1p(Y, out=Y)
        return Y
    if mode in {"frequency", "libsize10k", "log1CP10k"}:
        if libsize_rows is None:
            raise ValueError(f"libsize_rows required for {mode}")
        scale = 1.0 / safe_library_size(libsize_rows)
        if mode in {"libsize10k", "log1CP10k"}:
            scale = scale * CP10K_SCALE
        Y *= scale[:, None].astype(WORK_DTYPE)
        if mode == "log1CP10k":
            np.log1p(Y, out=Y)
        return Y
    raise ValueError(f"Unknown mode={mode}")


# --------------------------------------------------------------------------- #
# pflog
# --------------------------------------------------------------------------- #
def fit_pflog_alpha(X0_raw, gene_names=None):
    """Fit the pflog dispersion ``alpha`` from a raw *control* matrix.

    Returns ``(alpha, pseudocount, meanvar_df, bin_df)`` where
    ``pseudocount = 1 / (4 * alpha)``.
    """
    mean0, var0 = mean_var(X0_raw)
    mu = np.asarray(mean0, dtype=np.float64)
    var = np.asarray(var0, dtype=np.float64)
    excess = var - mu
    alpha_gene = excess / np.maximum(mu * mu, 1e-30)
    alpha_gene[(~np.isfinite(alpha_gene)) | (mu <= 0) | (var <= 0)] = np.nan
    used = np.isfinite(mu) & np.isfinite(var) & (mu > 0) & (var > 0)
    if gene_names is None:
        gene_names = np.arange(mu.size)
    meanvar_df = pd.DataFrame({
        "gene": np.asarray(gene_names, dtype=str),
        "raw_control_mean": mu,
        "raw_control_var": var,
        "raw_control_excess_var_minus_mean": excess,
        "genewise_alpha": alpha_gene,
        "used_for_fit": used,
    })

    mu_u, var_u = mu[used], var[used]
    if mu_u.size < 10:
        alpha = 1.0
        return alpha, 1.0 / (4.0 * alpha), meanvar_df, pd.DataFrame()

    order = np.argsort(mu_u)
    mu_u, var_u = mu_u[order], var_u[order]
    log_mu = np.log10(mu_u)
    edges = np.linspace(np.min(log_mu), np.max(log_mu), PFLOG_N_BINS + 1)
    rows = []
    for i in range(PFLOG_N_BINS):
        lo, hi = edges[i], edges[i + 1]
        m = (log_mu >= lo) & (log_mu <= hi if i == PFLOG_N_BINS - 1 else log_mu < hi)
        if np.sum(m) < PFLOG_MIN_GENES_PER_BIN:
            continue
        rows.append({"bin": i, "n_genes": int(np.sum(m)),
                     "mu_bin": float(np.mean(mu_u[m])), "var_bin": float(np.mean(var_u[m]))})
    bin_df = pd.DataFrame(rows)
    if len(bin_df) >= 3:
        mu_fit = bin_df["mu_bin"].to_numpy(np.float64)
        var_fit = bin_df["var_bin"].to_numpy(np.float64)
    else:
        mu_fit, var_fit = mu_u, var_u

    x = mu_fit ** 2
    y = var_fit - mu_fit
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if np.sum(finite) >= 3:
        alpha = float(np.sum(x[finite] * y[finite]) / max(np.sum(x[finite] * x[finite]), 1e-30))
    else:
        alpha = 1.0
    if not np.isfinite(alpha) or alpha <= 0:
        ratios = (y[finite] / np.maximum(x[finite], 1e-30))
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        alpha = float(np.median(ratios)) if ratios.size else 1.0
    alpha = float(np.clip(alpha, PFLOG_MIN_ALPHA, PFLOG_MAX_ALPHA))
    return alpha, float(1.0 / (4.0 * alpha)), meanvar_df, bin_df


def pflog_row_centers(X_filtered_rows, pseudocount):
    """Cellwise mean of ``log(x + pc)`` over the (kept) genes — the pflog centering term."""
    pc = float(pseudocount)
    n, p = int(X_filtered_rows.shape[0]), int(X_filtered_rows.shape[1])
    log_pc = float(np.log(pc))
    if issparse(X_filtered_rows):
        Xcsr = X_filtered_rows.tocsr()
        Xcsr.eliminate_zeros()
        centers = np.full(n, p * log_pc, dtype=np.float64)
        if Xcsr.nnz > 0:
            delta = np.log(Xcsr.data.astype(np.float64) + pc) - log_pc
            row_ids = np.repeat(np.arange(n, dtype=np.int64), np.diff(Xcsr.indptr))
            np.add.at(centers, row_ids, delta)
        return centers / float(p)
    return np.mean(np.log(np.asarray(X_filtered_rows, dtype=np.float64) + pc), axis=1)


def transform_pflog(X_chunk, center_rows, pseudocount):
    """Dense pflog transform of a raw chunk given precomputed row centers."""
    pc = float(pseudocount)
    if issparse(X_chunk):
        Y = X_chunk.toarray().astype(WORK_DTYPE, copy=False)
    else:
        Y = np.asarray(X_chunk, dtype=WORK_DTYPE).copy()
    Y = np.log(Y.astype(np.float64, copy=False) + pc).astype(WORK_DTYPE)
    Y -= np.asarray(center_rows, dtype=WORK_DTYPE)[:, None]
    return Y


# --------------------------------------------------------------------------- #
# convenience dense dispatcher (used by the in-memory forward/reverse path)
# --------------------------------------------------------------------------- #
def normalize_matrix(X, mode="log1p", libsize=None, pseudocount=None, pflog_center=None):
    """Normalize a raw matrix with ``mode`` and return a dense float32 array.

    For ``pflog`` you must pass ``pseudocount`` (from :func:`fit_pflog_alpha`);
    ``pflog_center`` is computed from ``X`` if not given.
    """
    if mode == "pflog":
        if pseudocount is None:
            raise ValueError("pflog requires a pseudocount from fit_pflog_alpha().")
        if pflog_center is None:
            pflog_center = pflog_row_centers(X, pseudocount)
        return transform_pflog(X, pflog_center, pseudocount)
    if mode in {"frequency", "libsize10k", "log1CP10k"} and libsize is None:
        libsize = library_size(X)
    Y = transform_zero_preserving(X, mode, libsize_rows=libsize)
    return Y.toarray() if issparse(Y) else np.asarray(Y)
