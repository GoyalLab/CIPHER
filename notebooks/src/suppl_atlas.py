"""Stateless helpers for the Fig 3/4 CellxGene-atlas covariance-transfer supplement.

Covers the two CellxGene-atlas covariance-transfer snapshots: the Marson T-cell
CD4/CD8 atlas and the RPE + far-from-RPE atlas. NOT part of the installable
``cipher`` package — a notebook-only helper for reproducing the supplementary/main
atlas panels.

Only the stateless, self-contained helpers live here (no dependence on notebook
config globals such as EPS / OUTDIR / CENSUS_VERSION / CONTROL_LABELS). The
config-coupled and cell-divergent functions (choose_cellxgene_sources,
plot_composite_violin, evaluate_response_vs_*, compute_cov_columns_from_raw_counts,
etc.) intentionally stay inline in the notebook cells, because they read per-pipeline
constants and differ between the T-cell and RPE snapshots; keeping them inline is what
preserves 1:1 reproduction.
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
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import issparse, csr_matrix
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt

try:  # pragma: no cover - optional heavy deps
    from matplotlib.colors import LogNorm
except Exception:
    LogNorm = None

try:  # pragma: no cover
    import h5py
except Exception:
    h5py = None

try:  # pragma: no cover
    import anndata as ad
except Exception:
    ad = None

try:  # pragma: no cover
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, *a, **k):
        return x if x is not None else iter(())

try:  # pragma: no cover - not installed in the parse/test env; needed at run time
    import cellxgene_census
except Exception:
    cellxgene_census = None


def classify_marson_pair(response_dataset, source_dataset):
    if str(response_dataset) == str(source_dataset):
        return "Marson within dataset"

    c_resp = extract_marson_condition(response_dataset)
    c_src = extract_marson_condition(source_dataset)

    if c_resp is not None and c_src is not None and c_resp == c_src:
        return "Marson within condition"

    return "Marson cross condition"


def classify_rpe_pair(response, source):
    if response["dataset"] == source["dataset"]:
        return "RPE Pert-seq within dataset"
    return "RPE Pert-seq cross dataset"


def clean_dataset_label(name, maxlen=55):
    s = str(name)
    s = re.sub(r"__mean_ge_[0-9p.]+$", "", s)
    s = s.replace("_filtered", "")
    s = s.replace("_raw_singlecell_01", "")
    s = s.replace("_raw_singlecell", "")
    s = s.replace("__", "_")
    if len(s) > maxlen:
        s = s[: maxlen - 3] + "..."
    return s


def compute_gene_stats_from_raw_counts(X_raw):
    """
    Returns:
      mean
      detection fraction
      sample variance
      depth vector
    """
    if issparse(X_raw):
        X = X_raw.astype(np.float64).tocsr(copy=False)
        n = X.shape[0]
        mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel().astype(np.float64)
        det = np.asarray((X > 0).mean(axis=0)).ravel().astype(np.float64)
        depth = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    else:
        X = np.asarray(X_raw, dtype=np.float64)
        n = X.shape[0]
        mean = X.mean(axis=0).astype(np.float64)
        mean_sq = (X * X).mean(axis=0).astype(np.float64)
        det = (X > 0).mean(axis=0).astype(np.float64)
        depth = X.sum(axis=1).astype(np.float64)

    var = mean_sq - mean * mean
    if n > 1:
        var = var * n / (n - 1)
    var = np.maximum(var, 0.0)

    return {
        "mean": mean,
        "detection": det,
        "var": var,
        "depth": depth,
    }


def contains_any_keyword(text, keywords):
    text = str(text).lower()
    return any(k.lower() in text for k in keywords)


def decode_str_array(x):
    out = []
    for y in np.asarray(x, dtype=object):
        if isinstance(y, bytes):
            out.append(y.decode("utf-8"))
        else:
            out.append(str(y))
    return np.asarray(out, dtype=object)


def extract_marson_condition(dataset_name):
    s = str(dataset_name)
    s = re.sub(r"__mean_ge_[0-9p.]+$", "", s)
    s = s.replace("_filtered", "")

    m = re.search(r"_D\d+_([^_]+)$", s)
    if m:
        return m.group(1)

    toks = s.split("_")
    if len(toks) >= 2:
        return toks[-1]

    return None


def gene_hash(gene_names):
    return hashlib.md5("\n".join(map(str, gene_names)).encode()).hexdigest()[:12]


def get_raw_X(adata_obj, name="adata"):
    preferred_layers = [
        "counts",
        "raw_counts",
        "count",
        "X_counts",
        "umi_counts",
        "UMI",
        "umis",
    ]

    for layer in preferred_layers:
        if layer in adata_obj.layers:
            print(f"[raw X] {name}: using layer '{layer}'")
            return adata_obj.layers[layer]

    if adata_obj.raw is not None:
        raw_var = np.asarray(adata_obj.raw.var_names.astype(str))
        need = set(adata_obj.var_names.astype(str))

        if need.issubset(set(raw_var)):
            print(f"[raw X] {name}: using adata.raw.X")
            return adata_obj.raw[:, adata_obj.var_names].X

    print(f"[raw X] {name}: using adata.X")
    return adata_obj.X


def infer_gene_names(adata_obj):
    for k in ["gene_symbol", "gene_symbols", "feature_name", "gene_name", "Gene", "symbol"]:
        if k in adata_obj.var.columns:
            vals = safe_str_col(adata_obj.var, k).values
            if len(pd.unique(vals)) > 100:
                adata_obj.var_names = vals
                break

    adata_obj.var_names = adata_obj.var_names.astype(str).str.strip()
    adata_obj.var_names_make_unique()
    return adata_obj


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def make_skip_summary(response, source_dataset, source_label, source_type, composite_group, reason, n_overlap_genes=0):
    return {
        "response_dataset": response["dataset"],
        "response_label": response["label"],
        "response_condition": response.get("condition", ""),
        "source_dataset": source_dataset,
        "source_label": source_label,
        "source_condition": "",
        "source_type": source_type,
        "composite_group": composite_group,
        "covariance_variant": "",
        "covariance_mode": "",
        "n_eval": 0,
        "n_overlap_genes": int(n_overlap_genes),
        "skip_reason": reason,
    }


def mode_or_empty(x):
    s = pd.Series(x).astype("object")
    s = s.where(pd.notna(s), "")
    s = s.astype(str)
    s = s[(s != "") & (s.str.lower() != "nan")]
    if len(s) == 0:
        return ""
    return str(s.value_counts().index[0])


def pert_to_gene_safe(pert, gene_set):
    p0 = str(pert).strip()

    if p0 in gene_set:
        return p0

    p = p0
    p = re.sub(r"([_\-\s]+)(KD|KO|OE|overexp|overexpression)$", "", p, flags=re.IGNORECASE)
    p = re.sub(r"^(sg)(?=[A-Z0-9])", "", p)
    p = re.sub(r"^(sgRNA|gRNA|sgrna|grna|sg)([_\-\s]+)", "", p, flags=re.IGNORECASE)

    for sep in ["_", "+", "-", "|", " "]:
        if sep in p:
            p = p.split(sep)[0]
            break

    if p in gene_set:
        return p

    return None


def safe_str_col(df, col, default=""):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=object).astype(str)

    s = df[col].astype("object")
    s = s.where(pd.notna(s), default)
    s = s.astype(str)
    s = s.replace(
        {
            "nan": default,
            "NaN": default,
            "None": default,
            "<NA>": default,
            "nat": default,
            "NaT": default,
        }
    )
    return s


def sanitize_filename(s, maxlen=180):
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_")[:maxlen]


def sparse_or_dense_mean0(X):
    if issparse(X):
        return np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
    return np.asarray(X, dtype=np.float64).mean(axis=0)


def sparse_or_dense_rowsum(X):
    if issparse(X):
        return np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    return np.asarray(X, dtype=np.float64).sum(axis=1).astype(np.float64)


def stable_seed(*items, modulo=2**32 - 1):
    s = "||".join(map(str, items))
    h = hashlib.md5(s.encode()).hexdigest()
    return int(h[:12], 16) % modulo


def summarize_metric(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "sem": float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def summarize_pair_from_perts(pert_df, meta):
    summary = dict(meta)

    metric_cols = [
        "pearson",
        "r2_uncentered",
        "pearson_lto",
        "r2_uncentered_lto",
        "alpha",
        "alpha_lto",
    ]

    for col in metric_cols:
        if col not in pert_df.columns:
            continue

        s = summarize_metric(pd.to_numeric(pert_df[col], errors="coerce").values)

        stem = col
        summary[f"n_{stem}"] = s["n"]
        summary[f"mean_{stem}"] = s["mean"]
        summary[f"median_{stem}"] = s["median"]
        summary[f"sem_{stem}"] = s["sem"]
        summary[f"std_{stem}"] = s["std"]
        summary[f"min_{stem}"] = s["min"]
        summary[f"max_{stem}"] = s["max"]

    summary["n_eval"] = int(len(pert_df))
    summary["skip_reason"] = ""
    return summary


def threshold_to_tag(x):
    return str(x).replace(".", "p")


def unique_join_limited(x, max_items=8):
    s = pd.Series(x).astype("object")
    s = s.where(pd.notna(s), "")
    vals = s.astype(str).unique().tolist()
    vals = [v for v in vals if v and v.lower() not in {"nan", "none", "<na>"}]
    return "; ".join(vals[:max_items])
