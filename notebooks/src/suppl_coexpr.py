"""Shared co-expression engine for the Fig S18 supplements (GGI / PPI).

A notebook-only helper for reproducing the supplementary figures. NOT part of the
installable ``cipher`` package.

The engine measures whether interacting gene/protein pairs are more correlated in
control cells than random pairs, across datasets and preprocessing modes.
"""
from __future__ import annotations


# --- library imports required by the functions below
#     (resolved at call time so placement after the docstring is sufficient) ---
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
import importlib.util as _iu

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from scipy.stats import mannwhitneyu

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

__all__ = ['_control_keys', '_find_spec_no_torch', '_perturbation_columns', 'analyze_dataset_ppi_vs_random_sparse', 'clean_dataset_name', 'compute_logratio_curve', 'compute_mu_sd_sparse', 'corr_for_pairs_sparse_chunked', 'find_pair_file', 'find_pair_files', 'find_previous_pair_file', 'format_pvalue', 'get_control_sparse', 'infer_dataset_from_filename', 'load_and_compile_condition', 'load_condition_pairs', 'load_pair_file', 'load_ppi_unique_pairs', 'paired_ttest_pvalue', 'paired_wilcoxon_pvalue', 'safe_make_var_names_unique', 'sample_random_pairs', 'standardize_pair_df']


# --- notebook config globals ---
# The functions read these as module globals. The driving notebook
# (notebooks/suppl/figS18_coexpr.ipynb) overrides them at call time by injecting
# its UPPER-case config into this module's namespace. The defaults below keep the
# module importable / standalone.
OUTDIR = "coexpr_vs_random_outputs"   # output dir; the notebook injects the real path under its OUTDIR
Z = 1.96                              # approx 95% CI z-score for the tail log-ratio curves
USE_ABS = False                      # False: signed Pearson r; True: |r|

# Torch-avoid shim: capture the real
# importlib.util.find_spec so _find_spec_no_torch can delegate to it. The actual
# patch (_iu.find_spec = _find_spec_no_torch) is performed once by the notebook's
# config cell, so it is intentionally NOT applied here -- doing it at import time
# would make the notebook's own capture line re-wrap this shim and recurse.
_real_find_spec = _iu.find_spec


def _find_spec_no_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        return None
    return _real_find_spec(name, *args, **kwargs)


def safe_make_var_names_unique(adata, join="-"):
    try:
        adata.var_names = pd.Index([str(x) for x in list(adata.var_names)], dtype="object")
    except Exception:
        adata.var_names = pd.Index(pd.Series(adata.var_names).astype(str).tolist(), dtype="object")

    if hasattr(adata, "var") and isinstance(adata.var, pd.DataFrame):
        for c in adata.var.columns:
            if isinstance(adata.var[c].dtype, pd.CategoricalDtype):
                adata.var[c] = adata.var[c].astype(str)

    adata.var_names_make_unique(join=join)
    return adata


def _perturbation_columns():
    return set(["gene", "perturbation_1"])


def _control_keys():
    return set(["control", "NT", "non-targeting", "ctrl"])


def load_ppi_unique_pairs(ppi_csv_path):
    ppi = pd.read_csv(ppi_csv_path)
    need_cols = {"Interactor A", "Interactor B"}
    if not need_cols.issubset(ppi.columns):
        raise ValueError(f"PPI CSV must contain columns {need_cols}, found {set(ppi.columns)}")

    ppi = ppi.dropna(subset=["Interactor A", "Interactor B"]).copy()
    a = ppi["Interactor A"].astype(str).to_numpy()
    b = ppi["Interactor B"].astype(str).to_numpy()

    g1 = np.minimum(a, b)
    g2 = np.maximum(a, b)
    mask = g1 != g2
    g1 = g1[mask]
    g2 = g2[mask]

    pairs = list(set(zip(g1.tolist(), g2.tolist())))
    pairs.sort()
    return ppi, pairs


def sample_random_pairs(all_genes, forbidden_set, target_n, rng):
    seen = set()
    pairs = []
    nG = len(all_genes)
    batch_size = max(5000, 5 * max(1, target_n))

    while len(pairs) < target_n:
        idx1 = rng.integers(0, nG, size=batch_size)
        idx2 = rng.integers(0, nG, size=batch_size)

        for i1, i2 in zip(idx1, idx2):
            if i1 == i2:
                continue
            g1, g2 = all_genes[i1], all_genes[i2]
            if g1 == g2:
                continue
            key = (g1, g2) if g1 < g2 else (g2, g1)
            if key in seen or key in forbidden_set:
                continue
            seen.add(key)
            pairs.append(key)
            if len(pairs) == target_n:
                break

    return pairs


def get_control_sparse(
    data_path,
    expression_threshold=0.01,
    min_samples=2,
    unique_join="-",
    log1p=False,
    norm=False,
):
    """
    Returns:
      X0_csc : control matrix in CSC (n0 x G_kept), sparse
      genes  : np.array of kept gene names (length G_kept)
      n0     : # control cells
      adata  : filtered AnnData (for counts)
    """
    print(f"\n[load] dataset: {os.path.basename(str(data_path))}")
    adata = ad.read_h5ad(data_path)
    safe_make_var_names_unique(adata, join=unique_join)

    if norm:
        print("Normalizing adata (CPM scale to 1e6)")
        sc.pp.normalize_total(adata, target_sum=1e6)

    if log1p:
        print("Applying log1p transform...")
        sc.pp.log1p(adata)

    # perturbation column
    if "perturbation" not in adata.obs.columns:
        hits = list(set(adata.obs.columns).intersection(_perturbation_columns()))
        if not hits:
            raise ValueError(f"{os.path.basename(data_path)}: missing 'perturbation' and no fallback in {_perturbation_columns()}")
        adata.obs["perturbation"] = adata.obs[hits[0]].astype(str)
    else:
        adata.obs["perturbation"] = adata.obs["perturbation"].astype(str)

    # normalize control label to "control"
    uniq = set(adata.obs["perturbation"].unique())
    ctrl_hits = uniq.intersection(_control_keys())
    if not ctrl_hits:
        sample = sorted(list(uniq))[:15]
        raise ValueError(f"{os.path.basename(data_path)}: no control label found among {_control_keys()}; saw e.g. {sample}")
    ctrl_key = sorted(list(ctrl_hits))[0]
    adata.obs.loc[adata.obs["perturbation"] == ctrl_key, "perturbation"] = "control"

    # filter cells with >=1 total count (sparse-safe)
    X = adata.X
    if issparse(X):
        tot = np.asarray(X.sum(axis=1)).reshape(-1)
    else:
        tot = np.asarray(X).sum(axis=1)
    adata = adata[tot >= 1, :].copy()

    ctrl_mask = (adata.obs["perturbation"].to_numpy() == "control")
    n0 = int(ctrl_mask.sum())
    if n0 < 2:
        raise ValueError(f"{os.path.basename(data_path)}: too few control cells ({n0})")

    # control matrix (keep sparse)
    X0 = adata[ctrl_mask, :].X
    if not issparse(X0):
        # if dense for some reason, convert to CSR then CSC
        import scipy.sparse as sp
        X0 = sp.csr_matrix(np.asarray(X0))
    X0 = X0.tocsc()

    # gene means on control (sparse)
    # mean = sum/n
    sum_x = np.asarray(X0.sum(axis=0)).reshape(-1).astype(np.float64)
    mu = sum_x / float(n0)

    # force-keep perturbation genes if present
    pert_labels = [p for p in adata.obs["perturbation"].unique() if p != "control"]
    pert_set = set(map(str, pert_labels))
    force_keep = np.array([g in pert_set for g in adata.var_names], dtype=bool)

    keep_genes = (mu >= float(expression_threshold)) | force_keep

    # apply gene filter to adata and X0
    adata = adata[:, keep_genes].copy()
    genes = np.array(adata.var_names.tolist(), dtype=object)

    # recompute control X0 with kept genes
    ctrl_mask = (adata.obs["perturbation"].astype(str).to_numpy() == "control")
    X0 = adata[ctrl_mask, :].X
    if not issparse(X0):
        import scipy.sparse as sp
        X0 = sp.csr_matrix(np.asarray(X0))
    X0_csc = X0.tocsc()
    n0 = X0_csc.shape[0]

    # filter perturbations by min_samples (does not change X0, but keeps consistency)
    counts = pd.Series(adata.obs["perturbation"].astype(str)).value_counts()
    valid_perts = counts[counts >= int(min_samples)].index.tolist()
    adata = adata[adata.obs["perturbation"].astype(str).isin(valid_perts)].copy()

    print(f"[load] kept cells={adata.n_obs:,} genes={adata.n_vars:,} | control cells={n0:,}")
    return X0_csc, genes, n0, adata


def compute_mu_sd_sparse(X0_csc, n0, eps_var=1e-12):
    """
    mu_i = sum x / n
    var_i = (sum x^2 - n*mu^2) / (n-1)
    sd_i = sqrt(max(var_i, eps))
    """
    # sums
    sum_x = np.asarray(X0_csc.sum(axis=0)).reshape(-1).astype(np.float64)
    mu = sum_x / float(n0)

    # sumsq (sparse-safe)
    X2 = X0_csc.copy()
    X2.data = X2.data * X2.data
    sum_x2 = np.asarray(X2.sum(axis=0)).reshape(-1).astype(np.float64)

    denom = float(max(1, n0 - 1))
    var = (sum_x2 - float(n0) * (mu * mu)) / denom
    var = np.maximum(var, eps_var)
    sd = np.sqrt(var)
    return mu, sd, denom


def corr_for_pairs_sparse_chunked(X0_csc, mu, sd, denom, ia, ib, chunk_size=20000, cap=0.999999):
    """
    Computes r for matched pairs (ia[k], ib[k]) without making Z.
    Uses:
      cov_ij = (x_i^T x_j - n*mu_i*mu_j) / (n-1)
      r_ij   = cov_ij / (sd_i sd_j)
    where x_i^T x_j is computed via elementwise multiply in sparse column slices.
    """
    ia = np.asarray(ia, dtype=np.int64)
    ib = np.asarray(ib, dtype=np.int64)
    m = ia.size
    out = np.empty(m, dtype=np.float64)

    n = X0_csc.shape[0]
    for s in range(0, m, chunk_size):
        e = min(m, s + chunk_size)
        a = ia[s:e]
        b = ib[s:e]

        Xa = X0_csc[:, a]  # (n x k) CSC
        Xb = X0_csc[:, b]  # (n x k) CSC

        # xy_k = sum_c Xa[c,k] * Xb[c,k]
        xy = np.asarray(Xa.multiply(Xb).sum(axis=0)).reshape(-1).astype(np.float64)

        cov = (xy - float(n) * (mu[a] * mu[b])) / denom
        r = cov / (sd[a] * sd[b])

        if cap is not None:
            r = np.clip(r, -cap, cap)
        out[s:e] = r

    return out


def analyze_dataset_ppi_vs_random_sparse(
    data_path,
    ppi_pairs_unique,
    expression_threshold=0.01,
    min_samples=2,
    use_abs=False,
    save_dir="ppi_vs_random_outputs",
    rng_seed=0,
    plot_bins=80,
    tail_grid_points=200,
    eps_var=1e-12,
    chunk_size=20000,
    unique_join="-",
    show_plots=True,
    log1p=False,
    norm=False,
    preprocess_name="raw",
):
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(data_path).replace(".h5ad", "")

    # Load sparse control matrix + genes
    X0_csc, genes, n0, adata = get_control_sparse(
        data_path=data_path,
        expression_threshold=expression_threshold,
        min_samples=min_samples,
        unique_join=unique_join,
        log1p=log1p,
        norm=norm,
    )

    # Precompute mu, sd ONCE
    mu, sd, denom = compute_mu_sd_sparse(X0_csc, n0, eps_var=eps_var)

    # Map gene -> index
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # Filter PPI to usable
    usable_ppi = [(a, b) for (a, b) in ppi_pairs_unique if (a in gene_to_idx and b in gene_to_idx)]
    n_ppi = len(usable_ppi)
    print(f"[{base} | {preprocess_name}] usable PPI pairs: {n_ppi:,} (from {len(ppi_pairs_unique):,} unique undirected PPI total)")
    if n_ppi == 0:
        raise ValueError(f"[{base} | {preprocess_name}] No usable PPI pairs overlap with kept genes.")

    # Indices for PPI
    ia = np.fromiter((gene_to_idx[a] for a, b in usable_ppi), dtype=np.int64, count=n_ppi)
    ib = np.fromiter((gene_to_idx[b] for a, b in usable_ppi), dtype=np.int64, count=n_ppi)

    # PPI correlations (chunked)
    r_ppi = corr_for_pairs_sparse_chunked(
        X0_csc, mu, sd, denom, ia, ib, chunk_size=chunk_size
    )
    fin = np.isfinite(r_ppi)
    r_ppi = r_ppi[fin]
    usable_ppi = [p for p, ok in zip(usable_ppi, fin) if ok]
    ppi_score = np.abs(r_ppi) if use_abs else r_ppi
    n_ppi = len(ppi_score)

    # Random pairs: EXACTLY n_ppi
    rng = np.random.default_rng(rng_seed)
    all_genes = np.array(list(gene_to_idx.keys()), dtype=object)
    forbidden = set(usable_ppi)
    rand_pairs = sample_random_pairs(all_genes, forbidden, target_n=n_ppi, rng=rng)

    ja = np.fromiter((gene_to_idx[a] for a, b in rand_pairs), dtype=np.int64, count=n_ppi)
    jb = np.fromiter((gene_to_idx[b] for a, b in rand_pairs), dtype=np.int64, count=n_ppi)

    r_rand = corr_for_pairs_sparse_chunked(
        X0_csc, mu, sd, denom, ja, jb, chunk_size=chunk_size
    )
    finr = np.isfinite(r_rand)
    r_rand = r_rand[finr]
    rand_pairs = [p for p, ok in zip(rand_pairs, finr) if ok]
    rand_score = np.abs(r_rand) if use_abs else r_rand

    # Match counts if finiteness differs
    m = min(len(ppi_score), len(rand_score))
    ppi_score = ppi_score[:m]
    rand_score = rand_score[:m]
    usable_ppi = usable_ppi[:m]
    rand_pairs = rand_pairs[:m]
    r_ppi = r_ppi[:m]
    r_rand = r_rand[:m]
    n_ppi = n_rand = m

    print(f"[{base} | {preprocess_name}] final counts (finite): PPI={n_ppi:,} random={n_rand:,}")

    # Stats
    ks_stat, ks_p = ks_2samp(ppi_score, rand_score)
    y = np.concatenate([np.ones(n_ppi, dtype=int), np.zeros(n_rand, dtype=int)])
    s = np.concatenate([ppi_score, rand_score]).astype(float)
    try:
        auc = float(roc_auc_score(y, s))
    except Exception:
        auc = np.nan

    # Save per-pair CSV
    df_ppi = pd.DataFrame({
        "preprocess": [preprocess_name] * n_ppi,
        "type": ["PPI"] * n_ppi,
        "gene A": [a for a, b in usable_ppi],
        "gene B": [b for a, b in usable_ppi],
        "r": r_ppi,
        "score": ppi_score,
    })
    df_rnd = pd.DataFrame({
        "preprocess": [preprocess_name] * n_rand,
        "type": ["Random"] * n_rand,
        "gene A": [a for a, b in rand_pairs],
        "gene B": [b for a, b in rand_pairs],
        "r": r_rand,
        "score": rand_score,
    })
    df_pairs = pd.concat([df_ppi, df_rnd], ignore_index=True)
    pairs_csv = os.path.join(save_dir, f"{base}__{preprocess_name}__ppi_vs_random_pairs.csv")
    df_pairs.to_csv(pairs_csv, index=False)

    # Plots (normalized histogram)
    xlabel = "|r|" if use_abs else "r"
    plt.figure(figsize=(10, 7))
    plt.hist(ppi_score, bins=plot_bins, alpha=0.6, label="PPI", density=True)
    plt.hist(rand_score, bins=plot_bins, alpha=0.6, label="Random", density=True)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Density (log scale)")
    plt.title(f"{base} [{preprocess_name}]: PPI vs Random (normalized hist)")
    plt.legend()
    plt.text(
        0.98, 0.95,
        f"KS={ks_stat:.3f}\np={ks_p:.2e}\nAUC={auc:.3f}\nN={n_ppi:,}",
        transform=plt.gca().transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
    )
    plt.tight_layout()
    p_hist = os.path.join(save_dir, f"{base}__{preprocess_name}__ppi_vs_random_hist.png")
    plt.savefig(p_hist, dpi=300)
    plt.savefig(p_hist.replace(".png", ".svg"))
    if show_plots:
        plt.show()
    plt.close()

    # Tail survival
    lo = float(np.nanmin(np.concatenate([ppi_score, rand_score])))
    hi = float(np.nanmax(np.concatenate([ppi_score, rand_score])))
    if hi > lo:
        grid = np.linspace(lo, hi, tail_grid_points)
        surv_ppi = np.array([np.mean(ppi_score >= c) for c in grid], dtype=float)
        surv_rnd = np.array([np.mean(rand_score >= c) for c in grid], dtype=float)

        plt.figure(figsize=(10, 7))
        plt.plot(grid, surv_ppi, label="PPI")
        plt.plot(grid, surv_rnd, label="Random")
        plt.yscale("log")
        plt.xlabel(f"{xlabel} cutoff c")
        plt.ylabel("P(score ≥ c) (log)")
        plt.title(f"{base} [{preprocess_name}]: tail survival")
        plt.legend()
        plt.tight_layout()
        p_tail = os.path.join(save_dir, f"{base}__{preprocess_name}__ppi_vs_random_tail.png")
        plt.savefig(p_tail, dpi=300)
        plt.savefig(p_tail.replace(".png", ".svg"))
        if show_plots:
            plt.show()
        plt.close()
    else:
        p_tail = None

    summary = {
        "dataset": base,
        "preprocess": preprocess_name,
        "log1p": bool(log1p),
        "norm": bool(norm),
        "pairs_csv": pairs_csv,
        "n_cells_control": int(n0),
        "n_genes_kept": int(len(genes)),
        "n_pairs_ppi": int(n_ppi),
        "n_pairs_random": int(n_rand),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "hist_plot": p_hist,
        "tail_plot": p_tail,
        "chunk_size": int(chunk_size),
    }
    return summary


def clean_dataset_name(x):
    return str(x).replace(".h5ad", "")


def find_previous_pair_file(base_output_dir, preprocess_short, dataset_clean, aliases=None):
    """
    Finds files produced by the GGI run.

    Compatible with current GGI script outputs that may still use PPI-style names:

        ggi_vs_random_outputs/raw/DATASET__raw__ppi_vs_random_pairs.csv
        ggi_vs_random_outputs/log1p/DATASET__log1p__ppi_vs_random_pairs.csv
        ggi_vs_random_outputs/norm/DATASET__norm__ppi_vs_random_pairs.csv
        ggi_vs_random_outputs/norm_plust_log1p/DATASET__norm_plust_log1p__ppi_vs_random_pairs.csv

    Also supports future renamed GGI-style names:

        DATASET__raw__ggi_vs_random_pairs.csv
    """
    if aliases is None:
        aliases = [preprocess_short]

    aliases = list(dict.fromkeys([preprocess_short] + list(aliases)))

    patterns = []

    for key in aliases:
        patterns.extend([
            # Current PPI-style filenames from GGI-generating script
            os.path.join(
                base_output_dir,
                key,
                f"{dataset_clean}__{key}__ppi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                key,
                f"{dataset_clean}*ppi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                "**",
                f"{dataset_clean}__{key}__ppi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                "**",
                f"{dataset_clean}*{key}*ppi_vs_random_pairs.csv",
            ),

            # Future GGI-style filenames
            os.path.join(
                base_output_dir,
                key,
                f"{dataset_clean}__{key}__ggi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                key,
                f"{dataset_clean}*ggi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                "**",
                f"{dataset_clean}__{key}__ggi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                "**",
                f"{dataset_clean}*{key}*ggi_vs_random_pairs.csv",
            ),
        ])

    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat, recursive=True))

    hits = sorted(set(hits))

    # Prefer explicit alias matches
    preferred = []
    for h in hits:
        h_norm = h.replace("\\", "/")
        base = os.path.basename(h_norm)

        for key in aliases:
            if f"/{key}/" in h_norm or f"__{key}__" in base or key in base:
                preferred.append(h)
                break

    preferred = sorted(set(preferred))
    if len(preferred) > 0:
        return preferred[0]

    return hits[0] if len(hits) > 0 else None


def standardize_pair_df(df, dataset_clean, condition_short):
    """
    Converts previous-script pair CSV format:

        type, gene A, gene B, r, score

    into long gene-centered format:

        gene, partner, group, corr, dataset, condition

    Each pair contributes twice:
        gene A -> gene B
        gene B -> gene A

    IMPORTANT:
      The GGI-generating script may still write type == "PPI".
      Here, type == "PPI" is intentionally interpreted as GGI.
    """
    required_cols = ["type", "gene A", "gene B", "r"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_cols].copy()
    df["dataset"] = dataset_clean
    df["condition"] = condition_short

    df["gene A"] = df["gene A"].astype(str)
    df["gene B"] = df["gene B"].astype(str)
    df["type_clean"] = df["type"].astype(str).str.strip().str.lower()
    df["r"] = pd.to_numeric(df["r"], errors="coerce")

    df = df.dropna(subset=["gene A", "gene B", "type_clean", "r"])

    # Compatible with exact output of the GGI run:
    # type may still be "PPI", but those pairs are actually genetic interactions.
    ggi_type_labels = [
        "ggi",
        "gi",
        "genetic_interaction",
        "genetic interaction",
        "genetic-interaction",
        "ppi",
    ]

    ggi_mask = df["type_clean"].isin(ggi_type_labels)

    rand_mask = (
        df["type_clean"].eq("random")
        | df["type_clean"].eq("random_non_ggi")
        | df["type_clean"].eq("random_non_ppi")
        | df["type_clean"].str.startswith("random")
    )

    df = df[ggi_mask | rand_mask].copy()
    df["group"] = np.where(df["type_clean"].isin(ggi_type_labels), "GGI", "Random")

    left = df.rename(columns={"gene A": "gene", "gene B": "partner", "r": "corr"})[
        ["dataset", "condition", "gene", "partner", "group", "corr"]
    ].copy()

    right = df.rename(columns={"gene B": "gene", "gene A": "partner", "r": "corr"})[
        ["dataset", "condition", "gene", "partner", "group", "corr"]
    ].copy()

    long_df = pd.concat([left, right], axis=0, ignore_index=True)
    return long_df


def load_and_compile_condition(
    base_output_dir,
    condition_name,
    condition_short,
    condition_aliases,
    datasets,
    threshold,
):
    """
    For one preprocessing condition:
      1) Load all previous pair-level CSVs.
      2) Expand each pair into two gene-centered rows.
      3) For each gene, compute:

            GGI survival    = P(r_GGI > threshold)
            Random survival = P(r_Random > threshold)

      4) Save compiled pair-level and gene-level tables.
    """
    all_long_rows = []
    missing_files = []
    skipped_files = []

    for dataset in tqdm(datasets, desc=f"{condition_name}: datasets", leave=False):
        dataset_clean = clean_dataset_name(dataset)

        file_path = find_previous_pair_file(
            base_output_dir=base_output_dir,
            preprocess_short=condition_short,
            dataset_clean=dataset_clean,
            aliases=condition_aliases,
        )

        if file_path is None:
            missing_files.append(dataset_clean)
            continue

        try:
            df = pd.read_csv(file_path)
            long_df = standardize_pair_df(
                df=df,
                dataset_clean=dataset_clean,
                condition_short=condition_short,
            )
            all_long_rows.append(long_df)

        except Exception as e:
            skipped_files.append((dataset_clean, str(e)))
            continue

    if len(all_long_rows) == 0:
        raise RuntimeError(
            f"No usable pair files found for condition {condition_name} "
            f"using aliases {condition_aliases} in {base_output_dir}"
        )

    if missing_files:
        print(f"\n[{condition_name}] Missing files:")
        for m in missing_files:
            print("   ", m)

    if skipped_files:
        print(f"\n[{condition_name}] Skipped files:")
        for m, err in skipped_files:
            print(f"   {m}: {err}")

    pair_long_all = pd.concat(all_long_rows, axis=0, ignore_index=True)

    compiled_gene_rows = []

    for gene, gdf in tqdm(
        pair_long_all.groupby("gene", sort=False),
        desc=f"{condition_name}: genes",
        leave=False,
    ):
        ggi_corr = gdf.loc[gdf["group"].eq("GGI"), "corr"].to_numpy()
        rand_corr = gdf.loc[gdf["group"].eq("Random"), "corr"].to_numpy()

        if len(ggi_corr) == 0 or len(rand_corr) == 0:
            continue

        ggi_survival = np.mean(ggi_corr > threshold)
        random_survival = np.mean(rand_corr > threshold)

        compiled_gene_rows.append({
            "gene": gene,
            "condition": condition_short,
            "threshold": threshold,
            "n_datasets": gdf["dataset"].nunique(),
            "datasets": ";".join(sorted(gdf["dataset"].unique())),
            "n_ggi_pairs_total": len(ggi_corr),
            "n_random_pairs_total": len(rand_corr),
            "ggi_survival": ggi_survival,
            "random_survival": random_survival,
            "survival_delta": ggi_survival - random_survival,
            "survival_ratio": (ggi_survival + 1e-12) / (random_survival + 1e-12),
            "mean_ggi_corr": np.mean(ggi_corr),
            "mean_random_corr": np.mean(rand_corr),
            "median_ggi_corr": np.median(ggi_corr),
            "median_random_corr": np.median(rand_corr),
        })

    compiled_gene_df = pd.DataFrame(compiled_gene_rows)

    compiled_gene_df = compiled_gene_df.sort_values(
        ["survival_delta", "ggi_survival", "n_ggi_pairs_total"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    pair_out = os.path.join(
        OUTDIR,
        f"{condition_short}_compiled_gene_centered_pair_rows_t{threshold}.csv",
    )
    gene_out = os.path.join(
        OUTDIR,
        f"{condition_short}_compiled_gene_survival_t{threshold}.csv",
    )

    pair_long_all.to_csv(pair_out, index=False)
    compiled_gene_df.to_csv(gene_out, index=False)

    summary = {
        "condition": condition_short,
        "condition_name": condition_name,
        "n_pair_rows_gene_centered": len(pair_long_all),
        "n_genes": len(compiled_gene_df),
        "mean_ggi_survival": compiled_gene_df["ggi_survival"].mean(),
        "mean_random_survival": compiled_gene_df["random_survival"].mean(),
        "mean_survival_delta": compiled_gene_df["survival_delta"].mean(),
        "median_survival_delta": compiled_gene_df["survival_delta"].median(),
        "frac_ggi_above_random": np.mean(
            compiled_gene_df["ggi_survival"] > compiled_gene_df["random_survival"]
        ),
        "frac_both_zero": np.mean(
            (compiled_gene_df["ggi_survival"] == 0)
            & (compiled_gene_df["random_survival"] == 0)
        ),
        "pair_rows_csv": pair_out,
        "gene_survival_csv": gene_out,
    }

    return pair_long_all, compiled_gene_df, summary


def find_pair_files(base_output_dir, condition_key, aliases=None):
    """
    Finds pair CSVs produced by the GGI script.

    Compatible with current GGI script outputs like:

        ggi_vs_random_outputs/raw/DATASET__raw__ppi_vs_random_pairs.csv
        ggi_vs_random_outputs/log1p/DATASET__log1p__ppi_vs_random_pairs.csv
        ggi_vs_random_outputs/norm/DATASET__norm__ppi_vs_random_pairs.csv
        ggi_vs_random_outputs/norm_plust_log1p/DATASET__norm_plust_log1p__ppi_vs_random_pairs.csv

    Also supports future-renamed files like:

        DATASET__raw__ggi_vs_random_pairs.csv
    """
    if aliases is None:
        aliases = [condition_key]

    aliases = list(dict.fromkeys([condition_key] + list(aliases)))

    patterns = []

    for key in aliases:
        patterns.extend([
            # Actual/current GGI outputs with PPI-style file names
            os.path.join(base_output_dir, key, f"*__{key}__ppi_vs_random_pairs.csv"),
            os.path.join(base_output_dir, key, "*ppi_vs_random_pairs.csv"),
            os.path.join(base_output_dir, "**", f"*__{key}__ppi_vs_random_pairs.csv"),
            os.path.join(base_output_dir, "**", f"*{key}*ppi_vs_random_pairs.csv"),

            # Future fully renamed GGI outputs
            os.path.join(base_output_dir, key, f"*__{key}__ggi_vs_random_pairs.csv"),
            os.path.join(base_output_dir, key, "*ggi_vs_random_pairs.csv"),
            os.path.join(base_output_dir, "**", f"*__{key}__ggi_vs_random_pairs.csv"),
            os.path.join(base_output_dir, "**", f"*{key}*ggi_vs_random_pairs.csv"),
        ])

    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat, recursive=True))

    hits = sorted(set(hits))

    # Avoid accidentally loading summaries/errors if a broad pattern matched them.
    hits = [
        h for h in hits
        if (
            h.endswith("_ppi_vs_random_pairs.csv")
            or h.endswith("__ppi_vs_random_pairs.csv")
            or h.endswith("_ggi_vs_random_pairs.csv")
            or h.endswith("__ggi_vs_random_pairs.csv")
        )
    ]

    return hits


def infer_dataset_from_filename(fp, condition_key, aliases=None):
    """
    Infer dataset name from filenames like:
        DATASET__raw__ppi_vs_random_pairs.csv
        DATASET__norm_plust_log1p__ppi_vs_random_pairs.csv
        DATASET__raw__ggi_vs_random_pairs.csv
    """
    base = os.path.basename(fp)

    if aliases is None:
        aliases = [condition_key]

    aliases = list(dict.fromkeys([condition_key] + list(aliases)))

    for key in aliases:
        token = f"__{key}__"
        if token in base:
            return base.split(token)[0]

    base = base.replace("_ppi_vs_random_pairs.csv", "")
    base = base.replace("__ppi_vs_random_pairs.csv", "")
    base = base.replace("_ggi_vs_random_pairs.csv", "")
    base = base.replace("__ggi_vs_random_pairs.csv", "")
    return base


def load_condition_pairs(base_output_dir, condition_key, aliases=None):
    files = find_pair_files(
        base_output_dir=base_output_dir,
        condition_key=condition_key,
        aliases=aliases,
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No pair files found for condition '{condition_key}' under {base_output_dir}. "
            f"Tried aliases: {aliases}"
        )

    dfs = []

    print(f"Found {len(files):,} files for condition '{condition_key}'")
    for fp in files[:5]:
        print(f"  example: {fp}")
    if len(files) > 5:
        print(f"  ... plus {len(files) - 5:,} more")

    for fp in files:
        df = pd.read_csv(fp)

        required = ["type", "r"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[SKIP] {fp}: missing columns {missing}")
            continue

        tmp = df.copy()
        tmp["source_file"] = fp
        tmp["dataset"] = infer_dataset_from_filename(
            fp=fp,
            condition_key=condition_key,
            aliases=aliases,
        )
        tmp["condition"] = condition_key

        tmp["type_clean"] = tmp["type"].astype(str).str.strip().str.lower()
        tmp["r"] = pd.to_numeric(tmp["r"], errors="coerce")
        tmp = tmp.dropna(subset=["type_clean", "r"])

        # IMPORTANT:
        # In the GGI run, the true GGI rows may still be labeled "PPI".
        # Treat "PPI" as GGI here.
        ggi_type_labels = [
            "ppi",
            "ggi",
            "gi",
            "genetic_interaction",
            "genetic interaction",
            "genetic-interaction",
        ]

        ggi_mask = tmp["type_clean"].isin(ggi_type_labels)

        rand_mask = (
            tmp["type_clean"].eq("random")
            | tmp["type_clean"].eq("random_non_ppi")
            | tmp["type_clean"].eq("random_non_ggi")
            | tmp["type_clean"].str.startswith("random")
        )

        tmp = tmp[ggi_mask | rand_mask].copy()
        tmp["group"] = np.where(tmp["type_clean"].isin(ggi_type_labels), "GGI", "Random")

        dfs.append(tmp[["condition", "dataset", "source_file", "group", "r"]])

    if len(dfs) == 0:
        raise RuntimeError(f"No usable pair files for condition '{condition_key}'")

    out = pd.concat(dfs, axis=0, ignore_index=True)
    return out


def compute_logratio_curve(
    pair_df,
    t_grid,
    min_surviving_pairs=50,
    min_total_pairs=500,
    use_abs=False,
):
    vals = np.abs(pair_df["r"].to_numpy()) if use_abs else pair_df["r"].to_numpy()
    groups = pair_df["group"].to_numpy()

    ggi_vals = vals[groups == "GGI"]
    rand_vals = vals[groups == "Random"]

    n_ggi = len(ggi_vals)
    n_rand = len(rand_vals)

    if n_ggi < min_total_pairs or n_rand < min_total_pairs:
        raise RuntimeError(
            f"Not enough total pairs: n_ggi={n_ggi}, n_rand={n_rand}, "
            f"MIN_TOTAL_PAIRS={min_total_pairs}"
        )

    rows = []

    for t in t_grid:
        k_ggi = int(np.sum(ggi_vals > t))
        k_rand = int(np.sum(rand_vals > t))

        # Enforce decent tail support
        valid = (k_ggi >= min_surviving_pairs) and (k_rand >= min_surviving_pairs)

        if not valid:
            rows.append({
                "t": t,
                "valid": False,
                "n_ggi": n_ggi,
                "n_rand": n_rand,
                "k_ggi": k_ggi,
                "k_rand": k_rand,
                "p_ggi": np.nan,
                "p_rand": np.nan,
                "log_ratio": np.nan,
                "se": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
            })
            continue

        p_ggi = k_ggi / n_ggi
        p_rand = k_rand / n_rand

        log_ratio = np.log(p_ggi / p_rand)

        # Delta-method SE for log ratio of two survival probabilities
        # Var(log p_hat) ~= (1-p)/(n p) = 1/k - 1/n
        se = np.sqrt(
            (1.0 / k_ggi) - (1.0 / n_ggi) +
            (1.0 / k_rand) - (1.0 / n_rand)
        )

        ci_low = log_ratio - Z * se
        ci_high = log_ratio + Z * se

        rows.append({
            "t": t,
            "valid": True,
            "n_ggi": n_ggi,
            "n_rand": n_rand,
            "k_ggi": k_ggi,
            "k_rand": k_rand,
            "p_ggi": p_ggi,
            "p_rand": p_rand,
            "log_ratio": log_ratio,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    return pd.DataFrame(rows)


def find_pair_file(base_output_dir, condition_key, dataset_clean, aliases=None):
    """
    Finds GGI/random pair CSVs.

    Supports current files that may still use PPI-style names:

        ggi_vs_random_outputs/raw/DATASET__raw__ppi_vs_random_pairs.csv
        ggi_vs_random_outputs/log1p/DATASET__log1p__ppi_vs_random_pairs.csv
        ggi_vs_random_outputs/norm/DATASET__norm__ppi_vs_random_pairs.csv
        ggi_vs_random_outputs/norm_plust_log1p/DATASET__norm_plust_log1p__ppi_vs_random_pairs.csv

    Also supports future renamed files:

        DATASET__raw__ggi_vs_random_pairs.csv
    """
    if aliases is None:
        aliases = [condition_key]

    aliases = list(dict.fromkeys([condition_key] + list(aliases)))

    patterns = []

    for key in aliases:
        patterns.extend([
            # Current PPI-style GGI output filenames
            os.path.join(
                base_output_dir,
                key,
                f"{dataset_clean}__{key}__ppi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                key,
                f"{dataset_clean}*ppi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                "**",
                f"{dataset_clean}__{key}__ppi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                "**",
                f"{dataset_clean}*{key}*ppi_vs_random_pairs.csv",
            ),

            # Future GGI-style filenames
            os.path.join(
                base_output_dir,
                key,
                f"{dataset_clean}__{key}__ggi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                key,
                f"{dataset_clean}*ggi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                "**",
                f"{dataset_clean}__{key}__ggi_vs_random_pairs.csv",
            ),
            os.path.join(
                base_output_dir,
                "**",
                f"{dataset_clean}*{key}*ggi_vs_random_pairs.csv",
            ),
        ])

    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat, recursive=True))

    hits = sorted(set(hits))

    # Prefer hits that explicitly include one of the aliases
    preferred = []
    for h in hits:
        h_norm = h.replace("\\", "/")
        base = os.path.basename(h_norm)

        for key in aliases:
            if f"/{key}/" in h_norm or f"__{key}__" in base or key in base:
                preferred.append(h)
                break

    preferred = sorted(set(preferred))
    if len(preferred) > 0:
        return preferred[0]

    return hits[0] if len(hits) > 0 else None


def load_pair_file(fp):
    df = pd.read_csv(fp)

    required = ["type", "r"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fp} missing columns {missing}")

    df = df.copy()
    df["type_clean"] = df["type"].astype(str).str.strip().str.lower()
    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    df = df.dropna(subset=["type_clean", "r"])

    # IMPORTANT:
    # In your GGI run, true GGI rows may still be labeled as "PPI".
    # Treat "PPI" as GGI here.
    ggi_type_labels = [
        "ppi",
        "ggi",
        "gi",
        "genetic_interaction",
        "genetic interaction",
        "genetic-interaction",
    ]

    ggi_mask = df["type_clean"].isin(ggi_type_labels)

    rand_mask = (
        df["type_clean"].eq("random")
        | df["type_clean"].eq("random_non_ppi")
        | df["type_clean"].eq("random_non_ggi")
        | df["type_clean"].str.startswith("random")
    )

    df = df[ggi_mask | rand_mask].copy()
    df["group"] = np.where(df["type_clean"].isin(ggi_type_labels), "GGI", "Random")

    if USE_ABS:
        df["score"] = np.abs(df["r"].to_numpy())
    else:
        df["score"] = df["r"].to_numpy()

    return df


def format_pvalue(p):
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.3g}"


def paired_ttest_pvalue(random_vals, ggi_vals):
    random_vals = np.asarray(random_vals, dtype=float)
    ggi_vals = np.asarray(ggi_vals, dtype=float)

    mask = np.isfinite(random_vals) & np.isfinite(ggi_vals)
    random_vals = random_vals[mask]
    ggi_vals = ggi_vals[mask]

    if len(random_vals) < 2:
        return np.nan

    if ttest_rel is None:
        return np.nan

    return float(ttest_rel(ggi_vals, random_vals).pvalue)


def paired_wilcoxon_pvalue(random_vals, ggi_vals):
    random_vals = np.asarray(random_vals, dtype=float)
    ggi_vals = np.asarray(ggi_vals, dtype=float)

    mask = np.isfinite(random_vals) & np.isfinite(ggi_vals)
    random_vals = random_vals[mask]
    ggi_vals = ggi_vals[mask]

    if len(random_vals) < 2:
        return np.nan

    if wilcoxon is None:
        return np.nan

    try:
        return float(wilcoxon(ggi_vals, random_vals).pvalue)
    except ValueError:
        return np.nan
