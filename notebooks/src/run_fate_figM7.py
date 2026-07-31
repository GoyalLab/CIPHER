"""Run module for notebooks/suppl/fate_commitment_figM7.ipynb (Fig M7).

Main-flow orchestration for the LARRY prospective fate-commitment supplement. Each
function is one section/figure block from a notebook cell; the thinned notebook drives
them as ``R.<section>()`` calls.

Notebook config (DATA_DIR / SUPPL / OUT_BASE) is injected from the notebook via
``R.__dict__.update(...)`` and resolved as MODULE GLOBALS. Cross-section state
(everything a later section reads from an earlier one) is likewise carried as
module globals: each function declares its cell's top-level assignments ``global``
so they accumulate in this module's namespace, exactly as they did in the original
single-namespace notebook.

Helpers in notebooks/src (not part of the cipher package).
"""
# --- library imports mirroring notebooks/src/suppl_fate.py (resolved at call time) ---
import os, re, glob, json, math, warnings, sys
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
    import seaborn as sns
except Exception:
    sns = None
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
    def tqdm(x, *a, **k):
        return x
# --- end library imports ---

# self-contained helpers in notebooks/src (not part of the cipher package)
from src.suppl_fate import *



def cipher_larry_prospective_prediction():
    global os, gzip, warnings, np, pd, plt, sns, mmread, \
        issparse, StratifiedKFold, LogisticRegression, roc_auc_score, average_precision_score, confusion_matrix, PCA, OUTDIR, \
        COUNTS_PATH, GENES_PATH, CLONE_PATH, META_PATH, TIME_COL, CELLTYPE_COL, WELL_COL, START_COL, \
        EARLY_TIME, EARLY_CELLTYPE, EARLY_WELL, TERMINAL_TIME, TERMINAL_WELL, EXCLUDE_FATES, MIN_TOTAL_CELLS_PER_CLONE, MIN_EARLY_CELLS_PER_CLONE, \
        MIN_TERMINAL_CELLS_PER_CLONE, MIN_DOMINANT_FATE_COUNT, MIN_DOMINANT_FATE_FRAC, MAX_FATE_ENTROPY, MIN_CLONES_PER_FATE, MAX_FATES, N_VAR_GENES, MAX_COV_CELLS, \
        RIDGE, COV_SHRINK_TO_DIAG, MODELS, MAIN_MODEL, N_SPLITS, SEED, rng, safe_name, \
        mask_early_cells, mask_terminal_cells, get_cell_to_clone, get_cells_x_genes, zscore_train, fate_entropy_from_counts, select_hvgs_sparse, make_covariance, \
        clone_mean_matrix, fit_calibrator, compute_metrics, counts, f, gene_names, clone_mat, meta, \
        cell_to_clone, has_clone, fate_labels, early_all_mask, early_cloned_mask, terminal_cloned_mask, early_all_idx, early_cloned_idx, \
        terminal_cloned_idx, clone_records, clone_id, cells, early_cells, terminal_cells, terminal_fates, vc, \
        dominant_fate, dominant_count, total_terminal, dominant_frac, rec, fate, count, \
        s, clone_table_all, fate_counts, selected_fates, clone_table, eligible_clones, eligible_early_mask, eligible_early_idx, \
        clone_to_fate, clone_to_frac, clone_to_n_total, clone_to_n_early, clone_to_n_terminal, fig, axes, hvg_idx, \
        gene_vars, hvg_genes, cov_idx, Xcov_raw, mu_ref, sd_ref, Xcov, Sigma, \
        evals, evecs, diag, X_clones, y_clones, min_class_n, n_splits, splitter, \
        all_cell_rows, all_clone_rows, force_rows, fold, train_pos, test_pos, train_clones, test_clones, \
        Xtrain_clone, train_clone_ids_used, n_train_early, y_train, y_train_shuffled, force_info, pos, neg, \
        delta, pos_shuf, neg_shuf, delta_shuf, U_dict, y_binary, model_name, u, \
        train_scores, smu, ssd, train_scores_scaled, calibrator, key, Xtest_clone, test_clone_ids_used, \
        n_test_early, y_test_clone, base_clone, test_early_idx, Xtest_cell_raw, Xtest_cell, test_cell_clone_ids, y_test_cell, \
        base_cell, U, calibrators, level_name, Xscore, base_df, collector, raw_scores, \
        scaled_scores, p_ovr, j, clf, p_norm, pred_idx, pred_fates, rows, \
        DELTAS, top_pos, top_neg, rank, gi, early_cell_probs, clone_probs, force_df, \
        metric_rows, df_clone, df_cell, m_clone, m_cell, metrics, summary_metrics, acc_rows, \
        df, sub, acc_df, clone_auc, pivot_auc, base, metric_name, delta_cols, \
        long_delta, main_clone, p_norm_cols, mean_prob, cm, cm_norm, score_rows, col, \
        tmp, score_df, cipher_force, mean_force, top_genes, TOP_GENES_PER_FATE, heat, MAX_PLOT_CELLS, \
        main_cell, plot_df, plot_cells, X_plot, Z, sc
    # ============================================================
    # CIPHER-LARRY: prospective early-cell fate prediction
    # ============================================================
    # This version is designed for the fact that LARRY has few early cells per clone.
    #
    # Main fixes:
    #   1. Do NOT require many early cells per clone.
    #      Early day-2 cells are the prospective measurement, so one early cell is OK.
    #
    #   2. Require reliable terminal fate labels:
    #      enough terminal cells, enough dominant-fate cells, high dominant-fate fraction.
    #
    #   3. Estimate Sigma from ALL early undifferentiated cells, not only fate-labeled clones.
    #      CIPHER covariance does not need clone labels.
    #
    #   4. Define fate vectors using clone-balanced early means:
    #        Delta_f = mean_clone(X_early | clone later f)
    #                  - mean_clone(X_early | clone later not f)
    #
    #   5. Compare:
    #        CIPHER  = Sigma^{-1} Delta_f
    #        diag    = diag(Sigma)^{-1} Delta_f
    #        direct  = Delta_f
    #        shuffled-label null
    #
    #   6. Proper held-out clone-level CV. No in-sample fallback.
    #
    # ============================================================

    import os, gzip, warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.io import mmread
    from scipy.sparse import issparse
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    from sklearn.decomposition import PCA

    warnings.filterwarnings("ignore")

    # ============================================================
    # CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_prospective_fate_final")
    os.makedirs(OUTDIR, exist_ok=True)

    COUNTS_PATH = os.path.join(SUPPL, "stateFate_inVitro_normed_counts.mtx.gz")
    GENES_PATH  = os.path.join(SUPPL, "stateFate_inVitro_gene_names.txt.gz")
    CLONE_PATH  = os.path.join(SUPPL, "stateFate_inVitro_clone_matrix.mtx.gz")
    META_PATH   = os.path.join(SUPPL, "stateFate_inVitro_metadata.txt.gz")

    TIME_COL = "Time point"
    CELLTYPE_COL = "Cell type annotation"
    WELL_COL = "Well"
    START_COL = "Starting population"

    # Early cells used for prediction and covariance.
    EARLY_TIME = 2.0
    EARLY_CELLTYPE = "Undifferentiated"

    # Important:
    #   None uses all wells and gives more early cells for Sigma.
    #   0 reproduces your earlier day-2/well-0 setup.
    EARLY_WELL = None

    # Terminal fate assignment.
    TERMINAL_TIME = 6.0
    TERMINAL_WELL = None

    EXCLUDE_FATES = {
        "Undifferentiated", "Unknown", "unknown", "nan", "NaN",
        "Ambiguous", "ambiguous", "None", ""
    }

    # Clone fate-label QC.
    # Keep MIN_EARLY_CELLS_PER_CLONE low. This is intentional.
    MIN_TOTAL_CELLS_PER_CLONE = 8
    MIN_EARLY_CELLS_PER_CLONE = 1
    MIN_TERMINAL_CELLS_PER_CLONE = 5
    MIN_DOMINANT_FATE_COUNT = 4
    MIN_DOMINANT_FATE_FRAC = 0.80
    MAX_FATE_ENTROPY = 0.75

    # Fate selection.
    MIN_CLONES_PER_FATE = 15
    MAX_FATES = 5

    # Gene/covariance settings.
    N_VAR_GENES = 1000
    MAX_COV_CELLS = 50000

    # CIPHER regularization.
    RIDGE = 1e-8
    COV_SHRINK_TO_DIAG = 0.0

    # Models.
    MODELS = ["cipher", "diag","direct", "shuffled"]
    MAIN_MODEL = "cipher"

    # CV.
    N_SPLITS = 5
    SEED = 0
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    # Plots.
    plt.rcParams.update({"font.size": 14})
    sns.set_context("talk")

    # ============================================================
    # HELPERS
    # ============================================================


    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )


    def mask_early_cells(meta):
        m = meta[TIME_COL].astype(float).values == float(EARLY_TIME)
        if EARLY_CELLTYPE is not None:
            m &= meta[CELLTYPE_COL].astype(str).values == str(EARLY_CELLTYPE)
        if EARLY_WELL is not None and WELL_COL in meta.columns:
            m &= meta[WELL_COL].astype(float).values == float(EARLY_WELL)
        return m

    def mask_terminal_cells(meta):
        m = meta[TIME_COL].astype(float).values == float(TERMINAL_TIME)
        if TERMINAL_WELL is not None and WELL_COL in meta.columns:
            m &= meta[WELL_COL].astype(float).values == float(TERMINAL_WELL)
        ann = meta[CELLTYPE_COL].astype(str).values
        m &= ~np.isin(ann, list(EXCLUDE_FATES))
        return m

    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()
        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row
        return cell_to_clone

    def get_cells_x_genes(counts, cell_idx, gene_idx):
        # counts is genes x cells
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)

    def zscore_train(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu, sd


    def fate_entropy_from_counts(counts_vec):
        counts_vec = np.asarray(counts_vec, dtype=float)
        counts_vec = counts_vec[counts_vec > 0]
        if len(counts_vec) <= 1:
            return 0.0
        p = counts_vec / counts_vec.sum()
        return float(-(p * np.log(p)).sum())

    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        X = counts[:, cell_idx]
        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means**2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]
        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_var_genes:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_

    def make_covariance(X):
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D
        Sigma = Sigma + RIDGE * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)

    def clone_mean_matrix(clone_ids, early_mask, cell_to_clone, counts, hvg_idx, mu, sd):
        rows = []
        out_ids = []
        out_n = []

        for cid in clone_ids:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]
            if len(idx) == 0:
                continue

            X = get_cells_x_genes(counts, idx, hvg_idx)
            X = apply_zscore(X, mu, sd)

            rows.append(X.mean(axis=0))
            out_ids.append(cid)
            out_n.append(len(idx))

        if len(rows) == 0:
            return np.empty((0, len(hvg_idx))), np.array([], dtype=int), np.array([], dtype=int)

        return np.vstack(rows), np.asarray(out_ids, dtype=int), np.asarray(out_n, dtype=int)

    def fit_calibrator(scores, y):
        scores = np.asarray(scores).reshape(-1, 1)
        y = np.asarray(y).astype(int)

        if len(np.unique(y)) < 2:
            return None

        clf = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        )
        clf.fit(scores, y)
        return clf


    def compute_metrics(df, selected_fates, prob_prefix="p_ovr", label_col="true_future_lineage"):
        rows = []

        for fate in selected_fates:
            col = f"{prob_prefix}__{safe_name(fate)}"

            y = (df[label_col].astype(str).values == str(fate)).astype(int)
            s = df[col].values.astype(float)

            if len(np.unique(y)) < 2:
                auroc = np.nan
                auprc = np.nan
            else:
                auroc = roc_auc_score(y, s)
                auprc = average_precision_score(y, s)

            baseline = y.mean()
            cutoff = np.quantile(s, 0.90)
            top = s >= cutoff

            if top.sum() > 0 and baseline > 0:
                top_rate = y[top].mean()
                enrichment = top_rate / baseline
            else:
                top_rate = np.nan
                enrichment = np.nan

            rows.append({
                "fate": fate,
                "n": len(y),
                "n_positive": int(y.sum()),
                "positive_fraction": float(baseline),
                "AUROC": auroc,
                "AUPRC": auprc,
                "top_decile_positive_rate": top_rate,
                "top_decile_enrichment": enrichment,
            })

        return pd.DataFrame(rows)

    # ============================================================
    # LOAD DATA
    # ============================================================

    counts = mmread(COUNTS_PATH).T.tocsr()  # genes x cells
    print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} cells | nnz={counts.nnz:,}")

    with gzip.open(GENES_PATH, "rt") as f:
        gene_names = np.array([line.strip() for line in f])
    print(f"Genes loaded: {len(gene_names)}")

    clone_mat = mmread(CLONE_PATH).T.tocsr()  # clones x cells
    print(f"Clone matrix: {clone_mat.shape[0]} clones x {clone_mat.shape[1]} cells")
    print(f"% cells with clone label: {(clone_mat.sum(axis=0) > 0).mean() * 100:.2f}%")

    meta = pd.read_csv(META_PATH, sep="\t")
    print(f"Meta: {meta.shape[0]} rows x {meta.shape[1]} cols")
    print("Meta columns:", list(meta.columns))

    assert counts.shape[1] == meta.shape[0] == clone_mat.shape[1], "cells mismatch"
    assert counts.shape[0] == len(gene_names), "genes mismatch"

    meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")

    print("\nTimepoints:")
    print(np.sort(meta[TIME_COL].dropna().unique()))

    print("\nCell annotations:")
    print(meta[CELLTYPE_COL].value_counts())

    cell_to_clone = get_cell_to_clone(clone_mat)
    has_clone = cell_to_clone >= 0
    fate_labels = meta[CELLTYPE_COL].astype(str).values

    # ============================================================
    # DEFINE EARLY AND TERMINAL CELLS
    # ============================================================

    early_all_mask = mask_early_cells(meta)
    early_cloned_mask = early_all_mask & has_clone
    terminal_cloned_mask = mask_terminal_cells(meta) & has_clone

    early_all_idx = np.where(early_all_mask)[0]
    early_cloned_idx = np.where(early_cloned_mask)[0]
    terminal_cloned_idx = np.where(terminal_cloned_mask)[0]

    print(f"\nAll early/precommitted cells for covariance: {len(early_all_idx):,}")
    print(f"Cloned early/precommitted cells: {len(early_cloned_idx):,}")
    print(f"Cloned terminal cells: {len(terminal_cloned_idx):,}")

    if len(early_all_idx) == 0:
        raise RuntimeError("No early cells found. Check EARLY_TIME / EARLY_CELLTYPE / EARLY_WELL.")

    if len(terminal_cloned_idx) == 0:
        raise RuntimeError("No terminal cloned cells found. Check TERMINAL_TIME / TERMINAL_WELL.")

    # ============================================================
    # BUILD STRICT TERMINAL FATE LABELS
    # ============================================================

    clone_records = []

    for clone_id in range(clone_mat.shape[0]):
        cells = clone_mat[clone_id].indices

        if len(cells) < MIN_TOTAL_CELLS_PER_CLONE:
            continue

        early_cells = cells[early_cloned_mask[cells]]
        terminal_cells = cells[terminal_cloned_mask[cells]]

        if len(early_cells) < MIN_EARLY_CELLS_PER_CLONE:
            continue

        if len(terminal_cells) < MIN_TERMINAL_CELLS_PER_CLONE:
            continue

        terminal_fates = pd.Series(fate_labels[terminal_cells].astype(str))
        terminal_fates = terminal_fates[~terminal_fates.isin(EXCLUDE_FATES)]

        if len(terminal_fates) < MIN_TERMINAL_CELLS_PER_CLONE:
            continue

        vc = terminal_fates.value_counts()
        if len(vc) == 0:
            continue

        dominant_fate = str(vc.index[0])
        dominant_count = int(vc.iloc[0])
        total_terminal = int(vc.sum())
        dominant_frac = dominant_count / max(total_terminal, 1)
        entropy = fate_entropy_from_counts(vc.values)

        if dominant_count < MIN_DOMINANT_FATE_COUNT:
            continue
        if dominant_frac < MIN_DOMINANT_FATE_FRAC:
            continue
        if MAX_FATE_ENTROPY is not None and entropy > MAX_FATE_ENTROPY:
            continue

        rec = {
            "clone_id": int(clone_id),
            "n_total_clone_cells": int(len(cells)),
            "n_early": int(len(early_cells)),
            "n_terminal": int(total_terminal),
            "n_terminal_raw": int(len(terminal_cells)),
            "n_terminal_fate_types": int(len(vc)),
            "dominant_fate": dominant_fate,
            "dominant_count": dominant_count,
            "dominant_frac": float(dominant_frac),
            "fate_entropy": float(entropy),
        }

        for fate, count in vc.items():
            s = safe_name(fate)
            rec[f"terminal_count__{s}"] = int(count)
            rec[f"terminal_frac__{s}"] = float(count / total_terminal)

        clone_records.append(rec)

    clone_table_all = pd.DataFrame(clone_records)

    if clone_table_all.empty:
        raise RuntimeError("No clones passed terminal fate QC. Relax terminal clone filters.")

    print("\nClones passing terminal fate QC:")
    print(f"n={len(clone_table_all):,}")
    print(clone_table_all["dominant_fate"].value_counts())

    fate_counts = clone_table_all["dominant_fate"].value_counts()
    selected_fates = fate_counts[fate_counts >= MIN_CLONES_PER_FATE].index.tolist()
    selected_fates = selected_fates[:MAX_FATES]

    if len(selected_fates) < 2:
        raise RuntimeError(
            "Fewer than two fates have enough clones. "
            "Lower MIN_CLONES_PER_FATE or MAX_FATES, but do not allow single-clone fates."
        )

    clone_table = clone_table_all[clone_table_all["dominant_fate"].isin(selected_fates)].copy()
    eligible_clones = clone_table["clone_id"].values.astype(int)

    eligible_early_mask = early_cloned_mask & np.isin(cell_to_clone, eligible_clones)
    eligible_early_idx = np.where(eligible_early_mask)[0]

    clone_to_fate = dict(zip(clone_table["clone_id"], clone_table["dominant_fate"]))
    clone_to_frac = dict(zip(clone_table["clone_id"], clone_table["dominant_frac"]))
    clone_to_n_total = dict(zip(clone_table["clone_id"], clone_table["n_total_clone_cells"]))
    clone_to_n_early = dict(zip(clone_table["clone_id"], clone_table["n_early"]))
    clone_to_n_terminal = dict(zip(clone_table["clone_id"], clone_table["n_terminal"]))

    print("\nSelected fates:")
    print(clone_table["dominant_fate"].value_counts())

    print(f"\nEligible clones: {len(eligible_clones):,}")
    print(f"Eligible early cells: {len(eligible_early_idx):,}")

    clone_table_all.to_csv(os.path.join(OUTDIR, "clone_table_all_terminal_qc.csv"), index=False)
    clone_table.to_csv(os.path.join(OUTDIR, "clone_table_selected_fates.csv"), index=False)

    # ============================================================
    # PLOT CLONE QC
    # ============================================================

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    sns.countplot(data=clone_table, x="dominant_fate", order=selected_fates, ax=axes[0, 0])
    axes[0, 0].set_title("High-confidence clones per future fate")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].set_xlabel("future fate")
    axes[0, 0].set_ylabel("clone count")

    sns.histplot(data=clone_table, x="n_early", bins=30, ax=axes[0, 1])
    axes[0, 1].set_title("Early cells per retained clone")
    axes[0, 1].set_xlabel("early cells per clone")

    sns.histplot(data=clone_table, x="n_terminal", bins=40, ax=axes[0, 2])
    axes[0, 2].set_title("Terminal cells per retained clone")
    axes[0, 2].set_xlabel("terminal cells per clone")

    sns.histplot(data=clone_table, x="dominant_frac", bins=25, ax=axes[1, 0])
    axes[1, 0].set_title("Dominant fate purity")
    axes[1, 0].set_xlabel("dominant fate fraction")

    sns.scatterplot(
        data=clone_table,
        x="n_terminal",
        y="dominant_frac",
        hue="dominant_fate",
        hue_order=selected_fates,
        ax=axes[1, 1],
        s=45,
    )
    axes[1, 1].set_title("Purity vs terminal clone size")
    axes[1, 1].set_xlabel("terminal cells")
    axes[1, 1].set_ylabel("dominant fate fraction")
    axes[1, 1].legend(fontsize=9, frameon=False)

    sns.scatterplot(
        data=clone_table,
        x="n_early",
        y="n_terminal",
        hue="dominant_fate",
        hue_order=selected_fates,
        ax=axes[1, 2],
        s=45,
    )
    axes[1, 2].set_title("Early vs terminal representation")
    axes[1, 2].set_xlabel("early cells")
    axes[1, 2].set_ylabel("terminal cells")
    axes[1, 2].legend(fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "clone_qc_summary.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "clone_qc_summary.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # SELECT HVGs AND BUILD GLOBAL EARLY COVARIANCE
    # ============================================================

    print("\nSelecting HVGs from all early/precommitted cells...")

    hvg_idx, gene_vars = select_hvgs_sparse(
        counts=counts,
        cell_idx=early_all_idx,
        n_var_genes=N_VAR_GENES,
    )

    hvg_genes = gene_names[hvg_idx]

    pd.DataFrame({
        "gene": hvg_genes,
        "gene_index": hvg_idx,
        "early_variance": gene_vars[hvg_idx],
    }).to_csv(os.path.join(OUTDIR, "selected_early_hvgs.csv"), index=False)

    print(f"Using top {len(hvg_idx)} early-variable genes.")

    # Covariance cells can include uncloned early cells.
    cov_idx = early_all_idx.copy()
    if len(cov_idx) > MAX_COV_CELLS:
        cov_idx = rng.choice(cov_idx, size=MAX_COV_CELLS, replace=False)

    print(f"Using {len(cov_idx):,} early cells for Sigma.")

    Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx)
    mu_ref, sd_ref = zscore_train(Xcov_raw)
    Xcov = apply_zscore(Xcov_raw, mu_ref, sd_ref)

    Sigma = make_covariance(Xcov)

    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-8)

    diag = np.diag(Sigma).copy()
    diag[diag < 1e-8] = 1e-8

    pd.DataFrame({
        "eigenvalue": evals[::-1],
        "rank": np.arange(1, len(evals) + 1),
    }).to_csv(os.path.join(OUTDIR, "early_covariance_eigenvalues.csv"), index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(evals) + 1), evals[::-1], marker="o", linewidth=1, markersize=3)
    plt.yscale("log")
    plt.xlabel("eigenvalue rank")
    plt.ylabel("eigenvalue")
    plt.title("Early progenitor covariance spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # CROSS-VALIDATED FATE PREDICTION
    # ============================================================

    X_clones = clone_table["clone_id"].values.astype(int)
    y_clones = clone_table["dominant_fate"].values.astype(str)

    min_class_n = clone_table["dominant_fate"].value_counts().min()
    n_splits = min(N_SPLITS, int(min_class_n))

    if n_splits < 2:
        raise RuntimeError(
            f"Cannot do held-out CV. Smallest selected fate has only {min_class_n} clones."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    print(f"\nUsing clone-level stratified {n_splits}-fold CV.")

    all_cell_rows = []
    all_clone_rows = []
    force_rows = []

    for fold, (train_pos, test_pos) in enumerate(splitter.split(X_clones, y_clones)):
        train_clones = X_clones[train_pos]
        test_clones = X_clones[test_pos]

        print(f"\nFold {fold + 1}/{n_splits}: train clones={len(train_clones)}, test clones={len(test_clones)}")

        Xtrain_clone, train_clone_ids_used, n_train_early = clone_mean_matrix(
            clone_ids=train_clones,
            early_mask=eligible_early_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        y_train = np.array([clone_to_fate[c] for c in train_clone_ids_used])

        if len(train_clone_ids_used) < 5:
            raise RuntimeError("Too few train clone means.")

        y_train_shuffled = y_train.copy()
        rng.shuffle(y_train_shuffled)

        force_info = {
            model: {
                "U": [],
                "DELTAS": [],
                "calibrators": [],
                "score_mu": [],
                "score_sd": [],
            }
            for model in MODELS
        }

        for fate in selected_fates:
            pos = y_train == fate
            neg = y_train != fate

            if pos.sum() == 0 or neg.sum() == 0:
                raise RuntimeError(f"Missing train positives/negatives for {fate}")

            delta = Xtrain_clone[pos].mean(axis=0) - Xtrain_clone[neg].mean(axis=0)

            pos_shuf = y_train_shuffled == fate
            neg_shuf = y_train_shuffled != fate

            if pos_shuf.sum() == 0 or neg_shuf.sum() == 0:
                delta_shuf = rng.normal(size=delta.shape)
                delta_shuf = delta_shuf / (np.linalg.norm(delta_shuf) + 1e-8) * (np.linalg.norm(delta) + 1e-8)
            else:
                delta_shuf = Xtrain_clone[pos_shuf].mean(axis=0) - Xtrain_clone[neg_shuf].mean(axis=0)

            U_dict = {
                "cipher": evecs @ ((evecs.T @ delta) / evals),
                "diag": delta / diag,
                "direct": delta.copy(),
                "shuffled": evecs @ ((evecs.T @ delta_shuf) / evals),
            }

            y_binary = (y_train == fate).astype(int)

            for model_name in MODELS:
                u = U_dict[model_name]
                train_scores = Xtrain_clone @ u

                smu = train_scores.mean()
                ssd = train_scores.std()
                if ssd < 1e-6:
                    ssd = 1.0

                train_scores_scaled = (train_scores - smu) / ssd
                calibrator = fit_calibrator(train_scores_scaled, y_binary)

                force_info[model_name]["U"].append(u)
                force_info[model_name]["DELTAS"].append(delta if model_name != "shuffled" else delta_shuf)
                force_info[model_name]["calibrators"].append(calibrator)
                force_info[model_name]["score_mu"].append(smu)
                force_info[model_name]["score_sd"].append(ssd)

        for model_name in MODELS:
            for key in ["U", "DELTAS", "score_mu", "score_sd"]:
                force_info[model_name][key] = np.asarray(force_info[model_name][key])

        # ----------------------
        # Score held-out clones.
        # ----------------------
        Xtest_clone, test_clone_ids_used, n_test_early = clone_mean_matrix(
            clone_ids=test_clones,
            early_mask=eligible_early_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        y_test_clone = np.array([clone_to_fate[c] for c in test_clone_ids_used])

        base_clone = pd.DataFrame({
            "fold": fold,
            "level": "clone",
            "clone_id": test_clone_ids_used,
            "true_future_lineage": y_test_clone,
            "true_future_lineage_frac": [clone_to_frac[c] for c in test_clone_ids_used],
            "n_early_scored": n_test_early,
            "n_total_clone_cells": [clone_to_n_total[c] for c in test_clone_ids_used],
            "n_early_clone_cells": [clone_to_n_early[c] for c in test_clone_ids_used],
            "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_clone_ids_used],
        })

        # ----------------------
        # Score held-out early cells.
        # ----------------------
        test_early_idx = np.where(eligible_early_mask & np.isin(cell_to_clone, test_clones))[0]
        Xtest_cell_raw = get_cells_x_genes(counts, test_early_idx, hvg_idx)
        Xtest_cell = apply_zscore(Xtest_cell_raw, mu_ref, sd_ref)

        test_cell_clone_ids = cell_to_clone[test_early_idx]
        y_test_cell = np.array([clone_to_fate[c] for c in test_cell_clone_ids])

        base_cell = pd.DataFrame({
            "fold": fold,
            "level": "cell",
            "cell_index": test_early_idx,
            "clone_id": test_cell_clone_ids,
            "true_future_lineage": y_test_cell,
            "true_future_lineage_frac": [clone_to_frac[c] for c in test_cell_clone_ids],
            "n_total_clone_cells": [clone_to_n_total[c] for c in test_cell_clone_ids],
            "n_early_clone_cells": [clone_to_n_early[c] for c in test_cell_clone_ids],
            "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_cell_clone_ids],
        })

        for model_name in MODELS:
            U = force_info[model_name]["U"]
            smu = force_info[model_name]["score_mu"]
            ssd = force_info[model_name]["score_sd"]
            calibrators = force_info[model_name]["calibrators"]

            for level_name, Xscore, base_df, collector in [
                ("clone", Xtest_clone, base_clone, all_clone_rows),
                ("cell", Xtest_cell, base_cell, all_cell_rows),
            ]:
                raw_scores = Xscore @ U.T
                scaled_scores = (raw_scores - smu[None, :]) / ssd[None, :]

                p_ovr = np.zeros_like(scaled_scores, dtype=float)

                for j, clf in enumerate(calibrators):
                    p_ovr[:, j] = predict_calibrator(clf, scaled_scores[:, j])

                p_ovr = np.nan_to_num(p_ovr, nan=0.0, posinf=1.0, neginf=0.0)
                p_ovr = np.clip(p_ovr, 1e-6, 1.0 - 1e-6)
                p_norm = soft_normalize(p_ovr)

                pred_idx = np.argmax(p_norm, axis=1)
                pred_fates = np.array(selected_fates, dtype=object)[pred_idx]

                rows = base_df.copy()
                rows["model"] = model_name
                rows["predicted_lineage_norm"] = pred_fates
                rows["max_pseudoprob_norm"] = p_norm.max(axis=1)

                for j, fate in enumerate(selected_fates):
                    s = safe_name(fate)
                    rows[f"score_raw__{s}"] = raw_scores[:, j]
                    rows[f"score_scaled__{s}"] = scaled_scores[:, j]
                    rows[f"p_ovr__{s}"] = p_ovr[:, j]
                    rows[f"p_norm__{s}"] = p_norm[:, j]

                rows["p_ovr_true_future_lineage"] = [
                    p_ovr[i, selected_fates.index(tf)]
                    for i, tf in enumerate(rows["true_future_lineage"].values)
                ]

                rows["p_norm_true_future_lineage"] = [
                    p_norm[i, selected_fates.index(tf)]
                    for i, tf in enumerate(rows["true_future_lineage"].values)
                ]

                collector.append(rows)

        # Save top force genes.
        for model_name in MODELS:
            U = force_info[model_name]["U"]
            DELTAS = force_info[model_name]["DELTAS"]

            for j, fate in enumerate(selected_fates):
                u = U[j]
                delta = DELTAS[j]

                top_pos = np.argsort(u)[::-1][:50]
                top_neg = np.argsort(u)[:50]

                for rank, gi in enumerate(top_pos, start=1):
                    force_rows.append({
                        "fold": fold,
                        "model": model_name,
                        "fate": fate,
                        "direction": "positive",
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),
                        "u": float(u[gi]),
                        "delta_early": float(delta[gi]),
                    })

                for rank, gi in enumerate(top_neg, start=1):
                    force_rows.append({
                        "fold": fold,
                        "model": model_name,
                        "fate": fate,
                        "direction": "negative",
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),
                        "u": float(u[gi]),
                        "delta_early": float(delta[gi]),
                    })

    early_cell_probs = pd.concat(all_cell_rows, ignore_index=True)
    clone_probs = pd.concat(all_clone_rows, ignore_index=True)
    force_df = pd.DataFrame(force_rows)

    early_cell_probs.to_csv(os.path.join(OUTDIR, "early_cell_probs_all_models.csv"), index=False)
    clone_probs.to_csv(os.path.join(OUTDIR, "clone_probs_all_models.csv"), index=False)
    force_df.to_csv(os.path.join(OUTDIR, "top_force_genes_all_models.csv"), index=False)

    print("\nSaved core outputs:")
    print(os.path.join(OUTDIR, "early_cell_probs_all_models.csv"))
    print(os.path.join(OUTDIR, "clone_probs_all_models.csv"))
    print(os.path.join(OUTDIR, "top_force_genes_all_models.csv"))

    # ============================================================
    # METRICS
    # ============================================================

    metric_rows = []

    for model_name in MODELS:
        for fold in sorted(clone_probs["fold"].unique()):
            df_clone = clone_probs[(clone_probs["model"] == model_name) & (clone_probs["fold"] == fold)]
            df_cell = early_cell_probs[(early_cell_probs["model"] == model_name) & (early_cell_probs["fold"] == fold)]

            m_clone = compute_metrics(df_clone, selected_fates)
            m_clone["model"] = model_name
            m_clone["fold"] = fold
            m_clone["level"] = "clone"

            m_cell = compute_metrics(df_cell, selected_fates)
            m_cell["model"] = model_name
            m_cell["fold"] = fold
            m_cell["level"] = "cell"

            metric_rows.append(m_clone)
            metric_rows.append(m_cell)

    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics.to_csv(os.path.join(OUTDIR, "prediction_metrics_by_fold.csv"), index=False)

    summary_metrics = (
        metrics
        .groupby(["level", "model", "fate"], as_index=False)
        .agg(
            AUROC_mean=("AUROC", "mean"),
            AUROC_sd=("AUROC", "std"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_sd=("AUPRC", "std"),
            top_decile_enrichment_mean=("top_decile_enrichment", "mean"),
            top_decile_enrichment_sd=("top_decile_enrichment", "std"),
            n_positive_mean=("n_positive", "mean"),
            positive_fraction_mean=("positive_fraction", "mean"),
        )
    )

    summary_metrics.to_csv(os.path.join(OUTDIR, "prediction_metrics_summary.csv"), index=False)

    acc_rows = []

    for model_name in MODELS:
        for level_name, df in [("clone", clone_probs), ("cell", early_cell_probs)]:
            sub = df[df["model"] == model_name]
            acc_rows.append({
                "model": model_name,
                "level": level_name,
                "argmax_accuracy": np.mean(sub["predicted_lineage_norm"] == sub["true_future_lineage"]),
                "mean_p_true_ovr": sub["p_ovr_true_future_lineage"].mean(),
                "mean_p_true_norm": sub["p_norm_true_future_lineage"].mean(),
            })

    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(os.path.join(OUTDIR, "argmax_accuracy_summary.csv"), index=False)

    clone_auc = metrics[metrics["level"] == "clone"].copy()

    pivot_auc = clone_auc.pivot_table(
        index=["fold", "fate"],
        columns="model",
        values="AUROC",
    ).reset_index()

    for base in ["direct", "diag", "shuffled"]:
        pivot_auc[f"cipher_minus_{base}"] = pivot_auc["cipher"] - pivot_auc[base]

    pivot_auc.to_csv(os.path.join(OUTDIR, "paired_clone_AUROC_deltas.csv"), index=False)

    print("\nClone-level CIPHER summary:")
    print(
        summary_metrics[
            (summary_metrics["level"] == "clone") &
            (summary_metrics["model"] == MAIN_MODEL)
        ][[
            "fate",
            "n_positive_mean",
            "positive_fraction_mean",
            "AUROC_mean",
            "AUROC_sd",
            "AUPRC_mean",
            "AUPRC_sd",
            "top_decile_enrichment_mean",
        ]]
    )

    print("\nArgmax summary:")
    print(acc_df)

    # ============================================================
    # PLOTS
    # ============================================================

    # --------------------------
    # AUROC / AUPRC model comparison.
    # --------------------------
    for metric_name in ["AUROC", "AUPRC"]:
        plt.figure(figsize=(11, 5))
        sub = metrics[metrics["level"] == "clone"].copy()

        sns.barplot(
            data=sub,
            x="fate",
            y=metric_name,
            hue="model",
            order=selected_fates,
            hue_order=MODELS,
            ci="sd",
        )

        sns.stripplot(
            data=sub,
            x="fate",
            y=metric_name,
            hue="model",
            order=selected_fates,
            hue_order=MODELS,
            dodge=True,
            color="black",
            alpha=0.45,
            size=3,
            legend=False,
        )

        if metric_name == "AUROC":
            plt.axhline(0.5, color="gray", linestyle="--", linewidth=2)

        plt.ylim(0, 1)
        plt.title(f"Clone-level {metric_name}: held-out future fate prediction")
        plt.xlabel("future lineage")
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"clone_model_comparison_{metric_name}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"clone_model_comparison_{metric_name}.svg"), bbox_inches="tight")
        plt.show()

    # --------------------------
    # Paired AUROC deltas.
    # --------------------------
    delta_cols = ["cipher_minus_direct", "cipher_minus_diag", "cipher_minus_shuffled"]

    long_delta = pivot_auc.melt(
        id_vars=["fold", "fate"],
        value_vars=delta_cols,
        var_name="comparison",
        value_name="delta_AUROC",
    )

    long_delta["comparison"] = (
        long_delta["comparison"]
        .str.replace("cipher_minus_", "CIPHER - ", regex=False)
    )

    plt.figure(figsize=(11, 5))
    sns.barplot(
        data=long_delta,
        x="fate",
        y="delta_AUROC",
        hue="comparison",
        order=selected_fates,
        ci="sd",
    )
    sns.stripplot(
        data=long_delta,
        x="fate",
        y="delta_AUROC",
        hue="comparison",
        order=selected_fates,
        dodge=True,
        color="black",
        alpha=0.45,
        size=3,
        legend=False,
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=2)
    plt.title("Paired clone-level AUROC improvement")
    plt.xlabel("future lineage")
    plt.ylabel("ΔAUROC")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "paired_CIPHER_AUROC_improvement.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "paired_CIPHER_AUROC_improvement.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # Top-decile enrichment.
    # --------------------------
    plt.figure(figsize=(11, 5))
    sub = metrics[metrics["level"] == "clone"].copy()
    sns.barplot(
        data=sub,
        x="fate",
        y="top_decile_enrichment",
        hue="model",
        order=selected_fates,
        hue_order=MODELS,
        ci="sd",
    )
    plt.axhline(1, color="gray", linestyle="--", linewidth=2)
    plt.title("Top-decile enrichment for future fate")
    plt.xlabel("future lineage")
    plt.ylabel("enrichment among top 10% scores")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "top_decile_enrichment.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "top_decile_enrichment.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # CIPHER heatmap + confusion matrix.
    # --------------------------
    main_clone = clone_probs[clone_probs["model"] == MAIN_MODEL].copy()
    p_norm_cols = [f"p_norm__{safe_name(f)}" for f in selected_fates]

    mean_prob = (
        main_clone
        .groupby("true_future_lineage")[p_norm_cols]
        .mean()
        .reindex(selected_fates)
    )
    mean_prob.columns = selected_fates

    cm = confusion_matrix(
        main_clone["true_future_lineage"],
        main_clone["predicted_lineage_norm"],
        labels=selected_fates,
    )

    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        mean_prob,
        ax=axes[0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "mean normalized pseudo-prob"},
    )
    axes[0].set_title("Clone mean CIPHER pseudo-probabilities")
    axes[0].set_xlabel("predicted future lineage")
    axes[0].set_ylabel("true future lineage")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        pd.DataFrame(cm_norm, index=selected_fates, columns=selected_fates),
        ax=axes[1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "row-normalized fraction"},
    )
    axes[1].set_title("Argmax CIPHER prediction")
    axes[1].set_xlabel("predicted future lineage")
    axes[1].set_ylabel("true future lineage")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_probability_heatmap_confusion.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_probability_heatmap_confusion.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # CIPHER p(true fate) distributions.
    # --------------------------
    plt.figure(figsize=(11, 5))
    sns.boxplot(
        data=main_clone,
        x="true_future_lineage",
        y="p_ovr_true_future_lineage",
        order=selected_fates,
        showfliers=False,
    )
    sns.stripplot(
        data=main_clone,
        x="true_future_lineage",
        y="p_ovr_true_future_lineage",
        order=selected_fates,
        color="black",
        alpha=0.35,
        size=3,
    )
    plt.ylim(0, 1)
    plt.title("CIPHER probability assigned to true future lineage")
    plt.xlabel("true future lineage")
    plt.ylabel("p(true future lineage | early clone)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_p_true_lineage_by_fate.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_p_true_lineage_by_fate.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # Future-fate vs other score distributions.
    # --------------------------
    score_rows = []

    for fate in selected_fates:
        col = f"p_ovr__{safe_name(fate)}"
        tmp = main_clone[["clone_id", "true_future_lineage", col]].copy()
        tmp["tested_fate"] = fate
        tmp["is_future_fate"] = np.where(tmp["true_future_lineage"] == fate, "future fate", "other")
        tmp["p_ovr"] = tmp[col]
        score_rows.append(tmp[["clone_id", "tested_fate", "is_future_fate", "p_ovr"]])

    score_df = pd.concat(score_rows, ignore_index=True)

    plt.figure(figsize=(12, 5))
    sns.boxplot(
        data=score_df,
        x="tested_fate",
        y="p_ovr",
        hue="is_future_fate",
        order=selected_fates,
        showfliers=False,
    )
    plt.ylim(0, 1)
    plt.title("CIPHER one-vs-rest scores: future-fate clones vs others")
    plt.xlabel("tested lineage")
    plt.ylabel("one-vs-rest probability")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_positive_vs_rest_scores.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_positive_vs_rest_scores.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # Top force genes heatmap.
    # --------------------------
    cipher_force = force_df[
        (force_df["model"] == MAIN_MODEL) &
        (force_df["direction"] == "positive")
    ].copy()

    mean_force = (
        cipher_force
        .groupby(["fate", "gene"], as_index=False)
        .agg(mean_u=("u", "mean"), mean_delta=("delta_early", "mean"), mean_rank=("rank", "mean"))
    )

    top_genes = []
    TOP_GENES_PER_FATE = 12

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(TOP_GENES_PER_FATE)
        )
        top_genes.extend(sub["gene"].tolist())

    top_genes = list(dict.fromkeys(top_genes))

    heat = (
        mean_force
        .pivot_table(index="gene", columns="fate", values="mean_u", fill_value=0)
        .reindex(top_genes)
        .reindex(columns=selected_fates)
    )

    plt.figure(figsize=(1.4 * len(selected_fates) + 6, 0.28 * len(top_genes) + 4))
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "mean CIPHER force u"},
    )
    plt.title("Top positive early-bias CIPHER force genes")
    plt.xlabel("future lineage")
    plt.ylabel("gene")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "top_CIPHER_force_genes_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "top_CIPHER_force_genes_heatmap.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # PCA of early cells.
    # --------------------------
    MAX_PLOT_CELLS = 7000
    main_cell = early_cell_probs[early_cell_probs["model"] == MAIN_MODEL].copy()

    plot_df = main_cell.copy()
    if len(plot_df) > MAX_PLOT_CELLS:
        plot_df = plot_df.sample(MAX_PLOT_CELLS, random_state=SEED)

    plot_cells = plot_df["cell_index"].values.astype(int)
    X_plot = get_cells_x_genes(counts, plot_cells, hvg_idx)
    X_plot = apply_zscore(X_plot, mu_ref, sd_ref)

    Z = PCA(n_components=2, random_state=SEED).fit_transform(X_plot)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=plot_df["p_ovr_true_future_lineage"].values,
        s=8,
        alpha=0.8,
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    plt.colorbar(sc, label="p(true future lineage | early cell)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Early cells colored by CIPHER probability of true future lineage")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "early_cells_pca_p_true_lineage.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "early_cells_pca_p_true_lineage.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # PRINT FINAL SUMMARY
    # ============================================================

    print("\n============================================================")
    print("FINAL CLONE-LEVEL CIPHER SUMMARY")
    print("============================================================")
    print(
        summary_metrics[
            (summary_metrics["level"] == "clone") &
            (summary_metrics["model"] == MAIN_MODEL)
        ][[
            "fate",
            "n_positive_mean",
            "positive_fraction_mean",
            "AUROC_mean",
            "AUROC_sd",
            "AUPRC_mean",
            "AUPRC_sd",
            "top_decile_enrichment_mean",
        ]].sort_values("AUROC_mean", ascending=False)
    )

    print("\n============================================================")
    print("ARGMAX ACCURACY")
    print("============================================================")
    print(acc_df)

    print("\n============================================================")
    print("PAIRED AUROC DELTAS")
    print("============================================================")
    print(pivot_auc)

    print("\n============================================================")
    print("TOP POSITIVE CIPHER FORCE GENES")
    print("============================================================")

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(20)
        )
        print(f"\n{fate}")
        print(", ".join(sub["gene"].astype(str).tolist()))

    print("\nDone. Outputs in:", OUTDIR)



def strict_clone_qc_global_covariance():
    global os, gzip, warnings, np, pd, plt, sns, mmread, \
        issparse, StratifiedKFold, LogisticRegression, roc_auc_score, average_precision_score, confusion_matrix, PCA, OUTDIR, \
        COUNTS_PATH, GENES_PATH, CLONE_PATH, META_PATH, TIME_COL, CELLTYPE_COL, WELL_COL, START_COL, \
        EARLY_TIME, EARLY_CELLTYPE, EARLY_WELL, TERMINAL_TIME, TERMINAL_WELL, EXCLUDE_FATES, CLONE_FILTER_GRID, MIN_CLONES_PER_FATE, \
        MAX_FATES, N_VAR_GENES, MAX_COV_CELLS, RIDGE, COV_SHRINK_TO_DIAG, MODELS, MAIN_MODEL, N_SPLITS, \
        SEED, rng, safe_name, barplot_sd, mask_early_cells, mask_terminal_cells, get_cell_to_clone, get_cells_x_genes, \
        zscore_train, fate_entropy_from_counts, select_hvgs_sparse, make_covariance, clone_mean_matrix, fit_score_calibrator, compute_metrics, build_clone_table_with_filters, \
        choose_clone_table, score_matrix_with_model, counts, f, gene_names, clone_mat, meta, cell_to_clone, \
        has_clone, fate_labels, early_all_mask, early_cloned_mask, terminal_cloned_mask, early_all_idx, early_cloned_idx, terminal_cloned_idx, \
        clone_table_all, selected_fates, chosen_filter, tried_filters, name, n_clones, n_fates, fates, \
        clone_table, eligible_clones, eligible_early_mask, eligible_early_idx, clone_to_fate, clone_to_frac, clone_to_n_total, clone_to_n_early, \
        clone_to_n_terminal, fig, axes, start_rows, cid, early_cells, start_counts, start_df, \
        tab, hvg_idx, gene_vars, hvg_genes, cov_idx, Xcov_raw, mu_ref, sd_ref, \
        Xcov, Sigma, evals, evecs, diag, X_clones, y_clones, min_class_n, \
        n_splits, splitter, all_cell_rows, all_clone_rows, force_rows, fold, train_pos, test_pos, \
        train_clones, test_clones, Xtrain_clone, train_clone_ids_used, n_train_early, y_train, y_train_shuffled, force_info, \
        fate, pos, neg, delta, pos_shuf, neg_shuf, delta_shuf, U_dict, \
        y_binary, model_name, u, train_scores, smu, ssd, train_scores_scaled, calibrator, \
        key, Xtest_clone, test_clone_ids_used, n_test_early, y_test_clone, base_clone, test_early_idx, Xtest_cell_raw, \
        Xtest_cell, test_cell_clone_ids, y_test_cell, base_cell, model, level_name, Xscore, base_df, \
        collector, raw_scores, scaled_scores, p_ovr, p_norm, pred_idx, pred_fates, rows, \
        j, s, U, DELTAS, top_pos, top_neg, rank, gi, \
        early_cell_probs, clone_probs, force_df, metric_rows, df_clone, df_cell, m_clone, m_cell, \
        metrics, summary_metrics, acc_rows, df, sub, acc_df, clone_auc, pivot_auc, \
        base, metric_name, handles, labels, delta_cols, long_delta, uniq, uniq_labels, \
        h, l, main_clone, p_norm_cols, mean_prob, cm, cm_norm, score_rows, \
        col, tmp, score_df, cipher_force, mean_force, top_genes, TOP_GENES_PER_FATE, heat, \
        MAX_PLOT_CELLS, main_cell, plot_df, plot_cells, X_plot, Z, sc
    # ============================================================
    # CIPHER-LARRY STRICT CLONE QC + GLOBAL EARLY COVARIANCE
    # ============================================================
    # This is the corrected version of the "strict clone QC + more plots" block.
    #
    # Core design:
    #   1. Strict terminal fate labels:
    #        clones need enough terminal cells, dominant fate count, high purity.
    #
    #   2. Lenient early clone requirement:
    #        one early cell per clone is OK because the early cell is the
    #        prospective measurement.
    #
    #   3. CIPHER covariance Sigma is estimated from ALL early undifferentiated
    #      cells, not only fate-labeled clones.
    #
    #   4. Fate vectors are clone-balanced:
    #        Delta_f = mean_clone(X_early | future fate f)
    #                  - mean_clone(X_early | future fate not f)
    #
    #   5. Compares:
    #        cipher   = full covariance inverse
    #        diag     = diagonal covariance inverse
    #        direct   = raw Delta_f
    #        shuffled = shuffled future-fate-label null
    #
    #   6. Proper held-out clone-level CV.
    #      If too few clones per fate, the code raises an error instead of
    #      falling back to in-sample scoring.
    #
    # ============================================================

    import os, gzip, warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.io import mmread
    from scipy.sparse import issparse
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    from sklearn.decomposition import PCA

    warnings.filterwarnings("ignore")

    # ============================================================
    # 0. CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_strict_qc_global_sigma_final")
    os.makedirs(OUTDIR, exist_ok=True)

    COUNTS_PATH = os.path.join(SUPPL, "stateFate_inVitro_normed_counts.mtx.gz")
    GENES_PATH  = os.path.join(SUPPL, "stateFate_inVitro_gene_names.txt.gz")
    CLONE_PATH  = os.path.join(SUPPL, "stateFate_inVitro_clone_matrix.mtx.gz")
    META_PATH   = os.path.join(SUPPL, "stateFate_inVitro_metadata.txt.gz")

    TIME_COL = "Time point"
    CELLTYPE_COL = "Cell type annotation"
    WELL_COL = "Well"
    START_COL = "Starting population"

    # Early/precommitted cells.
    EARLY_TIME = 4.0
    EARLY_CELLTYPE = "Undifferentiated"

    # Important:
    #   None uses all early wells and gives better Sigma.
    #   0 reproduces the earlier well-0 setup.
    EARLY_WELL = None

    # Terminal fate cells.
    TERMINAL_TIME = 6.0
    TERMINAL_WELL = None

    EXCLUDE_FATES = {
        "Undifferentiated",
        "Unknown",
        "unknown",
        "nan",
        "NaN",
        "Ambiguous",
        "ambiguous",
        "None",
        "",
    }

    # Strict terminal fate QC, lenient early cell requirement.
    CLONE_FILTER_GRID = [
        dict(
            name="strict_terminal_lenient_early",
            min_total=12,
            min_early=1,
            min_terminal=8,
            min_dom_count=6,
            min_dom_frac=0.85,
            max_entropy=0.65,
        ),
        dict(
            name="medium_terminal_lenient_early",
            min_total=10,
            min_early=1,
            min_terminal=5,
            min_dom_count=4,
            min_dom_frac=0.80,
            max_entropy=0.75,
        ),
        dict(
            name="lenient_terminal_still_qc",
            min_total=8,
            min_early=1,
            min_terminal=4,
            min_dom_count=3,
            min_dom_frac=0.75,
            max_entropy=0.85,
        ),
    ]

    # Keep this high enough to avoid one-clone/few-clone fake-perfect fates.
    MIN_CLONES_PER_FATE = 2
    MAX_FATES = 5

    # Expression/covariance settings.
    N_VAR_GENES = 10000
    MAX_COV_CELLS = 50000

    # CIPHER regularization. Do not set these to zero for this dataset.
    RIDGE = 0.000000001
    COV_SHRINK_TO_DIAG = 0.

    # Models.
    MODELS = ["cipher", "diag", "direct", "shuffled"]
    MAIN_MODEL = "cipher"

    # CV.
    N_SPLITS = 2
    SEED = 0
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    # Plotting.
    plt.rcParams.update({"font.size": 14})
    sns.set_context("talk")

    # ============================================================
    # 1. HELPERS
    # ============================================================


    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )


    def barplot_sd(*args, **kwargs):
        """
        Seaborn compatibility wrapper:
        new seaborn uses errorbar='sd'; older seaborn uses ci='sd'.
        """
        try:
            return sns.barplot(*args, errorbar="sd", **kwargs)
        except Exception:
            return sns.barplot(*args, ci="sd", **kwargs)

    def mask_early_cells(meta):
        m = meta[TIME_COL].astype(float).values == float(EARLY_TIME)
        if EARLY_CELLTYPE is not None:
            m &= meta[CELLTYPE_COL].astype(str).values == str(EARLY_CELLTYPE)
        if EARLY_WELL is not None and WELL_COL in meta.columns:
            m &= meta[WELL_COL].astype(float).values == float(EARLY_WELL)
        return m

    def mask_terminal_cells(meta):
        m = meta[TIME_COL].astype(float).values == float(TERMINAL_TIME)
        if TERMINAL_WELL is not None and WELL_COL in meta.columns:
            m &= meta[WELL_COL].astype(float).values == float(TERMINAL_WELL)
        ann = meta[CELLTYPE_COL].astype(str).values
        m &= ~np.isin(ann, list(EXCLUDE_FATES))
        return m

    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()
        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row
        return cell_to_clone

    def get_cells_x_genes(counts, cell_idx, gene_idx):
        # counts is genes x cells
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)

    def zscore_train(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu, sd


    def fate_entropy_from_counts(counts_vec):
        counts_vec = np.asarray(counts_vec, dtype=float)
        counts_vec = counts_vec[counts_vec > 0]
        if counts_vec.size <= 1:
            return 0.0
        p = counts_vec / counts_vec.sum()
        return float(-(p * np.log(p)).sum())

    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        """
        Select HVGs from sparse matrix without densifying all genes x cells.
        """
        X = counts[:, cell_idx]
        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means**2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]

        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_var_genes:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_

    def make_covariance(X):
        """
        X: samples x genes, already z-scored.
        Samples are all early cells for the global Sigma.
        """
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D
        Sigma = Sigma + RIDGE * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)

    def clone_mean_matrix(clone_ids, early_mask, cell_to_clone, counts, hvg_idx, mu, sd):
        """
        Clone-balanced early mean matrix.
        Returns:
            X_clone_mean: n_clones x n_genes
            clone_ids_out
            n_early_out
        """
        rows = []
        out_ids = []
        out_n = []

        for cid in clone_ids:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]
            if len(idx) == 0:
                continue

            X = get_cells_x_genes(counts, idx, hvg_idx)
            X = apply_zscore(X, mu, sd)
            rows.append(X.mean(axis=0))
            out_ids.append(cid)
            out_n.append(len(idx))

        if len(rows) == 0:
            return np.empty((0, len(hvg_idx))), np.array([], dtype=int), np.array([], dtype=int)

        return np.vstack(rows), np.asarray(out_ids, dtype=int), np.asarray(out_n, dtype=int)

    def fit_score_calibrator(scores_train, y_train):
        scores_train = np.asarray(scores_train).reshape(-1, 1)
        y_train = np.asarray(y_train).astype(int)

        if len(np.unique(y_train)) < 2:
            return None

        clf = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        )
        clf.fit(scores_train, y_train)
        return clf


    def compute_metrics(df, selected_fates, prob_prefix="p_ovr", label_col="true_future_lineage"):
        rows = []

        for fate in selected_fates:
            col = f"{prob_prefix}__{safe_name(fate)}"
            y = (df[label_col].astype(str).values == str(fate)).astype(int)
            s = df[col].values.astype(float)

            if len(np.unique(y)) < 2:
                auroc = np.nan
                auprc = np.nan
            else:
                auroc = roc_auc_score(y, s)
                auprc = average_precision_score(y, s)

            baseline = y.mean()
            cutoff = np.quantile(s, 0.90)
            top = s >= cutoff

            if top.sum() > 0 and baseline > 0:
                top_rate = y[top].mean()
                enrichment = top_rate / baseline
            else:
                top_rate = np.nan
                enrichment = np.nan

            rows.append({
                "fate": fate,
                "n": len(y),
                "n_positive": int(y.sum()),
                "positive_fraction": float(baseline),
                "AUROC": auroc,
                "AUPRC": auprc,
                "top_decile_positive_rate": top_rate,
                "top_decile_enrichment": enrichment,
            })

        return pd.DataFrame(rows)

    def build_clone_table_with_filters(clone_mat, early_mask, terminal_mask, fate_labels, config):
        records = []

        for clone_id in range(clone_mat.shape[0]):
            cells = clone_mat[clone_id].indices

            if len(cells) < config["min_total"]:
                continue

            early_cells = cells[early_mask[cells]]
            terminal_cells = cells[terminal_mask[cells]]

            if len(early_cells) < config["min_early"]:
                continue

            if len(terminal_cells) < config["min_terminal"]:
                continue

            terminal_fates = pd.Series(fate_labels[terminal_cells].astype(str))
            terminal_fates = terminal_fates[~terminal_fates.isin(EXCLUDE_FATES)]

            if len(terminal_fates) < config["min_terminal"]:
                continue

            fate_counts = terminal_fates.value_counts()

            if len(fate_counts) == 0:
                continue

            dominant_fate = str(fate_counts.index[0])
            dominant_count = int(fate_counts.iloc[0])
            total_terminal = int(fate_counts.sum())
            dominant_frac = dominant_count / max(total_terminal, 1)
            entropy = fate_entropy_from_counts(fate_counts.values)

            if dominant_count < config["min_dom_count"]:
                continue
            if dominant_frac < config["min_dom_frac"]:
                continue
            if config["max_entropy"] is not None and entropy > config["max_entropy"]:
                continue

            rec = {
                "clone_id": int(clone_id),
                "n_total_clone_cells": int(len(cells)),
                "n_early": int(len(early_cells)),
                "n_terminal": int(total_terminal),
                "n_terminal_raw": int(len(terminal_cells)),
                "n_terminal_fate_types": int(len(fate_counts)),
                "dominant_fate": dominant_fate,
                "dominant_count": dominant_count,
                "dominant_frac": float(dominant_frac),
                "fate_entropy": float(entropy),
                "filter_config": config["name"],
            }

            for fate, count in fate_counts.items():
                s = safe_name(fate)
                rec[f"terminal_count__{s}"] = int(count)
                rec[f"terminal_frac__{s}"] = float(count / total_terminal)

            records.append(rec)

        return pd.DataFrame(records)

    def choose_clone_table():
        """
        Try strict clone filters first. Use first filter level with >=2 fates passing
        MIN_CLONES_PER_FATE. No single-clone fate fallback.
        """
        tried = []

        for cfg in CLONE_FILTER_GRID:
            ct = build_clone_table_with_filters(
                clone_mat=clone_mat,
                early_mask=early_cloned_mask,
                terminal_mask=terminal_cloned_mask,
                fate_labels=fate_labels,
                config=cfg,
            )

            if ct.empty:
                tried.append((cfg["name"], 0, 0, []))
                continue

            fate_counts = ct["dominant_fate"].value_counts()
            selected = fate_counts[fate_counts >= MIN_CLONES_PER_FATE].index.tolist()
            selected = selected[:MAX_FATES]

            tried.append((cfg["name"], len(ct), len(selected), selected))

            if len(selected) >= 2:
                return ct, selected, cfg, tried

        raise RuntimeError(
            "No clone-filter setting produced at least 2 fates with enough clones. "
            "Lower MIN_CLONES_PER_FATE or relax terminal QC, but do not use MIN_CLONES_PER_FATE=1."
        )

    def score_matrix_with_model(X, model):
        U = model["U"]
        raw_scores = X @ U.T

        scaled_scores = (
            raw_scores - model["score_mu"][None, :]
        ) / model["score_sd"][None, :]

        p_ovr = np.zeros_like(scaled_scores, dtype=float)

        for j, clf in enumerate(model["calibrators"]):
            p_ovr[:, j] = calibrate_scores(clf, scaled_scores[:, j])

        p_ovr = np.nan_to_num(p_ovr, nan=0.0, posinf=1.0, neginf=0.0)
        p_ovr = np.clip(p_ovr, 1e-6, 1.0 - 1e-6)
        p_norm = soft_normalize(p_ovr)

        return raw_scores, scaled_scores, p_ovr, p_norm

    # ============================================================
    # 2. LOAD DATA EXACTLY LIKE YOUR LARRY BLOCK
    # ============================================================

    counts = mmread(COUNTS_PATH).T.tocsr()  # genes x cells
    print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} cells | nnz={counts.nnz:,}")

    with gzip.open(GENES_PATH, "rt") as f:
        gene_names = np.array([line.strip() for line in f])
    print(f"Genes loaded: {len(gene_names)}")

    clone_mat = mmread(CLONE_PATH).T.tocsr()  # clones x cells
    print(f"Clone matrix: {clone_mat.shape[0]} clones x {clone_mat.shape[1]} cells")
    print(f"% cells with clone label: {(clone_mat.sum(axis=0) > 0).mean() * 100:.2f}%")

    meta = pd.read_csv(META_PATH, sep="\t")
    print(f"Meta: {meta.shape[0]} rows x {meta.shape[1]} cols")
    print("Meta columns:", list(meta.columns))

    assert counts.shape[1] == meta.shape[0] == clone_mat.shape[1], "cells mismatch"
    assert counts.shape[0] == len(gene_names), "genes mismatch"

    meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")

    print("\nTimepoints:")
    print(np.sort(meta[TIME_COL].dropna().unique()))

    print("\nCell annotations:")
    print(meta[CELLTYPE_COL].value_counts())

    cell_to_clone = get_cell_to_clone(clone_mat)
    has_clone = cell_to_clone >= 0
    fate_labels = meta[CELLTYPE_COL].astype(str).values

    # ============================================================
    # 3. DEFINE EARLY / TERMINAL CELLS
    # ============================================================

    early_all_mask = mask_early_cells(meta)
    early_cloned_mask = early_all_mask & has_clone
    terminal_cloned_mask = mask_terminal_cells(meta) & has_clone

    early_all_idx = np.where(early_all_mask)[0]
    early_cloned_idx = np.where(early_cloned_mask)[0]
    terminal_cloned_idx = np.where(terminal_cloned_mask)[0]

    print(f"\nAll early/precommitted cells for Sigma: {len(early_all_idx):,}")
    print(f"Cloned early/precommitted cells: {len(early_cloned_idx):,}")
    print(f"Cloned terminal cells: {len(terminal_cloned_idx):,}")

    if len(early_all_idx) == 0:
        raise RuntimeError("No early cells found. Check EARLY_TIME / EARLY_CELLTYPE / EARLY_WELL.")

    if len(terminal_cloned_idx) == 0:
        raise RuntimeError("No terminal cloned cells found. Check TERMINAL_TIME / TERMINAL_WELL.")

    # ============================================================
    # 4. STRICT CLONE TABLE / FUTURE FATE LABELS
    # ============================================================

    clone_table_all, selected_fates, chosen_filter, tried_filters = choose_clone_table()

    print("\nClone-filter attempts:")
    for name, n_clones, n_fates, fates in tried_filters:
        print(f"  {name:28s} n_clones={n_clones:4d} n_fates={n_fates} fates={fates}")

    print("\nUsing clone filter:")
    print(chosen_filter)

    print("\nClone table after chosen clone QC:")
    print(f"n clones passing filters: {len(clone_table_all):,}")
    print(clone_table_all["dominant_fate"].value_counts())

    clone_table = clone_table_all[
        clone_table_all["dominant_fate"].isin(selected_fates)
    ].copy()

    eligible_clones = clone_table["clone_id"].values.astype(int)
    eligible_early_mask = early_cloned_mask & np.isin(cell_to_clone, eligible_clones)
    eligible_early_idx = np.where(eligible_early_mask)[0]

    clone_to_fate = dict(zip(clone_table["clone_id"], clone_table["dominant_fate"]))
    clone_to_frac = dict(zip(clone_table["clone_id"], clone_table["dominant_frac"]))
    clone_to_n_total = dict(zip(clone_table["clone_id"], clone_table["n_total_clone_cells"]))
    clone_to_n_early = dict(zip(clone_table["clone_id"], clone_table["n_early"]))
    clone_to_n_terminal = dict(zip(clone_table["clone_id"], clone_table["n_terminal"]))

    print("\nSelected fates:")
    print(clone_table["dominant_fate"].value_counts())

    print(f"\nEligible clones: {len(eligible_clones):,}")
    print(f"Eligible early cells: {len(eligible_early_idx):,}")

    clone_table_all.to_csv(os.path.join(OUTDIR, "clone_table_all_passing_qc.csv"), index=False)
    clone_table.to_csv(os.path.join(OUTDIR, "clone_table_selected_fates.csv"), index=False)

    # ============================================================
    # 5. CLONE QC PLOTS
    # ============================================================

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    sns.countplot(data=clone_table, x="dominant_fate", order=selected_fates, ax=axes[0, 0])
    axes[0, 0].set_title("Selected high-confidence clones per future fate")
    axes[0, 0].set_xlabel("future fate")
    axes[0, 0].set_ylabel("clone count")
    axes[0, 0].tick_params(axis="x", rotation=45)

    sns.histplot(data=clone_table, x="n_total_clone_cells", bins=40, ax=axes[0, 1])
    axes[0, 1].set_title("Total cells per retained clone")
    axes[0, 1].set_xlabel("total clone size")

    sns.histplot(data=clone_table, x="n_early", bins=30, ax=axes[0, 2])
    axes[0, 2].set_title("Early cells per retained clone")
    axes[0, 2].set_xlabel("early cells per clone")

    sns.histplot(data=clone_table, x="n_terminal", bins=40, ax=axes[1, 0])
    axes[1, 0].set_title("Terminal cells per retained clone")
    axes[1, 0].set_xlabel("terminal cells per clone")

    sns.scatterplot(
        data=clone_table,
        x="n_terminal",
        y="dominant_frac",
        hue="dominant_fate",
        hue_order=selected_fates,
        ax=axes[1, 1],
        s=45,
    )
    axes[1, 1].set_title("Clone purity vs terminal clone size")
    axes[1, 1].set_xlabel("terminal cells")
    axes[1, 1].set_ylabel("dominant fate fraction")
    axes[1, 1].legend(fontsize=9, frameon=False)

    sns.scatterplot(
        data=clone_table,
        x="n_early",
        y="n_terminal",
        hue="dominant_fate",
        hue_order=selected_fates,
        ax=axes[1, 2],
        s=45,
    )
    axes[1, 2].set_title("Early vs terminal clone representation")
    axes[1, 2].set_xlabel("early cells")
    axes[1, 2].set_ylabel("terminal cells")
    axes[1, 2].legend(fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "clone_qc_summary.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "clone_qc_summary.svg"), bbox_inches="tight")
    plt.show()

    if START_COL in meta.columns:
        start_rows = []
        for cid in eligible_clones:
            early_cells = np.where(eligible_early_mask & (cell_to_clone == cid))[0]
            if len(early_cells) == 0:
                continue

            start_counts = meta.iloc[early_cells][START_COL].astype(str).value_counts()

            start_rows.append({
                "clone_id": cid,
                "future_fate": clone_to_fate[cid],
                "dominant_starting_population": start_counts.index[0],
                "n_early": len(early_cells),
            })

        start_df = pd.DataFrame(start_rows)
        start_df.to_csv(os.path.join(OUTDIR, "clone_starting_population_summary.csv"), index=False)

        plt.figure(figsize=(10, 5))
        tab = pd.crosstab(start_df["future_fate"], start_df["dominant_starting_population"])
        tab = tab.reindex(selected_fates)
        sns.heatmap(tab, annot=True, fmt="d", cmap="viridis")
        plt.title("Future fate vs early starting population")
        plt.xlabel("dominant starting population among early cells")
        plt.ylabel("future fate")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "future_fate_vs_starting_population.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "future_fate_vs_starting_population.svg"), bbox_inches="tight")
        plt.show()

    # ============================================================
    # 6. GLOBAL EARLY HVGs + GLOBAL EARLY COVARIANCE
    # ============================================================

    print("\nSelecting HVGs from all early/precommitted cells...")

    hvg_idx, gene_vars = select_hvgs_sparse(
        counts=counts,
        cell_idx=early_all_idx,
        n_var_genes=N_VAR_GENES,
    )

    hvg_genes = gene_names[hvg_idx]

    pd.DataFrame({
        "gene": hvg_genes,
        "gene_index": hvg_idx,
        "early_variance": gene_vars[hvg_idx],
    }).to_csv(os.path.join(OUTDIR, "selected_early_hvgs.csv"), index=False)

    print(f"Using top {len(hvg_idx)} early-variable genes.")

    cov_idx = early_all_idx.copy()
    if len(cov_idx) > MAX_COV_CELLS:
        cov_idx = rng.choice(cov_idx, size=MAX_COV_CELLS, replace=False)

    print(f"Using {len(cov_idx):,} early cells for global Sigma.")

    Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx)
    mu_ref, sd_ref = zscore_train(Xcov_raw)
    Xcov = apply_zscore(Xcov_raw, mu_ref, sd_ref)

    Sigma = make_covariance(Xcov)

    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-8)

    diag = np.diag(Sigma).copy()
    diag[diag < 1e-8] = 1e-8

    pd.DataFrame({
        "rank": np.arange(1, len(evals) + 1),
        "eigenvalue": evals[::-1],
    }).to_csv(os.path.join(OUTDIR, "early_covariance_eigenvalues.csv"), index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(evals) + 1), evals[::-1], marker="o", linewidth=1, markersize=3)
    plt.yscale("log")
    plt.xlabel("eigenvalue rank")
    plt.ylabel("eigenvalue")
    plt.title("Early progenitor covariance spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # 7. CROSS-VALIDATED CLONE-BALANCED EARLY-BIAS CIPHER
    # ============================================================

    X_clones = clone_table["clone_id"].values.astype(int)
    y_clones = clone_table["dominant_fate"].values.astype(str)

    min_class_n = clone_table["dominant_fate"].value_counts().min()
    n_splits = int(min(N_SPLITS, min_class_n))

    if n_splits < 2:
        raise RuntimeError(
            f"Cannot do held-out CV: smallest selected fate has only {min_class_n} clones. "
            "Increase MIN_CLONES_PER_FATE filtering or reduce MAX_FATES."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    print(f"\nUsing clone-level stratified {n_splits}-fold CV.")

    all_cell_rows = []
    all_clone_rows = []
    force_rows = []

    for fold, (train_pos, test_pos) in enumerate(splitter.split(X_clones, y_clones)):
        train_clones = X_clones[train_pos]
        test_clones = X_clones[test_pos]

        print(f"\nFold {fold + 1}/{n_splits}: train clones={len(train_clones)}, test clones={len(test_clones)}")

        Xtrain_clone, train_clone_ids_used, n_train_early = clone_mean_matrix(
            clone_ids=train_clones,
            early_mask=eligible_early_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        if Xtrain_clone.shape[0] < 5:
            raise RuntimeError("Too few training clone means.")

        y_train = np.array([clone_to_fate[c] for c in train_clone_ids_used])

        y_train_shuffled = y_train.copy()
        rng.shuffle(y_train_shuffled)

        force_info = {
            model: {
                "U": [],
                "DELTAS": [],
                "calibrators": [],
                "score_mu": [],
                "score_sd": [],
            }
            for model in MODELS
        }

        for fate in selected_fates:
            pos = y_train == fate
            neg = y_train != fate

            if pos.sum() == 0 or neg.sum() == 0:
                raise RuntimeError(f"Missing train positives/negatives for fate {fate}")

            delta = Xtrain_clone[pos].mean(axis=0) - Xtrain_clone[neg].mean(axis=0)

            pos_shuf = y_train_shuffled == fate
            neg_shuf = y_train_shuffled != fate

            if pos_shuf.sum() == 0 or neg_shuf.sum() == 0:
                delta_shuf = rng.normal(size=delta.shape)
                delta_shuf = delta_shuf / (np.linalg.norm(delta_shuf) + 1e-8) * (np.linalg.norm(delta) + 1e-8)
            else:
                delta_shuf = Xtrain_clone[pos_shuf].mean(axis=0) - Xtrain_clone[neg_shuf].mean(axis=0)

            U_dict = {
                "cipher": evecs @ ((evecs.T @ delta) / evals),
                "diag": delta / diag,
                "direct": delta.copy(),
                "shuffled": evecs @ ((evecs.T @ delta_shuf) / evals),
            }

            y_binary = (y_train == fate).astype(int)

            for model_name in MODELS:
                u = U_dict[model_name]
                train_scores = Xtrain_clone @ u

                smu = train_scores.mean()
                ssd = train_scores.std()
                if ssd < 1e-6:
                    ssd = 1.0

                train_scores_scaled = (train_scores - smu) / ssd
                calibrator = fit_score_calibrator(train_scores_scaled, y_binary)

                force_info[model_name]["U"].append(u)
                force_info[model_name]["DELTAS"].append(delta if model_name != "shuffled" else delta_shuf)
                force_info[model_name]["calibrators"].append(calibrator)
                force_info[model_name]["score_mu"].append(smu)
                force_info[model_name]["score_sd"].append(ssd)

        for model_name in MODELS:
            for key in ["U", "DELTAS", "score_mu", "score_sd"]:
                force_info[model_name][key] = np.asarray(force_info[model_name][key])

        # Score held-out clone means.
        Xtest_clone, test_clone_ids_used, n_test_early = clone_mean_matrix(
            clone_ids=test_clones,
            early_mask=eligible_early_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        y_test_clone = np.array([clone_to_fate[c] for c in test_clone_ids_used])

        base_clone = pd.DataFrame({
            "fold": fold,
            "level": "clone",
            "clone_id": test_clone_ids_used,
            "true_future_lineage": y_test_clone,
            "true_future_lineage_frac": [clone_to_frac[c] for c in test_clone_ids_used],
            "n_early_scored": n_test_early,
            "n_total_clone_cells": [clone_to_n_total[c] for c in test_clone_ids_used],
            "n_early_clone_cells": [clone_to_n_early[c] for c in test_clone_ids_used],
            "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_clone_ids_used],
        })

        # Score held-out early cells.
        test_early_idx = np.where(eligible_early_mask & np.isin(cell_to_clone, test_clones))[0]
        Xtest_cell_raw = get_cells_x_genes(counts, test_early_idx, hvg_idx)
        Xtest_cell = apply_zscore(Xtest_cell_raw, mu_ref, sd_ref)

        test_cell_clone_ids = cell_to_clone[test_early_idx]
        y_test_cell = np.array([clone_to_fate[c] for c in test_cell_clone_ids])

        base_cell = pd.DataFrame({
            "fold": fold,
            "level": "cell",
            "cell_index": test_early_idx,
            "clone_id": test_cell_clone_ids,
            "true_future_lineage": y_test_cell,
            "true_future_lineage_frac": [clone_to_frac[c] for c in test_cell_clone_ids],
            "n_total_clone_cells": [clone_to_n_total[c] for c in test_cell_clone_ids],
            "n_early_clone_cells": [clone_to_n_early[c] for c in test_cell_clone_ids],
            "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_cell_clone_ids],
        })

        for model_name in MODELS:
            model = force_info[model_name]

            for level_name, Xscore, base_df, collector in [
                ("clone", Xtest_clone, base_clone, all_clone_rows),
                ("cell", Xtest_cell, base_cell, all_cell_rows),
            ]:
                raw_scores, scaled_scores, p_ovr, p_norm = score_matrix_with_model(Xscore, model)

                pred_idx = np.argmax(p_norm, axis=1)
                pred_fates = np.array(selected_fates, dtype=object)[pred_idx]

                rows = base_df.copy()
                rows["model"] = model_name
                rows["predicted_lineage_norm"] = pred_fates
                rows["max_pseudoprob_norm"] = p_norm.max(axis=1)

                for j, fate in enumerate(selected_fates):
                    s = safe_name(fate)
                    rows[f"score_raw__{s}"] = raw_scores[:, j]
                    rows[f"score_scaled__{s}"] = scaled_scores[:, j]
                    rows[f"p_ovr__{s}"] = p_ovr[:, j]
                    rows[f"p_norm__{s}"] = p_norm[:, j]

                rows["p_ovr_true_future_lineage"] = [
                    p_ovr[i, selected_fates.index(tf)]
                    for i, tf in enumerate(rows["true_future_lineage"].values)
                ]

                rows["p_norm_true_future_lineage"] = [
                    p_norm[i, selected_fates.index(tf)]
                    for i, tf in enumerate(rows["true_future_lineage"].values)
                ]

                collector.append(rows)

        # Save top force genes.
        for model_name in MODELS:
            U = force_info[model_name]["U"]
            DELTAS = force_info[model_name]["DELTAS"]

            for j, fate in enumerate(selected_fates):
                u = U[j]
                delta = DELTAS[j]

                top_pos = np.argsort(u)[::-1][:50]
                top_neg = np.argsort(u)[:50]

                for rank, gi in enumerate(top_pos, start=1):
                    force_rows.append({
                        "fold": fold,
                        "model": model_name,
                        "fate": fate,
                        "direction": "positive",
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),
                        "u": float(u[gi]),
                        "delta_early": float(delta[gi]),
                    })

                for rank, gi in enumerate(top_neg, start=1):
                    force_rows.append({
                        "fold": fold,
                        "model": model_name,
                        "fate": fate,
                        "direction": "negative",
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),
                        "u": float(u[gi]),
                        "delta_early": float(delta[gi]),
                    })

    early_cell_probs = pd.concat(all_cell_rows, ignore_index=True)
    clone_probs = pd.concat(all_clone_rows, ignore_index=True)
    force_df = pd.DataFrame(force_rows)

    early_cell_probs.to_csv(os.path.join(OUTDIR, "early_cell_probs_all_models.csv"), index=False)
    clone_probs.to_csv(os.path.join(OUTDIR, "clone_probs_all_models.csv"), index=False)
    force_df.to_csv(os.path.join(OUTDIR, "top_force_genes_all_models.csv"), index=False)

    print("\nSaved core outputs:")
    print(os.path.join(OUTDIR, "early_cell_probs_all_models.csv"))
    print(os.path.join(OUTDIR, "clone_probs_all_models.csv"))
    print(os.path.join(OUTDIR, "top_force_genes_all_models.csv"))

    # ============================================================
    # 8. METRICS
    # ============================================================

    metric_rows = []

    for model_name in MODELS:
        for fold in sorted(clone_probs["fold"].unique()):
            df_clone = clone_probs[(clone_probs["model"] == model_name) & (clone_probs["fold"] == fold)]
            df_cell = early_cell_probs[(early_cell_probs["model"] == model_name) & (early_cell_probs["fold"] == fold)]

            m_clone = compute_metrics(df_clone, selected_fates)
            m_clone["model"] = model_name
            m_clone["fold"] = fold
            m_clone["level"] = "clone"

            m_cell = compute_metrics(df_cell, selected_fates)
            m_cell["model"] = model_name
            m_cell["fold"] = fold
            m_cell["level"] = "cell"

            metric_rows.append(m_clone)
            metric_rows.append(m_cell)

    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics.to_csv(os.path.join(OUTDIR, "prediction_metrics_by_fold.csv"), index=False)

    summary_metrics = (
        metrics
        .groupby(["level", "model", "fate"], as_index=False)
        .agg(
            AUROC_mean=("AUROC", "mean"),
            AUROC_sd=("AUROC", "std"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_sd=("AUPRC", "std"),
            top_decile_enrichment_mean=("top_decile_enrichment", "mean"),
            top_decile_enrichment_sd=("top_decile_enrichment", "std"),
            n_positive_mean=("n_positive", "mean"),
            positive_fraction_mean=("positive_fraction", "mean"),
        )
    )

    summary_metrics.to_csv(os.path.join(OUTDIR, "prediction_metrics_summary.csv"), index=False)

    acc_rows = []
    for model_name in MODELS:
        for level_name, df in [("clone", clone_probs), ("cell", early_cell_probs)]:
            sub = df[df["model"] == model_name]
            acc_rows.append({
                "model": model_name,
                "level": level_name,
                "argmax_accuracy": np.mean(sub["predicted_lineage_norm"] == sub["true_future_lineage"]),
                "mean_p_true_ovr": sub["p_ovr_true_future_lineage"].mean(),
                "mean_p_true_norm": sub["p_norm_true_future_lineage"].mean(),
            })

    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(os.path.join(OUTDIR, "argmax_accuracy_summary.csv"), index=False)

    clone_auc = metrics[metrics["level"] == "clone"].copy()
    pivot_auc = clone_auc.pivot_table(
        index=["fold", "fate"],
        columns="model",
        values="AUROC",
    ).reset_index()

    for base in ["direct", "diag", "shuffled"]:
        pivot_auc[f"cipher_minus_{base}"] = pivot_auc["cipher"] - pivot_auc[base]

    pivot_auc.to_csv(os.path.join(OUTDIR, "paired_clone_AUROC_deltas.csv"), index=False)

    print("\nClone-level CIPHER summary:")
    print(
        summary_metrics[
            (summary_metrics["level"] == "clone") &
            (summary_metrics["model"] == MAIN_MODEL)
        ][[
            "fate",
            "n_positive_mean",
            "positive_fraction_mean",
            "AUROC_mean",
            "AUROC_sd",
            "AUPRC_mean",
            "AUPRC_sd",
            "top_decile_enrichment_mean",
        ]]
    )

    print("\nArgmax summary:")
    print(acc_df)

    # ============================================================
    # 9. PLOTS
    # ============================================================

    # AUROC/AUPRC model comparison.
    for metric_name in ["AUROC", "AUPRC"]:
        plt.figure(figsize=(11, 5))
        sub = metrics[metrics["level"] == "clone"].copy()

        barplot_sd(
            data=sub,
            x="fate",
            y=metric_name,
            hue="model",
            order=selected_fates,
            hue_order=MODELS,
        )

        sns.stripplot(
            data=sub,
            x="fate",
            y=metric_name,
            hue="model",
            order=selected_fates,
            hue_order=MODELS,
            dodge=True,
            color="black",
            alpha=0.45,
            size=3,
        )

        if metric_name == "AUROC":
            plt.axhline(0.5, color="gray", linestyle="--", linewidth=2)

        plt.ylim(0, 1)
        plt.title(f"Clone-level {metric_name}: held-out future fate prediction")
        plt.xlabel("future lineage")
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha="right")

        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > len(MODELS):
            handles = handles[:len(MODELS)]
            labels = labels[:len(MODELS)]
        plt.legend(handles, labels, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"clone_model_comparison_{metric_name}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"clone_model_comparison_{metric_name}.svg"), bbox_inches="tight")
        plt.show()

    # Paired AUROC deltas.
    delta_cols = ["cipher_minus_direct", "cipher_minus_diag", "cipher_minus_shuffled"]

    long_delta = pivot_auc.melt(
        id_vars=["fold", "fate"],
        value_vars=delta_cols,
        var_name="comparison",
        value_name="delta_AUROC",
    )

    long_delta["comparison"] = long_delta["comparison"].str.replace("cipher_minus_", "CIPHER - ", regex=False)

    plt.figure(figsize=(11, 5))
    barplot_sd(
        data=long_delta,
        x="fate",
        y="delta_AUROC",
        hue="comparison",
        order=selected_fates,
    )
    sns.stripplot(
        data=long_delta,
        x="fate",
        y="delta_AUROC",
        hue="comparison",
        order=selected_fates,
        dodge=True,
        color="black",
        alpha=0.45,
        size=3,
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=2)
    plt.title("Paired clone-level AUROC improvement")
    plt.xlabel("future lineage")
    plt.ylabel("ΔAUROC")
    plt.xticks(rotation=45, ha="right")

    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l not in uniq_labels:
            uniq.append(h)
            uniq_labels.append(l)
    plt.legend(uniq, uniq_labels, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "paired_CIPHER_AUROC_improvement.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "paired_CIPHER_AUROC_improvement.svg"), bbox_inches="tight")
    plt.show()

    # Top-decile enrichment.
    plt.figure(figsize=(11, 5))
    sub = metrics[metrics["level"] == "clone"].copy()
    barplot_sd(
        data=sub,
        x="fate",
        y="top_decile_enrichment",
        hue="model",
        order=selected_fates,
        hue_order=MODELS,
    )
    plt.axhline(1, color="gray", linestyle="--", linewidth=2)
    plt.title("Top-decile enrichment for future fate")
    plt.xlabel("future lineage")
    plt.ylabel("enrichment among top 10% scores")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "top_decile_enrichment.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "top_decile_enrichment.svg"), bbox_inches="tight")
    plt.show()

    # CIPHER heatmap + confusion matrix.
    main_clone = clone_probs[clone_probs["model"] == MAIN_MODEL].copy()
    p_norm_cols = [f"p_norm__{safe_name(f)}" for f in selected_fates]

    mean_prob = (
        main_clone
        .groupby("true_future_lineage")[p_norm_cols]
        .mean()
        .reindex(selected_fates)
    )
    mean_prob.columns = selected_fates

    cm = confusion_matrix(
        main_clone["true_future_lineage"],
        main_clone["predicted_lineage_norm"],
        labels=selected_fates,
    )

    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        mean_prob,
        ax=axes[0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "mean normalized pseudo-prob"},
    )
    axes[0].set_title("Clone mean CIPHER pseudo-probabilities")
    axes[0].set_xlabel("predicted future lineage")
    axes[0].set_ylabel("true future lineage")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        pd.DataFrame(cm_norm, index=selected_fates, columns=selected_fates),
        ax=axes[1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "row-normalized fraction"},
    )
    axes[1].set_title("Argmax CIPHER prediction")
    axes[1].set_xlabel("predicted future lineage")
    axes[1].set_ylabel("true future lineage")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_probability_heatmap_confusion.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_probability_heatmap_confusion.svg"), bbox_inches="tight")
    plt.show()

    # CIPHER p(true fate).
    plt.figure(figsize=(11, 5))
    sns.boxplot(
        data=main_clone,
        x="true_future_lineage",
        y="p_ovr_true_future_lineage",
        order=selected_fates,
        showfliers=False,
    )
    sns.stripplot(
        data=main_clone,
        x="true_future_lineage",
        y="p_ovr_true_future_lineage",
        order=selected_fates,
        color="black",
        alpha=0.35,
        size=3,
    )
    plt.ylim(0, 1)
    plt.title("CIPHER probability assigned to true future lineage")
    plt.xlabel("true future lineage")
    plt.ylabel("p(true future lineage | early clone)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_p_true_lineage_by_fate.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_p_true_lineage_by_fate.svg"), bbox_inches="tight")
    plt.show()

    # Future-fate vs other score distributions.
    score_rows = []

    for fate in selected_fates:
        col = f"p_ovr__{safe_name(fate)}"
        tmp = main_clone[["clone_id", "true_future_lineage", col]].copy()
        tmp["tested_fate"] = fate
        tmp["is_future_fate"] = np.where(tmp["true_future_lineage"] == fate, "future fate", "other")
        tmp["p_ovr"] = tmp[col]
        score_rows.append(tmp[["clone_id", "tested_fate", "is_future_fate", "p_ovr"]])

    score_df = pd.concat(score_rows, ignore_index=True)

    plt.figure(figsize=(12, 5))
    sns.boxplot(
        data=score_df,
        x="tested_fate",
        y="p_ovr",
        hue="is_future_fate",
        order=selected_fates,
        showfliers=False,
    )
    plt.ylim(0, 1)
    plt.title("CIPHER one-vs-rest scores: future-fate clones vs others")
    plt.xlabel("tested lineage")
    plt.ylabel("one-vs-rest probability")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_positive_vs_rest_scores.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_positive_vs_rest_scores.svg"), bbox_inches="tight")
    plt.show()

    # Top CIPHER force genes heatmap.
    cipher_force = force_df[
        (force_df["model"] == MAIN_MODEL) &
        (force_df["direction"] == "positive")
    ].copy()

    mean_force = (
        cipher_force
        .groupby(["fate", "gene"], as_index=False)
        .agg(mean_u=("u", "mean"), mean_delta=("delta_early", "mean"), mean_rank=("rank", "mean"))
    )

    top_genes = []
    TOP_GENES_PER_FATE = 12

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(TOP_GENES_PER_FATE)
        )
        top_genes.extend(sub["gene"].tolist())

    top_genes = list(dict.fromkeys(top_genes))

    heat = (
        mean_force
        .pivot_table(index="gene", columns="fate", values="mean_u", fill_value=0)
        .reindex(top_genes)
        .reindex(columns=selected_fates)
    )

    plt.figure(figsize=(1.4 * len(selected_fates) + 6, 0.28 * len(top_genes) + 4))
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "mean CIPHER force u"},
    )
    plt.title("Top positive early-bias CIPHER force genes")
    plt.xlabel("future lineage")
    plt.ylabel("gene")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "top_CIPHER_force_genes_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "top_CIPHER_force_genes_heatmap.svg"), bbox_inches="tight")
    plt.show()

    # PCA of early cells.
    MAX_PLOT_CELLS = 7000
    main_cell = early_cell_probs[early_cell_probs["model"] == MAIN_MODEL].copy()

    plot_df = main_cell.copy()
    if len(plot_df) > MAX_PLOT_CELLS:
        plot_df = plot_df.sample(MAX_PLOT_CELLS, random_state=SEED)

    plot_cells = plot_df["cell_index"].values.astype(int)
    X_plot = get_cells_x_genes(counts, plot_cells, hvg_idx)
    X_plot = apply_zscore(X_plot, mu_ref, sd_ref)

    Z = PCA(n_components=2, random_state=SEED).fit_transform(X_plot)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=plot_df["p_ovr_true_future_lineage"].values,
        s=8,
        alpha=0.8,
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    plt.colorbar(sc, label="p(true future lineage | early cell)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Early cells colored by CIPHER probability of true future lineage")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "early_cells_pca_p_true_lineage.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "early_cells_pca_p_true_lineage.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # 10. FINAL PRINTS
    # ============================================================

    print("\n============================================================")
    print("FINAL CLONE-LEVEL CIPHER SUMMARY")
    print("============================================================")
    print(
        summary_metrics[
            (summary_metrics["level"] == "clone") &
            (summary_metrics["model"] == MAIN_MODEL)
        ][[
            "fate",
            "n_positive_mean",
            "positive_fraction_mean",
            "AUROC_mean",
            "AUROC_sd",
            "AUPRC_mean",
            "AUPRC_sd",
            "top_decile_enrichment_mean",
        ]].sort_values("AUROC_mean", ascending=False)
    )

    print("\n============================================================")
    print("ARGMAX ACCURACY")
    print("============================================================")
    print(acc_df)

    print("\n============================================================")
    print("PAIRED AUROC DELTAS")
    print("============================================================")
    print(pivot_auc)

    print("\n============================================================")
    print("TOP POSITIVE CIPHER FORCE GENES")
    print("============================================================")

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(20)
        )
        print(f"\n{fate}")
        print(", ".join(sub["gene"].astype(str).tolist()))

    print("\nDone. Outputs in:", OUTDIR)



def test_u_predicts_future_fate():
    global os, gzip, warnings, np, pd, plt, sns, mmread, \
        issparse, StratifiedKFold, LogisticRegression, roc_auc_score, average_precision_score, confusion_matrix, PCA, OUTDIR, \
        COUNTS_PATH, GENES_PATH, CLONE_PATH, META_PATH, TIME_COL, CELLTYPE_COL, WELL_COL, START_COL, \
        EARLY_TIME, EARLY_CELLTYPE, EARLY_WELL, TERMINAL_TIME, TERMINAL_WELL, EXCLUDE_FATES, CLONE_FILTER_GRID, MIN_CLONES_PER_FATE, \
        MAX_FATES, N_VAR_GENES, MAX_COV_CELLS, RIDGE, COV_SHRINK_TO_DIAG, N_NULLS, N_SPLITS, SEED, \
        rng, safe_name, barplot_sd, mask_early_cells, mask_terminal_cells, get_cell_to_clone, get_cells_x_genes, zscore_train, \
        fate_entropy_from_counts, select_hvgs_sparse, make_covariance, clone_mean_matrix, fit_score_calibrator, compute_metrics, build_clone_table_with_filters, choose_clone_table, \
        make_cipher_model, score_matrix_with_model, rows_from_scores, counts, f, gene_names, clone_mat, meta, \
        cell_to_clone, has_clone, fate_labels, early_all_mask, early_cloned_mask, terminal_cloned_mask, early_all_idx, early_cloned_idx, \
        terminal_cloned_idx, clone_table_all, selected_fates, chosen_filter, tried_filters, name, n_clones, n_fates, \
        fates, clone_table, eligible_clones, eligible_early_mask, eligible_early_idx, clone_to_fate, clone_to_frac, clone_to_n_total, \
        clone_to_n_early, clone_to_n_terminal, fig, axes, start_rows, cid, early_cells, start_counts, \
        start_df, tab, hvg_idx, gene_vars, hvg_genes, cov_idx, Xcov_raw, mu_ref, \
        sd_ref, Xcov, Sigma, evals, evecs, X_clones, y_clones, min_class_n, \
        n_splits, splitter, all_cell_rows, all_clone_rows, force_rows, null_metric_rows, fold, train_pos, \
        test_pos, train_clones, test_clones, Xtrain_clone, train_clone_ids_used, n_train_early, y_train, Xtest_clone, \
        test_clone_ids_used, n_test_early, y_test_clone, base_clone, test_early_idx, Xtest_cell_raw, Xtest_cell, test_cell_clone_ids, \
        y_test_cell, base_cell, cipher_model, level_name, Xscore, base_df, collector, raw_scores, \
        scaled_scores, p_ovr, p_norm, rows, U, DELTAS, j, fate, \
        u, delta, top_pos, top_neg, rank, gi, null_id, y_train_shuffled, \
        null_model, tmp, m, early_cell_probs, clone_probs, force_df, null_metrics, metric_rows, \
        df_clone, df_cell, m_clone, m_cell, cipher_metrics, all_metrics, summary_metrics, p_rows, \
        level, metric_name, real_vals, null_vals, p_emp, real_mean, null_mean, pvals, \
        acc_rows, df, acc_df, sub, sub_points, handles, labels, uniq, \
        uniq_labels, h, l, p_plot, main_clone, p_norm_cols, mean_prob, cm, \
        cm_norm, score_rows, col, score_df, cipher_force, mean_force, top_genes, TOP_GENES_PER_FATE, \
        heat, MAX_PLOT_CELLS, main_cell, plot_df, plot_cells, X_plot, Z, sc
    # ============================================================
    # CIPHER-LARRY: test whether CIPHER u predicts future fate
    # ============================================================
    # ONLY tests:
    #   1. CIPHER u_f = Sigma_early^{-1} Delta_f
    #   2. shuffled-label null u_f^null = Sigma_early^{-1} Delta_f^shuffled
    #
    # No direct baseline.
    # No diagonal baseline.
    #
    # Main question:
    #   Does the CIPHER-inferred early fate force u_f predict held-out
    #   future clone fate better than shuffled-label null forces?
    #
    # ============================================================

    import os, gzip, warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.io import mmread
    from scipy.sparse import issparse
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    from sklearn.decomposition import PCA

    warnings.filterwarnings("ignore")

    # ============================================================
    # 0. CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_cipher_vs_null_only")
    os.makedirs(OUTDIR, exist_ok=True)

    COUNTS_PATH = os.path.join(SUPPL, "stateFate_inVitro_normed_counts.mtx.gz")
    GENES_PATH  = os.path.join(SUPPL, "stateFate_inVitro_gene_names.txt.gz")
    CLONE_PATH  = os.path.join(SUPPL, "stateFate_inVitro_clone_matrix.mtx.gz")
    META_PATH   = os.path.join(SUPPL, "stateFate_inVitro_metadata.txt.gz")

    TIME_COL = "Time point"
    CELLTYPE_COL = "Cell type annotation"
    WELL_COL = "Well"
    START_COL = "Starting population"

    # Prediction horizon.
    # Set EARLY_TIME = 2.0 for earlier prospective prediction.
    # Set EARLY_TIME = 4.0 for easier/later prediction.
    EARLY_TIME = 4.0
    EARLY_CELLTYPE = "Undifferentiated"
    EARLY_WELL = None

    TERMINAL_TIME = 6.0
    TERMINAL_WELL = None

    EXCLUDE_FATES = {
        "Undifferentiated",
        "Unknown",
        "unknown",
        "nan",
        "NaN",
        "Ambiguous",
        "ambiguous",
        "None",
        "",
    }

    # Strict terminal fate labels, lenient early requirement.
    CLONE_FILTER_GRID = [
        dict(
            name="strict_terminal_lenient_early",
            min_total=12,
            min_early=1,
            min_terminal=8,
            min_dom_count=6,
            min_dom_frac=0.85,
            max_entropy=0.65,
        ),
        dict(
            name="medium_terminal_lenient_early",
            min_total=10,
            min_early=1,
            min_terminal=5,
            min_dom_count=4,
            min_dom_frac=0.80,
            max_entropy=0.75,
        ),
        dict(
            name="lenient_terminal_still_qc",
            min_total=8,
            min_early=1,
            min_terminal=4,
            min_dom_count=3,
            min_dom_frac=0.75,
            max_entropy=0.85,
        ),
    ]

    # Avoid single/few-clone fake-perfect fates.
    MIN_CLONES_PER_FATE = 2
    MAX_FATES = 7

    # Expression/covariance settings.
    N_VAR_GENES = 500
    MAX_COV_CELLS = 50000

    # CIPHER regularization.
    RIDGE = 0.0
    COV_SHRINK_TO_DIAG = 0.0

    # Nulls.
    N_NULLS = 100

    # CV.
    N_SPLITS = 2
    SEED = 0
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    plt.rcParams.update({"font.size": 14})
    sns.set_context("talk")

    # ============================================================
    # 1. HELPERS
    # ============================================================


    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )


    def barplot_sd(*args, **kwargs):
        try:
            return sns.barplot(*args, errorbar="sd", **kwargs)
        except Exception:
            return sns.barplot(*args, ci="sd", **kwargs)

    def mask_early_cells(meta):
        m = meta[TIME_COL].astype(float).values == float(EARLY_TIME)
        if EARLY_CELLTYPE is not None:
            m &= meta[CELLTYPE_COL].astype(str).values == str(EARLY_CELLTYPE)
        if EARLY_WELL is not None and WELL_COL in meta.columns:
            m &= meta[WELL_COL].astype(float).values == float(EARLY_WELL)
        return m

    def mask_terminal_cells(meta):
        m = meta[TIME_COL].astype(float).values == float(TERMINAL_TIME)
        if TERMINAL_WELL is not None and WELL_COL in meta.columns:
            m &= meta[WELL_COL].astype(float).values == float(TERMINAL_WELL)
        ann = meta[CELLTYPE_COL].astype(str).values
        m &= ~np.isin(ann, list(EXCLUDE_FATES))
        return m

    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()
        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row
        return cell_to_clone

    def get_cells_x_genes(counts, cell_idx, gene_idx):
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)

    def zscore_train(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu, sd


    def fate_entropy_from_counts(counts_vec):
        counts_vec = np.asarray(counts_vec, dtype=float)
        counts_vec = counts_vec[counts_vec > 0]
        if counts_vec.size <= 1:
            return 0.0
        p = counts_vec / counts_vec.sum()
        return float(-(p * np.log(p)).sum())

    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        X = counts[:, cell_idx]
        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means**2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]

        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_var_genes:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_

    def make_covariance(X):
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D
        Sigma = Sigma + RIDGE * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)

    def clone_mean_matrix(clone_ids, early_mask, cell_to_clone, counts, hvg_idx, mu, sd):
        rows = []
        out_ids = []
        out_n = []

        for cid in clone_ids:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]
            if len(idx) == 0:
                continue

            X = get_cells_x_genes(counts, idx, hvg_idx)
            X = apply_zscore(X, mu, sd)

            rows.append(X.mean(axis=0))
            out_ids.append(cid)
            out_n.append(len(idx))

        if len(rows) == 0:
            return np.empty((0, len(hvg_idx))), np.array([], dtype=int), np.array([], dtype=int)

        return np.vstack(rows), np.asarray(out_ids, dtype=int), np.asarray(out_n, dtype=int)

    def fit_score_calibrator(scores_train, y_train):
        scores_train = np.asarray(scores_train).reshape(-1, 1)
        y_train = np.asarray(y_train).astype(int)

        if len(np.unique(y_train)) < 2:
            return None

        clf = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        )
        clf.fit(scores_train, y_train)
        return clf


    def compute_metrics(df, selected_fates, prob_prefix="p_ovr", label_col="true_future_lineage"):
        rows = []

        for fate in selected_fates:
            col = f"{prob_prefix}__{safe_name(fate)}"
            y = (df[label_col].astype(str).values == str(fate)).astype(int)
            s = df[col].values.astype(float)

            if len(np.unique(y)) < 2:
                auroc = np.nan
                auprc = np.nan
            else:
                auroc = roc_auc_score(y, s)
                auprc = average_precision_score(y, s)

            baseline = y.mean()
            cutoff = np.quantile(s, 0.90)
            top = s >= cutoff

            if top.sum() > 0 and baseline > 0:
                top_rate = y[top].mean()
                enrichment = top_rate / baseline
            else:
                top_rate = np.nan
                enrichment = np.nan

            rows.append({
                "fate": fate,
                "n": len(y),
                "n_positive": int(y.sum()),
                "positive_fraction": float(baseline),
                "AUROC": auroc,
                "AUPRC": auprc,
                "top_decile_positive_rate": top_rate,
                "top_decile_enrichment": enrichment,
            })

        return pd.DataFrame(rows)

    def build_clone_table_with_filters(clone_mat, early_mask, terminal_mask, fate_labels, config):
        records = []

        for clone_id in range(clone_mat.shape[0]):
            cells = clone_mat[clone_id].indices

            if len(cells) < config["min_total"]:
                continue

            early_cells = cells[early_mask[cells]]
            terminal_cells = cells[terminal_mask[cells]]

            if len(early_cells) < config["min_early"]:
                continue
            if len(terminal_cells) < config["min_terminal"]:
                continue

            terminal_fates = pd.Series(fate_labels[terminal_cells].astype(str))
            terminal_fates = terminal_fates[~terminal_fates.isin(EXCLUDE_FATES)]

            if len(terminal_fates) < config["min_terminal"]:
                continue

            fate_counts = terminal_fates.value_counts()
            if len(fate_counts) == 0:
                continue

            dominant_fate = str(fate_counts.index[0])
            dominant_count = int(fate_counts.iloc[0])
            total_terminal = int(fate_counts.sum())
            dominant_frac = dominant_count / max(total_terminal, 1)
            entropy = fate_entropy_from_counts(fate_counts.values)

            if dominant_count < config["min_dom_count"]:
                continue
            if dominant_frac < config["min_dom_frac"]:
                continue
            if config["max_entropy"] is not None and entropy > config["max_entropy"]:
                continue

            rec = {
                "clone_id": int(clone_id),
                "n_total_clone_cells": int(len(cells)),
                "n_early": int(len(early_cells)),
                "n_terminal": int(total_terminal),
                "n_terminal_raw": int(len(terminal_cells)),
                "n_terminal_fate_types": int(len(fate_counts)),
                "dominant_fate": dominant_fate,
                "dominant_count": dominant_count,
                "dominant_frac": float(dominant_frac),
                "fate_entropy": float(entropy),
                "filter_config": config["name"],
            }

            for fate, count in fate_counts.items():
                s = safe_name(fate)
                rec[f"terminal_count__{s}"] = int(count)
                rec[f"terminal_frac__{s}"] = float(count / total_terminal)

            records.append(rec)

        return pd.DataFrame(records)

    def choose_clone_table():
        tried = []

        for cfg in CLONE_FILTER_GRID:
            ct = build_clone_table_with_filters(
                clone_mat=clone_mat,
                early_mask=early_cloned_mask,
                terminal_mask=terminal_cloned_mask,
                fate_labels=fate_labels,
                config=cfg,
            )

            if ct.empty:
                tried.append((cfg["name"], 0, 0, []))
                continue

            fate_counts = ct["dominant_fate"].value_counts()
            selected = fate_counts[fate_counts >= MIN_CLONES_PER_FATE].index.tolist()
            selected = selected[:MAX_FATES]

            tried.append((cfg["name"], len(ct), len(selected), selected))

            if len(selected) >= 2:
                return ct, selected, cfg, tried

        raise RuntimeError(
            "No clone-filter setting produced at least 2 fates with enough clones. "
            "Lower MIN_CLONES_PER_FATE or relax terminal QC, but do not use tiny single-clone fates."
        )

    def make_cipher_model(
        Xtrain_clone,
        y_train_for_delta,
        selected_fates,
        evals,
        evecs,
        y_train_for_calibrator=None,
    ):
        """
        Build CIPHER model:
            Delta_f = mean X of clones labeled f - mean X of clones not f
            u_f = Sigma^{-1} Delta_f

        For real CIPHER:
            y_train_for_delta = true train labels
            y_train_for_calibrator = true train labels

        For shuffled null:
            y_train_for_delta = shuffled train labels
            y_train_for_calibrator = shuffled train labels
        """
        if y_train_for_calibrator is None:
            y_train_for_calibrator = y_train_for_delta

        U = []
        DELTAS = []
        calibrators = []
        score_mu = []
        score_sd = []

        for fate in selected_fates:
            pos = y_train_for_delta == fate
            neg = y_train_for_delta != fate

            if pos.sum() == 0 or neg.sum() == 0:
                raise RuntimeError(f"Missing pos/neg training clones for fate {fate}")

            delta = Xtrain_clone[pos].mean(axis=0) - Xtrain_clone[neg].mean(axis=0)
            u = evecs @ ((evecs.T @ delta) / evals)

            y_binary_cal = (y_train_for_calibrator == fate).astype(int)

            train_scores = Xtrain_clone @ u
            smu = train_scores.mean()
            ssd = train_scores.std()

            if ssd < 1e-6:
                ssd = 1.0

            train_scores_scaled = (train_scores - smu) / ssd
            clf = fit_score_calibrator(train_scores_scaled, y_binary_cal)

            U.append(u)
            DELTAS.append(delta)
            calibrators.append(clf)
            score_mu.append(smu)
            score_sd.append(ssd)

        return {
            "U": np.asarray(U),
            "DELTAS": np.asarray(DELTAS),
            "calibrators": calibrators,
            "score_mu": np.asarray(score_mu),
            "score_sd": np.asarray(score_sd),
        }

    def score_matrix_with_model(X, model):
        U = model["U"]
        raw_scores = X @ U.T

        scaled_scores = (
            raw_scores - model["score_mu"][None, :]
        ) / model["score_sd"][None, :]

        p_ovr = np.zeros_like(scaled_scores, dtype=float)

        for j, clf in enumerate(model["calibrators"]):
            p_ovr[:, j] = calibrate_scores(clf, scaled_scores[:, j])

        p_ovr = np.nan_to_num(p_ovr, nan=0.0, posinf=1.0, neginf=0.0)
        p_ovr = np.clip(p_ovr, 1e-6, 1.0 - 1e-6)
        p_norm = soft_normalize(p_ovr)

        return raw_scores, scaled_scores, p_ovr, p_norm

    def rows_from_scores(base_df, model_name, raw_scores, scaled_scores, p_ovr, p_norm):
        pred_idx = np.argmax(p_norm, axis=1)
        pred_fates = np.array(selected_fates, dtype=object)[pred_idx]

        rows = base_df.copy()
        rows["model"] = model_name
        rows["predicted_lineage_norm"] = pred_fates
        rows["max_pseudoprob_norm"] = p_norm.max(axis=1)

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            rows[f"score_raw__{s}"] = raw_scores[:, j]
            rows[f"score_scaled__{s}"] = scaled_scores[:, j]
            rows[f"p_ovr__{s}"] = p_ovr[:, j]
            rows[f"p_norm__{s}"] = p_norm[:, j]

        rows["p_ovr_true_future_lineage"] = [
            p_ovr[i, selected_fates.index(tf)]
            for i, tf in enumerate(rows["true_future_lineage"].values)
        ]

        rows["p_norm_true_future_lineage"] = [
            p_norm[i, selected_fates.index(tf)]
            for i, tf in enumerate(rows["true_future_lineage"].values)
        ]

        return rows

    # ============================================================
    # 2. LOAD DATA
    # ============================================================

    counts = mmread(COUNTS_PATH).T.tocsr()
    print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} cells | nnz={counts.nnz:,}")

    with gzip.open(GENES_PATH, "rt") as f:
        gene_names = np.array([line.strip() for line in f])
    print(f"Genes loaded: {len(gene_names)}")

    clone_mat = mmread(CLONE_PATH).T.tocsr()
    print(f"Clone matrix: {clone_mat.shape[0]} clones x {clone_mat.shape[1]} cells")
    print(f"% cells with clone label: {(clone_mat.sum(axis=0) > 0).mean() * 100:.2f}%")

    meta = pd.read_csv(META_PATH, sep="\t")
    print(f"Meta: {meta.shape[0]} rows x {meta.shape[1]} cols")
    print("Meta columns:", list(meta.columns))

    assert counts.shape[1] == meta.shape[0] == clone_mat.shape[1], "cells mismatch"
    assert counts.shape[0] == len(gene_names), "genes mismatch"

    meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")

    print("\nTimepoints:")
    print(np.sort(meta[TIME_COL].dropna().unique()))

    print("\nCell annotations:")
    print(meta[CELLTYPE_COL].value_counts())

    cell_to_clone = get_cell_to_clone(clone_mat)
    has_clone = cell_to_clone >= 0
    fate_labels = meta[CELLTYPE_COL].astype(str).values

    # ============================================================
    # 3. DEFINE EARLY / TERMINAL CELLS
    # ============================================================

    early_all_mask = mask_early_cells(meta)
    early_cloned_mask = early_all_mask & has_clone
    terminal_cloned_mask = mask_terminal_cells(meta) & has_clone

    early_all_idx = np.where(early_all_mask)[0]
    early_cloned_idx = np.where(early_cloned_mask)[0]
    terminal_cloned_idx = np.where(terminal_cloned_mask)[0]

    print(f"\nAll early/precommitted cells for Sigma: {len(early_all_idx):,}")
    print(f"Cloned early/precommitted cells: {len(early_cloned_idx):,}")
    print(f"Cloned terminal cells: {len(terminal_cloned_idx):,}")

    if len(early_all_idx) == 0:
        raise RuntimeError("No early cells found. Check EARLY_TIME / EARLY_CELLTYPE / EARLY_WELL.")

    if len(terminal_cloned_idx) == 0:
        raise RuntimeError("No terminal cloned cells found. Check TERMINAL_TIME / TERMINAL_WELL.")

    # ============================================================
    # 4. CLONE TABLE / FUTURE FATE LABELS
    # ============================================================

    clone_table_all, selected_fates, chosen_filter, tried_filters = choose_clone_table()

    print("\nClone-filter attempts:")
    for name, n_clones, n_fates, fates in tried_filters:
        print(f"  {name:28s} n_clones={n_clones:4d} n_fates={n_fates} fates={fates}")

    print("\nUsing clone filter:")
    print(chosen_filter)

    print("\nClone table after chosen clone QC:")
    print(f"n clones passing filters: {len(clone_table_all):,}")
    print(clone_table_all["dominant_fate"].value_counts())

    clone_table = clone_table_all[
        clone_table_all["dominant_fate"].isin(selected_fates)
    ].copy()

    eligible_clones = clone_table["clone_id"].values.astype(int)
    eligible_early_mask = early_cloned_mask & np.isin(cell_to_clone, eligible_clones)
    eligible_early_idx = np.where(eligible_early_mask)[0]

    clone_to_fate = dict(zip(clone_table["clone_id"], clone_table["dominant_fate"]))
    clone_to_frac = dict(zip(clone_table["clone_id"], clone_table["dominant_frac"]))
    clone_to_n_total = dict(zip(clone_table["clone_id"], clone_table["n_total_clone_cells"]))
    clone_to_n_early = dict(zip(clone_table["clone_id"], clone_table["n_early"]))
    clone_to_n_terminal = dict(zip(clone_table["clone_id"], clone_table["n_terminal"]))

    print("\nSelected fates:")
    print(clone_table["dominant_fate"].value_counts())

    print(f"\nEligible clones: {len(eligible_clones):,}")
    print(f"Eligible early cells: {len(eligible_early_idx):,}")

    clone_table_all.to_csv(os.path.join(OUTDIR, "clone_table_all_passing_qc.csv"), index=False)
    clone_table.to_csv(os.path.join(OUTDIR, "clone_table_selected_fates.csv"), index=False)

    # ============================================================
    # 5. CLONE QC PLOTS
    # ============================================================

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    sns.countplot(data=clone_table, x="dominant_fate", order=selected_fates, ax=axes[0, 0])
    axes[0, 0].set_title("Selected high-confidence clones per future fate")
    axes[0, 0].set_xlabel("future fate")
    axes[0, 0].set_ylabel("clone count")
    axes[0, 0].tick_params(axis="x", rotation=45)

    sns.histplot(data=clone_table, x="n_total_clone_cells", bins=40, ax=axes[0, 1])
    axes[0, 1].set_title("Total cells per retained clone")
    axes[0, 1].set_xlabel("total clone size")

    sns.histplot(data=clone_table, x="n_early", bins=30, ax=axes[0, 2])
    axes[0, 2].set_title("Early cells per retained clone")
    axes[0, 2].set_xlabel("early cells per clone")

    sns.histplot(data=clone_table, x="n_terminal", bins=40, ax=axes[1, 0])
    axes[1, 0].set_title("Terminal cells per retained clone")
    axes[1, 0].set_xlabel("terminal cells per clone")

    sns.scatterplot(
        data=clone_table,
        x="n_terminal",
        y="dominant_frac",
        hue="dominant_fate",
        hue_order=selected_fates,
        ax=axes[1, 1],
        s=45,
    )
    axes[1, 1].set_title("Clone purity vs terminal clone size")
    axes[1, 1].set_xlabel("terminal cells")
    axes[1, 1].set_ylabel("dominant fate fraction")
    axes[1, 1].legend(fontsize=9, frameon=False)

    sns.scatterplot(
        data=clone_table,
        x="n_early",
        y="n_terminal",
        hue="dominant_fate",
        hue_order=selected_fates,
        ax=axes[1, 2],
        s=45,
    )
    axes[1, 2].set_title("Early vs terminal clone representation")
    axes[1, 2].set_xlabel("early cells")
    axes[1, 2].set_ylabel("terminal cells")
    axes[1, 2].legend(fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "clone_qc_summary.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "clone_qc_summary.svg"), bbox_inches="tight")
    plt.show()

    if START_COL in meta.columns:
        start_rows = []
        for cid in eligible_clones:
            early_cells = np.where(eligible_early_mask & (cell_to_clone == cid))[0]
            if len(early_cells) == 0:
                continue

            start_counts = meta.iloc[early_cells][START_COL].astype(str).value_counts()

            start_rows.append({
                "clone_id": cid,
                "future_fate": clone_to_fate[cid],
                "dominant_starting_population": start_counts.index[0],
                "n_early": len(early_cells),
            })

        start_df = pd.DataFrame(start_rows)
        start_df.to_csv(os.path.join(OUTDIR, "clone_starting_population_summary.csv"), index=False)

        plt.figure(figsize=(10, 5))
        tab = pd.crosstab(start_df["future_fate"], start_df["dominant_starting_population"])
        tab = tab.reindex(selected_fates)
        sns.heatmap(tab, annot=True, fmt="d", cmap="viridis")
        plt.title("Future fate vs early starting population")
        plt.xlabel("dominant starting population among early cells")
        plt.ylabel("future fate")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "future_fate_vs_starting_population.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "future_fate_vs_starting_population.svg"), bbox_inches="tight")
        plt.show()

    # ============================================================
    # 6. GLOBAL EARLY HVGs + GLOBAL EARLY COVARIANCE
    # ============================================================

    print("\nSelecting HVGs from all early/precommitted cells...")

    hvg_idx, gene_vars = select_hvgs_sparse(
        counts=counts,
        cell_idx=early_all_idx,
        n_var_genes=N_VAR_GENES,
    )

    hvg_genes = gene_names[hvg_idx]

    pd.DataFrame({
        "gene": hvg_genes,
        "gene_index": hvg_idx,
        "early_variance": gene_vars[hvg_idx],
    }).to_csv(os.path.join(OUTDIR, "selected_early_hvgs.csv"), index=False)

    print(f"Using top {len(hvg_idx)} early-variable genes.")

    cov_idx = early_all_idx.copy()
    if len(cov_idx) > MAX_COV_CELLS:
        cov_idx = rng.choice(cov_idx, size=MAX_COV_CELLS, replace=False)

    print(f"Using {len(cov_idx):,} early cells for global Sigma.")

    Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx)
    mu_ref, sd_ref = zscore_train(Xcov_raw)
    Xcov = apply_zscore(Xcov_raw, mu_ref, sd_ref)

    Sigma = make_covariance(Xcov)

    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-8)

    pd.DataFrame({
        "rank": np.arange(1, len(evals) + 1),
        "eigenvalue": evals[::-1],
    }).to_csv(os.path.join(OUTDIR, "early_covariance_eigenvalues.csv"), index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(evals) + 1), evals[::-1], marker="o", linewidth=1, markersize=3)
    plt.yscale("log")
    plt.xlabel("eigenvalue rank")
    plt.ylabel("eigenvalue")
    plt.title("Early progenitor covariance spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # 7. CROSS-VALIDATED CIPHER VS NULL
    # ============================================================

    X_clones = clone_table["clone_id"].values.astype(int)
    y_clones = clone_table["dominant_fate"].values.astype(str)

    min_class_n = clone_table["dominant_fate"].value_counts().min()
    n_splits = int(min(N_SPLITS, min_class_n))

    if n_splits < 2:
        raise RuntimeError(
            f"Cannot do held-out CV: smallest selected fate has only {min_class_n} clones. "
            "Increase MIN_CLONES_PER_FATE or reduce MAX_FATES."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    print(f"\nUsing clone-level stratified {n_splits}-fold CV.")
    print(f"Running {N_NULLS} shuffled-label nulls per fold.")

    all_cell_rows = []
    all_clone_rows = []
    force_rows = []
    null_metric_rows = []

    for fold, (train_pos, test_pos) in enumerate(splitter.split(X_clones, y_clones)):
        train_clones = X_clones[train_pos]
        test_clones = X_clones[test_pos]

        print(f"\nFold {fold + 1}/{n_splits}: train clones={len(train_clones)}, test clones={len(test_clones)}")

        Xtrain_clone, train_clone_ids_used, n_train_early = clone_mean_matrix(
            clone_ids=train_clones,
            early_mask=eligible_early_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        if Xtrain_clone.shape[0] < 5:
            raise RuntimeError("Too few training clone means.")

        y_train = np.array([clone_to_fate[c] for c in train_clone_ids_used])

        # Held-out clone means.
        Xtest_clone, test_clone_ids_used, n_test_early = clone_mean_matrix(
            clone_ids=test_clones,
            early_mask=eligible_early_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        y_test_clone = np.array([clone_to_fate[c] for c in test_clone_ids_used])

        base_clone = pd.DataFrame({
            "fold": fold,
            "level": "clone",
            "clone_id": test_clone_ids_used,
            "true_future_lineage": y_test_clone,
            "true_future_lineage_frac": [clone_to_frac[c] for c in test_clone_ids_used],
            "n_early_scored": n_test_early,
            "n_total_clone_cells": [clone_to_n_total[c] for c in test_clone_ids_used],
            "n_early_clone_cells": [clone_to_n_early[c] for c in test_clone_ids_used],
            "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_clone_ids_used],
        })

        # Held-out early cells.
        test_early_idx = np.where(eligible_early_mask & np.isin(cell_to_clone, test_clones))[0]
        Xtest_cell_raw = get_cells_x_genes(counts, test_early_idx, hvg_idx)
        Xtest_cell = apply_zscore(Xtest_cell_raw, mu_ref, sd_ref)

        test_cell_clone_ids = cell_to_clone[test_early_idx]
        y_test_cell = np.array([clone_to_fate[c] for c in test_cell_clone_ids])

        base_cell = pd.DataFrame({
            "fold": fold,
            "level": "cell",
            "cell_index": test_early_idx,
            "clone_id": test_cell_clone_ids,
            "true_future_lineage": y_test_cell,
            "true_future_lineage_frac": [clone_to_frac[c] for c in test_cell_clone_ids],
            "n_total_clone_cells": [clone_to_n_total[c] for c in test_cell_clone_ids],
            "n_early_clone_cells": [clone_to_n_early[c] for c in test_cell_clone_ids],
            "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_cell_clone_ids],
        })

        # --------------------------
        # Real CIPHER model.
        # --------------------------
        cipher_model = make_cipher_model(
            Xtrain_clone=Xtrain_clone,
            y_train_for_delta=y_train,
            selected_fates=selected_fates,
            evals=evals,
            evecs=evecs,
            y_train_for_calibrator=y_train,
        )

        for level_name, Xscore, base_df, collector in [
            ("clone", Xtest_clone, base_clone, all_clone_rows),
            ("cell", Xtest_cell, base_cell, all_cell_rows),
        ]:
            raw_scores, scaled_scores, p_ovr, p_norm = score_matrix_with_model(Xscore, cipher_model)
            rows = rows_from_scores(base_df, "cipher", raw_scores, scaled_scores, p_ovr, p_norm)
            collector.append(rows)

        # Save top CIPHER genes.
        U = cipher_model["U"]
        DELTAS = cipher_model["DELTAS"]

        for j, fate in enumerate(selected_fates):
            u = U[j]
            delta = DELTAS[j]

            top_pos = np.argsort(u)[::-1][:50]
            top_neg = np.argsort(u)[:50]

            for rank, gi in enumerate(top_pos, start=1):
                force_rows.append({
                    "fold": fold,
                    "model": "cipher",
                    "fate": fate,
                    "direction": "positive",
                    "rank": rank,
                    "gene": hvg_genes[gi],
                    "gene_index": int(hvg_idx[gi]),
                    "u": float(u[gi]),
                    "delta_early": float(delta[gi]),
                })

            for rank, gi in enumerate(top_neg, start=1):
                force_rows.append({
                    "fold": fold,
                    "model": "cipher",
                    "fate": fate,
                    "direction": "negative",
                    "rank": rank,
                    "gene": hvg_genes[gi],
                    "gene_index": int(hvg_idx[gi]),
                    "u": float(u[gi]),
                    "delta_early": float(delta[gi]),
                })

        # --------------------------
        # Shuffled-label nulls.
        # --------------------------
        for null_id in range(N_NULLS):
            y_train_shuffled = y_train.copy()
            rng.shuffle(y_train_shuffled)

            null_model = make_cipher_model(
                Xtrain_clone=Xtrain_clone,
                y_train_for_delta=y_train_shuffled,
                selected_fates=selected_fates,
                evals=evals,
                evecs=evecs,
                y_train_for_calibrator=y_train_shuffled,
            )

            for level_name, Xscore, base_df in [
                ("clone", Xtest_clone, base_clone),
                ("cell", Xtest_cell, base_cell),
            ]:
                raw_scores, scaled_scores, p_ovr, p_norm = score_matrix_with_model(Xscore, null_model)
                tmp = rows_from_scores(base_df, "shuffled_null", raw_scores, scaled_scores, p_ovr, p_norm)

                m = compute_metrics(tmp, selected_fates)
                m["model"] = "shuffled_null"
                m["fold"] = fold
                m["null_id"] = null_id
                m["level"] = level_name

                null_metric_rows.append(m)

    early_cell_probs = pd.concat(all_cell_rows, ignore_index=True)
    clone_probs = pd.concat(all_clone_rows, ignore_index=True)
    force_df = pd.DataFrame(force_rows)
    null_metrics = pd.concat(null_metric_rows, ignore_index=True)

    early_cell_probs.to_csv(os.path.join(OUTDIR, "early_cell_cipher_probs.csv"), index=False)
    clone_probs.to_csv(os.path.join(OUTDIR, "clone_cipher_probs.csv"), index=False)
    force_df.to_csv(os.path.join(OUTDIR, "cipher_top_force_genes.csv"), index=False)
    null_metrics.to_csv(os.path.join(OUTDIR, "shuffled_null_metrics.csv"), index=False)

    print("\nSaved core outputs:")
    print(os.path.join(OUTDIR, "early_cell_cipher_probs.csv"))
    print(os.path.join(OUTDIR, "clone_cipher_probs.csv"))
    print(os.path.join(OUTDIR, "cipher_top_force_genes.csv"))
    print(os.path.join(OUTDIR, "shuffled_null_metrics.csv"))

    # ============================================================
    # 8. REAL CIPHER METRICS + EMPIRICAL NULL P-VALUES
    # ============================================================

    metric_rows = []

    for fold in sorted(clone_probs["fold"].unique()):
        df_clone = clone_probs[clone_probs["fold"] == fold].copy()
        df_cell = early_cell_probs[early_cell_probs["fold"] == fold].copy()

        m_clone = compute_metrics(df_clone, selected_fates)
        m_clone["model"] = "cipher"
        m_clone["fold"] = fold
        m_clone["level"] = "clone"

        m_cell = compute_metrics(df_cell, selected_fates)
        m_cell["model"] = "cipher"
        m_cell["fold"] = fold
        m_cell["level"] = "cell"

        metric_rows.append(m_clone)
        metric_rows.append(m_cell)

    cipher_metrics = pd.concat(metric_rows, ignore_index=True)
    cipher_metrics.to_csv(os.path.join(OUTDIR, "cipher_prediction_metrics_by_fold.csv"), index=False)

    all_metrics = pd.concat([cipher_metrics, null_metrics], ignore_index=True)
    all_metrics.to_csv(os.path.join(OUTDIR, "cipher_vs_null_metrics_all.csv"), index=False)

    summary_metrics = (
        all_metrics
        .groupby(["level", "model", "fate"], as_index=False)
        .agg(
            AUROC_mean=("AUROC", "mean"),
            AUROC_sd=("AUROC", "std"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_sd=("AUPRC", "std"),
            top_decile_enrichment_mean=("top_decile_enrichment", "mean"),
            top_decile_enrichment_sd=("top_decile_enrichment", "std"),
            n_positive_mean=("n_positive", "mean"),
            positive_fraction_mean=("positive_fraction", "mean"),
        )
    )

    summary_metrics.to_csv(os.path.join(OUTDIR, "cipher_vs_null_metrics_summary.csv"), index=False)

    # Empirical p-values.
    p_rows = []

    for level in ["clone", "cell"]:
        for fate in selected_fates:
            for metric_name in ["AUROC", "AUPRC", "top_decile_enrichment"]:
                real_vals = cipher_metrics[
                    (cipher_metrics["level"] == level) &
                    (cipher_metrics["fate"] == fate)
                ][metric_name].dropna().values

                null_vals = null_metrics[
                    (null_metrics["level"] == level) &
                    (null_metrics["fate"] == fate)
                ][metric_name].dropna().values

                if len(real_vals) == 0 or len(null_vals) == 0:
                    p_emp = np.nan
                    real_mean = np.nan
                    null_mean = np.nan
                else:
                    real_mean = float(np.mean(real_vals))
                    null_mean = float(np.mean(null_vals))
                    p_emp = float((1 + np.sum(null_vals >= real_mean)) / (1 + len(null_vals)))

                p_rows.append({
                    "level": level,
                    "fate": fate,
                    "metric": metric_name,
                    "cipher_mean": real_mean,
                    "null_mean": null_mean,
                    "empirical_p": p_emp,
                    "n_null": len(null_vals),
                })

    pvals = pd.DataFrame(p_rows)
    pvals.to_csv(os.path.join(OUTDIR, "cipher_vs_null_empirical_pvalues.csv"), index=False)

    # Accuracy summary for CIPHER only.
    acc_rows = []

    for level_name, df in [("clone", clone_probs), ("cell", early_cell_probs)]:
        acc_rows.append({
            "model": "cipher",
            "level": level_name,
            "argmax_accuracy": np.mean(df["predicted_lineage_norm"] == df["true_future_lineage"]),
            "mean_p_true_ovr": df["p_ovr_true_future_lineage"].mean(),
            "mean_p_true_norm": df["p_norm_true_future_lineage"].mean(),
        })

    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(os.path.join(OUTDIR, "cipher_argmax_accuracy_summary.csv"), index=False)

    print("\nClone-level CIPHER vs null summary:")
    print(
        summary_metrics[
            (summary_metrics["level"] == "clone")
        ][[
            "model",
            "fate",
            "n_positive_mean",
            "positive_fraction_mean",
            "AUROC_mean",
            "AUROC_sd",
            "AUPRC_mean",
            "AUPRC_sd",
            "top_decile_enrichment_mean",
        ]]
    )

    print("\nEmpirical p-values, clone AUROC:")
    print(
        pvals[
            (pvals["level"] == "clone") &
            (pvals["metric"] == "AUROC")
        ][["fate", "cipher_mean", "null_mean", "empirical_p", "n_null"]]
    )

    print("\nArgmax summary:")
    print(acc_df)

    # ============================================================
    # 9. PLOTS: CIPHER VS NULL
    # ============================================================

    # --------------------------
    # AUROC / AUPRC / enrichment distributions.
    # --------------------------
    for metric_name in ["AUROC", "AUPRC", "top_decile_enrichment"]:
        plt.figure(figsize=(11, 5))

        sub = all_metrics[all_metrics["level"] == "clone"].copy()
        sub["model_label"] = sub["model"].map({
            "cipher": "CIPHER",
            "shuffled_null": "shuffled null",
        })

        sns.boxplot(
            data=sub,
            x="fate",
            y=metric_name,
            hue="model_label",
            order=selected_fates,
            showfliers=False,
        )

        # Plot CIPHER fold points only.
        sub_points = sub[sub["model"] == "cipher"].copy()
        sns.stripplot(
            data=sub_points,
            x="fate",
            y=metric_name,
            hue="model_label",
            order=selected_fates,
            dodge=True,
            color="black",
            alpha=0.6,
            size=4,
            legend=False,
        )

        if metric_name == "AUROC":
            plt.axhline(0.5, color="gray", linestyle="--", linewidth=2)
            plt.ylim(0, 1)
        elif metric_name == "AUPRC":
            plt.ylim(0, 1)
        else:
            plt.axhline(1.0, color="gray", linestyle="--", linewidth=2)

        plt.title(f"CIPHER vs shuffled-label null: clone-level {metric_name}")
        plt.xlabel("future lineage")
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha="right")

        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = []
        uniq_labels = []
        for h, l in zip(handles, labels):
            if l not in uniq_labels:
                uniq.append(h)
                uniq_labels.append(l)
        plt.legend(uniq, uniq_labels, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"cipher_vs_null_clone_{metric_name}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"cipher_vs_null_clone_{metric_name}.svg"), bbox_inches="tight")
        plt.show()

    # --------------------------
    # Empirical p-values.
    # --------------------------
    p_plot = pvals[
        (pvals["level"] == "clone") &
        (pvals["metric"] == "AUROC")
    ].copy()

    p_plot["minus_log10_p"] = -np.log10(np.maximum(p_plot["empirical_p"], 1e-300))

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=p_plot,
        x="fate",
        y="minus_log10_p",
        order=selected_fates,
    )
    plt.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=2, label="p=0.05")
    plt.title("Empirical shuffled-label null p-values")
    plt.xlabel("future lineage")
    plt.ylabel("-log10 empirical p, AUROC")
    plt.xticks(rotation=45, ha="right")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cipher_vs_null_empirical_pvalues_AUROC.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "cipher_vs_null_empirical_pvalues_AUROC.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # CIPHER probability heatmap + confusion matrix.
    # --------------------------
    main_clone = clone_probs.copy()
    p_norm_cols = [f"p_norm__{safe_name(f)}" for f in selected_fates]

    mean_prob = (
        main_clone
        .groupby("true_future_lineage")[p_norm_cols]
        .mean()
        .reindex(selected_fates)
    )
    mean_prob.columns = selected_fates

    cm = confusion_matrix(
        main_clone["true_future_lineage"],
        main_clone["predicted_lineage_norm"],
        labels=selected_fates,
    )

    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        mean_prob,
        ax=axes[0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "mean normalized pseudo-prob"},
    )
    axes[0].set_title("Clone mean CIPHER pseudo-probabilities")
    axes[0].set_xlabel("predicted future lineage")
    axes[0].set_ylabel("true future lineage")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        pd.DataFrame(cm_norm, index=selected_fates, columns=selected_fates),
        ax=axes[1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "row-normalized fraction"},
    )
    axes[1].set_title("Argmax CIPHER prediction")
    axes[1].set_xlabel("predicted future lineage")
    axes[1].set_ylabel("true future lineage")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_probability_heatmap_confusion.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_probability_heatmap_confusion.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # CIPHER p(true lineage).
    # --------------------------
    plt.figure(figsize=(11, 5))
    sns.boxplot(
        data=main_clone,
        x="true_future_lineage",
        y="p_ovr_true_future_lineage",
        order=selected_fates,
        showfliers=False,
    )
    sns.stripplot(
        data=main_clone,
        x="true_future_lineage",
        y="p_ovr_true_future_lineage",
        order=selected_fates,
        color="black",
        alpha=0.35,
        size=3,
    )
    plt.ylim(0, 1)
    plt.title("CIPHER probability assigned to true future lineage")
    plt.xlabel("true future lineage")
    plt.ylabel("p(true future lineage | early clone)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_p_true_lineage_by_fate.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_p_true_lineage_by_fate.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # Future-fate vs other score distributions.
    # --------------------------
    score_rows = []

    for fate in selected_fates:
        col = f"p_ovr__{safe_name(fate)}"
        tmp = main_clone[["clone_id", "true_future_lineage", col]].copy()
        tmp["tested_fate"] = fate
        tmp["is_future_fate"] = np.where(tmp["true_future_lineage"] == fate, "future fate", "other")
        tmp["p_ovr"] = tmp[col]
        score_rows.append(tmp[["clone_id", "tested_fate", "is_future_fate", "p_ovr"]])

    score_df = pd.concat(score_rows, ignore_index=True)

    plt.figure(figsize=(12, 5))
    sns.boxplot(
        data=score_df,
        x="tested_fate",
        y="p_ovr",
        hue="is_future_fate",
        order=selected_fates,
        showfliers=False,
    )
    plt.ylim(0, 1)
    plt.title("CIPHER one-vs-rest scores: future-fate clones vs others")
    plt.xlabel("tested lineage")
    plt.ylabel("one-vs-rest probability")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_positive_vs_rest_scores.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_positive_vs_rest_scores.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # Top CIPHER force genes heatmap.
    # --------------------------
    cipher_force = force_df[force_df["direction"] == "positive"].copy()

    mean_force = (
        cipher_force
        .groupby(["fate", "gene"], as_index=False)
        .agg(mean_u=("u", "mean"), mean_delta=("delta_early", "mean"), mean_rank=("rank", "mean"))
    )

    top_genes = []
    TOP_GENES_PER_FATE = 12

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(TOP_GENES_PER_FATE)
        )
        top_genes.extend(sub["gene"].tolist())

    top_genes = list(dict.fromkeys(top_genes))

    heat = (
        mean_force
        .pivot_table(index="gene", columns="fate", values="mean_u", fill_value=0)
        .reindex(top_genes)
        .reindex(columns=selected_fates)
    )

    plt.figure(figsize=(1.4 * len(selected_fates) + 6, 0.28 * len(top_genes) + 4))
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "mean CIPHER force u"},
    )
    plt.title("Top positive early-bias CIPHER force genes")
    plt.xlabel("future lineage")
    plt.ylabel("gene")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "top_CIPHER_force_genes_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "top_CIPHER_force_genes_heatmap.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # PCA of early cells.
    # --------------------------
    MAX_PLOT_CELLS = 7000
    main_cell = early_cell_probs.copy()

    plot_df = main_cell.copy()
    if len(plot_df) > MAX_PLOT_CELLS:
        plot_df = plot_df.sample(MAX_PLOT_CELLS, random_state=SEED)

    plot_cells = plot_df["cell_index"].values.astype(int)
    X_plot = get_cells_x_genes(counts, plot_cells, hvg_idx)
    X_plot = apply_zscore(X_plot, mu_ref, sd_ref)

    Z = PCA(n_components=2, random_state=SEED).fit_transform(X_plot)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=plot_df["p_ovr_true_future_lineage"].values,
        s=8,
        alpha=0.8,
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    plt.colorbar(sc, label="p(true future lineage | early cell)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Early cells colored by CIPHER probability of true future lineage")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "early_cells_pca_p_true_lineage.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "early_cells_pca_p_true_lineage.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # 10. FINAL PRINTS
    # ============================================================

    print("\n============================================================")
    print("FINAL CLONE-LEVEL CIPHER SUMMARY")
    print("============================================================")
    print(
        summary_metrics[
            (summary_metrics["level"] == "clone") &
            (summary_metrics["model"] == "cipher")
        ][[
            "fate",
            "n_positive_mean",
            "positive_fraction_mean",
            "AUROC_mean",
            "AUROC_sd",
            "AUPRC_mean",
            "AUPRC_sd",
            "top_decile_enrichment_mean",
        ]].sort_values("AUROC_mean", ascending=False)
    )

    print("\n============================================================")
    print("NULL-COMPARISON EMPIRICAL P-VALUES, CLONE AUROC")
    print("============================================================")
    print(
        pvals[
            (pvals["level"] == "clone") &
            (pvals["metric"] == "AUROC")
        ][["fate", "cipher_mean", "null_mean", "empirical_p", "n_null"]]
    )

    print("\n============================================================")
    print("ARGMAX ACCURACY")
    print("============================================================")
    print(acc_df)

    print("\n============================================================")
    print("TOP POSITIVE CIPHER FORCE GENES")
    print("============================================================")

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(20)
        )
        print(f"\n{fate}")
        print(", ".join(sub["gene"].astype(str).tolist()))

    print("\nDone. Outputs in:", OUTDIR)



def cipher_vs_shuffled_gaussian_scoring():
    global os, gzip, warnings, np, pd, plt, sns, mmread, \
        issparse, StratifiedKFold, roc_auc_score, average_precision_score, confusion_matrix, PCA, OUTDIR, COUNTS_PATH, \
        GENES_PATH, CLONE_PATH, META_PATH, TIME_COL, CELLTYPE_COL, WELL_COL, START_COL, EARLY_TIME, \
        EARLY_CELLTYPE, EARLY_WELL, TERMINAL_TIME, TERMINAL_WELL, EXCLUDE_FATES, CLONE_FILTER_GRID, MIN_CLONES_PER_FATE, MAX_FATES, \
        N_VAR_GENES, MAX_COV_CELLS, RIDGE, COV_SHRINK_TO_DIAG, USE_FATE_PRIOR, N_NULLS, N_SPLITS, SEED, \
        rng, safe_name, softmax_logits, barplot_sd, mask_early_cells, mask_terminal_cells, get_cell_to_clone, get_cells_x_genes, \
        zscore_train, fate_entropy_from_counts, select_hvgs_sparse, make_covariance, clone_mean_matrix, compute_metrics, build_clone_table_with_filters, choose_clone_table, \
        make_cipher_model, score_matrix_with_model, rows_from_scores, counts, f, gene_names, clone_mat, meta, \
        cell_to_clone, has_clone, fate_labels, early_all_mask, early_cloned_mask, terminal_cloned_mask, early_all_idx, early_cloned_idx, \
        terminal_cloned_idx, clone_table_all, selected_fates, chosen_filter, tried_filters, name, n_clones, n_fates, \
        fates, clone_table, eligible_clones, eligible_early_mask, eligible_early_idx, clone_to_fate, clone_to_frac, clone_to_n_total, \
        clone_to_n_early, clone_to_n_terminal, fig, axes, start_rows, cid, early_cells, start_counts, \
        start_df, tab, hvg_idx, gene_vars, hvg_genes, cov_idx, Xcov_raw, mu_ref, \
        sd_ref, Xcov, Sigma, evals, evecs, X_clones, y_clones, min_class_n, \
        n_splits, splitter, all_cell_rows, all_clone_rows, force_rows, null_metric_rows, fold, train_pos, \
        test_pos, train_clones, test_clones, Xtrain_clone, train_clone_ids_used, n_train_early, y_train, Xtest_clone, \
        test_clone_ids_used, n_test_early, y_test_clone, base_clone, test_early_idx, Xtest_cell_raw, Xtest_cell, test_cell_clone_ids, \
        y_test_cell, base_cell, cipher_model, Xscore, base_df, collector, raw_scores, log_enrichment, \
        p_norm, rows, U, DELTAS, j, fate, u, delta, \
        top_pos, top_neg, rank, gi, null_id, y_train_shuffled, null_model, level_name, \
        tmp, m, early_cell_probs, clone_probs, force_df, null_metrics, metric_rows, df_clone, \
        df_cell, m_clone, m_cell, cipher_metrics, all_metrics, summary_metrics, p_rows, level, \
        metric_name, real_vals, null_vals, p_emp, real_mean, null_mean, pvals, acc_rows, \
        df, acc_df, sub, sub_points, handles, labels, uniq, uniq_labels, \
        h, l, p_plot, main_clone, p_norm_cols, mean_prob, cm, cm_norm, \
        score_rows, col, score_df, cipher_force, mean_force, top_genes, TOP_GENES_PER_FATE, heat, \
        MAX_PLOT_CELLS, main_cell, plot_df, plot_cells, X_plot, Z, sc
    # ============================================================
    # CIPHER-LARRY: CIPHER vs shuffled-label null only
    # with full Gaussian log-enrichment scoring
    # ============================================================
    #
    # Main test:
    #
    #   Does the CIPHER-inferred early fate force
    #
    #       u_f = Sigma_early^{-1} Delta_f
    #
    #   predict future clone fate better than shuffled-label nulls?
    #
    # Scoring uses the full Gaussian exponential-tilt log-enrichment:
    #
    #       log_enrichment_f(x)
    #       =
    #       u_f^T x
    #       -
    #       1/2 u_f^T Sigma u_f
    #
    # In this code, x is already z-scored early expression, so x is centered.
    #
    # Optional posterior-style prior:
    #
    #       logit_f(x)
    #       =
    #       u_f^T x
    #       -
    #       1/2 u_f^T Sigma u_f
    #       +
    #       log pi_f
    #
    # Set USE_FATE_PRIOR=True to include log pi_f.
    #
    # Outputs:
    #   - clone_cipher_probs.csv
    #   - early_cell_cipher_probs.csv
    #   - cipher_top_force_genes.csv
    #   - shuffled_null_metrics.csv
    #   - cipher_vs_null_metrics_all.csv
    #   - empirical p-value table
    #   - summary figures
    #
    # ============================================================

    import os
    import gzip
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.io import mmread
    from scipy.sparse import issparse
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    from sklearn.decomposition import PCA

    warnings.filterwarnings("ignore")

    # ============================================================
    # 0. CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_cipher_vs_null_full_log_enrichment")
    os.makedirs(OUTDIR, exist_ok=True)

    COUNTS_PATH = os.path.join(SUPPL, "stateFate_inVitro_normed_counts.mtx.gz")
    GENES_PATH  = os.path.join(SUPPL, "stateFate_inVitro_gene_names.txt.gz")
    CLONE_PATH  = os.path.join(SUPPL, "stateFate_inVitro_clone_matrix.mtx.gz")
    META_PATH   = os.path.join(SUPPL, "stateFate_inVitro_metadata.txt.gz")

    TIME_COL = "Time point"
    CELLTYPE_COL = "Cell type annotation"
    WELL_COL = "Well"
    START_COL = "Starting population"

    # Prediction horizon.
    # Use 2.0 for harder, earlier prediction.
    # Use 4.0 for later, easier prediction.
    EARLY_TIME = 4.0
    EARLY_CELLTYPE = "Undifferentiated"
    EARLY_WELL = None

    TERMINAL_TIME = 6.0
    TERMINAL_WELL = None

    EXCLUDE_FATES = {
        "Undifferentiated",
        "Unknown",
        "unknown",
        "nan",
        "NaN",
        "Ambiguous",
        "ambiguous",
        "None",
        "",
    }

    # Strict terminal fate labels, lenient early requirement.
    CLONE_FILTER_GRID = [
        dict(
            name="strict_terminal_lenient_early",
            min_total=12,
            min_early=1,
            min_terminal=8,
            min_dom_count=6,
            min_dom_frac=0.85,
            max_entropy=0.65,
        ),
        dict(
            name="medium_terminal_lenient_early",
            min_total=10,
            min_early=1,
            min_terminal=5,
            min_dom_count=4,
            min_dom_frac=0.80,
            max_entropy=0.75,
        ),
        dict(
            name="lenient_terminal_still_qc",
            min_total=8,
            min_early=1,
            min_terminal=4,
            min_dom_count=3,
            min_dom_frac=0.75,
            max_entropy=0.85,
        ),
    ]

    # Avoid fake-perfect tiny fates.
    MIN_CLONES_PER_FATE = 5
    MAX_FATES = 5

    # Expression/covariance settings.
    N_VAR_GENES = 500
    MAX_COV_CELLS = 50000

    # CIPHER regularization. Do not set these to zero unless you are debugging.
    RIDGE = 0.0
    COV_SHRINK_TO_DIAG = 0.0

    # Whether to add log class prior to the cross-fate softmax logits.
    # False = pure log-enrichment.
    # True  = posterior-like logits.
    USE_FATE_PRIOR = False

    # Null settings.
    N_NULLS = 100

    # CV.
    N_SPLITS = 5
    SEED = 0
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    plt.rcParams.update({"font.size": 14})
    sns.set_context("talk")

    # ============================================================
    # 1. HELPERS
    # ============================================================


    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )

    def softmax_logits(logits, eps=1e-12):
        z = logits - np.max(logits, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.maximum(ez.sum(axis=1, keepdims=True), eps)

    def barplot_sd(*args, **kwargs):
        try:
            return sns.barplot(*args, errorbar="sd", **kwargs)
        except Exception:
            return sns.barplot(*args, ci="sd", **kwargs)

    def mask_early_cells(meta):
        m = meta[TIME_COL].astype(float).values == float(EARLY_TIME)
        if EARLY_CELLTYPE is not None:
            m &= meta[CELLTYPE_COL].astype(str).values == str(EARLY_CELLTYPE)
        if EARLY_WELL is not None and WELL_COL in meta.columns:
            m &= meta[WELL_COL].astype(float).values == float(EARLY_WELL)
        return m

    def mask_terminal_cells(meta):
        m = meta[TIME_COL].astype(float).values == float(TERMINAL_TIME)
        if TERMINAL_WELL is not None and WELL_COL in meta.columns:
            m &= meta[WELL_COL].astype(float).values == float(TERMINAL_WELL)

        ann = meta[CELLTYPE_COL].astype(str).values
        m &= ~np.isin(ann, list(EXCLUDE_FATES))
        return m

    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()
        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row
        return cell_to_clone

    def get_cells_x_genes(counts, cell_idx, gene_idx):
        # counts is genes x cells
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)

    def zscore_train(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu, sd


    def fate_entropy_from_counts(counts_vec):
        counts_vec = np.asarray(counts_vec, dtype=float)
        counts_vec = counts_vec[counts_vec > 0]
        if counts_vec.size <= 1:
            return 0.0
        p = counts_vec / counts_vec.sum()
        return float(-(p * np.log(p)).sum())

    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        X = counts[:, cell_idx]
        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means**2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]

        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_var_genes:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_

    def make_covariance(X):
        """
        X: samples x genes, already z-scored.
        """
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D
        Sigma = Sigma + RIDGE * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)

    def clone_mean_matrix(clone_ids, early_mask, cell_to_clone, counts, hvg_idx, mu, sd):
        """
        Returns clone-balanced early mean matrix.
        """
        rows = []
        out_ids = []
        out_n = []

        for cid in clone_ids:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]
            if len(idx) == 0:
                continue

            X = get_cells_x_genes(counts, idx, hvg_idx)
            X = apply_zscore(X, mu, sd)

            rows.append(X.mean(axis=0))
            out_ids.append(cid)
            out_n.append(len(idx))

        if len(rows) == 0:
            return (
                np.empty((0, len(hvg_idx))),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        return np.vstack(rows), np.asarray(out_ids, dtype=int), np.asarray(out_n, dtype=int)

    def compute_metrics(df, selected_fates, score_prefix="log_enrichment", label_col="true_future_lineage"):
        """
        One-vs-rest metrics using the full log-enrichment score.

        For each fate f:
            positive = future fate f
            score    = log_enrichment_f(x)

        AUROC/AUPRC do not depend on the cross-fate softmax.
        """
        rows = []

        for fate in selected_fates:
            col = f"{score_prefix}__{safe_name(fate)}"

            y = (df[label_col].astype(str).values == str(fate)).astype(int)
            s = df[col].values.astype(float)

            if len(np.unique(y)) < 2:
                auroc = np.nan
                auprc = np.nan
            else:
                auroc = roc_auc_score(y, s)
                auprc = average_precision_score(y, s)

            baseline = y.mean()
            cutoff = np.quantile(s, 0.90)
            top = s >= cutoff

            if top.sum() > 0 and baseline > 0:
                top_rate = y[top].mean()
                enrichment = top_rate / baseline
            else:
                top_rate = np.nan
                enrichment = np.nan

            rows.append({
                "fate": fate,
                "n": len(y),
                "n_positive": int(y.sum()),
                "positive_fraction": float(baseline),
                "AUROC": auroc,
                "AUPRC": auprc,
                "top_decile_positive_rate": top_rate,
                "top_decile_enrichment": enrichment,
            })

        return pd.DataFrame(rows)

    def build_clone_table_with_filters(clone_mat, early_mask, terminal_mask, fate_labels, config):
        records = []

        for clone_id in range(clone_mat.shape[0]):
            cells = clone_mat[clone_id].indices

            if len(cells) < config["min_total"]:
                continue

            early_cells = cells[early_mask[cells]]
            terminal_cells = cells[terminal_mask[cells]]

            if len(early_cells) < config["min_early"]:
                continue
            if len(terminal_cells) < config["min_terminal"]:
                continue

            terminal_fates = pd.Series(fate_labels[terminal_cells].astype(str))
            terminal_fates = terminal_fates[~terminal_fates.isin(EXCLUDE_FATES)]

            if len(terminal_fates) < config["min_terminal"]:
                continue

            fate_counts = terminal_fates.value_counts()
            if len(fate_counts) == 0:
                continue

            dominant_fate = str(fate_counts.index[0])
            dominant_count = int(fate_counts.iloc[0])
            total_terminal = int(fate_counts.sum())
            dominant_frac = dominant_count / max(total_terminal, 1)
            entropy = fate_entropy_from_counts(fate_counts.values)

            if dominant_count < config["min_dom_count"]:
                continue
            if dominant_frac < config["min_dom_frac"]:
                continue
            if config["max_entropy"] is not None and entropy > config["max_entropy"]:
                continue

            rec = {
                "clone_id": int(clone_id),
                "n_total_clone_cells": int(len(cells)),
                "n_early": int(len(early_cells)),
                "n_terminal": int(total_terminal),
                "n_terminal_raw": int(len(terminal_cells)),
                "n_terminal_fate_types": int(len(fate_counts)),
                "dominant_fate": dominant_fate,
                "dominant_count": dominant_count,
                "dominant_frac": float(dominant_frac),
                "fate_entropy": float(entropy),
                "filter_config": config["name"],
            }

            for fate, count in fate_counts.items():
                s = safe_name(fate)
                rec[f"terminal_count__{s}"] = int(count)
                rec[f"terminal_frac__{s}"] = float(count / total_terminal)

            records.append(rec)

        return pd.DataFrame(records)

    def choose_clone_table():
        tried = []

        for cfg in CLONE_FILTER_GRID:
            ct = build_clone_table_with_filters(
                clone_mat=clone_mat,
                early_mask=early_cloned_mask,
                terminal_mask=terminal_cloned_mask,
                fate_labels=fate_labels,
                config=cfg,
            )

            if ct.empty:
                tried.append((cfg["name"], 0, 0, []))
                continue

            fate_counts = ct["dominant_fate"].value_counts()
            selected = fate_counts[fate_counts >= MIN_CLONES_PER_FATE].index.tolist()
            selected = selected[:MAX_FATES]

            tried.append((cfg["name"], len(ct), len(selected), selected))

            if len(selected) >= 2:
                return ct, selected, cfg, tried

        raise RuntimeError(
            "No clone-filter setting produced at least 2 fates with enough clones. "
            "Lower MIN_CLONES_PER_FATE or relax terminal QC, but avoid single-clone fates."
        )

    def make_cipher_model(
        Xtrain_clone,
        y_train_for_delta,
        selected_fates,
        evals,
        evecs,
        Sigma,
        use_fate_prior=False,
    ):
        """
        Build CIPHER model.

        For each fate f:
            Delta_f = mean clone-mean early X for future-f clones
                      - mean clone-mean early X for non-f clones

            u_f = Sigma^{-1} Delta_f

            penalty_f = 0.5 u_f^T Sigma u_f

            log_enrichment_f(x) = x^T u_f - penalty_f

        Optional:
            + log prior_f if use_fate_prior=True
        """

        U = []
        DELTAS = []
        penalties = []
        log_priors = []

        eps = 1e-12

        for fate in selected_fates:
            pos = y_train_for_delta == fate
            neg = y_train_for_delta != fate

            if pos.sum() == 0 or neg.sum() == 0:
                raise RuntimeError(f"Missing positive/negative training clones for fate {fate}")

            delta = Xtrain_clone[pos].mean(axis=0) - Xtrain_clone[neg].mean(axis=0)

            # CIPHER inverse: u = Sigma^{-1} Delta.
            u = evecs @ ((evecs.T @ delta) / evals)

            # Full Gaussian tilt correction.
            penalty = 0.5 * float(u @ Sigma @ u)

            if use_fate_prior:
                prior = float(np.mean(y_train_for_delta == fate))
                log_prior = np.log(max(prior, eps))
            else:
                log_prior = 0.0

            U.append(u)
            DELTAS.append(delta)
            penalties.append(penalty)
            log_priors.append(log_prior)

        return {
            "U": np.asarray(U),
            "DELTAS": np.asarray(DELTAS),
            "penalty": np.asarray(penalties),
            "log_prior": np.asarray(log_priors),
        }

    def score_matrix_with_model(X, model):
        """
        Returns:
            raw_scores:
                x^T u_f

            log_enrichment:
                x^T u_f - 0.5 u_f^T Sigma u_f + optional log prior

            p_norm:
                softmax over fates using full log_enrichment
        """
        U = model["U"]

        raw_scores = X @ U.T

        log_enrichment = (
            raw_scores
            - model["penalty"][None, :]
            + model["log_prior"][None, :]
        )

        p_norm = softmax_logits(log_enrichment)

        return raw_scores, log_enrichment, p_norm

    def rows_from_scores(base_df, model_name, raw_scores, log_enrichment, p_norm):
        pred_idx = np.argmax(p_norm, axis=1)
        pred_fates = np.array(selected_fates, dtype=object)[pred_idx]

        rows = base_df.copy()
        rows["model"] = model_name
        rows["predicted_lineage_norm"] = pred_fates
        rows["max_pseudoprob_norm"] = p_norm.max(axis=1)

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            rows[f"score_raw__{s}"] = raw_scores[:, j]
            rows[f"log_enrichment__{s}"] = log_enrichment[:, j]
            rows[f"p_norm__{s}"] = p_norm[:, j]

        rows["log_enrichment_true_future_lineage"] = [
            log_enrichment[i, selected_fates.index(tf)]
            for i, tf in enumerate(rows["true_future_lineage"].values)
        ]

        rows["p_norm_true_future_lineage"] = [
            p_norm[i, selected_fates.index(tf)]
            for i, tf in enumerate(rows["true_future_lineage"].values)
        ]

        return rows

    # ============================================================
    # 2. LOAD DATA
    # ============================================================

    counts = mmread(COUNTS_PATH).T.tocsr()
    print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} cells | nnz={counts.nnz:,}")

    with gzip.open(GENES_PATH, "rt") as f:
        gene_names = np.array([line.strip() for line in f])
    print(f"Genes loaded: {len(gene_names)}")

    clone_mat = mmread(CLONE_PATH).T.tocsr()
    print(f"Clone matrix: {clone_mat.shape[0]} clones x {clone_mat.shape[1]} cells")
    print(f"% cells with clone label: {(clone_mat.sum(axis=0) > 0).mean() * 100:.2f}%")

    meta = pd.read_csv(META_PATH, sep="\t")
    print(f"Meta: {meta.shape[0]} rows x {meta.shape[1]} cols")
    print("Meta columns:", list(meta.columns))

    assert counts.shape[1] == meta.shape[0] == clone_mat.shape[1], "cells mismatch"
    assert counts.shape[0] == len(gene_names), "genes mismatch"

    meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")

    print("\nTimepoints:")
    print(np.sort(meta[TIME_COL].dropna().unique()))

    print("\nCell annotations:")
    print(meta[CELLTYPE_COL].value_counts())

    cell_to_clone = get_cell_to_clone(clone_mat)
    has_clone = cell_to_clone >= 0
    fate_labels = meta[CELLTYPE_COL].astype(str).values

    # ============================================================
    # 3. DEFINE EARLY / TERMINAL CELLS
    # ============================================================

    early_all_mask = mask_early_cells(meta)
    early_cloned_mask = early_all_mask & has_clone
    terminal_cloned_mask = mask_terminal_cells(meta) & has_clone

    early_all_idx = np.where(early_all_mask)[0]
    early_cloned_idx = np.where(early_cloned_mask)[0]
    terminal_cloned_idx = np.where(terminal_cloned_mask)[0]

    print(f"\nAll early/precommitted cells for Sigma: {len(early_all_idx):,}")
    print(f"Cloned early/precommitted cells: {len(early_cloned_idx):,}")
    print(f"Cloned terminal cells: {len(terminal_cloned_idx):,}")

    if len(early_all_idx) == 0:
        raise RuntimeError("No early cells found. Check EARLY_TIME / EARLY_CELLTYPE / EARLY_WELL.")

    if len(terminal_cloned_idx) == 0:
        raise RuntimeError("No terminal cloned cells found. Check TERMINAL_TIME / TERMINAL_WELL.")

    # ============================================================
    # 4. CLONE TABLE / FUTURE FATE LABELS
    # ============================================================

    clone_table_all, selected_fates, chosen_filter, tried_filters = choose_clone_table()

    print("\nClone-filter attempts:")
    for name, n_clones, n_fates, fates in tried_filters:
        print(f"  {name:28s} n_clones={n_clones:4d} n_fates={n_fates} fates={fates}")

    print("\nUsing clone filter:")
    print(chosen_filter)

    print("\nClone table after chosen clone QC:")
    print(f"n clones passing filters: {len(clone_table_all):,}")
    print(clone_table_all["dominant_fate"].value_counts())

    clone_table = clone_table_all[
        clone_table_all["dominant_fate"].isin(selected_fates)
    ].copy()

    eligible_clones = clone_table["clone_id"].values.astype(int)
    eligible_early_mask = early_cloned_mask & np.isin(cell_to_clone, eligible_clones)
    eligible_early_idx = np.where(eligible_early_mask)[0]

    clone_to_fate = dict(zip(clone_table["clone_id"], clone_table["dominant_fate"]))
    clone_to_frac = dict(zip(clone_table["clone_id"], clone_table["dominant_frac"]))
    clone_to_n_total = dict(zip(clone_table["clone_id"], clone_table["n_total_clone_cells"]))
    clone_to_n_early = dict(zip(clone_table["clone_id"], clone_table["n_early"]))
    clone_to_n_terminal = dict(zip(clone_table["clone_id"], clone_table["n_terminal"]))

    print("\nSelected fates:")
    print(clone_table["dominant_fate"].value_counts())

    print(f"\nEligible clones: {len(eligible_clones):,}")
    print(f"Eligible early cells: {len(eligible_early_idx):,}")

    clone_table_all.to_csv(os.path.join(OUTDIR, "clone_table_all_passing_qc.csv"), index=False)
    clone_table.to_csv(os.path.join(OUTDIR, "clone_table_selected_fates.csv"), index=False)

    # ============================================================
    # 5. CLONE QC PLOTS
    # ============================================================

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    sns.countplot(data=clone_table, x="dominant_fate", order=selected_fates, ax=axes[0, 0])
    axes[0, 0].set_title("Selected high-confidence clones per future fate")
    axes[0, 0].set_xlabel("future fate")
    axes[0, 0].set_ylabel("clone count")
    axes[0, 0].tick_params(axis="x", rotation=45)

    sns.histplot(data=clone_table, x="n_total_clone_cells", bins=40, ax=axes[0, 1])
    axes[0, 1].set_title("Total cells per retained clone")
    axes[0, 1].set_xlabel("total clone size")

    sns.histplot(data=clone_table, x="n_early", bins=30, ax=axes[0, 2])
    axes[0, 2].set_title("Early cells per retained clone")
    axes[0, 2].set_xlabel("early cells per clone")

    sns.histplot(data=clone_table, x="n_terminal", bins=40, ax=axes[1, 0])
    axes[1, 0].set_title("Terminal cells per retained clone")
    axes[1, 0].set_xlabel("terminal cells per clone")

    sns.scatterplot(
        data=clone_table,
        x="n_terminal",
        y="dominant_frac",
        hue="dominant_fate",
        hue_order=selected_fates,
        ax=axes[1, 1],
        s=45,
    )
    axes[1, 1].set_title("Clone purity vs terminal clone size")
    axes[1, 1].set_xlabel("terminal cells")
    axes[1, 1].set_ylabel("dominant fate fraction")
    axes[1, 1].legend(fontsize=9, frameon=False)

    sns.scatterplot(
        data=clone_table,
        x="n_early",
        y="n_terminal",
        hue="dominant_fate",
        hue_order=selected_fates,
        ax=axes[1, 2],
        s=45,
    )
    axes[1, 2].set_title("Early vs terminal clone representation")
    axes[1, 2].set_xlabel("early cells")
    axes[1, 2].set_ylabel("terminal cells")
    axes[1, 2].legend(fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "clone_qc_summary.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "clone_qc_summary.svg"), bbox_inches="tight")
    plt.show()

    if START_COL in meta.columns:
        start_rows = []

        for cid in eligible_clones:
            early_cells = np.where(eligible_early_mask & (cell_to_clone == cid))[0]
            if len(early_cells) == 0:
                continue

            start_counts = meta.iloc[early_cells][START_COL].astype(str).value_counts()

            start_rows.append({
                "clone_id": cid,
                "future_fate": clone_to_fate[cid],
                "dominant_starting_population": start_counts.index[0],
                "n_early": len(early_cells),
            })

        start_df = pd.DataFrame(start_rows)
        start_df.to_csv(os.path.join(OUTDIR, "clone_starting_population_summary.csv"), index=False)

        plt.figure(figsize=(10, 5))
        tab = pd.crosstab(start_df["future_fate"], start_df["dominant_starting_population"])
        tab = tab.reindex(selected_fates)
        sns.heatmap(tab, annot=True, fmt="d", cmap="viridis")
        plt.title("Future fate vs early starting population")
        plt.xlabel("dominant starting population among early cells")
        plt.ylabel("future fate")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "future_fate_vs_starting_population.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "future_fate_vs_starting_population.svg"), bbox_inches="tight")
        plt.show()

    # ============================================================
    # 6. GLOBAL EARLY HVGs + GLOBAL EARLY COVARIANCE
    # ============================================================

    print("\nSelecting HVGs from all early/precommitted cells...")

    hvg_idx, gene_vars = select_hvgs_sparse(
        counts=counts,
        cell_idx=early_all_idx,
        n_var_genes=N_VAR_GENES,
    )

    hvg_genes = gene_names[hvg_idx]

    pd.DataFrame({
        "gene": hvg_genes,
        "gene_index": hvg_idx,
        "early_variance": gene_vars[hvg_idx],
    }).to_csv(os.path.join(OUTDIR, "selected_early_hvgs.csv"), index=False)

    print(f"Using top {len(hvg_idx)} early-variable genes.")

    cov_idx = early_all_idx.copy()
    if len(cov_idx) > MAX_COV_CELLS:
        cov_idx = rng.choice(cov_idx, size=MAX_COV_CELLS, replace=False)

    print(f"Using {len(cov_idx):,} early cells for global Sigma.")

    Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx)
    mu_ref, sd_ref = zscore_train(Xcov_raw)
    Xcov = apply_zscore(Xcov_raw, mu_ref, sd_ref)

    Sigma = make_covariance(Xcov)

    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-8)

    pd.DataFrame({
        "rank": np.arange(1, len(evals) + 1),
        "eigenvalue": evals[::-1],
    }).to_csv(os.path.join(OUTDIR, "early_covariance_eigenvalues.csv"), index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(evals) + 1), evals[::-1], marker="o", linewidth=1, markersize=3)
    plt.yscale("log")
    plt.xlabel("eigenvalue rank")
    plt.ylabel("eigenvalue")
    plt.title("Early progenitor covariance spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # 7. CROSS-VALIDATED CIPHER VS NULL
    # ============================================================

    X_clones = clone_table["clone_id"].values.astype(int)
    y_clones = clone_table["dominant_fate"].values.astype(str)

    min_class_n = clone_table["dominant_fate"].value_counts().min()
    n_splits = int(min(N_SPLITS, min_class_n))

    if n_splits < 2:
        raise RuntimeError(
            f"Cannot do held-out CV: smallest selected fate has only {min_class_n} clones. "
            "Increase MIN_CLONES_PER_FATE or reduce MAX_FATES."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    print(f"\nUsing clone-level stratified {n_splits}-fold CV.")
    print(f"Running {N_NULLS} shuffled-label nulls per fold.")

    all_cell_rows = []
    all_clone_rows = []
    force_rows = []
    null_metric_rows = []

    for fold, (train_pos, test_pos) in enumerate(splitter.split(X_clones, y_clones)):
        train_clones = X_clones[train_pos]
        test_clones = X_clones[test_pos]

        print(f"\nFold {fold + 1}/{n_splits}: train clones={len(train_clones)}, test clones={len(test_clones)}")

        Xtrain_clone, train_clone_ids_used, n_train_early = clone_mean_matrix(
            clone_ids=train_clones,
            early_mask=eligible_early_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        if Xtrain_clone.shape[0] < 5:
            raise RuntimeError("Too few training clone means.")

        y_train = np.array([clone_to_fate[c] for c in train_clone_ids_used])

        # Held-out clone means.
        Xtest_clone, test_clone_ids_used, n_test_early = clone_mean_matrix(
            clone_ids=test_clones,
            early_mask=eligible_early_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        y_test_clone = np.array([clone_to_fate[c] for c in test_clone_ids_used])

        base_clone = pd.DataFrame({
            "fold": fold,
            "level": "clone",
            "clone_id": test_clone_ids_used,
            "true_future_lineage": y_test_clone,
            "true_future_lineage_frac": [clone_to_frac[c] for c in test_clone_ids_used],
            "n_early_scored": n_test_early,
            "n_total_clone_cells": [clone_to_n_total[c] for c in test_clone_ids_used],
            "n_early_clone_cells": [clone_to_n_early[c] for c in test_clone_ids_used],
            "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_clone_ids_used],
        })

        # Held-out early cells.
        test_early_idx = np.where(eligible_early_mask & np.isin(cell_to_clone, test_clones))[0]

        Xtest_cell_raw = get_cells_x_genes(counts, test_early_idx, hvg_idx)
        Xtest_cell = apply_zscore(Xtest_cell_raw, mu_ref, sd_ref)

        test_cell_clone_ids = cell_to_clone[test_early_idx]
        y_test_cell = np.array([clone_to_fate[c] for c in test_cell_clone_ids])

        base_cell = pd.DataFrame({
            "fold": fold,
            "level": "cell",
            "cell_index": test_early_idx,
            "clone_id": test_cell_clone_ids,
            "true_future_lineage": y_test_cell,
            "true_future_lineage_frac": [clone_to_frac[c] for c in test_cell_clone_ids],
            "n_total_clone_cells": [clone_to_n_total[c] for c in test_cell_clone_ids],
            "n_early_clone_cells": [clone_to_n_early[c] for c in test_cell_clone_ids],
            "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_cell_clone_ids],
        })

        # --------------------------
        # Real CIPHER model.
        # --------------------------
        cipher_model = make_cipher_model(
            Xtrain_clone=Xtrain_clone,
            y_train_for_delta=y_train,
            selected_fates=selected_fates,
            evals=evals,
            evecs=evecs,
            Sigma=Sigma,
            use_fate_prior=USE_FATE_PRIOR,
        )

        for Xscore, base_df, collector in [
            (Xtest_clone, base_clone, all_clone_rows),
            (Xtest_cell, base_cell, all_cell_rows),
        ]:
            raw_scores, log_enrichment, p_norm = score_matrix_with_model(Xscore, cipher_model)
            rows = rows_from_scores(base_df, "cipher", raw_scores, log_enrichment, p_norm)
            collector.append(rows)

        # Save top CIPHER genes.
        U = cipher_model["U"]
        DELTAS = cipher_model["DELTAS"]

        for j, fate in enumerate(selected_fates):
            u = U[j]
            delta = DELTAS[j]

            top_pos = np.argsort(u)[::-1][:50]
            top_neg = np.argsort(u)[:50]

            for rank, gi in enumerate(top_pos, start=1):
                force_rows.append({
                    "fold": fold,
                    "model": "cipher",
                    "fate": fate,
                    "direction": "positive",
                    "rank": rank,
                    "gene": hvg_genes[gi],
                    "gene_index": int(hvg_idx[gi]),
                    "u": float(u[gi]),
                    "delta_early": float(delta[gi]),
                    "penalty": float(cipher_model["penalty"][j]),
                    "log_prior": float(cipher_model["log_prior"][j]),
                })

            for rank, gi in enumerate(top_neg, start=1):
                force_rows.append({
                    "fold": fold,
                    "model": "cipher",
                    "fate": fate,
                    "direction": "negative",
                    "rank": rank,
                    "gene": hvg_genes[gi],
                    "gene_index": int(hvg_idx[gi]),
                    "u": float(u[gi]),
                    "delta_early": float(delta[gi]),
                    "penalty": float(cipher_model["penalty"][j]),
                    "log_prior": float(cipher_model["log_prior"][j]),
                })

        # --------------------------
        # Shuffled-label nulls.
        # --------------------------
        for null_id in range(N_NULLS):
            y_train_shuffled = y_train.copy()
            rng.shuffle(y_train_shuffled)

            null_model = make_cipher_model(
                Xtrain_clone=Xtrain_clone,
                y_train_for_delta=y_train_shuffled,
                selected_fates=selected_fates,
                evals=evals,
                evecs=evecs,
                Sigma=Sigma,
                use_fate_prior=USE_FATE_PRIOR,
            )

            for level_name, Xscore, base_df in [
                ("clone", Xtest_clone, base_clone),
                ("cell", Xtest_cell, base_cell),
            ]:
                raw_scores, log_enrichment, p_norm = score_matrix_with_model(Xscore, null_model)
                tmp = rows_from_scores(base_df, "shuffled_null", raw_scores, log_enrichment, p_norm)

                m = compute_metrics(tmp, selected_fates, score_prefix="log_enrichment")
                m["model"] = "shuffled_null"
                m["fold"] = fold
                m["null_id"] = null_id
                m["level"] = level_name

                null_metric_rows.append(m)

    early_cell_probs = pd.concat(all_cell_rows, ignore_index=True)
    clone_probs = pd.concat(all_clone_rows, ignore_index=True)
    force_df = pd.DataFrame(force_rows)
    null_metrics = pd.concat(null_metric_rows, ignore_index=True)

    early_cell_probs.to_csv(os.path.join(OUTDIR, "early_cell_cipher_probs.csv"), index=False)
    clone_probs.to_csv(os.path.join(OUTDIR, "clone_cipher_probs.csv"), index=False)
    force_df.to_csv(os.path.join(OUTDIR, "cipher_top_force_genes.csv"), index=False)
    null_metrics.to_csv(os.path.join(OUTDIR, "shuffled_null_metrics.csv"), index=False)

    print("\nSaved core outputs:")
    print(os.path.join(OUTDIR, "early_cell_cipher_probs.csv"))
    print(os.path.join(OUTDIR, "clone_cipher_probs.csv"))
    print(os.path.join(OUTDIR, "cipher_top_force_genes.csv"))
    print(os.path.join(OUTDIR, "shuffled_null_metrics.csv"))

    # ============================================================
    # 8. REAL CIPHER METRICS + EMPIRICAL NULL P-VALUES
    # ============================================================

    metric_rows = []

    for fold in sorted(clone_probs["fold"].unique()):
        df_clone = clone_probs[clone_probs["fold"] == fold].copy()
        df_cell = early_cell_probs[early_cell_probs["fold"] == fold].copy()

        m_clone = compute_metrics(df_clone, selected_fates, score_prefix="log_enrichment")
        m_clone["model"] = "cipher"
        m_clone["fold"] = fold
        m_clone["level"] = "clone"

        m_cell = compute_metrics(df_cell, selected_fates, score_prefix="log_enrichment")
        m_cell["model"] = "cipher"
        m_cell["fold"] = fold
        m_cell["level"] = "cell"

        metric_rows.append(m_clone)
        metric_rows.append(m_cell)

    cipher_metrics = pd.concat(metric_rows, ignore_index=True)
    cipher_metrics.to_csv(os.path.join(OUTDIR, "cipher_prediction_metrics_by_fold.csv"), index=False)

    all_metrics = pd.concat([cipher_metrics, null_metrics], ignore_index=True)
    all_metrics.to_csv(os.path.join(OUTDIR, "cipher_vs_null_metrics_all.csv"), index=False)

    summary_metrics = (
        all_metrics
        .groupby(["level", "model", "fate"], as_index=False)
        .agg(
            AUROC_mean=("AUROC", "mean"),
            AUROC_sd=("AUROC", "std"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_sd=("AUPRC", "std"),
            top_decile_enrichment_mean=("top_decile_enrichment", "mean"),
            top_decile_enrichment_sd=("top_decile_enrichment", "std"),
            n_positive_mean=("n_positive", "mean"),
            positive_fraction_mean=("positive_fraction", "mean"),
        )
    )

    summary_metrics.to_csv(os.path.join(OUTDIR, "cipher_vs_null_metrics_summary.csv"), index=False)

    # Empirical p-values.
    p_rows = []

    for level in ["clone", "cell"]:
        for fate in selected_fates:
            for metric_name in ["AUROC", "AUPRC", "top_decile_enrichment"]:
                real_vals = cipher_metrics[
                    (cipher_metrics["level"] == level) &
                    (cipher_metrics["fate"] == fate)
                ][metric_name].dropna().values

                null_vals = null_metrics[
                    (null_metrics["level"] == level) &
                    (null_metrics["fate"] == fate)
                ][metric_name].dropna().values

                if len(real_vals) == 0 or len(null_vals) == 0:
                    p_emp = np.nan
                    real_mean = np.nan
                    null_mean = np.nan
                else:
                    real_mean = float(np.mean(real_vals))
                    null_mean = float(np.mean(null_vals))
                    p_emp = float((1 + np.sum(null_vals >= real_mean)) / (1 + len(null_vals)))

                p_rows.append({
                    "level": level,
                    "fate": fate,
                    "metric": metric_name,
                    "cipher_mean": real_mean,
                    "null_mean": null_mean,
                    "empirical_p": p_emp,
                    "n_null": len(null_vals),
                })

    pvals = pd.DataFrame(p_rows)
    pvals.to_csv(os.path.join(OUTDIR, "cipher_vs_null_empirical_pvalues.csv"), index=False)

    # Accuracy summary.
    acc_rows = []

    for level_name, df in [("clone", clone_probs), ("cell", early_cell_probs)]:
        acc_rows.append({
            "model": "cipher",
            "level": level_name,
            "argmax_accuracy": np.mean(df["predicted_lineage_norm"] == df["true_future_lineage"]),
            "mean_log_enrichment_true": df["log_enrichment_true_future_lineage"].mean(),
            "mean_p_true_norm": df["p_norm_true_future_lineage"].mean(),
        })

    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(os.path.join(OUTDIR, "cipher_argmax_accuracy_summary.csv"), index=False)

    print("\nClone-level CIPHER vs null summary:")
    print(
        summary_metrics[
            summary_metrics["level"] == "clone"
        ][[
            "model",
            "fate",
            "n_positive_mean",
            "positive_fraction_mean",
            "AUROC_mean",
            "AUROC_sd",
            "AUPRC_mean",
            "AUPRC_sd",
            "top_decile_enrichment_mean",
        ]]
    )

    print("\nEmpirical p-values, clone AUROC:")
    print(
        pvals[
            (pvals["level"] == "clone") &
            (pvals["metric"] == "AUROC")
        ][["fate", "cipher_mean", "null_mean", "empirical_p", "n_null"]]
    )

    print("\nArgmax summary:")
    print(acc_df)

    # ============================================================
    # 9. PLOTS: CIPHER VS NULL
    # ============================================================

    # --------------------------
    # AUROC / AUPRC / enrichment distributions.
    # --------------------------
    for metric_name in ["AUROC", "AUPRC", "top_decile_enrichment"]:
        plt.figure(figsize=(11, 5))

        sub = all_metrics[all_metrics["level"] == "clone"].copy()
        sub["model_label"] = sub["model"].map({
            "cipher": "CIPHER",
            "shuffled_null": "shuffled null",
        })

        sns.boxplot(
            data=sub,
            x="fate",
            y=metric_name,
            hue="model_label",
            order=selected_fates,
            showfliers=False,
        )

        sub_points = sub[sub["model"] == "cipher"].copy()
        sns.stripplot(
            data=sub_points,
            x="fate",
            y=metric_name,
            hue="model_label",
            order=selected_fates,
            dodge=True,
            color="black",
            alpha=0.6,
            size=4,
            legend=False,
        )

        if metric_name == "AUROC":
            plt.axhline(0.5, color="gray", linestyle="--", linewidth=2)
            plt.ylim(0, 1)
        elif metric_name == "AUPRC":
            plt.ylim(0, 1)
        else:
            plt.axhline(1.0, color="gray", linestyle="--", linewidth=2)

        plt.title(f"CIPHER vs shuffled-label null: clone-level {metric_name}")
        plt.xlabel("future lineage")
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha="right")

        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = []
        uniq_labels = []
        for h, l in zip(handles, labels):
            if l not in uniq_labels:
                uniq.append(h)
                uniq_labels.append(l)
        plt.legend(uniq, uniq_labels, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"cipher_vs_null_clone_{metric_name}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"cipher_vs_null_clone_{metric_name}.svg"), bbox_inches="tight")
        plt.show()

    # --------------------------
    # Empirical p-values.
    # --------------------------
    p_plot = pvals[
        (pvals["level"] == "clone") &
        (pvals["metric"] == "AUROC")
    ].copy()

    p_plot["minus_log10_p"] = -np.log10(np.maximum(p_plot["empirical_p"], 1e-300))

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=p_plot,
        x="fate",
        y="minus_log10_p",
        order=selected_fates,
    )
    plt.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=2, label="p=0.05")
    plt.title("Empirical shuffled-label null p-values")
    plt.xlabel("future lineage")
    plt.ylabel("-log10 empirical p, AUROC")
    plt.xticks(rotation=45, ha="right")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cipher_vs_null_empirical_pvalues_AUROC.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "cipher_vs_null_empirical_pvalues_AUROC.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # CIPHER probability heatmap + confusion matrix.
    # --------------------------
    main_clone = clone_probs.copy()
    p_norm_cols = [f"p_norm__{safe_name(f)}" for f in selected_fates]

    mean_prob = (
        main_clone
        .groupby("true_future_lineage")[p_norm_cols]
        .mean()
        .reindex(selected_fates)
    )
    mean_prob.columns = selected_fates

    cm = confusion_matrix(
        main_clone["true_future_lineage"],
        main_clone["predicted_lineage_norm"],
        labels=selected_fates,
    )

    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        mean_prob,
        ax=axes[0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "mean normalized pseudo-prob"},
    )
    axes[0].set_title("Clone mean CIPHER probabilities")
    axes[0].set_xlabel("predicted future lineage")
    axes[0].set_ylabel("true future lineage")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        pd.DataFrame(cm_norm, index=selected_fates, columns=selected_fates),
        ax=axes[1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "row-normalized fraction"},
    )
    axes[1].set_title("Argmax CIPHER prediction")
    axes[1].set_xlabel("predicted future lineage")
    axes[1].set_ylabel("true future lineage")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_probability_heatmap_confusion.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_probability_heatmap_confusion.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # CIPHER p(true lineage).
    # --------------------------
    plt.figure(figsize=(11, 5))
    sns.boxplot(
        data=main_clone,
        x="true_future_lineage",
        y="p_norm_true_future_lineage",
        order=selected_fates,
        showfliers=False,
    )
    sns.stripplot(
        data=main_clone,
        x="true_future_lineage",
        y="p_norm_true_future_lineage",
        order=selected_fates,
        color="black",
        alpha=0.35,
        size=3,
    )
    plt.ylim(0, 1)
    plt.title("CIPHER normalized probability assigned to true future lineage")
    plt.xlabel("true future lineage")
    plt.ylabel("p(true future lineage | early clone)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_p_true_lineage_by_fate.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_p_true_lineage_by_fate.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # Future-fate vs other log-enrichment distributions.
    # --------------------------
    score_rows = []

    for fate in selected_fates:
        col = f"log_enrichment__{safe_name(fate)}"
        tmp = main_clone[["clone_id", "true_future_lineage", col]].copy()
        tmp["tested_fate"] = fate
        tmp["is_future_fate"] = np.where(tmp["true_future_lineage"] == fate, "future fate", "other")
        tmp["log_enrichment"] = tmp[col]
        score_rows.append(tmp[["clone_id", "tested_fate", "is_future_fate", "log_enrichment"]])

    score_df = pd.concat(score_rows, ignore_index=True)

    plt.figure(figsize=(12, 5))
    sns.boxplot(
        data=score_df,
        x="tested_fate",
        y="log_enrichment",
        hue="is_future_fate",
        order=selected_fates,
        showfliers=False,
    )
    plt.title("CIPHER log-enrichment: future-fate clones vs others")
    plt.xlabel("tested lineage")
    plt.ylabel("full Gaussian log-enrichment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "CIPHER_positive_vs_rest_log_enrichment.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "CIPHER_positive_vs_rest_log_enrichment.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # Top CIPHER force genes heatmap.
    # --------------------------
    cipher_force = force_df[force_df["direction"] == "positive"].copy()

    mean_force = (
        cipher_force
        .groupby(["fate", "gene"], as_index=False)
        .agg(
            mean_u=("u", "mean"),
            mean_delta=("delta_early", "mean"),
            mean_rank=("rank", "mean"),
            mean_penalty=("penalty", "mean"),
        )
    )

    top_genes = []
    TOP_GENES_PER_FATE = 12

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(TOP_GENES_PER_FATE)
        )
        top_genes.extend(sub["gene"].tolist())

    top_genes = list(dict.fromkeys(top_genes))

    heat = (
        mean_force
        .pivot_table(index="gene", columns="fate", values="mean_u", fill_value=0)
        .reindex(top_genes)
        .reindex(columns=selected_fates)
    )

    plt.figure(figsize=(1.4 * len(selected_fates) + 6, 0.28 * len(top_genes) + 4))
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "mean CIPHER force u"},
    )
    plt.title("Top positive early-bias CIPHER force genes")
    plt.xlabel("future lineage")
    plt.ylabel("gene")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "top_CIPHER_force_genes_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "top_CIPHER_force_genes_heatmap.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # PCA of early cells.
    # --------------------------
    MAX_PLOT_CELLS = 7000
    main_cell = early_cell_probs.copy()

    plot_df = main_cell.copy()
    if len(plot_df) > MAX_PLOT_CELLS:
        plot_df = plot_df.sample(MAX_PLOT_CELLS, random_state=SEED)

    plot_cells = plot_df["cell_index"].values.astype(int)
    X_plot = get_cells_x_genes(counts, plot_cells, hvg_idx)
    X_plot = apply_zscore(X_plot, mu_ref, sd_ref)

    Z = PCA(n_components=2, random_state=SEED).fit_transform(X_plot)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=plot_df["p_norm_true_future_lineage"].values,
        s=8,
        alpha=0.8,
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    plt.colorbar(sc, label="p(true future lineage | early cell)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Early cells colored by CIPHER probability of true future lineage")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "early_cells_pca_p_true_lineage.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "early_cells_pca_p_true_lineage.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # 10. FINAL PRINTS
    # ============================================================

    print("\n============================================================")
    print("FINAL CLONE-LEVEL CIPHER SUMMARY")
    print("============================================================")
    print(
        summary_metrics[
            (summary_metrics["level"] == "clone") &
            (summary_metrics["model"] == "cipher")
        ][[
            "fate",
            "n_positive_mean",
            "positive_fraction_mean",
            "AUROC_mean",
            "AUROC_sd",
            "AUPRC_mean",
            "AUPRC_sd",
            "top_decile_enrichment_mean",
        ]].sort_values("AUROC_mean", ascending=False)
    )

    print("\n============================================================")
    print("NULL-COMPARISON EMPIRICAL P-VALUES, CLONE AUROC")
    print("============================================================")
    print(
        pvals[
            (pvals["level"] == "clone") &
            (pvals["metric"] == "AUROC")
        ][["fate", "cipher_mean", "null_mean", "empirical_p", "n_null"]]
    )

    print("\n============================================================")
    print("ARGMAX ACCURACY")
    print("============================================================")
    print(acc_df)

    print("\n============================================================")
    print("TOP POSITIVE CIPHER FORCE GENES")
    print("============================================================")

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(20)
        )
        print(f"\n{fate}")
        print(", ".join(sub["gene"].astype(str).tolist()))

    print("\nDone. Outputs in:", OUTDIR)



def controls_startpop_and_horizons():
    global os, gzip, warnings, np, pd, plt, sns, mmread, \
        issparse, StratifiedKFold, roc_auc_score, average_precision_score, confusion_matrix, PCA, OUTDIR, COUNTS_PATH, \
        GENES_PATH, CLONE_PATH, META_PATH, TIME_COL, CELLTYPE_COL, WELL_COL, START_COL, EARLY_TIMES_TO_RUN, \
        EARLY_CELLTYPE, EARLY_WELL, TERMINAL_TIME, TERMINAL_WELL, EXCLUDE_FATES, CLONE_FILTER_GRID, MIN_CLONES_PER_FATE_GLOBAL, MIN_CLONES_PER_FATE_WITHIN_START, \
        MAX_FATES, N_VAR_GENES, MAX_COV_CELLS, RIDGE, COV_SHRINK_TO_DIAG, USE_FATE_PRIOR, N_NULLS, N_SPLITS, \
        SEED, rng, safe_name, softmax_logits, barplot_sd, get_cell_to_clone, get_cells_x_genes, zscore_train, \
        fate_entropy_from_counts, select_hvgs_sparse, make_covariance, clone_mean_matrix, compute_metrics, build_masks, build_clone_table_with_filters, annotate_starting_population, \
        choose_clone_table_for_masks, shuffle_labels_within_groups, make_cipher_model, score_matrix_with_model, rows_from_scores, fit_startpop_baseline, score_startpop_baseline, make_analysis_dir, \
        counts, f, gene_names, clone_mat, meta, cell_to_clone, has_clone, fate_labels, \
        run_one_analysis, all_results, early_time, res_global, early_all_mask, _, starts_this_time, start_pop, \
        res_within, combined_metrics, combined_summary, combined_pvals, combined_accuracy, plot_df
    # ============================================================
    # CIPHER-LARRY controls:
    #   1. CIPHER vs shuffled-label null
    #   2. starting-population-preserving shuffled null
    #   3. starting-population-only baseline
    #   4. within-starting-population analyses
    #   5. optional day-2 and day-4 horizons
    #
    # Uses full Gaussian log-enrichment:
    #
    #   ell_f(x) = u_f^T x - 1/2 u_f^T Sigma u_f + log prior_f
    #
    # where:
    #
    #   u_f = Sigma^{-1} Delta_f
    #
    # and Delta_f is computed from clone-balanced early means.
    # ============================================================

    import os
    import gzip
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.io import mmread
    from scipy.sparse import issparse
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    from sklearn.decomposition import PCA

    warnings.filterwarnings("ignore")

    # ============================================================
    # CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_startpop_controls_full")
    os.makedirs(OUTDIR, exist_ok=True)

    COUNTS_PATH = os.path.join(SUPPL, "stateFate_inVitro_normed_counts.mtx.gz")
    GENES_PATH  = os.path.join(SUPPL, "stateFate_inVitro_gene_names.txt.gz")
    CLONE_PATH  = os.path.join(SUPPL, "stateFate_inVitro_clone_matrix.mtx.gz")
    META_PATH   = os.path.join(SUPPL, "stateFate_inVitro_metadata.txt.gz")

    TIME_COL = "Time point"
    CELLTYPE_COL = "Cell type annotation"
    WELL_COL = "Well"
    START_COL = "Starting population"

    # Run both for the strongest story.
    # Day 2 = harder, earlier prospective prediction.
    # Day 4 = later, easier prediction.
    EARLY_TIMES_TO_RUN = [2.0, 4.0]

    EARLY_CELLTYPE = "Undifferentiated"
    EARLY_WELL = None

    TERMINAL_TIME = 6.0
    TERMINAL_WELL = None

    EXCLUDE_FATES = {
        "Undifferentiated",
        "Unknown",
        "unknown",
        "nan",
        "NaN",
        "Ambiguous",
        "ambiguous",
        "None",
        "",
    }

    # Strict terminal labels, lenient early cells.
    CLONE_FILTER_GRID = [
        dict(
            name="strict_terminal_lenient_early",
            min_total=12,
            min_early=1,
            min_terminal=8,
            min_dom_count=6,
            min_dom_frac=0.85,
            max_entropy=0.65,
        ),
        dict(
            name="medium_terminal_lenient_early",
            min_total=10,
            min_early=1,
            min_terminal=5,
            min_dom_count=4,
            min_dom_frac=0.80,
            max_entropy=0.75,
        ),
        dict(
            name="lenient_terminal_still_qc",
            min_total=8,
            min_early=1,
            min_terminal=4,
            min_dom_count=3,
            min_dom_frac=0.75,
            max_entropy=0.85,
        ),
    ]

    MIN_CLONES_PER_FATE_GLOBAL = 5
    MIN_CLONES_PER_FATE_WITHIN_START = 5
    MAX_FATES = 5

    N_VAR_GENES = 500
    MAX_COV_CELLS = 50000

    RIDGE = 1e-6
    COV_SHRINK_TO_DIAG = 0.0

    # Use False for pure log-enrichment, True for posterior-like probabilities.
    USE_FATE_PRIOR = False

    N_NULLS = 100
    N_SPLITS = 5

    SEED = 0
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    plt.rcParams.update({"font.size": 14})
    sns.set_context("talk")

    # ============================================================
    # HELPERS
    # ============================================================


    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )

    def softmax_logits(logits, eps=1e-12):
        z = logits - np.max(logits, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.maximum(ez.sum(axis=1, keepdims=True), eps)

    def barplot_sd(*args, **kwargs):
        try:
            return sns.barplot(*args, errorbar="sd", **kwargs)
        except Exception:
            return sns.barplot(*args, ci="sd", **kwargs)

    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()
        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row
        return cell_to_clone

    def get_cells_x_genes(counts, cell_idx, gene_idx):
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)

    def zscore_train(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu, sd


    def fate_entropy_from_counts(counts_vec):
        counts_vec = np.asarray(counts_vec, dtype=float)
        counts_vec = counts_vec[counts_vec > 0]
        if counts_vec.size <= 1:
            return 0.0
        p = counts_vec / counts_vec.sum()
        return float(-(p * np.log(p)).sum())

    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        X = counts[:, cell_idx]
        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means**2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]

        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_var_genes:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_

    def make_covariance(X):
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D
        Sigma = Sigma + RIDGE * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)

    def clone_mean_matrix(clone_ids, early_mask, cell_to_clone, counts, hvg_idx, mu, sd):
        rows = []
        out_ids = []
        out_n = []

        for cid in clone_ids:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]
            if len(idx) == 0:
                continue

            X = get_cells_x_genes(counts, idx, hvg_idx)
            X = apply_zscore(X, mu, sd)

            rows.append(X.mean(axis=0))
            out_ids.append(cid)
            out_n.append(len(idx))

        if len(rows) == 0:
            return (
                np.empty((0, len(hvg_idx))),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        return np.vstack(rows), np.asarray(out_ids, dtype=int), np.asarray(out_n, dtype=int)

    def compute_metrics(df, selected_fates, score_prefix="log_enrichment", label_col="true_future_lineage"):
        rows = []

        for fate in selected_fates:
            col = f"{score_prefix}__{safe_name(fate)}"

            y = (df[label_col].astype(str).values == str(fate)).astype(int)
            s = df[col].values.astype(float)

            if len(np.unique(y)) < 2:
                auroc = np.nan
                auprc = np.nan
            else:
                auroc = roc_auc_score(y, s)
                auprc = average_precision_score(y, s)

            baseline = y.mean()
            cutoff = np.quantile(s, 0.90)
            top = s >= cutoff

            if top.sum() > 0 and baseline > 0:
                top_rate = y[top].mean()
                enrichment = top_rate / baseline
            else:
                top_rate = np.nan
                enrichment = np.nan

            rows.append({
                "fate": fate,
                "n": len(y),
                "n_positive": int(y.sum()),
                "positive_fraction": float(baseline),
                "AUROC": auroc,
                "AUPRC": auprc,
                "top_decile_positive_rate": top_rate,
                "top_decile_enrichment": enrichment,
            })

        return pd.DataFrame(rows)

    def build_masks(meta, early_time, restrict_start=None):
        early_all_mask = meta[TIME_COL].astype(float).values == float(early_time)

        if EARLY_CELLTYPE is not None:
            early_all_mask &= meta[CELLTYPE_COL].astype(str).values == str(EARLY_CELLTYPE)

        if EARLY_WELL is not None and WELL_COL in meta.columns:
            early_all_mask &= meta[WELL_COL].astype(float).values == float(EARLY_WELL)

        if restrict_start is not None and START_COL in meta.columns:
            early_all_mask &= meta[START_COL].astype(str).values == str(restrict_start)

        terminal_mask = meta[TIME_COL].astype(float).values == float(TERMINAL_TIME)

        if TERMINAL_WELL is not None and WELL_COL in meta.columns:
            terminal_mask &= meta[WELL_COL].astype(float).values == float(TERMINAL_WELL)

        ann = meta[CELLTYPE_COL].astype(str).values
        terminal_mask &= ~np.isin(ann, list(EXCLUDE_FATES))

        return early_all_mask, terminal_mask

    def build_clone_table_with_filters(clone_mat, early_mask, terminal_mask, fate_labels, config):
        records = []

        for clone_id in range(clone_mat.shape[0]):
            cells = clone_mat[clone_id].indices

            if len(cells) < config["min_total"]:
                continue

            early_cells = cells[early_mask[cells]]
            terminal_cells = cells[terminal_mask[cells]]

            if len(early_cells) < config["min_early"]:
                continue
            if len(terminal_cells) < config["min_terminal"]:
                continue

            terminal_fates = pd.Series(fate_labels[terminal_cells].astype(str))
            terminal_fates = terminal_fates[~terminal_fates.isin(EXCLUDE_FATES)]

            if len(terminal_fates) < config["min_terminal"]:
                continue

            fate_counts = terminal_fates.value_counts()
            if len(fate_counts) == 0:
                continue

            dominant_fate = str(fate_counts.index[0])
            dominant_count = int(fate_counts.iloc[0])
            total_terminal = int(fate_counts.sum())
            dominant_frac = dominant_count / max(total_terminal, 1)
            entropy = fate_entropy_from_counts(fate_counts.values)

            if dominant_count < config["min_dom_count"]:
                continue
            if dominant_frac < config["min_dom_frac"]:
                continue
            if config["max_entropy"] is not None and entropy > config["max_entropy"]:
                continue

            rec = {
                "clone_id": int(clone_id),
                "n_total_clone_cells": int(len(cells)),
                "n_early": int(len(early_cells)),
                "n_terminal": int(total_terminal),
                "n_terminal_raw": int(len(terminal_cells)),
                "n_terminal_fate_types": int(len(fate_counts)),
                "dominant_fate": dominant_fate,
                "dominant_count": dominant_count,
                "dominant_frac": float(dominant_frac),
                "fate_entropy": float(entropy),
                "filter_config": config["name"],
            }

            for fate, count in fate_counts.items():
                s = safe_name(fate)
                rec[f"terminal_count__{s}"] = int(count)
                rec[f"terminal_frac__{s}"] = float(count / total_terminal)

            records.append(rec)

        return pd.DataFrame(records)

    def annotate_starting_population(clone_table, early_mask, cell_to_clone, meta):
        if START_COL not in meta.columns:
            clone_table["dominant_starting_population"] = "unknown"
            clone_table["dominant_starting_population_frac"] = 1.0
            return clone_table

        starts = []
        start_fracs = []

        for cid in clone_table["clone_id"].values:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]

            if len(idx) == 0:
                starts.append("unknown")
                start_fracs.append(np.nan)
                continue

            vc = meta.iloc[idx][START_COL].astype(str).value_counts()
            starts.append(vc.index[0])
            start_fracs.append(float(vc.iloc[0] / vc.sum()))

        clone_table = clone_table.copy()
        clone_table["dominant_starting_population"] = starts
        clone_table["dominant_starting_population_frac"] = start_fracs

        return clone_table

    def choose_clone_table_for_masks(
        clone_mat,
        early_cloned_mask,
        terminal_cloned_mask,
        fate_labels,
        min_clones_per_fate,
    ):
        tried = []

        for cfg in CLONE_FILTER_GRID:
            ct = build_clone_table_with_filters(
                clone_mat=clone_mat,
                early_mask=early_cloned_mask,
                terminal_mask=terminal_cloned_mask,
                fate_labels=fate_labels,
                config=cfg,
            )

            if ct.empty:
                tried.append((cfg["name"], 0, 0, []))
                continue

            ct = annotate_starting_population(ct, early_cloned_mask, cell_to_clone, meta)

            fate_counts = ct["dominant_fate"].value_counts()
            selected = fate_counts[fate_counts >= min_clones_per_fate].index.tolist()
            selected = selected[:MAX_FATES]

            tried.append((cfg["name"], len(ct), len(selected), selected))

            if len(selected) >= 2:
                return ct, selected, cfg, tried

        return None, None, None, tried

    def shuffle_labels_within_groups(y, groups):
        y = np.asarray(y).copy()
        groups = np.asarray(groups).astype(str)

        out = y.copy()

        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) > 1:
                out[idx] = rng.permutation(out[idx])

        return out

    def make_cipher_model(
        Xtrain_clone,
        y_train_for_delta,
        selected_fates,
        evals,
        evecs,
        Sigma,
        use_fate_prior=False,
    ):
        U = []
        DELTAS = []
        penalties = []
        log_priors = []

        eps = 1e-12

        for fate in selected_fates:
            pos = y_train_for_delta == fate
            neg = y_train_for_delta != fate

            if pos.sum() == 0 or neg.sum() == 0:
                raise RuntimeError(f"Missing positive/negative training clones for fate {fate}")

            delta = Xtrain_clone[pos].mean(axis=0) - Xtrain_clone[neg].mean(axis=0)

            u = evecs @ ((evecs.T @ delta) / evals)
            penalty = 0.5 * float(u @ Sigma @ u)

            if use_fate_prior:
                prior = float(np.mean(y_train_for_delta == fate))
                log_prior = np.log(max(prior, eps))
            else:
                log_prior = 0.0

            U.append(u)
            DELTAS.append(delta)
            penalties.append(penalty)
            log_priors.append(log_prior)

        return {
            "U": np.asarray(U),
            "DELTAS": np.asarray(DELTAS),
            "penalty": np.asarray(penalties),
            "log_prior": np.asarray(log_priors),
        }

    def score_matrix_with_model(X, model):
        U = model["U"]
        raw_scores = X @ U.T

        log_enrichment = (
            raw_scores
            - model["penalty"][None, :]
            + model["log_prior"][None, :]
        )

        p_norm = softmax_logits(log_enrichment)

        return raw_scores, log_enrichment, p_norm

    def rows_from_scores(base_df, model_name, raw_scores, log_enrichment, p_norm, selected_fates):
        pred_idx = np.argmax(p_norm, axis=1)
        pred_fates = np.array(selected_fates, dtype=object)[pred_idx]

        rows = base_df.copy()
        rows["model"] = model_name
        rows["predicted_lineage_norm"] = pred_fates
        rows["max_pseudoprob_norm"] = p_norm.max(axis=1)

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            rows[f"score_raw__{s}"] = raw_scores[:, j]
            rows[f"log_enrichment__{s}"] = log_enrichment[:, j]
            rows[f"p_norm__{s}"] = p_norm[:, j]

        rows["log_enrichment_true_future_lineage"] = [
            log_enrichment[i, selected_fates.index(tf)]
            for i, tf in enumerate(rows["true_future_lineage"].values)
        ]

        rows["p_norm_true_future_lineage"] = [
            p_norm[i, selected_fates.index(tf)]
            for i, tf in enumerate(rows["true_future_lineage"].values)
        ]

        return rows

    def fit_startpop_baseline(y_train, start_train, selected_fates, alpha=1.0):
        y_train = np.asarray(y_train).astype(str)
        start_train = np.asarray(start_train).astype(str)

        fates = list(selected_fates)
        starts = np.unique(start_train).tolist()

        global_counts = pd.Series(y_train).value_counts()
        global_probs = np.array([
            (global_counts.get(f, 0) + alpha) / (len(y_train) + alpha * len(fates))
            for f in fates
        ])
        global_probs = global_probs / global_probs.sum()

        table = {}

        for s in starts:
            idx = start_train == s
            ys = y_train[idx]
            counts = pd.Series(ys).value_counts()

            probs = np.array([
                (counts.get(f, 0) + alpha) / (len(ys) + alpha * len(fates))
                for f in fates
            ])
            probs = probs / probs.sum()
            table[s] = probs

        return {
            "fates": fates,
            "starts": starts,
            "table": table,
            "global_probs": global_probs,
        }

    def score_startpop_baseline(start_test, model):
        start_test = np.asarray(start_test).astype(str)
        fates = model["fates"]

        probs = []
        for s in start_test:
            if s in model["table"]:
                probs.append(model["table"][s])
            else:
                probs.append(model["global_probs"])

        p_norm = np.vstack(probs)
        log_scores = np.log(np.clip(p_norm, 1e-12, 1.0))
        raw_scores = log_scores.copy()

        return raw_scores, log_scores, p_norm

    def make_analysis_dir(label):
        path = os.path.join(OUTDIR, safe_name(label))
        os.makedirs(path, exist_ok=True)
        return path

    # ============================================================
    # LOAD DATA ONCE
    # ============================================================

    counts = mmread(COUNTS_PATH).T.tocsr()
    print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} cells | nnz={counts.nnz:,}")

    with gzip.open(GENES_PATH, "rt") as f:
        gene_names = np.array([line.strip() for line in f])
    print(f"Genes loaded: {len(gene_names)}")

    clone_mat = mmread(CLONE_PATH).T.tocsr()
    print(f"Clone matrix: {clone_mat.shape[0]} clones x {clone_mat.shape[1]} cells")
    print(f"% cells with clone label: {(clone_mat.sum(axis=0) > 0).mean() * 100:.2f}%")

    meta = pd.read_csv(META_PATH, sep="\t")
    print(f"Meta: {meta.shape[0]} rows x {meta.shape[1]} cols")
    print("Meta columns:", list(meta.columns))

    assert counts.shape[1] == meta.shape[0] == clone_mat.shape[1], "cells mismatch"
    assert counts.shape[0] == len(gene_names), "genes mismatch"

    meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")

    print("\nTimepoints:")
    print(np.sort(meta[TIME_COL].dropna().unique()))

    print("\nCell annotations:")
    print(meta[CELLTYPE_COL].value_counts())

    cell_to_clone = get_cell_to_clone(clone_mat)
    has_clone = cell_to_clone >= 0
    fate_labels = meta[CELLTYPE_COL].astype(str).values

    # ============================================================
    # CORE ANALYSIS FUNCTION
    # ============================================================

    def run_one_analysis(
        early_time,
        restrict_start=None,
        min_clones_per_fate=10,
        run_startpop_baseline=True,
    ):
        analysis_label = f"early{early_time}"
        if restrict_start is None:
            analysis_label += "_all_startpops"
        else:
            analysis_label += f"_within_{safe_name(restrict_start)}"

        AOUT = make_analysis_dir(analysis_label)

        print("\n" + "=" * 80)
        print(f"RUNNING ANALYSIS: {analysis_label}")
        print("=" * 80)

        early_all_mask, terminal_mask = build_masks(meta, early_time, restrict_start=restrict_start)

        early_cloned_mask = early_all_mask & has_clone
        terminal_cloned_mask = terminal_mask & has_clone

        early_all_idx = np.where(early_all_mask)[0]
        early_cloned_idx = np.where(early_cloned_mask)[0]
        terminal_cloned_idx = np.where(terminal_cloned_mask)[0]

        print(f"All early cells for Sigma: {len(early_all_idx):,}")
        print(f"Cloned early cells: {len(early_cloned_idx):,}")
        print(f"Cloned terminal cells: {len(terminal_cloned_idx):,}")

        if len(early_all_idx) == 0 or len(early_cloned_idx) == 0 or len(terminal_cloned_idx) == 0:
            print("[skip] missing early/terminal cells")
            return None

        clone_table_all, selected_fates, chosen_filter, tried_filters = choose_clone_table_for_masks(
            clone_mat=clone_mat,
            early_cloned_mask=early_cloned_mask,
            terminal_cloned_mask=terminal_cloned_mask,
            fate_labels=fate_labels,
            min_clones_per_fate=min_clones_per_fate,
        )

        print("\nClone-filter attempts:")
        for name, n_clones, n_fates, fates in tried_filters:
            print(f"  {name:28s} n_clones={n_clones:4d} n_fates={n_fates} fates={fates}")

        if clone_table_all is None:
            print("[skip] no valid clone table")
            return None

        clone_table = clone_table_all[
            clone_table_all["dominant_fate"].isin(selected_fates)
        ].copy()

        eligible_clones = clone_table["clone_id"].values.astype(int)
        eligible_early_mask = early_cloned_mask & np.isin(cell_to_clone, eligible_clones)
        eligible_early_idx = np.where(eligible_early_mask)[0]

        clone_to_fate = dict(zip(clone_table["clone_id"], clone_table["dominant_fate"]))
        clone_to_frac = dict(zip(clone_table["clone_id"], clone_table["dominant_frac"]))
        clone_to_n_total = dict(zip(clone_table["clone_id"], clone_table["n_total_clone_cells"]))
        clone_to_n_early = dict(zip(clone_table["clone_id"], clone_table["n_early"]))
        clone_to_n_terminal = dict(zip(clone_table["clone_id"], clone_table["n_terminal"]))
        clone_to_start = dict(zip(clone_table["clone_id"], clone_table["dominant_starting_population"]))

        print("\nSelected fates:")
        print(clone_table["dominant_fate"].value_counts())
        print("\nStarting populations:")
        print(clone_table["dominant_starting_population"].value_counts())
        print(f"\nEligible clones: {len(eligible_clones):,}")
        print(f"Eligible early cells: {len(eligible_early_idx):,}")

        clone_table_all.to_csv(os.path.join(AOUT, "clone_table_all_passing_qc.csv"), index=False)
        clone_table.to_csv(os.path.join(AOUT, "clone_table_selected_fates.csv"), index=False)

        # --------------------------
        # Clone QC plots.
        # --------------------------
        fig, axes = plt.subplots(2, 3, figsize=(20, 11))

        sns.countplot(data=clone_table, x="dominant_fate", order=selected_fates, ax=axes[0, 0])
        axes[0, 0].set_title("Selected clones per future fate")
        axes[0, 0].set_xlabel("future fate")
        axes[0, 0].set_ylabel("clone count")
        axes[0, 0].tick_params(axis="x", rotation=45)

        sns.histplot(data=clone_table, x="n_total_clone_cells", bins=40, ax=axes[0, 1])
        axes[0, 1].set_title("Total cells per retained clone")

        sns.histplot(data=clone_table, x="n_early", bins=30, ax=axes[0, 2])
        axes[0, 2].set_title("Early cells per retained clone")

        sns.histplot(data=clone_table, x="n_terminal", bins=40, ax=axes[1, 0])
        axes[1, 0].set_title("Terminal cells per retained clone")

        sns.scatterplot(
            data=clone_table,
            x="n_terminal",
            y="dominant_frac",
            hue="dominant_fate",
            hue_order=selected_fates,
            ax=axes[1, 1],
            s=45,
        )
        axes[1, 1].set_title("Clone purity vs terminal size")
        axes[1, 1].legend(fontsize=9, frameon=False)

        sns.scatterplot(
            data=clone_table,
            x="n_early",
            y="n_terminal",
            hue="dominant_fate",
            hue_order=selected_fates,
            ax=axes[1, 2],
            s=45,
        )
        axes[1, 2].set_title("Early vs terminal representation")
        axes[1, 2].legend(fontsize=9, frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "clone_qc_summary.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "clone_qc_summary.svg"), bbox_inches="tight")
        plt.show()

        if START_COL in meta.columns:
            plt.figure(figsize=(10, 5))
            tab = pd.crosstab(
                clone_table["dominant_fate"],
                clone_table["dominant_starting_population"]
            )
            tab = tab.reindex(selected_fates)
            sns.heatmap(tab, annot=True, fmt="d", cmap="viridis")
            plt.title("Future fate vs early starting population")
            plt.xlabel("dominant starting population among early cells")
            plt.ylabel("future fate")
            plt.tight_layout()
            plt.savefig(os.path.join(AOUT, "future_fate_vs_starting_population.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join(AOUT, "future_fate_vs_starting_population.svg"), bbox_inches="tight")
            plt.show()

        # --------------------------
        # HVGs and Sigma.
        # --------------------------
        print("\nSelecting HVGs and building Sigma...")

        hvg_idx, gene_vars = select_hvgs_sparse(
            counts=counts,
            cell_idx=early_all_idx,
            n_var_genes=N_VAR_GENES,
        )
        hvg_genes = gene_names[hvg_idx]

        pd.DataFrame({
            "gene": hvg_genes,
            "gene_index": hvg_idx,
            "early_variance": gene_vars[hvg_idx],
        }).to_csv(os.path.join(AOUT, "selected_early_hvgs.csv"), index=False)

        cov_idx = early_all_idx.copy()
        if len(cov_idx) > MAX_COV_CELLS:
            cov_idx = rng.choice(cov_idx, size=MAX_COV_CELLS, replace=False)

        Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx)
        mu_ref, sd_ref = zscore_train(Xcov_raw)
        Xcov = apply_zscore(Xcov_raw, mu_ref, sd_ref)

        Sigma = make_covariance(Xcov)

        evals, evecs = np.linalg.eigh(Sigma)
        evals = np.maximum(evals, 1e-8)

        pd.DataFrame({
            "rank": np.arange(1, len(evals) + 1),
            "eigenvalue": evals[::-1],
        }).to_csv(os.path.join(AOUT, "early_covariance_eigenvalues.csv"), index=False)

        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(evals) + 1), evals[::-1], marker="o", linewidth=1, markersize=3)
        plt.yscale("log")
        plt.xlabel("eigenvalue rank")
        plt.ylabel("eigenvalue")
        plt.title("Early progenitor covariance spectrum")
        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "early_covariance_spectrum.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "early_covariance_spectrum.svg"), bbox_inches="tight")
        plt.show()

        # --------------------------
        # CV.
        # --------------------------
        X_clones = clone_table["clone_id"].values.astype(int)
        y_clones = clone_table["dominant_fate"].values.astype(str)

        min_class_n = clone_table["dominant_fate"].value_counts().min()
        n_splits = int(min(N_SPLITS, min_class_n))

        if n_splits < 2:
            print(f"[skip] cannot do CV; smallest fate has {min_class_n} clones.")
            return None

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED,
        )

        print(f"\nUsing clone-level stratified {n_splits}-fold CV.")
        print(f"Running {N_NULLS} shuffled-label nulls per fold.")
        print("Global analysis uses start-pop-preserving shuffled null.")
        print("Within-start analyses use standard shuffled null because start pop is fixed.")

        all_cell_rows = []
        all_clone_rows = []
        force_rows = []
        null_metric_rows = []
        startpop_metric_rows = []
        startpop_clone_rows = []

        for fold, (train_pos, test_pos) in enumerate(splitter.split(X_clones, y_clones)):
            train_clones = X_clones[train_pos]
            test_clones = X_clones[test_pos]

            print(f"\nFold {fold + 1}/{n_splits}: train={len(train_clones)}, test={len(test_clones)}")

            Xtrain_clone, train_clone_ids_used, n_train_early = clone_mean_matrix(
                clone_ids=train_clones,
                early_mask=eligible_early_mask,
                cell_to_clone=cell_to_clone,
                counts=counts,
                hvg_idx=hvg_idx,
                mu=mu_ref,
                sd=sd_ref,
            )

            y_train = np.array([clone_to_fate[c] for c in train_clone_ids_used])
            start_train = np.array([clone_to_start.get(c, "unknown") for c in train_clone_ids_used])

            Xtest_clone, test_clone_ids_used, n_test_early = clone_mean_matrix(
                clone_ids=test_clones,
                early_mask=eligible_early_mask,
                cell_to_clone=cell_to_clone,
                counts=counts,
                hvg_idx=hvg_idx,
                mu=mu_ref,
                sd=sd_ref,
            )

            y_test_clone = np.array([clone_to_fate[c] for c in test_clone_ids_used])
            start_test_clone = np.array([clone_to_start.get(c, "unknown") for c in test_clone_ids_used])

            base_clone = pd.DataFrame({
                "analysis": analysis_label,
                "fold": fold,
                "level": "clone",
                "clone_id": test_clone_ids_used,
                "true_future_lineage": y_test_clone,
                "dominant_starting_population": start_test_clone,
                "true_future_lineage_frac": [clone_to_frac[c] for c in test_clone_ids_used],
                "n_early_scored": n_test_early,
                "n_total_clone_cells": [clone_to_n_total[c] for c in test_clone_ids_used],
                "n_early_clone_cells": [clone_to_n_early[c] for c in test_clone_ids_used],
                "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_clone_ids_used],
            })

            test_early_idx = np.where(eligible_early_mask & np.isin(cell_to_clone, test_clones))[0]
            Xtest_cell_raw = get_cells_x_genes(counts, test_early_idx, hvg_idx)
            Xtest_cell = apply_zscore(Xtest_cell_raw, mu_ref, sd_ref)

            test_cell_clone_ids = cell_to_clone[test_early_idx]
            y_test_cell = np.array([clone_to_fate[c] for c in test_cell_clone_ids])
            start_test_cell = np.array([clone_to_start.get(c, "unknown") for c in test_cell_clone_ids])

            base_cell = pd.DataFrame({
                "analysis": analysis_label,
                "fold": fold,
                "level": "cell",
                "cell_index": test_early_idx,
                "clone_id": test_cell_clone_ids,
                "true_future_lineage": y_test_cell,
                "dominant_starting_population": start_test_cell,
                "true_future_lineage_frac": [clone_to_frac[c] for c in test_cell_clone_ids],
                "n_total_clone_cells": [clone_to_n_total[c] for c in test_cell_clone_ids],
                "n_early_clone_cells": [clone_to_n_early[c] for c in test_cell_clone_ids],
                "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_cell_clone_ids],
            })

            # Real CIPHER.
            cipher_model = make_cipher_model(
                Xtrain_clone=Xtrain_clone,
                y_train_for_delta=y_train,
                selected_fates=selected_fates,
                evals=evals,
                evecs=evecs,
                Sigma=Sigma,
                use_fate_prior=USE_FATE_PRIOR,
            )

            for Xscore, base_df, collector in [
                (Xtest_clone, base_clone, all_clone_rows),
                (Xtest_cell, base_cell, all_cell_rows),
            ]:
                raw_scores, log_enrichment, p_norm = score_matrix_with_model(Xscore, cipher_model)
                rows = rows_from_scores(base_df, "cipher", raw_scores, log_enrichment, p_norm, selected_fates)
                collector.append(rows)

            # Top genes.
            U = cipher_model["U"]
            DELTAS = cipher_model["DELTAS"]

            for j, fate in enumerate(selected_fates):
                u = U[j]
                delta = DELTAS[j]

                top_pos = np.argsort(u)[::-1][:50]
                top_neg = np.argsort(u)[:50]

                for rank, gi in enumerate(top_pos, start=1):
                    force_rows.append({
                        "analysis": analysis_label,
                        "fold": fold,
                        "model": "cipher",
                        "fate": fate,
                        "direction": "positive",
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),
                        "u": float(u[gi]),
                        "delta_early": float(delta[gi]),
                        "penalty": float(cipher_model["penalty"][j]),
                        "log_prior": float(cipher_model["log_prior"][j]),
                    })

                for rank, gi in enumerate(top_neg, start=1):
                    force_rows.append({
                        "analysis": analysis_label,
                        "fold": fold,
                        "model": "cipher",
                        "fate": fate,
                        "direction": "negative",
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),
                        "u": float(u[gi]),
                        "delta_early": float(delta[gi]),
                        "penalty": float(cipher_model["penalty"][j]),
                        "log_prior": float(cipher_model["log_prior"][j]),
                    })

            # Starting-population-only baseline.
            if run_startpop_baseline and restrict_start is None and START_COL in meta.columns:
                start_model = fit_startpop_baseline(
                    y_train=y_train,
                    start_train=start_train,
                    selected_fates=selected_fates,
                    alpha=1.0,
                )

                raw_scores, log_enrichment, p_norm = score_startpop_baseline(start_test_clone, start_model)
                rows = rows_from_scores(
                    base_clone,
                    "starting_population_only",
                    raw_scores,
                    log_enrichment,
                    p_norm,
                    selected_fates,
                )
                startpop_clone_rows.append(rows)

                m = compute_metrics(rows, selected_fates, score_prefix="log_enrichment")
                m["analysis"] = analysis_label
                m["model"] = "starting_population_only"
                m["fold"] = fold
                m["level"] = "clone"
                startpop_metric_rows.append(m)

            # Shuffled-label nulls.
            for null_id in range(N_NULLS):
                if restrict_start is None and START_COL in meta.columns:
                    y_train_null = shuffle_labels_within_groups(y_train, start_train)
                    null_type = "startpop_preserving_shuffled_null"
                else:
                    y_train_null = rng.permutation(y_train)
                    null_type = "shuffled_null"

                null_model = make_cipher_model(
                    Xtrain_clone=Xtrain_clone,
                    y_train_for_delta=y_train_null,
                    selected_fates=selected_fates,
                    evals=evals,
                    evecs=evecs,
                    Sigma=Sigma,
                    use_fate_prior=USE_FATE_PRIOR,
                )

                for level_name, Xscore, base_df in [
                    ("clone", Xtest_clone, base_clone),
                    ("cell", Xtest_cell, base_cell),
                ]:
                    raw_scores, log_enrichment, p_norm = score_matrix_with_model(Xscore, null_model)
                    tmp = rows_from_scores(
                        base_df,
                        null_type,
                        raw_scores,
                        log_enrichment,
                        p_norm,
                        selected_fates,
                    )

                    m = compute_metrics(tmp, selected_fates, score_prefix="log_enrichment")
                    m["analysis"] = analysis_label
                    m["model"] = null_type
                    m["fold"] = fold
                    m["null_id"] = null_id
                    m["level"] = level_name

                    null_metric_rows.append(m)

        early_cell_probs = pd.concat(all_cell_rows, ignore_index=True)
        clone_probs = pd.concat(all_clone_rows, ignore_index=True)
        force_df = pd.DataFrame(force_rows)
        null_metrics = pd.concat(null_metric_rows, ignore_index=True)

        if len(startpop_clone_rows) > 0:
            startpop_clone_probs = pd.concat(startpop_clone_rows, ignore_index=True)
            startpop_metrics = pd.concat(startpop_metric_rows, ignore_index=True)
        else:
            startpop_clone_probs = pd.DataFrame()
            startpop_metrics = pd.DataFrame()

        early_cell_probs.to_csv(os.path.join(AOUT, "early_cell_cipher_probs.csv"), index=False)
        clone_probs.to_csv(os.path.join(AOUT, "clone_cipher_probs.csv"), index=False)
        force_df.to_csv(os.path.join(AOUT, "cipher_top_force_genes.csv"), index=False)
        null_metrics.to_csv(os.path.join(AOUT, "null_metrics.csv"), index=False)
        startpop_clone_probs.to_csv(os.path.join(AOUT, "starting_population_only_clone_probs.csv"), index=False)
        startpop_metrics.to_csv(os.path.join(AOUT, "starting_population_only_metrics.csv"), index=False)

        # Real metrics.
        metric_rows = []
        for fold in sorted(clone_probs["fold"].unique()):
            df_clone = clone_probs[clone_probs["fold"] == fold].copy()
            df_cell = early_cell_probs[early_cell_probs["fold"] == fold].copy()

            m_clone = compute_metrics(df_clone, selected_fates, score_prefix="log_enrichment")
            m_clone["analysis"] = analysis_label
            m_clone["model"] = "cipher"
            m_clone["fold"] = fold
            m_clone["level"] = "clone"

            m_cell = compute_metrics(df_cell, selected_fates, score_prefix="log_enrichment")
            m_cell["analysis"] = analysis_label
            m_cell["model"] = "cipher"
            m_cell["fold"] = fold
            m_cell["level"] = "cell"

            metric_rows.append(m_clone)
            metric_rows.append(m_cell)

        cipher_metrics = pd.concat(metric_rows, ignore_index=True)

        metric_parts = [cipher_metrics, null_metrics]
        if len(startpop_metrics) > 0:
            metric_parts.append(startpop_metrics)

        all_metrics = pd.concat(metric_parts, ignore_index=True)
        all_metrics.to_csv(os.path.join(AOUT, "all_metrics.csv"), index=False)

        summary_metrics = (
            all_metrics
            .groupby(["analysis", "level", "model", "fate"], as_index=False)
            .agg(
                AUROC_mean=("AUROC", "mean"),
                AUROC_sd=("AUROC", "std"),
                AUPRC_mean=("AUPRC", "mean"),
                AUPRC_sd=("AUPRC", "std"),
                top_decile_enrichment_mean=("top_decile_enrichment", "mean"),
                top_decile_enrichment_sd=("top_decile_enrichment", "std"),
                n_positive_mean=("n_positive", "mean"),
                positive_fraction_mean=("positive_fraction", "mean"),
            )
        )
        summary_metrics.to_csv(os.path.join(AOUT, "summary_metrics.csv"), index=False)

        # Empirical p-values vs null.
        p_rows = []
        null_model_names = [m for m in null_metrics["model"].unique()]

        for null_model_name in null_model_names:
            for level in ["clone", "cell"]:
                for fate in selected_fates:
                    for metric_name in ["AUROC", "AUPRC", "top_decile_enrichment"]:
                        real_vals = cipher_metrics[
                            (cipher_metrics["level"] == level) &
                            (cipher_metrics["fate"] == fate)
                        ][metric_name].dropna().values

                        null_vals = null_metrics[
                            (null_metrics["level"] == level) &
                            (null_metrics["fate"] == fate) &
                            (null_metrics["model"] == null_model_name)
                        ][metric_name].dropna().values

                        if len(real_vals) == 0 or len(null_vals) == 0:
                            p_emp = np.nan
                            real_mean = np.nan
                            null_mean = np.nan
                        else:
                            real_mean = float(np.mean(real_vals))
                            null_mean = float(np.mean(null_vals))
                            p_emp = float((1 + np.sum(null_vals >= real_mean)) / (1 + len(null_vals)))

                        p_rows.append({
                            "analysis": analysis_label,
                            "level": level,
                            "fate": fate,
                            "metric": metric_name,
                            "null_model": null_model_name,
                            "cipher_mean": real_mean,
                            "null_mean": null_mean,
                            "empirical_p": p_emp,
                            "n_null": len(null_vals),
                        })

        pvals = pd.DataFrame(p_rows)
        pvals.to_csv(os.path.join(AOUT, "empirical_pvalues.csv"), index=False)

        # Accuracy.
        acc_rows = []
        for level_name, df in [("clone", clone_probs), ("cell", early_cell_probs)]:
            acc_rows.append({
                "analysis": analysis_label,
                "model": "cipher",
                "level": level_name,
                "argmax_accuracy": np.mean(df["predicted_lineage_norm"] == df["true_future_lineage"]),
                "mean_log_enrichment_true": df["log_enrichment_true_future_lineage"].mean(),
                "mean_p_true_norm": df["p_norm_true_future_lineage"].mean(),
            })

        if len(startpop_clone_probs) > 0:
            acc_rows.append({
                "analysis": analysis_label,
                "model": "starting_population_only",
                "level": "clone",
                "argmax_accuracy": np.mean(startpop_clone_probs["predicted_lineage_norm"] == startpop_clone_probs["true_future_lineage"]),
                "mean_log_enrichment_true": startpop_clone_probs["log_enrichment_true_future_lineage"].mean(),
                "mean_p_true_norm": startpop_clone_probs["p_norm_true_future_lineage"].mean(),
            })

        acc_df = pd.DataFrame(acc_rows)
        acc_df.to_csv(os.path.join(AOUT, "accuracy_summary.csv"), index=False)

        # --------------------------
        # Plots.
        # --------------------------
        plot_metrics = all_metrics[all_metrics["level"] == "clone"].copy()

        label_map = {
            "cipher": "CIPHER",
            "shuffled_null": "shuffled null",
            "startpop_preserving_shuffled_null": "startpop-preserving null",
            "starting_population_only": "starting-pop only",
        }
        plot_metrics["model_label"] = plot_metrics["model"].map(label_map).fillna(plot_metrics["model"])

        for metric_name in ["AUROC", "AUPRC", "top_decile_enrichment"]:
            plt.figure(figsize=(12, 5))

            sns.boxplot(
                data=plot_metrics,
                x="fate",
                y=metric_name,
                hue="model_label",
                order=selected_fates,
                showfliers=False,
            )

            point_df = plot_metrics[
                plot_metrics["model"].isin(["cipher", "starting_population_only"])
            ].copy()

            if len(point_df) > 0:
                sns.stripplot(
                    data=point_df,
                    x="fate",
                    y=metric_name,
                    hue="model_label",
                    order=selected_fates,
                    dodge=True,
                    color="black",
                    alpha=0.65,
                    size=4,
                    legend=False,
                )

            if metric_name == "AUROC":
                plt.axhline(0.5, color="gray", linestyle="--", linewidth=2)
                plt.ylim(0, 1)
            elif metric_name == "AUPRC":
                plt.ylim(0, 1)
            else:
                plt.axhline(1.0, color="gray", linestyle="--", linewidth=2)

            plt.title(f"{analysis_label}: clone-level {metric_name}")
            plt.xlabel("future lineage")
            plt.ylabel(metric_name)
            plt.xticks(rotation=45, ha="right")

            handles, labels = plt.gca().get_legend_handles_labels()
            uniq = []
            uniq_labels = []
            for h, l in zip(handles, labels):
                if l not in uniq_labels:
                    uniq.append(h)
                    uniq_labels.append(l)
            plt.legend(uniq, uniq_labels, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.tight_layout()
            plt.savefig(os.path.join(AOUT, f"clone_{metric_name}_model_comparison.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join(AOUT, f"clone_{metric_name}_model_comparison.svg"), bbox_inches="tight")
            plt.show()

        # p-values plot.
        p_plot = pvals[
            (pvals["level"] == "clone") &
            (pvals["metric"] == "AUROC")
        ].copy()

        if len(p_plot) > 0:
            p_plot["minus_log10_p"] = -np.log10(np.maximum(p_plot["empirical_p"], 1e-300))

            plt.figure(figsize=(9, 5))
            sns.barplot(
                data=p_plot,
                x="fate",
                y="minus_log10_p",
                hue="null_model",
                order=selected_fates,
            )
            plt.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=2, label="p=0.05")
            plt.title(f"{analysis_label}: empirical null p-values")
            plt.xlabel("future lineage")
            plt.ylabel("-log10 empirical p, AUROC")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(AOUT, "empirical_pvalues_AUROC.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join(AOUT, "empirical_pvalues_AUROC.svg"), bbox_inches="tight")
            plt.show()

        # CIPHER probability heatmap + confusion matrix.
        main_clone = clone_probs.copy()
        p_norm_cols = [f"p_norm__{safe_name(f)}" for f in selected_fates]

        mean_prob = (
            main_clone
            .groupby("true_future_lineage")[p_norm_cols]
            .mean()
            .reindex(selected_fates)
        )
        mean_prob.columns = selected_fates

        cm = confusion_matrix(
            main_clone["true_future_lineage"],
            main_clone["predicted_lineage_norm"],
            labels=selected_fates,
        )
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        sns.heatmap(
            mean_prob,
            ax=axes[0],
            cmap="viridis",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "mean normalized pseudo-prob"},
        )
        axes[0].set_title("Clone mean CIPHER probabilities")
        axes[0].set_xlabel("predicted future lineage")
        axes[0].set_ylabel("true future lineage")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].tick_params(axis="y", rotation=0)

        sns.heatmap(
            pd.DataFrame(cm_norm, index=selected_fates, columns=selected_fates),
            ax=axes[1],
            cmap="viridis",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "row-normalized fraction"},
        )
        axes[1].set_title("Argmax CIPHER prediction")
        axes[1].set_xlabel("predicted future lineage")
        axes[1].set_ylabel("true future lineage")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].tick_params(axis="y", rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "CIPHER_probability_heatmap_confusion.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "CIPHER_probability_heatmap_confusion.svg"), bbox_inches="tight")
        plt.show()

        # p(true lineage).
        plt.figure(figsize=(11, 5))
        sns.boxplot(
            data=main_clone,
            x="true_future_lineage",
            y="p_norm_true_future_lineage",
            order=selected_fates,
            showfliers=False,
        )
        sns.stripplot(
            data=main_clone,
            x="true_future_lineage",
            y="p_norm_true_future_lineage",
            order=selected_fates,
            color="black",
            alpha=0.35,
            size=3,
        )
        plt.ylim(0, 1)
        plt.title(f"{analysis_label}: CIPHER p(true future lineage)")
        plt.xlabel("true future lineage")
        plt.ylabel("p(true future lineage | early clone)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "CIPHER_p_true_lineage_by_fate.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "CIPHER_p_true_lineage_by_fate.svg"), bbox_inches="tight")
        plt.show()

        # positive vs rest log-enrichment.
        score_rows = []
        for fate in selected_fates:
            col = f"log_enrichment__{safe_name(fate)}"
            tmp = main_clone[["clone_id", "true_future_lineage", col]].copy()
            tmp["tested_fate"] = fate
            tmp["is_future_fate"] = np.where(tmp["true_future_lineage"] == fate, "future fate", "other")
            tmp["log_enrichment"] = tmp[col]
            score_rows.append(tmp[["clone_id", "tested_fate", "is_future_fate", "log_enrichment"]])

        score_df = pd.concat(score_rows, ignore_index=True)

        plt.figure(figsize=(12, 5))
        sns.boxplot(
            data=score_df,
            x="tested_fate",
            y="log_enrichment",
            hue="is_future_fate",
            order=selected_fates,
            showfliers=False,
        )
        plt.title(f"{analysis_label}: CIPHER log-enrichment")
        plt.xlabel("tested lineage")
        plt.ylabel("full Gaussian log-enrichment")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "CIPHER_positive_vs_rest_log_enrichment.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "CIPHER_positive_vs_rest_log_enrichment.svg"), bbox_inches="tight")
        plt.show()

        # Top CIPHER genes heatmap.
        cipher_force = force_df[force_df["direction"] == "positive"].copy()

        mean_force = (
            cipher_force
            .groupby(["fate", "gene"], as_index=False)
            .agg(
                mean_u=("u", "mean"),
                mean_delta=("delta_early", "mean"),
                mean_rank=("rank", "mean"),
                mean_penalty=("penalty", "mean"),
            )
        )

        top_genes = []
        TOP_GENES_PER_FATE = 12
        for fate in selected_fates:
            sub = (
                mean_force[mean_force["fate"] == fate]
                .sort_values("mean_u", ascending=False)
                .head(TOP_GENES_PER_FATE)
            )
            top_genes.extend(sub["gene"].tolist())

        top_genes = list(dict.fromkeys(top_genes))

        heat = (
            mean_force
            .pivot_table(index="gene", columns="fate", values="mean_u", fill_value=0)
            .reindex(top_genes)
            .reindex(columns=selected_fates)
        )

        plt.figure(figsize=(1.4 * len(selected_fates) + 6, 0.28 * len(top_genes) + 4))
        sns.heatmap(
            heat,
            cmap="vlag",
            center=0,
            cbar_kws={"label": "mean CIPHER force u"},
        )
        plt.title(f"{analysis_label}: top positive early-bias CIPHER force genes")
        plt.xlabel("future lineage")
        plt.ylabel("gene")
        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "top_CIPHER_force_genes_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "top_CIPHER_force_genes_heatmap.svg"), bbox_inches="tight")
        plt.show()

        # PCA of early cells.
        MAX_PLOT_CELLS = 7000
        main_cell = early_cell_probs.copy()

        plot_df = main_cell.copy()
        if len(plot_df) > MAX_PLOT_CELLS:
            plot_df = plot_df.sample(MAX_PLOT_CELLS, random_state=SEED)

        if len(plot_df) > 3:
            plot_cells = plot_df["cell_index"].values.astype(int)
            X_plot = get_cells_x_genes(counts, plot_cells, hvg_idx)
            X_plot = apply_zscore(X_plot, mu_ref, sd_ref)

            Z = PCA(n_components=2, random_state=SEED).fit_transform(X_plot)

            plt.figure(figsize=(7, 6))
            sc = plt.scatter(
                Z[:, 0],
                Z[:, 1],
                c=plot_df["p_norm_true_future_lineage"].values,
                s=8,
                alpha=0.8,
                vmin=0,
                vmax=1,
                cmap="viridis",
            )
            plt.colorbar(sc, label="p(true future lineage | early cell)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"{analysis_label}: early cells")
            plt.tight_layout()
            plt.savefig(os.path.join(AOUT, "early_cells_pca_p_true_lineage.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join(AOUT, "early_cells_pca_p_true_lineage.svg"), bbox_inches="tight")
            plt.show()

        print("\nFinal clone-level CIPHER summary:")
        print(
            summary_metrics[
                (summary_metrics["level"] == "clone") &
                (summary_metrics["model"] == "cipher")
            ][[
                "fate",
                "n_positive_mean",
                "positive_fraction_mean",
                "AUROC_mean",
                "AUROC_sd",
                "AUPRC_mean",
                "AUPRC_sd",
                "top_decile_enrichment_mean",
            ]].sort_values("AUROC_mean", ascending=False)
        )

        print("\nEmpirical p-values, clone AUROC:")
        print(
            pvals[
                (pvals["level"] == "clone") &
                (pvals["metric"] == "AUROC")
            ][["fate", "null_model", "cipher_mean", "null_mean", "empirical_p", "n_null"]]
        )

        print("\nAccuracy:")
        print(acc_df)

        return {
            "analysis": analysis_label,
            "outdir": AOUT,
            "selected_fates": selected_fates,
            "clone_table": clone_table,
            "cipher_metrics": cipher_metrics,
            "null_metrics": null_metrics,
            "startpop_metrics": startpop_metrics,
            "all_metrics": all_metrics,
            "summary_metrics": summary_metrics,
            "pvals": pvals,
            "accuracy": acc_df,
        }

    # ============================================================
    # RUN GLOBAL + WITHIN-STARTING-POPULATION ANALYSES
    # ============================================================

    all_results = []

    for early_time in EARLY_TIMES_TO_RUN:
        # Global analysis: uses all starting populations and start-pop-preserving null.
        res_global = run_one_analysis(
            early_time=early_time,
            restrict_start=None,
            min_clones_per_fate=MIN_CLONES_PER_FATE_GLOBAL,
            run_startpop_baseline=True,
        )
        if res_global is not None:
            all_results.append(res_global)

        # Determine which starting populations exist at this early time.
        early_all_mask, _ = build_masks(meta, early_time, restrict_start=None)
        if START_COL in meta.columns:
            starts_this_time = (
                meta.loc[early_all_mask, START_COL]
                .astype(str)
                .value_counts()
                .index
                .tolist()
            )
        else:
            starts_this_time = []

        # Within-starting-population analyses.
        for start_pop in starts_this_time:
            res_within = run_one_analysis(
                early_time=early_time,
                restrict_start=start_pop,
                min_clones_per_fate=MIN_CLONES_PER_FATE_WITHIN_START,
                run_startpop_baseline=False,
            )
            if res_within is not None:
                all_results.append(res_within)

    # ============================================================
    # COMBINE ACROSS ANALYSES
    # ============================================================

    if len(all_results) > 0:
        combined_metrics = pd.concat([r["all_metrics"] for r in all_results], ignore_index=True)
        combined_summary = pd.concat([r["summary_metrics"] for r in all_results], ignore_index=True)
        combined_pvals = pd.concat([r["pvals"] for r in all_results], ignore_index=True)
        combined_accuracy = pd.concat([r["accuracy"] for r in all_results], ignore_index=True)

        combined_metrics.to_csv(os.path.join(OUTDIR, "combined_all_metrics.csv"), index=False)
        combined_summary.to_csv(os.path.join(OUTDIR, "combined_summary_metrics.csv"), index=False)
        combined_pvals.to_csv(os.path.join(OUTDIR, "combined_empirical_pvalues.csv"), index=False)
        combined_accuracy.to_csv(os.path.join(OUTDIR, "combined_accuracy.csv"), index=False)

        # Summary plot across analyses: clone AUROC.
        plot_df = combined_metrics[combined_metrics["level"] == "clone"].copy()
        plot_df["model_label"] = plot_df["model"].map({
            "cipher": "CIPHER",
            "shuffled_null": "shuffled null",
            "startpop_preserving_shuffled_null": "startpop-preserving null",
            "starting_population_only": "starting-pop only",
        }).fillna(plot_df["model"])

        plt.figure(figsize=(16, 6))
        sns.boxplot(
            data=plot_df,
            x="analysis",
            y="AUROC",
            hue="model_label",
            showfliers=False,
        )
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=2)
        plt.ylim(0, 1)
        plt.title("CIPHER fate prediction across horizons and starting-pop controls")
        plt.xlabel("analysis")
        plt.ylabel("clone-level AUROC")
        plt.xticks(rotation=60, ha="right")
        plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "combined_clone_AUROC_by_analysis.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "combined_clone_AUROC_by_analysis.svg"), bbox_inches="tight")
        plt.show()

        print("\nCombined outputs saved in:", OUTDIR)
        print("\nCombined clone-level CIPHER summary:")
        print(
            combined_summary[
                (combined_summary["level"] == "clone") &
                (combined_summary["model"] == "cipher")
            ][[
                "analysis",
                "fate",
                "n_positive_mean",
                "AUROC_mean",
                "AUPRC_mean",
                "top_decile_enrichment_mean",
            ]].sort_values(["analysis", "AUROC_mean"], ascending=[True, False])
        )

        print("\nCombined empirical p-values, clone AUROC:")
        print(
            combined_pvals[
                (combined_pvals["level"] == "clone") &
                (combined_pvals["metric"] == "AUROC")
            ][[
                "analysis",
                "fate",
                "null_model",
                "cipher_mean",
                "null_mean",
                "empirical_p",
                "n_null",
            ]].sort_values(["analysis", "fate"])
        )

    else:
        print("No analyses completed. Relax filters or check EARLY_TIMES_TO_RUN.")



def controls_with_roc_pr_curves():
    global os, gzip, warnings, np, pd, plt, sns, mmread, \
        issparse, StratifiedKFold, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, PCA, \
        OUTDIR, COUNTS_PATH, GENES_PATH, CLONE_PATH, META_PATH, TIME_COL, CELLTYPE_COL, WELL_COL, \
        START_COL, EARLY_TIMES_TO_RUN, EARLY_CELLTYPE, EARLY_WELL, TERMINAL_TIME, TERMINAL_WELL, EXCLUDE_FATES, CLONE_FILTER_GRID, \
        MIN_CLONES_PER_FATE_GLOBAL, MIN_CLONES_PER_FATE_WITHIN_START, MAX_FATES, GLOBAL_FATE_NAME, N_VAR_GENES, MAX_COV_CELLS, RIDGE, COV_SHRINK_TO_DIAG, \
        USE_FATE_PRIOR, N_NULLS, N_SPLITS, SEED, rng, safe_name, softmax_logits, barplot_sd, \
        get_cell_to_clone, get_cells_x_genes, zscore_train, fate_entropy_from_counts, select_hvgs_sparse, make_covariance, clone_mean_matrix, make_global_ovr_vectors, \
        compute_metrics, plot_global_ovr_curves, build_masks, build_clone_table_with_filters, annotate_starting_population, choose_clone_table_for_masks, shuffle_labels_within_groups, make_cipher_model, \
        score_matrix_with_model, rows_from_scores, fit_startpop_baseline, score_startpop_baseline, make_analysis_dir, counts, f, gene_names, \
        clone_mat, meta, cell_to_clone, has_clone, fate_labels, run_one_analysis, all_results, early_time, \
        res_global, early_all_mask, _, starts_this_time, start_pop, res_within, combined_metrics, combined_summary, \
        combined_pvals, combined_accuracy, combined_global_curves, combined_global_gains, plot_df, global_plot_df, metric_name
    # ============================================================
    # CIPHER-LARRY controls:
    #   1. CIPHER vs shuffled-label null
    #   2. starting-population-preserving shuffled null
    #   3. starting-population-only baseline
    #   4. within-starting-population analyses
    #   5. optional day-2 and day-4 horizons
    #
    # Uses full Gaussian log-enrichment:
    #
    #   ell_f(x) = u_f^T x - 1/2 u_f^T Sigma u_f + log prior_f
    #
    # where:
    #
    #   u_f = Sigma^{-1} Delta_f
    #
    # and Delta_f is computed from clone-balanced early means.
    # ============================================================

    import os
    import gzip
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.io import mmread
    from scipy.sparse import issparse
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
    from sklearn.decomposition import PCA

    warnings.filterwarnings("ignore")

    # ============================================================
    # CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_startpop_controls_full")
    os.makedirs(OUTDIR, exist_ok=True)

    COUNTS_PATH = os.path.join(SUPPL, "stateFate_inVitro_normed_counts.mtx.gz")
    GENES_PATH  = os.path.join(SUPPL, "stateFate_inVitro_gene_names.txt.gz")
    CLONE_PATH  = os.path.join(SUPPL, "stateFate_inVitro_clone_matrix.mtx.gz")
    META_PATH   = os.path.join(SUPPL, "stateFate_inVitro_metadata.txt.gz")

    TIME_COL = "Time point"
    CELLTYPE_COL = "Cell type annotation"
    WELL_COL = "Well"
    START_COL = "Starting population"

    # Run both for the strongest story.
    # Day 2 = harder, earlier prospective prediction.
    # Day 4 = later, easier prediction.
    EARLY_TIMES_TO_RUN = [2.0, 4.0]

    EARLY_CELLTYPE = "Undifferentiated"
    EARLY_WELL = None

    TERMINAL_TIME = 6.0
    TERMINAL_WELL = None

    EXCLUDE_FATES = {
        "Undifferentiated",
        "Unknown",
        "unknown",
        "nan",
        "NaN",
        "Ambiguous",
        "ambiguous",
        "None",
        "",
    }

    # Strict terminal labels, lenient early cells.
    CLONE_FILTER_GRID = [
        dict(
            name="strict_terminal_lenient_early",
            min_total=12,
            min_early=1,
            min_terminal=8,
            min_dom_count=6,
            min_dom_frac=0.85,
            max_entropy=0.65,
        ),
        dict(
            name="medium_terminal_lenient_early",
            min_total=10,
            min_early=1,
            min_terminal=5,
            min_dom_count=4,
            min_dom_frac=0.80,
            max_entropy=0.75,
        ),
        dict(
            name="lenient_terminal_still_qc",
            min_total=8,
            min_early=1,
            min_terminal=4,
            min_dom_count=3,
            min_dom_frac=0.75,
            max_entropy=0.85,
        ),
    ]

    MIN_CLONES_PER_FATE_GLOBAL = 5
    MIN_CLONES_PER_FATE_WITHIN_START = 5
    MAX_FATES = 5

    # Extra one-vs-rest row added to metric tables/plots.
    # This treats every (clone, candidate fate) pair as a binary prediction.
    GLOBAL_FATE_NAME = "GLOBAL_OVR"

    N_VAR_GENES = 500
    MAX_COV_CELLS = 50000

    RIDGE = 1e-6
    COV_SHRINK_TO_DIAG = 0.0

    # Use False for pure log-enrichment, True for posterior-like probabilities.
    USE_FATE_PRIOR = False

    N_NULLS = 100
    N_SPLITS = 5

    SEED = 0
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    plt.rcParams.update({"font.size": 14})
    sns.set_context("talk")

    # ============================================================
    # HELPERS
    # ============================================================


    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )

    def softmax_logits(logits, eps=1e-12):
        z = logits - np.max(logits, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.maximum(ez.sum(axis=1, keepdims=True), eps)

    def barplot_sd(*args, **kwargs):
        try:
            return sns.barplot(*args, errorbar="sd", **kwargs)
        except Exception:
            return sns.barplot(*args, ci="sd", **kwargs)

    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()
        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row
        return cell_to_clone

    def get_cells_x_genes(counts, cell_idx, gene_idx):
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)

    def zscore_train(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu, sd


    def fate_entropy_from_counts(counts_vec):
        counts_vec = np.asarray(counts_vec, dtype=float)
        counts_vec = counts_vec[counts_vec > 0]
        if counts_vec.size <= 1:
            return 0.0
        p = counts_vec / counts_vec.sum()
        return float(-(p * np.log(p)).sum())

    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        X = counts[:, cell_idx]
        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means**2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]

        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_var_genes:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_

    def make_covariance(X):
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D
        Sigma = Sigma + RIDGE * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)

    def clone_mean_matrix(clone_ids, early_mask, cell_to_clone, counts, hvg_idx, mu, sd):
        rows = []
        out_ids = []
        out_n = []

        for cid in clone_ids:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]
            if len(idx) == 0:
                continue

            X = get_cells_x_genes(counts, idx, hvg_idx)
            X = apply_zscore(X, mu, sd)

            rows.append(X.mean(axis=0))
            out_ids.append(cid)
            out_n.append(len(idx))

        if len(rows) == 0:
            return (
                np.empty((0, len(hvg_idx))),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        return np.vstack(rows), np.asarray(out_ids, dtype=int), np.asarray(out_n, dtype=int)

    def make_global_ovr_vectors(df, selected_fates, score_prefix="log_enrichment", label_col="true_future_lineage"):
        """
        Stack all one-vs-rest fate decisions into one global vector.

        For each sample i and fate f:
          y_if = 1[true_future_lineage_i == f]
          s_if = score_f(x_i)

        This gives a micro/global one-vs-rest AUROC/AUPRC across all fates.
        It is useful as a single cumulative/global summary in addition to
        per-fate AUROC/AUPRC.
        """
        y_all = []
        s_all = []
        fate_all = []

        labels = df[label_col].astype(str).values

        for fate in selected_fates:
            col = f"{score_prefix}__{safe_name(fate)}"
            if col not in df.columns:
                continue

            y = (labels == str(fate)).astype(int)
            s = df[col].values.astype(float)
            ok = np.isfinite(s)

            y_all.append(y[ok])
            s_all.append(s[ok])
            fate_all.extend([fate] * int(ok.sum()))

        if len(y_all) == 0:
            return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=object)

        y_all = np.concatenate(y_all).astype(int)
        s_all = np.concatenate(s_all).astype(float)
        fate_all = np.asarray(fate_all, dtype=object)

        return y_all, s_all, fate_all


    def compute_metrics(df, selected_fates, score_prefix="log_enrichment", label_col="true_future_lineage", add_global=True):
        rows = []

        for fate in selected_fates:
            col = f"{score_prefix}__{safe_name(fate)}"

            y = (df[label_col].astype(str).values == str(fate)).astype(int)
            s = df[col].values.astype(float)
            ok = np.isfinite(s)
            y = y[ok]
            s = s[ok]

            if len(np.unique(y)) < 2:
                auroc = np.nan
                auprc = np.nan
            else:
                auroc = roc_auc_score(y, s)
                auprc = average_precision_score(y, s)

            baseline = y.mean() if len(y) > 0 else np.nan
            if len(s) > 0:
                cutoff = np.quantile(s, 0.90)
                top = s >= cutoff
            else:
                top = np.array([], dtype=bool)

            if top.sum() > 0 and baseline > 0:
                top_rate = y[top].mean()
                enrichment = top_rate / baseline
            else:
                top_rate = np.nan
                enrichment = np.nan

            rows.append({
                "fate": fate,
                "n": len(y),
                "n_positive": int(y.sum()) if len(y) else 0,
                "positive_fraction": float(baseline) if np.isfinite(baseline) else np.nan,
                "AUROC": auroc,
                "AUPRC": auprc,
                "top_decile_positive_rate": top_rate,
                "top_decile_enrichment": enrichment,
                "metric_scope": "per_fate",
            })

        if add_global:
            y_global, s_global, _ = make_global_ovr_vectors(
                df=df,
                selected_fates=selected_fates,
                score_prefix=score_prefix,
                label_col=label_col,
            )

            if len(y_global) == 0 or len(np.unique(y_global)) < 2:
                auroc = np.nan
                auprc = np.nan
            else:
                auroc = roc_auc_score(y_global, s_global)
                auprc = average_precision_score(y_global, s_global)

            baseline = y_global.mean() if len(y_global) > 0 else np.nan
            if len(s_global) > 0:
                cutoff = np.quantile(s_global, 0.90)
                top = s_global >= cutoff
            else:
                top = np.array([], dtype=bool)

            if top.sum() > 0 and baseline > 0:
                top_rate = y_global[top].mean()
                enrichment = top_rate / baseline
            else:
                top_rate = np.nan
                enrichment = np.nan

            rows.append({
                "fate": GLOBAL_FATE_NAME,
                "n": len(y_global),
                "n_positive": int(y_global.sum()) if len(y_global) else 0,
                "positive_fraction": float(baseline) if np.isfinite(baseline) else np.nan,
                "AUROC": auroc,
                "AUPRC": auprc,
                "top_decile_positive_rate": top_rate,
                "top_decile_enrichment": enrichment,
                "metric_scope": "global_micro_ovr",
            })

        return pd.DataFrame(rows)


    def plot_global_ovr_curves(
        AOUT,
        analysis_label,
        selected_fates,
        clone_probs,
        early_cell_probs=None,
        startpop_clone_probs=None,
        score_prefix="log_enrichment",
    ):
        """
        Plot global one-vs-rest ROC, PR, and cumulative-gain curves.

        These curves pool all clone/fate binary decisions:
          (clone, Monocyte), (clone, Neutrophil), ...
        into a single global prediction problem.
        """
        curve_inputs = [("CIPHER clone", clone_probs)]

        if early_cell_probs is not None and len(early_cell_probs) > 0:
            curve_inputs.append(("CIPHER cell", early_cell_probs))

        if startpop_clone_probs is not None and len(startpop_clone_probs) > 0:
            curve_inputs.append(("starting-pop only clone", startpop_clone_probs))

        curve_rows = []
        gain_rows = []

        # ROC curve.
        plt.figure(figsize=(6.5, 5.5))
        for label, df in curve_inputs:
            y, s, _ = make_global_ovr_vectors(df, selected_fates, score_prefix=score_prefix)
            if len(y) == 0 or len(np.unique(y)) < 2:
                continue

            fpr, tpr, thr = roc_curve(y, s)
            auroc = roc_auc_score(y, s)
            plt.plot(fpr, tpr, linewidth=2, label=f"{label} AUROC={auroc:.3f}")

            for a, b, c in zip(fpr, tpr, thr):
                curve_rows.append({
                    "analysis": analysis_label,
                    "curve": "ROC",
                    "model_level": label,
                    "x": a,
                    "y": b,
                    "threshold": c,
                    "global_AUROC": auroc,
                    "global_AUPRC": average_precision_score(y, s),
                })

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title(f"{analysis_label}: global one-vs-rest ROC")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "global_OVR_ROC_curve.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "global_OVR_ROC_curve.svg"), bbox_inches="tight")
        plt.show()

        # Precision-recall curve.
        plt.figure(figsize=(6.5, 5.5))
        for label, df in curve_inputs:
            y, s, _ = make_global_ovr_vectors(df, selected_fates, score_prefix=score_prefix)
            if len(y) == 0 or len(np.unique(y)) < 2:
                continue

            precision, recall, thr = precision_recall_curve(y, s)
            auprc = average_precision_score(y, s)
            baseline = y.mean()
            plt.plot(recall, precision, linewidth=2, label=f"{label} AUPRC={auprc:.3f}")

            # precision_recall_curve returns len(thresholds)=len(precision)-1.
            thr_pad = np.r_[thr, np.nan]
            for a, b, c in zip(recall, precision, thr_pad):
                curve_rows.append({
                    "analysis": analysis_label,
                    "curve": "PR",
                    "model_level": label,
                    "x": a,
                    "y": b,
                    "threshold": c,
                    "global_AUROC": roc_auc_score(y, s),
                    "global_AUPRC": auprc,
                })

        if len(curve_inputs) > 0:
            y0, _, _ = make_global_ovr_vectors(curve_inputs[0][1], selected_fates, score_prefix=score_prefix)
            if len(y0) > 0:
                plt.axhline(y0.mean(), color="gray", linestyle="--", linewidth=1.5, label="global baseline")
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title(f"{analysis_label}: global one-vs-rest precision-recall")
        plt.ylim(0, 1.02)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "global_OVR_precision_recall_curve.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "global_OVR_precision_recall_curve.svg"), bbox_inches="tight")
        plt.show()

        # Cumulative-gain / cumulative-precision curve.
        plt.figure(figsize=(6.5, 5.5))
        for label, df in curve_inputs:
            y, s, _ = make_global_ovr_vectors(df, selected_fates, score_prefix=score_prefix)
            if len(y) == 0 or y.sum() == 0:
                continue

            order = np.argsort(s)[::-1]
            y_sorted = y[order]
            frac_screened = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
            cum_recall = np.cumsum(y_sorted) / np.maximum(y_sorted.sum(), 1)
            cum_precision = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)

            plt.plot(frac_screened, cum_recall, linewidth=2, label=f"{label} cumulative recall")

            for a, b, c in zip(frac_screened, cum_recall, cum_precision):
                gain_rows.append({
                    "analysis": analysis_label,
                    "model_level": label,
                    "fraction_scored": a,
                    "cumulative_recall": b,
                    "cumulative_precision": c,
                })

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)
        plt.xlabel("fraction of all clone-fate scores sorted by log-enrichment")
        plt.ylabel("cumulative recall of true clone-fate pairs")
        plt.title(f"{analysis_label}: global cumulative gain")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "global_OVR_cumulative_gain_curve.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "global_OVR_cumulative_gain_curve.svg"), bbox_inches="tight")
        plt.show()

        curve_df = pd.DataFrame(curve_rows)
        gain_df = pd.DataFrame(gain_rows)
        curve_df.to_csv(os.path.join(AOUT, "global_OVR_ROC_PR_curve_points.csv"), index=False)
        gain_df.to_csv(os.path.join(AOUT, "global_OVR_cumulative_gain_curve_points.csv"), index=False)

        return curve_df, gain_df

    def build_masks(meta, early_time, restrict_start=None):
        early_all_mask = meta[TIME_COL].astype(float).values == float(early_time)

        if EARLY_CELLTYPE is not None:
            early_all_mask &= meta[CELLTYPE_COL].astype(str).values == str(EARLY_CELLTYPE)

        if EARLY_WELL is not None and WELL_COL in meta.columns:
            early_all_mask &= meta[WELL_COL].astype(float).values == float(EARLY_WELL)

        if restrict_start is not None and START_COL in meta.columns:
            early_all_mask &= meta[START_COL].astype(str).values == str(restrict_start)

        terminal_mask = meta[TIME_COL].astype(float).values == float(TERMINAL_TIME)

        if TERMINAL_WELL is not None and WELL_COL in meta.columns:
            terminal_mask &= meta[WELL_COL].astype(float).values == float(TERMINAL_WELL)

        ann = meta[CELLTYPE_COL].astype(str).values
        terminal_mask &= ~np.isin(ann, list(EXCLUDE_FATES))

        return early_all_mask, terminal_mask

    def build_clone_table_with_filters(clone_mat, early_mask, terminal_mask, fate_labels, config):
        records = []

        for clone_id in range(clone_mat.shape[0]):
            cells = clone_mat[clone_id].indices

            if len(cells) < config["min_total"]:
                continue

            early_cells = cells[early_mask[cells]]
            terminal_cells = cells[terminal_mask[cells]]

            if len(early_cells) < config["min_early"]:
                continue
            if len(terminal_cells) < config["min_terminal"]:
                continue

            terminal_fates = pd.Series(fate_labels[terminal_cells].astype(str))
            terminal_fates = terminal_fates[~terminal_fates.isin(EXCLUDE_FATES)]

            if len(terminal_fates) < config["min_terminal"]:
                continue

            fate_counts = terminal_fates.value_counts()
            if len(fate_counts) == 0:
                continue

            dominant_fate = str(fate_counts.index[0])
            dominant_count = int(fate_counts.iloc[0])
            total_terminal = int(fate_counts.sum())
            dominant_frac = dominant_count / max(total_terminal, 1)
            entropy = fate_entropy_from_counts(fate_counts.values)

            if dominant_count < config["min_dom_count"]:
                continue
            if dominant_frac < config["min_dom_frac"]:
                continue
            if config["max_entropy"] is not None and entropy > config["max_entropy"]:
                continue

            rec = {
                "clone_id": int(clone_id),
                "n_total_clone_cells": int(len(cells)),
                "n_early": int(len(early_cells)),
                "n_terminal": int(total_terminal),
                "n_terminal_raw": int(len(terminal_cells)),
                "n_terminal_fate_types": int(len(fate_counts)),
                "dominant_fate": dominant_fate,
                "dominant_count": dominant_count,
                "dominant_frac": float(dominant_frac),
                "fate_entropy": float(entropy),
                "filter_config": config["name"],
            }

            for fate, count in fate_counts.items():
                s = safe_name(fate)
                rec[f"terminal_count__{s}"] = int(count)
                rec[f"terminal_frac__{s}"] = float(count / total_terminal)

            records.append(rec)

        return pd.DataFrame(records)

    def annotate_starting_population(clone_table, early_mask, cell_to_clone, meta):
        if START_COL not in meta.columns:
            clone_table["dominant_starting_population"] = "unknown"
            clone_table["dominant_starting_population_frac"] = 1.0
            return clone_table

        starts = []
        start_fracs = []

        for cid in clone_table["clone_id"].values:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]

            if len(idx) == 0:
                starts.append("unknown")
                start_fracs.append(np.nan)
                continue

            vc = meta.iloc[idx][START_COL].astype(str).value_counts()
            starts.append(vc.index[0])
            start_fracs.append(float(vc.iloc[0] / vc.sum()))

        clone_table = clone_table.copy()
        clone_table["dominant_starting_population"] = starts
        clone_table["dominant_starting_population_frac"] = start_fracs

        return clone_table

    def choose_clone_table_for_masks(
        clone_mat,
        early_cloned_mask,
        terminal_cloned_mask,
        fate_labels,
        min_clones_per_fate,
    ):
        tried = []

        for cfg in CLONE_FILTER_GRID:
            ct = build_clone_table_with_filters(
                clone_mat=clone_mat,
                early_mask=early_cloned_mask,
                terminal_mask=terminal_cloned_mask,
                fate_labels=fate_labels,
                config=cfg,
            )

            if ct.empty:
                tried.append((cfg["name"], 0, 0, []))
                continue

            ct = annotate_starting_population(ct, early_cloned_mask, cell_to_clone, meta)

            fate_counts = ct["dominant_fate"].value_counts()
            selected = fate_counts[fate_counts >= min_clones_per_fate].index.tolist()
            selected = selected[:MAX_FATES]

            tried.append((cfg["name"], len(ct), len(selected), selected))

            if len(selected) >= 2:
                return ct, selected, cfg, tried

        return None, None, None, tried

    def shuffle_labels_within_groups(y, groups):
        y = np.asarray(y).copy()
        groups = np.asarray(groups).astype(str)

        out = y.copy()

        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) > 1:
                out[idx] = rng.permutation(out[idx])

        return out

    def make_cipher_model(
        Xtrain_clone,
        y_train_for_delta,
        selected_fates,
        evals,
        evecs,
        Sigma,
        use_fate_prior=False,
    ):
        U = []
        DELTAS = []
        penalties = []
        log_priors = []

        eps = 1e-12

        for fate in selected_fates:
            pos = y_train_for_delta == fate
            neg = y_train_for_delta != fate

            if pos.sum() == 0 or neg.sum() == 0:
                raise RuntimeError(f"Missing positive/negative training clones for fate {fate}")

            delta = Xtrain_clone[pos].mean(axis=0) - Xtrain_clone[neg].mean(axis=0)

            u = evecs @ ((evecs.T @ delta) / evals)
            penalty = 0.5 * float(u @ Sigma @ u)

            if use_fate_prior:
                prior = float(np.mean(y_train_for_delta == fate))
                log_prior = np.log(max(prior, eps))
            else:
                log_prior = 0.0

            U.append(u)
            DELTAS.append(delta)
            penalties.append(penalty)
            log_priors.append(log_prior)

        return {
            "U": np.asarray(U),
            "DELTAS": np.asarray(DELTAS),
            "penalty": np.asarray(penalties),
            "log_prior": np.asarray(log_priors),
        }

    def score_matrix_with_model(X, model):
        U = model["U"]
        raw_scores = X @ U.T

        log_enrichment = (
            raw_scores
            - model["penalty"][None, :]
            + model["log_prior"][None, :]
        )

        p_norm = softmax_logits(log_enrichment)

        return raw_scores, log_enrichment, p_norm

    def rows_from_scores(base_df, model_name, raw_scores, log_enrichment, p_norm, selected_fates):
        pred_idx = np.argmax(p_norm, axis=1)
        pred_fates = np.array(selected_fates, dtype=object)[pred_idx]

        rows = base_df.copy()
        rows["model"] = model_name
        rows["predicted_lineage_norm"] = pred_fates
        rows["max_pseudoprob_norm"] = p_norm.max(axis=1)

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            rows[f"score_raw__{s}"] = raw_scores[:, j]
            rows[f"log_enrichment__{s}"] = log_enrichment[:, j]
            rows[f"p_norm__{s}"] = p_norm[:, j]

        rows["log_enrichment_true_future_lineage"] = [
            log_enrichment[i, selected_fates.index(tf)]
            for i, tf in enumerate(rows["true_future_lineage"].values)
        ]

        rows["p_norm_true_future_lineage"] = [
            p_norm[i, selected_fates.index(tf)]
            for i, tf in enumerate(rows["true_future_lineage"].values)
        ]

        return rows

    def fit_startpop_baseline(y_train, start_train, selected_fates, alpha=1.0):
        y_train = np.asarray(y_train).astype(str)
        start_train = np.asarray(start_train).astype(str)

        fates = list(selected_fates)
        starts = np.unique(start_train).tolist()

        global_counts = pd.Series(y_train).value_counts()
        global_probs = np.array([
            (global_counts.get(f, 0) + alpha) / (len(y_train) + alpha * len(fates))
            for f in fates
        ])
        global_probs = global_probs / global_probs.sum()

        table = {}

        for s in starts:
            idx = start_train == s
            ys = y_train[idx]
            counts = pd.Series(ys).value_counts()

            probs = np.array([
                (counts.get(f, 0) + alpha) / (len(ys) + alpha * len(fates))
                for f in fates
            ])
            probs = probs / probs.sum()
            table[s] = probs

        return {
            "fates": fates,
            "starts": starts,
            "table": table,
            "global_probs": global_probs,
        }

    def score_startpop_baseline(start_test, model):
        start_test = np.asarray(start_test).astype(str)
        fates = model["fates"]

        probs = []
        for s in start_test:
            if s in model["table"]:
                probs.append(model["table"][s])
            else:
                probs.append(model["global_probs"])

        p_norm = np.vstack(probs)
        log_scores = np.log(np.clip(p_norm, 1e-12, 1.0))
        raw_scores = log_scores.copy()

        return raw_scores, log_scores, p_norm

    def make_analysis_dir(label):
        path = os.path.join(OUTDIR, safe_name(label))
        os.makedirs(path, exist_ok=True)
        return path

    # ============================================================
    # LOAD DATA ONCE
    # ============================================================

    counts = mmread(COUNTS_PATH).T.tocsr()
    print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} cells | nnz={counts.nnz:,}")

    with gzip.open(GENES_PATH, "rt") as f:
        gene_names = np.array([line.strip() for line in f])
    print(f"Genes loaded: {len(gene_names)}")

    clone_mat = mmread(CLONE_PATH).T.tocsr()
    print(f"Clone matrix: {clone_mat.shape[0]} clones x {clone_mat.shape[1]} cells")
    print(f"% cells with clone label: {(clone_mat.sum(axis=0) > 0).mean() * 100:.2f}%")

    meta = pd.read_csv(META_PATH, sep="\t")
    print(f"Meta: {meta.shape[0]} rows x {meta.shape[1]} cols")
    print("Meta columns:", list(meta.columns))

    assert counts.shape[1] == meta.shape[0] == clone_mat.shape[1], "cells mismatch"
    assert counts.shape[0] == len(gene_names), "genes mismatch"

    meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")

    print("\nTimepoints:")
    print(np.sort(meta[TIME_COL].dropna().unique()))

    print("\nCell annotations:")
    print(meta[CELLTYPE_COL].value_counts())

    cell_to_clone = get_cell_to_clone(clone_mat)
    has_clone = cell_to_clone >= 0
    fate_labels = meta[CELLTYPE_COL].astype(str).values

    # ============================================================
    # CORE ANALYSIS FUNCTION
    # ============================================================

    def run_one_analysis(
        early_time,
        restrict_start=None,
        min_clones_per_fate=10,
        run_startpop_baseline=True,
    ):
        analysis_label = f"early{early_time}"
        if restrict_start is None:
            analysis_label += "_all_startpops"
        else:
            analysis_label += f"_within_{safe_name(restrict_start)}"

        AOUT = make_analysis_dir(analysis_label)

        print("\n" + "=" * 80)
        print(f"RUNNING ANALYSIS: {analysis_label}")
        print("=" * 80)

        early_all_mask, terminal_mask = build_masks(meta, early_time, restrict_start=restrict_start)

        early_cloned_mask = early_all_mask & has_clone
        terminal_cloned_mask = terminal_mask & has_clone

        early_all_idx = np.where(early_all_mask)[0]
        early_cloned_idx = np.where(early_cloned_mask)[0]
        terminal_cloned_idx = np.where(terminal_cloned_mask)[0]

        print(f"All early cells for Sigma: {len(early_all_idx):,}")
        print(f"Cloned early cells: {len(early_cloned_idx):,}")
        print(f"Cloned terminal cells: {len(terminal_cloned_idx):,}")

        if len(early_all_idx) == 0 or len(early_cloned_idx) == 0 or len(terminal_cloned_idx) == 0:
            print("[skip] missing early/terminal cells")
            return None

        clone_table_all, selected_fates, chosen_filter, tried_filters = choose_clone_table_for_masks(
            clone_mat=clone_mat,
            early_cloned_mask=early_cloned_mask,
            terminal_cloned_mask=terminal_cloned_mask,
            fate_labels=fate_labels,
            min_clones_per_fate=min_clones_per_fate,
        )

        print("\nClone-filter attempts:")
        for name, n_clones, n_fates, fates in tried_filters:
            print(f"  {name:28s} n_clones={n_clones:4d} n_fates={n_fates} fates={fates}")

        if clone_table_all is None:
            print("[skip] no valid clone table")
            return None

        clone_table = clone_table_all[
            clone_table_all["dominant_fate"].isin(selected_fates)
        ].copy()

        eligible_clones = clone_table["clone_id"].values.astype(int)
        eligible_early_mask = early_cloned_mask & np.isin(cell_to_clone, eligible_clones)
        eligible_early_idx = np.where(eligible_early_mask)[0]

        clone_to_fate = dict(zip(clone_table["clone_id"], clone_table["dominant_fate"]))
        clone_to_frac = dict(zip(clone_table["clone_id"], clone_table["dominant_frac"]))
        clone_to_n_total = dict(zip(clone_table["clone_id"], clone_table["n_total_clone_cells"]))
        clone_to_n_early = dict(zip(clone_table["clone_id"], clone_table["n_early"]))
        clone_to_n_terminal = dict(zip(clone_table["clone_id"], clone_table["n_terminal"]))
        clone_to_start = dict(zip(clone_table["clone_id"], clone_table["dominant_starting_population"]))

        print("\nSelected fates:")
        print(clone_table["dominant_fate"].value_counts())
        print("\nStarting populations:")
        print(clone_table["dominant_starting_population"].value_counts())
        print(f"\nEligible clones: {len(eligible_clones):,}")
        print(f"Eligible early cells: {len(eligible_early_idx):,}")

        clone_table_all.to_csv(os.path.join(AOUT, "clone_table_all_passing_qc.csv"), index=False)
        clone_table.to_csv(os.path.join(AOUT, "clone_table_selected_fates.csv"), index=False)

        # --------------------------
        # Clone QC plots.
        # --------------------------
        fig, axes = plt.subplots(2, 3, figsize=(20, 11))

        sns.countplot(data=clone_table, x="dominant_fate", order=selected_fates, ax=axes[0, 0])
        axes[0, 0].set_title("Selected clones per future fate")
        axes[0, 0].set_xlabel("future fate")
        axes[0, 0].set_ylabel("clone count")
        axes[0, 0].tick_params(axis="x", rotation=45)

        sns.histplot(data=clone_table, x="n_total_clone_cells", bins=40, ax=axes[0, 1])
        axes[0, 1].set_title("Total cells per retained clone")

        sns.histplot(data=clone_table, x="n_early", bins=30, ax=axes[0, 2])
        axes[0, 2].set_title("Early cells per retained clone")

        sns.histplot(data=clone_table, x="n_terminal", bins=40, ax=axes[1, 0])
        axes[1, 0].set_title("Terminal cells per retained clone")

        sns.scatterplot(
            data=clone_table,
            x="n_terminal",
            y="dominant_frac",
            hue="dominant_fate",
            hue_order=selected_fates,
            ax=axes[1, 1],
            s=45,
        )
        axes[1, 1].set_title("Clone purity vs terminal size")
        axes[1, 1].legend(fontsize=9, frameon=False)

        sns.scatterplot(
            data=clone_table,
            x="n_early",
            y="n_terminal",
            hue="dominant_fate",
            hue_order=selected_fates,
            ax=axes[1, 2],
            s=45,
        )
        axes[1, 2].set_title("Early vs terminal representation")
        axes[1, 2].legend(fontsize=9, frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "clone_qc_summary.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "clone_qc_summary.svg"), bbox_inches="tight")
        plt.show()

        if START_COL in meta.columns:
            plt.figure(figsize=(10, 5))
            tab = pd.crosstab(
                clone_table["dominant_fate"],
                clone_table["dominant_starting_population"]
            )
            tab = tab.reindex(selected_fates)
            sns.heatmap(tab, annot=True, fmt="d", cmap="viridis")
            plt.title("Future fate vs early starting population")
            plt.xlabel("dominant starting population among early cells")
            plt.ylabel("future fate")
            plt.tight_layout()
            plt.savefig(os.path.join(AOUT, "future_fate_vs_starting_population.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join(AOUT, "future_fate_vs_starting_population.svg"), bbox_inches="tight")
            plt.show()

        # --------------------------
        # HVGs and Sigma.
        # --------------------------
        print("\nSelecting HVGs and building Sigma...")

        hvg_idx, gene_vars = select_hvgs_sparse(
            counts=counts,
            cell_idx=early_all_idx,
            n_var_genes=N_VAR_GENES,
        )
        hvg_genes = gene_names[hvg_idx]

        pd.DataFrame({
            "gene": hvg_genes,
            "gene_index": hvg_idx,
            "early_variance": gene_vars[hvg_idx],
        }).to_csv(os.path.join(AOUT, "selected_early_hvgs.csv"), index=False)

        cov_idx = early_all_idx.copy()
        if len(cov_idx) > MAX_COV_CELLS:
            cov_idx = rng.choice(cov_idx, size=MAX_COV_CELLS, replace=False)

        Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx)
        mu_ref, sd_ref = zscore_train(Xcov_raw)
        Xcov = apply_zscore(Xcov_raw, mu_ref, sd_ref)

        Sigma = make_covariance(Xcov)

        evals, evecs = np.linalg.eigh(Sigma)
        evals = np.maximum(evals, 1e-8)

        pd.DataFrame({
            "rank": np.arange(1, len(evals) + 1),
            "eigenvalue": evals[::-1],
        }).to_csv(os.path.join(AOUT, "early_covariance_eigenvalues.csv"), index=False)

        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(evals) + 1), evals[::-1], marker="o", linewidth=1, markersize=3)
        plt.yscale("log")
        plt.xlabel("eigenvalue rank")
        plt.ylabel("eigenvalue")
        plt.title("Early progenitor covariance spectrum")
        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "early_covariance_spectrum.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "early_covariance_spectrum.svg"), bbox_inches="tight")
        plt.show()

        # --------------------------
        # CV.
        # --------------------------
        X_clones = clone_table["clone_id"].values.astype(int)
        y_clones = clone_table["dominant_fate"].values.astype(str)

        min_class_n = clone_table["dominant_fate"].value_counts().min()
        n_splits = int(min(N_SPLITS, min_class_n))

        if n_splits < 2:
            print(f"[skip] cannot do CV; smallest fate has {min_class_n} clones.")
            return None

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED,
        )

        print(f"\nUsing clone-level stratified {n_splits}-fold CV.")
        print(f"Running {N_NULLS} shuffled-label nulls per fold.")
        print("Global analysis uses start-pop-preserving shuffled null.")
        print("Within-start analyses use standard shuffled null because start pop is fixed.")

        all_cell_rows = []
        all_clone_rows = []
        force_rows = []
        null_metric_rows = []
        startpop_metric_rows = []
        startpop_clone_rows = []

        for fold, (train_pos, test_pos) in enumerate(splitter.split(X_clones, y_clones)):
            train_clones = X_clones[train_pos]
            test_clones = X_clones[test_pos]

            print(f"\nFold {fold + 1}/{n_splits}: train={len(train_clones)}, test={len(test_clones)}")

            Xtrain_clone, train_clone_ids_used, n_train_early = clone_mean_matrix(
                clone_ids=train_clones,
                early_mask=eligible_early_mask,
                cell_to_clone=cell_to_clone,
                counts=counts,
                hvg_idx=hvg_idx,
                mu=mu_ref,
                sd=sd_ref,
            )

            y_train = np.array([clone_to_fate[c] for c in train_clone_ids_used])
            start_train = np.array([clone_to_start.get(c, "unknown") for c in train_clone_ids_used])

            Xtest_clone, test_clone_ids_used, n_test_early = clone_mean_matrix(
                clone_ids=test_clones,
                early_mask=eligible_early_mask,
                cell_to_clone=cell_to_clone,
                counts=counts,
                hvg_idx=hvg_idx,
                mu=mu_ref,
                sd=sd_ref,
            )

            y_test_clone = np.array([clone_to_fate[c] for c in test_clone_ids_used])
            start_test_clone = np.array([clone_to_start.get(c, "unknown") for c in test_clone_ids_used])

            base_clone = pd.DataFrame({
                "analysis": analysis_label,
                "fold": fold,
                "level": "clone",
                "clone_id": test_clone_ids_used,
                "true_future_lineage": y_test_clone,
                "dominant_starting_population": start_test_clone,
                "true_future_lineage_frac": [clone_to_frac[c] for c in test_clone_ids_used],
                "n_early_scored": n_test_early,
                "n_total_clone_cells": [clone_to_n_total[c] for c in test_clone_ids_used],
                "n_early_clone_cells": [clone_to_n_early[c] for c in test_clone_ids_used],
                "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_clone_ids_used],
            })

            test_early_idx = np.where(eligible_early_mask & np.isin(cell_to_clone, test_clones))[0]
            Xtest_cell_raw = get_cells_x_genes(counts, test_early_idx, hvg_idx)
            Xtest_cell = apply_zscore(Xtest_cell_raw, mu_ref, sd_ref)

            test_cell_clone_ids = cell_to_clone[test_early_idx]
            y_test_cell = np.array([clone_to_fate[c] for c in test_cell_clone_ids])
            start_test_cell = np.array([clone_to_start.get(c, "unknown") for c in test_cell_clone_ids])

            base_cell = pd.DataFrame({
                "analysis": analysis_label,
                "fold": fold,
                "level": "cell",
                "cell_index": test_early_idx,
                "clone_id": test_cell_clone_ids,
                "true_future_lineage": y_test_cell,
                "dominant_starting_population": start_test_cell,
                "true_future_lineage_frac": [clone_to_frac[c] for c in test_cell_clone_ids],
                "n_total_clone_cells": [clone_to_n_total[c] for c in test_cell_clone_ids],
                "n_early_clone_cells": [clone_to_n_early[c] for c in test_cell_clone_ids],
                "n_terminal_clone_cells": [clone_to_n_terminal[c] for c in test_cell_clone_ids],
            })

            # Real CIPHER.
            cipher_model = make_cipher_model(
                Xtrain_clone=Xtrain_clone,
                y_train_for_delta=y_train,
                selected_fates=selected_fates,
                evals=evals,
                evecs=evecs,
                Sigma=Sigma,
                use_fate_prior=USE_FATE_PRIOR,
            )

            for Xscore, base_df, collector in [
                (Xtest_clone, base_clone, all_clone_rows),
                (Xtest_cell, base_cell, all_cell_rows),
            ]:
                raw_scores, log_enrichment, p_norm = score_matrix_with_model(Xscore, cipher_model)
                rows = rows_from_scores(base_df, "cipher", raw_scores, log_enrichment, p_norm, selected_fates)
                collector.append(rows)

            # Top genes.
            U = cipher_model["U"]
            DELTAS = cipher_model["DELTAS"]

            for j, fate in enumerate(selected_fates):
                u = U[j]
                delta = DELTAS[j]

                top_pos = np.argsort(u)[::-1][:50]
                top_neg = np.argsort(u)[:50]

                for rank, gi in enumerate(top_pos, start=1):
                    force_rows.append({
                        "analysis": analysis_label,
                        "fold": fold,
                        "model": "cipher",
                        "fate": fate,
                        "direction": "positive",
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),
                        "u": float(u[gi]),
                        "delta_early": float(delta[gi]),
                        "penalty": float(cipher_model["penalty"][j]),
                        "log_prior": float(cipher_model["log_prior"][j]),
                    })

                for rank, gi in enumerate(top_neg, start=1):
                    force_rows.append({
                        "analysis": analysis_label,
                        "fold": fold,
                        "model": "cipher",
                        "fate": fate,
                        "direction": "negative",
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),
                        "u": float(u[gi]),
                        "delta_early": float(delta[gi]),
                        "penalty": float(cipher_model["penalty"][j]),
                        "log_prior": float(cipher_model["log_prior"][j]),
                    })

            # Starting-population-only baseline.
            if run_startpop_baseline and restrict_start is None and START_COL in meta.columns:
                start_model = fit_startpop_baseline(
                    y_train=y_train,
                    start_train=start_train,
                    selected_fates=selected_fates,
                    alpha=1.0,
                )

                raw_scores, log_enrichment, p_norm = score_startpop_baseline(start_test_clone, start_model)
                rows = rows_from_scores(
                    base_clone,
                    "starting_population_only",
                    raw_scores,
                    log_enrichment,
                    p_norm,
                    selected_fates,
                )
                startpop_clone_rows.append(rows)

                m = compute_metrics(rows, selected_fates, score_prefix="log_enrichment")
                m["analysis"] = analysis_label
                m["model"] = "starting_population_only"
                m["fold"] = fold
                m["level"] = "clone"
                startpop_metric_rows.append(m)

            # Shuffled-label nulls.
            for null_id in range(N_NULLS):
                if restrict_start is None and START_COL in meta.columns:
                    y_train_null = shuffle_labels_within_groups(y_train, start_train)
                    null_type = "startpop_preserving_shuffled_null"
                else:
                    y_train_null = rng.permutation(y_train)
                    null_type = "shuffled_null"

                null_model = make_cipher_model(
                    Xtrain_clone=Xtrain_clone,
                    y_train_for_delta=y_train_null,
                    selected_fates=selected_fates,
                    evals=evals,
                    evecs=evecs,
                    Sigma=Sigma,
                    use_fate_prior=USE_FATE_PRIOR,
                )

                for level_name, Xscore, base_df in [
                    ("clone", Xtest_clone, base_clone),
                    ("cell", Xtest_cell, base_cell),
                ]:
                    raw_scores, log_enrichment, p_norm = score_matrix_with_model(Xscore, null_model)
                    tmp = rows_from_scores(
                        base_df,
                        null_type,
                        raw_scores,
                        log_enrichment,
                        p_norm,
                        selected_fates,
                    )

                    m = compute_metrics(tmp, selected_fates, score_prefix="log_enrichment")
                    m["analysis"] = analysis_label
                    m["model"] = null_type
                    m["fold"] = fold
                    m["null_id"] = null_id
                    m["level"] = level_name

                    null_metric_rows.append(m)

        early_cell_probs = pd.concat(all_cell_rows, ignore_index=True)
        clone_probs = pd.concat(all_clone_rows, ignore_index=True)
        force_df = pd.DataFrame(force_rows)
        null_metrics = pd.concat(null_metric_rows, ignore_index=True)

        if len(startpop_clone_rows) > 0:
            startpop_clone_probs = pd.concat(startpop_clone_rows, ignore_index=True)
            startpop_metrics = pd.concat(startpop_metric_rows, ignore_index=True)
        else:
            startpop_clone_probs = pd.DataFrame()
            startpop_metrics = pd.DataFrame()

        early_cell_probs.to_csv(os.path.join(AOUT, "early_cell_cipher_probs.csv"), index=False)
        clone_probs.to_csv(os.path.join(AOUT, "clone_cipher_probs.csv"), index=False)
        force_df.to_csv(os.path.join(AOUT, "cipher_top_force_genes.csv"), index=False)
        null_metrics.to_csv(os.path.join(AOUT, "null_metrics.csv"), index=False)
        startpop_clone_probs.to_csv(os.path.join(AOUT, "starting_population_only_clone_probs.csv"), index=False)
        startpop_metrics.to_csv(os.path.join(AOUT, "starting_population_only_metrics.csv"), index=False)

        # Real metrics.
        metric_rows = []
        for fold in sorted(clone_probs["fold"].unique()):
            df_clone = clone_probs[clone_probs["fold"] == fold].copy()
            df_cell = early_cell_probs[early_cell_probs["fold"] == fold].copy()

            m_clone = compute_metrics(df_clone, selected_fates, score_prefix="log_enrichment")
            m_clone["analysis"] = analysis_label
            m_clone["model"] = "cipher"
            m_clone["fold"] = fold
            m_clone["level"] = "clone"

            m_cell = compute_metrics(df_cell, selected_fates, score_prefix="log_enrichment")
            m_cell["analysis"] = analysis_label
            m_cell["model"] = "cipher"
            m_cell["fold"] = fold
            m_cell["level"] = "cell"

            metric_rows.append(m_clone)
            metric_rows.append(m_cell)

        cipher_metrics = pd.concat(metric_rows, ignore_index=True)

        metric_parts = [cipher_metrics, null_metrics]
        if len(startpop_metrics) > 0:
            metric_parts.append(startpop_metrics)

        all_metrics = pd.concat(metric_parts, ignore_index=True)
        all_metrics.to_csv(os.path.join(AOUT, "all_metrics.csv"), index=False)

        summary_metrics = (
            all_metrics
            .groupby(["analysis", "level", "model", "fate"], as_index=False)
            .agg(
                AUROC_mean=("AUROC", "mean"),
                AUROC_sd=("AUROC", "std"),
                AUPRC_mean=("AUPRC", "mean"),
                AUPRC_sd=("AUPRC", "std"),
                top_decile_enrichment_mean=("top_decile_enrichment", "mean"),
                top_decile_enrichment_sd=("top_decile_enrichment", "std"),
                n_positive_mean=("n_positive", "mean"),
                positive_fraction_mean=("positive_fraction", "mean"),
            )
        )
        summary_metrics.to_csv(os.path.join(AOUT, "summary_metrics.csv"), index=False)

        # Empirical p-values vs null.
        p_rows = []
        null_model_names = [m for m in null_metrics["model"].unique()]

        for null_model_name in null_model_names:
            for level in ["clone", "cell"]:
                for fate in list(selected_fates) + [GLOBAL_FATE_NAME]:
                    for metric_name in ["AUROC", "AUPRC", "top_decile_enrichment"]:
                        real_vals = cipher_metrics[
                            (cipher_metrics["level"] == level) &
                            (cipher_metrics["fate"] == fate)
                        ][metric_name].dropna().values

                        null_vals = null_metrics[
                            (null_metrics["level"] == level) &
                            (null_metrics["fate"] == fate) &
                            (null_metrics["model"] == null_model_name)
                        ][metric_name].dropna().values

                        if len(real_vals) == 0 or len(null_vals) == 0:
                            p_emp = np.nan
                            real_mean = np.nan
                            null_mean = np.nan
                        else:
                            real_mean = float(np.mean(real_vals))
                            null_mean = float(np.mean(null_vals))
                            p_emp = float((1 + np.sum(null_vals >= real_mean)) / (1 + len(null_vals)))

                        p_rows.append({
                            "analysis": analysis_label,
                            "level": level,
                            "fate": fate,
                            "metric": metric_name,
                            "null_model": null_model_name,
                            "cipher_mean": real_mean,
                            "null_mean": null_mean,
                            "empirical_p": p_emp,
                            "n_null": len(null_vals),
                        })

        pvals = pd.DataFrame(p_rows)
        pvals.to_csv(os.path.join(AOUT, "empirical_pvalues.csv"), index=False)

        # Accuracy.
        acc_rows = []
        for level_name, df in [("clone", clone_probs), ("cell", early_cell_probs)]:
            acc_rows.append({
                "analysis": analysis_label,
                "model": "cipher",
                "level": level_name,
                "argmax_accuracy": np.mean(df["predicted_lineage_norm"] == df["true_future_lineage"]),
                "mean_log_enrichment_true": df["log_enrichment_true_future_lineage"].mean(),
                "mean_p_true_norm": df["p_norm_true_future_lineage"].mean(),
            })

        if len(startpop_clone_probs) > 0:
            acc_rows.append({
                "analysis": analysis_label,
                "model": "starting_population_only",
                "level": "clone",
                "argmax_accuracy": np.mean(startpop_clone_probs["predicted_lineage_norm"] == startpop_clone_probs["true_future_lineage"]),
                "mean_log_enrichment_true": startpop_clone_probs["log_enrichment_true_future_lineage"].mean(),
                "mean_p_true_norm": startpop_clone_probs["p_norm_true_future_lineage"].mean(),
            })

        acc_df = pd.DataFrame(acc_rows)
        acc_df.to_csv(os.path.join(AOUT, "accuracy_summary.csv"), index=False)

        # Global one-vs-rest cumulative ROC/PR/cumulative-gain curves.
        global_curve_df, global_gain_df = plot_global_ovr_curves(
            AOUT=AOUT,
            analysis_label=analysis_label,
            selected_fates=selected_fates,
            clone_probs=clone_probs,
            early_cell_probs=early_cell_probs,
            startpop_clone_probs=startpop_clone_probs if len(startpop_clone_probs) > 0 else None,
            score_prefix="log_enrichment",
        )

        # --------------------------
        # Plots.
        # --------------------------
        plot_metrics = all_metrics[all_metrics["level"] == "clone"].copy()

        label_map = {
            "cipher": "CIPHER",
            "shuffled_null": "shuffled null",
            "startpop_preserving_shuffled_null": "startpop-preserving null",
            "starting_population_only": "starting-pop only",
        }
        plot_metrics["model_label"] = plot_metrics["model"].map(label_map).fillna(plot_metrics["model"])

        metric_fate_order = list(selected_fates) + [GLOBAL_FATE_NAME]

        for metric_name in ["AUROC", "AUPRC", "top_decile_enrichment"]:
            plt.figure(figsize=(14, 5))

            sns.boxplot(
                data=plot_metrics,
                x="fate",
                y=metric_name,
                hue="model_label",
                order=metric_fate_order,
                showfliers=False,
            )

            point_df = plot_metrics[
                plot_metrics["model"].isin(["cipher", "starting_population_only"])
            ].copy()

            if len(point_df) > 0:
                sns.stripplot(
                    data=point_df,
                    x="fate",
                    y=metric_name,
                    hue="model_label",
                    order=metric_fate_order,
                    dodge=True,
                    color="black",
                    alpha=0.65,
                    size=4,
                    legend=False,
                )

            if metric_name == "AUROC":
                plt.axhline(0.5, color="gray", linestyle="--", linewidth=2)
                plt.ylim(0, 1)
            elif metric_name == "AUPRC":
                plt.ylim(0, 1)
            else:
                plt.axhline(1.0, color="gray", linestyle="--", linewidth=2)

            plt.title(f"{analysis_label}: clone-level {metric_name}")
            plt.xlabel("future lineage")
            plt.ylabel(metric_name)
            plt.xticks(rotation=45, ha="right")

            handles, labels = plt.gca().get_legend_handles_labels()
            uniq = []
            uniq_labels = []
            for h, l in zip(handles, labels):
                if l not in uniq_labels:
                    uniq.append(h)
                    uniq_labels.append(l)
            plt.legend(uniq, uniq_labels, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.tight_layout()
            plt.savefig(os.path.join(AOUT, f"clone_{metric_name}_model_comparison.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join(AOUT, f"clone_{metric_name}_model_comparison.svg"), bbox_inches="tight")
            plt.show()

        # p-values plot.
        p_plot = pvals[
            (pvals["level"] == "clone") &
            (pvals["metric"] == "AUROC")
        ].copy()

        if len(p_plot) > 0:
            p_plot["minus_log10_p"] = -np.log10(np.maximum(p_plot["empirical_p"], 1e-300))

            plt.figure(figsize=(9, 5))
            sns.barplot(
                data=p_plot,
                x="fate",
                y="minus_log10_p",
                hue="null_model",
                order=metric_fate_order,
            )
            plt.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=2, label="p=0.05")
            plt.title(f"{analysis_label}: empirical null p-values")
            plt.xlabel("future lineage")
            plt.ylabel("-log10 empirical p, AUROC")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(AOUT, "empirical_pvalues_AUROC.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join(AOUT, "empirical_pvalues_AUROC.svg"), bbox_inches="tight")
            plt.show()

        # CIPHER probability heatmap + confusion matrix.
        main_clone = clone_probs.copy()
        p_norm_cols = [f"p_norm__{safe_name(f)}" for f in selected_fates]

        mean_prob = (
            main_clone
            .groupby("true_future_lineage")[p_norm_cols]
            .mean()
            .reindex(selected_fates)
        )
        mean_prob.columns = selected_fates

        cm = confusion_matrix(
            main_clone["true_future_lineage"],
            main_clone["predicted_lineage_norm"],
            labels=selected_fates,
        )
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        sns.heatmap(
            mean_prob,
            ax=axes[0],
            cmap="viridis",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "mean normalized pseudo-prob"},
        )
        axes[0].set_title("Clone mean CIPHER probabilities")
        axes[0].set_xlabel("predicted future lineage")
        axes[0].set_ylabel("true future lineage")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].tick_params(axis="y", rotation=0)

        sns.heatmap(
            pd.DataFrame(cm_norm, index=selected_fates, columns=selected_fates),
            ax=axes[1],
            cmap="viridis",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "row-normalized fraction"},
        )
        axes[1].set_title("Argmax CIPHER prediction")
        axes[1].set_xlabel("predicted future lineage")
        axes[1].set_ylabel("true future lineage")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].tick_params(axis="y", rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "CIPHER_probability_heatmap_confusion.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "CIPHER_probability_heatmap_confusion.svg"), bbox_inches="tight")
        plt.show()

        # p(true lineage).
        plt.figure(figsize=(11, 5))
        sns.boxplot(
            data=main_clone,
            x="true_future_lineage",
            y="p_norm_true_future_lineage",
            order=selected_fates,
            showfliers=False,
        )
        sns.stripplot(
            data=main_clone,
            x="true_future_lineage",
            y="p_norm_true_future_lineage",
            order=selected_fates,
            color="black",
            alpha=0.35,
            size=3,
        )
        plt.ylim(0, 1)
        plt.title(f"{analysis_label}: CIPHER p(true future lineage)")
        plt.xlabel("true future lineage")
        plt.ylabel("p(true future lineage | early clone)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "CIPHER_p_true_lineage_by_fate.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "CIPHER_p_true_lineage_by_fate.svg"), bbox_inches="tight")
        plt.show()

        # positive vs rest log-enrichment.
        score_rows = []
        for fate in selected_fates:
            col = f"log_enrichment__{safe_name(fate)}"
            tmp = main_clone[["clone_id", "true_future_lineage", col]].copy()
            tmp["tested_fate"] = fate
            tmp["is_future_fate"] = np.where(tmp["true_future_lineage"] == fate, "future fate", "other")
            tmp["log_enrichment"] = tmp[col]
            score_rows.append(tmp[["clone_id", "tested_fate", "is_future_fate", "log_enrichment"]])

        score_df = pd.concat(score_rows, ignore_index=True)

        plt.figure(figsize=(12, 5))
        sns.boxplot(
            data=score_df,
            x="tested_fate",
            y="log_enrichment",
            hue="is_future_fate",
            order=selected_fates,
            showfliers=False,
        )
        plt.title(f"{analysis_label}: CIPHER log-enrichment")
        plt.xlabel("tested lineage")
        plt.ylabel("full Gaussian log-enrichment")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "CIPHER_positive_vs_rest_log_enrichment.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "CIPHER_positive_vs_rest_log_enrichment.svg"), bbox_inches="tight")
        plt.show()

        # Top CIPHER genes heatmap.
        cipher_force = force_df[force_df["direction"] == "positive"].copy()

        mean_force = (
            cipher_force
            .groupby(["fate", "gene"], as_index=False)
            .agg(
                mean_u=("u", "mean"),
                mean_delta=("delta_early", "mean"),
                mean_rank=("rank", "mean"),
                mean_penalty=("penalty", "mean"),
            )
        )

        top_genes = []
        TOP_GENES_PER_FATE = 12
        for fate in selected_fates:
            sub = (
                mean_force[mean_force["fate"] == fate]
                .sort_values("mean_u", ascending=False)
                .head(TOP_GENES_PER_FATE)
            )
            top_genes.extend(sub["gene"].tolist())

        top_genes = list(dict.fromkeys(top_genes))

        heat = (
            mean_force
            .pivot_table(index="gene", columns="fate", values="mean_u", fill_value=0)
            .reindex(top_genes)
            .reindex(columns=selected_fates)
        )

        plt.figure(figsize=(1.4 * len(selected_fates) + 6, 0.28 * len(top_genes) + 4))
        sns.heatmap(
            heat,
            cmap="vlag",
            center=0,
            cbar_kws={"label": "mean CIPHER force u"},
        )
        plt.title(f"{analysis_label}: top positive early-bias CIPHER force genes")
        plt.xlabel("future lineage")
        plt.ylabel("gene")
        plt.tight_layout()
        plt.savefig(os.path.join(AOUT, "top_CIPHER_force_genes_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(AOUT, "top_CIPHER_force_genes_heatmap.svg"), bbox_inches="tight")
        plt.show()

        # PCA of early cells.
        MAX_PLOT_CELLS = 7000
        main_cell = early_cell_probs.copy()

        plot_df = main_cell.copy()
        if len(plot_df) > MAX_PLOT_CELLS:
            plot_df = plot_df.sample(MAX_PLOT_CELLS, random_state=SEED)

        if len(plot_df) > 3:
            plot_cells = plot_df["cell_index"].values.astype(int)
            X_plot = get_cells_x_genes(counts, plot_cells, hvg_idx)
            X_plot = apply_zscore(X_plot, mu_ref, sd_ref)

            Z = PCA(n_components=2, random_state=SEED).fit_transform(X_plot)

            plt.figure(figsize=(7, 6))
            sc = plt.scatter(
                Z[:, 0],
                Z[:, 1],
                c=plot_df["p_norm_true_future_lineage"].values,
                s=8,
                alpha=0.8,
                vmin=0,
                vmax=1,
                cmap="viridis",
            )
            plt.colorbar(sc, label="p(true future lineage | early cell)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"{analysis_label}: early cells")
            plt.tight_layout()
            plt.savefig(os.path.join(AOUT, "early_cells_pca_p_true_lineage.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join(AOUT, "early_cells_pca_p_true_lineage.svg"), bbox_inches="tight")
            plt.show()

        print("\nFinal clone-level CIPHER summary:")
        print(
            summary_metrics[
                (summary_metrics["level"] == "clone") &
                (summary_metrics["model"] == "cipher")
            ][[
                "fate",
                "n_positive_mean",
                "positive_fraction_mean",
                "AUROC_mean",
                "AUROC_sd",
                "AUPRC_mean",
                "AUPRC_sd",
                "top_decile_enrichment_mean",
            ]].sort_values("AUROC_mean", ascending=False)
        )

        print("\nEmpirical p-values, clone AUROC:")
        print(
            pvals[
                (pvals["level"] == "clone") &
                (pvals["metric"] == "AUROC")
            ][["fate", "null_model", "cipher_mean", "null_mean", "empirical_p", "n_null"]]
        )

        print("\nAccuracy:")
        print(acc_df)

        return {
            "analysis": analysis_label,
            "outdir": AOUT,
            "selected_fates": selected_fates,
            "clone_table": clone_table,
            "cipher_metrics": cipher_metrics,
            "null_metrics": null_metrics,
            "startpop_metrics": startpop_metrics,
            "all_metrics": all_metrics,
            "summary_metrics": summary_metrics,
            "pvals": pvals,
            "accuracy": acc_df,
            "global_curve_df": global_curve_df,
            "global_gain_df": global_gain_df,
        }

    # ============================================================
    # RUN GLOBAL + WITHIN-STARTING-POPULATION ANALYSES
    # ============================================================

    all_results = []

    for early_time in EARLY_TIMES_TO_RUN:
        # Global analysis: uses all starting populations and start-pop-preserving null.
        res_global = run_one_analysis(
            early_time=early_time,
            restrict_start=None,
            min_clones_per_fate=MIN_CLONES_PER_FATE_GLOBAL,
            run_startpop_baseline=True,
        )
        if res_global is not None:
            all_results.append(res_global)

        # Determine which starting populations exist at this early time.
        early_all_mask, _ = build_masks(meta, early_time, restrict_start=None)
        if START_COL in meta.columns:
            starts_this_time = (
                meta.loc[early_all_mask, START_COL]
                .astype(str)
                .value_counts()
                .index
                .tolist()
            )
        else:
            starts_this_time = []

        # Within-starting-population analyses.
        for start_pop in starts_this_time:
            res_within = run_one_analysis(
                early_time=early_time,
                restrict_start=start_pop,
                min_clones_per_fate=MIN_CLONES_PER_FATE_WITHIN_START,
                run_startpop_baseline=False,
            )
            if res_within is not None:
                all_results.append(res_within)

    # ============================================================
    # COMBINE ACROSS ANALYSES
    # ============================================================

    if len(all_results) > 0:
        combined_metrics = pd.concat([r["all_metrics"] for r in all_results], ignore_index=True)
        combined_summary = pd.concat([r["summary_metrics"] for r in all_results], ignore_index=True)
        combined_pvals = pd.concat([r["pvals"] for r in all_results], ignore_index=True)
        combined_accuracy = pd.concat([r["accuracy"] for r in all_results], ignore_index=True)
        combined_global_curves = pd.concat([r["global_curve_df"] for r in all_results if "global_curve_df" in r], ignore_index=True)
        combined_global_gains = pd.concat([r["global_gain_df"] for r in all_results if "global_gain_df" in r], ignore_index=True)

        combined_metrics.to_csv(os.path.join(OUTDIR, "combined_all_metrics.csv"), index=False)
        combined_summary.to_csv(os.path.join(OUTDIR, "combined_summary_metrics.csv"), index=False)
        combined_pvals.to_csv(os.path.join(OUTDIR, "combined_empirical_pvalues.csv"), index=False)
        combined_accuracy.to_csv(os.path.join(OUTDIR, "combined_accuracy.csv"), index=False)
        combined_global_curves.to_csv(os.path.join(OUTDIR, "combined_global_OVR_ROC_PR_curve_points.csv"), index=False)
        combined_global_gains.to_csv(os.path.join(OUTDIR, "combined_global_OVR_cumulative_gain_curve_points.csv"), index=False)

        # Summary plot across analyses: clone AUROC.
        plot_df = combined_metrics[combined_metrics["level"] == "clone"].copy()
        plot_df["model_label"] = plot_df["model"].map({
            "cipher": "CIPHER",
            "shuffled_null": "shuffled null",
            "startpop_preserving_shuffled_null": "startpop-preserving null",
            "starting_population_only": "starting-pop only",
        }).fillna(plot_df["model"])

        plt.figure(figsize=(16, 6))
        sns.boxplot(
            data=plot_df,
            x="analysis",
            y="AUROC",
            hue="model_label",
            showfliers=False,
        )
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=2)
        plt.ylim(0, 1)
        plt.title("CIPHER fate prediction across horizons and starting-pop controls")
        plt.xlabel("analysis")
        plt.ylabel("clone-level AUROC")
        plt.xticks(rotation=60, ha="right")
        plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "combined_clone_AUROC_by_analysis.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "combined_clone_AUROC_by_analysis.svg"), bbox_inches="tight")
        plt.show()


        # Global-only rows: easier summary of the cumulative/micro one-vs-rest metric.
        global_plot_df = plot_df[plot_df["fate"] == GLOBAL_FATE_NAME].copy()
        if len(global_plot_df) > 0:
            for metric_name in ["AUROC", "AUPRC", "top_decile_enrichment"]:
                plt.figure(figsize=(16, 6))
                sns.boxplot(
                    data=global_plot_df,
                    x="analysis",
                    y=metric_name,
                    hue="model_label",
                    showfliers=False,
                )
                if metric_name == "AUROC":
                    plt.axhline(0.5, color="gray", linestyle="--", linewidth=2)
                    plt.ylim(0, 1)
                elif metric_name == "AUPRC":
                    plt.ylim(0, 1)
                else:
                    plt.axhline(1.0, color="gray", linestyle="--", linewidth=2)
                plt.title(f"Global one-vs-rest {metric_name} across analyses")
                plt.xlabel("analysis")
                plt.ylabel(f"global clone-level {metric_name}")
                plt.xticks(rotation=60, ha="right")
                plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTDIR, f"combined_global_OVR_{metric_name}_by_analysis.png"), dpi=300, bbox_inches="tight")
                plt.savefig(os.path.join(OUTDIR, f"combined_global_OVR_{metric_name}_by_analysis.svg"), bbox_inches="tight")
                plt.show()

        print("\nCombined outputs saved in:", OUTDIR)
        print("\nCombined clone-level CIPHER summary:")
        print(
            combined_summary[
                (combined_summary["level"] == "clone") &
                (combined_summary["model"] == "cipher")
            ][[
                "analysis",
                "fate",
                "n_positive_mean",
                "AUROC_mean",
                "AUPRC_mean",
                "top_decile_enrichment_mean",
            ]].sort_values(["analysis", "AUROC_mean"], ascending=[True, False])
        )

        print("\nCombined empirical p-values, clone AUROC:")
        print(
            combined_pvals[
                (combined_pvals["level"] == "clone") &
                (combined_pvals["metric"] == "AUROC")
            ][[
                "analysis",
                "fate",
                "null_model",
                "cipher_mean",
                "null_mean",
                "empirical_p",
                "n_null",
            ]].sort_values(["analysis", "fate"])
        )

    else:
        print("No analyses completed. Relax filters or check EARLY_TIMES_TO_RUN.")



def day4_lfc_baseline_all_startpops():
    global os, gzip, warnings, np, pd, plt, sns, mmread, \
        issparse, StratifiedKFold, roc_auc_score, average_precision_score, OUTDIR, COUNTS_PATH, GENES_PATH, CLONE_PATH, \
        META_PATH, TIME_COL, CELLTYPE_COL, START_COL, WELL_COL, EARLY_TIME, EARLY_CELLTYPE, TERMINAL_TIME, \
        EXCLUDE_FATES, MIN_TOTAL_CELLS_PER_CLONE, MIN_EARLY_CELLS_PER_CLONE, MIN_TERMINAL_CELLS_PER_CLONE, MIN_DOMINANT_FATE_FRAC, MIN_DOMINANT_FATE_COUNT, MIN_CLONES_PER_FATE, MAX_FATES, \
        PREFERRED_FATE_ORDER, N_VAR_GENES, MAX_COV_CELLS, RIDGE, COV_SHRINK_TO_DIAG, LFC_PSEUDOCOUNT, LFC_CLIP, USE_FATE_PRIOR, \
        N_SPLITS, N_NULLS, SEED, GLOBAL_FATE_NAME, rng, safe_name, softmax_logits, get_cell_to_clone, \
        get_cells_x_genes, zscore_train, select_hvgs_sparse, make_covariance, clone_mean_raw_and_z, shuffle_labels_within_groups, score_cipher_model, fit_terminal_vs_undiff_lfc_model, \
        score_terminal_vs_undiff_lfc_model, fit_startpop_baseline, score_startpop_baseline, rows_from_scores, make_global_ovr_vectors, compute_metrics, counts, f, \
        gene_names, clone_mat, meta, cell_to_clone, has_clone, cell_fates, early_mask, terminal_mask, \
        early_cloned_mask, terminal_cloned_mask, records, clone_id, cells, early_cells, terminal_cells, fate_series, \
        vc, dominant_fate, dominant_count, total_terminal, dominant_frac, start_vc, dom_start, dom_start_frac, \
        clone_table, fate_counts, selected_fates, ordered, early_all_idx, hvg_idx, gene_vars, hvg_genes, \
        cov_idx, Xcov_raw, mu_ref, sd_ref, Xcov_z, Sigma, evals, evecs, \
        all_clone_ids, all_labels, all_starts, min_class_n, n_splits, splitter, all_pred_rows, all_metric_rows, \
        fold, train_pos, test_pos, train_clones, test_clones, Xraw_train, Xz_train, train_ids_used, \
        n_early_train, Xraw_test, Xz_test, test_ids_used, n_early_test, clone_to_label, clone_to_start, y_train, \
        y_test, start_train, start_test, base_test_df, cipher_model, raw, logits, probs, \
        pred_cipher, m, lfc_model, pred_lfc, sp_model, pred_sp, null_id, y_train_null, \
        null_model, pred_null, pred_df, metrics_df, pred_path, metrics_path, model_label_map, model_order_raw, \
        model_order_labels, plot_df, fate_order, palette, fig, axes, ax, metric_name, \
        point_df, tick, handles, labels, seen, uniq_handles, uniq_labels, h, \
        l, png, svg, pdf, summary, summary_path
    # ============================================================
    # CIPHER-LARRY controls: DAY 4 ALL STARTING POPULATIONS ONLY
    #
    # Correct LFC baseline:
    #
    #   LFC[f,g] = log((mean_raw(terminal cells of future fate f, g) + eps)
    #                  /(mean_raw(day-4 undifferentiated control cells, g) + eps))
    #
    # i.e.
    #   numerator   = terminal fate-f cells
    #   denominator = undifferentiated early cells / control state
    #
    # This is NOT Delta x.
    # This is true log fold-change ln(x_f / x_0).
    #
    # Models:
    #   1. CIPHER:
    #        u_f = Sigma^{-1} Delta_f
    #        Delta_f = mean_z(day4 undiff clones with future fate f)
    #                  - mean_z(day4 undiff clones with future fate not-f)
    #
    #   2. LFC terminal-vs-undiff baseline:
    #        w_f = log(mean_raw(terminal fate f) / mean_raw(day4 undiff))
    #
    #   3. startpop-preserving shuffled-label null:
    #        shuffle future fate labels within starting population, refit CIPHER
    #
    #   4. starting-pop only:
    #        predict future fate from dominant early starting population
    #
    # Output:
    #   OUTDIR/early4_all_startpops_predictions_with_terminal_LFC.csv
    #   OUTDIR/early4_all_startpops_metrics_with_terminal_LFC.csv
    #   OUTDIR/early4_all_startpops_clone_AUROC_AUPRC_with_terminal_LFC.png/svg/pdf
    # ============================================================

    import os
    import gzip
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.io import mmread
    from scipy.sparse import issparse
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score

    warnings.filterwarnings("ignore")

    # ============================================================
    # CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_day4_all_startpops_with_terminal_vs_undiff_LFC")
    os.makedirs(OUTDIR, exist_ok=True)

    COUNTS_PATH = os.path.join(SUPPL, "stateFate_inVitro_normed_counts.mtx.gz")
    GENES_PATH  = os.path.join(SUPPL, "stateFate_inVitro_gene_names.txt.gz")
    CLONE_PATH  = os.path.join(SUPPL, "stateFate_inVitro_clone_matrix.mtx.gz")
    META_PATH   = os.path.join(SUPPL, "stateFate_inVitro_metadata.txt.gz")

    TIME_COL = "Time point"
    CELLTYPE_COL = "Cell type annotation"
    START_COL = "Starting population"
    WELL_COL = "Well"

    EARLY_TIME = 4.0
    EARLY_CELLTYPE = "Undifferentiated"

    TERMINAL_TIME = 6.0

    EXCLUDE_FATES = {
        "Undifferentiated", "Unknown", "unknown", "nan", "NaN",
        "Ambiguous", "ambiguous", "None", ""
    }

    # Clone filtering.
    MIN_TOTAL_CELLS_PER_CLONE = 10
    MIN_EARLY_CELLS_PER_CLONE = 1
    MIN_TERMINAL_CELLS_PER_CLONE = 5
    MIN_DOMINANT_FATE_FRAC = 0.80
    MIN_DOMINANT_FATE_COUNT = 4
    MIN_CLONES_PER_FATE = 8
    MAX_FATES = 5

    PREFERRED_FATE_ORDER = [
        "Monocyte", "Neutrophil", "Baso", "Meg", "Mast", "Erythroid", "Eos"
    ]

    # Features.
    N_VAR_GENES = 500
    MAX_COV_CELLS = 50000

    # CIPHER covariance.
    RIDGE = 1e-6
    COV_SHRINK_TO_DIAG = 0.0

    # LFC.
    LFC_PSEUDOCOUNT = 1e-3
    LFC_CLIP = 8.0

    # Modeling.
    USE_FATE_PRIOR = False
    N_SPLITS = 5
    N_NULLS = 100
    SEED = 0

    GLOBAL_FATE_NAME = "GLOBAL_OVR"

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    sns.set_context("talk")
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    # ============================================================
    # HELPERS
    # ============================================================


    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )

    def softmax_logits(logits, eps=1e-12):
        logits = np.asarray(logits, dtype=float)
        z = logits - np.max(logits, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.maximum(ez.sum(axis=1, keepdims=True), eps)

    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()
        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row
        return cell_to_clone

    def get_cells_x_genes(counts, cell_idx, gene_idx):
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)

    def zscore_train(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu, sd


    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        X = counts[:, cell_idx]
        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means**2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]

        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_var_genes:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_

    def make_covariance(Xz):
        Xc = Xz - Xz.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D
        Sigma = Sigma + RIDGE * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)

    def clone_mean_raw_and_z(clone_ids, early_mask, cell_to_clone, counts, hvg_idx, mu_ref, sd_ref):
        raw_rows = []
        z_rows = []
        out_ids = []
        out_n = []

        for cid in clone_ids:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]
            if len(idx) == 0:
                continue

            Xraw = get_cells_x_genes(counts, idx, hvg_idx)
            Xz = apply_zscore(Xraw, mu_ref, sd_ref)

            raw_rows.append(Xraw.mean(axis=0))
            z_rows.append(Xz.mean(axis=0))
            out_ids.append(int(cid))
            out_n.append(len(idx))

        if len(raw_rows) == 0:
            n_genes = len(hvg_idx)
            return (
                np.empty((0, n_genes)),
                np.empty((0, n_genes)),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        return (
            np.vstack(raw_rows).astype(np.float64),
            np.vstack(z_rows).astype(np.float64),
            np.asarray(out_ids, dtype=int),
            np.asarray(out_n, dtype=int),
        )

    def shuffle_labels_within_groups(y, groups):
        y = np.asarray(y).copy()
        groups = np.asarray(groups).astype(str)
        out = y.copy()

        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) > 1:
                out[idx] = rng.permutation(out[idx])

        return out


    # ============================================================
    # MODEL HELPERS
    # ============================================================


    def score_cipher_model(Xz, model):
        raw = Xz @ model["W"].T
        logits = raw - model["penalty"][None, :] + model["log_prior"][None, :]
        probs = softmax_logits(logits)
        return raw, logits, probs

    def fit_terminal_vs_undiff_lfc_model(
        counts,
        hvg_idx,
        cell_to_clone,
        train_clone_ids,
        y_train_by_clone,
        selected_fates,
        early_undiff_mask,
        terminal_mask,
        cell_fates,
        pseudocount=1e-3,
        clip=8.0,
        use_prior=False,
    ):
        """
        Correct LFC model.

        For each fate f and gene g:

            LFC[f,g] = log((mean_raw(terminal cells of fate f, g) + eps)
                           /(mean_raw(day4 undifferentiated control cells, g) + eps))

        The denominator is the control/starting state.
        The numerator is terminal cells of that fate.
        Both are computed using training clones only to avoid leakage.
        """

        train_clone_ids = np.asarray(train_clone_ids, dtype=int)
        y_train_by_clone = np.asarray(y_train_by_clone).astype(str)

        # Control cells: day-4 undifferentiated cells from training clones.
        control_cells = cells_for_clone_set(
            cell_to_clone=cell_to_clone,
            clone_ids=train_clone_ids,
            mask=early_undiff_mask,
        )

        if len(control_cells) == 0:
            raise RuntimeError("No training-clone day-4 undifferentiated control cells for LFC denominator.")

        X0 = get_cells_x_genes(counts, control_cells, hvg_idx)
        mu0 = X0.mean(axis=0).astype(np.float64)

        W_lfc = []
        log_priors = []
        mu_f_all = []
        n_terminal_cells_by_fate = []

        eps = 1e-12

        for fate in selected_fates:
            # Training clones whose dominant future fate is f.
            fate_train_clones = train_clone_ids[y_train_by_clone == fate]

            # Numerator cells:
            # terminal cells that both:
            #   1. belong to training clones of future fate f
            #   2. are annotated as terminal cell type f
            fate_terminal_mask = terminal_mask & (cell_fates.astype(str) == str(fate))
            fate_terminal_cells = cells_for_clone_set(
                cell_to_clone=cell_to_clone,
                clone_ids=fate_train_clones,
                mask=fate_terminal_mask,
            )

            if len(fate_terminal_cells) == 0:
                # fallback: terminal cells annotated as this fate from all training clones
                # still no test-clone leakage
                fate_terminal_mask = terminal_mask & (cell_fates.astype(str) == str(fate))
                fate_terminal_cells = cells_for_clone_set(
                    cell_to_clone=cell_to_clone,
                    clone_ids=train_clone_ids,
                    mask=fate_terminal_mask,
                )

            if len(fate_terminal_cells) == 0:
                raise RuntimeError(f"No terminal training cells for fate {fate}")

            Xf = get_cells_x_genes(counts, fate_terminal_cells, hvg_idx)
            muf = Xf.mean(axis=0).astype(np.float64)

            lfc = np.log((muf + pseudocount) / (mu0 + pseudocount))

            if clip is not None:
                lfc = np.clip(lfc, -float(clip), float(clip))

            if use_prior:
                prior = max(float(np.mean(y_train_by_clone == fate)), eps)
                log_prior = np.log(prior)
            else:
                log_prior = 0.0

            W_lfc.append(lfc)
            log_priors.append(log_prior)
            mu_f_all.append(muf)
            n_terminal_cells_by_fate.append(len(fate_terminal_cells))

        return {
            "model_type": "terminal_vs_undiff_lfc",
            "W_lfc": np.asarray(W_lfc),
            "log_prior": np.asarray(log_priors),
            "mu0_undiff_raw": mu0,
            "muf_terminal_raw": np.asarray(mu_f_all),
            "n_control_cells": len(control_cells),
            "n_terminal_cells_by_fate": dict(zip(selected_fates, n_terminal_cells_by_fate)),
            "pseudocount": pseudocount,
        }

    def score_terminal_vs_undiff_lfc_model(Xraw_day4_undiff_clone_means, model):
        """
        Score day-4 undifferentiated clone means by their projection onto terminal-vs-undiff LFC vectors:

            score_f(x) = x^T LFC_f + log prior_f

        This tests whether early clone expression already aligns with terminal fate marker enrichment.
        """
        raw = Xraw_day4_undiff_clone_means @ model["W_lfc"].T
        logits = raw + model["log_prior"][None, :]
        probs = softmax_logits(logits)
        return raw, logits, probs

    def fit_startpop_baseline(y_train, start_train, selected_fates, alpha=1.0):
        y_train = np.asarray(y_train).astype(str)
        start_train = np.asarray(start_train).astype(str)

        fates = list(selected_fates)

        global_counts = pd.Series(y_train).value_counts()
        global_probs = np.array([
            (global_counts.get(f, 0) + alpha) / (len(y_train) + alpha * len(fates))
            for f in fates
        ])
        global_probs = global_probs / global_probs.sum()

        table = {}
        for s in np.unique(start_train):
            idx = start_train == s
            ys = y_train[idx]
            counts = pd.Series(ys).value_counts()

            probs = np.array([
                (counts.get(f, 0) + alpha) / (len(ys) + alpha * len(fates))
                for f in fates
            ])
            probs = probs / probs.sum()
            table[s] = probs

        return {"table": table, "global_probs": global_probs, "fates": fates}

    def score_startpop_baseline(start_test, model):
        P = []
        for s in np.asarray(start_test).astype(str):
            P.append(model["table"].get(s, model["global_probs"]))

        P = np.vstack(P)
        logits = np.log(np.clip(P, 1e-12, 1.0))
        raw = logits.copy()
        return raw, logits, P

    def rows_from_scores(base_df, model_name, raw_scores, logits, probs, selected_fates, null_id=None):
        rows = base_df.copy()
        rows["model"] = model_name
        rows["null_id"] = null_id

        pred_idx = np.argmax(probs, axis=1)
        rows["predicted_lineage_norm"] = np.asarray(selected_fates, dtype=object)[pred_idx]
        rows["max_pseudoprob_norm"] = probs.max(axis=1)

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            rows[f"score_raw__{s}"] = raw_scores[:, j]
            rows[f"log_enrichment__{s}"] = logits[:, j]
            rows[f"p_norm__{s}"] = probs[:, j]

        true_vals = rows["true_future_lineage"].astype(str).values
        idx_map = {f: j for j, f in enumerate(selected_fates)}

        rows["log_enrichment_true_future_lineage"] = [
            logits[i, idx_map[t]] if t in idx_map else np.nan
            for i, t in enumerate(true_vals)
        ]
        rows["p_norm_true_future_lineage"] = [
            probs[i, idx_map[t]] if t in idx_map else np.nan
            for i, t in enumerate(true_vals)
        ]

        return rows

    # ============================================================
    # METRICS
    # ============================================================

    def make_global_ovr_vectors(df, selected_fates, score_prefix="log_enrichment"):
        y_all = []
        s_all = []

        labels = df["true_future_lineage"].astype(str).values

        for fate in selected_fates:
            col = f"{score_prefix}__{safe_name(fate)}"
            if col not in df.columns:
                continue

            y = (labels == fate).astype(int)
            s = df[col].values.astype(float)
            ok = np.isfinite(s)

            y_all.append(y[ok])
            s_all.append(s[ok])

        if len(y_all) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        return np.concatenate(y_all), np.concatenate(s_all)

    def compute_metrics(pred_df, selected_fates):
        rows = []

        for fate in selected_fates:
            col = f"log_enrichment__{safe_name(fate)}"
            y = (pred_df["true_future_lineage"].astype(str).values == fate).astype(int)
            s = pred_df[col].values.astype(float)
            ok = np.isfinite(s)

            y = y[ok]
            s = s[ok]

            if len(np.unique(y)) < 2:
                auroc = np.nan
                auprc = np.nan
            else:
                auroc = roc_auc_score(y, s)
                auprc = average_precision_score(y, s)

            baseline = y.mean() if len(y) else np.nan
            if len(s) > 0:
                cutoff = np.quantile(s, 0.90)
                top = s >= cutoff
                top_rate = y[top].mean() if top.sum() > 0 else np.nan
                top_enrich = top_rate / baseline if baseline and baseline > 0 else np.nan
            else:
                top_rate = np.nan
                top_enrich = np.nan

            rows.append({
                "fate": fate,
                "metric_scope": "per_fate",
                "n": len(y),
                "n_positive": int(y.sum()) if len(y) else 0,
                "positive_fraction": baseline,
                "AUROC": auroc,
                "AUPRC": auprc,
                "top_decile_positive_rate": top_rate,
                "top_decile_enrichment": top_enrich,
            })

        yg, sg = make_global_ovr_vectors(pred_df, selected_fates)

        if len(yg) == 0 or len(np.unique(yg)) < 2:
            auroc = np.nan
            auprc = np.nan
        else:
            auroc = roc_auc_score(yg, sg)
            auprc = average_precision_score(yg, sg)

        baseline = yg.mean() if len(yg) else np.nan
        if len(sg) > 0:
            cutoff = np.quantile(sg, 0.90)
            top = sg >= cutoff
            top_rate = yg[top].mean() if top.sum() > 0 else np.nan
            top_enrich = top_rate / baseline if baseline and baseline > 0 else np.nan
        else:
            top_rate = np.nan
            top_enrich = np.nan

        rows.append({
            "fate": GLOBAL_FATE_NAME,
            "metric_scope": "global_micro_ovr",
            "n": len(yg),
            "n_positive": int(yg.sum()) if len(yg) else 0,
            "positive_fraction": baseline,
            "AUROC": auroc,
            "AUPRC": auprc,
            "top_decile_positive_rate": top_rate,
            "top_decile_enrichment": top_enrich,
        })

        return pd.DataFrame(rows)

    # ============================================================
    # LOAD DATA
    # ============================================================

    counts = mmread(COUNTS_PATH).T.tocsr()
    print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} cells | nnz={counts.nnz:,}")

    with gzip.open(GENES_PATH, "rt") as f:
        gene_names = np.array([line.strip() for line in f])
    print(f"Genes loaded: {len(gene_names)}")

    clone_mat = mmread(CLONE_PATH).T.tocsr()
    print(f"Clone matrix: {clone_mat.shape[0]} clones x {clone_mat.shape[1]} cells")

    meta = pd.read_csv(META_PATH, sep="\t")
    meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")

    assert counts.shape[1] == meta.shape[0] == clone_mat.shape[1], "cell mismatch"
    assert counts.shape[0] == len(gene_names), "gene mismatch"

    print(f"Meta: {meta.shape[0]} rows x {meta.shape[1]} cols")
    print("Meta columns:", list(meta.columns))

    cell_to_clone = get_cell_to_clone(clone_mat)
    has_clone = cell_to_clone >= 0
    cell_fates = meta[CELLTYPE_COL].astype(str).values

    # ============================================================
    # DEFINE DAY 4 UNDIF / TERMINAL CELLS
    # ============================================================

    early_mask = meta[TIME_COL].astype(float).values == float(EARLY_TIME)
    early_mask &= meta[CELLTYPE_COL].astype(str).values == str(EARLY_CELLTYPE)

    terminal_mask = meta[TIME_COL].astype(float).values == float(TERMINAL_TIME)
    terminal_mask &= ~np.isin(cell_fates, list(EXCLUDE_FATES))

    early_cloned_mask = early_mask & has_clone
    terminal_cloned_mask = terminal_mask & has_clone

    print(f"\nDay 4 undifferentiated cells: {early_mask.sum():,}")
    print(f"Day 4 cloned undifferentiated cells: {early_cloned_mask.sum():,}")
    print(f"Terminal cloned cells: {terminal_cloned_mask.sum():,}")

    # ============================================================
    # BUILD CLONE TABLE
    # ============================================================

    records = []

    for clone_id in range(clone_mat.shape[0]):
        cells = clone_mat[clone_id].indices

        if len(cells) < MIN_TOTAL_CELLS_PER_CLONE:
            continue

        early_cells = cells[early_cloned_mask[cells]]
        terminal_cells = cells[terminal_cloned_mask[cells]]

        if len(early_cells) < MIN_EARLY_CELLS_PER_CLONE:
            continue
        if len(terminal_cells) < MIN_TERMINAL_CELLS_PER_CLONE:
            continue

        fate_series = pd.Series(cell_fates[terminal_cells].astype(str))
        fate_series = fate_series[~fate_series.isin(EXCLUDE_FATES)]

        if len(fate_series) < MIN_TERMINAL_CELLS_PER_CLONE:
            continue

        vc = fate_series.value_counts()
        dominant_fate = str(vc.index[0])
        dominant_count = int(vc.iloc[0])
        total_terminal = int(vc.sum())
        dominant_frac = dominant_count / max(total_terminal, 1)

        if dominant_count < MIN_DOMINANT_FATE_COUNT:
            continue
        if dominant_frac < MIN_DOMINANT_FATE_FRAC:
            continue

        if START_COL in meta.columns:
            start_vc = meta.iloc[early_cells][START_COL].astype(str).value_counts()
            dom_start = str(start_vc.index[0])
            dom_start_frac = float(start_vc.iloc[0] / start_vc.sum())
        else:
            dom_start = "unknown"
            dom_start_frac = 1.0

        records.append({
            "clone_id": int(clone_id),
            "n_total_clone_cells": int(len(cells)),
            "n_early": int(len(early_cells)),
            "n_terminal": int(total_terminal),
            "dominant_fate": dominant_fate,
            "dominant_count": dominant_count,
            "dominant_frac": float(dominant_frac),
            "dominant_starting_population": dom_start,
            "dominant_starting_population_frac": dom_start_frac,
        })

    clone_table = pd.DataFrame(records)

    if clone_table.empty:
        raise RuntimeError("No clones passed filters.")

    fate_counts = clone_table["dominant_fate"].value_counts()

    selected_fates = fate_counts[fate_counts >= MIN_CLONES_PER_FATE].index.tolist()
    selected_fates = selected_fates[:MAX_FATES]

    ordered = [f for f in PREFERRED_FATE_ORDER if f in selected_fates]
    ordered += [f for f in selected_fates if f not in ordered]
    selected_fates = ordered

    clone_table = clone_table[clone_table["dominant_fate"].isin(selected_fates)].copy()
    clone_table = clone_table.reset_index(drop=True)

    print("\nSelected fates:")
    print(clone_table["dominant_fate"].value_counts())

    print("\nStarting populations among retained clones:")
    print(clone_table["dominant_starting_population"].value_counts())

    if len(selected_fates) < 2:
        raise RuntimeError("Need at least 2 selected fates.")

    # ============================================================
    # HVGs + COVARIANCE
    # ============================================================

    early_all_idx = np.where(early_mask)[0]

    print("\nSelecting HVGs from day-4 undifferentiated cells...")
    hvg_idx, gene_vars = select_hvgs_sparse(counts, early_all_idx, N_VAR_GENES)
    hvg_genes = gene_names[hvg_idx]

    pd.DataFrame({"gene": hvg_genes, "gene_idx": hvg_idx}).to_csv(
        os.path.join(OUTDIR, "selected_early_hvgs.csv"),
        index=False,
    )

    cov_idx = early_all_idx.copy()
    if len(cov_idx) > MAX_COV_CELLS:
        cov_idx = rng.choice(cov_idx, size=MAX_COV_CELLS, replace=False)

    Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx)
    mu_ref, sd_ref = zscore_train(Xcov_raw)
    Xcov_z = apply_zscore(Xcov_raw, mu_ref, sd_ref)

    Sigma = make_covariance(Xcov_z)
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-8)

    print(f"Using {len(hvg_idx)} HVGs.")
    print(f"Sigma shape: {Sigma.shape}")

    # ============================================================
    # CROSS-VALIDATION
    # ============================================================

    all_clone_ids = clone_table["clone_id"].values.astype(int)
    all_labels = clone_table["dominant_fate"].astype(str).values
    all_starts = clone_table["dominant_starting_population"].astype(str).values

    min_class_n = clone_table["dominant_fate"].value_counts().min()
    n_splits = min(N_SPLITS, int(min_class_n))

    if n_splits < 2:
        raise RuntimeError(f"Cannot do CV. Smallest class has {min_class_n} clones.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    all_pred_rows = []
    all_metric_rows = []

    for fold, (train_pos, test_pos) in enumerate(splitter.split(all_clone_ids, all_labels), start=1):
        print(f"\nFold {fold}/{n_splits}")

        train_clones = all_clone_ids[train_pos]
        test_clones = all_clone_ids[test_pos]

        Xraw_train, Xz_train, train_ids_used, n_early_train = clone_mean_raw_and_z(
            clone_ids=train_clones,
            early_mask=early_cloned_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu_ref=mu_ref,
            sd_ref=sd_ref,
        )

        Xraw_test, Xz_test, test_ids_used, n_early_test = clone_mean_raw_and_z(
            clone_ids=test_clones,
            early_mask=early_cloned_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu_ref=mu_ref,
            sd_ref=sd_ref,
        )

        clone_to_label = dict(zip(all_clone_ids, all_labels))
        clone_to_start = dict(zip(all_clone_ids, all_starts))

        y_train = np.array([clone_to_label[int(c)] for c in train_ids_used])
        y_test = np.array([clone_to_label[int(c)] for c in test_ids_used])

        start_train = np.array([clone_to_start[int(c)] for c in train_ids_used])
        start_test = np.array([clone_to_start[int(c)] for c in test_ids_used])

        base_test_df = pd.DataFrame({
            "analysis": "early4.0_all_startpops",
            "level": "clone",
            "fold": fold,
            "clone_id": test_ids_used,
            "true_future_lineage": y_test,
            "dominant_starting_population": start_test,
            "n_early": n_early_test,
        })

        # -----------------------
        # CIPHER
        # -----------------------
        cipher_model = fit_cipher_model(
            Xz_train=Xz_train,
            y_train=y_train,
            selected_fates=selected_fates,
            Sigma=Sigma,
            evals=evals,
            evecs=evecs,
            use_prior=USE_FATE_PRIOR,
        )

        raw, logits, probs = score_cipher_model(Xz_test, cipher_model)

        pred_cipher = rows_from_scores(
            base_test_df,
            "cipher",
            raw,
            logits,
            probs,
            selected_fates,
            null_id=None,
        )

        all_pred_rows.append(pred_cipher)

        m = compute_metrics(pred_cipher, selected_fates)
        m["model"] = "cipher"
        m["fold"] = fold
        m["null_id"] = np.nan
        all_metric_rows.append(m)

        # -----------------------
        # CORRECT TERMINAL-vs-UNDIF LFC BASELINE
        # -----------------------
        lfc_model = fit_terminal_vs_undiff_lfc_model(
            counts=counts,
            hvg_idx=hvg_idx,
            cell_to_clone=cell_to_clone,
            train_clone_ids=train_ids_used,
            y_train_by_clone=y_train,
            selected_fates=selected_fates,
            early_undiff_mask=early_cloned_mask,
            terminal_mask=terminal_cloned_mask,
            cell_fates=cell_fates,
            pseudocount=LFC_PSEUDOCOUNT,
            clip=LFC_CLIP,
            use_prior=USE_FATE_PRIOR,
        )

        print("  LFC denominator n day4-undiff control cells:", lfc_model["n_control_cells"])
        print("  LFC numerator n terminal cells:", lfc_model["n_terminal_cells_by_fate"])

        raw, logits, probs = score_terminal_vs_undiff_lfc_model(Xraw_test, lfc_model)

        pred_lfc = rows_from_scores(
            base_test_df,
            "terminal_vs_undiff_LFC",
            raw,
            logits,
            probs,
            selected_fates,
            null_id=None,
        )

        all_pred_rows.append(pred_lfc)

        m = compute_metrics(pred_lfc, selected_fates)
        m["model"] = "terminal_vs_undiff_LFC"
        m["fold"] = fold
        m["null_id"] = np.nan
        all_metric_rows.append(m)

        # -----------------------
        # STARTING-POP ONLY
        # -----------------------
        sp_model = fit_startpop_baseline(y_train, start_train, selected_fates, alpha=1.0)
        raw, logits, probs = score_startpop_baseline(start_test, sp_model)

        pred_sp = rows_from_scores(
            base_test_df,
            "starting_population_only",
            raw,
            logits,
            probs,
            selected_fates,
            null_id=None,
        )

        all_pred_rows.append(pred_sp)

        m = compute_metrics(pred_sp, selected_fates)
        m["model"] = "starting_population_only"
        m["fold"] = fold
        m["null_id"] = np.nan
        all_metric_rows.append(m)

        # -----------------------
        # STARTPOP-PRESERVING SHUFFLED CIPHER NULL
        # -----------------------
        for null_id in range(N_NULLS):
            y_train_null = shuffle_labels_within_groups(y_train, start_train)

            null_model = fit_cipher_model(
                Xz_train=Xz_train,
                y_train=y_train_null,
                selected_fates=selected_fates,
                Sigma=Sigma,
                evals=evals,
                evecs=evecs,
                use_prior=USE_FATE_PRIOR,
            )

            raw, logits, probs = score_cipher_model(Xz_test, null_model)

            pred_null = rows_from_scores(
                base_test_df,
                "startpop_preserving_null",
                raw,
                logits,
                probs,
                selected_fates,
                null_id=null_id,
            )

            all_pred_rows.append(pred_null)

            m = compute_metrics(pred_null, selected_fates)
            m["model"] = "startpop_preserving_null"
            m["fold"] = fold
            m["null_id"] = null_id
            all_metric_rows.append(m)

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    pred_df = pd.concat(all_pred_rows, ignore_index=True)
    metrics_df = pd.concat(all_metric_rows, ignore_index=True)

    # ------------------------------------------------------------
    # SEM-across-fates for the GLOBAL_OVR rows.
    #
    # This column is consumed by the standalone plot_auroc_auprc()
    # section, which overlays SEM-across-fates error bars on the
    # GLOBAL_OVR points. For each (model, fold, null_id) group it is
    # the standard error of that group's per-fate AUROC / AUPRC
    # values; it is only defined on the GLOBAL_OVR rows (NaN
    # elsewhere).
    # ------------------------------------------------------------
    for _se_metric in ("AUROC", "AUPRC"):
        metrics_df[f"{_se_metric}_SE_across_fates"] = np.nan

    _perfate = metrics_df[metrics_df["fate"] != GLOBAL_FATE_NAME]
    for _grp_key, _grp in _perfate.groupby(["model", "fold", "null_id"], dropna=False):
        _model, _fold, _null_id = _grp_key
        _global_mask = (
            (metrics_df["model"] == _model)
            & (metrics_df["fold"] == _fold)
            & (metrics_df["fate"] == GLOBAL_FATE_NAME)
        )
        if pd.isna(_null_id):
            _global_mask &= metrics_df["null_id"].isna()
        else:
            _global_mask &= (metrics_df["null_id"] == _null_id)
        for _se_metric in ("AUROC", "AUPRC"):
            _vals = _grp[_se_metric].dropna().values
            if len(_vals) > 1:
                _se = _vals.std(ddof=1) / np.sqrt(len(_vals))
            else:
                _se = np.nan
            metrics_df.loc[_global_mask, f"{_se_metric}_SE_across_fates"] = _se

    pred_path = os.path.join(OUTDIR, "early4_all_startpops_predictions_with_terminal_LFC.csv")
    metrics_path = os.path.join(OUTDIR, "early4_all_startpops_metrics_with_terminal_LFC.csv")

    pred_df.to_csv(pred_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    print("\nSaved:")
    print(" ", pred_path)
    print(" ", metrics_path)

    # ============================================================
    # PLOT AUROC/AUPRC
    # ============================================================

    model_label_map = {
        "cipher": "CIPHER",
        "terminal_vs_undiff_LFC": "terminal-vs-undiff LFC",
        "starting_population_only": "starting-pop only",
        "startpop_preserving_null": "startpop-preserving null",
    }

    model_order_raw = [
        "cipher",
        "terminal_vs_undiff_LFC",
        "startpop_preserving_null",
        "starting_population_only",
    ]

    model_order_labels = [model_label_map[m] for m in model_order_raw]

    plot_df = metrics_df.copy()
    plot_df["model_label"] = plot_df["model"].map(model_label_map).fillna(plot_df["model"])

    fate_order = [f for f in PREFERRED_FATE_ORDER if f in selected_fates]
    fate_order += [f for f in selected_fates if f not in fate_order]
    fate_order += [GLOBAL_FATE_NAME]

    palette = {
        "CIPHER": sns.color_palette("tab10")[0],
        "terminal-vs-undiff LFC": sns.color_palette("tab10")[3],
        "startpop-preserving null": sns.color_palette("tab10")[1],
        "starting-pop only": sns.color_palette("tab10")[2],
    }

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    for ax, metric_name in zip(axes, ["AUROC", "AUPRC"]):
        sns.boxplot(
            data=plot_df,
            x="fate",
            y=metric_name,
            hue="model_label",
            order=fate_order,
            hue_order=model_order_labels,
            palette=palette,
            showfliers=False,
            linewidth=1.5,
            ax=ax,
        )

        point_df = plot_df[
            plot_df["model"].isin([
                "cipher",
                "terminal_vs_undiff_LFC",
                "starting_population_only",
            ])
        ].copy()

        try:
            sns.stripplot(
                data=point_df,
                x="fate",
                y=metric_name,
                hue="model_label",
                order=fate_order,
                hue_order=model_order_labels,
                dodge=True,
                color="black",
                alpha=0.6,
                size=4,
                jitter=0.12,
                legend=False,
                ax=ax,
            )
        except TypeError:
            sns.stripplot(
                data=point_df,
                x="fate",
                y=metric_name,
                hue="model_label",
                order=fate_order,
                hue_order=model_order_labels,
                dodge=True,
                color="black",
                alpha=0.6,
                size=4,
                jitter=0.12,
                ax=ax,
            )

        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_yticklabels([f"{x:.1f}" for x in np.arange(0, 1.01, 0.2)])

        if metric_name == "AUROC":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=2)

        ax.set_title(f"early4.0_all_startpops: clone-level {metric_name}")
        ax.set_ylabel(metric_name)
        ax.set_xlabel("")

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    axes[-1].set_xlabel("future lineage")
    axes[-1].tick_params(axis="x", rotation=45)
    for tick in axes[-1].get_xticklabels():
        tick.set_ha("right")

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in model_order_labels and l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)

    fig.legend(
        uniq_handles,
        uniq_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 0.84, 1])

    png = os.path.join(OUTDIR, "early4_all_startpops_clone_AUROC_AUPRC_with_terminal_LFC.png")
    svg = os.path.join(OUTDIR, "early4_all_startpops_clone_AUROC_AUPRC_with_terminal_LFC.svg")
    pdf = os.path.join(OUTDIR, "early4_all_startpops_clone_AUROC_AUPRC_with_terminal_LFC.pdf")

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.show()

    print("\nSaved plot:")
    print(" ", png)
    print(" ", svg)
    print(" ", pdf)

    # ============================================================
    # SUMMARY
    # ============================================================

    summary = (
        plot_df
        .groupby(["fate", "model_label"], as_index=False)
        .agg(
            AUROC_mean=("AUROC", "mean"),
            AUROC_sd=("AUROC", "std"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_sd=("AUPRC", "std"),
            n=("AUROC", "size"),
        )
    )

    summary["fate"] = pd.Categorical(summary["fate"], categories=fate_order, ordered=True)
    summary["model_label"] = pd.Categorical(summary["model_label"], categories=model_order_labels, ordered=True)
    summary = summary.sort_values(["fate", "model_label"])

    summary_path = os.path.join(OUTDIR, "early4_all_startpops_summary_with_terminal_LFC.csv")
    summary.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nSaved summary:")
    print(" ", summary_path)



def plot_auroc_auprc():
    global model_label_map, model_order_raw, model_order_labels, plot_df, fate_order, palette, fig, axes, \
        ax, metric_name, sem_col, point_df, global_point_df, n_hue, group_width, global_x, \
        _, r, label, hue_idx, offset, x, y, yerr, \
        tick, handles, labels, seen, uniq_handles, uniq_labels, h, l, \
        png, svg, pdf
    # ============================================================
    # PLOT AUROC/AUPRC
    # ============================================================

    model_label_map = {
        "cipher": "CIPHER:\ncovariance-corrected early fate score",
        "terminal_vs_undiff_LFC": "Terminal marker\nLFC score",
        "startpop_preserving_null": "Shuffled fate labels\nwithin start population",
        "starting_population_only": "Start population\nonly",
    }

    model_order_raw = [
        "cipher",
        "terminal_vs_undiff_LFC",
        "startpop_preserving_null",
        "starting_population_only",
    ]

    model_order_labels = [model_label_map[m] for m in model_order_raw]

    plot_df = metrics_df.copy()
    plot_df["model_label"] = plot_df["model"].map(model_label_map).fillna(plot_df["model"])

    fate_order = [f for f in PREFERRED_FATE_ORDER if f in selected_fates]
    fate_order += [f for f in selected_fates if f not in fate_order]
    fate_order += [GLOBAL_FATE_NAME]

    palette = {
        "CIPHER:\ncovariance-corrected early fate score": sns.color_palette("tab10")[0],
        "Terminal marker\nLFC score": sns.color_palette("tab10")[3],
        "Shuffled fate labels\nwithin start population": sns.color_palette("tab10")[1],
        "Start population\nonly": sns.color_palette("tab10")[2],
    }

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    for ax, metric_name, sem_col in zip(
        axes,
        ["AUROC", "AUPRC"],
        ["AUROC_SE_across_fates", "AUPRC_SE_across_fates"],
    ):
        sns.boxplot(
            data=plot_df,
            x="fate",
            y=metric_name,
            hue="model_label",
            order=fate_order,
            hue_order=model_order_labels,
            palette=palette,
            showfliers=False,
            linewidth=1.5,
            ax=ax,
        )

        point_df = plot_df[
            plot_df["model"].isin([
                "cipher",
                "terminal_vs_undiff_LFC",
                "starting_population_only",
            ])
        ].copy()

        try:
            sns.stripplot(
                data=point_df,
                x="fate",
                y=metric_name,
                hue="model_label",
                order=fate_order,
                hue_order=model_order_labels,
                dodge=True,
                color="black",
                alpha=0.6,
                size=4,
                jitter=0.12,
                legend=False,
                ax=ax,
            )
        except TypeError:
            sns.stripplot(
                data=point_df,
                x="fate",
                y=metric_name,
                hue="model_label",
                order=fate_order,
                hue_order=model_order_labels,
                dodge=True,
                color="black",
                alpha=0.6,
                size=4,
                jitter=0.12,
                ax=ax,
            )

        # Overlay SEM-across-fates error bars on GLOBAL_MEAN points
        # for the real, non-null models only.
        global_point_df = point_df[point_df["fate"] == GLOBAL_FATE_NAME].copy()

        n_hue = len(model_order_labels)
        group_width = 0.8
        global_x = fate_order.index(GLOBAL_FATE_NAME)

        for _, r in global_point_df.iterrows():
            label = r["model_label"]
            if label not in model_order_labels:
                continue

            hue_idx = model_order_labels.index(label)
            offset = (hue_idx - (n_hue - 1) / 2.0) * (group_width / n_hue)

            x = global_x + offset
            y = r[metric_name]
            yerr = r[sem_col]

            if np.isfinite(y) and np.isfinite(yerr):
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.2,
                    capsize=3,
                    alpha=0.75,
                    zorder=20,
                )

        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_yticklabels([f"{x:.1f}" for x in np.arange(0, 1.01, 0.2)])

        if metric_name == "AUROC":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=2)

        ax.set_title(f"early4.0 all start populations: clone-level {metric_name}")
        ax.set_ylabel(metric_name)
        ax.set_xlabel("")

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    axes[-1].set_xlabel("future lineage")
    axes[-1].tick_params(axis="x", rotation=45)
    for tick in axes[-1].get_xticklabels():
        tick.set_ha("right")

    handles, labels = axes[0].get_legend_handles_labels()

    seen = set()
    uniq_handles = []
    uniq_labels = []

    for h, l in zip(handles, labels):
        if l in model_order_labels and l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)

    fig.legend(
        uniq_handles,
        uniq_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="score / control",
        title_fontsize=15,
    )

    fig.tight_layout(rect=[0, 0, 0.80, 1])

    png = os.path.join(
        OUTDIR,
        "early4_all_startpops_clone_AUROC_AUPRC_with_readable_labels.png",
    )
    svg = os.path.join(
        OUTDIR,
        "early4_all_startpops_clone_AUROC_AUPRC_with_readable_labels.svg",
    )
    pdf = os.path.join(
        OUTDIR,
        "early4_all_startpops_clone_AUROC_AUPRC_with_readable_labels.pdf",
    )

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.show()

    print("\nSaved plot:")
    print(" ", png)
    print(" ", svg)
    print(" ", pdf)



def plot_multipanel_day4_all_startpops():
    global os, np, pd, plt, sns, OUTDIR, ANALYSIS_TO_PLOT, GLOBAL_FATE_NAME, \
        SAVE_PREFIX, metrics_df, metrics_path, model_label_map, model_order_raw, model_order_labels, plot_df, required_cols, \
        missing, preferred_fates, present_fates, fate_order, extra_fates, point_df, palette, add_stripplot_no_null, \
        fig, axes, ax, metric_name, tick, handles, labels, seen, \
        uniq_handles, uniq_labels, h, l, png_path, svg_path, pdf_path, summary
    # ============================================================
    # Single multipanel day-4/all-startpops plot:
    #   clone-level AUROC + AUPRC
    #
    # Fixes:
    #   1. Only plots early4.0_all_startpops
    #   2. Makes one 2-row multipanel figure
    #   3. Keeps orange null as boxplot only
    #   4. Only overlays black points for CIPHER + starting-pop only
    #   5. y-range is 0 to 1.1, but tick labels only go up to 1.0
    # ============================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # -----------------------------
    # Config
    # -----------------------------
    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_startpop_controls_full")
    ANALYSIS_TO_PLOT = "early4.0_all_startpops"
    GLOBAL_FATE_NAME = "GLOBAL_OVR"

    SAVE_PREFIX = "early4_all_startpops_clone_AUROC_AUPRC_multipanel_clean"

    os.makedirs(OUTDIR, exist_ok=True)

    sns.set_context("talk")
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    })

    # -----------------------------
    # Load combined metrics
    # -----------------------------
    if "combined_metrics" in globals():
        metrics_df = combined_metrics.copy()
    else:
        metrics_path = os.path.join(OUTDIR, "combined_all_metrics.csv")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"Could not find {metrics_path}. Run the full analysis first "
                f"or make sure combined_metrics is already loaded."
            )
        metrics_df = pd.read_csv(metrics_path)

    # -----------------------------
    # Standardize model names
    # -----------------------------
    model_label_map = {
        "cipher": "CIPHER",
        "startpop_preserving_shuffled_null": "startpop-preserving null",
        "starting_population_only": "starting-pop only",
        "shuffled_null": "shuffled null",
    }

    model_order_raw = [
        "cipher",
        "startpop_preserving_shuffled_null",
        "starting_population_only",
    ]

    model_order_labels = [
        model_label_map[m] for m in model_order_raw
    ]

    # -----------------------------
    # Filter to day 4 all-startpops clone-level rows
    # -----------------------------
    plot_df = metrics_df.copy()

    required_cols = {"analysis", "level", "model", "fate", "AUROC", "AUPRC"}
    missing = required_cols - set(plot_df.columns)
    if missing:
        raise ValueError(f"metrics dataframe missing columns: {missing}")

    plot_df = plot_df[
        (plot_df["analysis"].astype(str) == ANALYSIS_TO_PLOT) &
        (plot_df["level"].astype(str) == "clone") &
        (plot_df["model"].astype(str).isin(model_order_raw))
    ].copy()

    if len(plot_df) == 0:
        raise ValueError(f"No clone-level rows found for analysis={ANALYSIS_TO_PLOT}")

    plot_df["model_label"] = plot_df["model"].map(model_label_map)

    # -----------------------------
    # Fate order
    # -----------------------------
    preferred_fates = ["Monocyte", "Neutrophil", "Baso", "Meg", "Mast"]
    present_fates = plot_df["fate"].dropna().astype(str).unique().tolist()

    fate_order = [f for f in preferred_fates if f in present_fates]
    extra_fates = sorted([
        f for f in present_fates
        if f not in fate_order and f != GLOBAL_FATE_NAME
    ])
    fate_order = fate_order + extra_fates

    if GLOBAL_FATE_NAME in present_fates:
        fate_order = fate_order + [GLOBAL_FATE_NAME]

    # -----------------------------
    # Only overlay real-model points, not null points
    # -----------------------------
    point_df = plot_df[
        plot_df["model"].isin(["cipher", "starting_population_only"])
    ].copy()

    # -----------------------------
    # Colors
    # -----------------------------
    palette = {
        "CIPHER": sns.color_palette("tab10")[0],
        "startpop-preserving null": sns.color_palette("tab10")[1],
        "starting-pop only": sns.color_palette("tab10")[2],
    }

    # -----------------------------
    # Helper: robust stripplot for different seaborn versions
    # -----------------------------
    def add_stripplot_no_null(ax, data, ycol):
        try:
            sns.stripplot(
                data=data,
                x="fate",
                y=ycol,
                hue="model_label",
                order=fate_order,
                hue_order=model_order_labels,
                dodge=True,
                color="black",
                alpha=0.65,
                size=4,
                jitter=0.18,
                legend=False,
                ax=ax,
            )
        except TypeError:
            sns.stripplot(
                data=data,
                x="fate",
                y=ycol,
                hue="model_label",
                order=fate_order,
                hue_order=model_order_labels,
                dodge=True,
                color="black",
                alpha=0.65,
                size=4,
                jitter=0.18,
                ax=ax,
            )
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        sharex=True,
    )

    for ax, metric_name in zip(axes, ["AUROC", "AUPRC"]):

        sns.boxplot(
            data=plot_df,
            x="fate",
            y=metric_name,
            hue="model_label",
            order=fate_order,
            hue_order=model_order_labels,
            palette=palette,
            showfliers=False,
            linewidth=1.5,
            ax=ax,
        )

        # Overlay black points only for CIPHER and starting-pop only.
        # This prevents the orange null from having hundreds of visible dots.
        add_stripplot_no_null(ax, point_df, metric_name)

        # Range from 0 to 1.1, but only label ticks up to 1.0.
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_yticklabels([f"{x:.1f}" for x in np.arange(0, 1.01, 0.2)])

        if metric_name == "AUROC":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=2)
        else:
            # Optional AUPRC baseline is not drawn because baseline differs by fate.
            pass

        ax.set_title(f"{ANALYSIS_TO_PLOT}: clone-level {metric_name}")
        ax.set_ylabel(metric_name)
        ax.set_xlabel("")

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # X-axis labels only on bottom panel.
    axes[-1].set_xlabel("future lineage")
    axes[-1].tick_params(axis="x", rotation=45)
    for tick in axes[-1].get_xticklabels():
        tick.set_ha("right")

    # Shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []

    for h, l in zip(handles, labels):
        if l not in seen and l in model_order_labels:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)

    fig.legend(
        uniq_handles,
        uniq_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 0.86, 1])

    # -----------------------------
    # Save
    # -----------------------------
    png_path = os.path.join(OUTDIR, f"{SAVE_PREFIX}.png")
    svg_path = os.path.join(OUTDIR, f"{SAVE_PREFIX}.svg")
    pdf_path = os.path.join(OUTDIR, f"{SAVE_PREFIX}.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    print("Saved:")
    print(" ", png_path)
    print(" ", svg_path)
    print(" ", pdf_path)

    # -----------------------------
    # Optional: print compact summary
    # -----------------------------
    summary = (
        plot_df
        .groupby(["fate", "model_label"], as_index=False)
        .agg(
            AUROC_mean=("AUROC", "mean"),
            AUROC_sd=("AUROC", "std"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_sd=("AUPRC", "std"),
            n_rows=("AUROC", "size"),
        )
    )

    summary = summary[
        summary["fate"].isin(fate_order)
    ].copy()

    summary["fate"] = pd.Categorical(summary["fate"], categories=fate_order, ordered=True)
    summary["model_label"] = pd.Categorical(summary["model_label"], categories=model_order_labels, ordered=True)
    summary = summary.sort_values(["fate", "model_label"])

    print("\nSummary:")
    print(summary.to_string(index=False))



def composition_prediction():
    global os, gzip, warnings, np, pd, plt, sns, Counter, \
        mmread, issparse, pearsonr, spearmanr, StratifiedKFold, PCA, confusion_matrix, OUTDIR, \
        COUNTS_PATH, GENES_PATH, CLONE_PATH, META_PATH, TIME_COL, CELLTYPE_COL, START_COL, WELL_COL, \
        EARLY_TIME, EARLY_CELLTYPE, EARLY_WELL, RESTRICT_STARTING_POPULATION, TERMINAL_TIME, TERMINAL_WELL, EXCLUDE_FATES, MANUAL_SELECTED_FATES, \
        MAX_FATES, MIN_CLONES_WITH_FATE, MIN_TERMINAL_CELLS_PER_CLONE, MIN_EARLY_CELLS_PER_CLONE, MIN_TOTAL_CELLS_PER_CLONE, MIN_SELECTED_FATE_COVERAGE, MIN_SELECTED_TERMINAL_CELLS, N_VAR_GENES, \
        MAX_COV_CELLS, RIDGE, COV_SHRINK_TO_DIAG, USE_FATE_PRIOR, N_NULLS, USE_STARTPOP_PRESERVING_NULL, N_SPLITS, SEED, \
        rng, safe_name, softmax_logits, js_div, cosine_similarity, safe_corr, safe_r2, get_cell_to_clone, \
        get_cells_x_genes, zscore_train, select_hvgs_sparse, make_covariance, weighted_mean, clone_mean_matrix, shuffle_rows_within_groups, counts, \
        f, gene_names, clone_mat, meta, cell_to_clone, has_clone, cell_fates, early_mask, \
        terminal_mask, early_all_idx, early_cloned_mask, terminal_cloned_mask, early_cloned_idx, terminal_cloned_idx, candidate_records, global_fate_counts, \
        global_fate_clone_counts, clone_id, cells, early_cells, terminal_cells, fates, vc, terminal_counts_dict, \
        c, starts, dominant_start, dominant_start_frac, candidate_table, fate_summary, selected_fates, clone_table, \
        fate, s, selected_count_cols, obs_frac_cols, Y_all, dominant_idx, clone_table_save, fig, \
        axes, obs_mean_by_dom, tab, hvg_idx, gene_vars, hvg_genes, cov_idx, Xcov_raw, \
        mu_ref, sd_ref, Xcov, Sigma, evals, evecs, make_composition_cipher_model, score_composition_cipher, \
        fit_startpop_composition_baseline, score_startpop_composition_baseline, add_prediction_columns, add_composition_errors, summarize_predictions, X_clones_all, strat_y, min_class_n, \
        n_splits, splitter, clone_to_obs, clone_to_counts, clone_to_start, all_pred_rows, all_null_rows_for_error_plots, summary_rows, \
        force_rows, fold, train_pos, test_pos, train_clones, test_clones, Xtrain, train_clone_ids_used, \
        n_train_early, Xtest, test_clone_ids_used, n_test_early, Ytrain, Ytest, Ctest, start_train, \
        start_test, true_dom_test, base, j, cipher_model, raw_scores, logits, Ptest, \
        pred_df, u, delta, direction, idxs, rank, gi, sp_model, \
        raw_sp, logits_sp, P_sp, sp_df, null_id, Ytrain_null, null_name, null_model, \
        raw_null, logits_null, P_null, null_df, predictions, null_clone_errors, summary_metrics, force_df, \
        composition_summary, per_fate_summary, p_rows, null_models, metric, real_vals, null_vals, real_mean, \
        p_emp, pvals, cipher_pred, n_fates, ncols, nrows, ax, x, \
        y, r, rho, r2, k, obs_cols, pred_cols, obs_heat, \
        pred_heat, cipher_error_compact, error_plot_df, sp_error, model_label_map, perf, point_df, handles, \
        labels, uniq_h, uniq_l, h, l, mx, cm, cm_norm, \
        mixed_df, cipher_force, mean_force, top_genes, TOP_GENES_PER_FATE, sub, heat, MAX_PLOT_CLONES, \
        plot_df, plot_clone_ids, Xplot, ids_plot_used, _, Z, color_col, label, \
        sc
    # ============================================================
    # CIPHER-LARRY: predict terminal clone fate COMPOSITION
    # ============================================================
    # Instead of collapsing each clone to a dominant fate, this uses the full
    # terminal fate fraction vector:
    #
    #   y_c(f) = terminal cells in fate f / terminal cells in selected fates
    #
    # CIPHER learns one force per fate using weighted clone means:
    #
    #   Delta_f = weighted_mean(X_early_clone, weights=y_c(f))
    #             - weighted_mean(X_early_clone, weights=1-y_c(f))
    #
    #   u_f = Sigma_early^{-1} Delta_f
    #
    # Held-out clone composition is predicted by:
    #
    #   logit_f(x_c) = u_f^T x_c - 0.5 u_f^T Sigma u_f + optional log prior_f
    #   p_hat_c(f)  = softmax_f(logit_f)
    #
    # Outputs:
    #   - predicted vs observed fate fractions
    #   - KL/JSD/Brier composition error
    #   - entropy prediction
    #   - per-fate correlations
    #   - CIPHER vs null and starting-population-only baseline
    #   - force gene heatmap
    # ============================================================

    import os
    import gzip
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from collections import Counter
    from scipy.io import mmread
    from scipy.sparse import issparse
    from scipy.stats import pearsonr, spearmanr
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix

    warnings.filterwarnings("ignore")

    # ============================================================
    # CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_clone_fate_composition")
    os.makedirs(OUTDIR, exist_ok=True)

    COUNTS_PATH = os.path.join(SUPPL, "stateFate_inVitro_normed_counts.mtx.gz")
    GENES_PATH  = os.path.join(SUPPL, "stateFate_inVitro_gene_names.txt.gz")
    CLONE_PATH  = os.path.join(SUPPL, "stateFate_inVitro_clone_matrix.mtx.gz")
    META_PATH   = os.path.join(SUPPL, "stateFate_inVitro_metadata.txt.gz")

    TIME_COL = "Time point"
    CELLTYPE_COL = "Cell type annotation"
    START_COL = "Starting population"
    WELL_COL = "Well"

    # Main analysis. Change to 2.0 for earlier/harder prediction.
    EARLY_TIME = 4.0
    EARLY_CELLTYPE = "Undifferentiated"
    EARLY_WELL = None

    # Set this to "Lin-Kit+Sca1+" or "Lin-Kit+Sca1-" for within-starting-population analysis.
    RESTRICT_STARTING_POPULATION = None

    TERMINAL_TIME = 6.0
    TERMINAL_WELL = None

    EXCLUDE_FATES = {
        "Undifferentiated", "Unknown", "unknown", "nan", "NaN",
        "Ambiguous", "ambiguous", "None", ""
    }

    # If None, choose most common terminal fates automatically.
    MANUAL_SELECTED_FATES = None
    # Example:
    # MANUAL_SELECTED_FATES = ["Monocyte", "Neutrophil", "Baso", "Meg"]

    MAX_FATES = 5
    MIN_CLONES_WITH_FATE = 8
    MIN_TERMINAL_CELLS_PER_CLONE = 5
    MIN_EARLY_CELLS_PER_CLONE = 1
    MIN_TOTAL_CELLS_PER_CLONE = 8

    # Keep clones whose selected fates explain most terminal output.
    MIN_SELECTED_FATE_COVERAGE = 0.75
    MIN_SELECTED_TERMINAL_CELLS = 5

    # Gene / covariance settings.
    N_VAR_GENES = 500
    MAX_COV_CELLS = 50000

    RIDGE = 0.50
    COV_SHRINK_TO_DIAG = 0.30

    # Whether to add log mean fate fraction as prior in the softmax.
    USE_FATE_PRIOR = False

    # Nulls.
    N_NULLS = 100
    USE_STARTPOP_PRESERVING_NULL = True

    # CV.
    N_SPLITS = 5
    SEED = 0
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    plt.rcParams.update({"font.size": 13})
    sns.set_context("talk")

    # ============================================================
    # HELPERS
    # ============================================================


    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )

    def softmax_logits(logits, eps=1e-12):
        z = logits - np.max(logits, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.maximum(ez.sum(axis=1, keepdims=True), eps)


    def js_div(P, Q, eps=1e-12):
        M = 0.5 * (P + Q)
        return 0.5 * kl_div(P, M, eps=eps) + 0.5 * kl_div(Q, M, eps=eps)

    def cosine_similarity(P, Q, eps=1e-12):
        num = np.sum(P * Q, axis=1)
        den = np.sqrt(np.sum(P * P, axis=1)) * np.sqrt(np.sum(Q * Q, axis=1))
        return num / np.maximum(den, eps)

    def safe_corr(x, y, method="pearson"):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        try:
            if method == "pearson":
                return pearsonr(x, y)[0]
            return spearmanr(x, y)[0]
        except Exception:
            return np.nan

    def safe_r2(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        if ss_tot < 1e-12:
            return np.nan
        return 1.0 - ss_res / ss_tot

    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()
        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row
        return cell_to_clone

    def get_cells_x_genes(counts, cell_idx, gene_idx):
        # counts is genes x cells
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)

    def zscore_train(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu, sd


    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        X = counts[:, cell_idx]
        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means**2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]

        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_var_genes:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_

    def make_covariance(X):
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D
        Sigma = Sigma + RIDGE * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)

    def weighted_mean(X, w, eps=1e-12):
        w = np.asarray(w, dtype=float)
        return (w[:, None] * X).sum(axis=0) / max(w.sum(), eps)

    def clone_mean_matrix(clone_ids, early_mask, cell_to_clone, counts, hvg_idx, mu, sd):
        rows = []
        out_ids = []
        out_n = []

        for cid in clone_ids:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]
            if len(idx) == 0:
                continue

            X = get_cells_x_genes(counts, idx, hvg_idx)
            X = apply_zscore(X, mu, sd)

            rows.append(X.mean(axis=0))
            out_ids.append(cid)
            out_n.append(len(idx))

        if len(rows) == 0:
            return (
                np.empty((0, len(hvg_idx))),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        return np.vstack(rows), np.asarray(out_ids, dtype=int), np.asarray(out_n, dtype=int)

    def shuffle_rows_within_groups(Y, groups):
        Y = np.asarray(Y).copy()
        groups = np.asarray(groups).astype(str)
        out = Y.copy()

        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) > 1:
                out[idx] = out[rng.permutation(idx)]

        return out

    # ============================================================
    # LOAD DATA
    # ============================================================

    counts = mmread(COUNTS_PATH).T.tocsr()
    print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} cells | nnz={counts.nnz:,}")

    with gzip.open(GENES_PATH, "rt") as f:
        gene_names = np.array([line.strip() for line in f])
    print(f"Genes loaded: {len(gene_names)}")

    clone_mat = mmread(CLONE_PATH).T.tocsr()
    print(f"Clone matrix: {clone_mat.shape[0]} clones x {clone_mat.shape[1]} cells")
    print(f"% cells with clone label: {(clone_mat.sum(axis=0) > 0).mean() * 100:.2f}%")

    meta = pd.read_csv(META_PATH, sep="\t")
    meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")

    print(f"Meta: {meta.shape[0]} rows x {meta.shape[1]} cols")
    print("Meta columns:", list(meta.columns))

    assert counts.shape[1] == meta.shape[0] == clone_mat.shape[1], "cells mismatch"
    assert counts.shape[0] == len(gene_names), "genes mismatch"

    print("\nTimepoints:")
    print(np.sort(meta[TIME_COL].dropna().unique()))

    print("\nCell annotations:")
    print(meta[CELLTYPE_COL].value_counts())

    cell_to_clone = get_cell_to_clone(clone_mat)
    has_clone = cell_to_clone >= 0
    cell_fates = meta[CELLTYPE_COL].astype(str).values

    # ============================================================
    # DEFINE EARLY / TERMINAL MASKS
    # ============================================================

    early_mask = meta[TIME_COL].astype(float).values == float(EARLY_TIME)

    if EARLY_CELLTYPE is not None:
        early_mask &= meta[CELLTYPE_COL].astype(str).values == str(EARLY_CELLTYPE)

    if EARLY_WELL is not None and WELL_COL in meta.columns:
        early_mask &= meta[WELL_COL].astype(float).values == float(EARLY_WELL)

    if RESTRICT_STARTING_POPULATION is not None and START_COL in meta.columns:
        early_mask &= meta[START_COL].astype(str).values == str(RESTRICT_STARTING_POPULATION)

    terminal_mask = meta[TIME_COL].astype(float).values == float(TERMINAL_TIME)

    if TERMINAL_WELL is not None and WELL_COL in meta.columns:
        terminal_mask &= meta[WELL_COL].astype(float).values == float(TERMINAL_WELL)

    terminal_mask &= ~np.isin(cell_fates, list(EXCLUDE_FATES))

    early_all_idx = np.where(early_mask)[0]
    early_cloned_mask = early_mask & has_clone
    terminal_cloned_mask = terminal_mask & has_clone

    early_cloned_idx = np.where(early_cloned_mask)[0]
    terminal_cloned_idx = np.where(terminal_cloned_mask)[0]

    print(f"\nAll early cells for Sigma: {len(early_all_idx):,}")
    print(f"Cloned early cells: {len(early_cloned_idx):,}")
    print(f"Cloned terminal cells: {len(terminal_cloned_idx):,}")

    if len(early_all_idx) == 0:
        raise RuntimeError("No early cells found. Check EARLY_TIME / EARLY_CELLTYPE / EARLY_WELL / RESTRICT_STARTING_POPULATION.")

    if len(terminal_cloned_idx) == 0:
        raise RuntimeError("No terminal cloned cells found. Check TERMINAL_TIME / TERMINAL_WELL.")

    # ============================================================
    # BUILD CLONE TABLE WITH TERMINAL COMPOSITION
    # ============================================================

    candidate_records = []
    global_fate_counts = Counter()
    global_fate_clone_counts = Counter()

    for clone_id in range(clone_mat.shape[0]):
        cells = clone_mat[clone_id].indices

        if len(cells) < MIN_TOTAL_CELLS_PER_CLONE:
            continue

        early_cells = cells[early_cloned_mask[cells]]
        terminal_cells = cells[terminal_cloned_mask[cells]]

        if len(early_cells) < MIN_EARLY_CELLS_PER_CLONE:
            continue
        if len(terminal_cells) < MIN_TERMINAL_CELLS_PER_CLONE:
            continue

        fates = pd.Series(cell_fates[terminal_cells].astype(str))
        fates = fates[~fates.isin(EXCLUDE_FATES)]

        if len(fates) < MIN_TERMINAL_CELLS_PER_CLONE:
            continue

        vc = fates.value_counts()
        terminal_counts_dict = {str(k): int(v) for k, v in vc.items()}

        for f, c in terminal_counts_dict.items():
            global_fate_counts[f] += c
            if c > 0:
                global_fate_clone_counts[f] += 1

        if START_COL in meta.columns:
            starts = meta.iloc[early_cells][START_COL].astype(str).value_counts()
            dominant_start = starts.index[0]
            dominant_start_frac = float(starts.iloc[0] / starts.sum())
        else:
            dominant_start = "unknown"
            dominant_start_frac = 1.0

        candidate_records.append({
            "clone_id": int(clone_id),
            "n_total_clone_cells": int(len(cells)),
            "n_early": int(len(early_cells)),
            "n_terminal": int(len(fates)),
            "terminal_counts_dict": terminal_counts_dict,
            "dominant_starting_population": dominant_start,
            "dominant_starting_population_frac": dominant_start_frac,
        })

    candidate_table = pd.DataFrame(candidate_records)

    if candidate_table.empty:
        raise RuntimeError("No clones passed initial early/terminal filters.")

    fate_summary = pd.DataFrame({
        "fate": list(global_fate_counts.keys()),
        "terminal_cell_count": [global_fate_counts[f] for f in global_fate_counts.keys()],
        "clone_count_with_fate": [global_fate_clone_counts[f] for f in global_fate_counts.keys()],
    }).sort_values("terminal_cell_count", ascending=False)

    fate_summary.to_csv(os.path.join(OUTDIR, "terminal_fate_summary_before_selection.csv"), index=False)

    if MANUAL_SELECTED_FATES is None:
        selected_fates = (
            fate_summary[fate_summary["clone_count_with_fate"] >= MIN_CLONES_WITH_FATE]
            .head(MAX_FATES)["fate"]
            .tolist()
        )
    else:
        selected_fates = list(MANUAL_SELECTED_FATES)

    if len(selected_fates) < 2:
        raise RuntimeError("Fewer than two selected fates. Lower MIN_CLONES_WITH_FATE or use MANUAL_SELECTED_FATES.")

    print("\nSelected fates for composition:")
    print(selected_fates)

    clone_table = candidate_table.copy()

    for fate in selected_fates:
        s = safe_name(fate)
        clone_table[f"terminal_count__{s}"] = clone_table["terminal_counts_dict"].apply(lambda d: int(d.get(fate, 0)))

    selected_count_cols = [f"terminal_count__{safe_name(f)}" for f in selected_fates]
    clone_table["n_terminal_selected"] = clone_table[selected_count_cols].sum(axis=1)
    clone_table["selected_fate_coverage"] = clone_table["n_terminal_selected"] / clone_table["n_terminal"]

    clone_table = clone_table[
        (clone_table["n_terminal_selected"] >= MIN_SELECTED_TERMINAL_CELLS) &
        (clone_table["selected_fate_coverage"] >= MIN_SELECTED_FATE_COVERAGE)
    ].copy()

    if clone_table.empty:
        raise RuntimeError("No clones passed selected fate coverage filtering. Lower MIN_SELECTED_FATE_COVERAGE.")

    for fate in selected_fates:
        s = safe_name(fate)
        clone_table[f"obs_frac__{s}"] = clone_table[f"terminal_count__{s}"] / clone_table["n_terminal_selected"]

    obs_frac_cols = [f"obs_frac__{safe_name(f)}" for f in selected_fates]

    Y_all = clone_table[obs_frac_cols].values.astype(float)
    dominant_idx = np.argmax(Y_all, axis=1)
    clone_table["dominant_selected_fate"] = np.array(selected_fates, dtype=object)[dominant_idx]
    clone_table["terminal_entropy_selected"] = entropy(Y_all)

    clone_table = clone_table.reset_index(drop=True)

    print("\nClone table after composition filters:")
    print(f"n clones: {len(clone_table):,}")
    print("Dominant selected fate counts:")
    print(clone_table["dominant_selected_fate"].value_counts())
    print("\nMean selected fate coverage:", clone_table["selected_fate_coverage"].mean())

    clone_table_save = clone_table.drop(columns=["terminal_counts_dict"])
    clone_table_save.to_csv(os.path.join(OUTDIR, "clone_terminal_composition_table.csv"), index=False)

    # ============================================================
    # CLONE QC PLOTS
    # ============================================================

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    sns.countplot(
        data=clone_table,
        x="dominant_selected_fate",
        order=clone_table["dominant_selected_fate"].value_counts().index,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Selected clones by dominant terminal fate")
    axes[0, 0].set_xlabel("dominant terminal fate")
    axes[0, 0].set_ylabel("clone count")
    axes[0, 0].tick_params(axis="x", rotation=45)

    sns.histplot(data=clone_table, x="n_total_clone_cells", bins=40, ax=axes[0, 1])
    axes[0, 1].set_title("Total cells per retained clone")

    sns.histplot(data=clone_table, x="n_early", bins=30, ax=axes[0, 2])
    axes[0, 2].set_title("Early cells per retained clone")

    sns.histplot(data=clone_table, x="n_terminal_selected", bins=40, ax=axes[1, 0])
    axes[1, 0].set_title("Selected terminal cells per clone")

    sns.histplot(data=clone_table, x="terminal_entropy_selected", bins=30, ax=axes[1, 1])
    axes[1, 1].set_title("Observed terminal composition entropy")

    sns.scatterplot(
        data=clone_table,
        x="n_early",
        y="n_terminal_selected",
        hue="dominant_selected_fate",
        ax=axes[1, 2],
        s=45,
    )
    axes[1, 2].set_title("Early vs terminal representation")
    axes[1, 2].legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "clone_composition_qc_summary.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "clone_composition_qc_summary.svg"), bbox_inches="tight")
    plt.show()

    # Observed terminal composition by dominant fate.
    obs_mean_by_dom = clone_table.groupby("dominant_selected_fate")[obs_frac_cols].mean().reindex(selected_fates)
    obs_mean_by_dom.columns = selected_fates

    plt.figure(figsize=(1.2 * len(selected_fates) + 5, 1.0 * len(selected_fates) + 3))
    sns.heatmap(
        obs_mean_by_dom,
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "mean observed terminal fraction"},
    )
    plt.title("Observed terminal clone composition")
    plt.xlabel("terminal fate fraction")
    plt.ylabel("dominant terminal fate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "observed_terminal_composition_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "observed_terminal_composition_heatmap.svg"), bbox_inches="tight")
    plt.show()

    if START_COL in meta.columns:
        plt.figure(figsize=(10, 5))
        tab = pd.crosstab(
            clone_table["dominant_selected_fate"],
            clone_table["dominant_starting_population"],
        ).reindex(selected_fates)
        sns.heatmap(tab, annot=True, fmt="d", cmap="viridis")
        plt.title("Dominant terminal fate vs early starting population")
        plt.xlabel("dominant early starting population")
        plt.ylabel("dominant terminal fate")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "dominant_fate_vs_starting_population.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "dominant_fate_vs_starting_population.svg"), bbox_inches="tight")
        plt.show()

    # ============================================================
    # HVGs + GLOBAL EARLY COVARIANCE
    # ============================================================

    print("\nSelecting HVGs from early cells...")

    hvg_idx, gene_vars = select_hvgs_sparse(
        counts=counts,
        cell_idx=early_all_idx,
        n_var_genes=N_VAR_GENES,
    )
    hvg_genes = gene_names[hvg_idx]

    pd.DataFrame({
        "gene": hvg_genes,
        "gene_index": hvg_idx,
        "early_variance": gene_vars[hvg_idx],
    }).to_csv(os.path.join(OUTDIR, "selected_early_hvgs.csv"), index=False)

    cov_idx = early_all_idx.copy()
    if len(cov_idx) > MAX_COV_CELLS:
        cov_idx = rng.choice(cov_idx, size=MAX_COV_CELLS, replace=False)

    print(f"Using {len(cov_idx):,} cells for Sigma.")

    Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx)
    mu_ref, sd_ref = zscore_train(Xcov_raw)
    Xcov = apply_zscore(Xcov_raw, mu_ref, sd_ref)

    Sigma = make_covariance(Xcov)

    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-8)

    pd.DataFrame({
        "rank": np.arange(1, len(evals) + 1),
        "eigenvalue": evals[::-1],
    }).to_csv(os.path.join(OUTDIR, "early_covariance_eigenvalues.csv"), index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(evals) + 1), evals[::-1], marker="o", linewidth=1, markersize=3)
    plt.yscale("log")
    plt.xlabel("eigenvalue rank")
    plt.ylabel("eigenvalue")
    plt.title("Early progenitor covariance spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # MODEL FUNCTIONS
    # ============================================================

    def make_composition_cipher_model(Xtrain_clone, Ytrain, selected_fates, evals, evecs, Sigma, use_fate_prior=False):
        U = []
        DELTAS = []
        penalties = []
        log_priors = []

        eps = 1e-12
        Ytrain = np.asarray(Ytrain, dtype=float)

        for j, fate in enumerate(selected_fates):
            w_pos = Ytrain[:, j].copy()
            w_neg = 1.0 - w_pos

            if w_pos.sum() < eps or w_neg.sum() < eps:
                delta = np.zeros(Xtrain_clone.shape[1])
            else:
                mu_pos = weighted_mean(Xtrain_clone, w_pos)
                mu_neg = weighted_mean(Xtrain_clone, w_neg)
                delta = mu_pos - mu_neg

            u = evecs @ ((evecs.T @ delta) / evals)
            penalty = 0.5 * float(u @ Sigma @ u)

            if use_fate_prior:
                prior = max(float(Ytrain[:, j].mean()), eps)
                log_prior = np.log(prior)
            else:
                log_prior = 0.0

            U.append(u)
            DELTAS.append(delta)
            penalties.append(penalty)
            log_priors.append(log_prior)

        return {
            "U": np.asarray(U),
            "DELTAS": np.asarray(DELTAS),
            "penalty": np.asarray(penalties),
            "log_prior": np.asarray(log_priors),
        }

    def score_composition_cipher(X, model):
        U = model["U"]
        raw_scores = X @ U.T
        logits = raw_scores - model["penalty"][None, :] + model["log_prior"][None, :]
        P = softmax_logits(logits)
        return raw_scores, logits, P

    def fit_startpop_composition_baseline(Ytrain, start_train, alpha=2.0):
        Ytrain = np.asarray(Ytrain, dtype=float)
        start_train = np.asarray(start_train).astype(str)

        global_p = Ytrain.mean(axis=0)
        global_p = global_p / global_p.sum()

        table = {}
        for s in np.unique(start_train):
            idx = np.where(start_train == s)[0]
            if len(idx) == 0:
                continue
            p = (Ytrain[idx].sum(axis=0) + alpha * global_p) / (len(idx) + alpha)
            p = p / p.sum()
            table[s] = p

        return {
            "global_p": global_p,
            "table": table,
        }

    def score_startpop_composition_baseline(start_test, model):
        start_test = np.asarray(start_test).astype(str)
        P = []
        for s in start_test:
            P.append(model["table"].get(s, model["global_p"]))
        P = np.vstack(P)
        logits = np.log(np.clip(P, 1e-12, 1.0))
        raw_scores = logits.copy()
        return raw_scores, logits, P

    def add_prediction_columns(base_df, raw_scores, logits, P, selected_fates, model_name):
        rows = base_df.copy()
        rows["model"] = model_name

        pred_idx = np.argmax(P, axis=1)
        rows["pred_dominant_fate"] = np.array(selected_fates, dtype=object)[pred_idx]
        rows["pred_entropy"] = entropy(P)
        rows["pred_max_prob"] = P.max(axis=1)

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            rows[f"score_raw__{s}"] = raw_scores[:, j]
            rows[f"logit__{s}"] = logits[:, j]
            rows[f"pred_frac__{s}"] = P[:, j]

        return rows

    def add_composition_errors(df, selected_fates):
        obs = df[[f"obs_frac__{safe_name(f)}" for f in selected_fates]].values.astype(float)
        pred = df[[f"pred_frac__{safe_name(f)}" for f in selected_fates]].values.astype(float)

        df = df.copy()
        df["composition_KL_obs_pred"] = kl_div(obs, pred)
        df["composition_JS"] = js_div(obs, pred)
        df["composition_Brier"] = np.mean((obs - pred) ** 2, axis=1)
        df["composition_L1"] = np.sum(np.abs(obs - pred), axis=1)
        df["composition_cosine"] = cosine_similarity(obs, pred)
        df["obs_entropy"] = entropy(obs)
        df["pred_entropy"] = entropy(pred)
        df["true_dominant_fate"] = np.array(selected_fates, dtype=object)[np.argmax(obs, axis=1)]
        df["pred_dominant_fate"] = np.array(selected_fates, dtype=object)[np.argmax(pred, axis=1)]
        df["dominant_fate_correct"] = df["true_dominant_fate"].values == df["pred_dominant_fate"].values

        return df

    def summarize_predictions(df, selected_fates, model_name, fold, null_id=None):
        rows = []

        obs = df[[f"obs_frac__{safe_name(f)}" for f in selected_fates]].values.astype(float)
        pred = df[[f"pred_frac__{safe_name(f)}" for f in selected_fates]].values.astype(float)

        rows.append({
            "model": model_name,
            "fold": fold,
            "null_id": null_id,
            "metric_type": "composition",
            "fate": "ALL",
            "mean_KL": np.mean(kl_div(obs, pred)),
            "mean_JS": np.mean(js_div(obs, pred)),
            "mean_Brier": np.mean(np.mean((obs - pred) ** 2, axis=1)),
            "mean_L1": np.mean(np.sum(np.abs(obs - pred), axis=1)),
            "mean_cosine": np.mean(cosine_similarity(obs, pred)),
            "top1_accuracy": np.mean(np.argmax(obs, axis=1) == np.argmax(pred, axis=1)),
            "entropy_pearson": safe_corr(entropy(obs), entropy(pred), method="pearson"),
            "entropy_spearman": safe_corr(entropy(obs), entropy(pred), method="spearman"),
            "n_clones": len(df),
        })

        for j, fate in enumerate(selected_fates):
            y = obs[:, j]
            p = pred[:, j]

            rows.append({
                "model": model_name,
                "fold": fold,
                "null_id": null_id,
                "metric_type": "per_fate_fraction",
                "fate": fate,
                "pearson": safe_corr(y, p, method="pearson"),
                "spearman": safe_corr(y, p, method="spearman"),
                "r2": safe_r2(y, p),
                "mae": np.mean(np.abs(y - p)),
                "rmse": np.sqrt(np.mean((y - p) ** 2)),
                "mean_obs_fraction": np.mean(y),
                "mean_pred_fraction": np.mean(p),
                "n_clones": len(df),
            })

        return pd.DataFrame(rows)

    # ============================================================
    # CROSS-VALIDATED COMPOSITION PREDICTION
    # ============================================================

    X_clones_all = clone_table["clone_id"].values.astype(int)
    strat_y = clone_table["dominant_selected_fate"].values.astype(str)

    min_class_n = clone_table["dominant_selected_fate"].value_counts().min()
    n_splits = int(min(N_SPLITS, min_class_n))

    if n_splits < 2:
        raise RuntimeError(f"Cannot do CV. Smallest dominant fate has only {min_class_n} clones.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    clone_to_obs = {
        int(row["clone_id"]): row[obs_frac_cols].values.astype(float)
        for _, row in clone_table.iterrows()
    }

    clone_to_counts = {
        int(row["clone_id"]): row[selected_count_cols].values.astype(int)
        for _, row in clone_table.iterrows()
    }

    clone_to_start = dict(zip(clone_table["clone_id"].astype(int), clone_table["dominant_starting_population"].astype(str)))

    all_pred_rows = []
    all_null_rows_for_error_plots = []
    summary_rows = []
    force_rows = []

    for fold, (train_pos, test_pos) in enumerate(splitter.split(X_clones_all, strat_y)):
        train_clones = X_clones_all[train_pos]
        test_clones = X_clones_all[test_pos]

        print(f"\nFold {fold + 1}/{n_splits}: train={len(train_clones)}, test={len(test_clones)}")

        Xtrain, train_clone_ids_used, n_train_early = clone_mean_matrix(
            clone_ids=train_clones,
            early_mask=early_cloned_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        Xtest, test_clone_ids_used, n_test_early = clone_mean_matrix(
            clone_ids=test_clones,
            early_mask=early_cloned_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        Ytrain = np.vstack([clone_to_obs[int(c)] for c in train_clone_ids_used])
        Ytest = np.vstack([clone_to_obs[int(c)] for c in test_clone_ids_used])
        Ctest = np.vstack([clone_to_counts[int(c)] for c in test_clone_ids_used])

        start_train = np.array([clone_to_start.get(int(c), "unknown") for c in train_clone_ids_used])
        start_test = np.array([clone_to_start.get(int(c), "unknown") for c in test_clone_ids_used])

        true_dom_test = np.array(selected_fates, dtype=object)[np.argmax(Ytest, axis=1)]

        base = pd.DataFrame({
            "fold": fold,
            "clone_id": test_clone_ids_used,
            "n_early_scored": n_test_early,
            "dominant_starting_population": start_test,
            "true_dominant_fate": true_dom_test,
        })

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            base[f"obs_frac__{s}"] = Ytest[:, j]
            base[f"terminal_count__{s}"] = Ctest[:, j]

        base["obs_entropy"] = entropy(Ytest)
        base["n_terminal_selected"] = Ctest.sum(axis=1)

        # --------------------------
        # CIPHER composition model.
        # --------------------------
        cipher_model = make_composition_cipher_model(
            Xtrain_clone=Xtrain,
            Ytrain=Ytrain,
            selected_fates=selected_fates,
            evals=evals,
            evecs=evecs,
            Sigma=Sigma,
            use_fate_prior=USE_FATE_PRIOR,
        )

        raw_scores, logits, Ptest = score_composition_cipher(Xtest, cipher_model)
        pred_df = add_prediction_columns(base, raw_scores, logits, Ptest, selected_fates, "cipher")
        pred_df = add_composition_errors(pred_df, selected_fates)

        all_pred_rows.append(pred_df)
        summary_rows.append(summarize_predictions(pred_df, selected_fates, "cipher", fold))

        # Save force genes.
        for j, fate in enumerate(selected_fates):
            u = cipher_model["U"][j]
            delta = cipher_model["DELTAS"][j]

            for direction, idxs in [
                ("positive", np.argsort(u)[::-1][:50]),
                ("negative", np.argsort(u)[:50]),
            ]:
                for rank, gi in enumerate(idxs, start=1):
                    force_rows.append({
                        "fold": fold,
                        "model": "cipher",
                        "fate": fate,
                        "direction": direction,
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),
                        "u": float(u[gi]),
                        "delta_weighted_composition": float(delta[gi]),
                        "penalty": float(cipher_model["penalty"][j]),
                        "log_prior": float(cipher_model["log_prior"][j]),
                    })

        # --------------------------
        # Starting-population-only composition baseline.
        # --------------------------
        if START_COL in meta.columns and RESTRICT_STARTING_POPULATION is None:
            sp_model = fit_startpop_composition_baseline(Ytrain, start_train)
            raw_sp, logits_sp, P_sp = score_startpop_composition_baseline(start_test, sp_model)

            sp_df = add_prediction_columns(base, raw_sp, logits_sp, P_sp, selected_fates, "starting_population_only")
            sp_df = add_composition_errors(sp_df, selected_fates)

            all_pred_rows.append(sp_df)
            summary_rows.append(summarize_predictions(sp_df, selected_fates, "starting_population_only", fold))

        # --------------------------
        # Nulls: shuffle terminal composition vectors.
        # --------------------------
        for null_id in range(N_NULLS):
            if USE_STARTPOP_PRESERVING_NULL and START_COL in meta.columns and RESTRICT_STARTING_POPULATION is None:
                Ytrain_null = shuffle_rows_within_groups(Ytrain, start_train)
                null_name = "startpop_preserving_null"
            else:
                Ytrain_null = Ytrain[rng.permutation(Ytrain.shape[0])]
                null_name = "shuffled_null"

            null_model = make_composition_cipher_model(
                Xtrain_clone=Xtrain,
                Ytrain=Ytrain_null,
                selected_fates=selected_fates,
                evals=evals,
                evecs=evecs,
                Sigma=Sigma,
                use_fate_prior=USE_FATE_PRIOR,
            )

            raw_null, logits_null, P_null = score_composition_cipher(Xtest, null_model)

            null_df = add_prediction_columns(base, raw_null, logits_null, P_null, selected_fates, null_name)
            null_df = add_composition_errors(null_df, selected_fates)
            null_df["null_id"] = null_id

            # Store compact null clone errors for boxplots.
            all_null_rows_for_error_plots.append(
                null_df[[
                    "fold", "null_id", "model", "clone_id",
                    "composition_KL_obs_pred", "composition_JS",
                    "composition_Brier", "composition_L1", "composition_cosine",
                    "dominant_fate_correct", "obs_entropy", "pred_entropy",
                ]].copy()
            )

            summary_rows.append(summarize_predictions(null_df, selected_fates, null_name, fold, null_id=null_id))

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    null_clone_errors = pd.concat(all_null_rows_for_error_plots, ignore_index=True)
    summary_metrics = pd.concat(summary_rows, ignore_index=True)
    force_df = pd.DataFrame(force_rows)

    predictions.to_csv(os.path.join(OUTDIR, "clone_composition_predictions.csv"), index=False)
    null_clone_errors.to_csv(os.path.join(OUTDIR, "null_clone_composition_errors.csv"), index=False)
    summary_metrics.to_csv(os.path.join(OUTDIR, "composition_prediction_summary_metrics.csv"), index=False)
    force_df.to_csv(os.path.join(OUTDIR, "composition_CIPHER_force_genes.csv"), index=False)

    print("\nSaved:")
    print(os.path.join(OUTDIR, "clone_composition_predictions.csv"))
    print(os.path.join(OUTDIR, "composition_prediction_summary_metrics.csv"))
    print(os.path.join(OUTDIR, "composition_CIPHER_force_genes.csv"))

    # ============================================================
    # FINAL SUMMARIES
    # ============================================================

    composition_summary = (
        summary_metrics[summary_metrics["metric_type"] == "composition"]
        .groupby("model", as_index=False)
        .agg(
            mean_KL=("mean_KL", "mean"),
            mean_JS=("mean_JS", "mean"),
            mean_Brier=("mean_Brier", "mean"),
            mean_L1=("mean_L1", "mean"),
            mean_cosine=("mean_cosine", "mean"),
            top1_accuracy=("top1_accuracy", "mean"),
            entropy_pearson=("entropy_pearson", "mean"),
            entropy_spearman=("entropy_spearman", "mean"),
        )
    )

    per_fate_summary = (
        summary_metrics[summary_metrics["metric_type"] == "per_fate_fraction"]
        .groupby(["model", "fate"], as_index=False)
        .agg(
            pearson_mean=("pearson", "mean"),
            pearson_sd=("pearson", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_sd=("spearman", "std"),
            r2_mean=("r2", "mean"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            mean_obs_fraction=("mean_obs_fraction", "mean"),
            mean_pred_fraction=("mean_pred_fraction", "mean"),
        )
    )

    composition_summary.to_csv(os.path.join(OUTDIR, "composition_summary_by_model.csv"), index=False)
    per_fate_summary.to_csv(os.path.join(OUTDIR, "per_fate_fraction_summary_by_model.csv"), index=False)

    print("\nComposition summary:")
    print(composition_summary)

    print("\nPer-fate CIPHER fraction prediction summary:")
    print(per_fate_summary[per_fate_summary["model"] == "cipher"])

    # Empirical p-values vs null for composition metrics.
    p_rows = []

    null_models = [m for m in summary_metrics["model"].unique() if "null" in m]

    for null_model in null_models:
        for metric in ["mean_KL", "mean_JS", "mean_Brier", "mean_L1"]:
            real_vals = summary_metrics[
                (summary_metrics["model"] == "cipher") &
                (summary_metrics["metric_type"] == "composition")
            ][metric].dropna().values

            null_vals = summary_metrics[
                (summary_metrics["model"] == null_model) &
                (summary_metrics["metric_type"] == "composition")
            ][metric].dropna().values

            if len(real_vals) > 0 and len(null_vals) > 0:
                real_mean = np.mean(real_vals)
                # Lower is better for these error metrics.
                p_emp = (1 + np.sum(null_vals <= real_mean)) / (1 + len(null_vals))
            else:
                real_mean = np.nan
                p_emp = np.nan

            p_rows.append({
                "null_model": null_model,
                "metric": metric,
                "cipher_mean": real_mean,
                "null_mean": np.mean(null_vals) if len(null_vals) else np.nan,
                "empirical_p_lower_is_better": p_emp,
                "n_null": len(null_vals),
            })

        for metric in ["mean_cosine", "top1_accuracy", "entropy_pearson", "entropy_spearman"]:
            real_vals = summary_metrics[
                (summary_metrics["model"] == "cipher") &
                (summary_metrics["metric_type"] == "composition")
            ][metric].dropna().values

            null_vals = summary_metrics[
                (summary_metrics["model"] == null_model) &
                (summary_metrics["metric_type"] == "composition")
            ][metric].dropna().values

            if len(real_vals) > 0 and len(null_vals) > 0:
                real_mean = np.mean(real_vals)
                # Higher is better for these metrics.
                p_emp = (1 + np.sum(null_vals >= real_mean)) / (1 + len(null_vals))
            else:
                real_mean = np.nan
                p_emp = np.nan

            p_rows.append({
                "null_model": null_model,
                "metric": metric,
                "cipher_mean": real_mean,
                "null_mean": np.mean(null_vals) if len(null_vals) else np.nan,
                "empirical_p_higher_is_better": p_emp,
                "n_null": len(null_vals),
            })

    pvals = pd.DataFrame(p_rows)
    pvals.to_csv(os.path.join(OUTDIR, "composition_empirical_pvalues_vs_null.csv"), index=False)

    print("\nEmpirical p-values vs null:")
    print(pvals)

    # ============================================================
    # PLOTS
    # ============================================================

    cipher_pred = predictions[predictions["model"] == "cipher"].copy()

    # --------------------------
    # 1. Predicted vs observed fate fractions.
    # --------------------------
    n_fates = len(selected_fates)
    ncols = min(3, n_fates)
    nrows = int(np.ceil(n_fates / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for j, fate in enumerate(selected_fates):
        ax = axes[j // ncols][j % ncols]
        s = safe_name(fate)

        x = cipher_pred[f"obs_frac__{s}"].values
        y = cipher_pred[f"pred_frac__{s}"].values

        ax.scatter(x, y, s=35, alpha=0.75)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)

        r = safe_corr(x, y, "pearson")
        rho = safe_corr(x, y, "spearman")
        r2 = safe_r2(x, y)

        ax.set_title(f"{fate}\nPearson={r:.2f}, Spearman={rho:.2f}, R²={r2:.2f}")
        ax.set_xlabel("observed terminal fraction")
        ax.set_ylabel("predicted CIPHER fraction")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)

    for k in range(n_fates, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "predicted_vs_observed_fate_fractions.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "predicted_vs_observed_fate_fractions.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # 2. Observed vs predicted composition heatmaps by dominant fate.
    # --------------------------
    obs_cols = [f"obs_frac__{safe_name(f)}" for f in selected_fates]
    pred_cols = [f"pred_frac__{safe_name(f)}" for f in selected_fates]

    obs_heat = cipher_pred.groupby("true_dominant_fate")[obs_cols].mean().reindex(selected_fates)
    pred_heat = cipher_pred.groupby("true_dominant_fate")[pred_cols].mean().reindex(selected_fates)

    obs_heat.columns = selected_fates
    pred_heat.columns = selected_fates

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        obs_heat,
        ax=axes[0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "observed fraction"},
    )
    axes[0].set_title("Observed terminal composition")
    axes[0].set_xlabel("terminal fate")
    axes[0].set_ylabel("dominant terminal fate")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        pred_heat,
        ax=axes[1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "predicted fraction"},
    )
    axes[1].set_title("Predicted CIPHER composition")
    axes[1].set_xlabel("predicted terminal fate")
    axes[1].set_ylabel("dominant terminal fate")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_composition_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_composition_heatmaps.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # 3. Composition error distributions.
    # --------------------------
    cipher_error_compact = cipher_pred[[
        "model", "fold", "clone_id",
        "composition_KL_obs_pred", "composition_JS",
        "composition_Brier", "composition_L1",
        "composition_cosine", "dominant_fate_correct",
        "obs_entropy", "pred_entropy",
    ]].copy()

    error_plot_df = pd.concat([cipher_error_compact, null_clone_errors], ignore_index=True)

    if "starting_population_only" in predictions["model"].unique():
        sp_error = predictions[predictions["model"] == "starting_population_only"][[
            "model", "fold", "clone_id",
            "composition_KL_obs_pred", "composition_JS",
            "composition_Brier", "composition_L1",
            "composition_cosine", "dominant_fate_correct",
            "obs_entropy", "pred_entropy",
        ]].copy()
        error_plot_df = pd.concat([error_plot_df, sp_error], ignore_index=True)

    model_label_map = {
        "cipher": "CIPHER",
        "shuffled_null": "shuffled null",
        "startpop_preserving_null": "startpop-preserving null",
        "starting_population_only": "starting-pop only",
    }
    error_plot_df["model_label"] = error_plot_df["model"].map(model_label_map).fillna(error_plot_df["model"])

    for metric in ["composition_JS", "composition_Brier", "composition_L1", "composition_cosine"]:
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=error_plot_df,
            x="model_label",
            y=metric,
            showfliers=False,
        )
        sns.stripplot(
            data=error_plot_df[error_plot_df["model"] == "cipher"],
            x="model_label",
            y=metric,
            color="black",
            alpha=0.35,
            size=3,
        )
        plt.title(f"Clone composition prediction: {metric}")
        plt.xlabel("")
        plt.ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"composition_error_{metric}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"composition_error_{metric}.svg"), bbox_inches="tight")
        plt.show()

    # --------------------------
    # 4. Per-fate correlation vs null/baseline.
    # --------------------------
    perf = summary_metrics[summary_metrics["metric_type"] == "per_fate_fraction"].copy()
    perf["model_label"] = perf["model"].map(model_label_map).fillna(perf["model"])

    for metric in ["pearson", "spearman", "r2"]:
        plt.figure(figsize=(11, 5))
        sns.boxplot(
            data=perf,
            x="fate",
            y=metric,
            hue="model_label",
            order=selected_fates,
            showfliers=False,
        )

        point_df = perf[perf["model"].isin(["cipher", "starting_population_only"])]
        if len(point_df) > 0:
            sns.stripplot(
                data=point_df,
                x="fate",
                y=metric,
                hue="model_label",
                order=selected_fates,
                dodge=True,
                color="black",
                alpha=0.6,
                size=4,
                legend=False,
            )

        plt.axhline(0, color="gray", linestyle="--", linewidth=1.5)
        if metric != "r2":
            plt.ylim(-1, 1)

        plt.title(f"Predicted vs observed terminal fate fraction: {metric}")
        plt.xlabel("terminal fate")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha="right")

        handles, labels = plt.gca().get_legend_handles_labels()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l not in uniq_l:
                uniq_h.append(h)
                uniq_l.append(l)
        plt.legend(uniq_h, uniq_l, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"per_fate_fraction_{metric}_vs_null.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"per_fate_fraction_{metric}_vs_null.svg"), bbox_inches="tight")
        plt.show()

    # --------------------------
    # 5. Observed vs predicted entropy.
    # --------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(
        cipher_pred["obs_entropy"],
        cipher_pred["pred_entropy"],
        s=40,
        alpha=0.75,
    )
    mx = max(cipher_pred["obs_entropy"].max(), cipher_pred["pred_entropy"].max())
    plt.plot([0, mx], [0, mx], linestyle="--", color="gray", linewidth=2)
    r = safe_corr(cipher_pred["obs_entropy"], cipher_pred["pred_entropy"], "pearson")
    rho = safe_corr(cipher_pred["obs_entropy"], cipher_pred["pred_entropy"], "spearman")
    plt.title(f"Clone fate entropy\nPearson={r:.2f}, Spearman={rho:.2f}")
    plt.xlabel("observed terminal fate entropy")
    plt.ylabel("predicted CIPHER entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_fate_entropy.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_fate_entropy.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # 6. Confusion matrix for dominant fate as a sanity check.
    # --------------------------
    cm = confusion_matrix(
        cipher_pred["true_dominant_fate"],
        cipher_pred["pred_dominant_fate"],
        labels=selected_fates,
    )
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        pd.DataFrame(cm_norm, index=selected_fates, columns=selected_fates),
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "row-normalized fraction"},
    )
    plt.title("Dominant fate prediction from composition softmax")
    plt.xlabel("predicted dominant fate")
    plt.ylabel("observed dominant fate")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "dominant_fate_confusion_from_composition.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "dominant_fate_confusion_from_composition.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # 7. Mixed-clone focus plot.
    # --------------------------
    mixed_df = cipher_pred[cipher_pred["obs_entropy"] > np.quantile(cipher_pred["obs_entropy"], 0.50)].copy()

    if len(mixed_df) >= 10:
        plt.figure(figsize=(6, 5))
        plt.scatter(
            mixed_df["composition_JS"],
            mixed_df["obs_entropy"],
            c=mixed_df["composition_cosine"],
            s=45,
            alpha=0.8,
            cmap="viridis",
        )
        plt.colorbar(label="composition cosine similarity")
        plt.xlabel("JS divergence, observed vs predicted")
        plt.ylabel("observed terminal fate entropy")
        plt.title("Prediction quality for mixed-fate clones")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "mixed_clone_prediction_quality.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "mixed_clone_prediction_quality.svg"), bbox_inches="tight")
        plt.show()

    # --------------------------
    # 8. Force gene heatmap.
    # --------------------------
    cipher_force = force_df[
        (force_df["model"] == "cipher") &
        (force_df["direction"] == "positive")
    ].copy()

    mean_force = (
        cipher_force
        .groupby(["fate", "gene"], as_index=False)
        .agg(
            mean_u=("u", "mean"),
            mean_delta=("delta_weighted_composition", "mean"),
            mean_rank=("rank", "mean"),
            mean_penalty=("penalty", "mean"),
        )
    )

    top_genes = []
    TOP_GENES_PER_FATE = 12

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(TOP_GENES_PER_FATE)
        )
        top_genes.extend(sub["gene"].tolist())

    top_genes = list(dict.fromkeys(top_genes))

    heat = (
        mean_force
        .pivot_table(index="gene", columns="fate", values="mean_u", fill_value=0)
        .reindex(top_genes)
        .reindex(columns=selected_fates)
    )

    plt.figure(figsize=(1.4 * len(selected_fates) + 6, 0.28 * len(top_genes) + 4))
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "mean CIPHER force u"},
    )
    plt.title("Top positive CIPHER force genes for terminal fate composition")
    plt.xlabel("terminal fate")
    plt.ylabel("gene")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "composition_CIPHER_force_gene_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "composition_CIPHER_force_gene_heatmap.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # 9. PCA colored by composition error and entropy.
    # --------------------------
    MAX_PLOT_CLONES = 3000
    plot_df = cipher_pred.copy()
    if len(plot_df) > MAX_PLOT_CLONES:
        plot_df = plot_df.sample(MAX_PLOT_CLONES, random_state=SEED)

    plot_clone_ids = plot_df["clone_id"].values.astype(int)
    Xplot, ids_plot_used, _ = clone_mean_matrix(
        clone_ids=plot_clone_ids,
        early_mask=early_cloned_mask,
        cell_to_clone=cell_to_clone,
        counts=counts,
        hvg_idx=hvg_idx,
        mu=mu_ref,
        sd=sd_ref,
    )

    plot_df = plot_df.set_index("clone_id").loc[ids_plot_used].reset_index()

    if Xplot.shape[0] >= 3:
        Z = PCA(n_components=2, random_state=SEED).fit_transform(Xplot)

        for color_col, label in [
            ("composition_JS", "JS divergence"),
            ("obs_entropy", "observed fate entropy"),
            ("pred_entropy", "predicted fate entropy"),
            ("composition_cosine", "composition cosine similarity"),
        ]:
            plt.figure(figsize=(7, 6))
            sc = plt.scatter(
                Z[:, 0],
                Z[:, 1],
                c=plot_df[color_col].values,
                s=35,
                alpha=0.8,
                cmap="viridis",
            )
            plt.colorbar(sc, label=label)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"Early clone means colored by {label}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTDIR, f"clone_pca_{color_col}.png"), dpi=300, bbox_inches="tight")
            plt.savefig(os.path.join(OUTDIR, f"clone_pca_{color_col}.svg"), bbox_inches="tight")
            plt.show()

    # ============================================================
    # FINAL PRINTS
    # ============================================================

    print("\n============================================================")
    print("FINAL COMPOSITION SUMMARY")
    print("============================================================")
    print(composition_summary)

    print("\n============================================================")
    print("PER-FATE CIPHER FRACTION PREDICTION")
    print("============================================================")
    print(per_fate_summary[per_fate_summary["model"] == "cipher"])

    print("\n============================================================")
    print("EMPIRICAL NULL P-VALUES")
    print("============================================================")
    print(pvals)

    print("\n============================================================")
    print("TOP POSITIVE COMPOSITION CIPHER FORCE GENES")
    print("============================================================")

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(20)
        )
        print(f"\n{fate}")
        print(", ".join(sub["gene"].astype(str).tolist()))

    print("\nDone. Outputs in:", OUTDIR)



def composition_temperature_calibration():
    global os, gzip, warnings, np, pd, plt, sns, Counter, \
        mmread, issparse, pearsonr, spearmanr, minimize_scalar, StratifiedKFold, PCA, confusion_matrix, \
        OUTDIR, COUNTS_PATH, GENES_PATH, CLONE_PATH, META_PATH, TIME_COL, CELLTYPE_COL, START_COL, \
        WELL_COL, EARLY_TIME, EARLY_CELLTYPE, EARLY_WELL, RESTRICT_STARTING_POPULATION, TERMINAL_TIME, TERMINAL_WELL, EXCLUDE_FATES, \
        MANUAL_SELECTED_FATES, MAX_FATES, MIN_CLONES_WITH_FATE, MIN_TERMINAL_CELLS_PER_CLONE, MIN_EARLY_CELLS_PER_CLONE, MIN_TOTAL_CELLS_PER_CLONE, MIN_SELECTED_FATE_COVERAGE, MIN_SELECTED_TERMINAL_CELLS, \
        N_VAR_GENES, MAX_COV_CELLS, RIDGE, COV_SHRINK_TO_DIAG, USE_FATE_PRIOR, CALIBRATE_TEMPERATURE, TEMP_MIN, TEMP_MAX, \
        N_NULLS, USE_STARTPOP_PRESERVING_NULL, N_SPLITS, SEED, rng, safe_name, softmax_logits, js_div, \
        cosine_similarity, safe_corr, safe_r2, get_cell_to_clone, get_cells_x_genes, zscore_train, select_hvgs_sparse, make_covariance, \
        weighted_mean, clone_mean_matrix, shuffle_rows_within_groups, fit_temperature_from_counts, counts, f, gene_names, clone_mat, \
        meta, cell_to_clone, has_clone, cell_fates, early_mask, terminal_mask, early_all_idx, early_cloned_mask, \
        terminal_cloned_mask, early_cloned_idx, terminal_cloned_idx, candidate_records, global_fate_counts, global_fate_clone_counts, clone_id, cells, \
        early_cells, terminal_cells, fates, vc, terminal_counts_dict, c, starts, dominant_start, \
        dominant_start_frac, candidate_table, fate_summary, selected_fates, clone_table, fate, s, selected_count_cols, \
        obs_frac_cols, Y_all, dominant_idx, clone_table_save, fig, axes, tab, hvg_idx, \
        gene_vars, hvg_genes, cov_idx, Xcov_raw, mu_ref, sd_ref, Xcov, Sigma, \
        evals, evecs, make_composition_cipher_model, get_logits, score_composition_cipher, fit_startpop_composition_baseline, score_startpop_composition_baseline, add_prediction_columns, \
        add_composition_errors, summarize_predictions, X_clones_all, strat_y, min_class_n, n_splits, splitter, clone_to_obs, \
        clone_to_counts, clone_to_start, all_pred_rows, all_null_rows_for_error_plots, summary_rows, force_rows, temperature_rows, fold, \
        train_pos, test_pos, train_clones, test_clones, Xtrain, train_clone_ids_used, n_train_early, Xtest, \
        test_clone_ids_used, n_test_early, Ytrain, Ctrain, Ytest, Ctest, start_train, start_test, \
        true_dom_test, base, j, cipher_model, _, train_logits, T_cipher, raw_scores, \
        logits, Ptest, pred_df, u, delta, direction, idxs, rank, \
        gi, sp_model, raw_sp, logits_sp, P_sp, sp_df, null_id, Ytrain_null, \
        Ctrain_null, null_name, perm, null_model, null_train_logits, T_null, raw_null, logits_null, \
        P_null, null_df, predictions, null_clone_errors, summary_metrics, force_df, temperature_df, composition_summary, \
        per_fate_summary, p_rows, null_models, metric, real_vals, null_vals, real_mean, p_emp, \
        pvals, cipher_pred, n_fates, ncols, nrows, ax, x, y, \
        r, rho, r2, k, obs_cols, pred_cols, obs_heat, pred_heat, \
        cipher_error_compact, error_plot_df, sp_error, model_label_map, perf, point_df, handles, labels, \
        uniq_h, uniq_l, h, l, mx, cm, cm_norm, cipher_force, \
        mean_force, top_genes, TOP_GENES_PER_FATE, sub, heat
    # ============================================================
    # CIPHER-LARRY: terminal clone fate COMPOSITION with
    # fold-wise temperature calibration of CIPHER softmax
    # ============================================================
    #
    # Adds calibrated probabilities:
    #
    #   p_hat_c(f; T) = softmax_f(logit_c,f / T)
    #
    # where T is fit on the TRAIN clones in each CV fold by minimizing
    # multinomial NLL against terminal fate counts:
    #
    #   L(T) = - sum_c sum_f n_c,f log p_hat_c(f; T)
    #
    # This should reduce overconfident 0/1 probabilities while preserving
    # CIPHER score ranking.
    # ============================================================

    import os
    import gzip
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from collections import Counter
    from scipy.io import mmread
    from scipy.sparse import issparse
    from scipy.stats import pearsonr, spearmanr
    from scipy.optimize import minimize_scalar
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix

    warnings.filterwarnings("ignore")

    # ============================================================
    # CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_clone_fate_composition_temperature_calibrated")
    os.makedirs(OUTDIR, exist_ok=True)

    COUNTS_PATH = os.path.join(SUPPL, "stateFate_inVitro_normed_counts.mtx.gz")
    GENES_PATH  = os.path.join(SUPPL, "stateFate_inVitro_gene_names.txt.gz")
    CLONE_PATH  = os.path.join(SUPPL, "stateFate_inVitro_clone_matrix.mtx.gz")
    META_PATH   = os.path.join(SUPPL, "stateFate_inVitro_metadata.txt.gz")

    TIME_COL = "Time point"
    CELLTYPE_COL = "Cell type annotation"
    START_COL = "Starting population"
    WELL_COL = "Well"

    EARLY_TIME = 4.0
    EARLY_CELLTYPE = "Undifferentiated"
    EARLY_WELL = None

    # Set to None for all starting populations.
    # Or use "Lin-Kit+Sca1+" / "Lin-Kit+Sca1-" for within-start-pop analysis.
    RESTRICT_STARTING_POPULATION = None

    TERMINAL_TIME = 6.0
    TERMINAL_WELL = None

    EXCLUDE_FATES = {
        "Undifferentiated", "Unknown", "unknown", "nan", "NaN",
        "Ambiguous", "ambiguous", "None", ""
    }

    MANUAL_SELECTED_FATES = None
    # Example:
    # MANUAL_SELECTED_FATES = ["Monocyte", "Neutrophil", "Baso", "Mast", "Meg"]

    MAX_FATES = 5
    MIN_CLONES_WITH_FATE = 8
    MIN_TERMINAL_CELLS_PER_CLONE = 5
    MIN_EARLY_CELLS_PER_CLONE = 1
    MIN_TOTAL_CELLS_PER_CLONE = 8

    MIN_SELECTED_FATE_COVERAGE = 0.75
    MIN_SELECTED_TERMINAL_CELLS = 5

    N_VAR_GENES = 500
    MAX_COV_CELLS = 50000

    RIDGE = 0.00000001
    COV_SHRINK_TO_DIAG = 0.0

    USE_FATE_PRIOR = False

    # Temperature calibration.
    CALIBRATE_TEMPERATURE = True
    TEMP_MIN = 0.1
    TEMP_MAX = 100.0

    # Nulls.
    N_NULLS = 100
    USE_STARTPOP_PRESERVING_NULL = True

    N_SPLITS = 5
    SEED = 0
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    plt.rcParams.update({"font.size": 13})
    sns.set_context("talk")

    # ============================================================
    # HELPERS
    # ============================================================


    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )

    def softmax_logits(logits, temperature=1.0, eps=1e-12):
        logits = np.asarray(logits, dtype=float) / max(float(temperature), eps)
        z = logits - np.max(logits, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.maximum(ez.sum(axis=1, keepdims=True), eps)


    def js_div(P, Q, eps=1e-12):
        M = 0.5 * (P + Q)
        return 0.5 * kl_div(P, M, eps=eps) + 0.5 * kl_div(Q, M, eps=eps)

    def cosine_similarity(P, Q, eps=1e-12):
        num = np.sum(P * Q, axis=1)
        den = np.sqrt(np.sum(P * P, axis=1)) * np.sqrt(np.sum(Q * Q, axis=1))
        return num / np.maximum(den, eps)

    def safe_corr(x, y, method="pearson"):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        try:
            if method == "pearson":
                return pearsonr(x, y)[0]
            return spearmanr(x, y)[0]
        except Exception:
            return np.nan

    def safe_r2(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        if ss_tot < 1e-12:
            return np.nan
        return 1.0 - ss_res / ss_tot

    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()
        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row
        return cell_to_clone

    def get_cells_x_genes(counts, cell_idx, gene_idx):
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)

    def zscore_train(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu, sd


    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        X = counts[:, cell_idx]
        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means**2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]

        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_var_genes:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_

    def make_covariance(X):
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D
        Sigma = Sigma + RIDGE * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)

    def weighted_mean(X, w, eps=1e-12):
        w = np.asarray(w, dtype=float)
        return (w[:, None] * X).sum(axis=0) / max(w.sum(), eps)

    def clone_mean_matrix(clone_ids, early_mask, cell_to_clone, counts, hvg_idx, mu, sd):
        rows = []
        out_ids = []
        out_n = []

        for cid in clone_ids:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]
            if len(idx) == 0:
                continue

            X = get_cells_x_genes(counts, idx, hvg_idx)
            X = apply_zscore(X, mu, sd)

            rows.append(X.mean(axis=0))
            out_ids.append(cid)
            out_n.append(len(idx))

        if len(rows) == 0:
            return (
                np.empty((0, len(hvg_idx))),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        return np.vstack(rows), np.asarray(out_ids, dtype=int), np.asarray(out_n, dtype=int)

    def shuffle_rows_within_groups(Y, groups):
        Y = np.asarray(Y).copy()
        groups = np.asarray(groups).astype(str)
        out = Y.copy()

        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) > 1:
                out[idx] = out[rng.permutation(idx)]

        return out

    def fit_temperature_from_counts(logits, counts, temp_min=0.25, temp_max=100.0):
        """
        Fit scalar temperature T by minimizing multinomial NLL:

            -sum_c sum_f n_cf log softmax(logits_c / T)_f

        logits: clones x fates
        counts: clones x fates terminal cell counts
        """
        logits = np.asarray(logits, dtype=float)
        counts = np.asarray(counts, dtype=float)

        if logits.shape != counts.shape:
            raise ValueError("logits and counts must have same shape")

        if counts.sum() <= 0:
            return 1.0

        def nll_logT(logT):
            T = float(np.exp(logT))
            P = softmax_logits(logits, temperature=T)
            return -float(np.sum(counts * np.log(np.clip(P, 1e-12, 1.0))))

        res = minimize_scalar(
            nll_logT,
            bounds=(np.log(temp_min), np.log(temp_max)),
            method="bounded",
            options={"xatol": 1e-4}
        )

        if not res.success or not np.isfinite(res.fun):
            return 1.0

        return float(np.exp(res.x))

    # ============================================================
    # LOAD DATA
    # ============================================================

    counts = mmread(COUNTS_PATH).T.tocsr()
    print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} cells | nnz={counts.nnz:,}")

    with gzip.open(GENES_PATH, "rt") as f:
        gene_names = np.array([line.strip() for line in f])
    print(f"Genes loaded: {len(gene_names)}")

    clone_mat = mmread(CLONE_PATH).T.tocsr()
    print(f"Clone matrix: {clone_mat.shape[0]} clones x {clone_mat.shape[1]} cells")
    print(f"% cells with clone label: {(clone_mat.sum(axis=0) > 0).mean() * 100:.2f}%")

    meta = pd.read_csv(META_PATH, sep="\t")
    meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")

    print(f"Meta: {meta.shape[0]} rows x {meta.shape[1]} cols")
    print("Meta columns:", list(meta.columns))

    assert counts.shape[1] == meta.shape[0] == clone_mat.shape[1], "cells mismatch"
    assert counts.shape[0] == len(gene_names), "genes mismatch"

    print("\nTimepoints:")
    print(np.sort(meta[TIME_COL].dropna().unique()))

    print("\nCell annotations:")
    print(meta[CELLTYPE_COL].value_counts())

    cell_to_clone = get_cell_to_clone(clone_mat)
    has_clone = cell_to_clone >= 0
    cell_fates = meta[CELLTYPE_COL].astype(str).values

    # ============================================================
    # DEFINE EARLY / TERMINAL MASKS
    # ============================================================

    early_mask = meta[TIME_COL].astype(float).values == float(EARLY_TIME)

    if EARLY_CELLTYPE is not None:
        early_mask &= meta[CELLTYPE_COL].astype(str).values == str(EARLY_CELLTYPE)

    if EARLY_WELL is not None and WELL_COL in meta.columns:
        early_mask &= meta[WELL_COL].astype(float).values == float(EARLY_WELL)

    if RESTRICT_STARTING_POPULATION is not None and START_COL in meta.columns:
        early_mask &= meta[START_COL].astype(str).values == str(RESTRICT_STARTING_POPULATION)

    terminal_mask = meta[TIME_COL].astype(float).values == float(TERMINAL_TIME)

    if TERMINAL_WELL is not None and WELL_COL in meta.columns:
        terminal_mask &= meta[WELL_COL].astype(float).values == float(TERMINAL_WELL)

    terminal_mask &= ~np.isin(cell_fates, list(EXCLUDE_FATES))

    early_all_idx = np.where(early_mask)[0]
    early_cloned_mask = early_mask & has_clone
    terminal_cloned_mask = terminal_mask & has_clone

    early_cloned_idx = np.where(early_cloned_mask)[0]
    terminal_cloned_idx = np.where(terminal_cloned_mask)[0]

    print(f"\nAll early cells for Sigma: {len(early_all_idx):,}")
    print(f"Cloned early cells: {len(early_cloned_idx):,}")
    print(f"Cloned terminal cells: {len(terminal_cloned_idx):,}")

    if len(early_all_idx) == 0:
        raise RuntimeError("No early cells found.")

    if len(terminal_cloned_idx) == 0:
        raise RuntimeError("No terminal cloned cells found.")

    # ============================================================
    # BUILD CLONE TABLE WITH TERMINAL COMPOSITION
    # ============================================================

    candidate_records = []
    global_fate_counts = Counter()
    global_fate_clone_counts = Counter()

    for clone_id in range(clone_mat.shape[0]):
        cells = clone_mat[clone_id].indices

        if len(cells) < MIN_TOTAL_CELLS_PER_CLONE:
            continue

        early_cells = cells[early_cloned_mask[cells]]
        terminal_cells = cells[terminal_cloned_mask[cells]]

        if len(early_cells) < MIN_EARLY_CELLS_PER_CLONE:
            continue
        if len(terminal_cells) < MIN_TERMINAL_CELLS_PER_CLONE:
            continue

        fates = pd.Series(cell_fates[terminal_cells].astype(str))
        fates = fates[~fates.isin(EXCLUDE_FATES)]

        if len(fates) < MIN_TERMINAL_CELLS_PER_CLONE:
            continue

        vc = fates.value_counts()
        terminal_counts_dict = {str(k): int(v) for k, v in vc.items()}

        for f, c in terminal_counts_dict.items():
            global_fate_counts[f] += c
            if c > 0:
                global_fate_clone_counts[f] += 1

        if START_COL in meta.columns:
            starts = meta.iloc[early_cells][START_COL].astype(str).value_counts()
            dominant_start = starts.index[0]
            dominant_start_frac = float(starts.iloc[0] / starts.sum())
        else:
            dominant_start = "unknown"
            dominant_start_frac = 1.0

        candidate_records.append({
            "clone_id": int(clone_id),
            "n_total_clone_cells": int(len(cells)),
            "n_early": int(len(early_cells)),
            "n_terminal": int(len(fates)),
            "terminal_counts_dict": terminal_counts_dict,
            "dominant_starting_population": dominant_start,
            "dominant_starting_population_frac": dominant_start_frac,
        })

    candidate_table = pd.DataFrame(candidate_records)

    if candidate_table.empty:
        raise RuntimeError("No clones passed initial early/terminal filters.")

    fate_summary = pd.DataFrame({
        "fate": list(global_fate_counts.keys()),
        "terminal_cell_count": [global_fate_counts[f] for f in global_fate_counts.keys()],
        "clone_count_with_fate": [global_fate_clone_counts[f] for f in global_fate_counts.keys()],
    }).sort_values("terminal_cell_count", ascending=False)

    fate_summary.to_csv(os.path.join(OUTDIR, "terminal_fate_summary_before_selection.csv"), index=False)

    if MANUAL_SELECTED_FATES is None:
        selected_fates = (
            fate_summary[fate_summary["clone_count_with_fate"] >= MIN_CLONES_WITH_FATE]
            .head(MAX_FATES)["fate"]
            .tolist()
        )
    else:
        selected_fates = list(MANUAL_SELECTED_FATES)

    if len(selected_fates) < 2:
        raise RuntimeError("Fewer than two selected fates.")

    print("\nSelected fates for composition:")
    print(selected_fates)

    clone_table = candidate_table.copy()

    for fate in selected_fates:
        s = safe_name(fate)
        clone_table[f"terminal_count__{s}"] = clone_table["terminal_counts_dict"].apply(lambda d: int(d.get(fate, 0)))

    selected_count_cols = [f"terminal_count__{safe_name(f)}" for f in selected_fates]
    clone_table["n_terminal_selected"] = clone_table[selected_count_cols].sum(axis=1)
    clone_table["selected_fate_coverage"] = clone_table["n_terminal_selected"] / clone_table["n_terminal"]

    clone_table = clone_table[
        (clone_table["n_terminal_selected"] >= MIN_SELECTED_TERMINAL_CELLS) &
        (clone_table["selected_fate_coverage"] >= MIN_SELECTED_FATE_COVERAGE)
    ].copy()

    if clone_table.empty:
        raise RuntimeError("No clones passed selected fate coverage filtering.")

    for fate in selected_fates:
        s = safe_name(fate)
        clone_table[f"obs_frac__{s}"] = clone_table[f"terminal_count__{s}"] / clone_table["n_terminal_selected"]

    obs_frac_cols = [f"obs_frac__{safe_name(f)}" for f in selected_fates]

    Y_all = clone_table[obs_frac_cols].values.astype(float)
    dominant_idx = np.argmax(Y_all, axis=1)
    clone_table["dominant_selected_fate"] = np.array(selected_fates, dtype=object)[dominant_idx]
    clone_table["terminal_entropy_selected"] = entropy(Y_all)

    clone_table = clone_table.reset_index(drop=True)

    print("\nClone table after composition filters:")
    print(f"n clones: {len(clone_table):,}")
    print("Dominant selected fate counts:")
    print(clone_table["dominant_selected_fate"].value_counts())
    print("\nMean selected fate coverage:", clone_table["selected_fate_coverage"].mean())

    clone_table_save = clone_table.drop(columns=["terminal_counts_dict"])
    clone_table_save.to_csv(os.path.join(OUTDIR, "clone_terminal_composition_table.csv"), index=False)

    # ============================================================
    # QC PLOTS
    # ============================================================

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    sns.countplot(
        data=clone_table,
        x="dominant_selected_fate",
        order=clone_table["dominant_selected_fate"].value_counts().index,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Selected clones by dominant terminal fate")
    axes[0, 0].set_xlabel("dominant terminal fate")
    axes[0, 0].set_ylabel("clone count")
    axes[0, 0].tick_params(axis="x", rotation=45)

    sns.histplot(data=clone_table, x="n_total_clone_cells", bins=40, ax=axes[0, 1])
    axes[0, 1].set_title("Total cells per retained clone")

    sns.histplot(data=clone_table, x="n_early", bins=30, ax=axes[0, 2])
    axes[0, 2].set_title("Early cells per retained clone")

    sns.histplot(data=clone_table, x="n_terminal_selected", bins=40, ax=axes[1, 0])
    axes[1, 0].set_title("Selected terminal cells per clone")

    sns.histplot(data=clone_table, x="terminal_entropy_selected", bins=30, ax=axes[1, 1])
    axes[1, 1].set_title("Observed terminal composition entropy")

    sns.scatterplot(
        data=clone_table,
        x="n_early",
        y="n_terminal_selected",
        hue="dominant_selected_fate",
        ax=axes[1, 2],
        s=45,
    )
    axes[1, 2].set_title("Early vs terminal representation")
    axes[1, 2].legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "clone_composition_qc_summary.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "clone_composition_qc_summary.svg"), bbox_inches="tight")
    plt.show()

    if START_COL in meta.columns:
        plt.figure(figsize=(10, 5))
        tab = pd.crosstab(
            clone_table["dominant_selected_fate"],
            clone_table["dominant_starting_population"],
        ).reindex(selected_fates)
        sns.heatmap(tab, annot=True, fmt="d", cmap="viridis")
        plt.title("Dominant terminal fate vs early starting population")
        plt.xlabel("dominant early starting population")
        plt.ylabel("dominant terminal fate")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "dominant_fate_vs_starting_population.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "dominant_fate_vs_starting_population.svg"), bbox_inches="tight")
        plt.show()

    # ============================================================
    # HVGs + SIGMA
    # ============================================================

    print("\nSelecting HVGs from early cells...")

    hvg_idx, gene_vars = select_hvgs_sparse(
        counts=counts,
        cell_idx=early_all_idx,
        n_var_genes=N_VAR_GENES,
    )
    hvg_genes = gene_names[hvg_idx]

    pd.DataFrame({
        "gene": hvg_genes,
        "gene_index": hvg_idx,
        "early_variance": gene_vars[hvg_idx],
    }).to_csv(os.path.join(OUTDIR, "selected_early_hvgs.csv"), index=False)

    cov_idx = early_all_idx.copy()
    if len(cov_idx) > MAX_COV_CELLS:
        cov_idx = rng.choice(cov_idx, size=MAX_COV_CELLS, replace=False)

    print(f"Using {len(cov_idx):,} cells for Sigma.")

    Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx)
    mu_ref, sd_ref = zscore_train(Xcov_raw)
    Xcov = apply_zscore(Xcov_raw, mu_ref, sd_ref)

    Sigma = make_covariance(Xcov)

    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-8)

    pd.DataFrame({
        "rank": np.arange(1, len(evals) + 1),
        "eigenvalue": evals[::-1],
    }).to_csv(os.path.join(OUTDIR, "early_covariance_eigenvalues.csv"), index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(evals) + 1), evals[::-1], marker="o", linewidth=1, markersize=3)
    plt.yscale("log")
    plt.xlabel("eigenvalue rank")
    plt.ylabel("eigenvalue")
    plt.title("Early progenitor covariance spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "early_covariance_spectrum.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # MODEL FUNCTIONS
    # ============================================================

    def make_composition_cipher_model(Xtrain_clone, Ytrain, selected_fates, evals, evecs, Sigma, use_fate_prior=False):
        U = []
        DELTAS = []
        penalties = []
        log_priors = []

        eps = 1e-12
        Ytrain = np.asarray(Ytrain, dtype=float)

        for j, fate in enumerate(selected_fates):
            w_pos = Ytrain[:, j].copy()
            w_neg = 1.0 - w_pos

            if w_pos.sum() < eps or w_neg.sum() < eps:
                delta = np.zeros(Xtrain_clone.shape[1])
            else:
                mu_pos = weighted_mean(Xtrain_clone, w_pos)
                mu_neg = weighted_mean(Xtrain_clone, w_neg)
                delta = mu_pos - mu_neg

            u = evecs @ ((evecs.T @ delta) / evals)
            penalty = 0.5 * float(u @ Sigma @ u)

            if use_fate_prior:
                prior = max(float(Ytrain[:, j].mean()), eps)
                log_prior = np.log(prior)
            else:
                log_prior = 0.0

            U.append(u)
            DELTAS.append(delta)
            penalties.append(penalty)
            log_priors.append(log_prior)

        return {
            "U": np.asarray(U),
            "DELTAS": np.asarray(DELTAS),
            "penalty": np.asarray(penalties),
            "log_prior": np.asarray(log_priors),
            "temperature": 1.0,
        }

    def get_logits(X, model):
        U = model["U"]
        raw_scores = X @ U.T
        logits = raw_scores - model["penalty"][None, :] + model["log_prior"][None, :]
        return raw_scores, logits

    def score_composition_cipher(X, model, temperature=None):
        raw_scores, logits = get_logits(X, model)

        if temperature is None:
            temperature = model.get("temperature", 1.0)

        P = softmax_logits(logits, temperature=temperature)
        return raw_scores, logits, P

    def fit_startpop_composition_baseline(Ytrain, start_train, alpha=2.0):
        Ytrain = np.asarray(Ytrain, dtype=float)
        start_train = np.asarray(start_train).astype(str)

        global_p = Ytrain.mean(axis=0)
        global_p = global_p / global_p.sum()

        table = {}
        for s in np.unique(start_train):
            idx = np.where(start_train == s)[0]
            if len(idx) == 0:
                continue
            p = (Ytrain[idx].sum(axis=0) + alpha * global_p) / (len(idx) + alpha)
            p = p / p.sum()
            table[s] = p

        return {
            "global_p": global_p,
            "table": table,
        }

    def score_startpop_composition_baseline(start_test, model):
        start_test = np.asarray(start_test).astype(str)
        P = []
        for s in start_test:
            P.append(model["table"].get(s, model["global_p"]))
        P = np.vstack(P)
        logits = np.log(np.clip(P, 1e-12, 1.0))
        raw_scores = logits.copy()
        return raw_scores, logits, P

    def add_prediction_columns(base_df, raw_scores, logits, P, selected_fates, model_name, temperature=1.0):
        rows = base_df.copy()
        rows["model"] = model_name
        rows["temperature"] = float(temperature)

        pred_idx = np.argmax(P, axis=1)
        rows["pred_dominant_fate"] = np.array(selected_fates, dtype=object)[pred_idx]
        rows["pred_entropy"] = entropy(P)
        rows["pred_max_prob"] = P.max(axis=1)

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            rows[f"score_raw__{s}"] = raw_scores[:, j]
            rows[f"logit__{s}"] = logits[:, j]
            rows[f"pred_frac__{s}"] = P[:, j]

        return rows

    def add_composition_errors(df, selected_fates):
        obs = df[[f"obs_frac__{safe_name(f)}" for f in selected_fates]].values.astype(float)
        pred = df[[f"pred_frac__{safe_name(f)}" for f in selected_fates]].values.astype(float)

        df = df.copy()
        df["composition_KL_obs_pred"] = kl_div(obs, pred)
        df["composition_JS"] = js_div(obs, pred)
        df["composition_Brier"] = np.mean((obs - pred) ** 2, axis=1)
        df["composition_L1"] = np.sum(np.abs(obs - pred), axis=1)
        df["composition_cosine"] = cosine_similarity(obs, pred)
        df["obs_entropy"] = entropy(obs)
        df["pred_entropy"] = entropy(pred)
        df["true_dominant_fate"] = np.array(selected_fates, dtype=object)[np.argmax(obs, axis=1)]
        df["pred_dominant_fate"] = np.array(selected_fates, dtype=object)[np.argmax(pred, axis=1)]
        df["dominant_fate_correct"] = df["true_dominant_fate"].values == df["pred_dominant_fate"].values

        return df

    def summarize_predictions(df, selected_fates, model_name, fold, null_id=None):
        rows = []

        obs = df[[f"obs_frac__{safe_name(f)}" for f in selected_fates]].values.astype(float)
        pred = df[[f"pred_frac__{safe_name(f)}" for f in selected_fates]].values.astype(float)

        rows.append({
            "model": model_name,
            "fold": fold,
            "null_id": null_id,
            "metric_type": "composition",
            "fate": "ALL",
            "mean_temperature": df["temperature"].mean() if "temperature" in df.columns else 1.0,
            "mean_KL": np.mean(kl_div(obs, pred)),
            "mean_JS": np.mean(js_div(obs, pred)),
            "mean_Brier": np.mean(np.mean((obs - pred) ** 2, axis=1)),
            "mean_L1": np.mean(np.sum(np.abs(obs - pred), axis=1)),
            "mean_cosine": np.mean(cosine_similarity(obs, pred)),
            "top1_accuracy": np.mean(np.argmax(obs, axis=1) == np.argmax(pred, axis=1)),
            "entropy_pearson": safe_corr(entropy(obs), entropy(pred), method="pearson"),
            "entropy_spearman": safe_corr(entropy(obs), entropy(pred), method="spearman"),
            "n_clones": len(df),
        })

        for j, fate in enumerate(selected_fates):
            y = obs[:, j]
            p = pred[:, j]

            rows.append({
                "model": model_name,
                "fold": fold,
                "null_id": null_id,
                "metric_type": "per_fate_fraction",
                "fate": fate,
                "mean_temperature": df["temperature"].mean() if "temperature" in df.columns else 1.0,
                "pearson": safe_corr(y, p, method="pearson"),
                "spearman": safe_corr(y, p, method="spearman"),
                "r2": safe_r2(y, p),
                "mae": np.mean(np.abs(y - p)),
                "rmse": np.sqrt(np.mean((y - p) ** 2)),
                "mean_obs_fraction": np.mean(y),
                "mean_pred_fraction": np.mean(p),
                "n_clones": len(df),
            })

        return pd.DataFrame(rows)

    # ============================================================
    # CROSS-VALIDATED COMPOSITION PREDICTION
    # ============================================================

    X_clones_all = clone_table["clone_id"].values.astype(int)
    strat_y = clone_table["dominant_selected_fate"].values.astype(str)

    min_class_n = clone_table["dominant_selected_fate"].value_counts().min()
    n_splits = int(min(N_SPLITS, min_class_n))

    if n_splits < 2:
        raise RuntimeError(f"Cannot do CV. Smallest dominant fate has only {min_class_n} clones.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    clone_to_obs = {
        int(row["clone_id"]): row[obs_frac_cols].values.astype(float)
        for _, row in clone_table.iterrows()
    }

    clone_to_counts = {
        int(row["clone_id"]): row[selected_count_cols].values.astype(int)
        for _, row in clone_table.iterrows()
    }

    clone_to_start = dict(zip(clone_table["clone_id"].astype(int), clone_table["dominant_starting_population"].astype(str)))

    all_pred_rows = []
    all_null_rows_for_error_plots = []
    summary_rows = []
    force_rows = []
    temperature_rows = []

    for fold, (train_pos, test_pos) in enumerate(splitter.split(X_clones_all, strat_y)):
        train_clones = X_clones_all[train_pos]
        test_clones = X_clones_all[test_pos]

        print(f"\nFold {fold + 1}/{n_splits}: train={len(train_clones)}, test={len(test_clones)}")

        Xtrain, train_clone_ids_used, n_train_early = clone_mean_matrix(
            clone_ids=train_clones,
            early_mask=early_cloned_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        Xtest, test_clone_ids_used, n_test_early = clone_mean_matrix(
            clone_ids=test_clones,
            early_mask=early_cloned_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        Ytrain = np.vstack([clone_to_obs[int(c)] for c in train_clone_ids_used])
        Ctrain = np.vstack([clone_to_counts[int(c)] for c in train_clone_ids_used])

        Ytest = np.vstack([clone_to_obs[int(c)] for c in test_clone_ids_used])
        Ctest = np.vstack([clone_to_counts[int(c)] for c in test_clone_ids_used])

        start_train = np.array([clone_to_start.get(int(c), "unknown") for c in train_clone_ids_used])
        start_test = np.array([clone_to_start.get(int(c), "unknown") for c in test_clone_ids_used])

        true_dom_test = np.array(selected_fates, dtype=object)[np.argmax(Ytest, axis=1)]

        base = pd.DataFrame({
            "fold": fold,
            "clone_id": test_clone_ids_used,
            "n_early_scored": n_test_early,
            "dominant_starting_population": start_test,
            "true_dominant_fate": true_dom_test,
        })

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            base[f"obs_frac__{s}"] = Ytest[:, j]
            base[f"terminal_count__{s}"] = Ctest[:, j]

        base["obs_entropy"] = entropy(Ytest)
        base["n_terminal_selected"] = Ctest.sum(axis=1)

        # --------------------------
        # CIPHER model.
        # --------------------------
        cipher_model = make_composition_cipher_model(
            Xtrain_clone=Xtrain,
            Ytrain=Ytrain,
            selected_fates=selected_fates,
            evals=evals,
            evecs=evecs,
            Sigma=Sigma,
            use_fate_prior=USE_FATE_PRIOR,
        )

        _, train_logits = get_logits(Xtrain, cipher_model)

        if CALIBRATE_TEMPERATURE:
            T_cipher = fit_temperature_from_counts(
                logits=train_logits,
                counts=Ctrain,
                temp_min=TEMP_MIN,
                temp_max=TEMP_MAX,
            )
        else:
            T_cipher = 1.0

        cipher_model["temperature"] = T_cipher

        temperature_rows.append({
            "fold": fold,
            "model": "cipher",
            "null_id": None,
            "temperature": T_cipher,
        })

        raw_scores, logits, Ptest = score_composition_cipher(Xtest, cipher_model, temperature=T_cipher)

        pred_df = add_prediction_columns(
            base,
            raw_scores,
            logits,
            Ptest,
            selected_fates,
            "cipher",
            temperature=T_cipher,
        )
        pred_df = add_composition_errors(pred_df, selected_fates)

        all_pred_rows.append(pred_df)
        summary_rows.append(summarize_predictions(pred_df, selected_fates, "cipher", fold))

        # Save force genes.
        for j, fate in enumerate(selected_fates):
            u = cipher_model["U"][j]
            delta = cipher_model["DELTAS"][j]

            for direction, idxs in [
                ("positive", np.argsort(u)[::-1][:50]),
                ("negative", np.argsort(u)[:50]),
            ]:
                for rank, gi in enumerate(idxs, start=1):
                    force_rows.append({
                        "fold": fold,
                        "model": "cipher",
                        "fate": fate,
                        "direction": direction,
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),
                        "u": float(u[gi]),
                        "delta_weighted_composition": float(delta[gi]),
                        "penalty": float(cipher_model["penalty"][j]),
                        "log_prior": float(cipher_model["log_prior"][j]),
                        "temperature": float(T_cipher),
                    })

        # --------------------------
        # Starting-population-only baseline.
        # --------------------------
        if START_COL in meta.columns and RESTRICT_STARTING_POPULATION is None:
            sp_model = fit_startpop_composition_baseline(Ytrain, start_train)
            raw_sp, logits_sp, P_sp = score_startpop_composition_baseline(start_test, sp_model)

            sp_df = add_prediction_columns(
                base,
                raw_sp,
                logits_sp,
                P_sp,
                selected_fates,
                "starting_population_only",
                temperature=1.0,
            )
            sp_df = add_composition_errors(sp_df, selected_fates)

            all_pred_rows.append(sp_df)
            summary_rows.append(summarize_predictions(sp_df, selected_fates, "starting_population_only", fold))

        # --------------------------
        # Nulls: shuffled composition vectors.
        # --------------------------
        for null_id in range(N_NULLS):
            if USE_STARTPOP_PRESERVING_NULL and START_COL in meta.columns and RESTRICT_STARTING_POPULATION is None:
                Ytrain_null = shuffle_rows_within_groups(Ytrain, start_train)
                Ctrain_null = shuffle_rows_within_groups(Ctrain, start_train)
                null_name = "startpop_preserving_null"
            else:
                perm = rng.permutation(Ytrain.shape[0])
                Ytrain_null = Ytrain[perm]
                Ctrain_null = Ctrain[perm]
                null_name = "shuffled_null"

            null_model = make_composition_cipher_model(
                Xtrain_clone=Xtrain,
                Ytrain=Ytrain_null,
                selected_fates=selected_fates,
                evals=evals,
                evecs=evecs,
                Sigma=Sigma,
                use_fate_prior=USE_FATE_PRIOR,
            )

            _, null_train_logits = get_logits(Xtrain, null_model)

            if CALIBRATE_TEMPERATURE:
                T_null = fit_temperature_from_counts(
                    logits=null_train_logits,
                    counts=Ctrain_null,
                    temp_min=TEMP_MIN,
                    temp_max=TEMP_MAX,
                )
            else:
                T_null = 1.0

            null_model["temperature"] = T_null

            temperature_rows.append({
                "fold": fold,
                "model": null_name,
                "null_id": null_id,
                "temperature": T_null,
            })

            raw_null, logits_null, P_null = score_composition_cipher(Xtest, null_model, temperature=T_null)

            null_df = add_prediction_columns(
                base,
                raw_null,
                logits_null,
                P_null,
                selected_fates,
                null_name,
                temperature=T_null,
            )
            null_df = add_composition_errors(null_df, selected_fates)
            null_df["null_id"] = null_id

            all_null_rows_for_error_plots.append(
                null_df[[
                    "fold", "null_id", "model", "temperature", "clone_id",
                    "composition_KL_obs_pred", "composition_JS",
                    "composition_Brier", "composition_L1", "composition_cosine",
                    "dominant_fate_correct", "obs_entropy", "pred_entropy",
                ]].copy()
            )

            summary_rows.append(summarize_predictions(null_df, selected_fates, null_name, fold, null_id=null_id))

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    null_clone_errors = pd.concat(all_null_rows_for_error_plots, ignore_index=True)
    summary_metrics = pd.concat(summary_rows, ignore_index=True)
    force_df = pd.DataFrame(force_rows)
    temperature_df = pd.DataFrame(temperature_rows)

    predictions.to_csv(os.path.join(OUTDIR, "clone_composition_predictions_temperature_calibrated.csv"), index=False)
    null_clone_errors.to_csv(os.path.join(OUTDIR, "null_clone_composition_errors_temperature_calibrated.csv"), index=False)
    summary_metrics.to_csv(os.path.join(OUTDIR, "composition_prediction_summary_metrics_temperature_calibrated.csv"), index=False)
    force_df.to_csv(os.path.join(OUTDIR, "composition_CIPHER_force_genes.csv"), index=False)
    temperature_df.to_csv(os.path.join(OUTDIR, "fold_temperature_values.csv"), index=False)

    print("\nSaved:")
    print(os.path.join(OUTDIR, "clone_composition_predictions_temperature_calibrated.csv"))
    print(os.path.join(OUTDIR, "composition_prediction_summary_metrics_temperature_calibrated.csv"))
    print(os.path.join(OUTDIR, "fold_temperature_values.csv"))

    # ============================================================
    # FINAL SUMMARIES
    # ============================================================

    composition_summary = (
        summary_metrics[summary_metrics["metric_type"] == "composition"]
        .groupby("model", as_index=False)
        .agg(
            mean_temperature=("mean_temperature", "mean"),
            mean_KL=("mean_KL", "mean"),
            mean_JS=("mean_JS", "mean"),
            mean_Brier=("mean_Brier", "mean"),
            mean_L1=("mean_L1", "mean"),
            mean_cosine=("mean_cosine", "mean"),
            top1_accuracy=("top1_accuracy", "mean"),
            entropy_pearson=("entropy_pearson", "mean"),
            entropy_spearman=("entropy_spearman", "mean"),
        )
    )

    per_fate_summary = (
        summary_metrics[summary_metrics["metric_type"] == "per_fate_fraction"]
        .groupby(["model", "fate"], as_index=False)
        .agg(
            mean_temperature=("mean_temperature", "mean"),
            pearson_mean=("pearson", "mean"),
            pearson_sd=("pearson", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_sd=("spearman", "std"),
            r2_mean=("r2", "mean"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            mean_obs_fraction=("mean_obs_fraction", "mean"),
            mean_pred_fraction=("mean_pred_fraction", "mean"),
        )
    )

    composition_summary.to_csv(os.path.join(OUTDIR, "composition_summary_by_model_temperature_calibrated.csv"), index=False)
    per_fate_summary.to_csv(os.path.join(OUTDIR, "per_fate_fraction_summary_by_model_temperature_calibrated.csv"), index=False)

    print("\nTemperature summary:")
    print(
        temperature_df
        .groupby("model", as_index=False)
        .agg(
            mean_T=("temperature", "mean"),
            sd_T=("temperature", "std"),
            min_T=("temperature", "min"),
            max_T=("temperature", "max"),
        )
    )

    print("\nComposition summary:")
    print(composition_summary)

    print("\nPer-fate CIPHER fraction prediction summary:")
    print(per_fate_summary[per_fate_summary["model"] == "cipher"])

    # Empirical p-values vs null.
    p_rows = []
    null_models = [m for m in summary_metrics["model"].unique() if "null" in m]

    for null_model in null_models:
        for metric in ["mean_KL", "mean_JS", "mean_Brier", "mean_L1"]:
            real_vals = summary_metrics[
                (summary_metrics["model"] == "cipher") &
                (summary_metrics["metric_type"] == "composition")
            ][metric].dropna().values

            null_vals = summary_metrics[
                (summary_metrics["model"] == null_model) &
                (summary_metrics["metric_type"] == "composition")
            ][metric].dropna().values

            if len(real_vals) > 0 and len(null_vals) > 0:
                real_mean = np.mean(real_vals)
                p_emp = (1 + np.sum(null_vals <= real_mean)) / (1 + len(null_vals))
            else:
                real_mean = np.nan
                p_emp = np.nan

            p_rows.append({
                "null_model": null_model,
                "metric": metric,
                "cipher_mean": real_mean,
                "null_mean": np.mean(null_vals) if len(null_vals) else np.nan,
                "empirical_p_lower_is_better": p_emp,
                "n_null": len(null_vals),
            })

        for metric in ["mean_cosine", "top1_accuracy", "entropy_pearson", "entropy_spearman"]:
            real_vals = summary_metrics[
                (summary_metrics["model"] == "cipher") &
                (summary_metrics["metric_type"] == "composition")
            ][metric].dropna().values

            null_vals = summary_metrics[
                (summary_metrics["model"] == null_model) &
                (summary_metrics["metric_type"] == "composition")
            ][metric].dropna().values

            if len(real_vals) > 0 and len(null_vals) > 0:
                real_mean = np.mean(real_vals)
                p_emp = (1 + np.sum(null_vals >= real_mean)) / (1 + len(null_vals))
            else:
                real_mean = np.nan
                p_emp = np.nan

            p_rows.append({
                "null_model": null_model,
                "metric": metric,
                "cipher_mean": real_mean,
                "null_mean": np.mean(null_vals) if len(null_vals) else np.nan,
                "empirical_p_higher_is_better": p_emp,
                "n_null": len(null_vals),
            })

    pvals = pd.DataFrame(p_rows)
    pvals.to_csv(os.path.join(OUTDIR, "composition_empirical_pvalues_vs_null_temperature_calibrated.csv"), index=False)

    print("\nEmpirical p-values vs null:")
    print(pvals)

    # ============================================================
    # PLOTS
    # ============================================================

    cipher_pred = predictions[predictions["model"] == "cipher"].copy()

    # --------------------------
    # 1. Temperature distribution.
    # --------------------------
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=temperature_df, x="model", y="temperature", showfliers=False)
    sns.stripplot(data=temperature_df[temperature_df["model"] == "cipher"], x="model", y="temperature", color="black", size=6)
    plt.yscale("log")
    plt.title("Fitted softmax temperature")
    plt.xlabel("")
    plt.ylabel("temperature T")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fitted_temperature_distribution.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "fitted_temperature_distribution.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # 2. Predicted vs observed fate fractions.
    # --------------------------
    n_fates = len(selected_fates)
    ncols = min(3, n_fates)
    nrows = int(np.ceil(n_fates / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for j, fate in enumerate(selected_fates):
        ax = axes[j // ncols][j % ncols]
        s = safe_name(fate)

        x = cipher_pred[f"obs_frac__{s}"].values
        y = cipher_pred[f"pred_frac__{s}"].values

        ax.scatter(x, y, s=35, alpha=0.75)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)

        r = safe_corr(x, y, "pearson")
        rho = safe_corr(x, y, "spearman")
        r2 = safe_r2(x, y)

        ax.set_title(f"{fate}\nPearson={r:.2f}, Spearman={rho:.2f}, R²={r2:.2f}")
        ax.set_xlabel("observed terminal fraction")
        ax.set_ylabel("temperature-calibrated CIPHER fraction")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)

    for k in range(n_fates, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "predicted_vs_observed_fate_fractions_temperature_calibrated.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "predicted_vs_observed_fate_fractions_temperature_calibrated.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # 3. Observed vs predicted composition heatmaps.
    # --------------------------
    obs_cols = [f"obs_frac__{safe_name(f)}" for f in selected_fates]
    pred_cols = [f"pred_frac__{safe_name(f)}" for f in selected_fates]

    obs_heat = cipher_pred.groupby("true_dominant_fate")[obs_cols].mean().reindex(selected_fates)
    pred_heat = cipher_pred.groupby("true_dominant_fate")[pred_cols].mean().reindex(selected_fates)

    obs_heat.columns = selected_fates
    pred_heat.columns = selected_fates

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        obs_heat,
        ax=axes[0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "observed fraction"},
    )
    axes[0].set_title("Observed terminal composition")
    axes[0].set_xlabel("terminal fate")
    axes[0].set_ylabel("dominant terminal fate")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        pred_heat,
        ax=axes[1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "predicted fraction"},
    )
    axes[1].set_title("Temperature-calibrated CIPHER composition")
    axes[1].set_xlabel("predicted terminal fate")
    axes[1].set_ylabel("dominant terminal fate")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_composition_heatmaps_temperature_calibrated.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_composition_heatmaps_temperature_calibrated.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # 4. Composition errors vs null / baseline.
    # --------------------------
    cipher_error_compact = cipher_pred[[
        "model", "fold", "temperature", "clone_id",
        "composition_KL_obs_pred", "composition_JS",
        "composition_Brier", "composition_L1",
        "composition_cosine", "dominant_fate_correct",
        "obs_entropy", "pred_entropy",
    ]].copy()

    error_plot_df = pd.concat([cipher_error_compact, null_clone_errors], ignore_index=True)

    if "starting_population_only" in predictions["model"].unique():
        sp_error = predictions[predictions["model"] == "starting_population_only"][[
            "model", "fold", "temperature", "clone_id",
            "composition_KL_obs_pred", "composition_JS",
            "composition_Brier", "composition_L1",
            "composition_cosine", "dominant_fate_correct",
            "obs_entropy", "pred_entropy",
        ]].copy()
        error_plot_df = pd.concat([error_plot_df, sp_error], ignore_index=True)

    model_label_map = {
        "cipher": "CIPHER",
        "shuffled_null": "shuffled null",
        "startpop_preserving_null": "startpop-preserving null",
        "starting_population_only": "starting-pop only",
    }
    error_plot_df["model_label"] = error_plot_df["model"].map(model_label_map).fillna(error_plot_df["model"])

    for metric in ["composition_JS", "composition_Brier", "composition_L1", "composition_cosine"]:
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=error_plot_df,
            x="model_label",
            y=metric,
            showfliers=False,
        )
        sns.stripplot(
            data=error_plot_df[error_plot_df["model"] == "cipher"],
            x="model_label",
            y=metric,
            color="black",
            alpha=0.35,
            size=3,
        )
        plt.title(f"Temperature-calibrated clone composition prediction: {metric}")
        plt.xlabel("")
        plt.ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"composition_error_{metric}_temperature_calibrated.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"composition_error_{metric}_temperature_calibrated.svg"), bbox_inches="tight")
        plt.show()

    # --------------------------
    # 5. Per-fate correlation vs null / baseline.
    # --------------------------
    perf = summary_metrics[summary_metrics["metric_type"] == "per_fate_fraction"].copy()
    perf["model_label"] = perf["model"].map(model_label_map).fillna(perf["model"])

    for metric in ["pearson", "spearman", "r2"]:
        plt.figure(figsize=(11, 5))
        sns.boxplot(
            data=perf,
            x="fate",
            y=metric,
            hue="model_label",
            order=selected_fates,
            showfliers=False,
        )

        point_df = perf[perf["model"].isin(["cipher", "starting_population_only"])]
        if len(point_df) > 0:
            sns.stripplot(
                data=point_df,
                x="fate",
                y=metric,
                hue="model_label",
                order=selected_fates,
                dodge=True,
                color="black",
                alpha=0.6,
                size=4,
                legend=False,
            )

        plt.axhline(0, color="gray", linestyle="--", linewidth=1.5)
        if metric != "r2":
            plt.ylim(-1, 1)

        plt.title(f"Predicted vs observed terminal fate fraction: {metric}")
        plt.xlabel("terminal fate")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha="right")

        handles, labels = plt.gca().get_legend_handles_labels()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l not in uniq_l:
                uniq_h.append(h)
                uniq_l.append(l)
        plt.legend(uniq_h, uniq_l, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"per_fate_fraction_{metric}_temperature_calibrated.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"per_fate_fraction_{metric}_temperature_calibrated.svg"), bbox_inches="tight")
        plt.show()

    # --------------------------
    # 6. Observed vs predicted entropy.
    # --------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(
        cipher_pred["obs_entropy"],
        cipher_pred["pred_entropy"],
        s=40,
        alpha=0.75,
    )
    mx = max(cipher_pred["obs_entropy"].max(), cipher_pred["pred_entropy"].max())
    plt.plot([0, mx], [0, mx], linestyle="--", color="gray", linewidth=2)
    r = safe_corr(cipher_pred["obs_entropy"], cipher_pred["pred_entropy"], "pearson")
    rho = safe_corr(cipher_pred["obs_entropy"], cipher_pred["pred_entropy"], "spearman")
    plt.title(f"Clone fate entropy\nPearson={r:.2f}, Spearman={rho:.2f}")
    plt.xlabel("observed terminal fate entropy")
    plt.ylabel("temperature-calibrated CIPHER entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_fate_entropy_temperature_calibrated.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_fate_entropy_temperature_calibrated.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # 7. Dominant-fate confusion matrix.
    # --------------------------
    cm = confusion_matrix(
        cipher_pred["true_dominant_fate"],
        cipher_pred["pred_dominant_fate"],
        labels=selected_fates,
    )
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        pd.DataFrame(cm_norm, index=selected_fates, columns=selected_fates),
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "row-normalized fraction"},
    )
    plt.title("Dominant fate prediction from calibrated composition")
    plt.xlabel("predicted dominant fate")
    plt.ylabel("observed dominant fate")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "dominant_fate_confusion_temperature_calibrated.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "dominant_fate_confusion_temperature_calibrated.svg"), bbox_inches="tight")
    plt.show()

    # --------------------------
    # 8. Force gene heatmap.
    # --------------------------
    cipher_force = force_df[
        (force_df["model"] == "cipher") &
        (force_df["direction"] == "positive")
    ].copy()

    mean_force = (
        cipher_force
        .groupby(["fate", "gene"], as_index=False)
        .agg(
            mean_u=("u", "mean"),
            mean_delta=("delta_weighted_composition", "mean"),
            mean_rank=("rank", "mean"),
            mean_penalty=("penalty", "mean"),
            mean_temperature=("temperature", "mean"),
        )
    )

    top_genes = []
    TOP_GENES_PER_FATE = 12

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(TOP_GENES_PER_FATE)
        )
        top_genes.extend(sub["gene"].tolist())

    top_genes = list(dict.fromkeys(top_genes))

    heat = (
        mean_force
        .pivot_table(index="gene", columns="fate", values="mean_u", fill_value=0)
        .reindex(top_genes)
        .reindex(columns=selected_fates)
    )

    plt.figure(figsize=(1.4 * len(selected_fates) + 6, 0.28 * len(top_genes) + 4))
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "mean CIPHER force u"},
    )
    plt.title("Top positive CIPHER force genes for terminal fate composition")
    plt.xlabel("terminal fate")
    plt.ylabel("gene")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "composition_CIPHER_force_gene_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "composition_CIPHER_force_gene_heatmap.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # FINAL PRINTS
    # ============================================================

    print("\n============================================================")
    print("FINAL TEMPERATURE-CALIBRATED COMPOSITION SUMMARY")
    print("============================================================")
    print(composition_summary)

    print("\n============================================================")
    print("PER-FATE CIPHER FRACTION PREDICTION")
    print("============================================================")
    print(per_fate_summary[per_fate_summary["model"] == "cipher"])

    print("\n============================================================")
    print("EMPIRICAL NULL P-VALUES")
    print("============================================================")
    print(pvals)

    print("\n============================================================")
    print("FITTED TEMPERATURES")
    print("============================================================")
    print(temperature_df[temperature_df["model"] == "cipher"])

    print("\n============================================================")
    print("TOP POSITIVE COMPOSITION CIPHER FORCE GENES")
    print("============================================================")

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(20)
        )
        print(f"\n{fate}")
        print(", ".join(sub["gene"].astype(str).tolist()))

    print("\nDone. Outputs in:", OUTDIR)



def composition_pearson_only():
    global os, gzip, warnings, np, pd, plt, sns, Counter, \
        mmread, issparse, pearsonr, spearmanr, minimize_scalar, StratifiedKFold, OUTDIR, COUNTS_PATH, \
        GENES_PATH, CLONE_PATH, META_PATH, TIME_COL, CELLTYPE_COL, START_COL, WELL_COL, EARLY_TIME, \
        EARLY_CELLTYPE, EARLY_WELL, RESTRICT_STARTING_POPULATION, TERMINAL_TIME, TERMINAL_WELL, EXCLUDE_FATES, MANUAL_SELECTED_FATES, MAX_FATES, \
        MIN_CLONES_WITH_FATE, MIN_TERMINAL_CELLS_PER_CLONE, MIN_EARLY_CELLS_PER_CLONE, MIN_TOTAL_CELLS_PER_CLONE, MIN_SELECTED_FATE_COVERAGE, MIN_SELECTED_TERMINAL_CELLS, N_VAR_GENES, MAX_COV_CELLS, \
        RIDGE, COV_SHRINK_TO_DIAG, USE_FATE_PRIOR, CALIBRATE_TEMPERATURE, TEMP_MIN, TEMP_MAX, N_NULLS, USE_STARTPOP_PRESERVING_NULL, \
        N_SPLITS, SEED, rng, safe_name, softmax_logits, safe_corr, get_cell_to_clone, get_cells_x_genes, \
        zscore_train, select_hvgs_sparse, make_covariance, weighted_mean, clone_mean_matrix, shuffle_rows_within_groups, fit_temperature_from_counts, make_composition_cipher_model, \
        get_logits, score_composition_cipher, fit_startpop_composition_baseline, score_startpop_composition_baseline, summarize_per_fate_pearson, counts, f, gene_names, \
        clone_mat, meta, cell_to_clone, has_clone, cell_fates, early_mask, terminal_mask, early_all_idx, \
        early_cloned_mask, terminal_cloned_mask, candidate_records, global_fate_counts, global_fate_clone_counts, clone_id, cells, early_cells, \
        terminal_cells, fates, vc, terminal_counts_dict, c, starts, dominant_start, dominant_start_frac, \
        candidate_table, fate_summary, selected_fates, clone_table, fate, s, selected_count_cols, obs_frac_cols, \
        Y_all, dominant_idx, hvg_idx, gene_vars, hvg_genes, cov_idx, Xcov_raw, mu_ref, \
        sd_ref, Xcov, Sigma, evals, evecs, X_clones_all, strat_y, min_class_n, \
        n_splits, splitter, clone_to_obs, clone_to_counts, clone_to_start, metric_rows, fold, train_pos, \
        test_pos, train_clones, test_clones, Xtrain, train_clone_ids_used, n_train_early, Xtest, test_clone_ids_used, \
        n_test_early, Ytrain, Ctrain, Ytest, start_train, start_test, cipher_model, _, \
        train_logits, T_cipher, P_cipher, sp_model, P_sp, null_id, Ytrain_null, Ctrain_null, \
        null_name, perm, null_model, null_train_logits, T_null, P_null, perf, model_label_map, \
        model_order, palette, preferred_fate_order, fate_order, ax, point_df, tick, handles, \
        labels, seen, uniq_handles, uniq_labels, h, l, png_path, svg_path, \
        pdf_path, summary
    # ============================================================
    # SELF-CONTAINED CIPHER-LARRY composition analysis
    # ONLY OUTPUT:
    #   Predicted vs observed terminal fate fraction: pearson
    #
    # Models:
    #   1. CIPHER
    #   2. starting-pop only
    #   3. startpop-preserving null
    #
    # Output files:
    #   OUTDIR/per_fate_fraction_pearson_only.png
    #   OUTDIR/per_fate_fraction_pearson_only.svg
    #   OUTDIR/per_fate_fraction_pearson_only.pdf
    #   OUTDIR/per_fate_fraction_pearson_metrics.csv
    # ============================================================

    import os
    import gzip
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from collections import Counter
    from scipy.io import mmread
    from scipy.sparse import issparse
    from scipy.stats import pearsonr, spearmanr
    from scipy.optimize import minimize_scalar
    from sklearn.model_selection import StratifiedKFold

    warnings.filterwarnings("ignore")

    # ============================================================
    # CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_pearson_only_selfcontained")
    os.makedirs(OUTDIR, exist_ok=True)

    COUNTS_PATH = os.path.join(SUPPL, "stateFate_inVitro_normed_counts.mtx.gz")
    GENES_PATH  = os.path.join(SUPPL, "stateFate_inVitro_gene_names.txt.gz")
    CLONE_PATH  = os.path.join(SUPPL, "stateFate_inVitro_clone_matrix.mtx.gz")
    META_PATH   = os.path.join(SUPPL, "stateFate_inVitro_metadata.txt.gz")

    TIME_COL = "Time point"
    CELLTYPE_COL = "Cell type annotation"
    START_COL = "Starting population"
    WELL_COL = "Well"

    EARLY_TIME = 4.0
    EARLY_CELLTYPE = "Undifferentiated"
    EARLY_WELL = None

    RESTRICT_STARTING_POPULATION = None

    TERMINAL_TIME = 6.0
    TERMINAL_WELL = None

    EXCLUDE_FATES = {
        "Undifferentiated", "Unknown", "unknown", "nan", "NaN",
        "Ambiguous", "ambiguous", "None", ""
    }

    MANUAL_SELECTED_FATES = None
    # Example:
    # MANUAL_SELECTED_FATES = ["Monocyte", "Neutrophil", "Baso", "Mast", "Meg"]

    MAX_FATES = 5
    MIN_CLONES_WITH_FATE = 8
    MIN_TERMINAL_CELLS_PER_CLONE = 5
    MIN_EARLY_CELLS_PER_CLONE = 1
    MIN_TOTAL_CELLS_PER_CLONE = 8

    MIN_SELECTED_FATE_COVERAGE = 0.75
    MIN_SELECTED_TERMINAL_CELLS = 5

    N_VAR_GENES = 500
    MAX_COV_CELLS = 50000

    RIDGE = 1e-8
    COV_SHRINK_TO_DIAG = 0.0

    USE_FATE_PRIOR = False

    CALIBRATE_TEMPERATURE = True
    TEMP_MIN = 0.1
    TEMP_MAX = 100.0

    N_NULLS = 100
    USE_STARTPOP_PRESERVING_NULL = True

    N_SPLITS = 5
    SEED = 0

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    sns.set_context("talk")
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 16,
    })

    # ============================================================
    # HELPERS
    # ============================================================


    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )

    def softmax_logits(logits, temperature=1.0, eps=1e-12):
        logits = np.asarray(logits, dtype=float) / max(float(temperature), eps)
        z = logits - np.max(logits, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.maximum(ez.sum(axis=1, keepdims=True), eps)


    def safe_corr(x, y, method="pearson"):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        try:
            if method == "pearson":
                return pearsonr(x, y)[0]
            if method == "spearman":
                return spearmanr(x, y)[0]
        except Exception:
            return np.nan
        return np.nan

    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()
        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row
        return cell_to_clone

    def get_cells_x_genes(counts, cell_idx, gene_idx):
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)

    def zscore_train(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu, sd


    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        X = counts[:, cell_idx]
        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means**2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]

        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_var_genes:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_

    def make_covariance(X):
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D
        Sigma = Sigma + RIDGE * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)

    def weighted_mean(X, w, eps=1e-12):
        w = np.asarray(w, dtype=float)
        return (w[:, None] * X).sum(axis=0) / max(w.sum(), eps)

    def clone_mean_matrix(clone_ids, early_mask, cell_to_clone, counts, hvg_idx, mu, sd):
        rows = []
        out_ids = []
        out_n = []

        for cid in clone_ids:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]
            if len(idx) == 0:
                continue

            X = get_cells_x_genes(counts, idx, hvg_idx)
            X = apply_zscore(X, mu, sd)

            rows.append(X.mean(axis=0))
            out_ids.append(cid)
            out_n.append(len(idx))

        if len(rows) == 0:
            return (
                np.empty((0, len(hvg_idx))),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        return np.vstack(rows), np.asarray(out_ids, dtype=int), np.asarray(out_n, dtype=int)

    def shuffle_rows_within_groups(Y, groups):
        Y = np.asarray(Y).copy()
        groups = np.asarray(groups).astype(str)
        out = Y.copy()

        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) > 1:
                out[idx] = out[rng.permutation(idx)]

        return out

    def fit_temperature_from_counts(logits, counts, temp_min=0.1, temp_max=100.0):
        logits = np.asarray(logits, dtype=float)
        counts = np.asarray(counts, dtype=float)

        if logits.shape != counts.shape:
            raise ValueError("logits and counts must have same shape")

        if counts.sum() <= 0:
            return 1.0

        def nll_logT(logT):
            T = float(np.exp(logT))
            P = softmax_logits(logits, temperature=T)
            return -float(np.sum(counts * np.log(np.clip(P, 1e-12, 1.0))))

        res = minimize_scalar(
            nll_logT,
            bounds=(np.log(temp_min), np.log(temp_max)),
            method="bounded",
            options={"xatol": 1e-4},
        )

        if not res.success or not np.isfinite(res.fun):
            return 1.0

        return float(np.exp(res.x))

    # ============================================================
    # MODEL FUNCTIONS
    # ============================================================

    def make_composition_cipher_model(
        Xtrain_clone,
        Ytrain,
        selected_fates,
        evals,
        evecs,
        Sigma,
        use_fate_prior=False,
    ):
        U = []
        DELTAS = []
        penalties = []
        log_priors = []

        eps = 1e-12
        Ytrain = np.asarray(Ytrain, dtype=float)

        for j, fate in enumerate(selected_fates):
            w_pos = Ytrain[:, j].copy()
            w_neg = 1.0 - w_pos

            if w_pos.sum() < eps or w_neg.sum() < eps:
                delta = np.zeros(Xtrain_clone.shape[1])
            else:
                mu_pos = weighted_mean(Xtrain_clone, w_pos)
                mu_neg = weighted_mean(Xtrain_clone, w_neg)
                delta = mu_pos - mu_neg

            u = evecs @ ((evecs.T @ delta) / evals)
            penalty = 0.5 * float(u @ Sigma @ u)

            if use_fate_prior:
                prior = max(float(Ytrain[:, j].mean()), eps)
                log_prior = np.log(prior)
            else:
                log_prior = 0.0

            U.append(u)
            DELTAS.append(delta)
            penalties.append(penalty)
            log_priors.append(log_prior)

        return {
            "U": np.asarray(U),
            "DELTAS": np.asarray(DELTAS),
            "penalty": np.asarray(penalties),
            "log_prior": np.asarray(log_priors),
            "temperature": 1.0,
        }

    def get_logits(X, model):
        U = model["U"]
        raw_scores = X @ U.T
        logits = raw_scores - model["penalty"][None, :] + model["log_prior"][None, :]
        return raw_scores, logits

    def score_composition_cipher(X, model, temperature=None):
        raw_scores, logits = get_logits(X, model)

        if temperature is None:
            temperature = model.get("temperature", 1.0)

        P = softmax_logits(logits, temperature=temperature)
        return raw_scores, logits, P

    def fit_startpop_composition_baseline(Ytrain, start_train, alpha=2.0):
        Ytrain = np.asarray(Ytrain, dtype=float)
        start_train = np.asarray(start_train).astype(str)

        global_p = Ytrain.mean(axis=0)
        global_p = global_p / global_p.sum()

        table = {}
        for s in np.unique(start_train):
            idx = np.where(start_train == s)[0]
            if len(idx) == 0:
                continue
            p = (Ytrain[idx].sum(axis=0) + alpha * global_p) / (len(idx) + alpha)
            p = p / p.sum()
            table[s] = p

        return {
            "global_p": global_p,
            "table": table,
        }

    def score_startpop_composition_baseline(start_test, model):
        start_test = np.asarray(start_test).astype(str)
        P = []
        for s in start_test:
            P.append(model["table"].get(s, model["global_p"]))
        P = np.vstack(P)
        logits = np.log(np.clip(P, 1e-12, 1.0))
        raw_scores = logits.copy()
        return raw_scores, logits, P

    def summarize_per_fate_pearson(model_name, fold, Ytrue, Ppred, selected_fates, null_id=None):
        rows = []
        for j, fate in enumerate(selected_fates):
            rows.append({
                "model": model_name,
                "fold": fold,
                "null_id": null_id,
                "fate": fate,
                "pearson": safe_corr(Ytrue[:, j], Ppred[:, j], method="pearson"),
                "spearman": safe_corr(Ytrue[:, j], Ppred[:, j], method="spearman"),
                "n_clones": Ytrue.shape[0],
            })
        return rows

    # ============================================================
    # LOAD DATA
    # ============================================================

    counts = mmread(COUNTS_PATH).T.tocsr()
    print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} cells | nnz={counts.nnz:,}")

    with gzip.open(GENES_PATH, "rt") as f:
        gene_names = np.array([line.strip() for line in f])
    print(f"Genes loaded: {len(gene_names)}")

    clone_mat = mmread(CLONE_PATH).T.tocsr()
    print(f"Clone matrix: {clone_mat.shape[0]} clones x {clone_mat.shape[1]} cells")
    print(f"% cells with clone label: {(clone_mat.sum(axis=0) > 0).mean() * 100:.2f}%")

    meta = pd.read_csv(META_PATH, sep="\t")
    meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")

    print(f"Meta: {meta.shape[0]} rows x {meta.shape[1]} cols")
    print("Meta columns:", list(meta.columns))

    assert counts.shape[1] == meta.shape[0] == clone_mat.shape[1], "cells mismatch"
    assert counts.shape[0] == len(gene_names), "genes mismatch"

    print("\nTimepoints:")
    print(np.sort(meta[TIME_COL].dropna().unique()))

    print("\nCell annotations:")
    print(meta[CELLTYPE_COL].value_counts())

    cell_to_clone = get_cell_to_clone(clone_mat)
    has_clone = cell_to_clone >= 0
    cell_fates = meta[CELLTYPE_COL].astype(str).values

    # ============================================================
    # DEFINE EARLY / TERMINAL MASKS
    # ============================================================

    early_mask = meta[TIME_COL].astype(float).values == float(EARLY_TIME)

    if EARLY_CELLTYPE is not None:
        early_mask &= meta[CELLTYPE_COL].astype(str).values == str(EARLY_CELLTYPE)

    if EARLY_WELL is not None and WELL_COL in meta.columns:
        early_mask &= meta[WELL_COL].astype(float).values == float(EARLY_WELL)

    if RESTRICT_STARTING_POPULATION is not None and START_COL in meta.columns:
        early_mask &= meta[START_COL].astype(str).values == str(RESTRICT_STARTING_POPULATION)

    terminal_mask = meta[TIME_COL].astype(float).values == float(TERMINAL_TIME)

    if TERMINAL_WELL is not None and WELL_COL in meta.columns:
        terminal_mask &= meta[WELL_COL].astype(float).values == float(TERMINAL_WELL)

    terminal_mask &= ~np.isin(cell_fates, list(EXCLUDE_FATES))

    early_all_idx = np.where(early_mask)[0]
    early_cloned_mask = early_mask & has_clone
    terminal_cloned_mask = terminal_mask & has_clone

    print(f"\nAll early cells for Sigma: {len(early_all_idx):,}")
    print(f"Cloned early cells: {early_cloned_mask.sum():,}")
    print(f"Cloned terminal cells: {terminal_cloned_mask.sum():,}")

    if len(early_all_idx) == 0:
        raise RuntimeError("No early cells found.")

    if terminal_cloned_mask.sum() == 0:
        raise RuntimeError("No terminal cloned cells found.")

    # ============================================================
    # BUILD CLONE TABLE WITH TERMINAL COMPOSITION
    # ============================================================

    candidate_records = []
    global_fate_counts = Counter()
    global_fate_clone_counts = Counter()

    for clone_id in range(clone_mat.shape[0]):
        cells = clone_mat[clone_id].indices

        if len(cells) < MIN_TOTAL_CELLS_PER_CLONE:
            continue

        early_cells = cells[early_cloned_mask[cells]]
        terminal_cells = cells[terminal_cloned_mask[cells]]

        if len(early_cells) < MIN_EARLY_CELLS_PER_CLONE:
            continue
        if len(terminal_cells) < MIN_TERMINAL_CELLS_PER_CLONE:
            continue

        fates = pd.Series(cell_fates[terminal_cells].astype(str))
        fates = fates[~fates.isin(EXCLUDE_FATES)]

        if len(fates) < MIN_TERMINAL_CELLS_PER_CLONE:
            continue

        vc = fates.value_counts()
        terminal_counts_dict = {str(k): int(v) for k, v in vc.items()}

        for f, c in terminal_counts_dict.items():
            global_fate_counts[f] += c
            if c > 0:
                global_fate_clone_counts[f] += 1

        if START_COL in meta.columns:
            starts = meta.iloc[early_cells][START_COL].astype(str).value_counts()
            dominant_start = starts.index[0]
            dominant_start_frac = float(starts.iloc[0] / starts.sum())
        else:
            dominant_start = "unknown"
            dominant_start_frac = 1.0

        candidate_records.append({
            "clone_id": int(clone_id),
            "n_total_clone_cells": int(len(cells)),
            "n_early": int(len(early_cells)),
            "n_terminal": int(len(fates)),
            "terminal_counts_dict": terminal_counts_dict,
            "dominant_starting_population": dominant_start,
            "dominant_starting_population_frac": dominant_start_frac,
        })

    candidate_table = pd.DataFrame(candidate_records)

    if candidate_table.empty:
        raise RuntimeError("No clones passed initial early/terminal filters.")

    fate_summary = pd.DataFrame({
        "fate": list(global_fate_counts.keys()),
        "terminal_cell_count": [global_fate_counts[f] for f in global_fate_counts.keys()],
        "clone_count_with_fate": [global_fate_clone_counts[f] for f in global_fate_counts.keys()],
    }).sort_values("terminal_cell_count", ascending=False)

    if MANUAL_SELECTED_FATES is None:
        selected_fates = (
            fate_summary[fate_summary["clone_count_with_fate"] >= MIN_CLONES_WITH_FATE]
            .head(MAX_FATES)["fate"]
            .tolist()
        )
    else:
        selected_fates = list(MANUAL_SELECTED_FATES)

    if len(selected_fates) < 2:
        raise RuntimeError("Fewer than two selected fates.")

    print("\nSelected fates for composition:")
    print(selected_fates)

    clone_table = candidate_table.copy()

    for fate in selected_fates:
        s = safe_name(fate)
        clone_table[f"terminal_count__{s}"] = clone_table["terminal_counts_dict"].apply(
            lambda d: int(d.get(fate, 0))
        )

    selected_count_cols = [f"terminal_count__{safe_name(f)}" for f in selected_fates]
    clone_table["n_terminal_selected"] = clone_table[selected_count_cols].sum(axis=1)
    clone_table["selected_fate_coverage"] = clone_table["n_terminal_selected"] / clone_table["n_terminal"]

    clone_table = clone_table[
        (clone_table["n_terminal_selected"] >= MIN_SELECTED_TERMINAL_CELLS) &
        (clone_table["selected_fate_coverage"] >= MIN_SELECTED_FATE_COVERAGE)
    ].copy()

    if clone_table.empty:
        raise RuntimeError("No clones passed selected fate coverage filtering.")

    for fate in selected_fates:
        s = safe_name(fate)
        clone_table[f"obs_frac__{s}"] = (
            clone_table[f"terminal_count__{s}"] / clone_table["n_terminal_selected"]
        )

    obs_frac_cols = [f"obs_frac__{safe_name(f)}" for f in selected_fates]

    Y_all = clone_table[obs_frac_cols].values.astype(float)
    dominant_idx = np.argmax(Y_all, axis=1)
    clone_table["dominant_selected_fate"] = np.array(selected_fates, dtype=object)[dominant_idx]
    clone_table["terminal_entropy_selected"] = entropy(Y_all)

    clone_table = clone_table.reset_index(drop=True)

    print("\nClone table after composition filters:")
    print(f"n clones: {len(clone_table):,}")
    print("Dominant selected fate counts:")
    print(clone_table["dominant_selected_fate"].value_counts())
    print("Mean selected fate coverage:", clone_table["selected_fate_coverage"].mean())

    # ============================================================
    # HVGs + SIGMA
    # ============================================================

    print("\nSelecting HVGs from early cells...")

    hvg_idx, gene_vars = select_hvgs_sparse(
        counts=counts,
        cell_idx=early_all_idx,
        n_var_genes=N_VAR_GENES,
    )

    hvg_genes = gene_names[hvg_idx]

    cov_idx = early_all_idx.copy()
    if len(cov_idx) > MAX_COV_CELLS:
        cov_idx = rng.choice(cov_idx, size=MAX_COV_CELLS, replace=False)

    print(f"Using {len(cov_idx):,} cells for Sigma.")

    Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx)
    mu_ref, sd_ref = zscore_train(Xcov_raw)
    Xcov = apply_zscore(Xcov_raw, mu_ref, sd_ref)

    Sigma = make_covariance(Xcov)

    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-8)

    # ============================================================
    # CROSS-VALIDATED COMPOSITION PREDICTION
    # ============================================================

    X_clones_all = clone_table["clone_id"].values.astype(int)
    strat_y = clone_table["dominant_selected_fate"].values.astype(str)

    min_class_n = clone_table["dominant_selected_fate"].value_counts().min()
    n_splits = int(min(N_SPLITS, min_class_n))

    if n_splits < 2:
        raise RuntimeError(f"Cannot do CV. Smallest dominant fate has only {min_class_n} clones.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    clone_to_obs = {
        int(row["clone_id"]): row[obs_frac_cols].values.astype(float)
        for _, row in clone_table.iterrows()
    }

    clone_to_counts = {
        int(row["clone_id"]): row[selected_count_cols].values.astype(int)
        for _, row in clone_table.iterrows()
    }

    clone_to_start = dict(
        zip(
            clone_table["clone_id"].astype(int),
            clone_table["dominant_starting_population"].astype(str),
        )
    )

    metric_rows = []

    for fold, (train_pos, test_pos) in enumerate(splitter.split(X_clones_all, strat_y)):
        train_clones = X_clones_all[train_pos]
        test_clones = X_clones_all[test_pos]

        print(f"\nFold {fold + 1}/{n_splits}: train={len(train_clones)}, test={len(test_clones)}")

        Xtrain, train_clone_ids_used, n_train_early = clone_mean_matrix(
            clone_ids=train_clones,
            early_mask=early_cloned_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        Xtest, test_clone_ids_used, n_test_early = clone_mean_matrix(
            clone_ids=test_clones,
            early_mask=early_cloned_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu=mu_ref,
            sd=sd_ref,
        )

        Ytrain = np.vstack([clone_to_obs[int(c)] for c in train_clone_ids_used])
        Ctrain = np.vstack([clone_to_counts[int(c)] for c in train_clone_ids_used])

        Ytest = np.vstack([clone_to_obs[int(c)] for c in test_clone_ids_used])

        start_train = np.array([clone_to_start.get(int(c), "unknown") for c in train_clone_ids_used])
        start_test = np.array([clone_to_start.get(int(c), "unknown") for c in test_clone_ids_used])

        # --------------------------
        # CIPHER
        # --------------------------
        cipher_model = make_composition_cipher_model(
            Xtrain_clone=Xtrain,
            Ytrain=Ytrain,
            selected_fates=selected_fates,
            evals=evals,
            evecs=evecs,
            Sigma=Sigma,
            use_fate_prior=USE_FATE_PRIOR,
        )

        _, train_logits = get_logits(Xtrain, cipher_model)

        if CALIBRATE_TEMPERATURE:
            T_cipher = fit_temperature_from_counts(
                logits=train_logits,
                counts=Ctrain,
                temp_min=TEMP_MIN,
                temp_max=TEMP_MAX,
            )
        else:
            T_cipher = 1.0

        _, _, P_cipher = score_composition_cipher(
            Xtest,
            cipher_model,
            temperature=T_cipher,
        )

        metric_rows.extend(
            summarize_per_fate_pearson(
                model_name="cipher",
                fold=fold,
                Ytrue=Ytest,
                Ppred=P_cipher,
                selected_fates=selected_fates,
                null_id=None,
            )
        )

        # --------------------------
        # Starting-pop only
        # --------------------------
        if START_COL in meta.columns and RESTRICT_STARTING_POPULATION is None:
            sp_model = fit_startpop_composition_baseline(Ytrain, start_train)
            _, _, P_sp = score_startpop_composition_baseline(start_test, sp_model)

            metric_rows.extend(
                summarize_per_fate_pearson(
                    model_name="starting_population_only",
                    fold=fold,
                    Ytrue=Ytest,
                    Ppred=P_sp,
                    selected_fates=selected_fates,
                    null_id=None,
                )
            )

        # --------------------------
        # Startpop-preserving nulls
        # --------------------------
        for null_id in range(N_NULLS):
            if (
                USE_STARTPOP_PRESERVING_NULL
                and START_COL in meta.columns
                and RESTRICT_STARTING_POPULATION is None
            ):
                Ytrain_null = shuffle_rows_within_groups(Ytrain, start_train)
                Ctrain_null = shuffle_rows_within_groups(Ctrain, start_train)
                null_name = "startpop_preserving_null"
            else:
                perm = rng.permutation(Ytrain.shape[0])
                Ytrain_null = Ytrain[perm]
                Ctrain_null = Ctrain[perm]
                null_name = "shuffled_null"

            null_model = make_composition_cipher_model(
                Xtrain_clone=Xtrain,
                Ytrain=Ytrain_null,
                selected_fates=selected_fates,
                evals=evals,
                evecs=evecs,
                Sigma=Sigma,
                use_fate_prior=USE_FATE_PRIOR,
            )

            _, null_train_logits = get_logits(Xtrain, null_model)

            if CALIBRATE_TEMPERATURE:
                T_null = fit_temperature_from_counts(
                    logits=null_train_logits,
                    counts=Ctrain_null,
                    temp_min=TEMP_MIN,
                    temp_max=TEMP_MAX,
                )
            else:
                T_null = 1.0

            _, _, P_null = score_composition_cipher(
                Xtest,
                null_model,
                temperature=T_null,
            )

            metric_rows.extend(
                summarize_per_fate_pearson(
                    model_name=null_name,
                    fold=fold,
                    Ytrue=Ytest,
                    Ppred=P_null,
                    selected_fates=selected_fates,
                    null_id=null_id,
                )
            )

    # ============================================================
    # PLOT ONLY THE PEARSON FIGURE
    # ============================================================

    perf = pd.DataFrame(metric_rows)

    perf.to_csv(os.path.join(OUTDIR, "per_fate_fraction_pearson_metrics.csv"), index=False)

    model_label_map = {
        "cipher": "CIPHER",
        "starting_population_only": "starting-pop only",
        "startpop_preserving_null": "startpop-preserving null",
        "shuffled_null": "shuffled null",
    }

    perf["model_label"] = perf["model"].map(model_label_map).fillna(perf["model"])

    model_order = [
        "CIPHER",
        "starting-pop only",
        "startpop-preserving null",
    ]

    palette = {
        "CIPHER": sns.color_palette("tab10")[0],
        "starting-pop only": sns.color_palette("tab10")[1],
        "startpop-preserving null": sns.color_palette("tab10")[2],
    }

    preferred_fate_order = ["Monocyte", "Neutrophil", "Baso", "Mast", "Meg"]
    fate_order = [f for f in preferred_fate_order if f in selected_fates]
    fate_order += [f for f in selected_fates if f not in fate_order]

    plt.figure(figsize=(12, 5.5))

    ax = sns.boxplot(
        data=perf,
        x="fate",
        y="pearson",
        hue="model_label",
        order=fate_order,
        hue_order=model_order,
        palette=palette,
        showfliers=False,
        linewidth=1.5,
    )

    # Only overlay points for CIPHER and starting-pop only.
    # Do NOT overlay all null points; the null box already shows the null distribution.
    point_df = perf[perf["model"].isin(["cipher", "starting_population_only"])].copy()

    try:
        sns.stripplot(
            data=point_df,
            x="fate",
            y="pearson",
            hue="model_label",
            order=fate_order,
            hue_order=model_order,
            dodge=True,
            color="black",
            alpha=0.6,
            size=4,
            jitter=0.12,
            legend=False,
            ax=ax,
        )
    except TypeError:
        sns.stripplot(
            data=point_df,
            x="fate",
            y="pearson",
            hue="model_label",
            order=fate_order,
            hue_order=model_order,
            dodge=True,
            color="black",
            alpha=0.6,
            size=4,
            jitter=0.12,
            ax=ax,
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=2)
    ax.set_ylim(-1, 1)

    ax.set_title("Predicted vs observed terminal fate fraction: pearson")
    ax.set_xlabel("terminal fate")
    ax.set_ylabel("pearson")

    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in model_order and l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)

    ax.legend(
        uniq_handles,
        uniq_labels,
        frameon=False,
        bbox_to_anchor=(1.03, 1.0),
        loc="upper left",
    )

    plt.tight_layout()

    png_path = os.path.join(OUTDIR, "per_fate_fraction_pearson_only.png")
    svg_path = os.path.join(OUTDIR, "per_fate_fraction_pearson_only.svg")
    pdf_path = os.path.join(OUTDIR, "per_fate_fraction_pearson_only.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    print("\nSaved:")
    print(" ", png_path)
    print(" ", svg_path)
    print(" ", pdf_path)
    print(" ", os.path.join(OUTDIR, "per_fate_fraction_pearson_metrics.csv"))

    print("\nPearson summary:")
    summary = (
        perf
        .groupby(["fate", "model_label"], as_index=False)
        .agg(
            pearson_mean=("pearson", "mean"),
            pearson_sd=("pearson", "std"),
            n=("pearson", "size"),
        )
    )

    summary["fate"] = pd.Categorical(summary["fate"], categories=fate_order, ordered=True)
    summary["model_label"] = pd.Categorical(summary["model_label"], categories=model_order, ordered=True)
    summary = summary.sort_values(["fate", "model_label"])

    print(summary.to_string(index=False))



def gseapy_list_marker_libraries():
    global gp, get_library_name, get_library, libs
    import gseapy as gp
    from gseapy.parser import get_library_name, get_library

    libs = get_library_name(organism="Mouse")
    [x for x in libs if any(k.lower() in x.lower() for k in [
        "panglao", "cellmarker", "tabula", "mouse", "immune", "hematopo"
    ])]



def gseapy_enrichr_demo():
    global gp, top_genes, enr
    import gseapy as gp

    top_genes = ["Elane", "Prtn3", "Mpo", "Ctsg", "Pf4", "Gata2"]

    enr = gp.enrichr(
        gene_list=top_genes,
        gene_sets=[
            "PanglaoDB_Augmented_2021",
            "CellMarker_Augmented_2021",
        ],
        organism="h. sapiens",
        outdir=None,
    )

    enr.results.sort_values("Adjusted P-value").head(20)



def cipher_u_marker_association():
    global os, re, math, warnings, np, pd, plt, sns, \
        hypergeom, multipletests, CIPHER_OUTDIR, FORCE_PATH, HVG_PATH, OUTDIR, TOP_N_VALUES, MAIN_TOP_N, \
        USE_DIRECTION, FILTER_TO_RELEVANT_TERMS, TERM_RELEVANCE_KEYWORDS, FETCH_GSEAPY_LIBRARIES_IF_NEEDED, GSEAPY_ORGANISM, GSEAPY_LIBRARY_KEYWORDS, safe_name, bh_fdr, \
        hypergeom_enrich, term_is_relevant, flatten_marker_libraries, CANONICAL_MARKERS, FATE_CATEGORY_SYNONYMS, force_df, hvg_df, background_genes, \
        force_pos, agg, selected_fates, top_gene_rows, top_genes_by_fate, fate, sub, top_n, \
        genes, rank, _, row, top_genes_df, loaded_marker_libraries, varname, gp, \
        get_library_name, get_library, all_libs, chosen_libs, lib, marker_terms_df, canonical_rows, term, \
        canonical_df, all_terms_df, canonical_results, query, marker_category, marker_genes, res, cat_order, \
        fate_order, heat_fdr, heat_overlap, db_results, term_row, idx, top_db, display_cols, \
        terms_for_category, category_results, category, term_sub, union_genes, source_terms, n_terms, union, \
        cat_order2, cat_heat, cat_odds, best_cat, canon_gene_to_categories, cat, g, annot_rows, \
        gnorm, cats, db_hits, tr, annot_df, expected_category_for_fate, expected_rows, exp_cat, \
        r, expected_df, fig, axes, fn
    # ============================================================
    # CIPHER u-gene marker association / enrichment analysis
    # ============================================================
    # Inputs:
    #   - composition_CIPHER_force_genes.csv from CIPHER composition run
    #   - selected_early_hvgs.csv from same run, used as background
    #
    # Optional:
    #   - If you already loaded marker libraries, define one of:
    #       marker_libraries = {"LibraryName": {"term": ["Gene1", ...], ...}, ...}
    #       MARKER_LIBRARIES = {"LibraryName": {"term": ["Gene1", ...], ...}, ...}
    #
    # If not already loaded, this tries to fetch mouse Enrichr/GSEApy libraries.
    #
    # Outputs:
    #   - per-fate top CIPHER u genes
    #   - canonical marker overlap/enrichment
    #   - database marker-term enrichment
    #   - fate-by-marker-category enrichment heatmaps
    # ============================================================

    import os
    import re
    import math
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import hypergeom
    from statsmodels.stats.multitest import multipletests

    warnings.filterwarnings("ignore")

    # ----------------------------
    # CONFIG
    # ----------------------------

    CIPHER_OUTDIR = os.path.join(OUT_BASE, "cipher_larry_clone_fate_composition_temperature_calibrated")
    FORCE_PATH = os.path.join(CIPHER_OUTDIR, "composition_CIPHER_force_genes.csv")
    HVG_PATH = os.path.join(CIPHER_OUTDIR, "selected_early_hvgs.csv")

    OUTDIR = os.path.join(CIPHER_OUTDIR, "marker_association_analysis")
    os.makedirs(OUTDIR, exist_ok=True)

    TOP_N_VALUES = [10, 20, 50]
    MAIN_TOP_N = 20

    # Only positive u genes are usually interpretable as "toward fate f"
    USE_DIRECTION = "positive"

    # If True, use only terms that contain broad immune/hematopoietic keywords.
    FILTER_TO_RELEVANT_TERMS = True

    TERM_RELEVANCE_KEYWORDS = [
        "neutrophil", "granulocyte", "monocyte", "macrophage",
        "basophil", "mast", "megakaryocyte", "platelet",
        "erythroid", "erythroblast", "lymphoid", "lymphocyte",
        "b cell", "t cell", "nk cell", "dendritic", "dc",
        "hematopo", "myeloid", "progenitor", "stem cell",
    ]

    # Libraries to try if marker libraries are not already loaded.
    # Exact availability changes with Enrichr/GSEApy, so the code searches names.
    FETCH_GSEAPY_LIBRARIES_IF_NEEDED = True
    GSEAPY_ORGANISM = "Mouse"
    GSEAPY_LIBRARY_KEYWORDS = [
        "Panglao",
        "CellMarker",
        "Tabula",
        "Mouse",
        "Immune",
        "Hematopo",
    ]

    plt.rcParams.update({"font.size": 14})
    sns.set_context("talk")

    # ----------------------------
    # BASIC HELPERS
    # ----------------------------

    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )


    def bh_fdr(pvals):
        pvals = np.asarray(pvals, dtype=float)
        ok = np.isfinite(pvals)
        out = np.full_like(pvals, np.nan, dtype=float)
        if ok.sum() > 0:
            out[ok] = multipletests(pvals[ok], method="fdr_bh")[1]
        return out

    def hypergeom_enrich(query_genes, marker_genes, background_genes):
        """
        One-sided overlap enrichment:
          N = background size
          K = marker genes in background
          n = query genes in background
          k = overlap
          p = P[X >= k]
        """
        bg = set(map(norm_gene, background_genes))
        q = set(map(norm_gene, query_genes)) & bg
        m = set(map(norm_gene, marker_genes)) & bg

        N = len(bg)
        K = len(m)
        n = len(q)
        k = len(q & m)

        if N == 0 or K == 0 or n == 0 or k == 0:
            p = 1.0
        else:
            p = hypergeom.sf(k - 1, N, K, n)

        expected = (n * K / N) if N > 0 else np.nan
        odds = (k / expected) if expected and expected > 0 else np.nan

        return {
            "N_background": N,
            "K_marker_in_background": K,
            "n_query_in_background": n,
            "k_overlap": k,
            "expected_overlap": expected,
            "odds_enrichment": odds,
            "p_value": p,
            "overlap_genes_norm": sorted(q & m),
        }

    def term_is_relevant(term):
        t = clean_term(term).lower()
        return any(k.lower() in t for k in TERM_RELEVANCE_KEYWORDS)

    def flatten_marker_libraries(marker_libraries):
        """
        Converts:
          {"Lib": {"term": [genes]}}
        into a dataframe with library, term, genes.
        """
        rows = []
        for lib, terms in marker_libraries.items():
            if terms is None:
                continue

            # gseapy get_library can return dict(term -> list genes)
            if isinstance(terms, dict):
                iterator = terms.items()
            else:
                continue

            for term, genes in iterator:
                genes = [str(g).strip() for g in genes if str(g).strip()]
                if len(genes) == 0:
                    continue
                rows.append({
                    "library": lib,
                    "term": str(term),
                    "term_clean": clean_term(term),
                    "genes": genes,
                    "genes_norm": sorted(set(map(norm_gene, genes))),
                    "n_genes_raw": len(set(map(norm_gene, genes))),
                })

        df = pd.DataFrame(rows)
        if FILTER_TO_RELEVANT_TERMS and len(df) > 0:
            df = df[df["term_clean"].apply(term_is_relevant)].copy()
        return df.reset_index(drop=True)

    # ----------------------------
    # CANONICAL MARKER SETS
    # ----------------------------

    CANONICAL_MARKERS = {
        "Neutrophil": [
            "Elane", "Prtn3", "Mpo", "Ctsg", "Ngp", "Lcn2", "S100a8", "S100a9",
            "Camp", "Ltf", "Mmp8", "Retnlg", "Cebpe", "Ly6g", "Cxcr2", "Fcgr3"
        ],
        "Monocyte": [
            "Lyz2", "Csf1r", "Ctss", "Ctsb", "Lgals3", "Mpeg1", "Ccr2", "Ly6c2",
            "Itgam", "Cd14", "Fcgr1", "Sirpa", "Cybb", "Ms4a7", "Plac8", "S100a4"
        ],
        "Baso": [
            "Gata2", "Cpa3", "Prg2", "Hdc", "Ms4a2", "Fcer1a", "Alox5", "Srgn",
            "Ccr3", "Il4", "Il13", "Mcpt8", "Csf2rb", "Csf2rb2", "Kit"
        ],
        "Mast": [
            "Cma1", "Cpa3", "Tpsb2", "Tpsab1", "Hdc", "Kit", "Ms4a2", "Fcer1a",
            "Gata2", "Mcpt4", "Mcpt8", "Srgn", "Alox5"
        ],
        "Meg": [
            "Pf4", "Ppbp", "Itga2b", "Itgb3", "Gp9", "Gp1ba", "Gp1bb", "Nfe2",
            "Mpl", "Vwf", "Rap1b", "Tubb1", "Fli1", "Gata1", "Pbx1"
        ],
        "Erythroid": [
            "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2", "Alas2", "Klf1", "Gata1",
            "Nfe2", "Tfrc", "Car2", "Epor", "Hemgn", "Ermap"
        ],
        "Lymphoid": [
            "Cd3d", "Cd3e", "Cd3g", "Trac", "Ms4a1", "Cd79a", "Cd79b", "Nkg7",
            "Gzma", "Gzmb", "Il7r", "Dntt", "Rag1", "Rag2", "Ly6d"
        ],
        "Dendritic": [
            "Itgax", "Flt3", "Ccr7", "H2-Aa", "H2-Ab1", "Cd74", "Clec9a",
            "Siglech", "Irf8", "Batf3", "Zbtb46", "Xcr1"
        ],
    }

    FATE_CATEGORY_SYNONYMS = {
        "Neutrophil": ["neutrophil", "granulocyte", "pmn", "polymorphonuclear"],
        "Monocyte": ["monocyte", "macrophage"],
        "Baso": ["basophil", "baso"],
        "Mast": ["mast"],
        "Meg": ["megakaryocyte", "platelet", "thrombocyte"],
        "Erythroid": ["erythroid", "erythroblast", "red blood"],
        "Lymphoid": ["lymphoid", "lymphocyte", "b cell", "t cell", "nk cell"],
        "Dendritic": ["dendritic", "dc", "pdc", "cdc"],
        "HSPC": ["hematopoietic stem", "hsc", "progenitor", "stem cell", "lsk"],
    }

    # ----------------------------
    # LOAD CIPHER U GENES
    # ----------------------------

    force_df = pd.read_csv(FORCE_PATH)
    print("Loaded force_df:", force_df.shape)
    print("Columns:", list(force_df.columns))

    if os.path.exists(HVG_PATH):
        hvg_df = pd.read_csv(HVG_PATH)
        background_genes = hvg_df["gene"].astype(str).tolist()
        print(f"Background: {len(background_genes)} HVGs from {HVG_PATH}")
    else:
        background_genes = force_df["gene"].astype(str).unique().tolist()
        print(f"Background: {len(background_genes)} genes from force table only")

    force_pos = force_df[force_df["direction"].astype(str).str.lower() == USE_DIRECTION.lower()].copy()

    # Aggregate across folds. Since the file usually stores top 50 per fold, frequency is useful.
    agg = (
        force_pos
        .groupby(["fate", "gene"], as_index=False)
        .agg(
            mean_u=("u", "mean"),
            median_u=("u", "median"),
            max_u=("u", "max"),
            mean_rank=("rank", "mean"),
            min_rank=("rank", "min"),
            n_folds_present=("fold", "nunique"),
            mean_delta=("delta_weighted_composition", "mean")
                if "delta_weighted_composition" in force_pos.columns else ("u", "mean"),
        )
    )

    agg["gene_norm"] = agg["gene"].map(norm_gene)
    agg = agg.sort_values(["fate", "mean_u"], ascending=[True, False]).reset_index(drop=True)

    agg.to_csv(os.path.join(OUTDIR, "aggregated_positive_CIPHER_u_genes.csv"), index=False)

    selected_fates = agg["fate"].drop_duplicates().tolist()
    print("\nFates found:")
    print(selected_fates)

    top_gene_rows = []
    top_genes_by_fate = {}

    for fate in selected_fates:
        sub = agg[agg["fate"] == fate].sort_values("mean_u", ascending=False).copy()
        top_genes_by_fate[fate] = {}

        for top_n in TOP_N_VALUES:
            genes = sub.head(top_n)["gene"].astype(str).tolist()
            top_genes_by_fate[fate][top_n] = genes

            for rank, (_, row) in enumerate(sub.head(top_n).iterrows(), start=1):
                top_gene_rows.append({
                    "fate": fate,
                    "top_n": top_n,
                    "rank": rank,
                    "gene": row["gene"],
                    "mean_u": row["mean_u"],
                    "mean_rank_across_folds": row["mean_rank"],
                    "n_folds_present": row["n_folds_present"],
                })

    top_genes_df = pd.DataFrame(top_gene_rows)
    top_genes_df.to_csv(os.path.join(OUTDIR, "top_CIPHER_u_genes_by_fate.csv"), index=False)

    print("\nTop positive CIPHER u genes:")
    for fate in selected_fates:
        print(f"\n{fate}")
        print(", ".join(top_genes_by_fate[fate][MAIN_TOP_N]))

    # ----------------------------
    # LOAD OR FETCH MARKER LIBRARIES
    # ----------------------------

    loaded_marker_libraries = None

    for varname in ["marker_libraries", "MARKER_LIBRARIES", "marker_sets_by_library", "MARKER_SETS_BY_LIBRARY"]:
        if varname in globals():
            loaded_marker_libraries = globals()[varname]
            print(f"\nUsing existing marker library variable: {varname}")
            break

    if loaded_marker_libraries is None and FETCH_GSEAPY_LIBRARIES_IF_NEEDED:
        try:
            import gseapy as gp
            from gseapy.parser import get_library_name, get_library

            print("\nNo marker_libraries variable found. Trying to fetch GSEApy/Enrichr libraries...")

            all_libs = get_library_name(organism=GSEAPY_ORGANISM)
            chosen_libs = [
                lib for lib in all_libs
                if any(k.lower() in lib.lower() for k in GSEAPY_LIBRARY_KEYWORDS)
            ]

            print("Candidate libraries:")
            for lib in chosen_libs:
                print("  ", lib)

            loaded_marker_libraries = {}
            for lib in chosen_libs:
                try:
                    loaded_marker_libraries[lib] = get_library(name=lib, organism=GSEAPY_ORGANISM)
                    print(f"Loaded {lib}: {len(loaded_marker_libraries[lib])} terms")
                except Exception as e:
                    print(f"[skip] Could not load {lib}: {e}")

        except Exception as e:
            print("\nCould not fetch GSEApy libraries.")
            print("Reason:", repr(e))
            loaded_marker_libraries = {}

    if loaded_marker_libraries is None:
        loaded_marker_libraries = {}

    marker_terms_df = flatten_marker_libraries(loaded_marker_libraries)

    # Add canonical marker sets as an internal benchmark library.
    canonical_rows = []
    for term, genes in CANONICAL_MARKERS.items():
        canonical_rows.append({
            "library": "canonical_manual_markers",
            "term": term,
            "term_clean": term,
            "genes": genes,
            "genes_norm": sorted(set(map(norm_gene, genes))),
            "n_genes_raw": len(set(map(norm_gene, genes))),
        })

    canonical_df = pd.DataFrame(canonical_rows)

    all_terms_df = pd.concat([canonical_df, marker_terms_df], ignore_index=True)

    if len(all_terms_df) == 0:
        raise RuntimeError(
            "No marker terms loaded. Define marker_libraries = {'lib': {'term': [genes]}} "
            "or set FETCH_GSEAPY_LIBRARIES_IF_NEEDED=True with internet access."
        )

    all_terms_df["n_genes_in_background"] = all_terms_df["genes_norm"].apply(
        lambda gs: len(set(gs) & set(map(norm_gene, background_genes)))
    )

    all_terms_df = all_terms_df[all_terms_df["n_genes_in_background"] > 0].copy()
    all_terms_df.to_csv(os.path.join(OUTDIR, "all_marker_terms_used.csv"), index=False)

    print(f"\nMarker terms used: {len(all_terms_df)}")
    print(all_terms_df["library"].value_counts().head(20))

    # ----------------------------
    # 1) CANONICAL MARKER OVERLAP / ENRICHMENT
    # ----------------------------

    canonical_results = []

    for fate in selected_fates:
        query = top_genes_by_fate[fate][MAIN_TOP_N]

        for marker_category, marker_genes in CANONICAL_MARKERS.items():
            res = hypergeom_enrich(query, marker_genes, background_genes)

            canonical_results.append({
                "cipher_fate": fate,
                "marker_category": marker_category,
                "top_n": MAIN_TOP_N,
                **{k: v for k, v in res.items() if k != "overlap_genes_norm"},
                "overlap_genes": ", ".join(res["overlap_genes_norm"]),
            })

    canonical_results = pd.DataFrame(canonical_results)
    canonical_results["fdr"] = bh_fdr(canonical_results["p_value"].values)
    canonical_results["minus_log10_fdr"] = -np.log10(np.maximum(canonical_results["fdr"].values, 1e-300))
    canonical_results["overlap_fraction_of_query"] = (
        canonical_results["k_overlap"] / canonical_results["n_query_in_background"].replace(0, np.nan)
    )

    canonical_results.to_csv(os.path.join(OUTDIR, "canonical_marker_overlap_enrichment.csv"), index=False)

    print("\nCanonical marker enrichment, best matches:")
    print(
        canonical_results
        .sort_values(["cipher_fate", "p_value"])
        .groupby("cipher_fate")
        .head(3)
        [["cipher_fate", "marker_category", "k_overlap", "odds_enrichment", "p_value", "fdr", "overlap_genes"]]
    )

    # Heatmap: CIPHER fate x canonical marker category
    cat_order = list(CANONICAL_MARKERS.keys())
    fate_order = selected_fates

    heat_fdr = (
        canonical_results
        .pivot_table(index="cipher_fate", columns="marker_category", values="minus_log10_fdr", fill_value=0)
        .reindex(index=fate_order, columns=cat_order)
    )

    heat_overlap = (
        canonical_results
        .pivot_table(index="cipher_fate", columns="marker_category", values="overlap_fraction_of_query", fill_value=0)
        .reindex(index=fate_order, columns=cat_order)
    )

    plt.figure(figsize=(1.2 * len(cat_order) + 4, 0.7 * len(fate_order) + 3))
    sns.heatmap(
        heat_fdr,
        cmap="viridis",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "-log10(FDR)"},
    )
    plt.title(f"CIPHER top {MAIN_TOP_N} u genes vs canonical markers")
    plt.xlabel("canonical marker category")
    plt.ylabel("CIPHER fate force")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "canonical_marker_enrichment_heatmap_minuslog10FDR.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "canonical_marker_enrichment_heatmap_minuslog10FDR.svg"), bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(1.2 * len(cat_order) + 4, 0.7 * len(fate_order) + 3))
    sns.heatmap(
        heat_overlap,
        cmap="mako",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "overlap fraction of top genes"},
    )
    plt.title(f"CIPHER top {MAIN_TOP_N} u genes: canonical marker overlap")
    plt.xlabel("canonical marker category")
    plt.ylabel("CIPHER fate force")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "canonical_marker_overlap_fraction_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "canonical_marker_overlap_fraction_heatmap.svg"), bbox_inches="tight")
    plt.show()

    # ----------------------------
    # 2) DATABASE TERM ENRICHMENT FOR EACH CIPHER FATE
    # ----------------------------

    db_results = []

    for fate in selected_fates:
        for top_n in TOP_N_VALUES:
            query = top_genes_by_fate[fate][top_n]

            for _, term_row in all_terms_df.iterrows():
                res = hypergeom_enrich(query, term_row["genes_norm"], background_genes)

                db_results.append({
                    "cipher_fate": fate,
                    "top_n": top_n,
                    "library": term_row["library"],
                    "term": term_row["term"],
                    "term_clean": term_row["term_clean"],
                    "n_term_genes_raw": term_row["n_genes_raw"],
                    **{k: v for k, v in res.items() if k != "overlap_genes_norm"},
                    "overlap_genes": ", ".join(res["overlap_genes_norm"]),
                })

    db_results = pd.DataFrame(db_results)
    db_results["fdr_within_topn"] = np.nan

    for top_n in sorted(db_results["top_n"].unique()):
        idx = db_results["top_n"] == top_n
        db_results.loc[idx, "fdr_within_topn"] = bh_fdr(db_results.loc[idx, "p_value"].values)

    db_results["minus_log10_fdr"] = -np.log10(np.maximum(db_results["fdr_within_topn"].values, 1e-300))
    db_results["overlap_fraction_of_query"] = (
        db_results["k_overlap"] / db_results["n_query_in_background"].replace(0, np.nan)
    )

    db_results = db_results.sort_values(["cipher_fate", "top_n", "p_value"]).reset_index(drop=True)
    db_results.to_csv(os.path.join(OUTDIR, "all_marker_database_enrichment_results.csv"), index=False)

    top_db = (
        db_results[db_results["top_n"] == MAIN_TOP_N]
        .sort_values(["cipher_fate", "p_value"])
        .groupby("cipher_fate")
        .head(15)
        .copy()
    )
    top_db.to_csv(os.path.join(OUTDIR, f"top_marker_database_terms_top{MAIN_TOP_N}.csv"), index=False)

    print(f"\nTop database marker terms for top {MAIN_TOP_N} CIPHER genes:")
    for fate in selected_fates:
        print(f"\n=== {fate} ===")
        display_cols = ["library", "term_clean", "k_overlap", "odds_enrichment", "p_value", "fdr_within_topn", "overlap_genes"]
        print(top_db[top_db["cipher_fate"] == fate][display_cols].head(10).to_string(index=False))

    # Bar plot of top enriched terms per fate
    for fate in selected_fates:
        sub = (
            db_results[
                (db_results["cipher_fate"] == fate) &
                (db_results["top_n"] == MAIN_TOP_N) &
                (db_results["k_overlap"] > 0)
            ]
            .sort_values("p_value")
            .head(12)
            .copy()
        )

        if len(sub) == 0:
            continue

        sub["plot_label"] = sub["term_clean"].str.slice(0, 70)

        plt.figure(figsize=(10, max(4, 0.45 * len(sub))))
        sns.barplot(
            data=sub,
            y="plot_label",
            x="minus_log10_fdr",
            hue="library",
            dodge=False,
        )
        plt.xlabel("-log10(FDR)")
        plt.ylabel("")
        plt.title(f"{fate}: enriched marker terms among top {MAIN_TOP_N} CIPHER u genes")
        plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"{safe_name(fate)}_top_marker_terms_barplot.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"{safe_name(fate)}_top_marker_terms_barplot.svg"), bbox_inches="tight")
        plt.show()

    # ----------------------------
    # 3) TERM-SYNONYM CATEGORY ASSOCIATION
    # ----------------------------

    def terms_for_category(category, all_terms_df):
        synonyms = FATE_CATEGORY_SYNONYMS.get(category, [category])
        syn_low = [s.lower() for s in synonyms]

        mask = all_terms_df["term_clean"].astype(str).str.lower().apply(
            lambda t: any(s in t for s in syn_low)
        )
        return all_terms_df[mask].copy()

    category_results = []

    for fate in selected_fates:
        query = top_genes_by_fate[fate][MAIN_TOP_N]

        for category in FATE_CATEGORY_SYNONYMS.keys():
            term_sub = terms_for_category(category, all_terms_df)

            # Use union of all marker genes from matching terms.
            if len(term_sub) == 0:
                union_genes = CANONICAL_MARKERS.get(category, [])
                source_terms = "canonical_only_if_available"
                n_terms = 0
            else:
                union = set()
                for genes in term_sub["genes_norm"]:
                    union.update(genes)
                union_genes = sorted(union)
                source_terms = "; ".join(term_sub["term_clean"].head(10).astype(str).tolist())
                n_terms = len(term_sub)

            # Add canonical genes too, so category is not empty if DB naming is weird.
            if category in CANONICAL_MARKERS:
                union_genes = sorted(set(map(norm_gene, union_genes)) | set(map(norm_gene, CANONICAL_MARKERS[category])))

            res = hypergeom_enrich(query, union_genes, background_genes)

            category_results.append({
                "cipher_fate": fate,
                "marker_category": category,
                "top_n": MAIN_TOP_N,
                "n_matching_database_terms": n_terms,
                "example_matching_terms": source_terms,
                **{k: v for k, v in res.items() if k != "overlap_genes_norm"},
                "overlap_genes": ", ".join(res["overlap_genes_norm"]),
            })

    category_results = pd.DataFrame(category_results)
    category_results["fdr"] = bh_fdr(category_results["p_value"].values)
    category_results["minus_log10_fdr"] = -np.log10(np.maximum(category_results["fdr"].values, 1e-300))
    category_results["overlap_fraction_of_query"] = (
        category_results["k_overlap"] / category_results["n_query_in_background"].replace(0, np.nan)
    )

    category_results.to_csv(os.path.join(OUTDIR, "category_level_marker_association.csv"), index=False)

    cat_order2 = list(FATE_CATEGORY_SYNONYMS.keys())

    cat_heat = (
        category_results
        .pivot_table(index="cipher_fate", columns="marker_category", values="minus_log10_fdr", fill_value=0)
        .reindex(index=fate_order, columns=cat_order2)
    )

    cat_odds = (
        category_results
        .pivot_table(index="cipher_fate", columns="marker_category", values="odds_enrichment", fill_value=0)
        .reindex(index=fate_order, columns=cat_order2)
    )

    plt.figure(figsize=(1.15 * len(cat_order2) + 4, 0.7 * len(fate_order) + 3))
    sns.heatmap(
        cat_heat,
        cmap="viridis",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "-log10(FDR)"},
    )
    plt.title(f"CIPHER top {MAIN_TOP_N} u genes vs marker-term categories")
    plt.xlabel("marker-term category")
    plt.ylabel("CIPHER fate force")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "category_marker_association_minuslog10FDR_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "category_marker_association_minuslog10FDR_heatmap.svg"), bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(1.15 * len(cat_order2) + 4, 0.7 * len(fate_order) + 3))
    sns.heatmap(
        np.log2(cat_odds.replace(0, np.nan)),
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "log2 odds enrichment"},
    )
    plt.title(f"CIPHER top {MAIN_TOP_N} u genes: marker category odds")
    plt.xlabel("marker-term category")
    plt.ylabel("CIPHER fate force")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "category_marker_association_log2odds_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "category_marker_association_log2odds_heatmap.svg"), bbox_inches="tight")
    plt.show()

    print("\nBest marker category per CIPHER fate:")
    best_cat = (
        category_results
        .sort_values(["cipher_fate", "p_value"])
        .groupby("cipher_fate")
        .head(3)
    )
    print(
        best_cat[
            ["cipher_fate", "marker_category", "k_overlap", "odds_enrichment", "p_value", "fdr", "overlap_genes"]
        ].to_string(index=False)
    )

    # ----------------------------
    # 4) MARKER HIT TABLE FOR EACH TOP U GENE
    # ----------------------------

    # Annotate every top CIPHER gene by which canonical marker categories it appears in.
    canon_gene_to_categories = {}
    for cat, genes in CANONICAL_MARKERS.items():
        for g in genes:
            canon_gene_to_categories.setdefault(norm_gene(g), []).append(cat)

    annot_rows = []

    for _, row in top_genes_df[top_genes_df["top_n"] == MAIN_TOP_N].iterrows():
        gnorm = norm_gene(row["gene"])
        cats = canon_gene_to_categories.get(gnorm, [])

        # Also list database terms containing this gene among relevant terms.
        db_hits = []
        for _, tr in all_terms_df.iterrows():
            if gnorm in set(tr["genes_norm"]):
                db_hits.append(f"{tr['library']}::{tr['term_clean']}")

        annot_rows.append({
            "cipher_fate": row["fate"],
            "rank": row["rank"],
            "gene": row["gene"],
            "mean_u": row["mean_u"],
            "n_folds_present": row["n_folds_present"],
            "canonical_marker_categories": "; ".join(cats),
            "n_database_marker_terms": len(db_hits),
            "example_database_terms": "; ".join(db_hits[:8]),
        })

    annot_df = pd.DataFrame(annot_rows)
    annot_df.to_csv(os.path.join(OUTDIR, f"top{MAIN_TOP_N}_CIPHER_genes_marker_annotations.csv"), index=False)

    print(f"\nAnnotated top {MAIN_TOP_N} CIPHER genes saved:")
    print(os.path.join(OUTDIR, f"top{MAIN_TOP_N}_CIPHER_genes_marker_annotations.csv"))

    # ----------------------------
    # 5) SIMPLE SUMMARY PLOT: expected category match
    # ----------------------------

    # Map CIPHER fate names to closest canonical categories.
    def expected_category_for_fate(fate):
        f = str(fate).lower()
        if "neut" in f:
            return "Neutrophil"
        if "mono" in f:
            return "Monocyte"
        if "baso" in f:
            return "Baso"
        if "mast" in f:
            return "Mast"
        if "meg" in f:
            return "Meg"
        if "ery" in f:
            return "Erythroid"
        if "lymph" in f:
            return "Lymphoid"
        if "dc" in f or "dend" in f:
            return "Dendritic"
        return None

    expected_rows = []
    for fate in selected_fates:
        exp_cat = expected_category_for_fate(fate)
        if exp_cat is None:
            continue
        sub = category_results[
            (category_results["cipher_fate"] == fate) &
            (category_results["marker_category"] == exp_cat)
        ]
        if len(sub) == 0:
            continue
        r = sub.iloc[0].to_dict()
        expected_rows.append({
            "cipher_fate": fate,
            "expected_marker_category": exp_cat,
            "k_overlap": r["k_overlap"],
            "overlap_fraction_of_query": r["overlap_fraction_of_query"],
            "odds_enrichment": r["odds_enrichment"],
            "minus_log10_fdr": r["minus_log10_fdr"],
            "overlap_genes": r["overlap_genes"],
        })

    expected_df = pd.DataFrame(expected_rows)
    expected_df.to_csv(os.path.join(OUTDIR, "expected_fate_marker_category_hits.csv"), index=False)

    if len(expected_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.barplot(data=expected_df, x="cipher_fate", y="minus_log10_fdr", ax=axes[0])
        axes[0].set_title("Expected marker category significance")
        axes[0].set_ylabel("-log10(FDR)")
        axes[0].set_xlabel("CIPHER fate")
        axes[0].tick_params(axis="x", rotation=45)

        sns.barplot(data=expected_df, x="cipher_fate", y="odds_enrichment", ax=axes[1])
        axes[1].set_title("Expected marker category enrichment")
        axes[1].set_ylabel("odds enrichment")
        axes[1].set_xlabel("CIPHER fate")
        axes[1].tick_params(axis="x", rotation=45)

        sns.barplot(data=expected_df, x="cipher_fate", y="overlap_fraction_of_query", ax=axes[2])
        axes[2].set_title("Top-gene overlap with expected markers")
        axes[2].set_ylabel("overlap fraction")
        axes[2].set_xlabel("CIPHER fate")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "expected_marker_category_summary.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "expected_marker_category_summary.svg"), bbox_inches="tight")
        plt.show()

    # ----------------------------
    # FINAL PRINTS
    # ----------------------------

    print("\n============================================================")
    print("DONE: marker association analysis")
    print("============================================================")
    print("Outputs in:", OUTDIR)

    print("\nKey output files:")
    for fn in [
        "aggregated_positive_CIPHER_u_genes.csv",
        "top_CIPHER_u_genes_by_fate.csv",
        "canonical_marker_overlap_enrichment.csv",
        "all_marker_database_enrichment_results.csv",
        f"top_marker_database_terms_top{MAIN_TOP_N}.csv",
        "category_level_marker_association.csv",
        f"top{MAIN_TOP_N}_CIPHER_genes_marker_annotations.csv",
        "expected_fate_marker_category_hits.csv",
    ]:
        print(" -", os.path.join(OUTDIR, fn))

    print("\nExpected category hits:")
    if len(expected_df) > 0:
        print(expected_df.to_string(index=False))
    else:
        print("No expected fate/category mappings found. Check selected_fates names.")



def composite_marker_validation_top50():
    global os, re, math, warnings, np, pd, plt, sns, \
        hypergeom, multipletests, CIPHER_OUTDIR, FORCE_PATH, HVG_PATH, OUTDIR, MAIN_TOP_N, TOP_N_VALUES, \
        USE_DIRECTION, FILTER_TO_RELEVANT_TERMS, TERM_RELEVANCE_KEYWORDS, FETCH_GSEAPY_LIBRARIES_IF_NEEDED, GSEAPY_ORGANISM, GSEAPY_LIBRARY_KEYWORDS, N_TOP_TERMS_TO_PLOT, safe_name, \
        bh_fdr, term_is_relevant, hypergeom_enrich, flatten_marker_libraries, expected_category_for_fate, terms_for_category, CANONICAL_MARKERS, FATE_CATEGORY_SYNONYMS, \
        force_df, required_cols, missing, hvg_df, background_genes, background_norm, force_pos, agg_dict, \
        agg, selected_fates, top_genes_by_fate, top_gene_rows, fate, sub, top_n, genes, \
        rank, _, row, top_genes_df, top50_union, u_heat, loaded_marker_libraries, varname, \
        gp, get_library_name, get_library, all_libs, chosen_libs, lib, marker_terms_df, canonical_rows, \
        term, canonical_df, all_terms_df, canonical_results, query, marker_category, marker_genes, res, \
        cat_order, fate_order, heat_fdr, heat_overlap, heat_log2odds, db_results, term_row, idx, \
        sub_idx, top_db, display_cols, category_results, category, term_sub, union_genes, source_terms, \
        n_terms, union, cat_order2, cat_heat, cat_odds, cat_overlap, expected_rows, exp_cat, \
        canon_sub, cat_sub, c, g, expected_df, fig, axes, canon_gene_to_categories, \
        cat, annot_rows, gnorm, canonical_cats, db_hits, tr, annot_df, summary_rows, \
        best_canonical, best_db, expected_canonical, top50_genes, b, e, d, summary_df, \
        fn
    # ============================================================
    # CIPHER-LARRY composite marker validation using TOP 50 u genes
    # ============================================================
    # What this does:
    #   1. Loads CIPHER force genes from composition_CIPHER_force_genes.csv
    #   2. Aggregates u genes across CV folds
    #   3. Takes top 50 positive u genes per CIPHER fate/cell type
    #   4. Tests canonical marker overlap/enrichment
    #   5. Tests database marker-term enrichment if marker libraries are loaded
    #   6. Makes composite plots:
    #        - top 50 u gene heatmap
    #        - canonical marker -log10(FDR)
    #        - canonical marker overlap fraction
    #        - marker-category -log10(FDR)
    #        - marker-category log2 odds
    #        - expected cell-type marker summary
    #        - per-fate database term barplots
    #        - top 50 gene annotation table
    #
    # Assumes one of these may already exist in memory:
    #   marker_libraries / MARKER_LIBRARIES / marker_sets_by_library
    #
    # If not, it tries to fetch Enrichr/GSEApy libraries.
    # ============================================================

    import os
    import re
    import math
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.stats import hypergeom
    from statsmodels.stats.multitest import multipletests

    warnings.filterwarnings("ignore")

    # ============================================================
    # CONFIG
    # ============================================================

    CIPHER_OUTDIR = os.path.join(OUT_BASE, "cipher_larry_clone_fate_composition_temperature_calibrated")

    FORCE_PATH = os.path.join(CIPHER_OUTDIR, "composition_CIPHER_force_genes.csv")
    HVG_PATH   = os.path.join(CIPHER_OUTDIR, "selected_early_hvgs.csv")

    OUTDIR = os.path.join(CIPHER_OUTDIR, "marker_association_top50_composite")
    os.makedirs(OUTDIR, exist_ok=True)

    MAIN_TOP_N = 25
    TOP_N_VALUES = [25]

    USE_DIRECTION = "positive"

    # Set False if you want literally every database term.
    FILTER_TO_RELEVANT_TERMS = True

    TERM_RELEVANCE_KEYWORDS = [
        "neutrophil", "granulocyte", "monocyte", "macrophage",
        "basophil", "baso", "mast", "megakaryocyte", "platelet",
        "erythroid", "erythroblast", "lymphoid", "lymphocyte",
        "b cell", "t cell", "nk cell", "dendritic", "dc",
        "hematopo", "haematopo", "myeloid", "progenitor",
        "stem cell", "hspc", "hsc", "lsk",
    ]

    FETCH_GSEAPY_LIBRARIES_IF_NEEDED = True
    GSEAPY_ORGANISM = "Mouse"

    GSEAPY_LIBRARY_KEYWORDS = [
        "Panglao",
        "CellMarker",
        "Tabula",
        "Mouse_Gene_Atlas",
        "HDSigDB",
        "KEGG",
        "WikiPathways",
        "PerturbAtlas",
    ]

    # For database barplots.
    N_TOP_TERMS_TO_PLOT = 12

    # Plot style.
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    sns.set_context("talk")

    # ============================================================
    # HELPERS
    # ============================================================

    def safe_name(x):
        return (
            str(x)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "_")
        )


    def bh_fdr(pvals):
        pvals = np.asarray(pvals, dtype=float)
        ok = np.isfinite(pvals)
        out = np.full_like(pvals, np.nan, dtype=float)
        if ok.sum() > 0:
            out[ok] = multipletests(pvals[ok], method="fdr_bh")[1]
        return out

    def term_is_relevant(term):
        t = clean_term(term).lower()
        return any(k.lower() in t for k in TERM_RELEVANCE_KEYWORDS)

    def hypergeom_enrich(query_genes, marker_genes, background_genes):
        bg = set(map(norm_gene, background_genes))
        q = set(map(norm_gene, query_genes)) & bg
        m = set(map(norm_gene, marker_genes)) & bg

        N = len(bg)
        K = len(m)
        n = len(q)
        k = len(q & m)

        if N == 0 or K == 0 or n == 0 or k == 0:
            p = 1.0
        else:
            p = float(hypergeom.sf(k - 1, N, K, n))

        expected = (n * K / N) if N > 0 else np.nan
        odds = (k / expected) if expected and expected > 0 else np.nan

        return {
            "N_background": N,
            "K_marker_in_background": K,
            "n_query_in_background": n,
            "k_overlap": k,
            "expected_overlap": expected,
            "odds_enrichment": odds,
            "p_value": p,
            "overlap_genes_norm": sorted(q & m),
        }

    def flatten_marker_libraries(marker_libraries):
        rows = []

        for lib, terms in marker_libraries.items():
            if terms is None:
                continue

            if not isinstance(terms, dict):
                continue

            for term, genes in terms.items():
                genes = [str(g).strip() for g in genes if str(g).strip()]
                if len(genes) == 0:
                    continue

                rows.append({
                    "library": str(lib),
                    "term": str(term),
                    "term_clean": clean_term(term),
                    "genes": genes,
                    "genes_norm": sorted(set(map(norm_gene, genes))),
                    "n_genes_raw": len(set(map(norm_gene, genes))),
                })

        df = pd.DataFrame(rows)

        if len(df) == 0:
            return df

        if FILTER_TO_RELEVANT_TERMS:
            df = df[df["term_clean"].apply(term_is_relevant)].copy()

        return df.reset_index(drop=True)

    def expected_category_for_fate(fate):
        f = str(fate).lower()
        if "neut" in f:
            return "Neutrophil"
        if "mono" in f:
            return "Monocyte"
        if "baso" in f:
            return "Baso"
        if "mast" in f:
            return "Mast"
        if "meg" in f:
            return "Meg"
        if "ery" in f:
            return "Erythroid"
        if "lymph" in f:
            return "Lymphoid"
        if "dc" in f or "dend" in f:
            return "Dendritic"
        return None

    def terms_for_category(category, all_terms_df):
        synonyms = FATE_CATEGORY_SYNONYMS.get(category, [category])
        syn_low = [s.lower() for s in synonyms]

        mask = all_terms_df["term_clean"].astype(str).str.lower().apply(
            lambda t: any(s in t for s in syn_low)
        )

        return all_terms_df[mask].copy()

    # ============================================================
    # CANONICAL MARKER SETS
    # Edit/expand these if desired.
    # ============================================================

    CANONICAL_MARKERS = {
        "Neutrophil": [
            "Elane", "Prtn3", "Mpo", "Ctsg", "Ngp", "Lcn2", "S100a8", "S100a9",
            "Camp", "Ltf", "Mmp8", "Retnlg", "Cebpe", "Ly6g", "Cxcr2", "Fcgr3",
            "Mmp9", "Lcn2", "Sell", "Cd177", "Csf3r",
        ],
        "Monocyte": [
            "Lyz2", "Csf1r", "Ctss", "Ctsb", "Lgals3", "Mpeg1", "Ccr2", "Ly6c2",
            "Itgam", "Cd14", "Fcgr1", "Sirpa", "Cybb", "Ms4a7", "Plac8", "S100a4",
            "Cebpb", "Irf8", "Lpl", "Aif1", "Tyrobp", "Ccl6", "Lst1",
        ],
        "Baso": [
            "Gata2", "Cpa3", "Prg2", "Hdc", "Ms4a2", "Fcer1a", "Alox5", "Srgn",
            "Ccr3", "Il4", "Il13", "Mcpt8", "Csf2rb", "Csf2rb2", "Kit",
            "Il3ra", "Fcerg1", "Fcer1g", "Cd200r3", "Slc6a4",
        ],
        "Mast": [
            "Cma1", "Cpa3", "Tpsb2", "Tpsab1", "Hdc", "Kit", "Ms4a2", "Fcer1a",
            "Gata2", "Mcpt4", "Mcpt8", "Srgn", "Alox5", "Fcer1g", "Ctsg",
            "Il1rl1", "Cd63", "Cd9",
        ],
        "Meg": [
            "Pf4", "Ppbp", "Itga2b", "Itgb3", "Gp9", "Gp1ba", "Gp1bb", "Nfe2",
            "Mpl", "Vwf", "Rap1b", "Tubb1", "Fli1", "Gata1", "Pbx1",
            "F2r", "Treml1", "Clec1b", "Cd9", "Mmrn1",
        ],
        "Erythroid": [
            "Hbb-bs", "Hbb-bt", "Hba-a1", "Hba-a2", "Alas2", "Klf1", "Gata1",
            "Nfe2", "Tfrc", "Car2", "Epor", "Hemgn", "Ermap", "Sptb",
            "Gypa", "Ank1", "Slc4a1",
        ],
        "Lymphoid": [
            "Cd3d", "Cd3e", "Cd3g", "Trac", "Ms4a1", "Cd79a", "Cd79b", "Nkg7",
            "Gzma", "Gzmb", "Il7r", "Dntt", "Rag1", "Rag2", "Ly6d",
            "Lck", "Cd2", "Cd8a", "Cd4", "Klrb1c",
        ],
        "Dendritic": [
            "Itgax", "Flt3", "Ccr7", "H2-Aa", "H2-Ab1", "Cd74", "Clec9a",
            "Siglech", "Irf8", "Batf3", "Zbtb46", "Xcr1", "Cst3",
            "Fcgr1", "Itgae",
        ],
        "HSPC": [
            "Kit", "Ly6a", "Cd34", "Procr", "Meis1", "Hlf", "Mecom", "Mpl",
            "Flt3", "Gata2", "Tal1", "Lmo2", "Runx1", "Slamf1", "Thy1",
        ],
    }

    FATE_CATEGORY_SYNONYMS = {
        "Neutrophil": ["neutrophil", "granulocyte", "pmn", "polymorphonuclear"],
        "Monocyte": ["monocyte", "macrophage"],
        "Baso": ["basophil", "baso"],
        "Mast": ["mast"],
        "Meg": ["megakaryocyte", "platelet", "thrombocyte"],
        "Erythroid": ["erythroid", "erythroblast", "red blood"],
        "Lymphoid": ["lymphoid", "lymphocyte", "b cell", "t cell", "nk cell"],
        "Dendritic": ["dendritic", "dc", "pdc", "cdc"],
        "HSPC": ["hematopoietic stem", "haematopoietic stem", "hsc", "hspc", "progenitor", "stem cell", "lsk"],
    }

    # ============================================================
    # LOAD CIPHER FORCE GENES
    # ============================================================

    force_df = pd.read_csv(FORCE_PATH)
    print("Loaded force_df:", force_df.shape)
    print("Columns:", list(force_df.columns))

    required_cols = {"fold", "fate", "direction", "rank", "gene", "u"}
    missing = required_cols - set(force_df.columns)
    if missing:
        raise ValueError(f"Missing required columns from force file: {missing}")

    if os.path.exists(HVG_PATH):
        hvg_df = pd.read_csv(HVG_PATH)
        background_genes = hvg_df["gene"].astype(str).tolist()
        print(f"Background: {len(background_genes)} HVGs from {HVG_PATH}")
    else:
        background_genes = force_df["gene"].astype(str).unique().tolist()
        print(f"Background: {len(background_genes)} genes from force table only")

    background_norm = sorted(set(map(norm_gene, background_genes)))

    force_pos = force_df[
        force_df["direction"].astype(str).str.lower() == USE_DIRECTION.lower()
    ].copy()

    if force_pos.empty:
        raise RuntimeError(f"No rows found with direction={USE_DIRECTION}")

    # ============================================================
    # AGGREGATE TOP u GENES ACROSS FOLDS
    # ============================================================

    agg_dict = {
        "mean_u": ("u", "mean"),
        "median_u": ("u", "median"),
        "max_u": ("u", "max"),
        "mean_rank": ("rank", "mean"),
        "min_rank": ("rank", "min"),
        "n_folds_present": ("fold", "nunique"),
    }

    if "delta_weighted_composition" in force_pos.columns:
        agg_dict["mean_delta"] = ("delta_weighted_composition", "mean")

    agg = (
        force_pos
        .groupby(["fate", "gene"], as_index=False)
        .agg(**agg_dict)
    )

    agg["gene_norm"] = agg["gene"].map(norm_gene)

    # Important ranking:
    # 1. Prefer genes that appear in many folds.
    # 2. Then high mean u.
    # 3. Then lower mean rank.
    agg = agg.sort_values(
        ["fate", "n_folds_present", "mean_u", "mean_rank"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    agg.to_csv(os.path.join(OUTDIR, "aggregated_positive_CIPHER_u_genes_all.csv"), index=False)

    selected_fates = agg["fate"].drop_duplicates().tolist()
    print("\nFates found:")
    print(selected_fates)

    top_genes_by_fate = {}
    top_gene_rows = []

    for fate in selected_fates:
        sub = agg[agg["fate"] == fate].copy()
        top_genes_by_fate[fate] = {}

        for top_n in TOP_N_VALUES:
            genes = sub.head(top_n)["gene"].astype(str).tolist()
            top_genes_by_fate[fate][top_n] = genes

            for rank, (_, row) in enumerate(sub.head(top_n).iterrows(), start=1):
                top_gene_rows.append({
                    "fate": fate,
                    "top_n": top_n,
                    "rank": rank,
                    "gene": row["gene"],
                    "mean_u": row["mean_u"],
                    "median_u": row["median_u"],
                    "max_u": row["max_u"],
                    "mean_rank_across_folds": row["mean_rank"],
                    "min_rank": row["min_rank"],
                    "n_folds_present": row["n_folds_present"],
                    "mean_delta": row["mean_delta"] if "mean_delta" in row.index else np.nan,
                })

    top_genes_df = pd.DataFrame(top_gene_rows)
    top_genes_df.to_csv(os.path.join(OUTDIR, f"top{MAIN_TOP_N}_and_top20_CIPHER_u_genes_by_fate.csv"), index=False)

    print(f"\nTop {MAIN_TOP_N} positive CIPHER u genes per fate:")
    for fate in selected_fates:
        print(f"\n{fate}")
        print(", ".join(top_genes_by_fate[fate][MAIN_TOP_N]))

    # ============================================================
    # TOP 50 U GENE HEATMAP
    # ============================================================

    top50_union = []
    for fate in selected_fates:
        top50_union.extend(top_genes_by_fate[fate][MAIN_TOP_N])
    top50_union = list(dict.fromkeys(top50_union))

    u_heat = (
        agg
        .pivot_table(index="gene", columns="fate", values="mean_u", fill_value=0.0)
        .reindex(top50_union)
        .reindex(columns=selected_fates)
    )

    plt.figure(figsize=(1.2 * len(selected_fates) + 5, 0.20 * len(top50_union) + 5))
    sns.heatmap(
        u_heat,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "mean CIPHER force u"},
    )
    plt.title(f"Top {MAIN_TOP_N} positive CIPHER u genes per fate")
    plt.xlabel("CIPHER fate force")
    plt.ylabel("gene")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"top{MAIN_TOP_N}_CIPHER_u_gene_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"top{MAIN_TOP_N}_CIPHER_u_gene_heatmap.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # LOAD MARKER LIBRARIES
    # ============================================================

    loaded_marker_libraries = None

    for varname in [
        "marker_libraries",
        "MARKER_LIBRARIES",
        "marker_sets_by_library",
        "MARKER_SETS_BY_LIBRARY",
    ]:
        if varname in globals():
            loaded_marker_libraries = globals()[varname]
            print(f"\nUsing existing marker library variable: {varname}")
            break

    if loaded_marker_libraries is None and FETCH_GSEAPY_LIBRARIES_IF_NEEDED:
        try:
            import gseapy as gp
            from gseapy.parser import get_library_name, get_library

            print("\nNo marker_libraries variable found. Trying to fetch GSEApy/Enrichr libraries...")

            all_libs = get_library_name(organism=GSEAPY_ORGANISM)
            chosen_libs = [
                lib for lib in all_libs
                if any(k.lower() in lib.lower() for k in GSEAPY_LIBRARY_KEYWORDS)
            ]

            print("\nCandidate libraries:")
            for lib in chosen_libs:
                print("  ", lib)

            loaded_marker_libraries = {}

            for lib in chosen_libs:
                try:
                    loaded_marker_libraries[lib] = get_library(name=lib, organism=GSEAPY_ORGANISM)
                    print(f"Loaded {lib}: {len(loaded_marker_libraries[lib])} terms")
                except Exception as e:
                    print(f"[skip] Could not load {lib}: {e}")

        except Exception as e:
            print("\nCould not fetch GSEApy libraries.")
            print("Reason:", repr(e))
            loaded_marker_libraries = {}

    if loaded_marker_libraries is None:
        loaded_marker_libraries = {}

    marker_terms_df = flatten_marker_libraries(loaded_marker_libraries)

    # Add canonical marker sets as one library.
    canonical_rows = []
    for term, genes in CANONICAL_MARKERS.items():
        canonical_rows.append({
            "library": "canonical_manual_markers",
            "term": term,
            "term_clean": term,
            "genes": genes,
            "genes_norm": sorted(set(map(norm_gene, genes))),
            "n_genes_raw": len(set(map(norm_gene, genes))),
        })

    canonical_df = pd.DataFrame(canonical_rows)

    all_terms_df = pd.concat([canonical_df, marker_terms_df], ignore_index=True)

    if len(all_terms_df) == 0:
        raise RuntimeError(
            "No marker terms loaded. Define marker_libraries = {'lib': {'term': [genes]}} "
            "or install/use gseapy with internet access."
        )

    all_terms_df["n_genes_in_background"] = all_terms_df["genes_norm"].apply(
        lambda gs: len(set(gs) & set(background_norm))
    )

    all_terms_df = all_terms_df[all_terms_df["n_genes_in_background"] > 0].copy()
    all_terms_df.to_csv(os.path.join(OUTDIR, "all_marker_terms_used.csv"), index=False)

    print(f"\nMarker terms used: {len(all_terms_df)}")
    print(all_terms_df["library"].value_counts().head(20))

    # ============================================================
    # 1) CANONICAL MARKER ENRICHMENT USING TOP 50
    # ============================================================

    canonical_results = []

    for fate in selected_fates:
        query = top_genes_by_fate[fate][MAIN_TOP_N]

        for marker_category, marker_genes in CANONICAL_MARKERS.items():
            res = hypergeom_enrich(query, marker_genes, background_genes)

            canonical_results.append({
                "cipher_fate": fate,
                "marker_category": marker_category,
                "top_n": MAIN_TOP_N,
                **{k: v for k, v in res.items() if k != "overlap_genes_norm"},
                "overlap_genes": ", ".join(res["overlap_genes_norm"]),
            })

    canonical_results = pd.DataFrame(canonical_results)
    canonical_results["fdr"] = bh_fdr(canonical_results["p_value"].values)
    canonical_results["minus_log10_fdr"] = -np.log10(np.maximum(canonical_results["fdr"].values, 1e-300))
    canonical_results["overlap_fraction_of_query"] = (
        canonical_results["k_overlap"] /
        canonical_results["n_query_in_background"].replace(0, np.nan)
    )
    canonical_results["log2_odds_enrichment"] = np.log2(
        canonical_results["odds_enrichment"].replace(0, np.nan)
    )

    canonical_results.to_csv(
        os.path.join(OUTDIR, f"canonical_marker_enrichment_top{MAIN_TOP_N}.csv"),
        index=False,
    )

    print("\nCanonical marker enrichment, best matches:")
    print(
        canonical_results
        .sort_values(["cipher_fate", "p_value"])
        .groupby("cipher_fate")
        .head(4)
        [["cipher_fate", "marker_category", "k_overlap", "odds_enrichment", "p_value", "fdr", "overlap_genes"]]
        .to_string(index=False)
    )

    cat_order = list(CANONICAL_MARKERS.keys())
    fate_order = selected_fates

    heat_fdr = (
        canonical_results
        .pivot_table(index="cipher_fate", columns="marker_category", values="minus_log10_fdr", fill_value=0)
        .reindex(index=fate_order, columns=cat_order)
    )

    heat_overlap = (
        canonical_results
        .pivot_table(index="cipher_fate", columns="marker_category", values="overlap_fraction_of_query", fill_value=0)
        .reindex(index=fate_order, columns=cat_order)
    )

    heat_log2odds = (
        canonical_results
        .pivot_table(index="cipher_fate", columns="marker_category", values="log2_odds_enrichment", fill_value=0)
        .reindex(index=fate_order, columns=cat_order)
    )

    plt.figure(figsize=(1.15 * len(cat_order) + 4, 0.7 * len(fate_order) + 3))
    sns.heatmap(
        heat_fdr,
        cmap="viridis",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "-log10(FDR)"},
    )
    plt.title(f"CIPHER top {MAIN_TOP_N} u genes vs canonical markers")
    plt.xlabel("canonical marker category")
    plt.ylabel("CIPHER fate force")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"canonical_marker_enrichment_top{MAIN_TOP_N}_minuslog10FDR.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"canonical_marker_enrichment_top{MAIN_TOP_N}_minuslog10FDR.svg"), bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(1.15 * len(cat_order) + 4, 0.7 * len(fate_order) + 3))
    sns.heatmap(
        heat_overlap,
        cmap="mako",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": f"overlap fraction of top {MAIN_TOP_N}"},
    )
    plt.title(f"CIPHER top {MAIN_TOP_N} u genes: canonical marker overlap")
    plt.xlabel("canonical marker category")
    plt.ylabel("CIPHER fate force")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"canonical_marker_overlap_top{MAIN_TOP_N}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"canonical_marker_overlap_top{MAIN_TOP_N}.svg"), bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(1.15 * len(cat_order) + 4, 0.7 * len(fate_order) + 3))
    sns.heatmap(
        heat_log2odds,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "log2 odds enrichment"},
    )
    plt.title(f"CIPHER top {MAIN_TOP_N} u genes: canonical marker odds")
    plt.xlabel("canonical marker category")
    plt.ylabel("CIPHER fate force")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"canonical_marker_log2odds_top{MAIN_TOP_N}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"canonical_marker_log2odds_top{MAIN_TOP_N}.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # 2) DATABASE MARKER TERM ENRICHMENT USING TOP 50
    # ============================================================

    db_results = []

    for fate in selected_fates:
        for top_n in TOP_N_VALUES:
            query = top_genes_by_fate[fate][top_n]

            for _, term_row in all_terms_df.iterrows():
                res = hypergeom_enrich(query, term_row["genes_norm"], background_genes)

                db_results.append({
                    "cipher_fate": fate,
                    "top_n": top_n,
                    "library": term_row["library"],
                    "term": term_row["term"],
                    "term_clean": term_row["term_clean"],
                    "n_term_genes_raw": term_row["n_genes_raw"],
                    "n_term_genes_in_background": term_row["n_genes_in_background"],
                    **{k: v for k, v in res.items() if k != "overlap_genes_norm"},
                    "overlap_genes": ", ".join(res["overlap_genes_norm"]),
                })

    db_results = pd.DataFrame(db_results)

    # FDR separately within each top_n across all fates/terms.
    db_results["fdr_within_topn"] = np.nan
    for top_n in sorted(db_results["top_n"].unique()):
        idx = db_results["top_n"] == top_n
        db_results.loc[idx, "fdr_within_topn"] = bh_fdr(db_results.loc[idx, "p_value"].values)

    # Also FDR within each fate/top_n, useful for per-fate barplots.
    db_results["fdr_within_fate_topn"] = np.nan
    for (fate, top_n), sub_idx in db_results.groupby(["cipher_fate", "top_n"]).groups.items():
        idx = list(sub_idx)
        db_results.loc[idx, "fdr_within_fate_topn"] = bh_fdr(db_results.loc[idx, "p_value"].values)

    db_results["minus_log10_fdr"] = -np.log10(np.maximum(db_results["fdr_within_topn"].values, 1e-300))
    db_results["minus_log10_fdr_within_fate"] = -np.log10(np.maximum(db_results["fdr_within_fate_topn"].values, 1e-300))
    db_results["overlap_fraction_of_query"] = (
        db_results["k_overlap"] /
        db_results["n_query_in_background"].replace(0, np.nan)
    )
    db_results["log2_odds_enrichment"] = np.log2(
        db_results["odds_enrichment"].replace(0, np.nan)
    )

    db_results = db_results.sort_values(["cipher_fate", "top_n", "p_value"]).reset_index(drop=True)
    db_results.to_csv(
        os.path.join(OUTDIR, f"all_marker_database_enrichment_results_top{MAIN_TOP_N}.csv"),
        index=False,
    )

    top_db = (
        db_results[db_results["top_n"] == MAIN_TOP_N]
        .sort_values(["cipher_fate", "p_value"])
        .groupby("cipher_fate")
        .head(20)
        .copy()
    )

    top_db.to_csv(
        os.path.join(OUTDIR, f"top_marker_database_terms_top{MAIN_TOP_N}.csv"),
        index=False,
    )

    print(f"\nTop database marker terms for top {MAIN_TOP_N} CIPHER genes:")
    for fate in selected_fates:
        print(f"\n=== {fate} ===")
        display_cols = [
            "library", "term_clean", "k_overlap", "odds_enrichment",
            "p_value", "fdr_within_topn", "fdr_within_fate_topn", "overlap_genes"
        ]
        print(
            top_db[top_db["cipher_fate"] == fate][display_cols]
            .head(10)
            .to_string(index=False)
        )

    # Per-fate database term barplots.
    for fate in selected_fates:
        sub = (
            db_results[
                (db_results["cipher_fate"] == fate) &
                (db_results["top_n"] == MAIN_TOP_N) &
                (db_results["k_overlap"] > 0)
            ]
            .sort_values("p_value")
            .head(N_TOP_TERMS_TO_PLOT)
            .copy()
        )

        if len(sub) == 0:
            continue

        sub["plot_label"] = sub["term_clean"].str.slice(0, 80)

        plt.figure(figsize=(11, max(4.5, 0.5 * len(sub))))
        sns.barplot(
            data=sub,
            y="plot_label",
            x="minus_log10_fdr_within_fate",
            hue="library",
            dodge=False,
        )
        plt.xlabel("-log10(FDR within fate)")
        plt.ylabel("")
        plt.title(f"{fate}: enriched marker terms among top {MAIN_TOP_N} CIPHER u genes")
        plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"{safe_name(fate)}_top{MAIN_TOP_N}_marker_terms_barplot.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"{safe_name(fate)}_top{MAIN_TOP_N}_marker_terms_barplot.svg"), bbox_inches="tight")
        plt.show()

    # ============================================================
    # 3) CATEGORY-LEVEL MARKER ASSOCIATION USING TOP 50
    # ============================================================

    category_results = []

    for fate in selected_fates:
        query = top_genes_by_fate[fate][MAIN_TOP_N]

        for category in FATE_CATEGORY_SYNONYMS.keys():
            term_sub = terms_for_category(category, all_terms_df)

            if len(term_sub) == 0:
                union_genes = CANONICAL_MARKERS.get(category, [])
                source_terms = "canonical_only_if_available"
                n_terms = 0
            else:
                union = set()
                for genes in term_sub["genes_norm"]:
                    union.update(genes)
                union_genes = sorted(union)
                source_terms = "; ".join(term_sub["term_clean"].head(10).astype(str).tolist())
                n_terms = len(term_sub)

            # Add canonical genes into the union.
            if category in CANONICAL_MARKERS:
                union_genes = sorted(
                    set(map(norm_gene, union_genes)) |
                    set(map(norm_gene, CANONICAL_MARKERS[category]))
                )

            res = hypergeom_enrich(query, union_genes, background_genes)

            category_results.append({
                "cipher_fate": fate,
                "marker_category": category,
                "top_n": MAIN_TOP_N,
                "n_matching_database_terms": n_terms,
                "example_matching_terms": source_terms,
                **{k: v for k, v in res.items() if k != "overlap_genes_norm"},
                "overlap_genes": ", ".join(res["overlap_genes_norm"]),
            })

    category_results = pd.DataFrame(category_results)
    category_results["fdr"] = bh_fdr(category_results["p_value"].values)
    category_results["minus_log10_fdr"] = -np.log10(np.maximum(category_results["fdr"].values, 1e-300))
    category_results["overlap_fraction_of_query"] = (
        category_results["k_overlap"] /
        category_results["n_query_in_background"].replace(0, np.nan)
    )
    category_results["log2_odds_enrichment"] = np.log2(
        category_results["odds_enrichment"].replace(0, np.nan)
    )

    category_results.to_csv(
        os.path.join(OUTDIR, f"category_level_marker_association_top{MAIN_TOP_N}.csv"),
        index=False,
    )

    cat_order2 = list(FATE_CATEGORY_SYNONYMS.keys())

    cat_heat = (
        category_results
        .pivot_table(index="cipher_fate", columns="marker_category", values="minus_log10_fdr", fill_value=0)
        .reindex(index=fate_order, columns=cat_order2)
    )

    cat_odds = (
        category_results
        .pivot_table(index="cipher_fate", columns="marker_category", values="log2_odds_enrichment", fill_value=0)
        .reindex(index=fate_order, columns=cat_order2)
    )

    cat_overlap = (
        category_results
        .pivot_table(index="cipher_fate", columns="marker_category", values="overlap_fraction_of_query", fill_value=0)
        .reindex(index=fate_order, columns=cat_order2)
    )

    plt.figure(figsize=(1.15 * len(cat_order2) + 4, 0.7 * len(fate_order) + 3))
    sns.heatmap(
        cat_heat,
        cmap="viridis",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "-log10(FDR)"},
    )
    plt.title(f"CIPHER top {MAIN_TOP_N} u genes vs marker-term categories")
    plt.xlabel("marker-term category")
    plt.ylabel("CIPHER fate force")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"category_marker_association_top{MAIN_TOP_N}_minuslog10FDR.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"category_marker_association_top{MAIN_TOP_N}_minuslog10FDR.svg"), bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(1.15 * len(cat_order2) + 4, 0.7 * len(fate_order) + 3))
    sns.heatmap(
        cat_odds,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "log2 odds enrichment"},
    )
    plt.title(f"CIPHER top {MAIN_TOP_N} u genes: marker category odds")
    plt.xlabel("marker-term category")
    plt.ylabel("CIPHER fate force")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"category_marker_association_top{MAIN_TOP_N}_log2odds.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"category_marker_association_top{MAIN_TOP_N}_log2odds.svg"), bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(1.15 * len(cat_order2) + 4, 0.7 * len(fate_order) + 3))
    sns.heatmap(
        cat_overlap,
        cmap="mako",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": f"overlap fraction of top {MAIN_TOP_N}"},
    )
    plt.title(f"CIPHER top {MAIN_TOP_N} u genes: marker category overlap")
    plt.xlabel("marker-term category")
    plt.ylabel("CIPHER fate force")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"category_marker_association_top{MAIN_TOP_N}_overlap_fraction.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"category_marker_association_top{MAIN_TOP_N}_overlap_fraction.svg"), bbox_inches="tight")
    plt.show()

    # ============================================================
    # 4) EXPECTED FATE-CATEGORY SUMMARY
    # ============================================================

    expected_rows = []

    for fate in selected_fates:
        exp_cat = expected_category_for_fate(fate)

        if exp_cat is None:
            continue

        canon_sub = canonical_results[
            (canonical_results["cipher_fate"] == fate) &
            (canonical_results["marker_category"] == exp_cat)
        ]

        cat_sub = category_results[
            (category_results["cipher_fate"] == fate) &
            (category_results["marker_category"] == exp_cat)
        ]

        row = {
            "cipher_fate": fate,
            "expected_marker_category": exp_cat,
        }

        if len(canon_sub) > 0:
            c = canon_sub.iloc[0]
            row.update({
                "canonical_k_overlap": c["k_overlap"],
                "canonical_overlap_fraction": c["overlap_fraction_of_query"],
                "canonical_odds_enrichment": c["odds_enrichment"],
                "canonical_p_value": c["p_value"],
                "canonical_fdr": c["fdr"],
                "canonical_minus_log10_fdr": c["minus_log10_fdr"],
                "canonical_overlap_genes": c["overlap_genes"],
            })

        if len(cat_sub) > 0:
            g = cat_sub.iloc[0]
            row.update({
                "category_k_overlap": g["k_overlap"],
                "category_overlap_fraction": g["overlap_fraction_of_query"],
                "category_odds_enrichment": g["odds_enrichment"],
                "category_p_value": g["p_value"],
                "category_fdr": g["fdr"],
                "category_minus_log10_fdr": g["minus_log10_fdr"],
                "category_overlap_genes": g["overlap_genes"],
            })

        expected_rows.append(row)

    expected_df = pd.DataFrame(expected_rows)
    expected_df.to_csv(
        os.path.join(OUTDIR, f"expected_fate_marker_category_hits_top{MAIN_TOP_N}.csv"),
        index=False,
    )

    if len(expected_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.barplot(
            data=expected_df,
            x="cipher_fate",
            y="canonical_minus_log10_fdr",
            ax=axes[0],
        )
        axes[0].set_title("Expected canonical marker significance")
        axes[0].set_ylabel("-log10(FDR)")
        axes[0].set_xlabel("CIPHER fate")
        axes[0].tick_params(axis="x", rotation=45)

        sns.barplot(
            data=expected_df,
            x="cipher_fate",
            y="canonical_odds_enrichment",
            ax=axes[1],
        )
        axes[1].set_title("Expected canonical marker enrichment")
        axes[1].set_ylabel("odds enrichment")
        axes[1].set_xlabel("CIPHER fate")
        axes[1].tick_params(axis="x", rotation=45)

        sns.barplot(
            data=expected_df,
            x="cipher_fate",
            y="canonical_overlap_fraction",
            ax=axes[2],
        )
        axes[2].set_title(f"Top {MAIN_TOP_N} overlap with expected markers")
        axes[2].set_ylabel("overlap fraction")
        axes[2].set_xlabel("CIPHER fate")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"expected_marker_category_summary_top{MAIN_TOP_N}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"expected_marker_category_summary_top{MAIN_TOP_N}.svg"), bbox_inches="tight")
        plt.show()

    # ============================================================
    # 5) ANNOTATE EACH TOP 50 U GENE
    # ============================================================

    canon_gene_to_categories = {}
    for cat, genes in CANONICAL_MARKERS.items():
        for g in genes:
            canon_gene_to_categories.setdefault(norm_gene(g), []).append(cat)

    annot_rows = []

    for _, row in top_genes_df[top_genes_df["top_n"] == MAIN_TOP_N].iterrows():
        gnorm = norm_gene(row["gene"])

        canonical_cats = canon_gene_to_categories.get(gnorm, [])

        db_hits = []
        for _, tr in all_terms_df.iterrows():
            if gnorm in set(tr["genes_norm"]):
                db_hits.append(f"{tr['library']}::{tr['term_clean']}")

        annot_rows.append({
            "cipher_fate": row["fate"],
            "rank": row["rank"],
            "gene": row["gene"],
            "mean_u": row["mean_u"],
            "median_u": row["median_u"],
            "max_u": row["max_u"],
            "mean_rank_across_folds": row["mean_rank_across_folds"],
            "n_folds_present": row["n_folds_present"],
            "canonical_marker_categories": "; ".join(canonical_cats),
            "is_expected_canonical_marker": expected_category_for_fate(row["fate"]) in canonical_cats,
            "n_database_marker_terms": len(db_hits),
            "example_database_terms": "; ".join(db_hits[:10]),
        })

    annot_df = pd.DataFrame(annot_rows)
    annot_df.to_csv(
        os.path.join(OUTDIR, f"top{MAIN_TOP_N}_CIPHER_genes_marker_annotations.csv"),
        index=False,
    )

    # ============================================================
    # 6) COMPOSITE SUMMARY TABLE
    # ============================================================

    summary_rows = []

    for fate in selected_fates:
        exp_cat = expected_category_for_fate(fate)

        best_canonical = (
            canonical_results[canonical_results["cipher_fate"] == fate]
            .sort_values("p_value")
            .head(1)
        )

        best_db = (
            db_results[
                (db_results["cipher_fate"] == fate) &
                (db_results["top_n"] == MAIN_TOP_N)
            ]
            .sort_values("p_value")
            .head(1)
        )

        expected_canonical = canonical_results[
            (canonical_results["cipher_fate"] == fate) &
            (canonical_results["marker_category"] == exp_cat)
        ] if exp_cat is not None else pd.DataFrame()

        top50_genes = top_genes_by_fate[fate][MAIN_TOP_N]

        row = {
            "cipher_fate": fate,
            "expected_marker_category": exp_cat,
            "top50_genes": ", ".join(top50_genes),
        }

        if len(best_canonical) > 0:
            b = best_canonical.iloc[0]
            row.update({
                "best_canonical_category": b["marker_category"],
                "best_canonical_k_overlap": b["k_overlap"],
                "best_canonical_odds": b["odds_enrichment"],
                "best_canonical_fdr": b["fdr"],
                "best_canonical_overlap_genes": b["overlap_genes"],
            })

        if len(expected_canonical) > 0:
            e = expected_canonical.iloc[0]
            row.update({
                "expected_canonical_k_overlap": e["k_overlap"],
                "expected_canonical_overlap_fraction": e["overlap_fraction_of_query"],
                "expected_canonical_odds": e["odds_enrichment"],
                "expected_canonical_fdr": e["fdr"],
                "expected_canonical_overlap_genes": e["overlap_genes"],
            })

        if len(best_db) > 0:
            d = best_db.iloc[0]
            row.update({
                "best_database_library": d["library"],
                "best_database_term": d["term_clean"],
                "best_database_k_overlap": d["k_overlap"],
                "best_database_odds": d["odds_enrichment"],
                "best_database_fdr_within_topn": d["fdr_within_topn"],
                "best_database_fdr_within_fate": d["fdr_within_fate_topn"],
                "best_database_overlap_genes": d["overlap_genes"],
            })

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUTDIR, f"top{MAIN_TOP_N}_marker_validation_composite_summary.csv"),
        index=False,
    )

    print("\n============================================================")
    print(f"DONE: TOP {MAIN_TOP_N} CIPHER marker validation")
    print("============================================================")
    print("Outputs in:", OUTDIR)

    print("\nComposite summary:")
    print(summary_df.to_string(index=False))

    print("\nExpected fate/category hits:")
    if len(expected_df) > 0:
        print(expected_df.to_string(index=False))
    else:
        print("No expected fate/category mappings found.")

    print("\nKey files:")
    for fn in [
        f"top{MAIN_TOP_N}_CIPHER_u_gene_heatmap.png",
        f"canonical_marker_enrichment_top{MAIN_TOP_N}_minuslog10FDR.png",
        f"canonical_marker_overlap_top{MAIN_TOP_N}.png",
        f"canonical_marker_log2odds_top{MAIN_TOP_N}.png",
        f"category_marker_association_top{MAIN_TOP_N}_minuslog10FDR.png",
        f"category_marker_association_top{MAIN_TOP_N}_log2odds.png",
        f"category_marker_association_top{MAIN_TOP_N}_overlap_fraction.png",
        f"expected_marker_category_summary_top{MAIN_TOP_N}.png",
        f"top{MAIN_TOP_N}_CIPHER_genes_marker_annotations.csv",
        f"top{MAIN_TOP_N}_marker_validation_composite_summary.csv",
        f"top_marker_database_terms_top{MAIN_TOP_N}.csv",
    ]:
        print(" -", os.path.join(OUTDIR, fn))
