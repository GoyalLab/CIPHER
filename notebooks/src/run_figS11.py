"""Notebook-only run module for Fig S11 (sci-Plex drug inverse).

Thin-driver orchestration for ``notebooks/suppl/figS11_drug_inverse.ipynb``.
Each large inline main-flow cell of the notebook was relocated here VERBATIM as a
single function (bodies unchanged: same variables, same plt/savefig calls, same
logic).  Config values (DATA_DIR, OUTDIR_BASE, SUPPL, ...) are resolved as MODULE
GLOBALS -- they are injected at runtime by the notebook's injection cell, so they
appear undefined to static analysis.  This module is NOT part of the installable
``cipher`` package; it is a notebook-only helper.

Sections (one function per notebook code cell):
  * ``drug_identification``          -- sub-analysis 1 (LR-TRUE/LR-MF/LFC/CLF + ROC/PR + top-k)
  * ``summary_5panel``               -- sub-analysis 2 (5-panel summary figure)
  * ``global_clustering_permutation``-- sub-analysis 4 (drug-class clustering permutation test)

Sub-analyses 3 and 5 already live in ``src.suppl_druginv``
(``run_dense_ustar_heatmap`` / ``run_cell_line_specific``) and are called directly
from the notebook.
"""
from src.suppl_druginv import *

import os
import glob
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch, Rectangle
from scipy.sparse import issparse, csr_matrix
from scipy.cluster.hierarchy import linkage, leaves_list, cophenet
from scipy.spatial.distance import pdist, squareform

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, *a, **k):
            return x

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (roc_curve, auc,
                                 precision_recall_curve, average_precision_score)
except Exception:  # pragma: no cover
    LogisticRegression = None
    roc_curve = auc = precision_recall_curve = average_precision_score = None


# ============================================================
# (notebook cell 3) sub-analysis 1: covariance-aware drug identification
# ============================================================
def drug_identification():
    import os

    import numpy as np

    import anndata as ad

    import matplotlib.pyplot as plt

    from tqdm import tqdm

    from scipy.sparse import issparse, csr_matrix

    from sklearn.linear_model import LogisticRegression

    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    DATA_PATH = os.path.join(DATA_DIR, "SrivatsanTrapnell2020_sciplex3.h5ad")

    PERT_KEY = "perturbation"

    CONTROL_LABEL = "control"

    MIN_CELLS_PER_DRUG = 100

    test_frac = 0.50

    min_test_per_drug = 20

    seed = 0

    USE_HVG = True

    N_HVG = 1000

    cov_shrink0 = 1e-3

    cov_shrinkd = 5e-2

    jitter = 1e-8

    ridge_lambda = 1.0

    use_hdiag = True

    LFC_EPS = 1e-8

    CLF_C = 1.0

    CLF_MAXITER = 2000

    TOPK_LIST = [1, 2, 3, 4,5,6,7,8,9, 10]

    CONFUSION_TOPK = 189

    TOPK_BARS = 10

    N_EXAMPLES_SAVE = 12

    SAVE_FIG_DPI = 220

    LR_SAVE_TOPLL_K = 20

    OUTDIR = os.path.join(OUTDIR_BASE, "saved_lr_true_sigma_artifacts")

    os.makedirs(OUTDIR, exist_ok=True)

    COLORS = {
        "LR-TRUE": "#1f77b4",  # blue
        "LR-MF":   "#1f77b4",  # blue (dashed)
        "LFC":     "#9467bd",  # purple
        "CLF":     "#d62728",  # red
    }

    dataset_name = os.path.basename(DATA_PATH).replace(".h5ad", "")

    print(f"\n=== LOAD: {dataset_name} ===")

    adata = ad.read_h5ad(DATA_PATH)

    adata.var_names = adata.var_names.astype(str)

    adata.var_names_make_unique()

    labels_all = adata.obs[PERT_KEY].astype(str).values

    if CONTROL_LABEL not in set(labels_all):
        raise ValueError(f"Control label '{CONTROL_LABEL}' not found in obs['{PERT_KEY}'].")

    vc = adata.obs[PERT_KEY].astype(str).value_counts()

    drugs = [
        d for d in vc.index.astype(str).tolist()
        if (d != CONTROL_LABEL) and (not _is_bad_label(d)) and (vc[d] >= int(MIN_CELLS_PER_DRUG))
    ]

    adata = adata[adata.obs[PERT_KEY].astype(str).isin([CONTROL_LABEL] + drugs)].copy()

    labels_all = adata.obs[PERT_KEY].astype(str).values

    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()

    ctrl_idx = np.where(labels_all == CONTROL_LABEL)[0]

    print(f"Cells={adata.n_obs}  Genes={adata.n_vars}  Drugs={len(drugs)}  Control={ctrl_idx.size}")

    if USE_HVG:
        print(f"Selecting HVGs (top {N_HVG})...")
        X = adata.X
        mu = np.asarray(X.mean(axis=0)).ravel()
        mu2 = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = np.maximum(mu2 - mu * mu, 0.0)
        disp = var / np.maximum(mu, 1e-8)
        disp[~np.isfinite(disp)] = 0.0
        hv = np.argsort(disp)[::-1][:min(N_HVG, adata.n_vars)]
        adata = adata[:, hv].copy()
        labels_all = adata.obs[PERT_KEY].astype(str).values
        adata.X = adata.X.tocsr() if issparse(adata.X) else csr_matrix(adata.X)
        ctrl_idx = np.where(labels_all == CONTROL_LABEL)[0]
        print(f"After HVG: Genes={adata.n_vars}")

    drug_mask = (labels_all != CONTROL_LABEL)

    drug_idx = np.where(drug_mask)[0]

    drug_labels = labels_all[drug_mask]

    train_rel, test_rel = _split_train_test_indices(drug_labels, test_frac=test_frac, seed=seed, min_test=min_test_per_drug)

    train_idx = drug_idx[train_rel]

    test_idx = drug_idx[test_rel]

    print(f"Train drug cells={train_idx.size}  Test drug cells={test_idx.size}")

    print("\nComputing pseudo-bulk means on shared gene set...")

    mu0 = _mean_rows_sparse(adata.X[ctrl_idx])

    logmu0 = _log1p_mean_from_mean(mu0, eps=LFC_EPS)

    drugs_fit = sorted(list(set(labels_all[train_idx]).intersection(set(labels_all[test_idx]))))

    drugs_fit = [d for d in drugs_fit if (d != CONTROL_LABEL) and (not _is_bad_label(d))]

    print(f"Drugs with both train+test: {len(drugs_fit)}")

    mu_train, mu_test, n_train, n_test = {}, {}, {}, {}

    for d in tqdm(drugs_fit, desc="Pseudo-bulk means"):
        idx_tr = train_idx[labels_all[train_idx] == d]
        idx_te = test_idx[labels_all[test_idx] == d]
        if idx_tr.size < 2 or idx_te.size < 2:
            continue
        mu_train[d] = _mean_rows_sparse(adata.X[idx_tr])
        mu_test[d]  = _mean_rows_sparse(adata.X[idx_te])
        n_train[d] = int(idx_tr.size)
        n_test[d]  = int(idx_te.size)

    drugs_fit = sorted(mu_train.keys())

    nD = len(drugs_fit)

    if nD < 2:
        raise ValueError("Need at least 2 drugs in drugs_fit after filtering.")

    print(f"Kept drugs after nonempty: {nD}")

    print("\nBuilding control covariances...")

    X0_dense = adata.X[ctrl_idx].toarray().astype(np.float64)

    n0 = int(X0_dense.shape[0])

    Sigma0_true = _shrink_cov(np.cov(X0_dense, rowvar=False), shrink=cov_shrink0)

    print("Building MEAN-FIELD control covariance by column-wise shuffling...")

    X0_mf = _shuffle_columns_independently(X0_dense, seed=seed + 123)

    Sigma0_mf = _shrink_cov(np.cov(X0_mf, rowvar=False), shrink=cov_shrink0)

    diag_diff = np.max(np.abs(np.diag(Sigma0_true) - np.diag(Sigma0_mf)))

    print(f"Max |diag(Sigma_true)-diag(Sigma_mf)| = {diag_diff:.3e}")

    print("\n=== (A1) LR-TRUE ===")

    y_true_lr, y_pred_lr, rank_lr, margin_lr, orders_lr, top_labels_lr, top_lls_lr = _run_lr(
        drugs_fit=drugs_fit, mu_train=mu_train, mu_test=mu_test, mu0=mu0,
        train_idx=train_idx, test_idx=test_idx, labels_all=labels_all, X_csr=adata.X,
        Sigma0=Sigma0_true, cov_shrinkd=cov_shrinkd, jitter=jitter, ridge_lambda=ridge_lambda,
        use_hdiag=use_hdiag, control_label=CONTROL_LABEL, tag="LR-TRUE",
        save_topll_k=LR_SAVE_TOPLL_K
    )

    acc1_lr = float(np.mean(y_true_lr == y_pred_lr))

    print(f"LR-TRUE Top-1 acc: {acc1_lr:.4f}")

    top_drugs_lr = []

    top_lls_lr_bars = []

    for i in range(nD):
        top_idx = orders_lr[i, :int(min(TOPK_BARS, nD))]
        top_drugs_lr.append([drugs_fit[j] for j in top_idx])
        # keep bar LLs if available; else NaN
        if top_lls_lr is not None:
            kbar = int(min(TOPK_BARS, top_lls_lr.shape[1]))
            top_lls_lr_bars.append(top_lls_lr[i, :kbar].tolist())
            if kbar < int(min(TOPK_BARS, nD)):
                top_lls_lr_bars[-1].extend([np.nan] * (int(min(TOPK_BARS, nD)) - kbar))
        else:
            top_lls_lr_bars.append([np.nan for _ in top_idx])

    print("\n=== (A2) LR-MF ===")

    y_true_mf, y_pred_mf, rank_mf, margin_mf, orders_mf, _, _ = _run_lr(
        drugs_fit=drugs_fit, mu_train=mu_train, mu_test=mu_test, mu0=mu0,
        train_idx=train_idx, test_idx=test_idx, labels_all=labels_all, X_csr=adata.X,
        Sigma0=Sigma0_mf, cov_shrinkd=cov_shrinkd, jitter=jitter, ridge_lambda=ridge_lambda,
        use_hdiag=use_hdiag, control_label=CONTROL_LABEL, tag="LR-MF",
        save_topll_k=0
    )

    acc1_mf = float(np.mean(y_true_mf == y_pred_mf))

    print(f"LR-MF   Top-1 acc: {acc1_mf:.4f}")

    print("\n=== (B) LFC ===")

    L_tr = np.stack([_log1p_mean_from_mean(mu_train[d], eps=LFC_EPS) - logmu0 for d in drugs_fit], axis=0)

    L_te = np.stack([_log1p_mean_from_mean(mu_test[d],  eps=LFC_EPS) - logmu0 for d in drugs_fit], axis=0)

    S_lfc = _cosine_sim(L_te, L_tr)

    y_true_lfc = np.array(drugs_fit, dtype=object)

    y_pred_lfc = np.empty(nD, dtype=object)

    rank_lfc = np.zeros(nD, dtype=int)

    margin_lfc = np.zeros(nD, dtype=float)

    orders_lfc = np.zeros((nD, nD), dtype=int)

    for i in range(nD):
        sims = S_lfc[i]
        order = np.argsort(sims)[::-1]
        orders_lfc[i, :] = order
        y_pred_lfc[i] = drugs_fit[int(order[0])]
        rank_lfc[i] = int(np.where(order == i)[0][0] + 1)
        margin_lfc[i] = float(sims[order[0]] - sims[order[1]]) if nD >= 2 else 0.0

    acc1_lfc = float(np.mean(y_true_lfc == y_pred_lfc))

    print(f"LFC Top-1 acc: {acc1_lfc:.4f}")

    print("\n=== (C) CLF ===")

    Xtr = np.stack([(mu_train[d] - mu0) for d in drugs_fit], axis=0).astype(np.float64)

    Xte = np.stack([(mu_test[d]  - mu0) for d in drugs_fit], axis=0).astype(np.float64)

    Xtr_z, zparams = _zscore_fit_transform(Xtr)

    Xte_z = _zscore_transform(Xte, zparams)

    clf = LogisticRegression(
        penalty="l2",
        C=float(CLF_C),
        solver="lbfgs",
        max_iter=int(CLF_MAXITER),
        multi_class="auto",
    )

    clf.fit(Xtr_z, np.array(drugs_fit, str))

    proba = clf.predict_proba(Xte_z)

    classes = clf.classes_.astype(str).tolist()

    class_to_j = {c: j for j, c in enumerate(classes)}

    logp = np.log(np.maximum(proba, 1e-300))

    S_clf = np.zeros((nD, nD), dtype=float)

    for j, d in enumerate(drugs_fit):
        S_clf[:, j] = logp[:, class_to_j[d]]

    y_true_clf = np.array(drugs_fit, dtype=object)

    y_pred_clf = np.empty(nD, dtype=object)

    rank_clf = np.zeros(nD, dtype=int)

    margin_clf = np.zeros(nD, dtype=float)

    orders_clf = np.zeros((nD, nD), dtype=int)

    for i in range(nD):
        scores = S_clf[i]
        order = np.argsort(scores)[::-1]
        orders_clf[i, :] = order
        y_pred_clf[i] = drugs_fit[int(order[0])]
        rank_clf[i] = int(np.where(order == i)[0][0] + 1)
        margin_clf[i] = float(scores[order[0]] - scores[order[1]]) if nD >= 2 else 0.0

    acc1_clf = float(np.mean(y_true_clf == y_pred_clf))

    print(f"CLF Top-1 acc: {acc1_clf:.4f}")

    topk_acc = {"K": np.array(TOPK_LIST, int)}

    for name, rank_vec in [
        ("LR_TRUE", rank_lr),
        ("LR_MF",   rank_mf),
        ("LFC",     rank_lfc),
        ("CLF",     rank_clf),
    ]:
        topk_acc[name] = np.array([_topk_acc_from_rank(rank_vec, K) for K in TOPK_LIST], float)

    print("\n=== TOP-K ACCURACY ===")

    for iK, K in enumerate(TOPK_LIST):
        print(
            f"K={K:>2d} | "
            f"LR-TRUE {topk_acc['LR_TRUE'][iK]:.3f}  "
            f"LR-MF {topk_acc['LR_MF'][iK]:.3f}  "
            f"LFC {topk_acc['LFC'][iK]:.3f}  "
            f"CLF {topk_acc['CLF'][iK]:.3f}"
        )

    conf_png = os.path.join(OUTDIR, f"{dataset_name}__LR_TRUE_SIGMA__confusion_top{CONFUSION_TOPK}.png")

    keep_lr, C_counts_lr, C_row_norm_lr = _plot_confusion_topk(
        y_true_lr, y_pred_lr, topk=CONFUSION_TOPK,
        title=f"{dataset_name}: LR-TRUE confusion (TRUE Sigma0)",
        savepath=conf_png
    )

    rank_png = os.path.join(OUTDIR, f"{dataset_name}__LR_TRUE_SIGMA__rank_hist.png")

    _plot_rank_hist(rank_lr, f"{dataset_name}: LR-TRUE true-rank", savepath=rank_png)

    conf_fn = os.path.join(OUTDIR, f"{dataset_name}__LR_TRUE_SIGMA__confusion_topK.npz")

    np.savez_compressed(
        conf_fn,
        keep_labels=np.array(keep_lr, dtype=object),
        C_counts=C_counts_lr.astype(np.float64),
        C_row_norm=C_row_norm_lr.astype(np.float64),
        y_true=y_true_lr.astype(object),
        y_pred=y_pred_lr.astype(object),
        conf_topk=int(CONFUSION_TOPK),
        seed=int(seed),
        test_frac=float(test_frac),
        use_hvg=bool(USE_HVG),
        n_hvg=int(N_HVG),
    )

    print("Saved:", conf_fn)

    print("Saved:", conf_png)

    print("Saved:", rank_png)

    correct_mask = (y_true_lr == y_pred_lr)

    idx_pool = np.where(correct_mask)[0]

    if idx_pool.size == 0:
        idx_pool = np.arange(nD)

    idx_sorted = idx_pool[np.argsort(margin_lr[idx_pool])[::-1]]

    idx_ex = idx_sorted[:int(min(N_EXAMPLES_SAVE, idx_sorted.size))]

    ex_true = y_true_lr[idx_ex]

    ex_pred = y_pred_lr[idx_ex]

    ex_correct = (ex_true == ex_pred).astype(int)

    ex_margin = margin_lr[idx_ex]

    ex_rank = rank_lr[idx_ex]

    top_drugs_ex = _pad_2d([top_drugs_lr[i] for i in idx_ex], fill="", width=TOPK_BARS, dtype=object)

    top_lls_ex   = _pad_2d([top_lls_lr_bars[i] for i in idx_ex], fill=np.nan, width=TOPK_BARS, dtype=float)

    if top_labels_lr is not None and top_lls_lr is not None:
        ex_top20_labels = top_labels_lr[idx_ex, :].astype(object)
        ex_top20_lls = top_lls_lr[idx_ex, :].astype(float)
        ex_top20_k = int(ex_top20_labels.shape[1])
    else:
        ex_top20_labels = np.empty((len(idx_ex), 0), dtype=object)
        ex_top20_lls = np.empty((len(idx_ex), 0), dtype=float)
        ex_top20_k = 0

    ex_fn = os.path.join(OUTDIR, f"{dataset_name}__LR_TRUE_SIGMA__top_examples.npz")

    np.savez_compressed(
        ex_fn,
        examples_true=ex_true.astype(object),
        examples_pred=ex_pred.astype(object),
        examples_correct=ex_correct.astype(int),
        examples_margin_pred=ex_margin.astype(float),
        examples_rank_true=ex_rank.astype(int),
        top_drugs=top_drugs_ex,
        top_lls=top_lls_ex,
        topk_bars=int(TOPK_BARS),
        # NEW: just top 20 LL (TRUE sigma) for these examples
        top20_labels=ex_top20_labels,
        top20_lls=ex_top20_lls,
        top20_k=int(ex_top20_k),
    )

    print("Saved:", ex_fn)

    print("\n=== ROC/PR (Correctness; y=top1 correct, score=margin_pred) ===")

    ypos_lr  = (y_true_lr  == y_pred_lr ).astype(int)

    ypos_mf  = (y_true_mf  == y_pred_mf ).astype(int)

    ypos_lfc = (y_true_lfc == y_pred_lfc).astype(int)

    ypos_clf = (y_true_clf == y_pred_clf).astype(int)

    rp_lr  = _roc_pr_from_scores(ypos_lr,  margin_lr)

    rp_mf  = _roc_pr_from_scores(ypos_mf,  margin_mf)

    rp_lfc = _roc_pr_from_scores(ypos_lfc, margin_lfc)

    rp_clf = _roc_pr_from_scores(ypos_clf, margin_clf)

    curves = [
        ("LR-TRUE", rp_lr,  acc1_lr,  COLORS["LR-TRUE"], "-"),
        ("LR-MF",   rp_mf,  acc1_mf,  COLORS["LR-MF"],   "--"),
        ("LFC",     rp_lfc, acc1_lfc, COLORS["LFC"],     "-"),
        ("CLF",     rp_clf, acc1_clf, COLORS["CLF"],     "-"),
    ]

    plt.figure(figsize=(14, 6))

    gs = plt.gcf().add_gridspec(1, 2, wspace=0.28)

    ax1 = plt.gcf().add_subplot(gs[0, 0])

    ax2 = plt.gcf().add_subplot(gs[0, 1])

    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=2.8, color="0.6")

    ax1.set_xlabel("False Positive Rate", fontsize=18)

    ax1.set_ylabel("True Positive Rate", fontsize=18)

    ax1.set_title("ROC", fontsize=20)

    ax1.tick_params(labelsize=14)

    ax2.set_xlabel("Recall", fontsize=18)

    ax2.set_ylabel("Precision", fontsize=18)

    ax2.set_title("Precision–Recall", fontsize=20)

    ax2.tick_params(labelsize=14)

    for name, rp, acc, color, ls in curves:
        if rp is None:
            continue
        fpr, tpr, roc_auc, prec, rec, ap = rp
        ax1.plot(fpr, tpr, linewidth=3.6, color=color, linestyle=ls, label=f"{name}  AUC={roc_auc:.3f}")
        ax2.plot(rec, prec, linewidth=3.6, color=color, linestyle=ls, label=f"{name}  AP={ap:.3f}")

    ax1.legend(fontsize=12, frameon=True)

    ax2.legend(fontsize=12, frameon=True)

    plt.tight_layout()

    rocpr_png = os.path.join(OUTDIR, f"{dataset_name}__correctness_ROC_PR__margin_pred.png")

    plt.savefig(rocpr_png, dpi=SAVE_FIG_DPI, bbox_inches="tight")

    plt.show()

    print("Saved:", rocpr_png)

    everything_fn = os.path.join(OUTDIR, f"{dataset_name}__ALL_METHODS__predictions_and_metrics.npz")

    np.savez_compressed(
        everything_fn,
        # meta
        dataset_name=np.array(dataset_name, dtype=object),
        data_path=np.array(DATA_PATH, dtype=object),
        pert_key=np.array(PERT_KEY, dtype=object),
        control_label=np.array(CONTROL_LABEL, dtype=object),
        seed=int(seed),
        test_frac=float(test_frac),
        min_test_per_drug=int(min_test_per_drug),
        min_cells_per_drug=int(MIN_CELLS_PER_DRUG),
        use_hvg=bool(USE_HVG),
        n_hvg=int(N_HVG),
        n_drugs_fit=int(nD),

        # drug list / counts
        drugs_fit=np.array(drugs_fit, dtype=object),
        n_train=np.array([n_train[d] for d in drugs_fit], dtype=int),
        n_test=np.array([n_test[d] for d in drugs_fit], dtype=int),

        # cov diagnostics
        cov_shrink0=float(cov_shrink0),
        cov_shrinkd=float(cov_shrinkd),
        jitter=float(jitter),
        ridge_lambda=float(ridge_lambda),
        use_hdiag=bool(use_hdiag),
        diag_diff_true_vs_mf=float(diag_diff),

        # TOP-K
        topk_list=np.array(TOPK_LIST, dtype=int),
        topk_acc_lr_true=topk_acc["LR_TRUE"],
        topk_acc_lr_mf=topk_acc["LR_MF"],
        topk_acc_lfc=topk_acc["LFC"],
        topk_acc_clf=topk_acc["CLF"],

        # LR-TRUE outputs
        y_true_lr=y_true_lr.astype(object),
        y_pred_lr=y_pred_lr.astype(object),
        rank_lr=rank_lr.astype(int),
        margin_lr=margin_lr.astype(float),
        orders_lr=orders_lr.astype(int),

        # NEW: just top 20 LLs for LR-TRUE (TRUE sigma)
        lr_top20_k=int(top_lls_lr.shape[1]) if (top_lls_lr is not None) else int(0),
        lr_top20_labels=(top_labels_lr.astype(object) if (top_labels_lr is not None) else np.empty((nD, 0), dtype=object)),
        lr_top20_lls=(top_lls_lr.astype(float) if (top_lls_lr is not None) else np.empty((nD, 0), dtype=float)),

        # LR-MF outputs
        y_pred_mf=y_pred_mf.astype(object),
        rank_mf=rank_mf.astype(int),
        margin_mf=margin_mf.astype(float),
        orders_mf=orders_mf.astype(int),

        # LFC outputs
        y_pred_lfc=y_pred_lfc.astype(object),
        rank_lfc=rank_lfc.astype(int),
        margin_lfc=margin_lfc.astype(float),
        orders_lfc=orders_lfc.astype(int),

        # CLF outputs
        y_pred_clf=y_pred_clf.astype(object),
        rank_clf=rank_clf.astype(int),
        margin_clf=margin_clf.astype(float),
        orders_clf=orders_clf.astype(int),

        # accuracies
        acc1_lr_true=float(acc1_lr),
        acc1_lr_mf=float(acc1_mf),
        acc1_lfc=float(acc1_lfc),
        acc1_clf=float(acc1_clf),

        # ROC/PR (store as arrays; empty if undefined)
        lr_fpr=(rp_lr[0] if rp_lr else np.array([], float)),
        lr_tpr=(rp_lr[1] if rp_lr else np.array([], float)),
        lr_roc_auc=(rp_lr[2] if rp_lr else np.nan),
        lr_prec=(rp_lr[3] if rp_lr else np.array([], float)),
        lr_rec=(rp_lr[4] if rp_lr else np.array([], float)),
        lr_ap=(rp_lr[5] if rp_lr else np.nan),

        mf_fpr=(rp_mf[0] if rp_mf else np.array([], float)),
        mf_tpr=(rp_mf[1] if rp_mf else np.array([], float)),
        mf_roc_auc=(rp_mf[2] if rp_mf else np.nan),
        mf_prec=(rp_mf[3] if rp_mf else np.array([], float)),
        mf_rec=(rp_mf[4] if rp_mf else np.array([], float)),
        mf_ap=(rp_mf[5] if rp_mf else np.nan),

        lfc_fpr=(rp_lfc[0] if rp_lfc else np.array([], float)),
        lfc_tpr=(rp_lfc[1] if rp_lfc else np.array([], float)),
        lfc_roc_auc=(rp_lfc[2] if rp_lfc else np.nan),
        lfc_prec=(rp_lfc[3] if rp_lfc else np.array([], float)),
        lfc_rec=(rp_lfc[4] if rp_lfc else np.array([], float)),
        lfc_ap=(rp_lfc[5] if rp_lfc else np.nan),

        clf_fpr=(rp_clf[0] if rp_clf else np.array([], float)),
        clf_tpr=(rp_clf[1] if rp_clf else np.array([], float)),
        clf_roc_auc=(rp_clf[2] if rp_clf else np.nan),
        clf_prec=(rp_clf[3] if rp_clf else np.array([], float)),
        clf_rec=(rp_clf[4] if rp_clf else np.array([], float)),
        clf_ap=(rp_clf[5] if rp_clf else np.nan),

        # paths to saved figs
        rocpr_png=np.array(rocpr_png, dtype=object),
        conf_png=np.array(conf_png, dtype=object),
        rank_png=np.array(rank_png, dtype=object),
    )

    print("Saved:", everything_fn)

    print("\n=== SUMMARY ===")

    print(f"Top-1 acc: LR-TRUE {acc1_lr:.4f} | LR-MF {acc1_mf:.4f} | LFC {acc1_lfc:.4f} | CLF {acc1_clf:.4f}")

    print("Confidence score for ALL methods: margin_pred = top1_score - top2_score (prediction-only)")

    print(f"Saved LR-TRUE top-{LR_SAVE_TOPLL_K} loglikelihoods per test drug (TRUE Sigma0) in EVERYTHING + TOP_EXAMPLES NPZ.")

    print("\nSaved artifacts:")

    print(" ", conf_fn)

    print(" ", ex_fn)

    print(" ", rocpr_png)

    print(" ", everything_fn)

# ============================================================
# (notebook cell 4) sub-analysis 2: 5-panel summary figure
# ============================================================
def summary_5panel():
    import os, glob

    import numpy as np

    import matplotlib.pyplot as plt

    from matplotlib.colors import LinearSegmentedColormap

    OUTDIR = os.path.join(OUTDIR_BASE, "saved_lr_true_sigma_artifacts")

    DATASET_NAME = None

    EXAMPLE_INDEX = 0

    TOPK_BARS = 10

    FONTSIZE = 20

    DPI = 220

    CORNFLOWER = "#6495ED"

    PURPLE     = "#9467bd"

    SALMON     = "#FA8072"

    COLORS = {"LR-TRUE": CORNFLOWER, "LR-MF": CORNFLOWER, "LFC": PURPLE, "CLF": SALMON}

    LINESTYLE = {"LR-TRUE": "-", "LR-MF": "--", "LFC": "-", "CLF": "-"}

    CONF_CMAP = LinearSegmentedColormap.from_list(
        "conf_cmap",
        [(0.0, "#000000"), (0.55, SALMON), (1.0, CORNFLOWER)]
    )

    if DATASET_NAME is None:
        DATASET_NAME = _pick_dataset_name(OUTDIR)

    conf_fn = os.path.join(OUTDIR, f"{DATASET_NAME}__LR_TRUE_SIGMA__confusion_topK.npz")

    ex_fn   = os.path.join(OUTDIR, f"{DATASET_NAME}__LR_TRUE_SIGMA__top_examples.npz")

    all_fn  = os.path.join(OUTDIR, f"{DATASET_NAME}__ALL_METHODS__predictions_and_metrics.npz")

    for f in [conf_fn, ex_fn, all_fn]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing: {f}")

    conf = np.load(conf_fn, allow_pickle=True)

    ex   = np.load(ex_fn, allow_pickle=True)

    allz = np.load(all_fn, allow_pickle=True)

    C_row_norm = conf["C_row_norm"]

    examples_true = ex["examples_true"]

    drugs_fit = _safe_arr(allz, "drugs_fit", None)

    topk_list     = _safe_arr(allz, "topk_list", None)

    topk_lr_true  = _safe_arr(allz, "topk_acc_lr_true", None)

    topk_lr_mf    = _safe_arr(allz, "topk_acc_lr_mf", None)

    topk_lfc      = _safe_arr(allz, "topk_acc_lfc", None)

    topk_clf      = _safe_arr(allz, "topk_acc_clf", None)

    roc = {
        "LR-TRUE": (_safe_arr(allz, "lr_fpr", None),  _safe_arr(allz, "lr_tpr", None),  float(_safe_arr(allz, "lr_roc_auc", np.nan))),
        "LR-MF":   (_safe_arr(allz, "mf_fpr", None),  _safe_arr(allz, "mf_tpr", None),  float(_safe_arr(allz, "mf_roc_auc", np.nan))),
        "LFC":     (_safe_arr(allz, "lfc_fpr", None), _safe_arr(allz, "lfc_tpr", None), float(_safe_arr(allz, "lfc_roc_auc", np.nan))),
        "CLF":     (_safe_arr(allz, "clf_fpr", None), _safe_arr(allz, "clf_tpr", None), float(_safe_arr(allz, "clf_roc_auc", np.nan))),
    }

    pr = {
        "LR-TRUE": (_safe_arr(allz, "lr_rec", None),  _safe_arr(allz, "lr_prec", None),  float(_safe_arr(allz, "lr_ap", np.nan))),
        "LR-MF":   (_safe_arr(allz, "mf_rec", None),  _safe_arr(allz, "mf_prec", None),  float(_safe_arr(allz, "mf_ap", np.nan))),
        "LFC":     (_safe_arr(allz, "lfc_rec", None), _safe_arr(allz, "lfc_prec", None), float(_safe_arr(allz, "lfc_ap", np.nan))),
        "CLF":     (_safe_arr(allz, "clf_rec", None), _safe_arr(allz, "clf_prec", None), float(_safe_arr(allz, "clf_ap", np.nan))),
    }

    if drugs_fit is None:
        raise ValueError("Missing 'drugs_fit' in ALL_METHODS NPZ.")

    drugs_fit = np.array(drugs_fit, dtype=object).tolist()

    best_i, best_labels, best_lls = _choose_best_example_by_ll(allz, TOPK_BARS, fallback_index=EXAMPLE_INDEX)

    if best_labels is None or best_lls is None:
        if EXAMPLE_INDEX < 0 or EXAMPLE_INDEX >= len(examples_true):
            raise ValueError(f"EXAMPLE_INDEX={EXAMPLE_INDEX} out of range (0..{len(examples_true)-1})")
        labs_lr = ex["top_drugs"][EXAMPLE_INDEX, :TOPK_BARS].astype(object).tolist()
        ll_lr = ex["top_lls"][EXAMPLE_INDEX, :TOPK_BARS].astype(float)
    else:
        labs_lr = best_labels.astype(object).tolist()
        ll_lr = best_lls.astype(float)

    if topk_list is None:
        if topk_lr_true is None:
            topk_list = np.arange(1, 11, dtype=int)
        else:
            topk_list = np.arange(1, len(topk_lr_true) + 1, dtype=int)

    curves_topk = {"LR-TRUE": topk_lr_true, "LR-MF": topk_lr_mf, "LFC": topk_lfc, "CLF": topk_clf}

    from matplotlib.colors import LinearSegmentedColormap

    CONF_CMAP = LinearSegmentedColormap.from_list(
        "blue_to_salmon",
        ["#6495ED", "#FA8072"]   # cornflower to salmon
    )

    plt.rcParams.update({"font.size": FONTSIZE})

    fig = plt.figure(figsize=(16, 9))

    gs = fig.add_gridspec(
        2, 6,
        height_ratios=[2.0, 2.0],   # <-- bottom panels taller than before
        wspace=0.55,
        hspace=0.35
    )

    axA = fig.add_subplot(gs[0, 0:2])

    axB = fig.add_subplot(gs[0, 2:4])

    axC = fig.add_subplot(gs[0, 4:6])

    axD = fig.add_subplot(gs[1, 0:3])

    axE = fig.add_subplot(gs[1, 3:6])

    im = axA.imshow(
        C_row_norm,
        aspect="auto",
        interpolation="nearest",
        cmap=CONF_CMAP,
        vmin=0.0,
        vmax=1.0
    )

    axA.set_xlabel("Predicted drug", fontsize=FONTSIZE)

    axA.set_ylabel("True drug", fontsize=FONTSIZE)

    axA.set_xticks([])

    axA.set_yticks([])

    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.02)

    cb.ax.tick_params(labelsize=FONTSIZE * 0.7)

    try:
        ex_true_label = str(drugs_fit[int(best_i)])
    except Exception:
        ex_true_label = str(ex["examples_true"][int(EXAMPLE_INDEX)])

    _plot_example(axB, labs_lr, ll_lr, true_label=ex_true_label,
                  other_color=CORNFLOWER, true_color=SALMON)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=SALMON, edgecolor="none", label="True drug"),
        Patch(facecolor=CORNFLOWER, edgecolor="none", label="Others"),
    ]

    axB.legend(
        handles=legend_handles,
        fontsize=FONTSIZE * 0.7,
        frameon=True,
        loc="lower right"
    )

    axB.set_yticks([])

    axB.set_yticklabels([])

    axB.set_ylabel("Candidate drugs", fontsize=FONTSIZE)

    axB.set_title("")

    cand_labels_sorted = list(map(str, labs_lr))

    scores_sorted = np.asarray(ll_lr, float)

    ord_print = np.argsort(scores_sorted)[::-1]

    top1_name = cand_labels_sorted[int(ord_print[0])] if len(ord_print) > 0 else None

    top2_name = cand_labels_sorted[int(ord_print[1])] if len(ord_print) > 1 else None

    print(f"[Top-center example] top1 drug: {top1_name}")

    print(f"[Top-center example] top2 drug: {top2_name}")

    _plot_topk(axC, topk_list, curves_topk)

    axC.set_title("")

    _plot_roc(axD, roc)

    _plot_pr(axE, pr)

    for ax in [axA, axB, axC, axD, axE]:
        ax.set_title("")

    out_png = os.path.join(OUTDIR, f"{DATASET_NAME}__5panel_summary_v4.png")

    out_svg = os.path.join(OUTDIR, f"{DATASET_NAME}__5panel_summary_v4.svg")

    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")

    plt.savefig(out_svg, bbox_inches="tight")

    plt.show()

    print("Saved:")

    print(" ", out_png)

    print(" ", out_svg)

    print(f"Example chosen (row index): {best_i}")

# ============================================================
# (notebook cell 8) sub-analysis 4: global drug-class clustering permutation test
# ============================================================
def global_clustering_permutation():
    import os

    from pathlib import Path

    import re

    import warnings

    import numpy as np

    import pandas as pd

    import matplotlib.pyplot as plt

    from matplotlib import gridspec

    from matplotlib.colors import ListedColormap

    from matplotlib.patches import Patch, Rectangle

    from scipy.cluster.hierarchy import (
        linkage,
        leaves_list,
        cophenet,
    )

    from scipy.spatial.distance import (
        pdist,
        squareform,
    )

    from tqdm.auto import tqdm

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    INDIR = Path(
        os.path.join(OUTDIR_BASE, "sciplex_u_star_heatmap")
    )

    OUTDIR = INDIR

    MATRIX_CSV = (
        INDIR
        / "drug_level_u_star_heatmap_z.csv"
    )

    DISTANCE_METRIC = "correlation"

    LINKAGE_METHOD = "average"

    N_PERMUTATIONS = 100_000

    SEED = 7

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

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    if not MATRIX_CSV.exists():
        raise FileNotFoundError(
            f"Missing matrix:\n"
            f"{MATRIX_CSV.resolve()}"
        )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_df = pd.read_csv(
        MATRIX_CSV,
        index_col=0,
    )

    matrix_df.index = (
        matrix_df.index
        .astype(str)
    )

    matrix_df.columns = (
        matrix_df.columns
        .astype(str)
    )

    matrix_df = matrix_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if matrix_df.isna().any().any():
        n_missing = int(
            matrix_df
            .isna()
            .sum()
            .sum()
        )

        raise ValueError(
            f"The u* matrix contains {n_missing} "
            "missing or nonnumeric entries."
        )

    matrix_unordered = matrix_df.to_numpy(
        dtype=float,
    )

    drug_names_unordered = (
        matrix_df.index
        .to_numpy(
            dtype=object,
        )
    )

    gene_names_unordered = (
        matrix_df.columns
        .to_numpy(
            dtype=object,
        )
    )

    print(
        "Loaded drug x gene matrix:",
        matrix_unordered.shape,
    )

    valid_drugs = (
        np.all(
            np.isfinite(
                matrix_unordered
            ),
            axis=1,
        )
        & (
            np.std(
                matrix_unordered,
                axis=1,
            )
            > 1e-12
        )
    )

    valid_genes = (
        np.all(
            np.isfinite(
                matrix_unordered
            ),
            axis=0,
        )
        & (
            np.std(
                matrix_unordered,
                axis=0,
            )
            > 1e-12
        )
    )

    if not np.all(
        valid_drugs
    ):
        print(
            "Removing constant/invalid drugs:",
            int(
                np.sum(
                    ~valid_drugs
                )
            ),
        )

    if not np.all(
        valid_genes
    ):
        print(
            "Removing constant/invalid genes:",
            int(
                np.sum(
                    ~valid_genes
                )
            ),
        )

    matrix_unordered = matrix_unordered[
        np.ix_(
            valid_drugs,
            valid_genes,
        )
    ]

    drug_names_unordered = (
        drug_names_unordered[
            valid_drugs
        ]
    )

    gene_names_unordered = (
        gene_names_unordered[
            valid_genes
        ]
    )

    categories_unordered = np.array(
        [
            categorize_drug(
                drug
            )
            for drug
            in drug_names_unordered
        ],
        dtype=object,
    )

    pd.DataFrame({
        "drug":
            drug_names_unordered,
        "category":
            categories_unordered,
        "category_color": [
            category_color(
                category
            )
            for category
            in categories_unordered
        ],
    }).to_csv(
        OUTDIR
        / "drug_categories_coarse.csv",
        index=False,
    )

    (
        drug_order,
        drug_linkage_full,
        drug_distances_full,
    ) = hierarchical_order(
        matrix_unordered,
        cluster_rows=True,
    )

    (
        gene_order,
        gene_linkage_full,
        gene_distances_full,
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

    pd.DataFrame({
        "cluster_order":
            np.arange(
                len(
                    drug_names_clustered
                )
            ),
        "drug":
            drug_names_clustered,
    }).to_csv(
        OUTDIR
        / "clustered_drug_order.csv",
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
        OUTDIR
        / "clustered_gene_order.csv",
        index=False,
    )

    clustered_category_table = pd.DataFrame({
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
    })

    clustered_category_table.to_csv(
        OUTDIR
        / "clustered_drug_order_with_categories.csv",
        index=False,
    )

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

    blocks_df[
        "highlight_rank"
    ] = np.nan

    for block in top_blocks_df.itertuples(
        index=False
    ):
        blocks_df.loc[
            block.original_block_index,
            "highlight_rank",
        ] = int(
            block.highlight_rank
        )

    blocks_df.to_csv(
        OUTDIR
        / "contiguous_category_blocks_ranked.csv",
        index=False,
    )

    top_blocks_df.to_csv(
        OUTDIR
        / "top_5_contiguous_category_blocks.csv",
        index=False,
    )

    print(
        "\nTop contiguous mechanism blocks:"
    )

    if len(
        top_blocks_df
    ):
        print(
            top_blocks_df[
                [
                    "highlight_rank",
                    "category",
                    "start_row",
                    "end_row",
                    "n_rows",
                ]
            ].to_string(
                index=False
            )
        )

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
            "Fewer than two categories passed "
            "the permutation-test filters.\n"
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

    print(
        "\nCategories included in test:"
    )

    print(
        pd.Series(
            test_labels
        )
        .value_counts()
        .to_string()
    )

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

    print(
        "\nObserved global clustering score:",
        observed_global_score,
    )

    print(
        "Cophenetic correlation:",
        cophenetic_correlation,
    )

    rng = np.random.default_rng(
        SEED
    )

    null_global_scores = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    null_category_scores = {
        category: np.empty(
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
        desc="Permuting drug-class labels",
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
            "n_drugs":
                len(
                    test_drug_names
                ),
            "n_categories":
                len(
                    categories_to_test
                ),
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
        }
    ])

    global_statistics_df.to_csv(
        OUTDIR
        / "global_clustering_statistics.csv",
        index=False,
    )

    category_statistic_rows = []

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

        empirical_p = empirical_upper_tail_p(
            observed[
                "cluster_score"
            ],
            null_values,
        )

        z_score = safe_z_score(
            observed[
                "cluster_score"
            ],
            null_values,
        )

        category_statistic_rows.append({
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
                z_score,
            "empirical_p_value":
                empirical_p,
        })

    category_statistics_df = pd.DataFrame(
        category_statistic_rows
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

    category_statistics_df.to_csv(
        OUTDIR
        / "drug_category_clustering_statistics.csv",
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
        safe_category_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            category.lower(),
        ).strip(
            "_"
        )

        null_payload[
            f"null_{safe_category_name}"
        ] = (
            null_category_scores[
                category
            ]
        )

        null_payload[
            f"observed_{safe_category_name}"
        ] = np.array([
            observed_category_results[
                category
            ][
                "cluster_score"
            ]
        ])

    np.savez_compressed(
        OUTDIR
        / "drug_clustering_null_distributions.npz",
        **null_payload,
    )

    categories_present = [
        category
        for category in CATEGORY_ORDER
        if category
        in set(
            categories_clustered
        )
    ]

    categories_present += [
        category
        for category
        in pd.unique(
            categories_clustered
        )
        if category
        not in categories_present
    ]

    category_code_lookup = {
        category: index
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

        # Leave a clean, centered band for the shared legend.
        top=0.885,

        wspace=0.28,
        hspace=0.30,
    )

    heatmap_grid = gridspec.GridSpecFromSubplotSpec(
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
        heatmap_axis.set_yticks([])

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
        heatmap_axis.set_xticks([])

    if SHOW_HEATMAP_AXIS_LABELS:
        heatmap_axis.set_xlabel(
            f"{len(gene_names_clustered):,} clustered genes"
        )

        heatmap_axis.set_ylabel(
            f"{len(drug_names_clustered):,} clustered drugs"
        )

    else:
        heatmap_axis.set_xlabel("")
        heatmap_axis.set_ylabel("")

    heatmap_axis.set_title(
        "Inferred dense intervention programs",
        pad=8,
    )

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

    category_strip_axis.set_xticks([])

    category_strip_axis.set_yticks([])

    category_strip_axis.set_title(
        "Class",
        fontsize=8,
        pad=8,
    )

    for spine in category_strip_axis.spines.values():
        spine.set_linewidth(
            0.6
        )

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
                width=matrix_clustered.shape[1],
                height=block_height,
                fill=False,
                edgecolor=block_color,
                linewidth=BLOCK_LINEWIDTH,
                zorder=30,
                clip_on=False,
            )
        )

        y_center = (
            y_start
            + block_height / 2
        )

        block_label_axis.text(
            0.98,
            y_center,
            (
                f"#{rank} {block.category}\n"
                f"n={int(block.n_rows)}"
            ),
            ha="right",
            va="center",
            fontsize=BLOCK_LABEL_FONTSIZE,
            fontweight="bold",
            color=block_color,
            bbox={
                "facecolor":
                    "white",
                "edgecolor":
                    block_color,
                "linewidth":
                    0.8,
                "alpha":
                    0.94,
                "boxstyle":
                    "round,pad=0.22",
            },
            zorder=40,
        )

    heatmap_colorbar = figure.colorbar(
        heatmap_image,
        cax=heatmap_colorbar_axis,
    )

    heatmap_colorbar.set_label(
        "Row-standardized aggregated $u^*$",
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

    null_global_mean = float(
        np.mean(
            null_global_scores
        )
    )

    global_null_axis.axvline(
        null_global_mean,
        color="#4D4D4D",
        linewidth=1.8,
        linestyle="--",
        label="Null mean",
        zorder=10,
    )

    global_null_axis.set_title(
        "Global drug-class clustering"
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
            f"Observed = {observed_global_score:.4f}\n"
            f"Null mean = {null_global_mean:.4f}\n"
            f"$z$ = {global_z_score:.2f}\n"
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

    bar_y = np.arange(
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
        bar_y,
        category_plot_df[
            "z_score"
        ],
        color=bar_colors,
        edgecolor="none",
        alpha=0.90,
    )

    category_score_axis.set_yticks(
        bar_y
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
        "Category-specific clustering"
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

    for y_position, result in enumerate(
        category_plot_df.itertuples(
            index=False
        )
    ):
        q_label = (
            f"$q$="
            f"{format_probability(result.fdr_q_value)}"
        )

        category_score_axis.text(
            result.z_score
            + (
                label_offset
                if result.z_score >= 0
                else -label_offset
            ),
            y_position,
            q_label,
            va="center",
            ha=(
                "left"
                if result.z_score >= 0
                else "right"
            ),
            fontsize=7.5,
            fontweight=(
                "bold"
                if result.fdr_q_value < 0.05
                else "normal"
            ),
        )

    x_left = min(
        -0.5,
        minimum_z
        - 0.10 * z_span,
    )

    x_right = (
        maximum_z
        + 0.25 * z_span
    )

    category_score_axis.set_xlim(
        x_left,
        x_right,
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

    category_legend_handles = [
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
            category_legend_handles,

        # Center the legend relative to the complete figure.
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

    figure.text(
        0.73,
        0.025,
        (
            "Null generated by permuting drug-class labels "
            "while preserving category sizes."
        ),
        ha="center",
        va="center",
        fontsize=7.5,
        color="0.35",
    )

    png_path = (
        OUTDIR
        / "drug_u_star_clustering_publication_figure_sleek.png"
    )

    pdf_path = (
        OUTDIR
        / "drug_u_star_clustering_publication_figure_sleek.pdf"
    )

    svg_path = (
        OUTDIR
        / "drug_u_star_clustering_publication_figure_sleek.svg"
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
        "\n" + "=" * 76
    )

    print(
        "UNIFIED DRUG-CLASS CLUSTERING ANALYSIS COMPLETE"
    )

    print(
        "=" * 76
    )

    print(
        f"Drugs in heatmap: "
        f"{matrix_clustered.shape[0]}"
    )

    print(
        f"Genes in heatmap: "
        f"{matrix_clustered.shape[1]}"
    )

    print(
        f"Categories tested: "
        f"{len(categories_to_test)}"
    )

    print(
        f"Permutations: "
        f"{N_PERMUTATIONS:,}"
    )

    print(
        f"Global clustering score: "
        f"{observed_global_score:.6f}"
    )

    print(
        f"Global z-score: "
        f"{global_z_score:.3f}"
    )

    print(
        f"Global empirical P-value: "
        f"{global_p_value:.6g}"
    )

    print(
        "\nCategory-specific results:"
    )

    print(
        category_statistics_df[
            [
                "category",
                "n_drugs",
                "observed_cluster_score",
                "z_score",
                "empirical_p_value",
                "fdr_q_value",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved publication figure:"
    )

    print(
        f"  {png_path.resolve()}"
    )

    print(
        f"  {pdf_path.resolve()}"
    )

    print(
        f"  {svg_path.resolve()}"
    )

    if SHOW_FIGURE:
        plt.show()

    else:
        plt.close(
            figure
        )
