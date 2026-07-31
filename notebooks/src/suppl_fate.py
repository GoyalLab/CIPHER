"""Shared helpers for the LARRY fate-commitment supplements (Fig M7 / Fig S17).

Collects the subset of helper functions that are shared, byte-identical, across the
two supplement notebooks AND are fully self-contained (no reliance on notebook-level
config globals). Cell-specific / diverging helper variants are kept inline in the
notebooks so each analysis reproduces 1:1.

Helpers in notebooks/src (not part of the cipher package) -- a notebook-only helper
module for reproducing the supplementary figures.
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
import math
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.special import logsumexp
from scipy.stats import norm

__all__ = ['aggregate_pearson_tables', 'apply_zscore', 'calibrate_scores', 'canonical_model_label', 'cells_for_clone_set', 'clean_term', 'clone_weight_from_n', 'count_nll', 'credible_zero_excluded', 'empirical_log_partition', 'entropy', 'fit_cipher_model', 'fit_startpop_prior_from_counts', 'gaussian_sign_probability', 'kl_div', 'norm_gene', 'predict_calibrator', 'predict_startpop_prior', 'safe_toarray', 'smoothed_observed_fractions', 'soft_normalize', 'softmax_np']


def aggregate_pearson_tables(pearson_metrics):
    df = pearson_metrics.copy()
    df = df[np.isfinite(df["pearson"])].copy()

    per_cell_type = (
        df.groupby(["model", "cell_type"], as_index=False)
        .agg(
            pearson_mean=("pearson", "mean"),
            pearson_sd=("pearson", "std"),
            n_estimates=("pearson", "count"),
            n_clones_mean=("n_clones", "mean"),
        )
    )

    per_cell_type["pearson_sd"] = per_cell_type["pearson_sd"].fillna(0.0)
    per_cell_type["pearson_sem_over_folds_or_nulls"] = (
        per_cell_type["pearson_sd"] /
        np.sqrt(np.maximum(per_cell_type["n_estimates"], 1))
    )

    model_mean = (
        per_cell_type.groupby("model", as_index=False)
        .agg(
            mean_pearson_across_cell_types=("pearson_mean", "mean"),
            sd_across_cell_types=("pearson_mean", "std"),
            n_cell_types=("cell_type", "nunique"),
        )
    )

    model_mean["sd_across_cell_types"] = model_mean["sd_across_cell_types"].fillna(0.0)
    model_mean["sem_across_cell_types"] = (
        model_mean["sd_across_cell_types"] /
        np.sqrt(np.maximum(model_mean["n_cell_types"], 1))
    )

    return per_cell_type, model_mean


def apply_zscore(X, mu, sd):
    return (X - mu) / sd


def calibrate_scores(clf, scores):
    scores = np.asarray(scores).reshape(-1, 1)
    if clf is None:
        return np.full(scores.shape[0], np.nan)
    return clf.predict_proba(scores)[:, 1]


def canonical_model_label(x):
    x0 = str(x)
    xl = x0.lower()

    if "lfc" in xl or "terminal" in xl:
        return "terminal-vs-undiff LFC"

    if "null" in xl or "shuffle" in xl:
        return "startpop-preserving null"

    if "starting_population_only" in xl or "starting-pop" in xl or "starting_pop" in xl:
        return "starting-pop only"

    if "cipher" in xl or "bayes" in xl or "h0" in xl:
        return "CIPHER"

    return x0


def cells_for_clone_set(cell_to_clone, clone_ids, mask):
    clone_ids = np.asarray(clone_ids, dtype=int)
    return np.where(mask & np.isin(cell_to_clone, clone_ids))[0]


def clean_term(t):
    return str(t).replace("_", " ").replace("-", " ").strip()


def clone_weight_from_n(n_early, mode="none", saturating_n0=5.0):
    n = np.asarray(n_early, dtype=float)

    if mode == "none":
        return np.ones_like(n, dtype=float)

    if mode == "sqrt":
        return np.sqrt(np.maximum(n, 1.0))

    if mode == "raw":
        return np.maximum(n, 1.0)

    if mode == "saturating":
        return n / np.maximum(n + float(saturating_n0), 1e-12)

    raise ValueError("clone weight mode must be: none, sqrt, raw, saturating")


def count_nll(counts, probs, eps=1e-12):
    counts = np.asarray(counts, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return -float(np.sum(counts * np.log(np.clip(probs, eps, 1.0))))


def credible_zero_excluded(mu, std, level=0.95):
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)

    alpha = 1.0 - float(level)
    z = norm.ppf(1.0 - alpha / 2.0)

    lo = mu - z * std
    hi = mu + z * std

    return (lo > 0) | (hi < 0), lo, hi


def empirical_log_partition(X_baseline, U):
    """
    A_f = logmeanexp_{x0~p0}(x0^T u_f)
    """
    X_baseline = np.asarray(X_baseline, dtype=float)
    U = np.asarray(U, dtype=float)

    scores0 = X_baseline @ U.T
    return logsumexp(scores0, axis=0) - np.log(scores0.shape[0])


def entropy(P, eps=1e-12):
    P = np.asarray(P, dtype=float)
    return -np.sum(P * np.log(np.clip(P, eps, 1.0)), axis=1)


def fit_cipher_model(Xz_train, y_train, selected_fates, Sigma, evals, evecs, use_prior=False):
    W = []
    penalties = []
    log_priors = []
    deltas = []

    eps = 1e-12

    for fate in selected_fates:
        pos = y_train == fate
        neg = y_train != fate

        if pos.sum() == 0 or neg.sum() == 0:
            raise RuntimeError(f"Missing positives/negatives for {fate}")

        delta = Xz_train[pos].mean(axis=0) - Xz_train[neg].mean(axis=0)

        # CIPHER force:
        # u = Sigma^{-1} Delta
        u = evecs @ ((evecs.T @ delta) / evals)

        penalty = 0.5 * float(u @ Sigma @ u)

        if use_prior:
            prior = max(float(pos.mean()), eps)
            log_prior = np.log(prior)
        else:
            log_prior = 0.0

        W.append(u)
        penalties.append(penalty)
        log_priors.append(log_prior)
        deltas.append(delta)

    return {
        "model_type": "cipher",
        "W": np.asarray(W),
        "penalty": np.asarray(penalties),
        "log_prior": np.asarray(log_priors),
        "delta_z": np.asarray(deltas),
    }


def fit_startpop_prior_from_counts(Ctrain, start_train, alpha=2.0, eps=1e-12):
    """
    Fits P(fate | starting population) from terminal fate counts.
    """
    Ctrain = np.asarray(Ctrain, dtype=float)
    start_train = np.asarray(start_train).astype(str)

    global_counts = Ctrain.sum(axis=0) + eps
    global_p = global_counts / global_counts.sum()

    table = {}

    for s in np.unique(start_train):
        idx = np.where(start_train == s)[0]
        counts_s = Ctrain[idx].sum(axis=0)
        p = (counts_s + alpha * global_p) / np.maximum(counts_s.sum() + alpha, eps)
        p = p / np.maximum(p.sum(), eps)
        table[s] = p

    return {
        "global_p": global_p,
        "table": table,
    }


def gaussian_sign_probability(mu, std):
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)

    p_pos = 1.0 - norm.cdf((0.0 - mu) / (std + 1e-12))
    p_neg = norm.cdf((0.0 - mu) / (std + 1e-12))

    return p_pos, p_neg


def kl_div(P, Q, eps=1e-12):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    return np.sum(P * (np.log(np.clip(P, eps, 1.0)) - np.log(np.clip(Q, eps, 1.0))), axis=1)


def norm_gene(g):
    if pd.isna(g):
        return ""
    return str(g).strip().upper()


def predict_calibrator(clf, scores):
    scores = np.asarray(scores).reshape(-1, 1)
    if clf is None:
        return np.full(scores.shape[0], np.nan)
    return clf.predict_proba(scores)[:, 1]


def predict_startpop_prior(start_values, prior_model):
    start_values = np.asarray(start_values).astype(str)

    P = []
    for s in start_values:
        P.append(prior_model["table"].get(s, prior_model["global_p"]))

    return np.vstack(P)


def safe_toarray(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def smoothed_observed_fractions(counts, prior=None, alpha=5.0, eps=1e-12):
    counts = np.asarray(counts, dtype=float)

    if prior is None:
        prior = counts.sum(axis=0) + eps
        prior = prior / prior.sum()

    prior = np.asarray(prior, dtype=float)
    prior = prior / np.maximum(prior.sum(), eps)

    denom = counts.sum(axis=1, keepdims=True) + alpha
    return (counts + alpha * prior[None, :]) / np.maximum(denom, eps)


def soft_normalize(P, eps=1e-12):
    P = np.asarray(P, dtype=float)
    return P / np.maximum(P.sum(axis=1, keepdims=True), eps)


def softmax_np(logits, eps=1e-12):
    logits = np.asarray(logits, dtype=float)
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.maximum(ez.sum(axis=1, keepdims=True), eps)
