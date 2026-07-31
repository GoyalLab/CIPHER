"""Per-notebook run module for ``suppl/fate_commitment_bayesian_figM7_S17.ipynb`` (Fig M7 / Fig S17).

Main-flow orchestration for the analytic Bayesian CIPHER LARRY fate-commitment
supplement. Each function is one main-flow cell wrapped in a ``def`` (same variables,
same plt/savefig calls, same logic). The cells share a single kernel namespace, so
cross-cell state is persisted here via module-level ``global`` declarations; calling the
functions in notebook order reproduces the single-namespace execution exactly. Notebook
config (``SUPPL``, ``OUT_BASE``, ...) is read as MODULE GLOBALS, injected at runtime by
the notebook's injection cell.

Helpers in notebooks/src (not part of the cipher package).
"""
from src.suppl_fate import *

# same library imports the cluster module (src.suppl_fate) uses, plus seaborn (used by the
# cross-validation section)
import os, re, glob, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Patch
from matplotlib import gridspec
import seaborn as sns
import scipy
from scipy.sparse import issparse, csr_matrix
from scipy.special import logsumexp
from scipy.stats import wilcoxon, ttest_rel, ks_2samp, mannwhitneyu, pearsonr, spearmanr, norm
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


def analytic_bayesian_cipher_complete():
    global os, gzip, warnings, np, pd, plt, sns, Counter
    global mmread, issparse, pearsonr, spearmanr, norm, minimize_scalar, StratifiedKFold, confusion_matrix
    global OUTDIR, COUNTS_PATH, GENES_PATH, CLONE_PATH, META_PATH, TIME_COL, CELLTYPE_COL, START_COL
    global WELL_COL, EARLY_TIME, EARLY_CELLTYPE, EARLY_WELL, RESTRICT_STARTING_POPULATION, TERMINAL_TIME, TERMINAL_WELL, EXCLUDE_FATES
    global MANUAL_SELECTED_FATES, MAX_FATES, MIN_CLONES_WITH_FATE, MIN_TERMINAL_CELLS_PER_CLONE, MIN_EARLY_CELLS_PER_CLONE, MIN_TOTAL_CELLS_PER_CLONE, MIN_SELECTED_FATE_COVERAGE, MIN_SELECTED_TERMINAL_CELLS
    global N_VAR_GENES, MAX_COV_CELLS, RIDGE, COV_SHRINK_TO_DIAG, USE_FATE_PRIOR, POSTERIOR_TAU2, H0_MIN_SCALE, USE_POSTERIOR_UNCERTAINTY_PENALTY
    global POSTERIOR_ACTIVITY_THRESHOLD, CALIBRATE_TEMPERATURE, TEMP_MIN, TEMP_MAX, N_NULLS, USE_STARTPOP_PRESERVING_NULL, N_SPLITS, SEED
    global TOP_FORCE_GENES_PER_DIRECTION, TOP_GENES_PER_FATE_FOR_HEATMAP, rng, check_file, safe_name, softmax_logits, js_div, cosine_similarity
    global safe_corr, safe_r2, r2_from_pred, get_cell_to_clone, get_cells_x_genes, zscore_train, select_hvgs_sparse, make_covariance
    global weighted_mean, clone_mean_matrix, shuffle_YC_within_groups, fit_temperature_from_counts, gaussian_activity_probability, weighted_delta_and_H0_scale, analytic_bayesian_cipher_posterior_H0_equals_scaleSigma, make_composition_cipher_bayes_model
    global get_logits, score_composition_cipher, fit_startpop_composition_baseline, score_startpop_composition_baseline, add_prediction_columns, add_composition_errors, summarize_predictions, p
    global counts, f, gene_names, clone_mat, meta, cell_to_clone, has_clone, cell_fates
    global early_mask, terminal_mask, early_all_idx, early_cloned_mask, terminal_cloned_mask, early_cloned_idx, terminal_cloned_idx, candidate_records
    global global_fate_counts, global_fate_clone_counts, clone_id, cells, early_cells, terminal_cells, fates, vc
    global terminal_counts_dict, c, starts, dominant_start, dominant_start_frac, candidate_table, fate_summary, selected_fates
    global clone_table, fate, s, selected_count_cols, obs_frac_cols, Y_all, dominant_idx, clone_table_save
    global fig, axes, tab, hvg_idx, gene_vars, hvg_genes, cov_idx, Xcov_raw
    global mu_ref, sd_ref, Xcov, Sigma, evals, evecs, X_clones_all, strat_y
    global min_class_n, n_splits, splitter, clone_to_obs, clone_to_counts, clone_to_start, all_pred_rows, all_null_rows_for_error_plots
    global summary_rows, force_rows, temperature_rows, posterior_fit_rows, fold, train_pos, test_pos, train_clones
    global test_clones, Xtrain, train_clone_ids_used, n_train_early, Xtest, test_clone_ids_used, n_test_early, Ytrain
    global Ctrain, Ytest, Ctest, start_train, start_test, true_dom_test, base, j
    global cipher_model, _, train_logits, T_cipher, raw_scores, logits, Ptest, pred_df
    global u, delta, yhat, std, z, pip, sign_conf, p_pos
    global p_neg, ci_lo, ci_hi, zero_exc, ranking_sets, direction, idxs, rank
    global gi, sp_model, raw_sp, logits_sp, P_sp, sp_df, null_id, Ytrain_null
    global Ctrain_null, null_name, perm, null_model, null_train_logits, T_null, raw_null, logits_null
    global P_null, null_df, predictions, summary_metrics, force_df, temperature_df, posterior_fit_df, null_clone_errors
    global composition_summary, per_fate_summary, posterior_fit_summary, p_rows, null_models, metric, real_vals, null_vals
    global real_mean, p_emp, pvals, cipher_pred, model_label_map, n_fates, ncols, nrows
    global ax, x, y, r, rho, r2, k, obs_cols
    global pred_cols, obs_heat, pred_heat, cipher_error_compact, error_plot_df, sp_error, perf, point_df
    global handles, labels, uniq_h, uniq_l, h, l, mx, cm
    global cm_norm, cipher_force, mean_force, top_genes, sub, heat, pip_heat
    # ============================================================
    # CIPHER-LARRY: terminal clone fate COMPOSITION
    # COMPLETE STANDALONE VERSION WITH ANALYTIC BAYESIAN POSTERIOR
    # ============================================================
    #
    # Model change:
    #
    #   Old CIPHER:
    #       u_f = Sigma^{-1} Delta_f
    #
    #   New Bayesian CIPHER:
    #       Delta_f = Sigma u_f + eps
    #       u_f ~ N(0, tau2 I)
    #       eps ~ N(0, H0_f)
    #
    # Here:
    #       H0_f = alpha_f * Sigma
    #
    # where alpha_f is the weighted sample-mean noise scale:
    #
    #       alpha_f = sum_i wp_i^2 + sum_i wn_i^2
    #
    # with wp/wn normalized fate-vs-rest clone-composition weights.
    #
    # For uniform weights this reduces to:
    #
    #       H0 = (1/n_pos + 1/n_neg) Sigma
    #
    # This is the clone-composition analogue of the second block's
    # analytic Gaussian posterior with H = H0.
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
    from scipy.stats import pearsonr, spearmanr, norm
    from scipy.optimize import minimize_scalar
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix

    warnings.filterwarnings("ignore")


    # ============================================================
    # CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_clone_fate_composition_analytic_bayes_H0_complete")
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
    MAX_FATES = 7

    MIN_CLONES_WITH_FATE = 8
    MIN_TERMINAL_CELLS_PER_CLONE = 5
    MIN_EARLY_CELLS_PER_CLONE = 1
    MIN_TOTAL_CELLS_PER_CLONE = 8

    MIN_SELECTED_FATE_COVERAGE = 0.75
    MIN_SELECTED_TERMINAL_CELLS = 5

    # Important:
    # Full eigendecomposition scales badly with gene number.
    # 3000-6000 is usually realistic on a laptop.
    # Increase only if you have enough RAM/CPU.
    N_VAR_GENES = 500
    MAX_COV_CELLS = 50000

    RIDGE = 1e-8
    COV_SHRINK_TO_DIAG = 0.0

    USE_FATE_PRIOR = False

    # Bayesian posterior
    POSTERIOR_TAU2 = 1.0
    H0_MIN_SCALE = 1e-8
    USE_POSTERIOR_UNCERTAINTY_PENALTY = True
    POSTERIOR_ACTIVITY_THRESHOLD = None

    # Temperature calibration
    CALIBRATE_TEMPERATURE = True
    TEMP_MIN = 0.1
    TEMP_MAX = 100.0

    # Nulls
    N_NULLS = 100
    USE_STARTPOP_PRESERVING_NULL = True

    N_SPLITS = 5
    SEED = 0

    TOP_FORCE_GENES_PER_DIRECTION = 50
    TOP_GENES_PER_FATE_FOR_HEATMAP = 12

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    plt.rcParams.update({"font.size": 13})
    sns.set_context("talk")


    # ============================================================
    # BASIC HELPERS
    # ============================================================

    def check_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}\nCurrent working directory: {os.getcwd()}")
        print(f"[OK] Found {path}")


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
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)

        num = np.sum(P * Q, axis=1)
        den = np.sqrt(np.sum(P * P, axis=1)) * np.sqrt(np.sum(Q * Q, axis=1))

        return num / np.maximum(den, eps)


    def safe_corr(x, y, method="pearson"):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) < 3:
            return np.nan

        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan

        try:
            if method == "pearson":
                return pearsonr(x, y)[0]
            return spearmanr(x, y)[0]
        except Exception:
            return np.nan


    def safe_r2(y_true, y_pred, eps=1e-12):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)

        if ss_tot < eps:
            return np.nan

        return 1.0 - ss_res / ss_tot


    def r2_from_pred(y, yhat, eps=1e-12):
        y = np.asarray(y, float).reshape(-1)
        yhat = np.asarray(yhat, float).reshape(-1)

        return 1.0 - np.sum((y - yhat) ** 2) / (np.sum(y ** 2) + eps)


    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()

        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row

        return cell_to_clone


    def get_cells_x_genes(counts, cell_idx, gene_idx):
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)


    def zscore_train(X):
        X = np.asarray(X, dtype=np.float64)

        mu = X.mean(axis=0)
        sd = X.std(axis=0)

        sd[sd < 1e-6] = 1.0

        return mu, sd


    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        X = counts[:, cell_idx]

        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means ** 2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]

        n_keep = int(min(n_var_genes, len(valid_idx)))

        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_keep:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_


    def make_covariance(X):
        X = np.asarray(X, dtype=np.float64)

        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D

        scale = np.mean(np.diag(Sigma)) + 1e-12
        Sigma = Sigma + RIDGE * scale * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)


    def weighted_mean(X, w, eps=1e-12):
        X = np.asarray(X, dtype=np.float64)
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


    def shuffle_YC_within_groups(Y, C, groups):
        Y = np.asarray(Y).copy()
        C = np.asarray(C).copy()
        groups = np.asarray(groups).astype(str)

        Yout = Y.copy()
        Cout = C.copy()

        for g in np.unique(groups):
            idx = np.where(groups == g)[0]

            if len(idx) > 1:
                perm_idx = idx[rng.permutation(len(idx))]
                Yout[idx] = Y[perm_idx]
                Cout[idx] = C[perm_idx]

        return Yout, Cout


    def fit_temperature_from_counts(logits, counts, temp_min=0.25, temp_max=100.0):
        logits = np.asarray(logits, dtype=float)
        counts = np.asarray(counts, dtype=float)

        if logits.shape != counts.shape:
            raise ValueError("logits and counts must have same shape.")

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
    # BAYESIAN POSTERIOR HELPERS
    # ============================================================

    def gaussian_activity_probability(mu, std, effect_threshold=None):
        mu = np.asarray(mu, dtype=float)
        std = np.asarray(std, dtype=float)

        if effect_threshold is None:
            effect_threshold = float(np.median(std))

        z_upper = ( effect_threshold - mu) / (std + 1e-12)
        z_lower = (-effect_threshold - mu) / (std + 1e-12)

        pip = 1.0 - (norm.cdf(z_upper) - norm.cdf(z_lower))

        return pip, float(effect_threshold)


    def weighted_delta_and_H0_scale(X, w_pos, w_neg, eps=1e-12):
        X = np.asarray(X, dtype=np.float64)
        w_pos = np.asarray(w_pos, dtype=np.float64).reshape(-1)
        w_neg = np.asarray(w_neg, dtype=np.float64).reshape(-1)

        sp = float(w_pos.sum())
        sn = float(w_neg.sum())

        if sp <= eps or sn <= eps:
            delta = np.zeros(X.shape[1], dtype=np.float64)
            h0_scale = 1.0
            n_eff_pos = 0.0
            n_eff_neg = 0.0

            return delta, h0_scale, n_eff_pos, n_eff_neg

        wp = w_pos / sp
        wn = w_neg / sn

        mu_pos = wp @ X
        mu_neg = wn @ X

        delta = mu_pos - mu_neg

        sum_wp2 = float(np.sum(wp ** 2))
        sum_wn2 = float(np.sum(wn ** 2))

        h0_scale = max(sum_wp2 + sum_wn2, H0_MIN_SCALE)

        n_eff_pos = 1.0 / max(sum_wp2, eps)
        n_eff_neg = 1.0 / max(sum_wn2, eps)

        return delta, h0_scale, n_eff_pos, n_eff_neg


    def analytic_bayesian_cipher_posterior_H0_equals_scaleSigma(
        delta,
        Sigma,
        evals,
        evecs,
        h0_scale,
        tau2=1.0,
        use_uncertainty_penalty=True,
        activity_threshold=None,
        eps=1e-12,
    ):
        """
    Analytic posterior for:

        delta = Sigma u + eps
        u ~ N(0, tau2 I)
        eps ~ N(0, H0)
        H0 = h0_scale * Sigma

    With Sigma = V diag(lambda) V.T:

        posterior mean eig coeff:
            u_k = delta_k / (lambda_k + h0_scale/tau2)

        posterior covariance eig value:
            cov_k = 1 / (lambda_k/h0_scale + 1/tau2)
    """

        delta = np.asarray(delta, dtype=np.float64).reshape(-1)
        Sigma = np.asarray(Sigma, dtype=np.float64)
        evals = np.asarray(evals, dtype=np.float64).reshape(-1)
        evecs = np.asarray(evecs, dtype=np.float64)

        tau2 = float(tau2)
        h0_scale = max(float(h0_scale), H0_MIN_SCALE)

        if tau2 <= 0:
            raise ValueError("tau2 must be positive.")

        lam = np.maximum(evals, eps)
        delta_eig = evecs.T @ delta

        denom_mu = lam + h0_scale / tau2
        u_eig = delta_eig / np.maximum(denom_mu, eps)
        mu_u = evecs @ u_eig

        cov_eig = 1.0 / np.maximum(lam / h0_scale + 1.0 / tau2, eps)
        std_u = np.sqrt(np.maximum((evecs ** 2) @ cov_eig, 0.0))

        yhat_eig = lam * u_eig
        yhat = evecs @ yhat_eig

        r2 = r2_from_pred(delta, yhat)

        penalty_mean = 0.5 * float(np.sum(lam * (u_eig ** 2)))
        penalty_uncertainty = 0.5 * float(np.sum(lam * cov_eig))

        if use_uncertainty_penalty:
            penalty = penalty_mean + penalty_uncertainty
        else:
            penalty = penalty_mean

        C_eig = h0_scale * lam + tau2 * (lam ** 2)
        C_eig = np.maximum(C_eig, eps)

        logdetC = float(np.sum(np.log(C_eig)))
        quad = float(np.sum((delta_eig ** 2) / C_eig))
        d = len(delta)

        log_marginal = -0.5 * (d * np.log(2.0 * np.pi) + logdetC + quad)

        pip, used_activity_threshold = gaussian_activity_probability(
            mu_u,
            std_u,
            effect_threshold=activity_threshold,
        )

        p_pos, p_neg = gaussian_sign_probability(mu_u, std_u)
        sign_conf = np.maximum(p_pos, p_neg)

        zero_exc, ci95_lo, ci95_hi = credible_zero_excluded(mu_u, std_u, level=0.95)

        z = mu_u / (std_u + eps)

        return {
            "mu": mu_u,
            "std": std_u,
            "z": z,
            "pip": pip,
            "p_pos": p_pos,
            "p_neg": p_neg,
            "sign_conf": sign_conf,
            "ci95_lo": ci95_lo,
            "ci95_hi": ci95_hi,
            "zero_excluded": zero_exc.astype(bool),

            "u_eig": u_eig,
            "cov_eig": cov_eig,

            "yhat": yhat,
            "r2": r2,

            "penalty": penalty,
            "penalty_mean": penalty_mean,
            "penalty_uncertainty": penalty_uncertainty,

            "h0_scale": h0_scale,
            "tau2": tau2,
            "activity_threshold": used_activity_threshold,
            "log_marginal": log_marginal,
        }


    # ============================================================
    # MODEL FUNCTIONS
    # ============================================================

    def make_composition_cipher_bayes_model(
        Xtrain_clone,
        Ytrain,
        selected_fates,
        evals,
        evecs,
        Sigma,
        tau2=1.0,
        use_fate_prior=False,
        use_uncertainty_penalty=True,
        activity_threshold=None,
    ):
        Xtrain_clone = np.asarray(Xtrain_clone, dtype=np.float64)
        Ytrain = np.asarray(Ytrain, dtype=np.float64)

        U = []
        DELTAS = []
        YHATS = []

        posterior_std = []
        posterior_z = []
        posterior_pip = []
        posterior_sign_conf = []
        posterior_p_pos = []
        posterior_p_neg = []
        posterior_ci95_lo = []
        posterior_ci95_hi = []
        posterior_zero_excluded = []

        penalties = []
        penalty_mean = []
        penalty_uncertainty = []

        h0_scales = []
        n_eff_pos_list = []
        n_eff_neg_list = []
        r2_list = []
        log_marginals = []
        activity_thresholds = []
        log_priors = []

        eps = 1e-12

        for j, fate in enumerate(selected_fates):
            w_pos = Ytrain[:, j].copy()
            w_neg = 1.0 - w_pos

            delta, h0_scale, n_eff_pos, n_eff_neg = weighted_delta_and_H0_scale(
                X=Xtrain_clone,
                w_pos=w_pos,
                w_neg=w_neg,
            )

            post = analytic_bayesian_cipher_posterior_H0_equals_scaleSigma(
                delta=delta,
                Sigma=Sigma,
                evals=evals,
                evecs=evecs,
                h0_scale=h0_scale,
                tau2=tau2,
                use_uncertainty_penalty=use_uncertainty_penalty,
                activity_threshold=activity_threshold,
            )

            if use_fate_prior:
                prior = max(float(Ytrain[:, j].mean()), eps)
                log_prior = np.log(prior)
            else:
                log_prior = 0.0

            U.append(post["mu"])
            DELTAS.append(delta)
            YHATS.append(post["yhat"])

            posterior_std.append(post["std"])
            posterior_z.append(post["z"])
            posterior_pip.append(post["pip"])
            posterior_sign_conf.append(post["sign_conf"])
            posterior_p_pos.append(post["p_pos"])
            posterior_p_neg.append(post["p_neg"])
            posterior_ci95_lo.append(post["ci95_lo"])
            posterior_ci95_hi.append(post["ci95_hi"])
            posterior_zero_excluded.append(post["zero_excluded"])

            penalties.append(post["penalty"])
            penalty_mean.append(post["penalty_mean"])
            penalty_uncertainty.append(post["penalty_uncertainty"])

            h0_scales.append(post["h0_scale"])
            n_eff_pos_list.append(n_eff_pos)
            n_eff_neg_list.append(n_eff_neg)
            r2_list.append(post["r2"])
            log_marginals.append(post["log_marginal"])
            activity_thresholds.append(post["activity_threshold"])
            log_priors.append(log_prior)

        return {
            "U": np.asarray(U),
            "DELTAS": np.asarray(DELTAS),
            "YHAT_DELTAS": np.asarray(YHATS),

            "posterior_std": np.asarray(posterior_std),
            "posterior_z": np.asarray(posterior_z),
            "posterior_pip": np.asarray(posterior_pip),
            "posterior_sign_conf": np.asarray(posterior_sign_conf),
            "posterior_p_pos": np.asarray(posterior_p_pos),
            "posterior_p_neg": np.asarray(posterior_p_neg),
            "posterior_ci95_lo": np.asarray(posterior_ci95_lo),
            "posterior_ci95_hi": np.asarray(posterior_ci95_hi),
            "posterior_zero_excluded": np.asarray(posterior_zero_excluded),

            "penalty": np.asarray(penalties),
            "penalty_mean": np.asarray(penalty_mean),
            "penalty_uncertainty": np.asarray(penalty_uncertainty),

            "h0_scale": np.asarray(h0_scales),
            "n_eff_pos": np.asarray(n_eff_pos_list),
            "n_eff_neg": np.asarray(n_eff_neg_list),
            "posterior_delta_r2": np.asarray(r2_list),
            "log_marginal": np.asarray(log_marginals),
            "activity_threshold": np.asarray(activity_thresholds),

            "log_prior": np.asarray(log_priors),
            "tau2": float(tau2),
            "temperature": 1.0,
        }


    def get_logits(X, model):
        X = np.asarray(X, dtype=np.float64)

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
        global_p = global_p / np.maximum(global_p.sum(), 1e-12)

        table = {}

        for s in np.unique(start_train):
            idx = np.where(start_train == s)[0]

            if len(idx) == 0:
                continue

            p = (Ytrain[idx].sum(axis=0) + alpha * global_p) / (len(idx) + alpha)
            p = p / np.maximum(p.sum(), 1e-12)

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
    # LOAD DATA
    # ============================================================

    for p in [COUNTS_PATH, GENES_PATH, CLONE_PATH, META_PATH]:
        check_file(p)

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
    print(f"Using {len(hvg_idx):,} HVGs for Sigma/eigendecomposition.")

    Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx)
    mu_ref, sd_ref = zscore_train(Xcov_raw)
    Xcov = apply_zscore(Xcov_raw, mu_ref, sd_ref)

    print("Computing covariance...")
    Sigma = make_covariance(Xcov)

    print("Computing eigendecomposition of Sigma...")
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-10)

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

    all_pred_rows = []
    all_null_rows_for_error_plots = []
    summary_rows = []
    force_rows = []
    temperature_rows = []
    posterior_fit_rows = []

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

        if Xtrain.shape[0] == 0 or Xtest.shape[0] == 0:
            print(f"[skip] Fold {fold} has no usable train/test clone means.")
            continue

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

        # --------------------------------------------------------
        # Bayesian CIPHER model
        # --------------------------------------------------------
        cipher_model = make_composition_cipher_bayes_model(
            Xtrain_clone=Xtrain,
            Ytrain=Ytrain,
            selected_fates=selected_fates,
            evals=evals,
            evecs=evecs,
            Sigma=Sigma,
            tau2=POSTERIOR_TAU2,
            use_fate_prior=USE_FATE_PRIOR,
            use_uncertainty_penalty=USE_POSTERIOR_UNCERTAINTY_PENALTY,
            activity_threshold=POSTERIOR_ACTIVITY_THRESHOLD,
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
            "model": "cipher_bayes_H0",
            "null_id": None,
            "temperature": T_cipher,
            "tau2": POSTERIOR_TAU2,
        })

        raw_scores, logits, Ptest = score_composition_cipher(
            Xtest,
            cipher_model,
            temperature=T_cipher,
        )

        pred_df = add_prediction_columns(
            base,
            raw_scores,
            logits,
            Ptest,
            selected_fates,
            "cipher_bayes_H0",
            temperature=T_cipher,
        )

        pred_df = add_composition_errors(pred_df, selected_fates)

        all_pred_rows.append(pred_df)
        summary_rows.append(summarize_predictions(pred_df, selected_fates, "cipher_bayes_H0", fold))

        for j, fate in enumerate(selected_fates):
            posterior_fit_rows.append({
                "fold": fold,
                "model": "cipher_bayes_H0",
                "fate": fate,
                "tau2": POSTERIOR_TAU2,
                "temperature": float(T_cipher),
                "h0_scale": float(cipher_model["h0_scale"][j]),
                "n_eff_pos": float(cipher_model["n_eff_pos"][j]),
                "n_eff_neg": float(cipher_model["n_eff_neg"][j]),
                "posterior_delta_r2": float(cipher_model["posterior_delta_r2"][j]),
                "log_marginal": float(cipher_model["log_marginal"][j]),
                "penalty": float(cipher_model["penalty"][j]),
                "penalty_mean": float(cipher_model["penalty_mean"][j]),
                "penalty_uncertainty": float(cipher_model["penalty_uncertainty"][j]),
                "activity_threshold": float(cipher_model["activity_threshold"][j]),
                "log_prior": float(cipher_model["log_prior"][j]),
            })

        for j, fate in enumerate(selected_fates):
            u = cipher_model["U"][j]
            delta = cipher_model["DELTAS"][j]
            yhat = cipher_model["YHAT_DELTAS"][j]
            std = cipher_model["posterior_std"][j]
            z = cipher_model["posterior_z"][j]
            pip = cipher_model["posterior_pip"][j]
            sign_conf = cipher_model["posterior_sign_conf"][j]
            p_pos = cipher_model["posterior_p_pos"][j]
            p_neg = cipher_model["posterior_p_neg"][j]
            ci_lo = cipher_model["posterior_ci95_lo"][j]
            ci_hi = cipher_model["posterior_ci95_hi"][j]
            zero_exc = cipher_model["posterior_zero_excluded"][j]

            ranking_sets = [
                ("positive", np.argsort(u)[::-1][:TOP_FORCE_GENES_PER_DIRECTION]),
                ("negative", np.argsort(u)[:TOP_FORCE_GENES_PER_DIRECTION]),
                ("high_abs_z", np.argsort(np.abs(z))[::-1][:TOP_FORCE_GENES_PER_DIRECTION]),
                ("high_pip", np.argsort(pip)[::-1][:TOP_FORCE_GENES_PER_DIRECTION]),
            ]

            for direction, idxs in ranking_sets:
                for rank, gi in enumerate(idxs, start=1):
                    force_rows.append({
                        "fold": fold,
                        "model": "cipher_bayes_H0",
                        "fate": fate,
                        "direction": direction,
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),

                        "posterior_mu_u": float(u[gi]),
                        "posterior_std_u": float(std[gi]),
                        "posterior_z_u": float(z[gi]),
                        "posterior_pip": float(pip[gi]),
                        "posterior_sign_conf": float(sign_conf[gi]),
                        "posterior_p_pos": float(p_pos[gi]),
                        "posterior_p_neg": float(p_neg[gi]),
                        "ci95_lo": float(ci_lo[gi]),
                        "ci95_hi": float(ci_hi[gi]),
                        "zero_excluded_95": int(zero_exc[gi]),

                        "delta_weighted_composition": float(delta[gi]),
                        "delta_posterior_yhat": float(yhat[gi]),

                        "h0_scale": float(cipher_model["h0_scale"][j]),
                        "n_eff_pos": float(cipher_model["n_eff_pos"][j]),
                        "n_eff_neg": float(cipher_model["n_eff_neg"][j]),
                        "posterior_delta_r2": float(cipher_model["posterior_delta_r2"][j]),

                        "penalty": float(cipher_model["penalty"][j]),
                        "penalty_mean": float(cipher_model["penalty_mean"][j]),
                        "penalty_uncertainty": float(cipher_model["penalty_uncertainty"][j]),
                        "log_prior": float(cipher_model["log_prior"][j]),
                        "log_marginal": float(cipher_model["log_marginal"][j]),
                        "temperature": float(T_cipher),
                        "tau2": float(POSTERIOR_TAU2),
                    })

        # --------------------------------------------------------
        # Starting-population-only baseline
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Nulls
        # --------------------------------------------------------
        for null_id in range(N_NULLS):
            if USE_STARTPOP_PRESERVING_NULL and START_COL in meta.columns and RESTRICT_STARTING_POPULATION is None:
                Ytrain_null, Ctrain_null = shuffle_YC_within_groups(Ytrain, Ctrain, start_train)
                null_name = "startpop_preserving_null_bayes_H0"
            else:
                perm = rng.permutation(Ytrain.shape[0])
                Ytrain_null = Ytrain[perm]
                Ctrain_null = Ctrain[perm]
                null_name = "shuffled_null_bayes_H0"

            null_model = make_composition_cipher_bayes_model(
                Xtrain_clone=Xtrain,
                Ytrain=Ytrain_null,
                selected_fates=selected_fates,
                evals=evals,
                evecs=evecs,
                Sigma=Sigma,
                tau2=POSTERIOR_TAU2,
                use_fate_prior=USE_FATE_PRIOR,
                use_uncertainty_penalty=USE_POSTERIOR_UNCERTAINTY_PENALTY,
                activity_threshold=POSTERIOR_ACTIVITY_THRESHOLD,
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
                "tau2": POSTERIOR_TAU2,
            })

            raw_null, logits_null, P_null = score_composition_cipher(
                Xtest,
                null_model,
                temperature=T_null,
            )

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


    # ============================================================
    # SAVE OUTPUT TABLES
    # ============================================================

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    summary_metrics = pd.concat(summary_rows, ignore_index=True)
    force_df = pd.DataFrame(force_rows)
    temperature_df = pd.DataFrame(temperature_rows)
    posterior_fit_df = pd.DataFrame(posterior_fit_rows)

    if len(all_null_rows_for_error_plots) > 0:
        null_clone_errors = pd.concat(all_null_rows_for_error_plots, ignore_index=True)
    else:
        null_clone_errors = pd.DataFrame()

    predictions.to_csv(
        os.path.join(OUTDIR, "clone_composition_predictions_bayes_H0.csv"),
        index=False,
    )

    summary_metrics.to_csv(
        os.path.join(OUTDIR, "composition_prediction_summary_metrics_bayes_H0.csv"),
        index=False,
    )

    force_df.to_csv(
        os.path.join(OUTDIR, "composition_CIPHER_bayes_H0_force_genes.csv"),
        index=False,
    )

    temperature_df.to_csv(
        os.path.join(OUTDIR, "fold_temperature_values_bayes_H0.csv"),
        index=False,
    )

    posterior_fit_df.to_csv(
        os.path.join(OUTDIR, "posterior_fit_summary_by_fate_fold_bayes_H0.csv"),
        index=False,
    )

    if len(null_clone_errors) > 0:
        null_clone_errors.to_csv(
            os.path.join(OUTDIR, "null_clone_composition_errors_bayes_H0.csv"),
            index=False,
        )

    print("\nSaved main outputs:")
    print(os.path.join(OUTDIR, "clone_composition_predictions_bayes_H0.csv"))
    print(os.path.join(OUTDIR, "composition_prediction_summary_metrics_bayes_H0.csv"))
    print(os.path.join(OUTDIR, "composition_CIPHER_bayes_H0_force_genes.csv"))
    print(os.path.join(OUTDIR, "posterior_fit_summary_by_fate_fold_bayes_H0.csv"))


    # ============================================================
    # SUMMARY TABLES
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

    composition_summary.to_csv(
        os.path.join(OUTDIR, "composition_summary_by_model_bayes_H0.csv"),
        index=False,
    )

    per_fate_summary.to_csv(
        os.path.join(OUTDIR, "per_fate_fraction_summary_by_model_bayes_H0.csv"),
        index=False,
    )

    posterior_fit_summary = (
        posterior_fit_df
        .groupby("fate", as_index=False)
        .agg(
            mean_h0_scale=("h0_scale", "mean"),
            mean_n_eff_pos=("n_eff_pos", "mean"),
            mean_n_eff_neg=("n_eff_neg", "mean"),
            mean_delta_r2=("posterior_delta_r2", "mean"),
            mean_log_marginal=("log_marginal", "mean"),
            mean_penalty=("penalty", "mean"),
            mean_penalty_uncertainty=("penalty_uncertainty", "mean"),
        )
    )

    posterior_fit_summary.to_csv(
        os.path.join(OUTDIR, "posterior_fit_summary_aggregated_bayes_H0.csv"),
        index=False,
    )

    print("\nComposition summary:")
    print(composition_summary)

    print("\nPer-fate Bayesian CIPHER summary:")
    print(per_fate_summary[per_fate_summary["model"] == "cipher_bayes_H0"])

    print("\nPosterior fit summary:")
    print(posterior_fit_summary)


    # ============================================================
    # EMPIRICAL P-VALUES VS NULL
    # ============================================================

    p_rows = []

    null_models = [m for m in summary_metrics["model"].unique() if "null" in m]

    for null_model in null_models:
        for metric in ["mean_KL", "mean_JS", "mean_Brier", "mean_L1"]:
            real_vals = summary_metrics[
                (summary_metrics["model"] == "cipher_bayes_H0") &
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
                "direction": "lower_is_better",
                "cipher_mean": real_mean,
                "null_mean": np.mean(null_vals) if len(null_vals) else np.nan,
                "empirical_p": p_emp,
                "n_null": len(null_vals),
            })

        for metric in ["mean_cosine", "top1_accuracy", "entropy_pearson", "entropy_spearman"]:
            real_vals = summary_metrics[
                (summary_metrics["model"] == "cipher_bayes_H0") &
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
                "direction": "higher_is_better",
                "cipher_mean": real_mean,
                "null_mean": np.mean(null_vals) if len(null_vals) else np.nan,
                "empirical_p": p_emp,
                "n_null": len(null_vals),
            })

    pvals = pd.DataFrame(p_rows)

    pvals.to_csv(
        os.path.join(OUTDIR, "composition_empirical_pvalues_vs_null_bayes_H0.csv"),
        index=False,
    )

    print("\nEmpirical p-values vs null:")
    print(pvals)


    # ============================================================
    # PLOTS
    # ============================================================

    cipher_pred = predictions[predictions["model"] == "cipher_bayes_H0"].copy()

    model_label_map = {
        "cipher_bayes_H0": "Bayesian CIPHER, H0",
        "shuffled_null_bayes_H0": "shuffled null",
        "startpop_preserving_null_bayes_H0": "startpop-preserving null",
        "starting_population_only": "starting-pop only",
    }


    # ------------------------------------------------------------
    # Temperature distribution
    # ------------------------------------------------------------

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=temperature_df,
        x="model",
        y="temperature",
        showfliers=False,
    )

    sns.stripplot(
        data=temperature_df[temperature_df["model"] == "cipher_bayes_H0"],
        x="model",
        y="temperature",
        color="black",
        size=6,
    )

    plt.yscale("log")
    plt.title("Fitted softmax temperature")
    plt.xlabel("")
    plt.ylabel("temperature T")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, "fitted_temperature_distribution_bayes_H0.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "fitted_temperature_distribution_bayes_H0.svg"), bbox_inches="tight")
    plt.show()


    # ------------------------------------------------------------
    # Predicted vs observed fate fractions
    # ------------------------------------------------------------

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
        ax.set_ylabel("Bayesian CIPHER fraction")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)

    for k in range(n_fates, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, "predicted_vs_observed_fate_fractions_bayes_H0.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "predicted_vs_observed_fate_fractions_bayes_H0.svg"), bbox_inches="tight")
    plt.show()


    # ------------------------------------------------------------
    # Observed vs predicted composition heatmaps
    # ------------------------------------------------------------

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

    axes[1].set_title("Bayesian CIPHER composition")
    axes[1].set_xlabel("predicted terminal fate")
    axes[1].set_ylabel("dominant terminal fate")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_composition_heatmaps_bayes_H0.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_composition_heatmaps_bayes_H0.svg"), bbox_inches="tight")
    plt.show()


    # ------------------------------------------------------------
    # Composition errors vs null / baseline
    # ------------------------------------------------------------

    cipher_error_compact = cipher_pred[[
        "model", "fold", "temperature", "clone_id",
        "composition_KL_obs_pred", "composition_JS",
        "composition_Brier", "composition_L1",
        "composition_cosine", "dominant_fate_correct",
        "obs_entropy", "pred_entropy",
    ]].copy()

    error_plot_df = cipher_error_compact.copy()

    if len(null_clone_errors) > 0:
        error_plot_df = pd.concat([error_plot_df, null_clone_errors], ignore_index=True)

    if "starting_population_only" in predictions["model"].unique():
        sp_error = predictions[predictions["model"] == "starting_population_only"][[
            "model", "fold", "temperature", "clone_id",
            "composition_KL_obs_pred", "composition_JS",
            "composition_Brier", "composition_L1",
            "composition_cosine", "dominant_fate_correct",
            "obs_entropy", "pred_entropy",
        ]].copy()

        error_plot_df = pd.concat([error_plot_df, sp_error], ignore_index=True)

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
            data=error_plot_df[error_plot_df["model"] == "cipher_bayes_H0"],
            x="model_label",
            y=metric,
            color="black",
            alpha=0.35,
            size=3,
        )

        plt.title(f"Bayesian clone composition prediction: {metric}")
        plt.xlabel("")
        plt.ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        plt.savefig(os.path.join(OUTDIR, f"composition_error_{metric}_bayes_H0.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"composition_error_{metric}_bayes_H0.svg"), bbox_inches="tight")
        plt.show()


    # ------------------------------------------------------------
    # Per-fate correlation vs null / baseline
    # ------------------------------------------------------------

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

        point_df = perf[perf["model"].isin(["cipher_bayes_H0", "starting_population_only"])]

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

        plt.savefig(os.path.join(OUTDIR, f"per_fate_fraction_{metric}_bayes_H0.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"per_fate_fraction_{metric}_bayes_H0.svg"), bbox_inches="tight")
        plt.show()


    # ------------------------------------------------------------
    # Observed vs predicted entropy
    # ------------------------------------------------------------

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
    plt.ylabel("Bayesian CIPHER entropy")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_fate_entropy_bayes_H0.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "observed_vs_predicted_fate_entropy_bayes_H0.svg"), bbox_inches="tight")
    plt.show()


    # ------------------------------------------------------------
    # Dominant-fate confusion matrix
    # ------------------------------------------------------------

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

    plt.title("Dominant fate prediction from Bayesian composition")
    plt.xlabel("predicted dominant fate")
    plt.ylabel("observed dominant fate")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, "dominant_fate_confusion_bayes_H0.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "dominant_fate_confusion_bayes_H0.svg"), bbox_inches="tight")
    plt.show()


    # ------------------------------------------------------------
    # Posterior force gene heatmap
    # ------------------------------------------------------------

    cipher_force = force_df[
        (force_df["model"] == "cipher_bayes_H0") &
        (force_df["direction"] == "positive")
    ].copy()

    mean_force = (
        cipher_force
        .groupby(["fate", "gene"], as_index=False)
        .agg(
            mean_u=("posterior_mu_u", "mean"),
            mean_std=("posterior_std_u", "mean"),
            mean_z=("posterior_z_u", "mean"),
            mean_pip=("posterior_pip", "mean"),
            mean_delta=("delta_weighted_composition", "mean"),
            mean_yhat=("delta_posterior_yhat", "mean"),
            mean_rank=("rank", "mean"),
            mean_h0_scale=("h0_scale", "mean"),
            mean_delta_r2=("posterior_delta_r2", "mean"),
            mean_penalty=("penalty", "mean"),
            mean_temperature=("temperature", "mean"),
        )
    )

    top_genes = []

    for fate in selected_fates:
        sub = (
            mean_force[mean_force["fate"] == fate]
            .sort_values("mean_u", ascending=False)
            .head(TOP_GENES_PER_FATE_FOR_HEATMAP)
        )

        top_genes.extend(sub["gene"].tolist())

    top_genes = list(dict.fromkeys(top_genes))

    if len(top_genes) > 0:
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
            cbar_kws={"label": "mean posterior force E[u]"},
        )

        plt.title("Top positive Bayesian CIPHER posterior force genes")
        plt.xlabel("terminal fate")
        plt.ylabel("gene")
        plt.tight_layout()

        plt.savefig(os.path.join(OUTDIR, "composition_CIPHER_bayes_H0_force_gene_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "composition_CIPHER_bayes_H0_force_gene_heatmap.svg"), bbox_inches="tight")
        plt.show()

        pip_heat = (
            mean_force
            .pivot_table(index="gene", columns="fate", values="mean_pip", fill_value=0)
            .reindex(top_genes)
            .reindex(columns=selected_fates)
        )

        plt.figure(figsize=(1.4 * len(selected_fates) + 6, 0.28 * len(top_genes) + 4))

        sns.heatmap(
            pip_heat,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "mean posterior activity probability"},
        )

        plt.title("Posterior activity probability for top Bayesian CIPHER genes")
        plt.xlabel("terminal fate")
        plt.ylabel("gene")
        plt.tight_layout()

        plt.savefig(os.path.join(OUTDIR, "composition_CIPHER_bayes_H0_pip_gene_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "composition_CIPHER_bayes_H0_pip_gene_heatmap.svg"), bbox_inches="tight")
        plt.show()


    # ============================================================
    # FINAL PRINTS
    # ============================================================

    print("\n============================================================")
    print("FINAL BAYESIAN H0 COMPOSITION SUMMARY")
    print("============================================================")
    print(composition_summary)

    print("\n============================================================")
    print("PER-FATE BAYESIAN CIPHER FRACTION PREDICTION")
    print("============================================================")
    print(per_fate_summary[per_fate_summary["model"] == "cipher_bayes_H0"])

    print("\n============================================================")
    print("EMPIRICAL NULL P-VALUES")
    print("============================================================")
    print(pvals)

    print("\n============================================================")
    print("POSTERIOR FIT SUMMARY")
    print("============================================================")
    print(posterior_fit_summary)

    print("\n============================================================")
    print("FITTED TEMPERATURES")
    print("============================================================")
    print(temperature_df[temperature_df["model"] == "cipher_bayes_H0"])

    print("\n============================================================")
    print("TOP POSITIVE BAYESIAN CIPHER POSTERIOR FORCE GENES")
    print("============================================================")

    if len(force_df) > 0:
        for fate in selected_fates:
            sub = (
                mean_force[mean_force["fate"] == fate]
                .sort_values("mean_u", ascending=False)
                .head(20)
            )

            print(f"\n{fate}")
            print(", ".join(sub["gene"].astype(str).tolist()))

    print("\nDone. Outputs in:", OUTDIR)



def composition_pearson_with_lfc():
    global os, gzip, warnings, np, pd, plt, sns, Counter
    global mmread, issparse, pearsonr, norm, minimize_scalar, StratifiedKFold, OUTDIR, COUNTS_PATH
    global GENES_PATH, CLONE_PATH, META_PATH, TIME_COL, CELLTYPE_COL, START_COL, WELL_COL, EARLY_TIME
    global EARLY_CELLTYPE, EARLY_WELL, TERMINAL_TIME, TERMINAL_WELL, RESTRICT_STARTING_POPULATION, EXCLUDE_FATES, MANUAL_SELECTED_FATES, PREFERRED_FATE_ORDER
    global MAX_FATES, MIN_CLONES_WITH_FATE, MIN_TERMINAL_CELLS_PER_CLONE, MIN_EARLY_CELLS_PER_CLONE, MIN_TOTAL_CELLS_PER_CLONE, MIN_SELECTED_FATE_COVERAGE, MIN_SELECTED_TERMINAL_CELLS, N_VAR_GENES
    global MAX_COV_CELLS, RIDGE, COV_SHRINK_TO_DIAG, USE_FATE_PRIOR, POSTERIOR_TAU2, H0_MIN_SCALE, USE_POSTERIOR_UNCERTAINTY_PENALTY, POSTERIOR_ACTIVITY_THRESHOLD
    global CALIBRATE_TEMPERATURE, TEMP_MIN, TEMP_MAX, LFC_PSEUDOCOUNT, LFC_CLIP, N_NULLS, USE_STARTPOP_PRESERVING_NULL, N_SPLITS
    global SEED, TOP_FORCE_GENES_PER_DIRECTION, rng, check_file, safe_name, softmax_logits, r2_from_pred
    global get_cell_to_clone, get_cells_x_genes, zscore_train, select_hvgs_sparse, make_covariance, clone_mean_raw_and_z, shuffle_YC_within_groups, fit_temperature_from_counts
    global gaussian_activity_probability, weighted_delta_and_H0_scale, analytic_bayesian_cipher_posterior_H0_equals_scaleSigma_v2, make_composition_cipher_bayes_model_v2, get_logits, score_composition_cipher, fit_terminal_vs_undiff_lfc_composition_model, score_terminal_vs_undiff_lfc_model
    global fit_startpop_composition_baseline, score_startpop_composition_baseline, add_prediction_columns, summarize_pearson_by_cell_type, p, counts, f, gene_names
    global clone_mat, meta, cell_to_clone, has_clone, cell_fates, early_mask, terminal_mask, early_all_idx
    global early_cloned_mask, terminal_cloned_mask, early_cloned_idx, terminal_cloned_idx, candidate_records, global_fate_counts, global_fate_clone_counts, clone_id
    global cells, early_cells, terminal_cells, fates, vc, terminal_counts_dict, c, starts
    global dominant_start, dominant_start_frac, candidate_table, fate_summary, selected_fates, ordered, clone_table, fate
    global s, selected_count_cols, obs_frac_cols, Y_all, dominant_idx, clone_table_save, hvg_idx, gene_vars
    global hvg_genes, cov_idx, Xcov_raw, mu_ref, sd_ref, Xcov_z, Sigma, evals
    global evecs, X_clones_all, strat_y, min_class_n, n_splits, splitter, clone_to_obs, clone_to_counts
    global clone_to_start, all_pred_rows, pearson_rows, temperature_rows, posterior_fit_rows, force_rows, lfc_fit_rows, fold
    global train_pos, test_pos, train_clones, test_clones, Xraw_train, Xtrain, train_clone_ids_used, n_train_early
    global Xraw_test, Xtest, test_clone_ids_used, n_test_early, Ytrain, Ctrain, Ytest, Ctest
    global start_train, start_test, true_dom_test, base, j, cipher_model, _, train_logits
    global T_cipher, raw_scores, logits, Ptest, pred_cipher, order, rank, gi
    global lfc_model, raw_lfc_train, logits_lfc_train, T_lfc, raw_lfc, logits_lfc, P_lfc, pred_lfc
    global sp_model, raw_sp, logits_sp, P_sp, pred_sp, null_id, Ytrain_null, Ctrain_null
    global null_name, perm, null_model, null_train_logits, T_null, raw_null, logits_null, P_null
    global null_df, predictions, pearson_metrics, pearson_by_cell_type, pearson_model_mean, force_df, temperature_df, posterior_fit_df
    global lfc_fit_df, plot_df, model_label_map, model_order_raw, model_order, fate_order, palette, fig
    global ax, handles, labels, n, tick, spine, mean_plot, x
    # ============================================================
    # CIPHER-LARRY CLONE FATE COMPOSITION
    # FULL STANDALONE SCRIPT
    #
    # Pearson-only evaluation:
    #   - Pearson correlation per terminal cell type / fate
    #   - Boxplot style like your reference figure
    #
    # Models:
    #   1. Bayesian CIPHER, H0
    #   2. terminal-vs-undiff LFC
    #   3. starting-pop only
    #   4. startpop-preserving null
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
    from scipy.stats import pearsonr, norm
    from scipy.optimize import minimize_scalar
    from sklearn.model_selection import StratifiedKFold

    warnings.filterwarnings("ignore")


    # ============================================================
    # CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_composition_pearson_with_LFC_boxplot")
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

    TERMINAL_TIME = 6.0
    TERMINAL_WELL = None

    RESTRICT_STARTING_POPULATION = None

    EXCLUDE_FATES = {
        "Undifferentiated", "Unknown", "unknown", "nan", "NaN",
        "Ambiguous", "ambiguous", "None", ""
    }

    MANUAL_SELECTED_FATES = None

    PREFERRED_FATE_ORDER = [
        "Monocyte",
        "Neutrophil",
        "Baso",
        "Mast",
        "Meg",
        "Erythroid",
        "Eos",
    ]

    MAX_FATES = 7

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

    POSTERIOR_TAU2 = 1.0
    H0_MIN_SCALE = 1e-8
    USE_POSTERIOR_UNCERTAINTY_PENALTY = True
    POSTERIOR_ACTIVITY_THRESHOLD = None

    CALIBRATE_TEMPERATURE = True
    TEMP_MIN = 0.1
    TEMP_MAX = 100.0

    LFC_PSEUDOCOUNT = 1e-3
    LFC_CLIP = 8.0

    N_NULLS = 100
    USE_STARTPOP_PRESERVING_NULL = True

    N_SPLITS = 5
    SEED = 0

    TOP_FORCE_GENES_PER_DIRECTION = 50

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)


    # ============================================================
    # HELPERS
    # ============================================================

    def check_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file: {path}\nCurrent working directory: {os.getcwd()}"
            )
        print(f"[OK] Found {path}")


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


    def safe_corr(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]

        if len(x) < 3:
            return np.nan

        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan

        try:
            return pearsonr(x, y)[0]
        except Exception:
            return np.nan


    def r2_from_pred(y, yhat, eps=1e-12):
        y = np.asarray(y, float).reshape(-1)
        yhat = np.asarray(yhat, float).reshape(-1)
        return 1.0 - np.sum((y - yhat) ** 2) / (np.sum(y ** 2) + eps)


    def get_cell_to_clone(clone_mat):
        coo = clone_mat.tocoo()
        cell_to_clone = -np.ones(clone_mat.shape[1], dtype=int)
        cell_to_clone[coo.col] = coo.row
        return cell_to_clone


    def get_cells_x_genes(counts, cell_idx, gene_idx):
        return safe_toarray(counts[gene_idx][:, cell_idx]).T.astype(np.float32)


    def zscore_train(X):
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return mu, sd


    def select_hvgs_sparse(counts, cell_idx, n_var_genes):
        X = counts[:, cell_idx]

        means = np.asarray(X.mean(axis=1)).ravel()
        seconds = np.asarray(X.multiply(X).mean(axis=1)).ravel()
        vars_ = seconds - means ** 2

        valid = np.isfinite(vars_) & (vars_ > 0)
        valid_idx = np.where(valid)[0]

        if len(valid_idx) == 0:
            raise RuntimeError("No valid variable genes found.")

        n_keep = int(min(n_var_genes, len(valid_idx)))

        hvg_idx = valid_idx[np.argsort(vars_[valid_idx])[-n_keep:]]
        hvg_idx = np.sort(hvg_idx)

        return hvg_idx, vars_


    def make_covariance(X):
        X = np.asarray(X, dtype=np.float64)
        Xc = X - X.mean(axis=0, keepdims=True)

        Sigma = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)

        D = np.diag(np.diag(Sigma))
        Sigma = (1.0 - COV_SHRINK_TO_DIAG) * Sigma + COV_SHRINK_TO_DIAG * D

        scale = np.mean(np.diag(Sigma)) + 1e-12
        Sigma = Sigma + RIDGE * scale * np.eye(Sigma.shape[0])

        return Sigma.astype(np.float64)


    def clone_mean_raw_and_z(
        clone_ids,
        early_mask,
        cell_to_clone,
        counts,
        hvg_idx,
        mu_ref,
        sd_ref,
    ):
        raw_rows = []
        z_rows = []
        out_ids = []
        out_n = []

        for cid in clone_ids:
            idx = np.where(early_mask & (cell_to_clone == cid))[0]

            if len(idx) == 0:
                continue

            Xraw = get_cells_x_genes(counts, idx, hvg_idx).astype(np.float64)
            Xz = apply_zscore(Xraw, mu_ref, sd_ref)

            raw_rows.append(Xraw.mean(axis=0))
            z_rows.append(Xz.mean(axis=0))
            out_ids.append(int(cid))
            out_n.append(int(len(idx)))

        if len(raw_rows) == 0:
            n_genes = len(hvg_idx)

            return (
                np.empty((0, n_genes), dtype=np.float64),
                np.empty((0, n_genes), dtype=np.float64),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        return (
            np.vstack(raw_rows).astype(np.float64),
            np.vstack(z_rows).astype(np.float64),
            np.asarray(out_ids, dtype=int),
            np.asarray(out_n, dtype=int),
        )


    def shuffle_YC_within_groups(Y, C, groups):
        Y = np.asarray(Y).copy()
        C = np.asarray(C).copy()
        groups = np.asarray(groups).astype(str)

        Yout = Y.copy()
        Cout = C.copy()

        for g in np.unique(groups):
            idx = np.where(groups == g)[0]

            if len(idx) > 1:
                perm_idx = idx[rng.permutation(len(idx))]
                Yout[idx] = Y[perm_idx]
                Cout[idx] = C[perm_idx]

        return Yout, Cout


    def fit_temperature_from_counts(logits, counts, temp_min=0.1, temp_max=100.0):
        logits = np.asarray(logits, dtype=float)
        counts = np.asarray(counts, dtype=float)

        if logits.shape != counts.shape:
            raise ValueError("logits and counts must have same shape.")

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
    # BAYESIAN CIPHER H0
    # ============================================================

    def gaussian_activity_probability(mu, std, effect_threshold=None):
        mu = np.asarray(mu, dtype=float)
        std = np.asarray(std, dtype=float)

        if effect_threshold is None:
            effect_threshold = float(np.median(std))

        z_upper = (effect_threshold - mu) / (std + 1e-12)
        z_lower = (-effect_threshold - mu) / (std + 1e-12)

        pip = 1.0 - (norm.cdf(z_upper) - norm.cdf(z_lower))
        return pip, float(effect_threshold)


    def weighted_delta_and_H0_scale(X, w_pos, w_neg, eps=1e-12):
        X = np.asarray(X, dtype=np.float64)
        w_pos = np.asarray(w_pos, dtype=np.float64).reshape(-1)
        w_neg = np.asarray(w_neg, dtype=np.float64).reshape(-1)

        sp = float(w_pos.sum())
        sn = float(w_neg.sum())

        if sp <= eps or sn <= eps:
            delta = np.zeros(X.shape[1], dtype=np.float64)
            return delta, 1.0, 0.0, 0.0

        wp = w_pos / sp
        wn = w_neg / sn

        delta = (wp @ X) - (wn @ X)

        sum_wp2 = float(np.sum(wp ** 2))
        sum_wn2 = float(np.sum(wn ** 2))

        h0_scale = max(sum_wp2 + sum_wn2, H0_MIN_SCALE)

        n_eff_pos = 1.0 / max(sum_wp2, eps)
        n_eff_neg = 1.0 / max(sum_wn2, eps)

        return delta, h0_scale, n_eff_pos, n_eff_neg


    def analytic_bayesian_cipher_posterior_H0_equals_scaleSigma_v2(
        delta,
        evals,
        evecs,
        h0_scale,
        tau2=1.0,
        use_uncertainty_penalty=True,
        activity_threshold=None,
        eps=1e-12,
    ):
        delta = np.asarray(delta, dtype=np.float64).reshape(-1)
        evals = np.asarray(evals, dtype=np.float64).reshape(-1)
        evecs = np.asarray(evecs, dtype=np.float64)

        tau2 = float(tau2)
        h0_scale = max(float(h0_scale), H0_MIN_SCALE)

        if tau2 <= 0:
            raise ValueError("tau2 must be positive.")

        lam = np.maximum(evals, eps)
        delta_eig = evecs.T @ delta

        denom_mu = lam + h0_scale / tau2
        u_eig = delta_eig / np.maximum(denom_mu, eps)
        mu_u = evecs @ u_eig

        cov_eig = 1.0 / np.maximum(lam / h0_scale + 1.0 / tau2, eps)
        std_u = np.sqrt(np.maximum((evecs ** 2) @ cov_eig, 0.0))

        yhat_eig = lam * u_eig
        yhat = evecs @ yhat_eig

        r2 = r2_from_pred(delta, yhat)

        penalty_mean = 0.5 * float(np.sum(lam * (u_eig ** 2)))
        penalty_uncertainty = 0.5 * float(np.sum(lam * cov_eig))

        if use_uncertainty_penalty:
            penalty = penalty_mean + penalty_uncertainty
        else:
            penalty = penalty_mean

        C_eig = h0_scale * lam + tau2 * (lam ** 2)
        C_eig = np.maximum(C_eig, eps)

        logdetC = float(np.sum(np.log(C_eig)))
        quad = float(np.sum((delta_eig ** 2) / C_eig))
        d = len(delta)

        log_marginal = -0.5 * (d * np.log(2.0 * np.pi) + logdetC + quad)

        pip, used_activity_threshold = gaussian_activity_probability(
            mu_u,
            std_u,
            effect_threshold=activity_threshold,
        )

        p_pos, p_neg = gaussian_sign_probability(mu_u, std_u)
        sign_conf = np.maximum(p_pos, p_neg)

        zero_exc, ci95_lo, ci95_hi = credible_zero_excluded(mu_u, std_u, level=0.95)

        z = mu_u / (std_u + eps)

        return {
            "mu": mu_u,
            "std": std_u,
            "z": z,
            "pip": pip,
            "p_pos": p_pos,
            "p_neg": p_neg,
            "sign_conf": sign_conf,
            "ci95_lo": ci95_lo,
            "ci95_hi": ci95_hi,
            "zero_excluded": zero_exc.astype(bool),
            "u_eig": u_eig,
            "cov_eig": cov_eig,
            "yhat": yhat,
            "r2": r2,
            "penalty": penalty,
            "penalty_mean": penalty_mean,
            "penalty_uncertainty": penalty_uncertainty,
            "h0_scale": h0_scale,
            "tau2": tau2,
            "activity_threshold": used_activity_threshold,
            "log_marginal": log_marginal,
        }


    def make_composition_cipher_bayes_model_v2(
        Xtrain_clone,
        Ytrain,
        selected_fates,
        evals,
        evecs,
        tau2=1.0,
        use_fate_prior=False,
        use_uncertainty_penalty=True,
        activity_threshold=None,
    ):
        Xtrain_clone = np.asarray(Xtrain_clone, dtype=np.float64)
        Ytrain = np.asarray(Ytrain, dtype=np.float64)

        U = []
        DELTAS = []
        YHATS = []

        posterior_std = []
        posterior_z = []
        posterior_pip = []
        posterior_sign_conf = []

        penalties = []
        penalty_mean = []
        penalty_uncertainty = []

        h0_scales = []
        n_eff_pos_list = []
        n_eff_neg_list = []
        r2_list = []
        log_marginals = []
        activity_thresholds = []
        log_priors = []

        eps = 1e-12

        for j, fate in enumerate(selected_fates):
            w_pos = Ytrain[:, j].copy()
            w_neg = 1.0 - w_pos

            delta, h0_scale, n_eff_pos, n_eff_neg = weighted_delta_and_H0_scale(
                X=Xtrain_clone,
                w_pos=w_pos,
                w_neg=w_neg,
            )

            post = analytic_bayesian_cipher_posterior_H0_equals_scaleSigma_v2(
                delta=delta,
                evals=evals,
                evecs=evecs,
                h0_scale=h0_scale,
                tau2=tau2,
                use_uncertainty_penalty=use_uncertainty_penalty,
                activity_threshold=activity_threshold,
            )

            if use_fate_prior:
                prior = max(float(Ytrain[:, j].mean()), eps)
                log_prior = np.log(prior)
            else:
                log_prior = 0.0

            U.append(post["mu"])
            DELTAS.append(delta)
            YHATS.append(post["yhat"])

            posterior_std.append(post["std"])
            posterior_z.append(post["z"])
            posterior_pip.append(post["pip"])
            posterior_sign_conf.append(post["sign_conf"])

            penalties.append(post["penalty"])
            penalty_mean.append(post["penalty_mean"])
            penalty_uncertainty.append(post["penalty_uncertainty"])

            h0_scales.append(post["h0_scale"])
            n_eff_pos_list.append(n_eff_pos)
            n_eff_neg_list.append(n_eff_neg)
            r2_list.append(post["r2"])
            log_marginals.append(post["log_marginal"])
            activity_thresholds.append(post["activity_threshold"])
            log_priors.append(log_prior)

        return {
            "model_type": "cipher_bayes_H0",
            "U": np.asarray(U),
            "DELTAS": np.asarray(DELTAS),
            "YHAT_DELTAS": np.asarray(YHATS),

            "posterior_std": np.asarray(posterior_std),
            "posterior_z": np.asarray(posterior_z),
            "posterior_pip": np.asarray(posterior_pip),
            "posterior_sign_conf": np.asarray(posterior_sign_conf),

            "penalty": np.asarray(penalties),
            "penalty_mean": np.asarray(penalty_mean),
            "penalty_uncertainty": np.asarray(penalty_uncertainty),

            "h0_scale": np.asarray(h0_scales),
            "n_eff_pos": np.asarray(n_eff_pos_list),
            "n_eff_neg": np.asarray(n_eff_neg_list),
            "posterior_delta_r2": np.asarray(r2_list),
            "log_marginal": np.asarray(log_marginals),
            "activity_threshold": np.asarray(activity_thresholds),

            "log_prior": np.asarray(log_priors),
            "tau2": float(tau2),
            "temperature": 1.0,
        }


    def get_logits(X, model):
        X = np.asarray(X, dtype=np.float64)
        raw_scores = X @ model["U"].T
        logits = raw_scores - model["penalty"][None, :] + model["log_prior"][None, :]
        return raw_scores, logits


    def score_composition_cipher(X, model, temperature=None):
        raw_scores, logits = get_logits(X, model)

        if temperature is None:
            temperature = model.get("temperature", 1.0)

        P = softmax_logits(logits, temperature=temperature)
        return raw_scores, logits, P


    # ============================================================
    # TERMINAL-vs-UNDIF LFC BASELINE
    # ============================================================

    def fit_terminal_vs_undiff_lfc_composition_model(
        counts,
        hvg_idx,
        cell_to_clone,
        train_clone_ids,
        selected_fates,
        early_undiff_mask,
        terminal_mask,
        cell_fates,
        Ytrain=None,
        pseudocount=1e-3,
        clip=8.0,
        use_prior=False,
    ):
        train_clone_ids = np.asarray(train_clone_ids, dtype=int)
        cell_fates_str = np.asarray(cell_fates).astype(str)

        control_cells = cells_for_clone_set(
            cell_to_clone=cell_to_clone,
            clone_ids=train_clone_ids,
            mask=early_undiff_mask,
        )

        if len(control_cells) == 0:
            raise RuntimeError(
                "No training-clone day-4 undifferentiated cells for LFC denominator."
            )

        X0 = get_cells_x_genes(counts, control_cells, hvg_idx).astype(np.float64)
        mu0 = X0.mean(axis=0)

        W_lfc = []
        log_priors = []
        mu_f_all = []
        n_terminal_cells_by_fate = {}

        eps = 1e-12

        for j, fate in enumerate(selected_fates):
            fate_terminal_mask = terminal_mask & (cell_fates_str == str(fate))

            fate_terminal_cells = cells_for_clone_set(
                cell_to_clone=cell_to_clone,
                clone_ids=train_clone_ids,
                mask=fate_terminal_mask,
            )

            if len(fate_terminal_cells) == 0:
                raise RuntimeError(
                    f"No training terminal cells annotated as fate/cell type {fate!r}."
                )

            Xf = get_cells_x_genes(counts, fate_terminal_cells, hvg_idx).astype(np.float64)
            muf = Xf.mean(axis=0)

            lfc = np.log((muf + pseudocount) / (mu0 + pseudocount))

            if clip is not None:
                lfc = np.clip(lfc, -float(clip), float(clip))

            if use_prior and Ytrain is not None:
                prior = max(float(np.mean(Ytrain[:, j])), eps)
                log_prior = np.log(prior)
            else:
                log_prior = 0.0

            W_lfc.append(lfc)
            log_priors.append(log_prior)
            mu_f_all.append(muf)
            n_terminal_cells_by_fate[str(fate)] = int(len(fate_terminal_cells))

        return {
            "model_type": "terminal_vs_undiff_LFC",
            "W_lfc": np.asarray(W_lfc, dtype=np.float64),
            "log_prior": np.asarray(log_priors, dtype=np.float64),
            "mu0_undiff_raw": mu0,
            "muf_terminal_raw": np.asarray(mu_f_all, dtype=np.float64),
            "n_control_cells": int(len(control_cells)),
            "n_terminal_cells_by_fate": n_terminal_cells_by_fate,
            "pseudocount": float(pseudocount),
            "clip": clip,
        }


    def score_terminal_vs_undiff_lfc_model(Xraw_day4_undiff_clone_means, model, temperature=1.0):
        raw = np.asarray(Xraw_day4_undiff_clone_means, dtype=np.float64) @ model["W_lfc"].T
        logits = raw + model["log_prior"][None, :]
        probs = softmax_logits(logits, temperature=temperature)
        return raw, logits, probs


    # ============================================================
    # STARTPOP BASELINE
    # ============================================================

    def fit_startpop_composition_baseline(Ytrain, start_train, alpha=2.0):
        Ytrain = np.asarray(Ytrain, dtype=float)
        start_train = np.asarray(start_train).astype(str)

        global_p = Ytrain.mean(axis=0)
        global_p = global_p / np.maximum(global_p.sum(), 1e-12)

        table = {}

        for s in np.unique(start_train):
            idx = np.where(start_train == s)[0]

            if len(idx) == 0:
                continue

            p = (Ytrain[idx].sum(axis=0) + alpha * global_p) / (len(idx) + alpha)
            p = p / np.maximum(p.sum(), 1e-12)

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


    # ============================================================
    # PREDICTION + PEARSON HELPERS
    # ============================================================

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


    def summarize_pearson_by_cell_type(df, selected_fates, model_name, fold, null_id=None):
        rows = []

        for fate in selected_fates:
            s = safe_name(fate)

            obs_col = f"obs_frac__{s}"
            pred_col = f"pred_frac__{s}"

            x = df[obs_col].values.astype(float)
            y = df[pred_col].values.astype(float)

            rows.append({
                "model": model_name,
                "fold": fold,
                "null_id": null_id,
                "cell_type": fate,
                "pearson": safe_corr(x, y),
                "n_clones": int(np.sum(np.isfinite(x) & np.isfinite(y))),
                "mean_observed_fraction": float(np.nanmean(x)),
                "mean_predicted_fraction": float(np.nanmean(y)),
            })

        return pd.DataFrame(rows)


    # ============================================================
    # LOAD DATA
    # ============================================================

    for p in [COUNTS_PATH, GENES_PATH, CLONE_PATH, META_PATH]:
        check_file(p)

    counts = mmread(COUNTS_PATH).T.tocsr()
    print(f"Counts: {counts.shape[0]} genes x {counts.shape[1]} cells | nnz={counts.nnz:,}")

    with gzip.open(GENES_PATH, "rt") as f:
        gene_names = np.array([line.strip() for line in f])

    print(f"Genes loaded: {len(gene_names)}")

    clone_mat = mmread(CLONE_PATH).T.tocsr()
    print(f"Clone matrix: {clone_mat.shape[0]} clones x {clone_mat.shape[1]} cells")

    meta = pd.read_csv(META_PATH, sep="\t")
    meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")

    print(f"Meta: {meta.shape[0]} rows x {meta.shape[1]} cols")
    print("Meta columns:", list(meta.columns))

    assert counts.shape[1] == meta.shape[0] == clone_mat.shape[1], "cells mismatch"
    assert counts.shape[0] == len(gene_names), "genes mismatch"

    cell_to_clone = get_cell_to_clone(clone_mat)
    has_clone = cell_to_clone >= 0
    cell_fates = meta[CELLTYPE_COL].astype(str).values

    print("\nTimepoints:")
    print(np.sort(meta[TIME_COL].dropna().unique()))

    print("\nCell annotations:")
    print(meta[CELLTYPE_COL].value_counts())


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
            dominant_start = str(starts.index[0])
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

    fate_summary.to_csv(
        os.path.join(OUTDIR, "terminal_fate_summary_before_selection.csv"),
        index=False,
    )

    if MANUAL_SELECTED_FATES is None:
        selected_fates = (
            fate_summary[fate_summary["clone_count_with_fate"] >= MIN_CLONES_WITH_FATE]
            .head(MAX_FATES)["fate"]
            .tolist()
        )
    else:
        selected_fates = list(MANUAL_SELECTED_FATES)

    ordered = [f for f in PREFERRED_FATE_ORDER if f in selected_fates]
    ordered += [f for f in selected_fates if f not in ordered]
    selected_fates = ordered

    if len(selected_fates) < 2:
        raise RuntimeError("Fewer than two selected fates.")

    print("\nSelected terminal cell types / fates:")
    print(selected_fates)

    clone_table = candidate_table.copy()

    for fate in selected_fates:
        s = safe_name(fate)
        clone_table[f"terminal_count__{s}"] = clone_table["terminal_counts_dict"].apply(
            lambda d: int(d.get(fate, 0))
        )

    selected_count_cols = [f"terminal_count__{safe_name(f)}" for f in selected_fates]

    clone_table["n_terminal_selected"] = clone_table[selected_count_cols].sum(axis=1)
    clone_table["selected_fate_coverage"] = (
        clone_table["n_terminal_selected"] / clone_table["n_terminal"]
    )

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
    print("\nMean selected fate coverage:", clone_table["selected_fate_coverage"].mean())

    clone_table_save = clone_table.drop(columns=["terminal_counts_dict"])
    clone_table_save.to_csv(
        os.path.join(OUTDIR, "clone_terminal_composition_table.csv"),
        index=False,
    )


    # ============================================================
    # HVGs + COVARIANCE
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
    print(f"Using {len(hvg_idx):,} HVGs.")

    Xcov_raw = get_cells_x_genes(counts, cov_idx, hvg_idx).astype(np.float64)
    mu_ref, sd_ref = zscore_train(Xcov_raw)
    Xcov_z = apply_zscore(Xcov_raw, mu_ref, sd_ref)

    print("Computing covariance...")
    Sigma = make_covariance(Xcov_z)

    print("Computing eigendecomposition of Sigma...")
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-10)

    pd.DataFrame({
        "rank": np.arange(1, len(evals) + 1),
        "eigenvalue": evals[::-1],
    }).to_csv(os.path.join(OUTDIR, "early_covariance_eigenvalues.csv"), index=False)


    # ============================================================
    # CROSS-VALIDATED PREDICTION
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

    all_pred_rows = []
    pearson_rows = []
    temperature_rows = []
    posterior_fit_rows = []
    force_rows = []
    lfc_fit_rows = []

    for fold, (train_pos, test_pos) in enumerate(splitter.split(X_clones_all, strat_y)):
        train_clones = X_clones_all[train_pos]
        test_clones = X_clones_all[test_pos]

        print(f"\nFold {fold + 1}/{n_splits}: train={len(train_clones)}, test={len(test_clones)}")

        Xraw_train, Xtrain, train_clone_ids_used, n_train_early = clone_mean_raw_and_z(
            clone_ids=train_clones,
            early_mask=early_cloned_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu_ref=mu_ref,
            sd_ref=sd_ref,
        )

        Xraw_test, Xtest, test_clone_ids_used, n_test_early = clone_mean_raw_and_z(
            clone_ids=test_clones,
            early_mask=early_cloned_mask,
            cell_to_clone=cell_to_clone,
            counts=counts,
            hvg_idx=hvg_idx,
            mu_ref=mu_ref,
            sd_ref=sd_ref,
        )

        if Xtrain.shape[0] == 0 or Xtest.shape[0] == 0:
            print(f"[skip] Fold {fold} has no usable train/test clone means.")
            continue

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
            "n_terminal_selected": Ctest.sum(axis=1),
            "obs_entropy": entropy(Ytest),
        })

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            base[f"obs_frac__{s}"] = Ytest[:, j]
            base[f"terminal_count__{s}"] = Ctest[:, j]

        # --------------------------------------------------------
        # 1. Bayesian CIPHER H0
        # --------------------------------------------------------
        cipher_model = make_composition_cipher_bayes_model_v2(
            Xtrain_clone=Xtrain,
            Ytrain=Ytrain,
            selected_fates=selected_fates,
            evals=evals,
            evecs=evecs,
            tau2=POSTERIOR_TAU2,
            use_fate_prior=USE_FATE_PRIOR,
            use_uncertainty_penalty=USE_POSTERIOR_UNCERTAINTY_PENALTY,
            activity_threshold=POSTERIOR_ACTIVITY_THRESHOLD,
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

        raw_scores, logits, Ptest = score_composition_cipher(
            Xtest,
            cipher_model,
            temperature=T_cipher,
        )

        pred_cipher = add_prediction_columns(
            base,
            raw_scores,
            logits,
            Ptest,
            selected_fates,
            "cipher_bayes_H0",
            temperature=T_cipher,
        )

        pred_cipher["null_id"] = np.nan
        all_pred_rows.append(pred_cipher)

        pearson_rows.append(
            summarize_pearson_by_cell_type(
                pred_cipher,
                selected_fates,
                model_name="cipher_bayes_H0",
                fold=fold,
                null_id=np.nan,
            )
        )

        temperature_rows.append({
            "fold": fold,
            "model": "cipher_bayes_H0",
            "null_id": np.nan,
            "temperature": T_cipher,
            "tau2": POSTERIOR_TAU2,
        })

        for j, fate in enumerate(selected_fates):
            posterior_fit_rows.append({
                "fold": fold,
                "model": "cipher_bayes_H0",
                "fate": fate,
                "h0_scale": float(cipher_model["h0_scale"][j]),
                "n_eff_pos": float(cipher_model["n_eff_pos"][j]),
                "n_eff_neg": float(cipher_model["n_eff_neg"][j]),
                "posterior_delta_r2": float(cipher_model["posterior_delta_r2"][j]),
                "penalty": float(cipher_model["penalty"][j]),
                "penalty_mean": float(cipher_model["penalty_mean"][j]),
                "penalty_uncertainty": float(cipher_model["penalty_uncertainty"][j]),
                "log_marginal": float(cipher_model["log_marginal"][j]),
                "temperature": float(T_cipher),
                "tau2": float(POSTERIOR_TAU2),
            })

            order = np.argsort(np.abs(cipher_model["U"][j]))[::-1]

            for rank, gi in enumerate(order[:TOP_FORCE_GENES_PER_DIRECTION], start=1):
                force_rows.append({
                    "fold": fold,
                    "model": "cipher_bayes_H0",
                    "fate": fate,
                    "rank_abs": rank,
                    "gene": hvg_genes[gi],
                    "gene_index_original": int(hvg_idx[gi]),
                    "u_posterior_mean": float(cipher_model["U"][j, gi]),
                    "u_posterior_std": float(cipher_model["posterior_std"][j, gi]),
                    "u_posterior_z": float(cipher_model["posterior_z"][j, gi]),
                    "u_pip": float(cipher_model["posterior_pip"][j, gi]),
                    "u_sign_conf": float(cipher_model["posterior_sign_conf"][j, gi]),
                })

        # --------------------------------------------------------
        # 2. Terminal-vs-undiff LFC
        # --------------------------------------------------------
        lfc_model = fit_terminal_vs_undiff_lfc_composition_model(
            counts=counts,
            hvg_idx=hvg_idx,
            cell_to_clone=cell_to_clone,
            train_clone_ids=train_clone_ids_used,
            selected_fates=selected_fates,
            early_undiff_mask=early_cloned_mask,
            terminal_mask=terminal_cloned_mask,
            cell_fates=cell_fates,
            Ytrain=Ytrain,
            pseudocount=LFC_PSEUDOCOUNT,
            clip=LFC_CLIP,
            use_prior=USE_FATE_PRIOR,
        )

        raw_lfc_train, logits_lfc_train, _ = score_terminal_vs_undiff_lfc_model(
            Xraw_train,
            lfc_model,
            temperature=1.0,
        )

        if CALIBRATE_TEMPERATURE:
            T_lfc = fit_temperature_from_counts(
                logits=logits_lfc_train,
                counts=Ctrain,
                temp_min=TEMP_MIN,
                temp_max=TEMP_MAX,
            )
        else:
            T_lfc = 1.0

        raw_lfc, logits_lfc, P_lfc = score_terminal_vs_undiff_lfc_model(
            Xraw_test,
            lfc_model,
            temperature=T_lfc,
        )

        pred_lfc = add_prediction_columns(
            base,
            raw_lfc,
            logits_lfc,
            P_lfc,
            selected_fates,
            "terminal_vs_undiff_LFC",
            temperature=T_lfc,
        )

        pred_lfc["null_id"] = np.nan
        all_pred_rows.append(pred_lfc)

        pearson_rows.append(
            summarize_pearson_by_cell_type(
                pred_lfc,
                selected_fates,
                model_name="terminal_vs_undiff_LFC",
                fold=fold,
                null_id=np.nan,
            )
        )

        temperature_rows.append({
            "fold": fold,
            "model": "terminal_vs_undiff_LFC",
            "null_id": np.nan,
            "temperature": T_lfc,
            "tau2": np.nan,
        })

        lfc_fit_rows.append({
            "fold": fold,
            "model": "terminal_vs_undiff_LFC",
            "n_control_cells": lfc_model["n_control_cells"],
            "n_terminal_cells_by_fate": str(lfc_model["n_terminal_cells_by_fate"]),
            "pseudocount": lfc_model["pseudocount"],
            "clip": lfc_model["clip"],
            "temperature": T_lfc,
        })

        print("  LFC denominator n day4-undiff control cells:", lfc_model["n_control_cells"])
        print("  LFC numerator n terminal cells:", lfc_model["n_terminal_cells_by_fate"])

        # --------------------------------------------------------
        # 3. Starting-pop only
        # --------------------------------------------------------
        if START_COL in meta.columns:
            sp_model = fit_startpop_composition_baseline(Ytrain, start_train, alpha=2.0)
            raw_sp, logits_sp, P_sp = score_startpop_composition_baseline(start_test, sp_model)

            pred_sp = add_prediction_columns(
                base,
                raw_sp,
                logits_sp,
                P_sp,
                selected_fates,
                "starting_population_only",
                temperature=1.0,
            )

            pred_sp["null_id"] = np.nan
            all_pred_rows.append(pred_sp)

            pearson_rows.append(
                summarize_pearson_by_cell_type(
                    pred_sp,
                    selected_fates,
                    model_name="starting_population_only",
                    fold=fold,
                    null_id=np.nan,
                )
            )

        # --------------------------------------------------------
        # 4. Startpop-preserving shuffled CIPHER null
        # --------------------------------------------------------
        for null_id in range(N_NULLS):
            if USE_STARTPOP_PRESERVING_NULL and START_COL in meta.columns:
                Ytrain_null, Ctrain_null = shuffle_YC_within_groups(Ytrain, Ctrain, start_train)
                null_name = "startpop_preserving_null_bayes_H0"
            else:
                perm = rng.permutation(Ytrain.shape[0])
                Ytrain_null = Ytrain[perm]
                Ctrain_null = Ctrain[perm]
                null_name = "shuffled_null_bayes_H0"

            null_model = make_composition_cipher_bayes_model_v2(
                Xtrain_clone=Xtrain,
                Ytrain=Ytrain_null,
                selected_fates=selected_fates,
                evals=evals,
                evecs=evecs,
                tau2=POSTERIOR_TAU2,
                use_fate_prior=USE_FATE_PRIOR,
                use_uncertainty_penalty=USE_POSTERIOR_UNCERTAINTY_PENALTY,
                activity_threshold=POSTERIOR_ACTIVITY_THRESHOLD,
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

            raw_null, logits_null, P_null = score_composition_cipher(
                Xtest,
                null_model,
                temperature=T_null,
            )

            null_df = add_prediction_columns(
                base,
                raw_null,
                logits_null,
                P_null,
                selected_fates,
                null_name,
                temperature=T_null,
            )

            null_df["null_id"] = null_id

            pearson_rows.append(
                summarize_pearson_by_cell_type(
                    null_df,
                    selected_fates,
                    model_name=null_name,
                    fold=fold,
                    null_id=null_id,
                )
            )


    # ============================================================
    # SAVE TABLES
    # ============================================================

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    pearson_metrics = pd.concat(pearson_rows, ignore_index=True)

    pearson_by_cell_type, pearson_model_mean = aggregate_pearson_tables(pearson_metrics)

    force_df = pd.DataFrame(force_rows)
    temperature_df = pd.DataFrame(temperature_rows)
    posterior_fit_df = pd.DataFrame(posterior_fit_rows)
    lfc_fit_df = pd.DataFrame(lfc_fit_rows)

    predictions.to_csv(
        os.path.join(OUTDIR, "clone_composition_predictions_with_LFC_bayes_H0.csv"),
        index=False,
    )

    pearson_metrics.to_csv(
        os.path.join(OUTDIR, "pearson_metrics_by_fold_celltype_with_LFC_bayes_H0.csv"),
        index=False,
    )

    pearson_by_cell_type.to_csv(
        os.path.join(OUTDIR, "pearson_by_celltype_with_LFC_bayes_H0.csv"),
        index=False,
    )

    pearson_model_mean.to_csv(
        os.path.join(OUTDIR, "mean_pearson_across_celltypes_with_LFC_bayes_H0.csv"),
        index=False,
    )

    force_df.to_csv(
        os.path.join(OUTDIR, "composition_CIPHER_bayes_H0_force_genes.csv"),
        index=False,
    )

    temperature_df.to_csv(
        os.path.join(OUTDIR, "fold_temperature_values_with_LFC_bayes_H0.csv"),
        index=False,
    )

    posterior_fit_df.to_csv(
        os.path.join(OUTDIR, "posterior_fit_summary_by_fate_fold_bayes_H0.csv"),
        index=False,
    )

    lfc_fit_df.to_csv(
        os.path.join(OUTDIR, "terminal_vs_undiff_LFC_fit_summary.csv"),
        index=False,
    )

    print("\nSaved:")
    print(os.path.join(OUTDIR, "clone_composition_predictions_with_LFC_bayes_H0.csv"))
    print(os.path.join(OUTDIR, "pearson_metrics_by_fold_celltype_with_LFC_bayes_H0.csv"))
    print(os.path.join(OUTDIR, "pearson_by_celltype_with_LFC_bayes_H0.csv"))


    # ============================================================
    # REFERENCE-STYLE BOXPLOT
    # ============================================================

    plot_df = pearson_metrics.copy()
    plot_df = plot_df[np.isfinite(plot_df["pearson"])].copy()

    model_label_map = {
        "cipher_bayes_H0": "Bayesian CIPHER, H0",
        "terminal_vs_undiff_LFC": "terminal-vs-undiff LFC",
        "starting_population_only": "starting-pop only",
        "startpop_preserving_null_bayes_H0": "startpop-preserving null",
        "shuffled_null_bayes_H0": "shuffled null",
    }

    model_order_raw = [
        "cipher_bayes_H0",
        "terminal_vs_undiff_LFC",
        "starting_population_only",
        "startpop_preserving_null_bayes_H0",
        "shuffled_null_bayes_H0",
    ]

    model_order_raw = [m for m in model_order_raw if m in plot_df["model"].unique()]
    model_order = [model_label_map[m] for m in model_order_raw]

    plot_df["model_label"] = plot_df["model"].map(model_label_map).fillna(plot_df["model"])

    fate_order = [f for f in selected_fates if f in plot_df["cell_type"].unique()]

    plot_df["cell_type"] = pd.Categorical(
        plot_df["cell_type"],
        categories=fate_order,
        ordered=True,
    )

    plot_df["model_label"] = pd.Categorical(
        plot_df["model_label"],
        categories=model_order,
        ordered=True,
    )

    plot_df = plot_df.sort_values(["cell_type", "model_label"]).copy()

    sns.set_context("talk")
    sns.set_style("white")

    palette = {
        "Bayesian CIPHER, H0": "#1f77b4",
        "terminal-vs-undiff LFC": "#9467bd",
        "starting-pop only": "#ff7f0e",
        "startpop-preserving null": "#2ca02c",
        "shuffled null": "#7f7f7f",
    }

    palette = {k: v for k, v in palette.items() if k in model_order}

    fig, ax = plt.subplots(figsize=(9.2, 4.7))

    sns.boxplot(
        data=plot_df,
        x="cell_type",
        y="pearson",
        hue="model_label",
        order=fate_order,
        hue_order=model_order,
        palette=palette,
        width=0.72,
        linewidth=1.4,
        fliersize=0,
        ax=ax,
    )

    sns.stripplot(
        data=plot_df,
        x="cell_type",
        y="pearson",
        hue="model_label",
        order=fate_order,
        hue_order=model_order,
        dodge=True,
        color="black",
        alpha=0.55,
        size=3.4,
        jitter=0.15,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    n = len(model_order)
    handles = handles[:n]
    labels = labels[:n]

    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title=None,
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.4)

    ax.set_title("Predicted vs observed terminal fate fraction: pearson")
    ax.set_xlabel("")
    ax.set_ylabel("pearson")
    ax.set_ylim(-1.0, 1.0)

    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")

    ax.grid(False)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTDIR, "pearson_boxplot_by_celltype_with_LFC_reference_style.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        os.path.join(OUTDIR, "pearson_boxplot_by_celltype_with_LFC_reference_style.svg"),
        bbox_inches="tight",
    )

    plt.savefig(
        os.path.join(OUTDIR, "pearson_boxplot_by_celltype_with_LFC_reference_style.pdf"),
        bbox_inches="tight",
    )

    plt.show()


    # ============================================================
    # OPTIONAL: MEAN ACROSS CELL TYPES ± SEM
    # ============================================================

    mean_plot = pearson_model_mean.copy()
    mean_plot["model_label"] = mean_plot["model"].map(model_label_map).fillna(mean_plot["model"])

    mean_plot = (
        mean_plot
        .set_index("model")
        .loc[model_order_raw]
        .reset_index()
    )

    x = np.arange(len(mean_plot))

    plt.figure(figsize=(7.5, 4.8))

    plt.bar(
        x,
        mean_plot["mean_pearson_across_cell_types"].values,
        yerr=mean_plot["sem_across_cell_types"].values,
        capsize=6,
        edgecolor="black",
        linewidth=1.2,
    )

    plt.axhline(0, color="gray", linestyle="--", linewidth=1.4)

    plt.xticks(
        x,
        mean_plot["model_label"].values,
        rotation=30,
        ha="right",
    )

    plt.ylabel("mean pearson across cell types")
    plt.title("Mean terminal fate fraction prediction: pearson ± SEM")
    plt.ylim(-1.0, 1.0)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")

    plt.grid(False)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTDIR, "mean_pearson_across_celltypes_with_LFC_reference_style.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        os.path.join(OUTDIR, "mean_pearson_across_celltypes_with_LFC_reference_style.svg"),
        bbox_inches="tight",
    )

    plt.savefig(
        os.path.join(OUTDIR, "mean_pearson_across_celltypes_with_LFC_reference_style.pdf"),
        bbox_inches="tight",
    )

    plt.show()


    print("\nPearson by cell type:")
    try:
        display(pearson_by_cell_type)
    except NameError:
        print(pearson_by_cell_type)

    print("\nMean Pearson across cell types:")
    try:
        display(pearson_model_mean)
    except NameError:
        print(pearson_model_mean)



def pearson_boxplot():
    global os, np, pd, plt, mpatches, pearson_metrics, plot_df, STRICT_MODEL_ORDER
    global MODEL_COLORS, fate_order, preferred, observed, global_rows, tmp_global, model, model_label
    global fold, null_id, sub, vals, global_df, base_cols, col, plot_df2
    global x_order, model_rank, fate_rank, counts_table, fig, ax, box_width, offset_values
    global rng_plot, fate, xi, mi, pos, jitter, spine, legend_handles
    global EARLY_TIME, OUTDIR, selected_fates
    # ============================================================
    # FINAL PEARSON BOXPLOT
    #
    # Exact model order inside EVERY cell type:
    #   1. CIPHER
    #   2. terminal-vs-undiff LFC
    #   3. startpop-preserving null
    #   4. starting-pop only
    #
    # GLOBAL_OVR:
    #   simple unweighted average over individual cell-type Pearson values
    #
    # Style:
    #   - same AUPRC-style colors
    #   - no black outlines around colored boxes
    #   - no jitter points for startpop-preserving null
    # ============================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ------------------------------------------------------------
    # Load Pearson metrics if needed
    # ------------------------------------------------------------

    if "pearson_metrics" not in globals():
        pearson_metrics = pd.read_csv(
            os.path.join(
                OUTDIR,
                "pearson_metrics_by_fold_celltype_with_LFC_bayes_H0.csv",
            )
        )

    plot_df = pearson_metrics.copy()
    plot_df = plot_df[np.isfinite(plot_df["pearson"])].copy()

    # ------------------------------------------------------------
    # Canonical labels, strict order, and colors
    # ------------------------------------------------------------


    STRICT_MODEL_ORDER = [
        "CIPHER",
        "terminal-vs-undiff LFC",
        "startpop-preserving null",
        "starting-pop only",
    ]

    MODEL_COLORS = {
        "CIPHER": "#1f77b4",
        "terminal-vs-undiff LFC": "#d62728",
        "startpop-preserving null": "#ff7f0e",
        "starting-pop only": "#2ca02c",
    }

    plot_df["model_label"] = plot_df["model"].apply(canonical_model_label)
    plot_df = plot_df[plot_df["model_label"].isin(STRICT_MODEL_ORDER)].copy()

    if "null_id" not in plot_df.columns:
        plot_df["null_id"] = np.nan

    # ------------------------------------------------------------
    # Cell-type / fate order
    # ------------------------------------------------------------

    if "selected_fates" in globals():
        fate_order = [
            f for f in selected_fates
            if f in set(plot_df["cell_type"].astype(str))
        ]
    else:
        preferred = ["Monocyte", "Neutrophil", "Baso", "Mast", "Meg", "Erythroid", "Eos"]
        observed = [
            x for x in plot_df["cell_type"].astype(str).drop_duplicates().tolist()
            if x != "GLOBAL_OVR"
        ]

        fate_order = [f for f in preferred if f in observed]
        fate_order += [f for f in observed if f not in fate_order]

    # Make sure GLOBAL_OVR is not already included before recomputing it.
    plot_df = plot_df[plot_df["cell_type"].astype(str) != "GLOBAL_OVR"].copy()

    # ------------------------------------------------------------
    # Build GLOBAL_OVR as simple average over per-cell-type Pearson
    #
    # For each model/fold/null:
    #
    #   GLOBAL_OVR = mean([
    #       pearson_Monocyte,
    #       pearson_Neutrophil,
    #       pearson_Baso,
    #       ...
    #   ])
    #
    # No clone x fate pooling.
    # No weighting by clone count.
    # ------------------------------------------------------------

    global_rows = []

    tmp_global = plot_df[
        plot_df["cell_type"].isin(fate_order) &
        plot_df["model_label"].isin(STRICT_MODEL_ORDER)
    ].copy()

    for (model, model_label, fold, null_id), sub in tmp_global.groupby(
        ["model", "model_label", "fold", "null_id"],
        dropna=False,
    ):
        vals = sub["pearson"].values.astype(float)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            continue

        global_rows.append({
            "model": model,
            "model_label": model_label,
            "fold": fold,
            "null_id": null_id,
            "cell_type": "GLOBAL_OVR",
            "pearson": float(np.mean(vals)),
            "n_clones": np.nan,
            "n_cell_types_used": int(len(vals)),
        })

    global_df = pd.DataFrame(global_rows)

    # ------------------------------------------------------------
    # Combine individual cell types + GLOBAL_OVR
    # ------------------------------------------------------------

    base_cols = [
        "model",
        "model_label",
        "fold",
        "null_id",
        "cell_type",
        "pearson",
        "n_clones",
    ]

    for col in base_cols:
        if col not in plot_df.columns:
            plot_df[col] = np.nan

    for col in base_cols:
        if col not in global_df.columns:
            global_df[col] = np.nan

    plot_df2 = pd.concat(
        [
            plot_df[base_cols].copy(),
            global_df[base_cols].copy(),
        ],
        ignore_index=True,
    )

    plot_df2 = plot_df2[np.isfinite(plot_df2["pearson"])].copy()
    plot_df2 = plot_df2[plot_df2["model_label"].isin(STRICT_MODEL_ORDER)].copy()

    x_order = fate_order + ["GLOBAL_OVR"]

    # ------------------------------------------------------------
    # Numeric ranks enforce exact order.
    # This avoids seaborn hue-order weirdness completely.
    # ------------------------------------------------------------

    model_rank = {m: i for i, m in enumerate(STRICT_MODEL_ORDER)}
    fate_rank = {f: i for i, f in enumerate(x_order)}

    plot_df2["model_rank"] = plot_df2["model_label"].map(model_rank)
    plot_df2["fate_rank"] = plot_df2["cell_type"].map(fate_rank)

    plot_df2 = plot_df2.dropna(subset=["model_rank", "fate_rank"]).copy()
    plot_df2["model_rank"] = plot_df2["model_rank"].astype(int)
    plot_df2["fate_rank"] = plot_df2["fate_rank"].astype(int)

    plot_df2 = plot_df2.sort_values(
        ["fate_rank", "model_rank", "fold", "null_id"],
        kind="mergesort",
    ).copy()

    print("\nCounts by fate/model:")
    counts_table = (
        plot_df2
        .groupby(["cell_type", "model_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=x_order, columns=STRICT_MODEL_ORDER)
        .fillna(0)
        .astype(int)
    )
    print(counts_table)

    print("\nGLOBAL_OVR medians:")
    print(
        plot_df2[plot_df2["cell_type"].astype(str) == "GLOBAL_OVR"]
        .groupby("model_label")["pearson"]
        .median()
        .reindex(STRICT_MODEL_ORDER)
    )

    # ------------------------------------------------------------
    # Manual boxplot with exact physical positions
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6.8, 3.2))

    box_width = 0.16

    # Exact left-to-right order:
    # CIPHER, LFC, null, starting-pop.
    offset_values = np.array([-1.5, -0.5, 0.5, 1.5]) * box_width

    rng_plot = np.random.default_rng(123)

    for fate in x_order:
        xi = fate_rank[fate]

        for model in STRICT_MODEL_ORDER:
            mi = model_rank[model]
            pos = xi + offset_values[mi]

            vals = plot_df2[
                (plot_df2["cell_type"].astype(str) == str(fate)) &
                (plot_df2["model_label"].astype(str) == str(model))
            ]["pearson"].values.astype(float)

            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                continue

            ax.boxplot(
                vals,
                positions=[pos],
                widths=box_width * 0.82,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,

                # No black outlines around boxes.
                boxprops=dict(
                    facecolor=MODEL_COLORS[model],
                    edgecolor=MODEL_COLORS[model],
                    linewidth=0.0,
                    alpha=0.90,
                ),

                medianprops=dict(
                    color="black",
                    linewidth=1.1,
                ),
                whiskerprops=dict(
                    color="0.35",
                    linewidth=0.9,
                ),
                capprops=dict(
                    color="0.35",
                    linewidth=0.9,
                ),
            )

            # Jitter points for all except startpop-preserving null.
            if model != "startpop-preserving null":
                jitter = rng_plot.normal(
                    loc=0.0,
                    scale=box_width * 0.06,
                    size=len(vals),
                )

                ax.scatter(
                    np.full(len(vals), pos) + jitter,
                    vals,
                    s=8,
                    color="black",
                    alpha=0.55,
                    zorder=3,
                    linewidths=0,
                )

    # ------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)

    ax.set_xticks(np.arange(len(x_order)))
    ax.set_xticklabels(x_order, rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("pearson", fontsize=10)
    ax.set_xlabel("")
    ax.set_title(f"early{EARLY_TIME}_all_startpops: clone-level pearson", fontsize=11)

    # Match your reference-style positive Pearson plot.
    ax.set_ylim(-0.1, 1.0)

    ax.tick_params(axis="y", labelsize=9)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    ax.grid(False)

    legend_handles = [
        mpatches.Patch(
            facecolor=MODEL_COLORS[m],
            edgecolor=MODEL_COLORS[m],
            linewidth=0.0,
            label=m,
            alpha=0.90,
        )
        for m in STRICT_MODEL_ORDER
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.08, -0.42),
        ncol=1,
        title=None,
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTDIR,
            "early4_all_startpops_clone_level_pearson_boxplot_SIMPLE_AVG_GLOBAL_OVR.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        os.path.join(
            OUTDIR,
            "early4_all_startpops_clone_level_pearson_boxplot_SIMPLE_AVG_GLOBAL_OVR.svg",
        ),
        bbox_inches="tight",
    )

    plt.savefig(
        os.path.join(
            OUTDIR,
            "early4_all_startpops_clone_level_pearson_boxplot_SIMPLE_AVG_GLOBAL_OVR.pdf",
        ),
        bbox_inches="tight",
    )

    plt.show()



def cv_fullmatrix_l2_sweep():
    global logsumexp, minimize, StratifiedKFold, confusion_matrix, OUTDIR, FULL_MATRIX_L2_VALUES, DIAG_AFFINE_L2_VALUES, INCLUDE_TEMPERATURE_BASELINE
    global INCLUDE_EMPIRICAL_PARTITION, INCLUDE_STARTPOP_VARIANTS, INCLUDE_CLONE_WEIGHT_SCALING_VARIANTS, EMPIRICAL_PARTITION_BASELINE, SATURATING_N0, SMOOTH_ALPHA, CALIB_MAXITER, CALIB_MAXFUN
    global CALIB_FTOL, CALIB_GTOL, TOP_FORCE_GENES_PER_DIRECTION, MODEL_SPECS, l2, mode, mean_count_nll, fit_logit_calibrator
    global build_raw_logits_variant, add_smoothed_obs_columns, add_prediction_columns_from_probs, add_composition_errors_extended, summarize_predictions_extended, X_clones_all, strat_y, min_class_n
    global n_splits, splitter, clone_to_obs, clone_to_counts, clone_to_start, all_pred_rows, summary_rows, posterior_fit_rows
    global calibration_rows, force_rows, calibration_param_rows, fold, train_pos, test_pos, train_clones, test_clones
    global Xtrain, train_clone_ids_used, n_train_early, Xtest, test_clone_ids_used, n_test_early, Ytrain, Ctrain
    global Ytest, Ctest, start_train, start_test, true_dom_test, train_fate_prior, base_test, j
    global fate, s, cipher_model, startpop_prior_model, X_baseline_partition, u, delta, yhat
    global std, z, pip, sign_conf, p_pos, p_neg, ci_lo, ci_hi
    global zero_exc, ranking_sets, direction, idxs, rank, gi, spec, model_name
    global partition, use_startpop, clone_weight_mode, calibrator_mode, logits_train_raw, logits_test_raw, cal, logits_train_cal
    global logits_test_cal, Ptrain, Ptest, train_nll, train_mean_nll, test_nll, test_mean_nll, W
    global b, i, fate_out, fate_in, a, pred_df, predictions, summary_metrics
    global calibration_df, posterior_fit_df, force_df, calibration_params_df, composition_summary, per_fate_summary, best_row, best_model
    global full_sweep, plot_top, plot_order, best_pred, n_fates, ncols, nrows, fig
    global axes, ax, x, xs, y, r, rho, r2
    global rs, rhos, r2s, k, obs_cols, obs_s_cols, pred_cols, obs_heat
    global obs_s_heat, pred_heat, cm, cm_norm, best_params, W_mean, add_composition_errors, cell_to_clone
    global clone_mean_matrix, clone_table, cosine_similarity, counts, early_cloned_mask, evals, evecs, hvg_genes
    global hvg_idx, js_div, make_composition_cipher_bayes_model, meta, mu_ref, N_SPLITS, obs_frac_cols, POSTERIOR_ACTIVITY_THRESHOLD
    global POSTERIOR_TAU2, RESTRICT_STARTING_POPULATION, safe_corr, safe_name, safe_r2, sd_ref, SEED, selected_count_cols
    global selected_fates, Sigma, START_COL, USE_FATE_PRIOR, USE_POSTERIOR_UNCERTAINTY_PENALTY, Xcov
    # ============================================================
    # CROSS-VALIDATED COMPOSITION PREDICTION
    # KEEP WHAT WORKED + ADD FULL-MATRIX L2 SWEEP
    # ============================================================
    #
    # Keeps:
    #   - Gaussian Bayesian CIPHER posterior force model
    #   - Full-matrix logit calibration, because it worked best
    #   - Diagonal affine and temperature as baselines
    #   - Empirical variants as comparison
    #
    # Adds:
    #   - Much larger optimizer budget for full-matrix calibration
    #   - Ridge/L2 sweep for full_matrix calibration
    #   - Best-model selection by CV test count NLL
    #   - Separate "winner" plots and outputs
    #
    # This block assumes all earlier setup has already run:
    #   counts, meta, clone_table, Sigma, evals, evecs, hvg_idx, hvg_genes,
    #   Xcov, mu_ref, sd_ref, selected_fates, obs_frac_cols, selected_count_cols,
    #   early_cloned_mask, cell_to_clone, etc.
    # ============================================================

    from scipy.special import logsumexp
    from scipy.optimize import minimize
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix

    # ============================================================
    # OUTPUT / MODEL CONFIG
    # ============================================================

    OUTDIR = os.path.join(OUT_BASE, "cipher_larry_clone_fate_composition_bayes_H0_fullmatrix_L2_sweep")
    os.makedirs(OUTDIR, exist_ok=True)

    # Full-matrix L2 sweep.
    # The previous run used 1e-1 and worked best among tested models,
    # but train/test gap suggests we should sweep around it.
    FULL_MATRIX_L2_VALUES = [
        1e-4, 3e-4,
        1e-3, 3e-3,
        1e-2, 3e-2,
        1e-1, 3e-1,
        1.0, 3.0,
    ]

    # Diagonal affine baseline L2 sweep, smaller because fewer params.
    DIAG_AFFINE_L2_VALUES = [
        1e-4, 1e-3, 1e-2, 1e-1,
    ]

    # Keep temperature as weak baseline.
    INCLUDE_TEMPERATURE_BASELINE = True

    # Keep empirical partition comparison, but based on previous result it is not expected to win.
    INCLUDE_EMPIRICAL_PARTITION = True

    # Keep startpop comparison, but do not make it the default because it hurt last run.
    INCLUDE_STARTPOP_VARIANTS = True

    # Do NOT use clone-count scaling by default because it hurt last run.
    INCLUDE_CLONE_WEIGHT_SCALING_VARIANTS = False

    # Baseline for empirical log-partition.
    # "train_clones" is much cheaper and was essentially same as Gaussian after calibration.
    # "early_cells" is possible but can be more expensive.
    EMPIRICAL_PARTITION_BASELINE = "train_clones"

    SATURATING_N0 = 5.0

    # Smoothed observed fractions used only for diagnostics/plots.
    SMOOTH_ALPHA = 5.0

    # Optimizer budget. This fixes the warning from previous run.
    CALIB_MAXITER = 10000
    CALIB_MAXFUN = 50000
    CALIB_FTOL = 1e-10
    CALIB_GTOL = 1e-6

    # Save force genes once per fold.
    TOP_FORCE_GENES_PER_DIRECTION = 50


    # ============================================================
    # MODEL SPECS
    # ============================================================

    MODEL_SPECS = []

    if INCLUDE_TEMPERATURE_BASELINE:
        MODEL_SPECS.append({
            "name": "gaussian_temperature",
            "partition": "gaussian",
            "startpop_prior": False,
            "clone_weight_mode": "none",
            "calibrator": "temperature",
            "l2": 0.0,
        })

    # Diagonal affine baselines.
    for l2 in DIAG_AFFINE_L2_VALUES:
        MODEL_SPECS.append({
            "name": f"gaussian_diag_affine_l2_{l2:g}",
            "partition": "gaussian",
            "startpop_prior": False,
            "clone_weight_mode": "none",
            "calibrator": "diag_affine",
            "l2": float(l2),
        })

    # Main winner family: Gaussian full matrix with L2 sweep.
    for l2 in FULL_MATRIX_L2_VALUES:
        MODEL_SPECS.append({
            "name": f"gaussian_full_matrix_l2_{l2:g}",
            "partition": "gaussian",
            "startpop_prior": False,
            "clone_weight_mode": "none",
            "calibrator": "full_matrix",
            "l2": float(l2),
        })

    # Empirical partition variants.
    if INCLUDE_EMPIRICAL_PARTITION:
        for l2 in DIAG_AFFINE_L2_VALUES:
            MODEL_SPECS.append({
                "name": f"empirical_diag_affine_l2_{l2:g}",
                "partition": "empirical",
                "startpop_prior": False,
                "clone_weight_mode": "none",
                "calibrator": "diag_affine",
                "l2": float(l2),
            })

        for l2 in FULL_MATRIX_L2_VALUES:
            MODEL_SPECS.append({
                "name": f"empirical_full_matrix_l2_{l2:g}",
                "partition": "empirical",
                "startpop_prior": False,
                "clone_weight_mode": "none",
                "calibrator": "full_matrix",
                "l2": float(l2),
            })

    # Start-pop prior variants, kept as comparison only.
    if INCLUDE_STARTPOP_VARIANTS:
        for l2 in FULL_MATRIX_L2_VALUES:
            MODEL_SPECS.append({
                "name": f"empirical_startpop_full_matrix_l2_{l2:g}",
                "partition": "empirical",
                "startpop_prior": True,
                "clone_weight_mode": "none",
                "calibrator": "full_matrix",
                "l2": float(l2),
            })

    # Clone-count scaling variants are off by default because they hurt before.
    if INCLUDE_CLONE_WEIGHT_SCALING_VARIANTS:
        for mode in ["sqrt", "saturating"]:
            for l2 in [1e-2, 1e-1, 1.0]:
                MODEL_SPECS.append({
                    "name": f"empirical_startpop_{mode}_full_matrix_l2_{l2:g}",
                    "partition": "empirical",
                    "startpop_prior": True,
                    "clone_weight_mode": mode,
                    "calibrator": "full_matrix",
                    "l2": float(l2),
                })

    print(f"Testing {len(MODEL_SPECS)} calibrated models.")


    # ============================================================
    # CALIBRATION HELPERS
    # ============================================================


    def mean_count_nll(counts, probs, eps=1e-12):
        counts = np.asarray(counts, dtype=float)
        return count_nll(counts, probs, eps=eps) / np.maximum(counts.sum(), eps)


    def fit_logit_calibrator(
        train_logits,
        train_counts,
        mode="full_matrix",
        l2=1e-1,
        maxiter=10000,
        maxfun=50000,
        ftol=1e-10,
        gtol=1e-6,
    ):
        """
    Fits calibration from raw CIPHER logits to terminal fate probabilities.

    mode:
      temperature:
          Z = L / T

      diag_affine:
          Z_f = a_f L_f + b_f

      full_matrix:
          Z = L W^T + b

    All calibrators minimize terminal-count NLL on training clones.
    """
        L = np.asarray(train_logits, dtype=float)
        C = np.asarray(train_counts, dtype=float)

        n, k = L.shape

        if C.shape != L.shape:
            raise ValueError("train_logits and train_counts must have same shape")

        def nll_from_logits(Z):
            logP = Z - logsumexp(Z, axis=1, keepdims=True)
            return -np.sum(C * logP)

        if mode == "temperature":

            def transform_from_theta(theta, X):
                T = np.exp(theta[0])
                return X / np.maximum(T, 1e-12)

            def obj(theta):
                Z = transform_from_theta(theta, L)
                return nll_from_logits(Z)

            theta0 = np.array([0.0])

        elif mode == "diag_affine":

            def transform_from_theta(theta, X):
                log_a = theta[:k]
                b = theta[k:]
                a = np.exp(log_a)
                return X * a[None, :] + b[None, :]

            def obj(theta):
                log_a = theta[:k]
                b = theta[k:]
                Z = transform_from_theta(theta, L)

                # identity regularization
                reg = l2 * (np.sum(log_a ** 2) + np.sum(b ** 2))
                return nll_from_logits(Z) + reg

            theta0 = np.concatenate([
                np.zeros(k),  # log_a
                np.zeros(k),  # b
            ])

        elif mode == "full_matrix":

            def transform_from_theta(theta, X):
                W = theta[:k * k].reshape(k, k)
                b = theta[k * k:]
                return X @ W.T + b[None, :]

            def obj(theta):
                W = theta[:k * k].reshape(k, k)
                b = theta[k * k:]
                Z = transform_from_theta(theta, L)

                # identity regularization
                I = np.eye(k)
                reg = l2 * (np.sum((W - I) ** 2) + np.sum(b ** 2))
                return nll_from_logits(Z) + reg

            theta0 = np.concatenate([
                np.eye(k).reshape(-1),
                np.zeros(k),
            ])

        else:
            raise ValueError("mode must be: temperature, diag_affine, full_matrix")

        res = minimize(
            obj,
            theta0,
            method="L-BFGS-B",
            options={
                "maxiter": int(maxiter),
                "maxfun": int(maxfun),
                "ftol": float(ftol),
                "gtol": float(gtol),
            },
        )

        if not res.success:
            print(f"[warning] calibration did not fully converge: {res.message}")

        theta = res.x

        def transform(new_logits):
            return transform_from_theta(theta, np.asarray(new_logits, dtype=float))

        train_cal_logits = transform(L)
        train_probs = softmax_np(train_cal_logits)

        # Extract interpretable calibration pieces.
        out = {
            "mode": mode,
            "theta": theta,
            "success": bool(res.success),
            "message": str(res.message),
            "nfev": getattr(res, "nfev", np.nan),
            "nit": getattr(res, "nit", np.nan),
            "train_nll": count_nll(C, train_probs),
            "train_mean_nll": mean_count_nll(C, train_probs),
            "transform": transform,
            "train_probs": train_probs,
        }

        if mode == "temperature":
            out["temperature"] = float(np.exp(theta[0]))

        elif mode == "diag_affine":
            log_a = theta[:k]
            b = theta[k:]
            out["diag_scale"] = np.exp(log_a)
            out["bias"] = b

        elif mode == "full_matrix":
            W = theta[:k * k].reshape(k, k)
            b = theta[k * k:]
            out["W"] = W
            out["bias"] = b
            out["W_fro_norm_minus_I"] = float(np.linalg.norm(W - np.eye(k)))

        return out


    def build_raw_logits_variant(
        X,
        n_early,
        cipher_model,
        partition="gaussian",
        X_baseline_for_partition=None,
        start_values=None,
        startpop_prior_model=None,
        use_startpop_prior=False,
        clone_weight_mode="none",
        saturating_n0=5.0,
        eps=1e-12,
    ):
        """
    Builds raw logits before calibration.

    Gaussian partition:
        logit_cf = x_c^T u_f - posterior penalty_f

    Empirical partition:
        logit_cf = x_c^T u_f - logmeanexp_{x0}(x0^T u_f)

    Optional start-pop prior:
        logit_cf += log P(f | starting population)

    Optional clone-count confidence scaling:
        logit_cf *= w(n_early)
    """
        X = np.asarray(X, dtype=float)
        U = np.asarray(cipher_model["U"], dtype=float)

        raw = X @ U.T

        if partition == "gaussian":
            A = np.asarray(cipher_model["penalty"], dtype=float)

        elif partition == "empirical":
            if X_baseline_for_partition is None:
                raise ValueError("X_baseline_for_partition is required for empirical partition")
            A = empirical_log_partition(X_baseline_for_partition, U)

        else:
            raise ValueError("partition must be: gaussian or empirical")

        logits = raw - A[None, :]

        w = clone_weight_from_n(
            n_early,
            mode=clone_weight_mode,
            saturating_n0=saturating_n0,
        )
        logits = logits * w[:, None]

        if "log_prior" in cipher_model:
            logits = logits + np.asarray(cipher_model["log_prior"], dtype=float)[None, :]

        if use_startpop_prior:
            if start_values is None or startpop_prior_model is None:
                raise ValueError("start_values and startpop_prior_model are required for startpop prior")

            P_start = predict_startpop_prior(start_values, startpop_prior_model)
            logits = logits + np.log(np.clip(P_start, eps, 1.0))

        return logits


    # ============================================================
    # PREDICTION / SUMMARY HELPERS
    # ============================================================

    def add_smoothed_obs_columns(df, selected_fates, prior, alpha=5.0):
        count_cols = [f"terminal_count__{safe_name(f)}" for f in selected_fates]
        C = df[count_cols].values.astype(float)

        Ys = smoothed_observed_fractions(C, prior=prior, alpha=alpha)

        out = df.copy()

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            out[f"obs_frac_smooth__{s}"] = Ys[:, j]

        out["obs_entropy_smooth"] = entropy(Ys)

        return out


    def add_prediction_columns_from_probs(
        base_df,
        logits_raw,
        logits_cal,
        P,
        selected_fates,
        model_name,
        calibrator_name,
        partition,
        startpop_prior,
        clone_weight_mode,
        l2,
        train_mean_nll=np.nan,
    ):
        rows = base_df.copy()

        rows["model"] = model_name
        rows["calibrator"] = calibrator_name
        rows["partition"] = partition
        rows["use_startpop_prior"] = bool(startpop_prior)
        rows["clone_weight_mode"] = clone_weight_mode
        rows["l2"] = float(l2)
        rows["train_mean_nll"] = float(train_mean_nll)

        pred_idx = np.argmax(P, axis=1)

        rows["pred_dominant_fate"] = np.array(selected_fates, dtype=object)[pred_idx]
        rows["pred_entropy"] = entropy(P)
        rows["pred_max_prob"] = P.max(axis=1)

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)

            rows[f"logit_raw__{s}"] = logits_raw[:, j]
            rows[f"logit_cal__{s}"] = logits_cal[:, j]
            rows[f"pred_frac__{s}"] = P[:, j]

        return rows


    def add_composition_errors_extended(df, selected_fates):
        df = add_composition_errors(df, selected_fates)

        obs_smooth_cols = [f"obs_frac_smooth__{safe_name(f)}" for f in selected_fates]
        pred_cols = [f"pred_frac__{safe_name(f)}" for f in selected_fates]

        if all(c in df.columns for c in obs_smooth_cols):
            obs_s = df[obs_smooth_cols].values.astype(float)
            pred = df[pred_cols].values.astype(float)

            df["composition_JS_smooth"] = js_div(obs_s, pred)
            df["composition_Brier_smooth"] = np.mean((obs_s - pred) ** 2, axis=1)
            df["composition_L1_smooth"] = np.sum(np.abs(obs_s - pred), axis=1)
            df["composition_cosine_smooth"] = cosine_similarity(obs_s, pred)

        count_cols = [f"terminal_count__{safe_name(f)}" for f in selected_fates]

        if all(c in df.columns for c in count_cols):
            C = df[count_cols].values.astype(float)
            pred = df[pred_cols].values.astype(float)

            per_clone_nll = -np.sum(C * np.log(np.clip(pred, 1e-12, 1.0)), axis=1)
            n_terminal = np.maximum(C.sum(axis=1), 1.0)

            df["count_nll"] = per_clone_nll
            df["mean_count_nll_per_cell"] = per_clone_nll / n_terminal

        return df


    def summarize_predictions_extended(df, selected_fates, model_name, fold):
        rows = []

        obs = df[[f"obs_frac__{safe_name(f)}" for f in selected_fates]].values.astype(float)
        pred = df[[f"pred_frac__{safe_name(f)}" for f in selected_fates]].values.astype(float)

        count_cols = [f"terminal_count__{safe_name(f)}" for f in selected_fates]
        C = df[count_cols].values.astype(float)

        base_row = {
            "model": model_name,
            "fold": fold,
            "metric_type": "composition",
            "fate": "ALL",
            "partition": df["partition"].iloc[0],
            "calibrator": df["calibrator"].iloc[0],
            "use_startpop_prior": bool(df["use_startpop_prior"].iloc[0]),
            "clone_weight_mode": df["clone_weight_mode"].iloc[0],
            "l2": float(df["l2"].iloc[0]),
            "train_mean_nll": float(df["train_mean_nll"].iloc[0]),
            "test_count_nll": count_nll(C, pred),
            "test_mean_count_nll": mean_count_nll(C, pred),
            "mean_KL": np.mean(kl_div(obs, pred)),
            "mean_JS": np.mean(js_div(obs, pred)),
            "mean_Brier": np.mean(np.mean((obs - pred) ** 2, axis=1)),
            "mean_L1": np.mean(np.sum(np.abs(obs - pred), axis=1)),
            "mean_cosine": np.mean(cosine_similarity(obs, pred)),
            "top1_accuracy": np.mean(np.argmax(obs, axis=1) == np.argmax(pred, axis=1)),
            "entropy_pearson": safe_corr(entropy(obs), entropy(pred), method="pearson"),
            "entropy_spearman": safe_corr(entropy(obs), entropy(pred), method="spearman"),
            "n_clones": len(df),
            "n_terminal_cells": int(C.sum()),
        }

        obs_s_cols = [f"obs_frac_smooth__{safe_name(f)}" for f in selected_fates]

        if all(c in df.columns for c in obs_s_cols):
            obs_s = df[obs_s_cols].values.astype(float)

            base_row.update({
                "mean_JS_smooth": np.mean(js_div(obs_s, pred)),
                "mean_Brier_smooth": np.mean(np.mean((obs_s - pred) ** 2, axis=1)),
                "mean_L1_smooth": np.mean(np.sum(np.abs(obs_s - pred), axis=1)),
                "mean_cosine_smooth": np.mean(cosine_similarity(obs_s, pred)),
                "top1_accuracy_smooth": np.mean(np.argmax(obs_s, axis=1) == np.argmax(pred, axis=1)),
            })

        rows.append(base_row)

        for j, fate in enumerate(selected_fates):
            y = obs[:, j]
            p = pred[:, j]

            row = {
                "model": model_name,
                "fold": fold,
                "metric_type": "per_fate_fraction",
                "fate": fate,
                "partition": df["partition"].iloc[0],
                "calibrator": df["calibrator"].iloc[0],
                "use_startpop_prior": bool(df["use_startpop_prior"].iloc[0]),
                "clone_weight_mode": df["clone_weight_mode"].iloc[0],
                "l2": float(df["l2"].iloc[0]),
                "train_mean_nll": float(df["train_mean_nll"].iloc[0]),
                "pearson": safe_corr(y, p, method="pearson"),
                "spearman": safe_corr(y, p, method="spearman"),
                "r2": safe_r2(y, p),
                "mae": np.mean(np.abs(y - p)),
                "rmse": np.sqrt(np.mean((y - p) ** 2)),
                "mean_obs_fraction": np.mean(y),
                "mean_pred_fraction": np.mean(p),
                "n_clones": len(df),
            }

            if all(c in df.columns for c in obs_s_cols):
                ys = df[f"obs_frac_smooth__{safe_name(fate)}"].values.astype(float)

                row.update({
                    "pearson_smooth": safe_corr(ys, p, method="pearson"),
                    "spearman_smooth": safe_corr(ys, p, method="spearman"),
                    "r2_smooth": safe_r2(ys, p),
                    "mae_smooth": np.mean(np.abs(ys - p)),
                    "rmse_smooth": np.sqrt(np.mean((ys - p) ** 2)),
                    "mean_obs_fraction_smooth": np.mean(ys),
                })

            rows.append(row)

        return pd.DataFrame(rows)


    # ============================================================
    # CROSS-VALIDATED MODEL COMPARISON
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

    all_pred_rows = []
    summary_rows = []
    posterior_fit_rows = []
    calibration_rows = []
    force_rows = []
    calibration_param_rows = []

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

        if Xtrain.shape[0] == 0 or Xtest.shape[0] == 0:
            print(f"[skip] Fold {fold} has no usable train/test clone means.")
            continue

        Ytrain = np.vstack([clone_to_obs[int(c)] for c in train_clone_ids_used])
        Ctrain = np.vstack([clone_to_counts[int(c)] for c in train_clone_ids_used])

        Ytest = np.vstack([clone_to_obs[int(c)] for c in test_clone_ids_used])
        Ctest = np.vstack([clone_to_counts[int(c)] for c in test_clone_ids_used])

        start_train = np.array([clone_to_start.get(int(c), "unknown") for c in train_clone_ids_used])
        start_test = np.array([clone_to_start.get(int(c), "unknown") for c in test_clone_ids_used])

        true_dom_test = np.array(selected_fates, dtype=object)[np.argmax(Ytest, axis=1)]

        train_fate_prior = Ctrain.sum(axis=0) + 1e-12
        train_fate_prior = train_fate_prior / train_fate_prior.sum()

        base_test = pd.DataFrame({
            "fold": fold,
            "clone_id": test_clone_ids_used,
            "n_early_scored": n_test_early,
            "dominant_starting_population": start_test,
            "true_dominant_fate": true_dom_test,
        })

        for j, fate in enumerate(selected_fates):
            s = safe_name(fate)
            base_test[f"obs_frac__{s}"] = Ytest[:, j]
            base_test[f"terminal_count__{s}"] = Ctest[:, j]

        base_test["obs_entropy"] = entropy(Ytest)
        base_test["n_terminal_selected"] = Ctest.sum(axis=1)

        base_test = add_smoothed_obs_columns(
            base_test,
            selected_fates,
            prior=train_fate_prior,
            alpha=SMOOTH_ALPHA,
        )

        # --------------------------------------------------------
        # Fit Bayesian CIPHER posterior force model.
        # --------------------------------------------------------
        cipher_model = make_composition_cipher_bayes_model(
            Xtrain_clone=Xtrain,
            Ytrain=Ytrain,
            selected_fates=selected_fates,
            evals=evals,
            evecs=evecs,
            Sigma=Sigma,
            tau2=POSTERIOR_TAU2,
            use_fate_prior=USE_FATE_PRIOR,
            use_uncertainty_penalty=USE_POSTERIOR_UNCERTAINTY_PENALTY,
            activity_threshold=POSTERIOR_ACTIVITY_THRESHOLD,
        )

        # Starting-pop prior model.
        startpop_prior_model = None

        if START_COL in meta.columns and RESTRICT_STARTING_POPULATION is None:
            startpop_prior_model = fit_startpop_prior_from_counts(
                Ctrain=Ctrain,
                start_train=start_train,
                alpha=2.0,
            )

        # Baseline for empirical log-partition.
        if EMPIRICAL_PARTITION_BASELINE == "train_clones":
            X_baseline_partition = Xtrain
        elif EMPIRICAL_PARTITION_BASELINE == "early_cells":
            X_baseline_partition = Xcov
        else:
            raise ValueError("EMPIRICAL_PARTITION_BASELINE must be 'train_clones' or 'early_cells'")

        # Posterior fit summaries.
        for j, fate in enumerate(selected_fates):
            posterior_fit_rows.append({
                "fold": fold,
                "fate": fate,
                "tau2": POSTERIOR_TAU2,
                "h0_scale": float(cipher_model["h0_scale"][j]),
                "n_eff_pos": float(cipher_model["n_eff_pos"][j]),
                "n_eff_neg": float(cipher_model["n_eff_neg"][j]),
                "posterior_delta_r2": float(cipher_model["posterior_delta_r2"][j]),
                "log_marginal": float(cipher_model["log_marginal"][j]),
                "penalty": float(cipher_model["penalty"][j]),
                "penalty_mean": float(cipher_model["penalty_mean"][j]),
                "penalty_uncertainty": float(cipher_model["penalty_uncertainty"][j]),
                "activity_threshold": float(cipher_model["activity_threshold"][j]),
                "log_prior": float(cipher_model["log_prior"][j]),
            })

        # Save top posterior force genes once per fold.
        for j, fate in enumerate(selected_fates):
            u = cipher_model["U"][j]
            delta = cipher_model["DELTAS"][j]
            yhat = cipher_model["YHAT_DELTAS"][j]
            std = cipher_model["posterior_std"][j]
            z = cipher_model["posterior_z"][j]
            pip = cipher_model["posterior_pip"][j]
            sign_conf = cipher_model["posterior_sign_conf"][j]
            p_pos = cipher_model["posterior_p_pos"][j]
            p_neg = cipher_model["posterior_p_neg"][j]
            ci_lo = cipher_model["posterior_ci95_lo"][j]
            ci_hi = cipher_model["posterior_ci95_hi"][j]
            zero_exc = cipher_model["posterior_zero_excluded"][j]

            ranking_sets = [
                ("positive", np.argsort(u)[::-1][:TOP_FORCE_GENES_PER_DIRECTION]),
                ("negative", np.argsort(u)[:TOP_FORCE_GENES_PER_DIRECTION]),
                ("high_abs_z", np.argsort(np.abs(z))[::-1][:TOP_FORCE_GENES_PER_DIRECTION]),
                ("high_pip", np.argsort(pip)[::-1][:TOP_FORCE_GENES_PER_DIRECTION]),
            ]

            for direction, idxs in ranking_sets:
                for rank, gi in enumerate(idxs, start=1):
                    force_rows.append({
                        "fold": fold,
                        "model": "cipher_bayes_H0",
                        "fate": fate,
                        "direction": direction,
                        "rank": rank,
                        "gene": hvg_genes[gi],
                        "gene_index": int(hvg_idx[gi]),
                        "posterior_mu_u": float(u[gi]),
                        "posterior_std_u": float(std[gi]),
                        "posterior_z_u": float(z[gi]),
                        "posterior_pip": float(pip[gi]),
                        "posterior_sign_conf": float(sign_conf[gi]),
                        "posterior_p_pos": float(p_pos[gi]),
                        "posterior_p_neg": float(p_neg[gi]),
                        "ci95_lo": float(ci_lo[gi]),
                        "ci95_hi": float(ci_hi[gi]),
                        "zero_excluded_95": int(zero_exc[gi]),
                        "delta_weighted_composition": float(delta[gi]),
                        "delta_posterior_yhat": float(yhat[gi]),
                        "h0_scale": float(cipher_model["h0_scale"][j]),
                        "n_eff_pos": float(cipher_model["n_eff_pos"][j]),
                        "n_eff_neg": float(cipher_model["n_eff_neg"][j]),
                        "posterior_delta_r2": float(cipher_model["posterior_delta_r2"][j]),
                        "penalty": float(cipher_model["penalty"][j]),
                        "penalty_mean": float(cipher_model["penalty_mean"][j]),
                        "penalty_uncertainty": float(cipher_model["penalty_uncertainty"][j]),
                        "log_prior": float(cipher_model["log_prior"][j]),
                        "log_marginal": float(cipher_model["log_marginal"][j]),
                        "tau2": float(POSTERIOR_TAU2),
                    })

        # --------------------------------------------------------
        # Test all calibrated probability models.
        # --------------------------------------------------------
        for spec in MODEL_SPECS:
            model_name = spec["name"]
            partition = spec["partition"]
            use_startpop = bool(spec["startpop_prior"])
            clone_weight_mode = spec["clone_weight_mode"]
            calibrator_mode = spec["calibrator"]
            l2 = float(spec["l2"])

            if use_startpop and startpop_prior_model is None:
                print(f"[skip] {model_name}: no starting population prior available.")
                continue

            logits_train_raw = build_raw_logits_variant(
                X=Xtrain,
                n_early=n_train_early,
                cipher_model=cipher_model,
                partition=partition,
                X_baseline_for_partition=X_baseline_partition,
                start_values=start_train,
                startpop_prior_model=startpop_prior_model,
                use_startpop_prior=use_startpop,
                clone_weight_mode=clone_weight_mode,
                saturating_n0=SATURATING_N0,
            )

            logits_test_raw = build_raw_logits_variant(
                X=Xtest,
                n_early=n_test_early,
                cipher_model=cipher_model,
                partition=partition,
                X_baseline_for_partition=X_baseline_partition,
                start_values=start_test,
                startpop_prior_model=startpop_prior_model,
                use_startpop_prior=use_startpop,
                clone_weight_mode=clone_weight_mode,
                saturating_n0=SATURATING_N0,
            )

            cal = fit_logit_calibrator(
                train_logits=logits_train_raw,
                train_counts=Ctrain,
                mode=calibrator_mode,
                l2=l2,
                maxiter=CALIB_MAXITER,
                maxfun=CALIB_MAXFUN,
                ftol=CALIB_FTOL,
                gtol=CALIB_GTOL,
            )

            logits_train_cal = cal["transform"](logits_train_raw)
            logits_test_cal = cal["transform"](logits_test_raw)

            Ptrain = softmax_np(logits_train_cal)
            Ptest = softmax_np(logits_test_cal)

            train_nll = count_nll(Ctrain, Ptrain)
            train_mean_nll = mean_count_nll(Ctrain, Ptrain)
            test_nll = count_nll(Ctest, Ptest)
            test_mean_nll = mean_count_nll(Ctest, Ptest)

            calibration_rows.append({
                "fold": fold,
                "model": model_name,
                "partition": partition,
                "calibrator": calibrator_mode,
                "use_startpop_prior": use_startpop,
                "clone_weight_mode": clone_weight_mode,
                "l2": l2,
                "success": cal["success"],
                "message": cal["message"],
                "nfev": cal["nfev"],
                "nit": cal["nit"],
                "train_nll": train_nll,
                "train_mean_nll": train_mean_nll,
                "test_nll": test_nll,
                "test_mean_nll": test_mean_nll,
                "temperature": cal.get("temperature", np.nan),
                "W_fro_norm_minus_I": cal.get("W_fro_norm_minus_I", np.nan),
            })

            # Save calibration matrix/vector rows for interpretability.
            if calibrator_mode == "full_matrix":
                W = cal["W"]
                b = cal["bias"]

                for i, fate_out in enumerate(selected_fates):
                    calibration_param_rows.append({
                        "fold": fold,
                        "model": model_name,
                        "param_type": "bias",
                        "output_fate": fate_out,
                        "input_fate": "",
                        "value": float(b[i]),
                        "l2": l2,
                    })

                    for j, fate_in in enumerate(selected_fates):
                        calibration_param_rows.append({
                            "fold": fold,
                            "model": model_name,
                            "param_type": "W",
                            "output_fate": fate_out,
                            "input_fate": fate_in,
                            "value": float(W[i, j]),
                            "l2": l2,
                        })

            elif calibrator_mode == "diag_affine":
                a = cal["diag_scale"]
                b = cal["bias"]

                for i, fate in enumerate(selected_fates):
                    calibration_param_rows.append({
                        "fold": fold,
                        "model": model_name,
                        "param_type": "diag_scale",
                        "output_fate": fate,
                        "input_fate": fate,
                        "value": float(a[i]),
                        "l2": l2,
                    })

                    calibration_param_rows.append({
                        "fold": fold,
                        "model": model_name,
                        "param_type": "bias",
                        "output_fate": fate,
                        "input_fate": "",
                        "value": float(b[i]),
                        "l2": l2,
                    })

            pred_df = add_prediction_columns_from_probs(
                base_df=base_test,
                logits_raw=logits_test_raw,
                logits_cal=logits_test_cal,
                P=Ptest,
                selected_fates=selected_fates,
                model_name=model_name,
                calibrator_name=calibrator_mode,
                partition=partition,
                startpop_prior=use_startpop,
                clone_weight_mode=clone_weight_mode,
                l2=l2,
                train_mean_nll=train_mean_nll,
            )

            pred_df = add_composition_errors_extended(pred_df, selected_fates)

            all_pred_rows.append(pred_df)
            summary_rows.append(
                summarize_predictions_extended(
                    pred_df,
                    selected_fates,
                    model_name=model_name,
                    fold=fold,
                )
            )

            print(
                f"  {model_name:55s} "
                f"train mean NLL={train_mean_nll:.4f} | "
                f"test mean NLL={test_mean_nll:.4f} | "
                f"success={cal['success']}"
            )


    # ============================================================
    # SAVE OUTPUTS
    # ============================================================

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    summary_metrics = pd.concat(summary_rows, ignore_index=True)
    calibration_df = pd.DataFrame(calibration_rows)
    posterior_fit_df = pd.DataFrame(posterior_fit_rows)
    force_df = pd.DataFrame(force_rows)
    calibration_params_df = pd.DataFrame(calibration_param_rows)

    predictions.to_csv(
        os.path.join(OUTDIR, "clone_composition_predictions_calibrated_models.csv"),
        index=False,
    )

    summary_metrics.to_csv(
        os.path.join(OUTDIR, "composition_prediction_summary_calibrated_models.csv"),
        index=False,
    )

    calibration_df.to_csv(
        os.path.join(OUTDIR, "calibration_fit_summary.csv"),
        index=False,
    )

    posterior_fit_df.to_csv(
        os.path.join(OUTDIR, "posterior_fit_summary_by_fate_fold.csv"),
        index=False,
    )

    force_df.to_csv(
        os.path.join(OUTDIR, "composition_CIPHER_bayes_H0_force_genes.csv"),
        index=False,
    )

    calibration_params_df.to_csv(
        os.path.join(OUTDIR, "calibration_parameters_by_fold.csv"),
        index=False,
    )

    print("\nSaved:")
    print(os.path.join(OUTDIR, "clone_composition_predictions_calibrated_models.csv"))
    print(os.path.join(OUTDIR, "composition_prediction_summary_calibrated_models.csv"))
    print(os.path.join(OUTDIR, "calibration_fit_summary.csv"))
    print(os.path.join(OUTDIR, "composition_CIPHER_bayes_H0_force_genes.csv"))
    print(os.path.join(OUTDIR, "calibration_parameters_by_fold.csv"))


    # ============================================================
    # MODEL-LEVEL SUMMARY
    # ============================================================

    composition_summary = (
        summary_metrics[summary_metrics["metric_type"] == "composition"]
        .groupby(["model", "partition", "calibrator", "use_startpop_prior", "clone_weight_mode", "l2"], as_index=False)
        .agg(
            train_mean_nll=("train_mean_nll", "mean"),
            test_mean_count_nll=("test_mean_count_nll", "mean"),
            test_mean_count_nll_sd=("test_mean_count_nll", "std"),
            test_count_nll=("test_count_nll", "sum"),
            mean_JS=("mean_JS", "mean"),
            mean_JS_smooth=("mean_JS_smooth", "mean"),
            mean_Brier=("mean_Brier", "mean"),
            mean_Brier_smooth=("mean_Brier_smooth", "mean"),
            mean_L1=("mean_L1", "mean"),
            mean_L1_smooth=("mean_L1_smooth", "mean"),
            mean_cosine=("mean_cosine", "mean"),
            mean_cosine_smooth=("mean_cosine_smooth", "mean"),
            top1_accuracy=("top1_accuracy", "mean"),
            top1_accuracy_smooth=("top1_accuracy_smooth", "mean"),
            entropy_pearson=("entropy_pearson", "mean"),
            entropy_spearman=("entropy_spearman", "mean"),
            n_clones=("n_clones", "sum"),
            n_terminal_cells=("n_terminal_cells", "sum"),
        )
        .sort_values("test_mean_count_nll")
    )

    composition_summary.to_csv(
        os.path.join(OUTDIR, "composition_summary_by_model.csv"),
        index=False,
    )

    per_fate_summary = (
        summary_metrics[summary_metrics["metric_type"] == "per_fate_fraction"]
        .groupby(["model", "fate", "partition", "calibrator", "use_startpop_prior", "clone_weight_mode", "l2"], as_index=False)
        .agg(
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
            r2=("r2", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            pearson_smooth=("pearson_smooth", "mean"),
            spearman_smooth=("spearman_smooth", "mean"),
            r2_smooth=("r2_smooth", "mean"),
            mae_smooth=("mae_smooth", "mean"),
            rmse_smooth=("rmse_smooth", "mean"),
            mean_obs_fraction=("mean_obs_fraction", "mean"),
            mean_obs_fraction_smooth=("mean_obs_fraction_smooth", "mean"),
            mean_pred_fraction=("mean_pred_fraction", "mean"),
        )
        .sort_values(["model", "fate"])
    )

    per_fate_summary.to_csv(
        os.path.join(OUTDIR, "per_fate_summary_by_model.csv"),
        index=False,
    )

    best_row = composition_summary.iloc[0].copy()
    best_model = best_row["model"]

    print("\n============================================================")
    print("MODEL SUMMARY, SORTED BY TEST COUNT NLL")
    print("============================================================")
    print(composition_summary.to_string(index=False))

    print("\nBest model by test mean count NLL:", best_model)

    print("\nBest model details:")
    print(best_row.to_string())

    print("\nPer-fate summary for best model:")
    print(
        per_fate_summary[per_fate_summary["model"] == best_model]
        .to_string(index=False)
    )


    # ============================================================
    # PLOTS: L2 SWEEP FOR FULL MATRIX
    # ============================================================

    full_sweep = composition_summary[
        composition_summary["calibrator"].eq("full_matrix") &
        composition_summary["clone_weight_mode"].eq("none")
    ].copy()

    if len(full_sweep) > 0:
        plt.figure(figsize=(8, 5))

        sns.lineplot(
            data=full_sweep,
            x="l2",
            y="test_mean_count_nll",
            hue="partition",
            style="use_startpop_prior",
            marker="o",
        )

        plt.xscale("log")
        plt.xlabel("full-matrix calibration L2")
        plt.ylabel("CV test count NLL per terminal cell")
        plt.title("Full-matrix calibration L2 sweep")
        plt.tight_layout()

        plt.savefig(os.path.join(OUTDIR, "full_matrix_L2_sweep_test_count_nll.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, "full_matrix_L2_sweep_test_count_nll.svg"), bbox_inches="tight")
        plt.show()


    # ============================================================
    # PLOTS: MODEL COMPARISON
    # ============================================================

    plot_top = composition_summary.head(min(20, len(composition_summary))).copy()
    plot_order = plot_top["model"].tolist()

    plt.figure(figsize=(14, 5))

    sns.barplot(
        data=plot_top,
        x="model",
        y="test_mean_count_nll",
        order=plot_order,
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("test count NLL per terminal cell")
    plt.xlabel("")
    plt.title("Top calibrated models by terminal-count likelihood")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, "model_comparison_test_count_nll_top20.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "model_comparison_test_count_nll_top20.svg"), bbox_inches="tight")
    plt.show()


    plt.figure(figsize=(14, 5))

    sns.barplot(
        data=plot_top,
        x="model",
        y="mean_JS_smooth",
        order=plot_order,
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mean JS divergence vs smoothed fractions")
    plt.xlabel("")
    plt.title("Top calibrated models: smoothed composition JS")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, "model_comparison_JS_smooth_top20.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, "model_comparison_JS_smooth_top20.svg"), bbox_inches="tight")
    plt.show()


    # ============================================================
    # PLOTS: BEST MODEL PREDICTED VS OBSERVED
    # ============================================================

    best_pred = predictions[predictions["model"] == best_model].copy()

    n_fates = len(selected_fates)
    ncols = min(3, n_fates)
    nrows = int(np.ceil(n_fates / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for j, fate in enumerate(selected_fates):
        ax = axes[j // ncols][j % ncols]
        s = safe_name(fate)

        x = best_pred[f"obs_frac__{s}"].values
        xs = best_pred[f"obs_frac_smooth__{s}"].values
        y = best_pred[f"pred_frac__{s}"].values

        ax.scatter(x, y, s=28, alpha=0.35, label="raw observed")
        ax.scatter(xs, y, s=28, alpha=0.75, label="smoothed observed")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)

        r = safe_corr(x, y, "pearson")
        rho = safe_corr(x, y, "spearman")
        r2 = safe_r2(x, y)

        rs = safe_corr(xs, y, "pearson")
        rhos = safe_corr(xs, y, "spearman")
        r2s = safe_r2(xs, y)

        ax.set_title(
            f"{fate}\n"
            f"raw: r={r:.2f}, rho={rho:.2f}, R²={r2:.2f}\n"
            f"smooth: r={rs:.2f}, rho={rhos:.2f}, R²={r2s:.2f}"
        )

        ax.set_xlabel("observed terminal fraction")
        ax.set_ylabel("predicted fate probability")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)

        if j == 0:
            ax.legend(frameon=False, fontsize=9)

    for k in range(n_fates, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, f"best_model_{safe_name(best_model)}_predicted_vs_observed.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"best_model_{safe_name(best_model)}_predicted_vs_observed.svg"), bbox_inches="tight")
    plt.show()


    # ============================================================
    # PLOTS: BEST MODEL COMPOSITION HEATMAPS
    # ============================================================

    obs_cols = [f"obs_frac__{safe_name(f)}" for f in selected_fates]
    obs_s_cols = [f"obs_frac_smooth__{safe_name(f)}" for f in selected_fates]
    pred_cols = [f"pred_frac__{safe_name(f)}" for f in selected_fates]

    obs_heat = best_pred.groupby("true_dominant_fate")[obs_cols].mean().reindex(selected_fates)
    obs_s_heat = best_pred.groupby("true_dominant_fate")[obs_s_cols].mean().reindex(selected_fates)
    pred_heat = best_pred.groupby("true_dominant_fate")[pred_cols].mean().reindex(selected_fates)

    obs_heat.columns = selected_fates
    obs_s_heat.columns = selected_fates
    pred_heat.columns = selected_fates

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    sns.heatmap(
        obs_heat,
        ax=axes[0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "raw observed fraction"},
    )

    axes[0].set_title("Raw observed composition")
    axes[0].set_xlabel("terminal fate")
    axes[0].set_ylabel("dominant terminal fate")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        obs_s_heat,
        ax=axes[1],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "smoothed observed fraction"},
    )

    axes[1].set_title("Smoothed observed composition")
    axes[1].set_xlabel("terminal fate")
    axes[1].set_ylabel("dominant terminal fate")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    sns.heatmap(
        pred_heat,
        ax=axes[2],
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "predicted probability"},
    )

    axes[2].set_title(f"Predicted composition\n{best_model}")
    axes[2].set_xlabel("terminal fate")
    axes[2].set_ylabel("dominant terminal fate")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].tick_params(axis="y", rotation=0)

    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, f"best_model_{safe_name(best_model)}_composition_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"best_model_{safe_name(best_model)}_composition_heatmaps.svg"), bbox_inches="tight")
    plt.show()


    # ============================================================
    # PLOTS: BEST MODEL CONFUSION MATRIX
    # ============================================================

    cm = confusion_matrix(
        best_pred["true_dominant_fate"],
        best_pred["pred_dominant_fate"],
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

    plt.title(f"Dominant fate prediction\n{best_model}")
    plt.xlabel("predicted dominant fate")
    plt.ylabel("observed dominant fate")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, f"best_model_{safe_name(best_model)}_dominant_fate_confusion.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"best_model_{safe_name(best_model)}_dominant_fate_confusion.svg"), bbox_inches="tight")
    plt.show()


    # ============================================================
    # PLOTS: BEST FULL-MATRIX CALIBRATION MATRIX
    # ============================================================

    best_params = calibration_params_df[
        (calibration_params_df["model"] == best_model) &
        (calibration_params_df["param_type"] == "W")
    ].copy()

    if len(best_params) > 0:
        W_mean = (
            best_params
            .groupby(["output_fate", "input_fate"], as_index=False)["value"]
            .mean()
            .pivot(index="output_fate", columns="input_fate", values="value")
            .reindex(index=selected_fates, columns=selected_fates)
        )

        plt.figure(figsize=(8, 7))

        sns.heatmap(
            W_mean,
            cmap="vlag",
            center=0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "mean calibration W"},
        )

        plt.title(f"Mean full-matrix calibration W\n{best_model}")
        plt.xlabel("input CIPHER logit")
        plt.ylabel("output calibrated logit")
        plt.tight_layout()

        plt.savefig(os.path.join(OUTDIR, f"best_model_{safe_name(best_model)}_calibration_W_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"best_model_{safe_name(best_model)}_calibration_W_heatmap.svg"), bbox_inches="tight")
        plt.show()


    # ============================================================
    # FINAL PRINTS
    # ============================================================

    print("\n============================================================")
    print("BEST MODEL")
    print("============================================================")
    print(best_model)

    print("\n============================================================")
    print("BEST MODEL ROW")
    print("============================================================")
    print(best_row.to_string())

    print("\n============================================================")
    print("TOP 15 MODELS")
    print("============================================================")
    print(composition_summary.head(15).to_string(index=False))

    print("\n============================================================")
    print("PER-FATE SUMMARY FOR BEST MODEL")
    print("============================================================")
    print(
        per_fate_summary[per_fate_summary["model"] == best_model]
        .to_string(index=False)
    )

    print("\nDone. Outputs in:", OUTDIR)



def bayesian_marker_validation():
    global os, re, math, warnings, np, pd, plt, sns
    global hypergeom, multipletests, CIPHER_OUTDIR, FORCE_PATH, HVG_PATH, OUTDIR, MAIN_TOP_N, TOP_N_VALUES
    global USE_DIRECTION, PREFERRED_EFFECT_COL, FILTER_TO_RELEVANT_TERMS, TERM_RELEVANCE_KEYWORDS, FETCH_GSEAPY_LIBRARIES_IF_NEEDED, GSEAPY_ORGANISM, GSEAPY_LIBRARY_KEYWORDS, N_TOP_TERMS_TO_PLOT
    global check_file, safe_name, bh_fdr, term_is_relevant, hypergeom_enrich, flatten_marker_libraries, expected_category_for_fate, terms_for_category
    global first_existing_column, CANONICAL_MARKERS, FATE_CATEGORY_SYNONYMS, force_df, required_base, missing_base, EFFECT_COL, STD_COL
    global Z_COL, PIP_COL, SIGN_CONF_COL, P_POS_COL, P_NEG_COL, CI_LO_COL, CI_HI_COL, DELTA_COL
    global YHAT_COL, hvg_df, background_genes, background_norm, force_use, available_dirs, agg_dict, agg
    global sort_cols, ascending, selected_fates, top_genes_by_fate, top_gene_rows, fate, sub, top_n
    global top_sub, genes, rank, _, row, out_row, extra_col, top_genes_df
    global top_union, u_heat, pip_heat, z_heat, loaded_marker_libraries, varname, gp, get_library_name
    global get_library, all_libs, chosen_libs, lib, e, marker_terms_df, canonical_rows, term
    global canonical_df, all_terms_df, canonical_results, query, marker_category, marker_genes, res, cat_order
    global fate_order, heat_fdr, heat_overlap, heat_log2odds, db_results, term_row, idx, sub_idx
    global top_db, display_cols, category_results, category, term_sub, union_genes, source_terms, n_terms
    global union, cat_order2, cat_heat, cat_odds, cat_overlap, expected_rows, exp_cat, canon_sub
    global cat_sub, c, g, expected_df, fig, axes, canon_gene_to_categories, cat
    global gene_to_db_hits, tr, label, annot_rows, gnorm, canonical_cats, db_hits, annot_row
    global annot_df, summary_rows, best_canonical, best_db, expected_canonical, top_genes, b, d
    global summary_df, fn
    # ============================================================
    # CIPHER-LARRY Bayesian posterior marker validation
    # using TOP 50 posterior u genes
    # ============================================================
    #
    # Compatible with output from:
    #
    #   cipher_larry_clone_fate_composition_analytic_bayes_H0_complete
    #
    # Specifically expects:
    #
    #   composition_CIPHER_bayes_H0_force_genes.csv
    #
    # and uses:
    #
    #   posterior_mu_u      instead of old legacy u
    #   posterior_pip       posterior activity probability
    #   posterior_z_u       posterior z score
    #   posterior_std_u     posterior uncertainty
    #   ci95_lo / ci95_hi   credible intervals
    #
    # What this does:
    #   1. Loads Bayesian CIPHER force genes.
    #   2. Aggregates posterior u genes across CV folds.
    #   3. Takes top 50 positive posterior u genes per CIPHER fate.
    #   4. Tests canonical marker overlap/enrichment.
    #   5. Tests database marker-term enrichment if marker libraries are loaded.
    #   6. Makes composite plots:
    #        - top 50 posterior u gene heatmap
    #        - posterior PIP heatmap
    #        - posterior z heatmap
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
    # If that fails, it still runs with canonical manual markers.
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

    CIPHER_OUTDIR = os.path.join(OUT_BASE, "cipher_larry_clone_fate_composition_analytic_bayes_H0_complete")

    FORCE_PATH = os.path.join(CIPHER_OUTDIR, "composition_CIPHER_bayes_H0_force_genes.csv")
    HVG_PATH   = os.path.join(CIPHER_OUTDIR, "selected_early_hvgs.csv")

    OUTDIR = os.path.join(CIPHER_OUTDIR, "bayes_marker_association_top50_composite")
    os.makedirs(OUTDIR, exist_ok=True)

    MAIN_TOP_N = 25
    TOP_N_VALUES = [25]

    # For posterior Bayesian force table, valid directions are usually:
    #   positive, negative, high_abs_z, high_pip
    USE_DIRECTION = "positive"

    # Ranking column from Bayesian CIPHER force output.
    # Auto-fallbacks are handled below.
    PREFERRED_EFFECT_COL = "posterior_mu_u"

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
    GSEAPY_ORGANISM = "Human"

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

    N_TOP_TERMS_TO_PLOT = 12

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

    def check_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File not found: {path}\n"
                f"Current working directory: {os.getcwd()}"
            )
        print(f"[OK] Found: {path}")


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


    def first_existing_column(df, candidates, required=True, label="column"):
        for c in candidates:
            if c in df.columns:
                return c

        if required:
            raise ValueError(
                f"Could not find required {label}. Tried: {candidates}\n"
                f"Available columns: {list(df.columns)}"
            )

        return None


    # ============================================================
    # CANONICAL MARKER SETS
    # ============================================================

    CANONICAL_MARKERS = {
        "Neutrophil": [
            "Elane", "Prtn3", "Mpo", "Ctsg", "Ngp", "Lcn2", "S100a8", "S100a9",
            "Camp", "Ltf", "Mmp8", "Retnlg", "Cebpe", "Ly6g", "Cxcr2", "Fcgr3",
            "Mmp9", "Sell", "Cd177", "Csf3r",
        ],
        "Monocyte": [
            "Lyz2", "Csf1r", "Ctss", "Ctsb", "Lgals3", "Mpeg1", "Ccr2", "Ly6c2",
            "Itgam", "Cd14", "Fcgr1", "Sirpa", "Cybb", "Ms4a7", "Plac8", "S100a4",
            "Cebpb", "Irf8", "Lpl", "Aif1", "Tyrobp", "Ccl6", "Lst1",
        ],
        "Baso": [
            "Gata2", "Cpa3", "Prg2", "Hdc", "Ms4a2", "Fcer1a", "Alox5", "Srgn",
            "Ccr3", "Il4", "Il13", "Mcpt8", "Csf2rb", "Csf2rb2", "Kit",
            "Il3ra", "Fcer1g", "Cd200r3", "Slc6a4",
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
    # LOAD BAYESIAN CIPHER FORCE GENES
    # ============================================================

    check_file(FORCE_PATH)

    force_df = pd.read_csv(FORCE_PATH)

    print("Loaded force_df:", force_df.shape)
    print("Columns:", list(force_df.columns))

    required_base = {"fold", "fate", "direction", "rank", "gene"}
    missing_base = required_base - set(force_df.columns)

    if missing_base:
        raise ValueError(f"Missing required base columns from force file: {missing_base}")

    EFFECT_COL = first_existing_column(
        force_df,
        [
            PREFERRED_EFFECT_COL,
            "posterior_mu_u",
            "mean_u",
            "u",
        ],
        required=True,
        label="effect/u column",
    )

    STD_COL = first_existing_column(
        force_df,
        ["posterior_std_u", "std_u", "std"],
        required=False,
        label="posterior std column",
    )

    Z_COL = first_existing_column(
        force_df,
        ["posterior_z_u", "z_u", "z"],
        required=False,
        label="posterior z column",
    )

    PIP_COL = first_existing_column(
        force_df,
        ["posterior_pip", "pip", "activity_probability"],
        required=False,
        label="posterior PIP column",
    )

    SIGN_CONF_COL = first_existing_column(
        force_df,
        ["posterior_sign_conf", "sign_conf"],
        required=False,
        label="posterior sign confidence column",
    )

    P_POS_COL = first_existing_column(
        force_df,
        ["posterior_p_pos", "p_pos"],
        required=False,
        label="posterior p positive column",
    )

    P_NEG_COL = first_existing_column(
        force_df,
        ["posterior_p_neg", "p_neg"],
        required=False,
        label="posterior p negative column",
    )

    CI_LO_COL = first_existing_column(
        force_df,
        ["ci95_lo", "posterior_ci95_lo"],
        required=False,
        label="credible interval lower column",
    )

    CI_HI_COL = first_existing_column(
        force_df,
        ["ci95_hi", "posterior_ci95_hi"],
        required=False,
        label="credible interval upper column",
    )

    DELTA_COL = first_existing_column(
        force_df,
        ["delta_weighted_composition", "mean_delta", "delta"],
        required=False,
        label="delta column",
    )

    YHAT_COL = first_existing_column(
        force_df,
        ["delta_posterior_yhat", "yhat", "delta_yhat"],
        required=False,
        label="posterior yhat column",
    )

    print("\nUsing columns:")
    print("  effect:", EFFECT_COL)
    print("  std:", STD_COL)
    print("  z:", Z_COL)
    print("  pip:", PIP_COL)
    print("  delta:", DELTA_COL)
    print("  yhat:", YHAT_COL)

    if os.path.exists(HVG_PATH):
        hvg_df = pd.read_csv(HVG_PATH)
        background_genes = hvg_df["gene"].astype(str).tolist()
        print(f"\nBackground: {len(background_genes)} HVGs from {HVG_PATH}")
    else:
        background_genes = force_df["gene"].astype(str).unique().tolist()
        print(f"\nBackground: {len(background_genes)} genes from force table only")

    background_norm = sorted(set(map(norm_gene, background_genes)))

    force_use = force_df[
        force_df["direction"].astype(str).str.lower() == USE_DIRECTION.lower()
    ].copy()

    if force_use.empty:
        available_dirs = sorted(force_df["direction"].astype(str).unique().tolist())
        raise RuntimeError(
            f"No rows found with direction={USE_DIRECTION}. "
            f"Available directions: {available_dirs}"
        )

    force_use[EFFECT_COL] = pd.to_numeric(force_use[EFFECT_COL], errors="coerce")
    force_use = force_use[np.isfinite(force_use[EFFECT_COL])].copy()

    if force_use.empty:
        raise RuntimeError(f"No finite values found in effect column {EFFECT_COL}.")


    # ============================================================
    # AGGREGATE TOP POSTERIOR u GENES ACROSS FOLDS
    # ============================================================

    agg_dict = {
        "mean_effect": (EFFECT_COL, "mean"),
        "median_effect": (EFFECT_COL, "median"),
        "max_effect": (EFFECT_COL, "max"),
        "min_effect": (EFFECT_COL, "min"),
        "mean_rank": ("rank", "mean"),
        "min_rank": ("rank", "min"),
        "n_folds_present": ("fold", "nunique"),
    }

    if STD_COL is not None:
        agg_dict["mean_posterior_std"] = (STD_COL, "mean")

    if Z_COL is not None:
        agg_dict["mean_posterior_z"] = (Z_COL, "mean")
        agg_dict["mean_abs_posterior_z"] = (Z_COL, lambda x: np.nanmean(np.abs(x)))

    if PIP_COL is not None:
        agg_dict["mean_posterior_pip"] = (PIP_COL, "mean")
        agg_dict["max_posterior_pip"] = (PIP_COL, "max")

    if SIGN_CONF_COL is not None:
        agg_dict["mean_sign_conf"] = (SIGN_CONF_COL, "mean")

    if P_POS_COL is not None:
        agg_dict["mean_p_pos"] = (P_POS_COL, "mean")

    if P_NEG_COL is not None:
        agg_dict["mean_p_neg"] = (P_NEG_COL, "mean")

    if CI_LO_COL is not None:
        agg_dict["mean_ci95_lo"] = (CI_LO_COL, "mean")

    if CI_HI_COL is not None:
        agg_dict["mean_ci95_hi"] = (CI_HI_COL, "mean")

    if DELTA_COL is not None:
        agg_dict["mean_delta"] = (DELTA_COL, "mean")

    if YHAT_COL is not None:
        agg_dict["mean_delta_yhat"] = (YHAT_COL, "mean")

    if "zero_excluded_95" in force_use.columns:
        agg_dict["frac_zero_excluded_95"] = ("zero_excluded_95", "mean")

    if "h0_scale" in force_use.columns:
        agg_dict["mean_h0_scale"] = ("h0_scale", "mean")

    if "posterior_delta_r2" in force_use.columns:
        agg_dict["mean_posterior_delta_r2"] = ("posterior_delta_r2", "mean")

    if "temperature" in force_use.columns:
        agg_dict["mean_temperature"] = ("temperature", "mean")

    if "log_marginal" in force_use.columns:
        agg_dict["mean_log_marginal"] = ("log_marginal", "mean")

    agg = (
        force_use
        .groupby(["fate", "gene"], as_index=False)
        .agg(**agg_dict)
    )

    agg["gene_norm"] = agg["gene"].map(norm_gene)

    # Ranking:
    #   1. genes present in more folds
    #   2. high posterior mean u
    #   3. high posterior PIP if available
    #   4. lower mean rank
    sort_cols = ["fate", "n_folds_present", "mean_effect"]
    ascending = [True, False, False]

    if "mean_posterior_pip" in agg.columns:
        sort_cols.append("mean_posterior_pip")
        ascending.append(False)

    sort_cols.append("mean_rank")
    ascending.append(True)

    agg = agg.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    agg.to_csv(
        os.path.join(OUTDIR, "aggregated_bayesian_positive_posterior_u_genes_all.csv"),
        index=False,
    )

    selected_fates = agg["fate"].drop_duplicates().tolist()

    print("\nFates found:")
    print(selected_fates)

    top_genes_by_fate = {}
    top_gene_rows = []

    for fate in selected_fates:
        sub = agg[agg["fate"] == fate].copy()
        top_genes_by_fate[fate] = {}

        for top_n in TOP_N_VALUES:
            top_sub = sub.head(top_n).copy()
            genes = top_sub["gene"].astype(str).tolist()

            top_genes_by_fate[fate][top_n] = genes

            for rank, (_, row) in enumerate(top_sub.iterrows(), start=1):
                out_row = {
                    "fate": fate,
                    "top_n": top_n,
                    "rank": rank,
                    "gene": row["gene"],
                    "mean_posterior_mu_u": row["mean_effect"],
                    "median_posterior_mu_u": row["median_effect"],
                    "max_posterior_mu_u": row["max_effect"],
                    "min_posterior_mu_u": row["min_effect"],
                    "mean_rank_across_folds": row["mean_rank"],
                    "min_rank": row["min_rank"],
                    "n_folds_present": row["n_folds_present"],
                }

                for extra_col in [
                    "mean_posterior_std",
                    "mean_posterior_z",
                    "mean_abs_posterior_z",
                    "mean_posterior_pip",
                    "max_posterior_pip",
                    "mean_sign_conf",
                    "mean_p_pos",
                    "mean_p_neg",
                    "mean_ci95_lo",
                    "mean_ci95_hi",
                    "frac_zero_excluded_95",
                    "mean_delta",
                    "mean_delta_yhat",
                    "mean_h0_scale",
                    "mean_posterior_delta_r2",
                    "mean_temperature",
                    "mean_log_marginal",
                ]:
                    if extra_col in row.index:
                        out_row[extra_col] = row[extra_col]

                top_gene_rows.append(out_row)

    top_genes_df = pd.DataFrame(top_gene_rows)

    top_genes_df.to_csv(
        os.path.join(OUTDIR, f"top{MAIN_TOP_N}_bayesian_posterior_u_genes_by_fate.csv"),
        index=False,
    )

    print(f"\nTop {MAIN_TOP_N} positive Bayesian posterior u genes per fate:")

    for fate in selected_fates:
        print(f"\n{fate}")
        print(", ".join(top_genes_by_fate[fate][MAIN_TOP_N]))


    # ============================================================
    # TOP 50 POSTERIOR U HEATMAPS
    # ============================================================

    top_union = []

    for fate in selected_fates:
        top_union.extend(top_genes_by_fate[fate][MAIN_TOP_N])

    top_union = list(dict.fromkeys(top_union))

    u_heat = (
        agg
        .pivot_table(index="gene", columns="fate", values="mean_effect", fill_value=0.0)
        .reindex(top_union)
        .reindex(columns=selected_fates)
    )

    plt.figure(figsize=(1.2 * len(selected_fates) + 5, 0.20 * len(top_union) + 5))

    sns.heatmap(
        u_heat,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "mean posterior force E[u]"},
    )

    plt.title(f"Top {MAIN_TOP_N} Bayesian posterior CIPHER u genes per fate")
    plt.xlabel("CIPHER fate force")
    plt.ylabel("gene")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTDIR, f"top{MAIN_TOP_N}_bayesian_posterior_u_gene_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"top{MAIN_TOP_N}_bayesian_posterior_u_gene_heatmap.svg"), bbox_inches="tight")
    plt.show()


    if "mean_posterior_pip" in agg.columns:
        pip_heat = (
            agg
            .pivot_table(index="gene", columns="fate", values="mean_posterior_pip", fill_value=0.0)
            .reindex(top_union)
            .reindex(columns=selected_fates)
        )

        plt.figure(figsize=(1.2 * len(selected_fates) + 5, 0.20 * len(top_union) + 5))

        sns.heatmap(
            pip_heat,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "mean posterior activity probability"},
        )

        plt.title(f"Top {MAIN_TOP_N} Bayesian posterior CIPHER gene PIP")
        plt.xlabel("CIPHER fate force")
        plt.ylabel("gene")
        plt.tight_layout()

        plt.savefig(os.path.join(OUTDIR, f"top{MAIN_TOP_N}_bayesian_posterior_pip_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"top{MAIN_TOP_N}_bayesian_posterior_pip_heatmap.svg"), bbox_inches="tight")
        plt.show()


    if "mean_posterior_z" in agg.columns:
        z_heat = (
            agg
            .pivot_table(index="gene", columns="fate", values="mean_posterior_z", fill_value=0.0)
            .reindex(top_union)
            .reindex(columns=selected_fates)
        )

        plt.figure(figsize=(1.2 * len(selected_fates) + 5, 0.20 * len(top_union) + 5))

        sns.heatmap(
            z_heat,
            cmap="vlag",
            center=0,
            cbar_kws={"label": "mean posterior z"},
        )

        plt.title(f"Top {MAIN_TOP_N} Bayesian posterior CIPHER gene z-scores")
        plt.xlabel("CIPHER fate force")
        plt.ylabel("gene")
        plt.tight_layout()

        plt.savefig(os.path.join(OUTDIR, f"top{MAIN_TOP_N}_bayesian_posterior_z_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTDIR, f"top{MAIN_TOP_N}_bayesian_posterior_z_heatmap.svg"), bbox_inches="tight")
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
            print("\nCould not fetch GSEApy libraries. Continuing with canonical markers only.")
            print("Reason:", repr(e))
            loaded_marker_libraries = {}

    if loaded_marker_libraries is None:
        loaded_marker_libraries = {}

    marker_terms_df = flatten_marker_libraries(loaded_marker_libraries)

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

    all_terms_df["n_genes_in_background"] = all_terms_df["genes_norm"].apply(
        lambda gs: len(set(gs) & set(background_norm))
    )

    all_terms_df = all_terms_df[all_terms_df["n_genes_in_background"] > 0].copy()

    if len(all_terms_df) == 0:
        raise RuntimeError(
            "No marker terms overlap the background genes. "
            "Check gene symbol casing/species or background file."
        )

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

    plt.title(f"Bayesian CIPHER top {MAIN_TOP_N} posterior u genes vs canonical markers")
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

    plt.title(f"Bayesian CIPHER top {MAIN_TOP_N} posterior u genes: canonical marker overlap")
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

    plt.title(f"Bayesian CIPHER top {MAIN_TOP_N} posterior u genes: canonical marker odds")
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

    db_results["fdr_within_topn"] = np.nan

    for top_n in sorted(db_results["top_n"].unique()):
        idx = db_results["top_n"] == top_n
        db_results.loc[idx, "fdr_within_topn"] = bh_fdr(db_results.loc[idx, "p_value"].values)

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

    print(f"\nTop database marker terms for top {MAIN_TOP_N} Bayesian posterior CIPHER genes:")

    for fate in selected_fates:
        print(f"\n=== {fate} ===")

        display_cols = [
            "library",
            "term_clean",
            "k_overlap",
            "odds_enrichment",
            "p_value",
            "fdr_within_topn",
            "fdr_within_fate_topn",
            "overlap_genes",
        ]

        print(
            top_db[top_db["cipher_fate"] == fate][display_cols]
            .head(10)
            .to_string(index=False)
        )


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
        plt.title(f"{fate}: enriched marker terms among top {MAIN_TOP_N} Bayesian CIPHER genes")
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

    plt.title(f"Bayesian CIPHER top {MAIN_TOP_N} posterior u genes vs marker-term categories")
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

    plt.title(f"Bayesian CIPHER top {MAIN_TOP_N} posterior u genes: marker category odds")
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

    plt.title(f"Bayesian CIPHER top {MAIN_TOP_N} posterior u genes: marker category overlap")
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
    # 5) ANNOTATE EACH TOP 50 POSTERIOR U GENE
    # ============================================================

    canon_gene_to_categories = {}

    for cat, genes in CANONICAL_MARKERS.items():
        for g in genes:
            canon_gene_to_categories.setdefault(norm_gene(g), []).append(cat)

    gene_to_db_hits = {}

    for _, tr in all_terms_df.iterrows():
        label = f"{tr['library']}::{tr['term_clean']}"

        for g in tr["genes_norm"]:
            gene_to_db_hits.setdefault(g, []).append(label)

    annot_rows = []

    for _, row in top_genes_df[top_genes_df["top_n"] == MAIN_TOP_N].iterrows():
        gnorm = norm_gene(row["gene"])

        canonical_cats = canon_gene_to_categories.get(gnorm, [])
        db_hits = gene_to_db_hits.get(gnorm, [])

        annot_row = {
            "cipher_fate": row["fate"],
            "rank": row["rank"],
            "gene": row["gene"],
            "mean_posterior_mu_u": row["mean_posterior_mu_u"],
            "median_posterior_mu_u": row["median_posterior_mu_u"],
            "max_posterior_mu_u": row["max_posterior_mu_u"],
            "mean_rank_across_folds": row["mean_rank_across_folds"],
            "n_folds_present": row["n_folds_present"],
            "canonical_marker_categories": "; ".join(canonical_cats),
            "is_expected_canonical_marker": expected_category_for_fate(row["fate"]) in canonical_cats,
            "n_database_marker_terms": len(db_hits),
            "example_database_terms": "; ".join(db_hits[:10]),
        }

        for extra_col in [
            "mean_posterior_std",
            "mean_posterior_z",
            "mean_abs_posterior_z",
            "mean_posterior_pip",
            "max_posterior_pip",
            "mean_sign_conf",
            "mean_p_pos",
            "mean_p_neg",
            "mean_ci95_lo",
            "mean_ci95_hi",
            "frac_zero_excluded_95",
            "mean_delta",
            "mean_delta_yhat",
            "mean_h0_scale",
            "mean_posterior_delta_r2",
            "mean_temperature",
            "mean_log_marginal",
        ]:
            if extra_col in row.index:
                annot_row[extra_col] = row[extra_col]

        annot_rows.append(annot_row)

    annot_df = pd.DataFrame(annot_rows)

    annot_df.to_csv(
        os.path.join(OUTDIR, f"top{MAIN_TOP_N}_bayesian_CIPHER_genes_marker_annotations.csv"),
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

        expected_canonical = (
            canonical_results[
                (canonical_results["cipher_fate"] == fate) &
                (canonical_results["marker_category"] == exp_cat)
            ]
            if exp_cat is not None else pd.DataFrame()
        )

        top_genes = top_genes_by_fate[fate][MAIN_TOP_N]

        row = {
            "cipher_fate": fate,
            "expected_marker_category": exp_cat,
            f"top{MAIN_TOP_N}_genes": ", ".join(top_genes),
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
        os.path.join(OUTDIR, f"top{MAIN_TOP_N}_bayesian_marker_validation_composite_summary.csv"),
        index=False,
    )


    # ============================================================
    # FINAL PRINTS
    # ============================================================

    print("\n============================================================")
    print(f"DONE: TOP {MAIN_TOP_N} Bayesian CIPHER marker validation")
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
        f"top{MAIN_TOP_N}_bayesian_posterior_u_gene_heatmap.png",
        f"top{MAIN_TOP_N}_bayesian_posterior_pip_heatmap.png",
        f"top{MAIN_TOP_N}_bayesian_posterior_z_heatmap.png",
        f"canonical_marker_enrichment_top{MAIN_TOP_N}_minuslog10FDR.png",
        f"canonical_marker_overlap_top{MAIN_TOP_N}.png",
        f"canonical_marker_log2odds_top{MAIN_TOP_N}.png",
        f"category_marker_association_top{MAIN_TOP_N}_minuslog10FDR.png",
        f"category_marker_association_top{MAIN_TOP_N}_log2odds.png",
        f"category_marker_association_top{MAIN_TOP_N}_overlap_fraction.png",
        f"expected_marker_category_summary_top{MAIN_TOP_N}.png",
        f"top{MAIN_TOP_N}_bayesian_CIPHER_genes_marker_annotations.csv",
        f"top{MAIN_TOP_N}_bayesian_marker_validation_composite_summary.csv",
        f"top_marker_database_terms_top{MAIN_TOP_N}.csv",
    ]:
        print(" -", os.path.join(OUTDIR, fn))



def bayesian_marker_three_plots():
    global os, warnings, np, pd, plt, sns, hypergeom, multipletests
    global CIPHER_OUTDIR, FORCE_PATH, HVG_PATH, OUTDIR, TOP_N, USE_DIRECTION, PREFERRED_FATE_ORDER, safe_name
    global bh_fdr, hypergeom_enrich, expected_category_for_fate, first_existing_column, CANONICAL_MARKERS, force_df, required_cols, missing
    global EFFECT_COL, PIP_COL, hvg_df, background_genes, force_use, agg, fates_found, selected_fates
    global top_genes_by_fate, fate, sub, top_union, pip_heat, pip_png, pip_svg, canonical_results
    global query, marker_category, marker_genes, res, expected_rows, expected_cat, row, expected_df
    global fig, axes, tick, bar_png, bar_svg
    # ============================================================
    # CIPHER-LARRY Bayesian posterior: ONLY 3 plots
    #
    # Makes:
    #   1. Top N Bayesian posterior CIPHER gene PIP heatmap
    #   2. Expected canonical marker significance barplot
    #   3. Expected canonical marker enrichment barplot
    #
    # Requires existing files:
    #   CIPHER_OUTDIR/composition_CIPHER_bayes_H0_force_genes.csv
    #   CIPHER_OUTDIR/selected_early_hvgs.csv  [optional but preferred]
    # ============================================================

    import os
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

    CIPHER_OUTDIR = os.path.join(OUT_BASE, "cipher_larry_clone_fate_composition_analytic_bayes_H0_complete")

    FORCE_PATH = os.path.join(CIPHER_OUTDIR, "composition_CIPHER_bayes_H0_force_genes.csv")
    HVG_PATH   = os.path.join(CIPHER_OUTDIR, "selected_early_hvgs.csv")

    OUTDIR = os.path.join(CIPHER_OUTDIR, "bayes_marker_three_plots_only")
    os.makedirs(OUTDIR, exist_ok=True)

    TOP_N = 20
    USE_DIRECTION = "positive"

    # If you want a specific order:
    PREFERRED_FATE_ORDER = ["Baso", "Eos", "Erythroid", "Mast", "Meg", "Monocyte", "Neutrophil"]

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
        if "eos" in f:
            return "Eos"
        if "lymph" in f:
            return "Lymphoid"
        if "dc" in f or "dend" in f:
            return "Dendritic"
        return None

    def first_existing_column(df, candidates, required=True):
        for c in candidates:
            if c in df.columns:
                return c
        if required:
            raise ValueError(
                f"Could not find any of columns {candidates}.\n"
                f"Available columns: {list(df.columns)}"
            )
        return None

    # ============================================================
    # CANONICAL MARKERS
    # ============================================================

    CANONICAL_MARKERS = {
        "Neutrophil": [
            "Elane", "Prtn3", "Mpo", "Ctsg", "Ngp", "Lcn2", "S100a8", "S100a9",
            "Camp", "Ltf", "Mmp8", "Retnlg", "Cebpe", "Ly6g", "Cxcr2", "Fcgr3",
            "Mmp9", "Sell", "Cd177", "Csf3r",
        ],
        "Monocyte": [
            "Lyz2", "Csf1r", "Ctss", "Ctsb", "Lgals3", "Mpeg1", "Ccr2", "Ly6c2",
            "Itgam", "Cd14", "Fcgr1", "Sirpa", "Cybb", "Ms4a7", "Plac8", "S100a4",
            "Cebpb", "Irf8", "Lpl", "Aif1", "Tyrobp", "Ccl6", "Lst1",
        ],
        "Baso": [
            "Gata2", "Cpa3", "Prg2", "Hdc", "Ms4a2", "Fcer1a", "Alox5", "Srgn",
            "Ccr3", "Il4", "Il13", "Mcpt8", "Csf2rb", "Csf2rb2", "Kit",
            "Il3ra", "Fcer1g", "Cd200r3", "Slc6a4",
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
        "Eos": [
            "Prg2", "Epx", "Ear1", "Ear2", "Ear6", "Ccr3", "Siglecf", "Il5ra",
            "Alox15", "Rnase2a", "Rnase2b", "Lgals10",
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
    }

    # ============================================================
    # LOAD FORCE TABLE
    # ============================================================

    if not os.path.exists(FORCE_PATH):
        raise FileNotFoundError(f"Could not find FORCE_PATH:\n{FORCE_PATH}")

    force_df = pd.read_csv(FORCE_PATH)
    print("Loaded force_df:", force_df.shape)
    print("Columns:", list(force_df.columns))

    required_cols = {"fold", "fate", "direction", "rank", "gene"}
    missing = required_cols - set(force_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    EFFECT_COL = first_existing_column(
        force_df,
        ["posterior_mu_u", "mean_u", "u"],
        required=True,
    )

    PIP_COL = first_existing_column(
        force_df,
        ["posterior_pip", "pip", "activity_probability"],
        required=True,
    )

    print("Using effect column:", EFFECT_COL)
    print("Using PIP column:", PIP_COL)

    # Background genes
    if os.path.exists(HVG_PATH):
        hvg_df = pd.read_csv(HVG_PATH)
        background_genes = hvg_df["gene"].astype(str).tolist()
        print(f"Background: {len(background_genes)} HVGs from {HVG_PATH}")
    else:
        background_genes = force_df["gene"].astype(str).unique().tolist()
        print(f"Background: {len(background_genes)} genes from force table")

    # Direction filter
    force_use = force_df[
        force_df["direction"].astype(str).str.lower() == USE_DIRECTION.lower()
    ].copy()

    if force_use.empty:
        raise ValueError(
            f"No rows found with direction={USE_DIRECTION}. "
            f"Available directions: {sorted(force_df['direction'].astype(str).unique())}"
        )

    force_use[EFFECT_COL] = pd.to_numeric(force_use[EFFECT_COL], errors="coerce")
    force_use[PIP_COL] = pd.to_numeric(force_use[PIP_COL], errors="coerce")
    force_use = force_use[np.isfinite(force_use[EFFECT_COL])].copy()

    # ============================================================
    # AGGREGATE ACROSS FOLDS
    # ============================================================

    agg = (
        force_use
        .groupby(["fate", "gene"], as_index=False)
        .agg(
            mean_effect=(EFFECT_COL, "mean"),
            median_effect=(EFFECT_COL, "median"),
            max_effect=(EFFECT_COL, "max"),
            mean_pip=(PIP_COL, "mean"),
            max_pip=(PIP_COL, "max"),
            mean_rank=("rank", "mean"),
            min_rank=("rank", "min"),
            n_folds_present=("fold", "nunique"),
        )
    )

    agg["gene_norm"] = agg["gene"].map(norm_gene)

    # rank genes by fold recurrence, then posterior mean effect, then PIP, then rank
    agg = agg.sort_values(
        ["fate", "n_folds_present", "mean_effect", "mean_pip", "mean_rank"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    # Fate order
    fates_found = agg["fate"].drop_duplicates().tolist()
    selected_fates = [f for f in PREFERRED_FATE_ORDER if f in fates_found]
    selected_fates += [f for f in fates_found if f not in selected_fates]

    print("\nFates:")
    print(selected_fates)

    # Top genes per fate
    top_genes_by_fate = {}

    for fate in selected_fates:
        sub = agg[agg["fate"] == fate].copy()
        top_genes_by_fate[fate] = sub.head(TOP_N)["gene"].astype(str).tolist()

    print(f"\nTop {TOP_N} genes per fate:")
    for fate in selected_fates:
        print(f"\n{fate}")
        print(", ".join(top_genes_by_fate[fate]))

    # ============================================================
    # PLOT 1: TOP N PIP HEATMAP
    # ============================================================

    top_union = []
    for fate in selected_fates:
        top_union.extend(top_genes_by_fate[fate])
    top_union = list(dict.fromkeys(top_union))

    pip_heat = (
        agg
        .pivot_table(index="gene", columns="fate", values="mean_pip", fill_value=0.0)
        .reindex(top_union)
        .reindex(columns=selected_fates)
    )

    plt.figure(figsize=(1.2 * len(selected_fates) + 5, 0.20 * len(top_union) + 5))

    sns.heatmap(
        pip_heat,
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "mean posterior activity probability"},
    )

    plt.title(f"Top {TOP_N} Bayesian posterior CIPHER gene PIP")
    plt.xlabel("CIPHER fate force")
    plt.ylabel("gene")
    plt.tight_layout()

    pip_png = os.path.join(OUTDIR, f"top{TOP_N}_bayesian_posterior_pip_heatmap_ONLY.png")
    pip_svg = os.path.join(OUTDIR, f"top{TOP_N}_bayesian_posterior_pip_heatmap_ONLY.svg")

    plt.savefig(pip_png, dpi=300, bbox_inches="tight")
    plt.savefig(pip_svg, bbox_inches="tight")
    plt.show()

    # ============================================================
    # CANONICAL EXPECTED MARKER ENRICHMENT
    # ============================================================

    canonical_results = []

    for fate in selected_fates:
        query = top_genes_by_fate[fate]

        for marker_category, marker_genes in CANONICAL_MARKERS.items():
            res = hypergeom_enrich(query, marker_genes, background_genes)

            canonical_results.append({
                "cipher_fate": fate,
                "marker_category": marker_category,
                "top_n": TOP_N,
                **{k: v for k, v in res.items() if k != "overlap_genes_norm"},
                "overlap_genes": ", ".join(res["overlap_genes_norm"]),
            })

    canonical_results = pd.DataFrame(canonical_results)
    canonical_results["fdr"] = bh_fdr(canonical_results["p_value"].values)
    canonical_results["minus_log10_fdr"] = -np.log10(np.maximum(canonical_results["fdr"].values, 1e-300))
    canonical_results["log2_odds_enrichment"] = np.log2(
        canonical_results["odds_enrichment"].replace(0, np.nan)
    )

    canonical_results.to_csv(
        os.path.join(OUTDIR, f"canonical_marker_enrichment_top{TOP_N}_ONLY.csv"),
        index=False,
    )

    expected_rows = []

    for fate in selected_fates:
        expected_cat = expected_category_for_fate(fate)
        if expected_cat is None:
            continue

        sub = canonical_results[
            (canonical_results["cipher_fate"] == fate) &
            (canonical_results["marker_category"] == expected_cat)
        ].copy()

        if len(sub) == 0:
            continue

        row = sub.iloc[0].to_dict()
        row["expected_marker_category"] = expected_cat
        expected_rows.append(row)

    expected_df = pd.DataFrame(expected_rows)

    if expected_df.empty:
        raise RuntimeError("No expected fate/category matches found.")

    expected_df["cipher_fate"] = pd.Categorical(
        expected_df["cipher_fate"],
        categories=[f for f in selected_fates if f in expected_df["cipher_fate"].astype(str).tolist()],
        ordered=True,
    )
    expected_df = expected_df.sort_values("cipher_fate")

    expected_df.to_csv(
        os.path.join(OUTDIR, f"expected_canonical_marker_summary_top{TOP_N}_ONLY.csv"),
        index=False,
    )

    # ============================================================
    # PLOTS 2 + 3: EXPECTED SIGNIFICANCE + ENRICHMENT
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.barplot(
        data=expected_df,
        x="cipher_fate",
        y="minus_log10_fdr",
        ax=axes[0],
        color=sns.color_palette("tab10")[0],
    )
    axes[0].set_title("Expected canonical marker significance")
    axes[0].set_ylabel("-log10(FDR)")
    axes[0].set_xlabel("CIPHER fate")
    axes[0].tick_params(axis="x", rotation=45)
    for tick in axes[0].get_xticklabels():
        tick.set_ha("right")

    sns.barplot(
        data=expected_df,
        x="cipher_fate",
        y="odds_enrichment",
        ax=axes[1],
        color=sns.color_palette("tab10")[0],
    )
    axes[1].set_title("Expected canonical marker enrichment")
    axes[1].set_ylabel("odds enrichment")
    axes[1].set_xlabel("CIPHER fate")
    axes[1].tick_params(axis="x", rotation=45)
    for tick in axes[1].get_xticklabels():
        tick.set_ha("right")

    plt.tight_layout()

    bar_png = os.path.join(OUTDIR, f"expected_canonical_marker_significance_enrichment_top{TOP_N}_ONLY.png")
    bar_svg = os.path.join(OUTDIR, f"expected_canonical_marker_significance_enrichment_top{TOP_N}_ONLY.svg")

    plt.savefig(bar_png, dpi=300, bbox_inches="tight")
    plt.savefig(bar_svg, bbox_inches="tight")
    plt.show()

    print("\nSaved:")
    print(" ", pip_png)
    print(" ", pip_svg)
    print(" ", bar_png)
    print(" ", bar_svg)
    print(" ", os.path.join(OUTDIR, f"canonical_marker_enrichment_top{TOP_N}_ONLY.csv"))
    print(" ", os.path.join(OUTDIR, f"expected_canonical_marker_summary_top{TOP_N}_ONLY.csv"))

    print("\nExpected marker summary:")
    print(
        expected_df[
            [
                "cipher_fate",
                "expected_marker_category",
                "k_overlap",
                "odds_enrichment",
                "p_value",
                "fdr",
                "minus_log10_fdr",
                "overlap_genes",
            ]
        ].to_string(index=False)
    )

