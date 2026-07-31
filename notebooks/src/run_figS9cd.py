"""Fig S9 C/D -- full-H0 posterior double-perturbation inverse recovery (per-dataset ROC + PR).

Draws two independent control pseudobulks / posterior scores under the full-H0 whitening model and
evaluates target-recovery ROC/PR for double perturbations. Helpers in notebooks/src (not part of the
cipher package). Config constants live inside the entry functions; DATA_DIR / SUPPL / OUTDIR are module
globals injected by the notebook via R.__dict__.update.
"""
import os

DATA_DIR = None
SUPPL = None
OUTDIR = None


def run_double_pert_inverse():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # FAST RAW h5ad -> FULL-H0 POSTERIOR INVERSE FOR DOUBLE PERTURBATIONS
    #
    # FINAL PLOTS ONLY:
    #   1. AUROC plot with one curve per dataset
    #   2. PRC plot with one curve per dataset
    #
    # Uses FULL H0 whitening:
    #
    #   H0 = full control covariance
    #   H0 = Q diag(lambda) Q.T
    #   W  = diag(lambda)^(-1/2) Q.T
    #
    #   y = W DeltaX
    #   A = W Sigma
    #
    # Here, since we only load data and calculate Sigma from controls:
    #
    #   Sigma = H0 = full control covariance
    #
    # so the operator has the shortcut:
    #
    #   A = H0^(-1/2) H0 = H0^(1/2)
    #
    # Double perturbations:
    #   Each perturbation has two positive target genes.
    #
    # Main options:
    #   FORCE_INCLUDE_PERTURBED_GENES = True / False
    #   MAX_GENES_AFTER_FILTER controls full-H0 eigendecomposition size.
    # ============================================================

    import os
    import re
    import gc
    import json
    import time
    from pathlib import Path

    import h5py
    import numpy as np
    import pandas as pd
    import anndata as ad
    import matplotlib.pyplot as plt

    from tqdm import tqdm
    from scipy.sparse import issparse
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import (
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score,
    )

    # ============================================================
    # CONFIG
    # ============================================================

    DATA_PATHS = [os.path.join(DATA_DIR, _n) for _n in [
        "proper_filtered.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]]

    OUT_ROOT = os.path.join(OUTDIR, "FAST_fullH0_double_pert_inverse_from_h5ad")

    PERTURBATION_COL = "perturbation"

    CONTROL_ALIASES = {
        "negative",
        "control",
        "ctrl",
        "NTC",
        "ntc",
        "non-targeting",
        "non_targeting",
        "negative_control",
        "safe-targeting",
        "safe_targeting",
    }

    EXPRESSION_CUTOFF = 0.01

    # ------------------------------------------------------------
    # IMPORTANT OPTION
    # ------------------------------------------------------------
    # False:
    #   only genes with mean >= EXPRESSION_CUTOFF are kept.
    #
    # True:
    #   perturbed target genes are kept even if below expression cutoff,
    #   as long as they appear in adata.var_names.
    # ------------------------------------------------------------
    FORCE_INCLUDE_PERTURBED_GENES = True

    MIN_CELLS_PER_PERT = 2

    # Full H0 eigendecomposition scales roughly O(p^3).
    # Start with 1000-2000 for speed, then increase.
    MAX_GENES_AFTER_FILTER = 1000

    # Controls sampled for full H0/Sigma covariance.
    MAX_CONTROL_CELLS_FOR_COV = 10000

    # Exactly double perturbations.
    MIN_N_TARGETS = 2
    MAX_N_TARGETS = 2

    # all_present:
    #   both genes in the double perturbation must be present after filtering.
    #
    # any_present:
    #   at least one target must be present.
    #
    # Usually use all_present for double perturbation recovery.
    TARGET_MATCH_POLICY = "all_present"

    RANDOM_SEED = 1

    # Full H0 eigenvalue handling.
    H0_EIG_FLOOR_REL = 1e-4
    H0_EIG_FLOOR_ABS = 1e-8
    H0_MAX_MODES = None  # e.g. 1500 for low-rank full-H0 whitening

    # Ridge added to control covariance before eigendecomposition.
    COV_RIDGE_REL = 1e-4

    LOGTAU2_BOUNDS = (6., 8.0)
    EB_GRID_N = 100
    PLATEAU_DELTA = 1.92
    PLATEAU_PREFER = "largest"  # "largest" or "smallest"
    USE_PLATEAU_FOR_SCORING = True

    PERT_BATCH = 64

    ROC_NEGATIVES_PER_PERT = 1024
    ROC_SEED = 0

    SAVE_POSTERIOR_MU = True
    SAVE_POSTERIOR_SCORE_MATRIX = False
    SAVE_COVARIANCE = False

    PLOT_DPI = 300
    SHOW_FINAL_PLOTS = True


    # ============================================================
    # BASIC HELPERS
    # ============================================================

    def ensure_dir(x):
        x = Path(x)
        x.mkdir(parents=True, exist_ok=True)
        return x


    def safe_name(x):
        x = str(x)
        x = re.sub(r"[^\w\-.]+", "_", x)
        x = re.sub(r"_+", "_", x).strip("_")
        return x[:180]


    def sym(A):
        return 0.5 * (A + A.T)


    def nan0(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


    def as_csr_or_dense_float32(X):
        if issparse(X):
            return X.tocsr().astype(np.float32)
        return np.asarray(X, dtype=np.float32)


    def mean_axis0_fast(X):
        if issparse(X):
            return np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
        return np.asarray(X, dtype=np.float64).mean(axis=0)


    def mean_var_axis0_fast(X, ddof=1):
        n = X.shape[0]

        if issparse(X):
            mu = np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
            mu2 = np.asarray(X.multiply(X).mean(axis=0)).ravel().astype(np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)
            mu = X.mean(axis=0)
            mu2 = (X * X).mean(axis=0)

        var = mu2 - mu * mu
        var = nan0(var)
        var = np.maximum(var, 0.0)

        if ddof == 1 and n > 1:
            var *= n / (n - 1)

        return mu, var


    def h5_strings(h5, name, values):
        dt = h5py.string_dtype("utf-8")
        h5.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dt)


    def find_control_label(perturbations):
        counts = pd.Series(perturbations.astype(str)).value_counts()

        for alias in CONTROL_ALIASES:
            if alias in counts.index:
                return alias

        lower_map = {str(v).lower(): str(v) for v in counts.index}
        for alias in CONTROL_ALIASES:
            if alias.lower() in lower_map:
                return lower_map[alias.lower()]

        raise ValueError(
            "Could not infer control label. "
            f"Available examples: {list(counts.index[:30])}"
        )


    def clean_target_token(x):
        x = str(x).strip()

        x = re.sub(r"^(sgRNA|gRNA|sgrna|grna|sg)[_\-\s]*", "", x, flags=re.IGNORECASE)
        x = re.sub(r"([_\-\s]*g\d+)$", "", x, flags=re.IGNORECASE)
        x = re.sub(
            r"([_\-\s]+)(KD|KO|OE|CRISPRi|CRISPRa|overexp|overexpression)$",
            "",
            x,
            flags=re.IGNORECASE,
        )

        return x.strip()


    def parse_targets_from_perturbation(pert):
        p = str(pert).strip()

        if p.lower() in {x.lower() for x in CONTROL_ALIASES}:
            return []

        if "_" in p:
            parts = p.split("_")
        elif "+" in p:
            parts = p.split("+")
        elif "|" in p:
            parts = p.split("|")
        elif ";" in p:
            parts = p.split(";")
        elif "," in p:
            parts = p.split(",")
        else:
            parts = [p]

        targets = []
        for part in parts:
            g = clean_target_token(part)
            if g:
                targets.append(g)

        return list(dict.fromkeys(targets))


    def get_target_genes_from_perts(perts):
        out = set()
        for p in perts:
            for g in parse_targets_from_perturbation(p):
                out.add(str(g))
        return out


    def choose_top_genes_after_expression_filter(
        X_relevant,
        gene_names_all,
        expr_keep,
        force_keep_mask,
        candidate_perts,
    ):
        gene_names_all = np.asarray(gene_names_all, dtype=object)

        base_keep = expr_keep | force_keep_mask

        if MAX_GENES_AFTER_FILTER is None:
            keep = np.where(base_keep)[0]
            print(f"[gene cap] no cap, final genes={len(keep):,}")
            return keep

        print("[gene cap] computing variance on relevant cells")
        _, rel_var = mean_var_axis0_fast(X_relevant, ddof=1)

        gene_to_idx_all = {str(g): i for i, g in enumerate(gene_names_all)}
        candidate_target_genes = get_target_genes_from_perts(candidate_perts)

        target_idxs_to_keep = set()
        for g in candidate_target_genes:
            j = gene_to_idx_all.get(str(g), -1)
            if j >= 0 and base_keep[j]:
                target_idxs_to_keep.add(j)

        eligible = np.where(base_keep)[0]

        eligible_non_target = np.asarray(
            [j for j in eligible if j not in target_idxs_to_keep],
            dtype=np.int64,
        )

        n_target = len(target_idxs_to_keep)
        n_take_non_target = max(int(MAX_GENES_AFTER_FILTER) - n_target, 0)

        if eligible_non_target.size > n_take_non_target:
            order = np.argsort(rel_var[eligible_non_target])[::-1]
            top_non_target = eligible_non_target[order[:n_take_non_target]]
        else:
            top_non_target = eligible_non_target

        keep = np.asarray(
            sorted(set(top_non_target.tolist()) | target_idxs_to_keep),
            dtype=np.int64,
        )

        print(
            f"[gene cap] expr-pass={int(expr_keep.sum()):,}, "
            f"force-target-pass={int(force_keep_mask.sum()):,}, "
            f"target genes kept={n_target:,}, final genes={len(keep):,}"
        )

        return keep


    # ============================================================
    # MULTI-POSITIVE ROC/PRC ACCUMULATOR
    # ============================================================

    class MultiPositiveScoreAccumulator:
        def __init__(self, target_indices_list, n_genes, seed=0, negatives_per_pert=1024):
            self.target_indices_list = [
                np.asarray(x, dtype=np.int64) for x in target_indices_list
            ]

            self.n_perts = len(self.target_indices_list)
            self.n_genes = int(n_genes)
            self.rng = np.random.default_rng(seed)
            self.negatives_per_pert = int(negatives_per_pert)

            self.per_auc = np.full(self.n_perts, np.nan, dtype=np.float64)
            self.best_rank = np.full(self.n_perts, np.nan, dtype=np.float64)
            self.mean_target_rank = np.full(self.n_perts, np.nan, dtype=np.float64)

            self.pos_parts = []
            self.neg_parts = []

        def update(self, score_batch, row_start):
            S = np.asarray(score_batch, dtype=np.float64)
            S = np.nan_to_num(S, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

            bsz, p = S.shape

            for ii in range(bsz):
                row = row_start + ii
                if row >= self.n_perts:
                    continue

                targets = self.target_indices_list[row]
                targets = targets[(targets >= 0) & (targets < p)]

                if targets.size == 0:
                    continue

                srow = S[ii]
                pos_scores = srow[targets]
                pos_scores = pos_scores[np.isfinite(pos_scores)]

                if pos_scores.size == 0:
                    continue

                neg_mask = np.ones(p, dtype=bool)
                neg_mask[targets] = False

                neg_scores_all = srow[neg_mask]
                neg_scores_all = neg_scores_all[np.isfinite(neg_scores_all)]

                if neg_scores_all.size == 0:
                    continue

                aucs = []
                ranks = []

                for ps in pos_scores:
                    aucs.append(np.mean((ps > neg_scores_all) + 0.5 * (ps == neg_scores_all)))
                    ranks.append(1 + np.sum(srow > ps))

                self.per_auc[row] = float(np.mean(aucs))
                self.best_rank[row] = float(np.min(ranks))
                self.mean_target_rank[row] = float(np.mean(ranks))

                self.pos_parts.append(pos_scores.astype(np.float32, copy=False))

                k = min(self.negatives_per_pert, neg_scores_all.size)
                if k > 0:
                    idx = self.rng.choice(neg_scores_all.size, size=k, replace=False)
                    self.neg_parts.append(neg_scores_all[idx].astype(np.float32, copy=False))

        def finish(self):
            roc = None
            prc = None
            pooled_auc = np.nan
            pooled_average_precision = np.nan
            pooled_auprc_trapz = np.nan

            if self.pos_parts and self.neg_parts:
                pos = np.concatenate(self.pos_parts).astype(np.float64)
                neg = np.concatenate(self.neg_parts).astype(np.float64)

                scores = np.concatenate([pos, neg])
                labels = np.concatenate([
                    np.ones(pos.size, dtype=np.int8),
                    np.zeros(neg.size, dtype=np.int8),
                ])

                good = np.isfinite(scores)
                scores = scores[good]
                labels = labels[good]

                if labels.size >= 2 and labels.sum() > 0 and labels.sum() < labels.size:
                    fpr, tpr, _ = roc_curve(labels, scores)
                    precision, recall, _ = precision_recall_curve(labels, scores)

                    pooled_auc = float(auc(fpr, tpr))
                    pooled_average_precision = float(average_precision_score(labels, scores))

                    order = np.argsort(recall)
                    pooled_auprc_trapz = float(auc(recall[order], precision[order]))

                    roc = (fpr.astype(np.float32), tpr.astype(np.float32))
                    prc = (precision.astype(np.float32), recall.astype(np.float32))

            good_auc = np.isfinite(self.per_auc)
            good_rank = np.isfinite(self.best_rank)

            summary = {
                "n_valid_perts": int(good_auc.sum()),
                "pooled_auc": pooled_auc,
                "pooled_average_precision": pooled_average_precision,
                "pooled_auprc_trapz": pooled_auprc_trapz,
                "mean_per_pert_auc": float(np.nanmean(self.per_auc)) if good_auc.any() else np.nan,
                "median_per_pert_auc": float(np.nanmedian(self.per_auc)) if good_auc.any() else np.nan,
                "median_best_rank": float(np.nanmedian(self.best_rank)) if good_rank.any() else np.nan,
                "median_mean_target_rank": float(np.nanmedian(self.mean_target_rank)) if np.isfinite(self.mean_target_rank).any() else np.nan,
                "top1_any_target": float(np.nanmean(self.best_rank[good_rank] <= 1)) if good_rank.any() else np.nan,
                "top5_any_target": float(np.nanmean(self.best_rank[good_rank] <= 5)) if good_rank.any() else np.nan,
                "top10_any_target": float(np.nanmean(self.best_rank[good_rank] <= 10)) if good_rank.any() else np.nan,
            }

            return summary, roc, prc


    # ============================================================
    # FAST LOAD + FILTER
    # ============================================================

    def load_and_filter_fast(data_path):
        data_path = Path(data_path)
        dataset = data_path.name.replace(".h5ad", "")

        print("\n" + "=" * 100)
        print(f"[dataset] {dataset}")
        print(f"[path]    {data_path}")
        print("=" * 100)

        print("[load] reading h5ad")
        adata = ad.read_h5ad(data_path)
        adata.var_names = adata.var_names.astype(str)
        adata.var_names_make_unique()

        if PERTURBATION_COL not in adata.obs.columns:
            raise KeyError(f"Missing obs column: {PERTURBATION_COL!r}")

        pert_all = adata.obs[PERTURBATION_COL].astype(str).values
        gene_names_all = np.asarray(adata.var_names, dtype=object)
        gene_set_all = set(map(str, gene_names_all))

        print("[X] preparing matrix")
        X_all = as_csr_or_dense_float32(adata.X)

        control_label = find_control_label(pert_all)
        print(f"[control] {control_label!r}")

        print("[obs] finding candidate double perturbations before gene filtering")
        counts = pd.Series(pert_all).value_counts()

        candidate_perts = []

        for pert, n in counts.items():
            pert = str(pert)

            if pert == str(control_label):
                continue

            targets = parse_targets_from_perturbation(pert)

            if MIN_N_TARGETS <= len(targets) <= MAX_N_TARGETS and n >= MIN_CELLS_PER_PERT:
                candidate_perts.append(pert)

        print(f"[candidate double perts] {len(candidate_perts):,}")

        if len(candidate_perts) == 0:
            raise RuntimeError("No candidate double perturbations before gene filtering.")

        candidate_set = set(candidate_perts)

        relevant_mask = (pert_all == str(control_label)) | np.asarray(
            [p in candidate_set for p in pert_all],
            dtype=bool,
        )

        relevant_idx = np.where(relevant_mask)[0]

        print(
            f"[cell prefilter] relevant cells={len(relevant_idx):,}/{X_all.shape[0]:,} "
            f"(controls + candidate double perts)"
        )

        X_relevant_allgenes = X_all[relevant_idx, :]

        print("[gene filter] computing means on relevant cells only")
        gene_means = mean_axis0_fast(X_relevant_allgenes)
        expr_keep = gene_means >= float(EXPRESSION_CUTOFF)

        print(
            f"[gene filter] mean >= {EXPRESSION_CUTOFF}: "
            f"{int(expr_keep.sum()):,}/{len(expr_keep):,} genes"
        )

        # --------------------------------------------------------
        # Force include target genes, if requested.
        # --------------------------------------------------------
        force_keep_mask = np.zeros(len(gene_names_all), dtype=bool)

        if FORCE_INCLUDE_PERTURBED_GENES:
            print("[force include] adding perturbed target genes regardless of expression cutoff")

            target_genes = get_target_genes_from_perts(candidate_perts)
            gene_to_idx_all = {str(g): i for i, g in enumerate(gene_names_all)}

            found = 0
            missing = 0

            for g in target_genes:
                j = gene_to_idx_all.get(str(g), -1)
                if j >= 0:
                    force_keep_mask[j] = True
                    found += 1
                else:
                    missing += 1

            print(f"[force include] target genes found in var_names={found:,}, missing={missing:,}")
        else:
            print("[force include] OFF")

        keep_gene_idx = choose_top_genes_after_expression_filter(
            X_relevant=X_relevant_allgenes,
            gene_names_all=gene_names_all,
            expr_keep=expr_keep,
            force_keep_mask=force_keep_mask,
            candidate_perts=candidate_perts,
        )

        gene_names = gene_names_all[keep_gene_idx]
        gene_to_idx = {str(g): i for i, g in enumerate(gene_names)}

        print("[subset] creating relevant cell x filtered gene matrix")
        X_rel = X_relevant_allgenes[:, keep_gene_idx]

        if issparse(X_rel):
            X_rel = X_rel.tocsr().astype(np.float32)
        else:
            X_rel = np.asarray(X_rel, dtype=np.float32)

        pert_rel = pert_all[relevant_idx]

        print("[targets] validating target genes after filtering")
        final_perts = []
        target_names_list = []
        target_indices_list = []

        rel_counts = pd.Series(pert_rel).value_counts()

        for pert in candidate_perts:
            if rel_counts.get(pert, 0) < MIN_CELLS_PER_PERT:
                continue

            targets = parse_targets_from_perturbation(pert)
            idxs = [gene_to_idx.get(str(g), -1) for g in targets]
            present = [i for i in idxs if i >= 0]

            if TARGET_MATCH_POLICY == "all_present":
                if len(present) != len(targets):
                    continue
            elif TARGET_MATCH_POLICY == "any_present":
                if len(present) == 0:
                    continue
            else:
                raise ValueError("TARGET_MATCH_POLICY must be 'all_present' or 'any_present'")

            final_perts.append(pert)
            target_names_list.append(targets)
            target_indices_list.append(np.asarray(present, dtype=np.int64))

        print(f"[final double perts] {len(final_perts):,}")

        if len(final_perts) == 0:
            raise RuntimeError("No final double perturbations after gene filtering.")

        final_set = set(final_perts)

        final_cell_mask = (pert_rel == str(control_label)) | np.asarray(
            [p in final_set for p in pert_rel],
            dtype=bool,
        )

        X = X_rel[final_cell_mask, :]
        pert = pert_rel[final_cell_mask]

        if issparse(X):
            X = X.tocsr().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        control_mask = pert == str(control_label)

        print(
            f"[final matrix] cells={X.shape[0]:,}, genes={X.shape[1]:,}, "
            f"controls={int(control_mask.sum()):,}"
        )

        del adata, X_all, X_relevant_allgenes, X_rel
        gc.collect()

        return {
            "dataset": dataset,
            "X": X,
            "pert": pert,
            "control_label": control_label,
            "control_mask": control_mask,
            "gene_names": gene_names,
            "perts": np.asarray(final_perts, dtype=object),
            "target_names_list": target_names_list,
            "target_indices_list": target_indices_list,
        }


    # ============================================================
    # DX STATS
    # ============================================================

    def compute_dx_fast(data):
        X = data["X"]
        pert = data["pert"]
        control_mask = data["control_mask"]
        perts = data["perts"]

        X0 = X[control_mask, :]

        print("[control stats] mean/variance")
        mu0, var0 = mean_var_axis0_fast(X0, ddof=1)

        n_perts = len(perts)
        p = X.shape[1]

        DX = np.zeros((n_perts, p), dtype=np.float32)
        n_cells_pert = np.zeros(n_perts, dtype=np.int64)

        print("[pert stats] dx")
        for i, pertx in enumerate(tqdm(perts, desc="pert stats", dynamic_ncols=True)):
            idx = np.where(pert == str(pertx))[0]
            Xi = X[idx, :]
            n_cells_pert[i] = Xi.shape[0]

            mu1 = mean_axis0_fast(Xi)
            DX[i, :] = (mu1 - mu0).astype(np.float32)

        return mu0, var0, DX, n_cells_pert


    # ============================================================
    # FULL H0 / SIGMA / WHITENING
    # ============================================================

    def compute_full_control_covariance(data):
        X = data["X"]
        control_mask = data["control_mask"]

        X0 = X[control_mask, :]
        n0 = X0.shape[0]

        rng = np.random.default_rng(RANDOM_SEED)

        if n0 > MAX_CONTROL_CELLS_FOR_COV:
            pick = np.sort(rng.choice(n0, size=MAX_CONTROL_CELLS_FOR_COV, replace=False))
            Xc = X0[pick, :]
        else:
            Xc = X0

        print(f"[full H0/Sigma] densifying sampled controls: {Xc.shape[0]:,} x {Xc.shape[1]:,}")

        if issparse(Xc):
            Xc = Xc.toarray()
        else:
            Xc = np.asarray(Xc)

        Xc = np.asarray(Xc, dtype=np.float64)
        Xc -= Xc.mean(axis=0, keepdims=True)

        print("[full H0/Sigma] covariance")
        H0 = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
        H0 = sym(nan0(H0))

        diag = np.diag(H0)
        good_diag = diag[np.isfinite(diag) & (diag > 0)]
        ridge = float(np.median(good_diag) * COV_RIDGE_REL) if good_diag.size else COV_RIDGE_REL

        H0 += ridge * np.eye(H0.shape[0], dtype=np.float64)
        H0 = sym(nan0(H0))

        # In this raw-data version, Sigma is calculated directly from controls,
        # so Sigma == H0.
        Sigma = H0.copy()

        print(
            f"[full H0/Sigma] shape={H0.shape}, ridge={ridge:.3e}, "
            f"diag median={np.median(np.diag(H0)):.3e}"
        )

        del Xc
        gc.collect()

        return H0, Sigma, ridge


    def build_full_H0_operator(H0):
        """
        Full H0 whitening with Sigma == H0.

        H0 = Q Lambda Q.T
        W = Lambda^{-1/2} Q.T

        A = W H0 = Lambda^{1/2} Q.T

        SVD shortcut:
            U  = None / identity in whitened eigenbasis
            s  = sqrt(lambda)
            Vt = Q.T
        """
        p = H0.shape[0]

        print("[full H0] eigendecomposition")
        evals, Q = np.linalg.eigh(H0)
        evals = nan0(evals)

        scale = np.median(evals[evals > 0]) if np.any(evals > 0) else 1.0
        floor = max(float(H0_EIG_FLOOR_ABS), float(H0_EIG_FLOOR_REL) * float(scale))

        keep = evals > floor

        if H0_MAX_MODES is not None and np.sum(keep) > int(H0_MAX_MODES):
            idx = np.argsort(evals)[-int(H0_MAX_MODES):]
            keep = np.zeros_like(keep, dtype=bool)
            keep[idx] = True

        if np.sum(keep) == 0:
            raise RuntimeError("No H0 eigenmodes kept. Lower H0_EIG_FLOOR_REL or H0_EIG_FLOOR_ABS.")

        evals_keep = evals[keep].astype(np.float64)
        Q_keep = Q[:, keep].astype(np.float64)

        order = np.argsort(evals_keep)[::-1]
        evals_keep = evals_keep[order]
        Q_keep = Q_keep[:, order]

        invsqrt = 1.0 / np.sqrt(evals_keep)

        print(
            f"[full H0] kept modes={len(evals_keep):,}/{p:,}, "
            f"floor={floor:.3e}, min_kept={evals_keep.min():.3e}, "
            f"max_kept={evals_keep.max():.3e}"
        )

        U = None
        s = np.sqrt(evals_keep).astype(np.float64)
        Vt = Q_keep.T.astype(np.float64)

        whitener = {
            "Q": Q_keep,
            "evals": evals_keep,
            "invsqrt": invsqrt,
        }

        del Q
        gc.collect()

        return U, s, Vt, whitener


    def whiten_dx_batch_fullH0(DX_batch, whitener):
        """
        DX_batch: batch x genes
        returns: kept_modes x batch
        """
        DX = np.asarray(DX_batch, dtype=np.float64)
        Y = whitener["Q"].T @ DX.T
        Y *= whitener["invsqrt"][:, None]
        return nan0(Y)


    def project_Ut(U, Y):
        if U is None:
            return Y
        return U.T @ Y


    # ============================================================
    # EB + POSTERIOR
    # ============================================================

    def fit_tau2_from_svd_and_dx(U, s, whitener, DX):
        n_perts = DX.shape[0]
        z2_sum = np.zeros_like(s, dtype=np.float64)

        for a in tqdm(range(0, n_perts, PERT_BATCH), desc="EB dx", leave=False, dynamic_ncols=True):
            b = min(a + PERT_BATCH, n_perts)
            Y = whiten_dx_batch_fullH0(DX[a:b, :], whitener)
            Z = project_Ut(U, Y)
            z2_sum += np.sum(Z * Z, axis=1)

        s2 = s * s

        def nll(log_tau2):
            tau2 = float(np.exp(log_tau2))
            C = np.maximum(1.0 + tau2 * s2, 1e-12)
            return 0.5 * float(n_perts * np.sum(np.log(C)) + np.sum(z2_sum / C))

        res = minimize_scalar(nll, bounds=LOGTAU2_BOUNDS, method="bounded")

        grid = np.linspace(LOGTAU2_BOUNDS[0], LOGTAU2_BOUNDS[1], EB_GRID_N)
        nll_grid = np.asarray([nll(x) for x in grid], dtype=np.float64)

        jmin = int(np.nanargmin(nll_grid))
        nll_min = float(nll_grid[jmin])
        ok = np.where(nll_grid <= nll_min + PLATEAU_DELTA)[0]

        if len(ok) == 0:
            jpl = jmin
        else:
            jpl = int(ok[-1] if PLATEAU_PREFER == "largest" else ok[0])

        tau2_opt = float(np.exp(res.x))
        tau2_plateau = float(np.exp(grid[jpl]))
        tau2_use = tau2_plateau if USE_PLATEAU_FOR_SCORING else tau2_opt

        return {
            "tau2_opt": tau2_opt,
            "logtau2_opt": float(res.x),
            "nll_opt": float(res.fun),
            "tau2_plateau": tau2_plateau,
            "logtau2_plateau": float(grid[jpl]),
            "nll_plateau": float(nll_grid[jpl]),
            "tau2_use": float(tau2_use),
            "grid_logtau2": grid.astype(np.float32),
            "grid_nll": nll_grid.astype(np.float32),
        }


    def posterior_score_batches(U, s, Vt, whitener, DX, tau2, acc):
        n_perts, p = DX.shape

        V = Vt.T
        s2 = s * s

        post_var_eig = 1.0 / np.maximum(s2 + 1.0 / float(tau2), 1e-12)
        gain = s * post_var_eig

        # If all p modes kept, this is exact.
        # If modes are truncated, add tau2 in the discarded orthogonal complement.
        row_energy_kept = np.sum(V * V, axis=1)
        row_energy_missing = np.maximum(1.0 - row_energy_kept, 0.0)

        posterior_var_diag = (V * V) @ post_var_eig
        posterior_var_diag += float(tau2) * row_energy_missing

        std = np.sqrt(np.maximum(posterior_var_diag, 0.0))
        std = nan0(std)

        score_save = np.zeros((n_perts, p), dtype=np.float32) if SAVE_POSTERIOR_SCORE_MATRIX else None
        mu_save = np.zeros((n_perts, p), dtype=np.float32) if SAVE_POSTERIOR_MU else None

        for a in tqdm(range(0, n_perts, PERT_BATCH), desc="posterior", leave=False, dynamic_ncols=True):
            b = min(a + PERT_BATCH, n_perts)

            Y = whiten_dx_batch_fullH0(DX[a:b, :], whitener)
            Z = project_Ut(U, Y)

            MU = (V @ (gain[:, None] * Z)).T
            MU = nan0(MU)

            SCORE = np.maximum(
                np.abs(MU + std[None, :]),
                np.abs(MU - std[None, :]),
            ).astype(np.float32)

            acc.update(SCORE, a)

            if score_save is not None:
                score_save[a:b, :] = SCORE

            if mu_save is not None:
                mu_save[a:b, :] = MU.astype(np.float32)

        return score_save, mu_save, std.astype(np.float32)


    # ============================================================
    # SAVING
    # ============================================================

    def save_dataset_outputs(
        out_dir,
        data,
        mu0,
        var0,
        H0,
        Sigma,
        whitener,
        eb,
        DX,
        n_cells_pert,
        posterior_score,
        posterior_mu,
        posterior_std,
        summary,
        roc,
        prc,
    ):
        out_dir = Path(out_dir)

        pd.DataFrame([summary]).to_csv(out_dir / "inverse_summary.csv", index=False)

        pd.DataFrame([{
            "dataset": data["dataset"],
            "full_H0": True,
            "sigma_equals_H0": True,
            "n_h0_modes": len(whitener["evals"]),
            "tau2_opt": eb["tau2_opt"],
            "logtau2_opt": eb["logtau2_opt"],
            "nll_opt": eb["nll_opt"],
            "tau2_plateau": eb["tau2_plateau"],
            "logtau2_plateau": eb["logtau2_plateau"],
            "nll_plateau": eb["nll_plateau"],
            "tau2_use": eb["tau2_use"],
        }]).to_csv(out_dir / "eb_tau2_summary.csv", index=False)

        np.savez(
            out_dir / "posterior_roc_prc_curves.npz",
            fpr=np.asarray([] if roc is None else roc[0], dtype=np.float32),
            tpr=np.asarray([] if roc is None else roc[1], dtype=np.float32),
            precision=np.asarray([] if prc is None else prc[0], dtype=np.float32),
            recall=np.asarray([] if prc is None else prc[1], dtype=np.float32),
        )

        target_joined = np.asarray(
            ["_".join(map(str, x)) for x in data["target_names_list"]],
            dtype=object,
        )

        np.savez(
            out_dir / "perpert_metrics.npz",
            perturbations=np.asarray(data["perts"], dtype=object),
            target_genes_joined=target_joined,
            n_cells_pert=n_cells_pert.astype(np.int64),
        )

        with h5py.File(out_dir / "inverse_outputs.h5", "w") as h5:
            h5.attrs["dataset"] = data["dataset"]
            h5.attrs["control_label"] = str(data["control_label"])
            h5.attrs["expression_cutoff"] = float(EXPRESSION_CUTOFF)
            h5.attrs["force_include_perturbed_genes"] = bool(FORCE_INCLUDE_PERTURBED_GENES)
            h5.attrs["max_genes_after_filter"] = -1 if MAX_GENES_AFTER_FILTER is None else int(MAX_GENES_AFTER_FILTER)
            h5.attrs["max_control_cells_for_cov"] = int(MAX_CONTROL_CELLS_FOR_COV)
            h5.attrs["min_cells_per_pert"] = int(MIN_CELLS_PER_PERT)
            h5.attrs["target_match_policy"] = str(TARGET_MATCH_POLICY)
            h5.attrs["full_H0"] = True
            h5.attrs["sigma_equals_H0"] = True
            h5.attrs["tau2_use"] = float(eb["tau2_use"])

            h5_strings(h5, "gene_names", data["gene_names"])
            h5_strings(h5, "perturbations", data["perts"])
            h5_strings(h5, "target_genes_joined", target_joined)

            max_t = max(len(x) for x in data["target_indices_list"])
            target_idx = np.full((len(data["target_indices_list"]), max_t), -1, dtype=np.int64)

            for i, x in enumerate(data["target_indices_list"]):
                target_idx[i, :len(x)] = np.asarray(x, dtype=np.int64)

            h5.create_dataset("target_idx", data=target_idx)
            h5.create_dataset("n_cells_pert", data=n_cells_pert.astype(np.int64))

            h5.create_dataset("control_mean", data=mu0.astype(np.float32))
            h5.create_dataset("control_var_diag_reference", data=var0.astype(np.float32))
            h5.create_dataset("h0_evals_kept", data=whitener["evals"].astype(np.float32))

            h5.create_dataset("dx", data=DX.astype(np.float32))
            h5.create_dataset("posterior_std_u", data=posterior_std.astype(np.float32))

            if posterior_mu is not None:
                h5.create_dataset("posterior_mu_u", data=posterior_mu.astype(np.float32))

            if posterior_score is not None:
                h5.create_dataset("score_posterior", data=posterior_score.astype(np.float32))

            if SAVE_COVARIANCE:
                h5.create_dataset("H0_full_control_cov", data=H0.astype(np.float32))
                h5.create_dataset("Sigma_full", data=Sigma.astype(np.float32))


    # ============================================================
    # ONE DATASET
    # ============================================================

    def run_one_dataset(data_path, run_out):
        data = load_and_filter_fast(data_path)

        dataset = data["dataset"]
        out_dir = ensure_dir(Path(run_out) / safe_name(dataset))

        mu0, var0, DX, n_cells_pert = compute_dx_fast(data)

        H0, Sigma, cov_ridge = compute_full_control_covariance(data)

        U, s, Vt, whitener = build_full_H0_operator(H0)

        print("[EB] fitting tau2")
        eb = fit_tau2_from_svd_and_dx(U, s, whitener, DX)
        tau2_use = float(eb["tau2_use"])

        print(
            f"[EB] tau2_opt={eb['tau2_opt']:.4g}, "
            f"tau2_plateau={eb['tau2_plateau']:.4g}, "
            f"tau2_use={tau2_use:.4g}"
        )

        print("[score] full-H0 posterior")

        acc = MultiPositiveScoreAccumulator(
            target_indices_list=data["target_indices_list"],
            n_genes=len(data["gene_names"]),
            seed=ROC_SEED,
            negatives_per_pert=ROC_NEGATIVES_PER_PERT,
        )

        posterior_score, posterior_mu, posterior_std = posterior_score_batches(
            U=U,
            s=s,
            Vt=Vt,
            whitener=whitener,
            DX=DX,
            tau2=tau2_use,
            acc=acc,
        )

        summary, roc, prc = acc.finish()

        summary = {
            "dataset": dataset,
            "method": "score_posterior",
            "label": "full-H0 posterior",
            "force_include_perturbed_genes": bool(FORCE_INCLUDE_PERTURBED_GENES),
            "expression_cutoff": float(EXPRESSION_CUTOFF),
            "max_genes_after_filter": -1 if MAX_GENES_AFTER_FILTER is None else int(MAX_GENES_AFTER_FILTER),
            "n_genes": int(len(data["gene_names"])),
            "n_double_perts": int(len(data["perts"])),
            "n_h0_modes": int(len(whitener["evals"])),
            "tau2_use": float(tau2_use),
            **summary,
        }

        save_dataset_outputs(
            out_dir=out_dir,
            data=data,
            mu0=mu0,
            var0=var0,
            H0=H0,
            Sigma=Sigma,
            whitener=whitener,
            eb=eb,
            DX=DX,
            n_cells_pert=n_cells_pert,
            posterior_score=posterior_score,
            posterior_mu=posterior_mu,
            posterior_std=posterior_std,
            summary=summary,
            roc=roc,
            prc=prc,
        )

        print(f"[saved] {out_dir}")

        result = {
            "dataset": dataset,
            "summary": summary,
            "roc": roc,
            "prc": prc,
        }

        del data, H0, Sigma, U, s, Vt, whitener
        del DX, posterior_score, posterior_mu, posterior_std
        gc.collect()

        return result


    # ============================================================
    # FINAL AGGREGATE PLOTS
    # ============================================================

    def plot_final_dataset_curves(results, run_out):
        run_out = Path(run_out)

        valid_roc = [r for r in results if r.get("roc") is not None]
        valid_prc = [r for r in results if r.get("prc") is not None]

        # --------------------------------------------------------
        # Final AUROC plot: one curve per dataset
        # --------------------------------------------------------
        fig1 = plt.figure(figsize=(6.6, 6.2))
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")

        for r in valid_roc:
            ds = r["dataset"]
            fpr, tpr = r["roc"]
            auc_val = float(r["summary"]["pooled_auc"])

            plt.plot(
                fpr,
                tpr,
                linewidth=2.2,
                label=f"{ds}: AUROC={auc_val:.3f}",
            )

        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title("Full-H0 posterior recovery of double perturbation targets")
        plt.legend(frameon=False, fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(run_out / "FINAL_AUROC_one_curve_per_dataset.png", dpi=PLOT_DPI)
        plt.savefig(run_out / "FINAL_AUROC_one_curve_per_dataset.svg")

        # --------------------------------------------------------
        # Final PRC plot: one curve per dataset
        # --------------------------------------------------------
        fig2 = plt.figure(figsize=(6.6, 6.2))

        for r in valid_prc:
            ds = r["dataset"]
            precision, recall = r["prc"]
            ap_val = float(r["summary"]["pooled_average_precision"])

            plt.plot(
                recall,
                precision,
                linewidth=2.2,
                label=f"{ds}: AP={ap_val:.3f}",
            )

        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("Full-H0 posterior recovery of double perturbation targets")
        plt.legend(frameon=False, fontsize=8, loc="upper right")
        plt.tight_layout()
        plt.savefig(run_out / "FINAL_PRC_one_curve_per_dataset.png", dpi=PLOT_DPI)
        plt.savefig(run_out / "FINAL_PRC_one_curve_per_dataset.svg")

        if SHOW_FINAL_PLOTS:
            plt.show()
        else:
            plt.close(fig1)
            plt.close(fig2)


    # ============================================================
    # RUN ALL
    # ============================================================

    stamp = time.strftime("%Y%m%d_%H%M%S")

    run_out = ensure_dir(
        Path(OUT_ROOT)
        / f"mean_ge_{str(EXPRESSION_CUTOFF).replace('.', 'p')}"
        / f"forceinclude_{int(FORCE_INCLUDE_PERTURBED_GENES)}"
        / f"topgenes_{MAX_GENES_AFTER_FILTER if MAX_GENES_AFTER_FILTER is not None else 'all'}"
        / f"run_{stamp}"
    )

    print("[output]", run_out)
    print("[mode] raw h5ad full-H0 posterior double perturbations")
    print("[expression cutoff]", EXPRESSION_CUTOFF)
    print("[FORCE_INCLUDE_PERTURBED_GENES]", FORCE_INCLUDE_PERTURBED_GENES)
    print("[max genes after filter]", MAX_GENES_AFTER_FILTER)
    print("[max control cells for covariance]", MAX_CONTROL_CELLS_FOR_COV)

    results = []
    errors = []

    for path in tqdm(DATA_PATHS, desc="datasets", dynamic_ncols=True):
        try:
            result = run_one_dataset(path, run_out)
            results.append(result)

        except Exception as e:
            print(f"[ERROR] {path}: {repr(e)}")
            errors.append({
                "path": str(path),
                "error": repr(e),
            })

    # Save all summaries.
    if len(results):
        all_summary = pd.DataFrame([r["summary"] for r in results])
    else:
        all_summary = pd.DataFrame()

    all_summary.to_csv(run_out / "ALL_DATASETS_inverse_summary.csv", index=False)

    with open(run_out / "ALL_DATASETS_errors.json", "w") as f:
        json.dump(errors, f, indent=2)

    # Show exactly two final plots.
    if len(results):
        plot_final_dataset_curves(results, run_out)

    print("\nDONE")
    print("[saved]", run_out / "ALL_DATASETS_inverse_summary.csv")
    print("[saved]", run_out / "ALL_DATASETS_errors.json")
    print("[output root]", run_out)


def plot_roc_prc_side_by_side():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # LOAD SAVED DOUBLE-PERTURBATION INVERSE RESULTS
    # AND MAKE A PUBLICATION-READY SIDE-BY-SIDE ROC + PRC FIGURE
    #
    # Loads:
    #   <RUN_DIR>/<dataset>/inverse_summary.csv
    #   <RUN_DIR>/<dataset>/posterior_roc_prc_curves.npz
    #
    # Saves:
    #   <RUN_DIR>/FINAL_ROC_PRC_side_by_side.svg
    #
    # No model recomputation is performed.
    # ============================================================

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D


    # ============================================================
    # CONFIGURATION
    # ============================================================

    OUT_ROOT = Path(OUTDIR) / "FAST_fullH0_double_pert_inverse_from_h5ad"

    # Set this to a specific run directory to avoid automatic discovery:
    #
    # RUN_DIR = Path(
    #     "FAST_fullH0_double_pert_inverse_from_h5ad/"
    #     "mean_ge_0p01/"
    #     "forceinclude_1/"
    #     "topgenes_1000/"
    #     "run_20260716_123456"
    # )
    #
    # Leave as None to load the newest run automatically.
    RUN_DIR = None

    OUTPUT_FILENAME = "FINAL_ROC_PRC_side_by_side.svg"

    FIGSIZE = (10.2, 4.5)
    LINEWIDTH = 2.4
    REFERENCE_LINEWIDTH = 1.2
    LEGEND_FONTSIZE = 8.5
    AXIS_LABEL_FONTSIZE = 11
    TICK_LABEL_FONTSIZE = 9.5
    PANEL_LABEL_FONTSIZE = 13

    SHOW_FIGURE = True


    # Optional publication-facing dataset names.
    # Any dataset not listed here is automatically cleaned.
    DISPLAY_NAMES = {
        "NormanWeissman2019_filtered": "Norman et al. 2019",
        "TianKampmann2019_day7neuron": "Tian et al. 2019, neuron",
        "TianKampmann2019_iPSC": "Tian et al. 2019, iPSC",
        "proper_filtered": "Proper filtered",
    }

    # Optional explicit ordering. Datasets absent from this list are placed last.
    DATASET_ORDER = [
        "proper_filtered",
        "NormanWeissman2019_filtered",
        "TianKampmann2019_day7neuron",
        "TianKampmann2019_iPSC",
    ]


    # ============================================================
    # MATPLOTLIB SETTINGS
    # ============================================================

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "axes.titlesize": 11,
        "xtick.labelsize": TICK_LABEL_FONTSIZE,
        "ytick.labelsize": TICK_LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,

        # Preserve editable text in the SVG.
        "svg.fonttype": "none",

        # Useful if the figure is later saved as PDF as well.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


    # ============================================================
    # HELPERS
    # ============================================================

    def find_latest_run(out_root):
        """
        Find the newest run_* directory recursively beneath OUT_ROOT.
        """
        out_root = Path(out_root)

        if not out_root.exists():
            raise FileNotFoundError(
                f"Output root does not exist:\n{out_root.resolve()}"
            )

        candidates = [
            path for path in out_root.rglob("run_*")
            if path.is_dir()
        ]

        if not candidates:
            raise FileNotFoundError(
                f"No run_* directories were found beneath:\n{out_root.resolve()}"
            )

        # Timestamped run names sort correctly lexicographically.
        candidates = sorted(
            candidates,
            key=lambda path: (path.name, path.stat().st_mtime),
        )

        return candidates[-1]


    def clean_dataset_name(dataset):
        """
        Convert an internal dataset name into a readable fallback label.
        """
        dataset = str(dataset)

        if dataset in DISPLAY_NAMES:
            return DISPLAY_NAMES[dataset]

        cleaned = dataset
        cleaned = cleaned.replace("_filtered", "")
        cleaned = cleaned.replace("_", " ")
        cleaned = " ".join(cleaned.split())

        return cleaned


    def load_saved_results(run_dir):
        """
        Load every valid dataset result stored directly beneath RUN_DIR.
        """
        run_dir = Path(run_dir)
        results = []

        dataset_dirs = sorted(
            path for path in run_dir.iterdir()
            if path.is_dir()
        )

        for dataset_dir in dataset_dirs:
            summary_path = dataset_dir / "inverse_summary.csv"
            curves_path = dataset_dir / "posterior_roc_prc_curves.npz"

            if not summary_path.exists() or not curves_path.exists():
                continue

            try:
                summary_df = pd.read_csv(summary_path)

                if summary_df.empty:
                    print(f"[skip] Empty summary: {summary_path}")
                    continue

                summary = summary_df.iloc[0].to_dict()

                with np.load(curves_path, allow_pickle=False) as saved:
                    fpr = np.asarray(saved["fpr"], dtype=float)
                    tpr = np.asarray(saved["tpr"], dtype=float)
                    precision = np.asarray(saved["precision"], dtype=float)
                    recall = np.asarray(saved["recall"], dtype=float)

                dataset = str(
                    summary.get("dataset", dataset_dir.name)
                )

                pooled_auc = pd.to_numeric(
                    summary.get("pooled_auc", np.nan),
                    errors="coerce",
                )

                average_precision = pd.to_numeric(
                    summary.get("pooled_average_precision", np.nan),
                    errors="coerce",
                )

                valid_roc = (
                    fpr.size > 1
                    and tpr.size == fpr.size
                    and np.isfinite(fpr).any()
                    and np.isfinite(tpr).any()
                )

                valid_prc = (
                    precision.size > 1
                    and recall.size == precision.size
                    and np.isfinite(precision).any()
                    and np.isfinite(recall).any()
                )

                if not valid_roc and not valid_prc:
                    print(f"[skip] No valid curves: {dataset_dir}")
                    continue

                results.append({
                    "dataset": dataset,
                    "display_name": clean_dataset_name(dataset),
                    "pooled_auc": float(pooled_auc),
                    "average_precision": float(average_precision),
                    "fpr": fpr,
                    "tpr": tpr,
                    "precision": precision,
                    "recall": recall,
                    "valid_roc": valid_roc,
                    "valid_prc": valid_prc,
                })

                print(
                    f"[loaded] {dataset}: "
                    f"AUROC={pooled_auc:.4f}, "
                    f"AP={average_precision:.4f}"
                )

            except Exception as exc:
                print(
                    f"[skip] Failed to load {dataset_dir.name}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if not results:
            raise RuntimeError(
                "No valid saved dataset curves were found in:\n"
                f"{run_dir.resolve()}"
            )

        order_lookup = {
            dataset: index
            for index, dataset in enumerate(DATASET_ORDER)
        }

        results.sort(
            key=lambda result: (
                order_lookup.get(
                    result["dataset"],
                    len(DATASET_ORDER),
                ),
                result["display_name"],
            )
        )

        return results


    def finite_curve(x, y):
        """
        Remove nonfinite curve points while preserving their saved ordering.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        keep = np.isfinite(x) & np.isfinite(y)

        return x[keep], y[keep]


    # ============================================================
    # LOCATE AND LOAD RESULTS
    # ============================================================

    if RUN_DIR is None:
        RUN_DIR = find_latest_run(OUT_ROOT)
    else:
        RUN_DIR = Path(RUN_DIR)

    if not RUN_DIR.exists():
        raise FileNotFoundError(
            f"RUN_DIR does not exist:\n{RUN_DIR.resolve()}"
        )

    print("=" * 80)
    print(f"[run directory] {RUN_DIR.resolve()}")
    print("=" * 80)

    results = load_saved_results(RUN_DIR)

    print(f"[datasets loaded] {len(results)}")


    # ============================================================
    # BUILD PUBLICATION-READY SIDE-BY-SIDE FIGURE
    # ============================================================

    fig, (ax_roc, ax_pr) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=FIGSIZE,
    )

    # Use the same color for a dataset in both panels.
    cmap = plt.get_cmap(
        "tab10" if len(results) <= 10 else "tab20"
    )

    dataset_colors = {
        result["dataset"]: cmap(i % cmap.N)
        for i, result in enumerate(results)
    }


    # ------------------------------------------------------------
    # A. ROC curves
    # ------------------------------------------------------------

    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle=(0, (4, 3)),
        linewidth=REFERENCE_LINEWIDTH,
        color="0.55",
        zorder=1,
    )

    roc_legend_handles = []

    for result in results:
        if not result["valid_roc"]:
            continue

        fpr, tpr = finite_curve(
            result["fpr"],
            result["tpr"],
        )

        color = dataset_colors[result["dataset"]]

        ax_roc.plot(
            fpr,
            tpr,
            linewidth=LINEWIDTH,
            color=color,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )

        if np.isfinite(result["pooled_auc"]):
            label = (
                f"{result['display_name']} "
                f"({result['pooled_auc']:.3f})"
            )
        else:
            label = result["display_name"]

        roc_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=LINEWIDTH,
                label=label,
            )
        )

    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_xlim(-0.015, 1.015)
    ax_roc.set_ylim(-0.015, 1.015)
    ax_roc.set_xticks(np.linspace(0, 1, 6))
    ax_roc.set_yticks(np.linspace(0, 1, 6))
    ax_roc.set_aspect("equal", adjustable="box")

    ax_roc.legend(
        handles=roc_legend_handles,
        title="Dataset (AUROC)",
        title_fontsize=LEGEND_FONTSIZE,
        loc="lower right",
        frameon=False,
        borderaxespad=0.6,
        handlelength=2.3,
        handletextpad=0.7,
        labelspacing=0.45,
    )


    # ------------------------------------------------------------
    # B. Precision-recall curves
    # ------------------------------------------------------------

    pr_legend_handles = []

    for result in results:
        if not result["valid_prc"]:
            continue

        recall, precision = finite_curve(
            result["recall"],
            result["precision"],
        )

        color = dataset_colors[result["dataset"]]

        ax_pr.plot(
            recall,
            precision,
            linewidth=LINEWIDTH,
            color=color,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )

        if np.isfinite(result["average_precision"]):
            label = (
                f"{result['display_name']} "
                f"({result['average_precision']:.3f})"
            )
        else:
            label = result["display_name"]

        pr_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=LINEWIDTH,
                label=label,
            )
        )

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(-0.015, 1.015)
    ax_pr.set_ylim(-0.015, 1.015)
    ax_pr.set_xticks(np.linspace(0, 1, 6))
    ax_pr.set_yticks(np.linspace(0, 1, 6))
    ax_pr.set_aspect("equal", adjustable="box")

    ax_pr.legend(
        handles=pr_legend_handles,
        title="Dataset (average precision)",
        title_fontsize=LEGEND_FONTSIZE,
        loc="upper right",
        frameon=False,
        borderaxespad=0.6,
        handlelength=2.3,
        handletextpad=0.7,
        labelspacing=0.45,
    )


    # ------------------------------------------------------------
    # Shared styling
    # ------------------------------------------------------------

    for ax in (ax_roc, ax_pr):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            top=False,
            right=False,
        )

        ax.grid(False)


    # Panel labels.
    ax_roc.text(
        -0.16,
        1.04,
        "A",
        transform=ax_roc.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    ax_pr.text(
        -0.16,
        1.04,
        "B",
        transform=ax_pr.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.14,
        top=0.965,
        wspace=0.30,
    )


    # ============================================================
    # SAVE
    # ============================================================

    output_path = RUN_DIR / OUTPUT_FILENAME

    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=False,
    )

    print("=" * 80)
    print(f"[saved SVG] {output_path.resolve()}")
    print("=" * 80)

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
