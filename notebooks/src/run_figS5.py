"""Fig S5 -- frequency / percentile normalization comparison for the forward problem.

Compares raw, partial-log, frequency, and cross-normalization schemes across percentile-window gene
filters on the forward (Sigma @ u) prediction. This figure is intentionally ABOUT normalization, so the
analyses are NOT forced to raw counts. Helpers in notebooks/src (not part of the cipher package). Config
constants live inside each entry function; DATA_DIR / SUPPL / OUTDIR are module globals injected by the
notebook via R.__dict__.update. The inline control covariance equals cipher.compute_covariance(ridge_rel=).
"""
import os
import traceback

DATA_DIR = None
SUPPL = None
OUTDIR = None


def run_percentile_raw_partial_freq():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # PATCH: robustly handle var_names_make_unique() failures when
    # adata.var_names is a pandas Categorical (common in some h5ad).
    #
    # Drop-in replacement for analyze_dataset_percentiles()
    # (everything else in your script can stay the same).
    #
    # What changed:
    #   - safe_make_var_names_unique(adata): coerces var_names to plain strings
    #     (Index of dtype object) BEFORE making unique, avoiding:
    #       "Cannot setitem on a Categorical with a new category (...)"
    #   - also strips any accidental pandas categoricals in adata.var columns
    #     that can poison downstream ops.
    # ============================================================

    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.sparse import issparse
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, rankdata

    sns.set(style="whitegrid", context="talk")

    # -----------------------------
    # Keep your existing helpers (or paste if needed)
    # -----------------------------
    def _to_dense(X):
        return X.toarray() if issparse(X) else np.asarray(X)

    def safe_spearman(x, y):
        x = np.asarray(x); y = np.asarray(y)
        if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        xr = rankdata(x)
        yr = rankdata(y)
        r = np.corrcoef(xr, yr)[0, 1]
        if not np.isfinite(r):
            return np.nan
        return float(np.clip(r, -1, 1))

    def compute_covariance(X, ridge=1e-6):
        X = _to_dense(X)
        C = np.cov(X, rowvar=False)
        C = 0.5 * (C + C.T)
        tr = float(np.trace(C))
        if not np.isfinite(tr) or tr <= 0:
            tr = 1.0
        C += ridge * tr / C.shape[0] * np.eye(C.shape[0])
        return C

    def normalization_mats(X):
        X = _to_dense(X).astype(np.float64, copy=False)
        totals = X.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        median_total = np.median(totals)
        if not np.isfinite(median_total) or median_total <= 0:
            median_total = 1.0
        size_factors = totals / median_total
        X_partial = np.log1p(X / size_factors)
        X_freq    = X / totals
        return X, X_partial, X_freq

    def bounded_kde_overlay(df, cols, labels, title, xlabel, outpath_svg):
        plt.figure(figsize=(8, 6))
        any_plotted = False
        for col, lab in zip(cols, labels):
            vals = df[col].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            vals = np.clip(vals, -1, 1)
            sns.kdeplot(vals, label=lab, fill=True, alpha=0.30, lw=2, clip=(-1, 1))
            any_plotted = True
        plt.xlim(-1, 1)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.title(title)
        if any_plotted:
            plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outpath_svg)
        plt.show()

    def run_eval_for_pert(delta_X, Sigma, k, lam=1e-8, mask_mode="finite"):
        sigma = Sigma[:, k]
        denom = float(np.dot(sigma, sigma) + lam)
        alpha = float(np.dot(sigma, delta_X) / denom)
        pred  = alpha * sigma

        if mask_mode == "finite":
            mask = np.isfinite(delta_X) & np.isfinite(pred)
        else:
            mask = np.isfinite(delta_X) & np.isfinite(pred) & (np.abs(delta_X) > 0)

        if mask.sum() < 3:
            return alpha, np.nan, np.nan, np.nan

        obs = delta_X[mask]
        pr  = pred[mask]
        if np.std(obs) < 1e-12 or np.std(pr) < 1e-12:
            return alpha, np.nan, np.nan, np.nan

        ss_res = float(np.sum((obs - pr) ** 2))
        ss_tot = float(np.sum(obs ** 2) + 1e-8)
        r2 = 1.0 - ss_res / ss_tot

        p, _ = pearsonr(obs, pr)
        p = float(np.clip(p, -1, 1))
        s = safe_spearman(obs, pr)
        return alpha, float(r2), p, s

    # -----------------------------
    # NEW: robust var_names unique
    # -----------------------------
    def safe_make_var_names_unique(adata, join="-"):
        """
        Fixes cases where adata.var_names is a pandas Categorical (or other non-object dtype)
        that causes adata.var_names_make_unique() to crash.

        Strategy:
          - force var_names -> plain string Index (dtype object)
          - then call var_names_make_unique()
        """
        # Coerce index to plain python strings (NOT categorical)
        try:
            adata.var_names = pd.Index([str(x) for x in list(adata.var_names)], dtype="object")
        except Exception:
            # ultra-safe fallback
            adata.var_names = pd.Index(pd.Series(adata.var_names).astype(str).tolist(), dtype="object")

        # Also sanitize var columns that might be categorical (optional but helps)
        if hasattr(adata, "var") and isinstance(adata.var, pd.DataFrame):
            for c in adata.var.columns:
                if pd.api.types.is_categorical_dtype(adata.var[c]):
                    adata.var[c] = adata.var[c].astype(str)

        # Now this should not throw the categorical setitem error
        adata.var_names_make_unique(join=join)
        return adata

    # -----------------------------
    # REWRITTEN MAIN FUNCTION
    # -----------------------------
    def analyze_dataset_percentiles(
        data_path,
        out_root = os.path.join(OUTDIR, "raw_partial_freq_percentile_summary"),
        min_percentiles=(0, 10, 20, 30, 40),
        max_percentiles=(100, 90, 80, 70, 60),
        min_cells_per_pert=2,
        ridge=1e-6,
        lam=1e-8,
        unique_join="-",
    ):
        os.makedirs(out_root, exist_ok=True)
        base = os.path.basename(data_path).replace(".h5ad", "")
        print(f"\n==============================\nDATASET: {base}\n==============================")

        adata = sc.read_h5ad(data_path)

        # --- critical fix ---
        safe_make_var_names_unique(adata, join=unique_join)

        if "perturbation" not in adata.obs.columns:
            raise ValueError(f"{base}: missing adata.obs['perturbation']")
        if "control" not in set(adata.obs["perturbation"].unique()):
            raise ValueError(f"{base}: no 'control' label in adata.obs['perturbation']")

        gene_names_full = np.array(adata.var_names.tolist(), dtype=object)
        gene_name_set   = set(gene_names_full)

        # Single-gene perts matching gene names
        all_perts = [p for p in adata.obs["perturbation"].unique() if p != "control"]
        single_gene_perts = [p for p in all_perts if p in gene_name_set]
        if len(single_gene_perts) == 0:
            print(f"[WARN] {base}: no single-gene perturbations matching var_names.")
            return None

        # Control matrix once (full genes)
        ctrl_mask = (adata.obs["perturbation"].to_numpy() == "control")
        X0_full = _to_dense(adata[ctrl_mask, :].X).astype(np.float64, copy=False)
        if X0_full.shape[0] < 2:
            raise ValueError(f"{base}: too few control cells ({X0_full.shape[0]})")

        gene_means_full = X0_full.mean(axis=0)

        # Force-keep: pert genes must survive filtering
        pert_set = set(single_gene_perts)
        force_keep_full = np.array([g in pert_set for g in gene_names_full], dtype=bool)

        # Control normalizations + means (full genes)
        X0_raw_full, X0_part_full, X0_freq_full = normalization_mats(X0_full)
        mu0_raw_full  = X0_raw_full.mean(axis=0)
        mu0_part_full = X0_part_full.mean(axis=0)
        mu0_freq_full = X0_freq_full.mean(axis=0)

        # Precompute pert cell indices for speed
        pert_to_idx = {}
        pert_col = adata.obs["perturbation"].to_numpy()
        for p in single_gene_perts:
            idx = np.where(pert_col == p)[0]
            if idx.size >= min_cells_per_pert:
                pert_to_idx[p] = idx

        if len(pert_to_idx) == 0:
            print(f"[WARN] {base}: no perturbations with >= {min_cells_per_pert} cells.")
            return None

        for min_pct in min_percentiles:
            for max_pct in max_percentiles:
                if max_pct < min_pct:
                    continue

                low_cut  = np.percentile(gene_means_full, min_pct)
                high_cut = np.percentile(gene_means_full, max_pct)

                valid_genes = (gene_means_full >= low_cut) & (gene_means_full <= high_cut)
                keep_mask   = valid_genes | force_keep_full
                kept_idx    = np.where(keep_mask)[0]
                kept_genes  = gene_names_full[kept_idx]

                kept_set = set(kept_genes)
                perts_here = [p for p in pert_to_idx.keys() if p in kept_set]
                if len(perts_here) == 0:
                    print(f"[skip] {base} min{min_pct} max{max_pct}: no perts after filtering")
                    continue

                save_dir = os.path.join(out_root, base, f"min{min_pct:03d}_max{max_pct:03d}")
                os.makedirs(save_dir, exist_ok=True)

                print(f"\n[RUN] {base} | min_pct={min_pct} max_pct={max_pct} | kept_genes={len(kept_idx)} | perts={len(perts_here)}")

                # Slice control matrices/means
                X0_raw  = X0_raw_full[:, kept_idx]
                X0_part = X0_part_full[:, kept_idx]
                X0_freq = X0_freq_full[:, kept_idx]
                mu0_raw  = mu0_raw_full[kept_idx]
                mu0_part = mu0_part_full[kept_idx]
                mu0_freq = mu0_freq_full[kept_idx]

                # Covariances
                Sigma_raw  = compute_covariance(X0_raw,  ridge=ridge)
                Sigma_part = compute_covariance(X0_part, ridge=ridge)
                Sigma_freq = compute_covariance(X0_freq, ridge=ridge)

                gene_to_k = {g:i for i,g in enumerate(kept_genes)}

                rows = []
                for pert in tqdm(perts_here, desc=f"{base} min{min_pct} max{max_pct}", leave=False):
                    idx_cells = pert_to_idx[pert]

                    # Pull pert cells in one go; slice genes
                    X1_full = _to_dense(adata[idx_cells, :].X).astype(np.float64, copy=False)
                    X1_full = X1_full[:, kept_idx]

                    X1_raw, X1_part, X1_freq = normalization_mats(X1_full)

                    delta_raw  = X1_raw.mean(axis=0)  - mu0_raw
                    delta_part = X1_part.mean(axis=0) - mu0_part
                    delta_freq = X1_freq.mean(axis=0) - mu0_freq

                    k = gene_to_k.get(pert, None)
                    if k is None:
                        continue

                    a_raw,  r2_raw,  p_raw,  s_raw  = run_eval_for_pert(delta_raw,  Sigma_raw,  k, lam=lam)
                    a_part, r2_part, p_part, s_part = run_eval_for_pert(delta_part, Sigma_part, k, lam=lam)
                    a_freq, r2_freq, p_freq, s_freq = run_eval_for_pert(delta_freq, Sigma_freq, k, lam=lam)

                    rows.append({
                        "dataset": base,
                        "min_pct": int(min_pct),
                        "max_pct": int(max_pct),
                        "n_genes_kept": int(len(kept_idx)),
                        "perturbation": pert,
                        "n_cells_pert": int(idx_cells.size),

                        "alpha_raw": a_raw,   "R2_raw": r2_raw,   "Pearson_raw": p_raw,   "Spearman_raw": s_raw,
                        "alpha_part": a_part, "R2_part": r2_part, "Pearson_part": p_part, "Spearman_part": s_part,
                        "alpha_freq": a_freq, "R2_freq": r2_freq, "Pearson_freq": p_freq, "Spearman_freq": s_freq,
                    })

                df = pd.DataFrame(rows)
                out_csv = os.path.join(save_dir, f"{base}_metrics_min{min_pct:03d}_max{max_pct:03d}.csv")
                df.to_csv(out_csv, index=False)
                print(f"[SAVE] {out_csv}  (n_rows={len(df)})")

                if len(df) == 0:
                    print("[WARN] empty results (no perts passed filters/min_cells).")
                    continue

                bounded_kde_overlay(
                    df,
                    cols=["R2_raw", "R2_part", "R2_freq"],
                    labels=["Raw", "Partial (log1p/sf)", "Frequency"],
                    title=f"R² across normalizations — {base} (min{min_pct}, max{max_pct})",
                    xlabel="R² (predicted vs observed ΔX)",
                    outpath_svg=os.path.join(save_dir, f"{base}_R2_kde.svg"),
                )

                bounded_kde_overlay(
                    df,
                    cols=["Pearson_raw", "Pearson_part", "Pearson_freq"],
                    labels=["Raw", "Partial (log1p/sf)", "Frequency"],
                    title=f"Pearson across normalizations — {base} (min{min_pct}, max{max_pct})",
                    xlabel="Pearson correlation",
                    outpath_svg=os.path.join(save_dir, f"{base}_Pearson_kde.svg"),
                )

                bounded_kde_overlay(
                    df,
                    cols=["Spearman_raw", "Spearman_part", "Spearman_freq"],
                    labels=["Raw", "Partial (log1p/sf)", "Frequency"],
                    title=f"Spearman across normalizations — {base} (min{min_pct}, max{max_pct})",
                    xlabel="Spearman correlation",
                    outpath_svg=os.path.join(save_dir, f"{base}_Spearman_kde.svg"),
                )

        return True

    # -----------------------------
    # Example batch run (unchanged)
    # -----------------------------x
    datapaths = [os.path.join(DATA_DIR, _n) for _n in [
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]]

    min_percentiles = [0, 10, 20, 30, 40]
    max_percentiles = [100, 90, 80, 70, 60]

    for data_path in datapaths:
        try:
            analyze_dataset_percentiles(
                data_path,
                out_root = os.path.join(OUTDIR, "raw_partial_freq_percentile_summary"),
                min_percentiles=min_percentiles,
                max_percentiles=max_percentiles,
                min_cells_per_pert=2,
                ridge=1e-6,
                lam=1e-8,
                unique_join="-",
            )
        except Exception as e:
            print(f"\n[ERROR] on {data_path}: {e}")
            traceback.print_exc()


def run_percentile_train_test():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # TRAIN/TEST SPLIT VERSION
    #
    #   - Split control cells 50/50 into train/test once per dataset
    #   - Split each perturbation's cells 50/50 into train/test
    #   - Fit on TRAIN only:
    #         * Sigma from train control cells
    #         * mu0 from train control cells
    #         * alpha from train perturbation delta_X
    #   - Evaluate on TEST only:
    #         * observed delta_X from test perturbation cells minus test control mean
    #         * prediction = alpha_train * sigma_train[:, k]
    #
    # Notes:
    #   - Because of the train/test split, perturbations now need enough cells
    #     in both halves. So min_cells_per_pert should generally be >= 4.
    #   - Control cells also need enough samples in both halves.
    #   - The robust var_names fix is preserved.
    # ============================================================

    import os
    import traceback
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.sparse import issparse
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, rankdata

    sns.set(style="whitegrid", context="talk")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _to_dense(X):
        return X.toarray() if issparse(X) else np.asarray(X)

    def safe_spearman(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        xr = rankdata(x)
        yr = rankdata(y)
        r = np.corrcoef(xr, yr)[0, 1]
        if not np.isfinite(r):
            return np.nan
        return float(np.clip(r, -1, 1))

    def compute_covariance(X, ridge=1e-6):
        X = _to_dense(X)
        C = np.cov(X, rowvar=False)
        C = 0.5 * (C + C.T)
        tr = float(np.trace(C))
        if not np.isfinite(tr) or tr <= 0:
            tr = 1.0
        C += ridge * tr / C.shape[0] * np.eye(C.shape[0])
        return C

    def normalization_mats(X):
        X = _to_dense(X).astype(np.float64, copy=False)
        totals = X.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        median_total = np.median(totals)
        if not np.isfinite(median_total) or median_total <= 0:
            median_total = 1.0
        size_factors = totals / median_total
        X_partial = np.log1p(X / size_factors)
        X_freq = X / totals
        return X, X_partial, X_freq

    def bounded_kde_overlay(df, cols, labels, title, xlabel, outpath_svg):
        plt.figure(figsize=(8, 6))
        any_plotted = False
        for col, lab in zip(cols, labels):
            vals = df[col].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            vals = np.clip(vals, -1, 1)
            sns.kdeplot(vals, label=lab, fill=True, alpha=0.30, lw=2, clip=(-1, 1))
            any_plotted = True
        plt.xlim(-1, 1)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.title(title)
        if any_plotted:
            plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outpath_svg)
        plt.show()

    def fit_alpha_train(delta_X_train, Sigma_train, k, lam=1e-8):
        sigma = Sigma_train[:, k]
        denom = float(np.dot(sigma, sigma) + lam)
        alpha = float(np.dot(sigma, delta_X_train) / denom)
        return alpha

    def eval_alpha_on_test(delta_X_test, Sigma_train, k, alpha, mask_mode="finite"):
        sigma = Sigma_train[:, k]
        pred = alpha * sigma

        if mask_mode == "finite":
            mask = np.isfinite(delta_X_test) & np.isfinite(pred)
        else:
            mask = np.isfinite(delta_X_test) & np.isfinite(pred) & (np.abs(delta_X_test) > 0)

        if mask.sum() < 3:
            return np.nan, np.nan, np.nan

        obs = delta_X_test[mask]
        pr = pred[mask]

        if np.std(obs) < 1e-12 or np.std(pr) < 1e-12:
            return np.nan, np.nan, np.nan

        ss_res = float(np.sum((obs - pr) ** 2))
        ss_tot = float(np.sum(obs ** 2) + 1e-8)
        r2 = 1.0 - ss_res / ss_tot

        p, _ = pearsonr(obs, pr)
        p = float(np.clip(p, -1, 1))
        s = safe_spearman(obs, pr)
        return float(r2), p, s

    def split_indices_half(idx, rng):
        idx = np.asarray(idx, dtype=int).copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = n // 2
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        return train_idx, test_idx

    # -----------------------------
    # Robust var_names unique
    # -----------------------------
    def safe_make_var_names_unique(adata, join="-"):
        try:
            adata.var_names = pd.Index([str(x) for x in list(adata.var_names)], dtype="object")
        except Exception:
            adata.var_names = pd.Index(pd.Series(adata.var_names).astype(str).tolist(), dtype="object")

        if hasattr(adata, "var") and isinstance(adata.var, pd.DataFrame):
            for c in adata.var.columns:
                if pd.api.types.is_categorical_dtype(adata.var[c]):
                    adata.var[c] = adata.var[c].astype(str)

        adata.var_names_make_unique(join=join)
        return adata

    # -----------------------------
    # Main function: fit on train, evaluate on test
    # -----------------------------
    def analyze_dataset_percentiles_train_test(
        data_path,
        out_root = os.path.join(OUTDIR, "raw_partial_freq_percentile_summary_train_test"),
        min_percentiles=(0, 10, 20, 30, 40),
        max_percentiles=(100, 90, 80, 70, 60),
        min_cells_per_pert=4,
        min_cells_control=4,
        ridge=1e-6,
        lam=1e-8,
        unique_join="-",
        random_seed=0,
    ):
        os.makedirs(out_root, exist_ok=True)
        base = os.path.basename(data_path).replace(".h5ad", "")
        print(f"\n==============================\nDATASET: {base}\n==============================")

        rng = np.random.default_rng(random_seed)
        adata = sc.read_h5ad(data_path)

        # --- critical fix ---
        safe_make_var_names_unique(adata, join=unique_join)

        if "perturbation" not in adata.obs.columns:
            raise ValueError(f"{base}: missing adata.obs['perturbation']")
        if "control" not in set(adata.obs["perturbation"].unique()):
            raise ValueError(f"{base}: no 'control' label in adata.obs['perturbation']")

        gene_names_full = np.array(adata.var_names.tolist(), dtype=object)
        gene_name_set = set(gene_names_full)

        # Single-gene perts matching gene names
        all_perts = [p for p in adata.obs["perturbation"].unique() if p != "control"]
        single_gene_perts = [p for p in all_perts if p in gene_name_set]
        if len(single_gene_perts) == 0:
            print(f"[WARN] {base}: no single-gene perturbations matching var_names.")
            return None

        # -----------------------------------------
        # Split control cells 50/50 into train/test
        # -----------------------------------------
        pert_col = adata.obs["perturbation"].to_numpy()
        ctrl_idx_all = np.where(pert_col == "control")[0]

        if ctrl_idx_all.size < min_cells_control:
            raise ValueError(f"{base}: too few control cells total ({ctrl_idx_all.size})")

        ctrl_train_idx, ctrl_test_idx = split_indices_half(ctrl_idx_all, rng)
        if len(ctrl_train_idx) < 2 or len(ctrl_test_idx) < 2:
            raise ValueError(
                f"{base}: after 50/50 split, control cells are insufficient "
                f"(train={len(ctrl_train_idx)}, test={len(ctrl_test_idx)})"
            )

        X0_train_full = _to_dense(adata[ctrl_train_idx, :].X).astype(np.float64, copy=False)
        X0_test_full  = _to_dense(adata[ctrl_test_idx,  :].X).astype(np.float64, copy=False)

        gene_means_full = X0_train_full.mean(axis=0)

        # Force-keep: pert genes must survive filtering
        pert_set = set(single_gene_perts)
        force_keep_full = np.array([g in pert_set for g in gene_names_full], dtype=bool)

        # Control normalizations + means (train and test)
        X0_train_raw_full, X0_train_part_full, X0_train_freq_full = normalization_mats(X0_train_full)
        X0_test_raw_full,  X0_test_part_full,  X0_test_freq_full  = normalization_mats(X0_test_full)

        mu0_train_raw_full  = X0_train_raw_full.mean(axis=0)
        mu0_train_part_full = X0_train_part_full.mean(axis=0)
        mu0_train_freq_full = X0_train_freq_full.mean(axis=0)

        mu0_test_raw_full   = X0_test_raw_full.mean(axis=0)
        mu0_test_part_full  = X0_test_part_full.mean(axis=0)
        mu0_test_freq_full  = X0_test_freq_full.mean(axis=0)

        # -----------------------------------------
        # Precompute 50/50 split indices per perturbation
        # -----------------------------------------
        pert_to_split = {}
        for p in single_gene_perts:
            idx = np.where(pert_col == p)[0]
            if idx.size < min_cells_per_pert:
                continue

            train_idx, test_idx = split_indices_half(idx, rng)
            if len(train_idx) < 1 or len(test_idx) < 1:
                continue

            pert_to_split[p] = {
                "train_idx": train_idx,
                "test_idx": test_idx,
                "n_total": int(idx.size),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }

        if len(pert_to_split) == 0:
            print(f"[WARN] {base}: no perturbations with >= {min_cells_per_pert} cells and nonempty train/test split.")
            return None

        # -----------------------------------------
        # Sweep percentile filters
        # -----------------------------------------
        for min_pct in min_percentiles:
            for max_pct in max_percentiles:
                if max_pct < min_pct:
                    continue

                low_cut = np.percentile(gene_means_full, min_pct)
                high_cut = np.percentile(gene_means_full, max_pct)

                valid_genes = (gene_means_full >= low_cut) & (gene_means_full <= high_cut)
                keep_mask = valid_genes | force_keep_full
                kept_idx = np.where(keep_mask)[0]
                kept_genes = gene_names_full[kept_idx]

                kept_set = set(kept_genes)
                perts_here = [p for p in pert_to_split.keys() if p in kept_set]
                if len(perts_here) == 0:
                    print(f"[skip] {base} min{min_pct} max{max_pct}: no perts after filtering")
                    continue

                save_dir = os.path.join(out_root, base, f"min{min_pct:03d}_max{max_pct:03d}")
                os.makedirs(save_dir, exist_ok=True)

                print(
                    f"\n[RUN] {base} | min_pct={min_pct} max_pct={max_pct} | "
                    f"kept_genes={len(kept_idx)} | perts={len(perts_here)}"
                )

                # -----------------------------------------
                # Slice train/test control matrices/means
                # -----------------------------------------
                X0_train_raw  = X0_train_raw_full[:, kept_idx]
                X0_train_part = X0_train_part_full[:, kept_idx]
                X0_train_freq = X0_train_freq_full[:, kept_idx]

                X0_test_raw   = X0_test_raw_full[:, kept_idx]
                X0_test_part  = X0_test_part_full[:, kept_idx]
                X0_test_freq  = X0_test_freq_full[:, kept_idx]

                mu0_train_raw  = mu0_train_raw_full[kept_idx]
                mu0_train_part = mu0_train_part_full[kept_idx]
                mu0_train_freq = mu0_train_freq_full[kept_idx]

                mu0_test_raw   = mu0_test_raw_full[kept_idx]
                mu0_test_part  = mu0_test_part_full[kept_idx]
                mu0_test_freq  = mu0_test_freq_full[kept_idx]

                # Fit Sigma on TRAIN control only
                Sigma_train_raw  = compute_covariance(X0_train_raw, ridge=ridge)
                Sigma_train_part = compute_covariance(X0_train_part, ridge=ridge)
                Sigma_train_freq = compute_covariance(X0_train_freq, ridge=ridge)

                gene_to_k = {g: i for i, g in enumerate(kept_genes)}

                rows = []
                for pert in tqdm(perts_here, desc=f"{base} min{min_pct} max{max_pct}", leave=False):
                    split_info = pert_to_split[pert]
                    idx_train = split_info["train_idx"]
                    idx_test = split_info["test_idx"]

                    # Pull TRAIN perturbation cells
                    X1_train_full = _to_dense(adata[idx_train, :].X).astype(np.float64, copy=False)
                    X1_train_full = X1_train_full[:, kept_idx]

                    # Pull TEST perturbation cells
                    X1_test_full = _to_dense(adata[idx_test, :].X).astype(np.float64, copy=False)
                    X1_test_full = X1_test_full[:, kept_idx]

                    X1_train_raw, X1_train_part, X1_train_freq = normalization_mats(X1_train_full)
                    X1_test_raw,  X1_test_part,  X1_test_freq  = normalization_mats(X1_test_full)

                    # TRAIN deltas used for fitting alpha
                    delta_train_raw  = X1_train_raw.mean(axis=0)  - mu0_train_raw
                    delta_train_part = X1_train_part.mean(axis=0) - mu0_train_part
                    delta_train_freq = X1_train_freq.mean(axis=0) - mu0_train_freq

                    # TEST deltas used for evaluation
                    delta_test_raw   = X1_test_raw.mean(axis=0)   - mu0_test_raw
                    delta_test_part  = X1_test_part.mean(axis=0)  - mu0_test_part
                    delta_test_freq  = X1_test_freq.mean(axis=0)  - mu0_test_freq

                    k = gene_to_k.get(pert, None)
                    if k is None:
                        continue

                    # Fit alpha on TRAIN
                    a_raw  = fit_alpha_train(delta_train_raw,  Sigma_train_raw,  k, lam=lam)
                    a_part = fit_alpha_train(delta_train_part, Sigma_train_part, k, lam=lam)
                    a_freq = fit_alpha_train(delta_train_freq, Sigma_train_freq, k, lam=lam)

                    # Evaluate on TEST
                    r2_raw,  p_raw,  s_raw  = eval_alpha_on_test(delta_test_raw,  Sigma_train_raw,  k, a_raw)
                    r2_part, p_part, s_part = eval_alpha_on_test(delta_test_part, Sigma_train_part, k, a_part)
                    r2_freq, p_freq, s_freq = eval_alpha_on_test(delta_test_freq, Sigma_train_freq, k, a_freq)

                    rows.append({
                        "dataset": base,
                        "min_pct": int(min_pct),
                        "max_pct": int(max_pct),
                        "n_genes_kept": int(len(kept_idx)),
                        "perturbation": pert,

                        "n_cells_control_train": int(len(ctrl_train_idx)),
                        "n_cells_control_test": int(len(ctrl_test_idx)),
                        "n_cells_pert_total": int(split_info["n_total"]),
                        "n_cells_pert_train": int(split_info["n_train"]),
                        "n_cells_pert_test": int(split_info["n_test"]),

                        "alpha_raw_train": a_raw,
                        "R2_raw_test": r2_raw,
                        "Pearson_raw_test": p_raw,
                        "Spearman_raw_test": s_raw,

                        "alpha_part_train": a_part,
                        "R2_part_test": r2_part,
                        "Pearson_part_test": p_part,
                        "Spearman_part_test": s_part,

                        "alpha_freq_train": a_freq,
                        "R2_freq_test": r2_freq,
                        "Pearson_freq_test": p_freq,
                        "Spearman_freq_test": s_freq,
                    })

                df = pd.DataFrame(rows)
                out_csv = os.path.join(
                    save_dir,
                    f"{base}_metrics_train_test_min{min_pct:03d}_max{max_pct:03d}.csv"
                )
                df.to_csv(out_csv, index=False)
                print(f"[SAVE] {out_csv}  (n_rows={len(df)})")

                if len(df) == 0:
                    print("[WARN] empty results (no perts passed filters/min_cells).")
                    continue

                bounded_kde_overlay(
                    df,
                    cols=["R2_raw_test", "R2_part_test", "R2_freq_test"],
                    labels=["Raw", "Partial (log1p/sf)", "Frequency"],
                    title=f"TEST R² across normalizations — {base} (min{min_pct}, max{max_pct})",
                    xlabel="R² on held-out test ΔX",
                    outpath_svg=os.path.join(save_dir, f"{base}_R2_test_kde.svg"),
                )

                bounded_kde_overlay(
                    df,
                    cols=["Pearson_raw_test", "Pearson_part_test", "Pearson_freq_test"],
                    labels=["Raw", "Partial (log1p/sf)", "Frequency"],
                    title=f"TEST Pearson across normalizations — {base} (min{min_pct}, max{max_pct})",
                    xlabel="Pearson correlation on held-out test ΔX",
                    outpath_svg=os.path.join(save_dir, f"{base}_Pearson_test_kde.svg"),
                )

                bounded_kde_overlay(
                    df,
                    cols=["Spearman_raw_test", "Spearman_part_test", "Spearman_freq_test"],
                    labels=["Raw", "Partial (log1p/sf)", "Frequency"],
                    title=f"TEST Spearman across normalizations — {base} (min{min_pct}, max{max_pct})",
                    xlabel="Spearman correlation on held-out test ΔX",
                    outpath_svg=os.path.join(save_dir, f"{base}_Spearman_test_kde.svg"),
                )

        return True


    # -----------------------------
    # Example batch run
    # -----------------------------
    datapaths = [os.path.join(DATA_DIR, _n) for _n in [
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]]

    min_percentiles = [0]
    max_percentiles = [100]

    for data_path in datapaths:
        try:
            analyze_dataset_percentiles_train_test(
                data_path,
                out_root = os.path.join(OUTDIR, "raw_partial_freq_percentile_summary_train_test"),
                min_percentiles=min_percentiles,
                max_percentiles=max_percentiles,
                min_cells_per_pert=4,
                min_cells_control=4,
                ridge=1e-6,
                lam=1e-8,
                unique_join="-",
                random_seed=0,
            )
        except Exception as e:
            print(f"\n[ERROR] on {data_path}: {e}")
            traceback.print_exc()


def run_raw_only_train_test():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # TRAIN/TEST SPLIT VERSION — RAW COUNTS ONLY
    #
    # Flexible perturbation train/test split fractions
    #
    # Desired behavior:
    #   - Only raw expression counts are used
    #   - Use ALL control cells once per dataset:
    #         * Sigma from all control raw counts
    #         * mu0 from all control raw counts
    #   - Split each perturbation's cells according to train_fraction
    #         * e.g. train_fraction=0.8 -> 80/20 split
    #         * e.g. train_fraction=0.6 -> 60/40 split
    #   - Fit on TRAIN perturbation cells only:
    #         * delta_X_train = mean(train pert cells) - mu0_all
    #         * alpha from delta_X_train using Sigma_all
    #   - Evaluate on TEST perturbation cells only:
    #         * delta_X_test = mean(test pert cells) - mu0_all
    #         * prediction = alpha_train * Sigma_all[:, k]
    # ============================================================

    import os
    import math
    import traceback
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.sparse import issparse
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, rankdata

    sns.set(style="whitegrid", context="talk")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _to_dense(X):
        return X.toarray() if issparse(X) else np.asarray(X)

    def safe_spearman(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        xr = rankdata(x)
        yr = rankdata(y)
        r = np.corrcoef(xr, yr)[0, 1]
        if not np.isfinite(r):
            return np.nan
        return float(np.clip(r, -1, 1))

    def compute_covariance(X, ridge=1e-6):
        X = _to_dense(X).astype(np.float64, copy=False)
        C = np.cov(X, rowvar=False)
        tr = float(np.trace(C))
        if not np.isfinite(tr) or tr <= 0:
            tr = 1.0
        C += ridge * tr / C.shape[0] * np.eye(C.shape[0])
        return C

    def bounded_kde_overlay(df, cols, labels, title, xlabel, outpath_svg):
        plt.figure(figsize=(8, 6))
        any_plotted = False

        for col, lab in zip(cols, labels):
            vals = df[col].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            vals = np.clip(vals, -1, 1)

            # KDE
            sns.kdeplot(vals, label=lab, fill=True, alpha=0.30, lw=2, clip=(-1, 1))
            any_plotted = True

            # ---- MEAN LINE ----
            mean_val = float(np.mean(vals))
            plt.axvline(
                mean_val,
                linestyle="--",
                linewidth=2,
                label=f"{lab} mean = {mean_val:.3f}"
            )

        plt.xlim(-1, 1)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.title(title)

        if any_plotted:
            plt.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(outpath_svg)
        plt.show()

    def fit_alpha_train(delta_X_train, Sigma_all, k, lam=1e-8):
        sigma_k = Sigma_all[:, k]
        denom = float(np.dot(sigma_k, sigma_k) + lam)
        alpha = float(np.dot(sigma_k, delta_X_train) / denom)
        return alpha

    def eval_alpha_on_test(delta_X_test, Sigma_all, k, alpha, mask_mode="finite"):
        sigma_k = Sigma_all[:, k]
        pred = alpha * sigma_k

        if mask_mode == "finite":
            mask = np.isfinite(delta_X_test) & np.isfinite(pred)
        else:
            mask = np.isfinite(delta_X_test) & np.isfinite(pred) & (np.abs(delta_X_test) > 0)

        if mask.sum() < 3:
            return np.nan, np.nan, np.nan

        obs = delta_X_test[mask]
        pr = pred[mask]

        if np.std(obs) < 1e-12 or np.std(pr) < 1e-12:
            return np.nan, np.nan, np.nan

        ss_res = float(np.sum((obs - pr) ** 2))
        ss_tot = float(np.sum(obs ** 2) + 1e-8)
        r2 = 1.0 - ss_res / ss_tot

        p, _ = pearsonr(obs, pr)
        p = float(np.clip(p, -1, 1))
        s = safe_spearman(obs, pr)
        return float(r2), p, s

    def split_indices_fraction(idx, rng, train_fraction=0.5, min_train=1, min_test=1):
        """
        Flexible random split for one perturbation.

        train_fraction:
            fraction of cells assigned to train

        Guarantees:
            - integer split
            - at least min_train in train if possible
            - at least min_test in test if possible
        """
        idx = np.asarray(idx, dtype=int).copy()
        n = len(idx)

        if n < (min_train + min_test):
            return None, None

        rng.shuffle(idx)

        n_train = int(round(train_fraction * n))
        n_train = max(min_train, n_train)
        n_train = min(n_train, n - min_test)

        if n_train < min_train or (n - n_train) < min_test:
            return None, None

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        return train_idx, test_idx

    # -----------------------------
    # Robust var_names unique
    # -----------------------------
    def safe_make_var_names_unique(adata, join="-"):
        try:
            adata.var_names = pd.Index([str(x) for x in list(adata.var_names)], dtype="object")
        except Exception:
            adata.var_names = pd.Index(pd.Series(adata.var_names).astype(str).tolist(), dtype="object")

        if hasattr(adata, "var") and isinstance(adata.var, pd.DataFrame):
            for c in adata.var.columns:
                if pd.api.types.is_categorical_dtype(adata.var[c]):
                    adata.var[c] = adata.var[c].astype(str)

        adata.var_names_make_unique(join=join)
        return adata

    # -----------------------------
    # Main function
    # -----------------------------
    def analyze_dataset_percentiles_train_test_raw_only(
        data_path,
        out_root = os.path.join(OUTDIR, "raw_counts_only_percentile_summary_train_test"),
        min_percentiles=(0, 10, 20, 30, 40),
        max_percentiles=(100, 90, 80, 70, 60),
        min_cells_per_pert=4,
        min_cells_control=4,
        train_fraction=0.5,          # <-- flexible split
        min_train_cells=1,           # <-- enforce minimum train cells per perturbation
        min_test_cells=1,            # <-- enforce minimum test cells per perturbation
        ridge=1e-6,
        lam=1e-8,
        unique_join="-",
        random_seed=0,
    ):
        if not (0 < train_fraction < 1):
            raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

        os.makedirs(out_root, exist_ok=True)
        base = os.path.basename(data_path).replace(".h5ad", "")
        print(f"\n==============================\nDATASET: {base}\n==============================")
        print(f"Using train_fraction = {train_fraction:.3f}  (test_fraction = {1.0 - train_fraction:.3f})")

        rng = np.random.default_rng(random_seed)
        adata = sc.read_h5ad(data_path)

        safe_make_var_names_unique(adata, join=unique_join)

        if "perturbation" not in adata.obs.columns:
            raise ValueError(f"{base}: missing adata.obs['perturbation']")
        if "control" not in set(adata.obs["perturbation"].unique()):
            raise ValueError(f"{base}: no 'control' label in adata.obs['perturbation']")

        gene_names_full = np.array(adata.var_names.tolist(), dtype=object)
        gene_name_set = set(gene_names_full)

        all_perts = [p for p in adata.obs["perturbation"].unique() if p != "control"]
        single_gene_perts = [p for p in all_perts if p in gene_name_set]
        if len(single_gene_perts) == 0:
            print(f"[WARN] {base}: no single-gene perturbations matching var_names.")
            return None

        pert_col = adata.obs["perturbation"].to_numpy()

        # -----------------------------------------
        # Use ALL control cells for Sigma and mu0
        # -----------------------------------------
        ctrl_idx_all = np.where(pert_col == "control")[0]
        if ctrl_idx_all.size < min_cells_control:
            raise ValueError(f"{base}: too few control cells total ({ctrl_idx_all.size})")

        X0_all_full = _to_dense(adata[ctrl_idx_all, :].X).astype(np.float64, copy=False)
        mu0_all_full = X0_all_full.mean(axis=0)
        gene_means_full = mu0_all_full.copy()

        # Force-keep perturbation genes
        pert_set = set(single_gene_perts)
        force_keep_full = np.array([g in pert_set for g in gene_names_full], dtype=bool)

        # -----------------------------------------
        # Precompute flexible split per perturbation
        # -----------------------------------------
        pert_to_split = {}
        min_required = max(min_cells_per_pert, min_train_cells + min_test_cells)

        for p in single_gene_perts:
            idx = np.where(pert_col == p)[0]
            if idx.size < min_required:
                continue

            train_idx, test_idx = split_indices_fraction(
                idx,
                rng,
                train_fraction=train_fraction,
                min_train=min_train_cells,
                min_test=min_test_cells,
            )
            if train_idx is None or test_idx is None:
                continue

            pert_to_split[p] = {
                "train_idx": train_idx,
                "test_idx": test_idx,
                "n_total": int(idx.size),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }

        if len(pert_to_split) == 0:
            print(
                f"[WARN] {base}: no perturbations passed split constraints "
                f"(min_cells_per_pert={min_cells_per_pert}, "
                f"min_train_cells={min_train_cells}, min_test_cells={min_test_cells})."
            )
            return None

        # -----------------------------------------
        # Sweep percentile filters
        # -----------------------------------------
        for min_pct in min_percentiles:
            for max_pct in max_percentiles:
                if max_pct < min_pct:
                    continue

                low_cut = np.percentile(gene_means_full, min_pct)
                high_cut = np.percentile(gene_means_full, max_pct)

                valid_genes = (gene_means_full >= low_cut) & (gene_means_full <= high_cut)
                keep_mask = valid_genes | force_keep_full
                kept_idx = np.where(keep_mask)[0]
                kept_genes = gene_names_full[kept_idx]

                kept_set = set(kept_genes)
                perts_here = [p for p in pert_to_split.keys() if p in kept_set]
                if len(perts_here) == 0:
                    print(f"[skip] {base} min{min_pct} max{max_pct}: no perts after filtering")
                    continue

                split_label = f"train{int(round(100 * train_fraction)):02d}_test{int(round(100 * (1-train_fraction))):02d}"
                save_dir = os.path.join(
                    out_root,
                    base,
                    split_label,
                    f"min{min_pct:03d}_max{max_pct:03d}",
                )
                os.makedirs(save_dir, exist_ok=True)

                print(
                    f"\n[RUN] {base} | split={split_label} | "
                    f"min_pct={min_pct} max_pct={max_pct} | "
                    f"kept_genes={len(kept_idx)} | perts={len(perts_here)}"
                )

                X0_all = X0_all_full[:, kept_idx]
                mu0_all = mu0_all_full[kept_idx]
                Sigma_all = compute_covariance(X0_all, ridge=ridge)

                gene_to_k = {g: i for i, g in enumerate(kept_genes)}

                rows = []
                for pert in tqdm(perts_here, desc=f"{base} {split_label} min{min_pct} max{max_pct}", leave=False):
                    split_info = pert_to_split[pert]
                    idx_train = split_info["train_idx"]
                    idx_test = split_info["test_idx"]

                    X1_train = _to_dense(adata[idx_train, :].X).astype(np.float64, copy=False)
                    X1_train = X1_train[:, kept_idx]

                    X1_test = _to_dense(adata[idx_test, :].X).astype(np.float64, copy=False)
                    X1_test = X1_test[:, kept_idx]

                    delta_train = X1_train.mean(axis=0) - mu0_all
                    delta_test = X1_test.mean(axis=0) - mu0_all

                    k = gene_to_k.get(pert, None)
                    if k is None:
                        continue

                    alpha_train = fit_alpha_train(delta_train, Sigma_all, k, lam=lam)
                    r2_test, pearson_test, spearman_test = eval_alpha_on_test(
                        delta_test, Sigma_all, k, alpha_train
                    )

                    rows.append({
                        "dataset": base,
                        "train_fraction": float(train_fraction),
                        "test_fraction": float(1.0 - train_fraction),
                        "min_pct": int(min_pct),
                        "max_pct": int(max_pct),
                        "n_genes_kept": int(len(kept_idx)),
                        "perturbation": pert,

                        "n_cells_control_all": int(len(ctrl_idx_all)),
                        "n_cells_pert_total": int(split_info["n_total"]),
                        "n_cells_pert_train": int(split_info["n_train"]),
                        "n_cells_pert_test": int(split_info["n_test"]),

                        "alpha_train": alpha_train,
                        "R2_test": r2_test,
                        "Pearson_test": pearson_test,
                        "Spearman_test": spearman_test,
                    })

                df = pd.DataFrame(rows)
                out_csv = os.path.join(
                    save_dir,
                    f"{base}_{split_label}_raw_only_metrics_train_test_min{min_pct:03d}_max{max_pct:03d}.csv"
                )
                df.to_csv(out_csv, index=False)
                print(f"[SAVE] {out_csv}  (n_rows={len(df)})")

                if len(df) == 0:
                    print("[WARN] empty results.")
                    continue

                bounded_kde_overlay(
                    df,
                    cols=["R2_test"],
                    labels=["Raw counts"],
                    title=f"TEST R² — {base} — {split_label} (min{min_pct}, max{max_pct})",
                    xlabel="R² on held-out test ΔX",
                    outpath_svg=os.path.join(save_dir, f"{base}_{split_label}_R2_test_kde.svg"),
                )

                bounded_kde_overlay(
                    df,
                    cols=["Pearson_test"],
                    labels=["Raw counts"],
                    title=f"TEST Pearson — {base} — {split_label} (min{min_pct}, max{max_pct})",
                    xlabel="Pearson correlation on held-out test ΔX",
                    outpath_svg=os.path.join(save_dir, f"{base}_{split_label}_Pearson_test_kde.svg"),
                )

                bounded_kde_overlay(
                    df,
                    cols=["Spearman_test"],
                    labels=["Raw counts"],
                    title=f"TEST Spearman — {base} — {split_label} (min{min_pct}, max{max_pct})",
                    xlabel="Spearman correlation on held-out test ΔX",
                    outpath_svg=os.path.join(save_dir, f"{base}_{split_label}_Spearman_test_kde.svg"),
                )

        return True

    # -----------------------------
    # Example batch run
    # -----------------------------
    datapaths = [os.path.join(DATA_DIR, _n) for _n in [
        "ReplogleWeissman2022_rpe1.h5ad",
        # "ReplogleWeissman2022_K562_essential.h5ad",
        # "GSE264667_jurkat_raw_singlecell_01.h5ad",
        # "GSE264667_hepg2_raw_singlecell_01.h5ad",
        # "NormanWeissman2019_filtered.h5ad",
        # "FrangiehIzar2021_RNA.h5ad",
        # "TianKampmann2019_day7neuron.h5ad",
        # "TianKampmann2021_CRISPRi.h5ad",
        # "TianKampmann2021_CRISPRa.h5ad",
        # "TianKampmann2019_iPSC.h5ad",
    ]]

    min_percentiles = [10]
    max_percentiles = [90]

    # Example 1: 80/20 split
    for data_path in datapaths:
        try:
            analyze_dataset_percentiles_train_test_raw_only(
                data_path,
                out_root = os.path.join(OUTDIR, "raw_counts_only_percentile_summary_train_test"),
                min_percentiles=min_percentiles,
                max_percentiles=max_percentiles,
                min_cells_per_pert=200,
                min_cells_control=200,
                train_fraction=0.6,
                min_train_cells=5,
                min_test_cells=5,
                ridge=0.,
                lam=1.,
                unique_join="-",
                random_seed=0,
            )
        except Exception as e:
            print(f"\n[ERROR] on {data_path}: {e}")
            traceback.print_exc()

    # Example 2: run several split fractions
    # split_fractions = [0.5, 0.6, 0.7, 0.8, 0.9]
    # for frac in split_fractions:
    #     for data_path in datapaths:
    #         try:
    #             analyze_dataset_percentiles_train_test_raw_only(
    #                 data_path,
    #                 out_root = os.path.join(OUTDIR, "raw_counts_only_percentile_summary_train_test"),
    #                 min_percentiles=min_percentiles,
    #                 max_percentiles=max_percentiles,
    #                 min_cells_per_pert=50,
    #                 min_cells_control=50,
    #                 train_fraction=frac,
    #                 min_train_cells=5,
    #                 min_test_cells=5,
    #                 ridge=1e-6,
    #                 lam=1e-10,
    #                 unique_join="-",
    #                 random_seed=0,
    #             )
    #         except Exception as e:
    #             print(f"\n[ERROR] on {data_path} with frac={frac}: {e}")
    #             traceback.print_exc()


def run_raw_only_all_cells():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # RAW COUNTS ONLY — NO TRAIN/TEST SPLIT
    #
    # Desired behavior:
    #   - Only raw expression counts are used
    #   - Use ALL control cells once per dataset:
    #         * Sigma from all control raw counts
    #         * mu0 from all control raw counts
    #   - Use ALL perturbation cells for both fitting and evaluation
    #   - For each single-gene perturbation k:
    #         * delta_X = mean(all pert cells) - mu0_all
    #         * alpha = argmin ||delta_X - alpha * Sigma[:, k]||^2
    #         * prediction = alpha * Sigma[:, k]
    #         * evaluate on the same delta_X
    #   - KDE plots include dashed vertical mean lines
    # ============================================================

    import os
    import traceback
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.sparse import issparse
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, rankdata

    sns.set(style="whitegrid", context="talk")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _to_dense(X):
        return X.toarray() if issparse(X) else np.asarray(X)

    def safe_spearman(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        xr = rankdata(x)
        yr = rankdata(y)
        r = np.corrcoef(xr, yr)[0, 1]
        if not np.isfinite(r):
            return np.nan
        return float(np.clip(r, -1, 1))

    def compute_covariance(X, ridge=1e-6):
        X = _to_dense(X).astype(np.float64, copy=False)
        C = np.cov(X, rowvar=False)
        tr = float(np.trace(C))
        if not np.isfinite(tr) or tr <= 0:
            tr = 1.0
        C += ridge * tr / C.shape[0] * np.eye(C.shape[0])
        return C

    def bounded_kde_overlay(df, cols, labels, title, xlabel, outpath_svg):
        plt.figure(figsize=(8, 6))
        any_plotted = False

        for col, lab in zip(cols, labels):
            vals = df[col].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            vals = np.clip(vals, -1, 1)

            sns.kdeplot(vals, fill=True, alpha=0.30, lw=2, clip=(-1, 1))
            mean_val = float(np.mean(vals))
            plt.axvline(
                mean_val,
                linestyle="--",
                linewidth=2,
                label=f"{lab} (mean={mean_val:.3f})"
            )
            print(f"{col} mean = {mean_val:.4f}")
            any_plotted = True

        plt.xlim(-1, 1)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.title(title)
        if any_plotted:
            plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outpath_svg)
        plt.show()

    def fit_alpha(delta_X, Sigma, k, lam=1e-8):
        sigma_k = Sigma[:, k]
        denom = float(np.dot(sigma_k, sigma_k) + lam)
        alpha = float(np.dot(sigma_k, delta_X) / denom)
        return alpha

    def evaluate_delta(delta_X, Sigma, k, alpha, mask_mode="finite"):
        sigma_k = Sigma[:, k]
        pred = alpha * sigma_k

        if mask_mode == "finite":
            mask = np.isfinite(delta_X) & np.isfinite(pred)
        else:
            mask = np.isfinite(delta_X) & np.isfinite(pred) & (np.abs(delta_X) > 0)

        if mask.sum() < 3:
            return np.nan, np.nan, np.nan

        obs = delta_X[mask]
        pr = pred[mask]

        if np.std(obs) < 1e-12 or np.std(pr) < 1e-12:
            return np.nan, np.nan, np.nan

        ss_res = float(np.sum((obs - pr) ** 2))
        ss_tot = float(np.sum(obs ** 2) + 1e-8)
        r2 = 1.0 - ss_res / ss_tot

        p, _ = pearsonr(obs, pr)
        p = float(np.clip(p, -1, 1))
        s = safe_spearman(obs, pr)
        return float(r2), p, s

    # -----------------------------
    # Robust var_names unique
    # -----------------------------
    def safe_make_var_names_unique(adata, join="-"):
        try:
            adata.var_names = pd.Index([str(x) for x in list(adata.var_names)], dtype="object")
        except Exception:
            adata.var_names = pd.Index(pd.Series(adata.var_names).astype(str).tolist(), dtype="object")

        if hasattr(adata, "var") and isinstance(adata.var, pd.DataFrame):
            for c in adata.var.columns:
                if pd.api.types.is_categorical_dtype(adata.var[c]):
                    adata.var[c] = adata.var[c].astype(str)

        adata.var_names_make_unique(join=join)
        return adata

    # -----------------------------
    # Main function
    # -----------------------------
    def analyze_dataset_percentiles_raw_only_all_cells(
        data_path,
        out_root = os.path.join(OUTDIR, "raw_counts_only_percentile_summary_all_cells"),
        min_percentiles=(0, 10, 20, 30, 40),
        max_percentiles=(100, 90, 80, 70, 60),
        min_cells_per_pert=4,
        min_cells_control=4,
        ridge=1e-6,
        lam=1e-8,
        unique_join="-",
    ):
        os.makedirs(out_root, exist_ok=True)
        base = os.path.basename(data_path).replace(".h5ad", "")
        print(f"\n==============================\nDATASET: {base}\n==============================")

        adata = sc.read_h5ad(data_path)
        safe_make_var_names_unique(adata, join=unique_join)

        if "perturbation" not in adata.obs.columns:
            raise ValueError(f"{base}: missing adata.obs['perturbation']")
        if "control" not in set(adata.obs["perturbation"].unique()):
            raise ValueError(f"{base}: no 'control' label in adata.obs['perturbation']")

        gene_names_full = np.array(adata.var_names.tolist(), dtype=object)
        gene_name_set = set(gene_names_full)
        pert_col = adata.obs["perturbation"].to_numpy()

        # Single-gene perturbations matching var_names
        all_perts = [p for p in adata.obs["perturbation"].unique() if p != "control"]
        single_gene_perts = [p for p in all_perts if p in gene_name_set]
        if len(single_gene_perts) == 0:
            print(f"[WARN] {base}: no single-gene perturbations matching var_names.")
            return None

        # All control cells
        ctrl_idx_all = np.where(pert_col == "control")[0]
        if ctrl_idx_all.size < min_cells_control:
            raise ValueError(f"{base}: too few control cells total ({ctrl_idx_all.size})")

        X0_all_full = _to_dense(adata[ctrl_idx_all, :].X).astype(np.float64, copy=False)
        mu0_all_full = X0_all_full.mean(axis=0)
        gene_means_full = mu0_all_full.copy()

        # Force-keep perturbation genes
        pert_set = set(single_gene_perts)
        force_keep_full = np.array([g in pert_set for g in gene_names_full], dtype=bool)

        # Precompute perturbation indices
        pert_to_idx = {}
        for p in single_gene_perts:
            idx = np.where(pert_col == p)[0]
            if idx.size < min_cells_per_pert:
                continue
            pert_to_idx[p] = idx

        if len(pert_to_idx) == 0:
            print(f"[WARN] {base}: no perturbations with >= {min_cells_per_pert} cells.")
            return None

        # Sweep percentile filters
        for min_pct in min_percentiles:
            for max_pct in max_percentiles:
                if max_pct < min_pct:
                    continue

                low_cut = np.percentile(gene_means_full, min_pct)
                high_cut = np.percentile(gene_means_full, max_pct)

                valid_genes = (gene_means_full >= low_cut) & (gene_means_full <= high_cut)
                keep_mask = valid_genes | force_keep_full
                kept_idx = np.where(keep_mask)[0]
                kept_genes = gene_names_full[kept_idx]

                kept_set = set(kept_genes)
                perts_here = [p for p in pert_to_idx.keys() if p in kept_set]
                if len(perts_here) == 0:
                    print(f"[skip] {base} min{min_pct} max{max_pct}: no perts after filtering")
                    continue

                save_dir = os.path.join(out_root, base, f"min{min_pct:03d}_max{max_pct:03d}")
                os.makedirs(save_dir, exist_ok=True)

                print(
                    f"\n[RUN] {base} | min_pct={min_pct} max_pct={max_pct} | "
                    f"kept_genes={len(kept_idx)} | perts={len(perts_here)}"
                )

                X0_all = X0_all_full[:, kept_idx]
                mu0_all = mu0_all_full[kept_idx]
                Sigma_all = compute_covariance(X0_all, ridge=ridge)

                gene_to_k = {g: i for i, g in enumerate(kept_genes)}

                rows = []
                for pert in tqdm(perts_here, desc=f"{base} min{min_pct} max{max_pct}", leave=False):
                    idx_pert = pert_to_idx[pert]

                    X1_all = _to_dense(adata[idx_pert, :].X).astype(np.float64, copy=False)
                    X1_all = X1_all[:, kept_idx]

                    delta_X = X1_all.mean(axis=0) - mu0_all

                    k = gene_to_k.get(pert, None)
                    if k is None:
                        continue

                    alpha = fit_alpha(delta_X, Sigma_all, k, lam=lam)
                    r2, pearson_r, spearman_r = evaluate_delta(delta_X, Sigma_all, k, alpha)

                    rows.append({
                        "dataset": base,
                        "min_pct": int(min_pct),
                        "max_pct": int(max_pct),
                        "n_genes_kept": int(len(kept_idx)),
                        "perturbation": pert,
                        "n_cells_control_all": int(len(ctrl_idx_all)),
                        "n_cells_pert_all": int(len(idx_pert)),
                        "alpha": alpha,
                        "R2_all": r2,
                        "Pearson_all": pearson_r,
                        "Spearman_all": spearman_r,
                    })

                df = pd.DataFrame(rows)
                out_csv = os.path.join(
                    save_dir,
                    f"{base}_raw_only_metrics_all_cells_min{min_pct:03d}_max{max_pct:03d}.csv"
                )
                df.to_csv(out_csv, index=False)
                print(f"[SAVE] {out_csv}  (n_rows={len(df)})")

                if len(df) == 0:
                    print("[WARN] empty results.")
                    continue

                bounded_kde_overlay(
                    df,
                    cols=["R2_all"],
                    labels=["Raw counts"],
                    title=f"ALL-CELL R² — {base} (min{min_pct}, max{max_pct})",
                    xlabel="R² on all-cell ΔX",
                    outpath_svg=os.path.join(save_dir, f"{base}_R2_all_kde.svg"),
                )

                bounded_kde_overlay(
                    df,
                    cols=["Pearson_all"],
                    labels=["Raw counts"],
                    title=f"ALL-CELL Pearson — {base} (min{min_pct}, max{max_pct})",
                    xlabel="Pearson correlation on all-cell ΔX",
                    outpath_svg=os.path.join(save_dir, f"{base}_Pearson_all_kde.svg"),
                )

                bounded_kde_overlay(
                    df,
                    cols=["Spearman_all"],
                    labels=["Raw counts"],
                    title=f"ALL-CELL Spearman — {base} (min{min_pct}, max{max_pct})",
                    xlabel="Spearman correlation on all-cell ΔX",
                    outpath_svg=os.path.join(save_dir, f"{base}_Spearman_all_kde.svg"),
                )

        return True

    # -----------------------------
    # Example batch run
    # -----------------------------
    datapaths = [os.path.join(DATA_DIR, _n) for _n in [
        "ReplogleWeissman2022_rpe1.h5ad",
        # "ReplogleWeissman2022_K562_essential.h5ad",
        # "GSE264667_jurkat_raw_singlecell_01.h5ad",
        # "GSE264667_hepg2_raw_singlecell_01.h5ad",
        # "NormanWeissman2019_filtered.h5ad",
        # "FrangiehIzar2021_RNA.h5ad",
        # "TianKampmann2019_day7neuron.h5ad",
        # "TianKampmann2021_CRISPRi.h5ad",
        # "TianKampmann2021_CRISPRa.h5ad",
        # "TianKampmann2019_iPSC.h5ad",
    ]]

    min_percentiles = [10]
    max_percentiles = [80]

    for data_path in datapaths:
        try:
            analyze_dataset_percentiles_raw_only_all_cells(
                data_path,
                out_root = os.path.join(OUTDIR, "raw_counts_only_percentile_summary_all_cells"),
                min_percentiles=min_percentiles,
                max_percentiles=max_percentiles,
                min_cells_per_pert=50,
                min_cells_control=50,
                ridge=1e-6,
                lam=1e-10,
                unique_join="-",
            )
        except Exception as e:
            print(f"\n[ERROR] on {data_path}: {e}")
            traceback.print_exc()


def run_cross_norm_raw_freq():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # CROSS-NORMALIZATION LR TEST (RAW vs FREQ)
    #
    # For each dataset × (min_pct,max_pct) in your usual grid:
    #   Build Σ0 from CONTROL under:
    #     - RAW counts
    #     - FREQ normalized
    #   Build Δx from (PERT mean - CTRL mean) under:
    #     - RAW counts
    #     - FREQ normalized
    #
    # Evaluate 4 combos:
    #   1) Σ_raw  + Δx_raw   (raw/raw)
    #   2) Σ_freq + Δx_freq  (freq/freq)
    #   3) Σ_freq + Δx_raw   (freq/raw)   <-- your requested cross
    #   4) Σ_raw  + Δx_freq  (raw/freq)   <-- your requested cross
    #
    # Metrics per perturbation per combo:
    #   - R² (through-origin)
    #   - Pearson
    #   - Spearman
    #
    # Saves:
    #   <out_root>/<dataset>/minXXX_maxYYY/<dataset>_xnorm_metrics_minXXX_maxYYY.csv
    #
    # Plots (NO GRID):
    #   KDE overlays (bounded [-1,1]) for:
    #     - R² (4 curves)
    #     - Pearson (4 curves)
    #     - Spearman (4 curves)
    # ============================================================

    import os, glob, traceback
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.sparse import issparse
    from scipy.stats import pearsonr, rankdata
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="white", context="talk")
    plt.rcParams["axes.grid"] = False

    # -----------------------------
    # helpers
    # -----------------------------
    def _to_dense(X):
        return X.toarray() if issparse(X) else np.asarray(X)

    def safe_spearman(x, y):
        x = np.asarray(x); y = np.asarray(y)
        if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        xr = rankdata(x)
        yr = rankdata(y)
        r = np.corrcoef(xr, yr)[0, 1]
        if not np.isfinite(r):
            return np.nan
        return float(np.clip(r, -1, 1))

    def compute_covariance(X, ridge=1e-6):
        X = _to_dense(X)
        C = np.cov(X, rowvar=False)
        C = 0.5 * (C + C.T)
        tr = float(np.trace(C))
        if not np.isfinite(tr) or tr <= 0:
            tr = 1.0
        C += ridge * tr / C.shape[0] * np.eye(C.shape[0])
        return C

    def freq_norm(X):
        X = _to_dense(X).astype(np.float64, copy=False)
        totals = X.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return X / totals

    def bounded_kde_overlay(df, cols, labels, title, xlabel, outpath_svg):
        plt.figure(figsize=(8, 6))
        any_plotted = False
        for col, lab in zip(cols, labels):
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            vals = np.clip(vals, -1, 1)
            sns.kdeplot(vals, label=lab, fill=True, alpha=0.28, lw=2, clip=(-1, 1))
            any_plotted = True
        plt.xlim(-1, 1)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.title(title)
        if any_plotted:
            plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outpath_svg)
        plt.show()

    def run_eval(delta_X, Sigma, k, lam=1e-8):
        sigma = Sigma[:, k]
        denom = float(np.dot(sigma, sigma) + lam)
        alpha = float(np.dot(sigma, delta_X) / denom)
        pred  = alpha * sigma

        mask = np.isfinite(delta_X) & np.isfinite(pred)
        if mask.sum() < 3:
            return alpha, np.nan, np.nan, np.nan

        obs = delta_X[mask]
        pr  = pred[mask]
        if np.std(obs) < 1e-12 or np.std(pr) < 1e-12:
            return alpha, np.nan, np.nan, np.nan

        ss_res = float(np.sum((obs - pr) ** 2))
        ss_tot = float(np.sum(obs ** 2) + 1e-8)
        r2 = 1.0 - ss_res / ss_tot

        p, _ = pearsonr(obs, pr)
        p = float(np.clip(p, -1, 1))
        s = safe_spearman(obs, pr)
        return alpha, float(r2), p, s

    def safe_make_var_names_unique(adata, join="-"):
        try:
            adata.var_names = pd.Index([str(x) for x in list(adata.var_names)], dtype="object")
        except Exception:
            adata.var_names = pd.Index(pd.Series(adata.var_names).astype(str).tolist(), dtype="object")

        if hasattr(adata, "var") and isinstance(adata.var, pd.DataFrame):
            for c in adata.var.columns:
                if pd.api.types.is_categorical_dtype(adata.var[c]):
                    adata.var[c] = adata.var[c].astype(str)

        adata.var_names_make_unique(join=join)
        return adata

    # -----------------------------
    # main
    # -----------------------------
    def analyze_cross_norm_raw_freq(
        data_path,
        out_root = os.path.join(OUTDIR, "raw_freq_crossnorm_summary"),
        min_percentiles=(0, 10, 20, 30, 40),
        max_percentiles=(100, 90, 80, 70, 60),
        min_cells_per_pert=2,
        ridge=1e-6,
        lam=1e-8,
        unique_join="-",
    ):
        os.makedirs(out_root, exist_ok=True)
        base = os.path.basename(data_path).replace(".h5ad", "")
        print(f"\n==============================\nDATASET: {base}\n==============================")

        adata = sc.read_h5ad(data_path)
        safe_make_var_names_unique(adata, join=unique_join)

        if "perturbation" not in adata.obs.columns:
            raise ValueError(f"{base}: missing adata.obs['perturbation']")
        if "control" not in set(adata.obs["perturbation"].unique()):
            raise ValueError(f"{base}: no 'control' label in adata.obs['perturbation']")

        gene_names_full = np.array(adata.var_names.tolist(), dtype=object)
        gene_name_set   = set(gene_names_full)

        # single-gene perts only (label matches a gene)
        all_perts = [p for p in adata.obs["perturbation"].unique() if p != "control"]
        single_gene_perts = [p for p in all_perts if p in gene_name_set]
        if len(single_gene_perts) == 0:
            print(f"[WARN] {base}: no single-gene perturbations matching var_names.")
            return None

        # control cells once
        pert_col = adata.obs["perturbation"].to_numpy()
        ctrl_mask = (pert_col == "control")
        X0_full = _to_dense(adata[ctrl_mask, :].X).astype(np.float64, copy=False)
        if X0_full.shape[0] < 2:
            raise ValueError(f"{base}: too few control cells ({X0_full.shape[0]})")

        mu0_raw_full  = X0_full.mean(axis=0)
        X0_freq_full  = freq_norm(X0_full)
        mu0_freq_full = X0_freq_full.mean(axis=0)

        gene_means_full = mu0_raw_full  # percentile filter based on raw control mean

        # force-keep pert genes
        pert_set = set(single_gene_perts)
        force_keep_full = np.array([g in pert_set for g in gene_names_full], dtype=bool)

        # precompute pert indices
        pert_to_idx = {}
        for p in single_gene_perts:
            idx = np.where(pert_col == p)[0]
            if idx.size >= min_cells_per_pert:
                pert_to_idx[p] = idx
        if len(pert_to_idx) == 0:
            print(f"[WARN] {base}: no perts with >= {min_cells_per_pert} cells.")
            return None

        for min_pct in min_percentiles:
            for max_pct in max_percentiles:
                if max_pct < min_pct:
                    continue

                low_cut  = np.percentile(gene_means_full, min_pct)
                high_cut = np.percentile(gene_means_full, max_pct)

                valid_genes = (gene_means_full >= low_cut) & (gene_means_full <= high_cut)
                keep_mask   = valid_genes | force_keep_full
                kept_idx    = np.where(keep_mask)[0]
                kept_genes  = gene_names_full[kept_idx]
                kept_set    = set(kept_genes)

                perts_here = [p for p in pert_to_idx.keys() if p in kept_set]
                if len(perts_here) == 0:
                    continue

                save_dir = os.path.join(out_root, base, f"min{min_pct:03d}_max{max_pct:03d}")
                os.makedirs(save_dir, exist_ok=True)

                print(f"\n[RUN] {base} | min_pct={min_pct} max_pct={max_pct} | kept_genes={len(kept_idx)} | perts={len(perts_here)}")

                # slice control means + covariances for BOTH norms
                X0_raw  = X0_full[:, kept_idx]
                mu0_raw = mu0_raw_full[kept_idx]

                X0_freq  = X0_freq_full[:, kept_idx]
                mu0_freq = mu0_freq_full[kept_idx]

                Sigma_raw  = compute_covariance(X0_raw,  ridge=ridge)
                Sigma_freq = compute_covariance(X0_freq, ridge=ridge)

                gene_to_k = {g:i for i,g in enumerate(kept_genes)}

                rows = []
                for pert in tqdm(perts_here, desc=f"{base} min{min_pct} max{max_pct}", leave=False):
                    idx_cells = pert_to_idx[pert]

                    X1_raw_full = _to_dense(adata[idx_cells, :].X).astype(np.float64, copy=False)
                    X1_raw      = X1_raw_full[:, kept_idx]
                    X1_freq     = freq_norm(X1_raw_full)[:, kept_idx]

                    mu1_raw  = X1_raw.mean(axis=0)
                    mu1_freq = X1_freq.mean(axis=0)

                    dx_raw  = mu1_raw  - mu0_raw
                    dx_freq = mu1_freq - mu0_freq

                    k = gene_to_k.get(pert, None)
                    if k is None:
                        continue

                    # 4 combos
                    a_rr, r2_rr, p_rr, s_rr = run_eval(dx_raw,  Sigma_raw,  k, lam=lam)   # raw/raw
                    a_ff, r2_ff, p_ff, s_ff = run_eval(dx_freq, Sigma_freq, k, lam=lam)   # freq/freq
                    a_fr, r2_fr, p_fr, s_fr = run_eval(dx_raw,  Sigma_freq, k, lam=lam)   # freq/raw  (Sigma_freq, dx_raw)
                    a_rf, r2_rf, p_rf, s_rf = run_eval(dx_freq, Sigma_raw,  k, lam=lam)   # raw/freq  (Sigma_raw, dx_freq)

                    rows.append({
                        "dataset": base,
                        "min_pct": int(min_pct),
                        "max_pct": int(max_pct),
                        "n_genes_kept": int(len(kept_idx)),
                        "perturbation": pert,
                        "n_cells_pert": int(idx_cells.size),

                        # raw/raw
                        "alpha_rr": a_rr, "R2_rr": r2_rr, "Pearson_rr": p_rr, "Spearman_rr": s_rr,
                        # freq/freq
                        "alpha_ff": a_ff, "R2_ff": r2_ff, "Pearson_ff": p_ff, "Spearman_ff": s_ff,
                        # Sigma_freq + dx_raw
                        "alpha_fr": a_fr, "R2_fr": r2_fr, "Pearson_fr": p_fr, "Spearman_fr": s_fr,
                        # Sigma_raw + dx_freq
                        "alpha_rf": a_rf, "R2_rf": r2_rf, "Pearson_rf": p_rf, "Spearman_rf": s_rf,
                    })

                df = pd.DataFrame(rows)
                out_csv = os.path.join(save_dir, f"{base}_xnorm_metrics_min{min_pct:03d}_max{max_pct:03d}.csv")
                df.to_csv(out_csv, index=False)
                print(f"[SAVE] {out_csv} (n_rows={len(df)})")

                if len(df) == 0:
                    continue

                # KDE overlays for 4-combo comparison
                bounded_kde_overlay(
                    df,
                    cols=["R2_rr","R2_ff","R2_fr","R2_rf"],
                    labels=["Σ_raw + Δx_raw", "Σ_freq + Δx_freq", "Σ_freq + Δx_raw", "Σ_raw + Δx_freq"],
                    title=f"R² cross-norm — {base} (min{min_pct}, max{max_pct})",
                    xlabel="R²",
                    outpath_svg=os.path.join(save_dir, f"{base}_xnorm_R2_kde.svg"),
                )
                bounded_kde_overlay(
                    df,
                    cols=["Pearson_rr","Pearson_ff","Pearson_fr","Pearson_rf"],
                    labels=["Σ_raw + Δx_raw", "Σ_freq + Δx_freq", "Σ_freq + Δx_raw", "Σ_raw + Δx_freq"],
                    title=f"Pearson cross-norm — {base} (min{min_pct}, max{max_pct})",
                    xlabel="Pearson correlation",
                    outpath_svg=os.path.join(save_dir, f"{base}_xnorm_Pearson_kde.svg"),
                )
                bounded_kde_overlay(
                    df,
                    cols=["Spearman_rr","Spearman_ff","Spearman_fr","Spearman_rf"],
                    labels=["Σ_raw + Δx_raw", "Σ_freq + Δx_freq", "Σ_freq + Δx_raw", "Σ_raw + Δx_freq"],
                    title=f"Spearman cross-norm — {base} (min{min_pct}, max{max_pct})",
                    xlabel="Spearman correlation",
                    outpath_svg=os.path.join(save_dir, f"{base}_xnorm_Spearman_kde.svg"),
                )

        return True


    # -----------------------------
    # Batch run
    # -----------------------------
    datapaths = [os.path.join(DATA_DIR, _n) for _n in [
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]]

    min_percentiles = [0, 10, 20, 30, 40]
    max_percentiles = [100, 90, 80, 70, 60]

    for data_path in datapaths:
        try:
            analyze_cross_norm_raw_freq(
                data_path,
                out_root = os.path.join(OUTDIR, "raw_freq_crossnorm_summary"),
                min_percentiles=min_percentiles,
                max_percentiles=max_percentiles,
                min_cells_per_pert=2,
                ridge=1e-6,
                lam=1e-8,
                unique_join="-",
            )
        except Exception as e:
            print(f"\n[ERROR] on {data_path}: {e}")
            traceback.print_exc()


def plot_pearson_multipanel():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # LOAD ALL RESULTS + MULTIPANEL FIGURE (Pearson, minmax 0-100)
    #   - Reads per-pert CSVs produced by analyze_dataset_percentiles()
    #   - Filters to min_pct=0, max_pct=100 only
    #   - Makes one subplot per dataset
    #   - Each subplot overlays Pearson KDEs: Raw vs Partial vs Frequency
    #   - Saves: pearson_multipanel_min000_max100.(png,svg)
    # ============================================================

    import os, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="white", context="talk")
    plt.rcParams["axes.grid"] = False

    sns.set(style="whitegrid", context="talk")

    OUT_ROOT = os.path.join(OUTDIR, "raw_partial_freq_percentile_summary")   # <-- where your script wrote outputs
    MIN_PCT  = 0
    MAX_PCT  = 100

    # -----------------------------
    # 1) Load all CSVs under OUT_ROOT
    # -----------------------------
    csvs = glob.glob(os.path.join(OUT_ROOT, "**", "*_metrics_min*_max*.csv"), recursive=True)
    if len(csvs) == 0:
        raise FileNotFoundError(f"No metrics CSVs found under {OUT_ROOT}")

    dfs = []
    for fp in csvs:
        try:
            df = pd.read_csv(fp)
            # robustly ensure required columns exist
            need = {"dataset","min_pct","max_pct","Pearson_raw","Pearson_part","Pearson_freq"}
            if not need.issubset(df.columns):
                continue
            dfs.append(df)
        except Exception:
            pass

    if len(dfs) == 0:
        raise RuntimeError("Found CSVs but none contained required columns.")

    all_df = pd.concat(dfs, ignore_index=True)

    # -----------------------------
    # 2) Filter to minmax 0-100
    # -----------------------------
    df = all_df[(all_df["min_pct"] == MIN_PCT) & (all_df["max_pct"] == MAX_PCT)].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows found for min_pct={MIN_PCT}, max_pct={MAX_PCT}. "
                           f"Double-check outputs exist for min000_max100.")

    # Clip Pearson to [-1,1] and drop non-finite
    for c in ["Pearson_raw","Pearson_part","Pearson_freq"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[~np.isfinite(df[c].to_numpy()), c] = np.nan
        df[c] = df[c].clip(-1, 1)

    # -----------------------------
    # 3) Multipanel KDEs: one subplot per dataset
    # -----------------------------
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    n = len(datasets)
    if n == 0:
        raise RuntimeError("No datasets found after filtering.")

    # grid shape (pretty)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))

    fig_w = 6.0 * ncols
    fig_h = 4.8 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # We won't set explicit colors (your preference); seaborn defaults will be distinct.
    labels = ["Raw", "Partial (log1p/sf)", "Frequency"]
    cols   = ["Pearson_raw", "Pearson_part", "Pearson_freq"]

    for i, ds in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        dsub = df[df["dataset"] == ds]

        plotted_any = False
        for col, lab in zip(cols, labels):
            vals = dsub[col].dropna().to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size < 5:
                continue
            vals = np.clip(vals, -1, 1)

            sns.kdeplot(
                vals,
                ax=ax,
                label=lab,
                fill=True,
                alpha=0.28,
                lw=2,
                clip=(-1, 1),
                common_norm=False,   # per-dataset compare shapes; not forced to integrate the same
            )
            plotted_any = True

        ax.set_title(ds)
        ax.set_xlim(-0.5, 1)
        ax.set_xlabel("Pearson correlation")
        ax.set_ylabel("Density")
        if plotted_any:
            ax.legend(frameon=False, fontsize=11)
        else:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)

    # Turn off unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    plt.suptitle(f"Pearson distributions by dataset (min{MIN_PCT:03d}_max{MAX_PCT:03d})", y=1.02)
    plt.tight_layout()

    out_png = os.path.join(OUT_ROOT, f"pearson_multipanel_min{MIN_PCT:03d}_max{MAX_PCT:03d}.png")
    out_svg = os.path.join(OUT_ROOT, f"pearson_multipanel_min{MIN_PCT:03d}_max{MAX_PCT:03d}.svg")
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_svg)
    plt.show()

    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_svg}")
    print(f"[INFO] Loaded {len(all_df)} total rows across all percentiles; "
          f"using {len(df)} rows for min{MIN_PCT:03d}_max{MAX_PCT:03d} across {n} datasets.")


def plot_spearman_multipanel():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # LOAD ALL RESULTS + MULTIPANEL FIGURE (Spearman, minmax 0-100)
    #   - Reads per-pert CSVs produced by analyze_dataset_percentiles()
    #   - Filters to min_pct=0, max_pct=100 only
    #   - Makes one subplot per dataset
    #   - Each subplot overlays Spearman KDEs: Raw vs Partial vs Frequency
    #   - Saves: spearman_multipanel_min000_max100.(png,svg)
    # ============================================================

    import os, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="white", context="talk")
    plt.rcParams["axes.grid"] = False

    sns.set(style="whitegrid", context="talk")

    OUT_ROOT = os.path.join(OUTDIR, "raw_partial_freq_percentile_summary")   # <-- where your script wrote outputs
    MIN_PCT  = 0
    MAX_PCT  = 100

    # -----------------------------
    # 1) Load all CSVs under OUT_ROOT
    # -----------------------------
    csvs = glob.glob(os.path.join(OUT_ROOT, "**", "*_metrics_min*_max*.csv"), recursive=True)
    if len(csvs) == 0:
        raise FileNotFoundError(f"No metrics CSVs found under {OUT_ROOT}")

    dfs = []
    for fp in csvs:
        try:
            df = pd.read_csv(fp)
            # robustly ensure required columns exist
            need = {"dataset","min_pct","max_pct","Spearman_raw","Spearman_part","Spearman_freq"}
            if not need.issubset(df.columns):
                continue
            dfs.append(df)
        except Exception:
            pass

    if len(dfs) == 0:
        raise RuntimeError("Found CSVs but none contained required columns.")

    all_df = pd.concat(dfs, ignore_index=True)

    # -----------------------------
    # 2) Filter to minmax 0-100
    # -----------------------------
    df = all_df[(all_df["min_pct"] == MIN_PCT) & (all_df["max_pct"] == MAX_PCT)].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows found for min_pct={MIN_PCT}, max_pct={MAX_PCT}. "
                           f"Double-check outputs exist for min000_max100.")

    # Clip Spearman to [-1,1] and drop non-finite
    for c in ["Spearman_raw","Spearman_part","Spearman_freq"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[~np.isfinite(df[c].to_numpy()), c] = np.nan
        df[c] = df[c].clip(-1, 1)

    # -----------------------------
    # 3) Multipanel KDEs: one subplot per dataset
    # -----------------------------
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    n = len(datasets)
    if n == 0:
        raise RuntimeError("No datasets found after filtering.")

    # grid shape (pretty)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))

    fig_w = 6.0 * ncols
    fig_h = 4.8 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # We won't set explicit colors (your preference); seaborn defaults will be distinct.
    labels = ["Raw", "Partial (log1p/sf)", "Frequency"]
    cols   = ["Spearman_raw", "Spearman_part", "Spearman_freq"]

    for i, ds in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        dsub = df[df["dataset"] == ds]

        plotted_any = False
        for col, lab in zip(cols, labels):
            vals = dsub[col].dropna().to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size < 5:
                continue
            vals = np.clip(vals, -1, 1)

            sns.kdeplot(
                vals,
                ax=ax,
                label=lab,
                fill=True,
                alpha=0.28,
                lw=2,
                clip=(-1, 1),
                common_norm=False,   # per-dataset compare shapes; not forced to integrate the same
            )
            plotted_any = True

        ax.set_title(ds)
        ax.set_xlim(-0.5, 1)
        ax.set_xlabel("Spearman correlation")
        ax.set_ylabel("Density")
        if plotted_any:
            ax.legend(frameon=False, fontsize=11)
        else:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)

    # Turn off unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    plt.suptitle(f"Spearman distributions by dataset (min{MIN_PCT:03d}_max{MAX_PCT:03d})", y=1.02)
    plt.tight_layout()

    out_png = os.path.join(OUT_ROOT, f"spearman_multipanel_min{MIN_PCT:03d}_max{MAX_PCT:03d}.png")
    out_svg = os.path.join(OUT_ROOT, f"spearman_multipanel_min{MIN_PCT:03d}_max{MAX_PCT:03d}.svg")
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_svg)
    plt.show()

    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_svg}")
    print(f"[INFO] Loaded {len(all_df)} total rows across all percentiles; "
          f"using {len(df)} rows for min{MIN_PCT:03d}_max{MAX_PCT:03d} across {n} datasets.")


def plot_mean_sem_grids():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # MULTIPANEL SUMMARY FIGURES (NO GRID LINES)
    #   - 2×5 grid (9 datasets, one empty panel)
    #   - 2×4 grid (first 8 datasets)
    #   - Make BOTH metrics:
    #       (A) Pearson mean ± SEM (Raw / Partial / Frequency)
    #       (B) R²      mean ± SEM (Raw / Partial / Frequency)
    #   - Saves PNG + SVG for each figure.
    # ============================================================

    import os, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import sem

    # -----------------------------
    # CONFIG
    # -----------------------------
    datasets = [
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "GSE264667_jurkat_raw_singlecell_01",
        "GSE264667_hepg2_raw_singlecell_01",
        "NormanWeissman2019_filtered",
        "FrangiehIzar2021_RNA",
        "TianKampmann2019_day7neuron",
        "TianKampmann2021_CRISPRi",
        "TianKampmann2021_CRISPRa",
    ]

    min_percentiles = [0, 10, 20, 30]
    max_percentiles = [70, 80, 90, 100]

    base_dir = os.path.join(OUTDIR, "raw_partial_freq_percentile_summary")
    out_dir  = os.path.join(base_dir, "summary_plots")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["axes.grid"] = False  # hard-disable grids everywhere

    # -----------------------------
    # Helpers
    # -----------------------------
    def mean_sem(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        if x.size == 0:
            return np.nan, np.nan
        return float(x.mean()), float(sem(x)) if x.size > 1 else np.nan

    def find_csv(folder):
        hits = glob.glob(os.path.join(folder, "*_metrics_min*_max*.csv"))
        return hits[0] if hits else None

    def load_dataset_metric(dataset, metric_base):
        """
        metric_base in {"Pearson", "R2"}
        returns labels and dict of mean/sem arrays for raw/part/freq
        """
        labels = []
        out = {k: [] for k in ["raw_m","raw_s","part_m","part_s","freq_m","freq_s"]}

        for min_pct in min_percentiles:
            for max_pct in max_percentiles:
                # keep ONLY: (max == 100) OR (min == 0)
                if not ((max_pct == 100) or (min_pct == 0)):
                    continue

                labels.append(f"min{min_pct}_max{max_pct}")

                folder = os.path.join(base_dir, dataset, f"min{min_pct:03d}_max{max_pct:03d}")
                csv_path = find_csv(folder)
                if (csv_path is None) or (not os.path.exists(csv_path)):
                    for k in out:
                        out[k].append(np.nan)
                    continue

                df = pd.read_csv(csv_path)

                c_raw  = f"{metric_base}_raw"
                c_part = f"{metric_base}_part"
                c_freq = f"{metric_base}_freq"

                m,s = mean_sem(df.get(c_raw,  np.nan)); out["raw_m"].append(m);  out["raw_s"].append(s)
                m,s = mean_sem(df.get(c_part, np.nan)); out["part_m"].append(m); out["part_s"].append(s)
                m,s = mean_sem(df.get(c_freq, np.nan)); out["freq_m"].append(m); out["freq_s"].append(s)

        return labels, out

    def plot_grid(datasets_to_plot, nrows, ncols, metric_base, fname_prefix, ylim):
        """
        metric_base: "Pearson" or "R2"
        fname_prefix: e.g. "pearson" or "r2"
        ylim: tuple (ymin,ymax)
        """
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.8*nrows), squeeze=False)

        for i, ds in enumerate(datasets_to_plot):
            ax = axes[i // ncols][i % ncols]
            labels, d = load_dataset_metric(ds, metric_base)
            x = np.arange(len(labels))

            ax.errorbar(x, d["raw_m"],  yerr=d["raw_s"],  fmt="o-", lw=2, capsize=3, markersize=5, label="Raw")
            ax.errorbar(x, d["part_m"], yerr=d["part_s"], fmt="o-", lw=2, capsize=3, markersize=5, label="Partial")
            ax.errorbar(x, d["freq_m"], yerr=d["freq_s"], fmt="o-", lw=2, capsize=3, markersize=5, label="Frequency")

            ax.set_title(ds, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylim(*ylim)
            ax.set_ylabel(f"Mean {metric_base} ± SEM")
            ax.grid(False)

            if i == 0:
                ax.legend(frameon=False, fontsize=9)

        # Turn off unused panels
        for j in range(len(datasets_to_plot), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        plt.tight_layout()

        png = os.path.join(out_dir, f"{fname_prefix}_{nrows}x{ncols}.png")
        svg = os.path.join(out_dir, f"{fname_prefix}_{nrows}x{ncols}.svg")
        plt.savefig(png, dpi=200)
        plt.savefig(svg)
        plt.show()

        print(f"[SAVED] {png}")
        print(f"[SAVED] {svg}")

    # ============================================================
    # PEARSON FIGURES
    # ============================================================
    plot_grid(
        datasets_to_plot=datasets, nrows=2, ncols=5,
        metric_base="Pearson",
        fname_prefix="pearson_mean_sem_raw_part_freq",
        ylim=(-1, 1),
    )
    plot_grid(
        datasets_to_plot=datasets[:8], nrows=2, ncols=4,
        metric_base="Pearson",
        fname_prefix="pearson_mean_sem_raw_part_freq",
        ylim=(-1, 1),
    )

    # ============================================================
    # R² FIGURES
    #   (your R² can be negative; keep a symmetric-ish view)
    #   If you want [0,1], change ylim=(0,1)
    # ============================================================
    plot_grid(
        datasets_to_plot=datasets, nrows=2, ncols=5,
        metric_base="R2",
        fname_prefix="r2_mean_sem_raw_part_freq",
        ylim=(-1, 1),
    )
    plot_grid(
        datasets_to_plot=datasets[:8], nrows=2, ncols=4,
        metric_base="R2",
        fname_prefix="r2_mean_sem_raw_part_freq",
        ylim=(-1, 1),
    )


def plot_r2_raw_sem():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # RAW-ONLY R² MULTIPANEL (NO GRID LINES)
    #   - One panel per dataset
    #   - Plots mean R²_raw ± SEM across the SAME percentile grid subset:
    #       keep ONLY: (max_pct == 100) OR (min_pct == 0)
    #   - Saves PNG + SVG
    #
    # Reads CSVs produced by analyze_dataset_percentiles():
    #   raw_partial_freq_percentile_summary/<dataset>/minXXX_maxYYY/*_metrics_minXXX_maxYYY.csv
    # ============================================================

    import os, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import sem

    # -----------------------------
    # CONFIG
    # -----------------------------
    datasets = [
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "GSE264667_jurkat_raw_singlecell_01",
        "GSE264667_hepg2_raw_singlecell_01",
        "NormanWeissman2019_filtered",
        "FrangiehIzar2021_RNA",
        "TianKampmann2019_day7neuron",
        "TianKampmann2021_CRISPRi",
        "TianKampmann2021_CRISPRa",
    ]

    min_percentiles = [0, 10, 20, 30]
    max_percentiles = [70, 80, 90, 100]

    base_dir = os.path.join(OUTDIR, "raw_partial_freq_percentile_summary")
    out_dir  = os.path.join(base_dir, "summary_plots")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["axes.grid"] = False  # hard-disable grids everywhere

    # -----------------------------
    # Helpers
    # -----------------------------
    def mean_sem(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        if x.size == 0:
            return np.nan, np.nan
        return float(x.mean()), float(sem(x)) if x.size > 1 else np.nan

    def find_csv(folder):
        hits = glob.glob(os.path.join(folder, "*_metrics_min*_max*.csv"))
        return hits[0] if hits else None

    def load_dataset_r2_raw(dataset):
        labels = []
        m_list = []
        s_list = []

        for min_pct in min_percentiles:
            for max_pct in max_percentiles:
                # keep ONLY: (max == 100) OR (min == 0)
                if not ((max_pct == 100) or (min_pct == 0)):
                    continue

                labels.append(f"min{min_pct}_max{max_pct}")

                folder = os.path.join(base_dir, dataset, f"min{min_pct:03d}_max{max_pct:03d}")
                csv_path = find_csv(folder)
                if (csv_path is None) or (not os.path.exists(csv_path)):
                    m_list.append(np.nan); s_list.append(np.nan)
                    continue

                df = pd.read_csv(csv_path)
                m, s = mean_sem(df.get("R2_raw", np.nan))
                m_list.append(m); s_list.append(s)

        return labels, np.array(m_list), np.array(s_list)

    def plot_grid_raw_r2(datasets_to_plot, nrows, ncols, fname_prefix="r2_raw_mean_sem", ylim=(-1, 1)):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.8*nrows), squeeze=False)

        for i, ds in enumerate(datasets_to_plot):
            ax = axes[i // ncols][i % ncols]
            labels, mean_r2, sem_r2 = load_dataset_r2_raw(ds)
            x = np.arange(len(labels))

            ax.errorbar(x, mean_r2, yerr=sem_r2, fmt="o-", lw=2, capsize=3, markersize=5)
            ax.set_title(ds, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylim(*ylim)
            ax.set_ylabel("Mean R²_raw ± SEM")
            ax.grid(False)

        # Turn off unused panels
        for j in range(len(datasets_to_plot), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        plt.tight_layout()
        png = os.path.join(out_dir, f"{fname_prefix}_{nrows}x{ncols}.png")
        svg = os.path.join(out_dir, f"{fname_prefix}_{nrows}x{ncols}.svg")
        plt.savefig(png, dpi=200)
        plt.savefig(svg)
        plt.show()
        print(f"[SAVED] {png}")
        print(f"[SAVED] {svg}")

    # -----------------------------
    # Make the two multipanels you were using
    # -----------------------------
    plot_grid_raw_r2(datasets,   nrows=3, ncols=4, fname_prefix="r2_raw_mean_sem", ylim=(0., 1))
    # plot_grid_raw_r2(datasets[:8], nrows=2, ncols=4, fname_prefix="r2_raw_mean_sem", ylim=(-1, 1))


def plot_r2_raw_std():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # RAW-ONLY R² MULTIPANEL (NO GRID LINES)
    #   - One panel per dataset
    #   - Plots mean R²_raw ± STD across the SAME percentile grid subset:
    #       keep ONLY: (max_pct == 100) OR (min_pct == 0)
    #   - Saves PNG + SVG
    #
    # Reads CSVs produced by analyze_dataset_percentiles():
    #   raw_partial_freq_percentile_summary/<dataset>/minXXX_maxYYY/*_metrics_minXXX_maxYYY.csv
    # ============================================================

    import os, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # -----------------------------
    # CONFIG
    # -----------------------------
    datasets = [
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "GSE264667_jurkat_raw_singlecell_01",
        "GSE264667_hepg2_raw_singlecell_01",
        "NormanWeissman2019_filtered",
        "FrangiehIzar2021_RNA",
        "TianKampmann2019_day7neuron",
        "TianKampmann2021_CRISPRi",
        "TianKampmann2021_CRISPRa",
    ]

    min_percentiles = [0, 10, 20, 30]
    max_percentiles = [70, 80, 90, 100]

    base_dir = os.path.join(OUTDIR, "raw_partial_freq_percentile_summary")
    out_dir  = os.path.join(base_dir, "summary_plots")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["axes.grid"] = False  # hard-disable grids everywhere

    # -----------------------------
    # Helpers
    # -----------------------------
    def mean_std(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        if x.size == 0:
            return np.nan, np.nan
        return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else np.nan

    def find_csv(folder):
        hits = glob.glob(os.path.join(folder, "*_metrics_min*_max*.csv"))
        return hits[0] if hits else None

    def load_dataset_r2_raw(dataset):
        labels = []
        m_list = []
        s_list = []

        for min_pct in min_percentiles:
            for max_pct in max_percentiles:
                # keep ONLY: (max == 100) OR (min == 0)
                if not ((max_pct == 100) or (min_pct == 0)):
                    continue

                labels.append(f"min{min_pct}_max{max_pct}")

                folder = os.path.join(base_dir, dataset, f"min{min_pct:03d}_max{max_pct:03d}")
                csv_path = find_csv(folder)
                if (csv_path is None) or (not os.path.exists(csv_path)):
                    m_list.append(np.nan); s_list.append(np.nan)
                    continue

                df = pd.read_csv(csv_path)
                m, s = mean_std(df.get("R2_raw", np.nan))
                m_list.append(m); s_list.append(s)

        return labels, np.array(m_list), np.array(s_list)

    def plot_grid_raw_r2(datasets_to_plot, nrows, ncols, fname_prefix="r2_raw_mean_std", ylim=(-1, 1)):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.8*nrows), squeeze=False)

        for i, ds in enumerate(datasets_to_plot):
            ax = axes[i // ncols][i % ncols]
            labels, mean_r2, std_r2 = load_dataset_r2_raw(ds)
            x = np.arange(len(labels))

            ax.errorbar(x, mean_r2, yerr=std_r2, fmt="o-", lw=2, capsize=3, markersize=5)
            ax.set_title(ds, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylim(*ylim)
            ax.set_ylabel("Mean R²_raw ± STD")
            ax.grid(False)

        # Turn off unused panels
        for j in range(len(datasets_to_plot), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        plt.tight_layout()
        png = os.path.join(out_dir, f"{fname_prefix}_{nrows}x{ncols}.png")
        svg = os.path.join(out_dir, f"{fname_prefix}_{nrows}x{ncols}.svg")
        plt.savefig(png, dpi=200)
        plt.savefig(svg)
        plt.show()
        print(f"[SAVED] {png}")
        print(f"[SAVED] {svg}")

    # -----------------------------
    # Make the multipanel
    # -----------------------------
    plot_grid_raw_r2(datasets, nrows=3, ncols=4, fname_prefix="r2_raw_mean_std", ylim=(0., 1))


def plot_xnorm_4combo():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # CROSS-NORMALIZATION SUMMARY MULTIPANEL (NO GRID LINES)
    #   - One panel per dataset
    #   - Plots mean ± STD across perturbations for the SAME percentile grid subset:
    #       keep ONLY: (max_pct == 100) OR (min_pct == 0)
    #   - 4 curves per panel (RAW vs FREQ cross-combos):
    #       1) rr = Σ_raw  + Δx_raw
    #       2) rf = Σ_raw  + Δx_freq
    #       3) fr = Σ_freq + Δx_raw
    #       4) ff = Σ_freq + Δx_freq
    #   - Saves PNG + SVG
    #
    # Reads CSVs produced by analyze_cross_norm_raw_freq():
    #   raw_freq_crossnorm_summary/<dataset>/minXXX_maxYYY/<dataset>_xnorm_metrics_minXXX_maxYYY.csv
    # ============================================================

    import os, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # -----------------------------
    # CONFIG
    # -----------------------------
    datasets = [
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "GSE264667_jurkat_raw_singlecell_01",
        "GSE264667_hepg2_raw_singlecell_01",
        "NormanWeissman2019_filtered",
        "FrangiehIzar2021_RNA",
        "TianKampmann2019_day7neuron",
        "TianKampmann2021_CRISPRi",
        "TianKampmann2021_CRISPRa",
    ]

    min_percentiles = [0, 10, 20, 30]
    max_percentiles = [70, 80, 90, 100]

    base_dir = os.path.join(OUTDIR, "raw_freq_crossnorm_summary")
    out_dir  = os.path.join(base_dir, "summary_plots")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["axes.grid"] = False  # hard-disable grids everywhere

    # -----------------------------
    # Helpers
    # -----------------------------
    def mean_std(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        if x.size == 0:
            return np.nan, np.nan
        return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else np.nan

    def find_csv(folder):
        hits = glob.glob(os.path.join(folder, "*_xnorm_metrics_min*_max*.csv"))
        return hits[0] if hits else None

    def load_dataset_metric_4combo(dataset, metric_base):
        """
        metric_base in {"R2","Pearson","Spearman"}
        returns labels and dict of mean/std arrays for rr/rf/fr/ff
        """
        labels = []
        out = {k: [] for k in ["rr_m","rr_s","rf_m","rf_s","fr_m","fr_s","ff_m","ff_s"]}

        for min_pct in min_percentiles:
            for max_pct in max_percentiles:
                # keep ONLY: (max == 100) OR (min == 0)
                if not ((max_pct == 100) or (min_pct == 0)):
                    continue

                labels.append(f"min{min_pct}_max{max_pct}")

                folder = os.path.join(base_dir, dataset, f"min{min_pct:03d}_max{max_pct:03d}")
                csv_path = find_csv(folder)
                if (csv_path is None) or (not os.path.exists(csv_path)):
                    for k in out:
                        out[k].append(np.nan)
                    continue

                df = pd.read_csv(csv_path)

                for tag in ["rr","rf","fr","ff"]:
                    col = f"{metric_base}_{tag}"
                    m, s = mean_std(df.get(col, np.nan))
                    out[f"{tag}_m"].append(m)
                    out[f"{tag}_s"].append(s)

        return labels, out

    def plot_grid_4combo(datasets_to_plot, nrows, ncols, metric_base,
                         fname_prefix, ylim=(-1,1)):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.8*nrows), squeeze=False)

        for i, ds in enumerate(datasets_to_plot):
            ax = axes[i // ncols][i % ncols]
            labels, d = load_dataset_metric_4combo(ds, metric_base)
            x = np.arange(len(labels))

            ax.errorbar(x, d["rr_m"], yerr=d["rr_s"], fmt="o-", lw=2, capsize=3, markersize=5, label="Σ_raw + Δx_raw (rr)")
            ax.errorbar(x, d["rf_m"], yerr=d["rf_s"], fmt="o-", lw=2, capsize=3, markersize=5, label="Σ_raw + Δx_freq (rf)")
            ax.errorbar(x, d["fr_m"], yerr=d["fr_s"], fmt="o-", lw=2, capsize=3, markersize=5, label="Σ_freq + Δx_raw (fr)")
            ax.errorbar(x, d["ff_m"], yerr=d["ff_s"], fmt="o-", lw=2, capsize=3, markersize=5, label="Σ_freq + Δx_freq (ff)")

            ax.set_title(ds, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylim(*ylim)
            ax.set_ylabel(f"Mean {metric_base} ± STD")
            ax.grid(False)

            if i == 0:
                ax.legend(frameon=False, fontsize=8)

        # Turn off unused panels
        for j in range(len(datasets_to_plot), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        plt.tight_layout()
        png = os.path.join(out_dir, f"{fname_prefix}_{nrows}x{ncols}.png")
        svg = os.path.join(out_dir, f"{fname_prefix}_{nrows}x{ncols}.svg")
        plt.savefig(png, dpi=200)
        plt.savefig(svg)
        plt.show()
        print(f"[SAVED] {png}")
        print(f"[SAVED] {svg}")

    # ============================================================
    # R² MULTIPANELS (4 curves per dataset)
    # ============================================================
    plot_grid_4combo(
        datasets_to_plot=datasets,
        nrows=3, ncols=4,
        metric_base="R2",
        fname_prefix="xnorm_r2_mean_std_4combo",
        ylim=(-0.25, 1),   # set to (0,1) if you want
    )

    # If you also want Pearson/Spearman, uncomment:
    plot_grid_4combo(datasets, 3, 4, "Pearson",  "xnorm_pearson_mean_std_4combo", ylim=(-0.25,1))
    plot_grid_4combo(datasets, 3, 4, "Spearman", "xnorm_spearman_mean_std_4combo", ylim=(-0.25,1))


def plot_xnorm_category_summary():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # CROSS-NORMALIZATION CATEGORY SUMMARY (NO GRID LINES)
    #
    # Make THREE plots *per metric*:
    #   (1) CRISPRa datasets only
    #   (2) CRISPRi datasets only
    #   (3) ALL datasets together
    #
    # Metrics: R2, Pearson, Spearman
    #
    # For each plot:
    #   - Load all saved per-pert CSVs from analyze_cross_norm_raw_freq()
    #   - Keep ONLY percentile grid subset:
    #       keep ONLY: (max_pct == 100) OR (min_pct == 0)
    #   - Collapse duplicates at the PERTURBATION level (avg) if any
    #   - Aggregate within CATEGORY at each grid point:
    #       mean = mean over perturbations (pooled across datasets)
    #       std  = std  over perturbations (pooled across datasets)
    #   - Plot mean ± STD for 4 combos:
    #       rr = Σ_raw  + Δx_raw
    #       rf = Σ_raw  + Δx_freq
    #       fr = Σ_freq + Δx_raw
    #       ff = Σ_freq + Δx_freq
    #
    # Saves PNG + SVG for each figure.
    #
    # Reads:
    #   raw_freq_crossnorm_summary/<dataset>/minXXX_maxYYY/<dataset>_xnorm_metrics_minXXX_maxYYY.csv
    # ============================================================

    import os, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.rcParams["axes.grid"] = False  # hard-disable grids everywhere

    # -----------------------------
    # CONFIG
    # -----------------------------
    base_dir = os.path.join(OUTDIR, "raw_freq_crossnorm_summary")
    out_dir  = os.path.join(base_dir, "category_summary_plots")
    os.makedirs(out_dir, exist_ok=True)

    min_percentiles = [0, 10, 20, 30]
    max_percentiles = [70, 80, 90, 100]

    # Your 9 datasets
    ALL_DATASETS = [
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "GSE264667_jurkat_raw_singlecell_01",
        "GSE264667_hepg2_raw_singlecell_01",
        "NormanWeissman2019_filtered",
        "FrangiehIzar2021_RNA",
        "TianKampmann2019_day7neuron",
        "TianKampmann2021_CRISPRi",
        "TianKampmann2021_CRISPRa",
        "TianKampmann2019_iPSC"
    ]

    # Categories (edit as desired)
    CRISPRa_DATASETS = ["TianKampmann2021_CRISPRa", 'NormanWeissman2019_filtered']
    CRISPRi_DATASETS = [
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "GSE264667_jurkat_raw_singlecell_01",
        "GSE264667_hepg2_raw_singlecell_01",
        "FrangiehIzar2021_RNA",
        "TianKampmann2019_day7neuron",
        "TianKampmann2021_CRISPRi",
        "TianKampmann2019_iPSC"
    ]

    METRICS = ["R2", "Pearson", "Spearman"]

    # -----------------------------
    # Helpers
    # -----------------------------
    def grid_labels():
        labels = []
        pairs = []
        for min_pct in min_percentiles:
            for max_pct in max_percentiles:
                if not ((max_pct == 100) or (min_pct == 0)):
                    continue
                labels.append(f"min{min_pct}_max{max_pct}")
                pairs.append((int(min_pct), int(max_pct)))
        return labels, pairs

    LABELS, GRID = grid_labels()

    def find_csvs_for_dataset(dataset):
        patt = os.path.join(base_dir, dataset, "min*_max*", "*_xnorm_metrics_min*_max*.csv")
        return sorted(glob.glob(patt))

    def mean_std(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        if x.size == 0:
            return np.nan, np.nan
        return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else np.nan

    def load_category_long_df(datasets, metric_base):
        """
        Returns perturbation-level DF with:
          dataset, min_pct, max_pct, perturbation,
          <metric>_rr, <metric>_rf, <metric>_fr, <metric>_ff

        Filters to your grid subset and collapses duplicates per perturbation.
        """
        need_cols = {"dataset","min_pct","max_pct","perturbation",
                     f"{metric_base}_rr", f"{metric_base}_rf", f"{metric_base}_fr", f"{metric_base}_ff"}

        dfs = []
        for ds in datasets:
            fps = find_csvs_for_dataset(ds)
            for fp in fps:
                try:
                    d = pd.read_csv(fp)
                except Exception:
                    continue

                if not need_cols.issubset(d.columns):
                    continue

                d = d.copy()
                d["min_pct"] = pd.to_numeric(d["min_pct"], errors="coerce")
                d["max_pct"] = pd.to_numeric(d["max_pct"], errors="coerce")
                d = d[(d["max_pct"] == 100) | (d["min_pct"] == 0)]
                if len(d) == 0:
                    continue

                # numeric metrics
                for tag in ["rr","rf","fr","ff"]:
                    c = f"{metric_base}_{tag}"
                    d[c] = pd.to_numeric(d[c], errors="coerce")

                # collapse duplicates: dataset × min/max × perturbation
                d = (
                    d.groupby(["dataset","min_pct","max_pct","perturbation"], as_index=False)
                     [[f"{metric_base}_{t}" for t in ["rr","rf","fr","ff"]]]
                     .mean()
                )

                dfs.append(d)

        if len(dfs) == 0:
            return pd.DataFrame(columns=list(need_cols))

        return pd.concat(dfs, ignore_index=True)

    def summarize_category(df_long, metric_base):
        """
        Pooled across datasets within category:
          mean/std computed over perturbations at each grid point.
        """
        out = {k: np.full(len(GRID), np.nan) for k in ["rr_m","rr_s","rf_m","rf_s","fr_m","fr_s","ff_m","ff_s"]}

        for gi, (min_pct, max_pct) in enumerate(GRID):
            dsub = df_long[(df_long["min_pct"] == min_pct) & (df_long["max_pct"] == max_pct)]
            if len(dsub) == 0:
                continue

            for tag in ["rr","rf","fr","ff"]:
                col = f"{metric_base}_{tag}"
                m, s = mean_std(dsub[col])
                out[f"{tag}_m"][gi] = m
                out[f"{tag}_s"][gi] = s

        return out

    def metric_ylim(metric_base):
        if metric_base == "R2":
            return (-0.25, 1.0)
        return (-1.0, 1.0)

    def plot_category_summary(title, labels, stats, metric_base, out_prefix):
        x = np.arange(len(labels))
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        ax.errorbar(x, stats["rr_m"], yerr=stats["rr_s"], fmt="o-", lw=2, capsize=3, markersize=5,
                    label="Σ_raw + Δx_raw (rr)")
        ax.errorbar(x, stats["rf_m"], yerr=stats["rf_s"], fmt="o-", lw=2, capsize=3, markersize=5,
                    label="Σ_raw + Δx_freq (rf)")
        ax.errorbar(x, stats["fr_m"], yerr=stats["fr_s"], fmt="o-", lw=2, capsize=3, markersize=5,
                    label="Σ_freq + Δx_raw (fr)")
        ax.errorbar(x, stats["ff_m"], yerr=stats["ff_s"], fmt="o-", lw=2, capsize=3, markersize=5,
                    label="Σ_freq + Δx_freq (ff)")

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(f"Mean {metric_base} ± STD (pooled perturbations)")
        ax.set_ylim(*metric_ylim(metric_base))
        ax.grid(False)
        ax.legend(frameon=False, fontsize=10)

        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{out_prefix}.png")
        out_svg = os.path.join(out_dir, f"{out_prefix}.svg")
        plt.savefig(out_png, dpi=200)
        plt.savefig(out_svg)
        plt.show()

        print(f"[SAVED] {out_png}")
        print(f"[SAVED] {out_svg}")

    def run_category(metric_base, category_name, datasets):
        df_long = load_category_long_df(datasets, metric_base)
        stats = summarize_category(df_long, metric_base)
        plot_category_summary(
            title=f"{metric_base} cross-normalization ({category_name})",
            labels=LABELS,
            stats=stats,
            metric_base=metric_base,
            out_prefix=f"{metric_base.lower()}_xnorm_category_{category_name}_mean_std",
        )
        return df_long

    # -----------------------------
    # Run all metrics × categories
    # -----------------------------
    for metric in METRICS:
        print(f"\n==============================\nMETRIC: {metric}\n==============================")

        df_a = run_category(metric, "CRISPRa", CRISPRa_DATASETS)
        df_i = run_category(metric, "CRISPRi", CRISPRi_DATASETS)
        df_all = run_category(metric, "ALL", ALL_DATASETS)

        print(f"[INFO] {metric} CRISPRa pooled perturbations: {len(df_a)} rows (pert-level).")
        print(f"[INFO] {metric} CRISPRi pooled perturbations: {len(df_i)} rows (pert-level).")
        print(f"[INFO] {metric} ALL pooled perturbations:     {len(df_all)} rows (pert-level).")

    print("\n[INFO] Done.")


def plot_pearson_rr_3panel():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # PEARSON CROSS-NORMALIZATION CATEGORY SUMMARY (rr only)
    #   - ONE VERTICAL 3-PANEL FIGURE (CRISPRa / CRISPRi / ALL)
    #   - Font size = 20 everywhere
    #   - Relabeled x-axis:
    #       bottom 70%, bottom 80%, bottom 90%, full, top 90%, top 80%, top 70%
    #   - No grid lines
    #
    # Reads:
    #   raw_freq_crossnorm_summary/<dataset>/minXXX_maxYYY/<dataset>_xnorm_metrics_minXXX_maxYYY.csv
    #
    # Saves:
    #   raw_freq_crossnorm_summary/category_summary_plots_pearson_rr_only/
    #     Pearson_rr__CRISPRa_CRISPRi_ALL__3panel_vertical.(png/svg)
    # ============================================================

    import os, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # -----------------------------
    # GLOBAL STYLE
    # -----------------------------
    FS = 20
    plt.rcParams.update({
        "axes.grid": False,
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
    })

    # -----------------------------
    # CONFIG
    # -----------------------------
    base_dir = os.path.join(OUTDIR, "raw_freq_crossnorm_summary")
    out_dir  = os.path.join(base_dir, "category_summary_plots_pearson_rr_only")
    os.makedirs(out_dir, exist_ok=True)

    min_percentiles = [0, 10, 20, 30]
    max_percentiles = [70, 80, 90, 100]

    ALL_DATASETS = [
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "GSE264667_jurkat_raw_singlecell_01",
        "GSE264667_hepg2_raw_singlecell_01",
        "NormanWeissman2019_filtered",
        "FrangiehIzar2021_RNA",
        "TianKampmann2019_day7neuron",
        "TianKampmann2021_CRISPRi",
        "TianKampmann2021_CRISPRa",
        "TianKampmann2019_iPSC"
    ]

    CRISPRa_DATASETS = ["TianKampmann2021_CRISPRa", "NormanWeissman2019_filtered"]
    CRISPRi_DATASETS = [
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "GSE264667_jurkat_raw_singlecell_01",
        "GSE264667_hepg2_raw_singlecell_01",
        "FrangiehIzar2021_RNA",
        "TianKampmann2019_day7neuron",
        "TianKampmann2021_CRISPRi",
        "TianKampmann2019_iPSC"
    ]

    METRIC = "Pearson"  # ONLY

    # -----------------------------
    # Helpers
    # -----------------------------
    def grid_pairs():
        pairs = []
        for min_pct in min_percentiles:
            for max_pct in max_percentiles:
                if (max_pct == 100) or (min_pct == 0):
                    pairs.append((int(min_pct), int(max_pct)))
        return pairs

    GRID = grid_pairs()

    def pretty_grid_labels(grid):
        labels = []
        for min_pct, max_pct in grid:
            if min_pct == 0 and max_pct < 100:
                labels.append(f"bottom {max_pct}%")
            elif min_pct == 0 and max_pct == 100:
                labels.append("full")
            elif min_pct > 0 and max_pct == 100:
                labels.append(f"top {100 - min_pct}%")
            else:
                labels.append(f"min{min_pct}_max{max_pct}")
        return labels

    XTICK_LABELS = pretty_grid_labels(GRID)

    def find_csvs_for_dataset(dataset):
        patt = os.path.join(base_dir, dataset, "min*_max*", "*_xnorm_metrics_min*_max*.csv")
        return sorted(glob.glob(patt))

    def mean_std(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        if x.size == 0:
            return np.nan, np.nan
        return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else np.nan

    def load_category_long_df(datasets):
        need_cols = {"dataset","min_pct","max_pct","perturbation",f"{METRIC}_rr"}
        dfs = []

        for ds in datasets:
            for fp in find_csvs_for_dataset(ds):
                try:
                    d = pd.read_csv(fp)
                except Exception:
                    continue
                if not need_cols.issubset(d.columns):
                    continue

                d = d.copy()
                d["min_pct"] = pd.to_numeric(d["min_pct"], errors="coerce")
                d["max_pct"] = pd.to_numeric(d["max_pct"], errors="coerce")
                d = d[(d["max_pct"] == 100) | (d["min_pct"] == 0)]
                if len(d) == 0:
                    continue

                c = f"{METRIC}_rr"
                d[c] = pd.to_numeric(d[c], errors="coerce")

                # collapse duplicates: dataset × min/max × perturbation
                d = (
                    d.groupby(["dataset","min_pct","max_pct","perturbation"], as_index=False)[[c]]
                     .mean()
                )
                dfs.append(d)

        if len(dfs) == 0:
            return pd.DataFrame(columns=list(need_cols))

        return pd.concat(dfs, ignore_index=True)

    def summarize_rr(df_long):
        rr_m = np.full(len(GRID), np.nan)
        rr_s = np.full(len(GRID), np.nan)

        for i, (min_pct, max_pct) in enumerate(GRID):
            dsub = df_long[(df_long["min_pct"] == min_pct) & (df_long["max_pct"] == max_pct)]
            if len(dsub) == 0:
                continue
            rr_m[i], rr_s[i] = mean_std(dsub[f"{METRIC}_rr"])

        return rr_m, rr_s, int(df_long.shape[0])

    def get_stats_for_category(datasets):
        df = load_category_long_df(datasets)
        rr_m, rr_s, nrows = summarize_rr(df)
        return rr_m, rr_s, nrows

    # -----------------------------
    # Compute stats
    # -----------------------------
    rrA_m, rrA_s, nA = get_stats_for_category(CRISPRa_DATASETS)
    rrI_m, rrI_s, nI = get_stats_for_category(CRISPRi_DATASETS)
    rrAll_m, rrAll_s, nAll = get_stats_for_category(ALL_DATASETS)

    # -----------------------------
    # Plot: vertical 3 panels
    # -----------------------------
    x = np.arange(len(GRID))
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True, sharey=True)

    panels = [
        ("CRISPRa", rrA_m, rrA_s, nA),
        ("CRISPRi", rrI_m, rrI_s, nI),
        ("ALL",     rrAll_m, rrAll_s, nAll),
    ]

    for ax, (name, rr_m, rr_s, nrows) in zip(axes, panels):
        ax.errorbar(
            x, rr_m, yerr=rr_s,
            fmt="o-", lw=2.5, capsize=4, markersize=6,
            label="Σ_raw + Δx_raw"
        )
        ax.set_title(f"{name}")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Mean Pearson ± STD")
        ax.grid(False)
        ax.legend(frameon=False, loc="lower right")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(XTICK_LABELS, rotation=45, ha="right")
    axes[-1].set_xlabel("Gene filter")

    plt.tight_layout()

    png = os.path.join(out_dir, "Pearson_rr__CRISPRa_CRISPRi_ALL__3panel_vertical.png")
    svg = os.path.join(out_dir, "Pearson_rr__CRISPRa_CRISPRi_ALL__3panel_vertical.svg")
    plt.savefig(png, dpi=250, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.show()

    print("[SAVED]", png)
    print("[SAVED]", svg)
    print("\n[INFO] Done.")


def plot_big_figure_final():
    global DATA_DIR, SUPPL, OUTDIR
    # ============================================================
    # ONE BIG FIGURE (FINAL TWEAK):
    #    KDE panels: bottom + left ticks only
    #    RR panels: ADD y-axis ticks (left) with same "dash" style
    #    font size = 20 everywhere, no grids
    # ============================================================

    import os, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # -----------------------------
    # GLOBAL STYLE
    # -----------------------------
    FS = 20
    plt.rcParams.update({
        "axes.grid": False,
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
    })
    sns.set(style="white", context="talk")
    plt.rcParams["axes.grid"] = False

    # -----------------------------
    # GROUP DEFINITIONS
    # -----------------------------
    CRISPRa_DATASETS = ["TianKampmann2021_CRISPRa", "NormanWeissman2019_filtered"]
    CRISPRi_DATASETS = [
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "GSE264667_jurkat_raw_singlecell_01",
        "GSE264667_hepg2_raw_singlecell_01",
        "FrangiehIzar2021_RNA",
        "TianKampmann2019_day7neuron",
        "TianKampmann2021_CRISPRi",
        "TianKampmann2019_iPSC",
    ]
    ALL_DATASETS = sorted(list(set(CRISPRa_DATASETS + CRISPRi_DATASETS)))

    # -----------------------------
    # INPUTS
    # -----------------------------
    OUT_ROOT_KDE = "raw_partial_freq_percentile_summary"
    MIN_PCT_KDE, MAX_PCT_KDE = 0, 100

    base_dir_rr = "raw_freq_crossnorm_summary"
    out_dir_rr  = os.path.join(base_dir_rr, "category_summary_plots_pearson_rr_only")
    os.makedirs(out_dir_rr, exist_ok=True)

    min_percentiles = [0, 10, 20, 30]
    max_percentiles = [70, 80, 90, 100]
    METRIC = "Pearson"

    # -----------------------------
    # Helpers
    # -----------------------------
    def grid_pairs():
        return [(int(mn), int(mx))
                for mn in min_percentiles
                for mx in max_percentiles
                if (mx == 100) or (mn == 0)]

    GRID = grid_pairs()

    def pretty_grid_labels(grid):
        out = []
        for mn, mx in grid:
            if mn == 0 and mx < 100:
                out.append(f"bottom {mx}%")
            elif mn == 0 and mx == 100:
                out.append("full")
            else:
                out.append(f"top {100-mn}%")
        return out

    XTICK_LABELS = pretty_grid_labels(GRID)

    def add_bottom_left_ticks(ax):
        ax.tick_params(axis="both", which="both",
                       direction="in",
                       length=7, width=1.5,
                       bottom=True, left=True,
                       top=False, right=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)

    # -----------------------------
    # Load KDE data
    # -----------------------------
    def load_kde_df():
        dfs = []
        need = {"dataset","min_pct","max_pct","Pearson_raw"}
        for fp in glob.glob(os.path.join(OUT_ROOT_KDE, "**", "*_metrics_min*_max*.csv"),
                            recursive=True):
            try:
                d = pd.read_csv(fp)
            except Exception:
                continue
            if not need.issubset(d.columns):
                continue
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)
        df = df[(df.min_pct == MIN_PCT_KDE) & (df.max_pct == MAX_PCT_KDE)].copy()
        df["Pearson_raw"] = pd.to_numeric(df["Pearson_raw"], errors="coerce").clip(-1, 1)
        return df.dropna(subset=["Pearson_raw"])

    def subset(df, datasets):
        return df[df.dataset.isin(datasets)]

    # -----------------------------
    # Load RR data
    # -----------------------------
    def load_rr_long_df(datasets):
        dfs = []
        for ds in datasets:
            for fp in glob.glob(os.path.join(base_dir_rr, ds, "min*_max*", "*_xnorm_metrics_min*_max*.csv")):
                try:
                    d = pd.read_csv(fp)
                except Exception:
                    continue
                if not {"min_pct","max_pct","perturbation",f"{METRIC}_rr"}.issubset(d.columns):
                    continue
                d = d[((d.max_pct == 100) | (d.min_pct == 0))].copy()
                d[f"{METRIC}_rr"] = pd.to_numeric(d[f"{METRIC}_rr"], errors="coerce")
                dfs.append(
                    d.groupby(["min_pct","max_pct","perturbation"], as_index=False)[f"{METRIC}_rr"].mean()
                )
        return pd.concat(dfs, ignore_index=True)

    def summarize_rr(df):
        m = np.full(len(GRID), np.nan)
        s = np.full(len(GRID), np.nan)
        for i, (mn, mx) in enumerate(GRID):
            vals = df[(df.min_pct == mn) & (df.max_pct == mx)][f"{METRIC}_rr"]
            vals = vals.dropna()
            if len(vals):
                m[i] = vals.mean()
                s[i] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        return m, s

    # -----------------------------
    # Build datasets
    # -----------------------------
    kde_df = load_kde_df()
    kde_A, kde_I, kde_ALL = subset(kde_df, CRISPRa_DATASETS), subset(kde_df, CRISPRi_DATASETS), kde_df

    rrA_m, rrA_s = summarize_rr(load_rr_long_df(CRISPRa_DATASETS))
    rrI_m, rrI_s = summarize_rr(load_rr_long_df(CRISPRi_DATASETS))
    rrAll_m, rrAll_s = summarize_rr(load_rr_long_df(ALL_DATASETS))

    # -----------------------------
    # Plot
    # -----------------------------
    fig = plt.figure(figsize=(22, 22))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.1, 1.2, 1.2, 1.2], hspace=0.65)

    def kde_panel(ax, data, title):
        sns.kdeplot(data["Pearson_raw"], ax=ax, fill=True, alpha=0.28, lw=2.5, clip=(-1, 1))
        ax.set_title(title)
        ax.set_xlim(-0.5, 1.0)
        ax.set_xlabel("Pearson correlation")
        ax.set_ylabel("Density")
        add_bottom_left_ticks(ax)

    kde_panel(fig.add_subplot(gs[0,0]), kde_A,   "CRISPRa")
    kde_panel(fig.add_subplot(gs[0,1]), kde_I,   "CRISPRi")
    kde_panel(fig.add_subplot(gs[0,2]), kde_ALL, "ALL")

    def rr_panel(ax, title, m, s):
        x = np.arange(len(GRID))
        ax.errorbar(x, m, yerr=s, fmt="o-", lw=2.5, capsize=4)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean Pearson ± STD")
        ax.set_xticks(x)
        ax.set_xticklabels(XTICK_LABELS, rotation=45, ha="right")
        add_bottom_left_ticks(ax)   # <-- THIS is the key change

    rr_panel(fig.add_subplot(gs[1,:]), "CRISPRa", rrA_m, rrA_s)
    rr_panel(fig.add_subplot(gs[2,:]), "CRISPRi", rrI_m, rrI_s)
    ax_last = fig.add_subplot(gs[3,:])
    rr_panel(ax_last, "ALL", rrAll_m, rrAll_s)
    ax_last.set_xlabel("Gene filter")

    out_png = os.path.join(out_dir_rr, "big_figure__pearson_kde_and_rr_summary.png")
    out_svg = os.path.join(out_dir_rr, "big_figure__pearson_kde_and_rr_summary.svg")
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.show()

    print("[SAVED]", out_png)
    print("[SAVED]", out_svg)


