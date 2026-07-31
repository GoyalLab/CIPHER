"""Run module for Fig S7 (covariance-vs-correlation coordinates + high-variance-filtered forward prediction).

Notebook-only driver module: each function is one relocated main-flow cell from
``notebooks/suppl/figS7_covariance_correlation.ipynb``, moved here VERBATIM so the
notebook stays a thin driver. NOT part of the installable ``cipher`` package.

Config (DATA_DIR, SUPPL, OUTDIR and any UPPERCASE globals) is injected into this
module's namespace at runtime from the notebook config cell; the functions read those
as module globals. Per-section config that the original cells defined inline stays inline.
"""
from src.suppl_covcorr import *

import os, re, gc, json
from pathlib import Path
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def covcorr_variant1_raw_preserving():
    PRECOMPUTE_ROOT = os.path.join(SUPPL, "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED")

    EXPRESSION_THRESHOLD = 1.0

    DATASET_FOLDERS = None

    OUT_SCORE_CSV = "cipher_forward_PEARSON_TRAINTEST_CORRCOORD_TRUE_AND_MF.csv"

    OUT_SCORE_NPZ = "cipher_forward_PEARSON_TRAINTEST_CORRCOORD_TRUE_AND_MF.npz"

    OUT_SCORE_JSON = "cipher_forward_PEARSON_TRAINTEST_CORRCOORD_TRUE_AND_MF_metadata.json"

    OUT_MERGED_CSV = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_CORRCOORD_TRUE_AND_MF_merged.csv"

    OUT_DATASET_MEANS_CSV = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_CORRCOORD_TRUE_AND_MF_dataset_means.csv"

    OUT_FIG_PNG = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_CORRCOORD_TRUE_AND_MF_composite_MEDIAN.png"

    OUT_FIG_SVG = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_CORRCOORD_TRUE_AND_MF_composite_MEDIAN.svg"

    OUT_SUMMARY_JSON = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_CORRCOORD_TRUE_AND_MF_summary.json"

    TRAIN_FRAC = 0.5

    N_SPLITS = 5

    SPLIT_SEED = 0

    BATCH_SIZE = 64

    EPS = 1e-12

    TQDM_NCOLS = 110

    N_HIST_BINS = 90

    FIGSIZE = (13, 5)

    DPI = 300

    USE_QUANTILE_XLIM = True

    LOW_Q = 0.005

    HIGH_Q = 0.995

    CLIP_HIST_TO_XLIM = True

    COLOR_TRUE = "#1f77b4"

    COLOR_MF = "#ff7f0e"

    COLOR_PAIR = "0.72"

    COLOR_MEDIAN_MARKER_EDGE = "black"

    TRUE_SIGMA_CANDIDATES = [
        "Sigma_full_ridge.npy",
        "Sigma_true_ridge.npy",
        "Sigma_full.npy",
        "Sigma_true.npy",
    ]

    MEANFIELD_SIGMA_CANDIDATES = [
        "Sigma_meanfield_ridge.npy",
        "Sigma_mean_field_ridge.npy",
        "Sigma_mf_ridge.npy",
        "Sigma_MF_ridge.npy",
        "Sigma_full_meanfield_ridge.npy",
        "Sigma_full_mean_field_ridge.npy",
        "Sigma_full_mf_ridge.npy",
        "Sigma_shuffle_ridge.npy",
        "Sigma_shuffled_ridge.npy",
        "Sigma_full_shuffle_ridge.npy",
        "Sigma_full_shuffled_ridge.npy",
        "Sigma_meanfield.npy",
        "Sigma_mean_field.npy",
        "Sigma_mf.npy",
        "Sigma_MF.npy",
        "Sigma_full_meanfield.npy",
        "Sigma_full_mean_field.npy",
        "Sigma_shuffle.npy",
        "Sigma_shuffled.npy",
    ]

    MEANFIELD_GLOB_PATTERNS = [
        "*meanfield*.npy",
        "*mean_field*.npy",
        "*MeanField*.npy",
        "*MF*.npy",
        "*mf*.npy",
        "*shuffle*.npy",
        "*shuffled*.npy",
    ]

    def process_one_folder(folder):
        folder = Path(folder)
        dataset = folder.name

        print("\n" + "=" * 110)
        print(f"[dataset] {dataset}")
        print(f"[folder]  {folder}")
        print("=" * 110)

        genes_path = folder / "genes.npy"
        perts_path = folder / "perturbations.npy"
        stats_path = folder / "perturbation_stats.h5"
        sigdir = folder / "sigmas"

        required_base = [
            genes_path,
            perts_path,
            stats_path,
            sigdir,
        ]

        missing = [str(x) for x in required_base if not x.exists()]
        if missing:
            raise FileNotFoundError("Missing required files/folders:\n" + "\n".join(missing))

        sigma_true_path = find_sigma_path(
            sigdir=sigdir,
            candidates=TRUE_SIGMA_CANDIDATES,
            glob_patterns=None,
            exclude_paths=None,
            label="true/full",
        )

        sigma_mf_path = find_sigma_path(
            sigdir=sigdir,
            candidates=MEANFIELD_SIGMA_CANDIDATES,
            glob_patterns=MEANFIELD_GLOB_PATTERNS,
            exclude_paths=[sigma_true_path],
            label="mean-field",
        )

        genes = decode_str_array(np.load(genes_path, allow_pickle=True))
        perts = decode_str_array(np.load(perts_path, allow_pickle=True))

        p = len(genes)
        n_perts = len(perts)

        target_genes, target_idx, matched = load_target_indices(folder, perts, genes)

        keep_pert_idx = np.flatnonzero(matched).astype(np.int64)
        matched_perts = perts[keep_pert_idx]
        matched_targets = target_genes[keep_pert_idx]
        matched_target_idx = target_idx[keep_pert_idx]

        n_match = len(keep_pert_idx)

        print(f"[load] genes={p:,}, perts={n_perts:,}, matched={n_match:,}")

        if n_match == 0:
            raise ValueError("No matched perturbation targets found.")

        Sigma_true = np.load(sigma_true_path, mmap_mode="r")
        Sigma_mf = np.load(sigma_mf_path, mmap_mode="r")

        if Sigma_true.shape != (p, p):
            raise ValueError(f"Sigma_true shape {Sigma_true.shape}, expected {(p, p)}")

        if Sigma_mf.shape != (p, p):
            raise ValueError(f"Sigma_mf shape {Sigma_mf.shape}, expected {(p, p)}")

        sd_true = get_sigma_sd(Sigma_true, eps=EPS)
        sd_mf = get_sigma_sd(Sigma_mf, eps=EPS)

        splits = make_gene_splits(
            p=p,
            n_splits=N_SPLITS,
            train_frac=TRAIN_FRAC,
            seed=SPLIT_SEED,
        )

        rows_out = []

        with h5py.File(stats_path, "r") as h5:
            dx_ds = h5["dx"]

            if dx_ds.shape != (n_perts, p):
                raise ValueError(f"dx shape {dx_ds.shape}, expected {(n_perts, p)}")

            if "n_cells_pert" in h5:
                n_cells_all = np.asarray(h5["n_cells_pert"][:], dtype=np.int64)
                n_cells_match = n_cells_all[keep_pert_idx]
            else:
                n_cells_match = np.full(n_match, -1, dtype=np.int64)

            for start in tqdm(
                range(0, n_match, BATCH_SIZE),
                desc=f"{dataset}: corr-coordinate train/test Pearson batches",
                ncols=TQDM_NCOLS,
            ):
                end = min(start + BATCH_SIZE, n_match)

                pert_rows = keep_pert_idx[start:end]
                gidx = matched_target_idx[start:end]

                dx_raw = np.asarray(dx_ds[pert_rows, :], dtype=np.float64)

                Sigma_basis_true = full_sigma_columns(Sigma_true, gidx)
                Sigma_basis_mf = full_sigma_columns(Sigma_mf, gidx)

                R_basis_true = corr_columns_from_sigma(Sigma_true, gidx, sd_true, eps=EPS)
                R_basis_mf = corr_columns_from_sigma(Sigma_mf, gidx, sd_mf, eps=EPS)

                for split_obj in splits:
                    split_id = split_obj["split"]
                    train_idx = split_obj["train_idx"]
                    test_idx = split_obj["test_idx"]

                    fit_true = weighted_beta_fit_corrcoords(
                        dx_raw=dx_raw,
                        R_basis=R_basis_true,
                        Sigma_basis=Sigma_basis_true,
                        sd=sd_true,
                        target_gene_idx=gidx,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        eps=EPS,
                    )

                    fit_mf = weighted_beta_fit_corrcoords(
                        dx_raw=dx_raw,
                        R_basis=R_basis_mf,
                        Sigma_basis=Sigma_basis_mf,
                        sd=sd_mf,
                        target_gene_idx=gidx,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        eps=EPS,
                    )

                    for local_i in range(end - start):
                        global_i = start + local_i
                        gi = int(matched_target_idx[global_i])

                        rows_out.append(
                            {
                                "dataset": dataset,
                                "expression_threshold": float(EXPRESSION_THRESHOLD),
                                "coordinate_system": "correlation",
                                "fit_inner_product": "weighted_by_sd_squared_to_preserve_raw_LS",
                                "split": int(split_id),
                                "train_frac": float(TRAIN_FRAC),
                                "n_train_genes": int(len(train_idx)),
                                "n_test_genes": int(len(test_idx)),
                                "perturbation": str(matched_perts[global_i]),
                                "target_gene": str(matched_targets[global_i]),
                                "target_gene_index": gi,
                                "n_cells_pert": int(n_cells_match[global_i]),

                                "sd_true_target": float(sd_true[gi]) if np.isfinite(sd_true[gi]) else np.nan,
                                "sd_mf_target": float(sd_mf[gi]) if np.isfinite(sd_mf[gi]) else np.nan,

                                # true/full Sigma, correlation-coordinate fit
                                "beta_true_train": float(fit_true["beta"][local_i]),
                                "alpha_true_train_from_beta": float(fit_true["alpha"][local_i]),

                                "pearson_true_train_raw": float(fit_true["pearson_raw_train"][local_i]),
                                "pearson_true_test_raw": float(fit_true["pearson_raw_test"][local_i]),
                                "train_mse_true_raw": float(fit_true["mse_raw_train"][local_i]),
                                "test_mse_true_raw": float(fit_true["mse_raw_test"][local_i]),

                                "pearson_true_train_z": float(fit_true["pearson_z_train"][local_i]),
                                "pearson_true_test_z": float(fit_true["pearson_z_test"][local_i]),
                                "train_mse_true_z": float(fit_true["mse_z_train"][local_i]),
                                "test_mse_true_z": float(fit_true["mse_z_test"][local_i]),

                                # mean-field Sigma, correlation-coordinate fit
                                "beta_mf_train": float(fit_mf["beta"][local_i]),
                                "alpha_mf_train_from_beta": float(fit_mf["alpha"][local_i]),

                                "pearson_mf_train_raw": float(fit_mf["pearson_raw_train"][local_i]),
                                "pearson_mf_test_raw": float(fit_mf["pearson_raw_test"][local_i]),
                                "train_mse_mf_raw": float(fit_mf["mse_raw_train"][local_i]),
                                "test_mse_mf_raw": float(fit_mf["mse_raw_test"][local_i]),

                                "pearson_mf_train_z": float(fit_mf["pearson_z_train"][local_i]),
                                "pearson_mf_test_z": float(fit_mf["pearson_z_test"][local_i]),
                                "train_mse_mf_z": float(fit_mf["mse_z_train"][local_i]),
                                "test_mse_mf_z": float(fit_mf["mse_z_test"][local_i]),
                            }
                        )

                del dx_raw
                del Sigma_basis_true, Sigma_basis_mf
                del R_basis_true, R_basis_mf
                gc.collect()

        out = pd.DataFrame(rows_out)

        csv_path = folder / OUT_SCORE_CSV
        out.to_csv(csv_path, index=False)

        npz_path = folder / OUT_SCORE_NPZ
        np.savez_compressed(
            npz_path,
            dataset=np.asarray(dataset, dtype=object),
            expression_threshold=np.asarray(EXPRESSION_THRESHOLD, dtype=np.float64),
            rows=out.to_records(index=False),
        )

        meta = {
            "dataset": dataset,
            "folder": str(folder),
            "expression_threshold": float(EXPRESSION_THRESHOLD),
            "n_genes": int(p),
            "n_perturbations_total": int(n_perts),
            "n_perturbations_matched": int(n_match),
            "n_perturbations_unmatched": int(n_perts - n_match),
            "train_frac": float(TRAIN_FRAC),
            "n_splits": int(N_SPLITS),
            "split_seed": int(SPLIT_SEED),
            "definition": (
                "Gene-held-out Pearson using normalized correlation coordinates. "
                "D=diag(sqrt(Sigma_ii)), R=D^{-1} Sigma D^{-1}, dz=D^{-1} dx. "
                "For single-gene perturbation g, dz ≈ beta_g R[:,g], with beta_g=alpha_g sd_g. "
                "Beta is fit with the D^2-weighted inner product in z-space, which exactly preserves "
                "the original raw-space least-squares alpha. Raw-space Pearson is therefore directly "
                "comparable to the old covariance-coordinate code. Z-space Pearson is also saved."
            ),
            "median_test_pearson_raw": {
                "true": float(np.nanmedian(out["pearson_true_test_raw"])),
                "mean_field": float(np.nanmedian(out["pearson_mf_test_raw"])),
            },
            "median_test_pearson_z": {
                "true": float(np.nanmedian(out["pearson_true_test_z"])),
                "mean_field": float(np.nanmedian(out["pearson_mf_test_z"])),
            },
            "files": {
                "csv": str(csv_path),
                "npz": str(npz_path),
                "sigma_true": str(sigma_true_path),
                "sigma_mean_field": str(sigma_mf_path),
                "stats_h5": str(stats_path),
            },
        }

        json_path = folder / OUT_SCORE_JSON

        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2, default=json_default)

        print(f"[saved] {csv_path}")
        print(f"[summary median held-out raw Pearson true] {np.nanmedian(out['pearson_true_test_raw']):.4f}")
        print(f"[summary median held-out raw Pearson MF]   {np.nanmedian(out['pearson_mf_test_raw']):.4f}")
        print(f"[summary median held-out z Pearson true]   {np.nanmedian(out['pearson_true_test_z']):.4f}")
        print(f"[summary median held-out z Pearson MF]     {np.nanmedian(out['pearson_mf_test_z']):.4f}")

        del Sigma_true, Sigma_mf
        del sd_true, sd_mf
        gc.collect()

        return {
            "dataset": dataset,
            "folder": str(folder),
            "csv": str(csv_path),
            "npz": str(npz_path),
            "n_genes": int(p),
            "n_matched": int(n_match),
            "median_pearson_true_test_raw": float(np.nanmedian(out["pearson_true_test_raw"])),
            "median_pearson_mf_test_raw": float(np.nanmedian(out["pearson_mf_test_raw"])),
            "median_pearson_true_test_z": float(np.nanmedian(out["pearson_true_test_z"])),
            "median_pearson_mf_test_z": float(np.nanmedian(out["pearson_mf_test_z"])),
        }

    root = Path(PRECOMPUTE_ROOT)
    OUTROOT = Path(OUTDIR)
    OUTROOT.mkdir(parents=True, exist_ok=True)

    if DATASET_FOLDERS is None:
        folders = find_dataset_folders(root, EXPRESSION_THRESHOLD)
    else:
        folders = [Path(x) for x in DATASET_FOLDERS]

    print(f"[run] found {len(folders)} precomputed dataset folders")

    if len(folders) == 0:
        raise FileNotFoundError(
            f"No folders found under {PRECOMPUTE_ROOT} for EXPRESSION_THRESHOLD={EXPRESSION_THRESHOLD}"
        )

    all_results = []

    all_errors = []

    for folder in tqdm(folders, desc="datasets", ncols=TQDM_NCOLS):
        try:
            res = process_one_folder(folder)
            all_results.append(res)
        except Exception as e:
            print("\n" + "!" * 110)
            print(f"[ERROR] {folder}")
            print(repr(e))
            print("!" * 110 + "\n")
            all_errors.append({"folder": str(folder), "error": repr(e)})
            gc.collect()

    threshold_tag = threshold_to_tag(EXPRESSION_THRESHOLD)

    score_files = sorted(root.glob(f"*__mean_ge_{threshold_tag}/{OUT_SCORE_CSV}"))

    print(f"[plot] found {len(score_files)} corr-coordinate train/test files")

    dfs = []

    for path in tqdm(score_files, desc="loading corr-coordinate train/test files", ncols=TQDM_NCOLS):
        df = pd.read_csv(path)

        if "dataset" not in df.columns:
            df["dataset"] = Path(path).parent.name

        df["expression_threshold"] = float(EXPRESSION_THRESHOLD)
        df["source_folder"] = str(Path(path).parent)
        df["source_file"] = str(path)

        numeric_cols = [
            "beta_true_train",
            "alpha_true_train_from_beta",
            "pearson_true_train_raw",
            "pearson_true_test_raw",
            "train_mse_true_raw",
            "test_mse_true_raw",
            "pearson_true_train_z",
            "pearson_true_test_z",
            "train_mse_true_z",
            "test_mse_true_z",

            "beta_mf_train",
            "alpha_mf_train_from_beta",
            "pearson_mf_train_raw",
            "pearson_mf_test_raw",
            "train_mse_mf_raw",
            "test_mse_mf_raw",
            "pearson_mf_train_z",
            "pearson_mf_test_z",
            "train_mse_mf_z",
            "test_mse_mf_z",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        dfs.append(df)

    if len(dfs) == 0:
        summary_path = OUTROOT / OUT_SUMMARY_JSON

        summary = {
            "precompute_root": str(root),
            "expression_threshold": float(EXPRESSION_THRESHOLD),
            "threshold_tag": threshold_tag,
            "n_folders": int(len(folders)),
            "n_success": int(len(all_results)),
            "n_errors": int(len(all_errors)),
            "results": all_results,
            "errors": all_errors,
            "message": "No corr-coordinate Pearson files produced. All datasets failed or were skipped.",
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=json_default)

        print("\n" + "=" * 110)
        print("NO CORR-COORDINATE PEARSON FILES WERE PRODUCED.")
        print(f"successful datasets: {len(all_results)}")
        print(f"errored datasets:    {len(all_errors)}")
        print(f"summary json:        {summary_path}")
        print("\nErrors:")
        for err in all_errors:
            print(f"- {err['folder']}")
            print(f"  {err['error']}")
        print("=" * 110)

        raise SystemExit

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    finite_mask = (
        np.isfinite(all_df["pearson_true_test_raw"].to_numpy(float))
        & np.isfinite(all_df["pearson_mf_test_raw"].to_numpy(float))
    )

    all_df = all_df.loc[finite_mask].copy()

    if len(all_df) == 0:
        raise RuntimeError("Files were produced, but all true/MF raw test Pearson values were non-finite.")

    merged_path = OUTROOT / OUT_MERGED_CSV

    all_df.to_csv(merged_path, index=False)

    print(f"[saved] {merged_path}")

    dataset_means = (
        all_df
        .groupby("dataset", dropna=False)
        .agg(
            n_rows=("perturbation", "count"),
            n_perturbations=("perturbation", "nunique"),

            pearson_true_train_raw_mean=("pearson_true_train_raw", "mean"),
            pearson_true_test_raw_mean=("pearson_true_test_raw", "mean"),
            pearson_true_test_raw_median=("pearson_true_test_raw", "median"),

            pearson_mf_train_raw_mean=("pearson_mf_train_raw", "mean"),
            pearson_mf_test_raw_mean=("pearson_mf_test_raw", "mean"),
            pearson_mf_test_raw_median=("pearson_mf_test_raw", "median"),

            pearson_true_train_z_mean=("pearson_true_train_z", "mean"),
            pearson_true_test_z_mean=("pearson_true_test_z", "mean"),
            pearson_true_test_z_median=("pearson_true_test_z", "median"),

            pearson_mf_train_z_mean=("pearson_mf_train_z", "mean"),
            pearson_mf_test_z_mean=("pearson_mf_test_z", "mean"),
            pearson_mf_test_z_median=("pearson_mf_test_z", "median"),

            beta_true_median=("beta_true_train", "median"),
            beta_mf_median=("beta_mf_train", "median"),
            alpha_true_median=("alpha_true_train_from_beta", "median"),
            alpha_mf_median=("alpha_mf_train_from_beta", "median"),
        )
        .reset_index()
    )

    dataset_means["pearson_true_minus_mf_raw_median"] = (
        dataset_means["pearson_true_test_raw_median"] - dataset_means["pearson_mf_test_raw_median"]
    )

    dataset_means["pearson_true_minus_mf_z_median"] = (
        dataset_means["pearson_true_test_z_median"] - dataset_means["pearson_mf_test_z_median"]
    )

    dataset_means_path = OUTROOT / OUT_DATASET_MEANS_CSV

    dataset_means.to_csv(dataset_means_path, index=False)

    print(f"[saved] {dataset_means_path}")

    print("\n[dataset summaries]")

    print(
        dataset_means[
            [
                "dataset",
                "n_perturbations",
                "pearson_mf_test_raw_median",
                "pearson_true_test_raw_median",
                "pearson_true_minus_mf_raw_median",
                "pearson_mf_test_z_median",
                "pearson_true_test_z_median",
                "pearson_true_minus_mf_z_median",
            ]
        ].to_string(index=False)
    )

    hist_true = all_df["pearson_true_test_raw"].to_numpy(float)

    hist_mf = all_df["pearson_mf_test_raw"].to_numpy(float)

    hist_true = hist_true[np.isfinite(hist_true)]

    hist_mf = hist_mf[np.isfinite(hist_mf)]

    hist_true_median = safe_nanmedian(hist_true)

    hist_mf_median = safe_nanmedian(hist_mf)

    hist_all = np.concatenate([hist_true, hist_mf])

    if USE_QUANTILE_XLIM:
        xmin = float(np.nanquantile(hist_all, LOW_Q))
        xmax = float(np.nanquantile(hist_all, HIGH_Q))
    else:
        xmin = float(np.nanmin(hist_all))
        xmax = float(np.nanmax(hist_all))

    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0

    pad = 0.05 * max(xmax - xmin, 1e-9)

    xmin -= pad

    xmax += pad

    bins = np.linspace(xmin, xmax, N_HIST_BINS + 1)

    hist_true_plot = np.clip(hist_true, xmin, xmax) if CLIP_HIST_TO_XLIM else hist_true

    hist_mf_plot = np.clip(hist_mf, xmin, xmax) if CLIP_HIST_TO_XLIM else hist_mf

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    ax_hist, ax_box = axes

    ax_hist.hist(
        hist_mf_plot,
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLOR_MF,
        label=f"mean-field R\nmedian={hist_mf_median:.3g}",
    )

    ax_hist.hist(
        hist_true_plot,
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLOR_TRUE,
        label=f"real/full R\nmedian={hist_true_median:.3g}",
    )

    ax_hist.axvline(hist_mf_median, linestyle="--", linewidth=2.0, color=COLOR_MF)

    ax_hist.axvline(hist_true_median, linestyle="--", linewidth=2.0, color=COLOR_TRUE)

    ax_hist.axvline(0.0, linestyle=":", linewidth=1.5, color="black", alpha=0.75)

    ax_hist.set_xlim(xmin, xmax)

    ax_hist.set_xlabel(
        r"Held-out Pearson in raw space: $\rho(\Delta x_{\rm test}, \hat\alpha \Sigma_{{\rm test},g})$"
    )

    ax_hist.set_ylabel("probability density")

    ax_hist.set_title(
        f"Raw-space held-out Pearson\n"
        f"fit done in correlation coordinates; threshold={EXPRESSION_THRESHOLD}; splits={N_SPLITS}"
    )

    ax_hist.legend(frameon=False)

    y_mf = dataset_means["pearson_mf_test_raw_median"].to_numpy(float)

    y_true = dataset_means["pearson_true_test_raw_median"].to_numpy(float)

    dataset_mf_median = safe_nanmedian(y_mf)

    dataset_true_median = safe_nanmedian(y_true)

    rng = np.random.default_rng(0)

    x_mf = np.ones_like(y_mf) * 1.0

    x_true = np.ones_like(y_true) * 2.0

    x_mf_jit = x_mf + rng.normal(0, 0.035, size=len(x_mf))

    x_true_jit = x_true + rng.normal(0, 0.035, size=len(x_true))

    for ym, yt in zip(y_mf, y_true):
        if np.isfinite(ym) and np.isfinite(yt):
            ax_box.plot(
                [1.0, 2.0],
                [ym, yt],
                linewidth=0.8,
                alpha=0.35,
                color=COLOR_PAIR,
                zorder=1,
            )

    ax_box.scatter(
        x_mf_jit,
        y_mf,
        s=45,
        alpha=0.75,
        color=COLOR_MF,
        edgecolor="none",
        zorder=2,
    )

    ax_box.scatter(
        x_true_jit,
        y_true,
        s=45,
        alpha=0.75,
        color=COLOR_TRUE,
        edgecolor="none",
        zorder=2,
    )

    ax_box.scatter(
        [1.0],
        [dataset_mf_median],
        s=190,
        color=COLOR_MF,
        edgecolor=COLOR_MEDIAN_MARKER_EDGE,
        linewidth=1.3,
        marker="s",
        zorder=5,
        label=f"MF median={dataset_mf_median:.3g}",
    )

    ax_box.scatter(
        [2.0],
        [dataset_true_median],
        s=190,
        color=COLOR_TRUE,
        edgecolor=COLOR_MEDIAN_MARKER_EDGE,
        linewidth=1.3,
        marker="D",
        zorder=5,
        label=f"real/full median={dataset_true_median:.3g}",
    )

    ax_box.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    ax_box.set_xlim(0.55, 2.45)

    ax_box.set_xticks([1.0, 2.0])

    ax_box.set_xticklabels(["mean-field R", "real/full R"])

    ax_box.set_ylabel("median held-out raw Pearson per dataset")

    ax_box.set_title(f"Dataset-median held-out Pearson\nn={len(dataset_means):,} datasets")

    ax_box.grid(axis="y", alpha=0.25)

    ax_box.legend(frameon=False)

    fig.suptitle(
        "CIPHER forward problem in correlation coordinates: same raw predictions, beta-scaled alphas",
        fontsize=15,
        y=1.03,
    )

    plt.tight_layout()

    png_path = OUTROOT / OUT_FIG_PNG

    svg_path = OUTROOT / OUT_FIG_SVG

    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")

    plt.savefig(svg_path, bbox_inches="tight")

    plt.show()

    print(f"[saved] {png_path}")

    print(f"[saved] {svg_path}")

    summary = {
        "precompute_root": str(root),
        "expression_threshold": float(EXPRESSION_THRESHOLD),
        "threshold_tag": threshold_tag,
        "n_folders": int(len(folders)),
        "n_success": int(len(all_results)),
        "n_errors": int(len(all_errors)),
        "results": all_results,
        "errors": all_errors,
        "merged_csv": str(merged_path),
        "dataset_means_csv": str(dataset_means_path),
        "figure_png": str(png_path),
        "figure_svg": str(svg_path),
        "overall": {
            "perturbation_split_level": {
                "pearson_true_train_raw": summarize(all_df["pearson_true_train_raw"]),
                "pearson_true_test_raw": summarize(all_df["pearson_true_test_raw"]),
                "pearson_mf_train_raw": summarize(all_df["pearson_mf_train_raw"]),
                "pearson_mf_test_raw": summarize(all_df["pearson_mf_test_raw"]),
                "pearson_true_minus_mf_test_raw": summarize(
                    all_df["pearson_true_test_raw"].to_numpy(float)
                    - all_df["pearson_mf_test_raw"].to_numpy(float)
                ),

                "pearson_true_train_z": summarize(all_df["pearson_true_train_z"]),
                "pearson_true_test_z": summarize(all_df["pearson_true_test_z"]),
                "pearson_mf_train_z": summarize(all_df["pearson_mf_train_z"]),
                "pearson_mf_test_z": summarize(all_df["pearson_mf_test_z"]),
                "pearson_true_minus_mf_test_z": summarize(
                    all_df["pearson_true_test_z"].to_numpy(float)
                    - all_df["pearson_mf_test_z"].to_numpy(float)
                ),

                "beta_true_train": summarize(all_df["beta_true_train"]),
                "beta_mf_train": summarize(all_df["beta_mf_train"]),
                "alpha_true_train_from_beta": summarize(all_df["alpha_true_train_from_beta"]),
                "alpha_mf_train_from_beta": summarize(all_df["alpha_mf_train_from_beta"]),
            },
            "dataset_level": {
                "pearson_true_test_raw_median_per_dataset": summarize(dataset_means["pearson_true_test_raw_median"]),
                "pearson_mf_test_raw_median_per_dataset": summarize(dataset_means["pearson_mf_test_raw_median"]),
                "pearson_true_minus_mf_raw_median_per_dataset": summarize(dataset_means["pearson_true_minus_mf_raw_median"]),
                "pearson_true_test_z_median_per_dataset": summarize(dataset_means["pearson_true_test_z_median"]),
                "pearson_mf_test_z_median_per_dataset": summarize(dataset_means["pearson_mf_test_z_median"]),
                "pearson_true_minus_mf_z_median_per_dataset": summarize(dataset_means["pearson_true_minus_mf_z_median"]),
            },
        },
        "config": {
            "train_frac": float(TRAIN_FRAC),
            "n_splits": int(N_SPLITS),
            "split_seed": int(SPLIT_SEED),
            "batch_size": int(BATCH_SIZE),
            "primary_plot_metric": "raw-space held-out Pearson after fitting beta in correlation coordinates",
            "coordinate_change": {
                "D": "diag(sqrt(Sigma_ii))",
                "R": "D^{-1} Sigma D^{-1}",
                "dz": "D^{-1} dx",
                "beta": "alpha * sd_g",
                "alpha": "beta / sd_g",
            },
            "fit_definition": (
                "Fit beta in correlation coordinates with weights sd_i^2 over train genes. "
                "This preserves the original raw-space least-squares alpha exactly. "
                "Then evaluate held-out Pearson in raw space using alpha Sigma[:,g], "
                "and also save standardized z-space Pearson using beta R[:,g]."
            ),
            "plot_order": ["mean-field R", "real/full R"],
            "colors": {
                "mean_field": COLOR_MF,
                "real_full": COLOR_TRUE,
            },
        },
    }

    summary_path = OUTROOT / OUT_SUMMARY_JSON

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print("\n" + "=" * 110)

    print("DONE")

    print(f"successful datasets: {len(all_results)}")

    print(f"errored datasets:    {len(all_errors)}")

    print(f"merged csv:          {merged_path}")

    print(f"dataset summaries csv:{dataset_means_path}")

    print(f"summary json:        {summary_path}")

    print("=" * 110)


def covcorr_variant2_ordinary():
    PRECOMPUTE_ROOT = os.path.join(SUPPL, "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED")

    EXPRESSION_THRESHOLD = 1.0

    DATASET_FOLDERS = None

    OUT_SCORE_CSV = "cipher_forward_ORDINARY_CORRSPACE_TRUE_AND_MF.csv"

    OUT_SCORE_NPZ = "cipher_forward_ORDINARY_CORRSPACE_TRUE_AND_MF.npz"

    OUT_SCORE_JSON = "cipher_forward_ORDINARY_CORRSPACE_TRUE_AND_MF_metadata.json"

    OUT_MERGED_CSV = "ALL_DATASETS__cipher_forward_ORDINARY_CORRSPACE_TRUE_AND_MF_merged.csv"

    OUT_DATASET_MEANS_CSV = "ALL_DATASETS__cipher_forward_ORDINARY_CORRSPACE_TRUE_AND_MF_dataset_means.csv"

    OUT_FIG_PNG = "ALL_DATASETS__cipher_forward_ORDINARY_CORRSPACE_TRUE_AND_MF_composite_MEDIAN.png"

    OUT_FIG_SVG = "ALL_DATASETS__cipher_forward_ORDINARY_CORRSPACE_TRUE_AND_MF_composite_MEDIAN.svg"

    OUT_SUMMARY_JSON = "ALL_DATASETS__cipher_forward_ORDINARY_CORRSPACE_TRUE_AND_MF_summary.json"

    TRAIN_FRAC = 0.5

    N_SPLITS = 5

    SPLIT_SEED = 0

    BATCH_SIZE = 64

    EPS = 1e-12

    TQDM_NCOLS = 110

    N_HIST_BINS = 90

    FIGSIZE = (13, 5)

    DPI = 300

    COLOR_TRUE = "#1f77b4"

    COLOR_MF = "#ff7f0e"

    COLOR_PAIR = "0.72"

    TRUE_SIGMA_CANDIDATES = [
        "Sigma_full_ridge.npy",
        "Sigma_true_ridge.npy",
        "Sigma_full.npy",
        "Sigma_true.npy",
    ]

    MEANFIELD_SIGMA_CANDIDATES = [
        "Sigma_meanfield_ridge.npy",
        "Sigma_mean_field_ridge.npy",
        "Sigma_mf_ridge.npy",
        "Sigma_MF_ridge.npy",
        "Sigma_full_meanfield_ridge.npy",
        "Sigma_full_mean_field_ridge.npy",
        "Sigma_full_mf_ridge.npy",
        "Sigma_shuffle_ridge.npy",
        "Sigma_shuffled_ridge.npy",
        "Sigma_full_shuffle_ridge.npy",
        "Sigma_full_shuffled_ridge.npy",
        "Sigma_meanfield.npy",
        "Sigma_mean_field.npy",
        "Sigma_mf.npy",
        "Sigma_MF.npy",
        "Sigma_full_meanfield.npy",
        "Sigma_full_mean_field.npy",
        "Sigma_shuffle.npy",
        "Sigma_shuffled.npy",
    ]

    MEANFIELD_GLOB_PATTERNS = [
        "*meanfield*.npy",
        "*mean_field*.npy",
        "*MeanField*.npy",
        "*MF*.npy",
        "*mf*.npy",
        "*shuffle*.npy",
        "*shuffled*.npy",
    ]

    def process_one_folder(folder):
        folder = Path(folder)
        dataset = folder.name

        print("\n" + "=" * 110)
        print(f"[dataset] {dataset}")
        print(f"[folder]  {folder}")
        print("=" * 110)

        genes_path = folder / "genes.npy"
        perts_path = folder / "perturbations.npy"
        stats_path = folder / "perturbation_stats.h5"
        sigdir = folder / "sigmas"

        required_base = [genes_path, perts_path, stats_path, sigdir]
        missing = [str(x) for x in required_base if not x.exists()]

        if missing:
            raise FileNotFoundError("Missing required files/folders:\n" + "\n".join(missing))

        sigma_true_path = find_sigma_path(
            sigdir=sigdir,
            candidates=TRUE_SIGMA_CANDIDATES,
            glob_patterns=None,
            exclude_paths=None,
            label="true/full",
        )

        sigma_mf_path = find_sigma_path(
            sigdir=sigdir,
            candidates=MEANFIELD_SIGMA_CANDIDATES,
            glob_patterns=MEANFIELD_GLOB_PATTERNS,
            exclude_paths=[sigma_true_path],
            label="mean-field",
        )

        genes = decode_str_array(np.load(genes_path, allow_pickle=True))
        perts = decode_str_array(np.load(perts_path, allow_pickle=True))

        p = len(genes)
        n_perts = len(perts)

        target_genes, target_idx, matched = load_target_indices(folder, perts, genes)

        keep_pert_idx = np.flatnonzero(matched).astype(np.int64)
        matched_perts = perts[keep_pert_idx]
        matched_targets = target_genes[keep_pert_idx]
        matched_target_idx = target_idx[keep_pert_idx]

        n_match = len(keep_pert_idx)

        print(f"[load] genes={p:,}, perts={n_perts:,}, matched={n_match:,}")

        if n_match == 0:
            raise ValueError("No matched perturbation targets found.")

        Sigma_true = np.load(sigma_true_path, mmap_mode="r")
        Sigma_mf = np.load(sigma_mf_path, mmap_mode="r")

        if Sigma_true.shape != (p, p):
            raise ValueError(f"Sigma_true shape {Sigma_true.shape}, expected {(p, p)}")

        if Sigma_mf.shape != (p, p):
            raise ValueError(f"Sigma_mf shape {Sigma_mf.shape}, expected {(p, p)}")

        sd_true = get_sigma_sd(Sigma_true, eps=EPS)
        sd_mf = get_sigma_sd(Sigma_mf, eps=EPS)

        splits = make_gene_splits(
            p=p,
            n_splits=N_SPLITS,
            train_frac=TRAIN_FRAC,
            seed=SPLIT_SEED,
        )

        rows_out = []

        with h5py.File(stats_path, "r") as h5:
            dx_ds = h5["dx"]

            if dx_ds.shape != (n_perts, p):
                raise ValueError(f"dx shape {dx_ds.shape}, expected {(n_perts, p)}")

            if "n_cells_pert" in h5:
                n_cells_all = np.asarray(h5["n_cells_pert"][:], dtype=np.int64)
                n_cells_match = n_cells_all[keep_pert_idx]
            else:
                n_cells_match = np.full(n_match, -1, dtype=np.int64)

            for start in tqdm(
                range(0, n_match, BATCH_SIZE),
                desc=f"{dataset}: ordinary corr-space batches",
                ncols=TQDM_NCOLS,
            ):
                end = min(start + BATCH_SIZE, n_match)

                pert_rows = keep_pert_idx[start:end]
                gidx = matched_target_idx[start:end]

                dx_raw = np.asarray(dx_ds[pert_rows, :], dtype=np.float64)

                R_basis_true = corr_columns_from_sigma(Sigma_true, gidx, sd_true, eps=EPS)
                R_basis_mf = corr_columns_from_sigma(Sigma_mf, gidx, sd_mf, eps=EPS)

                for split_obj in splits:
                    split_id = split_obj["split"]
                    train_idx = split_obj["train_idx"]
                    test_idx = split_obj["test_idx"]

                    fit_true = ordinary_beta_fit_corrspace(
                        dx_raw=dx_raw,
                        R_basis=R_basis_true,
                        sd=sd_true,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        eps=EPS,
                    )

                    fit_mf = ordinary_beta_fit_corrspace(
                        dx_raw=dx_raw,
                        R_basis=R_basis_mf,
                        sd=sd_mf,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        eps=EPS,
                    )

                    for local_i in range(end - start):
                        global_i = start + local_i
                        gi = int(matched_target_idx[global_i])

                        rows_out.append(
                            {
                                "dataset": dataset,
                                "expression_threshold": float(EXPRESSION_THRESHOLD),
                                "coordinate_system": "ordinary_correlation_space",
                                "objective": "min_beta_sum_i_dz_minus_beta_Rig_squared",
                                "split": int(split_id),
                                "train_frac": float(TRAIN_FRAC),
                                "n_train_genes": int(len(train_idx)),
                                "n_test_genes": int(len(test_idx)),
                                "perturbation": str(matched_perts[global_i]),
                                "target_gene": str(matched_targets[global_i]),
                                "target_gene_index": gi,
                                "n_cells_pert": int(n_cells_match[global_i]),

                                "sd_true_target": float(sd_true[gi]) if np.isfinite(sd_true[gi]) else np.nan,
                                "sd_mf_target": float(sd_mf[gi]) if np.isfinite(sd_mf[gi]) else np.nan,

                                "beta_true_train": float(fit_true["beta"][local_i]),
                                "pearson_true_train_z": float(fit_true["pearson_z_train"][local_i]),
                                "pearson_true_test_z": float(fit_true["pearson_z_test"][local_i]),
                                "train_mse_true_z": float(fit_true["mse_z_train"][local_i]),
                                "test_mse_true_z": float(fit_true["mse_z_test"][local_i]),

                                "pearson_true_train_raw_reconstructed": float(fit_true["pearson_raw_train"][local_i]),
                                "pearson_true_test_raw_reconstructed": float(fit_true["pearson_raw_test"][local_i]),
                                "train_mse_true_raw_reconstructed": float(fit_true["mse_raw_train"][local_i]),
                                "test_mse_true_raw_reconstructed": float(fit_true["mse_raw_test"][local_i]),

                                "beta_mf_train": float(fit_mf["beta"][local_i]),
                                "pearson_mf_train_z": float(fit_mf["pearson_z_train"][local_i]),
                                "pearson_mf_test_z": float(fit_mf["pearson_z_test"][local_i]),
                                "train_mse_mf_z": float(fit_mf["mse_z_train"][local_i]),
                                "test_mse_mf_z": float(fit_mf["mse_z_test"][local_i]),

                                "pearson_mf_train_raw_reconstructed": float(fit_mf["pearson_raw_train"][local_i]),
                                "pearson_mf_test_raw_reconstructed": float(fit_mf["pearson_raw_test"][local_i]),
                                "train_mse_mf_raw_reconstructed": float(fit_mf["mse_raw_train"][local_i]),
                                "test_mse_mf_raw_reconstructed": float(fit_mf["mse_raw_test"][local_i]),
                            }
                        )

                del dx_raw, R_basis_true, R_basis_mf
                gc.collect()

        out = pd.DataFrame(rows_out)

        csv_path = folder / OUT_SCORE_CSV
        out.to_csv(csv_path, index=False)

        npz_path = folder / OUT_SCORE_NPZ
        np.savez_compressed(
            npz_path,
            dataset=np.asarray(dataset, dtype=object),
            expression_threshold=np.asarray(EXPRESSION_THRESHOLD, dtype=np.float64),
            rows=out.to_records(index=False),
        )

        meta = {
            "dataset": dataset,
            "folder": str(folder),
            "expression_threshold": float(EXPRESSION_THRESHOLD),
            "n_genes": int(p),
            "n_perturbations_total": int(n_perts),
            "n_perturbations_matched": int(n_match),
            "train_frac": float(TRAIN_FRAC),
            "n_splits": int(N_SPLITS),
            "split_seed": int(SPLIT_SEED),
            "definition": (
                "Ordinary correlation-space forward fit. "
                "R=D^{-1} Sigma D^{-1}; dz=D^{-1} dx. "
                "Fit beta on train genes using unweighted OLS: "
                "beta=<dz_train,R_train>/<R_train,R_train>. "
                "Evaluate primarily in z-space. Raw-space reconstructed metrics are also saved, "
                "but this fit is not equivalent to raw covariance CIPHER."
            ),
            "median_test_pearson_z": {
                "true": float(np.nanmedian(out["pearson_true_test_z"])),
                "mean_field": float(np.nanmedian(out["pearson_mf_test_z"])),
            },
            "median_test_pearson_raw_reconstructed": {
                "true": float(np.nanmedian(out["pearson_true_test_raw_reconstructed"])),
                "mean_field": float(np.nanmedian(out["pearson_mf_test_raw_reconstructed"])),
            },
            "files": {
                "csv": str(csv_path),
                "npz": str(npz_path),
                "sigma_true": str(sigma_true_path),
                "sigma_mean_field": str(sigma_mf_path),
                "stats_h5": str(stats_path),
            },
        }

        json_path = folder / OUT_SCORE_JSON

        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2, default=json_default)

        print(f"[saved] {csv_path}")
        print(f"[median z Pearson true] {np.nanmedian(out['pearson_true_test_z']):.4f}")
        print(f"[median z Pearson MF]   {np.nanmedian(out['pearson_mf_test_z']):.4f}")

        del Sigma_true, Sigma_mf, sd_true, sd_mf
        gc.collect()

        return {
            "dataset": dataset,
            "folder": str(folder),
            "csv": str(csv_path),
            "npz": str(npz_path),
            "n_genes": int(p),
            "n_matched": int(n_match),
            "median_pearson_true_test_z": float(np.nanmedian(out["pearson_true_test_z"])),
            "median_pearson_mf_test_z": float(np.nanmedian(out["pearson_mf_test_z"])),
            "median_pearson_true_test_raw_reconstructed": float(np.nanmedian(out["pearson_true_test_raw_reconstructed"])),
            "median_pearson_mf_test_raw_reconstructed": float(np.nanmedian(out["pearson_mf_test_raw_reconstructed"])),
        }

    root = Path(PRECOMPUTE_ROOT)
    OUTROOT = Path(OUTDIR)
    OUTROOT.mkdir(parents=True, exist_ok=True)

    if DATASET_FOLDERS is None:
        folders = find_dataset_folders(root, EXPRESSION_THRESHOLD)
    else:
        folders = [Path(x) for x in DATASET_FOLDERS]

    print(f"[run] found {len(folders)} precomputed dataset folders")

    if len(folders) == 0:
        raise FileNotFoundError(
            f"No folders found under {PRECOMPUTE_ROOT} for EXPRESSION_THRESHOLD={EXPRESSION_THRESHOLD}"
        )

    all_results = []

    all_errors = []

    for folder in tqdm(folders, desc="datasets", ncols=TQDM_NCOLS):
        try:
            res = process_one_folder(folder)
            all_results.append(res)
        except Exception as e:
            print("\n" + "!" * 110)
            print(f"[ERROR] {folder}")
            print(repr(e))
            print("!" * 110 + "\n")
            all_errors.append({"folder": str(folder), "error": repr(e)})
            gc.collect()

    threshold_tag = threshold_to_tag(EXPRESSION_THRESHOLD)

    score_files = sorted(root.glob(f"*__mean_ge_{threshold_tag}/{OUT_SCORE_CSV}"))

    print(f"[plot] found {len(score_files)} ordinary corr-space files")

    dfs = []

    for path in tqdm(score_files, desc="loading files", ncols=TQDM_NCOLS):
        df = pd.read_csv(path)

        if "dataset" not in df.columns:
            df["dataset"] = Path(path).parent.name

        df["expression_threshold"] = float(EXPRESSION_THRESHOLD)
        df["source_folder"] = str(Path(path).parent)
        df["source_file"] = str(path)

        for col in df.columns:
            if (
                col.startswith("pearson_")
                or col.startswith("train_mse_")
                or col.startswith("test_mse_")
                or col.startswith("beta_")
                or col.startswith("sd_")
            ):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        dfs.append(df)

    if len(dfs) == 0:
        raise RuntimeError("No ordinary corr-space output files produced.")

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    finite_mask = (
        np.isfinite(all_df["pearson_true_test_z"].to_numpy(float))
        & np.isfinite(all_df["pearson_mf_test_z"].to_numpy(float))
    )

    all_df = all_df.loc[finite_mask].copy()

    if len(all_df) == 0:
        raise RuntimeError("All z-space test Pearson values were non-finite.")

    merged_path = OUTROOT / OUT_MERGED_CSV

    all_df.to_csv(merged_path, index=False)

    print(f"[saved] {merged_path}")

    dataset_means = (
        all_df
        .groupby("dataset", dropna=False)
        .agg(
            n_rows=("perturbation", "count"),
            n_perturbations=("perturbation", "nunique"),

            pearson_true_test_z_mean=("pearson_true_test_z", "mean"),
            pearson_true_test_z_median=("pearson_true_test_z", "median"),
            pearson_mf_test_z_mean=("pearson_mf_test_z", "mean"),
            pearson_mf_test_z_median=("pearson_mf_test_z", "median"),

            pearson_true_test_raw_reconstructed_mean=("pearson_true_test_raw_reconstructed", "mean"),
            pearson_true_test_raw_reconstructed_median=("pearson_true_test_raw_reconstructed", "median"),
            pearson_mf_test_raw_reconstructed_mean=("pearson_mf_test_raw_reconstructed", "mean"),
            pearson_mf_test_raw_reconstructed_median=("pearson_mf_test_raw_reconstructed", "median"),

            beta_true_median=("beta_true_train", "median"),
            beta_mf_median=("beta_mf_train", "median"),
        )
        .reset_index()
    )

    dataset_means["pearson_true_minus_mf_z_median"] = (
        dataset_means["pearson_true_test_z_median"]
        - dataset_means["pearson_mf_test_z_median"]
    )

    dataset_means["pearson_true_minus_mf_raw_reconstructed_median"] = (
        dataset_means["pearson_true_test_raw_reconstructed_median"]
        - dataset_means["pearson_mf_test_raw_reconstructed_median"]
    )

    dataset_means_path = OUTROOT / OUT_DATASET_MEANS_CSV

    dataset_means.to_csv(dataset_means_path, index=False)

    print(f"[saved] {dataset_means_path}")

    print("\n[dataset summaries]")

    print(
        dataset_means[
            [
                "dataset",
                "n_perturbations",
                "pearson_mf_test_z_median",
                "pearson_true_test_z_median",
                "pearson_true_minus_mf_z_median",
                "pearson_mf_test_raw_reconstructed_median",
                "pearson_true_test_raw_reconstructed_median",
            ]
        ].to_string(index=False)
    )

    hist_true = all_df["pearson_true_test_z"].to_numpy(float)

    hist_mf = all_df["pearson_mf_test_z"].to_numpy(float)

    hist_true = hist_true[np.isfinite(hist_true)]

    hist_mf = hist_mf[np.isfinite(hist_mf)]

    hist_true_median = safe_nanmedian(hist_true)

    hist_mf_median = safe_nanmedian(hist_mf)

    hist_all = np.concatenate([hist_true, hist_mf])

    xmin = float(np.nanquantile(hist_all, 0.005))

    xmax = float(np.nanquantile(hist_all, 0.995))

    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0

    pad = 0.05 * max(xmax - xmin, 1e-9)

    xmin -= pad

    xmax += pad

    bins = np.linspace(xmin, xmax, N_HIST_BINS + 1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    ax_hist, ax_box = axes

    ax_hist.hist(
        np.clip(hist_mf, xmin, xmax),
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLOR_MF,
        label=f"mean-field R\nmedian={hist_mf_median:.3g}",
    )

    ax_hist.hist(
        np.clip(hist_true, xmin, xmax),
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLOR_TRUE,
        label=f"real/full R\nmedian={hist_true_median:.3g}",
    )

    ax_hist.axvline(hist_mf_median, linestyle="--", linewidth=2.0, color=COLOR_MF)

    ax_hist.axvline(hist_true_median, linestyle="--", linewidth=2.0, color=COLOR_TRUE)

    ax_hist.axvline(0.0, linestyle=":", linewidth=1.5, color="black", alpha=0.75)

    ax_hist.set_xlim(xmin, xmax)

    ax_hist.set_xlabel(
        r"Held-out Pearson in standardized space: $\rho(\Delta z_{\rm test}, \hat\beta R_{{\rm test},g})$"
    )

    ax_hist.set_ylabel("probability density")

    ax_hist.set_title("Ordinary correlation-space fit")

    ax_hist.legend(frameon=False)

    y_mf = dataset_means["pearson_mf_test_z_median"].to_numpy(float)

    y_true = dataset_means["pearson_true_test_z_median"].to_numpy(float)

    dataset_mf_median = safe_nanmedian(y_mf)

    dataset_true_median = safe_nanmedian(y_true)

    rng = np.random.default_rng(0)

    x_mf = np.ones_like(y_mf) * 1.0

    x_true = np.ones_like(y_true) * 2.0

    x_mf_jit = x_mf + rng.normal(0, 0.035, size=len(x_mf))

    x_true_jit = x_true + rng.normal(0, 0.035, size=len(x_true))

    for ym, yt in zip(y_mf, y_true):
        if np.isfinite(ym) and np.isfinite(yt):
            ax_box.plot(
                [1.0, 2.0],
                [ym, yt],
                linewidth=0.8,
                alpha=0.35,
                color=COLOR_PAIR,
                zorder=1,
            )

    ax_box.scatter(
        x_mf_jit,
        y_mf,
        s=45,
        alpha=0.75,
        color=COLOR_MF,
        edgecolor="none",
        zorder=2,
    )

    ax_box.scatter(
        x_true_jit,
        y_true,
        s=45,
        alpha=0.75,
        color=COLOR_TRUE,
        edgecolor="none",
        zorder=2,
    )

    ax_box.scatter(
        [1.0],
        [dataset_mf_median],
        s=190,
        color=COLOR_MF,
        edgecolor="black",
        linewidth=1.3,
        marker="s",
        zorder=5,
        label=f"MF median={dataset_mf_median:.3g}",
    )

    ax_box.scatter(
        [2.0],
        [dataset_true_median],
        s=190,
        color=COLOR_TRUE,
        edgecolor="black",
        linewidth=1.3,
        marker="D",
        zorder=5,
        label=f"real/full median={dataset_true_median:.3g}",
    )

    ax_box.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    ax_box.set_xlim(0.55, 2.45)

    ax_box.set_xticks([1.0, 2.0])

    ax_box.set_xticklabels(["mean-field R", "real/full R"])

    ax_box.set_ylabel("median held-out z-Pearson per dataset")

    ax_box.set_title(f"Dataset medians\nn={len(dataset_means):,} datasets")

    ax_box.grid(axis="y", alpha=0.25)

    ax_box.legend(frameon=False)

    fig.suptitle(
        "CIPHER-style forward problem with ordinary correlation-space least squares",
        fontsize=15,
        y=1.03,
    )

    plt.tight_layout()

    png_path = OUTROOT / OUT_FIG_PNG

    svg_path = OUTROOT / OUT_FIG_SVG

    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")

    plt.savefig(svg_path, bbox_inches="tight")

    plt.show()

    print(f"[saved] {png_path}")

    print(f"[saved] {svg_path}")

    summary = {
        "precompute_root": str(root),
        "expression_threshold": float(EXPRESSION_THRESHOLD),
        "threshold_tag": threshold_tag,
        "n_folders": int(len(folders)),
        "n_success": int(len(all_results)),
        "n_errors": int(len(all_errors)),
        "results": all_results,
        "errors": all_errors,
        "merged_csv": str(merged_path),
        "dataset_means_csv": str(dataset_means_path),
        "figure_png": str(png_path),
        "figure_svg": str(svg_path),
        "overall": {
            "perturbation_split_level": {
                "pearson_true_test_z": summarize(all_df["pearson_true_test_z"]),
                "pearson_mf_test_z": summarize(all_df["pearson_mf_test_z"]),
                "pearson_true_minus_mf_test_z": summarize(
                    all_df["pearson_true_test_z"].to_numpy(float)
                    - all_df["pearson_mf_test_z"].to_numpy(float)
                ),
                "pearson_true_test_raw_reconstructed": summarize(
                    all_df["pearson_true_test_raw_reconstructed"]
                ),
                "pearson_mf_test_raw_reconstructed": summarize(
                    all_df["pearson_mf_test_raw_reconstructed"]
                ),
                "beta_true_train": summarize(all_df["beta_true_train"]),
                "beta_mf_train": summarize(all_df["beta_mf_train"]),
            },
            "dataset_level": {
                "pearson_true_test_z_median_per_dataset": summarize(
                    dataset_means["pearson_true_test_z_median"]
                ),
                "pearson_mf_test_z_median_per_dataset": summarize(
                    dataset_means["pearson_mf_test_z_median"]
                ),
                "pearson_true_minus_mf_z_median_per_dataset": summarize(
                    dataset_means["pearson_true_minus_mf_z_median"]
                ),
            },
        },
        "config": {
            "objective": "ordinary unweighted correlation-space least squares",
            "model": "dz ≈ beta_g R[:,g]",
            "coordinate_change": {
                "sd_i": "sqrt(Sigma_ii)",
                "R_ij": "Sigma_ij / (sd_i sd_j)",
                "dz_i": "dx_i / sd_i",
            },
            "interpretation": (
                "This is linear response for standardized variables z, not the original "
                "raw expression variables x. It matches standardized shifts and therefore "
                "weights genes equally after variance normalization. It is not equivalent "
                "to the raw covariance CIPHER fit."
            ),
            "train_frac": float(TRAIN_FRAC),
            "n_splits": int(N_SPLITS),
            "split_seed": int(SPLIT_SEED),
        },
    }

    summary_path = OUTROOT / OUT_SUMMARY_JSON

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print("\n" + "=" * 110)

    print("DONE")

    print(f"successful datasets: {len(all_results)}")

    print(f"errored datasets:    {len(all_errors)}")

    print(f"merged csv:          {merged_path}")

    print(f"dataset summaries csv:{dataset_means_path}")

    print(f"summary json:        {summary_path}")

    print("=" * 110)


def covcorr_variant3_zonly():
    PRECOMPUTE_ROOT = os.path.join(SUPPL, "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED")

    EXPRESSION_THRESHOLD = 1.0

    DATASET_FOLDERS = None

    OUT_SCORE_CSV = "cipher_forward_ORDINARY_CORRSPACE_ZONLY_TRUE_AND_MF.csv"

    OUT_SCORE_NPZ = "cipher_forward_ORDINARY_CORRSPACE_ZONLY_TRUE_AND_MF.npz"

    OUT_SCORE_JSON = "cipher_forward_ORDINARY_CORRSPACE_ZONLY_TRUE_AND_MF_metadata.json"

    OUT_MERGED_CSV = "ALL_DATASETS__cipher_forward_ORDINARY_CORRSPACE_ZONLY_TRUE_AND_MF_merged.csv"

    OUT_DATASET_MEANS_CSV = "ALL_DATASETS__cipher_forward_ORDINARY_CORRSPACE_ZONLY_TRUE_AND_MF_dataset_means.csv"

    OUT_FIG_PNG = "ALL_DATASETS__cipher_forward_ORDINARY_CORRSPACE_ZONLY_TRUE_AND_MF_composite_MEDIAN.png"

    OUT_FIG_SVG = "ALL_DATASETS__cipher_forward_ORDINARY_CORRSPACE_ZONLY_TRUE_AND_MF_composite_MEDIAN.svg"

    OUT_SUMMARY_JSON = "ALL_DATASETS__cipher_forward_ORDINARY_CORRSPACE_ZONLY_TRUE_AND_MF_summary.json"

    TRAIN_FRAC = 0.5

    N_SPLITS = 5

    SPLIT_SEED = 0

    BATCH_SIZE = 64

    EPS = 1e-12

    TQDM_NCOLS = 110

    N_HIST_BINS = 90

    FIGSIZE = (13, 5)

    DPI = 300

    USE_QUANTILE_XLIM = True

    LOW_Q = 0.005

    HIGH_Q = 0.995

    CLIP_HIST_TO_XLIM = True

    COLOR_TRUE = "#1f77b4"

    COLOR_MF = "#ff7f0e"

    COLOR_PAIR = "0.72"

    COLOR_MEDIAN_MARKER_EDGE = "black"

    TRUE_SIGMA_CANDIDATES = [
        "Sigma_full_ridge.npy",
        "Sigma_true_ridge.npy",
        "Sigma_full.npy",
        "Sigma_true.npy",
    ]

    MEANFIELD_SIGMA_CANDIDATES = [
        "Sigma_meanfield_ridge.npy",
        "Sigma_mean_field_ridge.npy",
        "Sigma_mf_ridge.npy",
        "Sigma_MF_ridge.npy",
        "Sigma_full_meanfield_ridge.npy",
        "Sigma_full_mean_field_ridge.npy",
        "Sigma_full_mf_ridge.npy",
        "Sigma_shuffle_ridge.npy",
        "Sigma_shuffled_ridge.npy",
        "Sigma_full_shuffle_ridge.npy",
        "Sigma_full_shuffled_ridge.npy",
        "Sigma_meanfield.npy",
        "Sigma_mean_field.npy",
        "Sigma_mf.npy",
        "Sigma_MF.npy",
        "Sigma_full_meanfield.npy",
        "Sigma_full_mean_field.npy",
        "Sigma_shuffle.npy",
        "Sigma_shuffled.npy",
    ]

    MEANFIELD_GLOB_PATTERNS = [
        "*meanfield*.npy",
        "*mean_field*.npy",
        "*MeanField*.npy",
        "*MF*.npy",
        "*mf*.npy",
        "*shuffle*.npy",
        "*shuffled*.npy",
    ]

    def process_one_folder(folder):
        folder = Path(folder)
        dataset = folder.name

        print("\n" + "=" * 110)
        print(f"[dataset] {dataset}")
        print(f"[folder]  {folder}")
        print("=" * 110)

        genes_path = folder / "genes.npy"
        perts_path = folder / "perturbations.npy"
        stats_path = folder / "perturbation_stats.h5"
        sigdir = folder / "sigmas"

        required_base = [genes_path, perts_path, stats_path, sigdir]
        missing = [str(x) for x in required_base if not x.exists()]

        if missing:
            raise FileNotFoundError("Missing required files/folders:\n" + "\n".join(missing))

        sigma_true_path = find_sigma_path(
            sigdir=sigdir,
            candidates=TRUE_SIGMA_CANDIDATES,
            glob_patterns=None,
            exclude_paths=None,
            label="true/full",
        )

        sigma_mf_path = find_sigma_path(
            sigdir=sigdir,
            candidates=MEANFIELD_SIGMA_CANDIDATES,
            glob_patterns=MEANFIELD_GLOB_PATTERNS,
            exclude_paths=[sigma_true_path],
            label="mean-field",
        )

        genes = decode_str_array(np.load(genes_path, allow_pickle=True))
        perts = decode_str_array(np.load(perts_path, allow_pickle=True))

        p = len(genes)
        n_perts = len(perts)

        target_genes, target_idx, matched = load_target_indices(folder, perts, genes)

        keep_pert_idx = np.flatnonzero(matched).astype(np.int64)
        matched_perts = perts[keep_pert_idx]
        matched_targets = target_genes[keep_pert_idx]
        matched_target_idx = target_idx[keep_pert_idx]

        n_match = len(keep_pert_idx)

        print(f"[load] genes={p:,}, perts={n_perts:,}, matched={n_match:,}")

        if n_match == 0:
            raise ValueError("No matched perturbation targets found.")

        Sigma_true = np.load(sigma_true_path, mmap_mode="r")
        Sigma_mf = np.load(sigma_mf_path, mmap_mode="r")

        if Sigma_true.shape != (p, p):
            raise ValueError(f"Sigma_true shape {Sigma_true.shape}, expected {(p, p)}")

        if Sigma_mf.shape != (p, p):
            raise ValueError(f"Sigma_mf shape {Sigma_mf.shape}, expected {(p, p)}")

        sd_true = get_sigma_sd(Sigma_true, eps=EPS)
        sd_mf = get_sigma_sd(Sigma_mf, eps=EPS)

        splits = make_gene_splits(
            p=p,
            n_splits=N_SPLITS,
            train_frac=TRAIN_FRAC,
            seed=SPLIT_SEED,
        )

        rows_out = []

        with h5py.File(stats_path, "r") as h5:
            dx_ds = h5["dx"]

            if dx_ds.shape != (n_perts, p):
                raise ValueError(f"dx shape {dx_ds.shape}, expected {(n_perts, p)}")

            if "n_cells_pert" in h5:
                n_cells_all = np.asarray(h5["n_cells_pert"][:], dtype=np.int64)
                n_cells_match = n_cells_all[keep_pert_idx]
            else:
                n_cells_match = np.full(n_match, -1, dtype=np.int64)

            for start in tqdm(
                range(0, n_match, BATCH_SIZE),
                desc=f"{dataset}: ordinary corr-space z-only batches",
                ncols=TQDM_NCOLS,
            ):
                end = min(start + BATCH_SIZE, n_match)

                pert_rows = keep_pert_idx[start:end]
                gidx = matched_target_idx[start:end]

                dx_raw = np.asarray(dx_ds[pert_rows, :], dtype=np.float64)

                R_basis_true = corr_columns_from_sigma(Sigma_true, gidx, sd_true, eps=EPS)
                R_basis_mf = corr_columns_from_sigma(Sigma_mf, gidx, sd_mf, eps=EPS)

                for split_obj in splits:
                    split_id = split_obj["split"]
                    train_idx = split_obj["train_idx"]
                    test_idx = split_obj["test_idx"]

                    fit_true = ordinary_beta_fit_corrspace_zonly(
                        dx_raw=dx_raw,
                        R_basis=R_basis_true,
                        sd=sd_true,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        eps=EPS,
                    )

                    fit_mf = ordinary_beta_fit_corrspace_zonly(
                        dx_raw=dx_raw,
                        R_basis=R_basis_mf,
                        sd=sd_mf,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        eps=EPS,
                    )

                    for local_i in range(end - start):
                        global_i = start + local_i
                        gi = int(matched_target_idx[global_i])

                        rows_out.append(
                            {
                                "dataset": dataset,
                                "expression_threshold": float(EXPRESSION_THRESHOLD),
                                "coordinate_system": "ordinary_correlation_space_zonly",
                                "objective": "min_beta_sum_i_dz_minus_beta_Rig_squared",
                                "split": int(split_id),
                                "train_frac": float(TRAIN_FRAC),
                                "n_train_genes": int(len(train_idx)),
                                "n_test_genes": int(len(test_idx)),
                                "perturbation": str(matched_perts[global_i]),
                                "target_gene": str(matched_targets[global_i]),
                                "target_gene_index": gi,
                                "n_cells_pert": int(n_cells_match[global_i]),

                                "sd_true_target": float(sd_true[gi]) if np.isfinite(sd_true[gi]) else np.nan,
                                "sd_mf_target": float(sd_mf[gi]) if np.isfinite(sd_mf[gi]) else np.nan,

                                "beta_true_train": float(fit_true["beta"][local_i]),
                                "pearson_true_train_z": float(fit_true["pearson_z_train"][local_i]),
                                "pearson_true_test_z": float(fit_true["pearson_z_test"][local_i]),
                                "train_mse_true_z": float(fit_true["mse_z_train"][local_i]),
                                "test_mse_true_z": float(fit_true["mse_z_test"][local_i]),

                                "beta_mf_train": float(fit_mf["beta"][local_i]),
                                "pearson_mf_train_z": float(fit_mf["pearson_z_train"][local_i]),
                                "pearson_mf_test_z": float(fit_mf["pearson_z_test"][local_i]),
                                "train_mse_mf_z": float(fit_mf["mse_z_train"][local_i]),
                                "test_mse_mf_z": float(fit_mf["mse_z_test"][local_i]),
                            }
                        )

                del dx_raw, R_basis_true, R_basis_mf
                gc.collect()

        out = pd.DataFrame(rows_out)

        csv_path = folder / OUT_SCORE_CSV
        out.to_csv(csv_path, index=False)

        npz_path = folder / OUT_SCORE_NPZ
        np.savez_compressed(
            npz_path,
            dataset=np.asarray(dataset, dtype=object),
            expression_threshold=np.asarray(EXPRESSION_THRESHOLD, dtype=np.float64),
            rows=out.to_records(index=False),
        )

        json_path = folder / OUT_SCORE_JSON

        meta = {
            "dataset": dataset,
            "folder": str(folder),
            "expression_threshold": float(EXPRESSION_THRESHOLD),
            "n_genes": int(p),
            "n_perturbations_total": int(n_perts),
            "n_perturbations_matched": int(n_match),
            "train_frac": float(TRAIN_FRAC),
            "n_splits": int(N_SPLITS),
            "split_seed": int(SPLIT_SEED),
            "definition": (
                "Ordinary correlation-space forward fit, scored only in standardized z-space. "
                "R=D^{-1} Sigma D^{-1}; dz=D^{-1} dx. "
                "Fit beta on train genes using unweighted OLS: "
                "beta=<dz_train,R_train>/<R_train,R_train>. "
                "Evaluate Pearson and MSE only in z-space. "
                "No raw-space reconstruction is performed."
            ),
            "median_test_pearson_z": {
                "true": float(np.nanmedian(out["pearson_true_test_z"])),
                "mean_field": float(np.nanmedian(out["pearson_mf_test_z"])),
            },
            "files": {
                "csv": str(csv_path),
                "npz": str(npz_path),
                "sigma_true": str(sigma_true_path),
                "sigma_mean_field": str(sigma_mf_path),
                "stats_h5": str(stats_path),
            },
        }

        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2, default=json_default)

        print(f"[saved] {csv_path}")
        print(f"[median z Pearson true] {np.nanmedian(out['pearson_true_test_z']):.4f}")
        print(f"[median z Pearson MF]   {np.nanmedian(out['pearson_mf_test_z']):.4f}")

        del Sigma_true, Sigma_mf, sd_true, sd_mf
        gc.collect()

        return {
            "dataset": dataset,
            "folder": str(folder),
            "csv": str(csv_path),
            "npz": str(npz_path),
            "n_genes": int(p),
            "n_matched": int(n_match),
            "median_pearson_true_test_z": float(np.nanmedian(out["pearson_true_test_z"])),
            "median_pearson_mf_test_z": float(np.nanmedian(out["pearson_mf_test_z"])),
        }

    root = Path(PRECOMPUTE_ROOT)
    OUTROOT = Path(OUTDIR)
    OUTROOT.mkdir(parents=True, exist_ok=True)

    if DATASET_FOLDERS is None:
        folders = find_dataset_folders(root, EXPRESSION_THRESHOLD)
    else:
        folders = [Path(x) for x in DATASET_FOLDERS]

    print(f"[run] found {len(folders)} precomputed dataset folders")

    if len(folders) == 0:
        raise FileNotFoundError(
            f"No folders found under {PRECOMPUTE_ROOT} for EXPRESSION_THRESHOLD={EXPRESSION_THRESHOLD}"
        )

    all_results = []

    all_errors = []

    for folder in tqdm(folders, desc="datasets", ncols=TQDM_NCOLS):
        try:
            res = process_one_folder(folder)
            all_results.append(res)
        except Exception as e:
            print("\n" + "!" * 110)
            print(f"[ERROR] {folder}")
            print(repr(e))
            print("!" * 110 + "\n")
            all_errors.append({"folder": str(folder), "error": repr(e)})
            gc.collect()

    threshold_tag = threshold_to_tag(EXPRESSION_THRESHOLD)

    score_files = sorted(root.glob(f"*__mean_ge_{threshold_tag}/{OUT_SCORE_CSV}"))

    print(f"[plot] found {len(score_files)} ordinary corr-space z-only files")

    dfs = []

    for path in tqdm(score_files, desc="loading files", ncols=TQDM_NCOLS):
        df = pd.read_csv(path)

        if "dataset" not in df.columns:
            df["dataset"] = Path(path).parent.name

        df["expression_threshold"] = float(EXPRESSION_THRESHOLD)
        df["source_folder"] = str(Path(path).parent)
        df["source_file"] = str(path)

        for col in df.columns:
            if (
                col.startswith("pearson_")
                or col.startswith("train_mse_")
                or col.startswith("test_mse_")
                or col.startswith("beta_")
                or col.startswith("sd_")
            ):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        dfs.append(df)

    if len(dfs) == 0:
        summary_path = OUTROOT / OUT_SUMMARY_JSON

        summary = {
            "precompute_root": str(root),
            "expression_threshold": float(EXPRESSION_THRESHOLD),
            "threshold_tag": threshold_tag,
            "n_folders": int(len(folders)),
            "n_success": int(len(all_results)),
            "n_errors": int(len(all_errors)),
            "results": all_results,
            "errors": all_errors,
            "message": "No ordinary corr-space z-only output files produced.",
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=json_default)

        raise RuntimeError("No ordinary corr-space z-only output files produced.")

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    finite_mask = (
        np.isfinite(all_df["pearson_true_test_z"].to_numpy(float))
        & np.isfinite(all_df["pearson_mf_test_z"].to_numpy(float))
    )

    all_df = all_df.loc[finite_mask].copy()

    if len(all_df) == 0:
        raise RuntimeError("All z-space test Pearson values were non-finite.")

    merged_path = OUTROOT / OUT_MERGED_CSV

    all_df.to_csv(merged_path, index=False)

    print(f"[saved] {merged_path}")

    dataset_means = (
        all_df
        .groupby("dataset", dropna=False)
        .agg(
            n_rows=("perturbation", "count"),
            n_perturbations=("perturbation", "nunique"),

            pearson_true_train_z_mean=("pearson_true_train_z", "mean"),
            pearson_true_test_z_mean=("pearson_true_test_z", "mean"),
            pearson_true_test_z_median=("pearson_true_test_z", "median"),

            pearson_mf_train_z_mean=("pearson_mf_train_z", "mean"),
            pearson_mf_test_z_mean=("pearson_mf_test_z", "mean"),
            pearson_mf_test_z_median=("pearson_mf_test_z", "median"),

            train_mse_true_z_mean=("train_mse_true_z", "mean"),
            test_mse_true_z_mean=("test_mse_true_z", "mean"),

            train_mse_mf_z_mean=("train_mse_mf_z", "mean"),
            test_mse_mf_z_mean=("test_mse_mf_z", "mean"),

            beta_true_median=("beta_true_train", "median"),
            beta_mf_median=("beta_mf_train", "median"),
        )
        .reset_index()
    )

    dataset_means["pearson_true_minus_mf_z_mean"] = (
        dataset_means["pearson_true_test_z_mean"]
        - dataset_means["pearson_mf_test_z_mean"]
    )

    dataset_means["pearson_true_minus_mf_z_median"] = (
        dataset_means["pearson_true_test_z_median"]
        - dataset_means["pearson_mf_test_z_median"]
    )

    dataset_means_path = OUTROOT / OUT_DATASET_MEANS_CSV

    dataset_means.to_csv(dataset_means_path, index=False)

    print(f"[saved] {dataset_means_path}")

    print("\n[dataset summaries]")

    print(
        dataset_means[
            [
                "dataset",
                "n_perturbations",
                "pearson_mf_test_z_median",
                "pearson_true_test_z_median",
                "pearson_true_minus_mf_z_median",
                "pearson_mf_test_z_mean",
                "pearson_true_test_z_mean",
                "pearson_true_minus_mf_z_mean",
            ]
        ].to_string(index=False)
    )

    hist_true = all_df["pearson_true_test_z"].to_numpy(float)

    hist_mf = all_df["pearson_mf_test_z"].to_numpy(float)

    hist_true = hist_true[np.isfinite(hist_true)]

    hist_mf = hist_mf[np.isfinite(hist_mf)]

    hist_true_median = safe_nanmedian(hist_true)

    hist_mf_median = safe_nanmedian(hist_mf)

    hist_all = np.concatenate([hist_true, hist_mf])

    if USE_QUANTILE_XLIM:
        xmin = float(np.nanquantile(hist_all, LOW_Q))
        xmax = float(np.nanquantile(hist_all, HIGH_Q))
    else:
        xmin = float(np.nanmin(hist_all))
        xmax = float(np.nanmax(hist_all))

    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0

    pad = 0.05 * max(xmax - xmin, 1e-9)

    xmin -= pad

    xmax += pad

    bins = np.linspace(xmin, xmax, N_HIST_BINS + 1)

    hist_true_plot = np.clip(hist_true, xmin, xmax) if CLIP_HIST_TO_XLIM else hist_true

    hist_mf_plot = np.clip(hist_mf, xmin, xmax) if CLIP_HIST_TO_XLIM else hist_mf

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    ax_hist, ax_box = axes

    ax_hist.hist(
        hist_mf_plot,
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLOR_MF,
        label=f"mean-field R\nmedian={hist_mf_median:.3g}",
    )

    ax_hist.hist(
        hist_true_plot,
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLOR_TRUE,
        label=f"real/full R\nmedian={hist_true_median:.3g}",
    )

    ax_hist.axvline(hist_mf_median, linestyle="--", linewidth=2.0, color=COLOR_MF)

    ax_hist.axvline(hist_true_median, linestyle="--", linewidth=2.0, color=COLOR_TRUE)

    ax_hist.axvline(0.0, linestyle=":", linewidth=1.5, color="black", alpha=0.75)

    ax_hist.set_xlim(xmin, xmax)

    ax_hist.set_xlabel(
        r"Held-out Pearson in standardized space: $\rho(\Delta z_{\rm test}, \hat\beta R_{{\rm test},g})$"
    )

    ax_hist.set_ylabel("probability density")

    ax_hist.set_title(
        f"Ordinary correlation-space fit\n"
        f"threshold={EXPRESSION_THRESHOLD}; splits={N_SPLITS}"
    )

    ax_hist.legend(frameon=False)

    y_mf = dataset_means["pearson_mf_test_z_median"].to_numpy(float)

    y_true = dataset_means["pearson_true_test_z_median"].to_numpy(float)

    dataset_mf_median = safe_nanmedian(y_mf)

    dataset_true_median = safe_nanmedian(y_true)

    rng = np.random.default_rng(0)

    x_mf = np.ones_like(y_mf) * 1.0

    x_true = np.ones_like(y_true) * 2.0

    x_mf_jit = x_mf + rng.normal(0, 0.035, size=len(x_mf))

    x_true_jit = x_true + rng.normal(0, 0.035, size=len(x_true))

    for ym, yt in zip(y_mf, y_true):
        if np.isfinite(ym) and np.isfinite(yt):
            ax_box.plot(
                [1.0, 2.0],
                [ym, yt],
                linewidth=0.8,
                alpha=0.35,
                color=COLOR_PAIR,
                zorder=1,
            )

    ax_box.scatter(
        x_mf_jit,
        y_mf,
        s=45,
        alpha=0.75,
        color=COLOR_MF,
        edgecolor="none",
        zorder=2,
    )

    ax_box.scatter(
        x_true_jit,
        y_true,
        s=45,
        alpha=0.75,
        color=COLOR_TRUE,
        edgecolor="none",
        zorder=2,
    )

    ax_box.scatter(
        [1.0],
        [dataset_mf_median],
        s=190,
        color=COLOR_MF,
        edgecolor=COLOR_MEDIAN_MARKER_EDGE,
        linewidth=1.3,
        marker="s",
        zorder=5,
        label=f"MF median={dataset_mf_median:.3g}",
    )

    ax_box.scatter(
        [2.0],
        [dataset_true_median],
        s=190,
        color=COLOR_TRUE,
        edgecolor=COLOR_MEDIAN_MARKER_EDGE,
        linewidth=1.3,
        marker="D",
        zorder=5,
        label=f"real/full median={dataset_true_median:.3g}",
    )

    ax_box.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    ax_box.set_xlim(0.55, 2.45)

    ax_box.set_xticks([1.0, 2.0])

    ax_box.set_xticklabels(["mean-field R", "real/full R"])

    ax_box.set_ylabel("median held-out z-Pearson per dataset")

    ax_box.set_title(f"Dataset-median held-out z-Pearson\nn={len(dataset_means):,} datasets")

    ax_box.grid(axis="y", alpha=0.25)

    ax_box.legend(frameon=False)

    fig.suptitle(
        "CIPHER forward problem with ordinary correlation-space least squares",
        fontsize=15,
        y=1.03,
    )

    plt.tight_layout()

    png_path = OUTROOT / OUT_FIG_PNG

    svg_path = OUTROOT / OUT_FIG_SVG

    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")

    plt.savefig(svg_path, bbox_inches="tight")

    plt.show()

    print(f"[saved] {png_path}")

    print(f"[saved] {svg_path}")

    summary = {
        "precompute_root": str(root),
        "expression_threshold": float(EXPRESSION_THRESHOLD),
        "threshold_tag": threshold_tag,
        "n_folders": int(len(folders)),
        "n_success": int(len(all_results)),
        "n_errors": int(len(all_errors)),
        "results": all_results,
        "errors": all_errors,
        "merged_csv": str(merged_path),
        "dataset_means_csv": str(dataset_means_path),
        "figure_png": str(png_path),
        "figure_svg": str(svg_path),
        "overall": {
            "perturbation_split_level": {
                "pearson_true_train_z": summarize(all_df["pearson_true_train_z"]),
                "pearson_true_test_z": summarize(all_df["pearson_true_test_z"]),
                "pearson_mf_train_z": summarize(all_df["pearson_mf_train_z"]),
                "pearson_mf_test_z": summarize(all_df["pearson_mf_test_z"]),
                "pearson_true_minus_mf_test_z": summarize(
                    all_df["pearson_true_test_z"].to_numpy(float)
                    - all_df["pearson_mf_test_z"].to_numpy(float)
                ),
                "train_mse_true_z": summarize(all_df["train_mse_true_z"]),
                "test_mse_true_z": summarize(all_df["test_mse_true_z"]),
                "train_mse_mf_z": summarize(all_df["train_mse_mf_z"]),
                "test_mse_mf_z": summarize(all_df["test_mse_mf_z"]),
                "beta_true_train": summarize(all_df["beta_true_train"]),
                "beta_mf_train": summarize(all_df["beta_mf_train"]),
            },
            "dataset_level": {
                "pearson_true_test_z_median_per_dataset": summarize(
                    dataset_means["pearson_true_test_z_median"]
                ),
                "pearson_mf_test_z_median_per_dataset": summarize(
                    dataset_means["pearson_mf_test_z_median"]
                ),
                "pearson_true_minus_mf_z_median_per_dataset": summarize(
                    dataset_means["pearson_true_minus_mf_z_median"]
                ),
                "pearson_true_test_z_mean_per_dataset": summarize(
                    dataset_means["pearson_true_test_z_mean"]
                ),
                "pearson_mf_test_z_mean_per_dataset": summarize(
                    dataset_means["pearson_mf_test_z_mean"]
                ),
                "pearson_true_minus_mf_z_mean_per_dataset": summarize(
                    dataset_means["pearson_true_minus_mf_z_mean"]
                ),
            },
        },
        "config": {
            "objective": "ordinary unweighted correlation-space least squares",
            "model": "dz ≈ beta_g R[:,g]",
            "coordinate_change": {
                "sd_i": "sqrt(Sigma_ii)",
                "R_ij": "Sigma_ij / (sd_i sd_j)",
                "dz_i": "dx_i / sd_i",
            },
            "evaluation": (
                "Only standardized z-space metrics are computed. "
                "No raw-space reconstruction or raw-space Pearson/MSE is reported."
            ),
            "interpretation": (
                "This is linear response for standardized variables z, not the original "
                "raw expression variables x. It matches standardized shifts and therefore "
                "weights genes equally after variance normalization. It is not equivalent "
                "to the raw covariance CIPHER fit."
            ),
            "train_frac": float(TRAIN_FRAC),
            "n_splits": int(N_SPLITS),
            "split_seed": int(SPLIT_SEED),
            "batch_size": int(BATCH_SIZE),
            "plot_order": ["mean-field R", "real/full R"],
            "colors": {
                "mean_field": COLOR_MF,
                "real_full": COLOR_TRUE,
            },
        },
    }

    summary_path = OUTROOT / OUT_SUMMARY_JSON

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print("\n" + "=" * 110)

    print("DONE")

    print(f"successful datasets: {len(all_results)}")

    print(f"errored datasets:    {len(all_errors)}")

    print(f"merged csv:          {merged_path}")

    print(f"dataset summaries csv:{dataset_means_path}")

    print(f"summary json:        {summary_path}")

    print("=" * 110)


def forward_highvar_config1():
    PRECOMPUTE_ROOT = os.path.join(SUPPL, "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED")

    EXPRESSION_THRESHOLD = 0.1

    DATASET_FOLDERS = None

    OUT_SCORE_CSV = "cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED.csv"

    OUT_SCORE_NPZ = "cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED.npz"

    OUT_SCORE_JSON = "cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_metadata.json"

    OUT_MERGED_CSV = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_merged.csv"

    OUT_DATASET_MEANS_CSV = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_dataset_means.csv"

    OUT_FIG_PNG = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_composite_MEDIAN.png"

    OUT_FIG_SVG = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_composite_MEDIAN.svg"

    OUT_SUMMARY_JSON = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_summary.json"

    OUT_REMOVED_GENES_CSV = "removed_high_variance_genes_for_forward_pearson.csv"

    OUT_ALL_REMOVED_GENES_CSV = "ALL_DATASETS__removed_high_variance_genes_for_forward_pearson.csv"

    EXPLICIT_REMOVE_GENES = [
        "MALAT1",
    ]

    REMOVE_GENE_REGEXES = [
        # r"^MT-",
        # r"^RPL",
        # r"^RPS",
        # r"^HB[ABDEGQMZ]",
    ]

    REMOVE_HIGH_VARIANCE_GENES = True

    HIGH_VAR_QUANTILE = 0.995

    MAX_HIGH_VAR_GENES_TO_REMOVE = None

    PRINT_ALL_REMOVED_GENES = True

    TRAIN_FRAC = 0.5

    N_SPLITS = 5

    SPLIT_SEED = 0

    BATCH_SIZE = 64

    EPS = 1e-12

    TQDM_NCOLS = 110

    N_HIST_BINS = 90

    FIGSIZE = (13, 5)

    DPI = 300

    USE_QUANTILE_XLIM = True

    LOW_Q = 0.005

    HIGH_Q = 0.995

    CLIP_HIST_TO_XLIM = True

    COLOR_TRUE = "#1f77b4"

    COLOR_MF = "#ff7f0e"

    COLOR_PAIR = "0.72"

    COLOR_MEDIAN_MARKER_EDGE = "black"

    TRUE_SIGMA_CANDIDATES = [
        "Sigma_full_ridge.npy",
        "Sigma_true_ridge.npy",
        "Sigma_full.npy",
        "Sigma_true.npy",
    ]

    MEANFIELD_SIGMA_CANDIDATES = [
        "Sigma_meanfield_ridge.npy",
        "Sigma_mean_field_ridge.npy",
        "Sigma_mf_ridge.npy",
        "Sigma_MF_ridge.npy",
        "Sigma_full_meanfield_ridge.npy",
        "Sigma_full_mean_field_ridge.npy",
        "Sigma_full_mf_ridge.npy",
        "Sigma_shuffle_ridge.npy",
        "Sigma_shuffled_ridge.npy",
        "Sigma_full_shuffle_ridge.npy",
        "Sigma_full_shuffled_ridge.npy",
        "Sigma_meanfield.npy",
        "Sigma_mean_field.npy",
        "Sigma_mf.npy",
        "Sigma_MF.npy",
        "Sigma_full_meanfield.npy",
        "Sigma_full_mean_field.npy",
        "Sigma_shuffle.npy",
        "Sigma_shuffled.npy",
    ]

    MEANFIELD_GLOB_PATTERNS = [
        "*meanfield*.npy",
        "*mean_field*.npy",
        "*MeanField*.npy",
        "*MF*.npy",
        "*mf*.npy",
        "*shuffle*.npy",
        "*shuffled*.npy",
    ]

    def make_gene_removal_mask(genes, Sigma_true, folder):
        """
        Builds a gene-level removal mask.

        Removed genes include:
          1. EXPLICIT_REMOVE_GENES, e.g. MALAT1
          2. Genes matching REMOVE_GENE_REGEXES
          3. Top high-variance genes by diag(Sigma_true)

        Returns:
          keep_gene_mask_old: bool array length p over the original gene system.
          removed_df: dataframe describing removed genes.
          diag: diag(Sigma_true)
        """
        genes = np.asarray(genes, dtype=object)
        p = len(genes)

        remove_mask = np.zeros(p, dtype=bool)
        reasons = [[] for _ in range(p)]

        gene_upper_to_idx = {str(g).upper(): i for i, g in enumerate(genes)}

        # --------------------------------------------------------
        # Explicit removals, case-insensitive.
        # --------------------------------------------------------

        for g in EXPLICIT_REMOVE_GENES:
            j = gene_upper_to_idx.get(str(g).upper(), None)
            if j is not None:
                remove_mask[j] = True
                reasons[j].append("explicit_remove")

        # --------------------------------------------------------
        # Regex removals.
        # --------------------------------------------------------

        for j, g in enumerate(genes):
            gs = str(g)

            for pat in REMOVE_GENE_REGEXES:
                if re.search(pat, gs):
                    remove_mask[j] = True
                    reasons[j].append(f"regex:{pat}")

        # --------------------------------------------------------
        # High-variance removals by diag(Sigma_true).
        # --------------------------------------------------------

        diag = np.asarray(Sigma_true.diagonal(), dtype=np.float64)
        diag = np.nan_to_num(diag, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

        high_var_idx = np.asarray([], dtype=np.int64)
        threshold = np.nan

        if REMOVE_HIGH_VARIANCE_GENES:
            finite = np.isfinite(diag)

            if finite.sum() > 0:
                if MAX_HIGH_VAR_GENES_TO_REMOVE is not None:
                    n_remove = int(MAX_HIGH_VAR_GENES_TO_REMOVE)
                    n_remove = max(0, min(n_remove, int(finite.sum())))

                    if n_remove > 0:
                        finite_idx = np.flatnonzero(finite)
                        order = np.argsort(diag[finite_idx])[::-1]
                        high_var_idx = finite_idx[order[:n_remove]]
                        threshold = float(diag[high_var_idx[-1]])
                    else:
                        high_var_idx = np.asarray([], dtype=np.int64)
                        threshold = np.nan

                else:
                    threshold = float(np.nanquantile(diag[finite], HIGH_VAR_QUANTILE))
                    high_var_idx = np.flatnonzero(finite & (diag >= threshold))

                for j in high_var_idx:
                    remove_mask[j] = True
                    reasons[j].append("high_sigma_diag_variance")

                print(
                    f"[gene removal] high variance threshold diag(Sigma_true) >= {threshold:.6g}; "
                    f"removed high-var genes={len(high_var_idx):,}"
                )

        keep_mask = ~remove_mask

        removed_rows = []

        for j in np.flatnonzero(remove_mask):
            removed_rows.append(
                {
                    "dataset_folder": str(Path(folder).name),
                    "old_gene_index": int(j),
                    "gene": str(genes[j]),
                    "sigma_true_diag_variance": float(diag[j]) if np.isfinite(diag[j]) else np.nan,
                    "reason": ";".join(reasons[j]) if len(reasons[j]) else "removed",
                    "high_var_quantile": float(HIGH_VAR_QUANTILE),
                    "high_var_threshold": float(threshold) if np.isfinite(threshold) else np.nan,
                    "explicit_remove": bool("explicit_remove" in reasons[j]),
                    "high_sigma_diag_variance": bool("high_sigma_diag_variance" in reasons[j]),
                }
            )

        removed_df = pd.DataFrame(removed_rows)

        if len(removed_df) > 0:
            removed_df = removed_df.sort_values(
                ["sigma_true_diag_variance", "gene"],
                ascending=[False, True],
            ).reset_index(drop=True)

            removed_df.insert(
                0,
                "removed_rank_in_dataset",
                np.arange(1, len(removed_df) + 1),
            )
        else:
            removed_df = pd.DataFrame(
                columns=[
                    "removed_rank_in_dataset",
                    "dataset_folder",
                    "old_gene_index",
                    "gene",
                    "sigma_true_diag_variance",
                    "reason",
                    "high_var_quantile",
                    "high_var_threshold",
                    "explicit_remove",
                    "high_sigma_diag_variance",
                ]
            )

        removed_path = Path(folder) / OUT_REMOVED_GENES_CSV
        removed_df.to_csv(removed_path, index=False)

        print(f"[gene removal] removed total={int(remove_mask.sum()):,}/{p:,}")
        print(f"[gene removal] remaining genes={int(keep_mask.sum()):,}/{p:,}")
        print(f"[saved] {removed_path}")

        genes_str = genes.astype(str)

        malat1_matches = np.where(np.char.upper(genes_str.astype(str)) == "MALAT1")[0]
        if len(malat1_matches) > 0:
            malat1_idx = int(malat1_matches[0])
            print(
                f"[gene removal] MALAT1 present, removed={bool(remove_mask[malat1_idx])}, "
                f"diag_var={diag[malat1_idx]:.6g}"
            )
        else:
            print("[gene removal] MALAT1 not present in genes.npy")

        # --------------------------------------------------------
        # PRINT EXACT REMOVED GENES, UNTRUNCATED
        # --------------------------------------------------------

        if PRINT_ALL_REMOVED_GENES:
            print("\n" + "-" * 130)
            print(f"[removed genes: {Path(folder).name}]")
            print("-" * 130)

            if len(removed_df) == 0:
                print("No genes removed.")
            else:
                show_cols = [
                    "removed_rank_in_dataset",
                    "gene",
                    "old_gene_index",
                    "sigma_true_diag_variance",
                    "reason",
                ]

                with pd.option_context(
                    "display.max_rows", None,
                    "display.max_columns", None,
                    "display.width", 260,
                    "display.max_colwidth", None,
                ):
                    print(removed_df[show_cols].to_string(index=False))

            print("-" * 130 + "\n")

        if keep_mask.sum() < 3:
            raise ValueError("Fewer than 3 genes remain after high-variance filtering.")

        return keep_mask, removed_df, diag

    def process_one_folder(folder):
        folder = Path(folder)
        dataset = folder.name

        print("\n" + "=" * 110)
        print(f"[dataset] {dataset}")
        print(f"[folder]  {folder}")
        print("=" * 110)

        genes_path = folder / "genes.npy"
        perts_path = folder / "perturbations.npy"
        stats_path = folder / "perturbation_stats.h5"
        sigdir = folder / "sigmas"

        required_base = [
            genes_path,
            perts_path,
            stats_path,
            sigdir,
        ]

        missing = [str(x) for x in required_base if not x.exists()]

        if missing:
            raise FileNotFoundError("Missing required files/folders:\n" + "\n".join(missing))

        sigma_true_path = find_sigma_path(
            sigdir=sigdir,
            candidates=TRUE_SIGMA_CANDIDATES,
            glob_patterns=None,
            exclude_paths=None,
            label="true/full",
        )

        sigma_mf_path = find_sigma_path(
            sigdir=sigdir,
            candidates=MEANFIELD_SIGMA_CANDIDATES,
            glob_patterns=MEANFIELD_GLOB_PATTERNS,
            exclude_paths=[sigma_true_path],
            label="mean-field",
        )

        genes = decode_str_array(np.load(genes_path, allow_pickle=True))
        perts = decode_str_array(np.load(perts_path, allow_pickle=True))

        p_old = len(genes)
        n_perts = len(perts)

        Sigma_true = np.load(sigma_true_path, mmap_mode="r")
        Sigma_mf = np.load(sigma_mf_path, mmap_mode="r")

        if Sigma_true.shape != (p_old, p_old):
            raise ValueError(f"Sigma_true shape {Sigma_true.shape}, expected {(p_old, p_old)}")

        if Sigma_mf.shape != (p_old, p_old):
            raise ValueError(f"Sigma_mf shape {Sigma_mf.shape}, expected {(p_old, p_old)}")

        # --------------------------------------------------------
        # Remove MALAT1 / high-variance genes from the equation.
        # --------------------------------------------------------

        keep_gene_mask_old, removed_genes_df, sigma_diag = make_gene_removal_mask(
            genes=genes,
            Sigma_true=Sigma_true,
            folder=folder,
        )

        keep_gene_idx_old = np.flatnonzero(keep_gene_mask_old).astype(np.int64)
        genes_clean = genes[keep_gene_idx_old]
        p_clean = len(genes_clean)

        old_to_clean_idx = np.full(p_old, -1, dtype=np.int64)
        old_to_clean_idx[keep_gene_idx_old] = np.arange(p_clean, dtype=np.int64)

        # --------------------------------------------------------
        # Match perturbation targets in the original gene system,
        # then drop perturbations whose target gene was removed.
        # --------------------------------------------------------

        target_genes, target_idx_old, matched = load_target_indices(folder, perts, genes)

        target_not_removed = matched & (target_idx_old >= 0) & keep_gene_mask_old[target_idx_old]

        keep_pert_idx = np.flatnonzero(target_not_removed).astype(np.int64)

        matched_perts = perts[keep_pert_idx]
        matched_targets = target_genes[keep_pert_idx]
        matched_target_idx_old = target_idx_old[keep_pert_idx]
        matched_target_idx_clean = old_to_clean_idx[matched_target_idx_old]

        n_match = len(keep_pert_idx)

        n_matched_before_gene_filter = int(np.sum(matched))
        n_dropped_target_removed = int(n_matched_before_gene_filter - n_match)

        print(
            f"[load] genes old={p_old:,}, genes clean={p_clean:,}, "
            f"perts={n_perts:,}, matched before gene filter={n_matched_before_gene_filter:,}, "
            f"matched after gene filter={n_match:,}"
        )
        print(f"[target filter] dropped because target gene removed={n_dropped_target_removed:,}")

        if n_match == 0:
            raise ValueError("No matched perturbation targets remain after high-variance gene removal.")

        splits = make_gene_splits(
            p=p_clean,
            n_splits=N_SPLITS,
            train_frac=TRAIN_FRAC,
            seed=SPLIT_SEED,
        )

        rows_out = []

        with h5py.File(stats_path, "r") as h5:
            dx_ds = h5["dx"]

            if dx_ds.shape != (n_perts, p_old):
                raise ValueError(f"dx shape {dx_ds.shape}, expected {(n_perts, p_old)}")

            if "n_cells_pert" in h5:
                n_cells_all = np.asarray(h5["n_cells_pert"][:], dtype=np.int64)
                n_cells_match = n_cells_all[keep_pert_idx]
            else:
                n_cells_match = np.full(n_match, -1, dtype=np.int64)

            for start in tqdm(
                range(0, n_match, BATCH_SIZE),
                desc=f"{dataset}: highvar-filtered Pearson batches",
                ncols=TQDM_NCOLS,
            ):
                end = min(start + BATCH_SIZE, n_match)

                pert_rows = keep_pert_idx[start:end]
                gidx_old = matched_target_idx_old[start:end]
                gidx_clean = matched_target_idx_clean[start:end]

                # dx = xu - x0, restricted to kept genes.
                y = np.asarray(dx_ds[pert_rows, :][:, keep_gene_idx_old], dtype=np.float32)

                # Sigma restricted to kept rows and target columns.
                basis_true = full_sigma_columns_masked(
                    Sigma=Sigma_true,
                    old_gene_idx=gidx_old,
                    keep_gene_idx_old=keep_gene_idx_old,
                )

                basis_mf = full_sigma_columns_masked(
                    Sigma=Sigma_mf,
                    old_gene_idx=gidx_old,
                    keep_gene_idx_old=keep_gene_idx_old,
                )

                for split_obj in splits:
                    split_id = split_obj["split"]
                    train_idx = split_obj["train_idx"]
                    test_idx = split_obj["test_idx"]

                    fit_true = fit_a_train_eval_test_pearson(
                        y=y,
                        basis=basis_true,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        eps=EPS,
                    )

                    fit_mf = fit_a_train_eval_test_pearson(
                        y=y,
                        basis=basis_mf,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        eps=EPS,
                    )

                    for local_i in range(end - start):
                        global_i = start + local_i

                        rows_out.append(
                            {
                                "dataset": dataset,
                                "expression_threshold": float(EXPRESSION_THRESHOLD),
                                "split": int(split_id),
                                "train_frac": float(TRAIN_FRAC),
                                "n_train_genes": int(len(train_idx)),
                                "n_test_genes": int(len(test_idx)),
                                "n_genes_original": int(p_old),
                                "n_genes_clean": int(p_clean),
                                "n_genes_removed": int(p_old - p_clean),
                                "perturbation": str(matched_perts[global_i]),
                                "target_gene": str(matched_targets[global_i]),
                                "target_gene_index_original": int(matched_target_idx_old[global_i]),
                                "target_gene_index_clean": int(matched_target_idx_clean[global_i]),
                                "n_cells_pert": int(n_cells_match[global_i]),

                                # true/full Sigma
                                "pearson_true_train": float(fit_true["pearson_train"][local_i]),
                                "pearson_true_test": float(fit_true["pearson_test"][local_i]),
                                "a_true_train": float(fit_true["a"][local_i]),
                                "train_mse_true": float(fit_true["train_mse"][local_i]),
                                "test_mse_true": float(fit_true["test_mse"][local_i]),

                                # mean-field Sigma
                                "pearson_mf_train": float(fit_mf["pearson_train"][local_i]),
                                "pearson_mf_test": float(fit_mf["pearson_test"][local_i]),
                                "a_mf_train": float(fit_mf["a"][local_i]),
                                "train_mse_mf": float(fit_mf["train_mse"][local_i]),
                                "test_mse_mf": float(fit_mf["test_mse"][local_i]),
                            }
                        )

                del y, basis_true, basis_mf
                gc.collect()

        out = pd.DataFrame(rows_out)

        csv_path = folder / OUT_SCORE_CSV
        out.to_csv(csv_path, index=False)

        npz_path = folder / OUT_SCORE_NPZ
        np.savez_compressed(
            npz_path,
            dataset=np.asarray(dataset, dtype=object),
            expression_threshold=np.asarray(EXPRESSION_THRESHOLD, dtype=np.float64),
            rows=out.to_records(index=False),
        )

        meta = {
            "dataset": dataset,
            "folder": str(folder),
            "expression_threshold": float(EXPRESSION_THRESHOLD),
            "n_genes_original": int(p_old),
            "n_genes_clean": int(p_clean),
            "n_genes_removed": int(p_old - p_clean),
            "n_perturbations_total": int(n_perts),
            "n_perturbations_matched_before_gene_filter": int(n_matched_before_gene_filter),
            "n_perturbations_matched_after_gene_filter": int(n_match),
            "n_perturbations_dropped_because_target_removed": int(n_dropped_target_removed),
            "train_frac": float(TRAIN_FRAC),
            "n_splits": int(N_SPLITS),
            "split_seed": int(SPLIT_SEED),
            "gene_removal": {
                "explicit_remove_genes": EXPLICIT_REMOVE_GENES,
                "remove_gene_regexes": REMOVE_GENE_REGEXES,
                "remove_high_variance_genes": bool(REMOVE_HIGH_VARIANCE_GENES),
                "high_var_quantile": float(HIGH_VAR_QUANTILE),
                "max_high_var_genes_to_remove": (
                    None if MAX_HIGH_VAR_GENES_TO_REMOVE is None else int(MAX_HIGH_VAR_GENES_TO_REMOVE)
                ),
                "removed_genes_csv": str(folder / OUT_REMOVED_GENES_CSV),
                "removed_genes": removed_genes_df.to_dict(orient="records"),
            },
            "definition": (
                "High-variance-filtered gene-held-out Pearson for real/full Sigma and mean-field Sigma. "
                "MALAT1 and top high-variance genes are removed from dx=xu-x0 and Sigma rows. "
                "Perturbations whose target gene is removed are excluded. "
                "Fit scalar a on train genes using "
                "a=<dx_train,Sigma_col_train>/<Sigma_col_train,Sigma_col_train>. "
                "Evaluate Pearson(dx_test, a*Sigma_col_test). "
                "Same gene splits used for true and mean-field."
            ),
            "median_test_pearson": {
                "true": float(np.nanmedian(out["pearson_true_test"])),
                "mean_field": float(np.nanmedian(out["pearson_mf_test"])),
            },
            "mean_test_pearson": {
                "true": float(np.nanmean(out["pearson_true_test"])),
                "mean_field": float(np.nanmean(out["pearson_mf_test"])),
            },
            "files": {
                "csv": str(csv_path),
                "npz": str(npz_path),
                "sigma_true": str(sigma_true_path),
                "sigma_mean_field": str(sigma_mf_path),
                "stats_h5": str(stats_path),
                "removed_genes_csv": str(folder / OUT_REMOVED_GENES_CSV),
            },
        }

        json_path = folder / OUT_SCORE_JSON

        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2, default=json_default)

        print(f"[saved] {csv_path}")
        print(f"[summary median held-out Pearson true] {np.nanmedian(out['pearson_true_test']):.4f}")
        print(f"[summary median held-out Pearson MF]   {np.nanmedian(out['pearson_mf_test']):.4f}")

        del Sigma_true, Sigma_mf
        gc.collect()

        return {
            "dataset": dataset,
            "folder": str(folder),
            "csv": str(csv_path),
            "npz": str(npz_path),
            "removed_genes_csv": str(folder / OUT_REMOVED_GENES_CSV),
            "n_genes_original": int(p_old),
            "n_genes_clean": int(p_clean),
            "n_genes_removed": int(p_old - p_clean),
            "n_matched_before_gene_filter": int(n_matched_before_gene_filter),
            "n_matched_after_gene_filter": int(n_match),
            "n_dropped_target_removed": int(n_dropped_target_removed),
            "mean_pearson_true_test": float(np.nanmean(out["pearson_true_test"])),
            "median_pearson_true_test": float(np.nanmedian(out["pearson_true_test"])),
            "mean_pearson_mf_test": float(np.nanmean(out["pearson_mf_test"])),
            "median_pearson_mf_test": float(np.nanmedian(out["pearson_mf_test"])),
        }

    root = Path(PRECOMPUTE_ROOT)
    OUTROOT = Path(OUTDIR)
    OUTROOT.mkdir(parents=True, exist_ok=True)

    if DATASET_FOLDERS is None:
        folders = find_dataset_folders(root, EXPRESSION_THRESHOLD)
    else:
        folders = [Path(x) for x in DATASET_FOLDERS]

    print(f"[run] found {len(folders)} precomputed dataset folders")

    if len(folders) == 0:
        raise FileNotFoundError(
            f"No folders found under {PRECOMPUTE_ROOT} for EXPRESSION_THRESHOLD={EXPRESSION_THRESHOLD}"
        )

    all_results = []

    all_errors = []

    for folder in tqdm(folders, desc="datasets", ncols=TQDM_NCOLS):
        try:
            res = process_one_folder(folder)
            all_results.append(res)
        except Exception as e:
            print("\n" + "!" * 110)
            print(f"[ERROR] {folder}")
            print(repr(e))
            print("!" * 110 + "\n")
            all_errors.append({"folder": str(folder), "error": repr(e)})
            gc.collect()

    all_removed_gene_tables = []

    for folder in folders:
        removed_path = Path(folder) / OUT_REMOVED_GENES_CSV

        if removed_path.exists():
            rdf = pd.read_csv(removed_path)
            rdf["source_folder"] = str(folder)
            all_removed_gene_tables.append(rdf)

    all_removed_genes_path = None

    if len(all_removed_gene_tables) > 0:
        all_removed_genes_df = pd.concat(all_removed_gene_tables, axis=0, ignore_index=True)

        all_removed_genes_df = all_removed_genes_df.sort_values(
            ["dataset_folder", "sigma_true_diag_variance", "gene"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

        all_removed_genes_path = OUTROOT / OUT_ALL_REMOVED_GENES_CSV
        all_removed_genes_df.to_csv(all_removed_genes_path, index=False)

        print("\n" + "=" * 150)
        print("[ALL REMOVED GENES ACROSS DATASETS]")
        print("=" * 150)

        show_cols = [
            "dataset_folder",
            "removed_rank_in_dataset",
            "gene",
            "old_gene_index",
            "sigma_true_diag_variance",
            "reason",
        ]

        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 300,
            "display.max_colwidth", None,
        ):
            print(all_removed_genes_df[show_cols].to_string(index=False))

        print("=" * 150)
        print(f"[saved] {all_removed_genes_path}")
        print("=" * 150 + "\n")

    else:
        print("[removed genes] No removed-gene CSVs found.")

    threshold_tag = threshold_to_tag(EXPRESSION_THRESHOLD)

    score_files = sorted(root.glob(f"*__mean_ge_{threshold_tag}/{OUT_SCORE_CSV}"))

    print(f"[plot] found {len(score_files)} highvar-filtered true+MF Pearson train/test files")

    dfs = []

    for path in tqdm(score_files, desc="loading highvar-filtered Pearson files", ncols=TQDM_NCOLS):
        df = pd.read_csv(path)

        if "dataset" not in df.columns:
            df["dataset"] = Path(path).parent.name

        df["expression_threshold"] = float(EXPRESSION_THRESHOLD)
        df["source_folder"] = str(Path(path).parent)
        df["source_file"] = str(path)

        numeric_cols = [
            "pearson_true_train",
            "pearson_true_test",
            "a_true_train",
            "train_mse_true",
            "test_mse_true",
            "pearson_mf_train",
            "pearson_mf_test",
            "a_mf_train",
            "train_mse_mf",
            "test_mse_mf",
            "n_genes_original",
            "n_genes_clean",
            "n_genes_removed",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        dfs.append(df)

    if len(dfs) == 0:
        summary_path = OUTROOT / OUT_SUMMARY_JSON

        summary = {
            "precompute_root": str(root),
            "expression_threshold": float(EXPRESSION_THRESHOLD),
            "threshold_tag": threshold_tag,
            "n_folders": int(len(folders)),
            "n_success": int(len(all_results)),
            "n_errors": int(len(all_errors)),
            "results": all_results,
            "errors": all_errors,
            "all_removed_genes_csv": None if all_removed_genes_path is None else str(all_removed_genes_path),
            "message": "No highvar-filtered true+MF Pearson files produced. All datasets failed or were skipped.",
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=json_default)

        print("\n" + "=" * 110)
        print("NO HIGHVAR-FILTERED TRUE+MF PEARSON FILES WERE PRODUCED.")
        print(f"successful datasets: {len(all_results)}")
        print(f"errored datasets:    {len(all_errors)}")
        print(f"summary json:        {summary_path}")
        print("\nErrors:")
        for err in all_errors:
            print(f"- {err['folder']}")
            print(f"  {err['error']}")
        print("=" * 110)

        raise SystemExit

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    finite_mask = (
        np.isfinite(all_df["pearson_true_test"].to_numpy(float))
        & np.isfinite(all_df["pearson_mf_test"].to_numpy(float))
    )

    all_df = all_df.loc[finite_mask].copy()

    if len(all_df) == 0:
        raise RuntimeError("Pearson files were produced, but all true/MF test Pearson values were non-finite.")

    merged_path = OUTROOT / OUT_MERGED_CSV

    all_df.to_csv(merged_path, index=False)

    print(f"[saved] {merged_path}")

    dataset_means = (
        all_df
        .groupby("dataset", dropna=False)
        .agg(
            n_rows=("perturbation", "count"),
            n_perturbations=("perturbation", "nunique"),

            n_genes_original=("n_genes_original", "first"),
            n_genes_clean=("n_genes_clean", "first"),
            n_genes_removed=("n_genes_removed", "first"),

            pearson_true_train_mean=("pearson_true_train", "mean"),
            pearson_true_test_mean=("pearson_true_test", "mean"),
            pearson_true_test_median=("pearson_true_test", "median"),
            train_mse_true_mean=("train_mse_true", "mean"),
            test_mse_true_mean=("test_mse_true", "mean"),

            pearson_mf_train_mean=("pearson_mf_train", "mean"),
            pearson_mf_test_mean=("pearson_mf_test", "mean"),
            pearson_mf_test_median=("pearson_mf_test", "median"),
            train_mse_mf_mean=("train_mse_mf", "mean"),
            test_mse_mf_mean=("test_mse_mf", "mean"),
        )
        .reset_index()
    )

    dataset_means["pearson_true_minus_mf_mean"] = (
        dataset_means["pearson_true_test_mean"] - dataset_means["pearson_mf_test_mean"]
    )

    dataset_means["pearson_true_minus_mf_median"] = (
        dataset_means["pearson_true_test_median"] - dataset_means["pearson_mf_test_median"]
    )

    dataset_means_path = OUTROOT / OUT_DATASET_MEANS_CSV

    dataset_means.to_csv(dataset_means_path, index=False)

    print(f"[saved] {dataset_means_path}")

    print("\n[dataset summaries]")

    print(
        dataset_means[
            [
                "dataset",
                "n_perturbations",
                "n_genes_original",
                "n_genes_clean",
                "n_genes_removed",
                "pearson_mf_test_median",
                "pearson_true_test_median",
                "pearson_true_minus_mf_median",
                "pearson_mf_test_mean",
                "pearson_true_test_mean",
                "pearson_true_minus_mf_mean",
            ]
        ].to_string(index=False)
    )

    hist_true = all_df["pearson_true_test"].to_numpy(float)

    hist_mf = all_df["pearson_mf_test"].to_numpy(float)

    hist_true = hist_true[np.isfinite(hist_true)]

    hist_mf = hist_mf[np.isfinite(hist_mf)]

    hist_true_median = safe_nanmedian(hist_true)

    hist_mf_median = safe_nanmedian(hist_mf)

    hist_all = np.concatenate([hist_true, hist_mf])

    if USE_QUANTILE_XLIM:
        xmin = float(np.nanquantile(hist_all, LOW_Q))
        xmax = float(np.nanquantile(hist_all, HIGH_Q))
    else:
        xmin = float(np.nanmin(hist_all))
        xmax = float(np.nanmax(hist_all))

    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0

    pad = 0.05 * max(xmax - xmin, 1e-9)

    xmin -= pad

    xmax += pad

    bins = np.linspace(xmin, xmax, N_HIST_BINS + 1)

    hist_true_plot = np.clip(hist_true, xmin, xmax) if CLIP_HIST_TO_XLIM else hist_true

    hist_mf_plot = np.clip(hist_mf, xmin, xmax) if CLIP_HIST_TO_XLIM else hist_mf

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    ax_hist, ax_box = axes

    ax_hist.hist(
        hist_mf_plot,
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLOR_MF,
        label=f"mean-field Σ\nmedian={hist_mf_median:.3g}",
    )

    ax_hist.hist(
        hist_true_plot,
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLOR_TRUE,
        label=f"real/full Σ\nmedian={hist_true_median:.3g}",
    )

    ax_hist.axvline(
        hist_mf_median,
        linestyle="--",
        linewidth=2.0,
        color=COLOR_MF,
    )

    ax_hist.axvline(
        hist_true_median,
        linestyle="--",
        linewidth=2.0,
        color=COLOR_TRUE,
    )

    ax_hist.axvline(0.0, linestyle=":", linewidth=1.5, color="black", alpha=0.75)

    ax_hist.set_xlim(xmin, xmax)

    ax_hist.set_xlabel(
        r"Held-out Pearson after removing MALAT1/high-variance genes: "
        r"$\rho(\Delta x_{\rm test}, \hat a \Sigma_{{\rm test},g})$"
    )

    ax_hist.set_ylabel("probability density")

    ax_hist.set_title(
        f"High-variance-filtered held-out forward Pearson\n"
        f"threshold={EXPRESSION_THRESHOLD}; splits={N_SPLITS}"
    )

    ax_hist.legend(frameon=False)

    y_mf = dataset_means["pearson_mf_test_median"].to_numpy(float)

    y_true = dataset_means["pearson_true_test_median"].to_numpy(float)

    dataset_mf_median = safe_nanmedian(y_mf)

    dataset_true_median = safe_nanmedian(y_true)

    rng = np.random.default_rng(0)

    x_mf = np.ones_like(y_mf) * 1.0

    x_true = np.ones_like(y_true) * 2.0

    x_mf_jit = x_mf + rng.normal(0, 0.035, size=len(x_mf))

    x_true_jit = x_true + rng.normal(0, 0.035, size=len(x_true))

    for ym, yt in zip(y_mf, y_true):
        if np.isfinite(ym) and np.isfinite(yt):
            ax_box.plot(
                [1.0, 2.0],
                [ym, yt],
                linewidth=0.8,
                alpha=0.35,
                color=COLOR_PAIR,
                zorder=1,
            )

    ax_box.scatter(
        x_mf_jit,
        y_mf,
        s=45,
        alpha=0.75,
        color=COLOR_MF,
        edgecolor="none",
        label="dataset medians",
        zorder=2,
    )

    ax_box.scatter(
        x_true_jit,
        y_true,
        s=45,
        alpha=0.75,
        color=COLOR_TRUE,
        edgecolor="none",
        zorder=2,
    )

    ax_box.scatter(
        [1.0],
        [dataset_mf_median],
        s=190,
        color=COLOR_MF,
        edgecolor=COLOR_MEDIAN_MARKER_EDGE,
        linewidth=1.3,
        marker="s",
        zorder=5,
        label=f"MF median={dataset_mf_median:.3g}",
    )

    ax_box.scatter(
        [2.0],
        [dataset_true_median],
        s=190,
        color=COLOR_TRUE,
        edgecolor=COLOR_MEDIAN_MARKER_EDGE,
        linewidth=1.3,
        marker="D",
        zorder=5,
        label=f"real/full median={dataset_true_median:.3g}",
    )

    ax_box.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    ax_box.set_xlim(0.55, 2.45)

    ax_box.set_xticks([1.0, 2.0])

    ax_box.set_xticklabels(["mean-field Σ", "real/full Σ"])

    ax_box.set_ylabel("median held-out Pearson per dataset")

    ax_box.set_title(f"Dataset-median held-out Pearson\nn={len(dataset_means):,} datasets")

    ax_box.grid(axis="y", alpha=0.25)

    ax_box.legend(frameon=False)

    fig.suptitle(
        "CIPHER forward problem after removing MALAT1 / high-variance genes",
        fontsize=15,
        y=1.03,
    )

    plt.tight_layout()

    png_path = OUTROOT / OUT_FIG_PNG

    svg_path = OUTROOT / OUT_FIG_SVG

    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")

    plt.savefig(svg_path, bbox_inches="tight")

    plt.show()

    print(f"[saved] {png_path}")

    print(f"[saved] {svg_path}")

    summary = {
        "precompute_root": str(root),
        "expression_threshold": float(EXPRESSION_THRESHOLD),
        "threshold_tag": threshold_tag,
        "n_folders": int(len(folders)),
        "n_success": int(len(all_results)),
        "n_errors": int(len(all_errors)),
        "results": all_results,
        "errors": all_errors,
        "merged_csv": str(merged_path),
        "dataset_means_csv": str(dataset_means_path),
        "all_removed_genes_csv": None if all_removed_genes_path is None else str(all_removed_genes_path),
        "figure_png": str(png_path),
        "figure_svg": str(svg_path),
        "gene_removal": {
            "explicit_remove_genes": EXPLICIT_REMOVE_GENES,
            "remove_gene_regexes": REMOVE_GENE_REGEXES,
            "remove_high_variance_genes": bool(REMOVE_HIGH_VARIANCE_GENES),
            "high_var_quantile": float(HIGH_VAR_QUANTILE),
            "max_high_var_genes_to_remove": (
                None if MAX_HIGH_VAR_GENES_TO_REMOVE is None else int(MAX_HIGH_VAR_GENES_TO_REMOVE)
            ),
        },
        "overall": {
            "perturbation_split_level": {
                "pearson_true_train": summarize(all_df["pearson_true_train"]),
                "pearson_true_test": summarize(all_df["pearson_true_test"]),
                "pearson_mf_train": summarize(all_df["pearson_mf_train"]),
                "pearson_mf_test": summarize(all_df["pearson_mf_test"]),
                "pearson_true_minus_mf_test": summarize(
                    all_df["pearson_true_test"].to_numpy(float)
                    - all_df["pearson_mf_test"].to_numpy(float)
                ),
                "train_mse_true": summarize(all_df["train_mse_true"]),
                "test_mse_true": summarize(all_df["test_mse_true"]),
                "train_mse_mf": summarize(all_df["train_mse_mf"]),
                "test_mse_mf": summarize(all_df["test_mse_mf"]),
            },
            "dataset_level": {
                "pearson_true_test_median_per_dataset": summarize(dataset_means["pearson_true_test_median"]),
                "pearson_mf_test_median_per_dataset": summarize(dataset_means["pearson_mf_test_median"]),
                "pearson_true_minus_mf_median_per_dataset": summarize(dataset_means["pearson_true_minus_mf_median"]),
                "pearson_true_test_mean_per_dataset": summarize(dataset_means["pearson_true_test_mean"]),
                "pearson_mf_test_mean_per_dataset": summarize(dataset_means["pearson_mf_test_mean"]),
                "pearson_true_minus_mf_mean_per_dataset": summarize(dataset_means["pearson_true_minus_mf_mean"]),
            },
        },
        "config": {
            "train_frac": float(TRAIN_FRAC),
            "n_splits": int(N_SPLITS),
            "split_seed": int(SPLIT_SEED),
            "batch_size": int(BATCH_SIZE),
            "plot_metric": "median",
            "plot_order": ["mean-field Sigma", "real/full Sigma"],
            "colors": {
                "mean_field": COLOR_MF,
                "real_full": COLOR_TRUE,
            },
            "pearson_definition": (
                "MALAT1 and top high-variance genes are removed from dx=xu-x0 and Sigma rows. "
                "Perturbations whose target gene is removed are excluded. "
                "Fit scalar a on train genes using "
                "a=<dx_train,Sigma_col_train>/<Sigma_col_train,Sigma_col_train>. "
                "Evaluate Pearson(dx_test, a*Sigma_col_test). "
                "Computed for both real/full Sigma and mean-field Sigma using identical cleaned-gene splits."
            ),
        },
    }

    summary_path = OUTROOT / OUT_SUMMARY_JSON

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print("\n" + "=" * 110)

    print("DONE")

    print(f"successful datasets:      {len(all_results)}")

    print(f"errored datasets:         {len(all_errors)}")

    print(f"merged csv:               {merged_path}")

    print(f"dataset summaries csv:    {dataset_means_path}")

    print(f"all removed genes csv:    {all_removed_genes_path}")

    print(f"summary json:             {summary_path}")

    print("=" * 110)


def forward_highvar_config2():
    PRECOMPUTE_ROOT = os.path.join(SUPPL, "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED")

    EXPRESSION_THRESHOLD = 1.0

    DATASET_FOLDERS = None

    OUT_SCORE_CSV = "cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED.csv"

    OUT_SCORE_NPZ = "cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED.npz"

    OUT_SCORE_JSON = "cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_metadata.json"

    OUT_MERGED_CSV = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_merged.csv"

    OUT_DATASET_MEANS_CSV = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_dataset_means.csv"

    OUT_FIG_PNG = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_composite_MEDIAN.png"

    OUT_FIG_SVG = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_composite_MEDIAN.svg"

    OUT_SUMMARY_JSON = "ALL_DATASETS__cipher_forward_PEARSON_TRAINTEST_TRUE_AND_MF_HIGHVAR_FILTERED_summary.json"

    OUT_REMOVED_GENES_CSV = "removed_high_variance_genes_for_forward_pearson.csv"

    OUT_ALL_REMOVED_GENES_CSV = "ALL_DATASETS__removed_high_variance_genes_for_forward_pearson.csv"

    EXPLICIT_REMOVE_GENES = [
        "MALAT1",
    ]

    REMOVE_GENE_REGEXES = [
        # r"^MT-",
        # r"^RPL",
        # r"^RPS",
        # r"^HB[ABDEGQMZ]",
    ]

    REMOVE_HIGH_VARIANCE_GENES = True

    HIGH_VAR_QUANTILE = 0.9

    MAX_HIGH_VAR_GENES_TO_REMOVE = None

    PRINT_ALL_REMOVED_GENES = True

    TRAIN_FRAC = 0.5

    N_SPLITS = 5

    SPLIT_SEED = 0

    BATCH_SIZE = 64

    EPS = 1e-12

    TQDM_NCOLS = 110

    N_HIST_BINS = 90

    FIGSIZE = (13, 5)

    DPI = 300

    USE_QUANTILE_XLIM = True

    LOW_Q = 0.005

    HIGH_Q = 0.995

    CLIP_HIST_TO_XLIM = True

    COLOR_TRUE = "#1f77b4"

    COLOR_MF = "#ff7f0e"

    COLOR_PAIR = "0.72"

    COLOR_MEDIAN_MARKER_EDGE = "black"

    TRUE_SIGMA_CANDIDATES = [
        "Sigma_full_ridge.npy",
        "Sigma_true_ridge.npy",
        "Sigma_full.npy",
        "Sigma_true.npy",
    ]

    MEANFIELD_SIGMA_CANDIDATES = [
        "Sigma_meanfield_ridge.npy",
        "Sigma_mean_field_ridge.npy",
        "Sigma_mf_ridge.npy",
        "Sigma_MF_ridge.npy",
        "Sigma_full_meanfield_ridge.npy",
        "Sigma_full_mean_field_ridge.npy",
        "Sigma_full_mf_ridge.npy",
        "Sigma_shuffle_ridge.npy",
        "Sigma_shuffled_ridge.npy",
        "Sigma_full_shuffle_ridge.npy",
        "Sigma_full_shuffled_ridge.npy",
        "Sigma_meanfield.npy",
        "Sigma_mean_field.npy",
        "Sigma_mf.npy",
        "Sigma_MF.npy",
        "Sigma_full_meanfield.npy",
        "Sigma_full_mean_field.npy",
        "Sigma_shuffle.npy",
        "Sigma_shuffled.npy",
    ]

    MEANFIELD_GLOB_PATTERNS = [
        "*meanfield*.npy",
        "*mean_field*.npy",
        "*MeanField*.npy",
        "*MF*.npy",
        "*mf*.npy",
        "*shuffle*.npy",
        "*shuffled*.npy",
    ]

    def make_gene_removal_mask(genes, Sigma_true, folder):
        """
        Builds a gene-level removal mask.

        Removed genes include:
          1. EXPLICIT_REMOVE_GENES, e.g. MALAT1
          2. Genes matching REMOVE_GENE_REGEXES
          3. Top high-variance genes by diag(Sigma_true)

        Returns:
          keep_gene_mask_old: bool array length p over the original gene system.
          removed_df: dataframe describing removed genes.
          diag: diag(Sigma_true)
        """
        genes = np.asarray(genes, dtype=object)
        p = len(genes)

        remove_mask = np.zeros(p, dtype=bool)
        reasons = [[] for _ in range(p)]

        gene_upper_to_idx = {str(g).upper(): i for i, g in enumerate(genes)}

        # --------------------------------------------------------
        # Explicit removals, case-insensitive.
        # --------------------------------------------------------

        for g in EXPLICIT_REMOVE_GENES:
            j = gene_upper_to_idx.get(str(g).upper(), None)
            if j is not None:
                remove_mask[j] = True
                reasons[j].append("explicit_remove")

        # --------------------------------------------------------
        # Regex removals.
        # --------------------------------------------------------

        for j, g in enumerate(genes):
            gs = str(g)

            for pat in REMOVE_GENE_REGEXES:
                if re.search(pat, gs):
                    remove_mask[j] = True
                    reasons[j].append(f"regex:{pat}")

        # --------------------------------------------------------
        # High-variance removals by diag(Sigma_true).
        # --------------------------------------------------------

        diag = np.asarray(Sigma_true.diagonal(), dtype=np.float64)
        diag = np.nan_to_num(diag, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

        high_var_idx = np.asarray([], dtype=np.int64)
        threshold = np.nan

        if REMOVE_HIGH_VARIANCE_GENES:
            finite = np.isfinite(diag)

            if finite.sum() > 0:
                if MAX_HIGH_VAR_GENES_TO_REMOVE is not None:
                    n_remove = int(MAX_HIGH_VAR_GENES_TO_REMOVE)
                    n_remove = max(0, min(n_remove, int(finite.sum())))

                    if n_remove > 0:
                        finite_idx = np.flatnonzero(finite)
                        order = np.argsort(diag[finite_idx])[::-1]
                        high_var_idx = finite_idx[order[:n_remove]]
                        threshold = float(diag[high_var_idx[-1]])
                    else:
                        high_var_idx = np.asarray([], dtype=np.int64)
                        threshold = np.nan

                else:
                    threshold = float(np.nanquantile(diag[finite], HIGH_VAR_QUANTILE))
                    high_var_idx = np.flatnonzero(finite & (diag >= threshold))

                for j in high_var_idx:
                    remove_mask[j] = True
                    reasons[j].append("high_sigma_diag_variance")

                print(
                    f"[gene removal] high variance threshold diag(Sigma_true) >= {threshold:.6g}; "
                    f"removed high-var genes={len(high_var_idx):,}"
                )

        keep_mask = ~remove_mask

        removed_rows = []

        for j in np.flatnonzero(remove_mask):
            removed_rows.append(
                {
                    "dataset_folder": str(Path(folder).name),
                    "old_gene_index": int(j),
                    "gene": str(genes[j]),
                    "sigma_true_diag_variance": float(diag[j]) if np.isfinite(diag[j]) else np.nan,
                    "reason": ";".join(reasons[j]) if len(reasons[j]) else "removed",
                    "high_var_quantile": float(HIGH_VAR_QUANTILE),
                    "high_var_threshold": float(threshold) if np.isfinite(threshold) else np.nan,
                    "explicit_remove": bool("explicit_remove" in reasons[j]),
                    "high_sigma_diag_variance": bool("high_sigma_diag_variance" in reasons[j]),
                }
            )

        removed_df = pd.DataFrame(removed_rows)

        if len(removed_df) > 0:
            removed_df = removed_df.sort_values(
                ["sigma_true_diag_variance", "gene"],
                ascending=[False, True],
            ).reset_index(drop=True)

            removed_df.insert(
                0,
                "removed_rank_in_dataset",
                np.arange(1, len(removed_df) + 1),
            )
        else:
            removed_df = pd.DataFrame(
                columns=[
                    "removed_rank_in_dataset",
                    "dataset_folder",
                    "old_gene_index",
                    "gene",
                    "sigma_true_diag_variance",
                    "reason",
                    "high_var_quantile",
                    "high_var_threshold",
                    "explicit_remove",
                    "high_sigma_diag_variance",
                ]
            )

        removed_path = Path(folder) / OUT_REMOVED_GENES_CSV
        removed_df.to_csv(removed_path, index=False)

        print(f"[gene removal] removed total={int(remove_mask.sum()):,}/{p:,}")
        print(f"[gene removal] remaining genes={int(keep_mask.sum()):,}/{p:,}")
        print(f"[saved] {removed_path}")

        genes_str = genes.astype(str)

        malat1_matches = np.where(np.char.upper(genes_str.astype(str)) == "MALAT1")[0]
        if len(malat1_matches) > 0:
            malat1_idx = int(malat1_matches[0])
            print(
                f"[gene removal] MALAT1 present, removed={bool(remove_mask[malat1_idx])}, "
                f"diag_var={diag[malat1_idx]:.6g}"
            )
        else:
            print("[gene removal] MALAT1 not present in genes.npy")

        # --------------------------------------------------------
        # PRINT EXACT REMOVED GENES, UNTRUNCATED
        # --------------------------------------------------------

        if PRINT_ALL_REMOVED_GENES:
            print("\n" + "-" * 130)
            print(f"[removed genes: {Path(folder).name}]")
            print("-" * 130)

            if len(removed_df) == 0:
                print("No genes removed.")
            else:
                show_cols = [
                    "removed_rank_in_dataset",
                    "gene",
                    "old_gene_index",
                    "sigma_true_diag_variance",
                    "reason",
                ]

                with pd.option_context(
                    "display.max_rows", None,
                    "display.max_columns", None,
                    "display.width", 260,
                    "display.max_colwidth", None,
                ):
                    print(removed_df[show_cols].to_string(index=False))

            print("-" * 130 + "\n")

        if keep_mask.sum() < 3:
            raise ValueError("Fewer than 3 genes remain after high-variance filtering.")

        return keep_mask, removed_df, diag

    def process_one_folder(folder):
        folder = Path(folder)
        dataset = folder.name

        print("\n" + "=" * 110)
        print(f"[dataset] {dataset}")
        print(f"[folder]  {folder}")
        print("=" * 110)

        genes_path = folder / "genes.npy"
        perts_path = folder / "perturbations.npy"
        stats_path = folder / "perturbation_stats.h5"
        sigdir = folder / "sigmas"

        required_base = [
            genes_path,
            perts_path,
            stats_path,
            sigdir,
        ]

        missing = [str(x) for x in required_base if not x.exists()]

        if missing:
            raise FileNotFoundError("Missing required files/folders:\n" + "\n".join(missing))

        sigma_true_path = find_sigma_path(
            sigdir=sigdir,
            candidates=TRUE_SIGMA_CANDIDATES,
            glob_patterns=None,
            exclude_paths=None,
            label="true/full",
        )

        sigma_mf_path = find_sigma_path(
            sigdir=sigdir,
            candidates=MEANFIELD_SIGMA_CANDIDATES,
            glob_patterns=MEANFIELD_GLOB_PATTERNS,
            exclude_paths=[sigma_true_path],
            label="mean-field",
        )

        genes = decode_str_array(np.load(genes_path, allow_pickle=True))
        perts = decode_str_array(np.load(perts_path, allow_pickle=True))

        p_old = len(genes)
        n_perts = len(perts)

        Sigma_true = np.load(sigma_true_path, mmap_mode="r")
        Sigma_mf = np.load(sigma_mf_path, mmap_mode="r")

        if Sigma_true.shape != (p_old, p_old):
            raise ValueError(f"Sigma_true shape {Sigma_true.shape}, expected {(p_old, p_old)}")

        if Sigma_mf.shape != (p_old, p_old):
            raise ValueError(f"Sigma_mf shape {Sigma_mf.shape}, expected {(p_old, p_old)}")

        # --------------------------------------------------------
        # Remove MALAT1 / high-variance genes from the equation.
        # --------------------------------------------------------

        keep_gene_mask_old, removed_genes_df, sigma_diag = make_gene_removal_mask(
            genes=genes,
            Sigma_true=Sigma_true,
            folder=folder,
        )

        keep_gene_idx_old = np.flatnonzero(keep_gene_mask_old).astype(np.int64)
        genes_clean = genes[keep_gene_idx_old]
        p_clean = len(genes_clean)

        old_to_clean_idx = np.full(p_old, -1, dtype=np.int64)
        old_to_clean_idx[keep_gene_idx_old] = np.arange(p_clean, dtype=np.int64)

        # --------------------------------------------------------
        # Match perturbation targets in the original gene system,
        # then drop perturbations whose target gene was removed.
        # --------------------------------------------------------

        target_genes, target_idx_old, matched = load_target_indices(folder, perts, genes)

        target_not_removed = matched & (target_idx_old >= 0) & keep_gene_mask_old[target_idx_old]

        keep_pert_idx = np.flatnonzero(target_not_removed).astype(np.int64)

        matched_perts = perts[keep_pert_idx]
        matched_targets = target_genes[keep_pert_idx]
        matched_target_idx_old = target_idx_old[keep_pert_idx]
        matched_target_idx_clean = old_to_clean_idx[matched_target_idx_old]

        n_match = len(keep_pert_idx)

        n_matched_before_gene_filter = int(np.sum(matched))
        n_dropped_target_removed = int(n_matched_before_gene_filter - n_match)

        print(
            f"[load] genes old={p_old:,}, genes clean={p_clean:,}, "
            f"perts={n_perts:,}, matched before gene filter={n_matched_before_gene_filter:,}, "
            f"matched after gene filter={n_match:,}"
        )
        print(f"[target filter] dropped because target gene removed={n_dropped_target_removed:,}")

        if n_match == 0:
            raise ValueError("No matched perturbation targets remain after high-variance gene removal.")

        splits = make_gene_splits(
            p=p_clean,
            n_splits=N_SPLITS,
            train_frac=TRAIN_FRAC,
            seed=SPLIT_SEED,
        )

        rows_out = []

        with h5py.File(stats_path, "r") as h5:
            dx_ds = h5["dx"]

            if dx_ds.shape != (n_perts, p_old):
                raise ValueError(f"dx shape {dx_ds.shape}, expected {(n_perts, p_old)}")

            if "n_cells_pert" in h5:
                n_cells_all = np.asarray(h5["n_cells_pert"][:], dtype=np.int64)
                n_cells_match = n_cells_all[keep_pert_idx]
            else:
                n_cells_match = np.full(n_match, -1, dtype=np.int64)

            for start in tqdm(
                range(0, n_match, BATCH_SIZE),
                desc=f"{dataset}: highvar-filtered Pearson batches",
                ncols=TQDM_NCOLS,
            ):
                end = min(start + BATCH_SIZE, n_match)

                pert_rows = keep_pert_idx[start:end]
                gidx_old = matched_target_idx_old[start:end]
                gidx_clean = matched_target_idx_clean[start:end]

                # dx = xu - x0, restricted to kept genes.
                y = np.asarray(dx_ds[pert_rows, :][:, keep_gene_idx_old], dtype=np.float32)

                # Sigma restricted to kept rows and target columns.
                basis_true = full_sigma_columns_masked(
                    Sigma=Sigma_true,
                    old_gene_idx=gidx_old,
                    keep_gene_idx_old=keep_gene_idx_old,
                )

                basis_mf = full_sigma_columns_masked(
                    Sigma=Sigma_mf,
                    old_gene_idx=gidx_old,
                    keep_gene_idx_old=keep_gene_idx_old,
                )

                for split_obj in splits:
                    split_id = split_obj["split"]
                    train_idx = split_obj["train_idx"]
                    test_idx = split_obj["test_idx"]

                    fit_true = fit_a_train_eval_test_pearson(
                        y=y,
                        basis=basis_true,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        eps=EPS,
                    )

                    fit_mf = fit_a_train_eval_test_pearson(
                        y=y,
                        basis=basis_mf,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        eps=EPS,
                    )

                    for local_i in range(end - start):
                        global_i = start + local_i

                        rows_out.append(
                            {
                                "dataset": dataset,
                                "expression_threshold": float(EXPRESSION_THRESHOLD),
                                "split": int(split_id),
                                "train_frac": float(TRAIN_FRAC),
                                "n_train_genes": int(len(train_idx)),
                                "n_test_genes": int(len(test_idx)),
                                "n_genes_original": int(p_old),
                                "n_genes_clean": int(p_clean),
                                "n_genes_removed": int(p_old - p_clean),
                                "perturbation": str(matched_perts[global_i]),
                                "target_gene": str(matched_targets[global_i]),
                                "target_gene_index_original": int(matched_target_idx_old[global_i]),
                                "target_gene_index_clean": int(matched_target_idx_clean[global_i]),
                                "n_cells_pert": int(n_cells_match[global_i]),

                                # true/full Sigma
                                "pearson_true_train": float(fit_true["pearson_train"][local_i]),
                                "pearson_true_test": float(fit_true["pearson_test"][local_i]),
                                "a_true_train": float(fit_true["a"][local_i]),
                                "train_mse_true": float(fit_true["train_mse"][local_i]),
                                "test_mse_true": float(fit_true["test_mse"][local_i]),

                                # mean-field Sigma
                                "pearson_mf_train": float(fit_mf["pearson_train"][local_i]),
                                "pearson_mf_test": float(fit_mf["pearson_test"][local_i]),
                                "a_mf_train": float(fit_mf["a"][local_i]),
                                "train_mse_mf": float(fit_mf["train_mse"][local_i]),
                                "test_mse_mf": float(fit_mf["test_mse"][local_i]),
                            }
                        )

                del y, basis_true, basis_mf
                gc.collect()

        out = pd.DataFrame(rows_out)

        csv_path = folder / OUT_SCORE_CSV
        out.to_csv(csv_path, index=False)

        npz_path = folder / OUT_SCORE_NPZ
        np.savez_compressed(
            npz_path,
            dataset=np.asarray(dataset, dtype=object),
            expression_threshold=np.asarray(EXPRESSION_THRESHOLD, dtype=np.float64),
            rows=out.to_records(index=False),
        )

        meta = {
            "dataset": dataset,
            "folder": str(folder),
            "expression_threshold": float(EXPRESSION_THRESHOLD),
            "n_genes_original": int(p_old),
            "n_genes_clean": int(p_clean),
            "n_genes_removed": int(p_old - p_clean),
            "n_perturbations_total": int(n_perts),
            "n_perturbations_matched_before_gene_filter": int(n_matched_before_gene_filter),
            "n_perturbations_matched_after_gene_filter": int(n_match),
            "n_perturbations_dropped_because_target_removed": int(n_dropped_target_removed),
            "train_frac": float(TRAIN_FRAC),
            "n_splits": int(N_SPLITS),
            "split_seed": int(SPLIT_SEED),
            "gene_removal": {
                "explicit_remove_genes": EXPLICIT_REMOVE_GENES,
                "remove_gene_regexes": REMOVE_GENE_REGEXES,
                "remove_high_variance_genes": bool(REMOVE_HIGH_VARIANCE_GENES),
                "high_var_quantile": float(HIGH_VAR_QUANTILE),
                "max_high_var_genes_to_remove": (
                    None if MAX_HIGH_VAR_GENES_TO_REMOVE is None else int(MAX_HIGH_VAR_GENES_TO_REMOVE)
                ),
                "removed_genes_csv": str(folder / OUT_REMOVED_GENES_CSV),
                "removed_genes": removed_genes_df.to_dict(orient="records"),
            },
            "definition": (
                "High-variance-filtered gene-held-out Pearson for real/full Sigma and mean-field Sigma. "
                "MALAT1 and top high-variance genes are removed from dx=xu-x0 and Sigma rows. "
                "Perturbations whose target gene is removed are excluded. "
                "Fit scalar a on train genes using "
                "a=<dx_train,Sigma_col_train>/<Sigma_col_train,Sigma_col_train>. "
                "Evaluate Pearson(dx_test, a*Sigma_col_test). "
                "Same gene splits used for true and mean-field."
            ),
            "median_test_pearson": {
                "true": float(np.nanmedian(out["pearson_true_test"])),
                "mean_field": float(np.nanmedian(out["pearson_mf_test"])),
            },
            "mean_test_pearson": {
                "true": float(np.nanmean(out["pearson_true_test"])),
                "mean_field": float(np.nanmean(out["pearson_mf_test"])),
            },
            "files": {
                "csv": str(csv_path),
                "npz": str(npz_path),
                "sigma_true": str(sigma_true_path),
                "sigma_mean_field": str(sigma_mf_path),
                "stats_h5": str(stats_path),
                "removed_genes_csv": str(folder / OUT_REMOVED_GENES_CSV),
            },
        }

        json_path = folder / OUT_SCORE_JSON

        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2, default=json_default)

        print(f"[saved] {csv_path}")
        print(f"[summary median held-out Pearson true] {np.nanmedian(out['pearson_true_test']):.4f}")
        print(f"[summary median held-out Pearson MF]   {np.nanmedian(out['pearson_mf_test']):.4f}")

        del Sigma_true, Sigma_mf
        gc.collect()

        return {
            "dataset": dataset,
            "folder": str(folder),
            "csv": str(csv_path),
            "npz": str(npz_path),
            "removed_genes_csv": str(folder / OUT_REMOVED_GENES_CSV),
            "n_genes_original": int(p_old),
            "n_genes_clean": int(p_clean),
            "n_genes_removed": int(p_old - p_clean),
            "n_matched_before_gene_filter": int(n_matched_before_gene_filter),
            "n_matched_after_gene_filter": int(n_match),
            "n_dropped_target_removed": int(n_dropped_target_removed),
            "mean_pearson_true_test": float(np.nanmean(out["pearson_true_test"])),
            "median_pearson_true_test": float(np.nanmedian(out["pearson_true_test"])),
            "mean_pearson_mf_test": float(np.nanmean(out["pearson_mf_test"])),
            "median_pearson_mf_test": float(np.nanmedian(out["pearson_mf_test"])),
        }

    root = Path(PRECOMPUTE_ROOT)
    OUTROOT = Path(OUTDIR)
    OUTROOT.mkdir(parents=True, exist_ok=True)

    if DATASET_FOLDERS is None:
        folders = find_dataset_folders(root, EXPRESSION_THRESHOLD)
    else:
        folders = [Path(x) for x in DATASET_FOLDERS]

    print(f"[run] found {len(folders)} precomputed dataset folders")

    if len(folders) == 0:
        raise FileNotFoundError(
            f"No folders found under {PRECOMPUTE_ROOT} for EXPRESSION_THRESHOLD={EXPRESSION_THRESHOLD}"
        )

    all_results = []

    all_errors = []

    for folder in tqdm(folders, desc="datasets", ncols=TQDM_NCOLS):
        try:
            res = process_one_folder(folder)
            all_results.append(res)
        except Exception as e:
            print("\n" + "!" * 110)
            print(f"[ERROR] {folder}")
            print(repr(e))
            print("!" * 110 + "\n")
            all_errors.append({"folder": str(folder), "error": repr(e)})
            gc.collect()

    all_removed_gene_tables = []

    for folder in folders:
        removed_path = Path(folder) / OUT_REMOVED_GENES_CSV

        if removed_path.exists():
            rdf = pd.read_csv(removed_path)
            rdf["source_folder"] = str(folder)
            all_removed_gene_tables.append(rdf)

    all_removed_genes_path = None

    if len(all_removed_gene_tables) > 0:
        all_removed_genes_df = pd.concat(all_removed_gene_tables, axis=0, ignore_index=True)

        all_removed_genes_df = all_removed_genes_df.sort_values(
            ["dataset_folder", "sigma_true_diag_variance", "gene"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

        all_removed_genes_path = OUTROOT / OUT_ALL_REMOVED_GENES_CSV
        all_removed_genes_df.to_csv(all_removed_genes_path, index=False)

        print("\n" + "=" * 150)
        print("[ALL REMOVED GENES ACROSS DATASETS]")
        print("=" * 150)

        show_cols = [
            "dataset_folder",
            "removed_rank_in_dataset",
            "gene",
            "old_gene_index",
            "sigma_true_diag_variance",
            "reason",
        ]

        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 300,
            "display.max_colwidth", None,
        ):
            print(all_removed_genes_df[show_cols].to_string(index=False))

        print("=" * 150)
        print(f"[saved] {all_removed_genes_path}")
        print("=" * 150 + "\n")

    else:
        print("[removed genes] No removed-gene CSVs found.")

    threshold_tag = threshold_to_tag(EXPRESSION_THRESHOLD)

    score_files = sorted(root.glob(f"*__mean_ge_{threshold_tag}/{OUT_SCORE_CSV}"))

    print(f"[plot] found {len(score_files)} highvar-filtered true+MF Pearson train/test files")

    dfs = []

    for path in tqdm(score_files, desc="loading highvar-filtered Pearson files", ncols=TQDM_NCOLS):
        df = pd.read_csv(path)

        if "dataset" not in df.columns:
            df["dataset"] = Path(path).parent.name

        df["expression_threshold"] = float(EXPRESSION_THRESHOLD)
        df["source_folder"] = str(Path(path).parent)
        df["source_file"] = str(path)

        numeric_cols = [
            "pearson_true_train",
            "pearson_true_test",
            "a_true_train",
            "train_mse_true",
            "test_mse_true",
            "pearson_mf_train",
            "pearson_mf_test",
            "a_mf_train",
            "train_mse_mf",
            "test_mse_mf",
            "n_genes_original",
            "n_genes_clean",
            "n_genes_removed",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        dfs.append(df)

    if len(dfs) == 0:
        summary_path = OUTROOT / OUT_SUMMARY_JSON

        summary = {
            "precompute_root": str(root),
            "expression_threshold": float(EXPRESSION_THRESHOLD),
            "threshold_tag": threshold_tag,
            "n_folders": int(len(folders)),
            "n_success": int(len(all_results)),
            "n_errors": int(len(all_errors)),
            "results": all_results,
            "errors": all_errors,
            "all_removed_genes_csv": None if all_removed_genes_path is None else str(all_removed_genes_path),
            "message": "No highvar-filtered true+MF Pearson files produced. All datasets failed or were skipped.",
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=json_default)

        print("\n" + "=" * 110)
        print("NO HIGHVAR-FILTERED TRUE+MF PEARSON FILES WERE PRODUCED.")
        print(f"successful datasets: {len(all_results)}")
        print(f"errored datasets:    {len(all_errors)}")
        print(f"summary json:        {summary_path}")
        print("\nErrors:")
        for err in all_errors:
            print(f"- {err['folder']}")
            print(f"  {err['error']}")
        print("=" * 110)

        raise SystemExit

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    finite_mask = (
        np.isfinite(all_df["pearson_true_test"].to_numpy(float))
        & np.isfinite(all_df["pearson_mf_test"].to_numpy(float))
    )

    all_df = all_df.loc[finite_mask].copy()

    if len(all_df) == 0:
        raise RuntimeError("Pearson files were produced, but all true/MF test Pearson values were non-finite.")

    merged_path = OUTROOT / OUT_MERGED_CSV

    all_df.to_csv(merged_path, index=False)

    print(f"[saved] {merged_path}")

    dataset_means = (
        all_df
        .groupby("dataset", dropna=False)
        .agg(
            n_rows=("perturbation", "count"),
            n_perturbations=("perturbation", "nunique"),

            n_genes_original=("n_genes_original", "first"),
            n_genes_clean=("n_genes_clean", "first"),
            n_genes_removed=("n_genes_removed", "first"),

            pearson_true_train_mean=("pearson_true_train", "mean"),
            pearson_true_test_mean=("pearson_true_test", "mean"),
            pearson_true_test_median=("pearson_true_test", "median"),
            train_mse_true_mean=("train_mse_true", "mean"),
            test_mse_true_mean=("test_mse_true", "mean"),

            pearson_mf_train_mean=("pearson_mf_train", "mean"),
            pearson_mf_test_mean=("pearson_mf_test", "mean"),
            pearson_mf_test_median=("pearson_mf_test", "median"),
            train_mse_mf_mean=("train_mse_mf", "mean"),
            test_mse_mf_mean=("test_mse_mf", "mean"),
        )
        .reset_index()
    )

    dataset_means["pearson_true_minus_mf_mean"] = (
        dataset_means["pearson_true_test_mean"] - dataset_means["pearson_mf_test_mean"]
    )

    dataset_means["pearson_true_minus_mf_median"] = (
        dataset_means["pearson_true_test_median"] - dataset_means["pearson_mf_test_median"]
    )

    dataset_means_path = OUTROOT / OUT_DATASET_MEANS_CSV

    dataset_means.to_csv(dataset_means_path, index=False)

    print(f"[saved] {dataset_means_path}")

    print("\n[dataset summaries]")

    print(
        dataset_means[
            [
                "dataset",
                "n_perturbations",
                "n_genes_original",
                "n_genes_clean",
                "n_genes_removed",
                "pearson_mf_test_median",
                "pearson_true_test_median",
                "pearson_true_minus_mf_median",
                "pearson_mf_test_mean",
                "pearson_true_test_mean",
                "pearson_true_minus_mf_mean",
            ]
        ].to_string(index=False)
    )

    hist_true = all_df["pearson_true_test"].to_numpy(float)

    hist_mf = all_df["pearson_mf_test"].to_numpy(float)

    hist_true = hist_true[np.isfinite(hist_true)]

    hist_mf = hist_mf[np.isfinite(hist_mf)]

    hist_true_median = safe_nanmedian(hist_true)

    hist_mf_median = safe_nanmedian(hist_mf)

    hist_all = np.concatenate([hist_true, hist_mf])

    if USE_QUANTILE_XLIM:
        xmin = float(np.nanquantile(hist_all, LOW_Q))
        xmax = float(np.nanquantile(hist_all, HIGH_Q))
    else:
        xmin = float(np.nanmin(hist_all))
        xmax = float(np.nanmax(hist_all))

    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0

    pad = 0.05 * max(xmax - xmin, 1e-9)

    xmin -= pad

    xmax += pad

    bins = np.linspace(xmin, xmax, N_HIST_BINS + 1)

    hist_true_plot = np.clip(hist_true, xmin, xmax) if CLIP_HIST_TO_XLIM else hist_true

    hist_mf_plot = np.clip(hist_mf, xmin, xmax) if CLIP_HIST_TO_XLIM else hist_mf

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    ax_hist, ax_box = axes

    ax_hist.hist(
        hist_mf_plot,
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLOR_MF,
        label=f"mean-field Σ\nmedian={hist_mf_median:.3g}",
    )

    ax_hist.hist(
        hist_true_plot,
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLOR_TRUE,
        label=f"real/full Σ\nmedian={hist_true_median:.3g}",
    )

    ax_hist.axvline(
        hist_mf_median,
        linestyle="--",
        linewidth=2.0,
        color=COLOR_MF,
    )

    ax_hist.axvline(
        hist_true_median,
        linestyle="--",
        linewidth=2.0,
        color=COLOR_TRUE,
    )

    ax_hist.axvline(0.0, linestyle=":", linewidth=1.5, color="black", alpha=0.75)

    ax_hist.set_xlim(xmin, xmax)

    ax_hist.set_xlabel(
        r"Held-out Pearson after removing MALAT1/high-variance genes: "
        r"$\rho(\Delta x_{\rm test}, \hat a \Sigma_{{\rm test},g})$"
    )

    ax_hist.set_ylabel("probability density")

    ax_hist.set_title(
        f"High-variance-filtered held-out forward Pearson\n"
        f"threshold={EXPRESSION_THRESHOLD}; splits={N_SPLITS}"
    )

    ax_hist.legend(frameon=False)

    y_mf = dataset_means["pearson_mf_test_median"].to_numpy(float)

    y_true = dataset_means["pearson_true_test_median"].to_numpy(float)

    dataset_mf_median = safe_nanmedian(y_mf)

    dataset_true_median = safe_nanmedian(y_true)

    rng = np.random.default_rng(0)

    x_mf = np.ones_like(y_mf) * 1.0

    x_true = np.ones_like(y_true) * 2.0

    x_mf_jit = x_mf + rng.normal(0, 0.035, size=len(x_mf))

    x_true_jit = x_true + rng.normal(0, 0.035, size=len(x_true))

    for ym, yt in zip(y_mf, y_true):
        if np.isfinite(ym) and np.isfinite(yt):
            ax_box.plot(
                [1.0, 2.0],
                [ym, yt],
                linewidth=0.8,
                alpha=0.35,
                color=COLOR_PAIR,
                zorder=1,
            )

    ax_box.scatter(
        x_mf_jit,
        y_mf,
        s=45,
        alpha=0.75,
        color=COLOR_MF,
        edgecolor="none",
        label="dataset medians",
        zorder=2,
    )

    ax_box.scatter(
        x_true_jit,
        y_true,
        s=45,
        alpha=0.75,
        color=COLOR_TRUE,
        edgecolor="none",
        zorder=2,
    )

    ax_box.scatter(
        [1.0],
        [dataset_mf_median],
        s=190,
        color=COLOR_MF,
        edgecolor=COLOR_MEDIAN_MARKER_EDGE,
        linewidth=1.3,
        marker="s",
        zorder=5,
        label=f"MF median={dataset_mf_median:.3g}",
    )

    ax_box.scatter(
        [2.0],
        [dataset_true_median],
        s=190,
        color=COLOR_TRUE,
        edgecolor=COLOR_MEDIAN_MARKER_EDGE,
        linewidth=1.3,
        marker="D",
        zorder=5,
        label=f"real/full median={dataset_true_median:.3g}",
    )

    ax_box.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    ax_box.set_xlim(0.55, 2.45)

    ax_box.set_xticks([1.0, 2.0])

    ax_box.set_xticklabels(["mean-field Σ", "real/full Σ"])

    ax_box.set_ylabel("median held-out Pearson per dataset")

    ax_box.set_title(f"Dataset-median held-out Pearson\nn={len(dataset_means):,} datasets")

    ax_box.grid(axis="y", alpha=0.25)

    ax_box.legend(frameon=False)

    fig.suptitle(
        "CIPHER forward problem after removing MALAT1 / high-variance genes",
        fontsize=15,
        y=1.03,
    )

    plt.tight_layout()

    png_path = OUTROOT / OUT_FIG_PNG

    svg_path = OUTROOT / OUT_FIG_SVG

    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")

    plt.savefig(svg_path, bbox_inches="tight")

    plt.show()

    print(f"[saved] {png_path}")

    print(f"[saved] {svg_path}")

    summary = {
        "precompute_root": str(root),
        "expression_threshold": float(EXPRESSION_THRESHOLD),
        "threshold_tag": threshold_tag,
        "n_folders": int(len(folders)),
        "n_success": int(len(all_results)),
        "n_errors": int(len(all_errors)),
        "results": all_results,
        "errors": all_errors,
        "merged_csv": str(merged_path),
        "dataset_means_csv": str(dataset_means_path),
        "all_removed_genes_csv": None if all_removed_genes_path is None else str(all_removed_genes_path),
        "figure_png": str(png_path),
        "figure_svg": str(svg_path),
        "gene_removal": {
            "explicit_remove_genes": EXPLICIT_REMOVE_GENES,
            "remove_gene_regexes": REMOVE_GENE_REGEXES,
            "remove_high_variance_genes": bool(REMOVE_HIGH_VARIANCE_GENES),
            "high_var_quantile": float(HIGH_VAR_QUANTILE),
            "max_high_var_genes_to_remove": (
                None if MAX_HIGH_VAR_GENES_TO_REMOVE is None else int(MAX_HIGH_VAR_GENES_TO_REMOVE)
            ),
        },
        "overall": {
            "perturbation_split_level": {
                "pearson_true_train": summarize(all_df["pearson_true_train"]),
                "pearson_true_test": summarize(all_df["pearson_true_test"]),
                "pearson_mf_train": summarize(all_df["pearson_mf_train"]),
                "pearson_mf_test": summarize(all_df["pearson_mf_test"]),
                "pearson_true_minus_mf_test": summarize(
                    all_df["pearson_true_test"].to_numpy(float)
                    - all_df["pearson_mf_test"].to_numpy(float)
                ),
                "train_mse_true": summarize(all_df["train_mse_true"]),
                "test_mse_true": summarize(all_df["test_mse_true"]),
                "train_mse_mf": summarize(all_df["train_mse_mf"]),
                "test_mse_mf": summarize(all_df["test_mse_mf"]),
            },
            "dataset_level": {
                "pearson_true_test_median_per_dataset": summarize(dataset_means["pearson_true_test_median"]),
                "pearson_mf_test_median_per_dataset": summarize(dataset_means["pearson_mf_test_median"]),
                "pearson_true_minus_mf_median_per_dataset": summarize(dataset_means["pearson_true_minus_mf_median"]),
                "pearson_true_test_mean_per_dataset": summarize(dataset_means["pearson_true_test_mean"]),
                "pearson_mf_test_mean_per_dataset": summarize(dataset_means["pearson_mf_test_mean"]),
                "pearson_true_minus_mf_mean_per_dataset": summarize(dataset_means["pearson_true_minus_mf_mean"]),
            },
        },
        "config": {
            "train_frac": float(TRAIN_FRAC),
            "n_splits": int(N_SPLITS),
            "split_seed": int(SPLIT_SEED),
            "batch_size": int(BATCH_SIZE),
            "plot_metric": "median",
            "plot_order": ["mean-field Sigma", "real/full Sigma"],
            "colors": {
                "mean_field": COLOR_MF,
                "real_full": COLOR_TRUE,
            },
            "pearson_definition": (
                "MALAT1 and top high-variance genes are removed from dx=xu-x0 and Sigma rows. "
                "Perturbations whose target gene is removed are excluded. "
                "Fit scalar a on train genes using "
                "a=<dx_train,Sigma_col_train>/<Sigma_col_train,Sigma_col_train>. "
                "Evaluate Pearson(dx_test, a*Sigma_col_test). "
                "Computed for both real/full Sigma and mean-field Sigma using identical cleaned-gene splits."
            ),
        },
    }

    summary_path = OUTROOT / OUT_SUMMARY_JSON

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print("\n" + "=" * 110)

    print("DONE")

    print(f"successful datasets:      {len(all_results)}")

    print(f"errored datasets:         {len(all_errors)}")

    print(f"merged csv:               {merged_path}")

    print(f"dataset summaries csv:    {dataset_means_path}")

    print(f"all removed genes csv:    {all_removed_genes_path}")

    print(f"summary json:             {summary_path}")

    print("=" * 110)
