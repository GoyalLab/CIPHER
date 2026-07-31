"""Fig S13D -- covariance eigenspectrum across preprocessing normalizations.

Computes and plots the eigenspectrum (eigenvalue vs rank, log-log) of the control
gene-gene covariance under several preprocessing normalizations across the perturb-seq
datasets. The light primary path (plot_all_normalizations_overlay) eigendecomposes
precomputed ridge covariances (Sigma_full_ridge.npy) for each dataset x normalization and
overlays their spectra colored by normalization (raw / pflog / log1p / frequency /
log1CP10k) on one panel. The heavy from-raw path (compute_three_way_eigenspectra) loads
each raw h5ad, infers controls, fits the NB/PFlog dispersion alpha, computes three
covariances (raw, PFlog/Lior shifted-CLR, library-size-10k) on the same selected genes and
eigendecomposes each; plot_two_row_three_col and plot_three_col_vs_rank replot those saved
per-dataset spectra. The local blockwise covariance accumulator and binned-trend alpha fit
correspond conceptually to cipher.compute_covariance / cipher.fit_pflog_alpha but are not
bit-identical, so the local implementations are kept (no package_swaps apply).

Helpers in notebooks/src (not part of the cipher package). Config constants are module
globals the notebook overrides via R.__dict__.update; DATA_DIR/SUPPL/OUTDIR injected.
"""
import os
import re
import gc
import json
import fnmatch
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import anndata as ad

import cipher  # noqa: F401  (conceptual correspondence only; no numerically-identical swaps)

# ------------------------------------------------------------------
# Injected by the notebook config cell via R.__dict__.update:
DATA_DIR = None
SUPPL = None
OUTDIR = None


def plot_all_normalizations_overlay():
    # ============================================================
    # ROBUST ALL-NORMALIZATION EIGENSPECTRA OVERLAY
    #
    # Goal:
    #   One single shared log-log plot:
    #       eigenvalue vs rank
    #
    #   All available normalization modes are included.
    #   Color is determined by normalization mode.
    #   Each dataset/mode eigenspectrum is loaded exactly once.
    #
    # Robust discovery:
    #   Searches recursively for:
    #       <anything>/<dataset>/normalizations/<normalization>/Sigma_full_ridge.npy
    #       <anything>/<dataset>/normalizations/<normalization>/Sigma_full_ridge_eigenvalues.npy
    #       <anything>/<dataset>/normalizations/<normalization>/eigenvalues.npy
    #
    # If cached eigenvalues exist, loads them.
    # Otherwise computes from Sigma_full_ridge.npy and caches:
    #       Sigma_full_ridge_eigenvalues.npy
    #
    # This is NOT limited to raw / Lior / libsize.
    # It uses every normalization folder it finds.
    # ============================================================




    # ============================================================
    # CONFIG
    # ============================================================

    # If a previous preprocessing cell defined OUT_ROOT, use it.
    # Otherwise set this explicitly to the root containing dataset/normalizations/mode.
    PREPROC_ROOT = os.path.join(SUPPL, "precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_SAFE_mean_control_ge_1p0")

    SEARCH_ROOTS = [PREPROC_ROOT]

    SAVE_DIR = Path(OUTDIR) / "ALL_NORMALIZATIONS_SIGMA_EIGENSPECTRA_OVERLAY"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    DPI = 300
    FIGSIZE = (8.4, 6.3)

    YMIN = 1e-5
    YMAX = 1e6

    ALPHA = 0.42
    LW = 1.10

    MAX_DATASETS = None
    MAX_RANK_TO_PLOT = None

    RECOMPUTE_EIGENVALUES = False
    SYMMETRIZE_SIGMA_BEFORE_EIGEN = True

    CHECK_FILE_STABLE_BEFORE_LOAD = True
    FILE_STABILITY_SLEEP_SEC = 0.5

    # If not None, only keep normalizations whose name contains one of these.
    NORMALIZATION_FILTER = None
    # NORMALIZATION_FILTER = ["raw", "pflog", "log1p", "frequency", "cp10k"]

    PNG_OUT = SAVE_DIR / "ALL_NORMALIZATIONS__single_plot_eigenvalues_vs_rank.png"
    SVG_OUT = SAVE_DIR / "ALL_NORMALIZATIONS__single_plot_eigenvalues_vs_rank.svg"
    SUMMARY_OUT = SAVE_DIR / "ALL_NORMALIZATIONS__loaded_eigenspectra_summary.tsv"
    MANIFEST_OUT = SAVE_DIR / "ALL_NORMALIZATIONS__manifest.json"


    # ============================================================
    # COLORS / LABELS
    # ============================================================

    CANONICAL_NORMALIZATION_ORDER = [
        "raw",
        "pflog",
        "lior",
        "lior_pflog",
        "log1p",
        "frequency",
        "freq",
        "libsize10k",
        "library_size10k",
        "library_size_10k",
        "cp10k",
        "log1cp10k",
        "log1_cp10k",
        "log1CP10k",
    ]

    CANONICAL_LABELS = {
        "raw": "Raw",
        "pflog": "Lior/PFlog",
        "PFlog": "Lior/PFlog",
        "lior": "Lior/PFlog",
        "lior_pflog": "Lior/PFlog",
        "log1p": "log1p",
        "log1p_raw": "log1p",
        "frequency": "Frequency",
        "freq": "Frequency",
        "libsize10k": "Library size 10k",
        "lib10k": "Library size 10k",
        "library_size10k": "Library size 10k",
        "library_size_10k": "Library size 10k",
        "cp10k": "Library size 10k",
        "CP10k": "Library size 10k",
        "log1CP10k": "log1CP10k",
        "log1cp10k": "log1CP10k",
        "log1_cp10k": "log1CP10k",
    }

    FIXED_COLORS = {
        "raw": "red",
        "lior/pflog": "purple",
        "pflog": "purple",
        "lior": "purple",
        "lior_pflog": "purple",
        "log1p": "green",
        "frequency": "orange",
        "freq": "orange",
        "library size 10k": "blue",
        "libsize10k": "blue",
        "lib10k": "blue",
        "cp10k": "blue",
        "log1cp10k": "blue",
    }


    # ============================================================
    # BASIC HELPERS
    # ============================================================

    def as_path_or_none(x):
        if x is None:
            return None
        try:
            return Path(x)
        except Exception:
            return None


    def file_exists_nonempty(path):
        path = Path(path)
        return path.exists() and path.is_file() and path.stat().st_size > 0


    def npy_exists_valid(path):
        path = Path(path)

        if not file_exists_nonempty(path):
            return False

        try:
            arr = np.load(path, mmap_mode="r")
            ok = arr.size > 0
            del arr
            return bool(ok)
        except Exception:
            return False


    def file_looks_stable(path, sleep_sec=0.5):
        path = Path(path)

        if not file_exists_nonempty(path):
            return False

        if not CHECK_FILE_STABLE_BEFORE_LOAD:
            return True

        try:
            s0 = path.stat().st_size
            time.sleep(sleep_sec)
            s1 = path.stat().st_size
            return s0 == s1 and s1 > 0
        except Exception:
            return False


    def normalize_mode_name(name):
        s = str(name).strip()
        low = s.lower()

        low = low.replace("-", "_")
        low = re.sub(r"\s+", "_", low)

        if low in {"pf_log", "pflog", "p_flog", "lior", "lior_pflog", "pf_log_lior"}:
            return "pflog"

        if low in {"freq", "frequency", "plain_frequency", "library_frequency"}:
            return "frequency"

        if low in {
            "libsize10k",
            "lib10k",
            "library_size10k",
            "library_size_10k",
            "cp10k",
            "counts_per_10k",
            "cpm10k",
        }:
            return "libsize10k"

        if low in {"log1cp10k", "log1_cp10k", "log1p_cp10k", "log1p_libsize10k"}:
            return "log1cp10k"

        if low in {"log1p", "log1p_raw", "plain_log1p"}:
            return "log1p"

        if low == "raw":
            return "raw"

        return low


    def display_label_for_mode(mode):
        key = normalize_mode_name(mode)
        return CANONICAL_LABELS.get(mode, CANONICAL_LABELS.get(key, str(mode)))


    def sort_mode_key(mode):
        key = normalize_mode_name(mode)

        if key in CANONICAL_NORMALIZATION_ORDER:
            return (0, CANONICAL_NORMALIZATION_ORDER.index(key), key)

        return (1, 999, key)


    def get_color_map(modes):
        modes = sorted(list(modes), key=sort_mode_key)

        cmap_names = [
            "tab10",
            "tab20",
            "tab20b",
            "tab20c",
        ]

        dynamic_colors = []

        for cmap_name in cmap_names:
            cmap = plt.get_cmap(cmap_name)
            for i in range(cmap.N):
                dynamic_colors.append(cmap(i))

        color_map = {}
        dyn_i = 0

        for mode in modes:
            key = normalize_mode_name(mode)
            label = display_label_for_mode(mode)
            label_key = label.lower()

            if key in FIXED_COLORS:
                color_map[mode] = FIXED_COLORS[key]
            elif label_key in FIXED_COLORS:
                color_map[mode] = FIXED_COLORS[label_key]
            else:
                color_map[mode] = dynamic_colors[dyn_i % len(dynamic_colors)]
                dyn_i += 1

        return color_map


    def positive_evals(evals):
        evals = np.asarray(evals, dtype=float)
        return evals[np.isfinite(evals) & (evals > 0)]


    def save_json(obj, path):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)


    # ============================================================
    # DISCOVERY
    # ============================================================

    def unique_existing_roots(search_roots):
        roots = []
        seen = set()

        for r in search_roots:
            r = as_path_or_none(r)
            if r is None:
                continue

            candidates = [r]

            try:
                candidates.append(r.resolve())
            except Exception:
                pass

            for c in candidates:
                if c is None:
                    continue

                if not c.exists() or not c.is_dir():
                    continue

                try:
                    key = str(c.resolve())
                except Exception:
                    key = str(c)

                if key in seen:
                    continue

                seen.add(key)
                roots.append(c)

        return roots


    def find_all_normalization_dirs(search_roots):
        roots = unique_existing_roots(search_roots)

        print("[search roots]")
        for r in roots:
            print(f"  - {r}")

        norm_mode_dirs = []

        for root in roots:
            # root itself may be a dataset dir
            if (root / "normalizations").is_dir():
                for mode_dir in (root / "normalizations").iterdir():
                    if mode_dir.is_dir():
                        norm_mode_dirs.append(mode_dir)

            # recursively find every normalizations directory
            for norm_root in root.rglob("normalizations"):
                if not norm_root.is_dir():
                    continue

                for mode_dir in norm_root.iterdir():
                    if mode_dir.is_dir():
                        norm_mode_dirs.append(mode_dir)

        # deduplicate by resolved path
        unique = []
        seen = set()

        for d in norm_mode_dirs:
            try:
                key = str(d.resolve())
            except Exception:
                key = str(d)

            if key in seen:
                continue

            seen.add(key)
            unique.append(d)

        # keep only dirs with sigma or cached eigenspectrum
        kept = []

        for d in unique:
            if (
                file_exists_nonempty(d / "Sigma_full_ridge.npy")
                or file_exists_nonempty(d / "Sigma_full.npy")
                or file_exists_nonempty(d / "Sigma.npy")
                or file_exists_nonempty(d / "covariance.npy")
                or file_exists_nonempty(d / "Sigma_full_ridge_eigenvalues.npy")
                or file_exists_nonempty(d / "Sigma_full_eigenvalues.npy")
                or file_exists_nonempty(d / "Sigma_eigenvalues.npy")
                or file_exists_nonempty(d / "eigenvalues.npy")
            ):
                kept.append(d)

        if NORMALIZATION_FILTER is not None:
            filt = [str(x).lower() for x in NORMALIZATION_FILTER]
            kept = [
                d for d in kept
                if any(f in d.name.lower() for f in filt)
            ]

        kept = sorted(kept, key=lambda p: str(p))

        print(f"\n[normalization dirs found] {len(kept)}")
        for d in kept[:50]:
            print(f"  - dataset={d.parent.parent.name} mode={d.name} :: {d}")
        if len(kept) > 50:
            print(f"  ... plus {len(kept) - 50} more")

        return kept


    def dedupe_dataset_mode_dirs(mode_dirs):
        # key = dataset path + canonical mode
        # choose highest scoring path if duplicates exist
        buckets = {}

        for d in mode_dirs:
            dataset_dir = d.parent.parent
            dataset = dataset_dir.name
            raw_mode = d.name
            mode = normalize_mode_name(raw_mode)

            key = (str(dataset_dir), mode)
            buckets.setdefault(key, []).append(d)

        selected = []

        def score(d):
            s = str(d)
            out = 0

            if file_exists_nonempty(d / "Sigma_full_ridge_eigenvalues.npy"):
                out += 100
            if file_exists_nonempty(d / "eigenvalues.npy"):
                out += 50
            if file_exists_nonempty(d / "Sigma_full_ridge.npy"):
                out += 25

            out -= len(s) * 0.001

            try:
                out += min(d.stat().st_mtime / 1e10, 1.0)
            except Exception:
                pass

            return out

        for _, ds in buckets.items():
            best = sorted(ds, key=score, reverse=True)[0]
            selected.append(best)

        selected = sorted(
            selected,
            key=lambda d: (d.parent.parent.name.lower(), sort_mode_key(d.name)),
        )

        if MAX_DATASETS is not None:
            seen_datasets = []
            limited = []

            for d in selected:
                dataset = d.parent.parent.name
                if dataset not in seen_datasets:
                    seen_datasets.append(dataset)

                if len(seen_datasets) <= int(MAX_DATASETS):
                    limited.append(d)

            selected = limited

        print(f"\n[deduped dataset/mode dirs] {len(selected)}")
        for d in selected[:50]:
            print(f"  - dataset={d.parent.parent.name} mode={normalize_mode_name(d.name)} raw_mode={d.name}")
        if len(selected) > 50:
            print(f"  ... plus {len(selected) - 50} more")

        return selected


    # ============================================================
    # EIGENVALUE LOADING / COMPUTATION
    # ============================================================

    def find_sigma_path(mode_dir):
        mode_dir = Path(mode_dir)

        candidates = [
            mode_dir / "Sigma_full_ridge.npy",
            mode_dir / "Sigma_full.npy",
            mode_dir / "Sigma.npy",
            mode_dir / "covariance.npy",
        ]

        for p in candidates:
            if file_exists_nonempty(p):
                return p

        return None


    def find_cached_eigen_path(mode_dir):
        mode_dir = Path(mode_dir)

        candidates = [
            mode_dir / "Sigma_full_ridge_eigenvalues.npy",
            mode_dir / "Sigma_full_eigenvalues.npy",
            mode_dir / "Sigma_eigenvalues.npy",
            mode_dir / "eigenvalues.npy",
        ]

        for p in candidates:
            if npy_exists_valid(p):
                return p

        return mode_dir / "Sigma_full_ridge_eigenvalues.npy"


    def compute_eigenvalues_from_sigma(sigma_path):
        sigma_path = Path(sigma_path)

        if not file_looks_stable(sigma_path, sleep_sec=FILE_STABILITY_SLEEP_SEC):
            raise RuntimeError(f"Sigma file is missing, empty, or still changing: {sigma_path}")

        S_mmap = np.load(sigma_path, mmap_mode="r")

        if S_mmap.ndim != 2 or S_mmap.shape[0] != S_mmap.shape[1]:
            raise ValueError(f"Sigma is not square: shape={S_mmap.shape}, path={sigma_path}")

        print(f"[eig] loading Sigma into float64: {os.path.basename(str(sigma_path))} shape={S_mmap.shape}")

        S = np.asarray(S_mmap, dtype=np.float64)
        del S_mmap

        if not np.all(np.isfinite(S)):
            bad = np.size(S) - int(np.isfinite(S).sum())
            raise FloatingPointError(f"Sigma has {bad} non-finite entries: {sigma_path}")

        if SYMMETRIZE_SIGMA_BEFORE_EIGEN:
            S = 0.5 * (S + S.T)

        evals = np.linalg.eigvalsh(S)[::-1]
        evals = np.maximum(evals, 0.0)

        del S

        return evals


    def load_or_compute_eigenvalues(mode_dir):
        mode_dir = Path(mode_dir)

        cache_path = find_cached_eigen_path(mode_dir)

        if cache_path.exists() and npy_exists_valid(cache_path) and not RECOMPUTE_EIGENVALUES:
            evals = np.load(cache_path).astype(float)
            return evals, cache_path, "loaded_cached"

        sigma_path = find_sigma_path(mode_dir)

        if sigma_path is None:
            raise FileNotFoundError(f"No Sigma file found in {mode_dir}")

        evals = compute_eigenvalues_from_sigma(sigma_path)

        cache_path = mode_dir / "Sigma_full_ridge_eigenvalues.npy"
        tmp_path = mode_dir / "Sigma_full_ridge_eigenvalues.tmp.npy"

        np.save(tmp_path, evals.astype(np.float32))

        if cache_path.exists():
            cache_path.unlink()

        tmp_path.rename(cache_path)

        print(f"[saved eig] {os.path.basename(str(cache_path))}")

        return evals, cache_path, "computed_from_sigma"


    # ============================================================
    # LOAD EACH DATASET/MODE EXACTLY ONCE
    # ============================================================

    mode_dirs = find_all_normalization_dirs(SEARCH_ROOTS)
    mode_dirs = dedupe_dataset_mode_dirs(mode_dirs)

    if len(mode_dirs) == 0:
        raise RuntimeError(
            "No normalization mode directories found. Expected layout like "
            "<root>/<dataset>/normalizations/<normalization>/Sigma_full_ridge.npy"
        )

    loaded = []
    errors = []

    seen_keys = set()

    for mode_dir in mode_dirs:
        dataset_dir = mode_dir.parent.parent
        dataset = dataset_dir.name
        raw_mode = mode_dir.name
        mode = normalize_mode_name(raw_mode)
        label = display_label_for_mode(raw_mode)

        key = (str(dataset_dir.resolve()), mode)

        if key in seen_keys:
            continue

        seen_keys.add(key)

        try:
            evals, eig_path, status = load_or_compute_eigenvalues(mode_dir)

            evals = np.asarray(evals, dtype=float)
            if evals.ndim != 1 or evals.size == 0:
                raise ValueError(f"Bad eigenvalue array shape={evals.shape}")

            loaded.append(
                {
                    "dataset": dataset,
                    "dataset_dir": str(dataset_dir),
                    "raw_mode": raw_mode,
                    "mode": mode,
                    "label": label,
                    "mode_dir": str(mode_dir),
                    "n_eigs": int(len(evals)),
                    "eig_path": str(eig_path),
                    "status": status,
                    "evals": evals,
                }
            )

            print(
                f"[loaded] dataset={dataset} mode={mode} raw_mode={raw_mode} "
                f"n_eigs={len(evals):,} status={status}"
            )

        except Exception as e:
            msg = {
                "dataset": dataset,
                "raw_mode": raw_mode,
                "mode": mode,
                "mode_dir": str(mode_dir),
                "error": repr(e),
            }
            errors.append(msg)
            print(f"[skip/error] {msg}")

    if len(loaded) == 0:
        raise RuntimeError(
            "Found normalization directories, but no usable eigenspectra/Sigmas were loaded."
        )


    # ============================================================
    # SAVE MANIFEST / SUMMARY
    # ============================================================

    summary_rows = []

    for x in loaded:
        evals = np.asarray(x["evals"], dtype=float)
        evals = evals[np.isfinite(evals)]
        total = np.nansum(evals)

        summary_rows.append(
            {
                "dataset": x["dataset"],
                "mode": x["mode"],
                "raw_mode": x["raw_mode"],
                "label": x["label"],
                "n_eigs": int(len(evals)),
                "trace": float(total),
                "lambda_max": float(np.nanmax(evals)) if len(evals) else np.nan,
                "cumvar_top10": float(np.nansum(evals[:10]) / total) if total > 0 else np.nan,
                "cumvar_top100": float(np.nansum(evals[:100]) / total) if total > 0 else np.nan,
                "status": x["status"],
                "eig_path": os.path.basename(str(x["eig_path"])),
                "mode_dir": os.path.basename(str(x["mode_dir"])),
                "dataset_dir": os.path.basename(str(x["dataset_dir"])),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUT, sep="\t", index=False)
    print(f"[saved summary] {SUMMARY_OUT}")

    manifest = []
    for x in loaded:
        y = dict(x)
        y.pop("evals", None)
        for _k in ("dataset_dir", "mode_dir", "eig_path"):
            if _k in y and y[_k] is not None:
                y[_k] = os.path.basename(str(y[_k]))
        manifest.append(y)

    save_json(
        {
            "save_dir": os.path.basename(str(SAVE_DIR)),
            "preproc_root": os.path.basename(str(PREPROC_ROOT)),
            "search_roots": [os.path.basename(str(x)) for x in SEARCH_ROOTS if x is not None],
            "normalization_filter": NORMALIZATION_FILTER,
            "n_loaded": len(loaded),
            "n_errors": len(errors),
            "loaded": manifest,
            "errors": errors,
        },
        MANIFEST_OUT,
    )
    print(f"[saved manifest] {MANIFEST_OUT}")


    # ============================================================
    # SINGLE-PANEL OVERLAY PLOT: ALL NORMALIZATIONS
    # ============================================================

    modes_present = sorted({x["mode"] for x in loaded}, key=sort_mode_key)
    color_map = get_color_map(modes_present)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    label_used = set()
    xmax = 1

    for mode in modes_present:
        mode_items = [x for x in loaded if x["mode"] == mode]

        # Use the first display label seen for this canonical mode.
        label = mode_items[0]["label"]

        for item in mode_items:
            evals = positive_evals(item["evals"])

            if len(evals) == 0:
                continue

            if MAX_RANK_TO_PLOT is not None:
                evals = evals[:int(MAX_RANK_TO_PLOT)]

            rank = np.arange(1, len(evals) + 1)

            plot_label = label if mode not in label_used else None

            ax.plot(
                rank,
                evals,
                color=color_map[mode],
                alpha=ALPHA,
                linewidth=LW,
                label=plot_label,
            )

            label_used.add(mode)
            xmax = max(xmax, len(evals))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, xmax)
    ax.set_ylim(1e-8, 1e6)

    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue")
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()

    fig.savefig(PNG_OUT, dpi=DPI, bbox_inches="tight")
    fig.savefig(SVG_OUT, bbox_inches="tight")

    plt.show()

    print(f"[saved] {PNG_OUT}")
    print(f"[saved] {SVG_OUT}")

    print("\n[loaded counts by normalization]")
    for mode in modes_present:
        n = sum(x["mode"] == mode for x in loaded)
        label = [x["label"] for x in loaded if x["mode"] == mode][0]
        print(f"  {mode} ({label}): {n}")

    if len(errors) > 0:
        print(f"\n[non-fatal skipped/error modes] {len(errors)}")
        for e in errors[:25]:
            print(e)
        if len(errors) > 25:
            print(f"... plus {len(errors) - 25} more")

    print("\nDONE")
    print(f"plots saved in: {SAVE_DIR}")


def compute_three_way_eigenspectra():
    # ============================================================
    # RAW vs LIOR/PFLOG vs LIBRARY-SIZE-10K covariance eigenspectra
    #
    # For each dataset:
    #   1. Load raw counts.
    #   2. Infer controls, or use all cells for control-only files.
    #   3. Compute raw control mean/variance.
    #   4. Fit Lior/Pachter alpha to binned mean-variance trend:
    #
    #          Var = Mu + alpha * Mu^2
    #
    #      using denoised binned trend points:
    #
    #          genes -> bin by log10(mean) -> median mean/variance per bin
    #          v_bin ≈ mu_bin + alpha * mu_bin^2
    #
    #   5. Compute three covariance matrices on the SAME selected genes/cells:
    #
    #          Sigma_raw       = cov(raw counts)
    #          Sigma_pflog     = cov(log(x + 1/(4alpha)) - cell mean)
    #          Sigma_lib10k    = cov(10000 * x / library_size)
    #
    #   6. Eigendecompose each covariance.
    #   7. Plot eigenspectrum vs rank in one row:
    #
    #          [raw sigma] [Lior/PFlog sigma] [library-size 10k sigma]
    #
    #   8. Save eigenvalues, spectrum CSV, summary, and plots.
    # ============================================================





    # ============================================================
    # CONFIG
    # ============================================================

    SCRIPT_VERSION = "three_way_eigenspectrum_raw_pflog_lib10k_v1"

    ALPHA_NOTATION = "Lior/Pachter: Var = mu + alpha * mu^2; PFlog pseudocount = 1/(4*alpha)"
    ALPHA_FIT_DESCRIPTION = "alpha fitted by least-squares to denoised log-mean-binned mean-variance trend points"

    OUT_ROOT = Path(OUTDIR) / "RAW_vs_LIOR_PFLOG_vs_LIBSIZE10K_COVARIANCE_EIGENSPECTRA"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # SKIP / OVERWRITE
    # ------------------------------------------------------------

    SKIP_IF_DATASET_OUTPUT_EXISTS = True
    OVERWRITE_EXISTING_DATASETS = False
    LOAD_SKIPPED_DATASET_RESULTS = True

    REQUIRED_DATASET_OUTPUT_KEYS_FOR_SKIP = [
        "summary_json",
        "gene_stats_csv",
        "alpha_bin_csv",
        "genes_npy",
        "raw_eigenvalues_npy",
        "pflog_eigenvalues_npy",
        "lib10k_eigenvalues_npy",
        "spectrum_csv",
    ]

    # ------------------------------------------------------------
    # DATASET DISCOVERY
    # ------------------------------------------------------------

    H5AD_SEARCH_ROOTS = [Path(DATA_DIR)]

    EXPLICIT_DATASET_FILES = [
        "TianKampmann2019_iPSC.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "akana_etal_2026_crispra_perturbseq.h5ad",
        "schemidt_etal_2022_crispra_perturbseq.h5ad",
        "XAtlas2025_HEK293T_filtered.h5ad",
        "Marson2025_D3_Stim8hr_filtered.h5ad",
        "Marson2025_D4_Stim48hr_filtered.h5ad",
        "Marson2025_D1_Stim48hr_filtered.h5ad",
        "Marson2025_D1_Rest_filtered.h5ad",
        "Marson2025_D4_Stim8hr_filtered.h5ad",
        "Marson2025_D1_Stim8hr_filtered.h5ad",
        "Marson2025_D4_Rest_filtered.h5ad",
        "Marson2025_D2_Stim48hr_filtered.h5ad",
        "Marson2025_D3_Stim48hr_filtered.h5ad",
        "Marson2025_D3_Rest_filtered.h5ad",
        "Marson2025_D2_Stim8hr_filtered.h5ad",
        "XAtlas2025_HCT116_filtered.h5ad",
        "kaden25_rpe1_ctrl_10k_min100_greedy_4gb.h5ad",
        "kaden25_fibroblast_ctrl_10k_min100_greedy_4gb.h5ad",
    ]

    AUTO_INCLUDE_PATTERNS = [
        "*Replogle*.h5ad",
        "*replogle*.h5ad",
        "*Weissman2022*.h5ad",
        "*Tian*.h5ad",
        "*tian*.h5ad",
        "*Kampmann*.h5ad",
        "*NormanWeissman2019*.h5ad",
        "*Norman*.h5ad",
    ]

    EXCLUDE_SUBSAMPLED_FILES = True
    EXCLUDE_NAME_PATTERNS = [
        "*subsample*",
        "*all_ctrl*",
        "*scdesign3*",
        "*synthetic*",
    ]

    # If False, explicitly listed kaden ctrl_10k files are kept.
    EXCLUDE_GREEDY_AUTO_FILES = False

    # ------------------------------------------------------------
    # MATRIX SOURCE
    # ------------------------------------------------------------

    RAW_LAYER = None
    PREFER_ADATA_RAW = True

    RAW_LAYER_CANDIDATES = [
        "counts",
        "raw_counts",
        "raw",
        "X_counts",
        "umi",
        "UMI",
        "count",
        "Count",
    ]

    GENE_NAME_COL_CANDIDATES = [
        "gene_name",
        "gene_names",
        "gene_symbol",
        "gene_symbols",
        "feature_name",
        "feature_names",
        "symbol",
        "symbols",
        "name",
    ]

    # ------------------------------------------------------------
    # CELL SELECTION
    # ------------------------------------------------------------

    USE_CONTROLS_ONLY = True
    FALLBACK_TO_ALL_CELLS = False
    AUTO_USE_ALL_CELLS_FOR_CONTROL_ONLY_FILES = True

    # ------------------------------------------------------------
    # COVARIANCE GENE SELECTION
    # ------------------------------------------------------------

    MIN_MEAN_FOR_COV = 0.10
    MAX_GENES_FOR_COV = 6000
    GENE_SELECTION_MODE = "highest_mean"  # "highest_mean" or "highest_variance"

    # ------------------------------------------------------------
    # LIOR/PACHTER ALPHA FIT
    # ------------------------------------------------------------

    ALPHA_FIT_MODE = "binned_trend_least_squares"
    N_ALPHA_BINS = 40
    MIN_GENES_PER_ALPHA_BIN = 8

    FIT_ONLY_OVER_POISSON_BINS_IF_AVAILABLE = True
    MIN_OVER_POISSON_BINS_FOR_FIT = 3

    MIN_ALPHA = 1e-12
    EPS = 1e-12

    # ------------------------------------------------------------
    # LIBRARY SIZE NORMALIZATION
    # ------------------------------------------------------------

    LIBSIZE_TARGET = 10000.0

    # ------------------------------------------------------------
    # COMPUTE
    # ------------------------------------------------------------

    MOMENT_BATCH_CELLS = 4096
    COV_BATCH_CELLS_RAW = 1024
    COV_BATCH_CELLS_PFLOG = 512
    COV_BATCH_CELLS_LIB10K = 1024

    SAVE_COVARIANCE_MATRICES = False

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------

    DPI = 300
    TOP_K_CUMVAR_REPORT = [10, 25, 50, 100, 250, 500]


    # ============================================================
    # FILE DISCOVERY / OUTPUT PATHS
    # ============================================================

    def matches_any_pattern(name, patterns):
        name = str(name)
        return any(fnmatch.fnmatch(name, pat) for pat in patterns)


    def find_file_by_name(filename):
        filename = str(filename)
        direct = Path(filename)

        if direct.exists():
            return direct

        hits = []
        for root in H5AD_SEARCH_ROOTS:
            root = Path(root)
            if root.exists():
                hits.extend(root.rglob(filename))

        hits = sorted(set(hits), key=lambda p: (len(str(p)), str(p)))
        return hits[0] if hits else None


    def discover_h5ad_files():
        explicit_paths = []

        for fname in EXPLICIT_DATASET_FILES:
            p = find_file_by_name(fname)
            if p is not None:
                explicit_paths.append(p)

        auto_files = []

        for root in H5AD_SEARCH_ROOTS:
            root = Path(root)
            if not root.exists():
                continue

            for pat in AUTO_INCLUDE_PATTERNS:
                for p in root.rglob(pat):
                    if p.suffix == ".h5ad":
                        auto_files.append(p)

        explicit_paths = sorted(
            set(explicit_paths),
            key=lambda p: EXPLICIT_DATASET_FILES.index(p.name) if p.name in EXPLICIT_DATASET_FILES else 999999,
        )
        auto_files = sorted(set(auto_files), key=lambda p: str(p))

        kept_auto = []
        skipped = []

        for p in auto_files:
            if p in explicit_paths:
                continue

            name_full = str(p)
            exclude_patterns = list(EXCLUDE_NAME_PATTERNS)

            if not EXCLUDE_GREEDY_AUTO_FILES:
                exclude_patterns = [
                    pat for pat in exclude_patterns
                    if pat not in {"*greedy*", "*ctrl_10k*"}
                ]

            if EXCLUDE_SUBSAMPLED_FILES and matches_any_pattern(name_full, exclude_patterns):
                skipped.append(p)
            else:
                kept_auto.append(p)

        out = explicit_paths + kept_auto

        seen = set()
        out2 = []

        for p in out:
            rp = str(p.resolve()) if p.exists() else str(p)
            if rp not in seen:
                seen.add(rp)
                out2.append(p)

        print("\n[discovery] files to process:")
        for p in out2:
            print(f"  - {p}")

        if skipped:
            print("\n[discovery] skipped derived/subsampled auto-discovered files:")
            for p in skipped[:30]:
                print(f"  - {p}")
            if len(skipped) > 30:
                print(f"  ... plus {len(skipped) - 30} more")

        if len(out2) == 0:
            raise FileNotFoundError("No h5ad files found. Check H5AD_SEARCH_ROOTS / patterns.")

        return out2


    def dataset_name_from_path(path):
        name = Path(path).stem
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


    def expected_dataset_outputs(h5ad_path):
        dataset = dataset_name_from_path(h5ad_path)
        outdir = OUT_ROOT / dataset

        return {
            "outdir": outdir,
            "summary_json": outdir / f"{dataset}__summary.json",
            "gene_stats_csv": outdir / f"{dataset}__gene_mean_variance_LIOR_ALPHA_BINNED_TREND_stats.csv",
            "alpha_bin_csv": outdir / f"{dataset}__LIOR_ALPHA_binned_trend_points.csv",
            "genes_npy": outdir / f"{dataset}__covariance_genes.npy",

            "raw_eigenvalues_npy": outdir / f"{dataset}__raw_covariance_eigenvalues.npy",
            "pflog_eigenvalues_npy": outdir / f"{dataset}__PFlog_covariance_eigenvalues.npy",
            "lib10k_eigenvalues_npy": outdir / f"{dataset}__libsize10k_covariance_eigenvalues.npy",

            "spectrum_csv": outdir / f"{dataset}__raw_PFlog_libsize10k_covariance_eigenspectrum.csv",

            "raw_covariance_npy": outdir / f"{dataset}__raw_covariance.npy",
            "pflog_covariance_npy": outdir / f"{dataset}__PFlog_covariance.npy",
            "lib10k_covariance_npy": outdir / f"{dataset}__libsize10k_covariance.npy",
        }


    def file_exists_nonempty(path):
        path = Path(path)
        return path.exists() and path.is_file() and path.stat().st_size > 0


    def dataset_outputs_complete(h5ad_path, verbose=True):
        dataset = dataset_name_from_path(h5ad_path)
        outputs = expected_dataset_outputs(h5ad_path)

        missing = []
        for key in REQUIRED_DATASET_OUTPUT_KEYS_FOR_SKIP:
            p = outputs[key]
            if not file_exists_nonempty(p):
                missing.append((key, p))

        if missing:
            if verbose:
                print(f"[skip check] {dataset} incomplete; missing:")
                for key, p in missing:
                    print(f"  - {key}: {p}")
            return False

        try:
            with open(outputs["summary_json"], "r") as f:
                summary = json.load(f)
        except Exception as e:
            if verbose:
                print(f"[skip check] summary JSON unreadable for {dataset}: {repr(e)}")
            return False

        if summary.get("script_version") != SCRIPT_VERSION:
            if verbose:
                print(f"[skip check] {dataset} exists but script_version differs.")
                print(f"  found:    {summary.get('script_version')}")
                print(f"  required: {SCRIPT_VERSION}")
            return False

        try:
            spectrum_df = pd.read_csv(outputs["spectrum_csv"])
            if len(spectrum_df) == 0:
                if verbose:
                    print(f"[skip check] spectrum CSV has zero rows for {dataset}")
                return False
        except Exception as e:
            if verbose:
                print(f"[skip check] spectrum CSV unreadable for {dataset}: {repr(e)}")
            return False

        if verbose:
            print(f"[skip check] {dataset} complete; will skip.")

        return True


    def load_existing_dataset_outputs(h5ad_path):
        outputs = expected_dataset_outputs(h5ad_path)

        with open(outputs["summary_json"], "r") as f:
            summary = json.load(f)

        evals_raw = np.load(outputs["raw_eigenvalues_npy"])
        evals_pflog = np.load(outputs["pflog_eigenvalues_npy"])
        evals_lib10k = np.load(outputs["lib10k_eigenvalues_npy"])

        return {
            "dataset": dataset_name_from_path(h5ad_path),
            "summary": summary,
            "evals_raw": evals_raw,
            "evals_pflog": evals_pflog,
            "evals_lib10k": evals_lib10k,
        }


    # ============================================================
    # H5AD / GENE HELPERS
    # ============================================================

    def read_h5ad_memory(path):
        return ad.read_h5ad(str(path))


    def clean_str_array(x):
        out = []

        for v in np.asarray(x, dtype=object):
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            out.append(str(v))

        return np.asarray(out, dtype=object)


    def looks_like_ensembl(names):
        names = clean_str_array(names)

        if len(names) == 0:
            return False

        n = min(len(names), 5000)
        frac = np.mean([
            str(x).startswith(("ENSG", "ENSMUSG", "ENSDARG", "ENS"))
            for x in names[:n]
        ])

        return frac > 0.5


    def extract_gene_names(var_df, var_names):
        var_names = clean_str_array(var_names)

        if not looks_like_ensembl(var_names):
            return var_names

        for col in GENE_NAME_COL_CANDIDATES:
            if col in var_df.columns:
                vals = clean_str_array(var_df[col].values)
                nonempty = np.mean([(v != "") and (v.lower() != "nan") for v in vals])

                if nonempty > 0.8:
                    print(f"[genes] var_names look like Ensembl IDs; using var[{col!r}]")
                    return vals

        print("[genes] WARNING: var_names look like Ensembl IDs but no symbol column was found.")
        return var_names


    def make_unique_first_indices(names):
        first = {}
        keep = []

        for i, g in enumerate(names):
            g = str(g)

            if g == "" or g.lower() == "nan":
                continue

            if g not in first:
                first[g] = i
                keep.append(i)

        return np.asarray(keep, dtype=np.int64)


    def get_count_matrix_and_genes(adata):
        if RAW_LAYER is not None:
            if RAW_LAYER not in adata.layers:
                raise KeyError(f"RAW_LAYER={RAW_LAYER!r} not found in adata.layers")

            X = adata.layers[RAW_LAYER]
            genes = extract_gene_names(adata.var, adata.var_names)
            source = f"layer:{RAW_LAYER}"

        else:
            layer_hits = [k for k in RAW_LAYER_CANDIDATES if k in adata.layers]

            if len(layer_hits) > 0:
                layer = layer_hits[0]
                X = adata.layers[layer]
                genes = extract_gene_names(adata.var, adata.var_names)
                source = f"layer:{layer}"

            elif PREFER_ADATA_RAW and adata.raw is not None:
                X = adata.raw.X
                genes = extract_gene_names(adata.raw.var, adata.raw.var_names)
                source = "adata.raw.X"

            else:
                X = adata.X
                genes = extract_gene_names(adata.var, adata.var_names)
                source = "adata.X"

        if sp.issparse(X):
            X = X.tocsr()

        return X, genes, source


    def check_raw_like_matrix(X, label="X"):
        if sp.issparse(X):
            data = X.data

            if data.size == 0:
                print(f"[raw check] {label}: sparse with no nonzero entries?")
                return

            sample = data[: min(data.size, 100000)]

        else:
            flat = np.asarray(X).ravel()
            sample = flat[: min(flat.size, 100000)]

        sample = np.asarray(sample)
        sample = sample[np.isfinite(sample)]

        if sample.size == 0:
            print(f"[raw check] {label}: no finite sampled entries")
            return

        mn = float(sample.min())
        mx = float(sample.max())
        non_integer_frac = float(np.mean(np.abs(sample - np.rint(sample)) > 1e-6))

        print(
            f"[raw check] {label}: sampled min={mn:.4g}, max={mx:.4g}, "
            f"non-integer fraction={non_integer_frac:.4g}"
        )

        if mn < -1e-8:
            print("[raw check] WARNING: negative values found; this is not raw counts.")

        if non_integer_frac > 0.01:
            print("[raw check] WARNING: many non-integers found; this may not be raw counts.")


    def json_default(o):
        if isinstance(o, np.integer):
            return int(o)

        if isinstance(o, np.floating):
            return float(o)

        if isinstance(o, np.ndarray):
            return o.tolist()

        return str(o)


    # ============================================================
    # CONTROL INFERENCE
    # ============================================================

    CONTROL_CANONICAL = {
        "control",
        "ctrl",
        "ntc",
        "nt",
        "neg",
        "negative",
        "negative control",
        "neg control",
        "non-targeting",
        "nontargeting",
        "non targeting",
        "non-target",
        "nontarget",
        "non target",
        "safe-targeting",
        "safe targeting",
        "scramble",
        "scrambled",
        "intergenic",
        "empty",
        "mock",
        "vehicle",
        "none",
        "nan",
    }


    def normalize_label(s):
        s = str(s).strip()
        s = re.sub(r"\s+", " ", s)
        return s


    def is_control_token(t):
        z = normalize_label(t).lower()
        z2 = z.replace("_", " ").replace("-", " ")
        z3 = z.replace("_", "-").replace(" ", "-")

        if z in CONTROL_CANONICAL or z2 in CONTROL_CANONICAL or z3 in CONTROL_CANONICAL:
            return True

        if re.fullmatch(r"ctrl\d*", z):
            return True

        if re.fullmatch(r"control\d*", z):
            return True

        if "non-target" in z or "nontarget" in z or "non target" in z:
            return True

        if "negative control" in z or "neg control" in z:
            return True

        return False


    def is_control_label(s):
        s = normalize_label(s)

        if is_control_token(s):
            return True

        toks = re.split(r"[+|;,/]+", s)
        toks = [t.strip() for t in toks if t.strip()]

        if len(toks) > 1 and all(is_control_token(t) for t in toks):
            return True

        return False


    def infer_control_column(obs):
        rows = []

        for col in obs.columns:
            s = obs[col]

            dtype_ok = (
                pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
            )

            if not dtype_ok:
                continue

            vals = s.astype(str)
            unique_vals = vals.dropna().unique()

            if len(unique_vals) < 2:
                continue

            if len(unique_vals) > max(50000, 0.5 * len(vals)):
                continue

            vals_test = unique_vals[: min(len(unique_vals), 20000)]
            n_ctrl_unique = sum(is_control_label(v) for v in vals_test)

            if n_ctrl_unique == 0:
                continue

            ctrl_mask = vals.map(is_control_label).to_numpy(bool)
            n_ctrl_cells = int(ctrl_mask.sum())

            score = 1000 * n_ctrl_unique + n_ctrl_cells

            rows.append(
                {
                    "column": col,
                    "n_unique": int(len(unique_vals)),
                    "n_control_like_unique": int(n_ctrl_unique),
                    "n_control_cells": n_ctrl_cells,
                    "score": int(score),
                }
            )

        if not rows:
            return None, None

        cand = pd.DataFrame(rows).sort_values("score", ascending=False)

        print("[infer_control_column] top candidates:")
        print(cand.head(10).to_string(index=False))

        best_col = cand.iloc[0]["column"]
        vals = obs[best_col].astype(str)
        ctrl_mask = vals.map(is_control_label).to_numpy(bool)

        print(f"[infer_control_column] using: {best_col}")

        return best_col, ctrl_mask


    def looks_like_control_only_file(path):
        s = str(path).lower()
        return (
            "ctrl" in s
            or "control" in s
            or "all_ctrl" in s
            or "ctrl_10k" in s
        )


    # ============================================================
    # MOMENTS AND LIOR/PACHTER ALPHA FIT
    # ============================================================

    def compute_raw_gene_moments(X, row_idx, col_idx, batch_size=MOMENT_BATCH_CELLS):
        row_idx = np.asarray(row_idx, dtype=np.int64)
        col_idx = np.asarray(col_idx, dtype=np.int64)

        p = len(col_idx)
        n = 0
        s1 = np.zeros(p, dtype=np.float64)
        s2 = np.zeros(p, dtype=np.float64)

        for start in tqdm(range(0, len(row_idx), batch_size), desc="raw mean/var chunks", ncols=100):
            rows = row_idx[start:start + batch_size]
            B = X[rows, :][:, col_idx]

            if sp.issparse(B):
                s1 += np.asarray(B.sum(axis=0)).ravel()
                s2 += np.asarray(B.multiply(B).sum(axis=0)).ravel()
                n += B.shape[0]
            else:
                B = np.asarray(B, dtype=np.float64)
                s1 += B.sum(axis=0)
                s2 += np.square(B).sum(axis=0)
                n += B.shape[0]

        mu = s1 / max(n, 1)

        if n > 1:
            var = (s2 - n * mu * mu) / (n - 1)
        else:
            var = np.full(p, np.nan)

        var = np.maximum(var, 0.0)

        return mu, var, n


    def make_log_mean_bins(mu, var, n_bins=N_ALPHA_BINS, min_genes_per_bin=MIN_GENES_PER_ALPHA_BIN):
        mu = np.asarray(mu, dtype=np.float64)
        var = np.asarray(var, dtype=np.float64)

        valid_mask = (
            np.isfinite(mu)
            & np.isfinite(var)
            & (mu > 0)
            & (var > 0)
        )

        if valid_mask.sum() < 10:
            return pd.DataFrame()

        mu_use = mu[valid_mask]
        var_use = var[valid_mask]
        log_mu = np.log10(mu_use)

        lo = np.nanmin(log_mu)
        hi = np.nanmax(log_mu)

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.DataFrame()

        edges = np.linspace(lo, hi, n_bins + 1)
        bin_id = np.digitize(log_mu, edges) - 1
        bin_id = np.clip(bin_id, 0, n_bins - 1)

        rows = []

        for b in range(n_bins):
            m = bin_id == b
            n_genes = int(m.sum())

            if n_genes < min_genes_per_bin:
                continue

            mu_bin = float(np.median(mu_use[m]))
            var_bin = float(np.median(var_use[m]))

            mu_mean = float(np.mean(mu_use[m]))
            var_mean = float(np.mean(var_use[m]))

            alpha_bin_ratio = float(max(var_bin - mu_bin, 0.0) / max(mu_bin ** 2, EPS))

            rows.append(
                {
                    "bin": int(b),
                    "n_genes": n_genes,
                    "log10_mu_left": float(edges[b]),
                    "log10_mu_right": float(edges[b + 1]),

                    "mu_bin": mu_bin,
                    "var_bin": var_bin,

                    "mu_median": mu_bin,
                    "var_median": var_bin,

                    "mu_mean": mu_mean,
                    "var_mean": var_mean,

                    "over_poisson_bin": bool(var_bin > mu_bin),
                    "alpha_bin_ratio_reference_only": alpha_bin_ratio,
                }
            )

        return pd.DataFrame(rows)


    def fit_nb_alpha(mu, var, mode=ALPHA_FIT_MODE):
        """
        Fit direct Lior/Pachter alpha:

            Var = Mu + alpha * Mu^2

        using binned trend least squares:

            y = v_bin - mu_bin
            x = mu_bin^2
            y ≈ alpha x

        No sqrt. No alpha^2.
        """
        mu = np.asarray(mu, dtype=np.float64)
        var = np.asarray(var, dtype=np.float64)

        full_mask = (
            np.isfinite(mu)
            & np.isfinite(var)
            & (mu > 0)
            & (var > 0)
        )

        over_mask = full_mask & (var > mu)

        bin_df = make_log_mean_bins(mu, var)

        if len(bin_df) < 3:
            warnings.warn("Too few binned trend points. Falling back to gene-level overdispersed LS.")

            m = mu[over_mask]
            v = var[over_mask]

            if len(m) < 5:
                alpha = MIN_ALPHA
                alpha_fit_info = {
                    "alpha_fit_mode": "fallback_min_alpha",
                    "alpha_notation": ALPHA_NOTATION,
                    "alpha_fit_description": ALPHA_FIT_DESCRIPTION,
                    "alpha_source": "too_few_binned_points_and_too_few_overdispersed_genes",
                    "fit_points": "fallback_min_alpha",
                    "fit_y_mode": "none",
                    "n_gene_points_total": int(full_mask.sum()),
                    "n_gene_points_overdispersed": int(over_mask.sum()),
                    "n_bins_total": int(len(bin_df)),
                    "n_bins_used": 0,
                }
                return alpha, full_mask, over_mask, bin_df, alpha_fit_info

            x = m ** 2
            y = v - m

            alpha = float(np.dot(x, y) / max(np.dot(x, x), EPS))
            alpha = float(max(alpha, MIN_ALPHA))

            alpha_fit_info = {
                "alpha_fit_mode": "fallback_gene_level_overdispersed_least_squares",
                "alpha_notation": ALPHA_NOTATION,
                "alpha_fit_description": ALPHA_FIT_DESCRIPTION,
                "alpha_source": "gene_level_overdispersed_points",
                "fit_points": "individual genes with var > mean",
                "fit_y_mode": "v_minus_mu",
                "n_gene_points_total": int(full_mask.sum()),
                "n_gene_points_overdispersed": int(over_mask.sum()),
                "n_bins_total": int(len(bin_df)),
                "n_bins_used": 0,
            }

            return alpha, full_mask, over_mask, bin_df, alpha_fit_info

        if mode != "binned_trend_least_squares":
            raise ValueError(f"Unknown ALPHA_FIT_MODE={mode!r}; use 'binned_trend_least_squares'.")

        m = bin_df["mu_bin"].to_numpy(float)
        v = bin_df["var_bin"].to_numpy(float)

        base_ok = np.isfinite(m) & np.isfinite(v) & (m > 0) & (v > 0)
        over_bin_ok = base_ok & (v > m)

        if FIT_ONLY_OVER_POISSON_BINS_IF_AVAILABLE and int(over_bin_ok.sum()) >= MIN_OVER_POISSON_BINS_FOR_FIT:
            fit_ok = over_bin_ok
            fit_points = "binned trend points with var_bin > mu_bin"
            y_mode = "v_bin_minus_mu_bin"
        else:
            fit_ok = base_ok
            fit_points = "all binned trend points with negative excess clipped to zero"
            y_mode = "max_v_bin_minus_mu_bin_zero"

        m_fit = m[fit_ok]
        v_fit = v[fit_ok]

        x = m_fit ** 2

        if y_mode == "v_bin_minus_mu_bin":
            y = v_fit - m_fit
        else:
            y = np.maximum(v_fit - m_fit, 0.0)

        alpha = float(np.dot(x, y) / max(np.dot(x, x), EPS))
        alpha = float(max(alpha, MIN_ALPHA))

        alpha_fit_info = {
            "alpha_fit_mode": mode,
            "alpha_notation": ALPHA_NOTATION,
            "alpha_fit_description": ALPHA_FIT_DESCRIPTION,
            "alpha_source": "denoised_log_mean_binned_mean_variance_trend",
            "fit_points": fit_points,
            "fit_y_mode": y_mode,
            "n_gene_points_total": int(full_mask.sum()),
            "n_gene_points_overdispersed": int(over_mask.sum()),
            "n_bins_total": int(len(bin_df)),
            "n_bins_over_poisson": int(over_bin_ok.sum()),
            "n_bins_used": int(fit_ok.sum()),
        }

        return alpha, full_mask, over_mask, bin_df, alpha_fit_info


    # ============================================================
    # COVARIANCE HELPERS
    # ============================================================

    def select_covariance_genes(genes, mu, var):
        genes = np.asarray(genes, dtype=object)
        mu = np.asarray(mu, dtype=np.float64)
        var = np.asarray(var, dtype=np.float64)

        keep = np.isfinite(mu) & np.isfinite(var) & (mu >= MIN_MEAN_FOR_COV)
        idx = np.flatnonzero(keep)

        if len(idx) == 0:
            raise RuntimeError("No genes passed MIN_MEAN_FOR_COV.")

        if MAX_GENES_FOR_COV is not None and len(idx) > int(MAX_GENES_FOR_COV):
            if GENE_SELECTION_MODE == "highest_mean":
                score = mu[idx]
            elif GENE_SELECTION_MODE == "highest_variance":
                score = var[idx]
            else:
                raise ValueError(f"Unknown GENE_SELECTION_MODE={GENE_SELECTION_MODE!r}")

            order = np.argsort(score)[::-1]
            idx = idx[order[: int(MAX_GENES_FOR_COV)]]
            idx = np.sort(idx)

        return idx.astype(np.int64)


    def dense_subset(X, rows, cols):
        B = X[rows, :][:, cols]

        if sp.issparse(B):
            return B.toarray()

        return np.asarray(B)


    def row_library_size(X, rows):
        B = X[rows, :]

        if sp.issparse(B):
            lib = np.asarray(B.sum(axis=1)).ravel()
        else:
            lib = np.asarray(B, dtype=np.float64).sum(axis=1)

        lib = np.asarray(lib, dtype=np.float64)
        return lib


    def transform_counts_dense_to_pflog(B, pc):
        """
        PFlog / shifted-CLR:

            y_ci = log(x_ci + pc) - mean_j log(x_cj + pc)

        with:
            pc = 1/(4*alpha)
        """
        B = np.asarray(B, dtype=np.float32)
        Y = np.log(B + np.float32(pc)).astype(np.float32)
        Y -= Y.mean(axis=1, keepdims=True)
        return Y


    def transform_counts_dense_to_libsize10k(B, lib):
        """
        Library-size normalization to total count 10,000:

            z_cg = 10000 * x_cg / sum_j x_cj

        This is NOT log1p. It is the plain size-normalized count matrix.
        """
        B = np.asarray(B, dtype=np.float32)
        lib = np.asarray(lib, dtype=np.float32)

        scale = np.zeros_like(lib, dtype=np.float32)
        ok = np.isfinite(lib) & (lib > 0)
        scale[ok] = np.float32(LIBSIZE_TARGET) / lib[ok]

        Z = B * scale[:, None]
        return Z.astype(np.float32, copy=False)


    def compute_covariance_from_blocks(
        X,
        row_idx,
        col_idx,
        transform=None,
        pc=None,
        batch_size=512,
        desc="covariance chunks",
    ):
        """
        transform:
            None          -> raw counts
            "pflog"       -> Lior/PFlog shifted CLR
            "libsize10k"  -> 10000 * x / library_size
        """
        row_idx = np.asarray(row_idx, dtype=np.int64)
        col_idx = np.asarray(col_idx, dtype=np.int64)

        p = len(col_idx)
        n = 0
        s = np.zeros(p, dtype=np.float64)
        xtx = np.zeros((p, p), dtype=np.float64)

        for start in tqdm(range(0, len(row_idx), batch_size), desc=desc, ncols=100):
            rows = row_idx[start:start + batch_size]
            B = dense_subset(X, rows, col_idx)

            if transform is None:
                Z = B.astype(np.float32, copy=False)

            elif transform == "pflog":
                Z = transform_counts_dense_to_pflog(B, pc)

            elif transform == "libsize10k":
                lib = row_library_size(X, rows)
                Z = transform_counts_dense_to_libsize10k(B, lib)
                del lib

            else:
                raise ValueError(f"Unknown transform={transform!r}")

            s += Z.sum(axis=0, dtype=np.float64)
            xtx += Z.T @ Z
            n += Z.shape[0]

            del B, Z
            gc.collect()

        mean = s / max(n, 1)

        if n > 1:
            cov = (xtx - n * np.outer(mean, mean)) / (n - 1)
        else:
            cov = np.full((p, p), np.nan, dtype=np.float64)

        cov = 0.5 * (cov + cov.T)

        return mean.astype(np.float32), cov.astype(np.float32), n


    def eigenspectrum_from_cov(cov):
        cov64 = np.asarray(cov, dtype=np.float64)
        evals = np.linalg.eigvalsh(cov64)
        evals = evals[::-1]
        evals = np.maximum(evals, 0.0)
        return evals


    def spectrum_summary(evals, prefix):
        evals = np.asarray(evals, dtype=np.float64)
        total = float(np.sum(evals))

        out = {
            f"{prefix}_n_eigs": int(len(evals)),
            f"{prefix}_trace": total,
            f"{prefix}_lambda_max": float(evals[0]) if len(evals) else np.nan,
            f"{prefix}_lambda_median": float(np.median(evals)) if len(evals) else np.nan,
        }

        for k in TOP_K_CUMVAR_REPORT:
            kk = min(k, len(evals))

            if kk > 0 and total > 0:
                out[f"{prefix}_cumvar_top{k}"] = float(np.sum(evals[:kk]) / total)
            else:
                out[f"{prefix}_cumvar_top{k}"] = np.nan

        return out


    # ============================================================
    # PLOTS
    # ============================================================

    def plot_mean_variance_fit(dataset, outdir, mu, var, alpha, over_mask, bin_df):
        fig, ax = plt.subplots(figsize=(5.8, 4.8))

        plot_mask = np.isfinite(mu) & np.isfinite(var) & (mu > 0) & (var > 0)

        ax.scatter(
            mu[plot_mask],
            var[plot_mask],
            s=5,
            alpha=0.16,
            linewidths=0,
            label="genes",
        )

        if over_mask is not None and over_mask.sum() > 0:
            ax.scatter(
                mu[over_mask],
                var[over_mask],
                s=5,
                alpha=0.16,
                linewidths=0,
                label="var > mean",
            )

        if bin_df is not None and len(bin_df) > 0:
            ax.scatter(
                bin_df["mu_bin"],
                bin_df["var_bin"],
                s=38,
                marker="o",
                edgecolor="black",
                linewidth=0.45,
                label="binned mean-var trend",
                zorder=10,
            )

        xmin = max(np.nanmin(mu[plot_mask]), 1e-8)
        xmax = max(np.nanmax(mu[plot_mask]), xmin * 10)

        grid = np.geomspace(xmin, xmax, 600)
        fit = grid + alpha * (grid ** 2)

        ax.plot(grid, grid, linestyle=":", linewidth=1.6, label=r"$v=\mu$")
        ax.plot(
            grid,
            fit,
            linewidth=2.4,
            label=fr"$v=\mu+\alpha\mu^2$, $\alpha={alpha:.4g}$",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("control mean raw count")
        ax.set_ylabel("control variance raw count")
        ax.set_title(dataset)
        ax.legend(frameon=False, fontsize=8)

        fig.tight_layout()

        png = outdir / f"{dataset}__LIOR_ALPHA_binned_trend_fit.png"
        svg = outdir / f"{dataset}__LIOR_ALPHA_binned_trend_fit.svg"

        fig.savefig(png, dpi=DPI)
        fig.savefig(svg)
        plt.show()

        print(f"[saved] {png}")
        print(f"[saved] {svg}")


    def plot_dataset_three_spectra(dataset, outdir, evals_raw, evals_pflog, evals_lib10k):
        spectra = [
            ("Raw Sigma", evals_raw),
            ("Lior/PFlog Sigma", evals_pflog),
            ("Library size 10k Sigma", evals_lib10k),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

        for ax, (title, evals) in zip(axes, spectra):
            evals = np.asarray(evals, dtype=np.float64)
            rank = np.arange(1, len(evals) + 1)

            ax.plot(rank, evals, linewidth=2)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Eigenvalue rank")
            ax.set_ylabel("Eigenvalue")
            ax.set_title(title)

        fig.suptitle(dataset, y=1.02)
        fig.tight_layout()

        png = outdir / f"{dataset}__THREE_PANEL_raw_PFlog_libsize10k_eigenspectrum_vs_rank.png"
        svg = outdir / f"{dataset}__THREE_PANEL_raw_PFlog_libsize10k_eigenspectrum_vs_rank.svg"

        fig.savefig(png, dpi=DPI, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.show()

        print(f"[saved] {png}")
        print(f"[saved] {svg}")


    def plot_all_overlay_three_spectra(all_items):
        if len(all_items) == 0:
            return

        panels = [
            ("Raw Sigma", "evals_raw"),
            ("Lior/PFlog Sigma", "evals_pflog"),
            ("Library size 10k Sigma", "evals_lib10k"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        for ax, (title, key) in zip(axes, panels):
            for item in all_items:
                dataset = item["dataset"]
                evals = np.asarray(item[key], dtype=np.float64)
                rank = np.arange(1, len(evals) + 1)

                ax.plot(rank, evals, linewidth=1.2, alpha=0.75, label=dataset)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Eigenvalue rank")
            ax.set_ylabel("Eigenvalue")
            ax.set_title(title)

        if len(all_items) <= 12:
            axes[-1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.tight_layout()

        png = OUT_ROOT / "ALL_DATASETS__THREE_PANEL_raw_PFlog_libsize10k_eigenspectrum_overlay.png"
        svg = OUT_ROOT / "ALL_DATASETS__THREE_PANEL_raw_PFlog_libsize10k_eigenspectrum_overlay.svg"

        fig.savefig(png, dpi=DPI, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.show()

        print(f"[saved] {png}")
        print(f"[saved] {svg}")

        # Also save trace-normalized overlay in the same 3-panel row.
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        for ax, (title, key) in zip(axes, panels):
            for item in all_items:
                dataset = item["dataset"]
                evals = np.asarray(item[key], dtype=np.float64)
                total = np.sum(evals)
                y = evals / total if total > 0 else evals * np.nan
                rank = np.arange(1, len(y) + 1)

                ax.plot(rank, y, linewidth=1.2, alpha=0.75, label=dataset)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Eigenvalue rank")
            ax.set_ylabel("Fraction of trace")
            ax.set_title(title)

        if len(all_items) <= 12:
            axes[-1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.tight_layout()

        png = OUT_ROOT / "ALL_DATASETS__THREE_PANEL_raw_PFlog_libsize10k_trace_normalized_eigenspectrum_overlay.png"
        svg = OUT_ROOT / "ALL_DATASETS__THREE_PANEL_raw_PFlog_libsize10k_trace_normalized_eigenspectrum_overlay.svg"

        fig.savefig(png, dpi=DPI, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.show()

        print(f"[saved] {png}")
        print(f"[saved] {svg}")


    # ============================================================
    # MAIN DATASET PROCESSOR
    # ============================================================

    def process_dataset(h5ad_path):
        h5ad_path = Path(h5ad_path)
        dataset = dataset_name_from_path(h5ad_path)

        outputs = expected_dataset_outputs(h5ad_path)
        outdir = outputs["outdir"]
        outdir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 120)
        print(f"[dataset] {dataset}")
        print(f"[h5ad]    {os.path.basename(str(h5ad_path))}")
        print("=" * 120)

        adata = read_h5ad_memory(h5ad_path)
        X, genes_all_raw, matrix_source = get_count_matrix_and_genes(adata)

        print(f"[matrix] source={matrix_source}")
        print(f"[matrix] cells={adata.n_obs:,}, genes={len(genes_all_raw):,}, sparse={sp.issparse(X)}")
        check_raw_like_matrix(X, label=matrix_source)

        unique_idx = make_unique_first_indices(genes_all_raw)

        if len(unique_idx) < len(genes_all_raw):
            print(
                f"[genes] keeping first copy of duplicate/nonempty gene symbols: "
                f"{len(genes_all_raw):,} -> {len(unique_idx):,}"
            )

        genes_all = genes_all_raw[unique_idx]
        X_cols_all = unique_idx.astype(np.int64)

        # Choose cells.
        if USE_CONTROLS_ONLY:
            ctrl_col, ctrl_mask = infer_control_column(adata.obs)

            if ctrl_mask is None or ctrl_mask.sum() < 10:
                if AUTO_USE_ALL_CELLS_FOR_CONTROL_ONLY_FILES and looks_like_control_only_file(h5ad_path):
                    print("[cells] control-only filename detected; using all cells as controls.")
                    cell_idx = np.arange(adata.n_obs, dtype=np.int64)
                    cell_source = "all_cells_control_only_file"
                    ctrl_col = None

                elif FALLBACK_TO_ALL_CELLS:
                    print("[cells] WARNING: could not infer enough controls; using all cells.")
                    cell_idx = np.arange(adata.n_obs, dtype=np.int64)
                    cell_source = "all_cells_fallback"
                    ctrl_col = None

                else:
                    raise RuntimeError("Could not infer enough controls and FALLBACK_TO_ALL_CELLS=False.")
            else:
                cell_idx = np.flatnonzero(ctrl_mask).astype(np.int64)
                cell_source = f"controls_from_{ctrl_col}"

        else:
            cell_idx = np.arange(adata.n_obs, dtype=np.int64)
            cell_source = "all_cells"
            ctrl_col = None

        print(f"[cells] covariance cells={len(cell_idx):,} source={cell_source}")

        # Raw mean/variance for alpha + gene selection.
        mu, var, n_mom = compute_raw_gene_moments(
            X=X,
            row_idx=cell_idx,
            col_idx=X_cols_all,
            batch_size=MOMENT_BATCH_CELLS,
        )

        # Lior/Pachter alpha.
        alpha, full_mask, over_mask, bin_df, alpha_info = fit_nb_alpha(mu, var)
        pc = 1.0 / (4.0 * alpha)

        print(f"[alpha] notation={ALPHA_NOTATION}")
        print(f"[alpha] fit description={ALPHA_FIT_DESCRIPTION}")
        print(f"[alpha] mode={alpha_info['alpha_fit_mode']}")
        print(f"[alpha] source={alpha_info['alpha_source']}")
        print(f"[alpha] fit points={alpha_info['fit_points']}")
        print(f"[alpha] gene points total={alpha_info['n_gene_points_total']:,}")
        print(f"[alpha] overdispersed gene points={alpha_info['n_gene_points_overdispersed']:,}")
        print(f"[alpha] bins total={alpha_info['n_bins_total']:,}")
        print(f"[alpha] bins used={alpha_info['n_bins_used']:,}")
        if "n_bins_over_poisson" in alpha_info:
            print(f"[alpha] bins over poisson={alpha_info['n_bins_over_poisson']:,}")
        print(f"[alpha] alpha={alpha:.8g}")
        print(f"[alpha] sqrt(alpha) for reference only={np.sqrt(alpha):.8g}")
        print(f"[alpha] PFlog pseudocount pc=1/(4alpha)={pc:.8g}")

        plot_mean_variance_fit(
            dataset=dataset,
            outdir=outdir,
            mu=mu,
            var=var,
            alpha=alpha,
            over_mask=over_mask,
            bin_df=bin_df,
        )

        # Select same genes for all three covariances.
        keep_idx_local = select_covariance_genes(genes_all, mu, var)
        genes_cov = genes_all[keep_idx_local]
        X_cols_cov = X_cols_all[keep_idx_local]

        print(f"[genes] covariance genes={len(genes_cov):,}")

        np.save(outputs["genes_npy"], genes_cov)
        print(f"[saved] {outputs['genes_npy']}")

        # Save gene-level stats.
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha_ratio_gene_level = np.full(len(mu), np.nan, dtype=np.float64)
            ok_ratio = (
                np.isfinite(mu)
                & np.isfinite(var)
                & (mu > 0)
                & (var > 0)
            )
            alpha_ratio_gene_level[ok_ratio] = (
                np.maximum(var[ok_ratio] - mu[ok_ratio], 0.0)
                / np.maximum(mu[ok_ratio] ** 2, EPS)
            )

        used_for_cov = np.zeros(len(genes_all), dtype=bool)
        used_for_cov[keep_idx_local] = True

        gene_stats = pd.DataFrame(
            {
                "gene": genes_all,
                "mean": mu,
                "variance": var,
                "alpha_ratio_gene_level_reference_only": alpha_ratio_gene_level,
                "valid_for_alpha": full_mask,
                "overdispersed_var_gt_mean": over_mask,
                "used_for_covariance": used_for_cov,
            }
        )

        gene_stats.to_csv(outputs["gene_stats_csv"], index=False)
        print(f"[saved] {outputs['gene_stats_csv']}")

        bin_df.to_csv(outputs["alpha_bin_csv"], index=False)
        print(f"[saved] {outputs['alpha_bin_csv']}")

        # --------------------------------------------------------
        # Three covariance matrices.
        # --------------------------------------------------------

        print("\n[raw covariance]")
        raw_mean, cov_raw, n_raw = compute_covariance_from_blocks(
            X=X,
            row_idx=cell_idx,
            col_idx=X_cols_cov,
            transform=None,
            pc=None,
            batch_size=COV_BATCH_CELLS_RAW,
            desc="raw covariance chunks",
        )

        print("\n[Lior/PFlog covariance]")
        pflog_mean, cov_pflog, n_pflog = compute_covariance_from_blocks(
            X=X,
            row_idx=cell_idx,
            col_idx=X_cols_cov,
            transform="pflog",
            pc=pc,
            batch_size=COV_BATCH_CELLS_PFLOG,
            desc="PFlog covariance chunks",
        )

        print("\n[library-size 10k covariance]")
        lib10k_mean, cov_lib10k, n_lib10k = compute_covariance_from_blocks(
            X=X,
            row_idx=cell_idx,
            col_idx=X_cols_cov,
            transform="libsize10k",
            pc=None,
            batch_size=COV_BATCH_CELLS_LIB10K,
            desc="library-size 10k covariance chunks",
        )

        # --------------------------------------------------------
        # Eigenspectra.
        # --------------------------------------------------------

        print("\n[eigendecomp] raw")
        evals_raw = eigenspectrum_from_cov(cov_raw)

        print("[eigendecomp] Lior/PFlog")
        evals_pflog = eigenspectrum_from_cov(cov_pflog)

        print("[eigendecomp] library-size 10k")
        evals_lib10k = eigenspectrum_from_cov(cov_lib10k)

        np.save(outputs["raw_eigenvalues_npy"], evals_raw)
        np.save(outputs["pflog_eigenvalues_npy"], evals_pflog)
        np.save(outputs["lib10k_eigenvalues_npy"], evals_lib10k)

        print(f"[saved] {outputs['raw_eigenvalues_npy']}")
        print(f"[saved] {outputs['pflog_eigenvalues_npy']}")
        print(f"[saved] {outputs['lib10k_eigenvalues_npy']}")

        spectrum_df = pd.DataFrame(
            {
                "rank": np.arange(1, len(evals_raw) + 1),

                "lambda_raw": evals_raw,
                "lambda_pflog": evals_pflog,
                "lambda_libsize10k": evals_lib10k,

                "lambda_raw_trace_fraction": evals_raw / np.sum(evals_raw) if np.sum(evals_raw) > 0 else np.nan,
                "lambda_pflog_trace_fraction": evals_pflog / np.sum(evals_pflog) if np.sum(evals_pflog) > 0 else np.nan,
                "lambda_libsize10k_trace_fraction": evals_lib10k / np.sum(evals_lib10k) if np.sum(evals_lib10k) > 0 else np.nan,

                "cumvar_raw": np.cumsum(evals_raw) / np.sum(evals_raw) if np.sum(evals_raw) > 0 else np.nan,
                "cumvar_pflog": np.cumsum(evals_pflog) / np.sum(evals_pflog) if np.sum(evals_pflog) > 0 else np.nan,
                "cumvar_libsize10k": np.cumsum(evals_lib10k) / np.sum(evals_lib10k) if np.sum(evals_lib10k) > 0 else np.nan,
            }
        )

        spectrum_df.to_csv(outputs["spectrum_csv"], index=False)
        print(f"[saved] {outputs['spectrum_csv']}")

        if SAVE_COVARIANCE_MATRICES:
            np.save(outputs["raw_covariance_npy"], cov_raw)
            np.save(outputs["pflog_covariance_npy"], cov_pflog)
            np.save(outputs["lib10k_covariance_npy"], cov_lib10k)

            print(f"[saved] {outputs['raw_covariance_npy']}")
            print(f"[saved] {outputs['pflog_covariance_npy']}")
            print(f"[saved] {outputs['lib10k_covariance_npy']}")

        plot_dataset_three_spectra(
            dataset=dataset,
            outdir=outdir,
            evals_raw=evals_raw,
            evals_pflog=evals_pflog,
            evals_lib10k=evals_lib10k,
        )

        summary = {
            "script_version": SCRIPT_VERSION,
            "alpha_notation": ALPHA_NOTATION,
            "alpha_fit_description": ALPHA_FIT_DESCRIPTION,
            "dataset": dataset,
            "h5ad_path": os.path.basename(str(h5ad_path)),
            "matrix_source": matrix_source,
            "cell_source": cell_source,
            "control_column": ctrl_col,

            "n_cells_total": int(adata.n_obs),
            "n_cells_covariance": int(len(cell_idx)),
            "n_cells_raw_covariance": int(n_raw),
            "n_cells_pflog_covariance": int(n_pflog),
            "n_cells_libsize10k_covariance": int(n_lib10k),

            "n_genes_raw_unique": int(len(genes_all)),
            "n_genes_covariance": int(len(genes_cov)),
            "min_mean_for_cov": float(MIN_MEAN_FOR_COV),
            "max_genes_for_cov": int(MAX_GENES_FOR_COV) if MAX_GENES_FOR_COV is not None else None,
            "gene_selection_mode": GENE_SELECTION_MODE,

            "alpha": float(alpha),
            "nb_alpha_lior": float(alpha),
            "sqrt_alpha_reference_only": float(np.sqrt(alpha)),
            "pflog_pseudocount": float(pc),
            "library_size_target": float(LIBSIZE_TARGET),
            "alpha_fit_info": alpha_info,

            **spectrum_summary(evals_raw, "raw"),
            **spectrum_summary(evals_pflog, "pflog"),
            **spectrum_summary(evals_lib10k, "libsize10k"),

            "files": {
                "genes_npy": os.path.basename(str(outputs["genes_npy"])),
                "gene_stats_csv": os.path.basename(str(outputs["gene_stats_csv"])),
                "alpha_bin_csv": os.path.basename(str(outputs["alpha_bin_csv"])),
                "raw_eigenvalues_npy": os.path.basename(str(outputs["raw_eigenvalues_npy"])),
                "pflog_eigenvalues_npy": os.path.basename(str(outputs["pflog_eigenvalues_npy"])),
                "lib10k_eigenvalues_npy": os.path.basename(str(outputs["lib10k_eigenvalues_npy"])),
                "spectrum_csv": os.path.basename(str(outputs["spectrum_csv"])),
            },
        }

        with open(outputs["summary_json"], "w") as f:
            json.dump(summary, f, indent=2, default=json_default)

        print(f"[saved] {outputs['summary_json']}")

        print("\n[summary]")
        print(pd.Series(summary).drop(labels=["files", "alpha_fit_info"], errors="ignore").to_string())

        del adata, X
        del cov_raw, cov_pflog, cov_lib10k
        del raw_mean, pflog_mean, lib10k_mean
        gc.collect()

        return {
            "dataset": dataset,
            "summary": summary,
            "evals_raw": evals_raw,
            "evals_pflog": evals_pflog,
            "evals_lib10k": evals_lib10k,
        }


    # ============================================================
    # RUN ALL DATASETS
    # ============================================================

    h5ad_files = discover_h5ad_files()

    all_items = []
    all_summaries = []
    errors = []
    skipped = []

    for h5ad_path in h5ad_files:
        dataset = dataset_name_from_path(h5ad_path)

        try:
            is_complete = (
                SKIP_IF_DATASET_OUTPUT_EXISTS
                and not OVERWRITE_EXISTING_DATASETS
                and dataset_outputs_complete(h5ad_path, verbose=True)
            )

            if is_complete:
                print("\n" + "-" * 120)
                print(f"[SKIP] {dataset}")
                print(f"[h5ad] {h5ad_path}")
                print("-" * 120)

                skipped.append(
                    {
                        "dataset": dataset,
                        "h5ad_path": os.path.basename(str(h5ad_path)),
                        "reason": "dataset_outputs_complete_three_way_eigenspectrum",
                    }
                )

                if LOAD_SKIPPED_DATASET_RESULTS:
                    try:
                        item_existing = load_existing_dataset_outputs(h5ad_path)
                        all_items.append(item_existing)
                        all_summaries.append(item_existing["summary"])
                        print(f"[loaded existing] {dataset}")
                    except Exception as e:
                        print(f"[WARNING] could not load skipped outputs for {dataset}: {repr(e)}")

                continue

            item = process_dataset(h5ad_path)
            all_items.append(item)
            all_summaries.append(item["summary"])

        except Exception as e:
            print("\n" + "!" * 120)
            print(f"[ERROR] {dataset}")
            print(f"[h5ad]  {h5ad_path}")
            print(repr(e))
            print("!" * 120 + "\n")

            errors.append(
                {
                    "dataset": dataset,
                    "h5ad_path": os.path.basename(str(h5ad_path)),
                    "error": repr(e),
                }
            )

            gc.collect()

    skipped_csv = OUT_ROOT / "ALL_DATASETS__skipped.csv"
    pd.DataFrame(skipped).to_csv(skipped_csv, index=False)
    print(f"[saved] {skipped_csv}")

    errors_csv = OUT_ROOT / "ALL_DATASETS__errors.csv"
    pd.DataFrame(errors).to_csv(errors_csv, index=False)
    print(f"[saved] {errors_csv}")

    if len(all_items) == 0:
        if len(skipped) > 0 and not LOAD_SKIPPED_DATASET_RESULTS:
            print("\nAll datasets were skipped, and LOAD_SKIPPED_DATASET_RESULTS=False.")
            print("No combined summary/plots were generated.")
            print("\nDone.")
        else:
            raise RuntimeError(f"No results available. Errors: {errors}")

    else:
        summary_df = pd.DataFrame(all_summaries)

        summary_csv = OUT_ROOT / "ALL_DATASETS__three_way_covariance_eigenspectrum_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[saved] {summary_csv}")

        summary_json = OUT_ROOT / "ALL_DATASETS__summary.json"

        with open(summary_json, "w") as f:
            json.dump(
                {
                    "script_version": SCRIPT_VERSION,
                    "alpha_notation": ALPHA_NOTATION,
                    "alpha_fit_description": ALPHA_FIT_DESCRIPTION,
                    "library_size_target": float(LIBSIZE_TARGET),
                    "summaries": all_summaries,
                    "errors": errors,
                    "skipped": skipped,
                    "skip_config": {
                        "SKIP_IF_DATASET_OUTPUT_EXISTS": SKIP_IF_DATASET_OUTPUT_EXISTS,
                        "OVERWRITE_EXISTING_DATASETS": OVERWRITE_EXISTING_DATASETS,
                        "LOAD_SKIPPED_DATASET_RESULTS": LOAD_SKIPPED_DATASET_RESULTS,
                        "REQUIRED_DATASET_OUTPUT_KEYS_FOR_SKIP": REQUIRED_DATASET_OUTPUT_KEYS_FOR_SKIP,
                    },
                    "definition": (
                        "Three covariance eigenspectra computed on the same selected genes and cells: "
                        "raw-count covariance, Lior/PFlog shifted-CLR covariance, and plain library-size "
                        "normalized covariance with total count scaled to 10,000. PFlog uses "
                        "y_cg = log(x_cg + pc) - mean_h log(x_ch + pc), with pc = 1/(4*alpha). "
                        "Alpha is the direct NB coefficient in Var = Mu + alpha * Mu^2, fit by least-squares "
                        "to denoised log-mean-binned mean-variance trend points."
                    ),
                },
                f,
                indent=2,
                default=json_default,
            )

        print(f"[saved] {summary_json}")

        print("\n[ALL DATASET SUMMARY]")
        show_cols = [
            "dataset",
            "n_cells_covariance",
            "n_genes_covariance",
            "alpha",
            "pflog_pseudocount",
            "raw_cumvar_top10",
            "pflog_cumvar_top10",
            "libsize10k_cumvar_top10",
            "raw_cumvar_top100",
            "pflog_cumvar_top100",
            "libsize10k_cumvar_top100",
            "raw_trace",
            "pflog_trace",
            "libsize10k_trace",
        ]

        show_cols = [c for c in show_cols if c in summary_df.columns]
        print(summary_df[show_cols].to_string(index=False))

        plot_all_overlay_three_spectra(all_items)

        print("\nDone.")


def plot_two_row_three_col():
    # ============================================================
    # LOAD SAVED RAW / LIOR-PFLOG / LIBSIZE10K EIGENSPECTRA
    # AND PLOT 2 x 3 WITH SHARED SCALES ACROSS COLUMNS
    #
    # Row 1: raw eigenvalues
    # Row 2: fraction of trace
    #
    # Columns:
    #   raw          = red
    #   Lior/PFlog   = blue
    #   libsize 10k = purple
    # ============================================================




    # ============================================================
    # CONFIG
    # ============================================================

    OUT_ROOT = Path(OUTDIR) / "RAW_vs_LIOR_PFLOG_vs_LIBSIZE10K_COVARIANCE_EIGENSPECTRA"

    SAVE_DIR = OUT_ROOT
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    DPI = 300

    RAW_COLOR = "red"
    LIOR_COLOR = "blue"
    LIB10K_COLOR = "purple"

    ALPHA = 0.65
    LW = 1.25

    FIGSIZE = (16, 9)

    PNG_OUT = SAVE_DIR / "ALL_DATASETS__TWO_ROW_THREE_COL_SHARED_SCALE_colored_eigenspectra.png"
    SVG_OUT = SAVE_DIR / "ALL_DATASETS__TWO_ROW_THREE_COL_SHARED_SCALE_colored_eigenspectra.svg"


    # ============================================================
    # HELPERS
    # ============================================================

    def dataset_from_raw_eig_path(path):
        """
        Converts:
          DatasetName__raw_covariance_eigenvalues.npy
        to:
          DatasetName
        """
        name = Path(path).name
        return re.sub(r"__raw_covariance_eigenvalues\.npy$", "", name)


    def load_all_saved_eigenspectra(out_root):
        out_root = Path(out_root)

        raw_files = sorted(out_root.glob("*/*__raw_covariance_eigenvalues.npy"))

        items = []

        for raw_path in raw_files:
            dataset = dataset_from_raw_eig_path(raw_path)
            ddir = raw_path.parent

            pflog_path = ddir / f"{dataset}__PFlog_covariance_eigenvalues.npy"
            lib10k_path = ddir / f"{dataset}__libsize10k_covariance_eigenvalues.npy"

            if not pflog_path.exists():
                print(f"[skip missing PFlog] {dataset}: {pflog_path}")
                continue

            if not lib10k_path.exists():
                print(f"[skip missing lib10k] {dataset}: {lib10k_path}")
                continue

            try:
                evals_raw = np.load(raw_path).astype(float)
                evals_pflog = np.load(pflog_path).astype(float)
                evals_lib10k = np.load(lib10k_path).astype(float)

                items.append(
                    {
                        "dataset": dataset,
                        "evals_raw": evals_raw,
                        "evals_pflog": evals_pflog,
                        "evals_lib10k": evals_lib10k,
                    }
                )

            except Exception as e:
                print(f"[skip load error] {dataset}: {repr(e)}")

        if len(items) == 0:
            raise RuntimeError(
                f"No saved eigenspectra found under {out_root}. "
                "Run the covariance/eigenspectrum script first."
            )

        print(f"[loaded] {len(items)} datasets")
        for item in items:
            print(
                f"  - {item['dataset']}: "
                f"raw={len(item['evals_raw'])}, "
                f"PFlog={len(item['evals_pflog'])}, "
                f"lib10k={len(item['evals_lib10k'])}"
            )

        return items


    def positive_values(arr):
        arr = np.asarray(arr, dtype=float)
        return arr[np.isfinite(arr) & (arr > 0)]


    def trace_fraction(evals):
        evals = np.asarray(evals, dtype=float)
        total = np.nansum(evals)

        if not np.isfinite(total) or total <= 0:
            return np.full_like(evals, np.nan, dtype=float)

        return evals / total


    def get_common_log_ylim(arrays, pad_decades=0.15):
        vals = []

        for arr in arrays:
            v = positive_values(arr)
            if len(v) > 0:
                vals.append(v)

        if len(vals) == 0:
            return (1e-12, 1.0)

        vals = np.concatenate(vals)

        ymin = np.nanmin(vals)
        ymax = np.nanmax(vals)

        log_min = np.log10(ymin)
        log_max = np.log10(ymax)

        span = max(log_max - log_min, 1e-6)

        log_min -= pad_decades * span
        log_max += pad_decades * span

        return (10 ** log_min, 10 ** log_max)


    def get_common_xmax(items):
        xmax = 1

        for item in items:
            xmax = max(
                xmax,
                len(item["evals_raw"]),
                len(item["evals_pflog"]),
                len(item["evals_lib10k"]),
            )

        return xmax


    def plot_curve(ax, evals, color):
        evals = np.asarray(evals, dtype=float)
        rank = np.arange(1, len(evals) + 1)

        ok = np.isfinite(evals) & (evals > 0)

        if np.any(ok):
            ax.plot(
                rank[ok],
                evals[ok],
                color=color,
                alpha=ALPHA,
                linewidth=LW,
            )

            plt.ylim(0.01, 1000000)


    # ============================================================
    # LOAD RESULTS
    # ============================================================

    items = load_all_saved_eigenspectra(OUT_ROOT)


    # ============================================================
    # COMMON SCALES
    # ============================================================

    all_raw_eig_values = []
    all_trace_fraction_values = []

    for item in items:
        all_raw_eig_values.extend(
            [
                item["evals_raw"],
                item["evals_pflog"],
                item["evals_lib10k"],
            ]
        )

        all_trace_fraction_values.extend(
            [
                trace_fraction(item["evals_raw"]),
                trace_fraction(item["evals_pflog"]),
                trace_fraction(item["evals_lib10k"]),
            ]
        )

    eig_ylim = get_common_log_ylim(all_raw_eig_values)
    frac_ylim = get_common_log_ylim(all_trace_fraction_values)

    xmax = get_common_xmax(items)


    # ============================================================
    # PLOT
    # ============================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=FIGSIZE,
        sharex=True,
    )

    columns = [
        {
            "title": "Raw Sigma",
            "key": "evals_raw",
            "color": RAW_COLOR,
        },
        {
            "title": "Lior/PFlog Sigma",
            "key": "evals_pflog",
            "color": LIOR_COLOR,
        },
        {
            "title": "Library size 10k Sigma",
            "key": "evals_lib10k",
            "color": LIB10K_COLOR,
        },
    ]

    for j, col in enumerate(columns):
        ax = axes[0, j]

        for item in items:
            plot_curve(
                ax=ax,
                evals=item[col["key"]],
                color=col["color"],
            )

        ax.set_title(col["title"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, xmax)
        ax.set_ylim(*eig_ylim)
        ax.set_ylabel("Eigenvalue")

    for j, col in enumerate(columns):
        ax = axes[1, j]

        for item in items:
            y = trace_fraction(item[col["key"]])
            plot_curve(
                ax=ax,
                evals=y,
                color=col["color"],
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, xmax)
        ax.set_ylim(*frac_ylim)
        ax.set_xlabel("Eigenvalue rank")
        ax.set_ylabel("Fraction of trace")

    fig.tight_layout()

    fig.savefig(PNG_OUT, dpi=DPI, bbox_inches="tight")
    fig.savefig(SVG_OUT, bbox_inches="tight")

    plt.show()

    print(f"[saved] {PNG_OUT}")
    print(f"[saved] {SVG_OUT}")


def plot_three_col_vs_rank():
    # ============================================================
    # LOAD SAVED RAW / LIOR-PFLOG / LIBSIZE10K EIGENSPECTRA
    # AND PLOT JUST EIGENVALUES vs RANK
    #
    # Shared y-scale across all three columns
    # y-axis fixed from 0.01 to 10**6
    #
    # Columns:
    #   raw          = red
    #   Lior/PFlog   = blue
    #   libsize 10k  = purple
    # ============================================================




    # ============================================================
    # CONFIG
    # ============================================================

    OUT_ROOT = Path(OUTDIR) / "RAW_vs_LIOR_PFLOG_vs_LIBSIZE10K_COVARIANCE_EIGENSPECTRA"

    SAVE_DIR = OUT_ROOT
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    DPI = 300

    RAW_COLOR = "red"
    LIOR_COLOR = "blue"
    LIB10K_COLOR = "purple"

    ALPHA = 0.65
    LW = 1.25

    FIGSIZE = (16, 5)

    YMIN = 1e-1
    YMAX = 1e6

    PNG_OUT = SAVE_DIR / "ALL_DATASETS__THREE_COL_SHARED_SCALE_colored_eigenvalues_vs_rank.png"
    SVG_OUT = SAVE_DIR / "ALL_DATASETS__THREE_COL_SHARED_SCALE_colored_eigenvalues_vs_rank.svg"


    # ============================================================
    # HELPERS
    # ============================================================

    def dataset_from_raw_eig_path(path):
        """
        Converts:
          DatasetName__raw_covariance_eigenvalues.npy
        to:
          DatasetName
        """
        name = Path(path).name
        return re.sub(r"__raw_covariance_eigenvalues\.npy$", "", name)


    def load_all_saved_eigenspectra(out_root):
        out_root = Path(out_root)

        raw_files = sorted(out_root.glob("*/*__raw_covariance_eigenvalues.npy"))

        items = []

        for raw_path in raw_files:
            dataset = dataset_from_raw_eig_path(raw_path)
            ddir = raw_path.parent

            pflog_path = ddir / f"{dataset}__PFlog_covariance_eigenvalues.npy"
            lib10k_path = ddir / f"{dataset}__libsize10k_covariance_eigenvalues.npy"

            if not pflog_path.exists():
                print(f"[skip missing PFlog] {dataset}: {pflog_path}")
                continue

            if not lib10k_path.exists():
                print(f"[skip missing lib10k] {dataset}: {lib10k_path}")
                continue

            try:
                evals_raw = np.load(raw_path).astype(float)
                evals_pflog = np.load(pflog_path).astype(float)
                evals_lib10k = np.load(lib10k_path).astype(float)

                items.append(
                    {
                        "dataset": dataset,
                        "evals_raw": evals_raw,
                        "evals_pflog": evals_pflog,
                        "evals_lib10k": evals_lib10k,
                    }
                )

            except Exception as e:
                print(f"[skip load error] {dataset}: {repr(e)}")

        if len(items) == 0:
            raise RuntimeError(
                f"No saved eigenspectra found under {out_root}. "
                "Run the covariance/eigenspectrum script first."
            )

        print(f"[loaded] {len(items)} datasets")
        for item in items:
            print(
                f"  - {item['dataset']}: "
                f"raw={len(item['evals_raw'])}, "
                f"PFlog={len(item['evals_pflog'])}, "
                f"lib10k={len(item['evals_lib10k'])}"
            )

        return items


    def get_common_xmax(items):
        xmax = 1
        for item in items:
            xmax = max(
                xmax,
                len(item["evals_raw"]),
                len(item["evals_pflog"]),
                len(item["evals_lib10k"]),
            )
        return xmax


    def plot_curve(ax, evals, color):
        evals = np.asarray(evals, dtype=float)
        rank = np.arange(1, len(evals) + 1)

        ok = np.isfinite(evals) & (evals > 0)

        if np.any(ok):
            ax.plot(
                rank[ok],
                evals[ok],
                color=color,
                alpha=ALPHA,
                linewidth=LW,
            )


    # ============================================================
    # LOAD RESULTS
    # ============================================================

    items = load_all_saved_eigenspectra(OUT_ROOT)
    xmax = get_common_xmax(items)


    # ============================================================
    # PLOT
    # ============================================================

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharex=True, sharey=True)

    columns = [
        {
            "title": "Raw Sigma",
            "key": "evals_raw",
            "color": RAW_COLOR,
        },
        {
            "title": "Lior/PFlog Sigma",
            "key": "evals_pflog",
            "color": LIOR_COLOR,
        },
        {
            "title": "Library size 10k Sigma",
            "key": "evals_lib10k",
            "color": LIB10K_COLOR,
        },
    ]

    for ax, col in zip(axes, columns):
        for item in items:
            plot_curve(
                ax=ax,
                evals=item[col["key"]],
                color=col["color"],
            )

        ax.set_title(col["title"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, xmax)
        ax.set_ylim(YMIN, YMAX)
        ax.set_xlabel("Eigenvalue rank")
        ax.set_ylabel("Eigenvalue")

    fig.tight_layout()

    fig.savefig(PNG_OUT, dpi=DPI, bbox_inches="tight")
    fig.savefig(SVG_OUT, bbox_inches="tight")

    plt.show()

    print(f"[saved] {PNG_OUT}")
    print(f"[saved] {SVG_OUT}")
