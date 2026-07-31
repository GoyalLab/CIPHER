"""Per-notebook run module for ``notebooks/suppl/fig3_atlas.ipynb``.

Holds the main-flow orchestration for the Fig 3/4 CellxGene-atlas covariance-transfer
supplement, relocated VERBATIM out of the notebook's two big main-flow cells so the
notebook itself is a thin driver (markdown section headers + one config cell + one
high-level call per section). This is NOT part of the installable ``cipher`` package;
it is a notebook-only helper.

Each ``def`` below is one full main-flow cell wrapped unchanged (statements dedented
from the cell then indented one level). Config constants live inside the functions
exactly as in the original cells; the notebook injects the shared config
(DATA_DIR / SUPPL / SUPPL_OUT / OUTDIR) into this module's globals before the calls,
and the functions read those (and the star-imported stateless helpers) at run time.
"""
from __future__ import annotations

from src.suppl_atlas import *

# --- the same library imports the cluster module (src.suppl_atlas) uses;
#     re-imported here so names resolve statically and at run time ---
import os, re, glob, json, math, warnings, gc, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Patch
from matplotlib import gridspec
from scipy.sparse import issparse, csr_matrix
import scipy.stats
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
    import h5py
except Exception:
    h5py = None
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, *a, **k):
        return x if x is not None else iter(())
try:
    import cellxgene_census
except Exception:
    cellxgene_census = None


def marson_cellxgene_robust_atlas_transfer():
    global display  # preserve the cell's module-level display fallback
    PRECOMPUTE_ROOT = Path(SUPPL) / "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED"

    EXPRESSION_THRESHOLD = 1.0

    MARSON_ONLY = True

    MARSON_NAME_PATTERNS = ["marson"]

    DATASET_FOLDERS = None

    OUTDIR = Path(SUPPL_OUT) / "marson_cellxgene_robust_atlas_transfer"

    OUTDIR.mkdir(parents=True, exist_ok=True)

    PAIR_CACHE_DIR = OUTDIR / "pair_metric_cache"

    PAIR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    CELLXGENE_COV_CACHE_DIR = OUTDIR / "cellxgene_cov_source_cache"

    CELLXGENE_COV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    CELLXGENE_SAMPLE_DIR = OUTDIR / "cellxgene_sample_obs"

    CELLXGENE_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    DIAGNOSTIC_DIR = OUTDIR / "diagnostics"

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

    GENES_FILE = "genes.npy"

    PERTS_FILE = "perturbations.npy"

    STATS_H5 = "perturbation_stats.h5"

    SIGMA_FILE = "sigmas/Sigma_full_ridge.npy"

    TARGET_MAP_FILE = "perturbation_target_map.tsv"

    CENSUS_VERSION = "2025-11-08"

    ORGANISM = "Homo sapiens"

    BASE_OBS_FILTER_SUFFIX = (
        "is_primary_data == True "
        "and disease == 'normal' "
        "and suspension_type == 'cell'"
    )

    N_CELLS_PER_CELLXGENE_SOURCE = 10000

    N_ATLAS_RESAMPLES_PER_DATASET = 3

    STRATUM_COLS = ["dataset_id", "donor_id", "cell_type"]

    MIN_STRATUM_CELLS = 200

    MAX_CELLS_PER_STRATUM = 1500

    WEIGHT_STRATA_BY_N_CELLS = False

    ALLOW_POOLED_FALLBACK_IF_NO_STRATA = True

    ATLAS_COVARIANCE_VARIANT = "depth_resid"

    N_SOURCE_DATASETS_PER_ATLAS_GROUP = 8

    MIN_OVERLAP_GENES = 250

    MIN_ATLAS_TARGET_MEAN = 0.03

    MIN_ATLAS_TARGET_DETECTION = 0.01

    MIN_ATLAS_ROW_MEAN = 0.01

    MIN_ATLAS_ROW_DETECTION = 0.005

    MIN_BASIS_NORM2 = 1e-10

    MAX_PERTS_PER_RESPONSE = None

    BATCH_SIZE = 64

    TARGET_BLOCK_SIZE = 64

    EPS = 1e-12

    FORCE_REQUERY_CELLXGENE_OBS = False

    FORCE_RESAMPLE_CELLXGENE_SOURCES = False

    FORCE_REBUILD_CELLXGENE_COV = False

    # parallel workers for building CELLxGENE covariance sources (I/O-bound S3 fetch);
    # each source is independent + self-caching, so a thread pool overlaps the fetches.
    CELLXGENE_COV_WORKERS = int(os.environ.get("CELLXGENE_COV_WORKERS", "8"))

    FORCE_RECOMPUTE_PAIR_METRICS = False

    SAVE_PER_PERTURBATION_TABLE = True

    RUN_ATLAS_TARGET_SHUFFLED_NEGATIVE = True

    DPI = 300

    VIOLIN_MAX_POINTS_PER_GROUP = 2500

    JITTER_MAX_POINTS_PER_GROUP = 1200

    RANDOM_SEED = 0

    rng = np.random.default_rng(RANDOM_SEED)

    TQDM_NCOLS = 110

    GENERIC_TCELL_TYPES = [
        "T cell",
    ]

    CD4_TCELL_TYPES = [
        "CD4-positive, alpha-beta T cell",
        "CD4-positive helper T cell",
        "naive thymus-derived CD4-positive, alpha-beta T cell",
        "central memory CD4-positive, alpha-beta T cell",
        "effector memory CD4-positive, alpha-beta T cell",
    ]

    CD8_TCELL_TYPES = [
        "CD8-positive, alpha-beta T cell",
        "CD8-positive, alpha-beta cytotoxic T cell",
        "naive thymus-derived CD8-positive, alpha-beta T cell",
        "central memory CD8-positive, alpha-beta T cell",
        "effector memory CD8-positive, alpha-beta T cell",
    ]

    TREG_TYPES = [
        "regulatory T cell",
        "CD4-positive, CD25-positive, alpha-beta regulatory T cell",
    ]

    PBMC_NON_T_TYPES = [
        "B cell",
        "naive B cell",
        "memory B cell",
        "monocyte",
        "classical monocyte",
        "non-classical monocyte",
        "macrophage",
        "natural killer cell",
        "dendritic cell",
    ]

    TISSUE_NON_T_TYPES = [
        "fibroblast",
        "endothelial cell",
        "epithelial cell",
        "macrophage",
        "monocyte",
        "smooth muscle cell",
        "stromal cell",
        "mesenchymal cell",
    ]

    PBMC_KEYWORDS = [
        "pbmc",
        "pbmcs",
        "peripheral blood",
        "blood",
        "venous blood",
        "leukapheresis",
        "buffy coat",
        "mononuclear",
    ]

    BLOODLIKE_TISSUE_KEYWORDS = [
        "blood",
        "peripheral blood",
        "venous blood",
        "umbilical cord blood",
    ]

    PROBABLE_TCELL_PATTERNS = [
        r"\bt cell\b",
        r"\bt-cell\b",
        r"alpha-beta",
        r"gamma-delta",
        r"\bcd4\b",
        r"\bcd8\b",
        r"regulatory t",
        r"natural killer t",
        r"\bnkt\b",
        r"\bth1\b",
        r"\bth2\b",
        r"\bth17\b",
    ]

    ATLAS_GROUPS = [
        {
            "group": "PBMC CD4 T cells",
            "cell_types": CD4_TCELL_TYPES,
            "pbmc_mode": "pbmc",
            "n_source_datasets": N_SOURCE_DATASETS_PER_ATLAS_GROUP,
        },
        {
            "group": "PBMC CD8 T cells",
            "cell_types": CD8_TCELL_TYPES,
            "pbmc_mode": "pbmc",
            "n_source_datasets": N_SOURCE_DATASETS_PER_ATLAS_GROUP,
        },
        {
            "group": "PBMC Treg cells",
            "cell_types": TREG_TYPES,
            "pbmc_mode": "pbmc",
            "n_source_datasets": N_SOURCE_DATASETS_PER_ATLAS_GROUP,
        },
        {
            "group": "PBMC generic T cells",
            "cell_types": GENERIC_TCELL_TYPES,
            "pbmc_mode": "pbmc",
            "n_source_datasets": N_SOURCE_DATASETS_PER_ATLAS_GROUP,
        },
        {
            "group": "Tissue CD4 T cells",
            "cell_types": CD4_TCELL_TYPES,
            "pbmc_mode": "tissue",
            "n_source_datasets": N_SOURCE_DATASETS_PER_ATLAS_GROUP,
        },
        {
            "group": "Tissue CD8 T cells",
            "cell_types": CD8_TCELL_TYPES,
            "pbmc_mode": "tissue",
            "n_source_datasets": N_SOURCE_DATASETS_PER_ATLAS_GROUP,
        },
        {
            "group": "Tissue generic T cells",
            "cell_types": GENERIC_TCELL_TYPES,
            "pbmc_mode": "tissue",
            "n_source_datasets": N_SOURCE_DATASETS_PER_ATLAS_GROUP,
        },
        {
            "group": "PBMC non-T cells",
            "cell_types": PBMC_NON_T_TYPES,
            "pbmc_mode": "pbmc",
            "n_source_datasets": N_SOURCE_DATASETS_PER_ATLAS_GROUP,
            "remove_probable_t_cells": True,
        },
        {
            "group": "Tissue non-T cells",
            "cell_types": TISSUE_NON_T_TYPES,
            "pbmc_mode": "tissue",
            "n_source_datasets": N_SOURCE_DATASETS_PER_ATLAS_GROUP,
            "remove_probable_t_cells": True,
        },
    ]

    MARSON_GROUP_ORDER = [
        "Marson within dataset",
        "Marson within condition",
        "Marson cross condition",
    ]

    ATLAS_GROUP_ORDER = [x["group"] for x in ATLAS_GROUPS]

    NEGATIVE_GROUP_ORDER = [
        "Atlas target-shuffled negative",
    ]

    COMPOSITE_GROUP_ORDER = (
        MARSON_GROUP_ORDER
        + ATLAS_GROUP_ORDER
        + (NEGATIVE_GROUP_ORDER if RUN_ATLAS_TARGET_SHUFFLED_NEGATIVE else [])
    )

    COMPOSITE_GROUP_LABELS = {
        "Marson within dataset": "Marson\nwithin dataset",
        "Marson within condition": "Marson\nwithin condition",
        "Marson cross condition": "Marson\ncross condition",
        "PBMC CD4 T cells": "PBMC\nCD4 T",
        "PBMC CD8 T cells": "PBMC\nCD8 T",
        "PBMC Treg cells": "PBMC\nTreg",
        "PBMC generic T cells": "PBMC\ngeneric T",
        "Tissue CD4 T cells": "Tissue\nCD4 T",
        "Tissue CD8 T cells": "Tissue\nCD8 T",
        "Tissue generic T cells": "Tissue\ngeneric T",
        "PBMC non-T cells": "PBMC\nnon-T",
        "Tissue non-T cells": "Tissue\nnon-T",
        "Atlas target-shuffled negative": "Atlas\ntarget-shuffled",
    }

    COMPOSITE_GROUP_COLORS = {
        "Marson within dataset": "#8E77B5",
        "Marson within condition": "#6C8EBF",
        "Marson cross condition": "#4E79A7",
        "PBMC CD4 T cells": "#2CA25F",
        "PBMC CD8 T cells": "#E15759",
        "PBMC Treg cells": "#9467BD",
        "PBMC generic T cells": "#59A14F",
        "Tissue CD4 T cells": "#86BC86",
        "Tissue CD8 T cells": "#FF9DA7",
        "Tissue generic T cells": "#E3B778",
        "PBMC non-T cells": "#9C755F",
        "Tissue non-T cells": "#999999",
        "Atlas target-shuffled negative": "#BBBBBB",
    }

    try:
        _real_display = display
    except NameError:
        _real_display = print

    def display(x):
        # Sanitize absolute paths out of DataFrame string columns so allocation paths are
        # never baked into the rendered cell output (rich display bypasses stdout routing).
        try:
            from src.logutil import sanitize as _sanitize_paths
            if isinstance(x, pd.DataFrame):
                x = x.copy()
                for _c in x.columns:
                    if x[_c].dtype == object:
                        x[_c] = x[_c].map(
                            lambda v: _sanitize_paths(v) if isinstance(v, str) else v
                        )
        except Exception:
            pass
        _real_display(x)

    def pair_cache_paths(response_dataset, source_dataset, mode="main"):
        stem = (
            f"response__{sanitize_filename(response_dataset)}__"
            f"source__{sanitize_filename(source_dataset)}__"
            f"mode__{sanitize_filename(mode)}"
        )
        return {
            "summary": PAIR_CACHE_DIR / f"{stem}__summary.json",
            "per_pert": PAIR_CACHE_DIR / f"{stem}__per_pert.csv",
        }

    def load_target_indices(folder, perts, genes):
        folder = Path(folder)
        gene_set = set(genes.tolist())
        gene_to_idx = {g: i for i, g in enumerate(genes.tolist())}

        target_map_path = folder / TARGET_MAP_FILE
        pert_to_target = {}

        if target_map_path.exists():
            tm = pd.read_csv(target_map_path, sep="\t")
            if "perturbation" in tm.columns and "target_gene" in tm.columns:
                for _, row in tm.iterrows():
                    p = str(row["perturbation"])
                    g = str(row["target_gene"])
                    if g not in {"", "nan", "None", "<NA>"}:
                        pert_to_target[p] = g

        target_genes = []
        target_idx = []
        matched = []

        for p in perts:
            p = str(p)

            if p in pert_to_target and pert_to_target[p] in gene_to_idx:
                g = pert_to_target[p]
            else:
                g = pert_to_gene_safe(p, gene_set)

            if g is None or g not in gene_to_idx:
                target_genes.append("")
                target_idx.append(-1)
                matched.append(False)
            else:
                target_genes.append(g)
                target_idx.append(gene_to_idx[g])
                matched.append(True)

        return (
            np.asarray(target_genes, dtype=object),
            np.asarray(target_idx, dtype=np.int64),
            np.asarray(matched, dtype=bool),
        )

    def find_precomputed_folders(root=PRECOMPUTE_ROOT, expression_threshold=EXPRESSION_THRESHOLD):
        root = Path(root)
        tag = threshold_to_tag(expression_threshold)
        pattern = f"*__mean_ge_{tag}"

        folders = sorted([p for p in root.glob(pattern) if p.is_dir()])

        if MARSON_ONLY:
            folders = [
                p for p in folders
                if any(pat.lower() in p.name.lower() for pat in MARSON_NAME_PATTERNS)
            ]

        print(f"[folders] root={root}")
        print(f"[folders] pattern={pattern}")
        print(f"[folders] MARSON_ONLY={MARSON_ONLY}")
        print(f"[folders] found={len(folders)}")

        return folders

    def load_precomputed_marson_dataset(folder):
        folder = Path(folder)

        genes_path = folder / GENES_FILE
        perts_path = folder / PERTS_FILE
        stats_path = folder / STATS_H5
        sigma_path = folder / SIGMA_FILE

        required = [genes_path, perts_path, stats_path, sigma_path]
        missing = [str(x) for x in required if not x.exists()]
        if missing:
            raise FileNotFoundError("Missing required precomputed files:\n" + "\n".join(missing))

        genes = decode_str_array(np.load(genes_path, allow_pickle=True))
        perts = decode_str_array(np.load(perts_path, allow_pickle=True))

        target_genes, target_idx, matched = load_target_indices(folder, perts, genes)
        matched_pos = np.flatnonzero(matched).astype(np.int64)

        if MAX_PERTS_PER_RESPONSE is not None and len(matched_pos) > MAX_PERTS_PER_RESPONSE:
            matched_pos = np.sort(
                rng.choice(matched_pos, size=MAX_PERTS_PER_RESPONSE, replace=False)
            ).astype(np.int64)

        label = clean_dataset_label(folder.name)
        gene_to_idx = {g: i for i, g in enumerate(genes.tolist())}

        with h5py.File(stats_path, "r") as h5:
            if "dx" not in h5:
                raise KeyError(f"{stats_path} missing key 'dx'")

            dx_shape = h5["dx"].shape

            if dx_shape[1] != len(genes):
                raise ValueError(f"{stats_path}: dx shape={dx_shape}, genes={len(genes)}")

            if "n_cells_pert" in h5:
                n_cells_pert = np.asarray(h5["n_cells_pert"][:], dtype=np.int64)
            else:
                n_cells_pert = np.full(len(perts), -1, dtype=np.int64)

        Sigma = np.load(sigma_path, mmap_mode="r")

        if Sigma.shape != (len(genes), len(genes)):
            raise ValueError(f"{sigma_path}: Sigma shape={Sigma.shape}, expected={(len(genes), len(genes))}")

        meta = {
            "dataset": folder.name,
            "label": label,
            "folder": folder,
            "genes": genes,
            "gene_to_idx": gene_to_idx,
            "perts": perts,
            "target_genes": target_genes,
            "target_idx": target_idx,
            "matched": matched,
            "matched_pos": matched_pos,
            "matched_perts": perts[matched_pos].astype(str),
            "matched_target_genes": target_genes[matched_pos].astype(str),
            "matched_target_idx": target_idx[matched_pos].astype(np.int64),
            "n_cells_pert": n_cells_pert,
            "stats_path": stats_path,
            "sigma_path": sigma_path,
            "Sigma": Sigma,
            "condition": extract_marson_condition(folder.name),
            "n_genes": int(len(genes)),
            "n_perts_total": int(len(perts)),
            "n_perts_matched": int(len(matched_pos)),
        }

        print(
            f"[loaded precomputed] {label}: "
            f"condition={meta['condition']}, genes={len(genes):,}, "
            f"perts={len(perts):,}, matched_eval={len(matched_pos):,}"
        )

        return meta

    def load_all_precomputed_marson():
        if DATASET_FOLDERS is None:
            folders = find_precomputed_folders()
        else:
            folders = [Path(x) for x in DATASET_FOLDERS]

        if len(folders) == 0:
            raise FileNotFoundError(
                f"No folders found under {PRECOMPUTE_ROOT} for EXPRESSION_THRESHOLD={EXPRESSION_THRESHOLD}"
            )

        datasets = []
        errors = []

        for folder in tqdm(folders, desc="loading precomputed Marson datasets", ncols=TQDM_NCOLS):
            try:
                datasets.append(load_precomputed_marson_dataset(folder))
            except Exception as e:
                print("\n" + "!" * 110)
                print(f"[ERROR loading] {folder}")
                print(repr(e))
                print("!" * 110 + "\n")
                errors.append({"folder": str(folder), "error": repr(e)})

        if len(datasets) == 0:
            raise RuntimeError("No usable precomputed Marson datasets loaded.")

        with open(OUTDIR / "load_errors.json", "w") as f:
            json.dump(errors, f, indent=2, default=json_default)

        metadata_df = pd.DataFrame([
            {
                "dataset": d["dataset"],
                "label": d["label"],
                "condition": d["condition"],
                "folder": str(d["folder"]),
                "n_genes": d["n_genes"],
                "n_perts_total": d["n_perts_total"],
                "n_perts_matched": d["n_perts_matched"],
                "stats_path": str(d["stats_path"]),
                "sigma_path": str(d["sigma_path"]),
            }
            for d in datasets
        ])

        metadata_path = OUTDIR / "loaded_precomputed_marson_datasets.csv"
        metadata_df.to_csv(metadata_path, index=False)

        print(f"\n[saved] {metadata_path}")
        # display a path-sanitized copy so absolute folder/stats/sigma paths are not
        # baked into the cell output (rich display bypasses the stdout log routing)
        _disp = metadata_df.copy()
        for _col in ("folder", "stats_path", "sigma_path"):
            if _col in _disp.columns:
                _disp[_col] = _disp[_col].map(lambda x: os.path.basename(str(x)))
        display(_disp)

        return datasets

    def validate_raw_counts_matrix(X, name="X", integer_check=True):
        if issparse(X):
            data = X.data

            if data.size:
                if not np.isfinite(data).all():
                    bad = int((~np.isfinite(data)).sum())
                    raise ValueError(f"{name}: {bad:,} non-finite sparse values.")

                xmin = float(data.min())
                xmax = float(data.max())

                if xmin < 0:
                    neg = int((data < 0).sum())
                    raise ValueError(f"{name}: found {neg:,} negative raw-count values, min={xmin}.")

                if integer_check:
                    vals = data
                    if vals.size > 200_000:
                        vals = rng.choice(vals, size=200_000, replace=False)

                    frac_integerish = float(np.mean(np.abs(vals - np.round(vals)) < 1e-6))

                    print(
                        f"{name}: sparse raw check shape={X.shape}, "
                        f"nonzero_min={xmin:.4g}, nonzero_max={xmax:.4g}, "
                        f"integerish={frac_integerish:.4f}"
                    )

                    if frac_integerish < 0.98:
                        raise ValueError(
                            f"{name}: not integer-like enough for raw counts "
                            f"(integerish={frac_integerish:.4f})."
                        )

            return X

        X = np.asarray(X, dtype=np.float64)

        if not np.isfinite(X).all():
            bad = int((~np.isfinite(X)).sum())
            raise ValueError(f"{name}: {bad:,} non-finite values.")

        xmin = float(X.min())
        xmax = float(X.max())

        if xmin < 0:
            neg = int((X < 0).sum())
            raise ValueError(f"{name}: found {neg:,} negative raw-count values, min={xmin}.")

        if integer_check:
            vals = X[X > 0]
            if vals.size > 200_000:
                vals = rng.choice(vals, size=200_000, replace=False)

            frac_integerish = (
                float(np.mean(np.abs(vals - np.round(vals)) < 1e-6))
                if vals.size
                else 1.0
            )

            print(
                f"{name}: dense raw check shape={X.shape}, "
                f"min={xmin:.4g}, max={xmax:.4g}, integerish={frac_integerish:.4f}"
            )

            if frac_integerish < 0.98:
                raise ValueError(
                    f"{name}: not integer-like enough for raw counts "
                    f"(integerish={frac_integerish:.4f})."
                )

        return X

    def compute_cov_columns_from_raw_counts(
        X_raw,
        target_idx,
        residualize_depth=False,
        name="raw X",
    ):
        """
        Computes covariance columns:
          Cov(X, Xt) = (X.T @ Xt - n * mu * mu_t.T) / (n - 1)

        If residualize_depth=True:
          Cov_resid(X_j, X_t) =
            Cov(X_j, X_t) - Cov(X_j, depth) Cov(X_t, depth) / Var(depth)

        Returns:
          cov_cols, gene_mean, gene_detection, gene_var, depth stats
        """
        target_idx = np.asarray(target_idx, dtype=np.int64)

        if target_idx.size == 0:
            raise ValueError(f"{name}: empty target_idx.")

        validate_raw_counts_matrix(X_raw, name=name, integer_check=True)

        stats = compute_gene_stats_from_raw_counts(X_raw)
        mu = stats["mean"]
        gene_det = stats["detection"]
        gene_var = stats["var"]
        depth = stats["depth"]

        if issparse(X_raw):
            X = X_raw.astype(np.float64).tocsr(copy=False)
        else:
            X = np.asarray(X_raw, dtype=np.float64)

        n, p = X.shape
        if n < 2:
            raise ValueError(f"{name}: need at least 2 cells.")

        print(
            f"{name}: covariance columns "
            f"n={n:,}, genes={p:,}, targets={len(target_idx):,}, "
            f"target_block={TARGET_BLOCK_SIZE}, residualize_depth={residualize_depth}"
        )

        depth_mean = float(np.mean(depth))
        depth_var = float(np.var(depth, ddof=1)) if n > 1 else 0.0

        if residualize_depth and depth_var <= EPS:
            print(f"[warn] {name}: depth variance too small; skipping depth residualization.")
            residualize_depth = False

        if residualize_depth:
            if issparse(X):
                x_depth = np.asarray(X.T @ depth).ravel().astype(np.float64)
            else:
                x_depth = np.asarray(X.T @ depth).ravel().astype(np.float64)

            cov_x_depth = (x_depth - float(n) * mu * depth_mean) / float(n - 1)
        else:
            cov_x_depth = None

        cov_cols = np.empty((p, len(target_idx)), dtype=np.float32)

        if issparse(X):
            XT = X.T.tocsr(copy=False)

            for s in tqdm(
                range(0, len(target_idx), TARGET_BLOCK_SIZE),
                desc=f"{name}: cov target blocks",
                ncols=TQDM_NCOLS,
            ):
                e = min(s + TARGET_BLOCK_SIZE, len(target_idx))
                tidx = target_idx[s:e]

                X_target = X[:, tidx].tocsc(copy=False)
                cross = XT @ X_target
                cross = cross.toarray() if issparse(cross) else np.asarray(cross)
                cross = cross.astype(np.float64, copy=False)

                mu_t = mu[tidx]
                block = (cross - float(n) * mu[:, None] * mu_t[None, :]) / float(n - 1)

                if residualize_depth:
                    cov_t_depth = cov_x_depth[tidx]
                    block = block - cov_x_depth[:, None] * cov_t_depth[None, :] / depth_var

                if not np.isfinite(block).all():
                    bad = int((~np.isfinite(block)).sum())
                    raise FloatingPointError(f"{name}: covariance block {s}:{e} has {bad:,} non-finite values.")

                cov_cols[:, s:e] = block.astype(np.float32)

                del X_target, cross, block
                gc.collect()

        else:
            XT = np.ascontiguousarray(X.T)

            for s in tqdm(
                range(0, len(target_idx), TARGET_BLOCK_SIZE),
                desc=f"{name}: cov target blocks",
                ncols=TQDM_NCOLS,
            ):
                e = min(s + TARGET_BLOCK_SIZE, len(target_idx))
                tidx = target_idx[s:e]

                X_target = np.ascontiguousarray(X[:, tidx])
                cross = XT @ X_target

                mu_t = mu[tidx]
                block = (cross - float(n) * mu[:, None] * mu_t[None, :]) / float(n - 1)

                if residualize_depth:
                    cov_t_depth = cov_x_depth[tidx]
                    block = block - cov_x_depth[:, None] * cov_t_depth[None, :] / depth_var

                if not np.isfinite(block).all():
                    bad = int((~np.isfinite(block)).sum())
                    raise FloatingPointError(f"{name}: covariance block {s}:{e} has {bad:,} non-finite values.")

                cov_cols[:, s:e] = block.astype(np.float32)

                del X_target, cross, block
                gc.collect()

        return {
            "cov_cols": cov_cols,
            "gene_mean": mu.astype(np.float32),
            "gene_detection": gene_det.astype(np.float32),
            "gene_var": gene_var.astype(np.float32),
            "depth_mean": depth_mean,
            "depth_median": float(np.median(depth)),
            "depth_var": depth_var,
            "n_cells": int(n),
        }

    def lr_metrics_batch_all(dx, basis, eps=EPS):
        dx = np.asarray(dx, dtype=np.float64)
        basis = np.asarray(basis, dtype=np.float64)

        dx_norm2 = np.einsum("ij,ij->i", dx, dx, optimize=True)
        basis_norm2 = np.einsum("ij,ij->i", basis, basis, optimize=True)
        numer = np.einsum("ij,ij->i", dx, basis, optimize=True)

        alpha = np.zeros(dx.shape[0], dtype=np.float64)
        good_basis = basis_norm2 > eps
        alpha[good_basis] = numer[good_basis] / basis_norm2[good_basis]

        pred = alpha[:, None] * basis

        resid = dx - pred
        resid_norm2 = np.einsum("ij,ij->i", resid, resid, optimize=True)

        r2 = np.full(dx.shape[0], np.nan, dtype=np.float64)
        good_dx = dx_norm2 > eps
        r2[good_dx] = 1.0 - resid_norm2[good_dx] / dx_norm2[good_dx]

        dx_c = dx - dx.mean(axis=1, keepdims=True)
        pred_c = pred - pred.mean(axis=1, keepdims=True)

        pear_num = np.einsum("ij,ij->i", dx_c, pred_c, optimize=True)
        pear_den = (
            np.sqrt(
                np.einsum("ij,ij->i", dx_c, dx_c, optimize=True)
                * np.einsum("ij,ij->i", pred_c, pred_c, optimize=True)
            )
            + eps
        )

        pearson = pear_num / pear_den
        pearson[~np.isfinite(pearson)] = np.nan

        return pearson, r2, alpha, dx_norm2, basis_norm2

    def lr_metrics_leave_one_coordinate(dx, basis, exclude_pos, eps=EPS):
        """
        Per-row metric excluding one coordinate, usually the target gene.
        exclude_pos[i] is the column index to remove for row i, or -1 for no removal.
        """
        dx = np.asarray(dx, dtype=np.float64)
        basis = np.asarray(basis, dtype=np.float64)
        exclude_pos = np.asarray(exclude_pos, dtype=np.int64)

        n, p = dx.shape

        pearson = np.full(n, np.nan, dtype=np.float64)
        r2 = np.full(n, np.nan, dtype=np.float64)
        alpha = np.full(n, np.nan, dtype=np.float64)
        dx_norm2 = np.full(n, np.nan, dtype=np.float64)
        basis_norm2 = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            mask = np.ones(p, dtype=bool)

            if 0 <= exclude_pos[i] < p:
                mask[exclude_pos[i]] = False

            xi = dx[i, mask]
            bi = basis[i, mask]

            if xi.size < 3:
                continue

            dxn = float(np.dot(xi, xi))
            bn = float(np.dot(bi, bi))
            num = float(np.dot(xi, bi))

            dx_norm2[i] = dxn
            basis_norm2[i] = bn

            if dxn <= eps or bn <= eps:
                continue

            ai = num / bn
            pi = ai * bi
            resid = xi - pi

            alpha[i] = ai
            r2[i] = 1.0 - float(np.dot(resid, resid)) / dxn

            xic = xi - xi.mean()
            pic = pi - pi.mean()
            den = np.sqrt(float(np.dot(xic, xic)) * float(np.dot(pic, pic))) + eps

            pearson[i] = float(np.dot(xic, pic)) / den

        return pearson, r2, alpha, dx_norm2, basis_norm2

    def get_census_dataset_table():
        cache_path = OUTDIR / f"census_datasets__{CENSUS_VERSION}.parquet"

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
            datasets = census["census_info"]["datasets"].read().concat().to_pandas()

        datasets["dataset_id"] = safe_str_col(datasets, "dataset_id")
        datasets.to_parquet(cache_path, index=False)

        return datasets

    def add_dataset_titles_to_obs(obs):
        datasets = get_census_dataset_table().copy()

        title_col = None
        for c in ["dataset_title", "title", "collection_name", "dataset_name"]:
            if c in datasets.columns:
                title_col = c
                break

        obs = obs.copy()
        obs["dataset_id"] = safe_str_col(obs, "dataset_id")

        if title_col is None or "dataset_id" not in datasets.columns:
            obs["dataset_title"] = obs["dataset_id"].astype(str)
            return obs

        ds = datasets[["dataset_id", title_col]].copy()
        ds["dataset_id"] = safe_str_col(ds, "dataset_id")
        ds[title_col] = safe_str_col(ds, title_col)
        ds = ds.drop_duplicates("dataset_id")
        ds = ds.rename(columns={title_col: "dataset_title"})

        obs = obs.merge(ds, on="dataset_id", how="left")

        title = safe_str_col(obs, "dataset_title")
        fallback = safe_str_col(obs, "dataset_id")
        obs["dataset_title"] = np.where(title.values == "", fallback.values, title.values)

        return obs

    def classify_obs_pbmc_like(obs):
        tissue = safe_str_col(obs, "tissue").str.lower()
        tissue_general = safe_str_col(obs, "tissue_general").str.lower()
        dataset_title = safe_str_col(obs, "dataset_title").str.lower()

        text = tissue + " " + tissue_general + " " + dataset_title

        is_bloodlike_tissue = tissue.apply(lambda s: contains_any_keyword(s, BLOODLIKE_TISSUE_KEYWORDS))
        is_bloodlike_general = tissue_general.apply(lambda s: contains_any_keyword(s, BLOODLIKE_TISSUE_KEYWORDS))
        is_pbmc_title = dataset_title.apply(lambda s: contains_any_keyword(s, PBMC_KEYWORDS))
        is_pbmc_text = text.apply(lambda s: contains_any_keyword(s, PBMC_KEYWORDS))

        return (is_bloodlike_tissue | is_bloodlike_general | is_pbmc_title | is_pbmc_text).values

    def is_probably_t_cell_name(cell_type):
        s = str(cell_type).lower()
        return any(re.search(pat, s) is not None for pat in PROBABLE_TCELL_PATTERNS)

    def query_cellxgene_obs_for_cell_types(cell_type_names, cache_prefix):
        cache_path = OUTDIR / (
            f"cellxgene_obs__{sanitize_filename(cache_prefix)}__"
            f"{CENSUS_VERSION}.parquet"
        )

        if cache_path.exists() and not FORCE_REQUERY_CELLXGENE_OBS:
            print(f"[CELLxGENE obs] loading cached {cache_path}")
            obs = pd.read_parquet(cache_path)
            for c in ["dataset_id", "dataset_title", "cell_type", "tissue", "tissue_general", "assay", "donor_id"]:
                obs[c] = safe_str_col(obs, c)
            if "obs_pbmc_like" not in obs.columns:
                obs["obs_pbmc_like"] = classify_obs_pbmc_like(obs)
            return obs

        all_obs = []

        for ct in cell_type_names:
            vf = f"cell_type == '{ct}' and {BASE_OBS_FILTER_SUFFIX}"
            print(f"[CELLxGENE obs] querying: {vf}")

            try:
                with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
                    obs = cellxgene_census.get_obs(
                        census,
                        ORGANISM,
                        value_filter=vf,
                        column_names=[
                            "soma_joinid",
                            "dataset_id",
                            "cell_type",
                            "tissue",
                            "tissue_general",
                            "disease",
                            "assay",
                            "donor_id",
                            "suspension_type",
                            "is_primary_data",
                        ],
                    )

                if len(obs):
                    obs["query_cell_type"] = ct
                    all_obs.append(obs)
                    print(f"  -> {len(obs):,} cells")
                else:
                    print("  -> 0 cells")

            except Exception as e:
                print(f"  -> query failed for {ct}: {repr(e)}")

        if len(all_obs) == 0:
            print(f"[WARN] No CELLxGENE obs found for {cache_prefix}")
            return pd.DataFrame()

        obs = pd.concat(all_obs, ignore_index=True)
        obs = obs.drop_duplicates("soma_joinid").reset_index(drop=True)
        obs = add_dataset_titles_to_obs(obs)

        for c in ["dataset_id", "dataset_title", "cell_type", "tissue", "tissue_general", "assay", "donor_id"]:
            obs[c] = safe_str_col(obs, c)

        obs["obs_pbmc_like"] = classify_obs_pbmc_like(obs)

        obs.to_parquet(cache_path, index=False)

        print(f"[CELLxGENE obs] saved {cache_path}")
        print(f"[CELLxGENE obs] cells={len(obs):,}, datasets={obs['dataset_id'].nunique():,}")
        print("[CELLxGENE obs] top cell types:")
        print(obs["cell_type"].value_counts().head(20))

        return obs

    def summarize_obs_by_dataset(obs):
        obs = obs.copy()

        for c in ["dataset_id", "dataset_title", "cell_type", "tissue", "tissue_general", "assay", "donor_id"]:
            obs[c] = safe_str_col(obs, c)

        rows = []

        for dsid, sub in obs.groupby("dataset_id", sort=False):
            donor_vals = sub["donor_id"].astype(str)
            donor_vals = donor_vals[(donor_vals != "") & (donor_vals.str.lower() != "nan")]

            rows.append({
                "dataset_id": str(dsid),
                "dataset_title": mode_or_empty(sub["dataset_title"]),
                "n_cells": int(len(sub)),
                "cell_type_top": mode_or_empty(sub["cell_type"]),
                "cell_types_seen": unique_join_limited(sub["cell_type"], max_items=10),
                "tissue_top": mode_or_empty(sub["tissue"]),
                "tissue_general_top": mode_or_empty(sub["tissue_general"]),
                "tissues_seen": unique_join_limited(sub["tissue"], max_items=10),
                "tissue_general_seen": unique_join_limited(sub["tissue_general"], max_items=10),
                "assays_seen": unique_join_limited(sub["assay"], max_items=5),
                "n_donors": int(donor_vals.nunique(dropna=True)),
            })

        return pd.DataFrame(rows)

    def make_short_cellxgene_label(rank, composite_group, row, rep):
        title = str(row.get("dataset_title", ""))
        title = sanitize_filename(title, maxlen=60).replace("_", " ")
        if len(title) > 32:
            title = title[:29] + "..."

        cell_type = str(row.get("cell_type_top", "")).replace("_", " ")
        tissue = str(row.get("tissue_general_top", row.get("tissue_top", ""))).replace("_", " ")

        if tissue and tissue.lower() not in {"nan", "none", "<na>"}:
            extra = tissue
        elif cell_type and cell_type.lower() not in {"nan", "none", "<na>"}:
            extra = cell_type
        else:
            extra = title

        group_short = composite_group.replace(" cells", "").replace("T cells", "T")
        group_short = group_short.replace("PBMC ", "PBMC ").replace("Tissue ", "Tissue ")

        return f"{group_short} {rank}.{rep}: {extra}"

    def sample_source_from_dataset(obs, dataset_id, n, seed):
        sub = obs[obs["dataset_id"].astype(str) == str(dataset_id)].copy()

        if len(sub) < n:
            raise ValueError(f"dataset {dataset_id} has only {len(sub)} cells, need {n}")

        return sub.sample(n=n, replace=False, random_state=seed).copy()

    def choose_cellxgene_sources():
        specs = []
        rows = []

        chosen_path = OUTDIR / (
            f"chosen_cellxgene_sources__groups{len(ATLAS_GROUPS)}__"
            f"nsource{N_SOURCE_DATASETS_PER_ATLAS_GROUP}__"
            f"resamples{N_ATLAS_RESAMPLES_PER_DATASET}__"
            f"cells{N_CELLS_PER_CELLXGENE_SOURCE}__"
            f"seed{RANDOM_SEED}.csv"
        )

        if chosen_path.exists() and not FORCE_RESAMPLE_CELLXGENE_SOURCES:
            chosen_df = pd.read_csv(chosen_path)
            print(f"[CELLxGENE sources] loading chosen sources: {chosen_path}")

            loaded_specs = []
            for _, row in chosen_df.iterrows():
                sample_obs = pd.read_csv(row["sample_obs_path"])
                loaded_specs.append({
                    "kind": "cellxgene",
                    "source_dataset": str(row["source_dataset"]),
                    "source_label": str(row["source_label"]),
                    "source_type": str(row["source_type"]),
                    "composite_group": str(row["composite_group"]),
                    "dataset_id": str(row["dataset_id"]),
                    "resample_id": int(row["resample_id"]),
                    "sample_obs": sample_obs,
                })

            display(chosen_df)
            return loaded_specs

        for group_cfg in ATLAS_GROUPS:
            group = group_cfg["group"]
            cell_types = group_cfg["cell_types"]
            pbmc_mode = group_cfg["pbmc_mode"]
            n_source_datasets = int(group_cfg.get("n_source_datasets", N_SOURCE_DATASETS_PER_ATLAS_GROUP))

            print("\n" + "=" * 120)
            print(f"[ATLAS GROUP] {group}")
            print("=" * 120)

            obs = query_cellxgene_obs_for_cell_types(
                cell_type_names=cell_types,
                cache_prefix=f"{group}__raw_obs",
            )

            if len(obs) == 0:
                print(f"[WARN] skipping {group}: no obs")
                continue

            obs["obs_pbmc_like"] = classify_obs_pbmc_like(obs)

            if pbmc_mode == "pbmc":
                obs = obs[obs["obs_pbmc_like"].astype(bool)].copy()
            elif pbmc_mode == "tissue":
                obs = obs[~obs["obs_pbmc_like"].astype(bool)].copy()
            else:
                raise ValueError(f"Unknown pbmc_mode={pbmc_mode}")

            if group_cfg.get("remove_probable_t_cells", False):
                t_like = obs["cell_type"].map(is_probably_t_cell_name).astype(bool)
                print(f"[{group}] removing probable T-cell rows: {int(t_like.sum()):,}")
                obs = obs[~t_like].copy()

            if len(obs) == 0:
                print(f"[WARN] skipping {group}: no cells after pbmc/tissue filtering")
                continue

            summary = summarize_obs_by_dataset(obs)
            summary_path = DIAGNOSTIC_DIR / f"cellxgene_group_dataset_summary__{sanitize_filename(group)}.csv"
            summary.to_csv(summary_path, index=False)
            print(f"[saved] {summary_path}")

            eligible = summary[summary["n_cells"] >= N_CELLS_PER_CELLXGENE_SOURCE].copy()

            if len(eligible) == 0:
                print(
                    f"[WARN] skipping {group}: no datasets with >= "
                    f"{N_CELLS_PER_CELLXGENE_SOURCE:,} cells"
                )
                continue

            n_pick = min(n_source_datasets, len(eligible))
            chosen_ids = rng.choice(eligible["dataset_id"].values, size=n_pick, replace=False).tolist()

            print(f"[{group}] eligible datasets={len(eligible):,}; chosen={n_pick:,}")

            for rank, dsid in enumerate(chosen_ids, start=1):
                r = eligible[eligible["dataset_id"] == dsid].iloc[0].to_dict()

                for rep in range(N_ATLAS_RESAMPLES_PER_DATASET):
                    seed = stable_seed(RANDOM_SEED, group, dsid, rep)
                    sample_obs = sample_source_from_dataset(
                        obs=obs,
                        dataset_id=dsid,
                        n=N_CELLS_PER_CELLXGENE_SOURCE,
                        seed=seed,
                    )

                    source_dataset = (
                        f"CELLxGENE__{sanitize_filename(group)}__"
                        f"{sanitize_filename(dsid, maxlen=80)}__rep{rep}"
                    )

                    source_label = make_short_cellxgene_label(
                        rank=rank,
                        composite_group=group,
                        row=r,
                        rep=rep,
                    )

                    source_type = f"CELLxGENE {group}"

                    sample_path = CELLXGENE_SAMPLE_DIR / f"sample_obs__{sanitize_filename(source_dataset)}.csv"
                    sample_obs.to_csv(sample_path, index=False)

                    specs.append({
                        "kind": "cellxgene",
                        "source_dataset": source_dataset,
                        "source_label": source_label,
                        "source_type": source_type,
                        "composite_group": group,
                        "dataset_id": str(dsid),
                        "resample_id": int(rep),
                        "sample_obs": sample_obs,
                    })

                    rows.append({
                        "source_dataset": source_dataset,
                        "source_label": source_label,
                        "source_type": source_type,
                        "composite_group": group,
                        "dataset_id": str(dsid),
                        "resample_id": int(rep),
                        "dataset_title": r["dataset_title"],
                        "cell_type_top": r["cell_type_top"],
                        "cell_types_seen": r["cell_types_seen"],
                        "tissue_top": r["tissue_top"],
                        "tissue_general_top": r["tissue_general_top"],
                        "tissues_seen": r["tissues_seen"],
                        "n_cells_available": int(r["n_cells"]),
                        "n_cells_sampled": int(len(sample_obs)),
                        "n_donors": int(r["n_donors"]),
                        "sample_obs_path": str(sample_path),
                    })

        chosen_df = pd.DataFrame(rows)
        chosen_df.to_csv(chosen_path, index=False)

        print(f"\n[saved] {chosen_path}")
        display(chosen_df)

        print("\n[CELLxGENE source counts by group]")
        if len(chosen_df):
            print(chosen_df["composite_group"].value_counts().reindex(ATLAS_GROUP_ORDER).fillna(0).astype(int))

        return specs

    def get_census_gene_table():
        cache_path = OUTDIR / f"census_human_var_feature_table__{CENSUS_VERSION}.parquet"

        if cache_path.exists():
            var = pd.read_parquet(cache_path)
            var["feature_name"] = safe_str_col(var, "feature_name").str.strip()
            return var

        print("[Census var] fetching human gene table")

        with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
            var = cellxgene_census.get_var(
                census,
                ORGANISM,
                column_names=["soma_joinid", "feature_id", "feature_name"],
            )

        var["feature_name"] = safe_str_col(var, "feature_name").str.strip()
        var = var.drop_duplicates("feature_name", keep="first").reset_index(drop=True)
        var.to_parquet(cache_path, index=False)

        return var

    def get_census_gene_map():
        var = get_census_gene_table()
        var["feature_name"] = safe_str_col(var, "feature_name").str.strip()
        var = var.drop_duplicates("feature_name", keep="first")
        return dict(zip(var["feature_name"].astype(str), var["soma_joinid"].astype(np.int64)))

    def build_union_gene_target_sets_for_cellxgene(datasets):
        census_gene_map = get_census_gene_map()
        census_genes = set(census_gene_map.keys())

        union_genes = sorted(set().union(*[
            set(d["genes"].astype(str).tolist())
            for d in datasets
        ]))
        union_genes = [g for g in union_genes if g in census_genes]

        union_targets = sorted(set().union(*[
            set(d["matched_target_genes"].astype(str).tolist())
            for d in datasets
        ]))
        union_targets = [g for g in union_targets if g in set(union_genes)]

        if len(union_genes) < MIN_OVERLAP_GENES:
            raise ValueError(f"CELLxGENE union genes too small: {len(union_genes)}")

        if len(union_targets) == 0:
            raise ValueError("No union target genes available in Census gene space.")

        print("\n[CELLxGENE union gene space]")
        print(f"union genes in Census: {len(union_genes):,}")
        print(f"union target genes:    {len(union_targets):,}")

        return union_genes, union_targets

    def make_stratum_labels(obs):
        obs = obs.copy()

        pieces = []
        for col in STRATUM_COLS:
            if col in obs.columns:
                pieces.append(safe_str_col(obs, col, default="unknown"))
            else:
                pieces.append(pd.Series("unknown", index=obs.index, dtype=object))

        out = pieces[0].astype(str)
        for p in pieces[1:]:
            out = out + "||" + p.astype(str)

        out = out.replace({"": "unknown"})
        return out.values.astype(str)

    def average_cov_over_strata(adata_cxg, target_idx_local, source_label):
        """
        Computes within-stratum covariance columns and averages them.

        Stratum = dataset_id/donor_id/cell_type by default.
        """
        obs = adata_cxg.obs.copy()
        stratum = make_stratum_labels(obs)

        vc = pd.Series(stratum).value_counts()
        valid_strata = vc[vc >= MIN_STRATUM_CELLS].index.tolist()

        print(f"[strata] {source_label}: total strata={len(vc):,}, valid strata={len(valid_strata):,}")

        cov_blocks = []
        weights = []
        stratum_rows = []

        residualize_depth = ATLAS_COVARIANCE_VARIANT == "depth_resid"

        for st in tqdm(valid_strata, desc=f"{source_label}: within-strata cov", ncols=TQDM_NCOLS):
            idx = np.flatnonzero(stratum == st)

            if len(idx) < MIN_STRATUM_CELLS:
                continue

            if len(idx) > MAX_CELLS_PER_STRATUM:
                seed = stable_seed(source_label, st, RANDOM_SEED, "stratum_sample")
                rr = np.random.default_rng(seed)
                idx = np.sort(rr.choice(idx, size=MAX_CELLS_PER_STRATUM, replace=False))

            Xst = adata_cxg.X[idx, :]

            try:
                res = compute_cov_columns_from_raw_counts(
                    Xst,
                    target_idx_local,
                    residualize_depth=residualize_depth,
                    name=f"{source_label} | stratum {st[:60]}",
                )
            except Exception as e:
                print(f"[WARN] skipping stratum {st[:80]} because: {repr(e)}")
                continue

            w = float(len(idx)) if WEIGHT_STRATA_BY_N_CELLS else 1.0

            cov_blocks.append(res["cov_cols"].astype(np.float32))
            weights.append(w)

            stratum_rows.append({
                "stratum": st,
                "n_cells_used": int(len(idx)),
                "weight": float(w),
                "depth_mean": float(res["depth_mean"]),
                "depth_median": float(res["depth_median"]),
                "depth_var": float(res["depth_var"]),
            })

            del Xst, res
            gc.collect()

        if len(cov_blocks) == 0:
            if not ALLOW_POOLED_FALLBACK_IF_NO_STRATA:
                raise RuntimeError(f"{source_label}: no valid strata for within-stratum covariance.")

            print(f"[WARN] {source_label}: no valid strata; falling back to pooled covariance.")
            res = compute_cov_columns_from_raw_counts(
                adata_cxg.X,
                target_idx_local,
                residualize_depth=residualize_depth,
                name=f"{source_label} | pooled fallback",
            )

            stratum_df = pd.DataFrame([{
                "stratum": "pooled_fallback",
                "n_cells_used": int(adata_cxg.n_obs),
                "weight": 1.0,
                "depth_mean": float(res["depth_mean"]),
                "depth_median": float(res["depth_median"]),
                "depth_var": float(res["depth_var"]),
            }])

            return res["cov_cols"].astype(np.float32), stratum_df, "pooled_fallback"

        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()

        cov_avg = np.zeros_like(cov_blocks[0], dtype=np.float64)
        for w, c in zip(weights, cov_blocks):
            cov_avg += w * c.astype(np.float64)

        cov_avg = cov_avg.astype(np.float32)

        stratum_df = pd.DataFrame(stratum_rows)
        return cov_avg, stratum_df, "within_strata"

    def compute_cellxgene_cov_source(source_spec, union_genes, union_targets):
        census_gene_map = get_census_gene_map()

        union_gene_to_idx = {g: i for i, g in enumerate(union_genes)}
        target_idx_local = np.asarray([union_gene_to_idx[g] for g in union_targets], dtype=np.int64)

        sample_obs = source_spec["sample_obs"].copy()
        source_dataset = source_spec["source_dataset"]
        source_label = source_spec["source_label"]
        source_type = source_spec["source_type"]
        composite_group = source_spec["composite_group"]

        gh = gene_hash(union_genes)
        th = gene_hash(union_targets)
        oh = hashlib.md5(
            "\n".join(sample_obs["soma_joinid"].astype(str).tolist()).encode()
        ).hexdigest()[:12]

        cov_tag = (
            f"{ATLAS_COVARIANCE_VARIANT}__"
            f"stratMin{MIN_STRATUM_CELLS}__"
            f"stratMax{MAX_CELLS_PER_STRATUM}__"
            f"wByN{int(WEIGHT_STRATA_BY_N_CELLS)}"
        )

        cache_path = CELLXGENE_COV_CACHE_DIR / (
            f"cov_source__{sanitize_filename(source_dataset)}__"
            f"{cov_tag}__"
            f"cells{len(sample_obs)}__genes{len(union_genes)}__targets{len(union_targets)}__"
            f"{gh}_{th}_{oh}__{CENSUS_VERSION}.npz"
        )

        diag_path = DIAGNOSTIC_DIR / f"source_diagnostics__{sanitize_filename(source_dataset)}__{cov_tag}.json"
        target_diag_path = DIAGNOSTIC_DIR / f"target_diagnostics__{sanitize_filename(source_dataset)}__{cov_tag}.csv"
        stratum_diag_path = DIAGNOSTIC_DIR / f"stratum_diagnostics__{sanitize_filename(source_dataset)}__{cov_tag}.csv"

        # the npz holds all data the cached branch returns; the diag sidecar is only a
        # diagnostics artifact (safe to be absent, e.g. after a caches cleanup), so do NOT
        # require it for cache validity -- otherwise valid covariances are needlessly rebuilt.
        if cache_path.exists() and not FORCE_REBUILD_CELLXGENE_COV:
            z = np.load(cache_path, allow_pickle=True)
            cov_cols = z["cov_cols"]

            if np.isfinite(cov_cols).all():
                print(f"[CELLxGENE cov] loading cached valid {cache_path}")

                union_genes_loaded = z["union_genes"].astype(str)
                union_targets_loaded = z["union_targets"].astype(str)

                return {
                    "kind": "cellxgene_cov",
                    "source_dataset": source_dataset,
                    "source_label": source_label,
                    "source_type": source_type,
                    "composite_group": composite_group,
                    "covariance_variant": str(z["covariance_variant"]),
                    "covariance_mode": str(z["covariance_mode"]),
                    "cov_cols": cov_cols.astype(np.float32),
                    "union_genes": union_genes_loaded,
                    "union_gene_to_idx": {g: i for i, g in enumerate(union_genes_loaded.tolist())},
                    "union_targets": union_targets_loaded,
                    "target_gene_to_col": {g: i for i, g in enumerate(union_targets_loaded.tolist())},
                    "gene_mean": z["gene_mean"].astype(np.float32),
                    "gene_detection": z["gene_detection"].astype(np.float32),
                    "gene_var": z["gene_var"].astype(np.float32),
                    "target_mean": z["target_mean"].astype(np.float32),
                    "target_detection": z["target_detection"].astype(np.float32),
                    "target_var": z["target_var"].astype(np.float32),
                    "n_cells": int(z["n_cells"]),
                    "n_valid_strata": int(z["n_valid_strata"]),
                    "cache_path": str(cache_path),
                }

            bad = int((~np.isfinite(cov_cols)).sum())
            print(f"[CELLxGENE cov] bad cache has {bad:,} non-finite values; deleting {cache_path}")
            try:
                cache_path.unlink()
            except Exception as e:
                print(f"[CELLxGENE cov] could not delete bad cache: {repr(e)}")

        print("\n" + "=" * 120)
        print(f"[CELLxGENE cov] computing source")
        print(f"group:              {composite_group}")
        print(f"source:             {source_label}")
        print(f"type:               {source_type}")
        print(f"covariance_variant: {ATLAS_COVARIANCE_VARIANT}")
        print(f"cells:              {len(sample_obs):,}")
        print(f"genes:              {len(union_genes):,}")
        print(f"targets:            {len(union_targets):,}")
        print("=" * 120)

        obs_coords = sample_obs["soma_joinid"].astype(np.int64).values
        var_coords = np.asarray([census_gene_map[g] for g in union_genes], dtype=np.int64)

        with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
            adata_cxg = cellxgene_census.get_anndata(
                census=census,
                organism=ORGANISM,
                obs_coords=obs_coords,
                var_coords=var_coords,
                X_name="raw",
                obs_column_names=[
                    "soma_joinid",
                    "dataset_id",
                    "cell_type",
                    "tissue",
                    "tissue_general",
                    "disease",
                    "assay",
                    "donor_id",
                    "suspension_type",
                    "is_primary_data",
                ],
                var_column_names=[
                    "soma_joinid",
                    "feature_id",
                    "feature_name",
                ],
            )

        if "feature_name" in adata_cxg.var.columns:
            adata_cxg.var_names = safe_str_col(adata_cxg.var, "feature_name").values

        adata_cxg.var_names = adata_cxg.var_names.astype(str)
        adata_cxg.var_names_make_unique()

        missing = [g for g in union_genes if g not in set(adata_cxg.var_names)]
        if len(missing):
            raise ValueError(f"CELLxGENE AnnData missing {len(missing)} genes. Example: {missing[:10]}")

        adata_cxg = adata_cxg[:, union_genes].copy()

        validate_raw_counts_matrix(adata_cxg.X, name=f"CELLxGENE raw source {source_label}", integer_check=True)

        # Gene stats over all sampled source cells.
        stats_all = compute_gene_stats_from_raw_counts(adata_cxg.X)
        gene_mean = stats_all["mean"].astype(np.float32)
        gene_detection = stats_all["detection"].astype(np.float32)
        gene_var = stats_all["var"].astype(np.float32)
        depth = stats_all["depth"]

        cov_cols, stratum_df, covariance_mode = average_cov_over_strata(
            adata_cxg=adata_cxg,
            target_idx_local=target_idx_local,
            source_label=source_label,
        )

        if not np.isfinite(cov_cols).all():
            bad = int((~np.isfinite(cov_cols)).sum())
            raise FloatingPointError(f"CELLxGENE covariance has {bad:,} non-finite values; refusing to save.")

        target_mean = gene_mean[target_idx_local]
        target_detection = gene_detection[target_idx_local]
        target_var = gene_var[target_idx_local]

        n_valid_strata = int(len(stratum_df))

        np.savez_compressed(
            cache_path,
            cov_cols=cov_cols.astype(np.float32),
            union_genes=np.asarray(union_genes, dtype=object),
            union_targets=np.asarray(union_targets, dtype=object),
            gene_mean=gene_mean,
            gene_detection=gene_detection,
            gene_var=gene_var,
            target_mean=target_mean,
            target_detection=target_detection,
            target_var=target_var,
            n_cells=np.asarray(len(sample_obs), dtype=np.int64),
            n_valid_strata=np.asarray(n_valid_strata, dtype=np.int64),
            covariance_variant=np.asarray(ATLAS_COVARIANCE_VARIANT, dtype=object),
            covariance_mode=np.asarray(covariance_mode, dtype=object),
        )

        target_diag = pd.DataFrame({
            "target_gene": np.asarray(union_targets, dtype=str),
            "mean": target_mean,
            "detection": target_detection,
            "var": target_var,
        })
        target_diag.to_csv(target_diag_path, index=False)

        stratum_df.to_csv(stratum_diag_path, index=False)

        diag = {
            "source_dataset": source_dataset,
            "source_label": source_label,
            "source_type": source_type,
            "composite_group": composite_group,
            "dataset_id": source_spec.get("dataset_id", ""),
            "resample_id": source_spec.get("resample_id", -1),
            "n_cells": int(len(sample_obs)),
            "n_genes": int(len(union_genes)),
            "n_targets": int(len(union_targets)),
            "n_valid_strata": int(n_valid_strata),
            "covariance_variant": ATLAS_COVARIANCE_VARIANT,
            "covariance_mode": covariance_mode,
            "min_stratum_cells": MIN_STRATUM_CELLS,
            "max_cells_per_stratum": MAX_CELLS_PER_STRATUM,
            "weight_strata_by_n_cells": WEIGHT_STRATA_BY_N_CELLS,
            "depth_mean": float(np.mean(depth)),
            "depth_median": float(np.median(depth)),
            "depth_std": float(np.std(depth, ddof=1)) if len(depth) > 1 else 0.0,
            "median_target_mean": float(np.median(target_mean)),
            "median_target_detection": float(np.median(target_detection)),
            "n_targets_passing_expression": int(
                np.sum((target_mean >= MIN_ATLAS_TARGET_MEAN) | (target_detection >= MIN_ATLAS_TARGET_DETECTION))
            ),
            "cache_path": str(cache_path),
            "target_diagnostics_path": str(target_diag_path),
            "stratum_diagnostics_path": str(stratum_diag_path),
        }

        with open(diag_path, "w") as f:
            json.dump(diag, f, indent=2, default=json_default)

        print(f"[CELLxGENE cov] saved {cache_path}")
        print(f"[diagnostics] saved {diag_path}")
        print(f"[diagnostics] saved {target_diag_path}")
        print(f"[diagnostics] saved {stratum_diag_path}")

        return {
            "kind": "cellxgene_cov",
            "source_dataset": source_dataset,
            "source_label": source_label,
            "source_type": source_type,
            "composite_group": composite_group,
            "covariance_variant": ATLAS_COVARIANCE_VARIANT,
            "covariance_mode": covariance_mode,
            "cov_cols": cov_cols.astype(np.float32),
            "union_genes": np.asarray(union_genes, dtype=str),
            "union_gene_to_idx": union_gene_to_idx,
            "union_targets": np.asarray(union_targets, dtype=str),
            "target_gene_to_col": {g: i for i, g in enumerate(union_targets)},
            "gene_mean": gene_mean,
            "gene_detection": gene_detection,
            "gene_var": gene_var,
            "target_mean": target_mean,
            "target_detection": target_detection,
            "target_var": target_var,
            "n_cells": len(sample_obs),
            "n_valid_strata": n_valid_strata,
            "cache_path": str(cache_path),
        }

    def evaluate_response_vs_marson_source(response, source):
        mode = "marson_full"
        cache = pair_cache_paths(response["dataset"], source["dataset"], mode=mode)

        if (
            cache["summary"].exists()
            and (cache["per_pert"].exists() or not SAVE_PER_PERTURBATION_TABLE)
            and not FORCE_RECOMPUTE_PAIR_METRICS
        ):
            with open(cache["summary"], "r") as f:
                summary = json.load(f)
            pert_df = pd.read_csv(cache["per_pert"]) if cache["per_pert"].exists() and SAVE_PER_PERTURBATION_TABLE else None
            return pert_df, summary

        composite_group = classify_marson_pair(response["dataset"], source["dataset"])
        overlap_genes = np.intersect1d(response["genes"], source["genes"])

        meta = {
            "response_dataset": response["dataset"],
            "response_label": response["label"],
            "response_condition": response["condition"],
            "source_dataset": source["dataset"],
            "source_label": source["label"],
            "source_condition": source["condition"],
            "source_type": "Marson precomputed Sigma",
            "composite_group": composite_group,
            "covariance_variant": "precomputed",
            "covariance_mode": "full",
            "n_overlap_genes": int(len(overlap_genes)),
        }

        if len(overlap_genes) < MIN_OVERLAP_GENES:
            summary = dict(meta)
            summary.update({"n_eval": 0, "skip_reason": f"too few overlap genes: {len(overlap_genes)}"})
            with open(cache["summary"], "w") as f:
                json.dump(summary, f, indent=2, default=json_default)
            return None, summary

        resp_idx = np.asarray([response["gene_to_idx"][g] for g in overlap_genes], dtype=np.int64)
        src_idx = np.asarray([source["gene_to_idx"][g] for g in overlap_genes], dtype=np.int64)
        overlap_pos = {g: i for i, g in enumerate(overlap_genes.tolist())}

        src_gene_to_idx = source["gene_to_idx"]
        valid_mask = np.asarray(
            [tg in src_gene_to_idx for tg in response["matched_target_genes"]],
            dtype=bool,
        )
        valid_pos = np.flatnonzero(valid_mask)

        if len(valid_pos) == 0:
            summary = dict(meta)
            summary.update({"n_eval": 0, "skip_reason": "no response targets present in source genes"})
            with open(cache["summary"], "w") as f:
                json.dump(summary, f, indent=2, default=json_default)
            return None, summary

        rows_out = []
        Sigma = source["Sigma"]

        with h5py.File(response["stats_path"], "r") as h5:
            dx_ds = h5["dx"]

            for start in range(0, len(valid_pos), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(valid_pos))
                pos = valid_pos[start:end]

                pert_rows = response["matched_pos"][pos]
                target_genes = response["matched_target_genes"][pos].astype(str)

                source_target_idx = np.asarray(
                    [src_gene_to_idx[g] for g in target_genes],
                    dtype=np.int64,
                )

                dx = np.asarray(dx_ds[pert_rows, :], dtype=np.float32)[:, resp_idx]
                basis = np.asarray(Sigma[np.ix_(src_idx, source_target_idx)], dtype=np.float32).T

                pearson, r2, alpha, dx_norm2, basis_norm2 = lr_metrics_batch_all(dx, basis, eps=EPS)

                exclude_pos = np.asarray([overlap_pos.get(g, -1) for g in target_genes], dtype=np.int64)
                pearson_lto, r2_lto, alpha_lto, dx_norm2_lto, basis_norm2_lto = lr_metrics_leave_one_coordinate(
                    dx,
                    basis,
                    exclude_pos=exclude_pos,
                    eps=EPS,
                )

                for j, ppos in enumerate(pos):
                    rows_out.append({
                        **meta,
                        "perturbation": str(response["matched_perts"][ppos]),
                        "target_gene": str(response["matched_target_genes"][ppos]),
                        "target_in_metric": int(exclude_pos[j] >= 0),
                        "pearson": float(pearson[j]),
                        "r2_uncentered": float(r2[j]),
                        "alpha": float(alpha[j]),
                        "dx_norm2": float(dx_norm2[j]),
                        "basis_norm2": float(basis_norm2[j]),
                        "pearson_lto": float(pearson_lto[j]),
                        "r2_uncentered_lto": float(r2_lto[j]),
                        "alpha_lto": float(alpha_lto[j]),
                        "dx_norm2_lto": float(dx_norm2_lto[j]),
                        "basis_norm2_lto": float(basis_norm2_lto[j]),
                        "source_target_mean": np.nan,
                        "source_target_detection": np.nan,
                        "source_target_var": np.nan,
                        "basis_mode": "matched_target",
                    })

                del dx, basis
                gc.collect()

        pert_df = pd.DataFrame(rows_out)
        summary = summarize_pair_from_perts(pert_df, meta)

        with open(cache["summary"], "w") as f:
            json.dump(summary, f, indent=2, default=json_default)

        if SAVE_PER_PERTURBATION_TABLE:
            pert_df.to_csv(cache["per_pert"], index=False)

        return pert_df, summary

    def evaluate_response_vs_cellxgene_source(response, cxg_source, target_shuffled=False):
        mode = "cellxgene_target_shuffled" if target_shuffled else "cellxgene_matched_target"

        source_dataset_for_cache = cxg_source["source_dataset"]
        if target_shuffled:
            source_dataset_for_cache = source_dataset_for_cache + "__target_shuffled"

        cache = pair_cache_paths(response["dataset"], source_dataset_for_cache, mode=mode)

        if (
            cache["summary"].exists()
            and (cache["per_pert"].exists() or not SAVE_PER_PERTURBATION_TABLE)
            and not FORCE_RECOMPUTE_PAIR_METRICS
        ):
            with open(cache["summary"], "r") as f:
                summary = json.load(f)
            pert_df = pd.read_csv(cache["per_pert"]) if cache["per_pert"].exists() and SAVE_PER_PERTURBATION_TABLE else None
            return pert_df, summary

        union_gene_to_idx = cxg_source["union_gene_to_idx"]
        target_gene_to_col = cxg_source["target_gene_to_col"]
        cov_cols = cxg_source["cov_cols"]

        if target_shuffled:
            composite_group = "Atlas target-shuffled negative"
            source_label = cxg_source["source_label"] + " | target-shuffled"
            source_dataset = cxg_source["source_dataset"] + "__target_shuffled"
            source_type = cxg_source["source_type"] + " | target-shuffled negative"
        else:
            composite_group = cxg_source["composite_group"]
            source_label = cxg_source["source_label"]
            source_dataset = cxg_source["source_dataset"]
            source_type = cxg_source["source_type"]

        # Filter row genes by source expression/detection.
        response_gene_list = response["genes"].astype(str).tolist()
        overlap_genes = []
        for g in response_gene_list:
            if g not in union_gene_to_idx:
                continue
            gi = union_gene_to_idx[g]
            if (
                cxg_source["gene_mean"][gi] >= MIN_ATLAS_ROW_MEAN
                or cxg_source["gene_detection"][gi] >= MIN_ATLAS_ROW_DETECTION
            ):
                overlap_genes.append(g)

        meta = {
            "response_dataset": response["dataset"],
            "response_label": response["label"],
            "response_condition": response["condition"],
            "source_dataset": source_dataset,
            "source_label": source_label,
            "source_condition": "",
            "source_type": source_type,
            "composite_group": composite_group,
            "covariance_variant": cxg_source["covariance_variant"],
            "covariance_mode": cxg_source["covariance_mode"],
            "n_overlap_genes": int(len(overlap_genes)),
            "n_source_cells": int(cxg_source["n_cells"]),
            "n_valid_strata": int(cxg_source["n_valid_strata"]),
        }

        if len(overlap_genes) < MIN_OVERLAP_GENES:
            summary = dict(meta)
            summary.update({"n_eval": 0, "skip_reason": f"too few expressed overlap genes: {len(overlap_genes)}"})
            with open(cache["summary"], "w") as f:
                json.dump(summary, f, indent=2, default=json_default)
            return None, summary

        resp_idx = np.asarray([response["gene_to_idx"][g] for g in overlap_genes], dtype=np.int64)
        cxg_row_idx = np.asarray([union_gene_to_idx[g] for g in overlap_genes], dtype=np.int64)
        overlap_pos = {g: i for i, g in enumerate(overlap_genes)}

        # Target gene must exist and be expressed/detected in source.
        valid_pos = []
        valid_target_cols = []

        for i, tg in enumerate(response["matched_target_genes"].astype(str)):
            if tg not in target_gene_to_col:
                continue

            col = target_gene_to_col[tg]
            target_mean = float(cxg_source["target_mean"][col])
            target_det = float(cxg_source["target_detection"][col])

            if target_mean < MIN_ATLAS_TARGET_MEAN and target_det < MIN_ATLAS_TARGET_DETECTION:
                continue

            valid_pos.append(i)
            valid_target_cols.append(col)

        valid_pos = np.asarray(valid_pos, dtype=np.int64)
        valid_target_cols = np.asarray(valid_target_cols, dtype=np.int64)

        if len(valid_pos) == 0:
            summary = dict(meta)
            summary.update({"n_eval": 0, "skip_reason": "no response targets pass atlas target expression/detection filters"})
            with open(cache["summary"], "w") as f:
                json.dump(summary, f, indent=2, default=json_default)
            return None, summary

        # For negative control, randomly map each perturbation to a wrong target column.
        if target_shuffled:
            rr = np.random.default_rng(stable_seed(response["dataset"], cxg_source["source_dataset"], "target_shuffle"))
            all_cols = np.arange(len(cxg_source["union_targets"]), dtype=np.int64)
            shuffled_cols = []

            for true_col in valid_target_cols:
                if len(all_cols) <= 1:
                    shuffled_cols.append(true_col)
                else:
                    choices = all_cols[all_cols != true_col]
                    shuffled_cols.append(int(rr.choice(choices)))
            valid_target_cols_eval = np.asarray(shuffled_cols, dtype=np.int64)
        else:
            valid_target_cols_eval = valid_target_cols

        rows_out = []

        with h5py.File(response["stats_path"], "r") as h5:
            dx_ds = h5["dx"]

            for start in range(0, len(valid_pos), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(valid_pos))
                pos = valid_pos[start:end]
                col_idx = valid_target_cols_eval[start:end]
                true_col_idx = valid_target_cols[start:end]

                pert_rows = response["matched_pos"][pos]
                target_genes = response["matched_target_genes"][pos].astype(str)

                dx = np.asarray(dx_ds[pert_rows, :], dtype=np.float32)[:, resp_idx]
                basis = np.asarray(cov_cols[np.ix_(cxg_row_idx, col_idx)], dtype=np.float32).T

                pearson, r2, alpha, dx_norm2, basis_norm2 = lr_metrics_batch_all(dx, basis, eps=EPS)

                exclude_pos = np.asarray([overlap_pos.get(g, -1) for g in target_genes], dtype=np.int64)
                pearson_lto, r2_lto, alpha_lto, dx_norm2_lto, basis_norm2_lto = lr_metrics_leave_one_coordinate(
                    dx,
                    basis,
                    exclude_pos=exclude_pos,
                    eps=EPS,
                )

                # Basis norm filter after computing, to keep diagnostics but mark low-basis as NaN.
                bad_basis = basis_norm2 < MIN_BASIS_NORM2
                pearson[bad_basis] = np.nan
                r2[bad_basis] = np.nan
                alpha[bad_basis] = np.nan

                bad_basis_lto = basis_norm2_lto < MIN_BASIS_NORM2
                pearson_lto[bad_basis_lto] = np.nan
                r2_lto[bad_basis_lto] = np.nan
                alpha_lto[bad_basis_lto] = np.nan

                for j, ppos in enumerate(pos):
                    true_col = int(true_col_idx[j])
                    eval_col = int(col_idx[j])

                    rows_out.append({
                        **meta,
                        "perturbation": str(response["matched_perts"][ppos]),
                        "target_gene": str(response["matched_target_genes"][ppos]),
                        "eval_basis_gene": str(cxg_source["union_targets"][eval_col]),
                        "target_in_metric": int(exclude_pos[j] >= 0),
                        "pearson": float(pearson[j]),
                        "r2_uncentered": float(r2[j]),
                        "alpha": float(alpha[j]),
                        "dx_norm2": float(dx_norm2[j]),
                        "basis_norm2": float(basis_norm2[j]),
                        "pearson_lto": float(pearson_lto[j]),
                        "r2_uncentered_lto": float(r2_lto[j]),
                        "alpha_lto": float(alpha_lto[j]),
                        "dx_norm2_lto": float(dx_norm2_lto[j]),
                        "basis_norm2_lto": float(basis_norm2_lto[j]),
                        "source_target_mean": float(cxg_source["target_mean"][true_col]),
                        "source_target_detection": float(cxg_source["target_detection"][true_col]),
                        "source_target_var": float(cxg_source["target_var"][true_col]),
                        "basis_mode": "target_shuffled" if target_shuffled else "matched_target",
                    })

                del dx, basis
                gc.collect()

        pert_df = pd.DataFrame(rows_out)
        summary = summarize_pair_from_perts(pert_df, meta)

        with open(cache["summary"], "w") as f:
            json.dump(summary, f, indent=2, default=json_default)

        if SAVE_PER_PERTURBATION_TABLE:
            pert_df.to_csv(cache["per_pert"], index=False)

        return pert_df, summary

    def composite_summary_table(df, metric_col):
        rows = []

        for group in COMPOSITE_GROUP_ORDER:
            vals = pd.to_numeric(
                df.loc[df["composite_group"] == group, metric_col],
                errors="coerce",
            ).values
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                rows.append({
                    "composite_group": group,
                    "n": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "sem": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                })
            else:
                rows.append({
                    "composite_group": group,
                    "n": int(len(vals)),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                })

        return pd.DataFrame(rows)

    def plot_composite_violin(
        df,
        metric_col,
        ylabel,
        title,
        save_prefix,
        pair_level=True,
    ):
        plot_df = df.copy()

        if metric_col not in plot_df.columns:
            raise ValueError(f"Missing metric column: {metric_col}")

        plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
        plot_df = plot_df[np.isfinite(plot_df[metric_col].values)].copy()
        plot_df = plot_df[plot_df["composite_group"].isin(COMPOSITE_GROUP_ORDER)].copy()

        values = []
        groups = []
        colors = []

        for group in COMPOSITE_GROUP_ORDER:
            vals = plot_df.loc[plot_df["composite_group"] == group, metric_col].values
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                print(f"[plot] skipping empty group: {group}")
                continue

            if len(vals) > VIOLIN_MAX_POINTS_PER_GROUP:
                vals_for_violin = rng.choice(vals, size=VIOLIN_MAX_POINTS_PER_GROUP, replace=False)
            else:
                vals_for_violin = vals

            values.append(vals_for_violin)
            groups.append(group)
            colors.append(COMPOSITE_GROUP_COLORS.get(group, "#999999"))

        if len(values) == 0:
            raise ValueError("No non-empty composite groups to plot.")

        summary = composite_summary_table(plot_df, metric_col)
        summary_path = OUTDIR / f"{save_prefix}__summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[saved] {summary_path}")
        display(summary)

        fig_w = max(14.5, 0.95 * len(groups) + 4.0)
        plt.figure(figsize=(fig_w, 6.5))

        parts = plt.violinplot(
            values,
            positions=np.arange(1, len(values) + 1),
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.9,
        )

        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.45)
            body.set_linewidth(1.0)

        bp = plt.boxplot(
            values,
            positions=np.arange(1, len(values) + 1),
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            medianprops={"linewidth": 2.5, "color": "black"},
            boxprops={"linewidth": 1.0, "color": "black"},
            whiskerprops={"linewidth": 1.0, "color": "black"},
            capprops={"linewidth": 1.0, "color": "black"},
        )

        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_alpha(0.78)

        for i, group in enumerate(groups, start=1):
            vals = plot_df.loc[plot_df["composite_group"] == group, metric_col].values
            vals = vals[np.isfinite(vals)]

            if len(vals) > JITTER_MAX_POINTS_PER_GROUP:
                vals = rng.choice(vals, size=JITTER_MAX_POINTS_PER_GROUP, replace=False)

            jitter = rng.normal(loc=0.0, scale=0.055, size=len(vals))

            plt.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=9 if pair_level else 6,
                alpha=0.28 if pair_level else 0.12,
                color="black",
                linewidths=0,
                zorder=3,
            )

        plt.axhline(0, color="black", linewidth=1.2, alpha=0.8)

        labels = [COMPOSITE_GROUP_LABELS.get(g, g) for g in groups]

        plt.xticks(
            np.arange(1, len(values) + 1),
            labels,
            rotation=35,
            ha="right",
            fontsize=11,
        )
        plt.ylabel(ylabel, fontsize=15)
        plt.title(title, fontsize=18, pad=12)
        plt.grid(axis="y", alpha=0.22)

        finite_vals = plot_df[metric_col].values
        finite_vals = finite_vals[np.isfinite(finite_vals)]

        if len(finite_vals):
            lo = float(np.nanpercentile(finite_vals, 0.5))
            hi = float(np.nanpercentile(finite_vals, 99.5))
            pad = 0.08 * (hi - lo + 1e-8)
            plt.ylim(lo - pad, hi + pad)

        plt.tight_layout()

        png = OUTDIR / f"{save_prefix}.png"
        svg = OUTDIR / f"{save_prefix}.svg"
        pdf = OUTDIR / f"{save_prefix}.pdf"

        plt.savefig(png, dpi=DPI, bbox_inches="tight")
        plt.savefig(svg, bbox_inches="tight")
        plt.savefig(pdf, bbox_inches="tight")

        print(f"[saved] {png}")
        print(f"[saved] {svg}")
        print(f"[saved] {pdf}")

        plt.show()

    print("\n" + "=" * 120)

    print("LOAD PRECOMPUTED MARSON DATASETS")

    print("No Marson h5ad is loaded. No Marson covariance/correlation is recomputed.")

    print("=" * 120)

    datasets = load_all_precomputed_marson()

    print("\n" + "=" * 120)

    print("CHOOSE CONTROLLED CELLXGENE ATLAS SOURCES")

    print("=" * 120)

    cellxgene_sample_specs = choose_cellxgene_sources()

    if len(cellxgene_sample_specs) == 0:
        raise RuntimeError("No CELLxGENE source specs selected.")

    print("\n" + "=" * 120)

    print("BUILD CELLXGENE RAW COVARIANCE SOURCES")

    print("Atlas covariance is within-stratum averaged; Marson covariance remains precomputed.")

    print("=" * 120)

    union_genes, union_targets = build_union_gene_target_sets_for_cellxgene(datasets)

    cellxgene_cov_sources = []

    source_errors = []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _build_one_cov_source(spec):
        return compute_cellxgene_cov_source(
            source_spec=spec,
            union_genes=union_genes,
            union_targets=union_targets,
        )

    _workers = max(1, int(CELLXGENE_COV_WORKERS))
    with ThreadPoolExecutor(max_workers=_workers) as _ex:
        _futs = {_ex.submit(_build_one_cov_source, spec): spec for spec in cellxgene_sample_specs}
        for _fut in tqdm(as_completed(_futs), total=len(_futs),
                         desc="CELLxGENE covariance sources", ncols=TQDM_NCOLS):
            spec = _futs[_fut]
            try:
                cxg_source = _fut.result()
                cellxgene_cov_sources.append(cxg_source)
            except Exception as e:
                print(f"[ERROR CELLxGENE cov source] {spec.get('source_label', 'unknown')}: {repr(e)}")
                source_errors.append({
                "source_dataset": spec.get("source_dataset", ""),
                "source_label": spec.get("source_label", ""),
                "composite_group": spec.get("composite_group", ""),
                "error": repr(e),
            })
            gc.collect()

    source_errors_path = OUTDIR / "cellxgene_cov_source_errors.json"

    with open(source_errors_path, "w") as f:
        json.dump(source_errors, f, indent=2, default=json_default)

    print(f"[saved] {source_errors_path}")

    if len(cellxgene_cov_sources) == 0:
        raise RuntimeError("No CELLxGENE covariance sources were built.")

    source_diag_rows = []

    for s in cellxgene_cov_sources:
        source_diag_rows.append({
            "source_dataset": s["source_dataset"],
            "source_label": s["source_label"],
            "source_type": s["source_type"],
            "composite_group": s["composite_group"],
            "covariance_variant": s["covariance_variant"],
            "covariance_mode": s["covariance_mode"],
            "n_cells": s["n_cells"],
            "n_valid_strata": s["n_valid_strata"],
            "median_gene_mean": float(np.median(s["gene_mean"])),
            "median_gene_detection": float(np.median(s["gene_detection"])),
            "median_target_mean": float(np.median(s["target_mean"])),
            "median_target_detection": float(np.median(s["target_detection"])),
            "n_targets_passing_expression": int(
                np.sum((s["target_mean"] >= MIN_ATLAS_TARGET_MEAN) | (s["target_detection"] >= MIN_ATLAS_TARGET_DETECTION))
            ),
            "cache_path": s["cache_path"],
        })

    source_diag_df = pd.DataFrame(source_diag_rows)

    source_diag_summary_path = OUTDIR / "cellxgene_cov_source_diagnostics_summary.csv"

    source_diag_df.to_csv(source_diag_summary_path, index=False)

    print(f"[saved] {source_diag_summary_path}")

    display(source_diag_df.head(30))

    print("\n" + "=" * 120)

    print("EVALUATE RESPONSE DATASETS")

    print("=" * 120)

    all_pair_summaries = []

    all_pert_rows = []

    all_errors = []

    for response in tqdm(datasets, desc="response Marson datasets", ncols=TQDM_NCOLS):
        print("\n" + "#" * 120)
        print(f"[response] {response['label']} | condition={response['condition']}")
        print("#" * 120)

        # Marson precomputed Sigma sources.
        for source in tqdm(
            datasets,
            desc=f"{response['label']}: Marson Sigma sources",
            ncols=TQDM_NCOLS,
            leave=False,
        ):
            try:
                pert_df, summary = evaluate_response_vs_marson_source(response, source)
                all_pair_summaries.append(summary)

                if SAVE_PER_PERTURBATION_TABLE and pert_df is not None and len(pert_df):
                    all_pert_rows.append(pert_df)

                print(
                    f"[Marson pair] {summary['composite_group']} | "
                    f"response={response['label']} | source={source['label']} | "
                    f"n={summary.get('n_eval', 0):,} | "
                    f"mean Pearson LTO={summary.get('mean_pearson_lto', np.nan):.4g} | "
                    f"mean R2 LTO={summary.get('mean_r2_uncentered_lto', np.nan):.4g}"
                )

            except Exception as e:
                print(f"[ERROR Marson pair] response={response['label']} source={source['label']}: {repr(e)}")
                all_errors.append({
                    "response_dataset": response["dataset"],
                    "source_dataset": source["dataset"],
                    "source_type": "Marson precomputed Sigma",
                    "error": repr(e),
                })
                gc.collect()

        # CELLxGENE matched target sources.
        for cxg_source in tqdm(
            cellxgene_cov_sources,
            desc=f"{response['label']}: CELLxGENE matched-target sources",
            ncols=TQDM_NCOLS,
            leave=False,
        ):
            try:
                pert_df, summary = evaluate_response_vs_cellxgene_source(
                    response=response,
                    cxg_source=cxg_source,
                    target_shuffled=False,
                )
                all_pair_summaries.append(summary)

                if SAVE_PER_PERTURBATION_TABLE and pert_df is not None and len(pert_df):
                    all_pert_rows.append(pert_df)

                print(
                    f"[Atlas pair] {summary['composite_group']} | "
                    f"response={response['label']} | source={cxg_source['source_label']} | "
                    f"n={summary.get('n_eval', 0):,} | "
                    f"mean Pearson LTO={summary.get('mean_pearson_lto', np.nan):.4g} | "
                    f"mean R2 LTO={summary.get('mean_r2_uncentered_lto', np.nan):.4g}"
                )

            except Exception as e:
                print(f"[ERROR CELLxGENE pair] response={response['label']} source={cxg_source['source_label']}: {repr(e)}")
                all_errors.append({
                    "response_dataset": response["dataset"],
                    "source_dataset": cxg_source["source_dataset"],
                    "source_type": cxg_source["source_type"],
                    "composite_group": cxg_source["composite_group"],
                    "error": repr(e),
                })
                gc.collect()

        # CELLxGENE target-shuffled negative.
        if RUN_ATLAS_TARGET_SHUFFLED_NEGATIVE:
            for cxg_source in tqdm(
                cellxgene_cov_sources,
                desc=f"{response['label']}: CELLxGENE target-shuffled negative",
                ncols=TQDM_NCOLS,
                leave=False,
            ):
                try:
                    pert_df, summary = evaluate_response_vs_cellxgene_source(
                        response=response,
                        cxg_source=cxg_source,
                        target_shuffled=True,
                    )
                    all_pair_summaries.append(summary)

                    if SAVE_PER_PERTURBATION_TABLE and pert_df is not None and len(pert_df):
                        all_pert_rows.append(pert_df)

                    print(
                        f"[Negative pair] {summary['composite_group']} | "
                        f"response={response['label']} | source={cxg_source['source_label']} | "
                        f"n={summary.get('n_eval', 0):,} | "
                        f"mean Pearson LTO={summary.get('mean_pearson_lto', np.nan):.4g}"
                    )

                except Exception as e:
                    print(f"[ERROR target-shuffled pair] response={response['label']} source={cxg_source['source_label']}: {repr(e)}")
                    all_errors.append({
                        "response_dataset": response["dataset"],
                        "source_dataset": cxg_source["source_dataset"] + "__target_shuffled",
                        "source_type": cxg_source["source_type"] + " | target-shuffled",
                        "composite_group": "Atlas target-shuffled negative",
                        "error": repr(e),
                    })
                    gc.collect()

        gc.collect()

    pair_summary = pd.DataFrame(all_pair_summaries)

    pair_summary_path = OUTDIR / "robust_pair_summary.csv"

    pair_summary.to_csv(pair_summary_path, index=False)

    print(f"[saved] {pair_summary_path}")

    if SAVE_PER_PERTURBATION_TABLE and len(all_pert_rows):
        per_pert_df = pd.concat(all_pert_rows, ignore_index=True)
    else:
        per_pert_df = pd.DataFrame()

    per_pert_path = OUTDIR / "robust_per_perturbation_metrics.csv"

    per_pert_df.to_csv(per_pert_path, index=False)

    print(f"[saved] {per_pert_path}")

    errors_path = OUTDIR / "robust_errors.json"

    with open(errors_path, "w") as f:
        json.dump(all_errors, f, indent=2, default=json_default)

    print(f"[saved] {errors_path}")

    print("\n[pair summary counts]")

    if len(pair_summary):
        cols_to_show = [
            "composite_group",
            "n_pairs",
            "mean_pair_pearson_lto",
            "mean_pair_r2_lto",
            "total_eval_perts",
        ]

        pair_counts = (
            pair_summary
            .groupby("composite_group", as_index=False)
            .agg(
                n_pairs=("source_dataset", "count"),
                mean_pair_pearson_lto=("mean_pearson_lto", "mean"),
                mean_pair_r2_lto=("mean_r2_uncentered_lto", "mean"),
                total_eval_perts=("n_eval", "sum"),
            )
            .set_index("composite_group")
            .reindex(COMPOSITE_GROUP_ORDER)
            .reset_index()
        )

        display(pair_counts[cols_to_show])

    print("\n[per-perturbation counts]")

    if len(per_pert_df):
        print(per_pert_df["composite_group"].value_counts().reindex(COMPOSITE_GROUP_ORDER).fillna(0).astype(int))

    if len(pair_summary) == 0:
        raise RuntimeError("No pair summaries were generated; cannot plot pair-level violins.")

    plot_composite_violin(
        df=pair_summary,
        metric_col="mean_pearson_lto",
        ylabel="Mean Pearson, leave-target-out",
        title="Pair-level transfer performance: leave-target-out Pearson",
        save_prefix="PAIR_LEVEL_VIOLIN_mean_pearson_leave_target_out",
        pair_level=True,
    )

    plot_composite_violin(
        df=pair_summary,
        metric_col="mean_r2_uncentered_lto",
        ylabel="Mean uncentered R², leave-target-out",
        title="Pair-level transfer performance: leave-target-out R²",
        save_prefix="PAIR_LEVEL_VIOLIN_mean_R2_leave_target_out",
        pair_level=True,
    )

    plot_composite_violin(
        df=pair_summary,
        metric_col="mean_pearson",
        ylabel="Mean Pearson, all genes",
        title="Pair-level transfer performance: all-gene Pearson",
        save_prefix="PAIR_LEVEL_VIOLIN_mean_pearson_all_genes",
        pair_level=True,
    )

    plot_composite_violin(
        df=pair_summary,
        metric_col="mean_r2_uncentered",
        ylabel="Mean uncentered R², all genes",
        title="Pair-level transfer performance: all-gene R²",
        save_prefix="PAIR_LEVEL_VIOLIN_mean_R2_all_genes",
        pair_level=True,
    )

    if len(per_pert_df):
        plot_composite_violin(
            df=per_pert_df,
            metric_col="pearson_lto",
            ylabel="Pearson per perturbation, leave-target-out",
            title="Per-perturbation transfer: leave-target-out Pearson",
            save_prefix="PER_PERT_VIOLIN_pearson_leave_target_out",
            pair_level=False,
        )

        plot_composite_violin(
            df=per_pert_df,
            metric_col="r2_uncentered_lto",
            ylabel="Uncentered R² per perturbation, leave-target-out",
            title="Per-perturbation transfer: leave-target-out R²",
            save_prefix="PER_PERT_VIOLIN_R2_leave_target_out",
            pair_level=False,
        )

    print("\nDone.")


def rpe_first3_plus_far_from_rpe_atlas_streamed():
    global display  # preserve the cell's module-level display fallback
    RPE_DATASETS = [
        {
            "path": os.path.join(DATA_DIR, "ReplogleWeissman2022_rpe1.h5ad"),
            "name": "ReplogleWeissman2022_rpe1",
            "label": "Replogle RPE1",
        },
        {
            "path": os.path.join(DATA_DIR, "kaden25_rpe1_ctrl_10k_min100_greedy_4gb.h5ad"),
            "name": "kaden25_rpe1_ctrl_10k_min100_greedy_4gb",
            "label": "Kaden RPE1",
        },
    ]

    PERT_KEY_OVERRIDES = {
        # "ReplogleWeissman2022_rpe1": "perturbation",
        # "kaden25_rpe1_ctrl_10k_min100_greedy_4gb": "perturbation",
    }

    OUTDIR = Path(SUPPL_OUT) / "rpe_first3_plus_far_from_rpe_atlas_streamed"

    OUTDIR.mkdir(parents=True, exist_ok=True)

    PAIR_CACHE_DIR = OUTDIR / "pair_metric_cache"

    PAIR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    CELLXGENE_COV_CACHE_DIR = OUTDIR / "cellxgene_cov_source_cache"

    CELLXGENE_COV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    CELLXGENE_SAMPLE_DIR = OUTDIR / "cellxgene_sample_obs"

    CELLXGENE_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    MIN_PERT_CELLS = 100

    MIN_CONTROL_CELLS = 100

    MIN_RAW_MEAN_CONTROL = 0.01

    FORCE_INCLUDE_TARGET_GENES = True

    MAX_GENES_PER_RPE_DATASET = None

    N_CONTROL_FOR_SIGMA = 10000

    N_CONTROL_FOR_DX_MEAN = None

    N_PERT_FOR_DX_MEAN = None

    CENSUS_VERSION = "2025-11-08"

    ORGANISM = "Homo sapiens"

    N_RPE_ATLAS_SOURCES = 4

    N_CELLS_PER_RPE_ATLAS_SOURCE = 5000

    N_FAR_ATLAS_DATASETS = 8

    N_CELLS_PER_FAR_ATLAS_DATASET = 5000

    CELLXGENE_FETCH_CHUNK_CELLS = 1000

    RPE_TARGET_BLOCK_SIZE = 64

    RPE_CELL_TYPE_NAMES = [
        "retinal pigment epithelial cell",
        "retinal pigment epithelial cell of the eye",
        "retinal pigment epithelium cell",
    ]

    FAR_FROM_RPE_CELL_TYPE_NAMES = [
        "T cell",
        "B cell",
        "natural killer cell",
        "monocyte",
        "macrophage",
        "dendritic cell",
        "fibroblast",
        "endothelial cell",
        "smooth muscle cell",
        "stromal cell",
        "mesenchymal cell",
        "pericyte",
    ]

    BASE_OBS_FILTER_SUFFIX = (
        "is_primary_data == True "
        "and disease == 'normal' "
        "and suspension_type == 'cell'"
    )

    RPE_OR_EYE_KEYWORDS = [
        "rpe",
        "retinal pigment",
        "retina",
        "retinal",
        "eye",
        "ocular",
        "macula",
        "fovea",
        "choroid",
        "optic",
        "neural retina",
        "pigment epithelium",
    ]

    RPE_CLOSE_CELLTYPE_KEYWORDS = [
        "epithelial",
        "epithelium",
        "pigment",
        "photoreceptor",
        "rod",
        "cone",
        "bipolar neuron",
        "muller",
        "müller",
        "retinal ganglion",
        "neuron",
        "neural",
        "glial",
        "melanocyte",
    ]

    BATCH_SIZE = 64

    MIN_OVERLAP_GENES = 250

    MAX_PERTS_PER_RESPONSE = None

    EPS = 1e-12

    FORCE_REBUILD_RPE_CACHE = False

    FORCE_REQUERY_CELLXGENE_OBS = False

    FORCE_RESAMPLE_CELLXGENE_SOURCES = False

    FORCE_REBUILD_CELLXGENE_COV = False

    FORCE_RECOMPUTE_PAIR_METRICS = False

    SAVE_PER_PERTURBATION_TABLE = True

    DPI = 300

    VIOLIN_MAX_POINTS_PER_GROUP = 2500

    JITTER_MAX_POINTS_PER_GROUP = 1200

    RANDOM_SEED = 0

    rng = np.random.default_rng(RANDOM_SEED)

    TQDM_NCOLS = 110

    COMPOSITE_GROUP_ORDER = [
        "RPE Pert-seq within dataset",
        "RPE Pert-seq cross dataset",
        "CELLxGENE RPE atlas cells",
        "CELLxGENE far-from-RPE atlas cells",
    ]

    COMPOSITE_GROUP_LABELS = {
        "RPE Pert-seq within dataset": "RPE Pert-seq\nwithin dataset",
        "RPE Pert-seq cross dataset": "RPE Pert-seq\ncross dataset",
        "CELLxGENE RPE atlas cells": "CELLxGENE\nRPE cells",
        "CELLxGENE far-from-RPE atlas cells": "CELLxGENE\nfar-from-RPE cells",
    }

    COMPOSITE_GROUP_COLORS = {
        "RPE Pert-seq within dataset": "#8E77B5",
        "RPE Pert-seq cross dataset": "#4E79A7",
        "CELLxGENE RPE atlas cells": "#59A14F",
        "CELLxGENE far-from-RPE atlas cells": "#E3B778",
    }

    try:
        _real_display = display
    except NameError:
        _real_display = print

    def display(x):
        # Sanitize absolute paths out of DataFrame string columns so allocation paths are
        # never baked into the rendered cell output (rich display bypasses stdout routing).
        try:
            from src.logutil import sanitize as _sanitize_paths
            if isinstance(x, pd.DataFrame):
                x = x.copy()
                for _c in x.columns:
                    if x[_c].dtype == object:
                        x[_c] = x[_c].map(
                            lambda v: _sanitize_paths(v) if isinstance(v, str) else v
                        )
        except Exception:
            pass
        _real_display(x)

    def pair_cache_paths(response_dataset, source_dataset):
        stem = (
            f"response__{sanitize_filename(response_dataset)}__"
            f"source__{sanitize_filename(source_dataset)}"
        )
        return {
            "summary": PAIR_CACHE_DIR / f"{stem}__summary.json",
            "per_pert": PAIR_CACHE_DIR / f"{stem}__per_pert.csv",
        }

    CONTROL_LABELS = {
        "control",
        "ctrl",
        "non-targeting",
        "non_targeting",
        "nt",
        "ntc",
        "safe-targeting",
        "safe_targeting",
        "scramble",
        "scrambled",
        "negative control",
        "negative_control",
        "intergenic",
        "non-targeting control",
    }

    def is_control_label(x):
        s = str(x).strip().lower()
        if s in CONTROL_LABELS:
            return True
        return (
            "non-targeting" in s
            or "non_targeting" in s
            or s.startswith("control")
            or s.startswith("ctrl")
            or s in {"nan", "none", ""}
        )

    def base_gene_from_pert(pert):
        s = str(pert).strip()

        if is_control_label(s):
            return None

        s = re.sub(r"_g\d+$", "", s)
        s = re.sub(r"[-_]?guide\d+$", "", s, flags=re.I)
        s = re.sub(r"[-_]?sg\d+$", "", s, flags=re.I)
        s = re.sub(r"^(sgRNA|gRNA|sg|grna)[-_ ]+", "", s, flags=re.I)

        for sep in ["+", "|", ";", ","]:
            if sep in s:
                return None

        s = re.sub(r"[_-]sgRNA.*$", "", s, flags=re.I)
        s = re.sub(r"[_-]gRNA.*$", "", s, flags=re.I)
        s = re.sub(r"[_-]guide.*$", "", s, flags=re.I)

        return s

    def infer_perturbation_key(adata_obj, dataset_name):
        if dataset_name in PERT_KEY_OVERRIDES:
            key = PERT_KEY_OVERRIDES[dataset_name]
            if key not in adata_obj.obs.columns:
                raise ValueError(f"PERT_KEY_OVERRIDES[{dataset_name}]={key}, but key not in obs.")
            print(f"[pert key] using override for {dataset_name}: {key}")
            return key

        candidates = [
            "perturbation",
            "perturbation_name",
            "perturbation_id",
            "gene",
            "target_gene",
            "gene_target",
            "target",
            "guide_target",
            "sgRNA_target",
            "condition",
            "crispr",
        ]

        scored = []

        for k in candidates:
            if k not in adata_obj.obs.columns:
                continue

            vals = safe_str_col(adata_obj.obs, k)
            nunique = vals.nunique(dropna=True)
            n_controlish = vals.map(is_control_label).sum()

            if nunique < 2:
                continue

            score = 0
            score += min(nunique, 5000)
            score += 1000 if n_controlish > 0 else 0
            score += 500 if "pert" in k.lower() else 0
            score += 300 if "gene" in k.lower() or "target" in k.lower() else 0

            scored.append((score, k, nunique, int(n_controlish)))

        if len(scored) == 0:
            print("[obs columns]")
            print(list(adata_obj.obs.columns))
            raise ValueError(
                f"Could not infer perturbation key for {dataset_name}. "
                "Set PERT_KEY_OVERRIDES."
            )

        scored = sorted(scored, reverse=True)
        score, key, nunique, n_controlish = scored[0]

        print(
            f"[pert key] inferred for {dataset_name}: {key} "
            f"(nunique={nunique:,}, controlish_cells={n_controlish:,})"
        )

        return key

    def validate_raw_counts_matrix(X, name="X", integer_check=True):
        if issparse(X):
            data = X.data

            if data.size:
                if not np.isfinite(data).all():
                    bad = int((~np.isfinite(data)).sum())
                    raise ValueError(f"{name}: {bad:,} non-finite sparse values.")

                xmin = float(data.min())
                xmax = float(data.max())

                if xmin < 0:
                    neg = int((data < 0).sum())
                    raise ValueError(f"{name}: found {neg:,} negative raw-count values, min={xmin}.")

                if integer_check:
                    vals = data
                    if vals.size > 200_000:
                        vals = rng.choice(vals, size=200_000, replace=False)

                    frac_integerish = float(np.mean(np.abs(vals - np.round(vals)) < 1e-6))

                    print(
                        f"{name}: sparse raw check shape={X.shape}, "
                        f"nonzero_min={xmin:.4g}, nonzero_max={xmax:.4g}, "
                        f"integerish={frac_integerish:.4f}"
                    )

                    if frac_integerish < 0.98:
                        raise ValueError(
                            f"{name}: not integer-like enough for raw counts "
                            f"(integerish={frac_integerish:.4f})."
                        )

            return X

        X = np.asarray(X, dtype=np.float64)

        if not np.isfinite(X).all():
            bad = int((~np.isfinite(X)).sum())
            raise ValueError(f"{name}: {bad:,} non-finite values.")

        xmin = float(X.min())
        xmax = float(X.max())

        if xmin < 0:
            neg = int((X < 0).sum())
            raise ValueError(f"{name}: found {neg:,} negative raw-count values, min={xmin}.")

        if integer_check:
            vals = X[X > 0]
            if vals.size > 200_000:
                vals = rng.choice(vals, size=200_000, replace=False)

            frac_integerish = (
                float(np.mean(np.abs(vals - np.round(vals)) < 1e-6))
                if vals.size
                else 1.0
            )

            print(
                f"{name}: dense raw check shape={X.shape}, "
                f"min={xmin:.4g}, max={xmax:.4g}, integerish={frac_integerish:.4f}"
            )

            if frac_integerish < 0.98:
                raise ValueError(
                    f"{name}: not integer-like enough for raw counts "
                    f"(integerish={frac_integerish:.4f})."
                )

        return X

    def compute_cov_columns_from_raw_counts(
        X_raw,
        target_idx,
        name="raw X",
        target_block_size=64,
    ):
        """
        Computes covariance columns only:
            Cov(X, X_target) =
            (X.T @ X_target - n * mean_X * mean_target.T) / (n - 1)

        This avoids making a dense centered cell x gene matrix.
        """
        target_idx = np.asarray(target_idx, dtype=np.int64)

        if target_idx.size == 0:
            raise ValueError(f"{name}: empty target_idx.")

        if issparse(X_raw):
            X = validate_raw_counts_matrix(X_raw, name=name, integer_check=True)
            X = X.astype(np.float64).tocsr(copy=False)

            n, p = X.shape
            if n < 2:
                raise ValueError(f"{name}: need at least 2 cells.")

            print(
                f"{name}: sparse covariance columns using raw-count identity "
                f"n={n:,}, genes={p:,}, targets={len(target_idx):,}, "
                f"target_block={target_block_size}"
            )

            mu = np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
            XT = X.T.tocsr(copy=False)

            cov_cols = np.empty((p, len(target_idx)), dtype=np.float32)

            for s in tqdm(
                range(0, len(target_idx), target_block_size),
                desc=f"{name}: cov target blocks",
                ncols=TQDM_NCOLS,
            ):
                e = min(s + target_block_size, len(target_idx))
                tidx = target_idx[s:e]

                X_target = X[:, tidx].tocsc(copy=False)
                cross = XT @ X_target

                if issparse(cross):
                    cross = cross.toarray()
                else:
                    cross = np.asarray(cross)

                cross = cross.astype(np.float64, copy=False)
                mu_t = mu[tidx]

                block = (cross - float(n) * mu[:, None] * mu_t[None, :]) / float(n - 1)

                if not np.isfinite(block).all():
                    bad = int((~np.isfinite(block)).sum())
                    raise FloatingPointError(
                        f"{name}: covariance block {s}:{e} has {bad:,} non-finite values."
                    )

                cov_cols[:, s:e] = block.astype(np.float32)

                del X_target, cross, block
                gc.collect()

            return cov_cols

        X = validate_raw_counts_matrix(X_raw, name=name, integer_check=True)
        X = np.asarray(X, dtype=np.float64)

        n, p = X.shape
        if n < 2:
            raise ValueError(f"{name}: need at least 2 cells.")

        print(
            f"{name}: dense covariance columns using raw-count identity "
            f"n={n:,}, genes={p:,}, targets={len(target_idx):,}, "
            f"target_block={target_block_size}"
        )

        mu = X.mean(axis=0).astype(np.float64)
        XT = np.ascontiguousarray(X.T)

        cov_cols = np.empty((p, len(target_idx)), dtype=np.float32)

        for s in tqdm(
            range(0, len(target_idx), target_block_size),
            desc=f"{name}: cov target blocks",
            ncols=TQDM_NCOLS,
        ):
            e = min(s + target_block_size, len(target_idx))
            tidx = target_idx[s:e]

            X_target = np.ascontiguousarray(X[:, tidx])
            cross = XT @ X_target
            mu_t = mu[tidx]

            block = (cross - float(n) * mu[:, None] * mu_t[None, :]) / float(n - 1)

            if not np.isfinite(block).all():
                bad = int((~np.isfinite(block)).sum())
                raise FloatingPointError(
                    f"{name}: covariance block {s}:{e} has {bad:,} non-finite values."
                )

            cov_cols[:, s:e] = block.astype(np.float32)

            del X_target, cross, block
            gc.collect()

        return cov_cols

    def update_streamed_cross_sums_from_X(X_raw, target_idx, name="chunk"):
        """
        For one streamed chunk, returns:
          n_chunk
          sum_x:      shape p
          sum_t:      shape q
          cross_sum:  shape p x q = X.T @ X_target

        Used for CELLxGENE streaming covariance.
        """
        target_idx = np.asarray(target_idx, dtype=np.int64)

        if issparse(X_raw):
            X = validate_raw_counts_matrix(X_raw, name=name, integer_check=True)
            X = X.astype(np.float64).tocsr(copy=False)

            n, p = X.shape
            sum_x = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
            sum_t = sum_x[target_idx].copy()

            X_target = X[:, target_idx].tocsc(copy=False)
            cross = X.T @ X_target

            if issparse(cross):
                cross = cross.toarray()
            else:
                cross = np.asarray(cross)

            cross = cross.astype(np.float64, copy=False)

            return n, sum_x, sum_t, cross

        X = validate_raw_counts_matrix(X_raw, name=name, integer_check=True)
        X = np.asarray(X, dtype=np.float64)

        n, p = X.shape
        sum_x = X.sum(axis=0).astype(np.float64)
        sum_t = sum_x[target_idx].copy()
        cross = X.T @ X[:, target_idx]

        return n, sum_x, sum_t, cross.astype(np.float64, copy=False)

    def lr_metrics_batch(dx, basis, eps=EPS):
        dx = np.asarray(dx, dtype=np.float64)
        basis = np.asarray(basis, dtype=np.float64)

        dx_norm2 = np.einsum("ij,ij->i", dx, dx, optimize=True)
        basis_norm2 = np.einsum("ij,ij->i", basis, basis, optimize=True)
        numer = np.einsum("ij,ij->i", dx, basis, optimize=True)

        alpha = np.zeros(dx.shape[0], dtype=np.float64)
        good_basis = basis_norm2 > eps
        alpha[good_basis] = numer[good_basis] / basis_norm2[good_basis]

        pred = alpha[:, None] * basis

        resid = dx - pred
        resid_norm2 = np.einsum("ij,ij->i", resid, resid, optimize=True)

        r2 = np.full(dx.shape[0], np.nan, dtype=np.float64)
        good_dx = dx_norm2 > eps
        r2[good_dx] = 1.0 - resid_norm2[good_dx] / dx_norm2[good_dx]

        dx_c = dx - dx.mean(axis=1, keepdims=True)
        pred_c = pred - pred.mean(axis=1, keepdims=True)

        pear_num = np.einsum("ij,ij->i", dx_c, pred_c, optimize=True)
        pear_den = (
            np.sqrt(
                np.einsum("ij,ij->i", dx_c, dx_c, optimize=True)
                * np.einsum("ij,ij->i", pred_c, pred_c, optimize=True)
            )
            + eps
        )

        pearson = pear_num / pear_den
        pearson[~np.isfinite(pearson)] = np.nan

        return pearson, r2, alpha, dx_norm2

    def build_rpe_dataset_cache(spec):
        dataset_name = spec["name"]
        dataset_label = spec["label"]
        path = Path(spec["path"])

        if not path.exists():
            raise FileNotFoundError(f"Missing RPE h5ad: {path}")

        cache_path = OUTDIR / (
            f"rpe_pertseq_cache__{sanitize_filename(dataset_name)}__"
            f"minmean{str(MIN_RAW_MEAN_CONTROL).replace('.', 'p')}__"
            f"minpert{MIN_PERT_CELLS}__"
            f"nctrlsigma{N_CONTROL_FOR_SIGMA}__"
            f"maxgenes{MAX_GENES_PER_RPE_DATASET}.npz"
        )
        meta_path = cache_path.with_suffix(".json")

        if cache_path.exists() and meta_path.exists() and not FORCE_REBUILD_RPE_CACHE:
            print(f"[RPE cache] loading {cache_path}")
            z = np.load(cache_path, allow_pickle=True)

            genes = z["genes"].astype(str)
            unique_target_genes = z["unique_target_genes"].astype(str)

            return {
                "kind": "rpe_pertseq",
                "dataset": str(z["dataset"]),
                "label": str(z["label"]),
                "path": str(z["path"]),
                "pert_key": str(z["pert_key"]),
                "genes": genes,
                "gene_to_idx": {g: i for i, g in enumerate(genes.tolist())},
                "pert_names": z["pert_names"].astype(str),
                "target_genes": z["target_genes"].astype(str),
                "target_idx": z["target_idx"].astype(np.int64),
                "unique_target_genes": unique_target_genes,
                "unique_target_idx": z["unique_target_idx"].astype(np.int64),
                "target_gene_to_col": {g: i for i, g in enumerate(unique_target_genes.tolist())},
                "target_col_for_pert": z["target_col_for_pert"].astype(np.int64),
                "dx": z["dx"].astype(np.float32),
                "cov_cols": z["cov_cols"].astype(np.float32),
                "n_control_total": int(z["n_control_total"]),
                "n_control_sigma": int(z["n_control_sigma"]),
                "n_perts": int(z["dx"].shape[0]),
            }

        print("\n" + "=" * 120)
        print(f"[RPE load] {dataset_label}: {path}")
        print("=" * 120)

        adata = ad.read_h5ad(path)
        adata = infer_gene_names(adata)

        pert_key = infer_perturbation_key(adata, dataset_name)
        labels = safe_str_col(adata.obs, pert_key).values

        control_mask = np.asarray([is_control_label(x) for x in labels], dtype=bool)
        n_control = int(control_mask.sum())

        if n_control < MIN_CONTROL_CELLS:
            raise ValueError(
                f"{dataset_label}: too few controls detected using key={pert_key}: {n_control}. "
                "Set PERT_KEY_OVERRIDES or adjust CONTROL_LABELS."
            )

        Xraw_full = get_raw_X(adata, name=dataset_label)
        validate_raw_counts_matrix(Xraw_full, name=f"{dataset_label} full raw X", integer_check=True)

        raw_control_mean = sparse_or_dense_mean0(Xraw_full[control_mask])

        local_genes = np.asarray(adata.var_names.astype(str))
        gene_set = set(local_genes.tolist())
        gene_to_idx_full = {g: i for i, g in enumerate(local_genes.tolist())}

        vc = pd.Series(labels).value_counts()

        valid_perts_raw = [
            p for p, n in vc.items()
            if n >= MIN_PERT_CELLS and not is_control_label(p)
        ]

        pert_names = []
        target_genes = []

        for p in valid_perts_raw:
            g = base_gene_from_pert(p)
            if g is None:
                continue
            if g not in gene_set:
                continue
            pert_names.append(str(p))
            target_genes.append(str(g))

        print(f"[RPE {dataset_label}] control cells: {n_control:,}")
        print(f"[RPE {dataset_label}] valid perturbations raw >= {MIN_PERT_CELLS}: {len(valid_perts_raw):,}")
        print(f"[RPE {dataset_label}] valid single-gene perturbations mapped to genes: {len(pert_names):,}")

        expressed = set(local_genes[raw_control_mean >= MIN_RAW_MEAN_CONTROL])
        target_set = set(target_genes)

        if FORCE_INCLUDE_TARGET_GENES:
            keep_genes = sorted(list(expressed | target_set))
        else:
            keep_genes = sorted(list(expressed))

        keep_genes = [g for g in keep_genes if g in gene_set]

        if MAX_GENES_PER_RPE_DATASET is not None and len(keep_genes) > MAX_GENES_PER_RPE_DATASET:
            target_keep = sorted(list(target_set & set(keep_genes)))
            gene_mean = dict(zip(local_genes, raw_control_mean))
            non_target = [g for g in keep_genes if g not in set(target_keep)]
            non_target = sorted(non_target, key=lambda g: gene_mean.get(g, 0.0), reverse=True)
            keep_genes = sorted(
                set(
                    target_keep
                    + non_target[: max(0, MAX_GENES_PER_RPE_DATASET - len(target_keep))]
                )
            )

        if len(keep_genes) < MIN_OVERLAP_GENES:
            raise ValueError(f"{dataset_label}: too few kept genes: {len(keep_genes)}")

        print(f"[RPE {dataset_label}] kept genes: {len(keep_genes):,}")

        adata_sub = adata[:, keep_genes].copy()
        genes = np.asarray(adata_sub.var_names.astype(str))
        gene_to_idx = {g: i for i, g in enumerate(genes.tolist())}

        labels_sub = safe_str_col(adata_sub.obs, pert_key).values
        control_mask_sub = np.asarray([is_control_label(x) for x in labels_sub], dtype=bool)

        Xraw = get_raw_X(adata_sub, name=f"{dataset_label} subset")
        validate_raw_counts_matrix(Xraw, name=f"{dataset_label} subset raw X", integer_check=True)

        pert_names_final = []
        target_genes_final = []
        target_idx_final = []

        for p, g in zip(pert_names, target_genes):
            if g not in gene_to_idx:
                continue
            idx = np.flatnonzero(labels_sub == p)
            if len(idx) < MIN_PERT_CELLS:
                continue

            pert_names_final.append(str(p))
            target_genes_final.append(str(g))
            target_idx_final.append(gene_to_idx[g])

        if MAX_PERTS_PER_RESPONSE is not None and len(pert_names_final) > MAX_PERTS_PER_RESPONSE:
            keep = rng.choice(len(pert_names_final), size=MAX_PERTS_PER_RESPONSE, replace=False)
            keep = np.sort(keep)
            pert_names_final = [pert_names_final[i] for i in keep]
            target_genes_final = [target_genes_final[i] for i in keep]
            target_idx_final = [target_idx_final[i] for i in keep]

        if len(pert_names_final) == 0:
            raise ValueError(f"{dataset_label}: no valid perturbations after gene filtering.")

        target_idx_final = np.asarray(target_idx_final, dtype=np.int64)

        control_indices = np.flatnonzero(control_mask_sub)

        if N_CONTROL_FOR_DX_MEAN is not None and len(control_indices) > N_CONTROL_FOR_DX_MEAN:
            control_for_dx_idx = rng.choice(control_indices, size=N_CONTROL_FOR_DX_MEAN, replace=False)
        else:
            control_for_dx_idx = control_indices

        mu0 = sparse_or_dense_mean0(Xraw[control_for_dx_idx])

        dx_rows = []
        n_cells_pert = []

        print(f"[RPE {dataset_label}] computing dx per perturbation...")
        for p in tqdm(pert_names_final, ncols=TQDM_NCOLS):
            idx = np.flatnonzero(labels_sub == p)

            if N_PERT_FOR_DX_MEAN is not None and len(idx) > N_PERT_FOR_DX_MEAN:
                idx = rng.choice(idx, size=N_PERT_FOR_DX_MEAN, replace=False)

            mu1 = sparse_or_dense_mean0(Xraw[idx])
            dx = mu1 - mu0

            if not np.isfinite(dx).all():
                bad = int((~np.isfinite(dx)).sum())
                raise ValueError(f"{dataset_label}: dx for {p} has {bad:,} bad values.")

            dx_rows.append(dx.astype(np.float32))
            n_cells_pert.append(int(len(idx)))

        dx = np.vstack(dx_rows).astype(np.float32)

        unique_target_idx = np.asarray(sorted(set(target_idx_final.tolist())), dtype=np.int64)
        unique_target_genes = genes[unique_target_idx].astype(str)

        target_gene_to_col = {g: i for i, g in enumerate(unique_target_genes.tolist())}
        target_col_for_pert = np.asarray(
            [target_gene_to_col[g] for g in target_genes_final],
            dtype=np.int64,
        )

        if len(control_indices) > N_CONTROL_FOR_SIGMA:
            sigma_control_idx = rng.choice(control_indices, size=N_CONTROL_FOR_SIGMA, replace=False)
        else:
            sigma_control_idx = control_indices

        print(
            f"[RPE {dataset_label}] computing control covariance columns "
            f"from {len(sigma_control_idx):,} control cells, "
            f"{len(genes):,} genes, {len(unique_target_idx):,} target genes"
        )

        cov_cols = compute_cov_columns_from_raw_counts(
            Xraw[sigma_control_idx],
            unique_target_idx,
            name=f"{dataset_label} control raw counts",
            target_block_size=RPE_TARGET_BLOCK_SIZE,
        )

        np.savez_compressed(
            cache_path,
            dataset=np.asarray(dataset_name, dtype=object),
            label=np.asarray(dataset_label, dtype=object),
            path=np.asarray(str(path), dtype=object),
            pert_key=np.asarray(pert_key, dtype=object),
            genes=np.asarray(genes, dtype=object),
            pert_names=np.asarray(pert_names_final, dtype=object),
            target_genes=np.asarray(target_genes_final, dtype=object),
            target_idx=target_idx_final,
            unique_target_genes=np.asarray(unique_target_genes, dtype=object),
            unique_target_idx=unique_target_idx,
            target_col_for_pert=target_col_for_pert,
            dx=dx,
            cov_cols=cov_cols.astype(np.float32),
            n_cells_pert=np.asarray(n_cells_pert, dtype=np.int64),
            n_control_total=np.asarray(n_control, dtype=np.int64),
            n_control_sigma=np.asarray(len(sigma_control_idx), dtype=np.int64),
        )

        with open(meta_path, "w") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "label": dataset_label,
                    "path": str(path),
                    "pert_key": pert_key,
                    "n_cells": int(adata.n_obs),
                    "n_genes_original": int(adata.n_vars),
                    "n_genes_kept": int(len(genes)),
                    "n_control_total": int(n_control),
                    "n_control_sigma": int(len(sigma_control_idx)),
                    "n_perts": int(len(pert_names_final)),
                    "n_unique_target_genes": int(len(unique_target_idx)),
                    "MIN_RAW_MEAN_CONTROL": MIN_RAW_MEAN_CONTROL,
                    "MIN_PERT_CELLS": MIN_PERT_CELLS,
                    "MAX_GENES_PER_RPE_DATASET": MAX_GENES_PER_RPE_DATASET,
                },
                f,
                indent=2,
            )

        print(f"[RPE cache] saved {cache_path}")

        return {
            "kind": "rpe_pertseq",
            "dataset": dataset_name,
            "label": dataset_label,
            "path": str(path),
            "pert_key": pert_key,
            "genes": genes.astype(str),
            "gene_to_idx": {g: i for i, g in enumerate(genes.astype(str).tolist())},
            "pert_names": np.asarray(pert_names_final, dtype=str),
            "target_genes": np.asarray(target_genes_final, dtype=str),
            "target_idx": target_idx_final,
            "unique_target_genes": unique_target_genes.astype(str),
            "unique_target_idx": unique_target_idx,
            "target_gene_to_col": target_gene_to_col,
            "target_col_for_pert": target_col_for_pert,
            "dx": dx.astype(np.float32),
            "cov_cols": cov_cols.astype(np.float32),
            "n_control_total": int(n_control),
            "n_control_sigma": int(len(sigma_control_idx)),
            "n_perts": int(dx.shape[0]),
        }

    def load_all_rpe_datasets():
        datasets = []
        for spec in RPE_DATASETS:
            try:
                datasets.append(build_rpe_dataset_cache(spec))
            except FileNotFoundError as e:
                # skip RPE datasets whose h5ad is not staged locally (identical behaviour to
                # a shorter RPE_DATASETS list when all listed files are present)
                print(f"[RPE skip] {spec.get('label', spec.get('name'))}: {e}")
        if not datasets:
            raise FileNotFoundError(
                "No RPE datasets available on disk; expected at least one of: "
                + ", ".join(str(s["path"]) for s in RPE_DATASETS)
            )

        summary = pd.DataFrame([
            {
                "dataset": d["dataset"],
                "label": d["label"],
                "path": d["path"],
                "pert_key": d["pert_key"],
                "n_genes": len(d["genes"]),
                "n_perts": d["n_perts"],
                "n_unique_targets": len(d["unique_target_genes"]),
                "n_control_total": d["n_control_total"],
                "n_control_sigma": d["n_control_sigma"],
            }
            for d in datasets
        ])

        summary_path = OUTDIR / "loaded_rpe_pertseq_datasets.csv"
        summary.to_csv(summary_path, index=False)

        print(f"[saved] {summary_path}")
        display(summary)

        return datasets

    def get_census_dataset_table():
        cache_path = OUTDIR / f"census_datasets__{CENSUS_VERSION}.parquet"

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
            datasets = census["census_info"]["datasets"].read().concat().to_pandas()

        datasets["dataset_id"] = safe_str_col(datasets, "dataset_id")
        datasets.to_parquet(cache_path, index=False)

        return datasets

    def add_dataset_titles_to_obs(obs):
        datasets = get_census_dataset_table().copy()

        title_col = None
        for c in ["dataset_title", "title", "collection_name", "dataset_name"]:
            if c in datasets.columns:
                title_col = c
                break

        obs = obs.copy()
        obs["dataset_id"] = safe_str_col(obs, "dataset_id")

        if title_col is None or "dataset_id" not in datasets.columns:
            obs["dataset_title"] = obs["dataset_id"].astype(str)
            return obs

        ds = datasets[["dataset_id", title_col]].copy()
        ds["dataset_id"] = safe_str_col(ds, "dataset_id")
        ds[title_col] = safe_str_col(ds, title_col)
        ds = ds.drop_duplicates("dataset_id")
        ds = ds.rename(columns={title_col: "dataset_title"})

        obs = obs.merge(ds, on="dataset_id", how="left")

        title = safe_str_col(obs, "dataset_title")
        fallback = safe_str_col(obs, "dataset_id")
        obs["dataset_title"] = np.where(title.values == "", fallback.values, title.values)

        return obs

    def is_rpe_or_eye_related_obs(obs):
        cell_type = safe_str_col(obs, "cell_type").str.lower()
        tissue = safe_str_col(obs, "tissue").str.lower()
        tissue_general = safe_str_col(obs, "tissue_general").str.lower()
        dataset_title = safe_str_col(obs, "dataset_title").str.lower()

        if "query_cell_type" in obs.columns:
            query_cell_type = safe_str_col(obs, "query_cell_type").str.lower()
        else:
            query_cell_type = pd.Series("", index=obs.index, dtype=object)

        text = (
            cell_type
            + " "
            + tissue
            + " "
            + tissue_general
            + " "
            + dataset_title
            + " "
            + query_cell_type
        )

        return text.apply(lambda s: contains_any_keyword(s, RPE_OR_EYE_KEYWORDS)).astype(bool).values

    def is_close_to_rpe_celltype_obs(obs):
        cell_type = safe_str_col(obs, "cell_type").str.lower()

        if "query_cell_type" in obs.columns:
            query_cell_type = safe_str_col(obs, "query_cell_type").str.lower()
        else:
            query_cell_type = pd.Series("", index=obs.index, dtype=object)

        text = cell_type + " " + query_cell_type
        return text.apply(lambda s: contains_any_keyword(s, RPE_CLOSE_CELLTYPE_KEYWORDS)).astype(bool).values

    def is_far_from_rpe_obs(obs):
        eye_like = is_rpe_or_eye_related_obs(obs)
        celltype_close = is_close_to_rpe_celltype_obs(obs)
        return (~eye_like) & (~celltype_close)

    def query_cellxgene_obs_for_cell_types(cell_type_names, cache_prefix):
        cache_path = OUTDIR / (
            f"cellxgene_obs__{sanitize_filename(cache_prefix)}__"
            f"{CENSUS_VERSION}.parquet"
        )

        if cache_path.exists() and not FORCE_REQUERY_CELLXGENE_OBS:
            print(f"[CELLxGENE obs] loading cached {cache_path}")
            obs = pd.read_parquet(cache_path)
            obs["dataset_id"] = safe_str_col(obs, "dataset_id")
            obs["dataset_title"] = safe_str_col(obs, "dataset_title")
            obs["cell_type"] = safe_str_col(obs, "cell_type")
            obs["tissue"] = safe_str_col(obs, "tissue")
            obs["tissue_general"] = safe_str_col(obs, "tissue_general")
            return obs

        all_obs = []

        for ct in cell_type_names:
            vf = (
                f"cell_type == '{ct}' "
                f"and {BASE_OBS_FILTER_SUFFIX}"
            )

            print(f"[CELLxGENE obs] querying: {vf}")

            try:
                with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
                    obs = cellxgene_census.get_obs(
                        census,
                        ORGANISM,
                        value_filter=vf,
                        column_names=[
                            "soma_joinid",
                            "dataset_id",
                            "cell_type",
                            "tissue",
                            "tissue_general",
                            "disease",
                            "assay",
                            "donor_id",
                            "suspension_type",
                            "is_primary_data",
                        ],
                    )

                if len(obs):
                    obs["query_cell_type"] = ct
                    all_obs.append(obs)
                    print(f"  -> {len(obs):,} cells")
                else:
                    print("  -> 0 cells")

            except Exception as e:
                print(f"  -> query failed for {ct}: {repr(e)}")

        if len(all_obs) == 0:
            raise ValueError(
                f"No CELLxGENE obs found for {cache_prefix}. "
                f"Tried cell types: {cell_type_names}"
            )

        obs = pd.concat(all_obs, ignore_index=True)
        obs = obs.drop_duplicates("soma_joinid").reset_index(drop=True)
        obs = add_dataset_titles_to_obs(obs)

        obs["dataset_id"] = safe_str_col(obs, "dataset_id")
        obs["dataset_title"] = safe_str_col(obs, "dataset_title")
        obs["cell_type"] = safe_str_col(obs, "cell_type")
        obs["tissue"] = safe_str_col(obs, "tissue")
        obs["tissue_general"] = safe_str_col(obs, "tissue_general")
        obs["assay"] = safe_str_col(obs, "assay")
        obs["donor_id"] = safe_str_col(obs, "donor_id")

        obs.to_parquet(cache_path, index=False)

        print(f"[CELLxGENE obs] saved {cache_path}")
        print(f"[CELLxGENE obs] cells={len(obs):,}, datasets={obs['dataset_id'].nunique():,}")
        print("[CELLxGENE obs] top cell types:")
        print(obs["cell_type"].value_counts().head(20))

        return obs

    def summarize_obs_by_dataset(obs):
        obs = obs.copy()

        obs["dataset_id"] = safe_str_col(obs, "dataset_id")
        obs["dataset_title"] = safe_str_col(obs, "dataset_title")
        obs["cell_type"] = safe_str_col(obs, "cell_type")
        obs["tissue"] = safe_str_col(obs, "tissue")
        obs["tissue_general"] = safe_str_col(obs, "tissue_general")
        obs["assay"] = safe_str_col(obs, "assay")
        obs["donor_id"] = safe_str_col(obs, "donor_id")

        rows = []

        for dsid, sub in obs.groupby("dataset_id", sort=False):
            donor_vals = sub["donor_id"].astype(str)
            donor_vals = donor_vals[(donor_vals != "") & (donor_vals.str.lower() != "nan")]

            rows.append({
                "dataset_id": str(dsid),
                "dataset_title": mode_or_empty(sub["dataset_title"]),
                "n_cells": int(len(sub)),
                "cell_type_top": mode_or_empty(sub["cell_type"]),
                "cell_types_seen": unique_join_limited(sub["cell_type"], max_items=8),
                "tissue_top": mode_or_empty(sub["tissue"]),
                "tissue_general_top": mode_or_empty(sub["tissue_general"]),
                "tissues_seen": unique_join_limited(sub["tissue"], max_items=8),
                "tissue_generals_seen": unique_join_limited(sub["tissue_general"], max_items=8),
                "assays_seen": unique_join_limited(sub["assay"], max_items=5),
                "n_donors": int(donor_vals.nunique(dropna=True)),
            })

        return pd.DataFrame(rows)

    def sample_source_from_dataset(obs, dataset_id, n, seed):
        sub = obs[obs["dataset_id"].astype(str) == str(dataset_id)].copy()
        if len(sub) < n:
            raise ValueError(f"dataset {dataset_id} has only {len(sub)} cells, need {n}")
        return sub.sample(n=n, replace=False, random_state=seed).copy()

    def make_short_cellxgene_label(rank, group, row):
        cell_type = str(row.get("cell_type_top", "")).replace("_", " ")
        tissue = str(row.get("tissue_general_top", row.get("tissue_top", ""))).replace("_", " ")

        if group == "CELLxGENE RPE atlas cells":
            prefix = f"RPE atlas{rank}"
        else:
            prefix = f"Far atlas{rank}"

        if cell_type and tissue and cell_type.lower() not in {"nan", "none", "<na>"}:
            return f"{prefix}: {cell_type} | {tissue}"

        if tissue and tissue.lower() not in {"nan", "none", "<na>"}:
            return f"{prefix}: {tissue}"

        return prefix

    def choose_cellxgene_sources():
        """
        Returns CELLxGENE source specs:
          - several RPE atlas sources, all plotted under column 3
          - one pooled far-from-RPE atlas source, plotted under column 4
        """
        chosen_path = OUTDIR / (
            f"chosen_cellxgene_sources__rpe{N_RPE_ATLAS_SOURCES}x{N_CELLS_PER_RPE_ATLAS_SOURCE}__"
            f"far{N_FAR_ATLAS_DATASETS}x{N_CELLS_PER_FAR_ATLAS_DATASET}__"
            f"seed{RANDOM_SEED}.csv"
        )

        if chosen_path.exists() and not FORCE_RESAMPLE_CELLXGENE_SOURCES:
            chosen_df = pd.read_csv(chosen_path)
            print(f"[CELLxGENE sources] loading chosen sources: {chosen_path}")

            loaded_specs = []

            for _, row in chosen_df.iterrows():
                sample_obs = pd.read_csv(row["sample_obs_path"])
                loaded_specs.append({
                    "kind": "cellxgene",
                    "source_dataset": str(row["source_dataset"]),
                    "source_label": str(row["source_label"]),
                    "source_type": str(row["source_type"]),
                    "composite_group": str(row["composite_group"]),
                    "dataset_id": str(row["dataset_id"]),
                    "sample_obs": sample_obs,
                })

            display(chosen_df)
            return loaded_specs

        specs = []
        rows = []

        # -----------------------------
        # RPE atlas cells: column 3
        # -----------------------------
        rpe_obs = query_cellxgene_obs_for_cell_types(
            RPE_CELL_TYPE_NAMES,
            cache_prefix="RPE_atlas_cells",
        )

        rpe_mask = is_rpe_or_eye_related_obs(rpe_obs)
        n_before = len(rpe_obs)
        rpe_obs = rpe_obs[rpe_mask].copy()

        print(f"[RPE atlas] kept RPE/eye-related rows: {len(rpe_obs):,} / {n_before:,}")

        rpe_summary = summarize_obs_by_dataset(rpe_obs)
        rpe_summary_path = OUTDIR / "cellxgene_RPE_atlas_dataset_summary.csv"
        rpe_summary.to_csv(rpe_summary_path, index=False)
        print(f"[saved] {rpe_summary_path}")

        rpe_eligible = rpe_summary[rpe_summary["n_cells"] >= N_CELLS_PER_RPE_ATLAS_SOURCE].copy()

        if len(rpe_eligible) == 0:
            raise ValueError(
                "No eligible CELLxGENE RPE atlas datasets. "
                "Try lowering N_CELLS_PER_RPE_ATLAS_SOURCE or editing RPE_CELL_TYPE_NAMES."
            )

        n_rpe = min(N_RPE_ATLAS_SOURCES, len(rpe_eligible))
        rpe_ids = rng.choice(rpe_eligible["dataset_id"].values, size=n_rpe, replace=False).tolist()

        for rank, dsid in enumerate(rpe_ids, start=1):
            r = rpe_eligible[rpe_eligible["dataset_id"] == dsid].iloc[0].to_dict()
            sample_obs = sample_source_from_dataset(
                rpe_obs,
                dsid,
                N_CELLS_PER_RPE_ATLAS_SOURCE,
                RANDOM_SEED + rank,
            )

            source_dataset = f"CELLxGENE_RPE_{sanitize_filename(dsid, maxlen=80)}"
            source_label = make_short_cellxgene_label(rank, "CELLxGENE RPE atlas cells", r)
            source_type = "CELLxGENE RPE atlas raw covariance"
            composite_group = "CELLxGENE RPE atlas cells"

            sample_path = CELLXGENE_SAMPLE_DIR / f"sample_obs__{sanitize_filename(source_dataset)}.csv"
            sample_obs.to_csv(sample_path, index=False)

            specs.append({
                "kind": "cellxgene",
                "source_dataset": source_dataset,
                "source_label": source_label,
                "source_type": source_type,
                "composite_group": composite_group,
                "dataset_id": str(dsid),
                "sample_obs": sample_obs,
            })

            rows.append({
                "source_dataset": source_dataset,
                "source_label": source_label,
                "source_type": source_type,
                "composite_group": composite_group,
                "dataset_id": str(dsid),
                "dataset_title": r["dataset_title"],
                "cell_type_top": r["cell_type_top"],
                "tissue_top": r["tissue_top"],
                "tissue_general_top": r["tissue_general_top"],
                "tissues_seen": r.get("tissues_seen", ""),
                "tissue_generals_seen": r.get("tissue_generals_seen", ""),
                "n_cells_available": int(r["n_cells"]),
                "n_cells_sampled": int(len(sample_obs)),
                "sample_obs_path": str(sample_path),
            })

        # -----------------------------
        # Far-from-RPE atlas cells: one pooled source, column 4
        # -----------------------------
        far_obs = query_cellxgene_obs_for_cell_types(
            FAR_FROM_RPE_CELL_TYPE_NAMES,
            cache_prefix="far_from_RPE_atlas_cells_raw_query",
        )

        n_before = len(far_obs)
        far_mask = is_far_from_rpe_obs(far_obs)
        far_obs = far_obs[far_mask].copy()

        print(
            f"[far atlas] kept far-from-RPE rows: "
            f"{len(far_obs):,} / {n_before:,}"
        )

        if len(far_obs) == 0:
            raise ValueError(
                "After removing RPE/retina/eye/epithelial/pigment/neural-like rows, "
                "no far-from-RPE atlas cells remain. "
                "Add more FAR_FROM_RPE_CELL_TYPE_NAMES or relax RPE_CLOSE_CELLTYPE_KEYWORDS."
            )

        far_summary = summarize_obs_by_dataset(far_obs)
        far_summary_path = OUTDIR / "cellxgene_far_from_RPE_dataset_summary.csv"
        far_summary.to_csv(far_summary_path, index=False)
        print(f"[saved] {far_summary_path}")

        print("[far atlas] top cell types after filtering:")
        print(far_obs["cell_type"].value_counts().head(25))

        print("[far atlas] top tissues after filtering:")
        print(far_obs["tissue_general"].value_counts().head(25))

        far_eligible = far_summary[far_summary["n_cells"] >= N_CELLS_PER_FAR_ATLAS_DATASET].copy()

        if len(far_eligible) == 0:
            raise ValueError(
                "No eligible far-from-RPE atlas datasets. "
                "Try lowering N_CELLS_PER_FAR_ATLAS_DATASET or adding more cell types."
            )

        # Prefer tissue/cell-type diversity.
        far_eligible = far_eligible.sort_values(
            ["tissue_general_top", "cell_type_top", "n_cells"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

        n_far = min(N_FAR_ATLAS_DATASETS, len(far_eligible))
        far_ids = rng.choice(far_eligible["dataset_id"].values, size=n_far, replace=False).tolist()

        pooled_far_obs = []

        far_source_rows = []

        for rank, dsid in enumerate(far_ids, start=1):
            r = far_eligible[far_eligible["dataset_id"] == dsid].iloc[0].to_dict()
            sample_obs = sample_source_from_dataset(
                far_obs,
                dsid,
                N_CELLS_PER_FAR_ATLAS_DATASET,
                RANDOM_SEED + 1000 + rank,
            )

            sampled_far_mask = is_far_from_rpe_obs(sample_obs)
            if not sampled_far_mask.all():
                bad = int((~sampled_far_mask).sum())
                raise RuntimeError(f"Far-from-RPE sample {dsid} still contains {bad} close/RPE-like rows.")

            pooled_far_obs.append(sample_obs)

            far_source_rows.append({
                "rank": rank,
                "dataset_id": str(dsid),
                "dataset_title": r["dataset_title"],
                "cell_type_top": r["cell_type_top"],
                "tissue_top": r["tissue_top"],
                "tissue_general_top": r["tissue_general_top"],
                "n_cells_available": int(r["n_cells"]),
                "n_cells_sampled": int(len(sample_obs)),
            })

        pooled_far_obs = pd.concat(pooled_far_obs, ignore_index=True)

        pooled_far_mask = is_far_from_rpe_obs(pooled_far_obs)
        if not pooled_far_mask.all():
            bad = int((~pooled_far_mask).sum())
            raise RuntimeError(f"Pooled far-from-RPE source contains {bad} close/RPE-like rows.")

        source_dataset = "CELLxGENE_POOLED_FAR_FROM_RPE_ATLAS"
        source_label = "Pooled CELLxGENE far-from-RPE atlas"
        source_type = "CELLxGENE pooled far-from-RPE raw covariance"
        composite_group = "CELLxGENE far-from-RPE atlas cells"

        sample_path = CELLXGENE_SAMPLE_DIR / "sample_obs__CELLxGENE_POOLED_FAR_FROM_RPE_ATLAS.csv"
        pooled_far_obs.to_csv(sample_path, index=False)

        specs.append({
            "kind": "cellxgene",
            "source_dataset": source_dataset,
            "source_label": source_label,
            "source_type": source_type,
            "composite_group": composite_group,
            "dataset_id": "pooled_far_from_rpe_atlas",
            "sample_obs": pooled_far_obs,
        })

        rows.append({
            "source_dataset": source_dataset,
            "source_label": source_label,
            "source_type": source_type,
            "composite_group": composite_group,
            "dataset_id": "pooled_far_from_rpe_atlas",
            "dataset_title": "pooled selected far-from-RPE atlas datasets",
            "cell_type_top": "mixed far-from-RPE cells",
            "tissue_top": "mixed far-from-RPE tissues",
            "tissue_general_top": "mixed far-from-RPE tissues",
            "tissues_seen": unique_join_limited(pooled_far_obs["tissue"], max_items=12),
            "tissue_generals_seen": unique_join_limited(pooled_far_obs["tissue_general"], max_items=12),
            "n_cells_available": int(len(pooled_far_obs)),
            "n_cells_sampled": int(len(pooled_far_obs)),
            "sample_obs_path": str(sample_path),
        })

        far_source_table = pd.DataFrame(far_source_rows)
        far_source_table_path = OUTDIR / "cellxgene_far_from_RPE_selected_component_sources.csv"
        far_source_table.to_csv(far_source_table_path, index=False)
        print(f"[saved] {far_source_table_path}")
        display(far_source_table)

        chosen_df = pd.DataFrame(rows)
        chosen_df.to_csv(chosen_path, index=False)

        print(f"[saved] {chosen_path}")
        display(chosen_df)

        print("\n[CELLxGENE source counts by group]")
        print(chosen_df["composite_group"].value_counts())

        return specs

    def get_census_gene_table():
        cache_path = OUTDIR / f"census_human_var_feature_table__{CENSUS_VERSION}.parquet"

        if cache_path.exists():
            var = pd.read_parquet(cache_path)
            var["feature_name"] = safe_str_col(var, "feature_name").str.strip()
            return var

        print("[Census var] fetching human gene table")

        with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
            var = cellxgene_census.get_var(
                census,
                ORGANISM,
                column_names=["soma_joinid", "feature_id", "feature_name"],
            )

        var["feature_name"] = safe_str_col(var, "feature_name").str.strip()
        var = var.drop_duplicates("feature_name", keep="first").reset_index(drop=True)
        var.to_parquet(cache_path, index=False)

        return var

    def get_census_gene_map():
        var = get_census_gene_table()
        var["feature_name"] = safe_str_col(var, "feature_name").str.strip()
        var = var.drop_duplicates("feature_name", keep="first")
        return dict(zip(var["feature_name"].astype(str), var["soma_joinid"].astype(np.int64)))

    def build_union_gene_target_sets_for_cellxgene(datasets):
        census_gene_map = get_census_gene_map()
        census_genes = set(census_gene_map.keys())

        union_genes = sorted(set().union(*[
            set(d["genes"].astype(str).tolist())
            for d in datasets
        ]))
        union_genes = [g for g in union_genes if g in census_genes]

        union_targets = sorted(set().union(*[
            set(d["target_genes"].astype(str).tolist())
            for d in datasets
        ]))
        union_targets = [g for g in union_targets if g in set(union_genes)]

        if len(union_genes) < MIN_OVERLAP_GENES:
            raise ValueError(f"CELLxGENE union genes too small: {len(union_genes)}")

        if len(union_targets) == 0:
            raise ValueError("No union target genes available in Census gene space.")

        print("\n[CELLxGENE union gene space]")
        print(f"union genes in Census: {len(union_genes):,}")
        print(f"union target genes:    {len(union_targets):,}")

        return union_genes, union_targets

    def streamed_cellxgene_cov_columns(sample_obs, union_genes, union_targets, source_label):
        """
        Streamed raw-count covariance from CELLxGENE.

        Accumulates:
            sum_x      = sum over cells of X
            sum_t      = sum over cells of target genes
            cross_sum  = X.T @ X_target over all cells

        Then:
            cov_cols = (cross_sum - sum_x[:, None] * sum_t[None, :] / n) / (n - 1)

        This avoids storing the full sampled atlas matrix in memory.
        """
        census_gene_map = get_census_gene_map()

        union_gene_to_idx = {g: i for i, g in enumerate(union_genes)}
        target_idx_local = np.asarray([union_gene_to_idx[g] for g in union_targets], dtype=np.int64)

        obs_coords_all = sample_obs["soma_joinid"].astype(np.int64).values
        var_coords = np.asarray([census_gene_map[g] for g in union_genes], dtype=np.int64)

        p = len(union_genes)
        q = len(union_targets)

        n_total = 0
        sum_x_total = np.zeros(p, dtype=np.float64)
        sum_t_total = np.zeros(q, dtype=np.float64)
        cross_total = np.zeros((p, q), dtype=np.float64)

        print(
            f"[CELLxGENE streamed cov] {source_label}: "
            f"cells={len(obs_coords_all):,}, genes={p:,}, targets={q:,}, "
            f"chunk_cells={CELLXGENE_FETCH_CHUNK_CELLS:,}"
        )

        with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
            for start in tqdm(
                range(0, len(obs_coords_all), CELLXGENE_FETCH_CHUNK_CELLS),
                desc=f"{source_label}: CELLxGENE chunks",
                ncols=TQDM_NCOLS,
            ):
                end = min(start + CELLXGENE_FETCH_CHUNK_CELLS, len(obs_coords_all))
                obs_coords = obs_coords_all[start:end]

                adata_cxg = cellxgene_census.get_anndata(
                    census=census,
                    organism=ORGANISM,
                    obs_coords=obs_coords,
                    var_coords=var_coords,
                    X_name="raw",
                    obs_column_names=[
                        "soma_joinid",
                        "dataset_id",
                        "cell_type",
                        "tissue",
                        "tissue_general",
                        "disease",
                        "assay",
                        "donor_id",
                        "suspension_type",
                        "is_primary_data",
                    ],
                    var_column_names=[
                        "soma_joinid",
                        "feature_id",
                        "feature_name",
                    ],
                )

                if "feature_name" in adata_cxg.var.columns:
                    adata_cxg.var_names = safe_str_col(adata_cxg.var, "feature_name").values

                adata_cxg.var_names = adata_cxg.var_names.astype(str)
                adata_cxg.var_names_make_unique()

                missing = [g for g in union_genes if g not in set(adata_cxg.var_names)]
                if len(missing):
                    raise ValueError(
                        f"CELLxGENE chunk missing {len(missing)} genes. "
                        f"Example: {missing[:10]}"
                    )

                adata_cxg = adata_cxg[:, union_genes].copy()

                n_chunk, sum_x, sum_t, cross = update_streamed_cross_sums_from_X(
                    adata_cxg.X,
                    target_idx_local,
                    name=f"CELLxGENE raw chunk {source_label} {start}:{end}",
                )

                n_total += int(n_chunk)
                sum_x_total += sum_x
                sum_t_total += sum_t
                cross_total += cross

                del adata_cxg, sum_x, sum_t, cross
                gc.collect()

        if n_total < 2:
            raise ValueError(f"{source_label}: need at least 2 streamed cells, got {n_total}.")

        cov_cols = (
            cross_total
            - (sum_x_total[:, None] * sum_t_total[None, :] / float(n_total))
        ) / float(n_total - 1)

        if not np.isfinite(cov_cols).all():
            bad = int((~np.isfinite(cov_cols)).sum())
            raise FloatingPointError(
                f"{source_label}: streamed covariance has {bad:,} non-finite values."
            )

        return cov_cols.astype(np.float32), int(n_total)

    def compute_cellxgene_cov_source_streamed(source_spec, union_genes, union_targets):
        sample_obs = source_spec["sample_obs"].copy()
        source_dataset = source_spec["source_dataset"]
        source_label = source_spec["source_label"]
        source_type = source_spec["source_type"]
        composite_group = source_spec["composite_group"]

        gh = gene_hash(union_genes)
        th = gene_hash(union_targets)
        oh = hashlib.md5(
            "\n".join(sample_obs["soma_joinid"].astype(str).tolist()).encode()
        ).hexdigest()[:12]

        cache_path = CELLXGENE_COV_CACHE_DIR / (
            f"streamed_cov_source__{sanitize_filename(source_dataset)}__"
            f"cells{len(sample_obs)}__genes{len(union_genes)}__targets{len(union_targets)}__"
            f"chunk{CELLXGENE_FETCH_CHUNK_CELLS}__"
            f"{gh}_{th}_{oh}__{CENSUS_VERSION}.npz"
        )

        if cache_path.exists() and not FORCE_REBUILD_CELLXGENE_COV:
            z = np.load(cache_path, allow_pickle=True)
            cov_cols = z["cov_cols"]

            if np.isfinite(cov_cols).all():
                print(f"[CELLxGENE streamed cov] loading cached valid {cache_path}")

                union_genes_loaded = z["union_genes"].astype(str)
                union_targets_loaded = z["union_targets"].astype(str)

                return {
                    "kind": "cellxgene_cov",
                    "source_dataset": source_dataset,
                    "source_label": source_label,
                    "source_type": source_type,
                    "composite_group": composite_group,
                    "cov_cols": cov_cols.astype(np.float32),
                    "union_genes": union_genes_loaded,
                    "union_gene_to_idx": {
                        g: i for i, g in enumerate(union_genes_loaded.tolist())
                    },
                    "union_targets": union_targets_loaded,
                    "target_gene_to_col": {
                        g: i for i, g in enumerate(union_targets_loaded.tolist())
                    },
                    "n_cells": int(z["n_cells"]),
                    "cache_path": str(cache_path),
                }

            bad = int((~np.isfinite(cov_cols)).sum())
            print(f"[CELLxGENE streamed cov] bad cache has {bad:,} non-finite values; deleting {cache_path}")
            try:
                cache_path.unlink()
            except Exception as e:
                print(f"[CELLxGENE streamed cov] could not delete bad cache: {repr(e)}")

        print("\n" + "=" * 120)
        print("[CELLxGENE streamed cov] computing raw covariance source")
        print(f"group:   {composite_group}")
        print(f"source:  {source_label}")
        print(f"type:    {source_type}")
        print(f"cells:   {len(sample_obs):,}")
        print(f"genes:   {len(union_genes):,}")
        print(f"targets: {len(union_targets):,}")
        print("=" * 120)

        cov_cols, n_cells_used = streamed_cellxgene_cov_columns(
            sample_obs=sample_obs,
            union_genes=union_genes,
            union_targets=union_targets,
            source_label=source_label,
        )

        np.savez_compressed(
            cache_path,
            cov_cols=cov_cols.astype(np.float32),
            union_genes=np.asarray(union_genes, dtype=object),
            union_targets=np.asarray(union_targets, dtype=object),
            n_cells=np.asarray(n_cells_used, dtype=np.int64),
        )

        print(f"[CELLxGENE streamed cov] saved {cache_path}")

        return {
            "kind": "cellxgene_cov",
            "source_dataset": source_dataset,
            "source_label": source_label,
            "source_type": source_type,
            "composite_group": composite_group,
            "cov_cols": cov_cols.astype(np.float32),
            "union_genes": np.asarray(union_genes, dtype=str),
            "union_gene_to_idx": {g: i for i, g in enumerate(union_genes)},
            "union_targets": np.asarray(union_targets, dtype=str),
            "target_gene_to_col": {g: i for i, g in enumerate(union_targets)},
            "n_cells": n_cells_used,
            "cache_path": str(cache_path),
        }

    def evaluate_response_vs_rpe_source(response, source):
        cache = pair_cache_paths(response["dataset"], source["dataset"])

        if (
            cache["summary"].exists()
            and (cache["per_pert"].exists() or not SAVE_PER_PERTURBATION_TABLE)
            and not FORCE_RECOMPUTE_PAIR_METRICS
        ):
            with open(cache["summary"], "r") as f:
                summary = json.load(f)
            pert_df = (
                pd.read_csv(cache["per_pert"])
                if cache["per_pert"].exists() and SAVE_PER_PERTURBATION_TABLE
                else None
            )
            return pert_df, summary

        composite_group = classify_rpe_pair(response, source)
        overlap_genes = np.intersect1d(response["genes"], source["genes"])

        if len(overlap_genes) < MIN_OVERLAP_GENES:
            summary = {
                "response_dataset": response["dataset"],
                "response_label": response["label"],
                "source_dataset": source["dataset"],
                "source_label": source["label"],
                "source_type": "RPE Pert-seq control Sigma",
                "composite_group": composite_group,
                "n_eval": 0,
                "n_overlap_genes": int(len(overlap_genes)),
                "mean_pearson": np.nan,
                "median_pearson": np.nan,
                "sem_pearson": np.nan,
                "mean_r2": np.nan,
                "median_r2": np.nan,
                "sem_r2": np.nan,
                "skip_reason": f"too few overlap genes: {len(overlap_genes)}",
            }
            with open(cache["summary"], "w") as f:
                json.dump(summary, f, indent=2, default=json_default)
            return None, summary

        resp_idx = np.asarray([response["gene_to_idx"][g] for g in overlap_genes], dtype=np.int64)
        src_idx = np.asarray([source["gene_to_idx"][g] for g in overlap_genes], dtype=np.int64)

        valid_mask = np.asarray(
            [tg in source["target_gene_to_col"] for tg in response["target_genes"]],
            dtype=bool,
        )
        valid_pos = np.flatnonzero(valid_mask)

        if len(valid_pos) == 0:
            summary = {
                "response_dataset": response["dataset"],
                "response_label": response["label"],
                "source_dataset": source["dataset"],
                "source_label": source["label"],
                "source_type": "RPE Pert-seq control Sigma",
                "composite_group": composite_group,
                "n_eval": 0,
                "n_overlap_genes": int(len(overlap_genes)),
                "mean_pearson": np.nan,
                "median_pearson": np.nan,
                "sem_pearson": np.nan,
                "mean_r2": np.nan,
                "median_r2": np.nan,
                "sem_r2": np.nan,
                "skip_reason": "no response targets present in source target columns",
            }
            with open(cache["summary"], "w") as f:
                json.dump(summary, f, indent=2, default=json_default)
            return None, summary

        pearson_all = []
        r2_all = []
        rows_out = []

        for start in range(0, len(valid_pos), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(valid_pos))
            pos = valid_pos[start:end]

            target_genes = response["target_genes"][pos].astype(str)
            source_col_idx = np.asarray(
                [source["target_gene_to_col"][g] for g in target_genes],
                dtype=np.int64,
            )

            dx = np.asarray(response["dx"][pos, :], dtype=np.float32)[:, resp_idx]
            basis = np.asarray(
                source["cov_cols"][np.ix_(src_idx, source_col_idx)],
                dtype=np.float32,
            ).T

            pearson, r2, alpha, dx_norm2 = lr_metrics_batch(dx, basis, eps=EPS)

            pearson_all.append(pearson)
            r2_all.append(r2)

            if SAVE_PER_PERTURBATION_TABLE:
                for j, ppos in enumerate(pos):
                    rows_out.append({
                        "response_dataset": response["dataset"],
                        "response_label": response["label"],
                        "source_dataset": source["dataset"],
                        "source_label": source["label"],
                        "source_type": "RPE Pert-seq control Sigma",
                        "composite_group": composite_group,
                        "perturbation": str(response["pert_names"][ppos]),
                        "target_gene": str(response["target_genes"][ppos]),
                        "n_overlap_genes": int(len(overlap_genes)),
                        "pearson": float(pearson[j]),
                        "r2_uncentered": float(r2[j]),
                        "alpha": float(alpha[j]),
                        "dx_norm2": float(dx_norm2[j]),
                    })

            del dx, basis
            gc.collect()

        pearson_all = np.concatenate(pearson_all) if pearson_all else np.asarray([])
        r2_all = np.concatenate(r2_all) if r2_all else np.asarray([])

        ps = summarize_metric(pearson_all)
        rs = summarize_metric(r2_all)

        summary = {
            "response_dataset": response["dataset"],
            "response_label": response["label"],
            "source_dataset": source["dataset"],
            "source_label": source["label"],
            "source_type": "RPE Pert-seq control Sigma",
            "composite_group": composite_group,
            "n_eval": int(ps["n"]),
            "n_overlap_genes": int(len(overlap_genes)),
            "mean_pearson": ps["mean"],
            "median_pearson": ps["median"],
            "sem_pearson": ps["sem"],
            "mean_r2": rs["mean"],
            "median_r2": rs["median"],
            "sem_r2": rs["sem"],
            "skip_reason": "",
        }

        pert_df = pd.DataFrame(rows_out) if SAVE_PER_PERTURBATION_TABLE else None

        with open(cache["summary"], "w") as f:
            json.dump(summary, f, indent=2, default=json_default)

        if SAVE_PER_PERTURBATION_TABLE and pert_df is not None:
            pert_df.to_csv(cache["per_pert"], index=False)

        return pert_df, summary

    def evaluate_response_vs_cellxgene_source(response, cxg_source):
        cache = pair_cache_paths(response["dataset"], cxg_source["source_dataset"])

        if (
            cache["summary"].exists()
            and (cache["per_pert"].exists() or not SAVE_PER_PERTURBATION_TABLE)
            and not FORCE_RECOMPUTE_PAIR_METRICS
        ):
            with open(cache["summary"], "r") as f:
                summary = json.load(f)
            pert_df = (
                pd.read_csv(cache["per_pert"])
                if cache["per_pert"].exists() and SAVE_PER_PERTURBATION_TABLE
                else None
            )
            return pert_df, summary

        union_gene_to_idx = cxg_source["union_gene_to_idx"]
        target_gene_to_col = cxg_source["target_gene_to_col"]
        cov_cols = cxg_source["cov_cols"]

        overlap_genes = [
            g for g in response["genes"].astype(str).tolist()
            if g in union_gene_to_idx
        ]

        if len(overlap_genes) < MIN_OVERLAP_GENES:
            summary = {
                "response_dataset": response["dataset"],
                "response_label": response["label"],
                "source_dataset": cxg_source["source_dataset"],
                "source_label": cxg_source["source_label"],
                "source_type": cxg_source["source_type"],
                "composite_group": cxg_source["composite_group"],
                "n_eval": 0,
                "n_overlap_genes": int(len(overlap_genes)),
                "mean_pearson": np.nan,
                "median_pearson": np.nan,
                "sem_pearson": np.nan,
                "mean_r2": np.nan,
                "median_r2": np.nan,
                "sem_r2": np.nan,
                "skip_reason": f"too few response genes in CELLxGENE source: {len(overlap_genes)}",
            }
            with open(cache["summary"], "w") as f:
                json.dump(summary, f, indent=2, default=json_default)
            return None, summary

        resp_idx = np.asarray([response["gene_to_idx"][g] for g in overlap_genes], dtype=np.int64)
        cxg_row_idx = np.asarray([union_gene_to_idx[g] for g in overlap_genes], dtype=np.int64)

        valid_mask = np.asarray(
            [tg in target_gene_to_col for tg in response["target_genes"]],
            dtype=bool,
        )
        valid_pos = np.flatnonzero(valid_mask)

        if len(valid_pos) == 0:
            summary = {
                "response_dataset": response["dataset"],
                "response_label": response["label"],
                "source_dataset": cxg_source["source_dataset"],
                "source_label": cxg_source["source_label"],
                "source_type": cxg_source["source_type"],
                "composite_group": cxg_source["composite_group"],
                "n_eval": 0,
                "n_overlap_genes": int(len(overlap_genes)),
                "mean_pearson": np.nan,
                "median_pearson": np.nan,
                "sem_pearson": np.nan,
                "mean_r2": np.nan,
                "median_r2": np.nan,
                "sem_r2": np.nan,
                "skip_reason": "no response targets present in CELLxGENE target set",
            }
            with open(cache["summary"], "w") as f:
                json.dump(summary, f, indent=2, default=json_default)
            return None, summary

        pearson_all = []
        r2_all = []
        rows_out = []

        for start in range(0, len(valid_pos), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(valid_pos))
            pos = valid_pos[start:end]

            target_genes = response["target_genes"][pos].astype(str)
            col_idx = np.asarray([target_gene_to_col[g] for g in target_genes], dtype=np.int64)

            dx = np.asarray(response["dx"][pos, :], dtype=np.float32)[:, resp_idx]
            basis = np.asarray(
                cov_cols[np.ix_(cxg_row_idx, col_idx)],
                dtype=np.float32,
            ).T

            pearson, r2, alpha, dx_norm2 = lr_metrics_batch(dx, basis, eps=EPS)

            pearson_all.append(pearson)
            r2_all.append(r2)

            if SAVE_PER_PERTURBATION_TABLE:
                for j, ppos in enumerate(pos):
                    rows_out.append({
                        "response_dataset": response["dataset"],
                        "response_label": response["label"],
                        "source_dataset": cxg_source["source_dataset"],
                        "source_label": cxg_source["source_label"],
                        "source_type": cxg_source["source_type"],
                        "composite_group": cxg_source["composite_group"],
                        "perturbation": str(response["pert_names"][ppos]),
                        "target_gene": str(response["target_genes"][ppos]),
                        "n_overlap_genes": int(len(overlap_genes)),
                        "pearson": float(pearson[j]),
                        "r2_uncentered": float(r2[j]),
                        "alpha": float(alpha[j]),
                        "dx_norm2": float(dx_norm2[j]),
                    })

            del dx, basis
            gc.collect()

        pearson_all = np.concatenate(pearson_all) if pearson_all else np.asarray([])
        r2_all = np.concatenate(r2_all) if r2_all else np.asarray([])

        ps = summarize_metric(pearson_all)
        rs = summarize_metric(r2_all)

        summary = {
            "response_dataset": response["dataset"],
            "response_label": response["label"],
            "source_dataset": cxg_source["source_dataset"],
            "source_label": cxg_source["source_label"],
            "source_type": cxg_source["source_type"],
            "composite_group": cxg_source["composite_group"],
            "n_eval": int(ps["n"]),
            "n_overlap_genes": int(len(overlap_genes)),
            "mean_pearson": ps["mean"],
            "median_pearson": ps["median"],
            "sem_pearson": ps["sem"],
            "mean_r2": rs["mean"],
            "median_r2": rs["median"],
            "sem_r2": rs["sem"],
            "skip_reason": "",
        }

        pert_df = pd.DataFrame(rows_out) if SAVE_PER_PERTURBATION_TABLE else None

        with open(cache["summary"], "w") as f:
            json.dump(summary, f, indent=2, default=json_default)

        if SAVE_PER_PERTURBATION_TABLE and pert_df is not None:
            pert_df.to_csv(cache["per_pert"], index=False)

        return pert_df, summary

    def composite_summary_table(per_pert_df, metric_col):
        rows = []

        for group in COMPOSITE_GROUP_ORDER:
            vals = pd.to_numeric(
                per_pert_df.loc[per_pert_df["composite_group"] == group, metric_col],
                errors="coerce",
            ).values
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                rows.append({
                    "composite_group": group,
                    "n": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "sem": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                })
            else:
                rows.append({
                    "composite_group": group,
                    "n": int(len(vals)),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                })

        return pd.DataFrame(rows)

    def plot_composite_violin(
        per_pert_df,
        metric_col,
        ylabel,
        title,
        save_prefix,
    ):
        df = per_pert_df.copy()
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
        df = df[np.isfinite(df[metric_col].values)].copy()
        df = df[df["composite_group"].isin(COMPOSITE_GROUP_ORDER)].copy()

        values = []
        groups = []
        colors = []

        for group in COMPOSITE_GROUP_ORDER:
            vals = df.loc[df["composite_group"] == group, metric_col].values
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                print(f"[plot] skipping empty group: {group}")
                continue

            if len(vals) > VIOLIN_MAX_POINTS_PER_GROUP:
                vals_for_violin = rng.choice(vals, size=VIOLIN_MAX_POINTS_PER_GROUP, replace=False)
            else:
                vals_for_violin = vals

            values.append(vals_for_violin)
            groups.append(group)
            colors.append(COMPOSITE_GROUP_COLORS[group])

        if len(values) == 0:
            raise ValueError("No non-empty composite groups to plot.")

        summary = composite_summary_table(df, metric_col)
        summary_path = OUTDIR / f"{save_prefix}__summary.csv"
        summary.to_csv(summary_path, index=False)

        print(f"[saved] {summary_path}")
        display(summary)

        plt.figure(figsize=(12.5, 6.2))

        parts = plt.violinplot(
            values,
            positions=np.arange(1, len(values) + 1),
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.9,
        )

        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.45)
            body.set_linewidth(1.0)

        bp = plt.boxplot(
            values,
            positions=np.arange(1, len(values) + 1),
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            medianprops={"linewidth": 2.5, "color": "black"},
            boxprops={"linewidth": 1.0, "color": "black"},
            whiskerprops={"linewidth": 1.0, "color": "black"},
            capprops={"linewidth": 1.0, "color": "black"},
        )

        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_alpha(0.78)

        for i, group in enumerate(groups, start=1):
            vals = df.loc[df["composite_group"] == group, metric_col].values
            vals = vals[np.isfinite(vals)]

            if len(vals) > JITTER_MAX_POINTS_PER_GROUP:
                vals = rng.choice(vals, size=JITTER_MAX_POINTS_PER_GROUP, replace=False)

            jitter = rng.normal(loc=0.0, scale=0.055, size=len(vals))

            plt.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=8,
                alpha=0.14,
                color="black",
                linewidths=0,
                zorder=3,
            )

        plt.axhline(0, color="black", linewidth=1.2, alpha=0.8)

        labels = [COMPOSITE_GROUP_LABELS[g] for g in groups]

        plt.xticks(
            np.arange(1, len(values) + 1),
            labels,
            rotation=35,
            ha="right",
            fontsize=12,
        )
        plt.ylabel(ylabel, fontsize=15)
        plt.title(title, fontsize=19, pad=12)
        plt.grid(axis="y", alpha=0.22)

        finite_vals = df[metric_col].values
        finite_vals = finite_vals[np.isfinite(finite_vals)]

        if len(finite_vals):
            lo = float(np.nanpercentile(finite_vals, 0.5))
            hi = float(np.nanpercentile(finite_vals, 99.5))
            pad = 0.08 * (hi - lo + 1e-8)
            plt.ylim(lo - pad, hi + pad)

        plt.tight_layout()

        png = OUTDIR / f"{save_prefix}.png"
        svg = OUTDIR / f"{save_prefix}.svg"
        pdf = OUTDIR / f"{save_prefix}.pdf"

        plt.savefig(png, dpi=DPI, bbox_inches="tight")
        plt.savefig(svg, bbox_inches="tight")
        plt.savefig(pdf, bbox_inches="tight")

        print(f"[saved] {png}")
        print(f"[saved] {svg}")
        print(f"[saved] {pdf}")

        plt.show()

    print("\n" + "=" * 120)

    print("LOAD / BUILD RPE PERTURB-SEQ DATASETS")

    print("=" * 120)

    rpe_datasets = load_all_rpe_datasets()

    print("\n" + "=" * 120)

    print("CHOOSE CELLXGENE RPE + ONE FAR-FROM-RPE ATLAS SOURCE")

    print("=" * 120)

    cellxgene_sample_specs = choose_cellxgene_sources()

    print("\n" + "=" * 120)

    print("BUILD CELLXGENE RAW COVARIANCE SOURCES BY STREAMING")

    print("=" * 120)

    union_genes, union_targets = build_union_gene_target_sets_for_cellxgene(rpe_datasets)

    cellxgene_cov_sources = []

    for spec in tqdm(cellxgene_sample_specs, desc="CELLxGENE streamed covariance sources", ncols=TQDM_NCOLS):
        try:
            cxg_source = compute_cellxgene_cov_source_streamed(
                source_spec=spec,
                union_genes=union_genes,
                union_targets=union_targets,
            )
            cellxgene_cov_sources.append(cxg_source)
        except Exception as e:
            print(f"[ERROR CELLXGENE cov source] {spec.get('source_label', 'unknown')}: {repr(e)}")
            gc.collect()

    if len(cellxgene_cov_sources) == 0:
        raise RuntimeError("No CELLxGENE covariance sources were built.")

    print("\n" + "=" * 120)

    print("EVALUATE RPE RESPONSES AGAINST RPE + STREAMED CELLXGENE SOURCES")

    print("=" * 120)

    all_pair_summaries = []

    all_pert_rows = []

    all_errors = []

    for response in tqdm(rpe_datasets, desc="response RPE datasets", ncols=TQDM_NCOLS):
        print("\n" + "#" * 120)
        print(f"[response] {response['label']}")
        print("#" * 120)

        # First two columns:
        #   1) within dataset
        #   2) cross dataset
        for source in tqdm(
            rpe_datasets,
            desc=f"{response['label']}: RPE Pert-seq sources",
            ncols=TQDM_NCOLS,
            leave=False,
        ):
            try:
                pert_df, summary = evaluate_response_vs_rpe_source(response, source)
                all_pair_summaries.append(summary)

                if SAVE_PER_PERTURBATION_TABLE and pert_df is not None and len(pert_df):
                    all_pert_rows.append(pert_df)

                print(
                    f"[pair] {summary['composite_group']} | "
                    f"response={response['label']} | source={source['label']} | "
                    f"n={summary['n_eval']:,} | "
                    f"mean Pearson={summary['mean_pearson']:.4g} | "
                    f"mean R2={summary['mean_r2']:.4g}"
                )

            except Exception as e:
                print(f"[ERROR RPE pair] response={response['label']} source={source['label']}: {repr(e)}")
                all_errors.append({
                    "response_dataset": response["dataset"],
                    "source_dataset": source["dataset"],
                    "source_type": "RPE Pert-seq control Sigma",
                    "error": repr(e),
                })
                gc.collect()

        # Third and fourth columns:
        #   3) CELLxGENE RPE atlas cells
        #   4) CELLxGENE far-from-RPE atlas cells
        for cxg_source in tqdm(
            cellxgene_cov_sources,
            desc=f"{response['label']}: CELLxGENE sources",
            ncols=TQDM_NCOLS,
            leave=False,
        ):
            try:
                pert_df, summary = evaluate_response_vs_cellxgene_source(response, cxg_source)
                all_pair_summaries.append(summary)

                if SAVE_PER_PERTURBATION_TABLE and pert_df is not None and len(pert_df):
                    all_pert_rows.append(pert_df)

                print(
                    f"[pair] {summary['composite_group']} | "
                    f"response={response['label']} | source={cxg_source['source_label']} | "
                    f"n={summary['n_eval']:,} | "
                    f"mean Pearson={summary['mean_pearson']:.4g} | "
                    f"mean R2={summary['mean_r2']:.4g}"
                )

            except Exception as e:
                print(f"[ERROR CELLxGENE pair] response={response['label']} source={cxg_source['source_label']}: {repr(e)}")
                all_errors.append({
                    "response_dataset": response["dataset"],
                    "source_dataset": cxg_source["source_dataset"],
                    "source_type": cxg_source["source_type"],
                    "composite_group": cxg_source["composite_group"],
                    "error": repr(e),
                })
                gc.collect()

        gc.collect()

    pair_summary = pd.DataFrame(all_pair_summaries)

    pair_summary_path = OUTDIR / "rpe_first3_plus_far_atlas_pair_summary.csv"

    pair_summary.to_csv(pair_summary_path, index=False)

    print(f"[saved] {pair_summary_path}")

    if SAVE_PER_PERTURBATION_TABLE and len(all_pert_rows):
        per_pert_df = pd.concat(all_pert_rows, ignore_index=True)
    else:
        per_pert_df = pd.DataFrame()

    per_pert_path = OUTDIR / "rpe_first3_plus_far_atlas_per_perturbation_metrics.csv"

    per_pert_df.to_csv(per_pert_path, index=False)

    print(f"[saved] {per_pert_path}")

    errors_path = OUTDIR / "rpe_first3_plus_far_atlas_errors.json"

    with open(errors_path, "w") as f:
        json.dump(all_errors, f, indent=2, default=json_default)

    print(f"[saved] {errors_path}")

    print("\n[pair summary counts]")

    if len(pair_summary):
        display(
            pair_summary
            .groupby("composite_group", as_index=False)
            .agg(
                n_pairs=("source_dataset", "count"),
                mean_pair_pearson=("mean_pearson", "mean"),
                mean_pair_r2=("mean_r2", "mean"),
                total_eval_perts=("n_eval", "sum"),
            )
            .set_index("composite_group")
            .reindex(COMPOSITE_GROUP_ORDER)
            .reset_index()
        )

    print("\n[per-perturbation counts]")

    if len(per_pert_df):
        print(
            per_pert_df["composite_group"]
            .value_counts()
            .reindex(COMPOSITE_GROUP_ORDER)
            .fillna(0)
            .astype(int)
        )

    if len(per_pert_df) == 0:
        raise RuntimeError("No per-perturbation rows were generated; cannot plot composite violins.")

    plot_composite_violin(
        per_pert_df=per_pert_df,
        metric_col="pearson",
        ylabel="Pearson per perturbation",
        title="RPE per-perturbation Pearson distributions",
        save_prefix="RPE_FIRST3_PLUS_FAR_ATLAS_VIOLIN_pearson",
    )

    plot_composite_violin(
        per_pert_df=per_pert_df,
        metric_col="r2_uncentered",
        ylabel="Uncentered R² per perturbation",
        title="RPE per-perturbation uncentered R² distributions",
        save_prefix="RPE_FIRST3_PLUS_FAR_ATLAS_VIOLIN_R2",
    )

    print("\nDone.")
