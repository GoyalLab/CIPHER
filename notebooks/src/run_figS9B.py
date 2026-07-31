"""Fig S9B -- inverse-problem sampling-noise model comparison (CLT/fullH_diag vs isotropic).

For each dataset it ranks every gene as the likely driver of a perturbation's observed
mean-shift dx under two sampling-noise models in the control-covariance eigenbasis: a
per-eigenmode diagonal "CLT/fullH_diag" noise (h = lambda/n0 + diag(V^T Sigma_pert V)/n_pert)
and a trace-matched "isotropic" scalar noise. It filters genes/perturbations, builds the
control covariance and its eigendecomposition, fits an empirical-Bayes prior tau2 (plateau
rule) separately per noise model, scores every perturbation (target rank percentile, top-K
recovery, parametric-null -log10 p, gene-level top-K false-positive rate vs control-variance
percentile), aggregates across datasets, and draws the comparison figures. The deliverable
panel is the top-K target-recovery curve (panel B), produced with panel D by make_two_panel_bd.
run_inverse_from_precomputed reads the precomputed Sigma/dx tree; run_inverse_from_raw
recomputes everything from raw h5ads; make_two_panel_bd renders Fig S9B from the aggregate CSVs.

Helpers in notebooks/src (not part of the cipher package). Config constants are module globals
the notebook overrides via R.__dict__.update; DATA_DIR/SUPPL/OUTDIR injected.
"""
import os
import re
import gc
import json
import time
import math
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import h5py
import matplotlib.pyplot as plt

from scipy.sparse import issparse
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from tqdm import tqdm

import cipher

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Injected by the notebook config cell via R.__dict__.update.
DATA_DIR = None
SUPPL = None
OUTDIR = None


def run_inverse_from_precomputed():
    global PRECOMP_ROOT, EXPRESSION_CUTOFF, ONLY_RUN_CRISPRI_A_LISTS, RUN_ONLY_N_DATASETS
    global SIGMA_FILENAME, PERT_EIGVAR_FILENAME, PERT_EIGVAR_H5_KEY, PERT_VAR_H5_KEYS
    global ALLOW_GENE_DIAG_FALLBACK, SAVE_DERIVED_PERT_EIGVAR, FORCE_REDERIVE_PERT_EIGVAR
    global DERIVED_PERT_EIGVAR_DTYPE, DERIVED_PERT_EIGVAR_BATCH, SIGMA_SYMMETRIZE
    global FULLH_RIDGE_REL, FULLH_RIDGE_ABS, LOGTAU2_BOUNDS, EB_GRID_N, PLATEAU_DELTA
    global PLATEAU_PREFER, USE_PLATEAU_FOR_SCORING, PERT_BATCH, N_PARAM_NULL_REPS
    global K_GRID, FP_TOPK, VAR_BINS, CONTROL_LABELS, EXCLUDE_OBVIOUS_MULTI_GENE_PERTS
    global C_ISO, C_CLT, C_BASE, SEED, OVERWRITE, CRISPRa_KEYWORDS, CRISPRi_KEYWORDS
    global EXPRESSION_TAG

    PRECOMP_ROOT = os.path.join(SUPPL, "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED")

    EXPRESSION_CUTOFF = 1.0

    # default = all preprocessed datasets at this cutoff
    ONLY_RUN_CRISPRI_A_LISTS = False
    RUN_ONLY_N_DATASETS = None

    SIGMA_FILENAME = "Sigma_full_ridge.npy"
    PERT_EIGVAR_FILENAME = "pert_eigvar_true.npy"
    PERT_EIGVAR_H5_KEY = "pert_eigvar_true"

    PERT_VAR_H5_KEYS = [
        "var_pert",
        "pert_var",
        "perturbation_var",
        "var_perturbed",
    ]

    ALLOW_GENE_DIAG_FALLBACK = True
    SAVE_DERIVED_PERT_EIGVAR = True
    FORCE_REDERIVE_PERT_EIGVAR = False
    DERIVED_PERT_EIGVAR_DTYPE = np.float32
    DERIVED_PERT_EIGVAR_BATCH = 64

    SIGMA_SYMMETRIZE = True
    FULLH_RIDGE_REL = 1e-8
    FULLH_RIDGE_ABS = 1e-12

    LOGTAU2_BOUNDS = (-2.0, -1.0)
    EB_GRID_N = 100
    PLATEAU_DELTA = 1.92
    PLATEAU_PREFER = "largest"
    USE_PLATEAU_FOR_SCORING = True

    PERT_BATCH = 128

    # Set to 0 to skip p-value panel.
    N_PARAM_NULL_REPS = 100

    K_GRID = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300], dtype=int)

    FP_TOPK = 100
    VAR_BINS = np.linspace(0, 1, 21)

    CONTROL_LABELS = {"control", "ctrl", "non-targeting", "non_targeting", "nt", "NT"}
    EXCLUDE_OBVIOUS_MULTI_GENE_PERTS = True

    C_ISO = "blue"
    C_CLT = "purple"
    C_BASE = "#9e9e9e"

    SEED = 0
    OVERWRITE = True


    CRISPRa_KEYWORDS = [
        "akana_etal_2026_crispra_perturbseq",
        "schemidt_etal_2022_crispra_perturbseq",
        "kaden25_rpe1_ctrl_10k_min100_greedy_4gb",
        "kaden25_fibroblast_ctrl_10k_min100_greedy_4gb",
        "NormanWeissman2019_filtered",
        "TianKampmann2021_CRISPRa",
    ]

    CRISPRi_KEYWORDS = [
        "XAtlas2025_HEK293T_filtered",
        "Marson2025_D3_Stim8hr_filtered",
        "Marson2025_D4_Stim48hr_filtered",
        "Marson2025_D1_Stim48hr_filtered",
        "Marson2025_D1_Rest_filtered",
        "Marson2025_D4_Stim8hr_filtered",
        "Marson2025_D1_Stim8hr_filtered",
        "Marson2025_D4_Rest_filtered",
        "Marson2025_D2_Stim48hr_filtered",
        "Marson2025_D3_Stim48hr_filtered",
        "Marson2025_D3_Rest_filtered",
        "Marson2025_D2_Stim8hr_filtered",
        "XAtlas2025_HCT116_filtered",
        "ReplogleWeissman2022_rpe1",
        "ReplogleWeissman2022_K562_essential",
        "GSE264667_jurkat_raw_singlecell_01",
        "GSE264667_hepg2_raw_singlecell_01",
        "FrangiehIzar2021_RNA",
        "TianKampmann2019_day7neuron",
        "TianKampmann2021_CRISPRi",
        "TianKampmann2019_iPSC",
    ]


    # ============================================================
    # BASIC HELPERS
    # ============================================================

    def ensure_dir(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path


    def cutoff_to_tag(value):
        return f"{float(value):.1f}".replace(".", "p")


    EXPRESSION_TAG = cutoff_to_tag(EXPRESSION_CUTOFF)


    def nan0(x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


    def sym(A):
        return 0.5 * (A + A.T)


    def decode_arr(x):
        out = []
        for value in np.asarray(x):
            if isinstance(value, bytes):
                out.append(value.decode("utf-8"))
            else:
                out.append(str(value))
        return np.asarray(out, dtype=object)


    def find_one(root, filename):
        root = Path(root)

        direct = root / filename
        if direct.exists():
            return direct

        hits = sorted(root.rglob(filename))
        if not hits:
            raise FileNotFoundError(f"Could not find {filename!r} under {root}")

        return hits[0]


    def is_match(name, keywords):
        name = str(name)
        return any(str(keyword) in name for keyword in keywords)


    def dataset_name_from_precomp_dir(path):
        return Path(path).name.split("__mean_ge_")[0]


    def read_h5_scalar(h5, names):
        expanded = list(names) + [
            "n_control",
            "n_controls",
            "control_cells",
            "n_ctrl",
            "ctrl_n",
            "n_cells_control",
            "n_control_cells",
            "control_n",
            "n0",
        ]

        seen = set()
        expanded = [x for x in expanded if not (x in seen or seen.add(x))]

        for name in expanded:
            if name in h5:
                x = np.asarray(h5[name][()]).reshape(-1)
                if x.size:
                    value = float(x[0])
                    if np.isfinite(value) and value > 0:
                        return value

            if name in h5.attrs:
                value = float(h5.attrs[name])
                if np.isfinite(value) and value > 0:
                    return value

        return None


    def read_h5_vector(h5, names, expected_len):
        for name in names:
            if name in h5:
                x = np.asarray(h5[name][:], dtype=np.float64).reshape(-1)
                if x.shape == (expected_len,):
                    if np.all(np.isfinite(x)) and np.all(x > 0):
                        return x
                    raise ValueError(f"{h5.filename}:{name} has nonpositive/nonfinite values.")

        raise KeyError(f"Could not find any vector in {names} with length {expected_len}.")


    def find_dataset_dirs():
        root = Path(PRECOMP_ROOT)
        pattern = f"*__mean_ge_{EXPRESSION_TAG}"

        if not root.exists():
            raise FileNotFoundError(f"PRECOMP_ROOT does not exist: {root}")

        all_dirs = sorted({p for p in root.rglob(pattern) if p.is_dir()})

        print(f"[precomp root]      {root.name}")
        print(f"[expression cutoff] mean >= {EXPRESSION_CUTOFF}")
        print(f"[folder tag]        __mean_ge_{EXPRESSION_TAG}")
        print(f"[found total]       {len(all_dirs)}")

        if not all_dirs:
            print("\n[available cutoff folders]")
            for path in sorted(root.rglob("*__mean_ge_*"))[:200]:
                print(" ", path.name)
            raise FileNotFoundError(f"No folders found for pattern {pattern!r}")

        if ONLY_RUN_CRISPRI_A_LISTS:
            selected_keywords = CRISPRa_KEYWORDS + CRISPRi_KEYWORDS
            selected = [
                path for path in all_dirs
                if is_match(dataset_name_from_precomp_dir(path), selected_keywords)
            ]
        else:
            selected = all_dirs

        print(f"[selected]          {len(selected)}")

        for path in selected:
            dataset = dataset_name_from_precomp_dir(path)
            if is_match(dataset, CRISPRa_KEYWORDS):
                group = "CRISPRa"
            elif is_match(dataset, CRISPRi_KEYWORDS):
                group = "CRISPRi"
            else:
                group = "all"
            print(f"  [{group}] {dataset}")

        if not selected:
            raise FileNotFoundError("No datasets matched the active selection.")

        if RUN_ONLY_N_DATASETS is not None:
            selected = selected[:int(RUN_ONLY_N_DATASETS)]
            print(f"[debug] truncated to first {len(selected)} datasets")

        return selected


    def _percentile_rank_values(x):
        x = np.asarray(x, dtype=float).ravel()
        ok = np.isfinite(x)

        out = np.full_like(x, fill_value=np.nan, dtype=float)

        if np.sum(ok) <= 1:
            out[ok] = 0.0
            return out

        xx = x[ok]
        order = np.argsort(xx)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(xx), dtype=float)
        out[ok] = ranks / float(len(xx) - 1)

        return out


    def _safe_neglog10p(p):
        p = np.asarray(p, dtype=float)
        p = np.maximum(p, 1e-300)
        return -np.log10(p)


    def _axis_equal_with_diag(ax, lo=0.0, hi=1.0):
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color=C_BASE)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)


    # ============================================================
    # TARGET MATCHING
    # ============================================================

    def pert_to_gene_safe(perturbation, gene_set=None):
        original = str(perturbation).strip()

        if gene_set is not None and original in gene_set:
            return original

        gene = original

        gene = re.sub(
            r"([_\-\s]+)(KD|KO|OE|overexp|overexpression)$",
            "",
            gene,
            flags=re.IGNORECASE,
        )
        gene = re.sub(r"^(sg)(?=[A-Z0-9])", "", gene)
        gene = re.sub(
            r"^(sgRNA|gRNA|sgrna|grna|sg)([_\-\s]+)",
            "",
            gene,
            flags=re.IGNORECASE,
        )

        for separator in ["+", "|", " "]:
            if separator in gene:
                gene = gene.split(separator)[0]
                break

        if gene_set is not None and gene in gene_set:
            return gene

        for separator in ["_", "-"]:
            if separator in gene:
                first = gene.split(separator)[0]
                if gene_set is None or first in gene_set:
                    gene = first
                    break

        if gene_set is not None and gene not in gene_set and original in gene_set:
            return original

        return gene


    def looks_like_obvious_multi_gene_pert(perturbation, gene_set):
        p = str(perturbation).strip()

        if p in gene_set:
            return False

        for separator in ["+", "|"]:
            if separator in p:
                parts = [x for x in p.split(separator) if x]
                hits = sum(part in gene_set for part in parts)
                if hits >= 2:
                    return True

        if "_" in p:
            parts = [x for x in p.split("_") if x]
            hits = sum(part in gene_set for part in parts)
            if hits >= 2:
                return True

        return False


    def load_target_map_from_tables(ds_dir, gene_set):
        target_map = {}

        for filename in ["perturbation_table.tsv", "perturbation_target_map.tsv"]:
            path = Path(ds_dir) / filename

            if not path.exists():
                continue

            table = pd.read_csv(path, sep="\t")
            required = {"perturbation", "target_gene"}

            if not required.issubset(table.columns):
                continue

            for _, row in table.iterrows():
                perturbation = str(row["perturbation"])
                target_gene = "" if pd.isna(row["target_gene"]) else str(row["target_gene"])

                if target_gene in gene_set:
                    target_map[perturbation] = target_gene

        return target_map


    def load_targets(ds_dir, genes, perturbations):
        gene_set = set(map(str, genes))
        gene_to_idx = {str(gene): index for index, gene in enumerate(genes)}

        target_map = load_target_map_from_tables(ds_dir, gene_set)

        target_genes = []
        target_idx = []
        target_source = []

        for perturbation in perturbations:
            perturbation = str(perturbation)

            if perturbation in CONTROL_LABELS:
                target_genes.append("")
                target_idx.append(-1)
                target_source.append("control")
                continue

            if perturbation in target_map:
                gene = target_map[perturbation]
                source = "table"
            else:
                if (
                    EXCLUDE_OBVIOUS_MULTI_GENE_PERTS
                    and looks_like_obvious_multi_gene_pert(perturbation, gene_set)
                ):
                    gene = ""
                    source = "excluded_multi"
                else:
                    gene = pert_to_gene_safe(perturbation, gene_set=gene_set)
                    source = "parsed"

            target_genes.append(gene)
            target_idx.append(gene_to_idx.get(gene, -1))
            target_source.append(source)

        return (
            np.asarray(target_genes, dtype=object),
            np.asarray(target_idx, dtype=np.int64),
            np.asarray(target_source, dtype=object),
        )


    # ============================================================
    # PROJECTED PERTURBATION VARIANCE LOADING
    # ============================================================

    def _load_existing_pert_eigvar(ds_dir, stats_h5, expected_shape):
        if PERT_EIGVAR_H5_KEY in stats_h5 and not FORCE_REDERIVE_PERT_EIGVAR:
            eigvar = np.asarray(stats_h5[PERT_EIGVAR_H5_KEY][:], dtype=np.float64)
            source = f"{os.path.basename(stats_h5.filename)}:{PERT_EIGVAR_H5_KEY}"

            if eigvar.shape != expected_shape:
                raise ValueError(f"{source} has shape {eigvar.shape}; expected {expected_shape}.")

            eigvar = np.nan_to_num(eigvar, nan=0.0, posinf=0.0, neginf=0.0)
            eigvar = np.maximum(eigvar, 0.0)

            print(f"[fullH_diag] perturbation eigvar source: {source}")
            return eigvar

        path = Path(ds_dir) / PERT_EIGVAR_FILENAME

        if path.exists() and not FORCE_REDERIVE_PERT_EIGVAR:
            eigvar = np.load(path, mmap_mode="r")

            if eigvar.shape != expected_shape:
                raise ValueError(f"{path} has shape {eigvar.shape}; expected {expected_shape}.")

            print(f"[fullH_diag] perturbation eigvar source: {os.path.basename(path)}")
            return eigvar

        return None


    def _find_pert_var_dataset(stats_h5, expected_shape):
        for key in PERT_VAR_H5_KEYS:
            if key in stats_h5:
                ds = stats_h5[key]

                if ds.shape != expected_shape:
                    raise ValueError(
                        f"{stats_h5.filename}:{key} has shape {ds.shape}; expected {expected_shape}."
                    )

                return key, ds

        available = list(stats_h5.keys())
        raise FileNotFoundError(
            "Missing projected perturbation variance and missing per-gene fallback. "
            f"Expected one of {PERT_VAR_H5_KEYS}. Available H5 datasets: {available}"
        )


    def load_projected_perturbation_variances(ds_dir, stats_h5, V, expected_shape):
        ds_dir = Path(ds_dir)

        eigvar = _load_existing_pert_eigvar(
            ds_dir=ds_dir,
            stats_h5=stats_h5,
            expected_shape=expected_shape,
        )

        if eigvar is not None:
            return eigvar

        if not ALLOW_GENE_DIAG_FALLBACK:
            raise FileNotFoundError(
                f"Missing {PERT_EIGVAR_FILENAME} and {PERT_EIGVAR_H5_KEY}; "
                "ALLOW_GENE_DIAG_FALLBACK=False."
            )

        var_key, var_ds = _find_pert_var_dataset(stats_h5, expected_shape)

        derived_path = ds_dir / PERT_EIGVAR_FILENAME
        tmp_path = derived_path.with_suffix(derived_path.suffix + ".tmp")

        print(
            f"[fullH_diag] deriving {derived_path.name} from "
            f"{os.path.basename(stats_h5.filename)}:{var_key}"
        )
        print("[fullH_diag] using gene-diagonal fallback: var_pert @ (V * V)")

        n_perts, _ = expected_shape
        V2 = np.asarray(V * V, dtype=np.float64)

        if SAVE_DERIVED_PERT_EIGVAR:
            if tmp_path.exists():
                tmp_path.unlink()

            eigvar_mm = np.lib.format.open_memmap(
                tmp_path,
                mode="w+",
                dtype=DERIVED_PERT_EIGVAR_DTYPE,
                shape=expected_shape,
            )

            for row_start in tqdm(
                range(0, n_perts, DERIVED_PERT_EIGVAR_BATCH),
                desc="derive pert_eigvar_true",
                leave=False,
            ):
                row_end = min(row_start + DERIVED_PERT_EIGVAR_BATCH, n_perts)

                var_batch = np.asarray(var_ds[row_start:row_end, :], dtype=np.float64)
                var_batch = np.nan_to_num(var_batch, nan=0.0, posinf=0.0, neginf=0.0)
                var_batch = np.maximum(var_batch, 0.0)

                eig_batch = var_batch @ V2
                eig_batch = np.nan_to_num(eig_batch, nan=0.0, posinf=0.0, neginf=0.0)
                eig_batch = np.maximum(eig_batch, 0.0)

                eigvar_mm[row_start:row_end, :] = eig_batch.astype(
                    DERIVED_PERT_EIGVAR_DTYPE,
                    copy=False,
                )
                eigvar_mm.flush()

                del var_batch, eig_batch

            del eigvar_mm, V2
            gc.collect()

            if derived_path.exists():
                derived_path.unlink()

            tmp_path.rename(derived_path)

            eigvar = np.load(derived_path, mmap_mode="r")
            print(f"[fullH_diag] saved derived perturbation eigvar: {derived_path.name}")

            return eigvar

        eigvar = np.empty(expected_shape, dtype=np.float64)

        for row_start in tqdm(
            range(0, n_perts, DERIVED_PERT_EIGVAR_BATCH),
            desc="derive pert_eigvar_true",
            leave=False,
        ):
            row_end = min(row_start + DERIVED_PERT_EIGVAR_BATCH, n_perts)

            var_batch = np.asarray(var_ds[row_start:row_end, :], dtype=np.float64)
            var_batch = np.nan_to_num(var_batch, nan=0.0, posinf=0.0, neginf=0.0)
            var_batch = np.maximum(var_batch, 0.0)

            eig_batch = var_batch @ V2
            eig_batch = np.nan_to_num(eig_batch, nan=0.0, posinf=0.0, neginf=0.0)
            eig_batch = np.maximum(eig_batch, 0.0)

            eigvar[row_start:row_end, :] = eig_batch

            del var_batch, eig_batch

        del V2
        gc.collect()

        return eigvar


    # ============================================================
    # MODEL LOADING
    # ============================================================

    def load_true_fullh_diag_model(ds_dir, stats_h5):
        n_perts, n_genes = stats_h5["dx"].shape

        n0 = read_h5_scalar(
            stats_h5,
            [
                "n_cells_control",
                "n_control_cells",
                "control_n",
                "n0",
                "n_control",
            ],
        )

        if n0 is None or not np.isfinite(n0) or n0 <= 0:
            raise ValueError(
                "Could not find positive n_control/n0 in perturbation_stats.h5 datasets or attrs."
            )

        nu = read_h5_vector(
            stats_h5,
            names=[
                "n_cells_pert",
                "n_cells_perturbed",
                "n_pert",
                "pert_n",
                "n1",
            ],
            expected_len=n_perts,
        )

        sigma_path = find_one(ds_dir, SIGMA_FILENAME)

        Sigma = np.asarray(np.load(sigma_path, mmap_mode="r"), dtype=np.float64)
        Sigma = nan0(Sigma)

        if Sigma.shape != (n_genes, n_genes):
            raise ValueError(
                f"{sigma_path} has shape {Sigma.shape}; expected {(n_genes, n_genes)}."
            )

        if SIGMA_SYMMETRIZE:
            Sigma = sym(Sigma)

        control_gene_var = np.diag(Sigma).astype(np.float64, copy=True)

        print(f"[fullH_diag] eigendecomposition true Sigma, shape={Sigma.shape}")
        eigenvalues, V = np.linalg.eigh(Sigma)

        del Sigma
        gc.collect()

        eigenvalues = np.nan_to_num(eigenvalues, nan=0.0, posinf=0.0, neginf=0.0)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        V = nan0(V)

        pert_eigvar = load_projected_perturbation_variances(
            ds_dir=ds_dir,
            stats_h5=stats_h5,
            V=V,
            expected_shape=(n_perts, n_genes),
        )

        positive = eigenvalues[eigenvalues > 0]
        scale = float(np.median(positive) / n0) if positive.size else 1.0
        ridge = max(float(FULLH_RIDGE_ABS), float(FULLH_RIDGE_REL) * scale)

        print(
            "[fullH_diag]"
            f" n0={n0:.0f}"
            f" | nu min/median/max={nu.min():.0f}/{np.median(nu):.0f}/{nu.max():.0f}"
            f" | ridge={ridge:.3e}"
        )

        return {
            "sigma_path": str(sigma_path),
            "eigenvalues": eigenvalues,
            "V": V,
            "V2T": (V * V).T,
            "pert_eigvar": pert_eigvar,
            "n0": float(n0),
            "nu": nu.astype(np.float64, copy=False),
            "ridge": float(ridge),
            "control_gene_var": control_gene_var,
        }


    def get_batch_y_h(model, dx_dataset, row_indices):
        row_indices = np.asarray(row_indices, dtype=np.int64)

        eigenvalues = model["eigenvalues"]
        V = model["V"]

        DX = np.asarray(dx_dataset[row_indices, :], dtype=np.float64)
        DX = nan0(DX)

        y = DX @ V

        h_clt = (
            eigenvalues[None, :] / model["n0"]
            + np.asarray(model["pert_eigvar"][row_indices, :], dtype=np.float64)
            / model["nu"][row_indices, None]
        )

        h_clt = np.nan_to_num(h_clt, nan=0.0, posinf=0.0, neginf=0.0)
        h_clt = np.maximum(h_clt, model["ridge"])

        h_iso_scalar = np.mean(h_clt, axis=1, keepdims=True)
        h_iso_scalar = np.maximum(h_iso_scalar, model["ridge"])

        return y, h_clt, h_iso_scalar


    # ============================================================
    # EMPIRICAL BAYES tau^2
    # ============================================================

    def _pick_plateau_logtau2(grid, nll_curve, delta, prefer="largest"):
        grid = np.asarray(grid, dtype=float)
        nll_curve = np.asarray(nll_curve, dtype=float)

        jmin = int(np.nanargmin(nll_curve))
        nll_min = float(nll_curve[jmin])

        ok = np.where(nll_curve <= nll_min + float(delta))[0]

        if ok.size == 0:
            return float(grid[jmin]), nll_min, nll_min

        j = int(ok[-1] if prefer == "largest" else ok[0])
        return float(grid[j]), nll_min, float(nll_curve[j])


    def build_eb_cache(model, dx_dataset, valid_row_indices, mode):
        assert mode in {"iso", "clt"}

        n_valid = int(len(valid_row_indices))
        n_genes = int(dx_dataset.shape[1])
        eigenvalues = model["eigenvalues"]

        d2_all = np.empty((n_valid, n_genes), dtype=np.float32)
        z2_all = np.empty((n_valid, n_genes), dtype=np.float32)

        for local_start in tqdm(
            range(0, n_valid, PERT_BATCH),
            desc=f"EB cache {mode}",
            leave=False,
        ):
            local_end = min(local_start + PERT_BATCH, n_valid)
            rows = valid_row_indices[local_start:local_end]

            y, h_clt, h_iso_scalar = get_batch_y_h(
                model=model,
                dx_dataset=dx_dataset,
                row_indices=rows,
            )

            h = h_clt if mode == "clt" else h_iso_scalar

            d2 = (eigenvalues[None, :] * eigenvalues[None, :]) / h
            z2 = (y * y) / h

            d2_all[local_start:local_end, :] = d2.astype(np.float32, copy=False)
            z2_all[local_start:local_end, :] = z2.astype(np.float32, copy=False)

            del y, h_clt, h_iso_scalar, h, d2, z2

        return d2_all, z2_all


    def fit_tau2_from_cache(d2_all, z2_all):
        def negative_log_likelihood(log_tau2):
            tau2 = float(np.exp(log_tau2))
            marginal_variance = np.maximum(1.0 + tau2 * d2_all, 1e-12)

            return 0.5 * float(
                np.sum(
                    np.log(marginal_variance) + z2_all / marginal_variance,
                    dtype=np.float64,
                )
            )

        result = minimize_scalar(
            negative_log_likelihood,
            bounds=LOGTAU2_BOUNDS,
            method="bounded",
        )

        logtau2_opt = float(result.x)
        tau2_opt = float(np.exp(logtau2_opt))
        nll_opt = float(result.fun)

        grid = np.linspace(LOGTAU2_BOUNDS[0], LOGTAU2_BOUNDS[1], int(EB_GRID_N))
        nll_grid = np.asarray(
            [negative_log_likelihood(value) for value in grid],
            dtype=np.float64,
        )

        logtau2_plateau, nll_min, nll_plateau = _pick_plateau_logtau2(
            grid=grid,
            nll_curve=nll_grid,
            delta=PLATEAU_DELTA,
            prefer=PLATEAU_PREFER,
        )

        tau2_plateau = float(np.exp(logtau2_plateau))
        tau2_use = tau2_plateau if USE_PLATEAU_FOR_SCORING else tau2_opt

        return {
            "tau2_opt": tau2_opt,
            "logtau2_opt": logtau2_opt,
            "nll_opt": nll_opt,
            "tau2_plateau": tau2_plateau,
            "logtau2_plateau": float(logtau2_plateau),
            "nll_min_grid": float(nll_min),
            "nll_plateau": float(nll_plateau),
            "tau2_use": float(tau2_use),
        }


    def fit_tau2_for_mode(model, dx_dataset, valid_row_indices, mode):
        d2_all, z2_all = build_eb_cache(
            model=model,
            dx_dataset=dx_dataset,
            valid_row_indices=valid_row_indices,
            mode=mode,
        )

        eb = fit_tau2_from_cache(d2_all, z2_all)

        del d2_all, z2_all
        gc.collect()

        return eb


    # ============================================================
    # POSTERIOR SCORES
    # ============================================================

    def posterior_score_batch(model, y, h, tau2):
        eigenvalues = model["eigenvalues"]
        V = model["V"]
        V2T = model["V2T"]

        prior_precision = 1.0 / float(tau2)

        d2 = (eigenvalues[None, :] * eigenvalues[None, :]) / h
        posterior_variance_eigenbasis = 1.0 / np.maximum(d2 + prior_precision, 1e-12)

        posterior_mean_eigenbasis = (
            posterior_variance_eigenbasis
            * eigenvalues[None, :]
            * y
            / h
        )

        posterior_mean = posterior_mean_eigenbasis @ V.T
        posterior_variance_diag = posterior_variance_eigenbasis @ V2T
        posterior_std = np.sqrt(np.maximum(posterior_variance_diag, 0.0))

        score = np.maximum(
            np.abs(posterior_mean + posterior_std),
            np.abs(posterior_mean - posterior_std),
        )

        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


    def ranks_and_target_scores(score, target_idx):
        score = np.asarray(score, dtype=np.float64)
        target_idx = np.asarray(target_idx, dtype=np.int64)

        target_scores = score[np.arange(score.shape[0]), target_idx]
        ranks = 1 + np.sum(score > target_scores[:, None], axis=1)

        return ranks.astype(np.int64), target_scores.astype(np.float64)


    def update_false_positive_counts(fp_counts, score, target_idx, topk):
        n_rows, n_genes = score.shape
        k_eff = min(int(topk), n_genes - 1)

        if k_eff <= 0:
            return

        for i in range(n_rows):
            t = int(target_idx[i])

            row = score[i].copy()
            row[t] = -np.inf

            idx = np.argpartition(-row, kth=k_eff - 1)[:k_eff]
            fp_counts[idx] += 1.0


    def target_scores_for_null_y(y_null, eigenvalues, V_row, V2_row, tau2, h):
        y_null = np.asarray(y_null, dtype=np.float64)

        hvec = np.asarray(h, dtype=np.float64).reshape(-1)
        if hvec.size == 1:
            hvec = np.full_like(eigenvalues, fill_value=float(hvec[0]), dtype=np.float64)

        hvec = np.maximum(hvec, 1e-300)

        posterior_variance_eigenbasis = 1.0 / np.maximum(
            (eigenvalues * eigenvalues) / hvec + 1.0 / float(tau2),
            1e-12,
        )

        coef = V_row * posterior_variance_eigenbasis * eigenvalues / hvec
        mu = y_null @ coef

        var_g = float(np.sum(V2_row * posterior_variance_eigenbasis))
        std_g = math.sqrt(max(var_g, 0.0))

        scores = np.maximum(np.abs(mu + std_g), np.abs(mu - std_g))

        return scores


    # ============================================================
    # RUN ONE DATASET
    # ============================================================

    def run_one_dataset(ds_dir, outdir, seed=0):
        rng = np.random.default_rng(seed)

        ds_dir = Path(ds_dir)
        dataset = dataset_name_from_precomp_dir(ds_dir)

        print("\n" + "=" * 100)
        print(f"[dataset] {dataset}")
        print(f"[path]    {ds_dir.name}")
        print("=" * 100)

        stats_path = find_one(ds_dir, "perturbation_stats.h5")

        with h5py.File(stats_path, "r") as stats_h5:
            required = ["dx", "gene_names", "perturbations"]
            missing = [x for x in required if x not in stats_h5]
            if missing:
                raise KeyError(f"Missing required datasets in {stats_path}: {missing}")

            dx_dataset = stats_h5["dx"]
            n_perts, n_genes = dx_dataset.shape

            genes = decode_arr(stats_h5["gene_names"][:])
            perturbations = decode_arr(stats_h5["perturbations"][:])

            if len(genes) != n_genes:
                raise ValueError(f"gene_names has length {len(genes)}, but dx has {n_genes} genes.")

            if len(perturbations) != n_perts:
                raise ValueError(
                    f"perturbations has length {len(perturbations)}, but dx has {n_perts} rows."
                )

            target_genes, target_idx, target_source = load_targets(
                ds_dir=ds_dir,
                genes=genes,
                perturbations=perturbations,
            )

            valid = (target_idx >= 0) & (target_idx < n_genes)
            valid_row_indices = np.where(valid)[0].astype(np.int64)
            valid_target_idx = target_idx[valid_row_indices].astype(np.int64)

            print(f"[data] n_perts total:      {n_perts:,}")
            print(f"[data] n_genes:            {n_genes:,}")
            print(f"[data] n_valid targets:    {len(valid_row_indices):,}")
            print(f"[data] target sources:     {dict(pd.Series(target_source).value_counts())}")

            if len(valid_row_indices) == 0:
                raise ValueError(f"No valid perturbation targets found for {dataset}.")

            model = load_true_fullh_diag_model(
                ds_dir=ds_dir,
                stats_h5=stats_h5,
            )

            # --------------------------------------------------------
            # EB fit
            # --------------------------------------------------------
            print("[EB] fitting isotropic tau2")
            eb_iso = fit_tau2_for_mode(
                model=model,
                dx_dataset=dx_dataset,
                valid_row_indices=valid_row_indices,
                mode="iso",
            )

            print("[EB] fitting CLT/fullH_diag tau2")
            eb_clt = fit_tau2_for_mode(
                model=model,
                dx_dataset=dx_dataset,
                valid_row_indices=valid_row_indices,
                mode="clt",
            )

            tau2_iso = float(eb_iso["tau2_use"])
            tau2_clt = float(eb_clt["tau2_use"])

            print(f"[EB] isotropic tau2 used: {tau2_iso:.6g}")
            print(f"[EB] CLT tau2 used:       {tau2_clt:.6g}")

            # --------------------------------------------------------
            # Real perturbation scoring
            # --------------------------------------------------------
            print("[real] scoring isotropic vs CLT")

            fp_count_iso = np.zeros(n_genes, dtype=np.float64)
            fp_count_clt = np.zeros(n_genes, dtype=np.float64)

            perpert_rows = []

            real_target_score_iso = np.empty(len(valid_row_indices), dtype=np.float64)
            real_target_score_clt = np.empty(len(valid_row_indices), dtype=np.float64)

            local_position_by_original_row = {
                int(row): i for i, row in enumerate(valid_row_indices)
            }

            for local_start in tqdm(
                range(0, len(valid_row_indices), PERT_BATCH),
                desc=f"{dataset}: real scoring",
                leave=False,
            ):
                local_end = min(local_start + PERT_BATCH, len(valid_row_indices))
                rows = valid_row_indices[local_start:local_end]
                targets = valid_target_idx[local_start:local_end]

                y, h_clt, h_iso_scalar = get_batch_y_h(
                    model=model,
                    dx_dataset=dx_dataset,
                    row_indices=rows,
                )

                score_iso = posterior_score_batch(
                    model=model,
                    y=y,
                    h=h_iso_scalar,
                    tau2=tau2_iso,
                )

                score_clt = posterior_score_batch(
                    model=model,
                    y=y,
                    h=h_clt,
                    tau2=tau2_clt,
                )

                rank_iso, target_score_iso = ranks_and_target_scores(score_iso, targets)
                rank_clt, target_score_clt = ranks_and_target_scores(score_clt, targets)

                update_false_positive_counts(
                    fp_counts=fp_count_iso,
                    score=score_iso,
                    target_idx=targets,
                    topk=FP_TOPK,
                )

                update_false_positive_counts(
                    fp_counts=fp_count_clt,
                    score=score_clt,
                    target_idx=targets,
                    topk=FP_TOPK,
                )

                for j, row in enumerate(rows):
                    local = local_start + j

                    real_target_score_iso[local] = float(target_score_iso[j])
                    real_target_score_clt[local] = float(target_score_clt[j])

                    r_iso = int(rank_iso[j])
                    r_clt = int(rank_clt[j])

                    perpert_rows.append({
                        "dataset": dataset,
                        "precomp_dir": str(ds_dir),
                        "perturbation_row": int(row),
                        "perturbation": str(perturbations[row]),
                        "target_gene": str(target_genes[row]),
                        "target_idx": int(targets[j]),
                        "target_source": str(target_source[row]),
                        "n_p": int(model["nu"][row]),

                        "rank_iso": r_iso,
                        "rank_clt": r_clt,
                        "rankpct_iso": float(r_iso / n_genes),
                        "rankpct_clt": float(r_clt / n_genes),
                        "rankpct_improvement_iso_minus_clt": float((r_iso - r_clt) / n_genes),

                        "target_score_iso": float(target_score_iso[j]),
                        "target_score_clt": float(target_score_clt[j]),
                    })

                del y, h_clt, h_iso_scalar, score_iso, score_clt

            perpert_df = pd.DataFrame(perpert_rows)
            perpert_df = perpert_df.sort_values("perturbation_row").reset_index(drop=True)

            sort_original_rows = perpert_df["perturbation_row"].values.astype(int)
            sort_local = np.asarray(
                [local_position_by_original_row[int(row)] for row in sort_original_rows],
                dtype=np.int64,
            )

            real_target_score_iso_sorted = real_target_score_iso[sort_local]
            real_target_score_clt_sorted = real_target_score_clt[sort_local]

            # --------------------------------------------------------
            # Parametric null p-values for target-gene score
            # --------------------------------------------------------
            if int(N_PARAM_NULL_REPS) > 0:
                print("[null] target-gene parametric p-values")

                p_iso = np.empty(len(perpert_df), dtype=np.float64)
                p_clt = np.empty(len(perpert_df), dtype=np.float64)

                eigenvalues = model["eigenvalues"]
                V = model["V"]

                for local_start in tqdm(
                    range(0, len(sort_original_rows), PERT_BATCH),
                    desc=f"{dataset}: null p-values",
                    leave=False,
                ):
                    local_end = min(local_start + PERT_BATCH, len(sort_original_rows))
                    rows = sort_original_rows[local_start:local_end]
                    targets = perpert_df["target_idx"].values[local_start:local_end].astype(int)

                    _, h_clt_batch, h_iso_scalar_batch = get_batch_y_h(
                        model=model,
                        dx_dataset=dx_dataset,
                        row_indices=rows,
                    )

                    for j, row in enumerate(rows):
                        target = int(targets[j])

                        h_clt = h_clt_batch[j]
                        h_iso_scalar = float(h_iso_scalar_batch[j, 0])

                        V_row = V[target, :]
                        V2_row = V_row * V_row

                        y_null = rng.normal(
                            size=(int(N_PARAM_NULL_REPS), n_genes)
                        ).astype(np.float64)

                        y_null *= np.sqrt(h_clt)[None, :]

                        null_scores_iso = target_scores_for_null_y(
                            y_null=y_null,
                            eigenvalues=eigenvalues,
                            V_row=V_row,
                            V2_row=V2_row,
                            tau2=tau2_iso,
                            h=np.asarray([h_iso_scalar], dtype=np.float64),
                        )

                        null_scores_clt = target_scores_for_null_y(
                            y_null=y_null,
                            eigenvalues=eigenvalues,
                            V_row=V_row,
                            V2_row=V2_row,
                            tau2=tau2_clt,
                            h=h_clt,
                        )

                        p_iso[local_start + j] = (
                            1.0
                            + float(np.sum(null_scores_iso >= real_target_score_iso_sorted[local_start + j]))
                        ) / float(N_PARAM_NULL_REPS + 1)

                        p_clt[local_start + j] = (
                            1.0
                            + float(np.sum(null_scores_clt >= real_target_score_clt_sorted[local_start + j]))
                        ) / float(N_PARAM_NULL_REPS + 1)

                    del h_clt_batch, h_iso_scalar_batch

                perpert_df["param_p_iso"] = p_iso
                perpert_df["param_p_clt"] = p_clt
                perpert_df["neglog10p_iso"] = _safe_neglog10p(p_iso)
                perpert_df["neglog10p_clt"] = _safe_neglog10p(p_clt)
                perpert_df["delta_neglog10p_clt_minus_iso"] = (
                    perpert_df["neglog10p_clt"] - perpert_df["neglog10p_iso"]
                )

            else:
                perpert_df["param_p_iso"] = np.nan
                perpert_df["param_p_clt"] = np.nan
                perpert_df["neglog10p_iso"] = np.nan
                perpert_df["neglog10p_clt"] = np.nan
                perpert_df["delta_neglog10p_clt_minus_iso"] = np.nan

            # --------------------------------------------------------
            # Gene-level false-positive variance bias table
            # --------------------------------------------------------
            n_scored = max(len(perpert_df), 1)

            control_gene_var = np.asarray(model["control_gene_var"], dtype=np.float64)
            control_gene_var_pct = _percentile_rank_values(control_gene_var)

            gene_fp_df = pd.DataFrame({
                "dataset": dataset,
                "precomp_dir": str(ds_dir),
                "gene_idx": np.arange(n_genes, dtype=int),
                "gene": genes.astype(str),
                "control_variance": control_gene_var,
                "control_variance_percentile": control_gene_var_pct,
                "fp_rate_iso": fp_count_iso / float(n_scored),
                "fp_rate_clt": fp_count_clt / float(n_scored),
            })

            gene_fp_df["fp_rate_diff_iso_minus_clt"] = (
                gene_fp_df["fp_rate_iso"] - gene_fp_df["fp_rate_clt"]
            )

            # --------------------------------------------------------
            # Dataset summary
            # --------------------------------------------------------
            topk_summary = {}
            for K in K_GRID:
                K_eff = min(int(K), int(n_genes))
                topk_summary[f"top{K}_iso"] = float(np.mean(perpert_df["rank_iso"].values <= K_eff))
                topk_summary[f"top{K}_clt"] = float(np.mean(perpert_df["rank_clt"].values <= K_eff))

            summary = {
                "dataset": dataset,
                "precomp_dir": str(ds_dir),
                "stats_path": str(stats_path),
                "sigma_path": model["sigma_path"],
                "n_perts_total": int(n_perts),
                "n_genes": int(n_genes),
                "n_control_cells": int(round(model["n0"])),
                "n_scored_perts": int(len(perpert_df)),

                "tau2_iso": float(tau2_iso),
                "tau2_clt": float(tau2_clt),
                "tau2_iso_opt": float(eb_iso["tau2_opt"]),
                "tau2_clt_opt": float(eb_clt["tau2_opt"]),
                "tau2_iso_plateau": float(eb_iso["tau2_plateau"]),
                "tau2_clt_plateau": float(eb_clt["tau2_plateau"]),

                "median_rankpct_iso": float(np.nanmedian(perpert_df["rankpct_iso"])),
                "median_rankpct_clt": float(np.nanmedian(perpert_df["rankpct_clt"])),
                "median_rankpct_improvement_iso_minus_clt": float(
                    np.nanmedian(perpert_df["rankpct_improvement_iso_minus_clt"])
                ),
                "fraction_rankpct_clt_better": float(
                    np.nanmean(perpert_df["rankpct_clt"] < perpert_df["rankpct_iso"])
                ),

                "median_neglog10p_iso": float(np.nanmedian(perpert_df["neglog10p_iso"])),
                "median_neglog10p_clt": float(np.nanmedian(perpert_df["neglog10p_clt"])),
                "median_delta_neglog10p_clt_minus_iso": float(
                    np.nanmedian(perpert_df["delta_neglog10p_clt_minus_iso"])
                ),
                "fraction_p_clt_better": float(
                    np.nanmean(perpert_df["param_p_clt"] < perpert_df["param_p_iso"])
                ),

                **topk_summary,
            }

            # --------------------------------------------------------
            # Save per-dataset outputs
            # --------------------------------------------------------
            perpert_csv = Path(outdir) / f"{dataset}__perpert_inverse_metrics.csv"
            genefp_csv = Path(outdir) / f"{dataset}__gene_false_positive_variance_bias.csv"
            summary_json = Path(outdir) / f"{dataset}__summary.json"

            perpert_df.to_csv(perpert_csv, index=False)
            gene_fp_df.to_csv(genefp_csv, index=False)

            with open(summary_json, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"[saved] {os.path.basename(perpert_csv)}")
            print(f"[saved] {os.path.basename(genefp_csv)}")
            print(
                f"[summary] median rankpct iso={summary['median_rankpct_iso']:.4g}, "
                f"CLT={summary['median_rankpct_clt']:.4g}"
            )
            print(f"[summary] frac CLT rank better={summary['fraction_rankpct_clt_better']:.3f}")

            del model
            gc.collect()

            return perpert_df, gene_fp_df, summary


    # ============================================================
    # AGGREGATE PLOTTING
    # ============================================================

    def make_final_four_panel_figure(all_perpert_df, all_gene_fp_df, summary_df, outdir):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        axA = axes[0, 0]
        axB = axes[0, 1]
        axC = axes[1, 0]
        axD = axes[1, 1]

        # A. Target-rank scatter
        x = all_perpert_df["rankpct_iso"].values.astype(float)
        y = all_perpert_df["rankpct_clt"].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)

        x = x[ok]
        y = y[ok]

        axA.scatter(x, y, s=18, alpha=0.45, edgecolor="none")
        _axis_equal_with_diag(axA, lo=0.0, hi=1.0)

        frac_clt_better = float(np.mean(y < x)) if x.size else np.nan
        med_improve = float(np.median(x - y)) if x.size else np.nan

        axA.set_xlabel("Target rank percentile, isotropic")
        axA.set_ylabel("Target rank percentile, CLT")
        axA.set_title("A. CLT improves target-gene ranks\nbelow diagonal = CLT better")

        axA.text(
            0.03,
            0.97,
            f"fraction CLT better = {frac_clt_better:.3f}\nmedian improvement = {med_improve:.3g}",
            transform=axA.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, linewidth=0.5),
        )

        # B. Top-K recovery
        topk_iso = []
        topk_clt = []

        for K in K_GRID:
            topk_iso.append(float(np.mean(all_perpert_df["rank_iso"].values <= K)))
            topk_clt.append(float(np.mean(all_perpert_df["rank_clt"].values <= K)))

        topk_iso = np.array(topk_iso)
        topk_clt = np.array(topk_clt)

        axB.plot(K_GRID, topk_iso, marker="o", linewidth=2.2, color=C_ISO, label="isotropic")
        axB.plot(K_GRID, topk_clt, marker="o", linewidth=2.2, color=C_CLT, label="CLT")

        axB.set_xscale("log")
        axB.set_xlabel("Top K genes")
        axB.set_ylabel("Fraction of perturbations recovered")
        axB.set_ylim(0, 1.02)
        axB.set_title("B. Practical top-K target recovery")
        axB.legend(frameon=False)

        for K_show in [10, 50, 100]:
            if K_show in set(K_GRID):
                i = int(np.where(K_GRID == K_show)[0][0])
                axB.text(
                    K_show,
                    topk_clt[i],
                    f" Δ={topk_clt[i] - topk_iso[i]:+.2f}",
                    fontsize=9,
                    va="bottom",
                    ha="left",
                )

        # C. Null-calibrated p-value scatter
        x = all_perpert_df["neglog10p_iso"].values.astype(float)
        y = all_perpert_df["neglog10p_clt"].values.astype(float)

        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]

        if x.size:
            maxv = max(1.0, float(np.nanpercentile(np.concatenate([x, y]), 99.5)))
            maxv = min(maxv * 1.08, max(float(np.max(x)), float(np.max(y)), 1.0) * 1.05)

            axC.scatter(x, y, s=18, alpha=0.45, edgecolor="none")
            axC.plot([0, maxv], [0, maxv], linestyle="--", linewidth=1.2, color=C_BASE)
            axC.set_xlim(0, maxv)
            axC.set_ylim(0, maxv)

            frac_p_better = float(np.mean(y > x))
            med_delta_p = float(np.median(y - x))
        else:
            axC.text(
                0.5,
                0.5,
                "p-values skipped\nN_PARAM_NULL_REPS = 0",
                transform=axC.transAxes,
                ha="center",
                va="center",
            )
            axC.set_xlim(0, 1)
            axC.set_ylim(0, 1)
            frac_p_better = np.nan
            med_delta_p = np.nan

        axC.set_xlabel(r"$-\log_{10} p$, isotropic")
        axC.set_ylabel(r"$-\log_{10} p$, CLT")
        axC.set_title("C. Null-calibrated target support\nabove diagonal = CLT stronger")

        axC.text(
            0.03,
            0.97,
            f"fraction CLT stronger = {frac_p_better:.3f}\nmedian Δ = {med_delta_p:.3g}",
            transform=axC.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, linewidth=0.5),
        )

        # D. False-positive variance bias
        df_fp = all_gene_fp_df.copy()
        df_fp = df_fp[np.isfinite(df_fp["control_variance_percentile"])]

        df_fp["var_bin"] = pd.cut(
            df_fp["control_variance_percentile"],
            bins=VAR_BINS,
            include_lowest=True,
            labels=False,
        )

        bin_rows = []

        for b in sorted(df_fp["var_bin"].dropna().unique()):
            sub = df_fp[df_fp["var_bin"] == b]

            if len(sub) == 0:
                continue

            bin_rows.append({
                "bin": int(b),
                "x": float(sub["control_variance_percentile"].mean()),
                "fp_iso_mean": float(sub["fp_rate_iso"].mean()),
                "fp_clt_mean": float(sub["fp_rate_clt"].mean()),
                "fp_iso_sem": float(sub["fp_rate_iso"].std(ddof=1) / math.sqrt(max(len(sub), 1))),
                "fp_clt_sem": float(sub["fp_rate_clt"].std(ddof=1) / math.sqrt(max(len(sub), 1))),
                "n_genes": int(len(sub)),
            })

        bin_df = pd.DataFrame(bin_rows)

        if len(df_fp) > 2:
            rho_iso, _ = spearmanr(
                df_fp["control_variance_percentile"].values,
                df_fp["fp_rate_iso"].values,
            )
            rho_clt, _ = spearmanr(
                df_fp["control_variance_percentile"].values,
                df_fp["fp_rate_clt"].values,
            )
        else:
            rho_iso = np.nan
            rho_clt = np.nan

        if len(bin_df):
            axD.errorbar(
                bin_df["x"],
                bin_df["fp_iso_mean"],
                yerr=bin_df["fp_iso_sem"],
                marker="o",
                linewidth=2.2,
                capsize=2,
                color=C_ISO,
                label=f"isotropic, ρ={rho_iso:.2f}",
            )

            axD.errorbar(
                bin_df["x"],
                bin_df["fp_clt_mean"],
                yerr=bin_df["fp_clt_sem"],
                marker="o",
                linewidth=2.2,
                capsize=2,
                color=C_CLT,
                label=f"CLT, ρ={rho_clt:.2f}",
            )

        axD.set_xlabel("Control variance percentile")
        axD.set_ylabel(f"False-positive top-{FP_TOPK} frequency")
        axD.set_title("D. Variance-driven false positives\nlower/flatter = less noise bias")
        axD.legend(frameon=False)

        n_datasets = summary_df.shape[0]
        n_perts = all_perpert_df.shape[0]
        n_gene_entries = all_gene_fp_df.shape[0]

        fig.suptitle(
            (
                "CLT/fullH_diag noise vs trace-matched isotropic noise\n"
                f"{n_datasets} datasets, {n_perts} scored perturbations, "
                f"{n_gene_entries} gene-level false-positive entries"
            ),
            fontsize=15,
            y=0.995,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        outdir = Path(outdir)
        png = outdir / "FINAL_FOUR_PANEL__inverse_noise_model_comparison.png"
        svg = outdir / "FINAL_FOUR_PANEL__inverse_noise_model_comparison.svg"
        pdf = outdir / "FINAL_FOUR_PANEL__inverse_noise_model_comparison.pdf"

        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")

        print("[saved]", os.path.basename(png))
        print("[saved]", os.path.basename(svg))
        print("[saved]", os.path.basename(pdf))

        plt.show()

        return bin_df


    # ============================================================
    # RUN ALL PRECOMPUTED DATASETS
    # ============================================================

    run_outdir = ensure_dir(Path(OUTDIR))

    dataset_dirs = find_dataset_dirs()

    all_perpert = []
    all_gene_fp = []
    all_summaries = []
    errors = {}

    for i, ds_dir in enumerate(dataset_dirs):
        dataset = dataset_name_from_precomp_dir(ds_dir)

        try:
            perpert_df, gene_fp_df, summary = run_one_dataset(
                ds_dir=ds_dir,
                outdir=run_outdir,
                seed=SEED + i,
            )

            all_perpert.append(perpert_df)
            all_gene_fp.append(gene_fp_df)
            all_summaries.append(summary)

        except Exception as e:
            errors[dataset] = {
                "precomp_dir": str(ds_dir),
                "error": repr(e),
            }

            print("\n" + "!" * 100)
            print(f"[ERROR] Failed on {dataset}")
            print(repr(e))
            print("!" * 100 + "\n")

        gc.collect()

    if len(all_perpert) == 0:
        errors_path = Path(run_outdir) / "ALL_DATASETS__errors.json"
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)

        raise RuntimeError(
            "No datasets completed successfully. "
            f"Errors saved to {errors_path}"
        )

    all_perpert_df = pd.concat(all_perpert, ignore_index=True)
    all_gene_fp_df = pd.concat(all_gene_fp, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    all_perpert_csv = Path(run_outdir) / "ALL_DATASETS__perpert_inverse_metrics.csv"
    all_gene_fp_csv = Path(run_outdir) / "ALL_DATASETS__gene_false_positive_variance_bias.csv"
    summary_csv = Path(run_outdir) / "ALL_DATASETS__summary.csv"
    summary_json = Path(run_outdir) / "ALL_DATASETS__summary.json"
    errors_json = Path(run_outdir) / "ALL_DATASETS__errors.json"

    all_perpert_df.to_csv(all_perpert_csv, index=False)
    all_gene_fp_df.to_csv(all_gene_fp_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(summary_json, "w") as f:
        json.dump(all_summaries, f, indent=2)

    with open(errors_json, "w") as f:
        json.dump(errors, f, indent=2)

    print("[saved]", os.path.basename(all_perpert_csv))
    print("[saved]", os.path.basename(all_gene_fp_csv))
    print("[saved]", os.path.basename(summary_csv))
    print("[saved]", os.path.basename(summary_json))
    print("[saved]", os.path.basename(errors_json))

    print("\nAggregate summary:")
    print(f"  datasets attempted:       {len(dataset_dirs)}")
    print(f"  datasets completed:       {summary_df.shape[0]}")
    print(f"  datasets failed:          {len(errors)}")
    print(f"  scored perturbations:     {all_perpert_df.shape[0]}")
    print(f"  median rankpct isotropic: {np.nanmedian(all_perpert_df['rankpct_iso']):.4g}")
    print(f"  median rankpct CLT:       {np.nanmedian(all_perpert_df['rankpct_clt']):.4g}")
    print(f"  fraction CLT rank better: {np.nanmean(all_perpert_df['rankpct_clt'] < all_perpert_df['rankpct_iso']):.4f}")

    if np.any(np.isfinite(all_perpert_df["neglog10p_iso"])):
        print(f"  median -log10p iso:       {np.nanmedian(all_perpert_df['neglog10p_iso']):.4g}")
        print(f"  median -log10p CLT:       {np.nanmedian(all_perpert_df['neglog10p_clt']):.4g}")
        print(f"  fraction CLT p stronger:  {np.nanmean(all_perpert_df['neglog10p_clt'] > all_perpert_df['neglog10p_iso']):.4f}")

    for K in K_GRID:
        rec_iso = np.mean(all_perpert_df["rank_iso"].values <= K)
        rec_clt = np.mean(all_perpert_df["rank_clt"].values <= K)
        print(f"  top-{K:<3d}: iso={rec_iso:.4f}, CLT={rec_clt:.4f}, delta={rec_clt - rec_iso:+.4f}")

    bin_df = make_final_four_panel_figure(
        all_perpert_df=all_perpert_df,
        all_gene_fp_df=all_gene_fp_df,
        summary_df=summary_df,
        outdir=run_outdir,
    )

    bin_csv = Path(run_outdir) / "FINAL_FIGURE__variance_bias_binned_values.csv"
    bin_df.to_csv(bin_csv, index=False)
    print("[saved]", os.path.basename(bin_csv))

    print("\nDONE.")
    print("All outputs saved under:")
    print(os.path.basename(run_outdir))


def run_inverse_from_raw():
    global DATA_PATHS, PERT_KEY, CONTROL_LABEL, SEP, EXPRESSION_THRESHOLD, MIN_SAMPLES
    global COV_SHRINK0, COV_SHRINKU, JITTER, COV_MAX_CELLS_PER_GROUP
    global LOGTAU2_BOUNDS, EB_GRID_N, PLATEAU_DELTA, PLATEAU_PREFER, USE_PLATEAU_FOR_SCORING
    global N_PARAM_NULL_REPS, K_GRID, FP_TOPK, VAR_BINS, C_ISO, C_CLT, C_BASE, SEED

    DATA_PATHS = [
        "NormanWeissman2019_filtered.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]

    DATA_PATHS = list(dict.fromkeys(DATA_PATHS))


    PERT_KEY = "perturbation"
    CONTROL_LABEL = "control"
    SEP = "_"

    EXPRESSION_THRESHOLD = 1.0
    MIN_SAMPLES = 100

    COV_SHRINK0 = 1e-3
    COV_SHRINKU = 5e-2
    JITTER = 1e-8
    COV_MAX_CELLS_PER_GROUP = 4000

    # Empirical Bayes prior fit
    LOGTAU2_BOUNDS = (-2, -1)
    EB_GRID_N = 250
    PLATEAU_DELTA = 1.92
    PLATEAU_PREFER = "largest"
    USE_PLATEAU_FOR_SCORING = True

    # Null-calibrated p-values
    N_PARAM_NULL_REPS = 100

    # Top-K recovery panel
    K_GRID = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300], dtype=int)

    # False-positive variance-bias panel
    FP_TOPK = 100
    VAR_BINS = np.linspace(0, 1, 21)

    # Plotting
    C_ISO = "blue"
    C_CLT = "purple"
    C_BASE = "#9e9e9e"

    SEED = 0


    # ============================================================
    # BASIC HELPERS
    # ============================================================

    def to_dense(X):
        if issparse(X):
            return X.toarray()
        return np.asarray(X)


    def _symmetrize(A):
        return 0.5 * (A + A.T)


    def _shrink_cov(S, shrink=1e-3):
        S = _symmetrize(S)
        dbar = float(np.mean(np.diag(S))) if S.size else 1.0
        return (1.0 - shrink) * S + shrink * dbar * np.eye(S.shape[0], dtype=S.dtype)


    def _eig_psd(S, jitter=1e-8):
        S = _symmetrize(S) + jitter * np.eye(S.shape[0], dtype=S.dtype)
        lam, V = np.linalg.eigh(S)
        lam = np.maximum(lam, jitter)
        return lam, V


    def _subsample_rows(X, max_rows, rng):
        n = X.shape[0]
        if max_rows is None or max_rows <= 0 or n <= max_rows:
            return X
        idx = rng.choice(n, size=max_rows, replace=False)
        return X[idx]


    def _mean_axis0(X):
        return np.asarray(X.mean(axis=0)).ravel()


    def _cov_rowvar_false(X):
        if X.shape[0] <= 1:
            return np.eye(X.shape[1], dtype=np.float64)
        return np.cov(X, rowvar=False)


    def _safe_gene_name_array(adata):
        adata.var_names = adata.var_names.astype(str)
        adata.var_names_make_unique()
        return np.array(adata.var_names.tolist(), dtype=str)


    def pert_to_gene_safe(pert: str) -> str:
        p = str(pert).strip()

        p = re.sub(r"([_\-\s]+)(KD|KO|OE|overexp|overexpression)$", "", p, flags=re.IGNORECASE)
        p = re.sub(r"^(sg)(?=[A-Z0-9])", "", p)
        p = re.sub(r"^(sgRNA|gRNA|sgrna|grna|sg)([_\-\s]+)", "", p, flags=re.IGNORECASE)

        for s in ["_", "+", "-", "|", " "]:
            if s in p:
                p = p.split(s)[0]
                break

        return p


    def _resolve_target_gene(pert, gene_set):
        parsed = pert_to_gene_safe(pert)

        if parsed in gene_set:
            return parsed

        if str(pert) in gene_set:
            return str(pert)

        return None


    def _rank_of_target(scores, target_idx):
        s = np.asarray(scores).ravel()
        st = float(s[target_idx])
        rank = 1 + int(np.sum(s > st))
        rank_pct = rank / float(s.size)
        return rank, rank_pct, st


    def _topk_indices_excluding_target(scores, target_idx, k):
        scores = np.asarray(scores).ravel()
        G = scores.size

        k_eff = min(int(k), G - 1)
        if k_eff <= 0:
            return np.array([], dtype=int)

        tmp = scores.copy()
        tmp[int(target_idx)] = -np.inf

        idx = np.argpartition(-tmp, kth=k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-tmp[idx])]
        return idx.astype(int)


    def _percentile_rank_values(x):
        x = np.asarray(x, dtype=float).ravel()
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x), dtype=float)
        if len(x) <= 1:
            return np.zeros_like(x, dtype=float)
        return ranks / float(len(x) - 1)


    def _safe_neglog10p(p):
        p = np.asarray(p, dtype=float)
        p = np.maximum(p, 1e-300)
        return -np.log10(p)


    def _axis_equal_with_diag(ax, x, y, lo=0.0, hi=1.0):
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color=C_BASE)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)


    # ============================================================
    # EB OBJECTIVES
    # ============================================================

    def _nll_diag_noise(log_tau2, lam, Y_list, H_list, jitter=1e-12):
        """
        Generic diagonal-noise EB objective in Sigma0 eigenbasis.

        Model:
            y_i = lambda_i u_i + eps_i
            eps_i ~ N(0, h_i)
            u_i ~ N(0, tau^2)

        Marginal:
            y_i ~ N(0, tau^2 lambda_i^2 + h_i)
        """
        tau2 = np.exp(log_tau2)
        lam2 = lam * lam

        nll = 0.0

        for y, h in zip(Y_list, H_list):
            h = np.maximum(np.asarray(h).ravel(), jitter)
            C = tau2 * lam2 + h
            C = np.maximum(C, jitter)
            nll += 0.5 * np.sum(np.log(C) + (y * y) / C)

        return float(nll)


    def _pick_plateau_logtau2(grid, nll_curve, delta, prefer="largest"):
        grid = np.asarray(grid, dtype=float)
        nll_curve = np.asarray(nll_curve, dtype=float)

        jmin = int(np.argmin(nll_curve))
        nll_min = float(nll_curve[jmin])

        ok = np.where(nll_curve <= nll_min + float(delta))[0]

        if ok.size == 0:
            return float(grid[jmin]), nll_min, nll_min

        j = int(ok[-1] if prefer == "largest" else ok[0])
        return float(grid[j]), nll_min, float(nll_curve[j])


    def fit_tau2_EB_with_plateau(
        obj,
        logtau2_bounds=(-2, -1),
        grid_n=250,
        plateau_delta=1.92,
        plateau_prefer="largest",
    ):
        res = minimize_scalar(obj, bounds=logtau2_bounds, method="bounded")

        logtau2_opt = float(res.x)
        tau2_opt = float(np.exp(logtau2_opt))
        nll_opt = float(res.fun)

        lo, hi = logtau2_bounds
        grid = np.linspace(lo, hi, int(grid_n))
        nll_curve = np.array([obj(g) for g in grid], dtype=float)

        logtau2_pl, nll_min, nll_pl = _pick_plateau_logtau2(
            grid,
            nll_curve,
            delta=float(plateau_delta),
            prefer=plateau_prefer,
        )

        tau2_pl = float(np.exp(logtau2_pl))

        return {
            "tau2_opt": tau2_opt,
            "logtau2_opt": logtau2_opt,
            "nll_opt": nll_opt,
            "tau2_plateau": tau2_pl,
            "logtau2_plateau": logtau2_pl,
            "nll_plateau": float(nll_pl),
            "grid": grid,
            "nll_curve": nll_curve,
        }


    # ============================================================
    # POSTERIOR SCORES
    # ============================================================

    def posterior_score_diag_noise(lam, V, V2, y, tau2, h, jitter=1e-12):
        """
        Posterior over u in eigenbasis:
            y_i = lambda_i u_i + eps_i
            eps_i ~ N(0, h_i)
            u_i ~ N(0, tau^2)

        Posterior:
            Var(u_i | y_i) = 1 / (lambda_i^2 / h_i + 1/tau^2)
            Mean(u_i | y_i) = Var * lambda_i y_i / h_i

        Gene-space score:
            score_g = max(|mu_g + std_g|, |mu_g - std_g|)
        """
        h = np.maximum(np.asarray(h).ravel(), jitter)

        prec = (lam * lam) / h + (1.0 / tau2)
        var_i = 1.0 / np.maximum(prec, jitter)
        mean_i = var_i * (lam * y / h)

        mu = V @ mean_i
        diag_cov = V2 @ var_i
        std = np.sqrt(np.maximum(diag_cov, 0.0))

        return np.maximum(np.abs(mu + std), np.abs(mu - std))


    def target_score_diag_noise(lam, V_row, V2_row, y, tau2, h, jitter=1e-12):
        """
        Same as posterior_score_diag_noise, but compute only one target gene score.
        Used for fast null p-values.
        """
        h = np.maximum(np.asarray(h).ravel(), jitter)

        prec = (lam * lam) / h + (1.0 / tau2)
        var_i = 1.0 / np.maximum(prec, jitter)
        mean_i = var_i * (lam * y / h)

        mu_g = float(V_row @ mean_i)
        var_g = float(V2_row @ var_i)
        std_g = math.sqrt(max(var_g, 0.0))

        return max(abs(mu_g + std_g), abs(mu_g - std_g))


    # ============================================================
    # MAIN DATASET ROUTINE
    # ============================================================

    def run_one_dataset(data_path, outdir, seed=0):
        rng = np.random.default_rng(seed)

        dataset_name = os.path.basename(data_path).replace(".h5ad", "")

        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        adata = ad.read_h5ad(data_path)
        _safe_gene_name_array(adata)

        if PERT_KEY not in adata.obs.columns:
            raise ValueError(f"{data_path} does not contain obs['{PERT_KEY}'].")

        # --------------------------------------------------------
        # Gene filtering
        # --------------------------------------------------------
        gene_means = adata.X.mean(axis=0).A1 if issparse(adata.X) else np.asarray(adata.X).mean(axis=0)
        valid_genes = set(adata.var_names[np.where(gene_means >= float(EXPRESSION_THRESHOLD))[0]])

        all_perturbed_genes = set()
        var_name_set = set(adata.var_names)

        for pert in adata.obs[PERT_KEY].astype(str).unique():
            if pert == CONTROL_LABEL:
                continue

            for g in str(pert).split(SEP):
                if g in var_name_set:
                    all_perturbed_genes.add(g)

            parsed = pert_to_gene_safe(pert)
            if parsed in var_name_set:
                all_perturbed_genes.add(parsed)

        keep_genes = list(valid_genes | all_perturbed_genes)
        adata = adata[:, adata.var_names.isin(keep_genes)].copy()

        gene_names = _safe_gene_name_array(adata)
        gene_set = set(gene_names.tolist())
        G = len(gene_names)

        # --------------------------------------------------------
        # Perturbation filtering
        # --------------------------------------------------------
        pert_counts = adata.obs[PERT_KEY].astype(str).value_counts()
        valid_perts = pert_counts[pert_counts >= int(MIN_SAMPLES)].index.astype(str).tolist()

        adata = adata[adata.obs[PERT_KEY].astype(str).isin(valid_perts)].copy()
        obs_pert = adata.obs[PERT_KEY].astype(str).values

        if CONTROL_LABEL not in set(obs_pert):
            raise ValueError(f"Control label '{CONTROL_LABEL}' not found in {dataset_name}.")

        perts_all = [p for p in np.unique(obs_pert) if p != CONTROL_LABEL]
        perts_single = [p for p in perts_all if SEP not in p]

        scored_perts = []
        target_genes = []
        target_indices = []

        for p in perts_single:
            tg = _resolve_target_gene(p, gene_set)
            if tg is None:
                continue

            idx = np.where(gene_names == tg)[0]
            if idx.size == 0:
                continue

            scored_perts.append(str(p))
            target_genes.append(str(tg))
            target_indices.append(int(idx[0]))

        print(f"n_cells after filtering:     {adata.n_obs}")
        print(f"n_genes after filtering:     {G}")
        print(f"n_perts excl control:        {len(perts_all)}")
        print(f"n_single perts:              {len(perts_single)}")
        print(f"n_scored single perts:       {len(scored_perts)}")

        if len(scored_perts) == 0:
            raise ValueError(f"No scored perturbations found for {dataset_name}.")

        # --------------------------------------------------------
        # Control mean and covariance
        # --------------------------------------------------------
        X0_all = to_dense(adata[obs_pert == CONTROL_LABEL].X).astype(np.float32, copy=False)
        n0_full = int(X0_all.shape[0])

        X0_cov = _subsample_rows(X0_all, COV_MAX_CELLS_PER_GROUP, rng).astype(np.float64, copy=False)
        n0_cov = int(X0_cov.shape[0])

        X0_mean = _mean_axis0(X0_all)

        Sigma0 = cipher.compute_covariance(X0_cov, shrink=COV_SHRINK0)
        lam, V = _eig_psd(Sigma0, jitter=JITTER)
        V2 = V * V

        # Control variance per gene in original gene coordinates
        control_gene_var = np.diag(Sigma0).astype(float)
        control_gene_var_pct = _percentile_rank_values(control_gene_var)

        print(f"n0_full={n0_full}, n0_cov={n0_cov}")

        # --------------------------------------------------------
        # Build perturbation records
        # --------------------------------------------------------
        records = []
        Y_list = []
        H_clt_list = []
        H_iso_list = []

        for pert, target_gene, target_idx in tqdm(
            list(zip(scored_perts, target_genes, target_indices)),
            desc=f"{dataset_name}: build records",
            leave=False,
        ):
            Xp_all = to_dense(adata[obs_pert == pert].X).astype(np.float32, copy=False)
            n_p = int(Xp_all.shape[0])

            Xp_cov = _subsample_rows(Xp_all, COV_MAX_CELLS_PER_GROUP, rng).astype(np.float64, copy=False)

            dX = _mean_axis0(Xp_all) - X0_mean
            y = V.T @ dX

            Sigma_p = cipher.compute_covariance(Xp_cov, shrink=COV_SHRINKU)

            # diag(V^T Sigma_p V) without storing V^T Sigma_p V
            Sigma_p_V = Sigma_p @ V
            diag_VtSpV = np.sum(V * Sigma_p_V, axis=0)

            # Diagonal CLT noise in Sigma0 eigenbasis
            h_clt = (lam / max(n0_full, 1)) + (diag_VtSpV / max(n_p, 1))
            h_clt = np.maximum(h_clt, JITTER)

            # Trace-matched isotropic scalar noise
            h_iso_scalar = float(np.mean(h_clt))
            h_iso = np.full_like(h_clt, fill_value=max(h_iso_scalar, JITTER))

            rec = {
                "dataset": dataset_name,
                "perturbation": str(pert),
                "target_gene": str(target_gene),
                "target_idx": int(target_idx),
                "n_p": int(n_p),
                "y": y.astype(np.float64),
                "h_clt": h_clt.astype(np.float64),
                "h_iso": h_iso.astype(np.float64),
            }

            records.append(rec)
            Y_list.append(rec["y"])
            H_clt_list.append(rec["h_clt"])
            H_iso_list.append(rec["h_iso"])

        # --------------------------------------------------------
        # Fit EB tau2 separately for isotropic and CLT models
        # --------------------------------------------------------
        print("[EB] fitting tau2 for isotropic and CLT models")

        eb_iso = fit_tau2_EB_with_plateau(
            obj=lambda logt: _nll_diag_noise(logt, lam, Y_list, H_iso_list),
            logtau2_bounds=LOGTAU2_BOUNDS,
            grid_n=EB_GRID_N,
            plateau_delta=PLATEAU_DELTA,
            plateau_prefer=PLATEAU_PREFER,
        )

        eb_clt = fit_tau2_EB_with_plateau(
            obj=lambda logt: _nll_diag_noise(logt, lam, Y_list, H_clt_list),
            logtau2_bounds=LOGTAU2_BOUNDS,
            grid_n=EB_GRID_N,
            plateau_delta=PLATEAU_DELTA,
            plateau_prefer=PLATEAU_PREFER,
        )

        tau2_iso = eb_iso["tau2_plateau"] if USE_PLATEAU_FOR_SCORING else eb_iso["tau2_opt"]
        tau2_clt = eb_clt["tau2_plateau"] if USE_PLATEAU_FOR_SCORING else eb_clt["tau2_opt"]

        print(f"iso tau2 used = {tau2_iso:.5g}")
        print(f"CLT tau2 used = {tau2_clt:.5g}")

        # --------------------------------------------------------
        # Real perturbation inverse scoring
        # --------------------------------------------------------
        print("[real] scoring perturbations")

        fp_count_iso = np.zeros(G, dtype=np.float64)
        fp_count_clt = np.zeros(G, dtype=np.float64)

        perpert_rows = []

        for rec in tqdm(records, desc=f"{dataset_name}: real scoring", leave=False):
            target_idx = int(rec["target_idx"])
            y = rec["y"]

            score_iso = posterior_score_diag_noise(
                lam=lam,
                V=V,
                V2=V2,
                y=y,
                tau2=tau2_iso,
                h=rec["h_iso"],
            )

            score_clt = posterior_score_diag_noise(
                lam=lam,
                V=V,
                V2=V2,
                y=y,
                tau2=tau2_clt,
                h=rec["h_clt"],
            )

            rank_iso, rankpct_iso, target_score_iso = _rank_of_target(score_iso, target_idx)
            rank_clt, rankpct_clt, target_score_clt = _rank_of_target(score_clt, target_idx)

            top_iso = _topk_indices_excluding_target(score_iso, target_idx, FP_TOPK)
            top_clt = _topk_indices_excluding_target(score_clt, target_idx, FP_TOPK)

            fp_count_iso[top_iso] += 1.0
            fp_count_clt[top_clt] += 1.0

            perpert_rows.append({
                "dataset": dataset_name,
                "perturbation": rec["perturbation"],
                "target_gene": rec["target_gene"],
                "target_idx": target_idx,
                "n_p": int(rec["n_p"]),

                "rank_iso": int(rank_iso),
                "rank_clt": int(rank_clt),
                "rankpct_iso": float(rankpct_iso),
                "rankpct_clt": float(rankpct_clt),
                "rankpct_improvement_iso_minus_clt": float(rankpct_iso - rankpct_clt),

                "target_score_iso": float(target_score_iso),
                "target_score_clt": float(target_score_clt),
            })

        perpert_df = pd.DataFrame(perpert_rows)

        # --------------------------------------------------------
        # Null-calibrated target p-values
        # --------------------------------------------------------
        print("[null] computing target-gene null p-values")

        null_target_scores_iso = np.zeros((len(records), N_PARAM_NULL_REPS), dtype=np.float32)
        null_target_scores_clt = np.zeros((len(records), N_PARAM_NULL_REPS), dtype=np.float32)

        for i, rec in enumerate(tqdm(records, desc=f"{dataset_name}: param null", leave=False)):
            target_idx = int(rec["target_idx"])
            V_row = V[target_idx, :]
            V2_row = V2[target_idx, :]

            h_clt = rec["h_clt"]
            h_iso = rec["h_iso"]

            for b in range(N_PARAM_NULL_REPS):
                # No-signal draw from CLT sampling-noise null
                y_null = np.sqrt(h_clt) * rng.normal(size=G)

                null_target_scores_iso[i, b] = target_score_diag_noise(
                    lam=lam,
                    V_row=V_row,
                    V2_row=V2_row,
                    y=y_null,
                    tau2=tau2_iso,
                    h=h_iso,
                )

                null_target_scores_clt[i, b] = target_score_diag_noise(
                    lam=lam,
                    V_row=V_row,
                    V2_row=V2_row,
                    y=y_null,
                    tau2=tau2_clt,
                    h=h_clt,
                )

        real_target_score_iso = perpert_df["target_score_iso"].values.astype(float)
        real_target_score_clt = perpert_df["target_score_clt"].values.astype(float)

        p_iso = (
            1.0 + np.sum(null_target_scores_iso >= real_target_score_iso[:, None], axis=1)
        ) / float(N_PARAM_NULL_REPS + 1)

        p_clt = (
            1.0 + np.sum(null_target_scores_clt >= real_target_score_clt[:, None], axis=1)
        ) / float(N_PARAM_NULL_REPS + 1)

        perpert_df["param_p_iso"] = p_iso
        perpert_df["param_p_clt"] = p_clt
        perpert_df["neglog10p_iso"] = _safe_neglog10p(p_iso)
        perpert_df["neglog10p_clt"] = _safe_neglog10p(p_clt)
        perpert_df["delta_neglog10p_clt_minus_iso"] = perpert_df["neglog10p_clt"] - perpert_df["neglog10p_iso"]

        # --------------------------------------------------------
        # Gene-level false-positive variance bias table
        # --------------------------------------------------------
        n_scored = max(len(records), 1)

        gene_fp_df = pd.DataFrame({
            "dataset": dataset_name,
            "gene": gene_names,
            "control_variance": control_gene_var,
            "control_variance_percentile": control_gene_var_pct,
            "fp_rate_iso": fp_count_iso / float(n_scored),
            "fp_rate_clt": fp_count_clt / float(n_scored),
        })

        gene_fp_df["fp_rate_diff_iso_minus_clt"] = gene_fp_df["fp_rate_iso"] - gene_fp_df["fp_rate_clt"]

        # --------------------------------------------------------
        # Dataset summaries
        # --------------------------------------------------------
        topk_summary = {}

        for K in K_GRID:
            topk_summary[f"top{K}_iso"] = float(np.mean(perpert_df["rank_iso"].values <= K))
            topk_summary[f"top{K}_clt"] = float(np.mean(perpert_df["rank_clt"].values <= K))

        summary = {
            "dataset": dataset_name,
            "data_path": data_path,
            "n_cells": int(adata.n_obs),
            "n_genes": int(G),
            "n_control_cells": int(n0_full),
            "n_scored_perts": int(len(records)),

            "tau2_iso": float(tau2_iso),
            "tau2_clt": float(tau2_clt),

            "median_rankpct_iso": float(np.median(perpert_df["rankpct_iso"])),
            "median_rankpct_clt": float(np.median(perpert_df["rankpct_clt"])),
            "median_rankpct_improvement_iso_minus_clt": float(np.median(perpert_df["rankpct_improvement_iso_minus_clt"])),
            "fraction_rankpct_clt_better": float(np.mean(perpert_df["rankpct_clt"] < perpert_df["rankpct_iso"])),

            "median_neglog10p_iso": float(np.median(perpert_df["neglog10p_iso"])),
            "median_neglog10p_clt": float(np.median(perpert_df["neglog10p_clt"])),
            "median_delta_neglog10p_clt_minus_iso": float(np.median(perpert_df["delta_neglog10p_clt_minus_iso"])),
            "fraction_p_clt_better": float(np.mean(perpert_df["param_p_clt"] < perpert_df["param_p_iso"])),

            **topk_summary,
        }

        # --------------------------------------------------------
        # Save per-dataset outputs
        # --------------------------------------------------------
        perpert_csv = os.path.join(outdir, f"{dataset_name}__perpert_inverse_metrics.csv")
        genefp_csv = os.path.join(outdir, f"{dataset_name}__gene_false_positive_variance_bias.csv")
        summary_json = os.path.join(outdir, f"{dataset_name}__summary.json")

        perpert_df.to_csv(perpert_csv, index=False)
        gene_fp_df.to_csv(genefp_csv, index=False)

        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[saved] {os.path.basename(perpert_csv)}")
        print(f"[saved] {os.path.basename(genefp_csv)}")
        print(f"[summary] median rankpct iso={summary['median_rankpct_iso']:.4g}, CLT={summary['median_rankpct_clt']:.4g}")
        print(f"[summary] frac CLT rank better={summary['fraction_rankpct_clt_better']:.3f}")
        print(f"[summary] median delta -log10p CLT-iso={summary['median_delta_neglog10p_clt_minus_iso']:.3g}")

        # Clean heavy objects
        del adata, X0_all, X0_cov, Sigma0, V, V2
        gc.collect()

        return perpert_df, gene_fp_df, summary


    # ============================================================
    # AGGREGATE PLOTTING
    # ============================================================

    def make_final_four_panel_figure(all_perpert_df, all_gene_fp_df, summary_df, outdir):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        axA = axes[0, 0]
        axB = axes[0, 1]
        axC = axes[1, 0]
        axD = axes[1, 1]

        # --------------------------------------------------------
        # A. Target-rank scatter
        # --------------------------------------------------------
        x = all_perpert_df["rankpct_iso"].values.astype(float)
        y = all_perpert_df["rankpct_clt"].values.astype(float)

        axA.scatter(x, y, s=18, alpha=0.45, edgecolor="none")
        _axis_equal_with_diag(axA, x, y, lo=0.0, hi=1.0)

        frac_clt_better = float(np.mean(y < x))
        med_improve = float(np.median(x - y))

        axA.set_xlabel("Target rank percentile, isotropic")
        axA.set_ylabel("Target rank percentile, CLT")
        axA.set_title("A. CLT improves target-gene ranks\nbelow diagonal = CLT better")

        axA.text(
            0.03,
            0.97,
            f"fraction CLT better = {frac_clt_better:.3f}\nmedian improvement = {med_improve:.3g}",
            transform=axA.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, linewidth=0.5),
        )

        # --------------------------------------------------------
        # B. Top-K recovery
        # --------------------------------------------------------
        topk_iso = []
        topk_clt = []

        for K in K_GRID:
            topk_iso.append(float(np.mean(all_perpert_df["rank_iso"].values <= K)))
            topk_clt.append(float(np.mean(all_perpert_df["rank_clt"].values <= K)))

        topk_iso = np.array(topk_iso)
        topk_clt = np.array(topk_clt)

        axB.plot(K_GRID, topk_iso, marker="o", linewidth=2.2, color=C_ISO, label="isotropic")
        axB.plot(K_GRID, topk_clt, marker="o", linewidth=2.2, color=C_CLT, label="CLT")

        axB.set_xscale("log")
        axB.set_xlabel("Top K genes")
        axB.set_ylabel("Fraction of perturbations recovered")
        axB.set_ylim(0, 1.02)
        axB.set_title("B. Practical top-K target recovery")
        axB.legend(frameon=False)

        for K_show in [10, 50, 100]:
            if K_show in set(K_GRID):
                i = int(np.where(K_GRID == K_show)[0][0])
                axB.text(
                    K_show,
                    topk_clt[i],
                    f" Δ={topk_clt[i] - topk_iso[i]:+.2f}",
                    fontsize=9,
                    va="bottom",
                    ha="left",
                )

        # --------------------------------------------------------
        # C. Null-calibrated p-value scatter
        # --------------------------------------------------------
        x = all_perpert_df["neglog10p_iso"].values.astype(float)
        y = all_perpert_df["neglog10p_clt"].values.astype(float)

        ok = np.isfinite(x) & np.isfinite(y)

        x = x[ok]
        y = y[ok]

        maxv = max(1.0, float(np.nanpercentile(np.concatenate([x, y]), 99.5)))
        maxv = min(maxv * 1.08, max(float(np.max(x)), float(np.max(y)), 1.0) * 1.05)

        axC.scatter(x, y, s=18, alpha=0.45, edgecolor="none")
        axC.plot([0, maxv], [0, maxv], linestyle="--", linewidth=1.2, color=C_BASE)
        axC.set_xlim(0, maxv)
        axC.set_ylim(0, maxv)

        frac_p_better = float(np.mean(y > x))
        med_delta_p = float(np.median(y - x))

        axC.set_xlabel(r"$-\log_{10} p$, isotropic")
        axC.set_ylabel(r"$-\log_{10} p$, CLT")
        axC.set_title("C. Null-calibrated target support\nabove diagonal = CLT stronger")

        axC.text(
            0.03,
            0.97,
            f"fraction CLT stronger = {frac_p_better:.3f}\nmedian Δ = {med_delta_p:.3g}",
            transform=axC.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, linewidth=0.5),
        )

        # --------------------------------------------------------
        # D. False-positive variance bias
        # --------------------------------------------------------
        df_fp = all_gene_fp_df.copy()
        df_fp = df_fp[np.isfinite(df_fp["control_variance_percentile"])]
        df_fp["var_bin"] = pd.cut(
            df_fp["control_variance_percentile"],
            bins=VAR_BINS,
            include_lowest=True,
            labels=False,
        )

        bin_rows = []

        for b in sorted(df_fp["var_bin"].dropna().unique()):
            sub = df_fp[df_fp["var_bin"] == b]

            if len(sub) == 0:
                continue

            bin_rows.append({
                "bin": int(b),
                "x": float(sub["control_variance_percentile"].mean()),
                "fp_iso_mean": float(sub["fp_rate_iso"].mean()),
                "fp_clt_mean": float(sub["fp_rate_clt"].mean()),
                "fp_iso_sem": float(sub["fp_rate_iso"].std(ddof=1) / math.sqrt(max(len(sub), 1))),
                "fp_clt_sem": float(sub["fp_rate_clt"].std(ddof=1) / math.sqrt(max(len(sub), 1))),
                "n_genes": int(len(sub)),
            })

        bin_df = pd.DataFrame(bin_rows)

        rho_iso, p_iso_corr = spearmanr(
            df_fp["control_variance_percentile"].values,
            df_fp["fp_rate_iso"].values,
        )

        rho_clt, p_clt_corr = spearmanr(
            df_fp["control_variance_percentile"].values,
            df_fp["fp_rate_clt"].values,
        )

        axD.errorbar(
            bin_df["x"],
            bin_df["fp_iso_mean"],
            yerr=bin_df["fp_iso_sem"],
            marker="o",
            linewidth=2.2,
            capsize=2,
            color=C_ISO,
            label=f"isotropic, ρ={rho_iso:.2f}",
        )

        axD.errorbar(
            bin_df["x"],
            bin_df["fp_clt_mean"],
            yerr=bin_df["fp_clt_sem"],
            marker="o",
            linewidth=2.2,
            capsize=2,
            color=C_CLT,
            label=f"CLT, ρ={rho_clt:.2f}",
        )

        axD.set_xlabel("Control variance percentile")
        axD.set_ylabel(f"False-positive top-{FP_TOPK} frequency")
        axD.set_title("D. Variance-driven false positives\nlower/flatter = less noise bias")
        axD.legend(frameon=False)

        # --------------------------------------------------------
        # Overall title and save
        # --------------------------------------------------------
        n_datasets = summary_df.shape[0]
        n_perts = all_perpert_df.shape[0]
        n_gene_entries = all_gene_fp_df.shape[0]

        fig.suptitle(
            (
                "CLT noise improves inverse perturbation inference while reducing variance-driven false positives\n"
                f"{n_datasets} datasets, {n_perts} scored perturbations, {n_gene_entries} gene-level false-positive entries"
            ),
            fontsize=15,
            y=0.995,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        png = os.path.join(outdir, "FINAL_FOUR_PANEL__inverse_noise_model_comparison.png")
        svg = os.path.join(outdir, "FINAL_FOUR_PANEL__inverse_noise_model_comparison.svg")
        pdf = os.path.join(outdir, "FINAL_FOUR_PANEL__inverse_noise_model_comparison.pdf")

        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")

        print("[saved]", os.path.basename(png))
        print("[saved]", os.path.basename(svg))
        print("[saved]", os.path.basename(pdf))

        plt.show()

        return bin_df


    # ============================================================
    # RUN ALL DATASETS
    # ============================================================

    run_outdir = OUTDIR
    os.makedirs(run_outdir, exist_ok=True)

    all_perpert = []
    all_gene_fp = []
    all_summaries = []

    for i, name in enumerate(DATA_PATHS):
        data_path = os.path.join(DATA_DIR, name)
        try:
            perpert_df, gene_fp_df, summary = run_one_dataset(
                data_path=data_path,
                outdir=run_outdir,
                seed=SEED + i,
            )

            all_perpert.append(perpert_df)
            all_gene_fp.append(gene_fp_df)
            all_summaries.append(summary)

        except Exception as e:
            print("\n" + "!" * 100)
            print(f"[ERROR] Failed on {data_path}")
            print(repr(e))
            print("!" * 100 + "\n")

    if len(all_perpert) == 0:
        raise RuntimeError("No datasets completed successfully.")

    all_perpert_df = pd.concat(all_perpert, ignore_index=True)
    all_gene_fp_df = pd.concat(all_gene_fp, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    # Save aggregate tables
    all_perpert_csv = os.path.join(run_outdir, "ALL_DATASETS__perpert_inverse_metrics.csv")
    all_gene_fp_csv = os.path.join(run_outdir, "ALL_DATASETS__gene_false_positive_variance_bias.csv")
    summary_csv = os.path.join(run_outdir, "ALL_DATASETS__summary.csv")
    summary_json = os.path.join(run_outdir, "ALL_DATASETS__summary.json")

    all_perpert_df.to_csv(all_perpert_csv, index=False)
    all_gene_fp_df.to_csv(all_gene_fp_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(summary_json, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("[saved]", os.path.basename(all_perpert_csv))
    print("[saved]", os.path.basename(all_gene_fp_csv))
    print("[saved]", os.path.basename(summary_csv))
    print("[saved]", os.path.basename(summary_json))

    # Print compact aggregate summary
    print("\nAggregate summary:")
    print(f"  datasets completed: {summary_df.shape[0]}")
    print(f"  scored perturbations: {all_perpert_df.shape[0]}")
    print(f"  median rankpct isotropic: {np.median(all_perpert_df['rankpct_iso']):.4g}")
    print(f"  median rankpct CLT:       {np.median(all_perpert_df['rankpct_clt']):.4g}")
    print(f"  fraction CLT rank better: {np.mean(all_perpert_df['rankpct_clt'] < all_perpert_df['rankpct_iso']):.4f}")
    print(f"  median -log10p iso:       {np.median(all_perpert_df['neglog10p_iso']):.4g}")
    print(f"  median -log10p CLT:       {np.median(all_perpert_df['neglog10p_clt']):.4g}")
    print(f"  fraction CLT p stronger:  {np.mean(all_perpert_df['neglog10p_clt'] > all_perpert_df['neglog10p_iso']):.4f}")

    for K in K_GRID:
        rec_iso = np.mean(all_perpert_df["rank_iso"].values <= K)
        rec_clt = np.mean(all_perpert_df["rank_clt"].values <= K)
        print(f"  top-{K:<3d}: iso={rec_iso:.4f}, CLT={rec_clt:.4f}, delta={rec_clt - rec_iso:+.4f}")

    # Make final 4-panel figure
    bin_df = make_final_four_panel_figure(
        all_perpert_df=all_perpert_df,
        all_gene_fp_df=all_gene_fp_df,
        summary_df=summary_df,
        outdir=run_outdir,
    )

    bin_csv = os.path.join(run_outdir, "FINAL_FIGURE__variance_bias_binned_values.csv")
    bin_df.to_csv(bin_csv, index=False)
    print("[saved]", os.path.basename(bin_csv))

    print("\nDONE.")
    print("All outputs saved under:")
    print(os.path.basename(run_outdir))


def make_two_panel_bd():
    global K_GRID, FP_TOPK, N_VAR_BINS, VAR_BINS, C_ISO, C_CLT, C_BASE
    global SAVE_FIG, SHOW_FIG, SAVE_SVG, DPI



    # Top-K curve settings
    K_GRID = np.arange(1,100)

    # False-positive panel settings
    FP_TOPK = 100

    # More bins than before:
    # Original used 21 edges -> 20 bins
    # Here use 41 edges -> 40 bins
    N_VAR_BINS = 100
    VAR_BINS = np.linspace(0.0, 1., N_VAR_BINS + 1)

    # Plot aesthetics
    C_ISO = "blue"
    C_CLT = "purple"
    C_BASE = "#9e9e9e"

    SAVE_FIG = True
    SHOW_FIG = True
    SAVE_SVG = True
    DPI = 300

    # -----------------------------
    # FIND RUN DIRECTORY
    # -----------------------------

    RUN_DIR = OUTDIR

    perpert_csv = os.path.join(RUN_DIR, "ALL_DATASETS__perpert_inverse_metrics.csv")
    gene_fp_csv = os.path.join(RUN_DIR, "ALL_DATASETS__gene_false_positive_variance_bias.csv")

    if not os.path.exists(perpert_csv):
        raise FileNotFoundError(perpert_csv)
    if not os.path.exists(gene_fp_csv):
        raise FileNotFoundError(gene_fp_csv)

    all_perpert_df = pd.read_csv(perpert_csv)
    all_gene_fp_df = pd.read_csv(gene_fp_csv)

    print("\nLoaded:")
    print(f"  per-pert rows: {len(all_perpert_df)}")
    print(f"  gene-level rows: {len(all_gene_fp_df)}")

    # ============================================================
    # PANEL B: TOP-K RECOVERY
    # ============================================================
    topk_iso = []
    topk_clt = []

    for K in K_GRID:
        topk_iso.append(float(np.mean(all_perpert_df["rank_iso"].values <= K)))
        topk_clt.append(float(np.mean(all_perpert_df["rank_clt"].values <= K)))

    topk_iso = np.array(topk_iso)
    topk_clt = np.array(topk_clt)

    # ============================================================
    # PANEL D: FALSE-POSITIVE VARIANCE BIAS WITH MORE BINS
    # ============================================================
    df_fp = all_gene_fp_df.copy()
    df_fp = df_fp[np.isfinite(df_fp["control_variance_percentile"])].copy()

    df_fp["var_bin"] = pd.cut(
        df_fp["control_variance_percentile"],
        bins=VAR_BINS,
        include_lowest=True,
        labels=False,
    )

    bin_rows = []

    for b in sorted(df_fp["var_bin"].dropna().unique()):
        sub = df_fp[df_fp["var_bin"] == b]

        if len(sub) == 0:
            continue

        n = len(sub)

        iso_mean = float(sub["fp_rate_iso"].mean())
        clt_mean = float(sub["fp_rate_clt"].mean())

        iso_sem = float(sub["fp_rate_iso"].std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        clt_sem = float(sub["fp_rate_clt"].std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0

        bin_rows.append({
            "bin": int(b),
            "x": float(sub["control_variance_percentile"].mean()),
            "fp_iso_mean": iso_mean,
            "fp_clt_mean": clt_mean,
            "fp_iso_sem": iso_sem,
            "fp_clt_sem": clt_sem,
            "n_genes": int(n),
        })

    bin_df = pd.DataFrame(bin_rows)

    rho_iso, p_iso = spearmanr(
        df_fp["control_variance_percentile"].values,
        df_fp["fp_rate_iso"].values,
    )

    rho_clt, p_clt = spearmanr(
        df_fp["control_variance_percentile"].values,
        df_fp["fp_rate_clt"].values,
    )

    # ============================================================
    # MAKE 2-PANEL FIGURE
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 11.0))

    # ------------------------------------------------------------
    # Panel B
    # ------------------------------------------------------------
    ax = axes[0]

    ax.plot(
        K_GRID,
        topk_iso,
        marker="o",
        linewidth=2.4,
        color=C_ISO,
        label="isotropic",
    )

    ax.plot(
        K_GRID,
        topk_clt,
        marker="o",
        linewidth=2.4,
        color=C_CLT,
        label="CLT",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Top K genes")
    ax.set_ylabel("Fraction of perturbations recovered")
    ax.set_ylim(0, 1.02)
    ax.set_title("B. Practical top-K target recovery")
    ax.legend(frameon=False, loc="upper left")

    for K_show in [10, 50, 100]:
        if K_show in set(K_GRID):
            i = int(np.where(K_GRID == K_show)[0][0])
            delta = topk_clt[i] - topk_iso[i]
            ax.text(
                K_show,
                topk_clt[i],
                f"Δ={delta:+.2f}",
                fontsize=10,
                va="bottom",
                ha="left",
            )

    # ------------------------------------------------------------
    # Panel D
    # ------------------------------------------------------------
    ax = axes[1]

    ax.errorbar(
        bin_df["x"],
        bin_df["fp_iso_mean"],
        yerr=bin_df["fp_iso_sem"],
        marker="o",
        markersize=4,
        linewidth=2.0,
        capsize=2,
        color=C_ISO,
        label=f"isotropic, ρ={rho_iso:.2f}",
    )

    ax.errorbar(
        bin_df["x"],
        bin_df["fp_clt_mean"],
        yerr=bin_df["fp_clt_sem"],
        marker="o",
        markersize=4,
        linewidth=2.0,
        capsize=2,
        color=C_CLT,
        label=f"CLT, ρ={rho_clt:.2f}",
    )

    ax.set_xlabel("Control variance percentile")
    ax.set_ylabel(f"False-positive top-{FP_TOPK} frequency")
    ax.set_title(
        f"D. Variance-dependent false positives\n"
        f"lower/flatter = less noise bias ({N_VAR_BINS} bins)"
    )
    # ax.legend(frameon=False, loc="upper right")

    # Optional: make x ticks denser and clean
    ax.set_xlim(-0.01, .2)
    ax.set_xticks(np.linspace(0, .2, 11))

    # ------------------------------------------------------------
    # Final layout and save
    # ------------------------------------------------------------
    plt.tight_layout()

    out_png = os.path.join(RUN_DIR, "TWO_PANEL__B_and_D_more_variance_bins.png")
    out_svg = os.path.join(RUN_DIR, "TWO_PANEL__B_and_D_more_variance_bins.svg")
    out_csv = os.path.join(RUN_DIR, "TWO_PANEL__D_more_variance_bins_values.csv")

    if SAVE_FIG:
        fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
        if SAVE_SVG:
            fig.savefig(out_svg, bbox_inches="tight")
        bin_df.to_csv(out_csv, index=False)

    print("\nSaved:")
    print(os.path.basename(out_png))
    if SAVE_SVG:
        print(os.path.basename(out_svg))
    print(os.path.basename(out_csv))

    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)

