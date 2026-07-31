"""Generate the total-RNA / mu-axis forward diagnostic used by Figure 6's depth panels.

For every perturbation this measures how much of the mean response ``dx`` (and of the rank-1
forward prediction ``pred = alpha * Sigma[:, target]``) lies along the **control-mean axis**
``mu = control_mean`` -- i.e. the global "total-RNA" / library-scaling direction -- versus the
covariance-structured residual. It reproduces the per-perturbation table that
``scratch/functional/depth_fig6.ipynb`` wrote to
``preprocessed_raw_forward_depth_totalRNA_diagnostic/``.

Input is the forward precompute (per dataset: ``genes.npy``, ``perturbations.npy``,
``target_gene_indices.npy``, ``target_genes.npy`` and ``normalizations/<mode>/{Sigma_full_ridge.npy,
perturbation_stats.h5}``) -- the same artifact ``regenerate_sigma.py`` / the forward precompute
produce. Output: one TSV per dataset plus an aggregated table + summary under ``--out``.

CPU-only; reads the precompute, no h5ad needed.
"""
import os
import glob
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

MODES = ("raw",)
MAX_PERTS_PER_DATASET = 1000
TRAIN_FRAC = 0.5
EXCLUDE_TARGET_GENE_FROM_EVAL = True
PROJECTION_FIT_MODE = "all"          # fit the mu-axis coefficient on all genes ("all") or "train"
RNG_SEED = 0

CELL_CYCLE_GENES = {
    "MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
    "CDK1", "CCNB1", "CCNB2", "CCNA2", "AURKA", "AURKB", "BUB1", "BUB1B",
    "UBE2C", "CDC20", "TYMS", "RRM2", "HMGB2", "CENPF", "NUSAP1",
}
TRANSLATION_GROWTH_GENES = {
    "MYC", "EIF4E", "EIF4A1", "EIF4G1", "EEF1A1", "EEF2",
    "RPLP0", "RPLP1", "RPLP2", "NPM1", "FBL", "NOP56", "NOP58", "DDX21", "UBTF",
}


# ------------------------------------------------------------------ metrics
def _uncentered_r2(y, yhat, eps=1e-12):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat); y, yhat = y[ok], yhat[ok]
    if len(y) == 0:
        return np.nan
    denom = np.sum(y ** 2)
    return np.nan if denom <= eps else float(1.0 - np.sum((y - yhat) ** 2) / denom)


def _pearson_safe(y, yhat, eps=1e-12):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat); y, yhat = y[ok], yhat[ok]
    if len(y) < 3:
        return np.nan
    y = y - y.mean(); yhat = yhat - yhat.mean()
    denom = np.sqrt(np.sum(y ** 2) * np.sum(yhat ** 2))
    return np.nan if denom <= eps else float(np.sum(y * yhat) / denom)


def _cosine_safe(a, b, eps=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) == 0:
        return np.nan
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return np.nan if denom <= eps else float(np.sum(a * b) / denom)


def _fraction_norm2(part, full, eps=1e-12):
    part, full = np.asarray(part, float), np.asarray(full, float)
    ok = np.isfinite(part) & np.isfinite(full); part, full = part[ok], full[ok]
    denom = np.sum(full ** 2)
    return np.nan if denom <= eps else float(np.sum(part ** 2) / denom)


def _project_onto_axis(y, axis, fit_idx=None, eps=1e-12):
    """Split ``y`` into its component along ``axis`` (coef fit on ``fit_idx``) and the residual."""
    y, axis = np.asarray(y, float), np.asarray(axis, float)
    fit_idx = np.arange(len(y)) if fit_idx is None else np.asarray(fit_idx, int)
    ok = np.isfinite(y[fit_idx]) & np.isfinite(axis[fit_idx]); fit_idx = fit_idx[ok]
    nan = np.full_like(y, np.nan)
    if len(fit_idx) == 0:
        return nan, nan, np.nan
    denom = np.sum(axis[fit_idx] ** 2)
    if denom <= eps:
        return nan, nan, np.nan
    coef = float(np.sum(axis[fit_idx] * y[fit_idx]) / denom)
    ypar = coef * axis
    return ypar, y - ypar, coef


def _marker_masks(gene_names):
    gu = np.asarray([str(g).upper() for g in gene_names])
    return {
        "cell_cycle": np.array([g in CELL_CYCLE_GENES for g in gu], bool),
        "translation": np.array([g in TRANSLATION_GROWTH_GENES for g in gu], bool),
        "ribosomal": np.array([g.startswith("RPL") or g.startswith("RPS") for g in gu], bool),
        "mitochondrial": np.array([g.startswith("MT-") or g.startswith("MT.") for g in gu], bool),
    }


def _marker_z(y, mask, eps=1e-12):
    y = np.asarray(y, float); mask = np.asarray(mask, bool)
    ok = np.isfinite(y); y, mask = y[ok], mask[ok]
    if mask.sum() < 3:
        return np.nan
    sd = y.std()
    if sd <= eps:
        return np.nan
    z = (y - y.mean()) / sd
    return float(z[mask].mean())


# ------------------------------------------------------------------ IO
def _decode(arr):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(arr).ravel()], dtype=object)


def _load(dataset_dir, mode):
    d = Path(dataset_dir); md = d / "normalizations" / mode
    genes = _decode(np.load(d / "genes.npy", allow_pickle=True))
    perts = _decode(np.load(d / "perturbations.npy", allow_pickle=True))
    tgi = np.load(d / "target_gene_indices.npy", allow_pickle=True).astype(int)
    tgp = d / "target_genes.npy"
    tgenes = _decode(np.load(tgp, allow_pickle=True)) if tgp.exists() else genes[np.clip(tgi, 0, len(genes) - 1)]
    Sigma = np.load(md / "Sigma_full_ridge.npy", mmap_mode="r")
    with h5py.File(md / "perturbation_stats.h5", "r") as h5:
        dx = np.asarray(h5["dx"][:], float)
        control_mean = np.asarray(h5["control_mean"][:], float)
        mean_pert = np.asarray(h5["mean_pert"][:], float)
        n_cells = np.asarray(h5["n_cells_pert"][:]) if "n_cells_pert" in h5 else np.full(len(perts), np.nan)
    return dict(genes=genes, perts=perts, tgi=tgi, tgenes=tgenes, Sigma=Sigma,
                dx=dx, mu=control_mean, mean_pert=mean_pert, n_cells=n_cells)


def compute_dataset(dataset_dir, dataset_name, mode):
    """Per-perturbation mu-axis (control-mean) forward diagnostic for one dataset/mode."""
    D = _load(dataset_dir, mode)
    genes, perts, tgi, tgenes = D["genes"], D["perts"], D["tgi"], D["tgenes"]
    Sigma, dx_all, mu, mean_pert_all, n_cells = D["Sigma"], D["dx"], D["mu"], D["mean_pert"], D["n_cells"]
    p = Sigma.shape[0]
    masks = _marker_masks(genes)
    rng = np.random.default_rng(RNG_SEED)
    order = np.argsort(-np.asarray(n_cells, float))            # most-covered perturbations first
    order = order[:min(MAX_PERTS_PER_DATASET, len(order))]
    ctrl_total = float(np.nansum(mu))
    rows = []
    for pi in order:
        target_idx = int(tgi[pi])
        if target_idx < 0 or target_idx >= p:
            continue
        dx = np.asarray(dx_all[pi, :], float)
        mean_pert = np.asarray(mean_pert_all[pi, :], float)
        basis = np.asarray(Sigma[:, target_idx], float)
        valid = np.flatnonzero(np.isfinite(dx) & np.isfinite(basis) & np.isfinite(mu))
        if EXCLUDE_TARGET_GENE_FROM_EVAL:
            valid = valid[valid != target_idx]
        if len(valid) < 100:
            continue
        rng.shuffle(valid)
        n_tr = int(round(TRAIN_FRAC * len(valid)))
        train_idx, test_idx = valid[:n_tr], valid[n_tr:]
        if len(train_idx) < 50 or len(test_idx) < 50:
            continue
        denom = float(np.sum(basis[train_idx] ** 2))
        if denom <= 1e-20:
            continue
        alpha = float(np.sum(dx[train_idx] * basis[train_idx]) / denom)
        pred = alpha * basis
        fit = None if PROJECTION_FIT_MODE == "all" else train_idx
        dx_mu, dx_res, dx_coef = _project_onto_axis(dx, mu, fit_idx=fit)
        pred_mu, pred_res, pred_coef = _project_onto_axis(pred, mu, fit_idx=fit)
        pert_total = float(np.nansum(mean_pert))
        r2_full = _uncentered_r2(dx[test_idx], pred[test_idx])
        r2_res = _uncentered_r2(dx_res[test_idx], pred_res[test_idx])
        pear_full = _pearson_safe(dx[test_idx], pred[test_idx])
        pear_res = _pearson_safe(dx_res[test_idx], pred_res[test_idx])
        rows.append({
            "dataset": dataset_name, "mode": mode, "perturbation": str(perts[pi]),
            "target_gene": str(tgenes[pi]), "target_gene_index": target_idx,
            "n_cells_pert": int(n_cells[pi]) if np.isfinite(n_cells[pi]) else np.nan,
            "n_genes": p, "n_train_genes": len(train_idx), "n_test_genes": len(test_idx), "alpha": alpha,
            "ctrl_total": ctrl_total, "pert_total": pert_total, "delta_total": pert_total - ctrl_total,
            "log2_total_fc": float(np.log2((pert_total + 1e-9) / (ctrl_total + 1e-9))),
            "cos_dx_mu": _cosine_safe(dx, mu), "cos_pred_mu": _cosine_safe(pred, mu),
            "dx_mu_coef": dx_coef, "pred_mu_coef": pred_coef,
            "dx_mu_fraction_norm2": _fraction_norm2(dx_mu, dx),
            "pred_mu_fraction_norm2": _fraction_norm2(pred_mu, pred),
            "r2_full": r2_full, "r2_mu_parallel": _uncentered_r2(dx_mu[test_idx], pred_mu[test_idx]),
            "r2_mu_residual": r2_res, "r2_full_minus_residual": (r2_full - r2_res) if (np.isfinite(r2_full) and np.isfinite(r2_res)) else np.nan,
            "pearson_full": pear_full, "pearson_mu_parallel": _pearson_safe(dx_mu[test_idx], pred_mu[test_idx]),
            "pearson_mu_residual": pear_res, "pearson_full_minus_residual": (pear_full - pear_res) if (np.isfinite(pear_full) and np.isfinite(pear_res)) else np.nan,
            "dx_resid_cell_cycle_z": _marker_z(dx_res, masks["cell_cycle"]),
            "dx_resid_translation_z": _marker_z(dx_res, masks["translation"]),
            "dx_resid_ribosomal_z": _marker_z(dx_res, masks["ribosomal"]),
            "dx_resid_mitochondrial_z": _marker_z(dx_res, masks["mitochondrial"]),
        })
    return pd.DataFrame(rows)


def run_all(precompute_root, out_dir, modes=MODES):
    """Discover forward-precompute dataset dirs, write per-dataset diagnostics + an aggregate."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(os.path.dirname(p) for p in glob.glob(
        os.path.join(precompute_root, "*", "normalizations", modes[0], "perturbation_stats.h5")))
    dirs = [str(Path(d).parent.parent) for d in dirs]                 # -> dataset dir
    frames = []
    for dd in dirs:
        name = Path(dd).name.split("__mean_control_ge_")[0].split("__mean_ge_")[0]
        for mode in modes:
            if not (Path(dd) / "normalizations" / mode / "perturbation_stats.h5").exists():
                continue
            df = compute_dataset(dd, name, mode)
            if len(df):
                df.to_csv(out / f"{name}__{mode}__depth_totalRNA_diagnostic.tsv", sep="\t", index=False)
                frames.append(df)
                print(f"[ok] {name}/{mode}: {len(df)} perturbations")
    if not frames:
        raise RuntimeError(f"no usable precompute datasets under {precompute_root}")
    alld = pd.concat(frames, ignore_index=True)
    alld.to_csv(out / "all_depth_totalRNA_diagnostics.tsv", sep="\t", index=False)
    summ = alld.groupby("dataset").agg(
        n_perturbations=("perturbation", "count"),
        mean_dx_mu_fraction_norm2=("dx_mu_fraction_norm2", "mean"),
        median_dx_mu_fraction_norm2=("dx_mu_fraction_norm2", "median"),
        mean_pred_mu_fraction_norm2=("pred_mu_fraction_norm2", "mean"),
        median_pred_mu_fraction_norm2=("pred_mu_fraction_norm2", "median"),
        mean_r2_full=("r2_full", "mean"), mean_r2_mu_residual=("r2_mu_residual", "mean"),
    ).reset_index()
    summ.to_csv(out / "summary_depth_totalRNA_diagnostic.tsv", sep="\t", index=False)
    print(f"[done] {len(alld)} rows across {len(frames)} datasets -> {out}")
    return alld


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    default_root = os.path.join(os.environ.get("CIPHER_DATA_DIR", ""), "suppl",
                                "precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_SAFE_mean_control_ge_1p0")
    ap.add_argument("--precompute-root", default=default_root,
                    help="forward precompute root (<dataset>/normalizations/<mode>/perturbation_stats.h5)")
    ap.add_argument("--out", required=True, help="output directory for the diagnostic TSVs")
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    a = ap.parse_args(argv)
    run_all(a.precompute_root, a.out, modes=tuple(a.modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
