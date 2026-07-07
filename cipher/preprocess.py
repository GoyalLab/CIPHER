"""Feature 0 — precompute Sigma + per-perturbation statistics to disk.

This is the heavy, memory-aware engine that turns a single Perturb-seq
``.h5ad`` file into the on-disk artifacts consumed by :mod:`cipher.io`
(:func:`cipher.io.load_precomputed`).  For every requested normalization it
writes, under ``out_root/<stem>__mean_<cutoff_source>_ge_<threshold>/``::

    genes.npy, perturbations.npy, target_genes.npy, target_gene_indices.npy
    metadata.json
    normalizations/<mode>/
        Sigma_full_ridge.npy      # (p, p) float32 gene-gene covariance (ridged)
        Sigma_full_mean.npy       # (p,)   per-gene mean of the normalized cov matrix
        perturbation_stats.h5     # dx, control_mean, mean_pert, (+var), n_cells_pert

Dataset loading / control-perturbation detection / gene filtering are delegated
entirely to :func:`cipher.data.load_dataset`; normalization is delegated to
:mod:`cipher.normalize`.  The covariance is streamed to a ``np.memmap`` in
column blocks so the full ``p x p`` matrix never has to be built from a dense
``cells x genes`` design matrix, and per-perturbation statistics are accumulated
one group at a time so only a single group's dense expression lives in memory.

The design is a faithful refactor of the canonical "all-dataset forward
precompute" notebook (``write_full_covariance_memmap_chunked`` and the
group-mean/variance logic), but re-expressed against the shared foundation
modules so nothing is re-implemented here.
"""
from __future__ import annotations

import gc
import os
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import h5py
from scipy.sparse import issparse
from tqdm.auto import tqdm

from .data import load_dataset
from .normalize import (
    NORMALIZATION_MODES,
    transform_zero_preserving,
    transform_pflog,
    pflog_row_centers,
    fit_pflog_alpha,
    library_size,
    mean_var,
    WORK_DTYPE,
)
from .utils import (
    ensure_dir,
    sanitize_filename,
    stable_seed,
    json_default,
    atomic_save_npy,
)

#: dtype used for every array written to disk (matches the io.py contract).
SAVE_DTYPE = np.float32

#: rows processed at a time when a group has to be densified (pflog).
_ROW_CHUNK = 4096


@dataclass
class PreprocessConfig:
    """Knobs controlling :func:`preprocess_dataset`.

    The defaults reproduce the canonical precompute (mean-over-control gene
    cutoff of 1.0, single-gene perturbations with >=100 cells, covariance from
    up to 10k control cells, tiny ridge for a stable inverse).
    """

    expression_threshold: float = 1.0
    cutoff_source: str = "control"          # "control" | "all"
    min_samples_per_pert: int = 100
    only_single_gene: bool = True
    require_target_in_var: bool = True
    force_include_targets: bool = True
    cov_max_cells: int | None = 10000
    ridge_abs: float = 1e-8
    ridge_rel: float = 0.0
    save_mean_var: bool = True
    max_sigma_disk_gb: float = 20.0
    cov_block: int = 512
    seed: int = 0


# --------------------------------------------------------------------------- #
# normalization helpers (memory-aware; keep sparse sparse, densify in chunks)
# --------------------------------------------------------------------------- #
def _normalized_mean_var(X_raw, mode, pseudocount, need_var, row_chunk=_ROW_CHUNK):
    """Column mean (and optional unbiased variance) of a normalized group.

    Non-``pflog`` modes stay sparse and use :func:`cipher.normalize.mean_var`.
    ``pflog`` is dense after centering, so it is streamed in row blocks to keep
    memory bounded.  For ``pflog`` the per-cell centering terms are computed on
    the group's own cells (row centers are per-cell and group-independent).
    """
    n = int(X_raw.shape[0])
    p = int(X_raw.shape[1])
    if mode != "pflog":
        Y = transform_zero_preserving(X_raw, mode, libsize_rows=library_size(X_raw))
        mean, var = mean_var(Y)
        return mean, (var if need_var else None)

    pc = float(pseudocount)
    centers = pflog_row_centers(X_raw, pc)
    s = np.zeros(p, dtype=np.float64)
    ss = np.zeros(p, dtype=np.float64) if need_var else None
    for r0 in range(0, n, row_chunk):
        r1 = min(r0 + row_chunk, n)
        Yc = np.asarray(transform_pflog(X_raw[r0:r1], centers[r0:r1], pc), dtype=np.float64)
        s += Yc.sum(axis=0)
        if need_var:
            ss += np.einsum("ij,ij->j", Yc, Yc, optimize=True)
    mean = s / float(n) if n else np.zeros(p, dtype=np.float64)
    if need_var:
        if n > 1:
            var = np.maximum((ss - n * mean * mean) / float(n - 1), 0.0)
        else:
            var = np.full(p, np.nan)
        var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        var = None
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    return mean, var


def _materialize_cov_matrix(X_cov_raw, mode, pseudocount, row_chunk=_ROW_CHUNK):
    """Normalized covariance design matrix for ``mode`` (sparse stays sparse)."""
    if mode != "pflog":
        return transform_zero_preserving(X_cov_raw, mode, libsize_rows=library_size(X_cov_raw))
    pc = float(pseudocount)
    n_cov = int(X_cov_raw.shape[0])
    p = int(X_cov_raw.shape[1])
    centers = pflog_row_centers(X_cov_raw, pc)
    X_cov = np.empty((n_cov, p), dtype=SAVE_DTYPE)
    for r0 in range(0, n_cov, row_chunk):
        r1 = min(r0 + row_chunk, n_cov)
        X_cov[r0:r1, :] = transform_pflog(X_cov_raw[r0:r1], centers[r0:r1], pc)
    return X_cov


# --------------------------------------------------------------------------- #
# chunked covariance -> memmap  (port of write_full_covariance_memmap_chunked)
# --------------------------------------------------------------------------- #
def _write_covariance_memmap(out_path, X_cov, mean_cov, ridge, block, max_disk_gb,
                             progress=True, name="Sigma"):
    """Stream ``cov(X_cov)`` to a ``.npy`` memmap in column blocks.

    Handles sparse and dense ``X_cov``.  For each block of columns ``[a, b)`` the
    (un-centered) Gram block ``X[:, a:b].T @ X`` is formed, mean-corrected
    (``- n * mean_a * mean_b``), divided by ``n - 1``, the diagonal is ridged and
    the result is ``nan_to_num``'d before being written.
    """
    n, p = int(X_cov.shape[0]), int(X_cov.shape[1])
    if n < 2:
        raise ValueError(f"{name}: need at least 2 cells to compute a covariance (got {n}).")
    disk_gb = (p * p * np.dtype(SAVE_DTYPE).itemsize) / 1e9
    if disk_gb > max_disk_gb:
        raise MemoryError(
            f"{name} would be {disk_gb:.2f} GB for p={p:,} genes, exceeding "
            f"max_sigma_disk_gb={max_disk_gb:.2f} GB. Raise the expression "
            f"threshold or the disk cap."
        )

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    tmp_path = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    sigma = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=SAVE_DTYPE, shape=(p, p))
    if issparse(X_cov):
        X_csr = X_cov.tocsr()
        X_csr.eliminate_zeros()
        X_csc = X_csr.tocsc()
    else:
        X_csr = np.asarray(X_cov, dtype=WORK_DTYPE)
        X_csc = None
    mean_cov = np.asarray(mean_cov, dtype=np.float64)

    try:
        block_iter = range(0, p, block)
        if progress:
            block_iter = tqdm(block_iter, desc=f"{name} blocks", leave=False)
        for bi, a in enumerate(block_iter):
            b = min(a + block, p)
            if X_csc is not None:
                A = X_csc[:, a:b]
                XtX = A.T @ X_csr
                block_cov = (XtX.toarray().astype(np.float64, copy=False)
                             if issparse(XtX) else np.asarray(XtX, dtype=np.float64))
            else:
                A = X_csr[:, a:b]
                block_cov = (A.T @ X_csr).astype(np.float64, copy=False)
            block_cov -= n * mean_cov[a:b, None] * mean_cov[None, :]
            block_cov /= float(n - 1)
            local_rows = np.arange(b - a)
            global_cols = np.arange(a, b)
            block_cov[local_rows, global_cols] += ridge
            sigma[a:b, :] = np.nan_to_num(block_cov, nan=0.0, posinf=0.0,
                                          neginf=0.0).astype(SAVE_DTYPE, copy=False)
            if bi % 4 == 0:
                sigma.flush()
            del block_cov
        sigma.flush()
        del sigma
        if out_path.exists():
            out_path.unlink()
        tmp_path.rename(out_path)
    except Exception:
        try:
            del sigma
        except Exception:
            pass
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# --------------------------------------------------------------------------- #
# per-perturbation statistics
# --------------------------------------------------------------------------- #
def _write_perturbation_stats(out_path, ds, control_raw, perts, gene_names, mode,
                              pseudocount, pflog_alpha, config, progress=True):
    """Write ``perturbation_stats.h5`` for one normalization mode.

    ``control_mean`` is recomputed over the *full* control set (not the covariance
    subsample) so it is on the same normalization/gene axis as ``Sigma``.  Each
    perturbation group is normalized the same way and stored in ``ds.perturbations``
    order.
    """
    out_path = Path(out_path)
    p = int(len(gene_names))
    n_perts = len(perts)
    need_var = bool(config.save_mean_var)

    control_mean, control_var = _normalized_mean_var(control_raw, mode, pseudocount, need_var)

    ensure_dir(out_path.parent)
    if out_path.exists():
        out_path.unlink()

    str_dt = h5py.string_dtype(encoding="utf-8")
    chunks = (1, min(p, 4096)) if (n_perts and p) else None

    with h5py.File(out_path, "w") as h5:
        h5.attrs["mode"] = str(mode)
        h5.attrs["normalization_formula"] = str(NORMALIZATION_MODES[mode])
        h5.attrs["control_label"] = str(ds.control_label)
        h5.attrs["n_genes"] = int(p)
        h5.attrs["n_control"] = int(control_raw.shape[0])
        h5.attrs["n_perturbations"] = int(n_perts)
        h5.attrs["save_mean_var"] = bool(need_var)
        if mode == "pflog":
            h5.attrs["pflog_alpha"] = float(pflog_alpha) if pflog_alpha is not None else float("nan")
            h5.attrs["pflog_pseudocount"] = float(pseudocount)

        h5.create_dataset("gene_names", data=np.asarray(gene_names, dtype=object), dtype=str_dt)
        h5.create_dataset("perturbations", data=np.asarray(perts, dtype=object), dtype=str_dt)
        # target gene index per perturbation (aligned to `perts`) so a mode dir is
        # self-describing even if the dataset-level .npy files are moved away
        h5.create_dataset("target_gene_indices",
                          data=np.asarray(ds.target_gene_indices, dtype=np.int64))
        h5.create_dataset("control_mean", data=control_mean.astype(SAVE_DTYPE))
        if need_var:
            h5.create_dataset("control_var", data=control_var.astype(SAVE_DTYPE))

        dx_ds = h5.create_dataset("dx", shape=(n_perts, p), dtype=SAVE_DTYPE, chunks=chunks)
        mean_ds = h5.create_dataset("mean_pert", shape=(n_perts, p), dtype=SAVE_DTYPE, chunks=chunks)
        var_ds = (h5.create_dataset("var_pert", shape=(n_perts, p), dtype=SAVE_DTYPE, chunks=chunks)
                  if need_var else None)

        n_cells_pert = np.zeros(n_perts, dtype=np.int64)
        pert_iter = tqdm(perts, desc=f"{mode} stats", leave=False) if progress else perts
        for i, pert in enumerate(pert_iter):
            Xp = ds.perturbation_matrix(pert, dense=False)
            n_cells_pert[i] = int(Xp.shape[0])
            mean_p, var_p = _normalized_mean_var(Xp, mode, pseudocount, need_var)
            mean_ds[i, :] = mean_p.astype(SAVE_DTYPE)
            dx_ds[i, :] = (mean_p - control_mean).astype(SAVE_DTYPE)
            if need_var:
                var_ds[i, :] = var_p.astype(SAVE_DTYPE)

        h5.create_dataset("n_cells_pert", data=n_cells_pert)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def preprocess_dataset(data_path, out_root, modes=None, config=None,
                       overwrite=False, progress=True) -> Path:
    """Precompute per-normalization Sigma + perturbation stats for one ``.h5ad``.

    Parameters
    ----------
    data_path : str | Path
        Path to a Perturb-seq ``.h5ad`` file.
    out_root : str | Path
        Root directory; a per-dataset subdirectory is created inside it.
    modes : sequence of str, optional
        Normalization modes to compute (default: every mode in
        :data:`cipher.normalize.NORMALIZATION_MODES`).
    config : PreprocessConfig, optional
        Filtering / covariance / output knobs (default: :class:`PreprocessConfig`).
    overwrite : bool
        Recompute modes even if their outputs already exist.
    progress : bool
        Show ``tqdm`` progress bars.

    Returns
    -------
    pathlib.Path
        ``out_root/<sanitized_stem>__mean_<cutoff_source>_ge_<threshold>/``.
        Modes whose ``Sigma_full_ridge.npy`` and ``perturbation_stats.h5`` both
        already exist are skipped unless ``overwrite=True``.
    """
    config = config or PreprocessConfig()
    modes = list(NORMALIZATION_MODES) if modes is None else list(modes)
    unknown = [m for m in modes if m not in NORMALIZATION_MODES]
    if unknown:
        raise ValueError(f"Unknown normalization mode(s) {unknown}; "
                         f"choose from {list(NORMALIZATION_MODES)}.")

    t0 = time.time()
    data_path = Path(data_path)
    dataset_name = sanitize_filename(data_path.stem)
    source_tag = "mean_control" if config.cutoff_source == "control" else "mean_all"
    cutoff_tag = str(config.expression_threshold).replace(".", "p")
    outdir = ensure_dir(Path(out_root) / f"{dataset_name}__{source_tag}_ge_{cutoff_tag}")
    norm_root = ensure_dir(outdir / "normalizations")

    # ---- load + filter once (all detection/triage handled by cipher.data) ----
    ds = load_dataset(
        str(data_path),
        expression_threshold=config.expression_threshold,
        min_samples=config.min_samples_per_pert,
        cutoff_source=config.cutoff_source,
        only_single_gene=config.only_single_gene,
        require_target_in_var=config.require_target_in_var,
        force_include_targets=config.force_include_targets,
    )
    gene_names = ds.gene_names
    perts = list(ds.perturbations)
    p = ds.n_genes
    n_perts = len(perts)

    # ---- disk guard (p is fixed across modes) ----
    disk_gb = (p * p * np.dtype(SAVE_DTYPE).itemsize) / 1e9
    if disk_gb > config.max_sigma_disk_gb:
        raise MemoryError(
            f"Each Sigma would be {disk_gb:.2f} GB for p={p:,} genes, exceeding "
            f"max_sigma_disk_gb={config.max_sigma_disk_gb:.2f} GB. Raise the "
            f"expression threshold or the disk cap."
        )

    # ---- dataset-level files (written once, order == ds.perturbations) ----
    atomic_save_npy(outdir / "genes.npy", np.asarray(gene_names, dtype=object), allow_pickle=True)
    atomic_save_npy(outdir / "perturbations.npy", np.asarray(perts, dtype=object), allow_pickle=True)
    atomic_save_npy(outdir / "target_genes.npy",
                    np.asarray(ds.target_genes, dtype=object), allow_pickle=True)
    atomic_save_npy(outdir / "target_gene_indices.npy",
                    np.asarray(ds.target_gene_indices, dtype=np.int64))

    control_raw = ds.control_matrix(dense=False)      # full control set (kept sparse)
    n_control = int(control_raw.shape[0])
    rng = np.random.default_rng(stable_seed(config.seed, dataset_name))

    pflog_alpha = None
    pseudocount = None
    computed_modes, skipped_modes = [], []

    mode_iter = tqdm(modes, desc=f"modes:{dataset_name}") if progress else modes
    for mode in mode_iter:
        mode_dir = ensure_dir(norm_root / mode)
        sigma_path = mode_dir / "Sigma_full_ridge.npy"
        stats_path = mode_dir / "perturbation_stats.h5"
        if sigma_path.exists() and stats_path.exists() and not overwrite:
            skipped_modes.append(mode)
            continue

        # pflog dispersion is fit once from the full raw control matrix
        pc = None
        if mode == "pflog":
            if pseudocount is None:
                pflog_alpha, pseudocount, _, _ = fit_pflog_alpha(control_raw, gene_names)
            pc = pseudocount

        # covariance design matrix from (optionally subsampled) control cells
        if config.cov_max_cells is not None and n_control > config.cov_max_cells:
            sel = np.sort(rng.choice(n_control, int(config.cov_max_cells), replace=False))
            cov_raw = control_raw[sel]
        else:
            cov_raw = control_raw

        X_cov = _materialize_cov_matrix(cov_raw, mode, pc)
        mean_cov, var_cov = mean_var(X_cov)
        good = np.isfinite(var_cov) & (var_cov > 0)
        scale = float(np.mean(var_cov[good])) if np.any(good) else 1.0
        ridge = float(config.ridge_abs + config.ridge_rel * scale)

        atomic_save_npy(mode_dir / "Sigma_full_mean.npy", mean_cov.astype(SAVE_DTYPE))
        _write_covariance_memmap(sigma_path, X_cov, mean_cov, ridge, config.cov_block,
                                 config.max_sigma_disk_gb, progress=progress,
                                 name=f"{mode}/Sigma")
        del X_cov, mean_cov, var_cov
        gc.collect()

        _write_perturbation_stats(stats_path, ds, control_raw, perts, gene_names, mode,
                                  pc, pflog_alpha, config, progress=progress)
        computed_modes.append(mode)
        gc.collect()

    # ---- dataset metadata ----
    metadata = {
        "dataset": dataset_name,
        "data_path": str(data_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pert_key": ds.pert_key,
        "control_label": ds.control_label,
        "n_cells": int(ds.adata.n_obs),
        "n_genes": int(p),
        "n_control": n_control,
        "n_perturbations": int(n_perts),
        "modes": list(modes),
        "computed_modes": computed_modes,
        "skipped_existing_modes": skipped_modes,
        "config": asdict(config),
        "dataset_stats": ds.stats,
        "save_dtype": str(np.dtype(SAVE_DTYPE)),
        "pflog_alpha": None if pflog_alpha is None else float(pflog_alpha),
        "pflog_pseudocount": None if pseudocount is None else float(pseudocount),
        "elapsed_seconds": float(time.time() - t0),
    }
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=json_default)

    return outdir
