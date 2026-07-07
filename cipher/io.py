"""Readers for the artifacts written by :mod:`cipher.preprocess`.

A preprocessed dataset directory looks like::

    <outdir>/
        genes.npy, perturbations.npy, target_genes.npy, target_gene_indices.npy
        metadata.json
        normalizations/<mode>/
            Sigma_full_ridge.npy         # (p, p) covariance memmap
            Sigma_full_mean.npy          # (p,) column means
            perturbation_stats.h5        # dx, control_mean, mean_pert, ...
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import h5py


def _read_h5_strings(h5, key):
    arr = h5[key][:]
    return np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr])


@dataclass
class PrecomputedMode:
    """One normalization's precomputed covariance + per-perturbation statistics."""
    mode: str
    mode_dir: Path
    gene_names: np.ndarray
    perturbations: np.ndarray
    dx: np.ndarray                 # (n_perts, p) mean shifts
    control_mean: np.ndarray       # (p,)
    mean_pert: np.ndarray          # (n_perts, p)
    target_gene_indices: np.ndarray  # (n_perts,) index into gene_names (-1 if absent)

    _sigma_path: Path = None

    def sigma(self, mmap: bool = True) -> np.ndarray:
        """Load the (p, p) covariance matrix (memory-mapped by default)."""
        return np.load(self._sigma_path, mmap_mode="r" if mmap else None)

    def perturbation_index(self, pert) -> int:
        idx = np.where(self.perturbations == str(pert))[0]
        if not idx.size:
            raise KeyError(f"{pert!r} not among precomputed perturbations.")
        return int(idx[0])


def list_modes(dataset_dir) -> list:
    """Available normalization modes in a preprocessed dataset directory."""
    norm_root = Path(dataset_dir) / "normalizations"
    if not norm_root.exists():
        return []
    return sorted(p.name for p in norm_root.iterdir()
                  if (p / "Sigma_full_ridge.npy").exists())


def load_precomputed(dataset_dir, mode: str) -> PrecomputedMode:
    """Load covariance + perturbation stats for one ``mode``."""
    dataset_dir = Path(dataset_dir)
    mode_dir = dataset_dir / "normalizations" / mode
    sigma_path = mode_dir / "Sigma_full_ridge.npy"
    stats_path = mode_dir / "perturbation_stats.h5"
    if not sigma_path.exists():
        raise FileNotFoundError(f"No covariance for mode {mode!r} at {sigma_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"No perturbation stats for mode {mode!r} at {stats_path}")

    with h5py.File(stats_path, "r") as h5:
        gene_names = _read_h5_strings(h5, "gene_names")
        perturbations = _read_h5_strings(h5, "perturbations")
        dx = np.asarray(h5["dx"][:], dtype=np.float64)
        control_mean = np.asarray(h5["control_mean"][:], dtype=np.float64)
        mean_pert = np.asarray(h5["mean_pert"][:], dtype=np.float64)
        # preferred source: indices embedded alongside the stats (already aligned)
        h5_target_indices = (np.asarray(h5["target_gene_indices"][:], dtype=np.int64)
                             if "target_gene_indices" in h5 else None)

    tgi_path = dataset_dir / "target_gene_indices.npy"
    perts_path = dataset_dir / "perturbations.npy"
    if h5_target_indices is not None and h5_target_indices.shape[0] == perturbations.shape[0]:
        target_indices = h5_target_indices
    elif tgi_path.exists() and perts_path.exists():
        # dataset-level files: align by perturbation label to the h5 order
        ds_perts = np.load(perts_path, allow_pickle=True).astype(str)
        ds_tgi = np.load(tgi_path).astype(np.int64)
        lookup = {p: i for p, i in zip(ds_perts, ds_tgi)}
        target_indices = np.array([lookup.get(p, -1) for p in perturbations], dtype=np.int64)
    else:
        # last-resort fallback assumes each perturbation label *is* its target gene
        import warnings
        warnings.warn(
            f"No target_gene_indices in {stats_path} and no dataset-level "
            f"target_gene_indices.npy/perturbations.npy under {dataset_dir}; "
            "falling back to perturbation-label==gene-name, which is wrong for "
            "guide-ID labels. Re-run preprocess_dataset to embed correct indices.",
            stacklevel=2,
        )
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        target_indices = np.array([gene_to_idx.get(p, -1) for p in perturbations], dtype=np.int64)

    return PrecomputedMode(
        mode=mode, mode_dir=mode_dir, gene_names=gene_names, perturbations=perturbations,
        dx=dx, control_mean=control_mean, mean_pert=mean_pert,
        target_gene_indices=target_indices, _sigma_path=sigma_path,
    )
