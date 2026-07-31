#!/usr/bin/env python
"""Regenerate the per-dataset control covariance files that `figS14_pcs_dx_sigma` loads.

`pcs123` expects a precomputed layout it only *reads* (no compute fallback):

    <PRECOMPUTE_ROOT>/<dataset>__mean_ge_<tag>/
        sigmas/Sigma_full_ridge.npy      # gene x gene control covariance (ridged)
        genes.npy                        # gene names (the covariance axis)

This script rebuilds that layout from the base Perturb-seq h5ads using the installed
`cipher` package, so the supplement is self-contained (no opaque precomputed blobs).

    export CIPHER_DATA_DIR=/path/to/cipher_data
    python notebooks/src/regenerate_sigma.py --out resources/repro/sigma_precompute
    # then point figS14_pcs_dx_sigma's PRECOMPUTE_ROOT at that dir.

Config via env: SIGMA_NORM (default "raw"), SIGMA_THRESHOLD (default 1.0),
SIGMA_COV_MAX_CELLS (default 10000), SIGMA_RIDGE (default 1e-8).
"""
from __future__ import annotations

import argparse
import glob
import os
import numpy as np

import cipher
from cipher import load_dataset
from cipher.covariance import compute_covariance
from cipher.normalize import normalize_matrix, library_size

NORM = os.environ.get("SIGMA_NORM", "raw")
THRESHOLD = float(os.environ.get("SIGMA_THRESHOLD", "1.0"))
COV_MAX_CELLS = int(os.environ.get("SIGMA_COV_MAX_CELLS", "10000"))
RIDGE = float(os.environ.get("SIGMA_RIDGE", "1e-8"))
SEED = 0


def threshold_to_tag(t: float) -> str:
    # pcs123 convention: 1.0 -> "1p0", 0.1 -> "0p1"
    return str(t).replace(".", "p")


def regenerate(data_dir: str, out_root: str, datasets=None) -> None:
    tag = threshold_to_tag(THRESHOLD)
    files = ([os.path.join(data_dir, d if d.endswith(".h5ad") else d + ".h5ad") for d in datasets]
             if datasets else sorted(glob.glob(os.path.join(data_dir, "*.h5ad"))))
    os.makedirs(out_root, exist_ok=True)
    for path in files:
        name = os.path.basename(path)[:-5]
        try:
            ds = load_dataset(path, expression_threshold=THRESHOLD, min_samples=100)
        except Exception as e:
            print(f"[skip] {name}: {e!r}"); continue
        Xc = ds.control_matrix(dense=True)
        pc = ds.pflog_pseudocount if NORM == "pflog" else None
        Xn = normalize_matrix(Xc, NORM, libsize=library_size(Xc), pseudocount=pc)
        rng = np.random.default_rng(cipher.utils.stable_seed(SEED, ds.name))
        if COV_MAX_CELLS and Xn.shape[0] > COV_MAX_CELLS:
            Xn = Xn[np.sort(rng.choice(Xn.shape[0], COV_MAX_CELLS, replace=False))]
        Sigma = compute_covariance(Xn, ridge_abs=RIDGE)     # Sigma_full_ridge
        folder = os.path.join(out_root, f"{name}__mean_ge_{tag}")
        os.makedirs(os.path.join(folder, "sigmas"), exist_ok=True)
        np.save(os.path.join(folder, "sigmas", "Sigma_full_ridge.npy"), Sigma)
        np.save(os.path.join(folder, "genes.npy"), np.asarray(ds.gene_names, dtype=object))
        print(f"[ok]   {name}: Sigma {Sigma.shape} -> {folder}/sigmas/Sigma_full_ridge.npy")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("CIPHER_DATA_DIR", ""))
    ap.add_argument("--out", default="resources/repro/sigma_precompute",
                    help="PRECOMPUTE_ROOT to write <dataset>__mean_ge_<tag>/sigmas/ into")
    ap.add_argument("--datasets", nargs="*", help="dataset basenames (default: all *.h5ad in data-dir)")
    args = ap.parse_args(argv)
    if not args.data_dir:
        raise SystemExit("set CIPHER_DATA_DIR or pass --data-dir")
    regenerate(args.data_dir, args.out, args.datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
