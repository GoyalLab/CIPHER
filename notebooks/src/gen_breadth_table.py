"""Precompute GENERATOR for Fig S15 (response breadth / effective number of
responding genes: observed vs CIPHER-predicted).

Consumer
--------
``notebooks/src/run_figS15.py`` (blocks 1-3, driven by
``notebooks/suppl/figS15_effective_N.ipynb``) reads a single tab-separated
table::

    BREADTH_PATH = <out_root>/figS15/response_breadth_per_perturbation.tsv

(the thin notebook sets ``out_root = $CIPHER_DATA_DIR/suppl``).  Every block
requires exactly these columns (verified against run_figS15 blocks 1/2/3):

    dataset_base
    perturbation
    target_gene
    status                                      # only rows == "ok" are used
    observed_excluding_target_neff_shannon
    predicted_excluding_target_neff_shannon
    r2_uncentered_excluding_target
    pearson_excluding_target                    # required by block1

The consumer keeps rows with ``status == "ok"`` and finite, strictly-positive
observed/predicted N_eff, then filters on the forward R2 / Pearson columns.

How the columns are produced (installable ``cipher`` package only)
------------------------------------------------------------------
For each CRISPRi/CRISPRa Perturb-seq h5ad:

1. ``cipher.preprocess_dataset`` (default :class:`PreprocessConfig`,
   ``mean_control_ge_1p0`` -- the same store the forward figures use) writes the
   per-dataset covariance + per-perturbation ``dx`` under ``precompute_root``.
2. ``cipher.io.load_precomputed`` gives, per perturbation, the observed mean
   shift ``dx`` and the control covariance ``Sigma``.
3. CIPHER forward prediction (``cipher.core.forward_predict``) fits the rank-1
   scalar ``a_hat`` on all genes and returns the predicted shift
   ``a_hat * Sigma[:, target]``.
4. The directly-perturbed target gene is dropped ("excluding target"), and on
   the remaining genes we compute:
     * ``r2_uncentered_excluding_target`` / ``pearson_excluding_target``
       via :func:`cipher.core.forward_metrics` (identical definitions to the
       package's forward-prediction fit metrics), and
     * the Shannon effective number ``N_eff = exp(H)`` of the response power
       distribution ``p_i = |r_i|^2 / sum_j |r_j|^2`` for the observed shift
       (``observed_excluding_target_neff_shannon``) and the predicted shift
       (``predicted_excluding_target_neff_shannon``).

Run the full generation with::

    PYTHONPATH=/gpfs/projects/p32655/lschwartz/projects/CIPHER \
    /home/xlv0877/.conda/envs/CTMM/bin/python -c \
    "from notebooks.src.gen_breadth_table import generate; \
     generate('/projects/b1042/GoyalLab/lschwartz/misc/cipher_data', \
              '/projects/b1042/GoyalLab/lschwartz/misc/cipher_data/suppl')"

(heavy: covariance is O(n_genes^2) per dataset -- run on SLURM, not the login
node).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

import cipher
from cipher import PreprocessConfig, preprocess_dataset
from cipher.core import forward_metrics, forward_predict
from cipher.io import load_precomputed

# Normalization used for the response-breadth panel.  ``log1p`` is the package's
# canonical default normalization; the preprocess config otherwise matches the
# forward figures' store (mean-over-control gene cutoff of 1.0).
MODE = "raw"

# Output table location relative to ``out_root`` -- MUST match the consumer's
# BREADTH_PATH = <out_root>/figS15/response_breadth_per_perturbation.tsv.
TSV_SUBDIR = "figS15"
TSV_NAME = "response_breadth_per_perturbation.tsv"

# Only single-gene perturbations that a fit could be scored on.
MIN_GENES_FOR_FIT = 3

# Columns the consumer (run_figS15) requires, in a stable order.
REQUIRED_COLUMNS = [
    "dataset_base",
    "perturbation",
    "target_gene",
    "status",
    "observed_excluding_target_neff_shannon",
    "predicted_excluding_target_neff_shannon",
    "r2_uncentered_excluding_target",
    "pearson_excluding_target",
]


def neff_shannon(response) -> float:
    """Shannon effective number of responding genes for a response vector.

    ``p_i = |r_i|^2 / sum_j |r_j|^2`` (fraction of response *power* per gene),
    ``H = -sum_i p_i ln p_i``, ``N_eff = exp(H)``.  N_eff ranges from 1 (all the
    response power in a single gene) up to the number of genes (a perfectly flat
    response).  Returns ``nan`` if the response carries no power.
    """
    r = np.abs(np.asarray(response, dtype=np.float64).ravel())
    r = r[np.isfinite(r)]
    power = r * r
    total = float(power.sum())
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    p = power / total
    p = p[p > 0.0]
    entropy = float(-np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def _resolve_datasets(data_dir, datasets):
    """Return ``[(dataset_base, h5ad_path), ...]`` for the CRISPRi/a panel."""
    data_dir = Path(data_dir)
    if datasets is not None:
        pairs = []
        for entry in datasets:
            p = Path(entry)
            if not p.is_absolute() and p.suffix != ".h5ad":
                p = data_dir / f"{entry}.h5ad"
            elif not p.is_absolute():
                p = data_dir / entry
            pairs.append((p.stem, p))
        return pairs

    pairs = []
    for path in sorted(glob.glob(str(data_dir / "*.h5ad"))):
        base = Path(path).stem
        if cipher.dataset_group(base) in ("CRISPRi", "CRISPRa"):
            pairs.append((base, Path(path)))
    return pairs


def _rows_for_dataset(dataset_base, h5ad_path, precompute_root, mode, progress):
    """Build per-perturbation breadth rows for one dataset."""
    dataset_dir = preprocess_dataset(
        str(h5ad_path),
        str(precompute_root),
        modes=[mode],
        config=PreprocessConfig(save_mean_var=True),
        progress=progress,
    )

    pc = load_precomputed(dataset_dir, mode)
    Sigma = pc.sigma(mmap=True)
    gene_names = np.asarray(pc.gene_names)
    n_perts = len(pc.perturbations)

    rows = []
    for i in range(n_perts):
        pert = str(pc.perturbations[i])
        gene_idx = int(pc.target_gene_indices[i])

        if gene_idx < 0 or gene_idx >= len(gene_names):
            rows.append({
                "dataset_base": dataset_base,
                "perturbation": pert,
                "target_gene": "",
                "status": "no_target",
                "observed_excluding_target_neff_shannon": np.nan,
                "predicted_excluding_target_neff_shannon": np.nan,
                "r2_uncentered_excluding_target": np.nan,
                "pearson_excluding_target": np.nan,
            })
            continue

        target_gene = str(gene_names[gene_idx])
        dx = np.asarray(pc.dx[i], dtype=np.float64)
        sigma_col = np.asarray(Sigma[:, gene_idx], dtype=np.float64)

        # CIPHER forward prediction: a_hat fit on all genes -> rank-1 shift.
        pred, _a_hat = forward_predict(Sigma, dx, gene_idx)

        # Exclude the directly-perturbed target gene from the response.
        finite = np.isfinite(dx) & np.isfinite(pred)
        mask = finite.copy()
        mask[gene_idx] = False

        if int(mask.sum()) < MIN_GENES_FOR_FIT:
            rows.append({
                "dataset_base": dataset_base,
                "perturbation": pert,
                "target_gene": target_gene,
                "status": "too_few_genes",
                "observed_excluding_target_neff_shannon": np.nan,
                "predicted_excluding_target_neff_shannon": np.nan,
                "r2_uncentered_excluding_target": np.nan,
                "pearson_excluding_target": np.nan,
            })
            continue

        obs_excl = dx[mask]
        pred_excl = pred[mask]

        fit = forward_metrics(obs_excl, pred_excl)

        rows.append({
            "dataset_base": dataset_base,
            "perturbation": pert,
            "target_gene": target_gene,
            "status": "ok",
            "observed_excluding_target_neff_shannon": neff_shannon(obs_excl),
            "predicted_excluding_target_neff_shannon": neff_shannon(pred_excl),
            "r2_uncentered_excluding_target": float(fit["r2_uncentered"]),
            "pearson_excluding_target": float(fit["pearson"]),
        })

    return rows


def generate(data_dir, out_root, datasets=None, mode=MODE,
             precompute_root=None, progress=True):
    """Produce ``<out_root>/figS15/response_breadth_per_perturbation.tsv``.

    Parameters
    ----------
    data_dir : str | Path
        Directory holding the base Perturb-seq ``.h5ad`` files
        (``$CIPHER_DATA_DIR``).
    out_root : str | Path
        The supplement root the consumer points ``SUPPL`` at
        (``$CIPHER_DATA_DIR/suppl``); the table is written to
        ``<out_root>/figS15/response_breadth_per_perturbation.tsv``.
    datasets : sequence of str | Path, optional
        Explicit dataset names (or paths).  Default: every ``*.h5ad`` in
        ``data_dir`` classified as CRISPRi/CRISPRa by ``cipher.dataset_group``.
    mode : str
        Normalization mode for the covariance / dx (default ``log1p``).
    precompute_root : str | Path, optional
        Where ``preprocess_dataset`` stores Sigma/dx (default
        ``<out_root>/figS15/_precompute``).  Reused across runs.

    Returns
    -------
    pathlib.Path
        The written TSV path.
    """
    out_root = Path(out_root)
    if precompute_root is None:
        precompute_root = out_root / TSV_SUBDIR / "_precompute"
    precompute_root = Path(precompute_root)
    precompute_root.mkdir(parents=True, exist_ok=True)

    pairs = _resolve_datasets(data_dir, datasets)
    if not pairs:
        raise FileNotFoundError(
            f"No CRISPRi/CRISPRa .h5ad datasets found under {data_dir}."
        )

    all_rows = []
    for dataset_base, h5ad_path in pairs:
        if not Path(h5ad_path).exists():
            print(f"[skip] missing h5ad: {h5ad_path}")
            continue
        print(f"[breadth] {dataset_base}  ({h5ad_path})")
        all_rows.extend(
            _rows_for_dataset(dataset_base, h5ad_path, precompute_root, mode, progress)
        )

    breadth_df = pd.DataFrame(all_rows, columns=REQUIRED_COLUMNS)

    out_dir = out_root / TSV_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / TSV_NAME
    breadth_df.to_csv(out_path, sep="\t", index=False)

    n_ok = int((breadth_df["status"] == "ok").sum())
    print(f"[breadth] wrote {out_path}  ({len(breadth_df)} rows, {n_ok} ok, "
          f"{breadth_df['dataset_base'].nunique()} datasets)")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the Fig S15 breadth table.")
    parser.add_argument("--data-dir", default=os.environ.get("CIPHER_DATA_DIR"))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--mode", default=MODE)
    parser.add_argument("--precompute-root", default=None)
    args = parser.parse_args()

    generate(
        args.data_dir,
        args.out_root,
        datasets=args.datasets,
        mode=args.mode,
        precompute_root=args.precompute_root,
    )
