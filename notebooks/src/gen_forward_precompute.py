"""Precompute generator for Fig S16 (forward dx/Sigma across normalizations).

The cleaned supplement notebook ``notebooks/suppl/figS16_forward.ipynb`` (engine
``notebooks/src/suppl_forward6.py``, driver ``notebooks/src/run_figS16.py``) loads a
PRECOMPUTE store and *recomputes* the forward-prediction metrics from it. This module
PRODUCES that store from the base Perturb-seq ``.h5ad`` files using the installed
``cipher`` package, matching the exact layout/schema the consumer reads.

Consumer contract (see ``suppl_forward6.py``)
--------------------------------------------
``choose_precompute_root`` / ``discover_dataset_dirs`` accept any root that contains
one or more ``<dataset>/normalizations/`` subdirectories. The reproduction notebook
points the root candidates at::

    $CIPHER_DATA_DIR/suppl/precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_SAFE_mean_control_ge_1p0
    $CIPHER_DATA_DIR/suppl/precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_mean_control_ge_1p0

So ``out_root`` should be the SAFE path above (the first candidate).

Per dataset the consumer needs (all produced by ``cipher.preprocess_dataset``):
  <dataset>/genes.npy                      (str, len = n_genes)
  <dataset>/perturbations.npy              (str, len = n_perts)
  <dataset>/target_gene_indices.npy        (int64, len = n_perts)
  <dataset>/target_genes.npy               (str, len = n_perts; optional but written)
  <dataset>/metadata.json                  (carries pflog_alpha for PFlog pushback)
  <dataset>/normalizations/<mode>/Sigma_full_ridge.npy      (n_genes x n_genes, memmapped)
  <dataset>/normalizations/<mode>/perturbation_stats.h5 with datasets:
        dx           (n_perts x n_genes)  <- required by compute_forward_for_dataset_mode
        control_mean (n_genes)            <- required by load_raw_reference_information (raw mode)
        perturbations, target_gene_indices, control_var, mean_pert, var_pert, n_cells_pert
        + attrs pflog_alpha / pflog_pseudocount on the pflog mode

Modes required by the consumer (``suppl_forward6.MODES``):
    raw, log1p, log1CP10k, frequency, pflog
The ``raw`` mode is mandatory (raw-space evaluation reference); ``pflog`` carries the
alpha/pseudocount the loader needs for its first-order pushback.

Dataset scope: the CRISPRi/CRISPRa set -- every ``*.h5ad`` in ``data_dir`` whose
``cipher.dataset_group`` is in {CRISPRi, CRISPRa}. The consumer's ``select_dataset_dirs``
further keeps all non-Marson datasets plus a single Marson dataset at plot time, but the
store itself contains every produced dataset.
"""
from __future__ import annotations

from pathlib import Path

import cipher
from cipher import PreprocessConfig, preprocess_dataset


# Modes the Fig S16 consumer scores (suppl_forward6.MODES). "raw" first so the
# raw-space reference stats exist before any other mode is loaded downstream.
FORWARD_MODES = ["raw", "log1p", "log1CP10k", "frequency", "pflog"]

# dataset_group values that belong to the forward-prediction figure.
FORWARD_DATASET_GROUPS = {"CRISPRi", "CRISPRa"}


def select_dataset_paths(data_dir, datasets=None):
    """Resolve the list of base ``.h5ad`` paths to precompute.

    ``datasets=None`` -> glob ``*.h5ad`` and keep dataset_group in {CRISPRi, CRISPRa}.
    Otherwise ``datasets`` is an explicit list of names (with or without ``.h5ad``).
    """
    data_dir = Path(data_dir)

    if datasets is not None:
        paths = []
        for name in datasets:
            name = str(name)
            candidate = data_dir / (name if name.endswith(".h5ad") else f"{name}.h5ad")
            if not candidate.exists():
                raise FileNotFoundError(f"Requested dataset not found: {candidate}")
            paths.append(candidate)
        return paths

    selected = []
    for path in sorted(data_dir.glob("*.h5ad")):
        try:
            group = cipher.dataset_group(path.name)
        except Exception:
            group = "unknown"
        if group in FORWARD_DATASET_GROUPS:
            selected.append(path)
    return selected


def generate(data_dir, out_root, datasets=None, overwrite=False, progress=True):
    """Produce the Fig S16 forward dx/Sigma precompute under ``out_root``.

    Parameters
    ----------
    data_dir : str | Path
        Directory holding the base Perturb-seq ``.h5ad`` files (``$CIPHER_DATA_DIR``).
    out_root : str | Path
        Precompute root the consumer will read. Use the SAFE candidate path
        ``$CIPHER_DATA_DIR/suppl/precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_SAFE_mean_control_ge_1p0``.
    datasets : list[str] | None
        Explicit dataset names, or ``None`` to auto-select the CRISPRi/CRISPRa set.
    overwrite : bool
        Recompute modes whose outputs already exist.
    progress : bool
        Show cipher's tqdm progress bars.

    Returns
    -------
    list[pathlib.Path]
        The per-dataset output directories that were produced.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = select_dataset_paths(data_dir, datasets=datasets)
    if not paths:
        raise FileNotFoundError(
            f"No CRISPRi/CRISPRa .h5ad datasets found under {data_dir}"
        )

    config = PreprocessConfig(save_mean_var=True)

    produced = []
    for path in paths:
        print(f"[gen_forward_precompute] preprocessing {path.name} -> {out_root}")
        outdir = preprocess_dataset(
            path,
            out_root,
            modes=FORWARD_MODES,
            config=config,
            overwrite=overwrite,
            progress=progress,
        )
        produced.append(Path(outdir))
        print(f"[gen_forward_precompute] wrote {outdir}")

    return produced


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("CIPHER_DATA_DIR"),
        help="Directory with base .h5ad files (default: $CIPHER_DATA_DIR).",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Precompute root to write (default: "
        "$CIPHER_DATA_DIR/suppl/precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_SAFE_mean_control_ge_1p0).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Explicit dataset names; omit to auto-select the CRISPRi/CRISPRa set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.data_dir:
        raise SystemExit("Set --data-dir or $CIPHER_DATA_DIR.")

    out_root = args.out_root or os.path.join(
        args.data_dir,
        "suppl",
        "precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_SAFE_mean_control_ge_1p0",
    )

    generate(args.data_dir, out_root, datasets=args.datasets, overwrite=args.overwrite)
