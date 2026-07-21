"""Shared CLI contract and split loading for the CIPHER benchmark drivers.

Every ``run_*.py`` driver in this directory takes the same core flags, so no path
is ever hardcoded:

    --splits-dir PATH   root holding <dataset>/{filtered.h5ad,control_idx.npy,
                        train_idx.npy,test_idx.npy}
    --dataset NAME ...  one or more dataset names (default: every dataset found
                        under --splits-dir)
    --output-dir PATH   results root; each driver writes
                        <output-dir>/<MODEL>/<dataset>/results.pkl

Model-specific drivers additionally take ``--resources-dir`` (GenePT embeddings,
gene-alias map, essential-gene list), ``--models-dir`` (vendored third-party model
source added to ``sys.path``) and ``--pretrained-dir`` (checkpoints).

Only numpy + anndata are imported here, so this module is safe to import from
every one of the per-model conda environments in ``benchmarks/conda_envs/``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

#: files expected inside <splits-dir>/<dataset>/
INDEX_FILES = ("control_idx.npy", "train_idx.npy", "test_idx.npy")
ADATA_FILE = "filtered.h5ad"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser(description, *, resources=False, models=False, pretrained=False,
                 device=False, epochs=None, seed=None, batch_size=None,
                 skip_existing=False):
    """Build an argparse parser with the shared benchmark flags.

    Each driver enables only the extras it actually needs; flag names and meanings
    are identical across all drivers.
    """
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--splits-dir", required=True, type=Path,
                   help="root containing <dataset>/{filtered.h5ad,control_idx.npy,"
                        "train_idx.npy,test_idx.npy}")
    p.add_argument("--dataset", nargs="+", default=None, metavar="NAME",
                   help="dataset name(s); default: every dataset under --splits-dir")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="results root; writes <output-dir>/<MODEL>/<dataset>/results.pkl")
    if resources:
        p.add_argument("--resources-dir", required=True, type=Path,
                       help="directory with the benchmark resource files (see "
                            "benchmarks/setup_resources.py)")
    if models:
        p.add_argument("--models-dir", required=True, type=Path,
                       help="directory holding the vendored third-party model source "
                            "(subdirs GEARS/, GenePert/, scGPT/, scLAMBDA/, scouter/)")
    if pretrained:
        p.add_argument("--pretrained-dir", type=Path, default=None,
                       help="pretrained checkpoint directory (default: derived from --models-dir)")
    if device:
        p.add_argument("--device", default="cuda",
                       help="torch device, e.g. cuda / cuda:0 / cpu")
    if epochs is not None:
        p.add_argument("--epochs", type=int, default=epochs, help="training epochs")
    if seed is not None:
        p.add_argument("--seed", type=int, default=seed, help="random seed")
    if batch_size is not None:
        p.add_argument("--batch-size", type=int, default=batch_size, help="batch size")
    if skip_existing:
        p.add_argument("--skip-existing", action="store_true",
                       help="skip a dataset whose results.pkl already exists")
        p.add_argument("--overwrite", action="store_true",
                       help="recompute even if results.pkl exists")
    return p


# --------------------------------------------------------------------------- #
# dataset discovery / validation
# --------------------------------------------------------------------------- #
def dataset_dir(splits_dir, dataset) -> Path:
    return Path(splits_dir) / str(dataset)


def split_paths(splits_dir, dataset, require_adata=True) -> dict:
    """Resolve and validate the per-dataset input paths.

    Raises ``FileNotFoundError`` naming the missing file rather than letting a
    downstream ``np.load`` fail obscurely.
    """
    d = dataset_dir(splits_dir, dataset)
    if not d.is_dir():
        raise FileNotFoundError(f"No such dataset directory: {d}")
    paths = {"dir": d, "adata": d / ADATA_FILE}
    for f in INDEX_FILES:
        paths[f.replace("_idx.npy", "")] = d / f
    missing = [str(paths[k]) for k in ("control", "train", "test") if not paths[k].exists()]
    if require_adata and not paths["adata"].exists():
        missing.insert(0, str(paths["adata"]))
    if missing:
        raise FileNotFoundError(
            f"Dataset {dataset!r} is incomplete under {d}; missing:\n  "
            + "\n  ".join(missing)
        )
    return paths


def discover_datasets(splits_dir, requested=None, require_adata=True):
    """Return the dataset names to run.

    ``requested`` (from ``--dataset``) is validated; when omitted every
    subdirectory of ``--splits-dir`` holding the split indices is used.
    """
    splits_dir = Path(splits_dir)
    if not splits_dir.is_dir():
        raise FileNotFoundError(f"--splits-dir does not exist: {splits_dir}")
    if requested:
        for name in requested:
            split_paths(splits_dir, name, require_adata=require_adata)
        return list(requested)
    found = []
    for sub in sorted(p for p in splits_dir.iterdir() if p.is_dir()):
        try:
            split_paths(splits_dir, sub.name, require_adata=require_adata)
        except FileNotFoundError:
            continue
        found.append(sub.name)
    if not found:
        raise FileNotFoundError(
            f"No complete datasets under {splits_dir} (need {ADATA_FILE} and "
            f"{', '.join(INDEX_FILES)} per dataset directory)."
        )
    return found


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_split_adata(splits_dir, dataset, add_counts_layer=False):
    """Load ``filtered.h5ad`` and attach the benchmark's ``condition``/``split`` columns.

    Reproduces the loading every driver performed inline: gene-symbol var index,
    ``perturbation`` -> ``condition`` with ``control`` -> ``ctrl``, and a ``split``
    column where train = train_idx + control_idx and test = test_idx.
    """
    import anndata as ad

    paths = split_paths(splits_dir, dataset)
    adata = ad.read_h5ad(str(paths["adata"]))

    if "gene_name" in adata.var.columns:
        adata.var.index = adata.var["gene_name"]
    adata.var_names_make_unique()

    if add_counts_layer:
        adata.layers["counts"] = adata.X.copy()

    adata.obs.drop(columns=["condition"], errors="ignore", inplace=True)
    adata.obs.rename(columns={"perturbation": "condition"}, inplace=True)
    adata.obs["condition"] = adata.obs["condition"].apply(
        lambda x: "ctrl" if x == "control" else x)

    control_indices = np.load(paths["control"])
    training_indices = np.load(paths["train"])
    testing_indices = np.load(paths["test"])
    training_indices = np.concatenate([training_indices, control_indices])

    split = np.full(adata.n_obs, None, dtype=object)
    split[training_indices] = "train"
    split[testing_indices] = "test"
    adata.obs["split"] = split
    return adata


# --------------------------------------------------------------------------- #
# outputs / resources / vendored model source
# --------------------------------------------------------------------------- #
def results_dir(output_dir, model, dataset, create=True) -> Path:
    """``<output-dir>/<MODEL>/<dataset>`` (created by default)."""
    d = Path(output_dir) / str(model) / str(dataset)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def resource_path(resources_dir, filename) -> Path:
    """Resolve a file in ``--resources-dir``, with an actionable error if absent."""
    p = Path(resources_dir) / filename
    if not p.exists():
        raise FileNotFoundError(
            f"Missing benchmark resource: {p}\n"
            "Fetch it with:  python benchmarks/setup_resources.py --resources-dir "
            f"{resources_dir}"
        )
    return p


def add_model_to_syspath(models_dir, name, subpath=None) -> Path:
    """Put vendored third-party model source on ``sys.path`` (replaces hardcoded hacks)."""
    root = Path(models_dir) / name
    if subpath:
        root = root / subpath
    if not root.is_dir():
        raise FileNotFoundError(
            f"Vendored model source not found: {root}\n"
            f"Expected --models-dir to contain a {name}/ directory."
        )
    sys.path.insert(0, str(root))
    return root
