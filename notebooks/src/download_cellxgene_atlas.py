#!/usr/bin/env python
"""Stage the CellxGene *atlas* objects consumed by the Fig 3/4 covariance-transfer
supplement (``notebooks/suppl/fig3_atlas.ipynb`` -> ``src.run_fig3_atlas``).

Background
----------
``src/run_fig3_atlas.py`` drives two atlas-transfer pipelines that pull raw-count cells
from the CellxGene **Census** at run time via ``cellxgene_census``:

  * ``marson_cellxgene_robust_atlas_transfer`` -- T-cell / PBMC / tissue groups,
    incl. **CD4 T** and **CD8 T**;
  * ``rpe_first3_plus_far_from_rpe_atlas_streamed`` -- **RPE / retina** cells and
    a far-from-RPE negative-control group.

The notebook markdown flags this as the pipeline's only "Data gap": the Census objects are
not staged locally, so the cells are cleaned faithfully but not runnable offline. This
script closes that gap: it reproduces the **exact** Census fetch (same census version, same
obs ``value_filter``, same ``cell_type`` values, same ``X_name='raw'`` and obs/var column
projections) and writes one atlas ``.h5ad`` per logical group under
``$CIPHER_DATA_DIR/suppl/cellxgene_atlas/`` so the fetch is reproducible and cacheable.

The Census query issued per cell type is::

    cell_type == '<CT>' and is_primary_data == True
        and disease == 'normal' and suspension_type == 'cell'

Everything downstream of that raw fetch (PBMC-vs-tissue classification, RPE-keyword
filtering, per-dataset/stratum sub-sampling, covariance) stays in ``run_fig3_atlas`` and is
NOT duplicated here -- this script only materialises the raw-count atlas cells.

Paths are entirely env-driven (``CIPHER_DATA_DIR``); nothing is hard-coded.

Usage
-----
    export CIPHER_DATA_DIR=/path/to/cipher_data
    python notebooks/src/download_cellxgene_atlas.py --dry-run          # plan only, no network
    python notebooks/src/download_cellxgene_atlas.py                    # all groups
    python notebooks/src/download_cellxgene_atlas.py --groups cd4_t cd8_t rpe

Requires ``cellxgene-census`` (pulls ``tiledbsoma``); see module docstring / task notes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# Census query constants -- these mirror src/run_fig3_atlas.py. Do not "improve" these
# values: they define which cells the published panels were computed from.
# ----------------------------------------------------------------------------------------
CENSUS_VERSION = "2025-11-08"
ORGANISM = "Homo sapiens"

BASE_OBS_FILTER_SUFFIX = (
    "is_primary_data == True "
    "and disease == 'normal' "
    "and suspension_type == 'cell'"
)

# obs / var column projections -- identical to the get_anndata() calls in run_fig3_atlas.
OBS_COLUMN_NAMES = [
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
]
VAR_COLUMN_NAMES = [
    "soma_joinid",
    "feature_id",
    "feature_name",
]

# ---- Marson (T-cell / PBMC / tissue) cell-type vocabularies ----------------------------
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

# ---- RPE-atlas cell-type vocabularies --------------------------------------------------
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

# Logical atlas groups -> the union of cell_type strings queried from Census for that group.
# Each becomes one staged .h5ad. The task's headline targets are cd4_t / cd8_t / rpe; the
# remaining groups complete the faithful reproduction of both atlas-transfer pipelines.
ATLAS_GROUPS = {
    "cd4_t": CD4_TCELL_TYPES,
    "cd8_t": CD8_TCELL_TYPES,
    "treg": TREG_TYPES,
    "generic_t": GENERIC_TCELL_TYPES,
    "pbmc_non_t": PBMC_NON_T_TYPES,
    "tissue_non_t": TISSUE_NON_T_TYPES,
    "rpe": RPE_CELL_TYPE_NAMES,
    "far_from_rpe": FAR_FROM_RPE_CELL_TYPE_NAMES,
}

# Reproducibility / sampling defaults. run_fig3_atlas sub-samples downstream
# (N_CELLS_PER_CELLXGENE_SOURCE=10000 per Marson dataset; N_CELLS_PER_RPE_ATLAS_SOURCE=5000;
# MIN_STRATUM_CELLS=200 / MAX_CELLS_PER_STRATUM=1500; MIN_PERT_CELLS/MIN_CONTROL_CELLS=100).
# To keep the staged object bounded but large enough to satisfy those caps, we cap total
# cells per group here (override with --max-cells-per-group / env CXG_MAX_CELLS_PER_GROUP).
RANDOM_SEED = 0
DEFAULT_MAX_CELLS_PER_GROUP = 200_000
# Fetch raw counts in chunks to bound memory (matches CELLXGENE_FETCH_CHUNK_CELLS in run_fig3_atlas).
FETCH_CHUNK_CELLS = 1000


# ----------------------------------------------------------------------------------------
# Path resolution (env-driven, no hard-coding)
# ----------------------------------------------------------------------------------------
def resolve_paths(out_dir: str | None = None) -> tuple[Path, Path, Path]:
    data_dir = os.environ.get("CIPHER_DATA_DIR")
    if not data_dir:
        raise SystemExit(
            "CIPHER_DATA_DIR is not set. Export it first, e.g.\n"
            "  export CIPHER_DATA_DIR=/projects/.../cipher_data"
        )
    data_dir = Path(data_dir)
    suppl = data_dir / "suppl"
    out = Path(out_dir) if out_dir else suppl / "cellxgene_atlas"
    return data_dir, suppl, out


def group_h5ad_path(out: Path, group: str, census_version: str) -> Path:
    return out / f"atlas__{group}__census_{census_version}.h5ad"


def value_filter_for_cell_type(cell_type: str) -> str:
    """The exact obs value_filter issued per cell type in run_fig3_atlas."""
    return f"cell_type == '{cell_type}' and {BASE_OBS_FILTER_SUFFIX}"


# ----------------------------------------------------------------------------------------
# Census fetch
# ----------------------------------------------------------------------------------------
def query_group_obs(census, cxg, cell_types, census_version: str) -> pd.DataFrame:
    """get_obs per cell type with the exact value_filter; dedup by soma_joinid."""
    frames = []
    for ct in cell_types:
        vf = value_filter_for_cell_type(ct)
        print(f"  [obs] cell_type={ct!r}  filter={vf}")
        obs = cxg.get_obs(
            census,
            ORGANISM,
            value_filter=vf,
            column_names=OBS_COLUMN_NAMES,
        )
        if len(obs):
            obs = obs.copy()
            obs["query_cell_type"] = ct
            frames.append(obs)
            print(f"    -> {len(obs):,} cells")
        else:
            print("    -> 0 cells")
    if not frames:
        return pd.DataFrame(columns=OBS_COLUMN_NAMES + ["query_cell_type"])
    obs = pd.concat(frames, ignore_index=True)
    obs = obs.drop_duplicates("soma_joinid").reset_index(drop=True)
    return obs


def subsample_obs(obs: pd.DataFrame, max_cells: int | None, seed: int) -> pd.DataFrame:
    if max_cells is None or len(obs) <= max_cells:
        return obs
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(obs), size=max_cells, replace=False))
    return obs.iloc[keep].reset_index(drop=True)


def fetch_group_anndata(census, cxg, ad, soma_joinids: np.ndarray):
    """get_anndata(X_name='raw', full gene space) for the given cells, chunked & concatenated.

    Uses the exact obs/var column projections from run_fig3_atlas. var_coords is left as
    None so the staged object carries the full Census gene space; run_fig3_atlas subsets to
    its union-gene space in memory.
    """
    soma_joinids = np.asarray(soma_joinids, dtype=np.int64)
    chunks = []
    for start in range(0, len(soma_joinids), FETCH_CHUNK_CELLS):
        end = min(start + FETCH_CHUNK_CELLS, len(soma_joinids))
        obs_coords = soma_joinids[start:end]
        print(f"    [X] cells {start:,}:{end:,} / {len(soma_joinids):,}")
        a = cxg.get_anndata(
            census=census,
            organism=ORGANISM,
            obs_coords=obs_coords,
            X_name="raw",
            obs_column_names=OBS_COLUMN_NAMES,
            var_column_names=VAR_COLUMN_NAMES,
        )
        chunks.append(a)
    if len(chunks) == 1:
        return chunks[0]
    return ad.concat(chunks, join="outer", merge="first")


def download_group(census, cxg, ad, group: str, cell_types, out: Path,
                   max_cells: int | None, census_version: str, overwrite: bool) -> Path:
    dest = group_h5ad_path(out, group, census_version)
    if dest.exists() and not overwrite:
        print(f"[{group}] exists, skipping (use --overwrite): {dest}")
        return dest

    print(f"\n=== atlas group '{group}' ({len(cell_types)} cell type(s)) ===")
    obs = query_group_obs(census, cxg, cell_types, census_version)
    if len(obs) == 0:
        print(f"[{group}] WARNING: no cells returned; nothing written.")
        return dest
    print(f"[{group}] total unique cells: {len(obs):,}; datasets: {obs['dataset_id'].nunique():,}")

    obs = subsample_obs(obs, max_cells, seed=RANDOM_SEED)
    print(f"[{group}] fetching raw counts for {len(obs):,} cells...")

    soma_joinids = obs["soma_joinid"].astype(np.int64).values
    adata = fetch_group_anndata(census, cxg, ad, soma_joinids)

    # normalise var_names to gene symbols (as run_fig3_atlas does before subsetting)
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str).values
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()

    adata.uns["cipher_atlas_group"] = group
    adata.uns["cipher_atlas_cell_types"] = list(cell_types)
    adata.uns["cipher_census_version"] = census_version
    adata.uns["cipher_base_obs_filter_suffix"] = BASE_OBS_FILTER_SUFFIX

    out.mkdir(parents=True, exist_ok=True)
    print(f"[{group}] writing {adata.shape[0]:,} x {adata.shape[1]:,} -> {dest}")
    adata.write_h5ad(dest)
    return dest


# ----------------------------------------------------------------------------------------
# Dry run (no network / no cellxgene_census import)
# ----------------------------------------------------------------------------------------
def dry_run(groups, out: Path, max_cells: int | None, census_version: str) -> None:
    print("DRY RUN -- no Census access, no files written\n")
    print(f"census_version : {census_version}")
    print(f"organism       : {ORGANISM}")
    print(f"base filter    : {BASE_OBS_FILTER_SUFFIX}")
    print(f"X_name         : raw")
    print(f"max cells/group: {max_cells}")
    print(f"output dir     : {out}\n")
    for g in groups:
        cts = ATLAS_GROUPS[g]
        dest = group_h5ad_path(out, g, census_version)
        print(f"[{g}] -> {dest}")
        for ct in cts:
            print(f"    value_filter: {value_filter_for_cell_type(ct)}")


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--groups", nargs="+", default=list(ATLAS_GROUPS),
        choices=list(ATLAS_GROUPS),
        help="Atlas groups to stage (default: all).",
    )
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: $CIPHER_DATA_DIR/suppl/cellxgene_atlas).")
    p.add_argument("--census-version", default=CENSUS_VERSION,
                   help=f"Census LTS version (default: {CENSUS_VERSION}).")
    p.add_argument(
        "--max-cells-per-group", type=int,
        default=int(os.environ.get("CXG_MAX_CELLS_PER_GROUP", DEFAULT_MAX_CELLS_PER_GROUP)),
        help="Cap total cells per group (<=0 means no cap).",
    )
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download groups whose .h5ad already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned filters/paths and exit (no network).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    max_cells = None if args.max_cells_per_group is not None and args.max_cells_per_group <= 0 \
        else args.max_cells_per_group

    _data_dir, _suppl, out = resolve_paths(args.out_dir)

    if args.dry_run:
        dry_run(args.groups, out, max_cells, args.census_version)
        return 0

    try:
        import cellxgene_census as cxg
        import anndata as ad
    except Exception as e:  # pragma: no cover - env-dependent
        raise SystemExit(
            f"cellxgene_census / anndata unavailable ({e!r}).\n"
            "Install with:  pip install cellxgene-census"
        )

    print(f"Opening Census {args.census_version} ...")
    written = []
    with cxg.open_soma(census_version=args.census_version) as census:
        for g in args.groups:
            dest = download_group(
                census, cxg, ad, g, ATLAS_GROUPS[g], out,
                max_cells=max_cells,
                census_version=args.census_version,
                overwrite=args.overwrite,
            )
            written.append(dest)

    print("\nDone. Staged atlas objects:")
    for d in written:
        print(f"  {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
