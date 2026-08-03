# Supplementary-figure notebooks

Minimal reproductions of the paper's supplementary figures.

## Layout

- **`notebooks/src/`** — shared helper modules used by these notebooks. **Not part of the
  installable `cipher` package** (the package only ships `cipher/`), so this code stays in
  git for reproducibility without bloating the wheel. Notebooks add it to `sys.path` and
  `from src.<module> import *`.
- **`notebooks/suppl/`** — the cleaned notebooks (one per supplementary figure).

## Notebooks, figures, and data

All notebooks read `$CIPHER_DATA_DIR` (base Perturb-seq h5ads) and, where noted,
`$CIPHER_DATA_DIR/suppl` (supplemental inputs); outputs go to `$SUPPL_OUT`. Populate both with
`python resources/download_data.py` (see the repo README). Precomputed inputs marked
*(regenerable)* are **not** in either Zenodo record and must be rebuilt first — see the
`gen_*.py` and `regenerate_sigma.py` generators in `notebooks/src`, whose exact command lines
`download_data.py` prints when it finishes.

The "data needed" column below was checked against what each `notebooks/src/run_*.py` actually
reads. Several entries used to say "base h5ads" for notebooks that read **only** a precompute
and never open an `.h5ad` at all.

| notebook | paper figure | data needed |
|---|---|---|
| `figS5_frequency_normalization.ipynb` | S5 | base h5ads (full-gene covariance across normalizations) |
| `figS7_covariance_correlation.ipynb` | S7 | **full-covariance precompute only** (*regenerable*); reads no h5ad. Needs both the `mean_ge_1p0` and `mean_ge_0p1` thresholds |
| `figS9_generanking.ipynb` | S9 | **`suppl/posterior_inverse_fast_from_prerun_fullH_diag/` only** (*regenerable*); reads no h5ad |
| `figS9A_analytical_H.ipynb` | S9 A | base h5ads (+ `Sigma_full_ridge` precompute, *regenerable*) |
| `figS9B_inverse_noise_model.ipynb` | S9 B | base h5ads + full-covariance precompute (*regenerable*) |
| `figS9cd_double_pert_inverse.ipynb` | S9 C/D | `proper_filtered.h5ad` (combinatorial screen) |
| `figS11_drug_inverse.ipynb` | Fig 5 / S11 | sci-Plex3 full-gene h5ad |
| `figS13D_cov_eigenspectrum.ipynb` | S13 D | base h5ads + forward precompute (*regenerable*) |
| `figS14_pcs_dx_sigma.ipynb` | S14 | base h5ads (+ per-dataset `Sigma*.npy`, *regenerable*) |
| `figS15_effective_N.ipynb` | S15 | response-breadth TSV (*regenerable*) |
| `figS16_forward.ipynb` | S16 | base h5ads (+ forward precompute, *regenerable*) |
| `figS18_coexpr.ipynb` | S18 | base h5ads + interaction-pairs CSVs (GGI + PPI) |
| `figS19_umap_cov_visualization.ipynb` | S19 | base h5ads + full-covariance precompute (*regenerable*) |
| `umapvis_figS.ipynb` | Supplement (UMAP covariance visualization) | base h5ads |
| `tau2_selection.ipynb` | Supplement (τ² choice) | resistance h5ads (+ `Sigma`/`H`, *regenerable*) |
| `kras_resistance_figM7_S17.ipynb` | M7 / S17 | KRAS naive-vs-resistant h5ad |
| `melanoma_resistance_figM7_S17.ipynb` | M7 / S17 | GSE233766-derived `Xtot_*` h5ads |
| `fate_commitment_figM7.ipynb` | M7 | LARRY `stateFate_inVitro_*` |
| `fate_commitment_bayesian_figM7_S17.ipynb` | M7 / S17 | LARRY `stateFate_inVitro_*` |
| `fig3_atlas.ipynb` | Fig 3/4 | CellxGene Census (via `cellxgene_census`) |

## Data requirements

`python resources/download_data.py` populates everything below except the precomputes, into
both `$CIPHER_DATA_DIR` and `$CIPHER_DATA_DIR/suppl`. Verify with `--check`. The supplemental
inputs it places under `suppl/`:

- **Interaction-pairs CSVs** (GGI + PPI) for `figS18_coexpr`.
- **Resistance objects** — KRAS naive-vs-resistant h5ad, and the GSE233766-derived melanoma
  `Xtot_*` h5ads. These are now published in the supplement record, so the raw GEO tar is no
  longer needed to reconstruct them.
- **LARRY `stateFate_inVitro_*`** lineage-tracing objects for the fate-commitment notebooks.
- **CellxGene atlas** for `fig3_atlas` — *not* downloaded by that script; it is pulled at run
  time from the Census, and the query is reproduced standalone by
  `notebooks/src/download_cellxgene_atlas.py`.

### Precomputes must be built before the notebooks that read them

`Sigma`/`H`/`.npz`/breadth-TSV inputs are **not** in either Zenodo record. Rebuild them from the
base h5ads with `notebooks/src/gen_*.py` and `regenerate_sigma.py`; `download_data.py` prints
the exact commands on success. Two things the flags make easy to get wrong:

- `--out-root` differs per generator. `gen_fullcov_scores.py` and `gen_forward_precompute.py`
  take the **full precompute directory path**; `gen_breadth_table.py` and
  `gen_inverse_summary.py` take `$CIPHER_DATA_DIR/suppl` and append their own subdirectory.
  Pointing the first two at bare `suppl/` scatters dozens of `<dataset>__mean_*` directories
  where no consumer looks for them.
- `gen_fullcov_scores.py` needs `--thresholds 1.0 0.1`. `figS7` reads the `mean_ge_0p1`
  variant, so building only the default `1.0` silently yields an empty panel.
