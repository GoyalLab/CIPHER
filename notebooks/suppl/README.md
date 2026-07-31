# Supplementary-figure notebooks

Cleaned, minimal reproductions of the paper's supplementary figures.

## Layout

- **`notebooks/src/`** — shared helper modules used by these notebooks. **Not part of the
  installable `cipher` package** (the package only ships `cipher/`), so this code stays in
  git for reproducibility without bloating the wheel. Notebooks add it to `sys.path` and
  `from src.<module> import *`.
- **`notebooks/suppl/`** — the cleaned notebooks (one per supplementary figure).

## Refactor rules (applied to every notebook)

1. **Dedup → `src`.** Helper functions shared across notebooks live in a single
   `notebooks/src/*.py` module (logic unchanged).
2. **Use the package where equivalent.** Local reimplementations that match a `cipher`
   function are replaced by it (see mapping below); where a local variant genuinely differs,
   it stays in `src` so the result is unchanged.
3. **No hardcoding.** Inline paths → `os.environ["CIPHER_DATA_DIR"]` (no fallback; set it in
   the sbatch script). Output dirs → `$SUPPL_OUT`.
4. **Minimal + faithful.** Main-flow logic is preserved so each cleaned notebook reproduces
   the same numbers and plots when run on the same data.

## Package-mapping

| local helper | cipher replacement |
|---|---|
| `compute_covariance` / `make_covariance` | `cipher.compute_covariance` (+ `shrink=`) |
| `build_H_from_sample_means` / `analytic_gaussian_posterior` | `cipher.build_model` + `cipher.recover_u` |
| `select_hvgs_sparse` | `cipher.select_hvg_dispersion` |
| `r2_from_pred` / `safe_pearson` / `safe_r2` | `cipher.forward_metrics` / `cipher.metrics` |
| `prepare_whitened_eigendecomposition` / `evaluate_tau2_grid` | `cipher.build_model` / `cipher.fit_tau2` |

## Notebooks, figures, and data

All notebooks read `$CIPHER_DATA_DIR` (base Perturb-seq h5ads) and, where noted,
`$CIPHER_DATA_DIR/suppl` (supplemental inputs); outputs go to `$SUPPL_OUT`. Precomputed
inputs marked *(regenerable)* can be rebuilt with the package — see the `gen_*.py` and
`regenerate_sigma.py` generators in `notebooks/src`.

| notebook | paper figure | data needed |
|---|---|---|
| `figS5_frequency_normalization.ipynb` | S5 | base h5ads (full-gene covariance across normalizations) |
| `figS7_covariance_correlation.ipynb` | S7 | base h5ads |
| `figS9_generanking.ipynb` | S9 | base h5ads |
| `figS9A_analytical_H.ipynb` | S9 A | base h5ads (+ `Sigma_full_ridge` precompute, *regenerable*) |
| `figS9B_inverse_noise_model.ipynb` | S9 B | base h5ads |
| `figS9cd_double_pert_inverse.ipynb` | S9 C/D | double-perturbation h5ad (`Xtot_*BC50*`) |
| `figS11_drug_inverse.ipynb` | Fig 5 / S11 | sci-Plex3 full-gene h5ad |
| `figS13D_cov_eigenspectrum.ipynb` | S13 D | base h5ads |
| `figS14_pcs_dx_sigma.ipynb` | S14 | base h5ads (+ per-dataset `Sigma*.npy`, *regenerable*) |
| `figS15_effective_N.ipynb` | S15 | response-breadth TSV (*regenerable*) |
| `figS16_forward.ipynb` | S16 | base h5ads (+ forward precompute, *regenerable*) |
| `figS18_coexpr.ipynb` | S18 | base h5ads + interaction-pairs CSVs (GGI + PPI) |
| `figS19_umap_cov_visualization.ipynb` | S19 | base h5ads |
| `umapvis_figS.ipynb` | Supplement (UMAP covariance visualization) | base h5ads |
| `tau2_selection.ipynb` | Supplement (τ² choice) | resistance h5ads (+ `Sigma`/`H`, *regenerable*) |
| `kras_resistance_figM7_S17.ipynb` | M7 / S17 | KRAS naive-vs-resistant h5ad |
| `melanoma_resistance_figM7_S17.ipynb` | M7 / S17 | GSE233766-derived `Xtot_*` h5ads |
| `fate_commitment_figM7.ipynb` | M7 | LARRY `stateFate_inVitro_*` |
| `fate_commitment_bayesian_figM7_S17.ipynb` | M7 / S17 | LARRY `stateFate_inVitro_*` |
| `fig3_atlas.ipynb` | Fig 3/4 | CellxGene Census (via `cellxgene_census`) |

## Data requirements

Most notebooks need only the base Perturb-seq h5ads under `$CIPHER_DATA_DIR`. Additional
inputs, placed under `$CIPHER_DATA_DIR/suppl`:

- **Interaction-pairs CSVs** (GGI + PPI) for `figS18_coexpr`.
- **Resistance objects** — KRAS naive-vs-resistant h5ad; GSE233766-derived melanoma `Xtot_*`
  h5ads (GEO publishes the raw tar only — see `resources/download_suppl.py`).
- **LARRY `stateFate_inVitro_*`** lineage-tracing objects for the fate-commitment notebooks.
- **CellxGene atlas** for `fig3_atlas` — pulled at run time from the Census; the query is
  also reproduced standalone by `notebooks/src/download_cellxgene_atlas.py`.

Precomputed `Sigma`/`H`/`.npz`/breadth-TSV inputs can be regenerated from the base h5ads with
the package (`notebooks/src/gen_*.py`, `regenerate_sigma.py`).
