# Main-figure notebooks

Cleaned, minimal reproductions of the paper's main figures (3–7), built on the same
pattern as the supplementary notebooks in [`../suppl/`](../suppl/README.md).

## Layout & conventions

- **`notebooks/src/`** — shared helper/engine modules (`run_fig*.py`). **Not part of the
  installable `cipher` package**; kept in git for reproducibility. Each notebook is thin: a
  config cell that adds `notebooks/` to `sys.path`, does `from src.run_<fig> import *`, injects
  config into the engine module, and then a few `R.*()` calls — all heavy logic lives in the
  engine.
- **No hardcoded paths.** Notebooks read `os.environ["CIPHER_DATA_DIR"]` (base Perturb-seq
  h5ads) and write to `SUPPL_OUT` (`resources/repro/<fig>/`); set both in your sbatch script.
- **Raw counts.** Expression is on raw counts (`NORM="raw"`, the package default).
- **Inline == saved.** Each panel function does `fig.savefig(...); plt.show()` on the same
  figure, so a notebook's inline figures are the same renders as the saved `.svg` outputs.

## Notebooks, figures, and data

| notebook | figure panels | data needed |
|---|---|---|
| `fig3_forward.ipynb` | Fig. 3 C, D, E (real-Σ vs mean-field accuracy) + G, H, I (held-out prediction / estimator convergence) | base h5ads |
| `fig3_benchmarks.ipynb` | Fig. 3 J–N (CIPHER vs GEARS / scGPT / scLAMBDA / scouter / GenePert / linear-mean) + Systema centroid accuracy | cached benchmark metrics (`resources/repro/fig3_benchmarks/*.csv`, scored once from `benchmarks/` result pkls) |
| `fig4_cross_dataset.ipynb` | Fig. 4 B, C (cross-dataset covariance transfer) + E, F (CellxGene-atlas transfer) | base h5ads (+ CellxGene Census for E/F) |
| `fig5_inverse.ipynb` | Fig. 5 — inverse ROC/PR, CRISPRi/a perturbation identification, sci-Plex drug identification | base h5ads + sci-Plex3 full-gene h5ad |
| `fig6_effective_dims.ipynb` | Fig. 6 A, B (participation ratio), C–F (μ-axis reconstruction), G (PR vs ΔR²), H, I (effective response dimension) | base h5ads (+ response-breadth TSV, *regenerable*) |
| `fig7_validation.ipynb` | Fig. 7 B, F (naive-vs-resistant UMAPs), C (melanoma), G (KRAS resistance), K, L (LARRY prospective fate) | resistance h5ads + LARRY `stateFate_inVitro_*` |

Schematic panels (Fig. 3 A/B/F, Fig. 4 A, Fig. 5 A/B, the Fig. 6 diagrams) and the wet-lab
panels of Fig. 7 (D, E, H–J) are not reproduced from the release data.

## Notes

- **Mean-field null.** The Fig. 3 C/D/E accuracy panels compare the real control covariance
  against the **analytic diagonal** mean-field (per-gene variance, no gene–gene covariance);
  the held-out panels (G/H/I) use the empirical shuffled-over-cells null.
- **`fig3_benchmarks` cache.** The per-perturbation scores are computed once (reading the
  `benchmarks/results/<model>/<dataset>/results.pkl` files) into
  `resources/repro/fig3_benchmarks/{benchmark_metrics,centroid_accuracy,perturbation_status}.csv`;
  the notebook loads those and draws the violins. Regenerate with
  `CIPHER_BENCH_ROOT` / `CIPHER_BENCH_SPLITS` pointing at the benchmark outputs and splits
  (see [`../../benchmarks/README.md`](../../benchmarks/README.md)).
