# CIPHER

**Covariance Inference for Perturbation and High-dimensional Expression Response**

CIPHER is a tool designed for analyzing covariance structures in high-dimensional gene expression data, particularly in response to perturbations.

## Features

- Covariance inference for gene expression datasets
- Designed for high-dimensional data
- Suitable for perturbation experiments

## Installation

```bash
# Example installation command
bash .init_conda.sh
mamba activate .conda/cipher
```

## Usage of CIPHER module

```python
from src.r2 import full_analysis_with_nulls_soft_and_plots
# Example usage of CIPHER module
adata_path = 'path/to/your/perturb-seq-adata.h5ad'
output_dir = 'path/to/output_dir'
full_analysis_with_nulls_soft_and_plots(adata_path, save_dir=output_dir)
```

## Reproduce paper figures
Notebooks to reproduce each figure of the paper can be found in the notebooks directory.
All notebooks work with the supplied conda environment.
Notebook figures:
---
notebooks/LR_fig2.ipynb
- Fig. 2 (All)
notebooks/LR_fig3_R2_hist.ipynb
- Fig. 3 (A-M)
notebooks/LF_double_pert_R2_and_inference.ipynb
- Fig. 3 (N, O, P)
- Fig. 4 H
notebooks/LR_fig3_cross_dataset.ipynb
- Fig. 3 Q, R
notebooks/LR_fig4.ipynb
- Fig. 4 (A-G)
notebooks/LR_fig5_TRADE_and_EGENES.ipynb
- Fig. 5



## License

[MIT License](LICENSE)