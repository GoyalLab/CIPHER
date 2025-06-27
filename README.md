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

## Usage

```python
from src.r2 import full_analysis_with_nulls_soft_and_plots
# Example usage
adata_path = 'path/to/your/perturb-seq-adata.h5ad'
output_dir = 'path/to/output_dir'
full_analysis_with_nulls_soft_and_plots(adata_path, save_dir=output_dir)
```

## License

[MIT License](LICENSE)