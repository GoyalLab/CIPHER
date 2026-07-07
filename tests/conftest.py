"""Shared fixtures: a small synthetic Perturb-seq dataset with real CIPHER signal.

The synthetic control cells are drawn from a factor model with a handful of
correlated gene *blocks* (each block shares a latent factor), sampled as
non-negative Poisson counts so mean expression is clearly positive.  Each
perturbation shifts its block's latent factor (which co-moves the whole
correlated block along the covariance structure) plus an extra boost on the
specific targeted gene, so that:

* the **forward** rank-1 projection onto the real ``Sigma[:, g]`` explains the
  shift far better than a marginals-only null, and
* the **reverse** solve ranks the true target gene at (or near) the top.

Perturbation labels are the targeted gene symbols (``G0``, ``G10``, ...), so the
``cipher.data`` detection heuristics resolve them without extra annotation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import pytest

# --- generator configuration ------------------------------------------------ #
N_GENES = 60
N_BLOCKS = 6                    # -> 10 genes per correlated block
N_CONTROL = 300
CELLS_PER_PERT = 40            # >= 5 cells per perturbation
TARGET_GENE_IDX = [0, 10, 20, 30, 40, 50, 5, 15]   # 8 distinct targeted genes
BLOCK_SHIFT = 1.1              # latent-factor shift applied to a target's block
GENE_BOOST = 0.9              # extra log-mean boost on the specific target gene


def _build_adata(seed: int = 0):
    """Build the synthetic :class:`anndata.AnnData` and its target labels."""
    rng = np.random.default_rng(seed)
    block_size = N_GENES // N_BLOCKS
    block_of = np.arange(N_GENES) // block_size
    base = rng.uniform(1.2, 2.4, N_GENES)          # per-gene base log-mean
    loading = rng.uniform(0.6, 1.0, N_GENES)       # per-gene factor loading
    fac_sd = 0.8

    def gen(n, z_shift=None, boost_gene=None):
        z = rng.normal(0.0, fac_sd, (n, N_BLOCKS))
        if z_shift is not None:
            z = z + z_shift[None, :]
        logmean = base[None, :] + loading[None, :] * z[:, block_of]
        if boost_gene is not None:
            logmean[:, boost_gene] = logmean[:, boost_gene] + GENE_BOOST
        return rng.poisson(np.exp(logmean)).astype(np.float32)

    blocks = [gen(N_CONTROL)]
    labels = ["control"] * N_CONTROL
    target_labels = []
    for g in TARGET_GENE_IDX:
        z_shift = np.zeros(N_BLOCKS)
        z_shift[block_of[g]] = BLOCK_SHIFT
        blocks.append(gen(CELLS_PER_PERT, z_shift=z_shift, boost_gene=g))
        lab = f"G{g}"
        labels += [lab] * CELLS_PER_PERT
        target_labels.append(lab)

    X = np.vstack(blocks)
    var_names = [f"G{i}" for i in range(N_GENES)]
    obs = pd.DataFrame({"perturbation": labels},
                       index=[f"cell{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=pd.Index(var_names, name="gene"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata, target_labels


class SynthData:
    """Bundle of the synthetic dataset plus the ground-truth it was built with."""

    def __init__(self, adata, target_labels):
        self.adata = adata
        self.target_labels = target_labels          # e.g. ['G0', 'G10', ...]
        self.target_indices = list(TARGET_GENE_IDX)
        self.n_genes = N_GENES
        self.n_control = N_CONTROL
        self.cells_per_pert = CELLS_PER_PERT
        self.gene_names = [f"G{i}" for i in range(N_GENES)]


@pytest.fixture(scope="session")
def synth() -> SynthData:
    adata, target_labels = _build_adata(seed=0)
    return SynthData(adata, target_labels)


@pytest.fixture(scope="session")
def h5ad_path(tmp_path_factory, synth) -> str:
    """Write the synthetic dataset to a temporary ``.h5ad`` and return its path."""
    path = tmp_path_factory.mktemp("cipher_data") / "synthetic.h5ad"
    synth.adata.write_h5ad(str(path))
    return str(path)


@pytest.fixture(scope="session")
def control_counts(synth) -> np.ndarray:
    """Raw (cells x genes) count matrix of the control cells only."""
    labels = synth.adata.obs["perturbation"].to_numpy().astype(str)
    return np.asarray(synth.adata.X[labels == "control"], dtype=np.float64)
