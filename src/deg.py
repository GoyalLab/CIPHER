# run a simple differential expression over every pertiurbation and dataset for comparing to cipher rocaucs in fig 4
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.sparse import issparse


def differential_expression_per_perturbation(adata, min_cells=5):
    """
    For each perturbation, compare it to control samples and compute:
    - log2 fold change
    - t-test p-values
    Returns a long-format DataFrame: (Perturbation, Gene, log2FC, pval)
    """
    gene_names = np.array(adata.var_names.tolist())
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    perturbations = adata.obs['perturbation'].unique()
    perturbations = [p for p in perturbations if p != 'control']

    control_mask = adata.obs['perturbation'] == 'control'
    X_control = X[control_mask]

    results = []

    for pert in perturbations:
        pert_mask = adata.obs['perturbation'] == pert
        if pert_mask.sum() < min_cells:
            print(f"⚠️ Skipping {pert} (only {pert_mask.sum()} cells)")
            continue

        X_pert = X[pert_mask]

        # Compute log2FC
        mean_pert = X_pert.mean(axis=0) + 1e-6
        mean_ctrl = X_control.mean(axis=0) + 1e-6
        log2fc = np.log2(mean_pert / mean_ctrl)

        # Compute p-values (Welch's t-test)
        pvals = ttest_ind(X_pert, X_control, axis=0, equal_var=False).pvalue

        for i, gene in enumerate(gene_names):
            results.append({
                "Perturbation": pert,
                "Gene": gene,
                "log2FC": log2fc[i],
                "pval": pvals[i]
            })

    return pd.DataFrame(results)
