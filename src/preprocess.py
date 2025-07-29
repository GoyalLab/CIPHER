import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import logging
from scipy.sparse import issparse


def _gene_name_keys() -> set[str]:
    return set(['gene_name', 'gene_names', 'name'])


def is_outlier(adata, column, nmads):
    vals = adata.obs[column]
    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    threshold = nmads * mad
    return (vals > median + threshold) | (vals < median - threshold)


def quality_control_filter(adata, percent_threshold=20, nmads=5, mt_nmads=5, mt_per=20):
    adata.var_names_make_unique()
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]')

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=['mt', 'ribo', 'hb'],
        inplace=True, percent_top=[percent_threshold],
        log1p=True
    )

    adata.obs['outlier'] = (
        is_outlier(adata, 'log1p_total_counts', nmads)
        | is_outlier(adata, 'log1p_n_genes_by_counts', nmads)
        | is_outlier(adata, 'pct_counts_in_top_20_genes', nmads)
    )

    adata.obs['mt_outlier'] = is_outlier(adata, 'pct_counts_mt', mt_nmads) | (
        adata.obs['pct_counts_mt'] > mt_per
    )

    adata = adata[adata.obs['n_genes_by_counts'] > 200]

    gene_counts = np.sum(adata.X > 0, axis=0)
    genes_to_keep = np.array(gene_counts).flatten() >= 5
    adata = adata[:, genes_to_keep]

    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)]
    return adata


def compute_sparsity(X, label=''):
    total = np.prod(X.shape)
    zeros = np.sum(X == 0)
    sparsity = zeros / total
    logging.info(f"{label} sparsity: {sparsity:.2%} ({zeros} of {total} zero entries)")
    return sparsity

def _perturbation_columns() -> set[str]:
    return set(['gene', 'perturbation_1'])

def _control_keys() -> set[str]:
    return set(['control', 'NT', 'non-targeting', 'ctrl'])

def get_data(selected_pert_index, data_path, expression_threshold=1., min_samples=100, qc=False, save=True):
    logging.info(f'Reading dataset: {data_path}')
    adata = ad.read_h5ad(data_path)

    # --- Sparsity before filtering ---
    full_X_pre = adata.X.toarray() if issparse(adata.X) else adata.X
    compute_sparsity(full_X_pre, label='Full matrix (pre-filtering)')

    # Optional: QC
    if qc:
        adata = quality_control_filter(adata)

    # Check if perturbation column is in adata
    if 'perturbation' not in adata.obs.columns:
        hits = list(adata.obs.columns.intersection(_perturbation_columns()))
        if len(hits) == 0:
            raise ValueError('No perturbation column found. Use one of: perturbation, gene, or perturbation_1')
        # Set to first hit
        adata.obs['perturbation'] = adata.obs[hits[0]]
    adata.obs['perturbation_base'] = adata.obs['perturbation'].str.replace(r'g\d+$', '', regex=True)
    perturbations = adata.obs['perturbation_base'].unique()
    print("Perturbations (cleaned):", perturbations)

    # Expression filtering (keep perturbed genes no matter what)
    gene_means = adata.X.mean(axis=0).A1 if issparse(adata.X) else adata.X.mean(axis=0)
    valid_genes = np.where(gene_means >= expression_threshold)[0]
    # Check if var names are gene names, change if not
    if len(set(perturbations).intersection(adata.var_names)) == 0:
        # Find gene column
        logging.info('No intersection between perturbations and gene names, converting adata.var_names to gene names.')
        hit_list = list(adata.var.columns.intersection(_gene_name_keys()))
        if len(hit_list) == 0:
            raise ValueError('Could not find a valid gene name column in adata.var.')
        # Select first hit out of search query
        gene_name_col = hit_list[0]
        # Reset current index and save as ens_id, update index and .var_names slot in adata
        adata.var.reset_index(names=['ens_id'], inplace=True)
        adata.var.index = adata.var[gene_name_col].tolist()
        # Ensure unique var names
        adata.var_names_make_unique()
    # Filter genes for expression or perturbation
    gene_names_to_keep = set(adata.var_names[valid_genes]) | set(perturbations)
    adata = adata[:, adata.var_names.isin(gene_names_to_keep)].copy()

    # Perturbation filtering
    perturbation_counts = adata.obs['perturbation'].value_counts()
    valid_perts = perturbation_counts[perturbation_counts >= min_samples].index.tolist()
    adata = adata[adata.obs['perturbation'].isin(valid_perts)].copy()

    # Save metadata
    perturbations = adata.obs['perturbation'].unique()
    selected_pert_name = perturbations[selected_pert_index]
    genes = adata.var_names.tolist()
    if save:
        np.save('perturbations.npy', perturbations)

    print(f"{len(perturbations)} perturbations")
    print(f"{len(genes)} genes after filtering")

    # Check control key
    ctrl_hits = set(perturbations).intersection(_control_keys())
    if len(ctrl_hits) == 0:
        raise ValueError(f'No control group found. Has to be one of {_control_keys()}.')
    ctrl_key = list(ctrl_hits)[0]
    # Set control as key
    ctrl_mask = adata.obs['perturbation'] == ctrl_key
    # Save as values only and change the categories
    adata.obs['perturbation'] = adata.obs['perturbation'].values.tolist()
    adata.obs.loc[ctrl_mask, 'perturbation'] = 'control'
    adata.obs['perturbation'] = pd.Categorical(adata.obs['perturbation'])
    # Filter for control and selected perturbation
    control_data = adata[ctrl_mask]
    selected_pert_data = adata[adata.obs['perturbation'] == selected_pert_name]

    X0 = control_data.X.toarray() if issparse(control_data.X) else control_data.X
    X1 = selected_pert_data.X.toarray() if issparse(selected_pert_data.X) else selected_pert_data.X

    print('shapes X0, X1:', X0.shape, X1.shape)

    # --- Sparsity after filtering ---
    full_X = adata.X.toarray() if issparse(adata.X) else adata.X
    compute_sparsity(full_X, label='Full matrix (post-filtering)')
    compute_sparsity(X0, label='X0 (control)')
    compute_sparsity(X1, label='X1 (perturbation)')

    return adata, X0, X1

def filter_and_prepare_adata(adata, expression_threshold=1.0, min_samples=100):
    full_X_pre = adata.X.toarray() if issparse(adata.X) else adata.X
    compute_sparsity(full_X_pre, 'Full matrix (pre-filtering)')

    # Add perturbation_base column (e.g. "TP53_g1" -> "TP53")
    adata.obs['perturbation_base'] = adata.obs['perturbation'].str.replace(r'_g\d+$', '', regex=True)

    # Filter perturbations by sample count FIRST
    pert_counts = adata.obs['perturbation'].value_counts()
    valid_perts = pert_counts[pert_counts >= min_samples].index.tolist()
    adata = adata[adata.obs['perturbation'].isin(valid_perts)].copy()

    # Get genes targeted by surviving perturbations
    target_genes = set(adata.obs['perturbation_base'].unique())
    target_genes_in_var = target_genes & set(adata.var_names)

    # Get genes with sufficient expression
    gene_means = adata.X.mean(axis=0).A1 if issparse(adata.X) else adata.X.mean(axis=0)
    expressed_genes = set(adata.var_names[np.where(gene_means >= expression_threshold)[0]])

    # Union: keep all expressed genes and all targeted genes (if in var_names)
    genes_to_keep = expressed_genes | target_genes_in_var
    adata = adata[:, adata.var_names.isin(genes_to_keep)]

    return adata

def get_matched_data(data_path_1, data_path_2, expression_threshold=1.0, min_samples=100):
    adata1 = ad.read_h5ad(data_path_1)
    adata2 = ad.read_h5ad(data_path_2)

    # Add base gene name from perturbations
    adata1.obs['perturbation_base'] = adata1.obs['perturbation'].str.replace(r'_g\d+$', '', regex=True)
    adata2.obs['perturbation_base'] = adata2.obs['perturbation'].str.replace(r'_g\d+$', '', regex=True)

    # Filter perturbations by sample count
    valid_perts1 = adata1.obs['perturbation'].value_counts()
    valid_perts1 = valid_perts1[valid_perts1 >= min_samples].index.tolist()
    adata1 = adata1[adata1.obs['perturbation'].isin(valid_perts1)].copy()

    valid_perts2 = adata2.obs['perturbation'].value_counts()
    valid_perts2 = valid_perts2[valid_perts2 >= min_samples].index.tolist()
    adata2 = adata2[adata2.obs['perturbation'].isin(valid_perts2)].copy()

    # Collect all targeted genes in both datasets
    target_genes = set(adata1.obs['perturbation_base']) | set(adata2.obs['perturbation_base'])

    # Filter gene expression
    def get_expressed_genes(adata):
        gene_means = adata.X.mean(axis=0).A1 if issparse(adata.X) else adata.X.mean(axis=0)
        return set(adata.var_names[np.where(gene_means >= expression_threshold)[0]])

    expressed1 = get_expressed_genes(adata1)
    expressed2 = get_expressed_genes(adata2)

    all_genes_to_keep = (expressed1 | expressed2 | target_genes) & set(adata1.var_names) & set(adata2.var_names)

    # Keep those genes in both datasets
    adata1 = adata1[:, adata1.var_names.isin(all_genes_to_keep)].copy()
    adata2 = adata2[:, adata2.var_names.isin(all_genes_to_keep)].copy()

    # Harmonize by intersection after ensuring target genes are included
    shared_genes = adata1.var_names.intersection(adata2.var_names)
    adata1 = adata1[:, shared_genes]
    adata2 = adata2[:, shared_genes]

    print(f"Matched {len(shared_genes)} genes after filtering")
    print(f"{adata1.obs['perturbation'].nunique()} perturbations in dataset 1")
    print(f"{adata2.obs['perturbation'].nunique()} perturbations in dataset 2")

    np.save("perturbations_1.npy", adata1.obs['perturbation'].unique())
    np.save("perturbations_2.npy", adata2.obs['perturbation'].unique())

    # Extract ctrl data from each adata
    control_data_1 = adata1[adata1.obs['perturbation']=='control']
    control_data_2 = adata2[adata2.obs['perturbation']=='control']
    X0_1 = control_data_1.X.toarray() if issparse(control_data_1.X) else control_data_1.X
    X0_2 = control_data_2.X.toarray() if issparse(control_data_2.X) else control_data_2.X

    compute_sparsity(X0_1, label='X0 (dataset 1, control)')
    compute_sparsity(X0_2, label='X0 (dataset 2, control)')

    return adata1, adata2, X0_1, X0_2
