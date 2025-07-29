import os
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import pandas as pd
import scipy.sparse as sp

from gears import PertData, GEARS
import argparse

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def ensure_gears_compatible_dataset(
        adata_p: str, 
        perturbation_col: str = 'perturbation', 
        condition_key: str = 'condition',
        full_condition_key: str = 'condition_name',
        ct_key: str = 'cell_type',
        orig_ct_key: str = 'celltype',
        orig_ctrl_key: str = 'control',
        ctrl_key: str = 'ctrl',
        gene_key: str = 'gene_name',
        cache: bool = False,
        pp: bool = True,
    ) -> None:
    logging.info(f'Checking dataset: {adata_p}')
    if not os.path.exists(adata_p):
        logging.info(f'Adata object does not yet exists, passing to GEARS to look for downloads.')
        return None
    # Load dataset in backed mode, i.e. only look at meta data and check if update is needed or not
    adata = sc.read(adata_p, backed='r')
    no_update = cache and condition_key in adata.obs.columns and adata.obs[condition_key].str.endswith(f'+{ctrl_key}').any() and gene_key in adata.var
    # Close dangling file connection
    adata.file.close()
    if no_update:
        logging.info(f'Dataset is compatible with GEARS. No updates needed.')
        return None
    logging.info(f'Dataset is not compatible with GEARS. Updating meta-data. Reading dataset.')
    # Read full adata to update it
    adata = sc.read(adata_p)
    if not sp.issparse(adata.X):
        logging.info('\t- .X: Converting to sparse matrix (csr).')
        adata.X = sp.csr_matrix(adata.X)
    if pp:
        logging.info('\t- .X: Normalizing to 10,000 counts, applying log1p tranformation.')
        sc.pp.normalize_total(adata, target_sum=1e5)
        sc.pp.log1p(adata)
    
    # .obs updates
    logging.info(f'\t- .obs: Adding "{condition_key}" from "{perturbation_col}"')
    # Add condition column to adata.obs TODO: handle multiple perturbations
    adata.obs[condition_key] = adata.obs[perturbation_col].astype(str) + f"+{ctrl_key}"
    # Re-name control cells to ctrl only
    ctrl_mask = adata.obs[perturbation_col] == orig_ctrl_key
    adata.obs.loc[ctrl_mask,condition_key] = ctrl_key
    logging.info(f'\t- .obs: Renaming "{orig_ctrl_key}" to "{ctrl_key}" in "{condition_key}"')
    # Rename control cells to gears control key
    adata.obs.loc[adata.obs[condition_key]==orig_ctrl_key,condition_key] = ctrl_key
    adata.obs[condition_key] = pd.Categorical(adata.obs[condition_key])
    # Adding full condition name
    logging.info(f'\t- .obs: Adding "{full_condition_key}" from "{condition_key}" and "{orig_ct_key}"')
    if orig_ct_key not in adata.obs.columns:
        adata.obs[orig_ct_key] = 'unknown'
    adata.obs[ct_key] = adata.obs[orig_ct_key]
    adata.obs[full_condition_key] = adata.obs[orig_ct_key].astype(str) + '_' + adata.obs[condition_key].astype(str)
    logging.info(f'\t- .obs: Adding "control" boolean flag from "{orig_ctrl_key}" in "{perturbation_col}"')
    adata.obs['control'] = (ctrl_mask).astype(int)
    logging.info(f'\t- .obs: Adding "dose_val" info from "{perturbation_col}"')
    adata.obs['dose_val'] = '1+1'
    logging.info(f'\t- .obs: Adding "cov_drug_dose_name" info from "{full_condition_key}" and "dose_val"')
    adata.obs['cov_drug_dose_name'] = adata.obs[full_condition_key].astype(str) + '_' + adata.obs['dose_val'].astype(str)

    # .var updates
    logging.info(f'\t- .var: Adding "{gene_key}" from adata.var.index')
    # Add gene names in adata.var
    adata.var[gene_key] = pd.Categorical(adata.var.index.astype(str))
    # Rename adata index to ensure that the name is not duplicated
    if adata.var.index.name == gene_key:
        idx_name = 'var_idx'
        logging.info(f'\t- .var: Renaming adata.var.index to "{idx_name}"')
        adata.var.index.name = idx_name
    # Save updated adata
    adata.write_h5ad(adata_p)

def compute_average_response(X0, X1):
    return X1.mean(axis=0) - X0.mean(axis=0)

def compute_soft_fraction(Sigma, u, threshold_mode='fraction_variance', threshold_value=0.7):
    lambda_vals, V = np.linalg.eigh(Sigma)
    idx = np.argsort(lambda_vals)[::-1]
    lambda_vals = lambda_vals[idx]
    V = V[:, idx]
    c = V.T @ u
    c2 = c ** 2
    if threshold_mode == 'fraction_variance':
        total_var = np.sum(lambda_vals)
        cum_var = np.cumsum(lambda_vals)
        soft_indices = np.where(cum_var <= threshold_value * total_var)[0]
    elif threshold_mode == 'relative_max':
        lambda_max = np.max(lambda_vals)
        soft_indices = np.where(lambda_vals >= threshold_value * lambda_max)[0]
    elif threshold_mode == 'elbow':
        diffs = np.diff(lambda_vals)
        second_diffs = np.diff(diffs)
        elbow_idx = np.argmax(second_diffs)
        soft_indices = np.arange(elbow_idx + 1)
    else:
        raise ValueError("Invalid threshold_mode.")
    f_soft = np.sum(c2[soft_indices]) / np.sum(c2)
    return f_soft, soft_indices


def parse_args():
    parser = argparse.ArgumentParser(description='Run GEARS model')
    parser.add_argument('--input', type=str, required=True, help='Path to h5ad input file')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--test_batch_size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def run_gears(params) -> None:
    # Unpack arguments
    adata_p = params.input                   # path/to/data/ds/perturb_processed.h5ad
    out_dir = params.outdir
    batch_size = params.batch_size
    test_batch_size = params.test_batch_size
    hidden_size = params.hidden_size
    epochs = params.epochs
    seed = params.seed
    # Set I/O
    ds_dir = os.path.dirname(adata_p)        # path/to/data/ds
    data_dir = os.path.dirname(ds_dir)       # path/to/data/
    ds_name = os.path.basename(ds_dir)       # ds
    o = os.path.join(out_dir, 'gears')
    pred_o = os.path.join(o, 'predictions.csv')
    os.makedirs(o, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Running on {device.upper()}')
    # get data
    logging.info(f'Setting up PertData directory: {data_dir}')
    pert_data = PertData(data_dir)
    # Create new data process
    logging.info(f'Creating new compatible dataset from {adata_p}')
    ensure_gears_compatible_dataset(adata_p)
    pert_data.new_data_process(dataset_name=ds_name, adata=sc.read(adata_p))
    logging.info(f'Loading dataset: {ds_name}')
    pert_data.load(data_path = ds_dir)
    # specify data split
    pert_data.prepare_split(split = 'simulation', seed = seed)
    # get dataloader with batch size
    pert_data.get_dataloader(batch_size = batch_size, test_batch_size = test_batch_size)

    # set up and train a model
    logging.info(f'Training model.')
    gears_model = GEARS(pert_data, device = device)
    gears_model.model_initialize(hidden_size = hidden_size)
    gears_model.train(epochs = epochs)

    # save/load model
    gears_model.save_model(o)
    logging.info(f'Done training. Saved model to: {o}')
    # Predict perturbation effects
    cells_per_perturbation = pert_data.adata.obs.perturbation.value_counts()
    # Filter perturbations
    valid_perturbations = cells_per_perturbation.index
    # Remove control group and convert to list
    valid_perturbations = valid_perturbations[valid_perturbations!='control'].tolist()
    # Convert to list of lists for prediction
    perturbation_list = [[v] for v in valid_perturbations]
    # Predict expression
    logging.info(f'Predicting perturbation effects for {cells_per_perturbation.shape[0]} perturbations')
    predictions = gears_model.predict(perturbation_list)
    # Save predictions to file
    logging.info(f'Done, saving predictions to: {pred_o}')
    pred_df = pd.DataFrame(predictions, index=pert_data.adata.var_names.tolist()).T
    pred_df.to_csv(pred_o)

if __name__ == '__main__':
    args = parse_args()
    run_gears(args)
