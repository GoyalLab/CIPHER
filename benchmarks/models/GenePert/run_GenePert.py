import pandas as pd
import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
import gzip

import pickle
import sys
sys.path.insert(0, "/gpfs/projects/p32655/irish/ExPert/Portable/GenePert")
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Reload the module
import utils
import GenePertExperiment
importlib.reload(utils)

# Reload the module
importlib.reload(GenePertExperiment)
from utils import plot_mse_corr_comparison, compare_embedding_correlations

import scanpy as sc
import anndata as ad

data_partition = "small"
gene_embedding_path = "/gpfs/projects/p32655/irish/ExPert_source/ExPert/resources/gene_embeddings/claude_sonnet_4.5/embeddings/claude-opus-4-7/openai-3-large-1024/openai-3-large/gene_embeddings.npz"

# Set up the gene embeddings
d = np.load(gene_embedding_path, allow_pickle=True)
emb_df = pd.DataFrame(d["embeddings"], index=d["genes"])
gene_embeddings = {gene: np.array(row) for gene, row in emb_df.iterrows()}
print("Loaded gene embeddings")

# Load the control and perturbed sets
if data_partition == "small":
    control_set = sc.read_h5ad("/gpfs/projects/b1255/expert/data/source/v1/ctrl.h5ad")
    perturbed_set = sc.read_h5ad("/gpfs/projects/b1255/expert/data/source/v1/small/source.h5ad", backed="r")
elif data_partition == "full":
    control_set = sc.read_h5ad("/gpfs/projects/p32655/expert/data/source/ctrl.h5ad")
    perturbed_set = sc.read_h5ad("/gpfs/projects/p32655/expert/data/source/full.h5ad", backed="r")
print("Loaded gene control and perturbed sets")

mask_ctrl = np.asarray(control_set.varm['feature_mask_per_dataset'])
mask_pert = np.asarray(perturbed_set.varm['feature_mask_per_dataset'])
shared_genes_mask = (mask_ctrl == 1).all(axis=1) & (mask_pert == 1).all(axis=1)
shared_gene_names = control_set.var_names[shared_genes_mask]
print("Shared genes: " + str(shared_genes_mask.sum()))

control_shared = control_set[:, shared_gene_names]


def build_training_means(control_adata, perturbed_backed, indices, genes):
    """
    GenePertExperiment.populate_dicts only needs per-perturbation mean expression.
    Build a compact AnnData (one row per perturbation) by loading one perturbation
    group at a time from the backed dataset, avoiding a full memory load.
    """
    mean_rows = []
    pert_list = []

    # Control cells are already in memory
    for pert in control_adata.obs['perturbation'].unique():
        mask = control_adata.obs['perturbation'] == pert
        mean_expr = np.asarray(control_adata[mask].X.mean(axis=0), dtype=np.float32).flatten()
        mean_rows.append(mean_expr)
        pert_list.append(pert)

    # Perturbed training cells: load one perturbation at a time from disk
    train_obs = perturbed_backed.obs.iloc[indices]
    indices_arr = np.array(indices)
    for pert in train_obs['perturbation'].unique():
        mask = (train_obs['perturbation'] == pert).values
        pert_global_idx = indices_arr[mask]
        pert_data = perturbed_backed[pert_global_idx].to_memory()[:, genes]
        mean_expr = np.asarray(pert_data.X.mean(axis=0), dtype=np.float32).flatten()
        mean_rows.append(mean_expr)
        pert_list.append(pert)
        del pert_data

    obs_df = pd.DataFrame({'perturbation': pert_list})
    return ad.AnnData(X=np.vstack(mean_rows), obs=obs_df, var=control_adata.var)

def pearson_matrix(A, B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    return A @ B.T

def run_genepert(fold):
    ridge_params = [(('alpha', 0.1),)]
    with gzip.open("/gpfs/projects/b1255/expert/data/source/v1/" + data_partition + "/fold_" + str(fold) + "/id_train.txt.gz", mode="rt") as file:
        training_indices = [int(line.rstrip()) for line in file]
    with gzip.open("/gpfs/projects/b1255/expert/data/source/v1/" + data_partition + "/fold_" + str(fold) + "/id_test.txt.gz", mode="rt") as file:
        id_testing_indices = [int(line.rstrip()) for line in file]
    with gzip.open("/gpfs/projects/b1255/expert/data/source/v1/" + data_partition + "/fold_" + str(fold) + "/id_val.txt.gz", mode="rt") as file:
        id_val_indices = [int(line.rstrip()) for line in file]
    with gzip.open("/gpfs/projects/b1255/expert/data/source/v1/" + data_partition + "/fold_" + str(fold) + "/ood_test.txt.gz", mode="rt") as file:
        ood_testing_indices = [int(line.rstrip()) for line in file]
    logging.info("Collected indices")

    # Load only the needed rows from disk, then subset to common genes in memory
    id_testing_set  = perturbed_set[id_testing_indices].to_memory()[:, shared_gene_names]
    ood_testing_set = perturbed_set[ood_testing_indices].to_memory()[:, shared_gene_names]

    held_out_datasets = ood_testing_set.obs['dataset'].unique().tolist()
    print("Held out datasets: " + str(held_out_datasets))

    # Build compact training AnnData (one row per perturbation) to avoid loading
    # all training cells at once — GenePertExperiment only needs per-perturbation means
    training_set = build_training_means(control_shared, perturbed_set, training_indices, shared_gene_names)

    print(training_set.shape, id_testing_set.shape, ood_testing_set.shape)

    id_experiment = GenePertExperiment.GenePertExperiment(embeddings=gene_embeddings)
    id_experiment.mean_expression  = np.array(control_shared.to_df().mean(), dtype=np.float32)
    id_results = id_experiment.run_experiment_with_adata(adata_train=training_set, adata_test=id_testing_set, ridge_params=[{'alpha': 0.1}])

    ood_experiment = GenePertExperiment.GenePertExperiment(embeddings=gene_embeddings)
    ood_experiment.mean_expression = np.array(control_shared[control_shared.obs["dataset"].isin(held_out_datasets)].to_df().mean(), dtype=np.float32)
    ood_results = ood_experiment.run_experiment_with_adata(adata_train=training_set, adata_test=ood_testing_set, ridge_params=[{'alpha': 0.1}])

    for ridge_param in ridge_params:
        print(ridge_param)
        for split_name, results, test_set, mean_expression in [
        ("id_test",  id_results,  id_testing_set,  id_experiment.mean_expression),
        ("ood_test", ood_results, ood_testing_set, ood_experiment.mean_expression),]:
            
            # Build perturbation to predicted mean lookup
            pertubation_names = list(results['per_gene'].keys())
            predicted_means = np.array([results['per_gene'][p]['ridge'][ridge_param][2] for p in pertubation_names], dtype=np.float32)

            # Centering the expressions 
            GT_cells = np.array(test_set.to_df(), dtype=np.float32)
            GT_centered = GT_cells - mean_expression
            P_centered = predicted_means - mean_expression

            score_matrix = pearson_matrix(GT_centered, P_centered)
            cell_perturbations  = test_set.obs['perturbation'].values
            missing_perts = set(cell_perturbations) - set(pertubation_names)
            if missing_perts:
                print(f"WARNING: {len(missing_perts)} perturbations in test set have no prediction: {missing_perts}")

            invalid_cell_idx = [i for i, p in enumerate(cell_perturbations) if p not in pertubation_names]
            score_matrix[invalid_cell_idx] = np.nan
            
            logits_save_folder = "/projects/p32655/expert/benchmarks/context/genepert/results/" + data_partition + "/fold_" + str(fold) + "/" + split_name
            logits_save_path = logits_save_folder + "/logits.csv"
            os.makedirs(logits_save_folder, exist_ok=True)
            df = pd.DataFrame(score_matrix, columns=pertubation_names).sort_index(axis=1, ascending=True)
            df.to_csv(logits_save_path, index=False)
            print("Exported logits file for fold " + str(fold) + " distribution " + data_partition + " to " + logits_save_path)

if __name__ == "__main__":
    fold = int(sys.argv[1])
    run_genepert(fold=fold)