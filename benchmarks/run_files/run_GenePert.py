"""GenePert benchmark driver (ridge regression on GenePT gene embeddings).

Every path comes from a CLI flag; nothing is hardcoded.  Each dataset runs in its
own ``multiprocessing.Process`` so that ``PEAK_CPU_RAM_GB`` is measured per
dataset.

Results are written to::

    <output-dir>/GenePert/<dataset>/<alpha>.pkl

i.e. the pickle is named after the ridge ``alpha`` (not ``results.pkl``), so an
alpha sweep accumulates side by side in the same folder -- e.g. ``0.1.pkl``,
``1.0.pkl``.  The default ``--alpha 0.1`` reproduces the original run.

Example::

    python run_GenePert.py \\
        --splits-dir   ../dataset_splits \\
        --output-dir   ../results \\
        --resources-dir ../resources \\
        --models-dir   ../models \\
        --dataset ReplogleWeissman2022_rpe1 NormanWeissman2019_filtered

Omitting ``--dataset`` runs every dataset found under ``--splits-dir``.

Requires ``<resources-dir>/GenePT_gene_protein_embedding_model_3_text.pickle``
and a vendored ``<models-dir>/GenePert/`` source tree.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench_common as bench

import logging
import multiprocessing as mp
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def build_training_means(control_adata, perturbed_backed, indices):
    import numpy as np
    import pandas as pd
    import anndata as ad

    indices_arr = np.asarray(indices)
    train_X = perturbed_backed.X[indices_arr]
    train_perts = perturbed_backed.obs['perturbation'].iloc[indices_arr].values
    mean_rows, pert_list = [], []
    for pert in np.unique(train_perts):
        mask = train_perts == pert
        mean_expr = np.asarray(train_X[mask].mean(axis=0), dtype=np.float32).flatten()
        mean_rows.append(mean_expr)
        pert_list.append(pert)
    obs_df = pd.DataFrame({'perturbation': pert_list})
    return ad.AnnData(X=np.vstack(mean_rows), obs=obs_df, var=control_adata.var)


def pearson_matrix(A, B):
    import numpy as np

    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    return A @ B.T


def run_genepert(dataset_name, splits_dir, output_dir, models_dir, gene_embeddings,
                 alpha=0.1):
    import importlib
    import resource
    import time

    import numpy as np
    from scipy.stats import spearmanr

    bench.add_model_to_syspath(models_dir, "GenePert")
    import utils  # noqa: F401  (vendored GenePert helper module)
    import GenePertExperiment
    importlib.reload(utils)
    importlib.reload(GenePertExperiment)

    import scanpy as sc

    start_time = time.time()

    # DATA LOADING =========================================================================

    paths = bench.split_paths(splits_dir, dataset_name)

    full_set = sc.read_h5ad(str(paths["adata"]))
    ram_before_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    full_set.layers['counts'] = full_set.X.copy()
    sc.pp.normalize_total(full_set, target_sum=1e4)
    sc.pp.log1p(full_set)
    ram_after_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    control_indices = np.load(paths["control"])
    training_indices = np.load(paths["train"])
    testing_indices = np.load(paths["test"])
    control_set = full_set[control_indices]

    ridge_param = {'alpha': alpha}
    logging.info("Collected indices")

    logging.info("Loading test set...")
    id_testing_set = full_set[testing_indices].to_memory()
    logging.info(f"Test set loaded: {id_testing_set.shape}")

    logging.info("Building training means...")
    training_set = build_training_means(control_set, full_set, training_indices)
    logging.info(f"Training set built: {training_set.shape}")

    known_embeddings = set(gene_embeddings.keys())
    missing_train = [p for p in training_set.obs['perturbation'].unique() if p not in known_embeddings]
    missing_test  = [p for p in id_testing_set.obs['perturbation'].unique() if p not in known_embeddings]
    if missing_train or missing_test:
        print(f"Dropping {len(missing_train)} train / {len(missing_test)} test perturbations with no gene embedding")
        training_set = training_set[~training_set.obs['perturbation'].isin(missing_train)].copy()
        id_testing_set = id_testing_set[~id_testing_set.obs['perturbation'].isin(missing_test)].copy()

    end_data_loading_time = time.time()
    data_loading_time = end_data_loading_time - start_time
    print("Data loading time: " + str(round(data_loading_time, 2)) + " s")

    # MODEL SETUP =========================================================================

    logging.info("Computing control mean...")
    id_experiment = GenePertExperiment.GenePertExperiment(embeddings=gene_embeddings)
    ctrl_mean = np.asarray(control_set.X.mean(axis=0), dtype=np.float32).flatten()
    raw_ctrl_mean = np.asarray(control_set.layers['counts'].mean(axis=0), dtype=np.float32).flatten()
    id_experiment.mean_expression = ctrl_mean

    end_setup_time = time.time()
    setup_time = end_setup_time - end_data_loading_time
    print("Setup time: " + str(round(setup_time, 2)) + " s")

    # MODEL TRAINING =========================================================================

    logging.info("Running experiment...")
    id_results = id_experiment.run_experiment_with_adata(adata_train=training_set, adata_test=id_testing_set, ridge_params=[ridge_param], knn_params=[])
    logging.info("Experiment done")

    end_training_time = time.time()
    training_time = end_training_time - end_setup_time
    print("Training time: " + str(round(training_time, 2)) + " s")

    # EVALUATION =========================================================================

    ridge_key = tuple(ridge_param.items())
    print(ridge_param)
    for split_name, results, test_set, mean_expression in [
    ("id_test",  id_results,  id_testing_set,  id_experiment.mean_expression),]:

        # Build perturbation to predicted mean lookup
        pertubation_names = list(results['per_gene'].keys())
        predicted_means = np.array([results['per_gene'][p]['ridge'][ridge_key][2] for p in pertubation_names], dtype=np.float32)

        # Centering the expressions
        GT_cells = np.array(test_set.to_df(), dtype=np.float32)
        GT_centered = GT_cells - mean_expression
        P_centered = predicted_means - mean_expression

        # Raw-space ground truth cells (aligned to test_set rows via the 'counts' layer)
        GT_cells_raw = np.array(test_set.to_df(layer='counts'), dtype=np.float32)

        score_matrix = pearson_matrix(GT_centered, P_centered)
        cell_perturbations  = test_set.obs['perturbation'].values
        missing_perts = set(cell_perturbations) - set(pertubation_names)
        if missing_perts:
            print(f"WARNING: {len(missing_perts)} perturbations in test set have no prediction: {missing_perts}")

        invalid_cell_idx = [i for i, p in enumerate(cell_perturbations) if p not in pertubation_names]
        score_matrix[invalid_cell_idx] = np.nan

        # Per-perturbation Pearson correlation, MSE, and R² (predicted mean vs actual mean)
        per_pert_corr = {p: results['per_gene'][p]['ridge'][ridge_key][0].item() for p in pertubation_names}
        # GenePertExperiment.evaluate_performance_rowwise returns RMSE, so square it back to MSE.
        per_pert_mse  = {p: results['per_gene'][p]['ridge'][ridge_key][1].item() ** 2 for p in pertubation_names}
        per_pert_r2 = {}
        per_pert_spearman = {}
        per_pert_l2 = {}
        per_pert_cosine = {}
        for p in pertubation_names:
            y_true_p = np.asarray(results['per_gene'][p]['ridge'][ridge_key][3]).flatten() - mean_expression
            y_pred_p = np.asarray(results['per_gene'][p]['ridge'][ridge_key][2]).flatten() - mean_expression
            ss_res = np.sum((y_true_p - y_pred_p) ** 2)
            ss_tot = np.sum((y_true_p - y_true_p.mean()) ** 2)
            per_pert_r2[p] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            per_pert_spearman[p] = spearmanr(y_true_p, y_pred_p).statistic
            per_pert_l2[p] = np.linalg.norm(y_true_p - y_pred_p)
            per_pert_cosine[p] = np.dot(y_true_p, y_pred_p) / (
                np.linalg.norm(y_true_p) * np.linalg.norm(y_pred_p))
        mean_corr = np.mean(list(per_pert_corr.values()))
        mean_mse = np.mean(list(per_pert_mse.values()))
        mean_r2 = np.mean(list(per_pert_r2.values()))
        mean_spearman = np.mean(list(per_pert_spearman.values()))
        mean_l2 = np.mean(list(per_pert_l2.values()))
        mean_cosine = np.mean(list(per_pert_cosine.values()))

        end_evaluation_time = time.time()
        evaluation_time = end_evaluation_time - end_training_time
        print("Evaluation time: " + str(round(evaluation_time, 2)) + " s")

        # predicted gene expression change over the control in the log space
        delta_y = {p: P_centered[i] for i, p in enumerate(pertubation_names)}

        # ground truth gene expression change over the control in raw space
        delta_x = {p: GT_cells_raw[cell_perturbations == p].mean(axis=0) - raw_ctrl_mean for p in pertubation_names}

        # mean control gene expression in raw space
        control = {p: raw_ctrl_mean for p in pertubation_names}

        total_time = time.time() - start_time
        print("Total Elapsed time: " + str(round(total_time, 2)) + " s")

        # Track resource usage
        peak_cpu_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        peak_cpu_gb -= ram_after_counts_copy - ram_before_counts_copy
        print(f"Peak CPU RAM: {peak_cpu_gb:.2f} GB")

        exported_results = {"CORRELATION": per_pert_corr,
                            "SPEARMAN": per_pert_spearman,
                            "MSE": per_pert_mse,
                            "R2": per_pert_r2,
                            "L2": per_pert_l2,
                            "COSINE": per_pert_cosine,
                            "DELTA_Y": delta_y,
                            "CONTROL": control,
                            "DELTA_X": delta_x,
                            "DATA_LOADING_TIME": data_loading_time,
                            "SETUP_TIME": setup_time,
                            "TRAINING_TIME": training_time,
                            "EVALUATION_TIME": evaluation_time,
                            "TOTAL_TIME": total_time,
                            "PEAK_CPU_RAM_GB": peak_cpu_gb}

        # Save the results, named after the ridge alpha so a sweep accumulates side by side
        exported_results_folder = bench.results_dir(output_dir, "GenePert", dataset_name)
        with open(exported_results_folder / (str(ridge_param["alpha"]) + ".pkl"), "wb") as file:
            pickle.dump(exported_results, file)

        print(f"  [{split_name}] mean Pearson: {mean_corr:.4f} | mean Spearman: {mean_spearman:.4f} | mean MSE: {mean_mse:.4f} | mean R²: {mean_r2:.4f} | mean L2: {mean_l2:.4f} | mean Cosine: {mean_cosine:.4f} over {len(pertubation_names)} perturbations")


def main(argv=None):
    parser = bench.build_parser(
        "Run the GenePert benchmark (ridge regression on GenePT embeddings).",
        resources=True, models=True,
    )
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="ridge regression alpha; also names the output pickle "
                             "(<output-dir>/GenePert/<dataset>/<alpha>.pkl)")
    args = parser.parse_args(argv)

    datasets = bench.discover_datasets(args.splits_dir, args.dataset)

    # Loaded once, before forking, so the embedding table is shared copy-on-write
    # across the per-dataset workers and does not inflate their PEAK_CPU_RAM_GB.
    embedding_path = bench.resource_path(
        args.resources_dir, "GenePT_gene_protein_embedding_model_3_text.pickle")
    with open(embedding_path, 'rb') as f:
        gene_embeddings = pickle.load(f)
    print("Loaded gene embeddings")

    for dataset_name in datasets:
        print(dataset_name)
        # Run in a subprocess so that peak RAM usage is localised to each dataset
        p = mp.Process(target=run_genepert, kwargs={
            "dataset_name": dataset_name,
            "splits_dir": args.splits_dir,
            "output_dir": args.output_dir,
            "models_dir": args.models_dir,
            "gene_embeddings": gene_embeddings,
            "alpha": args.alpha,
        })
        p.start()
        p.join()
        print("\n_____________________________________________")
    return 0


if __name__ == "__main__":
    sys.exit(main())
