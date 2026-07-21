"""Benchmark driver for scLAMBDA.

Trains one scLAMBDA model per dataset and writes the standard CIPHER benchmark
``results.pkl`` to ``<output-dir>/scLAMBDA/<dataset>/``.

Every path comes from a CLI flag; nothing is hardcoded. Each dataset is run in
its own ``multiprocessing.Process`` so that ``PEAK_CPU_RAM_GB`` (measured with
``resource.getrusage``) is per-dataset rather than cumulative.

Example
-------
    python run_scLAMBDA.py \
        --splits-dir   /path/to/benchmarks/dataset_splits \
        --output-dir   /path/to/benchmarks/results \
        --resources-dir /path/to/benchmarks/resources \
        --models-dir   /path/to/benchmarks/models \
        --dataset ReplogleWeissman2022_rpe1 NormanWeissman2019_filtered

Omit ``--dataset`` to run every dataset found under ``--splits-dir``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench_common as bench

import numpy as np

MODEL = "scLAMBDA"
GENEPT_PICKLE = "GenePT_gene_protein_embedding_model_3_text.pickle"


def to_dense(x):
    from scipy import sparse
    return np.asarray(x.toarray(), dtype=np.float32) if sparse.issparse(x) else np.asarray(x, dtype=np.float32)


def run_sclambda(dataset_name, splits_dir, output_dir, resources_dir, models_dir,
                 device="cuda"):
    print(dataset_name)

    # Heavy third-party imports live here so that --help works without the
    # scLAMBDA conda environment.
    if str(device).startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import pandas as pd
    import anndata as ad
    import scanpy as sc
    import torch
    import pickle
    import time
    import resource
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, r2_score

    bench.add_model_to_syspath(models_dir, MODEL)
    import sclambda

    start_time = time.time()
    if torch.cuda.is_available():
        if ":" in str(device):
            torch.cuda.set_device(str(device))
        torch.cuda.reset_peak_memory_stats()

    # DATA LOADING =========================================================================
    # NOTE: the loading below is deliberately kept inline rather than going through
    # bench.load_split_adata(); scLAMBDA needs normalisation/log1p applied before the
    # condition rename, and builds obs['split'] itself via sclambda.utils.data_split.
    # Only the *paths* come from the shared helper.
    paths = bench.split_paths(splits_dir, dataset_name)

    gene_embeddings = pd.read_pickle(str(bench.resource_path(resources_dir, GENEPT_PICKLE)))
    gene_embeddings = {k: np.array(v) for k, v in gene_embeddings.items()}

    adata_processed = ad.read_h5ad(str(paths["adata"]))
    if "gene_name" in adata_processed.var.columns:
        adata_processed.var.index = adata_processed.var["gene_name"]

    ram_before_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    adata_processed.layers['counts'] = adata_processed.X.copy()
    sc.pp.normalize_total(adata_processed, target_sum=1e4)
    sc.pp.log1p(adata_processed)
    ram_after_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

    adata_processed.obs.drop(columns=["condition"], errors="ignore", inplace=True)
    adata_processed.obs.rename(columns={"perturbation": "condition"}, inplace=True)
    adata_processed.obs["condition"] = adata_processed.obs["condition"].apply(lambda x: "ctrl" if x == "control" else x)

    control_indices = np.load(paths["control"])
    training_indices = np.load(paths["train"])
    testing_indices = np.load(paths["test"])

    training_indices = np.concatenate([training_indices, control_indices])

    adata_processed, split = sclambda.utils.data_split(adata_processed, split_type="CIPHER_benchmarking",
                                                    training_indices=training_indices, testing_indices=testing_indices)

    # This is just a sanity check, the upstream data generation already takes care of missing
    missing_gene_conditions = {
        cond for cond in adata_processed.obs['condition'].unique()
        if cond != 'ctrl' and any(gene not in gene_embeddings for gene in cond.split('+') if gene != 'ctrl')
    }

    if missing_gene_conditions:
        print(f"Skipping {len(missing_gene_conditions)} perturbation(s) with no GenePT embedding: "
              f"{sorted(missing_gene_conditions)}")
        adata_processed = adata_processed[~adata_processed.obs['condition'].isin(missing_gene_conditions)].copy()

    exported_results_folder = bench.results_dir(output_dir, MODEL, dataset_name)

    end_data_loading_time = time.time()
    data_loading_time = end_data_loading_time - start_time
    print("Data loading time: " + str(round(data_loading_time, 2)) + " s")

    # SETUP AND TRAIN MODEL =========================================================================

    model = sclambda.model.Model(adata_processed,
                             gene_embeddings,
                             model_path = str(exported_results_folder),
                             multi_gene = False)

    model.train()

    end_training_time = time.time()
    training_time = end_training_time - end_data_loading_time
    print("Training time: " + str(round(training_time, 2)) + " s")

    # EVALUATE MODEL =========================================================================

    ctrl_mean = np.asarray(model.ctrl_mean, dtype=np.float32).ravel()

    raw_ctrl_mask = adata_processed.obs['condition'].values == 'ctrl'
    raw_ctrl_mean = to_dense(adata_processed[raw_ctrl_mask].layers['counts']).mean(axis=0)

    pert_test = sorted(adata_processed.obs.loc[adata_processed.obs['split'] == 'test', 'condition'].unique().tolist())

    res = model.generate(pert_test, return_type='mean')

    per_pert_corr = {}
    per_pert_spearman = {}
    per_pert_mse = {}
    per_pert_r2 = {}
    per_pert_l2 = {}
    per_pert_cosine = {}
    delta_y = {}
    delta_x = {}
    control = {}

    for i in pert_test:
        pred_centered = np.asarray(res[i], dtype=np.float32).ravel() - ctrl_mean

        gt_mask = (adata_processed.obs['condition'].values == i) & (adata_processed.obs['split'].values == 'test')
        gt_cells = to_dense(adata_processed[gt_mask].X)
        gt_mean_centered = np.mean(gt_cells, axis=0) - ctrl_mean

        per_pert_corr[i] = np.corrcoef(gt_mean_centered, pred_centered)[0, 1]
        per_pert_spearman[i] = spearmanr(gt_mean_centered, pred_centered).statistic
        per_pert_mse[i] = mean_squared_error(gt_mean_centered, pred_centered)
        per_pert_r2[i] = r2_score(gt_mean_centered, pred_centered)
        per_pert_l2[i] = np.linalg.norm(gt_mean_centered - pred_centered)
        per_pert_cosine[i] = np.dot(gt_mean_centered, pred_centered) / (np.linalg.norm(gt_mean_centered) * np.linalg.norm(pred_centered))

        # predicted gene expression change over the control in the log space
        delta_y[i] = pred_centered

        # ground truth gene expression change over the control in raw space
        gt_cells_raw = to_dense(adata_processed[gt_mask].layers['counts'])
        delta_x[i] = np.mean(gt_cells_raw, axis=0) - raw_ctrl_mean

        # mean control gene expression in raw space
        control[i] = raw_ctrl_mean

    end_evaluation_time = time.time()
    evaluation_time = end_evaluation_time - end_training_time
    print("Evaluation time: " + str(round(evaluation_time, 2)) + " s")

    nan_perts = sorted(p for p, v in per_pert_corr.items() if np.isnan(v))
    if nan_perts:
        print(f"WARNING: {len(nan_perts)} perturbations have NaN correlation "
            f"(likely a zero-variance ground-truth or predicted profile): {nan_perts}")

    mean_corr = np.nanmean(list(per_pert_corr.values()))
    mean_spearman = np.nanmean(list(per_pert_spearman.values()))
    mean_mse = np.nanmean(list(per_pert_mse.values()))
    mean_r2 = np.nanmean(list(per_pert_r2.values()))
    mean_l2 = np.nanmean(list(per_pert_l2.values()))
    mean_cosine = np.nanmean(list(per_pert_cosine.values()))

    total_time = time.time() - start_time
    print("Total Elapsed time: " + str(round(total_time, 2)) + " s")

    peak_cpu_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    peak_cpu_gb -= ram_after_counts_copy - ram_before_counts_copy
    if torch.cuda.is_available():
        peak_gpu_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_gpu_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    else:
        peak_gpu_allocated_gb = None
        peak_gpu_reserved_gb = None
    print(f"Peak CPU RAM: {peak_cpu_gb:.2f} GB | Peak GPU allocated: {peak_gpu_allocated_gb} GB | Peak GPU reserved: {peak_gpu_reserved_gb} GB")

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
                        "TRAINING_TIME": training_time,
                        "EVALUATION_TIME": evaluation_time,
                        "TOTAL_TIME": total_time,
                        "PEAK_CPU_RAM_GB": peak_cpu_gb,
                        "PEAK_GPU_ALLOCATED_GB": peak_gpu_allocated_gb,
                        "PEAK_GPU_RESERVED_GB": peak_gpu_reserved_gb}

    with open(str(exported_results_folder / "results.pkl"), "wb") as file:
        pickle.dump(exported_results, file)

    # Remove the saved model as it will take too much space otherwise
    # NOTE THIS REMOVES FILES SO PLEASE BE CAREFUL!!!
    os.remove(str(exported_results_folder / "ckpt.pth"))

    print(f"mean Pearson: {mean_corr:.4f} | mean Spearman: {mean_spearman:.4f} | mean MSE: {mean_mse:.4f} | mean R²: {mean_r2:.4f} | mean L2: {mean_l2:.4f} | mean Cosine: {mean_cosine:.4f} over {len(per_pert_corr)} perturbations")


def main(argv=None):
    import multiprocessing as mp

    parser = bench.build_parser(
        "Run the scLAMBDA benchmark over one or more perturbation datasets.",
        resources=True, models=True, device=True,
    )
    args = parser.parse_args(argv)

    datasets = bench.discover_datasets(args.splits_dir, args.dataset)
    print(f"Running {MODEL} on {len(datasets)} dataset(s): {', '.join(datasets)}")

    for dataset_name in datasets:
        # Run in a subprocess so that peak RAM usage is localised to each dataset
        p = mp.Process(
            target=run_sclambda,
            kwargs={
                "dataset_name": dataset_name,
                "splits_dir": args.splits_dir,
                "output_dir": args.output_dir,
                "resources_dir": args.resources_dir,
                "models_dir": args.models_dir,
                "device": args.device,
            },
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: {dataset_name} exited with code {p.exitcode}")
        print("\n_____________________________________________")

    return 0


if __name__ == "__main__":
    sys.exit(main())
