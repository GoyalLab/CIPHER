"""Scouter benchmark driver.

Trains Scouter on each dataset's train split and scores the held-out
perturbations, writing ``<output-dir>/scouter/<dataset>/results.pkl``.

Every path comes from a CLI flag; nothing is hardcoded.  Each dataset is run in
its own ``multiprocessing.Process`` so that ``PEAK_CPU_RAM_GB`` (and the peak
GPU stats) are measured per dataset rather than across the whole sweep.

Example
-------
    python run_scouter.py \\
        --splits-dir   /path/to/benchmarks/dataset_splits \\
        --output-dir   /path/to/benchmarks/results \\
        --resources-dir /path/to/benchmarks/resources \\
        --models-dir   /path/to/benchmarks/models \\
        --dataset      ReplogleWeissman2022_rpe1 TianKampmann2019_iPSC

Omit ``--dataset`` to run every dataset found under ``--splits-dir``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench_common as bench

import multiprocessing as mp
import pickle
import random
import resource
import time

import numpy as np

MODEL = "scouter"

#: file in --resources-dir holding the GenePT gene/protein text embeddings
GENEPT_EMBEDDING_FILE = "GenePT_gene_protein_embedding_model_3_text.pickle"


def set_seeds(seed=24):
    import torch

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Normalize the condition name. Make "A+B" and "B+A" the same
def condition_sort(x):
    return '+'.join(sorted(x.split('+')))


def run_scouter(dataset_name, args):
    # Heavy third-party imports live here so that --help works without the
    # scouter conda environment.  They are done before the timer starts so the
    # reported data-loading time stays comparable across runs.
    import torch
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, r2_score

    bench.add_model_to_syspath(args.models_dir, "scouter")
    from scouter import Scouter, ScouterData

    paths = bench.split_paths(args.splits_dir, dataset_name)
    data_path = paths["adata"]
    embd_path = bench.resource_path(args.resources_dir, GENEPT_EMBEDDING_FILE)

    print(dataset_name)
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Set seeds for reproducibility
    set_seeds(args.seed)

    adata = ad.read_h5ad(str(data_path))

    ram_before_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    ram_after_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

    adata.obs.drop(columns=["condition"], errors="ignore", inplace=True)
    adata.obs.rename(columns={"perturbation": "condition"}, inplace=True)
    adata.obs["condition"] = adata.obs["condition"].apply(lambda x: "ctrl" if x == "control" else "ctrl+" + x)

    adata.obs['condition'] = adata.obs['condition'].astype(str).apply(lambda x: condition_sort(x)).astype('category')

    adata.uns = {}
    # adata.obs.drop('condition_name', axis=1, inplace=True)
    adata.var['gene_name'] = adata.var.index

    control_indices = np.load(paths["control"])
    training_indices = np.load(paths["train"])
    testing_indices = np.load(paths["test"])

    training_indices = np.concatenate([training_indices, control_indices])

    adata.obs['split'] = 'none'
    col = adata.obs.columns.get_loc('split')
    adata.obs.iloc[training_indices, col] = 'train'
    adata.obs.iloc[testing_indices, col] = 'test'

    with open(embd_path, 'rb') as f:
        embd = pd.DataFrame(pickle.load(f)).T
    ctrl_row = pd.DataFrame([np.zeros(embd.shape[1])], columns=embd.columns, index=['ctrl'])
    embd = pd.concat([ctrl_row, embd])

    scouterdata = ScouterData(adata=adata, embd=embd, key_label='condition', key_var_genename='gene_name')

    scouterdata.setup_ad('embd_index')
    scouterdata.gene_ranks()
    scouterdata.get_dropout_non_zero_genes()

    scouterdata.train_adata = scouterdata.adata[scouterdata.adata.obs['split'] == 'train'].copy()
    scouterdata.val_adata = scouterdata.adata[scouterdata.adata.obs['split'] == 'val'].copy()
    scouterdata.test_adata = scouterdata.adata[scouterdata.adata.obs['split'] == 'test'].copy()
    scouterdata.train_conds = sorted(scouterdata.train_adata.obs[scouterdata.key_label].unique().tolist())
    scouterdata.val_conds = sorted(scouterdata.val_adata.obs[scouterdata.key_label].unique().tolist())
    scouterdata.test_conds = sorted(scouterdata.test_adata.obs[scouterdata.key_label].unique().tolist())

    end_data_loading_time = time.time()
    data_loading_time = end_data_loading_time - start_time
    print("Data loading time: " + str(round(data_loading_time, 2)) + " s")

    scouter_model = Scouter(scouterdata, device=args.device)
    scouter_model.model_init()
    scouter_model.train(n_epochs=args.epochs)

    end_training_time = time.time()
    training_time = end_training_time - end_data_loading_time
    print("Training time: " + str(round(training_time, 2)) + " s")

    # Set to True to match Scouter's own paper convention (top 20 DEGs only, per the bioRxiv Methods section).
    # Set to False to match run_scLAMBDA.py / run_GenePert.py (score all genes).
    RESTRICT_TO_TOP20_DEGS = False
    degs_key = 'top20_degs_non_dropout'

    ctrl_mean_full = scouter_model.ctrl_adata.X.toarray().mean(axis=0)
    raw_ctrl_mean_full = scouter_model.ctrl_adata.layers['counts'].toarray().mean(axis=0)

    per_pert_corr = {}
    per_pert_spearman = {}
    per_pert_mse = {}
    per_pert_r2 = {}
    per_pert_l2 = {}
    per_pert_cosine = {}
    delta_y = {}
    delta_x = {}
    control = {}

    for condition in scouterdata.test_conds:
        if condition == 'ctrl':
            continue

        if RESTRICT_TO_TOP20_DEGS:
            degs = scouterdata.adata.uns[degs_key][condition]
            degs = np.setdiff1d(degs, np.where(np.isin(scouterdata.adata.var['gene_name'].values, condition.split('+'))))
        else:
            degs = np.arange(scouterdata.adata.shape[1])

        ctrl_mean = ctrl_mean_full[degs]
        pred_centered = scouter_model.pred([condition])[condition][:, degs].mean(axis=0) - ctrl_mean

        true_cells = scouterdata.test_adata[scouterdata.test_adata.obs['condition'] == condition].X.toarray()[:, degs]
        gt_centered = true_cells.mean(axis=0) - ctrl_mean

        per_pert_corr[condition] = np.corrcoef(gt_centered, pred_centered)[0, 1]
        per_pert_spearman[condition] = spearmanr(gt_centered, pred_centered).statistic
        per_pert_mse[condition] = mean_squared_error(gt_centered, pred_centered)
        per_pert_r2[condition] = r2_score(gt_centered, pred_centered)
        per_pert_l2[condition] = np.linalg.norm(gt_centered - pred_centered)
        per_pert_cosine[condition] = np.dot(gt_centered, pred_centered) / (
            np.linalg.norm(gt_centered) * np.linalg.norm(pred_centered))

        # predicted gene expression change over the control in the log space
        delta_y[condition] = pred_centered

        # ground truth gene expression change over the control in raw space
        raw_ctrl_mean = raw_ctrl_mean_full[degs]
        true_cells_raw = scouterdata.test_adata[scouterdata.test_adata.obs['condition'] == condition].layers['counts'].toarray()[:, degs]
        delta_x[condition] = true_cells_raw.mean(axis=0) - raw_ctrl_mean

        # mean control gene expression in raw space
        control[condition] = raw_ctrl_mean

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

    exported_results_folder = bench.results_dir(args.output_dir, MODEL, dataset_name)

    with open(exported_results_folder / "results.pkl", "wb") as file:
        pickle.dump(exported_results, file)

    print(f"mean Pearson: {mean_corr:.4f} | mean Spearman: {mean_spearman:.4f} | mean MSE: {mean_mse:.4f} | mean R²: {mean_r2:.4f} | mean L2: {mean_l2:.4f} | mean Cosine: {mean_cosine:.4f} over {len(per_pert_corr)} perturbations")


def main(argv=None):
    parser = bench.build_parser(
        "Run the Scouter perturbation-response benchmark.",
        resources=True,
        models=True,
        device=True,
        epochs=20,
        seed=24,
    )
    args = parser.parse_args(argv)

    datasets = bench.discover_datasets(args.splits_dir, args.dataset)

    failures = []
    for dataset_name in datasets:
        print(dataset_name)
        # Run in a subprocess so that peak RAM usage is localised to each dataset
        p = mp.Process(target=run_scouter, kwargs={"dataset_name": dataset_name, "args": args})
        p.start()
        p.join()
        if p.exitcode != 0:
            failures.append((dataset_name, p.exitcode))
            print(f"ERROR: {dataset_name} exited with code {p.exitcode}")
        print("\n_____________________________________________")

    if failures:
        print("Failed datasets: " + ", ".join(f"{d} (exit {c})" for d, c in failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
