"""Benchmark driver for the ``linear_mean`` baseline ("trainMean").

The "model" predicts, for every test perturbation, the same delta: the mean of all
perturbed training cells minus the mean of the control training cells. No
preprocessing is applied (normalising/log1p made the baseline worse).

Results are written to ``<output-dir>/linear_mean/<dataset>/results.pkl``. Note the
MODEL folder is now the standardised ``linear_mean`` (this script previously wrote
to a folder literally named ``linear_mean_raw``); the output root is user-supplied,
so point ``--output-dir`` wherever the old results tree lived if you need them
side by side.

Example
-------
    python run_linear_mean.py \
        --splits-dir /path/to/benchmarks/dataset_splits \
        --output-dir /path/to/benchmarks/results \
        --dataset NormanWeissman2019_filtered ReplogleWeissman2022_rpe1

Omit ``--dataset`` to run every dataset found under ``--splits-dir``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench_common as bench  # noqa: E402

import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

MODEL = "linear_mean"

epsilon = 1e-8


def to_dense(x):
    from scipy.sparse import issparse

    return np.asarray(x.toarray(), dtype=np.float32) if issparse(x) else np.asarray(x, dtype=np.float32)


def run_linear_mean(dataset_name, splits_dir, output_dir):
    import scanpy as sc  # noqa: F401  (kept for the commented-out preprocessing below)
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, r2_score

    start_time = time.time()

    # DATA LOADING =========================================================================
    adata_processed = bench.load_split_adata(splits_dir, dataset_name)
    # The counts copy is made *between* the two RAM probes, exactly as in the original
    # script, so PEAK_CPU_RAM_GB stays comparable across models: the correction below
    # subtracts only the counts-layer allocation, never the h5ad load itself.
    # (load_split_adata only touches .obs after reading X, so copying here is numerically
    # identical to copying immediately after read_h5ad.)
    ram_before_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    adata_processed.layers["counts"] = adata_processed.X.copy()
    # Note: Linear mean no longer does preprocessing as that gives worse results.
    # sc.pp.normalize_total(adata_processed)
    # sc.pp.log1p(adata_processed)
    ram_after_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

    exported_results_folder = bench.results_dir(output_dir, MODEL, dataset_name)

    end_data_loading_time = time.time()
    data_loading_time = end_data_loading_time - start_time
    print("Data loading time: " + str(round(data_loading_time, 2)) + " s")

    # "MODEL" =========================================================================
    training_start_time = time.time()

    train_ctrl_mask = (adata_processed.obs["condition"].values == "ctrl") & (adata_processed.obs["split"].values == "train")
    ctrl_mean = to_dense(adata_processed[train_ctrl_mask].X).mean(axis=0)
    raw_ctrl_mean = to_dense(adata_processed[train_ctrl_mask].layers["counts"]).mean(axis=0)

    training_time = time.time() - training_start_time

    pert_test = sorted(adata_processed.obs.loc[adata_processed.obs["split"] == "test", "condition"].unique().tolist())
    pert_test = [p for p in pert_test if p != "ctrl"]

    per_pert_corr = {}
    per_pert_spearman = {}
    per_pert_mse = {}
    per_pert_r2 = {}
    per_pert_l2 = {}
    per_pert_cosine = {}
    delta_y = {}
    delta_x = {}
    control = {}
    skipped_perts = []
    eval_time = 0.0

    train_pert_mask = (adata_processed.obs["condition"].values != "ctrl") & (adata_processed.obs["split"].values == "train")
    train_cells = to_dense(adata_processed[train_pert_mask].X)

    # trainMean: mean of ALL perturbed training cells, so it gives the same prediction for every perturbation
    fit_start_time = time.time()
    pred_centered = train_cells.mean(axis=0) - ctrl_mean
    training_time += time.time() - fit_start_time

    for pert in pert_test:
        # EVALUATE ON TEST CELLS =========================================================================
        eval_start_time = time.time()
        test_mask = (adata_processed.obs["condition"].values == pert) & (adata_processed.obs["split"].values == "test")
        test_cells = to_dense(adata_processed[test_mask].X)
        if test_cells.shape[0] == 0:
            skipped_perts.append(pert)
            continue
        gt_centered = test_cells.mean(axis=0) - ctrl_mean

        with np.errstate(invalid="ignore", divide="ignore"):
            per_pert_corr[pert] = np.corrcoef(gt_centered, pred_centered)[0, 1]
            per_pert_spearman[pert] = spearmanr(gt_centered, pred_centered).statistic
            per_pert_cosine[pert] = np.dot(gt_centered, pred_centered) / (
                np.linalg.norm(gt_centered) * np.linalg.norm(pred_centered))
        per_pert_mse[pert] = mean_squared_error(gt_centered, pred_centered)
        per_pert_r2[pert] = r2_score(gt_centered, pred_centered)
        per_pert_l2[pert] = np.linalg.norm(gt_centered - pred_centered)

        # predicted gene expression change over the control in the log space
        delta_y[pert] = pred_centered

        # ground truth gene expression change over the control in raw space
        test_cells_raw = to_dense(adata_processed[test_mask].layers["counts"])
        delta_x[pert] = test_cells_raw.mean(axis=0) - raw_ctrl_mean
        eval_time += time.time() - eval_start_time

        # mean control gene expression in raw space
        control[pert] = raw_ctrl_mean

    print("Training time: " + str(round(training_time, 2)) + " s")
    print("Evaluation time: " + str(round(eval_time, 2)) + " s")

    if skipped_perts:
        print("Skipped " + str(len(skipped_perts)) + " perturbation(s) with no train/test cells: " + str(skipped_perts))

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

    # SAVE RESULTS =========================================================================

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
                        "TRAINING_TIME": training_time,
                        "EVALUATION_TIME": eval_time,
                        "TOTAL_TIME": total_time,
                        "PEAK_CPU_RAM_GB": peak_cpu_gb}

    with open(os.path.join(str(exported_results_folder), "results.pkl"), "wb") as file:
        pickle.dump(exported_results, file)

    print(f"mean Pearson: {mean_corr:.4f} | mean Spearman: {mean_spearman:.4f} | mean MSE: {mean_mse:.4f} | mean R²: {mean_r2:.4f} | mean L2: {mean_l2:.4f} | mean Cosine: {mean_cosine:.4f} over {len(per_pert_corr)} perturbations")


def main(argv=None):
    parser = bench.build_parser(
        "Run the linear_mean (trainMean) baseline over one or more benchmark datasets.")
    args = parser.parse_args(argv)

    dataset_names = bench.discover_datasets(args.splits_dir, args.dataset)

    for dataset_name in dataset_names:
        print(dataset_name)
        # Run in a subprocess so that peak RAM usage is localised to each dataset
        p = mp.Process(target=run_linear_mean,
                       kwargs={"dataset_name": dataset_name,
                               "splits_dir": args.splits_dir,
                               "output_dir": args.output_dir})
        p.start()
        p.join()
        print("\n_____________________________________________")
    return 0


if __name__ == "__main__":
    sys.exit(main())
