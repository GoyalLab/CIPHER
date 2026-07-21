"""CIPHER benchmark driver — rank-1 covariance projection of the mean perturbation response.

For every dataset the control covariance ``Sigma`` is estimated on the training
control cells and each test perturbation's mean shift is predicted by projecting the
training shift onto ``Sigma[:, gene]`` (``cipher.forward_predict``).

Example
-------
    python run_CIPHER.py \
        --splits-dir  ../dataset_splits \
        --output-dir  ../results \
        --dataset     ReplogleWeissman2022_rpe1 NormanWeissman2019_filtered

Omit ``--dataset`` to run every dataset found under ``--splits-dir``.  Results are
written to ``<output-dir>/CIPHER/<dataset>/results.pkl``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench_common as bench

import multiprocessing as mp
import pickle
import resource
import time

import numpy as np
from scipy.sparse import issparse

# ``cipher`` (installed by the ``[benchmarks]`` extra) is imported inside
# ``run_cipher`` so that ``--help`` works even from an environment that lacks it.

MODEL = "CIPHER"

epsilon = 1e-8

#see how diff pruning techniques change the distribution
# Draw from training set multiple times at least 2 times, check the proportion that the alpha parameter (which is the fitting parameter) have the same sign. If they don't have the same sign for two separate draws from the training set
# If training set is 100 cells, draw 50 cells without replacement from those 100 cells. From the same set draw another 50, could have overlap between two sets
# Fit both alphas and see if they have the both sign. Keep only the perts that have the same sign

# Second method:# If the PCC between the two training set draws is below 0.85 (could change this value a bit), scrap the perts


def to_dense(x):
    return np.asarray(x.toarray(), dtype=np.float32) if issparse(x) else np.asarray(x, dtype=np.float32)


def compute_training_draw_proportionality_agreement_and_PCC(train_cells, sigma_col, ctrl_mean, n_trials=20, frac=0.5, rng=None):
    rng = rng or np.random.default_rng()
    n = train_cells.shape[0]
    draw_size = max(1, int(round(n * frac)))
    sign_agreement_array = np.empty(n_trials, dtype=bool)
    pcc_array = np.empty(n_trials, dtype=np.float32)

    for t in range(n_trials):
        idx1 = rng.choice(n, size=draw_size, replace=False)
        idx2 = rng.choice(n, size=draw_size, replace=False)

        delta1 = train_cells[idx1].mean(axis=0) - ctrl_mean
        delta2 = train_cells[idx2].mean(axis=0) - ctrl_mean

        u1 = np.dot(sigma_col, delta1) / (np.dot(sigma_col, sigma_col) + epsilon)
        u2 = np.dot(sigma_col, delta2) / (np.dot(sigma_col, sigma_col) + epsilon)

        sign_agreement_array[t] = np.sign(u1) == np.sign(u2)
        pcc_array[t] = np.corrcoef(delta1, delta2)[0, 1]

    return sign_agreement_array, pcc_array


def run_cipher(dataset_name, splits_dir, output_dir):
    import cipher

    start_time = time.time()

    # DATA LOADING =========================================================================
    adata_processed = bench.load_split_adata(splits_dir, dataset_name)

    exported_results_folder = bench.results_dir(output_dir, MODEL, dataset_name)

    end_data_loading_time = time.time()
    data_loading_time = end_data_loading_time - start_time
    print("Data loading time: " + str(round(data_loading_time, 2)) + " s")

    # MODEL TRAINING =========================================================================
    training_start_time = time.time()

    gene_names = np.array(adata_processed.var_names.tolist())

    train_ctrl_mask = (adata_processed.obs["condition"].values == "ctrl") & (adata_processed.obs["split"].values == "train")
    X0_train = to_dense(adata_processed[train_ctrl_mask].X)
    ctrl_mean = X0_train.mean(axis=0)

    ctrl_norm = ctrl_mean / ctrl_mean.sum() * 1e4
    ctrl_log = np.log1p(ctrl_norm)

    Sigma = cipher.compute_covariance(X0_train)

    training_time = time.time() - training_start_time

    pert_test = sorted(adata_processed.obs.loc[adata_processed.obs["split"] == "test", "condition"].unique().tolist())

    per_pert_corr = {}
    per_pert_spearman = {}
    per_pert_mse = {}
    per_pert_r2 = {}
    per_pert_l2 = {}
    per_pert_cosine = {}
    delta_y = {}
    delta_x = {}
    control = {}
    raw_prediction = {}
    training_draw_proportionality_agreements = {}
    training_draw_corr = {}
    skipped_perts = []
    eval_time = 0.0

    for i in pert_test:
        gene_hits = np.where(gene_names == i)[0]
        if len(gene_hits) == 0:
            skipped_perts.append(i)
            continue
        gene_idx = gene_hits[0]
        sigma_col = Sigma[:, gene_idx]

        train_mask = (adata_processed.obs["condition"].values == i) & (adata_processed.obs["split"].values == "train")
        train_cells = to_dense(adata_processed[train_mask].X)
        if train_cells.shape[0] == 0:
            skipped_perts.append(i)
            continue

        training_draw_proportionality_agreements[i], training_draw_corr[i] = compute_training_draw_proportionality_agreement_and_PCC(train_cells, sigma_col, ctrl_mean)
        fit_start_time = time.time()
        delta_train = train_cells.mean(axis=0) - ctrl_mean

        pred_centered, u_opt = cipher.forward_predict(Sigma, delta_train, gene_idx)
        training_time += time.time() - fit_start_time

        # EVALUATE ON TEST CELLS =========================================================================
        eval_start_time = time.time()
        test_mask = (adata_processed.obs["condition"].values == i) & (adata_processed.obs["split"].values == "test")
        test_cells = to_dense(adata_processed[test_mask].X)
        gt_mean_centered = test_cells.mean(axis=0) - ctrl_mean

        m = cipher.forward_metrics(gt_mean_centered, pred_centered)
        per_pert_corr[i] = m["pearson"]
        per_pert_spearman[i] = m["spearman"]
        per_pert_mse[i] = m["mse"]
        per_pert_r2[i] = m["r2_centered"]
        per_pert_cosine[i] = m["cosine"]
        per_pert_l2[i] = np.linalg.norm(gt_mean_centered - pred_centered)

        # predicted gene expression change over the control in the log space
        pred_absolute = np.clip(ctrl_mean + pred_centered, a_min=0, a_max=None)
        pred_norm = pred_absolute / pred_absolute.sum() * 1e4
        pred_log = np.log1p(pred_norm)
        delta_y[i] = pred_log - ctrl_log

        # ground truth gene expression change over the control in raw space
        delta_x[i] = gt_mean_centered

        # CIPHER's real prediction, natively in raw space (DELTA_Y is a lossy log
        # round-trip of this, kept only for schema compatibility with the other models)
        raw_prediction[i] = pred_centered
        eval_time += time.time() - eval_start_time

        # mean control gene expression in raw space
        control[i] = ctrl_mean

    print("Training time: " + str(round(training_time, 2)) + " s")
    print("Evaluation time: " + str(round(eval_time, 2)) + " s")

    if skipped_perts:
        print("Skipped " + str(len(skipped_perts)) + " perturbation(s) with no matching gene in the panel or no training cells: " + str(skipped_perts))

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
                        "RAW_PREDICTION": raw_prediction,
                        "DATA_LOADING_TIME": data_loading_time,
                        "TRAINING_TIME": training_time,
                        "EVALUATION_TIME": eval_time,
                        "TOTAL_TIME": total_time,
                        "PEAK_CPU_RAM_GB": peak_cpu_gb,
                        "TRAINING_DRAW_PROPORTIONALITY_AGREEMENT": training_draw_proportionality_agreements,
                        "TRAINING_DRAW_PROPORTIONALITY_CORRELATION": training_draw_corr}

    with open(str(exported_results_folder / "results.pkl"), "wb") as file:
        pickle.dump(exported_results, file)

    print(f"mean Pearson: {mean_corr:.4f} | mean Spearman: {mean_spearman:.4f} | mean MSE: {mean_mse:.4f} | mean R²: {mean_r2:.4f} | mean L2: {mean_l2:.4f} | mean Cosine: {mean_cosine:.4f} over {len(per_pert_corr)} perturbations")


def main(argv=None):
    parser = bench.build_parser(
        "Run the CIPHER rank-1 covariance benchmark on one or more dataset splits.")
    args = parser.parse_args(argv)

    datasets = bench.discover_datasets(args.splits_dir, args.dataset)

    for dataset_name in datasets:
        print(dataset_name)
        # Run in a subprocess so that peak RAM usage is localised to each dataset
        p = mp.Process(target=run_cipher,
                       kwargs={"dataset_name": dataset_name,
                               "splits_dir": args.splits_dir,
                               "output_dir": args.output_dir})
        p.start()
        p.join()
        print("\n_____________________________________________")
    return 0


if __name__ == "__main__":
    sys.exit(main())
