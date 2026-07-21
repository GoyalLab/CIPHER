"""GEARS benchmark driver for the CIPHER perturbation-prediction benchmark.

Trains one GEARS model per dataset and writes
``<output-dir>/GEARS/<dataset>/results.pkl`` (plus a ``loss_curve.png`` and a
``saved_data/`` scratch dir holding GEARS's own processed data / checkpoints).

Every path comes from a CLI flag; nothing is hardcoded. Example:

    python run_GEARS.py \
        --splits-dir  ../dataset_splits \
        --output-dir  ../results \
        --resources-dir ../resources \
        --models-dir  ../models \
        --dataset NormanWeissman2019_filtered TianKampmann2021_CRISPRa \
        --epochs 10 --device cuda

Omitting ``--dataset`` runs every complete dataset found under ``--splits-dir``.
Each dataset runs in its own subprocess so ``PEAK_CPU_RAM_GB`` stays per-dataset.
A dataset whose ``results.pkl`` already exists is skipped (the default, made
explicit with ``--skip-existing``); pass ``--overwrite`` to recompute it.

``--models-dir`` must contain a ``GEARS/`` checkout (the one holding the ``gears``
package). ``--resources-dir`` must contain ``gene_alias_to_symbol.pkl``; if
``essential_all_data_pert_genes.pkl`` is missing it is downloaded there on first
use (never at import time).
"""
# IMPORT LIBRARIES =========================================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench_common as bench  # noqa: E402

import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#: GEARS's own curated ~9,976-gene "essential gene" set (the same one used when
#: default_pert_graph=True), fetched on demand into --resources-dir.
ESSENTIAL_GENES_FILE = "essential_all_data_pert_genes.pkl"
ESSENTIAL_GENES_URL = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
#: Maps outdated/alias HGNC gene symbols (e.g. "KIAA0368") to the current approved
#: symbol (e.g. "ECPAS"). GEARS's perturbation graph is keyed by gene symbol, so a
#: dataset using an older symbol for a guide target would otherwise silently fail
#: to match and get dropped from training/scoring.
GENE_ALIAS_FILE = "gene_alias_to_symbol.pkl"

MODEL = "GEARS"

# HELPER FUNCTIONS =========================================================================


def load_gene_alias_map(resources_dir):
    """Load the alias -> current-HGNC-symbol map from --resources-dir."""
    with open(bench.resource_path(resources_dir, GENE_ALIAS_FILE), "rb") as f:
        return pickle.load(f)


def load_default_essential_genes(resources_dir):
    """Load GEARS's default essential-gene set, downloading it once if absent.

    Used as an extra candidate pool for GO-similarity neighbors on top of each
    dataset's own genes, so genes whose true GO neighbors aren't in a dataset's own
    (much smaller) panel can still find one, instead of ending up with zero
    neighbors and no biological signal.
    """
    from gears.utils import dataverse_download

    path = os.path.join(str(resources_dir), ESSENTIAL_GENES_FILE)
    if not os.path.exists(path):
        os.makedirs(str(resources_dir), exist_ok=True)
        logging.info("Essential-gene file missing, downloading to %s", path)
        dataverse_download(ESSENTIAL_GENES_URL, path)
    with open(bench.resource_path(resources_dir, ESSENTIAL_GENES_FILE), "rb") as f:
        return pickle.load(f)


def resolve_gene_symbols(condition_series, gene_alias_to_symbol):
    """Remap gene symbols in a 'condition' column (perturbation targets) to current
    HGNC symbols where an alias/previous-symbol mapping exists. 'control' is untouched."""
    # obs columns are often pandas Categorical; .apply() on a Categorical remaps the
    # category labels themselves, so the result can end up with a different .categories
    # list than the input, and Categorical `!=` refuses to compare across category sets.
    # Cast to plain strings first so both sides are ordinary object-dtype Series.
    condition_series = condition_series.astype(str)
    resolved = condition_series.apply(lambda name: gene_alias_to_symbol.get(name, name))
    changed = resolved != condition_series
    if changed.any():
        remapped_pairs = sorted(set(zip(condition_series[changed], resolved[changed])))
        print(f"Remapped {int(changed.sum())} cells' perturbation labels to current HGNC symbols: {remapped_pairs}")
    return resolved


def to_dense(x):
    from scipy import sparse

    return np.asarray(x.toarray(), dtype=np.float32) if sparse.issparse(x) else np.asarray(x, dtype=np.float32)


def plot_training_curve(model_save_folder, output_path, dataset_name):
    """Plot train/val loss and DE MSE vs. epoch from the epoch_*.npy arrays GEARS's
    own train() writes to model_save_folder, so we don't duplicate its loss tracking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_loss_path = os.path.join(model_save_folder, "epoch_losses.npy")
    val_loss_path = os.path.join(model_save_folder, "epoch_val_losses.npy")
    train_de_mse_path = os.path.join(model_save_folder, "epoch_de_mse.npy")
    val_de_mse_path = os.path.join(model_save_folder, "epoch_val_de_mse.npy")
    if not (os.path.exists(train_loss_path) and os.path.exists(val_loss_path)):
        logging.info("No epoch loss files found, skipping loss curve plot")
        return

    train_loss = np.load(train_loss_path)
    val_loss = np.load(val_loss_path)
    # Irish: across checkpoint-resume attempts the train/val loss arrays can end up
    # a step out of sync (e.g. a resume re-appends one array but not the other),
    # which crashes matplotlib on a shape mismatch and takes the whole run down
    # with it -- including the scoring step that hasn't even happened yet. Just
    # plot however many epochs both arrays agree on.
    n_loss = min(len(train_loss), len(val_loss))
    train_loss, val_loss = train_loss[:n_loss], val_loss[:n_loss]
    epochs = np.arange(1, n_loss + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, train_loss, label="Train loss", marker="o")
    ax1.plot(epochs, val_loss, label="Validation loss", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Overall MSE")
    ax1.set_title("Overall MSE")
    ax1.legend()

    if os.path.exists(train_de_mse_path) and os.path.exists(val_de_mse_path):
        train_de_mse = np.load(train_de_mse_path)
        val_de_mse = np.load(val_de_mse_path)
        n_de = min(len(train_de_mse), len(val_de_mse), n_loss)
        de_epochs = np.arange(1, n_de + 1)
        train_de_mse, val_de_mse = train_de_mse[:n_de], val_de_mse[:n_de]
        ax2.plot(de_epochs, train_de_mse, label="Train DE MSE", marker="o")
        ax2.plot(de_epochs, val_de_mse, label="Validation DE MSE", marker="o")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Top 20 DE MSE")
        ax2.set_title("Top 20 DE MSE")
        ax2.legend()
    else:
        ax2.axis("off")

    fig.suptitle(f"GEARS training curves: {dataset_name}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logging.info(f"Saved loss curve to {output_path}")


def pearson_matrix(A, B):
    """Pearson r of every row in A against every row in B -> (n_A, n_B)."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    An = np.linalg.norm(A, axis=1, keepdims=True); An[An == 0] = 1.0
    Bn = np.linalg.norm(B, axis=1, keepdims=True); Bn[Bn == 0] = 1.0
    return (A / An) @ (B / Bn).T


def predict_with_controls(model, ctrl_X, perturbation_names, batch_size=128, n_samples=300):
    """Run model on a random sample of ctrl_X for each perturbation."""
    import torch
    from gears.utils import create_cell_graph_for_prediction
    from torch_geometric.data import DataLoader as GeoDataLoader

    rng = np.random.default_rng(44)
    n = min(n_samples, ctrl_X.shape[0])
    ctrl_X = ctrl_X[rng.choice(ctrl_X.shape[0], size=n, replace=False)]

    predicted_means = []
    model.best_model.eval()  # This just switches the model to inference mode in PyTorch
    for pert in perturbation_names:
        pert_idx = [model.node_map_pert[pert]]
        graphs = [
            create_cell_graph_for_prediction(ctrl_X[i].reshape(1, -1), pert_idx, [pert])
            for i in range(ctrl_X.shape[0])
        ]
        loader = GeoDataLoader(graphs, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch.to(model.device)
                preds.append(model.best_model(batch).detach().cpu().numpy())
        predicted_means.append(np.concatenate(preds).mean(axis=0))
        logging.info("Predicted " + pert)
    return np.array(predicted_means, dtype=np.float32)


def run_gears(dataset_name, args):
    # The vendored GEARS checkout has to be importable before anything touches `gears`.
    bench.add_model_to_syspath(args.models_dir, MODEL)

    import scanpy as sc
    import torch
    from scipy import sparse
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, r2_score
    from gears import PertData, GEARS

    logging.info("Imported libraries")

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    exported_results_folder = str(bench.results_dir(args.output_dir, MODEL, dataset_name))
    results_pkl = os.path.join(exported_results_folder, "results.pkl")

    if os.path.exists(results_pkl) and not args.overwrite:
        print(f"results.pkl already exists for {dataset_name}, skipping.")
        return

    data_folder_GEARS = os.path.join(exported_results_folder, "saved_data")

    # DATA LOADING =========================================================================

    paths = bench.split_paths(args.splits_dir, dataset_name)

    # Load the full set
    full_set = sc.read_h5ad(str(paths["adata"]))
    logging.info("Loaded the full gene set")

    control_indices = np.load(paths["control"])
    training_indices = np.load(paths["train"])
    testing_indices = np.load(paths["test"])

    control_set = full_set[control_indices]

    logging.info("Collected indices")

    # DATA PREPROCESSING AND COMPATIBILITY =========================================================================

    full_set.obs.drop(columns=["condition"], errors="ignore", inplace=True)
    full_set.obs.rename(columns={"perturbation": "condition"}, inplace=True)
    # full_set.obs["condition"] = resolve_gene_symbols(
    #     full_set.obs["condition"], load_gene_alias_map(args.resources_dir))
    full_set.obs.rename(columns={"celltype": "cell_type"}, inplace=True)

    # GEARS Requires cell_type internally. As all datasets for the CIPHER benchmark are from a
    # single cell line, we can just set the dataset name as the cell type
    if "cell_type" not in full_set.obs.columns:
        full_set.obs["cell_type"] = dataset_name
    else:
        full_set.obs.rename(columns={"celltype": "cell_type"}, inplace=True)

    perturbed_obs_names = full_set.obs_names
    train_obs = perturbed_obs_names[training_indices]
    id_test_obs = perturbed_obs_names[testing_indices]

    # Restrict to the anndata to the cells in the control, train and test set. Other cells are unnecessary overhead
    used_positions = np.unique(np.concatenate([control_indices, training_indices, testing_indices]))
    full_set = full_set[used_positions].copy()
    # A few source h5ad files (e.g. the Replogle datasets) store X as a dense ndarray
    # rather than sparse. GEARS's get_dropout_non_zero_genes() unconditionally calls
    # adata.X.toarray(), which only exists on sparse matrices, so normalize here.
    if not sparse.issparse(full_set.X):
        full_set.X = sparse.csr_matrix(full_set.X)
    # gene_means = full_set.X.mean(axis=0).A1 if sparse.issparse(full_set.X) else np.asarray(full_set.X.mean(axis=0)).ravel()
    # expressed_genes = set(full_set.var_names[gene_means >= 1.0])
    # target_genes = set(full_set.obs["condition"].unique()) - {"control"}
    # genes_to_keep = expressed_genes | (target_genes & set(full_set.var_names))
    # full_set = full_set[:, full_set.var_names.isin(genes_to_keep)].copy()
    # logging.info(f"Filtered to {full_set.shape[1]} genes (expression_threshold=1.0 + perturbation targets)")

    full_set.obs["condition"] = full_set.obs["condition"].apply(lambda x: "ctrl" if x == "control" else "ctrl+" + x)
    print("Control cell types:", full_set.obs.loc[full_set.obs['condition'] == 'ctrl', 'cell_type'].unique())
    print("Perturbed cell types:", full_set.obs.loc[full_set.obs['condition'] != 'ctrl', 'cell_type'].unique())
    full_set.var["gene_name"] = full_set.var.index
    logging.info("Reassigned annotated data columns to be compatible with GEARS")

    # All 27 datasets are raw counts. Keep a raw-counts layer alongside the normalized X,
    # matching run_CIPHER.py/run_scLAMBDA.py/run_scouter.py/run_GenePert.py/run_scGPT.py,
    # so the final scoring step can report DELTA_X/CONTROL in raw space like the rest of
    # the benchmark.
    ram_before_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    full_set.layers['counts'] = full_set.X.copy()
    sc.pp.normalize_total(full_set)
    sc.pp.log1p(full_set)
    ram_after_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    logging.info(f"Normalised the dataset")

    end_data_loading_time = time.time()
    data_loading_time = end_data_loading_time - start_time
    print("Data loading time: " + str(round(data_loading_time, 2)) + " s")

    # GEARS SETUP =========================================================================

    start_setup_time = time.time()

    DEFAULT_ESSENTIAL_GENES = load_default_essential_genes(args.resources_dir)

    # Candidate pool for GO-similarity neighbors: this dataset's own measured genes and
    # perturbation targets (guarantees every gene we need is still included, same as
    # default_pert_graph=False alone) unioned with GEARS's default essential-gene set
    # (gives genes with no same-panel GO neighbor a much larger pool to find one in).
    panel_genes = set(full_set.var["gene_name"].values)
    pert_genes = {c.replace("ctrl+", "") for c in full_set.obs["condition"].unique() if c != "ctrl"}
    gene_set = sorted(panel_genes | pert_genes | set(DEFAULT_ESSENTIAL_GENES))
    os.makedirs(data_folder_GEARS, exist_ok=True)
    gene_set_path = os.path.join(data_folder_GEARS, "gene_set_union.pkl")
    with open(gene_set_path, "wb") as f:
        pickle.dump(gene_set, f)

    pert_data = PertData(data_folder_GEARS, gene_set_path=gene_set_path)
    pert_data.new_data_process(dataset_name=dataset_name, adata=full_set, save_anndata=False)
    pert_data.load(data_path=os.path.join(data_folder_GEARS, dataset_name.lower()), adata=full_set)

    train_conditions = list(set(full_set[train_obs].obs["condition"].tolist()))
    id_test_conditions = list(set(full_set[id_test_obs].obs["condition"].tolist()))

    split_dict = {
        'train': train_conditions,
        'val': train_conditions,  # Reusing the train set for the validation as there is no validation set. Risk of overfitting!
        'test': id_test_conditions,
    }
    print("Train perturbations:", len(split_dict['train']))
    print("Val (ID) perturbations:", len(split_dict['val']))
    print("Test (ID) perturbations:", len(split_dict['test']))

    split_path = os.path.join(data_folder_GEARS, dataset_name.lower(), "split.pkl")
    with open(split_path, 'wb') as f:
        pickle.dump(split_dict, f)

    pert_data.prepare_split(split='custom', split_dict_path=split_path)
    dataloaders = pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    logging.info("Dataloaders ready")

    end_setup_time = time.time()
    setup_time = end_setup_time - start_setup_time
    print("Setup time: " + str(round(setup_time, 2)) + " s")

    # GEARS TRAINING =========================================================================

    start_training_time = time.time()

    # Train the model
    gears_model = GEARS(pert_data, device=args.device, model_save_folder=data_folder_GEARS,
                        weight_bias_track=False, proj_name='pertnet', exp_name='pertnet')
    gears_model.model_initialize(hidden_size=64)

    missing = [c for c in train_conditions if c not in gears_model.dict_filter and c != 'ctrl']
    print(f"{len(missing)} training perturbations missing from dict_filter: {missing}")

    # force_retrain=False: resume from the per-epoch checkpoint in data_folder_GEARS if
    # one exists (see gears.py's train()), instead of always retraining from scratch --
    # matters now that runs can take longer than the 4hr SLURM time limit.
    gears_model.train(epochs=args.epochs, lr=1e-3, n_eval_cells = 500, force_retrain=False)
    logging.info("Finished training.")

    plot_training_curve(data_folder_GEARS,
                        os.path.join(exported_results_folder, "loss_curve.png"),
                        dataset_name)

    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    print("Training time: " + str(round(training_time, 2)) + " s")

    # SCORING PERFORMANCE =========================================================================

    start_evaluation_time = time.time()

    # Scoring
    train_pert_genes = {c.replace('ctrl+', '') for c in train_conditions if c != 'ctrl'}
    perturbation_names = [p for p in gears_model.pert_list if p in train_pert_genes and p in gears_model.node_map_pert]
    pert_name_to_idx = {p: i for i, p in enumerate(perturbation_names)}

    ctrl = pert_data.adata.obs['condition'] == 'ctrl'

    logging.info("Started scoring.")

    for split_name, test_obs, ctrl_obs in [
        ("id_test",  id_test_obs,  ctrl),
    ]:
        ctrl_X = to_dense(pert_data.adata[ctrl_obs].X)
        mean_expression = ctrl_X.mean(axis=0)

        # mean control gene expression in raw space
        raw_ctrl_mean = to_dense(pert_data.adata[ctrl_obs].layers['counts']).mean(axis=0)

        # Use context-specific controls for predictions
        predicted_means = predict_with_controls(gears_model, ctrl_X, perturbation_names)
        P_centered = predicted_means - mean_expression

        test_adata = pert_data.adata[pert_data.adata.obs_names.isin(test_obs)]
        GT_cells = to_dense(test_adata.X)
        GT_centered = GT_cells - mean_expression
        GT_cells_raw = to_dense(test_adata.layers['counts'])

        cell_perturbations = test_adata.obs["condition"].str.replace(r"^ctrl\+", "", regex=True).values

        # This is needlessly retained from ExPert OOD benchmarking, but its a nice sanity check
        valid_local_idx = [i for i, p in enumerate(cell_perturbations) if p in pert_name_to_idx]
        valid_cell_perts = cell_perturbations[valid_local_idx]

        corr_by_pert = defaultdict(list)
        spearman_by_pert = defaultdict(list)
        mse_by_pert = defaultdict(list)
        r2_by_pert = defaultdict(list)
        l2_by_pert = defaultdict(list)
        cosine_by_pert = defaultdict(list)
        raw_by_pert = defaultdict(list)
        for i, pert in enumerate(valid_cell_perts):
            gt, pred = GT_centered[valid_local_idx[i]], P_centered[pert_name_to_idx[pert]]
            corr_by_pert[pert].append(np.corrcoef(gt, pred)[0, 1])
            spearman_by_pert[pert].append(spearmanr(gt, pred).statistic)
            mse_by_pert[pert].append(mean_squared_error(gt, pred))
            r2_by_pert[pert].append(r2_score(gt, pred))
            l2_by_pert[pert].append(np.linalg.norm(gt - pred))
            cosine_by_pert[pert].append(
                np.dot(gt, pred) / (np.linalg.norm(gt) * np.linalg.norm(pred)))
            raw_by_pert[pert].append(GT_cells_raw[valid_local_idx[i]])

        per_pert_corr     = {p: np.nanmean(v) for p, v in corr_by_pert.items()}
        per_pert_spearman = {p: np.nanmean(v) for p, v in spearman_by_pert.items()}
        per_pert_mse      = {p: np.nanmean(v) for p, v in mse_by_pert.items()}
        per_pert_r2       = {p: np.nanmean(v) for p, v in r2_by_pert.items()}
        per_pert_l2       = {p: np.nanmean(v) for p, v in l2_by_pert.items()}
        per_pert_cosine   = {p: np.nanmean(v) for p, v in cosine_by_pert.items()}

        # predicted gene expression change over the control in the log space
        delta_y = {p: P_centered[pert_name_to_idx[p]] for p in per_pert_corr}

        # mean control gene expression in raw space
        control = {p: raw_ctrl_mean for p in per_pert_corr}

        # ground truth gene expression change over the control in raw space
        delta_x = {p: np.mean(raw_by_pert[p], axis=0) - raw_ctrl_mean for p in per_pert_corr}

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

        end_time = time.time()
        evaluation_time = end_time - start_evaluation_time
        total_time = end_time - start_time
        print("Evaluation time: " + str(round(evaluation_time, 2)) + " s")
        print("Total Elapsed time: " + str(round(total_time, 2)) + " s")

        # Track resource usage
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
                            "SETUP_TIME": setup_time,
                            "TRAINING_TIME": training_time,
                            "EVALUATION_TIME": evaluation_time,
                            "TOTAL_TIME": total_time,
                            "PEAK_CPU_RAM_GB": peak_cpu_gb,
                            "PEAK_GPU_ALLOCATED_GB": peak_gpu_allocated_gb,
                            "PEAK_GPU_RESERVED_GB": peak_gpu_reserved_gb}

        # Save the results
        with open(results_pkl, "wb") as file:
            pickle.dump(exported_results, file)

        # Remove the checkpoints and the pygraph as they will take too much space otherwise
        # NOTE THIS REMOVES FILES SO PLEASE BE CAREFUL!!!
        # import glob, shutil
        # for saved_file in glob.glob(os.path.join(data_folder_GEARS, "*_checkpoint.pt")):
        #     os.remove(saved_file)
        # shutil.rmtree(os.path.join(data_folder_GEARS, dataset_name.lower()))
        logging.info("Deleted checkpoints and pygraph")

        print(f"  [{split_name}] mean Pearson: {mean_corr:.4f} | mean Spearman: {mean_spearman:.4f} | mean MSE: {mean_mse:.4f} | mean R²: {mean_r2:.4f} | mean L2: {mean_l2:.4f} | mean Cosine: {mean_cosine:.4f} over {len(per_pert_corr)} perturbations")
    logging.info("Scoring complete.")


def main(argv=None):
    parser = bench.build_parser(
        "Train and score GEARS on one or more CIPHER benchmark datasets.",
        resources=True, models=True, device=True, epochs=10, skip_existing=True,
    )
    args = parser.parse_args(argv)

    datasets = bench.discover_datasets(args.splits_dir, args.dataset)
    print(f"Running {MODEL} on {len(datasets)} dataset(s): {', '.join(datasets)}")

    failed = []
    for dataset_name in datasets:
        print(dataset_name)
        # Run in a subprocess so that peak RAM (and GPU) usage is localised to each dataset
        p = mp.Process(target=run_gears, kwargs={"dataset_name": dataset_name, "args": args})
        p.start()
        p.join()
        if p.exitcode != 0:
            print(f"{dataset_name} exited with code {p.exitcode}")
            failed.append(dataset_name)
        print("\n_____________________________________________")

    if failed:
        print(f"{len(failed)} dataset(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
