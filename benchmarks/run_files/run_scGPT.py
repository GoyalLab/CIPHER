"""scGPT perturbation-response benchmark driver for CIPHER.

Fine-tunes the pretrained scGPT_human checkpoint on each dataset's train split
(using the GEARS ``PertData`` backend for graph construction / dataloading) and
scores held-out perturbations, writing ``<output-dir>/scGPT/<dataset>/results.pkl``.

Every path comes from a CLI flag; nothing is hardcoded. Example:

    python run_scGPT.py \
        --splits-dir  ../dataset_splits \
        --output-dir  ../results \
        --models-dir  ../models \
        --dataset     NormanWeissman2019_filtered ReplogleWeissman2022_rpe1 \
        --device cuda --epochs 15 --seed 44 --batch-size 64

Omit ``--dataset`` to run every dataset found under ``--splits-dir``. Each dataset
runs in its own ``multiprocessing.Process`` so that the reported PEAK_CPU_RAM_GB /
PEAK_GPU_* figures stay per-dataset. As before, a dataset whose ``results.pkl``
already exists is skipped (``--skip-existing`` makes that explicit, ``--overwrite``
forces a recompute).

``--pretrained-dir`` defaults to ``<models-dir>/scGPT/pretrained_models/scGPT_human``.
"""
import json
import os
import shutil
import sys
import time
import resource
import copy
import gc
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional
import warnings
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bench_common as bench

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

warnings.filterwarnings("ignore")

MODEL = "scGPT"

# settings for data processing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 1536

# settings for training
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
amp = True
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
lr = 1e-4
eval_batch_size = 64
schedule_interval = 1
early_stop = 10

# settings for the model
embsize = 512
d_hid = 512
nlayers = 12
nhead = 8
n_layers_cls = 3
dropout = 0
use_fast_transformer = True

log_interval = 100

DEFAULT_EPOCHS = 15
DEFAULT_SEED = 44
DEFAULT_BATCH_SIZE = 64


def resolve_pretrained_dir(args) -> Path:
    """``--pretrained-dir``, or the scGPT_human checkpoint shipped under ``--models-dir``."""
    if args.pretrained_dir is not None:
        p = Path(args.pretrained_dir)
    else:
        p = Path(args.models_dir) / MODEL / "pretrained_models" / "scGPT_human"
    if not p.is_dir():
        raise FileNotFoundError(
            f"Pretrained scGPT checkpoint directory not found: {p}\n"
            "Pass --pretrained-dir, or place the scGPT_human checkpoint at "
            "<models-dir>/scGPT/pretrained_models/scGPT_human."
        )
    return p


def run_scgpt(args, dataset_name: str) -> None:
    print(dataset_name)

    save_dir = bench.results_dir(args.output_dir, MODEL, dataset_name)
    print(f"saving to {save_dir}")

    results_path = save_dir / "results.pkl"
    if results_path.exists() and not args.overwrite:
        print(f"results.pkl already exists for {dataset_name}, skipping.")
        return

    split_files = bench.split_paths(args.splits_dir, dataset_name)
    pretrained_dir = resolve_pretrained_dir(args)

    # Heavy third-party imports live here so that --help works without the scGPT env.
    import scanpy as sc
    import torch
    import numpy as np
    import scipy.sparse as sp
    import pickle
    from torch import nn
    from torch_geometric.loader import DataLoader
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, r2_score

    # scGPT uses the GEARS backend
    bench.add_model_to_syspath(args.models_dir, "GEARS")
    from gears import PertData
    from gears.utils import create_cell_graph_dataset_for_prediction

    bench.add_model_to_syspath(args.models_dir, MODEL)

    import scgpt as scg
    from scgpt.model import TransformerGenerator
    from scgpt.loss import masked_mse_loss
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics

    set_seed(args.seed)

    batch_size = args.batch_size
    epochs = args.epochs

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA unavailable; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Written after every epoch and removed on successful completion (see CLEANUP).
    # If a job times out mid-training (8hr SLURM limit), re-running the same dataset
    # picks this up instead of retraining from the pretrained weights at epoch 1.
    checkpoint_path = save_dir / "checkpoint.pt"

    logger = scg.logger
    scg.utils.add_file_handler(logger, str(save_dir / "run.log"))
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # DATA LOADING =========================================================================
    full_set = sc.read_h5ad(str(split_files["adata"]))
    logging.info("Loaded the full gene set")

    # A few datasets store X as a dense ndarray. GEARS' get_dropout_non_zero_genes()
    # unconditionally calls .toarray() on it, so force it to sparse here.
    if not sp.issparse(full_set.X):
        full_set.X = sp.csr_matrix(full_set.X)

    data_folder = save_dir / "saved_data"

    control_indices = np.load(split_files["control"])
    training_indices = np.load(split_files["train"])
    testing_indices = np.load(split_files["test"])

    full_set.obs.drop(columns=["condition"], errors="ignore", inplace=True)
    full_set.obs.rename(columns={"perturbation": "condition"}, inplace=True)
    full_set.obs.rename(columns={"celltype": "cell_type"}, inplace=True)

    # GEARS requires cell_type internally. As all datasets for the CIPHER benchmark are from a
    # single cell line, we can just set the dataset name as the cell type.
    if "cell_type" not in full_set.obs.columns:
        full_set.obs["cell_type"] = dataset_name.replace("_", "-")
    else:
        full_set.obs.rename(columns={"celltype": "cell_type"}, inplace=True)

    perturbed_obs_names = full_set.obs_names
    train_obs = perturbed_obs_names[training_indices]
    id_test_obs = perturbed_obs_names[testing_indices]

    # Restrict the anndata to the cells in the control, train and test set. Other cells are unnecessary overhead
    used_positions = np.unique(np.concatenate([control_indices, training_indices, testing_indices]))
    full_set = full_set[used_positions].copy()

    full_set.obs["condition"] = full_set.obs["condition"].apply(lambda x: "ctrl" if x == "control" else "ctrl+" + x)
    print("Control cell types:", full_set.obs.loc[full_set.obs['condition'] == 'ctrl', 'cell_type'].unique())
    print("Perturbed cell types:", full_set.obs.loc[full_set.obs['condition'] != 'ctrl', 'cell_type'].unique())
    full_set.var["gene_name"] = full_set.var.index
    logging.info("Reassigned annotated data columns to be compatible with GEARS")

    ram_before_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    full_set.layers['counts'] = full_set.X.copy()
    sc.pp.normalize_total(full_set)
    sc.pp.log1p(full_set)
    ram_after_counts_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    logging.info("Normalised the dataset")

    pert_data = PertData(str(data_folder))
    # scGPT never uses PertData's GO-graph perturbation embeddings (only GEARS' own GNN
    # does), so the default curated "essential genes" list -- built from GEARS' CRISPRi
    # essentiality-screen training data, not from this dataset -- has no benefit here and
    # only drops perturbations whose target isn't on that list. default_pert_graph=False
    # switches to (this panel's genes + all perturbation targets) intersected with GO
    # annotations, which covers all of them.
    pert_data.default_pert_graph = False
    pert_data.new_data_process(dataset_name="pert_data", adata=full_set, save_anndata=False)
    pert_data.load(data_path=str(data_folder / "pert_data"), adata=full_set)

    train_conditions = list(set(full_set[train_obs].obs["condition"].tolist()))
    id_test_conditions = list(set(full_set[id_test_obs].obs["condition"].tolist()))

    split_dict = {
        'train': train_conditions,
        'val': train_conditions,  # Reusing the train set for validation as there is no validation set. Risk of overfitting!
        'test': id_test_conditions,
    }
    print("Train perturbations:", len(split_dict['train']))
    print("Val (ID) perturbations:", len(split_dict['val']))
    print("Test (ID) perturbations:", len(split_dict['test']))

    split_path = str(data_folder / "pert_data" / "split.pkl")
    with open(split_path, 'wb') as f:
        pickle.dump(split_dict, f)

    pert_data.prepare_split(split='custom', split_dict_path=split_path)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

    end_data_loading_time = time.time()
    data_loading_time = end_data_loading_time - start_time
    print("Data loading time: " + str(round(data_loading_time, 2)) + " s")

    # VOCAB / MODEL SETUP =========================================================================
    model_dir = Path(pretrained_dir)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    genes = pert_data.adata.var["gene_name"].tolist()

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    model_embsize = model_configs["embsize"]
    model_nhead = model_configs["nheads"]
    model_d_hid = model_configs["d_hid"]
    model_nlayers = model_configs["nlayers"]
    model_n_layers_cls = model_configs["n_layers_cls"]

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)

    ntokens = len(vocab)
    model = TransformerGenerator(
        ntokens,
        model_embsize,
        model_nhead,
        model_d_hid,
        model_nlayers,
        nlayers_cls=model_n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        use_fast_transformer=use_fast_transformer,
    )
    resume_checkpoint = None
    if checkpoint_path.exists():
        # weights_only=False: this is our own checkpoint (not a third-party download), and it
        # holds plain numpy scalars (best_val_corr) alongside tensors, which weights_only=True
        # (torch>=2.6 default) refuses to unpickle.
        resume_checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        model.to(device)
        logger.info(
            f"Resuming from checkpoint at epoch {resume_checkpoint['epoch']} "
            f"(best_val_corr so far: {resume_checkpoint['best_val_corr']:.4f})"
        )
    else:
        pretrained_dict = torch.load(model_file)
        # scg.utils.load_pretrained rewrites the checkpoint's flash-attn-1.x-style
        # "self_attn.Wqkv.*" keys to this repo's FlashMHA wrapper layout
        # ("self_attn._impl.Wqkv.*") and drops any still-unmatched/shape-mismatched
        # keys instead of raising, unlike a raw model.load_state_dict().
        model = scg.utils.load_pretrained(
            model, pretrained_dict, strict=False, prefix=load_param_prefixs
        )
        model.to(device)

    end_setup_time = time.time()
    setup_time = end_setup_time - end_data_loading_time
    print("Setup time: " + str(round(setup_time, 2)) + " s")

    # TRAIN / EVAL FUNCTIONS =========================================================================
    criterion = masked_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])

    def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, epoch: int) -> None:
        model.train()
        total_loss, total_mse = 0.0, 0.0
        start_time = time.time()

        num_batches = len(train_loader)
        for batch, batch_data in enumerate(train_loader):
            cur_batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(cur_batch_size, n_genes)
            pert_flags = x[:, 1].long().view(cur_batch_size, n_genes)
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[:max_seq_len]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(cur_batch_size, 1)

            src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                )
                output_values = output_dict["mlm_output"]
                masked_positions = torch.ones_like(input_values, dtype=torch.bool)
                loss = loss_mse = criterion(output_values, target_values, masked_positions)

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            if batch % log_interval == 0 and batch > 0:
                cur_lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {cur_lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
                )
                total_loss = 0
                total_mse = 0
                start_time = time.time()

                # Irish: the once-per-epoch empty_cache()/gc.collect() (after the
                # training loop returns) only runs after ALL of an epoch's batches
                # are done -- for the largest datasets (~7000 batches/epoch) that's
                # a lot of accumulation before it ever fires, and OOMs consistently
                # hit in the last few % of an epoch. Clean up at the same cadence
                # as progress logging so it can't build up that far.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    def eval_perturb(loader: DataLoader, model: TransformerGenerator, device: torch.device) -> Dict:
        model.eval()
        model.to(device)
        pert_cat = []
        pred = []
        truth = []
        pred_de = []
        truth_de = []
        results = {}

        for batch in loader:
            batch.to(device)
            pert_cat.extend(batch.pert)

            with torch.no_grad():
                p = model.pred_perturb(batch, include_zero_gene=include_zero_gene, gene_ids=gene_ids)
                t = batch.y
                pred.extend(p.cpu())
                truth.extend(t.cpu())

                for itr, de_idx in enumerate(batch.de_idx):
                    pred_de.append(p[itr, de_idx])
                    truth_de.append(t[itr, de_idx])

        results["pert_cat"] = np.array(pert_cat)
        pred = torch.stack(pred)
        truth = torch.stack(truth)
        results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
        results["truth"] = truth.detach().cpu().numpy().astype(np.float64)

        pred_de = torch.stack(pred_de)
        truth_de = torch.stack(truth_de)
        results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float64)
        results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float64)

        return results

    def predict(model: TransformerGenerator, pert_list: List[List[str]], pool_size: Optional[int] = None) -> Dict:
        adata = pert_data.adata
        ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
        if pool_size is None:
            pool_size = len(ctrl_adata.obs)
        gene_list = pert_data.gene_names.values.tolist()
        for pert in pert_list:
            for i in pert:
                if i not in gene_list:
                    raise ValueError(
                        "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                    )

        model.eval()
        model_device = next(model.parameters()).device
        with torch.no_grad():
            results_pred = {}
            for pert in pert_list:
                cell_graphs = create_cell_graph_dataset_for_prediction(
                    pert, ctrl_adata, gene_list, model_device, num_samples=pool_size
                )
                loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
                preds = []
                for batch_data in loader:
                    pred_gene_values = model.pred_perturb(
                        batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                    )
                    preds.append(pred_gene_values)
                preds = torch.cat(preds, dim=0)
                results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

        return results_pred

    # TRAINING LOOP =========================================================================
    start_epoch = 1
    best_val_corr = 0
    best_model = None
    patience = 0

    if resume_checkpoint is not None:
        start_epoch = resume_checkpoint["epoch"] + 1
        best_val_corr = resume_checkpoint["best_val_corr"]
        patience = resume_checkpoint["patience"]
        if resume_checkpoint["best_model_state_dict"] is not None:
            best_model = copy.deepcopy(model)
            best_model.load_state_dict(resume_checkpoint["best_model_state_dict"])
            best_model.to(device)

    if start_epoch > epochs:
        # Checkpoint already finished all training epochs (job must have timed out
        # during scoring instead) -- nothing left to train, go straight to scoring.
        logger.info(
            f"Checkpoint is already at the final epoch ({resume_checkpoint['epoch']}/{epochs}); "
            "skipping training and resuming at scoring."
        )

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        train(model, train_loader, epoch)

        val_res = eval_perturb(valid_loader, model, device)
        val_metrics = compute_perturbation_metrics(
            val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        )
        logger.info(f"val_metrics at epoch {epoch}: ")
        logger.info(val_metrics)

        elapsed = time.time() - epoch_start_time
        logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

        # Diagnostic: track whether memory grows epoch over epoch (suspected CUDA
        # allocator fragmentation from repeated best_model deepcopies + variably
        # shaped shuffled batches, since empty_cache()/gc.collect() were never
        # called anywhere in this loop). ru_maxrss is a running peak, so a growing
        # value here across epochs indicates real (not just per-epoch) growth.
        epoch_peak_cpu_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        if torch.cuda.is_available():
            gpu_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            gpu_peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            logger.info(
                f"| epoch {epoch:3d} memory | peak CPU RSS: {epoch_peak_cpu_gb:6.2f} GB | "
                f"GPU allocated: {gpu_allocated_gb:6.2f} GB | GPU reserved: {gpu_reserved_gb:6.2f} GB | "
                f"GPU peak reserved: {gpu_peak_reserved_gb:6.2f} GB |"
            )
        else:
            logger.info(f"| epoch {epoch:3d} memory | peak CPU RSS: {epoch_peak_cpu_gb:6.2f} GB |")

        val_score = val_metrics["pearson"]
        stop_early = False
        if val_score > best_val_corr:
            best_val_corr = val_score
            best_model = copy.deepcopy(model)
            logger.info(f"Best model with score {val_score:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                stop_early = True

        scheduler.step()


        checkpoint_tmp_path = str(checkpoint_path) + ".tmp"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_corr": best_val_corr,
            "patience": patience,
            "best_model_state_dict": best_model.state_dict() if best_model is not None else None,
        }, checkpoint_tmp_path)
        os.replace(checkpoint_tmp_path, str(checkpoint_path))

        # Return cached-but-unused GPU memory to the driver and collect any
        # unreferenced CPU-side objects (e.g. the old best_model after a deepcopy).
        # No effect on results -- purely memory hygiene between epochs.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if stop_early:
            break

    best_model_path = save_dir / "best_model.pt"
    torch.save(best_model.state_dict(), str(best_model_path))

    end_training_time = time.time()
    training_time = end_training_time - end_setup_time
    print("Training time: " + str(round(training_time, 2)) + " s")

    # SCORING =========================================================================
    def to_dense(x):
        return np.asarray(x.toarray(), dtype=np.float32) if hasattr(x, "toarray") else np.asarray(x, dtype=np.float32)

    ctrl_mask = pert_data.adata.obs["condition"] == "ctrl"
    mean_expression = to_dense(pert_data.adata[ctrl_mask].X).mean(axis=0)
    raw_ctrl_mean = to_dense(pert_data.adata[ctrl_mask].layers["counts"]).mean(axis=0)

    # id_test_conditions are "ctrl+GENE" (single) or "ctrl+GENE1+GENE2..." (pooled)
    id_test_pert_list = [cond[len("ctrl+"):].split("+") for cond in id_test_conditions]
    predicted_means_dict = predict(best_model, id_test_pert_list, pool_size=300)

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

    for cond, pert_genes in zip(id_test_conditions, id_test_pert_list):
        pred_centered = predicted_means_dict["_".join(pert_genes)] - mean_expression

        test_cells = pert_data.adata[pert_data.adata.obs["condition"] == cond]
        if test_cells.shape[0] == 0:
            skipped_perts.append(cond)
            continue
        gt_centered = to_dense(test_cells.X).mean(axis=0) - mean_expression

        with np.errstate(invalid="ignore", divide="ignore"):
            per_pert_corr[cond] = np.corrcoef(gt_centered, pred_centered)[0, 1]
            per_pert_spearman[cond] = spearmanr(gt_centered, pred_centered).statistic
            per_pert_cosine[cond] = np.dot(gt_centered, pred_centered) / (
                np.linalg.norm(gt_centered) * np.linalg.norm(pred_centered))
        per_pert_mse[cond] = mean_squared_error(gt_centered, pred_centered)
        per_pert_r2[cond] = r2_score(gt_centered, pred_centered)
        per_pert_l2[cond] = np.linalg.norm(gt_centered - pred_centered)

        # predicted gene expression change over the control in the log space
        delta_y[cond] = pred_centered

        # ground truth gene expression change over the control in raw space
        delta_x[cond] = to_dense(test_cells.layers["counts"]).mean(axis=0) - raw_ctrl_mean

        # mean control gene expression in raw space
        control[cond] = raw_ctrl_mean

    if skipped_perts:
        print(f"Skipped {len(skipped_perts)} perturbation(s) with no test cells: {skipped_perts}")

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

    end_evaluation_time = time.time()
    evaluation_time = end_evaluation_time - end_training_time
    total_time = end_evaluation_time - start_time
    print("Evaluation time: " + str(round(evaluation_time, 2)) + " s")
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
                         "SETUP_TIME": setup_time,
                         "TRAINING_TIME": training_time,
                         "EVALUATION_TIME": evaluation_time,
                         "TOTAL_TIME": total_time,
                         "PEAK_CPU_RAM_GB": peak_cpu_gb,
                         "PEAK_GPU_ALLOCATED_GB": peak_gpu_allocated_gb,
                         "PEAK_GPU_RESERVED_GB": peak_gpu_reserved_gb}

    with open(results_path, "wb") as file:
        pickle.dump(exported_results, file)

    print(f"mean Pearson: {mean_corr:.4f} | mean Spearman: {mean_spearman:.4f} | mean MSE: {mean_mse:.4f} | mean R²: {mean_r2:.4f} | mean L2: {mean_l2:.4f} | mean Cosine: {mean_cosine:.4f} over {len(per_pert_corr)} perturbations")

    # CLEANUP =========================================================================
    if data_folder.exists():
        shutil.rmtree(str(data_folder))
        print(f"Removed pyg cache/data folder: {data_folder}")

    if best_model_path.exists():
        os.remove(str(best_model_path))
        print(f"Removed checkpoint: {best_model_path}")

    if checkpoint_path.exists():
        os.remove(str(checkpoint_path))
        print(f"Removed resume checkpoint: {checkpoint_path}")


def _subprocess_entry(args, dataset_name: str) -> None:
    """Child-process wrapper: run one dataset, then hard-exit.

    The hard exit reproduces the previous ``os._exit(0)`` (torch / pyg worker teardown
    can otherwise hang) while keeping each dataset's peak RSS and CUDA stats isolated.
    """
    status = 0
    try:
        run_scgpt(args, dataset_name)
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(status)


def main(argv=None) -> int:
    parser = bench.build_parser(
        "Fine-tune and score scGPT on the CIPHER perturbation benchmark.",
        models=True,
        pretrained=True,
        device=True,
        epochs=DEFAULT_EPOCHS,
        seed=DEFAULT_SEED,
        batch_size=DEFAULT_BATCH_SIZE,
        skip_existing=True,
    )
    args = parser.parse_args(argv)

    datasets = bench.discover_datasets(args.splits_dir, args.dataset)
    print(f"Running {MODEL} on {len(datasets)} dataset(s): {', '.join(datasets)}")

    failures = []
    for dataset_name in datasets:
        # Run in a subprocess so that peak RAM usage is localised to each dataset
        p = mp.Process(target=_subprocess_entry, args=(args, dataset_name))
        p.start()
        p.join()
        if p.exitcode != 0:
            failures.append((dataset_name, p.exitcode))
            print(f"{dataset_name} failed with exit code {p.exitcode}")
        print("\n_____________________________________________")

    if failures:
        print("Failed datasets: " + ", ".join(f"{d} (exit {c})" for d, c in failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
