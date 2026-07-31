"""Figure 3 J-N -- external-method benchmark comparison (violin panels).

Compares CIPHER against published perturbation-response predictors (GenePert, scLAMBDA,
scouter, scGPT, GEARS, and a linear mean-shift baseline) on transcriptome-wide accuracy.
Methods that predict in log space are mapped back to raw count space via the first-order
Taylor correction ``delta_raw ~= control * delta_log`` before scoring, so every method is
compared on the same (raw) footing CIPHER predicts in natively. Panels: per-perturbation
Pearson / Spearman / Cosine / L2 / MSE violins, plus the Systema-style centroid-accuracy
violin (fraction of other perturbations whose ground-truth centroid is farther from the
prediction than the perturbation's own).

The heavy step -- reading the benchmark prediction pkls (~5 GB) and scoring every
perturbation -- runs once and caches small CSVs under OUTDIR; the notebook then just loads
those and draws the violins.  Config constants are module globals the notebook overrides via
``R.__dict__.update``; ``BENCH_ROOT`` / ``SPLITS_ROOT`` / ``OUTDIR`` are injected there.
"""
import os
import glob
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# --- config (injected by the notebook) ---
# Roots of the benchmark artifacts (kept out of the notebook to avoid baking in absolute
# paths / allocation ids -- the notebook passes them through from the environment).
BENCH_ROOT = os.environ.get("CIPHER_BENCH_ROOT", "")     # <BENCH_ROOT>/<model>/<dataset>/results.pkl
SPLITS_ROOT = os.environ.get("CIPHER_BENCH_SPLITS", "")  # <SPLITS_ROOT>/<dataset>/{train_idx.npy,filtered.h5ad}
OUTDIR = None

MODEL_NAMES = ["CIPHER", "GenePert", "scLAMBDA", "scouter", "scGPT", "GEARS", "linear_mean_raw"]
RAW_NATIVE_MODELS = {"CIPHER", "linear_mean_raw"}   # already scored in raw space -> stage "raw"
GENEPERT_RIDGE = "0.1"                               # GenePert result file is "<ridge>.pkl"
LOW_CORR_THRESHOLD = 0.85
MIN_TRAINING_CELLS = 50
METRIC_KEYS = {"PCC": "CORRELATION", "Spearman": "SPEARMAN",
               "Cosine": "COSINE", "L2": "L2", "MSE": "MSE"}
IO_WORKERS = int(os.environ.get("CIPHER_BENCH_WORKERS", "8"))

PAPER_CIPHER_COLOR = "#a1a1a1"    # dark grey: CIPHER
PAPER_OTHER_COLOR = "#cacbcc"     # light grey: every other method

# cross-run state populated by compute_all()
correction_df = None
centroid_df = None
pert_status_df = None
_ACCEPTABLE = None                 # dataset -> set(acceptable perturbations)


# ----------------------------------------------------------------------------- helpers
def normalize_pert_name(gene):
    if gene.endswith("+ctrl"):
        return gene[:-len("+ctrl")]
    if gene.startswith("ctrl+"):
        return gene[len("ctrl+"):]
    return gene


def _model_files(model_name):
    """(dataset_name, path) pairs for a model's per-dataset result pkls."""
    model_dir = os.path.join(BENCH_ROOT, model_name)
    out = []
    if model_name == "GenePert":
        for d in glob.glob(os.path.join(model_dir, "*")):
            p = os.path.join(d, f"{GENEPERT_RIDGE}.pkl")
            if os.path.isdir(d) and os.path.exists(p):
                out.append((os.path.basename(d), p))
    else:
        for p in glob.glob(os.path.join(model_dir, "*", "results.pkl")):
            out.append((os.path.basename(os.path.dirname(p)), p))
    return out


# --------------------------------------------------------------- perturbation pruning
def compute_pert_status():
    """Per-dataset acceptable perturbations (drops proportionality disagreements, low draw
    correlations, and perturbations with fewer than MIN_TRAINING_CELLS training cells)."""
    import scanpy as sc
    global _ACCEPTABLE, pert_status_df
    status_rows = []
    acceptable = {}
    for path in sorted(glob.glob(os.path.join(BENCH_ROOT, "CIPHER", "*"))):
        pkl = os.path.join(path, "results.pkl")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as fh:
            res = pickle.load(fh)
        name = os.path.basename(path)
        agree = res["TRAINING_DRAW_PROPORTIONALITY_AGREEMENT"]
        corr = res["TRAINING_DRAW_PROPORTIONALITY_CORRELATION"]
        disagreement = {p for p, a in agree.items() if not np.all(a)}
        low_corr = {p for p in agree if np.mean(corr[p]) < LOW_CORR_THRESHOLD}
        train_idx = np.load(os.path.join(SPLITS_ROOT, name, "train_idx.npy"))
        obs = sc.read_h5ad(os.path.join(SPLITS_ROOT, name, "filtered.h5ad"), backed="r").obs
        counts = obs["perturbation"].iloc[train_idx].value_counts()
        all_perts = set(agree.keys())
        low_cells = set(counts[counts < MIN_TRAINING_CELLS].index) & all_perts
        ok = all_perts - disagreement - low_corr - low_cells
        acceptable[name] = ok
        status_rows.append({"dataset": name, "total": len(all_perts),
                            "disagreement": len(disagreement), "low_corr": len(low_corr),
                            "low_cell_count": len(low_cells), "acceptable": len(ok)})
    _ACCEPTABLE = acceptable
    pert_status_df = pd.DataFrame(status_rows).sort_values("dataset").reset_index(drop=True)
    return pert_status_df


# --------------------------------------------------------------------- metric scoring
def _score_file(model_name, dataset_name, path):
    with open(path, "rb") as fh:
        res = pickle.load(fh)
    acc = _ACCEPTABLE.get(dataset_name, set())
    rows = []
    if model_name in RAW_NATIVE_MODELS:
        # already raw-space: read the stored metric values straight through as stage "raw".
        for metric, key in METRIC_KEYS.items():
            for pert, val in res.get(key, {}).items():
                if normalize_pert_name(pert) in acc:
                    rows.append((model_name, dataset_name, pert, metric, "raw", float(val)))
        return rows
    if "DELTA_Y" not in res:
        return rows
    for pert in res["DELTA_Y"]:
        if normalize_pert_name(pert) not in acc:
            continue
        control = np.asarray(res["CONTROL"][pert])
        delta_x = np.asarray(res["DELTA_X"][pert])          # observed raw shift
        lc = control * np.asarray(res["DELTA_Y"][pert])     # Taylor: log-shift -> raw shift
        with np.errstate(invalid="ignore"):
            after = {
                "PCC": np.corrcoef(lc, delta_x)[0, 1],
                "Spearman": spearmanr(lc, delta_x).statistic,
                "Cosine": float(np.dot(lc, delta_x) / (np.linalg.norm(lc) * np.linalg.norm(delta_x))),
                "L2": float(np.linalg.norm(lc - delta_x)),
                "MSE": float(np.mean((lc - delta_x) ** 2)),
            }
        for metric, key in METRIC_KEYS.items():
            rows.append((model_name, dataset_name, pert, metric, "before", float(res[key][pert])))
            rows.append((model_name, dataset_name, pert, metric, "after", after[metric]))
    return rows


def compute_correction_df():
    """Per-perturbation metrics for every model, raw and (for log-space models) Taylor-corrected."""
    global correction_df
    tasks = [(m, d, p) for m in MODEL_NAMES for d, p in _model_files(m)]
    records = []
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as ex:
        futs = {ex.submit(_score_file, m, d, p): (m, d) for m, d, p in tasks}
        for fut in as_completed(futs):
            records.extend(fut.result())
    correction_df = pd.DataFrame(records, columns=["model", "dataset", "perturbation",
                                                   "metric", "stage", "value"])
    return correction_df


def raw_space_df(df, metric_name):
    """Rows for one metric in the space each method is evaluated in (raw for native models,
    Taylor-corrected 'after' for the log-space models)."""
    m = df[df["metric"] == metric_name]
    return m[(m["model"].isin(RAW_NATIVE_MODELS) & (m["stage"] == "raw"))
             | (~m["model"].isin(RAW_NATIVE_MODELS) & (m["stage"] == "after"))]


# ---------------------------------------------------------- Systema centroid accuracy
def _absolute_profiles(model_name, dataset_name, path):
    with open(path, "rb") as fh:
        res = pickle.load(fh)
    if "DELTA_Y" not in res:
        return {}
    acc = _ACCEPTABLE.get(dataset_name, set())
    prof = {}
    for pert in res["DELTA_Y"]:
        g = normalize_pert_name(pert)
        if g not in acc:
            continue
        control = np.asarray(res["CONTROL"][pert])
        gt = control + np.asarray(res["DELTA_X"][pert])                       # O(X)
        if model_name == "CIPHER" and "RAW_PREDICTION" in res:
            pred_delta = np.asarray(res["RAW_PREDICTION"][pert])
        elif model_name == "linear_mean_raw":
            pred_delta = np.asarray(res["DELTA_Y"][pert])
        else:
            pred_delta = control * np.asarray(res["DELTA_Y"][pert])           # Taylor -> raw
        prof[g] = (gt, control + pred_delta)                                  # (O(X), O_pred(X))
    return prof


def _centroid_accuracy(dataset_name):
    import scanpy as sc
    from scipy.spatial.distance import cdist
    genes = sc.read_h5ad(os.path.join(SPLITS_ROOT, dataset_name, "filtered.h5ad"),
                         backed="r").var_names.astype(str).to_numpy()
    ground_truth, pred_frames = {}, []
    for model_name in MODEL_NAMES:
        files = dict(_model_files(model_name))
        if dataset_name not in files:
            continue
        prof = _absolute_profiles(model_name, dataset_name, files[dataset_name])
        if not prof:
            continue
        perts = list(prof.keys())
        pdf = pd.DataFrame(np.vstack([prof[p][1] for p in perts]), index=perts, columns=genes)
        pdf.index = pd.MultiIndex.from_arrays([pdf.index, [model_name] * len(pdf)],
                                              names=["condition", "method"])
        pred_frames.append(pdf)
        for p in perts:
            ground_truth[p] = prof[p][0]
    if not pred_frames:
        return None
    gt_df = pd.DataFrame.from_dict(ground_truth, orient="index", columns=genes)
    pred_df = pd.concat(pred_frames, axis=0)
    dist = pd.DataFrame(cdist(pred_df, gt_df, metric="euclidean"),
                        index=pred_df.index, columns=gt_df.index)
    idx = [g for g in dist.index.get_level_values(0) if g in dist.columns]
    multi = [(g, m) for g, m in dist.index if g in dist.columns]
    cols = [np.argwhere(dist.columns == g)[0][0] for g in idx]
    self_d = pd.DataFrame(np.diag(dist.iloc[np.arange(len(dist)), cols]),
                          index=pd.MultiIndex.from_tuples(multi))
    scores = {}
    for method in pred_df.index.get_level_values(1).unique():
        x = dist.xs(method, level=1).sort_index()
        y = self_d.xs(method, level=1).sort_index()
        scores[method] = (x > y.values).sum(axis=1) / (x.shape[-1] - 1)
    out = pd.DataFrame(scores).rename_axis("perturbation").reset_index().melt(
        id_vars="perturbation", var_name="model", value_name="centroid_accuracy")
    out["dataset"] = dataset_name
    return out


def compute_centroid_df():
    global centroid_df
    frames = []
    for dataset_name, ok in _ACCEPTABLE.items():
        if not ok:
            continue
        cur = _centroid_accuracy(dataset_name)
        if cur is not None:
            frames.append(cur)
    centroid_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return centroid_df


# --------------------------------------------------------------- cache-aware driver
def compute_all(force=False):
    """Load the cached benchmark CSVs from OUTDIR, or (re)build them from the pkls."""
    global correction_df, centroid_df, pert_status_df
    corr_csv = os.path.join(OUTDIR, "benchmark_metrics.csv")
    cent_csv = os.path.join(OUTDIR, "centroid_accuracy.csv")
    stat_csv = os.path.join(OUTDIR, "perturbation_status.csv")
    if not force and os.path.exists(corr_csv) and os.path.exists(cent_csv):
        correction_df = pd.read_csv(corr_csv)
        centroid_df = pd.read_csv(cent_csv)
        pert_status_df = pd.read_csv(stat_csv) if os.path.exists(stat_csv) else None
        print(f"loaded cached benchmark metrics: {len(correction_df):,} metric rows, "
              f"{len(centroid_df):,} centroid rows")
        return
    if not BENCH_ROOT or not SPLITS_ROOT:
        raise RuntimeError("benchmark cache missing and CIPHER_BENCH_ROOT/CIPHER_BENCH_SPLITS not set")
    print("building benchmark metrics from prediction pkls (one-time)...")
    compute_pert_status()
    compute_correction_df()
    compute_centroid_df()
    os.makedirs(OUTDIR, exist_ok=True)
    correction_df.to_csv(corr_csv, index=False)
    centroid_df.to_csv(cent_csv, index=False)
    pert_status_df.to_csv(stat_csv, index=False)
    print(f"cached {len(correction_df):,} metric rows and {len(centroid_df):,} centroid rows")


# --------------------------------------------------------------------------- plotting
def _violin(ax, series_by_model, ylabel, title=None, y_limits=(-1, 1)):
    plot_models = [m for m in MODEL_NAMES if m in series_by_model and len(series_by_model[m]) >= 2]
    ax.set_xlim(0.5, len(plot_models) + 0.5)
    for i, model_name in enumerate(plot_models):
        y = series_by_model[model_name].dropna()
        color = PAPER_CIPHER_COLOR if model_name == "CIPHER" else PAPER_OTHER_COLOR
        parts = ax.violinplot(y, positions=[i + 1], showmeans=False,
                              showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_edgecolor("black")
            pc.set_linewidth(0.8); pc.set_alpha(1)
        mean_val = y.mean()
        txt = ax.text(i + 1, mean_val + 0.04, f"{mean_val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.figure.canvas.draw()
        bb = txt.get_window_extent(renderer=ax.figure.canvas.get_renderer()).transformed(
            ax.transData.inverted())
        hw = (bb.x1 - bb.x0) / 2
        ax.hlines(mean_val, i + 1 - hw, i + 1 + hw, color="black", linewidth=1.8, zorder=5)
        txt.remove()
    ax.set_xticks(range(1, len(plot_models) + 1))
    ax.set_xticklabels(plot_models, rotation=30, ha="right")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)


# (metric, ylabel, title, y_limits) -- L2/MSE keep autoscaled limits
_METRIC_SPECS = [("PCC", "Pearson's", "Pearson Correlation", (-1, 1)),
                 ("Spearman", "Spearman", "Spearman Correlation", (-1, 1)),
                 ("Cosine", "Cosine", "Cosine Similarity", (-1, 1)),
                 ("L2", "L2 Error", "L2 Error", None),
                 ("MSE", "Mean Squared Error", "Mean Squared Error", None)]


def plot_metric_violins():
    """The five per-perturbation metric violins (J/K/L + supplements)."""
    for metric, _y, _t, _ in _METRIC_SPECS:
        s = raw_space_df(correction_df, metric)
        means = s.groupby("model")["value"].mean().reindex(
            [m for m in MODEL_NAMES if m in s["model"].unique()]).sort_values(ascending=False)
        print(f"[{metric}] raw-space mean by model: " +
              ", ".join(f"{m}={v:.3f}" for m, v in means.items()))
    for metric, ylabel, title, ylim in _METRIC_SPECS:
        s = raw_space_df(correction_df, metric)
        by_model = {m: s.loc[s["model"] == m, "value"] for m in MODEL_NAMES}
        fig, ax = plt.subplots(figsize=(6, 6))
        _violin(ax, by_model, ylabel, title=title, y_limits=ylim)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"fig3_benchmark_{metric}.svg"))
        plt.show()


def plot_centroid():
    """Systema-style centroid-accuracy violin."""
    means = centroid_df.groupby("model")["centroid_accuracy"].mean().sort_values(ascending=False)
    print("[centroid] mean by model: " + ", ".join(f"{m}={v:.3f}" for m, v in means.items()))
    by_model = {m: centroid_df.loc[centroid_df["model"] == m, "centroid_accuracy"] for m in MODEL_NAMES}
    fig, ax = plt.subplots(figsize=(6, 6))
    _violin(ax, by_model, "Centroid Accuracy", title="Centroid Accuracy", y_limits=(0, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig3_benchmark_centroid.svg"))
    plt.show()
