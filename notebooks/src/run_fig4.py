"""Figure 4 -- CIPHER transfers across datasets and cell atlases.

Engine for the published Fig 4 panels B (cross-dataset ΔR2 scatter across same-cell-type neuron
datasets, correlation ~0.73) and C (mean ΔR2 transfer ladder: shuffled -> mean-field -> cross-dataset
-> true, for same vs different cell type). Panels D/E/F are the CellxGene-atlas transfer, produced by
the shared atlas engine notebooks/src/run_fig3_atlas.py (marson T-cell = E, RPE = F, tissue = D) which
the notebook calls directly.

CIPHER predicts host-dataset perturbation shifts from a source covariance on the shared gene axis via
cipher.forward_predict. Helpers in notebooks/src (not part of the cipher package). Config constants are
module globals the notebook overrides via R.__dict__.update; DATA_DIR / OUTDIR are injected there.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, pearsonr
from sklearn.metrics import r2_score

import cipher
from cipher import (load_matched_datasets, compute_covariance,
                    meanfield_covariance, shuffled_covariance,
                    forward_predict, forward_metrics)
from cipher.normalize import normalize_matrix, library_size

# --- config (injected by the notebook) ---
DATA_DIR = None
OUTDIR = None
NORMALIZATION = "raw"
EXPRESSION_THRESHOLD = 1.0
MIN_SAMPLES = 100
COV_MAX_CELLS = 10000
SEED = 0

SAME_CT = {
    "Tian19_neuron":  "TianKampmann2019_day7neuron.h5ad",
    "Tian19_iPSC":    "TianKampmann2019_iPSC.h5ad",
    "Tian21_CRISPRi": "TianKampmann2021_CRISPRi.h5ad",
    "Tian21_CRISPRa": "TianKampmann2021_CRISPRa.h5ad",
}
DIFF_CT = {
    "Replogle_rpe1": "ReplogleWeissman2022_rpe1.h5ad",
    "Norman":        "NormanWeissman2019_filtered.h5ad",
    "Frangieh":      "FrangiehIzar2021_RNA.h5ad",
}
B_PAIRS = [("Tian19_neuron", "Tian21_CRISPRi"),
           ("Tian21_CRISPRa", "Tian21_CRISPRi"),
           ("Tian21_CRISPRi", "Tian21_CRISPRa")]


def _load_kw():
    return dict(expression_threshold=EXPRESSION_THRESHOLD, min_samples=MIN_SAMPLES)


def _all_map():
    return {**SAME_CT, **DIFF_CT}


def _path(key):
    return os.path.join(DATA_DIR, _all_map()[key])


def _norm(ds, X):
    pc = ds.pflog_pseudocount if NORMALIZATION == "pflog" else None
    return normalize_matrix(X, NORMALIZATION, libsize=library_size(X), pseudocount=pc)


def sigma_of(ds, kind="real", seed=SEED):
    Xc = _norm(ds, ds.control_matrix())
    if COV_MAX_CELLS and Xc.shape[0] > COV_MAX_CELLS:
        rng = np.random.default_rng(cipher.utils.stable_seed(seed, ds.name))
        Xc = Xc[np.sort(rng.choice(Xc.shape[0], COV_MAX_CELLS, replace=False))]
    if kind == "meanfield":
        return meanfield_covariance(Xc, seed=seed)
    if kind == "shuffled":
        return shuffled_covariance(Xc, seed=seed)
    return compute_covariance(Xc)


def matched_forward(host_ds, Sigma, source_label):
    control_mean = _norm(host_ds, host_ds.control_matrix()).mean(0)
    rows = []
    for pert, tgi in zip(host_ds.perturbations, host_ds.target_gene_indices):
        tgi = int(tgi)
        if tgi < 0:
            continue
        Xp = host_ds.perturbation_matrix(pert)
        if Xp.shape[0] < 1:
            continue
        dx = _norm(host_ds, Xp).mean(0) - control_mean
        pred, a_hat = forward_predict(Sigma, dx, tgi)
        rows.append(dict(host=host_ds.name, source=source_label, perturbation=str(pert),
                         target=str(host_ds.gene_names[tgi]), a_hat=float(a_hat),
                         n_cells=int(Xp.shape[0]), **forward_metrics(dx, pred)))
    return pd.DataFrame(rows)


def run_pair(host_key, src_key, seed=SEED):
    host_ds, src_ds = load_matched_datasets(_path(host_key), _path(src_key), **_load_kw())
    return dict(
        cross=matched_forward(host_ds, sigma_of(src_ds, "real"), src_key),
        self_=matched_forward(host_ds, sigma_of(host_ds, "real"), host_key + ":self"),
        mf=matched_forward(host_ds, sigma_of(src_ds, "meanfield", seed), src_key + ":mf"),
        rand=matched_forward(host_ds, sigma_of(src_ds, "shuffled", seed), src_key + ":shuf"),
        n_shared=int(host_ds.n_genes),
    )


def panel_b():
    """B -- R2 with the true (same-dataset) Sigma vs a Sigma from another same-cell-type dataset."""
    recs = []
    for host, src in B_PAIRS:
        r = run_pair(host, src)
        m = r["self_"][["perturbation", "r2_uncentered"]].merge(
            r["cross"][["perturbation", "r2_uncentered"]],
            on="perturbation", suffixes=("_true", "_cross"))
        recs.append(m.assign(pair=f"{host} vs {src}"))
        print(f"{host:14s} vs {src:14s}: {len(m):3d} perts, {r['n_shared']} shared genes")
    B = pd.concat(recs, ignore_index=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    for k, g in B.groupby("pair"):
        ax.scatter(g["r2_uncentered_true"], g["r2_uncentered_cross"], s=20, alpha=.6, label=k)
    lims = (min(B["r2_uncentered_true"].min(), B["r2_uncentered_cross"].min(), 0), 1)
    ax.plot(lims, lims, "k--", lw=1)
    # reference metric: coefficient of determination of the cross-dataset ΔR2 about the
    # same-dataset ΔR2 (y=x), matching r2_score(true, cross) in the published figure.
    r = r2_score(B["r2_uncentered_true"], B["r2_uncentered_cross"])
    ax.set(xlabel="ΔR2 (true Σ, same dataset)",
           ylabel="ΔR2 (Σ from another same-cell-type dataset)",
           title=f"B   ΔR2 correlation = {r:.2f}", xlim=lims, ylim=lims)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig4_panelB.svg")); plt.show()
    return B


def panel_c():
    """C -- mean ΔR2 transfer ladder: shuffled -> mean-field -> cross-dataset -> true, same vs diff CT."""
    hosts = ["Tian19_neuron", "Tian21_CRISPRi", "Tian21_CRISPRa"]
    sources = list(SAME_CT) + list(DIFF_CT)
    ladder = []
    for host in hosts:
        for src in sources:
            if src == host:
                continue
            r = run_pair(host, src)
            same = (host in SAME_CT) and (src in SAME_CT)
            ladder.append(dict(host=host, source=src, same=same,
                               rand=r["rand"]["r2_uncentered"].mean(),
                               mf=r["mf"]["r2_uncentered"].mean(),
                               cross=r["cross"]["r2_uncentered"].mean(),
                               true=r["self_"]["r2_uncentered"].mean()))
            print(f"{host:14s} <- {src:14s} [{'same' if same else 'diff'}]")
    L = pd.DataFrame(ladder)
    steps = ["rand", "mf", "cross", "true"]
    labels = ["shuffled Σ", "mean-field Σ", "cross-dataset Σ", "true Σ (same dataset)"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in L.iterrows():
        ax.plot(labels, row[steps].to_numpy(dtype=float),
                color=("salmon" if row["same"] else "blueviolet"), alpha=.25, lw=1)
    ax.plot(labels, L[L["same"]][steps].mean(), "o-", color="salmon", lw=2.5, label="same cell type")
    ax.plot(labels, L[~L["same"]][steps].mean(), "s-", color="blueviolet", lw=2.5, label="different cell type")
    ax.set(ylabel="mean ΔR2 per dataset", title="C   cross-dataset covariance transfer")
    ax.legend(); plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig4_panelC.svg")); plt.show()
    print("\nWilcoxon signed-rank across adjacent ladder steps (one-sided 'less'):")
    for a, b in zip(steps[:-1], steps[1:]):
        try:
            pv = wilcoxon(L[a], L[b], alternative="less").pvalue
        except ValueError:
            pv = float("nan")
        print(f"  {a:>5s} -> {b:<6s}  p = {pv:.2g}")
    return L
