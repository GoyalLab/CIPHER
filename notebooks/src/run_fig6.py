"""Figure 6 -- the effective dimensions of transcriptome-wide response.

Engine for the computable Fig 6 panels: A (participation-ratio / effective-dimension distributions of
the perturbation response across control-covariance principal components, per dataset), B (clustered
heatmaps of the response fraction along each PC), and G (mean participation ratio vs mean forward ΔR2
per dataset). Panels H/I (effective number of driving genes) reuse the suppl engine run_figS15; panels
J/K/L (scDesign3 raw-vs-library-size mixing / accuracy) reuse the scDesign3 analysis. Panels C/D/E/F
(mu-axis vs orthogonal decomposition and its gene-set enrichment) are a distinct analysis.

Participation ratio of a response dx: project dx onto the control-covariance eigenbasis V (Sigma =
V diag(lam) V^T), take fractions f_i = z_i^2 / sum_j z_j^2 with z = V^T dx, and PR = (sum f)^2 / sum f^2
= 1 / sum_i f_i^2 (the effective number of engaged modes).

Helpers in notebooks/src (not part of the cipher package). Config constants are module globals the
notebook overrides via R.__dict__.update; DATA_DIR / OUTDIR are injected there.
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cipher
from cipher import load_dataset, dataset_group
from cipher.covariance import compute_covariance
from cipher.core import forward_fit, forward_metrics
from cipher.normalize import normalize_matrix, library_size

# --- config (injected by the notebook) ---
DATA_DIR = None
OUTDIR = None
NORMALIZATION = "raw"
# 1.0 == the reference "mean_control_ge_1p0" gene filter used throughout the paper (fig3/fig5
# both needed it to reproduce); panel G's mean ΔR2 is the same forward metric as fig3 C/D, so it
# has to be measured on the same gene set.
EXPRESSION_THRESHOLD = 1.0
MIN_SAMPLES = 100
COV_MAX_CELLS = 10000
SEED = 0
N_PC_HEATMAP = 10                       # top PCs shown in the B heatmaps
# Cache the per-dataset participation ratios / response fractions / mean dR2 behind panels A/B/G.
# That sweep re-reads every CRISPRi/a h5ad and dominates the notebook's runtime, so it is computed
# once and reloaded afterwards; iterating on panel G then costs seconds instead of an hour.
# Delete OUTDIR/pr_cache (or set False) to force a recompute.
COMPUTE_CACHE = True
# Datasets shown in published panel G (Replogle22 a/b, Nadig25 a/b, Norman19, Frangieh21,
# Tian21 a/b, Tian19 a/b). Empty set => fit every available dataset.
PANEL_G_DATASETS = {
    "ReplogleWeissman2022_rpe1", "ReplogleWeissman2022_K562_essential",
    "GSE264667_hepg2_raw_singlecell_01", "GSE264667_jurkat_raw_singlecell_01",
    "NormanWeissman2019_filtered", "FrangiehIzar2021_RNA",
    "TianKampmann2021_CRISPRa", "TianKampmann2021_CRISPRi",
    "TianKampmann2019_day7neuron", "TianKampmann2019_iPSC",
}
A_DATASETS = ["TianKampmann2019_day7neuron", "NormanWeissman2019_filtered",
              "ReplogleWeissman2022_rpe1"]   # the three shown in panel A/B

# cross-section state
pr_per_dataset = None   # {dataset: np.array of per-perturbation participation ratios}
frac_per_dataset = None # {dataset: (n_pert x N_PC_HEATMAP) response fractions}
dr2_per_dataset = None  # {dataset: mean forward ΔR2 (real vs mean-field)}


def _norm(ds, X):
    pc = ds.pflog_pseudocount if NORMALIZATION == "pflog" else None
    return normalize_matrix(X, NORMALIZATION, libsize=library_size(X), pseudocount=pc)


def _dataset_pr(ds, seed=SEED):
    control_norm = _norm(ds, ds.control_matrix())
    control_mean = control_norm.mean(0)
    rng = np.random.default_rng(cipher.utils.stable_seed(seed, ds.name))
    cov_in = control_norm
    if COV_MAX_CELLS and control_norm.shape[0] > COV_MAX_CELLS:
        cov_in = control_norm[np.sort(rng.choice(control_norm.shape[0], COV_MAX_CELLS, replace=False))]
    Sigma = compute_covariance(cov_in)
    lam, V = np.linalg.eigh(Sigma)                  # ascending
    order = np.argsort(-lam); V = V[:, order]       # PCs by descending eigenvalue
    perts = [(p, int(g)) for p, g in zip(ds.perturbations, ds.target_gene_indices) if int(g) >= 0]
    pr, fracs, dr2_real, dr2_mf = [], [], [], []
    from cipher.covariance import meanfield_covariance
    Sigma_mf = meanfield_covariance(cov_in, seed=seed)
    for p, g in perts:
        dx = _norm(ds, ds.perturbation_matrix(p)).mean(0) - control_mean
        z = V.T @ dx
        f = z ** 2
        s = f.sum()
        if s <= 0:
            continue
        f = f / s
        pr.append(1.0 / np.sum(f ** 2))
        fracs.append(f[:N_PC_HEATMAP])
        a_r, _ = forward_fit(Sigma[:, g], dx); a_m, _ = forward_fit(Sigma_mf[:, g], dx)
        dr2_real.append(forward_metrics(dx, (a_r if np.isfinite(a_r) else 0) * Sigma[:, g])["r2_uncentered"])
        dr2_mf.append(forward_metrics(dx, (a_m if np.isfinite(a_m) else 0) * Sigma_mf[:, g])["r2_uncentered"])
    return (np.asarray(pr), np.asarray(fracs),
            float(np.nanmean(dr2_real)) if dr2_real else np.nan)


def _pr_cache_path(name):
    tag = f"{NORMALIZATION}_t{str(EXPRESSION_THRESHOLD).replace('.', 'p')}"
    return os.path.join(OUTDIR, "pr_cache", f"{name}__{tag}.npz")


def compute_all():
    """Per-perturbation participation ratios + response fractions + mean ΔR2 for all CRISPRi/a.

    Each dataset's result is cached (see COMPUTE_CACHE); the cache key carries the normalization
    and expression threshold so changing either recomputes rather than silently reusing.
    """
    global pr_per_dataset, frac_per_dataset, dr2_per_dataset
    pr_per_dataset, frac_per_dataset, dr2_per_dataset = {}, {}, {}
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5ad")))
    for f in files:
        name = os.path.basename(f)[:-5]
        if dataset_group(name) not in ("CRISPRi", "CRISPRa"):
            continue
        cache = _pr_cache_path(name)
        if COMPUTE_CACHE and os.path.exists(cache):
            z = np.load(cache)
            pr, fr, dr2 = z["pr"], z["frac"], float(z["dr2"])
            pr_per_dataset[name] = pr; frac_per_dataset[name] = fr; dr2_per_dataset[name] = dr2
            print(f"{name:40s} n_pert={len(pr):4d} mean PR={np.nanmean(pr):.2f} "
                  f"mean ΔR2={dr2:.3f} [cached]")
            continue
        try:
            ds = load_dataset(f, expression_threshold=EXPRESSION_THRESHOLD, min_samples=MIN_SAMPLES)
            pr, fr, dr2 = _dataset_pr(ds)
            pr_per_dataset[name] = pr; frac_per_dataset[name] = fr; dr2_per_dataset[name] = dr2
            if COMPUTE_CACHE:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.savez_compressed(cache, pr=np.asarray(pr), frac=np.asarray(fr),
                                    dr2=np.asarray(float(dr2)))
            print(f"{name:40s} n_pert={len(pr):4d} mean PR={np.nanmean(pr):.2f} mean ΔR2={dr2:.3f}")
        except Exception as e:  # noqa: BLE001
            print("SKIP", name, repr(e))


def panel_ab():
    """A -- participation-ratio distributions; B -- response-fraction PC heatmaps (3 example datasets)."""
    shown = [d for d in A_DATASETS if d in pr_per_dataset] or list(pr_per_dataset)[:3]
    fig, ax = plt.subplots(2, len(shown), figsize=(5 * len(shown), 8))
    for j, d in enumerate(shown):
        ax[0, j].hist(pr_per_dataset[d], bins=30, color="0.5", density=True)
        ax[0, j].set(xlabel="participation ratio", ylabel="density", title=f"A   {d}")
        fr = frac_per_dataset[d]
        col_order = np.argsort(-fr[:, 0]) if len(fr) else np.arange(0)
        im = ax[1, j].imshow(fr[col_order].T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax[1, j].set(xlabel="perturbations", ylabel="principal components", title=f"B   {d}")
        fig.colorbar(im, ax=ax[1, j], fraction=0.03)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig6_panels_AB.svg")); plt.show()


def panel_g():
    """G -- mean participation ratio vs mean forward ΔR2 per dataset.

    Restricted to the datasets the published panel plots (Replogle22 a/b, Nadig25 a/b,
    Norman19, Frangieh21, Tian21 a/b, Tian19 a/b); fitting every available dataset instead
    pulls in the many Marson/XAtlas/CRISPRa sets the panel does not show and changes the fit.
    """
    names = [d for d in pr_per_dataset
             if np.isfinite(dr2_per_dataset.get(d, np.nan))
             and (not PANEL_G_DATASETS or d in PANEL_G_DATASETS)]
    x = np.array([dr2_per_dataset[d] for d in names])
    y = np.array([np.nanmean(pr_per_dataset[d]) for d in names])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=40, color="#4C78A8")
    for d, xi, yi in zip(names, x, y):
        ax.annotate(d, (xi, yi), fontsize=6, alpha=0.7)
    if len(x) >= 2:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50); ax.plot(xs, a + b * xs, "k--", lw=1)
        ss_res = np.sum((y - (a + b * x)) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        ax.set_title(f"G   R2 = {r2:.2f}, slope = {b:.2f}")
    ax.set(xlabel="mean ΔR2 (real)", ylabel="mean participation ratio")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig6_panelG.svg")); plt.show()


# ======================================================================================
# Panels C/D/E/F -- mu-axis vs orthogonal response decomposition.
#
# RECONSTRUCTION NOTE: the published C/D/E/F consume a per-perturbation decomposition table
# (all_mu_component_diagnostics.tsv) whose original producer pipeline is not present in this
# repository. The functions below RE-DERIVE that decomposition in Python from the base h5ads
# with an explicit, documented definition, so C/D/F are reproduced up to the definition of the
# global mu-axis. mu is the per-dataset control-mean expression vector (the global "total-RNA"
# / library-scaling direction), unit-normalized. For each perturbation:
#   frac_mu    = <dx, mu_hat>^2 / ||dx||^2                 (fraction of response norm along mu)
#   pearson_full     = Pearson(dx, forward_pred(Sigma, dx, target))         (full covariance)
#   pearson_residual = Pearson(dx_perp, forward_pred(Sigma, dx_perp, target)), dx_perp = dx - proj_mu(dx)
#   mu_axis_gain     = pearson_full - pearson_residual
# C plots mu_axis_gain (x) vs pearson_residual (y), colored by frac_mu; D highlights the top-150
# by each; F shows the standardized growth-scaling c = 1 - dx/mu per gene (recurrent large-|c|).
# ======================================================================================

mu_axis_df = None   # populated by compute_mu_axis()


def _pearson_vec(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compute_mu_axis(seed=SEED):
    """Re-derive the per-perturbation mu-axis decomposition for all CRISPRi/a datasets."""
    global mu_axis_df
    rows = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5ad")))
    for f in files:
        name = os.path.basename(f)[:-5]
        if dataset_group(name) not in ("CRISPRi", "CRISPRa"):
            continue
        try:
            ds = load_dataset(f, expression_threshold=EXPRESSION_THRESHOLD, min_samples=MIN_SAMPLES)
        except Exception as e:  # noqa: BLE001
            print("SKIP", name, repr(e)); continue
        control_norm = _norm(ds, ds.control_matrix()); control_mean = control_norm.mean(0)
        rng = np.random.default_rng(cipher.utils.stable_seed(seed, ds.name))
        cov_in = control_norm
        if COV_MAX_CELLS and control_norm.shape[0] > COV_MAX_CELLS:
            cov_in = control_norm[np.sort(rng.choice(control_norm.shape[0], COV_MAX_CELLS, replace=False))]
        Sigma = compute_covariance(cov_in)
        perts = [(p, int(g)) for p, g in zip(ds.perturbations, ds.target_gene_indices) if int(g) >= 0]
        dxs, tg = [], []
        for p, g in perts:
            dxs.append(_norm(ds, ds.perturbation_matrix(p)).mean(0) - control_mean); tg.append(g)
        if not dxs:
            continue
        dxs = np.asarray(dxs)
        mu = control_mean                                 # control-mean (total-RNA) axis
        mu_hat = mu / (np.linalg.norm(mu) + 1e-12)
        for (p, g), dx in zip(perts, dxs):
            nrm2 = float(dx @ dx)
            if nrm2 <= 0:
                continue
            a = float(dx @ mu_hat)
            frac_mu = a * a / nrm2
            dx_perp = dx - a * mu_hat
            af, _ = forward_fit(Sigma[:, g], dx)
            ap, _ = forward_fit(Sigma[:, g], dx_perp)
            pf = _pearson_vec(dx, (af if np.isfinite(af) else 0) * Sigma[:, g])
            pr = _pearson_vec(dx_perp, (ap if np.isfinite(ap) else 0) * Sigma[:, g])
            # standardized growth scaling c = 1 - dx/mu, gene-median over engaged genes
            with np.errstate(divide="ignore", invalid="ignore"):
                c_gene = 1.0 - dx / mu
            c_med = float(np.nanmedian(c_gene[np.isfinite(c_gene)]))
            rows.append(dict(dataset=name, group=dataset_group(name), perturbation=str(p),
                             target=str(ds.gene_names[g]), frac_mu=frac_mu,
                             pearson_full=pf, pearson_residual=pr,
                             mu_axis_gain=(pf - pr) if (np.isfinite(pf) and np.isfinite(pr)) else np.nan,
                             c=c_med))
        print(f"{name:40s} {len(dxs):4d} perts, mean frac_mu={np.nanmean([r['frac_mu'] for r in rows if r['dataset']==name]):.3f}")
    mu_axis_df = pd.DataFrame(rows)
    mu_axis_df.to_csv(os.path.join(OUTDIR, "fig6_mu_component_diagnostics.csv"), index=False)
    return mu_axis_df


def panel_cd():
    """C/D -- mu-axis gain vs orthogonal residual Pearson (C colored by frac_mu; D top-150 groups)."""
    d = mu_axis_df.dropna(subset=["mu_axis_gain", "pearson_residual"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    sc = ax[0].scatter(d["mu_axis_gain"], d["pearson_residual"], c=d["frac_mu"], s=8, cmap="viridis", vmin=0, vmax=1)
    ax[0].axhline(0, color="0.6", ls="--", lw=0.8); ax[0].axvline(0, color="0.6", ls="--", lw=0.8)
    ax[0].set(xlabel="Pearson gain from μ axis: full - residual", ylabel="μ-orthogonal residual Pearson", title="C")
    fig.colorbar(sc, ax=ax[0], label="fraction of true Δx norm2 along μ")
    top_mu = set(d.nlargest(150, "mu_axis_gain").index)
    top_orth = set(d.nlargest(150, "pearson_residual").index)
    for idx, row in d.iterrows():
        col = "orange" if idx in top_mu else ("blue" if idx in top_orth else "0.8")
        ax[1].scatter(row["mu_axis_gain"], row["pearson_residual"], color=col, s=8)
    ax[1].axhline(0, color="0.6", ls="--", lw=0.8); ax[1].axvline(0, color="0.6", ls="--", lw=0.8)
    ax[1].set(xlabel="Pearson gain from μ axis: full - residual", ylabel="μ-orthogonal residual Pearson",
              title="D   top-150 mostly-μ-axis (orange) vs mostly-orthogonal (blue)")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig6_panels_CD.svg")); plt.show()


def panel_f():
    """F -- standardized growth-scaling factor c = 1 - dx/mu, recurrent large-|c| genes across datasets."""
    g = (mu_axis_df.dropna(subset=["c"]).groupby("target")
         .agg(c_mean=("c", "mean"), c_std=("c", "std"), n=("dataset", "nunique")).reset_index())
    g = g[g["n"] >= 2].reindex(g[g["n"] >= 2]["c_mean"].abs().sort_values(ascending=False).index).head(14)
    g = g.sort_values("c_mean")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(g["c_mean"], np.arange(len(g)), xerr=g["c_std"].fillna(0), fmt="s", color="0.3", ecolor="0.6")
    ax.set_yticks(np.arange(len(g))); ax.set_yticklabels([f"{t} (n={int(n)})" for t, n in zip(g["target"], g["n"])], fontsize=8)
    ax.axvline(0, color="k", ls="--", lw=1); ax.axvline(0.2, color="0.6", ls=":", lw=1); ax.axvline(-0.2, color="0.6", ls=":", lw=1)
    ax.set(xlabel="mean c across datasets within each perturbation modality", title="F")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig6_panelF.svg")); plt.show()


def panel_e_gsea():
    """E -- Hallmark/KEGG enrichment of the mostly-μ-axis vs mostly-orthogonal gene groups (gseapy)."""
    d = mu_axis_df.dropna(subset=["mu_axis_gain", "pearson_residual"])
    mu_genes = list(d.nlargest(150, "mu_axis_gain")["target"].unique())
    orth_genes = list(d.nlargest(150, "pearson_residual")["target"].unique())
    print(f"mostly-μ-axis genes: {len(mu_genes)}; mostly-orthogonal genes: {len(orth_genes)}")
    try:
        import gseapy as gp
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for a, genes, title in [(ax[0], mu_genes, "E   mostly-μ-axis"), (ax[1], orth_genes, "mostly-μ-orthogonal")]:
            enr = gp.enrichr(gene_list=genes, gene_sets=["MSigDB_Hallmark_2020", "KEGG_2021_Human"], outdir=None)
            res = enr.results.sort_values("Adjusted P-value").head(8)
            a.barh(np.arange(len(res))[::-1], -np.log10(res["Adjusted P-value"]), color="#E45756")
            a.set_yticks(np.arange(len(res))[::-1]); a.set_yticklabels(res["Term"], fontsize=7)
            a.axvline(-np.log10(0.05), color="k", ls="--", lw=1); a.set(xlabel="-log10(FDR)", title=title)
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig6_panelE_gsea.svg")); plt.show()
    except Exception as e:  # noqa: BLE001
        print(f"[E] GSEA skipped ({e!r}); gene lists saved. Needs gseapy + Enrichr network or local .gmt.")
        pd.DataFrame({"mu_axis_genes": pd.Series(mu_genes), "orthogonal_genes": pd.Series(orth_genes)}
                     ).to_csv(os.path.join(OUTDIR, "fig6_panelE_gene_groups.csv"), index=False)
