"""Figure 7 -- CIPHER predictions validate on therapy resistance and cell-fate traits.

Engine for the computationally-reproducible Fig 7 panels: B/F (naive vs resistant UMAPs for the
WM989 melanoma and MIA PaCa-2 KRAS datasets), C (melanoma resistance-driver ranking by posterior
mean mu; FN1/IGFBP7 top), G (KRAS resistance-driver ranking), and K/L (LARRY lineage-prediction
performance). The ranking / fate panels reuse the shared suppl engines run_melanoma_resistance,
run_kras_resistance and run_fate_figM7 (the same analytic-Gaussian-posterior and prospective-fate
pipelines as the supplementary notebooks); this module only wires their config and adds the UMAPs.
Panels D/E/H/I/J are wet-lab colony assays (plate images + counts) and are not computed here.

Helpers in notebooks/src (not part of the cipher package). Config constants are module globals the
notebook overrides via R.__dict__.update; DATA_DIR / SUPPL / OUTDIR are injected there.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

# --- config (injected by the notebook) ---
DATA_DIR = None
SUPPL = None
OUTDIR = None
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.3
SEED = 0

# naive/resistant datasets for the UMAP panels
MEL_H5AD_NAME = os.path.join("GSE233766", "Xtot_naive_resistant_melanoma_unbalanced.h5ad")
KRAS_H5AD_NAME = "pancreatic_naive_vs_resistant.h5ad"


def _naive_resistant_umap(ax, h5ad_path, title):
    import scanpy as sc
    import anndata as ad
    a = ad.read_h5ad(h5ad_path)
    # resolve a naive/resistant label column
    key = next((c for c in a.obs.columns
                if a.obs[c].astype(str).str.contains("resist|naive", case=False).any()), None)
    if key is None:
        ax.set_title(f"{title}: no naive/resistant label"); ax.axis("off"); return
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10); sc.tl.pca(a, n_comps=30, random_state=SEED)
    sc.pp.neighbors(a, n_neighbors=UMAP_N_NEIGHBORS, random_state=SEED)
    sc.tl.umap(a, min_dist=UMAP_MIN_DIST, random_state=SEED)
    lab = a.obs[key].astype(str).str.lower()
    is_res = lab.str.contains("resist").to_numpy()
    xy = a.obsm["X_umap"]
    ax.scatter(xy[~is_res, 0], xy[~is_res, 1], s=3, alpha=0.5, color="#1f77b4", label="naive", rasterized=True)
    ax.scatter(xy[is_res, 0], xy[is_res, 1], s=3, alpha=0.5, color="#ff7f0e", label="resistant", rasterized=True)
    ax.set(title=title, xlabel="UMAP1", ylabel="UMAP2"); ax.legend(markerscale=3, fontsize=8)


def panel_umaps_bf():
    """B/F -- naive vs resistant UMAPs (WM989 melanoma, MIA PaCa-2 KRAS)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    _naive_resistant_umap(ax[0], os.path.join(SUPPL, MEL_H5AD_NAME), "B   WM989 (BRAF/MEKi)")
    _naive_resistant_umap(ax[1], os.path.join(SUPPL, KRAS_H5AD_NAME), "F   MIA PaCa-2 (KRAS G12Ci)")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig7_panels_BF_umap.svg")); plt.show()


def _inject(mod, **cfg):
    mod.__dict__.update(cfg)


def panel_c_melanoma():
    """C -- melanoma (WM989) resistance-driver ranking by posterior mean mu (FN1/IGFBP7 top).
    Reuses run_melanoma_resistance (analytic-Gaussian posterior)."""
    import src.run_melanoma_resistance as ME
    base = os.path.join(OUTDIR, "melanoma_analytic_gaussian")
    outdir = os.path.join(base, "analytic_gaussian_FN1_IGFBP7_stable_diag")
    os.makedirs(outdir, exist_ok=True)
    _inject(ME, DATA_DIR=DATA_DIR, SUPPL=SUPPL, BASE_OUT=base, OUTDIR=outdir,
            H5AD_PATH=os.path.join(SUPPL, MEL_H5AD_NAME),
            MEL_H5AD=os.path.join(SUPPL, MEL_H5AD_NAME),
            BC50_H5AD=os.path.join(SUPPL, 'Xtot_naive_resistant_unbalanced_resistant_BC50_clone_size_gt1.h5ad'),
            FOUR_COND_H5AD=os.path.join(SUPPL, 'Xtot_four_conditions_balanced.h5ad'),
            POSTERIOR_SUMMARY_PATH=os.path.join(outdir, "posterior_summary.tsv"),
            POSTERIOR_MU_PATH=os.path.join(outdir, "posterior_mu.npy"),
            SELECTED_GENES_PATH=os.path.join(outdir, "selected_genes.npy"),
            SELECTED_DE_PATH=os.path.join(outdir, "selected_de.tsv"),
            ALL_GENES_DE_PATH=os.path.join(outdir, "all_genes_de.tsv"),
            FDR_ALPHA=0.05, DPI=300, HIGHLIGHT_SIZE=220, LABEL_FONTSIZE=8,
            HIGHLIGHT_FONTSIZE=11, ARROW_LINEWIDTH=1.2)
    ME.fit_analytic_posterior_pipeline()
    ME.plot_cipher_rank_highlight_groups()


def panel_g_kras():
    """G -- KRAS (MIA PaCa-2) resistance-driver ranking. Reuses run_kras_resistance."""
    import src.run_kras_resistance as KR
    outdir = os.path.join(OUTDIR, "kras_analytic_gaussian")
    os.makedirs(outdir, exist_ok=True)
    _inject(KR, DATA_DIR=DATA_DIR, SUPPL=SUPPL, OUTDIR=outdir, _BASE=outdir,
            H5AD_PATH=os.path.join(SUPPL, KRAS_H5AD_NAME))
    KR.run_gaussian_pipeline()
    KR.fig_cipher_rank_green_grey()


def panel_kl_fate():
    """K/L -- LARRY lineage-prediction performance (AUROC / AUPRC). Reuses run_fate_figM7.

    ``day4_lfc_baseline_all_startpops`` is the self-contained function that fits every model
    (CIPHER, terminal-LFC, start-population null, start-population-only) across all start
    populations and populates the module-global ``metrics_df`` that ``plot_auroc_auprc`` reads.
    (The earlier ``cipher_larry_prospective_prediction`` computes a different analysis and never
    sets ``metrics_df``, so pairing it with the plotter raised ``NameError``.)
    """
    import src.run_fate_figM7 as FA
    outdir = os.path.join(OUTDIR, "fate_larry")
    os.makedirs(outdir, exist_ok=True)
    _inject(FA, DATA_DIR=DATA_DIR, SUPPL=SUPPL, OUT_BASE=outdir)
    FA.day4_lfc_baseline_all_startpops()
    FA.plot_auroc_auprc()
