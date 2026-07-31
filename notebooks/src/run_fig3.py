"""Figure 3 -- the forward problem (real vs mean-field covariance).

Engine for the published Fig 3 panels C/D/E (per-dataset accuracy slopegraphs, mean-field ->
real), G (median held-out Pearson), and H/I (estimator convergence vs number of genes used to
fit the amplitude a_m). CIPHER predicts a perturbation mean shift as a rank-1 projection onto the
control-covariance column via cipher.forward_fit / cipher.forward_predict.

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
from cipher.covariance import compute_covariance, meanfield_covariance
from cipher.core import forward_fit, forward_metrics, gene_holdout_masks
from cipher.normalize import normalize_matrix, library_size

# --- config (injected by the notebook) ---
DATA_DIR = None
OUTDIR = None
NORMALIZATION = "raw"            # raw counts, per the paper
EXPRESSION_THRESHOLD = 1.0
MIN_SAMPLES = 100
COV_MAX_CELLS = 10000
SEED = 0
MAX_CONV_PERTS = 200
CONV_HOLDOUT_FRAC = 0.5
K_GRID = np.unique(np.round(np.logspace(np.log10(2), np.log10(2000), 12)).astype(int))
GROUP_COLOR = {"CRISPRi": "#D62728", "CRISPRa": "#1F77B4"}
METRICS = ("r2_uncentered", "pearson", "spearman")

# cross-section state populated by compute_all()
acc_df = None
conv_names = None
k_grid = None
HO = None
PE = None


def _norm(ds, X):
    pc = ds.pflog_pseudocount if NORMALIZATION == "pflog" else None
    return normalize_matrix(X, NORMALIZATION, libsize=library_size(X), pseudocount=pc)


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def dataset_worker(ds, seed=SEED):
    """Per-dataset accuracy (C/D/E) + held-out Pearson (G) + convergence (H/I).

    Two mean-field nulls, each matching its reference panel:
      * C/D/E accuracy use the *analytic* diagonal mean-field ``diag(diag(Sigma))`` --
        no gene-gene covariance, so a perturbation can only move its own target gene.
        (A single empirical shuffle keeps spurious finite-sample off-diagonal noise that
        inflates the mean-field Pearson well above the analytic limit for small controls.)
      * G/H/I held-out use the empirical shuffled-over-cells null ``meanfield_covariance``,
        which stays non-degenerate once the target gene is excluded from the fit/eval.
    """
    control_norm = _norm(ds, ds.control_matrix())
    control_mean = control_norm.mean(0)
    rng = np.random.default_rng(cipher.utils.stable_seed(seed, ds.name))
    cov_in = control_norm
    if COV_MAX_CELLS and control_norm.shape[0] > COV_MAX_CELLS:
        cov_in = control_norm[np.sort(rng.choice(control_norm.shape[0], COV_MAX_CELLS, replace=False))]
    real_cov = compute_covariance(cov_in)
    Sig = {"real": real_cov, "mf": meanfield_covariance(cov_in, seed=seed)}   # G/H/I null
    Sig_acc = {"real": real_cov, "mf": np.diag(np.diag(real_cov))}            # C/D/E null (diagonal)
    ng = real_cov.shape[0]
    perts = [(p, int(g)) for p, g in zip(ds.perturbations, ds.target_gene_indices) if int(g) >= 0]

    acc = {m: {"real": [], "mf": []} for m in METRICS}
    dx_cache = {}
    for p, g in perts:
        dx = _norm(ds, ds.perturbation_matrix(p)).mean(0) - control_mean
        dx_cache[(p, g)] = dx
        for lab in ("real", "mf"):
            a, _ = forward_fit(Sig_acc[lab][:, g], dx)
            met = forward_metrics(dx, (a if np.isfinite(a) else 0.0) * Sig_acc[lab][:, g])
            for m in METRICS:
                acc[m][lab].append(met[m])
    row = {"dataset": ds.name, "group": dataset_group(ds.name), "n_pert": len(perts)}
    for m in METRICS:
        # per-dataset mean over perturbations; the analytic diagonal mean-field is the
        # correct null (no [0,1] display filter -- that would inflate the Spearman panel).
        row[f"{m}_real"] = float(np.nanmean(acc[m]["real"]))
        row[f"{m}_mf"] = float(np.nanmean(acc[m]["mf"]))

    conv_perts = perts[:MAX_CONV_PERTS]
    ho = {lab: np.full((len(conv_perts), len(K_GRID)), np.nan) for lab in ("real", "mf")}
    pe = {lab: np.full((len(conv_perts), len(K_GRID)), np.nan) for lab in ("real", "mf")}
    heldout = {lab: [] for lab in ("real", "mf")}
    for j, (p, g) in enumerate(conv_perts):
        dx = dx_cache[(p, g)]
        train_mask, test_mask = gene_holdout_masks(
            ng, target_idx=g, holdout_frac=CONV_HOLDOUT_FRAC, rng=rng,
            exclude_target_fit=True, exclude_target_eval=True)
        train_idx = np.where(train_mask)[0]
        if train_idx.size < 2 or test_mask.sum() < 2:
            continue
        for lab in ("real", "mf"):
            col = Sig[lab][:, g]; yt = dx[test_mask]
            a_full, _ = forward_fit(col, dx, mask=train_mask)
            heldout[lab].append(_pearson(yt, (a_full if np.isfinite(a_full) else 0.0) * col[test_mask]))
            for ki, K in enumerate(K_GRID):
                sel = train_idx if K >= train_idx.size else rng.choice(train_idx, size=int(K), replace=False)
                m = np.zeros(ng, bool); m[sel] = True
                a, _ = forward_fit(col, dx, mask=m)
                if not np.isfinite(a):
                    continue
                ho[lab][j, ki] = _pearson(yt, a * col[test_mask])
                if np.isfinite(a_full) and abs(a_full) > 1e-12:
                    pe[lab][j, ki] = 100.0 * abs(a - a_full) / abs(a_full)
    for lab in ("real", "mf"):
        row[f"heldout_{lab}"] = float(np.nanmedian(heldout[lab])) if heldout[lab] else np.nan
    conv = {f"ho_{lab}": np.nanmedian(ho[lab], axis=0) for lab in ("real", "mf")}
    conv.update({f"pe_{lab}": np.nanmedian(pe[lab], axis=0) for lab in ("real", "mf")})
    return row, conv


def compute_all():
    """Loop all CRISPRi/a datasets; cache accuracy + convergence under OUTDIR."""
    global acc_df, conv_names, k_grid, HO, PE
    acc_csv = os.path.join(OUTDIR, "fig3_accuracy.csv")
    conv_npz = os.path.join(OUTDIR, "fig3_convergence.npz")
    datasets = sorted((os.path.basename(p)[:-5], p)
                      for p in glob.glob(os.path.join(DATA_DIR, "*.h5ad"))
                      if dataset_group(os.path.basename(p)[:-5]) in ("CRISPRi", "CRISPRa"))
    print(f"{len(datasets)} CRISPRi/a datasets")
    if os.path.exists(acc_csv) and os.path.exists(conv_npz):
        acc_df = pd.read_csv(acc_csv)
        z = np.load(conv_npz, allow_pickle=True)
        conv_names = list(z["names"]); k_grid = z["k_grid"]
        HO = {lab: z[f"ho_{lab}"] for lab in ("real", "mf")}
        PE = {lab: z[f"pe_{lab}"] for lab in ("real", "mf")}
        print("loaded cache:", len(acc_df), "datasets")
        return acc_df
    rows, conv_names = [], []
    HO = {"real": [], "mf": []}; PE = {"real": [], "mf": []}
    for name, path in datasets:
        try:
            ds = load_dataset(path, expression_threshold=EXPRESSION_THRESHOLD, min_samples=MIN_SAMPLES)
            row, conv = dataset_worker(ds)
            rows.append(row); conv_names.append(name)
            for lab in ("real", "mf"):
                HO[lab].append(conv[f"ho_{lab}"]); PE[lab].append(conv[f"pe_{lab}"])
            print(f"{name:40s} n_pert={row['n_pert']:4d} R2 real={row['r2_uncentered_real']:.3f} "
                  f"mf={row['r2_uncentered_mf']:.3f} heldout real={row['heldout_real']:.3f}")
        except Exception as e:  # noqa: BLE001
            print("SKIP", name, repr(e))
    acc_df = pd.DataFrame(rows); k_grid = K_GRID
    HO = {lab: np.array(HO[lab]) for lab in ("real", "mf")}
    PE = {lab: np.array(PE[lab]) for lab in ("real", "mf")}
    acc_df.to_csv(acc_csv, index=False)
    np.savez(conv_npz, names=np.array(conv_names), k_grid=k_grid,
             ho_real=HO["real"], ho_mf=HO["mf"], pe_real=PE["real"], pe_mf=PE["mf"])
    print("cached", len(acc_df), "datasets")
    return acc_df


def _slopegraph(ax, mf, real, groups, letter, ylabel, title, per_modality=False):
    mf = np.asarray(mf, float); real = np.asarray(real, float); groups = np.asarray(groups)
    for xm, xr, grp in zip(mf, real, groups):
        ax.plot([0, 1], [xm, xr], "-", color=GROUP_COLOR.get(grp, "#888"), alpha=0.4, lw=1, zorder=1)
    for grp in ("CRISPRi", "CRISPRa"):
        sel = groups == grp
        ax.scatter(np.zeros(sel.sum()), mf[sel], s=16, color=GROUP_COLOR[grp], zorder=2, edgecolor="k", linewidth=.2)
        ax.scatter(np.ones(sel.sum()), real[sel], s=16, color=GROUP_COLOR[grp], zorder=2, edgecolor="k", linewidth=.2)
    for x, v in ((0, mf), (1, real)):
        ax.plot(x, np.nanmean(v), "D", ms=12, color="k", zorder=3)
        ax.plot(x, np.nanmedian(v), "o", ms=9, mfc="white", mec="k", zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["mean-field Σ", "real/full Σ"])
    ax.set_ylabel(ylabel); ax.set_title(f"{letter}   {title}")
    if per_modality:
        lines = []
        for grp in ("CRISPRi", "CRISPRa"):
            sel = groups == grp
            if sel.any():
                lines.append(f"{grp} mean Δ={np.nanmean(real[sel] - mf[sel]):.2f}")
        ax.text(0.5, 0.02, "\n".join(lines), transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
    else:
        d = real - mf
        ax.text(0.5, 0.02, f"mean Δ={np.nanmean(d):.2f}\nmedian Δ={np.nanmedian(d):.2f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
    ax.set_xlim(-0.3, 1.3)


def panels_cdeg():
    """Panels C/D/E/G -- slopegraphs mean-field -> real/full."""
    grp = acc_df["group"].to_numpy()
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    _slopegraph(ax[0], acc_df["r2_uncentered_mf"], acc_df["r2_uncentered_real"], grp, "C", "ΔR2 (uncentered R2)", "prediction accuracy", per_modality=True)
    _slopegraph(ax[1], acc_df["pearson_mf"], acc_df["pearson_real"], grp, "D", "average Pearson per dataset", "Pearson")
    _slopegraph(ax[2], acc_df["spearman_mf"], acc_df["spearman_real"], grp, "E", "average Spearman per dataset", "Spearman")
    _slopegraph(ax[3], acc_df["heldout_mf"], acc_df["heldout_real"], grp, "G", "median held-out Pearson", "held-out (train/test)")
    for g in ("CRISPRi", "CRISPRa"):
        sel = grp == g
        if sel.any():
            print(f"C {g}: mean ΔR2 = {np.nanmean(acc_df['r2_uncentered_real'][sel] - acc_df['r2_uncentered_mf'][sel]):.3f}")
    print("G medians:  mean-field =", round(float(np.nanmedian(acc_df['heldout_mf'])), 4),
          " real =", round(float(np.nanmedian(acc_df['heldout_real'])), 4))
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig3_panels_CDEG.svg")); plt.show()


def panels_hi():
    """Panels H/I -- convergence vs number of genes used to fit a_m (real vs mean-field)."""
    import matplotlib.ticker as mtick
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for lab, color, name in (("mf", "#F0A000", "mean-field Σ"), ("real", "#2077B4", "real/full Σ")):
        med = np.nanmedian(PE[lab], axis=0)
        lo = np.nanpercentile(PE[lab], 25, axis=0); hi = np.nanpercentile(PE[lab], 75, axis=0)
        ax[0].plot(k_grid, med, "-o", ms=3, color=color, lw=2, label=name)
        ax[0].fill_between(k_grid, lo, hi, color=color, alpha=0.18)
    ax[0].set(xscale="log", xlabel="number of genes used to fit a_m",
              ylabel="median percent error across datasets", title="H   Estimator convergence")
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0)); ax[0].set_ylim(bottom=0)
    ax[0].legend(fontsize=8)
    for lab, color, name in (("mf", "#F0A000", "mean-field Σ"), ("real", "#2077B4", "real/full Σ")):
        med = np.nanmedian(HO[lab], axis=0)
        lo = np.nanpercentile(HO[lab], 25, axis=0); hi = np.nanpercentile(HO[lab], 75, axis=0)
        ax[1].plot(k_grid, med, "-o", ms=3, color=color, lw=2, label=name)
        ax[1].fill_between(k_grid, lo, hi, color=color, alpha=0.18)
    ax[1].set(xscale="log", xlabel="number of genes used to fit a_m",
              ylabel="median held-out Pearson across datasets", title="I   Held-out prediction")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig3_panels_HI.svg")); plt.show()
