"""Figure 5 -- the inverse problem: predicting causal drivers of perturbation response.

Engine for published Fig 5 panels C/D/E/F (per-dataset inverse ROC/PR for CRISPRi/a), H/I
(composite CIPHER linear-response vs log-fold-change margin ROC/PR), and J/K/L/M (sci-Plex drug
identification: candidate log-likelihood, top-k accuracy, ROC, PR). Uses the packaged CIPHER inverse
(cipher.build_model / identify_perturbations / recover_u / posterior_inverse_prediction).

Helpers in notebooks/src (not part of the cipher package). Config constants are module globals the
notebook overrides via R.__dict__.update; DATA_DIR / OUTDIR are injected there.
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cipher
from cipher import (load_dataset, build_model, recover_u, identify_perturbations,
                    identify_lfc, dataset_group, posterior_inverse_prediction,
                    within_group_covariance, select_hvg_dispersion, identification_eigvar)
from cipher.plotting import plot_inverse_group
from cipher.covariance import compute_covariance
from cipher.normalize import normalize_matrix, library_size, mean_var
from cipher import metrics as M

# --- config (injected by the notebook) ---
DATA_DIR = None
OUTDIR = None
NORM = "raw"
SEED = 0
CRISPRI_THRESH = 1.0
CRISPRI_SIGMA_RIDGE = 1e-8
DRUG_HVG = 1000
DRUG_SIGMA_SHRINK = 1e-3
DRUG_WITHIN_SHRINK = 5e-2
# Cache the per-dataset inverse curves behind panels C-F. That sweep (posterior inverse over
# every CRISPRi/a dataset) dominates the notebook's runtime and its results are stable, so it is
# computed once and reloaded on later runs -- iterating on the H/I / J-M panels then costs
# minutes instead of an hour. Delete OUTDIR/inverse_cache (or set False) to force a recompute.
CDEF_CACHE = True

# published colours: CRISPRi red / CRISPRa blue (C-F); CIPHER blue, LFC purple, mean-field grey (K-M)
GROUP_COLOR = {"CRISPRi": "#D62728", "CRISPRa": "#1F77B4"}
METHOD_COLOR = {"lr": "#1F77B4", "lfc": "#7E5FA4", "mf": "#9E9E9E"}

# cross-section state populated by the compute functions
results = None
Sp = p_lr = p_mf = p_lfc = p_SC = None
Sd = md_ = d_lr = d_mf = d_lfc = d_SC = None


def build_split_stats(ds, seed=SEED, min_cells=8, cov_max_cells=10000,
                      hvg_top_n=None, sigma_shrink=0.0, sigma_ridge=0.0, within_cov_shrink=None):
    gene_idx = None
    if hvg_top_n:
        tgt = [int(g) for g in ds.target_gene_indices if int(g) >= 0]
        gene_idx = select_hvg_dispersion(ds.adata.X, hvg_top_n, force_include=tgt)

    def _sub(X):
        return X[:, gene_idx] if gene_idx is not None else X
    pc = ds.pflog_pseudocount if NORM == "pflog" else None

    def _norm(Xsparse):
        Xs = _sub(Xsparse)
        return normalize_matrix(Xs, NORM, libsize=library_size(Xs), pseudocount=pc)

    def _rawmean(Xsparse):
        return np.asarray(_sub(Xsparse).mean(0)).ravel()

    Cn = _norm(ds.control_matrix(dense=False))
    control_mean = Cn.mean(0); control_var = Cn.var(0, ddof=1); n0 = float(Cn.shape[0])
    rng = np.random.default_rng(cipher.utils.stable_seed(seed, ds.name))
    Csub = Cn if Cn.shape[0] <= cov_max_cells else Cn[np.sort(rng.choice(Cn.shape[0], cov_max_cells, replace=False))]
    Sigma = compute_covariance(Csub, ridge_abs=sigma_ridge, shrink=sigma_shrink)
    Cmf = Csub.copy()
    for j in range(Cmf.shape[1]):
        rng.shuffle(Cmf[:, j])
    Sigma_mf = compute_covariance(Cmf, ridge_abs=sigma_ridge, shrink=sigma_shrink)
    logmu0 = np.log1p(np.maximum(_rawmean(ds.control_matrix(dense=False)), 0.0))

    cols = {k: [] for k in ["dx_tr", "dx_te", "var_tr", "var_te", "n_tr", "n_te", "lfc_tr", "lfc_te", "labels"]}
    cov_tr, cov_te = [], []
    for pert in ds.perturbations:
        Xp = ds.perturbation_matrix(pert, dense=False)
        if Xp.shape[0] < min_cells:
            continue
        idx = rng.permutation(Xp.shape[0]); h = Xp.shape[0] // 2
        Ar, Br = Xp[idx[:h]], Xp[idx[h:]]
        A, B = _norm(Ar), _norm(Br)
        ma, va = mean_var(A); mb, vb = mean_var(B)
        cols["dx_tr"].append(ma - control_mean); cols["dx_te"].append(mb - control_mean)
        cols["var_tr"].append(va); cols["var_te"].append(vb)
        cols["n_tr"].append(A.shape[0]); cols["n_te"].append(B.shape[0])
        cols["lfc_tr"].append(np.log1p(np.maximum(_rawmean(Ar), 0.0)) - logmu0)
        cols["lfc_te"].append(np.log1p(np.maximum(_rawmean(Br), 0.0)) - logmu0)
        cols["labels"].append(str(pert))
        if within_cov_shrink is not None:
            cov_tr.append(within_group_covariance(A, within_cov_shrink))
            cov_te.append(within_group_covariance(B, within_cov_shrink))
    def arr(L):
        return np.asarray(L, dtype=np.float64)
    out = dict(Sigma=Sigma, Sigma_mf=Sigma_mf, control_var=control_var, n0=n0,
               dx_train=arr(cols["dx_tr"]), dx_test=arr(cols["dx_te"]),
               var_train=arr(cols["var_tr"]), var_test=arr(cols["var_te"]),
               n_train=arr(cols["n_tr"]), n_test=arr(cols["n_te"]),
               lfc_train=arr(cols["lfc_tr"]), lfc_test=arr(cols["lfc_te"]),
               labels=np.asarray(cols["labels"]))
    if within_cov_shrink is not None:
        out["cov_train"] = np.stack(cov_tr); out["cov_test"] = np.stack(cov_te)
    return out


def _lr_scores(model, S, pev, tau2=1.0):
    """(n_queries, n_candidates) likelihood scores for every candidate.

    ``cipher.identify_perturbations`` keeps only the per-query rank/margin, but the published
    composite panels (H/I, L/M) need each candidate's score. This rebuilds the same quantity
    from the packaged primitives (``recover_u`` plus the model's eigenbasis) -- it is exactly
    the expression the panel-J drug ranking already uses, evaluated for every query rather
    than one -- so nothing has to change inside the installed package.
    """
    V, lam = model.V, model.eigenvalues
    pred_eig = (recover_u(model, S["dx_train"], tau2) @ V) * lam[None, :]
    nu = np.asarray(S["n_test"], dtype=np.float64).reshape(-1)
    h_test = np.maximum(lam[None, :] / model.n0 + np.asarray(pev, dtype=np.float64) / nu[:, None],
                        model.ridge)
    z = np.asarray(S["dx_test"], dtype=np.float64) @ V
    out = np.empty((z.shape[0], pred_eig.shape[0]), dtype=np.float64)
    for i in range(z.shape[0]):
        diff = pred_eig - z[i][None, :]
        out[i] = -0.5 * np.sum(diff * diff / h_test[i][None, :], axis=1)
    return out


def _lfc_scores(S):
    """(n_queries, n_candidates) cosine scores, mirroring ``cipher.identify_lfc``."""
    def _unit(Mx):
        Mx = np.asarray(Mx, dtype=np.float64)
        return Mx / np.maximum(np.linalg.norm(Mx, axis=1, keepdims=True), 1e-12)
    return _unit(S["lfc_test"]) @ _unit(S["lfc_train"]).T


def _margin_scores(r):
    """AUROC / AP of the published identification panels (H/I, L/M).

    Matches the supplementary drug-identification engine (``run_figS11``): the label is
    whether the top-1 call was correct and the score is that query's top1-minus-top2 margin,
    i.e. how well the model's confidence separates its hits from its misses. This is what the
    packaged ``IdentificationResult.roc``/``.prc`` already carry.
    """
    c = (r.rank <= 1).astype(int)
    m = np.asarray(r.margin, float)
    fin = np.isfinite(m)
    if c[fin].sum() in (0, fin.sum()):
        return np.nan, np.nan
    return M.roc_auc(c[fin], m[fin]), M.average_precision(c[fin], m[fin])


def _pooled_from_scores(scores):
    """Composite one-vs-rest AUROC / AP / curves over every (query, candidate) pair.

    Each query contributes its true candidate as the single positive and the rest as
    negatives. ``identify_*`` is called without ``true_index``, so query i's true candidate
    is candidate i.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.zeros(s.shape, dtype=np.int64)
    n = min(s.shape)
    y[np.arange(n), np.arange(n)] = 1
    # Standardise within each query before pooling. Each query's log-likelihoods sit on their
    # own scale (they depend on that query's dx magnitude and sampling variance), so pooling
    # them raw lets one query's worst candidate outrank another query's true candidate and the
    # composite curve collapses to chance. Per-query z-scores keep every query's ranking intact
    # while putting them on a common axis.
    mu = np.nanmean(s, axis=1, keepdims=True)
    sd = np.nanstd(s, axis=1, keepdims=True)
    s = (s - mu) / np.where(sd > 0, sd, 1.0)
    fs, fy = s.ravel(), y.ravel()
    fin = np.isfinite(fs)
    fs, fy = fs[fin], fy[fin]
    if fs.size < 2 or not (0 < fy.sum() < fy.size):
        return np.nan, np.nan, None, None
    return (M.roc_auc(fy, fs), M.average_precision(fy, fs),
            M.roc_curve(fy, fs), M.pr_curve(fy, fs))


def run_identify(S, name, h_mode="sigma_count"):
    K = tuple(range(1, 11)); n0 = S["n0"]; nq = len(S["labels"])
    if h_mode == "within_cov":
        m = build_model(S["Sigma"], None, n0, S["n_train"], h_mode="within_cov", pert_cov=S["cov_train"])
        mf = build_model(S["Sigma_mf"], None, n0, S["n_train"], h_mode="within_cov", pert_cov=S["cov_train"])
        pev = identification_eigvar(m, "within_cov", cov=S["cov_test"])
        pev_mf = identification_eigvar(mf, "within_cov", cov=S["cov_test"])
    elif h_mode == "sigma_count":
        m = build_model(S["Sigma"], None, n0, S["n_train"], h_mode="sigma_count")
        mf = build_model(S["Sigma_mf"], None, n0, S["n_train"], h_mode="sigma_count")
        pev = identification_eigvar(m, "sigma_count", n_query=nq)
        pev_mf = identification_eigvar(mf, "sigma_count", n_query=nq)
    else:
        m = build_model(S["Sigma"], S["var_train"], n0, S["n_train"], control_var=S["control_var"])
        mf = build_model(S["Sigma_mf"], S["var_train"], n0, S["n_train"], control_var=S["control_var"])
        pev, pev_mf = S["var_test"] @ m.V2, S["var_test"] @ mf.V2
    r_lr = identify_perturbations(m, S["dx_train"], S["dx_test"], tau2=1.0, labels=S["labels"],
                                  pert_eigvar_test=pev, nu_test=S["n_test"], topk=K, dataset_name=name)
    r_mf = identify_perturbations(mf, S["dx_train"], S["dx_test"], tau2=1.0, labels=S["labels"],
                                  pert_eigvar_test=pev_mf, nu_test=S["n_test"], topk=K, dataset_name=name + "_MF")
    r_lfc = identify_lfc(S["lfc_train"], S["lfc_test"], labels=S["labels"], topk=K, dataset_name=name + "_LFC")
    # Candidate score matrices for the composite (pooled one-vs-rest) panels; the packaged
    # results carry only per-query rank/margin, so these are rebuilt here from the same inputs.
    SC = {"lr": _lr_scores(m, S, pev), "mf": _lr_scores(mf, S, pev_mf), "lfc": _lfc_scores(S)}
    return m, r_lr, r_mf, r_lfc, SC


def _plot_identify(r_lr, r_mf, r_lfc, SC, title):
    """K/L/M -- top-k accuracy plus the composite one-vs-rest ROC / PR across all queries.

    Published colours: CIPHER linear response (blue), log fold change (purple), mean-field
    linear response (grey, top-k panel only, as in the paper).
    """
    ks = list(range(1, 11)); fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    series = [(r_lr, "lr", "linear response"),
              (r_mf, "mf", "linear response (mean-field)"),
              (r_lfc, "lfc", "log fold change")]
    for r, key, lab in series:
        ax[0].plot(ks, [r.topk_acc[k] for k in ks], marker="o", color=METHOD_COLOR[key], label=lab)
    for r, key, lab in series:
        if key == "mf":
            continue                      # L/M show CIPHER vs LFC only, matching the paper
        au, ap = _margin_scores(r)
        if r.roc:
            ax[1].plot(*r.roc, color=METHOD_COLOR[key], label=f"{lab} {au:.3f}")
        if r.prc:
            pr, rc = r.prc; ax[2].plot(rc, pr, color=METHOD_COLOR[key], label=f"{lab} {ap:.3f}")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(xlabel="k", ylabel="top-k accuracy", title=f"{title}: top-k")
    ax[1].set(xlabel="false positive rate", ylabel="true positive rate", title="AUROC")
    ax[2].set(xlabel="recall", ylabel="precision", title="AUPRC")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    return fig


class _CachedInverse:
    """Minimal stand-in for cipher.InverseResult holding only what plot_inverse_group reads."""
    __slots__ = ("roc", "prc", "summary")

    def __init__(self, roc, prc, summary):
        self.roc, self.prc, self.summary = roc, prc, summary


def _cdef_cache_path(name):
    tag = f"{NORM}_t{str(CRISPRI_THRESH).replace('.', 'p')}"
    return os.path.join(OUTDIR, "inverse_cache", f"{name}__{tag}.npz")


def _save_inverse(name, res):
    path = _cdef_cache_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = {"auc": float(res.summary.get("pooled_auc", np.nan)),
         "ap": float(res.summary.get("pooled_average_precision", np.nan))}
    if res.roc is not None:
        d["roc0"], d["roc1"] = np.asarray(res.roc[0]), np.asarray(res.roc[1])
    if res.prc is not None:
        d["prc0"], d["prc1"] = np.asarray(res.prc[0]), np.asarray(res.prc[1])
    np.savez_compressed(path, **d)


def _load_inverse(name):
    path = _cdef_cache_path(name)
    if not os.path.exists(path):
        return None
    z = np.load(path)
    roc = (z["roc0"], z["roc1"]) if "roc0" in z.files else None
    prc = (z["prc0"], z["prc1"]) if "prc0" in z.files else None
    return _CachedInverse(roc, prc, {"pooled_auc": float(z["auc"]),
                                     "pooled_average_precision": float(z["ap"])})


def panels_cdef():
    """C/D/E/F -- per-dataset inverse ROC and PR for CRISPRi and CRISPRa."""
    global results
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5ad")))
    grp = {"CRISPRi": [], "CRISPRa": []}
    for f in files:
        g = dataset_group(os.path.basename(f)[:-5])
        if g in grp:
            grp[g].append(f)
    results = {"CRISPRi": [], "CRISPRa": []}
    for g, paths in grp.items():
        for p in paths:
            try:
                name = os.path.basename(p)[:-5]
                cached = _load_inverse(name) if CDEF_CACHE else None
                if cached is not None:
                    results[g].append(cached)
                    print(f"{g:8s} {name:42s} [cached] pooled_auc={cached.summary['pooled_auc']:.3f} "
                          f"pooled_AP={cached.summary['pooled_average_precision']:.3f}")
                    continue
                # threshold 1.0 == the reference "mean_control_ge_1p0" gene filter; a lower
                # cut keeps near-zero genes that only add ranking noise to the inverse.
                res = posterior_inverse_prediction(p, normalization=NORM, method="posterior",
                                                   progress=False, expression_threshold=CRISPRI_THRESH,
                                                   min_samples=100)
                if CDEF_CACHE:
                    _save_inverse(name, res)
                results[g].append(res)
                print(f"{g:8s} {os.path.basename(p):42s} pooled_auc={res.summary.get('pooled_auc'):.3f} "
                      f"pooled_AP={res.summary.get('pooled_average_precision'):.3f}")
            except Exception as e:  # noqa: BLE001
                print("SKIP", os.path.basename(p), repr(e))
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for j, g in enumerate(("CRISPRi", "CRISPRa")):
        if results[g]:
            plot_inverse_group(results[g], curve="roc", ax=ax[0, j], title=f"{g} | inverse ROC",
                               color=GROUP_COLOR[g])
            plot_inverse_group(results[g], curve="pr", ax=ax[1, j], title=f"{g} | inverse PR",
                               color=GROUP_COLOR[g])
        else:
            for r in (0, 1):
                ax[r, j].set_title(f"{g} | no datasets"); ax[r, j].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig5_panels_CDEF.svg")); plt.show()


def run_crispri():
    """CRISPRi perturbation identification (Tian2021 CRISPRi) -- CIPHER vs mean-field vs LFC."""
    global Sp, p_lr, p_mf, p_lfc, p_SC
    ds_p = load_dataset(os.path.join(DATA_DIR, "TianKampmann2021_CRISPRi.h5ad"),
                        expression_threshold=CRISPRI_THRESH, min_samples=100)
    Sp = build_split_stats(ds_p, sigma_ridge=CRISPRI_SIGMA_RIDGE)
    _mp, p_lr, p_mf, p_lfc, p_SC = run_identify(Sp, ds_p.name, h_mode="sigma_count")
    print(f"{len(Sp['labels'])} candidate perturbations (chance acc1={1/len(Sp['labels']):.4f})")
    for r, lab in [(p_lr, "CIPHER"), (p_mf, "mean-fd"), (p_lfc, "LFC")]:
        print(f"  {lab:8s} acc1={r.acc1:.4f}  top5={r.topk_acc[5]:.4f}")


def plot_crispri():
    """H (CRISPRi/a) -- top-k / margin ROC / margin PR for CIPHER vs LFC."""
    fig = _plot_identify(p_lr, p_mf, p_lfc, p_SC, "CRISPRi perturbation ID")
    fig.savefig(os.path.join(OUTDIR, "fig5_crispri_identify.svg")); plt.show()


def run_sciplex():
    """sci-Plex drug identification (full-gene SrivatsanTrapnell2020) -- within-drug covariance H."""
    global Sd, md_, d_lr, d_mf, d_lfc, d_SC
    ds_d = load_dataset(os.path.join(DATA_DIR, "SrivatsanTrapnell2020_sciplex3.h5ad"),
                        pert_key="perturbation", require_target_in_var=False,
                        expression_threshold=0.0, min_samples=100)
    Sd = build_split_stats(ds_d, hvg_top_n=DRUG_HVG, sigma_shrink=DRUG_SIGMA_SHRINK,
                           within_cov_shrink=DRUG_WITHIN_SHRINK)
    md_, d_lr, d_mf, d_lfc, d_SC = run_identify(Sd, ds_d.name, h_mode="within_cov")
    print(f"{len(Sd['labels'])} candidate drugs (chance acc1={1/len(Sd['labels']):.4f})")
    for r, lab in [(d_lr, "CIPHER"), (d_mf, "mean-fd"), (d_lfc, "LFC")]:
        print(f"  {lab:8s} acc1={r.acc1:.4f}  top5={r.topk_acc[5]:.4f}")


def plot_sciplex(query="Fulvestrant"):
    """K/L/M -- sci-Plex top-k / ROC / PR (CIPHER vs LFC); J -- candidate drug log-likelihood."""
    fig = _plot_identify(d_lr, d_mf, d_lfc, d_SC, "sci-Plex drug ID")
    fig.savefig(os.path.join(OUTDIR, "fig5_sciplex_identify.svg")); plt.show()
    q = query if query in set(Sd["labels"]) else str(Sd["labels"][0])
    u_hat = recover_u(md_, Sd["dx_train"], 1.0)
    V, lam = md_.V, md_.eigenvalues
    pred_eig = (u_hat @ V) * lam[None, :]
    h_test = np.maximum(lam[None, :] / md_.n0
                        + identification_eigvar(md_, "within_cov", cov=Sd["cov_test"]) / Sd["n_test"][:, None],
                        md_.ridge)
    z = Sd["dx_test"] @ V
    qi = int(np.where(Sd["labels"] == q)[0][0])
    diff = pred_eig - z[qi][None, :]
    ll = -0.5 * np.sum(diff * diff / h_test[qi][None, :], axis=1)
    order = np.argsort(-ll)[:15]; lab = [str(Sd["labels"][i]) for i in order]
    # Published panel J plots the log-likelihood *ratio* against the worst shown candidate, so
    # bars run rightward from 0 and the true drug is the longest (raw LL is large-negative).
    lr = ll[order] - float(np.min(ll[order]))
    colors = [METHOD_COLOR["lr"] if Sd["labels"][i] == q else "0.6" for i in order]
    fig, ax = plt.subplots(figsize=(6, 5)); ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, lr, color=colors); ax.set_yticks(ypos); ax.set_yticklabels(lab, fontsize=8)
    ax.set(xlabel="log-likelihood (LR)", title=f"J   candidate drugs for query = {q}")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig5_candidate_ll.svg")); plt.show()


def panels_hi_composite():
    """H/I + L/M values -- composite one-vs-rest AUROC/AUPRC for CIPHER vs mean-field vs LFC."""
    rows = []
    for grp, (rl, rm, rf) in {"CRISPRi/a": (p_lr, p_mf, p_lfc), "sci-Plex": (d_lr, d_mf, d_lfc)}.items():
        for meth, r in [("linear response", rl), ("linear response (mean-field)", rm),
                        ("log fold change", rf)]:
            au, ap = _margin_scores(r)
            rows.append(dict(group=grp, method=meth, AUROC=au, AUPRC=ap))
    comp = pd.DataFrame(rows); print(comp.to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, metric in zip(ax, ["AUROC", "AUPRC"]):
        comp.pivot(index="group", columns="method", values=metric).plot(kind="bar", ax=a, rot=0)
        a.set(ylabel=metric, title=metric); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig5_method_bars.svg")); plt.show()
    return comp
