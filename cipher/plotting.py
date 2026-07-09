"""Small matplotlib helpers for visualizing CIPHER results.

These functions turn the result objects produced elsewhere in the package
(:class:`cipher.ForwardResult`, :class:`cipher.DriverResult`,
:class:`cipher.ReverseResult`) plus raw forward vectors into publication-ready
figures.

matplotlib is intentionally imported *lazily* inside each function so that
``import cipher`` never requires matplotlib to be installed.  Every helper
follows the same convention:

* if ``ax`` is ``None`` a new figure/axes pair is created, otherwise the
  supplied ``ax`` is drawn onto;
* axes/title labels are always set;
* if ``save`` is not ``None`` the figure is written with
  ``fig.savefig(save, bbox_inches="tight")``;
* the :class:`matplotlib.axes.Axes` is returned (``plt.show`` is never called).
"""
from __future__ import annotations

import numpy as np


def _prepare_axes(ax):
    """Return ``(fig, ax)``, creating a new figure/axes if ``ax`` is None."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    return fig, ax


def _finalize(fig, save):
    """Save the figure if a path was given (never calls ``plt.show``)."""
    if save is not None:
        fig.savefig(save, bbox_inches="tight")


def plot_metric_histograms(result, metric="r2_uncentered", bins=30, ax=None, save=None):
    """Histogram of a forward metric for the real ``Sigma`` versus each null.

    Parameters
    ----------
    result : cipher.ForwardResult
        Has a ``.results`` DataFrame with columns like ``r2_uncentered_real``,
        ``r2_uncentered_meanfield`` and a ``.nulls`` tuple naming the null models.
    metric : str
        Metric prefix to plot — any name in :data:`cipher.core.FORWARD_METRICS`
        (e.g. ``"r2_uncentered"``, ``"pearson"``). The real series is column
        ``f"{metric}_real"`` and each null (when present) is ``f"{metric}_{null}"``.
        Only ``r2_uncentered`` is stored for the nulls.
    bins : int
        Number of histogram bins (shared across all series for comparability).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    fig, ax = _prepare_axes(ax)
    df = result.results

    real_col = f"{metric}_real"
    if real_col not in df.columns:
        raise KeyError(
            f"Column {real_col!r} not found in result.results "
            f"(available: {list(df.columns)})."
        )

    # collect the series that are actually present
    series = [("real", real_col)]
    for null in getattr(result, "nulls", ()) or ():
        col = f"{metric}_{null}"
        if col in df.columns:
            series.append((null, col))

    # shared bin edges over all finite values so the overlays line up
    all_vals = np.concatenate(
        [np.asarray(df[col], dtype=float) for _, col in series]
    )
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        edges = np.linspace(float(all_vals.min()), float(all_vals.max()), bins + 1)
    else:
        edges = bins

    cmap = plt.get_cmap("tab10")
    for i, (label, col) in enumerate(series):
        vals = np.asarray(df[col], dtype=float)
        vals = vals[np.isfinite(vals)]
        ax.hist(
            vals,
            bins=edges,
            histtype="stepfilled",
            alpha=0.45 if label == "real" else 0.35,
            color="C3" if label == "real" else cmap(i % 10),
            edgecolor="black",
            linewidth=0.6,
            label=f"real ({metric})" if label == "real" else f"{label} (null)",
        )

    ax.set_yscale("log")
    ax.set_xlabel(metric)
    ax.set_ylabel("count (log scale)")
    ax.set_title(
        f"{metric} distribution: real vs null "
        f"({getattr(result, 'dataset_name', '')} / "
        f"{getattr(result, 'normalization', '')})".strip(" /")
    )
    ax.legend()
    _finalize(fig, save)
    return ax


def plot_forward_scatter(delta_x, prediction, gene_index=None, ax=None, save=None):
    """Scatter of observed ``delta_x`` vs predicted shift with a ``y=x`` line.

    Parameters
    ----------
    delta_x : array-like
        Observed mean expression shift (length = n_genes).
    prediction : array-like
        CIPHER-predicted shift (same length).
    gene_index : int, optional
        If given, the perturbed gene's point is highlighted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, ax = _prepare_axes(ax)

    delta_x = np.asarray(delta_x, dtype=float).ravel()
    prediction = np.asarray(prediction, dtype=float).ravel()

    ax.scatter(delta_x, prediction, s=8, alpha=0.4, color="C0",
               edgecolors="none", label="genes")

    # y = x reference line spanning the combined finite data range
    finite = np.isfinite(delta_x) & np.isfinite(prediction)
    if np.any(finite):
        lo = float(min(delta_x[finite].min(), prediction[finite].min()))
        hi = float(max(delta_x[finite].max(), prediction[finite].max()))
    else:
        lo, hi = -1.0, 1.0
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--",
            linewidth=1.0, label="y = x")

    if gene_index is not None and 0 <= int(gene_index) < delta_x.size:
        gi = int(gene_index)
        ax.scatter(delta_x[gi], prediction[gi], s=70, color="C3",
                   edgecolors="black", linewidths=0.8, zorder=5,
                   label=f"gene {gi}")

    ax.set_xlabel("observed delta_x")
    ax.set_ylabel("predicted shift")
    ax.set_title("Forward prediction: observed vs predicted")
    ax.legend()
    _finalize(fig, save)
    return ax


def plot_driver_ranking(driver_result, top_n=20, ax=None, save=None):
    """Horizontal bar chart of the top-``top_n`` driver genes.

    Parameters
    ----------
    driver_result : cipher.DriverResult
        Has a ``.ranking`` DataFrame with columns ``gene``, ``driver_score``,
        ``abs_score`` and ``rank`` (already sorted by descending ``abs_score``).
    top_n : int
        Number of top-ranked genes to show.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, ax = _prepare_axes(ax)

    ranking = driver_result.ranking.head(int(top_n))
    genes = np.asarray(ranking["gene"]).astype(str)
    scores = np.asarray(ranking["driver_score"], dtype=float)

    # highest-ranked gene at the top of the chart
    y = np.arange(len(genes))[::-1]
    colors = ["C3" if s < 0 else "C0" for s in scores]
    ax.barh(y, scores, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(genes)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("driver score")
    ax.set_ylabel("gene")
    ax.set_title(
        f"Top {len(genes)} candidate drivers "
        f"({getattr(driver_result, 'name', '')} / "
        f"{getattr(driver_result, 'method', '')})".strip(" /")
    )
    _finalize(fig, save)
    return ax


def plot_reverse_auc(reverse_result, bins=30, ax=None, save=None):
    """Histogram of the per-perturbation ``auc`` column of a reverse run.

    Parameters
    ----------
    reverse_result : cipher.ReverseResult
        Has a ``.results`` DataFrame with an ``auc`` column (one-vs-rest ROC-AUC
        of the true driver per perturbation).
    bins : int
        Number of histogram bins.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, ax = _prepare_axes(ax)

    df = reverse_result.results
    if "auc" not in df.columns:
        raise KeyError(
            f"Column 'auc' not found in reverse_result.results "
            f"(available: {list(df.columns)})."
        )
    vals = np.asarray(df["auc"], dtype=float)
    vals = vals[np.isfinite(vals)]

    ax.hist(vals, bins=bins, range=(0.0, 1.0), color="C0",
            edgecolor="black", linewidth=0.6, alpha=0.8)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0,
               label="chance (0.5)")
    mean_auc = float(np.mean(vals)) if vals.size else float("nan")
    ax.axvline(mean_auc, color="C3", linewidth=1.2,
               label=f"mean = {mean_auc:.3f}")
    ax.set_xlabel("one-vs-rest AUC")
    ax.set_ylabel("count")
    ax.set_title(
        f"Driver-recovery AUC "
        f"({getattr(reverse_result, 'dataset_name', '')} / "
        f"{getattr(reverse_result, 'method', '')})".strip(" /")
    )
    ax.legend()
    _finalize(fig, save)
    return ax


# --------------------------------------------------------------------------- #
# inverse-problem ROC / PR curves (cipher.InverseResult)
# --------------------------------------------------------------------------- #
def _interp_curve(x, y, grid, monotone_end=1.0):
    """Interpolate a staircase curve onto ``grid`` (dedup x by taking max y)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux = np.unique(x)
    uy = np.asarray([np.max(y[x == v]) for v in ux])
    if ux[0] > 0:
        ux = np.r_[0.0, ux]
        uy = np.r_[uy[0] if monotone_end is None else 0.0, uy]
    if ux[-1] < 1:
        ux = np.r_[ux, 1.0]
        uy = np.r_[uy, uy[-1] if monotone_end is None else monotone_end]
    return np.interp(grid, ux, uy)


def plot_inverse_roc(result, ax=None, save=None, color="C0", label=None):
    """Pooled ROC curve of a single :class:`cipher.InverseResult`."""
    fig, ax = _prepare_axes(ax)
    ax.plot([0, 1], [0, 1], "--", lw=1, color="black")
    if result.roc is not None:
        fpr, tpr = result.roc
        auc = result.summary.get("pooled_auc", float("nan"))
        ax.plot(fpr, tpr, lw=2, color=color,
                label=label or f"{result.method}: AUC={auc:.3f}")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{result.dataset_name} | {result.normalization} | inverse ROC")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _finalize(fig, save)
    return ax


def plot_inverse_prc(result, ax=None, save=None, color="C0", label=None):
    """Pooled precision-recall curve of a single :class:`cipher.InverseResult`."""
    fig, ax = _prepare_axes(ax)
    if result.prc is not None:
        precision, recall = result.prc
        ap = result.summary.get("pooled_average_precision", float("nan"))
        ax.plot(recall, precision, lw=2, color=color,
                label=label or f"{result.method}: AP={ap:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"{result.dataset_name} | {result.normalization} | inverse PRC")
    ax.legend(frameon=False, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _finalize(fig, save)
    return ax


def plot_inverse_group(results, curve="roc", ax=None, save=None, color="#d62728",
                       title=None, grid_n=1001, trace_alpha=0.55):
    """Per-dataset traces plus a mean trace for a group of datasets (e.g. CRISPRi).

    ``results`` is an iterable of :class:`cipher.InverseResult` (one per dataset).
    Each dataset's pooled ROC (or PR) curve is drawn thin; their interpolated mean
    is drawn as a thick black line labelled with the mean pooled AUC/AP.  Reproduces
    the reference two-panel figure when called once per group onto side-by-side axes.
    """
    fig, ax = _prepare_axes(ax)
    grid = np.linspace(0.0, 1.0, int(grid_n))
    interps, metrics = [], []
    for res in results:
        curve_obj = res.roc if curve == "roc" else res.prc
        if curve_obj is None:
            continue
        if curve == "roc":
            fpr, tpr = curve_obj
            interps.append(_interp_curve(fpr, tpr, grid))
            metrics.append(res.summary.get("pooled_auc", np.nan))
        else:
            precision, recall = curve_obj
            interps.append(_interp_curve(recall, precision, grid, monotone_end=None))
            metrics.append(res.summary.get("pooled_average_precision", np.nan))
        ax.plot(grid, interps[-1], lw=1.8, alpha=trace_alpha, color=color)

    if curve == "roc":
        ax.plot([0, 1], [0, 1], "--", lw=1, color="black")

    metric_name = "AUC" if curve == "roc" else "AP"
    if interps:
        mean_curve = np.mean(np.vstack(interps), axis=0)
        mean_metric = float(np.nanmean(metrics))
        ax.plot(grid, mean_curve, lw=4, color="black",
                label=f"mean {metric_name}={mean_metric:.3f}")
        ax.legend(frameon=False, loc="lower right" if curve == "roc" else "upper right")

    if curve == "roc":
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
    else:
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
    ax.set_title(title or (f"{metric_name} traces + mean" if title is None else title))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _finalize(fig, save)
    return ax
