"""Fig S18 co-expression run module (notebook-only; NOT shipped with the ``cipher`` package).

Per-notebook orchestration for ``notebooks/suppl/figS18_coexpr.ipynb``. Each function
is one figure section/panel: same variables, same plt/savefig calls, same logic. Config
values (SUPPL, OUTDIR, USE_ABS, Z, DATA_DIR, ...) are read as MODULE GLOBALS; the driving
notebook injects its UPPER-case config into this module's namespace at call time (see the
notebook's injection cell). The shared co-expression engine is pulled in via
``from src.suppl_coexpr import *``.
"""
from src.suppl_coexpr import *

# same library imports the cluster module (src.suppl_coexpr) uses
import os, re, glob, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Patch
from matplotlib import gridspec
from scipy.sparse import issparse, csr_matrix
from scipy.stats import wilcoxon, ttest_rel, ks_2samp, mannwhitneyu, pearsonr, spearmanr
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
except Exception:
    roc_auc_score = average_precision_score = roc_curve = precision_recall_curve = None
try:
    import scanpy as sc
except Exception:
    sc = None
try:
    import anndata as ad
except Exception:
    ad = None
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *a, **k): return x


def ggi_engine_run():
    DATA_ROOT = os.environ["CIPHER_DATA_DIR"]
    OUTDIR = os.path.join(os.environ.get("SUPPL_OUT", "resources/repro/figS18"), "ggi")
    os.makedirs(OUTDIR, exist_ok=True)
    PPI_CSV = os.path.join(SUPPL, "Human_genetic_interactions_collapsed.csv")
    ppi_df, ppi_pairs_unique = load_ppi_unique_pairs(PPI_CSV)
    print(f"[PPI] rows in file: {len(ppi_df):,}")
    print(f"[PPI] unique undirected interactions: {len(ppi_pairs_unique):,}")
    print("[PPI] example rows:")
    print(ppi_df[["Interactor A", "Interactor B"]].head(5).to_string(index=False))
    datapaths = [
            "ReplogleWeissman2022_rpe1.h5ad",
            "ReplogleWeissman2022_K562_essential.h5ad",
            "GSE264667_jurkat_raw_singlecell_01.h5ad",
            "GSE264667_hepg2_raw_singlecell_01.h5ad",
            "NormanWeissman2019_filtered.h5ad",
            "FrangiehIzar2021_RNA.h5ad",
            "TianKampmann2019_day7neuron.h5ad",
            "TianKampmann2021_CRISPRi.h5ad",
            "TianKampmann2021_CRISPRa.h5ad",
            "TianKampmann2019_iPSC.h5ad",
        ]
    datapaths = [os.path.join(DATA_ROOT, f) for f in datapaths]
    preprocess_configs = [
            {"preprocess_name": "raw",   "log1p": False, "norm": False},
            {"preprocess_name": "log1p", "log1p": True,  "norm": False},
            {"preprocess_name": "norm",  "log1p": False, "norm": True},
            {"preprocess_name": "norm_plus_log1p",  "log1p": True, "norm": True},
        ]
    expression_threshold = 0.01
    min_samples = 2
    use_abs = False
    rng_seed = 0
    plot_bins = 80
    chunk_size = 20000
    summaries, errors = [], []
    for cfg in preprocess_configs:
            preprocess_name = cfg["preprocess_name"]
            log1p = cfg["log1p"]
            norm = cfg["norm"]

            print("\n" + "=" * 100)
            print(f"PREPROCESSING MODE: {preprocess_name} | log1p={log1p} | norm={norm}")
            print("=" * 100)

            outdir_cfg = os.path.join(OUTDIR, preprocess_name)
            os.makedirs(outdir_cfg, exist_ok=True)

            for p in datapaths:
                if not os.path.exists(p):
                    print(f"[SKIP] missing: {p}")
                    errors.append({
                        "dataset": os.path.basename(p),
                        "preprocess": preprocess_name,
                        "log1p": log1p,
                        "norm": norm,
                        "error": "file not found",
                    })
                    continue

                try:
                    s = analyze_dataset_ppi_vs_random_sparse(
                        data_path=p,
                        ppi_pairs_unique=ppi_pairs_unique,
                        expression_threshold=expression_threshold,
                        min_samples=min_samples,
                        use_abs=use_abs,
                        save_dir=outdir_cfg,
                        rng_seed=rng_seed,
                        plot_bins=plot_bins,
                        chunk_size=chunk_size,
                        show_plots=True,
                        log1p=log1p,
                        norm=norm,
                        preprocess_name=preprocess_name,
                    )
                    summaries.append(s)
                    pd.DataFrame([s]).to_csv(
                        os.path.join(outdir_cfg, f"{s['dataset']}__{preprocess_name}__summary.csv"),
                        index=False
                    )
                except Exception as e:
                    print(f"[ERROR] {os.path.basename(p)} | {preprocess_name}: {e}")
                    errors.append({
                        "dataset": os.path.basename(p),
                        "preprocess": preprocess_name,
                        "log1p": log1p,
                        "norm": norm,
                        "error": str(e),
                    })
    pd.DataFrame(summaries).to_csv(os.path.join(OUTDIR, "ALL_DATASETS_ALL_PREPROCESS__summary.csv"), index=False)
    if errors:
            pd.DataFrame(errors).to_csv(os.path.join(OUTDIR, "ALL_DATASETS_ALL_PREPROCESS__errors.csv"), index=False)
    print("\nDone.")
    print(f"  summaries: {os.path.join(OUTDIR, 'ALL_DATASETS_ALL_PREPROCESS__summary.csv')}")
    if errors:
            print(f"  errors:    {os.path.join(OUTDIR, 'ALL_DATASETS_ALL_PREPROCESS__errors.csv')}")


def ggi_gene_survival_panels():
    DATASETS = [
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]
    CONDITIONS = [
        {
            "name": "Raw",
            "short": "raw",
            "aliases": ["raw"],
        },
        {
            "name": "log1p",
            "short": "log1p",
            "aliases": ["log1p"],
        },
        {
            "name": "Normalized",
            "short": "norm",
            "aliases": ["norm", "normalized"],
        },
        {
            # compatible with your upstream typo/key
            "name": "Normalized + log1p",
            "short": "norm_plust_log1p",
            "aliases": [
                "norm_plust_log1p",
                "normpluslog1p",
                "norm_plus_log1p",
                "norm_log1p",
                "normlog1p",
                "normalized_log1p",
                "normalized_plus_log1p",
            ],
        },
    ]
    BASE_OUTPUT_DIR = os.path.join(os.environ.get("SUPPL_OUT", "resources/repro/figS18"), "ggi")
    T_THRESHOLD = 0.5
    PLOT_FLOOR = 1e-2
    FILTER_FOR_PLOTTING = True
    BINS = 60
    N_CONDITIONS = len(CONDITIONS)
    OUTDIR = f"{N_CONDITIONS}_panel_gene_survival_from_ggi_pairs_t{T_THRESHOLD}_linear_axes"
    os.makedirs(OUTDIR, exist_ok=True)
    compiled_by_condition = {}
    summary_rows = []
    for cond in CONDITIONS:
        print("\n" + "=" * 100)
        print(f"Loading condition: {cond['name']}")
        print(f"Condition key:     {cond['short']}")
        print(f"Aliases:           {cond.get('aliases', [cond['short']])}")
        print(f"Base output dir:   {BASE_OUTPUT_DIR}")
        print("=" * 100)

        pair_long_all, gene_df, summary = load_and_compile_condition(
            base_output_dir=BASE_OUTPUT_DIR,
            condition_name=cond["name"],
            condition_short=cond["short"],
            condition_aliases=cond.get("aliases", [cond["short"]]),
            datasets=DATASETS,
            threshold=T_THRESHOLD,
        )

        compiled_by_condition[cond["short"]] = {
            "name": cond["name"],
            "pair_long_all": pair_long_all,
            "gene_df": gene_df,
            "summary": summary,
        }

        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTDIR, f"{N_CONDITIONS}_condition_summary_t{T_THRESHOLD}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("\nSummary across conditions:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {summary_csv}")
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    xedges = np.linspace(xmin, xmax, BINS + 1)
    yedges = np.linspace(ymin, ymax, BINS + 1)
    panel_data = []
    global_max_count = 1
    for cond in CONDITIONS:
        key = cond["short"]
        gene_df = compiled_by_condition[key]["gene_df"].copy()

        if FILTER_FOR_PLOTTING:
            gene_df = gene_df[
                (gene_df["ggi_survival"] > PLOT_FLOOR)
                | (gene_df["random_survival"] > PLOT_FLOOR)
            ].copy()

        x = gene_df["random_survival"].to_numpy()
        y = gene_df["ggi_survival"].to_numpy()

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        # Fraction above/below y=x among genes with x or y > PLOT_FLOOR
        line_mask = (x > PLOT_FLOOR) | (y > PLOT_FLOOR)
        n_line = int(np.sum(line_mask))
        n_above = int(np.sum((y > x) & line_mask))
        n_below = int(np.sum((y < x) & line_mask))
        n_equal = int(np.sum((y == x) & line_mask))

        frac_above_line = n_above / n_line if n_line > 0 else np.nan
        frac_below_line = n_below / n_line if n_line > 0 else np.nan
        frac_equal_line = n_equal / n_line if n_line > 0 else np.nan

        x_plot = x.copy()
        y_plot = y.copy()

        counts, _, _ = np.histogram2d(
            x_plot,
            y_plot,
            bins=[xedges, yedges],
        )

        global_max_count = max(global_max_count, counts.max())

        panel_data.append({
            "key": key,
            "display_name": cond["name"],
            "df": gene_df,
            "x": x,
            "y": y,
            "x_plot": x_plot,
            "y_plot": y_plot,
            "counts": counts,
            "n_line": n_line,
            "n_above": n_above,
            "n_below": n_below,
            "n_equal": n_equal,
            "frac_above_line": frac_above_line,
            "frac_below_line": frac_below_line,
            "frac_equal_line": frac_equal_line,
        })
    fig, axes = plt.subplots(
        1,
        N_CONDITIONS,
        figsize=(6.2 * N_CONDITIONS, 6.2),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    mappable = None
    for ax, pdata in zip(axes, panel_data):
        counts_masked = np.ma.masked_where(
            pdata["counts"].T == 0,
            pdata["counts"].T,
        )

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            counts_masked,
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=max(1, global_max_count)),
            shading="auto",
        )
        mappable = mesh

        ax.plot(
            [xmin, xmax],
            [ymin, ymax],
            linestyle="--",
            linewidth=1.3,
            color="white",
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_title(pdata["display_name"], fontsize=16)
        ax.tick_params(axis="both", labelsize=12)

        mean_x = np.mean(pdata["x"]) if len(pdata["x"]) > 0 else np.nan
        mean_y = np.mean(pdata["y"]) if len(pdata["y"]) > 0 else np.nan
        mean_d = np.mean(pdata["y"] - pdata["x"]) if len(pdata["x"]) > 0 else np.nan
        frac_above = np.mean(pdata["y"] > pdata["x"]) if len(pdata["x"]) > 0 else np.nan
        frac_below = np.mean(pdata["y"] < pdata["x"]) if len(pdata["x"]) > 0 else np.nan

        ax.text(
            0.03,
            0.97,
            (
                f"Genes shown: {len(pdata['x']):,}\n"
                f"mean GGI: {mean_y:.3g}\n"
                f"mean random: {mean_x:.3g}\n"
                f"mean Δ: {mean_d:.3g}\n"
                f"frac above x=y: {frac_above:.3g}\n"
                f"frac below x=y: {frac_below:.3g}\n"
                f"among x or y > {PLOT_FLOOR:g}:\n"
                f"  above: {pdata['frac_above_line']:.3g} ({pdata['n_above']:,}/{pdata['n_line']:,})\n"
                f"  below: {pdata['frac_below_line']:.3g} ({pdata['n_below']:,}/{pdata['n_line']:,})"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.2,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )
    axes[0].set_ylabel(f"GGI survival: P(r > {T_THRESHOLD})", fontsize=15)
    for ax in axes:
        ax.set_xlabel(f"Random survival: P(r > {T_THRESHOLD})", fontsize=15)
    fig.suptitle(
        f"Gene-level GGI survival enrichment across preprocessing choices\nthreshold: r > {T_THRESHOLD}",
        fontsize=19,
        y=1.04,
    )
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Number of genes per bin, log color scale", fontsize=14)
    plt.tight_layout()
    suffix = f"t{T_THRESHOLD}_floor{PLOT_FLOOR}_linear_axes".replace(".", "p")
    if FILTER_FOR_PLOTTING:
        suffix += "_filtered"
    png_out = os.path.join(
        OUTDIR,
        f"{N_CONDITIONS}_panel_gene_survival_density_{suffix}.png",
    )
    svg_out = os.path.join(
        OUTDIR,
        f"{N_CONDITIONS}_panel_gene_survival_density_{suffix}.svg",
    )
    pdf_out = os.path.join(
        OUTDIR,
        f"{N_CONDITIONS}_panel_gene_survival_density_{suffix}.pdf",
    )
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.savefig(svg_out, bbox_inches="tight")
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.show()
    print(f"\nSaved {N_CONDITIONS}-panel PNG: {png_out}")
    print(f"Saved {N_CONDITIONS}-panel SVG: {svg_out}")
    print(f"Saved {N_CONDITIONS}-panel PDF: {pdf_out}")
    fig, axes = plt.subplots(
        1,
        N_CONDITIONS,
        figsize=(6.2 * N_CONDITIONS, 6.2),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    mappable = None
    for ax, pdata in zip(axes, panel_data):
        counts_masked = np.ma.masked_where(
            pdata["counts"].T == 0,
            pdata["counts"].T,
        )

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            counts_masked,
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=max(1, global_max_count)),
            shading="auto",
        )
        mappable = mesh

        ax.scatter(
            pdata["x_plot"],
            pdata["y_plot"],
            s=5,
            alpha=0.12,
            edgecolors="none",
            color="white",
        )

        ax.plot(
            [xmin, xmax],
            [ymin, ymax],
            linestyle="--",
            linewidth=1.3,
            color="red",
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_title(pdata["display_name"], fontsize=16)
        ax.tick_params(axis="both", labelsize=12)

        mean_x = np.mean(pdata["x"]) if len(pdata["x"]) > 0 else np.nan
        mean_y = np.mean(pdata["y"]) if len(pdata["y"]) > 0 else np.nan
        mean_d = np.mean(pdata["y"] - pdata["x"]) if len(pdata["x"]) > 0 else np.nan

        ax.text(
            0.03,
            0.97,
            (
                f"Genes shown: {len(pdata['x']):,}\n"
                f"mean GGI: {mean_y:.3g}\n"
                f"mean random: {mean_x:.3g}\n"
                f"mean Δ: {mean_d:.3g}\n"
                f"among x or y > {PLOT_FLOOR:g}:\n"
                f"  above: {pdata['frac_above_line']:.3g} ({pdata['n_above']:,}/{pdata['n_line']:,})\n"
                f"  below: {pdata['frac_below_line']:.3g} ({pdata['n_below']:,}/{pdata['n_line']:,})"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.2,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )
    axes[0].set_ylabel(f"GGI survival: P(r > {T_THRESHOLD})", fontsize=15)
    for ax in axes:
        ax.set_xlabel(f"Random survival: P(r > {T_THRESHOLD})", fontsize=15)
    fig.suptitle(
        f"Gene-level GGI survival enrichment across preprocessing choices\nwith point overlay, threshold: r > {T_THRESHOLD}",
        fontsize=19,
        y=1.04,
    )
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Number of genes per bin, log color scale", fontsize=14)
    plt.tight_layout()
    png_out2 = os.path.join(
        OUTDIR,
        f"{N_CONDITIONS}_panel_gene_survival_density_with_points_{suffix}.png",
    )
    svg_out2 = os.path.join(
        OUTDIR,
        f"{N_CONDITIONS}_panel_gene_survival_density_with_points_{suffix}.svg",
    )
    pdf_out2 = os.path.join(
        OUTDIR,
        f"{N_CONDITIONS}_panel_gene_survival_density_with_points_{suffix}.pdf",
    )
    plt.savefig(png_out2, dpi=300, bbox_inches="tight")
    plt.savefig(svg_out2, bbox_inches="tight")
    plt.savefig(pdf_out2, bbox_inches="tight")
    plt.show()
    print(f"\nSaved {N_CONDITIONS}-panel overlay PNG: {png_out2}")
    print(f"Saved {N_CONDITIONS}-panel overlay SVG: {svg_out2}")
    print(f"Saved {N_CONDITIONS}-panel overlay PDF: {pdf_out2}")


def ggi_logratio_curves():
    BASE_OUTPUT_DIR = os.path.join(os.environ.get("SUPPL_OUT", "resources/repro/figS18"), "ggi")
    CONDITIONS = [
        {
            "key": "raw",
            "label": "Raw",
            "aliases": ["raw"],
        },
        {
            "key": "log1p",
            "label": "log1p",
            "aliases": ["log1p"],
        },
        {
            "key": "norm",
            "label": "Normalized",
            "aliases": ["norm", "normalized"],
        },
        {
            # compatible with current upstream typo/key
            "key": "norm_plust_log1p",
            "label": "Normalized + log1p",
            "aliases": [
                "norm_plust_log1p",
                "normpluslog1p",
                "norm_plus_log1p",
                "norm_log1p",
                "normlog1p",
                "normalized_log1p",
                "normalized_plus_log1p",
            ],
        },
    ]
    OUTDIR = "ggi_survival_logratio_curves_4types"
    os.makedirs(OUTDIR, exist_ok=True)
    T_GRID = np.linspace(0.0, 1.0, 100)
    MIN_SURVIVING_PAIRS = 10
    MIN_TOTAL_PAIRS = 500
    USE_ABS = False
    Z = 1.96
    ERRORBAR_EVERY = 5
    FIGSIZE = (9.5, 7.0)
    DPI = 300
    all_pair_dfs = {}
    all_curve_dfs = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]
        aliases = cond.get("aliases", [key])

        print("\n" + "=" * 80)
        print(f"Loading condition: {label} ({key})")
        print(f"Aliases: {aliases}")
        print("=" * 80)

        pair_df = load_condition_pairs(
            base_output_dir=BASE_OUTPUT_DIR,
            condition_key=key,
            aliases=aliases,
        )
        all_pair_dfs[key] = pair_df

        n_ggi = int(np.sum(pair_df["group"].eq("GGI")))
        n_rand = int(np.sum(pair_df["group"].eq("Random")))

        print(f"Loaded rows: {len(pair_df):,}")
        print(f"  GGI pairs:    {n_ggi:,}")
        print(f"  Random pairs: {n_rand:,}")
        print(f"  Datasets:     {pair_df['dataset'].nunique():,}")

        curve_df = compute_logratio_curve(
            pair_df=pair_df,
            t_grid=T_GRID,
            min_surviving_pairs=MIN_SURVIVING_PAIRS,
            min_total_pairs=MIN_TOTAL_PAIRS,
            use_abs=USE_ABS,
        )

        curve_df["condition"] = key
        curve_df["label"] = label
        all_curve_dfs.append(curve_df)
    curve_all = pd.concat(all_curve_dfs, axis=0, ignore_index=True)
    curve_csv = os.path.join(
        OUTDIR,
        f"ggi_survival_logratio_curves_4types_minTail{MIN_SURVIVING_PAIRS}.csv",
    )
    curve_all.to_csv(curve_csv, index=False)
    print(f"\nSaved curve table: {curve_csv}")
    plt.figure(figsize=FIGSIZE)
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]

        df = curve_all[
            (curve_all["condition"].eq(key)) &
            (curve_all["valid"])
        ].copy()

        if len(df) == 0:
            print(f"[WARN] No valid thresholds for {label}")
            continue

        x = df["t"].to_numpy()
        y = df["log_ratio"].to_numpy()
        lo = df["ci_low"].to_numpy()
        hi = df["ci_high"].to_numpy()

        plt.plot(
            x,
            y,
            linewidth=2.4,
            label=label,
        )

        plt.fill_between(
            x,
            lo,
            hi,
            alpha=0.18,
            linewidth=0,
        )

        # Sparse error bars
        err_idx = np.arange(0, len(df), ERRORBAR_EVERY)
        plt.errorbar(
            x[err_idx],
            y[err_idx],
            yerr=np.vstack([
                y[err_idx] - lo[err_idx],
                hi[err_idx] - y[err_idx],
            ]),
            fmt="none",
            capsize=2.5,
            linewidth=1.0,
            alpha=0.8,
        )
    plt.axhline(
        0,
        linestyle="--",
        linewidth=1.3,
        color="black",
        alpha=0.8,
    )
    xlabel = "|r| threshold t" if USE_ABS else "Pearson-r threshold t"
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(
        r"$\log\left[P_{\mathrm{GGI}}(r>t) / P_{\mathrm{rand}}(r>t)\right]$",
        fontsize=15,
    )
    plt.title(
        (
            "GGI enrichment in high-correlation tail\n"
            f"thresholds retained only if both groups have ≥ {MIN_SURVIVING_PAIRS} surviving pairs"
        ),
        fontsize=16,
    )
    plt.legend(frameon=True, fontsize=12)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    suffix = f"minTail{MIN_SURVIVING_PAIRS}"
    if USE_ABS:
        suffix += "_absr"
    else:
        suffix += "_signedr"
    png_out = os.path.join(OUTDIR, f"ggi_survival_logratio_4curves_{suffix}.png")
    svg_out = os.path.join(OUTDIR, f"ggi_survival_logratio_4curves_{suffix}.svg")
    pdf_out = os.path.join(OUTDIR, f"ggi_survival_logratio_4curves_{suffix}.pdf")
    plt.savefig(png_out, dpi=DPI, bbox_inches="tight")
    plt.savefig(svg_out, bbox_inches="tight")
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.show()
    print(f"\nSaved PNG: {png_out}")
    print(f"Saved SVG: {svg_out}")
    print(f"Saved PDF: {pdf_out}")
    SELECTED_T = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    summary_rows = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]
        df = curve_all[curve_all["condition"].eq(key)].copy()

        for t0 in SELECTED_T:
            idx = np.argmin(np.abs(df["t"].to_numpy() - t0))
            row = df.iloc[idx].to_dict()
            row["requested_t"] = t0
            row["condition_label"] = label
            summary_rows.append(row)
    selected_summary = pd.DataFrame(summary_rows)
    selected_summary = selected_summary[
        [
            "condition_label",
            "requested_t",
            "t",
            "valid",
            "k_ggi",
            "k_rand",
            "p_ggi",
            "p_rand",
            "log_ratio",
            "ci_low",
            "ci_high",
        ]
    ]
    selected_csv = os.path.join(OUTDIR, f"selected_threshold_summary_4types_{suffix}.csv")
    selected_summary.to_csv(selected_csv, index=False)
    print("\nSelected threshold summary:")
    print(selected_summary.to_string(index=False))
    print(f"\nSaved selected threshold summary: {selected_csv}")


def ggi_pair_mean_barscatter():
    BASE_OUTPUT_DIR = os.path.join(os.environ.get("SUPPL_OUT", "resources/repro/figS18"), "ggi")
    CONDITIONS = [
        {
            "key": "raw",
            "label": "Raw",
            "aliases": ["raw"],
        },
        {
            "key": "log1p",
            "label": "log1p",
            "aliases": ["log1p"],
        },
        {
            "key": "norm",
            "label": "Normalized",
            "aliases": ["norm", "normalized"],
        },
        {
            # Matches your current upstream typo/key if present
            "key": "norm_plust_log1p",
            "label": "Normalized + log1p",
            "aliases": [
                "norm_plust_log1p",
                "normpluslog1p",
                "norm_plus_log1p",
                "norm_log1p",
                "normlog1p",
                "normalized_log1p",
                "normalized_plus_log1p",
            ],
        },
    ]
    DATASETS = [
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]
    OUTDIR = "ggi_pair_mean_barscatter_4types"
    os.makedirs(OUTDIR, exist_ok=True)
    USE_ABS = False
    ERROR_KIND = "sem"
    JITTER = 0.055
    POINT_SIZE = 58
    LINE_ALPHA = 0.35
    DPI = 300
    rng = np.random.default_rng(0)
    rows = []
    missing_files = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]
        aliases = cond.get("aliases", [key])

        for dataset in DATASETS:
            dataset_clean = clean_dataset_name(dataset)

            fp = find_pair_file(
                base_output_dir=BASE_OUTPUT_DIR,
                condition_key=key,
                dataset_clean=dataset_clean,
                aliases=aliases,
            )

            if fp is None:
                missing_files.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "error": "file not found",
                })
                continue

            try:
                df = load_pair_file(fp)

                rand_vals = df.loc[df["group"].eq("Random"), "score"].to_numpy()
                ggi_vals = df.loc[df["group"].eq("GGI"), "score"].to_numpy()

                if len(rand_vals) == 0 or len(ggi_vals) == 0:
                    missing_files.append({
                        "condition": key,
                        "condition_label": label,
                        "dataset": dataset_clean,
                        "error": "missing Random or GGI rows",
                    })
                    continue

                rows.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "file": fp,
                    "mean_random": float(np.mean(rand_vals)),
                    "mean_ggi": float(np.mean(ggi_vals)),
                    "sem_random_pairs": float(np.std(rand_vals, ddof=1) / np.sqrt(len(rand_vals))),
                    "sem_ggi_pairs": float(np.std(ggi_vals, ddof=1) / np.sqrt(len(ggi_vals))),
                    "sd_random_pairs": float(np.std(rand_vals, ddof=1)),
                    "sd_ggi_pairs": float(np.std(ggi_vals, ddof=1)),
                    "n_random_pairs": int(len(rand_vals)),
                    "n_ggi_pairs": int(len(ggi_vals)),
                    "delta_ggi_minus_random": float(np.mean(ggi_vals) - np.mean(rand_vals)),
                })

            except Exception as e:
                missing_files.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "error": str(e),
                })
    summary_df = pd.DataFrame(rows)
    if len(summary_df) == 0:
        raise RuntimeError("No usable GGI/random pair files found.")
    summary_csv = os.path.join(OUTDIR, "dataset_level_mean_random_vs_ggi_4types.csv")
    summary_df.to_csv(summary_csv, index=False)
    if missing_files:
        missing_df = pd.DataFrame(missing_files)
        missing_csv = os.path.join(OUTDIR, "missing_or_skipped_files_4types.csv")
        missing_df.to_csv(missing_csv, index=False)
        print(f"[saved] missing/skipped file log: {missing_csv}")
    print(f"[saved] dataset-level summary: {summary_csv}")
    print(summary_df.head().to_string(index=False))
    dataset_order = [clean_dataset_name(d) for d in DATASETS]
    dataset_order = [d for d in dataset_order if d in set(summary_df["dataset"])]
    cmap = plt.get_cmap("tab10")
    dataset_to_color = {
        d: cmap(i % 10)
        for i, d in enumerate(dataset_order)
    }
    N_CONDITIONS = len(CONDITIONS)
    fig, axes = plt.subplots(
        1,
        N_CONDITIONS,
        figsize=(5.1 * N_CONDITIONS, 5.8),
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    x_rand = 0
    x_ggi = 1
    for ax, cond in zip(axes, CONDITIONS):
        key = cond["key"]
        label = cond["label"]

        sub = summary_df[summary_df["condition"].eq(key)].copy()
        sub["dataset"] = pd.Categorical(sub["dataset"], categories=dataset_order, ordered=True)
        sub = sub.sort_values("dataset")

        # Dataset paired points
        for _, row in sub.iterrows():
            d = row["dataset"]
            color = dataset_to_color.get(str(d), "gray")

            jr = rng.normal(0, JITTER)
            jg = rng.normal(0, JITTER)

            y_rand = row["mean_random"]
            y_ggi = row["mean_ggi"]

            ax.plot(
                [x_rand + jr, x_ggi + jg],
                [y_rand, y_ggi],
                color=color,
                alpha=LINE_ALPHA,
                linewidth=1.4,
                zorder=1,
            )

            ax.scatter(
                x_rand + jr,
                y_rand,
                s=POINT_SIZE,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
                zorder=2,
            )

            ax.scatter(
                x_ggi + jg,
                y_ggi,
                s=POINT_SIZE,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
                zorder=2,
            )

        # Black grand mean across datasets
        mean_rand = sub["mean_random"].mean()
        mean_ggi = sub["mean_ggi"].mean()

        if ERROR_KIND == "sem":
            err_rand = sub["mean_random"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.0
            err_ggi = sub["mean_ggi"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.0
        elif ERROR_KIND == "sd":
            err_rand = sub["mean_random"].std(ddof=1) if len(sub) > 1 else 0.0
            err_ggi = sub["mean_ggi"].std(ddof=1) if len(sub) > 1 else 0.0
        else:
            raise ValueError("ERROR_KIND must be 'sem' or 'sd'")

        ax.errorbar(
            [x_rand, x_ggi],
            [mean_rand, mean_ggi],
            yerr=[err_rand, err_ggi],
            fmt="o",
            color="black",
            markersize=9,
            capsize=5,
            elinewidth=2.0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
            label=f"Mean ± {ERROR_KIND.upper()}",
        )

        ax.plot(
            [x_rand, x_ggi],
            [mean_rand, mean_ggi],
            color="black",
            linewidth=2.0,
            zorder=4,
        )

        # Optional faint bars behind the points
        ax.bar(
            [x_rand, x_ggi],
            [mean_rand, mean_ggi],
            width=0.48,
            color="black",
            alpha=0.10,
            zorder=0,
        )

        delta = mean_ggi - mean_rand
        frac_ggi_gt_rand = (
            np.mean(sub["mean_ggi"].to_numpy() > sub["mean_random"].to_numpy())
            if len(sub) > 0 else np.nan
        )

        ax.text(
            0.5,
            0.97,
            (
                f"N datasets = {len(sub)}\n"
                f"mean Δ = {delta:.4g}\n"
                f"frac GGI>rand = {frac_ggi_gt_rand:.2g}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

        ax.set_title(label, fontsize=16)
        ax.set_xticks([x_rand, x_ggi])
        ax.set_xticklabels(["Random", "GGI"], fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(-0.55, 1.55)
    axes[0].set_ylabel("Mean |Pearson r|" if USE_ABS else "Mean Pearson r", fontsize=15)
    fig.suptitle(
        "Dataset-level mean GGI vs random gene-pair correlation",
        fontsize=18,
        y=1.03,
    )
    handles = []
    labels = []
    for d in dataset_order:
        h = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=dataset_to_color[d],
            markeredgecolor="black",
            markersize=8,
            label=d,
        )
        handles.append(h)
        labels.append(d)
    mean_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="black",
        markeredgecolor="white",
        markersize=9,
        linewidth=2,
        label=f"Mean ± {ERROR_KIND.upper()}",
    )
    handles.append(mean_handle)
    labels.append(f"Mean ± {ERROR_KIND.upper()}")
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=9,
    )
    plt.tight_layout()
    suffix = "absr" if USE_ABS else "signedr"
    png_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.png",
    )
    svg_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.svg",
    )
    pdf_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.pdf",
    )
    plt.savefig(png_out, dpi=DPI, bbox_inches="tight")
    plt.savefig(svg_out, bbox_inches="tight")
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.show()
    print(f"[saved] PNG: {png_out}")
    print(f"[saved] SVG: {svg_out}")
    print(f"[saved] PDF: {pdf_out}")
    stats_rows = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]

        sub = summary_df[summary_df["condition"].eq(key)].copy()

        diffs = sub["mean_ggi"].to_numpy() - sub["mean_random"].to_numpy()

        stats_rows.append({
            "condition": key,
            "label": label,
            "n_datasets": len(sub),
            "mean_random_across_datasets": sub["mean_random"].mean(),
            "mean_ggi_across_datasets": sub["mean_ggi"].mean(),
            "mean_delta_ggi_minus_random": np.mean(diffs),
            "sem_delta_across_datasets": (
                np.std(diffs, ddof=1) / np.sqrt(len(diffs))
                if len(diffs) > 1 else np.nan
            ),
            "frac_datasets_ggi_greater_random": (
                np.mean(diffs > 0) if len(diffs) > 0 else np.nan
            ),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = os.path.join(OUTDIR, "paired_dataset_level_stats_4types.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"[saved] paired stats: {stats_csv}")
    print(stats_df.to_string(index=False))


def ggi_pair_mean_barscatter_stats():
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except Exception:
        ttest_rel = None
        wilcoxon = None
    BASE_OUTPUT_DIR = os.path.join(os.environ.get("SUPPL_OUT", "resources/repro/figS18"), "ggi")
    CONDITIONS = [
        {
            "key": "raw",
            "label": "Raw",
            "aliases": ["raw"],
        },
        {
            "key": "log1p",
            "label": "log1p",
            "aliases": ["log1p"],
        },
        {
            "key": "norm",
            "label": "Normalized",
            "aliases": ["norm", "normalized"],
        },
        {
            # Matches your current upstream typo/key if present
            "key": "norm_plust_log1p",
            "label": "Normalized + log1p",
            "aliases": [
                "norm_plust_log1p",
                "normpluslog1p",
                "norm_plus_log1p",
                "norm_log1p",
                "normlog1p",
                "normalized_log1p",
                "normalized_plus_log1p",
            ],
        },
    ]
    DATASETS = [
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]
    OUTDIR = "ggi_pair_mean_barscatter_4types"
    os.makedirs(OUTDIR, exist_ok=True)
    USE_ABS = False
    ERROR_KIND = "sem"
    JITTER = 0.055
    POINT_SIZE = 58
    LINE_ALPHA = 0.35
    DPI = 300
    rng = np.random.default_rng(0)
    rows = []
    missing_files = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]
        aliases = cond.get("aliases", [key])

        for dataset in DATASETS:
            dataset_clean = clean_dataset_name(dataset)

            fp = find_pair_file(
                base_output_dir=BASE_OUTPUT_DIR,
                condition_key=key,
                dataset_clean=dataset_clean,
                aliases=aliases,
            )

            if fp is None:
                missing_files.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "error": "file not found",
                })
                continue

            try:
                df = load_pair_file(fp)

                rand_vals = df.loc[df["group"].eq("Random"), "score"].to_numpy()
                ggi_vals = df.loc[df["group"].eq("GGI"), "score"].to_numpy()

                if len(rand_vals) == 0 or len(ggi_vals) == 0:
                    missing_files.append({
                        "condition": key,
                        "condition_label": label,
                        "dataset": dataset_clean,
                        "error": "missing Random or GGI rows",
                    })
                    continue

                rows.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "file": fp,
                    "mean_random": float(np.mean(rand_vals)),
                    "mean_ggi": float(np.mean(ggi_vals)),
                    "sem_random_pairs": float(np.std(rand_vals, ddof=1) / np.sqrt(len(rand_vals))),
                    "sem_ggi_pairs": float(np.std(ggi_vals, ddof=1) / np.sqrt(len(ggi_vals))),
                    "sd_random_pairs": float(np.std(rand_vals, ddof=1)),
                    "sd_ggi_pairs": float(np.std(ggi_vals, ddof=1)),
                    "n_random_pairs": int(len(rand_vals)),
                    "n_ggi_pairs": int(len(ggi_vals)),
                    "delta_ggi_minus_random": float(np.mean(ggi_vals) - np.mean(rand_vals)),
                })

            except Exception as e:
                missing_files.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "error": str(e),
                })
    summary_df = pd.DataFrame(rows)
    if len(summary_df) == 0:
        raise RuntimeError("No usable GGI/random pair files found.")
    summary_csv = os.path.join(OUTDIR, "dataset_level_mean_random_vs_ggi_4types.csv")
    summary_df.to_csv(summary_csv, index=False)
    if missing_files:
        missing_df = pd.DataFrame(missing_files)
        missing_csv = os.path.join(OUTDIR, "missing_or_skipped_files_4types.csv")
        missing_df.to_csv(missing_csv, index=False)
        print(f"[saved] missing/skipped file log: {missing_csv}")
    print(f"[saved] dataset-level summary: {summary_csv}")
    print(summary_df.head().to_string(index=False))
    dataset_order = [clean_dataset_name(d) for d in DATASETS]
    dataset_order = [d for d in dataset_order if d in set(summary_df["dataset"])]
    cmap = plt.get_cmap("tab10")
    dataset_to_color = {
        d: cmap(i % 10)
        for i, d in enumerate(dataset_order)
    }
    N_CONDITIONS = len(CONDITIONS)
    fig, axes = plt.subplots(
        1,
        N_CONDITIONS,
        figsize=(5.1 * N_CONDITIONS, 5.8),
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    x_rand = 0
    x_ggi = 1
    for ax, cond in zip(axes, CONDITIONS):
        key = cond["key"]
        label = cond["label"]

        sub = summary_df[summary_df["condition"].eq(key)].copy()
        sub["dataset"] = pd.Categorical(sub["dataset"], categories=dataset_order, ordered=True)
        sub = sub.sort_values("dataset")

        # Dataset paired points
        for _, row in sub.iterrows():
            d = row["dataset"]
            color = dataset_to_color.get(str(d), "gray")

            jr = rng.normal(0, JITTER)
            jg = rng.normal(0, JITTER)

            y_rand = row["mean_random"]
            y_ggi = row["mean_ggi"]

            ax.plot(
                [x_rand + jr, x_ggi + jg],
                [y_rand, y_ggi],
                color=color,
                alpha=LINE_ALPHA,
                linewidth=1.4,
                zorder=1,
            )

            ax.scatter(
                x_rand + jr,
                y_rand,
                s=POINT_SIZE,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
                zorder=2,
            )

            ax.scatter(
                x_ggi + jg,
                y_ggi,
                s=POINT_SIZE,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
                zorder=2,
            )

        # Black grand mean across datasets
        mean_rand = sub["mean_random"].mean()
        mean_ggi = sub["mean_ggi"].mean()

        if ERROR_KIND == "sem":
            err_rand = sub["mean_random"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.0
            err_ggi = sub["mean_ggi"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.0
        elif ERROR_KIND == "sd":
            err_rand = sub["mean_random"].std(ddof=1) if len(sub) > 1 else 0.0
            err_ggi = sub["mean_ggi"].std(ddof=1) if len(sub) > 1 else 0.0
        else:
            raise ValueError("ERROR_KIND must be 'sem' or 'sd'")

        ax.errorbar(
            [x_rand, x_ggi],
            [mean_rand, mean_ggi],
            yerr=[err_rand, err_ggi],
            fmt="o",
            color="black",
            markersize=9,
            capsize=5,
            elinewidth=2.0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
            label=f"Mean ± {ERROR_KIND.upper()}",
        )

        ax.plot(
            [x_rand, x_ggi],
            [mean_rand, mean_ggi],
            color="black",
            linewidth=2.0,
            zorder=4,
        )

        # Optional faint bars behind the points
        ax.bar(
            [x_rand, x_ggi],
            [mean_rand, mean_ggi],
            width=0.48,
            color="black",
            alpha=0.10,
            zorder=0,
        )

        delta = mean_ggi - mean_rand
        frac_ggi_gt_rand = (
            np.mean(sub["mean_ggi"].to_numpy() > sub["mean_random"].to_numpy())
            if len(sub) > 0 else np.nan
        )

        paired_t_p = paired_ttest_pvalue(
            sub["mean_random"].to_numpy(),
            sub["mean_ggi"].to_numpy(),
        )

        paired_w_p = paired_wilcoxon_pvalue(
            sub["mean_random"].to_numpy(),
            sub["mean_ggi"].to_numpy(),
        )

        ax.text(
            0.5,
            0.97,
            (
                f"N datasets = {len(sub)}\n"
                f"black mean rand = {mean_rand:.4g}\n"
                f"black mean GGI = {mean_ggi:.4g}\n"
                f"paired t p = {format_pvalue(paired_t_p)}\n"
                f"Wilcoxon p = {format_pvalue(paired_w_p)}\n"
                f"mean Δ = {delta:.4g}\n"
                f"frac GGI>rand = {frac_ggi_gt_rand:.2g}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

        ax.set_title(label, fontsize=16)
        ax.set_xticks([x_rand, x_ggi])
        ax.set_xticklabels(["Random", "GGI"], fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(-0.55, 1.55)
    axes[0].set_ylabel("Mean |Pearson r|" if USE_ABS else "Mean Pearson r", fontsize=15)
    fig.suptitle(
        "Dataset-level mean GGI vs random gene-pair correlation",
        fontsize=18,
        y=1.03,
    )
    handles = []
    labels = []
    for d in dataset_order:
        h = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=dataset_to_color[d],
            markeredgecolor="black",
            markersize=8,
            label=d,
        )
        handles.append(h)
        labels.append(d)
    mean_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="black",
        markeredgecolor="white",
        markersize=9,
        linewidth=2,
        label=f"Mean ± {ERROR_KIND.upper()}",
    )
    handles.append(mean_handle)
    labels.append(f"Mean ± {ERROR_KIND.upper()}")
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=9,
    )
    plt.tight_layout()
    suffix = "absr" if USE_ABS else "signedr"
    png_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.png",
    )
    svg_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.svg",
    )
    pdf_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.pdf",
    )
    plt.savefig(png_out, dpi=DPI, bbox_inches="tight")
    plt.savefig(svg_out, bbox_inches="tight")
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.show()
    print(f"[saved] PNG: {png_out}")
    print(f"[saved] SVG: {svg_out}")
    print(f"[saved] PDF: {pdf_out}")
    stats_rows = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]

        sub = summary_df[summary_df["condition"].eq(key)].copy()

        diffs = sub["mean_ggi"].to_numpy() - sub["mean_random"].to_numpy()

        paired_t_p = paired_ttest_pvalue(
            sub["mean_random"].to_numpy(),
            sub["mean_ggi"].to_numpy(),
        )

        paired_w_p = paired_wilcoxon_pvalue(
            sub["mean_random"].to_numpy(),
            sub["mean_ggi"].to_numpy(),
        )

        stats_rows.append({
            "condition": key,
            "label": label,
            "n_datasets": len(sub),
            "mean_random_across_datasets": sub["mean_random"].mean(),
            "mean_ggi_across_datasets": sub["mean_ggi"].mean(),
            "mean_delta_ggi_minus_random": np.mean(diffs),
            "sem_delta_across_datasets": (
                np.std(diffs, ddof=1) / np.sqrt(len(diffs))
                if len(diffs) > 1 else np.nan
            ),
            "frac_datasets_ggi_greater_random": (
                np.mean(diffs > 0) if len(diffs) > 0 else np.nan
            ),
            "paired_ttest_pvalue_ggi_vs_random": paired_t_p,
            "paired_wilcoxon_pvalue_ggi_vs_random": paired_w_p,
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = os.path.join(OUTDIR, "paired_dataset_level_stats_4types.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"[saved] paired stats: {stats_csv}")
    print(stats_df.to_string(index=False))


def ppi_engine_run():
    DATA_ROOT = os.environ["CIPHER_DATA_DIR"]
    OUTDIR = os.path.join(os.environ.get("SUPPL_OUT", "resources/repro/figS18"), "ppi")
    os.makedirs(OUTDIR, exist_ok=True)
    PPI_CSV = os.path.join(SUPPL, "Human_protein_protein_interactions_collapsed.csv")
    ppi_df, ppi_pairs_unique = load_ppi_unique_pairs(PPI_CSV)
    print(f"[PPI] rows in file: {len(ppi_df):,}")
    print(f"[PPI] unique undirected interactions: {len(ppi_pairs_unique):,}")
    print("[PPI] example rows:")
    print(ppi_df[["Interactor A", "Interactor B"]].head(5).to_string(index=False))
    datapaths = [
            "ReplogleWeissman2022_rpe1.h5ad",
            "ReplogleWeissman2022_K562_essential.h5ad",
            "GSE264667_jurkat_raw_singlecell_01.h5ad",
            "GSE264667_hepg2_raw_singlecell_01.h5ad",
            "NormanWeissman2019_filtered.h5ad",
            "FrangiehIzar2021_RNA.h5ad",
            "TianKampmann2019_day7neuron.h5ad",
            "TianKampmann2021_CRISPRi.h5ad",
            "TianKampmann2021_CRISPRa.h5ad",
            "TianKampmann2019_iPSC.h5ad",
        ]
    datapaths = [os.path.join(DATA_ROOT, f) for f in datapaths]
    preprocess_configs = [
            {"preprocess_name": "norm_plust_log1p",  "log1p": True, "norm": True},
            {"preprocess_name": "raw",   "log1p": False, "norm": False},
            {"preprocess_name": "log1p", "log1p": True,  "norm": False},
            {"preprocess_name": "norm",  "log1p": False, "norm": True},
        ]
    expression_threshold = 0.01
    min_samples = 2
    use_abs = False
    rng_seed = 0
    plot_bins = 80
    chunk_size = 20000
    summaries, errors = [], []
    for cfg in preprocess_configs:
            preprocess_name = cfg["preprocess_name"]
            log1p = cfg["log1p"]
            norm = cfg["norm"]

            print("\n" + "=" * 100)
            print(f"PREPROCESSING MODE: {preprocess_name} | log1p={log1p} | norm={norm}")
            print("=" * 100)

            outdir_cfg = os.path.join(OUTDIR, preprocess_name)
            os.makedirs(outdir_cfg, exist_ok=True)

            for p in datapaths:
                if not os.path.exists(p):
                    print(f"[SKIP] missing: {p}")
                    errors.append({
                        "dataset": os.path.basename(p),
                        "preprocess": preprocess_name,
                        "log1p": log1p,
                        "norm": norm,
                        "error": "file not found",
                    })
                    continue

                try:
                    s = analyze_dataset_ppi_vs_random_sparse(
                        data_path=p,
                        ppi_pairs_unique=ppi_pairs_unique,
                        expression_threshold=expression_threshold,
                        min_samples=min_samples,
                        use_abs=use_abs,
                        save_dir=outdir_cfg,
                        rng_seed=rng_seed,
                        plot_bins=plot_bins,
                        chunk_size=chunk_size,
                        show_plots=True,
                        log1p=log1p,
                        norm=norm,
                        preprocess_name=preprocess_name,
                    )
                    summaries.append(s)
                    pd.DataFrame([s]).to_csv(
                        os.path.join(outdir_cfg, f"{s['dataset']}__{preprocess_name}__summary.csv"),
                        index=False
                    )
                except Exception as e:
                    print(f"[ERROR] {os.path.basename(p)} | {preprocess_name}: {e}")
                    errors.append({
                        "dataset": os.path.basename(p),
                        "preprocess": preprocess_name,
                        "log1p": log1p,
                        "norm": norm,
                        "error": str(e),
                    })
    pd.DataFrame(summaries).to_csv(os.path.join(OUTDIR, "ALL_DATASETS_ALL_PREPROCESS__summary.csv"), index=False)
    if errors:
            pd.DataFrame(errors).to_csv(os.path.join(OUTDIR, "ALL_DATASETS_ALL_PREPROCESS__errors.csv"), index=False)
    print("\nDone.")
    print(f"  summaries: {os.path.join(OUTDIR, 'ALL_DATASETS_ALL_PREPROCESS__summary.csv')}")
    if errors:
            print(f"  errors:    {os.path.join(OUTDIR, 'ALL_DATASETS_ALL_PREPROCESS__errors.csv')}")


def ppi_gene_survival_panels_t05():
    DATASETS = [
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]
    CONDITIONS = [
        {
            "name": "Raw",
            "short": "raw",
            "aliases": ["raw"],
        },
        {
            "name": "log1p",
            "short": "log1p",
            "aliases": ["log1p"],
        },
        {
            "name": "Normalized",
            "short": "norm",
            "aliases": ["norm", "normalized"],
        },
        {
            "name": "Normalized + log1p",
            "short": "norm_plust_log1p",
            "aliases": [
                "norm_plust_log1p",
                "normpluslog1p",
                "norm_plus_log1p",
                "norm_log1p",
                "normlog1p",
                "normalized_log1p",
                "normalized_plus_log1p",
            ],
        },
    ]
    BASE_OUTPUT_DIR = os.path.join(os.environ.get("SUPPL_OUT", "resources/repro/figS18"), "ppi")
    T_THRESHOLD = 0.5
    PLOT_FLOOR = 1e-2
    FILTER_FOR_PLOTTING = True
    BINS = 60
    AXIS_SCALE = "linear"
    N_CONDITIONS = len(CONDITIONS)
    OUTDIR = f"{N_CONDITIONS}_panel_gene_survival_from_ppi_pairs_t{T_THRESHOLD}_{AXIS_SCALE}axes"
    os.makedirs(OUTDIR, exist_ok=True)
    DPI = 300
    compiled_by_condition = {}
    gene_survival_tables = []
    summary_rows = []
    for cond in CONDITIONS:
        print("\n" + "=" * 100)
        print(f"Loading condition: {cond['name']}")
        print(f"Condition key:     {cond['short']}")
        print(f"Aliases:           {cond.get('aliases', [cond['short']])}")
        print(f"Base output dir:   {BASE_OUTPUT_DIR}")
        print("=" * 100)

        pair_long_all, gene_df, summary = load_and_compile_condition(
            base_output_dir=BASE_OUTPUT_DIR,
            condition_name=cond["name"],
            condition_short=cond["short"],
            condition_aliases=cond.get("aliases", [cond["short"]]),
            datasets=DATASETS,
            threshold=T_THRESHOLD,
        )

        compiled_by_condition[cond["short"]] = {
            "name": cond["name"],
            "gene_df": gene_df,
            "summary": summary,
        }

        gene_survival_tables.append(gene_df)
        summary_rows.append(summary)
    all_gene_survival_df = pd.concat(gene_survival_tables, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    plot_data_csv = os.path.join(
        OUTDIR,
        f"ppi_gene_survival_plot_data_{N_CONDITIONS}types_t{T_THRESHOLD}_{AXIS_SCALE}axes.csv",
    )
    summary_csv = os.path.join(
        OUTDIR,
        f"ppi_gene_survival_condition_summary_{N_CONDITIONS}types_t{T_THRESHOLD}_{AXIS_SCALE}axes.csv",
    )
    all_gene_survival_df.to_csv(plot_data_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[saved] minimal plot data: {plot_data_csv}")
    print(f"[saved] condition summary: {summary_csv}")
    print("\nSummary across conditions:")
    print(summary_df.to_string(index=False))
    if AXIS_SCALE not in ["log", "linear"]:
        raise ValueError("AXIS_SCALE must be either 'log' or 'linear'")
    if AXIS_SCALE == "log":
        xmin, xmax = PLOT_FLOOR, 1.0
        ymin, ymax = PLOT_FLOOR, 1.0
        xedges = np.logspace(np.log10(xmin), np.log10(xmax), BINS + 1)
        yedges = np.logspace(np.log10(ymin), np.log10(ymax), BINS + 1)
    else:
        xmin, xmax = 0.0, 1.0
        ymin, ymax = 0.0, 1.0
        xedges = np.linspace(xmin, xmax, BINS + 1)
        yedges = np.linspace(ymin, ymax, BINS + 1)
    panel_data = []
    global_max_count = 1
    for cond in CONDITIONS:
        key = cond["short"]
        gene_df = compiled_by_condition[key]["gene_df"].copy()

        if FILTER_FOR_PLOTTING:
            gene_df = gene_df[
                (gene_df["ggi_survival"] > PLOT_FLOOR)
                | (gene_df["random_survival"] > PLOT_FLOOR)
            ].copy()

        x = gene_df["random_survival"].to_numpy()
        y = gene_df["ggi_survival"].to_numpy()

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        line_mask = (x > PLOT_FLOOR) | (y > PLOT_FLOOR)
        n_line = int(np.sum(line_mask))
        n_above = int(np.sum((y > x) & line_mask))
        n_below = int(np.sum((y < x) & line_mask))
        n_equal = int(np.sum((y == x) & line_mask))

        frac_above_line = n_above / n_line if n_line > 0 else np.nan
        frac_below_line = n_below / n_line if n_line > 0 else np.nan
        frac_equal_line = n_equal / n_line if n_line > 0 else np.nan

        if AXIS_SCALE == "log":
            x_plot = np.maximum(x, PLOT_FLOOR)
            y_plot = np.maximum(y, PLOT_FLOOR)
        else:
            x_plot = x.copy()
            y_plot = y.copy()

        counts, _, _ = np.histogram2d(
            x_plot,
            y_plot,
            bins=[xedges, yedges],
        )

        global_max_count = max(global_max_count, counts.max())

        panel_data.append({
            "key": key,
            "display_name": cond["name"],
            "df": gene_df,
            "x": x,
            "y": y,
            "x_plot": x_plot,
            "y_plot": y_plot,
            "counts": counts,
            "n_line": n_line,
            "n_above": n_above,
            "n_below": n_below,
            "n_equal": n_equal,
            "frac_above_line": frac_above_line,
            "frac_below_line": frac_below_line,
            "frac_equal_line": frac_equal_line,
        })
    fig, axes = plt.subplots(
        1,
        N_CONDITIONS,
        figsize=(6.2 * N_CONDITIONS, 6.2),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    mappable = None
    for ax, pdata in zip(axes, panel_data):
        counts_masked = np.ma.masked_where(
            pdata["counts"].T == 0,
            pdata["counts"].T,
        )

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            counts_masked,
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=max(1, global_max_count)),
            shading="auto",
        )
        mappable = mesh

        ax.set_xscale(AXIS_SCALE)
        ax.set_yscale(AXIS_SCALE)

        ax.plot(
            [xmin, xmax],
            [ymin, ymax],
            linestyle="--",
            linewidth=1.3,
            color="white",
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_title(pdata["display_name"], fontsize=16)
        ax.tick_params(axis="both", labelsize=12)

        mean_x = np.mean(pdata["x"]) if len(pdata["x"]) > 0 else np.nan
        mean_y = np.mean(pdata["y"]) if len(pdata["y"]) > 0 else np.nan
        mean_d = np.mean(pdata["y"] - pdata["x"]) if len(pdata["x"]) > 0 else np.nan
        frac_above = np.mean(pdata["y"] > pdata["x"]) if len(pdata["x"]) > 0 else np.nan
        frac_below = np.mean(pdata["y"] < pdata["x"]) if len(pdata["x"]) > 0 else np.nan

        ax.text(
            0.03,
            0.97,
            (
                f"Genes shown: {len(pdata['x']):,}\n"
                f"mean PPI: {mean_y:.3g}\n"
                f"mean random: {mean_x:.3g}\n"
                f"mean Δ: {mean_d:.3g}\n"
                f"frac above x=y: {frac_above:.3g}\n"
                f"frac below x=y: {frac_below:.3g}\n"
                f"among x or y > {PLOT_FLOOR:g}:\n"
                f"  above: {pdata['frac_above_line']:.3g} ({pdata['n_above']:,}/{pdata['n_line']:,})\n"
                f"  below: {pdata['frac_below_line']:.3g} ({pdata['n_below']:,}/{pdata['n_line']:,})"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.2,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )
    axes[0].set_ylabel(f"PPI survival: P(r > {T_THRESHOLD})", fontsize=15)
    for ax in axes:
        ax.set_xlabel(f"Random survival: P(r > {T_THRESHOLD})", fontsize=15)
    fig.suptitle(
        f"Gene-level PPI survival enrichment across preprocessing choices\nthreshold: r > {T_THRESHOLD}, {AXIS_SCALE} axes",
        fontsize=19,
        y=1.04,
    )
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Number of genes per bin, log color scale", fontsize=14)
    plt.tight_layout()
    suffix = f"t{T_THRESHOLD}_floor{PLOT_FLOOR}_{AXIS_SCALE}axes".replace(".", "p")
    if FILTER_FOR_PLOTTING:
        suffix += "_filtered"
    png_out = os.path.join(
        OUTDIR,
        f"{N_CONDITIONS}_panel_gene_survival_density_{suffix}.png",
    )
    svg_out = os.path.join(
        OUTDIR,
        f"{N_CONDITIONS}_panel_gene_survival_density_{suffix}.svg",
    )
    plt.savefig(png_out, dpi=DPI, bbox_inches="tight")
    plt.savefig(svg_out, bbox_inches="tight")
    plt.show()
    print(f"\nSaved {N_CONDITIONS}-panel PNG: {png_out}")
    print(f"Saved {N_CONDITIONS}-panel SVG: {svg_out}")


def ppi_gene_survival_panels_t04():
    DATASETS = [
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]
    CONDITIONS = [
        {
            "name": "Raw",
            "short": "raw",
            "aliases": ["raw"],
        },
        {
            "name": "log1p",
            "short": "log1p",
            "aliases": ["log1p"],
        },
        {
            "name": "Normalized",
            "short": "norm",
            "aliases": ["norm", "normalized"],
        },
        {
            "name": "Normalized + log1p",
            "short": "norm_plust_log1p",
            "aliases": [
                "norm_plust_log1p",   # typo currently used in your upstream script
                "normpluslog1p",
                "norm_plus_log1p",
                "norm_log1p",
                "normlog1p",
                "normalized_log1p",
                "normalized_plus_log1p",
            ],
        },
    ]
    BASE_OUTPUT_DIR = os.path.join(os.environ.get("SUPPL_OUT", "resources/repro/figS18"), "ppi")
    T_THRESHOLD = 0.4
    PLOT_FLOOR = 1e-2
    FILTER_FOR_PLOTTING = True
    BINS = 60
    N_PANELS = len(CONDITIONS)
    OUTDIR = f"{N_PANELS}_panel_gene_survival_from_ppi_pairs_t{T_THRESHOLD}"
    os.makedirs(OUTDIR, exist_ok=True)
    compiled_by_condition = {}
    summary_rows = []
    for cond in CONDITIONS:
        print("\n" + "=" * 100)
        print(f"Loading condition: {cond['name']}")
        print(f"Condition key:     {cond['short']}")
        print(f"Aliases:           {cond.get('aliases', [cond['short']])}")
        print(f"Base output dir:   {BASE_OUTPUT_DIR}")
        print("=" * 100)

        pair_long_all, gene_df, summary = load_and_compile_condition(
            base_output_dir=BASE_OUTPUT_DIR,
            condition_name=cond["name"],
            condition_short=cond["short"],
            condition_aliases=cond.get("aliases", [cond["short"]]),
            datasets=DATASETS,
            threshold=T_THRESHOLD,
        )

        compiled_by_condition[cond["short"]] = {
            "name": cond["name"],
            "pair_long_all": pair_long_all,
            "gene_df": gene_df,
            "summary": summary,
        }

        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTDIR, f"{N_PANELS}_condition_summary_t{T_THRESHOLD}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("\nSummary across conditions:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {summary_csv}")
    xmin, xmax = PLOT_FLOOR, 1.0
    ymin, ymax = PLOT_FLOOR, 1.0
    xedges = np.logspace(np.log10(xmin), np.log10(xmax), BINS + 1)
    yedges = np.logspace(np.log10(ymin), np.log10(ymax), BINS + 1)
    panel_data = []
    global_max_count = 1
    for cond in CONDITIONS:
        key = cond["short"]
        gene_df = compiled_by_condition[key]["gene_df"].copy()

        if FILTER_FOR_PLOTTING:
            gene_df = gene_df[
                (gene_df["ggi_survival"] > PLOT_FLOOR)
                | (gene_df["random_survival"] > PLOT_FLOOR)
            ].copy()

        x = gene_df["random_survival"].to_numpy()
        y = gene_df["ggi_survival"].to_numpy()

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        x_plot = np.maximum(x, PLOT_FLOOR)
        y_plot = np.maximum(y, PLOT_FLOOR)

        counts, _, _ = np.histogram2d(
            x_plot,
            y_plot,
            bins=[xedges, yedges],
        )

        global_max_count = max(global_max_count, counts.max())

        panel_data.append({
            "key": key,
            "display_name": cond["name"],
            "df": gene_df,
            "x": x,
            "y": y,
            "x_plot": x_plot,
            "y_plot": y_plot,
            "counts": counts,
        })
    fig, axes = plt.subplots(
        1,
        N_PANELS,
        figsize=(6.2 * N_PANELS, 6.2),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    mappable = None
    for ax, pdata in zip(axes, panel_data):
        counts_masked = np.ma.masked_where(
            pdata["counts"].T == 0,
            pdata["counts"].T,
        )

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            counts_masked,
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=max(1, global_max_count)),
            shading="auto",
        )
        mappable = mesh

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.plot(
            [xmin, xmax],
            [ymin, ymax],
            linestyle="--",
            linewidth=1.3,
            color="white",
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_title(pdata["display_name"], fontsize=16)
        ax.tick_params(axis="both", labelsize=12)

        mean_x = np.mean(pdata["x"]) if len(pdata["x"]) > 0 else np.nan
        mean_y = np.mean(pdata["y"]) if len(pdata["y"]) > 0 else np.nan
        mean_d = np.mean(pdata["y"] - pdata["x"]) if len(pdata["x"]) > 0 else np.nan

        # Fraction above / below x=y among shown genes
        frac_above = np.mean(pdata["y"] > pdata["x"]) if len(pdata["x"]) > 0 else np.nan
        frac_below = np.mean(pdata["y"] < pdata["x"]) if len(pdata["x"]) > 0 else np.nan
        frac_equal = np.mean(pdata["y"] == pdata["x"]) if len(pdata["x"]) > 0 else np.nan

        ax.text(
            0.03,
            0.97,
            (
                f"Genes shown: {len(pdata['x']):,}\n"
                f"mean PPI: {mean_y:.3g}\n"
                f"mean random: {mean_x:.3g}\n"
                f"mean Δ: {mean_d:.3g}\n"
                f"frac above x=y: {frac_above:.3g}\n"
                f"frac below x=y: {frac_below:.3g}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )
    axes[0].set_ylabel(f"PPI survival: P(r > {T_THRESHOLD})", fontsize=15)
    for ax in axes:
        ax.set_xlabel(f"Random survival: P(r > {T_THRESHOLD})", fontsize=15)
    fig.suptitle(
        (
            f"Gene-level PPI survival enrichment across preprocessing choices\n"
            f"threshold: r > {T_THRESHOLD}"
        ),
        fontsize=19,
        y=1.04,
    )
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Number of genes per bin, log color scale", fontsize=14)
    plt.tight_layout()
    suffix = f"t{T_THRESHOLD}_floor{PLOT_FLOOR}".replace(".", "p")
    if FILTER_FOR_PLOTTING:
        suffix += "_filtered"
    png_out = os.path.join(
        OUTDIR,
        f"{N_PANELS}_panel_gene_survival_density_{suffix}.png",
    )
    svg_out = os.path.join(
        OUTDIR,
        f"{N_PANELS}_panel_gene_survival_density_{suffix}.svg",
    )
    pdf_out = os.path.join(
        OUTDIR,
        f"{N_PANELS}_panel_gene_survival_density_{suffix}.pdf",
    )
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.savefig(svg_out, bbox_inches="tight")
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.show()
    print(f"\nSaved multi-panel PNG: {png_out}")
    print(f"Saved multi-panel SVG: {svg_out}")
    print(f"Saved multi-panel PDF: {pdf_out}")
    fig, axes = plt.subplots(
        1,
        N_PANELS,
        figsize=(6.2 * N_PANELS, 6.2),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    mappable = None
    for ax, pdata in zip(axes, panel_data):
        counts_masked = np.ma.masked_where(
            pdata["counts"].T == 0,
            pdata["counts"].T,
        )

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            counts_masked,
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=max(1, global_max_count)),
            shading="auto",
        )
        mappable = mesh

        ax.scatter(
            pdata["x_plot"],
            pdata["y_plot"],
            s=5,
            alpha=0.12,
            edgecolors="none",
            color="white",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.plot(
            [xmin, xmax],
            [ymin, ymax],
            linestyle="--",
            linewidth=1.3,
            color="red",
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_title(pdata["display_name"], fontsize=16)
        ax.tick_params(axis="both", labelsize=12)

        frac_above = np.mean(pdata["y"] > pdata["x"]) if len(pdata["x"]) > 0 else np.nan
        frac_below = np.mean(pdata["y"] < pdata["x"]) if len(pdata["x"]) > 0 else np.nan

        ax.text(
            0.03,
            0.97,
            (
                f"Genes shown: {len(pdata['x']):,}\n"
                f"frac above x=y: {frac_above:.3g}\n"
                f"frac below x=y: {frac_below:.3g}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )
    axes[0].set_ylabel(f"PPI survival: P(r > {T_THRESHOLD})", fontsize=15)
    for ax in axes:
        ax.set_xlabel(f"Random survival: P(r > {T_THRESHOLD})", fontsize=15)
    fig.suptitle(
        (
            f"Gene-level PPI survival enrichment across preprocessing choices\n"
            f"with point overlay, threshold: r > {T_THRESHOLD}"
        ),
        fontsize=19,
        y=1.04,
    )
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Number of genes per bin, log color scale", fontsize=14)
    plt.tight_layout()
    png_out2 = os.path.join(
        OUTDIR,
        f"{N_PANELS}_panel_gene_survival_density_with_points_{suffix}.png",
    )
    svg_out2 = os.path.join(
        OUTDIR,
        f"{N_PANELS}_panel_gene_survival_density_with_points_{suffix}.svg",
    )
    pdf_out2 = os.path.join(
        OUTDIR,
        f"{N_PANELS}_panel_gene_survival_density_with_points_{suffix}.pdf",
    )
    plt.savefig(png_out2, dpi=300, bbox_inches="tight")
    plt.savefig(svg_out2, bbox_inches="tight")
    plt.savefig(pdf_out2, bbox_inches="tight")
    plt.show()
    print(f"\nSaved multi-panel overlay PNG: {png_out2}")
    print(f"Saved multi-panel overlay SVG: {svg_out2}")
    print(f"Saved multi-panel overlay PDF: {pdf_out2}")


def ppi_logratio_curves():
    BASE_OUTPUT_DIR = os.path.join(os.environ.get("SUPPL_OUT", "resources/repro/figS18"), "ppi")
    CONDITIONS = [
        {
            "key": "raw",
            "label": "Raw",
            "aliases": ["raw"],
        },
        {
            "key": "log1p",
            "label": "log1p",
            "aliases": ["log1p"],
        },
        {
            "key": "norm",
            "label": "Normalized",
            "aliases": ["norm", "normalized"],
        },
        {
            # This matches the current typo in your upstream script:
            # {"preprocess_name": "norm_plust_log1p", "log1p": True, "norm": True}
            "key": "norm_plust_log1p",
            "label": "Normalized + log1p",
            "aliases": [
                "norm_plust_log1p",
                "normpluslog1p",
                "norm_plus_log1p",
                "norm_log1p",
                "normlog1p",
                "normalized_log1p",
                "normalized_plus_log1p",
            ],
        },
    ]
    OUTDIR = "ppi_survival_logratio_curves_4types"
    os.makedirs(OUTDIR, exist_ok=True)
    T_GRID = np.linspace(0.0, 1.0, 100)
    MIN_SURVIVING_PAIRS = 10
    MIN_TOTAL_PAIRS = 500
    USE_ABS = False
    Z = 1.96
    ERRORBAR_EVERY = 5
    FIGSIZE = (9.5, 7.0)
    DPI = 300
    all_pair_dfs = {}
    all_curve_dfs = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]
        aliases = cond.get("aliases", [key])

        print("\n" + "=" * 80)
        print(f"Loading condition: {label} ({key})")
        print(f"Aliases: {aliases}")
        print("=" * 80)

        pair_df = load_condition_pairs(
            base_output_dir=BASE_OUTPUT_DIR,
            condition_key=key,
            aliases=aliases,
        )
        all_pair_dfs[key] = pair_df

        n_ppi = int(np.sum(pair_df["group"].eq("GGI")))
        n_rand = int(np.sum(pair_df["group"].eq("Random")))

        print(f"Loaded rows: {len(pair_df):,}")
        print(f"  PPI pairs:    {n_ppi:,}")
        print(f"  Random pairs: {n_rand:,}")
        print(f"  Datasets:     {pair_df['dataset'].nunique():,}")

        curve_df = compute_logratio_curve(
            pair_df=pair_df,
            t_grid=T_GRID,
            min_surviving_pairs=MIN_SURVIVING_PAIRS,
            min_total_pairs=MIN_TOTAL_PAIRS,
            use_abs=USE_ABS,
        )

        curve_df["condition"] = key
        curve_df["label"] = label
        all_curve_dfs.append(curve_df)
    curve_all = pd.concat(all_curve_dfs, axis=0, ignore_index=True)
    curve_csv = os.path.join(
        OUTDIR,
        f"ppi_survival_logratio_curves_4types_minTail{MIN_SURVIVING_PAIRS}.csv",
    )
    curve_all.to_csv(curve_csv, index=False)
    print(f"\nSaved curve table: {curve_csv}")
    plt.figure(figsize=FIGSIZE)
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]

        df = curve_all[
            (curve_all["condition"].eq(key)) &
            (curve_all["valid"])
        ].copy()

        if len(df) == 0:
            print(f"[WARN] No valid thresholds for {label}")
            continue

        x = df["t"].to_numpy()
        y = df["log_ratio"].to_numpy()
        lo = df["ci_low"].to_numpy()
        hi = df["ci_high"].to_numpy()

        plt.plot(
            x,
            y,
            linewidth=2.4,
            label=label,
        )

        plt.fill_between(
            x,
            lo,
            hi,
            alpha=0.18,
            linewidth=0,
        )

        # Sparse error bars
        err_idx = np.arange(0, len(df), ERRORBAR_EVERY)
        plt.errorbar(
            x[err_idx],
            y[err_idx],
            yerr=np.vstack([
                y[err_idx] - lo[err_idx],
                hi[err_idx] - y[err_idx],
            ]),
            fmt="none",
            capsize=2.5,
            linewidth=1.0,
            alpha=0.8,
        )
    plt.axhline(
        0,
        linestyle="--",
        linewidth=1.3,
        color="black",
        alpha=0.8,
    )
    xlabel = "|r| threshold t" if USE_ABS else "Pearson-r threshold t"
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(
        r"$\log\left[P_{\mathrm{PPI}}(r>t) / P_{\mathrm{rand}}(r>t)\right]$",
        fontsize=15,
    )
    plt.title(
        (
            "PPI enrichment in high-correlation tail\n"
            f"thresholds retained only if both groups have ≥ {MIN_SURVIVING_PAIRS} surviving pairs"
        ),
        fontsize=16,
    )
    plt.legend(frameon=True, fontsize=12)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    suffix = f"minTail{MIN_SURVIVING_PAIRS}"
    if USE_ABS:
        suffix += "_absr"
    else:
        suffix += "_signedr"
    png_out = os.path.join(OUTDIR, f"ppi_survival_logratio_4curves_{suffix}.png")
    svg_out = os.path.join(OUTDIR, f"ppi_survival_logratio_4curves_{suffix}.svg")
    pdf_out = os.path.join(OUTDIR, f"ppi_survival_logratio_4curves_{suffix}.pdf")
    plt.savefig(png_out, dpi=DPI, bbox_inches="tight")
    plt.savefig(svg_out, bbox_inches="tight")
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.show()
    print(f"\nSaved PNG: {png_out}")
    print(f"Saved SVG: {svg_out}")
    print(f"Saved PDF: {pdf_out}")
    SELECTED_T = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    summary_rows = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]
        df = curve_all[curve_all["condition"].eq(key)].copy()

        for t0 in SELECTED_T:
            idx = np.argmin(np.abs(df["t"].to_numpy() - t0))
            row = df.iloc[idx].to_dict()
            row["requested_t"] = t0
            row["condition_label"] = label
            summary_rows.append(row)
    selected_summary = pd.DataFrame(summary_rows)
    selected_summary = selected_summary[
        [
            "condition_label",
            "requested_t",
            "t",
            "valid",
            "k_ggi",
            "k_rand",
            "p_ggi",
            "p_rand",
            "log_ratio",
            "ci_low",
            "ci_high",
        ]
    ]
    selected_csv = os.path.join(OUTDIR, f"selected_threshold_summary_4types_{suffix}.csv")
    selected_summary.to_csv(selected_csv, index=False)
    print("\nSelected threshold summary:")
    print(selected_summary.to_string(index=False))
    print(f"\nSaved selected threshold summary: {selected_csv}")


def ppi_pair_mean_barscatter():
    BASE_OUTPUT_DIR = os.path.join(os.environ.get("SUPPL_OUT", "resources/repro/figS18"), "ppi")
    CONDITIONS = [
        {
            "key": "raw",
            "label": "Raw",
            "aliases": ["raw"],
        },
        {
            "key": "log1p",
            "label": "log1p",
            "aliases": ["log1p"],
        },
        {
            "key": "norm",
            "label": "Normalized",
            "aliases": ["norm", "normalized"],
        },
        {
            # Matches your current upstream typo:
            # preprocess_name = "norm_plust_log1p"
            "key": "norm_plust_log1p",
            "label": "Normalized + log1p",
            "aliases": [
                "norm_plust_log1p",
                "normpluslog1p",
                "norm_plus_log1p",
                "norm_log1p",
                "normlog1p",
                "normalized_log1p",
                "normalized_plus_log1p",
            ],
        },
    ]
    DATASETS = [
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]
    OUTDIR = "ppi_pair_mean_barscatter_4types"
    os.makedirs(OUTDIR, exist_ok=True)
    USE_ABS = False
    ERROR_KIND = "sem"
    JITTER = 0.055
    POINT_SIZE = 58
    LINE_ALPHA = 0.35
    DPI = 300
    rng = np.random.default_rng(0)
    rows = []
    missing_files = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]
        aliases = cond.get("aliases", [key])

        for dataset in DATASETS:
            dataset_clean = clean_dataset_name(dataset)

            fp = find_pair_file(
                base_output_dir=BASE_OUTPUT_DIR,
                condition_key=key,
                dataset_clean=dataset_clean,
                aliases=aliases,
            )

            if fp is None:
                missing_files.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "error": "file not found",
                })
                continue

            try:
                df = load_pair_file(fp)

                rand_vals = df.loc[df["group"].eq("Random"), "score"].to_numpy()
                ppi_vals = df.loc[df["group"].eq("GGI"), "score"].to_numpy()

                if len(rand_vals) == 0 or len(ppi_vals) == 0:
                    missing_files.append({
                        "condition": key,
                        "condition_label": label,
                        "dataset": dataset_clean,
                        "error": "missing Random or PPI rows",
                    })
                    continue

                rows.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "file": fp,
                    "mean_random": float(np.mean(rand_vals)),
                    "mean_ppi": float(np.mean(ppi_vals)),
                    "sem_random_pairs": float(np.std(rand_vals, ddof=1) / np.sqrt(len(rand_vals))),
                    "sem_ppi_pairs": float(np.std(ppi_vals, ddof=1) / np.sqrt(len(ppi_vals))),
                    "sd_random_pairs": float(np.std(rand_vals, ddof=1)),
                    "sd_ppi_pairs": float(np.std(ppi_vals, ddof=1)),
                    "n_random_pairs": int(len(rand_vals)),
                    "n_ppi_pairs": int(len(ppi_vals)),
                    "delta_ppi_minus_random": float(np.mean(ppi_vals) - np.mean(rand_vals)),
                })

            except Exception as e:
                missing_files.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "error": str(e),
                })
    summary_df = pd.DataFrame(rows)
    if len(summary_df) == 0:
        raise RuntimeError("No usable pair files found.")
    summary_csv = os.path.join(OUTDIR, "dataset_level_mean_random_vs_ppi_4types.csv")
    summary_df.to_csv(summary_csv, index=False)
    if missing_files:
        missing_df = pd.DataFrame(missing_files)
        missing_csv = os.path.join(OUTDIR, "missing_or_skipped_files_4types.csv")
        missing_df.to_csv(missing_csv, index=False)
        print(f"[saved] missing/skipped file log: {missing_csv}")
    print(f"[saved] dataset-level summary: {summary_csv}")
    print(summary_df.head().to_string(index=False))
    dataset_order = [clean_dataset_name(d) for d in DATASETS]
    dataset_order = [d for d in dataset_order if d in set(summary_df["dataset"])]
    cmap = plt.get_cmap("tab10")
    dataset_to_color = {
        d: cmap(i % 10)
        for i, d in enumerate(dataset_order)
    }
    N_CONDITIONS = len(CONDITIONS)
    fig, axes = plt.subplots(
        1,
        N_CONDITIONS,
        figsize=(5.1 * N_CONDITIONS, 5.8),
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    x_rand = 0
    x_ppi = 1
    for ax, cond in zip(axes, CONDITIONS):
        key = cond["key"]
        label = cond["label"]

        sub = summary_df[summary_df["condition"].eq(key)].copy()
        sub["dataset"] = pd.Categorical(sub["dataset"], categories=dataset_order, ordered=True)
        sub = sub.sort_values("dataset")

        # Dataset paired points
        for _, row in sub.iterrows():
            d = row["dataset"]
            color = dataset_to_color.get(str(d), "gray")

            jr = rng.normal(0, JITTER)
            jp = rng.normal(0, JITTER)

            y_rand = row["mean_random"]
            y_ppi = row["mean_ppi"]

            ax.plot(
                [x_rand + jr, x_ppi + jp],
                [y_rand, y_ppi],
                color=color,
                alpha=LINE_ALPHA,
                linewidth=1.4,
                zorder=1,
            )

            ax.scatter(
                x_rand + jr,
                y_rand,
                s=POINT_SIZE,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
                zorder=2,
            )

            ax.scatter(
                x_ppi + jp,
                y_ppi,
                s=POINT_SIZE,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
                zorder=2,
            )

        # Black grand mean across datasets
        mean_rand = sub["mean_random"].mean()
        mean_ppi = sub["mean_ppi"].mean()

        if ERROR_KIND == "sem":
            err_rand = sub["mean_random"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.0
            err_ppi = sub["mean_ppi"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.0
        elif ERROR_KIND == "sd":
            err_rand = sub["mean_random"].std(ddof=1) if len(sub) > 1 else 0.0
            err_ppi = sub["mean_ppi"].std(ddof=1) if len(sub) > 1 else 0.0
        else:
            raise ValueError("ERROR_KIND must be 'sem' or 'sd'")

        ax.errorbar(
            [x_rand, x_ppi],
            [mean_rand, mean_ppi],
            yerr=[err_rand, err_ppi],
            fmt="o",
            color="black",
            markersize=9,
            capsize=5,
            elinewidth=2.0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
            label=f"Mean ± {ERROR_KIND.upper()}",
        )

        ax.plot(
            [x_rand, x_ppi],
            [mean_rand, mean_ppi],
            color="black",
            linewidth=2.0,
            zorder=4,
        )

        # Optional faint bars behind the points
        ax.bar(
            [x_rand, x_ppi],
            [mean_rand, mean_ppi],
            width=0.48,
            color="black",
            alpha=0.10,
            zorder=0,
        )

        delta = mean_ppi - mean_rand
        frac_ppi_gt_rand = np.mean(sub["mean_ppi"].to_numpy() > sub["mean_random"].to_numpy()) if len(sub) > 0 else np.nan

        ax.text(
            0.5,
            0.97,
            (
                f"N datasets = {len(sub)}\n"
                f"mean Δ = {delta:.4g}\n"
                f"frac PPI>rand = {frac_ppi_gt_rand:.2g}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

        ax.set_title(label, fontsize=16)
        ax.set_xticks([x_rand, x_ppi])
        ax.set_xticklabels(["Random", "PPI"], fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(-0.55, 1.55)
    axes[0].set_ylabel("Mean |Pearson r|" if USE_ABS else "Mean Pearson r", fontsize=15)
    fig.suptitle(
        "Dataset-level mean PPI vs random gene-pair correlation",
        fontsize=18,
        y=1.03,
    )
    handles = []
    labels = []
    for d in dataset_order:
        h = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=dataset_to_color[d],
            markeredgecolor="black",
            markersize=8,
            label=d,
        )
        handles.append(h)
        labels.append(d)
    mean_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="black",
        markeredgecolor="white",
        markersize=9,
        linewidth=2,
        label=f"Mean ± {ERROR_KIND.upper()}",
    )
    handles.append(mean_handle)
    labels.append(f"Mean ± {ERROR_KIND.upper()}")
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=9,
    )
    plt.tight_layout()
    suffix = "absr" if USE_ABS else "signedr"
    png_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.png",
    )
    svg_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.svg",
    )
    pdf_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.pdf",
    )
    plt.savefig(png_out, dpi=DPI, bbox_inches="tight")
    plt.savefig(svg_out, bbox_inches="tight")
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.show()
    print(f"[saved] PNG: {png_out}")
    print(f"[saved] SVG: {svg_out}")
    print(f"[saved] PDF: {pdf_out}")
    stats_rows = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]

        sub = summary_df[summary_df["condition"].eq(key)].copy()

        diffs = sub["mean_ppi"].to_numpy() - sub["mean_random"].to_numpy()

        stats_rows.append({
            "condition": key,
            "label": label,
            "n_datasets": len(sub),
            "mean_random_across_datasets": sub["mean_random"].mean(),
            "mean_ppi_across_datasets": sub["mean_ppi"].mean(),
            "mean_delta_ppi_minus_random": np.mean(diffs),
            "sem_delta_across_datasets": np.std(diffs, ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else np.nan,
            "frac_datasets_ppi_greater_random": np.mean(diffs > 0) if len(diffs) > 0 else np.nan,
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = os.path.join(OUTDIR, "paired_dataset_level_stats_4types.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"[saved] paired stats: {stats_csv}")
    print(stats_df.to_string(index=False))


def ppi_pair_mean_barscatter_stats():
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except Exception:
        ttest_rel = None
        wilcoxon = None
    BASE_OUTPUT_DIR = os.path.join(os.environ.get("SUPPL_OUT", "resources/repro/figS18"), "ppi")
    CONDITIONS = [
        {
            "key": "raw",
            "label": "Raw",
            "aliases": ["raw"],
        },
        {
            "key": "log1p",
            "label": "log1p",
            "aliases": ["log1p"],
        },
        {
            "key": "norm",
            "label": "Normalized",
            "aliases": ["norm", "normalized"],
        },
        {
            # Matches your current upstream typo:
            # preprocess_name = "norm_plust_log1p"
            "key": "norm_plust_log1p",
            "label": "Normalized + log1p",
            "aliases": [
                "norm_plust_log1p",
                "normpluslog1p",
                "norm_plus_log1p",
                "norm_log1p",
                "normlog1p",
                "normalized_log1p",
                "normalized_plus_log1p",
            ],
        },
    ]
    DATASETS = [
        "ReplogleWeissman2022_rpe1.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "GSE264667_jurkat_raw_singlecell_01.h5ad",
        "GSE264667_hepg2_raw_singlecell_01.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "TianKampmann2019_day7neuron.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2019_iPSC.h5ad",
    ]
    OUTDIR = "ppi_pair_mean_barscatter_4types"
    os.makedirs(OUTDIR, exist_ok=True)
    USE_ABS = False
    ERROR_KIND = "sem"
    JITTER = 0.055
    POINT_SIZE = 58
    LINE_ALPHA = 0.35
    DPI = 300
    rng = np.random.default_rng(0)
    rows = []
    missing_files = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]
        aliases = cond.get("aliases", [key])

        for dataset in DATASETS:
            dataset_clean = clean_dataset_name(dataset)

            fp = find_pair_file(
                base_output_dir=BASE_OUTPUT_DIR,
                condition_key=key,
                dataset_clean=dataset_clean,
                aliases=aliases,
            )

            if fp is None:
                missing_files.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "error": "file not found",
                })
                continue

            try:
                df = load_pair_file(fp)

                rand_vals = df.loc[df["group"].eq("Random"), "score"].to_numpy()
                ppi_vals = df.loc[df["group"].eq("GGI"), "score"].to_numpy()

                if len(rand_vals) == 0 or len(ppi_vals) == 0:
                    missing_files.append({
                        "condition": key,
                        "condition_label": label,
                        "dataset": dataset_clean,
                        "error": "missing Random or PPI rows",
                    })
                    continue

                rows.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "file": fp,
                    "mean_random": float(np.mean(rand_vals)),
                    "mean_ppi": float(np.mean(ppi_vals)),
                    "sem_random_pairs": float(np.std(rand_vals, ddof=1) / np.sqrt(len(rand_vals))),
                    "sem_ppi_pairs": float(np.std(ppi_vals, ddof=1) / np.sqrt(len(ppi_vals))),
                    "sd_random_pairs": float(np.std(rand_vals, ddof=1)),
                    "sd_ppi_pairs": float(np.std(ppi_vals, ddof=1)),
                    "n_random_pairs": int(len(rand_vals)),
                    "n_ppi_pairs": int(len(ppi_vals)),
                    "delta_ppi_minus_random": float(np.mean(ppi_vals) - np.mean(rand_vals)),
                })

            except Exception as e:
                missing_files.append({
                    "condition": key,
                    "condition_label": label,
                    "dataset": dataset_clean,
                    "error": str(e),
                })
    summary_df = pd.DataFrame(rows)
    if len(summary_df) == 0:
        raise RuntimeError("No usable pair files found.")
    summary_csv = os.path.join(OUTDIR, "dataset_level_mean_random_vs_ppi_4types.csv")
    summary_df.to_csv(summary_csv, index=False)
    if missing_files:
        missing_df = pd.DataFrame(missing_files)
        missing_csv = os.path.join(OUTDIR, "missing_or_skipped_files_4types.csv")
        missing_df.to_csv(missing_csv, index=False)
        print(f"[saved] missing/skipped file log: {missing_csv}")
    print(f"[saved] dataset-level summary: {summary_csv}")
    print(summary_df.head().to_string(index=False))
    dataset_order = [clean_dataset_name(d) for d in DATASETS]
    dataset_order = [d for d in dataset_order if d in set(summary_df["dataset"])]
    cmap = plt.get_cmap("tab10")
    dataset_to_color = {
        d: cmap(i % 10)
        for i, d in enumerate(dataset_order)
    }
    N_CONDITIONS = len(CONDITIONS)
    fig, axes = plt.subplots(
        1,
        N_CONDITIONS,
        figsize=(5.1 * N_CONDITIONS, 5.8),
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)
    x_rand = 0
    x_ppi = 1
    for ax, cond in zip(axes, CONDITIONS):
        key = cond["key"]
        label = cond["label"]

        sub = summary_df[summary_df["condition"].eq(key)].copy()
        sub["dataset"] = pd.Categorical(sub["dataset"], categories=dataset_order, ordered=True)
        sub = sub.sort_values("dataset")

        # Dataset paired points
        for _, row in sub.iterrows():
            d = row["dataset"]
            color = dataset_to_color.get(str(d), "gray")

            jr = rng.normal(0, JITTER)
            jp = rng.normal(0, JITTER)

            y_rand = row["mean_random"]
            y_ppi = row["mean_ppi"]

            ax.plot(
                [x_rand + jr, x_ppi + jp],
                [y_rand, y_ppi],
                color=color,
                alpha=LINE_ALPHA,
                linewidth=1.4,
                zorder=1,
            )

            ax.scatter(
                x_rand + jr,
                y_rand,
                s=POINT_SIZE,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
                zorder=2,
            )

            ax.scatter(
                x_ppi + jp,
                y_ppi,
                s=POINT_SIZE,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
                zorder=2,
            )

        # Black grand mean across datasets
        mean_rand = sub["mean_random"].mean()
        mean_ppi = sub["mean_ppi"].mean()

        if ERROR_KIND == "sem":
            err_rand = sub["mean_random"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.0
            err_ppi = sub["mean_ppi"].std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else 0.0
        elif ERROR_KIND == "sd":
            err_rand = sub["mean_random"].std(ddof=1) if len(sub) > 1 else 0.0
            err_ppi = sub["mean_ppi"].std(ddof=1) if len(sub) > 1 else 0.0
        else:
            raise ValueError("ERROR_KIND must be 'sem' or 'sd'")

        ax.errorbar(
            [x_rand, x_ppi],
            [mean_rand, mean_ppi],
            yerr=[err_rand, err_ppi],
            fmt="o",
            color="black",
            markersize=9,
            capsize=5,
            elinewidth=2.0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
            label=f"Mean ± {ERROR_KIND.upper()}",
        )

        ax.plot(
            [x_rand, x_ppi],
            [mean_rand, mean_ppi],
            color="black",
            linewidth=2.0,
            zorder=4,
        )

        # Optional faint bars behind the points
        ax.bar(
            [x_rand, x_ppi],
            [mean_rand, mean_ppi],
            width=0.48,
            color="black",
            alpha=0.10,
            zorder=0,
        )

        delta = mean_ppi - mean_rand
        frac_ppi_gt_rand = np.mean(sub["mean_ppi"].to_numpy() > sub["mean_random"].to_numpy()) if len(sub) > 0 else np.nan

        paired_t_p = paired_ttest_pvalue(
            sub["mean_random"].to_numpy(),
            sub["mean_ppi"].to_numpy(),
        )

        paired_w_p = paired_wilcoxon_pvalue(
            sub["mean_random"].to_numpy(),
            sub["mean_ppi"].to_numpy(),
        )

        ax.text(
            0.5,
            0.97,
            (
                f"N datasets = {len(sub)}\n"
                f"black mean rand = {mean_rand:.4g}\n"
                f"black mean PPI = {mean_ppi:.4g}\n"
                f"paired t p = {format_pvalue(paired_t_p)}\n"
                f"Wilcoxon p = {format_pvalue(paired_w_p)}\n"
                f"mean Δ = {delta:.4g}\n"
                f"frac PPI>rand = {frac_ppi_gt_rand:.2g}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

        ax.set_title(label, fontsize=16)
        ax.set_xticks([x_rand, x_ppi])
        ax.set_xticklabels(["Random", "PPI"], fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(-0.55, 1.55)
    axes[0].set_ylabel("Mean |Pearson r|" if USE_ABS else "Mean Pearson r", fontsize=15)
    fig.suptitle(
        "Dataset-level mean PPI vs random gene-pair correlation",
        fontsize=18,
        y=1.03,
    )
    handles = []
    labels = []
    for d in dataset_order:
        h = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=dataset_to_color[d],
            markeredgecolor="black",
            markersize=8,
            label=d,
        )
        handles.append(h)
        labels.append(d)
    mean_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="black",
        markeredgecolor="white",
        markersize=9,
        linewidth=2,
        label=f"Mean ± {ERROR_KIND.upper()}",
    )
    handles.append(mean_handle)
    labels.append(f"Mean ± {ERROR_KIND.upper()}")
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=9,
    )
    plt.tight_layout()
    suffix = "absr" if USE_ABS else "signedr"
    png_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.png",
    )
    svg_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.svg",
    )
    pdf_out = os.path.join(
        OUTDIR,
        f"pair_mean_barscatter_raw_log1p_norm_normpluslog1p_{suffix}_{ERROR_KIND}.pdf",
    )
    plt.savefig(png_out, dpi=DPI, bbox_inches="tight")
    plt.savefig(svg_out, bbox_inches="tight")
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.show()
    print(f"[saved] PNG: {png_out}")
    print(f"[saved] SVG: {svg_out}")
    print(f"[saved] PDF: {pdf_out}")
    stats_rows = []
    for cond in CONDITIONS:
        key = cond["key"]
        label = cond["label"]

        sub = summary_df[summary_df["condition"].eq(key)].copy()

        diffs = sub["mean_ppi"].to_numpy() - sub["mean_random"].to_numpy()

        paired_t_p = paired_ttest_pvalue(
            sub["mean_random"].to_numpy(),
            sub["mean_ppi"].to_numpy(),
        )

        paired_w_p = paired_wilcoxon_pvalue(
            sub["mean_random"].to_numpy(),
            sub["mean_ppi"].to_numpy(),
        )

        stats_rows.append({
            "condition": key,
            "label": label,
            "n_datasets": len(sub),
            "mean_random_across_datasets": sub["mean_random"].mean(),
            "mean_ppi_across_datasets": sub["mean_ppi"].mean(),
            "mean_delta_ppi_minus_random": np.mean(diffs),
            "sem_delta_across_datasets": np.std(diffs, ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else np.nan,
            "frac_datasets_ppi_greater_random": np.mean(diffs > 0) if len(diffs) > 0 else np.nan,
            "paired_ttest_pvalue_ppi_vs_random": paired_t_p,
            "paired_wilcoxon_pvalue_ppi_vs_random": paired_w_p,
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = os.path.join(OUTDIR, "paired_dataset_level_stats_4types.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"[saved] paired stats: {stats_csv}")
    print(stats_df.to_string(index=False))
