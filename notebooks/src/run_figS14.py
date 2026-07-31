"""Run module for Fig S14 (covariance-PC top genes, panels A/B; forward held-out
Pearson vs response magnitude, panel C).

Notebook-only driver module: each function is one relocated main-flow cell from
``notebooks/suppl/figS14_pcs_dx_sigma.ipynb``, moved here VERBATIM so the notebook stays
a thin driver. NOT part of the installable ``cipher`` package.

Config (DATA_DIR, SUPPL, REPRO, REPO and any UPPERCASE globals) is injected into this
module's namespace at runtime from the notebook config cell; the functions read those as
module globals. Cross-section state (PRECOMPUTE_ROOT, PCS_TSV) is persisted as module
globals via ``global`` declarations. Per-section config that the original cells defined
inline stays inline.
"""
from src.suppl_pcs import *          # verbatim panels-A/B + panel-C helpers (not part of cipher)
import src.suppl_pcs as sp           # to override its path-valued config globals

import os, sys, re, gc, json, math, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.stats import linregress, spearmanr
from tqdm.auto import tqdm
try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None



def rebuild_sigma_layout():
    global PRECOMPUTE_ROOT, EXPRESSION_THRESHOLD
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "regenerate_sigma", os.path.join(os.path.dirname(os.path.abspath(__file__)), "regenerate_sigma.py")
    )
    regenerate_sigma = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(regenerate_sigma)

    SIGMA_ROOT = os.path.join(REPRO, "sigma_precompute")
    # builds <dataset>__mean_ge_1p0/sigmas/Sigma_full_ridge.npy + genes.npy for every *.h5ad in DATA_DIR
    regenerate_sigma.regenerate(DATA_DIR, SIGMA_ROOT)

    # point the pcs123 loader (panels A/B) at the regenerated layout
    sp.PRECOMPUTE_ROOT = Path(SIGMA_ROOT)
    sp.EXPRESSION_THRESHOLD = 1.0
    PRECOMPUTE_ROOT = sp.PRECOMPUTE_ROOT
    EXPRESSION_THRESHOLD = sp.EXPRESSION_THRESHOLD



def pcs123():
    global PCS_TSV
    # ============================================================
    # PANELS A/B OUTPUT PATHS (under the repro dir)
    # ============================================================

    OUTDIR = Path(REPRO) / "figS14AB" / "TOP3_COVARIANCE_PC_TOP_GENES"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    FIGDIR = OUTDIR / "figures"
    FIGDIR.mkdir(parents=True, exist_ok=True)

    ALL_LOADINGS_PATH = OUTDIR / "ALL_DATASETS__TOP25_GENES_PC1_PC2_PC3.tsv"
    PC_SUMMARY_PATH = OUTDIR / "ALL_DATASETS__PC1_PC2_PC3_SUMMARY.tsv"
    MULTIPAGE_PDF_PATH = OUTDIR / "ALL_DATASETS__TOP25_GENES_PC1_PC2_PC3.pdf"


    folders = find_dataset_folders()

    if len(folders) == 0:
        raise FileNotFoundError(
            f"No matching dataset folders under {PRECOMPUTE_ROOT}"
        )

    print(
        f"[run] found {len(folders)} datasets"
    )

    all_loading_tables = []
    all_summary_rows = []
    errors = []

    with PdfPages(
        MULTIPAGE_PDF_PATH
    ) as multipage_pdf:

        for folder in tqdm(
            folders,
            desc="datasets",
            ncols=110,
        ):
            try:
                dataset = clean_dataset_name(
                    folder
                )

                print("\n" + "=" * 100)
                print(f"DATASET: {dataset}")
                print("=" * 100)

                genes_path = (
                    folder / "genes.npy"
                )

                if not genes_path.exists():
                    raise FileNotFoundError(
                        f"Missing {genes_path}"
                    )

                sigma_path = find_sigma_path(
                    folder
                )

                genes = decode_str_array(
                    np.load(
                        genes_path,
                        allow_pickle=True,
                    )
                )

                Sigma = np.load(
                    sigma_path,
                    mmap_mode="r",
                )

                n_genes = len(genes)

                if Sigma.shape != (
                    n_genes,
                    n_genes,
                ):
                    raise ValueError(
                        f"Sigma shape={Sigma.shape}, "
                        f"but genes={n_genes:,}"
                    )

                total_variance = float(
                    np.sum(
                        np.asarray(
                            Sigma.diagonal(),
                            dtype=np.float64,
                        )
                    )
                )

                if (
                    not np.isfinite(
                        total_variance
                    )
                    or total_variance <= 0
                ):
                    raise ValueError(
                        f"Invalid covariance trace: "
                        f"{total_variance}"
                    )

                print(
                    f"[load] genes={n_genes:,}"
                )

                print(
                    f"[sigma] {sigma_path}"
                )

                eigenvalues, eigenvectors = (
                    compute_top_three_pcs(
                        Sigma
                    )
                )

                top_gene_table = (
                    make_top_gene_table(
                        dataset=dataset,
                        genes=genes,
                        eigenvalues=eigenvalues,
                        eigenvectors=eigenvectors,
                        total_variance=total_variance,
                        sigma_path=sigma_path,
                    )
                )

                all_loading_tables.append(
                    top_gene_table
                )

                summary_row = {
                    "dataset": dataset,
                    "n_genes": n_genes,
                    "sigma_file": str(
                        sigma_path
                    ),
                    "pc1_eigenvalue": float(
                        eigenvalues[0]
                    ),
                    "pc2_eigenvalue": float(
                        eigenvalues[1]
                    ),
                    "pc3_eigenvalue": float(
                        eigenvalues[2]
                    ),
                    "pc1_variance_percent": float(
                        100.0
                        * eigenvalues[0]
                        / total_variance
                    ),
                    "pc2_variance_percent": float(
                        100.0
                        * eigenvalues[1]
                        / total_variance
                    ),
                    "pc3_variance_percent": float(
                        100.0
                        * eigenvalues[2]
                        / total_variance
                    ),
                    "top3_variance_percent": float(
                        100.0
                        * np.sum(
                            eigenvalues[:3]
                        )
                        / total_variance
                    ),
                }

                all_summary_rows.append(
                    summary_row
                )

                # Print the top genes for quick inspection.
                for pc_number in range(
                    1,
                    4,
                ):
                    pc_top = (
                        top_gene_table.loc[
                            top_gene_table["pc"]
                            == pc_number
                        ]
                        .head(15)
                    )

                    gene_text = ", ".join(
                        (
                            f"{row.gene}"
                            f" ({row.loading:+.3f})"
                        )
                        for row in pc_top.itertuples()
                    )

                    print(
                        f"[PC{pc_number}] "
                        f"{gene_text}"
                    )

                figure = plot_dataset(
                    dataset=dataset,
                    top_gene_table=top_gene_table,
                    eigenvalues=eigenvalues,
                    total_variance=total_variance,
                    sigma_path=sigma_path,
                )

                safe_dataset = safe_filename(
                    dataset
                )

                png_path = (
                    FIGDIR
                    / f"{safe_dataset}"
                    "__TOP25_GENES_PC1_PC2_PC3.png"
                )

                pdf_path = (
                    FIGDIR
                    / f"{safe_dataset}"
                    "__TOP25_GENES_PC1_PC2_PC3.pdf"
                )

                svg_path = (
                    FIGDIR
                    / f"{safe_dataset}"
                    "__TOP25_GENES_PC1_PC2_PC3.svg"
                )

                figure.savefig(
                    png_path,
                    dpi=DPI,
                    bbox_inches="tight",
                )

                figure.savefig(
                    pdf_path,
                    bbox_inches="tight",
                )

                figure.savefig(
                    svg_path,
                    bbox_inches="tight",
                )

                multipage_pdf.savefig(
                    figure,
                    bbox_inches="tight",
                )

                if SHOW_FIGURES:
                    plt.show()

                plt.close(
                    figure
                )

                print(
                    f"[saved] {png_path}"
                )

                del Sigma
                del eigenvectors
                gc.collect()

            except Exception as error:
                print(
                    f"\n[ERROR] {folder}"
                )

                print(
                    repr(error)
                )

                errors.append(
                    {
                        "folder": str(folder),
                        "error": repr(error),
                    }
                )

                gc.collect()


    # ============================================================
    # SAVE COMBINED TABLES
    # ============================================================

    if len(all_loading_tables) == 0:
        raise RuntimeError(
            "No datasets completed successfully."
        )

    all_loadings = pd.concat(
        all_loading_tables,
        axis=0,
        ignore_index=True,
    )

    all_loadings.to_csv(
        ALL_LOADINGS_PATH,
        sep="\t",
        index=False,
    )

    pc_summary = pd.DataFrame(
        all_summary_rows
    )

    pc_summary.to_csv(
        PC_SUMMARY_PATH,
        sep="\t",
        index=False,
    )


    # ============================================================
    # DONE
    # ============================================================

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        f"Successful datasets: "
        f"{len(all_summary_rows)}"
    )

    print(
        f"Failed datasets:     "
        f"{len(errors)}"
    )

    print(
        f"Figures:             "
        f"{FIGDIR}"
    )

    print(
        f"Multipage PDF:       "
        f"{MULTIPAGE_PDF_PATH}"
    )

    print(
        f"Top-gene table:      "
        f"{ALL_LOADINGS_PATH}"
    )

    print(
        f"PC summary:          "
        f"{PC_SUMMARY_PATH}"
    )

    if errors:
        print("\nErrors:")

        for error in errors:
            print(
                f"- {error['folder']}"
            )
            print(
                f"  {error['error']}"
            )

    print("=" * 100)
    PCS_TSV = ALL_LOADINGS_PATH



def figure_celltype_curated():
    # ============================================================
    # PUBLICATION-READY 5-PANEL FIGURE:
    # CELL-TYPE-SPECIFIC COVARIANCE PCs
    #
    # Automatically locates:
    #   ALL_DATASETS__TOP25_GENES_PC1_PC2_PC3.tsv
    #
    # Panels:
    #   A. HepG2 PC2        — hepatocyte secretory program
    #   B. Activated T PC2 — effector cytokine program
    #   C. Neuron PC3      — neuronal identity program
    #   D. Fibroblast PC3  — matrix / contractile program
    #   E. Melanoma PC3    — inflammatory / remodeling program
    #
    # Bars show absolute loading normalized within each panel.
    # Eigenvector sign is arbitrary, so absolute loading is used.
    #
    # Saves:
    #   CELL_TYPE_SPECIFIC_PC_SUMMARY/
    #       CELL_TYPE_SPECIFIC_COVARIANCE_PCS.png
    #       CELL_TYPE_SPECIFIC_COVARIANCE_PCS.pdf
    #       CELL_TYPE_SPECIFIC_COVARIANCE_PCS.svg
    # ============================================================

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec


    # ============================================================
    # CONFIG
    # ============================================================

    INPUT_FILENAME = "ALL_DATASETS__TOP25_GENES_PC1_PC2_PC3.tsv"

    # Leave as None to search automatically.
    # Alternatively, provide an explicit path:
    #
    # TSV_PATH = Path(
    #     "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED/"
    #     "TOP3_COVARIANCE_PC_TOP_GENES/"
    #     "ALL_DATASETS__TOP25_GENES_PC1_PC2_PC3.tsv"
    # )
    TSV_PATH = PCS_TSV

    OUTDIR = Path(REPRO) / "figS14AB" / "CELL_TYPE_SPECIFIC_PC_SUMMARY"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    OUTBASE = OUTDIR / "CELL_TYPE_SPECIFIC_COVARIANCE_PCS"

    DPI = 400


    # ============================================================
    # REPRESENTATIVE BIOLOGICAL PROGRAMS
    #
    # Every listed gene comes from that dataset-PC's saved top-25
    # loading table. Broad housekeeping genes are omitted to make
    # the biological programs visually clear.
    # ============================================================

    PANELS = [
        {
            "dataset": "GSE264667_hepg2_raw_singlecell_01",
            "pc": 2,
            "title": "HepG2",
            "subtitle": "Hepatocyte secretory program",
            "genes": [
                "ALB",
                "AFP",
                "APOA1",
                "APOB",
                "SERPINA1",
                "APOE",
                "APOH",
                "APOA2",
            ],
        },
        {
            "dataset": "Marson2025_D1_Stim8hr_filtered",
            "pc": 2,
            "title": "Activated T cells",
            "subtitle": "Effector cytokine program",
            "genes": [
                "IFNG",
                "IL3",
                "CCL4",
                "CSF2",
                "GZMB",
                "IL2",
                "CCL1",
                "CD40LG",
            ],
        },
        {
            "dataset": "TianKampmann2021_CRISPRi",
            "pc": 3,
            "title": "Neurons",
            "subtitle": "Neuronal identity program",
            "genes": [
                "NEFM",
                "NEFL",
                "VGF",
                "NNAT",
                "MAP1B",
                "PBX1",
                "BEX1",
                "ISL1",
            ],
        },
        {
            "dataset": "FrangiehIzar2021_RNA",
            "pc": 3,
            "title": "Melanoma",
            "subtitle": "Inflammatory and remodeling state",
            "genes": [
                "TIMP1",
                "S100A6",
                "MT2A",
                "S100A9",
                "IL1B",
                "MSMP",
                "S100A8",
                "TFPI2",
            ],
        },
    ]

    PANEL_COLORS = [
        "#4C78A8",  # HepG2
        "#E45756",  # T cells
        "#7A5195",  # neurons
        "#F28E2B",  # melanoma
    ]


    # ============================================================
    # AUTOMATICALLY LOCATE INPUT TABLE
    # ============================================================

    def find_input_table(explicit_path=None):
        if explicit_path is not None:
            explicit_path = Path(explicit_path).expanduser()

            if explicit_path.exists():
                return explicit_path.resolve()

            raise FileNotFoundError(
                "TSV_PATH was explicitly set, but the file does not exist:\n"
                f"{explicit_path}"
            )

        cwd = Path.cwd()

        direct_candidates = [
            cwd / INPUT_FILENAME,

            cwd
            / "TOP3_COVARIANCE_PC_TOP_GENES"
            / INPUT_FILENAME,

            cwd
            / "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED"
            / "TOP3_COVARIANCE_PC_TOP_GENES"
            / INPUT_FILENAME,

            cwd
            / "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED"
            / INPUT_FILENAME,

            Path("/mnt/data") / INPUT_FILENAME,
        ]

        for candidate in direct_candidates:
            if candidate.exists():
                return candidate.resolve()

        # Final fallback: recursively search the current project.
        matches = sorted(
            cwd.rglob(INPUT_FILENAME),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if matches:
            return matches[0].resolve()

        searched = "\n".join(
            f"  - {path}"
            for path in direct_candidates
        )

        raise FileNotFoundError(
            f"Could not locate {INPUT_FILENAME}.\n\n"
            f"Searched:\n{searched}\n\n"
            "Set TSV_PATH near the top of the script to the actual file location."
        )


    input_path = find_input_table(TSV_PATH)
    print(f"[load] {input_path}")


    # ============================================================
    # LOAD AND VALIDATE
    # ============================================================

    df = pd.read_csv(input_path, sep="\t")

    required_columns = {
        "dataset",
        "pc",
        "gene",
        "loading",
        "absolute_loading",
        "variance_percent",
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input table is missing required columns: "
            + ", ".join(missing_columns)
        )

    df["pc"] = pd.to_numeric(
        df["pc"],
        errors="coerce",
    )

    df["loading"] = pd.to_numeric(
        df["loading"],
        errors="coerce",
    )

    df["absolute_loading"] = pd.to_numeric(
        df["absolute_loading"],
        errors="coerce",
    )

    df["variance_percent"] = pd.to_numeric(
        df["variance_percent"],
        errors="coerce",
    )


    # ============================================================
    # PREPARE PANEL DATA
    # ============================================================

    def prepare_panel(panel):
        subset = df.loc[
            (df["dataset"] == panel["dataset"])
            & (df["pc"] == panel["pc"])
        ].copy()

        if subset.empty:
            available = sorted(
                df["dataset"]
                .dropna()
                .unique()
            )

            raise ValueError(
                f"No rows found for:\n"
                f"  dataset = {panel['dataset']}\n"
                f"  PC      = {panel['pc']}\n\n"
                "Available datasets:\n"
                + "\n".join(
                    f"  - {name}"
                    for name in available
                )
            )

        subset = subset.drop_duplicates(
            subset=["gene"],
            keep="first",
        )

        subset = subset.set_index(
            "gene",
            drop=False,
        )

        found_genes = [
            gene
            for gene in panel["genes"]
            if gene in subset.index
        ]

        missing_genes = [
            gene
            for gene in panel["genes"]
            if gene not in subset.index
        ]

        if missing_genes:
            print(
                f"[warning] {panel['title']} PC{panel['pc']}: "
                "genes absent from saved top-25 table: "
                + ", ".join(missing_genes)
            )

        if len(found_genes) < 4:
            raise ValueError(
                f"Too few requested genes were found for "
                f"{panel['title']} PC{panel['pc']}.\n"
                f"Found: {found_genes}"
            )

        selected = subset.loc[
            found_genes
        ].copy()

        if isinstance(selected, pd.Series):
            selected = selected.to_frame().T

        selected["relative_loading"] = (
            selected["absolute_loading"]
            / selected["absolute_loading"].max()
        )

        # Ascending order places the largest bar at the top
        # of the horizontal bar chart.
        selected = selected.sort_values(
            "relative_loading",
            ascending=True,
        )

        variance_percent = float(
            subset["variance_percent"]
            .dropna()
            .iloc[0]
        )

        return selected, variance_percent


    prepared_panels = [
        prepare_panel(panel)
        for panel in PANELS
    ]


    # ============================================================
    # PUBLICATION STYLE
    # ============================================================

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,

            # Keep text editable in vector files.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


    # ============================================================
    # FIGURE LAYOUT
    #
    # Top row:     A, B, C
    # Bottom row:  D, E centered
    # ============================================================

    fig = plt.figure(
        figsize=(12.6, 7.1)
    )

    grid = GridSpec(
        2,
        6,
        figure=fig,
        height_ratios=[1, 1],
        hspace=0.66,
        wspace=1.15,
    )

    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]


    # ============================================================
    # DRAW PANELS
    # ============================================================

    for index, (
        ax,
        panel,
        prepared,
        color,
    ) in enumerate(
        zip(
            axes,
            PANELS,
            prepared_panels,
            PANEL_COLORS,
        )
    ):
        selected, variance_percent = prepared

        y_positions = np.arange(
            len(selected)
        )

        ax.barh(
            y_positions,
            selected["relative_loading"],
            height=0.68,
            color=color,
            edgecolor="none",
        )

        ax.set_yticks(
            y_positions
        )

        ax.set_yticklabels(
            selected["gene"],
            fontweight="semibold",
        )

        ax.set_xlim(
            0,
            1.04,
        )

        ax.set_xticks(
            [0, 0.5, 1.0]
        )

        ax.set_xlabel(
            "Relative absolute loading"
        )

        # Separate title and subtitle so only the cell type is bold.
        ax.text(
            0.5,
            1.14,
            panel["title"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
        )

        ax.text(
            0.5,
            1.045,
            (
                f"{panel['subtitle']}  |  "
                f"PC{panel['pc']}: {variance_percent:.1f}%"
            ),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
        )

        ax.grid(
            axis="x",
            color="0.90",
            linewidth=0.7,
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("0.35")

        ax.tick_params(
            axis="y",
            length=0,
            pad=3,
        )

        ax.tick_params(
            axis="x",
            length=3,
            color="0.35",
        )

        # Panel letter.
        ax.text(
            -0.18,
            1.16,
            chr(65 + index),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="top",
        )


    # ============================================================
    # GLOBAL TITLE AND NOTE
    # ============================================================

    fig.suptitle(
        "Subleading covariance modes recover cell-type-specific programs",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.018,
        (
            "Bars show absolute gene loadings normalized within each panel. "
            "Genes were selected from the saved top-25 loadings to emphasize "
            "the interpretable biological program; eigenvector sign is arbitrary."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.88,
        bottom=0.10,
    )


    # ============================================================
    # SAVE
    # ============================================================

    png_path = OUTBASE.with_suffix(".png")
    pdf_path = OUTBASE.with_suffix(".pdf")
    svg_path = OUTBASE.with_suffix(".svg")

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    print("\nSaved publication-ready figure:")
    print(f"  PNG: {png_path.resolve()}")
    print(f"  PDF: {pdf_path.resolve()}")
    print(f"  SVG: {svg_path.resolve()}")



def figure_celltype_all25():
    # ============================================================
    # PUBLICATION-READY 5-PANEL FIGURE OF CELL-TYPE-SPECIFIC PCs
    #
    # REQUIREMENTS IMPLEMENTED
    # ------------------------
    # 1. SHOW ALL 25 GENES for each selected dataset-PC
    # 2. NO RENORMALIZATION of the loadings
    # 3. Color genes either:
    #      - grey = not part of the highlighted cell-type/state program
    #      - colored = part of the highlighted program
    # 4. Use ABSOLUTE LOADINGS (not signed loadings), because the sign
    #    of an eigenvector is arbitrary
    #
    # INPUT
    # -----
    # Automatically locates:
    #   ALL_DATASETS__TOP25_GENES_PC1_PC2_PC3.tsv
    #
    # OUTPUT
    # ------
    # CELL_TYPE_SPECIFIC_PC_SUMMARY/
    #   CELL_TYPE_SPECIFIC_COVARIANCE_PCS_ALL25.png
    #   CELL_TYPE_SPECIFIC_COVARIANCE_PCS_ALL25.pdf
    #   CELL_TYPE_SPECIFIC_COVARIANCE_PCS_ALL25.svg
    #
    # PANELS
    # ------
    # A. HepG2 PC2        — hepatocyte secretory program
    # B. Activated T PC2 — effector cytokine program
    # C. Neuron PC3      — neuronal identity program
    # D. Fibroblast PC3  — matrix / contractile program
    # E. Melanoma PC3    — inflammatory / remodeling program
    # ============================================================

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch


    # ============================================================
    # CONFIG
    # ============================================================

    INPUT_FILENAME = "ALL_DATASETS__TOP25_GENES_PC1_PC2_PC3.tsv"

    # Set this manually if you want; otherwise the script searches.
    TSV_PATH = PCS_TSV

    OUTDIR = Path(REPRO) / "figS14AB" / "CELL_TYPE_SPECIFIC_PC_SUMMARY"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    OUTBASE = OUTDIR / "CELL_TYPE_SPECIFIC_COVARIANCE_PCS_ALL25"

    DPI = 400

    # Representative examples
    PANELS = [
        {
            "dataset": "GSE264667_hepg2_raw_singlecell_01",
            "pc": 2,
            "title": "HepG2",
            "subtitle": "Hepatocyte secretory program",
            "program_genes": {
                "ALB", "AFP", "APOA1", "APOB", "SERPINA1",
                "APOE", "APOH", "APOA2", "TF", "AHSG",
                "RBP4", "APOC3",
            },
            "color": "#4C78A8",
        },
        {
            "dataset": "Marson2025_D1_Stim8hr_filtered",
            "pc": 2,
            "title": "Activated T cells",
            "subtitle": "Effector cytokine program",
            "program_genes": {
                "IFNG", "IL3", "CCL4", "CSF2", "GZMB",
                "IL2", "CCL1", "CD40LG", "CCL3",
                "TNFRSF9", "IL13", "CXCL8",
            },
            "color": "#E45756",
        },
        {
            "dataset": "TianKampmann2021_CRISPRi",
            "pc": 3,
            "title": "Neurons",
            "subtitle": "Neuronal identity program",
            "program_genes": {
                "NEFM", "NEFL", "VGF", "NNAT", "MAP1B",
                "PBX1", "BEX1", "ISL1", "ATP1B1",
            },
            "color": "#7A5195",
        },
        {
            "dataset": "kaden25_fibroblast_subsampled",
            "pc": 3,
            "title": "Fibroblasts",
            "subtitle": "Matrix and contractile state",
            "program_genes": {
                "IGFBP5", "ALDH1A1", "TAGLN", "CAV1", "PRDX6",
                "CALD1", "COL1A1", "KRT19",
            },
            "color": "#59A14F",
        },
        {
            "dataset": "FrangiehIzar2021_RNA",
            "pc": 3,
            "title": "Melanoma",
            "subtitle": "Inflammatory and remodeling state",
            "program_genes": {
                "TIMP1", "S100A6", "MT2A", "S100A9", "IL1B",
                "MSMP", "S100A8", "TFPI2", "MMP1", "LGALS1",
                "VIM",
            },
            "color": "#F28E2B",
        },
    ]

    GREY_COLOR = "#C8C8C8"


    # ============================================================
    # FIND INPUT TABLE
    # ============================================================

    def find_input_table(explicit_path=None):
        if explicit_path is not None:
            explicit_path = Path(explicit_path).expanduser()
            if explicit_path.exists():
                return explicit_path.resolve()
            raise FileNotFoundError(
                "TSV_PATH was explicitly set, but does not exist:\n"
                f"{explicit_path}"
            )

        cwd = Path.cwd()

        candidates = [
            cwd / INPUT_FILENAME,
            cwd / "TOP3_COVARIANCE_PC_TOP_GENES" / INPUT_FILENAME,
            cwd / "precomputed_FULL_COV_FAST_FULLLOAD_CHUNKED" / "TOP3_COVARIANCE_PC_TOP_GENES" / INPUT_FILENAME,
            Path("/mnt/data") / INPUT_FILENAME,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        recursive_matches = sorted(
            cwd.rglob(INPUT_FILENAME),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if recursive_matches:
            return recursive_matches[0].resolve()

        raise FileNotFoundError(
            f"Could not locate {INPUT_FILENAME}.\n"
            "Set TSV_PATH manually near the top of this script."
        )


    # ============================================================
    # LOAD AND VALIDATE
    # ============================================================

    input_path = find_input_table(TSV_PATH)
    print(f"[load] {input_path}")

    df = pd.read_csv(input_path, sep="\t")

    required_columns = {
        "dataset",
        "pc",
        "gene",
        "loading",
        "absolute_loading",
        "variance_percent",
    }

    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Input table is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Numeric cleanup
    df["pc"] = pd.to_numeric(df["pc"], errors="coerce")
    df["loading"] = pd.to_numeric(df["loading"], errors="coerce")
    df["absolute_loading"] = pd.to_numeric(df["absolute_loading"], errors="coerce")
    df["variance_percent"] = pd.to_numeric(df["variance_percent"], errors="coerce")

    # Keep rank if present, otherwise recreate from absolute loading
    if "rank" not in df.columns:
        df = df.sort_values(
            ["dataset", "pc", "absolute_loading"],
            ascending=[True, True, False],
        ).copy()
        df["rank"] = (
            df.groupby(["dataset", "pc"]).cumcount() + 1
        )

    # Limit to top 25 in case the table contains more
    df = df.loc[df["rank"] <= 25].copy()


    # ============================================================
    # PREPARE PANEL DATA
    # ============================================================

    def prepare_panel(panel):
        subset = df.loc[
            (df["dataset"] == panel["dataset"])
            & (df["pc"] == panel["pc"])
        ].copy()

        if subset.empty:
            # Dataset not present in the PC table (e.g. not staged locally). Return a
            # placeholder so the A-E panel layout still matches the reference figure.
            print(f"[placeholder] {panel['title']} (dataset '{panel['dataset']}' unavailable)")
            return None, None

        # Make sure we have all 25 genes
        subset = subset.sort_values(
            "absolute_loading",
            ascending=False,
        ).head(25).copy()

        if len(subset) != 25:
            print(
                f"[warning] {panel['title']} PC{panel['pc']} has "
                f"{len(subset)} genes instead of 25 in the TSV."
            )

        subset["is_program_gene"] = subset["gene"].isin(panel["program_genes"])

        # For horizontal bar plot: smallest at bottom, largest at top
        subset = subset.sort_values(
            "absolute_loading",
            ascending=True,
        ).reset_index(drop=True)

        variance_percent = float(subset["variance_percent"].iloc[0])

        highlighted_found = subset.loc[
            subset["is_program_gene"], "gene"
        ].tolist()

        print("\n" + "=" * 100)
        print(f"{panel['title']} | dataset={panel['dataset']} | PC{panel['pc']}")
        print(f"Subtitle: {panel['subtitle']}")
        print(f"Variance explained: {variance_percent:.2f}%")
        print(f"Highlighted program genes found ({len(highlighted_found)}): {highlighted_found}")

        return subset, variance_percent


    prepared_panels = [prepare_panel(panel) for panel in PANELS]


    # ============================================================
    # STYLE
    # ============================================================

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


    # ============================================================
    # GLOBAL X LIMIT
    #
    # IMPORTANT:
    # No renormalization. We use the real absolute loadings.
    # To compare panels fairly, share a global x-axis maximum.
    # ============================================================

    global_xmax = max(
        float(panel_df["absolute_loading"].max())
        for panel_df, _ in prepared_panels
        if panel_df is not None
    )
    global_xmax *= 1.06


    # ============================================================
    # FIGURE LAYOUT
    # Top row:    A, B, C
    # Bottom row: D, E centered
    # ============================================================

    fig = plt.figure(figsize=(13.5, 11.0))
    grid = GridSpec(
        2,
        6,
        figure=fig,
        height_ratios=[1, 1],
        hspace=0.55,
        wspace=1.20,
    )

    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]


    # ============================================================
    # DRAW PANELS
    # ============================================================

    for i, (ax, panel, prepared) in enumerate(zip(axes, PANELS, prepared_panels)):
        panel_df, variance_percent = prepared

        if panel_df is None:
            # Data-unavailable placeholder: keep the panel letter + title so the A-E
            # layout matches the reference even when this dataset is not staged locally.
            ax.text(-0.18, 1.08, chr(65 + i), transform=ax.transAxes, fontsize=14,
                    fontweight="bold", ha="left", va="bottom")
            ax.set_title(
                f"{panel['title']}\n{panel['subtitle']}  |  PC{panel['pc']}",
                pad=10, fontweight="bold",
            )
            ax.text(0.5, 0.5, "data unavailable\n(dataset not staged locally)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="0.45", style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            for _spine in ax.spines.values():
                _spine.set_visible(False)
            continue

        y = np.arange(len(panel_df))

        colors = [
            panel["color"] if is_program else GREY_COLOR
            for is_program in panel_df["is_program_gene"]
        ]

        ax.barh(
            y,
            panel_df["absolute_loading"],
            color=colors,
            edgecolor="none",
            height=0.72,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(panel_df["gene"])

        # Make highlighted genes bold
        for tick_label, is_program in zip(ax.get_yticklabels(), panel_df["is_program_gene"]):
            if is_program:
                tick_label.set_fontweight("bold")

        ax.set_xlim(0, global_xmax)
        ax.set_xlabel("Absolute loading")
        ax.set_title(
            f"{panel['title']}\n"
            f"{panel['subtitle']}  |  PC{panel['pc']}: {variance_percent:.1f}%",
            pad=10,
            fontweight="bold",
        )

        ax.grid(axis="x", color="0.90", linewidth=0.7)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("0.35")

        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", length=3, color="0.35")

        # Panel letter
        ax.text(
            -0.18,
            1.08,
            chr(65 + i),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        # Small legend inside each panel
        legend_handles = [
            Patch(facecolor=panel["color"], edgecolor="none", label="Highlighted program gene"),
            Patch(facecolor=GREY_COLOR, edgecolor="none", label="Other top-loading gene"),
        ]
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=7.5,
            loc="lower right",
        )


    # ============================================================
    # TITLE + CAPTION NOTE
    # ============================================================

    fig.suptitle(
        "Subleading covariance modes contain cell-type-specific transcriptional programs",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.025,
        (
            "For each selected dataset, all top 25 genes from the indicated covariance principal component are shown. "
            "Bars represent the original absolute loadings from the saved PC table (no renormalization). "
            "Colored genes mark the interpretable cell-type or state-specific program within that PC; grey genes are the remaining top-loading genes. "
            "Because eigenvector sign is arbitrary, absolute loadings are shown."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.91,
        bottom=0.09,
    )


    # ============================================================
    # SAVE
    # ============================================================

    png_path = OUTBASE.with_suffix(".png")
    pdf_path = OUTBASE.with_suffix(".pdf")
    svg_path = OUTBASE.with_suffix(".svg")

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    print("\nSaved figure:")
    print(f"  PNG: {png_path.resolve()}")
    print(f"  PDF: {pdf_path.resolve()}")
    print(f"  SVG: {svg_path.resolve()}")



def forward_dx_main():
    # point the forward-dx loader + outputs at SUPPL / the repro dir
    sp.PRECOMPUTE_ROOT_CANDIDATES = [
        Path(os.path.join(SUPPL, "precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_SAFE_mean_control_ge_1p0")),
        Path(os.path.join(SUPPL, "precomputed_FORWARD_DX_SIGMA_ALL_NORMALIZATIONS_mean_control_ge_1p0")),
    ]
    sp.OUTDIR = Path(REPRO) / "figS14C" / "cipher_raw_pearson_vs_dx_magnitude"
    sp.PLOT_DIR = sp.OUTDIR / "plots"
    sp.PER_DATASET_PLOT_DIR = sp.PLOT_DIR / "per_dataset"
    DX_OUT = sp.OUTDIR

    main()



def panelC_r2_spearman():
    # ============================================================
    # RAW CIPHER:
    # HELD-OUT PEARSON VS WITHIN-DATASET RESPONSE STRENGTH
    #
    # Reports across ALL individual perturbations:
    #   - Linear-regression R²
    #   - Linear-regression slope and p-value
    #   - Spearman rho and p-value
    #   - Dataset-clustered slope p-value
    #
    # Black points:
    #   Dataset-balanced decile means ± SEM across datasets.
    #
    # Automatically searches recursively for:
    #   raw_pearson_vs_dx_per_perturbation.tsv
    #
    # Outputs:
    #   raw_individual_perturbation_statistics.tsv
    #   raw_dataset_balanced_decile_points.tsv
    #   raw_pearson_vs_response_strength_R2_spearman.png
    #   raw_pearson_vs_response_strength_R2_spearman.pdf
    #   raw_pearson_vs_response_strength_R2_spearman.svg
    # ============================================================

    from pathlib import Path
    import warnings

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from scipy.stats import linregress, spearmanr
    import statsmodels.api as sm


    # ============================================================
    # CONFIG
    # ============================================================

    SEARCH_ROOT = Path(REPRO) / "figS14C"

    INPUT_FILENAME = "raw_pearson_vs_dx_per_perturbation.tsv"

    OUTDIR = Path(REPRO) / "figS14C" / "raw_pearson_vs_response_strength_R2_spearman"

    N_BINS = 10

    DPI = 300

    POINT_SIZE = 13
    POINT_ALPHA = 0.10

    SHOW_FIGURE = True


    # ============================================================
    # CREATE OUTPUT DIRECTORY
    # ============================================================

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ============================================================
    # LOCATE INPUT FILE
    # ============================================================

    preferred_paths = [
        SEARCH_ROOT
        / INPUT_FILENAME,

        SEARCH_ROOT
        / "cipher_raw_pearson_vs_dx_magnitude"
        / INPUT_FILENAME,

        SEARCH_ROOT
        / "raw_pearson_vs_dx_magnitude"
        / INPUT_FILENAME,

        SEARCH_ROOT
        / "raw_pearson_vs_response_strength_individual_R2"
        / INPUT_FILENAME,
    ]

    INPUT_PATH = None

    for candidate in preferred_paths:
        if candidate.exists():
            INPUT_PATH = candidate.resolve()
            break


    if INPUT_PATH is None:
        matches = sorted(
            [
                path.resolve()
                for path in SEARCH_ROOT.rglob(INPUT_FILENAME)
                if path.is_file()
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if len(matches) == 0:
            raise FileNotFoundError(
                "\nCould not locate:\n"
                f"    {INPUT_FILENAME}\n\n"
                "Searched recursively under:\n"
                f"    {SEARCH_ROOT.resolve()}\n\n"
                "Run the raw CIPHER Pearson-versus-response-strength "
                "analysis first, or change SEARCH_ROOT."
            )

        INPUT_PATH = matches[0]

        if len(matches) > 1:
            print(
                "\n[WARNING] Multiple matching files were found."
            )

            print(
                "Using the most recently modified file:"
            )

            print(
                f"    {INPUT_PATH}"
            )

            print(
                "\nOther matching files:"
            )

            for alternative in matches[1:]:
                print(
                    f"    {alternative}"
                )


    print(
        "\n" + "=" * 90
    )

    print(
        "LOADING INPUT"
    )

    print(
        "=" * 90
    )

    print(
        f"Input file:\n    {INPUT_PATH}"
    )


    # ============================================================
    # LOAD DATA
    # ============================================================

    df = pd.read_csv(
        INPUT_PATH,
        sep="\t",
    )

    print(
        f"\nRows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns):,}"
    )


    # ============================================================
    # CHECK REQUIRED COLUMNS
    # ============================================================

    required_columns = [
        "dataset",
        "pearson",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nInput file is missing required columns:\n"
            + "\n".join(
                f"    {column}"
                for column in missing_columns
            )
        )


    # ============================================================
    # FIND OR CALCULATE WITHIN-DATASET RESPONSE-STRENGTH PERCENTILE
    # ============================================================

    PERCENTILE_COLUMN = "dx_strength_percentile"

    magnitude_candidates = [
        "dx_rms_train",
        "dx_train_rms",
        "dx_rms_test",
        "dx_test_rms",
        "dx_norm_train",
        "dx_norm_test",
    ]

    if PERCENTILE_COLUMN not in df.columns:
        magnitude_column = None

        for candidate in magnitude_candidates:
            if candidate in df.columns:
                magnitude_column = candidate
                break

        if magnitude_column is None:
            raise ValueError(
                "\nThe input file does not contain "
                "'dx_strength_percentile', and no response-magnitude "
                "column was found.\n\n"
                "Expected one of:\n"
                + "\n".join(
                    f"    {column}"
                    for column in magnitude_candidates
                )
            )

        print(
            "\nCalculating within-dataset response-strength "
            f"percentiles using:\n    {magnitude_column}"
        )

        df[magnitude_column] = pd.to_numeric(
            df[magnitude_column],
            errors="coerce",
        )

        df[PERCENTILE_COLUMN] = (
            df
            .groupby(
                "dataset",
                group_keys=False,
            )[magnitude_column]
            .rank(
                method="average",
                pct=True,
            )
        )


    # ============================================================
    # CLEAN DATA
    # ============================================================

    keep_columns = [
        "dataset",
        "pearson",
        PERCENTILE_COLUMN,
    ]

    if "perturbation" in df.columns:
        keep_columns.append(
            "perturbation"
        )

    plot_df = df[
        keep_columns
    ].copy()

    plot_df["dataset"] = (
        plot_df["dataset"]
        .astype(str)
    )

    plot_df["pearson"] = pd.to_numeric(
        plot_df["pearson"],
        errors="coerce",
    )

    plot_df[PERCENTILE_COLUMN] = pd.to_numeric(
        plot_df[PERCENTILE_COLUMN],
        errors="coerce",
    )

    plot_df = plot_df[
        np.isfinite(
            plot_df["pearson"]
        )
        & np.isfinite(
            plot_df[PERCENTILE_COLUMN]
        )
    ].copy()

    plot_df = plot_df[
        (
            plot_df[PERCENTILE_COLUMN]
            >= 0.0
        )
        & (
            plot_df[PERCENTILE_COLUMN]
            <= 1.0
        )
    ].copy()

    plot_df = plot_df.reset_index(
        drop=True
    )

    if len(plot_df) < 3:
        raise ValueError(
            "Fewer than three valid perturbations remained."
        )


    # ============================================================
    # NUMERIC ARRAYS
    # ============================================================

    x = plot_df[
        PERCENTILE_COLUMN
    ].to_numpy(
        dtype=np.float64
    )

    y = plot_df[
        "pearson"
    ].to_numpy(
        dtype=np.float64
    )

    dataset_groups = plot_df[
        "dataset"
    ].astype(str).to_numpy()

    n_perturbations = int(
        len(plot_df)
    )

    n_datasets = int(
        plot_df["dataset"].nunique()
    )

    if n_datasets < 2:
        warnings.warn(
            "Only one dataset is present. Dataset-clustered "
            "inference is not meaningful."
        )


    # ============================================================
    # LINEAR REGRESSION ACROSS ALL INDIVIDUAL PERTURBATIONS
    #
    # Model:
    #   held-out Pearson
    #       = intercept
    #       + slope * response-strength percentile
    #
    # R² measures the fraction of individual-perturbation variance
    # explained by the linear trend.
    # ============================================================

    linear_fit = linregress(
        x,
        y,
    )

    slope = float(
        linear_fit.slope
    )

    intercept = float(
        linear_fit.intercept
    )

    pearson_r = float(
        linear_fit.rvalue
    )

    individual_r2 = float(
        pearson_r**2
    )

    linear_pvalue = float(
        linear_fit.pvalue
    )

    linear_slope_se = float(
        linear_fit.stderr
    )


    # ============================================================
    # SPEARMAN CORRELATION ACROSS ALL INDIVIDUAL PERTURBATIONS
    #
    # Spearman rho measures the monotonic association without
    # requiring the relationship to be exactly linear.
    # ============================================================

    spearman_result = spearmanr(
        x,
        y,
        nan_policy="omit",
    )

    spearman_rho = float(
        spearman_result.statistic
    )

    spearman_pvalue = float(
        spearman_result.pvalue
    )


    # ============================================================
    # OLS AND DATASET-CLUSTERED INFERENCE
    #
    # Clustering changes the uncertainty and p-value for the slope.
    # It does not change the fitted line or ordinary R².
    # ============================================================

    design_matrix = sm.add_constant(
        x
    )

    ols_fit = sm.OLS(
        y,
        design_matrix,
    ).fit()

    ols_adjusted_r2 = float(
        ols_fit.rsquared_adj
    )

    ols_slope_ci = ols_fit.conf_int(
        alpha=0.05
    )

    ols_slope_ci_low = float(
        ols_slope_ci[1, 0]
    )

    ols_slope_ci_high = float(
        ols_slope_ci[1, 1]
    )

    clustered_slope_se = np.nan
    clustered_slope_pvalue = np.nan
    clustered_slope_ci_low = np.nan
    clustered_slope_ci_high = np.nan

    if n_datasets >= 2:
        clustered_fit = ols_fit.get_robustcov_results(
            cov_type="cluster",
            groups=dataset_groups,
        )

        clustered_slope_se = float(
            clustered_fit.bse[1]
        )

        clustered_slope_pvalue = float(
            clustered_fit.pvalues[1]
        )

        clustered_slope_ci = clustered_fit.conf_int(
            alpha=0.05
        )

        clustered_slope_ci_low = float(
            clustered_slope_ci[1, 0]
        )

        clustered_slope_ci_high = float(
            clustered_slope_ci[1, 1]
        )


    # ============================================================
    # ASSIGN RESPONSE-STRENGTH DECILES
    # ============================================================

    strength_bin = np.floor(
        plot_df[PERCENTILE_COLUMN].to_numpy()
        * N_BINS
    ).astype(int)

    strength_bin = np.clip(
        strength_bin,
        0,
        N_BINS - 1,
    )

    plot_df["strength_bin"] = (
        strength_bin
    )


    # ============================================================
    # DATASET-SPECIFIC DECILE MEANS
    # ============================================================

    dataset_bin_means = (
        plot_df
        .groupby(
            [
                "dataset",
                "strength_bin",
            ],
            as_index=False,
        )
        .agg(
            dataset_bin_mean_pearson=(
                "pearson",
                "mean",
            ),
            dataset_bin_median_pearson=(
                "pearson",
                "median",
            ),
            dataset_bin_n_perturbations=(
                "pearson",
                "size",
            ),
        )
    )


    # ============================================================
    # DATASET-BALANCED BLACK POINTS
    #
    # Each black point:
    #   mean of dataset-specific mean Pearson values in that bin.
    #
    # Error bar:
    #   SEM across dataset-specific means.
    # ============================================================

    black_rows = []

    for bin_number in range(
        N_BINS
    ):
        values = dataset_bin_means.loc[
            dataset_bin_means["strength_bin"]
            == bin_number,
            "dataset_bin_mean_pearson",
        ].to_numpy(
            dtype=np.float64
        )

        values = values[
            np.isfinite(values)
        ]

        bin_left = (
            bin_number
            / N_BINS
        )

        bin_right = (
            (bin_number + 1)
            / N_BINS
        )

        bin_center = (
            0.5
            * (
                bin_left
                + bin_right
            )
        )

        if values.size == 0:
            mean_pearson = np.nan
            sem_pearson = np.nan
            standard_deviation = np.nan

        else:
            mean_pearson = float(
                np.mean(values)
            )

            standard_deviation = (
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
                if values.size > 1
                else 0.0
            )

            sem_pearson = (
                float(
                    standard_deviation
                    / np.sqrt(values.size)
                )
                if values.size > 1
                else 0.0
            )

        black_rows.append(
            {
                "strength_bin": bin_number,
                "bin_left": bin_left,
                "bin_right": bin_right,
                "bin_center": bin_center,
                "dataset_balanced_mean_pearson": (
                    mean_pearson
                ),
                "standard_deviation_across_datasets": (
                    standard_deviation
                ),
                "sem_across_datasets": (
                    sem_pearson
                ),
                "n_datasets_in_bin": int(
                    values.size
                ),
                "n_individual_perturbations_in_bin": int(
                    (
                        plot_df["strength_bin"]
                        == bin_number
                    ).sum()
                ),
            }
        )

    black_df = pd.DataFrame(
        black_rows
    )


    # ============================================================
    # SAVE STATISTICS
    # ============================================================

    statistics_df = pd.DataFrame(
        [
            {
                "analysis": (
                    "all_individual_perturbations"
                ),
                "x_variable": (
                    "within_dataset_response_strength_percentile"
                ),
                "y_variable": (
                    "held_out_pearson"
                ),
                "n_individual_perturbations": (
                    n_perturbations
                ),
                "n_datasets": (
                    n_datasets
                ),
                "linear_intercept": (
                    intercept
                ),
                "linear_slope": (
                    slope
                ),
                "linear_pearson_r": (
                    pearson_r
                ),
                "linear_r2": (
                    individual_r2
                ),
                "linear_adjusted_r2": (
                    ols_adjusted_r2
                ),
                "linear_slope_standard_error": (
                    linear_slope_se
                ),
                "linear_slope_pvalue": (
                    linear_pvalue
                ),
                "linear_slope_95ci_low": (
                    ols_slope_ci_low
                ),
                "linear_slope_95ci_high": (
                    ols_slope_ci_high
                ),
                "spearman_rho": (
                    spearman_rho
                ),
                "spearman_pvalue": (
                    spearman_pvalue
                ),
                "dataset_clustered_slope_standard_error": (
                    clustered_slope_se
                ),
                "dataset_clustered_slope_pvalue": (
                    clustered_slope_pvalue
                ),
                "dataset_clustered_slope_95ci_low": (
                    clustered_slope_ci_low
                ),
                "dataset_clustered_slope_95ci_high": (
                    clustered_slope_ci_high
                ),
                "input_path": (
                    str(INPUT_PATH)
                ),
            }
        ]
    )

    statistics_path = (
        OUTDIR
        / "raw_individual_perturbation_statistics.tsv"
    )

    statistics_df.to_csv(
        statistics_path,
        sep="\t",
        index=False,
    )


    # ============================================================
    # SAVE DECILE TABLES
    # ============================================================

    black_points_path = (
        OUTDIR
        / "raw_dataset_balanced_decile_points.tsv"
    )

    black_df.to_csv(
        black_points_path,
        sep="\t",
        index=False,
    )

    dataset_bin_means_path = (
        OUTDIR
        / "raw_dataset_specific_decile_means.tsv"
    )

    dataset_bin_means.to_csv(
        dataset_bin_means_path,
        sep="\t",
        index=False,
    )

    cleaned_points_path = (
        OUTDIR
        / "raw_individual_perturbations_used.tsv"
    )

    plot_df.to_csv(
        cleaned_points_path,
        sep="\t",
        index=False,
    )


    # ============================================================
    # PRINT STATISTICS
    # ============================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "ALL INDIVIDUAL PERTURBATIONS"
    )

    print(
        "=" * 90
    )

    print(
        f"Number of perturbations:            {n_perturbations:,}"
    )

    print(
        f"Number of datasets:                 {n_datasets:,}"
    )

    print(
        f"Linear slope:                       {slope:.6f}"
    )

    print(
        f"Linear Pearson r:                   {pearson_r:.6f}"
    )

    print(
        f"Linear R²:                          {individual_r2:.6f}"
    )

    print(
        f"Adjusted R²:                        {ols_adjusted_r2:.6f}"
    )

    print(
        f"Linear slope p-value:               {linear_pvalue:.6e}"
    )

    print(
        f"Spearman rho:                       {spearman_rho:.6f}"
    )

    print(
        f"Spearman p-value:                   {spearman_pvalue:.6e}"
    )

    if np.isfinite(
        clustered_slope_pvalue
    ):
        print(
            "Dataset-clustered slope p-value:     "
            f"{clustered_slope_pvalue:.6e}"
        )

    print(
        "=" * 90
    )


    # ============================================================
    # MAKE FIGURE
    # ============================================================

    figure, axis = plt.subplots(
        figsize=(
            8.7,
            6.7,
        )
    )


    # ------------------------------------------------------------
    # INDIVIDUAL PERTURBATIONS
    # ------------------------------------------------------------

    axis.scatter(
        x,
        y,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="none",
        rasterized=True,
        label="Individual perturbations",
    )


    # ------------------------------------------------------------
    # LINEAR FIT THROUGH ALL INDIVIDUAL PERTURBATIONS
    # ------------------------------------------------------------

    x_line = np.linspace(
        0.0,
        1.0,
        400,
    )

    y_line = (
        intercept
        + slope * x_line
    )

    axis.plot(
        x_line,
        y_line,
        color="black",
        linestyle="--",
        linewidth=2.0,
        alpha=0.90,
        label="Linear fit to all perturbations",
    )


    # ------------------------------------------------------------
    # DATASET-BALANCED BLACK POINTS
    # ------------------------------------------------------------

    valid_black = black_df[
        np.isfinite(
            black_df[
                "dataset_balanced_mean_pearson"
            ]
        )
        & np.isfinite(
            black_df[
                "sem_across_datasets"
            ]
        )
    ].copy()

    axis.errorbar(
        valid_black[
            "bin_center"
        ],
        valid_black[
            "dataset_balanced_mean_pearson"
        ],
        yerr=valid_black[
            "sem_across_datasets"
        ],
        marker="o",
        markersize=6.5,
        linewidth=2.2,
        capsize=3.5,
        color="black",
        markerfacecolor="black",
        markeredgecolor="black",
        zorder=10,
        label="Dataset-balanced decile mean ± SEM",
    )


    # ------------------------------------------------------------
    # ZERO REFERENCE
    # ------------------------------------------------------------

    axis.axhline(
        0.0,
        color="black",
        linewidth=1.1,
        alpha=0.55,
    )


    # ------------------------------------------------------------
    # AXIS LABELS
    # ------------------------------------------------------------

    axis.set_xlabel(
        "Within-dataset response-magnitude percentile",
        fontsize=12,
    )

    axis.set_ylabel(
        "Held-out Pearson",
        fontsize=12,
    )


    # ------------------------------------------------------------
    # TITLE WITH R² AND SPEARMAN RHO FOR ALL BLUE POINTS
    # ------------------------------------------------------------

    title_line_1 = (
        "CIPHER raw counts: held-out Pearson versus response strength"
    )

    title_line_2 = (
        rf"All individual perturbations: "
        rf"$R^2={individual_r2:.3f}$, "
        rf"Spearman $\rho={spearman_rho:.3f}$, "
        rf"$n={n_perturbations:,}$"
    )

    title_line_3 = (
        rf"Linear slope $p={linear_pvalue:.1e}$; "
        rf"Spearman $p={spearman_pvalue:.1e}$"
    )

    if np.isfinite(
        clustered_slope_pvalue
    ):
        title_line_3 += (
            rf"; clustered slope "
            rf"$p={clustered_slope_pvalue:.1e}$"
        )

    axis.set_title(
        title_line_1
        + "\n"
        + title_line_2
        + "\n"
        + title_line_3,
        fontsize=12.5,
        pad=10,
    )


    # ------------------------------------------------------------
    # FORMATTING
    # ------------------------------------------------------------

    axis.set_xlim(
        -0.02,
        1.02,
    )

    axis.set_ylim(
        -1.05,
        1.05,
    )

    axis.grid(
        alpha=0.20,
    )

    axis.set_axisbelow(
        True
    )

    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=9,
    )

    axis.tick_params(
        axis="both",
        labelsize=10,
    )

    for spine_name in [
        "top",
        "right",
    ]:
        axis.spines[
            spine_name
        ].set_visible(
            False
        )

    figure.tight_layout()


    # ============================================================
    # SAVE FIGURE
    # ============================================================

    output_base = (
        OUTDIR
        / "raw_pearson_vs_response_strength_R2_spearman"
    )

    png_path = output_base.with_suffix(
        ".png"
    )

    pdf_path = output_base.with_suffix(
        ".pdf"
    )

    svg_path = output_base.with_suffix(
        ".svg"
    )

    figure.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )

    if SHOW_FIGURE:
        plt.show()

    plt.close(
        figure
    )


    # ============================================================
    # FINAL OUTPUT SUMMARY
    # ============================================================

    print(
        "\nSaved outputs:"
    )

    print(
        f"    {statistics_path.resolve()}"
    )

    print(
        f"    {black_points_path.resolve()}"
    )

    print(
        f"    {dataset_bin_means_path.resolve()}"
    )

    print(
        f"    {cleaned_points_path.resolve()}"
    )

    print(
        f"    {png_path.resolve()}"
    )

    print(
        f"    {pdf_path.resolve()}"
    )

    print(
        f"    {svg_path.resolve()}"
    )



def panelC_individual_fit():
    # ============================================================
    # RAW CIPHER:
    # HELD-OUT PEARSON VS RESPONSE STRENGTH
    #
    # Displays only:
    #   1. Individual perturbations as pale-blue points
    #   2. Dashed linear fit across all perturbations
    #
    # Saves PNG, PDF, and SVG.
    # ============================================================

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from scipy.stats import linregress, spearmanr


    # ============================================================
    # CONFIG
    # ============================================================

    SEARCH_ROOT = Path(REPRO) / "figS14C"

    INPUT_FILENAME = "raw_pearson_vs_dx_per_perturbation.tsv"

    OUTDIR = Path(REPRO) / "figS14C" / "raw_pearson_vs_response_strength_individual_only"

    DPI = 300

    POINT_SIZE = 13
    POINT_ALPHA = 0.10
    POINT_COLOR = "#6BAED6"

    LINE_WIDTH = 2.0

    SHOW_FIGURE = True


    # ============================================================
    # CREATE OUTPUT DIRECTORY
    # ============================================================

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ============================================================
    # LOCATE INPUT FILE
    # ============================================================

    preferred_paths = [
        SEARCH_ROOT / INPUT_FILENAME,

        SEARCH_ROOT
        / "cipher_raw_pearson_vs_dx_magnitude"
        / INPUT_FILENAME,

        SEARCH_ROOT
        / "raw_pearson_vs_dx_magnitude"
        / INPUT_FILENAME,

        SEARCH_ROOT
        / "raw_pearson_vs_response_strength_individual_R2"
        / INPUT_FILENAME,

        SEARCH_ROOT
        / "raw_pearson_vs_response_strength_R2_spearman"
        / INPUT_FILENAME,
    ]

    INPUT_PATH = None

    for candidate in preferred_paths:
        if candidate.exists():
            INPUT_PATH = candidate.resolve()
            break

    if INPUT_PATH is None:
        matches = sorted(
            [
                path.resolve()
                for path in SEARCH_ROOT.rglob(INPUT_FILENAME)
                if path.is_file()
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if len(matches) == 0:
            raise FileNotFoundError(
                f"\nCould not locate:\n"
                f"    {INPUT_FILENAME}\n\n"
                f"Searched recursively under:\n"
                f"    {SEARCH_ROOT.resolve()}"
            )

        INPUT_PATH = matches[0]

        if len(matches) > 1:
            print(
                "\n[WARNING] Multiple matching files were found.\n"
                "Using the most recently modified file:\n"
                f"    {INPUT_PATH}"
            )


    print("=" * 90)
    print("LOADING INPUT")
    print("=" * 90)
    print(f"Input file:\n    {INPUT_PATH}")


    # ============================================================
    # LOAD DATA
    # ============================================================

    df = pd.read_csv(
        INPUT_PATH,
        sep="\t",
    )

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")


    # ============================================================
    # CHECK REQUIRED COLUMNS
    # ============================================================

    required_columns = [
        "dataset",
        "pearson",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nInput file is missing required columns:\n"
            + "\n".join(
                f"    {column}"
                for column in missing_columns
            )
        )


    # ============================================================
    # FIND OR CALCULATE WITHIN-DATASET RESPONSE-STRENGTH PERCENTILE
    # ============================================================

    PERCENTILE_COLUMN = "dx_strength_percentile"

    magnitude_candidates = [
        "dx_rms_train",
        "dx_train_rms",
        "dx_rms_test",
        "dx_test_rms",
        "dx_norm_train",
        "dx_norm_test",
    ]

    if PERCENTILE_COLUMN not in df.columns:
        magnitude_column = None

        for candidate in magnitude_candidates:
            if candidate in df.columns:
                magnitude_column = candidate
                break

        if magnitude_column is None:
            raise ValueError(
                "\nThe input file does not contain "
                "'dx_strength_percentile', and no response-magnitude "
                "column was found.\n\n"
                "Expected one of:\n"
                + "\n".join(
                    f"    {column}"
                    for column in magnitude_candidates
                )
            )

        print(
            "\nCalculating within-dataset response-strength "
            f"percentiles using:\n    {magnitude_column}"
        )

        df[magnitude_column] = pd.to_numeric(
            df[magnitude_column],
            errors="coerce",
        )

        df[PERCENTILE_COLUMN] = (
            df
            .groupby(
                "dataset",
                group_keys=False,
            )[magnitude_column]
            .rank(
                method="average",
                pct=True,
            )
        )


    # ============================================================
    # CLEAN DATA
    # ============================================================

    keep_columns = [
        "dataset",
        "pearson",
        PERCENTILE_COLUMN,
    ]

    if "perturbation" in df.columns:
        keep_columns.append("perturbation")

    plot_df = df[keep_columns].copy()

    plot_df["pearson"] = pd.to_numeric(
        plot_df["pearson"],
        errors="coerce",
    )

    plot_df[PERCENTILE_COLUMN] = pd.to_numeric(
        plot_df[PERCENTILE_COLUMN],
        errors="coerce",
    )

    plot_df = plot_df[
        np.isfinite(plot_df["pearson"])
        & np.isfinite(plot_df[PERCENTILE_COLUMN])
    ].copy()

    plot_df = plot_df[
        plot_df[PERCENTILE_COLUMN].between(
            0.0,
            1.0,
            inclusive="both",
        )
    ].copy()

    plot_df = plot_df.reset_index(
        drop=True,
    )

    if len(plot_df) < 3:
        raise ValueError(
            "Fewer than three valid perturbations remained."
        )


    # ============================================================
    # NUMERIC ARRAYS
    # ============================================================

    x = plot_df[PERCENTILE_COLUMN].to_numpy(
        dtype=np.float64,
    )

    y = plot_df["pearson"].to_numpy(
        dtype=np.float64,
    )

    n_perturbations = len(plot_df)
    n_datasets = plot_df["dataset"].nunique()


    # ============================================================
    # LINEAR FIT
    # ============================================================

    linear_fit = linregress(
        x,
        y,
    )

    slope = float(linear_fit.slope)
    intercept = float(linear_fit.intercept)

    pearson_r = float(linear_fit.rvalue)
    individual_r2 = float(pearson_r**2)

    linear_pvalue = float(linear_fit.pvalue)

    spearman_result = spearmanr(
        x,
        y,
        nan_policy="omit",
    )

    spearman_rho = float(
        spearman_result.statistic
    )

    spearman_pvalue = float(
        spearman_result.pvalue
    )


    # ============================================================
    # PRINT STATISTICS
    # ============================================================

    print("\n" + "=" * 90)
    print("ALL INDIVIDUAL PERTURBATIONS")
    print("=" * 90)

    print(
        f"Number of perturbations:  {n_perturbations:,}"
    )

    print(
        f"Number of datasets:       {n_datasets:,}"
    )

    print(
        f"Linear intercept:         {intercept:.6f}"
    )

    print(
        f"Linear slope:             {slope:.6f}"
    )

    print(
        f"Linear Pearson r:         {pearson_r:.6f}"
    )

    print(
        f"Linear R²:                {individual_r2:.6f}"
    )

    print(
        f"Linear slope p-value:     {linear_pvalue:.6e}"
    )

    print(
        f"Spearman rho:             {spearman_rho:.6f}"
    )

    print(
        f"Spearman p-value:         {spearman_pvalue:.6e}"
    )

    print("=" * 90)


    # ============================================================
    # SAVE CLEANED POINTS AND STATISTICS
    # ============================================================

    cleaned_points_path = (
        OUTDIR
        / "raw_individual_perturbations_used.tsv"
    )

    plot_df.to_csv(
        cleaned_points_path,
        sep="\t",
        index=False,
    )

    statistics_df = pd.DataFrame(
        [
            {
                "n_individual_perturbations": n_perturbations,
                "n_datasets": n_datasets,
                "linear_intercept": intercept,
                "linear_slope": slope,
                "linear_pearson_r": pearson_r,
                "linear_r2": individual_r2,
                "linear_slope_pvalue": linear_pvalue,
                "spearman_rho": spearman_rho,
                "spearman_pvalue": spearman_pvalue,
                "input_path": str(INPUT_PATH),
            }
        ]
    )

    statistics_path = (
        OUTDIR
        / "raw_individual_perturbation_statistics.tsv"
    )

    statistics_df.to_csv(
        statistics_path,
        sep="\t",
        index=False,
    )


    # ============================================================
    # MAKE FIGURE
    # ============================================================

    figure, axis = plt.subplots(
        figsize=(8.7, 6.7),
    )


    # ------------------------------------------------------------
    # INDIVIDUAL PERTURBATIONS
    # ------------------------------------------------------------

    axis.scatter(
        x,
        y,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        color=POINT_COLOR,
        edgecolors="none",
        rasterized=True,
        label="Individual perturbations",
    )


    # ------------------------------------------------------------
    # DASHED LINEAR FIT
    # ------------------------------------------------------------

    x_line = np.linspace(
        0.0,
        1.0,
        400,
    )

    y_line = (
        intercept
        + slope * x_line
    )

    axis.plot(
        x_line,
        y_line,
        color="black",
        linestyle="--",
        linewidth=LINE_WIDTH,
        alpha=0.90,
        label="Linear fit",
    )


    # ------------------------------------------------------------
    # ZERO REFERENCE
    # ------------------------------------------------------------

    axis.axhline(
        0.0,
        color="black",
        linewidth=1.1,
        alpha=0.55,
    )


    # ------------------------------------------------------------
    # AXIS LABELS
    # ------------------------------------------------------------

    axis.set_xlabel(
        "Within-dataset response-magnitude percentile",
        fontsize=12,
    )

    axis.set_ylabel(
        "Held-out Pearson",
        fontsize=12,
    )


    # ------------------------------------------------------------
    # TITLE
    # ------------------------------------------------------------

    axis.set_title(
        "CIPHER raw counts: held-out Pearson versus response strength"
        "\n"
        rf"All individual perturbations: "
        rf"$R^2={individual_r2:.3f}$, "
        rf"Spearman $\rho={spearman_rho:.3f}$, "
        rf"$n={n_perturbations:,}$"
        "\n"
        rf"Linear slope $p={linear_pvalue:.1e}$; "
        rf"Spearman $p={spearman_pvalue:.1e}$",
        fontsize=12.5,
        pad=10,
    )


    # ------------------------------------------------------------
    # FORMATTING
    # ------------------------------------------------------------

    axis.set_xlim(
        -0.02,
        1.02,
    )

    axis.set_ylim(
        -1.05,
        1.05,
    )

    axis.grid(
        alpha=0.20,
    )

    axis.set_axisbelow(
        True,
    )

    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=9,
    )

    axis.tick_params(
        axis="both",
        labelsize=10,
    )

    for spine_name in [
        "top",
        "right",
    ]:
        axis.spines[spine_name].set_visible(
            False
        )

    figure.tight_layout()


    # ============================================================
    # SAVE FIGURE
    # ============================================================

    output_base = (
        OUTDIR
        / "raw_pearson_vs_response_strength_individual_and_linear_fit"
    )

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    svg_path = output_base.with_suffix(".svg")

    figure.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )

    if SHOW_FIGURE:
        plt.show()

    plt.close(figure)


    # ============================================================
    # FINAL OUTPUT SUMMARY
    # ============================================================

    print("\nSaved outputs:")

    print(
        f"    {statistics_path.resolve()}"
    )

    print(
        f"    {cleaned_points_path.resolve()}"
    )

    print(
        f"    {png_path.resolve()}"
    )

    print(
        f"    {pdf_path.resolve()}"
    )

    print(
        f"    {svg_path.resolve()}"
    )



def panelC_scatter_only():
    # ============================================================
    # RAW CIPHER:
    # HELD-OUT PEARSON VS WITHIN-DATASET RESPONSE STRENGTH
    #
    # Figure contains ONLY the individual perturbations.
    #
    # No black binned points.
    # No regression line.
    # No zero-reference line.
    #
    # Reports across all individual perturbations:
    #   - linear-regression R²
    #   - Spearman rho
    #   - linear slope and p-value
    #   - Spearman p-value
    #   - dataset-clustered slope p-value
    #
    # Automatically searches recursively for:
    #   raw_pearson_vs_dx_per_perturbation.tsv
    #
    # Outputs:
    #   raw_individual_perturbation_statistics.tsv
    #   raw_individual_perturbations_used.tsv
    #   raw_pearson_vs_response_strength_scatter.png
    #   raw_pearson_vs_response_strength_scatter.pdf
    #   raw_pearson_vs_response_strength_scatter.svg
    # ============================================================

    from pathlib import Path
    import warnings

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from scipy.stats import linregress, spearmanr
    import statsmodels.api as sm


    # ============================================================
    # CONFIG
    # ============================================================

    SEARCH_ROOT = Path(REPRO) / "figS14C"

    INPUT_FILENAME = "raw_pearson_vs_dx_per_perturbation.tsv"

    OUTDIR = Path(REPRO) / "figS14C" / "raw_pearson_vs_response_strength_scatter"

    DPI = 300

    POINT_SIZE = 14
    POINT_ALPHA = 0.12

    SHOW_FIGURE = True


    # ============================================================
    # CREATE OUTPUT DIRECTORY
    # ============================================================

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ============================================================
    # LOCATE INPUT FILE
    #
    # Check likely locations first. If not found, search recursively
    # and use the most recently modified matching file.
    # ============================================================

    preferred_paths = [
        SEARCH_ROOT
        / INPUT_FILENAME,

        SEARCH_ROOT
        / "cipher_raw_pearson_vs_dx_magnitude"
        / INPUT_FILENAME,

        SEARCH_ROOT
        / "raw_pearson_vs_dx_magnitude"
        / INPUT_FILENAME,

        SEARCH_ROOT
        / "raw_pearson_vs_response_strength_individual_R2"
        / INPUT_FILENAME,

        SEARCH_ROOT
        / "raw_pearson_vs_response_strength_R2_spearman"
        / INPUT_FILENAME,
    ]

    INPUT_PATH = None

    for candidate in preferred_paths:
        if candidate.exists():
            INPUT_PATH = candidate.resolve()
            break


    if INPUT_PATH is None:
        matches = sorted(
            [
                path.resolve()
                for path in SEARCH_ROOT.rglob(
                    INPUT_FILENAME
                )
                if path.is_file()
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if len(matches) == 0:
            raise FileNotFoundError(
                "\nCould not locate the required input file:\n"
                f"    {INPUT_FILENAME}\n\n"
                "Searched recursively under:\n"
                f"    {SEARCH_ROOT.resolve()}\n\n"
                "Run the raw CIPHER response-strength analysis first, "
                "or change SEARCH_ROOT to the appropriate directory."
            )

        INPUT_PATH = matches[0]

        if len(matches) > 1:
            print(
                "\n[WARNING] Multiple matching files were found."
            )

            print(
                "Using the most recently modified file:"
            )

            print(
                f"    {INPUT_PATH}"
            )

            print(
                "\nOther matches:"
            )

            for alternative in matches[1:]:
                print(
                    f"    {alternative}"
                )


    # ============================================================
    # LOAD DATA
    # ============================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "LOADING INPUT"
    )

    print(
        "=" * 90
    )

    print(
        f"Input file:\n    {INPUT_PATH}"
    )

    df = pd.read_csv(
        INPUT_PATH,
        sep="\t",
    )

    print(
        f"\nRows loaded:    {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns):,}"
    )


    # ============================================================
    # CHECK REQUIRED COLUMNS
    # ============================================================

    required_columns = [
        "dataset",
        "pearson",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nInput file is missing required columns:\n"
            + "\n".join(
                f"    {column}"
                for column in missing_columns
            )
        )


    # ============================================================
    # FIND OR CALCULATE WITHIN-DATASET RESPONSE-STRENGTH PERCENTILE
    #
    # Preferred existing column:
    #   dx_strength_percentile
    #
    # Otherwise calculate percentile ranks within each dataset from
    # the first available response-magnitude column.
    # ============================================================

    PERCENTILE_COLUMN = "dx_strength_percentile"

    magnitude_candidates = [
        "dx_rms_train",
        "dx_train_rms",
        "dx_rms_test",
        "dx_test_rms",
        "dx_norm_train",
        "dx_norm_test",
    ]

    magnitude_column_used = None


    if PERCENTILE_COLUMN not in df.columns:
        for candidate in magnitude_candidates:
            if candidate in df.columns:
                magnitude_column_used = candidate
                break

        if magnitude_column_used is None:
            raise ValueError(
                "\nThe input file does not contain "
                "'dx_strength_percentile', and no usable response-"
                "magnitude column was found.\n\n"
                "Expected one of:\n"
                + "\n".join(
                    f"    {column}"
                    for column in magnitude_candidates
                )
            )

        print(
            "\nCalculating within-dataset response-strength "
            "percentiles from:"
        )

        print(
            f"    {magnitude_column_used}"
        )

        df[magnitude_column_used] = pd.to_numeric(
            df[magnitude_column_used],
            errors="coerce",
        )

        df[PERCENTILE_COLUMN] = (
            df
            .groupby(
                "dataset",
                group_keys=False,
            )[magnitude_column_used]
            .rank(
                method="average",
                pct=True,
            )
        )

    else:
        magnitude_column_used = (
            "existing dx_strength_percentile"
        )


    # ============================================================
    # CLEAN DATA
    # ============================================================

    keep_columns = [
        "dataset",
        "pearson",
        PERCENTILE_COLUMN,
    ]

    optional_columns = [
        "perturbation",
        "target_gene",
        "dx_rms_train",
        "dx_train_rms",
        "dx_rms_test",
        "dx_test_rms",
        "dx_norm_train",
        "dx_norm_test",
    ]

    for column in optional_columns:
        if (
            column in df.columns
            and column not in keep_columns
        ):
            keep_columns.append(
                column
            )


    plot_df = df[
        keep_columns
    ].copy()

    plot_df["dataset"] = (
        plot_df["dataset"]
        .astype(str)
    )

    plot_df["pearson"] = pd.to_numeric(
        plot_df["pearson"],
        errors="coerce",
    )

    plot_df[PERCENTILE_COLUMN] = pd.to_numeric(
        plot_df[PERCENTILE_COLUMN],
        errors="coerce",
    )


    # Retain only finite values.

    plot_df = plot_df[
        np.isfinite(
            plot_df["pearson"]
        )
        & np.isfinite(
            plot_df[PERCENTILE_COLUMN]
        )
    ].copy()


    # Retain valid percentile values.

    plot_df = plot_df[
        (
            plot_df[PERCENTILE_COLUMN]
            >= 0.0
        )
        & (
            plot_df[PERCENTILE_COLUMN]
            <= 1.0
        )
    ].copy()

    plot_df = plot_df.reset_index(
        drop=True
    )


    if len(plot_df) < 3:
        raise ValueError(
            "Fewer than three valid perturbations remained "
            "after filtering."
        )


    # ============================================================
    # NUMERIC ARRAYS
    # ============================================================

    x = plot_df[
        PERCENTILE_COLUMN
    ].to_numpy(
        dtype=np.float64
    )

    y = plot_df[
        "pearson"
    ].to_numpy(
        dtype=np.float64
    )

    dataset_groups = plot_df[
        "dataset"
    ].astype(str).to_numpy()

    n_perturbations = int(
        len(plot_df)
    )

    n_datasets = int(
        plot_df["dataset"].nunique()
    )


    if n_datasets < 2:
        warnings.warn(
            "Only one dataset is present. Dataset-clustered "
            "inference cannot be calculated meaningfully."
        )


    # ============================================================
    # LINEAR REGRESSION ACROSS ALL INDIVIDUAL PERTURBATIONS
    #
    # Model:
    #
    #   held-out Pearson
    #       = intercept
    #       + slope * response-strength percentile
    #
    # The fitted line is used to calculate R² and the slope
    # statistics but is NOT drawn on the figure.
    # ============================================================

    linear_fit = linregress(
        x,
        y,
    )

    linear_intercept = float(
        linear_fit.intercept
    )

    linear_slope = float(
        linear_fit.slope
    )

    linear_pearson_r = float(
        linear_fit.rvalue
    )

    linear_r2 = float(
        linear_pearson_r**2
    )

    linear_slope_standard_error = float(
        linear_fit.stderr
    )

    linear_slope_pvalue = float(
        linear_fit.pvalue
    )


    # ============================================================
    # SPEARMAN CORRELATION ACROSS ALL INDIVIDUAL PERTURBATIONS
    #
    # Spearman rho measures monotonic association without assuming
    # that the relationship is exactly linear.
    # ============================================================

    spearman_result = spearmanr(
        x,
        y,
        nan_policy="omit",
    )

    spearman_rho = float(
        spearman_result.statistic
    )

    spearman_pvalue = float(
        spearman_result.pvalue
    )


    # ============================================================
    # OLS STATISTICS
    # ============================================================

    design_matrix = sm.add_constant(
        x
    )

    ols_fit = sm.OLS(
        y,
        design_matrix,
    ).fit()

    ols_r2 = float(
        ols_fit.rsquared
    )

    ols_adjusted_r2 = float(
        ols_fit.rsquared_adj
    )

    ols_slope_ci = ols_fit.conf_int(
        alpha=0.05
    )

    ols_slope_ci_low = float(
        ols_slope_ci[1, 0]
    )

    ols_slope_ci_high = float(
        ols_slope_ci[1, 1]
    )


    # ============================================================
    # DATASET-CLUSTERED SLOPE INFERENCE
    #
    # The fitted relationship and R² are unchanged.
    # Clustering adjusts the slope uncertainty and p-value for
    # dependence among perturbations from the same dataset.
    # ============================================================

    clustered_slope_standard_error = np.nan
    clustered_slope_pvalue = np.nan
    clustered_slope_ci_low = np.nan
    clustered_slope_ci_high = np.nan


    if n_datasets >= 2:
        clustered_fit = ols_fit.get_robustcov_results(
            cov_type="cluster",
            groups=dataset_groups,
        )

        clustered_slope_standard_error = float(
            clustered_fit.bse[1]
        )

        clustered_slope_pvalue = float(
            clustered_fit.pvalues[1]
        )

        clustered_slope_ci = clustered_fit.conf_int(
            alpha=0.05
        )

        clustered_slope_ci_low = float(
            clustered_slope_ci[1, 0]
        )

        clustered_slope_ci_high = float(
            clustered_slope_ci[1, 1]
        )


    # ============================================================
    # SAVE CLEANED INDIVIDUAL-PERTURBATION DATA
    # ============================================================

    cleaned_points_path = (
        OUTDIR
        / "raw_individual_perturbations_used.tsv"
    )

    plot_df.to_csv(
        cleaned_points_path,
        sep="\t",
        index=False,
    )


    # ============================================================
    # SAVE STATISTICS
    # ============================================================

    statistics_df = pd.DataFrame(
        [
            {
                "analysis": (
                    "all_individual_perturbations"
                ),
                "x_variable": (
                    "within_dataset_response_strength_percentile"
                ),
                "y_variable": (
                    "held_out_pearson"
                ),
                "n_individual_perturbations": (
                    n_perturbations
                ),
                "n_datasets": (
                    n_datasets
                ),
                "linear_intercept": (
                    linear_intercept
                ),
                "linear_slope": (
                    linear_slope
                ),
                "linear_pearson_r": (
                    linear_pearson_r
                ),
                "linear_r2": (
                    linear_r2
                ),
                "linear_adjusted_r2": (
                    ols_adjusted_r2
                ),
                "linear_slope_standard_error": (
                    linear_slope_standard_error
                ),
                "linear_slope_pvalue": (
                    linear_slope_pvalue
                ),
                "linear_slope_95ci_low": (
                    ols_slope_ci_low
                ),
                "linear_slope_95ci_high": (
                    ols_slope_ci_high
                ),
                "spearman_rho": (
                    spearman_rho
                ),
                "spearman_pvalue": (
                    spearman_pvalue
                ),
                "dataset_clustered_slope_standard_error": (
                    clustered_slope_standard_error
                ),
                "dataset_clustered_slope_pvalue": (
                    clustered_slope_pvalue
                ),
                "dataset_clustered_slope_95ci_low": (
                    clustered_slope_ci_low
                ),
                "dataset_clustered_slope_95ci_high": (
                    clustered_slope_ci_high
                ),
                "response_strength_source": (
                    magnitude_column_used
                ),
                "input_path": (
                    str(INPUT_PATH)
                ),
            }
        ]
    )

    statistics_path = (
        OUTDIR
        / "raw_individual_perturbation_statistics.tsv"
    )

    statistics_df.to_csv(
        statistics_path,
        sep="\t",
        index=False,
    )


    # ============================================================
    # PRINT STATISTICS
    # ============================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "ALL INDIVIDUAL PERTURBATIONS"
    )

    print(
        "=" * 90
    )

    print(
        f"Number of perturbations:            {n_perturbations:,}"
    )

    print(
        f"Number of datasets:                 {n_datasets:,}"
    )

    print(
        f"Linear intercept:                   {linear_intercept:.6f}"
    )

    print(
        f"Linear slope:                       {linear_slope:.6f}"
    )

    print(
        f"Linear Pearson r:                   {linear_pearson_r:.6f}"
    )

    print(
        f"Linear R²:                          {linear_r2:.6f}"
    )

    print(
        f"Adjusted R²:                        {ols_adjusted_r2:.6f}"
    )

    print(
        f"Linear slope p-value:               {linear_slope_pvalue:.6e}"
    )

    print(
        "Linear slope 95% CI:               "
        f"[{ols_slope_ci_low:.6f}, "
        f"{ols_slope_ci_high:.6f}]"
    )

    print(
        f"Spearman rho:                       {spearman_rho:.6f}"
    )

    print(
        f"Spearman p-value:                   {spearman_pvalue:.6e}"
    )

    if np.isfinite(
        clustered_slope_pvalue
    ):
        print(
            "Dataset-clustered slope p-value:     "
            f"{clustered_slope_pvalue:.6e}"
        )

        print(
            "Dataset-clustered slope 95% CI:      "
            f"[{clustered_slope_ci_low:.6f}, "
            f"{clustered_slope_ci_high:.6f}]"
        )

    print(
        "=" * 90
    )


    # ============================================================
    # MAKE SCATTER FIGURE
    #
    # Only the individual perturbations are drawn.
    #
    # No black binned points.
    # No fitted regression line.
    # No horizontal zero line.
    # ============================================================

    figure, axis = plt.subplots(
        figsize=(
            8.5,
            6.4,
        )
    )


    axis.scatter(
        x,
        y,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="none",
        rasterized=True,
    )


    # ============================================================
    # AXIS LABELS
    # ============================================================

    axis.set_xlabel(
        "Within-dataset response-magnitude percentile",
        fontsize=12,
    )

    axis.set_ylabel(
        "Held-out Pearson",
        fontsize=12,
    )


    # ============================================================
    # TITLE
    #
    # R² and Spearman rho both refer to all individual points.
    # ============================================================

    title_line_1 = (
        "CIPHER raw counts: held-out Pearson versus response strength"
    )

    title_line_2 = (
        rf"All individual perturbations: "
        rf"$R^2={linear_r2:.3f}$, "
        rf"Spearman $\rho={spearman_rho:.3f}$, "
        rf"$n={n_perturbations:,}$"
    )

    title_line_3 = (
        rf"Linear slope $p={linear_slope_pvalue:.1e}$; "
        rf"Spearman $p={spearman_pvalue:.1e}$"
    )

    if np.isfinite(
        clustered_slope_pvalue
    ):
        title_line_3 += (
            rf"; clustered slope "
            rf"$p={clustered_slope_pvalue:.1e}$"
        )


    axis.set_title(
        title_line_1
        + "\n"
        + title_line_2
        + "\n"
        + title_line_3,
        fontsize=12.5,
        pad=10,
    )


    # ============================================================
    # AXIS FORMATTING
    # ============================================================

    axis.set_xlim(
        -0.02,
        1.02,
    )

    axis.set_ylim(
        -1.05,
        1.05,
    )

    # No grid lines.
    axis.grid(
        False
    )

    axis.tick_params(
        axis="both",
        labelsize=10,
    )

    for spine_name in [
        "top",
        "right",
    ]:
        axis.spines[
            spine_name
        ].set_visible(
            False
        )

    figure.tight_layout()


    # ============================================================
    # SAVE FIGURE
    # ============================================================

    output_base = (
        OUTDIR
        / "raw_pearson_vs_response_strength_scatter"
    )

    png_path = output_base.with_suffix(
        ".png"
    )

    pdf_path = output_base.with_suffix(
        ".pdf"
    )

    svg_path = output_base.with_suffix(
        ".svg"
    )


    figure.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )


    if SHOW_FIGURE:
        plt.show()


    plt.close(
        figure
    )


    # ============================================================
    # FINAL OUTPUT SUMMARY
    # ============================================================

    print(
        "\nSaved outputs:"
    )

    print(
        f"    {statistics_path.resolve()}"
    )

    print(
        f"    {cleaned_points_path.resolve()}"
    )

    print(
        f"    {png_path.resolve()}"
    )

    print(
        f"    {pdf_path.resolve()}"
    )

    print(
        f"    {svg_path.resolve()}"
    )
