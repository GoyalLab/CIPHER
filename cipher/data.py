"""Dataset loading, control/perturbation detection and gene filtering.

The heuristics here let CIPHER ingest heterogeneous Perturb-seq ``.h5ad`` files
without the caller having to hand-annotate which ``obs`` column holds the
perturbation label, which value marks controls, or which gene each guide targets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse

from .utils import to_dense

# labels that (case-insensitively) denote a negative-control population
CONTROL_PATTERNS = [
    "control", "ctrl", "ntc", "non-targeting", "nontargeting", "non_targeting",
    "negative", "neg", "safe", "scramble", "scrambled", "mock", "untreated",
    "vehicle", "empty",
]

# obs columns that may hold the perturbation label, in preference order
PERT_KEY_CANDIDATES = [
    "perturbation", "perturbation_name", "target", "target_gene", "gene",
    "gene_name", "guide_target", "guide_targets", "sgRNA_target", "condition",
    "Condition", "treatment", "guide_id", "guide", "sgRNA", "sgRNA_ID",
]

CONTROL_LABEL = "control"


# --------------------------------------------------------------------------- #
# control / target parsing
# --------------------------------------------------------------------------- #
def find_control_label(values, desired: str = CONTROL_LABEL):
    """Best guess at the control label among ``values``.

    Returns ``(label, count, mode)`` where ``mode`` records how it was matched.
    """
    vals = pd.Series(values).astype(str)
    vc = vals.value_counts()
    if str(desired) in vc.index:
        return str(desired), int(vc.loc[str(desired)]), "exact_requested"
    for label in vc.index.astype(str):
        if label.lower() in CONTROL_PATTERNS:
            return label, int(vc.loc[label]), "exact_pattern"
    cands = [x for x in vc.index.astype(str) if any(p in x.lower() for p in CONTROL_PATTERNS)]
    if cands:
        sub = vc.loc[cands].sort_values(ascending=False)
        return str(sub.index[0]), int(sub.iloc[0]), "contains_pattern"
    return None, 0, "none"


def strip_pert_label(p: str) -> str:
    """Strip common guide/direction decorations (sgFOO, FOO_KO, ...) from a label."""
    p = str(p).strip()
    p = re.sub(r"([_\-\s]+)(KD|KO|OE|overexp|overexpression|activation|inhibition)$", "", p, flags=re.I)
    p = re.sub(r"^(sgRNA|gRNA|sgrna|grna)([_\-\s]+)", "", p, flags=re.I)
    p = re.sub(r"^(sg)(?=[A-Z0-9])", "", p)
    return p


def infer_target_gene(pert: str, gene_set, sep: str = "_"):
    """Map a perturbation label to the single gene it targets, if any.

    Returns ``(gene, reason)``; ``gene`` is ``""`` when the label is a
    multi-gene perturbation or cannot be matched to ``gene_set``.
    """
    p0 = str(pert).strip()
    if p0 in gene_set:
        return p0, "exact"
    p = strip_pert_label(p0)
    if p in gene_set:
        return p, "stripped"
    for splitter in [sep, "+", "|", ";", ",", " "]:
        if splitter in p:
            parts = [x for x in p.split(splitter) if x]
            hits = [x for x in parts if x in gene_set]
            if len(hits) == 1:
                return hits[0], "parsed_single"
            if len(hits) > 1:
                return "", "multi_gene"
            break
    return "", "unmatched"


def _score_pert_column(values, gene_set, min_samples: int) -> dict:
    vals = pd.Series(values).astype(str)
    vc = vals.value_counts()
    ctrl, n_ctrl, ctrl_mode = find_control_label(vals)
    if ctrl is None or n_ctrl <= 0:
        return {"score": -1e9, "control_label": "", "control_count": 0,
                "control_mode": "none", "n_groups": len(vc), "target_hit_rate": 0.0}
    labels = [x for x in vc.index.astype(str).tolist() if x != ctrl]
    hits = checked = 0
    for lab in labels[:3000]:
        if int(vc.loc[lab]) < min_samples:
            continue
        target, _ = infer_target_gene(lab, gene_set)
        checked += 1
        hits += int(bool(target))
    hit_rate = hits / max(checked, 1)
    score = 1000.0 + 200.0 * hit_rate + 10.0 * np.log10(n_ctrl + 1) + 2.0 * np.log10(len(vc) + 1)
    score += {"exact_requested": 25.0, "exact_pattern": 15.0, "contains_pattern": 5.0}.get(ctrl_mode, 0.0)
    return {"score": float(score), "control_label": str(ctrl), "control_count": int(n_ctrl),
            "control_mode": ctrl_mode, "n_groups": int(len(vc)), "target_hit_rate": float(hit_rate)}


def resolve_perturbation_key(adata, min_samples: int = 1):
    """Pick the ``obs`` column and control label that best look like a screen.

    Returns ``(pert_key, control_label, candidate_scores_df)``.
    """
    gene_set = set(np.asarray(adata.var_names.astype(str)).tolist())
    candidates = [c for c in dict.fromkeys(PERT_KEY_CANDIDATES) if c in adata.obs.columns]
    if not candidates:
        raise KeyError(f"No perturbation key found. obs columns={list(adata.obs.columns)}")
    rows = [{"column": c, **_score_pert_column(adata.obs[c].astype(str), gene_set, min_samples)}
            for c in candidates]
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    if float(best["score"]) < 0 or int(best["control_count"]) <= 0:
        raise ValueError("Could not detect control/perturbation column.\n" + df.to_string(index=False))
    return str(best["column"]), str(best["control_label"]), df


def ensure_gene_symbol_var_names(adata, sample_labels) -> None:
    """If ``var_names`` don't overlap perturbation labels, swap in a gene-symbol column."""
    if len(set(map(str, sample_labels)).intersection(adata.var_names)) > 0:
        return
    hits = [c for c in adata.var.columns if c in {"gene_name", "gene_names", "name", "gene_symbol", "symbol"}]
    if not hits:
        return
    adata.var.reset_index(names=["_orig_var_id"], inplace=True)
    adata.var.index = adata.var[hits[0]].astype(str).tolist()
    adata.var_names_make_unique()


# --------------------------------------------------------------------------- #
# Dataset container
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """A filtered Perturb-seq dataset with resolved control/perturbation metadata."""
    adata: "ad.AnnData"
    pert_key: str
    control_label: str
    gene_names: np.ndarray
    perturbations: list           # selected non-control perturbation labels
    target_genes: list            # gene targeted by each perturbation (parallel to `perturbations`)
    target_gene_indices: np.ndarray  # index of each target gene into `gene_names`
    control_mask: np.ndarray
    stats: dict = field(default_factory=dict)
    name: str = "dataset"
    #: per-gene control mean/variance over ALL genes before the expression filter,
    #: used to fit the pflog dispersion across the whole expression range.
    control_full_mean: Optional[np.ndarray] = None
    control_full_var: Optional[np.ndarray] = None

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    @property
    def n_perturbations(self) -> int:
        return len(self.perturbations)

    @property
    def pflog_alpha(self) -> float:
        """NB dispersion for pflog, fit over the FULL (un-filtered) control gene set."""
        from .normalize import _alpha_from_meanvar, fit_pflog_alpha
        if self.control_full_mean is not None and self.control_full_var is not None:
            return _alpha_from_meanvar(self.control_full_mean, self.control_full_var)
        # fallback: fit from the (filtered) control matrix if full stats are absent
        return fit_pflog_alpha(self.control_matrix(dense=False), self.gene_names)[0]

    @property
    def pflog_pseudocount(self) -> float:
        """pflog pseudocount ``1/(4*alpha)`` from :attr:`pflog_alpha`."""
        return float(1.0 / (4.0 * self.pflog_alpha))

    def control_matrix(self, dense: bool = True):
        """Raw expression of control cells (``dense`` -> ndarray)."""
        X = self.adata[self.control_mask].X
        return to_dense(X) if dense else X

    def perturbation_matrix(self, pert, dense: bool = True):
        """Raw expression of cells for perturbation ``pert``."""
        X = self.adata[self.adata.obs[self.pert_key].astype(str) == str(pert)].X
        return to_dense(X) if dense else X

    def gene_index(self, gene) -> Optional[int]:
        idx = np.where(self.gene_names == gene)[0]
        return int(idx[0]) if idx.size else None


def load_dataset(
    data_path,
    expression_threshold: float = 1.0,
    min_samples: int = 100,
    cutoff_source: str = "control",
    only_single_gene: bool = True,
    require_target_in_var: bool = True,
    force_include_targets: bool = True,
    pert_key: Optional[str] = None,
    control_label: Optional[str] = None,
) -> Dataset:
    """Load a Perturb-seq ``.h5ad`` and filter genes/perturbations for CIPHER.

    Genes are kept when their mean raw expression over ``cutoff_source`` cells
    exceeds ``expression_threshold`` (perturbed target genes are force-kept when
    ``force_include_targets``).  Perturbations are kept when they have at least
    ``min_samples`` cells and (optionally) map to a single gene present in
    ``var_names``.
    """
    adata = ad.read_h5ad(str(data_path))
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()

    if pert_key is None:
        pk, cl, _ = resolve_perturbation_key(adata, min_samples=min_samples)
        pert_key = pk
        if control_label is None:
            control_label = cl
    elif control_label is None:
        # honour the caller's column: detect the control label *within* it
        control_label, _, _ = find_control_label(adata.obs[pert_key])
    if control_label is None:
        raise ValueError(f"Could not determine a control label in obs[{pert_key!r}]; "
                         f"pass control_label= explicitly.")

    labels_all = adata.obs[pert_key].astype(str).to_numpy()
    ensure_gene_symbol_var_names(adata, pd.unique(labels_all))
    gene_names_all = np.asarray(adata.var_names.astype(str))
    gene_set = set(gene_names_all.tolist())

    counts = pd.Series(labels_all).value_counts()
    control_rows = np.flatnonzero(labels_all == str(control_label))
    if control_rows.size < 2:
        raise ValueError(f"Only {control_rows.size} control cells for {control_label!r}.")

    # full-gene control mean/variance (ALL genes, before any expression filter) — kept so
    # the pflog dispersion can be fit over the whole expression range, not just kept genes.
    from .normalize import mean_var as _mean_var
    control_full_mean, control_full_var = _mean_var(adata.X[control_rows, :])

    # gene expression cutoff
    if cutoff_source == "control":
        gene_mean = np.nan_to_num(np.asarray(control_full_mean, dtype=np.float64))
    elif cutoff_source == "all":
        Xc = adata.X
        gene_mean = np.nan_to_num((np.asarray(Xc.mean(axis=0)).ravel() if issparse(Xc)
                     else np.asarray(Xc).mean(axis=0)).astype(np.float64))
    else:
        raise ValueError("cutoff_source must be 'control' or 'all'.")
    expressed = set(gene_names_all[gene_mean >= expression_threshold].tolist())

    # perturbation selection
    candidate_perts = [str(x) for x in counts.index.astype(str)
                       if str(x) != str(control_label) and int(counts.loc[x]) >= min_samples]
    selected_perts, selected_targets = [], []
    for pert in candidate_perts:
        target, reason = infer_target_gene(pert, gene_set)
        if only_single_gene and reason == "multi_gene":
            continue
        if require_target_in_var and target == "":
            continue
        selected_perts.append(pert)
        selected_targets.append(target if target else pert)
    if not selected_perts:
        raise ValueError("No perturbations survived filtering (check min_samples / target parsing).")

    force = set(t for t in selected_targets if t) if force_include_targets else set()
    keep_genes = (expressed | force) & gene_set
    keep_col_mask = pd.Index(gene_names_all).isin(keep_genes)
    keep_labels = set([str(control_label)] + selected_perts)
    keep_row_mask = pd.Series(labels_all).isin(keep_labels).to_numpy()

    adata = adata[keep_row_mask, keep_col_mask].copy()
    gene_names = np.asarray(adata.var_names.astype(str))
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    target_indices = np.array([gene_to_idx.get(t, -1) for t in selected_targets], dtype=np.int64)

    labels = adata.obs[pert_key].astype(str).to_numpy()
    control_mask = labels == str(control_label)
    stats = {
        "n_cells": int(adata.n_obs), "n_genes": int(len(gene_names)),
        "n_control": int(control_mask.sum()), "n_perturbations": int(len(selected_perts)),
        "expression_threshold": float(expression_threshold), "min_samples": int(min_samples),
    }
    from pathlib import Path as _Path
    return Dataset(adata=adata, pert_key=pert_key, control_label=str(control_label),
                   gene_names=gene_names, perturbations=selected_perts,
                   target_genes=selected_targets, target_gene_indices=target_indices,
                   control_mask=control_mask, stats=stats, name=_Path(str(data_path)).stem,
                   control_full_mean=np.asarray(control_full_mean, dtype=np.float64),
                   control_full_var=np.asarray(control_full_var, dtype=np.float64))


def _subset_dataset_genes(ds: Dataset, shared, shared_pos) -> Dataset:
    """Return a copy of ``ds`` restricted (and reordered) to ``shared`` gene names."""
    adata = ds.adata[:, list(shared)].copy()
    gene_names = np.asarray(adata.var_names.astype(str))
    tgi = np.array([shared_pos.get(str(tg), -1) for tg in ds.target_genes], dtype=np.int64)
    stats = dict(ds.stats)
    stats["n_genes"] = int(len(gene_names))
    return Dataset(adata=adata, pert_key=ds.pert_key, control_label=ds.control_label,
                   gene_names=gene_names, perturbations=list(ds.perturbations),
                   target_genes=list(ds.target_genes), target_gene_indices=tgi,
                   control_mask=ds.control_mask, stats=stats, name=ds.name,
                   control_full_mean=ds.control_full_mean, control_full_var=ds.control_full_var)


def load_matched_datasets(path_a, path_b, **load_kwargs):
    """Load two Perturb-seq datasets restricted to their **shared gene set**.

    Both returned :class:`Dataset` objects share the same ``gene_names`` (in a common
    order), so a covariance estimated on one can be applied to the other's shifts —
    the cross-dataset covariance-transfer analysis.  ``load_kwargs`` are forwarded to
    :func:`load_dataset` for both datasets.
    """
    ds_a = load_dataset(path_a, **load_kwargs)
    ds_b = load_dataset(path_b, **load_kwargs)
    set_b = set(map(str, ds_b.gene_names))
    shared = [g for g in map(str, ds_a.gene_names) if g in set_b]   # ds_a order
    if len(shared) < 2:
        raise ValueError(f"Only {len(shared)} shared genes between {ds_a.name} and {ds_b.name}.")
    shared_pos = {g: i for i, g in enumerate(shared)}
    return _subset_dataset_genes(ds_a, shared, shared_pos), _subset_dataset_genes(ds_b, shared, shared_pos)
