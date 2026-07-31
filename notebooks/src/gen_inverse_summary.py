"""Precompute GENERATOR for Fig S9 (inverse-inference true-perturbation gene ranking).

Produces the exact precompute layout that ``notebooks/src/run_figS9.py`` (the
consumer for ``notebooks/suppl/figS9_generanking.ipynb``) reads. It regenerates
the ``posterior_inverse_fast_from_prerun_fullH_diag`` run tree from the base
Perturb-seq ``.h5ad`` files using the installed ``cipher`` package.

Consumer contract (see ``run_figS9.load_and_summarize`` +
``suppl_generanking.find_latest_run``)::

    <out_root>/posterior_inverse_fast_from_prerun_fullH_diag/
        mean_ge_1p0/
            run_<YYYYmmdd_HHMMSS>/
                ALL_DATASETS_inverse_summary.csv      # must have a "dataset" column
                <dataset>/
                    perpert_metrics.npz               # per-perturbation ranks

``perpert_metrics.npz`` keys read by the consumer:

    perturbations      (str,  n_pert)   required
    target_genes       (str,  n_pert)   required
    target_idx         (int,  n_pert)   required   (-1 == unmatched)
    <method>_rank      (float,n_pert)   optional; NaN for unmatched / invalid

    where <method> in METHODS_TO_LOAD =
        score_pval, score_pip_full, score_lfc_abs,
        score_shuffle, score_mean_field, score_true

    Saved rank is the 1-indexed absolute rank of the true target gene under the
    method's per-gene score (rank == 1 -> true gene ranked first). Matches the
    original prerun convention: rank = 1 + (# genes scoring strictly higher).

``out_root`` here corresponds to the consumer's ``SUPPL`` directory
(``$CIPHER_DATA_DIR/suppl``); ``posterior_inverse_fast_from_prerun_fullH_diag``
is created beneath it.

The two figure-consuming methods are ``score_lfc_abs`` and ``score_true``; the
other rank arrays are produced for completeness (the consumer loads whatever is
present and skips the rest). ``score_pval`` is not produced (no p-value pass);
the consumer prints ``[method skip] score_pval_rank not found`` and continues.

NOT part of the installable ``cipher`` package -- a notebook-only reproduction
helper.
"""
from __future__ import annotations

import glob
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import cipher
from cipher.inverse import (
    build_model,
    fit_tau2,
    posterior_scores_batch,
    pip_scores_batch,
)
from cipher.covariance import (
    compute_covariance,
    meanfield_covariance,
    shuffled_covariance,
)
from cipher.normalize import normalize_matrix, library_size, mean_var
from cipher.utils import stable_seed


# ------------------------------------------------------------------
# Config (module globals so the run can be reproduced deterministically)
# ------------------------------------------------------------------
PRECOMP_SUBDIR = "posterior_inverse_fast_from_prerun_fullH_diag"
EXPRESSION_CUTOFF = 1.0                # -> mean_ge_1p0 (matches find_latest_run)
NORMALIZATION = "raw"               # single normalization stored per dataset
PERT_BATCH = 128
COV_MAX_CELLS = 10000
NEGATIVES_PER_PERT = 512
SEED = 0

# methods written into perpert_metrics.npz (score_pval intentionally omitted)
METHODS = [
    "score_true",
    "score_mean_field",
    "score_shuffle",
    "score_pip_full",
    "score_lfc_abs",
]


def _cutoff_to_tag(value: float) -> str:
    return f"{float(value):.1f}".replace(".", "p")


def _abs_rank(scores: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    """1-indexed absolute rank of the true target under per-gene ``scores``.

    Mirrors ``cipher.inverse._ScoreAccumulator``: rank = 1 + (# genes scoring
    strictly higher than the target). NaN when the target is unmatched
    (``target_idx < 0``) or the target's score is non-finite.
    """
    scores = np.nan_to_num(
        np.asarray(scores, dtype=np.float64),
        nan=-np.inf, posinf=np.inf, neginf=-np.inf,
    )
    n_pert, n_gene = scores.shape
    ranks = np.full(n_pert, np.nan, dtype=np.float64)
    tgi = np.asarray(target_idx, dtype=np.int64)
    for i in range(n_pert):
        t = int(tgi[i])
        if t < 0 or t >= n_gene:
            continue
        tscore = scores[i, t]
        if not np.isfinite(tscore):
            continue
        ranks[i] = 1.0 + float(np.sum(scores[i] > tscore))
    return ranks


def _posterior_ranks(model, dx, target_idx, tau2, method):
    """Batch-score every perturbation and return 1-indexed absolute ranks."""
    n_pert, n_gene = dx.shape
    score_fn = pip_scores_batch if method == "pip" else posterior_scores_batch
    scores = np.empty((n_pert, n_gene), dtype=np.float64)
    for a in range(0, n_pert, PERT_BATCH):
        b = min(a + PERT_BATCH, n_pert)
        sc = score_fn(model, np.asarray(dx[a:b], dtype=np.float64), a, b, tau2)
        scores[a:b] = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
    return _abs_rank(scores, target_idx)


def _process_dataset(h5ad_path: str, run_dir: Path, normalization: str) -> dict:
    """Compute per-perturbation ranks for one dataset and write perpert_metrics.npz.

    Returns a summary-row dict (one row of ALL_DATASETS_inverse_summary.csv).
    """
    ds = cipher.load_dataset(h5ad_path)
    ds_name = ds.name

    control_raw = ds.control_matrix(dense=True)
    pseudocount = ds.pflog_pseudocount if normalization == "pflog" else None

    def _norm(X):
        return normalize_matrix(
            X, normalization, libsize=library_size(X), pseudocount=pseudocount
        )

    control_norm = _norm(control_raw)
    control_mean = control_norm.mean(axis=0)
    control_var = control_norm.var(axis=0, ddof=1)
    n0 = float(control_norm.shape[0])

    # covariance sample (mirrors posterior_inverse_prediction subsampling)
    cov_norm = control_norm
    if COV_MAX_CELLS is not None and control_norm.shape[0] > COV_MAX_CELLS:
        rng = np.random.default_rng(stable_seed(SEED, f"{ds_name}:{normalization}"))
        sel = np.sort(rng.choice(control_norm.shape[0], COV_MAX_CELLS, replace=False))
        cov_norm = control_norm[sel]

    perts = np.asarray(ds.perturbations).astype(str)
    tgi = np.asarray(ds.target_gene_indices, dtype=np.int64)
    n_pert, p = len(perts), ds.n_genes

    # per-perturbation dx / var_pert / nu (single pass)
    dx = np.empty((n_pert, p), dtype=np.float64)
    var_pert = np.empty((n_pert, p), dtype=np.float64)
    nu = np.empty(n_pert, dtype=np.float64)
    for i, pert in enumerate(perts):
        Yp = _norm(ds.perturbation_matrix(pert, dense=True))
        nu[i] = Yp.shape[0]
        m, v = mean_var(Yp)
        dx[i] = m - control_mean
        var_pert[i] = v

    gene_names = np.asarray(ds.gene_names).astype(str)
    target_genes = np.where(
        tgi >= 0, gene_names[np.clip(tgi, 0, p - 1)], ""
    ).astype(str)

    rank_arrays: dict[str, np.ndarray] = {}

    # --- |LFC| baseline: rank genes by |dx| ---
    rank_arrays["score_lfc_abs"] = _abs_rank(np.abs(dx), tgi)

    # --- true covariance posterior + PIP ---
    Sigma_true = compute_covariance(cov_norm)
    model_true = build_model(Sigma_true, var_pert, n0, nu, control_var=control_var)
    tau2_true = float(fit_tau2(model_true, dx, batch=PERT_BATCH, plateau=True)["tau2_use"])
    rank_arrays["score_true"] = _posterior_ranks(
        model_true, dx, tgi, tau2_true, method="posterior"
    )
    rank_arrays["score_pip_full"] = _posterior_ranks(
        model_true, dx, tgi, tau2_true, method="pip"
    )

    # --- mean-field covariance posterior ---
    Sigma_mf = meanfield_covariance(cov_norm, seed=SEED)
    model_mf = build_model(Sigma_mf, var_pert, n0, nu, control_var=control_var)
    tau2_mf = float(fit_tau2(model_mf, dx, batch=PERT_BATCH, plateau=True)["tau2_use"])
    rank_arrays["score_mean_field"] = _posterior_ranks(
        model_mf, dx, tgi, tau2_mf, method="posterior"
    )

    # --- shuffled covariance posterior ---
    Sigma_shuf = shuffled_covariance(cov_norm, seed=SEED)
    model_shuf = build_model(Sigma_shuf, var_pert, n0, nu, control_var=control_var)
    tau2_shuf = float(fit_tau2(model_shuf, dx, batch=PERT_BATCH, plateau=True)["tau2_use"])
    rank_arrays["score_shuffle"] = _posterior_ranks(
        model_shuf, dx, tgi, tau2_shuf, method="posterior"
    )

    # --- write perpert_metrics.npz ---
    dataset_dir = run_dir / ds_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "perturbations": perts.astype(object),
        "target_genes": target_genes.astype(object),
        "target_idx": tgi.astype(np.int64),
    }
    for method, ranks in rank_arrays.items():
        npz_payload[f"{method}_rank"] = ranks.astype(np.float64)
    np.savez(dataset_dir / "perpert_metrics.npz", **npz_payload)

    # --- summary row ---
    n_matched = int(np.sum(tgi >= 0))
    true_valid = rank_arrays["score_true"][np.isfinite(rank_arrays["score_true"])]
    summary = {
        "dataset": ds_name,
        "dataset_group": cipher.dataset_group(ds_name),
        "normalization": normalization,
        "n_perturbations": int(n_pert),
        "n_genes": int(p),
        "n_matched_targets": n_matched,
        "tau2_true": tau2_true,
        "tau2_mean_field": tau2_mf,
        "tau2_shuffle": tau2_shuf,
        "median_rank_true": float(np.median(true_valid)) if true_valid.size else np.nan,
        "top1_true": float(np.mean(true_valid <= 1)) if true_valid.size else np.nan,
        "top10_true": float(np.mean(true_valid <= 10)) if true_valid.size else np.nan,
        "methods": "|".join(rank_arrays.keys()),
    }
    return summary


def generate(data_dir, out_root, datasets=None, normalization: str = NORMALIZATION):
    """Generate the Fig S9 inverse precompute under ``out_root``.

    Parameters
    ----------
    data_dir : str
        Directory containing the base ``*.h5ad`` files.
    out_root : str
        Corresponds to the consumer's ``SUPPL`` directory. The
        ``posterior_inverse_fast_from_prerun_fullH_diag/mean_ge_1p0/run_<ts>/``
        tree is created beneath it.
    datasets : list[str] | None
        Dataset stems (basename without ``.h5ad``) to process. When ``None``,
        every ``*.h5ad`` whose ``cipher.dataset_group`` is ``CRISPRi`` or
        ``CRISPRa`` is processed.
    normalization : str
        Normalization mode passed to the inverse (single mode stored per dataset).

    Returns
    -------
    pathlib.Path
        The created ``run_<ts>`` directory (RUN_OUT).
    """
    data_dir = Path(data_dir)

    # resolve dataset -> h5ad path
    all_h5ads = sorted(glob.glob(str(data_dir / "*.h5ad")))
    stem_to_path = {Path(p).stem: p for p in all_h5ads}

    if datasets is None:
        selected = [
            stem for stem in stem_to_path
            if cipher.dataset_group(stem) in {"CRISPRi", "CRISPRa"}
        ]
    else:
        selected = list(datasets)

    selected = sorted(selected)
    missing = [s for s in selected if s not in stem_to_path]
    if missing:
        raise FileNotFoundError(
            f"Requested datasets not found under {data_dir}:\n{missing}"
        )

    expression_tag = _cutoff_to_tag(EXPRESSION_CUTOFF)
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(out_root) / PRECOMP_SUBDIR / f"mean_ge_{expression_tag}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen] RUN_OUT = {run_dir}")
    print(f"[gen] {len(selected)} dataset(s): {selected}")

    summary_rows = []
    errors: dict[str, str] = {}
    for index, stem in enumerate(selected, start=1):
        h5ad_path = stem_to_path[stem]
        print(f"[gen] [{index:02d}/{len(selected):02d}] {stem}")
        t0 = time.time()
        try:
            row = _process_dataset(h5ad_path, run_dir, normalization)
            summary_rows.append(row)
            print(
                f"    ok  n_pert={row['n_perturbations']} "
                f"median_rank_true={row['median_rank_true']} "
                f"({time.time() - t0:.1f}s)"
            )
        except Exception as exc:  # noqa: BLE001 - record & continue like the prerun
            errors[stem] = repr(exc)
            print(f"    [error] {exc!r}")

    if not summary_rows:
        raise RuntimeError("No datasets were processed successfully.")

    combined = pd.DataFrame(summary_rows)
    combined_path = run_dir / "ALL_DATASETS_inverse_summary.csv"
    combined.to_csv(combined_path, index=False)

    with open(run_dir / "ALL_DATASETS_errors.json", "w", encoding="utf-8") as handle:
        json.dump(errors, handle, indent=2)

    print(f"[gen] wrote {combined_path} ({len(combined)} rows)")
    return run_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Fig S9 inverse precompute.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("CIPHER_DATA_DIR"),
        help="Directory with base *.h5ad files (default: $CIPHER_DATA_DIR).",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Consumer SUPPL directory (default: $CIPHER_DATA_DIR/suppl).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset stems to process (default: all CRISPRi/CRISPRa).",
    )
    parser.add_argument("--normalization", default=NORMALIZATION)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_root = args.out_root or os.path.join(data_dir, "suppl")
    generate(data_dir, out_root, datasets=args.datasets, normalization=args.normalization)
