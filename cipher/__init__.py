"""CIPHER — Covariance Inference for Perturbation and High-dimensional Expression Response.

CIPHER models the mean transcriptomic response to a perturbation with the control
covariance ``Sigma``:  ``delta_x ~= Sigma @ u``.

Typical entry points
--------------------
* :func:`forward_prediction`  — predict a perturbation's transcriptomic shift (1.1)
* :func:`reverse_prediction`  — recover the perturbed gene from the shift (1.2)
* :func:`condition_drivers`   — rank driver genes for any control-vs-condition dataset (2)
* :func:`preprocess_dataset`  — precompute Sigma + per-perturbation stats to disk (0)

See the README for end-to-end examples.  Preprint:
https://www.biorxiv.org/content/10.1101/2025.06.27.661814v1
"""
from __future__ import annotations

__version__ = "0.1.0"

from .data import Dataset, load_dataset, resolve_perturbation_key, infer_target_gene
from .normalize import (
    NORMALIZATION_MODES,
    normalize_matrix,
    fit_pflog_alpha,
    library_size,
    mean_var,
)
from .covariance import (
    compute_covariance,
    null_covariance,
    meanfield_covariance,
    shuffled_covariance,
    zinb_covariance,
)
from .core import (
    forward_predict,
    forward_fit,
    forward_metrics,
    gene_holdout_masks,
    FORWARD_METRICS,
    reverse_scores,
    reverse_operator,
    matched_filter_scores,
    rank_of,
    top_k_hit,
    one_vs_rest_auc,
)
from .forward import forward_prediction, forward_from_precomputed, ForwardResult
from .reverse import (
    reverse_prediction,
    reverse_from_precomputed,
    bayesian_reverse,
    ReverseResult,
)
from .driver import condition_drivers, condition_drivers_from_matrices, DriverResult
from .inverse import (
    posterior_inverse_from_precomputed,
    posterior_inverse_prediction,
    build_model,
    fit_tau2,
    dataset_group,
    InverseResult,
    PosteriorInverseModel,
)
from .preprocess import preprocess_dataset, PreprocessConfig
from .io import load_precomputed, list_modes, PrecomputedMode

__all__ = [
    "__version__",
    # data
    "Dataset", "load_dataset", "resolve_perturbation_key", "infer_target_gene",
    # normalize
    "NORMALIZATION_MODES", "normalize_matrix", "fit_pflog_alpha", "library_size", "mean_var",
    # covariance
    "compute_covariance", "null_covariance", "meanfield_covariance",
    "shuffled_covariance", "zinb_covariance",
    # core
    "forward_predict", "forward_fit", "forward_metrics", "gene_holdout_masks",
    "FORWARD_METRICS", "reverse_scores", "reverse_operator",
    "matched_filter_scores", "rank_of", "top_k_hit", "one_vs_rest_auc",
    # forward
    "forward_prediction", "forward_from_precomputed", "ForwardResult",
    # reverse
    "reverse_prediction", "reverse_from_precomputed", "bayesian_reverse", "ReverseResult",
    # driver
    "condition_drivers", "condition_drivers_from_matrices", "DriverResult",
    # inverse (fullH_diag posterior)
    "posterior_inverse_from_precomputed", "posterior_inverse_prediction",
    "build_model", "fit_tau2", "dataset_group", "InverseResult", "PosteriorInverseModel",
    # preprocess / io
    "preprocess_dataset", "PreprocessConfig",
    "load_precomputed", "list_modes", "PrecomputedMode",
]
