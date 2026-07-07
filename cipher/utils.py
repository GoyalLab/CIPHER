"""Small filesystem / array helpers shared across CIPHER."""
from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import issparse


def ensure_dir(path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_filename(x) -> str:
    """Turn an arbitrary string into a safe file/dir name."""
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x[:180]


def stable_seed(base_seed: int, name) -> int:
    """Deterministic per-name seed so shuffling nulls are reproducible."""
    h = hashlib.md5(str(name).encode("utf-8")).hexdigest()
    return int((int(base_seed) + int(h[:8], 16)) % (2 ** 32 - 1))


def json_default(o):
    """Fallback JSON encoder for numpy scalars/arrays."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def to_dense(X) -> np.ndarray:
    """Return a dense float array view of a (possibly sparse) matrix."""
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def atomic_save_npy(path, arr, allow_pickle: bool = False) -> None:
    """Write an ``.npy`` file atomically (write to tmp, then rename)."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npy")
    if tmp.exists():
        tmp.unlink()
    np.save(tmp, arr, allow_pickle=allow_pickle)
    if path.exists():
        path.unlink()
    tmp.rename(path)


def compute_sparsity(X) -> float:
    """Fraction of zero entries in ``X`` (works for sparse or dense)."""
    if issparse(X):
        total = X.shape[0] * X.shape[1]
        return float((total - X.nnz) / total) if total else 0.0
    X = np.asarray(X)
    return float(np.mean(X == 0)) if X.size else 0.0
