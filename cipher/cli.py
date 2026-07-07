"""Command-line interface for CIPHER.

Installed as the ``cipher`` console script (see ``pyproject.toml``)::

    cipher preprocess data.h5ad -o out/ --modes log1p log1CP10k
    cipher forward    data.h5ad -o out/ --normalization log1p
    cipher reverse    data.h5ad -o out/ --normalization log1p --method pinv
    cipher driver     data.h5ad -o out/ --condition-key stim --control rest --condition stim
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _add_common(p):
    p.add_argument("input", help="Path to an input .h5ad file")
    p.add_argument("-o", "--output", required=True, help="Output directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cipher", description="CIPHER command-line interface")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    sub = parser.add_subparsers(dest="command")

    pp = sub.add_parser("preprocess", help="Precompute Sigma + perturbation stats to disk")
    _add_common(pp)
    pp.add_argument("--modes", nargs="+", default=None,
                    help="Normalization modes (default: all)")
    pp.add_argument("--expression-threshold", type=float, default=1.0)
    pp.add_argument("--min-samples", type=int, default=100)
    pp.add_argument("--cov-max-cells", type=int, default=10000)
    pp.add_argument("--overwrite", action="store_true")

    fw = sub.add_parser("forward", help="Forward prediction (transcriptomic shift)")
    _add_common(fw)
    fw.add_argument("--normalization", default="log1p")
    fw.add_argument("--nulls", nargs="*", default=["meanfield", "shuffled"])
    fw.add_argument("--expression-threshold", type=float, default=1.0)
    fw.add_argument("--min-samples", type=int, default=100)
    fw.add_argument("--max-perturbations", type=int, default=None)

    rv = sub.add_parser("reverse", help="Reverse prediction (recover perturbed gene)")
    _add_common(rv)
    rv.add_argument("--normalization", default="log1p")
    rv.add_argument("--method", default="matched_filter",
                    choices=["pinv", "ridge", "lstsq", "matched_filter"])
    rv.add_argument("--top-k", type=int, default=10)
    rv.add_argument("--expression-threshold", type=float, default=1.0)
    rv.add_argument("--min-samples", type=int, default=100)
    rv.add_argument("--max-perturbations", type=int, default=None)

    dr = sub.add_parser("driver", help="Condition-driver prediction (control vs condition)")
    _add_common(dr)
    dr.add_argument("--condition-key", required=True, help="obs column grouping cells")
    dr.add_argument("--control", required=True, dest="control_value",
                    help="value of condition-key marking controls")
    dr.add_argument("--condition", default=None, dest="condition_value",
                    help="value marking the condition (default: all non-control)")
    dr.add_argument("--normalization", default="log1p")
    dr.add_argument("--method", default="matched_filter",
                    choices=["pinv", "ridge", "lstsq", "matched_filter"])
    dr.add_argument("--top", type=int, default=25, help="rows to print")
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = build_parser().parse_args(argv)

    import cipher
    if getattr(args, "version", False) and not args.command:
        print(cipher.__version__)
        return 0
    if not args.command:
        build_parser().print_help()
        return 1

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.command == "preprocess":
        cfg = cipher.PreprocessConfig(
            expression_threshold=args.expression_threshold,
            min_samples_per_pert=args.min_samples,
            cov_max_cells=args.cov_max_cells,
        )
        out = cipher.preprocess_dataset(args.input, outdir, modes=args.modes,
                                        config=cfg, overwrite=args.overwrite)
        print(f"Preprocessed -> {out}")
        return 0

    if args.command == "forward":
        res = cipher.forward_prediction(
            args.input, normalization=args.normalization, nulls=args.nulls,
            max_perturbations=args.max_perturbations,
            expression_threshold=args.expression_threshold, min_samples=args.min_samples)
        path = res.save(outdir)
        print(json.dumps(res.summary, indent=2))
        print(f"Saved -> {path}")
        return 0

    if args.command == "reverse":
        res = cipher.reverse_prediction(
            args.input, normalization=args.normalization, method=args.method,
            top_k=args.top_k, max_perturbations=args.max_perturbations,
            expression_threshold=args.expression_threshold, min_samples=args.min_samples)
        path = res.save(outdir)
        print(json.dumps(res.summary, indent=2))
        print(f"Saved -> {path}")
        return 0

    if args.command == "driver":
        res = cipher.condition_drivers(
            args.input, condition_key=args.condition_key, control_value=args.control_value,
            condition_value=args.condition_value, normalization=args.normalization,
            method=args.method)
        path = res.save(outdir)
        print(res.top(args.top).to_string(index=False))
        print(f"Saved -> {path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
