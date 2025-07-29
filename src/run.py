import os
import argparse

from src.r2 import full_analysis_with_nulls_soft_and_plots

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def parse_args():
    parser = argparse.ArgumentParser(description='Run GEARS model')
    parser.add_argument('--input', type=str, required=True, help='Path to h5ad input file')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    return parser.parse_args()

def run_cipher(params) -> None:
    # Unpack arguments
    adata_p = params.input                   # path/to/data/ds/perturb_processed.h5ad
    out_dir = params.outdir
    # Set I/O
    o = os.path.join(out_dir, 'cipher')
    os.makedirs(o, exist_ok=True)
    # Usage of CIPHER module
    _ = full_analysis_with_nulls_soft_and_plots(adata_p, save_dir=o)


if __name__ == '__main__':
    args = parse_args()
    run_cipher(args)
