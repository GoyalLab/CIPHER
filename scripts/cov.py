import argparse
import os
import sys
import logging


parser = argparse.ArgumentParser(description="Process an input file.")
parser.add_argument('-i', '--input', type=str, help='Input file path')
parser.add_argument('-o', '--output', type=str, help='Output file directory')
args = parser.parse_args()

# Fix path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    from src.cov import save_u_samples_summary
    
    path = args.input
    logging.info(f'Started covariance for file: {path}')
    os.makedirs(args.output, exist_ok=True)
    O = os.path.join(args.output, 'samples_summaries')
    save_u_samples_summary(path, output_dir=O, mode="u_samples_summaries")
    save_u_samples_summary(path, output_dir=O, mode="u_samples_summaries_shuff_X0")
    save_u_samples_summary(path, output_dir=O, mode="u_samples_summaries_shuff_Sigma")
