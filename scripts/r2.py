import argparse
import os
import sys


parser = argparse.ArgumentParser(description="Process an input file.")
parser.add_argument('-i', '--input', type=str, help='Input file path')
parser.add_argument('-o', '--output', type=str, help='Output file directory')
args = parser.parse_args()

# Fix path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    from src.r2 import full_analysis_with_nulls_soft_and_plots

    os.makedirs(args.output, exist_ok=True)
    O = os.path.join(args.output, 'r2_histograms')
    full_analysis_with_nulls_soft_and_plots(args.input, O)
