"""
Compute MGC/Tumor Ratio Script

This script takes two input CSV files:
1. A CSV file with the number of detected MGCs (multi-nucleated giant cells).
2. A CSV file with the number of detected tumor cells.

For each WSI (whole slide image), the script computes the ratio of MGC to tumor cells and writes the results to a new CSV file.

Usage:
    python compute_mgc_tumor_ratio.py --mgc-csv <path_to_mgc_csv> --tumor-csv <path_to_tumor_csv> --output-csv <path_to_output_csv>

Example:
    python compute_mgc_tumor_ratio.py --mgc-csv mgc_detections.csv --tumor-csv tumor_detections.csv --output-csv mgc_tumor_ratio.csv
"""

import argparse

import pandas as pd


def compute_ratio(mgc_csv, tumor_csv, output_csv):
    """
    Compute the ratio of MGC to tumor cells for each WSI and write the results to a CSV file.

    Args:
        mgc_csv (str): Path to the CSV file with number of detected MGC.
        tumor_csv (str): Path to the CSV file with number of detected tumor cells.
        output_csv (str): Path to the output CSV file.
    """
    # Load the CSV files
    mgc_df = pd.read_csv(mgc_csv)
    tumor_df = pd.read_csv(tumor_csv)

    # Merge the dataframes on the WSI column
    merged_df = pd.merge(mgc_df, tumor_df, on='WSI', suffixes=('_mgc', '_tumor'))

    # Compute the ratio of MGC to tumor cells
    merged_df['MGC_Tumor_Ratio'] = merged_df['Detections_mgc'] / merged_df['Detections_tumor']

    # Write the results to the output CSV file
    merged_df.to_csv(output_csv, index=False)


def main():
    """
    Main function to parse arguments and compute the ratio.
    """
    parser = argparse.ArgumentParser(description="Compute MGC/Tumor Ratio Script")
    parser.add_argument('--mgc-csv', type=str, required=True, help='Path to the CSV file with number of detected MGC')
    parser.add_argument('--tumor-csv', type=str, required=True,
                        help='Path to the CSV file with number of detected tumor cells')
    parser.add_argument('--output-csv', type=str, required=True, help='Path to the output CSV file')
    args = parser.parse_args()

    # Compute the ratio and write the results
    compute_ratio(args.mgc_csv, args.tumor_csv, args.output_csv)


if __name__ == "__main__":
    main()
