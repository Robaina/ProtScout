#!/usr/bin/env python3
"""
Consolidate protein sequence results from multiple tools into single TSV files per protein group.

This script takes results from multiple tools (catpred, gatsol, geopoc, temberture, etc.)
and consolidates them into a single TSV file per protein group, preserving the sequence_id
and all calculated properties from each tool.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Consolidate protein sequence results from multiple tools into single TSV files per protein group."
    )
    parser.add_argument(
        '-i', '--input_dir',
        required=True,
        help="Input directory containing the results subdirectories for each tool"
    )
    parser.add_argument(
        '-o', '--output_dir',
        required=True,
        help="Output directory where consolidated TSV files will be saved"
    )
    parser.add_argument(
        '--tools',
        default=['catpred', 'gatsol', 'geopoc', 'temberture', 'classical_properties'],
        nargs='+',
        help="Tool names to consolidate (default: catpred gatsol geopoc temberture classical_properties)"
    )
    parser.add_argument(
        '--sort',
        help="Column name to sort by (e.g., 'TM')"
    )
    parser.add_argument(
        '--ascending',
        action='store_true',
        help="Sort in ascending order (default is descending)"
    )
    return parser.parse_args()

def find_protein_groups(input_dir, tools):
    """
    Identify all unique protein groups from the result files.

    Args:
        input_dir: Base input directory
        tools: List of tool names

    Returns:
        Set of unique protein group names
    """
    protein_groups = set()

    for tool in tools:
        # NOTE: The script expects tool results in a subdirectory named <tool_name>_results
        tool_dir = os.path.join(input_dir, f"{tool}_results")
        if not os.path.isdir(tool_dir):
            logger.warning(f"Tool directory not found: {tool_dir}")
            continue

        for filename in os.listdir(tool_dir):
            if filename.endswith('.tsv'):
                # The filename (without extension) is the protein group name
                protein_group = os.path.splitext(filename)[0]
                protein_groups.add(protein_group)

    if not protein_groups:
        logger.error("No protein groups found in any tool directory")

    return protein_groups

def read_tool_results(input_dir, tool, protein_group):
    """
    Read results from a specific tool for a specific protein group.

    Args:
        input_dir: Base input directory
        tool: Tool name
        protein_group: Protein group name

    Returns:
        DataFrame with the tool's results or None if file not found
    """
    filepath = os.path.join(input_dir, f"{tool}_results", f"{protein_group}.tsv")

    if not os.path.isfile(filepath):
        logger.warning(f"File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath, sep='\t')

        # Ensure sequence_id column exists
        if 'sequence_id' not in df.columns:
            logger.error(f"sequence_id column missing in {filepath}")
            return None

        return df

    except Exception as e:
        logger.error(f"Error reading {filepath}: {str(e)}")
        return None

def consolidate_results(input_dir, output_dir, tools, protein_groups, sort_column=None, ascending=False):
    """
    Consolidate results from all tools for each protein group.

    Args:
        input_dir: Base input directory
        output_dir: Output directory
        tools: List of tool names
        protein_groups: Set of protein group names
        sort_column: Column name to sort by (optional)
        ascending: Sort in ascending order if True, descending if False
    """
    for protein_group in protein_groups:
        logger.info(f"Processing protein group: {protein_group}")

        merged_df = None
        tools_found = []

        for tool in tools:
            df = read_tool_results(input_dir, tool, protein_group)
            if df is None:
                continue

            tools_found.append(tool)

            if merged_df is None:
                merged_df = df
            else:
                # Merge on sequence_id, keeping all rows
                merged_df = pd.merge(
                    merged_df, df,
                    on='sequence_id',
                    how='outer',
                    suffixes=('_x', None) # Prioritize columns from the new dataframe in case of overlap
                )
                # Drop duplicated columns suffixed with '_x'
                merged_df = merged_df[[c for c in merged_df.columns if not c.endswith('_x')]]

        if merged_df is None:
            logger.warning(f"No data found for protein group: {protein_group}")
            continue

        logger.info(f"  Merged data from tools: {', '.join(tools_found)}")

        # Rename specific columns for clarity
        column_renames = {
            "Prediction_(s^(-1))": "kcat_(s^(-1))",
            "Prediction_(mM)": "KM_(mM)"
        }
        merged_df = merged_df.rename(columns=column_renames)

        # Calculate catalytic efficiency if both kcat and KM columns exist
        if "kcat_(s^(-1))" in merged_df.columns and "KM_(mM)" in merged_df.columns:
            merged_df["catalytic_efficiency_(M^(-1)s^(-1))"] = merged_df["kcat_(s^(-1))"] / (merged_df["KM_(mM)"] * 0.001)
            logger.info("  Added catalytic efficiency column (kcat/KM in M^-1s^-1)")

        # Sort if requested before reordering columns
        if sort_column and sort_column in merged_df.columns:
            merged_df = merged_df.sort_values(by=sort_column, ascending=ascending)
            logger.info(f"  Sorted results by '{sort_column}' ({'ascending' if ascending else 'descending'})")
        elif sort_column:
            logger.warning(f"  Sort column '{sort_column}' not found in consolidated data")

        # Reorder columns for better readability
        all_cols = merged_df.columns.tolist()
        
        # Define the desired order
        col_order = [
            # Core identifiers
            'sequence_id', 'Substrate', 'SMILES',
            # Classical properties
            'sequence_length', 'molecular_weight_kda', 'isoelectric_point',
            'instability_index', 'avg_flexibility',
            'extinction_coefficient_reduced', 'extinction_coefficient_oxidized',
            'n_complete', 'c_complete',
            # Key predictions
            'kcat_(s^(-1))', 'KM_(mM)', 'catalytic_efficiency_(M^(-1)s^(-1))', 'TM',
            'solubility', 'pH', 'salt', 'temperature'
        ]
        
        # Build the final column list
        final_cols = [col for col in col_order if col in all_cols]
        remaining_cols = [col for col in all_cols if col not in final_cols and col != 'sequence']
        final_cols.extend(remaining_cols)

        merged_df = merged_df[final_cols]
        logger.info("  Reordered columns for better readability and removed 'sequence' column.")

        # Save consolidated results
        output_filepath = os.path.join(output_dir, f"{protein_group}.tsv")
        merged_df.to_csv(output_filepath, sep='\t', index=False, float_format='%.6f')
        logger.info(f"  Saved consolidated results to {output_filepath}")
        logger.info(f"  Total sequences: {len(merged_df)}")


def main():
    """Main function to execute the consolidation process."""
    args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all unique protein groups
    protein_groups = find_protein_groups(args.input_dir, args.tools)
    if protein_groups:
        logger.info(f"Found {len(protein_groups)} protein groups: {', '.join(sorted(protein_groups))}")
    else:
        logger.error("No protein groups found to consolidate. Check input directory and tool names.")
        return

    # Consolidate results
    consolidate_results(
        args.input_dir,
        args.output_dir,
        args.tools,
        protein_groups,
        args.sort,
        args.ascending
    )

    logger.info("Consolidation complete")

if __name__ == "__main__":
    main()