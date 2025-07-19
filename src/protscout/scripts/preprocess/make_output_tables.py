#!/usr/bin/env python3
"""
Consolidate protein sequence results from multiple tools into single TSV files per protein group.

This script takes results from multiple tools (catpred, catapro, gatsol, geopoc, temberture, temstapro, etc.)
and consolidates them into a single TSV file per protein group, preserving the sequence_id
and all calculated properties from each tool.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import logging
import numpy as np

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
        default=['catpred', 'catapro', 'gatsol', 'geopoc', 'temberture', 'temstapro', 'classical_properties'],
        nargs='+',
        help="Tool names to consolidate (default: catpred catapro gatsol geopoc temberture temstapro classical_properties)"
    )
    parser.add_argument(
        '--sort',
        help="Column name to sort by (e.g., 'TM', 'thermostability_score')"
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

def standardize_column_names(df, tool):
    """
    Standardize column names from different tools for consistency.
    
    Args:
        df: DataFrame with tool results
        tool: Tool name
        
    Returns:
        DataFrame with standardized column names
    """
    # Define column mappings for each tool
    column_mappings = {
        'catpred': {
            "Prediction_(s^(-1))": "kcat_(s^(-1))",
            "Prediction_(mM)": "KM_(mM)"
        },
        'catapro': {
            "pred_log10_kcat": "log10_kcat_(s^(-1))",
            "pred_log10_Km": "log10_KM_(mM)", 
            "pred_log10_kcat_Km": "log10_catalytic_efficiency_(M^(-1)s^(-1))"
        },
        'temstapro': {
            "thermostability_score": "thermostability_score_(°C)"
        },
        'temberture': {
            # TM column should already be correctly named
        },
        'geopoc': {
            # Should have temperature, pH, salt columns
        },
        'gatsol': {
            # Should have solubility column
        },
        'classical_properties': {
            # Should have various classical property columns
        }
    }
    
    if tool in column_mappings:
        df = df.rename(columns=column_mappings[tool])
    
    return df

def calculate_derived_properties(df):
    """
    Calculate derived properties from available data.
    
    Args:
        df: DataFrame with consolidated results
        
    Returns:
        DataFrame with additional derived properties
    """
    # Calculate linear kcat and KM from CataPro log10 values
    if "log10_kcat_(s^(-1))" in df.columns:
        df["kcat_catapro_(s^(-1))"] = 10 ** df["log10_kcat_(s^(-1))"]
        logger.info("  Calculated linear kcat from CataPro log10 values")
        
    if "log10_KM_(mM)" in df.columns:
        df["KM_catapro_(mM)"] = 10 ** df["log10_KM_(mM)"]
        logger.info("  Calculated linear KM from CataPro log10 values")

    # Calculate catalytic efficiency from CatPred if both kcat and KM columns exist
    if "kcat_(s^(-1))" in df.columns and "KM_(mM)" in df.columns:
        # Convert KM from mM to M and calculate catalytic efficiency (kcat/KM)
        df["catalytic_efficiency_catpred_(M^(-1)s^(-1))"] = df["kcat_(s^(-1))"] / (df["KM_(mM)"] * 0.001)
        logger.info("  Calculated CatPred catalytic efficiency (kcat/KM in M^-1s^-1)")
    
    # Calculate catalytic efficiency from CataPro derived values
    if "kcat_catapro_(s^(-1))" in df.columns and "KM_catapro_(mM)" in df.columns:
        # Convert KM from mM to M and calculate catalytic efficiency (kcat/KM)
        df["catalytic_efficiency_catapro_(M^(-1)s^(-1))"] = df["kcat_catapro_(s^(-1))"] / (df["KM_catapro_(mM)"] * 0.001)
        logger.info("  Calculated CataPro catalytic efficiency (kcat/KM in M^-1s^-1)")
    
    return df

def remove_unwanted_columns(df):
    """
    Remove unwanted columns from the final output.
    
    Args:
        df: DataFrame with all columns
        
    Returns:
        DataFrame with unwanted columns removed
    """
    # Remove log10 columns as we only want linear values
    log10_columns = [col for col in df.columns if 'log10_' in col]
    
    # Remove the 'sequence' column if present
    unwanted_columns = log10_columns + ['sequence']
    
    columns_to_remove = [col for col in unwanted_columns if col in df.columns]
    
    if columns_to_remove:
        df = df.drop(columns=columns_to_remove)
        logger.info(f"  Removed columns: {', '.join(columns_to_remove)}")
    
    return df

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
            
            # Standardize column names
            df = standardize_column_names(df, tool)

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

        # Calculate derived properties
        merged_df = calculate_derived_properties(merged_df)

        # Remove unwanted columns (log10 values and sequence column)
        merged_df = remove_unwanted_columns(merged_df)

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
            # CatPred predictions (linear values only)
            'kcat_(s^(-1))', 'KM_(mM)', 'catalytic_efficiency_catpred_(M^(-1)s^(-1))',
            # CataPro predictions (linear values only)
            'kcat_catapro_(s^(-1))', 'KM_catapro_(mM)', 'catalytic_efficiency_catapro_(M^(-1)s^(-1))',
            # Thermostability predictions
            'TM', 'thermostability_score_(°C)',
            # Environmental predictions
            'solubility', 'pH', 'salt', 'temperature'
        ]
        
        # Build the final column list
        final_cols = [col for col in col_order if col in all_cols]
        remaining_cols = [col for col in all_cols if col not in final_cols]
        final_cols.extend(remaining_cols)

        merged_df = merged_df[final_cols]
        logger.info("  Reordered columns for better readability")

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