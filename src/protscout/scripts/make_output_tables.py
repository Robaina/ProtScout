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
        help="Input directory containing the results subdirectories"
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
                    suffixes=('', f'_{tool}')
                )
        
        if merged_df is None:
            logger.warning(f"No data found for protein group: {protein_group}")
            continue
        
        logger.info(f"  Merged data from tools: {', '.join(tools_found)}")
        
        # Rename specific columns for clarity
        column_renames = {
            "Prediction_(s^(-1))": "kcat_(s^(-1))",
            "Prediction_(mM)": "KM_(mM)"
        }
        
        for old_col, new_col in column_renames.items():
            if old_col in merged_df.columns:
                merged_df = merged_df.rename(columns={old_col: new_col})
                logger.info(f"  Renamed column '{old_col}' to '{new_col}'")
        
        # Calculate catalytic efficiency if both kcat and KM columns exist
        if "kcat_(s^(-1))" in merged_df.columns and "KM_(mM)" in merged_df.columns:
            # Convert KM from mM to M for standard units (M^-1 s^-1)
            merged_df["catalytic_efficiency_(M^(-1)s^(-1))"] = merged_df["kcat_(s^(-1))"] / (merged_df["KM_(mM)"] * 0.001)
            logger.info("  Added catalytic efficiency column (kcat/KM in M^(-1)s^(-1))")
        
        # Handle duplicate columns (e.g., sequence, Substrate, SMILES might appear in multiple tools)
        # Keep only the first occurrence
        seen_cols = set()
        cols_to_keep = []
        for col in merged_df.columns:
            base_col = col.split('_')[0] if '_' in col and col.split('_')[-1] in tools else col
            if base_col in ['sequence', 'Substrate', 'SMILES'] and base_col in seen_cols:
                continue
            seen_cols.add(base_col)
            cols_to_keep.append(col)
        
        merged_df = merged_df[cols_to_keep]
        
        # Sort if requested
        if sort_column is not None:
            if sort_column in merged_df.columns:
                merged_df = merged_df.sort_values(by=sort_column, ascending=ascending)
                logger.info(f"  Sorted results by '{sort_column}' ({'ascending' if ascending else 'descending'})")
            else:
                logger.warning(f"  Sort column '{sort_column}' not found in consolidated data")
        
        # Remove sequence column if it exists (keep only sequence_id)
        if 'sequence' in merged_df.columns:
            merged_df = merged_df.drop(columns=['sequence'])
            logger.info("  Removed 'sequence' column from output")
        
        # Reorder columns to put sequence_id first
        cols = merged_df.columns.tolist()
        if 'sequence_id' in cols:
            cols.remove('sequence_id')
            cols = ['sequence_id'] + cols
            merged_df = merged_df[cols]
        
        # Save consolidated results
        output_filepath = os.path.join(output_dir, f"{protein_group}.tsv")
        merged_df.to_csv(output_filepath, sep='\t', index=False)
        logger.info(f"  Saved consolidated results to {output_filepath}")
        logger.info(f"  Total sequences: {len(merged_df)}")

def main():
    """Main function to execute the consolidation process."""
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all unique protein groups
    protein_groups = find_protein_groups(args.input_dir, args.tools)
    logger.info(f"Found {len(protein_groups)} protein groups: {', '.join(sorted(protein_groups))}")
    
    if not protein_groups:
        logger.error("No protein groups found to consolidate")
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