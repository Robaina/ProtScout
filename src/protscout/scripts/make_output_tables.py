#!/usr/bin/env python3
"""
Consolidate protein sequence results from multiple tools into single TSV files per plastic type.

This script takes results from four different tools (catpred, gatsol, geopoc, temberture)
and consolidates them into a single TSV file per plastic type, preserving the sequence_id
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
        description="Consolidate protein sequence results from multiple tools into single TSV files per plastic type."
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
        default=['catpred', 'gatsol', 'geopoc', 'temberture'],
        nargs='+',
        help="Tool names to consolidate (default: catpred gatsol geopoc temberture)"
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

def find_plastic_types(input_dir, tools):
    """
    Identify all unique plastic types from the result files.
    
    Args:
        input_dir: Base input directory
        tools: List of tool names
        
    Returns:
        Set of unique plastic type names
    """
    plastic_types = set()
    
    for tool in tools:
        tool_dir = os.path.join(input_dir, f"{tool}_results")
        if not os.path.isdir(tool_dir):
            logger.warning(f"Tool directory not found: {tool_dir}")
            continue
            
        for filename in os.listdir(tool_dir):
            if filename.endswith('.tsv'):
                # The filename is the plastic type
                plastic_type = os.path.splitext(filename)[0]
                plastic_types.add(plastic_type)
                
    if not plastic_types:
        logger.error("No plastic types found in any tool directory")
        
    return plastic_types

def read_tool_results(input_dir, tool, plastic_type):
    """
    Read results from a specific tool for a specific plastic type.
    
    Args:
        input_dir: Base input directory
        tool: Tool name
        plastic_type: Plastic type name
        
    Returns:
        DataFrame with the tool's results or None if file not found
    """
    filepath = os.path.join(input_dir, f"{tool}_results", f"{plastic_type}.tsv")
    
    if not os.path.isfile(filepath):
        logger.warning(f"File not found: {filepath}")
        return None
        
    try:
        df = pd.read_csv(filepath, sep='\t')
        
        # Ensure sequence_id column exists
        if 'sequence_id' not in df.columns:
            logger.error(f"sequence_id column missing in {filepath}")
            return None
            
        # No longer renaming columns since there are no conflicts
        return df
        
    except Exception as e:
        logger.error(f"Error reading {filepath}: {str(e)}")
        return None

def consolidate_results(input_dir, output_dir, tools, plastic_types, sort_column=None, ascending=False):
    """
    Consolidate results from all tools for each plastic type.
    
    Args:
        input_dir: Base input directory
        output_dir: Output directory
        tools: List of tool names
        plastic_types: Set of plastic type names
        sort_column: Column name to sort by (optional)
        ascending: Sort in ascending order if True, descending if False
    """
    for plastic_type in plastic_types:
        logger.info(f"Processing plastic type: {plastic_type}")
        
        merged_df = None
        
        for tool in tools:
            df = read_tool_results(input_dir, tool, plastic_type)
            if df is None:
                continue
                
            if merged_df is None:
                merged_df = df
            else:
                # Merge on sequence_id
                merged_df = pd.merge(
                    merged_df, df, 
                    on='sequence_id', 
                    how='outer'
                )
        
        if merged_df is None:
            logger.warning(f"No data found for plastic type: {plastic_type}")
            continue
        
        # Rename specific columns
        column_renames = {
            "Prediction_(s^(-1))": "kcat_(s^(-1))",
            "Prediction_(mM)": "KM_(mM)"
        }
        
        for old_col, new_col in column_renames.items():
            if old_col in merged_df.columns:
                merged_df = merged_df.rename(columns={old_col: new_col})
                logger.info(f"Renamed column '{old_col}' to '{new_col}'")
        
        # Calculate catalytic efficiency if both kcat and KM columns exist
        if "kcat_(s^(-1))" in merged_df.columns and "KM_(mM)" in merged_df.columns:
            # Convert KM from mM to M for standard units (M^-1 s^-1)
            merged_df["catalytic_efficiency_(M^(-1)s^(-1))"] = merged_df["kcat_(s^(-1))"] / (merged_df["KM_(mM)"] * 0.001)
            logger.info("Added catalytic efficiency column (kcat/KM in M^(-1)s^(-1))")
            
        # Sort if requested
        if sort_column is not None:
            if sort_column in merged_df.columns:
                merged_df = merged_df.sort_values(by=sort_column, ascending=ascending)
                logger.info(f"Sorted results by '{sort_column}' ({'ascending' if ascending else 'descending'})")
            else:
                logger.warning(f"Sort column '{sort_column}' not found in consolidated data for {plastic_type}")
        
        # Remove sequence column if it exists
        if 'sequence' in merged_df.columns:
            merged_df = merged_df.drop(columns=['sequence'])
            logger.info("Removed 'sequence' column from output")
                
        # Save consolidated results
        output_filepath = os.path.join(output_dir, f"{plastic_type}.tsv")
        merged_df.to_csv(output_filepath, sep='\t', index=False)
        logger.info(f"Saved consolidated results to {output_filepath}")

def main():
    """Main function to execute the consolidation process."""
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all unique plastic types
    plastic_types = find_plastic_types(args.input_dir, args.tools)
    logger.info(f"Found {len(plastic_types)} plastic types: {', '.join(plastic_types)}")
    
    # Consolidate results
    consolidate_results(
        args.input_dir,
        args.output_dir,
        args.tools,
        plastic_types,
        args.sort,
        args.ascending
    )
    
    logger.info("Consolidation complete")

if __name__ == "__main__":
    main()