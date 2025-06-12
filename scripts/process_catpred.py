#!/usr/bin/env python3
"""
Script to extract CatPred results into TSV files.
For each protein group, combines kcat and km predictions and maps sequences to their IDs.
"""

import os
import argparse
import pandas as pd
import logging
from Bio import SeqIO
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract CatPred results into TSV files per protein group"
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Input directory containing CatPred outputs"
    )
    parser.add_argument(
        "--faa_dir",
        type=str,
        required=True,
        help="Directory containing FASTA files with protein sequences"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Output directory where extracted TSV files will be saved"
    )
    return parser.parse_args()

def get_sequence_id_map(faa_file):
    """
    Create a mapping from sequence to sequence ID from a FASTA file.
    
    Args:
        faa_file (str): Path to the FASTA file
        
    Returns:
        dict: Dictionary mapping sequence to sequence ID
    """
    seq_to_id = {}
    for record in SeqIO.parse(faa_file, "fasta"):
        seq_to_id[str(record.seq)] = record.id
    return seq_to_id

def process_protein_group(group_name, input_dir, faa_dir):
    """
    Process CatPred results for a specific protein group.
    
    Args:
        group_name (str): Protein group name
        input_dir (str): Path to the directory containing CatPred outputs
        faa_dir (str): Path to the directory containing FASTA files
        
    Returns:
        pd.DataFrame: Combined results dataframe
        bool: Whether processing was successful
    """
    # Define file paths
    kcat_file = Path(input_dir) / group_name / "kcat" / "final_predictions_input.csv"
    km_file = Path(input_dir) / group_name / "km" / "final_predictions_input.csv"
    faa_file = Path(faa_dir) / f"{group_name}.faa"
    
    # Check if all required files exist
    if not all(f.exists() for f in [kcat_file, km_file, faa_file]):
        logger.warning(f"Missing required files for {group_name}, skipping...")
        for f, name in zip([kcat_file, km_file, faa_file], ["kcat file", "km file", "faa file"]):
            if not f.exists():
                logger.warning(f"  - Missing {name}: {f}")
        return None, False
    
    # Read prediction files
    try:
        kcat_df = pd.read_csv(kcat_file)
        km_df = pd.read_csv(km_file)
    except Exception as e:
        logger.error(f"Error reading prediction files for {group_name}: {e}")
        return None, False
    
    # Check if required columns exist
    kcat_cols = ["Substrate", "SMILES", "sequence", "Prediction_(s^(-1))"]
    km_cols = ["Substrate", "SMILES", "sequence", "Prediction_(mM)"]
    
    if not all(col in kcat_df.columns for col in kcat_cols):
        logger.warning(f"Missing required columns in kcat file for {group_name}: {set(kcat_cols) - set(kcat_df.columns)}")
        return None, False
        
    if not all(col in km_df.columns for col in km_cols):
        logger.warning(f"Missing required columns in km file for {group_name}: {set(km_cols) - set(km_df.columns)}")
        return None, False
    
    # Create sequence to ID mapping
    try:
        seq_to_id = get_sequence_id_map(faa_file)
        logger.info(f"Loaded {len(seq_to_id)} sequences from {faa_file}")
    except Exception as e:
        logger.error(f"Error processing FASTA file for {group_name}: {e}")
        return None, False
    
    # Extract relevant columns
    kcat_df = kcat_df[kcat_cols].copy()
    km_df = km_df[km_cols].copy()
    
    # Merge dataframes on common columns
    merged_df = pd.merge(
        kcat_df, 
        km_df[["sequence", "Prediction_(mM)"]],
        on="sequence",
        how="inner"
    )
    
    logger.info(f"Merged data has {len(merged_df)} entries for {group_name}")
    
    # Add sequence ID column
    merged_df["sequence_id"] = merged_df["sequence"].map(seq_to_id)
    
    # Check if all sequences were mapped
    unmapped = merged_df[merged_df["sequence_id"].isna()]
    if not unmapped.empty:
        logger.warning(f"Warning: {len(unmapped)} sequences in {group_name} could not be mapped to IDs")
    
    # Rearrange columns to put sequence_id first
    final_cols = ["sequence_id", "sequence", "Substrate", "SMILES", 
                  "Prediction_(s^(-1))", "Prediction_(mM)"]
    final_df = merged_df[final_cols]
    
    return final_df, True

def main():
    """
    Main function to process all protein groups and generate TSV files.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"FASTA directory: {args.faa_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all protein groups by listing subdirectories in catpred output
    try:
        input_path = Path(args.input_dir)
        protein_groups = [d.name for d in input_path.iterdir() if d.is_dir()]
    except FileNotFoundError:
        logger.error(f"Error: CatPred output directory not found: {args.input_dir}")
        return
    
    if not protein_groups:
        logger.warning(f"No protein groups found in the output directory: {args.input_dir}")
        return
    
    logger.info(f"Found {len(protein_groups)} protein groups: {', '.join(protein_groups)}")
    
    # Process each protein group
    success_count = 0
    for group_name in protein_groups:
        logger.info(f"Processing {group_name}...")
        result_df, success = process_protein_group(group_name, args.input_dir, args.faa_dir)
        
        if success:
            # Save to TSV
            output_file = Path(args.output_dir) / f"{group_name}.tsv"
            result_df.to_csv(output_file, sep='\t', index=False)
            logger.info(f"Successfully saved results for {group_name} to {output_file}")
            logger.info(f"  - Processed {len(result_df)} entries")
            success_count += 1
        else:
            logger.error(f"Failed to process {group_name}")
    
    logger.info(f"Processing complete. Successfully processed {success_count}/{len(protein_groups)} protein groups.")

if __name__ == "__main__":
    main()