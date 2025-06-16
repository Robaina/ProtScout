#!/usr/bin/env python3
"""
Script to extract CatPred results into TSV files.
Maps substrate-based results back to protein groups using the processed_files.txt mapping.
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
        "--input_dir",
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
        "--output_dir",
        type=str,
        required=True,
        help="Output directory where extracted TSV files will be saved"
    )
    parser.add_argument(
        "--catpred_input_dir",
        type=str,
        help="Directory containing CatPred input files and processed_files.txt mapping"
    )
    return parser.parse_args()

def read_processed_files_mapping(catpred_input_dir):
    """
    Read the processed_files.txt to get mapping between substrates and protein groups.
    
    Returns:
        dict: Mapping of substrate_id to list of (input_file, original_fasta) tuples
    """
    mapping_file = Path(catpred_input_dir) / "processed_files.txt"
    if not mapping_file.exists():
        logger.warning(f"Mapping file not found: {mapping_file}")
        return {}
    
    mapping = {}
    with open(mapping_file, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                substrate_id, input_file, original_fasta = parts
                if substrate_id not in mapping:
                    mapping[substrate_id] = []
                mapping[substrate_id].append((input_file, original_fasta))
    
    return mapping

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

def process_substrate_results(substrate_id, catpred_output_dir, original_fasta_path):
    """
    Process CatPred results for a specific substrate.
    
    Args:
        substrate_id (str): Substrate ID (e.g., "PET")
        catpred_output_dir (str): Path to the CatPred output directory
        original_fasta_path (str): Path to the original FASTA file
        
    Returns:
        pd.DataFrame: Combined results dataframe
        bool: Whether processing was successful
    """
    # Define file paths
    kcat_file = Path(catpred_output_dir) / substrate_id / "kcat" / "final_predictions_input.csv"
    km_file = Path(catpred_output_dir) / substrate_id / "km" / "final_predictions_input.csv"
    
    # Check if required files exist
    if not all(f.exists() for f in [kcat_file, km_file]):
        logger.warning(f"Missing required files for substrate {substrate_id}")
        for f, name in zip([kcat_file, km_file], ["kcat file", "km file"]):
            if not f.exists():
                logger.warning(f"  - Missing {name}: {f}")
        return None, False
    
    if not Path(original_fasta_path).exists():
        logger.warning(f"Original FASTA file not found: {original_fasta_path}")
        return None, False
    
    # Read prediction files
    try:
        kcat_df = pd.read_csv(kcat_file)
        km_df = pd.read_csv(km_file)
    except Exception as e:
        logger.error(f"Error reading prediction files for {substrate_id}: {e}")
        return None, False
    
    # Check if required columns exist
    kcat_cols = ["Substrate", "SMILES", "sequence", "Prediction_(s^(-1))"]
    km_cols = ["Substrate", "SMILES", "sequence", "Prediction_(mM)"]
    
    if not all(col in kcat_df.columns for col in kcat_cols):
        logger.warning(f"Missing required columns in kcat file for {substrate_id}: {set(kcat_cols) - set(kcat_df.columns)}")
        return None, False
        
    if not all(col in km_df.columns for col in km_cols):
        logger.warning(f"Missing required columns in km file for {substrate_id}: {set(km_cols) - set(km_df.columns)}")
        return None, False
    
    # Create sequence to ID mapping from original FASTA
    try:
        seq_to_id = get_sequence_id_map(original_fasta_path)
        logger.info(f"Loaded {len(seq_to_id)} sequences from {original_fasta_path}")
    except Exception as e:
        logger.error(f"Error processing FASTA file: {e}")
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
    
    logger.info(f"Merged data has {len(merged_df)} entries for substrate {substrate_id}")
    
    # Add sequence ID column
    merged_df["sequence_id"] = merged_df["sequence"].map(seq_to_id)
    
    # Check if all sequences were mapped
    unmapped = merged_df[merged_df["sequence_id"].isna()]
    if not unmapped.empty:
        logger.warning(f"Warning: {len(unmapped)} sequences could not be mapped to IDs")
    
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
    
    logger.info(f"CatPred output directory: {args.input_dir}")
    logger.info(f"FASTA directory: {args.faa_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the mapping file to understand the directory structure
    # The processed_files.txt is created in the catpred input directory by prepare_catpred_inputs.py
    # Look for it in several possible locations
    possible_locations = []
    
    # First, try the explicitly provided catpred_input_dir
    if args.catpred_input_dir:
        possible_locations.append(Path(args.catpred_input_dir))
    
    # Then try other common locations
    possible_locations.extend([
        Path(args.input_dir).parent.parent / "data" / "catpred_data",  # Standard location
        Path(args.input_dir).parent / "catpred_data",  # Alternative location
        Path(args.input_dir).parent / "catpred",  # Another alternative
        Path(args.input_dir),  # Try the input_dir itself
    ])
    
    mapping = {}
    for location in possible_locations:
        if location.exists():
            mapping = read_processed_files_mapping(location)
            if mapping:
                logger.info(f"Found mapping file in: {location}")
                break
    
    if not mapping:
        # If we still don't have a mapping, try to infer from the directory structure
        logger.warning("No mapping file found, attempting to infer from directory structure...")
        # Look for substrate directories in the output
        output_path = Path(args.input_dir)
        substrate_dirs = [d for d in output_path.iterdir() if d.is_dir() and (d / "kcat").exists()]
        
        if substrate_dirs:
            # Create a synthetic mapping
            for substrate_dir in substrate_dirs:
                substrate_id = substrate_dir.name
                # Find all fasta files in the faa_dir
                fasta_files = list(Path(args.faa_dir).glob("*.faa"))
                for fasta_file in fasta_files:
                    if substrate_id not in mapping:
                        mapping[substrate_id] = []
                    mapping[substrate_id].append(("dummy_input", str(fasta_file)))
            logger.info(f"Inferred mapping for substrates: {list(mapping.keys())}")
    
    if not mapping:
        logger.error("No processed files mapping found. Cannot determine protein groups.")
        return
    
    # Process each substrate and map back to protein groups
    processed_groups = {}
    
    for substrate_id, file_list in mapping.items():
        logger.info(f"Processing substrate: {substrate_id}")
        
        for input_file, original_fasta in file_list:
            # Extract protein group name from original FASTA path
            fasta_basename = Path(original_fasta).stem  # e.g., "sequences" from "sequences.faa"
            
            logger.info(f"  Processing results for protein group: {fasta_basename}")
            
            result_df, success = process_substrate_results(
                substrate_id, 
                args.input_dir, 
                original_fasta
            )
            
            if success:
                # Save to TSV using protein group name, not substrate name
                output_file = Path(args.output_dir) / f"{fasta_basename}.tsv"
                result_df.to_csv(output_file, sep='\t', index=False)
                logger.info(f"  Successfully saved results to {output_file}")
                logger.info(f"  - Processed {len(result_df)} entries")
                processed_groups[fasta_basename] = True
            else:
                logger.error(f"  Failed to process results for {fasta_basename}")
    
    logger.info(f"Processing complete. Successfully processed {len(processed_groups)} protein groups.")

if __name__ == "__main__":
    main()