#!/usr/bin/env python3
"""
Script to extract catapro results into TSV files.
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
        description="Extract catapro results into TSV files per protein group"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing catapro outputs"
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
        "--catapro_input_dir",
        type=str,
        help="Directory containing catapro input files and processed_files.txt mapping"
    )
    return parser.parse_args()

def read_processed_files_mapping(catapro_input_dir):
    """
    Read the processed_files.txt to get mapping between substrates and protein groups.
    
    Returns:
        dict: Mapping of substrate_id to list of (input_file, original_fasta) tuples
    """
    mapping_file = Path(catapro_input_dir) / "processed_files.txt"
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

def get_fasta_id_to_sequence_map(faa_file):
    """
    Create a mapping from FASTA ID to sequence from a FASTA file.
    
    Args:
        faa_file (str): Path to the FASTA file
        
    Returns:
        dict: Dictionary mapping FASTA ID to sequence
    """
    id_to_seq = {}
    for record in SeqIO.parse(faa_file, "fasta"):
        id_to_seq[record.id] = str(record.seq)
    return id_to_seq

def process_substrate_results(substrate_id, catapro_output_dir, original_fasta_path):
    """
    Process catapro results for a specific substrate.
    
    Args:
        substrate_id (str): Substrate ID (e.g., "PET")
        catapro_output_dir (str): Path to the catapro output directory
        original_fasta_path (str): Path to the original FASTA file
        
    Returns:
        pd.DataFrame: Combined results dataframe
        bool: Whether processing was successful
    """
    # Look for files ending in "prediction.csv" in the substrate directory
    substrate_dir = Path(catapro_output_dir) / substrate_id
    
    # Debug: List all files in the substrate directory
    if substrate_dir.exists():
        logger.info(f"Files in {substrate_dir}: {[f.name for f in substrate_dir.iterdir()]}")
    else:
        logger.warning(f"Substrate directory does not exist: {substrate_dir}")
        return None, False
    
    # Find files ending with "prediction.csv"
    prediction_files = list(substrate_dir.glob("*prediction.csv"))
    
    if not prediction_files:
        logger.warning(f"No files ending with 'prediction.csv' found in {substrate_dir}")
        return None, False
    
    if len(prediction_files) > 1:
        logger.warning(f"Multiple prediction files found: {[f.name for f in prediction_files]}")
        logger.info(f"Using the first one: {prediction_files[0].name}")
    
    result_file = prediction_files[0]
    logger.info(f"Found results file: {result_file}")
    
    if not Path(original_fasta_path).exists():
        logger.warning(f"Original FASTA file not found: {original_fasta_path}")
        return None, False
    
    # Read prediction file
    try:
        result_df = pd.read_csv(result_file, sep=',')
        logger.info(f"Loaded results from: {result_file}")
        logger.info(f"DataFrame shape: {result_df.shape}")
        logger.info(f"DataFrame columns: {list(result_df.columns)}")
    except Exception as e:
        logger.error(f"Error reading results file for {substrate_id}: {e}")
        return None, False
    
    # Handle cases where the first column might be unnamed (index column)
    if len(result_df.columns) > 0 and (result_df.columns[0] == "Unnamed: 0" or result_df.columns[0] == "" or result_df.columns[0].startswith("Unnamed")):
        logger.info(f"Dropping unnamed index column: {result_df.columns[0]}")
        result_df = result_df.drop(result_df.columns[0], axis=1)
        logger.info(f"Columns after dropping index: {list(result_df.columns)}")
    
    # Check if required columns exist
    required_cols = ["fasta_id", "smiles", "pred_log10[kcat(s^-1)]", "pred_log10[Km(mM)]", "pred_log10[kcat/Km(s^-1mM^-1)]"]
    
    missing_cols = set(required_cols) - set(result_df.columns)
    if missing_cols:
        logger.warning(f"Missing required columns in results file for {substrate_id}: {missing_cols}")
        logger.warning(f"Available columns: {list(result_df.columns)}")
        
        # Try to handle slight variations in column names
        column_mapping = {}
        for req_col in required_cols:
            for df_col in result_df.columns:
                if req_col.lower() == df_col.lower().strip():
                    column_mapping[df_col] = req_col
                    break
        
        if column_mapping:
            logger.info(f"Found column mapping: {column_mapping}")
            result_df = result_df.rename(columns=column_mapping)
            missing_cols = set(required_cols) - set(result_df.columns)
        
        if missing_cols:
            logger.error(f"Still missing required columns: {missing_cols}")
            return None, False
    
    # Create FASTA ID to sequence mapping from original FASTA
    try:
        id_to_seq = get_fasta_id_to_sequence_map(original_fasta_path)
        logger.info(f"Loaded {len(id_to_seq)} sequences from {original_fasta_path}")
    except Exception as e:
        logger.error(f"Error processing FASTA file: {e}")
        return None, False
    
    # Extract relevant columns
    result_df = result_df[required_cols].copy()
    
    # Clean up fasta_id - remove '_wild' suffix if present
    result_df["sequence_id"] = result_df["fasta_id"].str.replace("_wild", "", regex=False)
    
    # Debug: Show some example fasta_ids and sequence_ids
    logger.info(f"Example fasta_ids: {result_df['fasta_id'].head(3).tolist()}")
    logger.info(f"Example sequence_ids: {result_df['sequence_id'].head(3).tolist()}")
    logger.info(f"Example FASTA keys: {list(id_to_seq.keys())[:3]}")
    
    # Add full sequence column
    result_df["sequence"] = result_df["sequence_id"].map(id_to_seq)
    
    # Check if all sequences were mapped
    unmapped = result_df[result_df["sequence"].isna()]
    if not unmapped.empty:
        logger.warning(f"Warning: {len(unmapped)} sequences could not be mapped")
        logger.warning(f"Unmapped sequence IDs: {unmapped['sequence_id'].tolist()[:5]}")  # Show first 5
        
        # Try alternative mapping strategies
        # 1. Try mapping without any prefixes/suffixes
        def clean_id(seq_id):
            # Remove common prefixes and suffixes
            cleaned = seq_id.split('___')[-1] if '___' in seq_id else seq_id
            cleaned = cleaned.replace('_wild', '')
            return cleaned
        
        result_df["cleaned_sequence_id"] = result_df["sequence_id"].apply(clean_id)
        
        # Create alternative mapping
        alternative_mapping = {}
        for fasta_id in id_to_seq.keys():
            cleaned_fasta = clean_id(fasta_id)
            alternative_mapping[cleaned_fasta] = id_to_seq[fasta_id]
        
        # Fill in missing sequences using alternative mapping
        mask = result_df["sequence"].isna()
        result_df.loc[mask, "sequence"] = result_df.loc[mask, "cleaned_sequence_id"].map(alternative_mapping)
        
        # Check again
        still_unmapped = result_df[result_df["sequence"].isna()]
        if not still_unmapped.empty:
            logger.warning(f"Still unmapped after cleaning: {len(still_unmapped)} sequences")
        else:
            logger.info("All sequences successfully mapped after cleaning")
    
    # Rename columns for clarity and consistency
    result_df = result_df.rename(columns={
        "smiles": "SMILES",
        "pred_log10[kcat(s^-1)]": "pred_log10_kcat",
        "pred_log10[Km(mM)]": "pred_log10_Km", 
        "pred_log10[kcat/Km(s^-1mM^-1)]": "pred_log10_kcat_Km"
    })
    
    # Rearrange columns to put sequence_id and sequence first
    final_cols = ["sequence_id", "sequence", "SMILES", 
                  "pred_log10_kcat", "pred_log10_Km", "pred_log10_kcat_Km"]
    final_df = result_df[final_cols]
    
    # Remove rows where sequence mapping failed
    final_df = final_df.dropna(subset=["sequence"])
    
    logger.info(f"Processed data has {len(final_df)} entries for substrate {substrate_id}")
    
    return final_df, True

def main():
    """
    Main function to process all protein groups and generate TSV files.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info(f"catapro output directory: {args.input_dir}")
    logger.info(f"FASTA directory: {args.faa_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the mapping file to understand the directory structure
    # The processed_files.txt is created in the catapro input directory by prepare_catapro_input.py
    # Look for it in several possible locations
    possible_locations = []
    
    # First, try the explicitly provided catapro_input_dir
    if args.catapro_input_dir:
        possible_locations.append(Path(args.catapro_input_dir))
    
    # Then try other common locations
    possible_locations.extend([
        Path(args.input_dir).parent.parent / "data" / "catapro_data",  # Standard location
        Path(args.input_dir).parent / "catapro_data",  # Alternative location
        Path(args.input_dir).parent / "catapro",  # Another alternative
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
        substrate_dirs = [d for d in output_path.iterdir() if d.is_dir()]
        
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