#!/usr/bin/env python3
"""
Script to extract TemStaPro results into TSV files.
Maps substrate-based results back to protein groups using the processed_files.txt mapping.
FIXED: Now includes support for *_predictions_mean.tsv files
"""

import os
import argparse
import pandas as pd
import numpy as np
import logging
from Bio import SeqIO
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Dict, Literal
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract TemStaPro results into TSV files per protein group"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing TemStaPro outputs"
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
        "--temstapro_input_dir",
        type=str,
        help="Directory containing TemStaPro input files and processed_files.txt mapping"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tm_interpolation", "weighted_temp", "range_midpoint"],
        default="tm_interpolation",
        help="Thermostability scoring method (default: tm_interpolation)"
    )
    return parser.parse_args()

def read_processed_files_mapping(temstapro_input_dir):
    """
    Read the processed_files.txt to get mapping between substrates and protein groups.
    
    Returns:
        dict: Mapping of substrate_id to list of (input_file, original_fasta) tuples
    """
    mapping_file = Path(temstapro_input_dir) / "processed_files.txt"
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

def calculate_tm_interpolation(raw_scores, temperatures=[40, 45, 50, 55, 60, 65]):
    """Calculate interpolated melting temperature."""
    temps = np.array(temperatures)
    scores = np.array(raw_scores)
    
    try:
        interpolator = interp1d(temps, scores, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        
        temp_range = np.linspace(temps.min() - 10, temps.max() + 10, 1000)
        interpolated_scores = interpolator(temp_range)
        crossing_idx = np.argmin(np.abs(interpolated_scores - 0.5))
        
        return float(temp_range[crossing_idx])
    except:
        return np.nan

def calculate_weighted_temp(raw_scores, temperatures=[40, 45, 50, 55, 60, 65]):
    """Calculate weighted temperature score."""
    weights = np.array(raw_scores)
    temps = np.array(temperatures)
    total_weight = np.sum(weights)
    
    if total_weight == 0:
        return float(np.mean(temperatures))
    return float(np.sum(weights * temps) / total_weight)

def parse_range_midpoint(range_label):
    """Parse temperature range and return midpoint."""
    if pd.isna(range_label):
        return 50.0
    numbers = re.findall(r'\d+', str(range_label))
    if len(numbers) >= 2:
        return (float(numbers[0]) + float(numbers[1])) / 2
    return 50.0  # Default fallback

def find_temstapro_result_file(substrate_id, temstapro_output_dir):
    """
    Find the TemStaPro result file for a given substrate.
    UPDATED: Now includes *_predictions_mean.tsv pattern
    
    Args:
        substrate_id (str): Substrate ID (e.g., "PET")
        temstapro_output_dir (str): Path to the TemStaPro output directory
        
    Returns:
        Path or None: Path to the result file if found
    """
    base_path = Path(temstapro_output_dir)
    
    # Define possible result file patterns - UPDATED to include predictions_mean
    possible_result_files = [
        # Standard patterns in substrate subdirectory
        base_path / substrate_id / "output.tsv",
        base_path / substrate_id / "predictions.tsv",
        base_path / substrate_id / "results.tsv",
        base_path / substrate_id / f"{substrate_id}_predictions_mean.tsv",  # NEW
        base_path / substrate_id / "predictions_mean.tsv",  # NEW
        
        # Patterns with substrate prefix
        base_path / f"{substrate_id}_output.tsv",
        base_path / f"{substrate_id}_predictions.tsv",
        base_path / f"{substrate_id}_results.tsv",
        base_path / f"{substrate_id}_predictions_mean.tsv",  # NEW
        
        # CSV versions
        base_path / substrate_id / "output.csv",
        base_path / substrate_id / "predictions.csv",
        base_path / substrate_id / "results.csv",
        base_path / substrate_id / "predictions_mean.csv",  # NEW
        
        # Direct in substrate directory
        base_path / substrate_id / f"{substrate_id}_predictions_mean.tsv",  # NEW
    ]
    
    # Also search for any file matching the pattern *_predictions_mean.tsv
    # This handles cases like "sequences_predictions_mean.tsv"
    for pattern in ["*_predictions_mean.tsv", "*_predictions_mean.csv"]:
        # Search in substrate subdirectory
        matches = list((base_path / substrate_id).glob(pattern))
        possible_result_files.extend(matches)
        
        # Search in base directory
        matches = list(base_path.glob(pattern))
        possible_result_files.extend(matches)
    
    # Find the actual results file
    for file_path in possible_result_files:
        if file_path.exists():
            logger.info(f"Found result file: {file_path}")
            return file_path
    
    return None

def process_substrate_results(substrate_id, temstapro_output_dir, original_fasta_path, method):
    """
    Process TemStaPro results for a specific substrate.
    
    Args:
        substrate_id (str): Substrate ID (e.g., "PET")
        temstapro_output_dir (str): Path to the TemStaPro output directory
        original_fasta_path (str): Path to the original FASTA file
        method (str): Thermostability scoring method
        
    Returns:
        pd.DataFrame: Combined results dataframe
        bool: Whether processing was successful
    """
    # Find the result file
    result_file = find_temstapro_result_file(substrate_id, temstapro_output_dir)
    
    if result_file is None:
        logger.warning(f"No results file found for substrate {substrate_id}")
        logger.warning(f"Searched in directory: {temstapro_output_dir}")
        return None, False
    
    if not Path(original_fasta_path).exists():
        logger.warning(f"Original FASTA file not found: {original_fasta_path}")
        return None, False
    
    # Read prediction file
    try:
        # Determine separator based on file extension
        sep = '\t' if result_file.suffix == '.tsv' else ','
        result_df = pd.read_csv(result_file, sep=sep)
        logger.info(f"Loaded results from: {result_file}")
        logger.info(f"Data shape: {result_df.shape}")
        logger.info(f"Columns: {list(result_df.columns)}")
    except Exception as e:
        logger.error(f"Error reading results file for {substrate_id}: {e}")
        return None, False
    
    # Check if required columns exist
    required_cols = ["protein_id"]
    temperatures = [40, 45, 50, 55, 60, 65]
    
    # Handle cases where the first column might be unnamed (index column)
    if len(result_df.columns) > 0 and (result_df.columns[0] == "Unnamed: 0" or result_df.columns[0] == ""):
        result_df = result_df.drop(result_df.columns[0], axis=1)
    
    missing_cols = set(required_cols) - set(result_df.columns)
    if missing_cols:
        logger.warning(f"Missing required columns in results file for {substrate_id}: {missing_cols}")
        logger.warning(f"Available columns: {list(result_df.columns)}")
        return None, False
    
    # Create FASTA ID to sequence mapping from original FASTA
    try:
        id_to_seq = get_fasta_id_to_sequence_map(original_fasta_path)
        logger.info(f"Loaded {len(id_to_seq)} sequences from {original_fasta_path}")
    except Exception as e:
        logger.error(f"Error processing FASTA file: {e}")
        return None, False
    
    # Process thermostability scores
    thermostability_scores = []
    valid_proteins = []
    
    for _, row in result_df.iterrows():
        protein_id = row['protein_id']
        
        try:
            if method == "tm_interpolation":
                # Check if raw score columns exist
                raw_cols = [f't{temp}_raw' for temp in temperatures]
                if all(col in result_df.columns for col in raw_cols):
                    raw_scores = [float(row[col]) for col in raw_cols]
                    score = calculate_tm_interpolation(raw_scores, temperatures)
                else:
                    logger.warning(f"Missing raw score columns for tm_interpolation method")
                    score = np.nan
                    
            elif method == "weighted_temp":
                # Check if raw score columns exist
                raw_cols = [f't{temp}_raw' for temp in temperatures]
                if all(col in result_df.columns for col in raw_cols):
                    raw_scores = [float(row[col]) for col in raw_cols]
                    score = calculate_weighted_temp(raw_scores, temperatures)
                else:
                    logger.warning(f"Missing raw score columns for weighted_temp method")
                    score = np.nan
                    
            elif method == "range_midpoint":
                if 'left_hand_label' in result_df.columns:
                    range_label = row['left_hand_label']
                    score = parse_range_midpoint(range_label)
                else:
                    logger.warning(f"Missing left_hand_label column for range_midpoint method")
                    score = np.nan
            
            if not np.isnan(score):
                thermostability_scores.append(score)
                valid_proteins.append(protein_id)
            else:
                logger.warning(f"Could not calculate score for protein {protein_id}")
                
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping protein '{protein_id}' due to missing/invalid data: {e}")
            continue
    
    if not valid_proteins:
        logger.error(f"No valid thermostability scores calculated for substrate {substrate_id}")
        return None, False
    
    # Create final dataframe
    final_df = pd.DataFrame({
        'sequence_id': valid_proteins,
        'thermostability_score': thermostability_scores
    })
    
    # Add full sequence column
    final_df["sequence"] = final_df["sequence_id"].map(id_to_seq)
    
    # Check if all sequences were mapped
    unmapped = final_df[final_df["sequence"].isna()]
    if not unmapped.empty:
        logger.warning(f"Warning: {len(unmapped)} sequences could not be mapped")
        logger.warning(f"Unmapped sequence IDs: {unmapped['sequence_id'].tolist()}")
    
    # Remove rows where sequence mapping failed
    final_df = final_df.dropna(subset=["sequence"])
    
    # Rearrange columns
    final_df = final_df[["sequence_id", "sequence", "thermostability_score"]]
    
    logger.info(f"Processed {len(final_df)} entries for substrate {substrate_id}")
    if len(final_df) > 0:
        logger.info(f"Thermostability score range: {final_df['thermostability_score'].min():.1f}°C to {final_df['thermostability_score'].max():.1f}°C")
    
    return final_df, True

def main():
    """
    Main function to process all protein groups and generate TSV files.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info(f"TemStaPro output directory: {args.input_dir}")
    logger.info(f"FASTA directory: {args.faa_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Thermostability method: {args.method}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the mapping file to understand the directory structure
    possible_locations = []
    
    # First, try the explicitly provided temstapro_input_dir
    if args.temstapro_input_dir:
        possible_locations.append(Path(args.temstapro_input_dir))
    
    # Then try other common locations
    possible_locations.extend([
        Path(args.input_dir).parent.parent / "data" / "temstapro_data",  # Standard location
        Path(args.input_dir).parent / "temstapro_data",  # Alternative location
        Path(args.input_dir).parent / "temstapro",  # Another alternative
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
        
        # First, try to find substrate directories
        substrate_dirs = [d for d in output_path.iterdir() if d.is_dir()]
        
        if substrate_dirs:
            # Create a synthetic mapping based on directories
            for substrate_dir in substrate_dirs:
                substrate_id = substrate_dir.name
                # Find all fasta files in the faa_dir
                fasta_files = list(Path(args.faa_dir).glob("*.faa"))
                for fasta_file in fasta_files:
                    if substrate_id not in mapping:
                        mapping[substrate_id] = []
                    mapping[substrate_id].append(("dummy_input", str(fasta_file)))
            logger.info(f"Inferred mapping for substrates: {list(mapping.keys())}")
        else:
            # No subdirectories found, look for prediction files directly
            prediction_files = list(output_path.glob("*_predictions_mean.tsv"))
            prediction_files.extend(list(output_path.glob("*_predictions_mean.csv")))
            
            if prediction_files:
                logger.info(f"Found prediction files directly in output directory: {[f.name for f in prediction_files]}")
                # Extract substrate ID from filename (before _predictions_mean)
                for pred_file in prediction_files:
                    # Extract substrate ID from filename like "sequences_predictions_mean.tsv"
                    substrate_id = pred_file.stem.replace("_predictions_mean", "")
                    logger.info(f"Extracted substrate ID: {substrate_id} from file: {pred_file.name}")
                    
                    # Find matching FASTA files
                    fasta_files = list(Path(args.faa_dir).glob("*.faa"))
                    # Try to match by name
                    matching_fasta = None
                    for fasta_file in fasta_files:
                        if substrate_id in fasta_file.stem or fasta_file.stem in substrate_id:
                            matching_fasta = fasta_file
                            break
                    
                    if matching_fasta is None and fasta_files:
                        matching_fasta = fasta_files[0]  # Use first available
                        logger.warning(f"No matching FASTA found for {substrate_id}, using {matching_fasta}")
                    
                    if matching_fasta:
                        if substrate_id not in mapping:
                            mapping[substrate_id] = []
                        mapping[substrate_id].append(("dummy_input", str(matching_fasta)))
                
                logger.info(f"Created mapping from prediction files: {mapping}")
    
    if not mapping:
        logger.error("No processed files mapping found and could not infer structure. Cannot determine protein groups.")
        logger.error("Please check that:")
        logger.error("1. Your input directory contains TemStaPro output files")
        logger.error("2. Files are named with pattern *_predictions_mean.tsv or similar")
        logger.error("3. Your FASTA directory contains .faa files")
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
                original_fasta,
                args.method
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