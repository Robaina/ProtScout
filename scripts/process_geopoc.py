#!/usr/bin/env python3
import os
import pandas as pd
import argparse
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def consolidate_geopoc_results(input_dir, output_dir):
    """
    Consolidate GeoPoc results from multiple CSV files into a single TSV per protein group.
    
    Args:
        input_dir (str): Base directory containing the GeoPoc output folders
        output_dir (str): Directory where consolidated TSV files will be saved
    """
    # Validate input directory
    if not os.path.exists(input_dir):
        logger.error(f"Error: Input directory '{input_dir}' does not exist")
        return False
        
    if not os.path.isdir(input_dir):
        logger.error(f"Error: '{input_dir}' is not a directory")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all items in the base directory
    all_items = os.listdir(input_dir)
    
    # Filter for directories (protein groups)
    protein_groups = []
    for item in all_items:
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            protein_groups.append(item)
    
    if not protein_groups:
        logger.error(f"Error: No protein group directories found in '{input_dir}'")
        return False
    
    logger.info(f"Found {len(protein_groups)} protein groups: {', '.join(protein_groups)}")
    success_count = 0
    
    for group in protein_groups:
        group_dir = os.path.join(input_dir, group)
        logger.info(f"Processing {group}...")
        
        # Paths to the three CSV files
        ph_file = os.path.join(group_dir, "pH_preds.csv")
        salt_file = os.path.join(group_dir, "salt_preds.csv")
        temp_file = os.path.join(group_dir, "temp_preds.csv")
        
        # Check if all required files exist
        missing_files = []
        for file_path, file_name in [
            (ph_file, "pH_preds.csv"),
            (salt_file, "salt_preds.csv"),
            (temp_file, "temp_preds.csv")
        ]:
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            logger.warning(f"Warning: Missing files for {group}: {', '.join(missing_files)}")
            logger.warning(f"Available files: {os.listdir(group_dir)}")
            continue
        
        try:
            # Read each CSV file
            ph_data = pd.read_csv(ph_file)
            salt_data = pd.read_csv(salt_file)
            temp_data = pd.read_csv(temp_file)
            
            # Validate required columns
            if "SeqID" not in ph_data.columns or "class_pH" not in ph_data.columns:
                logger.error(f"Error: pH_preds.csv missing required columns. Available columns: {ph_data.columns.tolist()}")
                continue
                
            if "SeqID" not in salt_data.columns or "class_salt" not in salt_data.columns:
                logger.error(f"Error: salt_preds.csv missing required columns. Available columns: {salt_data.columns.tolist()}")
                continue
                
            if "SeqID" not in temp_data.columns or "temperature" not in temp_data.columns:
                logger.error(f"Error: temp_preds.csv missing required columns. Available columns: {temp_data.columns.tolist()}")
                continue
            
            # Extract required columns
            ph_subset = ph_data[["SeqID", "class_pH"]].rename(columns={"class_pH": "pH"})
            salt_subset = salt_data[["SeqID", "class_salt"]].rename(columns={"class_salt": "salt"})
            temp_subset = temp_data[["SeqID", "temperature"]]
            
            # Merge dataframes on SeqID
            merged_df = pd.merge(ph_subset, salt_subset, on="SeqID", how="outer")
            merged_df = pd.merge(merged_df, temp_subset, on="SeqID", how="outer")
            
            # Rename columns to match required output format
            merged_df.rename(columns={
                "SeqID": "sequence_id"
            }, inplace=True)
            
            # Sort by sequence_id for consistency
            merged_df.sort_values(by="sequence_id", inplace=True)
            
            # Write to TSV
            output_file = os.path.join(output_dir, f"{group}.tsv")
            merged_df.to_csv(output_file, sep='\t', index=False)
            
            logger.info(f"Successfully created {output_file} with {len(merged_df)} entries")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {group}: {str(e)}")
    
    if success_count > 0:
        logger.info(f"Summary: Successfully processed {success_count} out of {len(protein_groups)} protein groups")
        logger.info(f"Results saved to: {output_dir}")
        return True
    else:
        logger.error("Error: No protein groups were successfully processed")
        return False

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Consolidate GeoPoc results into TSV files per protein group')
    parser.add_argument('-i', '--input_dir', 
                        help='Base directory containing the GeoPoc output folders')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Output directory where consolidated TSV files will be saved')
    
    args = parser.parse_args()
    
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    success = consolidate_geopoc_results(args.input_dir, args.output_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()