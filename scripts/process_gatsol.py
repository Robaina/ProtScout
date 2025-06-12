#!/usr/bin/env python3
"""
Consolidate GATSol results from different protein groups into individual TSV files.
This script extracts the 'id' and 'Solubility_hat' columns from each protein group's
predictions.csv file and consolidates them into individual TSV files per group.
"""
import os
import argparse
import pandas as pd
import logging
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
        description="Consolidate GATSol results into TSV files per protein group"
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Input directory containing GATSol outputs"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Output directory where consolidated TSV files will be saved"
    )
    return parser.parse_args()

def consolidate_gatsol_results(input_dir, output_dir):
    """
    Consolidate GATSol results from different protein groups.
    
    Parameters:
    -----------
    input_dir : str
        Path to the directory containing GATSol outputs
    output_dir : str
        Path to the directory where consolidated TSV files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of protein group directories
    input_path = Path(input_dir)
    group_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not group_dirs:
        logger.warning(f"No protein group directories found in {input_dir}")
        return
    
    logger.info(f"Found {len(group_dirs)} protein group directories")
    
    # Process each protein group directory
    success_count = 0
    for group_dir in group_dirs:
        group_name = group_dir.name
        predictions_file = group_dir / "predictions.csv"
        
        if not predictions_file.exists():
            logger.warning(f"No predictions.csv found for {group_name}")
            continue
        
        try:
            # Read the predictions CSV file
            logger.info(f"Processing {group_name}")
            df = pd.read_csv(predictions_file)
            
            # Check if required columns exist
            if 'id' not in df.columns or 'Solubility_hat' not in df.columns:
                logger.warning(f"Required columns missing in {predictions_file}")
                continue
            
            # Extract and rename columns
            result_df = df[['id', 'Solubility_hat']].copy()
            result_df.columns = ['sequence_id', 'solubility']
            
            # Save to TSV
            output_file = Path(output_dir) / f"{group_name}.tsv"
            result_df.to_csv(output_file, sep='\t', index=False)
            logger.info(f"Created {output_file} with {len(result_df)} records")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {group_name}: {str(e)}")
    
    logger.info(f"Processing complete. Successfully processed {success_count}/{len(group_dirs)} protein groups")

def main():
    """Main function to run the script."""
    args = parse_arguments()
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    consolidate_gatsol_results(args.input_dir, args.output_dir)
    logger.info("Processing complete")

if __name__ == "__main__":
    main()