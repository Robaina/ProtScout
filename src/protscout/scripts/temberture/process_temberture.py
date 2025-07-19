#!/usr/bin/env python3
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

def process_temberture_results(input_dir, output_dir):
    """
    Process TemBERTure results from multiple protein groups and consolidate them.
    
    Args:
        input_dir: Path to the directory containing TemBERTure results
        output_dir: Path to output directory for consolidated results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subdirectories (protein groups)
    protein_groups = [d for d in os.listdir(input_dir) 
                     if os.path.isdir(os.path.join(input_dir, d))]
    
    logger.info(f"Found {len(protein_groups)} protein groups: {', '.join(protein_groups)}")
    
    success_count = 0
    for group in protein_groups:
        group_dir = os.path.join(input_dir, group)
        results_file = os.path.join(group_dir, f"{group}_results.tsv")
        
        if not os.path.exists(results_file):
            logger.warning(f"Warning: Results file not found for {group}. Skipping.")
            continue
        
        logger.info(f"Processing {group}...")
        
        # Read the TSV file
        try:
            df = pd.read_csv(results_file, sep='\t')
            
            # Calculate mean TM across tm1, tm2, tm3
            tm_columns = [col for col in df.columns if col.startswith('tm')]
            
            if not tm_columns:
                logger.warning(f"Warning: No TM columns found in {results_file}. Skipping.")
                continue
                
            # Calculate mean TM
            df['TM'] = df[tm_columns].mean(axis=1)
            
            # Select only sequence_id and TM
            result_df = df[['sequence_id', 'TM']]
            
            # Save consolidated results
            output_file = os.path.join(output_dir, f"{group}.tsv")
            result_df.to_csv(output_file, sep='\t', index=False)
            
            logger.info(f"Saved consolidated results for {group} with {len(result_df)} sequences")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {group}: {str(e)}")
    
    logger.info(f"Processing complete. Successfully processed {success_count}/{len(protein_groups)} groups.")
    logger.info(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Consolidate TemBERTure results across different protein groups'
    )
    
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        required=True,
        help='Path to the directory containing TemBERTure results'
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Path to output directory for consolidated results'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        parser.error(f"Input directory does not exist: {args.input_dir}")
    
    process_temberture_results(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()