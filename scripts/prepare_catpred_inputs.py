#!/usr/bin/env python3

import argparse
import csv
import os
from Bio import SeqIO

def read_substrate_info(tsv_file):
    """Read substrate information from TSV file."""
    substrates = {}
    with open(tsv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            substrates[row['plastic_id']] = {
                'substrate': row['substrate'],
                'smiles': row['smiles']
            }
    return substrates

def create_catpred_input(fasta_file, substrate_name, smiles, output_dir, dir_name=None):
    """Create CatPred input CSV file from FASTA sequences."""
    # Create output directory if it doesn't exist
    if dir_name:
        plastic_id = dir_name
    else:
        plastic_id = os.path.splitext(os.path.basename(fasta_file))[0]
    output_path = os.path.join(output_dir, plastic_id)
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, 'input.csv')
    
    # Read sequences from FASTA file
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    
    # Check if file already exists (for appending when using same plastic_id)
    file_exists = os.path.exists(output_file)
    seq_counter = 1
    
    if file_exists:
        # Read existing file to get the current sequence count
        with open(output_file, 'r') as f:
            lines = f.readlines()
            seq_counter = len(lines)  # Header + existing sequences
    
    # Write CatPred input CSV (append mode if file exists)
    mode = 'a' if file_exists else 'w'
    with open(output_file, mode, newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Substrate', 'SMILES', 'sequence', 'pdbpath'])
        for seq in sequences:
            writer.writerow([substrate_name, smiles, seq, f'seq{seq_counter}.pdb'])
            seq_counter += 1
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Prepare input files for CatPred')
    parser.add_argument('--fasta_dir', required=True, help='Directory containing FASTA files')
    parser.add_argument('--substrate_tsv', required=True, help='TSV file with substrate information')
    parser.add_argument('--output_dir', required=True, help='Output directory for CatPred input files')
    parser.add_argument('--plastic_id', default=None, help='Optional plastic ID to use for all input files')
    
    args = parser.parse_args()
    
    # Read substrate information
    substrates = read_substrate_info(args.substrate_tsv)
    
    # If plastic_id is provided, check if it exists in substrates
    if args.plastic_id and args.plastic_id not in substrates:
        print(f"Error: Plastic ID '{args.plastic_id}' not found in substrate information")
        print(f"Available plastic IDs: {', '.join(substrates.keys())}")
        return
    
    # Process each FASTA file
    processed_files = []
    processed_dirs = set()  # Track which directories we've already processed
    
    for fasta_file in os.listdir(args.fasta_dir):
        if fasta_file.endswith('.faa'):
            if args.plastic_id:
                # Use provided plastic_id for all files
                lookup_plastic_id = args.plastic_id
                actual_dir_name = args.plastic_id
                print(f"Processing {fasta_file} with provided plastic_id: {lookup_plastic_id}")
            else:
                # Extract plastic_id from filename as before
                lookup_plastic_id = os.path.splitext(fasta_file)[0]
                actual_dir_name = lookup_plastic_id
                print(f"Processing {fasta_file} with filename-derived plastic_id: {lookup_plastic_id}")
            
            # Skip if no substrate information available
            if lookup_plastic_id not in substrates:
                print(f"Skipping {fasta_file}: No substrate information available for {lookup_plastic_id}")
                continue
            
            substrate_info = substrates[lookup_plastic_id]
            
            # Use plastic_id as directory name if provided, otherwise use filename
            dir_name = args.plastic_id if args.plastic_id else None
            
            input_file = create_catpred_input(
                os.path.join(args.fasta_dir, fasta_file),
                substrate_info['substrate'],
                substrate_info['smiles'],
                args.output_dir,
                dir_name
            )
            
            # Store mapping: result_directory_name -> original_fasta_file
            # Only add to processed_files if we haven't seen this directory before
            if actual_dir_name not in processed_dirs:
                original_fasta_path = os.path.join(args.fasta_dir, fasta_file)
                processed_files.append((actual_dir_name, input_file, original_fasta_path))
                processed_dirs.add(actual_dir_name)
                
            print(f"Created input file for {fasta_file} using substrate: {substrate_info['substrate']}")
    
    # Write the list of processed files with enhanced mapping information
    with open(os.path.join(args.output_dir, 'processed_files.txt'), 'w') as f:
        f.write("result_directory\tcatpred_input_file\toriginal_fasta_file\n")
        for result_dir, input_file, original_fasta in processed_files:
            f.write(f"{result_dir}\t{input_file}\t{original_fasta}\n")

if __name__ == '__main__':
    main()