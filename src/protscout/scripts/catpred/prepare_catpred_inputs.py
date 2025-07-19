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
            substrates[row['substrate_id']] = {
                'substrate': row['substrate'],
                'smiles': row['smiles']
            }
    return substrates

def create_catpred_input(fasta_file, substrate_name, smiles, output_dir, substrate_id):
    """Create CatPred input CSV file from FASTA sequences."""
    # Create output directory using substrate_id
    output_path = os.path.join(output_dir, substrate_id)
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, 'input.csv')
    
    # Read sequences from FASTA file
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    
    # Check if file already exists (for appending when using same substrate_id)
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
    parser.add_argument('--substrate_id', required=True, help='Substrate ID to use for processing')
    
    args = parser.parse_args()
    
    # Read substrate information
    substrates = read_substrate_info(args.substrate_tsv)
    
    # Check if substrate_id exists in substrates
    if args.substrate_id not in substrates:
        print(f"Error: Substrate ID '{args.substrate_id}' not found in substrate information")
        print(f"Available substrate IDs: {', '.join(substrates.keys())}")
        return
    
    # Get substrate information for the provided substrate_id
    substrate_info = substrates[args.substrate_id]
    
    # Process all FASTA files in the directory
    processed_files = []
    
    for fasta_file in os.listdir(args.fasta_dir):
        if fasta_file.endswith('.faa') or fasta_file.endswith('.fasta'):
            print(f"Processing {fasta_file} with substrate_id: {args.substrate_id}")
            
            # Create CatPred input for this FASTA file
            input_file = create_catpred_input(
                os.path.join(args.fasta_dir, fasta_file),
                substrate_info['substrate'],
                substrate_info['smiles'],
                args.output_dir,
                args.substrate_id
            )
            
            # Store the original FASTA file path
            original_fasta_path = os.path.join(args.fasta_dir, fasta_file)
            processed_files.append((args.substrate_id, input_file, original_fasta_path))
            
            print(f"Created input file for {fasta_file} using substrate: {substrate_info['substrate']}")
    
    # Write the list of processed files with mapping information
    if processed_files:
        with open(os.path.join(args.output_dir, 'processed_files.txt'), 'w') as f:
            f.write("result_directory\tcatpred_input_file\toriginal_fasta_file\n")
            for result_dir, input_file, original_fasta in processed_files:
                f.write(f"{result_dir}\t{input_file}\t{original_fasta}\n")
        
        print(f"\nProcessed {len(processed_files)} FASTA files")
        print(f"All files were placed in directory: {os.path.join(args.output_dir, args.substrate_id)}")
    else:
        print("No FASTA files found to process")

if __name__ == '__main__':
    main()