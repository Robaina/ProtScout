import os
import argparse
from Bio import SeqIO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter FASTA files to keep only sequences with corresponding PDB files."
    )
    parser.add_argument(
        "--faa_dir", required=True, help="Directory containing .faa files"
    )
    parser.add_argument(
        "--pdb_dir", required=True, help="Directory containing PDB subdirectories"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Process each .faa file
    for faa_file in os.listdir(args.faa_dir):
        if not faa_file.endswith(".faa"):
            continue

        # Get the corresponding PDB subdirectory name (removing .faa extension)
        subdir_name = os.path.splitext(faa_file)[0]
        pdb_subdir = os.path.join(args.pdb_dir, subdir_name)

        # Get list of PDB files (without .pdb extension)
        pdb_files = {
            os.path.splitext(f)[0] for f in os.listdir(pdb_subdir) if f.endswith(".pdb")
        }

        # Read and filter sequences
        input_path = os.path.join(args.faa_dir, faa_file)
        sequences = []

        for record in SeqIO.parse(input_path, "fasta"):
            # Remove '>' if present in the sequence ID
            seq_id = record.id.lstrip(">")

            if seq_id in pdb_files:
                sequences.append(record)

        # Write filtered sequences to the same file
        SeqIO.write(sequences, input_path, "fasta")
        print(f"Processed {faa_file}: kept {len(sequences)} sequences")


if __name__ == "__main__":
    main()
