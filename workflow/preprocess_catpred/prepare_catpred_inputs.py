# prepare_inputs.py
import argparse
import csv
import os
from Bio import SeqIO


def read_substrate_info(tsv_file):
    """Read substrate information from TSV file."""
    substrates = {}
    with open(tsv_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            substrates[row["plastic_id"]] = {
                "substrate": row["substrate"],
                "smiles": row["smiles"],
            }
    return substrates


def create_catpred_input(fasta_file, substrate_name, smiles, output_dir):
    """Create CatPred input CSV file from FASTA sequences."""
    # Create output directory if it doesn't exist
    plastic_id = os.path.splitext(os.path.basename(fasta_file))[0]
    output_path = os.path.join(output_dir, plastic_id)
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, "input.csv")

    # Read sequences from FASTA file
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))

    # Write CatPred input CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Substrate", "SMILES", "sequence", "pdbpath"])
        for i, seq in enumerate(sequences, 1):
            writer.writerow([substrate_name, smiles, seq, f"seq{i}.pdb"])

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Prepare input files for CatPred")
    parser.add_argument(
        "--fasta_dir", required=True, help="Directory containing FASTA files"
    )
    parser.add_argument(
        "--substrate_tsv", required=True, help="TSV file with substrate information"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for CatPred input files"
    )

    args = parser.parse_args()

    # Read substrate information
    substrates = read_substrate_info(args.substrate_tsv)

    # Process each FASTA file
    processed_files = []
    for fasta_file in os.listdir(args.fasta_dir):
        if fasta_file.endswith(".faa"):
            plastic_id = os.path.splitext(fasta_file)[0]

            # Skip if no substrate information available
            if plastic_id not in substrates:
                print(f"Skipping {plastic_id}: No substrate information available")
                continue

            substrate_info = substrates[plastic_id]
            input_file = create_catpred_input(
                os.path.join(args.fasta_dir, fasta_file),
                substrate_info["substrate"],
                substrate_info["smiles"],
                args.output_dir,
            )
            processed_files.append((plastic_id, input_file))
            print(f"Created input file for {plastic_id}")

    # Write the list of processed files
    with open(os.path.join(args.output_dir, "processed_files.txt"), "w") as f:
        for plastic_id, input_file in processed_files:
            f.write(f"{plastic_id}\t{input_file}\n")


if __name__ == "__main__":
    main()
