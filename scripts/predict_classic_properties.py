#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def setup_logging(log_file: Optional[str] = None) -> None:
    """Configure logging to file if log_file provided, otherwise to console."""
    handlers = []
    if log_file:
        handlers = [logging.FileHandler(log_file)]
    else:
        handlers = [logging.StreamHandler()]
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )


def compute_basic_protein_properties(
    fasta_path: str, output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze properties of proteins from a FASTA file and save results to TSV.
    Args:
        fasta_path: Path to input FASTA file
        output_path: Optional path to save TSV output (defaults to input path with .tsv extension)
    Returns:
        DataFrame containing protein properties
    Properties calculated:
        - molecular_weight_kda: Mass of protein in kilodaltons, calculated from amino acid composition
        - isoelectric_point: pH at which protein has neutral net charge (affects solubility)
        - extinction_coefficient:
            - reduced: Protein absorption at 280nm without disulfide bonds
            - oxidized: Protein absorption at 280nm with disulfide bonds
            Used for protein concentration measurements in spectrophotometry
        - instability_index:
            Value predicting protein stability in vitro
            < 40: Probably stable
            > 40: Probably unstable
        - flexibility: Average protein chain flexibility based on amino acid composition
            Higher values indicate more flexible regions
            Used in structural predictions
        - n_complete: 1 if sequence starts with Methionine (M), 0 otherwise
        - c_complete: 1 if sequence ends with stop codon (*), 0 otherwise
    """
    results: List[Dict] = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        # Get the raw sequence with potential stop codon
        raw_sequence = str(record.seq)
        # Get sequence for ProteinAnalysis (without stop codon)
        sequence = raw_sequence.replace("*", "")
        
        try:
            analysis = ProteinAnalysis(sequence)
            # Calculate properties
            properties = {
                "sequence_id": record.id,
                "sequence_length": len(sequence),
                "molecular_weight_kda": analysis.molecular_weight() / 1000,
                "isoelectric_point": analysis.isoelectric_point(),
                "extinction_coefficient_reduced": analysis.molar_extinction_coefficient()[0],
                "extinction_coefficient_oxidized": analysis.molar_extinction_coefficient()[1],
                "instability_index": analysis.instability_index(),
                "avg_flexibility": sum(analysis.flexibility()) / len(analysis.flexibility()),
                # Add new termini completeness properties
                "n_complete": int(sequence.startswith('M')),
                "c_complete": int(raw_sequence.endswith('*'))
            }
            results.append(properties)
        except Exception as e:
            logging.error(f"Error processing {record.id}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to TSV if path provided
    if output_path is None:
        output_path = str(Path(fasta_path).with_suffix(".tsv"))
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    return df


def process_directory(input_dir: Path, output_dir: Path, extension: str = ".faa") -> None:
    """Process all FASTA files in the input directory.
    
    Args:
        input_dir: Path to input directory
        output_dir: Path to output directory
        extension: File extension to look for (default: ".faa")
    """
    # Remove the dot if user included it, then add it back to ensure consistent format
    extension = "." + extension.lstrip(".")
    fasta_files = list(input_dir.glob(f'*{extension}'))
    
    if not fasta_files:
        logging.warning(f"No files with extension {extension} found in {input_dir}")
        return
    
    for fasta_path in fasta_files:
        output_path = output_dir / f"{fasta_path.stem}.tsv"
        try:
            compute_basic_protein_properties(
                fasta_path=str(fasta_path),
                output_path=str(output_path)
            )
            logging.info(f"Processed {fasta_path.name} -> {output_path.name}")
        except Exception as e:
            logging.error(f"Failed to process {fasta_path.name}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Process protein sequences from FASTA files and compute their properties."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Input directory containing FASTA files or path to a single FASTA file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=str,
        help="Output directory for TSV files or path to output TSV file"
    )
    parser.add_argument(
        "-l", "--log",
        type=str,
        help="Path to log file (if not provided, logs to terminal)"
    )
    parser.add_argument(
        "-e", "--extension",
        type=str,
        default="faa",
        help="File extension to look for (default: faa)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Process either a single file or a directory
    try:
        if input_path.is_file():
            logging.info(f"Processing single file: {input_path}")
            compute_basic_protein_properties(
                fasta_path=str(input_path),
                output_path=str(output_path)
            )
        elif input_path.is_dir():
            logging.info(f"Processing directory: {input_path}")
            output_path.mkdir(parents=True, exist_ok=True)
            process_directory(input_path, output_path, args.extension)
        else:
            logging.error(f"Input path does not exist: {input_path}")
            return 1
        
        logging.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())