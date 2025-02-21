#!/usr/bin/env python
"""
Enzyme Property Prediction Script

This script predicts various enzymatic properties for a set of protein sequences
using multiple predictors. Users can select which predictors to use.

# Use all predictors
python predict_enzyme_properties.py --substrate BHET --workdir /path/to/workdir

# Use only kcat and thermostability predictors
python predict_enzyme_properties.py --substrate glucose --workdir /path/to/workdir --predictors kcat thermostability

# Specify custom SMILES file and output location
python predict_enzyme_properties.py --substrate custom_substrate --workdir /path/to/workdir \
    --smiles-file substrates.csv --output-file results/predictions.csv

# Use a specific FASTA file and PDB directory
python predict_enzyme_properties.py --substrate BHET --workdir /path/to/workdir \
    --fasta-file sequences.fasta --pdb-dir pdb_files/ --batch-size 8 --device cpu
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
import torch
from Bio import SeqIO
import pandas as pd

from proteus.fitness_predictors import (
    KineticPredictor,
    ThermostabilityPredictor,
    EpHodPredictor,
    SolubilityGATSolPredictor,
    Predictor,
)
from protscout.predictors import GeoPocPredictor


def load_sequences(fasta_file: str) -> Dict[str, str]:
    """Load sequences from a FASTA file."""
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def get_pdb_path(pdb_dir: str, seq_id: str) -> str:
    """Get the PDB file path for a given sequence ID."""
    return os.path.join(pdb_dir, f"{seq_id}.pdb")


def setup_predictors(
    substrate_name: str,
    substrate_smiles_dict: Dict[str, str],
    workdir: str,
    selected_predictors: Set[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    model_weights_dir: str = "/home/ec2-user/SageMaker/models",
) -> Dict[str, Predictor]:
    """
    Initialize selected predictors for a specific substrate.

    Args:
        substrate_name: The name of the substrate being analyzed
        substrate_smiles_dict: Dictionary mapping substrate names to their SMILES strings
        workdir: Working directory for temporary files and outputs
        selected_predictors: Set of predictor names to initialize
        device: Computation device to use
        verbose: Whether to print verbose output
        model_weights_dir: Directory containing model weights

    Returns:
        Dictionary of predictors with meaningful keys
    """
    if substrate_name not in substrate_smiles_dict:
        raise ValueError(f"No SMILES string provided for substrate: {substrate_name}")

    # Create required output directories
    os.makedirs(f"{workdir}/outputs/temp_dir_catpred", exist_ok=True)
    os.makedirs(f"{workdir}/outputs/temp_dir_temberture", exist_ok=True)
    os.makedirs(f"{workdir}/outputs/temp_dir_ephod", exist_ok=True)
    os.makedirs(f"{workdir}/outputs/temp_dir_gatsol", exist_ok=True)
    os.makedirs(f"{workdir}/outputs/temp_dir_geopoc", exist_ok=True)

    predictors = {}

    # Initialize only the selected predictors
    if "kcat" in selected_predictors:
        predictors["kcat"] = KineticPredictor(
            substrate_smiles=substrate_smiles_dict[substrate_name],
            substrate_name=substrate_name,
            parameter="kcat",
            docker_image="ghcr.io/new-atlantis-labs/catpred:latest",
            weights_dir=model_weights_dir,
            parent_temp_dir=f"{workdir}/outputs/temp_dir_catpred",
            verbose=verbose,
            device=device,
        )

    if "km" in selected_predictors:
        predictors["km"] = KineticPredictor(
            substrate_smiles=substrate_smiles_dict[substrate_name],
            substrate_name=substrate_name,
            parameter="km",
            docker_image="ghcr.io/new-atlantis-labs/catpred:latest",
            weights_dir=model_weights_dir,
            parent_temp_dir=f"{workdir}/outputs/temp_dir_catpred",
            verbose=verbose,
            device=device,
        )

    if "ki" in selected_predictors:
        predictors["ki"] = KineticPredictor(
            substrate_smiles=substrate_smiles_dict[substrate_name],
            substrate_name=substrate_name,
            parameter="ki",
            docker_image="ghcr.io/new-atlantis-labs/catpred:latest",
            weights_dir=model_weights_dir,
            parent_temp_dir=f"{workdir}/outputs/temp_dir_catpred",
            verbose=verbose,
            device=device,
        )

    if "thermostability" in selected_predictors:
        predictors["thermostability"] = ThermostabilityPredictor(
            device=device,
            docker_image="ghcr.io/new-atlantis-labs/temberture:latest",
            parent_temp_dir=f"{workdir}/outputs/temp_dir_temberture",
            verbose=verbose,
        )

    if "ph_optimum" in selected_predictors:
        predictors["ph_optimum"] = EpHodPredictor(
            device=device,
            docker_image="ghcr.io/new-atlantis-labs/ephod:latest",
            weights_dir=model_weights_dir,
            parent_temp_dir=f"{workdir}/outputs/temp_dir_ephod",
            verbose=verbose,
        )

    if "solubility" in selected_predictors:
        predictors["solubility"] = SolubilityGATSolPredictor(
            device=device,
            docker_image="ghcr.io/new-atlantis-labs/gatsol:latest",
            verbose=verbose,
            requires_pdbs=True,
            esm_weights_dir=model_weights_dir,
            gatsol_weights_dir=f"{model_weights_dir}/gatsol",
            parent_temp_dir=f"{workdir}/outputs/temp_dir_gatsol",
            save_directory=f"{workdir}/outputs/temp_dir_gatsol",
        )

    if "geopoc_temp" in selected_predictors:
        predictors["geopoc_temp"] = GeoPocPredictor(
            task="temp",
            device=device,
            verbose=verbose,
            requires_pdbs=True,
            save_directory=f"{workdir}/outputs/temp_dir_geopoc",
            parent_temp_dir=f"{workdir}/outputs/temp_dir_geopoc",
            docker_image="ghcr.io/new-atlantis-labs/geopoc:latest",
            model_weights_dir=model_weights_dir,
        )

    if "geopoc_ph" in selected_predictors:
        predictors["geopoc_ph"] = GeoPocPredictor(
            task="pH",
            device=device,
            verbose=verbose,
            requires_pdbs=True,
            save_directory=f"{workdir}/outputs/temp_dir_geopoc",
            parent_temp_dir=f"{workdir}/outputs/temp_dir_geopoc",
            docker_image="ghcr.io/new-atlantis-labs/geopoc:latest",
            model_weights_dir=model_weights_dir,
        )

    if "geopoc_salt" in selected_predictors:
        predictors["geopoc_salt"] = GeoPocPredictor(
            task="salt",
            device=device,
            verbose=verbose,
            requires_pdbs=True,
            save_directory=f"{workdir}/outputs/temp_dir_geopoc",
            parent_temp_dir=f"{workdir}/outputs/temp_dir_geopoc",
            docker_image="ghcr.io/new-atlantis-labs/geopoc:latest",
            model_weights_dir=model_weights_dir,
        )

    return predictors


def process_substrate(
    substrate_name: str,
    workdir: str,
    predictors: Dict[str, Predictor],
    batch_size: int = 32,
    fasta_path: str = None,
    pdb_dir_path: str = None,
    output_file: str = None,
) -> pd.DataFrame:
    """
    Process sequences for a specific substrate.

    Args:
        substrate_name: Name of the substrate to process
        workdir: Working directory containing input files and outputs
        predictors: Dictionary of predictors initialized for this substrate
        batch_size: Number of sequences to process in each batch
        fasta_path: Optional custom path to the FASTA file (if None, uses default path)
        pdb_dir_path: Optional custom path to the PDB directory (if None, uses default path)
        output_file: Optional path to save the output CSV file

    Returns:
        DataFrame with prediction results
    """

    # Setup paths
    pdb_dir = pdb_dir_path or f"{workdir}/pdbs/{substrate_name}/pdbs"
    fasta_file = (
        fasta_path
        or f"{workdir}/outputs/sorted_fastas_representatives/selected_sequences_{substrate_name}.fasta"
    )

    # Load sequences
    sequences = load_sequences(fasta_file)
    print(f"Loaded {len(sequences)} sequences for {substrate_name}")

    # Initialize results dictionary with sequence_id column
    results = {"sequence_id": []}

    # Initialize columns based on available predictors
    if "kcat" in predictors:
        results["kcat"] = []
    if "km" in predictors:
        results["km"] = []
    if "ki" in predictors:
        results["ki"] = []
    if "thermostability" in predictors:
        results["TM"] = []
    if "ph_optimum" in predictors:
        results["optimal_pH"] = []
    if "solubility" in predictors:
        results["solubility"] = []
    if "geopoc_temp" in predictors:
        results["geopoc_temperature"] = []
    if "geopoc_ph" in predictors:
        results["geopoc_pH"] = []
    if "geopoc_salt" in predictors:
        results["geopoc_salt"] = []

    # Process sequences in batches
    seq_ids = list(sequences.keys())
    for i in range(0, len(seq_ids), batch_size):
        batch_ids = seq_ids[i : i + batch_size]
        batch_seqs = [sequences[seq_id] for seq_id in batch_ids]
        batch_pdbs = [get_pdb_path(pdb_dir, seq_id) for seq_id in batch_ids]

        # Skip sequences without PDB files
        valid_indices = [i for i, pdb in enumerate(batch_pdbs) if os.path.exists(pdb)]
        if not valid_indices:
            print(
                f"No pdb files found for input sequence IDs for substrate: {substrate_name}"
            )
            continue

        valid_seqs = [batch_seqs[i].replace("*", "") for i in valid_indices]
        valid_pdbs = [batch_pdbs[i] for i in valid_indices]
        valid_ids = [batch_ids[i] for i in valid_indices]

        try:
            # Collect predictions from each predictor
            predictions = {}

            # Run only the selected predictors
            if "kcat" in predictors:
                print("Inferring kcat..")
                predictions["kcat"] = predictors["kcat"].infer_fitness(valid_seqs)

            if "km" in predictors:
                print("Inferring KM..")
                predictions["km"] = predictors["km"].infer_fitness(valid_seqs)

            if "ki" in predictors:
                print("Inferring Ki..")
                predictions["ki"] = predictors["ki"].infer_fitness(valid_seqs)

            if "thermostability" in predictors:
                print("Inferring TM..")
                predictions["TM"] = predictors["thermostability"].infer_fitness(
                    valid_seqs
                )

            if "ph_optimum" in predictors:
                print("Inferring optimum pH..")
                predictions["optimal_pH"] = predictors["ph_optimum"].infer_fitness(
                    valid_seqs
                )

            if "solubility" in predictors:
                print("Inferring solubility..")
                predictions["solubility"] = predictors["solubility"].infer_fitness(
                    sequences=valid_seqs, sequence_ids=valid_ids, pdb_files=valid_pdbs
                )

            if "geopoc_temp" in predictors:
                print("Inferring GeoPoc temperature..")
                predictions["geopoc_temperature"] = predictors[
                    "geopoc_temp"
                ].infer_fitness(
                    sequences=valid_seqs,
                    pdb_files=valid_pdbs,
                    generation_id=f"{substrate_name}_batch_{i//batch_size + 1}",
                )

            if "geopoc_ph" in predictors:
                print("Inferring GeoPoc pH..")
                predictions["geopoc_pH"] = predictors["geopoc_ph"].infer_fitness(
                    sequences=valid_seqs,
                    pdb_files=valid_pdbs,
                    generation_id=f"{substrate_name}_batch_{i//batch_size + 1}",
                )

            if "geopoc_salt" in predictors:
                print("Inferring GeoPoc salt tolerance..")
                predictions["geopoc_salt"] = predictors["geopoc_salt"].infer_fitness(
                    sequences=valid_seqs,
                    pdb_files=valid_pdbs,
                    generation_id=f"{substrate_name}_batch_{i//batch_size + 1}",
                )

            # Store results
            for j, seq_id in enumerate(valid_ids):
                results["sequence_id"].append(seq_id)

                # Add results for each predictor
                for pred_key, pred_results in predictions.items():
                    result_key = pred_key
                    if pred_key == "TM":
                        result_key = "TM"  # Keep TM as is for thermostability
                    elif pred_key == "optimal_pH":
                        result_key = "optimal_pH"  # Keep optimal_pH as is
                    elif pred_key == "geopoc_temperature":
                        result_key = "geopoc_temperature"
                    elif pred_key == "geopoc_pH":
                        result_key = "geopoc_pH"
                    elif pred_key == "geopoc_salt":
                        result_key = "geopoc_salt"

                    results[result_key].append(float(pred_results[j].item()))

        except Exception as e:
            print(f"Error processing batch for {substrate_name}: {e}")
            continue

        print(f"Processed batch {i//batch_size + 1} for {substrate_name}")

    # Create a DataFrame from results
    results_df = pd.DataFrame(results)

    # Save results if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    return results_df


def load_substrate_smiles(smiles_file: str) -> Dict[str, str]:
    """
    Load substrate SMILES from a CSV file.
    Expected format: substrate_name,smiles_string

    Args:
        smiles_file: Path to CSV file with substrate SMILES

    Returns:
        Dictionary mapping substrate names to SMILES strings
    """
    substrate_smiles_dict = {}
    try:
        df = pd.read_csv(smiles_file)
        for _, row in df.iterrows():
            substrate_smiles_dict[row["substrate_name"]] = row["smiles_string"]
    except Exception as e:
        print(f"Error loading SMILES file: {e}")
        print("Using example SMILES instead.")
        # Provide example substrates if file loading fails
        substrate_smiles_dict = {
            "BHET": "CC(=O)OCCOc1ccc(C(=O)OC)cc1",
            "heptane": "CCCCCCC",
            "ethylbenzene": "C1=CC=C(C=C1)CC",
            "glucose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
            "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        }
    return substrate_smiles_dict


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict enzymatic properties for a set of protein sequences."
    )

    # Required arguments
    parser.add_argument("--substrate", required=True, help="Name of the substrate")
    parser.add_argument(
        "--workdir", required=True, help="Working directory for input/output files"
    )

    # Optional file paths
    parser.add_argument(
        "--smiles-file",
        help="Path to CSV file with substrate SMILES (columns: substrate_name,smiles_string)",
    )
    parser.add_argument("--fasta-file", help="Path to FASTA file with sequences")
    parser.add_argument("--pdb-dir", help="Path to directory containing PDB files")
    parser.add_argument("--output-file", help="Path to save output CSV file")
    parser.add_argument(
        "--model-weights-dir",
        default="/home/ec2-user/SageMaker/models",
        help="Directory containing model weights",
    )

    # Computation options
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device to use",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for processing"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Predictor selection
    parser.add_argument(
        "--predictors",
        nargs="+",
        default="all",
        help="Specific predictors to use (default: all). Options: kcat, km, ki, thermostability, "
        "ph_optimum, solubility, geopoc_temp, geopoc_ph, geopoc_salt",
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Determine which predictors to use
    all_predictors = {
        "kcat",
        "km",
        "ki",
        "thermostability",
        "ph_optimum",
        "solubility",
        "geopoc_temp",
        "geopoc_ph",
        "geopoc_salt",
    }

    if args.predictors == "all" or "all" in args.predictors:
        selected_predictors = all_predictors
    else:
        selected_predictors = set(args.predictors) & all_predictors
        if not selected_predictors:
            print("No valid predictors selected. Using all predictors.")
            selected_predictors = all_predictors

    print(f"Using predictors: {', '.join(selected_predictors)}")

    # Load substrate SMILES dictionary
    if args.smiles_file:
        substrate_smiles_dict = load_substrate_smiles(args.smiles_file)
    else:
        # Use example SMILES if no file provided
        substrate_smiles_dict = {
            "BHET": "CC(=O)OCCOc1ccc(C(=O)OC)cc1",
            "heptane": "CCCCCCC",
            "ethylbenzene": "C1=CC=C(C=C1)CC",
            "glucose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
            "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        }

    # Check if SMILES is available for the selected substrate
    if args.substrate not in substrate_smiles_dict:
        print(
            f"No SMILES found for substrate '{args.substrate}'. Please add it to the SMILES file."
        )
        return

    # Setup predictors
    predictors = setup_predictors(
        substrate_name=args.substrate,
        substrate_smiles_dict=substrate_smiles_dict,
        workdir=args.workdir,
        selected_predictors=selected_predictors,
        device=args.device,
        verbose=args.verbose,
        model_weights_dir=args.model_weights_dir,
    )

    # Process sequences
    results_df = process_substrate(
        substrate_name=args.substrate,
        workdir=args.workdir,
        predictors=predictors,
        batch_size=args.batch_size,
        fasta_path=args.fasta_file,
        pdb_dir_path=args.pdb_dir,
        output_file=args.output_file,
    )

    # Print summary if no output file was specified
    if not args.output_file:
        print("\nResults Summary:")
        print(f"Total sequences processed: {len(results_df)}")
        print(results_df.head())


if __name__ == "__main__":
    main()
