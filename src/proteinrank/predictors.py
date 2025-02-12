from __future__ import annotations

import os
from pathlib import Path
import subprocess
import uuid
from typing import List, Optional
import shutil

import numpy as np
import pandas as pd
import torch

from proteinrank.predictor_helpers import minimize_gnina_affinity


class Predictor:
    """
    A base class for all predictors with common attributes and methods.
    """

    def __init__(self, weight: float = 1, requires_pdbs: bool = False):
        """
        Initializes the Predictor with a weight.

        Args:
            weight (float): Weight for the values predicted by this predictor in the combined score.
            requires_pdbs (bool): Whether the predictor requires PDB files for the input sequences.
        """
        self.weight = weight
        self.requires_pdbs = requires_pdbs

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ):
        """
        Predicts fitness values for given sequences.

        Args:
            sequences (List[str]): A list of sequences to evaluate.
            pdb_files (List[str]): A list of paths to PDB files corresponding to the input sequences.
            generation_id (str): Identifier for the generation of sequences.

        Returns:
            torch.Tensor: Predicted fitness values.
        """
        raise NotImplementedError("Subclasses should implement this method")


class CombinedPredictor(Predictor):
    """
    A class to combine predictions from multiple expert predictor classes
    using a linear combination of their scores.
    """

    def __init__(self, predictors: List[object], weights: List[float] = None):
        """
        Initializes CombinedPredictors with a list of expert predictors and corresponding weights.

        Args:
            predictors (List[object]): List of expert predictor instances.
            weights (List[float]): List of weights for linearly combining the scores from the predictors.
        """
        if weights is None:
            weights = [predictor.weight for predictor in predictors]
        assert len(predictors) == len(
            weights
        ), "The number of predictors and weights must match"
        self.predictors = predictors
        self.weights = weights

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ) -> torch.Tensor:
        """
        Predicts fitness values using the combined scores from the expert predictors.

        Args:
            sequences (List[str]): A list of protein sequences to evaluate.
            pdb_files (List[str]): A list of paths to PDB files corresponding to the input sequences.
            generation_id (str): Identifier for the generation of sequences.

        Returns:
            torch.Tensor: A tensor of combined fitness scores for the input sequences.
        """
        combined_scores = None
        for predictor, weight in zip(self.predictors, self.weights):
            scores = predictor.infer_fitness(sequences, generation_id)
            weighted_scores = scores * weight
            if combined_scores is None:
                combined_scores = weighted_scores
            else:
                combined_scores += weighted_scores
        return combined_scores


class KMpredictor(Predictor):
    """
    A class to predict KM values for given protein sequences using a Docker-based tool.
    """

    def __init__(self, substrate: str, data_dir: str, weight: float = 1):
        """
        Initializes the KMpredictor with protein sequences and substrate.

        Args:
            substrate (str): SMILES or INCHI string for the substrate of the input enzymes.
            data_dir (str): Path to the directory containing necessary data and model files for the prediction.
            weight (float): Weight for the KM values predicted by this predictor in the combined score.
        """
        super().__init__()
        self.substrate = substrate
        self.data_dir = data_dir
        self.weight = weight

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ) -> List[float]:
        """
        Predicts KM values using the initialized protein sequences and substrate.

        Args:
            sequences (List[str]): A list of protein sequences for which KM values are to be predicted.
            pdb_files (List[str]): A list of paths to PDB files corresponding to the input sequences.
            generation_id (str): Identifier for the generation of sequences.

        Returns:
            List[float]: A list of predicted KM values corresponding to the input protein sequences.
        """
        # Create temporary files for input sequences and output KM values
        input_sequences_path = self.data_dir
        input_filename = f"input_{uuid.uuid4()}.txt"
        output_values_path = self.data_dir

        # Write the protein sequences to the temporary input file
        with open(os.path.join(input_sequences_path, input_filename), "w") as seq_file:
            for sequence in sequences:
                seq_file.write(sequence + "\n")

        # Construct the Docker run command
        docker_command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.data_dir}:/app/data",
            "-v",
            f"{input_sequences_path}:/app/input",
            "-v",
            f"{output_values_path}:/app/output",
            "ghcr.io/new-atlantis-labs/predict_km_values_cpu:latest",
            "--input_sequences",
            f"/app/input/{input_filename}",
            "--substrate",
            self.substrate,
            "--data_dir",
            "/app/data",
            "--out_embeddings",
            "/app/output/embeddings.pt",
            "--out_values",
            "/app/output/kmvalues.pt",
        ]

        # Run the Docker command
        subprocess.run(docker_command, check=True)

        # Read the KM values from the output file
        KM_values = torch.load(f"{output_values_path}/kmvalues.pt")

        # Remove the temporary input and output files
        os.remove(os.path.join(input_sequences_path, input_filename))
        os.remove(f"{output_values_path}/kmvalues.pt")

        return 1 / KM_values.unsqueeze(0).transpose(0, 1)


class DeepStabPtmPredictor(Predictor):
    """
    A class to predict melting temperature (Tm) for given protein sequences using a Docker-based tool.
    """

    def __init__(
        self,
        model_dir: str,
        parent_temp_dir: str = None,
        growth_temp: int = 37,
        mt_mode: str = "Lysate",
        weight: float = 1,
        docker_image: str = None,
    ):
        """
        Initializes the DeepStabPtmPredictor

        Args:
            model_dir (str): Path to the directory containing necessary data and model files for the prediction.
            parent_temp_dir (str): Path to the parent directory for temporary files. If None, uses the directory of this module.
            growth_temp (int): Growth temperature for the protein sequences.
            mt_mode (str): Mode for melting temperature prediction (Lysate or Purified).
            weight (float): Weight for the TM values predicted by this predictor in the combined score.
            docker_image (str): Docker image for the DeepStabP tool.
        """
        super().__init__()
        self.model_dir = model_dir
        self.parent_temp_dir = parent_temp_dir or os.path.dirname(
            os.path.abspath(__file__)
        )
        self.growth_temp = growth_temp
        self.mt_mode = mt_mode
        self.weight = weight
        self.docker_image = (
            docker_image
            if docker_image is not None
            else "ghcr.io/new-atlantis-labs/deepstabp:latest"
        )

    def _create_temp_dir(self):
        """Creates a unique temporary directory."""
        unique_id = str(uuid.uuid4())
        temp_dir = os.path.join(self.parent_temp_dir, f"deepstabp_temp_{unique_id}")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ) -> torch.Tensor:
        """
        Predicts TM values of protein sequences

        Args:
            sequences (List[str]): A list of protein sequences for which TM values are to be predicted.
            pdb_files (List[str]): A list of paths to PDB files corresponding to the input sequences.
            generation_id (str): Identifier for the generation of sequences (not used in this predictor).

        Returns:
            torch.Tensor: A tensor of predicted TM values corresponding to the input protein sequences.
        """
        temp_dir = self._create_temp_dir()
        try:
            input_filename = f"input_{uuid.uuid4()}.txt"
            input_file_path = os.path.join(temp_dir, input_filename)
            output_file_path = os.path.join(temp_dir, "tmvalues.csv")

            # Write the protein sequences to the temporary input file
            with open(input_file_path, "w") as seq_file:
                for n, sequence in enumerate(sequences):
                    header = f">seq_{n}"
                    seq_file.write(header + "\n" + sequence + "\n")

            # Construct the Docker run command
            docker_command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{self.model_dir}:/app/model_data",
                "-v",
                f"{temp_dir}:/app/data",
                f"{self.docker_image}",
                "--fasta_file",
                f"/app/data/{input_filename}",
                "--growth_temp",
                f"{self.growth_temp}",
                "--mt_mode",
                f"{self.mt_mode}",
                "--output_csv",
                "/app/data/tmvalues.csv",
                "--deepstabp_path",
                "trained_model/epoch=1-step=2316.ckpt",
                "--prot_t5_xl_dir",
                "/app/model_data/model",
                "--tokenizer_dir",
                "/app/model_data/tokenizer",
            ]

            # Run the Docker command
            subprocess.run(docker_command, check=True)

            # Read the TM values from the output file
            dataframe = pd.read_csv(output_file_path)
            tm_values = dataframe["Tm"]
            tm_tensor = torch.tensor(tm_values.values)

            return tm_tensor.unsqueeze(0).transpose(0, 1)

        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)


class AffinityGNINApredictor(Predictor):
    """
    A class to predict binding affinity using a Docker-based tool.
    """

    def __init__(
        self,
        ligand_sdf: str,
        weight: float = 1,
        requires_pdbs: bool = True,
        device: str = "cpu",
        suppress_warnings: bool = True,
        save_directory: str = None,
        parent_temp_dir: str = None,
        gnina_docker_image: str = "gnina/gnina:latest",
    ):
        """
        Initializes the AffinityGNINApredictor

        Args:
            ligand_sdf (str): Path to the ligand SDF file.
            weight (float): Weight for the affinity values predicted by this predictor in the combined score.
            requires_pdbs (bool): Whether the predictor requires PDB files for the input sequences. Default is True.
            device (str): Device to use for the prediction (cpu or cuda).
            suppress_warnings (bool): Whether to suppress warnings during prediction.
            parent_temp_dir (str): Path to the parent directory for temporary files. If None, uses the directory of this module.
            save_directory (str): Path to the directory to save the output files.
            gnina_docker_image (str): Docker image for the Gnina tool.
        """
        super().__init__()
        self.ligand_sdf = ligand_sdf
        self.parent_temp_dir = parent_temp_dir or os.path.dirname(
            os.path.abspath(__file__)
        )
        self.weight = weight
        self.device = device
        self.suppress_warnings = suppress_warnings
        if save_directory and not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        self.save_directory = save_directory
        self.gnina_docker_image = gnina_docker_image
        self.requires_pdbs = requires_pdbs

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ) -> torch.Tensor:
        """
        Predicts binding affinity values for protein sequences.

        Args:
            sequences (List[str]): A list of protein sequences for which binding affinity values are to be predicted.
            generation_id (str): Identifier for the generation of sequences.
            pdb_files (List[str]): A list of paths to PDB files corresponding to the input sequences.

        Returns:
            torch.Tensor: A tensor of predicted binding affinity values corresponding to the input protein sequences.
        """
        if pdb_files is None or len(pdb_files) != len(sequences):
            raise ValueError("PDB files must be provided for all sequences")

        # Create output directory
        docking_outdir = (
            os.path.join(self.save_directory, f"gen{generation_id}_gnina_docks")
            if self.save_directory is not None
            else None
        )

        affinities = []
        for pdb_file_path in pdb_files:
            # Extract the PDB name from the file name
            pdb_name = os.path.splitext(os.path.basename(pdb_file_path))[0]

            res = minimize_gnina_affinity(
                pdb_file_no_hetatms=pdb_file_path,
                ligand_sdf=self.ligand_sdf,
                docker_image=self.gnina_docker_image,
                autobox_add=2,
                cpu_only=True if self.device == "cpu" else False,
                suppress_warnings=self.suppress_warnings,
                output_dir=docking_outdir,
                output_file_name=f"wdock_{pdb_name}.sdf.gz",
            )
            min_affinity = min(res["Affinity"])
            affinities.append(min_affinity)

        affinities_tensor = torch.tensor(affinities).unsqueeze(0).transpose(0, 1)
        return -1 * affinities_tensor


class KineticPredictor(Predictor):
    """
    A class to predict kinetic parameters (kcat) for given protein sequences using CatPred tool.
    This predictor implements the CatPred model for enzyme kinetics prediction, specifically
    focused on turnover numbers (kcat values).
    """

    def __init__(
        self,
        substrate_smiles: str,
        substrate_name: str = "substrate",
        weights_dir: str = None,
        parameter: str = "kcat",
        weight: float = 1,
        device: str = "cuda",
        parent_temp_dir: str = None,
        docker_image: str = "catpred:latest",
        verbose: bool = False,
    ):
        """
        Initializes the KineticPredictor with necessary parameters.

        Args:
            substrate_smiles (str): SMILES string representation of the substrate.
            substrate_name (str): Name of the substrate (default: "substrate").
            weights_dir (str, optional): Path to the directory containing CatPred model weights.
                                       If None, uses default weights from the Docker container.
            parameter (str): Kinetic parameter to predict ('kcat' by default).
            weight (float): Weight for the values predicted by this predictor in the combined score.
            device (str): Computation device ('cuda' or 'cpu').
            parent_temp_dir (str): Path to the parent directory for temporary files.
                                 If None, uses the directory of this module.
            docker_image (str): Docker image name for the CatPred tool.
            verbose (bool): If True, show docker command outputs. If False, suppress all outputs.
        """
        super().__init__()
        self.substrate_smiles = substrate_smiles
        self.substrate_name = substrate_name
        self.weights_dir = weights_dir
        self.parameter = parameter
        self.weight = weight
        self.device = device
        self.parent_temp_dir = parent_temp_dir or os.path.dirname(
            os.path.abspath(__file__)
        )
        self.docker_image = docker_image
        self.verbose = verbose

    def _create_temp_dir(self):
        """Creates a unique temporary directory for input/output files."""
        unique_id = str(uuid.uuid4())
        temp_dir = os.path.join(self.parent_temp_dir, f"catpred_temp_{unique_id}")
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        return temp_dir, input_dir, output_dir

    def _prepare_input_file(self, sequences: List[str], input_dir: str) -> str:
        """
        Prepares the input CSV file for CatPred.

        Args:
            sequences (List[str]): List of protein sequences to predict.
            input_dir (str): Directory to save the input file.

        Returns:
            str: Path to the created input file.
        """
        input_file = os.path.join(input_dir, f"batch_{self.parameter}.csv")

        # Create DataFrame with the required columns
        df = pd.DataFrame(
            {
                "Substrate": [self.substrate_name] * len(sequences),
                "SMILES": [self.substrate_smiles] * len(sequences),
                "sequence": sequences,
                "pdbpath": [
                    f"seq{i+1}.pdb" for i in range(len(sequences))
                ],  # Placeholder PDB paths
            }
        )

        df.to_csv(input_file, index=True)
        return input_file

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ) -> torch.Tensor:
        """
        Predicts kinetic parameters using CatPred for the given sequences.

        Args:
            sequences (List[str]): A list of protein sequences to evaluate.
            pdb_files (List[str]): A list of paths to PDB files (not used in this predictor).
            generation_id (str): Identifier for the generation of sequences.

        Returns:
            torch.Tensor: Predicted kinetic parameter values as a tensor.
        """
        temp_dir, input_dir, output_dir = self._create_temp_dir()
        try:
            input_file = self._prepare_input_file(sequences, input_dir)

            # Construct Docker command
            docker_command = ["docker", "run", "--rm"]

            # Add GPU flag if using CUDA
            if self.device == "cuda":
                docker_command.extend(["--gpus", "all"])

            # Create list of volume mounts
            volume_mounts = [
                "-v",
                f"{input_dir}:/input",
                "-v",
                f"{output_dir}:/output",
            ]

            # Add weights directory to volume mounts if specified
            if self.weights_dir is not None:
                volume_mounts.extend(["-v", f"{self.weights_dir}:/weights"])

            # Add volume mounts to command
            docker_command.extend(volume_mounts)

            # Add volume mounts and command parameters
            docker_command.extend(
                [
                    self.docker_image,
                    "--parameter",
                    self.parameter,
                    "--input_file",
                    f"/input/batch_{self.parameter}.csv",
                ]
            )

            # Add weights directory if specified
            if self.weights_dir is not None:
                docker_command.extend(["--weights_dir", "/weights"])

            # Add GPU flag to CatPred command if using CUDA
            if self.device == "cuda":
                docker_command.append("--use_gpu")

            # Run the Docker command
            if self.verbose:
                subprocess.run(docker_command, check=True)
            else:
                with open(os.devnull, "w") as devnull:
                    subprocess.run(
                        docker_command,
                        check=True,
                        stdout=devnull,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONWARNINGS": "ignore"},
                    )

            # Read the final predictions output file
            output_file = os.path.join(
                output_dir, f"final_predictions_batch_{self.parameter}.csv"
            )
            predictions_df = pd.read_csv(output_file)

            # Determine the target column based on parameter
            if self.parameter == "kcat":
                target_col = "log10kcat_max"
            elif self.parameter == "km":
                target_col = "log10km_mean"
            else:  # ki parameter
                target_col = "log10ki_mean"

            # Convert predictions to tensor
            predictions = torch.tensor(predictions_df[target_col].values)
            return predictions.unsqueeze(0).transpose(0, 1)

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def __str__(self):
        return f"KineticPredictor(parameter={self.parameter}, device={self.device})"


class ThermostabilityPredictor(Predictor):
    """
    A class to predict protein thermostability using TemBERTure.
    This predictor implements the TemBERTure model for melting temperature prediction
    using three model replicas and averaging their predictions.
    """

    def __init__(
        self,
        weight: float = 1,
        device: str = "cuda",
        parent_temp_dir: str = None,
        docker_image: str = "temberture:latest",
        verbose: bool = False,
    ):
        """
        Initializes the ThermostabilityPredictor with necessary parameters.

        Args:
            weight (float): Weight for the values predicted by this predictor in the combined score.
            device (str): Computation device ('cuda' or 'cpu').
            parent_temp_dir (str): Path to the parent directory for temporary files.
                                 If None, uses the directory of this module.
            docker_image (str): Docker image name for the TemBERTure tool.
            verbose (bool): If True, show docker command outputs. If False, suppress all outputs.
        """
        super().__init__()
        self.weight = weight
        self.device = device
        self.parent_temp_dir = parent_temp_dir or os.path.dirname(
            os.path.abspath(__file__)
        )
        self.docker_image = docker_image
        self.verbose = verbose

    def _create_temp_dir(self):
        """Creates a unique temporary directory for input/output files."""
        unique_id = str(uuid.uuid4())
        temp_dir = os.path.join(self.parent_temp_dir, f"temberture_temp_{unique_id}")
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        return temp_dir, input_dir, output_dir

    def _prepare_input_file(self, sequences: List[str], input_dir: str) -> str:
        """
        Prepares the input FASTA file for TemBERTure.

        Args:
            sequences (List[str]): List of protein sequences to predict.
            input_dir (str): Directory to save the input file.

        Returns:
            str: Path to the created input file.
        """
        input_file = os.path.join(input_dir, "input_sequences.fasta")

        with open(input_file, "w") as f:
            for i, seq in enumerate(sequences, 1):
                f.write(f">seq{i}\n{seq}\n")

        return input_file

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ) -> torch.Tensor:
        """
        Predicts melting temperatures using TemBERTure for the given sequences.
        Returns the average of predictions from three model replicas.

        Args:
            sequences (List[str]): A list of protein sequences to evaluate.
            pdb_files (List[str]): A list of paths to PDB files (not used in this predictor).
            generation_id (str): Identifier for the generation of sequences.

        Returns:
            torch.Tensor: Predicted melting temperatures as a tensor (averaged across replicas).
        """
        temp_dir, input_dir, output_dir = self._create_temp_dir()
        try:
            input_file = self._prepare_input_file(sequences, input_dir)
            output_file = os.path.join(output_dir, "predictions.tsv")

            # Construct Docker command
            docker_command = ["docker", "run", "--rm"]

            # Add GPU flag if using CUDA
            if self.device == "cuda":
                docker_command.extend(["--gpus", "all"])

            # Add volume mounts
            docker_command.extend(
                [
                    "-v",
                    f"{input_dir}:/input",
                    "-v",
                    f"{output_dir}:/output",
                    self.docker_image,
                    "--tm",
                    "/input/input_sequences.fasta",
                    "/output/predictions.tsv",
                ]
            )

            # Run the Docker command
            if self.verbose:
                subprocess.run(docker_command, check=True)
            else:
                with open(os.devnull, "w") as devnull:
                    subprocess.run(
                        docker_command,
                        check=True,
                        stdout=devnull,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONWARNINGS": "ignore"},
                    )

            # Read predictions and compute average
            predictions_df = pd.read_csv(output_file, sep="\t")
            tm_columns = ["tm1", "tm2", "tm3"]
            tm_values = predictions_df[tm_columns].values
            average_tm = np.mean(tm_values, axis=1)

            # Convert to tensor
            predictions = torch.tensor(average_tm)
            return predictions.unsqueeze(0).transpose(0, 1)

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def __str__(self):
        return f"ThermostabilityPredictor(device={self.device})"


class EpHodPredictor(Predictor):
    """
    A class to predict optimal pH for protein sequences using EpHod.
    This predictor implements the EpHod model which outputs three predictions
    (RLATtr, SVR, and Ensemble) and returns their mean as the final prediction.
    """

    def __init__(
        self,
        weight: float = 1,
        device: str = "cuda",
        weights_dir: str = None,
        parent_temp_dir: str = None,
        docker_image: str = "ephod:latest",
        verbose: bool = False,
    ):
        """
        Initializes the EpHodPredictor with necessary parameters.

        Args:
            weight (float): Weight for the values predicted by this predictor in the combined score.
            device (str): Computation device ('cuda' or 'cpu').
            parent_temp_dir (str): Path to the parent directory for temporary files.
                                 If None, uses the current working directory.
            docker_image (str): Docker image name for the EpHod tool.
            verbose (bool): If True, show docker command outputs. If False, suppress all outputs.
        """
        super().__init__()
        self.weight = weight
        self.device = device
        self.weights_dir = weights_dir
        self.parent_temp_dir = parent_temp_dir or os.getcwd()
        self.docker_image = docker_image
        self.verbose = verbose

    def _create_temp_dir(self):
        """Creates a unique temporary directory for input/output files."""
        unique_id = str(uuid.uuid4())
        temp_dir = os.path.join(self.parent_temp_dir, f"ephod_temp_{unique_id}")
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        return temp_dir, input_dir, output_dir

    def _prepare_input_file(self, sequences: List[str], input_dir: str) -> str:
        """
        Prepares the input FASTA file for EpHod.

        Args:
            sequences (List[str]): List of protein sequences to predict.
            input_dir (str): Directory to save the input file.

        Returns:
            str: Path to the created input file.
        """
        input_file = os.path.join(input_dir, "input_sequences.fasta")

        with open(input_file, "w") as f:
            for i, seq in enumerate(sequences, 1):
                f.write(f">seq{i}\n{seq}\n")

        return input_file

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ) -> torch.Tensor:
        """
        Predicts optimal pH using EpHod for the given sequences.
        Returns the mean of RLATtr, SVR, and Ensemble predictions.

        Args:
            sequences (List[str]): A list of protein sequences to evaluate.
            pdb_files (List[str]): A list of paths to PDB files (not used in this predictor).
            generation_id (str): Identifier for the generation of sequences.

        Returns:
            torch.Tensor: Predicted optimal pH values as a tensor (mean of the three predictions).
        """
        temp_dir, input_dir, output_dir = self._create_temp_dir()
        try:
            input_file = self._prepare_input_file(sequences, input_dir)
            output_file = os.path.join(output_dir, "predictions.csv")

            # Construct Docker command
            docker_command = ["docker", "run", "--rm"]

            # Add GPU flag if using CUDA
            if self.device == "cuda":
                docker_command.extend(["--gpus", "all"])

            # Create list of volume mounts
            volume_mounts = [
                "-v",
                f"{input_dir}:/input",
                "-v",
                f"{output_dir}:/output",
            ]

            # Add weights directory to volume mounts if specified
            if self.weights_dir is not None:
                volume_mounts.extend(["-v", f"{self.weights_dir}:/weights"])

            # Add volume mounts to command
            docker_command.extend(volume_mounts)

            # Add command arguments
            command_args = [
                self.docker_image,
                "--fasta_path",
                "/input/input_sequences.fasta",
                "--save_dir",
                "/output",
                "--csv_name",
                "predictions.csv",
                "--verbose",
                "0",
            ]

            # Add weights directory argument if specified
            if self.weights_dir is not None:
                command_args.extend(["--weights_dir", "/weights"])

            docker_command.extend(command_args)

            # Run the Docker command
            if self.verbose:
                subprocess.run(docker_command, check=True)
            else:
                with open(os.devnull, "w") as devnull:
                    subprocess.run(
                        docker_command,
                        check=True,
                        stdout=devnull,
                        stderr=subprocess.STDOUT,
                    )

            # Read predictions and compute mean
            predictions_df = pd.read_csv(output_file)
            prediction_columns = ["RLATtr", "SVR", "Ensemble"]
            ph_values = predictions_df[prediction_columns].values
            mean_ph = np.mean(ph_values, axis=1)

            # Convert to tensor
            predictions = torch.tensor(mean_ph, dtype=torch.float32)
            return predictions.unsqueeze(0).transpose(0, 1)

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def __str__(self):
        return f"EpHodPredictor(device={self.device})"


class SolubilityGATSolPredictor(Predictor):
    """
    A class to predict protein solubility using a Docker-based GATSol model.
    """

    def __init__(
        self,
        weight: float = 1,
        requires_pdbs: bool = True,
        device: str = "cpu",
        verbose: bool = False,
        save_directory: Optional[str] = None,
        parent_temp_dir: Optional[str] = None,
        docker_image: str = "gatsol/predictor:latest",
        gatsol_weights_dir: str = None,
        esm_weights_dir: Optional[str] = None,
    ):
        """
        Initializes the SolubilityGATSolPredictor

        Args:
            weight (float): Weight for the solubility values predicted by this predictor in the combined score.
            requires_pdbs (bool): Whether the predictor requires PDB files for the input sequences. Default is True.
            device (str): Device to use for the prediction (cpu or cuda).
            verbose (bool): If True, show docker command outputs. If False, suppress all outputs.
            save_directory (str, optional): Path to the directory to save the output files.
            parent_temp_dir (str, optional): Path to the parent directory for temporary files.
            docker_image (str): Docker image name for the GATSol predictor.
            esm_weights_dir (str, optional): Directory containing ESM model weights.
            gatsol_weights_dir (str): Directory containing GATSol model weights.
        """
        super().__init__()
        self.weight = weight
        self.device = device
        self.verbose = verbose
        self.parent_temp_dir = parent_temp_dir or os.path.dirname(
            os.path.abspath(__file__)
        )

        if save_directory and not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        self.save_directory = save_directory

        self.docker_image = docker_image
        self.requires_pdbs = requires_pdbs
        self.esm_weights_dir = esm_weights_dir
        self.gatsol_weights_dir = gatsol_weights_dir

    def _create_temp_dir(self) -> str:
        """
        Creates a temporary directory with a unique name.

        Returns:
            str: Path to the created temporary directory
        """
        temp_dir = os.path.join(self.parent_temp_dir, f"gatsol_temp_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _validate_pdb_files(
        self, sequence_ids: List[str], pdb_files: List[str]
    ) -> None:
        """
        Validates that there is a corresponding PDB file for each sequence ID.

        Args:
            sequence_ids (List[str]): List of sequence identifiers
            pdb_files (List[str]): List of PDB file paths

        Raises:
            ValueError: If PDB files are missing for any sequence or if there are mismatches
        """
        if not pdb_files:
            raise ValueError("No PDB files provided")

        # Extract base names without extension for comparison
        pdb_basenames = [
            os.path.splitext(os.path.basename(pdb))[0] for pdb in pdb_files
        ]

        # Check for missing PDB files
        missing_pdbs = set(sequence_ids) - set(pdb_basenames)
        if missing_pdbs:
            raise ValueError(f"Missing PDB files for sequence IDs: {missing_pdbs}")

        # Check for extra PDB files
        extra_pdbs = set(pdb_basenames) - set(sequence_ids)
        if extra_pdbs:
            raise ValueError(
                f"Found extra PDB files not matching any sequence ID: {extra_pdbs}"
            )

    def infer_fitness(
        self,
        sequences: List[str],
        sequence_ids: List[str],
        pdb_files: Optional[List[str]] = None,
        generation_id: str = "",
    ) -> torch.Tensor:
        """
        Predicts solubility values for protein sequences.

        Args:
            sequences (List[str]): A list of protein sequences for which solubility is to be predicted.
            sequence_ids (List[str]): A list of unique identifiers for each sequence.
            pdb_files (List[str], optional): A list of paths to PDB files corresponding to the input sequences.
                                           The basename of each PDB file (without extension) should match
                                           its corresponding sequence_id.
            generation_id (str): Identifier for the generation of sequences.

        Returns:
            torch.Tensor: A tensor of predicted solubility values corresponding to the input protein sequences.

        Raises:
            ValueError: If the number of sequences and sequence_ids don't match, or if required PDB files
                      are missing or don't correspond to sequence IDs.
        """
        # Validate inputs
        if len(sequences) != len(sequence_ids):
            raise ValueError("Number of sequences must match number of sequence IDs")

        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("Sequence IDs must be unique")

        if self.requires_pdbs:
            if not pdb_files:
                raise ValueError("PDB files are required but none were provided")
            self._validate_pdb_files(sequence_ids, pdb_files)

        # Create temporary directory
        temp_dir = self._create_temp_dir()
        try:
            temp_dir_path = Path(temp_dir)

            # Create sequences DataFrame with provided IDs
            sequence_data = [
                {"id": seq_id, "sequence": seq}
                for seq_id, seq in zip(sequence_ids, sequences)
            ]
            sequences_file = temp_dir_path / "sequences.csv"
            pd.DataFrame(sequence_data).to_csv(sequences_file, index=False)

            # Create PDB directory and copy files
            pdb_dir = temp_dir_path / "pdb_files"
            pdb_dir.mkdir(exist_ok=True)
            if pdb_files:
                for pdb_file in pdb_files:
                    shutil.copy2(pdb_file, pdb_dir)

            # Create output directory
            output_dir = (
                Path(self.save_directory) / f"gen{generation_id}_solubility_predictions"
                if self.save_directory is not None
                else temp_dir_path / "output"
            )
            output_dir.mkdir(exist_ok=True)

            # Construct Docker command
            docker_cmd = ["docker", "run", "--rm"]

            # Add GPU flag if using CUDA
            if self.device == "cuda":
                docker_cmd.extend(["--gpus", "all"])

            docker_cmd.extend(
                [
                    "-v",
                    f"{sequences_file}:/app/sequences.csv",
                    "-v",
                    f"{pdb_dir}:/app/pdb_files",
                    "-v",
                    f"{output_dir}:/app/output",
                ]
            )

            # Mount GATSol weights directory
            docker_cmd.extend(
                ["-v", f"{self.gatsol_weights_dir}:/app/check_point/best_model"]
            )

            # Mount ESM weights directory (optional)
            if self.esm_weights_dir:
                docker_cmd.extend(["-v", f"{self.esm_weights_dir}:/app/esm_weights"])

            # Add image and command arguments
            docker_cmd.extend(
                [
                    self.docker_image,
                    "--sequences",
                    "/app/sequences.csv",
                    "--pdb-dir",
                    "/app/pdb_files",
                    "--output-dir",
                    "/app/output",
                    "--gatsol-weights-dir",
                    "/app/check_point/best_model",
                    "--device",
                    self.device,
                ]
            )

            if self.esm_weights_dir:
                docker_cmd.extend(["--esm-weights-dir", "/app/esm_weights"])

            # Run Docker container
            try:
                subprocess.run(docker_cmd, check=True, capture_output=not self.verbose)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Docker container execution failed: {e}")

            # Read predictions
            predictions_file = output_dir / "predictions.csv"
            if not predictions_file.exists():
                raise RuntimeError("Predictions file not found in container output")

            predictions_df = pd.read_csv(predictions_file)

            # Ensure predictions are in the same order as input sequences
            predictions_df = predictions_df.set_index("id").loc[sequence_ids]
            solubility_values = predictions_df["Solubility_hat"].values

            # Convert to tensor
            solubility_tensor = (
                torch.tensor(solubility_values).unsqueeze(0).transpose(0, 1)
            )
            return solubility_tensor

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @property
    def name(self) -> str:
        """Returns the name of the predictor."""
        return "GATSol_Solubility_Predictor"


import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
import pandas as pd
import torch


class GeoPocPredictor:
    """
    A class to predict optimal temperature, pH, and salt conditions for proteins using a Docker-based GeoPoc model.
    """

    # Class constants for pH and salt range mappings
    PH_RANGES = {"0-5": 2.5, "5-9": 7.0, "9-14": 11.5}

    SALT_RANGES = {
        "0-0.05%": 0.025,
        "0.05-4%": 2.025,
        ">4%": 5.0,  # Assuming a reasonable value for >4%
    }

    def __init__(
        self,
        task: str = "temp",
        weight: float = 1,
        device: str = "cpu",
        verbose: bool = False,
        save_directory: Optional[str] = None,
        parent_temp_dir: Optional[str] = None,
        docker_image: str = "geopoc/predictor:latest",
        model_weights_dir: Optional[str] = None,
        pdb_dir: Optional[str] = None,
        gpu_id: str = "0",
    ):
        """
        Initializes the GeoPocPredictor

        Args:
            task (str): Prediction task - one of "temp" (temperature), "pH", or "salt"
            weight (float): Weight for the predicted values in the combined score
            device (str): Device to use for prediction ("cpu" or "cuda")
            verbose (bool): If True, show docker command outputs
            save_directory (str, optional): Path to save output files
            parent_temp_dir (str, optional): Path to parent directory for temporary files
            docker_image (str): Docker image name for the GeoPoc predictor
            model_weights_dir (str, optional): Directory containing model weights
            pdb_dir (str, optional): Directory containing PDB files for the proteins
            gpu_id (str): GPU ID to use if device is "cuda"
        """
        if task not in ["temp", "pH", "salt"]:
            raise ValueError("Task must be one of: 'temp', 'pH', 'salt'")

        self.task = task
        self.weight = weight
        self.device = device
        self.verbose = verbose
        self.gpu_id = gpu_id
        self.parent_temp_dir = parent_temp_dir or os.path.dirname(
            os.path.abspath(__file__)
        )
        self.docker_image = docker_image
        self.model_weights_dir = model_weights_dir

        # Handle PDB directory
        self.pdb_dir = None
        if pdb_dir:
            if not os.path.isdir(pdb_dir):
                raise ValueError(f"PDB directory does not exist: {pdb_dir}")
            self.pdb_dir = Path(pdb_dir)

        if save_directory and not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        self.save_directory = save_directory

    def _create_temp_dir(self) -> str:
        """Creates a temporary directory with a unique name."""
        temp_dir = os.path.join(self.parent_temp_dir, f"geopoc_temp_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _create_feature_dirs(self, base_path: Path):
        """Creates the required feature directory structure."""
        # Create main subdirectories
        feature_dirs = [
            "pdb",
            "DSSP",
            "embedding",
            f"embedding/{self.task}",  # Task-specific embedding directory
        ]

        # Also create embedding directories for other tasks to avoid potential issues
        all_tasks = ["temp", "pH", "salt"]
        feature_dirs.extend(
            [f"embedding/{task}" for task in all_tasks if task != self.task]
        )

        # Create all directories with proper permissions
        for subdir in feature_dirs:
            dir_path = base_path / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            # Set directory permissions to 777 (same as chmod -R 777)
            os.chmod(str(dir_path), 0o777)

    def _create_fasta_file(
        self, sequences: List[str], sequence_ids: List[str], fasta_path: str
    ):
        """Creates a FASTA file from sequences and their IDs."""
        with open(fasta_path, "w") as f:
            for seq_id, seq in zip(sequence_ids, sequences):
                f.write(f">{seq_id}\n{seq}\n")
        os.chmod(fasta_path, 0o777)  # Set permissions for FASTA file

    def _copy_pdb_files(self, sequence_ids: List[str], target_pdb_dir: Path):
        """
        Copies PDB files from the class's pdb_dir to the target directory if they exist.

        Args:
            sequence_ids (List[str]): List of sequence identifiers
            target_pdb_dir (Path): Target directory for PDB files
        """
        if not self.pdb_dir:
            return

        # Copy any matching PDB files
        for seq_id in sequence_ids:
            pdb_file = self.pdb_dir / f"{seq_id}.pdb"
            if pdb_file.exists():
                shutil.copy2(pdb_file, target_pdb_dir)
                # Set file permissions to 777
                os.chmod(str(target_pdb_dir / f"{seq_id}.pdb"), 0o777)

    def _process_predictions(self, predictions_file: Path) -> torch.Tensor:
        """
        Processes the prediction output file and returns a tensor of predictions.

        Args:
            predictions_file (Path): Path to the predictions CSV file

        Returns:
            torch.Tensor: Tensor of predictions
        """
        predictions_df = pd.read_csv(predictions_file)

        if self.task == "temp":
            # For temperature, use the values directly
            values = predictions_df["temperature"].values
        else:
            # For pH and salt, convert class ranges to numerical values
            range_mapping = self.PH_RANGES if self.task == "pH" else self.SALT_RANGES
            class_column = f"class_{self.task}"
            values = predictions_df[class_column].map(range_mapping).values

        return torch.tensor(values).unsqueeze(1).float()

    def infer_fitness(
        self, sequences: List[str], sequence_ids: List[str], generation_id: str = ""
    ) -> torch.Tensor:
        """
        Predicts optimal conditions for protein sequences.

        Args:
            sequences (List[str]): List of protein sequences
            sequence_ids (List[str]): List of unique identifiers for each sequence
            generation_id (str): Identifier for the generation of sequences

        Returns:
            torch.Tensor: Tensor of predicted values
        """
        if len(sequences) != len(sequence_ids):
            raise ValueError("Number of sequences must match number of sequence IDs")

        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("Sequence IDs must be unique")

        # Create temporary directory structure
        temp_dir = self._create_temp_dir()
        try:
            temp_dir_path = Path(temp_dir)

            # Create feature directory structure
            feature_path = temp_dir_path / "features"
            self._create_feature_dirs(feature_path)

            # Copy PDB files if available
            self._copy_pdb_files(sequence_ids, feature_path / "pdb")

            # Create input directory and FASTA file
            input_dir = temp_dir_path / "input"
            input_dir.mkdir(exist_ok=True)
            fasta_file = input_dir / "sequences_clean.fasta"
            self._create_fasta_file(sequences, sequence_ids, str(fasta_file))

            # Create output directory
            output_dir = (
                Path(self.save_directory) / f"gen{generation_id}_geopoc_predictions"
                if self.save_directory
                else temp_dir_path / "output"
            )
            output_dir.mkdir(exist_ok=True)
            os.chmod(str(output_dir), 0o777)  # Set permissions for output directory

            # Construct Docker command
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{input_dir}:/app/GeoPoc/input",
                "-v",
                f"{feature_path}:/app/GeoPoc/features",
                "-v",
                f"{output_dir}:/app/GeoPoc/output",
            ]

            if self.model_weights_dir:
                docker_cmd.extend(["-v", f"{self.model_weights_dir}:/app/model"])

            if self.device == "cuda":
                docker_cmd.extend(["--gpus", "all"])

            docker_cmd.extend(
                [
                    self.docker_image,
                    "-i",
                    "/app/GeoPoc/input/sequences_clean.fasta",
                    "--feature_path",
                    "/app/GeoPoc/features/",
                    "-o",
                    "/app/GeoPoc/output/",
                    "--task",
                    self.task,
                    "--gpu",
                    self.gpu_id if self.device == "cuda" else "-1",
                ]
            )

            # Run Docker container
            try:
                process = subprocess.run(
                    docker_cmd, check=True, capture_output=True, text=True
                )
                if self.verbose:
                    print(process.stdout)
                    print(process.stderr)
            except subprocess.CalledProcessError as e:
                print(f"Docker stdout: {e.stdout}")
                print(f"Docker stderr: {e.stderr}")
                raise RuntimeError(f"Docker container execution failed: {e}")

            # Process predictions
            predictions_file = output_dir / f"{self.task}_preds.csv"
            if not predictions_file.exists():
                raise RuntimeError("Predictions file not found in container output")

            return self._process_predictions(predictions_file)

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @property
    def name(self) -> str:
        """Returns the name of the predictor."""
        return f"GeoPoc_{self.task}_Predictor"
