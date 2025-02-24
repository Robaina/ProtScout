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

from protscout.predictor_helpers import minimize_gnina_affinity


###############################################################################
# Helper mixin for persistent Docker container management
###############################################################################
class DockerContainerMixin:
    def _get_container_name(self) -> str:
        # Use a fixed container name based on class name.
        return f"{self.__class__.__name__}_container"

    def _get_persistent_dir(self) -> str:
        # Use a persistent directory (to be mounted into the container)
        return os.path.join(
            self.parent_temp_dir, f"{self.__class__.__name__}_persistent"
        )

    def _start_container(self, additional_mounts: List[str] = None):
        """
        Starts a persistent container (if not already running) that mounts the persistent
        directory (and any additional volumes) and runs a dummy command to stay alive.
        """
        self.container_name = self._get_container_name()
        self._persistent_dir = self._get_persistent_dir()
        os.makedirs(self._persistent_dir, exist_ok=True)

        # Check if the container already exists
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f", f"name={self.container_name}"],
            stdout=subprocess.PIPE,
            text=True,
        )
        if not result.stdout.strip():
            cmd = ["docker", "run", "-d", "--name", self.container_name]
            if self.device == "cuda":
                cmd.extend(["--gpus", "all"])
            # Mount the persistent directory at /persistent inside the container
            cmd.extend(["-v", f"{self._persistent_dir}:/persistent"])
            # Add any additional static mounts (e.g. model or weight directories)
            if additional_mounts:
                cmd.extend(additional_mounts)
            # Start the container with a dummy command to keep it running
            cmd.append(self.docker_image)
            cmd.extend(["tail", "-f", "/dev/null"])
            subprocess.run(cmd, check=True)

    def _create_temp_subdir(self, prefix: str) -> str:
        """
        Creates a temporary subdirectory under the persistent directory.
        This directory will be automatically mounted (via /persistent) inside the container.
        """
        temp_dir = os.path.join(self._persistent_dir, f"{prefix}_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _to_container_path(self, host_path: str) -> str:
        """
        Given a host_path that is under the persistent directory, returns the corresponding
        container path (which will be /persistent/<relative_path>).
        """
        rel_path = os.path.relpath(host_path, self._persistent_dir)
        return os.path.join("/persistent", rel_path)

    def shutdown_container(self):
        """
        Stops and removes the persistent container.
        Call this method when the predictor is no longer needed.
        """
        subprocess.run(["docker", "rm", "-f", self.container_name], check=True)


###############################################################################
# Predictor Base Class and CombinedPredictor (unchanged)
###############################################################################
class Predictor:
    """
    A base class for all predictors with common attributes and methods.
    """

    def __init__(self, weight: float = 1, requires_pdbs: bool = False):
        self.weight = weight
        self.requires_pdbs = requires_pdbs

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ):
        raise NotImplementedError("Subclasses should implement this method")

    def __str__(self):
        return f"{self.__class__.__name__}(weight={self.weight})"


class CombinedPredictor(Predictor):
    """
    Combines predictions from multiple expert predictors.
    """

    def __init__(self, predictors: List[object], weights: List[float] = None):
        if weights is None:
            weights = [predictor.weight for predictor in predictors]
        assert len(predictors) == len(weights), "Predictors and weights must match"
        super().__init__(
            weight=1, requires_pdbs=any(p.requires_pdbs for p in predictors)
        )
        self.predictors = predictors
        self.weights = weights

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ) -> torch.Tensor:
        combined_scores = None
        for predictor, weight in zip(self.predictors, self.weights):
            scores = predictor.infer_fitness(sequences, pdb_files, generation_id)
            weighted_scores = scores * weight
            combined_scores = (
                weighted_scores
                if combined_scores is None
                else combined_scores + weighted_scores
            )
        return combined_scores

    def __str__(self):
        return f"CombinedPredictor(predictors={len(self.predictors)})"


###############################################################################
# AffinityGNINApredictor (unchanged because it calls an external helper)
###############################################################################
class AffinityGNINApredictor(Predictor):
    """
    Predicts binding affinity using a Docker-based tool.
    (Note: This predictor still calls the external minimize_gnina_affinity helper.)
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
        super().__init__(weight=weight, requires_pdbs=requires_pdbs)
        self.ligand_sdf = ligand_sdf
        self.parent_temp_dir = parent_temp_dir or os.path.dirname(
            os.path.abspath(__file__)
        )
        self.device = device
        self.suppress_warnings = suppress_warnings
        if save_directory and not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        self.save_directory = save_directory
        self.gnina_docker_image = gnina_docker_image

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ) -> torch.Tensor:
        if pdb_files is None or len(pdb_files) != len(sequences):
            raise ValueError("PDB files must be provided for all sequences")

        docking_outdir = (
            os.path.join(self.save_directory, f"gen{generation_id}_gnina_docks")
            if self.save_directory is not None
            else None
        )

        affinities = []
        for pdb_file_path in pdb_files:
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

        affinities_tensor = torch.tensor(affinities).unsqueeze(1)
        return -1 * affinities_tensor

    def __str__(self):
        return f"AffinityGNINApredictor(device={self.device}, weight={self.weight})"


###############################################################################
# KineticPredictor (now re-using a persistent container)
###############################################################################
class KineticPredictor(Predictor, DockerContainerMixin):
    """
    Predicts kinetic parameters (kcat) using the CatPred tool via Docker.
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
        super().__init__(weight=weight, requires_pdbs=False)
        self.substrate_smiles = substrate_smiles
        self.substrate_name = substrate_name
        self.weights_dir = weights_dir
        self.parameter = parameter
        self.device = device
        self.parent_temp_dir = parent_temp_dir or os.path.dirname(
            os.path.abspath(__file__)
        )
        self.docker_image = docker_image
        self.verbose = verbose

        # Start the persistent container.
        extra_mounts = []
        if self.weights_dir is not None:
            extra_mounts.extend(["-v", f"{self.weights_dir}:/weights"])
        self._start_container(additional_mounts=extra_mounts)

    def _prepare_input_file(self, sequences: List[str], input_dir: str) -> str:
        input_file = os.path.join(input_dir, f"batch_{self.parameter}.csv")
        df = pd.DataFrame(
            {
                "Substrate": [self.substrate_name] * len(sequences),
                "SMILES": [self.substrate_smiles] * len(sequences),
                "sequence": sequences,
                "pdbpath": [f"seq{i+1}.pdb" for i in range(len(sequences))],
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
        temp_dir = self._create_temp_subdir("catpred_temp")
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        input_file = self._prepare_input_file(sequences, input_dir)
        container_input = self._to_container_path(input_dir)
        container_output = self._to_container_path(output_dir)

        # Build the command to run inside the container.
        exec_cmd = [
            "docker",
            "exec",
            self.container_name,
            "catpred",
            "--parameter",
            self.parameter,
            "--input_file",
            f"{container_input}/batch_{self.parameter}.csv",
        ]
        if self.weights_dir is not None:
            exec_cmd.extend(["--weights_dir", "/weights"])
        if self.device == "cuda":
            exec_cmd.append("--use_gpu")

        try:
            if self.verbose:
                subprocess.run(exec_cmd, check=True)
            else:
                with open(os.devnull, "w") as devnull:
                    subprocess.run(
                        exec_cmd,
                        check=True,
                        stdout=devnull,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONWARNINGS": "ignore"},
                    )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Docker container execution failed: {e}")

        output_file = os.path.join(
            output_dir, f"final_predictions_batch_{self.parameter}.csv"
        )
        predictions_df = pd.read_csv(output_file)
        if self.parameter == "kcat":
            target_col = "log10kcat_max"
        elif self.parameter == "km":
            target_col = "log10km_mean"
        else:
            target_col = "log10ki_mean"
        predictions = torch.tensor(predictions_df[target_col].values)
        return predictions.unsqueeze(1)

    def __str__(self):
        return f"KineticPredictor(parameter={self.parameter}, device={self.device}, weight={self.weight})"


###############################################################################
# ThermostabilityPredictor (now re-using a persistent container)
###############################################################################
class ThermostabilityPredictor(Predictor, DockerContainerMixin):
    """
    Predicts protein thermostability using TemBERTure via Docker.
    """

    def __init__(
        self,
        weight: float = 1,
        device: str = "cuda",
        parent_temp_dir: str = None,
        docker_image: str = "temberture:latest",
        verbose: bool = False,
    ):
        super().__init__(weight=weight, requires_pdbs=False)
        self.device = device
        self.parent_temp_dir = parent_temp_dir or os.path.dirname(
            os.path.abspath(__file__)
        )
        self.docker_image = docker_image
        self.verbose = verbose

        self._start_container()

    def _prepare_input_file(self, sequences: List[str], input_dir: str) -> str:
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
        temp_dir = self._create_temp_subdir("temberture_temp")
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        input_file = self._prepare_input_file(sequences, input_dir)
        container_input = self._to_container_path(input_dir)
        container_output = self._to_container_path(output_dir)

        exec_cmd = [
            "docker",
            "exec",
            self.container_name,
            "temberture",
            "--tm",
            f"{container_input}/input_sequences.fasta",
            f"{container_output}/predictions.tsv",
        ]
        try:
            if self.verbose:
                subprocess.run(exec_cmd, check=True)
            else:
                with open(os.devnull, "w") as devnull:
                    subprocess.run(
                        exec_cmd,
                        check=True,
                        stdout=devnull,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONWARNINGS": "ignore"},
                    )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Docker container execution failed: {e}")

        predictions_df = pd.read_csv(
            os.path.join(output_dir, "predictions.tsv"), sep="\t"
        )
        tm_columns = ["tm1", "tm2", "tm3"]
        average_tm = np.mean(predictions_df[tm_columns].values, axis=1)
        predictions = torch.tensor(average_tm)
        return predictions.unsqueeze(1)

    def __str__(self):
        return f"ThermostabilityPredictor(device={self.device}, weight={self.weight})"


###############################################################################
# SolubilityGATSolPredictor (now re-using a persistent container)
###############################################################################
class SolubilityGATSolPredictor(Predictor, DockerContainerMixin):
    """
    Predicts protein solubility using a Docker-based GATSol model.
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
        super().__init__(weight=weight, requires_pdbs=requires_pdbs)
        self.device = device
        self.verbose = verbose
        self.parent_temp_dir = parent_temp_dir or os.path.dirname(
            os.path.abspath(__file__)
        )
        if save_directory and not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        self.save_directory = save_directory
        self.docker_image = docker_image
        self.gatsol_weights_dir = gatsol_weights_dir
        self.esm_weights_dir = esm_weights_dir

        extra_mounts = []
        if self.gatsol_weights_dir:
            extra_mounts.extend(
                ["-v", f"{self.gatsol_weights_dir}:/app/check_point/best_model"]
            )
        if self.esm_weights_dir:
            extra_mounts.extend(["-v", f"{self.esm_weights_dir}:/app/esm_weights"])
        self._start_container(additional_mounts=extra_mounts)

    def _generate_sequence_ids(
        self, sequences: List[str], pdb_files: List[str] = None
    ) -> List[str]:
        if pdb_files and len(pdb_files) == len(sequences):
            return [os.path.splitext(os.path.basename(pdb))[0] for pdb in pdb_files]
        return [f"seq{i+1}" for i in range(len(sequences))]

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ) -> torch.Tensor:
        sequence_ids = self._generate_sequence_ids(sequences, pdb_files)
        if self.requires_pdbs and (not pdb_files or len(pdb_files) != len(sequences)):
            raise ValueError(
                "PDB files are required but not provided for all sequences"
            )

        temp_dir = self._create_temp_subdir("gatsol_temp")
        temp_dir_path = Path(temp_dir)
        sequence_data = [
            {"id": sid, "sequence": seq} for sid, seq in zip(sequence_ids, sequences)
        ]
        sequences_file = temp_dir_path / "sequences.csv"
        pd.DataFrame(sequence_data).to_csv(sequences_file, index=False)

        pdb_dir = temp_dir_path / "pdb_files"
        pdb_dir.mkdir(exist_ok=True)
        if pdb_files:
            for pdb_file in pdb_files:
                shutil.copy2(pdb_file, pdb_dir)

        output_dir = (
            (Path(self.save_directory) / f"gen{generation_id}_solubility_predictions")
            if self.save_directory
            else temp_dir_path / "output"
        )
        output_dir.mkdir(exist_ok=True)

        container_sequences = self._to_container_path(str(sequences_file))
        container_pdb_dir = self._to_container_path(str(pdb_dir))
        container_output = self._to_container_path(str(output_dir))

        exec_cmd = [
            "docker",
            "exec",
            self.container_name,
            "predictor",
            "--sequences",
            container_sequences,
            "--pdb-dir",
            container_pdb_dir,
            "--output-dir",
            container_output,
            "--gatsol-weights-dir",
            "/app/check_point/best_model",
            "--device",
            self.device,
        ]
        if self.esm_weights_dir:
            exec_cmd.extend(["--esm-weights-dir", "/app/esm_weights"])

        try:
            subprocess.run(exec_cmd, check=True, capture_output=not self.verbose)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Docker container execution failed: {e}")

        predictions_file = output_dir / "predictions.csv"
        if not predictions_file.exists():
            raise RuntimeError("Predictions file not found in container output")
        predictions_df = pd.read_csv(predictions_file)
        predictions_df = predictions_df.set_index("id").loc[sequence_ids]
        solubility_values = predictions_df["Solubility_hat"].values
        solubility_tensor = torch.tensor(solubility_values).unsqueeze(1)
        return solubility_tensor

    def __str__(self):
        return f"SolubilityGATSolPredictor(device={self.device}, weight={self.weight})"


###############################################################################
# GeoPocPredictor (now re-using a persistent container)
###############################################################################
class GeoPocPredictor(Predictor, DockerContainerMixin):
    """
    Predicts optimal temperature, pH, and salt conditions using a Docker-based GeoPoc model.
    """

    PH_RANGES = {"0-5": 2.5, "5-9": 7.0, "9-14": 11.5}
    SALT_RANGES = {"0-0.05%": 0.025, "0.05-4%": 2.025, ">4%": 5.0}

    def __init__(
        self,
        task: str = "temp",
        weight: float = 1,
        requires_pdbs: bool = True,
        device: str = "cpu",
        verbose: bool = False,
        save_directory: Optional[str] = None,
        parent_temp_dir: Optional[str] = None,
        docker_image: str = "geopoc/predictor:latest",
        model_weights_dir: Optional[str] = None,
        gpu_id: str = "0",
    ):
        super().__init__(weight=weight, requires_pdbs=requires_pdbs)
        if task not in ["temp", "pH", "salt"]:
            raise ValueError("Task must be one of: 'temp', 'pH', 'salt'")
        self.task = task
        self.device = device
        self.verbose = verbose
        self.gpu_id = gpu_id
        self.parent_temp_dir = parent_temp_dir or os.path.dirname(
            os.path.abspath(__file__)
        )
        self.docker_image = docker_image
        self.model_weights_dir = model_weights_dir
        if save_directory and not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        self.save_directory = save_directory

        extra_mounts = []
        if self.model_weights_dir:
            extra_mounts.extend(["-v", f"{self.model_weights_dir}:/app/model"])
        self._start_container(additional_mounts=extra_mounts)

    def _create_feature_dirs(self, base_path: Path):
        feature_dirs = ["pdb", "DSSP", "embedding", f"embedding/{self.task}"]
        all_tasks = ["temp", "pH", "salt"]
        feature_dirs.extend([f"embedding/{t}" for t in all_tasks if t != self.task])
        for subdir in feature_dirs:
            dir_path = base_path / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            os.chmod(str(dir_path), 0o777)

    def _create_fasta_file(
        self, sequences: List[str], sequence_ids: List[str], fasta_path: str
    ):
        with open(fasta_path, "w") as f:
            for seq_id, seq in zip(sequence_ids, sequences):
                f.write(f">{seq_id}\n{seq}\n")
        os.chmod(fasta_path, 0o777)

    def _copy_pdb_files(self, pdb_files: List[str], target_pdb_dir: Path):
        if not pdb_files:
            return
        for pdb_file in pdb_files:
            if os.path.exists(pdb_file):
                file_name = os.path.basename(pdb_file)
                shutil.copy2(pdb_file, target_pdb_dir / file_name)
                os.chmod(str(target_pdb_dir / file_name), 0o777)

    def _process_predictions(self, predictions_file: Path) -> torch.Tensor:
        predictions_df = pd.read_csv(predictions_file)
        if self.task == "temp":
            values = predictions_df["temperature"].values
        else:
            mapping = self.PH_RANGES if self.task == "pH" else self.SALT_RANGES
            values = predictions_df[f"class_{self.task}"].map(mapping).values
        return torch.tensor(values).unsqueeze(1).float()

    def infer_fitness(
        self,
        sequences: List[str],
        pdb_files: List[str] = None,
        generation_id: str = "",
    ) -> torch.Tensor:
        if pdb_files and self.requires_pdbs:
            sequence_ids = [
                os.path.splitext(os.path.basename(pdb))[0] for pdb in pdb_files
            ]
            if len(sequences) != len(sequence_ids):
                raise ValueError("Number of sequences must match number of PDB files")
        else:
            sequence_ids = [f"seq{i+1}" for i in range(len(sequences))]
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("Sequence IDs must be unique")

        temp_dir = self._create_temp_subdir("geopoc_temp")
        temp_dir_path = Path(temp_dir)
        feature_path = temp_dir_path / "features"
        self._create_feature_dirs(feature_path)
        if pdb_files and self.requires_pdbs:
            self._copy_pdb_files(pdb_files, feature_path / "pdb")
        input_dir = temp_dir_path / "input"
        input_dir.mkdir(exist_ok=True)
        fasta_file = input_dir / "sequences_clean.fasta"
        self._create_fasta_file(sequences, sequence_ids, str(fasta_file))
        output_dir = (
            (Path(self.save_directory) / f"gen{generation_id}_geopoc_predictions")
            if self.save_directory
            else temp_dir_path / "output"
        )
        output_dir.mkdir(exist_ok=True)
        os.chmod(str(output_dir), 0o777)

        container_input = self._to_container_path(str(input_dir))
        container_features = self._to_container_path(str(feature_path))
        container_output = self._to_container_path(str(output_dir))

        exec_cmd = [
            "docker",
            "exec",
            self.container_name,
            "geopoc",
            "-i",
            f"{container_input}/sequences_clean.fasta",
            "--feature_path",
            f"{container_features}/",
            "-o",
            f"{container_output}/",
            "--task",
            self.task,
            "--gpu",
            self.gpu_id if self.device == "cuda" else "-1",
        ]
        try:
            process = subprocess.run(
                exec_cmd, check=True, capture_output=True, text=True
            )
            if self.verbose:
                print(process.stdout)
                print(process.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Docker stdout: {e.stdout}")
            print(f"Docker stderr: {e.stderr}")
            raise RuntimeError(f"Docker container execution failed: {e}")

        predictions_file = output_dir / f"{self.task}_preds.csv"
        if not predictions_file.exists():
            raise RuntimeError("Predictions file not found in container output")
        return self._process_predictions(predictions_file)

    def __str__(self):
        return f"GeoPocPredictor(task={self.task}, device={self.device}, weight={self.weight})"
