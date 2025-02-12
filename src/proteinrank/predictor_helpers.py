from __future__ import annotations

import os
import re
import shutil
import tempfile
import subprocess
from typing import Optional, List, Dict, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from proteus.wrappers import run_diffdock


def run_rasp(
    query_protein: str,
    chain: str,
    output_dir: str,
    device: str = "cpu",
    docker_image: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Runs the RaSP Docker container with specified arguments.

    Args:
        query_protein (str): Path to the query protein PDB file within the Docker context.
        chain (str): Protein chain to process.
        output_dir (str): Directory within the Docker context to store outputs.
        pdb_parser_scripts_dir (str): Directory within the Docker context containing PDB parser scripts.
        models_dir (str): Directory within the Docker context where models are stored.
        pdb_frequencies (str): Path to PDB frequencies file within the Docker context.
        device (str, optional): Computation device ('CPU' or 'GPU'). Defaults to 'CPU'.
        docker_image (Optional[str], optional): Docker image to use. Defaults to 'ghcr.io/new-atlantis-labs/rasp:latest'
        verbose (bool, optional): Whether to print subprocess output. Defaults to False.

    Returns:
        dict: A dictionary containing the variant and predicted ddG values.
    """
    if docker_image is None:
        docker_image = "ghcr.io/new-atlantis-labs/rasp:latest"

    input_dir = os.path.abspath(os.path.dirname(query_protein))
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(f"{output_dir}/raw"):
        os.makedirs(f"{output_dir}/raw")
    if not os.path.exists(f"{output_dir}/parsed"):
        os.makedirs(f"{output_dir}/parsed")
    if not os.path.exists(f"{output_dir}/cleaned"):
        os.makedirs(f"{output_dir}/cleaned")
    if not os.path.exists(f"{output_dir}/predictions"):
        os.makedirs(f"{output_dir}/predictions")

    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{input_dir}:/app/input",
        "-v",
        f"{output_dir}:/app/output",
        docker_image,
        "--query_protein",
        f"/app/input/{os.path.basename(query_protein)}",
        "--chain",
        chain,
        "--output_dir",
        "/app/output",
        "--device",
        device,
    ]
    process = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if verbose:
        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)
    df = pd.read_csv(f"{output_dir}/predictions/cavity_pred_CUSTOM_A.csv")
    mut_ddg = dict(zip(df["variant"], df["score_ml"]))
    return mut_ddg


def run_thermompnn(
    pdb_file: str,
    output_file: str,
    chain: str = "A",
    device: str = "cpu",
    docker_image: str = None,
    verbose: bool = False,
) -> dict:
    """
    Runs the ThermoMPNN Docker container with specified arguments.

    Args:
        pdb_file (str): Path to the PDB file.
        output_file (str): Path where the output CSV will be stored.
        chain (str, optional): Protein chain to process. Defaults to "A".
        device (str, optional): Computation device ('cpu' or 'gpu'). Defaults to 'cpu'.
        docker_image (Optional[str], optional): Docker image to use. Defaults to 'ghcr.io/new-atlantis-labs/thermompnn:latest'.
        verbose (bool, optional): Whether to print subprocess output. Defaults to False.

    Returns:
        dict: A dictionary containing the variant and predicted ddG values.
    """
    if docker_image is None:
        docker_image = "ghcr.io/new-atlantis-labs/thermompnn:latest"

    pdb_dir = os.path.abspath(os.path.dirname(pdb_file))
    output_dir = os.path.abspath(os.path.dirname(output_file))

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the Docker run command
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{pdb_dir}:/data",
        "-v",
        f"{output_dir}:/output",
        docker_image,
        "--pdb",
        f"/data/{os.path.basename(pdb_file)}",
        "--output",
        f"/output/{os.path.basename(output_file)}",
        "--chain",
        chain,
    ]
    if device == "cpu":
        command.append("--only-cpus")

    process = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if verbose:
        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)
    os.path.basename(pdb_file).split(".")[0]
    # df = pd.read_csv(f"{output_dir}/thermompnn_{pdb_name}.txt")
    df = pd.read_csv(f"{output_file}")
    mut_ddg = dict(zip(df["variant"], df["ddG_pred"]))
    return mut_ddg


def get_gnina_affinity(
    pdb_file_no_hetatms: str,
    ligand_sdf: str,
    docker_image: str = "gnina/gnina:latest",
    output_dir: str = None,
    output_file_name: str = "wdocking.sdf.gz",
    cpu_only: bool = False,
    suppress_warnings: bool = False,
) -> dict:
    """
    Calculates various scoring metrics for the affinity of a ligand to a receptor using scoring mode only, through a Docker container.

    Parameters:
        pdb_file_no_hetatms (str): The file path of the receptor PDB file without HETATM entries.
        ligand_sdf (str): The file path of the ligand in SDF format.
        docker_image (str): Docker image tag for gnina, defaults to "gnina/gnina:latest".
        output_dir (str, optional): The directory path where output files will be stored. If specified, this directory will be mounted to the Docker container.
        output_file_name (str, optional): The name of the output file to save the results. Defaults to "wdocking.sdf.gz".
        cpu_only (bool, optional): If True, run the prediction on CPU only.
        suppress_warnings (bool, optional): If True, suppress warnings from the gnina command.

    Returns:
        dict: A dictionary containing the following keys and values:
            - 'Affinity' (float, kcal/mol): Predicted binding energy of the ligand to the receptor.
            - 'CNNscore' (float, unitless): Score from 0 to 1 indicating the likelihood of correct binding pose.
            - 'CNNaffinity' (float, kcal/mol): Predicted binding affinity based on a convolutional neural network.
            - 'CNNvariance' (float, unitless): Variance of the CNN prediction, indicating prediction uncertainty.
            - 'Intramolecular energy' (float, kcal/mol): Energy associated with internal ligand stability.

    Description:
        This function runs the gnina command in score-only mode from within a Docker container for the given receptor and ligand files.
        It extracts multiple scoring metrics from the command output, including traditional and CNN-based scores, and can optionally save the output to a specified file.
    """
    # Resolve absolute paths and parent directories of the files
    pdb_path = os.path.abspath(pdb_file_no_hetatms)
    pdb_dir = os.path.dirname(pdb_path)
    ligand_path = os.path.abspath(ligand_sdf)
    ligand_dir = os.path.dirname(ligand_path)

    if cpu_only:
        gpu_flag = ""
    else:
        gpu_flag = "--gpus all"

    command = [
        f"docker run --rm {gpu_flag}",
        f'-v "{pdb_dir}:/pdb"',
        f'-v "{ligand_dir}:/ligand"',
    ]
    if output_dir:
        output_path = os.path.abspath(output_dir)
        output_mount = f'-v "{output_path}:/output"'
        command.append(output_mount)
        output_file = f"/output/{output_file_name}"
    else:
        output_file = ""

    command.append(
        f"{docker_image} gnina --score_only "
        f'-r "/pdb/{os.path.basename(pdb_file_no_hetatms)}" '
        f'-l "/ligand/{os.path.basename(ligand_sdf)}" '
        f'-o "{output_file}"'
    )

    # Execute the Docker command
    full_command = " ".join(command)
    if suppress_warnings:
        full_command += " 2>/dev/null"
    scored_stdout = subprocess.check_output(full_command, shell=True, text=True)

    # Extract metrics using regex
    scores = {
        "Affinity": float(re.findall("Affinity:\\s*([-\\.\\d]+)", scored_stdout)[0]),
        "CNNscore": float(re.findall("CNNscore:\\s*([-\\.\\d]+)", scored_stdout)[0]),
        "CNNaffinity": float(
            re.findall("CNNaffinity:\\s*([-\\.\\d]+)", scored_stdout)[0]
        ),
        "CNNvariance": float(
            re.findall("CNNvariance:\\s*([-\\.\\d]+)", scored_stdout)[0]
        ),
        "Intramolecular energy": float(
            re.findall("Intramolecular energy:\\s*([-\\.\\d]+)", scored_stdout)[0]
        ),
    }

    return scores


def minimize_gnina_affinity(
    pdb_file_no_hetatms: str,
    ligand_sdf: str,
    docker_image: str = "gnina/gnina:latest",
    output_dir: str = None,
    output_file_name: str = "wdocking_minimized.sdf.gz",
    autobox_add: int = 2,
    cpu_only: bool = False,
    suppress_warnings: bool = False,
) -> dict:
    """
    Calculates various metrics for the minimized affinity of a ligand to a receptor after local adjustments, through a Docker container.

    Parameters:
        pdb_file_no_hetatms (str): The file path of the receptor PDB file without HETATM entries.
        ligand_sdf (str): The file path of the ligand in SDF format.
        docker_image (str): Docker image tag for gnina, defaults to "gnina/gnina:latest".
        output_dir (str, optional): The directory path where output files will be stored. If specified, this directory will be mounted to the Docker container.
        output_file_name (str, optional): The name of the output file to save the results. Defaults to "wdocking_minimized.sdf.gz".
        autobox_add (int, optional): The number of angstroms to add to the autobox size. Defaults to 2.
        cpu_only (bool, optional): If True, run the prediction on CPU only.
        suppress_warnings (bool, optional): If True, suppress warnings from the gnina command.

    Returns:
        dict: A dictionary containing the following keys and values:
            - 'Affinity' (list of floats, kcal/mol): The binding energies of the ligand to the receptor. Multiple values may represent different stages or types of affinity calculation.
            - 'RMSD' (float, Angstroms): Root mean square deviation of the docking pose from a reference pose.
            - 'CNNscore' (float, unitless): A neural network-based score indicating the likelihood of a correct binding pose.
            - 'CNNaffinity' (float, kcal/mol): Predicted binding affinity based on a convolutional neural network.
            - 'CNNvariance' (float, unitless): Variance of the CNN prediction, indicating prediction uncertainty.

    Description:
        This function runs the gnina command in local-only minimization mode from within a Docker container for the given receptor and ligand files.
        It extracts multiple scoring metrics from the command output, including traditional and CNN-based scores, and can optionally save the output to a specified file.
    """
    # Resolve absolute paths and parent directories of the files
    pdb_path = os.path.abspath(pdb_file_no_hetatms)
    pdb_dir = os.path.dirname(pdb_path)
    ligand_path = os.path.abspath(ligand_sdf)
    ligand_dir = os.path.dirname(ligand_path)

    if cpu_only:
        gpu_flag = ""
    else:
        gpu_flag = "--gpus all"

    command = [
        f"docker run --rm {gpu_flag}",
        f'-v "{pdb_dir}:/pdb"',
        f'-v "{ligand_dir}:/ligand"',
    ]
    if output_dir:
        output_path = os.path.abspath(output_dir)
        output_mount = f'-v "{output_path}:/output"'
        command.append(output_mount)
        output_file = f"/output/{output_file_name}"
        output_cmd = f"-o {output_file}"
    else:
        output_cmd = ""

    command.append(
        f"{docker_image} gnina --local_only --minimize "
        f'-r "/pdb/{os.path.basename(pdb_file_no_hetatms)}" '
        f'-l "/ligand/{os.path.basename(ligand_sdf)}" '
        f'--autobox_ligand "/ligand/{os.path.basename(ligand_sdf)}" --autobox_add {autobox_add} '
        f"{output_cmd}"
    )

    # Execute the Docker command
    full_command = " ".join(command)
    if suppress_warnings:
        full_command += " 2>/dev/null"

    minimized_stdout = subprocess.check_output(full_command, shell=True, text=True)

    # Extract metrics using regex
    affinities = re.findall(
        "Affinity:\\s*([\\-\\.\\d]+\\s*[\\-\\.\\d]+)", minimized_stdout
    )
    affinity_values = (
        [float(val) for val in affinities[0].split()] if affinities else []
    )
    rmsd = float(re.findall("RMSD:\\s*([\\-\\.\\d]+)", minimized_stdout)[0])
    cnnscore = float(re.findall("CNNscore:\\s*([\\-\\.\\d]+)", minimized_stdout)[0])
    cnnaffinity = float(
        re.findall("CNNaffinity:\\s*([\\-\\.\\d]+)", minimized_stdout)[0]
    )
    cnnvariance = float(
        re.findall("CNNvariance:\\s*([\\-\\.\\d]+)", minimized_stdout)[0]
    )

    return {
        "Affinity": affinity_values,
        "RMSD": rmsd,
        "CNNscore": cnnscore,
        "CNNaffinity": cnnaffinity,
        "CNNvariance": cnnvariance,
    }


def setup_logging(output_dir: str, generation_id: str) -> logging.Logger:
    """
    Set up logging configuration for the folding process.

    Args:
        output_dir (str): Directory to save the log file
        generation_id (str): Generation ID for the log file name

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create a logger
    logger = logging.getLogger(f"fold_sequences_gen{generation_id}")
    logger.setLevel(logging.INFO)

    # Create file handler
    log_file = os.path.join(
        logs_dir,
        f"folding_gen{generation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def run_esmfold(
    fasta_path: str,
    pdb_output_path: str,
    model_dir: str = None,
    num_recycles: int = None,
    max_tokens_per_batch: int = None,
    chunk_size: int = None,
    cpu_only: bool = False,
    cpu_offload: bool = False,
    docker_image: str = None,
    logger: Optional[logging.Logger] = None,
) -> Union[None, Tuple[bool, str]]:
    """
    Runs the ESMFold protein structure prediction tool within a Docker container.

    Parameters:
        [previous parameters remain the same]
        logger (Optional[logging.Logger]): Logger instance for recording the process.
                                         If None, logging is disabled.

    Returns:
        If logger is None:
            None (original behavior)
        If logger is provided:
            Tuple[bool, str]: (Success status, Error message if any)
    """
    if docker_image is None:
        docker_image = "esmfold:base"

    if logger:
        logger.info(f"Starting ESMFold for {fasta_path}")

    pdb_output_path = os.path.abspath(pdb_output_path)
    input_dir = os.path.dirname(fasta_path)
    os.makedirs(os.path.join(pdb_output_path, "pdbs"), exist_ok=True)
    os.makedirs(os.path.join(pdb_output_path, "logs"), exist_ok=True)

    cpu_flag, gpu_flag = "", ""
    if cpu_only:
        cpu_flag += "--cpu-only"
        if cpu_offload:
            cpu_flag += " --cpu-offload"
        if logger:
            logger.info("Running in CPU-only mode")
    else:
        gpu_flag += "--gpus all"
        if logger:
            logger.info("Running with GPU support")

    command = f"""
    docker run --rm {gpu_flag} \\
      -v {input_dir}:/home/vscode/input \\
      -v {pdb_output_path}:/home/vscode/output \\
      {docker_image} \\
      -i /home/vscode/input/{os.path.basename(fasta_path)} \\
      -o /home/vscode/output/pdbs {cpu_flag} \\
    """

    if model_dir:
        command += f"-m {model_dir} "
    if num_recycles:
        command += f"--num-recycles {num_recycles} "
    if max_tokens_per_batch:
        command += f"--max-tokens-per-batch {max_tokens_per_batch} "
    if chunk_size:
        command += f"--chunk-size {chunk_size} "

    log_file = os.path.join(pdb_output_path, "logs/pred.log")
    err_file = os.path.join(pdb_output_path, "logs/pred.err")
    command += f"> {log_file} 2>{err_file}"

    try:
        if logger:
            logger.info(f"Executing ESMFold command")
        subprocess.run(command, shell=True, check=True)
        if logger:
            logger.info("ESMFold command completed successfully")
            return True, ""
        return None

    except subprocess.CalledProcessError as e:
        error_msg = f"Command '{e.cmd}' returned non-zero exit status {e.returncode}."

        if logger:
            logger.error(error_msg)
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_content = f.read()
                    logger.error(f"ESMFold log output:\n{log_content}")
            if os.path.exists(err_file):
                with open(err_file, "r") as f:
                    err_content = f.read()
                    logger.error(f"ESMFold error output:\n{err_content}")
            return False, error_msg
        else:
            # Original error handling behavior
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    print("Contents of pred.log:")
                    print(f.read())
            if os.path.exists(err_file):
                with open(err_file, "r") as f:
                    print("Contents of pred.err:")
                    print(f.read())
            raise


def fold_sequences_with_esmfold(
    sequences: List[str],
    output_dir: str,
    generation_id: str,
    esmfold_docker_image: str = "ghcr.io/new-atlantis-labs/esmfold:latest",
    cpu_only: bool = False,
    fasta_filename: str = None,
    enable_logging: bool = False,
) -> Union[List[str], Dict[str, List[str]]]:
    """
    Fold sequences using ESMFold and save the PDB files.

    Args:
        sequences (List[str]): List of sequences to be folded.
        output_dir (str): Directory to save the PDB files.
        generation_id (str): Generation ID for naming the files.
        esmfold_docker_image (str): Docker image for ESMFold.
        cpu_only (bool): Whether to use CPU only for folding.
        fasta_filename (str, optional): Custom filename for the input FASTA file.
        enable_logging (bool): If True, enables detailed logging and returns additional information
                             about failed sequences. Defaults to False for backward compatibility.

    Returns:
        If enable_logging is False:
            List[str]: List of paths to the generated PDB files (original behavior)
        If enable_logging is True:
            Dict[str, List[str]]: Dictionary containing:
                - 'successful': List of paths to successfully generated PDB files
                - 'failed': List of sequences that failed to fold
    """
    logger = setup_logging(output_dir, generation_id) if enable_logging else None

    if logger:
        logger.info(f"Starting folding process for {len(sequences)} sequences")
    else:
        print(f"Folding {len(sequences)} sequence variants.")

    # Create a temporary directory for the input FASTA file
    temp_dir = os.path.join(output_dir, f"temp_gen{generation_id}")
    os.makedirs(temp_dir, exist_ok=False)

    result = {"successful": [], "failed": []} if enable_logging else []

    try:
        if fasta_filename is None:
            fasta_filename = f"gen{generation_id}_input.fasta"
        fasta_path = os.path.join(temp_dir, fasta_filename)
        esm_output_path = os.path.join(temp_dir, "esmfold")
        pdb_output_path = os.path.join(esm_output_path, "pdbs")

        # Write sequences to FASTA file
        if logger:
            logger.info("Writing sequences to FASTA file")
        with open(fasta_path, "w") as fasta_file:
            for i, seq in enumerate(sequences):
                fasta_file.write(f">seq{i}\n{seq}\n")

        # Run ESMFold
        esmfold_result = run_esmfold(
            fasta_path=fasta_path,
            docker_image=esmfold_docker_image,
            pdb_output_path=esm_output_path,
            cpu_only=cpu_only,
            logger=logger,
        )

        if logger and esmfold_result is not None and not esmfold_result[0]:
            logger.error(f"ESMFold run failed: {esmfold_result[1]}")
            result["failed"] = sequences
            return result

        # Create permanent directory for successful PDB files
        permanent_pdb_dir = os.path.join(output_dir, f"gen{generation_id}")
        os.makedirs(permanent_pdb_dir, exist_ok=True)

        # Process output files
        if logger:
            logger.info("Processing ESMFold output files")
            successfully_folded_files = set()

        # Simple file handling like in the original version
        for pdb_file in os.listdir(pdb_output_path):
            if pdb_file.endswith(".pdb"):
                src = os.path.join(pdb_output_path, pdb_file)
                dst = os.path.join(permanent_pdb_dir, pdb_file)
                shutil.move(src, dst)

                if enable_logging:
                    result["successful"].append(dst)
                    successfully_folded_files.add(pdb_file)
                    logger.info(f"Successfully processed {pdb_file}")
                else:
                    result.append(dst)

        # If logging is enabled, identify which sequences failed to fold
        if logger:
            for i, seq in enumerate(sequences):
                expected_file = f"seq{i}.pdb"
                if expected_file not in successfully_folded_files:
                    result["failed"].append(seq)
                    logger.warning(f"Sequence {i} failed to fold: {seq[:50]}...")

            logger.info(
                f"Folding completed. Successful: {len(result['successful'])}, Failed: {len(result['failed'])}"
            )

        return result

    except Exception as e:
        if logger:
            logger.error(
                f"Unexpected error during folding process: {str(e)}", exc_info=True
            )
            result["failed"] = sequences
            return result
        else:
            raise

    finally:
        if logger:
            logger.info("Cleaning up temporary directory")
        shutil.rmtree(temp_dir, ignore_errors=True)


@dataclass
class DockingResult:
    """Container for docking and affinity results.

    Attributes:
        diffdock_output_dir: Path to directory containing DiffDock outputs
        pose_file: Path to the best pose SDF file
        affinity: Predicted binding affinity in kcal/mol
        cnn_score: CNN-based score for pose quality (0-1)
        cnn_affinity: CNN-based affinity prediction in kcal/mol
        cnn_variance: Uncertainty in CNN predictions
        rmsd: Root mean square deviation between initial and minimized pose
    """

    diffdock_output_dir: Path
    pose_file: Path
    affinity: float
    cnn_score: float
    cnn_affinity: float
    cnn_variance: float
    rmsd: float


def dock_and_score_affinity(
    protein_file: str | Path,
    ligand_file: str | Path,
    output_dir: Optional[str | Path] = None,
    diffdock_image: str = "rbgcsail/diffdock",
    diffdock_model_dir: str = None,
    suppress_diffdock_logs: bool = True,
    gnina_image: str = "gnina/gnina:latest",
    inference_steps: int = 20,
    samples_per_complex: int = 10,
    batch_size: int = 10,
    no_final_step_noise: bool = True,
    autobox_add: int = 2,
    cpu_only: bool = False,
    cleanup_intermediates: bool = True,
) -> DockingResult:
    """Run protein-ligand docking with DiffDock followed by GNINA affinity scoring.

    This function performs two steps:
    1. Uses DiffDock to generate protein-ligand binding poses
    2. Uses GNINA to minimize the best pose and calculate binding affinity

    Args:
        protein_file: Path to input protein structure file (PDB format)
        ligand_file: Path to input ligand file (SDF format)
        output_dir: Directory to store results (temporary dir used if None)
        diffdock_image: Docker image for DiffDock
        gnina_image: Docker image for GNINA
        inference_steps: Number of inference steps for DiffDock
        samples_per_complex: Number of poses to generate per complex
        batch_size: Batch size for DiffDock inference
        actual_steps: Actual number of steps to run in DiffDock
        no_final_step_noise: Whether to disable noise in final DiffDock step
        autobox_add: Padding in Angstroms for GNINA search box
        cpu_only: Whether to use CPU only (vs GPU)
        cleanup_intermediates: Whether to remove intermediate files

    Returns:
        DockingResult containing paths to output files and computed metrics

    Raises:
        subprocess.CalledProcessError: If either DiffDock or GNINA fails
        FileNotFoundError: If input files don't exist or required outputs not found
    """
    # Convert paths to Path objects
    protein_file = Path(protein_file)
    ligand_file = Path(ligand_file)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # First run DiffDock to get poses
    diffdock_results_dir_name = "diffdock_results"
    run_diffdock(
        protein_input=str(protein_file),
        ligand_path=str(ligand_file),
        output_dir=str(output_dir),
        complex_name=diffdock_results_dir_name,
        docker_image=diffdock_image,
        model_dir=diffdock_model_dir,
        only_cpu=False,
        inference_steps=inference_steps,
        samples_per_complex=samples_per_complex,
        batch_size=batch_size,
        no_final_step_noise=no_final_step_noise,
        suppress_logs=suppress_diffdock_logs,
    )
    # Find the DiffDock results directory and best pose
    diffdock_dir = Path(output_dir) / diffdock_results_dir_name
    if not diffdock_dir.exists():
        raise FileNotFoundError(
            f"DiffDock output directory not found at {diffdock_dir}"
        )
    best_pose = diffdock_dir / "rank1.sdf"

    if not best_pose.exists():
        raise FileNotFoundError(f"Best pose file not found at {best_pose}")

    # Create a directory for GNINA outputs
    gnina_dir = output_dir / "gnina_results"
    gnina_dir.mkdir(exist_ok=True)

    # Run GNINA minimization and scoring on the best pose
    scores = minimize_gnina_affinity(
        pdb_file_no_hetatms=str(protein_file),
        ligand_sdf=str(best_pose),
        docker_image=gnina_image,
        output_dir=str(gnina_dir),
        output_file_name="minimized_pose.sdf",
        autobox_add=autobox_add,
        cpu_only=cpu_only,
    )

    # Clean up intermediate files if requested, but keep the best pose
    if cleanup_intermediates:
        # Only remove other rank files if they exist
        for rank_file in diffdock_dir.glob("rank[2-9].sdf"):
            try:
                rank_file.unlink()
            except OSError:
                pass

    return DockingResult(
        diffdock_output_dir=diffdock_dir,
        pose_file=best_pose,
        affinity=scores["Affinity"][0] if scores["Affinity"] else 0.0,
        cnn_score=scores["CNNscore"],
        cnn_affinity=scores["CNNaffinity"],
        cnn_variance=scores["CNNvariance"],
        rmsd=scores["RMSD"],
    )
