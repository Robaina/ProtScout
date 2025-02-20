import os
import re
import logging
from .container_management import DockerContainerPool


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
    Calculates various metrics for the minimized affinity of a ligand to a receptor after local adjustments,
    using a persistent Docker container.

    Parameters:
        pdb_file_no_hetatms (str): The file path of the receptor PDB file without HETATM entries.
        ligand_sdf (str): The file path of the ligand in SDF format.
        docker_image (str): Docker image tag for gnina, defaults to "gnina/gnina:latest".
        output_dir (str, optional): The directory path where output files will be stored. If specified,
                                    this directory will be mounted to the Docker container.
        output_file_name (str, optional): The name of the output file to save the results.
                                         Defaults to "wdocking_minimized.sdf.gz".
        autobox_add (int, optional): The number of angstroms to add to the autobox size. Defaults to 2.
        cpu_only (bool, optional): If True, run the prediction on CPU only.
        suppress_warnings (bool, optional): If True, suppress warnings from the gnina command.

    Returns:
        dict: A dictionary containing the following keys and values:
            - 'Affinity' (list of floats, kcal/mol): The binding energies of the ligand to the receptor.
            - 'RMSD' (float, Angstroms): Root mean square deviation of the docking pose from a reference pose.
            - 'CNNscore' (float, unitless): A neural network-based score indicating the likelihood of a correct binding pose.
            - 'CNNaffinity' (float, kcal/mol): Predicted binding affinity based on a convolutional neural network.
            - 'CNNvariance' (float, unitless): Variance of the CNN prediction, indicating prediction uncertainty.
    """

    logger = logging.getLogger(__name__)

    # Resolve absolute paths and parent directories of the files
    pdb_path = os.path.abspath(pdb_file_no_hetatms)
    pdb_dir = os.path.dirname(pdb_path)
    ligand_path = os.path.abspath(ligand_sdf)
    ligand_dir = os.path.dirname(ligand_path)

    # Prepare volume mounts
    volume_mounts = [(pdb_dir, "/pdb"), (ligand_dir, "/ligand")]

    if output_dir:
        output_path = os.path.abspath(output_dir)
        volume_mounts.append((output_path, "/output"))
        output_file = f"/output/{output_file_name}"
        output_cmd = f"-o {output_file}"
    else:
        output_cmd = ""

    # Get or create container
    container_pool = DockerContainerPool.get_instance()
    container_id = container_pool.get_container(
        image_name=docker_image, volume_mounts=volume_mounts, gpu=not cpu_only
    )

    # Prepare command to execute in container
    command = [
        "gnina",
        "--local_only",
        "--minimize",
        "-r",
        f"/pdb/{os.path.basename(pdb_file_no_hetatms)}",
        "-l",
        f"/ligand/{os.path.basename(ligand_sdf)}",
        "--autobox_ligand",
        f"/ligand/{os.path.basename(ligand_sdf)}",
        "--autobox_add",
        str(autobox_add),
    ]

    if output_cmd:
        command.extend(output_cmd.split())

    # Execute command in the container
    try:
        result = container_pool.execute_command(
            container_id=container_id, command=command, capture_output=True
        )
        minimized_stdout = result.stdout
    except Exception as e:
        logger.error(f"Error running gnina in container: {e}")
        raise RuntimeError(f"Failed to run gnina command: {e}")

    # Extract metrics using regex
    affinities = re.findall(r"Affinity:\s*([\-\.\d]+\s*[\-\.\d]+)", minimized_stdout)
    affinity_values = (
        [float(val) for val in affinities[0].split()] if affinities else []
    )
    rmsd = float(re.findall(r"RMSD:\s*([\-\.\d]+)", minimized_stdout)[0])
    cnnscore = float(re.findall(r"CNNscore:\s*([\-\.\d]+)", minimized_stdout)[0])
    cnnaffinity = float(re.findall(r"CNNaffinity:\s*([\-\.\d]+)", minimized_stdout)[0])
    cnnvariance = float(re.findall(r"CNNvariance:\s*([\-\.\d]+)", minimized_stdout)[0])

    return {
        "Affinity": affinity_values,
        "RMSD": rmsd,
        "CNNscore": cnnscore,
        "CNNaffinity": cnnaffinity,
        "CNNvariance": cnnvariance,
    }
