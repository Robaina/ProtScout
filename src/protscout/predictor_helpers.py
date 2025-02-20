import os
import re
import logging


def read_fasta_to_dict(fasta_file):
    """
    Read a FASTA file and return a dictionary of sequence IDs and their corresponding sequences.

    This function parses a FASTA-formatted file, extracting the sequence IDs (without the '>' character)
    as keys and the full sequences (with all whitespace removed) as values. It handles multi-line
    sequences and ignores empty lines.

    Args:
        fasta_file (str): Path to the FASTA file to be read.

    Returns:
        Dict[str, str]: A dictionary where keys are sequence IDs and values are the corresponding sequences.

    Raises:
        FileNotFoundError: If the specified FASTA file does not exist.
        ValueError: If the FASTA file is empty or improperly formatted.

    Example:
        >>> sequences = read_fasta_to_dict("example.fasta")
        >>> print(sequences)
        {'seq1': 'ATGCATGC', 'seq2': 'GATTACA'}
    """
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"The file {fasta_file} does not exist.")

    sequences = {}
    current_id = None
    current_sequence = []

    with open(fasta_file, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_sequence)
                    current_sequence = []
                current_id = line[1:].split()[
                    0
                ]  # Remove '>' and take the first word as ID
            else:
                current_sequence.append(line)

        # Add the last sequence
        if current_id:
            sequences[current_id] = "".join(current_sequence)

    if not sequences:
        raise ValueError("The FASTA file is empty or improperly formatted.")

    return sequences

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