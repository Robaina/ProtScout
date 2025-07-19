#!/bin/bash

# Default values
DOCKER_IMAGE="ghcr.io/new-atlantis-labs/gatsol:latest"
BATCH_SIZE=32
PARALLEL=8
FEATURE_BATCH_SIZE=4
FEATURE_TIMEOUT=600
DEVICE="cuda"
QUIET=false
SHM_SIZE=""  # Empty by default
GPUS_VALUE="all"  # Default to all GPUs

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --input DIR       Input directory containing .faa files (required)"
    echo "  -o, --output DIR      Output directory for results (required)"
    echo "  -m, --model DIR       GATSol model directory (required)"
    echo "  -e, --esm DIR         ESM models directory (required)"
    echo "  -s, --structures DIR  PDB structures directory (required)"
    echo "  -g, --gpu DEVICE      Device to use (default: cuda)"
    echo "  -b, --batch-size N    Batch size for processing (default: ${BATCH_SIZE})"
    echo "  -p, --parallel N      Number of parallel processes (default: ${PARALLEL})"
    echo "  -f, --feature-batch N Feature batch size (default: ${FEATURE_BATCH_SIZE})"
    echo "  -t, --timeout N       Feature timeout in seconds (default: ${FEATURE_TIMEOUT})"
    echo "  -d, --docker-image IMG Docker image to use (default: ${DOCKER_IMAGE})"
    echo "  -S, --shm-size SIZE    Shared memory size for containers (e.g., 8g)"
    echo "  -G, --gpus VALUE       GPUs to use for Docker (default: all)"
    echo "  -q, --quiet           Suppress output messages"
    echo "  -h, --help            Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--structures)
            PDB_DIR="$2"
            shift 2
            ;;
        -m|--model)
            GATSOL_DIR="$2"
            shift 2
            ;;
        -e|--esm)
            ESM_MODELS_DIR="$2"
            shift 2
            ;;
        -g|--gpu)
            DEVICE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        -f|--feature-batch)
            FEATURE_BATCH_SIZE="$2"
            shift 2
            ;;
        -t|--timeout)
            FEATURE_TIMEOUT="$2"
            shift 2
            ;;
        -d|--docker-image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        -S|--shm-size)
            SHM_SIZE="$2"
            shift 2
            ;;
        -G|--gpus)
            GPUS_VALUE="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate required parameters
if [ -z "${INPUT_DIR}" ] || [ -z "${OUTPUT_DIR}" ] || [ -z "${GATSOL_DIR}" ] || [ -z "${ESM_MODELS_DIR}" ] || [ -z "${PDB_DIR}" ]; then
    echo "Error: Missing required parameters"
    show_help
fi

# Validate input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory ${INPUT_DIR} does not exist"
    exit 1
fi

# Validate PDB directory exists
if [ ! -d "${PDB_DIR}" ]; then
    echo "Error: PDB directory ${PDB_DIR} does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Display configuration
echo "====== GATSol Analysis Configuration ======"
echo "  Input: ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  PDB base: ${PDB_DIR}"
echo "  GATSol model: ${GATSOL_DIR}"
echo "  ESM models: ${ESM_MODELS_DIR}"
echo "Using device: ${DEVICE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Parallel: ${PARALLEL}"
echo "Feature batch size: ${FEATURE_BATCH_SIZE}"
echo "Feature timeout: ${FEATURE_TIMEOUT} seconds"
echo "Docker image: ${DOCKER_IMAGE}"
echo "Docker GPUs: ${GPUS_VALUE}"
echo "Shared Memory Size: ${SHM_SIZE:-"default"}"
echo "Quiet mode: ${QUIET}"
echo "=========================================="

# Process each .faa file in the input directory
for input_file in "${INPUT_DIR}"/*.faa; do
    if [ -f "$input_file" ]; then
        # Get filename without extension
        filename=$(basename "$input_file")
        filename_no_ext="${filename%.*}"
        
        # Create output subdirectory for this file
        current_output_dir="${OUTPUT_DIR}/${filename_no_ext}"
        mkdir -p "${current_output_dir}"
        
        # Set the PDB directory for this file
        protein_pdb_dir="${PDB_DIR}/${filename_no_ext}"
        if [ ! -d "${protein_pdb_dir}" ]; then
            echo "Warning: PDB subdirectory ${protein_pdb_dir} does not exist."
            echo "Looking for PDB files directly in ${PDB_DIR}..."
            if [ -d "${PDB_DIR}" ] && [ "$(ls -A "${PDB_DIR}")" ]; then
                protein_pdb_dir="${PDB_DIR}"
            else
                echo "Error: No valid PDB directory found for ${filename_no_ext}. Skipping this file."
                continue
            fi
        fi
        
        if [ "$QUIET" = false ]; then
            echo "Processing file: $filename"
            echo "Using PDB directory: ${protein_pdb_dir}"
            echo "Output will be saved to: ${current_output_dir}"
        fi
        
        # Run GATSol command using docker run with updated paths and parameters
        if [ "$QUIET" = true ]; then
            docker run --rm --gpus "${GPUS_VALUE}" $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
                -v "${INPUT_DIR}:/app/input" \
                -v "${current_output_dir}:/app/results:rw" \
                -v "${GATSOL_DIR}:/app/weights" \
                -v "${ESM_MODELS_DIR}:/root/.cache/torch/hub/checkpoints" \
                -v "${protein_pdb_dir}:/app/pdb_files" \
                ${DOCKER_IMAGE} \
                --input "/app/input/${filename}" \
                --pdb-dir "/app/pdb_files" \
                --output-dir "/app/results" \
                --gatsol-weights-dir "/app/weights" \
                --device "${DEVICE}" \
                --batch-size "${BATCH_SIZE}" \
                --workers "${PARALLEL}" \
                --feature-batch-size "${FEATURE_BATCH_SIZE}" \
                --feature-timeout "${FEATURE_TIMEOUT}" > /dev/null 2>&1
        else
            docker run --rm --gpus "${GPUS_VALUE}" $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
                -v "${INPUT_DIR}:/app/input" \
                -v "${current_output_dir}:/app/results:rw" \
                -v "${GATSOL_DIR}:/app/weights" \
                -v "${ESM_MODELS_DIR}:/root/.cache/torch/hub/checkpoints" \
                -v "${protein_pdb_dir}:/app/pdb_files" \
                ${DOCKER_IMAGE} \
                --input "/app/input/${filename}" \
                --pdb-dir "/app/pdb_files" \
                --output-dir "/app/results" \
                --gatsol-weights-dir "/app/weights" \
                --device "${DEVICE}" \
                --batch-size "${BATCH_SIZE}" \
                --workers "${PARALLEL}" \
                --feature-batch-size "${FEATURE_BATCH_SIZE}" \
                --feature-timeout "${FEATURE_TIMEOUT}"
        fi
            
        if [ "$QUIET" = false ]; then
            echo "Completed processing: $filename"
        fi
        sleep 2
    fi
done

# Cleanup: Remove all containers created by this script
if [ "$QUIET" = false ]; then
    echo "Cleaning up containers..."
fi
docker ps -a | grep "${DOCKER_IMAGE}" | awk '{print $1}' | xargs -r docker rm > /dev/null 2>&1
if [ "$QUIET" = false ]; then
    echo "All files processed and containers cleaned up!"
fi