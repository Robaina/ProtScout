#!/bin/bash

# Default values
DOCKER_IMAGE="ghcr.io/robaina/protscout-tools-temstapro:latest"
GPU_ENABLED=true
QUIET=false
CONTAINER_NAME="temstapro"
MAX_CONTAINERS=4  # Default maximum parallel containers
SHM_SIZE=""  # Empty by default
GPUS_VALUE="all"  # Default to all GPUs

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --input DIR       Input directory containing FASTA files (required)"
    echo "  -o, --output DIR      Output directory for results (required)"
    echo "  -m, --model DIR       Model directory (required)"
    echo "  -g, --gpu BOOL        Enable GPU usage (default: true)"
    echo "  -q, --quiet           Suppress output messages"
    echo "  -p, --parallel N      Maximum number of parallel containers (default: ${MAX_CONTAINERS})"
    echo "  -d, --docker-image IMG Docker image to use (default: ${DOCKER_IMAGE})"
    echo "  -S, --shm-size SIZE    Shared memory size for containers (e.g., 8g)"
    echo "  -G, --gpus VALUE       GPUs to use for Docker (default: all)"
    echo "  -h, --help            Show this help message"
    exit 1
}

# Function to count running TemStaPro containers
count_containers() {
    docker ps --format '{{.Names}}' | grep -c "^${CONTAINER_NAME}_" || true
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
        -m|--model)
            MODEL_DIR="$2"
            shift 2
            ;;
        -g|--gpu)
            if [[ "$2" == "false" || "$2" == "0" ]]; then
                GPU_ENABLED=false
            else
                GPU_ENABLED=true
            fi
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -p|--parallel)
            MAX_CONTAINERS="$2"
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
if [ -z "${INPUT_DIR}" ] || [ -z "${OUTPUT_DIR}" ] || [ -z "${MODEL_DIR}" ]; then
    echo "Error: Missing required parameters"
    show_help
fi

# Validate input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory ${INPUT_DIR} does not exist"
    exit 1
fi

# Set default model directory if not provided
if [ -z "${MODEL_DIR}" ]; then
    echo "Error: Model directory is required for TemStaPro"
    exit 1
fi

# Validate model directory exists
if [ ! -d "${MODEL_DIR}" ]; then
    echo "Error: Model directory ${MODEL_DIR} does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Display configuration
echo "====== TemStaPro Analysis Configuration ======"
echo "  Input: ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Model: ${MODEL_DIR}"
echo "GPU enabled: ${GPU_ENABLED}"
echo "Quiet mode: ${QUIET}"
echo "Docker image: ${DOCKER_IMAGE}"
echo "Maximum parallel containers: ${MAX_CONTAINERS}"
echo "Docker GPUs: ${GPUS_VALUE}"
echo "Shared Memory Size: ${SHM_SIZE:-"default"}"
echo "=========================================="

# Find all FASTA files in the input directory - FIXED
FASTA_FILES=()
mapfile -t FASTA_FILES < <(find "${INPUT_DIR}" -name "*.faa" -o -name "*.fasta" -o -name "*.fa")

if [ ${#FASTA_FILES[@]} -eq 0 ]; then
    echo "Error: No FASTA files found in ${INPUT_DIR}"
    echo "DEBUG: Directory contents:"
    ls -la "${INPUT_DIR}"
    echo "DEBUG: Find command output:"
    find "${INPUT_DIR}" -name "*.faa" -o -name "*.fasta" -o -name "*.fa"
    exit 1
fi

TOTAL_JOBS=${#FASTA_FILES[@]}
CURRENT_JOB=0

if [ "$QUIET" = false ]; then
    echo "Found ${TOTAL_JOBS} FASTA files to process..."
fi

# Create arrays to store job information
declare -a JOB_IDS
declare -a OUTPUT_DIRS
declare -a BASENAMES_NO_EXT

# Process each FASTA file
for FASTA_FILE in "${FASTA_FILES[@]}"; do
    # Extract basename without extension for output directory
    BASENAME=$(basename "${FASTA_FILE}")
    BASENAME_NO_EXT="${BASENAME%.*}"
    
    if [ "$QUIET" = false ]; then
        echo "Processing ${BASENAME}..."
    fi
    
    # Create file-specific output directory
    FILE_OUTPUT_DIR="${OUTPUT_DIR}/${BASENAME_NO_EXT}"
    mkdir -p "${FILE_OUTPUT_DIR}"
    
    # Process with TemStaPro
    CURRENT_JOB=$((CURRENT_JOB + 1))
    # Generate a unique container name for this job
    CONTAINER_ID="${CONTAINER_NAME}_${BASENAME_NO_EXT}_$(date +%s%N | md5sum | head -c 8)"
    
    # Wait until we have room to run another container
    while [ "$(count_containers)" -ge "$MAX_CONTAINERS" ]; do
        if [ "$QUIET" = false ]; then
            echo "Currently at max containers ($MAX_CONTAINERS). Waiting..."
        fi
        sleep 5
    done
    
    if [ "$QUIET" = false ]; then
        echo "[${CURRENT_JOB}/${TOTAL_JOBS}] Running TemStaPro for ${BASENAME} (Container: ${CONTAINER_ID})..."
    fi
    
    # Build docker command parts
    DOCKER_GPU_FLAG=""
    if [ "$GPU_ENABLED" = true ]; then
        DOCKER_GPU_FLAG="--gpus ${GPUS_VALUE}"
    fi
    
    DOCKER_SHM_FLAG=""
    if [ -n "${SHM_SIZE}" ]; then
        DOCKER_SHM_FLAG="--shm-size=${SHM_SIZE}"
    fi
    
    # Run TemStaPro container in background - FIXED PATHS
    if [ "$QUIET" = true ]; then
        docker run --name "${CONTAINER_ID}" --rm ${DOCKER_GPU_FLAG} ${DOCKER_SHM_FLAG} \
            -v "${MODEL_DIR}":/models:ro \
            -v "${INPUT_DIR}":/test_data:ro \
            -v "${FILE_OUTPUT_DIR}":/outputs:rw \
            ${DOCKER_IMAGE} \
            /models \
            -f "/test_data/${BASENAME}" \
            -p "/outputs" \
            --mean-output "/outputs/${BASENAME_NO_EXT}_predictions_mean.tsv" > /dev/null 2>&1 &
    else
        docker run --name "${CONTAINER_ID}" --rm ${DOCKER_GPU_FLAG} ${DOCKER_SHM_FLAG} \
            -v "${MODEL_DIR}":/models:ro \
            -v "${INPUT_DIR}":/test_data:ro \
            -v "${FILE_OUTPUT_DIR}":/outputs:rw \
            ${DOCKER_IMAGE} \
            /models \
            -f "/test_data/${BASENAME}" \
            -p "/outputs" \
            --mean-output "/outputs/${BASENAME_NO_EXT}_predictions_mean.tsv" &
    fi
    
    # Add container ID and output directory to the job lists
    JOB_IDS+=("$CONTAINER_ID")
    OUTPUT_DIRS+=("${FILE_OUTPUT_DIR}")
    BASENAMES_NO_EXT+=("${BASENAME_NO_EXT}")
    
    # Sleep a bit to let Docker register the container
    sleep 1
    
    if [ "$QUIET" = false ]; then
        echo "Started Docker container for ${BASENAME} (running: $(count_containers)/${MAX_CONTAINERS})"
    fi
done

# Wait for all containers to finish
if [ "$QUIET" = false ]; then
    echo "All jobs started, waiting for containers to finish..."
fi

while [ "$(count_containers)" -gt 0 ]; do
    RUNNING=$(count_containers)
    if [ "$QUIET" = false ]; then
        echo "Waiting for ${RUNNING} container(s) to finish..."
    fi
    sleep 10
done

# Check results by examining output files instead of container exit codes
if [ "$QUIET" = false ]; then
    echo "All containers have finished. Checking results..."
fi

SUCCESS=0
FAILURE=0

# Loop through each job and check if the output files exist - FIXED PATH
for i in "${!JOB_IDS[@]}"; do
    CONTAINER_ID="${JOB_IDS[$i]}"
    OUTPUT_DIR_PATH="${OUTPUT_DIRS[$i]}"
    BASENAME_NO_EXT="${BASENAMES_NO_EXT[$i]}"

    # Look for the mean predictions output file - CORRECTED PATH
    MEAN_OUTPUT_FILE="${OUTPUT_DIR_PATH}/${BASENAME_NO_EXT}_predictions_mean.tsv"
    
    if [ -f "${MEAN_OUTPUT_FILE}" ]; then
        SUCCESS=$((SUCCESS + 1))
        if [ "$QUIET" = false ]; then
            echo "Job ${CONTAINER_ID} completed successfully: $(basename "${MEAN_OUTPUT_FILE}")"
        fi
    else
        # Missing output file, job failed
        FAILURE=$((FAILURE + 1))
        if [ "$QUIET" = false ]; then
            echo "Error: Job ${CONTAINER_ID} failed: missing output file ${MEAN_OUTPUT_FILE}"
        fi
    fi
done

# Print summary
if [ "$QUIET" = false ]; then
    echo "==== TemStaPro Analysis Complete ===="
    echo "Total jobs: ${TOTAL_JOBS}"
    echo "Successful: ${SUCCESS}"
    echo "Failed: ${FAILURE}"
    echo "Results are available in: ${OUTPUT_DIR}"
    
    if [ ${FAILURE} -gt 0 ]; then
        echo "Warning: ${FAILURE} job(s) failed"
    else
        echo "All jobs completed successfully!"
    fi
fi

if [ ${FAILURE} -gt 0 ]; then
    exit 1
fi