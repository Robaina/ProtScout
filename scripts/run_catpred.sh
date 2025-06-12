#!/bin/bash

# Default values
DOCKER_IMAGE="ghcr.io/new-atlantis-labs/catpred:latest"
PROCESSED_FILE="processed_files.txt"
GPU_ENABLED=true
QUIET=false
CONTAINER_NAME="catpred"
MAX_CONTAINERS=4  # Default maximum parallel containers
SHM_SIZE=""  # Empty by default
GPUS_VALUE="all"  # Default to all GPUs

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --input DIR       Input directory containing catpred input files (required)"
    echo "  -o, --output DIR      Output directory for results (required)"
    echo "  -m, --model DIR       Model/weights directory (required)"
    echo "  -f, --file FILE       Input processed files list (default: ${PROCESSED_FILE})"
    echo "  -g, --gpu BOOL        Enable GPU usage (default: true)"
    echo "  -q, --quiet           Suppress output messages"
    echo "  -p, --parallel N      Maximum number of parallel containers (default: ${MAX_CONTAINERS})"
    echo "  -d, --docker-image IMG Docker image to use (default: ${DOCKER_IMAGE})"
    echo "  -S, --shm-size SIZE    Shared memory size for containers (e.g., 8g)"
    echo "  -G, --gpus VALUE       GPUs to use for Docker (default: all)"
    echo "  -h, --help            Show this help message"
    exit 1
}

# Function to count running CatPred containers
count_containers() {
    docker ps --format '{{.Names}}' | grep -c "^${CONTAINER_NAME}_"
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
            WEIGHTS_DIR="$2"
            shift 2
            ;;
        -f|--file)
            PROCESSED_FILE="$2"
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
if [ -z "${INPUT_DIR}" ] || [ -z "${OUTPUT_DIR}" ] || [ -z "${WEIGHTS_DIR}" ]; then
    echo "Error: Missing required parameters"
    show_help
fi

# Validate input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory ${INPUT_DIR} does not exist"
    exit 1
fi

# Validate processed files list exists
if [ ! -f "${INPUT_DIR}/${PROCESSED_FILE}" ]; then
    echo "Error: Processed files list ${INPUT_DIR}/${PROCESSED_FILE} does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Set GPU and quiet flags for Docker command
USE_GPU=""
if [ "$GPU_ENABLED" = true ]; then
    USE_GPU="--use_gpu"
fi

# Always pass the quiet flag to the Docker container
DOCKER_QUIET=""

# Display configuration
echo "====== CatPred Analysis Configuration ======"
echo "  Input: ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Model/weights: ${WEIGHTS_DIR}"
echo "  Processed files list: ${PROCESSED_FILE}"
echo "GPU enabled: ${GPU_ENABLED}"
echo "Quiet mode: ${QUIET}"
echo "Docker image: ${DOCKER_IMAGE}"
echo "Maximum parallel containers: ${MAX_CONTAINERS}"
echo "Docker GPUs: ${GPUS_VALUE}"
echo "Shared Memory Size: ${SHM_SIZE:-"default"}"
echo "=========================================="

# Create arrays to store job information
declare -a JOB_IDS
declare -a OUTPUT_DIRS

# Count total runs for progress tracking
TOTAL_JOBS=0
CURRENT_JOB=0

# First, count the number of jobs
while IFS=$'\t' read -r protein_group_id input_file; do
    # Two jobs per protein group (kcat and km)
    TOTAL_JOBS=$((TOTAL_JOBS + 2))
done < "${INPUT_DIR}/${PROCESSED_FILE}"

if [ "$QUIET" = false ]; then
    echo "Starting CatPred analysis on ${TOTAL_JOBS} total jobs..."
fi

# Process each protein group
while IFS=$'\t' read -r protein_group_id input_file; do
    if [ "$QUIET" = false ]; then
        echo "Processing ${protein_group_id}..."
    fi
    
    # Create protein-group-specific output directories for kcat and km
    PROTEIN_GROUP_OUTPUT_DIR="${OUTPUT_DIR}/${protein_group_id}"
    KCAT_OUTPUT_DIR="${PROTEIN_GROUP_OUTPUT_DIR}/kcat"
    KM_OUTPUT_DIR="${PROTEIN_GROUP_OUTPUT_DIR}/km"
    mkdir -p "${KCAT_OUTPUT_DIR}"
    mkdir -p "${KM_OUTPUT_DIR}"
    
    # Process kcat parameter
    CURRENT_JOB=$((CURRENT_JOB + 1))
    # Generate a unique container name for this job
    KCAT_CONTAINER_ID="${CONTAINER_NAME}_${protein_group_id}_kcat_$(date +%s%N | md5sum | head -c 8)"
    
    # Wait until we have room to run another container
    while [ "$(count_containers)" -ge "$MAX_CONTAINERS" ]; do
        if [ "$QUIET" = false ]; then
            echo "Currently at max containers ($MAX_CONTAINERS). Waiting..."
        fi
        sleep 5
    done
    
    if [ "$QUIET" = false ]; then
        echo "[${CURRENT_JOB}/${TOTAL_JOBS}] Running CatPred for ${protein_group_id} - kcat parameter (Container: ${KCAT_CONTAINER_ID})..."
    fi
    
    # Run CatPred container for kcat parameter in background
    if [ "$QUIET" = true ]; then
        docker run --name "${KCAT_CONTAINER_ID}" --rm $([ "$GPU_ENABLED" = true ] && echo "--gpus \"${GPUS_VALUE}\"") $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
            -v "${INPUT_DIR}":/input \
            -v "${KCAT_OUTPUT_DIR}":/output:rw \
            -v "${WEIGHTS_DIR}":/weights \
            ${DOCKER_IMAGE} \
            --parameter kcat \
            --input_file "/input/${protein_group_id}/input.csv" \
            --weights_dir /weights \
            $USE_GPU \
            $DOCKER_QUIET > /dev/null 2>&1 &
    else
        docker run --name "${KCAT_CONTAINER_ID}" --rm $([ "$GPU_ENABLED" = true ] && echo "--gpus \"${GPUS_VALUE}\"") $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
            -v "${INPUT_DIR}":/input \
            -v "${KCAT_OUTPUT_DIR}":/output:rw \
            -v "${WEIGHTS_DIR}":/weights \
            ${DOCKER_IMAGE} \
            --parameter kcat \
            --input_file "/input/${protein_group_id}/input.csv" \
            --weights_dir /weights \
            $USE_GPU \
            $DOCKER_QUIET &
    fi
    
    # Add container ID and output directory to the job lists
    JOB_IDS+=("$KCAT_CONTAINER_ID")
    OUTPUT_DIRS+=("${KCAT_OUTPUT_DIR}")
    
    # Process km parameter
    CURRENT_JOB=$((CURRENT_JOB + 1))
    # Generate a unique container name for this job
    KM_CONTAINER_ID="${CONTAINER_NAME}_${protein_group_id}_km_$(date +%s%N | md5sum | head -c 8)"
    
    # Wait until we have room to run another container
    while [ "$(count_containers)" -ge "$MAX_CONTAINERS" ]; do
        if [ "$QUIET" = false ]; then
            echo "Currently at max containers ($MAX_CONTAINERS). Waiting..."
        fi
        sleep 5
    done
    
    if [ "$QUIET" = false ]; then
        echo "[${CURRENT_JOB}/${TOTAL_JOBS}] Running CatPred for ${protein_group_id} - km parameter (Container: ${KM_CONTAINER_ID})..."
    fi
    
    # Run CatPred container for km parameter in background
    if [ "$QUIET" = true ]; then
        docker run --name "${KM_CONTAINER_ID}" --rm $([ "$GPU_ENABLED" = true ] && echo "--gpus \"${GPUS_VALUE}\"") $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
            -v "${INPUT_DIR}":/input \
            -v "${KM_OUTPUT_DIR}":/output:rw \
            -v "${WEIGHTS_DIR}":/weights \
            ${DOCKER_IMAGE} \
            --parameter km \
            --input_file "/input/${protein_group_id}/input.csv" \
            --weights_dir /weights \
            $USE_GPU \
            $DOCKER_QUIET > /dev/null 2>&1 &
    else
        docker run --name "${KM_CONTAINER_ID}" --rm $([ "$GPU_ENABLED" = true ] && echo "--gpus \"${GPUS_VALUE}\"") $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
            -v "${INPUT_DIR}":/input \
            -v "${KM_OUTPUT_DIR}":/output:rw \
            -v "${WEIGHTS_DIR}":/weights \
            ${DOCKER_IMAGE} \
            --parameter km \
            --input_file "/input/${protein_group_id}/input.csv" \
            --weights_dir /weights \
            $USE_GPU \
            $DOCKER_QUIET &
    fi
    
    # Add container ID and output directory to the job lists
    JOB_IDS+=("$KM_CONTAINER_ID")
    OUTPUT_DIRS+=("${KM_OUTPUT_DIR}")
    
    # Sleep a bit to let Docker register the container
    sleep 1
    
    if [ "$QUIET" = false ]; then
        echo "Started Docker containers for ${protein_group_id} (running: $(count_containers)/${MAX_CONTAINERS})"
    fi
    
done < "${INPUT_DIR}/${PROCESSED_FILE}"

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

# Loop through each job and check if the output file exists
for i in "${!JOB_IDS[@]}"; do
    CONTAINER_ID="${JOB_IDS[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
    OUTPUT_FILE="${OUTPUT_DIR}/final_predictions_input.csv"
    
    if [ -f "$OUTPUT_FILE" ]; then
        # File exists, job was successful
        SUCCESS=$((SUCCESS + 1))
        if [ "$QUIET" = false ]; then
            echo "Job ${CONTAINER_ID} completed successfully: ${OUTPUT_FILE} exists"
        fi
    else
        # File doesn't exist, job failed
        FAILURE=$((FAILURE + 1))
        if [ "$QUIET" = false ]; then
            echo "Error: Job ${CONTAINER_ID} failed: ${OUTPUT_FILE} not found"
        fi
    fi
done

# Print summary
if [ "$QUIET" = false ]; then
    echo "==== CatPred Analysis Complete ===="
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