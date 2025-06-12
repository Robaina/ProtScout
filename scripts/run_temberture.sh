#!/bin/bash

# Default values
DOCKER_IMAGE="ghcr.io/new-atlantis-labs/temberture"
GPU_ENABLED=true
QUIET=false
CONTAINER_NAME="temberture"
MAX_CONTAINERS=4  # Default maximum parallel containers
SHM_SIZE=""  # Empty by default
GPUS_VALUE="all"  # Default to all GPUs

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --input DIR       Input directory containing .faa files (required)"
    echo "  -o, --output DIR      Output directory for results (required)"
    echo "  -g, --gpu BOOL        Enable GPU usage (default: true)"
    echo "  -d, --docker-image IMG Docker image to use (default: ${DOCKER_IMAGE})"
    echo "  -p, --parallel N      Maximum number of parallel containers (default: ${MAX_CONTAINERS})"
    echo "  -S, --shm-size SIZE    Shared memory size for containers (e.g., 8g)"
    echo "  -G, --gpus VALUE       GPUs to use for Docker (default: all)"
    echo "  -q, --quiet           Suppress output messages"
    echo "  -h, --help            Show this help message"
    exit 1
}

# Function to count running Temberture containers
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
        -g|--gpu)
            if [[ "$2" == "false" || "$2" == "0" ]]; then
                GPU_ENABLED=false
            else
                GPU_ENABLED=true
            fi
            shift 2
            ;;
        -d|--docker-image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        -p|--parallel)
            MAX_CONTAINERS="$2"
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
if [ -z "${INPUT_DIR}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "Error: Missing required parameters"
    show_help
fi

# Validate input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory ${INPUT_DIR} does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Display configuration
echo "====== TemBERTure Analysis Configuration ======"
echo "  Input: ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "GPU enabled: ${GPU_ENABLED}"
echo "Docker image: ${DOCKER_IMAGE}"
echo "Maximum parallel containers: ${MAX_CONTAINERS}"
echo "Docker GPUs: ${GPUS_VALUE}"
echo "Shared Memory Size: ${SHM_SIZE:-"default"}"
echo "Quiet mode: ${QUIET}"
echo "=========================================="

# Create an array to store all job IDs
declare -a JOB_IDS

# Count total runs for progress tracking
TOTAL_JOBS=0
CURRENT_JOB=0

# First, count the number of jobs
for input_file in "${INPUT_DIR}"/*.faa; do
    if [ -f "$input_file" ]; then
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    fi
done

if [ "$QUIET" = false ]; then
    echo "Starting Temberture analysis on ${TOTAL_JOBS} total files..."
fi

# Process each .faa file in the input directory
for input_file in "${INPUT_DIR}"/*.faa; do
    if [ -f "$input_file" ]; then
        # Get filename without extension
        filename=$(basename "$input_file")
        filename_no_ext="${filename%.*}"
        
        # Create output subdirectory for this file
        current_output_dir="${OUTPUT_DIR}/${filename_no_ext}"
        mkdir -p "${current_output_dir}"
        
        CURRENT_JOB=$((CURRENT_JOB + 1))
        # Generate a unique container name for this job
        CONTAINER_ID="${CONTAINER_NAME}_${filename_no_ext}_$(date +%s%N | md5sum | head -c 8)"
        
        # Wait until we have room to run another container
        while [ "$(count_containers)" -ge "$MAX_CONTAINERS" ]; do
            if [ "$QUIET" = false ]; then
                echo "Currently at max containers ($MAX_CONTAINERS). Waiting..."
            fi
            sleep 5
        done
        
        if [ "$QUIET" = false ]; then
            echo "[${CURRENT_JOB}/${TOTAL_JOBS}] Processing file: $filename (Container: ${CONTAINER_ID})"
            echo "Output will be saved to: ${current_output_dir}"
        fi
        
        # Run temberture command using docker run in background
        if [ "$GPU_ENABLED" = true ]; then
            if [ "$QUIET" = true ]; then
                docker run --name "${CONTAINER_ID}" --rm --gpus "${GPUS_VALUE}" $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
                    -v "${INPUT_DIR}:/input" \
                    -v "${current_output_dir}:/output:rw" \
                    ${DOCKER_IMAGE} \
                    --tm "/input/${filename}" "/output/${filename_no_ext}_results.tsv" > /dev/null 2>&1 &
            else
                docker run --name "${CONTAINER_ID}" --rm --gpus "${GPUS_VALUE}" $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
                    -v "${INPUT_DIR}:/input" \
                    -v "${current_output_dir}:/output:rw" \
                    ${DOCKER_IMAGE} \
                    --tm "/input/${filename}" "/output/${filename_no_ext}_results.tsv" &
            fi
        else
            if [ "$QUIET" = true ]; then
                docker run --name "${CONTAINER_ID}" --rm $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
                    -v "${INPUT_DIR}:/input" \
                    -v "${current_output_dir}:/output:rw" \
                    ${DOCKER_IMAGE} \
                    --tm "/input/${filename}" "/output/${filename_no_ext}_results.tsv" > /dev/null 2>&1 &
            else
                docker run --name "${CONTAINER_ID}" --rm $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
                    -v "${INPUT_DIR}:/input" \
                    -v "${current_output_dir}:/output:rw" \
                    ${DOCKER_IMAGE} \
                    --tm "/input/${filename}" "/output/${filename_no_ext}_results.tsv" &
            fi
        fi
        
        # Add container ID to the job list
        JOB_IDS+=("$CONTAINER_ID")
        
        # Sleep a bit to let Docker register the container
        sleep 1
        
        if [ "$QUIET" = false ]; then
            echo "Started Docker container ${CONTAINER_ID} (running: $(count_containers)/${MAX_CONTAINERS})"
        fi
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

# Check exit status of all containers
if [ "$QUIET" = false ]; then
    echo "All containers have finished. Checking results..."
fi

SUCCESS=0
FAILURE=0

for CONTAINER_ID in "${JOB_IDS[@]}"; do
    EXIT_CODE=$(docker inspect "${CONTAINER_ID}" --format='{{.State.ExitCode}}' 2>/dev/null || echo "removed")
    
    if [ "$EXIT_CODE" = "0" ]; then
        SUCCESS=$((SUCCESS + 1))
    elif [ "$EXIT_CODE" = "removed" ]; then
        if [ "$QUIET" = false ]; then
            echo "Warning: Container ${CONTAINER_ID} was removed before we could check its status"
        fi
    else
        if [ "$QUIET" = false ]; then
            echo "Error: Container ${CONTAINER_ID} failed with exit code ${EXIT_CODE}"
        fi
        FAILURE=$((FAILURE + 1))
    fi
    
done

# Print summary
if [ "$QUIET" = false ]; then
    echo "==== Temberture Analysis Complete ===="
    echo "Total files processed: ${TOTAL_JOBS}"
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