#!/bin/bash

# Default values
CONTAINER_NAME="geopoc"
GPU_ID="0"
TASKS=("temp" "pH" "salt")
MAX_CONTAINERS=4
DOCKER_IMAGE="ghcr.io/new-atlantis-labs/geopoc"
QUIET=false
SHM_SIZE=""  # Empty by default
GPUS_VALUE="all"  # Default to all GPUs

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Run GeoPoc analysis on FASTA files with flexible paths and tasks."
    echo
    echo "Options:"
    echo "  -i, --input DIR          Path to input sequences directory (required)"
    echo "  -o, --output DIR         Path to output directory (required)"
    echo "  -m, --model DIR          Path to GeoPoc models directory (required)"
    echo "  -e, --esm DIR            Path to ESM models directory (required)"
    echo "  -s, --structures DIR     Path to structures directory (required)"
    echo "  -b, --embeddings DIR     Path to embeddings directory (required)"
    echo "  -t, --tasks LIST         Comma-separated list of tasks to run (default: temp,pH,salt)"
    echo "  -g, --gpu ID             GPU ID to use (default: 0)"
    echo "  -p, --parallel N         Maximum number of parallel containers (default: 4)"
    echo "  -d, --docker-image IMG   Docker image to use (default: $DOCKER_IMAGE)"
    echo "  -S, --shm-size SIZE      Shared memory size for containers (e.g., 8g)"
    echo "  -G, --gpus VALUE         GPUs to use for Docker (default: all)"
    echo "  -h, --help               Display this help message and exit"
    echo
    echo "Example:"
    echo "  $0 -i /data/sequences -o /data/outputs -m /models/geopoc -e /models/esm -s /data/structures -b /data/embeddings -p 8"
    echo
}

# Function to count running GeoPoc containers
count_containers() {
    docker ps --format '{{.Names}}' | grep -c "^${CONTAINER_NAME}_"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input)
            SEQUENCES_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            GEOPOC_MODEL_DIR="$2"
            shift 2
            ;;
        -e|--esm)
            ESM_MODEL_DIR="$2"
            shift 2
            ;;
        -s|--structures)
            STRUCTURES_DIR="$2"
            shift 2
            ;;
        -b|--embeddings)
            EMBEDDINGS_DIR="$2"
            shift 2
            ;;
        -t|--tasks)
            IFS=',' read -r -a TASKS <<< "$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
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
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$SEQUENCES_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$GEOPOC_MODEL_DIR" ] || [ -z "$ESM_MODEL_DIR" ] || [ -z "$STRUCTURES_DIR" ] || [ -z "$EMBEDDINGS_DIR" ]; then
    echo "Error: Missing required parameters"
    show_help
    exit 1
fi

# Display configuration
echo "====== GeoPoc Analysis Configuration ======"
echo "Input Sequences Dir:    ${SEQUENCES_DIR}"
echo "Output Directory:       ${OUTPUT_DIR}"
echo "GeoPoc Models Dir:      ${GEOPOC_MODEL_DIR}"
echo "ESM Models Dir:         ${ESM_MODEL_DIR}"
echo "Structures Directory:   ${STRUCTURES_DIR}"
echo "Embeddings Directory:   ${EMBEDDINGS_DIR}"
echo "Tasks to Run:           ${TASKS[*]}"
echo "GPU ID:                 ${GPU_ID}"
echo "Max Parallel Jobs:      ${MAX_CONTAINERS}"
echo "Docker Image:           ${DOCKER_IMAGE}"
echo "Docker GPUs:            ${GPUS_VALUE}"
echo "Shared Memory Size:     ${SHM_SIZE:-"default"}"
echo "=========================================="

# Ensure parent output directory exists
mkdir -p ${OUTPUT_DIR}

# Find the FASTA file(s) in sequences directory
mapfile -t FASTA_FILES < <(find ${SEQUENCES_DIR} -name "*.fasta" -o -name "*.faa")
if [ ${#FASTA_FILES[@]} -eq 0 ]; then
    echo "No FASTA files found in ${SEQUENCES_DIR}"
    exit 1
fi

# Count total runs for progress tracking
TOTAL_JOBS=$((${#FASTA_FILES[@]} * ${#TASKS[@]}))
CURRENT_JOB=0

# Create an array to store all job IDs
declare -a JOB_IDS

# Process each FASTA file
if [ "$QUIET" = false ]; then
    echo "Starting GeoPoc analysis on ${TOTAL_JOBS} total jobs..."
fi

for FASTA_FILE in "${FASTA_FILES[@]}"; do
    FILENAME=$(basename "$FASTA_FILE")
    # Extract base name without extension for subdirectory
    BASE_NAME="${FILENAME%.*}"
    
    if [ "$QUIET" = false ]; then
        echo "Processing $FILENAME (Base: $BASE_NAME)..."
    fi
    
    # Create file-specific output subdirectory
    FILE_OUTPUT_DIR="${OUTPUT_DIR}/${BASE_NAME}"
    if [ "$QUIET" = false ]; then
        echo "Creating output directory: $FILE_OUTPUT_DIR"
    fi
    mkdir -p "$FILE_OUTPUT_DIR"
    
    # Define subdirectory paths for this specific FASTA file
    FASTA_STRUCTURES_DIR="${STRUCTURES_DIR}/${BASE_NAME}"
    FASTA_EMBEDDINGS_DIR="${EMBEDDINGS_DIR}/${BASE_NAME}"
    
    # Check if the subdirectories exist
    if [ ! -d "$FASTA_STRUCTURES_DIR" ]; then
        if [ "$QUIET" = false ]; then
            echo "Creating structure directory: $FASTA_STRUCTURES_DIR"
        fi
        mkdir -p "$FASTA_STRUCTURES_DIR"
    fi
    
    if [ ! -d "$FASTA_EMBEDDINGS_DIR" ]; then
        if [ "$QUIET" = false ]; then
            echo "Creating embedding directory: $FASTA_EMBEDDINGS_DIR"
        fi
        mkdir -p "$FASTA_EMBEDDINGS_DIR"
    fi
    
    for TASK in "${TASKS[@]}"; do
        CURRENT_JOB=$((CURRENT_JOB + 1))
        
        # Generate a unique container name for this job
        CONTAINER_ID="${CONTAINER_NAME}_${BASE_NAME}_${TASK}_$(date +%s%N | md5sum | head -c 8)"
        
        # Wait until we have room to run another container
        while [ "$(count_containers)" -ge "$MAX_CONTAINERS" ]; do
            if [ "$QUIET" = false ]; then
                echo "Currently at max containers ($MAX_CONTAINERS). Waiting..."
            fi
            sleep 5
        done
        
        if [ "$QUIET" = false ]; then
            echo "[$CURRENT_JOB/$TOTAL_JOBS] Starting task: $TASK for $FILENAME (Container: $CONTAINER_ID)"
        fi
        
        # Run the container with the current task in the background
        if [ "$QUIET" = true ]; then
            docker run --name "${CONTAINER_ID}" --gpus "${GPUS_VALUE}" $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
              -v "${SEQUENCES_DIR}":/input \
              -v "${FILE_OUTPUT_DIR}":/output:rw \
              -v "${FASTA_STRUCTURES_DIR}":/pdb_base \
              -v "${FASTA_EMBEDDINGS_DIR}":/embeddings_base \
              -v "${GEOPOC_MODEL_DIR}":/models \
              -v "${ESM_MODEL_DIR}":/root/.cache/torch/hub/checkpoints \
              "${DOCKER_IMAGE}" \
              -i "/input/${FILENAME}" \
              -o "/output/" \
              --model_path /models/ \
              --pdb_dir /pdb_base/ \
              --embedding_dir /embeddings_base/ \
              --task "${TASK}" \
              --gpu "${GPU_ID}" > /dev/null 2>&1 &
        else
            docker run --name "${CONTAINER_ID}" --gpus "${GPUS_VALUE}" $([ -n "${SHM_SIZE}" ] && echo "--shm-size=${SHM_SIZE}") \
              -v "${SEQUENCES_DIR}":/input \
              -v "${FILE_OUTPUT_DIR}":/output:rw \
              -v "${FASTA_STRUCTURES_DIR}":/pdb_base \
              -v "${FASTA_EMBEDDINGS_DIR}":/embeddings_base \
              -v "${GEOPOC_MODEL_DIR}":/models \
              -v "${ESM_MODEL_DIR}":/root/.cache/torch/hub/checkpoints \
              "${DOCKER_IMAGE}" \
              -i "/input/${FILENAME}" \
              -o "/output/" \
              --model_path /models/ \
              --pdb_dir /pdb_base/ \
              --embedding_dir /embeddings_base/ \
              --task "${TASK}" \
              --gpu "${GPU_ID}" &
        fi
          
        # Add container ID to the job list
        JOB_IDS+=("$CONTAINER_ID")
        
        # Sleep a bit to let Docker register the container
        sleep 1
        
        if [ "$QUIET" = false ]; then
            echo "Started Docker container ${CONTAINER_ID} (running: $(count_containers)/${MAX_CONTAINERS})"
        fi
    done
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
    
    # Clean up container
    docker rm "${CONTAINER_ID}" 2>/dev/null || true
done

# Print summary
if [ "$QUIET" = false ]; then
    echo "==== GeoPoc Analysis Complete ===="
    echo "Total jobs: ${TOTAL_JOBS}"
    echo "Successful: ${SUCCESS}"
    echo "Failed: ${FAILURE}"
    echo "Results are available in subdirectories under: ${OUTPUT_DIR}"
    echo "Each input file has its own subdirectory named after the file's base name"
    
    if [ ${FAILURE} -gt 0 ]; then
        echo "Warning: ${FAILURE} job(s) failed"
    else
        echo "All jobs completed successfully!"
    fi
fi

if [ ${FAILURE} -gt 0 ]; then
    exit 1
fi