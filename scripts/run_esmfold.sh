#!/bin/bash
umask 000

# Default values
DEFAULT_INPUT_DIR="/home/ec2-user/SageMaker/mangrove-plastic-degrading/data/protein_sequences_plastic_degrading_representatives"
DEFAULT_OUTPUT_BASE_DIR="/home/ec2-user/SageMaker/mangrove-plastic-degrading/outputs/structures"
DEFAULT_MEMORY="16g"
DEFAULT_MAX_CONTAINERS=4  # Default maximum number of concurrent containers
DEFAULT_DOCKER_IMAGE="ghcr.io/new-atlantis-labs/esmfold:latest"  # Default Docker image

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --input DIR     Input directory containing .faa/.fasta files (default: ${DEFAULT_INPUT_DIR})"
    echo "  -o, --output DIR    Output base directory for results (default: ${DEFAULT_OUTPUT_BASE_DIR})"
    echo "  -m, --memory MEM    Memory allocation for container (default: ${DEFAULT_MEMORY})"
    echo "  -p, --max-containers NUM  Maximum number of concurrent containers (default: ${DEFAULT_MAX_CONTAINERS})"
    echo "  -d, --docker-image IMG    Docker image to use (default: ${DEFAULT_DOCKER_IMAGE})"
    echo "  -h, --help          Show this help message"
    exit 1
}

# Parse command line arguments
INPUT_DIR="${DEFAULT_INPUT_DIR}"
OUTPUT_BASE_DIR="${DEFAULT_OUTPUT_BASE_DIR}"
MEMORY="${DEFAULT_MEMORY}"
MAX_CONTAINERS="${DEFAULT_MAX_CONTAINERS}"
DOCKER_IMAGE="${DEFAULT_DOCKER_IMAGE}"

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        -p|--max-containers)
            MAX_CONTAINERS="$2"
            shift 2
            ;;
        -d|--docker-image)
            DOCKER_IMAGE="$2"
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

# Function to check if required directories exist and create them if needed
check_directories() {
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Error: Input directory does not exist: $INPUT_DIR"
        exit 1
    fi
    
    if [ ! -d "$OUTPUT_BASE_DIR" ]; then
        echo "Creating output base directory: $OUTPUT_BASE_DIR"
        mkdir -p "$OUTPUT_BASE_DIR"
    fi
}

# Function to count currently running ESMFold containers
count_running_containers() {
    docker ps | grep "$DOCKER_IMAGE" | wc -l
}

# Function to wait until we have room to run another container
wait_for_container_slot() {
    while true; do
        current_containers=$(count_running_containers)
        if [ "$current_containers" -lt "$MAX_CONTAINERS" ]; then
            break
        fi
        echo "Currently running $current_containers containers, maximum is $MAX_CONTAINERS. Waiting..."
        sleep 10
    done
}

# Function to process a single fasta/faa file
process_file() {
    local input_file="$1"
    local filename=$(basename "$input_file")
    local basename="${filename%.*}"
    local output_dir="${OUTPUT_BASE_DIR}/${basename}"
    
    # Create output directory if it doesn't exist
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    fi
    
    # Wait until we have room to run another container
    wait_for_container_slot
    
    echo "Processing $filename..."
    echo "Output directory: $output_dir"
    echo "Using memory: $MEMORY"
    echo "Using Docker image: $DOCKER_IMAGE"
    
    # Run ESMFold for the current file with memory setting in the background
    docker run --gpus all \
        -v "$(dirname "$input_file"):/home/vscode/input" \
        -v "$output_dir:/home/vscode/output:rw" \
        --memory="$MEMORY" \
        --name "esmfold_${basename}" \
        "$DOCKER_IMAGE" \
        -i "/home/vscode/input/$(basename "$input_file")" \
        -o "/home/vscode/output/" &
    
    # Record the container's process ID
    local container_pid=$!
    echo "Started container for $filename with PID $container_pid"
    
    # Return the PID so we can track it
    echo $container_pid
}

# Main execution
main() {
    echo "Using the following settings:"
    echo "  Input: ${INPUT_DIR}"
    echo "  Output: ${OUTPUT_BASE_DIR}"
    echo "  Memory: ${MEMORY}"
    echo "  Maximum concurrent containers: ${MAX_CONTAINERS}"
    echo "  Docker image: ${DOCKER_IMAGE}"
    
    # Check directories
    check_directories
    
    # Find all fasta/faa files in input directory
    file_count=0
    total_files=$(find "$INPUT_DIR" -type f \( -name "*.faa" -o -name "*.fasta" \) | wc -l)
    
    # Array to store files and their PIDs
    declare -A file_pids
    
    find "$INPUT_DIR" -type f \( -name "*.faa" -o -name "*.fasta" \) | while read -r file; do
        file_count=$((file_count + 1))
        echo "Starting file $file_count of $total_files: $(basename "$file")"
        
        # Process the file and capture the PID
        pid=$(process_file "$file")
        
        # Only add to tracking if a process was started (non-empty PID)
        if [ -n "$pid" ]; then
            file_pids["$pid"]="$file"
        fi
    done
    
    # Wait for all background processes to finish
    echo "Waiting for all processes to complete..."
    for pid in "${!file_pids[@]}"; do
        file="${file_pids[$pid]}"
        
        # Only wait if PID is a number (valid process)
        if [[ "$pid" =~ ^[0-9]+$ ]]; then
            wait "$pid"
            status=$?
            
            if [ $status -eq 0 ]; then
                echo "Process for $(basename "$file") completed successfully"
            else
                echo "Process for $(basename "$file") failed with exit code $status"
            fi
        fi
    done
    
    # Clean up containers at the end
    echo "Cleaning up containers..."
    docker ps -a | grep "$DOCKER_IMAGE" | awk '{print $1}' | xargs -r docker rm
    
    echo "All processing complete!"
}

# Run the script
main