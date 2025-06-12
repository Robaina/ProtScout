#!/bin/bash

# Default values
INPUT_DIR="/home/ec2-user/SageMaker/mangrove-plastic-degrading/data/protein_sequences_plastic_degrading_clean"
OUTPUT_BASE_DIR="/home/ec2-user/SageMaker/mangrove-plastic-degrading/outputs/embeddings"
MEMORY="16g"
MODEL="esm2_t36_3B_UR50D"
TOKS_PER_BATCH="4096"
INCLUDE="per_tok"
MAX_CONTAINERS=4
DOCKER_IMAGE="ghcr.io/new-atlantis-labs/esm2:latest"
WEIGHTS_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input) INPUT_DIR="$2"; shift 2 ;;
        -o|--output) OUTPUT_BASE_DIR="$2"; shift 2 ;;
        -m|--memory) MEMORY="$2"; shift 2 ;;
        -M|--model) MODEL="$2"; shift 2 ;;
        -t|--toks_per_batch) TOKS_PER_BATCH="$2"; shift 2 ;;
        -I|--include) INCLUDE="$2"; shift 2 ;;
        -c|--max-containers) MAX_CONTAINERS="$2"; shift 2 ;;
        -d|--docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
        -w|--weights) WEIGHTS_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -i, --input DIR          Input directory with .faa/.fasta files"
            echo "  -o, --output DIR         Output base directory"
            echo "  -m, --memory MEM         Memory allocation (default: 16g)"
            echo "  -M, --model MODEL        ESM model (default: esm2_t36_3B_UR50D)"
            echo "  -t, --toks_per_batch N   Tokens per batch (default: 4096)"
            echo "  -I, --include OPT        Include option (default: per_tok)"
            echo "  -c, --max-containers N   Max concurrent containers (default: 4)"
            echo "  -d, --docker-image IMG   Docker image (default: ghcr.io/new-atlantis-labs/esm2:latest)"
            echo "  -w, --weights DIR        Pre-downloaded weights directory (optional)"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Ensure directories exist
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

if [ ! -d "$OUTPUT_BASE_DIR" ]; then
    echo "Creating output directory: $OUTPUT_BASE_DIR"
    mkdir -p "$OUTPUT_BASE_DIR"
fi

if [ -n "$WEIGHTS_DIR" ] && [ ! -d "$WEIGHTS_DIR" ]; then
    echo "Error: Weights directory does not exist: $WEIGHTS_DIR"
    exit 1
fi

# Print settings
echo "Settings:"
echo "- Input directory: $INPUT_DIR"
echo "- Output directory: $OUTPUT_BASE_DIR"
echo "- Model: $MODEL"
echo "- Memory: $MEMORY"
echo "- Tokens per batch: $TOKS_PER_BATCH"
echo "- Include: $INCLUDE"
echo "- Max containers: $MAX_CONTAINERS"
echo "- Docker image: $DOCKER_IMAGE"
if [ -n "$WEIGHTS_DIR" ]; then
    echo "- Weights directory: $WEIGHTS_DIR"
fi

# Count running containers of our type
count_containers() {
    docker ps --format '{{.Names}}' | grep -c "^esm2_"
}

# Find input files (avoid using a pipeline that creates a subshell)
readarray -t FILES < <(find "$INPUT_DIR" -type f \( -name "*.faa" -o -name "*.fasta" \))
TOTAL_FILES=${#FILES[@]}
echo "Found $TOTAL_FILES files to process"

# Set up weights mount if needed
WEIGHTS_MOUNT=""
if [ -n "$WEIGHTS_DIR" ]; then
    WEIGHTS_MOUNT="-v $WEIGHTS_DIR:/root/.cache/torch/hub/checkpoints"
fi

# Process files with simple parallelism control
for i in "${!FILES[@]}"; do
    file="${FILES[$i]}"
    filename=$(basename "$file")
    basename="${filename%.*}"
    output_dir="${OUTPUT_BASE_DIR}/${basename}"
    
    # Wait until we have room to run another container
    while [ "$(count_containers)" -ge "$MAX_CONTAINERS" ]; do
        echo "Currently at max containers ($MAX_CONTAINERS). Waiting..."
        sleep 5
    done
    
    # Create output directory
    mkdir -p "$output_dir"
    
    echo "[$((i+1))/$TOTAL_FILES] Processing $filename"
    
    # Run Docker container
    docker run --rm --gpus all \
        -v "$INPUT_DIR:/app/input" \
        -v "$output_dir:/app/output:rw" \
        $WEIGHTS_MOUNT \
        --memory="$MEMORY" \
        --name "esm2_${basename}" \
        "$DOCKER_IMAGE" \
        --input "/app/input/$filename" \
        --model "$MODEL" \
        --include "$INCLUDE" \
        --toks_per_batch "$TOKS_PER_BATCH" &
    
    # Sleep a bit to let Docker register the container
    sleep 1
    
    echo "Started Docker container for $filename (running: $(count_containers)/$MAX_CONTAINERS)"
done

# Wait for all containers to finish
echo "All tasks started, waiting for containers to finish..."
while [ "$(count_containers)" -gt 0 ]; do
    echo "Waiting for $(count_containers) containers to finish..."
    sleep 10
done

echo "All processing complete!"