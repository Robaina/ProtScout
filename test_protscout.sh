#!/bin/bash
# bash run_test_workflow.sh > test_workflow.log 2>&1
# sudo docker rm -f $(docker ps -aq)

echo "Starting ProtScout test workflow..."

# Python executable path
PYTHON_EXECUTABLE="/home/ubuntu/.conda/envs/protscout/bin/python"

# Main configuration directories
WORKDIR="$(pwd)"
MODELDIR="/home/ubuntu/lab4/model_weights"
INPUT_FASTA_DIR="${WORKDIR}/tests/data/input"
FASTA_DIR="${WORKDIR}/tests/data/clean"
MEMORY="100g"
SUBSTRATE_ANALOGS="${WORKDIR}/tests/data/substrate_analogs.tsv"

# Output directories
OUTPUT_DIR="${WORKDIR}/tests/output/outputs"
RESULTS_DIR="${WORKDIR}/tests/output/results"

# Fix ESMFold permission error
sudo mkdir -p /home/vscode/output
sudo chown -R 1000:1000 /home/vscode/output

# Define tool output directories
STRUCTURES_DIR="${OUTPUT_DIR}/structures"
EMBEDDINGS_DIR="${OUTPUT_DIR}/embeddings"
CATPRED_INPUT="${WORKDIR}/tests/data/catpred_data"
TEMBERTURE_OUTPUT="${OUTPUT_DIR}/temberture"
GEOPOC_OUTPUT="${OUTPUT_DIR}/geopoc"
GATSOL_OUTPUT="${OUTPUT_DIR}/gatsol"
CATPRED_OUTPUT="${OUTPUT_DIR}/catpred"
CLASSICAL_PROPS_OUTPUT="${RESULTS_DIR}/classical_properties"

# Create tool results directories
TEMBERTURE_RESULTS="${RESULTS_DIR}/temberture_results"
GEOPOC_RESULTS="${RESULTS_DIR}/geopoc_results"
GATSOL_RESULTS="${RESULTS_DIR}/gatsol_results"
CATPRED_RESULTS="${RESULTS_DIR}/catpred_results"
CONSOLIDATED_RESULTS="${RESULTS_DIR}/consolidated_results"

# Model paths
GEOPOC_MODEL_PATH="${MODELDIR}/geopoc"
GATSOL_MODEL_PATH="${MODELDIR}/gatsol"

# Container configuration
ESMFOLD_MAX_CONTAINERS=1
ESM2_MAX_CONTAINERS=1 # out of CUDA memory with 2
ESM2_TOKS_PER_BATCH=4096
WORKERS=2
QUIET="--quiet"
# QUIET=""

# Docker resource parameters
GPUS_VALUE="all"
SHM_SIZE="100g"

# Docker images
ESMFOLD_IMAGE="ghcr.io/new-atlantis-labs/esmfold:latest"
ESM2_IMAGE="ghcr.io/new-atlantis-labs/esm2:latest"
CATPRED_IMAGE="ghcr.io/new-atlantis-labs/catpred:latest"
TEMBERTURE_IMAGE="ghcr.io/new-atlantis-labs/temberture:latest"
GEOPOC_IMAGE="ghcr.io/new-atlantis-labs/geopoc:latest"
GATSOL_IMAGE="ghcr.io/new-atlantis-labs/gatsol:latest"

# Create directories with proper permissions
echo "Setting up test directories with proper permissions..."

sudo mkdir -p "${OUTPUT_DIR}" && sudo chmod -R 777 "${OUTPUT_DIR}"
sudo mkdir -p "${FASTA_DIR}" && sudo chmod -R 777 "${FASTA_DIR}"
sudo mkdir -p "${STRUCTURES_DIR}" && sudo chmod -R 777 "${STRUCTURES_DIR}"
sudo mkdir -p "${EMBEDDINGS_DIR}" && sudo chmod -R 777 "${EMBEDDINGS_DIR}"
sudo mkdir -p "${CATPRED_INPUT}" && sudo chmod -R 777 "${CATPRED_INPUT}"
sudo mkdir -p "${TEMBERTURE_OUTPUT}" && sudo chmod -R 777 "${TEMBERTURE_OUTPUT}"
sudo mkdir -p "${GEOPOC_OUTPUT}" && sudo chmod -R 777 "${GEOPOC_OUTPUT}"
sudo mkdir -p "${GATSOL_OUTPUT}" && sudo chmod -R 777 "${GATSOL_OUTPUT}"
sudo mkdir -p "${CATPRED_OUTPUT}" && sudo chmod -R 777 "${CATPRED_OUTPUT}"
sudo mkdir -p "${CLASSICAL_PROPS_OUTPUT}" && sudo chmod -R 777 "${CLASSICAL_PROPS_OUTPUT}"
sudo mkdir -p "${RESULTS_DIR}" && sudo chmod -R 777 "${RESULTS_DIR}"
sudo mkdir -p "${TEMBERTURE_RESULTS}" && sudo chmod -R 777 "${TEMBERTURE_RESULTS}"
sudo mkdir -p "${GEOPOC_RESULTS}" && sudo chmod -R 777 "${GEOPOC_RESULTS}"
sudo mkdir -p "${GATSOL_RESULTS}" && sudo chmod -R 777 "${GATSOL_RESULTS}"
sudo mkdir -p "${CATPRED_RESULTS}" && sudo chmod -R 777 "${CATPRED_RESULTS}"
sudo mkdir -p "${CONSOLIDATED_RESULTS}" && sudo chmod -R 777 "${CONSOLIDATED_RESULTS}"
sudo mkdir -p "${WORKDIR}/tests/logs" && sudo chmod -R 777 "${WORKDIR}/tests/logs"

# Step 0: Clean input fasta files
echo "Cleaning input FASTA files"
${PYTHON_EXECUTABLE} src/protscout/clean_sequences.py \
  --input_dir "${INPUT_FASTA_DIR}" \
  --output_dir "${FASTA_DIR}" \
  --remove_duplicates \
  --prefix ""

# Step 1: Run ESMFold
echo "Running ESMFold structure prediction on test proteins"
sudo src/protscout/run_esmfold.sh \
  --input "${FASTA_DIR}" \
  --output "${STRUCTURES_DIR}" \
  --docker-image ${ESMFOLD_IMAGE} \
  --max-containers ${ESMFOLD_MAX_CONTAINERS} \
  --memory ${MEMORY}

# Step 2: Run ESM-2
echo "Running ESM-2 embedding generation on test proteins"
sudo src/protscout/run_esm2.sh \
  --input "${FASTA_DIR}" \
  --output "${EMBEDDINGS_DIR}" \
  --docker-image ${ESM2_IMAGE} \
  --include "per_tok" \
  --model esm2_t36_3B_UR50D \
  --weights "${MODELDIR}" \
  --max-containers ${ESM2_MAX_CONTAINERS} \
  --toks_per_batch ${ESM2_TOKS_PER_BATCH} \
  --memory ${MEMORY}

# Step 3: Remove sequences without PDB files
echo "Removing sequences without PDB files"
${PYTHON_EXECUTABLE} src/protscout/remove_sequences_without_pdb.py \
  --faa_dir "${FASTA_DIR}" \
  --pdb_dir "${STRUCTURES_DIR}" \
  --output_dir "${FASTA_DIR}"

# Step 4: Prepare CatPred inputs
echo "Preparing CatPred inputs"
${PYTHON_EXECUTABLE} src/protscout/prepare_catpred_inputs.py \
  --fasta_dir "${FASTA_DIR}" \
  --substrate_tsv "${SUBSTRATE_ANALOGS}" \
  --output_dir "${CATPRED_INPUT}"

# Step 5: Run CatPred
echo "=== Running CatPred catalytic activity prediction ==="
sudo src/protscout/run_catpred.sh \
  --input ${CATPRED_INPUT} \
  --output ${CATPRED_OUTPUT} \
  --model ${MODELDIR} \
  --parallel ${WORKERS} \
  --docker-image ${CATPRED_IMAGE} \
  --gpus "${GPUS_VALUE}" \
  --shm-size "${SHM_SIZE}" \
  ${QUIET}
  
# Step 6: Run Temberture
echo "=== Running Temberture temperature stability prediction ==="
sudo src/protscout/run_temberture.sh \
  --input ${FASTA_DIR} \
  --output ${TEMBERTURE_OUTPUT} \
  --parallel ${WORKERS} \
  --docker-image ${TEMBERTURE_IMAGE} \
  --gpus "${GPUS_VALUE}" \
  --shm-size "${SHM_SIZE}" \
  ${QUIET}
  
# Step 7: Run GeoPoc
echo "=== Running GeoPoc optimal condition prediction ==="
sudo src/protscout/run_geopoc.sh \
  --input ${FASTA_DIR} \
  --output ${GEOPOC_OUTPUT} \
  --esm ${MODELDIR} \
  --model ${GEOPOC_MODEL_PATH} \
  --structures ${STRUCTURES_DIR} \
  --embeddings ${EMBEDDINGS_DIR} \
  --tasks "temp,pH,salt" \
  --parallel ${WORKERS} \
  --docker-image ${GEOPOC_IMAGE} \
  --gpus "${GPUS_VALUE}" \
  --shm-size "${SHM_SIZE}" \
  ${QUIET}
  
# Step 8: Run GATSol
echo "=== Running GATSol solubility prediction ==="
sudo src/protscout/run_gatsol.sh \
  --input ${FASTA_DIR} \
  --output ${GATSOL_OUTPUT} \
  --structures ${STRUCTURES_DIR} \
  --model ${GATSOL_MODEL_PATH} \
  --esm ${MODELDIR} \
  --batch-size 32 \
  --parallel ${WORKERS} \
  --feature-batch 4 \
  --timeout 600 \
  --docker-image ${GATSOL_IMAGE} \
  --gpus "${GPUS_VALUE}" \
  --shm-size "${SHM_SIZE}" \
  ${QUIET}

# Step 8.5: Run Classical Properties Prediction
echo "=== Running Classical Properties Prediction ==="
LOG_FILE="${WORKDIR}/tests/logs/classical_properties.log"
echo "Computing classical properties for test proteins..."
${PYTHON_EXECUTABLE} src/protscout/predict_classic_properties.py \
  --input "${FASTA_DIR}" \
  --log "${LOG_FILE}" \
  --output "${CLASSICAL_PROPS_OUTPUT}"
  
# Step 9: Process analysis results
echo "=== Processing analysis results ==="

# Process Temberture results
echo "Processing Temberture results..."
${PYTHON_EXECUTABLE} src/protscout/process_temberture.py \
  -i ${TEMBERTURE_OUTPUT} \
  -o ${TEMBERTURE_RESULTS}

# Process GeoPoc results
echo "Processing GeoPoc results..."
${PYTHON_EXECUTABLE} src/protscout/process_geopoc.py \
  --input ${GEOPOC_OUTPUT} \
  --output ${GEOPOC_RESULTS}

# Process GATSol results
echo "Processing GATSol results..."
${PYTHON_EXECUTABLE} src/protscout/process_gatsol.py \
  --input_dir ${GATSOL_OUTPUT} \
  --output_dir ${GATSOL_RESULTS}

# Process CatPred results
echo "Processing CatPred results..."
${PYTHON_EXECUTABLE} src/protscout/process_catpred.py \
  --input_dir ${CATPRED_OUTPUT} \
  --faa_dir ${FASTA_DIR} \
  --output_dir ${CATPRED_RESULTS}

# Create consolidated output tables
echo "Creating consolidated output tables..."
${PYTHON_EXECUTABLE} src/protscout/make_output_tables.py \
  -i ${RESULTS_DIR} \
  -o ${CONSOLIDATED_RESULTS}

echo "ProtScout test workflow complete! All analyses finished successfully."
echo "Test results are available in: ${OUTPUT_DIR}"
echo "Final processed results are available in: ${RESULTS_DIR}"
echo "Consolidated results are available in: ${CONSOLIDATED_RESULTS}"