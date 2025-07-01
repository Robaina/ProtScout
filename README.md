<div style="text-align: center; width: 100%">
    <img src="imgs/protscout_logo.png" max-width=400px height="auto">
</div>

## 🎯 About

ProtScout is a Python package that enables ranking of protein sequences based on multiple properties predicted by state-of-the-art AI models. It provides a unified interface to assess and compare proteins using various characteristics such as stability, solubility, and catalytic efficiency.

## ✨ Features

- 🧬 Comprehensive protein property analysis (structure, embeddings, catalytic activity, thermal stability, environmental tolerances, solubility, classical properties)
- 🐳 Containerized execution of prediction tools with Docker
- 🚀 Modular, parallel workflow with configurable steps and automatic resume
- 🔄 Automatic retry and resume support for robust execution
- ✅ Validation and dry-run modes to preview workflow
- 🔧 Fully configurable via YAML files and environment variable overrides
- 📈 Detailed logging and resource monitoring

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Docker (for running containerized prediction tools)
- NVIDIA GPU with CUDA support (recommended)
- Conda or Poetry for environment management

### Install with Poetry (Recommended)

```bash
# Clone repository
git clone https://github.com/Robaina/ProtScout.git
cd ProtScout

# Install Poetry if you haven't already
pip install poetry


# Install package and dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Install with pip

```bash
# Clone repository
git clone https://github.com/Robaina/ProtScout.git
cd ProtScout


# Install in development mode
pip install -e .
```

## 📋 Quick Start

Generate a configuration file:

```bash
protscout init -o my_config.yaml
```

Edit the configuration file with your paths and settings:

```yaml
condition: ultra
workdir: /path/to/your/workdir
modeldir: /path/to/model/weights
python_executable: /path/to/conda/env/bin/python
memory: 100g
workers: 2
quiet: true
max_retries: 2
preserve_artifacts: false
```

Validate your setup:

```bash
protscout validate -c my_config.yaml
```

Run the workflow:

```bash
protscout run -c my_config.yaml
```

## 📖 Usage Examples

### Basic Workflow

```bash
# Run complete workflow
protscout run -c config.yaml

# Run specific steps only
protscout run -c config.yaml -s clean_sequences -s esmfold

# Override condition from command line
protscout run -c config.yaml --condition ultra
```

### Advanced Features

```bash
# Resume from last successful step after failure
protscout run -c config.yaml --resume

# Dry run to see what would be executed
protscout run -c config.yaml --dry-run

# Monitor logs in real-time
protscout logs logs/protscout_run_20240112_143022.log -f
```

### Parallel Execution

The workflow automatically runs compatible steps in parallel:

- ESMFold and ESM-2 run simultaneously
- All prediction tools (CatPred, Temberture, GeoPoc, GATSol) run in parallel
- Result processing steps are parallelized

## 🔧 Configuration

ProtScout uses YAML configuration files for workflow management. Key configuration sections:

```yaml
# Analysis condition
condition: ultra

# Core directories
workdir: /path/to/your/workdir
modeldir: /path/to/model/weights

# Execution settings
python_executable: /path/to/conda/env/bin/python
memory: 100g
workers: 2
quiet: true
max_retries: 2
preserve_artifacts: false

# Container images (optional overrides)
containers:
  esmfold:
    image: ghcr.io/new-atlantis-labs/esmfold:latest
    max_containers: 1
  # ... other containers: esm2, catpred, temberture, geopoc, gatsol

# GPU and shared memory settings
resources:
  gpus: all
  shm_size: 100g

# Workflow steps
steps:
  - clean_sequences
  - esmfold
  - esm2
  - remove_sequences_without_pdb
  - prepare_catpred
  - catpred
  - temberture
  - geopoc
  - gatsol
  - classical_properties
  - process_temberture
  - process_geopoc
  - process_gatsol
  - process_catpred
  - consolidate_results
```

See `configs/example_workflow.yaml` for a complete example.

## 🛠️ Workflow Steps

- `clean_sequences` - Clean and deduplicate input sequences
- `esmfold` - Predict protein structures using ESMFold
- `esm2` - Generate protein embeddings using ESM-2
- `remove_sequences_without_pdb` - Filter sequences without structures
- `prepare_catpred` - Prepare inputs for catalytic prediction
- `catpred` - Predict catalytic properties
- `temberture` - Predict temperature stability
- `geopoc` - Predict environmental conditions (temp, pH, salt)
- `gatsol` - Predict solubility
- `classical_properties` - Calculate classical protein properties
- `process_*` - Process results from each tool
- `consolidate_results` - Create final output tables

## 📊 Output Structure

```
<artifacts_dir>/                  # raw outputs (artifacts)
├── structures/                   # PDB files from ESMFold
├── embeddings/                   # ESM-2 embeddings
├── clean_sequences/              # cleaned FASTA files
├── catpred_data/                 # prepared inputs for CatPred
├── catpred/                      # CatPred raw output
├── temberture/                   # temperature stability predictions
├── geopoc/                       # environmental predictions (temp, pH, salt)
└── gatsol/                       # solubility predictions

<results_dir>/                    # processed results
├── classical_properties_results/ # classical property outputs
├── temberture_results/            # processed temperature results
├── geopoc_results/                # processed environmental results
├── gatsol_results/                # processed solubility results
├── catpred_results/               # processed CatPred results
└── consolidated_results/          # final consolidated tables
```

## 🔄 Resume Capability

ProtScout automatically saves workflow state and can resume from failures:

```bash
# If workflow fails at step 'gatsol'
protscout run -c config.yaml --resume
# Workflow will skip completed steps and continue from 'gatsol'
```

## 📝 Logging

Comprehensive logging with multiple levels:

- Console output: INFO level (progress and important messages)
- Log file: DEBUG level (detailed execution information)

Logs are saved to: `{workdir}/logs/protscout_run_YYYYMMDD_HHMMSS.log`

## 🐛 Troubleshooting

### Docker Issues

```bash
# Check if Docker is running
docker info

# Ensure user has Docker permissions
sudo usermod -aG docker $USER
```

### GPU Issues

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
nvcc --version
```

### Memory Issues

- Reduce `max_containers` in configuration
- Decrease `toks_per_batch` for ESM-2
- Lower batch sizes for prediction tools

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

## 📚 Citation

If you use ProtScout in your research, please cite:

```bibtex
@software{protscout2024,
  author = {Robaina-Estévez, Semidán},
  title = {ProtScout: AI-powered protein sequence ranking},
  year = {2025},
  url = {https://github.com/Robaina/ProtScout}
}
```

## 🙏 Acknowledgments

ProtScout integrates several state-of-the-art protein prediction tools:

- ESMFold for structure prediction
- ESM-2 for sequence embeddings
- CatPred for catalytic activity prediction
- Temberture for thermal stability prediction
- GeoPoc for environmental condition prediction
- GATSol for solubility prediction