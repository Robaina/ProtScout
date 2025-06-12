<div style="text-align: center; width: 100%">
    <img src="imgs/protscout_logo.png" max-width=400px height="auto">
</div>

## 🎯 About

ProtScout is a Python package that enables ranking of protein sequences based on multiple properties predicted by state-of-the-art AI models. It provides a unified interface to assess and compare proteins using various characteristics such as stability, solubility, and catalytic efficiency.

## ✨ Features

- 🧬 Comprehensive protein property prediction
- 🤖 Integration with multiple AI models 
- 📊 Flexible ranking system with customizable weights
- 🚀 Efficient batch processing
- 📈 Property visualization tools
- 🔄 Combined prediction support

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

# Copy your existing scripts to the package
mkdir -p src/protscout/scripts
cp /path/to/your/scripts/*.py src/protscout/scripts/
cp /path/to/your/scripts/*.sh src/protscout/scripts/

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

# Copy scripts and install
mkdir -p src/protscout/scripts
cp /path/to/your/scripts/*.py src/protscout/scripts/
cp /path/to/your/scripts/*.sh src/protscout/scripts/

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

# Directory paths
workdir: /home/ubuntu/lab4/mangrove-plastic-degrading
modeldir: /home/ubuntu/lab4/model_weights

# Resource allocation
memory: 100g
workers: 2
quiet: true

# Container configuration
containers:
  esmfold:
    image: ghcr.io/new-atlantis-labs/esmfold:latest
    max_containers: 1

# Steps to execute
steps:
  - clean_sequences
  - esmfold
  - esm2
  # ... more steps
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
results/
└── pazy_homologs_total/
    ├── outputs_ultra/
    │   ├── structures/      # PDB files from ESMFold
    │   ├── embeddings/      # ESM-2 embeddings
    │   ├── catpred/         # CatPred predictions
    │   ├── temberture/      # Temperature predictions
    │   ├── geopoc/          # Environmental predictions
    │   └── gatsol/          # Solubility predictions
    └── results_ultra/
        ├── classical_properties/
        ├── *_results/           # Processed results
        └── consolidated_results/ # Final tables
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
  year = {2024},
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