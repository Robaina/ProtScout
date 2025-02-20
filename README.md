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

## 🛠️ Installation

```bash
pip install protscout
```

Or install from source using Poetry:

```bash
git clone https://github.com/Robaina/ProtScout
cd protscout
poetry install
```

## 🚀 Quick Start

```python
from protscout import ProteinRanker

# Initialize a combined predictor with custom weights
ranker = ProteinRanker(
    predictors=[
        KineticPredictor(substrate_smiles="CC(=O)Oc1ccc([N+](=O)[O-])cc1", weight=1.0),
        ThermostabilityPredictor(weight=0.5),
        SolubilityGATSolPredictor(weight=0.5)
    ]
)

# Load sequences
sequences = ["MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", ...]

# Get combined rankings
rankings = ranker.rank(sequences)
```

## 🎮 Available Predictors

### 🔬 Kinetics and Binding
- **KineticPredictor**: Predicts enzyme kinetic parameters (kcat, KM, Ki)
- **KMpredictor**: Specialized predictor for Michaelis constants (KM)
- **AffinityGNINApredictor**: Predicts protein-ligand binding affinity

### 🌡️ Stability and Structure
- **DeepStabPtmPredictor**: Predicts protein melting temperature (Tm)
- **ThermostabilityPredictor**: Predicts protein thermostability using TemBERTure
- **GeoPocPredictor**: Predicts optimal temperature, pH, and salt conditions

### 💧 Solubility and pH
- **SolubilityGATSolPredictor**: Predicts protein solubility using GATSol
- **EpHodPredictor**: Predicts optimal pH conditions

### 🔄 Combined Prediction
- **CombinedPredictor**: Enables weighted combination of multiple predictors

## 📋 Predictor Details

| Predictor | Input Requirements | Properties Predicted | Model Base |
|-----------|-------------------|---------------------|------------|
| KineticPredictor (CatPred) | Sequences, SMILES | kcat, KM, Ki | esm2_t33_650M_UR50D |
| KMpredictor | Sequences, SMILES | KM | ESM-2 |
| AffinityGNINApredictor | Sequences, PDB files, SDF | Binding Affinity | GNINA |
| DeepStabPtmPredictor | Sequences | Melting Temperature | Prot-T5-XL |
| ThermostabilityPredictor | Sequences | Thermostability | ProtBERT |
| GeoPocPredictor | Sequences, (optional PDBs) | Temperature, pH, Salt | esm2_t36_3B_UR50D |
| SolubilityGATSolPredictor | Sequences, PDB files | Solubility | ESM-1b |
| EpHodPredictor | Sequences | Optimal pH | ESM-1v |

## 🎯 Roadmap

- [ ] Optimize containers to enable input of protein embeddings for ESM-2 models
- [ ] Standardize embedding input across all models
- [ ] Support direct embedding input for ESM-based models
- [ ] Unify embedding computation to avoid redundant calculations
- [ ] Add support for custom model weights
- [ ] Implement caching for embeddings
- [ ] Add batch prediction support for all predictors

## 📚 Documentation

For detailed documentation, visit [protscout.readthedocs.io](https://protscout.readthedocs.io)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all the teams behind the integrated AI models
- Contributors and maintainers of the essential dependencies
- The computational biology community for valuable feedback