import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

class Config:
    def __init__(self, config_file: Path):
        self.config_file = config_file
        self._load_config()
        self._setup_paths()
        self._setup_logging()
        self._validate_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_file) as f:
            self._config = yaml.safe_load(f)
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Override config values with environment variables"""
        for key, value in os.environ.items():
            if key.startswith('PROTSCOUT_'):
                config_key = key[10:].lower()
                if config_key in self._config:
                    self._config[config_key] = value

    def _setup_paths(self):
        """Setup all directory paths based on configuration"""
        condition = self.get('condition', 'default')
        workdir = Path(self.get('workdir'))
        modeldir = Path(self.get('modeldir', workdir / 'models'))
        path_config = self.get('paths', {})

        # Define base directories
        input_base = Path(path_config.get('input_base', workdir / "data"))
        output_base = Path(path_config.get('output_base', workdir / "results"))
        
        # Define the main results directory FIRST
        results_dir = Path(path_config.get('results_dir', output_base / f"results_{condition}"))
        
        # Define the main raw outputs directory
        output_dir = Path(path_config.get('output_dir', output_base / f"outputs_{condition}"))

        self.paths = {
            # Input paths
            'input_fasta': Path(path_config.get('input_fasta', input_base / condition)),
            'clean_fasta': Path(path_config.get('clean_fasta', input_base / f"clean_{condition}")),
            'substrate_analogs': Path(path_config.get('substrate_analogs', workdir / "data/substrate_analogs.tsv")),
            'catpred_input': Path(path_config.get('catpred_input', input_base / "catpred_data")),

            # Main Output & Results directories
            'output_dir': output_dir,
            'results_dir': results_dir,

            # Tool-specific RAW output directories (under output_dir)
            'structures': Path(path_config.get('structures', output_dir / "structures")),
            'embeddings': Path(path_config.get('embeddings', output_dir / "embeddings")),
            'temberture_output': Path(path_config.get('temberture_output', output_dir / "temberture")),
            'geopoc_output': Path(path_config.get('geopoc_output', output_dir / "geopoc")),
            'gatsol_output': Path(path_config.get('gatsol_output', output_dir / "gatsol")),
            'catpred_output': Path(path_config.get('catpred_output', output_dir / "catpred")),

            # PROCESSED results directories (now correctly based under results_dir)
            'classical_properties_results': Path(path_config.get('classical_properties_results', results_dir / "classical_properties_results")),
            'temberture_results': Path(path_config.get('temberture_results', results_dir / "temberture_results")),
            'geopoc_results': Path(path_config.get('geopoc_results', results_dir / "geopoc_results")),
            'gatsol_results': Path(path_config.get('gatsol_results', results_dir / "gatsol_results")),
            'catpred_results': Path(path_config.get('catpred_results', results_dir / "catpred_results")),
            'consolidated_results': Path(path_config.get('consolidated_results', results_dir / "consolidated_results")),

            # Model paths
            'modeldir': modeldir,
            'geopoc_model': Path(path_config.get('geopoc_model', modeldir / "geopoc")),
            'gatsol_model': Path(path_config.get('gatsol_model', modeldir / "gatsol")),
            
            # Logs
            'log_dir': Path(path_config.get('log_dir', workdir / "logs")),
        }

        # Create all directories
        for path in self.paths.values():
            if path and not path.suffix:
                path.mkdir(parents=True, exist_ok=True)
                if os.geteuid() == 0:
                    os.chmod(path, 0o777)

    def _setup_logging(self):
        """Setup logging directory"""
        log_dir = self.paths['log_dir']
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"protscout_{self.get('condition', 'default')}_{timestamp}.log"

    def _validate_config(self):
        """Validate required configuration parameters"""
        required_keys = ['workdir', 'workers']
        missing = [k for k in required_keys if not self.get(k)]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    def get(self, key: str, default=None):
        return self._config.get(key, default)

    def __getattr__(self, key):
        if key in self._config:
            return self._config[key]
        raise AttributeError(f"Config has no attribute '{key}'")