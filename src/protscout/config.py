import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
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
        
        # Allow environment variable overrides
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """Override config values with environment variables"""
        # Format: PROTSCOUT_CONDITION=ultra
        for key, value in os.environ.items():
            if key.startswith('PROTSCOUT_'):
                config_key = key[10:].lower()  # Remove prefix
                if config_key in self._config:
                    self._config[config_key] = value
    
    def _setup_paths(self):
        """Setup all directory paths based on condition"""
        condition = self._config['condition']
        workdir = Path(self._config['workdir'])
        modeldir = Path(self._config['modeldir'])
        
        # Input/Output paths
        input_base = workdir / "data/pazy_protein_homologs_total"
        output_base = workdir / "results/pazy_homologs_total"
        
        self.paths = {
            # Input paths
            'input_fasta': input_base / condition,
            'clean_fasta': input_base / f"clean_{condition}",
            'substrate_analogs': workdir / "data/substrate_analogs/substrate_analogs.tsv",
            'catpred_input': input_base / "catpred_data",
            
            # Output directories
            'output_dir': output_base / f"outputs_{condition}",
            'results_dir': output_base / f"results_{condition}",
            
            # Tool-specific outputs
            'structures': output_base / f"outputs_{condition}/structures",
            'embeddings': output_base / f"outputs_{condition}/embeddings",
            'temberture_output': output_base / f"outputs_{condition}/temberture",
            'geopoc_output': output_base / f"outputs_{condition}/geopoc",
            'gatsol_output': output_base / f"outputs_{condition}/gatsol",
            'catpred_output': output_base / f"outputs_{condition}/catpred",
            'classical_props_output': output_base / f"results_{condition}/classical_properties",
            
            # Results directories
            'temberture_results': output_base / f"results_{condition}/temberture_results",
            'geopoc_results': output_base / f"results_{condition}/geopoc_results",
            'gatsol_results': output_base / f"results_{condition}/gatsol_results",
            'catpred_results': output_base / f"results_{condition}/catpred_results",
            'consolidated_results': output_base / f"results_{condition}/consolidated_results",
            
            # Model paths
            'modeldir': modeldir,
            'geopoc_model': modeldir / "geopoc",
            'gatsol_model': modeldir / "gatsol",
            
            # Logs
            'log_dir': workdir / "logs",
        }
        
        # Create all directories with proper permissions
        for path in self.paths.values():
            if path.suffix == '':  # Only create directories, not files
                path.mkdir(parents=True, exist_ok=True)
                # Set permissions if running with sudo
                if os.geteuid() == 0:
                    os.chmod(path, 0o777)
    
    def _setup_logging(self):
        """Setup logging directory"""
        log_dir = self.paths['log_dir']
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"protscout_run_{timestamp}.log"
    
    def _validate_config(self):
        """Validate required configuration parameters"""
        required_keys = ['condition', 'workdir', 'modeldir', 'workers']
        missing = [k for k in required_keys if k not in self._config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
        
        # Validate paths exist
        if not Path(self._config['workdir']).exists():
            raise ValueError(f"Workdir does not exist: {self._config['workdir']}")
        if not Path(self._config['modeldir']).exists():
            raise ValueError(f"Modeldir does not exist: {self._config['modeldir']}")
    
    def get(self, key: str, default=None):
        """Get configuration value with optional default"""
        return self._config.get(key, default)
    
    def get_container_config(self, container_name: str) -> Dict[str, Any]:
        """Get container-specific configuration"""
        containers = self._config.get('containers', {})
        return containers.get(container_name, {})
    
    def __getattr__(self, key):
        """Allow attribute-style access to config values"""
        if key in self._config:
            return self._config[key]
        raise AttributeError(f"Config has no attribute '{key}'")