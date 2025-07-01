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
        # Base directory for raw outputs (artifacts)
        artifacts_base = Path(path_config.get('artifacts_base',
                                    path_config.get('output_base',
                                                    workdir / "results")))
        # Base directory for processed results
        results_base = Path(path_config.get('results_base',
                                   path_config.get('output_base',
                                                   workdir / "results")))
        
        # Define the main artifacts directory (raw outputs)
        # If not explicitly set, use artifacts_base directly (no nested condition)
        output_dir = Path(path_config.get('output_dir', artifacts_base))
        # Define the main processed results directory
        # If not explicitly set, use results_base directly (no nested condition)
        results_dir = Path(path_config.get('results_dir', results_base))

        # Determine input directory and cleaned sequences directory (as artifacts)
        input_fasta = Path(path_config.get('input_fasta', input_base / "sequences"))
        # clean_fasta defaults to an artifacts subdirectory unless overridden
        if 'clean_fasta' in path_config:
            clean_fasta = Path(path_config['clean_fasta'])
        else:
            clean_fasta = output_dir / f"clean_{input_fasta.name}"
        substrate_analogs = Path(path_config.get('substrate_analogs', input_base / "substrate_analogs.tsv"))
        # catpred_input defaults to an artifacts subdirectory unless overridden
        if 'catpred_input' in path_config:
            catpred_input = Path(path_config['catpred_input'])
        else:
            catpred_input = output_dir / "catpred_data"
        self.paths = {
            # Input paths
            'input_fasta': input_fasta,
            'clean_fasta': clean_fasta,
            'substrate_analogs': substrate_analogs,
            'catpred_input': catpred_input,

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
            
            # Logs (accept 'log_dir' or legacy 'logs')
            'log_dir': Path(path_config.get('log_dir',
                                         path_config.get('logs', workdir / "logs")) ),
        }

        # Create directories based on configured steps
        steps = self.get('steps', []) or []
        always_create = {'output_dir', 'results_dir', 'log_dir', 'modeldir'}
        step_required_dirs = {
            'input_fasta': ['clean_sequences', 'esmfold', 'esm2',
                             'remove_sequences_without_pdb', 'prepare_catpred', 'classical_properties'],
            'clean_fasta': ['clean_sequences', 'remove_sequences_without_pdb',
                            'prepare_catpred', 'catpred', 'temberture',
                            'geopoc', 'gatsol', 'classical_properties'],
            'substrate_analogs': ['prepare_catpred'],
            'catpred_input': ['prepare_catpred'],
            'structures': ['esmfold', 'geopoc', 'gatsol'],
            'embeddings': ['esm2', 'geopoc', 'gatsol'],
            'temberture_output': ['temberture'],
            'geopoc_output': ['geopoc'],
            'gatsol_output': ['gatsol'],
            'catpred_output': ['catpred'],
            'classical_properties_results': ['classical_properties'],
            'temberture_results': ['process_temberture'],
            'geopoc_results': ['process_geopoc'],
            'gatsol_results': ['process_gatsol'],
            'catpred_results': ['process_catpred'],
            'consolidated_results': ['consolidate_results']
        }
        for key, path in self.paths.items():
            create = False
            if key in always_create:
                create = True
            elif key in step_required_dirs and any(step in steps for step in step_required_dirs[key]):
                create = True
            if create and path and not path.suffix:
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