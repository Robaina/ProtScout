import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import re

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

    def _safe_path(self, path_input, fallback_path=None):
        """Safely convert input to Path object"""
        if path_input is None:
            if fallback_path is not None:
                return self._safe_path(fallback_path)
            return None
        
        if isinstance(path_input, Path):
            return path_input
        
        if isinstance(path_input, str):
            if path_input.strip():  # Non-empty string
                return Path(path_input)
            else:
                if fallback_path is not None:
                    return self._safe_path(fallback_path)
                return None
        
        # For any other type, try to convert to string first
        try:
            return Path(str(path_input))
        except:
            if fallback_path is not None:
                return self._safe_path(fallback_path)
            return None

    def _setup_paths(self):
        """Setup all directory paths based on configuration - FULLY DYNAMIC AND SAFE"""
        condition = self.get('condition', 'default')
        
        # Safely get workdir - this is required
        workdir_str = self.get('workdir')
        if not workdir_str:
            raise ValueError("workdir must be specified in configuration")
        workdir = self._safe_path(workdir_str)
        if not workdir:
            raise ValueError(f"Invalid workdir path: {workdir_str}")
        
        # Safely get other base paths
        path_config = self.get('paths', {})
        
        # Model directory
        modeldir = self._safe_path(
            self.get('modeldir'),
            workdir / 'models'
        )
        
        # Base directories
        input_base = self._safe_path(
            path_config.get('input_base'),
            workdir / "data"
        )
        
        artifacts_base = self._safe_path(
            path_config.get('artifacts_base') or path_config.get('output_base'),
            workdir / "artifacts"
        )
        
        results_base = self._safe_path(
            path_config.get('results_base'),
            workdir / "results"
        )
        
        # Main directories
        output_dir = self._safe_path(
            path_config.get('output_dir'),
            artifacts_base
        )
        
        results_dir = self._safe_path(
            path_config.get('results_dir'),
            results_base
        )
        
        # Log directory
        log_dir = self._safe_path(
            path_config.get('log_dir') or path_config.get('logs'),
            workdir / "logs"
        )

        # Initialize paths with safe base directories
        self.paths = {
            'workdir': workdir,
            'output_dir': output_dir,
            'results_dir': results_dir,
            'modeldir': modeldir,
            'log_dir': log_dir,
            'input_base': input_base,
            'artifacts_base': artifacts_base,
            'results_base': results_base,
        }
        
        # Add all explicitly defined paths from config safely
        for key, value in path_config.items():
            if key not in ['input_base', 'artifacts_base', 'results_base', 'output_base']:
                safe_path = self._safe_path(value)
                if safe_path:
                    self.paths[key] = safe_path
        
        # Apply intelligent defaults for missing paths
        self._apply_intelligent_defaults()
        
        # Create directories based on steps
        self._create_required_directories()

    def _apply_intelligent_defaults(self):
        """Apply intelligent defaults for missing paths based on naming patterns"""
        output_dir = self.paths['output_dir']
        results_dir = self.paths['results_dir']
        modeldir = self.paths['modeldir']
        input_base = self.paths.get('input_base', self.paths['workdir'] / "data")
        
        # Ensure all base paths are Path objects
        if not isinstance(output_dir, Path):
            output_dir = self.paths['workdir'] / "artifacts"
            self.paths['output_dir'] = output_dir
            
        if not isinstance(results_dir, Path):
            results_dir = self.paths['workdir'] / "results"
            self.paths['results_dir'] = results_dir
            
        if not isinstance(modeldir, Path):
            modeldir = self.paths['workdir'] / "models"
            self.paths['modeldir'] = modeldir
        
        # Common path patterns with safe Path operations
        path_patterns = {
            # Input paths
            r'.*_input$': lambda name: output_dir / f"{name.replace('_input', '')}_data",
            r'^input_.*': lambda name: input_base / name.replace('input_', ''),
            
            # Output paths
            r'.*_output$': lambda name: output_dir / name.replace('_output', ''),
            
            # Special paths
            r'^structures$': lambda name: output_dir / "structures",
            r'^embeddings$': lambda name: output_dir / "embeddings",
            r'^clean_fasta$': lambda name: output_dir / "clean_sequences",
            
            # Results paths
            r'.*_results$': lambda name: results_dir / name,
            r'.*_processed$': lambda name: results_dir / name.replace('_processed', '_results'),
            
            # Model paths
            r'.*_model$': lambda name: modeldir / name.replace('_model', ''),
            r'.*_models$': lambda name: modeldir / name.replace('_models', ''),
            
            # Special cases
            r'^clean_.*': lambda name: output_dir / name,
            r'.*_analogs$': lambda name: input_base / f"{name}.tsv",
            r'^consolidated_.*': lambda name: results_dir / name,
            r'^substrate_analogs$': lambda name: input_base / "substrate_analogs.tsv",
        }
        
        # Find all potential path keys
        potential_paths = set()
        
        # From explicit config
        path_config = self.get('paths', {})
        potential_paths.update(path_config.keys())
        
        # Standard paths always needed
        standard_paths = {
            'input_fasta', 'clean_fasta', 'substrate_analogs',
            'structures', 'embeddings', 'consolidated_results'
        }
        potential_paths.update(standard_paths)
        
        # From steps
        steps = self.get('steps', [])
        for step in steps:
            potential_paths.add(f"{step}_input")
            potential_paths.add(f"{step}_output") 
            potential_paths.add(f"{step}_results")
            potential_paths.add(f"{step}_model")
            
            if step.startswith('process_'):
                tool_name = step.replace('process_', '')
                potential_paths.add(f"{tool_name}_results")
                potential_paths.add(f"{tool_name}_output")
            
            if step.startswith('prepare_'):
                tool_name = step.replace('prepare_', '')
                potential_paths.add(f"{tool_name}_input")
                
            if step in ['esmfold', 'esm2']:
                potential_paths.update(['structures', 'embeddings'])
            elif step in ['geopoc', 'gatsol']:
                potential_paths.update(['structures', 'embeddings'])
        
        # Apply defaults for missing paths
        for path_key in potential_paths:
            if path_key not in self.paths:
                matched = False
                for pattern, path_func in path_patterns.items():
                    if re.match(pattern, path_key):
                        try:
                            generated_path = path_func(path_key)
                            self.paths[path_key] = generated_path
                            matched = True
                            break
                        except Exception as e:
                            print(f"Warning: Could not generate path for {path_key}: {e}")
                
                if not matched:
                    # Fallback: put it in output_dir
                    self.paths[path_key] = output_dir / path_key

    def _create_required_directories(self):
        """Create directories based on configured steps"""
        steps = self.get('steps', []) or []
        always_create = {'output_dir', 'results_dir', 'log_dir', 'modeldir'}
        
        # Determine required paths
        required_paths = set(always_create)
        
        for step in steps:
            if step == 'clean_sequences':
                required_paths.update(['input_fasta', 'clean_fasta'])
            elif step.startswith('prepare_'):
                tool_name = step.replace('prepare_', '')
                required_paths.update([
                    'clean_fasta', 'substrate_analogs', 
                    f'{tool_name}_input'
                ])
            elif step.startswith('process_'):
                tool_name = step.replace('process_', '')
                required_paths.update([
                    f'{tool_name}_output', 
                    f'{tool_name}_results',
                    'clean_fasta'
                ])
            elif step == 'consolidate_results':
                required_paths.add('consolidated_results')
            elif step in ['esmfold', 'esm2']:
                required_paths.update(['clean_fasta', 'structures' if step == 'esmfold' else 'embeddings'])
            else:
                required_paths.update([
                    'clean_fasta',
                    f'{step}_output',
                    f'{step}_results'
                ])
        
        # Create directories safely
        for path_key in required_paths:
            if path_key in self.paths:
                path = self.paths[path_key]
                if isinstance(path, Path) and not path.suffix:  # Directory, not file
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                        if os.geteuid() == 0:
                            os.chmod(path, 0o777)
                    except Exception as e:
                        print(f"Warning: Could not create directory {path}: {e}")

    def _setup_logging(self):
        """Setup logging directory"""
        log_dir = self.paths['log_dir']
        if isinstance(log_dir, Path):
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = log_dir / f"protscout_{self.get('condition', 'default')}_{timestamp}.log"
        else:
            # Fallback
            fallback_log = Path.cwd() / "protscout.log"
            self.log_file = fallback_log

    def _validate_config(self):
        """Validate required configuration parameters"""
        required_keys = ['workdir']
        missing = [k for k in required_keys if not self.get(k)]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
        
        if not self.get('workers'):
            self._config['workers'] = 2

    def get_path(self, path_key: str, default=None) -> Optional[Path]:
        """Safely get a path with fallback to default"""
        if path_key in self.paths:
            return self.paths[path_key]
        
        # Dynamic generation with safe fallbacks
        output_dir = self.paths.get('output_dir')
        results_dir = self.paths.get('results_dir')
        modeldir = self.paths.get('modeldir')
        
        if not output_dir:
            output_dir = Path.cwd() / 'artifacts'
        if not results_dir:
            results_dir = Path.cwd() / 'results'
        if not modeldir:
            modeldir = Path.cwd() / 'models'
        
        # Pattern-based path generation
        if path_key.endswith('_input'):
            tool_name = path_key.replace('_input', '')
            path = output_dir / f"{tool_name}_data"
        elif path_key.endswith('_output'):
            tool_name = path_key.replace('_output', '')
            path = output_dir / tool_name
        elif path_key.endswith('_results'):
            path = results_dir / path_key
        elif path_key.endswith('_model'):
            tool_name = path_key.replace('_model', '')
            path = modeldir / tool_name
        elif path_key in ['structures', 'embeddings']:
            path = output_dir / path_key
        elif default:
            path = self._safe_path(default, output_dir / path_key)
        else:
            path = output_dir / path_key
            
        # Cache and return
        self.paths[path_key] = path
        return path

    def get(self, key: str, default=None):
        return self._config.get(key, default)

    def __getattr__(self, key):
        if key in self._config:
            return self._config[key]
        raise AttributeError(f"Config has no attribute '{key}'")
    
    def get_tool_paths(self, tool_name: str) -> Dict[str, Path]:
        """Get all paths related to a specific tool"""
        tool_paths = {}
        for key, path in self.paths.items():
            if key.startswith(f"{tool_name}_"):
                tool_paths[key] = path
        return tool_paths
    
    def list_available_tools(self) -> List[str]:
        """List all tools that have paths defined"""
        tools = set()
        for key in self.paths.keys():
            for suffix in ['_input', '_output', '_results', '_model', '_models']:
                if key.endswith(suffix):
                    tool_name = key.replace(suffix, '')
                    tools.add(tool_name)
        return sorted(list(tools))
    
    def debug_paths(self):
        """Print all configured paths for debugging"""
        print("=== CONFIGURED PATHS ===")
        for key in sorted(self.paths.keys()):
            path = self.paths[key]
            path_type = type(path).__name__
            exists = "✅" if isinstance(path, Path) and path.exists() else "📁"
            print(f"{exists} {key:<25} -> {path} ({path_type})")
        
        print(f"\n=== DETECTED TOOLS ===")
        tools = self.list_available_tools()
        for tool in tools:
            print(f"🔧 {tool}")
            tool_paths = self.get_tool_paths(tool)
            for path_key, path_value in tool_paths.items():
                exists = "✅" if isinstance(path_value, Path) and path_value.exists() else "📁"
                print(f"   {exists} {path_key} -> {path_value}")