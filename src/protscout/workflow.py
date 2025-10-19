import subprocess
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import signal
import shutil

from .utils import format_duration, check_docker_running, get_gpu_info

class StepResult:
    """Container for step execution results"""
    def __init__(self, name: str, success: bool, duration: float, 
                 output: str = "", error: str = "", metrics: Dict = None):
        self.name = name
        self.success = success
        self.duration = duration
        self.output = output
        self.error = error
        self.metrics = metrics or {}
        self.timestamp = datetime.now()

class ProtScoutWorkflow:
    def __init__(self, config, resume: bool = False):
        self.config = config
        self.resume = resume
        self.logger = self._setup_logging()
        self.scripts_dir = Path(__file__).parent / "scripts"
        
        # Debug configuration loading
        self._debug_config()
        
        # Dynamic path resolution - this is the key fix!
        self._setup_dynamic_paths()
        
        self.step_commands = self._define_steps()
        self.results = {}
        self.state_file = self._get_safe_path('output_dir') / '.workflow_state.json'
        self.start_time = None
        self._interrupted = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Check prerequisites
        self._check_prerequisites()
        
        # Clear artifacts directory if needed
        self._handle_artifacts_cleanup()
        
        # Load previous state if resuming
        if resume:
            self._load_state()
    
    def _debug_config(self):
        """Debug configuration loading"""
        self.logger.info("=== Configuration Debug ===")
        self.logger.info(f"Config type: {type(self.config)}")
        
        if hasattr(self.config, 'paths'):
            self.logger.info(f"Config has paths attribute: True")
            if hasattr(self.config.paths, 'keys'):
                self.logger.info(f"Available path keys: {list(self.config.paths.keys())}")
            else:
                self.logger.info(f"Paths content: {self.config.paths}")
        else:
            self.logger.error("Config does not have paths attribute!")
    
    def _setup_dynamic_paths(self):
        """Setup dynamic path resolution to handle any tool names"""
        self.logger.info("Setting up dynamic path resolution...")
        
        # Create a dynamic path resolver that maps tool names
        self.path_resolver = {}
        
        # Get all available paths from config
        available_paths = set()
        if hasattr(self.config, 'paths'):
            if hasattr(self.config.paths, 'keys'):
                available_paths.update(self.config.paths.keys())
            elif isinstance(self.config.paths, dict):
                available_paths.update(self.config.paths.keys())
        
        # Get steps to understand what tools are being used
        steps = self.config.get('steps', [])
        
        # Build tool name mappings (no automatic cross-tool mapping)
        tool_mappings = {}
        for step in steps:
            if step.startswith('prepare_'):
                tool_name = step.replace('prepare_', '')
                tool_mappings[tool_name] = tool_name
            elif step.startswith('process_'):
                tool_name = step.replace('process_', '')
                tool_mappings[tool_name] = tool_name
            elif step not in ['clean_sequences', 'consolidate_results']:
                tool_mappings[step] = step
        
        self.tool_mappings = tool_mappings
        self.logger.info(f"Tool mappings: {tool_mappings}")
        self.logger.info(f"Available paths: {sorted(available_paths)}")
        
        for tool in tool_mappings:
            tool_paths = [p for p in available_paths if p.startswith(f"{tool}_")]
            if not tool_paths:
                self.logger.warning(f"No paths configured for tool '{tool}'. Steps using this tool may fail or use default paths.")
    
    def _get_safe_path(self, path_key: str) -> Optional[Path]:
        """Safely get a path with dynamic tool name resolution"""
        # First try direct access
        path = self._try_direct_path_access(path_key)
        if path:
            return path
        
        # Try tool name mapping
        for prefix in ['_input', '_output', '_results', '_model']:
            if path_key.endswith(prefix):
                tool_name = path_key.replace(prefix, '')
                if tool_name in self.tool_mappings:
                    mapped_tool = self.tool_mappings[tool_name]
                    mapped_path_key = f"{mapped_tool}{prefix}"
                    path = self._try_direct_path_access(mapped_path_key)
                    if path:
                        self.logger.info(f"Mapped {path_key} -> {mapped_path_key}")
                        return path
        
        # Try using config's get_path method if available
        if hasattr(self.config, 'get_path'):
            try:
                path = self.config.get_path(path_key)
                if path:
                    return Path(path) if isinstance(path, str) else path
            except Exception as e:
                self.logger.debug(f"Config.get_path failed for {path_key}: {e}")
        
        # Generate intelligent default
        return self._generate_default_path(path_key)
    
    def _try_direct_path_access(self, path_key: str) -> Optional[Path]:
        """Try multiple methods to access a path directly"""
        try:
            # Method 1: config.paths attribute access
            if hasattr(self.config, 'paths'):
                if hasattr(self.config.paths, path_key):
                    value = getattr(self.config.paths, path_key)
                    return Path(value) if value else None
                elif hasattr(self.config.paths, 'get'):
                    value = self.config.paths.get(path_key)
                    return Path(value) if value else None
                elif isinstance(self.config.paths, dict) and path_key in self.config.paths:
                    value = self.config.paths[path_key]
                    return Path(value) if value else None
            
            # Method 2: config dict access
            if isinstance(self.config, dict) and 'paths' in self.config:
                if path_key in self.config['paths']:
                    value = self.config['paths'][path_key]
                    return Path(value) if value else None
            
            # Method 3: direct config attribute
            if hasattr(self.config, path_key):
                value = getattr(self.config, path_key)
                return Path(value) if value else None
                
        except Exception as e:
            self.logger.debug(f"Direct path access failed for {path_key}: {e}")
        
        return None
    
    def _generate_default_path(self, path_key: str) -> Path:
        """Generate intelligent default path when not found in config"""
        # Get base directories
        output_dir = self._try_direct_path_access('output_dir') or Path.cwd() / 'artifacts'
        results_dir = self._try_direct_path_access('results_dir') or Path.cwd() / 'results'
        modeldir = self._try_direct_path_access('modeldir') or Path.cwd() / 'models'
        
        # Generate based on patterns
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
        elif path_key in ['structures', 'embeddings', 'clean_fasta']:
            path = output_dir / path_key
        elif path_key in ['output_dir']:
            path = output_dir
        elif path_key in ['results_dir']:
            path = results_dir
        else:
            path = output_dir / path_key
        
        self.logger.info(f"Generated default path for {path_key}: {path}")
        return path
    
    def _handle_artifacts_cleanup(self):
        """Handle artifacts directory cleanup"""
        preserve = self.config.get('preserve_artifacts', False)
        artifacts_dir = self._get_safe_path('output_dir')
        
        if not self.resume and not preserve and artifacts_dir and artifacts_dir.exists():
            for item in artifacts_dir.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception:
                    self.logger.warning(f"Could not remove artifact: {item}")
            self.logger.info(f"Cleared artifacts directory: {artifacts_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        self.logger.warning("Interrupt received, saving state...")
        self._interrupted = True
        self._save_state()
        raise KeyboardInterrupt("Workflow interrupted by user")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logger = logging.getLogger('ProtScout')
        logger.setLevel(logging.DEBUG)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console.setFormatter(console_format)
        
        # File handler
        log_file = None
        if hasattr(self.config, 'log_file'):
            log_file = self.config.log_file
        else:
            log_dir = self._try_direct_path_access('log_dir')
            if log_dir:
                log_file = log_dir / 'protscout.log'
            else:
                log_file = Path.cwd() / 'protscout.log'
        
        if log_file:
            try:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_format = logging.Formatter(
                    '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
                )
                file_handler.setFormatter(file_format)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not setup file logging to {log_file}: {e}")
        
        logger.addHandler(console)
        return logger
    
    def _check_prerequisites(self):
        """Check system prerequisites"""
        self.logger.info("Checking prerequisites...")
        
        # Check Docker
        if not check_docker_running():
            raise RuntimeError("Docker is not running. Please start Docker daemon.")
        
        # Check GPU availability
        gpu_info = get_gpu_info()
        if gpu_info:
            self.logger.info(f"GPU available: {gpu_info}")
        else:
            self.logger.warning("No GPU detected. Some steps may run slower.")
        
        # Check disk space
        output_path = self._get_safe_path('output_dir') or Path.cwd()
        try:
            free_space = psutil.disk_usage(output_path).free / (1024**3)
            if free_space < 50:
                self.logger.warning(f"Low disk space: {free_space:.1f} GB free")
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
    
    def _define_steps(self) -> Dict[str, List[str]]:
        """Define command mappings for each step with dynamic path resolution"""
        containers = self.config.get('containers', {})
        resources = self.config.get('resources', {})
        
        # Python executable
        python_exe = self.config.get('python_executable', 'python')
        
        # Helper function to safely get paths for commands
        def safe_path_str(path_key: str, default: str = "") -> str:
            path = self._get_safe_path(path_key)
            return str(path) if path else default
        
        # Get clean_sequences config
        clean_seq_config = self.config.get('clean_sequences', {})
        clean_seq_cmd = [
            python_exe, str(self.scripts_dir / 'preprocess/clean_sequences.py'),
            '--input_dir', safe_path_str('input_fasta'),
            '--output_dir', safe_path_str('clean_fasta'),
        ]
        if clean_seq_config.get('remove_duplicates', False):
            clean_seq_cmd.append('--remove_duplicates')
        # Add prefix parameter
        prefix = clean_seq_config.get('prefix', 'cleaned_')
        clean_seq_cmd.extend(['--prefix', prefix if prefix is not None else ''])

        steps = {
            'clean_sequences': clean_seq_cmd,

            'esmfold': [
                'sudo', 'bash', str(self.scripts_dir / 'esm/run_esmfold.sh'),
                '--input', safe_path_str('clean_fasta'),
                '--output', safe_path_str('structures'),
                '--memory', resources.get('memory', '16g'),
                '--max-containers', str(containers.get('esmfold', {}).get('max_containers', 1)),
                '--docker-image', containers.get('esmfold', {}).get('image', 'ghcr.io/robaina/protscout-tools-esmfold:latest')
            ],

            'esm2': [
                'sudo', 'bash', str(self.scripts_dir / 'esm/run_esm2.sh'),
                '--input', safe_path_str('clean_fasta'),
                '--output', safe_path_str('embeddings'),
                '--memory', resources.get('memory', '16g'),
                '--model', containers.get('esm2', {}).get('model', 'esm2_t36_3B_UR50D'),
                '--toks_per_batch', str(containers.get('esm2', {}).get('toks_per_batch', 4096)),
                '--include', containers.get('esm2', {}).get('include', 'per_tok'),
                '--max-containers', str(containers.get('esm2', {}).get('max_containers', 1)),
                '--docker-image', containers.get('esm2', {}).get('image', 'ghcr.io/robaina/protscout-tools-esm2:latest')
            ],

            'remove_sequences_without_pdb': [
                python_exe, str(self.scripts_dir / 'preprocess/remove_sequences_without_pdb.py'),
                '--faa_dir', safe_path_str('clean_fasta'),
                '--pdb_dir', safe_path_str('structures'),
                '--output_dir', safe_path_str('clean_fasta')
            ],

            'temberture': [
                'sudo', 'bash', str(self.scripts_dir / 'temberture/run_temberture.sh'),
                '--input', safe_path_str('clean_fasta'),
                '--output', safe_path_str('temberture_output'),
                '--parallel', str(self.config.get('workers', 2)),
                '--docker-image', containers.get('temberture', {}).get('image', 'ghcr.io/robaina/protscout-tools-temberture:latest'),
                '--gpus', resources.get('gpus', 'all'),
                '--shm-size', resources.get('shm_size', '100g')
            ],

            'geopoc': [
                'sudo', 'bash', str(self.scripts_dir / 'geopoc/run_geopoc.sh'),
                '--input', safe_path_str('clean_fasta'),
                '--output', safe_path_str('geopoc_output'),
                '--model', safe_path_str('geopoc_model', str(self._get_safe_path('modeldir') / 'geopoc')),
                '--esm', safe_path_str('modeldir'),
                '--structures', safe_path_str('structures'),
                '--embeddings', safe_path_str('embeddings'),
                '--tasks', containers.get('geopoc', {}).get('tasks', 'temp,pH,salt'),
                '--parallel', str(self.config.get('workers', 2)),
                '--docker-image', containers.get('geopoc', {}).get('image', 'ghcr.io/robaina/protscout-tools-geopoc:latest'),
                '--gpus', resources.get('gpus', 'all'),
                '--shm-size', resources.get('shm_size', '100g')
            ],

            'classical_properties': [
                python_exe, str(self.scripts_dir / 'preprocess/predict_classic_properties.py'),
                '--input', safe_path_str('clean_fasta'),
                '--output', safe_path_str('classical_properties_output')
            ],
            
            'prepare_catpred': [
                python_exe, str(self.scripts_dir / 'catpred/prepare_catpred_inputs.py'),
                '--fasta_dir', safe_path_str('clean_fasta'),
                '--substrate_tsv', safe_path_str('substrate_analogs'),
                '--output_dir', safe_path_str('catpred_input'),
                '--substrate_id', self.config.get('substrate_id', '') or ''
            ],
            
            'prepare_catapro': [
                python_exe, str(self.scripts_dir / 'catapro/prepare_catapro_inputs.py'),
                '--fasta_dir', safe_path_str('clean_fasta'),
                '--substrate_tsv', safe_path_str('substrate_analogs'),
                '--output_dir', safe_path_str('catapro_input'),
                '--substrate_id', self.config.get('substrate_id', '') or ''
            ],
            
            'catpred': [
                'sudo', 'bash', str(self.scripts_dir / 'catpred/run_catpred.sh'),
                '--input', safe_path_str('catpred_input'),
                '--output', safe_path_str('catpred_output'),
                '--model', safe_path_str('catpred_model', str(self._get_safe_path('modeldir') / 'catpred')),
                '--parallel', str(self.config.get('workers', 2)),
                '--docker-image', containers.get('catpred', {}).get('image', 'ghcr.io/robaina/catpred:latest'),
                '--gpus', resources.get('gpus', 'all'),
                '--shm-size', resources.get('shm_size', '100g')
            ],
            
            'catapro': [
                'sudo', 'bash', str(self.scripts_dir / 'catapro/run_catapro.sh'),
                '--input', safe_path_str('catapro_input'),
                '--output', safe_path_str('catapro_output'),
                '--model', safe_path_str('catapro_model', str(self._get_safe_path('modeldir') / 'catapro')),
                '--parallel', str(self.config.get('workers', 2)),
                '--docker-image', containers.get('catapro', {}).get('image', 'ghcr.io/robaina/catapro:latest'),
                '--gpus', resources.get('gpus', 'all'),
                '--shm-size', resources.get('shm_size', '100g')
            ],
            
            'temstapro': [
                'sudo', 'bash', str(self.scripts_dir / 'temstapro/run_temstapro.sh'),
                '--input', safe_path_str('clean_fasta'),
                '--output', safe_path_str('temstapro_output'),
                '--model', safe_path_str('temstapro_model', str(self._get_safe_path('modeldir') / 'temstapro')),
                '--parallel', str(self.config.get('workers', 2)),
                '--docker-image', containers.get('temstapro', {}).get('image', 'ghcr.io/robaina/temstapro:latest'),
                '--gpus', resources.get('gpus', 'all'),
                '--shm-size', resources.get('shm_size', '100g')
            ],
            
            'process_catpred': [
                python_exe, str(self.scripts_dir / 'catpred/process_catpred.py'),
                '--input_dir', safe_path_str('catpred_output'),
                '--faa_dir', safe_path_str('clean_fasta'),
                '--output_dir', safe_path_str('catpred_results'),
                '--catpred_input_dir', safe_path_str('catpred_input')
            ],
            
            'process_catapro': [
                python_exe, str(self.scripts_dir / 'catapro/process_catapro.py'),
                '--input_dir', safe_path_str('catapro_output'),
                '--faa_dir', safe_path_str('clean_fasta'),
                '--output_dir', safe_path_str('catapro_results'),
                '--catapro_input_dir', safe_path_str('catapro_input')
            ],
            
            'process_temstapro': [
                python_exe, str(self.scripts_dir / 'temstapro/process_temstapro.py'),
                '--input_dir', safe_path_str('temstapro_output'),
                '--faa_dir', safe_path_str('clean_fasta'),
                '--output_dir', safe_path_str('temstapro_results'),
                '--temstapro_input_dir', safe_path_str('temstapro_input', safe_path_str('temstapro_output')),
                '--method', self.config.get('temstapro_method', 'tm_interpolation')
            ],

            'process_temberture': [
                python_exe, str(self.scripts_dir / 'temberture/process_temberture.py'),
                '--input_dir', safe_path_str('temberture_output'),
                '--output_dir', safe_path_str('temberture_results')
            ],

            'process_geopoc': [
                python_exe, str(self.scripts_dir / 'geopoc/process_geopoc.py'),
                '--input_dir', safe_path_str('geopoc_output'),
                '--output_dir', safe_path_str('geopoc_results')
            ],

            'consolidate_results': [
                python_exe, str(self.scripts_dir / 'preprocess/make_output_tables.py'),
                '--input_dir', safe_path_str('results_dir'),
                '--output_dir', safe_path_str('consolidated_results'),
                '--tools', 'temberture', 'geopoc', 'classical_properties'
            ],
        }
        
        # Add quiet flag if specified
        if self.config.get('quiet', False):
            for step_name, cmd in steps.items():
                if step_name in ['catpred', 'catapro', 'temstapro', 'temberture', 'geopoc']:
                    cmd.append('--quiet')
        
        # Log commands for debugging
        self.logger.debug("=== Generated Step Commands ===")
        for step_name, cmd in steps.items():
            self.logger.debug(f"{step_name}: {' '.join(cmd[:3])}...")
            # Check for empty paths in the command
            for i, arg in enumerate(cmd):
                if not arg or arg == "None":
                    self.logger.warning(f"Empty argument in {step_name} command at position {i}")
        
        return steps
    
    def _count_input_files(self, directory_path: Optional[Path], pattern: str = "*.faa") -> int:
        """Count input files for progress tracking"""
        if directory_path and directory_path.exists():
            return len(list(directory_path.glob(pattern)))
        return 0
    
    def _is_warning_or_info_line(self, line: str) -> bool:
        """Check if a line is a warning or informational message"""
        line_lower = line.lower()
        warning_patterns = [
            'warning:', 'futurewarning:', 'deprecationwarning:', 'userwarning:',
            'runtimewarning:', 'pendingdeprecationwarning:', 'importwarning:',
            'will not inherit from', 'get rid of this warning by', 
            'weights_only=false', 'torch.load', 'experimental feature',
            'failed with exit code'
        ]
        progress_patterns = [
            '%|', 'it/s]', '/s]', '[00:00<', '██████████', 'completed in',
            'successful:', 'found', 'output files', 'analysis complete'
        ]
        return any(pat in line_lower for pat in warning_patterns + progress_patterns)

    def _is_real_error(self, line: str) -> bool:
        """Check if a line represents a real error"""
        if self._is_warning_or_info_line(line):
            return False
        
        line_lower = line.lower()
        error_patterns = [
            'error:', 'exception:', 'traceback (most recent call last):',
            'fatal:', 'critical:', 'abort', 'segmentation fault',
            'killed', 'out of memory', 'no space left on device',
            'permission denied', 'file not found', 'cannot open',
            'failed to', 'unable to', 'could not'
        ]
        return any(pat in line_lower for pat in error_patterns)
    
    def run_step(self, step_name: str, retry: int = 0) -> StepResult:
        """Run a single workflow step with comprehensive error handling"""
        if step_name not in self.step_commands:
            self.logger.error(f"Unknown step: {step_name}")
            return StepResult(step_name, False, 0, error="Unknown step")
        
        cmd = self.step_commands[step_name]
        start_time = time.time()
        
        # Count input files for progress context
        input_count = 0
        if step_name == 'clean_sequences':
            input_count = self._count_input_files(self._get_safe_path('input_fasta'))
        elif step_name in ['catpred', 'catapro', 'temstapro', 'esmfold', 'esm2', 'temberture', 'geopoc']:
            input_count = self._count_input_files(self._get_safe_path('clean_fasta'))
        
        if input_count > 0:
            self.logger.info(f"Processing {input_count} input files")
        
        # Log the actual command being run
        self.logger.info(f"Running command: {' '.join(cmd[:5])}...")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            error_lines = []
            warning_lines = []
            real_error_lines = []
            
            # Stream output in real-time
            for line in process.stdout:
                line = line.strip()
                if line:
                    output_lines.append(line)
                    
                    if self._is_warning_or_info_line(line):
                        warning_lines.append(line)
                        if not self.config.get('quiet', False):
                            self.logger.debug(f"  INFO/WARNING: {line}")
                    elif self._is_real_error(line):
                        real_error_lines.append(line)
                        self.logger.error(f"  ERROR: {line}")
                    elif any(keyword in line.lower() for keyword in ['complete', 'done', 'success', 'found', 'output']):
                        self.logger.info(f"  {line}")
            
            # Wait for completion and capture stderr
            _, stderr = process.communicate()
            
            if stderr:
                for line in stderr.strip().split('\n'):
                    if line.strip():
                        error_lines.append(line)
                        if self._is_warning_or_info_line(line):
                            warning_lines.append(line)
                            if not self.config.get('quiet', False):
                                self.logger.debug(f"  STDERR INFO/WARNING: {line}")
                        elif self._is_real_error(line):
                            real_error_lines.append(line)
                            self.logger.error(f"  STDERR ERROR: {line}")
            
            duration = time.time() - start_time
            
            # Check if output was created for container-based tools
            output_created = False
            output_file_count = 0
            if step_name in ['catpred', 'catapro', 'temstapro', 'temberture', 'geopoc', 'esmfold', 'esm2']:
                output_dir = self._get_safe_path(f'{step_name}_output')
                if not output_dir:
                    # For esmfold and esm2, use alternate path keys
                    if step_name == 'esmfold':
                        output_dir = self._get_safe_path('structures')
                    elif step_name == 'esm2':
                        output_dir = self._get_safe_path('embeddings')

                if output_dir and output_dir.exists():
                    if step_name in ['catpred', 'catapro']:
                        result_files = list(output_dir.glob('**/*.csv'))
                    elif step_name in ['temstapro', 'temberture', 'geopoc']:
                        result_files = list(output_dir.glob('**/*.tsv'))
                    elif step_name == 'esmfold':
                        result_files = list(output_dir.glob('**/*.pdb'))
                    elif step_name == 'esm2':
                        result_files = list(output_dir.glob('**/*.pt'))
                    else:
                        result_files = []

                    output_file_count = len(result_files)
                    output_created = output_file_count > 0

                    if output_created:
                        self.logger.info(f"  Found {output_file_count} output files")
            
            # Determine success
            success = (process.returncode == 0) or (output_created and len(real_error_lines) == 0)
            
            if success:
                if process.returncode != 0:
                    self.logger.info(
                        f"Step '{step_name}' exited with code {process.returncode} but created "
                        f"{output_file_count} output files and had no real errors. Treating as success."
                    )
                if warning_lines:
                    self.logger.info(f"  Note: {len(warning_lines)} warnings/info messages were emitted")
                    
                self.logger.info(f"✓ Step '{step_name}' completed in {format_duration(duration)}")
                return StepResult(
                    step_name, True, duration,
                    output='\n'.join(output_lines),
                    metrics={
                        'input_files': input_count,
                        'output_files': output_file_count,
                        'warnings': len(warning_lines),
                        'real_errors': len(real_error_lines)
                    }
                )
            else:
                # Build error message from real errors only
                error_msg = ""
                if real_error_lines:
                    error_msg = '\n'.join(real_error_lines[:10])
                elif not output_created and step_name in ['catpred', 'catapro', 'temstapro', 'temberture', 'geopoc', 'esmfold', 'esm2']:
                    error_msg = f"No output files created and process exited with code {process.returncode}"
                else:
                    error_msg = f"Process failed with exit code {process.returncode}"
                
                self.logger.error(f"✗ Step '{step_name}' failed after {format_duration(duration)}")
                self.logger.error(f"  Exit code: {process.returncode}")
                self.logger.error(f"  Real errors found: {len(real_error_lines)}")
                
                # Retry logic
                if retry < self.config.get('max_retries', 2):
                    self.logger.info(f"  Retrying step '{step_name}' (attempt {retry + 2})...")
                    time.sleep(5)
                    return self.run_step(step_name, retry + 1)
                
                return StepResult(step_name, False, duration, error=error_msg)
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Exception in step '{step_name}': {str(e)}")
            import traceback
            self.logger.error(f"  Traceback: {traceback.format_exc()}")
            return StepResult(step_name, False, duration, error=str(e))
    
    def _save_state(self):
        """Save workflow state for resume capability"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'condition': getattr(self.config, 'condition', 'unknown'),
            'completed_steps': list(self.results.keys()),
            'step_results': {
                name: {
                    'success': result.success,
                    'duration': result.duration,
                    'timestamp': result.timestamp.isoformat()
                }
                for name, result in self.results.items()
            }
        }
        
        state_file = self._get_safe_path('output_dir') / '.workflow_state.json'
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            self.logger.debug(f"State saved to {state_file}")
        except Exception as e:
            self.logger.error(f"Could not save state: {e}")
    
    def _load_state(self):
        """Load previous workflow state"""
        state_file = self._get_safe_path('output_dir') / '.workflow_state.json'
        
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                
                config_condition = getattr(self.config, 'condition', 'unknown')
                if state.get('condition') != config_condition:
                    self.logger.warning(
                        f"State condition '{state.get('condition')}' doesn't match "
                        f"current condition '{config_condition}'. Starting fresh."
                    )
                    return
                
                for step_name in state.get('completed_steps', []):
                    result_data = state['step_results'].get(step_name, {})
                    self.results[step_name] = StepResult(
                        step_name,
                        result_data.get('success', True),
                        result_data.get('duration', 0)
                    )
                
                self.logger.info(f"Resumed from previous state: {len(self.results)} steps completed")
            except Exception as e:
                self.logger.error(f"Could not load previous state: {e}")
    
    def run(self, steps: Optional[List[str]] = None):
        """Run the complete workflow"""
        self.start_time = time.time()
        steps_to_run = steps or self.config.get('steps', self._get_default_steps())
        
        if self.resume:
            steps_to_run = [s for s in steps_to_run if s not in self.results]
            if not steps_to_run:
                self.logger.info("All steps already completed!")
                return
        
        total_steps = len(steps_to_run)
        self.logger.info(f"Starting workflow with {total_steps} steps")
        self.logger.info(f"Condition: {getattr(self.config, 'condition', 'unknown')}")
        self.logger.info(f"Output directory: {self._get_safe_path('output_dir')}")
        
        completed = len(self.results)
        
        try:
            for step in steps_to_run:
                if self._interrupted:
                    break
                
                completed += 1
                self.logger.info(f"\n[{completed}/{total_steps}] Running step: {step}")
                self.logger.info("=" * 60)
                
                result = self.run_step(step)
                self.results[step] = result
                
                if not result.success:
                    self._save_state()
                    self.logger.error(f"Workflow failed at step: {step}")
                    self._print_summary()
                    raise RuntimeError(f"Step {step} failed")
                
                self._save_state()
            
            self._print_summary()
            self.logger.info("✅ Workflow completed successfully!")
            
        except KeyboardInterrupt:
            self.logger.warning("\n⚠️ Workflow interrupted by user")
            self._print_summary()
            raise
        except Exception as e:
            self.logger.error(f"\n❌ Workflow failed: {str(e)}")
            self._print_summary()
            raise
    
    def _get_default_steps(self) -> List[str]:
        """Get default step order"""
        return [
            'clean_sequences',
            'prepare_catapro',
            'catapro',
            'temstapro',
            'process_temstapro',
            'process_catapro',
            'consolidate_results'
        ]
    
    def _print_summary(self):
        """Print workflow execution summary"""
        if not self.results:
            return
        
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("WORKFLOW SUMMARY")
        self.logger.info("=" * 60)
        
        for step_name, result in self.results.items():
            status = "✓" if result.success else "✗"
            self.logger.info(
                f"{status} {step_name:<30} {format_duration(result.duration):>10}"
            )
        
        successful = sum(1 for r in self.results.values() if r.success)
        failed = len(self.results) - successful
        
        self.logger.info("-" * 60)
        self.logger.info(f"Total steps: {len(self.results)}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Total duration: {format_duration(total_duration)}")
        
        log_file = getattr(self.config, 'log_file', None) 
        if log_file:
            self.logger.info(f"Log file: {log_file}")
        
        if successful == len(self.results):
            consolidated_path = self._get_safe_path('consolidated_results')
            if consolidated_path:
                self.logger.info(f"Results: {consolidated_path}")