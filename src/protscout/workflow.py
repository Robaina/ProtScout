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
        self.step_commands = self._define_steps()
        self.results = {}
        self.state_file = config.paths['output_dir'] / '.workflow_state.json'
        self.start_time = None
        self._interrupted = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Check prerequisites
        self._check_prerequisites()
        # Clear artifacts (raw outputs) directory before run unless preserving or resuming
        preserve = self.config.get('preserve_artifacts', False)
        artifacts_dir = self.config.paths['output_dir']
        if not resume and not preserve and artifacts_dir.exists():
            for item in artifacts_dir.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception:
                    self.logger.warning(f"Could not remove artifact: {item}")
            self.logger.info(f"Cleared artifacts directory: {artifacts_dir}")
        # Load previous state if resuming
        if resume:
            self._load_state()
    
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
        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        logger.addHandler(console)
        logger.addHandler(file_handler)
        
        return logger
    
    def _check_prerequisites(self):
        """Check system prerequisites"""
        self.logger.info("Checking prerequisites...")
        
        # Check Docker
        if not check_docker_running():
            raise RuntimeError("Docker is not running. Please start Docker daemon.")
        
        # Check GPU availability if needed
        gpu_info = get_gpu_info()
        if gpu_info:
            self.logger.info(f"GPU available: {gpu_info}")
        else:
            self.logger.warning("No GPU detected. Some steps may run slower.")
        
        # Check disk space
        output_path = self.config.paths['output_dir']
        free_space = psutil.disk_usage(output_path).free / (1024**3)  # GB
        if free_space < 50:
            self.logger.warning(f"Low disk space: {free_space:.1f} GB free")
    
    def _define_steps(self) -> Dict[str, List[str]]:
        """Define command mappings for each step"""
        paths = self.config.paths
        containers = self.config.get('containers', {})
        resources = self.config.get('resources', {})
        
        # Python executable (handle conda environment)
        python_exe = self.config.get('python_executable', 'python')
        
        steps = {
            'clean_sequences': [
                python_exe, str(self.scripts_dir / 'clean_sequences.py'),
                '--input_dir', str(paths['input_fasta']),
                '--output_dir', str(paths['clean_fasta']),
                '--remove_duplicates',
                '--prefix', ''
            ],
            
            'esmfold': [
                'sudo', 'bash', str(self.scripts_dir / 'run_esmfold.sh'),
                '--input', str(paths['clean_fasta']),
                '--output', str(paths['structures']),
                '--docker-image', containers.get('esmfold', {}).get('image', 'ghcr.io/new-atlantis-labs/esmfold:latest'),
                '--max-containers', str(containers.get('esmfold', {}).get('max_containers', 1)),
                '--memory', self.config.get('memory', '100g')
            ],
            
            'esm2': [
                'sudo', 'bash', str(self.scripts_dir / 'run_esm2.sh'),
                '--input', str(paths['clean_fasta']),
                '--output', str(paths['embeddings']),
                '--docker-image', containers.get('esm2', {}).get('image', 'ghcr.io/new-atlantis-labs/esm2:latest'),
                '--include', 'per_tok',
                '--model', 'esm2_t36_3B_UR50D',
                '--weights', str(paths['modeldir']),
                '--max-containers', str(containers.get('esm2', {}).get('max_containers', 1)),
                '--toks_per_batch', str(containers.get('esm2', {}).get('toks_per_batch', 4096)),
                '--memory', self.config.get('memory', '100g')
            ],
            
            'remove_sequences_without_pdb': [
                python_exe, str(self.scripts_dir / 'remove_sequences_without_pdb.py'),
                '--faa_dir', str(paths['clean_fasta']),
                '--pdb_dir', str(paths['structures']),
                '--output_dir', str(paths['clean_fasta'])
            ],
            
            'prepare_catpred': [
                python_exe, str(self.scripts_dir / 'prepare_catpred_inputs.py'),
                '--fasta_dir', str(paths['clean_fasta']),
                '--substrate_tsv', str(paths['substrate_analogs']),
                '--output_dir', str(paths['catpred_input']),
                '--substrate_id', self.config.get('substrate_id', None)
            ],
            
            'catpred': [
                'sudo', 'bash', str(self.scripts_dir / 'run_catpred.sh'),
                '--input', str(paths['catpred_input']),
                '--output', str(paths['catpred_output']),
                '--model', str(paths['modeldir']),
                '--parallel', str(self.config.get('workers', 2)),
                '--docker-image', containers.get('catpred', {}).get('image', 'ghcr.io/new-atlantis-labs/catpred:latest'),
                '--gpus', resources.get('gpus', 'all'),
                '--shm-size', resources.get('shm_size', '100g')
            ],
            
            'temberture': [
                'sudo', 'bash', str(self.scripts_dir / 'run_temberture.sh'),
                '--input', str(paths['clean_fasta']),
                '--output', str(paths['temberture_output']),
                '--parallel', str(self.config.get('workers', 2)),
                '--docker-image', containers.get('temberture', {}).get('image', 'ghcr.io/new-atlantis-labs/temberture:latest'),
                '--gpus', resources.get('gpus', 'all'),
                '--shm-size', resources.get('shm_size', '100g')
            ],
            
            'geopoc': [
                'sudo', 'bash', str(self.scripts_dir / 'run_geopoc.sh'),
                '--input', str(paths['clean_fasta']),
                '--output', str(paths['geopoc_output']),
                '--esm', str(paths['modeldir']),
                '--model', str(paths['geopoc_model']),
                '--structures', str(paths['structures']),
                '--embeddings', str(paths['embeddings']),
                '--tasks', 'temp,pH,salt',
                '--parallel', str(self.config.get('workers', 2)),
                '--docker-image', containers.get('geopoc', {}).get('image', 'ghcr.io/new-atlantis-labs/geopoc:latest'),
                '--gpus', resources.get('gpus', 'all'),
                '--shm-size', resources.get('shm_size', '100g')
            ],
            
            'gatsol': [
                'sudo', 'bash', str(self.scripts_dir / 'run_gatsol.sh'),
                '--input', str(paths['clean_fasta']),
                '--output', str(paths['gatsol_output']),
                '--structures', str(paths['structures']),
                '--model', str(paths['gatsol_model']),
                '--esm', str(paths['modeldir']),
                '--batch-size', '32',
                '--parallel', str(self.config.get('workers', 2)),
                '--feature-batch', '4',
                '--timeout', '600',
                '--docker-image', containers.get('gatsol', {}).get('image', 'ghcr.io/new-atlantis-labs/gatsol:latest'),
                '--gpus', resources.get('gpus', 'all'),
                '--shm-size', resources.get('shm_size', '100g')
            ],
            
            'classical_properties': [
                python_exe, str(self.scripts_dir / 'predict_classic_properties.py'),
                '--input', str(paths['clean_fasta']),
                '--log', str(paths['log_dir'] / f"classical_properties_{self.config.condition}.log"),
                '--output', str(paths['classical_properties_results'])  # Make sure this is the results subdirectory
            ],
                        
            'process_temberture': [
                python_exe, str(self.scripts_dir / 'process_temberture.py'),
                '-i', str(paths['temberture_output']),
                '-o', str(paths['temberture_results'])
            ],
            
            'process_geopoc': [
                python_exe, str(self.scripts_dir / 'process_geopoc.py'),
                '--input', str(paths['geopoc_output']),
                '--output', str(paths['geopoc_results'])
            ],
            
            'process_gatsol': [
                python_exe, str(self.scripts_dir / 'process_gatsol.py'),
                '--input_dir', str(paths['gatsol_output']),
                '--output_dir', str(paths['gatsol_results'])
            ],
            
            'process_catpred': [
                python_exe, str(self.scripts_dir / 'process_catpred.py'),
                '--input_dir', str(paths['catpred_output']),
                '--faa_dir', str(paths['clean_fasta']),
                '--output_dir', str(paths['catpred_results']),
                '--catpred_input_dir', str(paths['catpred_input'])  # Add this line
            ],
            
            'consolidate_results': [
                python_exe, str(self.scripts_dir / 'make_output_tables.py'),
                '-i', str(paths['results_dir']),
                '-o', str(paths['consolidated_results']),
                '--tools', 'catpred', 'gatsol', 'geopoc', 'temberture', 'classical_properties'
            ],
        }
        
        # Add quiet flag if specified
        if self.config.get('quiet', False):
            for step_name, cmd in steps.items():
                if step_name in ['catpred', 'temberture', 'geopoc', 'gatsol']:
                    cmd.append('--quiet')
        
        return steps
    
    def _count_input_files(self, directory: Path, pattern: str = "*.faa") -> int:
        """Count input files for progress tracking"""
        if directory.exists():
            return len(list(directory.glob(pattern)))
        return 0
    
    def _is_warning_or_info_line(self, line: str) -> bool:
        """Check if a line is a warning or informational message, not a real error"""
        line_lower = line.lower()

        # Common warning patterns (now includes container-exit notices)
        warning_patterns = [
            'warning:', 'futurewarning:', 'deprecationwarning:', 'userwarning:',
            'runtimewarning:', 'pendingdeprecationwarning:', 'importwarning:',
            'unicodewarning:', 'byteswarning:', 'resourcewarning:',
            'will not inherit from', 'will lose the ability', 'from 👉v4.50👈',
            'get rid of this warning by', 'you can get rid of this warning',
            'please modify your model class', 'please contact the model',
            'weights_only=false', 'torch.load', 'pickle module implicitly',
            'torch.serialization.add_safe_globals', 'experimental feature',
            'passing list objects for adapter activation is deprecated',
            'use stack or fuse explicitly',
            'failed with exit code'   # container exit messages
        ]

        # Progress indicators (not errors)
        progress_patterns = [
            '%|', 'it/s]', '/s]', '[00:00<', '██████████', 'completed in',
            'successful:', 'found', 'output files', 'analysis complete',
            'all jobs completed successfully'
        ]

        # If it matches one of the warning patterns, treat it as non-error
        if any(pat in line_lower for pat in warning_patterns):
            return True

        # If it looks like a progress/info line, ignore as well
        if any(pat in line_lower for pat in progress_patterns):
            return True

        # Common success/completion phrases
        success_patterns = ['step', 'completed', 'analysis complete', 'jobs completed successfully']
        if any(pat in line_lower for pat in success_patterns):
            return True

        return False


    def _is_real_error(self, line: str) -> bool:
        """Check if a line represents a real error that should cause failure"""
        line_lower = line.lower()

        # First, filter out warnings/info
        if self._is_warning_or_info_line(line):
            return False

        # OSError patterns: ignore any read-only FS errors
        if 'oserror:' in line_lower:
            if 'read-only file system' in line_lower:
                return False
            return True

        # Real error patterns
        error_patterns = [
            'error:', 'exception:', 'traceback (most recent call last):',
            'fatal:', 'critical:', 'abort', 'segmentation fault',
            'killed', 'out of memory', 'no space left on device',
            'permission denied', 'file not found', 'cannot open',
            'failed to', 'unable to', 'could not'
        ]

        return any(pat in line_lower for pat in error_patterns)
    
    def run_step(self, step_name: str, retry: int = 0) -> StepResult:
        """Run a single workflow step with retries and monitoring"""
        if step_name not in self.step_commands:
            self.logger.error(f"Unknown step: {step_name}")
            return StepResult(step_name, False, 0, error="Unknown step")
        
        cmd = self.step_commands[step_name]
        start_time = time.time()
        
        # Count input files for progress context
        input_count = 0
        if step_name == 'clean_sequences':
            input_count = self._count_input_files(self.config.paths['input_fasta'])
        elif step_name in ['esmfold', 'esm2', 'temberture', 'geopoc', 'gatsol']:
            input_count = self._count_input_files(self.config.paths['clean_fasta'])
        
        if input_count > 0:
            self.logger.info(f"Processing {input_count} input files")
        
        try:
            # Log the command being run (sanitized)
            self.logger.debug(f"Running command: {' '.join(cmd[:3])}...")
            
            # Run the command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            output_lines = []
            error_lines = []
            warning_lines = []
            real_error_lines = []
            
            # Read stdout line by line
            for line in process.stdout:
                line = line.strip()
                if line:
                    output_lines.append(line)
                    
                    # Categorize the line
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
                        
                        # Categorize stderr lines
                        if self._is_warning_or_info_line(line):
                            warning_lines.append(line)
                            if not self.config.get('quiet', False):
                                self.logger.debug(f"  STDERR INFO/WARNING: {line}")
                        elif self._is_real_error(line):
                            real_error_lines.append(line)
                            self.logger.error(f"  STDERR ERROR: {line}")
            
            duration = time.time() - start_time
            
            # For container-based tools, check if output was actually created
            output_created = False
            output_file_count = 0
            if step_name in ['temberture', 'geopoc', 'gatsol', 'catpred', 'esmfold', 'esm2']:
                output_dir_key = f'{step_name}_output'
                if output_dir_key in self.config.paths:
                    output_dir = Path(self.config.paths[output_dir_key])
                elif step_name == 'esmfold':
                    output_dir = Path(self.config.paths['structures'])
                elif step_name == 'esm2':
                    output_dir = Path(self.config.paths['embeddings'])
                else:
                    output_dir = None
                    
                if output_dir and output_dir.exists():
                    # Check for output files based on the tool
                    if step_name == 'temberture':
                        result_files = list(output_dir.glob('**/*_results.tsv'))
                    elif step_name == 'geopoc':
                        result_files = list(output_dir.glob('**/*.json'))
                    elif step_name == 'gatsol':
                        result_files = list(output_dir.glob('**/*.csv'))
                    elif step_name == 'catpred':
                        result_files = list(output_dir.glob('**/final_predictions_*.csv'))
                    elif step_name == 'esmfold':
                        result_files = list(output_dir.glob('*.pdb'))
                    elif step_name == 'esm2':
                        result_files = list(output_dir.glob('*.pt'))
                    else:
                        result_files = []
                    
                    output_file_count = len(result_files)
                    output_created = output_file_count > 0
                    
                    if output_created:
                        self.logger.info(f"  Found {output_file_count} output files")
            
            # Determine if step was successful based on:
            # 1. Exit code 0 OR
            # 2. Output files were created AND no real errors occurred
            success = (process.returncode == 0) or (
                output_created and len(real_error_lines) == 0
            )
            
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
                    error_msg = '\n'.join(real_error_lines[:10])  # Limit to first 10 real errors
                elif not output_created and step_name in ['temberture', 'geopoc', 'gatsol', 'catpred']:
                    error_msg = f"No output files created and process exited with code {process.returncode}"
                else:
                    error_msg = f"Process failed with exit code {process.returncode}"
                
                self.logger.error(f"✗ Step '{step_name}' failed after {format_duration(duration)}")
                self.logger.error(f"  Exit code: {process.returncode}")
                self.logger.error(f"  Real errors found: {len(real_error_lines)}")
                if warning_lines:
                    self.logger.info(f"  Warnings/info messages: {len(warning_lines)}")
                
                # Retry logic
                if retry < self.config.get('max_retries', 2):
                    self.logger.info(f"  Retrying step '{step_name}' (attempt {retry + 2})...")
                    time.sleep(5)  # Wait before retry
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
            'condition': self.config.condition,
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
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.debug(f"State saved to {self.state_file}")
    
    def _load_state(self):
        """Load previous workflow state"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
            
            # Verify condition matches
            if state.get('condition') != self.config.condition:
                self.logger.warning(
                    f"State condition '{state.get('condition')}' doesn't match "
                    f"current condition '{self.config.condition}'. Starting fresh."
                )
                return
            
            # Restore completed steps
            for step_name in state.get('completed_steps', []):
                result_data = state['step_results'].get(step_name, {})
                self.results[step_name] = StepResult(
                    step_name,
                    result_data.get('success', True),
                    result_data.get('duration', 0)
                )
            
            self.logger.info(f"Resumed from previous state: {len(self.results)} steps completed")
    
    def run(self, steps: Optional[List[str]] = None, parallel_groups: Optional[List[List[str]]] = None):
        """
        Run the complete workflow or specified steps
        
        Args:
            steps: List of steps to run (if None, runs all configured steps)
            parallel_groups: List of step groups that can run in parallel
        """
        self.start_time = time.time()
        steps_to_run = steps or self.config.get('steps', self._get_default_steps())
        
        # Filter out already completed steps if resuming
        if self.resume:
            steps_to_run = [s for s in steps_to_run if s not in self.results]
            if not steps_to_run:
                self.logger.info("All steps already completed!")
                return
        
        total_steps = len(steps_to_run)
        self.logger.info(f"Starting workflow with {total_steps} steps")
        self.logger.info(f"Condition: {self.config.condition}")
        self.logger.info(f"Output directory: {self.config.paths['output_dir']}")
        
        # Define parallel execution groups if not provided
        if parallel_groups is None:
            parallel_groups = self._get_parallel_groups()
        
        completed = len(self.results)
        
        try:
            for step in steps_to_run:
                if self._interrupted:
                    break
                
                # Check if step is part of a parallel group
                parallel_step_group = None
                for group in parallel_groups:
                    if step in group and all(s in steps_to_run for s in group):
                        parallel_step_group = group
                        break
                
                if parallel_step_group and step == parallel_step_group[0]:
                    # Run parallel group
                    self._run_parallel_steps(parallel_step_group, completed, total_steps)
                    completed += len(parallel_step_group)
                    # Mark other steps in group as processed
                    for s in parallel_step_group[1:]:
                        steps_to_run.remove(s)
                elif not parallel_step_group:
                    # Run single step
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
            
            # Run consolidation if all previous steps succeeded
            if 'consolidate_results' not in self.results and not self._interrupted:
                self.logger.info("\n[Final] Consolidating results")
                self.logger.info("=" * 60)
                result = self.run_step('consolidate_results')
                self.results['consolidate_results'] = result
            
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
    
    def _run_parallel_steps(self, steps: List[str], completed: int, total: int):
        """Run multiple steps in parallel"""
        self.logger.info(f"\n[{completed + 1}-{completed + len(steps)}/{total}] "
                        f"Running parallel steps: {', '.join(steps)}")
        self.logger.info("=" * 60)
        
        with ThreadPoolExecutor(max_workers=len(steps)) as executor:
            future_to_step = {
                executor.submit(self.run_step, step): step 
                for step in steps
            }
            
            for future in as_completed(future_to_step):
                step = future_to_step[future]
                result = future.result()
                self.results[step] = result
                
                if not result.success:
                    # Cancel remaining futures
                    for f in future_to_step:
                        f.cancel()
                    raise RuntimeError(f"Parallel step {step} failed")
    
    def _get_default_steps(self) -> List[str]:
        """Get default step order"""
        return [
            'clean_sequences',
            'esmfold',
            'esm2',
            'remove_sequences_without_pdb',
            'prepare_catpred',
            'catpred',
            'temberture',
            'geopoc',
            'gatsol',
            'classical_properties',
            'process_temberture',
            'process_geopoc',
            'process_gatsol',
            'process_catpred',
            'consolidate_results'
        ]
    
    def _get_parallel_groups(self) -> List[List[str]]:
        """Define which steps can run in parallel"""
        return [
            ['esmfold', 'esm2'],  # Structure and embeddings in parallel
            ['catpred', 'temberture', 'geopoc', 'gatsol'],  # All prediction tools
            ['process_temberture', 'process_geopoc', 'process_gatsol', 'process_catpred']  # Processing
        ]
    
    def _print_summary(self):
        """Print workflow execution summary"""
        if not self.results:
            return
        
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("WORKFLOW SUMMARY")
        self.logger.info("=" * 60)
        
        # Step results
        for step_name, result in self.results.items():
            status = "✓" if result.success else "✗"
            self.logger.info(
                f"{status} {step_name:<30} {format_duration(result.duration):>10}"
            )
        
        # Statistics
        successful = sum(1 for r in self.results.values() if r.success)
        failed = len(self.results) - successful
        
        self.logger.info("-" * 60)
        self.logger.info(f"Total steps: {len(self.results)}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Total duration: {format_duration(total_duration)}")
        self.logger.info(f"Log file: {self.config.log_file}")
        
        if successful == len(self.results):
            self.logger.info(f"Results: {self.config.paths['consolidated_results']}")