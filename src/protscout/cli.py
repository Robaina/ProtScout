import click
import sys
from pathlib import Path
from typing import Optional, List
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import track

from .config import Config
from .workflow import ProtScoutWorkflow
from .utils import check_docker_running, get_gpu_info, format_duration

console = Console()

@click.group()
@click.version_option(version='0.1.0')
def cli():
    """ProtScout - Modular Protein Analysis Pipeline
    
    A comprehensive workflow for protein structure prediction and property analysis.
    """
    pass

@cli.command()
@click.option('--config', '-c', type=Path, required=True, help='Workflow configuration file')
@click.option('--condition', help='Override condition from config')
@click.option('--steps', '-s', multiple=True, help='Run only specific steps')
@click.option('--resume', is_flag=True, help='Resume from last successful step')
@click.option('--dry-run', is_flag=True, help='Show what would be executed')
@click.option('--validate-only', is_flag=True, help='Validate configuration and exit')
def run(config: Path, condition: Optional[str], steps: List[str], 
        resume: bool, dry_run: bool, validate_only: bool):
    """Run the ProtScout workflow"""
    
    try:
        # Load configuration
        if not config.exists():
            console.print(f"[red]Error: Configuration file not found: {config}[/red]")
            sys.exit(1)
        
        cfg = Config(config)
        
        # Override condition if provided
        if condition:
            cfg._config['condition'] = condition
            cfg._setup_paths()
            console.print(f"[yellow]Overriding condition to: {condition}[/yellow]")
        
        # Validate only mode
        if validate_only:
            console.print("[green]✓ Configuration is valid[/green]")
            console.print(f"Condition: {cfg.condition}")
            console.print(f"Workdir: {cfg.workdir}")
            console.print(f"Model dir: {cfg.modeldir}")
            return
        
        # Initialize workflow
        workflow = ProtScoutWorkflow(cfg, resume=resume)
        
        # Dry run mode
        if dry_run:
            console.print("[yellow]DRY RUN MODE - No commands will be executed[/yellow]\n")
            
            steps_to_run = list(steps) if steps else cfg.get('steps', workflow._get_default_steps())
            
            table = Table(title="Steps to be executed")
            table.add_column("Step", style="cyan")
            table.add_column("Description", style="white")
            
            descriptions = {
                'clean_sequences': "Clean and deduplicate input sequences",
                'esmfold': "Predict protein structures using ESMFold",
                'esm2': "Generate protein embeddings using ESM-2",
                'remove_sequences_without_pdb': "Filter sequences without structures",
                'prepare_catpred': "Prepare inputs for CatPred analysis",
                'catpred': "Predict catalytic properties",
                'temberture': "Predict temperature stability",
                'geopoc': "Predict environmental conditions (temp, pH, salt)",
                'gatsol': "Predict solubility",
                'classical_properties': "Calculate classical protein properties",
                'process_temberture': "Process Temberture results",
                'process_geopoc': "Process GeoPoc results",
                'process_gatsol': "Process GATSol results",
                'process_catpred': "Process CatPred results",
                'consolidate_results': "Create final output tables"
            }
            
            for step in steps_to_run:
                table.add_row(step, descriptions.get(step, ""))
            
            console.print(table)
            return
        
        # Run workflow
        console.print(f"[bold]Starting ProtScout workflow[/bold]")
        console.print(f"Condition: [cyan]{cfg.condition}[/cyan]")
        console.print(f"Output: [cyan]{cfg.paths['output_dir']}[/cyan]\n")
        
        workflow.run(list(steps) if steps else None)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Workflow interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', type=Path, required=True, help='Configuration file to validate')
def validate(config: Path):
    """Validate workflow configuration and system requirements"""
    
    console.print("[bold]Validating ProtScout configuration...[/bold]\n")
    
    # Check configuration file
    if not config.exists():
        console.print(f"[red]✗ Configuration file not found: {config}[/red]")
        sys.exit(1)
    
    try:
        cfg = Config(config)
        console.print("[green]✓ Configuration file is valid[/green]")
    except Exception as e:
        console.print(f"[red]✗ Configuration error: {str(e)}[/red]")
        sys.exit(1)
    
    # Check Docker
    if check_docker_running():
        console.print("[green]✓ Docker is running[/green]")
    else:
        console.print("[red]✗ Docker is not running[/red]")
    
    # Check GPU
    gpu_info = get_gpu_info()
    if gpu_info:
        console.print(f"[green]✓ GPU available: {gpu_info}[/green]")
    else:
        console.print("[yellow]⚠ No GPU detected (some steps may run slower)[/yellow]")
    
    # Check required directories
    console.print("\n[bold]Checking directories:[/bold]")
    for name, path in cfg.paths.items():
        if path.exists():
            console.print(f"  [green]✓[/green] {name}: {path}")
        else:
            console.print(f"  [yellow]○[/yellow] {name}: {path} (will be created)")
    
    # Check containers
    console.print("\n[bold]Container configuration:[/bold]")
    containers = cfg.get('containers', {})
    for name, config in containers.items():
        image = config.get('image', 'Not specified')
        console.print(f"  {name}: {image}")

@cli.command()
@click.option('--output', '-o', type=Path, default='workflow_config.yaml', help='Output file')
@click.option('--condition', default='ultra', help='Analysis condition')
@click.option('--workdir', type=Path, help='Working directory')
@click.option('--modeldir', type=Path, help='Model directory')
def init(output: Path, condition: str, workdir: Optional[Path], modeldir: Optional[Path]):
    """Generate a template configuration file"""
    
    template = {
        'condition': condition,
        'workdir': str(workdir or '/path/to/workdir'),
        'modeldir': str(modeldir or '/path/to/models'),
        'python_executable': '/path/to/conda/env/bin/python',
        'memory': '100g',
        'workers': 2,
        'quiet': True,
        'max_retries': 2,
        
        'containers': {
            'esmfold': {
                'image': 'ghcr.io/new-atlantis-labs/esmfold:latest',
                'max_containers': 1
            },
            'esm2': {
                'image': 'ghcr.io/new-atlantis-labs/esm2:latest',
                'max_containers': 1,
                'toks_per_batch': 4096
            },
            'catpred': {
                'image': 'ghcr.io/new-atlantis-labs/catpred:latest'
            },
            'temberture': {
                'image': 'ghcr.io/new-atlantis-labs/temberture:latest'
            },
            'geopoc': {
                'image': 'ghcr.io/new-atlantis-labs/geopoc:latest'
            },
            'gatsol': {
                'image': 'ghcr.io/new-atlantis-labs/gatsol:latest'
            }
        },
        
        'resources': {
            'gpus': 'all',
            'shm_size': '100g'
        },
        
        'steps': [
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
    }
    
    with open(output, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
    
    console.print(f"[green]✓ Created template configuration: {output}[/green]")
    console.print("\n[yellow]Please edit the configuration file to set:[/yellow]")
    console.print("  - workdir: Path to your working directory")
    console.print("  - modeldir: Path to model weights")
    console.print("  - python_executable: Path to your conda environment Python")

@cli.command()
@click.argument('log_file', type=Path)
@click.option('--tail', '-n', default=50, help='Number of lines to show')
@click.option('--follow', '-f', is_flag=True, help='Follow log file')
def logs(log_file: Path, tail: int, follow: bool):
    """View workflow logs"""
    
    if not log_file.exists():
        console.print(f"[red]Log file not found: {log_file}[/red]")
        sys.exit(1)
    
    if follow:
        # Follow mode
        import subprocess
        subprocess.run(['tail', '-f', str(log_file)])
    else:
        # Show last N lines
        with open(log_file) as f:
            lines = f.readlines()
            for line in lines[-tail:]:
                console.print(line.rstrip())

if __name__ == '__main__':
    cli()