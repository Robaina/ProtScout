import subprocess
import shutil
from typing import Optional, Dict
import re

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def check_docker_running() -> bool:
    """Check if Docker daemon is running"""
    try:
        subprocess.run(['docker', 'info'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_gpu_info() -> Optional[str]:
   """Get GPU information if available"""
   if shutil.which('nvidia-smi'):
       try:
           result = subprocess.run(
               ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
               capture_output=True, text=True, check=True
           )
           return result.stdout.strip()
       except subprocess.CalledProcessError:
           return None
   return None

def parse_docker_stats(output: str) -> Dict[str, str]:
   """Parse docker stats output for resource monitoring"""
   stats = {}
   lines = output.strip().split('\n')
   if len(lines) > 1:
       # Parse the stats line (skip header)
       parts = lines[1].split()
       if len(parts) >= 4:
           stats['cpu'] = parts[2]
           stats['memory'] = parts[3]
   return stats

def estimate_runtime(num_sequences: int, step: str) -> str:
   """Estimate runtime based on number of sequences"""
   # Rough estimates per sequence (in seconds)
   time_per_seq = {
       'esmfold': 30,
       'esm2': 10,
       'catpred': 5,
       'temberture': 3,
       'geopoc': 4,
       'gatsol': 6,
   }
   
   if step in time_per_seq:
       total_seconds = num_sequences * time_per_seq[step]
       return format_duration(total_seconds)
   return "Unknown"

def validate_fasta_file(filepath: str) -> bool:
   """Validate FASTA file format"""
   try:
       with open(filepath, 'r') as f:
           content = f.read()
           # Basic FASTA validation
           return content.startswith('>') and '\n' in content
   except Exception:
       return False

def count_sequences_in_fasta(filepath: str) -> int:
   """Count sequences in a FASTA file"""
   count = 0
   with open(filepath, 'r') as f:
       for line in f:
           if line.startswith('>'):
               count += 1
   return count