"""ProtScout - Modular Protein Analysis Pipeline"""

__version__ = '0.1.0'
__author__ = 'Your Name'

from .config import Config
from .workflow import ProtScoutWorkflow

__all__ = ['Config', 'ProtScoutWorkflow']