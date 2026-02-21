"""
Agent Module

Main negotiation agent with modular architecture.
The MainStrategy orchestrates all components.
"""

from .main_strategy import MainStrategy
import warnings
warnings.filterwarnings("ignore")

__all__ = ['MainStrategy'] 