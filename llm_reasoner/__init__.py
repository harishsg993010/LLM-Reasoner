"""
LLM-Reasoner: Transform any LLM into a methodical thinker that excels at systematic reasoning.

This module provides a high-level interface for generating structured reasoning
chains using various language models through LiteLLM.
"""

from .models import ModelConfig, ModelRegistry, model_registry
from .engine import ReasonChain, Step, ReasoningError
from .settings import SYSTEM_PROMPT

__all__ = [
    'ReasonChain',
    'Step',
    'ReasoningError',
    'ModelConfig',
    'ModelRegistry',
    'model_registry',
    'SYSTEM_PROMPT'
]

__version__ = "0.1.9"  # Incrementing version for README updates and fresh upload