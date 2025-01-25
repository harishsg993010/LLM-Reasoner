"""Model configuration and registry for reasoning chains."""

import os
from typing import Dict, Optional, Any
from dataclasses import dataclass

try:
    from litellm import completion
    from litellm.utils import ModelResponse
except ImportError:
    raise ImportError(
        "litellm is required for ReasonChain. "
        "Install it with `pip install litellm>=1.0.0`"
    )

from pydantic import BaseModel, ConfigDict, field_validator, Field

class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str = Field(..., description="Name of the model")
    provider: str = Field(..., description="Provider of the model")
    context_window: Optional[int] = Field(None, description="Maximum context window size")
    default: bool = Field(False, description="Whether this is the default model")

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "gpt-3.5-turbo",
                    "provider": "openai",
                    "context_window": 4096,
                    "default": True
                }
            ]
        }
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate model name."""
        if not v or not isinstance(v, str):
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider name."""
        # Allow any provider string to support custom providers
        return v.lower()

    @field_validator('context_window')
    @classmethod
    def validate_context_window(cls, v: Optional[int]) -> Optional[int]:
        """Validate context window size."""
        if v is not None and v <= 0:
            raise ValueError("Context window must be a positive integer")
        return v

class ModelRegistry:
    """Registry managing available models and their configurations."""

    def __init__(self):
        self._models = self._initialize_models()
        self._set_initial_default()

    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize available models."""
        models = {
            'gpt-3.5-turbo': ModelConfig(
                name='gpt-3.5-turbo',
                provider='openai',
                context_window=4096,
            ),
            'gpt-4': ModelConfig(
                name='gpt-4',
                provider='openai',
                context_window=8192,
            ),
            'claude-2': ModelConfig(
                name='claude-2',
                provider='anthropic',
                context_window=100000,
            ),
            'gemini-pro': ModelConfig(
                name='gemini-pro',
                provider='google',
                context_window=32768,
            ),
        }
        return models

    def _set_initial_default(self) -> None:
        """Set an initial default model based on available API keys."""
        if not any(model.default for model in self._models.values()):
            if os.getenv('OPENAI_API_KEY'):
                default_model = 'gpt-3.5-turbo'
            elif os.getenv('ANTHROPIC_API_KEY'):
                default_model = 'claude-2'
            elif os.getenv('GOOGLE_API_KEY'):
                default_model = 'gemini-pro'
            else:
                default_model = 'gpt-3.5-turbo'

            if default_model in self._models:
                model_dict = self._models[default_model].model_dump()
                model_dict['default'] = True
                self._models[default_model] = ModelConfig(**model_dict)

    def register_model(self, name: str, provider: str, context_window: Optional[int] = None) -> None:
        """Register a new model with the registry.

        Args:
            name: Name of the model
            provider: Provider of the model (can be custom)
            context_window: Optional maximum context window size
        """
        self._models[name] = ModelConfig(
            name=name,
            provider=provider,
            context_window=context_window,
            default=False
        )

    def get_model(self, model_name: str) -> ModelConfig:
        """Get model configuration by name."""
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found in available models")
        return self._models[model_name]

    def get_default_model(self) -> ModelConfig:
        """Get the current default model configuration."""
        default_model = next(
            (model for model in self._models.values() if model.default),
            None
        )
        if not default_model:
            raise RuntimeError("No default model configured")
        return default_model

    def set_default_model(self, model_name: str) -> None:
        """Set a new default model."""
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found in available models")

        for name, model in self._models.items():
            model_dict = model.model_dump()
            model_dict['default'] = (name == model_name)
            self._models[name] = ModelConfig(**model_dict)

    def list_models(self) -> Dict[str, ModelConfig]:
        """List all available models."""
        return self._models

# Initialize the global model registry
model_registry = ModelRegistry()