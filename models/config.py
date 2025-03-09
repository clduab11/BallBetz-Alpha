"""
Configuration module for BallBetz model inference.

This module provides configuration settings for both local Ollama models
and cloud-based OpenAI models, allowing seamless switching between them.
"""

import os
from enum import Enum
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))

class ModelProvider(str, Enum):
    """Enum for supported model providers."""
    SKLEARN = "sklearn"  # Original scikit-learn model
    OLLAMA = "ollama"    # Local Ollama models
    OPENAI = "openai"    # OpenAI API models

class ModelConfig:
    """Configuration settings for model inference."""
    
    # Default provider to use (can be overridden at runtime)
    DEFAULT_PROVIDER = os.getenv("BALLBETZ_DEFAULT_PROVIDER", ModelProvider.OLLAMA)
    
    # Enable fallback to alternative provider if primary fails
    ENABLE_FALLBACK = os.getenv("BALLBETZ_ENABLE_FALLBACK", "true").lower() == "true"
    
    # Fallback order: If primary provider fails, try these in order
    FALLBACK_ORDER = [ModelProvider.SKLEARN, ModelProvider.OLLAMA, ModelProvider.OPENAI]
    
    # Ollama configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:10000")
    # No default model - will be selected at runtime from available models
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))  # seconds
    
    # OpenAI configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_PRIMARY_MODEL = os.getenv("OPENAI_PRIMARY_MODEL", "o1-mini")
    OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-3.5-turbo")
    OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "30"))  # seconds
    
    # Common model parameters
    MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "1024"))
    TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.0"))  # Use 0 for deterministic outputs
    
    # Retry configuration
    MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "3"))
    RETRY_BACKOFF = float(os.getenv("MODEL_RETRY_BACKOFF", "2.0"))
    
    # Prompt templates optimized for different model sizes
    PROMPT_TEMPLATES = {
        # Template for small models (3B parameters)
        "small": """
You are the top daily fantasy sports and fantasy football analyst, specializing in UFL player performance predictions.
Analyze the following player statistics and predict their fantasy points:

Player: {player_name}
Position: {position}
Team: {team}
Stats: {stats}

Predict the fantasy points for this player in their next game.
Provide a single number as your prediction.
""",
        
        # Template for medium models (7B parameters)
        "medium": """
You are a fantasy football analyst specializing in UFL player performance predictions.
Analyze the following player statistics and predict their fantasy points:

Player: {player_name}
Position: {position}
Team: {team}
Stats: {stats}

Consider the player's recent performance, matchup, and position.
Predict the fantasy points for this player in their next game.
Provide a single number as your prediction.
""",
        
        # Template for large models (13B+ parameters)
        "large": """
You are a fantasy football analyst specializing in UFL player performance predictions.
Analyze the following player statistics and predict their fantasy points:

Player: {player_name}
Position: {position}
Team: {team}
Stats: {stats}

Consider the player's recent performance, matchup, position, and historical trends.
Analyze the offensive scheme of their team and the defensive strengths of their opponent.
Predict the fantasy points for this player in their next game.
Provide a single number as your prediction.
"""
    }
    
    @classmethod
    def get_prompt_template(cls, model_size: str = "small") -> str:
        """Get the appropriate prompt template based on model size."""
        return cls.PROMPT_TEMPLATES.get(model_size, cls.PROMPT_TEMPLATES["small"])
    
    @classmethod
    def get_model_size(cls, provider: ModelProvider) -> str:
        """Determine the model size based on provider and model name."""
        if provider == ModelProvider.OLLAMA and hasattr(cls, 'OLLAMA_MODEL'):
            # Determine size based on Ollama model name
            model_name = cls.OLLAMA_MODEL.lower()
            if "3b" in model_name:
                return "small"
            elif "7b" in model_name:
                return "medium"
            else:
                return "large"
        elif provider == ModelProvider.OPENAI:
            # Determine size based on OpenAI model name
            model_name = cls.OPENAI_PRIMARY_MODEL.lower()
            if "gpt-3.5" in model_name:
                return "medium"
            elif "gpt-4" in model_name:
                return "large"
            else:
                return "medium"
        else:
            # Default for sklearn
            return "small"