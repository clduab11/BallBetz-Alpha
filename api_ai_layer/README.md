# API/Local AI Layer

## Overview

The API/Local AI Layer is a critical component of the BallBetz-Alpha Triple-Layer Prediction Engine, providing a flexible and modular approach to transformer model predictions.

## Key Components

### Transformer Model Interface
- Abstracts different transformer model providers
- Supports both API-based and local model predictions
- Implements caching and rate limiting mechanisms

### Providers
- `OpenAIProvider`: Integration with OpenAI's GPT models
- `LocalTransformerProvider`: Support for local Hugging Face transformer models

### Prediction Orchestrator
- Combines predictions from multiple layers
- Implements confidence scoring
- Provides fallback mechanisms

## Configuration

Configuration is managed through environment variables and the `config.py` module. Key configuration areas include:

### Transformer Model Configuration
- `TRANSFORMER_DEFAULT_MODEL`: Default model to use
- `TRANSFORMER_MAX_TOKENS`: Maximum tokens for generation
- `TRANSFORMER_TEMPERATURE`: Sampling temperature
- `TRANSFORMER_BATCH_SIZE`: Batch size for predictions

### Provider Configuration
#### OpenAI Provider
- `OPENAI_API_KEY`: API key for OpenAI
- `OPENAI_BASE_URL`: Base URL for OpenAI API
- `OPENAI_TIMEOUT`: Request timeout
- `OPENAI_MAX_RETRIES`: Maximum retry attempts

#### Local Model Provider
- `LOCAL_MODEL_PATH`: Path to local transformer model
- `LOCAL_MODEL_DEVICE`: Device for model inference (cpu/cuda)

### Caching Configuration
- `TRANSFORMER_CACHE_ENABLED`: Enable/disable caching
- `TRANSFORMER_CACHE_MAX_SIZE`: Maximum cache size
- `TRANSFORMER_CACHE_TTL`: Cache time-to-live

### Rate Limiting
- `TRANSFORMER_MAX_REQUESTS_PER_MINUTE`: Request frequency limit
- `TRANSFORMER_MAX_TOKENS_PER_MINUTE`: Token usage limit

## Usage Example

```python
from api_ai_layer.config import APIAILayerConfig
from api_ai_layer.providers.openai_client import OpenAIProvider
from api_ai_layer.providers.local_model_client import LocalTransformerProvider
from api_ai_layer.orchestrator import PredictionOrchestrator

# Load configuration
config = APIAILayerConfig.load_config()

# Create providers
openai_provider = OpenAIProvider(config)
local_provider = LocalTransformerProvider(config)

# Create orchestrator with multiple providers
orchestrator = PredictionOrchestrator(
    transformer_interface=transformer_interface,
    providers=[openai_provider, local_provider]
)

# Generate predictions
inputs = ["Your prediction input here"]
predictions = orchestrator.predict(inputs)
```

## Error Handling

The layer provides comprehensive error handling:
- `ModelLoadError`: Issues with model loading
- `PredictionError`: Prediction generation failures
- `ProviderError`: Provider-specific errors
- `RateLimitError`: Rate limit exceeded

## Performance Considerations
- Implements caching to reduce redundant API calls
- Supports rate limiting to prevent overuse
- Provides fallback mechanisms for provider failures

## Security
- Never hardcodes secrets
- Uses environment-based configuration
- Implements secure token management

## Extensibility
- Easily add new providers by implementing `BaseTransformerProvider`
- Configurable through environment variables
- Modular design allows easy integration and modification