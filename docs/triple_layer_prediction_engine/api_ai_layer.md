# API/Local AI Layer Documentation

This document provides detailed information about the API/Local AI Layer of the BallBetz-Alpha Triple-Layer Prediction Engine.

## Overview

The API/Local AI Layer is the second layer of the Triple-Layer Prediction Engine, enhancing predictions using transformer-based models that can identify patterns and contextual relationships not captured by traditional machine learning. This layer provides deeper insights through natural language understanding and pattern recognition capabilities.

## Architecture

The API/Local AI Layer consists of three main components:

1. **Transformer Model Interface**: Abstracts different transformer model providers
2. **Provider-Specific Clients**: Implements connections to different model providers
3. **Prediction Orchestrator**: Manages prediction generation and fallback mechanisms

```
┌─────────────────────────────────────────────────────────────┐
│                    API/Local AI Layer                        │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌───────────────────┼───────────────────┐
        │                    │                   │
        ▼                    ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│Transformer Model│  │Provider-Specific│  │   Prediction    │
│    Interface    │  │     Clients     │  │   Orchestrator  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                    │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │   Predictions   │
                  └─────────────────┘
```

## Components

### 1. Transformer Model Interface

The Transformer Model Interface provides a unified API for interacting with different transformer models, whether they are hosted remotely or run locally.

#### Key Classes and Functions

- `TransformerModelInterface`: Main interface for transformer models
- `TransformerConfig`: Configuration for transformer models
- `PromptFormatter`: Formats data into prompts for transformer models

#### Supported Model Types

1. **Remote API Models**:
   - OpenAI GPT models
   - Anthropic Claude models
   - Other API-based models

2. **Local Models**:
   - Hugging Face transformer models
   - Ollama models
   - GGML/GGUF quantized models

#### Example Usage

```python
from api_ai_layer.transformer_interface import TransformerModelInterface
from api_ai_layer.config import TransformerConfig

# Create configuration
config = TransformerConfig(
    provider="openai",
    model_name="gpt-3.5-turbo",
    api_key="your_api_key",
    temperature=0.0,
    max_tokens=100
)

# Initialize interface
interface = TransformerModelInterface(config)

# Create prompt
prompt = """
Analyze the following player statistics and predict fantasy points:
Player: John Smith
Position: QB
Team: Birmingham Stallions
Opponent: Michigan Panthers
Home Game: Yes
Season Stats:
- Passing Yards/Game: 285.3
- Passing TDs/Game: 2.1
- Interceptions/Game: 0.7
- Rushing Yards/Game: 32.5
- Rushing TDs/Game: 0.3
"""

# Generate prediction
response = interface.generate(prompt)

print(f"Transformer model response: {response}")
```

### 2. Provider-Specific Clients

Provider-Specific Clients handle the communication with different model providers, implementing the specific API requirements for each.

#### Key Classes

- `BaseProvider`: Abstract base class for all providers
- `OpenAIProvider`: Client for OpenAI API
- `LocalModelProvider`: Client for local transformer models

#### OpenAI Provider

The OpenAI Provider connects to OpenAI's API to use models like GPT-3.5 and GPT-4.

```python
from api_ai_layer.providers.openai_client import OpenAIProvider
from api_ai_layer.config import ProviderConfig

# Configure OpenAI provider
config = ProviderConfig(
    api_key="your_openai_api_key",
    base_url="https://api.openai.com/v1",
    timeout=30,
    max_retries=3
)

# Initialize provider
provider = OpenAIProvider(config)

# Generate text
response = provider.generate(
    model="gpt-3.5-turbo",
    prompt="Predict fantasy points for QB John Smith",
    max_tokens=100,
    temperature=0.0
)

print(f"OpenAI response: {response}")
```

#### Local Model Provider

The Local Model Provider runs transformer models locally using libraries like Hugging Face Transformers or Ollama.

```python
from api_ai_layer.providers.local_model_client import LocalModelProvider
from api_ai_layer.config import ProviderConfig

# Configure local model provider
config = ProviderConfig(
    model_path="./models/local_transformer",
    device="cpu",  # or "cuda" for GPU
    max_memory=None  # None for no limit
)

# Initialize provider
provider = LocalModelProvider(config)

# Generate text
response = provider.generate(
    model="llama3.2-3b-instruct",
    prompt="Predict fantasy points for QB John Smith",
    max_tokens=100,
    temperature=0.0
)

print(f"Local model response: {response}")
```

### 3. Prediction Orchestrator

The Prediction Orchestrator manages the process of generating predictions, handling multiple providers, and implementing fallback mechanisms.

#### Key Classes and Functions

- `PredictionOrchestrator`: Main orchestration class
- `ProviderSelector`: Selects the appropriate provider
- `ResponseParser`: Parses model responses into structured predictions

#### Orchestration Process

1. **Provider Selection**:
   - Choose provider based on configuration
   - Consider availability and rate limits

2. **Prompt Generation**:
   - Format player data into a prompt
   - Include relevant context and instructions

3. **Prediction Generation**:
   - Send prompt to selected provider
   - Handle timeouts and retries

4. **Response Parsing**:
   - Extract prediction from response
   - Calculate confidence score
   - Format into standard structure

5. **Fallback Handling**:
   - Detect failures or low-confidence predictions
   - Try alternative providers if needed

#### Example Usage

```python
from api_ai_layer.orchestrator import PredictionOrchestrator
from api_ai_layer.transformer_interface import TransformerModelInterface
from api_ai_layer.providers.openai_client import OpenAIProvider
from api_ai_layer.providers.local_model_client import LocalModelProvider

# Create transformer interface
transformer_interface = TransformerModelInterface(config)

# Create providers
openai_provider = OpenAIProvider(openai_config)
local_provider = LocalModelProvider(local_config)

# Create orchestrator with multiple providers
orchestrator = PredictionOrchestrator(
    transformer_interface=transformer_interface,
    providers=[openai_provider, local_provider]
)

# Player data
player_data = {
    "player_id": "UFL12345",
    "name": "John Smith",
    "position": "QB",
    "team": "Birmingham Stallions",
    "opponent": "Michigan Panthers",
    "home_game": True,
    "stats": {
        "passing_yards_avg": 285.3,
        "passing_tds_avg": 2.1,
        "interceptions_avg": 0.7,
        "rushing_yards_avg": 32.5,
        "rushing_tds_avg": 0.3
    }
}

# Generate prediction
prediction = orchestrator.predict(player_data, prediction_type="fantasy_points")

print(f"Predicted fantasy points: {prediction['value']}")
print(f"Confidence: {prediction['confidence']}")
```

## Configuration

The API/Local AI Layer is highly configurable through environment variables and configuration files.

### Environment Variables

```bash
# Enable/disable Transformer Layer
PREDICTION_TRANSFORMER_ENABLED=true

# Provider settings
PREDICTION_TRANSFORMER_PROVIDER=ollama  # or 'openai'
PREDICTION_TRANSFORMER_BASE_URL=http://localhost:10000
PREDICTION_TRANSFORMER_MODEL=llama3.2-3b-instruct
PREDICTION_TRANSFORMER_TIMEOUT=30

# OpenAI specific settings
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_TIMEOUT=30
OPENAI_MAX_RETRIES=3

# Local model settings
LOCAL_MODEL_PATH=./models/local_transformer
LOCAL_MODEL_DEVICE=cpu  # or 'cuda' for GPU acceleration

# Caching settings
TRANSFORMER_CACHE_ENABLED=true
TRANSFORMER_CACHE_MAX_SIZE=1000
TRANSFORMER_CACHE_TTL=3600  # seconds

# Rate limiting
TRANSFORMER_MAX_REQUESTS_PER_MINUTE=60
TRANSFORMER_MAX_TOKENS_PER_MINUTE=90000
```

### Configuration Files

The API/Local AI Layer uses several configuration files:

1. **Main Configuration**:
   - `api_ai_layer/config.py`
   - Defines main configuration classes and loading functions

2. **Provider Configuration**:
   - Provider-specific configuration in respective client files
   - Defines API endpoints, authentication, and request formats

3. **Prompt Templates**:
   - Templates for different prediction types
   - Stored as text files or embedded in code

## Prompt Engineering

The API/Local AI Layer uses carefully crafted prompts to get the best results from transformer models.

### Prompt Structure

```
[SYSTEM INSTRUCTIONS]
You are a sports prediction expert specializing in UFL fantasy football.
Your task is to predict {prediction_type} for the player based on the provided statistics.
Provide your prediction as a number followed by a confidence score from 0.0 to 1.0.

[PLAYER INFORMATION]
Player: {player_name}
Position: {position}
Team: {team}
Opponent: {opponent}
Home Game: {home_game}

[STATISTICS]
{statistics}

[HISTORICAL CONTEXT]
{historical_context}

[PREDICTION FORMAT]
Prediction: <number>
Confidence: <0.0-1.0>
Reasoning: <brief explanation>
```

### Example Prompts

#### Fantasy Points Prediction

```
You are a sports prediction expert specializing in UFL fantasy football.
Your task is to predict fantasy points for the player based on the provided statistics.
Provide your prediction as a number followed by a confidence score from 0.0 to 1.0.

Player: John Smith
Position: QB
Team: Birmingham Stallions
Opponent: Michigan Panthers
Home Game: Yes

Season Statistics:
- Passing Yards/Game: 285.3
- Passing TDs/Game: 2.1
- Interceptions/Game: 0.7
- Rushing Yards/Game: 32.5
- Rushing TDs/Game: 0.3

Historical Context:
- Last 3 games fantasy points: 22.4, 18.7, 25.1
- Michigan Panthers allow 21.3 fantasy points to QBs on average
- Weather forecast: Clear, 72°F

Prediction: <number>
Confidence: <0.0-1.0>
Reasoning: <brief explanation>
```

## Integration with Other Layers

The API/Local AI Layer integrates with the other layers of the Triple-Layer Prediction Engine:

1. **Integration with ML Layer**:
   - Receives baseline predictions as context
   - Enhances statistical predictions with pattern recognition

2. **Integration with Cloud AI Layer**:
   - Provides transformer-based predictions
   - Contributes to weighted predictions

## Error Handling

The API/Local AI Layer includes comprehensive error handling:

1. **API Errors**:
   - Connection failures
   - Authentication errors
   - Rate limiting
   - Timeout handling

2. **Model Errors**:
   - Invalid responses
   - Parsing failures
   - Out-of-context predictions

3. **Fallback Mechanisms**:
   - Provider fallback (try alternative providers)
   - Model fallback (try alternative models)
   - Default predictions when all else fails

## Performance Considerations

For optimal performance when using the API/Local AI Layer:

1. **Caching**: Enable caching to reduce redundant API calls
2. **Batching**: Use batch predictions when possible
3. **Local Models**: Use local models for lower latency and cost
4. **Rate Limiting**: Implement rate limiting to prevent API throttling
5. **Prompt Optimization**: Keep prompts concise and focused

## Example Workflows

### Using Multiple Providers

```python
from api_ai_layer.config import APIAILayerConfig
from api_ai_layer.providers.openai_client import OpenAIProvider
from api_ai_layer.providers.local_model_client import LocalModelProvider
from api_ai_layer.orchestrator import PredictionOrchestrator

# Load configuration
config = APIAILayerConfig.load_config()

# Create providers
openai_provider = OpenAIProvider(config.openai)
local_provider = LocalModelProvider(config.local_model)

# Create orchestrator with provider priority
orchestrator = PredictionOrchestrator(
    providers=[openai_provider, local_provider],
    fallback_enabled=True
)

# Generate predictions with automatic fallback
try:
    prediction = orchestrator.predict(player_data)
    print(f"Prediction: {prediction}")
except Exception as e:
    print(f"All providers failed: {str(e)}")
```

### Custom Prompt Templates

```python
from api_ai_layer.transformer_interface import TransformerModelInterface, PromptTemplate

# Define custom prompt template
custom_template = PromptTemplate(
    template="""
    Analyze the following UFL player and predict {prediction_type}.
    
    Player: {name}
    Position: {position}
    Team: {team}
    
    Recent Performance:
    {recent_performance}
    
    Provide your prediction as a single number.
    """
)

# Initialize interface with custom template
interface = TransformerModelInterface(
    config=config,
    prompt_template=custom_template
)

# Generate prediction with custom template
prediction = interface.generate_prediction(
    player_data,
    prediction_type="fantasy_points"
)
```

## Testing

The API/Local AI Layer includes comprehensive tests:

1. **Unit Tests**:
   - `tests/unit/api_ai_layer/test_transformer_interface.py`
   - `tests/unit/api_ai_layer/test_providers.py`
   - `tests/unit/api_ai_layer/test_orchestrator.py`

2. **Integration Tests**:
   - `tests/integration/api_ai_layer/test_api_ai_layer_integration.py`

3. **Mock Tests**:
   - Tests using mocked API responses
   - Tests for fallback mechanisms

## Source Code

The API/Local AI Layer source code is available in the BallBetz-Alpha repository:

[https://github.com/clduab11/BallBetz-Alpha/tree/main/api_ai_layer](https://github.com/clduab11/BallBetz-Alpha/tree/main/api_ai_layer)