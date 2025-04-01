# BallBetz-Alpha: API/Local AI Layer Configuration

## Environment Variables

### Transformer Model Configuration
```bash
# Default model and generation parameters
TRANSFORMER_DEFAULT_MODEL=gpt-3.5-turbo
TRANSFORMER_MAX_TOKENS=2048
TRANSFORMER_TEMPERATURE=0.7
TRANSFORMER_BATCH_SIZE=16
```

### Provider Configuration
#### OpenAI Provider
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_TIMEOUT=30
OPENAI_MAX_RETRIES=3
```

#### Local Model Provider
```bash
# Local Transformer Model Configuration
LOCAL_MODEL_PATH=./models/local_transformer
LOCAL_MODEL_DEVICE=cpu  # or 'cuda' for GPU acceleration
```

### Caching Configuration
```bash
# Caching settings for transformer predictions
TRANSFORMER_CACHE_ENABLED=true
TRANSFORMER_CACHE_MAX_SIZE=1000
TRANSFORMER_CACHE_TTL=3600  # Cache time-to-live in seconds (1 hour)
```

### Rate Limiting Configuration
```bash
# Prevent excessive API calls and token usage
TRANSFORMER_MAX_REQUESTS_PER_MINUTE=60
TRANSFORMER_MAX_TOKENS_PER_MINUTE=90000
```

### Prediction Orchestration
```bash
# Confidence and fallback strategy configuration
PREDICTION_CONFIDENCE_THRESHOLD=0.7
PREDICTION_FALLBACK_STRATEGY=local_model
```

## Example .env File

```bash
# Transformer Model Settings
TRANSFORMER_DEFAULT_MODEL=gpt-3.5-turbo
TRANSFORMER_MAX_TOKENS=2048
TRANSFORMER_TEMPERATURE=0.7
TRANSFORMER_BATCH_SIZE=16

# OpenAI Provider
OPENAI_API_KEY=sk-your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_TIMEOUT=30
OPENAI_MAX_RETRIES=3

# Local Model Provider
LOCAL_MODEL_PATH=./models/local_transformer
LOCAL_MODEL_DEVICE=cpu

# Caching
TRANSFORMER_CACHE_ENABLED=true
TRANSFORMER_CACHE_MAX_SIZE=1000
TRANSFORMER_CACHE_TTL=3600

# Rate Limiting
TRANSFORMER_MAX_REQUESTS_PER_MINUTE=60
TRANSFORMER_MAX_TOKENS_PER_MINUTE=90000

# Prediction Orchestration
PREDICTION_CONFIDENCE_THRESHOLD=0.7
PREDICTION_FALLBACK_STRATEGY=local_model
```

## Security Considerations

1. Never commit your `.env` file to version control
2. Use a `.env.example` file as a template
3. Protect your API keys and sensitive configuration
4. Consider using a secrets management system for production environments

## Loading Environment Variables

### Python
```python
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env file
```

### Command Line
```bash
# Load environment variables before running the application
export $(cat .env | xargs)
python your_app.py