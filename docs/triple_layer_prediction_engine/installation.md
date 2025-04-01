# Installation and Configuration Guide

This guide provides detailed instructions for installing, configuring, and setting up the BallBetz-Alpha Triple-Layer Prediction Engine.

## Prerequisites

Before installing the Triple-Layer Prediction Engine, ensure your system meets the following requirements:

### System Requirements

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- 50GB disk space for models and data
- CUDA-compatible GPU (optional, for faster transformer model inference)

### Required Software

- Git
- Python 3.9+
- pip (Python package manager)
- virtualenv or conda (recommended for environment management)

### Required Accounts

- OpenAI API account (for API/Local AI Layer with OpenAI models)
- Weather API account (for Cloud AI Layer external factors)

## Installation Steps

Follow these steps to install the Triple-Layer Prediction Engine:

### 1. Clone the Repository

```bash
git clone https://github.com/your-organization/BallBetz-Alpha.git
cd BallBetz-Alpha
```

### 2. Create a Virtual Environment

Using virtualenv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using conda:

```bash
conda create -n ballbetz python=3.9
conda activate ballbetz
```

### 3. Install Dependencies

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Install layer-specific dependencies:

```bash
pip install -r ml_layer/requirements.txt
pip install -r api_ai_layer/requirements.txt
pip install -r cloud_ai_layer/requirements.txt
```

### 4. Download Pre-trained Models

For the ML Layer:

```bash
python scripts/download_ml_models.py
```

For local transformer models (optional):

```bash
python scripts/download_transformer_models.py
```

### 5. Set Up Environment Variables

Create a `.env` file in the project root by copying the example:

```bash
cp .env.example .env
```

Edit the `.env` file to include your API keys and configuration settings (see the Configuration section below).

## Configuration

The Triple-Layer Prediction Engine is configured through environment variables. These can be set in a `.env` file or directly in your environment.

### ML Layer Configuration

```bash
# Enable/disable ML Layer
PREDICTION_ML_ENABLED=true

# Model path
PREDICTION_ML_MODEL_PATH=models/checkpoints/latest.joblib

# Feature engineering settings
PREDICTION_ML_ROLLING_WINDOW=3
PREDICTION_ML_ADVANCED_METRICS=true
```

### API/Local AI Layer Configuration

```bash
# Enable/disable Transformer Layer
PREDICTION_TRANSFORMER_ENABLED=true

# Provider settings
PREDICTION_TRANSFORMER_PROVIDER=ollama  # or 'openai'
PREDICTION_TRANSFORMER_BASE_URL=http://localhost:10000
PREDICTION_TRANSFORMER_MODEL=llama3.2-3b-instruct
PREDICTION_TRANSFORMER_TIMEOUT=30

# OpenAI specific settings (if using OpenAI provider)
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_TIMEOUT=30
OPENAI_MAX_RETRIES=3

# Local model settings (if using local models)
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

### Cloud AI Layer Configuration

```bash
# Enable/disable Cloud Layer
PREDICTION_CLOUD_ENABLED=true

# External factors settings
PREDICTION_CLOUD_WEATHER=true
PREDICTION_CLOUD_INJURIES=true
PREDICTION_CLOUD_SOCIAL=false

# Cross-league settings
PREDICTION_CLOUD_NFL=true
PREDICTION_CLOUD_NCAA=true

# External data sources
EXTERNAL_WEATHER_SOURCE=your_weather_api_key_here
EXTERNAL_INJURIES_SOURCE=your_injuries_api_key_here
```

### Prediction Weights Configuration

```bash
# Layer weights for prediction combining
PREDICTION_WEIGHT_ML=0.4
PREDICTION_WEIGHT_TRANSFORMER=0.3
PREDICTION_WEIGHT_EXTERNAL=0.2
PREDICTION_WEIGHT_CROSS_LEAGUE=0.1
```

## Verifying Installation

To verify that the Triple-Layer Prediction Engine is installed correctly:

1. Run the test suite:

```bash
pytest tests/
```

2. Run the example prediction script:

```bash
python examples/triple_layer_prediction_example.py
```

This should output sample predictions using all three layers.

## Troubleshooting

### Common Issues

#### ML Layer Issues

- **Model not found**: Ensure the model path in `PREDICTION_ML_MODEL_PATH` is correct
- **Feature engineering errors**: Check that the data format matches the expected schema

#### API/Local AI Layer Issues

- **API connection errors**: Verify your API keys and network connectivity
- **Local model errors**: Ensure the model is downloaded and the path is correct

#### Cloud AI Layer Issues

- **External data source errors**: Check your API keys for weather and injury data
- **Integration errors**: Verify that all layers are enabled and configured correctly

### Getting Help

If you encounter issues not covered in this guide:

1. Check the logs in the `logs/` directory
2. Review the test results for specific errors
3. Consult the troubleshooting section in the maintenance guide
4. Contact the development team for support

## Next Steps

After installation and configuration:

1. Review the [Usage Guide](usage.md) for instructions on using the prediction engine
2. Explore the [ML Layer Documentation](ml_layer.md) for details on the statistical models
3. Learn about the [API/Local AI Layer](api_ai_layer.md) for transformer model integration
4. Understand the [Cloud AI Layer](cloud_ai_layer.md) for external factor integration