# BallBetz-Alpha: Triple-Layer Prediction Engine

## Overview

BallBetz-Alpha is an advanced, multi-layered prediction engine designed to generate intelligent, context-aware predictions by integrating machine learning, AI, and contextual analysis.

## Architecture

### Layers

1. **Machine Learning Layer**
   - Feature engineering
   - Model training and selection
   - Predictive modeling

2. **API/Local AI Layer**
   - External AI model integration
   - Transformer-based prediction
   - Provider abstraction

3. **Cloud AI Layer (NEW)**
   - External factors integration
   - Cross-league pattern analysis
   - Weighted prediction combination
   - Comprehensive prediction explanation

## Cloud AI Layer Features

The Cloud AI Layer introduces advanced capabilities to the prediction engine:

- **External Factors Integration**
  - Normalize and score contextual data
  - Support for weather, injuries, and team statistics
  - Dynamic impact scoring

- **Cross-League Pattern Analysis**
  - Recognize patterns across different leagues
  - Compute league similarity scores
  - Enable knowledge transfer between sports contexts

- **Intelligent Prediction Combination**
  - Weighted ensemble of predictions
  - Configurable source weights
  - Confidence scoring mechanism
  - Detailed prediction explanations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/BallBetz-Alpha.git

# Install dependencies
pip install -r requirements.txt
```

### Example Usage

```python
from cloud_ai_layer import PredictionCombiner, create_prediction_request

# Create a prediction request
request = create_prediction_request(context={'game_data': {...}})

# Combine predictions from multiple sources
combiner = PredictionCombiner()
result = combiner.combine_predictions({
    'ml_layer': 0.7,
    'api_ai_layer': 0.6,
    'external_factors': 0.5
})

print(result['prediction'])  # Final weighted prediction
print(result['confidence'])  # Prediction confidence score
```

## Configuration

Configuration is managed through environment variables. Key configuration options include:

- `EXTERNAL_WEATHER_SOURCE`: Source for weather data
- `CROSS_LEAGUE_SIMILARITY_THRESHOLD`: Threshold for pattern matching
- `ML_LAYER_WEIGHT`: Weight for ML layer predictions
- `API_AI_LAYER_WEIGHT`: Weight for API/AI layer predictions

Refer to `.env.example` for a complete list of configurable parameters.

## Testing

```bash
# Run unit tests
pytest tests/unit/cloud_ai_layer/

# Run integration tests
pytest tests/integration/
```

## Performance Considerations

- Implements intelligent caching mechanisms
- Dynamic weight computation
- Configurable logging levels
- Modular, extensible design

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/cloud-ai-layer`)
3. Commit your changes (`git commit -m 'Add Cloud AI Layer'`)
4. Push to the branch (`git push origin feature/cloud-ai-layer`)
5. Open a Pull Request

## License

[Your License Here]

## Contact

[Your Contact Information]
