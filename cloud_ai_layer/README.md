# Cloud AI Layer - BallBetz-Alpha Triple-Layer Prediction Engine

## Overview

The Cloud AI Layer is a sophisticated component of the BallBetz-Alpha Triple-Layer Prediction Engine, responsible for integrating, weighting, and enhancing predictions from multiple sources.

## Key Components

1. **External Factors Integration**
   - Normalizes and scores contextual data
   - Supports weather, injuries, team statistics
   - Provides impact scoring for different factor types

2. **Cross-League Pattern Analysis**
   - Recognizes patterns across different leagues
   - Provides similarity scoring and historical pattern matching
   - Enables knowledge transfer between different sports contexts

3. **Weighted Prediction Combiner**
   - Intelligently combines predictions from multiple sources
   - Configurable weighting mechanism
   - Generates confidence scores and prediction explanations

## Configuration

Configuration is managed through environment variables and the `config.py` module. Key configuration options include:

- External factor sources
- Cross-league similarity thresholds
- Prediction source weights
- Logging and explainability settings

### Environment Variables

- `EXTERNAL_WEATHER_SOURCE`: Source for weather data
- `EXTERNAL_INJURIES_SOURCE`: Source for injury information
- `CROSS_LEAGUE_SIMILARITY_THRESHOLD`: Threshold for cross-league pattern matching
- `ML_LAYER_WEIGHT`: Weight for ML layer predictions
- `API_AI_LAYER_WEIGHT`: Weight for API/AI layer predictions
- `EXTERNAL_FACTORS_WEIGHT`: Weight for external factors

## Interfaces

The layer provides standardized interfaces for:
- Prediction requests
- Prediction results
- Inter-layer communication
- Encryption and decryption of prediction data

## Usage Example

```python
from cloud_ai_layer.interfaces import create_prediction_request, PredictionSourceType
from cloud_ai_layer.prediction_combiner import PredictionCombiner

# Create a prediction request
request = create_prediction_request(
    context={'game_data': {...}},
    source_type=PredictionSourceType.ML_MODEL
)

# Use prediction combiner to process predictions
combiner = PredictionCombiner()
result = combiner.combine_predictions({
    'ml_layer': 0.7,
    'api_ai_layer': 0.6,
    'external_factors': 0.5
})

print(result['prediction'])  # Combined prediction
print(result['confidence'])  # Confidence score
```

## Error Handling

The layer includes comprehensive error handling through custom exceptions:
- `ExternalFactorIntegrationError`
- `PatternAnalysisError`
- `PredictionCombinerError`

## Performance Considerations

- Implements caching mechanisms
- Supports dynamic weight computation
- Provides configurable logging levels

## Testing

Comprehensive unit and integration tests are available in the `tests/` directory.

## Contributing

Please read the project's contribution guidelines before making changes to the Cloud AI Layer.