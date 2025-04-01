# Prediction Module

## Overview
The Prediction Module provides a comprehensive interface for generating predictions using machine learning models in the BallBetz-Alpha Triple-Layer Prediction Engine.

## Key Components
- `predictor.py`: Core prediction interface
- `config.py`: Prediction configuration management
- `exceptions.py`: Custom error handling

## Features
- Flexible prediction for multiple task types
- Confidence scoring
- Ensemble prediction support
- Configurable output formats
- Logging and monitoring
- Error handling

## Configuration Environment Variables
- `UFL_PREDICTION_TASK`: Prediction task type (default: 'classification')
- `UFL_CONFIDENCE_THRESHOLD`: Minimum confidence for predictions (default: 0.5)
- `UFL_MODEL_DIR`: Directory for storing model checkpoints
- `UFL_OUTPUT_FORMAT`: Prediction output format (default: 'json')
- `UFL_LOG_PREDICTIONS`: Enable prediction logging (default: 'true')
- `UFL_USE_ENSEMBLE`: Enable ensemble prediction (default: 'false')
- `UFL_MAX_ENSEMBLE_MODELS`: Maximum number of models in ensemble (default: 5)

## Usage Example
```python
from ml_layer.prediction import UFLPredictor

# Initialize predictor
predictor = UFLPredictor()

# Load models
predictor.load_model('model1', 'path/to/checkpoint1.joblib')
predictor.load_model('model2', 'path/to/checkpoint2.joblib')

# Generate predictions
results = predictor.predict(X_test)

# Access predictions and confidence
predictions = results['predictions']
confidence = results['confidence']
```

## Dependencies
- scikit-learn
- numpy
- pandas
- joblib