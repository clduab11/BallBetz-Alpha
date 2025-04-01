# ML Layer - BallBetz-Alpha Triple-Layer Prediction Engine

## Overview
The Machine Learning (ML) Layer provides a comprehensive, modular framework for advanced predictive modeling in the BallBetz-Alpha system, specifically tailored for University Football League (UFL) statistical analysis.

## Key Components

### 1. Feature Engineering Module
- Robust data preprocessing pipeline
- Automatic feature selection
- Handling of missing data
- Configurable feature engineering strategies

#### Key Features:
- Automatic column type detection
- Imputation strategies
- Feature scaling
- One-hot encoding
- Flexible feature selection

### 2. Model Selection Module
- Centralized model management
- Advanced model evaluation
- Hyperparameter tuning
- Support for multiple prediction tasks

#### Key Features:
- Model registry
- Cross-validation
- Performance metric calculation
- Hyperparameter search strategies
- Ensemble model support

### 3. Training Pipeline
- Comprehensive model training workflow
- Configurable training parameters
- Model checkpointing
- Performance tracking

#### Key Features:
- Flexible training for classification and regression
- Automatic data preprocessing
- Early stopping
- Model serialization
- Logging and monitoring

### 4. Prediction Module
- Flexible prediction interface
- Confidence scoring
- Ensemble prediction
- Configurable output formats

#### Key Features:
- Multiple task type support
- Confidence threshold filtering
- Prediction logging
- Flexible model loading
- Error handling

## Configuration

The ML Layer is highly configurable through environment variables, allowing easy customization without code changes. Key configuration areas include:
- Feature engineering strategies
- Model selection parameters
- Training hyperparameters
- Prediction settings

## Usage Example

```python
from ml_layer.feature_engineering import UFLDataIntegrator
from ml_layer.model_selection import ModelRegistry, ModelEvaluator
from ml_layer.training import TrainingPipeline
from ml_layer.prediction import UFLPredictor

# Prepare dataset
integrator = UFLDataIntegrator()
dataset = integrator.prepare_prediction_dataset(prediction_type='win_loss')

# Select and train model
registry = ModelRegistry()
model_class = registry.get_model('classification', 'random_forest')
model = model_class()

# Train the model
pipeline = TrainingPipeline()
training_result = pipeline.train(model, dataset['features'], dataset['target'])

# Generate predictions
predictor = UFLPredictor()
predictor.load_model('ufl_model', training_result['checkpoint_path'])
predictions = predictor.predict(X_new)
```

## Dependencies
- scikit-learn
- numpy
- pandas
- joblib
- xgboost
- scipy

## Environment Configuration
Customize the ML Layer's behavior using environment variables in `.env` or system environment.

## Testing
Comprehensive unit tests are provided for each module to ensure reliability and performance.