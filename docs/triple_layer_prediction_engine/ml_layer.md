# ML Layer Documentation

This document provides detailed information about the ML Layer of the BallBetz-Alpha Triple-Layer Prediction Engine.

## Overview

The Machine Learning (ML) Layer is the foundation of the Triple-Layer Prediction Engine, providing statistical analysis and baseline predictions using traditional machine learning techniques. It is specifically tailored for University Football League (UFL) statistical analysis and serves as the first layer in the prediction pipeline.

## Architecture

The ML Layer consists of four main components:

1. **Feature Engineering Module**: Processes raw data into meaningful features
2. **Model Selection Module**: Selects and configures appropriate models
3. **Training Pipeline**: Trains models on historical data
4. **Prediction Module**: Generates predictions using trained models

```
┌─────────────────────────────────────────────────────────────┐
│                         ML Layer                             │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌───────────────────┼───────────────────┐
        │                    │                   │
        ▼                    ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│Feature Engineering│  │Model Selection  │  │Training Pipeline│
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                    │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │Prediction Module│
                  └─────────────────┘
```

## Components

### 1. Feature Engineering Module

The Feature Engineering Module transforms raw player and game data into meaningful features that can be used by machine learning models.

#### Key Classes and Functions

- `UFLDataIntegrator`: Integrates data from various sources
- `Preprocessor`: Handles data cleaning and preprocessing
- `FeatureEngineer`: Creates features from raw data

#### Feature Types

1. **Base Features**:
   - Player statistics (passing yards, rushing yards, etc.)
   - Team statistics (offense rating, defense rating, etc.)
   - Game context (home/away, division game, etc.)

2. **Derived Features**:
   - Rolling averages (3-game, 5-game, season)
   - Performance trends
   - Efficiency metrics

3. **Contextual Features**:
   - Opponent strength
   - Weather conditions
   - Rest days

#### Example Usage

```python
from ml_layer.feature_engineering import UFLDataIntegrator, Preprocessor

# Initialize components
integrator = UFLDataIntegrator()
preprocessor = Preprocessor()

# Load and integrate data
raw_data = integrator.load_data(source="ufl_official")

# Preprocess data
cleaned_data = preprocessor.clean_data(raw_data)

# Engineer features
features = preprocessor.engineer_features(cleaned_data)

print(f"Generated {features.shape[1]} features for {features.shape[0]} players")
```

### 2. Model Selection Module

The Model Selection Module is responsible for selecting and configuring the appropriate machine learning models based on the prediction task and player position.

#### Key Classes and Functions

- `ModelRegistry`: Manages available models
- `ModelEvaluator`: Evaluates model performance
- `HyperparameterTuner`: Optimizes model hyperparameters

#### Supported Models

1. **Regression Models**:
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
   - Linear Regression

2. **Classification Models**:
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - XGBoost Classifier
   - Logistic Regression

#### Position-Specific Models

The module selects different models based on player position:

- **QB Models**: Specialized for quarterback statistics
- **RB Models**: Optimized for running back performance
- **WR/TE Models**: Tailored for receiver statistics
- **DEF Models**: Designed for defensive player predictions

#### Example Usage

```python
from ml_layer.model_selection import ModelRegistry, ModelEvaluator

# Initialize components
registry = ModelRegistry()
evaluator = ModelEvaluator()

# Get model for specific position and prediction type
model_class = registry.get_model(
    position="QB",
    prediction_type="fantasy_points",
    model_type="regression"
)

# Initialize model with default hyperparameters
model = model_class()

# Evaluate model on validation data
performance = evaluator.evaluate(model, X_val, y_val)

print(f"Model performance: {performance}")
```

### 3. Training Pipeline

The Training Pipeline handles the end-to-end process of training machine learning models, from data preparation to model serialization.

#### Key Classes and Functions

- `TrainingPipeline`: Orchestrates the training process
- `ModelCheckpoint`: Manages model versioning and storage
- `TrainingConfig`: Configures training parameters

#### Training Process

1. **Data Preparation**:
   - Split data into training and validation sets
   - Apply feature scaling
   - Handle class imbalance (if applicable)

2. **Model Training**:
   - Initialize model with hyperparameters
   - Fit model to training data
   - Apply early stopping (if configured)

3. **Model Evaluation**:
   - Evaluate on validation data
   - Calculate performance metrics
   - Generate feature importance

4. **Model Persistence**:
   - Save model checkpoint
   - Record metadata and performance
   - Version control

#### Example Usage

```python
from ml_layer.training import TrainingPipeline, TrainingConfig

# Configure training
config = TrainingConfig(
    cv_folds=5,
    early_stopping_rounds=10,
    random_state=42
)

# Initialize pipeline
pipeline = TrainingPipeline(config)

# Train model
result = pipeline.train(
    model=model,
    X=X_train,
    y=y_train,
    validation_data=(X_val, y_val)
)

print(f"Training complete. Model saved to: {result['checkpoint_path']}")
print(f"Validation score: {result['validation_score']}")
```

### 4. Prediction Module

The Prediction Module generates predictions using trained models and provides confidence scores and prediction intervals.

#### Key Classes and Functions

- `UFLPredictor`: Main prediction interface
- `MLLayerPredictor`: Wrapper for integration with other layers
- `PredictionConfig`: Configures prediction parameters

#### Prediction Types

1. **Fantasy Points**:
   - Total fantasy points
   - Position-specific fantasy points

2. **Statistical Performance**:
   - Passing yards, touchdowns, interceptions
   - Rushing yards, touchdowns
   - Receiving yards, receptions, touchdowns

3. **Game Outcomes**:
   - Win/loss probability
   - Point spread
   - Over/under

#### Confidence Scoring

The module provides confidence scores for predictions based on:

- Model uncertainty
- Historical accuracy for similar players
- Data quality and completeness

#### Example Usage

```python
from ml_layer.prediction import UFLPredictor

# Initialize predictor
predictor = UFLPredictor()

# Load model
predictor.load_model("fantasy_points_qb", "models/checkpoints/qb_fantasy_points.joblib")

# Generate prediction for a player
prediction = predictor.predict(player_data)

print(f"Predicted fantasy points: {prediction['value']}")
print(f"Confidence: {prediction['confidence']}")
print(f"Prediction interval: [{prediction['lower_bound']}, {prediction['upper_bound']}]")
```

## Configuration

The ML Layer is highly configurable through environment variables and configuration files.

### Environment Variables

```bash
# Enable/disable ML Layer
PREDICTION_ML_ENABLED=true

# Model path
PREDICTION_ML_MODEL_PATH=models/checkpoints/latest.joblib

# Feature engineering settings
PREDICTION_ML_ROLLING_WINDOW=3
PREDICTION_ML_ADVANCED_METRICS=true
```

### Configuration Files

The ML Layer uses several configuration files:

1. **Feature Engineering Configuration**:
   - `ml_layer/feature_engineering/config.py`
   - Controls feature generation parameters

2. **Model Selection Configuration**:
   - `ml_layer/model_selection/config.py`
   - Defines model hyperparameters

3. **Training Configuration**:
   - `ml_layer/training/config.py`
   - Sets training parameters

4. **Prediction Configuration**:
   - `ml_layer/prediction/config.py`
   - Configures prediction behavior

## Integration with Other Layers

The ML Layer integrates with the other layers of the Triple-Layer Prediction Engine:

1. **Integration with API/Local AI Layer**:
   - Provides baseline predictions
   - Shares feature data for context

2. **Integration with Cloud AI Layer**:
   - Contributes to weighted predictions
   - Provides confidence scores

## Error Handling

The ML Layer includes comprehensive error handling:

1. **Feature Engineering Errors**:
   - Missing data handling
   - Invalid data detection
   - Feature generation fallbacks

2. **Model Errors**:
   - Model loading failures
   - Prediction generation errors
   - Out-of-bounds predictions

## Performance Considerations

For optimal performance when using the ML Layer:

1. **Batch Processing**: Use batch predictions for multiple players
2. **Feature Caching**: Enable feature caching for repeated predictions
3. **Model Selection**: Use simpler models for real-time predictions
4. **Parallel Processing**: Enable parallel processing for batch predictions

## Example Workflows

### Training New Models

```python
from ml_layer.feature_engineering import UFLDataIntegrator
from ml_layer.model_selection import ModelRegistry, HyperparameterTuner
from ml_layer.training import TrainingPipeline

# Load and prepare data
integrator = UFLDataIntegrator()
data = integrator.prepare_training_dataset()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['features'], 
    data['target'], 
    test_size=0.2, 
    random_state=42
)

# Get model
registry = ModelRegistry()
model_class = registry.get_model("QB", "fantasy_points")

# Tune hyperparameters
tuner = HyperparameterTuner()
best_params = tuner.tune(model_class, X_train, y_train)

# Initialize model with tuned parameters
model = model_class(**best_params)

# Train model
pipeline = TrainingPipeline()
result = pipeline.train(model, X_train, y_train, validation_data=(X_test, y_test))

print(f"Model trained and saved to {result['checkpoint_path']}")
```

### Generating Predictions

```python
from ml_layer.prediction import UFLPredictor
from ml_layer.feature_engineering import Preprocessor

# Initialize components
predictor = UFLPredictor()
preprocessor = Preprocessor()

# Load model
predictor.load_model("fantasy_points", "models/checkpoints/latest.joblib")

# Prepare player data
player_data = {
    "player_id": "UFL12345",
    "name": "John Smith",
    "position": "QB",
    "team": "Birmingham Stallions",
    "opponent": "Michigan Panthers",
    "home_game": True,
    # Additional player statistics...
}

# Preprocess player data
processed_data = preprocessor.process_player(player_data)

# Generate prediction
prediction = predictor.predict(processed_data)

print(f"Predicted fantasy points: {prediction['value']}")
print(f"Confidence: {prediction['confidence']}")
```

## Testing

The ML Layer includes comprehensive tests:

1. **Unit Tests**:
   - `tests/unit/ml_layer/test_feature_engineering.py`
   - `tests/unit/ml_layer/test_model_selection.py`
   - `tests/unit/ml_layer/test_training_pipeline.py`
   - `tests/unit/ml_layer/test_prediction.py`

2. **Integration Tests**:
   - `tests/integration/ml_layer/test_ml_layer_integration.py`

3. **Performance Tests**:
   - `tests/performance/ml_layer/test_ml_layer_performance.py`

## Source Code

The ML Layer source code is available in the BallBetz-Alpha repository:

[https://github.com/clduab11/BallBetz-Alpha/tree/main/ml_layer](https://github.com/clduab11/BallBetz-Alpha/tree/main/ml_layer)