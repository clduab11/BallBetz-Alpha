# Triple-Layer Prediction Engine Specification

## Overview

The Triple-Layer Prediction Engine is a sophisticated prediction system for BallBetz-Alpha that combines statistical machine learning, transformer-based AI models, and cloud-based contextual analysis to generate high-accuracy predictions for UFL 2025 season with future expansion capabilities to NFL and NCAA.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Layer 1: ML Layer](#layer-1-ml-layer)
3. [Layer 2: API/Local AI Layer](#layer-2-apilocal-ai-layer)
4. [Layer 3: Cloud AI Layer](#layer-3-cloud-ai-layer)
5. [Data Sources and Preprocessing](#data-sources-and-preprocessing)
6. [Prediction Types and Accuracy Metrics](#prediction-types-and-accuracy-metrics)
7. [Integration Points](#integration-points)
8. [Environment Variable Strategy](#environment-variable-strategy)
9. [Error Handling and Fallback Mechanisms](#error-handling-and-fallback-mechanisms)
10. [Testing Strategy](#testing-strategy)
11. [Future Expansion](#future-expansion)

## Architecture Overview

The Triple-Layer Prediction Engine employs a hierarchical approach where each layer builds upon the previous one:

```
┌─────────────────────────────────────────────────────────────┐
│                  Triple-Layer Prediction Engine              │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: ML Layer (scikit-learn)                            │
│ - Statistical analysis                                       │
│ - Feature engineering                                        │
│ - Ensemble models                                            │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: API/Local AI Layer (Transformer Models)            │
│ - Pattern recognition                                        │
│ - Contextual understanding                                   │
│ - Anomaly detection                                          │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Cloud AI Layer (Advanced Contextual Analysis)      │
│ - External factors integration                               │
│ - Cross-league patterns                                      │
│ - Configurable weights                                       │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1: ML Layer

### Purpose
The ML Layer provides statistical analysis and baseline predictions using traditional machine learning techniques.

### Components

#### 1. Feature Engineering Module

```python
# TDD Anchor: test_feature_engineering_module
def engineer_features(player_data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw player data.
    
    Args:
        player_data: DataFrame containing raw player statistics
        
    Returns:
        DataFrame with engineered features
    """
    # Initialize feature set
    features = pd.DataFrame()
    
    # Extract base features
    features = extract_base_features(player_data)
    
    # Calculate rolling averages
    features = add_rolling_averages(features, player_data)
    
    # Add positional encoding
    features = add_positional_encoding(features)
    
    # Add matchup features
    features = add_matchup_features(features, player_data)
    
    # Add advanced metrics
    features = add_advanced_metrics(features)
    
    return features
```

#### 2. Model Selection Module

```python
# TDD Anchor: test_model_selection_module
def select_model(position: str, prediction_type: str) -> BaseEstimator:
    """
    Select the appropriate ML model based on player position and prediction type.
    
    Args:
        position: Player position (QB, RB, WR, TE, etc.)
        prediction_type: Type of prediction (fantasy_points, yards, touchdowns, etc.)
        
    Returns:
        Appropriate scikit-learn model
    """
    # Get model configuration from environment
    config = get_model_config()
    
    # Select model based on position and prediction type
    if position == "QB":
        if prediction_type == "fantasy_points":
            return RandomForestRegressor(**config.get("qb_fantasy_points", {}))
        elif prediction_type == "passing_yards":
            return GradientBoostingRegressor(**config.get("qb_passing_yards", {}))
        # Additional prediction types...
    
    # Handle other positions similarly
    elif position == "RB":
        # RB-specific models
        pass
    
    # Default model if no specific configuration
    return RandomForestRegressor(n_estimators=100, random_state=42)
```

#### 3. Training Pipeline

```python
# TDD Anchor: test_training_pipeline
def train_ml_models(historical_data: pd.DataFrame, 
                    target_col: str = "fantasy_points",
                    cv_splits: int = 5) -> Dict[str, Any]:
    """
    Train ML models for different positions and prediction types.
    
    Args:
        historical_data: DataFrame containing historical player data
        target_col: Target column for prediction
        cv_splits: Number of cross-validation splits
        
    Returns:
        Dictionary of trained models and performance metrics
    """
    # Group data by position
    position_groups = historical_data.groupby("position")
    
    # Initialize results dictionary
    results = {}
    
    # Train models for each position
    for position, group_data in position_groups:
        # Engineer features
        X = engineer_features(group_data)
        y = group_data[target_col]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Select and train model
        model = select_model(position, target_col)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        # Store results
        results[position] = {
            "model": model,
            "train_score": train_score,
            "val_score": val_score,
            "feature_importance": get_feature_importance(model, X.columns)
        }
    
    return results
```

#### 4. Prediction Module

```python
# TDD Anchor: test_ml_prediction_module
def predict_with_ml(player_data: pd.DataFrame, 
                   models: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate predictions using trained ML models.
    
    Args:
        player_data: DataFrame containing player data
        models: Dictionary of trained models by position
        
    Returns:
        DataFrame with predictions added
    """
    # Initialize results DataFrame
    results = pd.DataFrame(index=player_data.index)
    
    # Group data by position
    position_groups = player_data.groupby("position")
    
    # Generate predictions for each position
    for position, group_data in position_groups:
        if position in models:
            # Engineer features
            X = engineer_features(group_data)
            
            # Get model
            model = models[position]["model"]
            
            # Generate predictions
            predictions = model.predict(X)
            
            # Add predictions to results
            for idx, pred in zip(group_data.index, predictions):
                results.loc[idx, "ml_prediction"] = pred
                
                # Add prediction intervals if available
                if hasattr(model, "estimators_"):
                    lower, upper = calculate_prediction_intervals(model, X.loc[idx].values.reshape(1, -1))
                    results.loc[idx, "ml_lower_bound"] = lower
                    results.loc[idx, "ml_upper_bound"] = upper
    
    return results
```

## Layer 2: API/Local AI Layer

### Purpose
The API/Local AI Layer enhances predictions using transformer-based models that can identify patterns and contextual relationships not captured by traditional ML.

### Components

#### 1. Model Interface

```python
# TDD Anchor: test_transformer_model_interface
class TransformerModelInterface:
    """Interface for transformer-based models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformer model interface.
        
        Args:
            config: Configuration dictionary
        """
        self.provider = config.get("provider", "ollama")
        self.base_url = config.get("base_url", "")
        self.model_name = config.get("model_name", "")
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        
        # Initialize provider-specific client
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == "ollama":
            return OllamaClient(self.base_url, self.timeout)
        elif self.provider == "openai":
            return OpenAIClient(self.api_key, self.timeout)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_prediction(self, player_data: pd.Series) -> Dict[str, Any]:
        """
        Generate prediction for a player using the transformer model.
        
        Args:
            player_data: Series containing player data
            
        Returns:
            Dictionary with prediction and confidence
        """
        # Format prompt
        prompt = self._format_prompt(player_data)
        
        # Call model
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            max_tokens=100,
            temperature=0.0
        )
        
        # Parse response
        return self._parse_response(response)
    
    def _format_prompt(self, player_data: pd.Series) -> str:
        """Format player data into a prompt for the model."""
        # Implementation depends on model requirements
        pass
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response into structured prediction."""
        # Implementation depends on model output format
        pass