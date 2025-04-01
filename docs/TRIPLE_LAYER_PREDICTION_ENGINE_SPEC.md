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
```

#### 2. Provider-Specific Clients

```python
# TDD Anchor: test_ollama_client
class OllamaClient:
    """Client for Ollama API."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
    
    def generate(self, model: str, prompt: str, 
                max_tokens: int = 100, 
                temperature: float = 0.0) -> str:
        """
        Generate text using Ollama model.
        
        Args:
            model: Model name
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        response = self.session.post(
            url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        return response.json().get("response", "")
```

```python
# TDD Anchor: test_openai_client
class OpenAIClient:
    """Client for OpenAI API."""
    
    def __init__(self, api_key: str, timeout: int = 30):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        
        # Import OpenAI library
        import openai
        self.openai = openai
        self.openai.api_key = api_key
    
    def generate(self, model: str, prompt: str, 
                max_tokens: int = 100, 
                temperature: float = 0.0) -> str:
        """
        Generate text using OpenAI model.
        
        Args:
            model: Model name
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        response = self.openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=self.timeout
        )
        
        return response.choices[0].text.strip()
```

#### 3. Prediction Orchestrator

```python
# TDD Anchor: test_transformer_prediction_orchestrator
def predict_with_transformer(player_data: pd.DataFrame, 
                           config: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate predictions using transformer models.
    
    Args:
        player_data: DataFrame containing player data
        config: Configuration dictionary
        
    Returns:
        DataFrame with predictions added
    """
    # Initialize results DataFrame
    results = pd.DataFrame(index=player_data.index)
    
    # Initialize model interface
    model_interface = TransformerModelInterface(config)
    
    # Generate predictions for each player
    for idx, player in player_data.iterrows():
        try:
            # Generate prediction
            prediction = model_interface.generate_prediction(player)
            
            # Add prediction to results
            results.loc[idx, "transformer_prediction"] = prediction.get("value", 0.0)
            results.loc[idx, "transformer_confidence"] = prediction.get("confidence", 0.0)
            
        except Exception as e:
            logger.error(f"Error generating transformer prediction for player {player.get('name')}: {str(e)}")
            # Set default values
            results.loc[idx, "transformer_prediction"] = 0.0
            results.loc[idx, "transformer_confidence"] = 0.0
    
    return results
```

## Layer 3: Cloud AI Layer

### Purpose
The Cloud AI Layer provides advanced contextual analysis by incorporating external factors and cross-league patterns, with configurable weights for different prediction components.

### Components

#### 1. External Factors Integration

```python
# TDD Anchor: test_external_factors_integration
def integrate_external_factors(player_data: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate external factors that may affect player performance.
    
    Args:
        player_data: DataFrame containing player data
        
    Returns:
        DataFrame with external factors added
    """
    # Initialize results DataFrame
    results = pd.DataFrame(index=player_data.index)
    
    # Add weather data
    results = add_weather_data(player_data, results)
    
    # Add injury reports
    results = add_injury_data(player_data, results)
    
    # Add team matchup data
    results = add_matchup_data(player_data, results)
    
    # Add historical performance against opponent
    results = add_historical_matchup_data(player_data, results)
    
    # Add social media sentiment
    results = add_social_sentiment(player_data, results)
    
    return results
```

#### 2. Cross-League Pattern Analysis

```python
# TDD Anchor: test_cross_league_pattern_analysis
def analyze_cross_league_patterns(player_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze patterns across different leagues (UFL, NFL, NCAA).
    
    Args:
        player_data: DataFrame containing player data
        
    Returns:
        DataFrame with cross-league pattern features
    """
    # Initialize results DataFrame
    results = pd.DataFrame(index=player_data.index)
    
    # Add NFL comparison metrics
    results = add_nfl_comparison_metrics(player_data, results)
    
    # Add NCAA historical performance
    results = add_ncaa_historical_performance(player_data, results)
    
    # Add league transition success metrics
    results = add_league_transition_metrics(player_data, results)
    
    return results
```

#### 3. Weighted Prediction Combiner

```python
# TDD Anchor: test_weighted_prediction_combiner
def combine_predictions(ml_predictions: pd.DataFrame,
                       transformer_predictions: pd.DataFrame,
                       external_factors: pd.DataFrame,
                       cross_league_patterns: pd.DataFrame,
                       weights: Dict[str, float]) -> pd.DataFrame:
    """
    Combine predictions from different sources with configurable weights.
    
    Args:
        ml_predictions: DataFrame with ML predictions
        transformer_predictions: DataFrame with transformer predictions
        external_factors: DataFrame with external factors
        cross_league_patterns: DataFrame with cross-league patterns
        weights: Dictionary of weights for different components
        
    Returns:
        DataFrame with combined predictions
    """
    # Initialize results DataFrame
    results = pd.DataFrame(index=ml_predictions.index)
    
    # Get weights
    ml_weight = weights.get("ml", 0.4)
    transformer_weight = weights.get("transformer", 0.3)
    external_weight = weights.get("external", 0.2)
    cross_league_weight = weights.get("cross_league", 0.1)
    
    # Combine predictions
    for idx in results.index:
        # Get individual predictions
        ml_pred = ml_predictions.loc[idx, "ml_prediction"]
        transformer_pred = transformer_predictions.loc[idx, "transformer_prediction"]
        
        # Get external factor adjustment
        external_adjustment = calculate_external_adjustment(external_factors, idx)
        
        # Get cross-league adjustment
        cross_league_adjustment = calculate_cross_league_adjustment(cross_league_patterns, idx)
        
        # Calculate weighted prediction
        weighted_prediction = (
            ml_pred * ml_weight +
            transformer_pred * transformer_weight +
            external_adjustment * external_weight +
            cross_league_adjustment * cross_league_weight
        )
        
        # Add to results
        results.loc[idx, "final_prediction"] = weighted_prediction
        
        # Add component contributions for transparency
        results.loc[idx, "ml_contribution"] = ml_pred * ml_weight
        results.loc[idx, "transformer_contribution"] = transformer_pred * transformer_weight
        results.loc[idx, "external_contribution"] = external_adjustment * external_weight
        results.loc[idx, "cross_league_contribution"] = cross_league_adjustment * cross_league_weight
    
    return results
```

## Data Sources and Preprocessing

### Data Sources

1. **Primary Data Sources**
   - UFL official statistics
   - Team websites and press releases
   - Sports data APIs (e.g., SportRadar, Stats Perform)

2. **Secondary Data Sources**
   - Weather data for game locations
   - Social media sentiment analysis
   - Injury reports and team announcements
   - Historical player performance from NCAA and NFL (for cross-league analysis)

### Preprocessing Requirements

```python
# TDD Anchor: test_data_preprocessing_pipeline
def preprocess_data(raw_data: List[Dict], 
                   config: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess raw data for the prediction engine.
    
    Args:
        raw_data: List of dictionaries containing raw data
        config: Configuration dictionary
        
    Returns:
        Preprocessed DataFrame
    """
    # Convert to DataFrame
    df = pd.DataFrame(raw_data)
    
    # Clean data
    df = clean_data(df, config.get("cleaning_params", {}))
    
    # Handle missing values
    df = handle_missing_values(df, config.get("missing_value_strategy", "mean"))
    
    # Normalize numerical features
    df = normalize_features(df, config.get("normalization_method", "standard"))
    
    # Encode categorical features
    df = encode_categorical_features(df, config.get("encoding_method", "one-hot"))
    
    # Feature selection
    df = select_features(df, config.get("feature_selection", {}))
    
    return df
```

## Prediction Types and Accuracy Metrics

### Prediction Types

1. **Fantasy Points**
   - Total fantasy points per game
   - Position-specific fantasy points

2. **Statistical Performance**
   - Passing yards, touchdowns, interceptions
   - Rushing yards, touchdowns
   - Receiving yards, receptions, touchdowns
   - Defensive statistics

3. **Game Outcomes**
   - Win/loss predictions
   - Point spreads
   - Over/under totals

### Accuracy Metrics

```python
# TDD Anchor: test_accuracy_metrics
def evaluate_predictions(predictions: pd.DataFrame, 
                        actual: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate prediction accuracy using multiple metrics.
    
    Args:
        predictions: DataFrame containing predictions
        actual: DataFrame containing actual values
        
    Returns:
        Dictionary of accuracy metrics
    """
    # Initialize results dictionary
    results = {}
    
    # Calculate mean absolute error
    mae = calculate_mae(predictions, actual)
    results["mae"] = mae
    
    # Calculate root mean squared error
    rmse = calculate_rmse(predictions, actual)
    results["rmse"] = rmse
    
    # Calculate R-squared
    r2 = calculate_r2(predictions, actual)
    results["r2"] = r2
    
    # Calculate position-specific metrics
    position_metrics = calculate_position_metrics(predictions, actual)
    results["position_metrics"] = position_metrics
    
    # Calculate calibration metrics
    calibration = calculate_calibration(predictions, actual)
    results["calibration"] = calibration
    
    # Calculate prediction interval coverage
    interval_coverage = calculate_interval_coverage(predictions, actual)
    results["interval_coverage"] = interval_coverage
    
    return results
```

## Integration Points

### Integration with Existing Components

```python
# TDD Anchor: test_system_integration
class PredictionEngineIntegrator:
    """Integrates the Triple-Layer Prediction Engine with other system components."""
    
    def __init__(self, config_path: str):
        """
        Initialize the integrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.prediction_engine = initialize_prediction_engine(self.config)
    
    def integrate_with_data_pipeline(self, pipeline_output: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate with the data pipeline.
        
        Args:
            pipeline_output: Output from the data pipeline
            
        Returns:
            DataFrame with predictions added
        """
        # Preprocess data
        preprocessed_data = preprocess_data(pipeline_output, self.config.get("preprocessing", {}))
        
        # Generate predictions
        predictions = self.prediction_engine.predict(preprocessed_data)
        
        # Combine with original data
        result = pd.concat([pipeline_output, predictions], axis=1)
        
        return result
    
    def integrate_with_web_interface(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Format predictions for the web interface.
        
        Args:
            predictions: DataFrame containing predictions
            
        Returns:
            Dictionary formatted for web interface
        """
        # Format predictions for web display
        web_data = format_for_web(predictions)
        
        return web_data
    
    def integrate_with_lineup_optimizer(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Provide predictions to the lineup optimizer.
        
        Args:
            predictions: DataFrame containing predictions
            
        Returns:
            Dictionary for lineup optimizer
        """
        # Format predictions for lineup optimizer
        optimizer_data = format_for_optimizer(predictions)
        
        return optimizer_data
```

## Environment Variable Strategy

```python
# TDD Anchor: test_environment_config
def load_prediction_engine_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Configuration dictionary
    """
    # Load .env file if it exists
    load_dotenv()
    
    # ML Layer configuration
    ml_config = {
        "enabled": os.getenv("PREDICTION_ML_ENABLED", "true").lower() == "true",
        "model_path": os.getenv("PREDICTION_ML_MODEL_PATH", "models/checkpoints/latest.joblib"),
        "feature_engineering": {
            "rolling_window": int(os.getenv("PREDICTION_ML_ROLLING_WINDOW", "3")),
            "include_advanced_metrics": os.getenv("PREDICTION_ML_ADVANCED_METRICS", "true").lower() == "true"
        }
    }
    
    # Transformer Layer configuration
    transformer_config = {
        "enabled": os.getenv("PREDICTION_TRANSFORMER_ENABLED", "true").lower() == "true",
        "provider": os.getenv("PREDICTION_TRANSFORMER_PROVIDER", "ollama"),
        "base_url": os.getenv("PREDICTION_TRANSFORMER_BASE_URL", "http://localhost:10000"),
        "model_name": os.getenv("PREDICTION_TRANSFORMER_MODEL", "llama3.2-3b-instruct"),
        "api_key": os.getenv("PREDICTION_TRANSFORMER_API_KEY", ""),
        "timeout": int(os.getenv("PREDICTION_TRANSFORMER_TIMEOUT", "30"))
    }
    
    # Cloud AI Layer configuration
    cloud_config = {
        "enabled": os.getenv("PREDICTION_CLOUD_ENABLED", "true").lower() == "true",
        "external_factors": {
            "weather_enabled": os.getenv("PREDICTION_CLOUD_WEATHER", "true").lower() == "true",
            "injuries_enabled": os.getenv("PREDICTION_CLOUD_INJURIES", "true").lower() == "true",
            "social_enabled": os.getenv("PREDICTION_CLOUD_SOCIAL", "false").lower() == "true"
        },
        "cross_league": {
            "nfl_enabled": os.getenv("PREDICTION_CLOUD_NFL", "true").lower() == "true",
            "ncaa_enabled": os.getenv("PREDICTION_CLOUD_NCAA", "true").lower() == "true"
        }
    }
    
    # Weights configuration
    weights_config = {
        "ml": float(os.getenv("PREDICTION_WEIGHT_ML", "0.4")),
        "transformer": float(os.getenv("PREDICTION_WEIGHT_TRANSFORMER", "0.3")),
        "external": float(os.getenv("PREDICTION_WEIGHT_EXTERNAL", "0.2")),
        "cross_league": float(os.getenv("PREDICTION_WEIGHT_CROSS_LEAGUE", "0.1"))
    }
    
    # Combine all configurations
    config = {
        "ml_layer": ml_config,
        "transformer_layer": transformer_config,
        "cloud_layer": cloud_config,
        "weights": weights_config
    }
    
    return config
```

## Error Handling and Fallback Mechanisms

```python
# TDD Anchor: test_error_handling
class PredictionEngineErrorHandler:
    """Handles errors and provides fallback mechanisms for the prediction engine."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the error handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fallback_enabled = config.get("fallback_enabled", True)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
    def handle_ml_layer_error(self, error: Exception, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle errors in the ML Layer.
        
        Args:
            error: The exception that occurred
            player_data: Player data being processed
            
        Returns:
            DataFrame with fallback predictions
        """
        logger.error(f"ML Layer error: {str(error)}")
        
        if self.fallback_enabled:
            # Use simple statistical fallback
            return generate_statistical_fallback(player_data)
        else:
            # Re-raise the error
            raise error
    
    def handle_transformer_layer_error(self, error: Exception, player_data: pd.DataFrame,
                                     ml_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Handle errors in the Transformer Layer.
        
        Args:
            error: The exception that occurred
            player_data: Player data being processed
            ml_predictions: Predictions from the ML Layer
            
        Returns:
            DataFrame with fallback predictions
        """
        logger.error(f"Transformer Layer error: {str(error)}")
        
        if self.fallback_enabled:
            # Use ML predictions as fallback
            return ml_predictions
        else:
            # Re-raise the error
            raise error
    
    def handle_cloud_layer_error(self, error: Exception, player_data: pd.DataFrame,
                               ml_predictions: pd.DataFrame,
                               transformer_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Handle errors in the Cloud Layer.
        
        Args:
            error: The exception that occurred
            player_data: Player data being processed
            ml_predictions: Predictions from the ML Layer
            transformer_predictions: Predictions from the Transformer Layer
            
        Returns:
            DataFrame with fallback predictions
        """
        logger.error(f"Cloud Layer error: {str(error)}")
        
        if self.fallback_enabled:
            # Combine ML and transformer predictions with default weights
            return combine_ml_transformer_predictions(ml_predictions, transformer_predictions)
        else:
            # Re-raise the error
            raise error
```

## Testing Strategy

### Unit Tests

```python
# TDD Anchor: test_ml_layer
def test_ml_layer():
    """Test the ML Layer components."""
    # Test feature engineering
    test_feature_engineering()
    
    # Test model selection
    test_model_selection()
    
    # Test training pipeline
    test_training_pipeline()
    
    # Test prediction module
    test_ml_prediction_module()
```

```python
# TDD Anchor: test_transformer_layer
def test_transformer_layer():
    """Test the Transformer Layer components."""
    # Test model interface
    test_transformer_model_interface()
    
    # Test provider-specific clients
    test_ollama_client()
    test_openai_client()
    
    # Test prediction orchestrator
    test_transformer_prediction_orchestrator()
```

```python
# TDD Anchor: test_cloud_layer
def test_cloud_layer():
    """Test the Cloud Layer components."""
    # Test external factors integration
    test_external_factors_integration()
    
    # Test cross-league pattern analysis
    test_cross_league_pattern_analysis()
    
    # Test weighted prediction combiner
    test_weighted_prediction_combiner()
```

### Integration Tests

```python
# TDD Anchor: test_integration
def test_integration():
    """Test the integration of all layers."""
    # Test end-to-end prediction pipeline
    test_end_to_end_pipeline()
    
    # Test integration with data pipeline
    test_data_pipeline_integration()
    
    # Test integration with web interface
    test_web_interface_integration()
    
    # Test integration with lineup optimizer
    test_lineup_optimizer_integration()
```

## Future Expansion

### NFL and NCAA Expansion

```python
# TDD Anchor: test_league_expansion
