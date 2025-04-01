# Triple-Layer Prediction Engine Test Examples

This document provides example test implementations for the TDD anchors defined in the Triple-Layer Prediction Engine specification. These examples use pytest and can be adapted to your testing framework of choice.

## Table of Contents

1. [ML Layer Tests](#ml-layer-tests)
2. [Transformer Layer Tests](#transformer-layer-tests)
3. [Cloud AI Layer Tests](#cloud-ai-layer-tests)
4. [Integration Tests](#integration-tests)

## ML Layer Tests

### Feature Engineering Tests

```python
# tests/ml_layer/test_feature_engineering.py

import pytest
import pandas as pd
import numpy as np
from prediction_engine.ml_layer.feature_engineering import engineer_features

# TDD Anchor: test_feature_engineering_module
class TestFeatureEngineering:
    
    def setup_method(self):
        """Set up test data."""
        # Create sample player data
        self.player_data = pd.DataFrame({
            'name': ['Player A', 'Player B', 'Player C'],
            'position': ['QB', 'RB', 'WR'],
            'team': ['Team X', 'Team Y', 'Team Z'],
            'passing_yards': [300, 0, 0],
            'rushing_yards': [20, 120, 10],
            'receiving_yards': [0, 30, 150],
            'passing_touchdowns': [2, 0, 0],
            'rushing_touchdowns': [0, 1, 0],
            'receiving_touchdowns': [0, 0, 2],
            'games_played': [10, 10, 10]
        })
    
    def test_engineer_features_complete(self):
        """Test the complete feature engineering pipeline."""
        features = engineer_features(self.player_data)
        
        # Check that all expected feature types are present
        assert any('rolling_avg' in col for col in features.columns)
        assert any('pos_' in col for col in features.columns)  # Position encoding
        
        # Check that the number of features is reasonable
        assert len(features.columns) > 10
        
        # Check that no NaN values are present
        assert not features.isna().any().any()
```

### Model Selection Tests

```python
# tests/ml_layer/test_model_selection.py

import pytest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from prediction_engine.ml_layer.model_selection import select_model

# TDD Anchor: test_model_selection_module
class TestModelSelection:
    
    def test_select_model_qb_fantasy_points(self):
        """Test model selection for QB fantasy points."""
        model = select_model("QB", "fantasy_points")
        
        # Check that the correct model type is returned
        assert isinstance(model, RandomForestRegressor)
    
    def test_select_model_default(self):
        """Test default model selection."""
        model = select_model("UNKNOWN", "unknown_metric")
        
        # Check that a default model is returned
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 100
        assert model.random_state == 42
```

### Training Pipeline Tests

```python
# tests/ml_layer/test_training_pipeline.py

import pytest
import pandas as pd
import numpy as np
from prediction_engine.ml_layer.training import train_ml_models

# TDD Anchor: test_training_pipeline
class TestTrainingPipeline:
    
    def setup_method(self):
        """Set up test data."""
        # Create sample historical data
        np.random.seed(42)
        n_samples = 100
        
        self.historical_data = pd.DataFrame({
            'name': [f'Player {i}' for i in range(n_samples)],
            'position': np.random.choice(['QB', 'RB', 'WR'], n_samples),
            'team': np.random.choice(['Team A', 'Team B', 'Team C'], n_samples),
            'passing_yards': np.random.randint(0, 400, n_samples),
            'rushing_yards': np.random.randint(0, 150, n_samples),
            'receiving_yards': np.random.randint(0, 200, n_samples),
            'passing_touchdowns': np.random.randint(0, 4, n_samples),
            'rushing_touchdowns': np.random.randint(0, 3, n_samples),
            'receiving_touchdowns': np.random.randint(0, 3, n_samples),
            'games_played': np.random.randint(1, 17, n_samples),
            'fantasy_points': np.random.uniform(0, 30, n_samples)
        })
    
    def test_train_ml_models(self):
        """Test training of ML models."""
        results = train_ml_models(self.historical_data, target_col="fantasy_points", cv_splits=3)
        
        # Check that models are trained for each position
        assert "QB" in results
        assert "RB" in results
        assert "WR" in results
        
        # Check that each position has the expected results
        for position in ["QB", "RB", "WR"]:
            assert "model" in results[position]
            assert "train_score" in results[position]
            assert "val_score" in results[position]
            assert "feature_importance" in results[position]
```

## Transformer Layer Tests

### Model Interface Tests

```python
# tests/transformer_layer/test_model_interface.py

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from prediction_engine.transformer_layer.model_interface import TransformerModelInterface

# TDD Anchor: test_transformer_model_interface
class TestTransformerModelInterface:
    
    def setup_method(self):
        """Set up test data and mocks."""
        self.config = {
            "provider": "ollama",
            "base_url": "http://localhost:10000",
            "model_name": "llama3.2-3b-instruct",
            "timeout": 30,
            "max_retries": 3
        }
        
        self.player_data = pd.Series({
            'name': 'Player A',
            'position': 'QB',
            'team': 'Team X',
            'passing_yards': 300,
            'rushing_yards': 20,
            'receiving_yards': 0,
            'passing_touchdowns': 2,
            'rushing_touchdowns': 0,
            'receiving_touchdowns': 0,
            'games_played': 10
        })
    
    @patch('prediction_engine.transformer_layer.model_interface.OllamaClient')
    def test_initialize_client_ollama(self, mock_ollama_client):
        """Test initialization of Ollama client."""
        # Setup mock
        mock_client = MagicMock()
        mock_ollama_client.return_value = mock_client
        
        # Create interface
        interface = TransformerModelInterface(self.config)
        
        # Check that the correct client is initialized
        mock_ollama_client.assert_called_once_with(
            self.config["base_url"],
            self.config["timeout"]
        )
        assert interface.client == mock_client
```

### Provider-Specific Client Tests

```python
# tests/transformer_layer/test_clients.py

import pytest
import requests
from unittest.mock import MagicMock, patch
from prediction_engine.transformer_layer.clients import OllamaClient, OpenAIClient

# TDD Anchor: test_ollama_client
class TestOllamaClient:
    
    def setup_method(self):
        """Set up test data."""
        self.base_url = "http://localhost:10000"
        self.client = OllamaClient(self.base_url)
    
    @patch('requests.Session.post')
    def test_generate(self, mock_post):
        """Test generation of text using Ollama model."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "test response"}
        mock_post.return_value = mock_response
        
        # Generate text
        response = self.client.generate(
            model="test-model",
            prompt="test prompt",
            max_tokens=100,
            temperature=0.0
        )
        
        # Check that the correct request is made
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["url"] == f"{self.base_url}/api/generate"
        assert kwargs["json"]["model"] == "test-model"
        assert kwargs["json"]["prompt"] == "test prompt"
        assert kwargs["json"]["options"]["temperature"] == 0.0
        assert kwargs["json"]["options"]["num_predict"] == 100
        
        # Check that the response is correct
        assert response == "test response"
```

## Cloud AI Layer Tests

### External Factors Integration Tests

```python
# tests/cloud_layer/test_external_factors.py

import pytest
import pandas as pd
from unittest.mock import patch
from prediction_engine.cloud_layer.external_factors import integrate_external_factors

# TDD Anchor: test_external_factors_integration
class TestExternalFactorsIntegration:
    
    def setup_method(self):
        """Set up test data."""
        self.player_data = pd.DataFrame({
            'name': ['Player A', 'Player B', 'Player C'],
            'position': ['QB', 'RB', 'WR'],
            'team': ['Team X', 'Team Y', 'Team Z'],
            'opponent': ['Team Y', 'Team Z', 'Team X'],
            'game_id': [1001, 1001, 1002]
        })
    
    @patch('prediction_engine.cloud_layer.external_factors.add_weather_data')
    @patch('prediction_engine.cloud_layer.external_factors.add_injury_data')
    @patch('prediction_engine.cloud_layer.external_factors.add_matchup_data')
    def test_integrate_external_factors(self, mock_matchup, mock_injury, mock_weather):
        """Test integration of all external factors."""
        # Setup mocks
        mock_weather.return_value = pd.DataFrame(index=self.player_data.index).assign(weather_factor=0.1)
        mock_injury.return_value = pd.DataFrame(index=self.player_data.index).assign(injury_factor=-0.2)
        mock_matchup.return_value = pd.DataFrame(index=self.player_data.index).assign(matchup_factor=0.3)
        
        # Integrate external factors
        result = integrate_external_factors(self.player_data)
        
        # Check that all mocks are called
        mock_weather.assert_called_once()
        mock_injury.assert_called_once()
        mock_matchup.assert_called_once()
        
        # Check that all factors are in the result
        assert 'weather_factor' in result.columns
        assert 'injury_factor' in result.columns
        assert 'matchup_factor' in result.columns
```

### Weighted Prediction Combiner Tests

```python
# tests/cloud_layer/test_prediction_combiner.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from prediction_engine.cloud_layer.prediction_combiner import combine_predictions

# TDD Anchor: test_weighted_prediction_combiner
class TestWeightedPredictionCombiner:
    
    def setup_method(self):
        """Set up test data."""
        # Create sample prediction data
        self.ml_predictions = pd.DataFrame({
            'ml_prediction': [15.0, 20.0, 10.0]
        })
        
        self.transformer_predictions = pd.DataFrame({
            'transformer_prediction': [17.0, 18.0, 12.0],
            'transformer_confidence': [0.8, 0.7, 0.9]
        })
        
        self.external_factors = pd.DataFrame({
            'weather_factor': [0.1, -0.2, 0.0],
            'injury_factor': [-0.2, 0.0, -0.1],
            'matchup_factor': [0.3, 0.1, 0.2]
        })
        
        self.cross_league_patterns = pd.DataFrame({
            'nfl_comparison': [0.05, 0.1, -0.05],
            'ncaa_historical': [0.1, 0.0, 0.15]
        })
        
        # Set index for all DataFrames
        index = [101, 102, 103]
        self.ml_predictions.index = index
        self.transformer_predictions.index = index
        self.external_factors.index = index
        self.cross_league_patterns.index = index
        
        # Define weights
        self.weights = {
            'ml': 0.4,
            'transformer': 0.3,
            'external': 0.2,
            'cross_league': 0.1
        }
    
    @patch('prediction_engine.cloud_layer.prediction_combiner.calculate_external_adjustment')
    @patch('prediction_engine.cloud_layer.prediction_combiner.calculate_cross_league_adjustment')
    def test_combine_predictions(self, mock_cross_league, mock_external):
        """Test combination of predictions with weights."""
        # Setup mocks
        mock_external.side_effect = [0.2, -0.1, 0.1]
        mock_cross_league.side_effect = [0.15, 0.1, 0.1]
        
        # Combine predictions
        result = combine_predictions(
            self.ml_predictions,
            self.transformer_predictions,
            self.external_factors,
            self.cross_league_patterns,
            self.weights
        )
        
        # Check that result contains expected columns
        assert 'final_prediction' in result.columns
        assert 'ml_contribution' in result.columns
        assert 'transformer_contribution' in result.columns
        assert 'external_contribution' in result.columns
        assert 'cross_league_contribution' in result.columns
```

## Integration Tests

```python
# tests/integration/test_end_to_end.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from prediction_engine.prediction_engine import PredictionEngine

# TDD Anchor: test_integration
class TestEndToEndPipeline:
    
    def setup_method(self):
        """Set up test data."""
        # Create sample player data
        np.random.seed(42)
        n_samples = 10
        
        self.player_data = pd.DataFrame({
            'name': [f'Player {i}' for i in range(n_samples)],
            'position': np.random.choice(['QB', 'RB', 'WR'], n_samples),
            'team': np.random.choice(['Team A', 'Team B', 'Team C'], n_samples),
            'opponent': np.random.choice(['Team A', 'Team B', 'Team C'], n_samples),
            'passing_yards': np.random.randint(0, 400, n_samples),
            'rushing_yards': np.random.randint(0, 150, n_samples),
            'receiving_yards': np.random.randint(0, 200, n_samples),
            'passing_touchdowns': np.random.randint(0, 4, n_samples),
            'rushing_touchdowns': np.random.randint(0, 3, n_samples),
            'receiving_touchdowns': np.random.randint(0, 3, n_samples),
            'games_played': np.random.randint(1, 17, n_samples)
        })
    
    @patch('prediction_engine.prediction_engine.load_prediction_engine_config')
    def test_end_to_end_prediction(self, mock_load_config):
        """Test end-to-end prediction pipeline."""
        # Setup mock config
        mock_config = {
            'ml_layer': {'enabled': True},
            'transformer_layer': {'enabled': True},
            'cloud_layer': {'enabled': True},
            'weights': {
                'ml': 0.4,
                'transformer': 0.3,
                'external': 0.2,
                'cross_league': 0.1
            }
        }
        mock_load_config.return_value = mock_config
        
        # Create prediction engine with mocked components
        engine = PredictionEngine()
        engine._predict_with_ml = MagicMock()
        engine._predict_with_transformer = MagicMock()
        engine._integrate_external_factors = MagicMock()
        engine._analyze_cross_league_patterns = MagicMock()
        engine._combine_predictions = MagicMock()
        
        # Generate predictions
        engine.predict(self.player_data)
        
        # Check that all components are called
        engine._predict_with_ml.assert_called_once()
        engine._predict_with_transformer.assert_called_once()
        engine._integrate_external_factors.assert_called_once()
        engine._analyze_cross_league_patterns.assert_called_once()
        engine._combine_predictions.assert_called_once()
