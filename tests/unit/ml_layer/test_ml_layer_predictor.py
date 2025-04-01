"""
Unit tests for the MLLayerPredictor class.

This module tests the MLLayerPredictor class, which serves as a wrapper
around the UFLPredictor to make it compatible with the Triple-Layer
Prediction Engine orchestrator.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from ml_layer.prediction.ml_layer_predictor import MLLayerPredictor
from ml_layer.prediction.predictor import UFLPredictor
from ml_layer.prediction.exceptions import ModelLoadError, InferenceFailed


@pytest.fixture
def mock_ufl_predictor():
    """Create a mock UFLPredictor instance."""
    mock_predictor = MagicMock(spec=UFLPredictor)
    mock_predictor._models = {'default': MagicMock()}
    
    # Mock the predict method
    mock_predictor.predict.return_value = {
        'predictions': [20.5, 15.3, 18.7],
        'confidence': [0.8, 0.7, 0.9],
        'task_type': 'regression'
    }
    
    return mock_predictor


@pytest.fixture
def sample_player_data():
    """Create sample player data for testing."""
    return [
        "{'name': 'Player 1', 'position': 'QB', 'passing_yards': 2500, 'passing_touchdowns': 20}",
        "{'name': 'Player 2', 'position': 'RB', 'rushing_yards': 1000, 'rushing_touchdowns': 8}",
        "{'name': 'Player 3', 'position': 'WR', 'receiving_yards': 1200, 'receiving_touchdowns': 10}"
    ]


def test_ml_layer_predictor_initialization():
    """Test MLLayerPredictor initialization."""
    # Test with default parameters
    with patch('ml_layer.prediction.ml_layer_predictor.UFLPredictor') as mock_ufl:
        mock_ufl.return_value = MagicMock()
        mock_ufl.return_value._models = {}
        
        predictor = MLLayerPredictor()
        
        # Verify UFLPredictor was initialized
        assert mock_ufl.called
        
        # Verify model loading was attempted
        assert mock_ufl.return_value.load_model.called


def test_ml_layer_predictor_with_existing_predictor(mock_ufl_predictor):
    """Test MLLayerPredictor initialization with an existing UFLPredictor."""
    predictor = MLLayerPredictor(ufl_predictor=mock_ufl_predictor)
    
    # Verify the provided predictor was used
    assert predictor.ufl_predictor == mock_ufl_predictor
    
    # Verify model loading was not attempted since models already exist
    assert not mock_ufl_predictor.load_model.called


def test_convert_input(mock_ufl_predictor, sample_player_data):
    """Test the _convert_input method."""
    predictor = MLLayerPredictor(ufl_predictor=mock_ufl_predictor)
    
    # Convert input data
    df = predictor._convert_input(sample_player_data)
    
    # Verify conversion
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert 'name' in df.columns
    assert 'position' in df.columns
    assert df.iloc[0]['name'] == 'Player 1'
    assert df.iloc[1]['position'] == 'RB'


def test_convert_input_with_invalid_data(mock_ufl_predictor):
    """Test the _convert_input method with invalid data."""
    predictor = MLLayerPredictor(ufl_predictor=mock_ufl_predictor)
    
    # Test with invalid input
    invalid_data = ["not a valid dict", "{'name': 'Player 1'}", "also invalid"]
    
    # Should handle invalid data gracefully
    df = predictor._convert_input(invalid_data)
    
    # Verify conversion still produces a DataFrame
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    
    # At least one valid entry should be present
    assert 'name' in df.columns
    assert df.iloc[1]['name'] == 'Player 1'


def test_predict_method(mock_ufl_predictor, sample_player_data):
    """Test the predict method."""
    predictor = MLLayerPredictor(ufl_predictor=mock_ufl_predictor)
    
    # Call predict
    results = predictor.predict(sample_player_data)
    
    # Verify results
    assert isinstance(results, list)
    assert len(results) == 3
    
    # Check structure of each result
    for result in results:
        assert 'input' in result
        assert 'value' in result
        assert 'confidence' in result
        assert 'player_name' in result
    
    # Check specific values
    assert results[0]['value'] == 20.5
    assert results[1]['confidence'] == 0.7
    assert results[2]['player_name'] == 'Player 3'


def test_predict_method_handles_errors(mock_ufl_predictor, sample_player_data):
    """Test that the predict method handles errors gracefully."""
    predictor = MLLayerPredictor(ufl_predictor=mock_ufl_predictor)
    
    # Make the UFLPredictor raise an exception
    mock_ufl_predictor.predict.side_effect = InferenceFailed("Test error")
    
    # Call predict - should not raise an exception
    results = predictor.predict(sample_player_data)
    
    # Should return an empty list on error
    assert isinstance(results, list)
    assert len(results) == 0


def test_ensure_model_loaded_handles_errors():
    """Test that _ensure_model_loaded handles errors gracefully."""
    # Create a mock UFLPredictor that raises an error when load_model is called
    mock_predictor = MagicMock(spec=UFLPredictor)
    mock_predictor._models = {}
    mock_predictor.load_model.side_effect = ModelLoadError("Test error")
    
    # Initialize MLLayerPredictor with the mock
    # This should not raise an exception
    predictor = MLLayerPredictor(ufl_predictor=mock_predictor)
    
    # Verify load_model was called
    assert mock_predictor.load_model.called
    
    # The predictor should still be usable
    assert predictor.ufl_predictor == mock_predictor