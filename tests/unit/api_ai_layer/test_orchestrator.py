import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from api_ai_layer.config import APIAILayerConfig
from api_ai_layer.orchestrator import PredictionOrchestrator
from api_ai_layer.transformer_interface import TransformerModelInterface
from ml_layer.prediction.predictor import MLLayerPredictor

@pytest.fixture
def mock_config():
    """
    Create a mock configuration for orchestrator testing.
    """
    return {
        'orchestration': {
            'confidence_threshold': 0.7,
            'fallback_strategy': 'local_model'
        },
        'transformer_model': {
            'default_model': 'test_model',
            'max_tokens': 2048,
            'temperature': 0.7
        }
    }

@pytest.fixture
def mock_transformer_interface(mock_config):
    """
    Create a mock transformer interface.
    """
    interface = Mock(spec=TransformerModelInterface)
    interface.config = mock_config
    interface.predict_batch.return_value = [
        {
            'input': 'test input 1',
            'prediction': 'Transformer prediction 1',
            'confidence': 0.8
        },
        {
            'input': 'test input 2', 
            'prediction': 'Transformer prediction 2',
            'confidence': 0.75
        }
    ]
    return interface

@pytest.fixture
def mock_ml_layer_predictor():
    """
    Create a mock ML Layer predictor.
    """
    predictor = Mock(spec=MLLayerPredictor)
    predictor.predict.return_value = [
        {
            'input': 'test input 1',
            'prediction': 'ML Layer prediction 1',
            'confidence': 0.7
        },
        {
            'input': 'test input 2',
            'prediction': 'ML Layer prediction 2', 
            'confidence': 0.65
        }
    ]
    return predictor

@pytest.mark.asyncio
async def test_predict_async(mock_config, mock_transformer_interface, mock_ml_layer_predictor):
    """
    Test asynchronous prediction method.
    """
    orchestrator = PredictionOrchestrator(
        transformer_interface=mock_transformer_interface,
        ml_layer_predictor=mock_ml_layer_predictor,
        config=mock_config
    )
    
    inputs = ['test input 1', 'test input 2']
    predictions = await orchestrator.predict_async(inputs)
    
    assert len(predictions) == 2
    
    # Check transformer and ML layer predictions are included
    for pred in predictions:
        assert 'transformer_prediction' in pred
        assert 'ml_layer_prediction' in pred
        assert 'confidence' in pred
        assert 'weighted_prediction' in pred

def test_predict_sync(mock_config, mock_transformer_interface, mock_ml_layer_predictor):
    """
    Test synchronous prediction method.
    """
    orchestrator = PredictionOrchestrator(
        transformer_interface=mock_transformer_interface,
        ml_layer_predictor=mock_ml_layer_predictor,
        config=mock_config
    )
    
    inputs = ['test input 1', 'test input 2']
    predictions = orchestrator.predict(inputs)
    
    assert len(predictions) == 2

def test_confidence_calculation(mock_config, mock_transformer_interface, mock_ml_layer_predictor):
    """
    Test combined confidence calculation.
    """
    orchestrator = PredictionOrchestrator(
        transformer_interface=mock_transformer_interface,
        ml_layer_predictor=mock_ml_layer_predictor,
        config=mock_config
    )
    
    inputs = ['test input 1', 'test input 2']
    predictions = orchestrator.predict(inputs)
    
    # Check confidence calculation
    for pred in predictions:
        assert 0 <= pred['confidence'] <= 1
        
        # Verify confidence is a weighted average
        transformer_conf = pred['transformer_prediction']['confidence']
        ml_layer_conf = pred['ml_layer_prediction']['confidence']
        expected_conf = (transformer_conf + ml_layer_conf) / 2
        assert abs(pred['confidence'] - expected_conf) < 0.01

def test_weighted_prediction(mock_config, mock_transformer_interface, mock_ml_layer_predictor):
    """
    Test weighted prediction combination.
    """
    orchestrator = PredictionOrchestrator(
        transformer_interface=mock_transformer_interface,
        ml_layer_predictor=mock_ml_layer_predictor,
        config=mock_config
    )
    
    inputs = ['test input 1', 'test input 2']
    predictions = orchestrator.predict(inputs)
    
    for pred in predictions:
        weighted_pred = pred['weighted_prediction']
        
        # Verify weighted prediction structure
        assert 'transformer_result' in weighted_pred
        assert 'ml_layer_result' in weighted_pred
        assert 'weights' in weighted_pred
        
        # Check weights
        weights = weighted_pred['weights']
        assert 'transformer' in weights
        assert 'ml_layer' in weights
        assert 0 <= weights['transformer'] <= 1
        assert 0 <= weights['ml_layer'] <= 1
        assert abs(weights['transformer'] + weights['ml_layer'] - 1) < 0.01

@pytest.mark.asyncio
async def test_predict_async_error_handling(mock_config, mock_transformer_interface, mock_ml_layer_predictor):
    """
    Test error handling in asynchronous prediction.
    """
    # Simulate transformer interface failure
    mock_transformer_interface.predict_batch.side_effect = Exception("Transformer prediction failed")
    
    # Simulate ML layer predictor failure
    mock_ml_layer_predictor.predict.side_effect = Exception("ML Layer prediction failed")
    
    orchestrator = PredictionOrchestrator(
        transformer_interface=mock_transformer_interface,
        ml_layer_predictor=mock_ml_layer_predictor,
        config=mock_config
    )
    
    inputs = ['test input 1', 'test input 2']
    
    with pytest.raises(PredictionError):
        await orchestrator.predict_async(inputs)