import pytest
import os
from unittest.mock import Mock, patch

from api_ai_layer.config import APIAILayerConfig
from api_ai_layer.transformer_interface import TransformerModelInterface
from api_ai_layer.providers.base_provider import BaseTransformerProvider
from api_ai_layer.exceptions import ModelLoadError, PredictionError

class MockProvider(BaseTransformerProvider):
    """
    Mock provider for testing transformer interface.
    """
    def __init__(self, config):
        super().__init__(config)
        self.model_loaded = False
        self.prediction_count = 0

    def _validate_config(self):
        pass

    def load_model(self, model_name=None):
        self.model_loaded = True
        return {
            'model_name': model_name or 'mock_model',
            'max_tokens': 2048,
            'temperature': 0.7
        }

    def predict(self, inputs, model, max_tokens=2048, temperature=0.7):
        self.prediction_count += 1
        return [
            {
                'input': input_text,
                'prediction': f'Prediction for {input_text}',
                'confidence': 0.8
            } for input_text in inputs
        ]

    def calculate_confidence(self, predictions):
        return 0.8

@pytest.fixture
def mock_config():
    """
    Create a mock configuration for testing.
    """
    return {
        'transformer_model': {
            'default_model': 'mock_model',
            'max_tokens': 2048,
            'temperature': 0.7,
            'batch_size': 16
        },
        'providers': {
            'mock_provider': {
                'api_key': 'test_key'
            }
        },
        'cache': {
            'enabled': True,
            'max_size': 1000,
            'ttl': 3600
        }
    }

def test_transformer_interface_initialization(mock_config):
    """
    Test initialization of TransformerModelInterface.
    """
    interface = TransformerModelInterface(config=mock_config)
    
    assert interface.config == mock_config
    assert len(interface.providers) == 0
    assert isinstance(interface._cache, dict)

def test_register_provider(mock_config):
    """
    Test registering a provider with the interface.
    """
    interface = TransformerModelInterface(config=mock_config)
    mock_provider = MockProvider(mock_config)
    
    interface.register_provider(mock_provider)
    
    assert len(interface.providers) == 1
    assert interface.providers[0] == mock_provider

def test_load_model(mock_config):
    """
    Test loading a model through the interface.
    """
    interface = TransformerModelInterface(config=mock_config)
    mock_provider = MockProvider(mock_config)
    interface.register_provider(mock_provider)
    
    model = interface.load_model('test_model')
    
    assert mock_provider.model_loaded
    assert model['model_name'] == 'test_model'
    assert model['max_tokens'] == 2048
    assert model['temperature'] == 0.7

def test_load_model_no_providers(mock_config):
    """
    Test loading a model when no providers are registered.
    """
    interface = TransformerModelInterface(config=mock_config)
    
    with pytest.raises(ModelLoadError):
        interface.load_model('test_model')

def test_predict_batch(mock_config):
    """
    Test batch prediction through the interface.
    """
    interface = TransformerModelInterface(config=mock_config)
    mock_provider = MockProvider(mock_config)
    interface.register_provider(mock_provider)
    
    inputs = ['test input 1', 'test input 2']
    predictions = interface.predict_batch(inputs)
    
    assert len(predictions) == 2
    assert mock_provider.prediction_count == 1
    assert all('prediction' in pred for pred in predictions)
    assert all('confidence' in pred for pred in predictions)

def test_predict_batch_no_providers(mock_config):
    """
    Test batch prediction when no providers are registered.
    """
    interface = TransformerModelInterface(config=mock_config)
    
    with pytest.raises(PredictionError):
        interface.predict_batch(['test input'])

def test_caching(mock_config):
    """
    Test caching mechanism in the transformer interface.
    """
    interface = TransformerModelInterface(config=mock_config)
    mock_provider = MockProvider(mock_config)
    interface.register_provider(mock_provider)
    
    inputs = ['test input 1', 'test input 2']
    interface.predict_batch(inputs)
    
    # Check if cache is populated
    assert len(interface._cache) > 0

def test_confidence_calculation(mock_config):
    """
    Test confidence calculation across providers.
    """
    interface = TransformerModelInterface(config=mock_config)
    mock_provider = MockProvider(mock_config)
    interface.register_provider(mock_provider)
    
    inputs = ['test input 1', 'test input 2']
    predictions = interface.predict_batch(inputs)
    
    confidence = mock_provider.calculate_confidence(predictions)
    assert confidence == 0.8