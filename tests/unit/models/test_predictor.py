import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from enum import Enum
from models.predictor import PlayerPerformancePredictor
from models.model_interface import ModelInterface, ModelProvider
from models.config import ModelConfig

@pytest.fixture
def mock_model_interface():
    mock = MagicMock(spec=ModelInterface)
    mock.predict.return_value = pd.DataFrame({'predicted_points': [10, 15]})
    return mock

def test_train(mock_model_interface, mock_player_data):
    predictor = PlayerPerformancePredictor()
    predictor._model = mock_model_interface
    result = predictor.train(mock_player_data, cv_splits=2)  # Use fewer CV splits to avoid error with small dataset
    assert result == True

def test_predict(mock_model_interface, mock_player_data):
    predictor = PlayerPerformancePredictor()
    predictor._model = mock_model_interface
    predictions = predictor.predict(mock_player_data)
    assert isinstance(predictions, pd.DataFrame)
    assert 'predicted_points' in predictions.columns

def test_save_load_model(mock_model_interface, mock_player_data, tmp_path):
    predictor = PlayerPerformancePredictor()
    predictor._model = mock_model_interface
    predictor.train(mock_player_data, cv_splits=2)  # Use fewer CV splits to avoid error with small dataset
    
    model_path = tmp_path / "test_model.joblib"
    predictor.save_model(model_path)
    
    loaded_predictor = PlayerPerformancePredictor()
    loaded_predictor.load_model(model_path)
    
    assert loaded_predictor.sklearn_model is not None
    # Basic check, more detailed comparison might be needed depending on the model
    assert isinstance(loaded_predictor.model_interfaces[loaded_predictor.default_provider], ModelInterface)

# New tests for predict() with different providers and fallback
def test_predict_default_provider(mock_model_interface, mock_player_data):
    predictor = PlayerPerformancePredictor()
    predictor.model_interfaces[ModelConfig.DEFAULT_PROVIDER] = mock_model_interface
    predictions = predictor.predict(mock_player_data)
    assert isinstance(predictions, pd.DataFrame)
    mock_model_interface.predict.assert_called_once_with(mock_player_data)

def test_predict_specific_provider(mock_model_interface, mock_player_data):
    predictor = PlayerPerformancePredictor()
    predictor.model_interfaces[ModelProvider.SKLEARN] = mock_model_interface
    predictions = predictor.predict(mock_player_data, provider=ModelProvider.SKLEARN)
    assert isinstance(predictions, pd.DataFrame)
    mock_model_interface.predict.assert_called_once_with(mock_player_data)

def test_predict_fallback(mock_player_data):
    predictor = PlayerPerformancePredictor()
    # Mock no available primary provider
    predictor.model_interfaces = {}
    
    # Mock a fallback provider
    mock_fallback = MagicMock(spec=ModelInterface)
    mock_fallback.predict.return_value = pd.DataFrame({'predicted_points': [5, 8]})
    predictor.model_interfaces[ModelConfig.FALLBACK_ORDER[0]] = mock_fallback
    
    predictions = predictor.predict(mock_player_data)
    assert isinstance(predictions, pd.DataFrame)
    mock_fallback.predict.assert_called_once_with(mock_player_data)

def test_predict_no_provider(mock_player_data):
    predictor = PlayerPerformancePredictor()
    predictor.model_interfaces = {}  # No providers available
    predictions = predictor.predict(mock_player_data)
    assert predictions is None

def test_set_provider_valid(mock_model_interface):
    predictor = PlayerPerformancePredictor()
    predictor.model_interfaces[ModelProvider.SKLEARN] = mock_model_interface
    assert predictor.set_provider(ModelProvider.SKLEARN) == True
    assert predictor.default_provider == ModelProvider.SKLEARN

def test_set_provider_invalid(mock_model_interface):
    predictor = PlayerPerformancePredictor()
    # Use a provider that's guaranteed not to be initialized
    class MockProvider(Enum):
        INVALID = "invalid"

    assert predictor.set_provider(MockProvider.INVALID) == False

def test_get_available_providers(mock_model_interface):
    predictor = PlayerPerformancePredictor()
    predictor.model_interfaces[ModelProvider.SKLEARN] = mock_model_interface
    providers = predictor.get_available_providers()
    assert ModelProvider.SKLEARN in providers
    assert len(providers) == len(predictor.model_interfaces)