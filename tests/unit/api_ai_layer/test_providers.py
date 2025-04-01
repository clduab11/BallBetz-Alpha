import pytest
import os
from unittest.mock import Mock, patch

from api_ai_layer.config import APIAILayerConfig
from api_ai_layer.providers.openai_client import OpenAIProvider
from api_ai_layer.providers.local_model_client import LocalTransformerProvider
from api_ai_layer.exceptions import ModelLoadError, PredictionError, ProviderError

@pytest.fixture
def mock_openai_config():
    """
    Create a mock configuration for OpenAI provider testing.
    """
    return {
        'providers': {
            'openai': {
                'api_key': 'test_key',
                'base_url': 'https://api.openai.com/v1',
                'timeout': 30,
                'max_retries': 3
            }
        },
        'transformer_model': {
            'default_model': 'gpt-3.5-turbo',
            'max_tokens': 2048,
            'temperature': 0.7,
            'batch_size': 16
        },
        'rate_limit': {
            'max_requests_per_minute': 60,
            'max_tokens_per_minute': 90000
        },
        'cache': {
            'enabled': True,
            'max_size': 1000,
            'ttl': 3600
        }
    }

@pytest.fixture
def mock_local_model_config():
    """
    Create a mock configuration for local transformer provider testing.
    """
    return {
        'providers': {
            'local_model': {
                'model_path': './models/test_model',
                'device': 'cpu'
            }
        },
        'transformer_model': {
            'default_model': 'local_transformer',
            'max_tokens': 2048,
            'temperature': 0.7,
            'batch_size': 16
        }
    }

class TestOpenAIProvider:
    """
    Test suite for OpenAI Provider.
    """

    def test_initialization(self, mock_openai_config):
        """
        Test OpenAI provider initialization.
        """
        provider = OpenAIProvider(mock_openai_config)
        
        assert provider.config == mock_openai_config
        assert provider._timeout == 30
        assert provider._max_retries == 3

    def test_validate_config_valid(self, mock_openai_config):
        """
        Test configuration validation with valid config.
        """
        provider = OpenAIProvider(mock_openai_config)
        provider._validate_config()  # Should not raise an exception

    def test_validate_config_invalid_api_key(self, mock_openai_config):
        """
        Test configuration validation with missing API key.
        """
        invalid_config = mock_openai_config.copy()
        invalid_config['providers']['openai']['api_key'] = ''
        
        with pytest.raises(ProviderError, match="OpenAI API key is required"):
            provider = OpenAIProvider(invalid_config)
            provider._validate_config()

    @patch('openai.ChatCompletion.create')
    def test_predict(self, mock_chat_completion, mock_openai_config):
        """
        Test prediction generation with mocked OpenAI API.
        """
        # Mock API response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Test prediction"))
        ]
        mock_response.usage = Mock(total_tokens=50)
        mock_chat_completion.return_value = mock_response

        provider = OpenAIProvider(mock_openai_config)
        model = provider.load_model()
        
        inputs = ["Test input 1", "Test input 2"]
        predictions = provider.predict(inputs, model)
        
        assert len(predictions) == 2
        assert all('prediction' in pred for pred in predictions)
        assert all('confidence' in pred for pred in predictions)
        mock_chat_completion.assert_called()

    def test_rate_limit_tracking(self, mock_openai_config):
        """
        Test rate limit tracking mechanism.
        """
        provider = OpenAIProvider(mock_openai_config)
        
        # Simulate multiple requests
        for _ in range(60):
            provider._request_timestamps.append(provider._get_current_timestamp())
        
        # 61st request should trigger rate limit
        with pytest.raises(RateLimitError):
            provider._check_rate_limits()

class TestLocalTransformerProvider:
    """
    Test suite for Local Transformer Provider.
    """

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.pipeline')
    def test_load_model(
        self, 
        mock_pipeline, 
        mock_model_from_pretrained, 
        mock_tokenizer_from_pretrained, 
        mock_local_model_config
    ):
        """
        Test loading a local transformer model.
        """
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_generator = Mock()
        
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model
        mock_pipeline.return_value = mock_generator

        provider = LocalTransformerProvider(mock_local_model_config)
        model_config = provider.load_model()
        
        assert model_config['model_path'] == './models/test_model'
        assert model_config['device'] == 'cpu'
        mock_tokenizer_from_pretrained.assert_called_once()
        mock_model_from_pretrained.assert_called_once()
        mock_pipeline.assert_called_once()

    def test_load_model_invalid_path(self, mock_local_model_config):
        """
        Test loading model with invalid path.
        """
        invalid_config = mock_local_model_config.copy()
        invalid_config['providers']['local_model']['model_path'] = ''
        
        with pytest.raises(ModelLoadError):
            provider = LocalTransformerProvider(invalid_config)
            provider.load_model()

    @patch('transformers.pipeline')
    def test_predict(
        self, 
        mock_pipeline, 
        mock_local_model_config
    ):
        """
        Test prediction generation with local transformer model.
        """
        # Mock generation output
        mock_generation_output = [
            {'generated_text': 'Test prediction 1'},
            {'generated_text': 'Test prediction 2'}
        ]
        mock_generator = Mock()
        mock_generator.return_value = mock_generation_output
        mock_pipeline.return_value = mock_generator

        provider = LocalTransformerProvider(mock_local_model_config)
        
        # Mock model loading
        provider._generator = mock_generator
        provider._model_path = './models/test_model'
        
        inputs = ["Test input 1", "Test input 2"]
        predictions = provider.predict(inputs, {})
        
        assert len(predictions) == 2
        assert all('prediction' in pred for pred in predictions)
        assert all('confidence' in pred for pred in predictions)
        mock_generator.assert_called_once()