"""Security tests for the Triple-Layer Prediction Engine.

This module contains security tests for the Triple-Layer Prediction Engine,
focusing on input validation, API security, and data sanitization.
"""

import pytest
import json
import re
from unittest.mock import patch, MagicMock
import requests
from urllib.parse import urlparse

from cloud_ai_layer.external_factors import ExternalFactorsIntegrator
from cloud_ai_layer.config import CloudAILayerConfig
from cloud_ai_layer.exceptions import ValidationError
from api_ai_layer.orchestrator import PredictionOrchestrator
from ml_layer.prediction.ml_layer_predictor import MLLayerPredictor


class TestExternalFactorsInputValidation:
    """Test input validation for external factors integration."""

    def setup_method(self):
        """Set up test environment."""
        self.integrator = ExternalFactorsIntegrator()

    def test_factor_type_validation(self):
        """Test validation of factor type parameter."""
        # Test empty factor type
        with pytest.raises(ValidationError, match="Factor type cannot be empty"):
            self.integrator._validate_factor_type("")

        # Test non-string factor type
        with pytest.raises(ValidationError, match="Factor type must be a string"):
            self.integrator._validate_factor_type(123)

        # Test invalid factor type
        with pytest.raises(ValidationError, match="Invalid factor type"):
            self.integrator._validate_factor_type("invalid_type")

        # Test valid factor types
        for factor_type in CloudAILayerConfig.EXTERNAL_FACTORS_SOURCES.keys():
            # Should not raise an exception
            self.integrator._validate_factor_type(factor_type)

    def test_api_url_validation(self):
        """Test validation of API URLs."""
        # Test empty URL
        with pytest.raises(ValidationError, match="API URL cannot be empty"):
            self.integrator._validate_api_url("")

        # Test invalid URL format
        with pytest.raises(ValidationError, match="Invalid URL format"):
            self.integrator._validate_api_url("not-a-url")

        # Test valid URL format
        self.integrator._validate_api_url("https://api.example.com/data")

        # Test URL with allowed domain
        with patch.object(CloudAILayerConfig, 'ALLOWED_API_DOMAINS', ['api.example.com']):
            self.integrator._validate_api_url("https://api.example.com/data")

        # Test URL with disallowed domain
        with patch.object(CloudAILayerConfig, 'ALLOWED_API_DOMAINS', ['api.example.com']):
            with pytest.raises(ValidationError, match="Domain not in allowed list"):
                self.integrator._validate_api_url("https://malicious-site.com/data")

    def test_api_response_sanitization(self):
        """Test sanitization of API responses."""
        # Test valid JSON data
        valid_data = {"temperature": 25, "humidity": 60, "wind_speed": 10}
        sanitized = self.integrator._sanitize_api_response(valid_data)
        assert sanitized == valid_data

        # Test nested JSON data
        nested_data = {"weather": {"temperature": 25, "conditions": ["sunny", "clear"]}}
        sanitized = self.integrator._sanitize_api_response(nested_data)
        assert sanitized == nested_data

        # Test invalid data (not JSON serializable)
        class NonSerializable:
            pass

        with pytest.raises(ValidationError, match="Invalid API response format"):
            self.integrator._sanitize_api_response({"data": NonSerializable()})


class TestExternalFactorsAPIIntegration:
    """Test external API integration security."""

    def setup_method(self):
        """Set up test environment."""
        self.integrator = ExternalFactorsIntegrator()

    @patch('requests.get')
    def test_api_request_security(self, mock_get):
        """Test security of API requests."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"temperature": 25, "humidity": 60}
        mock_get.return_value = mock_response

        # Test with valid factor type
        with patch.object(self.integrator, '_validate_factor_type'):
            with patch.object(self.integrator, '_validate_api_url'):
                self.integrator.fetch_external_data('weather')

        # Verify request includes User-Agent header
        args, kwargs = mock_get.call_args
        assert 'headers' in kwargs
        assert 'User-Agent' in kwargs['headers']

        # Verify request includes timeout
        assert 'timeout' in kwargs

    @patch('requests.get')
    def test_api_error_handling(self, mock_get):
        """Test handling of API errors."""
        # Test connection error
        mock_get.side_effect = requests.ConnectionError("Connection failed")
        
        with patch.object(self.integrator, '_validate_factor_type'):
            with patch.object(self.integrator, '_validate_api_url'):
                with pytest.raises(Exception):
                    self.integrator.fetch_external_data('weather')

        # Test timeout error
        mock_get.side_effect = requests.Timeout("Request timed out")
        
        with patch.object(self.integrator, '_validate_factor_type'):
            with patch.object(self.integrator, '_validate_api_url'):
                with pytest.raises(Exception):
                    self.integrator.fetch_external_data('weather')

        # Test HTTP error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.side_effect = None
        mock_get.return_value = mock_response
        
        with patch.object(self.integrator, '_validate_factor_type'):
            with patch.object(self.integrator, '_validate_api_url'):
                with pytest.raises(Exception):
                    self.integrator.fetch_external_data('weather')


class TestTripleLayerIntegrationSecurity:
    """Test security of the Triple-Layer Prediction Engine integration."""

    def setup_method(self):
        """Set up test environment."""
        self.ml_predictor = MLLayerPredictor()
        self.orchestrator = PredictionOrchestrator(ml_layer_predictor=self.ml_predictor)

    def test_input_sanitization(self):
        """Test sanitization of inputs to the prediction engine."""
        # Test with valid input
        valid_input = [
            '{"name": "Player 1", "team": "Team A", "position": "QB", "games_played": 10}'
        ]
        
        with patch.object(self.ml_predictor, 'predict', return_value=[{'value': 20.5}]):
            # Should not raise an exception
            self.orchestrator.predict(valid_input)

        # Test with potentially malicious input
        malicious_input = [
            '{"name": "<script>alert(\'XSS\')</script>", "team": "Team A"}'
        ]
        
        with patch.object(self.ml_predictor, 'predict', return_value=[{'value': 20.5}]):
            # Should not raise an exception, but should sanitize the input
            result = self.orchestrator.predict(malicious_input)
            
            # Check that the result doesn't contain the script tag
            result_str = json.dumps(result)
            assert '<script>' not in result_str

    def test_output_sanitization(self):
        """Test sanitization of outputs from the prediction engine."""
        valid_input = ['{"name": "Player 1"}']
        
        # Mock ML predictor to return potentially dangerous output
        dangerous_output = [{'value': 20.5, 'metadata': '<script>alert("XSS")</script>'}]
        
        with patch.object(self.ml_predictor, 'predict', return_value=dangerous_output):
            result = self.orchestrator.predict(valid_input)
            
            # Check that the result doesn't contain the script tag
            result_str = json.dumps(result)
            assert '<script>' not in result_str


class TestConfigurationSecurity:
    """Test security of configuration settings."""

    def test_allowed_domains_configuration(self):
        """Test configuration of allowed API domains."""
        # Test with empty allowed domains
        with patch.object(CloudAILayerConfig, 'ALLOWED_API_DOMAINS', []):
            from cloud_ai_layer.exceptions import ConfigurationError
            
            with pytest.raises(ConfigurationError):
                CloudAILayerConfig.validate_config()

        # Test with valid allowed domains
        with patch.object(CloudAILayerConfig, 'ALLOWED_API_DOMAINS', ['api.example.com']):
            # Should not raise an exception
            try:
                CloudAILayerConfig.validate_config()
            except Exception as e:
                if isinstance(e, ConfigurationError) and "allowed API domains" in str(e):
                    pytest.fail(f"validate_config() raised {e} unexpectedly!")