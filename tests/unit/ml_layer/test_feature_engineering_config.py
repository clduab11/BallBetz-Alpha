import os
import pytest
from unittest.mock import patch

from ml_layer.feature_engineering.config import FeatureEngineeringConfig
from ml_layer.feature_engineering.exceptions import FeatureEngineeringConfigError

class TestFeatureEngineeringConfig:
    
    def test_default_configuration(self):
        """
        Test default configuration values
        """
        config = FeatureEngineeringConfig()
        
        assert config.numeric_imputation_strategy == 'mean'
        assert config.categorical_imputation_strategy == 'most_frequent'
        assert config.scale_numeric_features is True
        assert config.max_features_to_select == 10
        assert config.verbose is False
    
    def test_environment_variable_configuration(self):
        """
        Test configuration from environment variables
        """
        # Set environment variables
        with patch.dict(os.environ, {
            'UFL_NUMERIC_IMPUTATION': 'median',
            'UFL_CATEGORICAL_IMPUTATION': 'constant',
            'UFL_SCALE_NUMERIC': 'false',
            'UFL_MAX_FEATURES': '5',
            'UFL_VERBOSE_PREPROCESSING': 'true'
        }):
            config = FeatureEngineeringConfig.from_env()
            
            assert config.numeric_imputation_strategy == 'median'
            assert config.categorical_imputation_strategy == 'constant'
            assert config.scale_numeric_features is False
            assert config.max_features_to_select == 5
            assert config.verbose is True
    
    def test_validation_valid_config(self):
        """
        Test validation with valid configuration
        """
        config = FeatureEngineeringConfig(
            numeric_imputation_strategy='mean',
            categorical_imputation_strategy='most_frequent',
            max_features_to_select=5
        )
        
        # This should not raise an exception
        config.validate()
    
    def test_validation_invalid_numeric_strategy(self):
        """
        Test validation with invalid numeric imputation strategy
        """
        config = FeatureEngineeringConfig(
            numeric_imputation_strategy='invalid_strategy'
        )
        
        with pytest.raises(FeatureEngineeringConfigError, match="Invalid numeric imputation strategy"):
            config.validate()
    
    def test_validation_invalid_categorical_strategy(self):
        """
        Test validation with invalid categorical imputation strategy
        """
        config = FeatureEngineeringConfig(
            categorical_imputation_strategy='invalid_strategy'
        )
        
        with pytest.raises(FeatureEngineeringConfigError, match="Invalid categorical imputation strategy"):
            config.validate()
    
    def test_validation_invalid_max_features(self):
        """
        Test validation with invalid max_features_to_select
        """
        config = FeatureEngineeringConfig(
            max_features_to_select=0
        )
        
        with pytest.raises(FeatureEngineeringConfigError, match="Invalid max_features_to_select"):
            config.validate()
        
        config = FeatureEngineeringConfig(
            max_features_to_select=-5
        )
        
        with pytest.raises(FeatureEngineeringConfigError, match="Invalid max_features_to_select"):
            config.validate()
    
    def test_from_env_calls_validate(self):
        """
        Test that from_env calls validate
        """
        with patch.object(FeatureEngineeringConfig, 'validate') as mock_validate:
            FeatureEngineeringConfig.from_env()
            mock_validate.assert_called_once()