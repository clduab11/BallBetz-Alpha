import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from ml_layer.feature_engineering.preprocessor import UFLFeaturePreprocessor
from ml_layer.feature_engineering.exceptions import DataPreprocessingError, FeatureSelectionError

@pytest.fixture
def sample_data():
    """
    Create sample data for testing the preprocessor
    """
    return pd.DataFrame({
        'numeric_col1': [10, 20, 30, np.nan, 50],
        'numeric_col2': [1.1, 2.2, 3.3, 4.4, 5.5],
        'categorical_col1': ['A', 'B', 'A', 'C', 'B'],
        'categorical_col2': ['X', 'Y', 'Z', 'X', np.nan]
    })

@pytest.fixture
def mock_config():
    """
    Create a mock configuration for testing
    """
    class MockConfig:
        numeric_imputation_strategy = 'mean'
        categorical_imputation_strategy = 'most_frequent'
        scale_numeric_features = True
        max_features_to_select = 3
        verbose = True
        
        def validate(self):
            pass
        
        @classmethod
        def from_env(cls):
            return cls()
    
    return MockConfig()

class TestUFLFeaturePreprocessor:
    
    def test_initialization(self, mock_config):
        """
        Test initialization of UFLFeaturePreprocessor
        """
        preprocessor = UFLFeaturePreprocessor(config=mock_config)
        assert preprocessor is not None
        assert preprocessor.config is not None
        assert preprocessor.config.numeric_imputation_strategy == 'mean'
        assert preprocessor.config.categorical_imputation_strategy == 'most_frequent'
        assert preprocessor.config.scale_numeric_features is True
    
    def test_identify_column_types(self, sample_data, mock_config):
        """
        Test identification of numeric and categorical columns
        """
        preprocessor = UFLFeaturePreprocessor(config=mock_config)
        numeric_cols, categorical_cols = preprocessor._identify_column_types(sample_data)
        
        assert set(numeric_cols) == {'numeric_col1', 'numeric_col2'}
        assert set(categorical_cols) == {'categorical_col1', 'categorical_col2'}
    
    def test_create_preprocessing_pipeline(self, sample_data, mock_config):
        """
        Test creation of preprocessing pipeline
        """
        preprocessor = UFLFeaturePreprocessor(config=mock_config)
        numeric_cols, categorical_cols = preprocessor._identify_column_types(sample_data)
        
        pipeline = preprocessor._create_preprocessing_pipeline(numeric_cols, categorical_cols)
        
        assert pipeline is not None
        assert hasattr(pipeline, 'transformers')
        assert len(pipeline.transformers) == 2  # One for numeric, one for categorical
    
    def test_preprocess_without_target(self, sample_data, mock_config):
        """
        Test preprocessing without a target column
        """
        preprocessor = UFLFeaturePreprocessor(config=mock_config)
        X_transformed, y = preprocessor.preprocess(sample_data)
        
        assert X_transformed is not None
        assert X_transformed.shape[0] == sample_data.shape[0]
        assert y is None
    
    def test_preprocess_with_target(self, sample_data, mock_config):
        """
        Test preprocessing with a target column
        """
        # Add a target column
        data_with_target = sample_data.copy()
        data_with_target['target'] = [0, 1, 0, 1, 0]
        
        preprocessor = UFLFeaturePreprocessor(config=mock_config)
        X_transformed, y = preprocessor.preprocess(data_with_target, target_column='target')
        
        assert X_transformed is not None
        assert X_transformed.shape[0] == data_with_target.shape[0]
        assert y is not None
        assert len(y) == data_with_target.shape[0]
        assert list(y) == [0, 1, 0, 1, 0]
    
    def test_feature_selection(self, sample_data, mock_config):
        """
        Test feature selection with target
        """
        # Add a target column
        data_with_target = sample_data.copy()
        data_with_target['target'] = [0, 1, 0, 1, 0]
        
        preprocessor = UFLFeaturePreprocessor(config=mock_config)
        X_transformed, y = preprocessor.preprocess(data_with_target, target_column='target')
        
        # Feature selection should limit the number of features
        assert X_transformed.shape[1] <= mock_config.max_features_to_select
    
    def test_log_preprocessing_details(self, sample_data, mock_config):
        """
        Test logging of preprocessing details
        """
        preprocessor = UFLFeaturePreprocessor(config=mock_config)
        X_transformed, _ = preprocessor.preprocess(sample_data)
        
        # This should not raise an exception
        preprocessor.log_preprocessing_details(X_transformed)
    
    def test_error_handling_invalid_data(self, mock_config):
        """
        Test error handling with invalid data
        """
        preprocessor = UFLFeaturePreprocessor(config=mock_config)
        
        with pytest.raises(DataPreprocessingError):
            preprocessor.preprocess(None)
    
    @patch('ml_layer.feature_engineering.preprocessor.SelectKBest')
    def test_feature_selection_error(self, mock_select_kbest, sample_data, mock_config):
        """
        Test error handling during feature selection
        """
        # Setup mock to raise an exception
        mock_select_kbest.side_effect = Exception("Feature selection failed")
        
        # Add a target column
        data_with_target = sample_data.copy()
        data_with_target['target'] = [0, 1, 0, 1, 0]
        
        preprocessor = UFLFeaturePreprocessor(config=mock_config)
        
        with pytest.raises(DataPreprocessingError):
            preprocessor.preprocess(data_with_target, target_column='target')