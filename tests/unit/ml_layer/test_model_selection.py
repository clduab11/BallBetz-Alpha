import os
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml_layer.model_selection.model_registry import ModelRegistry
from ml_layer.model_selection.model_evaluator import ModelEvaluator
from ml_layer.model_selection.hyperparameter_tuner import HyperparameterTuner
from ml_layer.model_selection.config import ModelSelectionConfig

@pytest.fixture
def classification_data():
    """Generate synthetic classification dataset"""
    X, y = make_classification(
        n_samples=200, 
        n_features=10, 
        n_classes=2, 
        random_state=42
    )
    return X, y

@pytest.fixture
def regression_data():
    """Generate synthetic regression dataset"""
    X, y = make_regression(
        n_samples=200, 
        n_features=10, 
        noise=0.1, 
        random_state=42
    )
    return X, y

class TestModelRegistry:
    def test_get_model(self):
        """Test retrieving models from the registry"""
        registry = ModelRegistry()
        
        # Test classification model retrieval
        clf_model = registry.get_model('classification')
        assert clf_model is not None
        assert hasattr(clf_model, 'fit')
        
        # Test regression model retrieval
        reg_model = registry.get_model('regression')
        assert reg_model is not None
        assert hasattr(reg_model, 'fit')
    
    def test_register_custom_model(self):
        """Test registering a custom model"""
        class CustomModel:
            def fit(self, X, y):
                pass
        
        registry = ModelRegistry()
        registry.register_model('custom_task', 'custom_model', CustomModel)
        
        retrieved_model = registry.get_model('custom_task', 'custom_model')
        assert retrieved_model == CustomModel
    
    def test_list_models(self):
        """Test listing available models"""
        registry = ModelRegistry()
        models = registry.list_models()
        
        assert 'classification' in models
        assert 'regression' in models
        assert len(models['classification']) > 0
        assert len(models['regression']) > 0

class TestModelEvaluator:
    def test_evaluate_classification_model(self, classification_data):
        """Test model evaluation for classification"""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(
            model, X_test, y_test, 
            task_type='classification',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        assert 'mean_accuracy' in results
        assert 'mean_precision' in results
        assert 'mean_recall' in results
    
    def test_evaluate_regression_model(self, regression_data):
        """Test model evaluation for regression"""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(
            model, X_test, y_test, 
            task_type='regression',
            metrics=['mse', 'mae', 'r2']
        )
        
        assert 'mean_mse' in results
        assert 'mean_mae' in results
        assert 'mean_r2' in results
    
    def test_compare_models(self, classification_data):
        """Test model comparison"""
        X, y = classification_data
        
        models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier()
        }
        
        for name, model in models.items():
            model.fit(X, y)
        
        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models(models, X, y)
        
        assert len(comparison) == len(models)
        assert all('mean_accuracy' in results for results in comparison.values())

class TestHyperparameterTuner:
    def test_generate_param_grid(self):
        """Test generating parameter grids"""
        tuner = HyperparameterTuner()
        
        rf_grid = tuner.generate_param_grid('random_forest')
        assert 'n_estimators' in rf_grid
        assert 'max_depth' in rf_grid
        
        lr_grid = tuner.generate_param_grid('logistic_regression')
        assert 'C' in lr_grid
        assert 'penalty' in lr_grid
    
    def test_tune_hyperparameters_classification(self, classification_data):
        """Test hyperparameter tuning for classification"""
        X, y = classification_data
        
        model = RandomForestClassifier()
        tuner = HyperparameterTuner()
        
        param_grid = tuner.generate_param_grid('random_forest')
        tuned_result = tuner.tune_hyperparameters(
            model, X, y, 
            param_distributions=param_grid,
            task_type='classification'
        )
        
        assert 'best_model' in tuned_result
        assert 'best_params' in tuned_result
        assert 'best_score' in tuned_result
    
    def test_invalid_search_strategy(self):
        """Test handling of invalid search strategy"""
        tuner = HyperparameterTuner()
        
        with pytest.raises(ValueError, match="Unsupported search strategy"):
            tuner._get_search_strategy('invalid_strategy')

class TestModelSelectionConfig:
    def test_config_validation(self):
        """Test configuration validation"""
        # Test default configuration
        config = ModelSelectionConfig.from_env()
        config.validate()
        
        # Test invalid configurations
        with pytest.raises(ValueError, match="Invalid cross-validation folds"):
            invalid_config = ModelSelectionConfig(cv_folds=1)
            invalid_config.validate()
        
        with pytest.raises(ValueError, match="Unsupported primary metric"):
            invalid_config = ModelSelectionConfig(primary_metric='invalid_metric')
            invalid_config.validate()