import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from ml_layer.model_selection.hyperparameter_tuner import HyperparameterTuner

@pytest.fixture
def small_classification_dataset():
    """Generate a small synthetic classification dataset for testing"""
    X, y = make_classification(
        n_samples=50,  # Small dataset for fast testing
        n_features=5,
        n_classes=2,
        random_state=42
    )
    return X, y

class TestHyperparameterTuner:
    
    def test_initialization(self):
        """Test initialization of HyperparameterTuner"""
        tuner = HyperparameterTuner()
        assert tuner is not None
        assert hasattr(tuner, '_param_grids')
        assert len(tuner._param_grids) > 0
    
    def test_generate_param_grid_for_random_forest(self):
        """Test generating parameter grid for random forest"""
        tuner = HyperparameterTuner()
        param_grid = tuner.generate_param_grid('random_forest')
        
        assert param_grid is not None
        assert 'n_estimators' in param_grid
        assert 'max_depth' in param_grid
        assert isinstance(param_grid['n_estimators'], list)
        assert len(param_grid['n_estimators']) > 0
    
    def test_generate_param_grid_for_logistic_regression(self):
        """Test generating parameter grid for logistic regression"""
        tuner = HyperparameterTuner()
        param_grid = tuner.generate_param_grid('logistic_regression')
        
        assert param_grid is not None
        assert 'C' in param_grid
        assert 'penalty' in param_grid
        assert isinstance(param_grid['C'], list)
        assert len(param_grid['C']) > 0
    
    def test_generate_param_grid_for_unknown_model(self):
        """Test generating parameter grid for unknown model type"""
        tuner = HyperparameterTuner()
        param_grid = tuner.generate_param_grid('unknown_model')
        
        assert param_grid is not None
        assert isinstance(param_grid, dict)
        assert len(param_grid) == 0  # Should return empty dict for unknown models
    
    def test_get_search_strategy_grid(self):
        """Test getting grid search strategy"""
        tuner = HyperparameterTuner()
        strategy = tuner._get_search_strategy('grid')
        
        assert strategy is not None
        assert strategy == GridSearchCV
    
    def test_get_search_strategy_random(self):
        """Test getting random search strategy"""
        tuner = HyperparameterTuner()
        strategy = tuner._get_search_strategy('random')
        
        assert strategy is not None
        assert strategy == RandomizedSearchCV
    
    def test_get_search_strategy_invalid(self):
        """Test getting invalid search strategy"""
        tuner = HyperparameterTuner()
        
        with pytest.raises(ValueError, match="Unsupported search strategy"):
            tuner._get_search_strategy('invalid_strategy')
    
    def test_tune_hyperparameters_random_forest(self, small_classification_dataset):
        """Test hyperparameter tuning for random forest"""
        X, y = small_classification_dataset
        
        tuner = HyperparameterTuner()
        model = RandomForestClassifier(random_state=42)
        
        # Use a very small param grid for testing
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [None, 5]
        }
        
        result = tuner.tune_hyperparameters(
            model, X, y,
            param_distributions=param_grid,
            task_type='classification',
            search_strategy='random',
            n_iter=2,  # Small number for fast testing
            cv=2  # Small number for fast testing
        )
        
        assert result is not None
        assert 'best_model' in result
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'cv_results' in result
        assert isinstance(result['best_model'], RandomForestClassifier)
    
    def test_tune_hyperparameters_logistic_regression(self, small_classification_dataset):
        """Test hyperparameter tuning for logistic regression"""
        X, y = small_classification_dataset
        
        tuner = HyperparameterTuner()
        model = LogisticRegression(random_state=42)
        
        # Use a very small param grid for testing
        param_grid = {
            'C': [0.1, 1.0],
            'penalty': ['l2', None]
        }
        
        result = tuner.tune_hyperparameters(
            model, X, y,
            param_distributions=param_grid,
            task_type='classification',
            search_strategy='grid',  # Test grid search
            cv=2  # Small number for fast testing
        )
        
        assert result is not None
        assert 'best_model' in result
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'cv_results' in result
        assert isinstance(result['best_model'], LogisticRegression)
    
    def test_tune_hyperparameters_with_custom_scoring(self, small_classification_dataset):
        """Test hyperparameter tuning with custom scoring metric"""
        X, y = small_classification_dataset
        
        tuner = HyperparameterTuner()
        model = RandomForestClassifier(random_state=42)
        
        # Use a very small param grid for testing
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [None, 5]
        }
        
        result = tuner.tune_hyperparameters(
            model, X, y,
            param_distributions=param_grid,
            task_type='classification',
            search_strategy='random',
            n_iter=2,  # Small number for fast testing
            cv=2,  # Small number for fast testing
            scoring='precision'  # Custom scoring metric
        )
        
        assert result is not None
        assert 'best_model' in result
        assert 'best_params' in result
        assert 'best_score' in result
    
    def test_tune_hyperparameters_error_handling(self, small_classification_dataset):
        """Test error handling in hyperparameter tuning"""
        X, y = small_classification_dataset
        
        tuner = HyperparameterTuner()
        
        # Create a model that will fail during fitting
        class FailingModel:
            def fit(self, X, y):
                raise ValueError("Model fitting failed")
        
        model = FailingModel()
        
        # Use a very small param grid for testing
        param_grid = {
            'param1': [1, 2],
            'param2': [3, 4]
        }
        
        with pytest.raises(Exception):
            tuner.tune_hyperparameters(
                model, X, y,
                param_distributions=param_grid,
                task_type='classification',
                search_strategy='random',
                n_iter=2,
                cv=2
            )