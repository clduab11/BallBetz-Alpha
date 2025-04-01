import logging
from typing import Dict, Any, Optional, Union, List
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

class HyperparameterTuner:
    """
    Hyperparameter tuning utility for machine learning models
    
    Provides automated hyperparameter optimization for different
    model types using various search strategies.
    """
    
    def __init__(self):
        """
        Initialize the hyperparameter tuner
        """
        self.logger = logging.getLogger(__name__)
        
        # Default parameter grids for common models
        self._param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 200, 500]
            },
            'linear_regression': {
                'fit_intercept': [True, False],
                'normalize': [True, False],
                'copy_X': [True, False]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                'degree': [2, 3, 4]
            }
        }
    
    def generate_param_grid(self, model_type: str) -> Dict[str, List[Any]]:
        """
        Generate parameter grid for a specific model type
        
        :param model_type: Type of model
        :return: Parameter grid for hyperparameter tuning
        """
        if model_type in self._param_grids:
            return self._param_grids[model_type]
        
        self.logger.warning(f"No predefined parameter grid for {model_type}. Using empty grid.")
        return {}
    
    def _get_search_strategy(self, strategy: str):
        """
        Get the appropriate search strategy
        
        :param strategy: Search strategy name
        :return: Search strategy class
        """
        if strategy == 'grid':
            return GridSearchCV
        elif strategy == 'random':
            return RandomizedSearchCV
        else:
            raise ValueError(f"Unsupported search strategy: {strategy}")
    
    def tune_hyperparameters(
        self, 
        model: BaseEstimator, 
        X: np.ndarray, 
        y: np.ndarray,
        param_distributions: Dict[str, List[Any]],
        task_type: str = 'classification',
        search_strategy: str = 'random',
        n_iter: int = 10,
        cv: int = 5,
        scoring: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for a given model
        
        :param model: Base model to tune
        :param X: Feature matrix
        :param y: Target variable
        :param param_distributions: Parameter distributions for search
        :param task_type: Type of prediction task
        :param search_strategy: Strategy for hyperparameter search
        :param n_iter: Number of iterations for random search
        :param cv: Number of cross-validation folds
        :param scoring: Scoring metric
        :return: Dictionary with tuning results
        """
        try:
            # Get search strategy
            search_class = self._get_search_strategy(search_strategy)
            
            # Set default scoring based on task type
            if scoring is None:
                scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
            
            # Configure search
            if search_strategy == 'random':
                search = search_class(
                    model,
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                search = search_class(
                    model,
                    param_grid=param_distributions,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1
                )
            
            # Perform search
            search.fit(X, y)
            
            # Log results
            self.logger.info(f"Best parameters: {search.best_params_}")
            self.logger.info(f"Best score: {search.best_score_}")
            
            # Return results
            return {
                'best_model': search.best_estimator_,
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_
            }
        
        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {e}")
            raise