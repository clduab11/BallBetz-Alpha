import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score
)
from .config import ModelSelectionConfig

class ModelEvaluator:
    """
    Comprehensive model evaluation utility
    
    Provides advanced model evaluation capabilities with 
    cross-validation and multiple performance metrics.
    """
    
    def __init__(self, config: Optional[ModelSelectionConfig] = None):
        """
        Initialize the model evaluator
        
        :param config: Model selection configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or ModelSelectionConfig.from_env()
        
        self._metrics_map = {
            'classification': {
                'accuracy': accuracy_score,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score,
                'roc_auc': roc_auc_score,
                'average_precision': average_precision_score
            },
            'regression': {
                'mse': mean_squared_error,
                'mae': mean_absolute_error,
                'r2': r2_score
            }
        }
    
    def evaluate_model(
        self, 
        model: BaseEstimator, 
        X: np.ndarray, 
        y: np.ndarray, 
        task_type: str = 'classification',
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a model using cross-validation
        
        :param model: Scikit-learn compatible model
        :param X: Feature matrix
        :param y: Target variable
        :param task_type: Type of prediction task
        :param metrics: Optional list of metrics to calculate
        :return: Dictionary of performance metrics
        """
        try:
            # Validate task type
            if task_type not in self._metrics_map:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            # Use default metrics if not specified
            if metrics is None:
                metrics = list(self._metrics_map[task_type].keys())
            
            # Validate metrics
            invalid_metrics = set(metrics) - set(self._metrics_map[task_type].keys())
            if invalid_metrics:
                raise ValueError(f"Invalid metrics for {task_type}: {invalid_metrics}")
            
            # Prepare cross-validation strategy
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds, 
                shuffle=True, 
                random_state=42
            ) if task_type == 'classification' else self.config.cv_folds
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, 
                X, 
                y, 
                cv=cv, 
                scoring={metric: self._metrics_map[task_type][metric] for metric in metrics}
            )
            
            # Aggregate results
            results = {
                f'mean_{metric}': np.mean(cv_results[f'test_{metric}'])
                for metric in metrics
            }
            
            self.logger.info(f"Model evaluation results: {results}")
            return results
        
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise
    
    def compare_models(
        self, 
        models: Dict[str, BaseEstimator], 
        X: np.ndarray, 
        y: np.ndarray, 
        task_type: str = 'classification'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models
        
        :param models: Dictionary of models to compare
        :param X: Feature matrix
        :param y: Target variable
        :param task_type: Type of prediction task
        :return: Comparison of model performances
        """
        comparison_results = {}
        
        for name, model in models.items():
            try:
                comparison_results[name] = self.evaluate_model(
                    model, X, y, task_type
                )
            except Exception as e:
                self.logger.warning(f"Could not evaluate model {name}: {e}")
        
        # Sort models by primary metric
        sorted_models = sorted(
            comparison_results.items(), 
            key=lambda x: x[1].get(f'mean_{self.config.primary_metric}', 0), 
            reverse=True
        )
        
        self.logger.info("Model comparison complete")
        return dict(sorted_models)