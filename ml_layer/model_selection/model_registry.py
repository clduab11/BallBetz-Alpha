import logging
from typing import Dict, Any, Type, Optional
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

class ModelRegistry:
    """
    A centralized registry for machine learning models
    
    Provides a flexible mechanism to register, retrieve, 
    and manage different types of predictive models.
    """
    
    def __init__(self):
        """
        Initialize the model registry
        """
        self.logger = logging.getLogger(__name__)
        self._models: Dict[str, Dict[str, Type[BaseEstimator]]] = {
            'classification': {
                'logistic_regression': LogisticRegression,
                'random_forest': RandomForestClassifier,
                'svm': SVC,
                'neural_network': MLPClassifier,
                'xgboost': XGBClassifier
            },
            'regression': {
                'linear_regression': LinearRegression,
                'random_forest': RandomForestRegressor,
                'svm': SVR,
                'neural_network': MLPRegressor,
                'xgboost': XGBRegressor
            }
        }
    
    def get_model(
        self, 
        task_type: str, 
        model_name: Optional[str] = None
    ) -> Type[BaseEstimator]:
        """
        Retrieve a model class based on task type and optional model name
        
        :param task_type: Type of prediction task ('classification' or 'regression')
        :param model_name: Specific model name (optional)
        :return: Model class
        :raises ValueError: If task type or model is not found
        """
        if task_type not in self._models:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # If no specific model is provided, return the first available model
        if model_name is None:
            model_name = list(self._models[task_type].keys())[0]
        
        if model_name not in self._models[task_type]:
            raise ValueError(f"Model {model_name} not found for task type {task_type}")
        
        return self._models[task_type][model_name]
    
    def register_model(
        self, 
        task_type: str, 
        model_name: str, 
        model_class: Type[BaseEstimator]
    ) -> None:
        """
        Register a custom model in the registry
        
        :param task_type: Type of prediction task
        :param model_name: Name of the model
        :param model_class: Model class to register
        """
        if task_type not in self._models:
            self._models[task_type] = {}
        
        self._models[task_type][model_name] = model_class
        self.logger.info(f"Registered model {model_name} for {task_type}")
    
    def list_models(self, task_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available models
        
        :param task_type: Optional task type to filter models
        :return: Dictionary of available models
        """
        if task_type:
            return {task_type: list(self._models.get(task_type, {}).keys())}
        
        return {
            task: list(models.keys())
            for task, models in self._models.items()
        }