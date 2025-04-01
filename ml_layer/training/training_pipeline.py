import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from .config import TrainingConfig
from .model_checkpoint import ModelCheckpoint
from .exceptions import ModelTrainingError, DataPreparationError

class TrainingPipeline:
    """
    Comprehensive training pipeline for machine learning models
    
    Provides a flexible workflow for model training, evaluation,
    and checkpointing across different prediction tasks.
    """
    
    def __init__(
        self, 
        config: Optional[TrainingConfig] = None,
        checkpoint: Optional[ModelCheckpoint] = None
    ):
        """
        Initialize training pipeline
        
        :param config: Training configuration
        :param checkpoint: Model checkpoint utility
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or TrainingConfig.from_env()
        self.checkpoint = checkpoint or ModelCheckpoint(self.config)
    
    def _prepare_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        :param X: Feature matrix
        :param y: Target variable
        :return: Train and test splits with scaled features
        """
        try:
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=42, 
                stratify=y if self.config.task_type == 'classification' else None
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
        
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise DataPreparationError(f"Failed to prepare training data: {e}")
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate performance metrics based on task type
        
        :param y_true: True target values
        :param y_pred: Predicted values
        :return: Dictionary of performance metrics
        """
        metrics = {}
        
        if self.config.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
        elif self.config.task_type == 'regression':
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        
        return metrics
    
    def train(
        self, 
        model: BaseEstimator, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train a machine learning model
        
        :param model: Scikit-learn compatible model
        :param X: Feature matrix
        :param y: Target variable
        :return: Training results and metadata
        """
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self._prepare_data(X, y)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # Select primary metric for checkpointing
            primary_metric = (
                'accuracy' if self.config.task_type == 'classification' 
                else 'r2'
            )
            
            # Save checkpoint
            checkpoint_path = self.checkpoint.save(
                model, 
                score=metrics.get(primary_metric),
                metadata={
                    'task_type': self.config.task_type,
                    'metrics': metrics
                }
            )
            
            # Log training results
            self.logger.info(f"Training completed. Metrics: {metrics}")
            
            return {
                'model': model,
                'metrics': metrics,
                'checkpoint_path': checkpoint_path
            }
        
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise ModelTrainingError(f"Failed to train model: {e}")
    
    def predict(
        self, 
        model: BaseEstimator, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Generate predictions using a trained model
        
        :param model: Trained model
        :param X: Feature matrix
        :return: Predictions
        """
        try:
            # Scale input features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Generate predictions
            predictions = model.predict(X_scaled)
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise ModelTrainingError(f"Failed to generate predictions: {e}")