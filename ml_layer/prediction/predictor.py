import os
import logging
import numpy as np
import joblib
import json
import pandas as pd
from typing import Union, Dict, Any, Optional, List

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, VotingRegressor

from .config import PredictionConfig
from .exceptions import (
    ModelLoadError, 
    InferenceFailed, 
    ConfidenceThresholdError
)

class UFLPredictor:
    """
    Comprehensive prediction interface for UFL machine learning models
    
    Provides flexible prediction capabilities with support for 
    different task types, confidence scoring, and ensemble methods.
    """
    
    def __init__(
        self, 
        config: Optional[PredictionConfig] = None
    ):
        """
        Initialize the predictor
        
        :param config: Prediction configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or PredictionConfig.from_env()
        
        # Feature scaler for preprocessing
        self.scaler = StandardScaler()
        
        # Model cache
        self._models: Dict[str, BaseEstimator] = {}
    
    def load_model(
        self, 
        model_name: str, 
        checkpoint_path: Optional[str] = None
    ) -> BaseEstimator:
        """
        Load a model from checkpoint
        
        :param model_name: Name to assign to the loaded model
        :param checkpoint_path: Path to model checkpoint
        :return: Loaded model
        """
        try:
            # Use default checkpoint if not specified
            if checkpoint_path is None:
                checkpoints = sorted(
                    [f for f in os.listdir(self.config.model_dir) if f.endswith('.joblib')],
                    reverse=True
                )
                if not checkpoints:
                    raise FileNotFoundError("No model checkpoints found")
                checkpoint_path = os.path.join(self.config.model_dir, checkpoints[0])
            
            # Load checkpoint
            checkpoint_data = joblib.load(checkpoint_path)
            model = checkpoint_data.get('model')
            
            if model is None:
                raise ModelLoadError("No model found in checkpoint")
            
            # Cache the model
            self._models[model_name] = model
            
            self.logger.info(f"Model '{model_name}' loaded from {checkpoint_path}")
            return model
        
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Model loading failed: {e}")
    
    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess input data
        
        :param X: Input feature matrix
        :return: Scaled feature matrix
        """
        try:
            return self.scaler.fit_transform(X)
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise InferenceFailed(f"Data preprocessing error: {e}")
    
    def predict(
        self, 
        X: np.ndarray, 
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate predictions
        
        :param X: Input feature matrix
        :param model_name: Specific model to use for prediction
        :return: Prediction results with confidence
        """
        try:
            # Preprocess input data
            X_scaled = self._preprocess_data(X)
            
            # Use ensemble or single model
            if self.config.use_ensemble and len(self._models) > 1:
                predictions = self._ensemble_predict(X_scaled)
            else:
                # Use specified or first available model
                if model_name is None:
                    model_name = list(self._models.keys())[0]
                
                model = self._models.get(model_name)
                if model is None:
                    raise ModelLoadError(f"Model '{model_name}' not found")
                
                predictions = model.predict(X_scaled)
                probabilities = (
                    model.predict_proba(X_scaled) 
                    if hasattr(model, 'predict_proba') 
                    else None
                )
            
            # Calculate confidence
            confidence = self._calculate_confidence(predictions, probabilities)
            
            # Apply confidence threshold
            filtered_predictions = self._apply_confidence_threshold(
                predictions, confidence
            )
            
            # Log predictions if enabled
            if self.config.log_predictions:
                self._log_predictions(predictions, confidence)
            
            # Format output
            return self._format_output(
                predictions=filtered_predictions, 
                confidence=confidence
            )
        
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise InferenceFailed(f"Prediction error: {e}")
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using ensemble methods
        
        :param X: Scaled input feature matrix
        :return: Ensemble predictions
        """
        try:
            # Limit ensemble to configured max models
            ensemble_models = list(self._models.values())[:self.config.max_ensemble_models]
            
            if self.config.task_type == 'classification':
                ensemble = VotingClassifier(
                    estimators=[(f'model_{i}', model) for i, model in enumerate(ensemble_models)],
                    voting='soft'
                )
            elif self.config.task_type == 'regression':
                ensemble = VotingRegressor(
                    estimators=[(f'model_{i}', model) for i, model in enumerate(ensemble_models)]
                )
            else:
                raise ValueError(f"Ensemble not supported for task: {self.config.task_type}")
            
            return ensemble.predict(X)
        
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            raise InferenceFailed(f"Ensemble prediction error: {e}")
    
    def _calculate_confidence(
        self, 
        predictions: np.ndarray, 
        probabilities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate prediction confidence
        
        :param predictions: Model predictions
        :param probabilities: Class probabilities
        :return: Confidence scores
        """
        if probabilities is not None:
            # Use max probability as confidence for classification
            if self.config.task_type == 'classification':
                return np.max(probabilities, axis=1)
            
        # For regression or when probabilities are not available
        # Use a normalized confidence based on prediction distribution
        return np.abs(predictions - np.mean(predictions)) / np.std(predictions)
    
    def _apply_confidence_threshold(
        self, 
        predictions: np.ndarray, 
        confidence: np.ndarray
    ) -> np.ndarray:
        """
        Apply confidence threshold to predictions
        
        :param predictions: Original predictions
        :param confidence: Confidence scores
        :return: Filtered predictions
        """
        mask = confidence >= self.config.confidence_threshold
        
        if not np.any(mask):
            raise ConfidenceThresholdError(
                f"No predictions meet confidence threshold of {self.config.confidence_threshold}"
            )
        
        return predictions[mask]
    
    def _log_predictions(
        self, 
        predictions: np.ndarray, 
        confidence: np.ndarray
    ):
        """
        Log prediction details
        
        :param predictions: Model predictions
        :param confidence: Confidence scores
        """
        log_data = pd.DataFrame({
            'predictions': predictions,
            'confidence': confidence
        })
        
        log_path = os.path.join(self.config.model_dir, 'prediction_log.csv')
        log_data.to_csv(log_path, mode='a', header=not os.path.exists(log_path))
    
    def _format_output(
        self, 
        predictions: np.ndarray, 
        confidence: np.ndarray
    ) -> Dict[str, Any]:
        """
        Format prediction output
        
        :param predictions: Filtered predictions
        :param confidence: Confidence scores
        :return: Formatted prediction results
        """
        output = {
            'predictions': predictions.tolist(),
            'confidence': confidence.tolist(),
            'task_type': self.config.task_type
        }
        
        # Convert to specified output format
        if self.config.output_format == 'json':
            return json.dumps(output)
        elif self.config.output_format == 'csv':
            return pd.DataFrame(output).to_csv(index=False)
        
        return output