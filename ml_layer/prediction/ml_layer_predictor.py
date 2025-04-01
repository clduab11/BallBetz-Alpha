"""
ML Layer Predictor Module.

This module provides a wrapper around the UFLPredictor to make it compatible
with the Triple-Layer Prediction Engine orchestrator.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from .predictor import UFLPredictor
from .config import PredictionConfig
from .exceptions import ModelLoadError, InferenceFailed

class MLLayerPredictor:
    """
    ML Layer Predictor for the Triple-Layer Prediction Engine.
    
    Wraps the UFLPredictor to provide a standardized interface for the
    prediction orchestrator.
    """
    
    def __init__(
        self,
        ufl_predictor: Optional[UFLPredictor] = None,
        config: Optional[PredictionConfig] = None
    ):
        """
        Initialize the ML Layer Predictor.
        
        Args:
            ufl_predictor: Optional UFLPredictor instance
            config: Optional prediction configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or PredictionConfig.from_env()
        self.ufl_predictor = ufl_predictor or UFLPredictor(self.config)
        
        # Load default model if not already loaded
        self._ensure_model_loaded()
    
    def _ensure_model_loaded(self) -> None:
        """
        Ensure at least one model is loaded.
        """
        try:
            if not hasattr(self.ufl_predictor, '_models') or not self.ufl_predictor._models:
                self.logger.info("Loading default model")
                self.ufl_predictor.load_model("default")
        except ModelLoadError as e:
            self.logger.warning(f"Could not load default model: {e}")
    
    def _convert_input(self, input_data: List[str]) -> pd.DataFrame:
        """
        Convert input data to the format expected by UFLPredictor.
        
        Args:
            input_data: List of input strings (serialized player data)
            
        Returns:
            DataFrame with player data
        """
        try:
            # Convert string representations to dictionaries
            import ast
            player_dicts = []
            for player_str in input_data:
                try:
                    # Try to safely evaluate the string as a Python literal
                    player_dict = ast.literal_eval(player_str)
                    player_dicts.append(player_dict)
                except (SyntaxError, ValueError) as e:
                    self.logger.warning(f"Could not parse player data: {e}")
                    # Add a minimal placeholder to maintain index alignment
                    player_dicts.append({"name": "unknown", "position": "unknown"})
            
            # Convert to DataFrame
            return pd.DataFrame(player_dicts)
        except Exception as e:
            self.logger.error(f"Error converting input data: {e}")
            raise ValueError(f"Invalid input format: {e}")
    
    def predict(self, inputs: List[str]) -> List[Dict[str, Any]]:
        """
        Generate predictions for the given inputs.
        
        Args:
            inputs: List of input strings (serialized player data)
            
        Returns:
            List of prediction dictionaries
        """
        try:
            # Convert inputs to DataFrame
            player_data = self._convert_input(inputs)
            if player_data.empty:
                return []
            
            # Extract features for prediction
            import numpy as np
            
            # Select numerical features for prediction
            numerical_cols = player_data.select_dtypes(include=[np.number]).columns.tolist()
            if not numerical_cols:
                self.logger.warning("No numerical features found for prediction")
                # Create dummy features if none exist
                player_data['dummy_feature'] = 1.0
                numerical_cols = ['dummy_feature']
            
            # Prepare feature matrix
            X = player_data[numerical_cols].fillna(0).values
            
            # Generate predictions
            prediction_results = self.ufl_predictor.predict(X)
            
            # Format results
            formatted_results = []
            predictions = prediction_results.get('predictions', [])
            confidences = prediction_results.get('confidence', [])
            
            for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                player_name = player_data.iloc[i].get('name', f"Player_{i}") if i < len(player_data) else f"Player_{i}"
                
                result = {
                    'input': inputs[i] if i < len(inputs) else "",
                    'value': float(pred),
                    'confidence': float(conf),
                    'player_name': player_name
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except InferenceFailed as e:
            self.logger.error(f"Prediction failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error in ML Layer prediction: {e}")
            return []