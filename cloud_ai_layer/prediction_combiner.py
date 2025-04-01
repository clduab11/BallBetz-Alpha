"""Weighted Prediction Combiner Module.

This module combines predictions from multiple sources with configurable weights
and provides a final prediction with confidence scoring.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
from .config import CloudAILayerConfig
from .exceptions import PredictionCombinerError
from .external_factors import ExternalFactorsIntegrator
from .pattern_analyzer import CrossLeaguePatternAnalyzer

class PredictionCombiner:
    """
    Combines predictions from multiple sources with intelligent weighting.
    
    Integrates ML layer, API/Local AI layer, and external factors to generate
    a final weighted prediction with comprehensive explanation.
    """

    def __init__(self, 
                 config: CloudAILayerConfig = None,
                 external_factors_integrator: Optional[ExternalFactorsIntegrator] = None,
                 pattern_analyzer: Optional[CrossLeaguePatternAnalyzer] = None):
        """
        Initialize the Prediction Combiner.
        
        Args:
            config: Configuration for prediction combination
            external_factors_integrator: Optional external factors integrator
            pattern_analyzer: Optional cross-league pattern analyzer
        """
        self.config = config or CloudAILayerConfig
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.LOGGING_LEVEL)
        
        # Initialize integrators if not provided
        self.external_factors_integrator = (
            external_factors_integrator or ExternalFactorsIntegrator(self.config)
        )
        self.pattern_analyzer = (
            pattern_analyzer or CrossLeaguePatternAnalyzer(self.config)
        )
        
        # Dynamic weights cache
        self._dynamic_weights: Dict[str, float] = {}

    def _validate_predictions(self, predictions: Dict[str, Any]) -> None:
        """
        Validate input predictions for required fields.
        
        Args:
            predictions: Dictionary of predictions from different sources
        
        Raises:
            PredictionCombinerError: If predictions are invalid
        """
        required_keys = ['ml_layer', 'api_ai_layer']
        for key in required_keys:
            if key not in predictions:
                raise PredictionCombinerError(f"Missing prediction from {key}")
            
            if not isinstance(predictions[key], (int, float, dict)):
                raise PredictionCombinerError(f"Invalid prediction format for {key}")

    def compute_dynamic_weights(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """
        Dynamically compute prediction source weights based on recent performance.
        
        Args:
            predictions: Dictionary of predictions from different sources
        
        Returns:
            Updated weights for prediction sources
        """
        try:
            # Start with default weights
            weights = self.config.DEFAULT_PREDICTION_WEIGHTS.copy()
            
            # Adjust weights based on prediction consistency and historical accuracy
            external_factor_impacts = self.external_factors_integrator.integrate_external_factors()
            
            # Modify weights based on external factor impacts
            for factor, impact in external_factor_impacts.items():
                if factor in weights:
                    weights[factor] *= (1 + impact)
            
            # Normalize weights to ensure they sum to 1
            total_weight = sum(weights.values())
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            # Cache dynamic weights
            self._dynamic_weights = normalized_weights
            
            return normalized_weights
        except Exception as e:
            self.logger.warning(f"Dynamic weight computation failed: {e}")
            return self.config.DEFAULT_PREDICTION_WEIGHTS

    def combine_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine predictions from multiple sources with intelligent weighting.
        
        Args:
            predictions: Dictionary of predictions from different sources
        
        Returns:
            Final combined prediction with confidence and explanation
        """
        try:
            # Validate input predictions
            self._validate_predictions(predictions)
            
            # Compute dynamic weights
            weights = self.compute_dynamic_weights(predictions)
            
            # Normalize and weight predictions
            weighted_predictions = {}
            for source, prediction in predictions.items():
                if source in weights:
                    # Handle different prediction formats
                    if isinstance(prediction, (int, float)):
                        weighted_predictions[source] = prediction * weights[source]
                    elif isinstance(prediction, dict):
                        # For dictionary predictions, use a specific key
                        weighted_predictions[source] = (
                            prediction.get('value', 0) * weights[source]
                        )
            
            # Compute final prediction
            final_prediction = sum(weighted_predictions.values())
            
            # Compute confidence score
            confidence_score = self._calculate_confidence(
                predictions, weights, final_prediction
            )
            
            # Generate prediction explanation if enabled
            explanation = self._generate_prediction_explanation(
                predictions, weights, final_prediction, confidence_score
            ) if self.config.ENABLE_PREDICTION_EXPLANATIONS else None
            
            return {
                'prediction': final_prediction,
                'confidence': confidence_score,
                'weights': weights,
                'explanation': explanation
            }
        except Exception as e:
            self.logger.error(f"Prediction combination failed: {e}")
            raise PredictionCombinerError("Failed to combine predictions") from e

    def _calculate_confidence(self, 
                               predictions: Dict[str, Any], 
                               weights: Dict[str, float], 
                               final_prediction: float) -> float:
        """
        Calculate confidence score for the final prediction.
        
        Args:
            predictions: Original predictions
            weights: Source weights
            final_prediction: Combined prediction
        
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Compute variance of predictions
            prediction_values = [
                p if isinstance(p, (int, float)) else p.get('value', 0) 
                for p in predictions.values()
            ]
            variance = np.var(prediction_values)
            
            # Lower variance indicates higher confidence
            base_confidence = 1 - min(variance, 1)
            
            # Adjust confidence based on weight distribution
            weight_entropy = -sum(w * np.log(w) for w in weights.values() if w > 0)
            weight_confidence_factor = 1 - (weight_entropy / np.log(len(weights)))
            
            # Final confidence is a combination of base and weight confidence
            return min(max(base_confidence * weight_confidence_factor, 0), 1)
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5

    def _generate_prediction_explanation(self, 
                                         predictions: Dict[str, Any], 
                                         weights: Dict[str, float], 
                                         final_prediction: float, 
                                         confidence_score: float) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation for the prediction.
        
        Args:
            predictions: Original predictions
            weights: Source weights
            final_prediction: Combined prediction
            confidence_score: Prediction confidence
        
        Returns:
            Detailed prediction explanation
        """
        try:
            explanation = {
                'source_contributions': {},
                'weight_rationale': {},
                'confidence_factors': {}
            }
            
            # Source contributions
            for source, prediction in predictions.items():
                if source in weights:
                    value = prediction if isinstance(prediction, (int, float)) else prediction.get('value', 0)
                    contribution = value * weights[source]
                    explanation['source_contributions'][source] = {
                        'raw_prediction': value,
                        'weight': weights[source],
                        'weighted_contribution': contribution
                    }
            
            # Weight rationale
            explanation['weight_rationale'] = {
                'base_weights': self.config.DEFAULT_PREDICTION_WEIGHTS,
                'dynamic_weights': self._dynamic_weights
            }
            
            # Confidence factors
            explanation['confidence_factors'] = {
                'final_prediction': final_prediction,
                'confidence_score': confidence_score
            }
            
            return explanation
        except Exception as e:
            self.logger.warning(f"Prediction explanation generation failed: {e}")
            return {}