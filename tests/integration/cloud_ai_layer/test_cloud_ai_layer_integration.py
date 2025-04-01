"""Integration tests for the Cloud AI Layer.

This module tests the interaction between different components
of the Cloud AI Layer and its integration with other layers.
"""

import pytest
import numpy as np
from cloud_ai_layer import (
    PredictionCombiner,
    ExternalFactorsIntegrator,
    CrossLeaguePatternAnalyzer,
    create_prediction_request,
    PredictionSourceType
)
from ml_layer.prediction.predictor import MLPredictor
from api_ai_layer.orchestrator import AILayerOrchestrator

class TestCloudAILayerIntegration:
    """
    Integration test suite for the Cloud AI Layer
    comprehensive testing of cross-layer interactions
    and prediction generation.
    """

    @pytest.fixture
    def ml_predictor(self):
        """
        Fixture to create an ML Layer predictor.
        
        Returns:
            Configured MLPredictor instance
        """
        return MLPredictor()

    @pytest.fixture
    def ai_layer_orchestrator(self):
        """
        Fixture to create an AI Layer orchestrator.
        
        Returns:
            Configured AILayerOrchestrator instance
        """
        return AILayerOrchestrator()

    @pytest.fixture
    def cloud_ai_components(self):
        """
        Fixture to create Cloud AI Layer components.
        
        Returns:
            Dictionary of Cloud AI Layer components
        """
        return {
            'external_factors': ExternalFactorsIntegrator(),
            'pattern_analyzer': CrossLeaguePatternAnalyzer(),
            'prediction_combiner': PredictionCombiner()
        }

    def test_multi_layer_prediction_integration(
        self, 
        ml_predictor, 
        ai_layer_orchestrator, 
        cloud_ai_components
    ):
        """
        Test end-to-end prediction generation across layers.
        
        Validates:
        - Prediction generation from multiple sources
        - Cross-layer data integration
        - Weighted prediction combination
        """
        # Prepare test context
        test_context = {
            'game_id': 'test_game_001',
            'league': 'test_league',
            'season': '2025'
        }
        
        # Create prediction request
        prediction_request = create_prediction_request(
            context=test_context,
            source_type=PredictionSourceType.ML_MODEL
        )
        
        # Generate predictions from different layers
        ml_prediction = ml_predictor.predict(prediction_request)
        ai_layer_prediction = ai_layer_orchestrator.generate_prediction(prediction_request)
        
        # Integrate external factors
        external_factors = cloud_ai_components['external_factors'].integrate_external_factors()
        
        # Analyze cross-league patterns
        pattern_analysis = cloud_ai_components['pattern_analyzer'].analyze_cross_league_patterns(
            current_league_data={
                'league_id': test_context['league'],
                **test_context
            },
            historical_leagues=[
                {'league_id': 'historical_league_1'},
                {'league_id': 'historical_league_2'}
            ]
        )
        
        # Combine predictions
        combined_predictions = {
            'ml_layer': ml_prediction.prediction,
            'api_ai_layer': ai_layer_prediction.prediction,
            'external_factors': list(external_factors.values())[0] if external_factors else 0.5
        }
        
        final_prediction = cloud_ai_components['prediction_combiner'].combine_predictions(
            combined_predictions
        )
        
        # Assertions
        assert 'prediction' in final_prediction
        assert 'confidence' in final_prediction
        assert 0 <= final_prediction['prediction'] <= 1
        assert 0 <= final_prediction['confidence'] <= 1
        
        # Optional: More detailed assertions
        assert len(pattern_analysis['similar_patterns']) >= 0
        assert final_prediction['weights']  # Verify weights are computed

    def test_error_handling_and_fallback(
        self, 
        ml_predictor, 
        ai_layer_orchestrator, 
        cloud_ai_components
    ):
        """
        Test error handling and fallback mechanisms.
        
        Validates:
        - Graceful handling of prediction source failures
        - Fallback prediction generation
        """
        # Prepare test context with potential error scenarios
        test_context = {
            'game_id': 'error_test_game',
            'league': 'error_league',
            'simulate_error': True
        }
        
        prediction_request = create_prediction_request(
            context=test_context,
            source_type=PredictionSourceType.ML_MODEL
        )
        
        # Simulate potential errors in different layers
        try:
            ml_prediction = ml_predictor.predict(prediction_request)
            ai_layer_prediction = ai_layer_orchestrator.generate_prediction(prediction_request)
            
            # Combine predictions with potential error handling
            combined_predictions = {
                'ml_layer': ml_prediction.prediction if ml_prediction else 0.5,
                'api_ai_layer': ai_layer_prediction.prediction if ai_layer_prediction else 0.5,
                'external_factors': 0.5  # Default fallback
            }
            
            final_prediction = cloud_ai_components['prediction_combiner'].combine_predictions(
                combined_predictions
            )
            
            # Assertions for fallback scenario
            assert 'prediction' in final_prediction
            assert 'confidence' in final_prediction
            assert 0 <= final_prediction['prediction'] <= 1
            assert 0 <= final_prediction['confidence'] <= 1
        
        except Exception as e:
            pytest.fail(f"Unexpected error in multi-layer prediction: {e}")

    def test_prediction_explainability(
        self, 
        ml_predictor, 
        ai_layer_orchestrator, 
        cloud_ai_components
    ):
        """
        Test prediction explainability features.
        
        Validates:
        - Generation of prediction explanations
        - Detailed contribution breakdown
        """
        # Prepare test context
        test_context = {
            'game_id': 'explain_game_001',
            'league': 'explain_league',
            'season': '2025'
        }
        
        prediction_request = create_prediction_request(
            context=test_context,
            source_type=PredictionSourceType.ML_MODEL
        )
        
        # Generate predictions
        ml_prediction = ml_predictor.predict(prediction_request)
        ai_layer_prediction = ai_layer_orchestrator.generate_prediction(prediction_request)
        external_factors = cloud_ai_components['external_factors'].integrate_external_factors()
        
        # Combine predictions
        combined_predictions = {
            'ml_layer': ml_prediction.prediction,
            'api_ai_layer': ai_layer_prediction.prediction,
            'external_factors': list(external_factors.values())[0] if external_factors else 0.5
        }
        
        final_prediction = cloud_ai_components['prediction_combiner'].combine_predictions(
            combined_predictions
        )
        
        # Check explanation details
        explanation = final_prediction.get('explanation', {})
        assert 'source_contributions' in explanation
        assert 'weight_rationale' in explanation
        assert 'confidence_factors' in explanation
        
        # Verify source contributions
        source_contributions = explanation['source_contributions']
        assert all(source in source_contributions for source in combined_predictions.keys())