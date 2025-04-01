"""Unit tests for Cloud AI Layer components."""

import pytest
import numpy as np
from cloud_ai_layer.config import CloudAILayerConfig
from cloud_ai_layer.external_factors import ExternalFactorsIntegrator
from cloud_ai_layer.pattern_analyzer import CrossLeaguePatternAnalyzer
from cloud_ai_layer.prediction_combiner import PredictionCombiner
from cloud_ai_layer.interfaces import (
    create_prediction_request, 
    create_prediction_result, 
    PredictionSourceType
)
from cloud_ai_layer.exceptions import (
    ExternalFactorIntegrationError,
    PatternAnalysisError,
    PredictionCombinerError
)

class TestCloudAILayer:
    """Comprehensive test suite for Cloud AI Layer components."""

    def test_config_initialization(self):
        """Test CloudAILayerConfig initialization and validation."""
        config = CloudAILayerConfig
        
        # Test default configuration values
        assert 'weather' in config.EXTERNAL_FACTORS_SOURCES
        assert 0 <= config.CROSS_LEAGUE_SIMILARITY_THRESHOLD <= 1
        
        # Test weight validation
        config.validate_config()  # Should not raise an exception
        
        # Test external factor sources retrieval
        sources = config.get_external_factor_sources()
        assert isinstance(sources, dict)
        assert len(sources) > 0

    def test_external_factors_integrator(self):
        """Test ExternalFactorsIntegrator functionality."""
        integrator = ExternalFactorsIntegrator()
        
        # Mock test data
        mock_weather_data = {
            'temperature': 25,
            'humidity': 60,
            'wind_speed': 15
        }
        
        # Test normalization
        normalized_weather = integrator._normalize_weather(mock_weather_data)
        assert len(normalized_weather) == 3
        assert all(0 <= val <= 1 for val in normalized_weather)
        
        # Test factor impact calculation
        impact = integrator.calculate_factor_impact(normalized_weather, 'weather')
        assert 0 <= impact <= 1

    def test_pattern_analyzer(self):
        """Test CrossLeaguePatternAnalyzer functionality."""
        analyzer = CrossLeaguePatternAnalyzer()
        
        # Mock league data
        league1 = {
            'win_rate': 0.7,
            'avg_score': 100,
            'home_advantage': 0.6,
            'team_strength': 0.8
        }
        league2 = {
            'win_rate': 0.6,
            'avg_score': 95,
            'home_advantage': 0.5,
            'team_strength': 0.7
        }
        
        # Test league similarity
        similarity = analyzer.calculate_league_similarity(league1, league2)
        assert 0 <= similarity <= 1
        
        # Test pattern caching
        analyzer.cache_league_pattern('test_league', league1)
        cached_pattern = analyzer.get_cached_pattern('test_league')
        assert cached_pattern == league1

    def test_prediction_combiner(self):
        """Test PredictionCombiner functionality."""
        combiner = PredictionCombiner()
        
        # Test prediction combination
        predictions = {
            'ml_layer': 0.7,
            'api_ai_layer': 0.6,
            'external_factors': 0.5
        }
        
        result = combiner.combine_predictions(predictions)
        
        # Validate result structure
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'weights' in result
        assert 0 <= result['confidence'] <= 1
        
        # Test dynamic weight computation
        weights = combiner.compute_dynamic_weights(predictions)
        assert len(weights) > 0
        assert all(0 <= w <= 1 for w in weights.values())

    def test_interfaces(self):
        """Test prediction request and result creation."""
        # Test prediction request creation
        request = create_prediction_request(
            context={'game_data': 'sample'},
            source_type=PredictionSourceType.ML_MODEL
        )
        assert request.context == {'game_data': 'sample'}
        assert request.source_type == PredictionSourceType.ML_MODEL
        
        # Test prediction result creation
        result = create_prediction_result(
            prediction=0.75,
            confidence=0.85,
            source_type=PredictionSourceType.API_AI
        )
        assert result.prediction == 0.75
        assert result.confidence == 0.85
        assert result.source_type == PredictionSourceType.API_AI

    def test_error_handling(self):
        """Test error handling in Cloud AI Layer components."""
        # External Factors Integration Error
        with pytest.raises(ExternalFactorIntegrationError):
            integrator = ExternalFactorsIntegrator()
            integrator.normalize_factor({}, 'invalid_type')
        
        # Pattern Analysis Error
        with pytest.raises(PatternAnalysisError):
            analyzer = CrossLeaguePatternAnalyzer()
            analyzer.find_historical_patterns(
                {}, 
                [{'invalid': 'data'}] * (CloudAILayerConfig.MAX_HISTORICAL_PATTERNS + 1)
            )
        
        # Prediction Combiner Error
        with pytest.raises(PredictionCombinerError):
            combiner = PredictionCombiner()
            combiner.combine_predictions({})  # Invalid predictions

def test_cloud_ai_layer_integration():
    """
    Integration test demonstrating the end-to-end flow 
    of the Cloud AI Layer components.
    """
    # External Factors Integration
    external_integrator = ExternalFactorsIntegrator()
    external_impacts = external_integrator.integrate_external_factors()
    
    # Pattern Analysis
    pattern_analyzer = CrossLeaguePatternAnalyzer()
    league_data = {
        'win_rate': 0.7,
        'avg_score': 100,
        'home_advantage': 0.6,
        'team_strength': 0.8
    }
    historical_leagues = [
        {'win_rate': 0.6, 'avg_score': 95},
        {'win_rate': 0.8, 'avg_score': 105}
    ]
    pattern_analysis = pattern_analyzer.analyze_cross_league_patterns(
        league_data, 
        historical_leagues
    )
    
    # Prediction Combination
    prediction_combiner = PredictionCombiner()
    predictions = {
        'ml_layer': 0.7,
        'api_ai_layer': 0.6,
        'external_factors': list(external_impacts.values())[0]
    }
    final_prediction = prediction_combiner.combine_predictions(predictions)
    
    # Assertions
    assert 'prediction' in final_prediction
    assert 'confidence' in final_prediction
    assert len(pattern_analysis['similar_patterns']) >= 0