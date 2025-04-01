"""
Integration test for the Triple-Layer Prediction Engine.

This test verifies that the three layers of the prediction engine work together
correctly and that the app can use the engine to generate predictions.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import json

from app import app, prediction_orchestrator, prediction_combiner
from scrapers.ufl_scraper import UFLScraper
from ml_layer.prediction.ml_layer_predictor import MLLayerPredictor
from api_ai_layer.orchestrator import PredictionOrchestrator
from cloud_ai_layer.prediction_combiner import PredictionCombiner


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_player_data():
    """Create mock player data for testing."""
    return [
        {
            'name': 'Player 1',
            'team': 'Team A',
            'position': 'QB',
            'games_played': 10,
            'passing_yards': 2500,
            'passing_touchdowns': 20,
            'interceptions': 5,
            'rushing_yards': 200,
            'rushing_touchdowns': 2,
            'salary': 8000
        },
        {
            'name': 'Player 2',
            'team': 'Team B',
            'position': 'RB',
            'games_played': 10,
            'rushing_yards': 1000,
            'rushing_touchdowns': 8,
            'receiving_yards': 300,
            'receiving_touchdowns': 2,
            'receptions': 25,
            'salary': 7500
        },
        {
            'name': 'Player 3',
            'team': 'Team C',
            'position': 'WR',
            'games_played': 10,
            'receiving_yards': 1200,
            'receiving_touchdowns': 10,
            'receptions': 80,
            'salary': 8500
        }
    ]


def test_triple_layer_prediction_flow(mock_player_data):
    """
    Test the flow of data through the Triple-Layer Prediction Engine.
    
    This test verifies that:
    1. The ML Layer can generate predictions
    2. The API/Local AI Layer can enhance those predictions
    3. The Cloud AI Layer can combine and adjust predictions
    """
    # Create test instances
    ml_predictor = MLLayerPredictor()
    orchestrator = PredictionOrchestrator(ml_layer_predictor=ml_predictor)
    combiner = PredictionCombiner()
    
    # Mock the ML Layer prediction method
    with patch.object(ml_predictor, 'predict') as mock_ml_predict:
        # Set up mock return value for ML Layer
        mock_ml_predict.return_value = [
            {'value': 20.5, 'confidence': 0.8, 'player_name': 'Player 1'},
            {'value': 15.3, 'confidence': 0.7, 'player_name': 'Player 2'},
            {'value': 18.7, 'confidence': 0.9, 'player_name': 'Player 3'}
        ]
        
        # Test orchestrator prediction
        player_inputs = [str(player) for player in mock_player_data]
        predictions = orchestrator.predict(player_inputs)
        
        # Verify orchestrator output
        assert len(predictions) == len(mock_player_data)
        assert all('weighted_prediction' in pred for pred in predictions)
        assert all('transformer_prediction' in pred for pred in predictions)
        assert all('ml_layer_prediction' in pred for pred in predictions)
        
        # Test cloud layer combination
        cloud_input = {
            'ml_layer': mock_player_data,
            'api_ai_layer': predictions
        }
        
        # Mock the external factors integration
        with patch.object(combiner.external_factors_integrator, 'integrate_external_factors') as mock_external:
            mock_external.return_value = {'ml_layer': 0.1, 'api_ai_layer': 0.05}
            
            # Get combined predictions
            combined = combiner.combine_predictions(cloud_input)
            
            # Verify combiner output
            assert 'prediction' in combined
            assert 'confidence' in combined
            assert 'weights' in combined
            assert 'explanation' in combined


def test_generate_lineup_endpoint(client, mock_player_data):
    """
    Test the /generate_lineup endpoint with the Triple-Layer Prediction Engine.
    
    This test verifies that:
    1. The endpoint can handle requests
    2. The scraper, prediction engine, and optimizer work together
    3. The response contains the expected data
    """
    # Mock the scraper
    with patch.object(UFLScraper, 'scrape_player_data', return_value=mock_player_data):
        # Mock the fantasy prices
        with patch.object(UFLScraper, 'get_fantasy_prices') as mock_prices:
            mock_prices.return_value = pd.DataFrame([
                {'name': 'Player 1', 'position': 'QB', 'team': 'Team A', 'salary': 8000, 'platform': 'draftkings'},
                {'name': 'Player 2', 'position': 'RB', 'team': 'Team B', 'salary': 7500, 'platform': 'draftkings'},
                {'name': 'Player 3', 'position': 'WR', 'team': 'Team C', 'salary': 8500, 'platform': 'draftkings'}
            ])
            
            # Mock the prediction orchestrator
            with patch.object(prediction_orchestrator, 'predict') as mock_predict:
                mock_predict.return_value = [
                    {
                        'weighted_prediction': {
                            'transformer_result': {'value': 20.5},
                            'ml_layer_result': {'value': 19.8}
                        },
                        'confidence': 0.85
                    },
                    {
                        'weighted_prediction': {
                            'transformer_result': {'value': 15.3},
                            'ml_layer_result': {'value': 16.1}
                        },
                        'confidence': 0.75
                    },
                    {
                        'weighted_prediction': {
                            'transformer_result': {'value': 18.7},
                            'ml_layer_result': {'value': 17.9}
                        },
                        'confidence': 0.9
                    }
                ]
                
                # Mock the prediction combiner
                with patch.object(prediction_combiner, 'combine_predictions') as mock_combine:
                    mock_combine.return_value = {
                        'prediction': [20.5, 15.3, 18.7],
                        'confidence': [0.85, 0.75, 0.9],
                        'weights': {'ml_layer': 0.4, 'api_ai_layer': 0.6}
                    }
                    
                    # Mock the optimizer
                    with patch('optimizers.lineup_optimizer.LineupOptimizer.optimize') as mock_optimize:
                        # Create a mock lineup result
                        mock_lineup = pd.DataFrame([
                            {'name': 'Player 1', 'position': 'QB', 'team': 'Team A', 'salary': 8000, 'predicted_points': 20.5, 'lineup_number': 1},
                            {'name': 'Player 2', 'position': 'RB', 'team': 'Team B', 'salary': 7500, 'predicted_points': 15.3, 'lineup_number': 1},
                            {'name': 'Player 3', 'position': 'WR', 'team': 'Team C', 'salary': 8500, 'predicted_points': 18.7, 'lineup_number': 1}
                        ])
                        mock_optimize.return_value = mock_lineup
                        
                        # Make request to the endpoint
                        response = client.post('/generate_lineup', data={
                            'platform': 'draftkings',
                            'max_lineups': 1,
                            'min_salary': 0
                        })
                        
                        # Verify response
                        assert response.status_code == 200
                        data = json.loads(response.data)
                        assert data['success'] is True
                        assert 'lineup' in data
                        assert len(data['lineup']) == 3
                        assert 'metadata' in data
                        assert data['metadata']['platform'] == 'draftkings'