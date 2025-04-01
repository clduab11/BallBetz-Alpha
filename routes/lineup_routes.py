"""Lineup generation routes for BallBetz-Alpha.

This module handles lineup generation routes for the application.
"""

import logging
import pandas as pd
from datetime import datetime
from flask import request, jsonify
from flask_login import login_required

from . import lineup_bp
from utils.error_handlers import handle_errors
from scrapers.ufl_scraper import UFLScraper
from optimizers.lineup_optimizer import LineupOptimizer
from api_ai_layer.orchestrator import PredictionOrchestrator
from ml_layer.prediction.ml_layer_predictor import MLLayerPredictor
from cloud_ai_layer.prediction_combiner import PredictionCombiner

# Set up logging
logger = logging.getLogger(__name__)

# Initialize components
try:
    scraper = UFLScraper()
    
    # Initialize Triple-Layer Prediction Engine
    ml_layer_predictor = MLLayerPredictor()
    prediction_orchestrator = PredictionOrchestrator(ml_layer_predictor=ml_layer_predictor)
    prediction_combiner = PredictionCombiner()
    
    # Initialize lineup optimizers
    dk_optimizer = LineupOptimizer('draftkings') 
    fd_optimizer = LineupOptimizer('fanduel')
    logger.info("All lineup generation components initialized successfully")
    
except Exception as e:
    logger.error(f"Error initializing lineup generation components: {str(e)}")
    raise


@lineup_bp.route('/generate_lineup', methods=['POST'])
@handle_errors
def generate_lineup():
    """Generate optimal lineup based on form parameters."""
    logger.info("=== Generating Lineup - Parameter Diagnostics ===")
    # Validate input parameters
    platform = request.form.get('platform', 'draftkings').lower()
    max_lineups = int(request.form.get('max_lineups', 1))
    min_salary = float(request.form.get('min_salary', 0))
    logger.info(f"Received parameters - platform: {platform}, max_lineups: {max_lineups}, min_salary: {min_salary}")

    if platform not in ['draftkings', 'fanduel']:
        raise ValueError(f"Invalid platform: {platform}")

    # Scrape latest player data
    try:
        raw_player_data = scraper.scrape_player_data()
        if not raw_player_data:
            raise ValueError("No player data retrieved")
        
        # Convert to DataFrame
        player_data = pd.DataFrame(raw_player_data)
        logger.info(f"Scraped player data: {len(player_data)} records")
    except Exception as e:
        logger.error(f"Error scraping player data: {str(e)}")
        raise ValueError(f"Error retrieving player data: {str(e)}")

    # Get fantasy prices
    prices = scraper.get_fantasy_prices(platform)
    if not prices.empty:
        player_data = player_data.merge(prices, on='name', how='left')
        logger.info(f"Merged fantasy prices: {len(prices)} records")
    else: 
        logger.warning("Fantasy prices data is empty - using placeholder data")
        # Add placeholder salary for testing
        if 'salary' not in player_data.columns:
            player_data['salary'] = 5000
            logger.info("Added placeholder salary data for testing")

    # Generate predictions using Triple-Layer Prediction Engine
    try:
        # Convert player data to format expected by prediction engine
        player_inputs = player_data.to_dict('records') 
        
        # Get predictions from orchestrator
        predictions = prediction_orchestrator.predict([str(player) for player in player_inputs])
        
        # Add predictions to player data
        for i, prediction in enumerate(predictions):
            if i < len(player_data):
                player_data.loc[i, 'predicted_points'] = prediction.get('weighted_prediction', {}).get('transformer_result', {}).get('value', 0)
        
        # Apply cloud layer for additional context and adjustments
        cloud_predictions = prediction_combiner.combine_predictions({
            'ml_layer': player_data.to_dict('records'),
            'api_ai_layer': predictions
        })
        
        # Update predictions with cloud layer results
        for i, player in enumerate(player_data.index):
            if i < len(cloud_predictions.get('prediction', [])):
                player_data.loc[player, 'predicted_points'] = cloud_predictions.get('prediction', [])[i]
        
        logger.info(f"Generated predictions for {len(player_data)} players")
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise ValueError(f"Error generating predictions: {str(e)}")

    # Optimize lineup
    optimizer = dk_optimizer if platform == 'draftkings' else fd_optimizer
    lineup = optimizer.optimize(
        player_data,
        max_lineups=max_lineups,
        min_salary=min_salary
    )

    if lineup.empty:
        logger.error("Generated lineup is empty")
        raise ValueError("Could not generate valid lineup")

    # Format lineup for display
    result = lineup.to_dict(orient='records')
    for player in result:
        player['salary'] = f"${player['salary']:,}"
        player['predicted_points'] = f"{player['predicted_points']:.2f}"

    return jsonify({
        'success': True, 
        'lineup': result, 
        'metadata': {
            'platform': platform, 
            'generated_at': datetime.now().isoformat(), 
            'lineup_count': len(lineup['lineup_number'].unique())
        }
    })