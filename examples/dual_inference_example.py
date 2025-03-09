"""
Example script demonstrating the dual-inference capability of BallBetz.

This script shows how to use the PlayerPerformancePredictor with different model providers:
- scikit-learn: Traditional machine learning model
- Ollama: Local LLM inference
- OpenAI: Cloud-based LLM inference

It demonstrates how to:
1. Initialize the predictor
2. Switch between different model providers
3. Make predictions with each provider
4. Handle fallback mechanisms
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from BallBetz
sys.path.append(str(Path(__file__).parent.parent))

from models import PlayerPerformancePredictor, ModelProvider, ModelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create a sample dataset for demonstration."""
    return pd.DataFrame([
        {
            'name': 'Luis Perez',
            'position': 'QB',
            'team': 'ARL',
            'passing_yards': 2310,
            'passing_touchdowns': 18,
            'interceptions': 4,
            'rushing_yards': 120,
            'rushing_touchdowns': 2,
            'receiving_yards': 0,
            'receiving_touchdowns': 0,
            'receptions': 0,
            'games_played': 10
        },
        {
            'name': 'Adrian Martinez',
            'position': 'RB',
            'team': 'BIR',
            'passing_yards': 0,
            'passing_touchdowns': 0,
            'interceptions': 0,
            'rushing_yards': 528,
            'rushing_touchdowns': 3,
            'receiving_yards': 150,
            'receiving_touchdowns': 1,
            'receptions': 15,
            'games_played': 10
        },
        {
            'name': 'Justin Hall',
            'position': 'WR',
            'team': 'HOU',
            'passing_yards': 0,
            'passing_touchdowns': 0,
            'interceptions': 0,
            'rushing_yards': 45,
            'rushing_touchdowns': 0,
            'receiving_yards': 604,
            'receiving_touchdowns': 3,
            'receptions': 56,
            'games_played': 10
        }
    ])

def main():
    """Main function demonstrating dual-inference capability."""
    logger.info("BallBetz Dual-Inference Example")
    
    # Create sample data
    player_data = create_sample_data()
    logger.info(f"Created sample data with {len(player_data)} players")
    
    # Initialize the predictor
    predictor = PlayerPerformancePredictor()
    logger.info("Initialized PlayerPerformancePredictor")
    
    # Get available providers
    available_providers = predictor.get_available_providers()
    logger.info(f"Available model providers: {available_providers}")
    
    # Make predictions with each available provider
    results = {}
    
    for provider in available_providers:
        logger.info(f"Making predictions with {provider} provider")
        
        # Set the provider
        predictor.set_provider(provider)
        
        # Make predictions
        result = predictor.predict(player_data)
        
        if result is not None:
            results[provider] = result
            logger.info(f"Predictions from {provider}:")
            for _, player in result.iterrows():
                logger.info(f"  {player['name']} ({player['position']}): {player['predicted_points']} points")
        else:
            logger.error(f"Failed to make predictions with {provider} provider")
    
    # Compare predictions from different providers
    if len(results) > 1:
        logger.info("\nComparison of predictions from different providers:")
        
        for player_idx in range(len(player_data)):
            player_name = player_data.iloc[player_idx]['name']
            logger.info(f"\n{player_name} ({player_data.iloc[player_idx]['position']}):")
            
            for provider, result in results.items():
                points = result.iloc[player_idx]['predicted_points']
                lower = result.iloc[player_idx]['prediction_lower']
                upper = result.iloc[player_idx]['prediction_upper']
                logger.info(f"  {provider}: {points} points (range: {lower}-{upper})")
    
    # Demonstrate fallback mechanism
    logger.info("\nDemonstrating fallback mechanism:")
    
    # Try with a non-existent provider first
    try:
        # This will use the fallback mechanism if enabled
        result = predictor.predict(player_data, "non_existent_provider")
        
        if result is not None:
            logger.info(f"Fallback successful, used provider: {result.iloc[0]['model_provider']}")
        else:
            logger.info("Fallback not enabled or no available fallback providers")
    except Exception as e:
        logger.error(f"Error demonstrating fallback: {str(e)}")

if __name__ == "__main__":
    main()