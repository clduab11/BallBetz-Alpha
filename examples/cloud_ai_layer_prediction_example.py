"""Example demonstrating the usage of the Cloud AI Layer 
for intelligent prediction generation.
"""

import logging
from cloud_ai_layer import (
    PredictionCombiner,
    ExternalFactorsIntegrator,
    CrossLeaguePatternAnalyzer,
    create_prediction_request,
    PredictionSourceType
)

def setup_logging():
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def simulate_prediction_sources():
    """
    Simulate prediction data from different sources.
    
    Returns:
        Dictionary of predictions from various sources
    """
    return {
        'ml_layer': 0.7,  # Machine Learning Layer prediction
        'api_ai_layer': 0.6,  # API/AI Layer prediction
        'external_factors': 0.5  # External Factors prediction
    }

def analyze_league_patterns(pattern_analyzer):
    """
    Demonstrate cross-league pattern analysis.
    
    Args:
        pattern_analyzer: CrossLeaguePatternAnalyzer instance
    
    Returns:
        Pattern analysis results
    """
    # Sample league data for pattern analysis
    current_league = {
        'league_id': 'league_a',
        'win_rate': 0.7,
        'avg_score': 100,
        'home_advantage': 0.6,
        'team_strength': 0.8
    }
    
    historical_leagues = [
        {
            'league_id': 'league_b',
            'win_rate': 0.6,
            'avg_score': 95,
            'home_advantage': 0.5,
            'team_strength': 0.7
        },
        {
            'league_id': 'league_c',
            'win_rate': 0.8,
            'avg_score': 105,
            'home_advantage': 0.7,
            'team_strength': 0.9
        }
    ]
    
    return pattern_analyzer.analyze_cross_league_patterns(
        current_league, 
        historical_leagues
    )

def main():
    """
    Main function demonstrating Cloud AI Layer capabilities.
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize Cloud AI Layer components
        external_factors_integrator = ExternalFactorsIntegrator()
        pattern_analyzer = CrossLeaguePatternAnalyzer()
        prediction_combiner = PredictionCombiner()
        
        # Integrate external factors
        logger.info("Integrating external factors...")
        external_factor_impacts = external_factors_integrator.integrate_external_factors()
        logger.info(f"External Factor Impacts: {external_factor_impacts}")
        
        # Analyze cross-league patterns
        logger.info("Analyzing cross-league patterns...")
        pattern_analysis = analyze_league_patterns(pattern_analyzer)
        logger.info(f"Similar Patterns Found: {len(pattern_analysis['similar_patterns'])}")
        
        # Simulate predictions from different sources
        predictions = simulate_prediction_sources()
        
        # Create a prediction request
        prediction_request = create_prediction_request(
            context={
                'external_factors': external_factor_impacts,
                'league_patterns': pattern_analysis
            },
            source_type=PredictionSourceType.ML_MODEL
        )
        
        # Combine predictions
        logger.info("Combining predictions...")
        final_prediction = prediction_combiner.combine_predictions(predictions)
        
        # Display results
        logger.info("Prediction Results:")
        logger.info(f"Final Prediction: {final_prediction['prediction']}")
        logger.info(f"Confidence Score: {final_prediction['confidence']}")
        
        # Display prediction explanation if available
        if final_prediction.get('explanation'):
            logger.info("Prediction Explanation:")
            logger.info(final_prediction['explanation'])
    
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()