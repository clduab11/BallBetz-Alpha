#!/usr/bin/env python3
"""
Example script demonstrating the BallBetz-Alpha API/Local AI Layer
prediction capabilities.
"""

import logging
import sys
import os

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_ai_layer.config import APIAILayerConfig
from api_ai_layer.orchestrator import PredictionOrchestrator
from api_ai_layer.providers.openai_client import OpenAIProvider
from api_ai_layer.providers.local_model_client import LocalTransformerProvider

def setup_logging():
    """
    Configure logging for the example script.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """
    Demonstrate transformer model prediction capabilities.
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = APIAILayerConfig.load_config()
        logger.info("Configuration loaded successfully")

        # Create providers
        openai_provider = OpenAIProvider(config)
        local_provider = LocalTransformerProvider(config)
        logger.info("Providers initialized")

        # Create prediction orchestrator
        orchestrator = PredictionOrchestrator(
            providers=[openai_provider, local_provider],
            config=config
        )
        logger.info("Prediction orchestrator created")

        # Example prediction inputs
        prediction_inputs = [
            "Predict the next major trend in machine learning",
            "Explain the potential impact of quantum computing on AI",
            "Describe the future of sustainable technology"
        ]

        # Generate predictions
        logger.info(f"Generating predictions for {len(prediction_inputs)} inputs")
        predictions = orchestrator.predict(prediction_inputs)

        # Display predictions
        for i, prediction in enumerate(predictions, 1):
            print(f"\nPrediction {i}:")
            print(f"Input: {prediction_inputs[i-1]}")
            print(f"Transformer Prediction: {prediction['transformer_prediction']['prediction']}")
            print(f"ML Layer Prediction: {prediction['ml_layer_prediction']['prediction']}")
            print(f"Confidence: {prediction['confidence']:.2f}")
            print(f"Weighted Prediction: {prediction['weighted_prediction']}")

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()