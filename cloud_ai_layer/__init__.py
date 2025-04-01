"""Cloud AI Layer initialization module.

Exposes key components and interfaces for the Cloud AI Layer.
"""

from .config import CloudAILayerConfig
from .external_factors import ExternalFactorsIntegrator
from .pattern_analyzer import CrossLeaguePatternAnalyzer
from .prediction_combiner import PredictionCombiner
from .interfaces import (
    PredictionRequest,
    PredictionResult,
    PredictionSourceType,
    CloudAILayerInterface,
    LayerCommunicationProtocol,
    create_prediction_request,
    create_prediction_result
)
from .exceptions import (
    CloudAILayerError,
    ExternalFactorIntegrationError,
    PatternAnalysisError,
    PredictionCombinerError,
    ConfigurationError,
    CacheError
)

__all__ = [
    # Configuration
    'CloudAILayerConfig',
    
    # Core Components
    'ExternalFactorsIntegrator',
    'CrossLeaguePatternAnalyzer',
    'PredictionCombiner',
    
    # Interfaces and Protocols
    'PredictionRequest',
    'PredictionResult',
    'PredictionSourceType',
    'CloudAILayerInterface',
    'LayerCommunicationProtocol',
    'create_prediction_request',
    'create_prediction_result',
    
    # Exceptions
    'CloudAILayerError',
    'ExternalFactorIntegrationError',
    'PatternAnalysisError',
    'PredictionCombinerError',
    'ConfigurationError',
    'CacheError'
]

# Version information
__version__ = '0.1.0'