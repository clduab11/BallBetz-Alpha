"""Exceptions for the Cloud AI Layer.

This module defines custom exceptions used throughout the Cloud AI Layer.
"""

class CloudAILayerError(Exception):
    """Base exception for all Cloud AI Layer errors."""
    pass


class ExternalFactorIntegrationError(CloudAILayerError):
    """Exception raised when external factor integration fails."""
    pass


class ValidationError(CloudAILayerError):
    """Exception raised when input validation fails."""
    pass


class PredictionCombinerError(CloudAILayerError):
    """Exception raised when prediction combination fails."""
    pass


class PatternAnalysisError(CloudAILayerError):
    """Exception raised when pattern analysis fails."""
    pass


class ConfigurationError(CloudAILayerError):
    """Exception raised when configuration is invalid."""
    pass