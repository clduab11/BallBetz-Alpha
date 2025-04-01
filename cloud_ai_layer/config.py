"""Configuration module for Cloud AI Layer.

This module manages configuration settings for the Cloud AI Layer,
ensuring no hardcoded values and supporting environment-based configuration.
"""

import os
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CloudAILayerConfig:
    """Configuration class for Cloud AI Layer settings."""

    # External Factors Configuration
    EXTERNAL_FACTORS_SOURCES: Dict[str, str] = {
        'weather': os.getenv('EXTERNAL_WEATHER_SOURCE', 'default_weather_api'),
        'injuries': os.getenv('EXTERNAL_INJURIES_SOURCE', 'default_injuries_api'),
        'team_stats': os.getenv('EXTERNAL_TEAM_STATS_SOURCE', 'default_stats_api')
    }

    # API Security Configuration
    ALLOWED_API_DOMAINS: List[str] = []
    _allowed_domains = os.getenv('ALLOWED_API_DOMAINS', '')
    if _allowed_domains:
        ALLOWED_API_DOMAINS = [domain.strip() for domain in _allowed_domains.split(',')]
    else:
        # Default to allowing domains from the configured sources
        ALLOWED_API_DOMAINS = [urlparse(url).netloc for url in EXTERNAL_FACTORS_SOURCES.values() if url]

    # Pattern Analysis Configuration
    CROSS_LEAGUE_SIMILARITY_THRESHOLD: float = float(
        os.getenv('CROSS_LEAGUE_SIMILARITY_THRESHOLD', '0.7')
    )
    MAX_HISTORICAL_PATTERNS: int = int(
        os.getenv('MAX_HISTORICAL_PATTERNS', '100')
    )

    # Prediction Combiner Configuration
    DEFAULT_PREDICTION_WEIGHTS: Dict[str, float] = {
        'ml_layer': float(os.getenv('ML_LAYER_WEIGHT', '0.4')),
        'api_ai_layer': float(os.getenv('API_AI_LAYER_WEIGHT', '0.3')),
        'external_factors': float(os.getenv('EXTERNAL_FACTORS_WEIGHT', '0.3'))
    }

    # Logging and Explainability Configuration
    LOGGING_LEVEL: str = os.getenv('CLOUD_AI_LOGGING_LEVEL', 'INFO')
    ENABLE_PREDICTION_EXPLANATIONS: bool = os.getenv(
        'ENABLE_PREDICTION_EXPLANATIONS', 'True'
    ).lower() == 'true'

    # Caching Configuration
    CACHE_ENABLED: bool = os.getenv('CLOUD_AI_CACHE_ENABLED', 'True').lower() == 'true'
    CACHE_EXPIRATION_SECONDS: int = int(
        os.getenv('CLOUD_AI_CACHE_EXPIRATION', '3600')  # 1 hour default
    )
    
    # API Request Configuration
    API_REQUEST_TIMEOUT: int = int(
        os.getenv('API_REQUEST_TIMEOUT', '10')  # 10 seconds default
    )
    API_MAX_RETRIES: int = int(os.getenv('API_MAX_RETRIES', '3'))

    @classmethod
    def validate_config(cls) -> None:
        """
        Validate configuration settings.
        Raises ValueError if any configuration is invalid.
        """
        # Validate prediction weights
        total_weight = sum(cls.DEFAULT_PREDICTION_WEIGHTS.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Prediction weights must sum to 1, current sum: {total_weight}")

        # Validate similarity threshold
        if not (0 <= cls.CROSS_LEAGUE_SIMILARITY_THRESHOLD <= 1):
            raise ValueError(
                f"Similarity threshold must be between 0 and 1, "
                f"current value: {cls.CROSS_LEAGUE_SIMILARITY_THRESHOLD}"
            )
            
        # Validate API configuration
        if not cls.ALLOWED_API_DOMAINS:
            from .exceptions import ConfigurationError
            raise ConfigurationError(
                "No allowed API domains configured. This is a security risk.")

    @classmethod
    def get_external_factor_sources(cls) -> Dict[str, str]:
        """
        Retrieve external factor sources with fallback to defaults.
        
        Returns:
            Dict of external factor sources
        """
        return cls.EXTERNAL_FACTORS_SOURCES.copy()
        
    @classmethod
    def is_domain_allowed(cls, domain: str) -> bool:
        """
        Check if a domain is in the allowed list.
        
        Args:
            domain: Domain to check
            
        Returns:
            True if domain is allowed, False otherwise
        """
        return domain in cls.ALLOWED_API_DOMAINS