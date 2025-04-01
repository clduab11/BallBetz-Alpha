"""External Factors Integration Module.

This module handles the integration and normalization of contextual data
such as weather, injuries, and team statistics for prediction enhancement.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Union
import requests
import numpy as np
import re
from .config import CloudAILayerConfig
from .exceptions import ExternalFactorIntegrationError, ValidationError

class ExternalFactorsIntegrator:
    """
    Integrates and normalizes external contextual factors for prediction enhancement.
    
    Supports multiple data sources with configurable impact scoring and normalization.
    """

    def __init__(self, config: CloudAILayerConfig = None):
        """
        Initialize the External Factors Integrator.
        
        Args:
            config: Configuration for external factors integration
        """
        self.config = config or CloudAILayerConfig
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.LOGGING_LEVEL)
        
    def _validate_factor_type(self, factor_type: str) -> None:
        """
        Validate the factor type parameter.
        
        Args:
            factor_type: Type of external factor to validate
            
        Raises:
            ValidationError: If factor_type is invalid
        """
        valid_factor_types = list(self.config.EXTERNAL_FACTORS_SOURCES.keys())
        
        if not factor_type:
            raise ValidationError("Factor type cannot be empty")
            
        if not isinstance(factor_type, str):
            raise ValidationError(f"Factor type must be a string, got {type(factor_type).__name__}")
            
        if factor_type not in valid_factor_types:
            raise ValidationError(
                f"Invalid factor type: {factor_type}. Must be one of: {', '.join(valid_factor_types)}"
            )
    
    def _validate_api_url(self, url: str) -> None:
        """
        Validate an API URL for security.
        
        Args:
            url: The URL to validate
            
        Raises:
            ValidationError: If URL is invalid or potentially malicious
        """
        if not url:
            raise ValidationError("API URL cannot be empty")
            
        # Check for valid URL format
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            
        if not url_pattern.match(url):
            raise ValidationError(f"Invalid URL format: {url}")
            
        # Check for allowed domains (whitelist approach)
        allowed_domains = self.config.ALLOWED_API_DOMAINS
        if allowed_domains:
            domain = urlparse(url).netloc.split(':')[0]  # Extract domain without port
            if domain not in allowed_domains:
                raise ValidationError(f"Domain not in allowed list: {domain}")
    
    def _sanitize_api_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize API response data to prevent injection attacks.
        
        Args:
            response_data: Raw API response data
            
        Returns:
            Sanitized response data
        """
        # Convert to string and back to ensure proper JSON structure
        try:
            sanitized = json.loads(json.dumps(response_data))
            return sanitized
        except (TypeError, json.JSONDecodeError) as e:
            raise ValidationError(f"Invalid API response format: {e}")

    def fetch_external_data(self, factor_type: str) -> Dict[str, Any]:
        """
        Fetch external data from configured sources.
        
        Args:
            factor_type: Type of external factor (weather, injuries, team_stats)
        
        Returns:
            Dictionary of external factor data
        
        Raises:
            ExternalFactorIntegrationError: If data fetching fails
        """
        # Validate factor type
        self._validate_factor_type(factor_type)
        
        # Get API source URL
        source = self.config.EXTERNAL_FACTORS_SOURCES.get(factor_type, '')
        
        # Validate API URL
        self._validate_api_url(source)
        
        try:
            # Placeholder for actual API calls - replace with real implementation
            headers = {'User-Agent': 'BallBetz-Alpha/1.0'}
            response = requests.get(source, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._sanitize_api_response(data)
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch {factor_type} data: {e}")
            raise ExternalFactorIntegrationError(f"Could not fetch {factor_type} data") from e

    def normalize_factor(self, data: Dict[str, Any], factor_type: str) -> np.ndarray:
        """
        Normalize external factor data using statistical techniques.
        
        Args:
            data: Raw external factor data
            factor_type: Type of external factor
        
        Returns:
            Normalized numpy array of factor values
        """
        # Validate factor type
        self._validate_factor_type(factor_type)
        
        try:
            # Implement factor-specific normalization logic
            if factor_type == 'weather':
                return self._normalize_weather(data)
            elif factor_type == 'injuries':
                return self._normalize_injuries(data)
            elif factor_type == 'team_stats':
                return self._normalize_team_stats(data)
            else:
                raise ValueError(f"Unsupported factor type: {factor_type}")
        except Exception as e:
            self.logger.error(f"Normalization failed for {factor_type}: {e}")
            raise ExternalFactorIntegrationError(f"Normalization failed for {factor_type}") from e

    def _normalize_weather(self, weather_data: Dict[str, Any]) -> np.ndarray:
        """
        Normalize weather-related data.
        
        Args:
            weather_data: Raw weather data dictionary
        
        Returns:
            Normalized weather impact scores
        """
        # Validate input data
        required_fields = ['temperature', 'humidity', 'wind_speed']
        for field in required_fields:
            if field not in weather_data:
                self.logger.warning(f"Missing required field in weather data: {field}")
                weather_data[field] = 0
                
        # Ensure numeric values
        weather_data = {k: float(v) if isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit() else 0 for k, v in weather_data.items()}
        # Example normalization - replace with actual implementation
        temperature = weather_data.get('temperature', 0)
        humidity = weather_data.get('humidity', 0)
        wind_speed = weather_data.get('wind_speed', 0)
        
        # Min-Max normalization
        normalized_temp = (temperature - 0) / (100 - 0)
        normalized_humidity = (humidity - 0) / (100 - 0)
        normalized_wind = (wind_speed - 0) / (50 - 0)
        
        return np.array([normalized_temp, normalized_humidity, normalized_wind])

    def _normalize_injuries(self, injuries_data: Dict[str, Any]) -> np.ndarray:
        """
        Normalize injury-related data.
        
        Args:
            injuries_data: Raw injuries data dictionary
        
        Returns:
            Normalized injury impact scores
        """
        # Validate input data
        if not isinstance(injuries_data, dict):
            self.logger.warning(f"Invalid injuries data format: {type(injuries_data).__name__}")
            injuries_data = {}
            
        # Ensure key_players is a list
        if not isinstance(injuries_data.get('key_players', []), list):
            self.logger.warning("key_players is not a list")
            injuries_data['key_players'] = []
        # Example normalization - replace with actual implementation
        key_players_injured = len(injuries_data.get('key_players', []))
        total_team_players = injuries_data.get('total_players', 1)
        
        injury_impact = key_players_injured / total_team_players
        return np.array([injury_impact])

    def _normalize_team_stats(self, team_stats: Dict[str, Any]) -> np.ndarray:
        """
        Normalize team statistics.
        
        Args:
            team_stats: Raw team statistics dictionary
        
        Returns:
            Normalized team performance scores
        """
        # Validate input data
        if not isinstance(team_stats, dict):
            self.logger.warning(f"Invalid team stats format: {type(team_stats).__name__}")
            team_stats = {}
            
        # Ensure recent_performance is a list of numbers
        recent_performance = team_stats.get('recent_performance', [])
        if not isinstance(recent_performance, list) or not all(isinstance(x, (int, float)) for x in recent_performance):
            team_stats['recent_performance'] = []
        # Example normalization - replace with actual implementation
        win_rate = team_stats.get('win_rate', 0)
        recent_performance = team_stats.get('recent_performance', [])
        
        # Calculate recent performance trend
        performance_trend = np.mean(recent_performance) if recent_performance else 0
        
        return np.array([win_rate, performance_trend])

    def calculate_factor_impact(self, normalized_data: np.ndarray, factor_type: str) -> float:
        """
        Calculate the overall impact score for a given external factor.
        
        Args:
            normalized_data: Normalized factor data
            factor_type: Type of external factor
        
        Returns:
            Impact score between 0 and 1
        """
        # Validate factor type
        self._validate_factor_type(factor_type)
        
        # Validate normalized data
        if not isinstance(normalized_data, np.ndarray):
            raise ValidationError(f"Normalized data must be a numpy array, got {type(normalized_data).__name__}")
            
        # Implement factor-specific impact scoring
        if factor_type == 'weather':
            # Example: Weighted impact of temperature, humidity, wind
            return np.dot(normalized_data, [0.5, 0.3, 0.2])
        elif factor_type == 'injuries':
            return normalized_data[0]  # Simple injury impact
        elif factor_type == 'team_stats':
            return np.mean(normalized_data)
        else:
            return 0.0

    def integrate_external_factors(self) -> Dict[str, float]:
        """
        Integrate and score all configured external factors.
        
        Returns:
            Dictionary of factor types and their impact scores
        """
        factor_impacts = {}
        
        # Get list of configured factor types
        factor_types = list(self.config.EXTERNAL_FACTORS_SOURCES.keys())
        
        for factor_type in factor_types:
            try:
                self.logger.info(f"Processing external factor: {factor_type}")
                raw_data = self.fetch_external_data(factor_type)
                normalized_data = self.normalize_factor(raw_data, factor_type)
                impact_score = self.calculate_factor_impact(normalized_data, factor_type)
                factor_impacts[factor_type] = impact_score
                self.logger.info(f"Successfully processed {factor_type} with impact score: {impact_score}")
            except ValidationError as e:
                self.logger.error(f"Validation error for {factor_type}: {e}")
            except ExternalFactorIntegrationError as e:
                self.logger.warning(f"Skipping {factor_type} due to integration error: {e}")
        
        return factor_impacts