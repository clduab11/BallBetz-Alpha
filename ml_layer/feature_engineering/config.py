import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .exceptions import FeatureEngineeringConfigError

@dataclass
class FeatureEngineeringConfig:
    """
    Configuration for feature engineering processes
    
    Loads configuration from environment variables with fallback defaults
    """
    
    # Data imputation strategies
    numeric_imputation_strategy: str = field(
        default_factory=lambda: os.getenv('UFL_NUMERIC_IMPUTATION', 'mean')
    )
    
    categorical_imputation_strategy: str = field(
        default_factory=lambda: os.getenv('UFL_CATEGORICAL_IMPUTATION', 'most_frequent')
    )
    
    # Feature scaling
    scale_numeric_features: bool = field(
        default_factory=lambda: os.getenv('UFL_SCALE_NUMERIC', 'true').lower() == 'true'
    )
    
    # Feature selection
    max_features_to_select: int = field(
        default_factory=lambda: int(os.getenv('UFL_MAX_FEATURES', '10'))
    )
    
    # Logging and debugging
    verbose: bool = field(
        default_factory=lambda: os.getenv('UFL_VERBOSE_PREPROCESSING', 'false').lower() == 'true'
    )
    
    # Supported imputation strategies
    supported_numeric_strategies: List[str] = field(
        default_factory=lambda: ['mean', 'median', 'most_frequent', 'constant']
    )
    
    supported_categorical_strategies: List[str] = field(
        default_factory=lambda: ['most_frequent', 'constant']
    )
    
    def validate(self) -> None:
        """
        Validate configuration parameters
        
        Raises:
            FeatureEngineeringConfigError: If configuration is invalid
        """
        # Validate numeric imputation strategy
        if self.numeric_imputation_strategy not in self.supported_numeric_strategies:
            raise FeatureEngineeringConfigError(
                f"Invalid numeric imputation strategy: {self.numeric_imputation_strategy}. "
                f"Supported strategies: {self.supported_numeric_strategies}"
            )
        
        # Validate categorical imputation strategy
        if self.categorical_imputation_strategy not in self.supported_categorical_strategies:
            raise FeatureEngineeringConfigError(
                f"Invalid categorical imputation strategy: {self.categorical_imputation_strategy}. "
                f"Supported strategies: {self.supported_categorical_strategies}"
            )
        
        # Validate max features
        if self.max_features_to_select <= 0:
            raise FeatureEngineeringConfigError(
                f"Invalid max_features_to_select: {self.max_features_to_select}. "
                "Must be greater than 0."
            )
    
    @classmethod
    def from_env(cls) -> 'FeatureEngineeringConfig':
        """
        Create configuration from environment variables
        
        Returns:
            FeatureEngineeringConfig: Configured instance
        """
        config = cls()
        config.validate()
        return config