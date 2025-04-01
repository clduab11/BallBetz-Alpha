import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class ModelSelectionConfig:
    """
    Configuration for model selection processes
    
    Loads configuration from environment variables with fallback defaults
    """
    
    # Cross-validation settings
    cv_folds: int = field(
        default_factory=lambda: int(os.getenv('UFL_CV_FOLDS', '5'))
    )
    
    # Evaluation metrics
    primary_metric: str = field(
        default_factory=lambda: os.getenv('UFL_PRIMARY_METRIC', 'accuracy')
    )
    
    # Model selection strategy
    selection_strategy: str = field(
        default_factory=lambda: os.getenv('UFL_MODEL_SELECTION_STRATEGY', 'best_cv')
    )
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = field(
        default_factory=lambda: os.getenv('UFL_ENABLE_HYPERPARAMETER_TUNING', 'true').lower() == 'true'
    )
    
    # Maximum number of models to evaluate
    max_models_to_evaluate: int = field(
        default_factory=lambda: int(os.getenv('UFL_MAX_MODELS', '5'))
    )
    
    # Supported metrics
    supported_metrics: List[str] = field(
        default_factory=lambda: [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
            'mse', 'mae', 'r2'
        ]
    )
    
    # Supported selection strategies
    supported_strategies: List[str] = field(
        default_factory=lambda: ['best_cv', 'ensemble', 'weighted']
    )
    
    def validate(self) -> None:
        """
        Validate configuration parameters
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate cross-validation folds
        if self.cv_folds < 2:
            raise ValueError(f"Invalid cross-validation folds: {self.cv_folds}. Must be >= 2.")
        
        # Validate primary metric
        if self.primary_metric not in self.supported_metrics:
            raise ValueError(
                f"Unsupported primary metric: {self.primary_metric}. "
                f"Supported metrics: {self.supported_metrics}"
            )
        
        # Validate selection strategy
        if self.selection_strategy not in self.supported_strategies:
            raise ValueError(
                f"Unsupported selection strategy: {self.selection_strategy}. "
                f"Supported strategies: {self.supported_strategies}"
            )
        
        # Validate max models
        if self.max_models_to_evaluate <= 0:
            raise ValueError(f"Invalid max_models_to_evaluate: {self.max_models_to_evaluate}. Must be > 0.")
    
    @classmethod
    def from_env(cls) -> 'ModelSelectionConfig':
        """
        Create configuration from environment variables
        
        Returns:
            ModelSelectionConfig: Configured instance
        """
        config = cls()
        config.validate()
        return config