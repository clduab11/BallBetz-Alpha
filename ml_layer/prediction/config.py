import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class PredictionConfig:
    """
    Configuration for prediction processes
    
    Loads configuration from environment variables with fallback defaults
    """
    
    # Prediction task configuration
    task_type: str = field(
        default_factory=lambda: os.getenv('UFL_PREDICTION_TASK', 'classification')
    )
    
    # Confidence threshold settings
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv('UFL_CONFIDENCE_THRESHOLD', '0.5'))
    )
    
    # Supported prediction tasks
    supported_tasks: List[str] = field(
        default_factory=lambda: ['classification', 'regression', 'ranking']
    )
    
    # Model loading configuration
    model_dir: str = field(
        default_factory=lambda: os.getenv(
            'UFL_MODEL_DIR', 
            os.path.join(os.getcwd(), 'ml_layer', 'models')
        )
    )
    
    # Prediction output configuration
    output_format: str = field(
        default_factory=lambda: os.getenv('UFL_OUTPUT_FORMAT', 'json')
    )
    
    # Logging and monitoring
    log_predictions: bool = field(
        default_factory=lambda: os.getenv('UFL_LOG_PREDICTIONS', 'true').lower() == 'true'
    )
    
    # Ensemble prediction settings
    use_ensemble: bool = field(
        default_factory=lambda: os.getenv('UFL_USE_ENSEMBLE', 'false').lower() == 'true'
    )
    
    # Maximum number of ensemble models
    max_ensemble_models: int = field(
        default_factory=lambda: int(os.getenv('UFL_MAX_ENSEMBLE_MODELS', '5'))
    )
    
    def validate(self) -> None:
        """
        Validate configuration parameters
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate task type
        if self.task_type not in self.supported_tasks:
            raise ValueError(
                f"Unsupported prediction task: {self.task_type}. "
                f"Supported tasks: {self.supported_tasks}"
            )
        
        # Validate confidence threshold
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f"Invalid confidence threshold: {self.confidence_threshold}. "
                "Must be between 0 and 1."
            )
        
        # Validate output format
        supported_formats = ['json', 'csv', 'numpy']
        if self.output_format not in supported_formats:
            raise ValueError(
                f"Unsupported output format: {self.output_format}. "
                f"Supported formats: {supported_formats}"
            )
        
        # Validate ensemble settings
        if self.use_ensemble and self.max_ensemble_models <= 1:
            raise ValueError(
                f"Invalid max ensemble models: {self.max_ensemble_models}. "
                "Must be > 1 when ensemble prediction is enabled."
            )
        
        # Ensure model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    @classmethod
    def from_env(cls) -> 'PredictionConfig':
        """
        Create configuration from environment variables
        
        Returns:
            PredictionConfig: Configured instance
        """
        config = cls()
        config.validate()
        return config