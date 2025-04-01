import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class TrainingConfig:
    """
    Configuration for model training processes
    
    Loads configuration from environment variables with fallback defaults
    """
    
    # Training hyperparameters
    epochs: int = field(
        default_factory=lambda: int(os.getenv('UFL_TRAINING_EPOCHS', '100'))
    )
    
    batch_size: int = field(
        default_factory=lambda: int(os.getenv('UFL_BATCH_SIZE', '32'))
    )
    
    learning_rate: float = field(
        default_factory=lambda: float(os.getenv('UFL_LEARNING_RATE', '0.01'))
    )
    
    # Training monitoring
    early_stopping: bool = field(
        default_factory=lambda: os.getenv('UFL_EARLY_STOPPING', 'true').lower() == 'true'
    )
    
    patience: int = field(
        default_factory=lambda: int(os.getenv('UFL_EARLY_STOPPING_PATIENCE', '10'))
    )
    
    # Logging and tracking
    log_interval: int = field(
        default_factory=lambda: int(os.getenv('UFL_LOG_INTERVAL', '10'))
    )
    
    # Checkpointing
    checkpoint_dir: str = field(
        default_factory=lambda: os.getenv(
            'UFL_CHECKPOINT_DIR', 
            os.path.join(os.getcwd(), 'ml_layer', 'checkpoints')
        )
    )
    
    save_best_only: bool = field(
        default_factory=lambda: os.getenv('UFL_SAVE_BEST_ONLY', 'true').lower() == 'true'
    )
    
    # Prediction task configuration
    task_type: str = field(
        default_factory=lambda: os.getenv('UFL_TASK_TYPE', 'classification')
    )
    
    # Supported task types
    supported_tasks: List[str] = field(
        default_factory=lambda: ['classification', 'regression', 'ranking']
    )
    
    def validate(self) -> None:
        """
        Validate configuration parameters
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate epochs
        if self.epochs <= 0:
            raise ValueError(f"Invalid epochs: {self.epochs}. Must be > 0.")
        
        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch size: {self.batch_size}. Must be > 0.")
        
        # Validate learning rate
        if self.learning_rate <= 0:
            raise ValueError(f"Invalid learning rate: {self.learning_rate}. Must be > 0.")
        
        # Validate task type
        if self.task_type not in self.supported_tasks:
            raise ValueError(
                f"Unsupported task type: {self.task_type}. "
                f"Supported tasks: {self.supported_tasks}"
            )
        
        # Validate early stopping
        if self.early_stopping and self.patience <= 0:
            raise ValueError(f"Invalid patience: {self.patience}. Must be > 0 when early stopping is enabled.")
        
        # Validate checkpoint directory
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    @classmethod
    def from_env(cls) -> 'TrainingConfig':
        """
        Create configuration from environment variables
        
        Returns:
            TrainingConfig: Configured instance
        """
        config = cls()
        config.validate()
        return config