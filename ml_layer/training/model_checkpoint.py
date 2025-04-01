import os
import logging
import joblib
import json
import datetime
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.base import BaseEstimator

from .config import TrainingConfig
from .exceptions import CheckpointError

class ModelCheckpoint:
    """
    Model checkpointing utility
    
    Provides functionality to save, load, and manage model checkpoints
    during and after training.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the model checkpoint utility
        
        :param config: Training configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or TrainingConfig.from_env()
        
        # Ensure checkpoint directory exists
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def save(
        self, 
        model: BaseEstimator, 
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a model checkpoint
        
        :param model: Trained model to save
        :param score: Optional performance score
        :param metadata: Optional metadata to save with the model
        :return: Path to saved checkpoint
        """
        try:
            # Generate checkpoint filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            score_str = f"_{score:.4f}" if score is not None else ""
            filename = f"model_checkpoint_{timestamp}{score_str}.joblib"
            checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
            
            # Prepare checkpoint data
            checkpoint_data = {
                'model': model,
                'metadata': metadata or {},
                'timestamp': timestamp,
                'best_score': score
            }
            
            # Add task type to metadata if not present
            if metadata and 'task_type' not in metadata:
                checkpoint_data['metadata']['task_type'] = self.config.task_type
            
            # Save checkpoint
            joblib.dump(checkpoint_data, checkpoint_path)
            self.logger.info(f"Model checkpoint saved to {checkpoint_path}")
            
            # Clean up old checkpoints if configured
            if self.config.save_best_only:
                self.cleanup()
            
            return checkpoint_path
        
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise CheckpointError(f"Checkpoint save failed: {e}")
    
    def load(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a model checkpoint
        
        :param checkpoint_path: Path to checkpoint file
        :return: Dictionary with loaded model and metadata
        """
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            checkpoint_data = joblib.load(checkpoint_path)
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
            
            return checkpoint_data
        
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise CheckpointError(f"Checkpoint load failed: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List available checkpoints
        
        :return: List of checkpoint information
        """
        try:
            checkpoints = []
            
            for filename in os.listdir(self.config.checkpoint_dir):
                if filename.endswith('.joblib'):
                    checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
                    try:
                        # Load just the metadata without the full model
                        checkpoint_data = joblib.load(checkpoint_path)
                        checkpoints.append({
                            'filename': filename,
                            'path': checkpoint_path,
                            'timestamp': checkpoint_data.get('timestamp'),
                            'score': checkpoint_data.get('best_score'),
                            'task_type': checkpoint_data.get('metadata', {}).get('task_type')
                        })
                    except Exception as e:
                        self.logger.warning(f"Could not load checkpoint info for {filename}: {e}")
            
            # Sort by score (descending) and then timestamp (descending)
            checkpoints.sort(
                key=lambda x: (x.get('score', 0) or 0, x.get('timestamp', '')), 
                reverse=True
            )
            
            return checkpoints
        
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
            raise CheckpointError(f"Checkpoint listing failed: {e}")
    
    def cleanup(self, keep_last_n: int = None) -> None:
        """
        Clean up old checkpoints
        
        :param keep_last_n: Number of checkpoints to keep
        """
        try:
            # Use configured value if not specified
            if keep_last_n is None:
                keep_last_n = 5
            
            # List and sort checkpoints
            checkpoints = self.list_checkpoints()
            
            # Keep the best N checkpoints
            if len(checkpoints) > keep_last_n:
                for checkpoint in checkpoints[keep_last_n:]:
                    try:
                        os.remove(checkpoint['path'])
                        self.logger.info(f"Removed old checkpoint: {checkpoint['filename']}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove checkpoint {checkpoint['filename']}: {e}")
        
        except Exception as e:
            self.logger.error(f"Checkpoint cleanup failed: {e}")
            # Don't raise an exception for cleanup failures