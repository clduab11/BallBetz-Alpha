import os
import pytest
import tempfile
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import patch, MagicMock

from ml_layer.training.model_checkpoint import ModelCheckpoint
from ml_layer.training.config import TrainingConfig
from ml_layer.training.exceptions import CheckpointError

@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary checkpoint directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['UFL_CHECKPOINT_DIR'] = tmpdir
        yield tmpdir

@pytest.fixture
def sample_model():
    """Create a sample model for testing"""
    return RandomForestClassifier(n_estimators=10, random_state=42)

@pytest.fixture
def trained_model(sample_model):
    """Create a trained model for testing"""
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    sample_model.fit(X, y)
    return sample_model

class TestModelCheckpoint:
    
    def test_initialization(self, temp_checkpoint_dir):
        """Test initialization of ModelCheckpoint"""
        checkpoint = ModelCheckpoint()
        assert checkpoint is not None
        assert checkpoint.config is not None
        assert os.path.exists(checkpoint.config.checkpoint_dir)
    
    def test_initialization_with_custom_config(self, temp_checkpoint_dir):
        """Test initialization with custom config"""
        config = TrainingConfig(checkpoint_dir=temp_checkpoint_dir)
        checkpoint = ModelCheckpoint(config)
        
        assert checkpoint.config == config
        assert checkpoint.config.checkpoint_dir == temp_checkpoint_dir
    
    def test_save_checkpoint(self, temp_checkpoint_dir, trained_model):
        """Test saving a model checkpoint"""
        checkpoint = ModelCheckpoint()
        
        # Save checkpoint
        checkpoint_path = checkpoint.save(trained_model, score=0.85)
        
        # Verify checkpoint was saved
        assert os.path.exists(checkpoint_path)
        assert checkpoint_path.endswith('.joblib')
        assert '0.8500' in checkpoint_path  # Score should be in filename
    
    def test_save_checkpoint_with_metadata(self, temp_checkpoint_dir, trained_model):
        """Test saving a model checkpoint with metadata"""
        checkpoint = ModelCheckpoint()
        
        # Create metadata
        metadata = {
            'task_type': 'classification',
            'features': ['feature1', 'feature2'],
            'training_time': 10.5
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint.save(trained_model, score=0.9, metadata=metadata)
        
        # Load checkpoint to verify metadata
        checkpoint_data = joblib.load(checkpoint_path)
        
        assert 'metadata' in checkpoint_data
        assert checkpoint_data['metadata'] == metadata
    
    def test_load_checkpoint(self, temp_checkpoint_dir, trained_model):
        """Test loading a model checkpoint"""
        checkpoint = ModelCheckpoint()
        
        # Save checkpoint
        checkpoint_path = checkpoint.save(trained_model, score=0.85)
        
        # Load checkpoint
        loaded_data = checkpoint.load(checkpoint_path)
        
        assert 'model' in loaded_data
        assert 'metadata' in loaded_data
        assert 'timestamp' in loaded_data
        assert 'best_score' in loaded_data
        assert loaded_data['best_score'] == 0.85
        
        # Verify model is the same
        loaded_model = loaded_data['model']
        assert isinstance(loaded_model, RandomForestClassifier)
        assert loaded_model.n_estimators == trained_model.n_estimators
    
    def test_load_nonexistent_checkpoint(self, temp_checkpoint_dir):
        """Test loading a non-existent checkpoint"""
        checkpoint = ModelCheckpoint()
        
        with pytest.raises(CheckpointError):
            checkpoint.load('nonexistent_checkpoint.joblib')
    
    def test_list_checkpoints(self, temp_checkpoint_dir, trained_model):
        """Test listing available checkpoints"""
        checkpoint = ModelCheckpoint()
        
        # Save multiple checkpoints
        checkpoint_paths = []
        for i in range(3):
            path = checkpoint.save(trained_model, score=0.8 + 0.05 * i)
            checkpoint_paths.append(path)
        
        # List checkpoints
        checkpoints = checkpoint.list_checkpoints()
        
        assert len(checkpoints) == 3
        assert all('filename' in c for c in checkpoints)
        assert all('path' in c for c in checkpoints)
        assert all('score' in c for c in checkpoints)
        
        # Verify checkpoints are sorted by score (descending)
        scores = [c['score'] for c in checkpoints]
        assert scores == sorted(scores, reverse=True)
    
    def test_cleanup(self, temp_checkpoint_dir, trained_model):
        """Test checkpoint cleanup"""
        checkpoint = ModelCheckpoint()
        
        # Save multiple checkpoints
        for i in range(10):
            checkpoint.save(trained_model, score=0.8 + 0.01 * i)
        
        # Verify 10 checkpoints exist
        checkpoints_before = checkpoint.list_checkpoints()
        assert len(checkpoints_before) == 10
        
        # Cleanup, keeping last 5
        checkpoint.cleanup(keep_last_n=5)
        
        # Verify only 5 checkpoints remain
        checkpoints_after = checkpoint.list_checkpoints()
        assert len(checkpoints_after) == 5
        
        # Verify the best 5 checkpoints were kept
        scores_after = [c['score'] for c in checkpoints_after]
        assert min(scores_after) >= 0.85  # 0.8 + 0.01 * 5
    
    def test_cleanup_with_default_keep_last_n(self, temp_checkpoint_dir, trained_model):
        """Test checkpoint cleanup with default keep_last_n"""
        checkpoint = ModelCheckpoint()
        
        # Save multiple checkpoints
        for i in range(10):
            checkpoint.save(trained_model, score=0.8 + 0.01 * i)
        
        # Cleanup with default keep_last_n
        checkpoint.cleanup()
        
        # Verify only 5 checkpoints remain (default)
        checkpoints_after = checkpoint.list_checkpoints()
        assert len(checkpoints_after) == 5
    
    @patch('os.remove')
    def test_cleanup_error_handling(self, mock_remove, temp_checkpoint_dir, trained_model):
        """Test error handling during cleanup"""
        checkpoint = ModelCheckpoint()
        
        # Save multiple checkpoints
        for i in range(3):
            checkpoint.save(trained_model, score=0.8 + 0.05 * i)
        
        # Make os.remove raise an exception
        mock_remove.side_effect = Exception("Failed to remove file")
        
        # Cleanup should not raise an exception
        checkpoint.cleanup(keep_last_n=1)
        
        # Verify os.remove was called
        assert mock_remove.called
    
    @patch('joblib.load')
    def test_list_checkpoints_error_handling(self, mock_load, temp_checkpoint_dir, trained_model):
        """Test error handling when listing checkpoints"""
        checkpoint = ModelCheckpoint()
        
        # Save a checkpoint
        checkpoint.save(trained_model, score=0.85)
        
        # Make joblib.load raise an exception
        mock_load.side_effect = Exception("Failed to load checkpoint")
        
        # list_checkpoints should handle the exception
        checkpoints = checkpoint.list_checkpoints()
        
        # Verify joblib.load was called
        assert mock_load.called
        
        # No checkpoints should be returned due to the error
        assert len(checkpoints) == 0