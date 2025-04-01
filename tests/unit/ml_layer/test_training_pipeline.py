import os
import pytest
import numpy as np
import tempfile
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_layer.training.config import TrainingConfig
from ml_layer.training.training_pipeline import TrainingPipeline
from ml_layer.training.model_checkpoint import ModelCheckpoint
from ml_layer.training.exceptions import (
    ModelTrainingError, 
    DataPreparationError, 
    CheckpointError
)

@pytest.fixture
def classification_data():
    """Generate synthetic classification dataset"""
    X, y = make_classification(
        n_samples=200, 
        n_features=10, 
        n_classes=2, 
        random_state=42
    )
    return X, y

@pytest.fixture
def regression_data():
    """Generate synthetic regression dataset"""
    X, y = make_regression(
        n_samples=200, 
        n_features=10, 
        noise=0.1, 
        random_state=42
    )
    return X, y

@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary checkpoint directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['UFL_CHECKPOINT_DIR'] = tmpdir
        yield tmpdir

class TestTrainingConfig:
    def test_default_configuration(self):
        """Test default training configuration"""
        config = TrainingConfig.from_env()
        
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 0.01
        assert config.early_stopping is True
        assert config.task_type == 'classification'
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        config = TrainingConfig.from_env()
        config.validate()
        
        # Test invalid configurations
        with pytest.raises(ValueError, match="Invalid epochs"):
            invalid_config = TrainingConfig(epochs=0)
            invalid_config.validate()
        
        with pytest.raises(ValueError, match="Invalid batch size"):
            invalid_config = TrainingConfig(batch_size=0)
            invalid_config.validate()
        
        with pytest.raises(ValueError, match="Unsupported task type"):
            invalid_config = TrainingConfig(task_type='invalid_task')
            invalid_config.validate()

class TestTrainingPipeline:
    def test_classification_training(self, classification_data, temp_checkpoint_dir):
        """Test training a classification model"""
        X, y = classification_data
        
        # Configure for classification
        os.environ['UFL_TASK_TYPE'] = 'classification'
        config = TrainingConfig.from_env()
        
        # Initialize pipeline
        pipeline = TrainingPipeline(config)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = pipeline.train(model, X, y)
        
        assert 'model' in result
        assert 'metrics' in result
        assert 'checkpoint_path' in result
        
        # Check classification metrics
        metrics = result['metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
    
    def test_regression_training(self, regression_data, temp_checkpoint_dir):
        """Test training a regression model"""
        X, y = regression_data
        
        # Configure for regression
        os.environ['UFL_TASK_TYPE'] = 'regression'
        config = TrainingConfig.from_env()
        
        # Initialize pipeline
        pipeline = TrainingPipeline(config)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        result = pipeline.train(model, X, y)
        
        assert 'model' in result
        assert 'metrics' in result
        assert 'checkpoint_path' in result
        
        # Check regression metrics
        metrics = result['metrics']
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
    
    def test_prediction(self, classification_data, temp_checkpoint_dir):
        """Test model prediction"""
        X, y = classification_data
        
        # Configure for classification
        os.environ['UFL_TASK_TYPE'] = 'classification'
        config = TrainingConfig.from_env()
        
        # Initialize pipeline
        pipeline = TrainingPipeline(config)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = pipeline.train(model, X, y)
        
        # Generate predictions
        X_new = X[:10]  # Use first 10 samples for prediction
        predictions = pipeline.predict(result['model'], X_new)
        
        assert predictions.shape[0] == X_new.shape[0]
    
    def test_error_handling(self, classification_data):
        """Test error handling in training pipeline"""
        X, y = classification_data
        
        # Configure for classification
        os.environ['UFL_TASK_TYPE'] = 'classification'
        config = TrainingConfig.from_env()
        
        # Initialize pipeline
        pipeline = TrainingPipeline(config)
        
        # Test with invalid data
        with pytest.raises(DataPreparationError):
            pipeline.train(LogisticRegression(), None, None)
        
        # Test with incompatible model and task
        with pytest.raises(ModelTrainingError):
            # Use a regression model for a classification task
            pipeline.train(LinearRegression(), X, y)

class TestModelCheckpoint:
    def test_checkpoint_save_and_load(self, classification_data, temp_checkpoint_dir):
        """Test model checkpointing"""
        X, y = classification_data
        
        # Configure for classification
        os.environ['UFL_TASK_TYPE'] = 'classification'
        config = TrainingConfig.from_env()
        
        # Initialize checkpoint
        checkpoint = ModelCheckpoint(config)
        
        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Save checkpoint
        checkpoint_path = checkpoint.save(model, score=0.85)
        assert os.path.exists(checkpoint_path)
        
        # Load checkpoint
        loaded_checkpoint = checkpoint.load(checkpoint_path)
        assert 'model' in loaded_checkpoint
        assert 'metadata' in loaded_checkpoint
        assert 'best_score' in loaded_checkpoint
    
    def test_checkpoint_cleanup(self, classification_data, temp_checkpoint_dir):
        """Test checkpoint cleanup"""
        X, y = classification_data
        
        # Configure for classification
        os.environ['UFL_TASK_TYPE'] = 'classification'
        config = TrainingConfig.from_env()
        
        # Initialize checkpoint
        checkpoint = ModelCheckpoint(config)
        
        # Save multiple checkpoints
        for i in range(10):
            model = RandomForestClassifier(n_estimators=10, random_state=i)
            model.fit(X, y)
            checkpoint.save(model, score=0.8 + 0.01 * i)
        
        # Cleanup, keeping last 5 checkpoints
        checkpoint.cleanup(keep_last_n=5)
        
        # Check number of remaining checkpoints
        checkpoints = [f for f in os.listdir(config.checkpoint_dir) if f.endswith('.joblib')]
        assert len(checkpoints) <= 5