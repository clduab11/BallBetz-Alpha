import os
import json
import pytest
import numpy as np
import tempfile
import joblib
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from ml_layer.prediction.predictor import UFLPredictor
from ml_layer.prediction.config import PredictionConfig
from ml_layer.prediction.exceptions import (
    ModelLoadError, 
    InferenceFailed, 
    ConfidenceThresholdError
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
def temp_model_dir():
    """Create a temporary model directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['UFL_MODEL_DIR'] = tmpdir
        yield tmpdir

def save_model_checkpoint(model, directory, name='model'):
    """
    Save a model checkpoint for testing
    
    :param model: Trained model
    :param directory: Checkpoint directory
    :param name: Model name
    :return: Path to saved checkpoint
    """
    checkpoint_data = {
        'model': model,
        'metadata': {'task_type': 'classification'}
    }
    checkpoint_path = os.path.join(directory, f'{name}_checkpoint.joblib')
    joblib.dump(checkpoint_data, checkpoint_path)
    return checkpoint_path

class TestPredictionConfig:
    def test_default_configuration(self):
        """Test default prediction configuration"""
        config = PredictionConfig.from_env()
        
        assert config.task_type == 'classification'
        assert config.confidence_threshold == 0.5
        assert config.output_format == 'json'
        assert config.log_predictions is True
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        config = PredictionConfig.from_env()
        config.validate()
        
        # Test invalid task type
        with pytest.raises(ValueError, match="Unsupported prediction task"):
            invalid_config = PredictionConfig(task_type='invalid_task')
            invalid_config.validate()
        
        # Test invalid confidence threshold
        with pytest.raises(ValueError, match="Invalid confidence threshold"):
            invalid_config = PredictionConfig(confidence_threshold=1.5)
            invalid_config.validate()
        
        # Test invalid output format
        with pytest.raises(ValueError, match="Unsupported output format"):
            invalid_config = PredictionConfig(output_format='invalid_format')
            invalid_config.validate()

class TestUFLPredictor:
    def test_model_loading(self, classification_data, temp_model_dir):
        """Test model loading from checkpoint"""
        X, y = classification_data
        
        # Train and save models
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model1.fit(X, y)
        checkpoint1 = save_model_checkpoint(model1, temp_model_dir, 'model1')
        
        model2 = LogisticRegression(random_state=42)
        model2.fit(X, y)
        checkpoint2 = save_model_checkpoint(model2, temp_model_dir, 'model2')
        
        # Initialize predictor
        predictor = UFLPredictor()
        
        # Load models
        loaded_model1 = predictor.load_model('model1', checkpoint1)
        loaded_model2 = predictor.load_model('model2', checkpoint2)
        
        assert loaded_model1 is not None
        assert loaded_model2 is not None
        assert len(predictor._models) == 2
    
    def test_prediction_generation(self, classification_data, temp_model_dir):
        """Test prediction generation"""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train and save model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        checkpoint = save_model_checkpoint(model, temp_model_dir)
        
        # Initialize predictor
        predictor = UFLPredictor()
        predictor.load_model('test_model', checkpoint)
        
        # Generate predictions
        results = predictor.predict(X_test)
        
        # Parse results (assuming JSON output)
        parsed_results = json.loads(results)
        
        assert 'predictions' in parsed_results
        assert 'confidence' in parsed_results
        assert len(parsed_results['predictions']) > 0
        assert len(parsed_results['confidence']) > 0
    
    def test_ensemble_prediction(self, classification_data, temp_model_dir):
        """Test ensemble prediction"""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train and save multiple models
        os.environ['UFL_USE_ENSEMBLE'] = 'true'
        os.environ['UFL_MAX_ENSEMBLE_MODELS'] = '3'
        
        models = [
            RandomForestClassifier(n_estimators=10, random_state=i) 
            for i in range(3)
        ]
        
        checkpoints = []
        for i, model in enumerate(models):
            model.fit(X_train, y_train)
            checkpoint = save_model_checkpoint(model, temp_model_dir, f'model_{i}')
            checkpoints.append(checkpoint)
        
        # Initialize predictor
        predictor = UFLPredictor()
        for i, checkpoint in enumerate(checkpoints):
            predictor.load_model(f'model_{i}', checkpoint)
        
        # Generate predictions
        results = predictor.predict(X_test)
        parsed_results = json.loads(results)
        
        assert 'predictions' in parsed_results
        assert 'confidence' in parsed_results
    
    def test_confidence_threshold(self, classification_data, temp_model_dir):
        """Test confidence threshold filtering"""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train and save model
        os.environ['UFL_CONFIDENCE_THRESHOLD'] = '0.9'  # High threshold
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        checkpoint = save_model_checkpoint(model, temp_model_dir)
        
        # Initialize predictor
        predictor = UFLPredictor()
        predictor.load_model('test_model', checkpoint)
        
        # Attempt prediction with high confidence threshold
        with pytest.raises(ConfidenceThresholdError):
            predictor.predict(X_test)
    
    def test_error_handling(self, classification_data):
        """Test error handling in prediction"""
        X, y = classification_data
        
        # Initialize predictor without loading a model
        predictor = UFLPredictor()
        
        # Attempt prediction without a model
        with pytest.raises(ModelLoadError):
            predictor.predict(X)
        
        # Attempt to load non-existent checkpoint
        with pytest.raises(ModelLoadError):
            predictor.load_model('invalid_model', 'non_existent_path.joblib')