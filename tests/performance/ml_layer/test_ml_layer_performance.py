import os
import time
import pytest
import numpy as np
import pandas as pd
import tempfile
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_layer.feature_engineering.preprocessor import UFLFeaturePreprocessor
from ml_layer.model_selection.model_registry import ModelRegistry
from ml_layer.training.training_pipeline import TrainingPipeline
from ml_layer.prediction.predictor import UFLPredictor

@pytest.fixture
def large_classification_dataset():
    """
    Generate a large synthetic classification dataset
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    return X, y

@pytest.fixture
def large_regression_dataset():
    """
    Generate a large synthetic regression dataset
    """
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    return X, y

@pytest.fixture
def temp_model_dir():
    """
    Create a temporary model directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['UFL_MODEL_DIR'] = tmpdir
        os.environ['UFL_CHECKPOINT_DIR'] = tmpdir
        yield tmpdir

class TestMLLayerPerformance:
    
    def test_feature_preprocessing_performance(self, large_classification_dataset):
        """
        Test performance of feature preprocessing
        """
        X, y = large_classification_dataset
        
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        
        # Measure preprocessing time
        preprocessor = UFLFeaturePreprocessor()
        
        start_time = time.time()
        X_processed, y_processed = preprocessor.preprocess(df, target_column='target')
        preprocessing_time = time.time() - start_time
        
        # Log performance metrics
        print(f"\nFeature preprocessing time: {preprocessing_time:.4f} seconds")
        print(f"Input shape: {df.shape}, Output shape: {X_processed.shape}")
        
        # Assert reasonable performance (adjust thresholds as needed)
        assert preprocessing_time < 5.0, "Feature preprocessing took too long"
    
    def test_model_training_performance_classification(self, large_classification_dataset, temp_model_dir):
        """
        Test performance of model training for classification
        """
        X, y = large_classification_dataset
        
        # Configure for classification
        os.environ['UFL_TASK_TYPE'] = 'classification'
        
        # Initialize pipeline
        pipeline = TrainingPipeline()
        
        # Train with different dataset sizes to measure scaling
        dataset_sizes = [100, 500, 1000]
        training_times = []
        
        for size in dataset_sizes:
            X_subset = X[:size]
            y_subset = y[:size]
            
            # Create a small model for testing
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            
            # Measure training time
            start_time = time.time()
            result = pipeline.train(model, X_subset, y_subset)
            training_time = time.time() - start_time
            
            training_times.append(training_time)
            
            # Log performance metrics
            print(f"\nTraining time for {size} samples: {training_time:.4f} seconds")
            print(f"Training metrics: {result['metrics']}")
        
        # Assert reasonable scaling (should be roughly linear or better)
        # This is a simple check - adjust as needed
        assert training_times[1] < training_times[0] * 10, "Training time scaling is worse than expected"
        assert training_times[2] < training_times[1] * 5, "Training time scaling is worse than expected"
    
    def test_model_training_performance_regression(self, large_regression_dataset, temp_model_dir):
        """
        Test performance of model training for regression
        """
        X, y = large_regression_dataset
        
        # Configure for regression
        os.environ['UFL_TASK_TYPE'] = 'regression'
        
        # Initialize pipeline
        pipeline = TrainingPipeline()
        
        # Train with different dataset sizes to measure scaling
        dataset_sizes = [100, 500, 1000]
        training_times = []
        
        for size in dataset_sizes:
            X_subset = X[:size]
            y_subset = y[:size]
            
            # Create a small model for testing
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            
            # Measure training time
            start_time = time.time()
            result = pipeline.train(model, X_subset, y_subset)
            training_time = time.time() - start_time
            
            training_times.append(training_time)
            
            # Log performance metrics
            print(f"\nTraining time for {size} samples: {training_time:.4f} seconds")
            print(f"Training metrics: {result['metrics']}")
        
        # Assert reasonable scaling (should be roughly linear or better)
        # This is a simple check - adjust as needed
        assert training_times[1] < training_times[0] * 10, "Training time scaling is worse than expected"
        assert training_times[2] < training_times[1] * 5, "Training time scaling is worse than expected"
    
    def test_prediction_performance(self, large_classification_dataset, temp_model_dir):
        """
        Test performance of prediction generation
        """
        X, y = large_classification_dataset
        
        # Configure for classification
        os.environ['UFL_TASK_TYPE'] = 'classification'
        os.environ['UFL_CONFIDENCE_THRESHOLD'] = '0.0'  # Disable confidence filtering for testing
        
        # Train a model first
        pipeline = TrainingPipeline()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = pipeline.train(model, X, y)
        
        # Initialize predictor
        predictor = UFLPredictor()
        predictor.load_model('test_model', result['checkpoint_path'])
        
        # Test prediction with different batch sizes
        batch_sizes = [1, 10, 100, 1000]
        prediction_times = []
        
        for size in batch_sizes:
            X_batch = X[:size]
            
            # Measure prediction time
            start_time = time.time()
            predictions = predictor.predict(X_batch)
            prediction_time = time.time() - start_time
            
            prediction_times.append(prediction_time)
            
            # Log performance metrics
            print(f"\nPrediction time for {size} samples: {prediction_time:.4f} seconds")
            print(f"Prediction throughput: {size / prediction_time:.2f} samples/second")
        
        # Assert reasonable scaling (should be roughly linear or better)
        # This is a simple check - adjust as needed
        assert prediction_times[3] < prediction_times[0] * 1000, "Prediction time scaling is worse than expected"
    
    def test_end_to_end_performance(self, large_classification_dataset, temp_model_dir):
        """
        Test end-to-end performance from preprocessing to prediction
        """
        X, y = large_classification_dataset
        
        # Configure for classification
        os.environ['UFL_TASK_TYPE'] = 'classification'
        os.environ['UFL_CONFIDENCE_THRESHOLD'] = '0.0'  # Disable confidence filtering for testing
        
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        
        # Measure total processing time
        start_time = time.time()
        
        # Step 1: Feature Engineering
        preprocessor = UFLFeaturePreprocessor()
        X_processed, y_processed = preprocessor.preprocess(df, target_column='target')
        
        # Step 2: Model Selection
        registry = ModelRegistry()
        model_class = registry.get_model('classification', 'random_forest')
        model = model_class(n_estimators=10, random_state=42)
        
        # Step 3: Model Training
        pipeline = TrainingPipeline()
        training_result = pipeline.train(model, X_processed, y_processed)
        
        # Step 4: Prediction
        predictor = UFLPredictor()
        predictor.load_model('test_model', training_result['checkpoint_path'])
        predictions = predictor.predict(X_processed)
        
        total_time = time.time() - start_time
        
        # Log performance metrics
        print(f"\nEnd-to-end processing time: {total_time:.4f} seconds")
        print(f"Dataset size: {len(df)} samples, {df.shape[1]} features")
        
        # Assert reasonable performance (adjust threshold as needed)
        assert total_time < 30.0, "End-to-end processing took too long"