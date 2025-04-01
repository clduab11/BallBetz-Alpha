import os
import pytest
import numpy as np
import pandas as pd
import tempfile
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml_layer.feature_engineering.preprocessor import UFLFeaturePreprocessor
from ml_layer.feature_engineering.data_integrator import UFLDataIntegrator
from ml_layer.model_selection.model_registry import ModelRegistry
from ml_layer.model_selection.model_evaluator import ModelEvaluator
from ml_layer.training.training_pipeline import TrainingPipeline
from ml_layer.prediction.predictor import UFLPredictor

@pytest.fixture
def sample_player_data():
    """
    Create sample player data for testing
    """
    return pd.DataFrame({
        'name': ['Player A', 'Player B', 'Player C', 'Player D', 'Player E'],
        'position': ['QB', 'RB', 'WR', 'QB', 'RB'],
        'team': ['Team X', 'Team Y', 'Team Z', 'Team X', 'Team Z'],
        'passing_yards': [300, 0, 0, 250, 0],
        'rushing_yards': [20, 120, 10, 15, 95],
        'receiving_yards': [0, 30, 150, 0, 20],
        'passing_touchdowns': [2, 0, 0, 1, 0],
        'rushing_touchdowns': [0, 1, 0, 0, 1],
        'receiving_touchdowns': [0, 0, 2, 0, 0],
        'games_played': [10, 10, 10, 9, 8],
        'fantasy_points': [22.0, 18.0, 27.0, 16.5, 15.5]
    })

@pytest.fixture
def temp_model_dir():
    """
    Create a temporary model directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['UFL_MODEL_DIR'] = tmpdir
        os.environ['UFL_CHECKPOINT_DIR'] = tmpdir
        yield tmpdir

class TestMLLayerIntegration:
    
    def test_end_to_end_classification_workflow(self, sample_player_data, temp_model_dir):
        """
        Test end-to-end classification workflow
        
        This test covers the entire ML pipeline from feature engineering
        to model training and prediction for a classification task.
        """
        # Set environment variables for testing
        os.environ['UFL_TASK_TYPE'] = 'classification'
        os.environ['UFL_CONFIDENCE_THRESHOLD'] = '0.0'  # Disable confidence filtering for testing
        
        # Step 1: Feature Engineering
        preprocessor = UFLFeaturePreprocessor()
        
        # Add a binary target for classification
        data_with_target = sample_player_data.copy()
        data_with_target['target'] = (data_with_target['fantasy_points'] > 20).astype(int)
        
        # Preprocess data
        X_processed, y = preprocessor.preprocess(data_with_target, target_column='target')
        
        assert X_processed is not None
        assert X_processed.shape[0] == len(data_with_target)
        assert y is not None
        
        # Step 2: Model Selection
        registry = ModelRegistry()
        model_class = registry.get_model('classification', 'random_forest')
        model = model_class(n_estimators=10, random_state=42)  # Small model for testing
        
        # Step 3: Model Evaluation
        evaluator = ModelEvaluator()
        
        # Use a small subset for evaluation to speed up testing
        X_subset = X_processed[:4]
        y_subset = y[:4]
        
        # Evaluate model (with reduced CV to speed up testing)
        os.environ['UFL_CV_FOLDS'] = '2'
        evaluation_results = evaluator.evaluate_model(
            model, X_subset, y_subset, 
            task_type='classification',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        assert 'mean_accuracy' in evaluation_results
        
        # Step 4: Model Training
        pipeline = TrainingPipeline()
        training_result = pipeline.train(model, X_processed, y)
        
        assert 'model' in training_result
        assert 'metrics' in training_result
        assert 'checkpoint_path' in training_result
        
        # Step 5: Prediction
        predictor = UFLPredictor()
        predictor.load_model('test_model', training_result['checkpoint_path'])
        
        # Generate predictions
        prediction_results = predictor.predict(X_processed)
        
        # Verify prediction results
        assert prediction_results is not None
        if isinstance(prediction_results, dict):
            assert 'predictions' in prediction_results
            assert 'confidence' in prediction_results
        
    def test_end_to_end_regression_workflow(self, sample_player_data, temp_model_dir):
        """
        Test end-to-end regression workflow
        
        This test covers the entire ML pipeline from feature engineering
        to model training and prediction for a regression task.
        """
        # Set environment variables for testing
        os.environ['UFL_TASK_TYPE'] = 'regression'
        os.environ['UFL_CONFIDENCE_THRESHOLD'] = '0.0'  # Disable confidence filtering for testing
        
        # Step 1: Feature Engineering
        preprocessor = UFLFeaturePreprocessor()
        
        # Use fantasy_points as regression target
        data_with_target = sample_player_data.copy()
        
        # Preprocess data
        X_processed, y = preprocessor.preprocess(data_with_target, target_column='fantasy_points')
        
        assert X_processed is not None
        assert X_processed.shape[0] == len(data_with_target)
        assert y is not None
        
        # Step 2: Model Selection
        registry = ModelRegistry()
        model_class = registry.get_model('regression', 'random_forest')
        model = model_class(n_estimators=10, random_state=42)  # Small model for testing
        
        # Step 3: Model Evaluation
        evaluator = ModelEvaluator()
        
        # Use a small subset for evaluation to speed up testing
        X_subset = X_processed[:4]
        y_subset = y[:4]
        
        # Evaluate model (with reduced CV to speed up testing)
        os.environ['UFL_CV_FOLDS'] = '2'
        evaluation_results = evaluator.evaluate_model(
            model, X_subset, y_subset, 
            task_type='regression',
            metrics=['mse', 'mae', 'r2']
        )
        
        assert 'mean_mse' in evaluation_results
        
        # Step 4: Model Training
        pipeline = TrainingPipeline()
        training_result = pipeline.train(model, X_processed, y)
        
        assert 'model' in training_result
        assert 'metrics' in training_result
        assert 'checkpoint_path' in training_result
        
        # Step 5: Prediction
        predictor = UFLPredictor()
        predictor.load_model('test_model', training_result['checkpoint_path'])
        
        # Generate predictions
        prediction_results = predictor.predict(X_processed)
        
        # Verify prediction results
        assert prediction_results is not None
        if isinstance(prediction_results, dict):
            assert 'predictions' in prediction_results
            assert 'confidence' in prediction_results
    
    def test_data_integrator_to_prediction_workflow(self, temp_model_dir):
        """
        Test workflow from data integration to prediction
        
        This test covers the workflow starting from data integration
        through the UFLDataIntegrator to final prediction.
        """
        # Create a mock scraper
        class MockScraper:
            def scrape_player_data(self):
                return [
                    {
                        'name': 'John Doe',
                        'team': 'Team A',
                        'position': 'QB',
                        'games_played': 10,
                        'passing_yards': 2500,
                        'rushing_yards': 250,
                        'receiving_yards': 0,
                        'passing_touchdowns': 20,
                        'rushing_touchdowns': 3,
                        'receiving_touchdowns': 0
                    },
                    {
                        'name': 'Jane Smith',
                        'team': 'Team B',
                        'position': 'RB',
                        'games_played': 12,
                        'passing_yards': 0,
                        'rushing_yards': 800,
                        'receiving_yards': 300,
                        'passing_touchdowns': 0,
                        'rushing_touchdowns': 7,
                        'receiving_touchdowns': 2
                    }
                ]
        
        # Set environment variables for testing
        os.environ['UFL_TASK_TYPE'] = 'classification'
        os.environ['UFL_CONFIDENCE_THRESHOLD'] = '0.0'  # Disable confidence filtering for testing
        
        # Step 1: Data Integration
        integrator = UFLDataIntegrator(scraper=MockScraper())
        dataset = integrator.prepare_prediction_dataset(prediction_type='win_loss')
        
        assert 'features' in dataset
        assert 'target' in dataset
        
        # Step 2: Model Training
        pipeline = TrainingPipeline()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        training_result = pipeline.train(
            model, 
            dataset['features'], 
            dataset['target']
        )
        
        assert 'model' in training_result
        assert 'checkpoint_path' in training_result
        
        # Step 3: Prediction
        predictor = UFLPredictor()
        predictor.load_model('test_model', training_result['checkpoint_path'])
        
        # Generate predictions
        prediction_results = predictor.predict(dataset['features'])
        
        # Verify prediction results
        assert prediction_results is not None