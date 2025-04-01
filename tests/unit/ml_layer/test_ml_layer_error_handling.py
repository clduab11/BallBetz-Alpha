import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from ml_layer.feature_engineering.preprocessor import UFLFeaturePreprocessor
from ml_layer.feature_engineering.data_integrator import UFLDataIntegrator
from ml_layer.feature_engineering.exceptions import DataPreprocessingError, FeatureSelectionError

from ml_layer.model_selection.model_registry import ModelRegistry
from ml_layer.model_selection.model_evaluator import ModelEvaluator

from ml_layer.training.training_pipeline import TrainingPipeline
from ml_layer.training.model_checkpoint import ModelCheckpoint
from ml_layer.training.exceptions import ModelTrainingError, DataPreparationError, CheckpointError

from ml_layer.prediction.predictor import UFLPredictor
from ml_layer.prediction.exceptions import ModelLoadError, InferenceFailed, ConfidenceThresholdError

class TestFeatureEngineeringErrorHandling:
    
    def test_preprocessor_with_empty_data(self):
        """
        Test preprocessor with empty DataFrame
        """
        preprocessor = UFLFeaturePreprocessor()
        empty_df = pd.DataFrame()
        
        with pytest.raises(DataPreprocessingError):
            preprocessor.preprocess(empty_df)
    
    def test_preprocessor_with_invalid_data_type(self):
        """
        Test preprocessor with invalid data type
        """
        preprocessor = UFLFeaturePreprocessor()
        
        with pytest.raises(DataPreprocessingError):
            preprocessor.preprocess("not a dataframe")
    
    def test_preprocessor_with_all_null_column(self):
        """
        Test preprocessor with a column containing all null values
        """
        preprocessor = UFLFeaturePreprocessor()
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'all_null': [None, None, None, None, None]
        })
        
        # This should not raise an exception as imputation should handle it
        X_transformed, _ = preprocessor.preprocess(df)
        assert X_transformed is not None
    
    def test_data_integrator_with_invalid_prediction_type(self):
        """
        Test data integrator with invalid prediction type
        """
        mock_scraper = Mock()
        mock_scraper.scrape_player_data.return_value = []
        
        integrator = UFLDataIntegrator(scraper=mock_scraper)
        
        with pytest.raises(ValueError, match="Unsupported prediction type"):
            integrator.prepare_prediction_dataset(prediction_type="invalid_type")
    
    def test_data_integrator_with_scraper_error(self):
        """
        Test data integrator when scraper raises an exception
        """
        mock_scraper = Mock()
        mock_scraper.scrape_player_data.side_effect = Exception("Scraping failed")
        
        integrator = UFLDataIntegrator(scraper=mock_scraper)
        
        with pytest.raises(DataPreprocessingError):
            integrator.prepare_prediction_dataset()

class TestModelSelectionErrorHandling:
    
    def test_model_registry_with_invalid_task_type(self):
        """
        Test model registry with invalid task type
        """
        registry = ModelRegistry()
        
        with pytest.raises(ValueError, match="Unsupported task type"):
            registry.get_model("invalid_task")
    
    def test_model_registry_with_invalid_model_name(self):
        """
        Test model registry with invalid model name
        """
        registry = ModelRegistry()
        
        with pytest.raises(ValueError, match="Model .* not found"):
            registry.get_model("classification", "invalid_model")
    
    def test_model_evaluator_with_invalid_task_type(self):
        """
        Test model evaluator with invalid task type
        """
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError, match="Unsupported task type"):
            evaluator.evaluate_model(
                Mock(), 
                np.array([[1, 2], [3, 4]]), 
                np.array([0, 1]), 
                task_type="invalid_task"
            )
    
    def test_model_evaluator_with_invalid_metrics(self):
        """
        Test model evaluator with invalid metrics
        """
        evaluator = ModelEvaluator()
        
        with pytest.raises(ValueError, match="Invalid metrics"):
            evaluator.evaluate_model(
                Mock(), 
                np.array([[1, 2], [3, 4]]), 
                np.array([0, 1]), 
                task_type="classification",
                metrics=["invalid_metric"]
            )

class TestTrainingErrorHandling:
    
    def test_training_pipeline_with_invalid_data(self):
        """
        Test training pipeline with invalid data
        """
        pipeline = TrainingPipeline()
        
        with pytest.raises(DataPreparationError):
            pipeline.train(Mock(), None, None)
    
    def test_training_pipeline_with_incompatible_model(self):
        """
        Test training pipeline with incompatible model
        """
        # Create a model that will fail during training
        class FailingModel:
            def fit(self, X, y):
                raise ValueError("Model training failed")
        
        pipeline = TrainingPipeline()
        
        with pytest.raises(ModelTrainingError):
            pipeline.train(
                FailingModel(), 
                np.array([[1, 2], [3, 4]]), 
                np.array([0, 1])
            )
    
    def test_model_checkpoint_with_invalid_path(self):
        """
        Test model checkpoint with invalid path
        """
        checkpoint = ModelCheckpoint()
        
        with pytest.raises(CheckpointError):
            checkpoint.load("non_existent_path.joblib")

class TestPredictionErrorHandling:
    
    def test_predictor_without_model(self):
        """
        Test predictor without loading a model
        """
        predictor = UFLPredictor()
        
        with pytest.raises(ModelLoadError):
            predictor.predict(np.array([[1, 2], [3, 4]]))
    
    def test_predictor_with_invalid_checkpoint(self):
        """
        Test predictor with invalid checkpoint path
        """
        predictor = UFLPredictor()
        
        with pytest.raises(ModelLoadError):
            predictor.load_model("test_model", "non_existent_path.joblib")
    
    @patch('ml_layer.prediction.predictor.UFLPredictor._apply_confidence_threshold')
    def test_predictor_with_confidence_threshold_error(self, mock_apply_threshold):
        """
        Test predictor when no predictions meet confidence threshold
        """
        mock_apply_threshold.side_effect = ConfidenceThresholdError("No predictions meet threshold")
        
        predictor = UFLPredictor()
        # Mock the _models dictionary to avoid ModelLoadError
        predictor._models = {"test_model": Mock()}
        
        with pytest.raises(InferenceFailed):
            predictor.predict(np.array([[1, 2], [3, 4]]), model_name="test_model")
    
    @patch('ml_layer.prediction.predictor.UFLPredictor._preprocess_data')
    def test_predictor_with_preprocessing_error(self, mock_preprocess):
        """
        Test predictor when preprocessing fails
        """
        mock_preprocess.side_effect = Exception("Preprocessing failed")
        
        predictor = UFLPredictor()
        # Mock the _models dictionary to avoid ModelLoadError
        predictor._models = {"test_model": Mock()}
        
        with pytest.raises(InferenceFailed):
            predictor.predict(np.array([[1, 2], [3, 4]]), model_name="test_model")