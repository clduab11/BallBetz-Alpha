import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from ml_layer.feature_engineering.data_integrator import UFLDataIntegrator
from ml_layer.feature_engineering.config import FeatureEngineeringConfig
from ml_layer.feature_engineering.exceptions import DataPreprocessingError

class MockUFLScraper:
    """
    Mock scraper for testing feature engineering module
    """
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

@pytest.fixture
def mock_config():
    """
    Create a mock configuration for testing
    """
    return FeatureEngineeringConfig(
        numeric_imputation_strategy='mean',
        max_features_to_select=10,
        scale_numeric_features=True,
        verbose=False
    )

@pytest.fixture
def data_integrator(mock_config):
    """
    Create a UFLDataIntegrator with mock scraper
    """
    mock_scraper = MockUFLScraper()
    return UFLDataIntegrator(config=mock_config, scraper=mock_scraper)

def test_data_integrator_initialization(data_integrator):
    """
    Test initialization of UFLDataIntegrator
    """
    assert data_integrator is not None
    assert data_integrator.config is not None
    assert data_integrator.scraper is not None

def test_preprocess_player_data(data_integrator):
    """
    Test player data preprocessing
    """
    raw_data = data_integrator.scraper.scrape_player_data()
    preprocessed_df = data_integrator._preprocess_player_data(raw_data)
    
    assert not preprocessed_df.empty
    assert 'team_encoded' in preprocessed_df.columns
    assert 'position_encoded' in preprocessed_df.columns
    assert 'total_touchdowns' in preprocessed_df.columns
    assert 'yards_per_game' in preprocessed_df.columns

def test_select_features(data_integrator):
    """
    Test feature selection
    """
    raw_data = data_integrator.scraper.scrape_player_data()
    preprocessed_df = data_integrator._preprocess_player_data(raw_data)
    
    # Create a mock target variable
    preprocessed_df['target'] = (preprocessed_df['total_touchdowns'] > 5).astype(int)
    
    selected_features = data_integrator._select_features(
        preprocessed_df.drop('target', axis=1), 
        preprocessed_df['target']
    )
    
    assert selected_features is not None
    assert selected_features.shape[1] <= data_integrator.config.max_features_to_select

def test_prepare_prediction_dataset(data_integrator):
    """
    Test prediction dataset preparation
    """
    # Test win/loss prediction
    win_loss_dataset = data_integrator.prepare_prediction_dataset(prediction_type='win_loss')
    
    assert 'features' in win_loss_dataset
    assert 'target' in win_loss_dataset
    assert 'feature_names' in win_loss_dataset
    assert 'team_mapping' in win_loss_dataset
    assert 'position_mapping' in win_loss_dataset
    
    assert win_loss_dataset['features'].ndim == 2
    assert win_loss_dataset['target'].ndim == 1
    
    # Test point spread prediction
    point_spread_dataset = data_integrator.prepare_prediction_dataset(prediction_type='point_spread')
    
    assert 'features' in point_spread_dataset
    assert 'target' in point_spread_dataset

def test_invalid_prediction_type(data_integrator):
    """
    Test handling of invalid prediction type
    """
    with pytest.raises(ValueError, match="Unsupported prediction type"):
        data_integrator.prepare_prediction_dataset(prediction_type='invalid_type')

def test_error_handling(mock_config):
    """
    Test error handling in data preprocessing
    """
    # Create a scraper that will raise an exception
    class FailingScraper:
        def scrape_player_data(self):
            raise Exception("Scraping failed")
    
    failing_integrator = UFLDataIntegrator(config=mock_config, scraper=FailingScraper())
    
    with pytest.raises(DataPreprocessingError):
        failing_integrator.prepare_prediction_dataset()