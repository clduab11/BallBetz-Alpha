import pytest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import jwt
from datetime import datetime, timedelta
from supabase import create_client, Client

from models.model_interface import ModelInterface

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set a test-specific FLASK_SECRET_KEY
os.environ['FLASK_SECRET_KEY'] = 'test-secret-key'

# Mock Supabase credentials for testing
os.environ['SUPABASE_URL'] = 'http://localhost:54321'
os.environ['SUPABASE_KEY'] = 'test-supabase-key'

# ML Layer Test Configuration
os.environ['UFL_NUMERIC_IMPUTATION'] = 'mean'
os.environ['UFL_CATEGORICAL_IMPUTATION'] = 'most_frequent'
os.environ['UFL_SCALE_NUMERIC'] = 'true'
os.environ['UFL_MAX_FEATURES'] = '10'
os.environ['UFL_VERBOSE_PREPROCESSING'] = 'false'

# Existing fixtures remain unchanged...
# (Previous fixtures from the original conftest.py)

@pytest.fixture
def mock_ml_feature_data():
    """
    Generate mock data for ML feature engineering tests
    """
    return [
        {
            'name': 'John Quarterback',
            'team': 'Team Alpha',
            'position': 'QB',
            'games_played': 12,
            'passing_yards': 3500,
            'rushing_yards': 250,
            'receiving_yards': 0,
            'passing_touchdowns': 28,
            'rushing_touchdowns': 3,
            'receiving_touchdowns': 0,
            'interceptions': 8
        },
        {
            'name': 'Sarah Runningback',
            'team': 'Team Beta',
            'position': 'RB',
            'games_played': 14,
            'passing_yards': 0,
            'rushing_yards': 1200,
            'receiving_yards': 350,
            'passing_touchdowns': 0,
            'rushing_touchdowns': 10,
            'receiving_touchdowns': 2,
            'interceptions': 0
        },
        {
            'name': 'Mike Widereceiver',
            'team': 'Team Gamma',
            'position': 'WR',
            'games_played': 15,
            'passing_yards': 0,
            'rushing_yards': 50,
            'receiving_yards': 1100,
            'passing_touchdowns': 0,
            'rushing_touchdowns': 0,
            'receiving_touchdowns': 8,
            'interceptions': 0
        }
    ]

@pytest.fixture
def mock_ml_config():
    """
    Create a mock ML configuration for testing
    """
    from ml_layer.feature_engineering.config import FeatureEngineeringConfig
    return FeatureEngineeringConfig.from_env()

@pytest.fixture
def mock_ml_scraper(mock_ml_feature_data):
    """
    Create a mock scraper for ML Layer tests
    """
    class MockMLScraper:
        def scrape_player_data(self):
            return mock_ml_feature_data
    
    return MockMLScraper()