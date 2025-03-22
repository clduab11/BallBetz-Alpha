import pytest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from models.model_interface import ModelInterface

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set a test-specific FLASK_SECRET_KEY
os.environ['FLASK_SECRET_KEY'] = 'test-secret-key'

@pytest.fixture
def test_app():
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_player_data():
    data = {
        'name': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5'],
        'team': ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'],
        'position': ['QB', 'RB', 'WR', 'TE', 'K'],
        'passing_yards': [250, 100, 150, 50, 0],
        'rushing_yards': [50, 75, 25, 100, 0],
        'receiving_yards': [10, 20, 30, 40, 0],
        'fantasy_points': [20, 30, 15, 25, 5]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_scraped_data():
    return [
        {'name': 'Player 1', 'team': 'Team A', 'position': 'QB', 'passing_yards': '250', 'rushing_yards': '50'},
        {'name': 'Player 2', 'team': 'Team B', 'position': 'RB', 'passing_yards': '100', 'rushing_yards': '75'},
        {'name': 'Player 3', 'team': 'Team C', 'position': 'WR', 'passing_yards': '150', 'rushing_yards': '25'},
        {'name': 'Player 4', 'team': 'Team D', 'position': 'TE', 'passing_yards': '50', 'rushing_yards': '100'},
        {'name': 'Player 5', 'team': 'Team E', 'position': 'K', 'passing_yards': '0', 'rushing_yards': '0'}
    ]

@pytest.fixture
def mock_sklearn_model():
    mock = MagicMock()
    mock.predict.return_value = pd.DataFrame({'predicted_points': [1.0, 2.0, 3.0, 4.0, 5.0]})
    return mock