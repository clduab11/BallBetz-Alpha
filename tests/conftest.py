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

@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    mock_client = MagicMock(spec=Client)
    
    # Mock auth responses
    mock_auth = MagicMock()
    mock_auth.sign_up.return_value = {'user': {'id': 'test-user-id'}, 'session': {'access_token': 'test-token'}}
    mock_auth.sign_in.return_value = {'session': {'access_token': 'test-token'}}
    mock_client.auth = mock_auth
    
    # Mock database responses
    mock_client.table().select().execute.return_value = {'data': []}
    
    return mock_client

@pytest.fixture
def mock_jwt_token():
    """Generate a mock JWT token for testing."""
    payload = {
        'sub': 'test-user-id',
        'email': 'test@example.com',
        'role': 'authenticated',
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, 'test-secret', algorithm='HS256')

@pytest.fixture
def mock_invalid_jwt_token():
    """Generate an expired JWT token for testing."""
    payload = {
        'sub': 'test-user-id',
        'email': 'test@example.com',
        'role': 'authenticated',
        'exp': datetime.utcnow() - timedelta(hours=1)
    }
    return jwt.encode(payload, 'test-secret', algorithm='HS256')

@pytest.fixture
def test_app(mock_supabase):
    from app import app
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    
    # Override Supabase client with mock
    app.supabase_client = mock_supabase
    
    with app.test_client() as client:
        yield client

@pytest.fixture
def auth_headers(mock_jwt_token):
    """Generate test authentication headers."""
    return {
        'Authorization': f'Bearer {mock_jwt_token}',
        'Content-Type': 'application/json'
    }

@pytest.fixture
def invalid_auth_headers(mock_invalid_jwt_token):
    """Generate invalid authentication headers."""
    return {'Authorization': f'Bearer {mock_invalid_jwt_token}'}


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

@pytest.fixture
def mock_optimizer_data():
    data = [
        {'name': 'Player1', 'position': 'QB', 'team': 'A', 'salary': 6000, 'predicted_points': 20},
        {'name': 'Player2', 'position': 'RB', 'team': 'B', 'salary': 5500, 'predicted_points': 18},
        {'name': 'Player3', 'position': 'WR', 'team': 'A', 'salary': 5000, 'predicted_points': 16},
        {'name': 'Player4', 'position': 'TE', 'team': 'C', 'salary': 4000, 'predicted_points': 14},
        {'name': 'Player5', 'position': 'FLEX', 'team': 'B', 'salary': 4500, 'predicted_points': 15},
        {'name': 'Player6', 'position': 'DST', 'team': 'C', 'salary': 3500, 'predicted_points': 12},
    ]
    return pd.DataFrame(data)