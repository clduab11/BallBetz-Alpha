import pytest
import pandas as pd

from utils.data_processor import clean_data, process_player_data

def test_clean_data():
    raw_data = [
        {'name': 'Player 1', 'team': 'Team A', 'position': 'QB', 'passing_yards': '250', 'rushing_yards': '50'},
        {'name': 'Player 2', 'team': 'Team B', 'position': 'RB', 'passing_yards': '100', 'rushing_yards': '75'}
    ]
    cleaned_data = clean_data(raw_data)
    assert isinstance(cleaned_data, list)
    assert len(cleaned_data) == 2
    assert all(isinstance(item, dict) for item in cleaned_data)

def test_process_player_data():
    raw_data = [
        {'name': 'Player 1', 'team': 'Team A', 'position': 'QB', 'passing_yards': '250', 'rushing_yards': '50'},
        {'name': 'Player 2', 'team': 'Team B', 'position': 'RB', 'passing_yards': '100', 'rushing_yards': '75'}
    ]
    processed_data = process_player_data(raw_data)
    assert isinstance(processed_data, pd.DataFrame)
    assert 'name' in processed_data.columns
    assert 'team' in processed_data.columns
    assert 'position' in processed_data.columns
    assert 'passing_yards' in processed_data.columns
    assert 'rushing_yards' in processed_data.columns