import pytest
import pandas as pd
from unittest.mock import MagicMock

from scrapers.ufl_scraper import UFLScraper

def test_scrape_player_data():
    scraper = UFLScraper()
    
    # Mock the _make_request method
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body>Mock Data</body></html>"  # Replace with actual mock HTML
    scraper._make_request = MagicMock(return_value=mock_response)
    
    # Mock the individual stats methods
    scraper._get_passing_stats = MagicMock(return_value=[{'name': 'Player 1', 'passing_yards': 250}])
    scraper._get_rushing_stats = MagicMock(return_value=[{'name': 'Player 1', 'rushing_yards': 50}])
    scraper._get_receiving_stats = MagicMock(return_value=[{'name': 'Player 1', 'receiving_yards': 100}])
    
    # Call the scrape_player_data method
    player_data = scraper.scrape_player_data()
    
    # Assert that the result is a list
    assert isinstance(player_data, list)
    
    # Assert that the list contains the expected data
    assert len(player_data) > 0
    assert 'name' in player_data[0]
    assert 'passing_yards' in player_data[0]

def test_get_fantasy_prices():
    scraper = UFLScraper()
    
    # Mock the _make_request method
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body>Mock Prices</body></html>"  # Replace with actual mock HTML
    scraper._make_request = MagicMock(return_value=mock_response)
    
    # Call the get_fantasy_prices method
    prices = scraper.get_fantasy_prices()
    
    # Assert that the result is a pandas DataFrame
    assert isinstance(prices, pd.DataFrame)