import pandas as pd
from unittest.mock import patch
import pytest
from scrapers.ufl_scraper import UFLScraper
from optimizers.lineup_optimizer import LineupOptimizer

def test_scraper_optimizer_integration(mock_optimizer_data):
    """
    Integration test to verify the data flow from UFLScraper to LineupOptimizer.
    """
    # Instantiate scraper and optimizer
    scraper = UFLScraper()
    optimizer = LineupOptimizer()

    # Mock the scraper's output.  We assume scrape_player_data is the correct method.
    with patch.object(scraper, 'scrape_player_data', return_value=mock_optimizer_data.to_dict('records')) as mock_scrape:
        # Call the optimizer's optimize method with the mocked data
        result = optimizer.optimize(mock_optimizer_data)

        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ['name', 'team', 'position', 'salary', 'predicted_points', 'lineup_number', 'total_salary', 'total_projected', 'salary_remaining', 'timestamp'])
        assert len(result) > 0