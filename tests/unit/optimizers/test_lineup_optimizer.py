import pytest
import pandas as pd
from unittest.mock import MagicMock
import pulp

from optimizers.lineup_optimizer import LineupOptimizer

def test_optimize():
    optimizer = LineupOptimizer()
    
    # Create mock player data
    data = {'name': ['Player 1', 'Player 2', 'Player 3'],
            'team': ['Team A', 'Team B', 'Team C'],
            'position': ['QB', 'RB', 'WR'],
            'salary': [5000, 6000, 7000],
            'predicted_points': [20, 25, 30]}
    player_data = pd.DataFrame(data)
    
    # Mock the _prepare_player_pool method
    optimizer._prepare_player_pool = MagicMock(return_value=player_data)
    
    # Mock the _create_optimization_problem and related methods
    mock_prob = MagicMock(spec=pulp.LpProblem)
    optimizer._create_optimization_problem = MagicMock(return_value=mock_prob)
    optimizer._add_stacking_constraints = MagicMock()
    optimizer._add_lineup_difference_constraints = MagicMock()
    mock_prob.solve = MagicMock(return_value=pulp.LpStatusOptimal)  # Mock successful solve
    
    # Mock the _extract_lineup method
    lineup_data = {'name': ['Player 1', 'Player 2', 'Player 3'],
                   'team': ['Team A', 'Team B', 'Team C'],
                   'position': ['QB', 'RB', 'WR'],
                   'salary': [5000, 6000, 7000],
                   'predicted_points': [20, 25, 30]}
    mock_lineup = pd.DataFrame(lineup_data)
    optimizer._extract_lineup = MagicMock(return_value=mock_lineup)
    
    # Call the optimize method
    lineup = optimizer.optimize(player_data)
    
    # Assert that the result is a DataFrame
    assert isinstance(lineup, pd.DataFrame)
    
    # Assert that the DataFrame contains the expected data
    assert 'name' in lineup.columns
    assert 'position' in lineup.columns
    assert 'salary' in lineup.columns
    assert 'predicted_points' in lineup.columns