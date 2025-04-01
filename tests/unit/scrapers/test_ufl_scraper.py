import unittest
from unittest.mock import MagicMock

from scrapers.ufl_scraper import UFLScraper

class TestUFLScraper(unittest.TestCase):
    
    def test_scrape_player_data_merging(self):
        """Test that player data from different stat categories is properly merged."""
        scraper = UFLScraper()
        
        # Mock the _make_request method
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Mock Data</body></html>"
        scraper._make_request = MagicMock(return_value=mock_response)
        
        # Mock the individual stats methods with overlapping player data
        # Player 1 appears in all three categories
        # Player 2 appears in two categories
        # Player 3 appears in only one category
        scraper._get_passing_stats = MagicMock(return_value=[
            {'name': 'Player 1', 'team': 'Team A', 'position': 'QB', 'games_played': 10, 
             'passing_yards': 250, 'passing_attempts': 30, 'passing_touchdowns': 2},
            {'name': 'Player 2', 'team': 'Team B', 'position': 'QB', 'games_played': 8, 
             'passing_yards': 180, 'passing_attempts': 25, 'passing_touchdowns': 1}
        ])
        
        scraper._get_rushing_stats = MagicMock(return_value=[
            {'name': 'Player 1', 'team': 'Team A', 'games_played': 10, 
             'rushing_yards': 50, 'rushing_attempts': 15, 'rushing_touchdowns': 1},
            {'name': 'Player 3', 'team': 'Team C', 'games_played': 9, 
             'rushing_yards': 120, 'rushing_attempts': 25, 'rushing_touchdowns': 2}
        ])
        
        scraper._get_receiving_stats = MagicMock(return_value=[
            {'name': 'Player 1', 'team': 'Team A', 'games_played': 10, 
             'receiving_yards': 100, 'receptions': 8, 'receiving_touchdowns': 1},
            {'name': 'Player 2', 'team': 'Team B', 'games_played': 8, 
             'receiving_yards': 70, 'receptions': 6, 'receiving_touchdowns': 0}
        ])
        
        # Mock other stat methods to return empty lists
        scraper._get_kickoff_return_stats = MagicMock(return_value=[])
        scraper._get_punt_return_stats = MagicMock(return_value=[])
        scraper._get_defensive_stats = MagicMock(return_value=[])
        
        # Call the scrape_player_data method
        player_data = scraper.scrape_player_data()
        
        # Basic assertions
        self.assertIsInstance(player_data, list)
        self.assertEqual(len(player_data), 3)  # Should have 3 unique players
        
        # Find each player in the result
        player1 = next((p for p in player_data if p['name'] == 'Player 1'), None)
        player2 = next((p for p in player_data if p['name'] == 'Player 2'), None)
        player3 = next((p for p in player_data if p['name'] == 'Player 3'), None)
        
        # Assert that all players were found
        self.assertIsNotNone(player1)
        self.assertIsNotNone(player2)
        self.assertIsNotNone(player3)
        
        # Assert that Player 1's stats were properly merged
        self.assertEqual(player1['passing_yards'], 250)
        self.assertEqual(player1['rushing_yards'], 50)
        self.assertEqual(player1['receiving_yards'], 100)
        self.assertEqual(player1['passing_touchdowns'], 2)
        self.assertEqual(player1['rushing_touchdowns'], 1)
        self.assertEqual(player1['receiving_touchdowns'], 1)
        self.assertEqual(player1['position'], 'QB')  # Should be determined as QB due to passing_attempts
        
        # Assert that Player 2's stats were properly merged
        self.assertEqual(player2['passing_yards'], 180)
        self.assertEqual(player2['receiving_yards'], 70)
        self.assertIn('rushing_yards', player2)  # Should exist but be 0
        self.assertEqual(player2['rushing_yards'], 0)
        self.assertEqual(player2['position'], 'QB')  # Should be determined as QB
        
        # Assert that Player 3's stats were properly set
        self.assertEqual(player3['rushing_yards'], 120)
        self.assertIn('passing_yards', player3)  # Should exist but be 0
        self.assertEqual(player3['passing_yards'], 0)
        self.assertEqual(player3['position'], 'RB')  # Should be determined as RB due to rushing_attempts

    def test_scrape_player_data_position_determination(self):
        """Test that player positions are correctly determined based on stats."""
        scraper = UFLScraper()
        
        # Mock the _make_request method
        mock_response = MagicMock()
        mock_response.status_code = 200
        scraper._make_request = MagicMock(return_value=mock_response)
        
        # Create players with different stat profiles to test position determination
        qb_player = {'name': 'QB Player', 'team': 'Team A', 'passing_attempts': 20}
        rb_player = {'name': 'RB Player', 'team': 'Team B', 'rushing_attempts': 15}
        wr_player = {'name': 'WR Player', 'team': 'Team C', 'receptions': 10}
        def_player = {'name': 'DEF Player', 'team': 'Team D', 'tackles': 5, 'sacks': 2}
        
        # Mock the stat methods
        scraper._get_passing_stats = MagicMock(return_value=[qb_player])
        scraper._get_rushing_stats = MagicMock(return_value=[rb_player])
        scraper._get_receiving_stats = MagicMock(return_value=[wr_player])
        scraper._get_defensive_stats = MagicMock(return_value=[def_player])
        scraper._get_kickoff_return_stats = MagicMock(return_value=[])
        scraper._get_punt_return_stats = MagicMock(return_value=[])
        
        # Call the scrape_player_data method
        player_data = scraper.scrape_player_data()
        
        # Find each player in the result
        qb = next((p for p in player_data if p['name'] == 'QB Player'), None)
        rb = next((p for p in player_data if p['name'] == 'RB Player'), None)
        wr = next((p for p in player_data if p['name'] == 'WR Player'), None)
        def_p = next((p for p in player_data if p['name'] == 'DEF Player'), None)
        
        # Assert positions were correctly determined
        self.assertEqual(qb['position'], 'QB')
        self.assertEqual(rb['position'], 'RB')
        self.assertEqual(wr['position'], 'WR')
        self.assertEqual(def_p['position'], 'DEF')

if __name__ == '__main__':
    unittest.main()