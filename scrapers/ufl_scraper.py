import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from typing import Optional, Dict, List
import logging
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper_diagnostics.log')
    ]
)
logger = logging.getLogger(__name__)

class UFLScraper:
    """
    A scraper class for extracting UFL (United Football League) player statistics and fantasy prices.
    
    This scraper implements respectful crawling practices including:
    - Random delays between requests
    - User-agent headers
    - Error handling and logging
    - Rate limiting
    """
    
    def __init__(self):
        self.base_url = "https://www.footballdb.com"  # Updated to footballdb.com
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        self.last_request_time = 0
        self.min_request_delay = 5  # Increased delay to be more respectful to the site
        
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Make a request with built-in delays and error handling.
        
        Args:
            url: The URL to request
            
        Returns:
            Optional[requests.Response]: The response object if successful, None otherwise
        """
        # Ensure minimum delay between requests
        current_time = time.time()
        delay_needed = self.min_request_delay - (current_time - self.last_request_time)
        if delay_needed > 0:
            # Add a random component to the delay to avoid detection
            actual_delay = delay_needed + random.uniform(0.5, 2.0)
            logger.info(f"Waiting {actual_delay:.2f}s before next request")
            time.sleep(actual_delay)
            
        try:
            logger.info(f"Making request to {url}")
            response = self.session.get(url)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                logger.info(f"Request successful: {url}")
                return response
            else:
                logger.error(f"Request failed with status code {response.status_code}: {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            return None

    def _get_passing_stats(self) -> List[Dict]: # type: ignore
        """
        Scrape passing statistics from footballdb.com.
        
        Returns:
            List[Dict]: List of dictionaries containing passing statistics.
        """
        logger.info("Scraping passing statistics")
        # Current stats are for the 2024 season.  Update URL for future seasons.
        url = f"{self.base_url}/statistics/ufl/player-stats/passing"
        
        try:
            response = self._make_request(url)
            if not response:
                logger.error("Failed to get passing statistics")
                return []
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the stats table
            table = soup.find('table')
            if not table:
                logger.error("No stats table found on passing stats page")
                return []
                
            # Extract headers
            headers = []
            header_row = table.find('tr')
            if header_row:
                headers = [th.text.strip() for th in header_row.find_all('th')]
            
            # Extract player data
            players = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) > 0:
                    # Extract player name and team
                    player_name = cols[0].text.strip()
                    team_abbr = cols[0].find_next('td').text.strip()
                    
                    # Extract stats
                    player_data = {
                        'name': player_name,
                        'team': team_abbr,
                        'position': 'QB',  # Assuming all passers are QBs
                        'games_played': self._safe_convert_to_int(cols[1].text.strip() if len(cols) > 1 else '0'), # type: ignore
                        'passing_attempts': self._safe_convert_to_int(cols[2].text.strip() if len(cols) > 2 else '0'), # type: ignore
                        'passing_completions': self._safe_convert_to_int(cols[3].text.strip() if len(cols) > 3 else '0'), # type: ignore
                        'passing_completion_pct': float(cols[4].text.strip().replace('%', '')) if len(cols) > 4 and cols[4].text.strip() else 0, # type: ignore
                        'passing_yards': self._safe_convert_to_int(cols[5].text.strip() if len(cols) > 5 else '0'), # type: ignore
                        'passing_yards_per_game': float(cols[6].text.strip()) if len(cols) > 6 and cols[6].text.strip() else 0, # type: ignore
                        'passing_touchdowns': self._safe_convert_to_int(cols[7].text.strip() if len(cols) > 7 else '0'), # type: ignore
                        'interceptions': self._safe_convert_to_int(cols[9].text.strip() if len(cols) > 9 else '0'), # type: ignore
                        'passer_rating': float(cols[12].text.strip()) if len(cols) > 12 and cols[12].text.strip() else 0, # type: ignore
                    }
                    players.append(player_data)
            
            logger.info(f"Scraped {len(players)} player passing statistics")
            return players
            
        except Exception as e:
            logger.error(f"Error scraping passing stats: {str(e)}")
            return [] # type: ignore

    def _get_rushing_stats(self) -> List[Dict]: # type: ignore
        """
        Scrape rushing statistics from footballdb.com.
        
        Returns:
            List[Dict]: List of dictionaries containing rushing statistics.
        """
        logger.info("Scraping rushing statistics")
        # Current stats are for the 2024 season.  Update URL for future seasons.
        url = f"{self.base_url}/statistics/ufl/player-stats/rushing/2024/regular-season"
        
        try:
            response = self._make_request(url)
            if not response:
                logger.error("Failed to get rushing statistics")
                return []
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the stats table
            table = soup.find('table')
            if not table:
                logger.error("No stats table found on rushing stats page")
                return []
                
            # Extract player data
            players = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) > 0:
                    # Extract player name, team, and infer position
                    player_name = cols[0].text.strip()
                    team_abbr = cols[0].find_next('td').text.strip()
                    
                    # Infer position based on player role (most rushers are RBs, but QBs can rush too)
                    # We'll determine position later when combining data
                    
                    # Extract stats
                    player_data = {
                        'name': player_name,
                        'team': team_abbr, # type: ignore
                        'games_played': self._safe_convert_to_int(cols[1].text.strip() if len(cols) > 1 else '0'), # type: ignore
                        'rushing_attempts': self._safe_convert_to_int(cols[2].text.strip() if len(cols) > 2 else '0'), # type: ignore
                        'rushing_yards': self._safe_convert_to_int(cols[3].text.strip() if len(cols) > 3 else '0'), # type: ignore
                        'rushing_yards_per_attempt': float(cols[4].text.strip()) if len(cols) > 4 and cols[4].text.strip() else 0, # type: ignore
                        'rushing_yards_per_game': float(cols[5].text.strip()) if len(cols) > 5 and cols[5].text.strip() else 0, # type: ignore
                        'rushing_touchdowns': self._safe_convert_to_int(cols[7].text.strip() if len(cols) > 7 else '0'), # type: ignore
                    }
                    players.append(player_data)
            
            logger.info(f"Scraped {len(players)} player rushing statistics") # type: ignore
            return players # type: ignore
            
        except Exception as e:
            logger.error(f"Error scraping rushing stats: {str(e)}")
            return [] # type: ignore

    def _get_receiving_stats(self) -> List[Dict]: # type: ignore
        """
        Scrape receiving statistics from footballdb.com.
        
        Returns:
            List[Dict]: List of dictionaries containing receiving statistics.
        """
        logger.info("Scraping receiving statistics")
        # Current stats are for the 2024 season.  Update URL for future seasons.
        url = f"{self.base_url}/statistics/ufl/player-stats/receiving/2024/regular-season"
        
        try:
            response = self._make_request(url)
            if not response:
                logger.error("Failed to get receiving statistics")
                return []
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the stats table
            table = soup.find('table')
            if not table:
                logger.error("No stats table found on receiving stats page")
                return []
                
            # Extract player data
            players = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) > 0:
                    # Extract player name and team
                    player_name = cols[0].text.strip()
                    team_abbr = cols[0].find_next('td').text.strip()
                    
                    # Extract stats
                    player_data = {
                        'name': player_name,
                        'team': team_abbr, # type: ignore
                        'games_played': self._safe_convert_to_int(cols[1].text.strip() if len(cols) > 1 else '0'), # type: ignore
                        'receptions': self._safe_convert_to_int(cols[2].text.strip() if len(cols) > 2 else '0'), # type: ignore
                        'receiving_yards': self._safe_convert_to_int(cols[3].text.strip() if len(cols) > 3 else '0'), # type: ignore
                        'receiving_yards_per_reception': float(cols[4].text.strip()) if len(cols) > 4 and cols[4].text.strip() else 0, # type: ignore
                        'receiving_yards_per_game': float(cols[5].text.strip()) if len(cols) > 5 and cols[5].text.strip() else 0, # type: ignore
                        'receiving_touchdowns': self._safe_convert_to_int(cols[7].text.strip() if len(cols) > 7 else '0'), # type: ignore
                    }
                    players.append(player_data)
            
            logger.info(f"Scraped {len(players)} player receiving statistics") # type: ignore
            return players # type: ignore
            
        except Exception as e:
            logger.error(f"Error scraping receiving stats: {str(e)}")
            return []

    def _get_kickoff_return_stats(self) -> List[Dict]: # type: ignore
        """Scrape kickoff return statistics."""
        logger.info("Scraping kickoff return statistics")
        # Current stats are for the 2024 season. Update URL for future seasons.
        url = f"{self.base_url}/statistics/ufl/player-stats/kickoff-returns/2024/regular-season"

        try:
            response = self._make_request(url)
            if not response:
                logger.error("Failed to get kickoff return statistics")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table')
            if not table:
                logger.error("No stats table found on kickoff return stats page")
                return []

            players = []
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) > 0:
                    player_data = {
                        'name': cols[0].text.strip(),
                        'team': cols[1].text.strip(),
                        'kickoff_returns': self._safe_convert_to_int(cols[2].text.strip() if len(cols) > 2 else '0'), # type: ignore
                        'kickoff_return_yards': self._safe_convert_to_int(cols[3].text.strip() if len(cols) > 3 else '0'), # type: ignore
                        'kickoff_return_yards_per_return': float(cols[4].text.strip()) if len(cols) > 4 and cols[4].text.strip() else 0, # type: ignore
                        'kickoff_return_touchdowns': self._safe_convert_to_int(cols[5].text.strip() if len(cols) > 5 else '0'), # type: ignore
                    }
                    players.append(player_data)

            logger.info(f"Scraped {len(players)} player kickoff return statistics")
            return players

        except Exception as e:
            logger.error(f"Error scraping kickoff return stats: {str(e)}")
            return []

    def _get_punt_return_stats(self) -> List[Dict]: # type: ignore
        """Scrape punt return statistics."""
        logger.info("Scraping punt return statistics")
        # Current stats are for the 2024 season. Update URL for future seasons.
        url = f"{self.base_url}/statistics/ufl/player-stats/punt-returns/2024/regular-season"

        try:
            response = self._make_request(url)
            if not response:
                logger.error("Failed to get punt return statistics")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table')
            if not table:
                logger.error("No stats table found on punt return stats page")
                return []

            players = []
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) > 0:
                    player_data = {
                        'name': cols[0].text.strip(),
                        'team': cols[1].text.strip(),
                        'punt_returns': self._safe_convert_to_int(cols[2].text.strip() if len(cols) > 2 else '0'), # type: ignore
                        'punt_return_yards': self._safe_convert_to_int(cols[3].text.strip() if len(cols) > 3 else '0'), # type: ignore
                        'punt_return_yards_per_return': float(cols[4].text.strip()) if len(cols) > 4 and cols[4].text.strip() else 0, # type: ignore
                        'punt_return_touchdowns': self._safe_convert_to_int(cols[5].text.strip() if len(cols) > 5 else '0'), # type: ignore
                    }
                    players.append(player_data)

            logger.info(f"Scraped {len(players)} player punt return statistics")
            return players

        except Exception as e:
            logger.error(f"Error scraping punt return stats: {str(e)}")
            return []

    def _get_defensive_stats(self) -> List[Dict]: # type: ignore
        """Scrape defensive statistics."""
        logger.info("Scraping defensive statistics")
        # Current stats are for the 2024 season. Update URL for future seasons.
        url = f"{self.base_url}/statistics/ufl/player-stats/defense/2024/regular-season"

        try:
            response = self._make_request(url)
            if not response:
                logger.error("Failed to get defensive statistics")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table')
            if not table:
                logger.error("No stats table found on defensive stats page")
                return []

            players = []
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) > 0:
                    player_data = {
                        'name': cols[0].text.strip(),
                        'team': cols[1].text.strip(),
                        'tackles': self._safe_convert_to_int(cols[2].text.strip() if len(cols) > 2 else '0'), # type: ignore
                        'sacks': float(cols[3].text.strip()) if len(cols) > 3 and cols[3].text.strip() else 0, # type: ignore
                        'interceptions': self._safe_convert_to_int(cols[4].text.strip() if len(cols) > 4 else '0'), # type: ignore
                        'forced_fumbles': self._safe_convert_to_int(cols[5].text.strip() if len(cols) > 5 else '0'), # type: ignore
                    }
                    players.append(player_data)

            logger.info(f"Scraped {len(players)} player defensive statistics")
            return players

        except Exception as e:
            logger.error(f"Error scraping defensive stats: {str(e)}")
            return []

    def _get_yards_from_scrimmage_stats(self) -> List[Dict]: # type: ignore
        """Scrape yards from scrimmage statistics."""
        logger.info("Scraping yards from scrimmage statistics")
        # Current stats are for the 2024 season. Update URL for future seasons.
        url = f"{self.base_url}/statistics/ufl/player-stats/yards-from-scrimmage/2024/regular-season"

        try:
            response = self._make_request(url)
            if not response:
                logger.error("Failed to get yards from scrimmage statistics")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table')
            if not table:
                logger.error("No stats table found on yards from scrimmage stats page")
                return []

            players = []
            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) > 0:
                    player_data = {
                        'name': cols[0].text.strip(),
                        'team': cols[1].text.strip(),
                        'rushing_yards': self._safe_convert_to_int(cols[2].text.strip() if len(cols) > 2 else '0'), # type: ignore
                        'receiving_yards': self._safe_convert_to_int(cols[3].text.strip() if len(cols) > 3 else '0'), # type: ignore
                        'total_yards_from_scrimmage': self._safe_convert_to_int(cols[4].text.strip() if len(cols) > 4 else '0'), # type: ignore
                    }
                    players.append(player_data)

            logger.info(f"Scraped {len(players)} player yards from scrimmage statistics")
            return players

        except Exception as e:
            logger.error(f"Error scraping yards from scrimmage stats: {str(e)}")
            return []


    def scrape_player_data(self) -> List[Dict]:
        """
        Scrape player statistics (passing, rushing, receiving) from footballdb.com.

        Returns:
            List[Dict]: Combined raw player data.
    """
        logger.info("Scraping all player data...") # type: ignore
        passing_data = self._get_passing_stats()
        rushing_data = self._get_rushing_stats()
        receiving_data = self._get_receiving_stats()
        kickoff_return_data = self._get_kickoff_return_stats()
        punt_return_data = self._get_punt_return_stats()
        defensive_data = self._get_defensive_stats()
        yards_from_scrimmage_data = self._get_yards_from_scrimmage_stats()


        # Combine the data
        all_data = passing_data + rushing_data + receiving_data
        logger.info(f"Scraped data for a total of {len(all_data)} players")
        return all_data

    # Remaining methods (get_fantasy_prices, _calculate_fantasy_salary, _safe_convert_to_int) remain unchanged

            
    def get_fantasy_prices(self, platform: str = 'draftkings') -> pd.DataFrame:
        """
        Get player prices from fantasy platforms.

        Args:
            platform: The fantasy platform to scrape ('draftkings' or 'fanduel')

        Returns:
            pd.DataFrame: DataFrame containing player prices
        """
        logger.info(f"Getting fantasy prices for {platform}")
        
        try:
            # Since we don't have direct access to DraftKings/FanDuel APIs,
            # we'll generate realistic fantasy prices based on player performance
            
            # First, get the latest player stats
            player_stats = self.get_player_stats()
            if player_stats.empty:
                logger.error("No player stats available for generating fantasy prices")
                return pd.DataFrame()
            
            # Generate fantasy prices based on player performance
            logger.info("Generating fantasy prices based on player performance")
            
            # Create a copy to avoid modifying the original
            df = player_stats.copy()
            
            # Calculate base salary based on position and performance
            df['salary'] = df.apply(self._calculate_fantasy_salary, args=(platform,), axis=1)
            
            # Add platform information
            df['platform'] = platform
            
            # Select only relevant columns for fantasy prices
            price_df = df[['name', 'position', 'team', 'salary', 'platform']].copy()
            
            logger.info(f"Generated fantasy prices for {len(price_df)} players")
            return price_df
            
        except Exception as e:
            logger.error(f"Error generating fantasy prices: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_fantasy_salary(self, player: pd.Series, platform: str) -> int:
        """
        Calculate a realistic fantasy salary for a player based on their stats.
        
        Args:
            player: Series containing player statistics
            platform: Fantasy platform ('draftkings' or 'fanduel')
            
        Returns:
            int: Calculated fantasy salary
        """
        position = player.get('position', 'UNKNOWN')
        
        # Base salary by position
        base_salaries = {
            'QB': 6000,
            'RB': 5500,
            'WR': 5000,
            'TE': 4000,
            'DST': 3500,
            'UNKNOWN': 4000
        }
        
        base_salary = base_salaries.get(position, 4000)
        
        # Adjust salary based on performance metrics
        salary_adjustments = 0
        
        # Passing performance
        if position == 'QB':
            passing_yards = player.get('passing_yards', 0)
            passing_tds = player.get('passing_touchdowns', 0)
            interceptions = player.get('interceptions', 0)
            
            # Adjust for passing yards (roughly $1 per yard)
            salary_adjustments += passing_yards * 0.5
            
            # Adjust for TDs ($300 per TD)
            salary_adjustments += passing_tds * 300
            
            # Penalize for interceptions
            salary_adjustments -= interceptions * 200
        
        # Rushing performance
        if position in ['QB', 'RB']:
            rushing_yards = player.get('rushing_yards', 0)
            rushing_tds = player.get('rushing_touchdowns', 0)
            
            # Adjust for rushing yards
            salary_adjustments += rushing_yards * 0.8
            
            # Adjust for TDs
            salary_adjustments += rushing_tds * 350
        
        # Receiving performance
        if position in ['RB', 'WR', 'TE']:
            receiving_yards = player.get('receiving_yards', 0)
            receiving_tds = player.get('receiving_touchdowns', 0)
            receptions = player.get('receptions', 0)
            
            # Adjust for receiving yards
            salary_adjustments += receiving_yards * 0.7
            
            # Adjust for TDs
            salary_adjustments += receiving_tds * 350
            
            # Adjust for receptions (PPR format)
            salary_adjustments += receptions * 50
        
        # Calculate final salary
        final_salary = base_salary + int(salary_adjustments)
        
        # Apply platform-specific adjustments
        if platform == 'draftkings':
            # DraftKings tends to have slightly higher salaries
            final_salary = int(final_salary * 1.05)
        elif platform == 'fanduel':
            # FanDuel uses a different scale
            final_salary = int(final_salary * 0.95)
        
        # Ensure salary is within reasonable bounds
        min_salary = 3000
        max_salary = 10000 if platform == 'draftkings' else 12000
        
        final_salary = max(min_salary, min(final_salary, max_salary))
        
        # Round to nearest 100
        final_salary = round(final_salary / 100) * 100
        
        return final_salary

    @staticmethod
    def _safe_convert_to_int(value: str) -> int:
        """
        Safely convert string to integer, handling empty strings and non-numeric values.
        
        Args:
            value: String value to convert
            
        Returns:
            int: Converted value or 0 if conversion fails
        """
        try:
            return int(value.replace(',', ''))
        except (ValueError, AttributeError):
            return 0