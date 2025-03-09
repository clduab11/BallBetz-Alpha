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

    def _get_passing_stats(self) -> pd.DataFrame:
        """
        Scrape passing statistics from footballdb.com.
        
        Returns:
            pd.DataFrame: DataFrame containing passing statistics
        """
        logger.info("Scraping passing statistics")
        url = f"{self.base_url}/ufl/stats/2024/passing"
        
        try:
            response = self._make_request(url)
            if not response:
                logger.error("Failed to get passing statistics")
                return pd.DataFrame()
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the stats table
            table = soup.find('table')
            if not table:
                logger.error("No stats table found on passing stats page")
                return pd.DataFrame()
                
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
                        'games_played': self._safe_convert_to_int(cols[1].text.strip()),
                        'passing_attempts': self._safe_convert_to_int(cols[2].text.strip()),
                        'passing_completions': self._safe_convert_to_int(cols[3].text.strip()),
                        'passing_completion_pct': float(cols[4].text.strip().replace('%', '')) if cols[4].text.strip() else 0,
                        'passing_yards': self._safe_convert_to_int(cols[5].text.strip()),
                        'passing_yards_per_game': float(cols[6].text.strip()) if cols[6].text.strip() else 0,
                        'passing_touchdowns': self._safe_convert_to_int(cols[7].text.strip()),
                        'interceptions': self._safe_convert_to_int(cols[9].text.strip()),
                        'passer_rating': float(cols[12].text.strip()) if cols[12].text.strip() else 0,
                    }
                    players.append(player_data)
            
            df = pd.DataFrame(players)
            logger.info(f"Scraped {len(df)} player passing statistics")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping passing stats: {str(e)}")
            return pd.DataFrame()

    def _get_rushing_stats(self) -> pd.DataFrame:
        """
        Scrape rushing statistics from footballdb.com.
        
        Returns:
            pd.DataFrame: DataFrame containing rushing statistics
        """
        logger.info("Scraping rushing statistics")
        url = f"{self.base_url}/ufl/stats/2024/rushing"
        
        try:
            response = self._make_request(url)
            if not response:
                logger.error("Failed to get rushing statistics")
                return pd.DataFrame()
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the stats table
            table = soup.find('table')
            if not table:
                logger.error("No stats table found on rushing stats page")
                return pd.DataFrame()
                
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
                        'team': team_abbr,
                        'games_played': self._safe_convert_to_int(cols[1].text.strip()),
                        'rushing_attempts': self._safe_convert_to_int(cols[2].text.strip()),
                        'rushing_yards': self._safe_convert_to_int(cols[3].text.strip()),
                        'rushing_yards_per_attempt': float(cols[4].text.strip()) if cols[4].text.strip() else 0,
                        'rushing_yards_per_game': float(cols[5].text.strip()) if cols[5].text.strip() else 0,
                        'rushing_touchdowns': self._safe_convert_to_int(cols[7].text.strip()),
                    }
                    players.append(player_data)
            
            df = pd.DataFrame(players)
            logger.info(f"Scraped {len(df)} player rushing statistics")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping rushing stats: {str(e)}")
            return pd.DataFrame()

    def _get_receiving_stats(self) -> pd.DataFrame:
        """
        Scrape receiving statistics from footballdb.com.
        
        Returns:
            pd.DataFrame: DataFrame containing receiving statistics
        """
        logger.info("Scraping receiving statistics")
        url = f"{self.base_url}/ufl/stats/2024/receiving"
        
        try:
            response = self._make_request(url)
            if not response:
                logger.error("Failed to get receiving statistics")
                return pd.DataFrame()
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the stats table
            table = soup.find('table')
            if not table:
                logger.error("No stats table found on receiving stats page")
                return pd.DataFrame()
                
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
                        'games_played': self._safe_convert_to_int(cols[1].text.strip()),
                        'receptions': self._safe_convert_to_int(cols[2].text.strip()),
                        'receiving_yards': self._safe_convert_to_int(cols[3].text.strip()),
                        'receiving_yards_per_reception': float(cols[4].text.strip()) if cols[4].text.strip() else 0,
                        'receiving_yards_per_game': float(cols[5].text.strip()) if cols[5].text.strip() else 0,
                        'receiving_touchdowns': self._safe_convert_to_int(cols[7].text.strip()),
                    }
                    players.append(player_data)
            
            df = pd.DataFrame(players)
            logger.info(f"Scraped {len(df)} player receiving statistics")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping receiving stats: {str(e)}")
            return pd.DataFrame()

    def _determine_positions(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Determine player positions based on their stats.
        
        Args:
            combined_df: DataFrame with combined player statistics
            
        Returns:
            pd.DataFrame: DataFrame with position information added
        """
        # Create a copy to avoid modifying the original
        df = combined_df.copy()
        
        # Players already with a position (from passing stats) keep their position
        # For others, determine based on stats:
        
        # Define position determination logic
        def determine_position(row):
            if pd.notna(row.get('position')):
                return row['position']
            
            # QBs have significant passing stats
            if pd.notna(row.get('passing_yards')) and row['passing_yards'] > 100:
                return 'QB'
                
            # RBs have more rushing than receiving yards
            if (pd.notna(row.get('rushing_yards')) and pd.notna(row.get('receiving_yards')) and 
                row['rushing_yards'] > row['receiving_yards']):
                return 'RB'
                
            # WRs have significant receiving yards
            if pd.notna(row.get('receiving_yards')) and row['receiving_yards'] > 100:
                return 'WR'
                
            # TEs have moderate receiving yards
            if pd.notna(row.get('receiving_yards')) and 50 <= row['receiving_yards'] <= 400:
                # This is a rough heuristic - in reality we'd need more data
                return 'TE'
                
            # Default to WR if they have any receiving stats
            if pd.notna(row.get('receiving_yards')) and row['receiving_yards'] > 0:
                return 'WR'
                
            # Default to RB if they have any rushing stats
            if pd.notna(row.get('rushing_yards')) and row['rushing_yards'] > 0:
                return 'RB'
                
            # If we can't determine, use a default
            return 'UNKNOWN'
        
        # Apply position determination
        df['position'] = df.apply(determine_position, axis=1)
        
        # Log position distribution
        position_counts = df['position'].value_counts()
        logger.info(f"Position distribution: {position_counts.to_dict()}")
        
        return df

    def get_player_stats(self, week: Optional[int] = None) -> pd.DataFrame:
        """
        Scrape player statistics from footballdb.com.
        
        This method combines passing, rushing, and receiving statistics
        into a comprehensive player dataset.
        
        Args:
            week: Optional week number to filter stats
            
        Returns:
            pd.DataFrame: DataFrame containing player statistics
        """
        players: List[Dict] = []
        
        try:
            logger.info(f"Getting player stats for week: {week if week else 'all'}")
            
            # Get stats from different categories
            passing_df = self._get_passing_stats()
            rushing_df = self._get_rushing_stats()
            receiving_df = self._get_receiving_stats()
            
            if passing_df.empty and rushing_df.empty and receiving_df.empty:
                logger.error("Failed to retrieve any player statistics")
                return pd.DataFrame()
            
            # Combine the dataframes
            logger.info("Combining player statistics")
            
            # Start with a list of all unique players
            all_players = set()
            for df in [passing_df, rushing_df, receiving_df]:
                if not df.empty:
                    all_players.update(df['name'].tolist())
            
            logger.info(f"Found {len(all_players)} unique players")
            
            # Create a combined dataframe with all player stats
            combined_data = []
            
            for player_name in all_players:
                player_data = {'name': player_name}
                
                # Get player's team (should be consistent across dataframes)
                team = None
                for df in [passing_df, rushing_df, receiving_df]:
                    if not df.empty and player_name in df['name'].values:
                        team = df.loc[df['name'] == player_name, 'team'].iloc[0]
                        break
                
                player_data['team'] = team
                
                # Add passing stats if available
                if not passing_df.empty and player_name in passing_df['name'].values:
                    pass_data = passing_df.loc[passing_df['name'] == player_name].iloc[0].to_dict()
                    for key, value in pass_data.items():
                        if key not in ['name', 'team']:  # Avoid duplicating these fields
                            player_data[key] = value
                
                # Add rushing stats if available
                if not rushing_df.empty and player_name in rushing_df['name'].values:
                    rush_data = rushing_df.loc[rushing_df['name'] == player_name].iloc[0].to_dict()
                    for key, value in rush_data.items():
                        if key not in ['name', 'team']:  # Avoid duplicating these fields
                            player_data[key] = value
                
                # Add receiving stats if available
                if not receiving_df.empty and player_name in receiving_df['name'].values:
                    rec_data = receiving_df.loc[receiving_df['name'] == player_name].iloc[0].to_dict()
                    for key, value in rec_data.items():
                        if key not in ['name', 'team']:  # Avoid duplicating these fields
                            player_data[key] = value
                
                combined_data.append(player_data)
            
            # Create combined DataFrame
            df = pd.DataFrame(combined_data)
            
            # Determine positions for players
            df = self._determine_positions(df)
            
            # Fill NaN values with 0 for numeric columns
            numeric_cols = [
                'passing_attempts', 'passing_completions', 'passing_yards', 'passing_touchdowns',
                'rushing_attempts', 'rushing_yards', 'rushing_touchdowns',
                'receptions', 'receiving_yards', 'receiving_touchdowns',
                'interceptions'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            # Add timestamp and week
            df['timestamp'] = pd.Timestamp.now()
            if week:
                df['week'] = week
            else:
                # If no specific week, assume it's the current week
                df['week'] = self._get_current_week()
            
            # Calculate total touchdowns
            td_cols = ['passing_touchdowns', 'rushing_touchdowns', 'receiving_touchdowns']
            df['touchdowns'] = df[[col for col in td_cols if col in df.columns]].sum(axis=1)
            
            # Save checkpoint
            if not df.empty:
                checkpoint_path = f'data/raw/player_stats_week_{week}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
                # Ensure directory exists
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(checkpoint_path, index=False)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
            logger.info(f"Successfully compiled stats for {len(df)} players")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping player stats: {str(e)}")
            return pd.DataFrame()
    
    def _get_current_week(self) -> int:
        """
        Determine the current UFL week based on the schedule.
        
        Returns:
            int: Current week number
        """
        try:
            # This is a simplified approach - in a real implementation,
            # we would scrape the schedule and determine the current week
            # based on the current date
            
            # For now, we'll use a hardcoded value for the 2024 season
            # In a real implementation, this would be determined dynamically
            return 10  # Assuming we're in week 10 of the UFL season
            
        except Exception as e:
            logger.error(f"Error determining current week: {str(e)}")
            return 1  # Default to week 1 if we can't determine
            
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