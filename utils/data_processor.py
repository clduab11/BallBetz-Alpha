import pandas as pd
from typing import List, Dict, Optional
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing player data."""
    
    def __init__(self):
        """Initialize the DataProcessor."""
        pass
    
    def clean_player_data(self, player_data):
        """
        Clean player data.
        
        Args:
            player_data: The raw player data.
            
        Returns:
            A pandas DataFrame with the cleaned data.
        """
        if player_data.empty:
            logger.warning("No player data to clean.")
            return pd.DataFrame()
        
        # Use the existing process_player_data function
        return process_player_data(player_data)

def clean_data(raw_data: List[Dict]) -> List[Dict]:
    """
    Cleans the raw scraped data.  Currently a placeholder, but this is where
    data cleaning logic (type conversions, handling missing data, etc.)
    would be implemented.
    """
    cleaned_data = []
    for item in raw_data:
        # Example: Convert relevant fields to numeric, handling errors
        cleaned_item = item.copy()  # Avoid modifying original
        for key in item:
            if key in ['games_played', 'passing_attempts', 'passing_completions',
                       'passing_yards', 'passing_touchdowns', 'interceptions',
                       'rushing_attempts', 'rushing_yards', 'rushing_touchdowns',
                       'receptions', 'receiving_yards', 'receiving_touchdowns']:
                try:
                    cleaned_item[key] = int(item[key])
                except (ValueError, TypeError):
                    cleaned_item[key] = 0  # Or handle differently, e.g., use NaN
            elif key in ['passing_completion_pct', 'passing_yards_per_game',
                         'passer_rating', 'rushing_yards_per_attempt',
                         'rushing_yards_per_game', 'receiving_yards_per_reception',
                         'receiving_yards_per_game']:
                try:
                    cleaned_item[key] = float(item[key])
                except (ValueError, TypeError):
                    cleaned_item[key] = 0.0  # Or handle differently
        cleaned_data.append(cleaned_item)
    return cleaned_data

def process_player_data(raw_data: List[Dict], week: Optional[int] = None) -> pd.DataFrame:
    """
    Processes the raw scraped player data.  Combines data from different
    sources (passing, rushing, receiving), determines player positions,
    and adds timestamps.

    Args:
        raw_data: The raw data from the scraper.
        week: Optional week number.

    Returns:
        A Pandas DataFrame with the processed data.
    """
    if not raw_data:
        logger.warning("No raw data provided to process_player_data.")
        return pd.DataFrame()

    # Create a DataFrame
    df = pd.DataFrame(raw_data)

    # Determine positions
    df = _determine_positions(df)

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
        df['week'] = _get_current_week() # Need to implement or import

    # Calculate total touchdowns
    td_cols = ['passing_touchdowns', 'rushing_touchdowns', 'receiving_touchdowns']
    df['touchdowns'] = df[[col for col in td_cols if col in df.columns]].sum(axis=1)

    # Save checkpoint (optional, but good practice)
    if not df.empty:
        checkpoint_path = f'BallBetz-Alpha/data/raw/player_stats_week_{week}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
        # Ensure directory exists
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(checkpoint_path, index=False)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    return df


def _determine_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine player positions based on their stats.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

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

    # Log position distribution (optional, for debugging)
    # position_counts = df['position'].value_counts()
    # logger.info(f"Position distribution: {position_counts.to_dict()}")

    return df

def _get_current_week() -> int:
    """Placeholder for getting the current week."""
    return 1