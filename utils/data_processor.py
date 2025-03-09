import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('processor_diagnostics.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== DataProcessor Diagnostic Logs ===")

class DataProcessor:
    """
    Process and clean player statistics data for fantasy sports analysis.
    
    This class handles data cleaning, feature engineering, and fantasy point calculations
    for different fantasy sports platforms.
    """
    
    def __init__(self):
        self.scoring_rules: Dict[str, Dict] = {
            'draftkings': {
                'passing_yards': 0.04,
                'passing_touchdowns': 4,
                'rushing_yards': 0.1,
                'rushing_touchdowns': 6,
                'receiving_yards': 0.1,
                'receiving_touchdowns': 6,
                'interceptions': -1,
                'fumbles_lost': -1,
                'receptions': 1,  # PPR scoring
            },
            'fanduel': {
                'passing_yards': 0.04,
                'passing_touchdowns': 4,
                'rushing_yards': 0.1,
                'rushing_touchdowns': 6,
                'receiving_yards': 0.1,
                'receiving_touchdowns': 6,
                'interceptions': -1,
                'fumbles_lost': -1,
                'receptions': 0.5,  # Half PPR scoring
            }
        }
        
    def clean_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process player statistics data.
        
        Args:
            df: DataFrame containing player statistics
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with processed statistics
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided for cleaning")
                return df
                
            # Create copy to avoid modifying original data
            df = df.copy()
            
            # Convert numeric columns to float
            numeric_cols = [
                'passing_yards', 'rushing_yards', 'receiving_yards', 
                'touchdowns', 'interceptions', 'fumbles_lost', 'receptions'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Split touchdowns into passing, rushing, receiving if not already split
            if 'touchdowns' in df.columns and 'passing_touchdowns' not in df.columns:
                # Estimate touchdown types based on yards (simple heuristic)
                logger.info("Splitting touchdowns by type based on yards distribution")
                
                # Calculate total yards safely
                total_yards = df['passing_yards'] + df['rushing_yards'] + df['receiving_yards']
                
                # Check for potential division by zero
                zero_total_yards = (total_yards == 0).sum()
                if zero_total_yards > 0:
                    logger.warning(f"Found {zero_total_yards} players with zero total yards - using position-based allocation")
                
                # Create a safe division function
                def safe_divide(a, b):
                    return a / b if b != 0 else 0
                
                # Initialize touchdown columns with zeros
                df['passing_touchdowns'] = 0
                df['rushing_touchdowns'] = 0
                df['receiving_touchdowns'] = 0
                
                # For players with yards, allocate touchdowns proportionally
                mask_with_yards = total_yards > 0
                if mask_with_yards.any():
                    # Calculate proportions for each type
                    passing_prop = df.loc[mask_with_yards, 'passing_yards'] / total_yards[mask_with_yards]
                    rushing_prop = df.loc[mask_with_yards, 'rushing_yards'] / total_yards[mask_with_yards]
                    receiving_prop = df.loc[mask_with_yards, 'receiving_yards'] / total_yards[mask_with_yards]
                    
                    # Allocate touchdowns based on proportions
                    df.loc[mask_with_yards, 'passing_touchdowns'] = (df.loc[mask_with_yards, 'touchdowns'] * passing_prop).round()
                    df.loc[mask_with_yards, 'rushing_touchdowns'] = (df.loc[mask_with_yards, 'touchdowns'] * rushing_prop).round()
                    df.loc[mask_with_yards, 'receiving_touchdowns'] = (df.loc[mask_with_yards, 'touchdowns'] * receiving_prop).round()
                
                # For players with zero yards but touchdowns, allocate based on position
                mask_zero_yards = (total_yards == 0) & (df['touchdowns'] > 0)
                if mask_zero_yards.any():
                    logger.info(f"Allocating touchdowns based on position for {mask_zero_yards.sum()} players")
                    
                    for idx in df[mask_zero_yards].index:
                        position = df.loc[idx, 'position'] if 'position' in df.columns else 'UNKNOWN'
                        tds = df.loc[idx, 'touchdowns']
                        
                        if position == 'QB':
                            # QBs mostly score passing TDs
                            df.loc[idx, 'passing_touchdowns'] = round(tds * 0.7)
                            df.loc[idx, 'rushing_touchdowns'] = round(tds * 0.3)
                        elif position == 'RB':
                            # RBs mostly score rushing TDs
                            df.loc[idx, 'rushing_touchdowns'] = round(tds * 0.9)
                            df.loc[idx, 'receiving_touchdowns'] = round(tds * 0.1)
                        elif position in ['WR', 'TE']:
                            # WRs and TEs mostly score receiving TDs
                            df.loc[idx, 'receiving_touchdowns'] = tds
                        else:
                            # Unknown position - split evenly
                            df.loc[idx, 'rushing_touchdowns'] = round(tds / 2)
                            df.loc[idx, 'receiving_touchdowns'] = round(tds / 2)
                
                # Verify touchdown allocation and fix any discrepancies
                total_allocated = df['passing_touchdowns'] + df['rushing_touchdowns'] + df['receiving_touchdowns']
                discrepancy = df['touchdowns'] - total_allocated
                
                # Log any discrepancies
                mismatch_count = (discrepancy != 0).sum()
                if mismatch_count > 0:
                    logger.warning(f"Touchdown allocation mismatch for {mismatch_count} players")
                    
                    # Fix discrepancies by adjusting the largest category
                    for idx in df[discrepancy != 0].index:
                        diff = discrepancy[idx]
                        
                        # Find the largest TD category to adjust
                        td_values = [
                            ('passing_touchdowns', df.loc[idx, 'passing_touchdowns']),
                            ('rushing_touchdowns', df.loc[idx, 'rushing_touchdowns']),
                            ('receiving_touchdowns', df.loc[idx, 'receiving_touchdowns'])
                        ]
                        td_values.sort(key=lambda x: x[1], reverse=True)
                        
                        # Adjust the largest category
                        largest_category = td_values[0][0]
                        df.loc[idx, largest_category] += diff
                
                # Final verification
                total_allocated = df['passing_touchdowns'] + df['rushing_touchdowns'] + df['receiving_touchdowns']
                final_mismatch = (total_allocated != df['touchdowns']).sum()
                if final_mismatch > 0:
                    logger.warning(f"Final touchdown allocation still has {final_mismatch} mismatches")
                else:
                    logger.info("Touchdown allocation completed successfully")
            
            # Calculate fantasy points for both platforms
            logger.info("Calculating fantasy points for DraftKings and FanDuel")
            df['draftkings_points'] = self.calculate_fantasy_points(df, 'draftkings')
            df['fanduel_points'] = self.calculate_fantasy_points(df, 'fanduel')
            
            # Add timestamp if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.Timestamp.now()
            
            # Save checkpoint of cleaned data
            self._save_checkpoint(df)
            
            logger.info(f"Data cleaning completed successfully: {len(df)} records processed")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning player data: {str(e)}")
            raise
            
    def calculate_fantasy_points(self, df: pd.DataFrame, platform: str = 'draftkings') -> pd.Series:
        """
        Calculate fantasy points based on platform scoring rules.
        
        Args:
            df: DataFrame containing player statistics
            platform: Fantasy platform to use for scoring ('draftkings' or 'fanduel')
            
        Returns:
            pd.Series: Series containing calculated fantasy points
        """
        try:
            platform = platform.lower()
            if platform not in self.scoring_rules:
                raise ValueError(f"Unknown platform: {platform}")
                
            logger.info(f"Calculating fantasy points for platform: {platform}")
                
            rules = self.scoring_rules[platform]
            points = pd.Series(0, index=df.index)
            
            # Calculate points for each statistic based on platform rules
            missing_stats = []
            for stat, multiplier in rules.items():
                if stat in df.columns:
                    points += df[stat] * multiplier
                else:
                    missing_stats.append(stat)
                    
            if missing_stats:
                logger.warning(f"Missing statistics for {platform} scoring: {missing_stats}")
            
            return points.round(2)
            
        except Exception as e:
            logger.error(f"Error calculating fantasy points: {str(e)}")
            return pd.Series(0, index=df.index)
            
    def _save_checkpoint(self, df: pd.DataFrame) -> None:
        """
        Save a checkpoint of the processed data.
        
        Args:
            df: DataFrame to save
        """
        try:
            checkpoint_dir = Path('data/processed')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoint_dir / f'processed_data_{timestamp}.csv'
            
            df.to_csv(checkpoint_path, index=False)
            logger.info(f"Saved data checkpoint to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            
    def get_player_trends(self, df: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
        """
        Calculate player performance trends over specified number of weeks.
        
        Args:
            df: DataFrame containing player statistics
            weeks: Number of weeks to analyze for trends
            
        Returns:
            pd.DataFrame: DataFrame containing player trends
        """
        try:
            if df.empty or 'week' not in df.columns:
                return pd.DataFrame()
                
            logger.info(f"Calculating player trends over {weeks} weeks")
                
            trends = []
            for name in df['name'].unique():
                player_data = df[df['name'] == name].sort_values('week', ascending=False)
                recent_weeks = player_data.head(weeks)
                
                if len(recent_weeks) > 0:
                    trend = {
                        'name': name,
                        'avg_draftkings_points': recent_weeks['draftkings_points'].mean(),
                        'avg_fanduel_points': recent_weeks['fanduel_points'].mean(),
                        'point_trend': recent_weeks['draftkings_points'].diff().mean(),
                        'weeks_analyzed': len(recent_weeks),
                        'last_week_points': recent_weeks.iloc[0]['draftkings_points']
                    }
                    trends.append(trend)
            
            logger.info(f"Generated trends for {len(trends)} players")
                    
            return pd.DataFrame(trends)
            
        except Exception as e:
            logger.error(f"Error calculating player trends: {str(e)}")
            return pd.DataFrame()