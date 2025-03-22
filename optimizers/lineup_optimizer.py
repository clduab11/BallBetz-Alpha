import pandas as pd
import numpy as np
import inspect
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
from datetime import datetime

# Diagnostic: Check for PuLP dependency
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimizer_diagnostics.log')
    ]
)
logger = logging.getLogger(__name__)

# Log PuLP availability
logger.info(f"=== LineupOptimizer Diagnostic Logs ===")
logger.info(f"PuLP library available: {PULP_AVAILABLE}")
if PULP_AVAILABLE:
    logger.info(f"PuLP version: {pulp.__version__}")

class LineupOptimizer:
    """
    Optimizes fantasy sports lineups using linear programming.
    
    This class implements lineup optimization for multiple fantasy platforms
    using the PuLP library for linear programming optimization.
    """
    
    def __init__(self, platform: str = 'draftkings'):
        """
        Initialize the lineup optimizer.
        
        Args:
            platform: Fantasy platform to optimize for ('draftkings' or 'fanduel')
        """
        self.platform = platform.lower()
        self.lineup_history: List[List[int]] = []
        logger.info(f"LineupOptimizer initialized for platform: {platform}")
        
        # Platform-specific settings
        self.settings = {
            'draftkings': {
                'salary_cap': 50000,
                'positions': ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST'],
                'constraints': {
                    'QB': 1,
                    'RB': 2,
                    'WR': 3,
                    'TE': 1,
                    'FLEX': 1,
                    'DST': 1
                },
                'flex_positions': ['RB', 'WR', 'TE']
            },
            'fanduel': {
                'salary_cap': 60000,
                'positions': ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DEF'],
                'constraints': {
                    'QB': 1,
                    'RB': 2,
                    'WR': 3,
                    'TE': 1,
                    'FLEX': 1,
                    'DEF': 1
                },
                'flex_positions': ['RB', 'WR', 'TE']
            }
        }
        
    def optimize(self, 
                player_pool: pd.DataFrame, 
                salary_cap: Optional[int] = None, 
                min_salary: Optional[float] = None,
                max_lineups: int = 1,
                min_lineup_diff: int = 3,
                min_stack_size: int = 2) -> pd.DataFrame:
        """
        Generate optimal lineups from a pool of players.
        
        Args:
            player_pool: DataFrame containing available players and their details
            salary_cap: Optional custom salary cap (defaults to platform standard)
            min_salary: Optional minimum salary threshold for players to include
            max_lineups: Number of unique lineups to generate
            min_lineup_diff: Minimum number of different players between lineups
            min_stack_size: Minimum size for team stacks
            
        Returns:
            pd.DataFrame: Optimized lineups
        """
        try:
            # Diagnostic logging
            logger.info(f"=== DIAGNOSTIC: optimize method called ===")
            logger.info(f"Method signature: {inspect.signature(LineupOptimizer.optimize)}")
            logger.info(f"Received parameters: player_pool shape={player_pool.shape}, "
                       f"salary_cap={salary_cap}, min_salary={min_salary}, "
                       f"max_lineups={max_lineups}, min_lineup_diff={min_lineup_diff}, "
                       f"min_stack_size={min_stack_size}")
            
            # Check for required columns in player_pool
            required_cols = ['name', 'position', 'team', 'salary', 'predicted_points']
            missing_cols = [col for col in required_cols if col not in player_pool.columns]
            if missing_cols:
                logger.error(f"Missing required columns in player_pool: {missing_cols}")
                logger.info(f"Available columns: {player_pool.columns.tolist()}")
                return pd.DataFrame()
            
            if self.platform not in self.settings:
                logger.error(f"Unsupported platform: {self.platform}")
                return pd.DataFrame()
                
            # Use default salary cap if none provided
            if salary_cap is None:
                salary_cap = self.settings[self.platform]['salary_cap']
                logger.info(f"Using default salary cap: ${salary_cap:,}")
            
            # Handle min_salary parameter if provided (diagnostic fix)
            if min_salary is not None:
                logger.info(f"Filtering player pool by minimum salary: ${min_salary}")
                if 'salary' in player_pool.columns:
                    original_size = len(player_pool)
                    player_pool = player_pool[player_pool['salary'] >= min_salary]
                    logger.info(f"Filtered player pool by min_salary: {original_size} -> {len(player_pool)} players")
                else:
                    logger.warning("Cannot filter by min_salary: 'salary' column not found in player pool")
                
            logger.info(f"Optimizing {max_lineups} lineups for {self.platform}")
            logger.info(f"Salary cap: ${salary_cap:,}")
            
            return self._optimize_lineups(
                player_pool, 
                salary_cap, 
                min_salary,  # Pass through for diagnostic purposes
                max_lineups,
                min_lineup_diff,
                min_stack_size
            )
            
        except Exception as e:
            logger.error(f"Error in optimize: {str(e)}")
            return pd.DataFrame(columns=[
                'name', 'team', 'position', 'salary', 'predicted_points'
            ])
            
    def _optimize_lineups(self,
                         player_pool: pd.DataFrame,
                         salary_cap: int,
                         min_salary: Optional[float],
                         max_lineups: int,
                         min_lineup_diff: int,
                         min_stack_size: int) -> pd.DataFrame:
        """
        Core optimization logic for generating lineups.
        """
        try:
            # Check if PuLP is available
            if not PULP_AVAILABLE:
                logger.error("PuLP library not available - cannot optimize lineups")
                return pd.DataFrame(columns=[
                    'name', 'team', 'position', 'salary', 'predicted_points'
                ])
                
            # Filter and prepare player pool
            player_pool = self._prepare_player_pool(player_pool)
            if player_pool.empty:
                logger.error("Prepared player pool is empty - cannot optimize lineup")
                return pd.DataFrame(columns=[
                    'name', 'team', 'position', 'salary', 'predicted_points'
                ])
                
            platform_settings = self.settings[self.platform]
            all_lineups = []
            
            for lineup_count in range(max_lineups):
                # Create optimization problem
                prob = self._create_optimization_problem(
                    f"{self.platform}_Lineup_{lineup_count}",
                    player_pool,
                    salary_cap,
                    platform_settings,
                    min_lineup_diff,
                    min_stack_size
                )
                
                # Solve the problem
                status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
                
                if status == pulp.LpStatusOptimal:
                    lineup = self._extract_lineup(prob, player_pool)
                    if not lineup.empty:
                        lineup['lineup_number'] = lineup_count + 1
                        all_lineups.append(lineup)
                        self.lineup_history.append(lineup.index.tolist())
                        
                        # Save checkpoint
                        self._save_lineup_checkpoint(lineup)
                else:
                    logger.warning(f"Could not find optimal solution for lineup {lineup_count + 1}")
                    empty_lineup = pd.DataFrame(columns=[
                        'name', 'team', 'position', 'salary', 'predicted_points'
                    ])
                    return empty_lineup
                    
            if all_lineups:
                result_df = pd.concat(all_lineups)
                result_df = self._add_lineup_metadata(result_df)
                return result_df
            
            # Fix: Return empty DataFrame with correct columns if no lineups were created
            return pd.DataFrame(columns=[
                'name', 'team', 'position', 'salary', 'predicted_points'
            ])
            
        except Exception as e:
            logger.error(f"Error in _optimize_lineups: {str(e)}")
            return pd.DataFrame(columns=[
                'name', 'team', 'position', 'salary', 'predicted_points'
            ])
            
    def _prepare_player_pool(self, player_pool: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and validate the player pool for optimization.
        
        Args:
            player_pool: Raw player pool DataFrame
            
        Returns:
            pd.DataFrame: Cleaned and validated player pool
        """
        try:
            # Filter out players without predictions or salaries
            logger.info(f"Original player pool size: {len(player_pool)}")
            
            # Check if required columns exist
            if 'predicted_points' not in player_pool.columns:
                logger.error("'predicted_points' column missing from player pool")
                # Add placeholder predicted points for testing
                player_pool['predicted_points'] = player_pool.get('salary', 5000) / 1000
                logger.warning("Added placeholder predicted_points based on salary for testing")
                
            if 'salary' not in player_pool.columns:
                logger.error("'salary' column missing from player pool")
                logger.info(f"Available columns: {player_pool.columns.tolist()}")
                # Add placeholder salary for testing
                player_pool['salary'] = 5000
                logger.warning("Added placeholder salary data (5000) for testing")
            
            # Ensure position column exists
            if 'position' not in player_pool.columns:
                logger.error("'position' column missing from player pool")
                # Try to infer position from player name or other data
                # For now, assign a default position
                player_pool['position'] = 'FLEX'
                logger.warning("Added placeholder position data (FLEX) for testing")
            
            # Ensure team column exists
            if 'team' not in player_pool.columns:
                logger.error("'team' column missing from player pool")
                # Assign a default team
                player_pool['team'] = 'UNK'
                logger.warning("Added placeholder team data (UNK) for testing")
            
            pool = player_pool[
                (player_pool['predicted_points'] > 0) &
                (player_pool['salary'] > 0)
            ].copy()
            logger.info(f"Filtered player pool size: {len(pool)}")
            
            # Add risk-adjusted projections
            if 'prediction_lower' in pool.columns and 'prediction_upper' in pool.columns:
                pool['risk_adjusted_points'] = (
                    pool['predicted_points'] - 
                    (pool['prediction_upper'] - pool['prediction_lower']) * 0.5
                )
            else:
                pool['risk_adjusted_points'] = pool['predicted_points']
                
            logger.info(f"Final prepared player pool size: {len(pool)}")
            return pool
            
        except Exception as e:
            logger.error(f"Error preparing player pool: {str(e)}")
            return pd.DataFrame()
            
    def _create_optimization_problem(self,
                                   name: str,
                                   player_pool: pd.DataFrame,
                                   salary_cap: int,
                                   settings: Dict,
                                   min_lineup_diff: int,
                                   min_stack_size: int) -> pulp.LpProblem:
        """
        Create the linear programming optimization problem.
        
        Args:
            name: Name for the optimization problem
            player_pool: Prepared player pool
            salary_cap: Maximum salary cap
            settings: Platform-specific settings
            min_lineup_diff: Minimum lineup difference
            min_stack_size: Minimum stack size
            
        Returns:
            pulp.LpProblem: Configured optimization problem
        """
        # Create problem
        prob = pulp.LpProblem(name, pulp.LpMaximize)
        
        # Create player variables
        player_vars = pulp.LpVariable.dicts(
            "player",
            player_pool.index,
            cat='Binary'
        )
        
        # Objective function (maximize projected points)
        prob += pulp.lpSum([
            player_pool.loc[i, 'risk_adjusted_points'] * player_vars[i]
            for i in player_pool.index
        ])
        
        # Salary constraint
        prob += pulp.lpSum([
            player_pool.loc[i, 'salary'] * player_vars[i]
            for i in player_pool.index
        ]) <= salary_cap
        
        # Position constraints
        for position, count in settings['constraints'].items():
            if position == 'FLEX':
                # FLEX can be RB, WR, or TE
                flex_players = player_pool[
                    player_pool['position'].isin(settings['flex_positions'])
                ].index
                prob += pulp.lpSum([player_vars[i] for i in flex_players]) >= (
                    count + sum([
                        settings['constraints'][pos] 
                        for pos in settings['flex_positions']
                    ])
                )
            else:
                # Regular position constraint
                position_players = player_pool[
                    player_pool['position'] == position
                ].index
                prob += pulp.lpSum([player_vars[i] for i in position_players]) == count
                
        # Team stack constraints
        if min_stack_size > 1:
            self._add_stacking_constraints(
                prob,
                player_vars,
                player_pool,
                min_stack_size
            )
        
        # Previous lineup constraints
        if self.lineup_history:
            self._add_lineup_difference_constraints(
                prob,
                player_vars,
                min_lineup_diff
            )
            
        return prob
        
    def _add_stacking_constraints(self,
                                prob: pulp.LpProblem,
                                player_vars: Dict,
                                player_pool: pd.DataFrame,
                                min_stack_size: int) -> None:
        """
        Add team stacking constraints to the optimization problem.
        
        Args:
            prob: Optimization problem
            player_vars: Player decision variables
            player_pool: Player pool DataFrame
            min_stack_size: Minimum stack size
        """
        # Create team stack variables
        teams = player_pool['team'].unique()
        team_vars = pulp.LpVariable.dicts(
            "team_stack",
            teams,
            cat='Binary'
        )
        
        # Ensure minimum stack size when team is selected
        for team in teams:
            team_players = player_pool[player_pool['team'] == team].index
            prob += pulp.lpSum([player_vars[i] for i in team_players]) >= (
                min_stack_size * team_vars[team]
            )
            
        # Ensure at least one team stack
        prob += pulp.lpSum(team_vars.values()) >= 1
        
    def _add_lineup_difference_constraints(self,
                                         prob: pulp.LpProblem,
                                         player_vars: Dict,
                                         min_diff: int) -> None:
        """
        Add constraints to ensure lineup diversity.
        
        Args:
            prob: Optimization problem
            player_vars: Player decision variables
            min_diff: Minimum number of different players
        """
        for prev_lineup in self.lineup_history:
            prob += pulp.lpSum([player_vars[i] for i in prev_lineup]) <= (
                len(prev_lineup) - min_diff
            )
            
    def _extract_lineup(self,
                       prob: pulp.LpProblem,
                       player_pool: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the optimal lineup from solved problem.
        
        Args:
            prob: Solved optimization problem
            player_pool: Player pool DataFrame
            
        Returns:
            pd.DataFrame: Selected lineup
        """
        try:
            selected_players = []
            for v in prob.variables():
                if v.name.startswith('player_') and v.value() == 1:
                    idx = int(v.name.split('_')[1])
                    selected_players.append(idx)
                    
            return player_pool.loc[selected_players].copy()
            
        except Exception as e:
            logger.error(f"Error extracting lineup: {str(e)}")
            return pd.DataFrame()
            
    def _add_lineup_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add metadata to lineup DataFrame.
        
        Args:
            df: Lineup DataFrame
            
        Returns:
            pd.DataFrame: Enhanced lineup DataFrame
        """
        df = df.copy()
        df['total_salary'] = df.groupby('lineup_number')['salary'].transform('sum')
        df['total_projected'] = df.groupby('lineup_number')['predicted_points'].transform('sum')
        df['salary_remaining'] = self.settings[self.platform]['salary_cap'] - df['total_salary']
        df['timestamp'] = pd.Timestamp.now()
        return df
        
    def _save_lineup_checkpoint(self, lineup: pd.DataFrame) -> None:
        """
        Save a checkpoint of the generated lineup.
        
        Args:
            lineup: Lineup DataFrame to save
        """
        try:
            checkpoint_dir = Path('optimizers/checkpoints')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lineup_num = lineup['lineup_number'].iloc[0]
            checkpoint_path = checkpoint_dir / f'lineup_{lineup_num}_{timestamp}.csv'
            
            lineup.to_csv(checkpoint_path, index=True)
            logger.info(f"Saved lineup checkpoint to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving lineup checkpoint: {str(e)}")