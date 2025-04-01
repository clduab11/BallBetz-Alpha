"""Data sharing settings database operations for BallBetz-Alpha.

This module handles data sharing settings database operations.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
from postgrest import APIError

from .base import DatabaseInterface, QueryError

# Set up logging
logger = logging.getLogger(__name__)


class DataSharingDatabaseOperations:
    """Data sharing settings database operations."""
    
    def __init__(self, db: DatabaseInterface):
        """Initialize with a database interface.
        
        Args:
            db: Database interface to use
        """
        self.db = db
    
    async def get_user_data_sharing_settings(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's data sharing settings.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of data sharing settings
        """
        try:
            response = await self.db.client.table('user_data_sharing_settings').select('*').eq('user_id', user_id).execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get data sharing settings: {str(e)}")
            return []
    
    async def update_user_data_sharing_settings(self, user_id: str, settings: Dict[str, bool]) -> bool:
        """Update user's data sharing settings.
        
        Args:
            user_id: ID of the user
            settings: Dictionary of setting names and enabled flags
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update each setting
            for setting_name, is_enabled in settings.items():
                await self.db.client.table('user_data_sharing_settings').update({
                    'is_enabled': is_enabled
                }).eq('user_id', user_id).eq('setting_name', setting_name).execute()
            
            # Calculate new data sharing level
            response = await self.db.client.table('user_data_sharing_settings').select('is_enabled').eq('user_id', user_id).execute()
            enabled_count = sum(1 for setting in response.data if setting['is_enabled'])
            
            # Determine data sharing level based on enabled settings count
            if enabled_count >= 5:
                new_level = 3  # Comprehensive
            elif enabled_count >= 3:
                new_level = 2  # Moderate
            else:
                new_level = 1  # Basic
            
            # Update user profile with new data sharing level
            from .rewards_operations import RewardsDatabaseOperations
            rewards_ops = RewardsDatabaseOperations(self.db)
            await rewards_ops.update_user_rewards_data(user_id, {'data_sharing_level': new_level})
            
            # Check if tier upgrade is needed
            await rewards_ops.check_rewards_tier_change(user_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update data sharing settings: {str(e)}")
            return False
    
    async def initialize_user_data_sharing_settings(self, user_id: str) -> bool:
        """Initialize default data sharing settings for a new user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Define default settings
            default_settings = [
                {'setting_name': 'profile_visibility', 'is_enabled': True, 'description': 'Allow others to view your profile'},
                {'setting_name': 'betting_history', 'is_enabled': False, 'description': 'Share your betting history for analysis'},
                {'setting_name': 'favorite_teams', 'is_enabled': True, 'description': 'Share your favorite teams'},
                {'setting_name': 'betting_patterns', 'is_enabled': False, 'description': 'Allow analysis of your betting patterns'},
                {'setting_name': 'platform_usage', 'is_enabled': True, 'description': 'Share platform usage statistics'},
                {'setting_name': 'performance_stats', 'is_enabled': False, 'description': 'Share your performance statistics'}
            ]
            
            # Insert settings
            for setting in default_settings:
                await self.db.client.table('user_data_sharing_settings').insert({
                    'user_id': user_id,
                    'setting_name': setting['setting_name'],
                    'is_enabled': setting['is_enabled'],
                    'description': setting['description']
                }).execute()
            
            # Set initial data sharing level
            from .rewards_operations import RewardsDatabaseOperations
            rewards_ops = RewardsDatabaseOperations(self.db)
            await rewards_ops.update_user_rewards_data(user_id, {'data_sharing_level': 1})
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize data sharing settings: {str(e)}")
            return False
    
    async def get_data_sharing_level_description(self, level: int) -> str:
        """Get description for a data sharing level.
        
        Args:
            level: Data sharing level (1-3)
            
        Returns:
            Description of the level
        """
        descriptions = {
            1: "Basic - Minimal data sharing for essential functionality",
            2: "Moderate - Balanced data sharing for improved experience",
            3: "Comprehensive - Full data sharing for maximum benefits"
        }
        return descriptions.get(level, "Unknown level")
    
    async def get_data_sharing_statistics(self) -> Dict[str, Any]:
        """Get statistics about data sharing across users.
        
        Returns:
            Dictionary of statistics
        """
        try:
            # Get counts of users at each level
            response = await self.db.client.table('user_profiles').select('data_sharing_level').execute()
            
            if not response.data:
                return {
                    'total_users': 0,
                    'level_counts': {1: 0, 2: 0, 3: 0},
                    'level_percentages': {1: 0, 2: 0, 3: 0}
                }
            
            # Count users at each level
            level_counts = {1: 0, 2: 0, 3: 0}
            for user in response.data:
                level = user.get('data_sharing_level', 1)
                level_counts[level] = level_counts.get(level, 0) + 1
            
            total_users = len(response.data)
            
            # Calculate percentages
            level_percentages = {
                level: (count / total_users * 100) if total_users > 0 else 0
                for level, count in level_counts.items()
            }
            
            return {
                'total_users': total_users,
                'level_counts': level_counts,
                'level_percentages': level_percentages
            }
        except Exception as e:
            logger.error(f"Failed to get data sharing statistics: {str(e)}")
            return {
                'total_users': 0,
                'level_counts': {1: 0, 2: 0, 3: 0},
                'level_percentages': {1: 0, 2: 0, 3: 0}
            }