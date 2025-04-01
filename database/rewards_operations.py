"""Rewards database operations for BallBetz-Alpha.

This module handles rewards-related database operations.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone
from postgrest import APIError

from .base import DatabaseInterface, QueryError

# Set up logging
logger = logging.getLogger(__name__)


class RewardsDatabaseOperations:
    """Rewards-related database operations."""
    
    def __init__(self, db: DatabaseInterface):
        """Initialize with a database interface.
        
        Args:
            db: Database interface to use
        """
        self.db = db
    
    async def get_user_rewards_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's rewards data.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary of rewards data or None if not found
        """
        try:
            response = await self.db.client.table('user_profiles').select(
                'rewards_tier', 'rewards_points', 'data_sharing_level'
            ).eq('id', user_id).execute()
            
            if not response.data:
                return None
                
            return response.data[0]
        except Exception as e:
            logger.error(f"Failed to get user rewards data: {str(e)}")
            return None
    
    async def update_user_rewards_data(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update a user's rewards data.
        
        Args:
            user_id: ID of the user to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            allowed_fields = {'rewards_tier', 'rewards_points', 'data_sharing_level'}
            filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
            
            if not filtered_updates:
                return False
                
            response = await self.db.client.table('user_profiles').update(
                filtered_updates
            ).eq('id', user_id).execute()
            
            return bool(response.data)
        except Exception as e:
            logger.error(f"Failed to update user rewards data: {str(e)}")
            return False
    
    async def add_rewards_points(self, user_id: str, points: int, action: str, description: str = None) -> bool:
        """Add rewards points to a user's account and record in history.
        
        Args:
            user_id: ID of the user
            points: Number of points to add
            action: The action that earned the points
            description: Optional description
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current points
            user_data = await self.get_user_rewards_data(user_id)
            if not user_data:
                return False
                
            current_points = user_data.get('rewards_points', 0)
            new_points = current_points + points
            
            # Update user profile
            await self.update_user_rewards_data(user_id, {'rewards_points': new_points})
            
            # Record in history
            await self.db.client.table('rewards_points_history').insert({
                'user_id': user_id,
                'points': points,
                'action': action,
                'description': description
            }).execute()
            
            # Check if tier upgrade is needed
            await self.check_rewards_tier_change(user_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add rewards points: {str(e)}")
            return False
    
    async def redeem_rewards_points(self, user_id: str, points: int, reward_type: str, reward_description: str = None) -> bool:
        """Redeem rewards points for a reward.
        
        Args:
            user_id: ID of the user
            points: Number of points to redeem
            reward_type: Type of reward
            reward_description: Optional description
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current points
            user_data = await self.get_user_rewards_data(user_id)
            if not user_data:
                return False
                
            current_points = user_data.get('rewards_points', 0)
            
            # Check if user has enough points
            if current_points < points:
                return False
                
            # Update user's points
            new_points = current_points - points
            await self.update_user_rewards_data(user_id, {'rewards_points': new_points})
            
            # Record redemption
            await self.db.client.table('rewards_redemption_history').insert({
                'user_id': user_id,
                'points_spent': points,
                'reward_type': reward_type,
                'reward_description': reward_description
            }).execute()
            
            # Check if tier downgrade is needed
            await self.check_rewards_tier_change(user_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to redeem rewards points: {str(e)}")
            return False
    
    async def check_rewards_tier_change(self, user_id: str) -> bool:
        """Check if user's rewards tier should change based on points and data sharing level.
        
        Args:
            user_id: ID of the user
            
        Returns:
            True if tier changed, False otherwise
        """
        try:
            # Get user data
            user_data = await self.get_user_rewards_data(user_id)
            if not user_data:
                return False
                
            current_points = user_data.get('rewards_points', 0)
            current_tier = user_data.get('rewards_tier', 'rookie')
            data_sharing_level = user_data.get('data_sharing_level', 1)
            
            # Get tier requirements
            response = await self.db.client.table('rewards_tier_requirements').select('*').execute()
            if not response.data:
                return False
                
            # Find appropriate tier
            appropriate_tier = 'rookie'
            for tier_req in response.data:
                if (current_points >= tier_req['min_points'] and 
                    data_sharing_level >= tier_req['min_data_sharing_level']):
                    if tier_req['tier'] in ['mvp', 'all_star', 'rookie'] and tier_req['tier'] > appropriate_tier:
                        appropriate_tier = tier_req['tier']
            
            # Update tier if needed
            if current_tier != appropriate_tier:
                await self.update_user_rewards_data(user_id, {'rewards_tier': appropriate_tier})
                
                # Record tier change in audit log
                await self.db.create_audit_log(
                    user_id,
                    'rewards_tier_changed',
                    {
                        'previous_tier': current_tier,
                        'new_tier': appropriate_tier
                    }
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to check rewards tier change: {str(e)}")
            return False
    
    async def get_rewards_tier_benefits(self, tier: str = None) -> List[Dict[str, Any]]:
        """Get benefits for a specific tier or all tiers.
        
        Args:
            tier: Optional tier to filter by
            
        Returns:
            List of tier benefits
        """
        try:
            if tier:
                response = await self.db.client.table('rewards_tier_benefits').select('*').eq('tier', tier).execute()
            else:
                response = await self.db.client.table('rewards_tier_benefits').select('*').execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get tier benefits: {str(e)}")
            return []
    
    async def get_rewards_tier_requirements(self, tier: str = None) -> List[Dict[str, Any]]:
        """Get requirements for a specific tier or all tiers.
        
        Args:
            tier: Optional tier to filter by
            
        Returns:
            List of tier requirements
        """
        try:
            if tier:
                response = await self.db.client.table('rewards_tier_requirements').select('*').eq('tier', tier).execute()
            else:
                response = await self.db.client.table('rewards_tier_requirements').select('*').execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get tier requirements: {str(e)}")
            return []
    
    async def get_rewards_points_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get points history for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of points history entries
        """
        try:
            response = await self.db.client.table('rewards_points_history').select('*').eq('user_id', user_id).order('created_at', {'ascending': False}).execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get points history: {str(e)}")
            return []
    
    async def get_rewards_redemption_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get redemption history for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of redemption history entries
        """
        try:
            response = await self.db.client.table('rewards_redemption_history').select('*').eq('user_id', user_id).order('created_at', {'ascending': False}).execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get redemption history: {str(e)}")
            return []