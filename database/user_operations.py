"""User database operations for BallBetz-Alpha.

This module handles user-related database operations.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone, timedelta
import jwt
from postgrest import APIError
from flask import current_app

from .base import DatabaseInterface, QueryError

# Set up logging
logger = logging.getLogger(__name__)


class UserDatabaseOperations:
    """User-related database operations."""
    
    def __init__(self, db: DatabaseInterface):
        """Initialize with a database interface.
        
        Args:
            db: Database interface to use
        """
        self.db = db
    
    async def create_user_profile(self, user_data: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
        """Create a new user profile in the database.
        
        Args:
            user_data: Dictionary containing user data
            
        Returns:
            Tuple of (user_profile, error_message)
        """
        try:
            return await self.db.create_user_profile(user_data)
        except Exception as e:
            logger.error(f"Failed to create user profile: {str(e)}")
            return None, str(e)
    
    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Retrieve a user by email.
        
        Args:
            email: Email address to look up
            
        Returns:
            User data dictionary or None if not found
        """
        try:
            return await self.db.get_user_by_email(email)
        except Exception as e:
            logger.error(f"Failed to get user by email: {str(e)}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Retrieve a user by ID.
        
        Args:
            user_id: User ID to look up
            
        Returns:
            User data dictionary or None if not found
        """
        try:
            return await self.db.get_user_by_id(user_id)
        except Exception as e:
            logger.error(f"Failed to get user by ID: {str(e)}")
            return None
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Update a user's profile.
        
        Args:
            user_id: ID of the user to update
            updates: Dictionary of fields to update
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            return await self.db.update_user_profile(user_id, updates)
        except Exception as e:
            logger.error(f"Failed to update user profile: {str(e)}")
            return False, str(e)
    
    async def create_password_reset_token(self, user_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Create a password reset token.
        
        Args:
            user_id: ID of the user requesting password reset
            
        Returns:
            Tuple of (token, error_message)
        """
        try:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
            token = jwt.encode(
                {
                    'user_id': user_id,
                    'exp': int(expires_at.timestamp())
                },
                current_app.config['SUPABASE_JWT_SECRET'],
                algorithm='HS256'
            )

            response = await self.db.client.table('password_reset_tokens').insert({
                'user_id': user_id,
                'token': token,
                'expires_at': expires_at.isoformat()
            }).execute()

            if not response.data:
                return None, "Failed to create reset token"

            # Create audit log
            await self.db.create_audit_log(user_id, 'password_reset_requested', {})

            return token, None
        except Exception as e:
            logger.error(f"Failed to create reset token: {str(e)}")
            return None, str(e)
    
    async def validate_reset_token(self, token: str) -> Tuple[Optional[str], Optional[str]]:
        """Validate a password reset token.
        
        Args:
            token: Token to validate
            
        Returns:
            Tuple of (user_id, error_message)
        """
        try:
            response = await self.db.client.table('password_reset_tokens').select('*').eq('token', token).execute()
            if not response.data:
                return None, "Invalid token"

            token_data = response.data[0]
            expires_at = datetime.fromisoformat(token_data['expires_at'].replace('Z', '+00:00'))
            
            if datetime.now(timezone.utc) >= expires_at:
                return None, "Token expired"
            
            if token_data['used_at']:
                return None, "Token already used"

            return token_data['user_id'], None
        except Exception as e:
            logger.error(f"Failed to validate reset token: {str(e)}")
            return None, str(e)
    
    async def mark_reset_token_used(self, token: str) -> bool:
        """Mark a password reset token as used.
        
        Args:
            token: Token to mark as used
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = await self.db.client.table('password_reset_tokens').update({
                'used_at': datetime.now(timezone.utc).isoformat()
            }).eq('token', token).execute()
            
            return bool(response.data)
        except Exception as e:
            logger.error(f"Failed to mark token as used: {str(e)}")
            return False
    
    async def record_failed_login(self, user_id: str) -> Tuple[int, Optional[datetime]]:
        """Record a failed login attempt.
        
        Args:
            user_id: ID of the user who failed to log in
            
        Returns:
            Tuple of (attempts_count, locked_until)
        """
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return 0, None

            attempts = user['failed_login_attempts'] + 1
            updates = {'failed_login_attempts': attempts}

            if attempts >= current_app.config['MAX_LOGIN_ATTEMPTS']:
                locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
                updates['locked_until'] = locked_until.isoformat()
            
            await self.update_user_profile(user_id, updates)
            
            # Create audit log
            await self.db.create_audit_log(
                user_id,
                'failed_login_attempt',
                {'attempts': attempts}
            )

            return attempts, locked_until if 'locked_until' in updates else None
        except Exception as e:
            logger.error(f"Failed to record failed login: {str(e)}")
            return 0, None
    
    async def reset_failed_attempts(self, user_id: str) -> bool:
        """Reset failed login attempts counter.
        
        Args:
            user_id: ID of the user to reset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success, _ = await self.update_user_profile(user_id, {
                'failed_login_attempts': 0,
                'locked_until': None
            })
            return success
        except Exception as e:
            logger.error(f"Failed to reset failed attempts: {str(e)}")
            return False