"""Supabase database client for BallBetz-Alpha.

This module implements the database interface using Supabase.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone
from postgrest import APIError
from supabase import create_client, Client
from flask import current_app, request

from .base import DatabaseInterface, ConnectionError, QueryError

# Set up logging
logger = logging.getLogger(__name__)


class SupabaseClient(DatabaseInterface):
    """Supabase implementation of the database interface."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.client: Optional[Client] = None
        self.connect()
    
    def connect(self) -> None:
        """Create Supabase client connection."""
        try:
            self.client = create_client(
                current_app.config['SUPABASE_URL'],
                current_app.config['SUPABASE_KEY']
            )
            logger.info("Connected to Supabase")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {str(e)}")
            raise ConnectionError(f"Failed to connect to Supabase: {str(e)}")
    
    async def create_user_profile(self, user_data: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
        """Create a new user profile in the database.
        
        Args:
            user_data: Dictionary containing user data
            
        Returns:
            Tuple of (user_profile, error_message)
        """
        try:
            response = await self.client.table('user_profiles').insert({
                'id': user_data['id'],
                'username': user_data['username'],
                'email': user_data['email'],
                'role': user_data['role']
            }).execute()

            if not response.data:
                return None, "Failed to create user profile"

            # Create audit log
            await self.create_audit_log(
                user_data['id'],
                'user_created',
                {'username': user_data['username']}
            )

            return response.data[0], None
        except APIError as e:
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
            response = await self.client.table('user_profiles').select('*').eq('email', email).execute()
            return response.data[0] if response.data else None
        except APIError as e:
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
            response = await self.client.table('user_profiles').select('*').eq('id', user_id).execute()
            return response.data[0] if response.data else None
        except APIError as e:
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
            response = await self.client.table('user_profiles').update(updates).eq('id', user_id).execute()
            
            if not response.data:
                return False, "User not found"

            # Create audit log
            await self.create_audit_log(
                user_id,
                'profile_updated',
                {'updates': {k: v for k, v in updates.items() if k != 'password'}}
            )

            return True, None
        except APIError as e:
            logger.error(f"Failed to update user profile: {str(e)}")
            return False, str(e)
    
    async def create_session(self, user_id: str, session_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Create a new user session.
        
        Args:
            user_id: ID of the user
            session_data: Dictionary of session data
            
        Returns:
            Tuple of (session_id, error_message)
        """
        try:
            expires_at = datetime.now(timezone.utc) + timedelta(days=1)
            if session_data.get('remember_me'):
                expires_at = datetime.now(timezone.utc) + timedelta(days=30)

            response = await self.client.table('user_sessions').insert({
                'user_id': user_id,
                'device_id': session_data.get('device_id'),
                'ip_address': session_data.get('ip_address'),
                'user_agent': session_data.get('user_agent'),
                'expires_at': expires_at.isoformat(),
                'remember_me': session_data.get('remember_me', False)
            }).execute()

            if not response.data:
                return None, "Failed to create session"

            return response.data[0]['id'], None
        except APIError as e:
            logger.error(f"Failed to create session: {str(e)}")
            return None, str(e)
    
    async def validate_session(self, session_id: str) -> bool:
        """Validate a user session.
        
        Args:
            session_id: ID of the session to validate
            
        Returns:
            True if session is valid, False otherwise
        """
        try:
            response = await self.client.table('user_sessions').select('*').eq('id', session_id).execute()
            if not response.data:
                return False

            session = response.data[0]
            expires_at = datetime.fromisoformat(session['expires_at'].replace('Z', '+00:00'))
            
            return datetime.now(timezone.utc) < expires_at
        except APIError as e:
            logger.error(f"Failed to validate session: {str(e)}")
            return False
    
    async def create_audit_log(self, user_id: str, action: str, details: Dict[str, Any]) -> bool:
        """Create an audit log entry.
        
        Args:
            user_id: ID of the user
            action: Action performed
            details: Dictionary of action details
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = await self.client.table('audit_logs').insert({
                'user_id': user_id,
                'action': action,
                'details': details,
                'ip_address': request.remote_addr if request else None,
                'user_agent': request.user_agent.string if request and request.user_agent else None
            }).execute()
            
            return bool(response.data)
        except APIError as e:
            logger.error(f"Failed to create audit log: {str(e)}")
            return False


# Create a singleton instance
db = SupabaseClient()