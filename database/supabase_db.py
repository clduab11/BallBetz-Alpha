from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone, timedelta
import jwt
from postgrest import APIError
from supabase import create_client, Client
from flask import current_app
import logging

logger = logging.getLogger(__name__)

class SupabaseDB:
    """Interface for Supabase database operations."""
    
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
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {str(e)}")
            raise

    async def create_user_profile(self, user_data: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
        """Create a new user profile in the database."""
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
        """Retrieve a user by email."""
        try:
            response = await self.client.table('user_profiles').select('*').eq('email', email).execute()
            return response.data[0] if response.data else None
        except APIError as e:
            logger.error(f"Failed to get user by email: {str(e)}")
            return None

    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Update a user's profile."""
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
        """Create a new user session."""
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
        """Validate a user session."""
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

    async def create_password_reset_token(self, user_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Create a password reset token."""
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

            response = await self.client.table('password_reset_tokens').insert({
                'user_id': user_id,
                'token': token,
                'expires_at': expires_at.isoformat()
            }).execute()

            if not response.data:
                return None, "Failed to create reset token"

            # Create audit log
            await self.create_audit_log(user_id, 'password_reset_requested', {})

            return token, None
        except APIError as e:
            logger.error(f"Failed to create reset token: {str(e)}")
            return None, str(e)

    async def validate_reset_token(self, token: str) -> Tuple[Optional[str], Optional[str]]:
        """Validate a password reset token."""
        try:
            response = await self.client.table('password_reset_tokens').select('*').eq('token', token).execute()
            if not response.data:
                return None, "Invalid token"

            token_data = response.data[0]
            expires_at = datetime.fromisoformat(token_data['expires_at'].replace('Z', '+00:00'))
            
            if datetime.now(timezone.utc) >= expires_at:
                return None, "Token expired"
            
            if token_data['used_at']:
                return None, "Token already used"

            return token_data['user_id'], None
        except APIError as e:
            logger.error(f"Failed to validate reset token: {str(e)}")
            return None, str(e)

    async def mark_reset_token_used(self, token: str) -> bool:
        """Mark a password reset token as used."""
        try:
            response = await self.client.table('password_reset_tokens').update({
                'used_at': datetime.now(timezone.utc).isoformat()
            }).eq('token', token).execute()
            
            return bool(response.data)
        except APIError as e:
            logger.error(f"Failed to mark token as used: {str(e)}")
            return False

    async def record_failed_login(self, user_id: str) -> Tuple[int, Optional[datetime]]:
        """Record a failed login attempt."""
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
            await self.create_audit_log(
                user_id,
                'failed_login_attempt',
                {'attempts': attempts}
            )

            return attempts, locked_until if 'locked_until' in updates else None
        except APIError as e:
            logger.error(f"Failed to record failed login: {str(e)}")
            return 0, None

    async def reset_failed_attempts(self, user_id: str) -> bool:
        """Reset failed login attempts counter."""
        try:
            response = await self.client.table('user_profiles').update({
                'failed_login_attempts': 0,
                'locked_until': None
            }).eq('id', user_id).execute()
            
            return bool(response.data)
        except APIError as e:
            logger.error(f"Failed to reset failed attempts: {str(e)}")
            return False

    async def create_audit_log(self, user_id: str, action: str, details: Dict[str, Any]) -> bool:
        """Create an audit log entry."""
        try:
            response = await self.client.table('audit_logs').insert({
                'user_id': user_id,
                'action': action,
                'details': details,
                'ip_address': current_app.request.remote_addr if current_app.request else None,
                'user_agent': current_app.request.user_agent.string if current_app.request else None
            }).execute()
            
            return bool(response.data)
        except APIError as e:
            logger.error(f"Failed to create audit log: {str(e)}")
            return False
            
    async def get_user_rewards_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's rewards data."""
        try:
            response = await self.client.table('user_profiles').select(
                'rewards_tier', 'rewards_points', 'data_sharing_level'
            ).eq('id', user_id).execute()
            
            if not response.data:
                return None
                
            return response.data[0]
        except APIError as e:
            logger.error(f"Failed to get user rewards data: {str(e)}")
            return None
            
    async def update_user_rewards_data(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update a user's rewards data."""
        try:
            allowed_fields = {'rewards_tier', 'rewards_points', 'data_sharing_level'}
            filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
            
            if not filtered_updates:
                return False
                
            response = await self.client.table('user_profiles').update(
                filtered_updates
            ).eq('id', user_id).execute()
            
            return bool(response.data)
        except APIError as e:
            logger.error(f"Failed to update user rewards data: {str(e)}")
            return False
            
    async def add_rewards_points(self, user_id: str, points: int, action: str, description: str = None) -> bool:
        """Add rewards points to a user's account and record in history."""
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
            await self.client.table('rewards_points_history').insert({
                'user_id': user_id,
                'points': points,
                'action': action,
                'description': description
            }).execute()
            
            # Check if tier upgrade is needed
            await self.check_rewards_tier_change(user_id)
            
            return True
        except APIError as e:
            logger.error(f"Failed to add rewards points: {str(e)}")
            return False
            
    async def redeem_rewards_points(self, user_id: str, points: int, reward_type: str, reward_description: str = None) -> bool:
        """Redeem rewards points for a reward."""
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
            await self.client.table('rewards_redemption_history').insert({
                'user_id': user_id,
                'points_spent': points,
                'reward_type': reward_type,
                'reward_description': reward_description
            }).execute()
            
            # Check if tier downgrade is needed
            await self.check_rewards_tier_change(user_id)
            
            return True
        except APIError as e:
            logger.error(f"Failed to redeem rewards points: {str(e)}")
            return False
            
    async def check_rewards_tier_change(self, user_id: str) -> bool:
        """Check if user's rewards tier should change based on points and data sharing level."""
        try:
            # Get user data
            user_data = await self.get_user_rewards_data(user_id)
            if not user_data:
                return False
                
            current_points = user_data.get('rewards_points', 0)
            current_tier = user_data.get('rewards_tier', 'rookie')
            data_sharing_level = user_data.get('data_sharing_level', 1)
            
            # Get tier requirements
            response = await self.client.table('rewards_tier_requirements').select('*').execute()
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
                await self.create_audit_log(
                    user_id,
                    'rewards_tier_changed',
                    {
                        'previous_tier': current_tier,
                        'new_tier': appropriate_tier
                    }
                )
                return True
            return False
        except APIError as e:
            logger.error(f"Failed to check rewards tier change: {str(e)}")
            return False
            
    async def get_rewards_tier_benefits(self, tier: str = None) -> List[Dict[str, Any]]:
        """Get benefits for a specific tier or all tiers."""
        try:
            if tier:
                response = await self.client.table('rewards_tier_benefits').select('*').eq('tier', tier).execute()
            else:
                response = await self.client.table('rewards_tier_benefits').select('*').execute()
            return response.data
        except APIError as e:
            logger.error(f"Failed to get tier benefits: {str(e)}")
            return []
            
    async def get_rewards_tier_requirements(self, tier: str = None) -> List[Dict[str, Any]]:
        """Get requirements for a specific tier or all tiers."""
        try:
            if tier:
                response = await self.client.table('rewards_tier_requirements').select('*').eq('tier', tier).execute()
            else:
                response = await self.client.table('rewards_tier_requirements').select('*').execute()
            return response.data
        except APIError as e:
            logger.error(f"Failed to get tier requirements: {str(e)}")
            return []
            
    async def get_rewards_points_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get points history for a user."""
        try:
            response = await self.client.table('rewards_points_history').select('*').eq('user_id', user_id).order('created_at', {'ascending': False}).execute()
            return response.data
        except APIError as e:
            logger.error(f"Failed to get points history: {str(e)}")
            return []
            
    async def get_rewards_redemption_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get redemption history for a user."""
        try:
            response = await self.client.table('rewards_redemption_history').select('*').eq('user_id', user_id).order('created_at', {'ascending': False}).execute()
            return response.data
        except APIError as e:
            logger.error(f"Failed to get redemption history: {str(e)}")
            return []
            
    async def get_user_data_sharing_settings(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's data sharing settings."""
        try:
            response = await self.client.table('user_data_sharing_settings').select('*').eq('user_id', user_id).execute()
            return response.data
        except APIError as e:
            logger.error(f"Failed to get data sharing settings: {str(e)}")
            return []
            
    async def update_user_data_sharing_settings(self, user_id: str, settings: Dict[str, bool]) -> bool:
        """Update user's data sharing settings."""
        try:
            # Update each setting
            for setting_name, is_enabled in settings.items():
                await self.client.table('user_data_sharing_settings').update({
                    'is_enabled': is_enabled
                }).eq('user_id', user_id).eq('setting_name', setting_name).execute()
            
            # Calculate new data sharing level
            response = await self.client.table('user_data_sharing_settings').select('is_enabled').eq('user_id', user_id).execute()
            enabled_count = sum(1 for setting in response.data if setting['is_enabled'])
            
            # Determine data sharing level based on enabled settings count
            if enabled_count >= 5:
                new_level = 3  # Comprehensive
            elif enabled_count >= 3:
                new_level = 2  # Moderate
            else:
                new_level = 1  # Basic
            
            # Update user profile with new data sharing level
            await self.update_user_rewards_data(user_id, {'data_sharing_level': new_level})
            
            # Check if tier upgrade is needed
            await self.check_rewards_tier_change(user_id)
            
            return True
        except APIError as e:
            logger.error(f"Failed to update data sharing settings: {str(e)}")
            return False

# Initialize database interface
db = SupabaseDB()