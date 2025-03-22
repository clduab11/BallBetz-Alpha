from datetime import datetime, timedelta, timezone
from flask_bcrypt import Bcrypt
from typing import Optional, Dict, Any, Tuple, List
import re
import os
from flask import current_app
from supabase import Client, create_client

# Initialize bcrypt
bcrypt = Bcrypt()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL', ''),
    os.getenv('SUPABASE_KEY', '')
)

# Define rewards tier constants
REWARDS_TIER_ROOKIE = 'rookie'
REWARDS_TIER_ALL_STAR = 'all_star'
REWARDS_TIER_MVP = 'mvp'

# Define data sharing level constants
DATA_SHARING_LEVEL_BASIC = 1
DATA_SHARING_LEVEL_MODERATE = 2
DATA_SHARING_LEVEL_COMPREHENSIVE = 3
class User:
    """User model integrated with Supabase Auth and Database."""
    
    def __init__(self, username: str, email: str, role: str = 'free', id: Optional[str] = None,
                 active: bool = True, totp_secret: Optional[str] = None):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        self.active = active
        self.rewards_tier = REWARDS_TIER_ROOKIE
        self.rewards_points = 0
        self.data_sharing_level = DATA_SHARING_LEVEL_BASIC
        self.totp_secret = totp_secret
        self._jwt_token = None
        self._refresh_token = None 

    @staticmethod
    def validate_password(password: str) -> Tuple[bool, Optional[str]]:
        """Validate password complexity requirements.

        Requirements:
        - Minimum 12 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one number
        - At least one special character
        """
        if len(password) < 12:
            return False, "Password must be at least 12 characters long"
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r'\d', password):
            return False, "Password must contain at least one number"
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        return True, None

    @classmethod
    async def create(cls, username: str, password: str, email: str, role: str = 'free') -> Tuple['User', Optional[str]]:
        """Create a new user in Supabase Auth and Database."""
        # Validate password
        is_valid, error = cls.validate_password(password)
        if not is_valid:
            return None, error

        try:
            # Create user in Supabase Auth
            auth_response = supabase.auth.sign_up({
                "email": email,
                "password": password
            })

            if not auth_response.user:
                return None, "Failed to create user"

            # Create user profile in database with RLS
            user_data = {
                "id": auth_response.user.id,
                "username": username,
                "email": email,
                "role": role,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            db_response = supabase.table('users').insert(user_data).execute()
            
            if not db_response.data:
                # Rollback auth creation
                await supabase.auth.admin.delete_user(auth_response.user.id)
                return None, "Failed to create user profile"

            return cls(
                id=auth_response.user.id,
                username=username,
                email=email,
                role=role
            ), None

        except Exception as e:
            return None, str(e)

    @classmethod
    async def get_by_email(cls, email: str) -> Optional['User']:
        """Retrieve a user by email."""
        try:
            response = supabase.table('users').select('*').eq('email', email).execute()
            if response.data:
                user_data = response.data[0]
                return cls(**user_data)
            return None
        except Exception:
            return None

    def is_active(self) -> bool:
        return self.active
    
    def is_authenticated(self) -> bool:
        return bool(self._jwt_token and not self.is_token_expired())
    
    def is_anonymous(self) -> bool:
        return False
    
    def get_id(self) -> str:
        return str(self.id)

    def is_admin(self) -> bool:
        return self.role == 'admin'

    def is_premium(self) -> bool:
        return self.role == 'premium' or self.role == 'admin'

    def is_free(self) -> bool:
        return self.role == 'free'

    def is_enterprise(self) -> bool:
        return self.role == 'enterprise'

    async def get_rewards_tier(self) -> str:
        """Get the user's current rewards tier."""
        try:
            response = await supabase.table('user_profiles').select('rewards_tier').eq('id', self.id).execute()
            if response.data:
                return response.data[0]['rewards_tier']
            return REWARDS_TIER_ROOKIE
        except Exception as e:
            return REWARDS_TIER_ROOKIE

    async def get_rewards_points(self) -> int:
        """Get the user's current rewards points."""
        try:
            response = await supabase.table('user_profiles').select('rewards_points').eq('id', self.id).execute()
            if response.data:
                return response.data[0]['rewards_points']
            return 0
        except Exception as e:
            return 0

    async def get_data_sharing_level(self) -> int:
        """Get the user's current data sharing level."""
        try:
            response = await supabase.table('user_profiles').select('data_sharing_level').eq('id', self.id).execute()
            if response.data:
                return response.data[0]['data_sharing_level']
            return DATA_SHARING_LEVEL_BASIC
        except Exception as e:
            return DATA_SHARING_LEVEL_BASIC

    async def get_subscription_discount(self) -> float:
        """Get the user's subscription discount based on rewards tier."""
        try:
            tier = await self.get_rewards_tier()
            response = await supabase.table('rewards_tier_requirements').select('subscription_discount').eq('tier', tier).execute()
            if response.data:
                return float(response.data[0]['subscription_discount'])
            return 0.0
        except Exception as e:
            return 0.0

    async def add_rewards_points(self, points: int, action: str, description: str = None) -> bool:
        """Add rewards points to the user's account and record in history."""
        try:
            # Update user's points
            current_points = await self.get_rewards_points()
            new_points = current_points + points
            
            # Update user profile
            await supabase.table('user_profiles').update({
                'rewards_points': new_points
            }).eq('id', self.id).execute()
            
            # Record in history
            await supabase.table('rewards_points_history').insert({
                'user_id': self.id,
                'points': points,
                'action': action,
                'description': description
            }).execute()
            
            # Check if tier upgrade is needed
            await self.check_tier_upgrade()
            
            return True
        except Exception as e:
            return False

    async def redeem_rewards_points(self, points: int, reward_type: str, reward_description: str = None) -> bool:
        """Redeem rewards points for a reward."""
        try:
            # Check if user has enough points
            current_points = await self.get_rewards_points()
            if current_points < points:
                return False
            
            # Update user's points
            new_points = current_points - points
            await supabase.table('user_profiles').update({
                'rewards_points': new_points
            }).eq('id', self.id).execute()
            
            # Record redemption
            await supabase.table('rewards_redemption_history').insert({
                'user_id': self.id,
                'points_spent': points,
                'reward_type': reward_type,
                'reward_description': reward_description
            }).execute()
            
            # Check if tier downgrade is needed
            await self.check_tier_downgrade()
            
            return True
        except Exception as e:
            return False

    async def check_tier_upgrade(self) -> bool:
        """Check if user qualifies for a tier upgrade and apply if needed."""
        try:
            # Get current points and data sharing level
            current_points = await self.get_rewards_points()
            data_sharing_level = await self.get_data_sharing_level()
            
            # Get tier requirements
            response = await supabase.table('rewards_tier_requirements').select('*').order('min_points', {'ascending': False}).execute()
            
            # Check each tier starting from highest
            for tier_req in response.data:
                if current_points >= tier_req['min_points'] and data_sharing_level >= tier_req['min_data_sharing_level']:
                    # User qualifies for this tier
                    current_tier = await self.get_rewards_tier()
                    if tier_req['tier'] != current_tier:
                        # Update tier
                        await supabase.table('user_profiles').update({
                            'rewards_tier': tier_req['tier']
                        }).eq('id', self.id).execute()
                        
                        # Record tier change in audit log
                        await supabase.table('audit_logs').insert({
                            'user_id': self.id,
                            'action': 'rewards_tier_upgraded',
                            'details': {
                                'previous_tier': current_tier,
                                'new_tier': tier_req['tier']
                            }
                        }).execute()
                    return True
            return False
        except Exception as e:
            return False

    async def check_tier_downgrade(self) -> bool:
        """Check if user should be downgraded to a lower tier and apply if needed."""
        try:
            # Get current points and data sharing level
            current_points = await self.get_rewards_points()
            data_sharing_level = await self.get_data_sharing_level()
            current_tier = await self.get_rewards_tier()
            
            # Get tier requirements
            response = await supabase.table('rewards_tier_requirements').select('*').order('min_points', {'ascending': True}).execute()
            
            # Find the highest tier the user still qualifies for
            appropriate_tier = REWARDS_TIER_ROOKIE
            for tier_req in response.data:
                if current_points >= tier_req['min_points'] and data_sharing_level >= tier_req['min_data_sharing_level']:
                    appropriate_tier = tier_req['tier']
            
            # If current tier is higher than appropriate tier, downgrade
            if current_tier != appropriate_tier:
                # Update tier
                await supabase.table('user_profiles').update({
                    'rewards_tier': appropriate_tier
                }).eq('id', self.id).execute()
                
                # Record tier change in audit log
                await supabase.table('audit_logs').insert({
                    'user_id': self.id,
                    'action': 'rewards_tier_downgraded',
                    'details': {
                        'previous_tier': current_tier,
                        'new_tier': appropriate_tier
                    }
                }).execute()
                return True
            return False
        except Exception as e:
            return False

    async def update_data_sharing_settings(self, settings: Dict[str, bool]) -> bool:
        """Update user's data sharing settings."""
        try:
            # Update each setting
            for setting_name, is_enabled in settings.items():
                await supabase.table('user_data_sharing_settings').update({
                    'is_enabled': is_enabled
                }).eq('user_id', self.id).eq('setting_name', setting_name).execute()
            
            # Calculate new data sharing level
            response = await supabase.table('user_data_sharing_settings').select('is_enabled').eq('user_id', self.id).execute()
            enabled_count = sum(1 for setting in response.data if setting['is_enabled'])
            
            # Determine data sharing level based on enabled settings count
            if enabled_count >= 5:
                new_level = DATA_SHARING_LEVEL_COMPREHENSIVE
            elif enabled_count >= 3:
                new_level = DATA_SHARING_LEVEL_MODERATE
            else:
                new_level = DATA_SHARING_LEVEL_BASIC
            
            # Update user profile with new data sharing level
            await supabase.table('user_profiles').update({
                'data_sharing_level': new_level
            }).eq('id', self.id).execute()
            
            # Check if tier upgrade is needed
            await self.check_tier_upgrade()
            
            return True
        except Exception as e:
            return False

    async def get_data_sharing_settings(self) -> List[Dict[str, Any]]:
        """Get user's data sharing settings."""
        try:
            response = await supabase.table('user_data_sharing_settings').select('*').eq('user_id', self.id).execute()
            return response.data
        except Exception as e:
            return []
        
    async def request_password_reset(self) -> Tuple[bool, Optional[str]]:
        """Request a password reset."""
        try:
            await supabase.auth.reset_password_email(self.email)
            return True, None
        except Exception as e:
            return False, str(e)

    async def reset_password(self, new_password: str, token: str) -> Tuple[bool, Optional[str]]:
        """Reset password using token."""
        # Validate new password
        is_valid, error = self.validate_password(new_password)
        if not is_valid:
            return False, error

        try:
            await supabase.auth.verify_password_recovery(token, new_password)
            return True, None
        except Exception as e:
            return False, str(e)

    def set_jwt_token(self, token: str, refresh_token: str) -> None:
        """Set the JWT token and refresh token."""
        self._jwt_token = token
        self._refresh_token = refresh_token

    def is_token_expired(self) -> bool:
        """Check if the JWT token is expired."""
        if not self._jwt_token:
            return True
        try:
            # Get expiry from token without verification
            import jwt
            token_data = jwt.decode(self._jwt_token, options={"verify_signature": False})
            exp = datetime.fromtimestamp(token_data['exp'], timezone.utc)
            return datetime.now(timezone.utc) >= exp
        except Exception:
            return True


class Rewards:
    """Rewards system management."""
    
    @staticmethod
    async def get_tier_benefits(tier: str = None) -> List[Dict[str, Any]]:
        """Get benefits for a specific tier or all tiers."""
        try:
            if tier:
                response = await supabase.table('rewards_tier_benefits').select('*').eq('tier', tier).execute()
            else:
                response = await supabase.table('rewards_tier_benefits').select('*').execute()
            return response.data
        except Exception as e:
            return []
    
    @staticmethod
    async def get_tier_requirements(tier: str = None) -> List[Dict[str, Any]]:
        """Get requirements for a specific tier or all tiers."""
        try:
            if tier:
                response = await supabase.table('rewards_tier_requirements').select('*').eq('tier', tier).execute()
            else:
                response = await supabase.table('rewards_tier_requirements').select('*').execute()
            return response.data
        except Exception as e:
            return []
    
    @staticmethod
    async def get_points_history(user_id: str) -> List[Dict[str, Any]]:
        """Get points history for a user."""
        try:
            response = await supabase.table('rewards_points_history').select('*').eq('user_id', user_id).order('created_at', {'ascending': False}).execute()
            return response.data
        except Exception as e:
            return []
    
    @staticmethod
    async def get_redemption_history(user_id: str) -> List[Dict[str, Any]]:
        """Get redemption history for a user."""
        try:
            response = await supabase.table('rewards_redemption_history').select('*').eq('user_id', user_id).order('created_at', {'ascending': False}).execute()
            return response.data
        except Exception as e:
            return []
    
    @staticmethod
    async def award_points_for_action(user_id: str, action: str) -> bool:
        """Award points to a user based on a predefined action."""
        # Define point values for different actions
        point_values = {
            'daily_login': 10,
            'profile_completion': 50,
            'first_bet': 100,
            'winning_bet': 25,
            'refer_friend': 200,
            'feedback_submission': 30,
            'contest_participation': 75,
            'social_share': 15
        }
        
        if action not in point_values:
            return False
        
        try:
            # Get user
            response = await supabase.table('user_profiles').select('*').eq('id', user_id).execute()
            if not response.data:
                return False
            
            user_data = response.data[0]
            user = User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                role=user_data['role']
            )
            
            # Add points
            points = point_values[action]
            description = f"Points awarded for {action.replace('_', ' ')}"
            return await user.add_rewards_points(points, action, description)
        except Exception as e:
            return False
