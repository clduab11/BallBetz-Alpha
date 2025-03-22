from datetime import datetime, timedelta, timezone
from flask_bcrypt import Bcrypt
from typing import Optional, Dict, Any, Tuple
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

class User:
    """User model integrated with Supabase Auth and Database."""
    
    def __init__(self, username: str, email: str, role: str = 'free', id: Optional[str] = None,
                 active: bool = True, totp_secret: Optional[str] = None):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        self.active = active
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
