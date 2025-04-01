"""User management module.

This module handles user authentication, registration, and management.
It replaces the hardcoded user credentials with a proper user store
that can be backed by a database or other persistent storage.
"""

import os
import logging
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta
import pyotp
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize bcrypt for password hashing
bcrypt = Bcrypt()

class User:
    """User class for authentication and authorization."""
    
    def __init__(self, username: str, password: str, email: str, role: str = 'free'):
        """Initialize a user.
        
        Args:
            username: The user's username
            password: The user's password (will be hashed)
            email: The user's email address
            role: The user's role (admin, premium, or free)
        """
        self.id = username  # Use username as ID for simplicity
        self.username = username
        self.email = email
        self.role = role
        self.password_hash = self.hash_password(password)
        self.failed_login_attempts = 0
        self.locked_until = None
        self.totp_secret = None
        self.rewards_tier = 'rookie'
        self.rewards_points = 0
        self.data_sharing_level = 1
        
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt.
        
        Args:
            password: The password to hash
            
        Returns:
            The hashed password
        """
        return bcrypt.generate_password_hash(password).decode('utf-8')
        
    def check_password(self, password: str) -> bool:
        """Check if a password matches the stored hash.
        
        Args:
            password: The password to check
            
        Returns:
            True if the password matches, False otherwise
        """
        return bcrypt.check_password_hash(self.password_hash, password)
        
    def is_authenticated(self) -> bool:
        """Check if the user is authenticated.
        
        Returns:
            True if the user is authenticated
        """
        return True
        
    def is_active(self) -> bool:
        """Check if the user is active.
        
        Returns:
            True if the user is active
        """
        return True
        
    def is_anonymous(self) -> bool:
        """Check if the user is anonymous.
        
        Returns:
            False as this represents a registered user
        """
        return False
        
    def get_id(self) -> str:
        """Get the user's ID.
        
        Returns:
            The user's ID (username)
        """
        return self.username
        
    def is_admin(self) -> bool:
        """Check if the user has admin role.
        
        Returns:
            True if the user is an admin
        """
        return self.role == 'admin'
        
    def is_premium(self) -> bool:
        """Check if the user has premium role.
        
        Returns:
            True if the user is premium
        """
        return self.role in ['admin', 'premium']
        
    def increment_failed_attempts(self) -> None:
        """Increment the failed login attempts counter."""
        self.failed_login_attempts += 1
        
        # Lock account after max attempts
        max_attempts = int(os.environ.get('MAX_LOGIN_ATTEMPTS', '5'))
        if self.failed_login_attempts >= max_attempts:
            timeout_seconds = int(os.environ.get('FAILED_LOGIN_TIMEOUT', '900'))  # 15 minutes default
            self.locked_until = datetime.now() + timedelta(seconds=timeout_seconds)
            logger.warning(f"User {self.username} locked until {self.locked_until}")
        
    def reset_failed_attempts(self) -> None:
        """Reset the failed login attempts counter."""
        self.failed_login_attempts = 0
        self.locked_until = None
        
    def is_locked(self) -> bool:
        """Check if the user account is locked.
        
        Returns:
            True if the account is locked
        """
        if self.locked_until and datetime.now() < self.locked_until:
            return True
        return False
        
    def get_rewards_tier(self) -> str:
        """Get the user's rewards tier.
        
        Returns:
            The user's rewards tier
        """
        return self.rewards_tier
        
    def get_rewards_points(self) -> int:
        """Get the user's rewards points.
        
        Returns:
            The user's rewards points
        """
        return self.rewards_points
        
    def get_data_sharing_level(self) -> int:
        """Get the user's data sharing level.
        
        Returns:
            The user's data sharing level
        """
        return self.data_sharing_level
        
    def get_subscription_discount(self) -> int:
        """Get the user's subscription discount.
        
        Returns:
            The user's subscription discount percentage
        """
        # Example implementation - in a real app, this would be stored in the database
        tier_discounts = {
            'rookie': 0,
            'all_star': 10,
            'mvp': 20
        }
        return tier_discounts.get(self.rewards_tier, 0)
        
    def get_data_sharing_settings(self) -> List[Dict[str, Any]]:
        """Get the user's data sharing settings.
        
        Returns:
            List of data sharing settings
        """
        # Example implementation - in a real app, this would be stored in the database
        return [
            {'setting_name': 'profile_visibility', 'is_enabled': True},
            {'setting_name': 'betting_history', 'is_enabled': False},
            {'setting_name': 'favorite_teams', 'is_enabled': True},
            {'setting_name': 'betting_patterns', 'is_enabled': False},
            {'setting_name': 'platform_usage', 'is_enabled': True},
            {'setting_name': 'performance_stats', 'is_enabled': False}
        ]
        
    def update_data_sharing_settings(self, settings: Dict[str, bool]) -> bool:
        """Update the user's data sharing settings.
        
        Args:
            settings: Dictionary of setting names and enabled flags
            
        Returns:
            True if successful
        """
        # Example implementation - in a real app, this would update the database
        # For now, just log the settings
        logger.info(f"Updated data sharing settings for {self.username}: {settings}")
        return True
        
    def add_rewards_points(self, points: int, action: str, description: str = None) -> bool:
        """Add rewards points to the user's account.
        
        Args:
            points: Number of points to add
            action: The action that earned the points
            description: Optional description
            
        Returns:
            True if successful
        """
        self.rewards_points += points
        logger.info(f"Added {points} points to {self.username} for {action}")
        return True
        
    def redeem_rewards_points(self, points: int, reward_type: str, description: str = None) -> bool:
        """Redeem rewards points from the user's account.
        
        Args:
            points: Number of points to redeem
            reward_type: Type of reward
            description: Optional description
            
        Returns:
            True if successful, False if not enough points
        """
        if self.rewards_points < points:
            return False
            
        self.rewards_points -= points
        logger.info(f"Redeemed {points} points from {self.username} for {reward_type}")
        return True


class UserManager:
    """Manages user authentication and storage."""
    
    def __init__(self):
        """Initialize the user manager."""
        self.users: Dict[str, User] = {}
        self._load_default_users()
        
    def _load_default_users(self) -> None:
        """Load default users from environment variables.
        
        This is a fallback for development/testing. In production,
        users should be stored in a database.
        """
        # Check if admin credentials are provided in environment
        admin_username = os.environ.get('DEFAULT_ADMIN_USERNAME')
        admin_password = os.environ.get('DEFAULT_ADMIN_PASSWORD')
        admin_email = os.environ.get('DEFAULT_ADMIN_EMAIL')
        
        if admin_username and admin_password and admin_email:
            self.users[admin_username] = User(
                username=admin_username,
                password=admin_password,
                email=admin_email,
                role='admin'
            )
            logger.info(f"Loaded default admin user: {admin_username}")
            
        # Check if premium user credentials are provided
        premium_username = os.environ.get('DEFAULT_PREMIUM_USERNAME')
        premium_password = os.environ.get('DEFAULT_PREMIUM_PASSWORD')
        premium_email = os.environ.get('DEFAULT_PREMIUM_EMAIL')
        
        if premium_username and premium_password and premium_email:
            self.users[premium_username] = User(
                username=premium_username,
                password=premium_password,
                email=premium_email,
                role='premium'
            )
            logger.info(f"Loaded default premium user: {premium_username}")
            
        # Check if free user credentials are provided
        free_username = os.environ.get('DEFAULT_FREE_USERNAME')
        free_password = os.environ.get('DEFAULT_FREE_PASSWORD')
        free_email = os.environ.get('DEFAULT_FREE_EMAIL')
        
        if free_username and free_password and free_email:
            self.users[free_username] = User(
                username=free_username,
                password=free_password,
                email=free_email,
                role='free'
            )
            logger.info(f"Loaded default free user: {free_username}")
            
    def get_user(self, username: str) -> Optional[User]:
        """Get a user by username.
        
        Args:
            username: The username to look up
            
        Returns:
            The user object or None if not found
        """
        return self.users.get(username)
        
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email.
        
        Args:
            email: The email to look up
            
        Returns:
            The user object or None if not found
        """
        for user in self.users.values():
            if user.email == email:
                return user
        return None
        
    def create_user(self, username: str, password: str, email: str, role: str = 'free') -> Tuple[Optional[User], Optional[str]]:
        """Create a new user.
        
        Args:
            username: The username for the new user
            password: The password for the new user
            email: The email for the new user
            role: The role for the new user
            
        Returns:
            Tuple of (User, None) if successful, or (None, error_message) if failed
        """
        # Check if username already exists
        if username in self.users:
            return None, "Username already exists"
            
        # Check if email already exists
        if self.get_user_by_email(email):
            return None, "Email already exists"
            
        # Create new user
        user = User(username, password, email, role)
        self.users[username] = user
        logger.info(f"Created new user: {username}")
        
        return user, None
        
    def authenticate(self, username: str, password: str) -> Tuple[Optional[User], Optional[str]]:
        """Authenticate a user.
        
        Args:
            username: The username to authenticate
            password: The password to check
            
        Returns:
            Tuple of (User, None) if successful, or (None, error_message) if failed
        """
        user = self.get_user(username)
        
        if not user:
            return None, "Invalid username or password"
            
        if user.is_locked():
            return None, "Account is locked due to too many failed attempts"
            
        if not user.check_password(password):
            user.increment_failed_attempts()
            if user.is_locked():
                return None, "Account is now locked due to too many failed attempts"
            return None, "Invalid username or password"
            
        # Reset failed attempts on successful login
        user.reset_failed_attempts()
        
        return user, None


# Create a singleton instance
user_manager = UserManager()