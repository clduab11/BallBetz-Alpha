"""Authentication package for BallBetz-Alpha.

This package handles user authentication, authorization, and management.
"""

from .user_manager import User, UserManager, user_manager

__all__ = ['User', 'UserManager', 'user_manager']