"""Base database module for BallBetz-Alpha.

This module defines the base database interface and common functionality.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
from abc import ABC, abstractmethod

# Set up logging
logger = logging.getLogger(__name__)


class DatabaseInterface(ABC):
    """Abstract base class for database interfaces."""
    
    @abstractmethod
    def connect(self) -> None:
        """Create database connection."""
        pass
    
    @abstractmethod
    async def create_user_profile(self, user_data: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
        """Create a new user profile in the database."""
        pass
    
    @abstractmethod
    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Retrieve a user by email."""
        pass
    
    @abstractmethod
    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Retrieve a user by ID."""
        pass
    
    @abstractmethod
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Update a user's profile."""
        pass
    
    @abstractmethod
    async def create_audit_log(self, user_id: str, action: str, details: Dict[str, Any]) -> bool:
        """Create an audit log entry."""
        pass


class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class ConnectionError(DatabaseError):
    """Exception raised when database connection fails."""
    pass


class QueryError(DatabaseError):
    """Exception raised when a database query fails."""
    pass


class ValidationError(DatabaseError):
    """Exception raised when data validation fails."""
    pass