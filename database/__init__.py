"""Database package for BallBetz-Alpha.

This package provides database access and operations.
"""

from .base import DatabaseInterface, DatabaseError, ConnectionError, QueryError, ValidationError
from .supabase_client import SupabaseClient, db
from .user_operations import UserDatabaseOperations
from .rewards_operations import RewardsDatabaseOperations
from .data_sharing_operations import DataSharingDatabaseOperations

# Create operation instances with the Supabase client
user_ops = UserDatabaseOperations(db)
rewards_ops = RewardsDatabaseOperations(db)
data_sharing_ops = DataSharingDatabaseOperations(db)

__all__ = [
    'DatabaseInterface',
    'DatabaseError',
    'ConnectionError',
    'QueryError',
    'ValidationError',
    'SupabaseClient',
    'db',
    'user_ops',
    'rewards_ops',
    'data_sharing_ops',
    'UserDatabaseOperations',
    'RewardsDatabaseOperations',
    'DataSharingDatabaseOperations'
]