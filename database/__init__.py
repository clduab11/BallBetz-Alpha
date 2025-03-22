"""
Database initialization and configuration module.
This module handles the initialization of Supabase and ensures proper configuration.
"""
import os
import logging
from supabase import create_client, Client
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class SupabaseManager:
    """Manages Supabase client initialization and configuration."""
    
    _instance: Optional['SupabaseManager'] = None
    _client: Optional[Client] = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize SupabaseManager."""
        if not self._client:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Supabase client with environment variables."""
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')

        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set. "
                "Please check your .env file or environment configuration."
            )

        try:
            self._client = create_client(url, key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise

    @property
    def client(self) -> Client:
        """Get the Supabase client instance."""
        if not self._client:
            self._initialize_client()
        return self._client

    def health_check(self) -> bool:
        """Check if Supabase connection is healthy."""
        try:
            # Attempt a simple query to verify connection
            self.client.table('user_profiles').select('id').limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Supabase health check failed: {str(e)}")
            return False

    def configure_app(self, app) -> None:
        """Configure Flask application with Supabase settings."""
        app.supabase = self.client
        
        # Add middleware for auth token validation
        from middleware.security import validate_jwt_token
        app.before_request(validate_jwt_token())
        
        # Load RLS policies
        self._load_rls_policies()
        
        logger.info("Flask application configured with Supabase")

    def _load_rls_policies(self) -> None:
        """Load and verify Row Level Security policies."""
        try:
            # Read schema.sql
            schema_path = Path(__file__).parent.parent / 'schema.sql'
            if not schema_path.exists():
                logger.warning("schema.sql not found - RLS policies may not be properly configured")
                return

            # Execute schema to ensure RLS policies are up to date
            with schema_path.open() as f:
                schema_sql = f.read()
                self.client.postgrest.raw(schema_sql)

            logger.info("RLS policies loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RLS policies: {str(e)}")
            raise

    def verify_rls_policies(self) -> bool:
        """Verify that RLS policies are properly configured."""
        try:
            # Check if RLS is enabled on critical tables
            tables = ['user_profiles', 'user_sessions', 'password_reset_tokens', 
                     'user_devices', 'audit_logs']
            
            for table in tables:
                result = self.client.postgrest.raw(f"""
                    SELECT relrowsecurity 
                    FROM pg_class 
                    WHERE relname = '{table}'
                """)
                if not result or not result.data[0]['relrowsecurity']:
                    logger.error(f"RLS not enabled on table: {table}")
                    return False

            # Verify key policies exist
            policies = self.client.postgrest.raw("""
                SELECT schemaname, tablename, policyname 
                FROM pg_policies
            """)

            required_policies = {
                'user_profiles': ['Users can view their own profile', 'Admins can view all profiles'],
                'audit_logs': ['Users can view their own audit logs', 'Admins can view all audit logs']
            }

            for table, expected_policies in required_policies.items():
                table_policies = [p['policyname'] for p in policies.data 
                                if p['tablename'] == table]
                missing = set(expected_policies) - set(table_policies)
                if missing:
                    logger.error(f"Missing policies for {table}: {missing}")
                    return False

            logger.info("RLS policies verified successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to verify RLS policies: {str(e)}")
            return False

# Create singleton instance
supabase_manager = SupabaseManager()

# Helper function to get client
def get_supabase_client() -> Client:
    """Get the Supabase client instance."""
    return supabase_manager.client