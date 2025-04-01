"""Application configuration module.

This module centralizes all configuration settings for the application,
ensuring no hardcoded values and supporting environment-based configuration.
"""

import os
import secrets
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AppConfig:
    """Configuration class for application settings."""
    
    # Flask Configuration
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    DEBUG = FLASK_ENV == 'development'
    TESTING = os.environ.get('TESTING', 'False').lower() == 'true'
    
    # Security Configuration
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY')
    if not SECRET_KEY:
        if FLASK_ENV == 'production':
            raise ValueError("FLASK_SECRET_KEY must be set in production")
        # Generate a random secret key for development
        SECRET_KEY = secrets.token_hex(32)
        print("WARNING: Using a randomly generated secret key. This is fine for development but not for production.")
    
    # Session Configuration
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = FLASK_ENV == 'production'
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = int(os.environ.get('SESSION_LIFETIME', '86400'))  # 24 hours default
    
    # CSRF Protection
    WTF_CSRF_ENABLED = True
    WTF_CSRF_SECRET_KEY = os.environ.get('WTF_CSRF_SECRET_KEY', SECRET_KEY)
    
    # Authentication Configuration
    MIN_PASSWORD_LENGTH = int(os.environ.get('MIN_PASSWORD_LENGTH', '12'))
    BCRYPT_LOG_ROUNDS = int(os.environ.get('BCRYPT_LOG_ROUNDS', '12'))
    PASSWORD_RESET_TIMEOUT = int(os.environ.get('PASSWORD_RESET_TIMEOUT', '3600'))  # 1 hour default
    FAILED_LOGIN_TIMEOUT = int(os.environ.get('FAILED_LOGIN_TIMEOUT', '900'))  # 15 minutes default
    MAX_LOGIN_ATTEMPTS = int(os.environ.get('MAX_LOGIN_ATTEMPTS', '5'))
    
    # Rate Limiting
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '200 per day, 50 per hour')
    RATELIMIT_STORAGE_URL = os.environ.get('RATELIMIT_STORAGE_URL', 'memory://')
    
    # Supabase Configuration
    SUPABASE_URL = os.environ.get('SUPABASE_URL')
    SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
    SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')
    SUPABASE_JWT_SECRET = os.environ.get('SUPABASE_JWT_SECRET')
    
    # SSL/TLS Configuration (Production Only)
    SSL_CERT_PATH = os.environ.get('SSL_CERT_PATH')
    SSL_KEY_PATH = os.environ.get('SSL_KEY_PATH')
    
    # Default User Credentials (Development Only)
    DEFAULT_ADMIN_USERNAME = os.environ.get('DEFAULT_ADMIN_USERNAME')
    DEFAULT_ADMIN_PASSWORD = os.environ.get('DEFAULT_ADMIN_PASSWORD')
    DEFAULT_ADMIN_EMAIL = os.environ.get('DEFAULT_ADMIN_EMAIL')
    
    DEFAULT_PREMIUM_USERNAME = os.environ.get('DEFAULT_PREMIUM_USERNAME')
    DEFAULT_PREMIUM_PASSWORD = os.environ.get('DEFAULT_PREMIUM_PASSWORD')
    DEFAULT_PREMIUM_EMAIL = os.environ.get('DEFAULT_PREMIUM_EMAIL')
    
    DEFAULT_FREE_USERNAME = os.environ.get('DEFAULT_FREE_USERNAME')
    DEFAULT_FREE_PASSWORD = os.environ.get('DEFAULT_FREE_PASSWORD')
    DEFAULT_FREE_EMAIL = os.environ.get('DEFAULT_FREE_EMAIL')
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'ballbetz_diagnostics.log')
    
    @classmethod
    def get_flask_config(cls) -> Dict[str, Any]:
        """Get Flask configuration dictionary.
        
        Returns:
            Dictionary of Flask configuration settings
        """
        return {
            'DEBUG': cls.DEBUG,
            'TESTING': cls.TESTING,
            'SECRET_KEY': cls.SECRET_KEY,
            'SESSION_COOKIE_HTTPONLY': cls.SESSION_COOKIE_HTTPONLY,
            'SESSION_COOKIE_SECURE': cls.SESSION_COOKIE_SECURE,
            'SESSION_COOKIE_SAMESITE': cls.SESSION_COOKIE_SAMESITE,
            'PERMANENT_SESSION_LIFETIME': cls.PERMANENT_SESSION_LIFETIME,
            'WTF_CSRF_ENABLED': cls.WTF_CSRF_ENABLED,
            'WTF_CSRF_SECRET_KEY': cls.WTF_CSRF_SECRET_KEY,
        }
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration settings.
        
        Raises:
            ValueError: If any required configuration is missing or invalid
        """
        if cls.FLASK_ENV == 'production':
            # Check required production settings
            if not cls.SECRET_KEY or cls.SECRET_KEY == 'your-secret-key-here':
                raise ValueError("FLASK_SECRET_KEY must be set to a secure value in production")
            
            if not cls.SESSION_COOKIE_SECURE:
                raise ValueError("SESSION_COOKIE_SECURE should be True in production")
            
            # Check SSL/TLS configuration
            if not (cls.SSL_CERT_PATH and cls.SSL_KEY_PATH):
                raise ValueError("SSL_CERT_PATH and SSL_KEY_PATH must be set in production")
        
        # Check Supabase configuration if used
        if any([cls.SUPABASE_URL, cls.SUPABASE_KEY, cls.SUPABASE_SERVICE_KEY, cls.SUPABASE_JWT_SECRET]):
            missing = []
            if not cls.SUPABASE_URL:
                missing.append('SUPABASE_URL')
            if not cls.SUPABASE_KEY:
                missing.append('SUPABASE_KEY')
            if not cls.SUPABASE_SERVICE_KEY:
                missing.append('SUPABASE_SERVICE_KEY')
            if not cls.SUPABASE_JWT_SECRET:
                missing.append('SUPABASE_JWT_SECRET')
            
            if missing:
                raise ValueError(f"Missing required Supabase configuration: {', '.join(missing)}")