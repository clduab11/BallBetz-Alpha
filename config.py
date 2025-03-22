import os
from datetime import timedelta

class Config:
    """Base configuration."""
    # Flask Settings
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') or 'dev-secret-key-replace-in-production'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SECURE = True
    REMEMBER_COOKIE_DURATION = timedelta(days=30)
    WTF_CSRF_TIME_LIMIT = 3600  # 1 hour CSRF token expiry

    # Supabase Settings
    SUPABASE_URL = os.environ.get('SUPABASE_URL')
    SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
    SUPABASE_JWT_SECRET = os.environ.get('SUPABASE_JWT_SECRET')

    # Security Settings
    PASSWORD_RESET_TIMEOUT = 3600  # 1 hour
    FAILED_LOGIN_TIMEOUT = 900  # 15 minutes
    MAX_LOGIN_ATTEMPTS = 5
    BCRYPT_LOG_ROUNDS = 12
    MIN_PASSWORD_LENGTH = 12

    # Rate Limiting
    RATELIMIT_DEFAULT = "200 per day"
    RATELIMIT_HEADERS_ENABLED = True
    RATELIMIT_STRATEGY = 'fixed-window-elastic-expiry'
    RATELIMIT_STORAGE_URL = 'memory://'

    # Security Headers
    SECURITY_HEADERS = {
        'X-Frame-Options': 'DENY',
        'X-Content-Type-Options': 'nosniff',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; "
                                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                                "style-src 'self' 'unsafe-inline'; "
                                "img-src 'self' data: https:; "
                                "connect-src 'self' https://*.supabase.co",
        'Permissions-Policy': 'accelerometer=(), camera=(), geolocation=(), gyroscope=(), '
                            'magnetometer=(), microphone=(), payment=(), usb=()'
    }

    # 2FA Settings
    TOTP_ISSUER = 'BallBetz'
    REMEMBER_DEVICE_DAYS = 30

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_SECURE = False

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    WTF_CSRF_ENABLED = False
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_SECURE = False
    RATELIMIT_ENABLED = False
    SUPABASE_URL = 'http://localhost:54321'
    SUPABASE_KEY = 'test-supabase-key'
    SUPABASE_JWT_SECRET = 'test-jwt-secret'

class ProductionConfig(Config):
    """Production configuration."""
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    WTF_CSRF_ENABLED = True
    RATELIMIT_ENABLED = True

    def __init__(self):
        if not os.environ.get('FLASK_SECRET_KEY'):
            raise ValueError('FLASK_SECRET_KEY environment variable must be set in production')
        if not os.environ.get('SUPABASE_URL'):
            raise ValueError('SUPABASE_URL environment variable must be set in production')
        if not os.environ.get('SUPABASE_KEY'):
            raise ValueError('SUPABASE_KEY environment variable must be set in production')
        if not os.environ.get('SUPABASE_JWT_SECRET'):
            raise ValueError('SUPABASE_JWT_SECRET environment variable must be set in production')

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}