import pytest
from datetime import datetime, timedelta, timezone
import jwt
from unittest.mock import patch, MagicMock
from flask import Flask, request, g
from middleware.security import (
    configure_security_headers,
    validate_jwt_token,
    role_required,
    validate_csrf_token,
    validate_cors,
    rate_limit_by_ip
)

@pytest.fixture
def app():
    """Create a test Flask application."""
    app = Flask(__name__)
    app.config.update({
        'TESTING': True,
        'SECURITY_HEADERS': {
            'X-Frame-Options': 'DENY',
            'X-Content-Type-Options': 'nosniff',
            'X-XSS-Protection': '1; mode=block'
        },
        'SUPABASE_JWT_SECRET': 'test-secret',
        'CORS_ORIGINS': ['localhost:5000', 'example.com'],
        'RATELIMIT_ENABLED': True
    })
    return app

def create_token(payload, secret='test-secret', expired=False):
    """Create a test JWT token."""
    exp_time = datetime.now(timezone.utc) - timedelta(hours=1) if expired else datetime.now(timezone.utc) + timedelta(hours=1)
    token_payload = {
        **payload,
        'exp': int(exp_time.timestamp())
    }
    return jwt.encode(token_payload, secret, algorithm='HS256')

def test_security_headers(app):
    """Test security headers are properly set."""
    security_middleware = configure_security_headers()
    
    with app.test_request_context('/'):
        response = app.make_response('test')
        response = security_middleware(response)
        
        assert response.headers['X-Frame-Options'] == 'DENY'
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
        assert response.headers['X-XSS-Protection'] == '1; mode=block'

def test_jwt_validation_valid_token(app):
    """Test JWT validation with valid token."""
    token = create_token({'sub': 'user-1', 'role': 'admin', 'email': 'admin@test.com'})
    
    @validate_jwt_token()
    def protected_route():
        return {'success': True}, 200
    
    with app.test_request_context(headers={'Authorization': f'Bearer {token}'}):
        response, status_code = protected_route()
        assert status_code == 200
        assert g.user_id == 'user-1'
        assert g.user_role == 'admin'
        assert g.user_email == 'admin@test.com'

def test_jwt_validation_expired_token(app):
    """Test JWT validation with expired token."""
    token = create_token({'sub': 'user-1'}, expired=True)
    
    @validate_jwt_token()
    def protected_route():
        return {'success': True}, 200
    
    with app.test_request_context(headers={'Authorization': f'Bearer {token}'}):
        response, status_code = protected_route()
        assert status_code == 401
        assert response['error'] == 'Token has expired'

def test_jwt_validation_invalid_token(app):
    """Test JWT validation with invalid token."""
    token = create_token({'sub': 'user-1'}, secret='wrong-secret')
    
    @validate_jwt_token()
    def protected_route():
        return {'success': True}, 200
    
    with app.test_request_context(headers={'Authorization': f'Bearer {token}'}):
        response, status_code = protected_route()
        assert status_code == 401
        assert response['error'] == 'Invalid token'

def test_role_required_sufficient_permission(app):
    """Test role validation with sufficient permissions."""
    @role_required('free')
    def protected_route():
        return {'success': True}, 200
    
    with app.test_request_context():
        g.user_role = 'admin'
        response, status_code = protected_route()
        assert status_code == 200

def test_role_required_insufficient_permission(app):
    """Test role validation with insufficient permissions."""
    @role_required('admin')
    def protected_route():
        return {'success': True}, 200
    
    with app.test_request_context():
        g.user_role = 'free'
        response, status_code = protected_route()
        assert status_code == 403
        assert response['error'] == 'Insufficient permissions'

def test_csrf_validation_valid_token(app):
    """Test CSRF validation with valid token."""
    app.csrf = MagicMock()
    app.csrf.validate_csrf.return_value = True
    
    @validate_csrf_token()
    def protected_route():
        return {'success': True}, 200
    
    with app.test_request_context(method='POST', headers={'X-CSRF-Token': 'valid-token'}):
        response, status_code = protected_route()
        assert status_code == 200

def test_csrf_validation_missing_token(app):
    """Test CSRF validation with missing token."""
    @validate_csrf_token()
    def protected_route():
        return {'success': True}, 200
    
    with app.test_request_context(method='POST'):
        response, status_code = protected_route()
        assert status_code == 400
        assert response['error'] == 'CSRF token missing'

def test_cors_validation_allowed_origin(app):
    """Test CORS validation with allowed origin."""
    @validate_cors()
    def protected_route():
        return {'success': True}, 200
    
    with app.test_request_context(headers={'Origin': 'http://example.com'}):
        response, status_code = protected_route()
        assert status_code == 200

def test_cors_validation_disallowed_origin(app):
    """Test CORS validation with disallowed origin."""
    @validate_cors()
    def protected_route():
        return {'success': True}, 200
    
    with app.test_request_context(headers={'Origin': 'http://malicious.com'}):
        response, status_code = protected_route()
        assert status_code == 403
        assert response['error'] == 'Origin not allowed'

def test_rate_limit_not_exceeded(app):
    """Test rate limiting when limit not exceeded."""
    limiter = MagicMock()
    limiter.test.return_value = True
    app.extensions = {'limiter': limiter}
    
    @rate_limit_by_ip('5 per minute')
    def protected_route():
        return {'success': True}, 200
    
    with app.test_request_context():
        response, status_code = protected_route()
        assert status_code == 200

def test_rate_limit_exceeded(app):
    """Test rate limiting when limit exceeded."""
    limiter = MagicMock()
    limiter.test.return_value = False
    app.extensions = {'limiter': limiter}
    
    @rate_limit_by_ip('5 per minute')
    def protected_route():
        return {'success': True}, 200
    
    with app.test_request_context():
        response, status_code = protected_route()
        assert status_code == 429
        assert response['error'] == 'Rate limit exceeded. Try again later.'