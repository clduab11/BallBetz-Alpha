import pytest
import pyotp
import jwt
from flask import url_for, session
from datetime import datetime, timedelta, timezone
# Import User from models.py in the root directory
import importlib.util
from unittest.mock import patch, MagicMock

# Test user data
TEST_USER = {
    'username': 'testuser',
    'email': 'test@example.com',
    'password': 'Test@password123',
    'role': 'free'
}

ADMIN_USER = {
    'username': 'admin',
    'email': 'admin@example.com',
    'password': 'Admin@password123',
    'role': 'admin'
}

@pytest.mark.asyncio
async def test_user_registration_with_valid_data(client, mock_supabase):
    """Test successful user registration with valid data."""
    response = await client.post('/register', json=TEST_USER)
    assert response.status_code == 201
    assert 'id' in response.json
    assert 'email' in response.json
    assert response.json['role'] == 'free'

@pytest.mark.asyncio
async def test_user_registration_with_weak_password(client, mock_supabase):
    """Test registration fails with weak password."""
    weak_passwords = [
        ('short', 'Password must be at least 12 characters long'),
        ('nocapital123!', 'Password must contain at least one uppercase letter'),
        ('NOCAPS123!', 'Password must contain at least one lowercase letter'),
        ('NoNumbers!', 'Password must contain at least one number'),
        ('NoSpecial123', 'Password must contain at least one special character')
    ]

    for password, error_msg in weak_passwords:
        data = TEST_USER.copy()
        data['password'] = password
        response = await client.post('/register', json=data)
        assert response.status_code == 400
        assert error_msg in response.json['error']

@pytest.mark.asyncio
async def test_login_with_valid_credentials(client, mock_supabase, auth_headers):
    """Test successful login with valid credentials."""
    response = await client.post('/login', json={
        'email': TEST_USER['email'],
        'password': TEST_USER['password']
    })
    assert response.status_code == 200
    assert 'access_token' in response.json
    assert 'refresh_token' in response.json

@pytest.mark.asyncio
async def test_login_with_rate_limiting(client, mock_supabase):
    """Test login rate limiting."""
    for i in range(6):  # Exceed rate limit
        response = await client.post('/login', json={
            'email': TEST_USER['email'],
            'password': 'wrong_password'
        })
        if i < 5:
            assert response.status_code == 401
        else:
            assert response.status_code == 429
            assert 'Try again later' in response.json['error']

@pytest.mark.asyncio
async def test_protected_route_access(client, mock_supabase, auth_headers):
    """Test protected route access with valid JWT."""
    response = await client.get('/protected', headers=auth_headers)
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_protected_route_with_expired_token(client, mock_supabase, invalid_auth_headers):
    """Test protected route access with expired JWT."""
    response = await client.get('/protected', headers=invalid_auth_headers)
    assert response.status_code == 401
    assert 'Token has expired' in response.json['error']

@pytest.mark.asyncio
async def test_csrf_protection(client, mock_supabase):
    """Test CSRF protection on state-changing endpoints."""
    # Enable CSRF for this test
    client.application.config['WTF_CSRF_ENABLED'] = True
    
    response = await client.post('/login', json=TEST_USER)
    assert response.status_code == 400
    assert 'CSRF token missing' in response.json['error']

@pytest.mark.asyncio
async def test_role_based_access_control(client, mock_supabase):
    """Test RBAC for different user roles."""
    # Test admin access
    admin_token = create_test_token(ADMIN_USER)
    headers = {'Authorization': f'Bearer {admin_token}'}
    
    response = await client.get('/admin', headers=headers)
    assert response.status_code == 200

    # Test free user access
    free_token = create_test_token(TEST_USER)
    headers = {'Authorization': f'Bearer {free_token}'}
    
    response = await client.get('/admin', headers=headers)
    assert response.status_code == 403
    assert 'Insufficient permissions' in response.json['error']

@pytest.mark.asyncio
async def test_password_reset_flow(client, mock_supabase):
    """Test complete password reset flow."""
    # Request reset
    response = await client.post('/reset-password-request', json={'email': TEST_USER['email']})
    assert response.status_code == 200

    # Mock token verification
    token = create_test_token({'email': TEST_USER['email']})
    mock_supabase.auth.verify_password_recovery.return_value = {'user': {'id': 'test-id'}}

    # Reset password
    response = await client.post(f'/reset-password/{token}', json={
        'password': 'NewTest@password123'
    })
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_2fa_flow(client, mock_supabase):
    """Test two-factor authentication flow."""
    # Enable 2FA
    response = await client.post('/enable-2fa', headers=auth_headers)
    assert response.status_code == 200
    assert 'secret' in response.json
    
    secret = response.json['secret']
    totp = pyotp.TOTP(secret)
    valid_token = totp.now()

    # Verify 2FA setup
    response = await client.post('/verify-2fa', headers=auth_headers, json={'token': valid_token})
    assert response.status_code == 200
    
    # Login with 2FA
    login_response = await client.post('/login', json={
        'email': TEST_USER['email'],
        'password': TEST_USER['password']
    })
    assert login_response.status_code == 200
    assert login_response.json['requires_2fa'] is True
    
    # Complete 2FA
    response = await client.post('/complete-2fa', json={'token': valid_token})
    assert response.status_code == 200
    assert 'access_token' in response.json

def create_test_token(user_data, expired=False):
    """Helper function to create test JWT tokens."""
    exp_time = datetime.now(timezone.utc) - timedelta(hours=1) if expired else datetime.now(timezone.utc) + timedelta(hours=1)
    
    payload = {
        'sub': 'test-user-id',
        'email': user_data['email'],
        'role': user_data['role'],
        'exp': int(exp_time.timestamp())
    }
    
    return jwt.encode(payload, 'test-secret', algorithm='HS256')

@pytest.mark.asyncio
async def test_secure_headers(client):
    """Test secure headers are set correctly."""
    response = await client.get('/')
    headers = response.headers
    
    assert headers['X-Frame-Options'] == 'DENY'
    assert headers['X-Content-Type-Options'] == 'nosniff'
    assert headers['X-XSS-Protection'] == '1; mode=block'
    assert 'strict-origin-when-cross-origin' in headers['Referrer-Policy']
    assert 'max-age=31536000' in headers['Strict-Transport-Security']

@pytest.mark.asyncio
async def test_session_management(client, mock_supabase, auth_headers):
    """Test session management features."""
    # Test session creation
    response = await client.post('/login', json=TEST_USER)
    assert response.status_code == 200
    assert 'Set-Cookie' in response.headers
    
    cookie = response.headers['Set-Cookie']
    assert 'HttpOnly' in cookie
    assert 'SameSite=Lax' in cookie
    if not client.application.debug:
        assert 'Secure' in cookie

    # Test session expiry
    expired_token = create_test_token(TEST_USER, expired=True)
    headers = {'Authorization': f'Bearer {expired_token}'}
    
    response = await client.get('/protected', headers=headers)
    assert response.status_code == 401
    assert 'Token has expired' in response.json['error']

@pytest.mark.asyncio
async def test_refresh_token_flow(client, mock_supabase):
    """Test JWT refresh token flow."""
    # Get initial tokens
    response = await client.post('/login', json=TEST_USER)
    assert response.status_code == 200
    refresh_token = response.json['refresh_token']
    
    # Mock token refresh
    mock_supabase.auth.refresh_session.return_value = {
        'access_token': 'new_access_token',
        'refresh_token': 'new_refresh_token'
    }
    
    # Refresh tokens
    response = await client.post('/refresh-token', json={'refresh_token': refresh_token})
    assert response.status_code == 200
    assert 'access_token' in response.json
    assert response.json['access_token'] != 'new_access_token'
