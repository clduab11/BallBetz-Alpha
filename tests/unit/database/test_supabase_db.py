import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
import jwt
from database.supabase_db import SupabaseDB
from postgrest import APIError

# Test data
TEST_USER = {
    'id': 'test-user-id',
    'username': 'testuser',
    'email': 'test@example.com',
    'role': 'free'
}

@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    client = MagicMock()
    client.table = MagicMock(return_value=client)
    client.select = MagicMock(return_value=client)
    client.insert = MagicMock(return_value=client)
    client.update = MagicMock(return_value=client)
    client.eq = MagicMock(return_value=client)
    return client

@pytest.fixture
def db(mock_supabase_client):
    """Create a SupabaseDB instance with mocked client."""
    with patch('database.supabase_db.create_client', return_value=mock_supabase_client):
        db = SupabaseDB()
        return db

@pytest.mark.asyncio
async def test_create_user_profile_success(db, mock_supabase_client):
    """Test successful user profile creation."""
    mock_supabase_client.execute.return_value.data = [TEST_USER]
    
    profile, error = await db.create_user_profile(TEST_USER)
    
    assert profile == TEST_USER
    assert error is None
    mock_supabase_client.table.assert_called_with('user_profiles')
    mock_supabase_client.insert.assert_called_once()

@pytest.mark.asyncio
async def test_create_user_profile_failure(db, mock_supabase_client):
    """Test user profile creation failure."""
    mock_supabase_client.execute.side_effect = APIError('Database error')
    
    profile, error = await db.create_user_profile(TEST_USER)
    
    assert profile is None
    assert error == 'Database error'

@pytest.mark.asyncio
async def test_get_user_by_email_success(db, mock_supabase_client):
    """Test successful user retrieval by email."""
    mock_supabase_client.execute.return_value.data = [TEST_USER]
    
    user = await db.get_user_by_email(TEST_USER['email'])
    
    assert user == TEST_USER
    mock_supabase_client.table.assert_called_with('user_profiles')
    mock_supabase_client.eq.assert_called_with('email', TEST_USER['email'])

@pytest.mark.asyncio
async def test_get_user_by_email_not_found(db, mock_supabase_client):
    """Test user retrieval when user not found."""
    mock_supabase_client.execute.return_value.data = []
    
    user = await db.get_user_by_email('nonexistent@example.com')
    
    assert user is None

@pytest.mark.asyncio
async def test_update_user_profile_success(db, mock_supabase_client):
    """Test successful user profile update."""
    mock_supabase_client.execute.return_value.data = [TEST_USER]
    updates = {'username': 'newusername'}
    
    success, error = await db.update_user_profile(TEST_USER['id'], updates)
    
    assert success is True
    assert error is None
    mock_supabase_client.table.assert_called_with('user_profiles')
    mock_supabase_client.eq.assert_called_with('id', TEST_USER['id'])

@pytest.mark.asyncio
async def test_create_session_success(db, mock_supabase_client):
    """Test successful session creation."""
    session_id = 'test-session-id'
    mock_supabase_client.execute.return_value.data = [{'id': session_id}]
    session_data = {
        'device_id': 'test-device',
        'ip_address': '127.0.0.1',
        'user_agent': 'Test Browser',
        'remember_me': True
    }
    
    result_id, error = await db.create_session(TEST_USER['id'], session_data)
    
    assert result_id == session_id
    assert error is None
    mock_supabase_client.table.assert_called_with('user_sessions')

@pytest.mark.asyncio
async def test_validate_session_valid(db, mock_supabase_client):
    """Test session validation with valid session."""
    future_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    mock_supabase_client.execute.return_value.data = [{
        'id': 'test-session-id',
        'expires_at': future_time
    }]
    
    is_valid = await db.validate_session('test-session-id')
    
    assert is_valid is True

@pytest.mark.asyncio
async def test_validate_session_expired(db, mock_supabase_client):
    """Test session validation with expired session."""
    past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    mock_supabase_client.execute.return_value.data = [{
        'id': 'test-session-id',
        'expires_at': past_time
    }]
    
    is_valid = await db.validate_session('test-session-id')
    
    assert is_valid is False

@pytest.mark.asyncio
async def test_create_password_reset_token_success(db, mock_supabase_client):
    """Test successful password reset token creation."""
    token = 'test-token'
    mock_supabase_client.execute.return_value.data = [{'token': token}]
    
    result_token, error = await db.create_password_reset_token(TEST_USER['id'])
    
    assert result_token is not None
    assert error is None
    mock_supabase_client.table.assert_called_with('password_reset_tokens')

@pytest.mark.asyncio
async def test_validate_reset_token_valid(db, mock_supabase_client):
    """Test reset token validation with valid token."""
    future_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    mock_supabase_client.execute.return_value.data = [{
        'user_id': TEST_USER['id'],
        'expires_at': future_time,
        'used_at': None
    }]
    
    user_id, error = await db.validate_reset_token('test-token')
    
    assert user_id == TEST_USER['id']
    assert error is None

@pytest.mark.asyncio
async def test_validate_reset_token_expired(db, mock_supabase_client):
    """Test reset token validation with expired token."""
    past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    mock_supabase_client.execute.return_value.data = [{
        'user_id': TEST_USER['id'],
        'expires_at': past_time,
        'used_at': None
    }]
    
    user_id, error = await db.validate_reset_token('test-token')
    
    assert user_id is None
    assert error == 'Token expired'

@pytest.mark.asyncio
async def test_record_failed_login_under_limit(db, mock_supabase_client):
    """Test recording failed login attempt under limit."""
    mock_supabase_client.execute.return_value.data = [{
        'id': TEST_USER['id'],
        'failed_login_attempts': 1
    }]
    
    attempts, locked_until = await db.record_failed_login(TEST_USER['id'])
    
    assert attempts == 2
    assert locked_until is None

@pytest.mark.asyncio
async def test_record_failed_login_exceeds_limit(db, mock_supabase_client):
    """Test recording failed login attempt that exceeds limit."""
    with patch('database.supabase_db.current_app') as mock_app:
        mock_app.config = {'MAX_LOGIN_ATTEMPTS': 5}
        mock_supabase_client.execute.return_value.data = [{
            'id': TEST_USER['id'],
            'failed_login_attempts': 4
        }]
        
        attempts, locked_until = await db.record_failed_login(TEST_USER['id'])
        
        assert attempts == 5
        assert locked_until is not None

@pytest.mark.asyncio
async def test_create_audit_log_success(db, mock_supabase_client):
    """Test successful audit log creation."""
    mock_supabase_client.execute.return_value.data = [{'id': 'test-log-id'}]
    with patch('database.supabase_db.current_app') as mock_app:
        mock_app.request = MagicMock(
            remote_addr='127.0.0.1',
            user_agent=MagicMock(string='Test Browser')
        )
        
        success = await db.create_audit_log(
            TEST_USER['id'],
            'test_action',
            {'detail': 'test'}
        )
        
        assert success is True
        mock_supabase_client.table.assert_called_with('audit_logs')