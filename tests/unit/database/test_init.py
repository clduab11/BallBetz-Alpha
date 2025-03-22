import pytest
from unittest.mock import MagicMock, patch, mock_open
import os
from database import SupabaseManager, get_supabase_client
from supabase import Client
from pathlib import Path

@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables."""
    with patch.dict(os.environ, {
        'SUPABASE_URL': 'https://test.supabase.co',
        'SUPABASE_KEY': 'test-key'
    }):
        yield

@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    client = MagicMock(spec=Client)
    client.table = MagicMock(return_value=client)
    client.select = MagicMock(return_value=client)
    client.limit = MagicMock(return_value=client)
    client.execute = MagicMock(return_value=MagicMock(data=[{'id': 1}]))
    client.postgrest = MagicMock()
    return client

@pytest.fixture
def supabase_manager(mock_env_vars, mock_supabase_client):
    """Create a SupabaseManager instance with mocked client."""
    with patch('database.create_client', return_value=mock_supabase_client):
        manager = SupabaseManager()
        return manager

def test_singleton_pattern():
    """Test that SupabaseManager follows singleton pattern."""
    with patch('database.create_client'):
        manager1 = SupabaseManager()
        manager2 = SupabaseManager()
        assert manager1 is manager2

def test_initialization_missing_env_vars():
    """Test initialization fails when environment variables are missing."""
    with patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError) as exc_info:
            SupabaseManager()
        assert "SUPABASE_URL and SUPABASE_KEY environment variables must be set" in str(exc_info.value)

def test_initialization_success(supabase_manager, mock_supabase_client):
    """Test successful initialization."""
    assert supabase_manager.client is mock_supabase_client

def test_health_check_success(supabase_manager, mock_supabase_client):
    """Test successful health check."""
    assert supabase_manager.health_check() is True
    mock_supabase_client.table.assert_called_with('user_profiles')
    mock_supabase_client.select.assert_called_with('id')

def test_health_check_failure(supabase_manager, mock_supabase_client):
    """Test health check failure."""
    mock_supabase_client.table.side_effect = Exception("Connection error")
    assert supabase_manager.health_check() is False

def test_configure_app(supabase_manager, mock_supabase_client):
    """Test Flask app configuration."""
    mock_app = MagicMock()
    
    with patch('database.validate_jwt_token') as mock_validator:
        supabase_manager.configure_app(mock_app)
        
        assert mock_app.supabase is mock_supabase_client
        assert mock_app.before_request.called
        mock_validator.assert_called_once()

def test_load_rls_policies_success(supabase_manager, mock_supabase_client):
    """Test successful loading of RLS policies."""
    mock_schema = """
    create policy "Test policy" on users for select using (true);
    """
    
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.open', mock_open(read_data=mock_schema)):
        supabase_manager._load_rls_policies()
        mock_supabase_client.postgrest.raw.assert_called_with(mock_schema)

def test_load_rls_policies_missing_file(supabase_manager, caplog):
    """Test RLS policy loading when schema file is missing."""
    with patch('pathlib.Path.exists', return_value=False):
        supabase_manager._load_rls_policies()
        assert "schema.sql not found" in caplog.text

def test_verify_rls_policies_success(supabase_manager, mock_supabase_client):
    """Test successful RLS policy verification."""
    # Mock successful RLS checks
    mock_supabase_client.postgrest.raw.side_effect = [
        MagicMock(data=[{'relrowsecurity': True}]),  # Table check
        MagicMock(data=[                            # Policy check
            {'tablename': 'user_profiles', 'policyname': 'Users can view their own profile'},
            {'tablename': 'user_profiles', 'policyname': 'Admins can view all profiles'},
            {'tablename': 'audit_logs', 'policyname': 'Users can view their own audit logs'},
            {'tablename': 'audit_logs', 'policyname': 'Admins can view all audit logs'}
        ])
    ]
    
    assert supabase_manager.verify_rls_policies() is True

def test_verify_rls_policies_missing_policy(supabase_manager, mock_supabase_client):
    """Test RLS policy verification with missing policy."""
    # Mock RLS check success but missing policy
    mock_supabase_client.postgrest.raw.side_effect = [
        MagicMock(data=[{'relrowsecurity': True}]),  # Table check
        MagicMock(data=[                            # Policy check - missing one required policy
            {'tablename': 'user_profiles', 'policyname': 'Users can view their own profile'},
            {'tablename': 'audit_logs', 'policyname': 'Users can view their own audit logs'}
        ])
    ]
    
    assert supabase_manager.verify_rls_policies() is False

def test_verify_rls_policies_disabled(supabase_manager, mock_supabase_client):
    """Test RLS policy verification when RLS is disabled."""
    # Mock RLS disabled on table
    mock_supabase_client.postgrest.raw.return_value = MagicMock(data=[{'relrowsecurity': False}])
    
    assert supabase_manager.verify_rls_policies() is False

def test_get_supabase_client(mock_env_vars, mock_supabase_client):
    """Test get_supabase_client helper function."""
    with patch('database.create_client', return_value=mock_supabase_client):
        client = get_supabase_client()
        assert client is mock_supabase_client
        
        # Test singleton behavior
        second_client = get_supabase_client()
        assert second_client is client