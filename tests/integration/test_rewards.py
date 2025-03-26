import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import json
from datetime import datetime, timezone

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app import app
from models import User, Rewards, REWARDS_TIER_ROOKIE, REWARDS_TIER_ALL_STAR, REWARDS_TIER_MVP

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        yield client

@pytest.fixture
def authenticated_client(client):
    """Create an authenticated test client."""
    # Mock the current_user in Flask-Login
    with patch('flask_login.utils._get_user') as mock_get_user:
        # Create a mock user
        mock_user = MagicMock()
        mock_user.id = 'test-user-id'
        mock_user.username = 'testuser'
        mock_user.email = 'test@example.com'
        mock_user.role = 'premium'
        mock_user.is_authenticated = True
        mock_user.is_active = True
        mock_user.is_anonymous = False
        mock_user.get_id.return_value = 'test-user-id'
        
        # Mock rewards-related methods
        mock_user.get_rewards_tier.return_value = REWARDS_TIER_ALL_STAR
        mock_user.get_rewards_points.return_value = 1500
        mock_user.get_data_sharing_level.return_value = 2
        mock_user.get_subscription_discount.return_value = 25.0
        mock_user.add_rewards_points.return_value = True
        mock_user.redeem_rewards_points.return_value = True
        mock_user.update_data_sharing_settings.return_value = True
        mock_user.get_data_sharing_settings.return_value = [
            {'setting_name': 'profile_visibility', 'is_enabled': True},
            {'setting_name': 'betting_history', 'is_enabled': True},
            {'setting_name': 'favorite_teams', 'is_enabled': False},
            {'setting_name': 'betting_patterns', 'is_enabled': False},
            {'setting_name': 'platform_usage', 'is_enabled': True},
            {'setting_name': 'performance_stats', 'is_enabled': False}
        ]
        
        # Set the mock user as the current user
        mock_get_user.return_value = mock_user
        
        yield client

class TestRewardsRoutes:
    """Test the rewards-related routes."""
    
    def test_rewards_dashboard(self, authenticated_client):
        """Test the rewards dashboard route."""
        # Mock the Rewards.get_tier_benefits method
        with patch.object(Rewards, 'get_tier_benefits', return_value=[
            {
                'tier': REWARDS_TIER_ALL_STAR,
                'benefit_name': 'Priority Lineup Processing',
                'benefit_description': 'Your lineup optimization requests are processed with higher priority'
            },
            {
                'tier': REWARDS_TIER_ALL_STAR,
                'benefit_name': 'Exclusive Betting Guides',
                'benefit_description': 'Access to exclusive betting strategy guides'
            }
        ]):
            # Mock the Rewards.get_tier_requirements method
            with patch.object(Rewards, 'get_tier_requirements', return_value=[
                {
                    'tier': REWARDS_TIER_ROOKIE,
                    'min_points': 0,
                    'min_data_sharing_level': 1,
                    'subscription_discount': 10.00
                },
                {
                    'tier': REWARDS_TIER_ALL_STAR,
                    'min_points': 1000,
                    'min_data_sharing_level': 2,
                    'subscription_discount': 25.00
                },
                {
                    'tier': REWARDS_TIER_MVP,
                    'min_points': 5000,
                    'min_data_sharing_level': 3,
                    'subscription_discount': 40.00
                }
            ]):
                # Make the request
                response = authenticated_client.get('/rewards/dashboard')
                
                # Check the response
                assert response.status_code == 200
                assert b'StatShare Rewards' in response.data
                assert b'All Star Member' in response.data
                assert b'1500' in response.data  # Points
                assert b'25%' in response.data   # Discount
    
    def test_rewards_settings_get(self, authenticated_client):
        """Test the rewards settings route (GET)."""
        # Make the request
        response = authenticated_client.get('/rewards/settings')
        
        # Check the response
        assert response.status_code == 200
        assert b'Data Sharing Settings' in response.data
        assert b'How StatShare Rewards' in response.data
        assert b'Current Data Sharing Level' in response.data
    
    def test_rewards_settings_post(self, authenticated_client):
        """Test the rewards settings route (POST)."""
        # Prepare form data
        form_data = {
            'profile_visibility': 'y',
            'betting_history': 'y',
            'favorite_teams': 'y',
            'betting_patterns': 'y',
            'platform_usage': 'y',
            'performance_stats': 'y'
        }
        
        # Make the request
        response = authenticated_client.post('/rewards/settings', data=form_data, follow_redirects=True)
        
        # Check the response
        assert response.status_code == 200
        assert b'Your data sharing settings have been updated successfully' in response.data
    
    def test_rewards_redeem_get(self, authenticated_client):
        """Test the rewards redeem route (GET)."""
        # Make the request
        response = authenticated_client.get('/rewards/redeem')
        
        # Check the response
        assert response.status_code == 200
        assert b'Redeem Your Points' in response.data
        assert b'1500 Points Available' in response.data
        assert b'How Redemption Works' in response.data
    
    def test_rewards_redeem_post(self, authenticated_client):
        """Test the rewards redeem route (POST)."""
        # Prepare form data
        form_data = {
            'reward_type': 'subscription_discount',
            'points_to_spend': '1000'
        }
        
        # Make the request
        response = authenticated_client.post('/rewards/redeem', data=form_data, follow_redirects=True)
        
        # Check the response
        assert response.status_code == 200
        assert b'Your points have been redeemed successfully' in response.data
    
    def test_rewards_history(self, authenticated_client):
        """Test the rewards history route."""
        # Mock the Rewards.get_points_history method
        with patch.object(Rewards, 'get_points_history', return_value=[
            {
                'user_id': 'test-user-id',
                'points': 100,
                'action': 'daily_login',
                'description': 'Points awarded for daily login',
                'created_at': datetime.now(timezone.utc).isoformat()
            },
            {
                'user_id': 'test-user-id',
                'points': 50,
                'action': 'profile_completion',
                'description': 'Points awarded for completing profile',
                'created_at': datetime.now(timezone.utc).isoformat()
            }
        ]):
            # Mock the Rewards.get_redemption_history method
            with patch.object(Rewards, 'get_redemption_history', return_value=[
                {
                    'user_id': 'test-user-id',
                    'points_spent': 200,
                    'reward_type': 'subscription_discount',
                    'reward_description': '2% discount on next subscription payment',
                    'created_at': datetime.now(timezone.utc).isoformat()
                }
            ]):
                # Make the request
                response = authenticated_client.get('/rewards/history')
                
                # Check the response
                assert response.status_code == 200
                assert b'Rewards History' in response.data
                assert b'Points Earned' in response.data
                assert b'Points Redeemed' in response.data
                assert b'daily_login' in response.data
                assert b'profile_completion' in response.data
                assert b'subscription_discount' in response.data

class TestRewardsErrorHandling:
    """Test error handling in rewards routes."""
    
    def test_rewards_dashboard_error(self, authenticated_client):
        """Test error handling in rewards dashboard route."""
        # Mock the get_rewards_tier method to raise an exception
        with patch('flask_login.utils._get_user') as mock_get_user:
            mock_user = mock_get_user.return_value
            mock_user.get_rewards_tier.side_effect = Exception("Test error")
            
            # Make the request
            response = authenticated_client.get('/rewards/dashboard', follow_redirects=True)
            
            # Check the response
            assert response.status_code == 200
            assert b'An error occurred' in response.data
    
    def test_rewards_settings_error(self, authenticated_client):
        """Test error handling in rewards settings route."""
        # Mock the get_data_sharing_settings method to raise an exception
        with patch('flask_login.utils._get_user') as mock_get_user:
            mock_user = mock_get_user.return_value
            mock_user.get_data_sharing_settings.side_effect = Exception("Test error")
            
            # Make the request
            response = authenticated_client.get('/rewards/settings', follow_redirects=True)
            
            # Check the response
            assert response.status_code == 200
            assert b'An error occurred' in response.data
    
    def test_rewards_redeem_error(self, authenticated_client):
        """Test error handling in rewards redeem route."""
        # Prepare form data with invalid points
        form_data = {
            'reward_type': 'subscription_discount',
            'points_to_spend': '2000'  # More than available
        }
        
        # Mock the redeem_rewards_points method to return False
        with patch('flask_login.utils._get_user') as mock_get_user:
            mock_user = mock_get_user.return_value
            mock_user.redeem_rewards_points.return_value = False
            
            # Make the request
            response = authenticated_client.post('/rewards/redeem', data=form_data, follow_redirects=True)
            
            # Check the response
            assert response.status_code == 200
            assert b'Failed to redeem points' in response.data