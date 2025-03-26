import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models import User, Rewards, REWARDS_TIER_ROOKIE, REWARDS_TIER_ALL_STAR, REWARDS_TIER_MVP

class TestUserRewards:
    """Test the User model's rewards-related methods."""
    
    @pytest.fixture
    def mock_user(self):
        """Create a mock user for testing."""
        user = User(
            id='test-user-id',
            username='testuser',
            email='test@example.com',
            role='premium'
        )
        return user
    
    @pytest.mark.asyncio
    async def test_get_rewards_tier(self, mock_user):
        """Test getting a user's rewards tier."""
        # Mock the supabase response
        with patch('models.supabase.table') as mock_table:
            mock_execute = MagicMock()
            mock_execute.execute.return_value.data = [{'rewards_tier': REWARDS_TIER_ALL_STAR}]
            mock_table.return_value.select.return_value.eq.return_value = mock_execute
            
            # Call the method
            tier = await mock_user.get_rewards_tier()
            
            # Verify the result
            assert tier == REWARDS_TIER_ALL_STAR
            mock_table.assert_called_once_with('user_profiles')
            mock_table.return_value.select.assert_called_once_with('rewards_tier')
            mock_table.return_value.select.return_value.eq.assert_called_once_with('id', 'test-user-id')
    
    @pytest.mark.asyncio
    async def test_get_rewards_points(self, mock_user):
        """Test getting a user's rewards points."""
        # Mock the supabase response
        with patch('models.supabase.table') as mock_table:
            mock_execute = MagicMock()
            mock_execute.execute.return_value.data = [{'rewards_points': 500}]
            mock_table.return_value.select.return_value.eq.return_value = mock_execute
            
            # Call the method
            points = await mock_user.get_rewards_points()
            
            # Verify the result
            assert points == 500
            mock_table.assert_called_once_with('user_profiles')
            mock_table.return_value.select.assert_called_once_with('rewards_points')
            mock_table.return_value.select.return_value.eq.assert_called_once_with('id', 'test-user-id')
    
    @pytest.mark.asyncio
    async def test_get_data_sharing_level(self, mock_user):
        """Test getting a user's data sharing level."""
        # Mock the supabase response
        with patch('models.supabase.table') as mock_table:
            mock_execute = MagicMock()
            mock_execute.execute.return_value.data = [{'data_sharing_level': 2}]
            mock_table.return_value.select.return_value.eq.return_value = mock_execute
            
            # Call the method
            level = await mock_user.get_data_sharing_level()
            
            # Verify the result
            assert level == 2
            mock_table.assert_called_once_with('user_profiles')
            mock_table.return_value.select.assert_called_once_with('data_sharing_level')
            mock_table.return_value.select.return_value.eq.assert_called_once_with('id', 'test-user-id')
    
    @pytest.mark.asyncio
    async def test_get_subscription_discount(self, mock_user):
        """Test getting a user's subscription discount."""
        # Mock the get_rewards_tier method
        with patch.object(mock_user, 'get_rewards_tier', return_value=REWARDS_TIER_ALL_STAR):
            # Mock the supabase response
            with patch('models.supabase.table') as mock_table:
                mock_execute = MagicMock()
                mock_execute.execute.return_value.data = [{'subscription_discount': 25.00}]
                mock_table.return_value.select.return_value.eq.return_value = mock_execute
                
                # Call the method
                discount = await mock_user.get_subscription_discount()
                
                # Verify the result
                assert discount == 25.00
                mock_table.assert_called_once_with('rewards_tier_requirements')
                mock_table.return_value.select.assert_called_once_with('subscription_discount')
                mock_table.return_value.select.return_value.eq.assert_called_once_with('tier', REWARDS_TIER_ALL_STAR)
    
    @pytest.mark.asyncio
    async def test_add_rewards_points(self, mock_user):
        """Test adding rewards points to a user."""
        # Mock the get_rewards_points method
        with patch.object(mock_user, 'get_rewards_points', return_value=500):
            # Mock the check_tier_upgrade method
            with patch.object(mock_user, 'check_tier_upgrade', return_value=True):
                # Mock the supabase response for updating user profile
                with patch('models.supabase.table') as mock_table:
                    mock_execute = MagicMock()
                    mock_execute.execute.return_value.data = [{'rewards_points': 600}]
                    mock_table.return_value.update.return_value.eq.return_value = mock_execute
                    
                    # Call the method
                    success = await mock_user.add_rewards_points(100, 'test_action', 'Test description')
                    
                    # Verify the result
                    assert success is True
                    mock_table.assert_called_with('user_profiles')
                    mock_table.return_value.update.assert_called_once_with({'rewards_points': 600})
                    mock_table.return_value.update.return_value.eq.assert_called_once_with('id', 'test-user-id')
    
    @pytest.mark.asyncio
    async def test_redeem_rewards_points(self, mock_user):
        """Test redeeming rewards points."""
        # Mock the get_rewards_points method
        with patch.object(mock_user, 'get_rewards_points', return_value=500):
            # Mock the check_tier_downgrade method
            with patch.object(mock_user, 'check_tier_downgrade', return_value=True):
                # Mock the supabase response for updating user profile
                with patch('models.supabase.table') as mock_table:
                    mock_execute = MagicMock()
                    mock_execute.execute.return_value.data = [{'rewards_points': 300}]
                    mock_table.return_value.update.return_value.eq.return_value = mock_execute
                    
                    # Call the method
                    success = await mock_user.redeem_rewards_points(
                        200, 'test_reward', 'Test reward description'
                    )
                    
                    # Verify the result
                    assert success is True
                    mock_table.assert_called_with('user_profiles')
                    mock_table.return_value.update.assert_called_once_with({'rewards_points': 300})
                    mock_table.return_value.update.return_value.eq.assert_called_once_with('id', 'test-user-id')
    
    @pytest.mark.asyncio
    async def test_check_tier_upgrade(self, mock_user):
        """Test checking for tier upgrade."""
        # Mock the get_rewards_points method
        with patch.object(mock_user, 'get_rewards_points', return_value=1500):
            # Mock the get_data_sharing_level method
            with patch.object(mock_user, 'get_data_sharing_level', return_value=2):
                # Mock the get_rewards_tier method
                with patch.object(mock_user, 'get_rewards_tier', return_value=REWARDS_TIER_ROOKIE):
                    # Mock the supabase response for tier requirements
                    with patch('models.supabase.table') as mock_table:
                        # First call for tier requirements
                        mock_execute_req = MagicMock()
                        mock_execute_req.execute.return_value.data = [
                            {'tier': REWARDS_TIER_MVP, 'min_points': 5000, 'min_data_sharing_level': 3},
                            {'tier': REWARDS_TIER_ALL_STAR, 'min_points': 1000, 'min_data_sharing_level': 2},
                            {'tier': REWARDS_TIER_ROOKIE, 'min_points': 0, 'min_data_sharing_level': 1}
                        ]
                        
                        # Second call for updating user profile
                        mock_execute_update = MagicMock()
                        mock_execute_update.execute.return_value.data = [{'rewards_tier': REWARDS_TIER_ALL_STAR}]
                        
                        # Configure the mock to return different values for different calls
                        mock_table.return_value.select.return_value.order.return_value = mock_execute_req
                        mock_table.return_value.update.return_value.eq.return_value = mock_execute_update
                        
                        # Call the method
                        success = await mock_user.check_tier_upgrade()
                        
                        # Verify the result
                        assert success is True
                        mock_table.assert_called_with('user_profiles')
                        mock_table.return_value.update.assert_called_once_with({'rewards_tier': REWARDS_TIER_ALL_STAR})
                        mock_table.return_value.update.return_value.eq.assert_called_once_with('id', 'test-user-id')

class TestRewards:
    """Test the Rewards class methods."""
    
    @pytest.mark.asyncio
    async def test_get_tier_benefits(self):
        """Test getting tier benefits."""
        # Mock the supabase response
        with patch('models.supabase.table') as mock_table:
            mock_execute = MagicMock()
            mock_execute.execute.return_value.data = [
                {
                    'tier': REWARDS_TIER_ROOKIE,
                    'benefit_name': 'Custom Avatar Badges',
                    'benefit_description': 'Unique avatar badges to show your Rookie status'
                },
                {
                    'tier': REWARDS_TIER_ROOKIE,
                    'benefit_name': 'Basic Performance Insights',
                    'benefit_description': 'Access to basic performance metrics and insights'
                }
            ]
            mock_table.return_value.select.return_value.eq.return_value = mock_execute
            mock_table.return_value.select.return_value = mock_execute
            
            # Test with specific tier
            benefits = await Rewards.get_tier_benefits(REWARDS_TIER_ROOKIE)
            
            # Verify the result
            assert len(benefits) == 2
            assert benefits[0]['tier'] == REWARDS_TIER_ROOKIE
            assert benefits[0]['benefit_name'] == 'Custom Avatar Badges'
            
            # Test without tier (all tiers)
            benefits_all = await Rewards.get_tier_benefits()
            
            # Verify the result
            assert len(benefits_all) == 2
    
    @pytest.mark.asyncio
    async def test_get_tier_requirements(self):
        """Test getting tier requirements."""
        # Mock the supabase response
        with patch('models.supabase.table') as mock_table:
            mock_execute = MagicMock()
            mock_execute.execute.return_value.data = [
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
            ]
            mock_table.return_value.select.return_value.eq.return_value = mock_execute
            mock_table.return_value.select.return_value = mock_execute
            
            # Test with specific tier
            requirements = await Rewards.get_tier_requirements(REWARDS_TIER_ALL_STAR)
            
            # Verify the result
            assert len(requirements) == 3
            
            # Test without tier (all tiers)
            requirements_all = await Rewards.get_tier_requirements()
            
            # Verify the result
            assert len(requirements_all) == 3
    
    @pytest.mark.asyncio
    async def test_get_points_history(self):
        """Test getting points history."""
        # Mock the supabase response
        with patch('models.supabase.table') as mock_table:
            mock_execute = MagicMock()
            mock_execute.execute.return_value.data = [
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
            ]
            mock_table.return_value.select.return_value.eq.return_value.order.return_value = mock_execute
            
            # Call the method
            history = await Rewards.get_points_history('test-user-id')
            
            # Verify the result
            assert len(history) == 2
            assert history[0]['points'] == 100
            assert history[0]['action'] == 'daily_login'
            assert history[1]['points'] == 50
            assert history[1]['action'] == 'profile_completion'
    
    @pytest.mark.asyncio
    async def test_get_redemption_history(self):
        """Test getting redemption history."""
        # Mock the supabase response
        with patch('models.supabase.table') as mock_table:
            mock_execute = MagicMock()
            mock_execute.execute.return_value.data = [
                {
                    'user_id': 'test-user-id',
                    'points_spent': 200,
                    'reward_type': 'subscription_discount',
                    'reward_description': '2% discount on next subscription payment',
                    'created_at': datetime.now(timezone.utc).isoformat()
                },
                {
                    'user_id': 'test-user-id',
                    'points_spent': 300,
                    'reward_type': 'premium_feature',
                    'reward_description': 'Access to premium features for 3 days',
                    'created_at': datetime.now(timezone.utc).isoformat()
                }
            ]
            mock_table.return_value.select.return_value.eq.return_value.order.return_value = mock_execute
            
            # Call the method
            history = await Rewards.get_redemption_history('test-user-id')
            
            # Verify the result
            assert len(history) == 2
            assert history[0]['points_spent'] == 200
            assert history[0]['reward_type'] == 'subscription_discount'
            assert history[1]['points_spent'] == 300
            assert history[1]['reward_type'] == 'premium_feature'
    
    @pytest.mark.asyncio
    async def test_award_points_for_action(self):
        """Test awarding points for an action."""
        # Mock the supabase response for getting user
        with patch('models.supabase.table') as mock_table:
            mock_execute = MagicMock()
            mock_execute.execute.return_value.data = [
                {
                    'id': 'test-user-id',
                    'username': 'testuser',
                    'email': 'test@example.com',
                    'role': 'premium'
                }
            ]
            mock_table.return_value.select.return_value.eq.return_value = mock_execute
            
            # Mock the User.add_rewards_points method
            with patch('models.User.add_rewards_points', return_value=True):
                # Call the method
                success = await Rewards.award_points_for_action('test-user-id', 'daily_login')
                
                # Verify the result
                assert success is True
                mock_table.assert_called_with('user_profiles')
                mock_table.return_value.select.assert_called_once_with('*')
                mock_table.return_value.select.return_value.eq.assert_called_once_with('id', 'test-user-id')