"""Rewards routes for BallBetz-Alpha.

This module handles rewards-related routes for the application.
"""

import logging
from flask import render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user

from . import rewards_bp
from utils.error_handlers import handle_errors
from forms import DataSharingSettingsForm, RewardsRedemptionForm
from models import Rewards, REWARDS_TIER_ROOKIE, REWARDS_TIER_ALL_STAR, REWARDS_TIER_MVP

# Set up logging
logger = logging.getLogger(__name__)


@rewards_bp.route('/dashboard')
@login_required
@handle_errors
def rewards_dashboard():
    """Display the rewards dashboard."""
    try:
        # Get user's rewards data
        user_tier = current_user.get_rewards_tier()
        user_points = current_user.get_rewards_points()
        data_sharing_level = current_user.get_data_sharing_level()
        subscription_discount = current_user.get_subscription_discount()
        
        # Get tier benefits
        tier_benefits = Rewards.get_tier_benefits(user_tier)
        
        # Get tier requirements
        tier_requirements = Rewards.get_tier_requirements()
        
        # Calculate progress to next tier
        next_tier = REWARDS_TIER_ALL_STAR if user_tier == REWARDS_TIER_ROOKIE else REWARDS_TIER_MVP
        
        # Find current and next tier requirements
        current_tier_req = next((req for req in tier_requirements if req['tier'] == user_tier), None)
        next_tier_req = next((req for req in tier_requirements if req['tier'] == next_tier), None)
        
        if current_tier_req and next_tier_req:
            points_to_next_tier = next_tier_req['min_points'] - user_points
            points_to_next_tier = max(0, points_to_next_tier)
            
            # Calculate percentage progress
            points_range = next_tier_req['min_points'] - current_tier_req['min_points']
            points_progress = user_points - current_tier_req['min_points']
            tier_progress = min(100, max(0, int((points_progress / points_range) * 100))) if points_range > 0 else 0
        else:
            points_to_next_tier = 0
            tier_progress = 0
        
        # Map data sharing level to description
        data_sharing_descriptions = {
            1: "Basic",
            2: "Moderate",
            3: "Comprehensive"
        }
        data_sharing_description = data_sharing_descriptions.get(data_sharing_level, "Basic")
        
        return render_template('rewards_dashboard.html',
                              user_tier=user_tier,
                              user_points=user_points,
                              data_sharing_level=data_sharing_level,
                              data_sharing_description=data_sharing_description,
                              subscription_discount=subscription_discount,
                              tier_benefits=tier_benefits,
                              tier_requirements=tier_requirements,
                              points_to_next_tier=points_to_next_tier,
                              next_tier=next_tier,
                              tier_progress=tier_progress)
    except Exception as e:
        logger.error(f"Error in rewards_dashboard: {str(e)}")
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(url_for('main.index'))


@rewards_bp.route('/settings', methods=['GET', 'POST'])
@login_required
@handle_errors
def rewards_settings():
    """Manage data sharing settings."""
    try:
        form = DataSharingSettingsForm()
        
        # Get user's current data sharing settings
        data_sharing_settings = current_user.get_data_sharing_settings()
        data_sharing_level = current_user.get_data_sharing_level()
        
        # Map data sharing level to description
        data_sharing_descriptions = {
            1: "Basic",
            2: "Moderate",
            3: "Comprehensive"
        }
        data_sharing_description = data_sharing_descriptions.get(data_sharing_level, "Basic")
        
        # If form is submitted
        if form.validate_on_submit():
            # Prepare settings to update
            settings_to_update = {
                'profile_visibility': form.profile_visibility.data,
                'betting_history': form.betting_history.data,
                'favorite_teams': form.favorite_teams.data,
                'betting_patterns': form.betting_patterns.data,
                'platform_usage': form.platform_usage.data,
                'performance_stats': form.performance_stats.data
            }
            
            # Update settings
            success = current_user.update_data_sharing_settings(settings_to_update)
            
            if success:
                # Award points for updating settings
                current_user.add_rewards_points(
                    10, 
                    'settings_updated', 
                    'Points awarded for updating data sharing settings'
                )
                
                flash('Your data sharing settings have been updated successfully.', 'success')
                return redirect(url_for('rewards.rewards_dashboard'))
            else:
                flash('Failed to update data sharing settings. Please try again.', 'danger')
        elif request.method == 'GET':
            # Populate form with current settings
            for setting in data_sharing_settings:
                if hasattr(form, setting['setting_name']):
                    field = getattr(form, setting['setting_name'])
                    field.data = setting['is_enabled']
        
        return render_template('rewards_settings.html',
                              form=form,
                              data_sharing_level=data_sharing_level,
                              data_sharing_description=data_sharing_description)
    except Exception as e:
        logger.error(f"Error in rewards_settings: {str(e)}")
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(url_for('rewards.rewards_dashboard'))


@rewards_bp.route('/redeem', methods=['GET', 'POST'])
@login_required
@handle_errors
def rewards_redeem():
    """Redeem rewards points."""
    try:
        form = RewardsRedemptionForm()
        
        # Get user's rewards data
        user_points = current_user.get_rewards_points()
        
        # If form is submitted
        if form.validate_on_submit():
            reward_type = form.reward_type.data
            points_to_spend = form.points_to_spend.data
            
            # Validate points
            if points_to_spend > user_points:
                flash('You do not have enough points for this redemption.', 'danger')
                return redirect(url_for('rewards.rewards_redeem'))
            
            # Map reward types to descriptions
            reward_descriptions = {
                'subscription_discount': f"{points_to_spend // 100}% discount on next subscription payment",
                'premium_feature': f"Access to premium features for {points_to_spend // 100} days",
                'exclusive_content': "Access to exclusive betting strategy guides",
                'custom_analysis': "Personalized betting analysis report"
            }
            
            # Redeem points
            success = current_user.redeem_rewards_points(
                points_to_spend,
                reward_type,
                reward_descriptions.get(reward_type, "Reward redemption")
            )
            
            if success:
                flash('Your points have been redeemed successfully.', 'success')
                return redirect(url_for('rewards.rewards_dashboard'))
            else:
                flash('Failed to redeem points. Please try again.', 'danger')
        
        return render_template('rewards_redeem.html',
                              form=form,
                              user_points=user_points)
    except Exception as e:
        logger.error(f"Error in rewards_redeem: {str(e)}")
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(url_for('rewards.rewards_dashboard'))


@rewards_bp.route('/history')
@login_required
@handle_errors
def rewards_history():
    """View rewards history."""
    try:
        # Get user's rewards data
        user_points = current_user.get_rewards_points()
        
        # Get points history
        points_history = Rewards.get_points_history(current_user.id)
        
        # Get redemption history
        redemption_history = Rewards.get_redemption_history(current_user.id)
        
        # Get tier change history from audit logs
        tier_history = []  # This would come from audit logs in a real implementation
        
        # Calculate totals
        total_points_earned = sum(entry['points'] for entry in points_history)
        total_points_spent = sum(entry['points_spent'] for entry in redemption_history)
        
        # Calculate points by category
        points_by_category = []
        category_colors = {
            'daily_login': 'primary',
            'profile_completion': 'success',
            'first_bet': 'danger',
            'winning_bet': 'warning',
            'refer_friend': 'info',
            'feedback_submission': 'secondary',
            'settings_updated': 'dark',
            'other': 'light'
        }
        
        # Sample achievements (in a real app, these would be dynamic)
        achievements = []
        
        return render_template('rewards_history.html',
                              user_points=user_points,
                              points_history=points_history,
                              redemption_history=redemption_history,
                              tier_history=tier_history,
                              total_points_earned=total_points_earned,
                              total_points_spent=total_points_spent,
                              points_by_category=points_by_category,
                              achievements=achievements)
    except Exception as e:
        logger.error(f"Error in rewards_history: {str(e)}")
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(url_for('rewards.rewards_dashboard'))