from flask_wtf import FlaskForm
from wtforms import (
    StringField, 
    PasswordField, 
    BooleanField, 
    SubmitField, 
    SelectField,
    IntegerField,
    TextAreaField
)
from wtforms.validators import (
    DataRequired, 
    Email, 
    EqualTo, 
    Length, 
    ValidationError,
    Regexp,
    NumberRange
)
import re
from models import User

class LoginForm(FlaskForm):
    """Form for user login."""
    email = StringField('Email', validators=[
        DataRequired(),
        Email(message="Please enter a valid email address")
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=12, message="Password must be at least 12 characters long")
    ])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    """Form for user registration."""
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=64),
        Regexp(r'^[\w.]+$', message="Username can only contain letters, numbers, and dots")
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email(message="Please enter a valid email address")
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=12, message='Password must be at least 12 characters long'),
        Regexp(r'[A-Z]', message='Password must contain at least one uppercase letter'),
        Regexp(r'[a-z]', message='Password must contain at least one lowercase letter'),
        Regexp(r'\d', message='Password must contain at least one number'),
        Regexp(r'[!@#$%^&*(),.?":{}|<>]', message='Password must contain at least one special character')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])
    role = SelectField('Role', choices=[
        ('free', 'Free User'),
        ('premium', 'Premium User')
    ], validators=[DataRequired()])
    submit = SubmitField('Register')

    def validate_password(self, field):
        """Custom password validation."""
        is_valid, error = User.validate_password(field.data)
        if not is_valid:
            raise ValidationError(error)

    def validate_username(self, field):
        """Check for common username security issues."""
        if re.search(r'(admin|root|superuser)', field.data.lower()):
            raise ValidationError('Username cannot contain reserved words')
        if len(field.data) > 64:
            raise ValidationError('Username must be less than 64 characters')

class PasswordResetRequestForm(FlaskForm):
    """Form for requesting a password reset."""
    email = StringField('Email', validators=[
        DataRequired(),
        Email(message="Please enter a valid email address")
    ])
    submit = SubmitField('Request Password Reset')

class PasswordResetForm(FlaskForm):
    """Form for resetting a password."""
    password = PasswordField('New Password', validators=[
        DataRequired(),
        Length(min=12, message='Password must be at least 12 characters long'),
        Regexp(r'[A-Z]', message='Password must contain at least one uppercase letter'),
        Regexp(r'[a-z]', message='Password must contain at least one lowercase letter'),
        Regexp(r'\d', message='Password must contain at least one number'),
        Regexp(r'[!@#$%^&*(),.?":{}|<>]', message='Password must contain at least one special character')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])
    submit = SubmitField('Reset Password')
    
    def validate_password(self, field):
        """Custom password validation."""
        is_valid, error = User.validate_password(field.data)
        if not is_valid:
            raise ValidationError(error)

class TwoFactorForm(FlaskForm):
    """Form for two-factor authentication."""
    token = StringField('Authentication Code', validators=[
        DataRequired(),
        Length(min=6, max=6, message='Authentication code must be 6 digits'),
        Regexp(r'^\d{6}$', message='Authentication code must be exactly 6 digits')
    ])
    remember_device = BooleanField('Remember this device for 30 days')
    submit = SubmitField('Verify')

class DataSharingSettingsForm(FlaskForm):
    """Form for managing data sharing settings."""
    profile_visibility = BooleanField('Profile Visibility', 
        description='Allow other users to see your profile information')
    betting_history = BooleanField('Betting History', 
        description='Share your anonymized betting history to improve platform predictions')
    favorite_teams = BooleanField('Favorite Teams', 
        description='Share your favorite teams to receive personalized recommendations')
    betting_patterns = BooleanField('Betting Patterns', 
        description='Share your betting patterns to help improve the platform algorithms')
    platform_usage = BooleanField('Platform Usage', 
        description='Share your platform usage data to help us improve the user experience')
    performance_stats = BooleanField('Performance Stats', 
        description='Share your performance statistics to contribute to community insights')
    submit = SubmitField('Save Settings')

class RewardsRedemptionForm(FlaskForm):
    """Form for redeeming rewards points."""
    reward_type = SelectField('Reward Type', choices=[
        ('subscription_discount', 'Subscription Discount'),
        ('premium_feature', 'Premium Feature Access'),
        ('exclusive_content', 'Exclusive Content'),
        ('custom_analysis', 'Custom Analysis Report')
    ], validators=[DataRequired()])
    points_to_spend = IntegerField('Points to Spend', validators=[
        DataRequired(),
        NumberRange(min=100, message='Minimum redemption is 100 points')
    ])
    submit = SubmitField('Redeem Points')

class ManualPointsAdjustmentForm(FlaskForm):
    """Admin form for manually adjusting user points."""
    user_email = StringField('User Email', validators=[
        DataRequired(),
        Email(message="Please enter a valid email address")
    ])
    points = IntegerField('Points (positive or negative)', validators=[
        DataRequired()
    ])
    action = SelectField('Action Type', choices=[
        ('manual_adjustment', 'Manual Adjustment'),
        ('bonus_points', 'Bonus Points'),
        ('correction', 'Correction'),
        ('promotion', 'Promotional Award')
    ], validators=[DataRequired()])
    description = TextAreaField('Description', validators=[
        DataRequired(),
        Length(max=200, message='Description must be less than 200 characters')
    ])
    submit = SubmitField('Adjust Points')