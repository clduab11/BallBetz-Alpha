"""Authentication routes for BallBetz-Alpha.

This module handles user authentication, registration, and password reset.
"""

import logging
from datetime import datetime
from urllib.parse import urlparse, urljoin
from flask import request, render_template, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from . import auth_bp
from forms import LoginForm, RegistrationForm, PasswordResetRequestForm, PasswordResetForm, TwoFactorForm
from auth.user_manager import user_manager
from config.app_config import AppConfig
import pyotp

# Set up logging
logger = logging.getLogger(__name__)


def is_safe_url(target):
    """Check if the URL is safe to redirect to."""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc


def generate_reset_token(email):
    """Generate a secure token for password reset."""
    from itsdangerous import URLSafeTimedSerializer
    serializer = URLSafeTimedSerializer(AppConfig.SECRET_KEY)
    return serializer.dumps(email, salt='password-reset-salt')


def verify_reset_token(token, expiration=None):
    """Verify a password reset token."""
    if expiration is None:
        expiration = AppConfig.PASSWORD_RESET_TIMEOUT
        
    from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
    serializer = URLSafeTimedSerializer(AppConfig.SECRET_KEY)
    try:
        email = serializer.loads(
            token,
            salt='password-reset-salt',
            max_age=expiration
        )
        return email
    except (SignatureExpired, BadSignature):
        return None


def generate_totp_secret():
    """Generate a new TOTP secret."""
    return pyotp.random_base32()


def verify_totp(secret, token):
    """Verify a TOTP token."""
    totp = pyotp.TOTP(secret)
    return totp.verify(token)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = LoginForm()
    if form.validate_on_submit():
        user, error = user_manager.authenticate(form.username.data, form.password.data)
        
        if user:
            if user.totp_secret:
                session['username'] = user.username  # Store username in session for 2FA verification
                return redirect(url_for('auth.two_factor_auth'))

            login_user(user, remember=form.remember_me.data)
            logger.info(f"User {user.username} logged in successfully.")
            next_page = request.args.get('next')
            if not next_page or not is_safe_url(next_page):
                next_page = url_for('main.index')
            return redirect(next_page)
        else:
            flash(error or 'Invalid username or password.', 'danger')

    return render_template('login.html', form=form)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        user, error = user_manager.create_user(
            username=form.username.data,
            password=form.password.data,
            email=form.email.data,
            role=form.role.data
        )
        
        if user:
            logger.info(f"New user registered: {user.username}")
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash(error or 'Registration failed. Please try again.', 'danger')

    return render_template('register.html', form=form)


@auth_bp.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    logger.info("User logged out.")
    flash('You have been logged out.', 'success')
    return redirect(url_for('main.index'))


@auth_bp.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    """Handle password reset requests."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    form = PasswordResetRequestForm()
    if form.validate_on_submit():
        # Find user by email
        user = user_manager.get_user_by_email(form.email.data)
        if user:
            token = generate_reset_token(user.email)
            # In a real application, you would send an email with the reset link
            # For this example, we'll just log it
            reset_url = url_for('auth.reset_password', token=token, _external=True)
            logger.info(f"Password reset requested for {user.email}. Reset URL: {reset_url}")
            flash('Check your email for instructions to reset your password.', 'info')
        else:
            logger.warning(f"Password reset requested for unknown email: {form.email.data}")
            # Don't reveal that the email doesn't exist
            flash('Check your email for instructions to reset your password.', 'info')
        return redirect(url_for('auth.login'))
    
    return render_template('reset_password_request.html', form=form)


@auth_bp.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Handle password reset with token."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    email = verify_reset_token(token)
    if not email:
        flash('The password reset link is invalid or has expired.', 'danger')
        return redirect(url_for('auth.reset_password_request'))
    
    form = PasswordResetForm()
    if form.validate_on_submit():
        user = user_manager.get_user_by_email(email)
        if user:
            user.password_hash = user.hash_password(form.password.data)
            logger.info(f"Password reset successful for {user.email}")
            flash('Your password has been reset. You can now log in with your new password.', 'success')
            return redirect(url_for('auth.login'))
    
    return render_template('reset_password.html', form=form)


@auth_bp.route('/two_factor_auth', methods=['GET', 'POST'])
def two_factor_auth():
    """Handle two-factor authentication."""
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    
    user = user_manager.get_user(session['username'])
    if not user or not user.totp_secret:
        session.pop('username', None)
        return redirect(url_for('auth.login'))
    
    form = TwoFactorForm()
    if form.validate_on_submit():
        if verify_totp(user.totp_secret, form.token.data):
            login_user(user)
            user.reset_failed_attempts()
            session.pop('username', None)
            logger.info(f"User {user.username} passed 2FA verification")
            next_page = request.args.get('next')
            if not next_page or not is_safe_url(next_page):
                next_page = url_for('main.index')
            return redirect(next_page)
        else:
            logger.warning(f"Failed 2FA attempt for user {user.username}")
            flash('Invalid authentication code.', 'danger')
    
    return render_template('two_factor_auth.html', form=form)


@auth_bp.route('/setup_2fa', methods=['GET', 'POST'])
@login_required
def setup_2fa():
    """Set up two-factor authentication."""
    if current_user.totp_secret:
        flash('Two-factor authentication is already enabled.', 'info')
        return redirect(url_for('main.index'))
    
    # Generate a new TOTP secret
    secret = generate_totp_secret()
    current_user.totp_secret = secret
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(name=current_user.email, issuer_name="BallBetz")
    return render_template('setup_2fa.html', totp_uri=totp_uri)