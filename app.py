from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from flask_bcrypt import Bcrypt
# Import User from models.py in the root directory
import importlib.util
import sys
import os

# Load models.py as a module
spec = importlib.util.spec_from_file_location("models_module", os.path.join(os.path.dirname(__file__), "models.py"))
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)
User = models_module.User

from forms import LoginForm, RegistrationForm, PasswordResetRequestForm, PasswordResetForm, TwoFactorForm
from urllib.parse import urlparse, urljoin
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import pyotp
from scrapers.ufl_scraper import UFLScraper
from utils.data_processor import DataProcessor
from models.predictor import PlayerPerformancePredictor
from optimizers.lineup_optimizer import LineupOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ballbetz_diagnostics.log')
    ]
)
logger = logging.getLogger(__name__)

# Diagnostic: Check for required dependencies
logger.info("=== BallBetz Diagnostic Logs ===")
try:
    pulp_spec = importlib.util.find_spec('pulp')
    logger.info(f"PuLP dependency check: {'FOUND' if pulp_spec else 'MISSING'}")
except ImportError:
    logger.error("PuLP dependency check: MISSING - ImportError")

# Log Python and key package versions
logger.info(f"Python version: {importlib.sys.version}")

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')  # Change in production

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Set the login view

# Initialize Flask-Bcrypt
bcrypt = Bcrypt()
bcrypt.init_app(app)

# Initialize Flask-Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Configure secure session management
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,  # Only send cookies over HTTPS
    SESSION_COOKIE_SAMESITE='Lax',  # Protect against CSRF
    REMEMBER_COOKIE_DURATION=timedelta(days=7),  # Remember-me cookie duration
    REMEMBER_COOKIE_HTTPONLY=True,
    REMEMBER_COOKIE_SECURE=True,  # Only send remember-me cookie over HTTPS
)


# Create a simple user store (for demonstration purposes)
users = {}
users = {
    'admin': User('admin', 'admin_password', 'admin@example.com', role='admin'),
    'premium': User('premium', 'premium_password', 'premium@example.com', role='premium'),
    'free': User('free', 'free_password', 'free@example.com', role='free'),
}


@login_manager.user_loader
def load_user(user_id):
    if user_id == 'None' or user_id is None:
        return None
    try:
        return users.get(user_id)  # Use the username directly as the key
    except (ValueError, KeyError):
        return None

# Initialize components
try:
    scraper = UFLScraper()
    processor = DataProcessor()
    predictor = PlayerPerformancePredictor('models/ufl_predictor.joblib')
    dk_optimizer = LineupOptimizer('draftkings')
    fd_optimizer = LineupOptimizer('fanduel')
    logger.info("All components initialized successfully")
    # Check if model file exists
    logger.info(f"Model file exists: {Path('models/ufl_predictor.joblib').exists()}")
    
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    raise
# Commented out to avoid dependency on APScheduler
# from scheduler import create_scheduler




# Constants
BANKROLL_FILE = Path('data/bankroll.json')
DATA_DIR = Path('data')
REQUIRED_DIRS = ['data/raw', 'data/processed', 'models']

def setup_directories() -> None:
    """Create required directories if they don't exist."""
    for dir_path in REQUIRED_DIRS:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def handle_errors(f):
    """Decorator to handle errors in routes."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            flash(f"An error occurred: {str(e)}", 'error')
            return jsonify({'success': False, 'message': str(e)}), 500
    return wrapper

def load_bankroll() -> Dict[str, Any]:
    """
    Load bankroll data from file.
    
    Returns:
        Dict containing bankroll information
    """
    try:
        if BANKROLL_FILE.exists():
            with open(BANKROLL_FILE) as f:
                return json.load(f)
        return {
            'initial': 20,
            'current': 20,
            'history': [],
            'last_updated': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error loading bankroll: {str(e)}")
        return {
            'initial': 20,
            'current': 20,
            'history': [],
            'last_updated': datetime.now().isoformat()
        }

def save_bankroll(bankroll: Dict[str, Any]) -> bool:
    """
    Save bankroll data to file.
    
    Args:
        bankroll: Dictionary containing bankroll data
        
    Returns:
        bool: True if save was successful
    """
    try:
        BANKROLL_FILE.parent.mkdir(parents=True, exist_ok=True)
        bankroll['last_updated'] = datetime.now().isoformat()
        with open(BANKROLL_FILE, 'w') as f:
            json.dump(bankroll, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving bankroll: {str(e)}")
        return False

def generate_reset_token(email):
    """Generate a secure token for password reset."""
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    return serializer.dumps(email, salt='password-reset-salt')

def verify_reset_token(token, expiration=3600):
    """Verify a password reset token."""
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
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


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login', next=request.url))
        if not current_user.is_admin():
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('index'))  # Or another suitable page
        return f(*args, **kwargs)
    return decorated_function

def premium_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login', next=request.url))
        if not current_user.is_premium():
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('index'))  # Or another suitable page
        return f(*args, **kwargs)
    return decorated_function

def free_required(f):  # In this specific app, every logged in user can access "free" routes
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")  # Limit login attempts
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = LoginForm()
    if form.validate_on_submit():
        user = users.get(form.username.data)
        if user and user.check_password(form.password.data):
            if user.is_locked():
                flash('Your account is locked. Please try again later.', 'danger')
                return render_template('login.html', form=form)

            # Check if 2FA is enabled
            if user.totp_secret:
                session['username'] = user.username  # Store username in session for 2FA verification
                return redirect(url_for('two_factor_auth'))

            login_user(user, remember=form.remember_me.data)
            user.reset_failed_attempts()
            logger.info(f"User {user.username} logged in successfully.")
            next_page = request.args.get('next')
            if not next_page or urlparse(next_page).netloc != '':
                next_page = url_for('index')
            return redirect(next_page)
        else:
            if user:
                user.increment_failed_attempts()
                logger.warning(f"Failed login attempt for user {form.username.data}. Attempts: {user.failed_login_attempts}")
                if user.is_locked():
                    flash('Your account has been locked due to multiple failed login attempts.', 'danger')
                else:
                    flash('Invalid username or password.', 'danger')
            else:
                logger.warning(f"Failed login attempt for unknown user {form.username.data}")
                flash('Invalid username or password.', 'danger')

    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        # Check if username already exists
        if form.username.data in users:
            flash('Username already exists. Please choose a different one.', 'danger')
            return render_template('register.html', form=form)

        # Create new user
        new_user = User(
            username=form.username.data,
            password=form.password.data,
            email=form.email.data,
            role=form.role.data
        )
        users[new_user.username] = new_user # Use username as key for simplicity
        logger.info(f"New user registered: {new_user.username}")
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    logger.info("User logged out.")
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    """Handle password reset requests."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = PasswordResetRequestForm()
    if form.validate_on_submit():
        # Find user by email
        user = next((u for u in users.values() if u.email == form.email.data), None)
        if user:
            token = generate_reset_token(user.email)
            # In a real application, you would send an email with the reset link
            # For this example, we'll just log it
            reset_url = url_for('reset_password', token=token, _external=True)
            logger.info(f"Password reset requested for {user.email}. Reset URL: {reset_url}")
            flash('Check your email for instructions to reset your password.', 'info')
        else:
            logger.warning(f"Password reset requested for unknown email: {form.email.data}")
            # Don't reveal that the email doesn't exist
            flash('Check your email for instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    
    return render_template('reset_password_request.html', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Handle password reset with token."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    email = verify_reset_token(token)
    if not email:
        flash('The password reset link is invalid or has expired.', 'danger')
        return redirect(url_for('reset_password_request'))
    
    form = PasswordResetForm()
    if form.validate_on_submit():
        user = next((u for u in users.values() if u.email == email), None)
        if user:
            user.password_hash = user.hash_password(form.password.data)
            logger.info(f"Password reset successful for {user.email}")
            flash('Your password has been reset. You can now log in with your new password.', 'success')
            return redirect(url_for('login'))
    
    return render_template('reset_password.html', form=form)

@app.route('/two_factor_auth', methods=['GET', 'POST'])
def two_factor_auth():
    """Handle two-factor authentication."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    user = users.get(session['username'])
    if not user or not user.totp_secret:
        session.pop('username', None)
        return redirect(url_for('login'))
    
    form = TwoFactorForm()
    if form.validate_on_submit():
        if verify_totp(user.totp_secret, form.token.data):
            login_user(user)
            user.reset_failed_attempts()
            session.pop('username', None)
            logger.info(f"User {user.username} passed 2FA verification")
            next_page = request.args.get('next')
            if not next_page or urlparse(next_page).netloc != '':
                next_page = url_for('index')
            return redirect(next_page)
        else:
            logger.warning(f"Failed 2FA attempt for user {user.username}")
            flash('Invalid authentication code.', 'danger')
    
    return render_template('two_factor_auth.html', form=form)

@app.route('/setup_2fa', methods=['GET', 'POST'])
@login_required
@admin_required
def setup_2fa():
    """Set up two-factor authentication."""
    if current_user.totp_secret:
        flash('Two-factor authentication is already enabled.', 'info')
        return redirect(url_for('index'))
    
    # Generate a new TOTP secret
    secret = generate_totp_secret()
    current_user.totp_secret = secret
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(name=current_user.email, issuer_name="BallBetz")
    return render_template('setup_2fa.html', totp_uri=totp_uri)

@app.route('/')
@handle_errors
def index():
    """Render the main dashboard page."""
    bankroll = load_bankroll()
    
    # Calculate performance metrics
    total_profit = sum(entry['profit'] for entry in bankroll['history'])
    total_contests = len(bankroll['history'])
    win_rate = sum(1 for entry in bankroll['history'] if entry['profit'] > 0) / total_contests if total_contests > 0 else 0
    
    return render_template('index.html',
                         bankroll=bankroll,
                         total_profit=total_profit,
                         total_contests=total_contests,
                         win_rate=win_rate)

@app.route('/update_data', methods=['POST'])
@admin_required
@handle_errors
def update_data():
    """Update player data and process it."""
    # Scrape latest data
    player_data = scraper.get_player_stats()
    if player_data.empty:
        raise ValueError("No player data retrieved")
    
    # Process data
    processed_data = processor.clean_player_data(player_data)
    if processed_data.empty:
        raise ValueError("Error processing player data")
    
    # Save to file
    save_path = DATA_DIR / 'processed/latest_player_data.csv'
    processed_data.to_csv(save_path, index=False)
    
    return jsonify({
        'success': True,
        'message': 'Data updated successfully',
        'player_count': len(processed_data),
        'last_updated': datetime.now().isoformat()
    })

@app.route('/generate_lineup', methods=['POST'])
@handle_errors
def generate_lineup():
    """Generate optimal lineup based on form parameters."""
    logger.info("=== Generating Lineup - Parameter Diagnostics ===")
    # Validate input parameters
    platform = request.form.get('platform', 'draftkings').lower()
    max_lineups = int(request.form.get('max_lineups', 1))
    min_salary = float(request.form.get('min_salary', 0))
    logger.info(f"Received parameters - platform: {platform}, max_lineups: {max_lineups}, min_salary: {min_salary}")

    if platform not in ['draftkings', 'fanduel']:
        raise ValueError(f"Invalid platform: {platform}")

    # Load and validate player data
    try:
        player_data = pd.read_csv(DATA_DIR / 'processed/latest_player_data.csv')
        logger.info(f"Loaded player data: {len(player_data)} records")
    except FileNotFoundError:
        raise ValueError("No player data available. Please update data first.")

    # Get fantasy prices
    prices = scraper.get_fantasy_prices(platform)
    if not prices.empty:
        player_data = player_data.merge(prices, on='name', how='left')
        logger.info(f"Merged fantasy prices: {len(prices)} records")
    else:
        logger.warning("Fantasy prices data is empty - using placeholder data")
        # Add placeholder salary for testing
        if 'salary' not in player_data.columns:
            player_data['salary'] = 5000
            logger.info("Added placeholder salary data for testing")

    # Generate predictions
    player_data = predictor.predict(player_data)
    logger.info(f"Generated predictions for {len(player_data)} players")


    # Optimize lineup
    optimizer = dk_optimizer if platform == 'draftkings' else fd_optimizer
    lineup = optimizer.optimize(
        player_data,
        max_lineups=max_lineups,
        min_salary=min_salary
    )

    if lineup.empty:
        logger.error("Generated lineup is empty")
        raise ValueError("Could not generate valid lineup")

    # Format lineup for display
    result = lineup.to_dict(orient='records')
    for player in result:
        player['salary'] = f"${player['salary']:,}"
        player['predicted_points'] = f"{player['predicted_points']:.2f}"

    return jsonify({'success': True, 'lineup': result, 'metadata': {'platform': platform, 'generated_at': datetime.now().isoformat(), 'lineup_count': len(lineup['lineup_number'].unique())}})
    # Test comment

@app.route('/update_bankroll', methods=['POST'])
@admin_required
@handle_errors
def update_bankroll():
    """Update bankroll with contest results."""
    # Validate input
    try:
        contest_date = datetime.strptime(
            request.form.get('date', datetime.now().strftime('%Y-%m-%d')),
            '%Y-%m-%d'
        )
        platform = request.form.get('platform', 'draftkings')
        entry_fee = float(request.form.get('entry_fee', 0))
        winnings = float(request.form.get('winnings', 0))
    except ValueError as e:
        raise ValueError(f"Invalid input parameters: {str(e)}")
    
    # Calculate results
    profit = winnings - entry_fee
    roi = (profit / entry_fee * 100) if entry_fee > 0 else 0
    
    # Update bankroll
    bankroll = load_bankroll()
    bankroll['current'] += profit
    bankroll['history'].append({
        'date': contest_date.strftime('%Y-%m-%d'),
        'platform': platform,
        'entry_fee': entry_fee,
        'winnings': winnings,
        'profit': profit,
        'roi': f"{roi:.2f}%"
    })
    save_bankroll(bankroll)

    return jsonify({
        'success': True,
        'message': 'Bankroll updated successfully',
        'current_bankroll': f"${bankroll['current']:.2f}",
        'last_updated': bankroll['last_updated']
    })

@app.route('/analysis')
@handle_errors
def analysis():
    """Analyze historical performance."""
    bankroll = load_bankroll()
    
    # Convert history to DataFrame
    df = pd.DataFrame(bankroll['history'])
    if df.empty:
        return "No data available for analysis."
    
    # Calculate metrics
    df['date'] = pd.to_datetime(df['date'])
    df['profit'] = df['winnings'] - df['entry_fee']
    df['roi'] = df['profit'] / df['entry_fee'] * 100
    
    # Group by platform
    platform_summary = df.groupby('platform').agg({
        'profit': 'sum',
        'roi': 'mean',
        'entry_fee': 'sum',
        'winnings': 'sum'
    }).reset_index()
    
    # Overall metrics
    total_profit = df['profit'].sum()
    average_roi = df['roi'].mean()
    
    # Time-based analysis (e.g., monthly)
    df_monthly = df.set_index('date').resample('M').agg({
        'profit': 'sum',
        'roi': 'mean',
        'entry_fee': 'sum',
        'winnings': 'sum'
    })
    df_monthly.index.rename('month', inplace=True)
    df_monthly.reset_index(inplace=True)
  
    
    return render_template('analysis.html',
                           total_profit=f"{total_profit:.2f}",
                           average_roi=f"{average_roi:.2f}%",
                           platform_summary=platform_summary.to_dict(orient='records'),
                           monthly_summary=df_monthly.to_dict(orient='records'))

@app.route('/admin')
@admin_required
def admin():
    return "Admin page"

@app.route('/premium')
@premium_required
def premium():
    return "Premium page"