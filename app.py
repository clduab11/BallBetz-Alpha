"""Main application module for BallBetz-Alpha.

This module initializes the Flask application and sets up all necessary components.
"""

from flask import Flask
import logging
from pathlib import Path
from flask_login import LoginManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from flask_bcrypt import Bcrypt

# Import configuration
from config.app_config import AppConfig

# Import authentication
from auth.user_manager import user_manager, User

# Import routes
from routes import blueprints

# Set up logging
logging.basicConfig(
    level=getattr(logging, AppConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(AppConfig.LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Diagnostic: Check for required dependencies
logger.info("=== BallBetz Diagnostic Logs ===")
try:
    import importlib.util
    pulp_spec = importlib.util.find_spec('pulp')
    logger.info(f"PuLP dependency check: {'FOUND' if pulp_spec else 'MISSING'}")
except ImportError:
    logger.error("PuLP dependency check: MISSING - ImportError")

# Log Python and key package versions
logger.info(f"Python version: {importlib.sys.version}")

# Initialize Flask app
app = Flask(__name__)

# Apply configuration
app.config.from_mapping(AppConfig.get_flask_config())

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'  # Set the login view

# Initialize Flask-Bcrypt
bcrypt = Bcrypt()
bcrypt.init_app(app)

# Initialize Flask-Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[AppConfig.RATELIMIT_DEFAULT]
)

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Set up user loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    """Load a user by ID for Flask-Login."""
    if user_id == 'None' or user_id is None:
        return None
    try:
        return user_manager.get_user(user_id)
    except (ValueError, KeyError):
        return None

# Create required directories
def setup_directories() -> None:
    """Create required directories if they don't exist."""
    required_dirs = ['data/raw', 'data/processed', 'models']
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# Register blueprints
for blueprint in blueprints:
    app.register_blueprint(blueprint)

# Set up directories
setup_directories()

# Validate configuration
try:
    AppConfig.validate_config()
    logger.info("Configuration validated successfully")
except Exception as e:
    logger.error(f"Configuration validation failed: {str(e)}")
    if AppConfig.FLASK_ENV == 'production':
        raise

# Log initialization complete
logger.info("Application initialization complete")

if __name__ == '__main__':
    # Run the app
    if AppConfig.FLASK_ENV == 'production' and AppConfig.SSL_CERT_PATH and AppConfig.SSL_KEY_PATH:
        # Run with SSL in production
        app.run(
            host='0.0.0.0',
            port=5000,
            ssl_context=(AppConfig.SSL_CERT_PATH, AppConfig.SSL_KEY_PATH)
        )
    else:
        # Run without SSL in development
        app.run(debug=AppConfig.DEBUG)