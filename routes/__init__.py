"""Routes package for BallBetz-Alpha.

This package contains route handlers for the application.
"""

from flask import Blueprint

# Create blueprints
auth_bp = Blueprint('auth', __name__)
main_bp = Blueprint('main', __name__)
rewards_bp = Blueprint('rewards', __name__)
lineup_bp = Blueprint('lineup', __name__)

# Import routes to register them with blueprints
from . import auth_routes
from . import main_routes
from . import rewards_routes
from . import lineup_routes

# List of all blueprints to register with the app
blueprints = [auth_bp, main_bp, rewards_bp, lineup_bp]