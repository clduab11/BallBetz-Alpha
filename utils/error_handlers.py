"""Error handling utilities for BallBetz-Alpha.

This module provides error handling utilities for the application.
"""

import logging
from functools import wraps
from flask import jsonify, flash

# Set up logging
logger = logging.getLogger(__name__)


def handle_errors(f):
    """Decorator to handle errors in routes.
    
    Args:
        f: The function to decorate
        
    Returns:
        The decorated function
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            flash(f"An error occurred: {str(e)}", 'error')
            return jsonify({'success': False, 'message': str(e)}), 500
    return wrapper