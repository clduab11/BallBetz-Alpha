"""
Logging utilities for Hello SPARC application.

This module provides functions for logging application events
in a standardized format.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Name for the logger
        level: Optional logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    if level is None:
        level = logging.INFO
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handler if none exists to prevent duplicate handlers
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger


def log_greeting(logger: logging.Logger, name: str, greeting: str) -> None:
    """
    Log a greeting event.
    
    Args:
        logger: Logger instance
        name: Name used in greeting
        greeting: Formatted greeting message
    """
    logger.info(f"Generated greeting for {name}: '{greeting}'")


class GreetingLogger:
    """
    A class-based logger for greeting-related operations.
    
    This demonstrates an object-oriented approach to logging
    that can maintain state and provide specialized methods.
    """
    
    def __init__(self, app_name: str, level: Optional[int] = None):
        """
        Initialize the greeting logger.
        
        Args:
            app_name: Name of the application
            level: Optional logging level
        """
        self.logger = get_logger(f"{app_name}.greetings", level)
        self.greeting_count = 0
    
    def log_greeting(self, name: str, greeting: str) -> None:
        """
        Log a greeting event and increment counter.
        
        Args:
            name: Name used in greeting
            greeting: Formatted greeting message
        """
        self.greeting_count += 1
        self.logger.info(
            f"Greeting #{self.greeting_count} generated for {name}: '{greeting}'"
        )
    
    def log_error(self, message: str) -> None:
        """
        Log an error event.
        
        Args:
            message: Error message
        """
        self.logger.error(f"Greeting error: {message}")
    
    def get_stats(self) -> dict:
        """
        Get logger statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_greetings": self.greeting_count
        }