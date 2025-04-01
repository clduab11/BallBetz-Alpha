"""
Configuration module for Hello SPARC application.

This module demonstrates how to properly handle configuration 
and environment variables following SPARC methodology best practices.
"""

import os
from typing import Dict, Any


class Config:
    """
    Configuration class that loads settings from environment variables
    with sensible defaults.
    
    This class follows SPARC best practices:
    - No hardcoded secrets
    - Environment variable abstraction
    - Clear documentation
    - Sensible defaults
    """
    
    def __init__(self):
        """Initialize configuration with values from environment or defaults."""
        # Application settings
        self.app_name = os.environ.get('APP_NAME', 'Hello SPARC')
        self.debug_mode = os.environ.get('DEBUG_MODE', 'False').lower() == 'true'
        
        # User settings
        self.greeting_template = os.environ.get('GREETING_TEMPLATE', 'Hello, {name}!')
        self.default_name = os.environ.get('DEFAULT_NAME', 'World')
        
        # Customization
        self.text_color = os.environ.get('TEXT_COLOR', 'blue')
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        # Example validation: ensure greeting template has name placeholder
        if '{name}' not in self.greeting_template:
            raise ValueError("GREETING_TEMPLATE must contain '{name}' placeholder")
    
    def as_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return {
            'app_name': self.app_name,
            'debug_mode': self.debug_mode,
            'greeting_template': self.greeting_template,
            'default_name': self.default_name,
            'text_color': self.text_color
        }
    
    def __str__(self) -> str:
        """Return string representation of configuration."""
        # Only show non-sensitive configuration (no secrets or credentials)
        return f"Config(app_name='{self.app_name}', debug_mode={self.debug_mode})"


# Create a default configuration instance for import
config = Config()


if __name__ == "__main__":
    # If run directly, print the current configuration
    print("Current configuration:")
    for key, value in config.as_dict().items():
        print(f"  {key}: {value}")