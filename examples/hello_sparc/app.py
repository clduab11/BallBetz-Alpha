#!/usr/bin/env python3
"""
Hello SPARC Application

This is the main entry point for the Hello SPARC application, which
demonstrates SPARC methodology principles including modularity,
proper configuration management, and clean architecture.

Usage:
    python app.py [name]
"""

import sys
import argparse
from typing import Optional, List

# Import configuration
import sys
import os

# Add the parent directory to sys.path to make imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from hello_sparc.config import config

# Import utilities
from hello_sparc.utils.formatter import format_greeting, format_colored_text, wrap_in_box
from hello_sparc.utils.logger import GreetingLogger


class Greeter:
    """
    Greeter class responsible for generating and displaying greetings.
    
    This class follows SPARC methodology principles:
    - Uses external configuration
    - Single responsibility
    - Proper error handling
    - Testable design
    """
    
    def __init__(self, config_obj):
        """
        Initialize the Greeter with configuration.
        
        Args:
            config_obj: Configuration object
        """
        self.config = config_obj
        self.logger = GreetingLogger(self.config.app_name)
    
    def generate_greeting(self, name: Optional[str] = None) -> str:
        """
        Generate a greeting message.
        
        Args:
            name: Optional name to use in greeting (default: from config)
            
        Returns:
            Formatted greeting message
        """
        # Use provided name or default from configuration
        actual_name = name if name else self.config.default_name
        
        # Generate greeting using formatter utility
        greeting = format_greeting(self.config.greeting_template, actual_name)
        
        # Log the greeting generation
        self.logger.log_greeting(actual_name, greeting)
        
        return greeting
    
    def display_greeting(self, name: Optional[str] = None) -> None:
        """
        Generate and display a greeting message.
        
        Args:
            name: Optional name to use in greeting
        """
        try:
            # Generate the greeting
            greeting = self.generate_greeting(name)
            
            # Format with color from configuration
            colored_greeting = format_colored_text(greeting, self.config.text_color)
            
            # Wrap in a decorative box
            boxed_greeting = wrap_in_box(greeting)
            colored_boxed_greeting = format_colored_text(boxed_greeting, self.config.text_color)
            
            # Display the greeting
            print("\n" + colored_boxed_greeting + "\n")
            
            # Show debug info if in debug mode
            if self.config.debug_mode:
                print("Debug information:")
                print(f"  Configuration: {self.config}")
                print(f"  Logger stats: {self.logger.get_stats()}")
                
        except Exception as e:
            self.logger.log_error(f"Error displaying greeting: {e}")
            print(f"An error occurred: {e}")


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Hello SPARC Application')
    parser.add_argument('name', nargs='?', default=None,
                        help='Name to use in greeting (default: from config)')
    
    return parser.parse_args(args)


def main(args: List[str] = None) -> int:
    """
    Main entry point for the application.
    
    Args:
        args: Command line arguments (default: sys.argv[1:])
        
    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]
    
    try:
        # Parse command line arguments
        parsed_args = parse_args(args)
        
        # Create and use the Greeter
        greeter = Greeter(config)
        greeter.display_greeting(parsed_args.name)
        
        return 0
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())