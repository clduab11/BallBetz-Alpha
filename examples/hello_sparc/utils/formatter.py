"""
Text formatting utilities for Hello SPARC application.

This module provides functions for formatting text output,
including greeting messages and colored terminal text.
"""

from typing import Dict, Optional


# ANSI color codes
COLORS: Dict[str, str] = {
    'black': '\033[30m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'reset': '\033[0m'
}


def format_greeting(template: str, name: str) -> str:
    """
    Format a greeting message with the given name.
    
    Args:
        template: The greeting template containing {name} placeholder
        name: The name to insert into the greeting
        
    Returns:
        Formatted greeting message
        
    Examples:
        >>> format_greeting("Hello, {name}!", "SPARC")
        'Hello, SPARC!'
    """
    try:
        # Check if the template contains the name placeholder
        if '{name}' not in template:
            return f"Hello, {name}!"
        return template.format(name=name) 
    except KeyError:
        # Fallback in case the template is missing the name placeholder
        return f"Hello, {name}!"
    except Exception as e:
        # Log and handle any unexpected errors
        print(f"Error formatting greeting: {e}")
        return f"Hello, {name}!"


def format_colored_text(text: str, color: str = 'reset') -> str:
    """
    Format text with ANSI color codes for terminal output.
    
    Args:
        text: The text to color
        color: Color name (default: 'reset')
        
    Returns:
        Text with ANSI color codes
        
    Examples:
        >>> format_colored_text("Hello", "blue")
        '\033[34mHello\033[0m'
    """
    # Get color code or use reset if color not found
    color_code = COLORS.get(color.lower(), COLORS['reset'])
    
    # Return colored text with reset at the end
    return f"{color_code}{text}{COLORS['reset']}"


def wrap_in_box(text: str, width: Optional[int] = None) -> str:
    """
    Wrap text in a decorative ASCII box.
    
    Args:
        text: The text to wrap in a box
        width: Optional width of the box (default: length of text + 4)
        
    Returns:
        Text wrapped in an ASCII box
        
    Examples:
        >>> wrap_in_box("Hello")
        '┌──────┐\\n│ Hello │\\n└──────┘'
    """
    if width is None:
        width = len(text) + 4
    
    # Ensure width is sufficient for the text
    width = max(width, len(text) + 4)
    
    # Create the box
    top = f"┌{'─' * (width - 2)}┐"
    middle = f"│ {text.center(width - 4)} │"
    bottom = f"└{'─' * (width - 2)}┘"
    
    return f"{top}\n{middle}\n{bottom}"